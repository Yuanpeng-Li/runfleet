"""[WP1] tty-less subprocess spawning: new session, scrubbed env, no prompts (§4.2).

Every non-interactive ssh (and the local `sh -s`) is started here: `start_new_session=True` (no controlling terminal,
so nothing can open `/dev/tty`), stdin a pipe or `/dev/null`, `DISPLAY`/`SSH_ASKPASS` removed and
`SSH_ASKPASS_REQUIRE=never`. Any prompt therefore fails at once instead of writing onto the TUI.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

from runfleet import debugdump
from runfleet.errors import RunFleetError
from runfleet.models import ErrorKind, HostError
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult, RunResult

log = logging.getLogger("runfleet.transport.spawn")

READ_SIZE = 65536
STDOUT_CAP = 64 << 20          # client-side guard for one process
STDERR_CAP = 64 << 10
STREAM_STDERR_TAIL = 4096
DRAIN_GRACE = 0.5              # readers get this long after exit (a descendant may hold the pipe)
KILL_GRACE = 2.0               # SIGTERM -> SIGKILL


class SpawnError(RunFleetError):
    """The program could not be started (not found, permission, fork failure)."""


def scrubbed_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """`base` (default os.environ) minus DISPLAY and SSH_ASKPASS, plus SSH_ASKPASS_REQUIRE=never."""
    env = dict(os.environ if base is None else base)
    env.pop("DISPLAY", None)
    env.pop("SSH_ASKPASS", None)
    env["SSH_ASKPASS_REQUIRE"] = "never"
    return env


class _ExitProtocol(asyncio.subprocess.SubprocessStreamProtocol):
    """Signals process exit as soon as it happens. `Process.wait()` also waits for every pipe to close, which a
    backgrounded descendant (an `ssh -fN` master, a stray `sleep &`) can keep open indefinitely."""

    def __init__(self, limit: int, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(limit=limit, loop=loop)
        self.exited = asyncio.Event()

    def process_exited(self) -> None:
        super().process_exited()
        self.exited.set()


_CHILDREN: weakref.WeakKeyDictionary[asyncio.subprocess.Process, tuple[asyncio.Event, asyncio.SubprocessTransport]] = (
    weakref.WeakKeyDictionary())


async def spawn(argv: Sequence[str], *, stdin_pipe: bool, capture: bool = True,
                env: Mapping[str, str] | None = None) -> asyncio.subprocess.Process:
    """Start `argv` in a new session with the scrubbed environment. Raises SpawnError."""
    out = asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL
    loop = asyncio.get_running_loop()
    protocol: _ExitProtocol | None = None

    def factory() -> _ExitProtocol:
        nonlocal protocol
        protocol = _ExitProtocol(limit=READ_SIZE, loop=loop)
        return protocol

    try:
        transport, _ = await loop.subprocess_exec(
            factory, *argv,
            stdin=asyncio.subprocess.PIPE if stdin_pipe else asyncio.subprocess.DEVNULL,
            stdout=out, stderr=out,
            start_new_session=True,
            env=scrubbed_env(env),
        )
    except OSError as e:
        raise SpawnError(f"cannot run {argv[0]}: {e.strerror or e}") from e
    assert protocol is not None
    proc = asyncio.subprocess.Process(transport, protocol, loop)
    _CHILDREN[proc] = (protocol.exited, transport)
    return proc


async def wait_exit(proc: asyncio.subprocess.Process) -> int | None:
    """Wait for the process itself to exit (not for its pipes to close); returns the exit status."""
    entry = _CHILDREN.get(proc)
    if entry is None:
        return await proc.wait()
    await entry[0].wait()
    return proc.returncode


def release(proc: asyncio.subprocess.Process) -> None:
    """Close our ends of the pipes (a descendant may still hold theirs); the process must have exited."""
    entry = _CHILDREN.pop(proc, None)
    if entry is not None and proc.returncode is not None:
        entry[1].close()


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the process group the child leads (start_new_session), e.g. ssh plus its ProxyCommand."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


def descendants(pid: int) -> list[int]:
    """Best-effort list of `pid`'s descendants from /proc (Linux; [] elsewhere).

    GNU `timeout` moves itself into its own process group, so signalling our child's group does not reach a section
    that runs under it; on the local host we therefore signal the descendants individually as well.
    """
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read().decode(errors="replace")
        except OSError:
            continue
        rest = stat.rpartition(")")[2].split()
        if len(rest) > 1 and rest[1].isdigit():
            children.setdefault(int(rest[1]), []).append(int(entry))
    out: list[int] = []
    todo = [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _signal_tree(proc: asyncio.subprocess.Process, sig: int, extra: list[int]) -> None:
    _signal_group(proc, sig)
    for p in extra:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, sig)


async def terminate(proc: asyncio.subprocess.Process, grace: float = KILL_GRACE) -> None:
    """SIGTERM the group (and stray descendants), SIGKILL after `grace` seconds, reap."""
    if proc.returncode is not None:
        return
    tree = descendants(proc.pid)
    _signal_tree(proc, signal.SIGTERM, tree)
    try:
        await asyncio.wait_for(wait_exit(proc), grace)
    except TimeoutError:
        _signal_tree(proc, signal.SIGKILL, tree + descendants(proc.pid))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wait_exit(proc), grace)


async def _pump(reader: asyncio.StreamReader | None, buf: bytearray, cap: int, *, keep_tail: bool = False) -> None:
    if reader is None:
        return
    while chunk := await reader.read(READ_SIZE):
        buf += chunk
        if len(buf) > cap:
            if keep_tail:
                del buf[: len(buf) - cap]
            else:
                del buf[cap:]


async def _feed(proc: asyncio.subprocess.Process, data: bytes) -> None:
    assert proc.stdin is not None
    try:
        proc.stdin.write(data)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            proc.stdin.close()


# Intervals (time.monotonic) during which the event loop was frozen on purpose: the TUI suspended for an external
# command (`c`/`s`/`p` run synchronously inside App.suspend) or the whole process stopped by `^z`. A probe timer that
# expired inside such a window measured the freeze, not the remote side (§4.2).
_STALLS: deque[tuple[float, float]] = deque(maxlen=16)


def note_stall(start: float, end: float) -> None:
    _STALLS.append((start, end))


@contextlib.contextmanager
def loop_frozen() -> Iterator[None]:
    """Wrap code that blocks the event loop (a suspended TUI, SIGTSTP); in-flight probes then get a fresh budget."""
    start = time.monotonic()
    try:
        yield
    finally:
        note_stall(start, time.monotonic())


def stalled_since(t0: float) -> bool:
    """The loop was frozen at some point after `t0`."""
    return any(end > t0 for _start, end in _STALLS)


async def run(argv: Sequence[str], *, timeout: float, input: bytes | None = None, capture: bool = True,
              env: Mapping[str, str] | None = None) -> RunResult:
    """Run to completion (or `timeout`): readers in tasks, wait for exit, short drain, terminate on expiry.

    Raises SpawnError if the program cannot start. Cancellation kills the process group.
    """
    t0 = time.monotonic()
    proc = await spawn(argv, stdin_pipe=input is not None, capture=capture, env=env)
    out, err = bytearray(), bytearray()
    tasks = [asyncio.ensure_future(_pump(proc.stdout, out, STDOUT_CAP)),
             asyncio.ensure_future(_pump(proc.stderr, err, STDERR_CAP, keep_tail=True))]
    if input is not None:
        tasks.append(asyncio.ensure_future(_feed(proc, input)))
    timed_out = False
    try:
        try:
            await asyncio.wait_for(wait_exit(proc), timeout)
        except TimeoutError:
            if stalled_since(t0):
                # The timer ran out while the loop was frozen (suspended TUI / ^z): the probe may have finished (its
                # exit is still queued) or be blocked on a full pipe nobody read. One fresh budget, then a real verdict.
                try:
                    await asyncio.wait_for(wait_exit(proc), timeout)
                except TimeoutError:
                    timed_out = True
            else:
                timed_out = True
        if timed_out:
            await terminate(proc)
        _, pending = await asyncio.wait(tasks, timeout=DRAIN_GRACE)
        for t in pending:
            t.cancel()
    finally:
        if proc.returncode is None:          # cancelled: never leave a stray ssh behind
            # descendants too (GNU timeout runs a local section in its own group), then reap before release()
            _signal_tree(proc, signal.SIGKILL, descendants(proc.pid))
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.shield(asyncio.wait_for(wait_exit(proc), KILL_GRACE))
        for t in tasks:
            if not t.done():
                t.cancel()
        release(proc)
    return RunResult(rc=None if timed_out else proc.returncode, stdout=bytes(out), stderr=bytes(err),
                     elapsed_s=time.monotonic() - t0, timed_out=timed_out)


ProbeClassifier = Callable[[RunResult, framing.ParsedProbe], HostError | None]


async def run_probe(argv: Sequence[str], script: ProbeScript, *, timeout: float, classify: ProbeClassifier,
                    env: Mapping[str, str] | None = None, label: str | None = None) -> ProbeResult:
    """Render `script`, feed it to `argv` (a `sh -s` reader), parse the framing, classify transport failure.

    Never raises: a spawn failure or an internal error becomes a ProbeResult with every section NOT_RUN. `label` (the
    host name) enables the DEBUG summary line and, with `--debug`, the raw dump (`runfleet.debugdump`).
    """
    try:
        text = framing.render(script)          # purpose "attach": no temp dir on the login host (§7.16.4)
    except ValueError as e:
        return failed_probe(script, HostError(ErrorKind.INTERNAL, f"bad probe script: {e}"[:80], str(e)))
    try:
        res = await run(argv, timeout=timeout, input=text.encode(), env=env)
    except SpawnError as e:
        return failed_probe(script, HostError(ErrorKind.SSH, str(e)[:80], str(e)))
    parsed = framing.parse(res.stdout, script, timed_out=res.timed_out)
    error = classify(res, parsed)
    if label is not None:
        outcomes: dict[str, int] = {}
        for r in parsed.sections.values():
            outcomes[r.outcome.value] = outcomes.get(r.outcome.value, 0) + 1
        summary = (f"{label}: probe {script.purpose} {len(script.sections)} sections rc={res.rc} "
                   f"{res.elapsed_s * 1000:.0f} ms timed_out={res.timed_out} outcomes={outcomes} "
                   f"error={error.kind.value + ': ' + error.summary if error else None}")
        log.debug("%s", summary)
        if debugdump.enabled():
            names = " ".join(s.name for s in script.sections)
            meta = f"{summary}\nargv: {' '.join(argv)}\nsections: {names}"
            debugdump.dump_probe(label, script.purpose, text, res.stdout, res.stderr, meta)
    return ProbeResult(run=res, sections=parsed.sections, error=error)


def failed_probe(script: ProbeScript, error: HostError) -> ProbeResult:
    """A ProbeResult for a probe that never ran: every section NOT_RUN, `error` set."""
    parsed = framing.parse(b"", script)
    return ProbeResult(run=RunResult(rc=None, stdout=b"", stderr=b"", elapsed_s=0.0, timed_out=False),
                       sections=parsed.sections, error=error)


class ProcessStream:
    """StreamHandle over a spawned process: raw stdout blocks; stdin kept open as the remote watchdog (§4.7)."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._err = bytearray()
        self._err_task = asyncio.ensure_future(_pump(proc.stderr, self._err, STREAM_STDERR_TAIL, keep_tail=True))
        self._close_task: asyncio.Future[None] | None = None

    @classmethod
    async def start(cls, argv: Sequence[str], env: Mapping[str, str] | None = None) -> ProcessStream:
        return cls(await spawn(argv, stdin_pipe=True, env=env))

    async def chunks(self) -> AsyncIterator[bytes]:
        reader = self._proc.stdout
        if reader is None:
            return
        while chunk := await reader.read(READ_SIZE):
            yield chunk

    async def close(self) -> None:
        """Idempotent; the shutdown runs as one task, so a caller cancelled during the exit wait (screen pop, app
        quit) neither skips terminate/release nor turns later close() calls into no-ops."""
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._shutdown())
        await asyncio.shield(self._close_task)

    async def _shutdown(self) -> None:
        try:
            if self._proc.stdin is not None:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
                    self._proc.stdin.close()
            try:
                await asyncio.wait_for(wait_exit(self._proc), KILL_GRACE)
            except TimeoutError:
                await terminate(self._proc)
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._err_task, DRAIN_GRACE)
        finally:
            if not self._err_task.done():
                self._err_task.cancel()
            release(self._proc)

    @property
    def stderr_text(self) -> str:
        return bytes(self._err[-STREAM_STDERR_TAIL:]).decode("utf-8", errors="replace")

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode
