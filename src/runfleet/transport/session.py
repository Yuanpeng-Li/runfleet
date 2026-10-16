"""[WP1] SessionTransport: one long-lived `ssh -T` per host running the sh agent of `transport/agent.py` (§4.8).

Windows' OpenSSH has no ControlMaster, so a Duo or password host could not be polled from there: every probe would be
a new login. The session transport keeps ONE ssh per host open instead and multiplexes probes and log streams over
it through a small POSIX `sh` agent it sends on stdin (nothing is installed; the agent ends every job and removes
its temp dir when stdin closes). It is the default on Windows (`mux` other than "off") and opt-in elsewhere
(`mux = "session"`).

It reuses everything of `OpenSSHTransport` that does not involve a ControlMaster: `ssh -G` resolution, the
ProxyJump rewrite, `ssh_options`, the `-E` log and the ssh error classification. Its own parts:

- `prepare`: a BatchMode session (no prompt can appear), at most one attempt per 30 s unless forced; ready when the
  agent sends `R` within `connect_timeout` + 5 s.
- `login` (the optional interactive-login capability of §3.5): the same session WITHOUT BatchMode, started while
  the TUI is suspended, so ssh asks its Duo / password / host-key questions in the real terminal; it returns once
  the agent is ready (or ssh gave up) and the session then serves the background polls until RunFleet exits.
- `check_master`: the ssh is alive and the agent answers a ping within 3 s.
- `probe`: one agent job; a local timeout kills only that job (`K`), never the session.
- `stream`: one agent job whose stdin `close()` ends (`W`), then `K`.
- `reset_master` / `close`: end the session (closing its stdin ends the agent and every job).

I/O runs in three daemon threads per session (stdout, stderr, stdin) on a plain `subprocess.Popen`, so the same code
works on the Windows Proactor loop and while the event loop is blocked by a suspended TUI (the login waits for `R`
on a `threading.Event`); replies reach the loop through `call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from runfleet.config import HostConfig
from runfleet.errors import RunFleetError
from runfleet.models import ErrorKind, HostError
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, StreamSpec
from runfleet.transport import agent, classify, spawn
from runfleet.transport.base import ProbeResult, RunResult, StreamHandle
from runfleet.transport.openssh import CHECK_TIMEOUT, ENSURE_INTERVAL, OpenSSHTransport, _read_tail
from runfleet.util import platform

log = logging.getLogger("runfleet.transport.session")

READY_GRACE = 5.0               # on top of connect_timeout: the agent's `R` must arrive by then
LOGIN_CONNECT_TIMEOUT = 30      # ConnectTimeout of the interactive login (as `connect_argv`)
SESSION_KEEPALIVE = ("-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
STDERR_KEEP = 8192              # session stderr tail kept for diagnostics; a job's stderr keeps the same
STREAM_STDERR = 4096
CLOSE_GRACE = 2.0               # stdin closed -> ssh gone, else killed
EXIT_WAIT = 5.0                 # the stdout reader waits this long for ssh's exit status after EOF
POLL_STEP = 0.2                 # the blocking login re-checks Ctrl-C and ssh's exit this often
JOB_KILL_WAIT = 3.5             # after `K`: the agent's SIGKILL comes 2 s after its SIGTERM, and 1 s later a job that
                                # still holds its output stops counting toward MAX_JOBS; the client keeps the job's
                                # slot until then, so both sides count the same jobs
LOGIN_TERM_WAIT = 1.0           # kill_now: SIGTERM (ssh restores the terminal's echo), then SIGKILL after this


@runtime_checkable
class InteractiveLogin(Protocol):
    """The session transport's optional capability (§4.8; not part of the frozen §3.5 Transport): a login whose ssh
    must outlive the suspend it ran in. `login()` BLOCKS (call it inside App.suspend, after `before_connect()`): ssh
    prompts in the real terminal; 0 = logged in and serving the background polls, else ssh's exit status (130:
    Ctrl-C). The UI reaches it through `FleetAPI.connect_login` instead of running `connect_argv()`; `runfleet
    connect` cannot use it (the session would end with the CLI process)."""

    def login(self) -> int: ...


class _Closed:
    """Reader thread -> loop: stdout reached EOF (ssh ended); `rc` its exit status, None if still unknown."""

    __slots__ = ("rc",)

    def __init__(self, rc: int | None) -> None:
        self.rc = rc


class _Job:
    """One agent job (probe or stream) as the loop sees it."""

    __slots__ = ("id", "out", "err", "queue", "done", "rc", "lost")

    def __init__(self, job_id: str, loop: asyncio.AbstractEventLoop, *, stream: bool) -> None:
        self.id = job_id
        self.out = bytearray()
        self.err = bytearray()
        self.queue: asyncio.Queue[bytes | None] | None = asyncio.Queue() if stream else None
        self.done: asyncio.Future[int | None] = loop.create_future()
        self.rc: int | None = None
        self.lost = False                       # the session ended before the job's `Z`

    def finish(self, rc: int | None, *, lost: bool = False) -> None:
        if self.done.done():
            return
        self.rc, self.lost = rc, lost
        if self.queue is not None:
            self.queue.put_nowait(None)
        self.done.set_result(rc)


class AgentSession:
    """One ssh process running the agent: its threads, the jobs in flight, the pings. Create it on the loop thread
    (it binds the running loop). Raises spawn.SpawnError when ssh cannot start."""

    def __init__(self, argv: list[str], *, env: dict[str, str], interactive: bool, label: str) -> None:
        self.loop = asyncio.get_running_loop()
        self.label = label
        kwargs = platform.login_spawn_kwargs() if interactive else platform.background_spawn_kwargs()
        run_argv = platform.login_argv(argv) if interactive else list(argv)
        try:
            self.proc = subprocess.Popen(run_argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0, env=env, **kwargs)
        except OSError as e:
            raise spawn.SpawnError(f"cannot run {argv[0]}: {e.strerror or e}") from e
        if platform.IS_WINDOWS:
            platform.kill_on_exit(self.proc.pid)   # Job Object: the session (and its hop) never outlive RunFleet
        self.ready_event = threading.Event()        # set by the stdout thread: the blocking login waits on it
        self.ended_event = threading.Event()        # stdout EOF
        self.ready: agent.Ready | None = None
        self.fatal: str | None = None
        self.rc: int | None = None
        self.ready_future: asyncio.Future[bool] = self.loop.create_future()
        self.jobs: dict[str, _Job] = {}
        self.closed = False                         # loop side: ended (EOF seen or given up)
        self.last_rx = float("-inf")                # time.monotonic() of the last stdout bytes (reader thread)
        self._pings: dict[str, asyncio.Future[bool]] = {}
        self._tokens = itertools.count(1)
        self._err = bytearray()
        self._err_lock = threading.Lock()
        self._outq: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._stdin_open = True
        self._close_task: asyncio.Future[None] | None = None
        for target, name in ((self._read_stdout, "out"), (self._read_stderr, "err"), (self._write_stdin, "in")):
            threading.Thread(target=target, name=f"rf-session-{label}-{name}", daemon=True).start()
        self.send(agent.hello())

    # ------------------------------------------------------------ threads
    def _read_stdout(self) -> None:
        decoder = agent.ReplyDecoder()
        out = self.proc.stdout
        assert out is not None
        try:
            while data := out.read(spawn.READ_SIZE):
                self.last_rx = time.monotonic()
                replies = decoder.feed(data)
                for r in replies:
                    if isinstance(r, agent.Ready):
                        self.ready = r
                        self.ready_event.set()
                    elif isinstance(r, agent.Fatal):
                        self.fatal = r.text
                self._post(replies)
        except (OSError, ValueError):
            pass
        rest: list[Any] = list(decoder.eof())
        try:
            rc: int | None = self.proc.wait(timeout=EXIT_WAIT)
        except subprocess.TimeoutExpired:
            rc = None
        self.rc = rc
        self.ended_event.set()
        self._post([*rest, _Closed(rc)])

    def _read_stderr(self) -> None:
        err = self.proc.stderr
        assert err is not None
        try:
            while data := err.read(spawn.READ_SIZE):
                with self._err_lock:
                    self._err += data
                    del self._err[: max(0, len(self._err) - STDERR_KEEP)]
        except (OSError, ValueError):
            pass

    def _write_stdin(self) -> None:
        stdin = self.proc.stdin
        assert stdin is not None
        try:
            while (data := self._outq.get()) is not None:
                view = memoryview(data)
                while view:
                    n = stdin.write(view)
                    view = view[n or 0:]
        except (OSError, ValueError):
            pass
        finally:
            self._stdin_open = False
            with contextlib.suppress(OSError, ValueError):
                stdin.close()

    def _post(self, replies: list[Any]) -> None:
        with contextlib.suppress(RuntimeError):          # the loop is closed: RunFleet is exiting
            self.loop.call_soon_threadsafe(self._dispatch, replies)

    # ------------------------------------------------------------ loop side
    def _dispatch(self, replies: list[Any]) -> None:
        for r in replies:
            if isinstance(r, agent.Output):
                job = self.jobs.get(r.job)
                if job is None:
                    continue
                if r.stderr:
                    job.err += r.data
                    del job.err[: max(0, len(job.err) - STDERR_KEEP)]
                elif job.queue is not None:
                    job.queue.put_nowait(r.data)
                elif len(job.out) < spawn.STDOUT_CAP:
                    job.out += r.data
            elif isinstance(r, agent.Done):
                done = self.jobs.pop(r.job, None)
                if done is not None:
                    done.finish(r.rc)
            elif isinstance(r, agent.Pong):
                fut = self._pings.pop(r.token, None)
                if fut is not None and not fut.done():
                    fut.set_result(True)
            elif isinstance(r, agent.Ready):
                if not self.ready_future.done():
                    self.ready_future.set_result(True)
            elif isinstance(r, agent.Fatal):
                log.warning("%s: session agent: %s", self.label, r.text)
            elif isinstance(r, agent.Noise):
                if r.line.strip():
                    log.debug("%s: session: ignored %r", self.label, r.line[:200])
            elif isinstance(r, _Closed):
                self._ended()

    def _ended(self) -> None:
        self.closed = True
        self._outq.put(None)                      # the stdin thread ends too (it waits on this queue)
        for job in list(self.jobs.values()):
            job.finish(None, lost=True)
        self.jobs.clear()
        for fut in self._pings.values():
            if not fut.done():
                fut.set_result(False)
        self._pings.clear()
        if not self.ready_future.done():
            self.ready_future.set_result(False)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def alive(self) -> bool:
        return not self.closed and not self.ended_event.is_set() and self.proc.poll() is None

    @property
    def usable(self) -> bool:
        return self.alive and self.ready is not None and self._stdin_open

    def stderr_text(self) -> str:
        with self._err_lock:
            return bytes(self._err).decode("utf-8", errors="replace")

    def send(self, data: bytes) -> bool:
        if not self._stdin_open:
            return False
        self._outq.put(data)
        return True

    def start(self, job_id: str, request: bytes, *, stream: bool) -> _Job:
        job = _Job(job_id, self.loop, stream=stream)
        if not self.usable or not self.send(request):
            job.finish(None, lost=True)
            return job
        self.jobs[job_id] = job
        return job

    def forget(self, job_id: str) -> None:
        """Stop waiting for a job (it was killed and its `Z` is late); later replies for it are dropped."""
        job = self.jobs.pop(job_id, None)
        if job is not None:
            job.finish(None)

    async def wait_ready(self, timeout: float) -> bool:
        t0 = time.monotonic()
        for _ in range(2):              # one fresh budget when the loop was frozen meanwhile (§4.2)
            try:
                return await asyncio.wait_for(asyncio.shield(self.ready_future), timeout)
            except TimeoutError:
                if not spawn.stalled_since(t0):
                    return False
        return False

    async def ping(self, timeout: float) -> bool:
        """True when the agent answers within `timeout`, or when the session's output kept arriving meanwhile: the
        pong then waits behind bulk output (a big log fetch on a slow link, up to the channel window), and a
        session whose bytes still flow is alive."""
        if not self.usable:
            return False
        token = f"p{next(self._tokens)}"
        fut: asyncio.Future[bool] = self.loop.create_future()
        self._pings[token] = fut
        t0 = time.monotonic()
        try:
            if not self.send(agent.ping(token)):
                return False
            for _ in range(2):
                try:
                    return await asyncio.wait_for(asyncio.shield(fut), timeout)
                except TimeoutError:
                    if time.monotonic() - self.last_rx < timeout and self.usable:
                        return True
                    if not spawn.stalled_since(t0):
                        return False
            return False
        finally:
            self._pings.pop(token, None)

    async def close(self) -> None:
        """Close stdin (the agent ends every job and its temp dir, then ssh exits), kill after a grace. Idempotent."""
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._shutdown())
        await asyncio.shield(self._close_task)

    async def _shutdown(self) -> None:
        self._outq.put(None)
        if not await asyncio.to_thread(self._wait, CLOSE_GRACE):
            await asyncio.to_thread(platform.kill_tree, self.proc, hard=True)   # Windows: taskkill takes a moment
            await asyncio.to_thread(self._wait, CLOSE_GRACE)
        self._ended()

    def kill_now(self) -> None:
        """Blocking end (the login under a suspended TUI): close stdin, then SIGTERM (an ssh still prompting restores
        the terminal's echo on it, never on SIGKILL), then kill the tree, reap."""
        self._outq.put(None)
        if not self._wait(0.5):
            platform.kill_tree(self.proc, hard=False)
            if not self._wait(LOGIN_TERM_WAIT):
                platform.kill_tree(self.proc, hard=True)
                self._wait(CLOSE_GRACE)
        self.closed = True

    def _wait(self, timeout: float) -> bool:
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


class SessionStream:
    """StreamHandle over one agent stream job (§4.7 watchdog form: `close()` closes its stdin)."""

    def __init__(self, session: AgentSession, job: _Job, release: Callable[[], None]) -> None:
        self._session = session
        self._job = job
        self._release = release
        self._close_task: asyncio.Future[None] | None = None

    async def chunks(self) -> AsyncIterator[bytes]:
        q = self._job.queue
        assert q is not None
        while (data := await q.get()) is not None:
            yield data

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.ensure_future(self._shutdown())
        await asyncio.shield(self._close_task)

    async def _shutdown(self) -> None:
        job, s = self._job, self._session
        try:
            if not job.done.done():
                s.send(agent.close_stdin(job.id))
                try:
                    await asyncio.wait_for(asyncio.shield(job.done), spawn.KILL_GRACE)
                except TimeoutError:
                    s.send(agent.kill(job.id))
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(job.done), JOB_KILL_WAIT)
        finally:
            if not job.done.done():
                s.forget(job.id)
            self._release()

    @property
    def stderr_text(self) -> str:
        return bytes(self._job.err[-STREAM_STDERR:]).decode("utf-8", errors="replace")

    @property
    def returncode(self) -> int | None:
        if not self._job.done.done():
            return None
        return 255 if self._job.lost else self._job.rc


class SessionTransport(OpenSSHTransport):
    """OpenSSH without a ControlMaster: one session ssh per host running the agent (§4.8)."""

    kind: Literal["openssh", "local", "fake", "demo"] = "openssh"

    def __init__(self, host: HostConfig, *, ssh: str = "ssh", mux_dir: Path | None = None,
                 log_path: Path | None = None, monotonic: Callable[[], float] = time.monotonic) -> None:
        super().__init__(host, ssh=ssh, mux_dir=mux_dir, log_path=log_path, monotonic=monotonic)
        if host.mux != "session":         # the plan follows the transport (mux.decide then says "session")
            self.config = dataclasses.replace(host, mux="session")
        self._session: AgentSession | None = None
        self._lock: asyncio.Lock | None = None
        self._sem: asyncio.Semaphore | None = None
        self._ids = itertools.count(1)
        self._gen = 0                     # bumped whenever the session is replaced or dropped

    # ------------------------------------------------------------ command line
    def session_argv(self, *, batch: bool) -> list[str]:
        """`ssh -T [-o BatchMode=yes] -o ControlMaster=no -o ControlPath=none {jump} {ssh_options}
        -o ConnectTimeout=… -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR -E <log> --
        <alias> <bootstrap>` (§4.8). `batch=False` is the interactive login: prompts in the real terminal."""
        _, plan = self._require()
        lead = ("-T", "-o", "BatchMode=yes") if batch else ("-T",)
        timeout = int(self.config.connect_timeout) if batch else LOGIN_CONNECT_TIMEOUT
        return [self._ssh, *lead, *plan.probe_opts, *self._jump(batch=batch, hop_log=True), *self._user_opts(),
                "-o", f"ConnectTimeout={timeout}", *SESSION_KEEPALIVE, "-o", "LogLevel=ERROR",
                "-E", str(self._log_path), "--", self.alias, agent.remote_command(self.config.remote_shell)]

    # ------------------------------------------------------------ session lifecycle
    def _the_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _slots(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(agent.MAX_JOBS)
        return self._sem

    def _usable(self) -> AgentSession | None:
        s = self._session
        return s if s is not None and s.usable else None

    async def prepare(self, *, force: bool = False) -> HostError | None:
        """Resolve, then make sure a session is up: a live one must answer a ping; a dead one is replaced by a
        BatchMode session, at most once per 30 s unless `force` (a user retry)."""
        err = await self.resolve()
        if err is not None:
            return err
        async with self._the_lock():
            s = self._usable()
            if s is not None:
                if await s.ping(CHECK_TIMEOUT):
                    self._last_ensure_error = None
                    return None
                if s.usable:          # alive but silent: ServerAlive ends a dead link; reset_master() forces it
                    return HostError(ErrorKind.TIMEOUT, "ssh session not answering", s.stderr_text())
            if self._session is not None:
                await self._drop()
            now = self._monotonic()
            if not force and self._last_ensure is not None and now - self._last_ensure < ENSURE_INTERVAL:
                return self._last_ensure_error or HostError(ErrorKind.MUX, "ssh session not up yet")
            self._last_ensure = now
            err = await self._start()
            self._last_ensure_error = err
            return err

    async def _start(self) -> HostError | None:
        self._truncate_log()
        gen = self._gen
        try:
            s = AgentSession(self.session_argv(batch=True), env=spawn.scrubbed_env(), interactive=False,
                             label=self.host)
        except spawn.SpawnError as e:
            return HostError(ErrorKind.SSH, str(e)[: classify.SUMMARY_LIMIT], str(e))
        budget = self.config.connect_timeout + READY_GRACE
        try:
            ready = await s.wait_ready(budget)
        except BaseException:           # cancelled (the poller stopped): never leave this ssh logged in
            await asyncio.shield(s.close())
            raise
        if gen != self._gen:            # a login won meanwhile, or the transport was closed / reset: not installed
            await s.close()
            if self._usable() is not None:
                return None
            return HostError(ErrorKind.MUX, "ssh session closed while it was starting")
        if ready:
            self._session = s
            self._gen += 1
            log.info("%s: session up (agent pid %s, %s)", self.host, s.ready.pid if s.ready else "?",
                     "setsid" if s.ready and s.ready.setsid else "no setsid")
            return None
        timed_out = not s.closed
        await s.close()
        return self._failure(s, timed_out=timed_out, budget=budget)

    def _diagnostics(self, s: AgentSession) -> str:
        return "\n".join(t for t in (_read_tail(self._log_path).strip(), s.stderr_text().strip()) if t)

    def _failure(self, s: AgentSession, *, timed_out: bool, budget: float) -> HostError:
        """Why a session never became ready."""
        text = self._diagnostics(s)
        if s.fatal:
            return HostError(ErrorKind.REMOTE, f"session agent: {s.fatal}"[: classify.SUMMARY_LIMIT], text)
        if timed_out:
            return HostError(ErrorKind.TIMEOUT, f"ssh session not ready after {budget:.0f} s", text)
        rc = s.rc
        if rc is not None and rc < 0:
            return classify.killed_by_signal(rc, text)
        err = classify.classify_ssh(255 if rc is None else rc, text, jumps=self._jumps())
        if err is not None:
            return err
        return classify.remote_shell_error(rc, text)    # ssh logged in, but the remote shell could not run the agent

    def _lost(self, s: AgentSession) -> HostError:
        """Why a session that was up ended (jobs in flight fail with this)."""
        text = self._diagnostics(s)
        if s.rc is not None and s.rc < 0:
            return classify.killed_by_signal(s.rc, text)
        err = classify.classify_ssh(255, text, jumps=self._jumps()) if s.rc in (None, 255) and text else None
        if err is not None and err.kind is not ErrorKind.SSH:
            return err
        return HostError(ErrorKind.MUX, "ssh session ended", text or f"rc={s.rc}")

    async def _drop(self) -> None:
        s, self._session = self._session, None
        self._gen += 1
        if s is not None:
            await s.close()

    async def _ensure(self) -> tuple[AgentSession | None, HostError | None]:
        err = await self.resolve()
        if err is not None:
            return None, err
        s = self._usable()
        if s is None:
            err = await self.prepare()
            s = self._usable()
            if err is None and s is None:
                err = HostError(ErrorKind.MUX, "ssh session not up yet")
        return s, err

    async def check_master(self) -> bool:
        s = self._usable()
        return s is not None and await s.ping(CHECK_TIMEOUT)

    async def reset_master(self) -> None:
        """End the session (two probe timeouts in a row, `doctor --reset`); the next prepare starts a new one."""
        await self._drop()
        self._last_ensure = None

    async def close(self) -> None:
        await self._drop()

    async def before_connect(self) -> bool:
        if await self.resolve() is not None or self.plan is None:
            return False
        if await self.check_master():
            return True
        await self._drop()
        self._truncate_log()
        return False

    def login(self) -> int:
        """Interactive login (§3.5 optional capability, §4.8), BLOCKING: run it inside `App.suspend`. Starts the
        session without BatchMode, so ssh prompts in the real terminal; returns 0 once the agent is ready (the
        session then serves every background poll), ssh's exit status if it gave up, 130 on Ctrl-C. Call
        `before_connect()` first (it resolves the host)."""
        if self.plan is None:
            print(f"runfleet: {self.host}: `ssh -G` failed; see `runfleet doctor {self.host}`", flush=True)
            return 255
        old, self._session = self._session, None
        self._gen += 1
        if old is not None:
            old.kill_now()
        self._truncate_log()
        try:
            s = AgentSession(self.session_argv(batch=False), env=dict(os.environ), interactive=True, label=self.host)
        except spawn.SpawnError as e:
            print(f"runfleet: {e}", flush=True)
            return 127
        try:
            while not s.ready_event.wait(POLL_STEP):
                if s.ended_event.is_set():
                    break
        except KeyboardInterrupt:
            s.kill_now()
            return 130
        if s.ready_event.is_set() and not s.ended_event.is_set():
            self._session = s
            self._gen += 1
            self._last_ensure = None
            self._last_ensure_error = None
            return 0
        s.kill_now()
        if s.fatal:
            print(f"runfleet: session agent: {s.fatal}", flush=True)
        rc = s.proc.returncode
        return rc if rc not in (None, 0) else 255

    # ------------------------------------------------------------ probes and streams
    def _new_id(self) -> str:
        return f"j{next(self._ids)}"

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        s, err = await self._ensure()
        if err is not None or s is None:
            return spawn.failed_probe(script, err or HostError(ErrorKind.MUX, "ssh session not up yet"))
        try:
            text = framing.render(script)
        except ValueError as e:
            return spawn.failed_probe(script, HostError(ErrorKind.INTERNAL, f"bad probe script: {e}"[:80], str(e)))
        res, lost = await self._run_job(s, text.encode(), timeout)
        base = self._classify(timeout)

        def run_classify(r: RunResult, parsed: framing.ParsedProbe) -> HostError | None:
            if lost:                     # the session died under the probe (§4.2 partial rule still applies)
                return None if parsed.ended > 0 else self._lost(s)
            if r.rc == 125 and b"runfleet agent: busy" in r.stderr:
                return HostError(ErrorKind.MUX, "session agent busy", r.stderr.decode("utf-8", "replace"))
            return base(r, parsed)

        return spawn.finish_probe(script, text, res, run_classify, label=self.host, where=f"session pid {s.pid}")

    async def _run_job(self, s: AgentSession, data: bytes, timeout: float) -> tuple[RunResult, bool]:
        """Run one probe job; a local timeout kills that job only (`K`, then up to JOB_KILL_WAIT for its `Z`)."""
        async with self._slots():
            job_id = self._new_id()
            t0 = time.monotonic()
            job = s.start(job_id, agent.probe_request(job_id, data), stream=False)
            timed_out = False
            try:
                for attempt in range(2):
                    try:
                        await asyncio.wait_for(asyncio.shield(job.done), timeout)
                        break
                    except TimeoutError:
                        if attempt == 1 or not spawn.stalled_since(t0):
                            timed_out = True
                            break
                if timed_out:
                    s.send(agent.kill(job_id))
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(job.done), JOB_KILL_WAIT)
            finally:
                if not job.done.done():          # cancelled, or its `Z` is late: kill it and stop waiting
                    s.send(agent.kill(job_id))
                    s.forget(job_id)
        rc = None if timed_out or job.lost else job.rc
        return RunResult(rc=rc, stdout=bytes(job.out), stderr=bytes(job.err), elapsed_s=time.monotonic() - t0,
                         timed_out=timed_out), job.lost and not timed_out

    async def stream(self, spec: StreamSpec) -> StreamHandle:
        s, err = await self._ensure()
        if err is not None or s is None:
            raise RunFleetError((err.summary if err else "ssh session not up yet"))
        sem = self._slots()
        await sem.acquire()
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                sem.release()

        job_id = self._new_id()
        job = s.start(job_id, agent.stream_request(job_id, spec.cmd), stream=True)
        return SessionStream(s, job, release)

    # ------------------------------------------------------------ port forwards (P3)
    def forward_blocker(self) -> str | None:
        if self.plan is None:
            return "not connected yet"
        if platform.IS_WINDOWS:
            return "port forwards need a shared connection, which Windows ssh lacks (planned; use WSL for forwards)"
        return 'port forwards are not available with mux = "session" (use mux = "auto" for them)'

    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return HostError(ErrorKind.MUX, self.forward_blocker() or "no port forwards over a session")

    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return HostError(ErrorKind.MUX, self.forward_blocker() or "no port forwards over a session")
