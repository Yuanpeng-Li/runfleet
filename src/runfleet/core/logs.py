"""[WP1] LogService: follow streams, line assembly, peek, fetch, pager argv (§4.7).

Streams yield raw blocks (`StreamHandle.chunks()`), never lines, so a multi-MB CR-only tqdm line cannot overflow a
reader limit. `LineAssembler` turns blocks into display lines with terminal semantics: `\\n` ends a line, `\\r` restarts
it (CRLF is a plain line end), lines are capped at 16 KiB, and the unterminated tail after its last `\\r` is the live
line shown under the log.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import os
import re
import shlex
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from runfleet import paths
from runfleet.core.clock import Clock
from runfleet.core.service import LogPeek, LogUpdate
from runfleet.errors import RunFleetError
from runfleet.models import Capabilities, HostError, LogRef
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, StreamSpec
from runfleet.transport.base import ProbeResult, StreamHandle, Transport
from runfleet.util import shell

log = logging.getLogger("runfleet.core.logs")

MAX_LINE = 16 * 1024
BATCH_INTERVAL = 0.1           # ≤ 10 updates/s
MAX_PENDING_LINES = 20_000     # LogView keeps 20 000 lines; older pending lines are useless
PEEK_LINES = 8
PEEK_MAX_AGE = 15.0
PEEK_HARD_MAX_AGE = 300.0      # even an unchanged logstat does not keep a peek longer (logstat may itself be stale)
STDERR_POLL = 0.5


class LineAssembler:
    """Incremental bytes -> lines with terminal CR semantics and a per-line cap (§4.7)."""

    def __init__(self, max_line: int = MAX_LINE) -> None:
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._max = max_line
        self._cur: list[str] = []
        self._len = 0              # UTF-8 bytes in _cur (the cap is a byte budget, like the "…[+N bytes]" suffix)
        self._dropped = 0          # bytes cut from the current line
        self._pending_cr = False   # a chunk ended in "\r": CRLF or a restart, decided by the next chunk

    def feed(self, data: bytes) -> list[str]:
        """Returns the lines completed by `data` (without their newline)."""
        return self._feed_text(self._dec.decode(data))

    def finish(self) -> list[str]:
        """At EOF: flush the decoder; the unterminated tail stays the live line."""
        return self._feed_text(self._dec.decode(b"", final=True))

    @property
    def live(self) -> str | None:
        text = self._render()
        return text or None

    def break_line(self) -> str | None:
        """End the current line where it is (the file was replaced or truncated); returns it unless empty."""
        self._pending_cr = False
        text = self._render()
        self._restart()
        return text or None

    def _feed_text(self, text: str) -> list[str]:
        if not text:
            return []
        if self._pending_cr:
            self._pending_cr = False
            if not text.startswith("\n"):
                self._restart()
        out: list[str] = []
        parts = text.split("\n")
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if not last:
                self._append(part[:-1] if part.endswith("\r") else part)   # CRLF = plain line end
                out.append(self._render())
                self._restart()
            else:
                if part.endswith("\r"):
                    self._pending_cr = True
                    part = part[:-1]
                self._append(part)
        return out

    def _restart(self) -> None:
        self._cur, self._len, self._dropped = [], 0, 0

    def _append(self, part: str) -> None:
        cr = part.rfind("\r")
        if cr >= 0:                 # "\r" restarts the line: keep only what follows the last one
            self._restart()
            part = part[cr + 1:]
        if not part:
            return
        data = part.encode("utf-8", errors="replace")
        room = self._max - self._len
        if room <= 0:
            self._dropped += len(data)
            return
        if len(data) > room:
            part = data[:room].decode("utf-8", errors="ignore")    # cut at a character boundary
            kept = len(part.encode("utf-8"))
            self._dropped += len(data) - kept
            data = data[:kept]
        self._cur.append(part)
        self._len += len(data)

    def _render(self) -> str:
        text = "".join(self._cur)
        return f"{text}…[+{self._dropped} bytes]" if self._dropped else text


ROTATED_NOTE = "file was replaced or truncated"
_NOTE_RULES = (
    (re.compile(r"Permission denied", re.I), "permission denied"),
    (re.compile(r"No such file|cannot open", re.I), "waiting for {path} to appear"),
    (re.compile(r"has appeared|has become accessible", re.I), "{path} appeared"),
    (re.compile(r"has been replaced|truncated", re.I), ROTATED_NOTE),
)


def stream_note(stderr: str, path: str) -> str | None:
    """Viewer note from the last meaningful `tail -F` stderr line."""
    for line in reversed([ln for ln in stderr.splitlines() if ln.strip()]):
        for rx, text in _NOTE_RULES:
            if rx.search(line):
                return text.format(path=path)
        return line.strip()[:120]
    return None


def tail_spec(path: str, nbytes: int) -> StreamSpec:
    """`log.tail` (Appendix A): initial tail and follow from one command; stdin is the remote watchdog."""
    p = shell.quote(shell.path(path))
    cmd = f"tail -c {int(nbytes)} -F -- {p} & p=$!; cat >/dev/null; kill $p 2>/dev/null"
    return StreamSpec("log.tail", cmd=cmd, meta={"path": path, "bytes": int(nbytes)})


def peek_section(path: str, nbytes: int) -> Section:
    p = shell.quote(shell.path(path))
    # -L: describe the file `tail` reads, not a `latest.log` symlink's own constant size/mtime
    return Section("log.peek", cmd=f"stat -L -c '%s|%Y' -- {p}; tail -c {int(nbytes)} -- {p}", timeout=10,
                   meta={"path": path, "bytes": int(nbytes)})


def fetch_section(path: str, nbytes: int) -> Section:
    p = shell.quote(shell.path(path))
    return Section("log.fetch", cmd=f"tail -c {int(nbytes)} -- {p}", timeout=30,
                   meta={"path": path, "bytes": int(nbytes)}, max_stdout=max(8_000_000, 2 * int(nbytes)))


def _file_error(stderr: str) -> str | None:
    if re.search(r"No such file", stderr):
        return "No such file"
    if re.search(r"Permission denied", stderr):
        return "Permission denied"
    first = next((ln.strip() for ln in stderr.splitlines() if ln.strip()), "")
    return first[:120] or None


def parse_peek(stdout: str, stderr: str, nbytes: int, now: datetime) -> LogPeek:
    """`size|mtime` line, then the tail bytes -> last ≤ 8 complete lines (+ the live line)."""
    head, _, body = stdout.partition("\n")
    m = re.fullmatch(r"(\d+)\|(\d+)", head.strip())
    if m is None:
        return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error=_file_error(stderr) or "no output")
    size, mtime = int(m.group(1)), datetime.fromtimestamp(int(m.group(2)), tz=UTC)
    if size > nbytes and body:
        # `tail -c` started mid-line: drop the fragment before the first line boundary. A "\r" boundary counts too
        # (what follows it starts at column 0), so a CR-only tqdm tail keeps its last state; no boundary at all is
        # one huge line, shown marked as cut.
        cut = min((i for i in (body.find("\n"), body.find("\r")) if i >= 0), default=-1)
        if cut < 0:
            body = "…" + body
        else:
            body = body[cut + 2:] if body.startswith("\r\n", cut) else body[cut + 1:]
    # SectionResult.stdout is the exact command output (§4.3), so an unterminated tail is the live line (tqdm "\r"
    # progress collapses to its last state); it is appended after the complete lines.
    asm = LineAssembler()
    lines = asm.feed(body.encode("utf-8", errors="replace")) if body else []
    lines += asm.finish()
    live = asm.live
    if live is not None:
        lines.append(live)
    shown = lines[-PEEK_LINES:]
    return LogPeek(lines=tuple(shown), size=size, mtime=mtime, fetched_at=now)


# ======================================================================================= streaming sessions
class StreamSession:
    """LogSession over one StreamHandle: initial tail, then follow; batched ≤ 10/s.

    The ≤ 10/s limiter uses loop time (`asyncio.sleep`), not the injected Clock: it paces screen updates, it is not
    a schedule, and it must not stall under a ManualClock in UI tests.
    """

    def __init__(self, host: str, ref: LogRef, handle: StreamHandle, *,
                 on_close: Callable[[StreamSession], None] | None = None,
                 batch_interval: float = BATCH_INTERVAL) -> None:
        self.host = host
        self.ref = ref
        self.handle = handle
        self._on_close = on_close
        self._batch = batch_interval
        self._asm = LineAssembler()
        self._pending: deque[str] = deque(maxlen=MAX_PENDING_LINES)
        self._live: str | None = None
        self._live_sent: str | None = None
        self._note_sent: str | None = None
        self._err_seen = ""
        self._dirty = asyncio.Event()
        self._eof = False
        self._closed = False
        self._close_task: asyncio.Future[None] | None = None
        self._reader: asyncio.Task[None] | None = None

    async def _read(self) -> None:
        try:
            async for chunk in self.handle.chunks():
                self._pending.extend(self._asm.feed(chunk))
                self._live = self._asm.live
                self._dirty.set()
            self._pending.extend(self._asm.finish())
            self._live = self._asm.live
        except Exception as e:           # a dying ssh must end the stream, not the app
            log.info("%s: log stream error: %s", self.host, e)
        finally:
            self._eof = True
            self._dirty.set()

    def _note(self) -> tuple[str | None, bool]:
        """The current note, and whether new stderr just reported a rotation (repeated rotations repeat the note)."""
        err = self.handle.stderr_text
        note = stream_note(err, self.ref.path or "?")
        fresh, self._err_seen = err != self._err_seen, err
        return note, fresh and note == ROTATED_NOTE

    async def updates(self) -> AsyncIterator[LogUpdate]:
        if self._reader is None:
            self._reader = asyncio.ensure_future(self._read())
        while not self._closed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._dirty.wait(), STDERR_POLL)
            self._dirty.clear()
            note, rotated = self._note()
            if rotated:
                # The old file's unterminated last line must not run into the new file's first one. stderr and
                # stdout are not ordered, so the boundary is approximate.
                if (cut := self._asm.break_line()) is not None:
                    self._pending.append(cut)
                self._live = None
            lines = tuple(self._pending)
            self._pending.clear()
            live_changed = self._live != self._live_sent
            note_changed = note is not None and (note != self._note_sent or rotated)
            eof = self._eof and not self._pending
            if eof and note is None:
                note = "stream ended"
                note_changed = note != self._note_sent
            if lines or live_changed or note_changed or eof:
                self._live_sent = self._live
                if note_changed:
                    self._note_sent = note
                yield LogUpdate(lines=lines, live=self._live, note=note if note_changed else None, eof=eof)
            if eof:
                return
            await asyncio.sleep(self._batch)

    async def close(self) -> None:
        """Idempotent; every caller waits for the one shutdown (it survives a caller's cancellation)."""
        if self._close_task is None:
            self._closed = True
            self._dirty.set()
            self._close_task = asyncio.ensure_future(self._shutdown())
        await asyncio.shield(self._close_task)

    async def _shutdown(self) -> None:
        try:
            await self.handle.close()
        finally:
            if self._reader is not None and not self._reader.done():
                self._reader.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._reader
            if self._on_close is not None:
                self._on_close(self)


# ======================================================================================= the service
OnDemandProbe = Callable[[str, ProbeScript], Awaitable[ProbeResult]]


@dataclass
class _Peeked:
    peek: LogPeek
    at: float


class LogService:
    """Peek (cached one-shot probe), fetch (pager fallback), follow streams (lane-limited), pager argv."""

    def __init__(self, clock: Clock, *, transport_for: Callable[[str], Transport],
                 stream_limit_for: Callable[[str], int], probe: OnDemandProbe,
                 caps_for: Callable[[str], Capabilities | None], pager: str = "",
                 ensure_master: Callable[[str], Awaitable[HostError | None]] | None = None) -> None:
        self.clock = clock
        self._ensure_master = ensure_master
        self._transport_for = transport_for
        self._stream_limit_for = stream_limit_for
        self._probe = probe
        self._caps_for = caps_for
        self._pager = pager
        self._streams: dict[str, list[StreamSession]] = {}
        self._admit: dict[str, asyncio.Lock] = {}      # per host: lane check + open are one step
        self._peeks: dict[tuple[str, str], _Peeked] = {}

    # ------------------------------------------------------------ follow
    async def stream(self, host: str, ref: LogRef, *, tail_bytes: int = 65536) -> StreamSession:
        if not ref.path:
            raise RunFleetError("log path unknown")
        spec = tail_spec(ref.path, tail_bytes)
        async with self._admit.setdefault(host, asyncio.Lock()):
            open_ = self._streams.setdefault(host, [])
            while open_ and len(open_) >= max(1, self._stream_limit_for(host)):
                oldest = open_[0]
                await oldest.close()               # opening beyond the lane closes the oldest (§4.2)
                with contextlib.suppress(ValueError):
                    open_.remove(oldest)           # normally _forget did; a detached list must still shrink
            if self._ensure_master is not None:    # streams are bounded by stream_limit, not the global semaphore
                err = await self._ensure_master(host)
                if err is not None:
                    raise RunFleetError(err.summary)
            handle = await self._transport_for(host).stream(spec)
            if self._streams.get(host) is not open_:          # the host was removed meanwhile (close_host)
                await handle.close()
                raise RunFleetError(f"{host}: host removed")
            session = StreamSession(host, ref, handle, on_close=self._forget)
            open_.append(session)
            return session

    def _forget(self, session: StreamSession) -> None:
        lst = self._streams.get(session.host, [])
        if session in lst:
            lst.remove(session)

    def open_streams(self, host: str) -> int:
        return len(self._streams.get(host, []))

    async def close_host(self, host: str) -> None:
        """The host left the fleet (Settings): close its streams, forget its cached peeks."""
        for s in list(self._streams.pop(host, [])):
            await s.close()
        self._peeks = {k: v for k, v in self._peeks.items() if k[0] != host}

    def set_pager(self, pager: str) -> None:
        self._pager = pager

    async def close_all(self) -> None:
        for lst in list(self._streams.values()):
            for s in list(lst):
                await s.close()

    # ------------------------------------------------------------ peek / fetch
    async def peek(self, host: str, ref: LogRef, *, nbytes: int = 4096) -> LogPeek:
        now = self.clock.now()
        if not ref.path:
            return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error=ref.error or "log path unknown")
        key = (host, ref.path)
        cached = self._peeks.get(key)
        mono = self.clock.monotonic()
        if cached is not None:
            fresh = mono - cached.at < PEEK_MAX_AGE
            unchanged = (ref.size is not None and ref.size == cached.peek.size
                         and ref.mtime is not None and ref.mtime == cached.peek.mtime)
            if fresh or (unchanged and mono - cached.at < PEEK_HARD_MAX_AGE):
                return cached.peek
        script = framing.make_script([peek_section(ref.path, nbytes)], "log")
        res = await self._probe(host, script)
        if res.error is not None:
            return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error=res.error.summary)
        r = res.sections["log.peek"]
        peek = parse_peek(r.stdout, r.stderr, nbytes, now)
        self._peeks[key] = _Peeked(peek, mono)
        return peek

    async def fetch(self, host: str, ref: LogRef, *, nbytes: int = 4 << 20) -> Path:
        if not ref.path:
            raise RunFleetError("log path unknown")
        script = framing.make_script([fetch_section(ref.path, nbytes)], "log")
        res = await self._probe(host, script)
        if res.error is not None:
            raise RunFleetError(f"{host}: {res.error.summary}")
        r = res.sections["log.fetch"]
        if not r.ok and not r.stdout:
            raise RunFleetError(f"{host}: {_file_error(r.stderr) or r.outcome.value}")
        fd, name = tempfile.mkstemp(prefix=f"runfleet-{host}-", suffix=".log")   # 0600
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(r.stdout + ("\n" if r.stdout else ""))
        return Path(name)

    # ------------------------------------------------------------ pagers
    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None:
        caps = self._caps_for(host)
        if not ref.path or caps is None or "less" not in caps.tools:
            return None
        cmd = f"less -R +G -- {shell.quote(shell.path(ref.path))}"
        return self._transport_for(host).interactive_argv(cmd)

    def local_pager_argv(self, path: Path | None = None) -> list[str]:
        spec = self._pager or os.environ.get("PAGER") or "less -R +G"
        try:
            argv = shlex.split(spec)
        except ValueError:
            argv = ["less", "-R", "+G"]
        return [*(argv or ["less", "-R", "+G"]), str(path if path is not None else paths.log_file())]
