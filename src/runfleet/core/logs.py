"""[WP1] LogService: follow streams, line assembly, peek, read, fetch, search (§4.7.1), find (§4.7.2), pager argv.

Streams yield raw blocks (`StreamHandle.chunks()`), never lines, so a multi-MB CR-only tqdm line cannot overflow a
reader limit. `LineAssembler` turns blocks into display lines with terminal semantics: `\\n` ends a line, `\\r` restarts
it (CRLF is a plain line end), lines are capped at 16 KiB, and the unterminated tail after its last `\\r` is the live
line shown under the log.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import dataclasses
import logging
import math
import os
import posixpath
import re
import shutil
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from runfleet import paths
from runfleet.core import logsearch
from runfleet.core.clock import Clock
from runfleet.core.logsearch import LogQuery, LogSearchResult, SearchPlan
from runfleet.core.service import LogPeek, LogUpdate
from runfleet.errors import RunFleetError
from runfleet.models import Capabilities, ErrorKind, HostError, Job, JobKey, LogRef
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, StreamSpec
from runfleet.transport.base import ProbeResult, StreamHandle, Transport
from runfleet.util import platform, shell

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


def tail_spec(path: str, nbytes: int, start: int | None = None) -> StreamSpec:
    """`log.tail` (Appendix A): initial tail and follow from one command; stdin is the remote watchdog. ``start``
    (a log search hit, §4.7.1): from that absolute byte offset instead of the last ``nbytes`` (``tail -c +K``)."""
    p = shell.quote(shell.path(path))
    first = f"+{int(start) + 1}" if start is not None and int(start) >= 0 else str(int(nbytes))
    cmd = f"tail -c {first} -F -- {p} & p=$!; cat >/dev/null; kill $p 2>/dev/null"
    meta: dict[str, object] = {"path": path, "bytes": int(nbytes)}
    if start is not None and int(start) >= 0:
        meta["start"] = int(start)
    return StreamSpec("log.tail", cmd=cmd, meta=meta)


def peek_section(path: str, nbytes: int) -> Section:
    p = shell.quote(shell.path(path))
    # -L: describe the file `tail` reads, not a `latest.log` symlink's own constant size/mtime
    return Section("log.peek", cmd=f"stat -L -c '%s|%Y' -- {p}; tail -c {int(nbytes)} -- {p}", timeout=10,
                   meta={"path": path, "bytes": int(nbytes)})


HEAD_BYTES = 32768             # start of a log shown when its tail holds only "\r" progress (tqdm), §4.7
HEAD_LINES = 200


def head_section(path: str, nbytes: int) -> Section:
    """`log.head`: size/mtime, then the first `nbytes` (one read, on demand: the follower asks for it once when the
    tail has no complete line, so the pane shows the log's start instead of nothing)."""
    p = shell.quote(shell.path(path))
    return Section("log.head", cmd=f"stat -L -c '%s|%Y' -- {p}; head -c {int(nbytes)} -- {p}", timeout=10,
                   meta={"path": path, "bytes": int(nbytes)}, max_stdout=2 * int(nbytes) + 256)


def parse_head(stdout: str, stderr: str, nbytes: int, now: datetime) -> LogPeek:
    """`size|mtime` line, then the head bytes -> its complete lines (``\r`` progress collapsed to its last state, at
    most HEAD_LINES); an unterminated end (the read stopped mid-line, or a progress line) is left out."""
    head, _, body = stdout.partition("\n")
    m = re.fullmatch(r"(\d+)\|(\d+)", head.strip())
    if m is None:
        return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error=_file_error(stderr) or "no output")
    size, mtime = int(m.group(1)), datetime.fromtimestamp(int(m.group(2)), tz=UTC)
    asm = LineAssembler()
    lines = asm.feed(body.encode("utf-8", errors="replace")) if body else []
    return LogPeek(lines=tuple(lines[:HEAD_LINES]), size=size, mtime=mtime, fetched_at=now)


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


@dataclass(frozen=True, slots=True)
class LogRead:
    """One bounded read of a log for `runfleet agent log` (§4.7): every complete line of the window, in memory."""
    path: str
    lines: tuple[str, ...]          # complete lines, "\r" progress collapsed (ANSI and control characters kept)
    live: str | None                # the unterminated last segment (tqdm's current state); None in head mode
    size: int | None                # file size / mtime at the read (None: unknown)
    mtime: datetime | None
    window_bytes: int               # bytes the window covered: min(size, nbytes), else the requested nbytes
    started_mid_file: bool          # tail mode and the file is larger than the window (its first fragment dropped)
    fetched_at: datetime
    error: HostError | None = None  # transport failure (typed: auth, hostkey, timeout …); no lines then
    file_error: str | None = None   # "No such file", "Permission denied", "log path unknown", …
    partial: bool = False           # the section failed after some output (timeout, connection cut): lines may be few


def _stat_line(stdout: str) -> tuple[int, datetime, str] | None:
    head, _, body = stdout.partition("\n")
    m = re.fullmatch(r"(\d+)\|(\d+)", head.strip())
    if m is None:
        return None
    return int(m.group(1)), datetime.fromtimestamp(int(m.group(2)), tz=UTC), body


def parse_read(stdout: str, stderr: str, nbytes: int, now: datetime, path: str) -> LogRead:
    """`log.peek` output (`size|mtime`, then the last `nbytes`) -> ALL complete lines of the window plus the live
    segment; the cut first fragment is dropped when the file is larger than the window (as :func:`parse_peek`)."""
    parsed = _stat_line(stdout)
    if parsed is None:
        return LogRead(path=path, lines=(), live=None, size=None, mtime=None, window_bytes=nbytes,
                       started_mid_file=False, fetched_at=now, file_error=_file_error(stderr) or "no output")
    size, mtime, body = parsed
    mid = size > nbytes
    if mid and body:
        cut = min((i for i in (body.find("\n"), body.find("\r")) if i >= 0), default=-1)
        if cut < 0:
            body = "…" + body
        else:
            body = body[cut + 2:] if body.startswith("\r\n", cut) else body[cut + 1:]
    asm = LineAssembler()
    lines = asm.feed(body.encode("utf-8", errors="replace")) if body else []
    lines += asm.finish()
    return LogRead(path=path, lines=tuple(lines), live=asm.live, size=size, mtime=mtime,
                   window_bytes=min(size, nbytes), started_mid_file=mid, fetched_at=now)


def _read_from_head(peek: LogPeek, nbytes: int, path: str) -> LogRead:
    return LogRead(path=path, lines=peek.lines, live=None, size=peek.size, mtime=peek.mtime,
                   window_bytes=min(peek.size, nbytes) if peek.size is not None else nbytes,
                   started_mid_file=False, fetched_at=peek.fetched_at, file_error=peek.error)


# ======================================================================================= find other logs (§4.7.2)
FIND_MAX_FILES = 30
FIND_MAX_DEPTH = 6
FIND_SLACK = 60.0              # files written from 60 s before `since` (the job's start) on
FIND_WINDOW = 86400.0          # since None: the last 24 h
FIND_END_SLACK = 300.0         # an ended job (`until`): files written up to 5 min after its end (flushes, teardown)
FIND_TIMEOUT = 15              # the section (remote `timeout -k 2`)
FIND_WALK = 12                 # `find` alone, so sort/head still print what it found when a tree is slow
FIND_NAMES = ("*.log", "*.out", "*.err", "*.stdout", "*.stderr")      # wandb's files/output.log is a *.log
FIND_PRUNE = (".git", ".venv*", "venv", "node_modules", "__pycache__", ".cache", ".mypy_cache", ".pytest_cache",
              "site-packages")
FIND_MAX_STDOUT = (FIND_MAX_FILES + 1) * 4200 + 256    # PATH_MAX-long lines at worst
FIND_UNSUPPORTED = "unsupported: needs GNU find (-printf, -newermt)"


@dataclass(frozen=True, slots=True)
class FoundLog:
    """A file under a job's workdir that looks like a log and was written since the job started."""
    path: str                       # absolute, under LogFiles.workdir
    size: int
    mtime: datetime


@dataclass(frozen=True, slots=True)
class LogFiles:
    """One `log.find` listing (§4.7.2): newest first, at most FIND_MAX_FILES."""
    workdir: str                    # normalised (what the probe searched)
    files: tuple[FoundLog, ...]
    fetched_at: datetime
    error: HostError | None = None  # transport failure (typed: auth, hostkey, timeout …); no files then
    file_error: str | None = None   # "No such directory", "Permission denied", FIND_UNSUPPORTED, …
    truncated: bool = False         # more than FIND_MAX_FILES matched; only the newest are listed
    incomplete: str | None = None   # why the walk may have missed files: "timed out", "output cut", "sort failed",
                                    # "some directories unreadable", "find failed (rc N)"


def find_workdir(workdir: str) -> str:
    """The workdir a probe may search: absolute, one line (`util.shell.path`), no `..` (a remote symlink before it
    would make a lexical normalisation name another directory), normalised, never `/`."""
    if not workdir:
        raise ValueError("workdir unknown")
    if ".." in shell.path(workdir).split("/"):
        raise ValueError("refusing a workdir with '..'")
    wd = posixpath.normpath(workdir)
    wd = "/" + wd.lstrip("/")                      # normpath keeps a leading "//"
    if wd == "/":
        raise ValueError("refusing to search / (workdir is the filesystem root)")
    return wd


def find_cutoff(since: datetime | None, now: datetime) -> int:
    """Epoch seconds `find -newermt` compares with: `since` - FIND_SLACK, else now - FIND_WINDOW."""
    at = since - timedelta(seconds=FIND_SLACK) if since is not None else now - timedelta(seconds=FIND_WINDOW)
    return max(0, int(at.timestamp()))


def find_until(until: datetime | None) -> int | None:
    """Epoch seconds `! -newermt` compares with for an ended job: its end + FIND_END_SLACK; None = no upper bound."""
    return None if until is None else max(0, math.ceil(until.timestamp()) + int(FIND_END_SLACK))


def find_window(job: Job) -> tuple[datetime | None, datetime | None]:
    """(since, until) of a job's `log.find`, one rule for the TUI and the agent: since its start (a scheduler
    estimate is no start), until its end once it ended (a running job's end_time is only its time limit). An ended
    job with no usable start: the FIND_WINDOW before its end, never the default "last 24 h" from now (which misses,
    or inverts against `until`, a job that ended days ago). (None, None): the last 24 h."""
    until = job.end_time if job.state.is_terminal else None
    since = job.start_time if job.start_time is not None and not job.est_start else None
    if since is None and until is not None:
        since = until - timedelta(seconds=FIND_WINDOW)
    return since, until


def find_command(workdir: str, cutoff: int, until: int | None = None) -> str:
    """`log.find` (Appendix A): ``@@nodir`` | ``@@denied`` | ``@@unsupported`` alone, or up to FIND_MAX_FILES + 1
    ``MTIME SIZE PATH\\037`` lines (``%T@ %s %p\\037``, newest first; ``@@sort RC`` when sort failed), then
    ``@@end RC`` (find's exit status; 124/137 = the walk was stopped). ``-H``: only the workdir itself may be a
    symlink; symlinked files and directories under it are neither listed nor entered. Names with control characters
    (a newline) are skipped; `until`: only files modified at or before it (an ended job)."""
    d = shell.quote(find_workdir(workdir))
    prune = " -o ".join(f"-name {shell.quote(n)}" for n in FIND_PRUNE)
    names = " -o ".join(f"-name {shell.quote(n)}" for n in FIND_NAMES)
    walk = (f"$t find -H \"$d\" -mindepth 1 -maxdepth {FIND_MAX_DEPTH} \\( -type d \\( {prune} \\) \\) -prune "
            f"-o -type f \\( {names} \\) ! -path '*[[:cntrl:]]*' -newermt @{int(cutoff)} "
            + (f"! -newermt @{int(until)} " if until is not None else "") + "-printf '%T@ %s %p\\037\\n'")
    return "\n".join((
        f"d={d}",
        "[ -d \"$d\" ] || { echo @@nodir; exit 0; }",
        "{ [ -r \"$d\" ] && [ -x \"$d\" ]; } || { echo @@denied; exit 0; }",
        # a failed check is GNU find missing, unless the workdir went away since `[ -d ]`
        "find -H \"$d\" -maxdepth 0 -newermt @0 -printf '' >/dev/null 2>&1 || "
        "{ [ -d \"$d\" ] || { echo @@nodir; exit 0; }; echo @@unsupported; exit 0; }",
        f"if command -v timeout >/dev/null 2>&1; then t='timeout -k 1 {FIND_WALK}'; else t=; fi",
        "exec 3>&1",
        # find's status leaves the pipeline on fd 4; the sorted list goes straight to stdout (fd 3)
        f"r=$({{ {{ {walk}; echo $? >&4; }} | {{ sort -rn || echo \"@@sort $?\"; }} | head -n {FIND_MAX_FILES + 1} "
        ">&3; } 4>&1)",
        # a failed walk whose workdir is gone: find escapes the name it prints (LC_ALL=C), never match it in stderr
        "[ \"$r\" = 0 ] || [ -d \"$d\" ] || { echo @@nodir; exit 0; }",
        "echo \"@@end $r\"",
    ))


def find_section(workdir: str, since: datetime | None, now: datetime, until: datetime | None = None) -> Section:
    """`log.find`: recently written log-like files under a job's workdir (on demand, one probe); `until` = an ended
    job's end (files written after it + FIND_END_SLACK are left out). Raises ValueError for a workdir no probe may
    search (relative, several lines, `/`, `..`); OverflowError for a since / until out of range."""
    wd = find_workdir(workdir)
    cutoff = find_cutoff(since, now)
    end = find_until(until)
    return Section("log.find", cmd=find_command(wd, cutoff, end), timeout=FIND_TIMEOUT,
                   requires=("find", "sort", "head"),
                   meta={"workdir": wd, "since": cutoff, "until": end, "max_depth": FIND_MAX_DEPTH,
                         "max_files": FIND_MAX_FILES},
                   max_stdout=FIND_MAX_STDOUT)


_FIND_LINE = re.compile(r"(\d+(?:\.\d*)?) (\d+) (/[^\x1f]*)\x1f")      # \037 ends a whole record
_FIND_MARKS = {"@@nodir": "No such directory", "@@denied": "Permission denied", "@@unsupported": FIND_UNSUPPORTED}
_FIND_DENIED = re.compile(r"find: .*: Permission denied")    # LC_ALL=C (framing): a private subdirectory
# removed / renamed while find walked (a temp dir): that entry is not listed. Old findutils quote with `…'.
_FIND_GONE = re.compile(r"find: [`']?(.*?)'?: No such file or directory")
FIND_TIMED_OUT = "timed out"
FIND_CUT = "output cut"
FIND_SORT_FAILED = "sort failed"
FIND_UNREADABLE = "some directories unreadable"
FIND_NO_OUTPUT = "no output"


def _stderr_lines(stderr: str) -> list[str]:
    """find's stderr as whole, stripped lines. The framing keeps at most STDERR_LIMIT bytes (``head -c``): a capped
    stderr that does not end in a newline ends in a cut line (``find: '/w/pri``) that matches nothing: dropped."""
    lines = stderr.split("\n")
    if not stderr.endswith("\n") and len(stderr.encode("utf-8", "replace")) >= framing.STDERR_LIMIT:
        lines.pop()
    return [s for ln in lines if (s := ln.strip())]


def _find_failure(rc: str, stderr: str, workdir: str) -> tuple[str | None, str | None]:
    """find's non-zero status -> (file_error when nothing was listed, incomplete when something was). Unreadable
    subdirectories alone are no error, only an incomplete walk; an entry below the workdir removed or renamed during
    the walk is tolerated (it is simply not listed), the workdir itself gone is "No such directory"."""
    failed = f"find failed (rc {rc[:8] or '?'})"
    lines = _stderr_lines(stderr)
    gone = [m.group(1) for ln in lines if (m := _FIND_GONE.fullmatch(ln)) is not None]
    if workdir in gone or workdir.rstrip("/") in gone:     # removed / renamed after the `[ -d ]` check
        return "No such directory", failed
    rest = [ln for ln in lines if not _FIND_GONE.fullmatch(ln)]
    if not rest:
        return (None, None) if gone else (failed, failed)
    if all(_FIND_DENIED.fullmatch(ln) for ln in rest):
        return None, FIND_UNREADABLE
    other = "\n".join(ln for ln in rest if not _FIND_DENIED.fullmatch(ln))    # the real cause, not a private dir
    return _file_error(other) or failed, failed


def parse_find(stdout: str, stderr: str, workdir: str, now: datetime) -> LogFiles:
    """`log.find` output -> newest first, ≤ FIND_MAX_FILES (`truncated` when more matched). A path outside the
    workdir or not valid UTF-8 is dropped. A walk that may have missed files keeps what it listed and says why in
    `incomplete` (timed out, output cut, sort failed, unreadable subdirectories, find failed); with nothing listed,
    a real failure is a `file_error` instead (an I/O error is never "no log files")."""
    try:
        wd = find_workdir(workdir)
    except ValueError:
        wd = workdir
    prefix = wd.rstrip("/") + "/"
    rows: list[tuple[float, FoundLog]] = []
    records = 0                                      # whole records, kept or dropped: what `head` let through
    end: str | None = None
    sort_rc: str | None = None
    # A record cut mid-path (find killed by its timeout, then sort adds the newline; max_stdout) lacks its \037 end:
    # it never matches, so a prefix of a name is never offered.
    for line in stdout.split("\n"):
        if line.startswith("@@"):
            tag, _, rest = line.partition(" ")
            if tag in _FIND_MARKS:
                return LogFiles(workdir=wd, files=(), fetched_at=now, file_error=_FIND_MARKS[tag])
            if tag == "@@end":
                end = rest.strip()
            elif tag == "@@sort":
                sort_rc = rest.strip()
            continue
        m = _FIND_LINE.fullmatch(line)
        if m is None:
            continue
        records += 1
        path = m.group(3)
        if not path.startswith(prefix) or "\ufffd" in path or "/../" in path or path.endswith("/.."):
            continue
        try:
            ts = float(m.group(1))
            rows.append((ts, FoundLog(path=path, size=int(m.group(2)), mtime=datetime.fromtimestamp(ts, tz=UTC))))
        except (ValueError, OverflowError, OSError):
            continue
    rows.sort(key=lambda r: (-r[0], r[1].path))
    files = tuple(f for _, f in rows)
    error: str | None = None
    incomplete: str | None = None
    if end in ("124", "137"):
        incomplete = FIND_TIMED_OUT                  # `timeout` stopped the walk: what it found so far, if anything
    elif end is None:
        incomplete = FIND_CUT
        error = _file_error("\n".join(ln for ln in _stderr_lines(stderr) if not _FIND_DENIED.fullmatch(ln)))
        error = error or FIND_NO_OUTPUT
    elif sort_rc is not None:
        incomplete = FIND_SORT_FAILED
        error = next((ln.strip()[:120] for ln in stderr.splitlines() if ln.startswith("sort:")),
                     f"sort failed (rc {sort_rc[:8]})")
    elif end != "0":
        error, incomplete = _find_failure(end, stderr, wd)
    if error is not None and not files:
        return LogFiles(workdir=wd, files=(), fetched_at=now, file_error=error)
    return LogFiles(workdir=wd, files=files[:FIND_MAX_FILES], fetched_at=now, truncated=records > FIND_MAX_FILES,
                    incomplete=incomplete)


def norm_log_path(path: str) -> str:
    """The form `known_logs` / `suggest` compare paths in: normalised, one leading `/` (`//a/./b` -> `/a/b`)."""
    return "/" + posixpath.normpath(path).lstrip("/")


def known_logs(jobs: Iterable[Job], exclude: JobKey) -> dict[str, str]:
    """The stdout / stderr files of the OTHER jobs on `exclude`'s host (a shared workdir holds every job's
    ``logs/<name>_<id>.out``) -> "job <label> stdout|stderr", so a picker / agent marks them and never suggests one."""
    out: dict[str, str] = {}
    for job in jobs:
        if job.key == exclude or job.key.host != exclude.host:
            continue
        for ref in job.logs:
            if ref.path and ref.path.startswith("/") and ref.stream in ("stdout", "stderr"):
                out.setdefault(norm_log_path(ref.path), f"job {job.key.label} {ref.stream}")
    return out


def suggest(files: Sequence[FoundLog], skip: Collection[str]) -> FoundLog | None:
    """The newest found file worth opening first: not in `skip` (the job's own stdout / stderr and `known_logs`;
    compared normalised). None when every file is skipped: no suggestion then."""
    skipped = {norm_log_path(p) for p in skip if p}
    return next((f for f in sorted(files, key=lambda f: f.mtime, reverse=True)
                 if norm_log_path(f.path) not in skipped), None)


QUIET_TINY = 4096              # bytes: a running job's stdout this small …
QUIET_TINY_AGE = 600.0         # … and unchanged this long holds a launcher's few lines: the real log is elsewhere


def stdout_quiet_for(ref: LogRef | None, now: datetime, stall_after: float) -> float | None:
    """Seconds since a stdout last changed when it is "quiet": unchanged longer than `stall_after` (the "log idle"
    rule), or ≤ QUIET_TINY bytes and unchanged ≥ QUIET_TINY_AGE. None otherwise, or with no mtime. One rule for the
    TUI's hint and the agent's "next" line (§4.7.2)."""
    if ref is None or ref.mtime is None:
        return None
    quiet = (now - ref.mtime).total_seconds()
    if quiet > stall_after or (ref.size is not None and ref.size <= QUIET_TINY and quiet >= QUIET_TINY_AGE):
        return quiet
    return None


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
    async def stream(self, host: str, ref: LogRef, *, tail_bytes: int = 65536,
                     start: int | None = None) -> StreamSession:
        if not ref.path:
            raise RunFleetError("log path unknown")
        spec = tail_spec(ref.path, tail_bytes, start)
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

    async def head(self, host: str, ref: LogRef, *, nbytes: int = HEAD_BYTES) -> LogPeek:
        """The start of a log (its complete lines), read once on demand; never raises."""
        now = self.clock.now()
        if not ref.path:
            return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error="log path unknown")
        script = framing.make_script([head_section(ref.path, nbytes)], "log")
        res = await self._probe(host, script)
        if res.error is not None:
            return LogPeek(lines=(), size=None, mtime=None, fetched_at=now, error=res.error.summary)
        r = res.sections["log.head"]
        return parse_head(r.stdout, r.stderr, nbytes, now)

    async def read(self, host: str, ref: LogRef, *, nbytes: int, head: bool = False) -> LogRead:
        """ONE probe: the log's last `nbytes` (`log.peek`: stat + `tail -c`), or with `head` its start (`log.head`,
        ≤ HEAD_LINES lines). In memory, never cached; never raises for host problems (they are `error`,
        `file_error` or `partial`)."""
        now = self.clock.now()
        if not ref.path:
            return LogRead(path="", lines=(), live=None, size=None, mtime=None, window_bytes=nbytes,
                           started_mid_file=False, fetched_at=now, file_error="log path unknown")
        path = ref.path
        empty = LogRead(path=path, lines=(), live=None, size=None, mtime=None, window_bytes=nbytes,
                        started_mid_file=False, fetched_at=now)
        try:
            section = head_section(path, nbytes) if head else peek_section(path, nbytes)
        except ValueError as e:            # a remote path no shell command can carry (a newline in it)
            return dataclasses.replace(empty, file_error=(str(e) or "invalid log path")[:120])
        try:
            res = await self._probe(host, framing.make_script([section], "log"))
        except Exception as e:             # an unknown host, a broken transport: a typed error, not a crash
            log.info("%s: log read failed: %r", host, e)
            return dataclasses.replace(empty, error=HostError(ErrorKind.INTERNAL, (str(e) or type(e).__name__)[:80],
                                                              at=now))
        if res.error is not None:
            return dataclasses.replace(empty, error=res.error)
        r = res.sections.get(section.name)
        if r is None:
            return dataclasses.replace(empty, file_error="no output")
        if not r.ok and not r.stdout:
            known = _file_error(r.stderr)
            if known is None and r.outcome is SectionOutcome.TIMEOUT:      # no byte came: a typed timeout
                return dataclasses.replace(empty, error=HostError(ErrorKind.TIMEOUT, "log read timed out",
                                                                  section=section.name, at=now))
            return dataclasses.replace(empty, file_error=known or r.outcome.value)
        try:
            out = (_read_from_head(parse_head(r.stdout, r.stderr, nbytes, now), nbytes, path) if head
                   else parse_read(r.stdout, r.stderr, nbytes, now, path))
        except (ValueError, OverflowError, OSError):    # a stat line out of range: remote data, not our bug
            return dataclasses.replace(empty, file_error="unreadable stat output")
        if r.ok:
            return out
        # stat answered but the read failed (unreadable contents) or was cut: keep what came, say why
        return dataclasses.replace(out, partial=True, file_error=out.file_error or _file_error(r.stderr))

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

    # ------------------------------------------------------------ search (§4.7.1)
    async def search(self, host: str, search: SearchPlan, query: LogQuery, *,
                     max_matches: int = logsearch.MAX_MATCHES) -> LogSearchResult:
        """One `log.search` probe over the plan's files (on demand only); never raises."""
        problem = logsearch.pattern_problem(query)
        if problem is not None:
            return LogSearchResult(host, query, search, max_matches=max_matches, error=problem)
        if not search.files:
            return LogSearchResult(host, query, search, max_matches=max_matches)
        section = logsearch.search_section([f.path for f in search.files], query, max_matches=max_matches)
        try:
            res = await self._probe(host, framing.make_script([section], "log"))
        except Exception as e:             # a search problem is a line in the results, never an app error
            return LogSearchResult(host, query, search, max_matches=max_matches, error=str(e)[:120] or "failed")
        if res.error is not None:
            return LogSearchResult(host, query, search, max_matches=max_matches, error=res.error.summary)
        return logsearch.parse(host, query, search, res.sections[logsearch.SECTION], max_matches=max_matches)

    # ------------------------------------------------------------ find other logs (§4.7.2)
    async def find(self, host: str, workdir: str, *, since: datetime | None,
                   until: datetime | None = None) -> LogFiles:
        """ONE `log.find` probe: log-like files written under `workdir` since `since` (the job's start; None = the
        last 24 h) and, for an ended job, not after `until` (its end) + FIND_END_SLACK. On demand only, never
        cached; never raises (`error`, `file_error`, `incomplete`, `truncated`)."""
        now = self.clock.now()
        empty = LogFiles(workdir=workdir, files=(), fetched_at=now)
        try:
            section = find_section(workdir, since, now, until)
        except (ValueError, OverflowError) as e:     # relative, several lines, `/`, `..`, a since out of range
            return dataclasses.replace(empty, file_error=(str(e) or "invalid workdir")[:120])
        wd = str(section.meta["workdir"])
        empty = dataclasses.replace(empty, workdir=wd)
        try:
            res = await self._probe(host, framing.make_script([section], "log"))
        except Exception as e:             # an unknown host, a broken transport: a typed error, not a crash
            log.info("%s: log find failed: %r", host, e)
            return dataclasses.replace(empty, error=HostError(ErrorKind.INTERNAL, (str(e) or type(e).__name__)[:80],
                                                              at=now))
        if res.error is not None:
            return dataclasses.replace(empty, error=res.error)
        r = res.sections.get(section.name)
        if r is None:
            return dataclasses.replace(empty, file_error=FIND_NO_OUTPUT)
        if r.outcome is SectionOutcome.MISSING:
            tool = r.stderr.partition("missing:")[2].strip() or "find"
            return dataclasses.replace(empty, file_error=f"unsupported: {tool} not installed"[:120])
        if not r.ok and not r.stdout:
            known = _file_error(r.stderr)
            if known is None and r.outcome is SectionOutcome.TIMEOUT:
                return dataclasses.replace(empty, error=HostError(ErrorKind.TIMEOUT, "log find timed out",
                                                                  section=section.name, at=now))
            return dataclasses.replace(empty, file_error=known or r.outcome.value)
        out = parse_find(r.stdout, r.stderr, wd, now)
        if r.ok or out.incomplete == FIND_TIMED_OUT:
            return out
        why = FIND_TIMED_OUT if r.outcome is SectionOutcome.TIMEOUT else FIND_CUT     # cut short: a partial list
        if out.file_error == FIND_NO_OUTPUT and r.outcome is SectionOutcome.TIMEOUT:
            return dataclasses.replace(empty, incomplete=why)          # stopped before listing anything
        if out.file_error is None:
            return dataclasses.replace(out, incomplete=out.incomplete if out.incomplete != FIND_CUT else why)
        return out

    # ------------------------------------------------------------ pagers
    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None:
        caps = self._caps_for(host)
        if not ref.path or caps is None or "less" not in caps.tools:
            return None
        cmd = f"less -R +G -- {shell.quote(shell.path(ref.path))}"
        return self._transport_for(host).interactive_argv(cmd)

    def local_pager_argv(self, path: Path | None = None) -> list[str]:
        """general.pager, else $PAGER, else `less -R +G`; Windows: `less` when installed (Git for Windows, scoop),
        else `more`, and the setting is split with Windows rules (backslashes are path separators)."""
        default = ["less", "-R", "+G"]
        if platform.IS_WINDOWS and shutil.which("less") is None:
            default = ["more"]
        spec = self._pager or os.environ.get("PAGER")
        try:
            argv = platform.split_command(spec) if spec else default
        except ValueError:
            argv = default
        return [*(argv or default), str(path if path is not None else paths.log_file())]
