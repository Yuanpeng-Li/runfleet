"""[WP2] ps output parsers.

RunFleet's own commands use headerless `ps -o col=,…` lists (§5.3.1):

- `process.ps`: `date +%s` on the first line, then `pid ppid pgid etimes pcpu pmem rss stat args`;
- `nvidia.apps` (after `@@ps`): `pid ppid user etimes pcpu pmem rss stat args`;
- `process.fds`: `pid␟cwd␟fd1␟fd2` (\x1f-separated; old `|` rows still parse) from readlink.

`args` is everything after the fixed columns (it may contain any whitespace).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from runfleet.backends.common import ParseResult
from runfleet.models import Proc
from runfleet.util.slurmfmt import split_rows

OWN_COLUMNS = ("pid", "ppid", "pgid", "etimes", "pcpu", "pmem", "rss", "stat", "args")
APPS_COLUMNS = ("pid", "ppid", "user", "etimes", "pcpu", "pmem", "rss", "stat", "args")


def _pct(value: str) -> float:
    """ps %cpu/%mem; a non-finite value reads as 0 (keeps the process, keeps nan out of `snapshot --json`)."""
    f = float(value)
    return f if math.isfinite(f) else 0.0


def parse_ps_rows(lines: Sequence[str], columns: Sequence[str], *, user: str = "") -> ParseResult[Proc]:
    """Headerless ps rows with the given columns (`args` last) → Procs. `user` fills a missing user column."""
    pr: ParseResult[Proc] = ParseResult()
    fixed = len(columns) - 1
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split(None, fixed)
        if len(parts) < fixed:
            pr.bad(n, line, f"expected {fixed}+ columns")
            continue
        row = dict(zip(columns, parts + [""] * (len(columns) - len(parts)), strict=True))
        try:
            pr.items.append(Proc(
                pid=int(row["pid"]), ppid=int(row["ppid"]),
                pgid=int(row["pgid"]) if "pgid" in row else None,
                user=row.get("user", user) or user, etime_s=int(row["etimes"]),
                cpu_pct=_pct(row["pcpu"]), mem_pct=_pct(row["pmem"]),
                rss_bytes=int(row.get("rss", "0") or 0) * 1024, stat=row["stat"], args=row["args"].strip(),
            ))
        except (KeyError, ValueError) as e:
            first = parts[0] if parts and parts[0].isdigit() else None
            pr.bad(n, line, f"bad value: {e}", first)
    return pr


def parse_ps(text: str, *, user: str) -> tuple[int | None, ParseResult[Proc]]:
    """`process.ps` output → (remote epoch, own processes)."""
    lines = split_rows(text)
    epoch: int | None = None
    if lines and lines[0].strip().isdigit():
        epoch = int(lines[0].strip())
        lines = lines[1:]
    return epoch, parse_ps_rows(lines, OWN_COLUMNS, user=user)


@dataclass(frozen=True, slots=True)
class FdInfo:
    cwd: str | None
    stdout: str | None           # readlink /proc/<pid>/fd/1: a path, "/dev/pts/3", "pipe:[123]", ...
    stderr: str | None


def parse_fds(text: str) -> dict[int, FdInfo]:
    out: dict[int, FdInfo] = {}
    for line in split_rows(text):
        parts = line.split("\x1f") if "\x1f" in line else line.split("|")
        if len(parts) != 4 or not parts[0].strip().isdigit():
            continue
        out[int(parts[0])] = FdInfo(parts[1] or None, parts[2] or None, parts[3] or None)
    return out


def is_regular_log(target: str | None) -> bool:
    """A file path worth tailing (not a tty, pipe, socket, /dev/null or a deleted file)."""
    if not target or not target.startswith("/"):
        return False
    return not (target.startswith("/dev/") or target.endswith(" (deleted)"))
