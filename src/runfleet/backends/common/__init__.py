"""[WP2] Monitor collectors shared by schedulers: nvidia, procfs, GPU type table.

This package module also holds the small pieces every collector shares (§5 general rules): `ParseResult`, section
outcome helpers, `stat -c '%n|%s|%Y'` parsing for log files and the stall hints of §5.2.6.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Generic, TypeVar

from runfleet.models import ErrorKind, HostError, LogRef
from runfleet.probe.section import SectionOutcome, SectionResult
from runfleet.util.slurmfmt import split_rows

T = TypeVar("T")

# Outcomes that must never clear previously collected data (§5): the store keeps the last good value.
STALE_OUTCOMES = frozenset({SectionOutcome.STDERR_ONLY, SectionOutcome.TIMEOUT, SectionOutcome.INCOMPLETE,
                            SectionOutcome.NOT_RUN, SectionOutcome.MISSING})


@dataclass(slots=True)
class ParseResult(Generic[T]):
    """Rows that parsed, rows that did not (`(line number, line, why)`) and the ids of bad rows whose id was readable.

    Collectors never treat a bad id as vanished/ended (§5)."""
    items: list[T] = field(default_factory=list)
    bad_lines: list[tuple[int, str, str]] = field(default_factory=list)
    bad_ids: set[str] = field(default_factory=set)

    def bad(self, lineno: int, line: str, why: str, row_id: str | None = None) -> None:
        self.bad_lines.append((lineno, line, why))
        if row_id:
            self.bad_ids.add(row_id)

    @property
    def mostly_bad(self) -> bool:
        """More than 50 % of the rows failed validation → the section is a PARSE error (§5)."""
        total = len(self.items) + len(self.bad_lines)
        return total > 0 and len(self.bad_lines) * 2 > total


def usable(result: SectionResult | None) -> bool:
    """Parse this result? OK/EMPTY always; FAILED only when it printed something (multi-id partial output, §5)."""
    if result is None:
        return False
    return result.ok or (result.outcome is SectionOutcome.FAILED and result.stdout != "")


def complete(result: SectionResult | None) -> bool:
    """A full, authoritative answer (OK/EMPTY): the only case that may drop items absent from the output."""
    return result is not None and result.ok


def _first_line(text: str) -> str:
    for line in split_rows(text):
        if line.strip():
            return line.strip()
    return ""


def section_error(result: SectionResult, at: datetime | None, *, summary: str | None = None,
                  kind: ErrorKind | None = None) -> HostError | None:
    """HostError for a failing section (None when the outcome is OK/EMPTY)."""
    if result.ok:
        return None
    if kind is None:
        kind = {SectionOutcome.TIMEOUT: ErrorKind.TIMEOUT, SectionOutcome.MISSING: ErrorKind.MISSING}.get(
            result.outcome, ErrorKind.REMOTE)
    if summary is None:
        detail_line = _first_line(result.stderr)
        if result.outcome is SectionOutcome.MISSING:
            tool = result.stderr.partition("missing:")[2].strip() or detail_line
            summary = f"{result.name}: {tool or 'tool'} not installed"
        elif result.outcome is SectionOutcome.TIMEOUT:
            summary = f"{result.name} timed out"
        elif result.outcome in (SectionOutcome.INCOMPLETE, SectionOutcome.NOT_RUN):
            summary = f"{result.name} did not finish ({result.outcome.value})"
        else:
            summary = f"{result.name}: {detail_line or f'rc {result.rc}'}"
    return HostError(kind=kind, summary=summary[:80], detail=result.stderr[:4096], section=result.name, at=at)


def parse_error(name: str, pr: ParseResult[T], at: datetime | None, what: str | None = None) -> HostError:
    lineno, line, why = pr.bad_lines[0] if pr.bad_lines else (0, "", "")
    return HostError(kind=ErrorKind.PARSE, summary=f"could not parse {what or name} output (see log)"[:80],
                     detail=f"line {lineno}: {why}: {line[:200]}", section=name, at=at)


# ---------------------------------------------------------------- stat -c '%n|%s|%Y' (log size + mtime)
@dataclass(frozen=True, slots=True)
class StatInfo:
    size: int | None
    mtime: datetime | None
    error: str | None = None


_STAT_ERR = re.compile(r"^stat: cannot stat(?:x)? '(?P<path>.*)': (?P<why>.+)$")


def parse_stat_lines(text: str) -> dict[str, StatInfo]:
    """`/p/x.out|663|1788763833` per path (rsplit: paths may contain `|`); `stat: cannot stat 'p': No such file or
    directory` (stderr folded in with `2>&1`) → error entry."""
    out: dict[str, StatInfo] = {}
    for line in split_rows(text):
        m = _STAT_ERR.match(line)
        if m:
            out[m.group("path")] = StatInfo(None, None, m.group("why").strip())
            continue
        parts = line.rsplit("|", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        out[parts[0]] = StatInfo(int(parts[1]), datetime.fromtimestamp(int(parts[2]), tz=UTC))
    return out


def with_stat(ref: LogRef, stats: dict[str, StatInfo]) -> LogRef:
    """LogRef updated with fresh size/mtime/error when its path was stat'ed."""
    if ref.path is None or ref.path not in stats:
        return ref
    st = stats[ref.path]
    return dataclasses.replace(ref, size=st.size, mtime=st.mtime, error=st.error)


# ---------------------------------------------------------------- stall hints (§5.2.6)
GPU_IDLE_PCT = 5.0
GPU_IDLE_MIN_S = 15 * 60


def fmt_age(seconds: float) -> str:
    """Compact age for hints: `47s`, `12m`, `3h05m`, `2d4h`."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


def log_idle_hint(mtime: datetime | None, now: datetime, stall_after: float) -> str | None:
    if mtime is None:
        return None
    age = (now - mtime).total_seconds()
    return f"log idle {fmt_age(age)}" if age > stall_after else None


@dataclass(slots=True)
class _IdleRun:
    last_at: datetime
    since: datetime | None = None    # first sample of the current run of idle samples
    samples: int = 0                 # samples in that run


@dataclass(slots=True)
class GpuIdleTracker:
    """`GPU idle {dur}` once every utilisation sample over >= 15 min (>= 2 samples) was below 5 % (§5.2.6)."""
    _runs: dict[str, _IdleRun] = field(default_factory=dict)

    def add(self, key: str, at: datetime, util: float | None) -> None:
        if util is None:
            return
        run = self._runs.get(key)
        if run is not None and at <= run.last_at:
            return                                   # same sample seen again (no new GPU data this tick)
        if run is None:
            run = self._runs[key] = _IdleRun(last_at=at)
        if (at - run.last_at).total_seconds() > GPU_IDLE_MIN_S:
            run.since, run.samples = None, 0         # no telemetry for that long: the gap is not known to be idle
        run.last_at = at
        if util < GPU_IDLE_PCT:
            if run.since is None:
                run.since, run.samples = at, 0
            run.samples += 1
        else:
            run.since, run.samples = None, 0

    def hint(self, key: str, now: datetime) -> str | None:
        run = self._runs.get(key)
        if run is None or run.since is None or run.samples < 2:
            return None
        dur = (now - run.since).total_seconds()
        return f"GPU idle {fmt_age(dur)}" if dur >= GPU_IDLE_MIN_S else None

    def forget(self, keep: Iterable[str]) -> None:
        live = set(keep)
        for k in [k for k in self._runs if k not in live]:
            del self._runs[k]
