"""[WP4] `runfleet status`: a cache-only fleet summary for shell prompts and tmux status bars (DESIGN.md §8.5).

Never runs ssh and never writes: the cache is opened read-only (`file:…?mode=ro&immutable=0`; sqlite may add the
empty `-wal`/`-shm` companions of a WAL database itself, 0600 like the file) without `core.cache.Cache`, whose
constructor creates, migrates and repairs the database. A missing, locked, damaged or foreign file is "no data".

What it reads: `snapshots` (each host's last merged snapshot: my jobs by state, `taken_at` = data age) and
`host_status` (connection state since a time, written by FleetService on every change; absent in caches of RunFleet
≤ 0.5.0, then the state is unknown). Counts follow the header: R = my active jobs, PD = my queued jobs, F = my
failed or ended jobs whose end lies inside `general.recent_window`.

`--short`: `R4 PD2 F1 · lynx:auth · 3m` (`… · 14m stale` once the newest data is older than STALE_AFTER).
`--json`: schema "runfleet.status/1", see `document()`. Exit 0 a summary, 1 no data (`no data` / `"error"`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runfleet.models import ConnState, JobState, StateGroup
from runfleet.util import humanize

SCHEMA = "runfleet.status/1"
STALE_AFTER = 600.0                  # seconds: past this the newest data is marked "stale"
DEFAULT_WINDOW = 86400.0             # general.recent_window when there is no readable config
EXIT_OK, EXIT_NO_DATA = 0, 1
BUSY_TIMEOUT = 0.5                   # seconds sqlite waits for a lock before "cache busy"
_PROBLEM = {ConnState.NEEDS_AUTH: "auth", ConnState.HOSTKEY: "hostkey", ConnState.UNREACHABLE: "down"}
STATE_WIDTH = 40


class NoData(Exception):
    """Nothing to summarise; the message says why ("no cache yet", "cache busy", …)."""


@dataclass(frozen=True, slots=True)
class CachedRows:
    """Raw rows of one read: host -> (taken_at, snapshot json) and host -> (at, state, error kind, error)."""
    snapshots: Mapping[str, tuple[float, str]]
    status: Mapping[str, tuple[float, str, str | None, str | None]]


@dataclass(frozen=True, slots=True)
class HostLine:
    name: str
    state: ConnState | None          # None: never recorded (older cache, or no answer yet)
    error: str | None
    error_kind: str | None
    state_since: datetime | None
    data_at: datetime | None         # snapshot taken_at; None = no data for this host
    running: int = 0
    pending: int = 0
    failed: int = 0

    @property
    def problem(self) -> str | None:
        """`auth` / `hostkey` / `down` / `dns` for a host that needs you (the header's red chips), else None."""
        if self.state is ConnState.UNREACHABLE and self.error_kind == "dns":
            return "dns"
        return _PROBLEM.get(self.state) if self.state is not None else None


@dataclass(frozen=True, slots=True)
class Report:
    generated_at: datetime
    hosts: tuple[HostLine, ...]
    window: float = DEFAULT_WINDOW
    error: str | None = None         # set: no data (exit 1)

    @property
    def data_at(self) -> datetime | None:
        """Time of the newest data: the newest snapshot, else the newest recorded state."""
        snaps = [h.data_at for h in self.hosts if h.data_at is not None]
        if snaps:
            return max(snaps)
        states = [h.state_since for h in self.hosts if h.state_since is not None]
        return max(states) if states else None

    @property
    def age_s(self) -> float | None:
        at = self.data_at
        return None if at is None else max(0.0, (self.generated_at - at).total_seconds())

    @property
    def stale(self) -> bool:
        age = self.age_s
        return age is None or age > STALE_AFTER

    def totals(self) -> tuple[int, int, int]:
        return (sum(h.running for h in self.hosts), sum(h.pending for h in self.hosts),
                sum(h.failed for h in self.hosts))

    def problems(self) -> list[tuple[str, str]]:
        return [(h.name, p) for h in self.hosts if (p := h.problem) is not None]


# ====================================================================================================== reading
def read_cache(path: Path) -> CachedRows:
    """Both tables in one read transaction; raises NoData for a missing, locked, damaged or foreign file."""
    if not path.is_file():
        raise NoData("no cache yet (start runfleet once)")
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=0"
    try:
        db = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT)
    except sqlite3.Error as exc:
        raise NoData(f"cache unreadable ({exc})") from None
    try:
        db.execute("BEGIN")                     # both tables from the same moment (WAL: never blocks the writer)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "snapshots" not in tables:
            raise NoData("cache has no snapshots yet")
        snaps = {str(h): (float(t), str(j)) for h, t, j in
                 db.execute("SELECT host, taken_at, json FROM snapshots") if t is not None and j is not None}
        status: dict[str, tuple[float, str, str | None, str | None]] = {}
        if "host_status" in tables:             # caches written by RunFleet <= 0.5.0 lack it
            for h, at, state, kind, err in db.execute("SELECT host, at, state, error_kind, error FROM host_status"):
                if at is not None and state is not None:
                    status[str(h)] = (float(at), str(state), kind, err)
        return CachedRows(snaps, status)
    except sqlite3.OperationalError as exc:
        text = str(exc)
        raise NoData("cache busy" if "locked" in text or "busy" in text else f"cache unreadable ({text})") from None
    except (sqlite3.Error, ValueError, TypeError) as exc:      # "file is not a database", odd column types
        raise NoData(f"cache unreadable ({exc})") from None
    finally:
        db.close()


def _when(text: Any) -> datetime | None:
    if not isinstance(text, str) or not text:
        return None
    at = datetime.fromisoformat(text)
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def count_jobs(snapshot_json: str, now: datetime, window: float) -> tuple[int, int, int]:
    """(running, pending, failed) of MY jobs in one cached snapshot, like the header totals; a job row that does not
    decode is skipped."""
    try:
        jobs = json.loads(snapshot_json).get("jobs") or {}
    except (ValueError, AttributeError):
        return 0, 0, 0
    running = pending = failed = 0
    for job in (jobs.values() if isinstance(jobs, dict) else ()):
        try:
            if not job.get("is_mine", True):
                continue
            state = JobState(job["state"])
            group = state.group
            if group is StateGroup.ACTIVE:
                running += 1
            elif group is StateGroup.QUEUED:
                pending += 1
            elif group in (StateGroup.FAILED, StateGroup.ENDED):
                end = _when(job.get("end_time")) or _when(job.get("observed_at"))
                if end is None or (now - end).total_seconds() <= window:
                    failed += 1
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    return running, pending, failed


def select_hosts(cached: Iterable[str], configured: Sequence[str] | None, known: Iterable[str],
                 wanted: Sequence[str]) -> list[str]:
    """Hosts to summarise: `wanted` (each must be configured or cached), else the enabled `configured` hosts in
    config order (none enabled: none, never the cached hosts the config disabled), else (`configured` None: no readable
    config) every host in the cache by name. Raises ValueError for an unknown host."""
    cached_set = set(cached)
    if wanted:
        allowed = cached_set | set(known) | set(configured or ())
        unknown = [h for h in wanted if h not in allowed]
        if unknown:
            raise ValueError(f"unknown host {unknown[0]!r} (not configured, nothing cached)")
        return list(dict.fromkeys(wanted))
    if configured is not None:
        return list(configured)
    return sorted(cached_set)


def build_report(rows: CachedRows, hosts: Sequence[str], now: datetime, window: float = DEFAULT_WINDOW) -> Report:
    lines: list[HostLine] = []
    for name in hosts:
        snap = rows.snapshots.get(name)
        st = rows.status.get(name)
        try:
            state = ConnState(st[1]) if st is not None else None
        except ValueError:
            state = None
        counts = count_jobs(snap[1], now, window) if snap is not None else (0, 0, 0)
        lines.append(HostLine(
            name=name, state=state, error=st[3] if st else None, error_kind=st[2] if st else None,
            state_since=datetime.fromtimestamp(st[0], tz=UTC) if st else None,
            data_at=datetime.fromtimestamp(snap[0], tz=UTC) if snap else None,
            running=counts[0], pending=counts[1], failed=counts[2]))
    if not lines:
        return Report(now, (), window, error="no enabled hosts in the config")
    if not any(h.data_at is not None or h.state is not None for h in lines):
        return Report(now, tuple(lines), window, error="no data for these hosts yet")
    return Report(now, tuple(lines), window)


def no_data(now: datetime, reason: str, window: float = DEFAULT_WINDOW) -> Report:
    return Report(now, (), window, error=reason)


# ====================================================================================================== output
def _age_text(age: float | None) -> str:
    if age is None:
        return "—"
    text = humanize.age(age)
    return f"{text} stale" if age > STALE_AFTER else text


def short_line(report: Report) -> str:
    """`R4 PD2 F1 · lynx:auth · 3m` (my jobs by state across the hosts · hosts that need you · newest data)."""
    if report.error is not None:
        return "no data"
    r, pd, f = report.totals()
    parts = [f"R{r} PD{pd} F{f}"]
    problems = report.problems()
    if problems:
        parts.append(" ".join(f"{h}:{p}" for h, p in problems))
    parts.append(_age_text(report.age_s))
    return " · ".join(parts)


def table_rows(report: Report) -> list[tuple[str, ...]]:
    """Per-host table: HOST, STATE (with the error), R, PD, F, AGE (of that host's data), then a total row."""
    rows: list[tuple[str, ...]] = [("HOST", "STATE", "R", "PD", "F", "AGE")]
    for h in report.hosts:
        state = h.state.value if h.state is not None else "—"
        if h.error and h.state not in (None, ConnState.OK):
            state += f" ({h.error})"
        if len(state) > STATE_WIDTH:
            state = state[: STATE_WIDTH - 1] + "…"
        if h.data_at is None:
            rows.append((h.name, state, "—", "—", "—", "no data"))
            continue
        age = (report.generated_at - h.data_at).total_seconds()
        rows.append((h.name, state, str(h.running), str(h.pending), str(h.failed), _age_text(age)))
    r, pd, f = report.totals()
    rows.append(("total", "", str(r), str(pd), str(f), _age_text(report.age_s)))
    return rows


def _iso(at: datetime | None) -> str | None:
    return at.astimezone(UTC).isoformat() if at is not None else None


def document(report: Report) -> dict[str, Any]:
    """The `--json` document (schema "runfleet.status/1"); the same keys with and without data."""
    r, pd, f = report.totals()
    hosts = []
    for h in report.hosts:
        age = None if h.data_at is None else max(0.0, (report.generated_at - h.data_at).total_seconds())
        hosts.append({
            "name": h.name, "state": h.state.value if h.state is not None else None,
            "state_since": _iso(h.state_since), "error": h.error, "problem": h.problem,
            "data_at": _iso(h.data_at), "age_s": None if age is None else round(age, 1),
            "stale": age is None or age > STALE_AFTER,
            "running": h.running, "pending": h.pending, "failed": h.failed,
        })
    age = report.age_s
    return {
        "schema": SCHEMA, "generated_at": _iso(report.generated_at), "error": report.error,
        "data_at": _iso(report.data_at), "age_s": None if age is None else round(age, 1), "stale": report.stale,
        "stale_after_s": STALE_AFTER, "recent_window_s": report.window,
        "totals": {"running": r, "pending": pd, "failed": f},
        "problems": [{"host": h, "problem": p} for h, p in report.problems()],
        "hosts": hosts,
    }


def dumps(report: Report) -> str:
    return json.dumps(document(report), indent=2, ensure_ascii=False)
