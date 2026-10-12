"""[WP3] Filter language (§7.6): parse and match Job -> bool (pure).

Also the other pure job-selection rules of the Jobs view, so they are testable without Textual widgets: state
presets (§7.6), the default and column sort orders (§7.5) and the time-derived values the sort and the table share
(extrapolated elapsed, time left).

Grammar: space-separated terms, all must match. ``word`` matches as a substring of the job's host, id, name, state,
partition, nodes, reason or user, or of the name with its separators removed (``orbswp`` finds ``orb-swp--…``); no
subsequence fuzz: common words would match nearly every long sweep name. ``host:``/``h:``, ``state:``/``st:`` (codes
``R``/``PD``, groups ``active``/``failed``, names ``running``/``pending``/``oom``…) and ``user:``/``u:`` take comma lists;
``!term`` negates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import re

from runfleet.models import Job, JobState, StateGroup

TermKind = Literal["text", "host", "state", "user"]

_KEY_ALIASES: dict[str, TermKind] = {
    "host": "host", "h": "host",
    "state": "state", "st": "state", "s": "state",
    "user": "user", "u": "user",
}
_GROUP_NAMES = {g.value: g for g in StateGroup}
_STATE_CODES = {s.value.upper(): s for s in JobState}
_STATE_NAMES = {s.name.lower(): s for s in JobState} | {
    "oom": JobState.OUT_OF_MEMORY, "canceled": JobState.CANCELLED, "done": JobState.COMPLETED,
    "timeouts": JobState.TIMEOUT, "preempt": JobState.PREEMPTED, "node_fail": JobState.NODE_FAIL,
}
_SEPARATORS = re.compile(r"[^0-9a-z]+")


@dataclass(frozen=True, slots=True)
class Term:
    kind: TermKind
    values: tuple[str, ...]      # lower-cased
    negate: bool = False


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    terms: tuple[Term, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.terms


def parse(text: str) -> Query:
    """Parse filter text. Never raises: an unknown ``key:value`` is matched as plain text."""
    terms: list[Term] = []
    for raw in text.split():
        negate = raw.startswith("!")
        body = raw[1:] if negate else raw
        if not body:
            continue
        key, sep, rest = body.partition(":")
        kind = _KEY_ALIASES.get(key.lower()) if sep else None
        if kind is not None and rest:
            values = tuple(v.lower() for v in rest.split(",") if v)
            terms.append(Term(kind, values, negate))
        else:
            terms.append(Term("text", (body.lower(),), negate))
    return Query(text, tuple(terms))


def _fields(job: Job) -> tuple[str, ...]:
    return (
        job.key.host, job.key.label, job.key.id, job.name, job.state.value, job.state.name, job.partition or "",
        " ".join(job.nodes) or (job.nodelist or ""), job.reason or "", job.user,
    )


def _fuzzy(needle: str, haystacks: Iterable[str], name: str = "") -> bool:
    """Substring of any field; else the needle and the job name compared without separators."""
    fields = [h.lower() for h in haystacks if h]
    if any(needle in f for f in fields):
        return True
    compact = _SEPARATORS.sub("", needle)
    return len(compact) >= 3 and compact in _SEPARATORS.sub("", name.lower())


def _state_matches(values: Sequence[str], job: Job) -> bool:
    for v in values:
        group = _GROUP_NAMES.get(v)
        if group is not None and job.state.group is group:
            return True
        code = v.upper()
        if code.endswith("H") and code[:-1] in _STATE_CODES and job.held and job.state.value == code[:-1]:
            return True
        if _STATE_CODES.get(code) is job.state or _STATE_NAMES.get(v) is job.state:
            return True
    return False


def _term_matches(term: Term, job: Job) -> bool:
    if term.kind == "host":
        return job.key.host.lower() in term.values
    if term.kind == "state":
        return _state_matches(term.values, job)
    if term.kind == "user":
        return any((job.is_mine if v == "me" else job.user.lower() == v) for v in term.values)
    return _fuzzy(term.values[0], _fields(job), job.name)


def matches(query: Query, job: Job) -> bool:
    """True when every term matches (negated terms must not match)."""
    return all(_term_matches(t, job) != t.negate for t in query.terms)


# ---------------------------------------------------------------- state presets
Preset = Literal["fresh", "active", "queued", "failed", "all"]
PRESETS: tuple[Preset, ...] = ("fresh", "active", "queued", "failed", "all")
PRESET_LABELS: dict[Preset, str] = {
    "fresh": "fresh (active + ended < {window})",
    "active": "active",
    "queued": "queued",
    "failed": "failed (failed + ended)",
    "all": "all",
}


def next_preset(current: Preset) -> Preset:
    return PRESETS[(PRESETS.index(current) + 1) % len(PRESETS)]


def ended_at(job: Job) -> datetime | None:
    """When a terminal job ended (falls back to the observation time when Slurm gave no end)."""
    return job.end_time or (job.observed_at if job.state.is_terminal else None)


def preset_matches(preset: Preset, job: Job, now: datetime, recent_window: float) -> bool:
    group = job.state.group
    if preset == "all":
        return True
    if preset == "active":
        return group is StateGroup.ACTIVE
    if preset == "queued":
        return group is StateGroup.QUEUED
    if preset == "failed":
        return group in (StateGroup.FAILED, StateGroup.ENDED)
    # fresh: everything not terminal, plus terminal jobs that ended inside the window
    if not job.state.is_terminal:
        return True
    end = ended_at(job)
    return end is None or (now - end).total_seconds() <= recent_window


# ---------------------------------------------------------------- derived time values
def elapsed_s(job: Job, now: datetime) -> float | None:
    """Elapsed seconds, extrapolated from ``observed_at`` for ACTIVE jobs; ``None`` for queued jobs."""
    if job.state.group is StateGroup.QUEUED:
        return None
    if job.elapsed is None:
        if job.start_time is not None and job.state.group is StateGroup.ACTIVE and not job.est_start:
            return max(0.0, (now - job.start_time).total_seconds())
        return None
    base = job.elapsed.total_seconds()
    if job.state.group is StateGroup.ACTIVE and job.observed_at is not None:
        base += max(0.0, (now - job.observed_at).total_seconds())
    return base


def left_s(job: Job, now: datetime) -> float | None:
    """Seconds left before the time limit (ACTIVE); the limit itself for queued jobs; ``None`` otherwise."""
    if job.time_limit is None:
        return None
    limit = job.time_limit.total_seconds()
    group = job.state.group
    if group is StateGroup.QUEUED:
        return limit
    if group is StateGroup.ACTIVE:
        el = elapsed_s(job, now)
        return None if el is None else max(0.0, limit - el)
    return None


# ---------------------------------------------------------------- sorting
SortColumn = Literal["host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "submit", "user"]
SORT_CYCLE: tuple[SortColumn, ...] = ("host", "state", "elapsed", "gpu_util", "gpus", "name", "id", "submit")
SORT_LABELS: dict[SortColumn, str] = {
    "host": "host", "id": "id", "name": "name", "state": "state", "elapsed": "elapsed", "left": "left",
    "gpus": "gpus", "gpu_util": "GPU%", "submit": "submit", "user": "user",
}

# Default state order (§7.5): active, queued, failed, ended, cancelled, done.
GROUP_ORDER: dict[StateGroup, int] = {
    StateGroup.ACTIVE: 0, StateGroup.QUEUED: 1, StateGroup.FAILED: 2, StateGroup.ENDED: 3,
    StateGroup.CANCELLED: 4, StateGroup.DONE: 5, StateGroup.UNKNOWN: 6,
}


# numeric columns read best largest-first by default (busiest GPU, longest running, newest submit)
DESC_DEFAULT: frozenset[str] = frozenset({"elapsed", "gpus", "gpu_util", "submit"})


def sort_descending(column: str, reverse: bool) -> bool:
    """The effective direction of a sort (the status line and the column header share it)."""
    return (column in DESC_DEFAULT) != reverse


def sort_arrow(column: str, reverse: bool) -> str:
    return "▼" if sort_descending(column, reverse) else "▲"


def next_sort(current: SortColumn) -> SortColumn:
    if current not in SORT_CYCLE:
        return SORT_CYCLE[0]
    return SORT_CYCLE[(SORT_CYCLE.index(current) + 1) % len(SORT_CYCLE)]


def _id_key(job: Job) -> tuple[int, str]:
    digits = "".join(ch for ch in job.job_id_raw if ch.isdigit())
    return (int(digits) if digits and len(digits) < 19 else 0, job.key.id)


def _submit_ts(job: Job) -> float:
    t = job.submit_time or job.start_time
    return t.timestamp() if t is not None else 0.0


def sort_jobs(jobs: Iterable[Job], column: SortColumn, reverse: bool, *, host_order: Sequence[str],
              now: datetime) -> list[Job]:
    """Sort by ``column`` then by the default order (host, state group, newest submit first), stably.

    Unknown values (``None``) always sort last, whatever the direction.
    """
    rank = {h: i for i, h in enumerate(host_order)}

    def default_key(j: Job) -> tuple[int, int, float, tuple[int, str]]:
        return (rank.get(j.key.host, len(rank)), GROUP_ORDER[j.state.group], -_submit_ts(j), _id_key(j))

    ordered = sorted(jobs, key=default_key)
    if column == "host":
        return list(reversed(ordered)) if reverse else ordered

    value: Callable[[Job], float | str | tuple[int, str] | None]
    if column == "id":
        value = _id_key
    elif column == "name":
        value = lambda j: j.name.lower()  # noqa: E731
    elif column == "user":
        value = lambda j: j.user.lower()  # noqa: E731
    elif column == "state":
        value = lambda j: float(GROUP_ORDER[j.state.group])  # noqa: E731
    elif column == "elapsed":
        value = lambda j: elapsed_s(j, now)  # noqa: E731
    elif column == "left":
        value = lambda j: left_s(j, now)  # noqa: E731
    elif column == "gpus":
        value = lambda j: float(j.alloc.gpus) if j.alloc.gpus else None  # noqa: E731
    elif column == "gpu_util":
        value = lambda j: j.gpu.util_mean_pct if j.gpu is not None else None  # noqa: E731
    else:   # submit
        value = lambda j: _submit_ts(j) or None  # noqa: E731

    known = [j for j in ordered if value(j) is not None]
    unknown = [j for j in ordered if value(j) is None]
    known.sort(key=lambda j: value(j), reverse=sort_descending(column, reverse))  # type: ignore[arg-type,return-value]
    return known + unknown
