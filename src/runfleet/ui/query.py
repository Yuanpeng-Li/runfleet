"""[WP3] Filter language (§7.6): parse and match Job -> bool (pure).

Also the other pure job-selection rules of the Jobs view, so they are testable without Textual widgets: state
presets (§7.6), the default and column sort orders (§7.5) and the time-derived values the sort and the table share
(extrapolated elapsed, time left).

Grammar: space-separated terms, all must match. ``word`` matches as a substring of the job's host, id, name, state,
partition, nodes, reason or user, or of the name with its separators removed (``orbswp`` finds ``orb-swp--…``); no
subsequence fuzz: common words would match nearly every long sweep name. ``host:``/``h:``, ``state:``/``st:`` (codes
``R``/``PD``, groups ``active``/``failed``, names ``running``/``pending``/``oom``…), ``user:``/``u:``,
``acct:``/``account:`` and ``part:``/``p:`` take comma lists; ``su:`` and ``gpuh:`` compare the job's cost (Account,
§5.2.11) with ``> >= < <= =`` (``su:>10``, ``gpuh:>=5``); ``!term`` negates. The cost comes from a callback the screen
builds from the cached charging rules, so this module stays pure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import re

from runfleet.models import Job, JobState, StateGroup
from runfleet.util import cost as costs
from runfleet.util.cost import JobCost

TermKind = Literal["text", "host", "state", "user", "account", "partition", "su", "gpuh"]
CostFn = Callable[[Job, datetime], JobCost | None]      # util.cost.job_cost with the job's host caps (§7.5)

_KEY_ALIASES: dict[str, TermKind] = {
    "host": "host", "h": "host",
    "state": "state", "st": "state", "s": "state",
    "user": "user", "u": "user",
    "acct": "account", "account": "account",
    "part": "partition", "p": "partition", "partition": "partition",
    "su": "su", "gpuh": "gpuh",
}
_NUMERIC: frozenset[TermKind] = frozenset({"su", "gpuh"})
_COMPARE = re.compile(r"^(?P<op>>=|<=|>|<|=)?(?P<n>\d+(?:\.\d*)?|\.\d+)$")
_OPS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "=": lambda a, b: abs(a - b) < 0.05,      # (`_numeric_matches` compares `=` at the precision shown instead)
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
    values: tuple[str, ...]      # lower-cased; numeric terms: (operator,)
    negate: bool = False
    number: float = 0.0          # numeric terms (su, gpuh): the right-hand side


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
        if kind in _NUMERIC:
            m = _COMPARE.match(rest)
            if m is not None:        # a malformed number falls through to a text term
                terms.append(Term(kind, (m.group("op") or "=",), negate, float(m.group("n"))))
                continue
            kind = None
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


def su_value(job: Job, cost: JobCost | None) -> float | None:
    """The job's SU as the `su` column shows it (billing-h on usage-only clusters, via JobCost.as_billing_h): charged
    so far, a pending job's projection at its limit; None = unknown."""
    if cost is None:
        return None
    return cost.projected_su if job.state.group is StateGroup.QUEUED else cost.charged_su


def gpuh_value(job: Job, cost: JobCost | None) -> float | None:
    """The job's GPU-h so far (a pending job: 0); None = unknown."""
    return cost.gpu_h if cost is not None else None


def _numeric_matches(term: Term, job: Job, now: datetime | None, cost: CostFn | None) -> bool:
    """A job whose cost is unknown never matches (so `!su:>10` keeps it)."""
    if cost is None or now is None:
        return False
    c = cost(job, now)
    value = su_value(job, c) if term.kind == "su" else gpuh_value(job, c)
    if value is None:
        return False
    if term.values[0] == "=":           # as the column shows it (`33` is 32.5 … 33.49, `2.9`): `su:=0` = free jobs
        shown = costs.fmt_su if term.kind == "su" else costs.fmt_hours
        return shown(value) == shown(term.number)
    return _OPS[term.values[0]](value, term.number)


def _term_matches(term: Term, job: Job, now: datetime | None, cost: CostFn | None) -> bool:
    if term.kind == "host":
        return job.key.host.lower() in term.values
    if term.kind == "state":
        return _state_matches(term.values, job)
    if term.kind == "user":
        return any((job.is_mine if v == "me" else job.user.lower() == v) for v in term.values)
    if term.kind == "account":
        return (job.account or "").lower() in term.values
    if term.kind == "partition":       # a pending job's `gpu,free-gpu` list matches any of its partitions
        return any(p.strip().lower() in term.values for p in (job.partition or "").split(","))
    if term.kind in _NUMERIC:
        return _numeric_matches(term, job, now, cost)
    return _fuzzy(term.values[0], _fields(job), job.name)


def matches(query: Query, job: Job, now: datetime | None = None, cost: CostFn | None = None) -> bool:
    """True when every term matches (negated terms must not match). ``su:`` / ``gpuh:`` need ``now`` and ``cost``;
    without them the job's cost is unknown."""
    return all(_term_matches(t, job, now, cost) != t.negate for t in query.terms)


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
SortColumn = Literal["host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "submit", "user", "su",
                     "gpuh"]
# su / gpuh (Account) are not in the `o` cycle: a header click or the palette's Sort by SU / Sort by GPU-h
SORT_CYCLE: tuple[SortColumn, ...] = ("host", "state", "elapsed", "gpu_util", "gpus", "name", "id", "submit")
SORT_LABELS: dict[SortColumn, str] = {
    "host": "host", "id": "id", "name": "name", "state": "state", "elapsed": "elapsed", "left": "left",
    "gpus": "gpus", "gpu_util": "GPU%", "submit": "submit", "user": "user", "su": "SU", "gpuh": "GPU-h",
}
COST_SORTS: tuple[SortColumn, ...] = ("su", "gpuh")

# Default state order (§7.5): active, queued, failed, ended, cancelled, done.
GROUP_ORDER: dict[StateGroup, int] = {
    StateGroup.ACTIVE: 0, StateGroup.QUEUED: 1, StateGroup.FAILED: 2, StateGroup.ENDED: 3,
    StateGroup.CANCELLED: 4, StateGroup.DONE: 5, StateGroup.UNKNOWN: 6,
}


# numeric columns read best largest-first by default (busiest GPU, longest running, newest submit)
DESC_DEFAULT: frozenset[str] = frozenset({"elapsed", "gpus", "gpu_util", "submit", "su", "gpuh"})


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
              now: datetime, cost: CostFn | None = None) -> list[Job]:
    """Sort by ``column`` then by the default order (host, state group, newest submit first), stably.

    Unknown values (``None``) always sort last, whatever the direction; without ``cost`` every SU / GPU-h is unknown.
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
    elif column in COST_SORTS:
        pick = su_value if column == "su" else gpuh_value
        value = lambda j: pick(j, cost(j, now)) if cost is not None else None  # noqa: E731
    else:   # submit
        value = lambda j: _submit_ts(j) or None  # noqa: E731

    values = [(j, value(j)) for j in ordered]       # once per job: a cost is not free to compute
    known = [(j, v) for j, v in values if v is not None]
    unknown = [j for j, v in values if v is None]
    known.sort(key=lambda jv: jv[1], reverse=sort_descending(column, reverse))
    return [j for j, _v in known] + unknown
