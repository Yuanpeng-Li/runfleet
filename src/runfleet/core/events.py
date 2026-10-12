"""Job transitions worth telling the user about, derived from consecutive store snapshots (pure, §7.14).

`TransitionDetector` keeps the last observed jobs of each host and turns the next observation into `JobEvent`s:
started (queued → running, or a new task / job first seen running), ended (into a terminal state), restarted (the
restart count went up or the job went back to the queue), time_low (running with less than max(10 % of the limit,
10 min) left; once per run) and gpu_idle (the collector's `GPU idle …` hint first raised; once per run). Only the
user's own jobs count. The first observation of a host is a baseline and yields nothing, so a start, a reconnect or
a warm cache never floods; jobs that first appear already ended (sacct history arriving a tick late) only count when
they ended after the baseline.

`coalesce` folds one batch into notices (a burst on one host becomes one summary line), `wanted` applies the
`[notify]` event selection, `ended_since` and `count_line` back the "since you were away" digest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from runfleet.config import NotifySettings
from runfleet.models import Job, JobKey, JobKind, JobState, StateGroup
from runfleet.util import humanize
from runfleet.util.logscan import PENDING_EXIT_HINT

EventKind = Literal["started", "ended", "restarted", "time_low", "gpu_idle"]
Category = Literal["started", "ended_bad", "ended_ok", "ended_other", "restarted", "time_low", "gpu_idle"]
CATEGORIES: tuple[Category, ...] = ("started", "ended_bad", "ended_ok", "ended_other", "restarted", "time_low",
                                    "gpu_idle")
BAD_END = frozenset({JobState.FAILED, JobState.TIMEOUT, JobState.OUT_OF_MEMORY, JobState.NODE_FAIL,
                     JobState.PREEMPTED, JobState.BOOT_FAIL, JobState.DEADLINE})
TIME_LOW_FRACTION = 0.10
TIME_LOW_MIN_S = 600.0
BASELINE_SLACK_S = 120.0     # a job started / ended this long before the baseline still counts as news after it
BURST = 3                    # more events than this on one host in one batch -> one summary notice
GPU_IDLE_PREFIX = "GPU idle"

# how an end state reads in counts ("3 failed · 12 completed · 1 timeout")
_END_WORDS: dict[JobState, str] = {
    JobState.COMPLETED: "completed", JobState.FAILED: "failed", JobState.TIMEOUT: "timeout",
    JobState.OUT_OF_MEMORY: "out of memory", JobState.NODE_FAIL: "node failure", JobState.PREEMPTED: "preempted",
    JobState.BOOT_FAIL: "boot failure", JobState.DEADLINE: "deadline", JobState.CANCELLED: "cancelled",
    JobState.EXITED: "ended",
}
_KIND_WORDS: dict[str, str] = {"started": "started", "restarted": "restarted", "time_low": "low on time",
                               "gpu_idle": "GPU idle"}
# order of words in a count line: the bad news first
_WORD_ORDER = ("failed", "out of memory", "timeout", "node failure", "preempted", "boot failure", "deadline",
               "restarted", "low on time", "GPU idle", "cancelled", "ended", "completed", "started")


def end_word(state: JobState) -> str:
    return _END_WORDS.get(state, state.name.lower())


def end_summary(job: Job) -> str:
    """One line on why a job ended: the collector's summary (``hit time limit``, ``exit 1``, a task's log line)."""
    if job.state is JobState.COMPLETED:
        return ""
    summary = job.raw.get("reason_text") or (job.hints[0] if job.hints else "")
    if not summary and job.kind is JobKind.TASK and job.raw_state.lower() not in ("", "exited", "running"):
        summary = job.raw_state
    if not summary and job.exit_code:
        summary = f"exit {job.exit_code}"
    return summary


def ended_at(job: Job) -> datetime | None:
    """When a terminal job ended (the observation time when the scheduler gave no end)."""
    return job.end_time or (job.observed_at if job.state.is_terminal else None)


def time_left_s(job: Job, now: datetime) -> float | None:
    """Seconds before an active job hits its time limit (elapsed extrapolated from ``observed_at``); None if none."""
    if job.time_limit is None or job.state.group is not StateGroup.ACTIVE:
        return None
    if job.elapsed is not None:
        elapsed = job.elapsed.total_seconds()
        if job.observed_at is not None:
            elapsed += max(0.0, (now - job.observed_at).total_seconds())
    elif job.start_time is not None and not job.est_start:
        elapsed = max(0.0, (now - job.start_time).total_seconds())
    else:
        return None
    return max(0.0, job.time_limit.total_seconds() - elapsed)


def time_low(job: Job, now: datetime) -> bool:
    """Running with less than max(10 % of the limit, 10 min) left."""
    left = time_left_s(job, now)
    if left is None or job.state is not JobState.RUNNING or job.time_limit is None:
        return False
    return left < max(TIME_LOW_FRACTION * job.time_limit.total_seconds(), TIME_LOW_MIN_S)


def gpu_idle_hint(job: Job) -> str | None:
    return next((h for h in job.hints if h.startswith(GPU_IDLE_PREFIX)), None) \
        if job.state.group is StateGroup.ACTIVE else None


@dataclass(frozen=True, slots=True)
class JobEvent:
    kind: EventKind
    key: JobKey
    name: str
    state: JobState
    at: datetime
    summary: str = ""            # why it ended / "↻2" / "8m left of 2:00:00" / "GPU idle 20m"
    exit_code: int | None = None
    restarts: int = 0

    @property
    def category(self) -> Category:
        if self.kind != "ended":
            return self.kind
        if self.state in BAD_END:
            return "ended_bad"
        return "ended_ok" if self.state is JobState.COMPLETED else "ended_other"

    @property
    def word(self) -> str:
        """What happened, as counted in summaries: ``failed``, ``completed``, ``started``, ``low on time`` …"""
        return end_word(self.state) if self.kind == "ended" else _KIND_WORDS[self.kind]

    @property
    def bad(self) -> bool:
        return self.category in ("ended_bad", "time_low", "restarted", "gpu_idle")

    def text(self) -> str:
        """``k2:2332861 orb-swp--train… failed: exit 1`` (one line for a toast or the history)."""
        head = f"{self.key.host}:{self.key.label} {humanize.truncate(self.name, 40)} {self.word}"
        return f"{head}: {self.summary}" if self.summary else head


def _event(kind: EventKind, job: Job, now: datetime, summary: str = "") -> JobEvent:
    return JobEvent(kind, job.key, job.name, job.state, now, summary, job.exit_code, job.restarts)


def _end_pending(job: Job) -> bool:
    """A workstation task that just ended while its log is still being scanned: FAILED or EXITED comes next tick."""
    return job.state is JobState.EXITED and job.hints[:1] == (PENDING_EXIT_HINT,)


def _requeued(old: Job, new: Job) -> bool:
    if new.restarts > old.restarts:
        return True
    back = new.state.group is StateGroup.QUEUED
    return back and (old.state.group is StateGroup.ACTIVE or old.state in (JobState.PREEMPTED, JobState.NODE_FAIL))


class TransitionDetector:
    """Last observed own jobs per host -> events of the next observation (see the module docstring)."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Job]] = {}
        self._since: dict[str, datetime] = {}
        self._fired: dict[str, set[tuple[str, str, int]]] = {}    # host -> (kind, job id, restarts) fired once

    def known(self, host: str) -> bool:
        """The host has a baseline (observed at least once since start / the last reset)."""
        return host in self._jobs

    def hosts(self) -> list[str]:
        return list(self._jobs)

    def reset(self, host: str) -> None:
        """Forget the host: its next observation is a baseline again (outage, disabled, removed)."""
        self._jobs.pop(host, None)
        self._since.pop(host, None)
        self._fired.pop(host, None)

    def observe(self, host: str, jobs: Mapping[str, Job], now: datetime) -> list[JobEvent]:
        mine = {jid: j for jid, j in jobs.items() if j.is_mine}
        prev = self._jobs.get(host)
        if prev is None:                         # baseline: what is already low / idle is not news either
            self._jobs[host], self._since[host] = mine, now
            self._fired[host] = {(kind, jid, j.restarts) for jid, j in mine.items()
                                 for kind, hit in (("time_low", time_low(j, now)), ("gpu_idle", gpu_idle_hint(j)))
                                 if hit}
            return []
        since = self._since[host] - timedelta(seconds=BASELINE_SLACK_S)
        fired = self._fired[host]
        out: list[JobEvent] = []
        for jid, new in mine.items():
            old = prev.get(jid)
            if old is None:
                end = ended_at(new)
                if _end_pending(new):
                    pass                                 # told once classified (the branch below)
                elif new.state.is_terminal and end is not None and end >= since:
                    out.append(_event("ended", new, now, end_summary(new)))
                elif new.state.group is StateGroup.ACTIVE and new.start_time is not None and \
                        not new.est_start and new.start_time >= since:
                    out.append(_event("started", new, now, _where(new)))
            else:
                if old.state.group is StateGroup.QUEUED and new.state.group is StateGroup.ACTIVE:
                    out.append(_event("started", new, now, _where(new)))
                if _requeued(old, new):
                    out.append(_event("restarted", new, now, f"↻{new.restarts}" if new.restarts else "requeued"))
                verdict = _end_pending(old) and (ended_at(new) or now) >= since
                if (not old.state.is_terminal or verdict) and new.state.is_terminal \
                        and not _end_pending(new):     # an end still being classified waits for its verdict
                    out.append(_event("ended", new, now, end_summary(new)))
            if time_low(new, now) and ("time_low", jid, new.restarts) not in fired:
                fired.add(("time_low", jid, new.restarts))
                left = time_left_s(new, now) or 0.0
                limit = new.time_limit.total_seconds() if new.time_limit is not None else None
                out.append(_event("time_low", new, now, f"{humanize.left(left)} left of {humanize.clock(limit)}"))
            idle = gpu_idle_hint(new)
            if idle and ("gpu_idle", jid, new.restarts) not in fired:
                fired.add(("gpu_idle", jid, new.restarts))
                out.append(_event("gpu_idle", new, now, idle))
        self._fired[host] = {f for f in fired if f[1] in mine}
        self._jobs[host] = mine
        return out


def _where(job: Job) -> str:
    if job.kind is JobKind.TASK:
        return ""
    return f"on {job.nodelist}" if job.nodelist else ""


# ---------------------------------------------------------------- selection and coalescing
def wanted(event: JobEvent, settings: NotifySettings, *, watched: bool) -> bool:
    """`[notify] events` for every own job, plus `watched_events` for watched ones."""
    cat = event.category
    return cat in settings.events or (watched and cat in settings.watched_events)


def count_line(words: Iterable[str], sep: str = " · ") -> str:
    """``3 failed · 12 completed · 1 timeout`` (bad news first)."""
    counts = Counter(words)
    order = {w: i for i, w in enumerate(_WORD_ORDER)}
    return sep.join(f"{n} {w}" for w, n in sorted(counts.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))


@dataclass(frozen=True, slots=True)
class Notice:
    """One delivered notification: a single event, or a summary of a burst on one host."""
    host: str
    text: str                                # "k2: 12 failed, 3 completed" / JobEvent.text()
    events: tuple[JobEvent, ...]

    @property
    def summary(self) -> bool:
        return len(self.events) > 1

    @property
    def bad(self) -> bool:
        return any(e.bad for e in self.events)


def coalesce(events: Sequence[JobEvent], *, burst: int = BURST) -> list[Notice]:
    """One notice per event, except a host with more than ``burst`` events in this batch: one summary line."""
    by_host: dict[str, list[JobEvent]] = {}
    for e in events:
        by_host.setdefault(e.key.host, []).append(e)
    out: list[Notice] = []
    for host, evs in by_host.items():
        if len(evs) > burst:
            out.append(Notice(host, f"{host}: {count_line((e.word for e in evs), ', ')}", tuple(evs)))
        else:
            out.extend(Notice(host, e.text(), (e,)) for e in evs)
    return out


def ended_since(jobs: Iterable[Job], since: datetime) -> list[Job]:
    """Own jobs that ended after ``since``, newest first (sacct history included: it is in the store)."""
    out = [(end, j) for j in jobs if j.is_mine and j.state.is_terminal
           and (end := ended_at(j)) is not None and end > since]
    return [j for _end, j in sorted(out, key=lambda p: p[0], reverse=True)]
