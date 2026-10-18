"""[WP4] `runfleet agent events --since`: what happened on the fleet, rebuilt statelessly from the cached jobs (pure).

Sources (docs/AGENT_INTERFACE.md §3.8): submit times, real (not estimated) start times, ends (`core.events.ended_since`,
sacct history included) and the workstation tasks of the cache's `ended_tasks` table (7 days, beyond the snapshot's
`recent_window`). Conditions that hold now carry no time: `time_low` and each host's current state with its start
(`host_status.at` dates the current state only: transitions are not rebuilt).

Event time is not observation time: accounting can report at 12:05 a job that ended at 11:59, so a window starting at
12:00 never shows that end. Callers (the skill) query overlapping windows and dedupe by `Event.id`, which is stable
across calls: "<job key or host>:<printed kind>" (`k2:2331977:ended_bad`; a requeued run `k2:4:started:1`; a host
state `lynx:host:needs_auth[:<since epoch>]`). The job part is the full key (`JobKey`), which for Slurm is the display
id and for a workstation task adds the start time (`labstation:p363942-1788696459:started`): a reused pid is another
task, and two tasks sharing the display id `labstation:p363942` in one view never share an event id. `Event.job` is
the display form HOST:LABEL. Only the user's own jobs count; names and texts are raw (agent.view cleans them).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from runfleet.agent.data import OLD_AFTER, SECTION_GROUPS, STALE_AFTER, FleetView, HostView
from runfleet.core import events as core_events
from runfleet.core.cache import ENDED_TASKS_RETENTION
from runfleet.models import ConnState, Job, JobState, StateGroup
from runfleet.util import humanize
from runfleet.util.logscan import PENDING_EXIT_HINT

EventKind = Literal["submitted", "started", "ended_ok", "ended_bad", "ended_other", "time_low", "host"]
KINDS: tuple[EventKind, ...] = ("submitted", "started", "ended_ok", "ended_bad", "ended_other", "time_low", "host")
_QUIET = (None, ConnState.OK, ConnState.DISABLED, ConnState.NEW)     # host states that are not news


@dataclass(frozen=True, slots=True)
class Event:
    id: str                          # stable across calls: "k2:2331977:ended_bad", "lynx:host:needs_auth:1792310470",
                                     # "labstation:p363942-1788696459:started" (tasks: the full key, see above)
    at: datetime | None              # None: holds now (time_low, host state)
    kind: EventKind
    host: str
    job: str | None                  # display id "k2:2331977" / "labstation:p642229"; None for host items
    name: str                        # raw job name ("" for host items)
    text: str                        # raw detail: "OUT_OF_MEMORY · out of memory", "14m left of 1d", "needs_auth"
    state: str | None = None         # job state code ("OOM") or host state ("needs_auth")
    since: datetime | None = None    # host items: when the current state began


@dataclass(frozen=True, slots=True)
class Coverage:
    """What one host's data can tell about the window: anything outside it is unknown, not "nothing happened"."""
    host: str
    jobs_at: datetime | None         # newest job listing (None: no data)
    history_at: datetime | None      # newest accounting read (Slurm; None: never / workstation)
    history_from: datetime | None    # oldest end accounting covers (history_at - sacct window; None: unknown)
    tasks_from: datetime | None      # workstations: ended tasks are kept from here (7 days)
    no_data: bool                    # nothing cached for the host
    gap: bool                        # the window starts before what the data covers, or the data stops early


def _all_jobs(hv: HostView) -> dict[str, Job]:
    """The snapshot's jobs plus the cached ended tasks it no longer holds (the snapshot wins on the same id)."""
    jobs: dict[str, Job] = dict(hv.snapshot.jobs) if hv.snapshot is not None else {}
    for task in hv.ended_tasks:
        jobs.setdefault(task.key.id, task)
    return jobs


def host_jobs(hv: HostView) -> Mapping[str, Job]:
    """Every job agent views know for one host (snapshot + ended tasks), by JobKey.id."""
    return _all_jobs(hv)


def _ref(job: Job) -> str:
    return f"{job.key.host}:{job.key.label}"


def _event_key(job: Job) -> str:
    """The job part of an event id: the full key (Slurm: the display id; tasks: pid plus start time)."""
    return str(job.key)


def _end_kind(state: JobState) -> EventKind:
    if state in core_events.BAD_END:
        return "ended_bad"
    return "ended_ok" if state is JobState.COMPLETED else "ended_other"


def _end_text(job: Job) -> str:
    summary = core_events.end_summary(job)
    return f"{job.state.name} · {summary}" if summary else job.state.name


def _job_events(hv: HostView, since: datetime, now: datetime) -> list[Event]:
    jobs = [j for j in _all_jobs(hv).values() if j.is_mine]
    out: list[Event] = []
    for j in jobs:
        ref, base = _ref(j), _event_key(j)
        if j.submit_time is not None and since < j.submit_time <= now:
            out.append(Event(f"{base}:submitted", j.submit_time, "submitted", hv.name, ref, j.name, "", j.state.value))
        started = j.start_time is not None and not j.est_start and j.state.group is not StateGroup.QUEUED
        if started and j.start_time is not None and since < j.start_time <= now:
            where = " ".join(x for x in (_gpus(j), j.nodelist or "") if x)
            sid = f"{base}:started" + (f":{j.restarts}" if j.restarts else "")
            out.append(Event(sid, j.start_time, "started", hv.name, ref, j.name, where, j.state.value))
        # a job seen long ago: time left as of that sighting, marked (never run down to "0s left" meanwhile)
        seen = j.observed_at if j.observed_at is not None and (now - j.observed_at).total_seconds() > OLD_AFTER \
            else now
        if j.state.group is StateGroup.ACTIVE and core_events.time_low(j, seen):
            left = core_events.time_left_s(j, seen) or 0.0
            limit = j.time_limit.total_seconds() if j.time_limit is not None else None
            as_of = f" (as of {humanize.span((now - seen).total_seconds())} ago)" if seen != now else ""
            out.append(Event(f"{base}:time_low", None, "time_low", hv.name, ref, j.name,
                             f"{humanize.span(left)} left of {humanize.span(limit, dash='unknown')}{as_of}",
                             j.state.value))
    for j in core_events.ended_since(jobs, since):
        end = core_events.ended_at(j)
        if end is None or end > now or (j.state is JobState.EXITED and j.hints[:1] == (PENDING_EXIT_HINT,)):
            continue                 # an end still being classified shows (with its verdict) in the next window
        kind = _end_kind(j.state)
        out.append(Event(f"{_event_key(j)}:{kind}", end, kind, hv.name, _ref(j), j.name, _end_text(j), j.state.value))
    return out


def _gpus(job: Job) -> str:
    n = job.alloc.gpus
    if not n:
        return ""
    return f"{n}×{humanize.gpu_model(job.alloc.gpu_type)}" if job.alloc.gpu_type else f"{n} GPU"


def _host_event(hv: HostView) -> Event | None:
    if not hv.enabled or hv.state in _QUIET or hv.state is None:
        return None
    since = hv.state_since
    eid = f"{hv.name}:host:{hv.state.value}" + (f":{int(since.timestamp())}" if since is not None else "")
    return Event(eid, None, "host", hv.name, None, "", hv.state.value, hv.state.value, since)


def _failing(hv: HostView, group: str) -> bool:
    if hv.health is None:
        return False
    return any(not s.ok for n in SECTION_GROUPS.get(group, ()) if (s := hv.health.sections.get(n)) is not None)


def coverage(hv: HostView, since: datetime, now: datetime) -> Coverage:
    jobs_at = hv.data_at("jobs")
    history_at = hv.data_at("history")
    history_from = None
    if history_at is not None and hv.sacct_window_s is not None:
        history_from = history_at - timedelta(seconds=hv.sacct_window_s)
    tasks_from = now - timedelta(seconds=ENDED_TASKS_RETENTION) if hv.kind == "workstation" else None
    no_data = hv.snapshot is None
    stale = jobs_at is None or (now - jobs_at).total_seconds() > STALE_AFTER
    if hv.kind == "slurm":            # ends come from accounting: failing or old accounting is a gap too
        hist_bad = history_at is None or (now - history_at).total_seconds() > STALE_AFTER or _failing(hv, "history")
        early = hist_bad or history_from is None or since < history_from
    elif hv.kind == "workstation":
        early = tasks_from is not None and since < tasks_from
    else:
        early = True
    gap = hv.enabled and (no_data or stale or early)
    return Coverage(hv.name, jobs_at, history_at, history_from, tasks_from, no_data, gap)


def events_since(fv: FleetView, since: datetime) -> tuple[list[Event], list[Coverage]]:
    """Events after ``since`` up to ``fv.generated_at``, oldest first, then the "now" items (host states, then
    time_low), plus each host's coverage (config order). Disabled hosts give neither events nor a gap."""
    now = fv.generated_at
    timed: list[Event] = []
    hosts_now: list[Event] = []
    low_now: list[Event] = []
    covers: list[Coverage] = []
    order = {h.name: i for i, h in enumerate(fv.hosts)}
    for hv in fv.hosts:
        covers.append(coverage(hv, since, now))
        if not hv.enabled:
            continue
        hev = _host_event(hv)
        if hev is not None:
            hosts_now.append(hev)
        for ev in _job_events(hv, since, now):
            (low_now if ev.at is None else timed).append(ev)
    timed.sort(key=lambda e: (e.at, order.get(e.host, 0), KINDS.index(e.kind), e.id))
    low_now.sort(key=lambda e: (order.get(e.host, 0), e.id))
    return [*timed, *hosts_now, *low_now], covers


def dedupe(batches: Sequence[Sequence[Event]]) -> list[Event]:
    """Events of overlapping windows, each id once (the latest batch wins: an end classified meanwhile)."""
    seen: dict[str, Event] = {}
    for batch in batches:
        for ev in batch:
            seen[ev.id] = ev
    return list(seen.values())
