"""Resubmissions (§5.2.12): which later job of yours carries on a Slurm job that ended badly.

A partition with PreemptMode CANCEL ends a preempted job PREEMPTED; resuming it is a new `sbatch` (new job id, same
name — what hpc-run does), and so is resubmitting after a crash, a node failure or a cancel. Slurm keeps no link
between the two; this module draws it from the jobs RunFleet already holds, pure and per host, so the TUI, `agent`,
MCP and `snapshot --json` all tell the same story: the old job is **superseded** ("resubmitted as 58721477 · done"),
the new one **resumes** it.

The rule (exact, never fuzzy — a missing link only means "not seen", never "not resubmitted"):

* both are Slurm jobs of yours (`is_mine`, same user) on the same host, with a submit time;
* the earlier one ended badly: a terminal state outside DONE (FAILED, TIMEOUT, OOM, NODE_FAIL, PREEMPTED, BOOT_FAIL,
  DEADLINE, CANCELLED) with an end time;
* the same job name, or the same name with a higher hpc-run attempt (`…--s45--a01` → `…--s45--a02`; the seed and
  everything else must match);
* the later one was submitted strictly later, and did not run alongside it: not started yet, or started no earlier than
  `CONCURRENT_SLACK` before the old one ended (a chained `--dependency=singleton` segment is submitted ahead and waits,
  so the submit time alone may precede the end; two copies running at once are duplicates, not a resubmission);
* the same working directory when both are known; array tasks only link to the same task index of another array;
  an array summary row (`…_[1-9%2]`) never links.

The successor is the earliest such job; several bad endings may point at the same one (two failed duplicates, then
one resubmit). A chain (F → F → F → CD) is followed by `latest_of`. Same-id requeues (PreemptMode REQUEUE) are not
links: the job keeps its id and `restarts` counts them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta

from runfleet.models import Job, JobKey, JobKind, StateGroup

CONCURRENT_SLACK = timedelta(seconds=60)
_ATTEMPT = re.compile(r"^(?P<base>.+)--a(?P<n>\d+)$")
_BAD_END = (StateGroup.FAILED, StateGroup.CANCELLED, StateGroup.ENDED)


@dataclass(frozen=True, slots=True)
class Lineage:
    """Links among the jobs of one or more hosts. `successor[k]` is the job that carries on `k` (k is superseded);
    `predecessor[k]` the most recently submitted bad ending that `k` carries on."""
    successor: Mapping[JobKey, Job] = field(default_factory=dict)
    predecessor: Mapping[JobKey, Job] = field(default_factory=dict)

    def next_of(self, key: JobKey) -> Job | None:
        return self.successor.get(key)

    def prev_of(self, key: JobKey) -> Job | None:
        return self.predecessor.get(key)

    def superseded(self, key: JobKey) -> bool:
        return key in self.successor

    def latest_of(self, key: JobKey) -> Job | None:
        """The newest job of the chain after `key` (None when nothing carries it on)."""
        job, seen = self.successor.get(key), {key}
        while job is not None and job.key not in seen:
            seen.add(job.key)
            nxt = self.successor.get(job.key)
            if nxt is None:
                return job
            job = nxt
        return job

    def chain_after(self, key: JobKey) -> tuple[Job, ...]:
        """Every job after `key`, oldest first (cycle-safe)."""
        out: list[Job] = []
        job, seen = self.successor.get(key), {key}
        while job is not None and job.key not in seen:
            seen.add(job.key)
            out.append(job)
            job = self.successor.get(job.key)
        return tuple(out)


EMPTY = Lineage()


def split_attempt(name: str) -> tuple[str, int | None]:
    """`("…--s45", 1)` for `"…--s45--a01"`; `(name, None)` for a name without an hpc-run attempt field."""
    m = _ATTEMPT.match(name)
    return (m.group("base"), int(m.group("n"))) if m else (name, None)


def _eligible(job: Job) -> bool:
    return (job.kind is JobKind.SLURM and job.is_mine and job.submit_time is not None and bool(job.name)
            and not (job.array_task or "").startswith("["))


def _bad_end(job: Job) -> bool:
    return job.state.is_terminal and job.state.group in _BAD_END and job.end_time is not None


def _carries_on(old: Job, new: Job) -> bool:
    """`new` resumes `old` (both eligible, same host / user / base name; `old` a bad ending)."""
    if new.key == old.key or new.user != old.user:
        return False
    assert old.submit_time is not None and new.submit_time is not None and old.end_time is not None
    if new.submit_time <= old.submit_time:
        return False
    if new.start_time is not None and not new.est_start and new.start_time < old.end_time - CONCURRENT_SLACK:
        return False                                    # ran alongside it: a duplicate, not a resubmission
    if old.workdir and new.workdir and old.workdir.rstrip("/") != new.workdir.rstrip("/"):
        return False
    if old.array_task or new.array_task:
        if not (old.array_task and new.array_task) or old.array_task != new.array_task:
            return False
        if old.array_job_id and old.array_job_id == new.array_job_id:
            return False                                # two tasks of one array
    if old.name == new.name:
        return True
    _, a_old = split_attempt(old.name)
    _, a_new = split_attempt(new.name)
    return a_old is not None and a_new is not None and a_new > a_old


def link(jobs: Iterable[Job]) -> Lineage:
    """The resubmission links among `jobs` (any mix of hosts; only same-host pairs link)."""
    runs: dict[tuple[str, str, str], list[Job]] = defaultdict(list)
    for j in jobs:
        if _eligible(j):
            runs[(j.key.host, j.user, split_attempt(j.name)[0])].append(j)
    successor: dict[JobKey, Job] = {}
    predecessor: dict[JobKey, Job] = {}
    for group in runs.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda j: (j.submit_time, j.key.id))
        for i, old in enumerate(group):
            if not _bad_end(old):
                continue
            nxt = next((new for new in group[i + 1:] if _carries_on(old, new)), None)
            if nxt is None:
                continue
            successor[old.key] = nxt
            prev = predecessor.get(nxt.key)
            if prev is None or (prev.submit_time, prev.key.id) < (old.submit_time, old.key.id):
                predecessor[nxt.key] = old
    return Lineage(successor, predecessor) if successor else EMPTY
