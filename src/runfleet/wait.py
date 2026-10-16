"""[WP4] `runfleet wait HOST:JOBID … [--timeout SEC] [--json]`: block until jobs end (scripts, agents, `&&` chains).

The normal pollers run for just the named hosts, at their normal cadence (like an unattended RunFleet: 4× slower
after `idle_after` without input, but never paused), and every state change is printed. Nothing is ever changed on
a host. Exit codes: 0 every job ended COMPLETED, 1 at least one ended otherwise (failed, cancelled, timeout,
preempted, a workstation task's unknown exit, a job that vanished without a final state), 2 timeout, 3 error
(unknown host or job, a host that needs a login, a host unreachable for 10 minutes).

A target names one job (`k2:2332861`, `k2:31270818_3`), a whole array by its parent id (`k2:31270818`: every task,
pending ones included, until all ended) or a workstation task (`lynx:p642229`, the root pid, or the full id).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from runfleet.config import HostConfig, host_name_for
from runfleet.core.clock import Clock
from runfleet.core.service import FleetAPI
from runfleet.models import ConnState, Job, JobKey, JobKind, JobState, to_json

EXIT_COMPLETED, EXIT_FAILED, EXIT_TIMEOUT, EXIT_ERROR = 0, 1, 2, 3
UNREACHABLE_GIVE_UP = 600.0     # a host down (retrying / unreachable) this long ends the wait with an error
UNKNOWN_AFTER = 2               # answered polls of a host that never showed the job -> unknown job
NEVER = 1e9                     # idle_pause_after for `wait`: an unattended wait must keep polling
WAKE_EVERY = 5.0                # re-check deadlines at least this often (store changes wake the loop at once)
SCHEMA = "runfleet.wait/1"
JOB_SECTIONS = ("slurm.squeue", "process.ps")   # their success = an answer that would list the job


class TargetError(ValueError):
    """A HOST:JOBID argument that names no configured host or is malformed (exit 3)."""


@dataclass(frozen=True, slots=True)
class Target:
    host: str                    # RunFleet host name
    job: str                     # as given: "2332861", "31270818" (array), "p642229", "642229"
    text: str                    # the argument, for messages

    def matches(self, jobs: Mapping[str, Job]) -> list[Job]:
        """Jobs of the host snapshot this target names, in id order."""
        if self.job in jobs:
            return [jobs[self.job]]
        array = [j for j in jobs.values() if j.array_job_id == self.job]
        if array:
            return sorted(array, key=lambda j: j.key.id)
        return sorted((j for j in jobs.values()
                       if j.key.label == self.job or (j.kind is JobKind.TASK and j.job_id_raw == self.job)),
                      key=lambda j: j.key.id)


def split_target(text: str) -> tuple[str, str]:
    host, sep, job = text.rpartition(":")
    if not sep or not host or not job or any(c.isspace() for c in text):
        raise TargetError(f"expected HOST:JOBID, got {text!r}")
    return host, job


def resolve_targets(texts: Sequence[str], hosts: Sequence[HostConfig]) -> list[Target]:
    """HOST:JOBID arguments -> targets on configured host names (the host part may be the ssh alias)."""
    out: list[Target] = []
    for text in texts:
        host, job = split_target(text)
        cfg = next((h for h in hosts if host in (h.name, h.ssh) or host_name_for(host) == h.name), None)
        if cfg is None:
            raise TargetError(f"unknown host {host!r} in {text!r}")
        if not any((t.host, t.job) == (cfg.name, job) for t in out):     # `k2:101 k2-alias:101` waits once
            out.append(Target(cfg.name, job, text))
    return out


def is_final(job: Job) -> bool:
    """Ended for good: a terminal state, or a vanished job whose accounting never came (final state unknown)."""
    return job.state.is_terminal or (job.source == "vanished" and job.state is JobState.UNKNOWN)


def answer_mark(hs: Any) -> Any:
    """Changes once per freshly listed job set: the last success of the job-listing sections (squeue, ps); without
    those (stub collectors) the snapshot's job mapping, replaced whenever a collector delivers jobs. None = none yet
    (a failed squeue, or a tick that only refreshed metrics, is no answer)."""
    secs = hs.status.sections
    listed = [secs[n].last_ok for n in JOB_SECTIONS if n in secs]
    if listed:
        oks = tuple(at for at in listed if at is not None)
        return oks or None
    jobs = hs.snapshot.jobs if hs.snapshot is not None else None
    return (id(jobs), jobs) if jobs is not None else None        # kept in Waiter._marks: the id is not reused


class Waiter:
    """The bookkeeping of one `wait`, fed with store snapshots (`check`), pure apart from `emit`."""

    def __init__(self, targets: Sequence[Target], *, emit: Callable[[str], None], as_json: bool = False,
                 now: Callable[[], Any]) -> None:
        self.targets = list(dict.fromkeys(targets))
        self.emit_line = emit
        self.as_json = as_json
        self.now = now
        self.states: dict[JobKey, JobState] = {}
        self.seen: dict[Target, bool] = {t: False for t in self.targets}
        self.finals: dict[Target, list[Job]] = {}
        self.members: dict[Target, dict[JobKey, Job]] = {t: {} for t in self.targets}   # every job a target matched
        self.polls: dict[str, int] = {}         # host -> answered polls seen
        self._marks: dict[str, Any] = {}        # host -> answer_mark() last counted as an answered poll
        self.down_since: dict[str, float] = {}

    # ------------------------------------------------------------ output
    def emit(self, event: str, text: str, **fields: Any) -> None:
        if self.as_json:
            doc = {"schema": SCHEMA, "event": event, "at": to_json(self.now()), **fields}
            self.emit_line(json.dumps(doc, sort_keys=False, default=str))
        else:
            self.emit_line(f"{self.now().astimezone().strftime('%H:%M:%S')}  {text}")

    def _job_event(self, job: Job, old: JobState | None) -> None:
        exit_text = ""
        if job.state.is_terminal and job.exit_code is not None:
            exit_text = f"  exit {job.exit_code}" + (f":{job.exit_signal}" if job.exit_signal else "")
        where = f"  on {job.nodelist}" if job.nodelist and not job.state.is_terminal else ""
        if old is None:
            text = f"{job.key}  {job.state.value:<3} {job.name}{where}{exit_text}"
        else:
            text = f"{job.key}  {old.value} → {job.state.value}{where}{exit_text}"
        self.emit("state", text, job=str(job.key), name=job.name, state=job.state.value,
                  previous=old.value if old is not None else None, raw_state=job.raw_state,
                  exit_code=job.exit_code, final=is_final(job))

    # ------------------------------------------------------------ one look at the store
    def check(self, store: Any, mono: float) -> int | None:
        """Exit code once decided, else None."""
        for host in dict.fromkeys(t.host for t in self.targets):
            hs = store.hosts.get(host)
            if hs is None:
                return self.error(f"{host}: no longer configured")
            st = hs.status.state
            if st in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
                what = "needs a login" if st is ConnState.NEEDS_AUTH else "has a host key problem"
                if hs.status.mux == "session" and st is ConnState.NEEDS_AUTH:
                    # a session login lives in the process that made it: `runfleet connect` is only an auth check
                    return self.error(f"{host} {what} — `runfleet wait` cannot use a Duo/password login with the "
                                      "session transport (Windows): use a key or ssh-agent for this host, or watch "
                                      "it in RunFleet (press c on it)")
                return self.error(f"{host} {what} — run `runfleet connect {host}`, then wait again")
            if st in (ConnState.RETRYING, ConnState.UNREACHABLE):
                since = self.down_since.setdefault(host, mono)
                if mono - since >= UNREACHABLE_GIVE_UP:
                    err = hs.status.error.summary if hs.status.error is not None else st.value
                    return self.error(f"{host} unreachable for {mono - since:.0f} s ({err})")
            elif st in (ConnState.OK, ConnState.DEGRADED):
                self.down_since.pop(host, None)
                mark = answer_mark(hs)
                if mark is not None and self._marks.get(host) != mark:
                    self._marks[host] = mark
                    self.polls[host] = self.polls.get(host, 0) + 1
            elif st is ConnState.DISABLED:
                return self.error(f"{host} is disabled")
        for t in self.targets:
            if t in self.finals:
                continue
            hs = store.hosts[t.host]
            jobs = hs.snapshot.jobs if hs.snapshot is not None else {}
            matched = t.matches(jobs)
            for job in matched:
                self.members[t][job.key] = job
                old = self.states.get(job.key)
                if old is not job.state:
                    self.states[job.key] = job.state
                    self._job_event(job, old)
            if matched:
                self.seen[t] = True
                if all(is_final(j) for j in matched):
                    # array tasks that ended earlier and aged out of the snapshot still count (a failed one fails)
                    keys = {j.key for j in matched}
                    early = [j for k, j in self.members[t].items() if k not in keys and is_final(j)]
                    self.finals[t] = sorted([*matched, *early], key=lambda j: j.key.id)
            elif self.seen[t]:
                self.emit("gone", f"{t.text}  vanished without a final state", job=t.text)
                self.finals[t] = []
            elif self.polls.get(t.host, 0) >= UNKNOWN_AFTER:
                return self.error(f"{t.text}: no such job on {t.host} (not queued, running, nor ended recently)")
        if all(t in self.finals for t in self.targets):
            return self.finish()
        return None

    def error(self, message: str) -> int:
        self.emit("error", f"error: {message}", exit=EXIT_ERROR, message=message)
        return EXIT_ERROR

    def finish(self) -> int:
        jobs = [j for js in self.finals.values() for j in js]
        bad = [j for j in jobs if j.state is not JobState.COMPLETED]
        gone = [t.text for t, js in self.finals.items() if not js]
        if not bad and not gone:
            n = len(jobs)
            self.emit("done", f"all {n} job{'s' if n != 1 else ''} completed", exit=EXIT_COMPLETED,
                      result="completed", jobs={str(j.key): j.state.value for j in jobs})
            return EXIT_COMPLETED
        names = [f"{j.key} {j.state.name}" for j in bad] + [f"{t} vanished" for t in gone]
        self.emit("done", f"not all completed: {', '.join(names)}", exit=EXIT_FAILED, result="failed",
                  jobs={str(j.key): j.state.value for j in jobs})
        return EXIT_FAILED

    def timeout(self, seconds: float) -> int:
        left = [t.text for t in self.targets if t not in self.finals]
        self.emit("timeout", f"timed out after {seconds:g} s; still waiting for {', '.join(left)}",
                  exit=EXIT_TIMEOUT, result="timeout", waiting=left)
        return EXIT_TIMEOUT


async def wait_for(service: FleetAPI, targets: Sequence[Target], *, clock: Clock, timeout: float | None,
                   emit: Callable[[str], None], as_json: bool = False) -> int:
    """Start the pollers, follow the targets until decided, stop the pollers (always)."""
    waiter = Waiter(targets, emit=emit, as_json=as_json, now=clock.now)
    changed = asyncio.Event()
    unsubscribe = service.store.subscribe(lambda _cs: changed.set())
    started = clock.monotonic()
    try:
        await service.start()
        while True:
            changed.clear()
            code = waiter.check(service.store, clock.monotonic())
            if code is not None:
                return code
            left = None if timeout is None else timeout - (clock.monotonic() - started)
            if left is not None and left <= 0:
                return waiter.timeout(timeout or 0)
            step = WAKE_EVERY if left is None else min(WAKE_EVERY, left)
            waiters = {asyncio.ensure_future(changed.wait()), asyncio.ensure_future(clock.sleep(step))}
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()
    finally:
        unsubscribe()
        await service.stop()
