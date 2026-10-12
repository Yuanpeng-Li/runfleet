"""[WP1] FROZEN-API FleetStore, HostState, ChangeSet (DESIGN.md §3.7)."""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime

from runfleet.backends.base import CollectorOutput
from runfleet.config import HostConfig
from runfleet.models import (GPU, Capabilities, ConnState, HostSnapshot, HostStatus, Job, JobKey, StateGroup)

log = logging.getLogger("runfleet.core.store")
_FRESH_STATES = (ConnState.OK, ConnState.DEGRADED)

@dataclass
class HostState:
    config: HostConfig
    status: HostStatus
    snapshot: HostSnapshot | None        # last good merged snapshot (never cleared by failures)

    def stale(self, now: datetime) -> bool:
        """True if snapshot is None, state not in {OK, DEGRADED}, or
        now - status.last_ok > 2 * status.tick_interval + 5 s (effective interval: hidden/idle throttling is normal)."""
        st = self.status
        if self.snapshot is None or st.state not in _FRESH_STATES or st.last_ok is None:
            return True
        return (now - st.last_ok).total_seconds() > 2 * (st.tick_interval or 0.0) + 5.0

@dataclass(frozen=True, slots=True)
class ChangeSet:
    host: str
    status: bool = False
    jobs_added: frozenset[str] = frozenset()
    jobs_changed: frozenset[str] = frozenset()   # material changes only (job_material_eq)
    jobs_removed: frozenset[str] = frozenset()
    gpus: bool = False
    procs: bool = False
    nodes: bool = False
    metrics: bool = False
    paused: bool = False                         # idle pause toggled (host == "*")
    live: frozenset[str] = frozenset()           # job ids whose AttachState changed (§7.16): re-read attach_state()

class FleetStore:
    """Current state of every configured host. Single-threaded (the asyncio loop); callbacks run synchronously.

    The UI only reads (`hosts`, `jobs()`, `job()`, `gpus()`, `counts()`, `subscribe()`); core writes through `apply`,
    `set_status`, `set_idle_paused`, plus the core-only helpers `set_caps`, `set_snapshot`, `add_host`, `remove_host`
    and `reorder` (runtime host management from Settings, §7.13), and `notify_live` (Attach, §7.16).
    """

    hosts: dict[str, HostState]                  # insertion order = config order = scope numbers 1..9
    idle_paused: bool

    def __init__(self, configs: Iterable[HostConfig] = ()) -> None:
        self.hosts = {}
        self.idle_paused = False
        self._subscribers: list[Callable[[ChangeSet], None]] = []
        for cfg in configs:
            self.add_host(cfg)

    # ------------------------------------------------------------ read side
    def subscribe(self, cb: Callable[[ChangeSet], None]) -> Callable[[], None]:  # returns unsubscribe
        self._subscribers.append(cb)

        def unsubscribe() -> None:
            if cb in self._subscribers:
                self._subscribers.remove(cb)
        return unsubscribe

    def _selected(self, hosts: Iterable[str] | None) -> Iterator[HostState]:
        if hosts is None:
            yield from self.hosts.values()
            return
        for name in hosts:
            hs = self.hosts.get(name)
            if hs is not None:
                yield hs

    def jobs(self, hosts: Iterable[str] | None = None) -> Iterator[Job]:
        for hs in self._selected(hosts):
            if hs.snapshot is not None:
                yield from hs.snapshot.jobs.values()

    def job(self, key: JobKey) -> Job | None:
        hs = self.hosts.get(key.host)
        if hs is None or hs.snapshot is None:
            return None
        return hs.snapshot.jobs.get(key.id)

    def gpus(self, hosts: Iterable[str] | None = None) -> Iterator[GPU]:
        for hs in self._selected(hosts):
            if hs.snapshot is not None:
                yield from hs.snapshot.gpus

    def counts(self, hosts: Iterable[str] | None = None) -> Mapping[StateGroup, int]:  # header totals
        c = Counter(j.state.group for j in self.jobs(hosts))
        return {g: c.get(g, 0) for g in StateGroup}

    # ------------------------------------------------------------ core only
    def add_host(self, cfg: HostConfig) -> ChangeSet:
        state = ConnState.NEW if cfg.enabled else ConnState.DISABLED
        self.hosts[cfg.name] = HostState(config=cfg, status=HostStatus(name=cfg.name, state=state), snapshot=None)
        return self._notify(ChangeSet(host=cfg.name, status=True))

    def remove_host(self, host: str) -> ChangeSet:
        """Core only: the host leaves the fleet (its jobs disappear from every view)."""
        hs = self.hosts.pop(host, None)
        if hs is None:
            return ChangeSet(host=host)
        jobs = frozenset(hs.snapshot.jobs) if hs.snapshot is not None else frozenset()
        return self._notify(ChangeSet(host=host, status=True, jobs_removed=jobs, gpus=True, procs=True,
                                      nodes=True, metrics=True))

    def reorder(self, names: Iterable[str]) -> ChangeSet:
        """Core only: hosts named first in this order (scope numbers, card order); the rest keep their order."""
        wanted = [n for n in dict.fromkeys(names) if n in self.hosts]
        order = wanted + [n for n in self.hosts if n not in wanted]
        if order == list(self.hosts):
            return ChangeSet(host="*")
        items = [(n, self.hosts[n]) for n in order]
        self.hosts.clear()                       # in place: readers holding `store.hosts` see the new order
        self.hosts.update(items)
        return self._notify(ChangeSet(host="*", status=True))

    def _notify(self, change: ChangeSet) -> ChangeSet:
        for cb in list(self._subscribers):
            try:
                cb(change)
            except Exception:                     # a subscriber bug must never break polling
                log.exception("store subscriber failed")
        return change

    def _snapshot_for(self, host: str, taken_at: datetime) -> HostSnapshot:
        hs = self.hosts[host]
        if hs.snapshot is not None:
            return hs.snapshot
        return HostSnapshot(host=host, taken_at=taken_at, caps=None, jobs={}, gpus=(), procs=(), nodes=(),
                            partitions=(), metrics=None)

    def apply(self, host: str, out: CollectorOutput, taken_at: datetime) -> ChangeSet:
        """Merge one tick's merged CollectorOutput into the host snapshot (None fields keep the last value)."""
        hs = self.hosts[host]
        snap = self._snapshot_for(host, taken_at)
        added: frozenset[str] = frozenset()
        changed: frozenset[str] = frozenset()
        removed: frozenset[str] = frozenset()
        updates: dict[str, object] = {"taken_at": taken_at}
        if out.jobs is not None:
            old, new = snap.jobs, dict(out.jobs)
            added = frozenset(new.keys() - old.keys())
            removed = frozenset(old.keys() - new.keys())
            changed = frozenset(k for k in new.keys() & old.keys() if not job_material_eq(old[k], new[k]))
            updates["jobs"] = new
        gpus_changed = out.gpus is not None and tuple(out.gpus) != snap.gpus
        procs_changed = out.procs is not None and tuple(out.procs) != snap.procs
        nodes_changed = ((out.nodes is not None and tuple(out.nodes) != snap.nodes)
                         or (out.partitions is not None and tuple(out.partitions) != snap.partitions))
        metrics_changed = out.metrics is not None and out.metrics != snap.metrics
        if out.gpus is not None:
            updates["gpus"] = tuple(out.gpus)
        if out.procs is not None:
            updates["procs"] = tuple(out.procs)
        if out.nodes is not None:
            updates["nodes"] = tuple(out.nodes)
        if out.partitions is not None:
            updates["partitions"] = tuple(out.partitions)
        if out.metrics is not None:
            updates["metrics"] = out.metrics
        if out.caps_patch and snap.caps is not None:
            updates["caps"] = _patch_caps(snap.caps, out.caps_patch)
        hs.snapshot = dataclasses.replace(snap, **updates)   # type: ignore[arg-type]
        change = ChangeSet(host=host, jobs_added=added, jobs_changed=changed, jobs_removed=removed,
                           gpus=gpus_changed, procs=procs_changed, nodes=nodes_changed, metrics=metrics_changed)
        if added or changed or removed or gpus_changed or procs_changed or nodes_changed or metrics_changed:
            self._notify(change)
        return change

    def set_status(self, host: str, status: HostStatus) -> ChangeSet:
        hs = self.hosts[host]
        if hs.status == status:
            return ChangeSet(host=host)
        hs.status = status
        return self._notify(ChangeSet(host=host, status=True))

    def set_idle_paused(self, paused: bool) -> ChangeSet:
        if self.idle_paused == paused:
            return ChangeSet(host="*")
        self.idle_paused = paused
        return self._notify(ChangeSet(host="*", paused=True))

    def notify_live(self, host: str, job_ids: Iterable[str]) -> ChangeSet:
        """Core only (Attach, §7.16): the AttachState of these jobs changed. The store keeps no attach data itself;
        subscribers re-read `FleetAPI.attach_state(JobKey(host, id))`."""
        ids = frozenset(job_ids)
        if not ids or host not in self.hosts:
            return ChangeSet(host=host)
        return self._notify(ChangeSet(host=host, live=ids))

    def set_caps(self, host: str, caps: Capabilities, taken_at: datetime) -> ChangeSet:
        """Core only: capabilities from detection (or the cache) become part of the snapshot."""
        snap = self._snapshot_for(host, taken_at)
        if snap.caps == caps and self.hosts[host].snapshot is not None:
            return ChangeSet(host=host)
        self.hosts[host].snapshot = dataclasses.replace(snap, caps=caps)
        return self._notify(ChangeSet(host=host, status=True))

    def set_snapshot(self, host: str, snapshot: HostSnapshot) -> ChangeSet:
        """Core only: warm start from the cache (shown stale until the first successful poll)."""
        self.hosts[host].snapshot = snapshot
        return self._notify(ChangeSet(host=host, status=True, jobs_added=frozenset(snapshot.jobs), gpus=True,
                                      procs=True, nodes=True, metrics=True))


def _patch_caps(caps: Capabilities, patch: Mapping[str, object]) -> Capabilities:
    known = {f.name for f in dataclasses.fields(caps)}
    valid = {k: v for k, v in patch.items() if k in known}
    for k in patch.keys() - valid.keys():
        log.warning("ignoring unknown capability %r in caps_patch", k)
    return dataclasses.replace(caps, **valid)   # type: ignore[arg-type]


# raw fields that only restate the clock (squeue TimeUsed/TimeLeft, scontrol RunTime, sacct Elapsed of a running
# job): they changed on every poll and made every running job "materially changed" -> a repaint per tick
VOLATILE_RAW = frozenset({"TimeUsed", "TimeLeft", "RunTime", "Elapsed"})


def job_material_eq(a: Job, b: Job) -> bool:
    """== ignoring observed_at, elapsed and the clock-derived raw fields, so a pure time tick does not repaint rows
    (the UI extrapolates ELAPSED/LEFT itself)."""
    if a.raw.keys() != b.raw.keys() or any(a.raw[k] != b.raw[k] for k in a.raw if k not in VOLATILE_RAW):
        return False
    return dataclasses.replace(a, observed_at=None, elapsed=None, raw={}) == dataclasses.replace(
        b, observed_at=None, elapsed=None, raw={})
