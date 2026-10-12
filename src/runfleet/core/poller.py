"""[WP1] HostPoller: ticks, cadence modes, backoff, lanes, persistent MonitorView (§5.4).

One poller per enabled host. It owns the host's connection state machine (§4.4): prepare (ssh -G, mux, master),
detect (capabilities + collector profile), then a loop of *ticks*. At a tick every due cadence group contributes its
sections to **one** probe; results are routed to collectors by `Section.owner`, outputs merged (monitors first,
scheduler last, §3.6) and applied to the store.

Scheduling is drift-free: tick k after an anchor happens at `anchor + (k + u) × T` with `u ~ U(0, 0.2)`, so the
average period is exactly the effective tick `T` (the politeness budget of §1.1 holds over any hour) while individual
gaps vary within ±20 % to spread load. The grid is re-anchored whenever `T` changes, after a wake and after a failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import random
import re
import secrets
import statistics
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from runfleet.backends.base import (Cadence, Collector, CollectorOutput, HostContext, MonitorView, Role,
                                    SchedulerCollector)
from runfleet.config import GeneralSettings, HostConfig
from runfleet.core.cache import Cache, JobMeta, enrich_from_meta
from runfleet.core.clock import Clock
from runfleet.core.store import FleetStore, _patch_caps
from runfleet.errors import RunFleetError
from runfleet.models import (Capabilities, ConnState, ErrorKind, HostError, HostStatus, Job, JobKind, MuxMode,
                             SectionStatus, Visibility)
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, SectionResult
from runfleet.transport import classify
from runfleet.transport.base import ProbeResult, Transport

log = logging.getLogger("runfleet.core.poller")

MAX_BACKOFF = 300.0              # transport backoff cap (§4.4)
WAKE_FLOOR = 3.0                 # manual / post-action ticks: >= 3 s after the previous probe
REDETECT_AFTER = 6 * 3600.0      # background re-detection (§5.1)
SNAPSHOT_CAPS_MAX_AGE = 3600.0   # `snapshot` reuses cached capabilities this young instead of a detect probe
DEFAULT_TICK = 15.0              # only if a profile declares no interval group at all
JOBMETA_TOUCH = 3600.0           # rewrite an unchanged jobmeta row at most hourly (last_seen)
# Busy Slurm controller (§5.4): these sections talk to slurmctld / slurmdbd; their stderr or a slow answer slows the
# host's Slurm cadence ×2 per busy tick (up to ×8), halved again after BUSY_DECAY healthy ticks
BUSY_SECTIONS = frozenset({"slurm.squeue", "slurm.sacct", "slurm.sacct_ids", "slurm.sinfo"})
BUSY_STDERR = re.compile(r"Socket timed out|Rate limit|slurm_load_jobs error|Transport endpoint", re.IGNORECASE)
BUSY_MAX = 8.0
BUSY_RATIO = 3.0                 # a section this many times slower than its rolling median …
BUSY_MIN_MS = 5000               # … and slower than 5 s counts as busy
BUSY_WINDOW = 20                 # rolling window of section durations
BUSY_DECAY = 3                   # healthy ticks before the factor halves


# ======================================================================================= backend hooks
@dataclass(frozen=True, slots=True)
class BackendHooks:
    """The WP2 entry points core uses (§5.1, FROZEN signatures); swappable so core tests can use stub collectors."""
    collectors: Callable[[], Mapping[str, type[Collector]]]
    detect_sections: Callable[[HostConfig, Mapping[str, type[Collector]]], list[Section]]
    build_capabilities: Callable[[Mapping[str, SectionResult], HostConfig, datetime,
                                  Mapping[str, type[Collector]]], Capabilities]

    @classmethod
    def default(cls) -> BackendHooks:
        from runfleet.backends import detect, registry
        return cls(registry.collectors, detect.detect_sections, detect.build_capabilities)


# ======================================================================================= lanes (§4.2)
class Limiter:
    """A semaphore whose limit can shrink at runtime (a refused session drops the on-demand lane to 1)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._cond = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < self.limit)
            self.active += 1
        try:
            yield
        finally:
            async with self._cond:
                self.active -= 1
                self._cond.notify_all()

    async def set_limit(self, limit: int) -> None:
        async with self._cond:
            self.limit = limit
            self._cond.notify_all()


class HostLanes:
    """Per-host session budget: 1 poll probe (the poller loop is sequential), on-demand probes and log streams
    ≤ 1 each on a user master, ≤ 2 otherwise; all probes also share the global `max_parallel_ssh` semaphore."""

    def __init__(self, global_sem: asyncio.Semaphore, mux: MuxMode = "none") -> None:
        self.global_sem = global_sem
        self.on_demand = Limiter(self._limit_for(mux))
        self.stream_limit = self._limit_for(mux)
        self.degraded = False

    @staticmethod
    def _limit_for(mux: MuxMode) -> int:
        return 1 if mux == "user" else 2

    async def configure(self, mux: MuxMode) -> None:
        limit = 1 if self.degraded else self._limit_for(mux)
        self.stream_limit = limit
        await self.on_demand.set_limit(limit)

    async def degrade(self) -> None:
        """`Session open refused by peer`: on-demand lane -> 1 and one stream for the rest of the session."""
        self.degraded = True
        self.stream_limit = 1
        await self.on_demand.set_limit(1)

    @contextlib.asynccontextmanager
    async def poll(self) -> AsyncIterator[None]:
        async with self.global_sem:
            yield

    @contextlib.asynccontextmanager
    async def demand(self) -> AsyncIterator[None]:
        async with self.on_demand.slot(), self.global_sem:
            yield


# ======================================================================================= helpers
def merge_outputs(outputs: Sequence[CollectorOutput]) -> CollectorOutput:
    """§3.6 merge rule: in the given order, the last non-None value of every Optional field wins; errors are
    concatenated, followups unioned, caps patches combined."""
    values: dict[str, object] = {}
    caps: dict[str, object] = {}
    errors: list[HostError] = []
    followups: set[str] = set()
    for out in outputs:
        for f in ("jobs", "gpus", "gpu_procs", "procs", "nodes", "partitions", "metrics"):
            v = getattr(out, f)
            if v is not None:
                values[f] = v
        if out.caps_patch:
            caps.update(out.caps_patch)
        errors.extend(out.errors)
        followups |= out.followups
    return CollectorOutput(**values, caps_patch=caps or None, errors=tuple(errors),  # type: ignore[arg-type]
                           followups=frozenset(followups))


def update_monitor_view(view: MonitorView, out: CollectorOutput, now: datetime) -> MonitorView:
    """Persistent MonitorView: only fields a monitor actually returned replace the previous sample."""
    changes: dict[str, object] = {}
    if out.gpus is not None:
        changes["gpus"] = tuple(out.gpus)
        changes["gpus_at"] = now
    if out.gpu_procs is not None:
        changes["gpu_procs"] = tuple(out.gpu_procs)
        changes.setdefault("gpus_at", now)
    if out.metrics is not None:
        changes["metrics"] = out.metrics
    return dataclasses.replace(view, **changes) if changes else view   # type: ignore[arg-type]


def section_error(r: SectionResult, now: datetime) -> HostError | None:
    """Default HostError for a failed section when the collector did not report a better one."""
    if r.ok:
        return None
    first = next((ln.strip() for ln in r.stderr.splitlines() if ln.strip()), "")
    kind, text = {
        SectionOutcome.TIMEOUT: (ErrorKind.TIMEOUT, "timed out"),
        SectionOutcome.MISSING: (ErrorKind.MISSING, f"not available ({r.stderr or 'missing tool'})"),
        SectionOutcome.INCOMPLETE: (ErrorKind.SSH, "output cut off"),
        SectionOutcome.NOT_RUN: (ErrorKind.SSH, "did not run"),
        SectionOutcome.STDERR_ONLY: (ErrorKind.REMOTE, "error"),
        SectionOutcome.FAILED: (ErrorKind.REMOTE, f"failed (rc {r.rc})"),
    }[r.outcome]
    summary = f"{r.name}: {first or text}"
    return HostError(kind, summary[:80], r.stderr[:4096], section=r.name, at=now)


def is_failure(status: SectionStatus) -> bool:
    """Counts toward DEGRADED and group backoff (a MISSING tool is a capability fact, not a failure)."""
    return not status.ok and status.outcome != SectionOutcome.MISSING.value


class ControllerBackoff:
    """Slows a host's Slurm cadence while its controller is busy (§5.4): a controller section whose stderr says so
    (`Socket timed out`, `Rate limit`, …) or that took more than 3× its rolling median and over 5 s doubles the
    factor (up to ×8); BUSY_DECAY ticks with healthy controller sections (at least one answered) halve it again. A
    plain failure or timeout of one section is the group backoff's business, not this, and no sign of recovery."""

    def __init__(self) -> None:
        self.factor = 1.0
        self._healthy = 0
        self._ms: dict[str, deque[int]] = {}

    def observe(self, results: Mapping[str, SectionResult]) -> bool:
        """One tick's results; True when the factor changed."""
        seen = busy = answered = False
        for name, r in results.items():
            if name not in BUSY_SECTIONS:
                continue
            seen = True
            answered = answered or r.ok
            if BUSY_STDERR.search(r.stderr or ""):
                busy = True
            if r.ms is None:
                continue
            history = self._ms.setdefault(name, deque(maxlen=BUSY_WINDOW))
            if len(history) >= 3 and r.ms > BUSY_MIN_MS and r.ms > BUSY_RATIO * statistics.median(history):
                busy = True
            history.append(r.ms)
        if not seen:
            return False
        old = self.factor
        if busy:
            self.factor, self._healthy = min(BUSY_MAX, self.factor * 2), 0
        elif self.factor > 1 and answered:       # a round that only failed proves no recovery
            self._healthy += 1
            if self._healthy >= BUSY_DECAY:
                self.factor, self._healthy = max(1.0, self.factor / 2), 0
        return self.factor != old


@dataclass
class _GroupPlan:
    """Sections of this tick, with the cadence group each came from (for per-group backoff)."""
    sections: list[Section] = field(default_factory=list)
    group_of: dict[str, str] = field(default_factory=dict)
    owner_groups: dict[str, set[str]] = field(default_factory=dict)   # collector name -> due groups


# ======================================================================================= the poller
class HostPoller:
    def __init__(self, host: HostConfig, transport: Transport, store: FleetStore, clock: Clock, rng: random.Random,
                 lanes: HostLanes, *, general: GeneralSettings, hooks: BackendHooks, cache: Cache | None = None,
                 n_hosts: int = 1, caps: Capabilities | None = None, caps_at: datetime | None = None,
                 jobmeta: Mapping[str, JobMeta] | None = None, ended_tasks: Mapping[str, Job] | None = None) -> None:
        self.cfg = host
        self.name = host.name
        self.transport = transport
        self.store = store
        self.clock = clock
        self.rng = rng
        self.lanes = lanes
        self.general = general
        self.hooks = hooks
        self.cache = cache
        self.n_hosts = n_hosts
        self.caps = caps
        self.caps_at = caps_at
        self.profile: list[Collector] = []
        self.monitor_view = MonitorView()
        self._jobmeta: dict[str, JobMeta] = dict(jobmeta or {})
        # last_seen as last written to the cache (the in-memory meta's last_seen advances every poll)
        self._jobmeta_persisted: dict[str, float] = {jid: m.last_seen for jid, m in self._jobmeta.items()}
        self._ended_tasks: dict[str, Job] = dict(ended_tasks or {})
        self._persisted_ended: set[str] = set(self._ended_tasks)
        # scheduling state (monotonic seconds)
        self._last_run: dict[str, float] = {}
        self._group_fail: dict[str, int] = {}
        self._events: set[str] = set()
        self._forced: set[str] = set()
        self._unplanned: set[str] = set()   # due at the last tick but produced no section (nothing known yet)
        self.controller = ControllerBackoff()
        self._visibility = Visibility.VISIBLE
        self._multiplier = 1.0
        self._next_at = 0.0
        self._anchor: float | None = None
        self._anchor_k = 0
        self._anchor_t = 0.0
        self._last_tick_at: float | None = None
        self._last_probe_at: float | None = None
        # connection state
        self._need_prepare = True
        self._redetect = False
        self._redetected_for_missing = False
        self._consecutive = 0
        self._timeouts_in_row = 0
        self._auth_paused = False
        self._idle_paused = False
        self._wake_pending = False
        self._wake_floor = True
        self._manual_retry = False
        self._force_prepare = False       # next prepare is user-initiated: bypass the 30 s master throttle
        self._event = asyncio.Event()
        if caps is not None:
            self._rebuild_profile(caps)

    # ------------------------------------------------------------ public control (§5.4)
    @property
    def status(self) -> HostStatus:
        return self.store.hosts[self.name].status

    def wake(self, reason: str, groups: frozenset[str] = frozenset()) -> None:
        """Serve an immediate tick (≥ 3 s after the previous probe). `groups` are forced due; a manual wake without
        groups forces the fast groups (period ≤ 2 ticks) and retries a paused (auth) host exactly once."""
        if groups:
            self._forced |= set(groups)
        elif reason == "manual":
            self._forced |= self._fast_groups()
        if reason in ("manual", "connect") and self._auth_paused:
            self._manual_retry = True
        self._wake_pending = True
        self._event.set()

    def reconnect(self) -> None:
        """After an interactive connect: reset backoff, re-prepare, poll now (no 3 s floor)."""
        self._need_prepare = True
        self._redetect = True
        self._consecutive = 0
        self._timeouts_in_row = 0
        self._auth_paused = False
        self._manual_retry = False
        self._force_prepare = True
        self._wake_pending = True
        self._wake_floor = False
        self._publish(state=ConnState.CONNECTING, error=None, consecutive_failures=0)
        self._event.set()

    def set_visibility(self, v: Visibility) -> None:
        if v != self._visibility:
            self._visibility = v
            self._reschedule()

    def set_multiplier(self, m: float) -> None:
        if m != self._multiplier:
            self._multiplier = m
            self._reschedule()

    def set_paused(self, paused: bool) -> None:
        if paused == self._idle_paused:
            return
        self._idle_paused = paused
        if not paused:
            self._next_at = self.clock.monotonic()
            self._anchor = None
        self._event.set()

    # ------------------------------------------------------------ main loop
    async def run(self) -> None:          # until cancelled; never raises (exceptions -> INTERNAL error)
        try:
            while True:
                await self._wait()
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as e:      # a bug must never take the host (or the app) down
                    self._internal_error(e)
        except asyncio.CancelledError:
            if self.status.in_flight:
                self._publish(in_flight=False)
            raise

    async def _wait(self) -> None:
        while True:
            self._event.clear()
            now = self.clock.monotonic()
            delay: float | None
            if self._idle_paused:
                delay = None
            elif self._auth_paused:
                if self._manual_retry:
                    self._manual_retry = False
                    self._force_prepare = True     # "exactly one more attempt" must really attempt (§4.4)
                    return
                delay = None
            elif self._wake_pending:
                last = self._last_probe_at
                delay = 0.0 if last is None or not self._wake_floor else last + WAKE_FLOOR - now
            else:
                delay = self._next_at - now
            if delay is not None and delay <= 0:
                return
            await self._sleep_or_event(delay)

    async def _sleep_or_event(self, delay: float | None) -> None:
        tasks: set[asyncio.Future[Any]] = {asyncio.ensure_future(self._event.wait())}
        if delay is not None:
            tasks.add(asyncio.ensure_future(self.clock.sleep(delay)))
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def _cycle(self) -> None:
        was_wake = self._wake_pending
        self._wake_pending = False
        self._wake_floor = True
        prepared_now = False
        if self._need_prepare or self.caps is None or not self.profile:
            err = await self._prepare_and_detect()
            if err is not None:
                await self._on_transport_failure(err)
                return
            prepared_now = True        # a cached profile polls first; re-detection waits for the next cycle
        elif self.transport.mux in ("user", "managed") and not await self.transport.check_master():
            # §4.1: no probe while a master is being made; mux hosts never fall back to direct probes
            self._publish(state=ConnState.CONNECTING)
            err = await self._ensure_master()
            if err is not None:
                await self._on_transport_failure(err)
                return
            self._schedule_next(self.clock.monotonic())
            return
        if not prepared_now and (self._redetect or (
                self.caps_at is not None and (self.clock.now() - self.caps_at).total_seconds() >= REDETECT_AFTER)):
            self._redetect = False
            err = await self._detect()
            if err is not None:
                await self._on_transport_failure(err)
                return
        await self.poll_tick(anchor=was_wake)

    # ------------------------------------------------------------ prepare and detect
    async def _ensure_master(self) -> HostError | None:
        force, self._force_prepare = self._force_prepare, False
        async with self.lanes.poll():
            return await self.transport.prepare(force=force)

    async def _prepare_and_detect(self) -> HostError | None:
        if self.status.state not in (ConnState.RETRYING, ConnState.UNREACHABLE):
            self._publish(state=ConnState.CONNECTING)
        err = await self._ensure_master()
        if err is not None:
            return err
        self._need_prepare = False
        await self.lanes.configure(self.transport.mux)
        self._publish(mux=self.transport.mux, ssh_log=self.transport.ssh_log)
        if self.caps is None or not self.profile:
            return await self._detect()
        snap = self.store.hosts[self.name].snapshot
        if snap is None or snap.caps is None:
            self.store.set_caps(self.name, self.caps, self.clock.now())
        if self.caps_at is None or (self.clock.now() - self.caps_at).total_seconds() > 60:
            self._redetect = True     # cached profile used now; detection re-runs on the next cycle (§5.1)
        return None

    async def _detect(self) -> HostError | None:
        regs = self.hooks.collectors()
        script = framing.make_script(self.hooks.detect_sections(self.cfg, regs), "detect")
        res = await self._run_probe(script)
        if res.error is not None:
            return res.error
        self._on_transport_ok(res)
        now = self.clock.now()
        try:
            caps = self.hooks.build_capabilities(res.sections, self.cfg, now, regs)
        except RunFleetError as e:
            ident = res.sections.get("detect.id")
            kind = ErrorKind.PARSE if ident is not None and ident.ok else ErrorKind.REMOTE
            return HostError(kind, f"detection failed: {e}"[:80], str(e), at=now)
        self._set_caps(caps, now)
        return None

    def _set_caps(self, caps: Capabilities, now: datetime) -> None:
        changed_profile = self.caps is None or caps.profile != self.caps.profile or not self.profile
        self.caps, self.caps_at = caps, now
        self.store.set_caps(self.name, caps, now)
        if self.cache is not None:
            self.cache.put_caps(self.name, caps, now)
        if changed_profile:
            self._rebuild_profile(caps)

    def apply_caps_patch(self, patch: Mapping[str, object]) -> bool:
        """A capability learned outside a tick (Attach's ingest_attach caps_patch, §7.16: node_ssh): store caps +
        cache, keeping caps_at (it expires with the detected caps; a re-detect forgets it). True if anything changed."""
        if self.caps is None or not patch:
            return False
        caps = _patch_caps(self.caps, patch)
        if caps == self.caps:
            return False
        now = self.clock.now()
        self.caps = caps
        self.store.set_caps(self.name, caps, now)
        if self.cache is not None:
            self.cache.put_caps(self.name, caps, self.caps_at or now)
        return True

    def _rebuild_profile(self, caps: Capabilities) -> None:
        regs = self.hooks.collectors()
        profile: list[Collector] = []
        for name in caps.profile:
            cls = regs.get(name)
            if cls is None:
                log.warning("%s: collector %r of the profile is not registered", self.name, name)
                continue
            profile.append(cls(self.cfg))
        self.profile = profile
        hs = self.store.hosts.get(self.name)
        if hs is not None:            # a collector that left the profile must not keep the host DEGRADED
            owners = {c.name for c in profile}
            kept = {n: st for n, st in hs.status.sections.items() if n.split(".", 1)[0] in owners}
            if len(kept) != len(hs.status.sections):
                self._publish(sections=kept)
        self._last_run.clear()
        self._group_fail.clear()
        self._events.clear()
        self.monitor_view = MonitorView()

    @property
    def scheduler(self) -> SchedulerCollector | None:
        return next((c for c in self.profile if isinstance(c, SchedulerCollector)), None)

    def context(self, now: datetime | None = None) -> HostContext:
        if self.caps is None:
            raise RunFleetError(f"{self.name}: not detected yet")
        return HostContext(host=self.cfg, general=self.general, caps=self.caps, now=now or self.clock.now())

    # ------------------------------------------------------------ cadence arithmetic
    def _cadences(self, ctx: HostContext) -> dict[str, list[Cadence]]:
        out: dict[str, list[Cadence]] = {}
        for c in self.profile:
            try:
                out[c.name] = list(c.cadences(ctx))
            except Exception as e:
                self._internal_error(e, collector=c.name)
                out[c.name] = []
        return out

    def _host_factor(self) -> float:
        workstation = self.caps is not None and "slurm" not in self.caps.profile
        return 1 + (self.n_hosts - 10) / 20 if workstation and self.n_hosts > 10 else 1.0

    def effective_period(self, c: Cadence, *, with_backoff: bool = True) -> float:
        """interval × visibility × multiplier × host-count factor, clamped to the floor, plus group backoff."""
        vis = {Visibility.HIDDEN: c.hidden_factor, Visibility.FOCUSED: c.focused_factor}.get(self._visibility, 1.0)
        p = max(c.interval * vis * self._multiplier * self._host_factor(), c.floor)
        if self.controller.factor > 1 and c.group.startswith("slurm."):
            p *= self.controller.factor           # busy controller (§5.4): the whole Slurm cadence slows down
        k = self._group_fail.get(c.group, 0) if with_backoff else 0
        if k:
            p = min(p * 2 ** min(k, 16), max(c.max_backoff, p))    # capped exponent: no float overflow
        return p

    def tick_period(self, cadences: Mapping[str, list[Cadence]] | None = None) -> float:
        if cadences is None:
            cadences = self._cadences(self.context()) if self.caps is not None else {}
        periods = [self.effective_period(c) for cs in cadences.values() for c in cs if c.mode == "interval"]
        if not periods:   # no interval group: piggyback groups pace themselves
            periods = [self.effective_period(c) for cs in cadences.values() for c in cs if c.mode == "piggyback"]
        return min(periods) if periods else DEFAULT_TICK

    def _fast_groups(self) -> set[str]:
        if self.caps is None:
            return set()
        cad = self._cadences(self.context())
        t = self.tick_period(cad)
        return {c.group for cs in cad.values() for c in cs
                if c.mode != "event" and self.effective_period(c, with_backoff=False) <= 2 * t}

    def _due(self, cadences: Mapping[str, list[Cadence]], now_m: float, tick: float,
             *, force_all: bool = False) -> dict[str, set[str]]:
        """collector name -> due groups at this tick (interval/piggyback: period elapsed ± half a tick)."""
        due: dict[str, set[str]] = {}
        anchor = False
        for name, cs in cadences.items():
            for c in cs:
                if c.mode == "event":
                    hit = c.group in self._events or c.group in self._forced
                else:
                    last = self._last_run.get(c.group)
                    # a collector follow-up may name an interval group too (`slurm.nodes` once the partitions are
                    # known, an early `process.ps`): it is served at the next tick like an event group
                    hit = (force_all or c.group in self._forced or c.group in self._events or last is None
                           or now_m - last >= self.effective_period(c) - tick / 2)
                    anchor = anchor or (hit and c.mode == "interval")
                anchor = anchor or (hit and c.group in self._forced)     # explicit requests never wait
                if hit:
                    due.setdefault(name, set()).add(c.group)
        if not anchor:     # piggyback groups never ride alone; event groups wait for the next tick
            has_interval = any(c.mode == "interval" for cs in cadences.values() for c in cs)
            due = {n: {g for g in gs if not self._is_mode(cadences, g, "piggyback")} for n, gs in due.items()}
            if has_interval:
                due = {n: {g for g in gs if not self._is_mode(cadences, g, "event")} for n, gs in due.items()}
            due = {n: gs for n, gs in due.items() if gs}
        return due

    @staticmethod
    def _is_mode(cadences: Mapping[str, list[Cadence]], group: str, mode: str) -> bool:
        return any(c.group == group and c.mode == mode for cs in cadences.values() for c in cs)

    def _reschedule(self) -> None:
        """Visibility / multiplier changed: re-anchor so the new tick period applies from the last tick."""
        if self._last_tick_at is None or self.caps is None:
            return
        new_t = self.tick_period()
        if self._consecutive > 0 or self._auth_paused or self._idle_paused:
            # transport backoff / paused: the deadline set by _on_transport_failure stands (re-gridding from the
            # last good tick would retry an unreachable host at once on every focus or visibility change)
            self._publish(tick_interval=new_t)
            return
        self._anchor = None
        self._next_at = min(self._next_at, self._last_tick_at + new_t)
        self._publish(tick_interval=new_t, next_poll_at=self._wall(self._next_at))
        self._event.set()

    def _schedule_next(self, now_m: float, tick: float | None = None, *, anchor: bool = False) -> None:
        t = tick if tick is not None else self.tick_period()
        if anchor or self._anchor is None or t != self._anchor_t:
            self._anchor, self._anchor_k, self._anchor_t = now_m, 0, t
        self._anchor_k += 1
        target = self._anchor + (self._anchor_k + self.rng.uniform(0.0, 0.2)) * t
        if target <= now_m:                       # fell behind (slow probe): never catch up in a burst
            self._anchor, self._anchor_k = now_m, 1
            target = now_m + (1 + self.rng.uniform(0.0, 0.2)) * t
        self._next_at = target
        self._publish(tick_interval=t, next_poll_at=self._wall(target))

    def _wall(self, mono: float) -> datetime:
        return self.clock.now() + timedelta(seconds=max(0.0, mono - self.clock.monotonic()))

    # ------------------------------------------------------------ one tick
    def _plan(self, due: Mapping[str, set[str]], ctx: HostContext) -> _GroupPlan:
        plan = _GroupPlan()
        for c in self.profile:
            groups = due.get(c.name, set())
            if not groups:
                continue
            plan.owner_groups[c.name] = set(groups)
            for g in sorted(groups):
                try:
                    secs = c.sections(frozenset({g}), ctx)
                except Exception as e:
                    self._internal_error(e, collector=c.name)
                    continue
                for s in secs:
                    if s.name in plan.group_of:
                        continue
                    if s.owner != c.name:
                        log.warning("%s: section %s returned by %s ignored (owner mismatch)", self.name, s.name,
                                    c.name)
                        continue
                    plan.group_of[s.name] = g
                    plan.sections.append(s)
        return plan

    async def poll_tick(self, *, anchor: bool = False, force_all: bool = False) -> None:
        """Collect due groups, run one probe, route, merge, apply, update status and schedule the next tick."""
        now_m, now = self.clock.monotonic(), self.clock.now()
        ctx = self.context(now)
        cadences = self._cadences(ctx)
        tick = self.tick_period(cadences)
        due = self._due(cadences, now_m, tick, force_all=force_all)
        forced, self._forced = self._forced, set()
        self._last_tick_at = now_m
        if not due:
            self._schedule_next(now_m, tick, anchor=anchor)
            return
        served = {g for gs in due.values() for g in gs}
        plan = self._plan(due, ctx)
        self._unplanned = served - set(plan.group_of.values())
        results: Mapping[str, SectionResult] = {}
        received = now
        if plan.sections:
            res = await self._run_probe(framing.make_script(plan.sections, "poll"))
            received = self.clock.now()
            if res.error is not None:
                self._forced |= forced                         # due groups, events and forced groups stay due
                await self._on_transport_failure(res.error)
                return
            results = res.sections
            self._on_transport_ok(res)
            if self.controller.observe(results):
                log.info("%s: Slurm controller %s — Slurm cadence ×%g", self.name,
                         "busy" if self.controller.factor > 1 else "healthy again", self.controller.factor)
                self._publish(controller_slowdown=self.controller.factor)
        self._events -= served
        for g in served:
            self._last_run[g] = now_m
        self._ingest(results, plan, ctx, probed=bool(plan.sections), received=received)
        self._schedule_next(self.clock.monotonic(), self.tick_period(cadences), anchor=anchor)

    async def _run_probe(self, script: ProbeScript) -> ProbeResult:
        timeout = framing.local_timeout(script)
        self._publish(in_flight=True, last_attempt=self.clock.now())
        self._last_probe_at = self.clock.monotonic()
        try:
            async with self.lanes.poll():
                return await self.transport.probe(script, timeout=timeout)
        finally:
            self._publish(in_flight=False)

    def _ingest(self, results: Mapping[str, SectionResult], plan: _GroupPlan, ctx: HostContext, *,
                probed: bool = True, received: datetime | None = None) -> None:
        """`ctx.now` (taken before the probe) is the sample time; `received` (after it) dates last_ok, so a slow
        successful probe does not look stale the moment it lands."""
        by_owner: dict[str, dict[str, SectionResult]] = {}
        for name, r in results.items():
            by_owner.setdefault(name.split(".", 1)[0], {})[name] = r
        outputs: list[CollectorOutput] = []
        internal: list[HostError] = []
        monitors = [c for c in self.profile if c.role is Role.MONITOR]
        schedulers = [c for c in self.profile if c.role is not Role.MONITOR]
        for c in monitors + schedulers:      # §3.6: monitors first (profile order), scheduler last
            try:
                out = c.ingest(by_owner.get(c.name, {}), ctx, self.monitor_view)
            except Exception as e:
                internal.extend(self._internal_error(e, collector=c.name, sections=[
                    s.name for s in plan.sections if s.owner == c.name], publish=False))
                self._back_off(plan.owner_groups.get(c.name, set()))
                continue
            if c.role is Role.MONITOR:
                self.monitor_view = update_monitor_view(self.monitor_view, out, ctx.now)
            outputs.append(out)
        merged = merge_outputs(outputs)
        merged = self._post_process(merged, ctx.now)
        self._events |= set(merged.followups)
        self._maybe_redetect(results)
        self.store.apply(self.name, merged, ctx.now)
        if merged.caps_patch and self.caps is not None:
            snap = self.store.hosts[self.name].snapshot
            if snap is not None and snap.caps is not None:
                self.caps = snap.caps
                if self.cache is not None:
                    self.cache.put_caps(self.name, snap.caps, self.caps_at or ctx.now)
        self._update_sections(results, plan, list(merged.errors) + internal, received or ctx.now, probed=probed)
        if self.cache is not None:
            snap = self.store.hosts[self.name].snapshot
            if snap is not None:
                self.cache.put_snapshot(self.name, snap)

    def _maybe_redetect(self, results: Mapping[str, SectionResult]) -> None:
        """§9 row 9: the scheduler's own command went missing -> re-detect once (maybe a different profile)."""
        sched = self.scheduler
        if sched is None or self._redetected_for_missing:
            return
        if any(r.outcome is SectionOutcome.MISSING for n, r in results.items() if n.split(".", 1)[0] == sched.name):
            self._redetected_for_missing = True
            self._redetect = True

    def _post_process(self, out: CollectorOutput, now: datetime) -> CollectorOutput:
        """Core-side persistence the sans-IO collectors cannot do: jobmeta for Slurm history rows (§5.2.4) and
        ended workstation tasks across restarts (§5.3.2 step 8)."""
        if out.jobs is None:
            return out
        jobs = dict(out.jobs)
        t = now.timestamp()
        fresh_meta: list[JobMeta] = []
        for jid, job in jobs.items():
            if job.kind is JobKind.SLURM:
                prev = self._jobmeta.get(jid)
                if job.source == "squeue":
                    meta = JobMeta.from_job(job, t, prev)
                    if meta.worth_keeping() and (prev is None or dataclasses.replace(prev, last_seen=t) != meta
                                                 or t - self._jobmeta_persisted.get(jid, t) > JOBMETA_TOUCH):
                        fresh_meta.append(meta)
                        self._jobmeta_persisted[jid] = t
                    if meta.worth_keeping():
                        self._jobmeta[jid] = meta
                elif prev is not None and job.source in ("sacct", "vanished", "cache"):
                    jobs[jid] = enrich_from_meta(job, prev)
            elif job.kind is JobKind.TASK and job.state.is_terminal and jid not in self._persisted_ended:
                self._persisted_ended.add(jid)
                self._ended_tasks[jid] = job
                if self.cache is not None:
                    self.cache.put_ended_task(self.name, job, job.end_time or now)
        horizon = now - timedelta(seconds=self.general.recent_window)
        for jid, job in list(self._ended_tasks.items()):
            ended = job.end_time or job.observed_at
            if ended is not None and ended < horizon:
                del self._ended_tasks[jid]
            elif jid not in jobs:
                jobs[jid] = dataclasses.replace(job, source="cache")
        if fresh_meta and self.cache is not None:
            self.cache.put_jobmeta(self.name, fresh_meta)
        return dataclasses.replace(out, jobs=jobs)

    def _update_sections(self, results: Mapping[str, SectionResult], plan: _GroupPlan,
                         errors: Sequence[HostError], now: datetime, *, probed: bool = True) -> None:
        prev = dict(self.status.sections)
        by_section = {e.section: e for e in errors if e.section}
        host_errors = [e for e in errors if not e.section]
        failed_groups: set[str] = set()
        for name, r in results.items():
            err = by_section.get(name) or section_error(r, now)
            ok = r.ok and name not in by_section
            old = prev.get(name)
            st = SectionStatus(name=name, outcome=r.outcome.value, ok=ok, rc=r.rc, ms=r.ms,
                               last_ok=now if ok else (old.last_ok if old else None), error=None if ok else err)
            prev[name] = st
            if is_failure(st) and name in plan.group_of:
                failed_groups.add(plan.group_of[name])
        for name, err in by_section.items():      # errors about sections that did not run this tick
            if name not in results and name in prev:
                prev[name] = dataclasses.replace(prev[name], ok=False, error=err)
        ran = {plan.group_of[n] for n in results if n in plan.group_of}
        for g in ran - failed_groups:
            self._group_fail.pop(g, None)
        self._back_off(failed_groups)
        failing = [s for s in prev.values() if is_failure(s)]
        degraded = bool(failing or host_errors)
        error = host_errors[0] if host_errors else (failing[0].error if failing else None)
        changes: dict[str, object] = {"state": ConnState.DEGRADED if degraded else ConnState.OK, "error": error,
                                      "sections": prev}
        if probed:                     # last_ok = last time the host actually answered
            changes["last_ok"] = now
        self._publish(**changes)

    def _back_off(self, groups: Iterable[str]) -> None:
        for g in groups:
            self._group_fail[g] = self._group_fail.get(g, 0) + 1

    # ------------------------------------------------------------ transport outcomes (§4.4)
    def _on_transport_ok(self, res: ProbeResult) -> None:
        self._consecutive = 0
        self._timeouts_in_row = 0
        self._auth_paused = False
        self._publish(latency_s=res.run.elapsed_s, consecutive_failures=0)

    async def _on_transport_failure(self, err: HostError) -> None:
        self._consecutive += 1
        n = self._consecutive
        err = dataclasses.replace(err, at=err.at or self.clock.now())
        state = classify.state_for(err, n)
        if err.kind is ErrorKind.TIMEOUT:
            self._timeouts_in_row += 1
            if self._timeouts_in_row >= 2:
                if self.transport.mux == "managed":
                    await self.transport.reset_master()
                    self._need_prepare = True
                    self._timeouts_in_row = 0
                elif self.transport.mux == "user":
                    err = dataclasses.replace(err, summary="shared ssh master unresponsive — press c to restart it")
        else:
            self._timeouts_in_row = 0
        if err.kind is ErrorKind.MUX:
            if classify.session_refused(err):
                await self.lanes.degrade()
            self._need_prepare = True
        now_m = self.clock.monotonic()
        self._anchor = None
        if classify.pauses_polling(err):
            self._auth_paused = True
            self._need_prepare = True
            self._next_at = float("inf")
            self._publish(state=state, error=err, consecutive_failures=n, next_poll_at=None)
            return
        base = self.tick_period() if self.caps is not None else DEFAULT_TICK
        raw = MAX_BACKOFF if err.kind is ErrorKind.DNS else min(base * 2 ** min(n - 1, 16), MAX_BACKOFF)
        delay = raw * self.rng.uniform(0.8, 1.2)
        self._next_at = now_m + delay
        self._publish(state=state, error=err, consecutive_failures=n, next_poll_at=self._wall(self._next_at))

    def _internal_error(self, e: BaseException, *, collector: str | None = None,
                        sections: Sequence[str] = (), publish: bool = True) -> list[HostError]:
        """§9 row 20: log the traceback with a short id; the host goes DEGRADED (INTERNAL)."""
        ref = secrets.token_hex(2)
        log.error("%s: internal error %s in %s", self.name, ref, collector or "poller", exc_info=e)
        now = self.clock.now()
        summary = f"internal error (log id {ref})"
        errs = [HostError(ErrorKind.INTERNAL, summary, repr(e)[:4096], section=s, at=now) for s in sections]
        if not errs:
            errs = [HostError(ErrorKind.INTERNAL, summary, repr(e)[:4096], at=now)]
        if publish:
            self._publish(state=ConnState.DEGRADED, error=errs[0])
            if self._next_at <= self.clock.monotonic():
                self._next_at = self.clock.monotonic() + self.tick_period_safe()
        return errs

    def tick_period_safe(self) -> float:
        try:
            return self.tick_period()
        except Exception:
            return DEFAULT_TICK

    # ------------------------------------------------------------ status
    def _publish(self, **changes: object) -> None:
        hs = self.store.hosts.get(self.name)
        if hs is None:
            return
        self.store.set_status(self.name, dataclasses.replace(hs.status, **changes))   # type: ignore[arg-type]

    # ------------------------------------------------------------ one-shot collection (snapshot)
    async def collect_once(self) -> HostError | None:
        """prepare + detect + one full poll (+ one round of follow-ups); returns the transport error, if any.

        Capabilities younger than SNAPSHOT_CAPS_MAX_AGE (cache, e.g. written by `doctor`) skip the detect probe, so
        `doctor` + `snapshot` stay within 3 ssh sessions per host. The second round serves the collectors' follow-ups
        and every group that had nothing to do in the first one (sstat, sinfo, logstat once squeue named the jobs)."""
        self._need_prepare = True
        err = await self._prepare_and_detect()
        fresh = self.caps_at is not None and (self.clock.now() - self.caps_at).total_seconds() < SNAPSHOT_CAPS_MAX_AGE
        if err is None and self._redetect and not fresh:
            err = await self._detect()
        self._redetect = False
        if err is not None:
            await self._on_transport_failure(err)
            return err
        await self.poll_tick(force_all=True)
        if self.status.consecutive_failures:
            return self.status.error
        if self._events or self._unplanned:
            self._forced |= self._events | self._unplanned
            await self.poll_tick()
        return None if not self.status.consecutive_failures else self.status.error
