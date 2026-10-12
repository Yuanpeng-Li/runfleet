"""[WP1] AttachService (§7.16): attach policy, consent, auto-refresh timer, autostop, srun row accounting.

Owned by FleetService, which hands it its plumbing as callables (like ActionService): the host's on-demand lane
(`FleetService._on_demand`: master check, lane slot, global semaphore), the scheduler collector and its context, a
caps_patch sink (`HostPoller.apply_caps_patch`: store caps + cache, 24 h like every capability), a poller wake, and
the app's input / focus state. Every AttachState change is announced with `FleetStore.notify_live`.

One entry per job attached this session (kept after stop: live data, consent and srun rows stay readable); at most
one entry is *open* (its GPU/Procs tab visible). The open entry has a runner task that re-evaluates the state on every
poke (store changes of its host, idle pause, focus, a finished sample) and takes the ssh auto samples. Samples run as
tasks of their own (`_Entry.inflight`) so a concurrent `r` for the same mode and node awaits the one in flight instead
of probing twice. An automatic sample re-checks the auto-refresh rules once it holds the lane, right before dispatch
(`_auto_blocked`); a request waiting behind another sample re-checks that the service and its entry still exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, TypeVar

from runfleet.backends.base import HostContext, SchedulerCollector
from runfleet.core.clock import Clock
from runfleet.core.store import ChangeSet, FleetStore
from runfleet.errors import AttachBlocked, AttachConsentRequired
from runfleet.models import (AttachMode, AttachOffer, AttachState, AttachStatus, ConnState, ErrorKind, HostError, Job,
                             JobKey, JobKind, JobLive, JobState)
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult

log = logging.getLogger("runfleet.core.attach")

AUTOSTOP_S = 600.0              # auto-refresh stops after 10 min without user input
NOW_FLOOR_S = 3.0               # attach_now in ssh mode returns a sample younger than this instead of a new one
MAX_FAILURES = 3                # consecutive failed auto samples -> STOPPED ("3 reads failed — press r")
FAILURE_BACKOFF_MAX_S = 60.0    # after a failure the next auto sample waits 2 × attach_refresh, at most this
REFRESH_FLOOR_S = 10.0          # attach_refresh floor (config clamps too; SLURM_FLOORS)
WAKE_GROUPS = frozenset({"slurm.jobs"})   # a sample without data: maybe the job moved or ended -> poll squeue now

READONLY_REASON = "read-only mode: srun adds a permanent sacct row"
AUTOSTOP_REASON = "no input for 10 min — press r"
FAILURES_REASON = f"{MAX_FAILURES} reads failed — press r"
JOB_ENDED_REASON = "job is no longer running"
CLOSED_REASON = "attach stopped (RunFleet is shutting down)"
AUTO_OFF = "auto-refresh is off"
TAB_CLOSED = "tab closed"
ANOTHER_JOB = "another job attached"
BLURRED = "app in background"
IDLE_PAUSED = "polling paused (idle)"
_CONNECTED = (ConnState.OK, ConnState.DEGRADED)
_UNREACHABLE = frozenset({ErrorKind.HOSTKEY, ErrorKind.DNS, ErrorKind.REFUSED})   # ssh hop failures srun gets around
_NODE_SECTION = "slurm.attach.node"   # outer section of node i = AttachPlan.nodes[i] (§7.16.3)

Setting = Literal["ssh", "ssh+srun", "srun", "off"]
ProbeFn = Callable[..., Awaitable[ProbeResult]]   # FleetService._on_demand(host, script, *, recheck=None)
_F = TypeVar("_F", bound="asyncio.Future[Any]")


def srun_hint(host: str) -> str:
    return (f'add attach = "ssh+srun" to [[host]] {host} to read via srun; each read adds one permanent sacct '
            "row")


@dataclass(frozen=True, slots=True)
class _Policy:
    """§7.16.1 applied to one job now (no I/O)."""
    setting: Setting
    refresh_s: float = 15.0
    mode: AttachMode | None = None       # None: blocked
    blocked: str | None = None
    hint: str | None = None
    kind: Literal["", "job", "host", "gone", "config"] = ""   # why blocked: job not running / host not connected /
                                                              # host removed or disabled / policy (sticky)
    ssh: bool = False                    # ssh allowed by the setting and technically possible
    srun: bool = False                   # srun allowed by the setting, technically possible, not read-only
    srun_blocked: str | None = None      # why `r` with mode="srun" is refused
    offer: AttachOffer | None = None     # srun offer (srun mode; ssh mode when ssh landed in another job or could
                                         # not reach the node)


@dataclass(eq=False)
class _Entry:
    key: JobKey
    state: AttachState
    open: bool = False                   # tab visible (attach_start .. attach_stop)
    armed: bool = False                  # reached a readable state since it was opened (job end -> STOPPED)
    node: str | None = None
    auto: bool = True
    consent: Literal["none", "job"] = "none"
    live: JobLive | None = None          # last sample with data
    error: HostError | None = None
    srun_rows: int = 0
    failures: int = 0
    srun_read: bool = False              # an srun sample ran (status IDLE instead of NEEDS_CONSENT)
    halted: str | None = None            # sticky STOPPED reason while open (autostop, failures, job ended)
    stop_reason: str | None = None       # why it was closed
    first_pending: bool = False          # attach_start: sample at once (even with auto off)
    next_m: float | None = None          # next auto sample (monotonic) …
    next_wall: datetime | None = None    # … and as wall clock for AttachState.next_at
    last: JobLive | None = None          # last sample (with or without data), when it ended (monotonic) and the node
    last_m: float | None = None          # selection it read (None: backend default)
    last_node: str | None = None
    streak_broken: bool = False          # the last sample had no data: the next ingest gets no idle_streak
    input_m: float | None = None         # last attach_start / attach_now / attach_set_auto (counts as input)
    landed: tuple[str, str] | None = None   # (node, landed job) when the last ssh sample landed in another job
    unreachable: tuple[str, HostError] | None = None   # (node, error) when the last ssh sample could not reach it
    inflight: asyncio.Task[JobLive] | None = None
    inflight_mode: AttachMode | None = None
    inflight_node: str | None = None
    inflight_auto: bool = False          # an automatic sample (a joining `r` turns it into the user's)
    gone: bool = False                   # dropped (host removed, service closed): nothing may start for it
    runner: asyncio.Task[None] | None = None
    poke: asyncio.Event = field(default_factory=asyncio.Event)


class AttachService:
    def __init__(self, store: FleetStore, clock: Clock, *, readonly: Callable[[], bool],
                 scheduler_for: Callable[[str], SchedulerCollector | None],
                 context_for: Callable[[str], HostContext | None], probe: ProbeFn,
                 apply_caps: Callable[[str, Mapping[str, object]], None],
                 wake: Callable[[str, frozenset[str]], None], last_input: Callable[[], float],
                 focused: Callable[[], bool]) -> None:
        self.store = store
        self.clock = clock
        self._readonly = readonly
        self._scheduler = scheduler_for
        self._context = context_for
        self._probe = probe
        self._apply_caps = apply_caps
        self._wake = wake
        self._last_input = last_input
        self._focused = focused
        self._entries: dict[JobKey, _Entry] = {}
        self._open: _Entry | None = None
        self._tasks: set[asyncio.Future[Any]] = set()
        self._closed = False
        self._unsub: Callable[[], None] | None = store.subscribe(self._on_store)

    # ================================================================ FleetAPI
    def start(self, key: JobKey, *, node: str | None = None) -> AttachState:
        cur = self._open
        if cur is not None and cur.key != key:
            self._close(cur, ANOTHER_JOB)
        e = self._entry(key)
        now_m = self.clock.monotonic()
        e.input_m = now_m
        if e.open and e.node == node:                    # idempotent while the tab stays visible
            self._refresh(e)
            self._ensure_runner(e)
            return e.state
        if node != e.node:                               # another pin: the old node's offers and auto sample go
            e.landed = e.unreachable = None
            if e.inflight is not None and e.inflight_auto and e.inflight_node != node:
                e.inflight.cancel()
                e.inflight = e.inflight_mode = e.inflight_node = None
                e.inflight_auto = False
        e.node, e.open, e.armed = node, True, False
        e.halted = e.stop_reason = None
        e.failures = 0
        refresh = self._refresh_s(key.host)
        if (e.last_node == node and e.last is not None and e.last.mode == "ssh" and e.last_m is not None
                and now_m - e.last_m < refresh):
            e.first_pending = False                      # re-opened right after a sample: keep its schedule
            self._schedule(e, e.last_m + refresh - now_m)
        else:
            e.first_pending = True
        self._open = e
        self._refresh(e)
        self._ensure_runner(e)
        e.poke.set()                                     # a sleeping runner re-reads the selection and its deadline
        return e.state

    def stop(self, key: JobKey | None = None) -> None:
        e = self._open if key is None else self._entries.get(key)
        if e is not None and e.open:
            self._close(e, TAB_CLOSED)

    async def now(self, key: JobKey, *, mode: AttachMode | None = None,
                  consent: Literal["once", "job"] | None = None, node: str | None = None) -> JobLive:
        e = self._entry(key)
        now_m = e.input_m = self.clock.monotonic()
        pin = node or e.node
        pol = self._policy(e, node=pin)
        if pol.blocked is not None:
            raise AttachBlocked(pol.blocked, hint=pol.hint)
        use = mode or pol.mode
        if use == "srun":
            if not pol.srun:
                raise AttachBlocked(pol.srun_blocked or f"srun cannot read this job on {key.host}", hint=pol.hint)
            offer = pol.offer if pol.offer is not None and (node is None or pol.offer.node == node) else None
            if offer is None:
                offer = self._srun_offer(e, node=pin, reason=f"read job {key.id}'s GPUs with srun", lead="")
            if consent is None and e.consent != "job":
                raise AttachConsentRequired(key, offer)
            if consent == "job" and e.consent != "job":
                e.consent = "job"
                self._refresh(e)
            return await self._sample_now(e, "srun", node=offer.node)
        if not pol.ssh:
            raise AttachBlocked(f"ssh cannot read this job on {key.host}", hint=pol.hint)
        if e.open:                                       # `r` re-arms auto-refresh (also after STOPPED)
            e.halted = None
            e.failures = 0
        if e.inflight is None and e.last is not None and e.last.mode == "ssh" and e.last_node == pin \
                and e.last_m is not None and now_m - e.last_m < NOW_FLOOR_S:
            self._refresh(e)
            return e.last
        return await self._sample_now(e, "ssh", node=pin)

    def set_auto(self, key: JobKey, enabled: bool) -> AttachState:
        e = self._entry(key)
        now_m = e.input_m = self.clock.monotonic()
        e.auto = enabled
        if enabled and e.open:
            if e.halted in (AUTOSTOP_REASON, FAILURES_REASON):
                e.halted = None
                e.failures = 0
            if e.next_m is None:
                due = now_m if e.last_m is None else e.last_m + self._refresh_s(key.host)
                self._schedule(e, max(0.0, due - now_m))
        self._refresh(e)
        e.poke.set()
        return e.state

    def state(self, key: JobKey) -> AttachState | None:
        e = self._entries.get(key)
        return e.state if e is not None else None

    def live_for(self, key: JobKey) -> JobLive | None:
        e = self._entries.get(key)
        return e.live if e is not None else None

    # ================================================================ service hooks
    def poke(self) -> None:
        """Re-evaluate the open attach (focus change, read-only toggled)."""
        if self._open is not None:
            self._open.poke.set()

    async def forget_host(self, host: str) -> None:
        """The host leaves the fleet: stop its attach, cancel every sample of it (user ones too), drop the states."""
        tasks: list[asyncio.Future[Any]] = []
        for key, e in list(self._entries.items()):
            if key.host != host:
                continue
            pending: list[asyncio.Future[Any] | None] = [e.inflight, e.runner]
            e.gone = True                                # requests waiting behind a sample must not start one
            if e.open:
                self._close(e, f"{host} was removed")
            for task in pending:
                if task is not None:
                    task.cancel()
                    tasks.append(task)
            del self._entries[key]
        for fut in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await fut

    async def close(self) -> None:
        """Service stop: cancel every runner and sample (the transport kills a cancelled probe's ssh) and wait."""
        self._closed = True                              # first: a request woken by a cancellation must not spawn
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        for e in self._entries.values():
            e.gone = True
            if e.open:
                self._close(e, TAB_CLOSED, notify=False)
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()

    # ================================================================ policy
    def _refresh_s(self, host: str) -> float:
        hs = self.store.hosts.get(host)
        return max(REFRESH_FLOOR_S, float(hs.config.slurm.attach_refresh)) if hs is not None else 15.0

    def _policy(self, e: _Entry, *, node: str | None = None) -> _Policy:
        key = e.key
        host = key.host
        hs = self.store.hosts.get(host)
        if hs is None:
            return _Policy("off", blocked=f"{host} was removed", kind="gone")
        setting: Setting = hs.config.slurm.attach
        refresh = self._refresh_s(host)
        if not hs.config.enabled or hs.status.state is ConnState.DISABLED:
            return _Policy(setting, refresh, blocked=f"{host} is disabled", kind="gone")
        job = self.store.job(key)
        if job is None:
            return _Policy(setting, refresh, blocked="job is no longer listed", kind="job")
        if job.kind is not JobKind.SLURM:
            return _Policy(setting, refresh, blocked="live readings are for running Slurm jobs", kind="config")
        if job.state is not JobState.RUNNING:
            return _Policy(setting, refresh, blocked="job is not running", kind="job")
        if setting == "off":
            return _Policy(setting, refresh, blocked=f"attach is off for {host}", kind="config")
        sched, ctx = self._scheduler(host), self._context(host)
        if hs.status.state not in _CONNECTED or sched is None or ctx is None:
            return _Policy(setting, refresh, blocked=f"{host}: not connected", kind="host")
        try:
            modes = sched.attach_modes(job, ctx)
        except Exception as exc:
            log.exception("%s: attach_modes failed", key)
            return _Policy(setting, refresh, blocked=f"internal error: {exc}"[:80], kind="config")
        denied = ctx.caps.node_ssh == "denied"
        readonly = self._readonly()
        ssh_ok = "ssh" in modes and setting in ("ssh", "ssh+srun")
        srun_blocked: str | None = None
        if setting == "ssh":
            srun_blocked = f'srun is not enabled for {host} (attach = "ssh")'
        elif "srun" not in modes:
            srun_blocked = f"srun cannot read this job on {host}"
        elif readonly:
            srun_blocked = READONLY_REASON
        srun_ok = srun_blocked is None
        hint = srun_hint(host) if setting == "ssh" and denied and "srun" in modes and not readonly else None
        if ssh_ok:
            offer: AttachOffer | None = None
            if srun_ok and e.landed is not None:
                lnode, ljob = e.landed
                offer = self._offer(job, ctx, lnode, f"ssh landed in job {ljob}",
                                    f"ssh can only enter your newest job on {lnode} (job {ljob}). ")
            elif srun_ok and e.unreachable is not None:  # host key unknown, name unresolved, sshd refused
                unode, err = e.unreachable
                offer = self._offer(job, ctx, unode, err.summary, f"ssh from the login host cannot reach {unode} "
                                    f"({err.summary}). ")
            return _Policy(setting, refresh, mode="ssh", ssh=True, srun=srun_ok, srun_blocked=srun_blocked,
                           offer=offer)
        if srun_ok:
            if denied:
                reason, lead = f"{host} refuses ssh to compute nodes", f"{host} does not allow ssh to compute nodes. "
            elif setting == "srun":
                reason, lead = f'{host} reads jobs with srun (attach = "srun")', ""
            else:
                reason, lead = f"ssh cannot read this job on {host}", ""
            try:
                offer = self._offer(job, ctx, node, reason, lead, strict=True)
            except AttachBlocked as exc:
                return _Policy(setting, refresh, blocked=exc.reason, hint=exc.hint, kind="config")
            return _Policy(setting, refresh, mode="srun", srun=True, offer=offer)
        if setting != "ssh" and "srun" in modes and readonly:
            return _Policy(setting, refresh, blocked=READONLY_REASON, kind="config", srun_blocked=srun_blocked)
        if denied:
            return _Policy(setting, refresh, blocked=f"{host} refuses ssh to compute nodes", hint=hint,
                           kind="config", srun_blocked=srun_blocked)
        if setting == "srun":
            return _Policy(setting, refresh, blocked=f"srun cannot read this job on {host}", kind="config",
                           srun_blocked=srun_blocked)
        return _Policy(setting, refresh, blocked=f"{host}: live readings are not available for this job",
                       kind="config", srun_blocked=srun_blocked)

    def _offer(self, job: Job, ctx: HostContext, node: str | None, reason: str, lead: str, *,
               strict: bool = False) -> AttachOffer | None:
        """The srun confirmation texts; the preview is the exact command of the srun plan for `node`."""
        sched = self._scheduler(job.key.host)
        try:
            if sched is None:
                raise AttachBlocked(f"{job.key.host}: not connected")
            plan = sched.plan_attach(job, "srun", ctx, node=node)
        except AttachBlocked:
            if strict:
                raise
            return None
        except Exception as exc:
            log.exception("%s: plan_attach(srun) failed", job.key)
            if strict:
                raise AttachBlocked(f"internal error: {exc}"[:80]) from None
            return None
        target = plan.nodes[0] if plan.nodes else (node or "?")
        jid = job.key.id
        body = (f"{lead}Reading job {jid} runs a one-shot step inside it on {target} (srun --overlap, well under a "
                f"second, at most 1 minute). It leaves one permanent sacct row {plan.job_id_raw}.N rfprobe that "
                "cannot be deleted.")
        return AttachOffer(node=target, reason=reason, title=f"Read job {jid} with srun?", body=body,
                           preview=plan.preview)

    def _srun_offer(self, e: _Entry, *, node: str | None, reason: str, lead: str) -> AttachOffer:
        job, ctx = self.store.job(e.key), self._context(e.key.host)
        if job is None or ctx is None:
            raise AttachBlocked(f"{e.key.host}: not connected")
        offer = self._offer(job, ctx, node, reason, lead, strict=True)
        assert offer is not None
        return offer

    # ================================================================ state machine
    def _entry(self, key: JobKey) -> _Entry:
        e = self._entries.get(key)
        if e is None:
            hs = self.store.hosts.get(key.host)
            setting: Setting = hs.config.slurm.attach if hs is not None else "off"
            e = self._entries[key] = _Entry(key=key, state=AttachState(key=key, status=AttachStatus.BLOCKED,
                                                                       setting=setting))
            self._refresh(e)
        return e

    def _schedule(self, e: _Entry, delay: float | None) -> None:
        if delay is None:
            e.next_m = e.next_wall = None
            return
        e.next_m = self.clock.monotonic() + delay
        e.next_wall = self.clock.now() + timedelta(seconds=delay)

    def _refresh(self, e: _Entry) -> float | Literal["sample"] | None:
        """Recompute and publish e.state; returns what the runner should do: "sample", wait this many seconds, or
        wait for a poke (None)."""
        now_m = self.clock.monotonic()
        pol = self._policy(e, node=e.node)
        if e.open and pol.blocked is not None and e.halted is None:
            if pol.kind == "gone" or (pol.kind == "job" and e.armed):
                e.halted = JOB_ENDED_REASON if pol.kind == "job" else pol.blocked
        status: AttachStatus
        stopped: str | None = None
        next_at: datetime | None = None
        action: float | Literal["sample"] | None = None
        if e.inflight is not None:
            status = AttachStatus.SAMPLING
        elif not e.open:
            if pol.blocked is not None:
                status = AttachStatus.BLOCKED
            elif e.stop_reason is not None:
                status, stopped = AttachStatus.STOPPED, e.stop_reason
            elif pol.mode == "srun":
                status = AttachStatus.IDLE if (e.consent == "job" or e.srun_read) else AttachStatus.NEEDS_CONSENT
            else:
                status = AttachStatus.IDLE
        elif pol.blocked is None and pol.mode == "srun":
            e.armed = True
            status = AttachStatus.IDLE if (e.consent == "job" or e.srun_read) else AttachStatus.NEEDS_CONSENT
        elif pol.blocked is not None and pol.kind == "config":
            status = AttachStatus.BLOCKED
        elif e.halted is not None:
            status, stopped = AttachStatus.STOPPED, e.halted
        elif pol.blocked is not None:
            if pol.kind == "host" and e.armed:
                status, stopped = AttachStatus.PAUSED, pol.blocked
            else:
                status = AttachStatus.BLOCKED
        else:                                            # ssh, tab open
            e.armed = True
            idle_for = now_m - max(self._last_input(), e.input_m if e.input_m is not None else float("-inf"))
            if not (e.auto or e.first_pending):
                status = AttachStatus.IDLE
            elif idle_for >= AUTOSTOP_S:
                e.halted = AUTOSTOP_REASON
                status, stopped = AttachStatus.STOPPED, AUTOSTOP_REASON
            elif not self._focused():
                status, stopped, action = AttachStatus.PAUSED, BLURRED, AUTOSTOP_S - idle_for
            elif self.store.idle_paused:
                status, stopped, action = AttachStatus.PAUSED, IDLE_PAUSED, AUTOSTOP_S - idle_for
            else:
                status = AttachStatus.LIVE if e.auto else AttachStatus.IDLE
                if e.first_pending or e.next_m is None:
                    due, next_at = now_m, self.clock.now()
                else:
                    due, next_at = e.next_m, e.next_wall
                if not e.auto:
                    next_at = None
                action = "sample" if due <= now_m else min(due - now_m, AUTOSTOP_S - idle_for)
        blocked = status is AttachStatus.BLOCKED
        state = AttachState(key=e.key, status=status, setting=pol.setting, mode=pol.mode, live=e.live,
                            error=e.error, auto=e.auto, refresh_s=pol.refresh_s if pol.mode == "ssh" else None,
                            next_at=next_at, consent=e.consent, offer=pol.offer,
                            blocked=pol.blocked if blocked else None, hint=pol.hint if blocked else None,
                            stopped=stopped, srun_rows=e.srun_rows, node=e.node, failures=e.failures)
        self._publish(e, state)
        return action

    def _publish(self, e: _Entry, state: AttachState, *, notify: bool = True) -> None:
        if state == e.state:
            return
        e.state = state
        if notify:
            self.store.notify_live(e.key.host, (e.key.id,))

    def _close(self, e: _Entry, reason: str, *, notify: bool = True) -> None:
        e.open = False
        e.stop_reason = reason
        e.first_pending = False
        self._schedule(e, None)
        if self._open is e:
            self._open = None
        if e.runner is not None:
            e.runner.cancel()
            e.runner = None
        if e.inflight is not None and e.inflight_auto:   # an AUTO sample in flight goes with the tab (`r` stays)
            e.inflight.cancel()
            e.inflight = None
        if notify:
            self._refresh(e)

    def _on_store(self, change: ChangeSet) -> None:
        e = self._open
        if e is None or change.host not in (e.key.host, "*"):
            return
        if change.status or change.paused or change.jobs_changed or change.jobs_removed or change.jobs_added:
            e.poke.set()

    # ================================================================ runner and samples
    def _ensure_runner(self, e: _Entry) -> None:
        if e.runner is None or e.runner.done():
            e.runner = self._track(asyncio.ensure_future(self._run(e)))

    def _track(self, task: _F) -> _F:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, e: _Entry) -> None:
        while e.open:
            e.poke.clear()
            try:
                action = self._refresh(e)
            except Exception:
                log.exception("%s: attach state evaluation failed", e.key)
                action = None
            if action == "sample":
                if self._closed or e.gone:
                    return
                task = self._spawn(e, "ssh", auto=True, node=e.node)
                await asyncio.wait({task})
                continue
            await self._wait(e.poke, action if isinstance(action, float) else None)

    async def _wait(self, poke: asyncio.Event, delay: float | None) -> None:
        waiters: set[asyncio.Future[Any]] = {asyncio.ensure_future(poke.wait())}
        if delay is not None:
            waiters.add(asyncio.ensure_future(self.clock.sleep(max(0.0, delay))))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()

    async def _sample_now(self, e: _Entry, mode: AttachMode, *, node: str | None) -> JobLive:
        """A user sample (`r`): await one of the same mode and node selection in flight (it becomes the user's: no
        auto recheck, the tab closing keeps it), else wait for the other one, then probe — unless the service or
        the entry went away meanwhile."""
        while e.inflight is not None and not e.inflight.done():
            task = e.inflight
            if e.inflight_mode == mode and e.inflight_node == node:
                e.inflight_auto = False
                try:
                    return await self._await(task)
                except _Skipped:                         # dropped at dispatch just before we joined: read anew
                    pass
            else:
                await asyncio.wait({task})
            self._check_alive(e)
        return await self._await(self._spawn(e, mode, auto=False, node=node))

    def _check_alive(self, e: _Entry) -> None:
        """Nothing starts for a closed service or a dropped entry (host removed) — checked after every await."""
        if self._closed or e.gone or self._entries.get(e.key) is not e:
            raise AttachBlocked(CLOSED_REASON if self._closed else f"{e.key.host} was removed")

    async def _await(self, task: asyncio.Task[JobLive]) -> JobLive:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and (current is None or not current.cancelling()):
                raise AttachBlocked(CLOSED_REASON if self._closed else "the read was cancelled (tab closed)") from None
            raise

    def _spawn(self, e: _Entry, mode: AttachMode, *, auto: bool, node: str | None) -> asyncio.Task[JobLive]:
        self._check_alive(e)
        task: asyncio.Task[JobLive] = asyncio.ensure_future(self._sample(e, mode, node))
        e.inflight, e.inflight_mode, e.inflight_node, e.inflight_auto = task, mode, node, auto
        self._track(task)
        task.add_done_callback(lambda t: self._sample_done(e, t, mode, node, auto))
        self._refresh(e)
        return task

    def _sample_done(self, e: _Entry, task: asyncio.Task[JobLive], mode: AttachMode, node: str | None,
                     auto: bool) -> None:
        if e.inflight is task:
            auto = e.inflight_auto                       # a joining `r` made it the user's
            e.inflight = e.inflight_mode = e.inflight_node = None
            e.inflight_auto = False
        if not task.cancelled():
            exc = task.exception()
            if isinstance(exc, _Skipped):                # dropped at dispatch: nothing ran, nothing to record
                log.debug("%s: auto sample dropped before dispatch: %s", e.key, exc)
            elif isinstance(exc, AttachBlocked):
                if auto and e.open:
                    e.halted = exc.reason
            elif exc is not None:                        # a bug outside ingest: count it as a failed read (backoff,
                log.error("%s: attach sample failed", e.key, exc_info=exc)   # never a hot loop)
                now = self.clock.now()
                err = HostError(ErrorKind.INTERNAL, f"internal error: {exc}"[:80], at=now)
                self._record(e, JobLive(key=e.key, mode=mode, sampled_at=now, errors=(err,)), mode, node,
                             transport_ok=False)
        if self._entries.get(e.key) is e:
            self._refresh(e)
        e.poke.set()

    async def _sample(self, e: _Entry, mode: AttachMode, node: str | None) -> JobLive:
        key, host = e.key, e.key.host
        job, sched, ctx = self.store.job(key), self._scheduler(host), self._context(host)
        if job is None:
            raise AttachBlocked("job is no longer listed")
        if sched is None or ctx is None:
            raise AttachBlocked(f"{host}: not connected")
        plan = sched.plan_attach(job, mode, ctx, node=node)      # AttachBlocked propagates
        script: ProbeScript = framing.make_script(plan.sections, "attach")
        skipped: list[str] = []

        def recheck() -> str | None:                             # runs once the lane is held, right before dispatch
            if mode == "srun" and self._readonly():              # read-only switched on while waiting for the lane
                return READONLY_REASON
            mine = e.inflight is asyncio.current_task()
            reason = self._auto_blocked(e) if mine and e.inflight_auto else None
            if reason is not None:
                skipped.append(reason)
            return reason

        res = await self._probe(host, script, recheck=recheck)
        if skipped:
            raise _Skipped(skipped[0])
        after = self._context(host) or dataclasses.replace(ctx, now=self.clock.now())
        previous = e.live
        if previous is not None and e.streak_broken:             # a failed read ends "N samples in a row"; the
            previous = dataclasses.replace(previous, idle_streak={})   # counters stay for the cpu_cores delta
        patch: Mapping[str, object] = {}
        try:
            live, patch = sched.ingest_attach(plan, res.sections, after, previous=previous)
        except Exception as exc:
            log.exception("%s: ingest_attach failed", key)
            live = JobLive(key=key, mode=mode, sampled_at=after.now,
                           errors=(HostError(ErrorKind.INTERNAL, f"internal error: {exc}"[:80], at=after.now),))
        if res.error is not None:
            live = dataclasses.replace(live, errors=(*live.errors, res.error))
        if patch:
            try:
                self._apply_caps(host, patch)
            except Exception:
                log.exception("%s: applying attach caps_patch %r failed", host, patch)
        self._record(e, live, mode, node, transport_ok=res.error is None, nodes=plan.nodes)
        return live

    def _auto_blocked(self, e: _Entry) -> str | None:
        """Why an automatic sample must not start now — the §7.16.2 auto-refresh rules `_refresh` applies, re-checked
        right before dispatch (the lane may have been busy for a while)."""
        if self._closed or e.gone:
            return CLOSED_REASON
        if not e.open or self._open is not e:
            return TAB_CLOSED
        if not (e.auto or e.first_pending):
            return AUTO_OFF
        pol = self._policy(e, node=e.node)
        if pol.blocked is not None:
            return pol.blocked
        if pol.mode != "ssh":
            return f"{e.key.host} reads this job with srun"
        if e.halted is not None:
            return e.halted
        idle_for = self.clock.monotonic() - max(self._last_input(),
                                                e.input_m if e.input_m is not None else float("-inf"))
        if idle_for >= AUTOSTOP_S:
            return AUTOSTOP_REASON
        if not self._focused():
            return BLURRED
        if self.store.idle_paused:
            return IDLE_PAUSED
        return None

    def _record(self, e: _Entry, live: JobLive, mode: AttachMode, node: str | None, *, transport_ok: bool,
                nodes: tuple[str, ...] = ()) -> None:
        now_m = self.clock.monotonic()
        e.srun_rows += live.sacct_rows_added
        e.last, e.last_m, e.last_node = live, now_m, node
        ok = live.has_data
        e.streak_broken = not ok
        if ok:
            e.live, e.error, e.failures = live, None, 0
        else:
            e.error = live.errors[0] if live.errors else HostError(ErrorKind.REMOTE, "no live data in this read",
                                                                   section="slurm.attach", at=live.sampled_at)
            e.failures += 1
        if mode == "srun":
            e.srun_read = True
            return
        landed = next(((n.node, n.landed_job) for n in live.nodes
                       if n.gpus_of_target is False and n.landed_job), None)
        if ok or landed is not None:
            e.landed = landed
        e.unreachable = _unreachable(live, nodes)
        e.first_pending = False
        refresh = self._refresh_s(e.key.host)
        self._schedule(e, refresh if ok else min(2 * refresh, FAILURE_BACKOFF_MAX_S))
        if not ok and e.failures >= MAX_FAILURES and e.open:
            e.halted = FAILURES_REASON
        if not ok and transport_ok:
            self._wake(e.key.host, WAKE_GROUPS)


class _Skipped(Exception):
    """An automatic sample dropped at dispatch (`_auto_blocked`): not a failed read, nothing is recorded."""


def _unreachable(live: JobLive, nodes: tuple[str, ...]) -> tuple[str, HostError] | None:
    """(node, error) of the first planned node the ssh hop could not reach (unknown host key, name not resolved,
    sshd refused) — srun gets there through slurmd, so the srun offer appears (§7.16.6)."""
    for err in live.errors:
        if err.kind in _UNREACHABLE and err.section and err.section.startswith(_NODE_SECTION):
            idx = err.section[len(_NODE_SECTION):]
            if idx.isdigit() and int(idx) < len(nodes):
                return nodes[int(idx)], err
    return None
