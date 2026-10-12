"""[WP1] FROZEN FleetAPI protocol + FleetService (DESIGN.md §3.7)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import asyncio
import contextlib
import dataclasses
import logging
import os
import random
import re
import tomllib
from typing import TYPE_CHECKING, Any, Literal

from runfleet.backends.base import ActionPlan, ActionSpec, HostContext, SchedulerCollector
from runfleet.config import FleetConfig, GeneralSettings, HostConfig, NotifySettings, host_name_for
from runfleet.core.cache import Cache, JobMeta
from runfleet.core.clock import Clock
from runfleet.core.store import FleetStore
from runfleet.errors import ConfigError, RunFleetError
from runfleet.models import (ActionResult, AttachMode, AttachState, Capabilities, ConnState,
                             DetectReport, DoctorReport, ErrorKind, HostError, Job, JobDetail, JobKey, JobLive, LogRef,
                             Visibility)
from runfleet.probe.section import SectionResult
from runfleet.transport.base import ProbeResult, Transport

if TYPE_CHECKING:
    from runfleet.core.actions import ActionService
    from runfleet.core.forwards import PortForward
    from runfleet.core.logs import LogService
    from runfleet.core.poller import BackendHooks, HostLanes, HostPoller
    from runfleet.probe.section import ProbeScript

log = logging.getLogger("runfleet.core.service")

TransportFactory = Callable[[HostConfig], Transport]     # cli.py picks OpenSSH / Local / Demo per host

@dataclass(frozen=True, slots=True)
class LogUpdate:
    lines: tuple[str, ...] = ()  # newly completed lines (ANSI kept, CR-collapsed, each <= 16 KiB)
    live: str | None = None      # current unterminated line after its last "\r" (tqdm); replaces the previous one
    note: str | None = None      # "waiting for …/x.out to appear", "stream ended", "permission denied"
    eof: bool = False

class LogSession(Protocol):
    host: str
    ref: LogRef
    def updates(self) -> AsyncIterator[LogUpdate]: ...   # initial tail, then follow; batched <= 10/s
    async def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class LogPeek:
    lines: tuple[str, ...]       # last <= 8 complete lines (live line appended if any)
    size: int | None
    mtime: datetime | None
    fetched_at: datetime
    error: str | None = None

class FleetAPI(Protocol):
    store: FleetStore
    config: FleetConfig
    readonly: bool
    notices: list[str]           # append-only one-shot warnings the UI toasts once each (config warnings from cli,
                                 # corrupt cache moved aside, "another RunFleet (pid N) is polling", §7.7/§8.3)
    config_path: Path | None     # the file Settings edits (cli: -c PATH or the default path, even when the run shows
                                 # ad-hoc hosts); None = in memory only (--demo: nothing is ever written)

    # lifecycle
    async def start(self) -> None: ...           # load cache -> store; start one HostPoller per enabled host
    async def stop(self) -> None: ...
    async def collect_once(self, hosts: Iterable[str] | None = None, *, timeout: float = 60) -> None: ...
        # `snapshot`: prepare + detect + one full poll per host, no pollers left running
    # scheduling hints from the UI
    def set_visibility(self, levels: Mapping[str, Visibility]) -> None: ...
    def note_user_activity(self) -> None: ...    # every key/mouse event: idle throttle reset, idle pause resume
    def set_app_focus(self, focused: bool) -> None: ...   # AppFocus / AppBlur -> x4 when blurred
    def refresh(self, hosts: Iterable[str] | None = None) -> None: ...   # manual; 3 s per-host floor
    def set_rate_multiplier(self, m: float) -> None: ...   # global +/- (0.5 ... 4.0)
    def set_host_enabled(self, host: str, enabled: bool) -> None: ...
    # details and logs
    async def job_detail(self, key: JobKey, *, max_age_s: float = 30) -> JobDetail: ...
    async def peek_log(self, host: str, ref: LogRef, *, nbytes: int = 4096) -> LogPeek: ...
        # preview pane; cached per (host, path); re-fetched only if >= 15 s old and logstat size/mtime changed
    async def stream_log(self, host: str, ref: LogRef, *, tail_bytes: int = 65536) -> LogSession: ...
        # JobDetail log pane and LogScreen only; opening one beyond the host's stream lane closes the oldest
    async def fetch_log(self, host: str, ref: LogRef, *, nbytes: int = 4 << 20) -> Path: ...
        # last nbytes into a private temp file (local $PAGER fallback)
    # actions
    def actions_for(self, keys: Sequence[JobKey]) -> list[ActionSpec]: ...
        # merged per action id across hosts; disabled_reason precedence: read-only mode -> host not OK/DEGRADED
        # ("lynx: needs auth") -> stale data ("data is 4m old — refresh first") -> backend reason
    def plan(self, action_id: str, keys: Sequence[JobKey],
             params: Mapping[str, object] | None = None) -> list[ActionPlan]: ...
        # one plan per host with eligible targets; pure (no I/O); raises errors.ActionError if nothing is eligible
    async def execute(self, plans: Sequence[ActionPlan]) -> list[ActionResult]: ...
        # never raises; then wakes the affected hosts ("slurm.jobs", "slurm.history" / "process.ps")
    def action_log(self) -> Sequence[ActionResult]: ...   # last 200 results of this session
    # interactive argv; the UI runs them inside App.suspend
    async def prepare_connect(self, host: str) -> bool: ...  # resolve + clear stale socket; True = already connected
    def connect_argv(self, host: str) -> list[str] | None: ...   # call prepare_connect first
    async def after_connect(self, host: str, rc: int) -> None: ...   # reset backoff, re-prepare, poll now
    def shell_argv(self, host: str, key: JobKey | None = None, node: str | None = None) -> list[str]: ...
    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None: ...
        # `ssh -t <alias> less -R +G -- <path>` if `less` is in caps.tools, else None (use fetch_log + local pager)
    def local_pager_argv(self, path: Path | None = None) -> list[str]: ...
        # general.pager / $PAGER / `less -R +G` on a local file; None = runfleet.log (palette "RunFleet log")
    # hosts import and diagnostics (aliases need not be in the config)
    async def probe_alias(self, alias: str) -> DetectReport: ...
    def write_config(self, hosts: Sequence[HostConfig], path: Path | None = None) -> Path: ...
        # new file: dump_hosts_toml; existing file: appends [[host]] tables for names not yet present, never rewrites
    async def doctor(self, host: str, *, reset: bool = False) -> DoctorReport: ...
    # runtime host management and settings (Settings overlay §7.13, auto-remember §8.4); `save=False` changes only
    # this run. Every save is one comment-preserving, locked read-modify-write of config_path (configedit); a broken
    # or unwritable file is never rewritten: ConfigError is raised and nothing changes, neither the file nor the
    # running fleet.
    def ssh_aliases(self) -> list[str]: ...           # ~/.ssh/config Host aliases (demo: the simulated fleet's)
    def saved_hosts(self) -> list[HostConfig]: ...    # hosts in config_path, file order; raises ConfigError
    def custom_host_keys(self, host: str) -> list[str]: ...
        # keys of the saved [[host]] table beyond name/ssh (hand-tuned settings; [] in memory / unreadable file)
    def add_host(self, cfg: HostConfig, *, save: bool = True) -> HostConfig: ...
        # save a minimal [[host]] table (unless present; a table removed earlier this session comes back whole) and,
        # if not in the store yet, add it there and start its poller at once; returns the running HostConfig
        # ([defaults.*] of the file applied)
    async def remove_host(self, host: str, *, save: bool = True) -> None: ...
        # unsave, and if running: cancel its poller, close its log streams and transport, drop it from the store
    def reorder_hosts(self, names: Sequence[str], *, save: bool = True) -> None: ...
        # store order (= card order and scope numbers) and the order of the [[host]] tables
    def set_general(self, *, save: bool = True, **changes: Any) -> GeneralSettings: ...
        # [general] keys; applied live where the core can (readonly, refresh_multiplier, idle_*, pager)
    def set_notify(self, *, save: bool = True, **changes: Any) -> NotifySettings: ...
        # [notify] keys; live (the UI's Alerts reads config.notify on every batch)
    def remember_hosts(self, names: Sequence[str]) -> None: ...
        # hosts named on the command line: saved once they answer (or need auth) while general.remember_hosts,
        # with one notice per batch ("saved k2, vega — manage in Settings (,)")
    # everyday actions (copy menu, port forwards); nothing here runs a remote command
    def job_shell_command(self, host: str, key: JobKey | None = None, node: str | None = None) -> str | None: ...
        # the remote command shell_argv runs for this job (`srun --jobid … --pty` on its node, `cd WorkDir …`)
    def forward_blocker(self, host: str) -> str | None: ...
        # why `host` cannot forward a port now ("needs auth — press c", no usable master), None if it can
    async def forward_port(self, host: str, node: str, remote_port: int, *, job: JobKey | None = None,
                           scheme: str = "http", path: str = "", label: str = "") -> PortForward: ...
        # `ssh -O forward -L 127.0.0.1:<lport>:<node>:<rport>` on the host's master; lport = rport when free here;
        # the same (host, node, rport) again returns the existing forward; raises RunFleetError
    async def cancel_forward(self, local_port: int) -> None: ...   # `-O cancel`; stop() cancels every forward
        # raises RunFleetError when the master answered but did not cancel (the forward stays listed)
    def forwards(self) -> list[PortForward]: ...
    # Attach (§7.16): live per-GPU / per-process / cgroup readings of ONE running Slurm job, read from its compute
    # node only while JobDetail's GPU or Procs tab shows it. Every state change notifies store subscribers with
    # ChangeSet(host, live={job id}); the UI then re-reads attach_state(key). Never touches the background poll.
    def attach_start(self, key: JobKey, *, node: str | None = None) -> AttachState: ...
        # the GPU/Procs tab of `key` became visible (idempotent while it stays visible). Stops any other attach
        # ("another job attached"). No I/O itself: ssh mode schedules the first sample at once (then every
        # attach_refresh while the rules of §7.16 hold); srun mode never samples here (status NEEDS_CONSENT or
        # IDLE until attach_now). Returns the state right after the call.
    def attach_stop(self, key: JobKey | None = None) -> None: ...
        # tab hidden / screen closed (None = whatever is attached). Cancels the timer and an in-flight AUTO sample;
        # the state (live data, consent, srun_rows) stays readable as STOPPED ("tab closed")
    async def attach_now(self, key: JobKey, *, mode: AttachMode | None = None,
                         consent: Literal["once", "job"] | None = None, node: str | None = None) -> JobLive: ...
        # `r`: one sample now (mode None = the state's mode; "srun" explicitly = the "read with srun" offer).
        # Raises errors.AttachConsentRequired (srun without consent: show e.offer, call again with consent),
        # errors.AttachBlocked (reason [+ hint]). Remote failures do not raise: the JobLive carries errors and
        # has no nodes. ssh: re-arms auto-refresh if the tab is open; a sample < 3 s old is returned as is; an
        # in-flight sample of the same job is awaited, not duplicated
    def attach_set_auto(self, key: JobKey, enabled: bool) -> AttachState: ...
        # `a`: ssh auto-refresh on/off for this job (session only); srun ignores it (never auto-refreshes)
    def attach_state(self, key: JobKey) -> AttachState | None: ...
        # None = never attached this session; otherwise the latest state (also after stop)

class FleetService(FleetAPI):
    """The core facade: one HostPoller task per enabled host, plus on-demand details, logs, actions and doctor.

    Construction does no I/O. `hooks` (WP2 entry points) and `rng` may be replaced before `start()` /
    `collect_once()` (tests use stub collectors). `notices` collects one-shot warnings for the UI (corrupt cache,
    another instance polling, config warnings added by the CLI).
    """

    REFRESH_FLOOR = 3.0
    FLUSH_EVERY = 10.0

    def __init__(self, config: FleetConfig, transports: TransportFactory, clock: Clock,
                 cache: Cache | None) -> None:
        # Deferred imports: poller/logs/actions import this module's LogUpdate/LogPeek types.
        from runfleet.core.actions import ActionService
        from runfleet.core.attach import AttachService
        from runfleet.core.logs import LogService
        from runfleet.core.poller import BackendHooks

        self.config = config
        self.readonly = config.general.readonly
        self.store = FleetStore(config.hosts)
        self.clock = clock
        self.cache = cache
        self.hooks: BackendHooks = BackendHooks.default()
        self.rng = random.Random()
        self.notices: list[str] = list(cache.warnings) if cache is not None else []
        # Per-host instance locks: <lock_path>.<host key> (default <mux_dir>/instance.lock.*); False: none (demo).
        self.lock_path: Path | None | Literal[False] = None
        self._factory = transports
        self._global_sem = asyncio.Semaphore(max(1, config.general.max_parallel_ssh))
        self._transports: dict[str, Transport] = {}
        self._lanes: dict[str, HostLanes] = {}
        self._pollers: dict[str, HostPoller] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._visibility: dict[str, Visibility] = {}
        self._rate = min(4.0, max(0.5, config.general.refresh_multiplier))
        self._focused = True
        self._idle_throttled = False
        self._last_input = clock.monotonic()
        self._activity = asyncio.Event()
        self._last_refresh: dict[str, float] = {}
        self._details: dict[JobKey, tuple[float, JobDetail]] = {}
        self._cached_caps: dict[str, tuple[Capabilities, datetime | None]] = {}
        self._jobmeta: dict[str, dict[str, JobMeta]] = {}
        self._ended: dict[str, dict[str, Job]] = {}
        self._bg: list[asyncio.Task[None]] = []
        self._lock_fds: list[int] = []
        self._host_locks: dict[str, int] = {}
        self._started = False
        # Settings: where saves go (None = in memory, the list below), where alias candidates come from
        self.config_path: Path | None = config.source
        self.alias_source: Callable[[], list[str]] = _list_aliases
        self._memory_hosts: list[HostConfig] = list(config.hosts)
        self._removed: dict[str, str] = {}      # name -> its [[host]] text, restored if checked again this session
        self._removing: dict[str, asyncio.Task[None]] = {}   # name -> its running teardown (remove_host)
        self._remember: dict[str, HostConfig] = {}
        self._remembered: list[str] = []
        self._remember_order: list[str] = []
        self._remember_unsub: Callable[[], None] | None = None
        self._forwards: dict[int, PortForward] = {}   # local port -> active forward (cancelled by stop())
        self._forward_ops: set[asyncio.Task[Any]] = set()   # `-O forward|cancel` in flight (stop() waits for them)
        self.logs: LogService = LogService(
            clock, transport_for=self._transport, stream_limit_for=lambda h: self._lanes_for(h).stream_limit,
            probe=self._on_demand, caps_for=self._caps, pager=config.general.pager,
            ensure_master=self._ensure_master)
        self.actions: ActionService = ActionService(
            self.store, clock, readonly=self.readonly, scheduler_for=self._scheduler, context_for=self._context,
            probe=self._on_demand, wake=self._wake)
        self.attach: AttachService = AttachService(
            self.store, clock, readonly=lambda: self.readonly, scheduler_for=self._scheduler,
            context_for=self._context, probe=self._on_demand, apply_caps=self._apply_caps_patch, wake=self._wake,
            last_input=lambda: self._last_input, focused=lambda: self._focused)

    # ================================================================ plumbing
    def _host_cfg(self, host: str) -> HostConfig:
        hs = self.store.hosts.get(host)
        if hs is None:
            raise RunFleetError(f"unknown host {host!r}")
        return hs.config

    def _transport(self, host: str) -> Transport:
        t = self._transports.get(host)
        if t is None:
            t = self._transports[host] = self._factory(self._host_cfg(host))
        return t

    def _lanes_for(self, host: str) -> HostLanes:
        from runfleet.core.poller import HostLanes
        lanes = self._lanes.get(host)
        if lanes is None:
            lanes = self._lanes[host] = HostLanes(self._global_sem, self._transport(host).mux)
        return lanes

    def _poller(self, host: str) -> HostPoller:
        from runfleet.core.poller import HostPoller
        p = self._pollers.get(host)
        if p is None:
            caps, caps_at = self._cached_caps.get(host, (None, None))
            p = HostPoller(self._host_cfg(host), self._transport(host), self.store, self.clock, self.rng,
                           self._lanes_for(host), general=self.config.general, hooks=self.hooks, cache=self.cache,
                           n_hosts=len(self.store.hosts), caps=caps, caps_at=caps_at,
                           jobmeta=self._jobmeta.get(host), ended_tasks=self._ended.get(host))
            p.set_visibility(self._visibility.get(host, Visibility.VISIBLE))
            p.set_multiplier(self._effective_multiplier())
            self._pollers[host] = p
        return p

    def _caps(self, host: str) -> Capabilities | None:
        p = self._pollers.get(host)
        if p is not None and p.caps is not None:
            return p.caps
        snap = self.store.hosts[host].snapshot if host in self.store.hosts else None
        return snap.caps if snap is not None else None

    def _scheduler(self, host: str) -> SchedulerCollector | None:
        p = self._pollers.get(host)
        return p.scheduler if p is not None else None

    def _context(self, host: str) -> HostContext | None:
        p = self._pollers.get(host)
        if p is None or p.caps is None:
            return None
        return p.context()

    async def _ensure_master(self, host: str) -> HostError | None:
        """Revive a gone shared/managed master (ControlPersist expired, the user closed theirs) before an on-demand
        probe or a log stream; else ssh with ControlMaster=no falls back to a full BatchMode login (an extra cold
        login through the jump host, or an instant auth failure on Duo hosts)."""
        t = self._transport(host)
        if t.mux in ("user", "managed") and not await t.check_master():
            async with self._global_sem:          # a cold login counts against max_parallel_ssh (never nested)
                return await t.prepare()
        return None

    async def _on_demand(self, host: str, script: ProbeScript, *,
                         recheck: Callable[[], str | None] | None = None) -> ProbeResult:
        """On-demand probe (detail, action, log peek/fetch, doctor): master check, then the host's on-demand lane.
        `recheck` runs once the lane is held, right before dispatch; a reason it returns cancels the probe."""
        from runfleet.probe import framing
        from runfleet.transport import classify
        from runfleet.transport.spawn import failed_probe

        t = self._transport(host)
        lanes = self._lanes_for(host)
        err = await self._ensure_master(host)
        if err is not None:
            return failed_probe(script, err)
        async with lanes.demand():
            reason = recheck() if recheck is not None else None
            if reason is not None:
                return failed_probe(script, HostError(ErrorKind.PERMISSION, reason[:80]))
            res = await t.probe(script, timeout=framing.local_timeout(script))
        if res.error is not None and res.error.kind is ErrorKind.MUX and classify.session_refused(res.error):
            await lanes.degrade()
        return res

    def _apply_caps_patch(self, host: str, patch: Mapping[str, object]) -> None:
        """A capability learned on demand (Attach: node_ssh) goes through the host's poller: store caps + cache."""
        p = self._pollers.get(host)
        if p is not None:
            p.apply_caps_patch(patch)

    def _wake(self, host: str, groups: frozenset[str]) -> None:
        p = self._pollers.get(host)
        if p is not None and host in self._tasks:
            p.wake("action", groups)

    def _effective_multiplier(self) -> float:
        return self._rate * (4.0 if (self._idle_throttled or not self._focused) else 1.0)

    def _apply_multiplier(self) -> None:
        m = self._effective_multiplier()
        for p in self._pollers.values():
            p.set_multiplier(m)

    # ================================================================ lifecycle
    async def start(self) -> None:           # load cache -> store; start one HostPoller per enabled host
        if self._started:
            return
        self._started = True
        await self._load_cache()
        self._take_instance_lock()
        for name, hs in self.store.hosts.items():
            if hs.config.enabled:
                self._start_poller(name)
        self._bg.append(asyncio.ensure_future(self._idle_watch()))
        if self.cache is not None:
            self._bg.append(asyncio.ensure_future(self._flush_loop()))

    def _start_poller(self, host: str) -> None:
        if host in self._tasks:
            return
        p = self._poller(host)
        self._tasks[host] = asyncio.ensure_future(p.run())

    async def _load_cache(self, *, snapshots: bool = True) -> None:
        """Cache -> store (snapshots, only for the TUI: a `snapshot` must never print cached rows as current), caps,
        jobmeta, ended tasks."""
        if self.cache is None:
            return
        cache, now = self.cache, self.clock.now()
        from datetime import timedelta

        def load() -> tuple[Any, ...]:
            cache.prune(now)
            snaps = cache.load_snapshots()
            caps = {h: cache.load_caps(h, now) for h in self.store.hosts}
            meta = {h: cache.load_jobmeta(h) for h in self.store.hosts}
            since = now - timedelta(seconds=self.config.general.recent_window)
            ended = {h: cache.load_ended_tasks(h, since) for h in self.store.hosts}
            return snaps, caps, meta, ended

        try:
            snaps, caps, meta, ended = await asyncio.to_thread(load)
        except Exception as e:                 # a broken cache must never block startup
            log.warning("cache load failed: %s", e)
            self.notices.append(f"cache could not be read ({e}); starting without it")
            return
        for host, snap in snaps.items():
            if host in self.store.hosts and snapshots:
                self.store.set_snapshot(host, snap)
        for host, entry in caps.items():
            if entry is not None:
                self._cached_caps[host] = entry
            elif host in snaps and snaps[host].caps is not None:
                self._cached_caps[host] = (snaps[host].caps, None)   # profile known, re-detect soon
        self._jobmeta, self._ended = meta, ended

    @property
    def _lock_fd(self) -> int | None:            # tests / introspection: the first held lock, if any
        return self._lock_fds[0] if self._lock_fds else None

    def _take_instance_lock(self) -> None:
        """§5.4: warn (once) when another RunFleet already polls one of these hosts from this account. One lock per
        ssh destination, so two instances with disjoint host sets (a localhost-only config next to a full one)
        do not warn; only the overlapping hosts are named."""
        base = self._lock_base()
        if base is None:
            return
        busy: dict[str, list[str]] = {}
        for name, state in self.store.hosts.items():
            if not state.config.enabled:
                continue
            other = self._lock_host(name, base)
            if other is not None:
                busy.setdefault(other, []).append(name)
        for other, names in busy.items():
            shown = ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
            self.notices.append(f"another RunFleet (pid {other}) is polling {shown} — remote load doubles there")

    def _lock_base(self) -> Path | None:
        from runfleet import paths
        if self.lock_path is False:
            return None
        try:
            return self.lock_path or paths.ensure_private_dir(paths.mux_dir()) / "instance.lock"
        except OSError as e:
            log.info("instance lock unavailable: %s", e)
            return None

    def _lock_host(self, name: str, base: Path) -> str | None:
        """Take the host's instance lock; returns the pid text of the instance holding it, else None."""
        import fcntl
        import hashlib

        if name in self._host_locks:
            return None
        dest = self.store.hosts[name].config.ssh
        key = re.sub(r"[^A-Za-z0-9_.\-]", "_", dest)[:40] + "-" + hashlib.sha1(dest.encode()).hexdigest()[:8]
        try:
            fd = os.open(base.with_name(f"{base.name}.{key}"), os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            log.info("instance lock for %s unavailable: %s", name, e)
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            other = os.read(fd, 32).decode(errors="replace").strip() or "?"
            os.close(fd)
            return other
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fds.append(fd)
        self._host_locks[name] = fd
        return None

    def _unlock_host(self, name: str) -> None:
        fd = self._host_locks.pop(name, None)
        if fd is not None:
            if fd in self._lock_fds:
                self._lock_fds.remove(fd)
            with contextlib.suppress(OSError):
                os.close(fd)

    async def stop(self) -> None:
        if self._forward_ops:                          # a forward a cancelled worker started is recorded first …
            await asyncio.gather(*self._forward_ops, return_exceptions=True)
        await self._cancel_forwards()                  # … a persisting master must not keep our local listeners
        for t in list(self._removing.values()):        # a host teardown in flight finishes first (never cancelled)
            with contextlib.suppress(Exception):
                await t
        await self.attach.close()                      # attach samples first: their ssh dies with the task
        tasks = [*self._tasks.values(), *self._bg]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        self._bg.clear()
        await self.logs.close_all()
        if self.cache is not None:
            with contextlib.suppress(Exception):
                await self.cache.flush()
        for transport in self._transports.values():
            with contextlib.suppress(Exception):
                await transport.close()
        for fd in self._lock_fds:
            os.close(fd)
        self._lock_fds.clear()
        self._host_locks.clear()
        if self._remember_unsub is not None:
            self._remember_unsub()
            self._remember_unsub = None
        self._started = False

    async def collect_once(self, hosts: Iterable[str] | None = None, *, timeout: float = 60) -> None:
        """`snapshot`: prepare + detect + one full poll per host, no pollers left running. Failures end up in
        each host's status (state / error); this method does not raise for host problems."""
        if self._tasks:
            raise RunFleetError("collect_once() while pollers are running")
        names = list(hosts) if hosts is not None else [n for n, hs in self.store.hosts.items() if hs.config.enabled]
        for n in names:
            self._host_cfg(n)                                  # unknown host -> RunFleetError
        await self._load_cache(snapshots=False)                # cached caps skip the detect probe; jobmeta log paths

        async def one(host: str) -> None:
            p = self._poller(host)
            try:
                await asyncio.wait_for(p.collect_once(), timeout)
            except TimeoutError:
                st = self.store.hosts[host].status
                err = HostError(ErrorKind.TIMEOUT, f"snapshot timed out after {timeout:.0f} s", at=self.clock.now())
                self.store.set_status(host, _replace(st, state=ConnState.UNREACHABLE, error=err, in_flight=False,
                                                     consecutive_failures=st.consecutive_failures + 1))

        await asyncio.gather(*(one(n) for n in names))
        if self.cache is not None:
            await self.cache.flush()

    # ================================================================ scheduling hints
    def set_visibility(self, levels: Mapping[str, Visibility]) -> None:
        for host, v in levels.items():
            self._visibility[host] = v
            p = self._pollers.get(host)
            if p is not None:
                p.set_visibility(v)

    def note_user_activity(self) -> None:
        self._last_input = self.clock.monotonic()
        if self.store.idle_paused:
            self.store.set_idle_paused(False)
            for host, p in self._pollers.items():
                p.set_paused(False)
                if self._visibility.get(host, Visibility.VISIBLE) is not Visibility.HIDDEN:
                    p.wake("resume")
        if self._idle_throttled:
            self._idle_throttled = False
            self._apply_multiplier()
        self._activity.set()

    def set_app_focus(self, focused: bool) -> None:
        if focused != self._focused:
            self._focused = focused
            self._apply_multiplier()
            self.attach.poke()                         # Attach auto-refresh pauses while blurred

    def refresh(self, hosts: Iterable[str] | None = None) -> None:
        now = self.clock.monotonic()
        for host in (list(hosts) if hosts is not None else list(self._tasks)):
            last = self._last_refresh.get(host)
            if host not in self._tasks or (last is not None and now - last < self.REFRESH_FLOOR):
                continue
            self._last_refresh[host] = now
            self._pollers[host].wake("manual")

    def set_rate_multiplier(self, m: float) -> None:
        self._rate = min(4.0, max(0.5, m))
        self._apply_multiplier()

    def set_host_enabled(self, host: str, enabled: bool) -> None:
        hs = self.store.hosts[host]
        hs.config = _replace(hs.config, enabled=enabled)
        p = self._pollers.get(host)
        if p is not None:
            p.cfg = hs.config
        if not enabled:
            task = self._tasks.pop(host, None)
            if task is not None:
                task.cancel()
            self.store.set_status(host, _replace(hs.status, state=ConnState.DISABLED, in_flight=False,
                                                 next_poll_at=None))
        else:
            self.store.set_status(host, _replace(hs.status, state=ConnState.CONNECTING, error=None))
            if self._started:
                self._poller(host).reconnect()
                self._start_poller(host)

    async def _idle_watch(self) -> None:
        """No input for idle_after -> ×4; for idle_pause_after -> all polling paused until the next input. Reads the
        settings on every round: Settings may change them (set_general wakes this loop)."""
        while True:
            g = self.config.general
            self._activity.clear()
            idle = self.clock.monotonic() - self._last_input
            delay: float | None
            if idle >= g.idle_pause_after:
                if not self.store.idle_paused:
                    self.store.set_idle_paused(True)
                    for p in self._pollers.values():
                        p.set_paused(True)
                delay = None
            elif idle >= g.idle_after:
                if not self._idle_throttled:
                    self._idle_throttled = True
                    self._apply_multiplier()
                delay = g.idle_pause_after - idle
            else:
                if self._idle_throttled:            # a longer idle_after from Settings ends the throttle
                    self._idle_throttled = False
                    self._apply_multiplier()
                delay = g.idle_after - idle
            waiters: set[asyncio.Future[Any]] = {asyncio.ensure_future(self._activity.wait())}
            if delay is not None:
                waiters.add(asyncio.ensure_future(self.clock.sleep(delay)))
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()

    async def _flush_loop(self) -> None:
        assert self.cache is not None
        while True:
            await self.clock.sleep(self.FLUSH_EVERY)
            try:
                await self.cache.flush()
            except Exception as e:
                log.warning("cache flush failed: %s", e)

    # ================================================================ details and logs
    async def job_detail(self, key: JobKey, *, max_age_s: float = 30) -> JobDetail:
        from runfleet.probe import framing

        now_m = self.clock.monotonic()
        cached = self._details.get(key)
        if cached is not None and now_m - cached[0] < max_age_s:
            return self._with_live(cached[1])
        job = self.store.job(key)
        if job is None:
            raise RunFleetError(f"unknown job {key}")
        sched, ctx = self._scheduler(key.host), self._context(key.host)
        if sched is None or ctx is None:
            raise RunFleetError(f"{key.host}: not connected yet")
        sections = sched.detail_sections(job, ctx)
        results: Mapping[str, SectionResult] = {}
        if sections:
            res = await self._on_demand(key.host, framing.make_script(sections, "detail"))
            if res.error is not None:
                return self._with_live(_bare_detail(job, self.clock.now(), res.error))
            results = res.sections
        try:
            detail = sched.ingest_detail(job, results, self._context(key.host) or ctx)
        except Exception as e:
            log.exception("%s: ingest_detail failed", key)
            return self._with_live(_bare_detail(job, self.clock.now(),
                                                HostError(ErrorKind.INTERNAL, f"internal error: {e}"[:80])))
        self._details[key] = (now_m, detail)
        return self._with_live(detail)

    def _with_live(self, detail: JobDetail) -> JobDetail:
        """JobDetail.live = the job's latest Attach sample with data this session (§7.16; never cached)."""
        live = self.attach.live_for(detail.job.key)
        return detail if live is None else dataclasses.replace(detail, live=live)

    async def peek_log(self, host: str, ref: LogRef, *, nbytes: int = 4096) -> LogPeek:
        return await self.logs.peek(host, ref, nbytes=nbytes)

    async def stream_log(self, host: str, ref: LogRef, *, tail_bytes: int = 65536) -> LogSession:
        return await self.logs.stream(host, ref, tail_bytes=tail_bytes)

    async def fetch_log(self, host: str, ref: LogRef, *, nbytes: int = 4 << 20) -> Path:
        return await self.logs.fetch(host, ref, nbytes=nbytes)

    # ================================================================ actions
    def actions_for(self, keys: Sequence[JobKey]) -> list[ActionSpec]:
        return self.actions.actions_for(keys)

    def plan(self, action_id: str, keys: Sequence[JobKey],
             params: Mapping[str, object] | None = None) -> list[ActionPlan]:
        return self.actions.plan(action_id, keys, params)

    async def execute(self, plans: Sequence[ActionPlan]) -> list[ActionResult]:
        results = await self.actions.execute(plans)
        for r in results:
            self._details.pop(r.target, None)
        return results

    def action_log(self) -> Sequence[ActionResult]:
        return self.actions.action_log()

    # ================================================================ interactive argv
    async def prepare_connect(self, host: str) -> bool:
        """Before `connect_argv`: resolve the host's mux plan and clear a stale control socket. True if a working
        master is already up, so the interactive login would only start a second, non-multiplexed session."""
        return await self._transport(host).before_connect()

    def connect_argv(self, host: str) -> list[str] | None:
        return self._transport(host).connect_argv()

    async def after_connect(self, host: str, rc: int) -> None:
        self._details = {k: v for k, v in self._details.items() if k.host != host}
        if host in self._tasks:
            self._pollers[host].reconnect()
        elif self._started and self.store.hosts[host].config.enabled:
            self._poller(host).reconnect()
            self._start_poller(host)

    def shell_argv(self, host: str, key: JobKey | None = None, node: str | None = None) -> list[str]:
        cmd: str | None = None
        sched, ctx = self._scheduler(host), self._context(host)
        if sched is not None and ctx is not None:
            job = self.store.job(key) if key is not None else None
            try:
                cmd = sched.shell_command(job, ctx, node)
            except Exception:
                log.exception("%s: shell_command failed", host)
        return self._transport(host).interactive_argv(cmd)

    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None:
        return self.logs.pager_argv(host, ref)

    def job_shell_command(self, host: str, key: JobKey | None = None, node: str | None = None) -> str | None:
        sched, ctx = self._scheduler(host), self._context(host)
        job = self.store.job(key) if key is not None else None
        if sched is None or ctx is None or job is None:
            return None
        try:
            return sched.shell_command(job, ctx, node)
        except Exception:
            log.exception("%s: shell_command failed", host)
            return None

    # ================================================================ port forwards (local `ssh -O forward`)
    def forward_blocker(self, host: str) -> str | None:
        hs = self.store.hosts.get(host)
        if hs is None:
            return f"unknown host {host}"
        st = hs.status.state
        if st in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
            return f"{host} needs a login — press c first"
        if st not in (ConnState.OK, ConnState.DEGRADED):
            return f"{host} is not connected ({st.value})"
        return self._transport(host).forward_blocker()

    async def forward_port(self, host: str, node: str, remote_port: int, *, job: JobKey | None = None,
                           scheme: str = "http", path: str = "", label: str = "") -> PortForward:
        from runfleet.core import forwards

        reason = self.forward_blocker(host)
        if reason is not None:
            raise RunFleetError(reason)
        err = await self._ensure_master(host)
        if err is not None:
            raise RunFleetError(f"{host}: {err.summary}")
        t = self._transport(host)
        for local, fwd in list(self._forwards.items()):
            if (fwd.host, fwd.node, fwd.remote_port) == (host, node, remote_port):
                # the master may have been replaced since (reconnect) and its listener gone with it; OpenSSH answers
                # a forward it already has with OK, so asking again is free and restores a lost one
                if await self._forward_op(t.forward(local, node, remote_port)) is None:
                    return fwd
                self._forwards.pop(local, None)
        preferred: int | None = remote_port
        failure = "port forward failed"
        for _ in range(2):              # the probed local port can be taken before ssh binds it: once more, any port
            try:
                local = forwards.pick_local_port(preferred, self._forwards)
            except OSError as e:
                raise RunFleetError(f"no free local port: {e}") from None
            new = forwards.PortForward(host, node, remote_port, local,
                                       forwards.local_url(local, scheme=scheme, path=path), job, label,
                                       self.clock.now())
            err = await self._forward_op(t.forward(local, node, remote_port), record=new)
            if err is None:
                log.info("%s: forwarding 127.0.0.1:%d -> %s:%d", host, local, node, remote_port)
                return new
            preferred, failure = None, err.summary
        raise RunFleetError(f"{host}: {failure}")

    async def _forward_op(self, op: Awaitable[HostError | None], *,
                          record: PortForward | None = None) -> HostError | None:
        """Run one `-O forward|cancel` to its end even if the caller is cancelled (a new `P`, closing the screen):
        the master may already hold the listener, so a successful forward is recorded (and cancelled by stop())
        whether or not anyone still awaits it."""
        async def run() -> HostError | None:
            err = await op
            if err is None and record is not None:
                self._forwards[record.local_port] = record
            return err

        task = asyncio.ensure_future(run())
        self._forward_ops.add(task)
        task.add_done_callback(self._forward_ops.discard)
        return await asyncio.shield(task)

    async def cancel_forward(self, local_port: int) -> None:
        fwd = self._forwards.get(local_port)
        if fwd is None:
            return
        transport = self._transports.get(fwd.host)
        if transport is None:
            self._forwards.pop(local_port, None)
            return
        err = await self._forward_op(transport.cancel_forward(fwd.local_port, fwd.node, fwd.remote_port))
        if err is not None and not _master_gone(err):    # still listening, maybe: keep it listed (stop() retries)
            raise RunFleetError(f"{fwd.host}: {err.summary}")
        if err is not None:            # the master is gone already, taking the forward with it
            log.info("%s: cancel forward %d: %s", fwd.host, local_port, err.summary)
        if self._forwards.get(local_port) is fwd:
            del self._forwards[local_port]

    def forwards(self) -> list[PortForward]:
        return list(self._forwards.values())

    async def _cancel_forwards(self, host: str | None = None) -> None:
        for port in [p for p, f in self._forwards.items() if host is None or f.host == host]:
            try:
                await self.cancel_forward(port)
            except Exception as exc:   # exit / host removal: nothing left to retry with
                log.warning("cancel forward %d: %s", port, exc)
                self._forwards.pop(port, None)

    def local_pager_argv(self, path: Path | None = None) -> list[str]:
        return self.logs.local_pager_argv(path)

    # ================================================================ attach (§7.16): core/attach.py
    def attach_start(self, key: JobKey, *, node: str | None = None) -> AttachState:
        return self.attach.start(key, node=node)

    def attach_stop(self, key: JobKey | None = None) -> None:
        self.attach.stop(key)

    async def attach_now(self, key: JobKey, *, mode: AttachMode | None = None,
                         consent: Literal["once", "job"] | None = None, node: str | None = None) -> JobLive:
        return await self.attach.now(key, mode=mode, consent=consent, node=node)

    def attach_set_auto(self, key: JobKey, enabled: bool) -> AttachState:
        return self.attach.set_auto(key, enabled)

    def attach_state(self, key: JobKey) -> AttachState | None:
        return self.attach.state(key)

    # ================================================================ hosts import and diagnostics
    async def probe_alias(self, alias: str) -> DetectReport:
        from runfleet.probe import framing
        from runfleet.transport import classify
        from runfleet.transport.openssh import OpenSSHTransport

        name = host_name_for(alias) or "host"
        cfg = HostConfig(name=name, ssh=alias)
        t = self._factory(cfg)
        caps = None
        latency: float | None = None
        regs = self.hooks.collectors()
        try:
            # The whole probe (ssh -G, master creation, detect) holds one global permit: `hosts import --all` on a
            # 40-alias ~/.ssh/config behind one bastion must not open 40 unauthenticated logins at once (sshd
            # MaxStartups 10:30:100 resets them at random and the hosts would be dropped as unreachable).
            async with self._global_sem:
                err = await t.prepare()
                res = None
                if err is None:
                    script = framing.make_script(self.hooks.detect_sections(cfg, regs), "detect")
                    res = await t.probe(script, timeout=framing.local_timeout(script))
            if res is not None:
                err = res.error
                if err is None:
                    latency = res.run.elapsed_s
                    try:
                        caps = self.hooks.build_capabilities(res.sections, cfg, self.clock.now(), regs)
                    except RunFleetError as e:
                        err = HostError(ErrorKind.PARSE, f"detection failed: {e}"[:80], str(e))
            summary = t.settings.summary() if isinstance(t, OpenSSHTransport) and t.settings else {}
            state = ConnState.OK if err is None else classify.state_for(err, 3)
            return DetectReport(alias=alias, state=state, error=err, caps=caps, mux=t.mux, ssh_summary=summary,
                                latency_s=latency)
        finally:
            with contextlib.suppress(Exception):
                await t.close()

    def write_config(self, hosts: Sequence[HostConfig], path: Path | None = None) -> Path:
        from runfleet import paths
        from runfleet.config import dump_hosts_toml, host_table_toml

        target = path or self.config.source or paths.config_file()
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(dump_hosts_toml(hosts), encoding="utf-8")
            return target
        text = target.read_text(encoding="utf-8")
        try:
            existing = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("", f"{target}: invalid TOML: {e}") from None
        present = {h.get("name") for h in existing.get("host", []) if isinstance(h, dict)}
        new = [h for h in hosts if h.name not in present]
        if new:
            sep = "" if text.endswith("\n\n") or not text else ("\n" if text.endswith("\n") else "\n\n")
            with target.open("a", encoding="utf-8") as f:
                f.write(sep + "\n".join(host_table_toml(h) for h in new))
        return target

    async def doctor(self, host: str, *, reset: bool = False) -> DoctorReport:
        from runfleet.core.doctor import run_doctor

        if host in self.store.hosts:
            cfg, t = self.store.hosts[host].config, self._transport(host)
        else:
            cfg = HostConfig(name=host_name_for(host), ssh=host)
            t = self._factory(cfg)
        cache = self.cache if host in self.store.hosts else None

        def keep(caps: Capabilities) -> None:          # a following `snapshot` reuses them (no second detect probe)
            if cache is not None:
                cache.put_caps(host, caps, self.clock.now())

        from runfleet.core.poller import HostLanes
        lanes = self._lanes_for(host) if host in self.store.hosts else HostLanes(self._global_sem, t.mux)
        report = await run_doctor(cfg, t, hooks=self.hooks, clock=self.clock, reset=reset, on_caps=keep, lanes=lanes)
        if cache is not None:
            await cache.flush()
        return report

    # ================================================================ runtime hosts and settings (§7.13, §8.4)
    def ssh_aliases(self) -> list[str]:
        try:
            return list(self.alias_source())
        except Exception as e:                 # an unreadable ~/.ssh/config must not break Settings
            log.warning("ssh aliases unavailable: %s", e)
            return []

    def _file(self) -> Any:
        from runfleet.configedit import ConfigFile
        return ConfigFile(self.config_path) if self.config_path is not None else None

    def saved_hosts(self) -> list[HostConfig]:
        f = self._file()
        return f.hosts() if f is not None else list(self._memory_hosts)

    def custom_host_keys(self, host: str) -> list[str]:
        f = self._file()
        if f is None:
            return []
        try:
            keys: list[str] = f.host_keys(host)
        except ConfigError:
            return []
        return keys

    def add_host(self, cfg: HostConfig, *, save: bool = True) -> HostConfig:
        from runfleet.config import HOST_SCHEMA
        HOST_SCHEMA["name"](cfg.name, "host.name")             # ConfigError for a name the file would reject
        if not cfg.ssh or cfg.ssh.startswith("-") or any(c.isspace() for c in cfg.ssh):
            raise ConfigError("host.ssh", f"invalid ssh target {cfg.ssh!r}")
        if cfg.name in self._removing:
            raise ConfigError("host.name", f"{cfg.name} is still being removed — try again in a moment")

        def clash(hosts: Iterable[HostConfig]) -> None:     # same name, other endpoint: never a silent success
            other = next((h for h in hosts if h.name == cfg.name and h.ssh != cfg.ssh), None)
            if other is not None:
                raise ConfigError("host.name", f"{cfg.name} already exists (ssh = {other.ssh!r})")

        clash(hs.config for hs in self.store.hosts.values())
        running = cfg
        if save:
            f = self._file()
            if f is not None:
                clash(f.hosts())
                f.add_hosts([cfg], restore=self._removed)
                self._removed.pop(cfg.name, None)
                running = next((h for h in f.hosts() if h.name == cfg.name), cfg)   # [defaults.*] applied
            else:
                clash(self._memory_hosts)
                if not any(h.name == cfg.name or h.ssh == cfg.ssh != "local" for h in self._memory_hosts):
                    self._memory_hosts.append(HostConfig(name=cfg.name, ssh=cfg.ssh))
        if cfg.name in self.store.hosts:
            return self.store.hosts[cfg.name].config
        self.store.add_host(running)
        for p in self._pollers.values():
            p.n_hosts = len(self.store.hosts)
        if self._started and running.enabled:
            base = self._lock_base()
            other = self._lock_host(cfg.name, base) if base is not None else None
            if other is not None:
                self.notices.append(f"another RunFleet (pid {other}) is polling {cfg.name} — remote load doubles "
                                    "there")
            self._start_poller(cfg.name)
        return running

    async def remove_host(self, host: str, *, save: bool = True) -> None:
        pending = self._removing.get(host)
        if pending is not None:                    # a second remove of the same host waits for the first teardown
            await asyncio.shield(pending)
        if save:
            f = self._file()
            if f is not None:
                removed = f.remove_host(host)
                if removed:
                    self._removed[host] = removed
            else:
                self._memory_hosts = [h for h in self._memory_hosts if h.name != host]
        self._remember.pop(host, None)
        if host not in self.store.hosts:
            return
        # a tracked task awaited through shield: a cancelled caller (Settings closed mid-removal) cannot leave the
        # host in the store with its poller already cancelled
        teardown = self._removing[host] = asyncio.ensure_future(self._teardown_host(host))
        await asyncio.shield(teardown)

    async def _teardown_host(self, host: str) -> None:
        try:
            await self.attach.forget_host(host)
            task = self._tasks.pop(host, None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await self._cancel_forwards(host)
            await self.logs.close_host(host)
            transport = self._transports.pop(host, None)
            if transport is not None:
                with contextlib.suppress(Exception):
                    await transport.close()
            self._unlock_host(host)
            for d in (self._pollers, self._lanes, self._visibility, self._last_refresh, self._cached_caps,
                      self._jobmeta, self._ended):
                d.pop(host, None)
            self._details = {k: v for k, v in self._details.items() if k.host != host}
            self.store.remove_host(host)
            for p in self._pollers.values():
                p.n_hosts = len(self.store.hosts)
        finally:
            self._removing.pop(host, None)

    def reorder_hosts(self, names: Sequence[str], *, save: bool = True) -> None:
        if save:
            f = self._file()
            if f is not None:
                f.reorder_hosts(names)
            else:
                rank = {n: i for i, n in enumerate(dict.fromkeys(names))}
                self._memory_hosts.sort(key=lambda h: rank.get(h.name, len(rank)))
        self.store.reorder(names)

    def set_general(self, *, save: bool = True, **changes: Any) -> GeneralSettings:
        from runfleet.configedit import general_value
        for k, v in changes.items():
            general_value(k, v)                                 # unknown key / wrong type -> ConfigError
        if save:
            f = self._file()
            if f is not None:
                f.set_general(changes)
        g: GeneralSettings = dataclasses.replace(self.config.general, **changes)
        self.config = dataclasses.replace(self.config, general=g)
        if "readonly" in changes:
            self.readonly = self.actions.readonly = bool(g.readonly)
            self.attach.poke()                  # srun offers appear / disappear
        if "refresh_multiplier" in changes:
            self.set_rate_multiplier(g.refresh_multiplier)
        if "pager" in changes:
            self.logs.set_pager(g.pager)
        for p in self._pollers.values():
            p.general = g
        if changes.keys() & {"idle_after", "idle_pause_after"}:
            self._activity.set()            # re-evaluate the idle state with the new thresholds (not an input)
        return g

    def set_notify(self, *, save: bool = True, **changes: Any) -> NotifySettings:
        from runfleet.config import NOTIFY_SCHEMA
        from runfleet.configedit import table_value
        # unknown key / wrong type -> ConfigError; the live value is what loading the file gives ("2s" -> 2.0)
        values: dict[str, Any] = {}
        for k, v in changes.items():
            stored = table_value("notify", k, v)
            parsed = NOTIFY_SCHEMA[k](stored, f"notify.{k}")
            values[k] = tuple(parsed) if isinstance(parsed, list) else parsed
        if save:
            f = self._file()
            if f is not None:
                f.set_table("notify", changes)
        n: NotifySettings = dataclasses.replace(self.config.notify, **values)
        self.config = dataclasses.replace(self.config, notify=n)
        return n

    def remember_hosts(self, names: Sequence[str]) -> None:
        if not self.config.general.remember_hosts:
            return
        self._remember_order = list(names)
        try:
            saved = self.saved_hosts()
        except ConfigError as e:
            log.warning("remember hosts: %s", e)
            return
        known = {h.name for h in saved} | {h.ssh for h in saved}
        for n in names:
            hs = self.store.hosts.get(n)
            if hs is not None and n not in known and hs.config.ssh not in known:
                self._remember[n] = hs.config
        if self._remember and self._remember_unsub is None:
            self._remember_unsub = self.store.subscribe(self._on_remember_change)
        for n in list(self._remember):
            self._check_remember(n)

    def _on_remember_change(self, change: Any) -> None:
        if change.status and change.host in self._remember:
            with contextlib.suppress(RuntimeError):       # no running loop (sync tests): check right away
                asyncio.get_running_loop().call_soon(self._check_remember, change.host)
                return
            self._check_remember(change.host)

    def _check_remember(self, host: str) -> None:
        """Save a remembered host once it answered (or needs a login); one notice once none is still connecting."""
        cfg = self._remember.get(host)
        hs = self.store.hosts.get(host)
        if cfg is None or hs is None:
            self._remember.pop(host, None)
        elif not self.config.general.remember_hosts:
            self._remember.clear()
        elif hs.status.state in _ANSWERED:
            del self._remember[host]
            try:
                self.add_host(HostConfig(name=cfg.name, ssh=cfg.ssh))
            except ConfigError as e:        # also an unwritable file (read-only dotfiles dir, disk full)
                log.warning("could not save %s: %s", host, e)
                self.notices.append(f"could not save {host} to the config: {e}")
            else:
                self._remembered.append(host)
                try:
                    self._keep_command_line_order()
                except ConfigError as e:
                    log.warning("could not reorder the remembered hosts: %s", e)
        pending = [n for n in self._remember if n in self.store.hosts      # still connecting, or answered and
                   and self.store.hosts[n].status.state not in _GAVE_UP]    # about to be saved: one notice later
        if self._remembered and not pending:
            rank = {n: i for i, n in enumerate(self._remember_order)}
            names = sorted(self._remembered, key=lambda n: rank.get(n, len(rank)))
            self.notices.append(f"saved {', '.join(names)} — manage in Settings (,)")
            self._remembered = []
        if not self._remember and self._remember_unsub is not None:
            self._remember_unsub()
            self._remember_unsub = None

    def _keep_command_line_order(self) -> None:
        """Remembered hosts are saved as they answer; in the file they keep the command line's order (their slots
        among the saved hosts are re-filled in that order), so `runfleet k2 vega` numbers them 1, 2 next time."""
        rank = {n: i for i, n in enumerate(self._remember_order)}
        saved = [h.name for h in self.saved_hosts()]
        slots = [i for i, n in enumerate(saved) if n in rank]
        ordered = sorted((saved[i] for i in slots), key=lambda n: rank[n])
        wanted = list(saved)
        for i, n in zip(slots, ordered, strict=True):
            wanted[i] = n
        if wanted != saved:
            f = self._file()
            if f is not None:
                f.reorder_hosts(wanted)
            else:
                order = {n: i for i, n in enumerate(wanted)}
                self._memory_hosts.sort(key=lambda h: order.get(h.name, len(order)))


_ANSWERED = frozenset({ConnState.OK, ConnState.DEGRADED, ConnState.NEEDS_AUTH, ConnState.HOSTKEY})
# `ssh -O cancel` without a master to talk to: the forward went with the master
_NO_MASTER = ("No such file or directory", "Connection refused", "no ssh master", "not connected")
_GAVE_UP = frozenset({ConnState.UNREACHABLE, ConnState.RETRYING, ConnState.DISABLED})


def _master_gone(err: HostError) -> bool:
    return any(s in f"{err.summary}\n{err.detail}" for s in _NO_MASTER)


def _list_aliases() -> list[str]:
    from runfleet import sshconfig
    return sshconfig.list_aliases()


def _replace(obj: Any, **changes: Any) -> Any:
    return dataclasses.replace(obj, **changes)


def _bare_detail(job: Job, now: datetime, error: HostError) -> JobDetail:
    """JobDetail from store data only, when the detail probe could not run."""
    return JobDetail(job=job, fields=dict(job.raw), per_node=job.alloc.per_node, steps=(), logs=job.logs,
                     gpu=job.gpu, procs=(), procs_note="", env={}, explanation=(), fetched_at=now, errors=(error,))
