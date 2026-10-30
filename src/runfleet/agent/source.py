"""[WP4] IO side of `runfleet agent` (docs/AGENT_INTERFACE.md §4, §3.5): the read-only cache read, the fleet
view built from it (identity check, data source), the explicit `--refresh`, the `--demo` fleet polled in
memory, the one-probe log read and the one-probe find of other log files under a job's workdir.

The cache path never spawns a process and never writes: the database is opened `mode=ro` in one read
transaction, like `runfleet.status`, without `core.cache.Cache` (its constructor creates, migrates and
repairs). `--refresh` is the only writer: per host it skips what is disabled, fresh (< REFRESH_FLOOR), held
by another RunFleet, attempted by anyone < REFRESH_FLOOR ago (the lock file's mtime) or waiting for a login
nobody did (checked locally: `ssh -G` + `-O check`, zero network), takes the instance lock, decides again,
then runs one read-only `collect_once(keep_cached=True)` into a real Cache and confirms the flush, all under
one deadline; the locks are released only once the cache is closed."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from runfleet.agent.data import (REFRESH_FLOOR, STALE_AFTER, FleetView, HostHealth, HostView, Problem,
                                 ProblemCode, Source, dest_hash)
from runfleet.config import FleetConfig, HostConfig
from runfleet.core.cache import ENDED_TASKS_RETENTION, Cache, health_from_json, health_json
from runfleet.core.clock import Clock
from runfleet.core.logs import LogFiles, LogRead, find_window
from runfleet.core.service import (FleetService, TransportFactory, claim_instance_lock, instance_lock_file,
                                   lock_attempt_at, lock_holder, release_instance_lock, try_instance_lock_at)
from runfleet.models import ConnState, ErrorKind, HostError, HostSnapshot, HostStatus, Job, LogRef, from_json
from runfleet.transport.base import Transport

log = logging.getLogger("runfleet.agent.source")

BUSY_TIMEOUT = 0.5                   # s: sqlite waits this long for a lock: "cache busy" (as runfleet.status)
DEFAULT_TICK = 60.0                  # s: tick assumed for a host whose health row has none (older cache)
TUI_SLACK = 10.0                     # s: job data within 2 ticks + this of now = "the holder is polling it"
HEALTH_SKEW = 120.0                  # s: a snapshot row this much newer than its health row was written without it
AUTH_STATES = frozenset({ConnState.NEEDS_AUTH, ConnState.HOSTKEY})
NEEDS_MASTER = "needs_master"        # refresh_decision: only with a live master (needs_auth / hostkey cached)
Kind = Literal["slurm", "workstation", "unknown"]


# ==================================================================================================== reading
class NoData(Exception):
    """No cache to read; the message says why, in `runfleet status`'s words ("no cache yet", "cache busy", …).
    Not status.NoData itself: runfleet.status pulls in util.humanize, hence rich."""


@dataclass(frozen=True, slots=True)
class CacheRows:
    """Raw rows of one read-only read (one transaction): host -> (taken_at, snapshot json), host -> (at,
    state, error kind, error), host -> (at, health json), host -> ended task jsons (newest first), host ->
    caps profile."""
    snapshots: Mapping[str, tuple[float, str]] = field(default_factory=dict)
    status: Mapping[str, tuple[float, str, str | None, str | None]] = field(default_factory=dict)
    health: Mapping[str, tuple[float, str]] = field(default_factory=dict)
    ended: Mapping[str, Sequence[str]] = field(default_factory=dict)
    profiles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def read_rows(path: Path, hosts: Sequence[str] | None = None, *, ended_since: float | None = None) -> CacheRows:
    """Every table `runfleet agent` reads, from the same moment: only the rows of `hosts` (None: all) and the
    ended tasks since `ended_since` (POSIX time; None: all), both filtered in SQL. Raises NoData (the messages
    of `runfleet status`) for a missing, locked, damaged or foreign file; a missing newer table (older cache)
    reads as empty and a row that does not decode is dropped on its own."""
    if not path.is_file():
        raise NoData("no cache yet (start runfleet once)")
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=0"
    try:
        db = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT)
    except sqlite3.Error as exc:
        raise NoData(f"cache unreadable ({exc})") from None
    try:
        db.execute("BEGIN")                     # one snapshot of every table (WAL: never blocks the writer)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "snapshots" not in tables:
            raise NoData("cache has no snapshots yet")
        where, args = _host_filter(hosts)
        snaps: dict[str, tuple[float, str]] = {}
        for h, t, j in db.execute(f"SELECT host, taken_at, json FROM snapshots{where}", args):
            with contextlib.suppress(ValueError, TypeError):
                if h is not None and t is not None and isinstance(j, str):
                    snaps[str(h)] = (float(t), j)
        status: dict[str, tuple[float, str, str | None, str | None]] = {}
        if "host_status" in tables:             # RunFleet <= 0.5.0 lacks it
            query = f"SELECT host, at, state, error_kind, error FROM host_status{where}"
            for h, at, state, kind, err in db.execute(query, args):
                with contextlib.suppress(ValueError, TypeError):
                    if h is not None and at is not None and state is not None:
                        status[str(h)] = (float(at), str(state), _opt_str(kind), _opt_str(err))
        health: dict[str, tuple[float, str]] = {}
        if "host_health" in tables:             # RunFleet < 0.8 lacks it
            for h, at, j in db.execute(f"SELECT host, at, json FROM host_health{where}", args):
                with contextlib.suppress(ValueError, TypeError):
                    if h is not None and at is not None and isinstance(j, str):
                        health[str(h)] = (float(at), j)
        ended: dict[str, list[str]] = {}
        if "ended_tasks" in tables:
            cond, cargs = where, args
            if ended_since is not None:
                cond = f"{where} AND ended_at >= ?" if where else " WHERE ended_at >= ?"
                cargs = [*args, ended_since]
            for h, j in db.execute(f"SELECT host, json FROM ended_tasks{cond} ORDER BY ended_at DESC", cargs):
                if h is not None and isinstance(j, str):
                    ended.setdefault(str(h), []).append(j)
        profiles: dict[str, tuple[str, ...]] = {}
        if "caps" in tables:
            for h, j in db.execute(f"SELECT host, json FROM caps{where}", args):
                prof = _profile(j)
                if h is not None and prof is not None:
                    profiles[str(h)] = prof
        return CacheRows(snaps, status, health, ended, profiles)
    except sqlite3.OperationalError as exc:
        text = str(exc)
        busy = "locked" in text or "busy" in text
        raise NoData("cache busy" if busy else f"cache unreadable ({text})") from None
    except (sqlite3.Error, ValueError, TypeError) as exc:      # "file is not a database", odd column types
        raise NoData(f"cache unreadable ({exc})") from None
    finally:
        db.close()


def _host_filter(hosts: Sequence[str] | None) -> tuple[str, list[object]]:
    """`` WHERE host IN (?, …)`` and its parameters (an empty selection matches nothing); no filter for None."""
    if hosts is None:
        return "", []
    names: list[object] = list(dict.fromkeys(hosts))
    if not names:
        return " WHERE 0", []
    return f" WHERE host IN ({', '.join('?' * len(names))})", names


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _profile(text: Any) -> tuple[str, ...] | None:
    """`profile` of a cached Capabilities json (no full decode: only slurm vs workstation matters here)."""
    try:
        prof = json.loads(text).get("profile")
    except (ValueError, TypeError, AttributeError, RecursionError):
        return None
    if not isinstance(prof, list):
        return None
    return tuple(str(p) for p in prof)


def _ts(value: float) -> datetime | None:
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _snapshot(text: str) -> HostSnapshot | None:
    try:
        return from_json(HostSnapshot, json.loads(text))
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as e:   # drift
        log.info("agent: dropping unreadable snapshot (%s)", e)
        return None


def _health(row: tuple[float, str] | None) -> HostHealth | None:
    if row is None:
        return None
    at = _ts(row[0])
    if at is None:
        return None
    try:
        data = json.loads(row[1])
    except (ValueError, TypeError, RecursionError):
        return None
    try:
        return health_from_json(data, at)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):    # e.g. a count of 1e309
        return None


def _jobs(texts: Sequence[str]) -> tuple[Job, ...]:
    out: list[Job] = []
    for text in texts:
        try:
            out.append(from_json(Job, json.loads(text)))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            continue
    return tuple(out)


# ================================================================================================== decisions
def identity_ok(health: HostHealth | None, ssh: str) -> bool:
    """§4.3: the cached rows are this host's unless its health row names another ssh destination (a second
    `-c` config using the same display name). No health row or no hash (older cache): nothing to
    contradict."""
    return health is None or health.dest_hash is None or health.dest_hash == dest_hash(ssh)


def decide_source(holder: str | None, job_age_s: float | None, tick: float | None) -> Source:
    """TUI when another RunFleet holds the host's instance lock and its job data advances (age within two
    ticks of the holder's cadence plus slack), else CACHE: a held lock with old data is an idle or paused
    TUI."""
    if holder is None or job_age_s is None:
        return Source.CACHE
    return Source.TUI if job_age_s <= 2 * (tick or DEFAULT_TICK) + TUI_SLACK else Source.CACHE


def decide_kind(profile: Sequence[str] | None, backend: str) -> Kind:
    """From the detected collector profile (`slurm` in it: slurm; detected otherwise: workstation), else the
    config's `backend`; unknown when neither says."""
    if profile is not None:
        return "slurm" if "slurm" in profile else "workstation"
    if backend == "slurm":
        return "slurm"
    if backend == "workstation":
        return "workstation"
    return "unknown"


def is_session_host(cfg: HostConfig) -> bool:
    """The host would run over the session transport (`transport.make_transport`'s choice): auth recovery
    is the TUI's `c`, and no other process can share its login."""
    from runfleet.transport import mux
    return cfg.ssh != "local" and mux.wants_session(cfg)


def job_age(view: HostView, now: datetime) -> float | None:
    at = view.data_at("jobs")
    return None if at is None else (now - at).total_seconds()


def refresh_decision(view: HostView, now: datetime) -> str | None:
    """§4.2 for one host, before and again after taking its lock: "skipped:disabled", "skipped:mismatch"
    (the cache rows under this name belong to another ssh destination: a poll would load that machine's
    capabilities and job metadata, then overwrite the other config's rows), "skipped:fresh", NEEDS_MASTER
    (cached needs_auth / hostkey: refresh only over a live master), or None (refresh it)."""
    if not view.enabled:
        return "skipped:disabled"
    if any(p.code is ProblemCode.CONFIG_MISMATCH for p in view.problems):
        return "skipped:mismatch"
    age = job_age(view, now)
    if age is not None and age < REFRESH_FLOOR:
        return "skipped:fresh"
    if view.state in AUTH_STATES:
        return NEEDS_MASTER
    return None


def select_hosts(config: FleetConfig, names: Sequence[str] | None) -> list[HostConfig]:
    """The enabled hosts (names None or empty), else exactly the named ones; config order either way. Raises
    ValueError for a name the config does not have."""
    if not names:
        return [h for h in config.hosts if h.enabled]
    known = {h.name for h in config.hosts}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown host {unknown[0]!r}")
    wanted = set(names)
    return [h for h in config.hosts if h.name in wanted]


# ====================================================================================================== view
def host_view(cfg: HostConfig, rows: CacheRows, now: datetime, *, lock_base: Path | None,
              outcome: str | None = None) -> HostView:
    """One host from the cache rows: identity check, data source (TUI / CACHE / LIVE after a refresh), the
    problems only this side can tell (config_mismatch, why data is stale, what a refresh did)."""
    name = cfg.name
    health = _health(rows.health.get(name))
    snap_row = rows.snapshots.get(name)
    if health is not None and snap_row is not None and snap_row[0] - health.at.timestamp() > HEALTH_SKEW:
        health = None           # an older RunFleet wrote the snapshot since: identity unverified, age approximate
    problems: list[Problem] = []
    snapshot: HostSnapshot | None = None
    ended: tuple[Job, ...] = ()
    state: ConnState | None = None
    since: datetime | None = None
    error_kind: str | None = None
    profile: tuple[str, ...] | None = None
    if not identity_ok(health, cfg.ssh):
        health = None
        problems.append(Problem(ProblemCode.CONFIG_MISMATCH, host=name))
    else:
        snapshot = _snapshot(snap_row[1]) if snap_row is not None else None
        ended = _jobs(rows.ended.get(name, ()))
        st = rows.status.get(name)
        if st is not None:
            with contextlib.suppress(ValueError):
                state = ConnState(st[1])
            since = _ts(st[0])
            error_kind = st[2]
        elif health is not None:
            state = health.state
        if snapshot is not None and snapshot.caps is not None:
            profile = snapshot.caps.profile
        else:
            profile = rows.profiles.get(name)
    kind = decide_kind(profile, cfg.backend)
    view = HostView(name=name, kind=kind, enabled=cfg.enabled, state=state, state_since=since,
                    error_kind=error_kind, snapshot=snapshot, health=health, ended_tasks=ended,
                    session_transport=is_session_host(cfg),
                    sacct_window_s=cfg.slurm.sacct_window if kind == "slurm" else None)
    age = job_age(view, now)
    holder = lock_holder(instance_lock_file(lock_base, cfg.ssh)) if lock_base is not None else None
    source = Source.LIVE if outcome == "refreshed" else decide_source(
        holder, age, health.tick_interval if health is not None else None)
    data_at = view.data_at("jobs")
    if outcome == "failed:timeout":
        problems.append(Problem(ProblemCode.TIMEOUT, host=name, since=data_at, reason="refresh"))
    elif outcome == "failed:flush":
        problems.append(Problem(ProblemCode.STALE, host=name, since=data_at, reason="refresh_skipped"))
    elif outcome == "skipped:rate_limited" or (outcome == "skipped:in_progress"
                                               and (age is None or age >= REFRESH_FLOOR)):
        # in_progress: another refresh holds the lock right now (not an idle TUI: nobody needs to press a key)
        problems.append(Problem(ProblemCode.STALE, host=name, since=data_at, reason="rate_limited"))
    elif outcome is not None and outcome.startswith("failed:"):
        problems.append(Problem(ProblemCode.STALE, host=name, since=data_at, reason="refresh_failed"))
    elif (source is Source.CACHE and cfg.enabled and age is not None and age > STALE_AFTER
          and state not in AUTH_STATES and not problems):
        problems.append(Problem(ProblemCode.STALE, host=name, since=data_at,
                                reason="tui_idle" if holder is not None else "no_poller"))
    if (health is None and snapshot is not None and state not in AUTH_STATES
            and not any(p.code is ProblemCode.STALE for p in problems)):
        # no (usable) health row: the age is the snapshot's taken_at, which a failing listing does not hold back
        problems.append(Problem(ProblemCode.STALE, host=name, since=data_at, reason="age_unknown"))
    return dataclasses.replace(view, source=source, problems=tuple(problems))


def build_view(config: FleetConfig, hosts: Sequence[HostConfig], rows: CacheRows, now: datetime, *,
               lock_base: Path | None, outcomes: Mapping[str, str] | None = None,
               error: str | None = None) -> FleetView:
    fleet_problems = (Problem(ProblemCode.CACHE_BUSY),) if error == "cache busy" else ()
    done = outcomes or {}
    views = tuple(host_view(h, rows, now, lock_base=lock_base, outcome=done.get(h.name)) for h in hosts)
    return FleetView(generated_at=now, hosts=views, recent_window=config.general.recent_window, error=error,
                     stall_after=config.general.stall_after, problems=fleet_problems)


def load_view(config: FleetConfig, names: Sequence[str] | None, *, cache_path: Path, now: datetime,
              lock_base: Path | None, outcomes: Mapping[str, str] | None = None) -> FleetView:
    """The fleet as the cache has it, hosts in config order (enabled ones, or exactly `names`; ValueError
    for an unknown name). No ssh, no write. An unreadable cache sets FleetView.error (exit 1; CACHE_BUSY
    when locked) and lists the hosts without data. `lock_base` None (demo, no mux dir): nobody can be
    polling, never Source.TUI. `outcomes`: what :func:`refresh` just did (refreshed hosts become
    Source.LIVE, failures become problems)."""
    hosts = select_hosts(config, names)
    try:
        rows = read_rows(cache_path, [h.name for h in hosts], ended_since=_ended_since(now))
    except NoData as exc:
        return build_view(config, hosts, CacheRows(), now, lock_base=lock_base, outcomes=outcomes,
                          error=str(exc))
    return build_view(config, hosts, rows, now, lock_base=lock_base, outcomes=outcomes)


def _ended_since(now: datetime) -> float:
    return now.timestamp() - ENDED_TASKS_RETENTION      # what core.cache keeps (it prunes older rows)


def _rows_or_empty(path: Path, hosts: Sequence[str], now: datetime) -> CacheRows:
    try:
        return read_rows(path, hosts, ended_since=_ended_since(now))
    except NoData as exc:
        log.info("agent: refresh without cached rows (%s)", exc)
        return CacheRows()


# ==================================================================================================== refresh
async def has_live_master(transport: Transport) -> bool:
    """Whether a poll could reach the host without a login, decided locally (zero network): OpenSSH
    resolves first (`ssh -G`; an unresolved transport's check_master answers True), then `-O check` against
    the socket; a mux plan without a master, or the session transport (its login lives in another process),
    has none. Other transports (local, fake, demo) answer in memory."""
    from runfleet.transport.openssh import OpenSSHTransport
    from runfleet.transport.session import SessionTransport

    if isinstance(transport, SessionTransport):
        return False
    if isinstance(transport, OpenSSHTransport):
        if await transport.resolve() is not None:
            return False
        if transport.plan is None or not transport.plan.uses_master:
            return False
    return await transport.check_master()


def poll_outcome(status: HostStatus) -> str:
    """A collected host's outcome: "refreshed" once it answered, else "failed:<error kind>" (timeout, auth,
    dns …)."""
    if status.state in (ConnState.OK, ConnState.DEGRADED):
        return "refreshed"
    if status.error is not None:
        return f"failed:{status.error.kind.value}"
    return f"failed:{status.state.value}"


def _auth_outcome(view: HostView) -> str:
    return "skipped:hostkey" if view.state is ConnState.HOSTKEY else "skipped:auth"


def _readonly(config: FleetConfig) -> FleetConfig:
    return dataclasses.replace(config, general=dataclasses.replace(config.general, readonly=True))


async def refresh(config: FleetConfig, names: Sequence[str] | None, *, clock: Clock, timeout: float,
                  cache_path: Path, lock_base: Path | None,
                  transports: TransportFactory | None = None) -> dict[str, str]:
    """§4.2 `--refresh`: outcome per selected host, config order — "refreshed",
    "skipped:disabled|mismatch|fresh|busy|in_progress|rate_limited|auth|hostkey", "failed:timeout|flush|cache|
    <error kind>" (in_progress: another refresh holds the lock, taken < REFRESH_FLOOR ago). One
    end-to-end deadline (`timeout`) covers the decisions, the locks, the poll and the flush; a host still
    undecided when it passes is "failed:timeout" (its cache rows stay as they were). Never prompts (every
    ssh is BatchMode), never runs connect/login paths."""
    hosts = select_hosts(config, names)
    outcomes: dict[str, str] = {}
    try:
        work = _refresh(config, hosts, outcomes, clock=clock, timeout=timeout, cache_path=cache_path,
                        lock_base=lock_base, transports=transports)
        await asyncio.wait_for(work, timeout)
    except TimeoutError:
        log.info("agent: refresh deadline (%.0f s) passed", timeout)
    return {h.name: outcomes.get(h.name, "failed:timeout") for h in hosts}


async def _refresh(config: FleetConfig, hosts: Sequence[HostConfig], outcomes: dict[str, str], *,
                   clock: Clock, timeout: float, cache_path: Path, lock_base: Path | None,
                   transports: TransportFactory | None) -> None:
    from runfleet.transport import make_transport

    started = clock.monotonic()
    factory = transports or make_transport
    made: dict[str, Transport] = {}

    def transport_for(cfg: HostConfig) -> Transport:       # one per host: resolved once, reused by the poll
        t = made.get(cfg.name)
        if t is None:
            t = made[cfg.name] = factory(cfg)
        return t

    def lock_file(cfg: HostConfig) -> Path | None:
        return instance_lock_file(lock_base, cfg.ssh) if lock_base is not None else None

    async def master_or_skip(cfg: HostConfig, view: HostView) -> bool:
        if await has_live_master(transport_for(cfg)):
            return True
        outcomes[cfg.name] = _auth_outcome(view)
        return False

    def rate_limited(cfg: HostConfig) -> bool:
        path = lock_file(cfg)
        return path is not None and attempted_recently(path, clock.now())

    names = [h.name for h in hosts]

    # 1. decide from the cache as it is (no lock, no network)
    now = clock.now()
    rows = await asyncio.to_thread(_rows_or_empty, cache_path, names, now)   # a busy database waits off the loop
    candidates: list[HostConfig] = []
    auth_checks: list[tuple[HostConfig, HostView]] = []
    for cfg in hosts:
        view = host_view(cfg, rows, now, lock_base=None)
        decision = refresh_decision(view, now)
        if decision is not None and decision != NEEDS_MASTER:
            outcomes[cfg.name] = decision
            continue
        path = lock_file(cfg)
        recent = rate_limited(cfg)                  # first: a held lock taken just now is a refresh at work
        if path is not None and lock_holder(path) is not None:
            outcomes[cfg.name] = "skipped:in_progress" if recent else "skipped:busy"
            continue
        if recent:
            outcomes[cfg.name] = "skipped:rate_limited"
            continue
        if decision == NEEDS_MASTER:
            auth_checks.append((cfg, view))
        candidates.append(cfg)
    await asyncio.gather(*(master_or_skip(c, v) for c, v in auth_checks))
    candidates = [c for c in candidates if c.name not in outcomes]

    # 2. take the free locks (one per ssh destination), decide again: someone may have refreshed meanwhile
    fds: dict[str, int] = {}
    try:
        locked: list[HostConfig] = []
        for cfg in candidates:
            path = lock_file(cfg)
            if path is not None and cfg.ssh not in fds:
                if rate_limited(cfg):                       # attempted since step 1 (taking the lock dates it)
                    outcomes[cfg.name] = "skipped:rate_limited"
                    continue
                fd, other, previous = try_instance_lock_at(path, write_pid=False)
                if fd is None and other is not None:
                    outcomes[cfg.name] = "skipped:in_progress" if rate_limited(cfg) else "skipped:busy"
                    continue
                if fd is not None and previous is not None and _recent(previous, clock.now()):
                    release_instance_lock(fd)               # someone came and went between the check and the lock:
                    outcomes[cfg.name] = "skipped:rate_limited"     # let go untouched (no attempt of ours to date)
                    continue
                if fd is not None and not claim_instance_lock(fd):
                    release_instance_lock(fd)               # our pid cannot be written: as no lock file, unlocked
                    fd = None
                if fd is not None:                          # (None, None): no lock file possible: unlocked
                    fds[cfg.ssh] = fd
            locked.append(cfg)
        now = clock.now()
        rows = await asyncio.to_thread(_rows_or_empty, cache_path, names, now)
        go: list[str] = []
        for cfg in locked:
            view = host_view(cfg, rows, now, lock_base=None)
            decision = refresh_decision(view, now)
            if decision is not None and decision != NEEDS_MASTER:
                outcomes[cfg.name] = decision
                continue
            if decision == NEEDS_MASTER and not await master_or_skip(cfg, view):     # again: it may be gone
                continue
            go.append(cfg.name)
        if go:
            await _collect(config, go, outcomes, clock=clock, cache_path=cache_path,
                           transport_for=transport_for, locks=fds,
                           budget=_poll_budget(timeout, clock.monotonic() - started))
    finally:
        for fd in fds.values():                             # what _collect did not take over
            release_instance_lock(fd)


def attempted_recently(path: Path, now: datetime) -> bool:
    """D2 rate limit: the lock at `path` was taken (by a refresh, `snapshot` or a TUI, i.e. a poll attempt)
    less than REFRESH_FLOOR ago. A clock far behind the file's time (another clock) does not count."""
    at = lock_attempt_at(path)
    return at is not None and _recent(at, now)


def _recent(at: float, now: datetime) -> bool:
    return abs(now.timestamp() - at) < REFRESH_FLOOR


def _poll_budget(timeout: float, elapsed: float) -> float:
    """collect_once's per-host timeout: what is left of the deadline minus a reserve for the flush."""
    return max(0.1, timeout - elapsed - min(2.0, 0.2 * timeout))


async def _collect(config: FleetConfig, names: Sequence[str], outcomes: dict[str, str], *, clock: Clock,
                   cache_path: Path, transport_for: TransportFactory, budget: float,
                   locks: dict[str, int]) -> None:
    """One collect_once into the real cache. Takes over `locks` (emptied) once the cache is open: they are
    released by the thread that closes the cache, after its last write, even when the deadline cancels us."""
    held = list(locks.values())
    locks.clear()
    opening = _Opening(cache_path, held)
    try:
        cache = await asyncio.to_thread(opening.run)     # its open retries with blocking sleeps
    except asyncio.CancelledError:          # the deadline: the worker closes what it opens, then unlocks
        opening.abandon()
        raise
    except Exception as e:                  # locked while converting, read-only fs: nothing written or polled
        log.warning("agent: cache unavailable for --refresh: %s", e)
        for n in names:
            outcomes[n] = "failed:cache"
        return
    service = FleetService(_readonly(config), transport_for, clock, cache)
    service.lock_path = False               # the locks are ours (taken above), the service takes none
    try:
        # keep_cached: a section failing now keeps its cached rows (and, in host_health, its last success)
        await service.collect_once(names, timeout=budget, keep_cached=True)
        flushed = await cache.flush()       # collect_once flushed already; True also when nothing was left
        for n in names:
            outcomes[n] = poll_outcome(service.store.hosts[n].status) if flushed else "failed:flush"
    finally:
        try:
            with contextlib.suppress(Exception):
                await service.stop()        # session transports end their ssh; flushes again (harmless)
        finally:                            # also when the deadline cancels us inside stop()
            try:                            # submitted now (not when a task first runs: a closing loop may not)
                closing = asyncio.get_running_loop().run_in_executor(None, _close_then_unlock, cache, held)
            except RuntimeError:            # the executor is shut down already
                _close_then_unlock(cache, held)
            else:
                with contextlib.suppress(Exception):
                    await asyncio.shield(closing)   # cancelled: the thread still finishes, then unlocks


class _Opening:
    """The refresh's Cache open on a worker thread, owning the instance locks until someone takes the cache:
    abandoned by the deadline (or a loop shutting down), the worker itself closes what it opened, then unlocks —
    never an asyncio callback, which a closing loop would not run."""

    def __init__(self, path: Path, fds: Sequence[int]) -> None:
        self.path, self.fds = path, list(fds)
        self._lock = threading.Lock()
        self._abandoned = False
        self._done = False                  # run() finished: failed (locks released) or handed `cache` over
        self._cache: Cache | None = None

    def run(self) -> Cache:
        try:
            cache = Cache(self.path)
        except BaseException:
            with self._lock:
                self._done = True
            for fd in self.fds:             # nothing opened: nothing to protect
                release_instance_lock(fd)
            raise
        with self._lock:
            self._done = True
            if not self._abandoned:
                self._cache = cache
                return cache
        _close_then_unlock(cache, self.fds)
        raise asyncio.CancelledError        # nobody awaits this result any more

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            cache, done = self._cache, self._done
        if done and cache is not None:      # opened, but the awaiting task was cancelled before taking it
            threading.Thread(target=_close_then_unlock, args=(cache, self.fds), name="runfleet-agent-close").start()


def _close_then_unlock(cache: Cache, fds: Sequence[int]) -> None:
    """Worker thread: the last flush and close (waiting for a flush in flight; bounded by sqlite's busy
    timeout), then the instance locks: never free while this refresh's rows are half written."""
    try:
        with contextlib.suppress(Exception):
            cache.close_sync()
    finally:
        for fd in fds:
            release_instance_lock(fd)


# ====================================================================================================== demo
async def demo_view(config: FleetConfig, transports: TransportFactory, clock: Clock,
                    names: Sequence[str] | None = None, *, timeout: float = 30) -> FleetView:
    """`--demo`: no cache. One read-only collect_once of the simulated fleet in memory (no instance lock);
    health is synthesised from each live HostStatus, source DEMO."""
    hosts = select_hosts(config, names)
    service = FleetService(_readonly(config), transports, clock, None)
    service.lock_path = False
    try:
        await service.collect_once([h.name for h in hosts if h.enabled], timeout=timeout)
    finally:
        with contextlib.suppress(Exception):
            await service.stop()
    now = clock.now()
    views: list[HostView] = []
    for cfg in hosts:
        hs = service.store.hosts[cfg.name]
        st, snap = hs.status, hs.snapshot
        user = snap.caps.remote_user if snap is not None and snap.caps is not None else None
        doc = health_json(st, dest_hash=dest_hash(cfg.ssh), remote_user=user or None)
        health = health_from_json(doc, now)
        caps = snap.caps if snap is not None else None
        kind = decide_kind(caps.profile if caps is not None else None, cfg.backend)
        views.append(HostView(
            name=cfg.name, kind=kind, enabled=cfg.enabled, state=st.state,
            state_since=st.last_attempt or st.last_ok, error_kind=st.error.kind.value if st.error else None,
            snapshot=snap, health=health, source=Source.DEMO,
            sacct_window_s=cfg.slurm.sacct_window if kind == "slurm" else None))
    return FleetView(generated_at=now, hosts=tuple(views), recent_window=config.general.recent_window,
                     demo=True, stall_after=config.general.stall_after)


# ====================================================================================================== log
def find_job(view: HostView, job_ref: str) -> Job | None:
    """The job `job_ref` names on this host (`wait.Target` matching: exact id, array parent -> its first
    task, task label / root pid), in the snapshot, else among the cached ended tasks; then, for a full task
    id with no exact hit in either, the task with that root pid and a start within the tolerance (the id
    another call printed, its start a second off: `tasks.nearest_task`). None when nothing matches."""
    from runfleet.backends.process.tasks import nearest_task
    from runfleet.wait import Target

    target = Target(view.name, job_ref, f"{view.name}:{job_ref}")
    live = view.snapshot.jobs if view.snapshot is not None else {}
    ended = {j.key.id: j for j in reversed(view.ended_tasks)}       # newest wins on a repeated id
    for jobs in (live, ended):
        found = target.matches(jobs, near=False)
        if found:
            return found[0]
    return nearest_task(job_ref, {**ended, **live})                 # the closest start of either; live wins a tie


async def read_log(config: FleetConfig, host: HostView, job_ref: str | Job, *, stream: str = "auto",
                   nbytes: int, head: bool = False, clock: Clock, timeout: float,
                   transports: TransportFactory | None = None, log_ref: LogRef | None = None) -> LogRead | None:
    """`agent log`: ONE bounded read of the job's log (LogService.read), in memory, under one deadline.
    None when there is nothing to read (no such job — tell apart with :func:`find_job` — or no known path:
    no_log_path), with zero probes. A session-transport host waiting for a login gets an AUTH error without
    any network; a deadline that passes gives a TIMEOUT error. `log_ref`: read this ref instead of choosing
    one (agent.commands passes the one agent.view shows, so path line and contents always agree)."""
    job = job_ref if isinstance(job_ref, Job) else find_job(host, job_ref)
    if job is None:
        return None
    if log_ref is not None:
        ref: LogRef | None = log_ref
    else:
        from runfleet.agent.view import pick_log      # the one stream choice (agent.view shows the same ref)
        ref = pick_log(job, stream)
    if ref is None or not ref.path:
        return None
    now = clock.now()
    empty = LogRead(path=ref.path, lines=(), live=None, size=None, mtime=None, window_bytes=nbytes,
                    started_mid_file=False, fetched_at=now)
    if host.session_transport and host.state in AUTH_STATES:
        kind = ErrorKind.HOSTKEY if host.state is ConnState.HOSTKEY else ErrorKind.AUTH
        return dataclasses.replace(empty, error=HostError(kind, "login needed (session transport)", at=now))
    cfg = next((h for h in config.hosts if h.name == host.name), None)
    if cfg is None:
        raise ValueError(f"unknown host {host.name!r}")
    from runfleet.transport import make_transport

    service = FleetService(_readonly(config), transports or make_transport, clock, None)
    service.lock_path = False
    try:
        return await asyncio.wait_for(service.logs.read(cfg.name, ref, nbytes=nbytes, head=head), timeout)
    except TimeoutError:
        return dataclasses.replace(empty, error=HostError(ErrorKind.TIMEOUT, f"log read timed out after "
                                                          f"{timeout:.0f} s", at=clock.now()))
    finally:
        with contextlib.suppress(Exception):
            await service.stop()


async def find_logs(config: FleetConfig, host: HostView, job: Job, *, clock: Clock, timeout: float,
                    transports: TransportFactory | None = None) -> LogFiles | None:
    """`agent log --files` / `--file`: ONE bounded find of the log files written under the job's workdir since it
    started, up to shortly after an ended job's end (LogService.find), under one deadline. None when the job has no known workdir (zero probes). A session-
    transport host waiting for a login gets an AUTH error without any network; a deadline that passes gives a
    TIMEOUT error."""
    workdir = job.workdir
    if not workdir:
        return None
    now = clock.now()
    empty = LogFiles(workdir=workdir, files=(), fetched_at=now)
    if host.session_transport and host.state in AUTH_STATES:
        kind = ErrorKind.HOSTKEY if host.state is ConnState.HOSTKEY else ErrorKind.AUTH
        return dataclasses.replace(empty, error=HostError(kind, "login needed (session transport)", at=now))
    cfg = next((h for h in config.hosts if h.name == host.name), None)
    if cfg is None:
        raise ValueError(f"unknown host {host.name!r}")
    from runfleet.transport import make_transport

    service = FleetService(_readonly(config), transports or make_transport, clock, None)
    service.lock_path = False
    try:
        since, until = find_window(job)
        return await asyncio.wait_for(service.logs.find(cfg.name, workdir, since=since, until=until), timeout)
    except TimeoutError:
        return dataclasses.replace(empty, error=HostError(ErrorKind.TIMEOUT, f"log find timed out after "
                                                          f"{timeout:.0f} s", at=clock.now()))
    finally:
        with contextlib.suppress(Exception):
            await service.stop()
