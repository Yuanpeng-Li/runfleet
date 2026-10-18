"""[WP4] agent.source (docs/AGENT_INTERFACE.md §4, §3.5): the read-only cache read (missing / busy /
garbage / foreign / older caches, damaged rows), identity, data source, the --refresh decision table under
one deadline, the demo view and the one-probe log read. Temporary sqlite written by the real
core.cache.Cache from the demo world; no real host."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from runfleet.agent import source
from runfleet.agent.data import REFRESH_FLOOR, STALE_AFTER, HostView, ProblemCode, Source, dest_hash
from runfleet.config import FleetConfig, HostConfig
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.core.service import (FleetService, instance_lock_file, lock_holder, release_instance_lock,
                                   try_instance_lock)
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.models import (ConnState, ErrorKind, HostSnapshot, HostStatus, Job, JobKey, JobKind, JobState, LogRef,
                             SectionStatus, StateGroup)
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult, Transport
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START

T0 = CLOCK_START
DEMO_HOSTS = ("k2", "vega", "lyra", "labstation", "lynx")


@dataclasses.dataclass
class Demo:
    world: DemoWorld
    clock: ManualClock
    config: FleetConfig
    cache: Path
    lock_base: Path
    made: list[str] = dataclasses.field(default_factory=list)      # hosts a transport was built for

    def transports(self, cfg: HostConfig) -> Transport:
        self.made.append(cfg.name)
        return DemoTransport(cfg, self.world)


async def _fill(demo: Demo, hosts: tuple[str, ...] = DEMO_HOSTS) -> None:
    cache = Cache(demo.cache)
    svc = FleetService(demo.config, lambda h: DemoTransport(h, demo.world), demo.clock, cache)
    svc.lock_path = False
    await svc.collect_once(list(hosts), timeout=30)
    await svc.stop()
    await cache.close()


@pytest.fixture
async def demo(tmp_path: Path) -> Demo:
    """The demo fleet polled once at T0 into tmp/cache.sqlite (lynx: needs_auth, no snapshot)."""
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    d = Demo(world, clock, world.fleet_config(), tmp_path / "cache.sqlite", tmp_path / "instance.lock")
    await _fill(d)
    return d


@pytest.fixture
def no_spawn(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any process spawn (or a writable Cache) fails the test: the cache path is pure sqlite reads."""
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("process spawned / Cache constructed on the read-only path")
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", boom)
    monkeypatch.setattr("os.posix_spawn", boom, raising=False)
    monkeypatch.setattr(Cache, "__init__", boom)
    yield


def _db_exec(path: Path, sql: str, params: tuple[Any, ...] = ()) -> None:
    db = sqlite3.connect(path)
    with db:
        db.execute(sql, params)
    db.close()


def _set_status(path: Path, host: str, state: ConnState, kind: str | None = None, at: datetime = T0) -> None:
    _db_exec(path, "INSERT OR REPLACE INTO host_status VALUES (?, ?, ?, ?, ?)",
             (host, at.timestamp(), state.value, kind, None))


def _touch(path: Path, host: str, at: datetime) -> None:
    """Another refresher just polled `host`: its snapshot and every section's last success move to `at`."""
    db = sqlite3.connect(path)
    with db:
        row = db.execute("SELECT json FROM host_health WHERE host = ?", (host,)).fetchone()
        if row is not None:
            data = json.loads(row[0])
            for sec in data["sections"].values():
                sec["last_ok"] = at.isoformat()
            db.execute("UPDATE host_health SET json = ?, at = ? WHERE host = ?",
                       (json.dumps(data), at.timestamp(), host))
        db.execute("UPDATE snapshots SET taken_at = ? WHERE host = ?", (at.timestamp(), host))
    db.close()


# ==================================================================================================== reading
def test_missing_garbage_and_foreign_files_are_no_data(tmp_path: Path) -> None:
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"), HostConfig("vega", "vega")))
    missing = source.load_view(cfg, None, cache_path=tmp_path / "nope.sqlite", now=T0, lock_base=None)
    assert missing.error == "no cache yet (start runfleet once)" and not missing.problems
    assert [h.name for h in missing.hosts] == ["k2", "vega"]
    assert all(h.snapshot is None for h in missing.hosts)
    garbage = tmp_path / "garbage.sqlite"
    garbage.write_bytes(b"this is not a database\n" * 200)
    v = source.load_view(cfg, None, cache_path=garbage, now=T0, lock_base=None)
    assert v.error is not None and v.error.startswith("cache unreadable")
    foreign = tmp_path / "foreign.sqlite"
    _db_exec(foreign, "CREATE TABLE notes (id INTEGER, text TEXT)")
    assert source.load_view(cfg, None, cache_path=foreign, now=T0, lock_base=None).error == \
        "cache has no snapshots yet"
    assert garbage.read_bytes() == b"this is not a database\n" * 200      # never repaired, moved or written


async def test_busy_cache_is_cache_busy(demo: Demo) -> None:
    w = sqlite3.connect(demo.cache, isolation_level=None)
    try:
        w.execute("PRAGMA locking_mode=EXCLUSIVE")
        w.execute("BEGIN EXCLUSIVE")
        w.execute("INSERT OR REPLACE INTO snapshots VALUES ('zz', 1, '{}')")
        v = source.load_view(demo.config, None, cache_path=demo.cache, now=T0, lock_base=None)
    finally:
        w.close()
    assert v.error == "cache busy" and [p.code for p in v.problems] == [ProblemCode.CACHE_BUSY]
    assert len(v.hosts) == 5 and all(h.snapshot is None for h in v.hosts)


async def test_demo_cache_reads_every_table_without_spawning(demo: Demo, no_spawn: None) -> None:
    rows = source.read_rows(demo.cache)
    assert set(rows.snapshots) == {"k2", "vega", "lyra", "labstation"}       # lynx never answered
    assert rows.status["lynx"][1] == "needs_auth" and set(rows.health) >= {"k2", "labstation"}
    assert "slurm" in rows.profiles["k2"]
    v = source.load_view(demo.config, None, cache_path=demo.cache, now=T0 + timedelta(seconds=30),
                         lock_base=demo.lock_base)
    assert v.error is None and [h.name for h in v.hosts] == list(DEMO_HOSTS)
    k2, lab, lynx = v.host("k2"), v.host("labstation"), v.host("lynx")
    assert k2 is not None and lab is not None and lynx is not None
    assert k2.kind == "slurm" and k2.snapshot is not None and k2.health is not None
    assert k2.source is Source.CACHE
    assert k2.sacct_window_s == demo.config.hosts[0].slurm.sacct_window and k2.state in (ConnState.OK,
                                                                                           ConnState.DEGRADED)
    assert k2.health.dest_hash == dest_hash("k2") and k2.data_at("jobs") == k2.health.data_at("jobs")
    assert lab.kind == "workstation" and lab.sacct_window_s is None and lab.snapshot is not None
    assert lynx.state is ConnState.NEEDS_AUTH and lynx.error_kind == "auth" and lynx.snapshot is None
    assert lynx.state_since == T0
    assert not any(p.code is ProblemCode.STALE for h in v.hosts for p in h.problems)


def test_named_hosts_in_config_order_and_unknown_names(tmp_path: Path) -> None:
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"), HostConfig("vega", "vega", enabled=False),
                             HostConfig("lyra", "lyra")))
    path = tmp_path / "none.sqlite"
    assert [h.name for h in source.load_view(cfg, None, cache_path=path, now=T0, lock_base=None).hosts] == \
        ["k2", "lyra"]                                                           # enabled ones by default
    named = source.load_view(cfg, ["lyra", "vega", "lyra"], cache_path=path, now=T0, lock_base=None)
    assert [(h.name, h.enabled) for h in named.hosts] == [("vega", False), ("lyra", True)]
    with pytest.raises(ValueError, match="orb"):
        source.load_view(cfg, ["orb"], cache_path=path, now=T0, lock_base=None)


async def test_damaged_rows_are_dropped_one_by_one(demo: Demo) -> None:
    _db_exec(demo.cache, "UPDATE snapshots SET json = '{not json' WHERE host = 'vega'")
    _db_exec(demo.cache, "UPDATE snapshots SET taken_at = 'soon' WHERE host = 'lyra'")
    _db_exec(demo.cache, "INSERT OR REPLACE INTO host_health VALUES ('labstation', 1e300, '{}')")
    _db_exec(demo.cache, "INSERT OR REPLACE INTO host_status VALUES ('k2', 1.0, 'sleepy', NULL, NULL)")
    _db_exec(demo.cache, "INSERT OR REPLACE INTO ended_tasks VALUES ('labstation', 'p1-1', 5.0, '[1, 2]')")
    _db_exec(demo.cache, "INSERT OR REPLACE INTO caps VALUES ('vega', 5.0, 'garbage')")
    v = source.load_view(demo.config, None, cache_path=demo.cache, now=T0, lock_base=None)
    assert v.error is None
    vega, lyra, lab, k2 = (v.host(n) for n in ("vega", "lyra", "labstation", "k2"))
    assert vega is not None and vega.snapshot is None and vega.health is not None   # only its snapshot lost
    assert lyra is not None and lyra.snapshot is None
    assert lab is not None and lab.health is None and lab.snapshot is not None and lab.ended_tasks == ()
    assert k2 is not None and k2.state is None and k2.snapshot is not None               # unknown state word


def test_older_cache_without_health_or_ended_tasks(tmp_path: Path) -> None:
    """A cache written by RunFleet 0.5/0.7: snapshots + host_status (+ none of the newer tables)."""
    clock = ManualClock(T0 - timedelta(hours=1))
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)

    async def snap_json() -> str:
        svc = FleetService(world.fleet_config(["k2"]), lambda h: DemoTransport(h, world), clock, None)
        svc.lock_path = False
        await svc.collect_once(["k2"], timeout=30)
        snap = svc.store.hosts["k2"].snapshot
        assert snap is not None
        from runfleet.models import to_json
        return json.dumps(to_json(snap))

    text = asyncio.run(snap_json())
    path = tmp_path / "old.sqlite"
    _db_exec(path, "CREATE TABLE snapshots (host TEXT PRIMARY KEY, taken_at REAL, json TEXT)")
    _db_exec(path, "INSERT INTO snapshots VALUES ('k2', ?, ?)", ((T0 - timedelta(hours=1)).timestamp(), text))
    v = source.load_view(world.fleet_config(["k2"]), None, cache_path=path, now=T0, lock_base=None)
    k2 = v.hosts[0]
    assert v.error is None and k2.health is None and k2.ended_tasks == () and k2.state is None
    assert k2.snapshot is not None and k2.kind == "slurm" and k2.data_at("jobs") == T0 - timedelta(hours=1)
    assert [(p.code, p.reason) for p in k2.problems] == [(ProblemCode.STALE, "no_poller")]
    young = source.load_view(world.fleet_config(["k2"]), None, cache_path=path, now=T0 - timedelta(minutes=59),
                             lock_base=None).hosts[0]
    assert [(p.code, p.reason) for p in young.problems] == [(ProblemCode.STALE, "age_unknown")]   # approximate
    _db_exec(path, "CREATE TABLE host_status (host TEXT PRIMARY KEY, at REAL, state TEXT, error_kind TEXT, "
                   "error TEXT)")
    _db_exec(path, "INSERT INTO host_status VALUES ('k2', ?, 'hostkey', 'hostkey', 'x')", (T0.timestamp(),))
    k2 = source.load_view(world.fleet_config(["k2"]), None, cache_path=path, now=T0, lock_base=None).hosts[0]
    assert k2.state is ConnState.HOSTKEY and k2.problems == ()     # the hostkey problem says it, not "stale"


# =========================================================================================== identity, source
async def test_identity_mismatch_is_no_data(demo: Demo) -> None:
    cfg = _moved_k2(demo.config)
    v = source.load_view(cfg, ["k2", "vega"], cache_path=demo.cache, now=T0, lock_base=None)
    k2, vega = v.hosts
    assert k2.snapshot is None and k2.health is None and k2.state is None and k2.ended_tasks == ()
    assert [p.code for p in k2.problems] == [ProblemCode.CONFIG_MISMATCH] and k2.data_at() is None
    assert vega.snapshot is not None and vega.problems == ()


def _moved_k2(config: FleetConfig) -> FleetConfig:
    """The same display name, another machine (a second `-c` config)."""
    return dataclasses.replace(config, hosts=tuple(
        dataclasses.replace(h, ssh="alice@other-k2.example") if h.name == "k2" else h for h in config.hosts))


async def test_refresh_never_polls_a_host_whose_rows_belong_to_another_destination(demo: Demo) -> None:
    demo.clock.advance(300)
    cfg = _moved_k2(demo.config)
    assert await _refresh(demo, ["k2"], config=cfg) == {"k2": "skipped:mismatch"}
    assert demo.made == []


async def test_a_health_row_with_absurd_numbers_drops_only_its_health(demo: Demo) -> None:
    text = source.read_rows(demo.cache).health["k2"][1]
    _db_exec(demo.cache, "UPDATE host_health SET json = ? WHERE host = 'k2'",
             (text.replace('"consecutive_failures": 0', '"consecutive_failures": 1e999'),))
    assert "1e999" in source.read_rows(demo.cache).health["k2"][1]
    k2 = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=T0, lock_base=None).hosts[0]
    assert k2.health is None and k2.snapshot is not None


@pytest.mark.parametrize(("health_hash", "ok"),
                         [(None, True), (dest_hash("k2"), True), ("0123abcd4567", False)])
def test_identity_table(health_hash: str | None, ok: bool) -> None:
    from runfleet.agent.data import HostHealth
    h = HostHealth(at=T0, state=ConnState.OK, dest_hash=health_hash, remote_user="alice", mux=None,
                   last_ok=None, latency_s=None, consecutive_failures=0, controller_slowdown=1.0,
                   tick_interval=None)
    assert source.identity_ok(h, "k2") is ok
    assert source.identity_ok(None, "k2")


@pytest.mark.parametrize(("holder", "age", "tick", "expected"), [
    (None, 5.0, 15.0, Source.CACHE),          # nobody holds the lock
    ("4242", None, 15.0, Source.CACHE),       # held, but no job data at all
    ("4242", 41.0, 15.0, Source.CACHE),       # held, data older than 2 ticks + 10 s: an idle / paused TUI
    ("4242", 40.0, 15.0, Source.TUI),
    ("4242", 40.0, 15.0 * 2, Source.TUI),
    ("4242", 130.0, None, Source.TUI),        # no tick recorded: 60 s assumed (2 * 60 + 10)
    ("4242", 131.0, None, Source.CACHE),
])
def test_source_table(holder: str | None, age: float | None, tick: float | None, expected: Source) -> None:
    assert source.decide_source(holder, age, tick) is expected


@pytest.mark.parametrize(("profile", "backend", "kind"), [
    (("slurm", "procfs"), "auto", "slurm"), (("procfs", "nvidia"), "slurm", "workstation"), ((), "auto",
                                                                                              "workstation"),
    (None, "slurm", "slurm"), (None, "workstation", "workstation"), (None, "auto", "unknown"),
])
def test_kind_table(profile: tuple[str, ...] | None, backend: str, kind: str) -> None:
    assert source.decide_kind(profile, backend) == kind


async def test_lock_held_elsewhere_makes_fresh_data_tui(demo: Demo) -> None:
    fd, _ = try_instance_lock(instance_lock_file(demo.lock_base, "k2"))
    assert fd is not None
    try:
        fresh = source.load_view(demo.config, ["k2", "vega"], cache_path=demo.cache,
                                 now=T0 + timedelta(seconds=40), lock_base=demo.lock_base)
        assert [h.source for h in fresh.hosts] == [Source.TUI, Source.CACHE]     # vega: nobody holds it
        idle = source.load_view(demo.config, ["k2", "vega"], cache_path=demo.cache,
                                now=T0 + timedelta(hours=1), lock_base=demo.lock_base)
        k2, vega = idle.hosts
        assert k2.source is Source.CACHE and [(p.code, p.reason) for p in k2.problems] == \
            [(ProblemCode.STALE, "tui_idle")]
        assert [(p.code, p.reason) for p in vega.problems] == [(ProblemCode.STALE, "no_poller")]
        assert source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=T0 + timedelta(seconds=40),
                                lock_base=None).hosts[0].source is Source.CACHE      # no lock base: never TUI
    finally:
        release_instance_lock(fd)
    after = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=T0 + timedelta(seconds=40),
                             lock_base=demo.lock_base)
    assert after.hosts[0].source is Source.CACHE


def _view(name: str = "k2", *, enabled: bool = True, state: ConnState | None = ConnState.OK,
          snapshot_at: datetime | None = None) -> HostView:
    from runfleet.models import HostSnapshot
    snap = None if snapshot_at is None else HostSnapshot(name, snapshot_at, None, {}, (), (), (), (), None)
    return HostView(name=name, kind="slurm", enabled=enabled, state=state, state_since=None, error_kind=None,
                    snapshot=snap, health=None)


@pytest.mark.parametrize(("view", "expected"), [
    (_view(enabled=False, snapshot_at=T0), "skipped:disabled"),
    (_view(snapshot_at=T0 - timedelta(seconds=REFRESH_FLOOR - 1)), "skipped:fresh"),
    (_view(snapshot_at=T0 - timedelta(seconds=REFRESH_FLOOR + 1)), None),
    (_view(), None),                                                       # no data at all: refresh
    (_view(state=ConnState.NEEDS_AUTH, snapshot_at=T0 - timedelta(hours=2)), source.NEEDS_MASTER),
    (_view(state=ConnState.HOSTKEY), source.NEEDS_MASTER),
    (_view(state=ConnState.NEEDS_AUTH, snapshot_at=T0), "skipped:fresh"),
])
def test_refresh_decision_table(view: HostView, expected: str | None) -> None:
    assert source.refresh_decision(view, T0) == expected


# ==================================================================================================== refresh
async def _refresh(demo: Demo, names: list[str] | None, *, timeout: float = 30,
                   transports: Callable[[HostConfig], Transport] | None = None,
                   config: FleetConfig | None = None) -> dict[str, str]:
    return await source.refresh(config or demo.config, names, clock=demo.clock, timeout=timeout,
                                cache_path=demo.cache, lock_base=demo.lock_base,
                                transports=transports or demo.transports)


async def test_refresh_skips_fresh_disabled_and_busy_hosts_without_a_transport(demo: Demo) -> None:
    demo.clock.advance(30)
    cfg = dataclasses.replace(demo.config, hosts=tuple(
        dataclasses.replace(h, enabled=False) if h.name == "vega" else h for h in demo.config.hosts))
    out = await _refresh(demo, ["k2", "vega", "labstation"], config=cfg)
    assert out == {"k2": "skipped:fresh", "vega": "skipped:disabled", "labstation": "skipped:fresh"}
    demo.clock.advance(120)
    fd, _ = try_instance_lock(instance_lock_file(demo.lock_base, "k2"))
    assert fd is not None
    try:
        assert await _refresh(demo, ["k2"]) == {"k2": "skipped:busy"}
    finally:
        release_instance_lock(fd)
    assert demo.made == []                                   # no transport built, nothing resolved or probed


async def test_refresh_polls_writes_the_cache_and_reads_back_live(demo: Demo) -> None:
    demo.clock.advance(300)
    out = await _refresh(demo, ["k2", "labstation"])
    assert out == {"k2": "refreshed", "labstation": "refreshed"}
    rows = source.read_rows(demo.cache)
    assert rows.snapshots["k2"][0] == (T0 + timedelta(seconds=300)).timestamp()
    v = source.load_view(demo.config, ["k2", "labstation", "vega"], cache_path=demo.cache,
                         now=demo.clock.now(), lock_base=demo.lock_base, outcomes=out)
    assert {h.name: h.source for h in v.hosts} == {"k2": Source.LIVE, "vega": Source.CACHE,
                                                    "labstation": Source.LIVE}
    assert all(not h.problems for h in v.hosts)
    assert try_instance_lock(instance_lock_file(demo.lock_base, "k2"))[0] is not None   # released again


async def test_auth_without_a_master_is_zero_network(demo: Demo) -> None:
    demo.clock.advance(300)
    fakes: dict[str, FakeTransport] = {}

    def factory(cfg: HostConfig) -> Transport:
        fakes[cfg.name] = FakeTransport(cfg.name, master_ok=False)
        return fakes[cfg.name]

    _set_status(demo.cache, "vega", ConnState.HOSTKEY, "hostkey")
    _touch(demo.cache, "vega", T0 - timedelta(hours=1))
    out = await _refresh(demo, ["lynx", "vega"], transports=factory)
    assert out == {"lynx": "skipped:auth", "vega": "skipped:hostkey"}
    for t in fakes.values():
        assert t.calls == ["check_master"] and t.probes == []          # no prepare, no probe
    assert source.read_rows(demo.cache).status["lynx"][1] == "needs_auth"   # nothing written


async def test_auth_with_a_live_master_refreshes(demo: Demo) -> None:
    demo.clock.advance(300)
    _set_status(demo.cache, "k2", ConnState.NEEDS_AUTH, "auth")       # e.g. `runfleet connect k2` since then
    assert await _refresh(demo, ["k2"]) == {"k2": "refreshed"}
    assert source.read_rows(demo.cache).status["k2"][1] in ("ok", "degraded")


async def test_session_transport_needs_auth_is_skipped_without_any_spawn(
        demo: Demo, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.transport.session import SessionTransport
    demo.clock.advance(300)

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("spawned")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    out = await _refresh(demo, ["lynx"], transports=lambda cfg: SessionTransport(cfg))
    assert out == {"lynx": "skipped:auth"}


async def test_refresh_decides_again_once_the_lock_is_taken(demo: Demo,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    demo.clock.advance(300)
    real = source.try_instance_lock_at

    def racing(path: Path, **kw: Any) -> tuple[int | None, str | None, float | None]:
        _touch(demo.cache, "k2", demo.clock.now())                     # another agent refreshed k2 meanwhile
        return real(path, **kw)

    monkeypatch.setattr(source, "try_instance_lock_at", racing)
    assert await _refresh(demo, ["k2"]) == {"k2": "skipped:fresh"}
    assert demo.made == []
    assert try_instance_lock(instance_lock_file(demo.lock_base, "k2"))[0] is not None


async def test_refresh_whose_pid_cannot_be_written_polls_unlocked(demo: Demo,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """K3 in the refresh: the lock taken, our pid unwritable: let go at once and carry on as without a lock file."""
    demo.clock.advance(300)
    monkeypatch.setattr(source, "claim_instance_lock", lambda fd: False)
    assert await _refresh(demo, ["k2"]) == {"k2": "refreshed"}
    assert lock_holder(instance_lock_file(demo.lock_base, "k2")) is None


async def test_flush_that_fails_is_failed_flush(demo: Demo, monkeypatch: pytest.MonkeyPatch) -> None:
    demo.clock.advance(300)

    async def no(self: Cache) -> bool:
        return False
    monkeypatch.setattr(Cache, "flush", no)
    out = await _refresh(demo, ["k2"])
    assert out == {"k2": "failed:flush"}
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=demo.clock.now(),
                         lock_base=demo.lock_base, outcomes=out)
    assert v.hosts[0].source is Source.CACHE
    assert [(p.code, p.reason) for p in v.hosts[0].problems] == [(ProblemCode.STALE, "refresh_skipped")]


class Hanging(FakeTransport):
    async def prepare(self, *, force: bool = False) -> Any:
        await asyncio.sleep(3600)

    async def check_master(self) -> bool:
        if self.host == "lynx":
            await asyncio.sleep(3600)
        return True


async def test_refresh_is_bounded_by_one_deadline(demo: Demo) -> None:
    demo.clock.advance(300)
    started = time.monotonic()
    out = await _refresh(demo, ["k2", "lynx", "vega"], timeout=0.6, transports=lambda cfg: Hanging(cfg.name))
    assert time.monotonic() - started < 5
    assert out == {"k2": "failed:timeout", "lynx": "failed:timeout", "vega": "failed:timeout"}
    for name in ("k2", "vega", "lynx"):
        assert try_instance_lock(instance_lock_file(demo.lock_base, name))[0] is not None


async def test_refresh_timeout_inside_the_poll_is_recorded(demo: Demo) -> None:
    demo.clock.advance(300)
    out = await _refresh(demo, ["k2"], timeout=0.6, transports=lambda cfg: Hanging(cfg.name))
    assert out == {"k2": "failed:timeout"}
    assert source.read_rows(demo.cache).status["k2"][2] == "timeout"   # collect_once recorded it, flushed
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=demo.clock.now(), lock_base=None,
                         outcomes=out)
    assert [p.code for p in v.hosts[0].problems] == [ProblemCode.TIMEOUT]


async def test_refresh_unknown_host_and_unusable_cache(demo: Demo, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        await _refresh(demo, ["orb"])
    demo.clock.advance(300)

    def broken(self: Cache, path: Path | None) -> None:
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(Cache, "__init__", broken)
    assert await _refresh(demo, ["k2"]) == {"k2": "failed:cache"}


def test_poll_outcome_table() -> None:
    from runfleet.models import HostError
    assert source.poll_outcome(HostStatus("k2", ConnState.OK)) == "refreshed"
    assert source.poll_outcome(HostStatus("k2", ConnState.DEGRADED)) == "refreshed"
    assert source.poll_outcome(HostStatus("k2", ConnState.NEEDS_AUTH, HostError(ErrorKind.AUTH, "x"))) == \
        "failed:auth"
    assert source.poll_outcome(HostStatus("k2", ConnState.UNREACHABLE)) == "failed:unreachable"


async def test_openssh_master_check_resolves_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unresolved OpenSSH transport answers check_master True (no plan): has_live_master resolves (`ssh
    -G`, local) and then asks `-O check`; a plan without a master (mux off) has none, without any check."""
    from runfleet.transport.openssh import OpenSSHTransport
    calls: list[str] = []

    class T(OpenSSHTransport):
        async def resolve(self) -> Any:
            calls.append("resolve")
            from runfleet.sshconfig import SshSettings
            from runfleet.transport import mux
            settings = SshSettings.__new__(SshSettings)
            object.__setattr__(settings, "controlpath", None)
            object.__setattr__(settings, "controlpersist", "no")
            self.plan = mux.decide(settings, self.config, tmp_path)
            return None

        async def check_master(self) -> bool:
            calls.append("check")
            return False

    assert not await source.has_live_master(T(HostConfig("k2", "k2", mux="managed")))
    assert calls == ["resolve", "check"]
    calls.clear()
    if not sys.platform.startswith("win"):
        assert not await source.has_live_master(T(HostConfig("k2", "k2", mux="off")))
        assert calls == ["resolve"]


# ====================================================================================================== demo
async def test_demo_view_polls_the_simulated_fleet_in_memory(tmp_path: Path) -> None:
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    v = await source.demo_view(world.fleet_config(), lambda h: DemoTransport(h, world), clock)
    assert v.demo and v.error is None and [h.name for h in v.hosts] == list(DEMO_HOSTS)
    assert all(h.source is Source.DEMO for h in v.hosts)
    lynx = v.host("lynx")
    assert lynx is not None and lynx.state is ConnState.NEEDS_AUTH and lynx.snapshot is None
    for name in ("k2", "vega", "lyra", "labstation"):
        h = v.host(name)
        assert h is not None and h.snapshot is not None and h.health is not None
        assert h.state in (ConnState.OK, ConnState.DEGRADED) and h.health.dest_hash == dest_hash(name)
        assert h.data_at("jobs") is not None
    k2 = v.host("k2")
    assert k2 is not None and k2.kind == "slurm" and k2.health is not None
    assert "slurm.squeue" in k2.health.sections
    assert not list(tmp_path.iterdir())                                 # no cache, no lock file
    only = await source.demo_view(world.fleet_config(), lambda h: DemoTransport(h, world), clock, ["vega"])
    assert [h.name for h in only.hosts] == ["vega"]


# ====================================================================================================== log
def _job(demo_view: Any, logs: tuple[LogRef, ...]) -> Job:
    k2 = demo_view.host("k2")
    job = next(iter(k2.snapshot.jobs.values()))
    return dataclasses.replace(job, logs=logs)


@pytest.fixture
async def k2_view(demo: Demo) -> Any:
    return source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=T0, lock_base=None)


OUT = "/shared/lab/alice/orb/logs/orb_1.out"
ERR = "/shared/lab/alice/orb/logs/orb_1.err"


def test_find_job_uses_wait_matching_and_ended_tasks(k2_view: Any) -> None:
    k2 = k2_view.host("k2")
    some = next(iter(k2.snapshot.jobs.values()))
    assert source.find_job(k2, some.key.id) == some
    assert source.find_job(k2, "999999999") is None
    ended = dataclasses.replace(some, key=dataclasses.replace(some.key, id="p642229-1788696459"))
    with_ended = dataclasses.replace(k2, ended_tasks=(ended,))
    assert source.find_job(with_ended, "p642229-1788696459") == ended


def _task(view: Any, task_id: str) -> Job:
    some = next(iter(view.snapshot.jobs.values()))
    return dataclasses.replace(some, key=dataclasses.replace(some.key, id=task_id), kind=JobKind.TASK,
                               job_id_raw=task_id[1:].split("-")[0])


def test_find_job_tolerates_a_task_start_a_second_off(k2_view: Any) -> None:
    """M2: a workstation task id is p<pid>-<date - etimes>, recomputed by every collector: the id one call printed
    may be a second off the one the next call computes; same pid and a start within START_TOLERANCE_S match."""
    from runfleet.backends.process.tasks import START_TOLERANCE_S, nearest_task, parse_task_id

    k2 = k2_view.host("k2")
    live = _task(k2, "p363942-1788696459")
    host = dataclasses.replace(k2, snapshot=dataclasses.replace(k2.snapshot, jobs={live.key.id: live}))
    assert source.find_job(host, "p363942-1788696460") == live                    # +1 s
    assert source.find_job(host, "p363942-1788696458") == live                    # -1 s
    assert source.find_job(host, f"p363942-{1788696459 + START_TOLERANCE_S}") == live
    assert source.find_job(host, f"p363942-{1788696459 + START_TOLERANCE_S + 1}") is None   # another task
    assert source.find_job(host, "p363943-1788696459") is None                    # another pid
    # an exact hit anywhere (here: an ended task) beats a near one in the snapshot
    ended = _task(k2, "p363942-1788696461")
    both = dataclasses.replace(host, ended_tasks=(ended,))
    assert source.find_job(both, "p363942-1788696461") == ended
    assert source.find_job(both, "p363942-1788696462") == ended                 # the closest start, either set
    assert source.find_job(both, "p363942-1788696458") == live
    # the closest start wins
    near = _task(k2, "p363942-1788696463")
    two = dataclasses.replace(k2, snapshot=dataclasses.replace(k2.snapshot, jobs={live.key.id: live,
                                                                                  near.key.id: near}))
    assert source.find_job(two, "p363942-1788696462") == near
    assert parse_task_id("p1-2") == (1, 2) and parse_task_id("363942") is None and parse_task_id("p1-2x") is None
    assert nearest_task("p363942", {live.key.id: live}) is None                   # a label is not a full id
    slurm = dataclasses.replace(live, kind=JobKind.SLURM)
    assert nearest_task("p363942-1788696460", {live.key.id: slurm}) is None       # only tasks


def test_wait_target_matches_a_near_task_start_unless_told_not_to(k2_view: Any) -> None:
    from runfleet.wait import Target

    live = _task(k2_view.host("k2"), "p642229-1788696459")
    target = Target("k2", "p642229-1788696458", "k2:p642229-1788696458")
    assert target.matches({live.key.id: live}) == [live]
    assert target.matches({live.key.id: live}, near=False) == []


def _fail_factory(cfg: HostConfig) -> Transport:
    raise AssertionError("a transport was built")


async def test_no_log_path_and_no_job_are_zero_probes(k2_view: Any) -> None:
    clock = ManualClock(T0)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    k2 = k2_view.host("k2")
    job = _job(k2_view, (LogRef("stdout", None, pattern="%x_%j.out"),))
    assert await source.read_log(cfg, k2, job, nbytes=4096, clock=clock, timeout=5,
                                 transports=_fail_factory) is None
    assert await source.read_log(cfg, k2, "424242424", nbytes=4096, clock=clock, timeout=5,
                                 transports=_fail_factory) is None


async def test_session_host_waiting_for_login_is_auth_without_network(k2_view: Any) -> None:
    clock = ManualClock(T0)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2", mux="session"),))
    k2 = dataclasses.replace(k2_view.host("k2"), state=ConnState.NEEDS_AUTH, session_transport=True)
    r = await source.read_log(cfg, k2, _job(k2_view, (LogRef("stdout", OUT),)), nbytes=4096, clock=clock,
                              timeout=5, transports=_fail_factory)
    assert r is not None and r.error is not None and r.error.kind is ErrorKind.AUTH and r.path == OUT
    assert r.lines == ()


async def test_read_log_is_one_peek_probe(k2_view: Any) -> None:
    clock = ManualClock(T0)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    t = FakeTransport("k2")
    t.set_result("log.peek", "36|1788696459\nTraceback\nRuntimeError: CUDA oom\n")
    job = _job(k2_view, (LogRef("stdout", OUT), LogRef("stderr", ERR, size=36)))
    r = await source.read_log(cfg, k2_view.host("k2"), job, nbytes=4096, clock=clock, timeout=5,
                              transports=lambda c: t)
    assert r is not None and r.error is None and r.path == ERR
    assert r.lines == ("Traceback", "RuntimeError: CUDA oom")
    assert [p.names for p in t.probes] == [("log.peek",)] and "close" in t.calls
    t.set_result("log.head", "36|1788696459\nfirst\n")
    h = await source.read_log(cfg, k2_view.host("k2"), job, stream="out", nbytes=4096, head=True, clock=clock,
                              timeout=5, transports=lambda c: t)
    assert h is not None and h.path == OUT and h.lines == ("first",) and t.probes[-1].names == ("log.head",)


class SlowProbe(FakeTransport):
    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_read_log_deadline_is_a_timeout_error(k2_view: Any) -> None:
    clock = ManualClock(T0)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    started = time.monotonic()
    r = await source.read_log(cfg, k2_view.host("k2"), _job(k2_view, (LogRef("stdout", OUT),)), nbytes=4096,
                              clock=clock, timeout=0.3, transports=lambda c: SlowProbe("k2"))
    assert time.monotonic() - started < 5
    assert r is not None and r.error is not None and r.error.kind is ErrorKind.TIMEOUT and r.path == OUT


async def test_demo_log_read_end_to_end() -> None:
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config()
    v = await source.demo_view(config, lambda h: DemoTransport(h, world), clock, ["k2"])
    k2 = v.hosts[0]
    assert k2.snapshot is not None
    job = next(j for j in k2.snapshot.jobs.values() if j.state.group is StateGroup.ACTIVE
               and any(r.path and r.path in world.cluster("k2").logs for r in j.logs))
    r = await source.read_log(config, k2, job.key.id, stream="out", nbytes=8192, clock=clock, timeout=10,
                              transports=lambda h: DemoTransport(h, world))
    assert r is not None and r.error is None and (r.lines or r.live)


# ==================================================================================================== windows
def test_source_stays_portable_and_out_of_the_ui() -> None:
    path = Path(source.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names} | \
        {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    posix_only = {"fcntl", "termios", "pwd", "getuid", "geteuid", "killpg", "SIGKILL", "SIGHUP",
                  "start_new_session", "setsid", "fork"}
    assert not (names | imported) & posix_only
    assert not imported & {"textual", "rich", "runfleet.ui", "runfleet.status", "runfleet.core.actions",
                           "runfleet.core.attach"}
    code = ("import sys, runfleet.agent.source as s; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('rich', 'textual') or "
            "m.startswith('runfleet.ui')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]"


# ========================================================================= refresh keeps good data (D1, D2)
class SqueueDown(DemoTransport):
    """The demo host, except that squeue fails (slurmctld unreachable): the listing section errors."""

    def _answer(self, sim: Any, section: Any, now: float, purpose: str) -> Any:
        from runfleet.demo.transport import Out
        if section.name == "slurm.squeue":
            return Out("", 1, "slurm_load_jobs error: Unable to contact slurm controller (connect failure)")
        return super()._answer(sim, section, now, purpose)


async def test_refresh_whose_listing_fails_keeps_the_cached_jobs_and_their_age(demo: Demo) -> None:
    before = source.read_rows(demo.cache)
    cached_jobs = json.loads(before.snapshots["k2"][1])["jobs"]
    squeue_ok = json.loads(before.health["k2"][1])["sections"]["slurm.squeue"]["last_ok"]
    assert cached_jobs and squeue_ok is not None
    demo.clock.advance(300)
    out = await _refresh(demo, ["k2"], transports=lambda cfg: SqueueDown(cfg, demo.world))
    assert out == {"k2": "refreshed"}                                 # the host answered (degraded)
    rows = source.read_rows(demo.cache)
    kept = json.loads(rows.snapshots["k2"][1])["jobs"]
    assert kept.keys() == cached_jobs.keys()                          # not wiped to what accounting lists
    running = [k for k, j in cached_jobs.items() if j["state"] == "R"]
    assert sum(kept[k]["state"] == "R" for k in running) >= len(running) - 2     # live rows kept (fresh
    #                                                       accounting wins for a job that ended since)
    squeue = json.loads(rows.health["k2"][1])["sections"]["slurm.squeue"]
    assert squeue["ok"] is False and squeue["last_ok"] == squeue_ok                    # last success kept
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=demo.clock.now(),
                         lock_base=demo.lock_base, outcomes=out)
    k2 = v.hosts[0]
    assert k2.snapshot is not None and k2.snapshot.jobs.keys() == cached_jobs.keys()
    assert k2.data_at("jobs") == datetime.fromisoformat(squeue_ok)                     # the old age, not now
    assert k2.health is not None and [s.name for s in k2.health.failing()] == ["slurm.squeue"]


class SacctDown(DemoTransport):
    """The demo host, except that accounting fails (slurmdbd unreachable): squeue answers, sacct does not."""

    def _answer(self, sim: Any, section: Any, now: float, purpose: str) -> Any:
        from runfleet.demo.transport import Out
        if section.name.startswith("slurm.sacct"):
            return Out("", 1, "sacct: error: slurmdbd: connection refused")
        return super()._answer(sim, section, now, purpose)


async def test_refresh_whose_accounting_fails_keeps_the_cached_finished_jobs(demo: Demo) -> None:
    """C1: squeue fine, sacct down — a fresh collector lists only squeue's rows, which used to wipe every
    cached finished job (the failed / done ones an agent asks about first)."""
    before = source.read_rows(demo.cache)
    cached = json.loads(before.snapshots["k2"][1])["jobs"]
    finished = {k for k, j in cached.items() if JobState(j["state"]).is_terminal}
    sacct_ok = json.loads(before.health["k2"][1])["sections"]["slurm.sacct"]["last_ok"]
    assert finished and sacct_ok is not None
    demo.clock.advance(300)
    out = await _refresh(demo, ["k2"], transports=lambda cfg: SacctDown(cfg, demo.world))
    assert out == {"k2": "refreshed"}
    rows = source.read_rows(demo.cache)
    kept = json.loads(rows.snapshots["k2"][1])["jobs"]
    assert finished <= kept.keys()                                    # every cached finished job is still there
    assert all(kept[k] == cached[k] for k in finished)                # as cached (no fresh row for them)
    live = {k for k, j in kept.items() if not JobState(j["state"]).is_terminal}
    assert live and all(kept[k]["source"] == "squeue" for k in live)  # the live rows are this poll's
    sacct = json.loads(rows.health["k2"][1])["sections"]["slurm.sacct"]
    assert sacct["ok"] is False and sacct["last_ok"] == sacct_ok      # accounting's age: its last success
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=demo.clock.now(),
                         lock_base=demo.lock_base, outcomes=out)
    k2 = v.hosts[0]
    assert k2.snapshot is not None and finished <= k2.snapshot.jobs.keys()
    assert k2.data_at("history") == datetime.fromisoformat(sacct_ok)


async def test_refresh_attempted_by_anyone_under_the_floor_is_rate_limited(demo: Demo) -> None:
    demo.clock.advance(300)
    lock = instance_lock_file(demo.lock_base, "k2")
    lock.write_text("4242")                           # a pid written: an attempt (an empty file is none, K4)
    recent = demo.clock.now().timestamp() - (REFRESH_FLOOR - 5)
    os.utime(lock, (recent, recent))                  # someone's refresh attempt 55 s ago (it failed: data old)
    out = await _refresh(demo, ["k2", "labstation"])
    assert out == {"k2": "skipped:rate_limited", "labstation": "refreshed"}
    assert demo.made == ["labstation"]                                # no transport for k2: no ssh at all
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=demo.clock.now(),
                         lock_base=demo.lock_base, outcomes=out)
    assert [(p.code, p.reason) for p in v.hosts[0].problems] == [(ProblemCode.STALE, "rate_limited")]
    old = demo.clock.now().timestamp() - (REFRESH_FLOOR + 5)
    os.utime(lock, (old, old))
    assert await _refresh(demo, ["k2"]) == {"k2": "refreshed"}


async def test_two_refreshes_seconds_apart_poll_once(tmp_path: Path) -> None:
    """E2E-7: the second call within REFRESH_FLOOR builds no transport, even though the first one failed (the
    data stayed old, so "fresh" would not stop it). Real wall-clock times: the lock file's mtime is real."""
    clock = ManualClock(datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=300))
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    d = Demo(world, clock, world.fleet_config(), tmp_path / "cache.sqlite", tmp_path / "instance.lock")
    await _fill(d, ("k2",))
    clock.advance(300)
    first = await _refresh(d, ["k2"], timeout=0.6, transports=lambda cfg: Hanging(cfg.name))
    assert first == {"k2": "failed:timeout"}
    clock.advance(5)
    built: list[str] = []

    def factory(cfg: HostConfig) -> Transport:
        built.append(cfg.name)
        return DemoTransport(cfg, world)
    assert await _refresh(d, ["k2"], transports=factory) == {"k2": "skipped:rate_limited"}
    assert built == []


def _wall_demo(tmp_path: Path, ago: float = 300) -> Demo:
    """A demo whose clock runs `ago` s behind the wall clock (lock files carry real mtimes): fill, then advance."""
    clock = ManualClock(datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=ago))
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    return Demo(world, clock, world.fleet_config(), tmp_path / "cache.sqlite", tmp_path / "instance.lock")


class SlowDemo(DemoTransport):
    """The demo host, its probes taking `delay` wall seconds (a refresh holding the lock meanwhile)."""

    def __init__(self, cfg: HostConfig, world: DemoWorld, delay: float) -> None:
        super().__init__(cfg, world)
        self.delay = delay

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        await asyncio.sleep(self.delay)
        return await super().probe(script, timeout=timeout)


async def test_a_refresh_while_another_one_runs_is_in_progress_not_an_idle_tui(tmp_path: Path) -> None:
    """C3: refresh B starts while refresh A holds k2's lock. It used to read the held lock as an idle TUI
    ("ask the user to press a key"); a lock taken < REFRESH_FLOOR ago is a refresh at work."""
    d = _wall_demo(tmp_path, 300 + STALE_AFTER)
    await _fill(d, ("k2",))
    d.clock.advance(300 + STALE_AFTER)                         # old enough that an idle TUI would be "tui_idle"
    lock = instance_lock_file(d.lock_base, "k2")
    first = asyncio.ensure_future(_refresh(d, ["k2"], transports=lambda cfg: SlowDemo(cfg, d.world, 0.3)))
    for _ in range(200):
        if lock_holder(lock) is not None:
            break
        await asyncio.sleep(0.01)
    assert lock_holder(lock) == str(os.getpid())
    built: list[str] = []

    def factory(cfg: HostConfig) -> Transport:
        built.append(cfg.name)
        return DemoTransport(cfg, d.world)
    second = await _refresh(d, ["k2"], transports=factory)
    assert second == {"k2": "skipped:in_progress"} and built == []
    v = source.load_view(d.config, ["k2"], cache_path=d.cache, now=d.clock.now(), lock_base=d.lock_base,
                         outcomes=second)
    assert [(p.code, p.reason) for p in v.hosts[0].problems] == [(ProblemCode.STALE, "rate_limited")]
    assert await first == {"k2": "refreshed"}
    fd, _ = try_instance_lock(lock)                            # a TUI holding the lock since long ago: busy
    assert fd is not None
    old = d.clock.now().timestamp() - (REFRESH_FLOOR + 5)
    os.utime(lock, (old, old))
    try:
        _touch(d.cache, "k2", d.clock.now() - timedelta(seconds=STALE_AFTER))     # data old again
        assert await _refresh(d, ["k2"], transports=factory) == {"k2": "skipped:busy"}
    finally:
        release_instance_lock(fd)


async def test_a_refresh_that_came_and_went_before_the_lock_is_rate_limited(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """C3 race: between B's rate-limit check and B taking the lock, A takes it, fails (data stays old) and lets
    go. The lock's previous mtime, read under B's lock, tells: B lets go too, without a transport."""
    d = _wall_demo(tmp_path)
    await _fill(d, ("k2",))
    d.clock.advance(300)
    real = source.try_instance_lock_at

    seen: list[int] = []

    def racing(path: Path, **kw: Any) -> tuple[int | None, str | None, float | None]:
        fd, _ = try_instance_lock(path)                        # refresh A, just now
        assert fd is not None
        release_instance_lock(fd)
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns - 7_000_000))   # distinct from now
        seen.append(path.stat().st_mtime_ns)
        return real(path, **kw)

    monkeypatch.setattr(source, "try_instance_lock_at", racing)
    out = await _refresh(d, ["k2"])
    assert out == {"k2": "skipped:rate_limited"} and d.made == []
    lock = instance_lock_file(d.lock_base, "k2")
    assert lock_holder(lock) is None
    assert lock.stat().st_mtime_ns == seen[0]                  # K5: A's attempt time, not rewritten by B's skip
    assert lock.read_text() == str(os.getpid())                # A's pid (the same process here), not truncated


async def test_locks_are_released_only_after_the_cache_is_closed(demo: Demo,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex-12: also when the deadline cancels the refresh while the last flush is still writing."""
    demo.clock.advance(300)
    lock = instance_lock_file(demo.lock_base, "k2")
    seen: list[str | None] = []
    real = Cache.close_sync

    def slow_close(self: Cache) -> bool:
        time.sleep(0.4)                                  # a slow disk: the deadline passes meanwhile
        seen.append(lock_holder(lock))
        return real(self)
    monkeypatch.setattr(Cache, "close_sync", slow_close)
    out = await _refresh(demo, ["k2"], timeout=0.6, transports=lambda cfg: Hanging(cfg.name))
    assert out == {"k2": "failed:timeout"}
    for _ in range(100):                                 # the closing thread finishes on its own
        if seen and lock_holder(lock) is None:
            break
        await asyncio.sleep(0.05)
    assert seen == [str(os.getpid())]                    # still ours while the cache was being closed
    assert lock_holder(lock) is None


# ============================================================================ identity and age (Codex-9/10)
async def test_health_row_older_than_its_snapshot_is_not_trusted(demo: Demo) -> None:
    rows = source.read_rows(demo.cache)
    at = rows.health["k2"][0]
    _db_exec(demo.cache, "UPDATE host_health SET json = json_set(json, '$.dest_hash', 'feedfacecafe') "
                         "WHERE host = 'k2'")                     # it names another destination …
    later = at + source.HEALTH_SKEW + 60
    _db_exec(demo.cache, "UPDATE snapshots SET taken_at = ? WHERE host = 'k2'", (later,))
    now = datetime.fromtimestamp(later, UTC) + timedelta(seconds=30)
    v = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=now, lock_base=None)
    k2 = v.hosts[0]                                     # … but an older RunFleet wrote the snapshot since
    assert k2.health is None and k2.snapshot is not None
    assert k2.data_at("jobs") == k2.snapshot.taken_at                                   # approximate
    assert [(p.code, p.reason) for p in k2.problems] == [(ProblemCode.STALE, "age_unknown")]
    _db_exec(demo.cache, "UPDATE snapshots SET taken_at = ? WHERE host = 'k2'", (at + source.HEALTH_SKEW - 1,))
    close = source.load_view(demo.config, ["k2"], cache_path=demo.cache, now=now, lock_base=None).hosts[0]
    assert [p.code for p in close.problems] == [ProblemCode.CONFIG_MISMATCH]       # within the skew: trusted


# ===================================================================================== reading (Codex-15)
async def test_read_rows_filters_hosts_and_old_ended_tasks_in_sql(demo: Demo) -> None:
    some = Job(key=JobKey("labstation", "p1-1"), kind=JobKind.TASK, name="old", user="alice",
               state=JobState.EXITED, raw_state="EX", job_id_raw="p1")
    from runfleet.models import to_json
    old_t = (T0 - timedelta(days=8)).timestamp()
    new_t = (T0 - timedelta(hours=1)).timestamp()
    for jid, t in (("p1-1", old_t), ("p2-2", new_t)):
        doc = json.dumps(to_json(dataclasses.replace(some, key=JobKey("labstation", jid))))
        _db_exec(demo.cache, "INSERT OR REPLACE INTO ended_tasks VALUES ('labstation', ?, ?, ?)", (jid, t, doc))
    rows = source.read_rows(demo.cache, ["labstation", "k2"], ended_since=(T0 - timedelta(days=7)).timestamp())
    assert set(rows.snapshots) <= {"labstation", "k2"} and set(rows.health) <= {"labstation", "k2"}
    assert set(rows.status) <= {"labstation", "k2"} and set(rows.profiles) <= {"labstation", "k2"}
    assert len(rows.ended["labstation"]) == 1 and '"p2-2"' in rows.ended["labstation"][0]
    assert source.read_rows(demo.cache, []).snapshots == {}
    everything = source.read_rows(demo.cache)
    assert {"vega", "lyra"} <= set(everything.snapshots) and len(everything.ended["labstation"]) >= 2
    v = source.load_view(demo.config, ["labstation"], cache_path=demo.cache, now=T0, lock_base=None)
    assert [j.key.id for j in v.hosts[0].ended_tasks if j.name == "old"] == ["p2-2"]


def test_a_large_cache_renders_within_a_second(tmp_path: Path) -> None:
    """12 hosts x 600 jobs: cache -> FleetView -> brief / jobs documents, generously < 1 s."""
    from runfleet.agent import view
    from tests.cli.fakes import job

    names = [f"node{i:02d}" for i in range(12)]
    cfg = FleetConfig(hosts=tuple(HostConfig(n, n, backend="slurm") for n in names))
    path = tmp_path / "cache.sqlite"
    cache = Cache(path)
    states = (JobState.RUNNING, JobState.PENDING, JobState.COMPLETED, JobState.FAILED)
    for n in names:
        jobs = {}
        for k in range(600):
            j = dataclasses.replace(job(n, str(4000000 + k), states[k % 4]), is_mine=k % 3 != 0,
                                    submit_time=T0 - timedelta(hours=2), observed_at=T0)
            jobs[j.key.id] = j
        cache.put_snapshot(n, HostSnapshot(n, T0, None, jobs, (), (), (), (), None))
        st = HostStatus(name=n, state=ConnState.OK, last_ok=T0, sections={
            "slurm.squeue": SectionStatus("slurm.squeue", "ok", True, 0, 80, T0)})
        cache.put_health(n, st, dest_hash=dest_hash(n), remote_user="alice", at=T0)
    assert cache.flush_sync()
    cache._db.close()
    started = time.perf_counter()
    fv = source.load_view(cfg, None, cache_path=path, now=T0 + timedelta(seconds=30), lock_base=None)
    view.brief(fv)
    view.jobs(fv, view.JobFilter())
    took = time.perf_counter() - started
    assert sum(len(h.snapshot.jobs) for h in fv.hosts if h.snapshot is not None) == 7200
    assert took < 1.0, f"{took:.2f} s"


async def test_a_cache_open_cut_by_the_deadline_is_closed_before_the_locks_go(demo: Demo,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    demo.clock.advance(300)
    lock = instance_lock_file(demo.lock_base, "k2")
    real_init, real_close = Cache.__init__, Cache.close_sync
    seen: list[str | None] = []

    def slow_open(self: Cache, path: Path | None) -> None:
        time.sleep(0.5)                                  # a locked database: the open retries
        real_init(self, path)

    def close(self: Cache) -> bool:
        seen.append(lock_holder(lock))
        return real_close(self)
    monkeypatch.setattr(Cache, "__init__", slow_open)
    monkeypatch.setattr(Cache, "close_sync", close)
    assert await _refresh(demo, ["k2"], timeout=0.3) == {"k2": "failed:timeout"}
    for _ in range(100):
        if seen and lock_holder(lock) is None:
            break
        await asyncio.sleep(0.05)
    assert seen == [str(os.getpid())] and lock_holder(lock) is None     # closed while ours, then released


def test_an_abandoned_cache_open_survives_the_loop_shutting_down(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's asyncio.run ends right after the deadline: the worker still closes the cache, then unlocks."""
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    d = Demo(world, clock, world.fleet_config(), tmp_path / "cache.sqlite", tmp_path / "instance.lock")
    asyncio.run(_fill(d, ("k2",)))
    clock.advance(300)
    lock = instance_lock_file(d.lock_base, "k2")
    real_init, real_close = Cache.__init__, Cache.close_sync
    seen: list[str | None] = []

    def slow_open(self: Cache, path: Path | None) -> None:
        time.sleep(0.5)
        real_init(self, path)

    def close(self: Cache) -> bool:
        seen.append(lock_holder(lock))
        return real_close(self)
    monkeypatch.setattr(Cache, "__init__", slow_open)
    monkeypatch.setattr(Cache, "close_sync", close)
    assert asyncio.run(_refresh(d, ["k2"], timeout=0.2)) == {"k2": "failed:timeout"}
    for _ in range(100):
        if seen and lock_holder(lock) is None:
            break
        time.sleep(0.05)
    assert seen == [str(os.getpid())] and lock_holder(lock) is None


def test_the_last_close_runs_even_when_asyncio_run_ends_at_the_deadline(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    d = Demo(world, clock, world.fleet_config(), tmp_path / "cache.sqlite", tmp_path / "instance.lock")
    asyncio.run(_fill(d, ("k2",)))
    clock.advance(300)
    lock = instance_lock_file(d.lock_base, "k2")
    real_close = Cache.close_sync
    seen: list[str | None] = []

    def slow_close(self: Cache) -> bool:
        time.sleep(0.3)
        seen.append(lock_holder(lock))
        return real_close(self)
    monkeypatch.setattr(Cache, "close_sync", slow_close)
    out = asyncio.run(_refresh(d, ["k2"], timeout=0.5, transports=lambda cfg: Hanging(cfg.name)))
    assert out == {"k2": "failed:timeout"}
    for _ in range(100):
        if seen and lock_holder(lock) is None:
            break
        time.sleep(0.05)
    assert seen == [str(os.getpid())] and lock_holder(lock) is None
