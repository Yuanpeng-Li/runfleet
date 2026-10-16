"""[WP1] FleetService composed with FakeTransport + stub collectors (§11.1 WP1 acceptance): cadence, politeness,
backoff, actions, details, logs, idle handling, snapshot collection, hosts import and doctor."""

from __future__ import annotations

import asyncio
import json
import os
import tomllib
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from runfleet import status
from runfleet.config import FleetConfig, GeneralSettings, HostConfig
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.errors import ActionError, ConfigError, RunFleetError
from runfleet.models import (ConnState, ErrorKind, HostError, HostSnapshot, HostStatus, JobKey, LogRef, Visibility,
                             to_json)
from runfleet.transport.base import Transport
from runfleet.transport.fake import FakeStream, FakeTransport, result
from runfleet.transport.local import LocalTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, run_for, settle, stub_caps

K2 = HostConfig(name="k2", ssh="k2", backend="slurm")
VEGA = HostConfig(name="vega", ssh="vega", backend="slurm")
LYNX = HostConfig(name="lynx", ssh="lynx", backend="workstation")


class World:
    """One FleetService over FakeTransports (one per host) with a ManualClock."""

    def __init__(self, hosts: tuple[HostConfig, ...] = (K2, LYNX), *, general: GeneralSettings | None = None,
                 cache: Cache | None = None, mux: str = "none", latency: float = 0.0, tmp: Path | None = None) -> None:
        self.clock = ManualClock(CLOCK_START)
        self.transports: dict[str, FakeTransport] = {}
        self.mux, self.latency = mux, latency
        cfg = FleetConfig(hosts=hosts, general=general or GeneralSettings())
        self.svc = FleetService(cfg, self.factory, self.clock, cache)
        self.svc.hooks = STUB_HOOKS
        if tmp is not None:
            self.svc.lock_path = tmp / "instance.lock"
        for h in hosts:
            self.factory(h)

    def factory(self, host: HostConfig) -> Transport:
        t = self.transports.get(host.name)
        if t is None:
            t = FakeTransport(host.name, mux_mode=self.mux, clock=self.clock, latency=self.latency)  # type: ignore[arg-type]
            t.set_result("detect.id", "user=me")
            t.set_result("slurm.squeue", "101 R\n102 PD")
            t.set_result("process.ps", "642229 0")
            t.set_result("nvidia.gpu", "0 90\n1 5")
            self.transports[host.name] = t
        return t

    def status(self, host: str):  # type: ignore[no-untyped-def]
        return self.svc.store.hosts[host].status


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[Callable[..., World]]:
    made: list[World] = []

    def make(*hosts: HostConfig, **kw: object) -> World:
        w = World(hosts or (K2, LYNX), tmp=tmp_path, **kw)   # type: ignore[arg-type]
        made.append(w)
        return w

    yield make
    for w in made:
        await w.svc.stop()


# ---------------------------------------------------------------- lifecycle, cadence, politeness
async def test_start_polls_every_host_into_the_store(world: Callable[..., World]) -> None:
    w = world()
    await w.svc.start()
    await settle()
    assert w.status("k2").state is ConnState.OK and w.status("lynx").state is ConnState.OK
    assert sorted(str(j.key) for j in w.svc.store.jobs()) == ["k2:101", "k2:102", "lynx:p642229-1000"]
    assert [g.name for g in w.svc.store.gpus(["lynx"])] == ["annotated", "annotated"]
    assert w.status("k2").tick_interval == 15 and w.status("lynx").tick_interval == 5
    await w.svc.stop()
    n = len(w.transports["k2"].probes)
    await run_for(w.clock, 60)
    assert len(w.transports["k2"].probes) == n                        # no pollers left running


async def test_politeness_through_the_service(world: Callable[..., World]) -> None:
    w = world(K2, VEGA)
    await w.svc.start()
    w.svc.note_user_activity()
    for _ in range(60):                                                # an active user: no idle throttling
        await run_for(w.clock, 60, step=1.0)
        w.svc.note_user_activity()
    for name in ("k2", "vega"):
        polls = w.transports[name].count("poll")
        assert 225 <= polls <= 240, (name, polls)


async def test_hidden_hosts_slow_down_and_visibility_restores(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    w.svc.set_visibility({"k2": Visibility.HIDDEN})
    await run_for(w.clock, 30)
    assert w.status("k2").tick_interval == 45
    w.svc.set_visibility({"k2": Visibility.VISIBLE})
    await run_for(w.clock, 20)
    assert w.status("k2").tick_interval == 15


async def test_rate_multiplier_and_app_focus(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    w.svc.set_rate_multiplier(2.0)
    await run_for(w.clock, 20)
    assert w.status("k2").tick_interval == 30
    w.svc.set_rate_multiplier(0.1)                                     # clamped to 0.5 -> floor 10
    await run_for(w.clock, 40)
    assert w.status("k2").tick_interval == 10
    w.svc.set_rate_multiplier(1.0)
    w.svc.set_app_focus(False)                                         # AppBlur -> ×4
    await run_for(w.clock, 40)
    assert w.status("k2").tick_interval == 60
    w.svc.set_app_focus(True)
    await run_for(w.clock, 70)
    assert w.status("k2").tick_interval == 15


async def test_idle_throttle_and_idle_pause(world: Callable[..., World]) -> None:
    w = world(K2, general=GeneralSettings(idle_after=600, idle_pause_after=7200))
    await w.svc.start()
    await run_for(w.clock, 610, step=1.0)
    assert w.status("k2").tick_interval == 60                          # no input for 10 min -> ×4
    await run_for(w.clock, 6700, step=5.0)
    assert w.svc.store.idle_paused
    n = len(w.transports["k2"].probes)
    await run_for(w.clock, 600, step=5.0)
    assert len(w.transports["k2"].probes) == n                         # paused: nothing at all
    w.svc.note_user_activity()
    await run_for(w.clock, 4, step=1.0)
    assert not w.svc.store.idle_paused
    assert len(w.transports["k2"].probes) == n + 1                     # immediate refresh on input
    assert w.status("k2").tick_interval == 15


async def test_manual_refresh_floor(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await run_for(w.clock, 5)
    n = w.transports["k2"].count("poll")
    w.svc.refresh()
    await run_for(w.clock, 1)
    w.svc.refresh(["k2"])                                              # < 3 s per host: ignored
    await run_for(w.clock, 1)
    assert w.transports["k2"].count("poll") == n + 1
    assert "slurm.squeue" in w.transports["k2"].probes[-1].names


async def test_backoff_and_recovery_through_the_service(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    w.transports["k2"].fail_next(HostError(ErrorKind.REFUSED, "connection failed"), times=3)
    seen: set[ConnState] = set()
    for _ in range(200):
        await run_for(w.clock, 1)
        seen.add(w.status("k2").state)
        if w.status("k2").state is ConnState.OK and ConnState.UNREACHABLE in seen:
            break
    assert {ConnState.RETRYING, ConnState.UNREACHABLE, ConnState.OK} <= seen
    assert w.svc.store.hosts["k2"].snapshot is not None and len(w.svc.store.hosts["k2"].snapshot.jobs) == 2


async def test_disable_and_enable_host(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    w.svc.set_host_enabled("k2", False)
    await settle()
    assert w.status("k2").state is ConnState.DISABLED
    n = len(w.transports["k2"].probes)
    await run_for(w.clock, 60)
    assert len(w.transports["k2"].probes) == n
    assert w.svc.store.hosts["k2"].snapshot is not None                # data kept, shown stale
    w.svc.set_host_enabled("k2", True)
    await settle()
    assert w.status("k2").state is ConnState.OK and len(w.transports["k2"].probes) > n


async def test_needs_auth_connect_flow(world: Callable[..., World]) -> None:
    w = world(K2)
    w.transports["k2"].prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    await w.svc.start()
    await settle()
    assert w.status("k2").state is ConnState.NEEDS_AUTH
    assert w.svc.connect_argv("k2") == ["fake-ssh", "-fN", "k2"]
    await w.svc.after_connect("k2", 0)
    await settle()
    assert w.status("k2").state is ConnState.OK


# ---------------------------------------------------------------- actions (§6.3)
async def test_actions_plan_execute_and_wake(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await run_for(w.clock, 4)
    keys = [JobKey("k2", "101"), JobKey("k2", "102")]
    specs = w.svc.actions_for(keys)
    assert [s.id for s in specs] == ["slurm.cancel"] and specs[0].disabled_reason is None
    plans = w.svc.plan("slurm.cancel", keys)
    assert len(plans) == 1 and plans[0].preview == "scancel 101 102"
    t = w.transports["k2"]
    before = t.count("poll")
    results = await w.svc.execute(plans)
    assert [r.ok for r in results] == [True, True] and t.count("action") == 1
    assert results[0].started_at is not None and results[0].duration_s is not None
    assert list(w.svc.action_log()) == results
    await run_for(w.clock, 4)
    assert t.count("poll") == before + 1                               # woken ≥ 3 s after the last probe
    woke = t.probes[-1].names
    assert "slurm.squeue" in woke and "slurm.sacct" in woke            # {"slurm.jobs", "slurm.history"}


async def test_action_failure_is_reported_not_raised(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    plans = w.svc.plan("slurm.cancel", [JobKey("k2", "101")])
    w.transports["k2"].fail_next(HostError(ErrorKind.TIMEOUT, "probe timed out after 25 s"))
    results = await w.svc.execute(plans)
    assert len(results) == 1 and not results[0].ok and "timed out" in results[0].message


async def test_action_disabled_reasons(world: Callable[..., World]) -> None:
    w = world(K2, LYNX, general=GeneralSettings(readonly=True))
    await w.svc.start()
    await settle()
    specs = w.svc.actions_for([JobKey("k2", "101")])
    assert specs[0].disabled_reason == "read-only mode"
    with pytest.raises(ActionError, match="read-only"):
        w.svc.plan("slurm.cancel", [JobKey("k2", "101")])

    w2 = world(K2, LYNX)
    await w2.svc.start()
    await settle()
    w2.transports["k2"].fail_next(HostError(ErrorKind.AUTH, "authentication needed"))
    w2.svc.refresh(["k2"])
    await run_for(w2.clock, 4)
    assert w2.status("k2").state is ConnState.NEEDS_AUTH
    assert w2.svc.actions_for([JobKey("k2", "101")])[0].disabled_reason == "k2: needs auth"
    with pytest.raises(ActionError, match="needs auth"):
        w2.svc.plan("slurm.cancel", [JobKey("k2", "101")])
    # stale data: last ok long ago although the state is still OK
    hs = w2.svc.store.hosts["lynx"]
    hs.status = replace(hs.status, last_ok=w2.clock.now() - timedelta(minutes=4), tick_interval=5.0)
    assert w2.svc.actions_for([JobKey("lynx", "p642229-1000")])[0].disabled_reason == \
        "data is 4m old — refresh first"
    # mixed selection: enabled if any host can run it
    hs.status = replace(hs.status, last_ok=w2.clock.now())
    merged = w2.svc.actions_for([JobKey("k2", "101"), JobKey("lynx", "p642229-1000")])
    assert {s.id: s.disabled_reason for s in merged} == {"slurm.cancel": "k2: needs auth", "task.terminate": None}


async def test_actions_gate_on_the_job_section_not_host_last_ok(world: Callable[..., World]) -> None:
    """squeue failing while other sections answer keeps the host's last_ok fresh; its rows must not stay
    actionable."""
    from runfleet.models import SectionStatus
    w = world(K2)
    await w.svc.start()
    await settle()
    hs = w.svc.store.hosts["k2"]
    now = w.clock.now()
    failed = SectionStatus("slurm.squeue", "failed", ok=False, rc=1, ms=5, last_ok=now - timedelta(seconds=1000))
    hs.status = replace(hs.status, state=ConnState.DEGRADED, last_ok=now, tick_interval=15.0,
                        sections={**hs.status.sections, "slurm.squeue": failed})
    assert w.svc.actions_for([JobKey("k2", "101")])[0].disabled_reason == "data is 16m old — slurm.squeue failing"
    with pytest.raises(ActionError, match="slurm.squeue failing"):
        w.svc.plan("slurm.cancel", [JobKey("k2", "101")])
    recent = replace(failed, last_ok=now - timedelta(seconds=10))     # one failed run: rows still current
    hs.status = replace(hs.status, sections={**hs.status.sections, "slurm.squeue": recent})
    assert w.svc.actions_for([JobKey("k2", "101")])[0].disabled_reason is None


async def test_action_rechecks_readonly_after_the_master_wait(world: Callable[..., World]) -> None:
    w = world(K2, mux="managed")
    await w.svc.start()
    await settle()
    plans = w.svc.plan("slurm.cancel", [JobKey("k2", "101")])
    t = w.transports["k2"]
    gate = asyncio.Event()
    orig_prepare = t.prepare

    async def slow_prepare(*, force: bool = False) -> HostError | None:
        await gate.wait()
        return await orig_prepare(force=force)

    t.prepare = slow_prepare                                           # type: ignore[method-assign]
    t.master_ok = False
    running = asyncio.ensure_future(w.svc.execute(plans))
    await settle()
    w.svc.set_general(save=False, readonly=True)
    gate.set()
    results = await running
    assert t.count("action") == 0
    assert not results[0].ok and results[0].message == "read-only mode"


async def test_on_demand_master_prepares_share_max_parallel_ssh() -> None:
    live = peak = 0

    class SlowPrepare(FakeTransport):
        async def prepare(self, *, force: bool = False) -> HostError | None:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return None

    hosts = tuple(HostConfig(f"h{i}", f"h{i}") for i in range(4))
    ts = {h.name: SlowPrepare(h.name, mux_mode="managed", master_ok=False) for h in hosts}
    svc = FleetService(FleetConfig(hosts=hosts, general=GeneralSettings(max_parallel_ssh=1)),
                       lambda h: ts[h.name], ManualClock(CLOCK_START), None)
    svc.hooks = STUB_HOOKS
    await asyncio.gather(*(svc.peek_log(h.name, LogRef("stdout", "/w/x.out")) for h in hosts),
                         *(svc.doctor(h.name) for h in hosts))
    assert peak == 1


async def test_replan_on_param_change_is_pure(world: Callable[..., World]) -> None:
    w = world(LYNX)
    await w.svc.start()
    await settle()
    key = [JobKey("lynx", "p642229-1000")]
    n = len(w.transports["lynx"].probes)
    p1 = w.svc.plan("task.terminate", key, {"grace_s": 10})
    p2 = w.svc.plan("task.terminate", key, {"grace_s": 30})
    assert p1[0].preview != p2[0].preview and p2[0].sections[0].timeout == 45
    assert len(w.transports["lynx"].probes) == n                       # planning does no I/O


async def test_on_demand_lane_is_one_on_a_user_master(world: Callable[..., World]) -> None:
    w = world(K2, mux="user", latency=1.0)
    await w.svc.start()
    await run_for(w.clock, 2)
    t = w.transports["k2"]
    t.max_in_flight = 0
    tasks = [asyncio.ensure_future(w.svc.job_detail(JobKey("k2", "101"), max_age_s=0)),
             asyncio.ensure_future(w.svc.peek_log("k2", LogRef("stdout", "/w/101.out"))),
             asyncio.ensure_future(w.svc.fetch_log("k2", LogRef("stdout", "/w/101.out")))]
    await run_for(w.clock, 5, step=0.25)
    done = await asyncio.gather(*tasks, return_exceptions=True)
    assert not any(isinstance(d, BaseException) for d in done), done
    assert t.max_in_flight <= 2                                         # ≤ 1 on-demand + ≤ 1 poll at a time
    on_demand = [p for p in t.probes if p.script.purpose in ("detail", "log")]
    starts = sorted(p.at for p in on_demand if p.at is not None)
    assert all(b - a >= 1.0 for a, b in zip(starts, starts[1:], strict=False))   # serialized
    Path(done[2]).unlink()                                             # type: ignore[arg-type]


# ---------------------------------------------------------------- details, logs, argv
async def test_job_detail_cache_and_errors(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    t = w.transports["k2"]
    t.set_result("slurm.steps", "101.batch COMPLETED")
    d1 = await w.svc.job_detail(JobKey("k2", "101"))
    assert d1.fields["steps"] == "101.batch COMPLETED"
    await w.svc.job_detail(JobKey("k2", "101"))
    assert t.count("detail") == 1                                      # max_age_s = 30
    w.clock.advance(31)
    t.fail_next(HostError(ErrorKind.TIMEOUT, "probe timed out"))
    d3 = await w.svc.job_detail(JobKey("k2", "101"))
    assert d3.errors and d3.errors[0].kind is ErrorKind.TIMEOUT and d3.job.key.id == "101"
    with pytest.raises(RunFleetError):
        await w.svc.job_detail(JobKey("k2", "999"))


async def test_stream_log_through_service(world: Callable[..., World]) -> None:
    w = world(K2, mux="user")
    await w.svc.start()
    await settle()
    w.transports["k2"].set_stream("log.tail", lambda spec: FakeStream([b"a\nb\r c"], hold_open=True))
    s1 = await w.svc.stream_log("k2", LogRef("stdout", "/w/101.out"))
    first = await s1.updates().__anext__()
    assert first.lines == ("a",) and first.live == " c"
    s2 = await w.svc.stream_log("k2", LogRef("stdout", "/w/102.out"))
    assert w.transports["k2"].opened_streams[0].closed                 # user master: one stream
    await s2.close()


async def test_stream_revives_a_gone_master_first(world: Callable[..., World]) -> None:
    """Critic: a stream after ControlPersist expired must re-prepare the master like on-demand probes, not fall
    back to a full BatchMode login; a failed prepare is reported, no stream is opened."""
    w = world(K2, mux="managed")
    await w.svc.start()
    await settle()
    t = w.transports["k2"]
    t.set_stream("log.tail", lambda spec: FakeStream([b"x\n"], hold_open=True))
    t.master_ok = False
    before = t.calls.count("prepare")
    s = await w.svc.stream_log("k2", LogRef("stdout", "/w/101.out"))
    assert t.calls.count("prepare") == before + 1 and len(t.opened_streams) == 1
    await s.close()
    t.prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    with pytest.raises(RunFleetError, match="authentication needed"):
        await w.svc.stream_log("k2", LogRef("stdout", "/w/102.out"))
    assert len(t.opened_streams) == 1


async def test_shell_and_pager_argv(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    assert w.svc.shell_argv("k2") == ["fake-ssh", "-t", "k2"]
    assert w.svc.shell_argv("k2", JobKey("k2", "101")) == ["fake-ssh", "-t", "k2", "cd /w/101 && exec $SHELL -l"]
    assert w.svc.pager_argv("k2", LogRef("stdout", "/w/101.out")) == ["fake-ssh", "-t", "k2",
                                                                       "less -R +G -- /w/101.out"]
    assert w.svc.local_pager_argv(Path("/tmp/x"))[-1] == str(Path("/tmp/x"))


# ---------------------------------------------------------------- snapshot (`runfleet snapshot --json`)
async def test_collect_once_fake(world: Callable[..., World]) -> None:
    w = world(K2, LYNX)
    await w.svc.collect_once(timeout=10)
    k2 = w.transports["k2"]
    assert [p.script.purpose for p in k2.probes] == ["detect", "poll", "poll"]   # full poll + follow-ups
    assert "slurm.scontrol" in k2.probes[-1].names
    assert w.status("k2").state is ConnState.OK
    snap = w.svc.store.hosts["k2"].snapshot
    assert snap is not None and snap.caps is not None and len(snap.jobs) == 2
    json.dumps(to_json(snap))
    n = len(k2.probes)
    await run_for(w.clock, 60)
    assert len(k2.probes) == n                                         # nothing keeps polling


async def test_doctor_then_snapshot_stays_within_three_sessions(tmp_path: Path, world: Callable[..., World]) -> None:
    """M3 F4: doctor (1 probe) + snapshot (detect + 2 ticks) was 4 sessions per host against a budget of 3. doctor now
    stores the capabilities it detected; snapshot reuses fresh ones instead of probing again."""
    cache = Cache(tmp_path / "c.sqlite")
    w = world(K2, cache=cache)
    await w.svc.doctor("k2")
    snap_svc = World((K2,), cache=cache, tmp=tmp_path)            # a second process (`runfleet snapshot`)
    snap_svc.transports["k2"].probes = w.transports["k2"].probes
    await snap_svc.svc.collect_once(timeout=10)
    purposes = [p.script.purpose for p in w.transports["k2"].probes]
    assert purposes == ["doctor", "poll", "poll"], purposes
    snap = snap_svc.svc.store.hosts["k2"].snapshot
    assert snap is not None and snap.jobs
    await snap_svc.svc.stop()
    # without the cache the snapshot still detects first
    plain = world(K2)
    await plain.svc.collect_once(timeout=10)
    assert [p.script.purpose for p in plain.transports["k2"].probes] == ["detect", "poll", "poll"]


async def test_collect_once_reports_failures_in_status(world: Callable[..., World]) -> None:
    w = world(K2)
    w.transports["k2"].prepare_errors.append(HostError(ErrorKind.DNS, "dns lookup failed"))
    await w.svc.collect_once()
    assert w.status("k2").state is ConnState.UNREACHABLE and w.status("k2").error.kind is ErrorKind.DNS
    with pytest.raises(RunFleetError):
        await w.svc.collect_once(["nope"])


async def test_collect_once_timeout(world: Callable[..., World]) -> None:
    w = world(K2, latency=100.0)
    task = asyncio.ensure_future(w.svc.collect_once(timeout=0.2))
    done = await asyncio.wait_for(task, 5)
    assert done is None
    assert w.status("k2").state is ConnState.UNREACHABLE and "timed out" in w.status("k2").error.summary


@pytest.mark.posix                # LocalTransport runs sh
async def test_snapshot_of_the_local_host_with_real_sh() -> None:
    """`runfleet snapshot --json local`: FleetService + LocalTransport, real /bin/sh, stub collectors."""
    clock = ManualClock(CLOCK_START)
    host = HostConfig(name="atlas", ssh="local", backend="workstation")
    from runfleet.transport import make_transport
    svc = FleetService(FleetConfig(hosts=(host,)), make_transport, clock, None)
    svc.hooks = STUB_HOOKS
    await svc.collect_once(timeout=30)
    st = svc.store.hosts["atlas"].status
    assert st.state in (ConnState.OK, ConnState.DEGRADED) and st.consecutive_failures == 0
    assert st.mux == "local" and st.latency_s is not None
    snap = svc.store.hosts["atlas"].snapshot
    assert snap is not None and snap.caps is not None and snap.caps.profile == ("process", "nvidia", "procfs")
    assert snap.jobs                                                   # `ps` ran for real
    json.dumps(to_json(snap))
    await svc.stop()
    assert isinstance(make_transport(host), LocalTransport)


@pytest.mark.posix                # LocalTransport runs sh
async def test_snapshot_local_with_wp2_backends_if_available() -> None:
    """The same path with the real WP2 detection/collectors (skipped until they are implemented)."""
    from runfleet.core.poller import BackendHooks
    from runfleet.transport import make_transport

    hooks = BackendHooks.default()
    try:
        hooks.collectors()
        hooks.detect_sections(HostConfig("atlas", "local"), hooks.collectors())
    except NotImplementedError:
        pytest.skip("WP2 backends not implemented yet")
    from runfleet.core.clock import RealClock
    svc = FleetService(FleetConfig(hosts=(HostConfig("atlas", "local"),)), make_transport, RealClock(), None)
    try:
        await svc.collect_once(timeout=30)
    except NotImplementedError:
        pytest.skip("WP2 backends not implemented yet")
    st = svc.store.hosts["atlas"].status
    assert st.consecutive_failures == 0, st.error
    assert svc.store.hosts["atlas"].snapshot is not None


# ---------------------------------------------------------------- cache warm start
async def test_warm_start_from_cache(tmp_path: Path, world: Callable[..., World]) -> None:
    cache = Cache(tmp_path / "cache.sqlite")
    snap = HostSnapshot("k2", CLOCK_START - timedelta(hours=1), stub_caps(K2), {}, (), (), (), (), None)
    cache.put_snapshot("k2", snap)
    cache.put_caps("k2", stub_caps(K2), CLOCK_START - timedelta(hours=1))
    cache.flush_sync()
    w = world(K2, cache=Cache(tmp_path / "cache.sqlite"))
    seen_stale = []
    w.svc.store.subscribe(lambda c: seen_stale.append(w.svc.store.hosts["k2"].stale(w.clock.now())))
    await w.svc.start()
    await settle()
    assert seen_stale[0] is True                                       # cached snapshot shown stale first
    purposes = [p.script.purpose for p in w.transports["k2"].probes]
    assert purposes[0] == "poll"                                       # cached profile used immediately
    await run_for(w.clock, 20)
    assert w.transports["k2"].count("detect") == 1                     # re-detected in the background
    await w.svc.stop()
    again = Cache(tmp_path / "cache.sqlite").load_snapshots()
    assert set(again["k2"].jobs) == {"101", "102"}


async def test_connection_states_reach_the_cache_for_runfleet_status(tmp_path: Path,
                                                                     world: Callable[..., World]) -> None:
    """`runfleet status` shows `lynx:auth` from the cache: the service writes each host's state when it changes
    (never NEW / CONNECTING, and not again while it stays the same)."""
    cache = Cache(tmp_path / "cache.sqlite")
    written: list[str] = []
    put = cache.put_status

    def record(host: str, st: HostStatus, at: datetime) -> None:
        written.append(f"{host}:{st.state.value}")
        put(host, st, at)
    cache.put_status = record  # type: ignore[method-assign]
    w = world(K2, cache=cache)
    w.transports["k2"].prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    await w.svc.start()
    await settle()
    assert w.status("k2").state is ConnState.NEEDS_AUTH
    await cache.flush()
    rows = status.read_cache(tmp_path / "cache.sqlite")
    assert rows.status["k2"][1:] == ("needs_auth", "auth", "authentication needed")
    await w.svc.after_connect("k2", 0)
    await run_for(w.clock, 20)
    await cache.flush()
    assert status.read_cache(tmp_path / "cache.sqlite").status["k2"][1:] == ("ok", None, None)
    assert written == ["k2:needs_auth", "k2:ok"]


async def test_second_instance_notice(tmp_path: Path) -> None:
    clock = ManualClock(CLOCK_START)
    cfg = FleetConfig(hosts=(K2, LYNX))
    a = FleetService(cfg, lambda h: FakeTransport(h.name), clock, None)
    b = FleetService(FleetConfig(hosts=(K2, HostConfig("box", "box"))), lambda h: FakeTransport(h.name), clock, None)
    lonely = FleetService(FleetConfig(hosts=(HostConfig("localhost", "localhost"),)),
                          lambda h: FakeTransport(h.name), clock, None)
    a.lock_path = b.lock_path = lonely.lock_path = tmp_path / "instance.lock"
    await a.start()
    await b.start()
    await lonely.start()
    assert not a.notices
    assert b.notices == [f"another RunFleet (pid {os.getpid()}) is polling k2 — remote load doubles there"]
    assert not lonely.notices                               # disjoint host sets: no warning (critic)
    await lonely.stop()
    demo = FleetService(cfg, lambda h: FakeTransport(h.name), clock, None)
    demo.lock_path = False                                  # --demo: neither takes nor reports the real lock
    await demo.start()
    assert not demo.notices and demo._lock_fd is None
    await a.stop()
    await b.stop()
    await demo.stop()
    c = FleetService(cfg, lambda h: FakeTransport(h.name), clock, None)
    c.lock_path = tmp_path / "instance.lock"
    await c.start()                                         # locks are released on stop
    assert not c.notices
    await c.stop()


# ---------------------------------------------------------------- hosts import and doctor
async def test_probe_alias(world: Callable[..., World]) -> None:
    w = world(K2)
    rep = await w.svc.probe_alias("newbox")
    assert rep.state is ConnState.OK and rep.caps is not None and rep.caps.profile == ("slurm", "procfs")
    assert rep.latency_s is not None and rep.ssh_summary == {}
    bad = world(K2)
    t = bad.factory(HostConfig("broken", "broken"))
    assert isinstance(t, FakeTransport)
    t.prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    rep2 = await bad.svc.probe_alias("broken")
    assert rep2.state is ConnState.NEEDS_AUTH and rep2.caps is None


async def test_probe_alias_bounded_by_max_parallel_ssh() -> None:
    """`hosts import --all`: master creation (prepare) counts against max_parallel_ssh, not only the detect probe."""
    live = peak = 0

    class SlowPrepare(FakeTransport):
        async def prepare(self, *, force: bool = False) -> HostError | None:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return None

    def factory(h: HostConfig) -> Transport:
        t = SlowPrepare(h.name)
        t.set_result("detect.id", "user=me")
        return t

    svc = FleetService(FleetConfig(hosts=(), general=GeneralSettings(max_parallel_ssh=3)), factory,
                       ManualClock(CLOCK_START), None)
    svc.hooks = STUB_HOOKS
    reports = await asyncio.gather(*(svc.probe_alias(f"h{i}") for i in range(12)))
    assert peak == 3 and all(r.state is ConnState.OK for r in reports)


def test_write_config_new_and_append(tmp_path: Path) -> None:
    svc = FleetService(FleetConfig(hosts=()), lambda h: FakeTransport(h.name), ManualClock(CLOCK_START), None)
    path = tmp_path / "sub" / "config.toml"
    out = svc.write_config([HostConfig("k2", "k2"), HostConfig("lynx", "lynx")], path)
    assert out == path and [h["name"] for h in tomllib.loads(path.read_text())["host"]] == ["k2", "lynx"]
    path.write_text(path.read_text() + "# my own comment\n")
    svc.write_config([HostConfig("k2", "k2", connect_timeout=99), HostConfig("lyra", "lyra")], path)
    text = path.read_text()
    assert "# my own comment" in text                                  # never rewrites
    doc = tomllib.loads(text)
    assert [h["name"] for h in doc["host"]] == ["k2", "lynx", "lyra"] and "connect_timeout" not in doc["host"][0]
    path.write_text("[[host]\n")
    with pytest.raises(ConfigError):
        svc.write_config([HostConfig("x", "x")], path)


async def test_doctor_with_fake_transport(world: Callable[..., World]) -> None:
    w = world(K2)
    rep = await w.svc.doctor("k2")
    names = [c.name for c in rep.checks]
    assert names == ["ssh -G", "mux", "master", "probe", "profile", "tools", "clock", "gpu_tres"]
    assert {c.name: c.ok for c in rep.checks}["probe"] is True
    tools = next(c for c in rep.checks if c.name == "tools")
    assert tools.ok is False and "sacct" in tools.detail               # stub caps lack most Slurm tools
    w.transports["k2"].prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    rep2 = await w.svc.doctor("k2", reset=True)
    assert "reset_master" in w.transports["k2"].calls
    assert rep2.checks[-1].name == "master" and rep2.checks[-1].ok is False


@pytest.mark.posix                # fake ssh shell script
async def test_doctor_with_openssh_and_fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_openssh import FAKE_SSH, G_LYRA

    exe = tmp_path / "ssh"
    exe.write_text(FAKE_SSH, encoding="utf-8")
    exe.chmod(0o755)
    (tmp_path / "g.txt").write_text(G_LYRA + "controlpersist no\n")
    monkeypatch.setenv("RF_FAKE_LOG", str(tmp_path / "argv.log"))
    monkeypatch.setenv("RF_FAKE_DIR", str(tmp_path))
    monkeypatch.setenv("RF_FAKE_G", str(tmp_path / "g.txt"))
    from runfleet.core.doctor import run_doctor
    from runfleet.transport.openssh import OpenSSHTransport

    host = HostConfig("lyra", "lyra", backend="slurm")
    t = OpenSSHTransport(host, ssh=str(exe), mux_dir=tmp_path / "mux", log_path=tmp_path / "ssh.log")
    rep = await run_doctor(host, t, hooks=STUB_HOOKS, clock=ManualClock(CLOCK_START), reset=True)
    by = {c.name: c for c in rep.checks}
    assert "proxyjump=jumpvm" in by["ssh -G"].detail
    assert by["mux"].detail.startswith("managed") and "BatchMode ProxyCommand" in by["mux"].detail
    assert by["master"].ok is True and by["probe"].ok is True
    assert by["reset"].detail == "managed master closed"


def test_unknown_host_errors() -> None:
    svc = FleetService(FleetConfig(hosts=(K2,)), lambda h: FakeTransport(h.name), ManualClock(CLOCK_START), None)
    with pytest.raises(RunFleetError):
        svc.connect_argv("nope")
    assert result("x").ok
