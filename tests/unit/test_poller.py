"""[WP1] HostPoller with ManualClock + FakeTransport + stub collectors (§5.4, §4.4, §10.3)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import timedelta

import pytest

from runfleet.config import GeneralSettings, HostConfig, SlurmSettings
from runfleet.core.clock import ManualClock
from runfleet.core.poller import HostLanes, HostPoller, merge_outputs, update_monitor_view
from runfleet.core.store import FleetStore
from runfleet.backends.base import CollectorOutput, MonitorView
from runfleet.models import ConnState, ErrorKind, HostError, Visibility
from runfleet.probe.section import SectionOutcome
from runfleet.transport.fake import FakeTransport, result
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, StubProcess, StubProcfs, StubSlurm, run_for, settle, stub_caps

SLURM = HostConfig(name="k2", ssh="k2", backend="slurm")
WS = HostConfig(name="lynx", ssh="lynx", backend="workstation")


class Rig:
    def __init__(self, host: HostConfig, *, seed: int = 1, mux: str = "none", latency: float = 0.0,
                 general: GeneralSettings | None = None, cached: bool = False) -> None:
        self.clock = ManualClock(CLOCK_START)
        self.store = FleetStore([host])
        self.t = FakeTransport(host.name, mux_mode=mux, clock=self.clock, latency=latency)  # type: ignore[arg-type]
        self.t.set_result("detect.id", "user=me")
        self.lanes = HostLanes(asyncio.Semaphore(8), self.t.mux)
        caps = stub_caps(host) if cached else None
        self.p = HostPoller(host, self.t, self.store, self.clock, random.Random(seed), self.lanes,
                            general=general or GeneralSettings(), hooks=STUB_HOOKS, caps=caps,
                            caps_at=CLOCK_START if cached else None)
        self.task: asyncio.Task[None] | None = None

    @property
    def status(self):  # type: ignore[no-untyped-def]
        return self.store.hosts[self.p.name].status

    def polls(self) -> list[float]:
        return [p.at for p in self.t.probes if p.script.purpose == "poll" and p.at is not None]

    def poll_names(self) -> list[tuple[str, ...]]:
        return [p.names for p in self.t.probes if p.script.purpose == "poll"]


@pytest.fixture
async def rig_factory() -> AsyncIterator[Callable[..., Rig]]:
    rigs: list[Rig] = []

    def make(host: HostConfig = SLURM, **kw: object) -> Rig:
        r = Rig(host, **kw)   # type: ignore[arg-type]
        r.task = asyncio.ensure_future(r.p.run())
        rigs.append(r)
        return r

    yield make
    for r in rigs:
        assert r.task is not None
        r.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await r.task


# ---------------------------------------------------------------- ticks and cadence
async def test_first_cycle_detects_then_polls_with_piggyback(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    assert [p.script.purpose for p in rig.t.probes] == ["detect", "poll"]
    first = rig.poll_names()[0]
    assert set(first) == {"slurm.squeue", "slurm.sacct", "slurm.sinfo", "procfs.host"}
    assert rig.status.state is ConnState.OK
    assert rig.status.tick_interval == 15
    assert rig.status.last_ok == CLOCK_START


async def test_evaluation_only_at_ticks_and_slow_groups(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await run_for(rig.clock, 300)
    names = rig.poll_names()
    times = rig.polls()
    assert all("slurm.squeue" in n and "procfs.host" in n for n in names)   # every tick: jobs + piggyback
    sacct_times = [t for t, n in zip(times, names, strict=True) if "slurm.sacct" in n]
    gaps = [b - a for a, b in zip(sacct_times, sacct_times[1:], strict=False)]
    assert gaps and all(105 <= g <= 140 for g in gaps), gaps          # ~120 s, decided at ticks only
    assert len(times) == len(set(times))                               # one probe per tick


async def test_politeness_slurm_host_one_hour(rig_factory: Callable[..., Rig]) -> None:
    """§1.1/§10.3: a default Slurm host with no wakes makes ≤ 240 probes per simulated hour."""
    for seed in range(5):
        rig = rig_factory(seed=seed)
        await run_for(rig.clock, 3600, step=1.0)
        polls = rig.polls()
        assert 225 <= len(polls) <= 240, (seed, len(polls))
        assert rig.t.count("detect") == 1
        for i in range(len(polls)):                          # any 60 s window holds at most 5 ticks
            assert sum(1 for t in polls[i:] if t < polls[i] + 60) <= 5
        rig.task.cancel()  # type: ignore[union-attr]


async def test_jitter_bounds(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(seed=7)
    await run_for(rig.clock, 900)
    times = rig.polls()
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert min(gaps) >= 0.8 * 15 - 0.5 and max(gaps) <= 1.2 * 15 + 0.5
    assert len(set(round(g) for g in gaps)) > 2                       # it does jitter


async def test_hidden_and_focused_factors(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.p.set_visibility(Visibility.HIDDEN)
    await run_for(rig.clock, 60)
    assert rig.status.tick_interval == 45
    before = len(rig.polls())
    await run_for(rig.clock, 450)
    assert 8 <= len(rig.polls()) - before <= 12                         # ~450/45
    ws = rig_factory(WS)
    await settle()
    ws.p.set_visibility(Visibility.FOCUSED)
    await run_for(ws.clock, 10)
    assert ws.status.tick_interval == pytest.approx(3.0)                # nvidia 5 s × 0.6


async def test_floor_clamps_multiplier(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.p.set_multiplier(0.5)                                          # 15 × 0.5 = 7.5 < floor 10
    await run_for(rig.clock, 30)
    assert rig.status.tick_interval == 10
    rig.p.set_multiplier(4.0)
    await run_for(rig.clock, 30)
    assert rig.status.tick_interval == 60


async def test_event_groups_ride_the_next_tick(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.t.set_result("slurm.squeue", "101 R\n102 PD")
    await settle()
    names = rig.poll_names()
    assert len(names) == 1 and "slurm.scontrol" not in names[0]        # follow-up requested, not run yet
    await run_for(rig.clock, 19)
    names = rig.poll_names()
    assert "slurm.scontrol" in names[1]
    await run_for(rig.clock, 30)
    assert all("slurm.scontrol" not in n for n in rig.poll_names()[2:])  # served once


async def test_followup_naming_an_interval_group_is_served_next_tick(rig_factory: Callable[..., Rig],
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """`slurm.nodes` (interval, 120 s) requested as a follow-up runs at the next tick, not 2 minutes later."""
    orig = StubSlurm.ingest
    asked: list[bool] = []

    def ingest(self: StubSlurm, results, ctx, monitors):  # type: ignore[no-untyped-def]
        out = orig(self, results, ctx, monitors)
        if "slurm.squeue" in results and not asked:
            asked.append(True)
            out = replace(out, followups=out.followups | {"slurm.nodes"})
        return out

    monkeypatch.setattr(StubSlurm, "ingest", ingest)
    rig = rig_factory()
    await settle()
    assert "slurm.sinfo" in rig.poll_names()[0]
    await run_for(rig.clock, 19)
    assert "slurm.sinfo" in rig.poll_names()[1]                        # served at the next tick
    await run_for(rig.clock, 60)
    assert all("slurm.sinfo" not in n for n in rig.poll_names()[2:])   # then back on its 120 s cadence


async def test_piggyback_never_alone(rig_factory: Callable[..., Rig]) -> None:
    StubProcfs.piggyback_interval = 5.0
    try:
        rig = rig_factory()
        await run_for(rig.clock, 120)
        times = rig.polls()
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert min(gaps) >= 11                     # procfs every 5 s would add probes; it only rides jobs ticks
        assert all("procfs.host" in n for n in rig.poll_names())
    finally:
        StubProcfs.piggyback_interval = None


# ---------------------------------------------------------------- failures and backoff (§4.4)
async def test_transport_backoff_growth_and_reset(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.t.fail_next(ErrorKind.TIMEOUT, times=3)
    states, delays = [], []
    last = rig.polls()[-1]
    for _ in range(3):
        before = len(rig.t.probes)
        while len(rig.t.probes) == before:
            await run_for(rig.clock, 1)
        states.append(rig.status.state)
        delays.append(rig.t.probes[-1].at)
    gaps = [delays[0] - last] + [b - a for a, b in zip(delays, delays[1:], strict=False)]
    assert states[:2] == [ConnState.RETRYING, ConnState.RETRYING]
    await settle()
    assert rig.status.state is ConnState.UNREACHABLE and rig.status.consecutive_failures == 3
    assert 12 <= gaps[1] <= 18.5 and 24 <= gaps[2] <= 36.5             # 15 × 2^(n-1) × U(0.8, 1.2)
    while rig.status.state is not ConnState.OK:
        await run_for(rig.clock, 1)
    assert rig.status.consecutive_failures == 0 and rig.status.error is None


async def test_dns_failure_goes_unreachable_with_long_backoff(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.t.fail_next(ErrorKind.DNS)
    await run_for(rig.clock, 20)
    assert rig.status.state is ConnState.UNREACHABLE
    n = len(rig.t.probes)
    await run_for(rig.clock, 200)
    assert len(rig.t.probes) == n                                      # ≥ 240 s (300 × U(0.8, 1.2))


async def test_needs_auth_pauses_until_manual_retry(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.t.prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    await settle()
    assert rig.status.state is ConnState.NEEDS_AUTH
    assert rig.t.probes == []
    await run_for(rig.clock, 600)
    assert rig.t.probes == [] and rig.t.calls.count("prepare") == 1
    rig.t.prepare_errors.append(HostError(ErrorKind.AUTH, "authentication needed"))
    rig.p.wake("manual")                                               # exactly one more BatchMode attempt
    await run_for(rig.clock, 5)
    assert rig.t.calls.count("prepare") == 2 and rig.status.state is ConnState.NEEDS_AUTH
    assert rig.t.forced_prepares == 1                                  # the retry bypasses the 30 s master throttle
    await run_for(rig.clock, 300)
    assert rig.t.calls.count("prepare") == 2
    rig.p.reconnect()                                                  # after `c`: re-prepare, poll now
    await settle()
    assert rig.status.state is ConnState.OK and rig.t.count("poll") >= 1


async def test_focus_and_visibility_changes_keep_the_transport_backoff(rig_factory: Callable[..., Rig]) -> None:
    """Regression (review core): _reschedule re-gridded from the last good tick and retried an unreachable host at
    once on every alt-tab / scope change."""
    rig = rig_factory()
    await settle()
    rig.t.fail_next(ErrorKind.REFUSED, times=50)
    while rig.status.consecutive_failures < 4:
        await run_for(rig.clock, 1)
    assert rig.status.state is ConnState.UNREACHABLE
    n = len(rig.t.probes)
    nxt = rig.p._next_at
    for _ in range(5):                                                  # blur/focus toggles
        rig.p.set_multiplier(4.0)
        await run_for(rig.clock, 0.5)
        rig.p.set_multiplier(1.0)
        rig.p.set_visibility(Visibility.FOCUSED)
        rig.p.set_visibility(Visibility.VISIBLE)
        await run_for(rig.clock, 0.5)
    assert len(rig.t.probes) == n and rig.p._next_at == nxt


async def test_forced_groups_survive_a_transport_failure(rig_factory: Callable[..., Rig]) -> None:
    """wake(groups=…) after an action: a failed probe must not drop the forced refresh."""
    rig = rig_factory()
    await run_for(rig.clock, 20)
    rig.t.fail_next(ErrorKind.REFUSED)
    rig.p.wake("action", frozenset({"slurm.history"}))
    await run_for(rig.clock, 4)
    assert rig.status.consecutive_failures == 1
    while rig.status.consecutive_failures:
        await run_for(rig.clock, 1)
    assert "slurm.sacct" in rig.poll_names()[-1]


async def test_partial_timeout_is_degraded_not_transport_failure(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.t.queue("slurm.squeue", result("slurm.squeue", "", None, outcome=SectionOutcome.TIMEOUT))
    await run_for(rig.clock, 19)
    st = rig.status
    assert st.state is ConnState.DEGRADED and st.consecutive_failures == 0
    assert st.sections["slurm.squeue"].outcome == "timeout" and st.sections["slurm.squeue"].error is not None
    assert st.sections["procfs.host"].ok
    assert rig.p._group_fail.get("slurm.jobs") == 1                   # group backoff, transport untouched
    assert st.tick_interval == 30
    await run_for(rig.clock, 40)
    assert rig.status.state is ConnState.OK and "slurm.jobs" not in rig.p._group_fail


async def test_section_failure_backs_off_its_group_up_to_max(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.t.set_result("slurm.sacct", "", 1, "Problem talking to the database")
    await run_for(rig.clock, 2000, step=1.0)
    names, times = rig.poll_names(), rig.polls()
    sacct = [t for t, n in zip(times, names, strict=True) if "slurm.sacct" in n]
    gaps = [b - a for a, b in zip(sacct, sacct[1:], strict=False)]
    assert gaps[0] >= 230 and max(gaps) <= 620                         # 120 → 240 → 480 → capped at 600
    assert rig.status.state is ConnState.DEGRADED
    assert rig.status.error is not None and "database" in rig.status.error.summary


async def test_single_in_flight_and_manual_refresh_floor(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(latency=2.0)
    await run_for(rig.clock, 3)
    rig.p.wake("manual")
    rig.p.wake("manual")
    await run_for(rig.clock, 10)
    assert rig.t.max_in_flight == 1
    times = rig.polls()
    assert len(times) == 2 and times[1] - times[0] >= 3.0              # ≥ 3 s after the previous probe


async def test_check_master_failure_ensures_and_skips_the_tick(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(mux="user")
    await settle()
    polls = rig.t.count("poll")
    rig.t.master_ok = False
    await run_for(rig.clock, 18)
    assert rig.t.count("poll") == polls                                # no probe while (re)making the master
    assert rig.t.calls.count("prepare") == 2


async def test_two_timeouts_reset_a_managed_master(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(mux="managed")
    await settle()
    rig.t.fail_next(ErrorKind.TIMEOUT, times=2)
    await run_for(rig.clock, 60)
    assert "reset_master" in rig.t.calls


async def test_two_timeouts_on_user_master_suggest_restart(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(mux="user")
    await settle()
    rig.t.fail_next(ErrorKind.TIMEOUT, times=3)
    while rig.status.consecutive_failures < 2:
        await run_for(rig.clock, 1)
    assert "reset_master" not in rig.t.calls
    assert rig.status.error is not None and "press c to restart it" in rig.status.error.summary


async def test_session_refused_degrades_lanes(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(mux="managed")
    await settle()
    assert rig.lanes.on_demand.limit == 2
    rig.t.fail_next(HostError(ErrorKind.MUX, "ssh master trouble", "Session open refused by peer"))
    await run_for(rig.clock, 20)
    assert rig.lanes.on_demand.limit == 1 and rig.lanes.stream_limit == 1


async def test_collector_exception_is_internal_error(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    sched = rig.p.scheduler
    assert isinstance(sched, StubSlurm)
    sched.raise_on_ingest = True
    await run_for(rig.clock, 18)
    st = rig.status
    assert st.state is ConnState.DEGRADED and st.error is not None and st.error.kind is ErrorKind.INTERNAL
    assert "log id" in st.error.summary
    sched.raise_on_ingest = False
    await run_for(rig.clock, 70)
    assert rig.status.state is ConnState.OK


async def test_parse_error_from_collector_keeps_jobs(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.t.set_result("slurm.squeue", "101 R")
    await settle()
    rig.t.set_result("slurm.squeue", "garbage")
    await run_for(rig.clock, 18)
    assert rig.status.state is ConnState.DEGRADED
    assert rig.status.sections["slurm.squeue"].error.kind is ErrorKind.PARSE   # type: ignore[union-attr]
    assert [j.key.id for j in rig.store.jobs()] == ["101"]


# ---------------------------------------------------------------- monitors and merge (§3.6)
async def test_persistent_monitor_view_with_ages(rig_factory: Callable[..., Rig]) -> None:
    StubProcess.views.clear()
    rig = rig_factory(WS)
    rig.t.set_result("nvidia.gpu", "0 50\n1 10")
    rig.t.set_result("process.ps", "642229 0")
    await settle()
    t_first = rig.clock.now()
    rig.t.set_result("nvidia.gpu", "", None, outcome=SectionOutcome.TIMEOUT)   # hung nvidia-smi
    await run_for(rig.clock, 12)
    view = StubProcess.views[-1]
    assert view.gpus_at == t_first and len(view.gpus) == 2
    assert view.gpus_age(rig.clock.now()) is not None and view.gpus_age(rig.clock.now()) >= 4   # type: ignore[operator]
    gpus = list(rig.store.gpus())
    assert [g.name for g in gpus] == ["annotated", "annotated"]        # scheduler output wins (monitors first)
    assert rig.status.state is ConnState.DEGRADED


def test_merge_rule_last_non_none_wins() -> None:
    a = CollectorOutput(gpus=(), metrics=None, caps_patch={"x": 1}, followups=frozenset({"a"}))
    b = CollectorOutput(gpus=None, procs=(), caps_patch={"y": 2}, followups=frozenset({"b"}))
    m = merge_outputs([a, b])
    assert m.gpus == () and m.procs == () and m.metrics is None
    assert m.caps_patch == {"x": 1, "y": 2} and m.followups == {"a", "b"}


def test_monitor_view_updates_only_returned_fields() -> None:
    v = MonitorView()
    v2 = update_monitor_view(v, CollectorOutput(), CLOCK_START)
    assert v2 is v
    v3 = update_monitor_view(v, CollectorOutput(gpus=()), CLOCK_START)
    assert v3.gpus_at == CLOCK_START and v3.metrics is None


async def test_ingest_every_collector_every_tick(rig_factory: Callable[..., Rig]) -> None:
    StubSlurm.ingests.clear()
    rig = rig_factory()
    await run_for(rig.clock, 60)
    assert len(StubSlurm.ingests) == rig.t.count("poll")


# ---------------------------------------------------------------- cached profile and redetect
async def test_cached_profile_polls_first_and_redetects_in_background(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(cached=True)
    await settle()
    assert [p.script.purpose for p in rig.t.probes][:1] == ["poll"]
    await run_for(rig.clock, 70)
    assert rig.t.count("detect") == 0                                  # caps fresh (detected "now")
    rig2 = Rig(SLURM, cached=True)
    rig2.p.caps_at = CLOCK_START - timedelta(hours=7)
    task = asyncio.ensure_future(rig2.p.run())
    await run_for(rig2.clock, 20)
    assert rig2.t.probes[0].script.purpose == "poll" and rig2.t.count("detect") == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_idle_pause_stops_all_polling(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.p.set_paused(True)
    n = len(rig.t.probes)
    await run_for(rig.clock, 600)
    assert len(rig.t.probes) == n
    rig.p.set_paused(False)
    await settle()
    assert len(rig.t.probes) == n + 1                                  # immediate refresh on resume


async def test_custom_poll_settings_change_the_tick(rig_factory: Callable[..., Rig]) -> None:
    host = replace(SLURM, slurm=SlurmSettings(poll_jobs=30))
    rig = rig_factory(host)
    await run_for(rig.clock, 5)
    assert rig.status.tick_interval == 30


async def test_detection_failure_is_retried_not_paused(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    rig.t.queue("detect.id", result("detect.id", "", 1, "id: cannot find name"))
    await settle()
    assert rig.status.error is not None and "detection failed" in rig.status.error.summary
    assert rig.status.next_poll_at is not None                         # backoff scheduled, not paused
    await run_for(rig.clock, 40)
    assert rig.t.count("detect") == 2 and rig.status.state is ConnState.OK


async def test_remote_shell_failure_pauses(rig_factory: Callable[..., Rig]) -> None:
    from runfleet.transport.classify import remote_shell_error

    rig = rig_factory()
    rig.t.fail_next(remote_shell_error(127, "sh: not found"))
    await settle()
    assert rig.status.state is ConnState.DEGRADED and rig.status.next_poll_at is None
    await run_for(rig.clock, 120)
    assert rig.t.count() == 1


async def test_collect_once_second_round_serves_groups_that_had_nothing_to_do() -> None:
    """M3 F9: logstat (and sstat/sinfo) had no paths/ids at the first snapshot tick and were marked served, so the
    snapshot never had log sizes. The follow-up round now also serves groups that produced no section."""
    from runfleet.backends.base import Cadence
    from runfleet.core.poller import BackendHooks
    from tests.unit import stubs

    class LogSlurm(stubs.StubSlurm):
        def cadences(self, ctx):  # type: ignore[no-untyped-def]
            return [*super().cadences(ctx), Cadence("slurm.logstat", 120.0, floor=60)]

        def sections(self, due, ctx):  # type: ignore[no-untyped-def]
            out = super().sections(due, ctx)
            if "slurm.logstat" in due and self.jobs:
                out.append(stubs.Section("slurm.logstat", "stat", timeout=10))
            return out

    hooks = BackendHooks(collectors=lambda: {**stubs.REGISTRY, "slurm": LogSlurm},
                         detect_sections=stubs.stub_detect_sections,
                         build_capabilities=stubs.stub_build_capabilities)
    clock = ManualClock(CLOCK_START)
    store = FleetStore([SLURM])
    t = FakeTransport(SLURM.name, clock=clock)
    t.set_result("detect.id", "user=me")
    t.set_result("slurm.squeue", "101 R")
    p = HostPoller(SLURM, t, store, clock, random.Random(1), HostLanes(asyncio.Semaphore(8), t.mux),
                   general=GeneralSettings(), hooks=hooks)
    assert await p.collect_once() is None
    polls = [r.names for r in t.probes if r.script.purpose == "poll"]
    assert len(polls) == 2 and "slurm.logstat" not in polls[0] and "slurm.logstat" in polls[1]


# ---------------------------------------------------------------- review regressions
async def test_slow_successful_probe_is_fresh_when_it_lands(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory(latency=12.0)                                    # detect 0..12 s, first poll 12..24 s
    await run_for(rig.clock, 24)
    assert rig.t.count("poll") == 1
    assert rig.status.last_ok == CLOCK_START + timedelta(seconds=24)   # receipt time, not the 12 s sample time
    assert not rig.store.hosts["k2"].stale(rig.clock.now())


async def test_backoff_never_overflows_after_days_of_failures(rig_factory: Callable[..., Rig]) -> None:
    rig = rig_factory()
    await settle()
    rig.p._consecutive = 2000
    await rig.p._on_transport_failure(HostError(ErrorKind.TIMEOUT, "timed out"))
    assert rig.status.next_poll_at is not None
    from runfleet.backends.base import Cadence
    rig.p._group_fail["slurm.jobs"] = 5000
    assert rig.p.effective_period(Cadence("slurm.jobs", 15.0, floor=10)) <= 15.0 * 2 ** 16


async def test_profile_rebuild_drops_sections_of_departed_collectors(rig_factory: Callable[..., Rig]) -> None:
    from runfleet.models import SectionStatus
    rig = rig_factory()
    await settle()
    gone = SectionStatus("nvidia.gpu", "failed", ok=False, rc=9, ms=3, last_ok=None,
                         error=HostError(ErrorKind.REMOTE, "nvidia.gpu: failed (rc 9)", section="nvidia.gpu"))
    rig.store.set_status("k2", replace(rig.status, sections={**rig.status.sections, "nvidia.gpu": gone}))
    rig.p._rebuild_profile(stub_caps(SLURM))                           # slurm + procfs: no nvidia any more
    assert "nvidia.gpu" not in rig.status.sections and "slurm.squeue" in rig.status.sections
    rig.p.wake("manual")
    await run_for(rig.clock, 4)
    assert rig.status.state is ConnState.OK


async def test_unchanged_jobmeta_is_touched_hourly() -> None:
    class RecCache:
        def __init__(self) -> None:
            self.jobmeta: list[float] = []

        def put_jobmeta(self, host: str, metas: list) -> None:  # type: ignore[type-arg]
            self.jobmeta.extend(m.last_seen for m in metas)

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return lambda *a, **k: None

    clock = ManualClock(CLOCK_START)
    store = FleetStore([SLURM])
    t = FakeTransport(SLURM.name, clock=clock)
    t.set_result("detect.id", "user=me")
    t.set_result("slurm.squeue", "101 R")
    cache = RecCache()
    p = HostPoller(SLURM, t, store, clock, random.Random(1), HostLanes(asyncio.Semaphore(8), t.mux),
                   general=GeneralSettings(), hooks=STUB_HOOKS, cache=cache)  # type: ignore[arg-type]
    task = asyncio.ensure_future(p.run())
    try:
        await run_for(clock, 2 * 3600 + 60, step=5.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert 2 <= len(cache.jobmeta) <= 4, cache.jobmeta                 # first sighting + ~hourly touches
