"""[WP4] Pending priority evidence in the demo world (§5.2.10, §10.6): `--demo` shows all three cluster setups — k2
ranks by fair share (14-day half-life), vega's usage never decays, lyra does not rank at all — and the real poller
runs sprio + sshare at most every 10 min, only while an own job is pending, never on lyra."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from runfleet.config import GeneralSettings
from runfleet.core.clock import ManualClock
from runfleet.core.poller import BackendHooks, HostLanes, HostPoller
from runfleet.core.store import FleetStore
from runfleet.demo import render
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.models import ConnState, JobState
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult
from tests.conftest import CLOCK_START
from tests.demo.conftest import StepClock
from tests.demo.test_roundtrip import Driver
from tests.unit.stubs import run_for


async def test_k2_fairshare_evidence_round_trip(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "k2")
    caps = await d.run_detect()
    assert {"sprio", "sshare"} <= caps.tools and caps.slurm_config["PriorityWeightFairShare"] == "100000"
    out = await d.poll()
    assert not out.get("errors")
    p = out["jobs"]["3372066"].priority                    # smoke-test, PD Priority, gpu-preempt
    assert p is not None and p.ranked and p.dominant == "fairshare" and p.partition == "gpu-preempt"
    assert p.priority == sum(f.value for f in p.factors)
    assert ("fairshare 0.03: your group (lab) used much more than its share recently; it recovers with a half-life "
            "of 14 days") in p.hints
    assert out["jobs"]["3372143"].priority is not None    # PD Resources in lab: the partition tier counts there
    assert dict((f.name, f.value) for f in out["jobs"]["3372143"].priority.factors)["partition"] == 1000
    assert all(j.priority is None for j in out["jobs"].values() if j.state is not JobState.PENDING)


async def test_vega_and_lyra_setups(make_world: Any) -> None:
    world, clock = make_world()
    d = Driver(world, clock, "vega")
    await d.run_detect()
    out = await d.poll()
    p = out["jobs"]["824479"].priority                     # PD QOSMaxJobsPerUserLimit
    assert p is not None and p.ranked and p.half_life_s == 0
    assert p.hints[0].startswith("held back by per-user job cap (QOS), not by priority")
    assert [s.account for s in p.fairshare] == ["lab", "dept"]
    world, clock = make_world()
    d = Driver(world, clock, "lyra")
    await d.run_detect()
    sections = {s.name for col in d.profile for s in col.sections(frozenset({"slurm.priority"}), d.ctx())}
    assert not sections                                    # every weight 0: nothing to read
    out = await d.poll()
    pending = [j for j in out["jobs"].values() if j.state is JobState.PENDING]
    assert pending and all(j.priority is not None and not j.priority.ranked for j in pending)
    assert "submit order" in pending[0].priority.hints[0]


def test_demo_sprio_refuses_basic_and_has_no_jobs_message(world: DemoWorld) -> None:
    c = world.cluster("k2")
    c.config["PriorityType"] = "priority/basic"
    out = render.sprio(c, world.now())
    assert out.rc == 1 and "supported priority plugin" in out.stderr
    c.config["PriorityType"] = "priority/multifactor"
    for job in c.jobs.values():
        if job.state == "PENDING":
            job.held = True
    assert render.sprio(c, world.now()).stdout.startswith("Unable to find jobs")


class _Recording(DemoTransport):
    def __init__(self, *a: Any, clock: ManualClock, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.clock = clock
        self.calls: list[tuple[float, tuple[str, ...]]] = []

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        if script.purpose == "poll":
            self.calls.append((self.clock.monotonic(), tuple(s.name for s in script.sections)))
        return await super().probe(script, timeout=timeout)


async def _poll_hour(host: str) -> list[float]:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    cfg = world.configs[host]
    t = _Recording(cfg, world, clock=clock)
    p = HostPoller(cfg, t, FleetStore([cfg]), clock, random.Random(1), HostLanes(asyncio.Semaphore(8), t.mux),
                   general=GeneralSettings(), hooks=BackendHooks.default())
    task = asyncio.ensure_future(p.run())
    try:
        await run_for(clock, 3600, step=1.0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return [at for at, names in t.calls if "slurm.sprio" in names]


async def test_poller_reads_priority_at_most_every_ten_minutes() -> None:
    runs = await _poll_hour("k2")
    assert 5 <= len(runs) <= 7, runs                       # due at the first tick, then every ~600 s
    gaps = [b - a for a, b in zip(runs, runs[1:], strict=False)]
    assert all(g >= 600 - 15 for g in gaps), gaps          # the floor (a tick may land half a tick early)
    assert await _poll_hour("lyra") == []                  # unranked: never asked


async def _poll_k2_then(answer: Any, seconds: float = 3600) -> tuple[HostPoller, list[float]]:
    """Poll demo k2 until sprio has run once, then let ``answer(world)`` change what sprio says and poll on."""
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    cfg = world.configs["k2"]
    t = _Recording(cfg, world, clock=clock)
    p = HostPoller(cfg, t, FleetStore([cfg]), clock, random.Random(1), HostLanes(asyncio.Semaphore(8), t.mux),
                   general=GeneralSettings(), hooks=BackendHooks.default())
    task = asyncio.ensure_future(p.run())
    try:
        await run_for(clock, 60, step=1.0)
        assert any("slurm.sprio" in names for _at, names in t.calls)
        answer(world)
        await run_for(clock, seconds, step=1.0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return p, [at for at, names in t.calls if "slurm.sprio" in names]


async def test_sprio_refusal_stops_the_group_without_degrading_the_host() -> None:
    """The cluster went priority/basic after detection: sprio refuses (rc 1). The group stops for the session, and
    the host is not left DEGRADED by a section that will never run again to clear it."""
    def basic(world: DemoWorld) -> None:
        world.cluster("k2").config["PriorityType"] = "priority/basic"

    p, runs = await _poll_k2_then(basic)
    assert len(runs) == 2, runs                             # the first reading, then the refusal; never again
    st = p.status
    assert st.state is ConnState.OK and st.error is None, st.error
    assert st.sections["slurm.sprio"].ok


async def test_sprio_finding_no_job_on_stderr_is_no_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job started between squeue and sprio and sprio says so with a non-zero exit: no DEGRADED host, no backoff
    (the next reading still comes ~600 s later, not 1200 s)."""
    def gone(_world: DemoWorld) -> None:
        monkeypatch.setattr(render, "sprio", lambda _c, _now: render.Out(
            "", 1, "Unable to find jobs matching user/id(s) specified\n"))

    p, runs = await _poll_k2_then(gone, seconds=1300)
    assert len(runs) == 3, runs
    assert runs[2] - runs[1] < 900, runs                    # no failure backoff
    assert p.status.state is ConnState.OK and p.status.sections["slurm.sprio"].ok
