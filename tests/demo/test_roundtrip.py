"""Demo render -> WP2 parsers round trip (§10.6, §11.1 WP4 acceptance).

The real collectors plan sections, DemoTransport answers them in the real formats, the collectors ingest the answers;
the parsed jobs/GPUs/tasks must match the simulated truth. Only frozen contracts are used (registry.collectors,
detect.detect_sections/build_capabilities, Collector.cadences/sections/ingest, SchedulerCollector actions). A small
local driver plays the poller's merge rule (§3.6) so this runs without WP1's engine. Skips (with the reason) while
WP2's collectors are not implemented.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pytest

from runfleet.backends.base import CollectorOutput, HostContext, MonitorView, Role, SchedulerCollector
from runfleet.config import GeneralSettings
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld, SimJob
from runfleet.models import Capabilities, JobState
from runfleet.probe.section import ProbeScript, SectionResult
from tests.demo.conftest import StepClock

NONCE = "0123456789ab"
SLURM_TO_STATE = {"PENDING": JobState.PENDING, "RUNNING": JobState.RUNNING, "COMPLETING": JobState.COMPLETING,
                  "COMPLETED": JobState.COMPLETED, "FAILED": JobState.FAILED, "TIMEOUT": JobState.TIMEOUT,
                  "OUT_OF_MEMORY": JobState.OUT_OF_MEMORY, "PREEMPTED": JobState.PREEMPTED,
                  "CANCELLED": JobState.CANCELLED}


def _wp2() -> tuple[Any, Any]:
    try:
        from runfleet.backends import detect, registry
        cols = registry.collectors()
        detect.detect_sections(DemoWorld().configs["k2"], cols)
    except (ImportError, NotImplementedError) as exc:      # pragma: no cover - only before WP2 lands
        pytest.skip(f"WP2 collectors not importable/implemented yet ({exc!r}); round trip enabled at M2")
    return detect, cols


class Driver:
    """One host: detect, then poll ticks with the §3.6 merge rule (monitors first, scheduler last)."""

    def __init__(self, world: DemoWorld, clock: StepClock, host: str) -> None:
        self.detect, self.cols = _wp2()
        self.world, self.clock = world, clock
        self.cfg = world.configs[host]
        self.transport = DemoTransport(self.cfg, world)
        self.monitors = MonitorView()
        self.followups: set[str] = set()
        self.caps: Capabilities | None = None
        self.profile: list[Any] = []

    def ctx(self) -> HostContext:
        assert self.caps is not None
        return HostContext(host=self.cfg, general=GeneralSettings(), caps=self.caps, now=self.clock())

    async def run_detect(self) -> Capabilities:
        sections = self.detect.detect_sections(self.cfg, self.cols)
        res = await self.transport.probe(ProbeScript(tuple(sections), NONCE, "detect"), timeout=60)
        assert res.error is None, res.error
        self.caps = self.detect.build_capabilities(res.sections, self.cfg, self.clock(), self.cols)
        self.profile = [self.cols[name](self.cfg) for name in self.caps.profile]
        return self.caps

    @property
    def scheduler(self) -> SchedulerCollector:
        sched = next(c for c in self.profile if c.role is Role.SCHEDULER)
        assert isinstance(sched, SchedulerCollector)
        return sched

    async def tick(self) -> dict[str, Any]:
        ctx = self.ctx()
        cadences = [cad for col in self.profile for cad in col.cadences(ctx)]
        due = frozenset(c.group for c in cadences if c.mode != "event") | frozenset(self.followups)
        self.followups = set()
        sections = [s for col in self.profile for s in col.sections(due, ctx)]
        res = await self.transport.probe(ProbeScript(tuple(sections), NONCE, "poll"), timeout=60)
        assert res.error is None, res.error
        merged: dict[str, Any] = {}
        ordered = [c for c in self.profile if c.role is Role.MONITOR] + [self.scheduler]
        for col in ordered:
            mine: Mapping[str, SectionResult] = {n: r for n, r in res.sections.items() if n.split(".")[0] == col.name}
            out: CollectorOutput = col.ingest(mine, ctx, self.monitors)
            if col.role is Role.MONITOR:
                self.monitors = dataclasses.replace(
                    self.monitors,
                    gpus=out.gpus if out.gpus is not None else self.monitors.gpus,
                    gpu_procs=out.gpu_procs if out.gpu_procs is not None else self.monitors.gpu_procs,
                    gpus_at=ctx.now if out.gpus is not None else self.monitors.gpus_at,
                    metrics=out.metrics if out.metrics is not None else self.monitors.metrics)
            for f in dataclasses.fields(out):
                value = getattr(out, f.name)
                if f.name == "followups":
                    self.followups |= set(value)
                elif f.name == "errors":
                    merged.setdefault("errors", []).extend(value)
                elif value is not None:
                    merged[f.name] = value
            if out.caps_patch:
                self.caps = dataclasses.replace(self.caps, **dict(out.caps_patch))  # type: ignore[arg-type]
        return merged

    async def poll(self, ticks: int = 2, step: float = 15.0) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for _ in range(ticks):
            merged = await self.tick()
            self.clock.advance(step)
        return merged


def own_jobs(world: DemoWorld, host: str, window: float = 3 * 86400) -> dict[str, SimJob]:
    c = world.cluster(host)
    now = world.now()
    return {j.display_id: j for j in c.jobs.values()
            if j.user == c.user and (not j.terminal or (j.end or 0) >= now - window)}


async def test_k2_jobs_nodes_and_caps_round_trip(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "k2")
    caps = await d.run_detect()
    assert caps.profile[0] == "slurm" and "procfs" in caps.profile and "nvidia" not in caps.profile
    assert (caps.slurm_version, caps.cluster_name, caps.remote_user) == ("25.05.8", "k2", "alice")
    assert caps.slurm_config.get("PreemptMode") == "REQUEUE" and caps.acct_gather_freq_s == 30
    assert caps.gpu_tres is True and ("k2", "lab", "", "gpu-priority,normal,labpi") in caps.associations
    out = await d.poll()
    jobs = out["jobs"]
    truth = own_jobs(world, "k2")
    assert set(jobs) == set(truth)
    for jid, sim in truth.items():
        job = jobs[jid]
        assert job.state is SLURM_TO_STATE[sim.state], jid
        assert job.job_id_raw == str(sim.job_id)
        if sim.state == "RUNNING":
            assert job.alloc.gpus == sim.gpus and set(job.nodes) == set(sim.nodes), jid
    assert jobs["3372094"].nodes == ("n-37-01", "n-37-02") and jobs["3372094"].alloc.gpus == 8
    assert jobs["3371310"].alloc.gpu_type == "nvidia_h200_nvl"          # untyped tres-alloc, typed request
    assert jobs["3372066"].est_start and jobs["3372066"].reason == "Priority"
    assert jobs["3371311"].raw.get("cancelled_by_uid") == "10001"
    assert jobs["3368404"].exit_code == 1
    running = jobs["3370390"]
    assert running.gpu is not None and running.gpu.util_mean_pct is not None and running.gpu.util_mean_pct > 80
    assert {n.name for n in out["nodes"]} >= {"n-40-01", "n-31-05", "n-34-01"}
    parts = {p.name: p for p in out["partitions"]}
    assert "gpu-preempt" in parts and parts["gpu-preempt"].gpus_total > 0
    assert out["metrics"].ncpu == 32 and out["metrics"].mem_total is not None    # login node, procfs piggyback


async def test_lyra_array_preemption_and_no_gpu_tres(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "lyra")
    caps = await d.run_detect()
    assert caps.gpu_tres is False and caps.slurm_config.get("PreemptMode") == "CANCEL"
    jobs = (await d.poll())["jobs"]
    summary = jobs["32377076_[5-12%2]"]
    assert summary.state is JobState.PENDING and summary.job_id_raw == "32377076"
    assert jobs["32377076_3"].state is JobState.RUNNING and jobs["32377076_3"].job_id_raw == "32377079"
    assert jobs["32354789"].state is JobState.PREEMPTED
    assert jobs["32378138"].alloc.gpu_type in ("a30", "A30")


async def test_vega_qos_cap_reason(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "vega")
    await d.run_detect()
    jobs = (await d.poll())["jobs"]
    assert jobs["824479"].state is JobState.PENDING and jobs["824479"].reason == "QOSMaxJobsPerUserLimit"
    assert jobs["824507"].reason == "DependencyNeverSatisfied"
    assert sum(1 for j in jobs.values() if j.state is JobState.RUNNING) == 4


async def test_lynx_tasks_gpus_and_queue_runner(world: DemoWorld, clock: StepClock) -> None:
    world.clear("lynx")
    d = Driver(world, clock, "lynx")
    caps = await d.run_detect()
    assert set(caps.profile) == {"process", "nvidia", "procfs"} and caps.driver_version == "610.43.02"
    out = await d.poll(ticks=3, step=5.0)
    gpus = out["gpus"]
    assert len(gpus) == 2 and "Blackwell" in gpus[0].name and gpus[0].fan_pct is None
    mine = [j for j in out["jobs"].values() if j.is_mine and not j.state.is_terminal]
    assert len(mine) == 1
    task = mine[0]
    assert task.key.label == "p642229" and set(task.pids) == {642229, 642234, 786345}
    assert task.alloc.gpus == 2


async def test_labstation_other_user_and_failed_task(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "labstation")
    await d.run_detect()
    out = await d.poll(ticks=2, step=5.0)
    holders = {p.pid: p for g in out["gpus"] for p in g.processes}
    assert set(holders) == {2911042, 590112, 363948}
    assert holders[2911042].user == "carol" and holders[2911042].gpu_index == 1
    assert not any(j.is_mine and 2911042 in j.pids for j in out["jobs"].values())
    assert holders[363948].job is not None and holders[363948].job.id.startswith("p363942-")
    clock.advance(120)                                     # the eval task crashes with a Traceback
    out = await d.poll(ticks=3, step=5.0)
    ended = [j for j in out["jobs"].values() if j.key.label == "p590112"]
    assert ended and ended[0].state is JobState.FAILED and "Traceback" in ended[0].raw_state


async def test_cancel_action_round_trip(world: DemoWorld, clock: StepClock) -> None:
    d = Driver(world, clock, "k2")
    await d.run_detect()
    jobs = (await d.poll(ticks=1))["jobs"]
    ctx = d.ctx()
    plan = d.scheduler.plan_action("slurm.cancel", [jobs["3370390"]], {}, ctx)
    res = await d.transport.probe(ProbeScript(plan.sections, NONCE, "action"), timeout=30)
    results = d.scheduler.interpret_action(plan, res.sections, ctx)
    assert [r.ok for r in results] == [True]
    assert world.cluster("k2").jobs[3370390].state == "CANCELLED"
    jobs = (await d.poll(ticks=2))["jobs"]
    assert jobs["3370390"].state is JobState.CANCELLED


def test_driver_clock_is_aware(clock: StepClock) -> None:
    assert isinstance(clock(), datetime) and clock().tzinfo is not None
