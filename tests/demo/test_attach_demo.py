"""Demo Attach round trip (§7.16.8): the real SlurmCollector plans an attach sample, DemoTransport answers the node
hops and the audit in the real framed format (nested node script with the plan's inner nonce), the collector ingests
it. Cases: k2 three jobs on one node (ssh lands in the newest), vega GPU1 idle holding 85 GB, lyra ssh refused and
srun working."""

from __future__ import annotations

import asyncio

from runfleet.backends.slurm import attach
from runfleet.demo import render
from runfleet.demo.transport import DemoTransport, handler_for
from runfleet.demo.world import DemoWorld
from runfleet.models import ErrorKind, JobLive
from runfleet.probe import framing
from tests.demo.conftest import StepClock
from tests.demo.test_roundtrip import Driver


def sample(world: DemoWorld, clock: StepClock, host: str, job_id: str, mode: str, *, n: int = 1,
           ) -> tuple[list[JobLive], dict[str, object], Driver]:
    async def go() -> tuple[list[JobLive], dict[str, object], Driver]:
        d = Driver(world, clock, host)
        await d.run_detect()
        for _ in range(3):
            await d.tick()
            clock.advance(20)
        sched = d.scheduler
        job = sched._live[job_id]                                  # type: ignore[attr-defined]
        assert mode in sched.attach_modes(job, d.ctx())
        plan = sched.plan_attach(job, mode, d.ctx())               # type: ignore[arg-type]
        lives: list[JobLive] = []
        patch: dict[str, object] = {}
        prev = None
        for _ in range(n):
            res = await d.transport.probe(framing.make_script(plan.sections, "attach"), timeout=60)
            assert res.error is None
            live, p = sched.ingest_attach(plan, res.sections, d.ctx(), previous=prev)
            lives.append(live)
            patch.update(p)
            prev = live
            clock.advance(15)
        return lives, patch, d
    return asyncio.run(go())


def test_attach_sections_have_handlers() -> None:
    assert handler_for("slurm.attach.node0") is not None and handler_for("slurm.attach.node3") is not None
    assert handler_for(attach.AUDIT_SECTION) is not None


def test_demo_detects_the_attach_tools(world: DemoWorld, clock: StepClock) -> None:
    async def go() -> frozenset[str]:
        d = Driver(world, clock, "k2")
        return (await d.run_detect()).tools
    assert {"ssh", "srun"} <= asyncio.run(go())


def test_k2_ssh_lands_in_the_newest_job(world: DemoWorld, clock: StepClock) -> None:
    (live,), patch, _ = sample(world, clock, "k2", "3370390", "ssh")
    assert patch == {"node_ssh": "adopt"} and live.has_data and not live.errors
    (n,) = live.nodes
    assert (n.node, n.landed_job, n.gpus_of_target) == ("n-40-01", "3370392", False)
    assert set(n.cgroups) == {"3370390", "3370391", "3370392"}
    (g,) = n.gpus
    assert g.slurm_idx == 1 and g.name == "NVIDIA A100-SXM4-40GB"           # 3370392 holds IDX 1
    assert all(p.job is not None and p.job.id == "3370392" for p in g.processes)
    assert n.procs and not any(p.args.startswith(("sshd-session", "timeout -k")) for p in n.procs)
    assert (live.clean, live.sacct_rows_added) == (True, 0)


def test_k2_stalled_job_gets_the_idle_hint_on_the_second_sample(world: DemoWorld, clock: StepClock) -> None:
    (first, second), _, _ = sample(world, clock, "k2", "3370392", "ssh", n=2)
    assert first.hints == () and second.hints == ("GPU1 idle (1%) but holding 24.3G",)
    assert second.nodes[0].cgroups["3370392"].cpu_cores is not None


def test_vega_gpu1_idle_holding_85_gb(world: DemoWorld, clock: StepClock) -> None:
    (_, live), _, _ = sample(world, clock, "vega", "823954", "ssh", n=2)
    g0, g1 = live.nodes[0].gpus
    assert g0.util_pct is not None and g0.util_pct >= 95 and (g1.util_pct, g1.mem_used_mib) == (0.0, 84725.0)
    assert live.hints == ("GPU1 idle (0%) but holding 82.7G",)
    assert live.nodes[0].cgroups["823954"].shmem == 57555 * 1024 * 1024


def test_lyra_refuses_ssh_and_srun_reads_with_one_row(world: DemoWorld, clock: StepClock) -> None:
    (denied,), patch, d = sample(world, clock, "lyra", "32378138", "ssh")
    assert patch == {"node_ssh": "denied"} and denied.nodes == ()
    assert denied.errors[0].kind is ErrorKind.PERMISSION
    (live,), patch, _ = sample(world, clock, "lyra", "32378138", "srun")
    assert patch == {} and live.sacct_rows_added == 1 and live.has_data and live.clean is True
    (n,) = live.nodes
    assert n.landed_job == "32378138" and n.gpus_of_target is True
    assert "32378138" in n.cgroups                                          # SLUID directory (Slurm 26.05)
    (g,) = n.gpus
    assert g.slurm_idx == 0 and g.bus_id == "00000000:71:00.0"             # lyra: IDX0 sits on bus 71
    assert world.cluster("lyra").attach_steps[32378138] == 1


def test_srun_on_lyra_rtx6000_feeds_the_gpu_model(world: DemoWorld, clock: StepClock) -> None:
    (live,), _, d = sample(world, clock, "lyra", "32377076_3", "srun")
    sched = d.scheduler
    assert live.nodes[0].gpus[0].name == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    alloc = sched._enrich(sched._live["32377076_3"], d.ctx()).alloc                # type: ignore[attr-defined]
    assert (alloc.gpu_type, alloc.gpu_type_source) == ("nvidia_rtx_pro_6000_blackwell_server_edition", "nvidia-smi")


def test_ssh_to_a_node_without_a_job_is_a_bare_255(world: DemoWorld) -> None:
    t = DemoTransport(world.configs["k2"], world)
    out = render.attach_node(world.cluster("k2"), world.now(), {"mode": "ssh", "node": "n-31-01",
                                                                "job_id_raw": "3370390"})
    assert (out.stdout, out.rc, out.stderr) == ("", 255, "") and t.host == "k2"
