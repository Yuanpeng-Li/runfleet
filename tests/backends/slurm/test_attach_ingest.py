"""[WP2] ingest_attach (§7.16.5/§7.16.6) on the field fixtures and synthetic failures: landed job, IDX mapping, GPU
processes → steps → jobs, cgroups, CPU cores, hints and idle streaks, errors with their kinds, caps_patch, srun sacct
rows, the audit verdict, GPU-type feedback into the job's allocation."""

from __future__ import annotations

import dataclasses

import pytest

from runfleet.backends.base import AttachPlan
from runfleet.backends.slurm import attach
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.models import CgroupStats, ErrorKind, JobKey, JobLive
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.backends.factories import at, ctx, result
from tests.backends.slurm.test_attach import collector_with, make_job
from tests.conftest import frame
from tests.backends.slurm.test_attach_parse import HEALTHY, STUCK
from tests.fixtures.attach import Case, load, node_sections, results_for


def setup(name: str, *, job: str | None = None, gpus: int = 1,
          tres: str = "") -> tuple[Case, SlurmCollector, AttachPlan, dict[str, SectionResult]]:
    case = load(name)
    if job is not None:
        case = dataclasses.replace(case, job=job)
    cx = ctx(case.host)
    jobs = [make_job(case.host, j, case.nodes, gpus=gpus, tres=tres if j == case.job else "",
                     user=cx.caps.remote_user)
            for j in dict.fromkeys([case.job, *[j for js in case.jobs.values() for j in js]])]
    col = collector_with(case.host, *jobs)
    plan = attach.build_plan(jobs[0], case.mode, case.nodes, case.jobs_by_node())   # type: ignore[arg-type]
    return case, col, plan, results_for(case, plan)


def run(name: str, **kw: object) -> tuple[JobLive, dict[str, object], SlurmCollector]:
    case, col, plan, res = setup(name, **kw)                               # type: ignore[arg-type]
    live, patch = col.ingest_attach(plan, res, ctx(case.host))
    return live, dict(patch), col


# ---------------------------------------------------------------- field cases
def test_k2_ssh_lands_in_another_job_on_the_node() -> None:
    live, patch, _ = run("k2_three_jobs")
    assert patch == {"node_ssh": "adopt"} and live.mode == "ssh" and live.has_data and not live.errors
    (n,) = live.nodes
    assert (n.node, n.target_job, n.landed_job, n.gpus_of_target) == ("n-40-01", "2332861", "2332863", False)
    (g,) = n.gpus
    assert (g.slurm_idx, g.bus_id, g.uuid) == (1, "00000000:41:00.0", "GPU-741b3ce9-5226-576c-3c99-916aca612697")
    assert g.host == "k2" and g.index == 0 and g.util_pct == 100.0
    (p,) = g.processes
    assert (p.pid, p.job, p.used_mem_mib) == (2023741, JobKey("k2", "2332863"), 24870.0)
    assert p.command is not None and "orb.train" in p.command and p.user == "alice0001"
    assert list(n.cgroups) == ["2332861", "2332862", "2332863"]           # every own job's cgroup is readable
    assert len(n.procs) == 12 and all(pr.task is None for pr in n.procs)
    assert not any(pr.pid in (2102376, 2102377) for pr in n.procs)        # the probe's own session is not shown
    assert n.step_of_pid[2023738] == "2332861.batch" and n.stray_nvidia_smi == 0 and n.cost_ms == 618
    assert (live.clean, live.audit_detail, live.sacct_rows_added) == (True, "clean", 0)
    assert live.hints == () and live.idle_streak == {}                   # GPUs of another job: no idle hint for 2332861


def test_k2_ssh_landing_in_the_target_maps_minor_to_idx_and_cross_checks_scontrol() -> None:
    live, _, col = run("k2_three_jobs", job="2332863")
    (g,) = live.nodes[0].gpus
    assert live.nodes[0].gpus_of_target is True and g.slurm_idx == 1
    case, col, plan, res = setup("k2_three_jobs", job="2332863")
    col._known_idx = lambda raw, node: (3,)                              # type: ignore[method-assign]
    live2, _ = col.ingest_attach(plan, res, ctx("k2"))
    assert live2.nodes[0].gpus[0].slurm_idx is None                      # scontrol disagrees: never a guessed IDX


def test_vega_idle_gpu_needs_two_samples() -> None:
    case, col, plan, res = setup("vega_idle_gpu", gpus=2)
    first, patch = col.ingest_attach(plan, res, ctx("vega"))
    assert patch == {"node_ssh": "adopt"} and first.nodes[0].gpus_of_target is True
    g0, g1 = first.nodes[0].gpus
    assert g1.slurm_idx is None and g1.bus_id == "00000001:C5:00.0"     # no /proc/driver/nvidia read on vega
    assert first.idle_streak == {"GPU-3888ef1e-a214-afbd-a42a-a8e2121a8ef7": 1} and first.hints == ()
    second, _ = col.ingest_attach(plan, res, at(ctx("vega"), 15), previous=first)
    assert second.idle_streak == {"GPU-3888ef1e-a214-afbd-a42a-a8e2121a8ef7": 2}
    assert second.hints == ("GPU 0001:C5:00.0 idle (0%) but holding 82.7G",)
    c = second.nodes[0].cgroups["724655"]
    assert c.cpu_cores == 0.0 and first.nodes[0].cgroups["724655"].cpu_cores is None   # same counters 15 s apart
    rank = next(p for p in g1.processes if p.pid == 129361)
    assert rank.job == JobKey("vega", "724655") and rank.used_mem_mib == 84634.0
    assert {p.pid for p in g0.processes} == {129392, 129361}
    assert not any(p.pid in (161459, 161460, 161578, 161579) for p in first.nodes[0].procs)


def test_idle_streak_resets_when_the_gpu_works_again() -> None:
    case, col, plan, res = setup("vega_idle_gpu", gpus=2)
    prev = dataclasses.replace(col.ingest_attach(plan, res, ctx("vega"))[0],
                               idle_streak={"GPU-3888ef1e-a214-afbd-a42a-a8e2121a8ef7": 5, "GPU-gone": 3})
    busy = res["slurm.attach.node0"].stdout.replace("1, GPU-3888ef1e-a214-afbd-a42a-a8e2121a8ef7, NVIDIA H200 NVL, 0,",
                                                    "1, GPU-3888ef1e-a214-afbd-a42a-a8e2121a8ef7, NVIDIA H200 NVL, 87,")
    res2 = dict(res, **{"slurm.attach.node0": dataclasses.replace(res["slurm.attach.node0"], stdout=busy)})
    live, _ = col.ingest_attach(plan, res2, at(ctx("vega"), 15), previous=prev)
    assert live.idle_streak == {} and live.hints == ()


def test_cpu_cores_from_the_previous_sample() -> None:
    case, col, plan, res = setup("lyra_srun_sluid")
    first, _ = col.ingest_attach(plan, res, ctx("lyra"))
    old = first.nodes[0].cgroups["31367626"]
    prev = dataclasses.replace(first, nodes=(dataclasses.replace(first.nodes[0], cgroups={
        "31367626": dataclasses.replace(old, cpu_usage_usec=(old.cpu_usage_usec or 0) - 45_000_000)}),))
    live, _ = col.ingest_attach(plan, res, at(ctx("lyra"), 15), previous=prev)
    assert live.nodes[0].cgroups["31367626"].cpu_cores == pytest.approx(3.0)
    too_soon, _ = col.ingest_attach(plan, res, at(ctx("lyra"), 0.5), previous=prev)
    assert too_soon.nodes[0].cgroups["31367626"].cpu_cores is None


def test_lyra_srun_sluid_cgroup_idx_and_one_row() -> None:
    live, patch, col = run("lyra_srun_sluid", tres="billing=16,cpu=16,gres/gpu:rtx6000=1,gres/gpu=1,mem=96G,node=1")
    assert patch == {} and live.mode == "srun" and live.sacct_rows_added == 1 and live.clean is True
    (n,) = live.nodes
    assert (n.landed_job, n.gpus_of_target) == ("31367626", True)
    (g,) = n.gpus
    assert (g.index, g.slurm_idx, g.bus_id) == (0, 1, "00000000:01:00.0")    # step index 0 = Slurm IDX 1
    assert g.name == "NVIDIA RTX PRO 6000 Blackwell Server Edition" and g.mem_total_mib == 97887.0
    assert [p.pid for p in g.processes] == [1077036, 1077687, 1078432]
    assert all(p.job == JobKey("lyra", "31367626") for p in g.processes)
    c = n.cgroups["31367626"]
    assert c.shmem == 40595066880 and c.mem_max == 103079215104
    assert n.step_of_pid[1763482] == "31367626.0" and len(n.procs) == 13
    # 78.7 GB of 96 GiB = 76 %: no memory hint; nothing idle
    assert live.hints == ()


def test_gpu_type_feedback_fixes_lyra_rtx6000() -> None:
    _, _, col = run("lyra_srun_sluid", tres="billing=16,cpu=16,gres/gpu:rtx6000=1,gres/gpu=1,mem=96G,node=1")
    job = col._live["31367626"]
    alloc = col._enrich(job, ctx("lyra")).alloc
    assert (alloc.gpus, alloc.gpu_type, alloc.gpu_type_source) == (1, "nvidia_rtx_pro_6000_blackwell_server_edition",
                                                                   "nvidia-smi")
    assert col._meta["31367626"].gpu_type == "nvidia_rtx_pro_6000_blackwell_server_edition"   # jobmeta keeps it
    before = collector_with("lyra", make_job("lyra", "31367626", tres="gres/gpu:rtx6000=1,gres/gpu=1"))
    plain = before._enrich(before._live["31367626"], ctx("lyra")).alloc
    assert (plain.gpu_type, plain.gpu_type_source) == ("rtx6000", "tres-alloc")


def test_no_gpu_model_from_another_jobs_gpus() -> None:
    _, _, col = run("k2_three_jobs")
    assert col._smi_model == {}


def test_step_without_gpus_reports_no_devices() -> None:
    live, patch, _ = run("k2_gres_none")
    (n,) = live.nodes
    assert n.gpus == () and n.cgroups["2332862"].mem_current == 10682085376
    assert [e.summary for e in n.errors] == ["no GPU visible on n-40-01 (nvidia-smi: No devices found)"]
    assert n.errors[0].section == "slurm.attach.node0" and n.errors[0].kind is ErrorKind.REMOTE
    assert live.sacct_rows_added == 1 and patch == {}
    cpu_only, _, _ = run("k2_gres_none", gpus=0)
    assert cpu_only.nodes[0].errors == ()                               # a CPU-only job has no GPU to miss


def test_pam_slurm_adopt_deny() -> None:
    live, patch, _ = run("vega_adopt_deny")
    assert live.nodes == () and not live.has_data and patch == {}       # a stale node is no auth refusal
    (e,) = live.errors
    assert (e.kind, e.summary, e.section) == (ErrorKind.PERMISSION, "no job of yours on n-38-06 — job moved or ended?",
                                              "slurm.attach.node0")
    assert live.clean is True


def test_lyra_publickey_deny_learns_node_ssh_denied() -> None:
    live, patch, _ = run("lyra_publickey_deny")
    assert patch == {"node_ssh": "denied"} and live.nodes == ()
    (e,) = live.errors
    assert e.kind is ErrorKind.PERMISSION and e.summary == "ssh to lyra-gpu-s71-02 refused (Permission denied)"
    assert "publickey" in e.detail


# ---------------------------------------------------------------- synthetic hop failures
def hop(stderr: str, rc: int | None = 255, *, mode: str = "ssh", stdout: str = "",
        outcome: SectionOutcome | None = None) -> JobLive:
    job = make_job()
    col = collector_with("k2", job)
    plan = col.plan_attach(job, mode, ctx("k2"))                        # type: ignore[arg-type]
    res = {"slurm.attach.node0": result("slurm.attach.node0", stdout, rc, stderr, outcome=outcome)}
    live, patch = col.ingest_attach(plan, res, ctx("k2"))
    assert patch == {}
    return live


@pytest.mark.parametrize("stderr,rc,kind,summary", [
    ("@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key for n-40-01 has changed and you have "
     "requested strict checking.\nHost key verification failed.", 255, ErrorKind.HOSTKEY,
     "n-40-01's host key changed (login host's known_hosts)"),
    ("ssh: Could not resolve hostname n-40-01: Name or service not known", 255, ErrorKind.DNS,
     "login host cannot resolve n-40-01"),
    ("ssh: connect to host n-40-01 port 22: Connection timed out", 255, ErrorKind.TIMEOUT, "ssh to n-40-01 timed out"),
    ("ssh: connect to host n-40-01 port 22: Connection refused", 255, ErrorKind.REFUSED,
     "ssh to n-40-01: ssh: connect to host n-40-01 port 22: Connection refused"),
    ("", 255, ErrorKind.SSH, "ssh to n-40-01 closed without a message — no job of yours there?"),
    ("", 124, ErrorKind.TIMEOUT, "ssh to n-40-01 timed out"),
    ("kex_exchange_identification: read: Connection reset by peer", 255, ErrorKind.SSH,
     "ssh to n-40-01: kex_exchange_identification: read: Connection reset by peer"),
])
def test_ssh_hop_failures_are_classified(stderr: str, rc: int, kind: ErrorKind, summary: str) -> None:
    live = hop(stderr, rc)
    (e,) = live.errors
    assert (e.kind, e.summary, e.section) == (kind, summary, "slurm.attach.node0")
    if kind is ErrorKind.HOSTKEY:
        assert "ssh-keygen -R n-40-01" in e.detail and "ssh n-40-01 true" in e.detail


def test_hop_not_run_or_missing_tool() -> None:
    live = hop("", None, outcome=SectionOutcome.NOT_RUN)
    assert live.errors[0].summary == "slurm.attach.node0 did not finish (not_run)"
    live = hop("missing:ssh", 127)
    assert live.errors[0].kind is ErrorKind.MISSING and "ssh not installed" in live.errors[0].summary


@pytest.mark.parametrize("stderr,rc,outcome,rows", [
    ("srun: error: Unable to create step for job 2332861: Job/step already completing or completed", 1, None, 0),
    ("srun: error: Invalid job id specified", 1, None, 0),
    ("", 124, None, 1),                                                  # timed out: the step may exist
    ("srun: error: n-40-01: task 0: Exited with exit code 2", 2, None, 1),
    ("", None, SectionOutcome.NOT_RUN, 0),
    ("missing:srun", 127, None, 0),
])
def test_srun_rows_only_when_a_step_was_created(stderr: str, rc: int | None, outcome: SectionOutcome | None,
                                                rows: int) -> None:
    live = hop(stderr, rc, mode="srun", outcome=outcome)
    assert live.sacct_rows_added == rows
    if rc == 1:
        assert live.errors[0].summary.startswith("srun on n-40-01: srun: error:")


def test_srun_never_denies_node_ssh() -> None:
    live = hop("Permission denied (publickey).", 1, mode="srun")
    assert live.errors[0].kind is ErrorKind.REMOTE


# ---------------------------------------------------------------- node-level problems
def with_inner(name: str, **edits: tuple[str, int, str]) -> JobLive:
    case, col, plan, res = setup(name)
    inner = node_sections(case, 0)
    assert inner is not None
    inner.update(edits)
    order = [s.name for s in plan.inner["slurm.attach.node0"].sections if not s.late] + \
            [s.name for s in plan.inner["slurm.attach.node0"].sections if s.late]
    nonce = plan.inner["slurm.attach.node0"].nonce
    stdout = frame({s: inner[s] for s in order if s in inner}, nonce)[:-1]
    res["slurm.attach.node0"] = result("slurm.attach.node0", stdout, 0, "")
    return col.ingest_attach(plan, res, ctx(case.host))[0]


def test_hung_nvidia_smi_and_stray_count_make_the_audit_dirty() -> None:
    live = with_inner("lyra_srun_sluid", **{"node.gpu": ("", 124, ""), "node.apps": ("", 125, "nvidia-smi already "
                                                                                     "running (hung?)"),
                                            "node.self": (f"{STUCK}\n@@again\n{STUCK}\n@@done", 0, "")})
    (n,) = live.nodes
    assert n.gpus == () and [e.summary for e in n.errors] == ["nvidia-smi hung on lyra-gpu-s71-03"]
    assert (n.stray_nvidia_smi, n.other_nvidia_smi) == (1, 0)
    assert (live.clean, live.audit_detail) == (False, "1 hung nvidia-smi on lyra-gpu-s71-03")


def test_healthy_nvidia_smi_of_a_job_is_no_leftover() -> None:
    """A training job's own `nvidia-smi -l` / `dmon` (not in D state) is reported apart, never as hung."""
    live = with_inner("vega_idle_gpu", **{"node.self": (f"{HEALTHY}\n@@done", 0, "")})
    (n,) = live.nodes
    assert (n.stray_nvidia_smi, n.other_nvidia_smi) == (0, 2) and n.errors == ()
    assert (live.clean, live.audit_detail) == (True, "clean")


def test_d_state_for_a_moment_is_no_leftover() -> None:
    flicker = f"{HEALTHY}\n{STUCK}\n@@again\n{HEALTHY}\n{STUCK.replace(' D ', ' S ')}\n@@done"
    live = with_inner("vega_idle_gpu", **{"node.self": (flicker, 0, "")})
    assert (live.nodes[0].stray_nvidia_smi, live.nodes[0].other_nvidia_smi) == (0, 3)
    assert live.clean is True


def cut_before_self(name: str, tail: str = "") -> tuple[SlurmCollector, AttachPlan, dict[str, SectionResult]]:
    """The node timeout cut the script before node.self ended (fixture truncated; `tail` may begin node.self)."""
    case, col, plan, res = setup(name)
    inner = node_sections(case, 0)
    assert inner is not None
    nonce = plan.inner["slurm.attach.node0"].nonce
    inner.pop("node.self")
    order = [s for s in attach.NODE_SECTIONS if s in inner]
    stdout = frame({s: inner[s] for s in order}, nonce) + tail.format(nonce=nonce)
    res["slurm.attach.node0"] = result("slurm.attach.node0", stdout, 124, "")
    return col, plan, res


@pytest.mark.parametrize("tail", ["", "\n@@RF{nonce} B node.self\n"])
def test_audit_without_the_node_cleanup_check_is_inconclusive(tail: str) -> None:
    """A node that ran the script must answer node.self; cut off (never begun or begun only) the audit cannot say
    clean even though squeue and the login host are clean: a D-state nvidia-smi may remain."""
    col, plan, res = cut_before_self("vega_idle_gpu", tail)
    live, _ = col.ingest_attach(plan, res, ctx("vega"))
    (n,) = live.nodes
    assert n.stray_nvidia_smi is None and n.gpus                        # the readings themselves are kept
    assert live.clean is None
    assert live.audit_detail == "audit inconclusive: no leftover check from n-37-08 (node.self did not finish)"


def test_audit_with_an_unreadable_node_cleanup_check_is_inconclusive() -> None:
    live = with_inner("vega_idle_gpu", **{"node.self": ("0", 0, "")})
    assert live.nodes[0].stray_nvidia_smi is None and live.clean is None
    assert live.audit_detail == "audit inconclusive: no leftover check from n-37-08 (node.self output unreadable)"


def test_leftovers_found_elsewhere_still_make_the_audit_dirty() -> None:
    col, plan, res = cut_before_self("vega_idle_gpu")
    res["slurm.attach.audit"] = result("slurm.attach.audit", "@@steps\n724655.batch batch\n@@ssh\n1", 0, "")
    live, _ = col.ingest_attach(plan, res, ctx("vega"))
    assert live.clean is False and live.audit_detail == "1 node ssh client still running on the login node"


def test_hop_that_timed_out_before_any_output_is_inconclusive() -> None:
    """ssh/srun killed by its timeout with nothing printed: the node script may have started."""
    case, col, plan, res = setup("vega_idle_gpu")
    res["slurm.attach.node0"] = result("slurm.attach.node0", "", 124, "")
    live, _ = col.ingest_attach(plan, res, ctx("vega"))
    assert live.nodes == () and live.clean is None
    assert live.audit_detail == "audit inconclusive: no leftover check from n-37-08 (ssh timed out)"


@pytest.mark.parametrize(("stderr", "clean"), [
    ("Connection to n-37-08 closed by remote host.\nConnection reset by peer", None),   # after the command started
    ("Read from remote host n-37-08: Connection timed out", None),
    ("ssh: connect to host n-37-08 port 22: Connection timed out", True),               # never had a session
    ("ssh: connect to host n-37-08 port 22: Connection refused", True),
    ("alice0001@n-37-08: Permission denied (publickey,hostbased).", True),
    ("", None),                     # pam_slurm_adopt under LogLevel=ERROR, or ssh killed after the command started
])
def test_hop_failures_are_clean_only_when_nothing_started(stderr: str, clean: bool | None) -> None:
    case, col, plan, res = setup("vega_idle_gpu")
    res["slurm.attach.node0"] = result("slurm.attach.node0", "", 255, stderr)
    live, _ = col.ingest_attach(plan, res, ctx("vega"))
    assert live.nodes == () and live.clean is clean, live.audit_detail
    if clean is None:
        assert live.audit_detail == "audit inconclusive: no leftover check from n-37-08 (ssh failed (rc 255))"


def test_driver_not_loaded_and_cut_off_script() -> None:
    case, col, plan, res = setup("lyra_srun_sluid")
    inner = node_sections(case, 0)
    assert inner is not None
    nonce = plan.inner["slurm.attach.node0"].nonce
    early = {k: inner[k] for k in ("node.id", "node.cg", "node.ps", "node.pids", "node.map")}
    early["node.map"] = ("", 2, "grep: /proc/driver/nvidia/gpus/*/information: No such file or directory")
    text = frame(early, nonce) + f"\n@@RF{nonce} B node.gpu\n"
    res["slurm.attach.node0"] = result("slurm.attach.node0", text, 124, "")
    live, _ = col.ingest_attach(plan, res, ctx("lyra"))
    (n,) = live.nodes
    assert n.cgroups and n.procs and n.gpus == ()
    summaries = [e.summary for e in n.errors]
    assert "nvidia-smi hung on lyra-gpu-s71-03" in summaries                  # begun, never ended
    assert "node script on lyra-gpu-s71-03 was cut off before node.gpu" in summaries
    assert live.sacct_rows_added == 1                                      # the step ran

    drv = with_inner("lyra_srun_sluid", **{"node.gpu": ("", 9, "NVIDIA-SMI has failed because it couldn't "
                                                               "communicate with the NVIDIA driver.")})
    assert "NVIDIA driver not loaded on lyra-gpu-s71-03" in [e.summary for e in drv.nodes[0].errors]


def test_parse_error_and_missing_target_cgroup() -> None:
    live = with_inner("lyra_srun_sluid", **{"node.gpu": ("garbage line\nmore garbage", 0, ""),
                                            "node.cg": ("@@cg 31367626 -", 0, "")})
    errs = {e.summary: e.kind for e in live.nodes[0].errors}
    assert errs == {"lyra-gpu-s71-03: could not parse nvidia-smi output": ErrorKind.PARSE,
                    "lyra-gpu-s71-03: cgroup of job 31367626 not found": ErrorKind.REMOTE}
    assert live.nodes[0].cgroups == {}


# ---------------------------------------------------------------- hints
def test_oom_kill_and_memory_hints() -> None:
    cg = ("@@cg 31367626 /sys/fs/cgroup/system.slice/slurmstepd.scope/sQ4HR26FE80M00\nmemory.current 98000000000\n"
          "memory.max 103079215104\noom 2\noom_kill 2\nshmem 43594000000\n")
    live = with_inner("lyra_srun_sluid", **{"node.cg": (cg, 0, "")})
    assert live.hints == ("oom_kill 2 — a process of this job was killed for memory",
                          "memory 95% of the job's limit (incl. shmem 40.6G)")


def test_hints_name_the_node_on_multi_node_jobs() -> None:
    gpu = dataclasses.replace(load_gpu(), slurm_idx=3)
    nodes = tuple(attach.NodeInspect(node=f"n{i}", target_job="1", landed_job="1", gpus_of_target=True, gpus=(gpu,),
                                     cgroups={"1": CgroupStats(oom_kill=1)}) for i in (1, 2))
    plan = make_plan_for("1", ("n1", "n2"))
    prev = JobLive(key=plan.key, mode="ssh", sampled_at=ctx("k2").now, idle_streak={gpu.uuid or "": 1})
    hints, streak = attach._hints(plan, nodes, prev)
    assert hints[0] == "GPU3 idle (0%) but holding 82.7G on n1" and "oom_kill 1" in hints[1] and hints[1].endswith("n1")
    assert streak == {gpu.uuid: 2}


def load_gpu():  # type: ignore[no-untyped-def]
    _, col, plan, res = setup("vega_idle_gpu", gpus=2)
    return col.ingest_attach(plan, res, ctx("vega"))[0].nodes[0].gpus[1]


def make_plan_for(raw: str, nodes: tuple[str, ...]) -> AttachPlan:
    job = make_job("k2", raw, nodes)
    return attach.build_plan(job, "ssh", nodes, {n: [(raw, "")] for n in nodes})


# ---------------------------------------------------------------- audit
@pytest.mark.parametrize("stdout,rc,outcome,clean,detail", [
    ("@@steps\n2332861.batch batch\n2332861.3 rfprobe\n@@ssh\n0", 0, None, False, "rfprobe step 2332861.3 still running"),
    ("@@steps\n2332861.batch batch\n@@ssh\n2", 0, None, False, "2 node ssh clients still running on the login node"),
    ("@@steps\nslurm_load_jobs error: Invalid job id specified\n@@ssh\n0", 0, None, None,
     "audit inconclusive: slurm_load_jobs error: Invalid job id specified"),
    ("", None, SectionOutcome.NOT_RUN, None, "audit did not run"),
    ("@@steps\n2332861.batch batch\n", None, SectionOutcome.INCOMPLETE, None,
     "audit inconclusive: slurm.attach.audit did not finish (incomplete)"),
])
def test_audit_verdicts(stdout: str, rc: int | None, outcome: SectionOutcome | None, clean: bool | None,
                        detail: str) -> None:
    case, col, plan, res = setup("k2_three_jobs")
    res["slurm.attach.audit"] = result("slurm.attach.audit", stdout, rc, "", outcome=outcome)
    live, _ = col.ingest_attach(plan, res, ctx("k2"))
    assert (live.clean, live.audit_detail) == (clean, detail)


def test_audit_missing_from_results() -> None:
    case, col, plan, res = setup("k2_three_jobs")
    del res["slurm.attach.audit"]
    live, _ = col.ingest_attach(plan, res, ctx("k2"))
    assert (live.clean, live.audit_detail) == (None, "audit did not run") and live.has_data


def test_empty_results_never_raise() -> None:
    case, col, plan, _ = setup("k2_three_jobs")
    live, patch = col.ingest_attach(plan, {}, ctx("k2"))
    assert live.nodes == () and patch == {} and live.errors and live.clean is None
