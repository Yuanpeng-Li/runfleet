"""[WP2] Attach planning (§7.16.1/§7.16.3/§7.16.4): technical modes, node choice, exact login-side command lines, the
nested node script, section meta for the demo/fake transports."""

from __future__ import annotations

import base64
import dataclasses
import re
from collections.abc import Sequence

import pytest

from runfleet.backends.slurm import attach
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.backends.slurm.parse_scontrol import JobInfo
from runfleet.errors import AttachBlocked
from runfleet.models import Job, JobKey, JobKind, JobState, NodeAlloc, ResourceAlloc
from runfleet.probe import framing
from tests.backends.factories import ctx

B64 = "QUJD"


def make_job(host: str = "k2", raw: str = "2332861", nodes: Sequence[str] = ("n-40-01",), *,
             batch: str | None = None, state: JobState = JobState.RUNNING, gpus: int = 1, tres: str = "",
             display: str | None = None, user: str = "alice0001", start: float = 0.0) -> Job:
    from datetime import UTC, datetime
    return Job(key=JobKey(host, display or raw), kind=JobKind.SLURM, name="train", user=user, state=state,
               raw_state=state.name, job_id_raw=raw, nodes=tuple(nodes), batch_host=batch or (nodes[0] if nodes else None),
               alloc=ResourceAlloc(len(nodes), 8, None, gpus, None, {}),
               start_time=datetime.fromtimestamp(1788733357 + start, tz=UTC),
               raw={"tres-alloc": tres} if tres else {})


def collector_with(host: str, *jobs: Job) -> SlurmCollector:
    col = SlurmCollector(ctx(host).host)
    col._live = {j.key.id: j for j in jobs}
    col._have_jobs = True
    return col


# ---------------------------------------------------------------- modes
def test_modes_running_job_ssh_first() -> None:
    col = collector_with("k2")
    assert col.attach_modes(make_job(), ctx("k2")) == ("ssh", "srun")


def test_modes_drop_ssh_when_denied_and_srun_without_overlap() -> None:
    col = collector_with("lyra")
    assert col.attach_modes(make_job("lyra"), ctx("lyra", node_ssh="denied")) == ("srun",)
    assert col.attach_modes(make_job("lyra"), ctx("lyra", slurm_version="20.02.7")) == ("ssh",)
    assert col.attach_modes(make_job("lyra"), ctx("lyra", node_ssh="denied", slurm_version="20.02.7")) == ()


def test_modes_srun_needs_srun_on_the_login_host_when_detect_knew_to_look() -> None:
    col = collector_with("k2")
    tools = ctx("k2").caps.tools
    assert col.attach_modes(make_job(), ctx("k2", tools=tools | {"ssh"})) == ("ssh",)          # v0.2 detect: no srun
    assert col.attach_modes(make_job(), ctx("k2", tools=tools | {"ssh", "srun"})) == ("ssh", "srun")
    assert col.attach_modes(make_job(), ctx("k2", tools=tools)) == ("ssh", "srun")             # pre-v0.2 tool list


@pytest.mark.parametrize("job", [make_job(state=JobState.PENDING), make_job(nodes=()),
                                 make_job(raw="123+0"), dataclasses.replace(make_job(), kind=JobKind.TASK)])
def test_modes_none_for_jobs_that_cannot_be_read(job: Job) -> None:
    assert collector_with("k2").attach_modes(job, ctx("k2")) == ()


# ---------------------------------------------------------------- node choice
def test_ssh_plan_samples_up_to_four_nodes_batch_host_first() -> None:
    nodes = [f"n-37-0{i}" for i in range(1, 7)]
    job = make_job(nodes=nodes, batch="n-37-03")
    plan = collector_with("k2", job).plan_attach(job, "ssh", ctx("k2"))
    assert plan.nodes == ("n-37-03", "n-37-01", "n-37-02", "n-37-04")
    assert [s.name for s in plan.sections] == ["slurm.attach.node0", "slurm.attach.node1", "slurm.attach.node2",
                                               "slurm.attach.node3", "slurm.attach.audit"]
    assert plan.sacct_rows == 0 and plan.mode == "ssh" and plan.job_id_raw == "2332861"
    assert set(plan.inner) == {f"slurm.attach.node{i}" for i in range(4)}


def test_node_pins_one_node_and_must_belong_to_the_job() -> None:
    job = make_job(nodes=["n-37-01", "n-37-02"])
    col = collector_with("k2", job)
    assert col.plan_attach(job, "ssh", ctx("k2"), node="n-37-02").nodes == ("n-37-02",)
    with pytest.raises(AttachBlocked, match="not a node of job"):
        col.plan_attach(job, "ssh", ctx("k2"), node="n-29-99")


def test_srun_plan_is_one_node_one_row() -> None:
    job = make_job("lyra", "31367626", ["lyra-gpu-s71-03", "lyra-gpu-s71-02"], batch="lyra-gpu-s71-02")
    plan = collector_with("lyra", job).plan_attach(job, "srun", ctx("lyra"))
    assert plan.nodes == ("lyra-gpu-s71-02",) and plan.sacct_rows == 1
    assert plan.preview.startswith("timeout -k 2 25 srun --jobid=31367626 --overlap --job-name=rfprobe ")


@pytest.mark.parametrize("job,why", [
    (make_job(state=JobState.COMPLETED), "not running"),
    (make_job(nodes=()), "no node"),
    (make_job(raw="2332861_[1-3]"), "cannot be attached"),
    (make_job(nodes=["n-40-01;reboot"]), "unsafe node name"),
])
def test_plan_blocks_jobs_it_cannot_read(job: Job, why: str) -> None:
    with pytest.raises(AttachBlocked, match=why):
        collector_with("k2", job).plan_attach(job, "ssh", ctx("k2"))


def test_srun_blocked_without_overlap() -> None:
    job = make_job("lyra")
    with pytest.raises(AttachBlocked, match="20.11"):
        collector_with("lyra", job).plan_attach(job, "srun", ctx("lyra", slurm_version="20.02.7"))


# ---------------------------------------------------------------- exact command lines (§7.16.3)
def test_ssh_command_line_golden() -> None:
    assert attach.ssh_cmd("n-37-08", B64) == (
        "timeout -k 2 12 ssh -T -a -x -o BatchMode=yes -o ControlMaster=no -o ControlPath=none "
        "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "
        "-o \"GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts2 $HOME/.ssh/known_hosts "
        "$HOME/.ssh/known_hosts2\" -o UpdateHostKeys=no -o ClearAllForwardings=yes -o ConnectTimeout=5 "
        "-o ServerAliveInterval=5 -o ServerAliveCountMax=1 -o LogLevel=ERROR -- n-37-08 "
        "\"timeout -k 1 10 sh -c 'echo QUJD | base64 -d | sh'\"")


def test_srun_command_line_golden() -> None:
    cmd = attach.srun_cmd("31367626", "lyra-gpu-s71-03", B64)
    assert cmd == ("timeout -k 2 25 srun --jobid=31367626 --overlap --job-name=rfprobe -N1 -n1 -w lyra-gpu-s71-03 "
                   "--time=1 --quiet --input=none timeout -k 1 15 sh -c 'echo QUJD | base64 -d | sh'")
    for forbidden in ("--gres", "--pty", "--exclusive"):       # §7.16.4 layer 4
        assert forbidden not in cmd


def test_audit_command_line_golden() -> None:
    assert attach.audit_cmd("724655") == (
        "echo @@steps; squeue -h -s -j 724655 -o '%i %j' 2>&1; echo @@ssh; "
        "pgrep -u \"$(id -un)\" -f 'ControlPat[h]=none' | wc -l; true")
    assert "%T" not in attach.audit_cmd("724655")


def test_plan_sections_timeouts_requires_meta() -> None:
    job = make_job()
    plan = collector_with("k2", job).plan_attach(job, "ssh", ctx("k2"))
    node0, audit = plan.sections
    assert (node0.timeout, node0.requires, node0.late) == (14.0, ("timeout", "ssh"), False)
    assert (audit.timeout, audit.late) == (10.0, True)
    inner = plan.inner["slurm.attach.node0"]
    assert node0.meta == {"mode": "ssh", "node": "n-40-01", "job_id_raw": "2332861", "jobs_raw": ["2332861"],
                          "sluids": [""], "inner_nonce": inner.nonce, "inner": list(attach.NODE_SECTIONS)}
    assert audit.meta == {"mode": "ssh", "job_id_raw": "2332861", "nodes": ["n-40-01"]}
    assert framing.local_timeout(framing.make_script(plan.sections, "attach")) == 35.0   # max(20, 14) + 10 + 5
    srun = collector_with("k2", job).plan_attach(job, "srun", ctx("k2")).sections[0]
    assert (srun.timeout, srun.requires) == (27.0, ("timeout", "srun"))


def test_nested_script_is_the_rendered_node_probe_without_temp_dir() -> None:
    job = make_job()
    plan = collector_with("k2", job).plan_attach(job, "ssh", ctx("k2"))
    cmd = plan.sections[0].cmd
    b64 = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", cmd)
    assert b64 is not None
    script = base64.b64decode(b64.group(1)).decode()
    inner = plan.inner["slurm.attach.node0"]
    assert script == framing.render(inner, tmp=False)
    assert "mktemp" not in script and "\nRF_T=\n" in script            # layer 3: nothing written on the node
    assert inner.purpose == "attach" and inner.nonce != ""
    assert [s.name for s in framing.ordered(inner)] == list(attach.NODE_SECTIONS)
    assert all(s.timeout == 5.0 for s in inner.sections)
    assert {s.name for s in inner.sections if s.late} == {"node.gpu", "node.apps", "node.self"}
    assert "pci.bus_id" in next(s.cmd for s in inner.sections if s.name == "node.gpu")
    assert plan.preview == attach.preview_of(cmd, b64.group(1))
    assert "<read-only node script, " in plan.preview and b64.group(1) not in plan.preview


def test_ssh_hop_never_writes_known_hosts_or_keeps_a_master() -> None:
    """Known node keys must match (system + user files read as global files); an unknown node is accepted but never
    recorded: accept-new writes only to UserKnownHostsFile, which is /dev/null."""
    cmd = attach.ssh_cmd("n1", B64)
    for opt in ("ControlMaster=no", "ControlPath=none", "StrictHostKeyChecking=accept-new",
                "UserKnownHostsFile=/dev/null", "UpdateHostKeys=no", "BatchMode=yes"):
        assert f"-o {opt}" in cmd
    assert '"GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts2 $HOME/.ssh/known_hosts ' in cmd
    assert "StrictHostKeyChecking=no" not in cmd and "StrictHostKeyChecking=yes" not in cmd


def test_plan_reads_the_users_other_jobs_on_the_node_with_sluids() -> None:
    target = make_job("lyra", "31367725", ["lyra-gpu-s71-02"], user="alice07", start=0)
    other = make_job("lyra", "31367726", ["lyra-gpu-s71-02"], user="alice07", start=10)
    elsewhere = make_job("lyra", "31367626", ["lyra-gpu-s71-03"], user="alice07")
    col = collector_with("lyra", target, other, elsewhere)
    for job, sluid in ((target, "sQ4HR2Z1TS9G00"), (other, "sQ4HR2Z1TS9K00")):
        col._jobinfo[job.key.id] = JobInfo(job.job_id_raw, None, None, {"SLUID": sluid}, (), None, None, None, None)
    plan = col.plan_attach(target, "ssh", ctx("lyra"))
    meta = plan.sections[0].meta
    assert meta["jobs_raw"] == ["31367725", "31367726"]
    assert meta["sluids"] == ["sQ4HR2Z1TS9G00", "sQ4HR2Z1TS9K00"]
    cg = next(s.cmd for s in plan.inner["slurm.attach.node0"].sections if s.name == "node.cg")
    assert "for p in 31367725:sQ4HR2Z1TS9G00 31367726:sQ4HR2Z1TS9K00; do" in cg
    pids = next(s.cmd for s in plan.inner["slurm.attach.node0"].sections if s.name == "node.pids")
    assert pids == 'for j in 31367725 31367726; do scontrol listpids "$j"; done'


def test_unsafe_sluid_is_dropped() -> None:
    job = make_job()
    plan = attach.build_plan(job, "ssh", ("n-40-01",), {"n-40-01": [("2332861", "s1;rm -rf /")]})
    assert plan.sections[0].meta["sluids"] == [""]


def test_known_idx_reads_scontrol_per_node_indices() -> None:
    job = dataclasses.replace(make_job(), alloc=ResourceAlloc(1, 8, None, 1, None, {}, per_node=(
        NodeAlloc("n-40-01", "64-71", None, (2,), "gpu:nvidia_a100-sxm4-40gb:1(IDX:2)"),)))
    col = collector_with("k2", job)
    assert col._known_idx("2332861", "n-40-01") == (2,)
    assert col._known_idx("2332861", "n-40-02") == () and col._known_idx("1", "n-40-01") == ()
