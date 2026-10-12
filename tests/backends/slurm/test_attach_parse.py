"""[WP2] Attach parsers (§7.16.3/§7.16.5) on the field fixtures of tests/fixtures/attach: node.id (job_<N> and SLUID
cgroups), node.cg (cgroup v2 blocks), node.ps (probe tree removed), scontrol listpids, /proc/driver/nvidia, audit."""

from __future__ import annotations

import subprocess

import pytest

from runfleet.backends.common.cgroup import is_empty, parse_cgroup_block
from runfleet.backends.common.gpu_types import mem_mib
from runfleet.backends.common.nvidia import parse_gpu_query
from runfleet.backends.slurm import attach
from runfleet.backends.slurm import parse_attach as pa
from runfleet.models import CgroupStats
from tests.fixtures.attach import load, node_sections


def section(case: str, name: str, node: int = 0) -> str:
    inner = node_sections(load(case), node)
    assert inner is not None
    return inner[name][0]


# ---------------------------------------------------------------- node.id
def test_node_id_ssh_lands_in_the_newest_job_k2() -> None:
    nid = pa.parse_node_id(section("k2_three_jobs", "node.id"))
    assert nid.hostname == "n-40-01" and nid.cgroup_dir == "job_2332863"
    assert nid.cgroup == "/system.slice/n-40-01_slurmstepd.scope/job_2332863/step_extern/user/task_0"
    assert nid.env_job == "2332863" and nid.step_gpus is None
    assert pa.landed_job(nid, ["2332861", "2332862", "2332863"], ["", "", ""]) == "2332863"


def test_node_id_sluid_directory_maps_back_through_the_plan_lyra() -> None:
    nid = pa.parse_node_id(section("lyra_srun_sluid", "node.id"))
    assert nid.cgroup_dir == "sQ4HR26FE80M00" and nid.step_gpus == (1,)
    assert pa.landed_job(nid, ["31367626"], ["sQ4HR26FE80M00"]) == "31367626"
    unknown = pa.NodeId("n", "/system.slice/slurmstepd.scope/sXX/step_0", "sXX", None, None)
    assert pa.landed_job(unknown, ["1"], [""]) is None                  # neither job_<N>, a known SLUID nor $J


def test_node_id_outside_a_job_falls_back_to_slurm_job_id() -> None:
    nid = pa.parse_node_id("n1\n0::/user.slice/user-610001.slice/session-3.scope\nJ=4242 G=0,1\n")
    assert nid.cgroup_dir is None and pa.landed_job(nid, ["1"], [""]) == "4242" and nid.step_gpus == (0, 1)


@pytest.mark.parametrize("text,want", [("0,1", (0, 1)), ("2", (2,)), ("0-3", (0, 1, 2, 3)), ("1,4-5", (1, 4, 5)),
                                       ("", None), ("x", None), ("1-", None)])
def test_idx_list(text: str, want: tuple[int, ...] | None) -> None:
    assert pa.parse_idx_list(text) == want


def test_scope_sed_of_the_node_script_finds_both_layouts() -> None:
    """The node.cg script's own sed expression, run for real on the k2/vega and lyra cgroup lines."""
    expr = "s#^0::(/system\\.slice/[^/]*slurmstepd\\.scope)(/.*)?$#\\1#p"
    assert expr in attach.cg_cmd(["1"], [""])
    lines = {"0::/system.slice/n-40-01_slurmstepd.scope/job_2332863/step_extern/user/task_0":
             "/system.slice/n-40-01_slurmstepd.scope",
             "0::/system.slice/slurmstepd.scope/sQ4HR26FE80M00/step_0/user/task_0": "/system.slice/slurmstepd.scope",
             "0::/user.slice/user-1.slice/session-3.scope": ""}
    for line, want in lines.items():
        out = subprocess.run(["sed", "-n", "-E", expr], input=line + "\n", capture_output=True, text=True, check=True)
        assert out.stdout.strip() == want


# ---------------------------------------------------------------- node.cg / cgroup v2
def test_cgroups_three_jobs_on_one_node_k2() -> None:
    cg = pa.parse_cgroups(section("k2_three_jobs", "node.cg"))
    assert list(cg) == ["2332861", "2332862", "2332863"]
    c = cg["2332861"]
    assert c == CgroupStats(mem_current=10693746688, mem_peak=11822579712, mem_max=51539607552, anon=10391187456,
                            file=266776576, shmem=19664896, oom=0, oom_kill=0, cpu_usage_usec=60723561547)


def test_cgroup_sluid_dir_and_shmem_lyra() -> None:
    c = pa.parse_cgroups(section("lyra_srun_sluid", "node.cg"))["31367626"]
    assert c is not None
    assert (c.mem_current, c.mem_max, c.shmem, c.anon) == (78738329600, 103079215104, 40595066880, 36293398528)
    assert c.oom_kill == 0 and c.cpu_usage_usec == 115399681972


def test_cgroup_block_max_is_unlimited_and_missing_files_stay_none() -> None:
    # vega step_batch (vega/04): memory.max "max"; memory.peak absent on kernels < 5.19
    c = parse_cgroup_block("memory.current 143638327296\nmemory.max max\nlow 0\nhigh 0\nmax 0\noom 0\noom_kill 3\n"
                           "oom_group_kill 0\nanon 82757595136\nshmem 60351225856\nusage_usec 18521195620\n")
    assert c.mem_max is None and c.mem_peak is None and c.file is None
    assert (c.mem_current, c.oom_kill, c.oom, c.shmem) == (143638327296, 3, 0, 60351225856)
    assert is_empty(parse_cgroup_block("memory.peak \n")) and not is_empty(c)


def test_cgroup_not_found_is_none() -> None:
    cg = pa.parse_cgroups("@@cg 1 -\n@@cg 2 /sys/fs/cgroup/x/job_2\nmemory.current 5\n@@cg bad /x\nmemory.current 9\n")
    assert cg == {"1": None, "2": CgroupStats(mem_current=5)}


# ---------------------------------------------------------------- node.ps
def test_ps_drops_the_probe_tree_k2() -> None:
    self_pid, pr = pa.parse_node_ps(section("k2_three_jobs", "node.ps"), user="alice0001")
    assert self_pid == 2102377 and not pr.bad_lines and len(pr.items) == 17
    probe = pa.probe_pids(pr.items, self_pid)
    assert probe == {2102376, 2102377, 2102438, 2102439, 2102440}      # sshd session + everything below it
    assert {p.pid for p in pr.items} - probe >= {2023571, 2023741, 2023572, 2023738, 2023570, 2023739}
    main = next(p for p in pr.items if p.pid == 2023741)
    assert (main.user, main.cpu_pct, main.rss_bytes, main.stat) == ("alice0001", 106.0, 10778856 * 1024, "Sl")


def test_ps_srun_probe_tree_ends_at_slurmstepd_lyra() -> None:
    self_pid, pr = pa.parse_node_ps(section("lyra_srun_sluid", "node.ps"), user="alice07")
    assert pa.probe_pids(pr.items, self_pid) == {1763477, 1763479, 1763482, 1763509, 1763510}
    assert pa.probe_pids(pr.items, None) == frozenset() and pa.probe_pids(pr.items, 1) == frozenset()


# ---------------------------------------------------------------- node.pids
def test_listpids_three_tables_k2() -> None:
    steps = pa.parse_listpids(section("k2_three_jobs", "node.pids"))
    assert steps[2023738] == "2332861.batch" and steps[2023741] == "2332863.batch"
    assert steps[2102377] == "2332863.extern"                         # the ssh session was adopted into .extern
    assert -1 not in steps and len(steps) == 24


def test_listpids_srun_step_lyra() -> None:
    steps = pa.parse_listpids(section("lyra_srun_sluid", "node.pids"))
    assert steps[1763482] == "31367626.0" and steps[1077036] == "31367626.batch"


# ---------------------------------------------------------------- node.map / node.gpu
def test_gpu_map_k2_minor_equals_slurm_idx() -> None:
    gmap = pa.parse_gpu_map(section("k2_three_jobs", "node.map"))
    assert [(e.bus, e.minor) for e in gmap.values()] == [("0000:01:00.0", 0), ("0000:41:00.0", 1),
                                                         ("0000:81:00.0", 2), ("0000:c1:00.0", 3)]
    e = pa.map_by_uuid_or_bus(gmap, None, "00000000:41:00.0")
    assert e is not None and e.uuid == "GPU-741b3ce9-5226-576c-3c99-916aca612697"
    assert pa.map_by_uuid_or_bus(gmap, "GPU-d4fe0bdf-2969-58a4-cb31-1f8659ae8116", None).minor == 3  # type: ignore
    assert pa.parse_gpu_map(section("vega_idle_gpu", "node.map")) == {}


@pytest.mark.parametrize("bus,want", [("00000000:41:00.0", "0000:41:00.0"), ("00000001:C5:00.0", "0001:c5:00.0"),
                                      ("0000:c1:00.0", "0000:c1:00.0"), ("41:00.0", "0000:41:00.0"),
                                      ("", None), ("garbage", None), (None, None)])
def test_canon_bus(bus: str | None, want: str | None) -> None:
    assert pa.canon_bus(bus) == want


def test_gpu_query_with_bus_id_vega() -> None:
    pr = parse_gpu_query(section("vega_idle_gpu", "node.gpu"), attach.ATTACH_GPU_FIELDS, host="vega")
    g0, g1 = pr.items
    assert (g1.index, g1.util_pct, g1.mem_used_mib, g1.mem_total_mib, g1.bus_id) == (1, 0.0, 84725.0, 143771.0,
                                                                                      "00000001:C5:00.0")
    assert g0.name == "NVIDIA H200 NVL" and g0.power_limit_w == 600.0 and g0.pstate is None and g0.fan_pct is None


def test_no_devices() -> None:
    assert pa.no_devices(section("k2_gres_none", "node.gpu"))
    assert pa.no_devices("No devices were found") and not pa.no_devices("0, GPU-1, X")


def test_gpu_type_from_name_feeds_the_gpu_table() -> None:
    name = parse_gpu_query(section("lyra_srun_sluid", "node.gpu"), attach.ATTACH_GPU_FIELDS, host="lyra").items[0].name
    gtype = pa.gpu_type_from_name(name)
    assert gtype == "nvidia_rtx_pro_6000_blackwell_server_edition" and mem_mib(gtype) == 97887
    assert pa.gpu_type_from_name("NVIDIA A100-SXM4-40GB") == "nvidia_a100-sxm4-40gb"    # = k2's gres type
    assert pa.gpu_type_from_name("NVIDIA H200 NVL") == "nvidia_h200_nvl"


@pytest.mark.parametrize("text,want", [("0\n", 0), ("2", 2), ("", None), ("x", None)])
def test_count(text: str, want: int | None) -> None:
    assert pa.parse_count(text) == want


# ---------------------------------------------------------------- node.self
# `ps -o pid=,ppid=,stat=,etimes=,args=` rows of the user's nvidia-smi processes (field formats of k2/vega ps)
HEALTHY = " 2023901  2023738 Sl     5400 nvidia-smi -l 30\n 2023977  2023740 S      5300 /usr/bin/nvidia-smi dmon"
STUCK = " 2102390        1 D        12 nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits"


@pytest.mark.parametrize("text,stuck,other", [
    ("@@done", (), 0),
    (f"{HEALTHY}\n@@done", (), 2),                                       # a job's own `nvidia-smi -l`: not a leftover
    (f"{HEALTHY}\n{STUCK}\n@@again\n{HEALTHY}\n{STUCK}\n@@done", (2102390,), 2),   # D on both reads: hung
    (f"{STUCK}\n@@again\n{STUCK.replace(' D ', ' S ')}\n@@done", (), 1),  # D for a moment: a query in flight
    (f"{STUCK}\n@@again\n@@done", (), 0),                                # gone by the second read
])
def test_self_check(text: str, stuck: tuple[int, ...], other: int) -> None:
    chk = pa.parse_self(text)
    assert chk is not None and (chk.stuck, chk.other) == (stuck, other)


@pytest.mark.parametrize("text", [
    "",                                     # cut before any output
    "0",                                    # not this format
    HEALTHY,                                # no end marker: the read may be incomplete
    f"{STUCK}\n@@done",                     # a D-state row without its second read
    f"{STUCK}\n@@again\n{STUCK}",            # cut during the second read
    "garbage\n@@done",
])
def test_self_check_incomplete(text: str) -> None:
    assert pa.parse_self(text) is None


# ---------------------------------------------------------------- audit
def test_audit_clean_fixture() -> None:
    a = pa.parse_audit((load("k2_three_jobs").dir / "audit.txt").read_text())
    assert a.steps == (("2332861.batch", "batch"), ("2332861.extern", "extern")) and a.ssh_clients == 0


def test_audit_rfprobe_array_step_and_squeue_error() -> None:
    a = pa.parse_audit("@@steps\n32377076_3.batch batch\n32377076_3.4 rfprobe\n@@ssh\n2\n")
    assert a.steps == (("32377076_3.batch", "batch"), ("32377076_3.4", "rfprobe")) and a.ssh_clients == 2
    bad = pa.parse_audit("@@steps\nslurm_load_jobs error: Invalid job id specified\n@@ssh\n0\n")
    assert bad.steps is None and bad.note.startswith("slurm_load_jobs error") and bad.ssh_clients == 0
    assert pa.parse_audit("").steps is None
    assert pa.parse_audit("@@steps\n@@ssh\n0\n").steps == ()           # job without steps listed (just ended)
