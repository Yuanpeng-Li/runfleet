"""Byte-exact checks of demo/render.py against the field-probe outputs in tests/fixtures/samples (§10.6).

Each test rebuilds the fixture's jobs/nodes/GPUs as simulation objects and renders them with the fixture's command
options (`|` separator, naive local ISO timestamps); the output must equal the captured bytes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from runfleet.demo import render
from runfleet.demo.world import GresDef, SimCluster, SimGPU, SimJob, SimNode, SimProc, SimWorkstation, StepUsage
from tests.conftest import sample

UTC_M5 = timedelta(hours=-5)
UTC_M7 = timedelta(hours=-7)


def epoch(text: str, tz: timedelta) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone(tz)).timestamp()


def k2_cluster() -> SimCluster:
    return SimCluster(name="k2", slurm_version="25.05.8", login_host="login02", user="alice0001", uid=610001,
                      group="rs", gid=500, tz=UTC_M5, merged_stderr="repeat")


def orb_job(job_id: int, name: str, submit: str, *, tz: timedelta = UTC_M5, **kw: object) -> SimJob:
    fields: dict[str, object] = dict(
        user="alice0001", uid=610001, account="campus01", partition="gpu-preempt", qos="gpu-priority",
        time_limit_s=3 * 86400, cpus=8, mem_mib=48 * 1024, workdir="/shared/pilab/pilab/orb",
        command="/shared/pilab/pilab/orb/jobs/train.sbatch", stdout_pattern="/shared/pilab/pilab/orb/logs/%x_%j.out",
        stderr_pattern="/shared/pilab/pilab/orb/logs/%x_%j.out", gpus_per_node=1, gpu_type="nvidia_a100-sxm4-40gb")
    fields.update(kw)
    return SimJob(job_id=job_id, name=name, submit=epoch(submit, tz), **fields)  # type: ignore[arg-type]


def running(job: SimJob, start: str, nodes: tuple[str, ...], *, tz: timedelta = UTC_M5) -> SimJob:
    job.state, job.reason, job.start, job.nodes = "RUNNING", "None", epoch(start, tz), nodes
    return job


def k2_squeue_jobs() -> list[SimJob]:
    diag = "/shared/pilab/pilab/rl-benchmarks"
    lmft = "/shared/pilab/alix/lmft"
    pd = orb_job(2334564, "smoke-test", "2026-09-07T05:39:20", time_limit_s=3600,
                 gpu_type="nvidia_geforce_rtx_3090", workdir=diag, command=f"{diag}/jobs/train.sbatch",
                 stdout_pattern=f"{diag}/logs/%x_%j.out", stderr_pattern=f"{diag}/logs/%x_%j.out",
                 start=epoch("2026-09-07T06:31:41", UTC_M5))
    jobs = [pd]
    for jid, method, submit in ((2332861, "warmup", "2026-09-06T11:14:06"), (2332862, "const", "2026-09-06T11:14:07"),
                                (2332863, "cosinewarm", "2026-09-06T11:14:07")):
        jobs.append(running(orb_job(jid, f"orb-swp--train--cifar-{method}--s43--a02", submit),
                            "2026-09-07T01:50:27", ("n-40-01",)))
    jobs.append(running(orb_job(2333760, "lmft-x07-sampler-s0", "2026-09-06T20:54:51", qos="normal",
                                time_limit_s=2 * 86400, cpus=16, mem_mib=256 * 1024, gpus_per_node=2,
                                gpu_type="nvidia_h200_nvl", typed_alloc=False, typed_req_tres=False, workdir=lmft,
                                command=f"{lmft}/jobs/train.sbatch", stdout_pattern=f"{lmft}/logs/%x_%j.out",
                                stderr_pattern=None),
                        "2026-09-06T20:55:13", ("n-38-01",)))
    return jobs


def test_squeue_O_matches_k2_fixture() -> None:
    c = k2_cluster()
    for job in k2_squeue_jobs():
        c.jobs[job.job_id] = job
    out = render.squeue(c, epoch("2026-09-07T05:57:27", UTC_M5), sep="|", mode="iso")
    assert out.stdout == sample("k2", "squeue_me_O")


def test_squeue_O_matches_vega_pending_row() -> None:
    c = SimCluster(name="vega", user="alice0001", uid=610001, tz=UTC_M5)
    lmft = "/shared/pilab/alix/lmft"
    job = orb_job(724479, "lmft-x07-trainer-s0", "2026-09-06T21:33:10", account="pilab", partition="h200",
                  qos="normal", time_limit_s=2 * 86400, cpus=16, mem_mib=256 * 1024, gpus_per_node=2,
                  gpu_type="nvidia_h200_nvl", workdir=lmft, command=f"{lmft}/jobs/train.sbatch",
                  stdout_pattern=f"{lmft}/logs/%x_%j.out", stderr_pattern=None, reason="QOSMaxJobsPerUserLimit")
    c.jobs[job.job_id] = job
    out = render.squeue(c, epoch("2026-09-07T05:58:27", UTC_M5), sep="|", mode="iso")
    assert out.stdout == sample("vega", "squeue_me_O").splitlines(keepends=True)[0]


def test_squeue_O_matches_lyra_array_summary_row() -> None:
    c = SimCluster(name="lyra", user="rita", uid=1, tz=UTC_M7)
    wd = "/scratch/rita/md/run1"
    job = SimJob(job_id=31270818, name="train.sh", user="rita", uid=1, account="rita",
                 partition="free-gpu", qos="low", submit=epoch("2026-08-26T12:51:37", UTC_M7), time_limit_s=86400,
                 cpus=1, mem_mib=3 * 1024, workdir=wd, command="./train.sh", stdout_pattern="slurm-%A_%a.out",
                 gpus_per_node=1, gpu_type=None, typed_request=False, typed_alloc=False, reason="Dependency",
                 array_job_id=31270818, array_task="1-24%1", summary=True, array_pending=list(range(1, 25)))
    c.jobs[job.job_id] = job
    out = render.squeue(c, epoch("2026-09-07T03:42:37", UTC_M7), users=["rita"], sep="|", mode="iso")
    assert out.stdout == sample("lyra", "squeue_parts_O").splitlines(keepends=True)[1]


def test_sacct_matches_k2_ext_rows() -> None:
    fixture = sample("k2", "sacct_ext").splitlines(keepends=True)
    fields = ["JobID", "JobIDRaw", "JobName", "User", "Account", "QOS", "Partition", "State", "ExitCode",
              "DerivedExitCode", "Reason", "Submit", "Start", "End", "Elapsed", "Timelimit", "AllocCPUS", "ReqMem",
              "ReqTRES", "AllocTRES", "NNodes", "NodeList", "WorkDir"]
    c = k2_cluster()
    now = epoch("2026-09-07T05:57:28", UTC_M5)
    running_job = running(orb_job(2332863, "orb-swp--train--cifar-cosinewarm--s43--a02", "2026-09-06T11:14:07"),
                          "2026-09-07T01:50:27", ("n-40-01",))
    done = orb_job(2333095, "orb-ab--train--resnet18-bs512--s42--a02", "2026-09-06T13:01:36", time_limit_s=86400,
                   gpu_type="nvidia_rtx_pro_6000_blackwell_server_edition", typed_request=False)
    done.state, done.start, done.end, done.nodes = ("COMPLETED", epoch("2026-09-06T13:13:42", UTC_M5),
                                                    epoch("2026-09-06T18:53:20", UTC_M5), ("n-38-03",))
    lmft = "/shared/pilab/alix/lmft"
    cancelled = orb_job(2333761, "lmft-x07-trainer-s1", "2026-09-06T20:54:52", qos="normal", time_limit_s=2 * 86400,
                        cpus=16, mem_mib=256 * 1024, gpus_per_node=2, gpu_type="nvidia_h200_nvl",
                        typed_req_tres=False, workdir=lmft)
    cancelled.state, cancelled.end, cancelled.cancelled_by = "CANCELLED", epoch("2026-09-06T20:58:58", UTC_M5), 610001
    for job in (running_job, done, cancelled):
        c.jobs[job.job_id] = job
    out = render.sacct_window(c, now, since_s=86400, fields=fields, sep="|", mode="iso")
    assert out.stdout.splitlines(keepends=True) == [fixture[0], fixture[1], fixture[9]]


def test_sstat_matches_k2_fixture() -> None:
    c = k2_cluster()
    job = running(orb_job(2332861, "orb-swp--train--cifar-warmup--s43--a02", "2026-09-06T11:14:06"),
                  "2026-09-07T01:50:27", ("n-40-01",))
    job.batch = StepUsage(ntasks=1, ave_cpu_s=4 * 3600 + 20 * 60 + 2, ave_rss_kb=10398988, max_rss_kb=11506244,
                          disk_bytes=9430412075, pages=3652, gpuutil_ave=98, gpuutil_max=99, gpumem_ave_mib=24870,
                          gpumem_max_mib=24870)
    c.jobs[job.job_id] = job
    fields = ["JobID", "NTasks", "AveCPU", "AveRSS", "MaxRSS", "TRESUsageInAve", "TRESUsageInMax"]
    out = render.sstat(c, 0.0, ids_raw=["2332861"], fields=fields)
    assert out.stdout == sample("k2", "sstat_running")
    assert out.rc == 0


def test_sstat_without_gpu_tres_omits_gres_keys() -> None:
    c = k2_cluster()
    c.gpu_tres = False
    job = running(orb_job(1, "x", "2026-09-06T11:14:06"), "2026-09-07T01:50:27", ("n1",))
    c.jobs[1] = job
    out = render.sstat(c, epoch("2026-09-07T02:42:37", UTC_M5), ids_raw=["1", "2"])
    assert "gres/" not in out.stdout and out.stdout.count("\n") == 2
    assert "couldn't get steps for job 2" in out.stderr


def _k2_nodes() -> list[SimNode]:
    s07 = "(S:0-7)"
    return [
        SimNode("n-31-01", ("gpu-preempt",), 64, 512000, (GresDef("nvidia_geforce_rtx_3090", 8, s07),),
                bg_cpus=34, bg_mem_mib=288768, bg_gpu_idx=set(range(8)), free_mem_mib=23103),
        SimNode("n-31-02", ("gpu-preempt",), 64, 512000, (GresDef("nvidia_geforce_rtx_3090", 8, s07),),
                bg_cpus=40, bg_mem_mib=401408, bg_gpu_idx=set(range(8)), free_mem_mib=40339),
        SimNode("n-31-03", ("gpu-preempt",), 64, 512000, (GresDef("nvidia_geforce_rtx_3090", 8, s07),),
                bg_cpus=44, bg_mem_mib=296960, bg_gpu_idx=set(range(8)), free_mem_mib=22073),
        SimNode("n-31-04", ("gpu-preempt",), 56, 128000, (GresDef("tesla_v100-pcie-16gb", 4, "(S:0)"),),
                bg_cpus=6, bg_mem_mib=73728, bg_gpu_idx={0, 2, 3}, free_mem_mib=13725),
        SimNode("n-31-05", ("gpu-preempt",), 64, 515000,
                (GresDef("nvidia_a100_80gb_pcie", 1, s07), GresDef("nvidia_a100_80gb_pcie_3g.40gb", 2, s07),
                 GresDef("nvidia_a100_3g.20gb", 2, s07), GresDef("nvidia_a100_80gb_pcie_1g.10gb", 7, s07)),
                state="planned", free_mem_mib=59616),
    ]


def test_sinfo_matches_k2_rows() -> None:
    c = k2_cluster()
    c.nodes = _k2_nodes()
    out = render.sinfo_nodes(c, partitions=["lab", "gpu-preempt"], sep="|")
    assert out.stdout.splitlines(keepends=True) == sample("k2", "sinfo_nodes_O").splitlines(keepends=True)[:5]


def test_sinfo_matches_vega_reserved_and_lyra_rows() -> None:
    vega = SimCluster(name="vega")
    vega.nodes = [SimNode("n-37-09", ("h200",), 64, 384000, (GresDef("nvidia_h200_nvl", 2, "(S:3-4)"),),
                          state="reserved", free_mem_mib=245744)]
    line = render.sinfo_nodes(vega, partitions=["h200"], sep="|").stdout
    assert line in sample("vega", "sinfo_nodes_O").splitlines(keepends=True)
    lyra = SimCluster(name="lyra")
    lyra.nodes = [SimNode("lyra-gpu-36-00", ("free-gpu",), 40, 192000, (GresDef("V100", 4),), bg_cpus=40,
                          bg_mem_mib=143360, bg_gpu_idx={0, 1, 2, 3}, free_mem_mib=24485)]
    line = render.sinfo_nodes(lyra, partitions=["free-gpu"], sep="|").stdout
    assert line == sample("lyra", "sinfo_nodes_O").splitlines(keepends=True)[0]


def test_scontrol_matches_k2_running_record() -> None:
    c = k2_cluster()
    job = running(orb_job(2332861, "orb-swp--train--cifar-warmup--s43--a02", "2026-09-06T11:14:06",
                          scontrol_extra={"Priority": "1", "Scheduler": "Backfill",
                                          "AllocNode:Sid": "login02:1730264",
                                          "ExcNodeList": "n-31-[04-05],n-33-[04-06],n-36-[01-03],n-37-02"}),
                  "2026-09-07T01:50:27", ("n-40-01",))
    job.gpu_idx, job.cpu_ids = {"n-40-01": (2,)}, {"n-40-01": "64-71"}
    c.jobs[job.job_id] = job
    out = render.scontrol_record(c, job, epoch("2026-09-07T05:58:07", UTC_M5), mode="iso") + "\n"
    assert out == sample("k2", "scontrol_show_job_running_d")


def test_scontrol_matches_k2_pending_record() -> None:
    c = k2_cluster()
    diag = "/shared/pilab/pilab/rl-benchmarks"
    job = orb_job(2334564, "smoke-test", "2026-09-07T05:39:20", time_limit_s=3600, gpu_type="nvidia_geforce_rtx_3090",
                  workdir=diag, command=f"{diag}/jobs/train.sbatch", stdout_pattern=f"{diag}/logs/%x_%j.out",
                  stderr_pattern=f"{diag}/logs/%x_%j.out", start=epoch("2026-09-07T06:31:41", UTC_M5),
                  scontrol_extra={"Priority": "1", "LastSchedEval": "2026-09-07T05:57:25",
                                  "AllocNode:Sid": "login02:99955", "SchedNodeList": "n-32-03"})
    c.jobs[job.job_id] = job
    out = render.scontrol_record(c, job, epoch("2026-09-07T05:58:07", UTC_M5), mode="iso") + "\n"
    assert out == sample("k2", "scontrol_show_job_d")


def test_scontrol_matches_vega_running_record_with_submitline() -> None:
    c = SimCluster(name="vega", slurm_version="25.11.4", login_host="login-v01", user="alice0001", uid=610001,
                   group="rs", gid=500, tz=UTC_M5, merged_stderr="empty")
    lmft = "/shared/pilab/alix/lmft"
    job = orb_job(723954, "lmft-x07-sampler-s1", "2026-09-05T22:06:27", account="pilab", partition="h200",
                  qos="normal", time_limit_s=2 * 86400, cpus=16, mem_mib=256 * 1024, gpus_per_node=2,
                  gpu_type="nvidia_h200_nvl", workdir=lmft, command=f"{lmft}/jobs/train.sbatch",
                  stdout_pattern=f"{lmft}/logs/%x_%j.out", stderr_pattern=None,
                  submit_args="--config configs/exp_b.yaml seed=1 lr=1e-5 steps=100",
                  scontrol_extra={"Priority": "60592", "SLUID": "sB6TW9XCKGHD00", "AllocNode:Sid": "login-v01:2336902",
                                  "ExcNodeList": "n-38-06"})
    running(job, "2026-09-07T00:04:12", ("n-38-01",))
    job.gpu_idx, job.cpu_ids = {"n-38-01": (0, 1)}, {"n-38-01": "24-39"}
    out = render.scontrol_record(c, job, epoch("2026-09-07T05:58:28", UTC_M5), mode="iso") + "\n"
    assert out == sample("vega", "scontrol_show_job_running_d")


def test_scontrol_matches_lyra_untyped_a30_record() -> None:
    c = SimCluster(name="lyra", slurm_version="26.05.3", login_host="login-a03", tz=UTC_M7, merged_stderr="empty")
    wd = "/home/victor/vision"
    job = SimJob(job_id=31367493, name="train_a01", user="victor", uid=2304117, account="victor", partition="free-gpu",
                 qos="low", submit=epoch("2026-09-07T03:23:39", UTC_M7), time_limit_s=12 * 3600, cpus=4,
                 mem_mib=16 * 1024, workdir=wd, command=f"{wd}/train.sbatch", stdout_pattern="logs/%x_%j.out",
                 gpus_per_node=1, gpu_type=None, typed_request=False, typed_alloc=False, alloc_gpu_type="A30",
                 scontrol_extra={"Priority": "1874", "SLUID": "sQ4HR2VVK0FH00", "Scheduler": "Backfill",
                                 "AllocNode:Sid": "login-a03:3318840", "GroupId": "victor(2304117)",
                                 "SubmitLine": "sbatch -J train_a01 train.sbatch train_a01 0.003 0.9 8"})
    running(job, "2026-09-07T03:27:22", ("lyra-gpu-q71-01",), tz=UTC_M7)
    job.gpu_idx, job.cpu_ids = {"lyra-gpu-q71-01": (0,)}, {"lyra-gpu-q71-01": "0-3"}
    out = render.scontrol_record(c, job, epoch("2026-09-07T03:58:29", UTC_M7), mode="iso") + "\n"
    assert out == sample("lyra", "scontrol_show_job_d")


def test_scontrol_loop_marks_forgotten_jobs() -> None:
    c = k2_cluster()
    job = orb_job(2330539, "old", "2026-09-06T11:14:06")
    job.state, job.start, job.end, job.nodes = "COMPLETED", 0.0, 1000.0, ("n-40-01",)
    c.jobs[job.job_id] = job
    out = render.scontrol_loop(c, 1000.0 + 301, ids_raw=["2330539", "123"])
    assert out.stdout == ("@@job 2330539\nslurm_load_jobs error: Invalid job id specified\n"
                          "@@job 123\nslurm_load_jobs error: Invalid job id specified\n")
    assert render.scontrol_loop(c, 1000.0 + 299, ids_raw=["2330539"]).stdout.count("JobState=COMPLETED") == 1


# ---------------------------------------------------------------------------------------------------- workstations
def lbs_gpus() -> list[SimGPU]:
    uuids = ("GPU-4e050371-ee67-0bb4-bab4-c813e9a744d2", "GPU-56d77d13-b29e-6b87-795d-377913f8ad57",
             "GPU-f38d7a35-ee30-7409-6ef0-7e2de081d943", "GPU-d5f3487b-34d5-9bc6-16d5-f3d609021850")
    readings = ((0, 1, 30, 23.95), (0, 1, 31, 27.88), (0, 1, 29, 19.93), (70, 21464, 78, 286.58))
    return [SimGPU(index=i, uuid=u, name="NVIDIA RTX A6000", mem_total_mib=49140, power_limit_w=300.0, util=util,
                   mem_used_mib=mem, temp_c=temp, power_w=pw) for i, (u, (util, mem, temp, pw)) in
            enumerate(zip(uuids, readings))]


def test_nvidia_query_matches_labstation_fixture() -> None:
    ws = SimWorkstation(name="labstation", gpus=lbs_gpus())
    out = render.nvidia_gpu(ws, fields=render.NVIDIA_GPU_FIELDS[:9])
    assert out.stdout == sample("labstation", "gpu_query")


def test_nvidia_query_full_field_list_and_unknown_field() -> None:
    ws = SimWorkstation(name="lynx", gpus=[SimGPU(index=0, uuid="GPU-x", name="NVIDIA RTX PRO 6000 Blackwell Max-Q "
                                                  "Workstation Edition", mem_total_mib=97887, power_limit_w=300.0,
                                                  fan=None)])
    row = render.nvidia_gpu(ws).stdout
    assert row.count(", ") == len(render.NVIDIA_GPU_FIELDS) - 1
    assert ", [N/A], P8, " in row
    bad = render.nvidia_gpu(ws, fields=["index", "bogus"])
    assert bad.rc == 2 and "bogus" in bad.stderr and bad.stdout == ""


def lynx_ws() -> SimWorkstation:
    ws = SimWorkstation(name="lynx", user="alice07", mem_total_kb=593_520_640)
    ws.gpus = [SimGPU(index=0, uuid="GPU-c6faebeb-cf00-3f1a-1182-49cec7f3d256", name="X", mem_total_mib=97887,
                      power_limit_w=300.0),
               SimGPU(index=1, uuid="GPU-35765204-ac46-81e4-b43b-71a7d77aedd9", name="X", mem_total_mib=97887,
                      power_limit_w=300.0)]
    for line in sample("lynx", "my_procs").splitlines()[1:]:
        pid, ppid, etime, cpu, mem, stat, args = line.split(None, 6)
        ws.procs[int(pid)] = SimProc(pid=int(pid), ppid=int(ppid), pgid=int(pid), user="alice07",
                                     start=1_000_000 - int(etime), args=args, stat=stat, cpu_pct=float(cpu),
                                     mem_pct=float(mem))
    ws.procs[786345].gpu, ws.procs[786345].gpu_mem_mib = 0, 15234
    ws.procs[642234].gpu, ws.procs[642234].gpu_mem_mib = 1, 15234
    return ws


def test_compute_apps_matches_lynx_fixture() -> None:
    assert render.compute_apps(lynx_ws()) == sample("lynx", "compute_apps")


def test_ps_rows_match_procps_layout() -> None:
    ws = lynx_ws()
    cols = ("pid", "ppid", "etimes", "pcpu", "pmem", "stat", "args")
    expected = sample("lynx", "my_procs").splitlines()[1:]
    assert [render.ps_row(ws.procs[int(line.split()[0])], 1_000_000, cols, ws.mem_total_kb) for line in expected] \
        == expected
    apps_cols = ("pid", "ppid", "user", "etimes", "pcpu", "pmem", "args")
    ws.procs[642234].start, ws.procs[786345].start = 1_000_000 - 67512, 1_000_000 - 3103
    expected = sample("lynx", "gpu_pids_ps").splitlines()[1:]
    assert [render.ps_row(ws.procs[int(line.split()[0])], 1_000_000, apps_cols, ws.mem_total_kb)
            for line in expected] == expected


def test_nvidia_apps_payload_shape() -> None:
    ws = lynx_ws()
    out = render.nvidia_apps(ws, 1_000_000).stdout
    apps, ps = out.split("@@ps\n")
    assert apps == sample("lynx", "compute_apps")
    assert [line.split()[0] for line in ps.splitlines()] == ["642234", "786345"]
    ws.procs[786345].gpu = ws.procs[642234].gpu = None
    assert render.nvidia_apps(ws, 1_000_000).stdout == "\n@@ps\n"   # printf '%s\n' "" then the marker


def test_nvidia_version_matches_fixture() -> None:
    assert render.nvidia_version(SimWorkstation(name="lynx")).stdout == sample("lynx", "nvidia_smi_version")


def test_procfs_host_contains_real_meminfo_and_cpu_line() -> None:
    ws = SimWorkstation(name="labstation", ncpu=64, mem_total_kb=528008252, swap_total_kb=2097148,
                        swap_free_kb=2061820, mem_used_kb=528008252 - 510878396)
    ws.cpu_jiffies = [2005215382, 1007428288, 384747761, 68893959522, 18544027, 0, 6257322, 0, 0, 0]
    out = render.procfs_host(ws, 2_000_000.0).stdout
    assert sample("labstation", "meminfo") in out
    assert sample("labstation", "proc_stat").splitlines()[0] + "\n" in out
    assert out.startswith("ncpu=64\nepoch=2000000\nload=")


def test_slurm_capability_outputs_match_fixtures() -> None:
    c = k2_cluster()
    c.slurm_version = "25.05.8"
    assert render.sinfo_version(c).stdout == sample("k2", "sinfo_version")
    c.config = {line.split(" = ")[0].strip(): line.split(" = ", 1)[1]
                for line in sample("k2", "slurm_config").splitlines()}
    assert render.slurm_config(c).stdout == sample("k2", "slurm_config")
    c.associations = (("k2", "campus01", "", "gpu-priority,normal,labpi"),)
    assert render.sacctmgr_assoc(c).stdout == "k2|campus01||gpu-priority,normal,labpi\n"



def test_charging_rule_outputs_match_fixtures() -> None:
    """Account (§5.2.11): `sacctmgr show qos` byte-exact; partition lines carry AllowQos, QoS and
    TRESBillingWeights in scontrol's order (TRESBillingWeights after TRES, only where set)."""
    for host in ("k2", "vega", "lyra"):
        c = k2_cluster()
        c.qos = tuple((n, float(f)) for n, f in (ln.split("|") for ln in sample(host, "sacctmgr_qos").splitlines()))
        assert render.sacctmgr_qos(c).stdout == sample(host, "sacctmgr_qos")
    c = k2_cluster()
    c.nodes = [SimNode("x1", ("gpu", "free-gpu"), 32, 1000, (GresDef("A30", 4),))]
    c.partition_fields = {"gpu": {"AllowQos": "normal,high", "QoS": "gpu-part", "TRESBillingWeights": "cpu=1"}}
    lines = render.scontrol_partitions(c).stdout.splitlines()
    assert lines[2].startswith("PartitionName=gpu AllowGroups=ALL AllowAccounts=ALL AllowQos=normal,high QoS=gpu-part ")
    assert lines[2].endswith(",gres/gpu=4 TRESBillingWeights=cpu=1")
    assert " AllowQos=ALL QoS=N/A " in lines[3] and "TRESBillingWeights" not in lines[3]


def test_formatting_helpers() -> None:
    assert render.squeue_time(0) == "0:00" and render.squeue_time(4 * 3600 + 420) == "4:07:00"
    assert render.squeue_time(3 * 86400) == "3-00:00:00" and render.squeue_time(None) == "UNLIMITED"
    assert render.acct_time(3600) == "01:00:00" and render.acct_time(86400 + 1) == "1-00:00:01"
    assert render.mem_str(48 * 1024) == "48G" and render.mem_str(500) == "500M"
    assert render.hostlist(["n-37-01", "n-37-02", "n-37-04", "x"]) == "n-37-[01-02,04],x"
    assert render.hostlist(["lyra-gpu-s71-03"]) == "lyra-gpu-s71-03"
    assert render.tz_text(UTC_M5) == "-0500" and render.tz_text(timedelta(hours=5, minutes=30)) == "+0530"
    assert render.slurm_ts(1788696459, UTC_M5, "epoch", "N/A") == "1788696459"
    assert render.slurm_ts(None, UTC_M5, "epoch", "N/A") == "N/A"
