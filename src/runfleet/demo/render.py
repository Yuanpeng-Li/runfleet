"""[WP4] Demo world -> byte-exact command outputs (the formats captured in tests/fixtures/samples).

Every public function plays one remote command and returns its exact stdout (trailing newline included, as the
framing delivers it), rc and stderr. Slurm timestamps are epoch seconds by default, because the probe prelude exports
`SLURM_TIME_FORMAT=%s` (§4.3); `mode="iso"` renders the naive local ISO form the field-probe fixtures were captured in,
which is what the byte-exact fixture tests use.

The field lists below repeat DESIGN.md §5.2.2 on purpose: the demo plays the remote side and must not import WP2's
command builders (a demo that reused them could not catch a drift between collector and real format).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from runfleet.demo.world import (MIN_JOB_AGE_S, SimCluster, SimGPU, SimHost, SimJob, SimNode, SimProc,
                                 SimWorkstation, StepUsage, compress_ranges, expand_pattern, free_gpus, gpu_type_at,
                                 node_cpus_used, stable_hash)

TimeMode = Literal["epoch", "iso"]

SQUEUE_FIELDS = ["JobID", "ArrayJobID", "ArrayTaskID", "Name", "UserName", "State", "StateCompact", "Reason",
                 "TimeUsed", "TimeLimit", "TimeLeft", "NumNodes", "NumCPUs", "MinMemory", "Partition", "QOS",
                 "Account", "SubmitTime", "StartTime", "EndTime", "NodeList", "BatchHost", "tres-alloc",
                 "tres-per-node", "tres-per-job", "RestartCnt", "STDOUT", "STDERR", "WorkDir", "Command"]
SQUEUE_LEGACY = "%A|%F|%K|%j|%u|%T|%t|%r|%M|%l|%L|%D|%C|%m|%P|%q|%a|%V|%S|%e|%N|%B|%b|%Z|%o"
_LEGACY_CODES = {"A": "JobID", "F": "ArrayJobID", "K": "ArrayTaskID", "j": "Name", "u": "UserName", "T": "State",
                 "t": "StateCompact", "r": "Reason", "M": "TimeUsed", "l": "TimeLimit", "L": "TimeLeft",
                 "D": "NumNodes", "C": "NumCPUs", "m": "MinMemory", "P": "Partition", "q": "QOS", "a": "Account",
                 "V": "SubmitTime", "S": "StartTime", "e": "EndTime", "N": "NodeList", "B": "BatchHost",
                 "b": "tres-per-node", "Z": "WorkDir", "o": "Command", "i": "JobIDDisplay", "R": "NodeOrReason"}
SACCT_FIELDS = ["JobID", "JobIDRaw", "JobName", "User", "Account", "QOS", "Partition", "State", "ExitCode",
                "DerivedExitCode", "Reason", "Submit", "Start", "End", "Elapsed", "Timelimit", "AllocCPUS", "ReqMem",
                "ReqTRES", "AllocTRES", "NNodes", "NodeList", "StdOut", "StdErr", "WorkDir"]
STEP_FIELDS = ["JobID", "JobName", "State", "ExitCode", "DerivedExitCode", "Reason", "Elapsed", "Start", "End",
               "NodeList", "NTasks", "MaxRSS", "AveCPU", "TRESUsageInMax", "TRESUsageInTot"]
SSTAT_FIELDS = ["JobID", "NTasks", "AveCPU", "AveRSS", "MaxRSS", "TRESUsageInAve", "TRESUsageInMax",
                "TRESUsageInTot"]
OCCUPANT_FIELDS = ["JobID", "ArrayJobID", "ArrayTaskID", "UserName", "Account", "Partition", "StateCompact",
                   "TimeUsed", "TimeLimit", "NumNodes", "NodeList", "tres-alloc", "Name"]
SINFO_FIELDS = ["NodeList", "Partition", "StateLong", "CPUsState", "Memory", "AllocMem", "FreeMem", "Gres", "GresUsed",
                "Reason"]
NVIDIA_GPU_FIELDS = ["index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu",
                     "power.draw", "power.limit", "fan.speed", "pstate", "utilization.memory"]
DETECT_TOOLS = ["squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel", "sacctmgr", "nvidia-smi", "timeout",
                "less", "stat", "ps", "pgrep", "srun", "ssh"]
ENV_ALLOW = ("CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "OMP_NUM_THREADS", "WANDB_PROJECT",
             "WANDB_RUN_ID")

_COMPACT = {"PENDING": "PD", "RUNNING": "R", "COMPLETING": "CG", "COMPLETED": "CD", "CONFIGURING": "CF",
            "CANCELLED": "CA", "FAILED": "F", "TIMEOUT": "TO", "PREEMPTED": "PR", "NODE_FAIL": "NF",
            "OUT_OF_MEMORY": "OOM", "SUSPENDED": "S", "REQUEUED": "RQ", "BOOT_FAIL": "BF", "DEADLINE": "DL",
            "STOPPED": "ST"}
_EXTERN_AVECPU = "213503982334-14:25:51"   # what the .extern step really reports (field probes)


@dataclass(frozen=True, slots=True)
class Out:
    """One command's result as the remote side produced it."""

    stdout: str = ""
    rc: int = 0
    stderr: str = ""


# ====================================================================================================== formatting
def squeue_time(seconds: float | None) -> str:
    """squeue TimeUsed/TimeLimit/TimeLeft: `D-HH:MM:SS`, `H:MM:SS` or `M:SS`."""
    if seconds is None:
        return "UNLIMITED"
    s = max(0, int(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def acct_time(seconds: float | None) -> str:
    """sacct/scontrol/sstat durations: `D-HH:MM:SS` or `HH:MM:SS`."""
    if seconds is None:
        return "UNLIMITED"
    s = max(0, int(seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def mem_str(mib: int) -> str:
    return f"{mib // 1024}G" if mib % 1024 == 0 else f"{mib}M"


def slurm_ts(epoch: float | None, tz: timedelta, mode: TimeMode, missing: str) -> str:
    if epoch is None:
        return missing
    if mode == "epoch":
        return str(int(epoch))
    return datetime.fromtimestamp(int(epoch), timezone(tz)).strftime("%Y-%m-%dT%H:%M:%S")


def tz_text(tz: timedelta) -> str:
    """`date +%z`."""
    total = int(tz.total_seconds()) // 60
    sign = "-" if total < 0 else "+"
    h, m = divmod(abs(total), 60)
    return f"{sign}{h:02d}{m:02d}"


def hostlist(names: Sequence[str]) -> str:
    """Compress hostnames the way Slurm prints them: `n-37-[01-02]`, `n-40-01`, `a,b`."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for name in names:
        head = name.rstrip("0123456789")
        digits = name[len(head):]
        if not digits:
            groups.setdefault(name, [])
            order.append(name)
            continue
        if head not in groups:
            order.append(head)
            groups[head] = []
        groups[head].append(digits)
    out: list[str] = []
    for head in dict.fromkeys(order):
        nums = groups.get(head, [])
        if not nums:
            out.append(head)
        elif len(nums) == 1:
            out.append(head + nums[0])
        else:
            width = len(nums[0])
            spans = compress_ranges(int(n) for n in nums).split(",")
            padded = ["-".join(f"{int(x):0{width}d}" for x in sp.split("-")) for sp in spans]
            out.append(f"{head}[{','.join(padded)}]")
    return ",".join(out)


def _tres(job: SimJob, *, typed: bool, alpha: bool) -> str:
    """TRES string: squeue/scontrol order (`cpu,mem,node,billing,gres/gpu,gres/gpu:t`) or sacct's alphabetical."""
    mem = mem_str(job.mem_mib * job.num_nodes)
    gtype = (job.gpu_type or "").lower()
    if alpha:
        parts = [f"billing={job.cpus}", f"cpu={job.cpus}"]
        if job.gpus and typed and gtype:
            parts.append(f"gres/gpu:{gtype}={job.gpus}")
        if job.gpus:
            parts.append(f"gres/gpu={job.gpus}")
        parts += [f"mem={mem}", f"node={job.num_nodes}"]
        return ",".join(parts)
    parts = [f"cpu={job.cpus}", f"mem={mem}", f"node={job.num_nodes}", f"billing={job.cpus}"]
    if job.gpus:
        parts.append(f"gres/gpu={job.gpus}")
        if typed and gtype:
            parts.append(f"gres/gpu:{gtype}={job.gpus}")
    return ",".join(parts)


def _tres_per_node(job: SimJob) -> str:
    if not job.gpus_per_node:
        return "N/A"
    if job.typed_request and job.gpu_type:
        return f"gres/gpu:{job.gpu_type.lower()}:{job.gpus_per_node}"
    return f"gres/gpu:{job.gpus_per_node}"


def _limit_end(job: SimJob) -> float | None:
    if job.start is None or job.time_limit_s is None:
        return None
    return job.start + job.time_limit_s


def stdout_path(c: SimCluster, job: SimJob) -> str:
    return expand_pattern(job.stdout_pattern, job, c.user)


def stderr_path(c: SimCluster, job: SimJob) -> str:
    return expand_pattern(job.stderr_pattern, job, c.user) if job.stderr_pattern else stdout_path(c, job)


# ====================================================================================================== squeue
def _squeue_value(c: SimCluster, job: SimJob, fieldname: str, now: float, mode: TimeMode) -> str:
    running = job.state in ("RUNNING", "COMPLETING", "SUSPENDED")
    used = job.elapsed(now) if running else 0.0
    match fieldname:
        case "JobID":
            return str(job.job_id)
        case "ArrayJobID":
            return str(job.array_job_id if job.array_job_id is not None else job.job_id)
        case "ArrayTaskID":
            return job.array_task or "N/A"
        case "JobIDDisplay":
            return job.display_id
        case "Name":
            return job.name
        case "UserName":
            return job.user
        case "State":
            return job.state
        case "StateCompact":
            return _COMPACT.get(job.state, job.state)
        case "Reason":
            return "None" if running else job.reason
        case "NodeOrReason":
            return hostlist(job.nodes) if running else f"({job.reason})"
        case "TimeUsed":
            return squeue_time(used)
        case "TimeLimit":
            return squeue_time(job.time_limit_s)
        case "TimeLeft":
            return "UNLIMITED" if job.time_limit_s is None else squeue_time(job.time_limit_s - used)
        case "NumNodes":
            return str(job.num_nodes)
        case "NumCPUs":
            return str(job.cpus)
        case "MinMemory":
            return mem_str(job.mem_mib)
        case "Partition":
            return job.partition
        case "QOS":
            return job.qos
        case "Account":
            return job.account
        case "SubmitTime":
            return slurm_ts(job.submit, c.tz, mode, "N/A")
        case "StartTime":
            return slurm_ts(job.start, c.tz, mode, "N/A")
        case "EndTime":
            return slurm_ts(_limit_end(job), c.tz, mode, "N/A")
        case "NodeList":
            return hostlist(job.nodes) if running else ""
        case "BatchHost":
            return job.nodes[0] if running and job.nodes else "n/a"
        case "tres-alloc":
            return _tres(job, typed=job.typed_alloc if running else job.typed_request, alpha=False)
        case "tres-per-node":
            return _tres_per_node(job)
        case "tres-per-job":
            return "N/A"
        case "RestartCnt":
            return str(job.restarts)
        case "STDOUT":
            return job.stdout_pattern
        case "STDERR":
            return job.stderr_pattern if job.stderr_pattern is not None else "(null)"
        case "WorkDir":
            return job.workdir
        case "Command":
            return job.command
    return "N/A"


def _squeue_jobs(c: SimCluster, users: Sequence[str]) -> list[SimJob]:
    who = set(users) if users else {c.user}
    live = [j for j in c.jobs.values() if not j.terminal and j.user in who]
    # PENDING first (default squeue sort is by state, then priority), then by id
    return sorted(live, key=lambda j: (j.state != "PENDING", j.job_id))


def squeue(c: SimCluster, now: float, *, users: Sequence[str] = (), sep: str = "\x1f",
           fields: Sequence[str] = SQUEUE_FIELDS, mode: TimeMode = "epoch") -> Out:
    """`squeue {--me|-u U} -h -O 'F1:<sep>,F2:<sep>,…'`: every value followed by the separator."""
    rows = ["".join(_squeue_value(c, j, f, now, mode) + sep for f in fields) for j in _squeue_jobs(c, users)]
    return Out("".join(r + "\n" for r in rows))


def squeue_legacy(c: SimCluster, now: float, *, users: Sequence[str] = (), spec: str = SQUEUE_LEGACY,
                  mode: TimeMode = "epoch") -> Out:
    """`squeue -h -o '<spec>'` with `%x` codes and literal separators (no trailing separator)."""
    out: list[str] = []
    for job in _squeue_jobs(c, users):
        line: list[str] = []
        i = 0
        while i < len(spec):
            if spec[i] == "%" and i + 1 < len(spec):
                line.append(_squeue_value(c, job, _LEGACY_CODES.get(spec[i + 1], "?"), now, mode))
                i += 2
            else:
                line.append(spec[i])
                i += 1
        out.append("".join(line) + "\n")
    return Out("".join(out))


# ====================================================================================================== sacct
def _sacct_state(job: SimJob) -> str:
    if job.state == "CANCELLED" and job.cancelled_by is not None:
        return f"CANCELLED by {job.cancelled_by}"
    return "RUNNING" if job.state == "COMPLETING" else job.state


def _sacct_value(c: SimCluster, job: SimJob, fieldname: str, now: float, mode: TimeMode) -> str:
    started = job.started
    match fieldname:
        case "JobID":
            return job.display_id
        case "JobIDRaw":
            return str(job.job_id)
        case "JobName":
            return job.name
        case "User":
            return job.user
        case "Account":
            return job.account
        case "QOS":
            return job.qos
        case "Partition":
            return job.partition
        case "State":
            return _sacct_state(job)
        case "ExitCode":
            return f"{job.exit_code}:{job.exit_signal}" if job.terminal else "0:0"
        case "DerivedExitCode":
            return f"{job.exit_code}:0" if job.state == "FAILED" else "0:0"
        case "Reason":
            return job.reason if job.state == "PENDING" else "None"
        case "Submit":
            return slurm_ts(job.submit, c.tz, mode, "Unknown")
        case "Start":
            if job.state == "PENDING":
                return "Unknown"
            return slurm_ts(job.start if started else None, c.tz, mode, "None")
        case "End":
            return slurm_ts(job.end if job.terminal else None, c.tz, mode, "Unknown")
        case "Elapsed":
            return acct_time(job.elapsed(now))
        case "Timelimit":
            return acct_time(job.time_limit_s)
        case "AllocCPUS":
            return str(job.cpus if started else 0)
        case "ReqMem":
            return mem_str(job.mem_mib)
        case "ReqTRES":
            typed = job.typed_request if job.typed_req_tres is None else job.typed_req_tres
            return _tres(job, typed=typed, alpha=True)
        case "AllocTRES":
            return _tres(job, typed=job.typed_alloc, alpha=True) if started else ""
        case "NNodes":
            return str(job.num_nodes)
        case "NodeList":
            return hostlist(job.nodes) if started and job.nodes else "None assigned"
        case "WorkDir":
            return job.workdir
        case "StdOut":
            return stdout_path(c, job)
        case "StdErr":
            return stderr_path(c, job)
    return ""


def _sort_key(job: SimJob) -> tuple[int, int, int]:
    arr = job.array_job_id if job.array_job_id is not None else job.job_id
    task = int(job.array_task) if job.array_task and job.array_task.isdigit() else (1 << 30 if job.summary else 0)
    return (arr, task, job.job_id)


def _sacct_rows(c: SimCluster, jobs: Iterable[SimJob], fields: Sequence[str], sep: str, now: float,
                mode: TimeMode) -> str:
    return "".join(sep.join(_sacct_value(c, j, f, now, mode) for f in fields) + "\n" for j in jobs)


def sacct_window(c: SimCluster, now: float, *, since_s: int, fields: Sequence[str] = SACCT_FIELDS,
                 sep: str = "\x1f", mode: TimeMode = "epoch") -> Out:
    """`sacct -u $(id -un) -S now-<n>seconds -E now -X -P --delimiter=<sep> -n --format=…`."""
    start = now - since_s
    jobs = [j for j in c.jobs.values()
            if j.user == c.user and j.submit <= now and (not j.terminal or (j.end or 0) >= start)]
    return Out(_sacct_rows(c, sorted(jobs, key=_sort_key), fields, sep, now, mode))


def sacct_ids(c: SimCluster, now: float, *, ids_raw: Sequence[str], fields: Sequence[str] = SACCT_FIELDS,
              sep: str = "\x1f", mode: TimeMode = "epoch") -> Out:
    """`sacct -j <ids> -X -P …`: a raw id selects that job, an array job id every element of the array."""
    seen: list[SimJob] = []
    for raw in ids_raw:
        if not raw.isdigit():
            continue
        num = int(raw)
        for j in sorted(c.jobs.values(), key=_sort_key):
            if (j.job_id == num or j.array_job_id == num) and j not in seen:
                seen.append(j)
    return Out(_sacct_rows(c, seen, fields, sep, now, mode))


def _usage_tres(c: SimCluster, job: SimJob, u: StepUsage, which: Literal["ave", "max", "tot"]) -> str:
    n = u.ntasks if which == "tot" else 1
    cpu = u.ave_cpu_s * n
    rss = (u.max_rss_kb if which == "max" else u.ave_rss_kb) * n
    parts = [f"cpu={acct_time(cpu)}", "energy=0", f"fs/disk={u.disk_bytes * n}"]
    if c.gpu_tres and job.gpus:
        gmem = (u.gpumem_max_mib if which == "max" else u.gpumem_ave_mib) * n
        gutil = (u.gpuutil_max if which == "max" else u.gpuutil_ave) * n
        parts += [f"gres/gpumem={int(gmem)}M", f"gres/gpuutil={int(round(gutil))}"]
    parts += [f"mem={rss}K", f"pages={u.pages * n}", "vmem=0"]
    return ",".join(parts)


def _step_rows(c: SimCluster, job: SimJob, now: float, mode: TimeMode) -> list[dict[str, str]]:
    """The allocation row and its steps (`.batch`, `.extern`, `.0`) as field -> value maps."""
    def times(row: dict[str, str]) -> None:
        row["Elapsed"] = acct_time(job.elapsed(now))
        row["Start"] = slurm_ts(job.start if job.started else None, c.tz, mode, "None")
        row["End"] = slurm_ts(job.end if job.terminal else None, c.tz, mode, "Unknown")

    alloc = {"JobID": job.display_id, "JobName": job.name, "State": _sacct_state(job),
             "ExitCode": f"{job.exit_code}:{job.exit_signal}" if job.terminal else "0:0",
             "DerivedExitCode": f"{job.exit_code}:0" if job.state == "FAILED" else "0:0",
             "Reason": job.reason if job.state == "PENDING" else "None",
             "NodeList": hostlist(job.nodes) if job.started and job.nodes else "None assigned"}
    times(alloc)
    rows = [alloc]
    if not job.started:
        return rows
    ended = job.terminal
    batch_state, batch_exit = {
        "COMPLETED": ("COMPLETED", "0:0"), "FAILED": ("FAILED", f"{job.exit_code}:0"),
        "PREEMPTED": ("FAILED", "143:0") if c.preempt_mode == "CANCEL" else ("CANCELLED", "0:15"),
        "TIMEOUT": ("CANCELLED", "0:15"), "CANCELLED": ("CANCELLED", "0:15"),
        "OUT_OF_MEMORY": ("OUT_OF_MEMORY", "0:125"),
    }.get(job.state, ("RUNNING", "0:0"))
    batch = {"JobID": f"{job.display_id}.batch", "JobName": "batch", "State": batch_state, "ExitCode": batch_exit,
             "NodeList": job.nodes[0], "NTasks": "1", "MaxRSS": f"{job.batch.max_rss_kb}K",
             "AveCPU": acct_time(job.batch.ave_cpu_s)}
    if ended:
        batch["TRESUsageInMax"] = _usage_tres(c, job, job.batch, "max")
        batch["TRESUsageInTot"] = _usage_tres(c, job, job.batch, "tot")
        batch["TotalCPU"] = acct_time(job.batch.ave_cpu_s * job.batch.ntasks)
    times(batch)
    extern = {"JobID": f"{job.display_id}.extern", "JobName": "extern",
              "State": ("CANCELLED" if job.state == "CANCELLED" else "COMPLETED") if ended else "RUNNING",
              "ExitCode": "0:0", "NodeList": hostlist(job.nodes), "NTasks": str(job.num_nodes)}
    if ended:
        extern["TRESUsageInMax"] = extern["TRESUsageInTot"] = "energy=0"
    times(extern)
    rows += [batch, extern]
    if job.step0 is not None:
        step = {"JobID": f"{job.display_id}.0", "JobName": "torchrun",
                "State": batch_state if ended else "RUNNING", "ExitCode": batch_exit if ended else "0:0",
                "NodeList": hostlist(job.nodes), "NTasks": str(job.step0.ntasks),
                "MaxRSS": f"{job.step0.max_rss_kb}K", "AveCPU": acct_time(job.step0.ave_cpu_s)}
        if ended:
            step["TRESUsageInMax"] = _usage_tres(c, job, job.step0, "max")
            step["TRESUsageInTot"] = _usage_tres(c, job, job.step0, "tot")
            step["TotalCPU"] = acct_time(job.step0.ave_cpu_s * job.step0.ntasks)
        times(step)
        rows.append(step)
    if ended:                        # sacct sums TotalCPU over the steps on the allocation row (efficiency card)
        alloc["TotalCPU"] = acct_time(sum(u.ave_cpu_s * u.ntasks for u in (job.batch, job.step0) if u is not None))
    return rows


def sacct_steps(c: SimCluster, now: float, *, job_id_raw: str, sep: str = "\x1f",
                fields: Sequence[str] = STEP_FIELDS, mode: TimeMode = "epoch") -> Out:
    """`sacct -j <raw> -P --delimiter=<sep> -n --format=<STEP_FIELDS>` (no -X: allocation + steps)."""
    if not job_id_raw.isdigit() or int(job_id_raw) not in c.jobs:
        return Out("")
    job = c.jobs[int(job_id_raw)]
    rows = _step_rows(c, job, now, mode)
    return Out("".join(sep.join(r.get(f, "") for f in fields) + "\n" for r in rows))


# ====================================================================================================== scontrol
def _sluid(job_id: int) -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    h = stable_hash(f"sluid:{job_id}") | (job_id << 32)
    chars = []
    for _ in range(11):
        chars.append(alphabet[h % 32])
        h //= 32
    return "s" + "".join(chars) + "00"


def _in_controller(job: SimJob, now: float) -> bool:
    return not job.terminal or (job.end is not None and now - job.end <= MIN_JOB_AGE_S)


def scontrol_record(c: SimCluster, job: SimJob, now: float, mode: TimeMode = "epoch") -> str:
    """One `scontrol show job -d -o` line (version-specific keys and spacing quirks, see the fixtures)."""
    v = tuple(int(x) for x in c.slurm_version.split(".")[:2])
    started = job.started
    extra = job.scontrol_extra
    kv: list[tuple[str, str]] = [("JobId", str(job.job_id))]
    if job.array_job_id is not None:
        kv.append(("ArrayJobId", str(job.array_job_id)))
        kv.append(("ArrayTaskId", job.array_task or ""))
        if job.summary:
            kv.append(("ArrayTaskThrottle", str(job.array_throttle)))
    kv.append(("JobName", job.name))
    if v >= (25, 11):
        kv.append(("SLUID", extra.get("SLUID", _sluid(job.job_id))))
    priority = extra.get("Priority", "0" if job.held else str(1 + stable_hash(job.name) % 90000))

    def ts(epoch: float | None) -> str:
        return slurm_ts(epoch, c.tz, mode, "Unknown")

    reason = "None" if started else job.reason
    kv += [("UserId", f"{job.user}({job.uid})"), ("GroupId", extra.get("GroupId", f"{c.group}({c.gid})")),
           ("MCS_label", "N/A"),
           ("Priority", priority), ("Nice", "0"), ("Account", job.account), ("QOS", job.qos),
           ("JobState", job.state), ("Reason", reason), ("Dependency", job.dependency), ("Requeue", "1"),
           ("Restarts", str(job.restarts)), ("BatchFlag", "1"), ("Reboot", "0"),
           ("ExitCode", f"{job.exit_code}:{job.exit_signal}" if job.terminal else "0:0"),
           ("DerivedExitCode", f"{job.exit_code}:0" if job.state == "FAILED" else "0:0"),
           ("RunTime", acct_time(job.elapsed(now))), ("TimeLimit", acct_time(job.time_limit_s)),
           ("TimeMin", "N/A"), ("SubmitTime", ts(job.submit)), ("EligibleTime", ts(job.submit)),
           ("AccrueTime", ts(job.submit)), ("StartTime", ts(job.start)),
           ("EndTime", ts(job.end if job.terminal else _limit_end(job))), ("Deadline", "N/A")]
    if started:
        kv += [("PreemptEligibleTime", ts(job.start)), ("PreemptTime", "None")]
    kv += [("SuspendTime", "None"), ("SecsPreSuspend", "0"),
           ("LastSchedEval", extra.get("LastSchedEval", ts(job.start if started else job.submit + 60))),
           ("Scheduler", extra.get("Scheduler", "Main")), ("Partition", job.partition),
           ("AllocNode:Sid", extra.get("AllocNode:Sid", f"{c.login_host}:{1000 + job.job_id % 4_000_000}")),
           ("ReqNodeList", "(null)"), ("ExcNodeList", extra.get("ExcNodeList", "(null)")),
           ("NodeList", hostlist(job.nodes) if started else "")]
    if not started and "SchedNodeList" in extra:
        kv.append(("SchedNodeList", extra["SchedNodeList"]))
    if started:
        kv.append(("BatchHost", job.nodes[0] if job.nodes else ""))
    ntasks = job.step_tasks or 1
    kv += [("NumNodes", str(job.num_nodes)), ("NumCPUs", str(job.cpus)), ("NumTasks", str(ntasks)),
           ("CPUs/Task", str(job.cpus // ntasks)), ("ReqB:S:C:T", "0:0:*:*"),
           ("ReqTRES", _tres(job, typed=job.typed_request, alpha=False)),
           ("AllocTRES", _tres(job, typed=job.typed_alloc, alpha=False) if started else "(null)"),
           ("Socks/Node", "*"), ("NtasksPerN:B:S:C", "0:0:*:1" if v >= (26, 5) else "0:0:*:*"),
           ("CoreSpec", "*")]
    per_node: list[tuple[str, str]] = []
    if started:
        gtype = job.alloc_gpu_type or job.gpu_type
        if job.gpus:
            kv.append(("JOB_GRES", f"gpu:{gtype}:{job.gpus}" if gtype else f"gpu:{job.gpus}"))
        for node in job.nodes:
            idx = job.gpu_idx.get(node, ())
            gres = f"gpu:{gtype}:{len(idx)}(IDX:{compress_ranges(idx)})" if idx else ""
            per_node += [("Nodes", node), ("CPU_IDs", job.cpu_ids.get(node, "0")), ("Mem", str(job.mem_mib)),
                         ("GRES", gres)]
    tail: list[tuple[str, str]] = [
        ("MinCPUsNode", str(job.cpus // job.num_nodes)), ("MinMemoryNode", mem_str(job.mem_mib)),
        ("MinTmpDiskNode", "0"), ("Features", "(null)"), ("DelayBoot", "00:00:00")]
    tail += [("OverSubscribe", "NO"), ("Exclusive", "NO")] if v >= (26, 5) else [("OverSubscribe", "OK")]
    tail += [("Contiguous", "0"), ("Licenses", "(null)"), ("LicensesAlloc", "(null)"), ("Network", "(null)"),
             ("Command", job.command)]
    if v >= (25, 11):
        tail.append(("SubmitLine", extra.get("SubmitLine", _submit_line(c, job))))
    out_path = stdout_path(c, job)
    err_path = stderr_path(c, job)
    if job.stderr_pattern is None or err_path == out_path:
        err_value = out_path if c.merged_stderr == "repeat" else ""
    else:
        err_value = err_path
    tail += [("WorkDir", job.workdir), ("StdErr", err_value), ("StdIn", "/dev/null"), ("StdOut", out_path)]
    if job.gpus_per_node:
        tail.append(("TresPerNode", _tres_per_node(job)))
    tail.append(("TresPerTask", f"cpu={job.cpus // ntasks}"))
    head = " ".join(f"{k}={val}" for k, val in kv)
    node_txt = " ".join(f"{k}={val}" for k, val in per_node)
    tail_txt = " ".join(f"{k}={val}" for k, val in tail)
    if v < (25, 11):                     # 25.05 -o keeps the multi-line indent: "JOB_GRES=…   Nodes=…", trailing " "
        mid = f"   {node_txt}" if node_txt else ""
        return f"{head}{mid} {tail_txt} "
    return " ".join(p for p in (head, node_txt, tail_txt) if p)


def _submit_line(c: SimCluster, job: SimJob) -> str:
    if c.name == "lyra" or not job.gpu_type:
        return f"sbatch -J {job.name} {job.command} {job.submit_args}".rstrip()
    parts = [f"--job-name={job.name}", f"--chdir={job.workdir}", f"--time={acct_time(job.time_limit_s)}",
             f"--account={job.account}"]
    if "ExcNodeList" in job.scontrol_extra:
        parts.append(f"--exclude={job.scontrol_extra['ExcNodeList']}")
    parts.append(f"--partition={job.partition}")
    if job.gpus_per_node:
        parts.append(f"--gres=gpu:{job.gpu_type}:{job.gpus_per_node}")
    script = job.command.removeprefix(f"{job.workdir.rstrip('/')}/") if job.workdir else job.command
    parts += [f"--cpus-per-task={job.cpus // max(1, job.step_tasks or 1)}", f"--mem={mem_str(job.mem_mib)}",
              "--requeue", script]
    return " ".join(["sbatch", *parts, job.submit_args]).rstrip()


def scontrol_loop(c: SimCluster, now: float, *, ids_raw: Sequence[str], mode: TimeMode = "epoch") -> Out:
    """The §5.2.2 loop: `echo "@@job $j"; scontrol show job -d -o "$j" 2>&1` per id."""
    out: list[str] = []
    for raw in ids_raw:
        out.append(f"@@job {raw}\n")
        num = int(raw) if raw.isdigit() else -1
        records = [j for j in sorted(c.jobs.values(), key=_sort_key)
                   if (j.job_id == num or j.array_job_id == num) and _in_controller(j, now)]
        if not records:
            out.append("slurm_load_jobs error: Invalid job id specified\n")
        out += [scontrol_record(c, j, now, mode) + "\n" for j in records]
    return Out("".join(out))


# ====================================================================================================== sstat / sinfo
def sstat(c: SimCluster, now: float, *, ids_raw: Sequence[str], fields: Sequence[str] = SSTAT_FIELDS) -> Out:
    """`sstat -a -j <ids> -P -n --format=…`; `.extern` reports garbage AveCPU (field probes)."""
    out: list[str] = []
    errors: list[str] = []
    for raw in ids_raw:
        job = c.jobs.get(int(raw)) if raw.isdigit() else None
        if job is None or job.state not in ("RUNNING", "COMPLETING"):
            errors.append(f"sstat: error: couldn't get steps for job {raw}")
            continue
        extern = {"JobID": f"{job.job_id}.extern", "NTasks": str(job.num_nodes), "AveCPU": _EXTERN_AVECPU,
                  "AveRSS": "", "MaxRSS": "", "TRESUsageInAve": "energy=0", "TRESUsageInMax": "energy=0",
                  "TRESUsageInTot": "energy=0"}
        rows = [extern]
        for suffix, u in (("batch", job.batch), ("0", job.step0)):
            if u is None:
                continue
            rows.append({"JobID": f"{job.job_id}.{suffix}", "NTasks": str(u.ntasks), "AveCPU": acct_time(u.ave_cpu_s),
                         "AveRSS": f"{u.ave_rss_kb}K", "MaxRSS": f"{u.max_rss_kb}K",
                         "TRESUsageInAve": _usage_tres(c, job, u, "ave"),
                         "TRESUsageInMax": _usage_tres(c, job, u, "max"),
                         "TRESUsageInTot": _usage_tres(c, job, u, "tot")})
        out += ["|".join(r.get(f, "") for f in fields) + "\n" for r in rows]
    return Out("".join(out), 0 if out or not errors else 1, "\n".join(errors))


def node_state(c: SimCluster, node: SimNode) -> tuple[str, int]:
    """(StateLong as sinfo prints it, CPUs in use)."""
    used = node_cpus_used(c, node)
    if node.state is not None:
        base = node.state
    elif used >= node.cpus:
        base = "allocated"
    elif used > 0 or len(free_gpus(c, node, None)) < node.gpu_total:
        base = "mixed"
    else:
        base = "idle"
    return base + node.flag, used


def sinfo_nodes(c: SimCluster, *, partitions: Sequence[str], sep: str = "\x1f") -> Out:
    """`sinfo -h -N -p <parts> -O 'NodeList:<sep>,…,Reason:<sep>'`: one row per (node, partition)."""
    wanted = set(partitions)
    rows: list[str] = []
    for node in sorted(c.nodes, key=lambda n: n.name):
        for part in node.partitions:
            if part not in wanted:
                continue
            state, used = node_state(c, node)
            down = node.state in ("down", "drained")
            cpus = f"0/0/{node.cpus}/{node.cpus}" if down else f"{used}/{node.cpus - used}/0/{node.cpus}"
            alloc_mem = node.bg_mem_mib + sum(j.mem_mib for j in c.jobs.values()
                                              if j.state in ("RUNNING", "COMPLETING") and node.name in j.nodes)
            free = set(free_gpus(c, node, None))
            gres = ",".join(f"gpu:{g.type}:{g.count}{g.sockets or ''}" for g in node.gres) or "(null)"
            used_parts = []
            for g, rng in node.type_ranges():
                idx = [i for i in rng if i not in free]
                used_parts.append(f"gpu:{g.type}:{len(idx)}(IDX:{compress_ranges(idx) if idx else 'N/A'})")
            values = [node.name, part, state, cpus, str(node.mem_mib), str(min(alloc_mem, node.mem_mib)),
                      str(node.free_mem_mib), gres, ",".join(used_parts) or "(null)", node.reason]
            rows.append("".join(val + sep for val in values) + "\n")
    return Out("".join(rows))


# Other users of the simulated clusters: their jobs are the nodes' background load (`bg_gpu_idx`, `bg_cpus`).
OTHER_USERS = (("bob", "Bob Example", "lab"), ("carol", "Carol Example", "lab"), ("dave", "Dave Sample", "physics"),
               ("erin", "Erin Sample", "chem"), ("frank", "Frank Placeholder", "bio"),
               ("grace", "Grace Placeholder", "physics"), ("heidi", "Heidi Demo", "chem"))
OWN_NAME = "Alice Example"
_BG_JOB_NAMES = ("ddp-train", "llama-ft", "md-sim", "eval-sweep", "diffusion", "jupyter", "rl-ppo", "vasp")
_BG_LIMITS = (4 * 3600, 86400, 2 * 86400, 7 * 86400)


def _background_jobs(c: SimCluster, node: SimNode, part: str, now: float) -> list[dict[str, str]]:
    """The jobs behind a node's background load: its busy GPUs split into one or two jobs, else one CPU job."""
    h = stable_hash(f"{c.name}/{node.name}")
    idx = sorted(node.bg_gpu_idx)
    chunks: list[list[int]] = []
    if idx:
        cut = 1 + h % len(idx) if len(idx) > 1 and h % 3 else len(idx)
        chunks = [chunk for chunk in (idx[:cut], idx[cut:]) if chunk]
    elif node.bg_cpus:
        chunks = [[]]
    rows: list[dict[str, str]] = []
    for k, chunk in enumerate(chunks):
        hk = stable_hash(f"{c.name}/{node.name}/{k}/{','.join(map(str, chunk))}")
        user, _full, account = OTHER_USERS[hk % len(OTHER_USERS)]
        limit = _BG_LIMITS[hk % len(_BG_LIMITS)]
        used = (hk // 7) % max(600, limit - 1800) + int(now - c.boot) % 600
        cpus = max(1, node.bg_cpus // len(chunks))
        tres = f"cpu={cpus},mem={mem_str(max(1024, node.bg_mem_mib // len(chunks)))},node=1,billing={cpus}"
        types = {gpu_type_at(node, i) for i in chunk}
        if chunk:
            tres += f",gres/gpu={len(chunk)}"
            if len(types) == 1 and None not in types and c.gpu_tres:
                tres += f",gres/gpu:{next(iter(types))}={len(chunk)}"
        job_id = str(7_000_000 + hk % 900_000)
        rows.append({"JobID": job_id, "ArrayJobID": job_id, "ArrayTaskID": "N/A", "UserName": user,
                     "Account": account, "Partition": part, "StateCompact": "R", "TimeUsed": squeue_time(used),
                     "TimeLimit": squeue_time(limit), "NumNodes": "1", "NodeList": node.name, "tres-alloc": tres,
                     "Name": _BG_JOB_NAMES[hk % len(_BG_JOB_NAMES)]})
    return rows


def squeue_occupants(c: SimCluster, now: float, *, nodes: Sequence[str] = (), partitions: Sequence[str] = (),
                     sep: str = "\x1f") -> Out:
    """`squeue -a -h -t R,CG {-w <nodes>|-p <parts>} -O 'F:<sep>,…'`: the user's running jobs and the other users'
    (the nodes' background load)."""
    on = set(nodes) or {n.name for n in c.nodes if set(n.partitions) & set(partitions)}
    rows: list[str] = []
    for job in sorted(c.jobs.values(), key=lambda j: j.job_id):
        if job.state in ("RUNNING", "COMPLETING") and on & set(job.nodes) and (nodes or set(partitions) &
                                                                              set(job.partition.split(","))):
            rows.append("".join(_squeue_value(c, job, f, now, "epoch") + sep for f in OCCUPANT_FIELDS))
    for node in sorted(c.nodes, key=lambda n: n.name):
        if node.name not in on or node.state in ("down", "drained"):
            continue
        part = next((p for p in node.partitions if p in partitions), node.partitions[0])
        for row in _background_jobs(c, node, part, now):
            rows.append("".join(row[f] + sep for f in OCCUPANT_FIELDS))
    return Out("".join(r + "\n" for r in rows))


def getent_names(c: SimCluster, *, users: Sequence[str]) -> Out:
    """`getent passwd U… | cut -d: -f1,5`: the simulated users' GECOS."""
    known = {u: full for u, full, _a in OTHER_USERS} | {c.user: OWN_NAME}
    return Out("".join(f"{u}:{known[u]},,,\n" for u in users if u in known))


# ====================================================================================================== capabilities
def sinfo_version(c: SimCluster) -> Out:
    return Out(f"slurm {c.slurm_version}\n")


def scontrol_partitions(c: SimCluster) -> Out:
    """`printf 'groups=…'; scontrol show partition -o`: open to every account and QOS, TRES from the nodes."""
    parts: dict[str, list[SimNode]] = {}
    for node in c.nodes:
        for p in node.partitions:
            parts.setdefault(p, []).append(node)
    rows = [f"groups={c.group}\n", "parents=complete\n"]
    for name, nodes in parts.items():
        gpus = sum(n.gpu_total for n in nodes)
        tres = f"cpu={sum(n.cpus for n in nodes)},mem={sum(n.mem_mib for n in nodes)}M,node={len(nodes)}"
        rows.append(f"PartitionName={name} AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL Hidden=NO State=UP "
                    f"TotalNodes={len(nodes)} TRES={tres}{f',gres/gpu={gpus}' if gpus else ''}\n")
    return Out("".join(rows))


def slurm_config(c: SimCluster) -> Out:
    return Out("".join(f"{k:<23} = {val}\n" for k, val in c.config.items()))


def sacctmgr_assoc(c: SimCluster) -> Out:
    return Out("".join("|".join(row) + "\n" for row in c.associations))


_HELPFORMAT = """Account AdminComment AllocCPUS AllocNodes AllocTRES AssocID AveCPU AveCPUFreq AveDiskRead AveDiskWrite
AvePages AveRSS AveVMSize BlockID Cluster Comment Constraints ConsumedEnergy ConsumedEnergyRaw Container CPUTime
CPUTimeRAW DBIndex DerivedExitCode Elapsed ElapsedRaw Eligible End ExitCode Extra FailedNode Flags GID Group JobID
JobIDRaw JobName Layout Licenses MaxDiskRead MaxDiskReadNode MaxDiskReadTask MaxDiskWrite MaxDiskWriteNode
MaxDiskWriteTask MaxPages MaxPagesNode MaxPagesTask MaxRSS MaxRSSNode MaxRSSTask MaxVMSize MaxVMSizeNode
MaxVMSizeTask McsLabel MinCPU MinCPUNode MinCPUTask NCPUS NNodes NodeList NTasks Partition Planned PlannedCPU
PlannedCPURAW Priority QOS QOSRAW QOSREQ Reason ReqCPUFreq ReqCPUFreqGov ReqCPUFreqMax ReqCPUFreqMin ReqCPUS ReqMem
ReqNodes ReqTRES Reservation ReservationId Start State StdErr StdIn StdOut Submit SubmitLine Suspended SystemComment
SystemCPU Timelimit TimelimitRaw TotalCPU TRESUsageInAve TRESUsageInMax TRESUsageInMaxNode TRESUsageInMaxTask
TRESUsageInMin TRESUsageInMinNode TRESUsageInMinTask TRESUsageInTot TRESUsageOutAve TRESUsageOutMax
TRESUsageOutMaxNode TRESUsageOutMaxTask TRESUsageOutMin TRESUsageOutMinNode TRESUsageOutMinTask TRESUsageOutTot UID
User UserCPU WCKey WCKeyID WorkDir""".split()


def sacct_helpformat(c: SimCluster) -> Out:
    lines = []
    for i in range(0, len(_HELPFORMAT), 4):
        lines.append("".join(f"{f:<20}" for f in _HELPFORMAT[i:i + 4]) + "\n")
    return Out("".join(lines))


def sacct_gputres(c: SimCluster, now: float) -> Out:
    """The `slurm.gputres` probe: TRESUsageInMax of the user's latest ended GPU job (14 days), or nothing."""
    ended = [j for j in c.jobs.values() if j.user == c.user and j.gpus and j.started and j.terminal
             and j.end is not None and now - j.end < 14 * 86400]
    if not ended:
        return Out("")
    job = max(ended, key=_sort_key)
    return Out(f"{job.display_id}|\n{job.display_id}.batch|{_usage_tres(c, job, job.batch, 'max')}\n"
               f"{job.display_id}.extern|energy=0\n")


def detect_id(h: SimHost, now: float) -> Out:
    host = h.login_host if isinstance(h, SimCluster) else h.hostname
    return Out(f"user={h.user}\nuid={h.uid}\nhost={host}\ntz={tz_text(h.tz)}\nepoch={int(now)}\n")


def detect_tools(h: SimHost) -> Out:
    return Out("".join(f"{t}\n" for t in DETECT_TOOLS if t in h.tools))


def procfs_host(h: SimHost, now: float) -> Out:
    """The §5.3.1 `procfs.host` snippet: nproc, epoch, /proc/loadavg, meminfo subset, the cpu line, uptime."""
    if isinstance(h, SimWorkstation):
        total, used = h.mem_total_kb, h.mem_used_kb + sum(p.rss_kb for p in h.procs.values())
        swap_t, swap_f = h.swap_total_kb, h.swap_free_kb
        nprocs = 2104 + len(h.procs)
    else:
        total, used = h.mem_total_kb, int(h.mem_total_kb * 0.31)
        swap_t, swap_f = 0, 0
        nprocs = 15388
    avail = max(0, total - used)
    load = h.load
    loadavg = f"{load:.2f} {load * 0.97:.2f} {load * 1.02:.2f} {max(1, int(load))}/{nprocs} {400000 + int(now) % 99999}"
    mem = "".join(f"{label:<16}{value:>8} kB\n" for label, value in (
        ("MemTotal:", total), ("MemAvailable:", avail), ("SwapTotal:", swap_t), ("SwapFree:", swap_f)))
    cpu = "cpu  " + " ".join(str(x) for x in h.cpu_jiffies)
    return Out(f"ncpu={h.ncpu}\nepoch={int(now)}\nload={loadavg}\n{mem}{cpu}\nuptime={now - h.boot:.2f}\n")


# ====================================================================================================== files
def _stat_error(h: SimHost, path: str) -> str:
    verb = h.stat_verb if isinstance(h, SimCluster) else "statx"
    return f"stat: cannot {verb} '{path}': No such file or directory"


def logstat(h: SimHost, *, paths: Sequence[str]) -> Out:
    """`stat -L -c '%n|%s|%Y' -- <paths> 2>&1; true` (errors inline, rc 0)."""
    out: list[str] = []
    for path in paths:
        lg = h.logs.get(path)
        out.append(f"{path}|{lg.size}|{int(lg.mtime)}" if lg else _stat_error(h, path))
    return Out("".join(line + "\n" for line in out))


def log_peek(h: SimHost, *, path: str, nbytes: int) -> Out:
    """`stat -L -c '%s|%Y' -- P; tail -c N -- P`."""
    lg = h.logs.get(path)
    if lg is None:
        return Out("", 1, f"{_stat_error(h, path)}\ntail: cannot open '{path}' for reading: No such file or directory")
    return Out(f"{lg.size}|{int(lg.mtime)}\n" + lg.tail(nbytes).decode("utf-8", errors="replace"))


def log_fetch(h: SimHost, *, path: str, nbytes: int) -> Out:
    lg = h.logs.get(path)
    if lg is None:
        return Out("", 1, f"tail: cannot open '{path}' for reading: No such file or directory")
    return Out(lg.tail(nbytes).decode("utf-8", errors="replace"))


def exitscan(h: SimHost, *, tasks: Sequence[tuple[str, Sequence[str]]], nbytes: int = 8192) -> Out:
    """WP2's `process.exitscan` command per ended task: `printf '%s\\n' '@@rftask <id>'; tail -q -c 8192 -- <paths>
    2>/dev/null; echo` (no file headers, unreadable files silently skipped, one newline after each task)."""
    out: list[str] = []
    for job_id, paths in tasks:
        out.append(f"@@rftask {job_id}\n")
        for path in paths:
            lg = h.logs.get(path)
            if lg is not None:
                out.append(lg.tail(nbytes).decode("utf-8", errors="replace"))
        out.append("\n")
    return Out("".join(out))


# ====================================================================================================== processes
def _num(v: float) -> str:
    """procps %CPU/%MEM: one decimal below 100, integer from 100 on (`121`, `0.7`)."""
    return f"{v:.0f}" if v >= 99.95 else f"{v:.1f}"


def _user_col(user: str) -> str:
    return (user if len(user) <= 8 else user[:7] + "+").ljust(8)


def _etime(p: SimProc, now: float) -> int:
    return max(0, int(now) - p.start)


def ps_row(p: SimProc, now: float, cols: Sequence[str], mem_total_kb: int) -> str:
    """One procps row for `-o <col>=,…` (no header): numbers right-aligned to procps' default widths."""
    cells: list[str] = []
    for col in cols:
        match col:
            case "pid":
                cells.append(f"{p.pid:>7}")
            case "ppid":
                cells.append(f"{p.ppid:>7}")
            case "pgid":
                cells.append(f"{p.pgid:>7}")
            case "user":
                cells.append(_user_col(p.user))
            case "etimes":
                cells.append(f"{_etime(p, now):>7}")
            case "pcpu":
                cells.append(f"{_num(p.cpu_pct):>4}")
            case "pmem":
                cells.append(f"{_num(p.mem_pct or 100.0 * p.rss_kb / max(1, mem_total_kb)):>4}")
            case "rss":
                cells.append(f"{p.rss_kb:>5}")
            case "stat":
                cells.append(f"{p.stat:<4}")
            case "args":
                cells.append(p.args)
    return " ".join(cells)


PS_OWN_COLS = ("pid", "ppid", "pgid", "etimes", "pcpu", "pmem", "rss", "stat", "args")
PS_APPS_COLS = ("pid", "ppid", "user", "etimes", "pcpu", "pmem", "rss", "stat", "args")


def _probe_procs(ws: SimWorkstation, now: float, cmd: str) -> list[SimProc]:
    """The probe's own processes, as `ps -u` sees them while the section runs (all younger than 60 s)."""
    base = 900000 + int(now) % 50000
    sshd = SimProc(pid=base, ppid=base - 7, pgid=base - 7, user=ws.user, start=int(now) - 1,
                   args=f"sshd-session: {ws.user}@notty")
    sh = SimProc(pid=base + 2, ppid=base, pgid=base + 2, user=ws.user, start=int(now) - 1, args="sh -s", stat="Ss")
    to = SimProc(pid=base + 31, ppid=base + 2, pgid=base + 2, user=ws.user, start=int(now),
                 args=f"timeout -k 2 5 sh -c {cmd}")
    shc = SimProc(pid=base + 32, ppid=base + 31, pgid=base + 2, user=ws.user, start=int(now), args=f"sh -c {cmd}")
    ps = SimProc(pid=base + 34, ppid=base + 32, pgid=base + 2, user=ws.user, start=int(now), stat="R",
                 args=f"ps -ww -u {ws.user} -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args=")
    return [sshd, sh, to, shc, ps]


def process_ps(ws: SimWorkstation, now: float) -> Out:
    """`date +%s; ps -ww -u "$(id -un)" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args=`."""
    own = [p for p in ws.procs.values() if p.user == ws.user]
    own += _probe_procs(ws, now, "date +%s; ps -ww -u \"$(id -un)\" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,"
                                 "stat=,args=")
    rows = [ps_row(p, now, PS_OWN_COLS, ws.mem_total_kb) for p in sorted(own, key=lambda p: p.pid)]
    return Out(f"{int(now)}\n" + "".join(r + "\n" for r in rows))


def process_fds(ws: SimWorkstation, *, pids: Sequence[int]) -> Out:
    """`printf '%s\\037%s\\037%s\\037%s\\n' pid cwd fd1 fd2` per pid (empty readlinks for vanished pids), each followed by
    `@@stat path|size|mtime` for its regular-file stdout/stderr (missing files print nothing: `2>/dev/null`)."""
    out: list[str] = []
    for pid in pids:
        p = ws.procs.get(pid)
        if p is None:
            out.append(f"{pid}\x1f\x1f\x1f\n")
            continue
        out.append(f"{pid}\x1f{p.cwd}\x1f{p.fd1}\x1f{p.fd2}\n")
        for target in (p.fd1, p.fd2):
            lg = ws.logs.get(target) if target else None
            if target and target.startswith("/") and not target.startswith("/dev/") and lg is not None:
                out.append(f"@@stat {target}|{lg.size}|{int(lg.mtime)}\n")
    return Out("".join(out))


def process_detail(ws: SimWorkstation, now: float, *, pids: Sequence[int], root: int) -> Out:
    """WP2's `process.detail`: `ps -ww -o pid=,…,args= -p <pids>; echo '@@env'; tr '\\0' '\\n' <
    /proc/<root>/environ | grep -E '^(<allow-list>)='`."""
    rows = [ps_row(ws.procs[p], now, PS_OWN_COLS, ws.mem_total_kb) for p in sorted(pids) if p in ws.procs]
    env = ws.procs[root].env if root in ws.procs else {}
    env_lines = "".join(f"{k}={v}\n" for k, v in env.items() if k in ENV_ALLOW)
    return Out("".join(r + "\n" for r in rows) + "@@env\n" + env_lines)


# ====================================================================================================== nvidia-smi
def _gpu_value(g: SimGPU, fieldname: str) -> str | None:
    match fieldname:
        case "index":
            return str(g.index)
        case "uuid":
            return g.uuid
        case "name":
            return g.name
        case "utilization.gpu":
            return str(int(g.util))
        case "utilization.memory":
            return str(int(g.mem_util))
        case "memory.used":
            return str(g.mem_used_mib)
        case "memory.total":
            return str(g.mem_total_mib)
        case "temperature.gpu":
            return str(int(round(g.temp_c)))
        case "power.draw":
            return f"{g.power_w:.2f}"
        case "power.limit":
            return f"{g.power_limit_w:.2f}"
        case "fan.speed":
            return "[N/A]" if g.fan is None else str(g.fan)
        case "pstate":
            return g.pstate
        case "clocks.sm" | "clocks.current.sm":
            return str(g.clocks_sm)
    return None


def nvidia_gpu(ws: SimWorkstation, *, fields: Sequence[str] = NVIDIA_GPU_FIELDS) -> Out:
    """`nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`."""
    lines: list[str] = []
    for g in ws.gpus:
        vals = [_gpu_value(g, f) for f in fields]
        bad = [f for f, v in zip(fields, vals) if v is None]
        if bad:
            return Out("", 2, f'Field "{bad[0]}" is not a valid field to query.')
        lines.append(", ".join(v for v in vals if v is not None) + "\n")
    return Out("".join(lines))


def compute_apps(ws: SimWorkstation) -> str:
    """`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits` (GPU order)."""
    rows = sorted((p for p in ws.procs.values() if p.gpu is not None), key=lambda p: (p.gpu or 0, p.pid))
    return "".join(f"{ws.gpus[p.gpu or 0].uuid}, {p.pid}, {p.gpu_mem_mib}\n" for p in rows)


def nvidia_apps(ws: SimWorkstation, now: float) -> Out:
    """The `nvidia.apps` payload: apps CSV, `@@ps`, then `ps` rows of those pids (all users)."""
    apps = compute_apps(ws)
    out = apps.rstrip("\n") + "\n@@ps\n"
    pids = sorted({p.pid for p in ws.procs.values() if p.gpu is not None})
    out += "".join(ps_row(ws.procs[p], now, PS_APPS_COLS, ws.mem_total_kb) + "\n" for p in pids)
    return Out(out)


def nvidia_version(ws: SimWorkstation) -> Out:
    return Out(f"NVIDIA-SMI version  : {ws.driver}\nNVML version        : {ws.nvml}\n"
               'DRIVER version      : Deprecated, see "KMD version" instead\n'
               'CUDA version        : Deprecated, see "CUDA UMD version" instead\n'
               f"KMD version         : {ws.driver}\nCUDA UMD version    : {ws.cuda}\n")


# ====================================================================================================== faults
def garbage(rng: random.Random, lines: int = 6) -> str:
    """What a confused remote might print instead of the expected table (fault "garbage")."""
    junk = ["\x1b[?1049h\x1b[22;0;0t", "Welcome to the cluster! Maintenance on Tuesday.", "▒▒▒▒ 0x7f3a9c ▒▒▒▒",
            "%%%% unexpected token near |||", "Segmentation fault (core dumped)", "<html><body>502</body></html>"]
    return "".join(rng.choice(junk) + f" {rng.randrange(10**6)}\n" for _ in range(lines))


def meta_list(meta: Mapping[str, object], key: str) -> list[str]:
    value = meta.get(key, ())
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] if value not in (None, "") else []



# ====================================================================================================== attach (§7.16)
# The nested node script of an Attach hop, played on the simulated compute node. The section formats repeat
# backends/slurm/attach.py on purpose (see the module docstring); the framing is what the nested prelude prints.
ATTACH_GPU_FIELDS = [*NVIDIA_GPU_FIELDS, "pci.bus_id"]
_GPU_BOARDS: dict[str, tuple[str, int, float]] = {     # gres type (lower) -> nvidia-smi name, memory.total, power.limit
    "nvidia_a100-sxm4-40gb": ("NVIDIA A100-SXM4-40GB", 40960, 400.0),
    "nvidia_a100-sxm4-80gb": ("NVIDIA A100-SXM4-80GB", 81920, 400.0),
    "nvidia_a100_80gb_pcie": ("NVIDIA A100 80GB PCIe", 81920, 300.0),
    "nvidia_h200_nvl": ("NVIDIA H200 NVL", 143771, 600.0),
    "nvidia_h100_80gb_hbm3": ("NVIDIA H100 80GB HBM3", 81559, 700.0),
    "nvidia_h100_nvl": ("NVIDIA H100 NVL", 95830, 400.0),
    "nvidia_geforce_rtx_3090": ("NVIDIA GeForce RTX 3090", 24576, 350.0),
    "nvidia_a30": ("NVIDIA A30", 24576, 165.0),
    "a30": ("NVIDIA A30", 24576, 165.0),
    "rtx6000": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, 600.0),   # lyra's rtx6000 (Attach field probes)
    "nvidia_rtx_pro_6000_blackwell_server_edition": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, 600.0),
    "v100": ("Tesla V100-SXM2-32GB", 32768, 300.0),
    "l40s": ("NVIDIA L40S", 46068, 350.0),
    "tesla_v100-pcie-16gb": ("Tesla V100-PCIE-16GB", 16384, 250.0),
    "tesla_p100-pcie-16gb": ("Tesla P100-PCIE-16GB", 16384, 250.0),
}
_PS_ATTACH = "ps -ww -u {user} -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args="
_B64_HEAD = "IyBydW5mbGVldCBwcm9iZSA"           # what `ps` shows of the decoded script's argv (truncated)


@dataclass(frozen=True, slots=True)
class _NodeGpu:
    idx: int                         # Slurm IDX (= /dev/nvidia minor, as on k2)
    bus: str                         # "0000:41:00.0"
    uuid: str
    name: str
    total_mib: int
    power_limit: float


def _node_gpus(c: SimCluster, node: SimNode) -> list[_NodeGpu]:
    out = []
    for idx in range(node.gpu_total):
        gtype = (gpu_type_at(node, idx) or "").lower()
        name, total, limit = _GPU_BOARDS.get(gtype, (f"NVIDIA {gtype.upper()}", 24576, 300.0))
        bus_hi = c.gpu_buses[idx % len(c.gpu_buses)]
        h = f"{stable_hash(f'gpu:{c.name}:{node.name}:{idx}'):08x}{stable_hash(f'uuid:{node.name}:{idx}'):08x}"
        uuid = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[2:6]}-{h[4:16]}"
        out.append(_NodeGpu(idx, f"0000:{bus_hi}:00.0", uuid, name, total, limit))
    return out


def _running_on(c: SimCluster, node: str) -> list[SimJob]:
    return [j for j in c.jobs.values() if j.user == c.user and j.state == "RUNNING" and node in j.nodes]


def _landed(c: SimCluster, node: str) -> SimJob | None:
    """pam_slurm_adopt puts the session into the newest own job on the node (ties: the highest job id, k2 field)."""
    jobs = _running_on(c, node)
    return max(jobs, key=lambda j: (j.start or 0.0, j.job_id)) if jobs else None


def _job_base_pid(c: SimCluster, job: SimJob, node: str) -> int:
    return 100_000 + (stable_hash(f"pids:{c.name}:{job.job_id}:{node}") % 2_800_000) // 100 * 100


def _job_procs(c: SimCluster, job: SimJob, node: str, now: float) -> list[SimProc]:
    """slurm_script → launcher → one rank per GPU of the job on this node (rank 0 also holds idle GPUs' memory)."""
    base = _job_base_pid(c, job, node)
    start = int(job.start or now)
    ngpu = max(1, len(job.gpu_idx.get(node, ())))
    script = SimProc(pid=base, ppid=base - 9, pgid=base, user=c.user, start=start,
                     args=f"/bin/bash /var/spool/slurmd/job{job.job_id}/slurm_script {job.submit_args}".rstrip())
    launcher = SimProc(pid=base + 11, ppid=base, pgid=base, user=c.user, start=start + 1, rss_kb=21_236,
                       args=f"python -m orb.launch --job-name {job.name} --per-gpu 1")
    out = [script, launcher]
    busy = 0.1 if job.stalled else 0.9 + 0.1 * (job.job_id % 3)
    for k in range(ngpu):
        rss = (job.batch.ave_rss_kb // ngpu) if job.batch.ave_rss_kb else 4_000_000
        out.append(SimProc(pid=base + 40 + k, ppid=base + 11, pgid=base, user=c.user, start=start + 3,
                           cpu_pct=100.0 * busy, rss_kb=rss, stat="Rl" if not job.stalled else "Sl",
                           args=f"python -u train.py --rank {k} --run {job.name}"))
    return out


def _probe_tree(c: SimCluster, mode: str, now: float) -> tuple[int, list[SimProc]]:
    """The probe's own processes as `ps -u` sees them inside the node script; returns (node.ps shell pid, rows)."""
    base = 900_000 + int(now) % 50_000 * 10
    t = int(now)
    decoder = f"echo {_B64_HEAD}... | base64 -d | sh"
    rows: list[SimProc] = []
    if mode == "ssh":
        rows += [SimProc(pid=base, ppid=base - 1, pgid=base, user=c.user, start=t, args=f"sshd-session: {c.user}@notty"),
                 SimProc(pid=base + 1, ppid=base, pgid=base + 1, user=c.user, start=t, stat="Ss",
                         args=f"bash -c timeout -k 1 10 sh -c '{decoder}'")]
        top = SimProc(pid=base + 2, ppid=base + 1, pgid=base + 1, user=c.user, start=t,
                      args=f"timeout -k 1 10 sh -c {decoder}")
    else:
        top = SimProc(pid=base + 2, ppid=base - 1, pgid=base + 2, user=c.user, start=t,
                      args=f"timeout -k 1 15 sh -c {decoder}")
    ps_cmd = _PS_ATTACH.format(user=c.user)
    rows += [top,
             SimProc(pid=base + 3, ppid=base + 2, pgid=top.pgid, user=c.user, start=t, args=f"sh -c {decoder}"),
             SimProc(pid=base + 5, ppid=base + 3, pgid=top.pgid, user=c.user, start=t, args="sh"),
             SimProc(pid=base + 30, ppid=base + 5, pgid=top.pgid, user=c.user, start=t, args="sh"),
             SimProc(pid=base + 31, ppid=base + 30, pgid=top.pgid, user=c.user, start=t,
                     args=f'timeout -k 2 5 sh -c echo "$$"; {ps_cmd}'),
             SimProc(pid=base + 32, ppid=base + 31, pgid=top.pgid, user=c.user, start=t, args=f'sh -c echo "$$"; {ps_cmd}'),
             SimProc(pid=base + 34, ppid=base + 32, pgid=top.pgid, user=c.user, start=t, stat="R", args=ps_cmd)]
    return base + 32, rows


def _cg_dir(c: SimCluster, job: SimJob, node: str) -> tuple[str, str]:
    """(scope, job directory name) of the job's cgroup: lyra (Slurm 26.05) uses the SLUID."""
    v = tuple(int(x) for x in c.slurm_version.split(".")[:2])
    if v >= (26, 5):
        return "/system.slice/slurmstepd.scope", _sluid(job.job_id)
    return f"/system.slice/{node}_slurmstepd.scope", f"job_{job.job_id}"


def _gpu_reading(job: SimJob, pos: int, now: float) -> tuple[float, int]:
    """(util %, memory MiB) of the job's pos-th GPU on a node: idle ones sit at 0 % holding their memory; the busy
    ones carry the job's whole utilisation (sstat's sum stays util_target × n)."""
    if pos in job.idle_gpus:
        return 0.0, job.idle_gpus[pos]
    n = max(1, job.gpus_per_node)
    busy = max(1, n - len(job.idle_gpus))
    target = 1.0 if job.stalled else min(100.0, job.util_target * n / busy)
    wobble = (stable_hash(f"util:{job.job_id}:{pos}:{int(now) // 15}") % 5) - 2
    util = max(0.0, min(100.0, target + (0 if job.stalled else wobble)))
    return util, job.mem_target_mib + 11


def _attach_sections(c: SimCluster, now: float, meta: Mapping[str, object]) -> dict[str, Out]:
    mode = str(meta.get("mode", "ssh"))
    node_name = str(meta.get("node", ""))
    target_raw = str(meta.get("job_id_raw", ""))
    jobs_raw = meta_list(meta, "jobs_raw")
    sluids = meta_list(meta, "sluids") or [""] * len(jobs_raw)
    node = c.node(node_name)
    target = c.jobs.get(int(target_raw)) if target_raw.isdigit() else None
    assert node is not None and target is not None
    landed = target if mode == "srun" else _landed(c, node_name)
    assert landed is not None
    scope, jdir = _cg_dir(c, landed, node_name)
    step = str(c.attach_steps.get(target.job_id, 1) - 1) if mode == "srun" else "extern"
    gpus = _node_gpus(c, node)
    held = sorted((g for g in gpus if g.idx in landed.gpu_idx.get(node_name, ())), key=lambda g: g.bus)
    step_gpus = ",".join(str(i) for i in landed.gpu_idx.get(node_name, ())) if mode == "srun" else ""
    out: dict[str, Out] = {}
    out["node.id"] = Out(f"{node_name}\n0::{scope}/{jdir}/step_{step}/user/task_0\nJ={landed.job_id} G={step_gpus}\n")

    blocks = []
    for raw, sluid in zip(jobs_raw, sluids, strict=False):
        job = c.jobs.get(int(raw)) if raw.isdigit() else None
        found = job is not None and job.state == "RUNNING" and node_name in job.nodes
        if found:
            assert job is not None
            jscope, jname = _cg_dir(c, job, node_name)
            found = jname == f"job_{raw}" or jname == sluid       # the script tries job_<J>, then the SLUID it got
        if not found or job is None:
            blocks.append(f"@@cg {raw} -\n")
            continue
        rss = job.batch.ave_rss_kb * 1024
        shmem = job.shmem_mib * 1024 * 1024
        file_ = 2_400_000_000 + job.job_id % 997 * 1_000_000
        current = rss + shmem + file_
        limit = job.mem_mib * 1024 * 1024
        cores = 0.05 if job.stalled else 1.0 + job.cpus * 0.1
        usage = int(job.elapsed(now) * cores * 1e6)
        blocks.append(f"@@cg {raw} /sys/fs/cgroup{jscope}/{jname}\nmemory.current {current}\n"
                      f"memory.peak {min(limit, int(current * 1.25))}\nmemory.max {limit}\n"
                      f"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
                      f"anon {rss}\nfile {file_}\nshmem {shmem}\nusage_usec {usage}\n")
    out["node.cg"] = Out("".join(blocks))

    self_pid, probe = _probe_tree(c, mode, now)
    procs = [p for j in _running_on(c, node_name) for p in _job_procs(c, j, node_name, now)] + probe
    mem_kb = node.mem_mib * 1024
    out["node.ps"] = Out(f"{self_pid}\n" + "".join(ps_row(p, now, PS_OWN_COLS, mem_kb) + "\n"
                                                   for p in sorted(procs, key=lambda p: p.pid)))

    tables = []
    probe_step_of = landed.job_id
    for raw in jobs_raw:
        job = c.jobs.get(int(raw)) if raw.isdigit() else None
        if job is None or job.state != "RUNNING" or node_name not in job.nodes:
            continue
        base = _job_base_pid(c, job, node_name)
        rows = [(base, "batch", "0"), (base - 9, "batch", "-")] + [(p.pid, "batch", "-")
                                                                  for p in _job_procs(c, job, node_name, now)[1:]]
        rows += [(-1, "extern", "0"), (base - 20, "extern", "-")]
        if job.job_id == probe_step_of:
            rows += [(p.pid, step, "-") for p in probe]
        tables.append("PID      JOBID    STEPID   LOCALID GLOBALID\n" + "".join(
            f"{pid:<8} {job.job_id:<8} {sid:<8} {loc:<7} {loc:<9}\n" for pid, sid, loc in rows))
    out["node.pids"] = Out("".join(tables))

    out["node.map"] = Out("".join(
        f"/proc/driver/nvidia/gpus/{g.bus}/information:GPU UUID: \t {g.uuid}\n"
        f"/proc/driver/nvidia/gpus/{g.bus}/information:Bus Location: \t {g.bus}\n"
        f"/proc/driver/nvidia/gpus/{g.bus}/information:Device Minor: \t {g.idx}\n" for g in gpus))

    if not held:
        out["node.gpu"] = Out("No devices were found\n", 6)
        out["node.apps"] = Out("", 6, "No devices were found")
    else:
        rows_gpu = []
        apps = []
        order = list(landed.gpu_idx.get(node_name, ()))
        ranks = _job_procs(c, landed, node_name, now)[2:]
        for i, g in enumerate(held):
            pos = order.index(g.idx)
            util, mem = _gpu_reading(landed, pos, now)
            temp = 30 + util * 0.43
            power = g.power_limit * (0.15 + 0.8 * util / 100)
            dom, bus_rest = g.bus.split(":", 1)
            rows_gpu.append(f"{i}, {g.uuid}, {g.name}, {int(util)}, {mem}, {g.total_mib}, {int(temp)}, {power:.2f}, "
                            f"{g.power_limit:.2f}, [N/A], P0, {int(util * 0.6)}, "
                            f"0000{dom}:{bus_rest.upper()}\n")
            holder = ranks[0] if pos in landed.idle_gpus or pos >= len(ranks) else ranks[pos]
            apps.append(f"{g.uuid}, {holder.pid}, {mem - 11}\n")
        out["node.gpu"] = Out("".join(rows_gpu))
        out["node.apps"] = Out("".join(apps))
    out["node.self"] = Out("@@done\n")             # no nvidia-smi of the user left on the node (stuck or healthy)
    return out


def attach_node(c: SimCluster, now: float, meta: Mapping[str, object]) -> Out:
    """One `slurm.attach.node<i>` section: the login → node hop (ssh or srun) and, when it lands, the nested node
    script's framed output (nonce `meta.inner_nonce`, sections `meta.inner` in order)."""
    mode = str(meta.get("mode", "ssh"))
    node_name = str(meta.get("node", ""))
    target_raw = str(meta.get("job_id_raw", ""))
    target = c.jobs.get(int(target_raw)) if target_raw.isdigit() else None
    if c.node(node_name) is None:
        if mode == "ssh":
            return Out("", 255, f"ssh: Could not resolve hostname {node_name}: Name or service not known")
        return Out("", 1, f"srun: error: Invalid node name specified: {node_name}")
    if mode == "ssh":
        if c.node_ssh == "denied":
            return Out("", 255, f"{c.user}@{node_name}: Permission denied "
                                "(publickey,gssapi-keyex,gssapi-with-mic,password,hostbased).")
        if _landed(c, node_name) is None:
            return Out("", 255, "")          # pam_slurm_adopt: its banner is hidden by LogLevel=ERROR
    elif target is None or target.state != "RUNNING" or node_name not in target.nodes:
        return Out("", 1, f"srun: error: Unable to create step for job {target_raw}: "
                          "Job/step already completing or completed")
    else:
        c.attach_steps[target.job_id] = c.attach_steps.get(target.job_id, 0) + 1
    nonce = str(meta.get("inner_nonce", ""))
    outs = _attach_sections(c, now, meta)
    rf = f"@@RF{nonce}"
    text: list[str] = []
    for k, name in enumerate(meta_list(meta, "inner")):
        o = outs.get(name, Out("", 1, f"demo: no simulation for node section {name!r}"))
        text.append(f"\n{rf} B {name}\n{o.stdout}")
        if o.stderr:
            text.append(f"\n{rf} X {name}\n{o.stderr}\n{rf} Y {name}\n")
        text.append(f"\n{rf} E {name} {o.rc} {3 + (stable_hash(name) + k) % 30}\n")
    return Out("".join(text)[:-1])


def attach_audit(c: SimCluster, now: float, meta: Mapping[str, object]) -> Out:
    """`slurm.attach.audit`: the job's live steps (no rfprobe: demo steps end at once) and no ssh client left."""
    raw = str(meta.get("job_id_raw", ""))
    job = c.jobs.get(int(raw)) if raw.isdigit() else None
    if job is None or job.state not in ("RUNNING", "COMPLETING"):
        steps = "slurm_load_jobs error: Invalid job id specified\n"
    else:
        steps = f"{job.display_id}.batch batch\n{job.display_id}.extern extern\n"
        if job.step_tasks:
            steps += f"{job.display_id}.0 python\n"
    return Out(f"@@steps\n{steps}@@ssh\n0\n")
