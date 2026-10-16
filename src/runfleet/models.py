"""[WP1] FROZEN shared dataclasses and enums (DESIGN.md §3.3)."""

from __future__ import annotations

import dataclasses
import functools
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from typing import Any, Literal, TypeVar

T = TypeVar("T")

# ---------- hosts ----------
class ConnState(StrEnum):
    NEW = "new"                  # configured, never probed
    CONNECTING = "connecting"    # first probe / ensure_master in flight (no probe is sent while a master is being made)
    OK = "ok"
    DEGRADED = "degraded"        # transport fine, >= 1 section failing (slurmdbd down, nvidia-smi hung, parse errors)
    RETRYING = "retrying"        # 1-2 consecutive transport failures, backing off
    UNREACHABLE = "unreachable"  # >= 3 consecutive transport failures, or DNS failure
    NEEDS_AUTH = "needs_auth"    # BatchMode auth failure -> polling paused until connect
    HOSTKEY = "hostkey"          # unknown/changed host key -> paused until connect
    DISABLED = "disabled"        # user toggled off / enabled = false

class ErrorKind(StrEnum):        # defined ONLY here; errors.py holds exception classes only
    AUTH = "auth"
    HOSTKEY = "hostkey"
    DNS = "dns"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    JUMP = "jump"
    MUX = "mux"
    SSH = "ssh"
    REMOTE = "remote"
    MISSING = "missing"
    PARSE = "parse"
    PERMISSION = "permission"
    INTERNAL = "internal"

MuxMode = Literal["user", "managed", "off", "session", "local", "none"]   # "none" = fake/demo transports;
                                                                         # "session": one ssh + agent (§4.8)

class Visibility(StrEnum):
    HIDDEN = "hidden"
    VISIBLE = "visible"
    FOCUSED = "focused"

@dataclass(frozen=True, slots=True)
class HostError:
    kind: ErrorKind
    summary: str                 # one line for badges/toasts, <= 80 chars
    detail: str = ""             # stderr excerpt (<= 4 KiB)
    section: str | None = None   # None = transport level
    at: datetime | None = None

@dataclass(frozen=True, slots=True)
class SectionStatus:
    name: str
    outcome: str                 # SectionOutcome value (probe/section.py)
    ok: bool
    rc: int | None
    ms: int | None
    last_ok: datetime | None
    error: HostError | None = None

DETECT_REV = 2      # bump when detection learns something new: cached capabilities of an older revision are re-detected


@dataclass(frozen=True, slots=True)
class Capabilities:
    remote_user: str
    uid: int | None
    login_host: str                          # `hostname` of the ssh target
    tz_offset: timedelta                     # `date +%z` at detect; ONLY for ISO timestamps (fixtures/fallback)
    clock_skew: timedelta                    # remote `date +%s` - local, informational
    tools: frozenset[str]                    # names found by `command -v`
    profile: tuple[str, ...] = ()            # collector names, e.g. ("slurm", "procfs")
    slurm_version: str | None = None
    cluster_name: str | None = None
    slurm_config: Mapping[str, str] = field(default_factory=dict)   # AccountingStorageType, PreemptMode, MinJobAge, ...
    acct_gather_freq_s: int | None = None
    gpu_tres: bool | None = None             # cluster records gres/gpuutil (None = not known yet), §5.1
    sacct_fields: frozenset[str] = frozenset()
    associations: tuple[tuple[str, str, str, str], ...] = ()       # (cluster, account, partition, qos)
    gpu_partitions: tuple[str, ...] = ()     # GPU partitions the user may submit to (scontrol show partition), §5.1
    detect_rev: int = 0                      # DETECT_REV of the detection that produced these (0 = before 0.2.1)
    driver_version: str | None = None        # nvidia "KMD version" (fallback "DRIVER version")
    node_ssh: Literal["adopt", "denied"] | None = None   # Attach (§7.16), learned via caps_patch by the first node
                                             # ssh: "adopt" = login→node ssh lands in a job (pam_slurm_adopt);
                                             # "denied" = refused before/at auth (lyra). None = not tried yet

@dataclass(frozen=True, slots=True)
class HostMetrics:
    ncpu: int | None
    load: tuple[float, float, float] | None
    cpu_pct: float | None        # /proc/stat delta between two samples; None on the first
    mem_total: int | None
    mem_available: int | None
    swap_total: int | None
    swap_free: int | None
    uptime_s: float | None
    sampled_at: datetime

@dataclass(frozen=True, slots=True)
class HostStatus:
    name: str
    state: ConnState
    error: HostError | None = None
    last_ok: datetime | None = None
    last_attempt: datetime | None = None
    latency_s: float | None = None           # wall time of the last successful probe
    consecutive_failures: int = 0
    next_poll_at: datetime | None = None
    tick_interval: float | None = None       # current effective tick period after all factors (§5.4)
    in_flight: bool = False
    mux: MuxMode = "none"
    ssh_log: str | None = None               # `-E` log of master creation / connect, shown in HostErrorScreen
    sections: Mapping[str, SectionStatus] = field(default_factory=dict)
    controller_slowdown: float = 1.0         # busy Slurm controller: the Slurm cadence runs ×2 … ×8 (§5.4); 1 = normal

# ---------- jobs ----------
class JobKind(StrEnum):
    SLURM = "slurm"
    TASK = "task"                # workstation process tree

class StateGroup(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ENDED = "ended"              # workstation task gone, exit status unknown (shown next to failures)
    UNKNOWN = "unknown"

class JobState(StrEnum):         # value = short code shown in the ST column
    PENDING = "PD"
    CONFIGURING = "CF"
    REQUEUED = "RQ"
    RUNNING = "R"
    COMPLETING = "CG"
    SUSPENDED = "S"
    STOPPED = "ST"
    COMPLETED = "CD"
    EXITED = "EX"
    FAILED = "F"
    TIMEOUT = "TO"
    OUT_OF_MEMORY = "OOM"
    NODE_FAIL = "NF"
    PREEMPTED = "PR"
    BOOT_FAIL = "BF"
    DEADLINE = "DL"
    CANCELLED = "CA"
    UNKNOWN = "?"

    @property
    def group(self) -> StateGroup:
        return _STATE_GROUP[self]

    @property
    def is_terminal(self) -> bool:
        return _STATE_GROUP[self] not in (StateGroup.QUEUED, StateGroup.ACTIVE, StateGroup.UNKNOWN)

_STATE_GROUP: dict[JobState, StateGroup] = {
    JobState.PENDING: StateGroup.QUEUED, JobState.CONFIGURING: StateGroup.QUEUED, JobState.REQUEUED: StateGroup.QUEUED,
    JobState.RUNNING: StateGroup.ACTIVE, JobState.COMPLETING: StateGroup.ACTIVE, JobState.SUSPENDED: StateGroup.ACTIVE,
    JobState.STOPPED: StateGroup.ACTIVE,
    JobState.COMPLETED: StateGroup.DONE,
    JobState.EXITED: StateGroup.ENDED,
    JobState.FAILED: StateGroup.FAILED, JobState.TIMEOUT: StateGroup.FAILED, JobState.OUT_OF_MEMORY: StateGroup.FAILED,
    JobState.NODE_FAIL: StateGroup.FAILED, JobState.PREEMPTED: StateGroup.FAILED, JobState.BOOT_FAIL: StateGroup.FAILED,
    JobState.DEADLINE: StateGroup.FAILED,
    JobState.CANCELLED: StateGroup.CANCELLED,
    JobState.UNKNOWN: StateGroup.UNKNOWN,
}

_TASK_ID = re.compile(r"^p(\d+)-(\d+)$")

@dataclass(frozen=True, slots=True, order=True)
class JobKey:
    host: str                    # RunFleet host name (config `name`)
    id: str                      # Slurm display id: "2332861", "31270818_3", "31270818_[1-24%1]"
                                 # task: "p<root pid>-<root start epoch>", e.g. "p642229-1788696459"

    def __str__(self) -> str:
        return f"{self.host}:{self.id}"

    @property
    def label(self) -> str:      # what the ID column shows: task -> "p642229", Slurm unchanged
        m = _TASK_ID.match(self.id)
        return f"p{m.group(1)}" if m else self.id

@dataclass(frozen=True, slots=True)
class NodeAlloc:
    node: str
    cpu_ids: str | None          # "64-71"
    mem_bytes: int | None
    gpu_indices: tuple[int, ...] # physical IDX from scontrol -d GRES=...(IDX:0-1); () if unknown
    gres: str | None             # raw "gpu:nvidia_h200_nvl:2(IDX:0-1)"

@dataclass(frozen=True, slots=True)
class ResourceAlloc:
    nodes: int | None
    cpus: int | None
    mem_bytes: int | None        # total for the job
    gpus: int                    # 0 when none
    gpu_type: str | None         # normalized lower-case gres type: "nvidia_a100-sxm4-40gb", "rtx6000"
    tres: Mapping[str, str]      # raw TRES dict (alloc if running, req if pending)
    per_node: tuple[NodeAlloc, ...] = ()
    gpu_type_source: Literal["tres-alloc", "scontrol", "tres-req", "sinfo", "cache", "nvidia-smi"] | None = None

@dataclass(frozen=True, slots=True)
class LogRef:
    stream: Literal["stdout", "stderr"]
    path: str | None             # absolute, expanded; None if unknown
    pattern: str | None = None   # raw Slurm pattern ("…/logs/%x_%j.out") if that is all we have
    merged: bool = False         # stderr goes to the stdout file
    source: Literal["scontrol", "squeue", "sacct", "template", "procfd", "cache", "demo"] = "squeue"
    size: int | None = None      # from the last stat
    mtime: datetime | None = None
    error: str | None = None     # "No such file", "Permission denied"

@dataclass(frozen=True, slots=True)
class Proc:                      # one `ps` row (workstations)
    pid: int
    ppid: int
    pgid: int | None             # None when the source did not report it (nvidia.apps ps)
    user: str
    etime_s: int
    cpu_pct: float
    mem_pct: float
    rss_bytes: int
    stat: str
    args: str
    task: JobKey | None = None   # set by ProcessCollector for members of a task

@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    gpu_uuid: str
    gpu_index: int | None
    used_mem_mib: float | None
    user: str | None
    command: str | None
    etime_s: int | None
    cpu_pct: float | None
    job: JobKey | None           # task that owns the pid (set by ProcessCollector)

@dataclass(frozen=True, slots=True)
class GPU:
    host: str
    index: int
    uuid: str | None
    name: str                    # "NVIDIA RTX A6000"
    util_pct: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    mem_util_pct: float | None
    temp_c: float | None
    power_w: float | None
    power_limit_w: float | None
    fan_pct: float | None
    pstate: str | None
    source: Literal["nvidia-smi", "demo"]
    processes: tuple[GPUProcess, ...] = ()
    sampled_at: datetime | None = None
    bus_id: str | None = None    # "00000000:41:00.0" (nvidia-smi pci.bus_id); Attach pairs GPUs by uuid or bus id
    slurm_idx: int | None = None # Slurm GRES IDX (Attach, §7.16 IDX mapping); `index` stays nvidia-smi's own index,
                                 # which inside a job step is 0..n-1 and NOT the physical/Slurm number

@dataclass(frozen=True, slots=True)
class JobGpuUsage:
    source: Literal["sstat", "nvidia-smi", "demo"]
    n_gpus: int
    util_mean_pct: float | None  # sstat: Tot gpuutil / n_gpus clamped to 100; nvidia-smi: mean over the task's GPUs
    util_sum_pct: float | None   # sstat: TRESUsageInTot gres/gpuutil (sum over tasks)
    util_peak_pct: float | None  # sstat: min(100, Max gpuutil x NTasks / n_gpus) (peak per GPU, est.); nvidia-smi: max
    mem_used_mib: float | None   # total over the job's GPUs
    mem_total_mib: float | None  # nvidia-smi, or gpu_types table x n_gpus
    sampled_at: datetime | None = None
    note: str = ""               # "Slurm accounting · refreshed every 60 s", "GPU-level; shared GPUs count others"

@dataclass(frozen=True, slots=True)
class PriorityFactor:
    """One term of a pending job's priority/multifactor priority (§5.2.10), as `sprio` weights it."""
    name: Literal["age", "assoc", "fairshare", "jobsize", "partition", "qos", "tres", "nice"]
    value: float                 # weighted contribution (sprio %A %B %F %J %P %Q %T; nice: minus the adjustment)
    weight: float | None = None  # the cluster's PriorityWeight<Factor>: the most this factor can add; None for nice

@dataclass(frozen=True, slots=True)
class FairShare:
    """One `sshare -U` row of the user: fair-share standing in one account (§5.2.10)."""
    account: str
    user: str
    raw_shares: str | None       # "1", "parent"
    norm_shares: float | None    # share among the siblings, 0..1
    raw_usage: int | None        # decayed usage (TRES-seconds)
    effective_usage: float | None    # decayed usage relative to the siblings, 0..1
    fairshare: float | None      # the fair-share factor 0..1 (×PriorityWeightFairShare in the job's priority)
    level_fs: float | None       # NormShares / EffectvUsage at this level (> 1: under-served); None: "inf" (no
                                 # usage) or unknown

@dataclass(frozen=True, slots=True)
class PendingPriority:
    """Why a pending Slurm job waits, as evidence (§5.2.10): its priority factors weighted as the cluster weights
    them, the user's fair share and plain-language hints. Never a start time or a queue position."""
    ranked: bool                 # False: the cluster does not rank pending jobs by priority (priority/basic, or
                                 # priority/multifactor with every PriorityWeight* 0): submit order + backfill
    priority: float | None = None            # sprio %Y of the job's best partition; None: not read yet
    partition: str | None = None             # partition of that sprio row
    factors: tuple[PriorityFactor, ...] = () # every weighted factor (weight > 0), largest contribution first
    dominant: str | None = None              # name of the largest contribution (None: all 0 / not read)
    others: tuple[tuple[str, float], ...] = ()   # (partition, priority) of the job's other sprio rows
    fairshare: tuple[FairShare, ...] = ()    # the user's sshare rows; the job's account first
    hints: tuple[str, ...] = ()              # plain language, most telling first
    sampled_at: datetime | None = None       # when sprio was read (None: never)
    half_life_s: float | None = None         # PriorityDecayHalfLife (0 = usage does not decay)

@dataclass(frozen=True, slots=True)
class Job:
    key: JobKey
    kind: JobKind
    name: str
    user: str
    state: JobState
    raw_state: str               # "CANCELLED by 10001", "RUNNING", "exited (log: Traceback ...)"
    job_id_raw: str              # Slurm: squeue JobID / sacct JobIDRaw (for scontrol, sstat, sacct -j, %j); task: root pid
    reason: str | None = None    # Slurm Reason, raw
    held: bool = False
    submit_time: datetime | None = None
    start_time: datetime | None = None   # PD: scheduler estimate (est_start=True)
    est_start: bool = False
    end_time: datetime | None = None
    elapsed: timedelta | None = None     # at observed_at; the UI extrapolates for ACTIVE jobs
    time_limit: timedelta | None = None  # None = unlimited/unknown
    exit_code: int | None = None
    exit_signal: int | None = None
    derived_exit: str | None = None
    partition: str | None = None
    qos: str | None = None
    account: str | None = None
    alloc: ResourceAlloc = field(default_factory=lambda: ResourceAlloc(None, None, None, 0, None, {}))
    nodes: tuple[str, ...] = ()          # expanded hostnames
    nodelist: str | None = None          # raw compressed
    batch_host: str | None = None
    workdir: str | None = None
    command: str | None = None
    logs: tuple[LogRef, ...] = ()
    array_job_id: str | None = None
    array_task: str | None = None        # "3" or "[1-20%1]"
    restarts: int = 0
    pids: tuple[int, ...] = ()           # task: all current members, root first
    cpu_pct: float | None = None         # task: sum over members
    rss_bytes: int | None = None         # task: sum over members
    gpu: JobGpuUsage | None = None
    hints: tuple[str, ...] = ()          # "log idle 47m", "GPU idle 20m", "will never run: dependency failed"
    is_mine: bool = True
    source: Literal["squeue", "sacct", "vanished", "ps", "cache", "demo"] = "squeue"
    observed_at: datetime | None = None
    raw: Mapping[str, str] = field(default_factory=dict)   # every raw field (detail "Raw" tab)
    priority: PendingPriority | None = None  # own pending Slurm jobs: priority evidence (§5.2.10); else None

# ---------- cluster ----------
@dataclass(frozen=True, slots=True)
class GresCount:
    name: str                    # "gpu" | "shard"
    type: str | None             # "nvidia_a100_80gb_pcie_3g.40gb" (MIG profiles are distinct types)
    total: int
    used: int
    used_indices: str | None     # "0,2-3" / None for N/A

@dataclass(frozen=True, slots=True)
class Node:
    name: str
    partitions: tuple[str, ...]
    state: str                   # base state, lower-case: idle mixed allocated completing planned reserved
                                 # drained draining down fail failing maint future inval unknown power_down ...
    flags: frozenset[str]        # "*" not responding, "~" powered off, "#" powering up, "!" pending power down,
                                 # "%" powering down, "$" maintenance, "@" pending reboot, "^" rebooting,
                                 # "-" planned by backfill (still usable), "+" multiple states
    usable: bool                 # contributes free GPUs (rule §5.2.3)
    cpus_alloc: int | None
    cpus_idle: int | None
    cpus_other: int | None
    cpus_total: int | None
    mem_total_mib: int | None
    mem_alloc_mib: int | None
    mem_free_mib: int | None
    gres: tuple[GresCount, ...]
    reason: str | None
    jobs: tuple[NodeJob, ...] | None = None   # running jobs of every user (§5.2.9); None = not sampled lately
    jobs_at: datetime | None = None           # when `jobs` was sampled

    def jobs_seen(self, now: datetime) -> tuple[NodeJob, ...] | None:
        """`jobs` while the reading is fresh (NODE_JOBS_TTL), else None: an old reading is never shown."""
        if self.jobs is None or self.jobs_at is None or now - self.jobs_at > NODE_JOBS_TTL:
            return None
        return self.jobs

NODE_JOBS_TTL = timedelta(minutes=3)          # every user's jobs on a node: how long a reading is shown (§5.2.9)

@dataclass(frozen=True, slots=True)
class NodeJob:
    """A running job on one node, any user (§5.2.9: sampled while the host's detail is open)."""
    job_id: str                  # display id (`123_4` for an array task)
    user: str
    real_name: str | None        # GECOS of `getent passwd` (None: unknown or not looked up yet)
    account: str | None
    partition: str | None
    name: str
    state: str                   # squeue compact state: "R" / "CG"
    gpus: int | None             # GPUs on this node (None: unknown); a multi-node job's split when not exact
    gpus_exact: bool             # False: an even split of a multi-node job's total
    gpu_type: str | None
    elapsed_s: int | None
    limit_s: int | None          # None: UNLIMITED / unknown
    mine: bool

    @property
    def left_s(self) -> int | None:
        if self.limit_s is None or self.elapsed_s is None:
            return None
        return max(0, self.limit_s - self.elapsed_s)

@dataclass(frozen=True, slots=True)
class PartitionSummary:
    name: str
    nodes: int
    gpus_total: int
    gpus_free: int               # sum over usable nodes of (total - used)
    gpus_free_planned: int       # part of gpus_free on nodes that are planned for a pending job ("-" / planned)
    gpu_types: Mapping[str, tuple[int, int]]   # type -> (free, total)
    states: Mapping[str, int]    # base state -> node count

# ---------- details ----------
@dataclass(frozen=True, slots=True)
class JobStep:
    step_id: str                 # "2332861.batch", "31354789.0"
    name: str
    state: JobState
    raw_state: str
    exit_code: int | None
    exit_signal: int | None
    elapsed: timedelta | None
    start: datetime | None
    end: datetime | None
    nodelist: str | None
    ntasks: int | None
    max_rss_bytes: int | None
    ave_cpu: timedelta | None
    tres_usage_max: Mapping[str, str] = field(default_factory=dict)
    tres_usage_tot: Mapping[str, str] = field(default_factory=dict)
    total_cpu: timedelta | None = None   # sacct TotalCPU (detail probe, when caps.sacct_fields has it)

@dataclass(frozen=True, slots=True)
class JobDetail:
    job: Job
    fields: Mapping[str, str]            # scontrol key/values, or ps/procfs facts for tasks
    per_node: tuple[NodeAlloc, ...]
    steps: tuple[JobStep, ...]
    logs: tuple[LogRef, ...]             # with fresh stat info
    gpu: JobGpuUsage | None
    procs: tuple[Proc, ...]              # task members; () for Slurm in v0.1
    procs_note: str                      # e.g. "per-process view needs compute-node access (v0.2)"
    env: Mapping[str, str]               # allow-listed env of the task root (Appendix A process.detail)
    explanation: tuple[str, ...]         # "batch step FAILED 143:0 = SIGTERM (128+15): preempted"
    fetched_at: datetime
    errors: tuple[HostError, ...] = ()
    efficiency: tuple[str, ...] = ()     # plain efficiency card (util/efficiency.lines), Slurm jobs that ran
    live: JobLive | None = None          # Attach (§7.16): the job's latest live sample this session, filled in by
                                         # FleetService.job_detail from AttachState.live (collectors leave it None)

# ---------- attach (§7.16) ----------
AttachMode = Literal["ssh", "srun"]      # "ssh": login → `ssh <node>` landing via pam_slurm_adopt (no sacct row);
                                         # "srun": `srun --jobid --overlap` one-shot step (ONE permanent sacct row)

@dataclass(frozen=True, slots=True)
class CgroupStats:                       # one job's cgroup v2 directory on one node (bytes / counts)
    mem_current: int | None = None       # memory.current
    mem_peak: int | None = None          # memory.peak
    mem_max: int | None = None           # memory.max; "max" (unlimited) -> None
    anon: int | None = None              # memory.stat anon
    file: int | None = None              # memory.stat file
    shmem: int | None = None             # memory.stat shmem (counts against memory.max; part of sstat MaxRSS)
    oom: int | None = None               # memory.events oom
    oom_kill: int | None = None          # memory.events oom_kill
    cpu_usage_usec: int | None = None    # cpu.stat usage_usec (cumulative)
    cpu_cores: float | None = None       # Δcpu_usage_usec / Δwall between this and the previous sample of the same
                                         # (node, job); None on the first sample (computed by ingest_attach)

@dataclass(frozen=True, slots=True)
class NodeInspect:                       # one compute node of one Attach sample
    node: str                            # short hostname ("n-37-08")
    target_job: str                      # job_id_raw of the attached job
    landed_job: str | None = None        # raw job id the node session ran in: ssh = the job pam_slurm_adopt chose
                                         # (from /proc/self/cgroup, else $SLURM_JOB_ID); srun = target_job
    gpus_of_target: bool | None = None   # True: `gpus` are the target job's; False: they belong to landed_job
                                         # (ssh landed in another own job on this node); None: unknown
    gpus: tuple[GPU, ...] = ()           # GPU.host = RunFleet host name; bus_id/slurm_idx set when known;
                                         # GPU.processes = compute apps of this sample (GPUProcess.job set when the
                                         # pid's job is known from listpids)
    procs: tuple[Proc, ...] = ()         # `ps` rows of the user's processes on the node (all own jobs, task=None)
    step_of_pid: Mapping[int, str] = field(default_factory=dict)   # pid -> "<raw job id>.<step>" (scontrol listpids:
                                         # "724655.batch", "31367626.0", "2332863.extern")
    cgroups: Mapping[str, CgroupStats] = field(default_factory=dict)   # raw job id -> stats (target first; k2 can
                                         # read every own job's cgroup on the node)
    stray_nvidia_smi: int | None = None  # node.self: own nvidia-smi stuck after the sample (D state on two reads 1 s
                                         # apart: hung; >0 = leftover); None = no affirmative answer (cut, unreadable)
    cost_ms: int | None = None           # wall time of this node's hop (outer section ms)
    errors: tuple[HostError, ...] = ()   # node-level problems (section = outer section name)
    other_nvidia_smi: int | None = None  # node.self: the user's other, healthy nvidia-smi there (a job's
                                         # `nvidia-smi -l`, …): informational, never a leftover; None = unknown

@dataclass(frozen=True, slots=True)
class JobLive:                           # one Attach sample of one job (FleetAPI.attach_now, AttachState.live)
    key: JobKey
    mode: AttachMode
    sampled_at: datetime                 # when the login probe finished (local wall clock, UTC)
    nodes: tuple[NodeInspect, ...] = ()  # nodes that answered (possibly with node-level errors); () = no data
    cost_ms: int = 0                     # wall time of the whole login probe
    sacct_rows_added: int = 0            # permanent sacct rows THIS sample created (srun: steps launched; ssh: 0)
    clean: bool | None = None            # audit (§7.16 layer 5): True clean, False leftovers found, None not run or
                                         # inconclusive (e.g. a just-finished rfprobe step still listed)
    audit_detail: str = ""               # "clean" | "rfprobe step 31367626.3 still running" | "1 hung nvidia-smi on
                                         # s71-02" | "2 node ssh clients still running on the login node"
    hints: tuple[str, ...] = ()          # §7.16 hints, ready to show ("GPU1 idle (0%) but holding 85 GB")
    idle_streak: Mapping[str, int] = field(default_factory=dict)   # GPU uuid (else bus id) -> consecutive samples with
                                         # util < 5 % and mem > 1 GiB (carried over from `previous`)
    errors: tuple[HostError, ...] = ()   # sample-level problems (probe failed, audit failed, parse errors)

    @property
    def has_data(self) -> bool:
        return any(n.gpus or n.procs or n.cgroups for n in self.nodes)

class AttachStatus(StrEnum):
    BLOCKED = "blocked"                  # nothing can run now; AttachState.blocked says why (+ hint)
    NEEDS_CONSENT = "needs_consent"      # only srun can read this job: `r` asks for confirmation (AttachState.offer)
    SAMPLING = "sampling"                # a sample is in flight
    LIVE = "live"                        # ssh auto-refresh armed; next sample at AttachState.next_at
    IDLE = "idle"                        # no automatic sample pending (srun after a read, auto off): `r` reads once
    PAUSED = "paused"                    # auto-refresh suspended until the app is focused / input resumes polling
    STOPPED = "stopped"                  # auto-refresh ended (tab closed, 10 min without input, job ended, another job
                                         # attached, 3 failures); `r` or re-opening the tab starts again

@dataclass(frozen=True, slots=True)
class AttachOffer:                       # srun is possible behind a confirmation (texts ready for ConfirmScreen)
    node: str                            # node the srun step would run on
    reason: str                          # one line: "lyra refuses ssh to compute nodes" | "ssh landed in job 2332863"
    title: str                           # "Read job 31367626 with srun?"
    body: str                            # what runs + cost: "…leaves one permanent sacct row 31367626.N rfprobe…"
    preview: str                         # exact srun command line (AttachPlan.preview)

@dataclass(frozen=True, slots=True)
class AttachState:                       # FleetAPI.attach_state; one per job attached this session (kept after stop)
    key: JobKey
    status: AttachStatus
    setting: Literal["ssh", "ssh+srun", "srun", "off"]   # the host's effective `attach` config
    mode: AttachMode | None = None       # mode `r` uses without an explicit mode (None: blocked)
    live: JobLive | None = None          # last sample WITH data (kept across failed samples and stops)
    error: HostError | None = None       # last sample's failure; cleared by the next sample with data
    auto: bool = True                    # auto-refresh wanted (ssh only; `a` toggles); srun never auto-refreshes
    refresh_s: float | None = None       # the host's attach_refresh while mode == "ssh", else None
    next_at: datetime | None = None      # next automatic sample (None: none scheduled)
    consent: Literal["none", "job"] = "none"   # "job": srun allowed for this job for the rest of this session
    offer: AttachOffer | None = None     # srun available with confirmation (also while LIVE when ssh landed elsewhere)
    blocked: str | None = None           # why nothing can run (status BLOCKED): "attach is off for lyra", "read-only
                                         # mode: srun writes accounting", "job is not running", "lyra: not connected"
    hint: str | None = None              # how to enable more (attach="ssh" + node_ssh="denied": the config line + cost)
    stopped: str | None = None           # why PAUSED/STOPPED: "app in background", "polling paused (idle)", "no input
                                         # for 10 min", "tab closed", "job is no longer running", "another job attached"
    srun_rows: int = 0                   # sacct rows this session's srun samples added for this job
    node: str | None = None              # node pinned by attach_start(node=...); None = backend default
    failures: int = 0                    # consecutive failed samples (auto-refresh stops at 3)

# ---------- snapshot ----------
@dataclass(frozen=True, slots=True)
class HostSnapshot:
    host: str
    taken_at: datetime
    caps: Capabilities | None
    jobs: Mapping[str, Job]              # by JobKey.id
    gpus: tuple[GPU, ...]
    procs: tuple[Proc, ...]              # workstation process table: own task members + GPU pids of all users
    nodes: tuple[Node, ...]
    partitions: tuple[PartitionSummary, ...]
    metrics: HostMetrics | None

# ---------- actions, import, doctor ----------
@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str               # "slurm.cancel", "task.terminate", ...
    target: JobKey
    ok: bool
    message: str                 # "cancelled 2332861", "permission denied", "2 processes survived"
    details: tuple[str, ...] = ()   # per-pid / per-id lines ("sent 642234", "killed 786345", "new 790001")
    rc: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime | None = None
    duration_s: float | None = None

@dataclass(frozen=True, slots=True)
class DetectReport:              # FleetAPI.probe_alias (hosts import)
    alias: str
    state: ConnState             # OK, NEEDS_AUTH, HOSTKEY, UNREACHABLE, ...
    error: HostError | None
    caps: Capabilities | None    # caps.profile tells slurm vs workstation
    mux: MuxMode
    ssh_summary: Mapping[str, str]   # from `ssh -G`: user, hostname, port, proxyjump, controlpath, controlpersist
    latency_s: float | None

@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str                    # "ssh -G", "mux", "master", "probe", "profile", "tools", "clock", "gpu_tres"
    ok: bool | None              # None = informational
    detail: str

@dataclass(frozen=True, slots=True)
class DoctorReport:
    host: str
    checks: tuple[DoctorCheck, ...]
    warnings: tuple[str, ...] = ()

def to_json(obj: Any) -> Any:
    """Dataclass/enum/datetime/timedelta/frozenset -> JSON-compatible value (datetimes ISO 8601 UTC)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj.value if isinstance(obj, Enum) else obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        aware = obj if obj.tzinfo is not None else obj.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(to_json(k)): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((to_json(v) for v in obj), key=repr)    # deterministic output for caches and snapshots
    if isinstance(obj, (tuple, list)):
        return [to_json(v) for v in obj]
    raise TypeError(f"to_json: unsupported type {type(obj).__name__}")

def from_json(cls: type[T], data: Any) -> T:
    """Inverse of to_json for every dataclass in this module."""
    return typing.cast(T, _decode(cls, data))

@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)

def _decode(tp: Any, data: Any) -> Any:
    """Decode `data` (a to_json value) into type `tp` (a resolved annotation)."""
    if tp is Any:
        return data
    if tp is type(None):
        if data is not None:
            raise ValueError(f"expected null, got {data!r}")
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union or origin is types.UnionType:
        if data is None and type(None) in args:
            return None
        last: Exception | None = None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _decode(arg, data)
            except (TypeError, ValueError, KeyError) as e:
                last = e
        raise ValueError(f"no union member of {tp} accepts {data!r}") from last
    if origin is Literal:
        if data not in args:
            raise ValueError(f"{data!r} is not one of {args}")
        return data
    if origin in (tuple,):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v) for v in data)
        if not args:
            return tuple(data)
        if len(args) != len(data):
            raise ValueError(f"expected {len(args)} items, got {len(data)}")
        return tuple(_decode(a, v) for a, v in zip(args, data, strict=True))
    if origin in (frozenset, set):
        return frozenset(_decode(args[0], v) for v in data) if args else frozenset(data)
    if origin in (dict, Mapping):
        if not isinstance(data, Mapping):
            raise TypeError(f"expected an object, got {type(data).__name__}")
        kt, vt = args if args else (Any, Any)
        if kt is int:                             # JSON object keys are strings (NodeInspect.step_of_pid)
            return {_decode(int, int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _decode(vt, v)
                    for k, v in data.items()}
        return {_decode(kt, k): _decode(vt, v) for k, v in data.items()}
    if origin in (list,):
        return [_decode(args[0], v) for v in data] if args else list(data)
    if isinstance(tp, type):
        if issubclass(tp, Enum):
            return tp(data)
        if tp is datetime:
            if not isinstance(data, str):
                raise TypeError(f"expected an ISO timestamp, got {data!r}")
            dt = datetime.fromisoformat(data)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
        if tp is timedelta:
            try:
                return timedelta(seconds=float(data))
            except OverflowError as e:          # a damaged cache row must fail like any other bad value
                raise ValueError(f"duration out of range: {data!r}") from e
        if dataclasses.is_dataclass(tp):
            if not isinstance(data, Mapping):
                raise TypeError(f"expected an object for {tp.__name__}, got {type(data).__name__}")
            hints = _hints(tp)
            kwargs = {f.name: _decode(hints[f.name], data[f.name])
                      for f in dataclasses.fields(tp) if f.name in data}   # missing keys -> defaults
            return tp(**kwargs)
        if tp is bool:
            if not isinstance(data, bool):
                raise TypeError(f"expected a bool, got {data!r}")
            return data
        if tp is float:
            if isinstance(data, bool) or not isinstance(data, (int, float)):
                raise TypeError(f"expected a number, got {data!r}")
            return float(data)
        if tp is int:
            if isinstance(data, bool) or not isinstance(data, int):
                raise TypeError(f"expected an integer, got {data!r}")
            return data
        if tp is str:
            if not isinstance(data, str):
                raise TypeError(f"expected a string, got {data!r}")
            return data
    raise TypeError(f"from_json: unsupported type {tp!r}")
