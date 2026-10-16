"""[WP4] Simulated fleet ("default" scenario) with fault-injection hooks for tests (DESIGN.md §10.6).

The world is plain mutable state advanced on a fixed 5 s grid from its start time, so the same seed and the same
sequence of `sync()` targets always produce the same fleet. It knows nothing about output formats (`render.py`) or
the transport protocol (`transport.py`); it only answers "what is true on this host now" and applies actions.

Default scenario (mirrors the shapes of the field-probe fixtures in tests/fixtures/samples):

- ``k2``: Slurm 25.05, GPU TRES, PreemptMode REQUEUE, the PI's protected ``lab`` partition, a multi-node job, a
  stalled job, a pending job with an estimated start, history with a failed (Traceback) job, TIMEOUT, OOM, CANCELLED.
- ``vega``: Slurm 25.11, 4-running-job QOS cap, 2-GPU H200 jobs, ``mixed-`` nodes, a DependencyNeverSatisfied job.
- ``lyra``: Slurm 26.05, no GPU TRES, PreemptMode CANCEL, a throttled pending array, ProxyJump-shaped latency.
- ``labstation``: 4x RTX A6000, a queue-runner task, another user's GPU process, a task that crashes with a Traceback.
- ``lynx``: 2x RTX PRO 6000 Blackwell, a queue-runner task that starts its next item; starts NEEDS_AUTH until connect.
"""

from __future__ import annotations

import logging
import random
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Literal, TypeAlias

from runfleet.config import FleetConfig, HostConfig, SlurmSettings

log = logging.getLogger("runfleet.demo.world")

DEFAULT_SEED = 7
STEP_S = 5.0                     # simulation grid
COARSE_STEP_S = 60.0             # used for gaps longer than an hour (laptop slept), keeps catch-up cheap
FAULTS: tuple[str, ...] = ("unreachable", "auth", "nvidia_hang", "garbage", "slurmdbd_down")
Fault = Literal["unreachable", "auth", "nvidia_hang", "garbage", "slurmdbd_down"]

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "CANCELLED",
                             "NODE_FAIL", "BOOT_FAIL", "DEADLINE"})
MIN_JOB_AGE_S = 300              # scontrol forgets a job this long after it ended (field probes: MinJobAge=300)


def stable_hash(text: str) -> int:
    """Process-independent hash (Python's hash() is salted per process)."""
    return zlib.crc32(text.encode())


def local_ts(epoch: float, tz: timedelta, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.fromtimestamp(epoch, timezone(tz)).strftime(fmt)


# ====================================================================================================== logs
class SimLog:
    """An append-only file on a simulated host. Keeps at most ~4 MiB in memory; `size` stays the true file size."""

    MAX_BYTES = 4 << 20

    def __init__(self, path: str, created: float) -> None:
        self.path = path
        self.created = created
        self.mtime = created
        self._buf = bytearray()
        self._base = 0                  # bytes dropped from the head

    @property
    def size(self) -> int:
        return self._base + len(self._buf)

    def write(self, text: str, at: float) -> None:
        if not text:
            return
        self._buf += text.encode("utf-8")
        self.mtime = max(self.mtime, at)
        if len(self._buf) > self.MAX_BYTES:
            drop = len(self._buf) - self.MAX_BYTES * 3 // 4
            del self._buf[:drop]
            self._base += drop

    def tail(self, nbytes: int) -> bytes:
        return bytes(self._buf[-nbytes:]) if nbytes > 0 else b""

    def read_from(self, offset: int) -> bytes:
        """Bytes written after absolute offset `offset` (what `tail -F` delivers next)."""
        return bytes(self._buf[max(0, offset - self._base):])

    def text(self) -> str:
        return self._buf.decode("utf-8", errors="replace")


_BAR_PARTS = " ▏▎▍▌▋▊▉"


def tqdm_bar(done: int, total: int, width: int = 10) -> str:
    frac = 0.0 if total <= 0 else min(1.0, done / total)
    eighths = int(frac * width * 8)
    full, part = divmod(eighths, 8)
    bar = "█" * full + (_BAR_PARTS[part] if full < width and part else "")
    return bar.ljust(width)


def _mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class Trainer:
    """Writes a training log the way PyTorch scripts do: tqdm progress with carriage returns, then a summary line
    per epoch. `advance` returns the text to append for `dt` seconds of progress."""

    epochs: int
    epoch_s: float                   # wall time per epoch
    iters: int                       # iterations per epoch
    epoch: int = 1
    into: float = 0.0                # seconds into the current epoch
    loss: float = 1.2
    metric: float = 0.0              # vs_base (%), drifts negative = better
    label: str = "epoch"

    @property
    def rate(self) -> float:
        return self.iters / self.epoch_s

    def progress_line(self) -> str:
        done = min(self.iters, int(self.into / self.epoch_s * self.iters))
        pct = int(100 * done / self.iters)
        left = (self.iters - done) / self.rate
        return (f"{self.label} {self.epoch}/{self.epochs}: {pct:3d}%|{tqdm_bar(done, self.iters)}| "
                f"{done}/{self.iters} [{_mmss(self.into)}<{_mmss(left)}, {self.rate:.0f} it/s]")

    def summary_line(self, stamp: str, rng: random.Random) -> str:
        self.loss = max(0.05, self.loss * (0.985 + rng.random() * 0.01))
        self.metric -= rng.random() * 0.06
        return (f"[{stamp}] epoch {self.epoch}/{self.epochs}  train_loss {self.loss:.4f}  "
                f"vs_base {self.metric:+.2f}%  {self.rate * (0.97 + rng.random() * 0.06):.0f} it/s\n")

    def advance(self, dt: float, now: float, tz: timedelta, rng: random.Random) -> str:
        out: list[str] = []
        self.into += dt
        while self.into >= self.epoch_s and self.epoch <= self.epochs:
            self.into -= self.epoch_s
            out.append("\r" + self._final_bar() + "\n")
            out.append(self.summary_line(local_ts(now - self.into, tz), rng))
            self.epoch += 1
        if self.epoch <= self.epochs:
            out.append("\r" + self.progress_line())
        return "".join(out)

    def _final_bar(self) -> str:
        return (f"{self.label} {self.epoch}/{self.epochs}: 100%|{tqdm_bar(1, 1)}| {self.iters}/{self.iters} "
                f"[{_mmss(self.epoch_s)}<00:00, {self.rate:.0f} it/s]")

    def history(self, started: float, now: float, tz: timedelta, rng: random.Random, keep: int = 120) -> str:
        """Log content for a run that started at `started` and is still running at `now`."""
        done_epochs = min(self.epochs, int((now - started) / self.epoch_s))
        out: list[str] = []
        first = max(1, done_epochs - keep + 1)
        self.metric = -0.004 * first
        for e in range(first, done_epochs + 1):
            self.epoch = e
            self.into = 0.0
            out.append("\r" + self.progress_line() + "\r" + self._final_bar() + "\n")
            out.append(self.summary_line(local_ts(started + e * self.epoch_s, tz), rng))
        self.epoch = done_epochs + 1
        self.into = (now - started) - done_epochs * self.epoch_s
        if self.epoch <= self.epochs:
            out.append("\r" + self.progress_line())
        return "".join(out)


TRACEBACKS: dict[str, str] = {
    "cuda_oom": (
        "Traceback (most recent call last):\n"
        '  File "{root}/src/{pkg}/train.py", line 412, in <module>\n'
        "    main()\n"
        '  File "{root}/src/{pkg}/train.py", line 388, in main\n'
        "    loss = trainer.step(batch)\n"
        '  File "{root}/src/{pkg}/trainer.py", line 214, in step\n'
        "    loss.backward()\n"
        '  File "{root}/.venv/lib/python3.11/site-packages/torch/_tensor.py", line 626, in backward\n'
        "    torch.autograd.backward(\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of "
        "{gpu_gib} GiB of which 1.12 GiB is free. Including non-PyTorch memory, this process has {used_gib} GiB "
        "memory in use.\n"),
    "value": (
        "Traceback (most recent call last):\n"
        '  File "{root}/src/{pkg}/train.py", line 412, in <module>\n'
        "    main()\n"
        '  File "{root}/src/{pkg}/train.py", line 301, in main\n'
        "    cfg = load_config(args.config, overrides=args.overrides)\n"
        '  File "{root}/src/{pkg}/config.py", line 88, in load_config\n'
        "    raise ValueError(f\"unknown key {{key!r}} in {{path}}\")\n"
        "ValueError: unknown key 'optim.lr_shedule' in configs/exp_a.yaml\n"),
    "missing_ckpt": (
        "Traceback (most recent call last):\n"
        '  File "{root}/scripts/eval_ckpt.py", line 57, in <module>\n'
        "    state = torch.load(args.ckpt, map_location=\"cuda\")\n"
        '  File "{root}/.venv/lib/python3.11/site-packages/torch/serialization.py", line 1484, in load\n'
        "    with _open_file_like(f, \"rb\") as opened_file:\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'runs/voc/baseline-const/s44/"
        "best.pt'\n"),
    "nccl": (       # rank 1 waits on an all-reduce rank 0 never joins: the NCCL watchdog aborts (cause chip NCCL)
        "[rank1]:[E0906 22:55:21.512 ProcessGroupNCCL.cpp:616] [Rank 1] Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=48211, OpType=ALLREDUCE, NumelIn=26214400, NumelOut=26214400, Timeout(ms)=600000) ran for "
        "600043 milliseconds before timing out.\n"
        "[rank1]:[E0906 22:55:21.530 ProcessGroupNCCL.cpp:1785] [PG ID 0 PG GUID 0(default_pg) Rank 1] Exception "
        "(either an error before or timeout detected by watchdog)\n"
        "Traceback (most recent call last):\n"
        '  File "{root}/src/{pkg}/train_ddp.py", line 233, in <module>\n'
        "    main()\n"
        '  File "{root}/src/{pkg}/train_ddp.py", line 198, in main\n'
        "    dist.all_reduce(grads)\n"
        "torch.distributed.DistBackendError: NCCL communicator was aborted on rank 1.\n"
        "E0906 22:55:28.884 torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: -6) local_rank: 1 "
        "(pid: 2210894) of binary: {root}/.venv/bin/python\n"),
    "keyboard": (
        "Traceback (most recent call last):\n"
        '  File "<frozen runpy>", line 198, in _run_module_as_main\n'
        '  File "{root}/src/{pkg}/launch.py", line 141, in main\n'
        "    time.sleep(5)\n"
        "KeyboardInterrupt\n"),
}


def traceback_text(kind: str, root: str, pkg: str = "orb", gpu_gib: str = "39.38", used_gib: str = "37.91") -> str:
    return TRACEBACKS[kind].format(root=root, pkg=pkg, gpu_gib=gpu_gib, used_gib=used_gib)


# ====================================================================================================== slurm
@dataclass
class GresDef:
    type: str                        # as configured: "nvidia_h200_nvl", "V100", "A30"
    count: int
    sockets: str | None = None       # "(S:0-7)" suffix in sinfo Gres; None = no suffix (lyra)


@dataclass
class SimNode:
    name: str
    partitions: tuple[str, ...]
    cpus: int
    mem_mib: int
    gres: tuple[GresDef, ...] = ()
    state: str | None = None         # fixed base state ("planned", "reserved", "drained", "down"); None = derived
    flag: str = ""                   # "-" planned by backfill, "*" not responding, "~" powered off
    reason: str = "none"
    bg_cpus: int = 0                 # CPUs / memory / GPU indices held by other users' jobs
    bg_mem_mib: int = 0
    bg_gpu_idx: set[int] = field(default_factory=set)
    free_mem_mib: int = 20000        # sinfo FreeMem (OS view, unrelated to allocations)

    @property
    def gpu_total(self) -> int:
        return sum(g.count for g in self.gres)

    def type_ranges(self) -> list[tuple[GresDef, range]]:
        """Global GPU index range of every gres type (types get consecutive indices)."""
        out: list[tuple[GresDef, range]] = []
        start = 0
        for g in self.gres:
            out.append((g, range(start, start + g.count)))
            start += g.count
        return out


@dataclass
class StepUsage:
    """What sstat/sacct report for one step. GPU util is summed over a task's GPUs (Appendix C #1)."""

    ntasks: int = 1
    ave_cpu_s: float = 0.0
    ave_rss_kb: int = 0
    max_rss_kb: int = 0
    disk_bytes: int = 0
    pages: int = 0
    gpuutil_ave: float = 0.0         # per task
    gpuutil_max: float = 0.0
    gpumem_ave_mib: int = 0          # per task
    gpumem_max_mib: int = 0


@dataclass
class SimJob:
    job_id: int                      # JobIDRaw (array element: its own id; pending summary: the array job id)
    name: str
    user: str
    uid: int
    account: str
    partition: str
    qos: str
    submit: float
    time_limit_s: int | None         # None = UNLIMITED
    cpus: int                        # total over all nodes
    mem_mib: int                     # per node
    workdir: str
    command: str
    stdout_pattern: str
    stderr_pattern: str | None = None    # None = merged into stdout ("(null)" in squeue)
    num_nodes: int = 1
    gpus_per_node: int = 0
    gpu_type: str | None = None      # configured gres type; None = any
    typed_request: bool = True       # tres-per-node "gres/gpu:<type>:n" vs "gres/gpu:n"
    typed_alloc: bool = True         # tres-alloc carries "gres/gpu:<type>=n" (k2 3371310 does not)
    typed_req_tres: bool | None = None   # sacct ReqTRES typed; None = like typed_request (3371310: typed --gres,
                                         # untyped ReqTRES)
    alloc_gpu_type: str | None = None    # gres type of the allocated GPUs (JOB_GRES / per-node GRES)
    state: str = "PENDING"           # Slurm long state
    reason: str = "Priority"
    start: float | None = None       # actual start; for PENDING the scheduler estimate (None = N/A)
    end: float | None = None
    duration_s: float = 3600.0       # planned run time once started
    outcome: str = "COMPLETED"       # final state of the run
    exit_code: int = 0
    exit_signal: int = 0
    failure: str = "cuda_oom"        # traceback kind for FAILED outcomes
    nodes: tuple[str, ...] = ()
    gpu_idx: dict[str, tuple[int, ...]] = field(default_factory=dict)
    cpu_ids: dict[str, str] = field(default_factory=dict)
    array_job_id: int | None = None
    array_task: str | None = None    # element "3"; pending summary "5-12%2"
    summary: bool = False            # the pending meta record of an array ("A_[5-12%2]")
    array_pending: list[int] = field(default_factory=list)   # summary only: tasks not yet started
    array_throttle: int = 0
    restarts: int = 0
    held: bool = False
    dependency: str = "(null)"
    never_runs: bool = False         # DependencyNeverSatisfied
    cancelled_by: int | None = None
    stalled: bool = False            # log idle, GPU idle
    step_tasks: int = 0              # > 0: an srun/torchrun step ".0" with this many tasks
    cg_until: float | None = None    # COMPLETING until (then the final state)
    batch: StepUsage = field(default_factory=StepUsage)
    step0: StepUsage | None = None
    trainer: Trainer | None = None
    log_paths: tuple[str, ...] = ()  # expanded stdout (+ stderr when not merged)
    scontrol_extra: dict[str, str] = field(default_factory=dict)   # per-job scontrol fields (Priority, AllocNode:Sid…)
    submit_args: str = ""            # trailing script args in SubmitLine
    util_target: float = 95.0        # per GPU, percent
    mem_target_mib: int = 20000      # per GPU
    banner: str = ""                 # extra start-up log lines (a TensorBoard URL: `P` forwards it)
    idle_gpus: dict[int, int] = field(default_factory=dict)   # Attach: position in the job's GPU list on a node ->
                                     # MiB that GPU holds at 0 % (vega: GPU1 idle holding 85 GB, §7.16.8)
    shmem_mib: int = 0               # Attach: shared memory in the job's cgroup (vLLM, DataLoader workers)

    @property
    def is_summary(self) -> bool:
        return self.summary

    @property
    def gpus(self) -> int:
        return self.gpus_per_node * self.num_nodes

    @property
    def display_id(self) -> str:
        if self.array_job_id is None or self.array_task is None:
            return str(self.job_id)
        if self.is_summary:
            return f"{self.array_job_id}_[{self.array_task}]"
        return f"{self.array_job_id}_{self.array_task}"

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def started(self) -> bool:
        return self.state not in ("PENDING", "REQUEUED") and self.start is not None and (
            self.state != "CANCELLED" or bool(self.nodes))

    def elapsed(self, now: float) -> float:
        if not self.started or self.start is None:
            return 0.0
        return max(0.0, (self.end if self.end is not None else now) - self.start)


@dataclass
class SimCluster:
    name: str
    kind: Literal["slurm"] = "slurm"
    slurm_version: str = "25.05.8"
    login_host: str = "login"
    user: str = "user"
    uid: int = 1000
    group: str = "users"
    gid: int = 100
    tz: timedelta = timedelta(0)
    config: dict[str, str] = field(default_factory=dict)        # scontrol show config subset (in scontrol order)
    associations: tuple[tuple[str, str, str, str], ...] = ()
    tools: frozenset[str] = frozenset()
    nodes: list[SimNode] = field(default_factory=list)
    jobs: dict[int, SimJob] = field(default_factory=dict)
    next_job_id: int = 1
    gpu_tres: bool = True
    merged_stderr: Literal["repeat", "empty"] = "repeat"         # scontrol StdErr of a merged job
    stat_verb: Literal["stat", "statx"] = "statx"                # coreutils 8.30 says "cannot stat"
    qos_max_running: int | None = None                           # per-user running cap of QOS "normal"
    latency: tuple[float, float] = (0.05, 0.12)
    ncpu: int = 32                                               # login node
    mem_total_kb: int = 196_000_000
    load: float = 2.5
    logs: dict[str, SimLog] = field(default_factory=dict)
    submit_templates: tuple[tuple[str, str, int, str], ...] = ()  # (name, partition, gpus, gpu type) for new jobs
    next_submit: float = 0.0
    cpu_jiffies: list[int] = field(default_factory=lambda: [0] * 10)
    boot: float = 0.0
    node_ssh: Literal["adopt", "denied"] = "adopt"   # login → compute-node ssh: pam_slurm_adopt, or refused (lyra)
    gpu_buses: tuple[str, ...] = ("01", "41", "81", "c1", "21", "61", "a1", "e1")   # PCI bus of GPU IDX 0, 1, …
    attach_steps: dict[int, int] = field(default_factory=dict)   # job id -> rfprobe steps created this session
    # pending priority (§5.2.10): `sshare -U` rows of the user (Account, RawShares, NormShares, RawUsage, EffectvUsage,
    # FairShare, LevelFS) and the normalized partition factor sprio weights by PriorityWeightPartition
    shares: tuple[tuple[str, str, str, str, str, str, str], ...] = ()
    partition_tier: dict[str, float] = field(default_factory=dict)

    @property
    def preempt_mode(self) -> str:
        return self.config.get("PreemptMode", "OFF")

    def node(self, name: str) -> SimNode | None:
        return next((n for n in self.nodes if n.name == name), None)

    def live_jobs(self) -> list[SimJob]:
        return [j for j in self.jobs.values() if not j.terminal]

    def find(self, raw_or_display: str) -> list[SimJob]:
        """Jobs addressed by an id as scancel/scontrol accept it: raw id, array job id (all elements + summary),
        `A_T` element or `A_[range]` (summary and elements whose task is in range; throttle stripped)."""
        text = raw_or_display.strip()
        if "_" not in text:
            if not text.isdigit():
                return []
            num = int(text)
            exact = self.jobs.get(num)
            members = [j for j in self.jobs.values() if j.array_job_id == num]
            return members if members else ([exact] if exact else [])
        base, _, task = text.partition("_")
        if not base.isdigit():
            return []
        arr = int(base)
        if task.startswith("["):
            wanted = set(expand_ranges(task.strip("[]").split("%")[0]))
            out = [j for j in self.jobs.values() if j.array_job_id == arr and (
                j.is_summary and wanted & set(j.array_pending)
                or (not j.is_summary and j.array_task is not None and j.array_task.isdigit()
                    and int(j.array_task) in wanted))]
            return out
        return [j for j in self.jobs.values()
                if j.array_job_id == arr and not j.is_summary and j.array_task == task]


def expand_ranges(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def compress_ranges(values: Iterable[int]) -> str:
    vals = sorted(set(values))
    parts: list[str] = []
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[j] + 1:
            j += 1
        parts.append(str(vals[i]) if i == j else f"{vals[i]}-{vals[j]}")
        i = j + 1
    return ",".join(parts)


# ====================================================================================================== workstation
@dataclass
class SimGPU:
    index: int
    uuid: str
    name: str
    mem_total_mib: int
    power_limit_w: float
    idle_power_w: float = 22.0
    fan: int | None = 30             # None -> "[N/A]" (passively cooled boards)
    util: float = 0.0
    mem_util: float = 0.0
    mem_used_mib: int = 1
    temp_c: float = 30.0
    power_w: float = 22.0
    clocks_sm: int = 210

    @property
    def pstate(self) -> str:
        return "P2" if self.util > 5 or self.mem_used_mib > 500 else "P8"


@dataclass
class SimProc:
    pid: int
    ppid: int
    pgid: int
    user: str
    start: int                       # epoch seconds (what `now - etimes` gives)
    args: str
    stat: str = "S"
    cpu_pct: float = 0.0
    mem_pct: float = 0.0
    rss_kb: int = 2048
    cwd: str = "/"
    fd1: str = "/dev/null"
    fd2: str = "/dev/null"
    env: dict[str, str] = field(default_factory=dict)
    gpu: int | None = None
    gpu_mem_mib: int = 0
    gpu_util: float = 0.0
    end_at: float | None = None      # planned exit
    crash: str | None = None         # traceback kind written at exit
    trainer: Trainer | None = None
    item: str | None = None          # manifest item (queue-runner child)


@dataclass
class QueueRunner:
    """A launcher (`python -m orb.launch --manifest … --per-gpu 1`) that starts the next item when a child exits."""

    root: int
    items: list[str]                 # remaining manifest items
    next_no: int = 1                 # 1-based manifest position of the next item
    total: int = 0
    pending: list[tuple[float, int]] = field(default_factory=list)   # (start at, gpu)
    child_s: tuple[float, float] = (1800.0, 3600.0)


@dataclass
class SimWorkstation:
    name: str
    kind: Literal["workstation"] = "workstation"
    hostname: str = "ws"
    user: str = "user"
    uid: int = 1000
    tz: timedelta = timedelta(0)
    ncpu: int = 64
    mem_total_kb: int = 528_008_252
    swap_total_kb: int = 2_097_148
    swap_free_kb: int = 2_061_820
    driver: str = "610.43.02"
    nvml: str = "610.43"
    cuda: str = "13.3"
    tools: frozenset[str] = frozenset()
    gpus: list[SimGPU] = field(default_factory=list)
    procs: dict[int, SimProc] = field(default_factory=dict)
    runners: dict[int, QueueRunner] = field(default_factory=dict)
    next_pid: int = 800000
    latency: tuple[float, float] = (0.05, 0.1)
    load: float = 1.0
    bg_cpu_pct: float = 150.0        # system + other users' CPU use, percent of one core
    cpu_jiffies: list[int] = field(default_factory=lambda: [0] * 10)
    boot: float = 0.0
    mem_used_kb: int = 15_000_000
    logs: dict[str, SimLog] = field(default_factory=dict)
    project_root: str = "/home/user/projects/orb"

    def children(self, pid: int) -> list[SimProc]:
        return sorted((p for p in self.procs.values() if p.ppid == pid), key=lambda p: p.pid)

    def descendants(self, pid: int) -> list[int]:
        """BFS over the current process table (what the §6.2 `tree` helper prints)."""
        out: list[int] = []
        queue = [pid]
        while queue:
            cur = queue.pop(0)
            for child in self.children(cur):
                if child.pid not in out:
                    out.append(child.pid)
                    queue.append(child.pid)
        return out


SimHost: TypeAlias = SimCluster | SimWorkstation


# ====================================================================================================== world
class DemoWorld:
    """The simulated fleet. `now` is the time source (aware UTC); defaults to the wall clock."""

    def __init__(self, seed: int = DEFAULT_SEED, now: Callable[[], datetime] | None = None, *,
                 latency_scale: float = 1.0, stream_poll_s: float = 0.5, scenario: str = "default") -> None:
        if scenario != "default":
            raise ValueError(f"unknown demo scenario {scenario!r} (v0.1 has only 'default')")
        self.seed = seed
        self._now = now or (lambda: datetime.now(UTC))
        self.latency_scale = latency_scale   # 0 = instant probes (tests, snapshots)
        self.stream_poll_s = stream_poll_s   # how often a log.tail stream looks for new bytes (real seconds)
        self.rng = random.Random(seed)
        self.t0 = float(int(self._now().timestamp()))
        self.t = self.t0                 # simulated time (on the STEP_S grid)
        self.hosts: dict[str, SimHost] = {}
        self.faults: dict[str, set[str]] = {}
        self.configs: dict[str, HostConfig] = {}   # HostConfig per demo host, in fleet order
        self.events: list[str] = []      # human-readable log of world changes (tests, debugging)
        self.nvidia_hangs: dict[str, int] = {}
        _build_default(self)

    # ------------------------------------------------------------------ time
    def now(self) -> float:
        return self._now().timestamp()

    def sync(self) -> float:
        """Advance the simulation to the time source's `now` (fixed grid steps) and return it (epoch seconds)."""
        target = self.now()
        while self.t + STEP_S <= target:
            dt = COARSE_STEP_S if target - self.t > 3600 else STEP_S
            dt = min(dt, max(STEP_S, (target - self.t) // STEP_S * STEP_S))
            self.t += dt
            self._step(dt)
        return max(target, self.t)

    def _step(self, dt: float) -> None:
        for host in self.hosts.values():
            if isinstance(host, SimCluster):
                _step_cluster(self, host, dt)
            else:
                _step_workstation(self, host, dt)

    def note(self, msg: str) -> None:
        self.events.append(f"{local_ts(self.t, timedelta(0), '%H:%M:%S')} {msg}")
        log.debug("demo: %s", msg)

    # ------------------------------------------------------------------ hosts and config
    def host(self, name: str) -> SimHost | None:
        return self.hosts.get(name)

    def cluster(self, name: str) -> SimCluster:
        h = self.hosts[name]
        assert isinstance(h, SimCluster), name
        return h

    def workstation(self, name: str) -> SimWorkstation:
        h = self.hosts[name]
        assert isinstance(h, SimWorkstation), name
        return h

    def host_configs(self) -> tuple[HostConfig, ...]:
        return tuple(self.configs[name] for name in self.hosts)

    def fleet_config(self, hosts: Sequence[str] = ()) -> FleetConfig:
        """FleetConfig of the demo fleet (optionally only `hosts`, in the given order). Unknown names raise
        KeyError with the list of demo hosts."""
        cfgs = {c.name: c for c in self.host_configs()}
        if hosts:
            unknown = [h for h in hosts if h not in cfgs]
            if unknown:
                raise KeyError(f"not a demo host: {', '.join(unknown)} (demo hosts: {', '.join(cfgs)})")
            return FleetConfig(hosts=tuple(cfgs[h] for h in hosts))
        return FleetConfig(hosts=tuple(cfgs.values()))

    def logs_of(self, host: str) -> dict[str, SimLog]:
        h = self.hosts.get(host)
        return h.logs if h is not None else {}

    # ------------------------------------------------------------------ faults (tests and demos)
    def inject(self, host: str, fault: str) -> None:
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; known: {', '.join(FAULTS)}")
        if host not in self.hosts:
            raise KeyError(host)
        self.faults.setdefault(host, set()).add(fault)
        self.note(f"{host}: fault {fault} injected")

    def clear(self, host: str, fault: str | None = None) -> None:
        current = self.faults.get(host, set())
        if fault is None:
            current.clear()
        else:
            current.discard(fault)
        if fault in (None, "nvidia_hang"):
            self.nvidia_hangs.pop(host, None)
        self.note(f"{host}: fault {fault or 'all'} cleared")

    def has_fault(self, host: str, fault: str) -> bool:
        return fault in self.faults.get(host, ())

    # ------------------------------------------------------------------ Slurm actions
    def slurm_action(self, host: str, verb: str, ids: Sequence[str], *, hold: bool = False) -> tuple[int, str]:
        """Apply scancel / scontrol hold|release|requeue[hold] to display ids; returns (rc, stderr)."""
        c = self.cluster(host)
        now = self.t
        errors: list[str] = []
        for raw in ids:
            targets = c.find(raw)
            if "[" in raw:                      # a task range addresses the tasks still alive, not finished ones
                targets = [j for j in targets if not j.terminal] or targets
            if not targets:
                errors.append(_slurm_err(verb, raw, "Invalid job id specified"))
                continue
            for job in targets:
                if verb == "cancel" and job.is_summary and "[" in raw:
                    wanted = set(expand_ranges(raw.partition("[")[2].rstrip("]").split("%")[0]))
                    if set(job.array_pending) - wanted:
                        _cancel_array_tasks(self, c, job, sorted(wanted & set(job.array_pending)), now)
                        continue
                err = _apply_slurm_verb(self, c, job, verb, now, hold=hold)
                if err:
                    errors.append(_slurm_err(verb, raw, err))
        return (1 if errors else 0), "\n".join(errors)

    # ------------------------------------------------------------------ task signals (§6.2 semantics)
    def signal_task(self, host: str, root: int, start: int, sig: str, grace_s: int, escalate: bool) -> list[str]:
        ws = self.workstation(host)
        proc = ws.procs.get(root)
        if proc is None:
            return [f"gone {root}", "done"]
        if abs(proc.start - start) > 5:
            return [f"reused {root}", "done"]
        members = ws.descendants(root)
        set_text = "".join(f"{p} " for p in members)
        lines = [f"members {root} {set_text}"]
        lines += [f"sent {p}" for p in [*members, root] if p in ws.procs]
        if sig == "INT":
            for pid in [root, *members]:
                p = ws.procs.get(pid)
                if p is not None and "python" in p.args and pid == root:
                    _write_proc_log(ws, p, traceback_text("keyboard", ws.project_root), self.t)
        for pid in [*members, root]:
            _exit_proc(self, ws, pid, reason=f"SIG{sig}")
        ws.runners.pop(root, None)
        lines.append("done")
        self.note(f"{host}: task {root} signalled {sig} ({len(members) + 1} processes)")
        return lines


def _slurm_err(verb: str, ident: str, reason: str) -> str:
    if verb == "cancel":
        return f"scancel: error: Kill job error on job id {ident}: {reason}"
    return f"{verb} {ident}: {reason}" if reason.startswith("Job ") else f"slurm_update error: {reason}"


def _apply_slurm_verb(world: DemoWorld, c: SimCluster, job: SimJob, verb: str, now: float,
                      *, hold: bool) -> str | None:
    if verb == "cancel":
        if job.terminal or job.state == "COMPLETING":
            return "Job/step already completing or completed"
        _finish_job(world, c, job, "CANCELLED", now, cancelled_by=c.uid)
        return None
    if verb == "hold":
        if job.state != "PENDING":
            return "Job is no longer pending execution"
        job.held, job.reason = True, "JobHeldUser"
        world.note(f"{c.name}: {job.display_id} held")
        return None
    if verb == "release":
        if not job.held:
            return "Job is not held"
        job.held, job.reason = False, "Priority"
        world.note(f"{c.name}: {job.display_id} released")
        return None
    if verb == "requeue":
        if job.terminal and (job.end is None or now - job.end > MIN_JOB_AGE_S):
            return "Invalid job id specified"
        if job.state == "PENDING":
            return "Job is pending execution"
        if job.state in ("RUNNING", "SUSPENDED"):
            _write_job_log(c, job, f"slurmstepd: error: *** JOB {job.job_id} ON {job.nodes[0]} CANCELLED AT "
                                   f"{local_ts(now, c.tz, '%Y-%m-%dT%H:%M:%S')} DUE TO JOB REQUEUE ***\n", now)
        _requeue(c, job, now, held=hold)
        world.note(f"{c.name}: {job.display_id} requeued{' (held)' if hold else ''}")
        return None
    return f"unsupported action {verb}"


# ====================================================================================================== slurm sim
def _write_job_log(c: SimCluster, job: SimJob, text: str, at: float) -> None:
    for path in job.log_paths[:1]:
        lg = c.logs.get(path)
        if lg is None:
            lg = c.logs[path] = SimLog(path, at)
        lg.write(text, at)


def expand_pattern(pattern: str, job: SimJob, user: str) -> str:
    """Slurm filename pattern expansion (%x %j %A %a %u %N %%) as the controller does for scontrol StdOut."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch != "%" or i + 1 >= len(pattern):
            out.append(ch)
            i += 1
            continue
        nxt = pattern[i + 1]
        arr = job.array_job_id if job.array_job_id is not None else job.job_id
        table = {"%": "%", "x": job.name, "j": str(job.job_id), "J": str(job.job_id), "A": str(arr),
                 "u": user, "N": job.nodes[0] if job.nodes else "%N", "n": "0", "s": "batch", "t": "0"}
        if nxt == "a":
            out.append(job.array_task if job.array_task and job.array_task.isdigit() else "%a")   # summary: unknown
        else:
            out.append(table.get(nxt, "%" + nxt))
        i += 2
    path = "".join(out)
    return path if path.startswith("/") else f"{job.workdir.rstrip('/')}/{path}"


def _set_log_paths(c: SimCluster, job: SimJob) -> None:
    paths = [expand_pattern(job.stdout_pattern, job, c.user)]
    if job.stderr_pattern is not None:
        err = expand_pattern(job.stderr_pattern, job, c.user)
        if err != paths[0]:
            paths.append(err)
    job.log_paths = tuple(paths)


def free_gpus(c: SimCluster, node: SimNode, gpu_type: str | None) -> list[int]:
    used = set(node.bg_gpu_idx)
    for j in c.jobs.values():
        if j.state in ("RUNNING", "COMPLETING", "SUSPENDED") and node.name in j.gpu_idx:
            used |= set(j.gpu_idx[node.name])
    free: list[int] = []
    for g, rng in node.type_ranges():
        if gpu_type is None or g.type == gpu_type:
            free += [i for i in rng if i not in used]
    return free


def _type_of(node: SimNode, idx: int) -> str | None:
    return next((g.type for g, rng in node.type_ranges() if idx in rng), None)


def gpu_type_at(node: SimNode, idx: int) -> str | None:
    """The gres type of GPU IDX `idx` on `node`."""
    return _type_of(node, idx)


def _usable(node: SimNode) -> bool:
    return node.state not in ("reserved", "drained", "down") and node.flag not in ("*", "~")


def node_cpus_used(c: SimCluster, node: SimNode) -> int:
    return node.bg_cpus + sum(j.cpus // max(1, j.num_nodes) for j in c.jobs.values()
                              if j.state in ("RUNNING", "COMPLETING") and node.name in j.nodes)


def _allocate(c: SimCluster, job: SimJob) -> bool:
    """Pick nodes for `job` (first fit in its partition); False if nothing fits now."""
    chosen: list[tuple[SimNode, list[int]]] = []
    per_node_cpus = job.cpus // max(1, job.num_nodes)
    for node in c.nodes:
        if job.partition not in node.partitions or not _usable(node):
            continue
        if node.cpus - node_cpus_used(c, node) < per_node_cpus:
            continue
        free = free_gpus(c, node, job.gpu_type)
        if len(free) < job.gpus_per_node:
            continue
        chosen.append((node, free[:job.gpus_per_node]))
        if len(chosen) == job.num_nodes:
            break
    if len(chosen) < job.num_nodes:
        return False
    job.nodes = tuple(n.name for n, _ in chosen)
    job.gpu_idx = {n.name: tuple(idx) for n, idx in chosen if idx}
    job.alloc_gpu_type = _type_of(chosen[0][0], chosen[0][1][0]) if job.gpus_per_node else None
    job.cpu_ids = {}
    for n, _ in chosen:
        first = node_cpus_used(c, n)
        job.cpu_ids[n.name] = f"{first}-{first + per_node_cpus - 1}"
    return True


def _start_job(world: DemoWorld, c: SimCluster, job: SimJob, now: float) -> bool:
    if not _allocate(c, job):
        job.reason = "Resources"
        return False
    job.state, job.reason, job.start, job.end = "RUNNING", "None", now, None
    _set_log_paths(c, job)
    job.batch = StepUsage()
    job.step0 = StepUsage(ntasks=job.step_tasks) if job.step_tasks else None
    job.trainer = Trainer(epochs=300, epoch_s=max(30.0, job.duration_s / 300), iters=4500)
    _write_job_log(c, job, _job_header(c, job, now), now)
    world.note(f"{c.name}: {job.display_id} started on {','.join(job.nodes)}")
    return True


def _job_header(c: SimCluster, job: SimJob, at: float) -> str:
    stamp = local_ts(at, c.tz)
    vis = ",".join(str(i) for i in range(job.gpus_per_node)) or "none"
    return (f"[{stamp}] host {job.nodes[0] if job.nodes else '?'}  CUDA_VISIBLE_DEVICES={vis}  torch 2.9.1+cu130\n"
            f"[{stamp}] job {job.job_id} · {job.name} · workdir {job.workdir}\n"
            + "".join(f"[{stamp}] {line}\n" for line in job.banner.splitlines()))


def _finish_job(world: DemoWorld, c: SimCluster, job: SimJob, state: str, now: float,
                *, cancelled_by: int | None = None) -> None:
    """Move a job to a terminal state, writing what Slurm and the program would have written to its log."""
    stamp = local_ts(now, c.tz, "%Y-%m-%dT%H:%M:%S")
    node = job.nodes[0] if job.nodes else "?"
    was_running = job.state in ("RUNNING", "COMPLETING", "SUSPENDED")
    if job.is_summary:                       # cancelling a pending array summary cancels its pending tasks
        job.array_pending = []
    if state == "CANCELLED" and was_running:
        _write_job_log(c, job, f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT {stamp} ***\n", now)
    elif state == "TIMEOUT":
        _write_job_log(c, job, f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT {stamp} DUE TO "
                               "TIME LIMIT ***\n", now)
    elif state == "PREEMPTED":
        _write_job_log(c, job, f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT {stamp} DUE TO "
                               "PREEMPTION ***\n", now)
    elif state == "OUT_OF_MEMORY":
        _write_job_log(c, job, f"slurmstepd: error: Detected 1 oom_kill event in StepId={job.job_id}.batch. "
                               "Some of the step tasks have been OOM Killed.\n", now)
    elif state == "FAILED":
        _write_job_log(c, job, "\n" + traceback_text(job.failure, job.workdir), now)
    elif state == "COMPLETED":
        _write_job_log(c, job, f"\n[{local_ts(now, c.tz)}] done · best vs_base -3.12% · saved "  # tqdm closes its bar
                               f"{job.workdir}/runs/{job.name}/final.pt\n", now)
    exit_codes = {"COMPLETED": (0, 0), "FAILED": (job.exit_code or 1, 0), "TIMEOUT": (0, 0),
                  "OUT_OF_MEMORY": (0, 125), "PREEMPTED": (0, 0), "CANCELLED": (0, 0)}
    job.exit_code, job.exit_signal = exit_codes.get(state, (job.exit_code, job.exit_signal))
    if state == "CANCELLED" and was_running:
        job.exit_code, job.exit_signal = 0, 15
    job.state, job.end, job.cg_until = state, now, None
    job.reason = "None"
    job.cancelled_by = cancelled_by
    if not was_running:
        job.nodes, job.gpu_idx, job.cpu_ids = (), {}, {}
        if state == "CANCELLED":
            job.start = None
    world.note(f"{c.name}: {job.display_id} -> {state}")


def _cancel_array_tasks(world: DemoWorld, c: SimCluster, summary: SimJob, tasks: Sequence[int], now: float) -> None:
    """Cancel some pending tasks of an array: Slurm splits them off as CANCELLED records with their own ids."""
    for task in tasks:
        summary.array_pending.remove(task)
        rec = SimJob(job_id=c.next_job_id, name=summary.name, user=summary.user, uid=summary.uid,
                     account=summary.account, partition=summary.partition, qos=summary.qos, submit=summary.submit,
                     time_limit_s=summary.time_limit_s, cpus=summary.cpus, mem_mib=summary.mem_mib,
                     workdir=summary.workdir, command=summary.command, stdout_pattern=summary.stdout_pattern,
                     stderr_pattern=summary.stderr_pattern, gpus_per_node=summary.gpus_per_node,
                     gpu_type=summary.gpu_type, typed_request=summary.typed_request,
                     array_job_id=summary.array_job_id, array_task=str(task), state="CANCELLED", reason="None",
                     end=now, cancelled_by=c.uid)
        c.next_job_id += 1
        c.jobs[rec.job_id] = rec
    summary.array_task = f"{compress_ranges(summary.array_pending)}%{summary.array_throttle}"
    world.note(f"{c.name}: cancelled array tasks {compress_ranges(tasks)} of {summary.array_job_id}")


def _requeue(c: SimCluster, job: SimJob, now: float, *, held: bool) -> None:
    job.state, job.restarts = "PENDING", job.restarts + 1
    job.reason = "JobHeldUser" if held else "BeginTime"
    job.held = held
    job.start, job.end, job.cg_until = now + 120, None, None
    job.nodes, job.gpu_idx, job.cpu_ids = (), {}, {}


def _qos_blocked(c: SimCluster, job: SimJob) -> bool:
    if c.qos_max_running is None or job.qos != "normal":
        return False
    running = sum(1 for j in c.jobs.values() if j.state == "RUNNING" and j.user == job.user and j.qos == "normal")
    return running >= c.qos_max_running


def _spawn_array_element(world: DemoWorld, c: SimCluster, summary: SimJob, now: float) -> None:
    task = summary.array_pending.pop(0)
    last = not summary.array_pending
    raw = summary.job_id if last else c.next_job_id
    if not last:
        c.next_job_id += 1
    elem = SimJob(job_id=raw, name=summary.name, user=summary.user, uid=summary.uid, account=summary.account,
                  partition=summary.partition, qos=summary.qos, submit=summary.submit,
                  time_limit_s=summary.time_limit_s, cpus=summary.cpus, mem_mib=summary.mem_mib,
                  workdir=summary.workdir, command=summary.command, stdout_pattern=summary.stdout_pattern,
                  stderr_pattern=summary.stderr_pattern, gpus_per_node=summary.gpus_per_node,
                  gpu_type=summary.gpu_type, typed_request=summary.typed_request, typed_alloc=summary.typed_alloc,
                  duration_s=summary.duration_s * (0.8 + world.rng.random() * 0.4),
                  outcome="COMPLETED", array_job_id=summary.array_job_id, array_task=str(task),
                  submit_args=summary.submit_args, util_target=summary.util_target,
                  mem_target_mib=summary.mem_target_mib)
    if last:
        del c.jobs[summary.job_id]
    else:
        lo_hi = compress_ranges(summary.array_pending)
        summary.array_task = f"{lo_hi}%{summary.array_throttle}"
    c.jobs[elem.job_id] = elem
    if not _start_job(world, c, elem, now):
        elem.state, elem.reason, elem.start = "PENDING", "Resources", None


def _step_cluster(world: DemoWorld, c: SimCluster, dt: float) -> None:
    now = world.t
    rng = world.rng
    for job in list(c.jobs.values()):
        if job.state == "COMPLETING" and job.cg_until is not None and now >= job.cg_until:
            _finish_job(world, c, job, job.outcome, job.end or now)
        elif job.state == "PENDING":
            _step_pending(world, c, job, now)
        elif job.state == "RUNNING":
            _step_running(world, c, job, now, dt, rng)
    # other users' load drifts on the nodes (free GPU counts move a little)
    if rng.random() < dt / 120:
        node = rng.choice(c.nodes)
        if node.gpu_total and _usable(node):
            idx = rng.randrange(node.gpu_total)
            if idx in node.bg_gpu_idx:
                node.bg_gpu_idx.discard(idx)
            elif idx in free_gpus(c, node, None):
                node.bg_gpu_idx.add(idx)
    if c.submit_templates and now >= c.next_submit:
        _submit_new(world, c, now)
    _tick_login(c, dt, rng)


def _step_pending(world: DemoWorld, c: SimCluster, job: SimJob, now: float) -> None:
    if job.held or job.never_runs:
        return
    if job.is_summary:
        running = sum(1 for j in c.jobs.values() if j.array_job_id == job.array_job_id and not j.is_summary
                      and j.state in ("RUNNING", "COMPLETING"))
        if running < job.array_throttle and job.start is not None and now >= job.start:
            _spawn_array_element(world, c, job, now)
        return
    if job.start is None or now < job.start:
        return
    if _qos_blocked(c, job):
        job.reason = "QOSMaxJobsPerUserLimit"
        return
    _start_job(world, c, job, now)


def _step_running(world: DemoWorld, c: SimCluster, job: SimJob, now: float, dt: float, rng: random.Random) -> None:
    assert job.start is not None
    _update_usage(job, dt, rng)
    if job.trainer is not None and not job.stalled:
        _write_job_log(c, job, job.trainer.advance(dt, now, c.tz, rng), now)
    limit_end = job.start + job.time_limit_s if job.time_limit_s else None
    planned_end = job.start + job.duration_s
    if limit_end is not None and now >= limit_end and planned_end > limit_end:
        _end_run(world, c, job, "TIMEOUT", now)
    elif now >= planned_end:
        _end_run(world, c, job, job.outcome, now)


def _end_run(world: DemoWorld, c: SimCluster, job: SimJob, outcome: str, now: float) -> None:
    if outcome == "PREEMPTED" and c.preempt_mode == "REQUEUE":
        _write_job_log(c, job, f"slurmstepd: error: *** JOB {job.job_id} ON {job.nodes[0]} CANCELLED AT "
                               f"{local_ts(now, c.tz, '%Y-%m-%dT%H:%M:%S')} DUE TO PREEMPTION ***\n", now)
        _requeue(c, job, now, held=False)
        job.outcome = "COMPLETED"
        world.note(f"{c.name}: {job.display_id} preempted and requeued")
        return
    job.outcome = outcome
    job.state, job.end, job.cg_until = "COMPLETING", now, now + STEP_S
    world.note(f"{c.name}: {job.display_id} completing ({outcome})")


def _update_usage(job: SimJob, dt: float, rng: random.Random) -> None:
    """Random walk of the job's accounting numbers (what sstat reports). GPU util/memory are per task, summed over
    that task's GPUs; the batch step of an srun/torchrun job holds no GPU."""
    target = 1.0 if job.stalled else job.util_target
    steps = ((job.batch, 0 if job.step_tasks else job.gpus_per_node),
             (job.step0, job.gpus // max(1, job.step_tasks)))
    for usage, gpus_per_task in steps:
        if usage is None:
            continue
        cores = 0.1 if job.stalled else 0.9 + 0.2 * rng.random()
        usage.ave_cpu_s += cores * dt
        usage.ave_rss_kb = min(job.mem_mib * 1024 * 9 // 10, max(usage.ave_rss_kb, 2_000_000)
                               + int(rng.random() * 20_000))
        usage.max_rss_kb = max(usage.max_rss_kb, usage.ave_rss_kb + int(rng.random() * 200_000))
        usage.disk_bytes += int(rng.random() * 1_500_000 * dt)
        usage.pages = max(usage.pages, 3000 + usage.disk_bytes // 2_600_000)
        if gpus_per_task:
            per_gpu = usage.gpuutil_ave / gpus_per_task
            per_gpu = max(0.0, min(100.0, per_gpu + (target - per_gpu) * 0.3 + rng.uniform(-2.0, 2.0)))
            usage.gpuutil_ave = round(per_gpu * gpus_per_task)
            usage.gpuutil_max = max(usage.gpuutil_max, usage.gpuutil_ave)
            per_gpu_mem = min(job.mem_target_mib, usage.gpumem_ave_mib // gpus_per_task + 4000)
            usage.gpumem_ave_mib = per_gpu_mem * gpus_per_task
            usage.gpumem_max_mib = max(usage.gpumem_max_mib, usage.gpumem_ave_mib)


def _submit_new(world: DemoWorld, c: SimCluster, now: float) -> None:
    rng = world.rng
    c.next_submit = now + rng.uniform(900, 1500)
    if len(c.live_jobs()) >= 12:
        return
    name_t, partition, gpus, gpu_type = rng.choice(c.submit_templates)
    seed = rng.randrange(40, 60)
    name = name_t.format(seed=seed, bench=rng.choice(["cifar", "imagenette", "voc", "places"]),
                         method=rng.choice(["warmup", "const", "cosinewarm", "adam"]))
    outcome = rng.choices(["COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED"],
                          weights=[70, 12, 6, 6, 6])[0]
    duration = rng.uniform(1200, 3 * 3600)
    limit = 6 * 3600 if outcome != "TIMEOUT" else int(duration * 0.6) // 60 * 60
    job = SimJob(job_id=c.next_job_id, name=name, user=c.user, uid=c.uid, account=c.associations[0][1],
                 partition=partition, qos="normal", submit=now, time_limit_s=limit, cpus=8, mem_mib=48 * 1024,
                 workdir=_PROJECT[c.name], command=f"{_PROJECT[c.name]}/jobs/train.sbatch",
                 stdout_pattern=f"{_PROJECT[c.name]}/logs/%x_%j.out",
                 stderr_pattern=f"{_PROJECT[c.name]}/logs/%x_%j.out",
                 gpus_per_node=gpus, gpu_type=gpu_type, start=now + rng.uniform(60, 600),
                 duration_s=duration, outcome=outcome, exit_code=1, failure=rng.choice(["cuda_oom", "value"]))
    c.next_job_id += 1
    c.jobs[job.job_id] = job
    world.note(f"{c.name}: submitted {job.job_id} {name}")


def _tick_login(h: SimCluster | SimWorkstation, dt: float, rng: random.Random) -> None:
    """Login-node / host CPU counters (/proc/stat in USER_HZ) and load average."""
    total = int(h.ncpu * 100 * dt)
    busy_frac = min(0.95, max(0.01, h.load / h.ncpu))
    user = int(total * busy_frac * 0.8)
    system = int(total * busy_frac * 0.2)
    j = h.cpu_jiffies
    j[0] += user
    j[1] += int(user * 0.05)
    j[2] += system
    j[3] += max(0, total - user - system)
    j[4] += int(total * 0.001)
    h.load = max(0.05, h.load + (rng.uniform(-0.3, 0.3)) * dt / 15)


# ====================================================================================================== workstation sim
def _write_proc_log(ws: SimWorkstation, proc: SimProc, text: str, at: float) -> None:
    if not proc.fd1.startswith("/") or proc.fd1.startswith("/dev/"):
        return
    lg = ws.logs.get(proc.fd1)
    if lg is None:
        lg = ws.logs[proc.fd1] = SimLog(proc.fd1, at)
    lg.write(text, at)


def _exit_proc(world: DemoWorld, ws: SimWorkstation, pid: int, *, reason: str) -> None:
    proc = ws.procs.pop(pid, None)
    if proc is None:
        return
    for child in ws.children(pid):              # orphans are re-parented to init
        child.ppid = 1
    runner = next((r for r in ws.runners.values() if proc.ppid == r.root), None)
    if runner is not None and reason in ("exit", "crash") and proc.gpu is not None:
        if runner.items:
            runner.pending.append((world.t + 15, proc.gpu))
    world.note(f"{ws.name}: pid {pid} exited ({reason})")


def _spawn_child(world: DemoWorld, ws: SimWorkstation, runner: QueueRunner, gpu: int) -> None:
    root = ws.procs.get(runner.root)
    if root is None or not runner.items:
        return
    item = runner.items.pop(0)
    pid = ws.next_pid
    ws.next_pid += world.rng.randrange(3, 40)
    now = int(world.t)
    run_dir = f"runs/{item}"
    child = SimProc(pid=pid, ppid=root.pid, pgid=root.pgid, user=ws.user, start=now,
                    args=f"{ws.project_root}/.venv/bin/python -m orb.train --config configs/exp_a.yaml"
                         f" data={item.split('/')[0]} sched={_item_sched(item)} seed=44 run_dir={run_dir}",
                    stat="RNl", cpu_pct=110.0, mem_pct=1.9, rss_kb=10_500_000, cwd=root.cwd, fd1=root.fd1,
                    fd2=root.fd2, env=dict(root.env, CUDA_VISIBLE_DEVICES=str(gpu)), gpu=gpu,
                    gpu_mem_mib=15234, gpu_util=98.0,
                    end_at=world.t + world.rng.uniform(*runner.child_s), item=item,
                    trainer=Trainer(epochs=200, epoch_s=40.0, iters=1800))
    ws.procs[pid] = child
    _write_proc_log(ws, root, f"[{local_ts(world.t, ws.tz)}] launch: item {runner.next_no}/{runner.total} "
                              f"{item} on gpu {gpu} (pid {pid})\n", world.t)
    runner.next_no += 1
    world.note(f"{ws.name}: launcher {root.pid} started pid {pid} on gpu {gpu}")


def _step_workstation(world: DemoWorld, ws: SimWorkstation, dt: float) -> None:
    now = world.t
    rng = world.rng
    for proc in list(ws.procs.values()):
        if proc.end_at is not None and now >= proc.end_at:
            if proc.crash:
                _write_proc_log(ws, proc, "\n" + traceback_text(proc.crash, ws.project_root), now)
            elif proc.trainer is not None:
                _write_proc_log(ws, proc, f"[{local_ts(now, ws.tz)}] done · {proc.item or 'run'} · saved "
                                          "final.pt\n", now)
            _exit_proc(world, ws, proc.pid, reason="crash" if proc.crash else "exit")
            continue
        if proc.cpu_pct > 1:
            proc.cpu_pct = max(1.0, min(400.0, proc.cpu_pct + rng.uniform(-6, 6)))
        if proc.gpu is not None and proc.gpu_util:
            proc.gpu_util = max(40.0, min(100.0, proc.gpu_util + rng.uniform(-4, 4)))
        if proc.trainer is not None:
            _write_proc_log(ws, proc, proc.trainer.advance(dt, now, ws.tz, rng), now)
    for runner in ws.runners.values():
        due = [p for p in runner.pending if p[0] <= now]
        for entry in due:
            runner.pending.remove(entry)
            _spawn_child(world, ws, runner, entry[1])
    _update_gpus(ws, dt, rng)
    ws.load = sum(max(0.0, p.cpu_pct) for p in ws.procs.values()) / 100 + ws.bg_cpu_pct / 100
    _tick_login(ws, dt, rng)


def _update_gpus(ws: SimWorkstation, dt: float, rng: random.Random) -> None:
    for gpu in ws.gpus:
        procs = [p for p in ws.procs.values() if p.gpu == gpu.index]
        util = min(100.0, sum(p.gpu_util for p in procs))
        gpu.util = round(max(0.0, util + (rng.uniform(-3, 3) if util else 0.0)))
        gpu.util = min(100.0, gpu.util)
        gpu.mem_used_mib = 1 + sum(p.gpu_mem_mib + 10 for p in procs) if procs else 1
        gpu.mem_util = round(gpu.util * 0.8) if gpu.util else 0
        target_t = 30 + gpu.util * 0.48
        gpu.temp_c += (target_t - gpu.temp_c) * min(1.0, dt / 60)
        target_p = gpu.idle_power_w + (gpu.power_limit_w * 0.95 - gpu.idle_power_w) * gpu.util / 100
        gpu.power_w = max(gpu.idle_power_w * 0.8, target_p + rng.uniform(-6, 6) if gpu.util else
                          gpu.idle_power_w + rng.uniform(-3, 3))
        gpu.clocks_sm = 1860 if gpu.util > 5 else 210
        if gpu.fan is not None:
            gpu.fan = int(30 + max(0.0, gpu.temp_c - 45) * 0.7)


# ====================================================================================================== scenario
_PROJECT = {"k2": "/shared/lab/alice/orb", "vega": "/shared/lab/alice/orb", "lyra": "/vol/alice/orb"}
def _item_sched(item: str) -> str:
    """Queue item `<data>/baseline-<sched>/s<seed>` -> its LR schedule."""
    return item.split("/")[1].removeprefix("baseline-")

def _slurm_config(cluster: str, preempt: str, launch: str, prolog: str,
                  priority: dict[str, str] | None = None) -> dict[str, str]:
    # the keys the slurm.config section greps for, in `scontrol show config` order
    cfg = {"AccountingStorageType": "accounting_storage/slurmdbd", "ClusterName": cluster,
           "JobAcctGatherFrequency": "30", "JobAcctGatherType": "jobacct_gather/cgroup",
           "LaunchParameters": launch, "MinJobAge": "300 sec", "PreemptMode": preempt, "PrivateData": "none",
           "PrologFlags": prolog, **(priority or {})}
    return dict(sorted(cfg.items()))


def _priority(half_life: str, reset: str, **weights: str) -> dict[str, str]:
    """The priority keys of `scontrol show config` (§5.2.10); weights not given are 0, TRES "(null)"."""
    out = {"PriorityDecayHalfLife": half_life, "PriorityFlags": "", "PriorityMaxAge": "7-00:00:00",
           "PriorityType": "priority/multifactor", "PriorityUsageResetPeriod": reset}
    for factor in ("Age", "Assoc", "FairShare", "JobSize", "Partition", "QOS"):
        out[f"PriorityWeight{factor}"] = weights.get(factor, "0")
    out["PriorityWeightTRES"] = weights.get("TRES", "(null)")
    return out


# what `command -v` finds; detect.tools reports the subset it asks for (render.DETECT_TOOLS)
BASE_TOOLS = frozenset({"sh", "awk", "cat", "cut", "date", "getent", "grep", "head", "id", "kill", "nproc", "paste",
                        "readlink", "sleep", "sort", "tail", "tr", "wc", "timeout", "less", "stat", "ps", "pgrep"})
SLURM_TOOLS = BASE_TOOLS | {"squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel", "sacctmgr", "srun", "ssh",
                           "sprio", "sshare"}
WS_TOOLS = BASE_TOOLS | {"nvidia-smi"}


def _job(c: SimCluster, job_id: int, name: str, **kw: Any) -> SimJob:
    defaults: dict[str, Any] = dict(user=c.user, uid=c.uid, account=c.associations[0][1], qos="normal",
                                       time_limit_s=2 * 86400, cpus=8, mem_mib=48 * 1024,
                                       workdir=_PROJECT.get(c.name, "/tmp"),
                                       command=f"{_PROJECT.get(c.name, '/tmp')}/jobs/train.sbatch",
                                       stdout_pattern=f"{_PROJECT.get(c.name, '/tmp')}/logs/%x_%j.out")
    defaults.update(kw)
    job = SimJob(job_id=job_id, name=name, **defaults)
    c.jobs[job_id] = job
    return job


def _run(world: DemoWorld, c: SimCluster, job: SimJob, *, started_ago: float, nodes: Sequence[str],
         gpu_idx: Sequence[Sequence[int]], cpu_ids: Sequence[str], log_idle: float = 0.0) -> None:
    """Put a scenario job into RUNNING as if it had started `started_ago` seconds before t0."""
    now = world.t0
    job.state, job.reason, job.start = "RUNNING", "None", now - started_ago
    job.nodes = tuple(nodes)
    job.gpu_idx = {n: tuple(i) for n, i in zip(nodes, gpu_idx) if i}
    job.cpu_ids = dict(zip(nodes, cpu_ids))
    first = c.node(nodes[0])
    if job.gpus_per_node and first is not None and gpu_idx and gpu_idx[0]:
        job.alloc_gpu_type = _type_of(first, gpu_idx[0][0])
    _set_log_paths(c, job)
    rng = random.Random(world.seed + stable_hash(f"{c.name}:{job.job_id}"))
    job.trainer = Trainer(epochs=300, epoch_s=max(30.0, job.duration_s / 300), iters=4500)
    hist_end = now - log_idle
    text = _job_header(c, job, job.start) + job.trainer.history(job.start, hist_end, c.tz, rng)
    lg = SimLog(job.log_paths[0], job.start)
    lg.write(text, hist_end)
    c.logs[job.log_paths[0]] = lg
    job.batch = StepUsage(ntasks=1, ave_cpu_s=started_ago * 1.05, ave_rss_kb=10_398_988, max_rss_kb=11_506_244,
                          disk_bytes=int(started_ago * 640_000), pages=3652)
    gpu_step = job.batch
    if job.step_tasks:
        gpu_step = job.step0 = StepUsage(ntasks=job.step_tasks, ave_cpu_s=started_ago * 6.2, ave_rss_kb=31_250_000,
                                         max_rss_kb=33_900_000, disk_bytes=int(started_ago * 2_100_000), pages=12011)
    per_task_gpus = job.gpus // max(1, job.step_tasks) if job.step_tasks else job.gpus_per_node
    util = round((1.0 if job.stalled else job.util_target) * per_task_gpus)
    gpu_step.gpuutil_ave = util
    gpu_step.gpuutil_max = min(100 * per_task_gpus, util + 1 + rng.randrange(3))
    gpu_step.gpumem_ave_mib = job.mem_target_mib * per_task_gpus
    gpu_step.gpumem_max_mib = job.mem_target_mib * per_task_gpus + 1200


def _ended(world: DemoWorld, c: SimCluster, job: SimJob, *, state: str, started_ago: float | None, ran: float,
           nodes: Sequence[str] = (), exit_code: int = 0, exit_signal: int = 0, log: str = "") -> None:
    """A history job that ended before t0 (`started_ago` None = never started)."""
    job.state, job.reason = state, "None"
    job.exit_code, job.exit_signal = exit_code, exit_signal
    if started_ago is None:
        job.start, job.end = None, job.submit + ran
        job.nodes = ()
    else:
        job.start = world.t0 - started_ago
        job.end = job.start + ran
        job.nodes = tuple(nodes)
        _set_log_paths(c, job)
        lg = SimLog(job.log_paths[0], job.start)
        rng = random.Random(world.seed + stable_hash(f"{c.name}:{job.job_id}"))
        tr = Trainer(epochs=300, epoch_s=max(30.0, ran / 40), iters=4500)
        lg.write(_job_header(c, job, job.start) + tr.history(job.start, job.end, c.tz, rng, keep=40) + "\n" + log,
                 job.end)
        c.logs[job.log_paths[0]] = lg
        job.batch = StepUsage(ntasks=1, ave_cpu_s=ran * 0.98, ave_rss_kb=4_713_932, max_rss_kb=4_713_932,
                              disk_bytes=int(ran * 38_000), pages=4886,
                              gpuutil_ave=92 if job.gpus else 0, gpuutil_max=92 if job.gpus else 0,
                              gpumem_ave_mib=4756 if job.gpus else 0, gpumem_max_mib=4756 if job.gpus else 0)


def _build_default(world: DemoWorld) -> None:
    for build in (_build_k2, _build_vega, _build_lyra, _build_labstation, _build_lynx):
        build(world)
    _build_transitions(world)
    world.faults["lynx"] = {"auth"}          # ws-duo shape: needs a login until `connect`
    for host in world.hosts.values():
        if not any(host.cpu_jiffies):        # /proc/stat counters since boot, ~busy as the current load
            ticks = int(host.ncpu * 100 * (world.t0 - host.boot))
            busy = min(0.9, host.load / host.ncpu)
            host.cpu_jiffies = [int(ticks * busy * 0.8), int(ticks * busy * 0.04), int(ticks * busy * 0.16),
                                int(ticks * (1 - busy)), int(ticks * 0.001), 0, int(ticks * 0.0005), 0, 0, 0]


def _rtx3090(name: str, bg: Iterable[int], cpus_bg: int, **kw: Any) -> SimNode:
    return SimNode(name=name, partitions=("gpu-preempt",), cpus=64, mem_mib=512000,
                   gres=(GresDef("nvidia_geforce_rtx_3090", 8, "(S:0-7)"),), bg_gpu_idx=set(bg), bg_cpus=cpus_bg,
                   bg_mem_mib=cpus_bg * 4600, **kw)


def _build_k2(world: DemoWorld) -> None:
    t0 = world.t0
    c = SimCluster(name="k2", slurm_version="25.05.8", login_host="login02", user="alice", uid=10001,
                   group="rs", gid=500, tz=timedelta(hours=-5),
                   config=_slurm_config("k2", "REQUEUE", "(null)", "Alloc,Contain",
                                        _priority("14-00:00:00", "NONE", Age="1000", FairShare="100000",
                                                  JobSize="1000", Partition="1000")),
                   associations=(("k2", "lab", "", "gpu-priority,normal,labpi"),), tools=SLURM_TOOLS,
                   shares=(("lab", "1", "1.000000", "8123456", "0.833333", "0.030000", "1.200000"),),
                   partition_tier={"lab": 1.0, "gpu-preempt": 0.0},
                   gpu_tres=True, merged_stderr="repeat", stat_verb="stat", latency=(0.05, 0.12), ncpu=32,
                   mem_total_kb=196_000_000, load=2.3, next_job_id=3372155, boot=t0 - 41 * 86400,
                   submit_templates=(("orb-swp--train--{bench}-{method}--s{seed}--a01", "gpu-preempt", 1,
                                      "nvidia_geforce_rtx_3090"),
                                     ("orb-swp--train--{bench}-{method}--s{seed}--a01", "gpu-preempt", 1,
                                      "nvidia_a100-sxm4-40gb")),
                   next_submit=t0 + 600)
    c.nodes = [
        _rtx3090("n-31-01", range(8), 34, free_mem_mib=23103),
        _rtx3090("n-31-02", range(8), 40, free_mem_mib=40339),
        _rtx3090("n-31-03", range(8), 44, free_mem_mib=22073),
        SimNode("n-31-04", ("gpu-preempt",), 56, 128000, (GresDef("tesla_v100-pcie-16gb", 4, "(S:0)"),),
                bg_cpus=6, bg_mem_mib=73728, bg_gpu_idx={0, 2, 3}, free_mem_mib=13725),
        SimNode("n-31-05", ("gpu-preempt",), 64, 515000,
                (GresDef("nvidia_a100_80gb_pcie", 1, "(S:0-7)"), GresDef("nvidia_a100_80gb_pcie_3g.40gb", 2, "(S:0-7)"),
                 GresDef("nvidia_a100_3g.20gb", 2, "(S:0-7)"), GresDef("nvidia_a100_80gb_pcie_1g.10gb", 7, "(S:0-7)")),
                state="planned", free_mem_mib=59616),
        _rtx3090("n-32-01", range(6), 48, free_mem_mib=31877),
        _rtx3090("n-32-02", range(8), 26, free_mem_mib=11467),
        _rtx3090("n-32-03", range(5), 36, flag="-", free_mem_mib=7882),
        _rtx3090("n-32-04", range(7), 54, free_mem_mib=13730),
        _rtx3090("n-32-05", range(8), 58, free_mem_mib=85214),
        SimNode("n-33-06", ("gpu-preempt",), 32, 384000, (GresDef("tesla_p100-pcie-16gb", 1, "(S:0)"),),
                free_mem_mib=192271),
        SimNode("n-34-01", ("gpu-preempt",), 64, 512000, (GresDef("nvidia_geforce_rtx_3090", 8, "(S:0-7)"),),
                state="down", flag="*", reason="Not responding", free_mem_mib=0),
        SimNode("n-34-02", ("gpu-preempt",), 64, 512000, (GresDef("nvidia_geforce_rtx_3090", 8, "(S:0-7)"),),
                state="drained", reason="maint: PSU swap (ticket 4412)", free_mem_mib=501000),
        SimNode("n-37-01", ("gpu-preempt",), 112, 1031900, (GresDef("nvidia_a100-sxm4-80gb", 4, "(S:0-7)"),),
                bg_cpus=16, bg_mem_mib=65536, free_mem_mib=347459),
        SimNode("n-37-02", ("gpu-preempt",), 112, 1031900, (GresDef("nvidia_a100-sxm4-80gb", 4, "(S:0-7)"),),
                bg_cpus=24, bg_mem_mib=131072, free_mem_mib=301122),
        SimNode("n-38-01", ("gpu-preempt",), 64, 1031900, (GresDef("nvidia_h200_nvl", 4, "(S:0-1)"),),
                bg_cpus=24, bg_mem_mib=262144, bg_gpu_idx={2, 3}, free_mem_mib=88121),
        SimNode("n-38-02", ("gpu-preempt",), 64, 1031900, (GresDef("nvidia_h200_nvl", 4, "(S:0-1)"),),
                bg_cpus=8, bg_mem_mib=131072, bg_gpu_idx={0}, free_mem_mib=402311),
        SimNode("n-38-03", ("gpu-preempt",), 64, 515000,
                (GresDef("nvidia_rtx_pro_6000_blackwell_server_edition", 2, "(S:0-1)"),), free_mem_mib=480222),
        SimNode("n-39-01", ("lab",), 64, 1031900, (GresDef("nvidia_h100_80gb_hbm3", 4, "(S:0-1)"),),
                bg_cpus=64, bg_mem_mib=800000, bg_gpu_idx={0, 1, 2, 3}, free_mem_mib=12000),
        SimNode("n-39-02", ("lab",), 64, 1031900, (GresDef("nvidia_h100_80gb_hbm3", 4, "(S:0-1)"),),
                bg_cpus=48, bg_mem_mib=600000, bg_gpu_idx={0, 1, 2, 3}, free_mem_mib=33000),
        SimNode("n-40-01", ("gpu-preempt",), 128, 1031900, (GresDef("nvidia_a100-sxm4-40gb", 4, "(S:0-3)"),),
                bg_cpus=8, bg_mem_mib=65536, bg_gpu_idx={3}, free_mem_mib=211876),
    ]
    world.hosts["k2"] = c
    a100 = dict(partition="gpu-preempt", qos="gpu-priority", time_limit_s=3 * 86400, gpus_per_node=1,
                gpu_type="nvidia_a100-sxm4-40gb", duration_s=2 * 86400, util_target=97.0, mem_target_mib=24870)
    for jid, method, idx, cpus, util, stalled in ((3370390, "warmup", 2, "64-71", 98.0, False),
                                                  (3370391, "const", 0, "72-79", 97.0, False),
                                                  (3370392, "cosinewarm", 1, "80-87", 98.0, True)):
        name = f"orb-swp--train--cifar-{method}--s43--a02"
        job = _job(c, jid, name, submit=t0 - 18 * 3600 - 43 * 60 + (jid - 3370390),
                   stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out", stalled=stalled,
                   **dict(a100, util_target=util))
        _run(world, c, job, started_ago=4 * 3600 + 7 * 60, nodes=["n-40-01"], gpu_idx=[[idx]], cpu_ids=[cpus],
             log_idle=47 * 60 if stalled else 0.0)
    lmft = "/shared/lab/bob/lmft"
    job = _job(c, 3371310, "lmft-x07-sampler-s0", partition="gpu-preempt", submit=t0 - 9 * 3600 - 2 * 60 - 36,
               cpus=16, mem_mib=256 * 1024, gpus_per_node=2, gpu_type="nvidia_h200_nvl", typed_alloc=False,
               typed_req_tres=False, workdir=lmft, command=f"{lmft}/jobs/train.sbatch", stdout_pattern=f"{lmft}/logs/%x_%j.out",
               duration_s=40 * 3600, util_target=86.0, mem_target_mib=49200,
               banner="TensorBoard 2.20.0 at http://n-38-01:6006/ (Press CTRL+C to quit)")
    _run(world, c, job, started_ago=9 * 3600 + 2 * 60 + 14, nodes=["n-38-01"], gpu_idx=[[0, 1]], cpu_ids=["0-15"])
    job = _job(c, 3372094, "ddp-scale--imagenet-vit-b--n2", partition="gpu-preempt", submit=t0 - 2 * 3600,
               cpus=64, mem_mib=480 * 1024, num_nodes=2, gpus_per_node=4, gpu_type="nvidia_a100-sxm4-80gb",
               time_limit_s=86400, duration_s=20 * 3600, util_target=91.0, mem_target_mib=61000, step_tasks=2,
               stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.err",
               command=f"{_PROJECT['k2']}/scripts/ddp_vit.sbatch", submit_args="--nodes 2 --gpus-per-node 4")
    _run(world, c, job, started_ago=72 * 60, nodes=["n-37-01", "n-37-02"], gpu_idx=[[0, 1, 2, 3], [0, 1, 2, 3]],
         cpu_ids=["16-47", "24-55"])
    job = _job(c, 3371232, "orb-swp--eval--cifar-warmup--s43", partition="gpu-preempt", submit=t0 - 3600,
               time_limit_s=3600, gpus_per_node=1, gpu_type="nvidia_geforce_rtx_3090", duration_s=27 * 60,
               stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out", util_target=64.0, mem_target_mib=9100)
    _run(world, c, job, started_ago=25 * 60, nodes=["n-32-03"], gpu_idx=[[5]], cpu_ids=["36-43"])
    diag = "/shared/lab/alice/rl-bench"
    _job(c, 3372066, "smoke-test", partition="gpu-preempt", qos="gpu-priority", submit=t0 - 18 * 60,
         time_limit_s=3600, gpus_per_node=1, gpu_type="nvidia_geforce_rtx_3090", start=t0 + 35 * 60,
         duration_s=20 * 60, workdir=diag, command=f"{diag}/jobs/train.sbatch", stdout_pattern=f"{diag}/logs/%x_%j.out",
         stderr_pattern=f"{diag}/logs/%x_%j.out", scontrol_extra={"SchedNodeList": "n-32-03"})
    _job(c, 3372143, "orb-sweep-lab--s44", partition="lab", qos="labpi", reason="Resources", submit=t0 - 40 * 60,
         gpus_per_node=2, gpu_type="nvidia_h100_80gb_hbm3", start=None,
         stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out")
    # history (sacct)
    for k, (seed, node, ran, ago) in enumerate(((42, "n-38-03", 5 * 3600 + 39 * 60 + 38, 16 * 3600),
                                                (45, "n-38-03", 5 * 3600 + 43 * 60 + 34, 15.9 * 3600),
                                                (46, "n-32-03", 11 * 3600 + 24 * 60 + 33, 15.2 * 3600))):
        job = _job(c, 3370735 + k, f"orb-ab--train--resnet18-bs512--s{seed}--a02", partition="gpu-preempt",
                   qos="gpu-priority", submit=t0 - ago - 3600, time_limit_s=86400, gpus_per_node=1,
                   gpu_type="nvidia_geforce_rtx_3090" if node == "n-32-03" else
                   "nvidia_rtx_pro_6000_blackwell_server_edition", typed_request=False,
                   stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out")
        _ended(world, c, job, state="COMPLETED", started_ago=ago, ran=ran, nodes=[node],
               log=f"[{local_ts(t0 - ago + ran, c.tz)}] done · best vs_base -2.87% · saved final.pt\n")
        if node == "n-32-03":            # its GPU sat idle (the efficiency card's "GPU idle" note)
            job.batch.gpuutil_ave = job.batch.gpuutil_max = 2
    lmft_job = _job(c, 3368404, "lmft-x0-lr1e-5", partition="gpu-preempt", submit=t0 - 5 * 3600, cpus=16,
                    mem_mib=256 * 1024, gpus_per_node=2, gpu_type="nvidia_h200_nvl", workdir=lmft,
                    command=f"{lmft}/jobs/train.sbatch", stdout_pattern=f"{lmft}/logs/%x_%j.out")
    _ended(world, c, lmft_job, state="FAILED", started_ago=4 * 3600, ran=59 * 60 + 5, nodes=["n-38-01"],
           exit_code=1, log=traceback_text("cuda_oom", lmft, pkg="lmft", gpu_gib="139.80",
                                           used_gib="138.12"))
    job = _job(c, 3371193, "orb-swp--train--imagenette-warmup--s42--a01", partition="gpu-preempt",
               submit=t0 - 18 * 3600, time_limit_s=12 * 3600, gpus_per_node=1, gpu_type="nvidia_geforce_rtx_3090",
               stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out")
    _ended(world, c, job, state="TIMEOUT", started_ago=17 * 3600, ran=12 * 3600 + 21, nodes=["n-32-04"],
           log=f"slurmstepd: error: *** JOB 3371193 ON n-32-04 CANCELLED AT "
               f"{local_ts(t0 - 17 * 3600 + 12 * 3600 + 21, c.tz, '%Y-%m-%dT%H:%M:%S')} DUE TO TIME LIMIT ***\n")
    job = _job(c, 3371400, "lmft-x07-trainer-s2", partition="gpu-preempt", submit=t0 - 8 * 3600, cpus=16,
               mem_mib=256 * 1024, gpus_per_node=2, gpu_type="nvidia_h200_nvl", workdir=lmft,
               command=f"{lmft}/jobs/train.sbatch", stdout_pattern=f"{lmft}/logs/%x_%j.out")
    _ended(world, c, job, state="OUT_OF_MEMORY", started_ago=7.5 * 3600, ran=31 * 60 + 9, nodes=["n-38-02"],
           exit_signal=125, log="slurmstepd: error: Detected 1 oom_kill event in StepId=3371400.batch. Some of the "
                                "step tasks have been OOM Killed.\n")
    job = _job(c, 3371311, "lmft-x07-trainer-s1", partition="gpu-preempt", submit=t0 - 9 * 3600 - 120, cpus=16,
               mem_mib=256 * 1024, gpus_per_node=2, gpu_type="nvidia_h200_nvl", workdir=lmft,
               command=f"{lmft}/jobs/train.sbatch", stdout_pattern=f"{lmft}/logs/%x_%j.out")
    _ended(world, c, job, state="CANCELLED", started_ago=None, ran=246)
    job.cancelled_by = c.uid
    _add_config(world, HostConfig(name="k2", ssh="k2", group="clusters", slurm=_slurm_settings(
        protected=("labpi",), templates=("{workdir}/logs/{name}_{jobid}.out",))))


def _slurm_settings(*, protected: tuple[str, ...] = (), templates: tuple[str, ...] = ()) -> SlurmSettings:
    return SlurmSettings(protected_partitions=protected, protected_qos=protected, log_templates=templates)


def _add_config(world: DemoWorld, cfg: HostConfig) -> None:
    world.configs[cfg.name] = cfg


def _build_vega(world: DemoWorld) -> None:
    t0 = world.t0
    c = SimCluster(name="vega", slurm_version="25.11.4", login_host="login-v01", user="alice", uid=10001,
                   group="rs", gid=500, tz=timedelta(hours=-5),
                   config=_slurm_config("vega", "REQUEUE", "ulimit_pam_adopt", "Alloc,Contain",
                                        _priority("00:00:00", "MONTHLY", Age="10000", FairShare="10000",
                                                  QOS="10000")),
                   associations=(("vega", "lab", "", "normal"), ("vega", "dept", "", "normal")),
                   shares=(("lab", "1", "0.500000", "123456", "0.010000", "1.000000", "50.000000"),
                           ("dept", "1", "0.032258", "0", "0.000000", "0.950000", "inf")),
                   tools=SLURM_TOOLS, gpu_tres=True, merged_stderr="empty", stat_verb="statx", qos_max_running=4,
                   latency=(0.05, 0.12), ncpu=64, mem_total_kb=263_000_000, load=5.1, next_job_id=824547,
                   boot=t0 - 12 * 86400,
                   submit_templates=(("orb-swp--train--{bench}-{method}--s{seed}--a01", "h200", 1,
                                      "nvidia_h200_nvl"),),
                   next_submit=t0 + 1500)
    nodes: list[SimNode] = []
    for i, name in enumerate(("n-31-01", "n-32-01", "n-33-01", "n-34-01")):
        nodes.append(SimNode(name, ("a30",), 128, 1024000, (GresDef("nvidia_a30", 2, "(S:4,10)"),),
                             flag="-", bg_cpus=48 - 8 * (i % 2), bg_mem_mib=73728,
                             bg_gpu_idx={0, 1} if i != 2 else {0}, free_mem_mib=600000 + 70000 * i))
    nodes.append(SimNode("n-34-02", ("h100",), 64, 512000, (GresDef("nvidia_h100_80gb_hbm3", 4, "(S:0-1)"),),
                         flag="-", bg_cpus=48, bg_mem_mib=319488, bg_gpu_idx={0, 1, 2, 3}, free_mem_mib=11238))
    nodes.append(SimNode("n-35-01", ("h100",), 64, 512000, (GresDef("nvidia_h100_nvl", 1, "(S:1)"),),
                         flag="-", bg_cpus=42, bg_mem_mib=262144, bg_gpu_idx={0}, free_mem_mib=175340))
    nodes.append(SimNode("n-36-01", ("h100",), 64, 512000, (GresDef("nvidia_h100_nvl", 2, "(S:1)"),),
                         flag="-", bg_cpus=32, bg_mem_mib=188416, bg_gpu_idx={0, 1}, free_mem_mib=167870))
    for rack in (7, 8):
        for k in range(1, 13):
            name = f"n-{rack + 30:02d}-{k:02d}"
            if name == "n-37-09":
                nodes.append(SimNode(name, ("h200",), 64, 384000, (GresDef("nvidia_h200_nvl", 2, "(S:3-4)"),),
                                     state="reserved", free_mem_mib=245744))
                continue
            bg = {0, 1} if (rack * 13 + k) % 5 else {0}
            if name in ("n-38-01", "n-38-12", "n-37-05", "n-38-09"):
                bg = set()
            elif name in ("n-38-06", "n-37-12"):
                bg = {1}
            nodes.append(SimNode(name, ("h200",), 64, 384000, (GresDef("nvidia_h200_nvl", 2, "(S:3-4)"),),
                                 flag="-" if k % 4 else "", bg_cpus=16 + (k % 3) * 8, bg_mem_mib=131072,
                                 bg_gpu_idx=bg, free_mem_mib=20000 + 9000 * k))
    c.nodes = nodes
    world.hosts["vega"] = c
    lmft = "/shared/lab/bob/lmft"
    sk = dict(partition="h200", account="lab", cpus=16, mem_mib=256 * 1024, gpus_per_node=2,
              gpu_type="nvidia_h200_nvl", workdir=lmft, command=f"{lmft}/jobs/train.sbatch",
              stdout_pattern=f"{lmft}/logs/%x_%j.out", duration_s=36 * 3600,
              submit_args="--config configs/exp_b.yaml seed=1 lr=1e-5 steps=100")
    # GPU0 busy, GPU1 at 0 % holding 85 GB (the field case of vega 724655): sstat sums to ~100 % over 2 GPUs
    job = _job(c, 823954, "lmft-x07-sampler-s1", submit=t0 - 31 * 3600 - 51 * 60, util_target=50.0,
               mem_target_mib=91625, idle_gpus={1: 84725}, shmem_mib=57555,
               scontrol_extra={"Priority": "60592", "ExcNodeList": "n-38-06"}, **sk)
    _run(world, c, job, started_ago=5 * 3600 + 54 * 60 + 15, nodes=["n-38-01"], gpu_idx=[[0, 1]],
         cpu_ids=["24-39"])
    job = _job(c, 823952, "lmft-x07-trainer-s1", submit=t0 - 31 * 3600 - 52 * 60, util_target=91.0,
               mem_target_mib=104000, **sk)
    _run(world, c, job, started_ago=8 * 3600 + 26 * 60 + 12, nodes=["n-38-12"], gpu_idx=[[0, 1]],
         cpu_ids=["0-15"])
    orb = dict(partition="h200", account="lab", gpus_per_node=1, gpu_type="nvidia_h200_nvl",
               stderr_pattern=f"{_PROJECT['vega']}/logs/%x_%j.out", duration_s=14 * 3600, mem_target_mib=4756)
    for jid, seed, node, ago, util in ((824246, 48, "n-38-06", 6 * 3600 + 4 * 60 + 32, 71.0),
                                       (824245, 47, "n-37-12", 6 * 3600 + 5 * 60 + 32, 80.0)):
        job = _job(c, jid, f"orb-swp--train--voc-cosinewarm--s{seed}--a01", submit=t0 - 19 * 3600,
                   util_target=util, **orb)
        _run(world, c, job, started_ago=ago, nodes=[node], gpu_idx=[[0]], cpu_ids=["40-47"])
    _job(c, 824479, "lmft-x07-trainer-s0", reason="QOSMaxJobsPerUserLimit", submit=t0 - 8 * 3600 - 25 * 60,
         start=t0, **sk)
    _job(c, 824507, "lmft-x07-eval-s0", reason="DependencyNeverSatisfied", submit=t0 - 7 * 3600,
         dependency="afterok:824295(failed)", never_runs=True, start=None,
         **dict(sk, gpus_per_node=1, cpus=8, mem_mib=64 * 1024))
    job = _job(c, 824295, "lmft-x07-sampler-s0", submit=t0 - 9 * 3600, **sk)
    _ended(world, c, job, state="FAILED", started_ago=8 * 3600, ran=2 * 3600 + 13 * 60, nodes=["n-37-04"],
           exit_code=1, log=traceback_text("value", lmft, pkg="lmft"))
    # a 2-GPU DDP run whose rank 1 hit the NCCL watchdog (cause chip NCCL; triage demo, §5.2.8)
    job = _job(c, 824373, "lmft-x07-ddp-s0", submit=t0 - 11 * 3600, **dict(sk, cpus=32))
    _ended(world, c, job, state="FAILED", started_ago=10 * 3600, ran=3 * 3600 + 41 * 60, nodes=["n-37-06"],
           exit_code=1, log=traceback_text("nccl", lmft, pkg="lmft"))
    job = _job(c, 824244, "orb-swp--train--voc-cosinewarm--s46--a01", submit=t0 - 30 * 3600, **orb)
    _ended(world, c, job, state="COMPLETED", started_ago=22 * 3600, ran=10 * 3600 + 51 * 60 + 24, nodes=["n-37-03"],
           log="done · best vs_base -1.93% · saved final.pt\n")
    _add_config(world, HostConfig(name="vega", ssh="vega", group="clusters", connect_timeout=120,
                                slurm=_slurm_settings(templates=("{workdir}/logs/{name}_{jobid}.out",))))


def _build_lyra(world: DemoWorld) -> None:
    t0 = world.t0
    c = SimCluster(name="lyra", slurm_version="26.05.3", login_host="login-a02", user="alice", uid=20001,
                   group="alice", gid=20001, tz=timedelta(hours=-7),
                   config=_slurm_config("lyra", "CANCEL", "use_interactive_step", "Alloc,Contain,X11",
                                        _priority("7-00:00:00", "NONE")),    # every weight 0: submit order
                   associations=(("lyra", "alice", "", "high,low,normal,standard-mem"),
                                 ("lyra", "pilab_02", "", "high,low,normal,standard-mem"),
                                 ("lyra", "pilab_02_gpu", "", "gpu-highmem,high,low,normal")),
                   tools=SLURM_TOOLS, gpu_tres=False, merged_stderr="empty", stat_verb="statx",
                   latency=(0.2, 0.3), ncpu=48, mem_total_kb=395_000_000, load=11.4, next_job_id=32378161,
                   boot=t0 - 64 * 86400, node_ssh="denied", gpu_buses=("71", "01", "e1", "a1"))
    nodes = [SimNode(f"lyra-gpu-36-{k:02d}", ("free-gpu",), 40, 192000, (GresDef("V100", 4),),
                     bg_cpus=40 if k % 3 else 37, bg_mem_mib=143360, bg_gpu_idx={0, 1, 2, 3} if k % 4 else {0, 1, 2},
                     flag="-" if k == 5 else "", free_mem_mib=24485 - 1000 * k) for k in range(8)]
    nodes += [SimNode(f"lyra-gpu-q71-{k:02d}", ("free-gpu",), 32, 256000, (GresDef("A30", 4),),
                      bg_cpus=20, bg_mem_mib=98304, bg_gpu_idx={1, 2} if k != 1 else {2, 3},
                      free_mem_mib=88000 + 3000 * k) for k in range(4)]
    nodes += [SimNode(f"lyra-gpu-s71-{k:02d}", ("free-gpuxl",), 32, 256000, (GresDef("RTX6000", 4),),
                      bg_cpus=24, bg_mem_mib=147456, bg_gpu_idx={0, 1, 2}, free_mem_mib=41000) for k in range(4)]
    nodes += [SimNode(f"lyra-gpu-r71-{k:02d}", ("free-gpuxl",), 32, 256000, (GresDef("RTX6000", 4),),
                      bg_cpus=8, bg_mem_mib=49152, bg_gpu_idx={0}, free_mem_mib=161000) for k in range(3)]
    nodes.append(SimNode("lyra-gpu-p71-08", ("free-gpuxl",), 48, 1031000, (GresDef("L40S", 4),),
                         free_mem_mib=315712))
    c.nodes = nodes
    world.hosts["lyra"] = c
    proj = _PROJECT["lyra"]
    base = dict(partition="free-gpu", account="alice", qos="low", time_limit_s=3 * 86400, gpus_per_node=1,
                stderr_pattern=None, duration_s=10 * 3600)
    job = _job(c, 32378138, "orb-swp--train--places-warmup--s45--a01", submit=t0 - 3 * 3600, gpu_type="A30",
               typed_request=False, typed_alloc=False, util_target=88.0, mem_target_mib=11800,
               submit_args="places warmup 45", **base)
    _run(world, c, job, started_ago=2 * 3600 + 41 * 60, nodes=["lyra-gpu-q71-01"], gpu_idx=[[0]],
         cpu_ids=["0-7"])
    # throttled array: tasks 1-2 done, 3-4 running, 5-12 pending (%2)
    arr = 32377076
    sweep = dict(base, partition="free-gpu", gpu_type=None, typed_request=False, typed_alloc=False,
                 stdout_pattern=f"{proj}/logs/slurm-%A_%a.out", duration_s=40 * 60, util_target=76.0,
                 mem_target_mib=7300, submit_args="sweep.yaml")
    for task, raw, ago in ((3, 32377079, 21 * 60), (4, 32377080, 9 * 60)):
        job = _job(c, raw, "orb-sweep", submit=t0 - 2 * 3600, array_job_id=arr, array_task=str(task),
                   command=f"{proj}/scripts/sweep.sbatch", **sweep)
        _run(world, c, job, started_ago=ago, nodes=[f"lyra-gpu-r71-{task - 3:02d}"], gpu_idx=[[1]],
             cpu_ids=["8-15"])
    for task, raw, ago, ran in ((1, 32377077, 100 * 60, 38 * 60), (2, 32377078, 62 * 60, 41 * 60)):
        job = _job(c, raw, "orb-sweep", submit=t0 - 2 * 3600, array_job_id=arr, array_task=str(task),
                   command=f"{proj}/scripts/sweep.sbatch", **sweep)
        _ended(world, c, job, state="COMPLETED", started_ago=ago, ran=ran, nodes=["lyra-gpu-r71-00"],
               log="sweep point done\n")
    _job(c, arr, "orb-sweep", submit=t0 - 2 * 3600, array_job_id=arr, array_task="5-12%2",
         summary=True, array_pending=list(range(5, 13)), array_throttle=2, reason="JobArrayTaskLimit", start=t0,
         command=f"{proj}/scripts/sweep.sbatch", **sweep)
    rtx = dict(base, partition="free-gpuxl", gpu_type="RTX6000")
    job = _job(c, 32354789, "orb-swp--train--places-cosinewarm--s44--a01", submit=t0 - 19 * 3600, **rtx)
    _ended(world, c, job, state="PREEMPTED", started_ago=18 * 3600, ran=64, nodes=["lyra-gpu-s71-01"],
           log=f"slurmstepd: error: *** JOB 32354789 ON lyra-gpu-s71-01 CANCELLED AT "
               f"{local_ts(t0 - 18 * 3600 + 64, c.tz, '%Y-%m-%dT%H:%M:%S')} DUE TO PREEMPTION ***\n")
    for k, (method, seed, ran) in enumerate((("warmup", 49, 150), ("const", 49, 148))):
        job = _job(c, 32327454 + k, f"orb-swp--train--{method}--s{seed}--a01", submit=t0 - 40 * 3600 - 1200, **rtx)
        _ended(world, c, job, state="FAILED", started_ago=40 * 3600, ran=ran, nodes=["lyra-gpu-r71-00"],
               exit_code=2, log=f"python: can't open file '{proj}/train_{method}.py': [Errno 2] "
                                "No such file or directory\n")
    job = _job(c, 32303059, "orb-swp--train--warmup--s43--a02", submit=t0 - 60 * 3600, **rtx)
    _ended(world, c, job, state="COMPLETED", started_ago=58 * 3600, ran=13 * 3600 + 35 * 60 + 11,
           nodes=["lyra-gpu-s71-03"], log="done\n")
    job = _job(c, 32301780, "orb-swp--train--adam--s43--a02", submit=t0 - 62 * 3600, **rtx)
    _ended(world, c, job, state="CANCELLED", started_ago=None, ran=1030)
    job.cancelled_by = c.uid
    _add_config(world, HostConfig(name="lyra", ssh="lyra", group="clusters"))


def _proc(ws: SimWorkstation, t0: float, pid: int, ppid: int, args: str, *, ago: float, user: str | None = None,
          pgid: int | None = None, **kw: Any) -> SimProc:
    """A scenario process that has been running for `ago` seconds at t0."""
    p = SimProc(pid=pid, ppid=ppid, pgid=pgid if pgid is not None else pid, user=user or ws.user,
                start=int(t0 - ago), args=args, **kw)
    ws.procs[pid] = p
    return p


def _task_log(world: DemoWorld, ws: SimWorkstation, path: str, started: float, header: str,
              trainer: Trainer | None) -> None:
    rng = random.Random(world.seed + stable_hash(path))
    text = header
    if trainer is not None:
        text += trainer.history(max(started, world.t0 - 3 * 3600), world.t0, ws.tz, rng, keep=60)
    lg = SimLog(path, started)
    lg.write(text, world.t0)
    ws.logs[path] = lg


def _build_labstation(world: DemoWorld) -> None:
    t0 = world.t0
    home = "/home/alice"
    ws = SimWorkstation(name="labstation", hostname="labstation.example.edu", user="alice", uid=1203,
                        tz=timedelta(hours=-7), ncpu=64, mem_total_kb=528_008_252, swap_total_kb=2_097_148,
                        swap_free_kb=2_061_820, tools=WS_TOOLS, latency=(0.06, 0.12), load=2.1, bg_cpu_pct=60.0,
                        boot=t0 - (117 * 86400 + 19 * 3600 + 34 * 60), mem_used_kb=45_000_000,
                        project_root=f"{home}/projects/orb", next_pid=600000)
    ws.cpu_jiffies = [2005215382, 1007428288, 384747761, 68893959522, 18544027, 0, 6257322, 0, 0, 0]
    for i, uuid in enumerate(("GPU-4e050371-ee67-0bb4-bab4-c813e9a744d2", "GPU-56d77d13-b29e-6b87-795d-377913f8ad57",
                              "GPU-f38d7a35-ee30-7409-6ef0-7e2de081d943", "GPU-d5f3487b-34d5-9bc6-16d5-f3d609021850")):
        ws.gpus.append(SimGPU(index=i, uuid=uuid, name="NVIDIA RTX A6000", mem_total_mib=49140, power_limit_w=300.0,
                              idle_power_w=(23.95, 27.88, 19.93, 24.0)[i], temp_c=(30, 31, 29, 30)[i]))
    world.hosts["labstation"] = ws
    _proc(ws, t0, 360796, 360588, "sshd: alice@pts/3", ago=68483, user="alice")
    _proc(ws, t0, 374177, 360796, "-zsh", ago=65956, stat="Ss")
    _proc(ws, t0, 374190, 374177, "tmux new -s orb", ago=65950, stat="S+")
    _proc(ws, t0, 374231, 374177, f"{home}/.local/share/uv/tools/nvitop/bin/python {home}/.local/bin/nvitop",
          ago=65953, stat="Sl+", cpu_pct=2.0, rss_kb=88_000)
    launch_log = f"{home}/projects/orb/logs/launch_voc_s44.log"
    env = {"VIRTUAL_ENV": f"{home}/projects/orb/.venv", "OMP_NUM_THREADS": "16", "WANDB_PROJECT": "orb-sweep"}
    _proc(ws, t0, 363942, 1, "python -m orb.launch --manifest manifests/voc_s44.txt --per-gpu 1 --threads 16",
          ago=68047, stat="SN", cwd=f"{home}/projects/orb", fd1=launch_log, fd2=launch_log, rss_kb=61_000, env=env)
    tr = Trainer(epochs=200, epoch_s=300.0, iters=1800)
    _proc(ws, t0, 363948, 363942, f"{home}/projects/orb/.venv/bin/python -m orb.train --config configs/exp_a.yaml data=voc "
          "sched=cosinewarm seed=44 run_dir=runs/voc/baseline-cosinewarm/s44", ago=68046, stat="SNl", cpu_pct=100.0,
          mem_pct=0.8, rss_kb=4_400_000, cwd=f"{home}/projects/orb", fd1=launch_log, fd2=launch_log,
          env=dict(env, CUDA_VISIBLE_DEVICES="3", WANDB_RUN_ID="voc-s44-cw"), gpu=3, gpu_mem_mib=21454,
          gpu_util=70.0, trainer=tr, item="voc/baseline-cosinewarm/s44")
    ws.runners[363942] = QueueRunner(root=363942, items=["voc/baseline-warmup/s44",
                                                         "voc/baseline-step/s44"], next_no=7, total=8,
                                     child_s=(5400.0, 9000.0))
    _task_log(world, ws, launch_log, t0 - 68047,
              f"[{local_ts(t0 - 68047, ws.tz)}] launch: manifest manifests/voc_s44.txt · 8 items · per-gpu 1\n"
              f"[{local_ts(t0 - 68046, ws.tz)}] launch: item 6/8 voc/baseline-cosinewarm/s44 "
              "on gpu 3 (pid 363948)\n", tr)
    # a standalone eval that crashes with a Traceback shortly after start (ends -> exit scan finds it -> FAILED)
    eval_log = f"{home}/projects/orb/logs/eval_ckpt_voc_s44.log"
    _proc(ws, t0, 590112, 1, "python scripts/eval_ckpt.py --run runs/voc/baseline-const/s44 --split test",
          ago=12 * 60, stat="RNl", cpu_pct=97.0, mem_pct=0.6, rss_kb=3_100_000, cwd=f"{home}/projects/orb",
          fd1=eval_log, fd2=eval_log, env=dict(env, CUDA_VISIBLE_DEVICES="2"), gpu=2, gpu_mem_mib=6120,
          gpu_util=35.0, end_at=t0 + 100, crash="missing_ckpt")
    _task_log(world, ws, eval_log, t0 - 12 * 60,
              f"[{local_ts(t0 - 12 * 60, ws.tz)}] eval_ckpt: run runs/voc/baseline-const/s44\n"
              f"[{local_ts(t0 - 11 * 60, ws.tz)}] eval_ckpt: scoring 4 checkpoints on split test\n"
              f"[{local_ts(t0 - 2 * 60, ws.tz)}] eval_ckpt: 3/4 done (last.pt: vs_base -1.88%)\n", None)
    # another user's GPU process (shown with `u`, never actionable)
    _proc(ws, t0, 2911042, 2911001, "python sample_ddpm.py --ckpt /data/carol/ddpm/cifar10/ema_0.9999.pt --n 50000",
          ago=5 * 3600 + 1200, user="carol", stat="Rl", cpu_pct=101.0, mem_pct=1.2, rss_kb=6_600_000, gpu=1,
          gpu_mem_mib=30722, gpu_util=88.0)
    _proc(ws, t0, 2318537, 1, f"{home}/.vscode-server/code-8a67639f74080f56c5aa328f676226e236a23e23 --cli-data-dir "
          f"{home}/.vscode-server/cli agent host", ago=2497055, stat="Ssl", rss_kb=140_000)
    for k in range(3):
        _proc(ws, t0, 2613940 + 9603 * k, 1, f"Xvfb :{99 + k} -screen 0 1280x800x24 -nolisten tcp -auth "
              f"/tmp/xvfb-run.{'rCOX6Jhw7Hd11XzWL8'[6 * k:6 * k + 6]}/Xauthority", ago=5319375 - 1000 * k,
              rss_kb=52_000)
    _update_gpus(ws, 600.0, random.Random(0))     # settle temperatures/power to the current load
    _add_config(world, HostConfig(name="labstation", ssh="labstation", group="workstations"))


def _build_lynx(world: DemoWorld) -> None:
    t0 = world.t0
    home = "/data/home/alice"
    ws = SimWorkstation(name="lynx", hostname="gpu-box.example.edu", user="alice", uid=56021,
                        tz=timedelta(hours=-7), ncpu=64, mem_total_kb=593_520_640, swap_total_kb=4_194_300,
                        swap_free_kb=4_194_300, tools=WS_TOOLS, latency=(0.05, 0.09), load=3.2, bg_cpu_pct=40.0,
                        boot=t0 - 19 * 86400, mem_used_kb=38_000_000, project_root=f"{home}/projects/orb",
                        next_pid=801500)
    ws.cpu_jiffies = [812004415, 9001221, 120044871, 9120044551, 2004481, 0, 1400223, 0, 0, 0]
    for i, uuid in enumerate(("GPU-c6faebeb-cf00-3f1a-1182-49cec7f3d256", "GPU-35765204-ac46-81e4-b43b-71a7d77aedd9")):
        ws.gpus.append(SimGPU(index=i, uuid=uuid, name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                              mem_total_mib=97887, power_limit_w=300.0, idle_power_w=19.0, fan=None,
                              temp_c=(55, 57)[i]))
    world.hosts["lynx"] = ws
    _proc(ws, t0, 639192, 1, "/usr/lib/systemd/systemd --user", ago=68453, stat="Ss", rss_kb=12_000)
    _proc(ws, t0, 639193, 639192, "(sd-pam)", ago=68453, pgid=639192)
    _proc(ws, t0, 639209, 639186, "sshd-session: alice@pts/0", ago=68453, pgid=639186)
    _proc(ws, t0, 641936, 639209, "-zsh", ago=67580, stat="Ss")
    _proc(ws, t0, 641995, 641936, f"{home}/.local/share/uv/tools/nvitop/bin/python {home}/.local/bin/nvitop",
          ago=67578, stat="Sl+", cpu_pct=0.7, rss_kb=90_000)
    launch_log = f"{home}/projects/orb/logs/launch_cif_s44.log"
    env = {"VIRTUAL_ENV": f"{home}/projects/orb/.venv", "WANDB_PROJECT": "orb-sweep"}
    _proc(ws, t0, 642229, 1, "python -m orb.launch --manifest manifests/cif_s44.txt --per-gpu 1", ago=67514,
          stat="SN", cwd=f"{home}/projects/orb", fd1=launch_log, fd2=launch_log, rss_kb=58_000, env=env)
    common = "configs/exp_a.yaml data=cifar"
    tr1 = Trainer(epochs=200, epoch_s=420.0, iters=2400)
    _proc(ws, t0, 642234, 642229, f"{home}/projects/orb/.venv/bin/python -m orb.train --config {common} sched=const "
          "seed=44 run_dir=runs/cifar/baseline-const/s44", ago=67514, stat="RNl", cpu_pct=121.0,
          mem_pct=1.9, rss_kb=11_200_000, cwd=f"{home}/projects/orb", fd1=launch_log, fd2=launch_log,
          env=dict(env, CUDA_VISIBLE_DEVICES="1"), gpu=1, gpu_mem_mib=15234, gpu_util=99.0, trainer=tr1,
          item="cifar/baseline-const/s44", pgid=642229)
    _proc(ws, t0, 786345, 642229, f"{home}/projects/orb/.venv/bin/python -m orb.train --config {common} "
          "sched=cosinewarm seed=44 run_dir=runs/cifar/baseline-cosinewarm/s44",
          ago=3105, stat="SNl", cpu_pct=114.0, mem_pct=1.9, rss_kb=11_100_000, cwd=f"{home}/projects/orb",
          fd1=launch_log, fd2=launch_log, env=dict(env, CUDA_VISIBLE_DEVICES="0"), gpu=0, gpu_mem_mib=15234,
          gpu_util=97.0, end_at=t0 + 8 * 60, item="cifar/baseline-cosinewarm/s44", pgid=642229,
          trainer=Trainer(epochs=200, epoch_s=60.0, iters=2400, epoch=193))
    ws.runners[642229] = QueueRunner(root=642229, items=["cifar/baseline-warmup/s44",
                                                         "cifar/baseline-step/s44", "cifar/baseline-poly/s44"],
                                     next_no=5, total=7, child_s=(2400.0, 4200.0))
    _task_log(world, ws, launch_log, t0 - 67514,
              f"[{local_ts(t0 - 67514, ws.tz)}] launch: manifest manifests/cif_s44.txt · 7 items · per-gpu 1\n"
              f"[{local_ts(t0 - 3105, ws.tz)}] launch: item 4/7 cifar/baseline-cosinewarm/s44 "
              "on gpu 0 (pid 786345)\n", tr1)
    _update_gpus(ws, 600.0, random.Random(0))     # settle temperatures/power to the current load
    _add_config(world, HostConfig(name="lynx", ssh="lynx", group="workstations"))


# ---------------------------------------------------------------- notifications showcase (§7.14)
DEMO_WATCHED = "k2:3371232"          # the eval job that completes ~2 min in: a watched job's end notifies
DEMO_AWAY_S = 8 * 3600               # "since you were away": the last visit, 8 h before the demo starts


def demo_ui_state(now: datetime) -> dict[str, Any]:
    """In-memory ui.json for --demo: one watched job and a last visit, so alerts and `D` show up at once."""
    return {"watch": {DEMO_WATCHED: now.isoformat(timespec="seconds")},
            "last_seen": (now - timedelta(seconds=DEMO_AWAY_S)).isoformat(timespec="seconds")}


def _build_transitions(world: DemoWorld) -> None:
    """Jobs that change state in the first minutes, so notifications show up in --demo: an lyra eval runs low on
    time ~40 s in and times out ~13 min in, an lyra training fails ~80 s in, and a k2 job that was requeued once
    is preempted again ~2.5 min in (PreemptMode REQUEUE: Slurm requeues it, its restart count goes to 2)."""
    t0 = world.t0
    lyra, k2 = world.cluster("lyra"), world.cluster("k2")
    base = dict(partition="free-gpu", account="alice", qos="low", gpus_per_node=1, gpu_type="A30",
                typed_request=False, typed_alloc=False, stderr_pattern=None, mem_target_mib=9800)
    limit = 2 * 3600
    job = _job(lyra, 32378151, "orb-swp--eval--places-warmup--s45", submit=t0 - 2 * 3600, time_limit_s=limit,
               duration_s=3 * 3600, util_target=71.0, **base)
    _run(world, lyra, job, started_ago=limit - 12 * 60 - 40, nodes=["lyra-gpu-q71-02"], gpu_idx=[[0]],
         cpu_ids=["0-7"])
    ran = 47 * 60
    job = _job(lyra, 32378152, "orb-swp--train--places-adam--s46--a01", submit=t0 - 3600, time_limit_s=3 * 86400,
               duration_s=ran + 75, outcome="FAILED", exit_code=1, failure="value", util_target=83.0, **base)
    _run(world, lyra, job, started_ago=ran, nodes=["lyra-gpu-q71-03"], gpu_idx=[[0]], cpu_ids=["8-15"])
    ran = 3 * 3600 + 20 * 60
    job = _job(k2, 3372115, "orb-swp--train--voc-warmup--s44--a01", partition="gpu-preempt", submit=t0 - 6 * 3600,
               time_limit_s=86400, gpus_per_node=1, gpu_type="nvidia_geforce_rtx_3090", duration_s=ran + 150,
               outcome="PREEMPTED", restarts=1, stderr_pattern=f"{_PROJECT['k2']}/logs/%x_%j.out",
               util_target=93.0, mem_target_mib=17800)
    _run(world, k2, job, started_ago=ran, nodes=["n-32-01"], gpu_idx=[[6]], cpu_ids=["48-55"])
