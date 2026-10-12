"""Efficiency card (§5.2.8): how much of the requested time, CPUs, memory and GPUs a Slurm job actually used.

Pure and shared (reportseff / seff style): the Slurm collector puts the plain lines on ``JobDetail.efficiency``; job
detail renders the same rows with colours. Inputs are only what the detail probe already holds — the job's
Elapsed / Timelimit / AllocCPUS / AllocTRES mem and the steps' TotalCPU (when this sacct knows it), AveCPU × NTasks,
MaxRSS and ``gres/gpuutil``. Finished jobs get used / requested and a "next time" line with a +25 % margin; running
jobs get live headroom (sstat MaxRSS against the requested memory) instead. Unknown values read ``–``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from runfleet.models import Job, JobKind, JobState, JobStep, StateGroup
from runfleet.util import humanize
from runfleet.util.slurmfmt import parse_float, parse_mem_bytes

Level = Literal["ok", "warn", "bad", "none"]

MARGIN = 1.25                     # "next time" = used × 1.25, rounded up
CPU_LOW, MEM_LOW, TIME_LOW = 50.0, 25.0, 25.0     # reportseff colouring: below these the request was oversized
GPU_LOW = 50.0
GPU_IDLE = 5.0                    # mean GPU util below this: the "GPU idle" note (the stall hint's threshold)
LIVE_MEM_HIGH = 90.0              # running: RSS above this share of --mem is close to an OOM kill
_GIB = 1024 ** 3


@dataclass(frozen=True, slots=True)
class Efficiency:
    live: bool                        # a running job: headroom, no "next time"
    elapsed_s: float | None
    limit_s: float | None
    cpus: int | None
    cpu_time_s: float | None          # TotalCPU (or Σ AveCPU × NTasks); running: None
    cpu_pct: float | None             # cpu_time / (elapsed × cpus); running: sstat average / cpus
    mem_peak: int | None              # bytes: max over steps of MaxRSS × tasks (single node), running: sstat MaxRSS
    mem_req: int | None               # bytes requested (AllocTRES / ReqMem)
    gpus: int
    gpu_pct: float | None             # mean util per GPU (sacct gres/gpuutil Tot / GPUs; running: sstat)
    next_time: str | None             # "6:00:00"
    next_mem: str | None              # "24G"
    note: str | None = None           # why there is no "next time", or the GPU-idle warning
    completed: bool = False           # COMPLETED: the time used is what a run needs (time colouring, suggestions)

    @property
    def time_pct(self) -> float | None:
        return _pct(self.elapsed_s, self.limit_s)

    @property
    def mem_pct(self) -> float | None:
        return _pct(self.mem_peak, self.mem_req)


def _pct(used: float | None, total: float | None) -> float | None:
    return used / total * 100 if used is not None and total else None


def _secs(td: object) -> float | None:
    return td.total_seconds() if td is not None else None   # type: ignore[attr-defined]


def _work_steps(steps: Sequence[JobStep]) -> list[JobStep]:
    """Steps that did the work: the batch step and numbered steps (not the allocation row, not `.extern`, whose
    AveCPU is garbage)."""
    return [s for s in steps if "." in s.step_id and not s.step_id.endswith(".extern")]


def cpu_time_s(steps: Sequence[JobStep]) -> float | None:
    """CPU seconds the job used: the allocation row's TotalCPU (= the sum over steps), else the steps' TotalCPU,
    else Σ AveCPU × NTasks over the work steps."""
    alloc = next((s for s in steps if "." not in s.step_id and s.total_cpu is not None), None)
    if alloc is not None and alloc.total_cpu is not None:
        return alloc.total_cpu.total_seconds()
    work = _work_steps(steps)
    totals = [s.total_cpu.total_seconds() for s in work if s.total_cpu is not None]
    if totals:
        return sum(totals)
    aves = [s.ave_cpu.total_seconds() * max(1, s.ntasks or 1) for s in work if s.ave_cpu is not None]
    return sum(aves) if aves else None


def mem_peak(job: Job, steps: Sequence[JobStep]) -> int | None:
    """Peak memory: max over the work steps of MaxRSS (per task, sampled) × the step's tasks when the job has one
    node (all its tasks share it: an upper bound), else the per-task MaxRSS."""
    single = (job.alloc.nodes or len(job.nodes) or 1) == 1
    peaks = [s.max_rss_bytes * (max(1, s.ntasks or 1) if single else 1) for s in _work_steps(steps)
             if s.max_rss_bytes is not None]
    return max(peaks) if peaks else None


def mem_req(job: Job) -> int | None:
    return job.alloc.mem_bytes or parse_mem_bytes(job.raw.get("ReqMem") or job.raw.get("MinMemory"))


def gpu_mean_pct(job: Job, steps: Sequence[JobStep]) -> float | None:
    """Mean util per allocated GPU: the busiest work step's ``TRESUsageInTot gres/gpuutil`` (sum over its tasks, a
    task's value summed over its GPUs) / GPUs, clamped to 100 — recorded on k2 / vega, absent on lyra."""
    if not job.alloc.gpus:
        return None
    vals = [parse_float(s.tres_usage_tot.get("gres/gpuutil")) for s in _work_steps(steps)]
    known = [v for v in vals if v is not None]
    return min(100.0, max(known) / job.alloc.gpus) if known else None


def round_time(seconds: float, cap: float | None = None) -> str:
    """A --time value rounded up: 15 min steps below 1 h, whole hours below 1 day, then 6 h steps (``0:45:00``,
    ``6:00:00``, ``1-06:00:00``); never above ``cap`` (the request that was enough: then that, as it was)."""
    s = max(900.0, seconds)
    step = 900 if s <= 3600 else 3600 if s <= 86400 else 6 * 3600
    rounded = math.ceil(s / step) * step
    return humanize.clock(cap if cap is not None and rounded > cap else rounded)


def round_mem(n_bytes: float, cap: float | None = None) -> str:
    """A --mem value rounded up: whole GiB to 16G, then 4G steps to 64G, then 8G steps (``1G``, ``24G``, ``72G``);
    never above ``cap`` (then the request itself, in whole MiB unless it is whole GiB)."""
    g = max(1.0, n_bytes / _GIB)
    step = 1 if g <= 16 else 4 if g <= 64 else 8
    rounded = math.ceil(g / step - 1e-9) * step
    if cap is not None and rounded * _GIB > cap:
        return f"{round(cap / _GIB)}G" if cap % _GIB == 0 else f"{math.ceil(cap / (1 << 20))}M"
    return f"{rounded}G"


def efficiency(job: Job, steps: Sequence[JobStep]) -> Efficiency | None:
    """The card of a Slurm job that ran (None for tasks and jobs that never started)."""
    if job.kind is not JobKind.SLURM or job.state.group is StateGroup.QUEUED or job.start_time is None and (
            job.elapsed is None or not job.elapsed.total_seconds()):
        return None
    cpus = job.alloc.cpus or None
    elapsed = _secs(job.elapsed)
    limit = _secs(job.time_limit)
    req = mem_req(job)
    gpus = job.alloc.gpus
    if not job.state.is_terminal:
        cpu = job.cpu_pct / cpus if job.cpu_pct is not None and cpus else None
        util = job.gpu.util_mean_pct if job.gpu is not None else None
        rss = job.rss_bytes if job.rss_bytes is not None else mem_peak(job, steps)   # sstat, or sstat-filled steps
        return Efficiency(True, elapsed, limit, cpus, None, cpu, rss, req, gpus, util, None, None)
    used_cpu = cpu_time_s(steps)
    cpu_pct = used_cpu / (elapsed * cpus) * 100 if used_cpu is not None and elapsed and cpus else None
    peak = mem_peak(job, steps)
    util = gpu_mean_pct(job, steps)
    next_time = next_mem = note = None
    single = (job.alloc.nodes or len(job.nodes) or 1) == 1
    if job.state is JobState.COMPLETED:
        if elapsed:
            want = elapsed * MARGIN
            next_time = round_time(want, cap=limit or None)
        if peak is not None and single:
            want_b = peak * MARGIN
            next_mem = round_mem(want_b, cap=req or None)
        elif not single:
            note = "multi-node: --mem is per node, MaxRSS per task"
    elif job.state is JobState.TIMEOUT and limit:
        next_time = round_time(limit * MARGIN)
        note = "hit the limit: needs more; +25 % is a guess"
    elif job.state is JobState.OUT_OF_MEMORY and req:
        next_mem = round_mem(req * 1.5)
        note = "killed at --mem: needs more; +50 % is a guess"
    else:
        note = "did not finish: no request from a partial run"
    if util is not None and util < GPU_IDLE:
        note = f"GPU idle (mean {humanize.pct(util)}): CUDA used? else fewer GPUs"
    return Efficiency(False, elapsed, limit, cpus, used_cpu, cpu_pct, peak, req, gpus, util, next_time, next_mem,
                      note, job.state is JobState.COMPLETED)


def _level_low(value: float | None, low: float) -> Level:
    return "none" if value is None else "warn" if value < low else "ok"


def rows(eff: Efficiency) -> list[tuple[str, str, Level]]:
    """``(label, text, level)`` rows; ``warn`` = an oversized request (finished) or close to the limit (running)."""
    out: list[tuple[str, str, Level]] = []
    t = humanize.clock(eff.elapsed_s, dash=humanize.NA)
    if eff.limit_s is not None:
        t += f" of {humanize.clock(eff.limit_s)} ({humanize.pct(eff.time_pct)})"
    else:
        t += " (no limit)"
    tl: Level = _level_low(eff.time_pct, TIME_LOW) if eff.completed else "none"
    out.append(("time", t, tl))
    cpus = f"{eff.cpus} core{'s' if eff.cpus != 1 else ''}" if eff.cpus else "? cores"
    c = f"{humanize.pct(eff.cpu_pct)} of {cpus}"
    if eff.cpu_time_s is not None:
        c += f" (CPU time {humanize.clock(eff.cpu_time_s)})"
    elif eff.live and eff.cpu_pct is not None:
        c += " (sstat average)"
    out.append(("cpu", c, _level_low(eff.cpu_pct, CPU_LOW) if not eff.live else "none"))
    if eff.live:
        m = f"rss {humanize.size(eff.mem_peak)} of {humanize.size(eff.mem_req)} ({humanize.pct(eff.mem_pct)})"
        if eff.mem_peak is not None and eff.mem_req:
            m += f" · {humanize.size(max(0, eff.mem_req - eff.mem_peak))} headroom"
        ml: Level = "none" if eff.mem_pct is None else "bad" if eff.mem_pct >= LIVE_MEM_HIGH else "ok"
    else:
        m = f"peak {humanize.size(eff.mem_peak)} of {humanize.size(eff.mem_req)} ({humanize.pct(eff.mem_pct)})"
        ml = _level_low(eff.mem_pct, MEM_LOW)
    out.append(("mem", m, ml))
    if eff.gpus:
        src = "sstat" if eff.live else "sacct gres/gpuutil"
        g = (f"{humanize.pct(eff.gpu_pct)} mean over {eff.gpus} GPU{'s' if eff.gpus != 1 else ''} ({src})"
             if eff.gpu_pct is not None else f"{humanize.NA} (gres/gpuutil not recorded)")
        gl: Level = "bad" if eff.gpu_pct is not None and eff.gpu_pct < GPU_IDLE else _level_low(eff.gpu_pct, GPU_LOW)
        out.append(("gpu", g, gl))
    if not eff.live:
        bits = [f"--time={eff.next_time}" if eff.next_time else "", f"--mem={eff.next_mem}" if eff.next_mem else ""]
        nxt = " ".join(b for b in bits if b)
        out.append(("next", nxt or humanize.NA, "none"))
    if eff.note:
        out.append(("note", eff.note, "warn" if eff.note.startswith("GPU idle") else "none"))
    return out


def lines(eff: Efficiency | None) -> tuple[str, ...]:
    """Plain text of the card (``JobDetail.efficiency``, support bundle): ``time  5:12:03 of 24:00:00 (22%)``."""
    if eff is None:
        return ()
    return tuple(f"{label:<5} {text}" for label, text, _level in rows(eff))
