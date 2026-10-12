"""[WP2] sstat parser: job GPU/CPU usage from TRESUsageIn*.

Semantics (§5.2.3, Appendix C #1): `TRESUsageInTot gres/gpuutil` is the sum over the step's tasks, and one task's
value is the sum over that task's GPUs (vega 2-GPU job: Max 169). Hence

    util_sum  = Tot gpuutil
    util_mean = min(100, util_sum / alloc.gpus)
    util_peak = min(100, Max gpuutil × NTasks / alloc.gpus)      (Max is per task; shown as "peak", estimated)
    mem_used  = Tot gpumem

Fixtures predate the Tot field: without it Tot = Ave × NTasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from runfleet.backends.common import ParseResult
from runfleet.backends.common.gpu_types import mem_mib
from runfleet.backends.slurm.commands import SSTAT_FMT
from runfleet.backends.slurm.parse_sacct import field_names
from runfleet.models import JobGpuUsage
from runfleet.util.slurmfmt import parse_duration, parse_float, parse_int, parse_mem_bytes, parse_mem_mib, parse_tres
from runfleet.util.slurmfmt import split_rows


@dataclass(frozen=True, slots=True)
class SstatRow:
    job_id_raw: str
    step: str                    # "batch", "extern", "0", ...
    ntasks: int
    ave_cpu: timedelta | None
    ave_rss_bytes: int | None
    max_rss_bytes: int | None
    tres_ave: dict[str, str]
    tres_max: dict[str, str]
    tres_tot: dict[str, str]

    def tot(self, key: str) -> float | None:
        """Tot value of a numeric TRES (`gres/gpuutil`), falling back to Ave × NTasks."""
        conv = parse_mem_mib if key == "gres/gpumem" else parse_float
        if key in self.tres_tot:
            return conv(self.tres_tot[key])
        ave = conv(self.tres_ave.get(key))
        return None if ave is None else ave * self.ntasks

    def has_gpu_keys(self) -> bool:
        return any("gres/gpuutil" in d for d in (self.tres_ave, self.tres_max, self.tres_tot))


@dataclass(frozen=True, slots=True)
class StepUsage:
    """The step chosen for a job plus its derived numbers."""
    row: SstatRow
    gpu: JobGpuUsage | None
    ave_cpu: timedelta | None
    max_rss_bytes: int | None


def parse_sstat(text: str, fields: Sequence[str] = tuple(SSTAT_FMT.split(","))) -> ParseResult[SstatRow]:
    names = field_names(fields)
    pr: ParseResult[SstatRow] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != len(names):
            pr.bad(n, line, f"expected {len(names)} fields")
            continue
        row = dict(zip(names, parts, strict=True))
        jid, _, step = row["JobID"].strip().partition(".")
        ntasks = parse_int(row.get("NTasks"))
        if not jid.isdigit() or not step or ntasks is None:
            pr.bad(n, line, "bad JobID/NTasks", jid if jid.isdigit() else None)
            continue
        pr.items.append(SstatRow(
            job_id_raw=jid, step=step, ntasks=ntasks, ave_cpu=parse_duration(row.get("AveCPU")),
            ave_rss_bytes=parse_mem_bytes(row.get("AveRSS"), default_unit=""),
            max_rss_bytes=parse_mem_bytes(row.get("MaxRSS"), default_unit=""),
            tres_ave=parse_tres(row.get("TRESUsageInAve")), tres_max=parse_tres(row.get("TRESUsageInMax")),
            tres_tot=parse_tres(row.get("TRESUsageInTot")),
        ))
    return pr


def pick_step(rows: Sequence[SstatRow]) -> SstatRow | None:
    """Among non-extern steps with gres/gpumem > 0: largest NTasks, ties → largest Tot gpumem (batch for plain
    jobs, `.0` for srun/torchrun jobs). Without GPU memory: the non-extern step with the most tasks (batch first)."""
    candidates = [r for r in rows if r.step != "extern"]
    gpu_steps = [r for r in candidates if (r.tot("gres/gpumem") or 0) > 0]
    if gpu_steps:
        return max(gpu_steps, key=lambda r: (r.ntasks, r.tot("gres/gpumem") or 0))
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r.ntasks, r.step == "batch"))


def job_usage(rows: Sequence[SstatRow], *, gpus: int, gpu_type: str | None, gpu_tres: bool | None,
              sampled_at: datetime | None, interval_s: float) -> StepUsage | None:
    """Usage of one job from its sstat rows. GPU numbers only when the cluster records gres/gpuutil
    (`gpu_tres` is not False) and the job has GPUs; AveCPU/MaxRSS always."""
    row = pick_step(rows)
    if row is None:
        return None
    gpu: JobGpuUsage | None = None
    if gpus > 0:
        per_gpu_total = mem_mib(gpu_type)
        total = per_gpu_total * gpus if per_gpu_total is not None else None
        if gpu_tres is False or not row.has_gpu_keys():
            note = ("n/a — this cluster does not record GPU usage" if gpu_tres is False
                    else "no GPU usage recorded for this job yet")
            gpu = JobGpuUsage(source="sstat", n_gpus=gpus, util_mean_pct=None, util_sum_pct=None, util_peak_pct=None,
                              mem_used_mib=None, mem_total_mib=total, sampled_at=sampled_at, note=note)
        else:
            util_sum = row.tot("gres/gpuutil")
            peak_task = parse_float(row.tres_max.get("gres/gpuutil"))
            gpu = JobGpuUsage(
                source="sstat", n_gpus=gpus,
                util_mean_pct=None if util_sum is None else min(100.0, util_sum / gpus),
                util_sum_pct=util_sum,
                util_peak_pct=None if peak_task is None else min(100.0, peak_task * row.ntasks / gpus),
                mem_used_mib=row.tot("gres/gpumem"), mem_total_mib=total, sampled_at=sampled_at,
                note=f"Slurm accounting · per job · refreshed every {int(interval_s)} s")
    return StepUsage(row=row, gpu=gpu, ave_cpu=row.ave_cpu, max_rss_bytes=row.max_rss_bytes)
