"""[WP2] sstat usage semantics: TRESUsageInTot sums tasks, Max is per task (§5.2.3, Appendix C #1)."""

from __future__ import annotations

import pytest

from runfleet.backends.slurm.commands import SSTAT_FMT
from runfleet.backends.slurm.parse_sstat import job_usage, parse_sstat, pick_step
from tests.backends.factories import NOW, format_fields
from tests.conftest import meta, sample


def rows(host: str) -> list[object]:
    pr = parse_sstat(sample(host, "sstat_running"), format_fields(meta(host, "sstat_running")["cmd"]))
    assert not pr.bad_lines
    return pr.items


def test_k2_single_gpu() -> None:
    usage = job_usage(rows("k2"), gpus=1, gpu_type="nvidia_a100-sxm4-40gb", gpu_tres=True, sampled_at=NOW,
                      interval_s=60)
    assert usage is not None and usage.gpu is not None
    assert usage.row.step == "batch"                      # extern ignored (garbage AveCPU)
    g = usage.gpu
    assert g.util_mean_pct == 98 and g.util_sum_pct == 98 and g.util_peak_pct == 99
    assert g.mem_used_mib == 24870 and g.mem_total_mib == 40960
    assert g.n_gpus == 1 and g.source == "sstat" and "60 s" in g.note
    assert usage.ave_cpu is not None and usage.ave_cpu.total_seconds() == 4 * 3600 + 20 * 60 + 2
    assert usage.max_rss_bytes == 11506244 * 1024


def test_vega_two_gpus_sum_semantics() -> None:
    usage = job_usage(rows("vega"), gpus=2, gpu_type="nvidia_h200_nvl", gpu_tres=True, sampled_at=NOW,
                      interval_s=60)
    assert usage is not None and usage.gpu is not None
    g = usage.gpu
    assert g.util_sum_pct == 100 and g.util_mean_pct == 50
    assert g.util_peak_pct == pytest.approx(84.5)          # Max 169 is per task, i.e. summed over its 2 GPUs
    assert g.mem_total_mib == 2 * 143771


def test_synthetic_four_task_step_uses_tot() -> None:
    line = ("9.0|4|01:00:00|1G|2G|gres/gpuutil=90,gres/gpumem=1000M|gres/gpuutil=95,gres/gpumem=1100M|"
            "gres/gpuutil=360,gres/gpumem=4000M\n"
            "9.batch|1|00:01:00|1G|1G|gres/gpuutil=0,gres/gpumem=10M|gres/gpuutil=0|gres/gpuutil=0,gres/gpumem=10M\n"
            "9.extern|1|213503982334-14:25:51|||energy=0|energy=0|energy=0")
    pr = parse_sstat(line)
    assert pick_step(pr.items).step == "0"                 # largest NTasks among steps with GPU memory
    usage = job_usage(pr.items, gpus=4, gpu_type=None, gpu_tres=True, sampled_at=NOW, interval_s=60)
    assert usage is not None and usage.gpu is not None
    assert usage.gpu.util_mean_pct == 90                   # Ave (90) would have under-reported 4×
    assert usage.gpu.util_peak_pct == 95
    assert usage.gpu.mem_used_mib == 4000 and usage.gpu.mem_total_mib is None


def test_no_gpu_tres_keeps_cpu_numbers() -> None:
    usage = job_usage(rows("k2"), gpus=1, gpu_type="a30", gpu_tres=False, sampled_at=NOW, interval_s=60)
    assert usage is not None and usage.gpu is not None
    assert usage.gpu.util_mean_pct is None and "does not record" in usage.gpu.note
    assert usage.ave_cpu is not None
    cpu_only = job_usage(rows("k2"), gpus=0, gpu_type=None, gpu_tres=True, sampled_at=NOW, interval_s=60)
    assert cpu_only is not None and cpu_only.gpu is None and cpu_only.max_rss_bytes


def test_lyra_empty_and_default_fields() -> None:
    assert sample("lyra", "sstat_running") == ""
    assert parse_sstat("").items == []
    assert len(SSTAT_FMT.split(",")) == 8
    assert job_usage([], gpus=1, gpu_type=None, gpu_tres=None, sampled_at=NOW, interval_s=60) is None


def test_bad_rows() -> None:
    pr = parse_sstat("x|y\nnot.a|row|a|b|c|d|e|f")
    assert len(pr.bad_lines) == 2 and pr.mostly_bad
