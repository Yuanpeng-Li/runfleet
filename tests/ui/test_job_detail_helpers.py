"""Job detail helpers: "Why it ended" bullets (log scan, no duplicate exit, merged log once), the cores estimate of
multi-node jobs, and the exit-log scan."""

from __future__ import annotations

from datetime import timedelta

from runfleet.models import JobDetail, JobState, JobStep
from runfleet.ui.screens import job_detail as jd
from runfleet.util import humanize
from runfleet.util.logscan import exit_cause
from tests.ui.fakes import T0, slurm_job

OOM_TAIL = [
    "epoch 3 step 1200 loss 0.41",
    "Traceback (most recent call last):",
    '  File "/w/train.py", line 88, in <module>',
    "    loss.backward()",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.",
]


def detail(job, steps=(), explanation=()):  # type: ignore[no-untyped-def]
    return JobDetail(job=job, fields={}, per_node=(), steps=tuple(steps), logs=job.logs, gpu=job.gpu, procs=(),
                     procs_note="", env={}, explanation=tuple(explanation), fetched_at=T0, errors=())


def step(sid: str, ntasks: int | None, ave_cpu: timedelta | None, elapsed: timedelta) -> JobStep:
    return JobStep(sid, sid.rpartition(".")[2], JobState.RUNNING, "RUNNING", 0, 0, elapsed, T0 - elapsed, None,
                   "n-37-[01-02]", ntasks, None, ave_cpu)


def test_exit_cause_prefers_the_exception_line() -> None:
    assert exit_cause(OOM_TAIL) == "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB."
    assert exit_cause(["ok", "Traceback (most recent call last):"]) == "Traceback (most recent call last):"
    assert exit_cause(["all good", "done"]) is None
    assert exit_cause(["/bin/sh: line 1: 1234 Killed  python x.py"]) is not None


def test_why_it_ended_uses_log_scan_without_duplicates() -> None:
    job = slurm_job("k2", "2330539", "lmft-x0-lr1e-5", JobState.FAILED, elapsed=timedelta(hours=1),
                    ended_ago=3600, exit_code=1, hints=("exit 1",), node="n-38-01")
    d = detail(job, explanation=("batch step exited with code 1 (the job script / program failed)",))
    lines = [b.plain for b in jd.why_bullets(job, d, exit_cause(OOM_TAIL))]
    assert lines[0].startswith("• log: torch.OutOfMemoryError: CUDA out of memory")
    assert not any(line == "• exit 1" for line in lines)                      # said by the explanation already
    logs = [line for line in lines if "/logs/" in line]
    assert len(logs) == 1 and logs[0].startswith("• stdout+stderr: ") and logs[0].endswith("(merged)")
    done = slurm_job("k2", "1", "ok", JobState.COMPLETED, elapsed=timedelta(hours=1), ended_ago=60, exit_code=0)
    assert not any("log:" in b.plain for b in jd.why_bullets(done, detail(done), "Killed"))


def test_multi_node_cores_from_the_busy_step_and_no_999_clamp() -> None:
    """2-node DDP: the batch step is the launcher; the torchrun step (2 tasks) carries the CPU time."""
    el = timedelta(hours=1, minutes=12)
    job = slurm_job("k2", "2334607", "ddp-scale--n2", JobState.RUNNING, elapsed=el, cpus=64, node="n-37-01",
                    util=91, mem_used_gib=10, mem_total_gib=80)
    job = job.__class__(**{**{f: getattr(job, f) for f in job.__dataclass_fields__}, "cpu_pct": 1240.0})
    d = detail(job, steps=[step("2334607.batch", 1, timedelta(hours=1, minutes=15), el),
                           step("2334607.extern", 2, None, el),
                           step("2334607.0", 2, timedelta(hours=7, minutes=26, seconds=24), el)])
    gpu = " ".join(t.plain for t in jd.gpu_lines(job, d, True, 100, (30, 70), (30, 70)))
    assert "× 2 tasks (step 0)" in gpu and "≈ 12.40 of 64 cores" in gpu
    box = " ".join(t.plain for t in jd.job_box_lines(job, d, T0, 100))
    assert "≈ 12 cores avg" in box and "999%" not in box
    assert humanize.cpu_load(98.4) == "98%" and humanize.cpu_load(6100) == "≈ 61 cores"
