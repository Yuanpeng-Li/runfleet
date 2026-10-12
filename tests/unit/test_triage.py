"""Failure triage (§5.2.8): the cause chip (`util/logscan.cause_of`), error-line levels and the efficiency card
(`util/efficiency`)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from runfleet.models import (Job, JobDetail, JobGpuUsage, JobKey, JobKind, JobState, JobStep, NodeAlloc,
                             ResourceAlloc)
from runfleet.util import efficiency as eff_mod
from runfleet.util.efficiency import efficiency, lines, round_mem, round_time, rows
from runfleet.util.logscan import CAUSE_PATTERNS, cause_of, clean_line, line_level

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)
GiB = 1024 ** 3


def job(state: JobState = JobState.FAILED, *, exit_code: int | None = 1, exit_signal: int | None = 0,
        elapsed: timedelta | None = timedelta(hours=4), limit: timedelta | None = timedelta(hours=24), cpus: int = 8,
        mem_gib: int = 48, gpus: int = 1, nodes: int = 1, raw: dict[str, str] | None = None,
        kind: JobKind = JobKind.SLURM, **kw: object) -> Job:
    started = T0 - (elapsed or timedelta(0)) - timedelta(hours=1)
    return Job(key=JobKey("k2", "4001"), kind=kind, name="train", user="me", state=state, raw_state=state.name,
               job_id_raw="4001", start_time=started, end_time=T0 - timedelta(hours=1) if state.is_terminal else None,
               elapsed=elapsed, time_limit=limit, exit_code=exit_code, exit_signal=exit_signal,
               alloc=ResourceAlloc(nodes, cpus, mem_gib * GiB, gpus, "h200" if gpus else None, {},
                                   (NodeAlloc("g-01", None, None, (0,), None),)),
               nodes=tuple(f"g-0{i}" for i in range(nodes)), raw=raw or {}, **kw)  # type: ignore[arg-type]


def step(sid: str, *, ntasks: int | None = 1, rss_gib: float | None = None, ave_cpu: timedelta | None = None,
         total_cpu: timedelta | None = None, gpuutil: str | None = None, code: int | None = 0,
         sig: int | None = 0) -> JobStep:
    return JobStep(sid, sid.rpartition(".")[2], JobState.COMPLETED, "COMPLETED", code, sig, timedelta(hours=4), None,
                   None, "g-01", ntasks, int(rss_gib * GiB) if rss_gib is not None else None, ave_cpu,
                   tres_usage_tot={"gres/gpuutil": gpuutil} if gpuutil is not None else {}, total_cpu=total_cpu)


def detail_of(j: Job, steps: tuple[JobStep, ...]) -> JobDetail:
    return JobDetail(j, {}, (), steps, (), None, (), "", {}, (), T0)


# ------------------------------------------------------------------ cause chip
@pytest.mark.parametrize(("lines", "code", "evidence"), [
    (["epoch 3", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"], "OOM-GPU",
     "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"),
    (["jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate"],
     "OOM-GPU", None),
    (["slurmstepd: error: Detected 1 oom_kill event in StepId=4001.batch."], "OOM-HOST", None),
    (["OSError: [Errno 122] Disk quota exceeded: 'ckpt/last.pt'"], "QUOTA", None),
    (["OSError: [Errno 28] No space left on device"], "QUOTA", None),
    (["ModuleNotFoundError: No module named 'flash_attn'"], "IMPORT", None),
    (["ImportError: cannot import name 'x' from 'y'"], "IMPORT", None),
    (["/var/spool/slurmd/job4001/slurm_script: line 12: 1234 Segmentation fault (core dumped) python x.py"], "SEGV",
     None),
    (["step 400 loss nan lr 3e-4"], "NAN", None),
    (["ValueError: loss is inf at step 12"], "NAN", None),
    (["[rank1]:[E0906 ProcessGroupNCCL.cpp:616] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL"],
     "NCCL", None),
    (["torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1970, unhandled system error"],
     "NCCL", None),
])
def test_log_causes(lines: list[str], code: str, evidence: str | None) -> None:
    c = cause_of(job(), None, lines)
    assert c is not None and c.code == code
    assert c.evidence == (evidence or lines[-1].strip())
    assert c.pattern is CAUSE_PATTERNS[code]


def test_specific_cause_beats_nccl_and_evidence_is_the_first_line() -> None:
    """One rank's GPU OOM shows up as NCCL timeouts elsewhere: OOM-GPU wins; its first matching line is the evidence."""
    lines = ["NCCL error: watchdog timeout", "CUDA out of memory (rank 0)", "torch.OutOfMemoryError: CUDA out of memory"]
    c = cause_of(job(), None, lines)
    assert c is not None and c.code == "OOM-GPU" and c.evidence == "CUDA out of memory (rank 0)"


@pytest.mark.parametrize(("state", "code"), [(JobState.OUT_OF_MEMORY, "OOM-HOST"), (JobState.TIMEOUT, "TIMEOUT"),
                                             (JobState.DEADLINE, "TIMEOUT"), (JobState.PREEMPTED, "PREEMPTED"),
                                             (JobState.NODE_FAIL, "NODE_FAIL"), (JobState.BOOT_FAIL, "NODE_FAIL"),
                                             (JobState.CANCELLED, "CANCELLED")])
def test_state_causes_win_over_the_log(state: JobState, code: str) -> None:
    c = cause_of(job(state, exit_code=0), None, ["torch.OutOfMemoryError: CUDA out of memory"])
    assert c is not None and c.code == code


def test_oom_state_names_the_request_and_finds_the_kill_line() -> None:
    c = cause_of(job(JobState.OUT_OF_MEMORY, raw={"ReqMem": "48G"}), None,
                 ["x", "slurmstepd: error: Detected 1 oom_kill event in StepId=4001.batch."])
    assert c is not None and "--mem 48G" in c.text and c.evidence is not None and "oom_kill" in c.evidence


def test_cancelled_record_of_a_timeout_or_preemption() -> None:
    to = cause_of(job(JobState.CANCELLED), None, ["slurmstepd: error: *** JOB 4001 CANCELLED AT x DUE TO TIME LIMIT ***"])
    pr = cause_of(job(JobState.CANCELLED), None, ["slurmstepd: error: *** JOB 4001 CANCELLED AT x DUE TO PREEMPTION ***"])
    by = cause_of(job(JobState.CANCELLED, raw={"cancelled_by_uid": "10001"}), None, [])
    assert to is not None and to.code == "TIMEOUT" and to.evidence is not None
    assert pr is not None and pr.code == "PREEMPTED"
    assert by is not None and by.code == "CANCELLED" and "uid 10001" in by.text


def test_exit_status_and_signals() -> None:
    c9 = cause_of(job(exit_code=137))
    assert c9 is not None and c9.code == "SIGNAL 9" and "SIGKILL" in c9.text and "OOM killer" in c9.text
    c11 = cause_of(job(exit_code=139))
    assert c11 is not None and c11.code == "SEGV"
    sig = cause_of(job(exit_code=0, exit_signal=6))
    assert sig is not None and sig.code == "SIGNAL 6" and "SIGABRT" in sig.text
    e = cause_of(job(exit_code=2))
    assert e is not None and e.code == "EXIT 2" and e.evidence is None
    assert cause_of(job(exit_code=None, exit_signal=None)) is None


def test_step_signals_when_the_job_says_nothing() -> None:
    j = job(exit_code=0, exit_signal=0)
    d = detail_of(j, (step("4001.extern", sig=9), step("4001.batch", code=143)))
    c = cause_of(j, d)
    assert c is not None and c.code == "SIGNAL 15"
    oom_marker = detail_of(j, (step("4001.batch", code=0, sig=125),))
    assert cause_of(j, oom_marker) is None                      # 0:125 is Slurm's OOM marker, not signal 125


def test_no_chip_for_active_queued_completed() -> None:
    lines = ["torch.OutOfMemoryError: CUDA out of memory"]
    for st in (JobState.RUNNING, JobState.PENDING, JobState.COMPLETED):
        assert cause_of(job(st), None, lines) is None


def test_workstation_task_exit_from_log() -> None:
    c = cause_of(job(JobState.EXITED, exit_code=None, kind=JobKind.TASK), None, ["ModuleNotFoundError: No module x"])
    assert c is not None and c.code == "IMPORT"


def test_evidence_is_cleaned_and_capped() -> None:
    assert clean_line("\x1b[31mRuntimeError: boom\x1b[0m\r\n") == "RuntimeError: boom"
    assert clean_line("10%|█   \r100%|█████ done") == "100%|█████ done"
    c = cause_of(job(), None, ["CUDA out of memory " + "x" * 400], limit=50)
    assert c is not None and c.evidence is not None and len(c.evidence) == 50 and c.evidence.endswith("…")
    assert c.plain().startswith("OOM-GPU: GPU memory ran out") and "· log: CUDA out of memory" in c.plain()


@pytest.mark.parametrize(("line", "level"), [
    ("Traceback (most recent call last):", "error"), ("RuntimeError: x", "error"), ("error: bad option", "error"),
    ("slurmstepd: error: *** JOB 1 ON n CANCELLED AT t DUE TO TIME LIMIT ***", "error"), ("Killed", "error"),
    ("Segmentation fault (core dumped)", "error"), ("step 4 nan loss", "error"), ("Disk quota exceeded", "error"),
    ("NCCL error: unhandled system error", "error"), ("UserWarning: torch.cuda.amp is deprecated", "warn"),
    ("WARNING: no GPU found", "warn"), ("g-01:1:1 [0] NCCL INFO Bootstrap : Using ib0", None),
    ("epoch 4  train_loss 0.31  val_error 0.2", None), ("zoom room info", None),
])
def test_line_level(line: str, level: str | None) -> None:
    assert line_level(line) == level


# ------------------------------------------------------------------ efficiency card
def test_rounding() -> None:
    assert round_time(100) == "0:15:00" and round_time(50 * 60) == "1:00:00" and round_time(4.4 * 3600) == "5:00:00"
    assert round_time(30 * 3600) == "1-06:00:00" and round_time(31 * 3600) == "1-12:00:00"
    assert round_mem(0.2 * GiB) == "1G" and round_mem(5.6 * GiB) == "6G" and round_mem(22.5 * GiB) == "24G"
    assert round_mem(65 * GiB) == "72G" and round_mem(16 * GiB) == "16G"
    # never above the request that was enough, even when rounding up would pass it
    assert round_time(65 * 60 * 1.25, cap=70 * 60) == "1:10:00" and round_time(100, cap=600) == "0:10:00"
    assert round_mem(17 * 1.25 * GiB, cap=18 * GiB) == "18G" and round_mem(0.4 * GiB, cap=500 << 20) == "500M"


def test_completed_job_next_time_never_exceeds_unaligned_requests() -> None:
    j = job(JobState.COMPLETED, exit_code=0, elapsed=timedelta(minutes=65), limit=timedelta(minutes=70), mem_gib=18)
    e = efficiency(j, (step("4001.batch", rss_gib=17.0),))
    assert e is not None and e.next_time == "1:10:00" and e.next_mem == "18G"     # not 2:00:00 / 20G


def test_completed_job_card_and_next_time() -> None:
    j = job(JobState.COMPLETED, exit_code=0, elapsed=timedelta(hours=4, minutes=48), limit=timedelta(hours=24))
    steps = (step("4001", ntasks=None, total_cpu=timedelta(hours=19, minutes=12)),
             step("4001.batch", rss_gib=18.0, ave_cpu=timedelta(hours=1), gpuutil="92"),
             step("4001.extern", rss_gib=0.001, ave_cpu=timedelta(days=9999)))
    e = efficiency(j, steps)
    assert e is not None and not e.live and e.completed
    assert e.cpu_time_s == 19.2 * 3600 and e.cpu_pct == pytest.approx(50.0)
    assert e.mem_peak == 18 * GiB and e.mem_pct == pytest.approx(37.5) and e.gpu_pct == 92
    assert e.next_time == "6:00:00" and e.next_mem == "24G"
    text = lines(e)
    assert text[0] == "time  4:48:00 of 1-00:00:00 (20%)"
    assert text[1] == "cpu   50% of 8 cores (CPU time 19:12:00)"
    assert text[2] == "mem   peak 18G of 48G (38%)"
    assert text[3] == "gpu   92% mean over 1 GPU (sacct gres/gpuutil)"
    assert text[4] == "next  --time=6:00:00 --mem=24G"
    levels = {label: level for label, _t, level in rows(e)}
    assert levels == {"time": "warn", "cpu": "ok", "mem": "ok", "gpu": "ok", "next": "none"}


def test_cpu_time_falls_back_to_avecpu_and_mem_counts_tasks_on_one_node() -> None:
    j = job(JobState.COMPLETED, exit_code=0, elapsed=timedelta(hours=2), cpus=16)
    steps = (step("4001.batch", rss_gib=1.0, ave_cpu=timedelta(minutes=10)),
             step("4001.0", ntasks=4, rss_gib=5.0, ave_cpu=timedelta(hours=1, minutes=30)))
    e = efficiency(j, steps)
    assert e is not None and e.cpu_time_s == (10 + 4 * 90) * 60
    assert e.mem_peak == 20 * GiB                         # MaxRSS is per task: 4 tasks on the node
    multi = efficiency(job(JobState.COMPLETED, exit_code=0, nodes=2), steps)
    assert multi is not None and multi.mem_peak == 5 * GiB and multi.next_mem is None and multi.note is not None


def test_suggestions_never_exceed_what_worked() -> None:
    j = job(JobState.COMPLETED, exit_code=0, elapsed=timedelta(hours=23), limit=timedelta(hours=24), mem_gib=20)
    e = efficiency(j, (step("4001.batch", rss_gib=19.0),))
    assert e is not None and e.next_time == "1-00:00:00" and e.next_mem == "20G"


def test_timeout_oom_and_failed_cards() -> None:
    to = efficiency(job(JobState.TIMEOUT, exit_code=0, elapsed=timedelta(hours=12), limit=timedelta(hours=12)), ())
    assert to is not None and to.next_time == "15:00:00" and to.next_mem is None and "guess" in (to.note or "")
    oom = efficiency(job(JobState.OUT_OF_MEMORY, exit_code=0, mem_gib=16), ())
    assert oom is not None and oom.next_mem == "24G" and oom.next_time is None
    failed = efficiency(job(), (step("4001.batch", rss_gib=2.0),))
    assert failed is not None and failed.next_time is None and failed.next_mem is None
    assert "partial run" in (failed.note or "") and lines(failed)[-2] == "next  –"
    assert {label: lv for label, _t, lv in rows(failed)}["time"] == "none"   # a failed run's time says nothing


def test_idle_gpus_and_unknowns() -> None:
    e = efficiency(job(JobState.COMPLETED, exit_code=0, gpus=2), (step("4001.batch", gpuutil="6"),))
    assert e is not None and e.gpu_pct == 3 and e.note is not None and e.note.startswith("GPU idle (mean 3%)")
    assert dict((r[0], r[2]) for r in rows(e))["gpu"] == "bad"
    bare = efficiency(job(JobState.COMPLETED, exit_code=0, gpus=1), ())
    assert bare is not None and bare.cpu_pct is None and bare.mem_peak is None and bare.gpu_pct is None
    assert "–" in lines(bare)[1] and "gres/gpuutil not recorded" in lines(bare)[3]


def test_live_headroom_and_no_card() -> None:
    usage = JobGpuUsage("sstat", 1, 80.0, 80.0, 90.0, None, None)
    running = job(JobState.RUNNING, exit_code=None, cpus=4, mem_gib=10, cpu_pct=300.0, rss_bytes=int(9.5 * GiB),
                  gpu=usage)
    e = efficiency(running, ())
    assert e is not None and e.live and e.cpu_pct == 75 and e.gpu_pct == 80 and e.next_time is None
    mem = next(r for r in rows(e) if r[0] == "mem")
    assert mem[1] == "rss 9.5G of 10G (95%) · 512M headroom" and mem[2] == "bad"
    assert all(label != "next" for label, _t, _l in rows(e))
    filled = efficiency(dataclasses.replace(running, rss_bytes=None), (step("4001.0", ntasks=2, rss_gib=2.0),))
    assert filled is not None and filled.mem_peak == 4 * GiB          # sstat-filled steps when the job has no RSS
    assert efficiency(job(JobState.PENDING, exit_code=None), ()) is None
    assert efficiency(job(kind=JobKind.TASK), ()) is None
    assert lines(None) == ()


def test_thresholds_are_reportseff_like() -> None:
    assert (eff_mod.CPU_LOW, eff_mod.MEM_LOW, eff_mod.TIME_LOW, eff_mod.MARGIN) == (50.0, 25.0, 25.0, 1.25)
