"""[WP2] State mapping, reason humanizer, exit explanations and short summaries (§5.2.3, §5.2.6)."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from runfleet.backends.slurm.parse_sacct import parse_sacct, parse_steps
from runfleet.backends.slurm.states import (explain, humanize_reason, is_held, map_state, preempt_text,
                                            requeue_window_open, short_summary, will_never_run)
from runfleet.models import Job, JobKey, JobKind, JobState
from tests.backends.factories import NOW, SLURM_CONFIG, TZ, format_fields
from tests.conftest import meta, sample


@pytest.mark.parametrize(("raw", "state"), [
    ("PENDING", JobState.PENDING), ("CONFIGURING", JobState.CONFIGURING), ("REQUEUED", JobState.REQUEUED),
    ("REQUEUE_FED", JobState.REQUEUED), ("REQUEUE_HOLD", JobState.REQUEUED), ("RESV_DEL_HOLD", JobState.PENDING),
    ("RUNNING", JobState.RUNNING), ("RESIZING", JobState.RUNNING), ("SIGNALING", JobState.RUNNING),
    ("STAGE_OUT", JobState.RUNNING), ("COMPLETING", JobState.COMPLETING), ("SUSPENDED", JobState.SUSPENDED),
    ("STOPPED", JobState.STOPPED), ("COMPLETED", JobState.COMPLETED), ("FAILED", JobState.FAILED),
    ("SPECIAL_EXIT", JobState.REQUEUED), ("EXPEDITING", JobState.REQUEUED), ("RF", JobState.REQUEUED),
    ("TIMEOUT", JobState.TIMEOUT), ("OUT_OF_MEMORY", JobState.OUT_OF_MEMORY),
    ("NODE_FAIL", JobState.NODE_FAIL), ("PREEMPTED", JobState.PREEMPTED), ("BOOT_FAIL", JobState.BOOT_FAIL),
    ("DEADLINE", JobState.DEADLINE), ("CANCELLED", JobState.CANCELLED), ("REVOKED", JobState.CANCELLED),
    ("NEW_FANCY_STATE", JobState.UNKNOWN),
])
def test_map_state(raw: str, state: JobState) -> None:
    assert map_state(raw)[0] is state


def test_cancelled_by_and_held() -> None:
    assert map_state("CANCELLED by 610001") == (JobState.CANCELLED, {"cancelled_by_uid": "610001"})
    assert is_held("JobHeldUser", "PENDING") and is_held("JobHeldAdmin", "PENDING")
    assert is_held(None, "REQUEUE_HOLD") and not is_held("Priority", "PENDING")
    # SPECIAL_EXIT = requeued by slurmctld, PENDING with priority 0: still in squeue, held, not ended
    assert is_held(None, "SPECIAL_EXIT") and not map_state("SPECIAL_EXIT")[0].is_terminal
    assert not map_state("EXPEDITING")[0].is_terminal


def test_reasons() -> None:
    assert humanize_reason("Priority") == "waiting for priority"
    assert humanize_reason("QOSMaxJobsPerUserLimit") == "per-user job cap (QOS)"
    assert humanize_reason("DependencyNeverSatisfied") == "will never run: dependency failed"
    assert humanize_reason("(Resources)") == "waiting for resources"
    assert humanize_reason("SomethingNew") == "SomethingNew"
    assert humanize_reason("None") is None and humanize_reason(None) is None
    assert will_never_run("DependencyNeverSatisfied") and will_never_run("PartitionTimeLimit")
    assert not will_never_run("Priority")


def _history(host: str) -> dict[str, object]:
    return {j.key.id: j for j in parse_sacct(sample(host, "sacct_ext"), format_fields(meta(host, "sacct_ext")["cmd"]),
                                             host=host, tz=TZ[host], sep="|").items}


def test_lyra_preempted_explanation() -> None:
    steps = parse_steps(sample("lyra", "sacct_failed_job_steps"),
                        format_fields(meta("lyra", "sacct_failed_job_steps")["cmd"]), tz=TZ["lyra"], sep="|").items
    alloc_row = next(s for s in steps if s.step_id == "31354789")
    job = Job(key=JobKey("lyra", "31354789"), kind=JobKind.SLURM, name="x", user="alice07",
              state=alloc_row.state, raw_state=alloc_row.raw_state, job_id_raw="31354789")
    lines = explain(job, steps, uid=2100007, slurm_config=SLURM_CONFIG["lyra"])
    assert any("SIGTERM (128+15)" in ln for ln in lines)
    assert any("PreemptMode CANCEL: killed, resubmit" in ln for ln in lines)
    assert short_summary(job, uid=None, slurm_config=SLURM_CONFIG["lyra"]) == "preempted, killed"
    assert "requeue" in preempt_text(SLURM_CONFIG["k2"])


def test_k2_failed_batch_explanation() -> None:
    steps = parse_steps(sample("k2", "sacct_failed_job_steps"),
                        format_fields(meta("k2", "sacct_failed_job_steps")["cmd"]), tz=TZ["k2"], sep="|").items
    job = Job(key=JobKey("k2", "2330539"), kind=JobKind.SLURM, name="x", user="u", state=JobState.FAILED,
              raw_state="FAILED", job_id_raw="2330539", exit_code=1, exit_signal=0)
    lines = explain(job, steps, uid=1, slurm_config=SLURM_CONFIG["k2"])
    assert lines == ("batch step exited with code 1 (the job script / program failed; not a Slurm kill)",)
    assert short_summary(job, uid=1, slurm_config={}) == "exit 1"


def test_summaries_from_fixtures() -> None:
    h = _history("lyra")
    assert short_summary(h["31301780"], uid=2100007, slurm_config={}) == "cancelled by you"
    assert short_summary(h["31301780"], uid=1, slurm_config={}) == "cancelled by uid 2100007"
    assert short_summary(h["31327454"], uid=1, slurm_config={}) == "exit 2"
    assert short_summary(h["31303059"], uid=1, slurm_config={}) is None
    oom = dataclasses.replace(h["31327454"], state=JobState.OUT_OF_MEMORY)
    assert explain(oom, (), uid=None, slurm_config={})[0].startswith("out of memory: exceeded --mem 48G")
    # review: 0:125 is Slurm's OOM marker (not "killed by signal 125"); a sampled MaxRSS below --mem is explained
    from runfleet.models import JobStep
    step = JobStep(step_id="1.batch", name="batch", state=JobState.OUT_OF_MEMORY, raw_state="OUT_OF_MEMORY",
                   exit_code=0, exit_signal=125, elapsed=None, start=None, end=None, nodelist=None, ntasks=1,
                   max_rss_bytes=int(4.5 * 1024 ** 3), ave_cpu=None, tres_usage_max={}, tres_usage_tot={})
    lines = explain(oom, (step,), uid=None, slurm_config={})
    assert not any("signal 125" in ln for ln in lines)
    assert "sampled" in lines[0] and "missed between samples" in lines[0]
    to = dataclasses.replace(h["31327454"], state=JobState.TIMEOUT)
    assert explain(to, (), uid=None, slurm_config={}) == ("hit time limit 3-00:00:00",)
    sig = dataclasses.replace(h["31327454"], exit_code=137)
    assert short_summary(sig, uid=None, slurm_config={}) == "exit 137 = SIGKILL"


def test_requeue_window() -> None:
    h = _history("lyra")
    job = dataclasses.replace(h["31327454"], end_time=NOW - timedelta(seconds=100))
    assert requeue_window_open(job, NOW, {"MinJobAge": "300 sec"})
    assert not requeue_window_open(job, NOW + timedelta(seconds=250), {"MinJobAge": "300 sec"})
