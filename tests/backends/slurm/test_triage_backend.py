"""Failure triage on the Slurm side (§5.2.8): the full pending-reason set with advice, the start-estimate caveat,
TotalCPU on the on-demand steps query only (caps-gated), and the plain efficiency card on JobDetail."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.backends.slurm.commands import SEP
from runfleet.backends.slurm.parse_sacct import parse_steps
from runfleet.backends.slurm.states import (NEVER_RUNS, REASON_ADVICE, REASONS, explain, humanize_reason,
                                            reason_advice, reason_key, will_never_run)
from runfleet.models import Job, JobKey, JobKind, JobState, ResourceAlloc
from runfleet.ui import render
from tests.backends.factories import ctx, result

FULL_SET = ("Priority", "Resources", "Dependency", "DependencyNeverSatisfied", "AssocGrpGRES", "AssocGrpCPULimit",
            "AssocGrpMemLimit", "AssocGrpNodeLimit", "AssocMaxJobsLimit", "AssocGrpBillingMinutes",
            "QOSMaxGRESPerUser", "QOSMaxJobsPerUserLimit", "QOSGrpGRES", "QOSMaxWallDurationPerJobLimit",
            "PartitionTimeLimit", "PartitionNodeLimit", "ReqNodeNotAvail", "Reservation", "BeginTime", "JobHeldUser",
            "JobHeldAdmin", "launch failed requeued held", "BadConstraints", "InvalidAccount", "InvalidQOS",
            "MaxGRESPerAccount", "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority "
            "partitions")


@pytest.mark.parametrize("reason", FULL_SET)
def test_every_common_reason_has_text_advice_and_ui_wording(reason: str) -> None:
    key = reason_key(reason)
    assert key in REASONS and key in REASON_ADVICE
    assert humanize_reason(f"({reason})") == REASONS[key]
    advice = reason_advice(reason)
    assert advice and len(advice) <= 160
    assert render.known_reason(reason)                          # the UI's short / long wording knows it too
    assert render.reason_text(reason, short=False).plain == REASONS[key]


def test_never_runs_set_is_red_everywhere() -> None:
    for key in NEVER_RUNS:
        assert will_never_run(f"({key})") and "will never run" in REASONS[key]
        assert render.hint_level(REASONS[key]) == "red"
    assert not will_never_run("Priority") and not will_never_run(None)


def test_reason_parsing_and_generic_limits() -> None:
    assert reason_key("(ReqNodeNotAvail, UnavailableNodes:n-31-01)") == "ReqNodeNotAvail"
    assert humanize_reason("SomethingNew") == "SomethingNew" and reason_advice("SomethingNew") is None
    assert reason_advice("AssocGrpGRESRunMinutes") is not None and "account" in (reason_advice("AssocGrpX") or "")
    assert reason_advice("QOSGrpBillingMinutes") is not None
    assert humanize_reason("None") is None and reason_advice(None) is None


def _pending(reason: str, *, est: datetime | None = None) -> Job:
    return Job(key=JobKey("vega", "824479"), kind=JobKind.SLURM, name="x", user="me", state=JobState.PENDING,
               raw_state="PENDING", job_id_raw="824479", reason=reason, start_time=est, est_start=est is not None,
               alloc=ResourceAlloc(1, 8, None, 1, None, {}))


def test_explain_pending_gives_advice_and_an_honest_start_estimate() -> None:
    est = datetime(2026, 9, 7, 14, 12, 37, tzinfo=UTC)
    lines = explain(_pending("Resources", est=est), (), uid=1, slurm_config={})
    assert lines[0] == REASON_ADVICE["Resources"]
    assert "est. start" in lines[1] and "guess" in lines[1] and "moves" in lines[1]
    assert explain(_pending("Priority"), (), uid=1, slurm_config={}) == (REASON_ADVICE["Priority"],)


def test_collector_puts_advice_on_pending_jobs_and_in_the_detail() -> None:
    cx = ctx("vega")
    c = SlurmCollector(cx.host)
    job = _pending("QOSMaxGRESPerUser")
    enriched = c._enrich(job, cx)
    assert enriched.raw["reason_advice"] == REASON_ADVICE["QOSMaxGRESPerUser"]
    detail = c.ingest_detail(job, {}, cx)
    assert detail.explanation[:2] == (REASONS["QOSMaxGRESPerUser"], REASON_ADVICE["QOSMaxGRESPerUser"])


# ------------------------------------------------------------------ TotalCPU, efficiency on JobDetail
def test_step_fields_add_totalcpu_only_when_sacct_knows_it() -> None:
    assert cmds.step_fields(frozenset()) == cmds.STEP_FIELDS
    assert cmds.step_fields(frozenset({"TotalCPU", "JobID"})) == cmds.STEP_FIELDS + ",TotalCPU"
    assert "TotalCPU" not in cmds.SACCT_FIELDS                    # never on the periodic history query
    assert cmds.sacct_steps_cmd("4001", SEP, cmds.step_fields(frozenset({"TotalCPU"}))).endswith(",TotalCPU")


def test_parse_steps_reads_totalcpu() -> None:
    fields = ["JobID", "JobName", "State", "ExitCode", "TotalCPU"]
    pr = parse_steps("4001|x|COMPLETED|0:0|1-02:03:04\n4001.batch|batch|COMPLETED|0:0|05:08.123\n", fields,
                     tz=timedelta(0), sep="|")
    assert pr.items[0].total_cpu == timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert pr.items[1].total_cpu == timedelta(minutes=5, seconds=8, microseconds=123000)


def _ended() -> Job:
    return Job(key=JobKey("k2", "4001"), kind=JobKind.SLURM, name="x", user="me", state=JobState.COMPLETED,
               raw_state="COMPLETED", job_id_raw="4001", start_time=datetime(2026, 9, 7, 1, 42, 37, tzinfo=UTC),
               end_time=datetime(2026, 9, 7, 5, 42, 37, tzinfo=UTC), elapsed=timedelta(hours=4),
               time_limit=timedelta(hours=24), exit_code=0, exit_signal=0,
               alloc=ResourceAlloc(1, 8, 48 * 1024 ** 3, 0, None, {}), source="sacct")


def test_detail_requests_totalcpu_and_carries_the_plain_card() -> None:
    cx = ctx("k2", sacct_fields=frozenset({"TotalCPU", "JobID"}))
    c = SlurmCollector(cx.host)
    job = _ended()
    steps = next(s for s in c.detail_sections(job, cx) if s.name == "slurm.steps")
    assert "TotalCPU" in steps.cmd and steps.meta["fields"] == cmds.step_fields(cx.caps.sacct_fields).split(",")
    names = [n.split("%")[0] for n in cmds.step_fields(cx.caps.sacct_fields).split(",")]
    row = {"JobID": "4001", "JobName": "x", "State": "COMPLETED", "ExitCode": "0:0", "TotalCPU": "16:00:00"}
    batch = dict(row, JobID="4001.batch", JobName="batch", MaxRSS="12G", NTasks="1")
    text = "\n".join(SEP.join(r.get(n, "") for n in names) for r in (row, batch))
    detail = c.ingest_detail(job, {"slurm.steps": result("slurm.steps", text)}, cx)
    assert detail.steps[0].total_cpu == timedelta(hours=16)
    assert detail.efficiency[:3] == ("time  4:00:00 of 1-00:00:00 (17%)", "cpu   50% of 8 cores (CPU time 16:00:00)",
                                     "mem   peak 12G of 48G (25%)")
    assert detail.efficiency[-1] == "next  --time=5:00:00 --mem=15G"


def test_detail_without_totalcpu_capability_keeps_the_old_query() -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    steps = next(s for s in c.detail_sections(_ended(), cx) if s.name == "slurm.steps")
    assert "TotalCPU" not in steps.cmd
    queued = dataclasses.replace(_ended(), state=JobState.PENDING, start_time=None, elapsed=None)
    assert c.ingest_detail(queued, {}, cx).efficiency == ()


def test_demo_renders_totalcpu_and_the_nccl_failure() -> None:
    from runfleet.backends.slurm.parse_sacct import parse_steps as parse
    from runfleet.demo import render as demo_render
    from runfleet.demo.world import DemoWorld
    from runfleet.util.logscan import cause_of

    world = DemoWorld(seed=7, now=lambda: datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC), latency_scale=0.0)
    vega = world.cluster("vega")
    fields = [*demo_render.STEP_FIELDS, "TotalCPU"]
    out = demo_render.sacct_steps(vega, world.now(), job_id_raw="824373", sep="|", fields=fields).stdout
    steps = parse(out, fields, tz=timedelta(0), sep="|").items
    assert steps[0].step_id == "824373" and steps[0].total_cpu is not None
    log = vega.logs[vega.jobs[824373].log_paths[0]].text().splitlines()
    job = Job(key=JobKey("vega", "824373"), kind=JobKind.SLURM, name="x", user="me", state=JobState.FAILED,
              raw_state="FAILED", job_id_raw="824373", exit_code=1)
    c = cause_of(job, None, log)
    assert c is not None and c.code == "NCCL" and "Watchdog caught collective operation timeout" in (c.evidence or "")
