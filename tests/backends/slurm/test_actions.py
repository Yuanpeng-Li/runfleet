"""[WP2] Slurm action specs, eligibility, plans (exact remote commands) and result interpretation (§6.1, §6.3)."""

from __future__ import annotations

import dataclasses
import shlex
from datetime import timedelta

import pytest

from runfleet.backends.base import ActionPlan
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.errors import ActionError
from runfleet.models import Job, JobState
from tests.backends.factories import NOW, ctx, host_config, result, to_sep
from tests.backends.slurm.test_collector import squeue, tick
from tests.conftest import sample


@pytest.fixture
def k2() -> tuple[SlurmCollector, dict[str, Job]]:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, squeue())
    assert out.jobs is not None
    return c, dict(out.jobs)


def specs(c: SlurmCollector, jobs: list[Job], **kw: object) -> dict[str, object]:
    return {s.id: s for s in c.actions(jobs, ctx("k2", **kw))}


def test_specs_catalogue(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    s = specs(c, [jobs["2332861"]])
    assert list(s) == ["slurm.cancel", "slurm.hold", "slurm.release", "slurm.requeue"]
    assert all(x.mutating and x.confirm for x in s.values())
    assert s["slurm.cancel"].disabled_reason is None
    assert s["slurm.hold"].disabled_reason == "only pending jobs can be held"
    assert s["slurm.release"].disabled_reason == "not held"
    assert s["slurm.requeue"].disabled_reason is None
    assert [p.name for p in s["slurm.requeue"].params] == ["hold"]
    pd = specs(c, [jobs["2334564"]])
    assert pd["slurm.hold"].disabled_reason is None
    ro = specs(c, [jobs["2332861"]], readonly=True)
    assert all(x.disabled_reason == "read-only mode" for x in ro.values())


def test_protection_and_ownership(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    cfg = host_config("k2", slurm={"protected_partitions": ("labpi",), "protected_qos": ("labpi",)})
    cx = ctx("k2", cfg=cfg)
    protected = dataclasses.replace(jobs["2332861"], partition="labpi")
    s = {x.id: x for x in c.actions([protected], cx)}
    assert s["slurm.cancel"].disabled_reason == "protected partition labpi"
    qos = dataclasses.replace(jobs["2332861"], qos="labpi")
    assert {x.id: x for x in c.actions([qos], cx)}["slurm.cancel"].disabled_reason == "protected QOS labpi"
    other = dataclasses.replace(jobs["2332861"], user="alice", is_mine=False)
    assert {x.id: x for x in c.actions([other], cx)}["slurm.cancel"].disabled_reason == "owned by alice"
    # mixed selection: enabled, the ineligible ones are skipped with reasons
    plan = c.plan_action("slurm.cancel", [jobs["2332862"], protected, other], {}, cx)
    assert plan.targets == (jobs["2332862"].key,)
    assert dict(plan.skipped) == {protected.key: "protected partition labpi", other.key: "owned by alice"}
    with pytest.raises(ActionError):
        c.plan_action("slurm.cancel", [protected], {}, cx)


def test_cancel_plan_three_jobs(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    plan = c.plan_action("slurm.cancel", [jobs["2332861"], jobs["2332862"], jobs["2332863"]], {}, ctx("k2"))
    assert plan.preview == "scancel 2332861 2332862 2332863"
    assert plan.confirm_title == "Cancel 3 jobs on k2?"
    (sec,) = plan.sections
    assert sec.name == "slurm.act.cancel" and sec.owner == "slurm" and sec.meta == {"ids": ["2332861", "2332862",
                                                                                            "2332863"]}
    assert "2332861  orb-swp--train--cifar-warmup--s43--a02  R  4:07:00" in plan.confirm_body
    single = c.plan_action("slurm.cancel", [jobs["2334564"]], {}, ctx("k2"))
    assert single.confirm_title == "Cancel 1 job on k2?"


def test_array_summary_strips_throttle() -> None:
    cx = ctx("lyra")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, result("slurm.squeue", to_sep(sample("lyra", "squeue_parts_O").strip(), "\x1f")))
    assert out.jobs is not None
    job = dataclasses.replace(out.jobs["31270818_[1-24%1]"], is_mine=True)
    plan = c.plan_action("slurm.cancel", [job], {}, cx)
    assert shlex.split(plan.preview) == ["scancel", "31270818_[1-24]"]
    assert plan.sections[0].meta["ids"] == ["31270818_[1-24]"]
    hold = c.plan_action("slurm.hold", [job], {}, cx)
    assert shlex.split(hold.preview) == ["scontrol", "hold", "31270818_[1-24]"]


def test_hold_release_requeue(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    cx = ctx("k2")
    held = dataclasses.replace(jobs["2334564"], held=True, reason="JobHeldUser")
    assert c.plan_action("slurm.release", [held], {}, cx).preview == "scontrol release 2334564"
    with pytest.raises(ActionError, match="already held"):
        c.plan_action("slurm.hold", [held], {}, cx)
    rq = c.plan_action("slurm.requeue", [jobs["2332861"]], {"hold": True}, cx)
    assert rq.preview == "scontrol requeuehold 2332861" and rq.params == {"hold": True}
    assert rq.sections[0].meta == {"ids": ["2332861"], "hold": True}
    assert c.plan_action("slurm.requeue", [jobs["2332861"]], {}, cx).preview == "scontrol requeue 2332861"
    ended = dataclasses.replace(jobs["2332861"], state=JobState.FAILED, end_time=NOW - timedelta(seconds=60))
    assert c.plan_action("slurm.requeue", [ended], {}, cx).targets == (ended.key,)
    old = dataclasses.replace(ended, end_time=NOW - timedelta(hours=1))
    with pytest.raises(ActionError, match="no longer in controller memory"):
        c.plan_action("slurm.requeue", [old], {}, cx)
    with pytest.raises(ActionError, match="already ended"):
        c.plan_action("slurm.cancel", [old], {}, cx)


def _interpret(c: SlurmCollector, plan: ActionPlan, stdout: str = "", rc: int = 0, stderr: str = "") -> list[object]:
    return c.interpret_action(plan, {plan.sections[0].name: result(plan.sections[0].name, stdout, rc, stderr)},
                              ctx("k2"))


def test_interpret_success_and_per_id_errors(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    plan = c.plan_action("slurm.cancel", [jobs["2332861"], jobs["2332862"]], {}, ctx("k2"))
    ok = _interpret(c, plan)
    assert [(r.ok, r.message) for r in ok] == [(True, "cancelled 2332861"), (True, "cancelled 2332862")]
    mixed = _interpret(c, plan, rc=1,
                       stderr="scancel: error: Kill job error on job id 2332862: Invalid job id specified")
    assert [(r.ok, r.message) for r in mixed] == [(True, "cancelled 2332861"),
                                                  (False, "job already ended")]
    assert mixed[1].details == ("scancel: error: Kill job error on job id 2332862: Invalid job id specified",)
    denied = _interpret(c, plan, rc=1, stderr="scancel: error: Kill job error on job id 2332861: "
                                              "Access/permission denied")
    assert denied[0].message == "permission denied" and not denied[0].ok
    unknown = _interpret(c, plan, rc=1, stderr="scancel: error: Unable to contact slurm controller")
    assert not any(r.ok for r in unknown)


def test_interpret_timeout_and_missing(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    from runfleet.probe.section import SectionOutcome
    c, jobs = k2
    plan = c.plan_action("slurm.hold", [jobs["2334564"]], {}, ctx("k2"))
    res = result(plan.sections[0].name, "", 124)
    assert res.outcome is SectionOutcome.TIMEOUT
    (r,) = c.interpret_action(plan, {res.name: res}, ctx("k2"))
    assert not r.ok and r.message == "timed out"
    (r,) = c.interpret_action(plan, {}, ctx("k2"))
    assert not r.ok and r.message == "not run"
    (r,) = c.interpret_action(plan, {res.name: result(res.name, "", 0)}, ctx("k2"))
    assert r.ok and r.message == "held 2334564"


# ---------------------------------------------------------------- review 2026-10: odd array ids, scontrol summaries
def _pending_array(jobs: dict[str, Job], task: str) -> Job:
    base = jobs["2334564"]
    return dataclasses.replace(base, key=dataclasses.replace(base.key, id=f"500_[{task}]"), job_id_raw="500",
                               array_job_id="500", array_task=task, state=JobState.PENDING)


@pytest.mark.parametrize("task", ["1-99:2%5", "1,3,5,...,45,...%4"])
def test_abbreviated_array_summaries_plan_without_crashing(k2: tuple[SlurmCollector, dict[str, Job]],
                                                           task: str) -> None:
    """Slurm prints step ranges and truncated lists for big sparse arrays; plan() raised ValueError (TUI crash)."""
    c, jobs = k2
    job = _pending_array(jobs, task)
    plan = c.plan_action("slurm.cancel", [job], {}, ctx("k2"))
    mixed = c.plan_action("slurm.cancel", [job, jobs["2334564"]], {}, ctx("k2"))
    if "..." in task:                # truncated: only the whole array can be named, and the dialog says so
        assert plan.preview == "scancel --state=PENDING 500"
        assert mixed.preview == "scancel 2334564; a=$?; scancel --state=PENDING 500 && [ $a -eq 0 ]"
        assert "cancels ALL pending tasks of array 500" in plan.confirm_body
    else:
        assert "ALL pending" not in plan.confirm_body
    spec = specs(c, [job])["slurm.hold"]
    assert spec.disabled_reason is not None and "abbreviated" in spec.disabled_reason  # type: ignore[attr-defined]
    with pytest.raises(ActionError):
        c.plan_action("slurm.hold", [job], {}, ctx("k2"))


def test_step_range_summary_cancels_exactly_its_tasks(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    """Review BACK-01: `scancel --state=PENDING 500` for `500_[1-99:2]` also cancelled a separately held 500_2."""
    c, jobs = k2
    job = _pending_array(jobs, "1-99:2%5")
    odd = ",".join(str(i) for i in range(1, 100, 2))
    plan = c.plan_action("slurm.cancel", [job], {}, ctx("k2"))
    assert shlex.split(plan.preview) == ["scancel", f"500_[{odd}]"] and "--state" not in plan.preview
    mixed = c.plan_action("slurm.cancel", [job, jobs["2334564"]], {}, ctx("k2"))
    assert shlex.split(mixed.preview) == ["scancel", f"500_[{odd}]", "2334564"]


def test_stopped_job_can_be_cancelled(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    job = dataclasses.replace(jobs["2334564"], state=JobState.STOPPED, raw_state="STOPPED")
    assert specs(c, [job])["slurm.cancel"].disabled_reason is None  # type: ignore[attr-defined]
    assert c.plan_action("slurm.cancel", [job], {}, ctx("k2")).preview == "scancel 2334564"


def test_per_task_lines_only_match_the_summary_holding_that_task(
        k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    a, b = _pending_array(jobs, "1-3"), _pending_array(jobs, "4-6")
    plan = c.plan_action("slurm.cancel", [a, b], {}, ctx("k2"))
    res = result(plan.sections[0].name, "", 1, "scancel: error: Kill job error on job id 500_5: Job has already finished")
    got = {r.target.id: r.ok for r in c.interpret_action(plan, {plan.sections[0].name: res}, ctx("k2"))}
    assert got == {"500_[1-3]": True, "500_[4-6]": False}


def test_builder_value_errors_become_action_errors(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    bad = dataclasses.replace(jobs["2334564"], key=dataclasses.replace(jobs["2334564"].key, id="2334564; rm -rf /"))
    with pytest.raises(ActionError):
        c.plan_action("slurm.cancel", [bad], {}, ctx("k2"))


def test_scontrol_idless_summary_line_does_not_fail_other_targets(
        k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    a = jobs["2334564"]
    b = dataclasses.replace(a, key=dataclasses.replace(a.key, id="2334565"), job_id_raw="2334565")
    plan = c.plan_action("slurm.hold", [a, b], {}, ctx("k2"))
    res = result(plan.sections[0].name, "", 1,
                 "Invalid job id specified for job 2334565\nslurm_suspend error:Invalid job id specified")
    got = {r.target.id: (r.ok, r.message) for r in c.interpret_action(plan, {plan.sections[0].name: res}, ctx("k2"))}
    assert got == {"2334564": (True, "held 2334564"), "2334565": (False, "job already ended")}


def test_per_task_lines_match_summary_targets(k2: tuple[SlurmCollector, dict[str, Job]]) -> None:
    c, jobs = k2
    job = _pending_array(jobs, "2-20%1")
    plan = c.plan_action("slurm.cancel", [job, jobs["2334564"]], {}, ctx("k2"))
    res = result(plan.sections[0].name, "", 1, "scancel: error: Kill job error on job id 500_7: Job has already finished")
    got = {r.target.id: r for r in c.interpret_action(plan, {plan.sections[0].name: res}, ctx("k2"))}
    assert not got["500_[2-20%1]"].ok and "500_7" in got["500_[2-20%1]"].details[0]
    assert got["2334564"].ok                                       # the error belongs to the summary target
