"""[WP2] Task action specs, plans (§6.2 script rendering, previews, re-planning) and interpretation."""

from __future__ import annotations

import dataclasses

import pytest

from runfleet.backends.process.actions import SIGNAL_SCRIPT, interpret_lines, render_script
from runfleet.backends.process.collector import ProcessCollector
from runfleet.errors import ActionError
from runfleet.models import Job, JobState
from tests.backends.factories import ctx, result
from tests.backends.process.test_collector import ROOT, started


@pytest.fixture
def task() -> tuple[ProcessCollector, Job]:
    c, _cx, _mv, out = started()
    return c, out.jobs[ROOT]


def test_specs(task: tuple[ProcessCollector, Job]) -> None:
    c, job = task
    s = {x.id: x for x in c.actions([job], ctx("lynx"))}
    assert list(s) == ["task.terminate", "task.interrupt", "task.kill"]
    assert all(x.disabled_reason is None and x.mutating and x.confirm for x in s.values())
    assert [(p.name, p.default) for p in s["task.terminate"].params] == [("escalate", True), ("grace_s", 10)]
    assert s["task.kill"].params == () and s["task.interrupt"].params == ()
    ro = {x.id: x for x in c.actions([job], ctx("lynx", readonly=True))}
    assert all(x.disabled_reason == "read-only mode" for x in ro.values())
    other = dataclasses.replace(job, is_mine=False, user="alice")
    assert c.actions([other], ctx("lynx"))[0].disabled_reason == "owned by alice"
    ended = dataclasses.replace(job, state=JobState.EXITED)
    assert c.actions([ended], ctx("lynx"))[0].disabled_reason == "task already ended"


def test_terminate_plan_and_replan(task: tuple[ProcessCollector, Job]) -> None:
    c, job = task
    cx = ctx("lynx")
    plan = c.plan_action("task.terminate", [job], {}, cx)
    (sec,) = plan.sections
    assert sec.name == "process.kill.642229" and sec.owner == "process"
    assert sec.meta == {"root": 642229, "start": int(job.raw["root_start"]), "sig": "TERM", "grace_s": 10,
                        "escalate": True}
    assert sec.timeout == 25
    assert sec.cmd.splitlines()[0] == f"R=642229 RS={job.raw['root_start']} SIG=TERM G=10 ESC=1"
    assert plan.preview == "STOP 642229 · TERM its tree · KILL after 10 s"
    assert plan.confirm_title == "Terminate task p642229 on lynx?"
    assert "orb.launch ×2 (orb.train) · 3 processes · GPUs 0,1" in plan.confirm_body
    assert "launcher paused first" in plan.confirm_body
    # ConfirmScreen re-plans on every param change
    no_esc = c.plan_action("task.terminate", [job], {"escalate": False, "grace_s": 30}, cx)
    assert no_esc.preview == "STOP 642229 · TERM its tree" and no_esc.sections[0].meta["grace_s"] == 30
    assert "ESC=0" in no_esc.sections[0].cmd and no_esc.params == {"escalate": False, "grace_s": 30}
    assert c.plan_action("task.terminate", [job], {"grace_s": 9999}, cx).sections[0].meta["grace_s"] == 600
    assert c.plan_action("task.terminate", [job], {"grace_s": "x"}, cx).sections[0].meta["grace_s"] == 10


def test_interrupt_and_kill(task: tuple[ProcessCollector, Job]) -> None:
    c, job = task
    cx = ctx("lynx")
    i = c.plan_action("task.interrupt", [job], {}, cx)
    assert i.sections[0].meta["sig"] == "INT" and i.sections[0].meta["escalate"] is False
    assert i.preview == "STOP 642229 · INT its tree"
    k = c.plan_action("task.kill", [job], {}, cx)
    assert k.sections[0].meta == {"root": 642229, "start": int(job.raw["root_start"]), "sig": "KILL", "grace_s": 0,
                                  "escalate": True}
    assert k.preview == "STOP 642229 · KILL its tree"
    with pytest.raises(ActionError):
        c.plan_action("task.kill", [dataclasses.replace(job, is_mine=False, user="x")], {}, cx)
    with pytest.raises(ActionError):
        c.plan_action("task.nope", [job], {}, cx)


def test_render_validates() -> None:
    assert "kill -s STOP \"$R\"" in render_script(10, 0, "TERM", 5, True)
    for bad in [(1, 0, "TERM", 5, True), (10, 0, "HUP", 5, True), (10, 0, "TERM", -1, True), (10, -5, "INT", 0, False)]:
        with pytest.raises(ValueError):
            render_script(*bad)
    assert SIGNAL_SCRIPT.count("{root}") == 1


@pytest.mark.parametrize(("lines", "ok", "message"), [
    (["gone 5", "done"], True, "already exited"),
    (["reused 5", "done"], False, "pid reused — refresh"),
    (["members 5 6 7", "sent 6", "sent 7", "sent 5", "done"], True, "signalled 3 processes"),
    (["members 5 6", "sent 6", "sent 5", "killed 6", "done"], True, "signalled 2 processes, killed 1"),
    (["members 5 6 7", "sent 6", "alive 6", "alive 7", "done"], False, "2 processes survived"),
    (["members 5", "sent 5", "new 790001", "done"], False, "1 process started during termination: 790001"),
    (["members 5", "sent 5"], False, "signal script did not finish"),
])
def test_interpret_lines(lines: list[str], ok: bool, message: str) -> None:
    assert interpret_lines(lines) == (ok, message)


def test_interpret_results(task: tuple[ProcessCollector, Job]) -> None:
    c, job = task
    cx = ctx("lynx")
    plan = c.plan_action("task.terminate", [job], {}, cx)
    name = plan.sections[0].name
    (r,) = c.interpret_action(plan, {name: result(name, "members 642229 642234 786345\nsent 642234\nsent 786345\n"
                                                        "sent 642229\ndone")}, cx)
    assert r.ok and r.message == "signalled 3 processes" and r.target == job.key and len(r.details) == 5
    (r,) = c.interpret_action(plan, {}, cx)
    assert not r.ok and r.message == "not run"
    (r,) = c.interpret_action(plan, {name: result(name, "members 1\nsent 1", 124)}, cx)
    assert not r.ok and r.message.startswith("timed out")


def test_transport_budget_covers_long_grace_periods(task: tuple[ProcessCollector, Job]) -> None:
    """Review BACK-04 (with CORE-12): a 120 s grace (section timeout 135 s) was cut off by the 90 s probe cap, and
    several selected tasks run one after the other."""
    from runfleet.probe import framing
    c, job = task
    other = dataclasses.replace(job, key=dataclasses.replace(job.key, id="p999-1"),
                                raw={**job.raw, "root_pid": "999", "root_start": "1"})
    plan = c.plan_action("task.terminate", [job, other], {"grace_s": 120}, ctx("lynx"))
    assert [s.timeout for s in plan.sections] == [135.0, 135.0]
    assert framing.local_timeout(framing.make_script(plan.sections, "action")) >= 270
