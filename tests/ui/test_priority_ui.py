"""Pending priority evidence in the UI (§5.2.10): the job detail's Priority box (in place of a pending job's "n/a — job
pending" GPU box) and its one-line form in the compact detail and the Jobs preview. Evidence only: no start time, no
queue position."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import Any

from textual.widgets import Static

from runfleet.models import FairShare, JobKey, PendingPriority, PriorityFactor
from runfleet.ui import render
from runfleet.ui.screens.job_detail import JobDetailScreen, priority_box_lines, priority_box_title
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from tests.ui.fakes import T0, FakeService

PENDING = JobKey("k2", "2334564")
UNRANKED = ("this cluster does not rank by priority: pending jobs start in submit order as resources free up "
            "(backfill can start short jobs earlier)")
HINT = ("fairshare 0.03: your group (lab) used much more than its share recently; it recovers with a half-life of 14 "
        "days")


def _evidence(**kw: Any) -> PendingPriority:
    base = PendingPriority(
        ranked=True, priority=3017.0, partition="gpu-preempt",
        factors=(PriorityFactor("fairshare", 3000.0, 100000.0), PriorityFactor("jobsize", 15.0, 1000.0),
                 PriorityFactor("age", 2.0, 1000.0), PriorityFactor("partition", 0.0, 1000.0)),
        dominant="fairshare", others=(("lab", 2017.0),),
        fairshare=(FairShare("lab", "alice", "1", 1.0, 8123456, 0.833333, 0.03, 1.2),
                   FairShare("dept", "alice", "1", 0.032258, 0, 0.0, 0.95, None)),
        hints=("fairshare carries most of its priority (3,000 of 3,017): the fair-share standing of you and your "
               "account", HINT),
        sampled_at=T0 - timedelta(minutes=4), half_life_s=14 * 86400.0)
    return dataclasses.replace(base, **kw)


def _pending(svc: FakeService, p: PendingPriority | None) -> None:
    job = svc.store.job(PENDING)
    assert job is not None
    svc.put_job(dataclasses.replace(job, account="lab", priority=p))


# ---------------------------------------------------------------- pure renderers
def test_summary_line() -> None:
    line = render.priority_summary(_evidence(), "lab", T0).plain
    assert line == "prio 3,017 · fairshare dominates · fairshare 0.03 · 4m ago"
    assert render.priority_summary(_evidence(), "dept", T0).plain.endswith("fairshare 0.95 · 4m ago")
    unread = render.priority_summary(_evidence(priority=None, dominant=None, factors=(), sampled_at=None), "x", T0)
    assert unread.plain == "prio not read yet (sprio runs at most every 10 min)"   # two accounts, none of the job's
    unranked = render.priority_summary(PendingPriority(ranked=False, hints=(UNRANKED,)), "lab", T0).plain
    assert unranked.startswith("prio not ranked here: submit order") and "0" not in unranked


def test_priority_box() -> None:
    job = FakeService().store.job(PENDING)
    assert job is not None
    job = dataclasses.replace(job, account="lab", priority=_evidence())
    lines = [x.plain.rstrip() for x in priority_box_lines(job, 72)]
    assert lines[0].split() == ["priority", "3,017", "in", "gpu-preempt", "·", "lab", "2,017"]
    fs = next(x for x in lines if x.startswith("fairshare "))
    assert "3,000 / 100,000" in fs and fs.endswith("◀ dominates")
    assert any(x.startswith("job size ") and "15 / 1,000" in x for x in lines)
    share = next(x for x in lines if x.startswith("share "))
    assert "lab  0.03 · usage 83.3% of share 100.0%" in share
    assert any("dept  0.95" in x and "no recent usage" in x for x in lines)
    text = " ".join(x.strip() for x in lines)
    assert "• fairshare 0.03: your group (lab) used much more than its share recently" in text
    assert "half-life of 14 days" in text
    assert all(len(x) <= 72 for x in lines)
    assert priority_box_title(job, T0) == "Priority · sprio 4m ago"
    narrow = [x.plain.rstrip() for x in priority_box_lines(job, 46)]           # a narrow right column: nothing cut
    assert all(len(x) <= 46 and "…" not in x for x in narrow)
    assert any(x.startswith("fairshare ") and x.endswith("3,000 / 100,000 ◀") for x in narrow)
    assert "lab  0.03 · 83% of 100%" in narrow[next(i for i, x in enumerate(narrow) if x.startswith("share "))]
    unranked = dataclasses.replace(job, priority=PendingPriority(ranked=False, hints=(UNRANKED,)))
    ulines = [x.plain for x in priority_box_lines(unranked, 50)]
    assert ulines[0].startswith("• this cluster does not rank by priority") and not any("/" in x for x in ulines)
    assert priority_box_lines(dataclasses.replace(job, priority=None), 50)[0].plain == "n/a — job pending"


# ---------------------------------------------------------------- screens
async def test_job_detail_shows_the_priority_box(make_app: Any) -> None:
    app, svc = make_app()
    _pending(svc, _evidence())
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.open_job(PENDING)
        for _ in range(6):
            await pilot.pause(0.05)
        assert isinstance(app.screen, JobDetailScreen)
        box = app.screen.query_one("#gpu-box", Static)
        assert str(box.border_title).startswith("Priority · sprio")
        text = str(box.render())
        assert "3,000 / 100,000" in text and "◀ dominates" in text and "fairshare 0.03" in text
        assert "n/a — job pending" not in text


async def test_compact_detail_and_preview_carry_one_line(make_app: Any) -> None:
    app, svc = make_app()
    _pending(svc, _evidence())
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        app.open_job(PENDING)
        for _ in range(6):
            await pilot.pause(0.05)
        compact = str(app.screen.query_one("#summary-compact", Static).render())
        assert "prio 3,017 · fairshare dominates · fairshare 0.03" in compact
    app, svc = make_app()
    _pending(svc, _evidence())
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = app.screen.query_one(JobsView)
        app.screen.query_one(JobTable).move_to_key(str(PENDING))
        for _ in range(6):
            await pilot.pause(0.05)
        assert "prio 3,017 · fairshare dominates · fairshare 0.03" in str(view.preview.render())
    app, svc = make_app()
    _pending(svc, None)                                     # no evidence: no line
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = app.screen.query_one(JobsView)
        app.screen.query_one(JobTable).move_to_key(str(PENDING))
        for _ in range(6):
            await pilot.pause(0.05)
        assert "prio " not in str(view.preview.render())
