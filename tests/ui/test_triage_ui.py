"""Failure triage in the UI (§5.2.8): the cause chip in job detail and the jobs preview, `L` landing the log viewer on
the failure line, `]` / `[` error navigation, the Efficiency box, pending advice and the keymap entries."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from textual.widgets import Static

from runfleet.models import JobKey, JobState
from runfleet.ui import keymap, theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.job_detail import (JobDetailScreen, cause_line, efficiency_box_lines, efficiency_summary,
                                            headroom_line)
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.widgets.job_preview import cause_text
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.log_view import LogView
from runfleet.util.efficiency import efficiency
from runfleet.util.logscan import cause_of
from tests.ui.fakes import LOG_LINES, T0, FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
FAILED = JobKey("k2", "2330539")
OOM_LOG = [*LOG_LINES, "Traceback (most recent call last):", '  File "train.py", line 412, in <module>',
           "    loss.backward()",
           "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 139 GiB",
           "UserWarning: something deprecated"]


def _with_log(svc: FakeService, key: JobKey, lines: list[str]) -> None:
    job = svc.store.job(key)
    assert job is not None and job.logs
    svc.logs[job.logs[0].path or ""] = list(lines)


def _plain(app: RunFleetApp, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).render())


async def _settle(pilot: Any, rounds: int = 6) -> None:
    for _ in range(rounds):
        await pilot.pause(0.05)


async def test_job_detail_cause_chip_efficiency_and_landing(make_app: MakeApp) -> None:
    app, svc = make_app()
    _with_log(svc, FAILED, OOM_LOG)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_job(FAILED)
        await _settle(pilot)
        screen = app.screen
        assert isinstance(screen, JobDetailScreen)
        assert screen.cause is not None and screen.cause.code == "OOM-GPU"
        line = _plain(app, "#cause")
        assert "OOM-GPU" in line and "torch.OutOfMemoryError: CUDA out of memory" in line and "L: log" in line
        assert screen.query_one("#cause", Static).tooltip == screen.cause.plain()
        box = screen.query_one("#gpu-box", Static)
        assert box.border_title == "Efficiency"
        body = str(box.render())
        assert "cpu   55% of 8 cores" in body and "peak 9.9G of 48G" in body and "partial run" in body
        await pilot.press("L")
        await _settle(pilot)
        log = app.screen
        assert isinstance(log, LogScreen) and log.open_at is None
        view = log.query_one(LogView)
        assert view.search_note is not None and view.search_note.startswith("cause 1/1")
        assert "torch.OutOfMemoryError" in view.search_note and not view.follow
        await pilot.press("right_square_bracket")              # the OOM line itself is the next error stop
        await pilot.pause()
        assert view.search_note is not None and view.search_note.startswith("error / warning 3/3")
        assert "UserWarning" in view.search_note
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert "error / warning 2/3" in (view.search_note or "")
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert "Traceback" in (view.search_note or "")


async def test_cause_chip_without_log_evidence_and_toast(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        app.open_job(FAILED)
        await _settle(pilot)
        assert "EXIT 1" in _plain(app, "#cause") and "L: log" not in _plain(app, "#cause")
        await pilot.press("L")
        await _settle(pilot)
        assert isinstance(app.screen, LogScreen) and app.screen.open_at is None
        assert any("no failure line" in n.message for n in app._notifications)


async def test_running_job_has_no_chip_and_shows_headroom(make_app: MakeApp) -> None:
    app, svc = make_app()
    key = JobKey("k2", "2332861")
    job = svc.store.job(key)
    assert job is not None
    svc.put_job(dataclasses.replace(job, rss_bytes=40 * 1024 ** 3))
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_job(key)
        await _settle(pilot)
        assert not app.screen.query_one("#cause", Static).display
        box = app.screen.query_one("#gpu-box", Static)
        assert box.border_title != "Efficiency" and "room  rss 40G of 48G (83%) · 8.0G headroom" in str(box.render())


async def test_log_viewer_error_navigation_and_highlight(make_app: MakeApp) -> None:
    app, svc = make_app()
    key = JobKey("k2", "2332861")
    _with_log(svc, key, [*LOG_LINES, "RuntimeError: first", *LOG_LINES, "WARNING: second", "epoch 45"])
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_job(key)
        await _settle(pilot)
        await pilot.press("l")
        await _settle(pilot)
        log = app.screen
        assert isinstance(log, LogScreen)
        view = log.query_one(LogView)
        await pilot.press("left_square_bracket")              # from the bottom: the last stop on screen
        await pilot.pause()
        assert (view.search_note or "").startswith("error / warning 2/2") and not view.follow
        await pilot.press("right_square_bracket")             # wraps around
        await pilot.pause()
        assert (view.search_note or "").startswith("error / warning 1/2")
        styles = {str(seg.style) for strip in view.richlog.lines for seg in strip if "RuntimeError" in seg.text}
        assert styles and all(theme.RED in s for s in styles)            # error lines are highlighted
        hints = [k for k in keymap.hints(("log", "global")) if k[1] == "error"]
        assert hints == [("[/]", "error", "prev_error")]


async def test_preview_shows_the_chip_from_the_peek(make_app: MakeApp) -> None:
    app, svc = make_app()
    _with_log(svc, FAILED, OOM_LOG[:-1])
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = app.screen.query_one(JobsView)
        while view.preset != "all":
            await pilot.press("f")
        await pilot.pause()
        app.screen.query_one(JobTable).move_to_key(str(FAILED))
        await _settle(pilot, 10)
        text = str(view.preview.render())
        assert "OOM-GPU" in text and "torch.OutOfMemoryError" in text
        assert "! exit 1" not in text                          # the chip replaces an ended job's summary hint


def test_pure_renderers() -> None:
    svc = FakeService()
    failed = svc.store.job(FAILED)
    assert failed is not None
    line = cause_line(failed, cause_of(failed, None, OOM_LOG), 60)
    assert line is not None and line.cell_len == 60 and line.plain.startswith(" OOM-GPU  torch.OutOf")
    narrow = cause_line(failed, cause_of(failed, None, OOM_LOG), 30)
    assert narrow is not None and "L: log" not in narrow.plain and narrow.cell_len == 30
    pending = dataclasses.replace(failed, state=JobState.PENDING, reason="Resources", est_start=True, start_time=T0,
                                  raw={"reason_advice": "next in line, but the GPUs are busy"})
    p = cause_line(pending, None, 140)
    assert p is not None and p.plain.startswith(" resources  next in line") and "Slurm's guess" in p.plain
    pt = cause_text(pending, None, 80)
    assert pt is not None and pt.plain.startswith("→ next in line")
    running = svc.store.job(JobKey("k2", "2332861"))
    assert running is not None and cause_line(running, None, 80) is None and cause_text(running, None, 80) is None
    eff = efficiency(failed, ())
    assert [x.plain.split()[0] for x in efficiency_box_lines(eff, 50)] == ["time", "cpu", "mem", "gpu", "next", "note"]
    assert efficiency_box_lines(None, 50)[0].plain.startswith("n/a")
    summary = efficiency_summary(eff)
    assert summary is not None and summary.plain.startswith("eff  time 1% · cpu – · mem – · gpu –")
    assert headroom_line(eff, 50) is None and efficiency_summary(None) is None


def test_keymap_entries_and_no_conflicts() -> None:
    assert keymap.find("log", "next_error") is not None and keymap.find("log", "prev_error") is not None
    assert keymap.find("job_detail", "open_cause") is not None
    for ctx in ("log", "job_detail"):
        keys = [k for entry in keymap.keys_of(ctx) for k in entry.keys]
        assert len(keys) == len(set(keys)), ctx
    help_rows = dict(keymap.help_entries(("log", "global"))["nav"])
    assert help_rows["[ ]"] == "previous / next error or warning line"
