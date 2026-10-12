"""Regression tests for the ui_data review findings (Jobs view marks and cursor, GPUs view freshness, job detail data
and layout, log following)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from textual.widgets import DataTable, Static

from runfleet.core.service import LogUpdate
from runfleet.models import ConnState, ErrorKind, HostError, JobDetail, JobKey, JobState, LogRef
from runfleet.ui import keymap
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens import job_detail as jd
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.widgets.gpus_view import GPUsView
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.log_view import LogView
from tests.ui.fakes import T0, FakeService, slurm_job

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
FLUSH = 0.3


def jobs_view(app: RunFleetApp) -> JobsView:
    return app.screen.query_one(JobsView)


async def goto_jobs_all(pilot: Any, app: RunFleetApp) -> None:
    await pilot.press("tab")
    await pilot.pause()
    while jobs_view(app).preset != "all":
        await pilot.press("f")
    await pilot.pause()


async def open_detail(pilot: Any, app: RunFleetApp, key: JobKey, settle: float = 0.3) -> JobDetailScreen:
    app.open_job(key)
    await pilot.pause(settle)
    assert isinstance(app.screen, JobDetailScreen)
    return app.screen


def text_of(app: RunFleetApp, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).render())


# ---------------------------------------------------------------- UI-02 folded groups
async def test_folded_group_members_are_hidden_marks(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs_all(pilot, app)
        view = jobs_view(app)
        await pilot.press("V")
        await pilot.pause()
        assert any(k.startswith("k2:") for k in view.marks)
        await pilot.press("z")                                  # group by host
        await pilot.pause()
        t = app.screen.query_one(JobTable)
        assert t.order[0] == "§host:k2"
        t.move_cursor(row=0)
        await pilot.press("enter")                              # fold k2
        await pilot.pause()
        targets = view.action_targets()
        assert targets and not any(k.host == "k2" for k in targets)
        assert "hidden — not acted on" in view.status_text(160).plain
        header = "".join(t.get_cell("§host:k2", c).plain for c, _w in t.cols)
        assert "marked (folded: not acted on)" in header


# ---------------------------------------------------------------- UI-15 cursor after the selected row vanishes
async def test_cursor_stays_in_place_when_the_selected_job_disappears(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs_all(pilot, app)
        t = app.screen.query_one(JobTable)
        t.move_cursor(row=7)
        await pilot.pause()
        gone, below = t.order[7], t.order[8]
        host, _, jid = gone.partition(":")
        svc.store.remove_job(JobKey(host, jid))
        await pilot.pause(FLUSH)
        assert gone not in t.order
        assert t.cursor_key == below and t.cursor_row == 7


async def test_steps_cursor_survives_a_poll(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_detail(pilot, app, JobKey("k2", "2330539"))
        steps = app.screen.query_one("#steps-table", DataTable)
        assert steps.row_count == 3
        steps.move_cursor(row=1)
        svc.tick(1)                                             # a successful poll of k2 -> FleetChanged
        await pilot.pause(FLUSH)
        assert steps.cursor_row == 1


# ---------------------------------------------------------------- UI-04 GPU rows keep their own freshness
async def test_old_gpu_readings_of_an_answering_host_are_stale(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        assert not any(r.stale for r in view.rows if r.host == "labstation")
        snap = svc.store.hosts["labstation"].snapshot
        assert snap is not None
        old = tuple(dataclasses.replace(g, sampled_at=T0 - timedelta(minutes=10)) for g in snap.gpus)
        svc.store.replace_snapshot("labstation", gpus=old)
        st = svc.store.hosts["labstation"].status
        svc.store.hosts["labstation"].status = dataclasses.replace(st, state=ConnState.DEGRADED)
        view.refresh_data()
        rows = [r for r in view.rows if r.host == "labstation" and not r.aggregate]
        assert rows and all(r.stale and r.age_s == pytest.approx(600) for r in rows)
        assert "labstation 10m old" in str(view.query_one("#gpus-note").render())
        assert not any(r.stale for r in view.rows if r.host == "k2")


# ---------------------------------------------------------------- UI-05 newest GPU sample wins in job detail
def test_pick_usage_prefers_the_newer_sample() -> None:
    job = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(hours=1), util=98, node="n-40-01")
    assert job.gpu is not None
    newer = dataclasses.replace(job.gpu, util_mean_pct=1.0, sampled_at=T0)
    d = JobDetail(job, {}, (), (), (), job.gpu, (), "", {}, (), T0)
    assert jd.pick_usage(dataclasses.replace(job, gpu=newer), d) is newer
    assert jd.pick_usage(job, dataclasses.replace(d, gpu=newer)) is newer
    assert jd.pick_usage(dataclasses.replace(job, gpu=None), d) is job.gpu


async def test_detail_gpu_box_follows_newer_polled_samples(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        key = JobKey("k2", "2332861")
        await open_detail(pilot, app, key)
        assert "98%" in text_of(app, "#gpu-box")
        job = svc.store.job(key)
        assert job is not None and job.gpu is not None
        svc.put_job(dataclasses.replace(job, gpu=dataclasses.replace(job.gpu, util_mean_pct=1.0, sampled_at=T0)))
        await pilot.pause(FLUSH)
        box = text_of(app, "#gpu-box")
        assert "98%" not in box and " 1%" in box


# ---------------------------------------------------------------- UI-06 detail errors are shown
async def test_detail_probe_errors_reach_the_banner(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    real = svc.job_detail

    async def partial(key: JobKey, *, max_age_s: float = 30) -> JobDetail:
        d = await real(key, max_age_s=max_age_s)
        err = HostError(ErrorKind.TIMEOUT, "sacct steps timed out", section="slurm.steps")
        return dataclasses.replace(d, steps=(), errors=(err,))

    monkeypatch.setattr(svc, "job_detail", partial)
    async with app.run_test(size=BIG) as pilot:
        await open_detail(pilot, app, JobKey("k2", "2330539"))
        assert "timed out" in text_of(app, "#banner")
        assert "steps unavailable: sacct steps timed out" in text_of(app, "#why")


# ---------------------------------------------------------------- UI-08 / UI-09 / UI-10 / UI-07 log following
async def test_detail_starts_the_log_once_its_path_is_known(make_app: MakeApp) -> None:
    app, svc = make_app()
    pending = slurm_job("k2", "2338403", "late-start", JobState.PENDING, reason="Priority")
    unresolved = (LogRef("stdout", None, pattern="%N.out", source="scontrol"),)
    svc.put_job(dataclasses.replace(pending, logs=unresolved))
    async with app.run_test(size=BIG) as pilot:
        screen = await open_detail(pilot, app, pending.key)
        assert screen.follower is not None and screen.follower.state == "waiting"
        assert not svc.sessions
        started = slurm_job("k2", "2338403", "late-start", JobState.RUNNING, elapsed=timedelta(seconds=30),
                            node="n-40-02")
        svc.put_job(dataclasses.replace(started, logs=(LogRef("stdout", "/w/n-40-02.out", source="scontrol"),)))
        await pilot.pause(FLUSH)
        assert [s.ref.path for s in svc.sessions] == ["/w/n-40-02.out"]
        svc.tick(1)
        await pilot.pause(FLUSH)
        assert len(svc.sessions) == 1                           # same path: never restarted on a poll


async def test_switching_to_an_unresolved_stream_stops_the_old_one(make_app: MakeApp) -> None:
    app, svc = make_app()
    key = JobKey("k2", "2332861")
    job = svc.store.job(key)
    assert job is not None
    svc.put_job(dataclasses.replace(job, logs=(job.logs[0], LogRef("stderr", None, pattern="%x.err"))))
    async with app.run_test(size=BIG) as pilot:
        app.push_screen(LogScreen(key))
        await pilot.pause(0.2)
        view = app.screen.query_one(LogView)
        assert view.count == 7
        await pilot.press("e")
        await pilot.pause(0.2)
        assert view.count == 0 and svc.sessions[0].closed
        assert view.follow                                      # clearing the pane is no scroll-up
        svc.sessions[0].queue.put_nowait(LogUpdate(lines=("a stdout line",)))
        await pilot.pause(0.2)
        assert view.count == 0                                  # nothing of stdout under the stderr label


async def test_dead_stream_is_not_following_and_reconnects(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.push_screen(LogScreen(JobKey("k2", "2332861")))
        await pilot.pause(0.2)
        screen = app.screen
        assert isinstance(screen, LogScreen) and screen.follower is not None
        svc.push_log(LogUpdate(eof=True))
        await pilot.pause(0.2)
        assert screen.follower.dead
        assert "following" not in str(screen.query_one("#live").render())
        assert "following" not in str(screen.query_one("#streams").render())
        screen.refresh_time()
        assert len(svc.sessions) == 1                           # backoff: not before 5 s
        svc.clock.advance(6)
        screen.refresh_time()
        await pilot.pause(0.2)
        assert len(svc.sessions) == 2 and screen.follower.state == "live"
        assert screen.follower.attempts == 1


async def test_reload_reopens_a_dead_detail_stream(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_detail(pilot, app, JobKey("k2", "2332861"))
        svc.push_log(LogUpdate(eof=True))
        await pilot.pause(0.2)
        assert screen.follower is not None and screen.follower.dead
        await pilot.press("r")
        await pilot.pause(0.2)
        assert len(svc.sessions) == 2 and screen.follower.state == "live"


async def test_detail_restarts_its_stream_on_resume(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_detail(pilot, app, JobKey("k2", "2332861"))
        await pilot.press("l")
        await pilot.pause(0.2)
        assert isinstance(app.screen, LogScreen)
        svc.push_log(LogUpdate(eof=True), index=0)              # the viewer took the host's only stream slot
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert app.screen is screen and screen.follower is not None
        assert len(svc.sessions) == 3 and screen.follower.state == "live"


# ---------------------------------------------------------------- UI-11 scrolling up pauses following
async def test_scrolling_up_pauses_follow(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.push_screen(LogScreen(JobKey("k2", "2332861")))
        await pilot.pause(0.2)
        view = app.screen.query_one(LogView)
        svc.push_log(LogUpdate(lines=tuple(f"line {i}" for i in range(200))))
        await pilot.pause(0.2)
        log = view.richlog
        assert view.follow and log.scroll_y == log.max_scroll_y
        log.scroll_to(y=0, animate=False)
        await pilot.pause(0.1)
        assert not view.follow
        svc.push_log(LogUpdate(lines=("one more",)))
        await pilot.pause(0.2)
        assert log.scroll_y == 0 and "paused" in str(view.query_one("#live").render())
        log.scroll_end(animate=False)                           # back at the bottom: following again
        await pilot.pause(0.1)
        assert view.follow


# ---------------------------------------------------------------- UI-18 search across a wrap
async def test_search_finds_a_phrase_split_by_wrapping(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.push_screen(LogScreen(JobKey("k2", "2332861")))
        await pilot.pause(0.2)
        view = app.screen.query_one(LogView)
        width = view.richlog.lines[0].cell_length                # the render width (strips are padded to it)
        svc.push_log(LogUpdate(lines=("x" * (width - 7) + " out of memory",)))
        await pilot.pause(0.2)
        assert not any("out of memory" in s.text for s in view.richlog.lines)     # the wrap splits it
        assert view.search("out of memory") == len(view.richlog.lines) - 2


# ---------------------------------------------------------------- UI-13 layout follows the terminal size
async def test_detail_layout_switches_on_resize(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await open_detail(pilot, app, JobKey("k2", "2332861"))
        assert screen.summary_log
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.2)
        assert not screen.summary_log and screen.has_class("-compact")
        log = screen.query_one("#summary-log", LogView)
        assert log.region.height >= 3 and log.region.bottom <= 24
        assert screen.query_one("#summary-compact").display
        await pilot.resize_terminal(120, 40)
        await pilot.pause(0.2)
        assert screen.summary_log and not screen.query_one("#summary-compact").display
        assert log.count == 7                                   # same views: the stream stayed attached


# ---------------------------------------------------------------- UI-20 "c to log in" only where c connects
async def test_login_hints_match_the_keymap(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        app.set_scope("lynx")
        svc.store.hosts["lynx"].snapshot = None
        view.refresh_data()
        note = str(view.query_one("#gpus-note").render())
        bound = keymap.find("gpus", "app.connect_host") is not None
        assert ("c to log in" in note) == bound and ("log in from Fleet (c)" in note) != bound
    state = svc.store.hosts["lynx"]
    banner = jd.stale_banner(state, T0, context="job_detail")
    assert banner is not None
    in_detail = keymap.find("job_detail", "app.connect_host") is not None
    assert ("press c to log in" in banner.plain) == in_detail


# ---------------------------------------------------------------- UI-32 `r` never restarts a probe in flight
async def test_reload_coalesces_while_a_detail_probe_runs(make_app: MakeApp,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    real = svc.job_detail
    gate = asyncio.Event()
    calls: list[float] = []

    async def slow(key: JobKey, *, max_age_s: float = 30) -> JobDetail:
        calls.append(max_age_s)
        await gate.wait()
        return await real(key, max_age_s=max_age_s)

    monkeypatch.setattr(svc, "job_detail", slow)
    async with app.run_test(size=BIG) as pilot:
        await open_detail(pilot, app, JobKey("k2", "2332861"), settle=0.1)
        for _ in range(3):
            await pilot.press("r")
            await pilot.pause(0.05)
        assert len(calls) == 1                                  # the first probe still runs, never restarted
        gate.set()
        await pilot.pause(0.3)
        assert calls == [30, jd.DETAIL_MIN_GAP]                 # one queued refresh, served by the cache if recent
        assert app.screen.detail is not None                    # type: ignore[attr-defined]


# ---------------------------------------------------------------- UI-55 no peek after leaving the Jobs view
async def test_no_log_peek_after_leaving_the_jobs_view(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JobsView, "PEEK_DWELL", 0.3)
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs_all(pilot, app)
        await pilot.pause(0.5)
        before = len(svc.peeks)
        await pilot.press("j")                                  # arms the dwell timer
        await pilot.press("tab")                                # GPUs view before it fires
        await pilot.pause(0.6)
        assert isinstance(app.screen, MainScreen)
        assert len(svc.peeks) == before
