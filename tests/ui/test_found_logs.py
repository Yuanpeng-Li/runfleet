"""[WP3] `f` in the job detail and the log viewer: the files written under the job's workdir (`find_logs`, one probe in
the picker's worker), follow the picked one as the `file` stream, `o` / `e` back, `f` back to it; the "stdout quiet"
hint of the Logs tab; loading / empty / error states; own jobs with a workdir only."""

from __future__ import annotations

import asyncio
import dataclasses
import re
from datetime import timedelta
from typing import Any

import pytest
from rich.cells import cell_len
from rich.text import Text
from textual.widgets import OptionList, TabbedContent

from runfleet.core.logs import FIND_MAX_FILES, FIND_UNREADABLE, FoundLog, LogFiles, stdout_quiet_for
from runfleet.core.logs import find_window as logs_find_window
from runfleet.models import ErrorKind, HostError, Job, JobKey, JobState, LogRef
from runfleet.ui import keymap
from runfleet.ui.screens.found_logs import (FoundLogsScreen, find_problem, find_window, found_ref, incomplete_text,
                                            other_paths, own_paths, quiet_hint, rel_path, remember_pick, row_text,
                                            trailer_width, window_text)
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.log_view import LogView
from tests.ui.fakes import T0

BIG = (140, 40)
KEY = JobKey("k2", "2332861")
WD = "/shared/pilab/pilab/orb"
TRAIN = f"{WD}/runs/cifar/warmup/s42/train.log"
EVAL = f"{WD}/runs/cifar/warmup/s42/eval.log"
STDOUT = f"{WD}/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"


def _files(svc: Any) -> None:
    svc.found_files[WD] = [
        FoundLog(EVAL, 2_048, T0 - timedelta(minutes=5)),
        FoundLog(TRAIN, 3_250_000, T0 - timedelta(seconds=12)),
        FoundLog(STDOUT, 661, T0 - timedelta(hours=4)),
    ]
    svc.logs[TRAIN] = ["epoch 41 loss 0.31", "epoch 42 loss 0.30"]
    svc.logs[EVAL] = ["eval acc 0.91"]


def _job(svc: Any, key: JobKey = KEY) -> Job:
    job = svc.store.job(key)
    assert job is not None
    return job


async def _settle(pilot: Any, n: int = 12) -> None:
    for _ in range(n):
        await pilot.pause(0.03)


async def _detail(app: Any, pilot: Any, key: JobKey = KEY) -> JobDetailScreen:
    await pilot.pause()
    app.open_job(key)
    await _settle(pilot)
    screen = app.screen
    assert isinstance(screen, JobDetailScreen)
    return screen


async def _picker(app: Any, pilot: Any) -> FoundLogsScreen:
    await pilot.press("f")
    await _settle(pilot)
    screen = app.screen
    assert isinstance(screen, FoundLogsScreen)
    return screen


def _summary(screen: FoundLogsScreen) -> str:
    return str(screen.query_one("#fl-summary").render())


def _title(screen: Any) -> str:
    title = screen.query_one("#logs-log", LogView).border_title
    return title.plain if isinstance(title, Text) else Text.from_markup(str(title)).plain


# ---------------------------------------------------------------- pure helpers
def test_find_problem_own_jobs_with_a_workdir_only(service: Any) -> None:
    job = _job(service)
    assert find_problem(job) is None
    assert "your own jobs" in (find_problem(dataclasses.replace(job, is_mine=False, user="bob")) or "")
    assert "no workdir" in (find_problem(dataclasses.replace(job, workdir=None)) or "")
    assert find_problem(None) == "no job selected"


def test_since_is_the_start_and_none_before_a_real_start(service: Any) -> None:
    job = _job(service)
    assert job.start_time is not None
    assert find_window(job)[:2] == (job.start_time, None) == logs_find_window(job)      # the agent's rule too
    assert find_window(dataclasses.replace(job, est_start=True))[:2] == (None, None)
    assert find_window(dataclasses.replace(job, start_time=None))[:2] == (None, None)


def test_rel_path_and_own_paths(service: Any) -> None:
    assert rel_path(TRAIN, WD) == "runs/cifar/warmup/s42/train.log"
    assert rel_path(TRAIN, WD + "/") == "runs/cifar/warmup/s42/train.log"
    assert rel_path("/elsewhere/x.log", WD) == "/elsewhere/x.log"
    assert rel_path(WD + "x/a.log", WD) == WD + "x/a.log"             # a sibling directory is not under it
    assert own_paths(_job(service)) == {STDOUT: "stdout+stderr"}         # merged stderr: one path, both streams


def test_remember_pick_maps_the_job_s_own_stdout_back_to_its_stream(service: Any) -> None:
    job, found = _job(service), {}
    assert remember_pick(found, job, FoundLog(STDOUT, 661, T0)) == "stdout" and not found
    assert remember_pick(found, job, FoundLog(TRAIN, 9, T0)) == "file"
    assert found[job.key] == LogRef("file", TRAIN, source="found", size=9, mtime=T0)
    assert found_ref(FoundLog(TRAIN, 9, T0)).stream == "file"


def test_quiet_hint(service: Any) -> None:
    job = _job(service)
    stall = service.config.general.stall_after
    old = LogRef("stdout", STDOUT, size=661, mtime=T0 - timedelta(hours=13))
    assert quiet_hint(job, old, T0, stall) == "stdout quiet for 13h; f finds other log files in the workdir"
    fresh = LogRef("stdout", STDOUT, size=661, mtime=T0 - timedelta(seconds=30))
    assert quiet_hint(job, fresh, T0, stall) is None                     # tiny but just written
    tiny_old = LogRef("stdout", STDOUT, size=661, mtime=T0 - timedelta(minutes=12))
    assert quiet_hint(job, tiny_old, T0, stall) == "stdout quiet for 12m; f finds other log files in the workdir"
    big_recent = LogRef("stdout", STDOUT, size=1 << 20, mtime=T0 - timedelta(minutes=12))
    assert quiet_hint(job, big_recent, T0, stall) is None                # below stall_after, not tiny
    assert quiet_hint(dataclasses.replace(job, state=JobState.COMPLETED), old, T0, stall) is None
    assert quiet_hint(dataclasses.replace(job, is_mine=False), old, T0, stall) is None
    assert quiet_hint(job, LogRef("stdout", STDOUT), T0, stall) is None  # no stat yet


def test_the_key_is_documented_in_both_contexts_and_the_hint_bar() -> None:
    for ctx in ("job_detail", "log"):
        k = keymap.find(ctx, "find_logs")
        assert k is not None and k.keys == ("f",) and k.hint == "other logs"
        keys = [key for entry in keymap.KEYMAP[ctx] for key in entry.keys]
        assert keys.count("f") == 1                                       # free in that context
        assert any(label == "f" for label, _text in keymap.help_entries([ctx, "global"])["view"])
        assert ("f", "other logs", "find_logs") in keymap.hints([ctx, "global"])
    assert "found_logs" in keymap.CONTEXT_TITLES and "Other log files" in keymap.markdown()


# ---------------------------------------------------------------- job detail
async def test_pick_follows_the_file_and_o_and_f_switch(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        picker = await _picker(app, pilot)
        job = _job(svc)
        assert svc.called("find_logs") == [("k2", WD, job.start_time, None)]
        assert "3 files written under" in _summary(picker) and "newest first" in _summary(picker)
        rows = [str(picker.query_one("#fl-list").get_option_at_index(i).prompt) for i in range(3)]
        assert rows[0].startswith("runs/cifar/warmup/s42/train.log") and "3.1M" in rows[0]
        assert rows[1].startswith("runs/cifar/warmup/s42/eval.log") and "5m ago" in rows[1]
        assert "a02_2332861.out" in rows[2] and "· stdout+stderr  ◀ shown" in rows[2]   # its own, marked
        assert picker.query_one("#fl-list").highlighted == 0                         # newest non-own file first
        await pilot.press("enter")
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "file"
        assert svc.called("stream_log")[-1] == ("k2", "file", TRAIN)
        assert app.found_logs[KEY].path == TRAIN and app.found_logs[KEY].source == "found"
        assert list(detail.query_one("#logs-log", LogView)._raw) == svc.logs[TRAIN]
        title = _title(detail)
        assert title.startswith("train.log · ") and TRAIN in title and "f(train.log)" in title
        await pilot.press("o")                                   # back to stdout
        await _settle(pilot)
        assert detail.stream == "stdout" and svc.called("stream_log")[-1] == ("k2", "stdout", STDOUT)
        assert "f(train.log)" in _title(detail)                 # the way back stays on the tab line
        calls = len(svc.called("find_logs"))
        await pilot.press("f")                                   # remembered: straight back, no probe
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "file" and len(svc.called("find_logs")) == calls
        assert svc.called("stream_log")[-1] == ("k2", "file", TRAIN)
        picker = await _picker(app, pilot)                       # on the file: f picks another
        rows = [str(picker.query_one("#fl-list").get_option_at_index(i).prompt) for i in range(3)]
        assert "◀ shown" in rows[0]
        await pilot.press("j", "enter")
        await _settle(pilot)
        assert detail.stream == "file" and svc.called("stream_log")[-1] == ("k2", "file", EVAL)
        assert "f(eval.log)" in _title(detail)


async def test_picking_the_job_s_own_stdout_goes_back_to_it(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        await _picker(app, pilot)
        await pilot.press("j", "j", "enter")
        await _settle(pilot)
        assert detail.stream == "stdout" and KEY not in app.found_logs


async def test_loading_then_empty_then_rescan(make_app: Any) -> None:
    app, svc = make_app()
    svc.find_gate = asyncio.Event()
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        assert _summary(picker).startswith("looking for log files under") and "since the job started" in \
            _summary(picker)
        svc.find_gate.set()
        await _settle(pilot)
        assert _summary(picker).startswith("no log file (*.log *.out") and not picker.query_one("#fl-list").display
        _files(svc)
        await pilot.press("r")
        await _settle(pilot)
        assert len(svc.called("find_logs")) == 2 and "3 files" in _summary(picker)
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.stream == "stdout"


async def test_typed_error_and_file_error_and_truncation(make_app: Any) -> None:
    app, svc = make_app()
    svc.find_errors["k2"] = HostError(ErrorKind.TIMEOUT, "log.find timed out after 15 s", section="log.find")
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        assert _summary(picker) == "k2: log.find timed out after 15 s"
        del svc.find_errors["k2"]
        svc.find_file_error = "find on this host cannot list files by time (BusyBox): unsupported"
        await pilot.press("r")
        await _settle(pilot)
        assert "cannot list the files under" in _summary(picker) and "unsupported" in _summary(picker)
        svc.find_file_error = None
        svc.found_files[WD] = [FoundLog(f"{WD}/runs/r{i}/train.log", 10, T0 - timedelta(seconds=i))
                               for i in range(FIND_MAX_FILES + 4)]
        await pilot.press("r")
        await _settle(pilot)
        assert f"only the newest {FIND_MAX_FILES}: more were written" in _summary(picker)
        assert picker.query_one("#fl-list").option_count == FIND_MAX_FILES


async def test_a_raising_service_is_an_error_line(make_app: Any) -> None:
    app, svc = make_app()

    async def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no transport for k2")

    svc.find_logs = boom
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        assert "no transport for k2" in _summary(picker)


async def test_other_users_jobs_and_jobs_without_workdir_are_refused(make_app: Any) -> None:
    app, svc = make_app()
    job = _job(svc)
    svc.put_job(dataclasses.replace(job, is_mine=False, user="bob"))
    other = JobKey("k2", "2332862")
    svc.put_job(dataclasses.replace(_job(svc, other), workdir=None))
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        toasts: list[str] = []
        app.toast = lambda msg, **_k: toasts.append(msg)       # type: ignore[method-assign]
        await pilot.press("f")
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and "your own jobs" in toasts[-1]
        app.pop_screen()
        await _detail(app, pilot, other)
        await pilot.press("f")
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and "no workdir known" in toasts[-1]
        assert svc.called("find_logs") == []


async def test_logs_tab_hint_when_stdout_went_quiet(make_app: Any) -> None:
    app, svc = make_app()
    job = _job(svc)
    out = dataclasses.replace(job.logs[0], size=661, mtime=T0 - timedelta(hours=13))
    svc.put_job(dataclasses.replace(job, logs=(out, *job.logs[1:])))
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        hint = detail.query_one("#logs-hint")
        assert hint.display and str(hint.render()) == "stdout quiet for 13h; f finds other log files in the workdir"
        _files(svc)
        await _picker(app, pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert not hint.display                                # on the found file: nothing to hint
        await pilot.press("o")
        await _settle(pilot)
        assert hint.display


async def test_no_hint_for_a_fresh_stdout(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        assert not detail.query_one("#logs-hint").display


async def test_a_workstation_task_looks_in_its_cwd(make_app: Any) -> None:
    app, svc = make_app()
    task = next(j for j in svc.store.jobs(["labstation"]) if j.is_mine and j.workdir)
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot, task.key)
        await _picker(app, pilot)
        assert svc.called("find_logs") == [("labstation", task.workdir, *logs_find_window(task))]


# ---------------------------------------------------------------- log viewer
async def test_log_viewer_follows_a_picked_file(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.push_screen(LogScreen(KEY))
        await _settle(pilot)
        screen = app.screen
        assert isinstance(screen, LogScreen)
        await _picker(app, pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert app.screen is screen and screen.stream == "file"
        assert svc.called("stream_log")[-1] == ("k2", "file", TRAIN)
        assert screen.query_one(FleetHeader).crumbs[-1] == "train.log"
        streams = str(screen.query_one("#streams").render())
        assert "f(train.log)" in streams and TRAIN[-30:] in streams
        assert "loaded" in streams and " of " not in streams.split("loaded")[1].split("·")[0]
        copied: list[str] = []
        app.copy_to_clipboard = copied.append                   # type: ignore[method-assign]
        await pilot.press("y")
        assert copied == [TRAIN]
        await pilot.press("o")
        await _settle(pilot)
        assert screen.stream == "stdout" and screen.query_one(FleetHeader).crumbs[-1] == "stdout"
        await pilot.press("f")
        await _settle(pilot)
        assert app.screen is screen and screen.stream == "file"


async def test_the_pick_is_remembered_per_job_for_the_viewer_and_the_pager(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        await _picker(app, pilot)
        await pilot.press("enter")
        await _settle(pilot)
        await pilot.press("l")                                 # the full viewer opens on the same file
        await _settle(pilot)
        viewer = app.screen
        assert isinstance(viewer, LogScreen) and viewer.stream == "file"
        assert svc.called("stream_log")[-1] == ("k2", "file", TRAIN)
        refs: list[LogRef] = []
        app.open_pager = lambda host, ref: refs.append(ref)    # type: ignore[method-assign]
        app.action_pager()
        assert refs and refs[-1].path == TRAIN
        app.pop_screen()
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "file"
        other = JobKey("k2", "2332862")
        app.open_job(other)
        await _settle(pilot)
        assert app.screen.stream == "stdout" and other not in app.found_logs


async def test_log_viewer_hint_line(make_app: Any) -> None:
    app, svc = make_app()
    job = _job(svc)
    out = dataclasses.replace(job.logs[0], size=661, mtime=T0 - timedelta(hours=13))
    svc.put_job(dataclasses.replace(job, logs=(out, *job.logs[1:])))
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.push_screen(LogScreen(KEY))
        await _settle(pilot)
        hint = app.screen.query_one("#find-hint")
        assert hint.display and "stdout quiet for 13h; f finds other log files" in str(hint.render())


async def test_hint_bar_shows_the_key_on_a_wide_terminal(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(220, 40)) as pilot:
        detail = await _detail(app, pilot)
        assert "other logs" in detail.query_one(HintBar).text.plain


async def test_enter_selects_after_a_slow_rescan(make_app: Any) -> None:
    """Hiding the list while loading took its focus; the results give it back, so enter picks again."""
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        picker = await _picker(app, pilot)
        svc.find_gate = asyncio.Event()
        await pilot.press("r")
        await _settle(pilot)
        assert _summary(picker).startswith("looking for")
        svc.find_gate.set()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "file"


async def test_closing_the_picker_while_it_looks(make_app: Any) -> None:
    app, svc = make_app()
    svc.find_gate = asyncio.Event()
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        await _picker(app, pilot)
        await pilot.press("escape")
        await _settle(pilot)
        svc.find_gate.set()
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "stdout" and KEY not in app.found_logs


async def test_full_viewer_opens_on_a_picked_file_of_a_job_without_stdout_path(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    svc.put_job(dataclasses.replace(_job(svc), logs=()))
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        await _picker(app, pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert detail.stream == "file"
        await pilot.press("l")
        await _settle(pilot)
        assert isinstance(app.screen, LogScreen) and app.screen.stream == "file"
        assert svc.called("stream_log")[-1] == ("k2", "file", TRAIN)


async def test_a_path_with_brackets_is_text_not_markup(make_app: Any) -> None:
    app, svc = make_app()
    odd = f"{WD}/runs/[bold]x[/bold]/train.log"
    svc.found_files[WD] = [FoundLog(odd, 10, T0)]
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        picker = await _picker(app, pilot)
        assert str(picker.query_one("#fl-list").get_option_at_index(0).prompt).startswith("runs/[bold]x[/bold]/")
        await pilot.press("enter")
        await _settle(pilot)
        assert odd in _title(detail)


# ---------------------------------------------------------------- round 2: width, tab, incomplete, others, ended
OTHER = JobKey("k2", "2332862")
OTHER_OUT = f"{WD}/logs/orb-swp--train--cifar-const--s43--a02_2332862.out"
ENDED = JobKey("k2", "2331214")


def _rows(screen: FoundLogsScreen) -> list[str]:
    menu = screen.query_one("#fl-list", OptionList)
    return [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]


def test_row_text_fits_its_width_with_size_and_age_apart() -> None:
    long = f"{WD}/" + "/".join(["very-long-directory-name"] * 6) + "/train.log"
    for width in (20, 40, 70, 99, 140):
        for mark in ("", "stdout", "job 2332862_17 stdout"):
            row = row_text(FoundLog(long, 21_504, T0 - timedelta(seconds=3)), WD, T0, width, mark=mark,
                           current=True)
            assert row.cell_len <= width
        assert re.search(r"21K  +3s ago", row.plain)                      # never "21K3s ago"
    assert "· job 2332862_17 stdout" in row_text(FoundLog(TRAIN, 9, T0), WD, T0, 99, mark="job 2332862_17 stdout").plain
    trailer = trailer_width("job 2332862 stdout", False)                  # rows pad to the widest marks
    plain = row_text(FoundLog(TRAIN, 9, T0), WD, T0, 99, trailer=trailer).plain
    marked = row_text(FoundLog(STDOUT, 9, T0), WD, T0, 99, mark="job 2332862 stdout", trailer=trailer).plain
    assert len(plain) == len(marked) == 99 and plain.index(" ago") == marked.index(" ago")


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40), (200, 50)])
async def test_picker_rows_fit_the_list_at_every_terminal_size(make_app: Any, size: tuple[int, int]) -> None:
    """Round 1: `#fl-box { width: 104 }` in DEFAULT_CSS lost to `.dialog { width: 80 }`, rows laid out for 98 cells
    wrapped in a 76-cell list and the size ran into the age ("21K3s ago"). Rows now fit the list's real width."""
    app, svc = make_app()
    deep = "/".join(["a-rather-long-run-directory"] * 5)
    svc.found_files[WD] = [FoundLog(f"{WD}/runs/{deep}/r{i:02d}/train.log", 21_504 * (i + 1),
                                    T0 - timedelta(seconds=3 + i)) for i in range(FIND_MAX_FILES + 4)]
    svc.found_files[WD].append(FoundLog(OTHER_OUT, 90, T0 - timedelta(seconds=1)))
    async with app.run_test(size=size) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        menu = picker.query_one("#fl-list", OptionList)
        box = picker.query_one("#fl-box")
        assert box.region.width == min(104, size[0] * 96 // 100)          # the stylesheet's width, not .dialog's 80
        pad = menu.get_component_styles("option-list--option").padding.width
        room = menu.scrollable_content_region.width - pad
        assert room > 0 and picker.row_width == room
        rows = _rows(picker)
        assert len(rows) == FIND_MAX_FILES
        assert all(cell_len(r) <= room for r in rows)
        assert menu.virtual_size.height == menu.option_count                # one line per row: nothing wraps
        assert all(re.search(r"\S  +\d+[smhd] ago", r) for r in rows)
        summary = picker.query_one("#fl-summary")
        assert all(cell_len(line) <= summary.content_region.width for line in _summary(picker).split("\n"))


async def test_a_resize_lays_the_rows_out_again(make_app: Any) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=(200, 50)) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        wide = picker.row_width
        await pilot.resize_terminal(80, 24)
        await _settle(pilot)
        menu = picker.query_one("#fl-list", OptionList)
        assert picker.row_width < wide and all(cell_len(r) <= picker.row_width for r in _rows(picker))
        assert menu.virtual_size.height == menu.option_count


@pytest.mark.parametrize("tab", ["summary", "gpu", "procs"])
async def test_picking_from_another_tab_shows_the_logs_tab(make_app: Any, tab: str) -> None:
    app, svc = make_app()
    _files(svc)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        tabs = detail.query_one(TabbedContent)
        tabs.active = tab
        await _settle(pilot)
        await _picker(app, pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert detail.stream == "file" and tabs.active == "logs"
        await pilot.press("o")
        await _settle(pilot)
        tabs.active = tab
        await _settle(pilot)
        await pilot.press("f")                                    # back to the remembered file: in sight too
        await _settle(pilot)
        assert app.screen is detail and detail.stream == "file" and tabs.active == "logs"


def test_incomplete_text() -> None:
    files = (FoundLog(TRAIN, 9, T0), FoundLog(EVAL, 9, T0))
    assert incomplete_text(LogFiles(WD, files, T0)) is None
    assert incomplete_text(LogFiles(WD, files, T0, incomplete="timed out")) == \
        "search stopped early (timed out): 2 found so far"
    unreadable = incomplete_text(LogFiles(WD, files, T0, incomplete=FIND_UNREADABLE))
    assert unreadable == "some directories were unreadable: their files are not listed"      # the walk finished
    capped = incomplete_text(LogFiles(WD, files, T0, truncated=True, incomplete=FIND_UNREADABLE))
    assert capped == ("some directories were unreadable: their files are not listed\n"       # round 2: both
                      f"only the newest {FIND_MAX_FILES}: more were written")


async def test_a_walk_stopped_early_is_not_a_complete_or_capped_list(make_app: Any) -> None:
    """Round 1: a timed-out walk read "only the newest 30: more were written", or with nothing found "no log file …
    written", as if the search had finished."""
    app, svc = make_app()
    svc.find_incomplete = "timed out"
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        text = _summary(picker)
        assert text.startswith("no log file found under") and "so far" in text
        assert "search stopped early (timed out): 0 found so far" in text and "written under" not in text
        _files(svc)
        await pilot.press("r")
        await _settle(pilot)
        text = _summary(picker)
        assert "3 files written under" in text and "search stopped early (timed out): 3 found so far" in text
        assert "more were written" not in text
        many = [FoundLog(f"{WD}/runs/r{i}/train.log", 10, T0 - timedelta(seconds=i)) for i in range(FIND_MAX_FILES + 4)]
        svc.found_files[WD] = many
        await pilot.press("r")
        await _settle(pilot)
        assert f"search stopped early (timed out): over {FIND_MAX_FILES} found, the newest listed" in _summary(picker)
        svc.find_incomplete = FIND_UNREADABLE
        await pilot.press("r")
        await _settle(pilot)
        text = _summary(picker)                     # round 2: the cap was hidden behind the unreadable note
        assert "some directories were unreadable: their files are not listed" in text
        assert f"only the newest {FIND_MAX_FILES}: more were written" in text
        assert "stopped early" not in text and "so far" not in text
        svc.found_files[WD] = []
        await pilot.press("r")
        await _settle(pilot)
        text = _summary(picker)                     # nothing found, the walk finished: no "so far"
        assert text.startswith("no log file found under") and "so far" not in text
        assert "some directories were unreadable" in text and "stopped early" not in text
        svc.find_incomplete, svc.found_files[WD] = None, many
        await pilot.press("r")
        await _settle(pilot)
        assert f"only the newest {FIND_MAX_FILES}: more were written" in _summary(picker)


def test_other_jobs_logs_on_the_host(service: Any) -> None:
    job = _job(service)
    others = other_paths(service.store.jobs([job.key.host]), job)
    assert others[OTHER_OUT] == "job 2332862 stdout" and STDOUT not in others


async def test_other_jobs_logs_are_marked_and_never_suggested(make_app: Any) -> None:
    """Round 1: in a shared workdir the newest file is often another job's stdout; the cursor started on it."""
    app, svc = make_app()
    _files(svc)
    svc.found_files[WD].append(FoundLog(OTHER_OUT, 4_000, T0 - timedelta(seconds=2)))     # the newest of all
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot)
        picker = await _picker(app, pilot)
        rows = _rows(picker)
        assert "a02_2332862.out" in rows[0] and "· job 2332862 stdout" in rows[0]
        assert picker.query_one("#fl-list", OptionList).highlighted == 1 and rows[1].startswith("runs/cifar")
        await pilot.press("escape")
        await _settle(pilot)
        svc.found_files[WD] = [FoundLog(OTHER_OUT, 4_000, T0 - timedelta(seconds=2)),
                               FoundLog(STDOUT, 661, T0 - timedelta(hours=4))]
        picker = await _picker(app, pilot)
        rows = _rows(picker)                    # round 2: nothing to suggest, and the cursor sat on the other job's
        assert "◀ shown" in rows[1] and picker.query_one("#fl-list", OptionList).highlighted == 1    # but the shown


def test_the_cursor_never_starts_on_another_job_s_log_while_a_safer_row_is_left(service: Any) -> None:
    """The ◀ shown row, else core.logs.suggest, else the job's own log, and another job's log only when it is all
    that is listed (round 2: with nothing to suggest the cursor started on the first row, another job's stdout)."""
    job = _job(service)
    other, own_out, train = (FoundLog(OTHER_OUT, 4_000, T0), FoundLog(STDOUT, 661, T0 - timedelta(hours=4)),
                             FoundLog(TRAIN, 9, T0 - timedelta(hours=1)))

    def start(files: list[FoundLog], current: str | None) -> FoundLog:
        screen = FoundLogsScreen(job, current=current, others={OTHER_OUT: "job 2332862 stdout"})
        screen.files = {f"f{i}": f for i, f in enumerate(files)}
        oid = screen.start_option()
        assert oid is not None
        return screen.files[oid]

    assert start([other, own_out, train], STDOUT) is train                  # the suggestion over the shown stdout
    assert start([other, own_out, train], TRAIN) is train                   # the picked file shown
    assert start([other, own_out], STDOUT) is own_out                       # the shown row
    assert start([other, own_out], None) is own_out                         # the job's own: back to its stream
    assert start([other], None) is other and start([other], STDOUT) is other    # nothing else listed
    old_eval = FoundLog(EVAL, 9, T0 - timedelta(hours=3))                   # older than train.log, which suggest picks
    assert start([other, train, old_eval], EVAL) is old_eval                # the picked file shown wins
    stderr = f"{WD}/logs/a02_2332861.err"
    own_err = FoundLog(stderr, 9, T0 - timedelta(hours=5))
    with_err = dataclasses.replace(job, logs=(*job.logs, LogRef("stderr", stderr)))
    screen = FoundLogsScreen(with_err, current=stderr, others={OTHER_OUT: "job 2332862 stdout"})
    screen.files = {f"f{i}": f for i, f in enumerate([other, own_out, own_err])}
    oid = screen.start_option()
    assert oid is not None and screen.files[oid] is own_err                # the shown stderr, not the first own row


def test_find_until_is_the_end_of_an_ended_job_only(service: Any) -> None:
    ended = _job(service, ENDED)
    assert ended.end_time is not None and find_window(ended)[1] == ended.end_time
    running = _job(service)
    assert find_window(dataclasses.replace(running, end_time=T0 + timedelta(hours=3)))[1] is None    # a time limit
    assert window_text(T0 - timedelta(hours=4), T0).startswith("while the job ran (")
    assert window_text(T0, None).startswith("since the job started (") and window_text(None, None) == \
        "in the last 24 h"


async def test_an_ended_job_lists_what_was_written_while_it_ran(make_app: Any) -> None:
    app, svc = make_app()
    ended = _job(svc, ENDED)
    assert ended.start_time is not None and ended.end_time is not None
    svc.found_files[WD] = [
        FoundLog(f"{WD}/runs/imagenette/s40/train.log", 9_000, ended.end_time - timedelta(minutes=1)),
        FoundLog(f"{WD}/runs/next-job/train.log", 9_000, ended.end_time + timedelta(hours=1)),    # ran later
    ]
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot, ENDED)
        picker = await _picker(app, pilot)
        assert svc.called("find_logs") == [("k2", WD, ended.start_time, ended.end_time)]
        assert "while the job ran (" in _summary(picker)
        assert [r.split()[0] for r in _rows(picker)] == ["runs/imagenette/s40/train.log"]


def test_quiet_hint_is_the_core_rule(service: Any) -> None:
    """Round 1: the agent called a 202-B stdout written 0 s ago quiet; both now ask core.logs.stdout_quiet_for."""
    job = _job(service)
    stall = service.config.general.stall_after
    for size, ago in ((202, 0), (202, 599), (202, 600), (4096, 700), (4097, 700), (1 << 20, stall + 1)):
        ref = LogRef("stdout", STDOUT, size=size, mtime=T0 - timedelta(seconds=ago))
        assert (quiet_hint(job, ref, T0, stall) is None) == (stdout_quiet_for(ref, T0, stall) is None)
    assert quiet_hint(job, LogRef("stdout", STDOUT, size=202, mtime=T0), T0, stall) is None


def test_own_paths_take_the_detail_s_resolved_refs(service: Any) -> None:
    job = dataclasses.replace(_job(service), logs=(LogRef("stdout", None, pattern="logs/%x_%j.out"),))
    assert own_paths(job) == {}
    resolved = (LogRef("stdout", STDOUT), LogRef("stderr", "//" + STDOUT.lstrip("/")))     # normalised alike
    own = own_paths(job, resolved)
    assert own == {STDOUT: "stdout+stderr"}
    found: dict[Any, LogRef] = {}
    assert remember_pick(found, job, FoundLog(STDOUT, 661, T0), own) == "stdout" and not found


async def test_a_stdout_only_the_detail_resolved_is_still_the_job_s_own(make_app: Any) -> None:
    """The store's ref holds only a pattern; the detail probe resolved the path: the picker marks that row as the
    job's stdout, does not start on it, and picking it goes back to stdout instead of following it as a file."""
    app, svc = make_app()
    _files(svc)
    job = _job(svc)
    svc.put_job(dataclasses.replace(job, logs=(LogRef("stdout", None, pattern="logs/%x_%j.out"),)))
    svc.detail_logs[KEY] = (LogRef("stdout", STDOUT, size=661, mtime=T0 - timedelta(hours=4)),)
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        assert detail.detail is not None
        picker = await _picker(app, pilot)
        rows = _rows(picker)
        assert "a02_2332861.out" in rows[2] and "· stdout" in rows[2]
        assert picker.query_one("#fl-list", OptionList).highlighted == 0
        await pilot.press("j", "j", "enter")
        await _settle(pilot)
        assert detail.stream == "stdout" and KEY not in app.found_logs
        await pilot.press("l")                  # round 2: the viewer's picker forgot the refs the detail resolved
        await _settle(pilot)
        viewer = app.screen
        assert isinstance(viewer, LogScreen) and viewer.detail_logs == svc.detail_logs[KEY]
        picker = await _picker(app, pilot)
        rows = _rows(picker)
        assert "a02_2332861.out" in rows[2] and "· stdout" in rows[2]
        assert picker.query_one("#fl-list", OptionList).highlighted == 0
        await pilot.press("j", "j", "enter")
        await _settle(pilot)
        assert app.screen is viewer and viewer.stream == "stdout" and KEY not in app.found_logs
        assert viewer.follower is not None and viewer.follower.ref is not None
        assert viewer.follower.ref.path == STDOUT                          # followed as the detail resolved it


async def test_l_p_and_the_hint_use_the_stdout_the_detail_resolved(make_app: Any, monkeypatch: Any) -> None:
    """No polled stdout at all (job.logs empty), the detail probe resolved it: `l` opens the viewer, `p` pages the
    resolved path in both views, and the viewer's quiet hint judges the detail's fresher stat as the detail does."""
    app, svc = make_app()
    _files(svc)
    svc.put_job(dataclasses.replace(_job(svc), logs=()))
    svc.detail_logs[KEY] = (LogRef("stdout", STDOUT, size=661, mtime=T0 - timedelta(hours=13)),)
    paged: list[str | None] = []
    monkeypatch.setattr(app, "open_pager", lambda host, ref: paged.append(ref.path))
    async with app.run_test(size=BIG) as pilot:
        detail = await _detail(app, pilot)
        assert detail.detail is not None
        app.action_pager()
        await pilot.press("l")
        await _settle(pilot)
        viewer = app.screen
        assert isinstance(viewer, LogScreen) and viewer.shown_ref() is not None
        app.action_pager()
        assert paged == [STDOUT, STDOUT]
        hint = viewer.query_one("#find-hint")
        assert hint.display and "stdout quiet for 13h" in str(hint.render())


async def test_an_ended_job_without_a_start_looks_in_the_day_before_its_end(make_app: Any) -> None:
    """since None would be the core's "last 24 h" from now: empty for a job that ended days ago."""
    app, svc = make_app()
    ended = _job(svc, ENDED)
    assert ended.end_time is not None
    svc.put_job(dataclasses.replace(ended, start_time=None, end_time=T0 - timedelta(days=3)))
    svc.found_files[WD] = [FoundLog(f"{WD}/runs/old/train.log", 9, T0 - timedelta(days=3, hours=2))]
    async with app.run_test(size=BIG) as pilot:
        await _detail(app, pilot, ENDED)
        picker = await _picker(app, pilot)
        end = T0 - timedelta(days=3)
        assert svc.called("find_logs") == [("k2", WD, end - timedelta(hours=24), end)]
        assert "in the 24 h up to the job's end (" in _summary(picker) and len(_rows(picker)) == 1


async def test_a_wrapped_summary_keeps_the_list_inside_the_box(make_app: Any) -> None:
    """80x24, an ended job (a long window text) and a walk stopped early: the summary takes 3-4 lines and the list
    gives up those rows, so the highlighted last row is never below the box's bottom border."""
    app, svc = make_app()
    ended = _job(svc, ENDED)
    assert ended.end_time is not None
    svc.find_incomplete = "timed out"
    svc.found_files[WD] = [FoundLog(f"{WD}/runs/r{i}/train.log", 10, ended.end_time - timedelta(seconds=i))
                           for i in range(FIND_MAX_FILES)]
    async with app.run_test(size=(80, 24)) as pilot:
        await _detail(app, pilot, ENDED)
        picker = await _picker(app, pilot)
        menu = picker.query_one("#fl-list", OptionList)
        box = picker.query_one("#fl-box")
        assert picker.summary_rows >= 3 and menu.highlighted == 0
        for _ in range(FIND_MAX_FILES - 1):
            await pilot.press("j")
        await _settle(pilot)
        assert menu.highlighted == FIND_MAX_FILES - 1 and menu.region.bottom <= box.region.bottom - 1
