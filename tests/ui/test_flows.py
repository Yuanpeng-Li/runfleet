"""[WP3] Pilot flows of DESIGN.md §10.5 on FakeService (ManualClock, animations off)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
from textual.widgets import Checkbox, DataTable, OptionList, TabbedContent

from runfleet.core.service import LogUpdate
from runfleet.models import ConnState, JobKey, JobState
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.action_menu import ActionMenuScreen
from runfleet.ui.screens.confirm import ConfirmScreen
from runfleet.ui.screens.help import HelpScreen
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.host_error import HostErrorScreen
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen, TooSmallScreen
from runfleet.ui.widgets.filter_bar import FilterBar
from runfleet.ui.widgets.fleet_view import FleetTable, FleetView
from runfleet.ui.widgets.gpus_view import GPUsView
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.host_card import HostCard
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.log_view import LogView
from runfleet.ui.widgets.proc_table import ProcTable
from tests.ui.fakes import FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
SMALL = (80, 24)
FLUSH = 0.3          # > the 250 ms store->UI coalescing interval


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


def header_text(app: RunFleetApp) -> str:
    return app.screen.query_one(FleetHeader).text.plain


def jobs_view(app: RunFleetApp) -> JobsView:
    return app.screen.query_one(JobsView)


def table(app: RunFleetApp) -> JobTable:
    return app.screen.query_one(JobTable)


def cell(app: RunFleetApp, key: str, column: str) -> str:
    t = table(app)
    return str(t.get_cell(key, column).plain)


async def goto_jobs(pilot: Any) -> None:
    await pilot.press("tab")
    await pilot.pause()


async def all_preset(pilot: Any, app: RunFleetApp) -> None:
    while jobs_view(app).preset != "all":
        await pilot.press("f")
    await pilot.pause()


async def move_to(pilot: Any, app: RunFleetApp, key: str) -> None:
    table(app).move_to_key(key)
    await pilot.pause()


# ---------------------------------------------------------------- 1. views and scope
async def test_launch_views_and_scope(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, MainScreen) and app.screen.view == "fleet"
        assert len(app.screen.query(HostCard)) == 5
        assert svc.started
        await goto_jobs(pilot)
        assert app.screen.view == "jobs"
        await all_preset(pilot, app)
        assert set(table(app).order) == {str(j.key) for j in svc.store.jobs() if j.is_mine}
        await pilot.press("tab")
        assert app.screen.view == "gpus"
        gpus = app.screen.query_one(GPUsView)
        assert len([r for r in gpus.rows if not r.aggregate]) == 6
        await pilot.press("2")
        await pilot.pause()
        assert app.scope == "vega"
        assert {r.host for r in gpus.rows} == {"vega"}
        assert "vega" in header_text(app).split("RunFleet")[1][:10]
        assert svc.visibility["vega"].value == "visible" and svc.visibility["k2"].value == "hidden"
        await pilot.press("0")
        await pilot.pause()
        assert app.scope is None and "ALL" in header_text(app)
        await pilot.press("shift+tab", "shift+tab")
        assert app.screen.view == "fleet"


# ---------------------------------------------------------------- 2. filter and presets
async def test_filter_negation_and_presets(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await pilot.press("slash", "l", "m", "f", "t")
        await pilot.pause(0.2)
        view = jobs_view(app)
        assert view.query_text == "lmft"
        assert table(app).order and all("lmft" in table(app).rows_by_key[k].name for k in table(app).order)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert view.query_text == "" and not app.screen.query_one(FilterBar).display
        await pilot.press("slash", *"!state:R", "enter")
        await pilot.pause(0.2)
        rows = [table(app).rows_by_key[k] for k in table(app).order]
        assert rows and all(r.state is not JobState.RUNNING for r in rows)        # type: ignore[union-attr]
        assert app.screen.query_one(FilterBar).display                             # stays while active
        assert table(app).has_focus
        await pilot.press("escape")                                               # esc on table clears filter
        await pilot.pause(0.2)
        assert view.query_text == ""
        presets = [view.preset]
        for _ in range(4):
            await pilot.press("f")
            presets.append(view.preset)
        assert presets == ["fresh", "active", "queued", "failed", "all"]
        await pilot.press("f", "f", "f", "f")                                     # back to "failed"
        assert view.preset == "failed"
        ended_task = dataclasses.replace(next(j for j in svc.store.jobs(["labstation"]) if j.is_mine),
                                         state=JobState.EXITED, raw_state="exited", end_time=svc.clock.now())
        svc.put_job(ended_task)
        await pilot.pause(FLUSH)
        assert str(ended_task.key) in table(app).order                            # failed preset includes ENDED


# ---------------------------------------------------------------- 3. sorting, cursor stability
async def test_sort_and_cursor_stays_on_job(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        view = jobs_view(app)
        await pilot.press("o")
        assert view.sort == "state"
        t = table(app)
        assert str(t.columns["state"].label) == "ST▲"                           # 3 cells: the label stays whole
        await pilot.press("o")
        assert view.sort == "elapsed"
        assert str(t.columns["elapsed"].label).startswith("ELAPSED")
        col = [k for k, _w in t.cols].index("gpu_util")
        x = sum(w for _k, w in t.cols[:col]) + 1
        await pilot.click(JobTable, offset=(x, 0))                               # header row
        await pilot.pause()
        assert view.sort == "gpu_util" and not view.reverse
        utils = [r.gpu.util_mean_pct for r in (t.rows_by_key[k] for k in t.order)  # type: ignore[union-attr]
                 if r.gpu is not None and r.state.group.value == "active"]       # type: ignore[union-attr]
        assert utils == sorted(utils, reverse=True)
        await pilot.press("O")
        assert view.reverse
        target = "vega:724245"
        await move_to(pilot, app, target)
        before = t.cursor_row
        # a refresh that changes another job and adds one: the cursor follows the job, not the index
        changed = dataclasses.replace(svc.store.job(JobKey("k2", "2332861")), gpu=None)
        svc.put_job(changed)
        new = dataclasses.replace(svc.store.job(JobKey("vega", "724246")), key=JobKey("vega", "724690"),
                                  job_id_raw="724690")
        svc.put_job(new)
        await pilot.pause(FLUSH)
        assert t.cursor_key == target
        assert t.order.index(target) == t.cursor_row
        del before


async def test_cell_updates_in_place_without_rebuild(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        t = table(app)
        key = "k2:2332861"
        await move_to(pilot, app, key)
        calls: list[str] = []
        original_clear = t.clear
        t.clear = lambda columns=False: calls.append("clear") or original_clear(columns)  # type: ignore[method-assign]
        job = svc.store.job(JobKey("k2", "2332861"))
        usage = dataclasses.replace(job.gpu, util_mean_pct=42.0)
        svc.put_job(dataclasses.replace(job, gpu=usage))
        await pilot.pause(FLUSH)
        assert "42%" in cell(app, key, "gpu_util")
        assert calls == []                                                       # keyed diff, no rebuild
        assert t.cursor_key == key


# ---------------------------------------------------------------- 4. job detail
async def test_detail_tabs_and_return(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332862")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == JobKey("k2", "2332862")
        assert svc.called("job_detail")
        assert svc.visibility["k2"].value == "focused"
        tabs = app.screen.query_one(TabbedContent)
        seen = [tabs.active]
        for _ in range(6):
            await pilot.press("right_square_bracket")
            seen.append(tabs.active)
        assert seen == ["summary", "gpu", "logs", "steps", "procs", "raw", "summary"]
        await pilot.press("left_square_bracket")
        assert tabs.active == "raw"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        assert table(app).cursor_key == "k2:2332862"


async def test_header_logo_goes_home(make_app: MakeApp) -> None:
    """Clicking `RunFleet` in the header closes detail screens and shows the Fleet view."""
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332862")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, JobDetailScreen)
        app.action_open_host("k2")                             # a second level: host detail over job detail
        await pilot.pause()
        assert isinstance(app.screen, HostDetailScreen)
        await pilot.click(FleetHeader, offset=(3, 0))          # " RunFleet …": the logo starts at column 1
        await pilot.pause()
        assert isinstance(app.screen, MainScreen) and app.screen.view == "fleet"
        await pilot.press("tab")
        await pilot.pause()
        assert app.screen.view == "jobs"
        await pilot.click(FleetHeader, offset=(3, 0))          # on the main screen too: back to Fleet
        await pilot.pause()
        assert app.screen.view == "fleet"


async def test_failed_job_steps_and_explanation(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.open_job(JobKey("k2", "2330539"))
        await pilot.pause(0.2)
        steps = app.screen.query_one("#steps-table", DataTable)
        assert steps.row_count == 3
        why = str(app.screen.query_one("#why").render())
        assert "exited with code 1" in why
        assert "FAILED" in app.screen.query_one("#title").render().plain   # type: ignore[union-attr]


# ---------------------------------------------------------------- 5. actions
async def test_cancel_flow_updates_row_and_toasts(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await goto_jobs(pilot)
        key = "k2:2332861"
        await move_to(pilot, app, key)
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        body = app.screen.query_one("#confirm-body").render()
        assert "scancel 2332861" in str(body)
        assert app.screen.focused is not None and app.screen.focused.id == "cancel"     # default focus
        await pilot.press("y")
        await pilot.pause(FLUSH)
        assert isinstance(app.screen, MainScreen)
        assert svc.store.job(JobKey("k2", "2332861")).state is JobState.CANCELLED
        assert cell(app, key, "state") == "CA"
        assert any("Cancelled 1 job on k2" in m for m in toasts(app))


async def test_marks_make_one_dialog(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332863")
        await pilot.press("space", "space", "space")
        view = jobs_view(app)
        assert len(view.marks) == 3
        assert cell(app, "k2:2332863", "marker").endswith("+")
        await pilot.press("ctrl+k")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen)
        assert sum(len(p.targets) for p in screen.plans) == 3
        assert "scancel 2332861 2332862 2332863" in screen.plans[0].preview
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        assert all(j.state is JobState.RUNNING for j in svc.store.jobs(["k2"]) if j.key.id.startswith("40016"))
        await pilot.press("escape")                                               # esc clears marks
        assert not view.marks


async def test_terminate_task_replans_on_escalate_toggle(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("4")
        await goto_jobs(pilot)
        await pilot.press("ctrl+k")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen) and screen.action_id == "task.terminate"
        assert "KILL after 10 s" in screen.plans[0].preview
        n_plans = len(svc.called("plan"))
        screen.query_one(Checkbox).toggle()
        await pilot.pause()
        assert len(svc.called("plan")) == n_plans + 1
        assert "no escalation" in screen.plans[0].preview
        assert "no escalation" in str(screen.query_one("#confirm-body").render())
        await pilot.press("escape")


# ---------------------------------------------------------------- 6. read-only and host reasons
async def test_readonly_blocks_actions(make_app: MakeApp) -> None:
    app, svc = make_app(readonly=True)
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await goto_jobs(pilot)
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        assert any("read-only" in m for m in toasts(app))
        assert "R/O" in header_text(app)
        assert "^k" not in app.screen.query_one(HintBar).text.plain
        await pilot.press("x")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        options = app.screen.query_one(OptionList)
        assert options.option_count > 0
        assert all(options.get_option_at_index(i).disabled for i in range(options.option_count))
        assert "read-only mode" in str(options.get_option_at_index(0).prompt)


async def test_needs_auth_host_actions_disabled(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.press("5")
        await goto_jobs(pilot)
        await pilot.press("x")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        prompt = str(app.screen.query_one(OptionList).get_option_at_index(0).prompt)
        assert "lynx: needs auth" in prompt
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        assert any("needs auth" in m for m in toasts(app))


# ---------------------------------------------------------------- 7. fault injection
async def test_unreachable_host_goes_red_and_recovers(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await goto_jobs(pilot)
        svc.inject("vega", "unreachable")
        await pilot.pause(FLUSH)
        assert "✖ vega down" in header_text(app)
        rows = [k for k in table(app).order if k.startswith("vega:")]
        assert rows and all(cell(app, k, "marker")[1] in "!~" for k in rows)
        assert any("vega: unreachable" in m for m in toasts(app))
        svc.inject("vega", "ok")
        await pilot.pause(FLUSH)
        assert "● vega" in header_text(app)
        assert all(cell(app, k, "marker")[1] == " " for k in rows)
        assert any("vega is back" in m for m in toasts(app))


# ---------------------------------------------------------------- 8. connect
async def test_connect_needs_auth_host(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("5")                                   # scope lynx: fleet selection follows on c
        await pilot.pause()
        assert app.screen.query_one(FleetView).selected == "lynx"
        await pilot.press("c")
        await pilot.pause(FLUSH)
        assert runs == [svc.connect_argv("lynx")]
        assert svc.called("after_connect") == [("lynx", 0)]
        assert svc.store.hosts["lynx"].status.state is ConnState.OK


async def test_host_error_screen_from_red_chip(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_chip("lynx")
        await pilot.pause()
        assert isinstance(app.screen, HostErrorScreen)
        assert "Permission denied" in str(app.screen.query_one("#err-body").render())
        await pilot.press("r")
        await pilot.pause()
        assert svc.called("refresh")[-1] == (("lynx",),)
        app.action_chip("vega")                                  # healthy chip scopes instead
        await pilot.pause()
        assert app.scope == "vega"


async def test_shell_into_multi_node_job_asks_for_the_node(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    """`s` on a running multi-node job picks the node (batch host first); a single-node job goes straight in."""
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None
    svc.put_job(dataclasses.replace(job, nodes=("n-40-01", "n-40-02"), nodelist="n-40-[01-02]", batch_host="n-40-01"))
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332861")
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        menu = app.screen.query_one(OptionList)
        assert [menu.get_option_at_index(i).id for i in range(menu.option_count)] == ["n-40-01", "n-40-02"]
        await pilot.press("down", "enter")
        await pilot.pause(FLUSH)
        assert runs == [svc.shell_argv("k2", JobKey("k2", "2332861"), "n-40-02")]
        await move_to(pilot, app, "k2:2332862")
        await pilot.press("s")
        await pilot.pause(FLUSH)
        assert runs[-1] == svc.shell_argv("k2", JobKey("k2", "2332862"))


# ---------------------------------------------------------------- 9. log viewer
async def test_log_viewer_stream_live_follow_and_pager(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332861")
        await pilot.press("l")
        await pilot.pause(0.2)
        screen = app.screen
        assert isinstance(screen, LogScreen)
        view = screen.query_one(LogView)
        assert view.count == 7
        svc.push_log(LogUpdate(lines=("epoch 45/300 done",), live="epoch 46/300:  37%|███▋ | 211/570"))
        await pilot.pause(0.2)
        assert view.count == 8
        assert "37%" in str(screen.query_one("#live").render())
        svc.push_log(LogUpdate(live="epoch 46/300:  52%|█████▏| 296/570"))
        await pilot.pause(0.2)
        assert "52%" in str(screen.query_one("#live").render()) and view.count == 8
        assert view.follow
        await pilot.press("F")
        assert not view.follow
        await pilot.press("G")
        assert view.follow
        await pilot.press("e")
        await pilot.pause(0.2)
        assert screen.stream == "stderr"
        assert svc.called("stream_log")[-1][1] == "stderr"
        assert svc.sessions[0].closed                              # switching streams closes the old one
        await pilot.press("p")
        await pilot.pause(0.2)
        assert runs and runs[-1][:4] == ["ssh", "-t", "k2", "less"]


async def test_detail_log_pane_follows(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.open_job(JobKey("vega", "723952"))
        await pilot.pause(0.2)
        views = list(app.screen.query(LogView))
        assert len(views) == 2 and all(v.count == 7 for v in views)          # one stream, two panes
        assert len(svc.sessions) == 1


# ---------------------------------------------------------------- 10. responsive
async def test_resize_drops_columns_and_collapses_preview(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        cols = [k for k, _w in table(app).cols]
        assert {"left", "node_reason", "gpus", "gpu_util"} <= set(cols)
        assert jobs_view(app).preview.mode == "full"
        await pilot.resize_terminal(*SMALL)
        await pilot.pause(0.2)
        cols = [k for k, _w in table(app).cols]
        assert "left" not in cols and "gpus" not in cols        # critic: why it waits / ended beats the GPU type
        assert {"host", "id", "name", "state", "elapsed", "gpu_util", "node_reason"} <= set(cols)
        assert jobs_view(app).preview.mode == "line"
        await pilot.resize_terminal(80, 18)
        await pilot.pause(0.2)
        assert jobs_view(app).preview.mode == "hidden"
        await pilot.resize_terminal(50, 12)
        await pilot.pause(0.2)
        assert isinstance(app.screen, TooSmallScreen)
        await pilot.resize_terminal(*BIG)
        await pilot.pause(0.2)
        assert isinstance(app.screen, MainScreen)


async def test_fleet_switches_to_table_when_narrow(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=SMALL) as pilot:
        await pilot.pause()
        fleet = app.screen.query_one(FleetView)
        assert fleet.mode == "table"
        t = app.screen.query_one(FleetTable)
        assert t.row_count == 5 and t.display
        await pilot.press("down", "enter")
        await pilot.pause()
        assert isinstance(app.screen, HostDetailScreen) and app.screen.host == "vega"


# ---------------------------------------------------------------- 11. mouse
async def test_mouse_click_double_click_right_click(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        t = table(app)
        await pilot.click(JobTable, offset=(20, 3))              # row 2 (header is row 0)
        await pilot.pause()
        assert t.cursor_row == 2 and isinstance(app.screen, MainScreen)
        await pilot.click(JobTable, offset=(20, 3))              # single click on the selected row: no drill-in
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        key = t.cursor_key
        await pilot.double_click(JobTable, offset=(20, 3))
        await pilot.pause()
        assert isinstance(app.screen, JobDetailScreen) and str(app.screen.key) == key
        await pilot.press("escape")
        await pilot.pause()
        await pilot.click(JobTable, offset=(20, 5), button=3)
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)


async def test_mouse_on_cards_and_tabs(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.click("#card-2", offset=(5, 2))
        await pilot.pause()
        assert app.scope is None and app.focused is app.screen.query_one("#card-2")   # click selects, space scopes
        await pilot.press("space")
        await pilot.pause()
        assert app.scope == "vega"
        await pilot.double_click("#card-4", offset=(5, 2))
        await pilot.pause()
        assert isinstance(app.screen, HostDetailScreen) and app.screen.host == "labstation"
        await pilot.press("escape")
        await pilot.pause()
        app.screen.action_show_view("gpus")                     # what a click on the GPUs tab runs
        await pilot.pause()
        assert app.screen.view == "gpus"


# ---------------------------------------------------------------- 12. host detail
async def test_host_detail_processes_and_other_users(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_open_host("labstation")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostDetailScreen) and not screen.cluster
        procs = screen.query_one(ProcTable)
        assert set(procs.order) == {363948, 363942}
        await pilot.press("u")
        await pilot.pause()
        assert 771002 in procs.order
        assert procs.order[0] == 363948                         # sorted by GPU memory
        await pilot.press("o")
        assert screen.sort == "cpu"
        procs.move_cursor(row=procs.order.index(363942))
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key.host == "labstation"


async def test_host_detail_gpu_click_shows_its_processes(make_app: MakeApp) -> None:
    """Workstation: a GPU row is clickable and narrows the process table to that GPU, other users included."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_open_host("labstation")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostDetailScreen)
        procs = screen.query_one(ProcTable)
        assert 771002 not in procs.order                        # another user's process: hidden by default
        await pilot.click("#gpu-panel", offset=(10, 3))         # border, header, GPU 0, GPU 1
        await pilot.pause()
        assert screen.gpu == 1 and procs.order == [771002]      # alice's eval.py on GPU 1, shown without `u`
        assert screen.query_one("#procs-box").border_title == "Processes on GPU 1 (all users)"
        assert "▸ 1" in str(screen.query_one("#gpu-panel").render())
        assert app.focused is procs
        await pilot.click("#gpu-panel", offset=(10, 3))         # the same GPU again: all processes
        await pilot.pause()
        assert screen.gpu is None and set(procs.order) == {363948, 363942}
        await pilot.press("right_square_bracket")               # [ ] step all -> GPU 0 -> … -> all
        await pilot.pause()
        assert screen.gpu == 0 and procs.order == []
        await pilot.press("right_square_bracket", "right_square_bracket", "right_square_bracket")
        await pilot.pause()
        assert screen.gpu == 3 and procs.order == [363948]
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert screen.gpu is None
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert screen.gpu == 3
        assert "[/]" in [h[0] for h in __import__("runfleet.ui.keymap", fromlist=["x"]).hints(screen.help_contexts())]


async def test_host_detail_cluster_partitions(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_open_host("k2")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostDetailScreen) and screen.cluster
        assert screen.partition == "gpu-preempt"
        nodes = screen.query_one("#nodes")
        assert nodes.row_count == 6
        await pilot.press("f")
        await pilot.pause()
        assert nodes.row_count == 3                             # n-31-04, n-31-05, n-40-01 have free GPUs
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert screen.partition == "labpi"
        assert "f" in [h[0] for h in __import__("runfleet.ui.keymap", fromlist=["x"]).hints(screen.help_contexts())]


async def test_partition_switch_settles_without_a_highlight_loop(make_app: MakeApp) -> None:
    """Review (critical): `]` started an endless RowHighlighted ping-pong between two partitions (6000 events in
    3 s, the TUI froze). Selection is now one-way: refreshes stop and the partition stays put."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_open_host("k2")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostDetailScreen)
        calls: list[object] = []
        original = screen.refresh_data

        def counting(changes: object) -> None:
            calls.append(changes)
            original(changes)                                    # type: ignore[arg-type]

        screen.refresh_data = counting                           # type: ignore[method-assign]
        await pilot.press("right_square_bracket")
        for _ in range(10):
            await pilot.pause()
        assert screen.partition == "labpi" and len(calls) <= 2, len(calls)
        table = screen.query_one("#parts")
        assert table.cursor_row == table.names.index("labpi")   # type: ignore[attr-defined]
        await pilot.press("left_square_bracket")
        for _ in range(10):
            await pilot.pause()
        assert screen.partition == "gpu-preempt" and len(calls) <= 4, len(calls)
        screen.refresh_data(None)                                # a data refresh keeps the selection
        await pilot.pause()
        assert screen.partition == "gpu-preempt"


# ---------------------------------------------------------------- 13. palette
async def test_palette_connect_and_colon(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"conn lynx")
        await pilot.pause(0.5)
        await pilot.press("enter")
        await pilot.pause(FLUSH)
        assert runs == [svc.connect_argv("lynx")]
        await pilot.press("colon")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "CommandPalette"
        await pilot.press("escape")


# ---------------------------------------------------------------- 14. idle pause
async def test_idle_pause_and_resume(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        svc.clock.advance(svc.config.general.idle_pause_after + 1)
        svc.check_idle()
        await pilot.pause(FLUSH)
        assert "paused (idle)" in header_text(app)
        await pilot.press("j")
        await pilot.pause(FLUSH)
        assert not svc.store.idle_paused
        assert "paused (idle)" not in header_text(app)


async def test_idle_throttle_badge(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        svc.clock.advance(svc.config.general.idle_after + 1)
        app.refresh_headers()
        assert "idle ×4" in header_text(app)


# ---------------------------------------------------------------- misc keys
async def test_help_is_contextual_and_filterable(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        body = str(app.screen.query_one("#help-body").render())
        assert "mark / unmark" in body and "Help · Jobs view" in str(app.screen.query_one("#help-box").border_title)
        await pilot.press(*"sort")
        await pilot.pause()
        body = str(app.screen.query_one("#help-body").render())
        assert "next sort column" in body and "mark / unmark" not in body
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_refresh_rate_copy_and_quit(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("r")
        assert svc.called("refresh")[-1] == (("k2", "vega", "lyra", "labstation", "lynx"),)
        await pilot.press("ctrl+r")
        assert svc.called("refresh")[-1] == (None,)
        await pilot.press("minus", "minus", "minus")
        assert svc.multiplier == 4.0
        await pilot.press("plus")
        assert svc.multiplier == 2.0
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332861")
        await pilot.press("y", "Y")                 # Y: the copy menu (§7.15); its "stdout path" entry
        await pilot.pause()
        menu = app.screen.query_one(OptionList)
        menu.highlighted = next(i for i in range(menu.option_count) if menu.get_option_at_index(i).id == "stdout")
        await pilot.press("enter")
        await pilot.pause()
        assert copied == ["2332861", "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"]
        await pilot.press("q")
        await pilot.pause()
    assert app.return_code is not None or not app.is_running


async def test_toggle_host_and_shell(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause(FLUSH)
        assert svc.store.hosts["k2"].status.state is ConnState.DISABLED
        assert "○ k2 off" in header_text(app)
        await pilot.press("t")
        await pilot.pause(FLUSH)
        assert svc.store.hosts["k2"].status.state is ConnState.OK
        await pilot.press("s")
        assert runs[-1] == ["ssh", "-t", "k2"]


async def test_group_by_host_folds(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await pilot.press("z")
        t = table(app)
        assert t.order[0] == "§host:k2"
        n = len(t.order)
        t.move_cursor(row=0)
        await pilot.press("enter")
        await pilot.pause()
        assert t.order[0] == "§host:k2" and len(t.order) < n
        await pilot.press("ctrl+k")                             # a group header is never an action target
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        await pilot.press("z")
        assert jobs_view(app).group == "array"


async def test_worker_exception_never_kills_app(make_app: MakeApp) -> None:
    app, svc = make_app()

    async def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("detail exploded")

    svc.job_detail = boom   # type: ignore[method-assign]
    async with app.run_test(size=BIG) as pilot:
        app.open_job(JobKey("k2", "2332861"))
        await pilot.pause(0.2)
        assert isinstance(app.screen, JobDetailScreen)
        assert "detail exploded" in str(app.screen.query_one("#banner").render())


# ---------------------------------------------------------------- empty / first-probe states
async def test_no_hosts_shows_guidance(clock: Any) -> None:
    from runfleet.config import FleetConfig

    svc = FakeService(clock, world=(FleetConfig(hosts=()), []))
    app = RunFleetApp(svc, clock=clock, animations=False)   # type: ignore[arg-type]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#fleet-empty").display
        assert "(Settings)" in str(app.screen.query_one("#fleet-empty").render())
        await goto_jobs(pilot)
        assert table(app).row_count == 0
        assert "no jobs in scope" in str(jobs_view(app).preview.render())
        await pilot.press("tab")
        await pilot.press("r", "x", "ctrl+k", "enter", "l", "p", "y")       # nothing to act on: no crash
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_host_without_snapshot_is_waiting(clock: Any) -> None:
    from tests.ui.fakes import FakeHostState, build_world

    config, states = build_world()
    first = states[0]
    states[0] = FakeHostState(first.config, dataclasses.replace(first.status, state=ConnState.CONNECTING,
                                                                last_ok=None), None)
    svc = FakeService(clock, world=(config, states))
    app = RunFleetApp(svc, clock=clock, animations=False)   # type: ignore[arg-type]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        card = app.screen.query_one("#card-1", HostCard)
        assert "waiting for the first probe" in str(card.render())
        assert "◌ k2 …" in header_text(app)
        app.action_open_host("k2")
        await pilot.pause()
        assert isinstance(app.screen, HostDetailScreen)
        await pilot.press("escape")
        await goto_jobs(pilot)
        assert not any(k.startswith("k2:") for k in table(app).order)


@pytest.mark.posix                # asserts file mode 0600
async def test_crash_report_written(tmp_path: Any) -> None:
    from runfleet.ui.app import write_crash_report

    try:
        raise ValueError("boom")
    except ValueError as exc:
        path = write_crash_report(exc, tmp_path)
    assert path is not None and path.name.startswith("crash-")
    text = path.read_text()
    assert "ValueError: boom" in text and "RunFleet" in text and "Textual" in text
    assert oct(path.stat().st_mode & 0o777) == "0o600"


async def test_core_notices_become_toasts(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.notices = ["another RunFleet (pid 4242) is polling — remote load doubles"]
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause(0.2)
        assert any("pid 4242" in m for m in toasts(app))
        n = len(toasts(app))
        svc.inject("vega", "ok")                                   # a later flush does not repeat it
        await pilot.pause(FLUSH)
        assert sum("pid 4242" in m for m in toasts(app)) == 1 and len(toasts(app)) >= n


async def test_demo_app_never_writes_the_real_ui_state(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Leftover #1: `runfleet --demo` wrote ~/.local/state/runfleet/ui.json."""
    from runfleet.ui import app as app_module
    from runfleet.ui.app import RunFleetApp
    from tests.ui.fakes import FakeService

    target = tmp_path / "ui.json"
    monkeypatch.setattr(app_module, "ui_state_path", lambda: target)
    demo = RunFleetApp(FakeService(), animations=False, persist_state=False)  # type: ignore[arg-type]
    demo.save_ui_state(filter="x")
    assert demo.ui_state == {"filter": "x"} and not target.exists()
    real = RunFleetApp(FakeService(), animations=False)  # type: ignore[arg-type]
    real.save_ui_state(filter="y")
    assert target.exists()


# ---------------------------------------------------------------- review 2026-10: key bursts, exit races, small terminals
def _burst(app: RunFleetApp, keys: list[str]) -> None:
    from textual import events
    for k in keys:
        app.post_message(events.Key(k, k if len(k) == 1 else None))


async def test_key_bursts_never_crash_a_modal_mount(make_app: MakeApp) -> None:
    """Review: mashed / batched keys raised NoMatches('#menu-box') in ActionMenuScreen.on_mount (crash report)."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        for _ in range(20):
            _burst(app, ["x", "escape", "enter", "escape", "question_mark", "escape", "e", "escape"])
            await pilot.pause(0.01)
        await pilot.pause(0.5)
        assert app.crash_report is None and app.return_code is None
        assert isinstance(app.screen, MainScreen)


async def test_q_right_after_a_screen_change_does_not_quit(make_app: MakeApp) -> None:
    """Review: `x,q` in one burst — the q typed for the action menu landed on MainScreen and quit."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        app.action_action_menu()
        app.pop_screen()                                     # menu gone again before the q is handled
        await app.run_action("app.back_or_quit")
        await pilot.pause()
        assert app.return_code is None and isinstance(app.screen, MainScreen)
        app._last_push = float("-inf")                       # a deliberate q later still quits
        await app.run_action("app.back_or_quit")
        await pilot.pause()
    assert app.return_code == 0


async def test_keys_after_quit_open_nothing(make_app: MakeApp) -> None:
    """M3 F5: `q` then `e` — HostErrorScreen pushed while exiting crashed in on_mount (rc 1)."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app._last_push = float("-inf")
        _burst(app, ["q", "e", "question_mark", "x"])
        await pilot.pause(0.3)
    assert app.crash_report is None and app.return_code == 0


async def test_screens_opened_while_too_small_stay_below_the_placeholder(make_app: MakeApp) -> None:
    """Review: `?` while too small left a TooSmallScreen in the stack after the resize ('terminal too small' at
    full size); q while small popped the placeholder."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.resize_terminal(50, 12)
        await pilot.pause()
        assert isinstance(app.screen, TooSmallScreen)
        app.action_help()                                    # e.g. F1, or a worker opening a modal
        await pilot.pause()
        assert isinstance(app.screen, TooSmallScreen)
        await pilot.press("q")                               # swallowed while small
        await pilot.pause()
        assert isinstance(app.screen, TooSmallScreen) and app.return_code is None
        await pilot.resize_terminal(*BIG)
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        assert not any(isinstance(s, TooSmallScreen) for s in app.screen_stack)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_help_closes_on_question_mark_and_q_is_always_a_letter(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        await pilot.press("question_mark")
        await pilot.pause()
        await pilot.press("l", "o", "q")                     # q is a letter while searching
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen) and app.screen.query_one("#help-filter").value == "loq"  # type: ignore[attr-defined]
        for _ in range(3):
            await pilot.press("backspace")
        await pilot.pause()
        await pilot.press("q", "o", "s")                     # "qos": closing on q would sort and open a shell below
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen) and app.screen.query_one("#help-filter").value == "qos"  # type: ignore[attr-defined]
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen) and jobs_view(app).sort == "host"
        help_text = __import__("runfleet.ui.screens.help", fromlist=["x"]).legend(["jobs", "global"], 100).plain
        assert "Legend" in help_text and "marked" in help_text


async def test_no_color_cursor_row_is_marked_by_an_attribute(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: under NO_COLOR the cursor (background only) vanished in every table without a ▸ marker."""
    monkeypatch.setenv("NO_COLOR", "1")
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")                      # GPUs view: no marker column
        await pilot.pause()
        assert app.has_class("-no-color")
        table = app.screen.query_one(DataTable)
        style = table.get_component_styles("datatable--cursor")
        assert style.text_style.reverse or style.text_style.underline


async def test_reordered_rows_keep_the_selected_job_on_its_screen_row() -> None:
    """Review: a reorder (GPU%/elapsed sort, every sstat cycle) rebuilt the table and the selected row jumped."""
    from textual.app import App, ComposeResult

    from runfleet.ui.widgets.job_table import JobTable, RowContext, layout
    from tests.ui.fakes import T0, slurm_job

    jobs = [slurm_job("k2", str(2331789 + i), f"job{i}", JobState.RUNNING) for i in range(40)]
    ctx = RowContext(now=T0, host_colors={}, problem_hosts={}, stale_hosts=frozenset(), marked=frozenset(),
                     thresholds=(10.0, 75.0))

    class TableApp(App[None]):
        def compose(self) -> ComposeResult:
            yield JobTable(id="t")

    app = TableApp()
    async with app.run_test(size=(100, 20)) as pilot:
        table = app.query_one(JobTable)
        table.set_columns(layout(98, ("host", "id", "name", "state", "elapsed"), show_user=False))
        table.set_rows(jobs, ctx)
        await pilot.pause()
        table.move_cursor(row=30)
        await pilot.pause()
        table.move_cursor(row=25)                              # cursor somewhere in the middle of the viewport
        await pilot.pause()
        key, offset = table.cursor_key, table.cursor_row - int(table.scroll_y)
        table.set_rows(list(reversed(jobs)), ctx)              # same jobs, new order
        await pilot.pause()
        await pilot.pause()
        assert table.cursor_key == key
        assert table.cursor_row - int(table.scroll_y) == offset


async def test_slash_searches_the_log_instead_of_leaving_it(make_app: MakeApp) -> None:
    """Review: `/` in the log viewer / job detail popped every screen and opened the Jobs filter."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await move_to(pilot, app, "k2:2332861")
        await pilot.press("l")
        await pilot.pause(0.2)
        screen = app.screen
        assert isinstance(screen, LogScreen)
        svc.push_log(LogUpdate(lines=("alpha", "needle one", "beta", "needle two")))
        await pilot.pause(0.2)
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, LogScreen)
        await pilot.press(*"needle", "enter")
        await pilot.pause()
        view = screen.query_one(LogView)
        assert view.search_note is not None and "1/2" in view.search_note and not view.follow
        await pilot.press("n")
        assert view.search_note is not None and "2/2" in view.search_note
        await pilot.press("escape")                              # clears the search, stays in the log
        assert view.search_note is None and isinstance(app.screen, LogScreen)
        await pilot.press("q")
        await pilot.pause()
        await pilot.press("enter")                               # job detail: `/` opens the log viewer's search
        await pilot.pause(0.2)
        assert isinstance(app.screen, JobDetailScreen)
        await pilot.press("slash")
        await pilot.pause(0.2)
        assert isinstance(app.screen, LogScreen) and app.screen.query_one("#log-search").display


async def test_clicking_a_card_selects_without_scoping(make_app: MakeApp) -> None:
    """Review: a click (and so every double-click) on a Fleet card scoped the whole app to that host."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.click("#card-1")
        await pilot.pause()
        assert app.scope is None and app.focused is app.screen.query_one("#card-1")
        await pilot.click("#card-1", times=2)
        await pilot.pause()
        assert isinstance(app.screen, HostDetailScreen) and app.scope is None


async def test_gpus_view_says_sort_and_why_it_is_empty(make_app: MakeApp) -> None:
    """Review: sort only in a toast; an empty view for a host needing auth gave no reason."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        await pilot.press("o")
        await pilot.pause()
        assert "sort: util ▼" in view.status_text(120).plain
        idx = [(r.host, r.idx) for r in view.rows if not r.aggregate]
        view.sort = "host"
        view.refresh_data()
        hosts_in_order = [r for r in view.rows if not r.aggregate]
        for a, b in zip(hosts_in_order, hosts_in_order[1:], strict=False):
            if a.host == b.host and a.idx.isdigit() and b.idx.isdigit():
                assert int(a.idx) < int(b.idx)
        assert idx
        app.set_scope("lynx")                                  # needs auth in the fake world
        svc.store.hosts["lynx"].snapshot = None                 # and nothing collected yet
        view.refresh_data()
        note = str(view.query_one("#gpus-note").render())
        assert "lynx: needs auth" in note


async def test_hidden_marks_are_not_acted_on(make_app: MakeApp) -> None:
    """Review: V then `/eval` — 3 rows visible but the action menu targeted all 36 marked jobs."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        await all_preset(pilot, app)
        view = jobs_view(app)
        await pilot.press("V")
        await pilot.pause()
        total = len(view.marks)
        assert total > 3
        await pilot.press("slash", *"smoke", "enter")
        await pilot.pause(0.3)
        targets = view.action_targets()
        assert 0 < len(targets) < total and all(str(k) in view.visible_keys for k in targets)
        assert "hidden" in view.status_text(140).plain


async def test_sort_arrow_agrees_between_status_and_header(make_app: MakeApp) -> None:
    from runfleet.ui import query as q

    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot)
        view = jobs_view(app)
        for column in ("elapsed", "gpu_util", "name"):
            view.sort, view.reverse = column, False
            view.refresh_data()
            await pilot.pause()
            arrow = q.sort_arrow(column, False)
            assert arrow in view.status_text(140).plain
            key = {"gpu_util": "gpu_util"}.get(column, column)
            header = str(table(app).columns[key].label.plain) if key in table(app).columns else arrow  # type: ignore[index]
            assert header.endswith(arrow), header
        assert q.sort_arrow("elapsed", False) == "▼" and q.sort_arrow("name", False) == "▲"


def test_hint_bar_keeps_help_and_follows_the_selection() -> None:
    from runfleet.ui.widgets.hintbar import HintBar

    narrow = HintBar.build(["fleet", "main", "global"], 80).plain
    assert narrow.rstrip().endswith("? help")                    # reserved, never dropped
    jobs = HintBar.build(["jobs", "main", "global"], 120).plain
    assert "/ filter" in jobs and "? help" in jobs
    ended = HintBar.build(["job_detail", "global"], 140, overrides={"app.quick_kill": None}).plain
    assert "^k" not in ended
    task = HintBar.build(["host_detail", "host_cluster", "global"], 140, overrides={"app.quick_kill": "cancel"}).plain
    assert "^k cancel" in task and "terminate" not in task


def test_left_column_shows_estimated_start_for_queued_jobs() -> None:
    from datetime import timedelta

    from runfleet.ui.widgets.job_table import left_text
    from tests.ui.fakes import T0, slurm_job

    pd = dataclasses.replace(slurm_job("k2", "1", "x", JobState.PENDING), est_start=True,
                             start_time=T0 + timedelta(minutes=37))
    assert left_text(pd, T0).plain == "~in 37m"
    assert left_text(dataclasses.replace(pd, est_start=False, start_time=None), T0).plain == "—"


async def test_palette_lists_only_runfleet_commands(make_app: MakeApp) -> None:
    from runfleet.ui.commands import RunFleetCommands

    app, svc = make_app()
    assert set(app.COMMANDS) == {RunFleetCommands}


def test_id_column_sized_from_labels_and_cut_from_the_left() -> None:
    from runfleet.ui.widgets.job_table import job_cells, layout
    from tests.ui.fakes import T0, slurm_job
    from runfleet.ui.widgets.job_table import RowContext

    cols = layout(120, ("host", "id", "name", "state", "elapsed"), show_user=False, id_width=10)
    assert dict(cols)["id"] == 11                                 # 10 + gap
    job = dataclasses.replace(slurm_job("lyra", "31377076_3", "sweep", JobState.RUNNING))
    ctx = RowContext(now=T0, host_colors={}, problem_hosts={}, stale_hosts=frozenset(), marked=frozenset(),
                     thresholds=(10.0, 75.0))
    narrow = job_cells(job, [("marker", 2), ("id", 9)], ctx, cursor=False)[1].plain
    assert narrow.endswith("_3")
