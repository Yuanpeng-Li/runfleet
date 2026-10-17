"""[WP3] Regression tests for the UI-shell review findings (internal review notes), on FakeService."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from textual.app import SuspendNotSupported
from textual.widgets import Button, Checkbox, Input

import runfleet.ui.app as app_module
from runfleet.config import HostConfig
from runfleet.models import ConnState, JobKey, JobState, PartitionSummary
from runfleet.ui import render, theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.confirm import ConfirmScreen
from runfleet.ui.screens.help import HelpScreen
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.host_error import DoctorScreen, HostErrorScreen, ManualCommandScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.screens.settings import GENERAL, SettingsPanel, SettingsScreen
from runfleet.ui.widgets.filter_bar import FilterBar
from runfleet.ui.widgets.host_card import card_lines
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.node_table import NodeTable, PartitionTable
from runfleet.ui.widgets.proc_table import ProcTable
from tests.ui.fakes import T0, FakeService, slurm_job

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
SMALL = (80, 24)
TINY = (60, 16)
FLUSH = 0.3


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


def jobs_view(app: RunFleetApp) -> JobsView:
    return app.screen.query_one(JobsView)


async def goto_jobs(pilot: Any, app: RunFleetApp, preset: str = "all") -> None:
    await pilot.press("tab")
    await pilot.pause()
    while jobs_view(app).preset != preset:
        await pilot.press("f")
    await pilot.pause()


async def mark(pilot: Any, app: RunFleetApp, *keys: str) -> None:
    for key in keys:
        app.screen.query_one(JobTable).move_to_key(key)
        await pilot.pause()
        await pilot.press("space")
    await pilot.pause()


# ---------------------------------------------------------------- UI-01 dropped hosts and marks
async def test_bulk_action_reports_a_dropped_host_and_keeps_its_marks(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await goto_jobs(pilot, app)
        await mark(pilot, app, "k2:2332861", "vega:724479")
        svc.inject("vega", "auth")
        await pilot.pause(FLUSH)
        await pilot.press("ctrl+k")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen)
        assert [p.host for p in screen.plans] == ["k2"]
        assert "not acted on — vega: needs auth (1 job)" in str(screen.query_one("#confirm-body").render())
        await pilot.press("y")
        await pilot.pause(FLUSH)
        assert svc.store.job(JobKey("k2", "2332861")).state is JobState.CANCELLED
        assert jobs_view(app).marks == {"vega:724479"}          # the handled job is unmarked, the skipped one stays
        assert any("Cancelled 1 job on k2" in m and "1 skipped (vega: needs auth)" in m for m in toasts(app))


# ---------------------------------------------------------------- UI-03 removal outlives the Settings overlay
async def test_closing_settings_does_not_cancel_a_host_removal(make_app: MakeApp) -> None:
    app, svc = make_app()
    release = asyncio.Event()
    original = svc.remove_host

    async def slow_remove(host: str, *, save: bool = True) -> None:
        await release.wait()
        await original(host, save=save)

    svc.remove_host = slow_remove  # type: ignore[method-assign]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("comma")
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.selected["hosts"] = [r.name for r in screen.visible_rows()].index("labstation")
        screen.toggle(screen.visible_rows()[screen.selected["hosts"]])
        await pilot.pause()
        if screen.confirm:                                  # the host runs jobs of mine: second press
            screen.toggle(screen.visible_rows()[screen.selected["hosts"]])
            await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        release.set()
        await pilot.pause(FLUSH)
        assert "labstation" not in svc.store.hosts


# ---------------------------------------------------------------- UI-12 large confirm dialogs
async def test_large_confirm_keeps_its_buttons_on_screen(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=SMALL) as pilot:
        await goto_jobs(pilot, app)
        await pilot.press("V")
        await pilot.pause()
        await pilot.press("ctrl+k")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen)
        button = screen.query_one("#confirm", Button).region
        box = screen.query_one("#confirm-box").region
        assert button.y + button.height <= box.y + box.height <= app.size.height
        assert screen.query_one("#confirm-scroll").max_scroll_y > 0
        await pilot.press("n")


# ---------------------------------------------------------------- UI-30 every number field is validated
async def test_toggling_a_checkbox_never_bypasses_an_invalid_number(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("4")
        await goto_jobs(pilot, app, "fresh")
        await pilot.press("ctrl+k")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen) and screen.action_id == "task.terminate"
        screen.query_one("#p-grace_s", Input).value = "9999"
        await pilot.pause()
        assert screen.query_one("#confirm", Button).disabled
        box = screen.query_one(Checkbox)
        box.toggle()
        await pilot.pause()
        box.toggle()
        await pilot.pause()
        assert screen.query_one("#confirm", Button).disabled
        screen.action_confirm()
        await pilot.pause()
        assert app.screen is screen and not svc.called("execute")
        screen.query_one("#p-grace_s", Input).value = "20"
        await pilot.pause()
        assert not screen.query_one("#confirm", Button).disabled and "KILL after 20 s" in screen.plans[0].preview


# ---------------------------------------------------------------- UI-14 / UI-35 / UI-36 host detail
async def test_host_detail_panels_stay_reachable_at_the_minimum_size(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=TINY) as pilot:
        await pilot.pause()
        app.action_open_host("k2")
        await pilot.pause()
        layout = app.screen.query_one("#cluster-layout")
        caps = app.screen.query_one("#caps-box")
        assert layout.max_scroll_y > 0
        assert caps.virtual_region.bottom <= layout.virtual_size.height


async def test_host_detail_moves_focus_when_the_host_type_is_detected(make_app: MakeApp) -> None:
    app, svc = make_app()
    snap = svc.store.hosts["k2"].snapshot
    svc.store.hosts["k2"].snapshot = None
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.action_open_host("k2")
        await pilot.pause()
        assert isinstance(app.focused, ProcTable)
        svc.store.hosts["k2"].snapshot = snap
        svc.tick()
        await pilot.pause(FLUSH)
        assert isinstance(app.focused, NodeTable)


async def test_job_keys_on_a_partition_row_do_not_hit_the_node_cursor_job(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        app.action_open_host("k2")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostDetailScreen)
        nodes = screen.query_one(NodeTable)
        nodes.move_cursor(row=nodes.order.index("n-40-01"))
        await pilot.pause()
        assert screen.current_job() is not None
        screen.query_one(PartitionTable).focus()
        await pilot.pause()
        assert screen.current_job() is None
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmScreen) and any("nothing selected" in m for m in toasts(app))
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.focused, NodeTable)


# ---------------------------------------------------------------- UI-16 / UI-21 / UI-22 / UI-33 settings
async def test_settings_general_list_follows_the_selection_when_short(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=TINY) as pilot:
        await pilot.pause()
        await pilot.press("comma", "tab")
        await pilot.pause()
        for _ in range(5):
            await pilot.press("down")
        await pilot.pause()
        text = str(app.screen.query_one(SettingsPanel).render())
        assert "remember hosts" in text and f"6/{len(GENERAL)}" in text


async def test_p_polls_a_saved_host_this_run_left_out(make_app: MakeApp, service: FakeService) -> None:
    service.saved.append(HostConfig(name="edge1", ssh="edge1"))
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("comma")
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.selected["hosts"] = [r.name for r in screen.visible_rows()].index("edge1")
        await pilot.press("p")
        await pilot.pause()
        assert "edge1" in svc.store.hosts and not svc.called("remove_host")
        assert [h.name for h in svc.saved].count("edge1") == 1
        assert "edge1: polling started" in str(screen.query_one(SettingsPanel).render())


async def test_added_target_never_takes_over_an_existing_name(make_app: MakeApp, service: FakeService) -> None:
    service.saved.append(HostConfig(name="alice_box", ssh="other.example.edu"))
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("comma", "a", *"alice@box", "enter")
        await pilot.pause(0.1)
        assert ("alice_box_2", "alice@box", True) in svc.called("add_host")
        assert svc.store.hosts["alice_box_2"].config.ssh == "alice@box"


async def test_an_older_connection_test_never_overwrites_a_newer_one(make_app: MakeApp) -> None:
    from runfleet.models import DetectReport

    app, svc = make_app()
    first, second = asyncio.Event(), asyncio.Event()
    answers = iter([(first, ConnState.NEEDS_AUTH), (second, ConnState.OK)])

    async def probe(alias: str) -> DetectReport:
        gate, state = next(answers)
        await gate.wait()
        return DetectReport(alias, state, None, None, "user", {}, 0.1)

    svc.probe_alias = probe  # type: ignore[method-assign]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("comma")
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        row = screen.visible_rows()[0]
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        second.set()
        await pilot.pause()
        first.set()
        await pilot.pause()
        result = screen.tests[row.ssh]
        assert isinstance(result, DetectReport) and result.state is ConnState.OK


# ---------------------------------------------------------------- UI-23 hint clicks with the filter focused
async def test_hint_click_runs_the_key_even_while_the_filter_has_focus(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot, app, "fresh")
        await pilot.press("slash")
        await pilot.pause()
        bar = app.screen.query_one(FilterBar)
        assert bar.has_focus
        app.action_press("l")                       # what a click on "l logs" runs
        await pilot.pause()
        assert isinstance(app.screen, LogScreen) and bar.value == ""


# ---------------------------------------------------------------- UI-24 / UI-45 fleet card
def test_card_next_line_prefers_a_real_estimate(service: FakeService) -> None:
    state = service.store.hosts["k2"]
    assert state.snapshot is not None
    unknown = slurm_job("k2", "3331790", "no-estimate", JobState.PENDING, reason="Resources")
    soon = dataclasses.replace(slurm_job("k2", "3331791", "estimated", JobState.PENDING, reason="Priority"),
                               start_time=T0 + timedelta(minutes=10), est_start=True)
    jobs = {j.key.id: j for j in state.snapshot.jobs.values() if j.state is not JobState.PENDING}
    jobs |= {"3331790": unknown, "3331791": soon}
    state.snapshot = dataclasses.replace(state.snapshot, jobs=jobs)
    text = "\n".join(t.plain for t in card_lines(state, T0, width=100))
    nxt = next(line for line in text.splitlines() if line.startswith("next"))
    assert "estimated" in nxt and "(10m)" in nxt


def test_free_gpus_names_the_planned_part() -> None:
    p = PartitionSummary("gpu", 4, 16, 8, 6, {"a100": (8, 16)}, {"mixed": 4})
    assert render.free_gpus(p).plain == "8/16 (6 planned)"
    assert render.free_gpus(dataclasses.replace(p, gpus_free_planned=0)).plain == "8/16"


def test_card_free_gpus_fit_every_partition_or_say_how_many_more(service: FakeService) -> None:
    state = service.store.hosts["lyra"]
    assert state.snapshot is not None
    parts = tuple(PartitionSummary(n, 1, t, f, pl, {}, {}) for n, f, t, pl in (
        ("free-gpu", 6, 136, 2), ("free-gpuxl", 15, 40, 0), ("gpu", 3, 132, 1), ("gpu-highmem", 0, 4, 0),
        ("gpuxl", 2, 40, 0)))
    state.snapshot = dataclasses.replace(state.snapshot, partitions=parts)

    def free(width: int) -> list[str]:
        lines = [t.plain.rstrip() for t in card_lines(state, T0, width=width)]
        i = next(i for i, line in enumerate(lines) if line.startswith("free GPUs"))
        return [lines[i]] + [x for x in lines[i + 1:i + 2] if x.startswith(" " * 10)]

    wide = free(140)                                   # one line, "(planned)" notes kept
    assert len(wide) == 1 and "gpu 3/132 (1 planned)" in wide[0] and "gpuxl 2/40" in wide[0]
    two = free(58)                                     # the paid "gpu" partition is not cut off: a second line
    assert len(two) == 2 and "planned" not in "".join(two) and all(len(x) <= 58 for x in two)
    assert all(f" {n} " in " ".join(two) + " " for n in ("gpu", "gpuxl", "gpu-highmem"))
    narrow = free(40)
    assert len(narrow) == 2 and narrow[1].endswith("more") and all(len(x) <= 40 for x in narrow)


# ---------------------------------------------------------------- UI-46 GPU occupancy
def test_a_gpu_holding_memory_is_not_free(service: FakeService) -> None:
    snap = service.store.hosts["labstation"].snapshot
    assert snap is not None
    idle = next(g for g in snap.gpus if not g.processes and not g.util_pct)
    assert not render.gpu_in_use(idle)
    assert render.gpu_in_use(dataclasses.replace(idle, mem_used_mib=40 * 1024.0))   # another container, 0 %
    assert render.gpu_in_use(dataclasses.replace(idle, util_pct=None, mem_used_mib=None))   # unknown


# ---------------------------------------------------------------- UI-34 doctor under a dialog
async def test_a_late_doctor_result_does_not_cover_another_dialog(make_app: MakeApp) -> None:
    app, svc = make_app()
    release = asyncio.Event()
    original = svc.doctor

    async def slow_doctor(host: str, *, reset: bool = False) -> Any:
        await release.wait()
        return await original(host)

    svc.doctor = slow_doctor  # type: ignore[method-assign]
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        app.show_doctor("k2")
        await pilot.pause()
        app.action_host_error("lyra")
        await pilot.pause()
        release.set()
        await pilot.pause(FLUSH)
        assert isinstance(app.screen, HostErrorScreen)
        assert any("doctor k2 finished" in m for m in toasts(app))
        await pilot.press("escape")
        await pilot.pause()
        app.show_doctor("k2")                        # the palette entry shows the held report, no re-run
        await pilot.pause(FLUSH)
        assert isinstance(app.screen, DoctorScreen)


# ---------------------------------------------------------------- UI-37 / UI-41 / UI-40 / UI-31 external programs
async def test_manual_command_records_the_target_host(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()

    def no_suspend() -> Any:
        raise SuspendNotSupported()

    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "suspend", no_suspend)
        app.set_scope("k2")
        await pilot.pause()
        app.connect("lynx")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ManualCommandScreen) and screen.host == "lynx"
        await pilot.click("#done")
        await pilot.pause(FLUSH)
        assert svc.called("after_connect") == [("lynx", 0)]
        app.action_shell_host("k2")
        await pilot.pause()
        assert isinstance(app.screen, ManualCommandScreen) and not app.screen.query("#done")


async def test_a_shell_that_fails_at_once_waits_for_enter(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    prompts: list[str] = []
    monkeypatch.setattr(app_module.subprocess, "run", lambda argv, check: type("R", (), {"returncode": 1})())
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "")
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "suspend", contextlib.nullcontext)
        assert app.run_external(["ssh", "-t", "k2"], pause_on_error=True) == 1
        assert app.run_external(["less", "x"]) == 1                    # pagers / editors do not pause
        assert len(prompts) == 1 and "exited with 1" in prompts[0]


async def test_shell_picker_rejects_a_node_the_job_left(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None
    two = dataclasses.replace(job, nodes=("n-40-01", "n-40-02"), nodelist="n-40-[01-02]", batch_host="n-40-01")
    svc.put_job(two)
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await goto_jobs(pilot, app)
        app.screen.query_one(JobTable).move_to_key("k2:2332861")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        svc.put_job(dataclasses.replace(two, nodes=("n-41-01",), nodelist="n-41-01", batch_host="n-41-01"))
        await pilot.press("down", "enter")                             # n-40-02: no longer allocated
        await pilot.pause(FLUSH)
        assert runs == [] and any("n-40-02 is no longer part of job 2332861" in m for m in toasts(app))


async def test_a_second_connect_during_a_slow_check_starts_no_second_login(make_app: MakeApp,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    release = asyncio.Event()

    async def slow_prepare(host: str) -> bool:
        await release.wait()
        return False

    svc.prepare_connect = slow_prepare  # type: ignore[method-assign]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.connect("lynx")
        app.connect("lynx")
        await pilot.pause()
        release.set()
        await pilot.pause(FLUSH)
        assert runs == [svc.connect_argv("lynx")]


# ---------------------------------------------------------------- UI-42 / UI-43 host state changes
async def test_host_error_dialog_follows_the_host(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.action_host_error("lynx")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, HostErrorScreen)
        assert "needs auth" in str(screen.query_one("#err-box").border_title)
        svc.set_ok("lynx")
        await pilot.pause(FLUSH)
        assert str(screen.query_one("#err-box").border_title) == "lynx · ok"
        assert app.focused is screen.query_one("#close")


async def test_unreachable_to_needs_auth_is_announced(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        svc.inject("vega", "unreachable")
        await pilot.pause(FLUSH)
        svc.inject("vega", "auth")
        await pilot.pause(FLUSH)
        assert any("vega: needs auth — press c" in m for m in toasts(app))


# ---------------------------------------------------------------- UI-48 / UI-49 / UI-52 colour independence
def test_compact_chips_tell_problem_states_apart() -> None:
    from runfleet.models import HostStatus

    def chip(state: ConnState, form: Any) -> str:
        return render.host_chip(HostStatus("lynx", state), form=form, number=5).plain

    for form in ("name", "num"):
        texts = {chip(s, form) for s in (ConnState.UNREACHABLE, ConnState.NEEDS_AUTH, ConnState.HOSTKEY)}
        assert len(texts) == 3
    assert chip(ConnState.NEEDS_AUTH, "name") == "✖lynx:a" and chip(ConnState.OK, "num") == "●5"


def test_light_host_palette_is_readable_on_white(monkeypatch: pytest.MonkeyPatch) -> None:
    def luminance(color: str) -> float:
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    assert all(1.05 / (luminance(c) + 0.05) >= 4.5 for c in theme.HOST_PALETTE_LIGHT)
    theme.configure("light")
    try:
        assert theme.host_color(None, 2) == theme.HOST_PALETTE_LIGHT[2]
    finally:
        theme.configure("dark")


async def test_empty_no_color_means_unset_for_textual_too(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        assert not app.no_color and not app.has_class("-no-color")


# ---------------------------------------------------------------- UI-50 F1 inside dialogs
async def test_f1_inside_a_dialog_lists_its_keys(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await goto_jobs(pilot, app, "fresh")
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen) and app.screen.contexts == ["confirm"]
        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)


# ---------------------------------------------------------------- UI-19 MY JOBS column
async def test_node_table_lists_only_my_jobs(make_app: MakeApp) -> None:
    app, svc = make_app()
    other = slurm_job("k2", "2338403", "someone-else", JobState.RUNNING, elapsed=timedelta(hours=1),
                      node="n-31-01", user="alice")
    svc.put_job(other)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.action_open_host("k2")
        await pilot.pause()
        nodes = app.screen.query_one(NodeTable)
        assert "2338403" not in str(nodes.get_cell("n-31-01", "jobs"))


# ---------------------------------------------------------------- UI-57 screenshots show the current header
def test_readme_screenshots_show_the_settings_gear() -> None:
    from pathlib import Path

    shots = sorted((Path(__file__).resolve().parents[2] / "docs" / "screenshots").glob("*.svg"))
    assert shots
    for shot in shots:
        text = shot.read_text(encoding="utf-8")
        assert "⚙" in text or "&#9881;" in text, f"{shot.name}: regenerate with scripts/screenshots.py"


async def test_a_shell_on_a_host_waiting_for_login_reconnects_it(make_app: MakeApp,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """UI-39: a managed shell becomes the master, so after it returns a NEEDS_AUTH host is re-prepared and polled
    (after_connect) instead of waiting for a manual retry; a healthy host only gets a refresh."""
    app, svc = make_app()
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: 0)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        assert app.host_state("lynx").status.state is ConnState.NEEDS_AUTH  # type: ignore[union-attr]
        app.action_shell_host("lynx")
        app.action_shell_host("k2")
        await pilot.pause(FLUSH)
        assert svc.called("after_connect") == [("lynx", 0)] and svc.called("refresh") == [(("k2",),)]


async def test_proc_table_keeps_the_selected_pid_across_a_resize(make_app: MakeApp) -> None:
    """UI-15: a width change rebuilds the columns (clear(columns=True)); the selected PID must survive it."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.action_open_host("labstation")
        await pilot.pause()                                 # other users' processes are shown by default
        procs = app.screen.query_one(ProcTable)
        pid = procs.order[-1]
        procs.move_cursor(row=len(procs.order) - 1)
        await pilot.resize_terminal(90, 35)
        svc.tick()
        await pilot.pause(FLUSH)
        assert procs.cursor_key == str(pid) and procs.cursor_row == procs.order.index(pid) > 0


async def test_table_home_end_and_page_keys_move_the_cursor(make_app: MakeApp) -> None:
    """UI-51: the focused JobTable got Home/End/PgUp/PgDn before JobsView (DataTable scrolling, cursor unmoved)."""
    app, svc = make_app()
    async with app.run_test(size=(120, 20)) as pilot:
        await goto_jobs(pilot, app)
        table = app.screen.query_one(JobTable)
        table.focus()
        await pilot.press("end")
        assert table.cursor_row == table.row_count - 1 > 0
        await pilot.press("home")
        assert table.cursor_row == 0
        await pilot.press("ctrl+d")
        step = table.cursor_row
        await pilot.press("ctrl+u", "pagedown")
        assert step > 0 and table.cursor_row == step
        await pilot.press("pageup")
        assert table.cursor_row == 0


async def test_gpus_view_shows_only_my_slurm_jobs_unless_others_are_on(make_app: MakeApp) -> None:
    """UI-19: the view is "GPUs of my running Slurm jobs"; other users' jobs appear only with the Jobs view's `u`."""
    from runfleet.ui.widgets.gpus_view import GPUsView
    app, svc = make_app()
    svc.put_job(slurm_job("k2", "2338403", "theirs", JobState.RUNNING, user="bob", node="n-40-03",
                          elapsed=timedelta(hours=1), gpus=2, idx=(0, 1)))
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        assert not any(r.key == "job:k2:2338403" for r in view.build_rows())
        app.others_visible = True
        assert any(r.key == "job:k2:2338403" for r in view.build_rows())


async def test_c_logs_in_from_job_detail_and_the_log_viewer(make_app: MakeApp,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """UI-20: job detail and the log viewer say "c to log in"; `c` is bound there now and connects the job's host."""
    app, svc = make_app()
    connected: list[str] = []
    monkeypatch.setattr(app, "connect", connected.append)
    async with app.run_test(size=BIG) as pilot:
        app.open_job(JobKey("k2", "2332861"))
        await pilot.pause()
        await pilot.press("c")
        app.action_log_viewer()
        await pilot.pause()
        assert isinstance(app.screen, LogScreen)
        await pilot.press("c")
        assert connected == ["k2", "k2"]
