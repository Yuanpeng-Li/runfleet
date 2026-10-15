"""[WP3] Settings overlay (§7.13): open/close, sections by key and click, host toggles (keyboard and mouse), confirm,
test, add, filter, reorder, General values (notifications included), About, broken config, removed-host cleanup — on
FakeService."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from runfleet.config import NOTIFY_EVENTS
from runfleet.models import ConnState, DetectReport, ErrorKind, HostError, JobKey, JobState
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.screens.settings import (FAILURES, GENERAL, SettingsPanel, SettingsScreen, editor_argv,
                                          host_name_for, probe_summary)
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.host_card import HostCard
from tests.ui.fakes import FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
SMALL = (80, 24)
LIST_Y = 7                 # first list row inside the panel (title, tabs, rule, blank, heading, description, blank)


def settings(app: RunFleetApp) -> SettingsScreen:
    assert isinstance(app.screen, SettingsScreen), app.screen
    return app.screen


def panel_text(app: RunFleetApp) -> str:
    return str(settings(app).query_one(SettingsPanel).render())


def row_names(app: RunFleetApp) -> list[str]:
    return [r.name for r in settings(app).visible_rows()]


def cards(app: RunFleetApp) -> list[tuple[str, int]]:
    main = next(s for s in app.screen_stack if isinstance(s, MainScreen))
    return [(c.host, c.index) for c in main.query(HostCard)]


async def open_settings(pilot: Any, *keys: str) -> SettingsScreen:
    await pilot.pause()
    await pilot.press("comma", *keys)
    await pilot.pause(0.1)
    return settings(pilot.app)


# ---------------------------------------------------------------- open / close / sections
async def test_comma_opens_and_closes_and_esc_closes(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        assert screen.section == "hosts" and " settings" in panel_text(app) and "demo — not saved" in panel_text(app)
        await pilot.press("comma")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)
        await open_settings(pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, MainScreen)


async def test_sections_by_keyboard_and_by_click(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        for key, section in (("tab", "general"), ("right", "about"), ("right", "hosts"), ("left", "about"),
                             ("shift+tab", "general")):
            await pilot.press(key)
            assert screen.section == section, key
        await pilot.click("#settings-panel", offset=(3, 1))      # " Hosts " pill
        assert screen.section == "hosts"
        await pilot.click("#settings-panel", offset=(22, 1))     # " About " pill
        assert screen.section == "about"


async def test_header_gear_and_palette_open_settings(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        header = app.screen.query_one(FleetHeader)
        assert header.text.plain.rstrip().endswith("⚙")
        await pilot.click(FleetHeader, offset=(len(header.text.plain.rstrip()) - 1, 0))
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape")
        from runfleet.ui.commands import RunFleetCommands

        names = [n for n, _cb, _h in RunFleetCommands(app.screen)._commands()]   # type: ignore[arg-type]
        assert {"Settings", "Settings: general", "Settings: about"} <= set(names)


async def test_open_settings_at_start(clock: Any) -> None:
    from runfleet.config import FleetConfig

    svc = FakeService(clock, world=(FleetConfig(hosts=()), []))
    app = RunFleetApp(svc, clock=clock, animations=False, open_settings=True)   # type: ignore[arg-type]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(0.1)
        assert settings(app).section == "hosts"
        assert row_names(app) == svc.aliases
        await pilot.press("space")                                   # first alias: saved, polled, a card appears
        await pilot.pause(0.4)
        assert svc.called("add_host") == [("k2", "k2", True)]
        await pilot.press("escape")
        await pilot.pause(0.4)
        assert cards(app) == [("k2", 1)]


# ---------------------------------------------------------------- hosts
async def test_rows_merge_saved_adhoc_and_aliases(make_app: MakeApp, service: FakeService) -> None:
    from runfleet.config import HostConfig

    service.saved = [h for h in service.saved if h.name != "lynx"]     # lynx: ad hoc this run
    service.aliases = ["k2", "edge1", "me@box"]
    app, _ = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot)
        rows = settings(app).rows
        assert [(r.name, r.ssh, r.saved, r.running) for r in rows] == [
            ("k2", "k2", True, True), ("vega", "vega", True, True), ("lyra", "lyra", True, True),
            ("labstation", "labstation", True, True), ("lynx", "lynx", False, True),
            ("edge1", "edge1", False, False), ("me_box", "me@box", False, False)]
        text = panel_text(app)
        assert "unsaved" in text and "slurm 25.05.8" in text and "workstation · 4 GPUs" in text
        assert HostConfig  # noqa: B018 - imported for the type


async def test_space_saves_and_click_on_box_removes(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        names = row_names(app)
        i = names.index("edge1")
        screen.selected["hosts"] = i
        await pilot.press("space")
        await pilot.pause(0.4)
        assert svc.called("add_host")[-1] == ("edge1", "edge1", True)
        assert "edge1" in svc.store.hosts and any(h.name == "edge1" for h in svc.saved)
        assert ("edge1", 6) in cards(app) and "saved · polling started" in panel_text(app)
        lyra = row_names(app).index("lyra")
        await pilot.click("#settings-panel", offset=(4, LIST_Y + lyra))   # the [x] of lyra (no jobs of mine)
        await pilot.pause(0.4)
        assert svc.called("remove_host") == [("lyra", True)]
        assert "lyra" not in svc.store.hosts and [c for c, _ in cards(app)] == [
            "k2", "vega", "labstation", "lynx", "edge1"]
        assert [i for _c, i in cards(app)] == [1, 2, 3, 4, 5]           # renumbered


async def test_removing_a_host_with_my_jobs_needs_a_second_press(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("vega")
        await pilot.press("space")
        await pilot.pause()
        assert not svc.called("remove_host") and "space again removes it" in panel_text(app)
        await pilot.press("down", "up", "space")                         # moving cancels the confirm
        await pilot.pause()
        assert not svc.called("remove_host")
        await pilot.press("space")
        await pilot.pause(0.4)
        assert svc.called("remove_host") == [("vega", True)] and "vega removed" in panel_text(app)


async def test_click_on_a_row_selects_and_wheel_moves(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        await pilot.click("#settings-panel", offset=(30, LIST_Y + 3))
        assert screen.selected["hosts"] == 3
        from textual import events

        panel = screen.query_one(SettingsPanel)
        panel.post_message(events.MouseScrollDown(panel, 30, 9, 0, 1, 0, False, False, False))
        await pilot.pause()
        assert screen.selected["hosts"] == 4


async def test_test_key_probes_and_shows_the_result(make_app: MakeApp, service: FakeService) -> None:
    service.probe_reports["atlas"] = DetectReport("atlas", ConnState.UNREACHABLE,
                                                  HostError(ErrorKind.DNS, "could not resolve hostname atlas"),
                                                  None, "none", {}, None)
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("atlas")
        await pilot.press("t")
        await pilot.pause(0.2)
        assert svc.called("probe_alias") == [("atlas",)]
        assert "✖ could not resolve hostname" in panel_text(app)
        assert "atlas: could not resolve hostname atlas" in panel_text(app)


async def test_add_custom_target_by_typing(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot, "a")
        assert settings(app).editing == "add"
        await pilot.press(*"me@box.example")
        assert "+ ssh target: me@box.example" in panel_text(app)
        await pilot.press("enter")
        await pilot.pause(0.4)
        assert svc.called("add_host")[-1] == ("me_box.example", "me@box.example", True)
        assert settings(app).editing is None and "me_box.example" in row_names(app)
        await pilot.press("a", *"bad target", "enter")                   # whitespace: refused
        await pilot.pause()
        assert "not an ssh target" in panel_text(app) and len(svc.called("add_host")) == 1


async def test_filter_while_typing_and_keys_inside_the_input(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot, "slash")
        await pilot.press("e", "d")                                      # letters type, they are not bindings
        assert row_names(app) == ["edge1", "edge2"] and not svc.called("probe_alias")
        await pilot.press("t")                                           # still typing: no match, no probe
        assert row_names(app) == [] and not svc.called("probe_alias")
        await pilot.press("backspace", "enter")
        assert settings(app).filter == "ed" and settings(app).editing is None
        await pilot.press("slash", "escape")
        assert settings(app).filter == "" and len(row_names(app)) == 8
        assert isinstance(app.screen, SettingsScreen)                    # esc left the input, not the overlay


async def test_reorder_with_shift_keys(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("lynx")
        await pilot.press("K", "shift+up")
        await pilot.pause(0.4)
        assert svc.called("reorder_hosts")[-1] == (("k2", "vega", "lynx", "lyra", "labstation"), True)
        assert list(svc.store.hosts)[:3] == ["k2", "vega", "lynx"]
        assert row_names(app)[2] == "lynx" and screen.selected["hosts"] == 2
        screen.selected["hosts"] = row_names(app).index("edge1")
        await pilot.press("J")
        assert "save the host first" in panel_text(app)
        await pilot.press("escape")
        await pilot.pause(0.4)
        assert [c for c, _ in cards(app)][:3] == ["k2", "vega", "lynx"]


async def test_broken_config_is_shown_and_toggles_refused(make_app: MakeApp, service: FakeService) -> None:
    service.config_error = "config.toml: invalid TOML: line 3"
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot)
        assert "invalid TOML: line 3" in panel_text(app)
        await pilot.press("space")
        await pilot.pause()
        assert not svc.called("add_host") and not svc.called("remove_host")
        assert "cannot be read" in panel_text(app)


# ---------------------------------------------------------------- general / about
async def test_general_values_by_key_and_click(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot, "tab")
        await pilot.press("space")                                       # default view: fleet -> jobs
        assert svc.called("set_general")[-1] == ({"default_view": "jobs"}, True)
        assert "from the next start" in panel_text(app)
        await pilot.click("#settings-panel", offset=(39, LIST_Y + 1))    # refresh speed › " fast "
        await pilot.pause()
        assert svc.called("set_general")[-1] == ({"refresh_multiplier": 0.5}, True)
        assert app.multiplier == 0.5 and "×0.5" in app.screen_stack[1].query_one(FleetHeader).text.plain
        await pilot.press("down", "space")                               # read-only: off -> on
        assert svc.readonly and svc.config.general.readonly
        await pilot.press("down", "down", "space")                       # idle pause 2h -> 8h
        assert svc.config.general.idle_pause_after == 28800.0


def _row(key: str, table: str = "notify") -> int:
    return next(i for i, s in enumerate(GENERAL) if (s.table, s.key) == (table, key))


def _ended(svc: FakeService, jid: str, state: JobState) -> None:
    import dataclasses

    job = svc.store.job(JobKey("k2", jid))
    assert job is not None
    svc.put_job(dataclasses.replace(job, state=state, end_time=svc.clock.now()))


async def test_notification_rows_apply_live(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot, "tab")
        text = panel_text(app)
        assert "notifications" in text and "command hook: none (config.toml only)" in text
        assert screen._current(GENERAL[_row("events")]) == FAILURES             # the default is a named preset
        screen.selected["general"] = _row("enabled")
        await pilot.press("space")                                               # enabled: on -> off
        assert svc.called("set_notify")[-1] == ({"enabled": False}, True) and not svc.config.notify.enabled
        assert "notifications · enabled: off (this run only — demo)" in panel_text(app)
        await pilot.pause(0.3)
        _ended(svc, "2332861", JobState.FAILED)                                 # live: history, no pop-up
        await pilot.pause(0.3)
        assert [e.key.id for e in app.alerts.history] == ["2332861"]
        assert not any("2332861" in n.message for n in app._notifications)
        await pilot.press("space")                                               # back on: the next one pops up
        _ended(svc, "2332862", JobState.FAILED)
        await pilot.pause(0.3)
        assert any("2332862" in n.message for n in app._notifications)
        await pilot.press("down", "space")                                       # terminal: auto -> osc9
        assert svc.config.notify.terminal == "osc9"
        await pilot.click("#settings-panel", offset=(48, LIST_Y + _row("terminal") + 1))   # › " bell "
        await pilot.pause()
        assert svc.called("set_notify")[-1] == ({"terminal": "bell"}, True)
        assert "tmux: set -g allow-passthrough on" in panel_text(app)
        await pilot.press("down", "space")                                       # events: failures -> +gpu idle
        assert svc.config.notify.events == ("ended_bad", "restarted", "time_low", "gpu_idle")
        await pilot.press("space")
        assert svc.config.notify.events == NOTIFY_EVENTS
        _ended(svc, "2332863", JobState.COMPLETED)
        await pilot.pause(0.3)
        assert app.alerts.history[-1].category == "ended_ok"                     # "all" applies to every job


async def test_custom_events_are_shown_and_kept(make_app: MakeApp, service: FakeService) -> None:
    import dataclasses

    from runfleet.config import NotifySettings

    service.config = dataclasses.replace(service.config, notify=NotifySettings(
        events=("started", "ended_bad"), command=("notify-send", "RunFleet")))
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot, "tab")
        text = panel_text(app)
        assert "custom 2 events" in text and "command hook: set in config.toml" in text
        screen.selected["general"] = _row("enabled")
        await pilot.press("space", "down", "space")                              # other notify rows: events kept
        assert [c[0] for c in svc.called("set_notify")] == [{"enabled": False}, {"terminal": "osc9"}]
        assert svc.config.notify.events == ("started", "ended_bad")
        await pilot.press("down", "space")                                       # changing the row picks a preset
        assert svc.config.notify.events == FAILURES and "custom" not in panel_text(app)


async def test_a_notify_save_that_fails_is_a_message(make_app: MakeApp, service: FakeService) -> None:
    service.write_error = "config.toml: cannot write: Read-only file system"
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot, "tab")
        screen.selected["general"] = _row("enabled")
        await pilot.press("space")
        assert "cannot write: Read-only file system" in panel_text(app) and svc.config.notify.enabled


async def test_notification_rows_scroll_into_a_short_panel(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(80, 16)) as pilot:
        await open_settings(pilot, "tab", *["down"] * (len(GENERAL) - 1))
        text = panel_text(app)
        assert "▸ events" in text and f"{len(GENERAL)}/{len(GENERAL)}" in text and "default view" not in text


async def test_notify_settings_persist_on_the_real_service(tmp_path: Path) -> None:
    """The real FleetService with a file: the [notify] rows write the file (above the hosts, comments kept) and the
    running Alerts use the new values at once."""
    from runfleet.config import load_config
    from runfleet.core.clock import ManualClock
    from runfleet.core.service import FleetService
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld
    from tests.ui.fakes import T0

    path = tmp_path / "config.toml"
    path.write_text('[general]\nmouse = true   # keep\n\n# the main cluster\n[[host]]\nname = "k2"\n')
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.05)
    svc = FleetService(world.fleet_config(["k2"]), lambda h: DemoTransport(h, world), clock, None)
    svc.lock_path = False
    svc.config_path = path
    svc.alias_source = lambda: ["k2"]
    app = RunFleetApp(svc, clock=clock, animations=False, persist_state=False)
    try:
        async with app.run_test(size=BIG) as pilot:
            screen = await open_settings(pilot, "tab")
            screen.selected["general"] = _row("enabled")
            await pilot.press("space", "down", "down", "space")                  # enabled off, events +gpu idle
            assert not app.alerts.settings.enabled and "gpu_idle" in app.alerts.settings.events
            assert "this run only" not in panel_text(app)
        assert path.read_text() == ('[general]\nmouse = true   # keep\n\n[notify]\nenabled = false\n'
                                    'events = ["ended_bad", "restarted", "time_low", "gpu_idle"]\n\n'
                                    '# the main cluster\n[[host]]\nname = "k2"\n')
        notify = load_config(path).notify
        assert not notify.enabled and notify.events == ("ended_bad", "restarted", "time_low", "gpu_idle")
    finally:
        await svc.stop()


async def test_about_opens_the_config_in_the_editor(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path) -> None:
    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    monkeypatch.setenv("VISUAL", "nvim -u NONE")
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot, "e")
        assert not runs and "no file to edit" in panel_text(app)          # demo / in memory
        svc.config_path = tmp_path / "config.toml"
        await pilot.press("tab", "tab", "enter")
        assert runs == [["nvim", "-u", "NONE", str(tmp_path / "config.toml")]]
        assert "edited" in panel_text(app)


async def test_removed_host_closes_its_detail_screen_and_scope(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.set_scope("lyra")
        app.action_open_host("lyra")
        await pilot.pause(0.2)
        assert isinstance(app.screen, HostDetailScreen)
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("lyra")
        await pilot.press("space")
        await pilot.pause(0.5)
        assert isinstance(app.screen, SettingsScreen) and app.scope is None
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert isinstance(app.screen, MainScreen)


async def test_settings_fit_small_terminals(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(60, 16)) as pilot:
        await open_settings(pilot)
        box = settings(app).query_one("#settings-box")
        assert box.outer_size.width == 56 and box.outer_size.height == 14
        text = panel_text(app)
        assert "type / to filter" in text and "1/8" in text and "esc close" in text


# ---------------------------------------------------------------- helpers
def test_helpers() -> None:
    assert host_name_for("me@box:22") == "me_box_22" and host_name_for("k2") == "k2"
    assert editor_argv(Path("/c.toml"))[-1] == "/c.toml"
    ok = DetectReport("k2", ConnState.OK, None, None, "user", {}, 0.21)
    assert probe_summary(ok) == "ok · 0.21s · mux user"
    auth = DetectReport("lynx", ConnState.NEEDS_AUTH, None, None, "none", {}, None)
    assert "runfleet connect lynx" in probe_summary(auth)


async def test_help_inside_settings_lists_the_settings_keys(make_app: MakeApp) -> None:
    from runfleet.ui.screens.help import HelpScreen

    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_settings(pilot)
        await pilot.press("question_mark")
        await pilot.pause(0.1)
        assert isinstance(app.screen, HelpScreen) and app.screen.contexts == ["settings", "global"]


# ---------------------------------------------------------------- review regressions
async def test_keys_in_one_burst_after_slash_or_a_are_typed_not_run(make_app: MakeApp,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Over a slow ssh link `/vega` arrives in one read: every Key is queued before the first is handled. The keys
    after `/` (or `a`) must land in the input line, never run as Settings commands (j/k move, e edits, space
    removes)."""
    from textual import events

    app, svc = make_app()
    runs: list[list[str]] = []
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: runs.append(argv) or 0)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)

        def burst(text: str) -> None:
            names = {"/": "slash", " ": "space", "@": "at", ".": "full_stop"}
            for ch in text:                             # posted without awaiting: one terminal read
                app.post_message(events.Key(names.get(ch, ch), ch))

        burst("/vega")
        await pilot.pause(0.2)
        assert screen.editing == "filter" and screen.buffer == "vega" and row_names(app) == ["vega"]
        await pilot.press("escape")
        burst("ate k@j.e")                              # a + "te k@j.e": t would probe, e would open $EDITOR
        await pilot.pause(0.2)
        assert screen.editing == "add" and screen.buffer == "te k@j.e"
        assert not runs and not svc.called("probe_alias") and not svc.called("remove_host")
        assert screen.selected["hosts"] == 0


async def test_a_failed_save_is_a_message_never_a_crash_or_a_stuck_line(make_app: MakeApp,
                                                                        service: FakeService) -> None:
    service.write_error = "config.toml: cannot write: Read-only file system"
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("lyra")
        await pilot.press("space")                      # remove (worker): the error replaces "removing lyra …"
        await pilot.pause(0.3)
        assert "cannot write: Read-only file system" in panel_text(app) and "removing" not in panel_text(app)
        assert "lyra" in svc.store.hosts


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
async def test_an_unwritable_config_on_the_real_service_shows_the_error(tmp_path: Path) -> None:
    """The real FleetService (demo world, file-backed config) in a read-only directory: checking a host reports
    'cannot write' in the panel and the app keeps running (it used to exit with PermissionError)."""
    from runfleet.core.clock import ManualClock
    from runfleet.core.service import FleetService
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld
    from tests.ui.fakes import T0

    ro = tmp_path / "dotfiles"
    ro.mkdir()
    path = ro / "config.toml"
    path.write_text('[[host]]\nname = "k2"\n')
    ro.chmod(0o555)
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.05)
    svc = FleetService(world.fleet_config(["k2"]), lambda h: DemoTransport(h, world), clock, None)
    svc.lock_path = False
    svc.config_path = path
    svc.alias_source = lambda: ["k2", "vega"]
    app = RunFleetApp(svc, clock=clock, animations=False, persist_state=False)
    try:
        async with app.run_test(size=BIG) as pilot:
            screen = await open_settings(pilot)
            screen.selected["hosts"] = row_names(app).index("vega")
            await pilot.press("space")
            await pilot.pause(0.2)
            assert app.is_running and isinstance(app.screen, SettingsScreen)
            assert "cannot write: Permission denied" in panel_text(app) and "vega" not in svc.store.hosts
            screen.selected["hosts"] = row_names(app).index("k2")
            await pilot.press("space", "space")         # it runs jobs of mine: confirm, then the remove worker
            await pilot.pause(0.3)
            assert "cannot write" in panel_text(app) and "removing" not in panel_text(app)
            assert "k2" in svc.store.hosts
        assert app.return_code in (None, 0)
    finally:
        ro.chmod(0o755)
        await svc.stop()


async def test_removing_a_tuned_or_absent_host_needs_a_second_press(make_app: MakeApp,
                                                                    service: FakeService) -> None:
    from runfleet.config import HostConfig

    service.saved.append(HostConfig(name="edge1", ssh="edge1"))      # saved, but not in this run
    service.host_keys["lyra"] = ["group", "connect_timeout", "log_templates", "protected_qos"]
    app, svc = make_app(service)
    async with app.run_test(size=BIG) as pilot:
        screen = await open_settings(pilot)
        screen.selected["hosts"] = row_names(app).index("lyra")
        await pilot.press("space")
        await pilot.pause()
        assert not svc.called("remove_host")
        assert "has its own settings (group, connect_timeout, log_templates, …)" in panel_text(app)
        await pilot.press("space")
        await pilot.pause(0.3)
        assert svc.called("remove_host") == [("lyra", True)]
        screen.selected["hosts"] = row_names(app).index("edge1")
        await pilot.press("space")
        await pilot.pause()
        assert "not in this run" in panel_text(app) and len(svc.called("remove_host")) == 1
        await pilot.press("escape")                                      # esc drops the confirm, panel stays
        assert isinstance(app.screen, SettingsScreen) and screen.confirm is None


async def test_refresh_speed_shows_the_live_rate(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("plus")                      # ×0.5 for this run (not saved)
        assert app.multiplier == 0.5
        screen = await open_settings(pilot, "tab", "down")
        assert GENERAL[1].key == "refresh_multiplier" and svc.config.general.refresh_multiplier == 1.0
        assert screen._current(GENERAL[1]) == 0.5                     # the row highlights `fast`, as the header
        await pilot.press("space")                     # next value after fast (0.5) is slow (2.0): a visible change
        assert svc.called("set_general")[-1] == ({"refresh_multiplier": 2.0}, True) and app.multiplier == 2.0


async def test_the_base_hint_bar_hides_under_the_panel(make_app: MakeApp) -> None:
    from runfleet.ui.widgets.hintbar import HintBar

    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        main = app.screen
        bar = main.query_one(HintBar)
        assert bar.visible
        await open_settings(pilot)
        assert not bar.visible
        await pilot.press("question_mark")             # another screen on top: Settings no longer owns the bottom
        await pilot.pause(0.1)
        assert bar.visible
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not bar.visible
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert app.screen is main and bar.visible


def test_spread_keeps_a_gap_between_a_long_value_and_its_note() -> None:
    from rich.text import Text

    from runfleet.ui.screens.settings import _spread

    line = _spread(Text("   config file   /tmp/rf-set/config/runfleet/config.toml"), Text("↵ edit in $EDITOR "), 72)
    assert line.cell_len == 72 and "… ↵ edit" in line.plain
    assert _spread(Text("abc"), Text("de"), 5).plain == "a… de" and _spread(Text("abc"), Text("de"), 6).plain == "abc de"


async def test_about_theme_row_shows_the_mode_runfleet_theme_forces(make_app: MakeApp,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The row showed the stored choice ("light") while RUNFLEET_THEME kept the palette dark."""
    from runfleet.ui import theme

    app, _svc = make_app()
    try:
        async with app.run_test(size=BIG) as pilot:
            theme.configure("dark", "#1e1e2e", source="terminal background")
            monkeypatch.setenv("RUNFLEET_THEME", "dark")
            app.set_theme_choice("light")
            screen = await open_settings(pilot)
            screen.section = "about"
            screen.selected["about"] = 0
            screen.redraw()
            await pilot.pause()
            row = next(line for line in panel_text(app).splitlines() if line.lstrip(" ▸").startswith("theme "))
            assert "light → dark" in row and "RUNFLEET_THEME" in row and "#1e1e2e" not in row
            monkeypatch.delenv("RUNFLEET_THEME")
            app.set_theme_choice("light")
            screen.redraw()
            await pilot.pause()
            row = next(line for line in panel_text(app).splitlines() if line.lstrip(" ▸").startswith("theme "))
            assert "light → " not in row and "light" in row and "RUNFLEET_THEME" not in row
    finally:
        theme.configure("dark")
