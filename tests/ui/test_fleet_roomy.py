"""Roomy Fleet view in the app (DESIGN.md §7.4): the wrap fix, the ui.json ``fleet_layout`` switch, the fleet panel,
and heights that stay put while the demo world runs for an hour."""

from __future__ import annotations

from typing import Any

import pytest

from runfleet.core.clock import ManualClock
from runfleet.ui import fleet_layout
from runfleet.ui.app import RunFleetApp
from runfleet.ui.widgets.fleet_panel import FleetPanel
from runfleet.ui.widgets.fleet_view import FleetView
from runfleet.ui.widgets.host_card import HostCard
from tests.ui.fakes import T0, FakeService


def _app(**kwargs: Any) -> RunFleetApp:
    svc = FakeService(ManualClock(T0))
    return RunFleetApp(svc, clock=svc.clock, animations=False, **kwargs)  # type: ignore[arg-type]


def _rows(app: RunFleetApp) -> list[list[HostCard]]:
    cards = list(app.screen.query(HostCard))
    cols = fleet_layout.columns_for(app.screen.query_one(FleetView).size.width)
    return [cards[i:i + cols] for i in range(0, len(cards), cols)]


async def test_cards_never_wrap_at_100x30() -> None:
    """The scrollbar gutter is reserved: lines are fitted to the real card width, rows keep one height."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.3)
        view = app.screen.query_one(FleetView)
        assert view.plan is not None and view.plan.tier == 0
        for row in _rows(app):
            tallest = max(len(c.lines) for c in row)
            for card in row:                          # a wrapped line would make one card taller than its lines
                assert card.size.height == tallest, card.host
                assert all(line.cell_len <= card.size.width for line in card.lines), card.host


async def test_fleet_layout_switch_is_live_and_saved() -> None:
    app = _app(persist_state=False)
    async with app.run_test(size=(172, 62)) as pilot:
        await pilot.pause(0.3)
        view = app.screen.query_one(FleetView)
        panel = app.screen.query_one(FleetPanel)
        assert view.plan is not None and view.plan.tier == 1 and panel.display
        app.set_fleet_layout("compact")
        await pilot.pause(0.3)
        assert app.ui_state["fleet_layout"] == "compact"
        assert view.plan.tier == 0 and not panel.display
        app.set_fleet_layout("auto")
        await pilot.pause(0.3)
        assert app.ui_state["fleet_layout"] == "auto" and view.plan.tier == 1 and panel.display


async def test_palette_has_the_fleet_layout_commands() -> None:
    from runfleet.ui.commands import RunFleetCommands

    app = _app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.1)
        provider = RunFleetCommands(app.screen)
        names = [name for name, _cb, _help in provider._commands()]
        assert "Fleet: roomy cards" in names and "Fleet: compact cards" in names


async def test_panel_is_not_a_card() -> None:
    """Deep review X: a click on the panel moved the focus off the selected card (to the scroll container), and the
    old check here (arrow right from the last card) could not fail."""
    app = _app()
    async with app.run_test(size=(172, 62)) as pilot:
        await pilot.pause(0.3)
        panel = app.screen.query_one(FleetPanel)
        view = app.screen.query_one(FleetView)
        assert panel.display and not panel.can_focus
        names = app.host_names()
        view.selected = names[1]
        view.focus_selection()
        await pilot.pause(0.1)
        await pilot.click(FleetPanel, offset=(10, 5))
        await pilot.pause(0.2)
        focused = app.screen.focused
        assert isinstance(focused, HostCard) and focused.host == names[1] and view.selected == names[1]


async def test_empty_fleet_message_follows_the_theme() -> None:
    """Deep review N: the empty-fleet message was built once and kept the old palette after a theme switch."""
    import dataclasses

    from textual.widgets import Static

    from runfleet.ui import theme
    from tests.ui.fakes import build_world

    cfg, _states = build_world()
    svc = FakeService(ManualClock(T0), world=(dataclasses.replace(cfg, hosts=()), []))
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    try:
        async with app.run_test(size=(120, 35)) as pilot:
            await pilot.pause(0.3)
            empty = app.screen.query_one("#fleet-empty", Static)
            dark = theme.muted()
            app.set_theme_choice("light")
            await pilot.pause(0.3)
            assert theme.muted() != dark
            styles = {str(span.style) for span in empty.content.spans}    # type: ignore[union-attr]
            assert theme.muted() in styles and dark not in styles
    finally:
        theme.configure("dark")


@pytest.mark.parametrize(("size", "panel"), [((80, 24), False), ((172, 62), True)])
async def test_help_legend_has_panel_glyphs_only_with_the_panel(size: tuple[int, int], panel: bool) -> None:
    """Review: `?` at 80×24 listed the fleet panel's glyphs although there is no panel in table mode."""
    from runfleet.ui.screens.help import HelpScreen

    app = _app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        await pilot.press("question_mark")
        await pilot.pause(0.3)
        assert isinstance(app.screen, HelpScreen)
        assert ("fleet_panel" in app.screen.legend_contexts) is panel
        body = str(app.screen.query_one("#help-body").render())
        assert ("what changed" in body) is panel


def _demo() -> tuple[RunFleetApp, Any, ManualClock]:
    try:
        from runfleet.core.service import FleetService
        from runfleet.demo.transport import DemoTransport
        from runfleet.demo.world import DemoWorld, demo_ui_state
    except ImportError as exc:      # pragma: no cover
        pytest.skip(f"demo stack not available: {exc!r}")
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.05)
    service = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
    service.lock_path = False
    app = RunFleetApp(service, clock=clock, animations=False, persist_state=False,
                      initial_state=demo_ui_state(clock.now()))
    return app, service, clock


@pytest.mark.parametrize("size", [(172, 62), (120, 40)])
async def test_heights_stay_put_for_an_hour(size: tuple[int, int]) -> None:
    """60 simulated minutes in 5 s steps on the demo stack: jobs start, end and fail, the heights never move and no
    line is wider than its card."""
    app, service, clock = _demo()
    async with app.run_test(size=size) as pilot:
        for _ in range(60):
            await pilot.pause(0.05)
            if any(True for _ in service.store.jobs()):
                break
        clock.advance(5)
        await pilot.pause(0.5)
        view = app.screen.query_one(FleetView)
        panel = app.screen.query_one(FleetPanel)
        first = view.plan
        assert first is not None and first.tier == 1 and first.panel is not None
        heights = [c.size.height for c in app.screen.query(HostCard)]
        panel_h = panel.size.height
        for step in range(720):
            clock.advance(5)
            await pilot.pause(0.01 if step % 12 else 0.05)
            if step % 12 == 11:                       # once a simulated minute: everything settled
                await pilot.pause(0.05)
                assert view.plan == first
                assert [c.size.height for c in app.screen.query(HostCard)] == heights
                assert panel.size.height == panel_h
                for card in app.screen.query(HostCard):
                    assert all(line.cell_len <= first.inner for line in card.lines), card.host
                assert all(line.cell_len <= panel.content_region.width + 1 for line in panel.lines)


# ---------------------------------------------------------------- opening entries from the Fleet view
def _card(app: RunFleetApp, host: str) -> HostCard:
    return next(c for c in app.screen.query(HostCard) if c.host == host)


async def test_click_on_a_card_job_row_opens_that_job() -> None:
    """A click on a list row opens the job it shows; a click on the summary still only selects the card."""
    from runfleet.ui.screens.job_detail import JobDetailScreen
    from runfleet.ui.screens.main import MainScreen

    app = _app(persist_state=False)
    async with app.run_test(size=(172, 62)) as pilot:
        await pilot.pause(0.3)
        card = _card(app, "vega")
        row = next(i for i, k in enumerate(card.row_keys) if k is not None)
        key = card.row_keys[row]
        assert key is not None and app.store.job(key) is not None
        await pilot.click(card, offset=(20, 1))                   # the first summary line: select only
        await pilot.pause(0.2)
        assert isinstance(app.screen, MainScreen) and app.screen.focused is card
        await pilot.click(card, offset=(20, row + 1))             # + 1: the top border
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == key


async def test_click_on_a_panel_activity_row_opens_that_job() -> None:
    from runfleet.ui.screens.job_detail import JobDetailScreen

    app = _app(persist_state=False)
    async with app.run_test(size=(172, 62)) as pilot:
        await pilot.pause(0.3)
        panel = app.screen.query_one(FleetPanel)
        row = next(i for i, k in enumerate(panel.row_keys) if k is not None)
        key = panel.row_keys[row]
        job = app.store.job(key) if key is not None else None
        assert job is not None and job.name[:12] in panel.lines[row].plain
        await pilot.click(FleetPanel, offset=(10, row + 1))
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == key


def _with_needs(app: RunFleetApp) -> None:
    """A fresh failure on vega and a k2 job low on time, on top of lyra's two older failures."""
    from datetime import timedelta

    from runfleet.models import JobState
    from tests.ui.fakes import slurm_job

    svc: Any = app.service
    svc.put_job(slurm_job("vega", "724999", "orb-eval--voc--s47--a01", JobState.FAILED,
                          elapsed=timedelta(hours=1), ended_ago=120, exit_code=1))
    svc.put_job(slurm_job("k2", "2334999", "orb-swp--train--cifar-short--s43--a02", JobState.RUNNING,
                          elapsed=timedelta(minutes=55), limit=timedelta(hours=1), node="n-40-01"))


@pytest.mark.parametrize("size", [(172, 62), (100, 30), (80, 24)])
async def test_needs_you_picker_in_every_layout(size: tuple[int, int]) -> None:
    """`a` lists what needs the user fleet-wide in rank order (roomy, compact cards and the narrow table alike);
    enter opens the highlighted job."""
    from textual.widgets import OptionList

    from runfleet.models import JobKey
    from runfleet.ui.screens.attention import AttentionScreen
    from runfleet.ui.screens.job_detail import JobDetailScreen

    app = _app(persist_state=False)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        _with_needs(app)
        await pilot.pause(0.3)
        await pilot.press("a")
        await pilot.pause(0.3)
        assert isinstance(app.screen, AttentionScreen)
        assert [str(e.key) for e in app.screen.entries] == [
            "vega:724999", "k2:2334999", "lyra:31354789", "lyra:31327454"]
        menu = app.screen.query_one(OptionList)
        first = str(menu.get_option_at_index(0).prompt)
        assert "vega" in first and "724999" in first and "orb-eval--voc--s47--a01" in first and "exit 1" in first
        second = str(menu.get_option_at_index(1).prompt)
        assert "orb-swp--train--cifar-short--s43--a02" in second and "low on time" in second
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == JobKey("k2", "2334999")


async def test_needs_you_picker_closes_and_warns_about_a_gone_job() -> None:
    from runfleet.models import JobKey
    from runfleet.ui.screens.attention import AttentionScreen
    from runfleet.ui.screens.main import MainScreen

    app = _app(persist_state=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        await pilot.press("a")
        await pilot.pause(0.3)
        assert isinstance(app.screen, AttentionScreen)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert isinstance(app.screen, MainScreen)
        await pilot.press("a")
        await pilot.pause(0.3)
        svc: Any = app.service
        svc.store.remove_job(JobKey("lyra", "31354789"))            # gone while the picker is open
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, MainScreen)
        assert any("no longer listed" in str(n.message) for n in app._notifications)


async def test_needs_you_list_fits_a_short_terminal() -> None:
    """Review fix: with many two-line entries at 80×24 the list stays inside the dialog (a list taller than the box was
    clipped, and the highlighted entry could sit unseen below it)."""
    from datetime import timedelta

    from textual.widgets import OptionList

    from runfleet.models import JobState
    from runfleet.ui.screens.attention import AttentionScreen
    from tests.ui.fakes import slurm_job

    app = _app(persist_state=False)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.3)
        svc: Any = app.service
        for i in range(20):
            svc.put_job(slurm_job("vega", f"7250{i:02d}", f"orb-eval--voc--s47--a{i:02d}--a-long-name-that-wraps",
                                  JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=120 + i, exit_code=1))
        await pilot.pause(0.3)
        await pilot.press("a")
        await pilot.pause(0.3)
        assert isinstance(app.screen, AttentionScreen)
        menu = app.screen.query_one(OptionList)
        box = app.screen.query_one("#attention-box")
        assert menu.option_count >= 20
        for _ in range(11):
            await pilot.press("j")
        await pilot.pause(0.2)
        assert menu.region.bottom <= box.region.bottom - 1             # inside the bottom border
        assert menu.max_scroll_y > 0 and menu.scroll_y > 0                # it scrolls to the highlighted entry
        await pilot.resize_terminal(80, 40)
        await pilot.pause(0.3)
        assert menu.region.height == 24                                  # a tall terminal: the usual 24 rows
