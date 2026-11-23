"""The header ``F n/24h`` chip picks the recent window (app.recent_window, ui.json ``recent_window``): a click cycles
1h · 6h · 24h · 3d, every view follows, config.toml keeps general.recent_window, and the core keeps ended
workstation tasks for at least config.ENDED_RETAIN_S whatever the window."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from datetime import timedelta

from runfleet.config import ENDED_RETAIN_S, GeneralSettings, ended_retain_s
from runfleet.models import StateGroup
from runfleet.ui.app import RunFleetApp
from runfleet.ui.query import RECENT_WINDOWS, ended_at, preset_matches
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.jobs_view import JobsView
from tests.ui.fakes import FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
FLUSH = 0.3


def _header(app: RunFleetApp) -> str:
    return app.screen.query_one(FleetHeader).text.plain


def _f_total(app: RunFleetApp) -> tuple[int, str]:
    m = re.search(r"F (\d+)/(\S+)", _header(app))
    assert m is not None, _header(app)
    return int(m.group(1)), m.group(2)


def _expected(app: RunFleetApp, window: float) -> int:
    now = app.now()
    return sum(1 for j in app.store.jobs(app.scope_hosts()) if j.is_mine
               and j.state.group in (StateGroup.FAILED, StateGroup.ENDED)
               and ((end := ended_at(j)) is None or (now - end).total_seconds() <= window))


def test_retention_never_below_three_days() -> None:
    assert ended_retain_s(GeneralSettings(recent_window=3600)) == ENDED_RETAIN_S
    assert ended_retain_s(GeneralSettings(recent_window=7 * 86400)) == 7 * 86400
    assert max(RECENT_WINDOWS) <= ENDED_RETAIN_S          # the widest pick never outgrows what the core keeps


async def test_default_window_is_the_config_one(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(FLUSH)
        assert app.recent_window == svc.config.general.recent_window == 86400
        assert _f_total(app) == (_expected(app, 86400), "24h")


async def test_cycle_order_wraps_and_saves(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(FLUSH)
        seen = []
        for _ in range(len(RECENT_WINDOWS)):
            app.action_recent_window()
            await pilot.pause()
            seen.append(app.recent_window)
            assert app.ui_state["recent_window"] == app.recent_window
            count, label = _f_total(app)
            assert count == _expected(app, app.recent_window)
        assert seen == [3 * 86400, 3600, 6 * 3600, 86400]       # from 24h: 3d, then back round to the shortest
        assert svc.config.general.recent_window == 86400       # config.toml / the agent keep theirs
        app.action_recent_window(-1)
        await pilot.pause()
        assert app.recent_window == 6 * 3600


async def test_click_on_the_f_chip_cycles(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(FLUSH)
        text = _header(app)
        col = text.index("F ", text.index("PD"))
        await pilot.click(FleetHeader, offset=(col + 1, 0))
        await pilot.pause()
        assert app.recent_window == 3 * 86400
        assert _f_total(app)[1] == "3d"


async def test_shorter_window_hides_older_ends_everywhere(make_app: MakeApp) -> None:
    """A window shorter than a job's age drops it from the header count and the Jobs *fresh* list."""
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(FLUSH)
        now = app.now()
        ended = [j for j in app.store.jobs() if j.is_mine and j.state.is_terminal and ended_at(j) is not None]
        assert ended, "the fake fleet has ended jobs of mine"
        oldest = max((now - ended_at(j)).total_seconds() for j in ended if ended_at(j) is not None)
        app.set_recent_window(max(1.0, oldest / 2))
        await pilot.pause()
        hidden = [j for j in ended if not preset_matches("fresh", j, now, app.recent_window)]
        assert hidden
        assert _f_total(app)[0] == _expected(app, app.recent_window)
        await pilot.press("tab")
        await pilot.pause(FLUSH)
        view = app.screen.query_one(JobsView)
        assert view.preset == "fresh"
        shown = {j.key for j in view.visible_jobs()[0]}
        assert not any(j.key in shown for j in hidden)


async def test_saved_window_comes_back_and_odd_config_value_joins(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.config = dataclasses.replace(svc.config, general=dataclasses.replace(svc.config.general,
                                                                              recent_window=12 * 3600))
    app.ui_state["recent_window"] = 6 * 3600
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause(FLUSH)
        assert app.recent_window == 6 * 3600 and _f_total(app)[1] == "6h"
        assert 12 * 3600 in app.recent_window_choices()
        app.action_recent_window()
        await pilot.pause()
        assert app.recent_window == 12 * 3600
        app.ui_state["recent_window"] = "junk"                   # a hand-edited ui.json: the config value
        assert app.recent_window == 12 * 3600
    assert timedelta(seconds=app.recent_window) == timedelta(hours=12)
