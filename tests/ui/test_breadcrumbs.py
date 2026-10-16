"""[WP3] Clickable header breadcrumb on detail screens: the host name, `Jobs` / `Hosts` and a log's job go back to
that place (the same screen when it is further down the stack), the last part (this screen) is plain text."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from runfleet.models import JobKey
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.widgets.header import FleetHeader

BIG = (140, 40)
KEY = JobKey("k2", "2332861")


def clicks(header: FleetHeader) -> dict[str, str]:
    """Visible text of each clickable span of the header line -> its action."""
    text: Text = header.build(BIG[0])
    out: dict[str, str] = {}
    for span in text.spans:
        meta = getattr(span.style, "meta", None) or {}
        action = meta.get("@click")
        if action:
            out[text.plain[span.start:span.end].strip("▏▕ ")] = action
    return out


async def _settle(pilot: Any) -> None:
    await pilot.pause(0.2)


async def test_job_detail_breadcrumb_parts(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.open_job(KEY)
        await _settle(pilot)
        actions = clicks(app.screen.query_one(FleetHeader))
        assert actions["k2"] == "app.crumb('host', 'k2')"
        assert actions["Jobs"] == "app.crumb('view', 'jobs')"
        assert KEY.label not in actions and "2332861" not in actions          # the current screen: plain text
        assert actions["RunFleet"] == "app.home"


async def test_host_crumb_returns_to_the_host_detail_it_came_from(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.action_open_host("k2")
        await _settle(pilot)
        host_screen = app.screen
        assert isinstance(host_screen, HostDetailScreen)
        app.open_job(KEY)
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen)
        app.action_crumb("host", "k2")
        await _settle(pilot)
        assert app.screen is host_screen                                      # the same screen, as it was


async def test_host_crumb_opens_the_host_detail_when_not_on_the_way(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.open_job(KEY)                                                     # opened from the main screen
        await _settle(pilot)
        app.action_crumb("host", "k2")
        await _settle(pilot)
        assert isinstance(app.screen, HostDetailScreen) and app.screen.host == "k2"
        assert [type(s).__name__ for s in app.screen_stack[1:]] == ["MainScreen", "HostDetailScreen"]


async def test_jobs_crumb_shows_the_jobs_view(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.action_open_host("k2")
        await _settle(pilot)
        app.open_job(KEY)
        await _settle(pilot)
        app.action_crumb("view", "jobs")
        await _settle(pilot)
        assert isinstance(app.screen, MainScreen) and app.screen.view == "jobs"


async def test_log_viewer_job_crumb_goes_back_to_that_job(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.open_job(KEY)
        await _settle(pilot)
        detail = app.screen
        app.push_screen(LogScreen(KEY))
        await _settle(pilot)
        actions = clicks(app.screen.query_one(FleetHeader))
        assert actions[KEY.label] == "app.crumb('job', 'k2', '2332861')" and "stdout" not in actions
        app.action_crumb("job", "k2", "2332861")
        await _settle(pilot)
        assert app.screen is detail
        # opened straight from the Jobs view (no detail screen on the way): the crumb opens one
        app.go_main("jobs")
        await _settle(pilot)
        app.push_screen(LogScreen(KEY))
        await _settle(pilot)
        app.action_crumb("job", "k2", "2332861")
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == KEY


async def test_a_mouse_click_on_the_crumb_works(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.action_open_host("k2")
        await _settle(pilot)
        host_screen = app.screen
        app.open_job(KEY)
        await _settle(pilot)
        header = app.screen.query_one(FleetHeader)
        x = header.build(BIG[0]).plain.index(" Jobs") + 2
        await pilot.click(FleetHeader, offset=(x, 0))
        await _settle(pilot)
        assert isinstance(app.screen, MainScreen) and app.screen.view == "jobs"
        app.action_open_host("k2")
        await _settle(pilot)
        app.open_job(KEY)
        await _settle(pilot)
        header = app.screen.query_one(FleetHeader)
        x = header.build(BIG[0]).plain.index("k2") + 1
        await pilot.click(FleetHeader, offset=(x, 0))
        await _settle(pilot)
        assert isinstance(app.screen, HostDetailScreen) and app.screen is not host_screen   # a new one: popped above


async def test_unknown_crumb_targets_change_nothing(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await _settle(pilot)
        app.open_job(KEY)
        await _settle(pilot)
        detail = app.screen
        for args in (("bogus",), ("host",), ("job", "k2")):
            app.action_crumb(*args)
            await _settle(pilot)
            assert app.screen is detail
        assert isinstance(app, RunFleetApp)
