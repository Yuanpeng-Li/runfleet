"""[WP3] Updates in the TUI (§8.6): the start-up check and its one-time toast, Settings › About's version row (check,
arm, ↵ again quits with update.RESTART_CODE), palette › Update RunFleet, no network in --demo."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet import __version__, update
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.settings import SettingsPanel, SettingsScreen
from runfleet.util import platform
from tests.ui.fakes import FakeService


@pytest.fixture(autouse=True)
def _posix_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests pin the POSIX behaviour (pure UI/CLI logic, valid on any OS); the Windows branches are tested in
    test_windows_cli.py / test_windows_ui.py / tests/unit/test_platform.py."""
    monkeypatch.setattr(platform, "IS_WINDOWS", False)


BIG = (120, 35)
T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)


class Checks:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.latest = "9.0.0"
        self.calls: list[bool] = []
        monkeypatch.setattr(update, "state_file", lambda: tmp_path / "update.json")
        monkeypatch.setattr(update, "check", self._check)
        monkeypatch.setattr(update, "detect_install", lambda: update.Install(
            "uv", "git", (("uv", "tool", "upgrade", "runfleet"),), "uv tool upgrade runfleet"))

    def _check(self, now: datetime, *, force: bool = False) -> update.Status:
        self.calls.append(force)
        return update.Status(__version__, self.latest, T0)


@pytest.fixture
def checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Checks:
    return Checks(monkeypatch, tmp_path)


def app_for(clock: Any, *, check: bool = True, saved: bool = True) -> tuple[RunFleetApp, FakeService]:
    svc = FakeService(clock)
    if saved:
        svc.config_path = Path("/tmp/runfleet-test-config.toml")        # never written: no setting changes here
    return RunFleetApp(svc, clock=svc.clock, animations=False, check_updates=check), svc  # type: ignore[arg-type]


def panel(app: RunFleetApp) -> str:
    assert isinstance(app.screen, SettingsScreen), app.screen
    return str(app.screen.query_one(SettingsPanel).render())


async def settle(pilot: Any) -> None:
    await pilot.pause(0.1)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause(0.1)


async def test_start_up_check_announces_a_new_version_once(clock: Any, checks: Checks) -> None:
    app, _svc = app_for(clock)
    async with app.run_test(size=BIG) as pilot:
        await settle(pilot)
        assert checks.calls == [False] and app.update_status.latest == "9.0.0"
        assert any("RunFleet 9.0.0 is out" in str(n.message) for n in app._notifications)
    app2, _svc = app_for(clock)
    async with app2.run_test(size=BIG) as pilot:
        await settle(pilot)
        assert not any("is out" in str(n.message) for n in app2._notifications)     # announced already


async def test_no_check_when_off(clock: Any, checks: Checks) -> None:
    app, _svc = app_for(clock, check=False)
    async with app.run_test(size=BIG) as pilot:
        await settle(pilot)
        assert checks.calls == [] and app.update_status is None


async def test_about_version_row_checks_then_updates_on_a_second_enter(clock: Any, checks: Checks) -> None:
    app, _svc = app_for(clock, check=False)
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("comma", "tab", "tab")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.selected["about"] = 4                                          # version
        screen.redraw()
        assert "↵ check for updates" in panel(app) and "↵ check" in panel(app)
        checks.latest = __version__
        await pilot.press("enter")
        await settle(pilot)
        assert checks.calls == [True] and f"this is the newest version ({__version__})" in panel(app)
        checks.latest = "9.0.0"
        await pilot.press("enter")
        await settle(pilot)
        assert "RunFleet 9.0.0 is available" in panel(app) and "9.0.0 available · ↵ update" in panel(app)
        await pilot.press("enter")
        await pilot.pause()
        assert "↵ again: quit, update to 9.0.0 (uv tool upgrade runfleet)" in panel(app)
        assert "↵ update now" in panel(app)
        await pilot.press("escape")                                           # esc disarms, panel stays
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen) and "↵ again" not in panel(app)
        await pilot.press("enter", "enter")
        await pilot.pause()
    assert app.return_code == update.RESTART_CODE


async def test_palette_entry_lands_ready_to_update(clock: Any, checks: Checks) -> None:
    app, _svc = app_for(clock, check=False)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_update()
        await settle(pilot)
        assert checks.calls == [True]                                         # nothing known yet: asked first
        assert "↵ again: quit, update to 9.0.0" in panel(app)
        await pilot.press("enter")
        await pilot.pause()
    assert app.return_code == update.RESTART_CODE


async def test_demo_never_asks_github(clock: Any, checks: Checks) -> None:
    app, _svc = app_for(clock, check=False, saved=False)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_update()
        await settle(pilot)
        assert checks.calls == [] and "demo — no network" in panel(app)


async def test_windows_version_row_shows_the_manual_steps(clock: Any, checks: Checks,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows (P1): the running runfleet.exe cannot be replaced, so ↵ never arms an update that quits."""
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    app, _svc = app_for(clock, check=False)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.open_update()
        await settle(pilot)
        assert checks.calls == [True]
        text = panel(app)
        assert "9.0.0: quit RunFleet (q), then run uv tool upgrade runfleet" in text
        assert "how to update" in text and "↵ again" not in text
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen) and "↵ again" not in panel(app)
    assert app.return_code != update.RESTART_CODE
