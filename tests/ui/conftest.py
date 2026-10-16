"""[WP3] UI test fixtures: deterministic app (FakeService + ManualClock, no animations, UTC, no ui.json)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import runfleet.ui.app as app_module
from runfleet.core.clock import ManualClock
from runfleet.ui.app import RunFleetApp
from runfleet.ui.widgets.jobs_view import JobsView
from tests.ui.fakes import T0, FakeService


@pytest.fixture(autouse=True)
def _deterministic_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[None]:
    """UTC display times, no persisted UI state, immediate log peeks, colour on; state / config dirs in tmp (a crash
    report of a failing UI test must not land in the developer's real ~/.local/state/runfleet)."""
    for var in ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    # Windows: platformdirs reads these overrides (an absolute path with a drive) instead of the known folders
    for var in ("WIN_PD_OVERRIDE_APPDATA", "WIN_PD_OVERRIDE_LOCAL_APPDATA"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.setenv("TZ", "UTC")      # Windows has no tzset: its runners already run in UTC
    monkeypatch.delenv("NO_COLOR", raising=False)
    # a truecolor terminal, whatever runs the tests: CI runners have no TERM, which Rich reads as 16 colours (and the
    # theme then rightly picks reverse cursor rows); RUNFLEET_THEME / TEXTUAL_COLOR_SYSTEM must not leak in either
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    for var in ("RUNFLEET_THEME", "TEXTUAL_COLOR_SYSTEM", "COLORFGBG"):
        monkeypatch.delenv(var, raising=False)
    _tzset()
    monkeypatch.setattr(app_module, "ui_state_path", lambda: None)
    monkeypatch.setattr(JobsView, "PEEK_DWELL", 0.0)
    yield
    _tzset()


def _tzset() -> None:
    tzset = getattr(time, "tzset", None)          # POSIX only
    if tzset is not None:
        tzset()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(T0)


@pytest.fixture
def service(clock: ManualClock) -> FakeService:
    return FakeService(clock)


@pytest.fixture
def make_app(clock: ManualClock) -> Callable[..., tuple[RunFleetApp, FakeService]]:
    def factory(service: FakeService | None = None, **kwargs: Any) -> tuple[RunFleetApp, FakeService]:
        svc = service or FakeService(clock, **kwargs)
        return RunFleetApp(svc, clock=svc.clock, animations=False), svc   # type: ignore[arg-type]
    return factory


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("TEXTUAL_ANIMATIONS", "none")
