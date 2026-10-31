"""[WP3] SVG snapshots (DESIGN.md §10.5): the main screens at 120×35, fleet, jobs, job detail and settings at 80×24, the
Fleet view at 100×30 (compact cards), 120×40 and 172×62 (roomy cards, §7.4).

Accept intended changes with ``uv run pytest tests/ui/test_snapshots.py --snapshot-update`` and review the SVG diff.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from runfleet.core.clock import ManualClock
from runfleet.models import JobKey
from runfleet.ui.app import RunFleetApp
from runfleet.util import platform
from tests.ui.fakes import T0, FakeService, account_report


@pytest.fixture(autouse=True)
def _posix_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests pin the POSIX behaviour (pure UI/CLI logic, valid on any OS); the Windows branches are tested in
    test_windows_cli.py / test_windows_ui.py / tests/unit/test_platform.py."""
    monkeypatch.setattr(platform, "IS_WINDOWS", False)


BIG = (120, 35)
SMALL = (80, 24)
ROOMY = (172, 62)


def _app() -> RunFleetApp:
    svc = FakeService(ManualClock(T0))
    return RunFleetApp(svc, clock=svc.clock, animations=False)  # type: ignore[arg-type]


def _app_cost_columns() -> RunFleetApp:
    """The Jobs view with the optional SU / GPU-h columns on (Account, §7.5)."""
    svc = FakeService(ManualClock(T0))
    ui = svc.config.ui
    svc.config = dataclasses.replace(svc.config, ui=dataclasses.replace(
        ui, jobs_columns=(*ui.jobs_columns, "su", "gpuh")))
    return RunFleetApp(svc, clock=svc.clock, animations=False)  # type: ignore[arg-type]


def _app_k2_account() -> RunFleetApp:
    """k2 with its periodic account summary read (usage only): the host detail's `account` line (§7.17)."""
    svc = FakeService(ManualClock(T0))
    k2 = svc.store.hosts["k2"]
    report = account_report("k2", "30d", T0)
    assert k2.snapshot is not None and report is not None
    k2.snapshot = dataclasses.replace(k2.snapshot, account=report.summary)
    return RunFleetApp(svc, clock=svc.clock, animations=False)  # type: ignore[arg-type]


APPS: dict[str, Callable[[], RunFleetApp]] = {"jobs_su": _app_cost_columns, "host_cluster_account": _app_k2_account}


def keys(*presses: str, settle: float = 0.4) -> Callable[[Any], Awaitable[None]]:
    async def run(pilot: Any) -> None:
        await pilot.pause(0.1)
        for k in presses:
            await pilot.press(k)
            await pilot.pause(0.05)
        await pilot.pause(settle)
    return run


def call(fn: Callable[[RunFleetApp], None], settle: float = 0.4) -> Callable[[Any], Awaitable[None]]:
    async def run(pilot: Any) -> None:
        await pilot.pause(0.1)
        fn(pilot.app)
        await pilot.pause(settle)
    return run


async def _vega_needs_auth(pilot: Any) -> None:
    """vega asks for a login while its cached data stays: the error lines take the login slot and one list row."""
    await pilot.pause(0.1)
    pilot.app.service.inject("vega", "auth")
    await pilot.pause(0.4)


async def _failed_steps(pilot: Any) -> None:
    await pilot.pause(0.1)
    pilot.app.open_job(JobKey("k2", "2330539"))
    await pilot.pause(0.3)
    await pilot.press("right_square_bracket", "right_square_bracket", "right_square_bracket")
    await pilot.pause(0.3)


async def _sorted_by_su(pilot: Any) -> None:
    """Palette › Sort by SU, then the top row (the cursor otherwise keeps its screen row)."""
    await pilot.pause(0.1)
    pilot.app.jobs_setting("sort", "su")
    await pilot.pause(0.2)
    await pilot.press("g")
    await pilot.pause(0.3)


async def _cost_gpu_tab(pilot: Any) -> None:
    await pilot.pause(0.1)
    pilot.app.open_job(JobKey("k2", "2332861"))
    await pilot.pause(0.3)
    await pilot.press("right_square_bracket")
    await pilot.pause(0.3)


CASES: Sequence[tuple[str, tuple[int, int], Callable[[Any], Awaitable[None]]]] = (
    ("fleet", BIG, keys()),
    ("jobs", BIG, keys("tab")),
    ("gpus", BIG, keys("tab", "tab")),
    ("job_detail", BIG, call(lambda app: app.open_job(JobKey("k2", "2332861")))),
    ("job_detail_failed_steps", BIG, _failed_steps),
    ("host_ws", BIG, call(lambda app: app.action_open_host("labstation"))),
    ("host_cluster", BIG, call(lambda app: app.action_open_host("k2"))),
    ("log_viewer", BIG, keys("tab", "l")),
    ("help", BIG, keys("tab", "question_mark")),
    ("confirm", BIG, keys("tab", "ctrl+k")),
    ("settings", BIG, keys("comma")),
    ("settings_general", BIG, keys("comma", "tab", "down")),
    ("fleet_80x24", SMALL, keys()),
    ("fleet_100x30", (100, 30), keys()),
    ("fleet_120x40", (120, 40), keys()),
    ("fleet_172x62", ROOMY, keys()),
    ("fleet_172x62_auth", ROOMY, _vega_needs_auth),
    ("jobs_80x24", SMALL, keys("tab")),
    ("job_detail_80x24", SMALL, call(lambda app: app.open_job(JobKey("k2", "2332861")))),
    # Account (§7.17): SU / GPU-h columns sorted by SU; the k2 job's Cost box on the GPU tab (mockup P)
    ("jobs_su", BIG, _sorted_by_su),
    ("job_detail_cost_gpu_tab", BIG, _cost_gpu_tab),
    # the Account screen (mockups O / O2) and the cluster host detail's `account` line
    ("account", BIG, keys("dollar_sign", settle=0.6)),
    ("account_80x24", SMALL, keys("dollar_sign", settle=0.6)),
    ("account_7d", BIG, keys("dollar_sign", "left_square_bracket", settle=0.6)),
    ("account_rollup_task", BIG, keys("dollar_sign", "z", settle=0.6)),
    ("account_scope_k2", BIG, keys("dollar_sign", "1", settle=0.6)),
    ("host_cluster_account", BIG, call(lambda app: app.action_open_host("k2"))),
    ("settings_80x24", SMALL, keys("comma", "down", "down", "down", "down", "down")),
)


@pytest.mark.parametrize(("name", "size", "run_before"), CASES, ids=[c[0] for c in CASES])
def test_snapshot(snap_compare: Any, name: str, size: tuple[int, int],
                  run_before: Callable[[Any], Awaitable[None]]) -> None:
    assert snap_compare(APPS.get(name, _app)(), terminal_size=size, run_before=run_before)
