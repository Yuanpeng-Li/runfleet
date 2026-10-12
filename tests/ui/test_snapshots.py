"""[WP3] SVG snapshots (DESIGN.md §10.5): the main screens at 120×35, fleet, jobs, job detail and settings at 80×24.

Accept intended changes with ``uv run pytest tests/ui/test_snapshots.py --snapshot-update`` and review the SVG diff.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from runfleet.core.clock import ManualClock
from runfleet.models import JobKey
from runfleet.ui.app import RunFleetApp
from tests.ui.fakes import T0, FakeService

BIG = (120, 35)
SMALL = (80, 24)


def _app() -> RunFleetApp:
    svc = FakeService(ManualClock(T0))
    return RunFleetApp(svc, clock=svc.clock, animations=False)  # type: ignore[arg-type]


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


async def _failed_steps(pilot: Any) -> None:
    await pilot.pause(0.1)
    pilot.app.open_job(JobKey("k2", "2330539"))
    await pilot.pause(0.3)
    await pilot.press("right_square_bracket", "right_square_bracket", "right_square_bracket")
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
    ("jobs_80x24", SMALL, keys("tab")),
    ("job_detail_80x24", SMALL, call(lambda app: app.open_job(JobKey("k2", "2332861")))),
    ("settings_80x24", SMALL, keys("comma", "down", "down", "down", "down", "down")),
)


@pytest.mark.parametrize(("name", "size", "run_before"), CASES, ids=[c[0] for c in CASES])
def test_snapshot(snap_compare: Any, name: str, size: tuple[int, int],
                  run_before: Callable[[Any], Awaitable[None]]) -> None:
    assert snap_compare(_app(), terminal_size=size, run_before=run_before)
