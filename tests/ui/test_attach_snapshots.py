"""[WP3] SVG snapshots of Attach (§7.16.7): vega's live GPU tab (2nd sample: GPU1 idle), its Procs tab with STEP,
and lyra's srun confirmation.

Accept intended changes with ``uv run pytest tests/ui/test_attach_snapshots.py --snapshot-update`` and review the SVG.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from runfleet.core.clock import ManualClock
from runfleet.models import JobKey
from runfleet.ui.app import RunFleetApp
from tests.ui.fakes import LYRA_LIVE_KEY, T0, FakeService

BIG = (120, 35)


def _app() -> RunFleetApp:
    svc = FakeService(ManualClock(T0))
    svc.add_lyra_running_job()
    return RunFleetApp(svc, clock=svc.clock, animations=False)  # type: ignore[arg-type]


def open_then(key: JobKey, *presses: str) -> Callable[[Any], Awaitable[None]]:
    async def run(pilot: Any) -> None:
        await pilot.pause(0.1)
        pilot.app.open_job(key)
        await pilot.pause(0.3)
        for k in presses:
            await pilot.press(k)
            await pilot.pause(0.1)
        await pilot.pause(0.4)
    return run


CASES: Sequence[tuple[str, Callable[[Any], Awaitable[None]]]] = (
    ("gpu_live_vega", open_then(JobKey("vega", "723954"), "right_square_bracket", "r")),
    ("procs_live_vega", open_then(JobKey("vega", "723954"), *["right_square_bracket"] * 4)),
    ("srun_confirm_lyra", open_then(LYRA_LIVE_KEY, "right_square_bracket", "r")),
)


@pytest.mark.parametrize(("name", "run_before"), CASES, ids=[c[0] for c in CASES])
def test_attach_snapshot(snap_compare: Any, name: str, run_before: Callable[[Any], Awaitable[None]]) -> None:
    assert snap_compare(_app(), terminal_size=BIG, run_before=run_before)
