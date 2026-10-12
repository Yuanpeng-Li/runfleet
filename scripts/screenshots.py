#!/usr/bin/env python3
"""Render README screenshots from the demo fleet (headless, no network).

    uv run python scripts/screenshots.py [--out docs/screenshots] [--size 120x34] [--settle 70]

Starts the real stack (FleetService + collectors + DemoTransport + RunFleetApp) in Textual's headless test driver,
lets the first polls and the first Slurm accounting round land, then saves fleet.svg, jobs.svg, gpus.svg,
job_detail.svg and attach.svg (live per-GPU readings of a job with one idle GPU).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ATTACH_JOB = "823954"       # demo vega job: GPU0 busy, GPU1 idle while holding memory


async def shoot(out: Path, size: tuple[int, int], settle: float) -> list[Path]:
    from runfleet.core.clock import RealClock
    from runfleet.core.service import FleetService
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld
    from runfleet.ui.app import RunFleetApp

    clock = RealClock()
    world = DemoWorld(seed=7, now=clock.now)        # real-ish latencies, so the header shows probe times
    for faults in world.faults.values():            # a healthy fleet for the pictures (the demo itself keeps them)
        faults.clear()
    service = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
    service.lock_path = False                       # like --demo: never the real instance locks
    service.alias_source = lambda: [c.name for c in world.host_configs()]   # like --demo: never ~/.ssh/config
    app = RunFleetApp(service, clock=clock, animations=False, persist_state=False)   # never the real ui.json
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    cols, rows = size

    def save(name: str) -> None:
        app.save_screenshot(str(out / name))
        saved.append(out / name)

    async with app.run_test(size=size) as pilot:
        await asyncio.sleep(settle)                  # squeue/sacct, then the sstat round for running jobs
        await pilot.resize_terminal(cols, 27)        # the fleet cards fit; no empty half screen
        await asyncio.sleep(0.5)
        save("fleet.svg")
        await pilot.resize_terminal(cols, rows)
        await pilot.press("tab")
        await asyncio.sleep(1.5)                     # preview log peek after the cursor dwell
        save("jobs.svg")
        await pilot.press("tab")
        await asyncio.sleep(0.5)
        save("gpus.svg")
        await pilot.press("shift+tab")
        await asyncio.sleep(0.3)
        await pilot.press("enter")
        await asyncio.sleep(1.5)
        save("job_detail.svg")
        await pilot.press("escape")
        await asyncio.sleep(0.3)
        await pilot.press("slash")
        await pilot.press(*ATTACH_JOB)
        await asyncio.sleep(0.4)                     # filter debounce
        await pilot.press("enter")                   # back to the table, filter kept
        await pilot.press("enter")                   # open the job
        await asyncio.sleep(0.5)
        await pilot.press("right_square_bracket")    # GPU tab: starts the live readings
        await asyncio.sleep(20)                      # two samples, so the idle-GPU hint shows
        save("attach.svg")
        await pilot.press("ctrl+c")
    return saved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=REPO / "docs" / "screenshots")
    p.add_argument("--size", default="120x34", help="terminal size COLSxROWS (default 120x34)")
    p.add_argument("--settle", type=float, default=70.0,
                   help="seconds to let the first polls and the first sstat round land")
    args = p.parse_args()
    cols, rows = (int(x) for x in args.size.lower().split("x"))
    for path in asyncio.run(shoot(args.out, (cols, rows), args.settle)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
