"""[WP4] Cross-job log search in the demo world (§4.7.1, §10.6): `--demo` answers `log.search` from the simulated
files (Slurm clusters and workstations), streams a hit's window from its absolute offset, the UI lands on the hit,
and no timer ever sends the search."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from runfleet.config import GeneralSettings
from runfleet.core.clock import ManualClock
from runfleet.core.logs import tail_spec
from runfleet.core.logsearch import LogQuery
from runfleet.core.poller import BackendHooks, HostLanes, HostPoller
from runfleet.core.service import FleetService
from runfleet.core.store import FleetStore
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.log_search import LogSearchScreen, SearchPromptScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.widgets.log_view import LogView
from tests.conftest import CLOCK_START
from tests.ui.fakes import T0
from tests.ui.test_log_search_ui import landed
from tests.unit.stubs import run_for


class _Recording(DemoTransport):
    """Every section name of every probe, whatever its purpose."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.sections: list[str] = []

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        self.sections += [s.name for s in script.sections]
        return await super().probe(script, timeout=timeout)


def _service() -> tuple[FleetService, DemoWorld, ManualClock, dict[str, _Recording]]:
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    transports: dict[str, _Recording] = {}

    def make(h: Any) -> _Recording:
        transports[h.name] = _Recording(h, world)
        return transports[h.name]

    svc = FleetService(world.fleet_config(), make, clock, None)
    svc.lock_path = False
    return svc, world, clock, transports


async def test_demo_search_round_trip_on_clusters_and_workstations() -> None:
    svc, world, _clock, transports = _service()
    await svc.collect_once()
    r = await svc.search_logs("k2", LogQuery("cuda OUT of memory"))
    assert r.error is None and len(r.files) == 15 and [h.key.id for h in r.hits] == ["3368404"]
    hit = r.hits[0]
    data = bytes(world.cluster("k2").logs[hit.file.path].read_from(0))
    assert data[hit.offset:].startswith(hit.excerpt.encode()) and hit.needle.startswith("torch.OutOfMemoryError")
    assert hit.needle in data[hit.start:].split(b"\n")[hit.line - 1].decode()
    assert "missing" in [f.status for f in r.files]                        # logs the demo never wrote: said so
    assert transports["k2"].sections.count("log.search") == 1             # one probe for the host's 15 files
    ws = await svc.search_logs("labstation", LogQuery(r"epoch [0-9]+/200", regex=True), max_matches=3)
    assert len(ws.hits) == 3 and ws.match_capped and ws.hits[0].key.id.startswith("p")
    bad = await svc.search_logs("vega", LogQuery("loss=[0", regex=True))
    assert (bad.error or "").startswith("grep: ") and bad.hits == ()


async def test_demo_stream_starts_at_the_hit_window() -> None:
    svc, world, _clock, _t = _service()
    await svc.collect_once()
    hit = (await svc.search_logs("k2", LogQuery("cuda out of memory"))).hits[0]
    lg = world.cluster("k2").logs[hit.file.path]
    t = DemoTransport(world.fleet_config().hosts[0], world)
    handle = await t.stream(tail_spec(hit.file.path, 65536, 1000))
    chunks = handle.chunks()
    first = await chunks.__anext__()
    await handle.close()
    assert first == lg.read_from(1000)[:len(first)] and len(first) == min(65536, lg.size - 1000)


async def test_no_timer_ever_sends_a_search() -> None:
    """An hour of polling on a cluster and a workstation: not one `log.search` section."""
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    tasks, transports = [], []
    for host in ("k2", "labstation"):
        cfg = world.configs[host]
        t = _Recording(cfg, world)
        transports.append(t)
        p = HostPoller(cfg, t, FleetStore([cfg]), clock, random.Random(1), HostLanes(asyncio.Semaphore(8), t.mux),
                       general=GeneralSettings(), hooks=BackendHooks.default())
        tasks.append(asyncio.ensure_future(p.run()))
    try:
        await run_for(clock, 3600, step=2.0)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert all(t.sections for t in transports)
    assert not any("log.search" in t.sections for t in transports)


async def test_demo_ui_finds_and_lands() -> None:
    svc, _world, clock, transports = _service()
    app = RunFleetApp(svc, clock=clock, animations=False, persist_state=False)
    async with app.run_test(size=(140, 40)) as pilot:
        for _ in range(60):
            await pilot.pause(0.05)
            if any(True for _ in svc.store.jobs(["k2"])):
                break
        await pilot.pause(0.3)
        assert "log.search" not in transports["k2"].sections            # opening the app searches nothing
        app.go_main("jobs")
        await pilot.pause()
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert isinstance(app.screen, SearchPromptScreen)
        await pilot.press(*"out of memory")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if isinstance(app.screen, LogSearchScreen) and not app.screen.pending:
                break
        results = app.screen
        assert isinstance(results, LogSearchScreen) and results.hits
        hit = next(iter(results.hits.values()))
        assert hit.key.host == "k2" and "lynx" not in [h for h, r in results.results.items() if r.hits]
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            screen = app.screen
            if isinstance(screen, LogScreen) and screen.hit is None:
                break
        view = app.screen.query_one(LogView)
        assert view.search_note is not None and view.search_note.startswith("search hit “out of memory”")
        assert "out of memory" in landed(view)
        assert transports["k2"].sections.count("log.search") == 1
