"""[WP3] §10.5 flows on the real stack (M2): DemoWorld -> DemoTransport -> collectors -> FleetService -> UI.

Same ManualClock discipline as the FakeService flows; time only moves when a test advances it. Skipped as a whole
while a lower layer is still a stub (WP1/WP2/WP4 land in parallel).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from runfleet.core.clock import ManualClock
from runfleet.models import ConnState, JobKey, JobState
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.confirm import ConfirmScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.host_card import HostCard
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.log_view import LogView
from tests.ui.fakes import T0

BIG = (120, 35)


def _stack() -> tuple[RunFleetApp, Any, Any, ManualClock]:
    try:
        from runfleet.core.service import FleetService
        from runfleet.demo.transport import DemoTransport
        from runfleet.demo.world import DemoWorld

        clock = ManualClock(T0)
        world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.05)
        service = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
        service.lock_path = False       # like --demo: never take (or report) the user's real instance locks
    except (ImportError, NotImplementedError, TypeError) as exc:   # pragma: no cover - only before M2
        pytest.skip(f"real stack not available yet: {exc!r}")
    return RunFleetApp(service, clock=clock, animations=False), service, world, clock


async def _settle(pilot: Any, service: Any, *, rounds: int = 60) -> None:
    for _ in range(rounds):
        await pilot.pause(0.05)
        if any(True for _ in service.store.jobs()):
            break
    await pilot.pause(0.3)


@pytest.fixture
async def stack() -> AsyncIterator[tuple[RunFleetApp, Any, Any, ManualClock, Any]]:
    app, service, world, clock = _stack()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await _settle(pilot, service)
        yield app, service, world, clock, pilot


def header(app: RunFleetApp) -> str:
    return app.screen.query_one(FleetHeader).text.plain


async def test_real_stack_fleet_and_jobs(stack: Any) -> None:
    app, service, _world, _clock, pilot = stack
    assert isinstance(app.screen, MainScreen)
    assert len(app.screen.query(HostCard)) == 5
    assert "✖ lynx auth" in header(app)
    await pilot.press("tab")
    await pilot.pause(0.2)
    view = app.screen.query_one(JobsView)
    jobs, _total = view.visible_jobs()
    assert jobs and set(app.screen.query_one(JobTable).order) == {str(j.key) for j in jobs}


async def test_real_stack_cancel(stack: Any) -> None:
    app, service, _world, clock, pilot = stack
    await pilot.press("tab")
    await pilot.pause(0.2)
    table = app.screen.query_one(JobTable)
    table.move_to_key("k2:3370390")
    await pilot.pause()
    await pilot.press("ctrl+k")
    await pilot.pause(0.2)
    assert isinstance(app.screen, ConfirmScreen)
    assert "scancel 3370390" in app.screen.plans[0].preview
    await pilot.press("y")
    for _ in range(20):
        clock.advance(1)
        await pilot.pause(0.1)
        job = service.store.job(JobKey("k2", "3370390"))
        if job is not None and job.state is JobState.CANCELLED:
            break
    job = service.store.job(JobKey("k2", "3370390"))
    assert job is not None and job.state is JobState.CANCELLED
    await pilot.pause(0.3)
    if "k2:3370390" in table.order:
        assert table.get_cell("k2:3370390", "state").plain == "CA"
    assert any("Cancelled 1 job on k2" in n.message for n in app._notifications)


async def test_real_stack_connect_needs_auth(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, _world, clock, pilot = stack
    monkeypatch.setattr(app, "run_external", lambda argv, **kw: 0)
    app.connect("lynx")
    for _ in range(30):
        clock.advance(1)
        await pilot.pause(0.1)
        if service.store.hosts["lynx"].status.state is ConnState.OK:
            break
    assert service.store.hosts["lynx"].status.state is ConnState.OK
    await pilot.pause(0.3)
    assert "✖ lynx" not in header(app)


async def test_real_stack_fault_injection(stack: Any) -> None:
    app, service, world, clock, pilot = stack
    world.inject("vega", "unreachable")
    for _ in range(80):
        clock.advance(5)
        await pilot.pause(0.05)
        if service.store.hosts["vega"].status.state is ConnState.UNREACHABLE:
            break
    await pilot.pause(0.3)
    assert service.store.hosts["vega"].status.state in (ConnState.UNREACHABLE, ConnState.RETRYING)
    assert "vega" in header(app) and app.is_running
    world.clear("vega")
    for _ in range(80):
        clock.advance(10)
        await pilot.pause(0.05)
        if service.store.hosts["vega"].status.state is ConnState.OK:
            break
    await pilot.pause(0.3)
    assert service.store.hosts["vega"].status.state is ConnState.OK
    assert "● vega" in header(app)


async def test_real_stack_log_viewer(stack: Any) -> None:
    app, service, _world, _clock, pilot = stack
    await pilot.press("tab")
    await pilot.pause(0.2)
    app.screen.query_one(JobTable).move_to_key("k2:3370390")
    await pilot.pause()
    await pilot.press("l")
    for _ in range(40):
        await pilot.pause(0.05)
        if isinstance(app.screen, LogScreen) and app.screen.query_one(LogView).count > 0:
            break
    assert isinstance(app.screen, LogScreen)
    assert app.screen.query_one(LogView).count > 0


async def test_real_stack_demo_transitions_notify(stack: Any) -> None:
    """§7.14 on the real stack: no alert at start (baseline), then the demo's lyra failure ~80 s in."""
    app, service, _world, clock, pilot = stack
    assert not app.alerts.history
    for _ in range(60):
        clock.advance(5)
        await pilot.pause(0.05)
        if any(e.key == JobKey("lyra", "32378152") for e in app.alerts.history):
            break
    await pilot.pause(0.3)
    failed = [e for e in app.alerts.history if e.key == JobKey("lyra", "32378152")]
    assert failed and failed[0].category == "ended_bad"
    assert any(e.category == "time_low" and e.key == JobKey("lyra", "32378151") for e in app.alerts.history)


async def test_real_stack_forward_the_demo_tensorboard(stack: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """§7.15 on the real stack: `P` finds the TensorBoard URL in the demo job's log and forwards it (simulated)."""
    from textual.widgets import OptionList

    from runfleet.ui.screens.action_menu import ActionMenuScreen

    app, service, world, _clock, pilot = stack
    monkeypatch.delenv("TMUX", raising=False)
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    await pilot.press("tab")
    await pilot.pause(0.2)
    app.screen.query_one(JobTable).move_to_key("k2:3371310")
    await pilot.pause()
    await pilot.press("P")
    for _ in range(40):
        await pilot.pause(0.05)
        if isinstance(app.screen, ActionMenuScreen):
            break
    menu = app.screen.query_one(OptionList)
    assert menu.get_option_at_index(0).id == "url:6006"
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause(0.05)
        if copied:
            break
    fwd = service.forwards()
    assert len(fwd) == 1 and fwd[0].node == "n-38-01" and fwd[0].remote_port == 6006
    assert copied == [fwd[0].url] and any("demo forward" in e for e in world.events)


async def test_real_stack_find_other_log_files(stack: Any) -> None:
    """The field report on the real stack (§4.7.2): the demo launcher job's stdout is 6 lines from 13 h ago; the Logs
    tab says so, `f` lists what `log.find` found under its workdir (train.log first), `enter` follows train.log."""
    from runfleet.demo.world import LAUNCHER_JOB, LAUNCHER_TRAIN_LOG, LAUNCHER_WORKDIR
    from runfleet.ui.screens.found_logs import FoundLogsScreen
    from runfleet.ui.screens.job_detail import JobDetailScreen

    app, service, _world, _clock, pilot = stack
    key = JobKey("k2", str(LAUNCHER_JOB))
    assert service.store.job(key) is not None
    app.open_job(key)
    for _ in range(40):
        await pilot.pause(0.05)
    detail = app.screen
    assert isinstance(detail, JobDetailScreen)
    hint = detail.query_one("#logs-hint")
    assert hint.display and str(hint.render()).startswith("stdout quiet for 13h; f finds other log files")
    await pilot.press("f")
    for _ in range(60):
        await pilot.pause(0.05)
        if isinstance(app.screen, FoundLogsScreen) and app.screen.result is not None:
            break
    picker = app.screen
    assert isinstance(picker, FoundLogsScreen) and picker.result is not None
    paths = [f.path for f in picker.result.files]
    assert paths[0] == LAUNCHER_TRAIN_LOG and all(p.startswith(LAUNCHER_WORKDIR + "/") for p in paths)
    assert any(p.endswith("/eval.err") for p in paths) and any(p.endswith("/files/output.log") for p in paths)
    assert not any("/.venv" in p or p.endswith(".jsonl") for p in paths)
    rows = [str(picker.query_one("#fl-list").get_option_at_index(i).prompt) for i in range(len(paths))]
    assert rows[0].startswith("runs/in1k-b16/r0412/s42/train.log")
    await pilot.press("enter")
    for _ in range(60):
        await pilot.pause(0.05)
        if app.screen is detail and detail.query_one("#logs-log", LogView).count > 0:
            break
    assert app.screen is detail and detail.stream == "file" and not hint.display
    assert detail.follower is not None and detail.follower.ref is not None
    assert detail.follower.ref.path == LAUNCHER_TRAIN_LOG and detail.follower.ref.source == "found"
    assert detail.query_one("#logs-log", LogView).count > 0
