"""[WP3] Attach in JobDetail (§7.16.7): GPU / Procs tabs of running Slurm jobs, the srun confirmation, r / a / S."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from textual.widgets import TabbedContent

from runfleet.models import AttachOffer, AttachStatus, ErrorKind, HostError, JobKey
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.confirm import AttachConfirmScreen
from runfleet.ui.screens.help import HelpScreen
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.proc_table import ProcTable, stat_style
from runfleet.util import humanize
from tests.ui.fakes import LYRA_LIVE_KEY, FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 40)
VEGA = JobKey("vega", "723954")
K2 = JobKey("k2", "2332861")


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


def text(app: RunFleetApp, selector: str) -> str:
    r = app.screen.query_one(selector).render()
    return getattr(r, "plain", str(r))


def calls(svc: FakeService, name: str) -> list[tuple[Any, ...]]:
    return svc.called(name)


async def open_tab(pilot: Any, app: RunFleetApp, key: JobKey, tab: str = "gpu") -> JobDetailScreen:
    app.open_job(key)
    await pilot.pause(0.2)
    screen = app.screen
    assert isinstance(screen, JobDetailScreen)
    screen.query_one(TabbedContent).active = tab
    await pilot.pause(0.2)
    return screen


# ---------------------------------------------------------------- GPU tab
async def test_vega_live_rows_and_idle_flag_on_the_second_sample(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.open_job(VEGA)
        await pilot.pause(0.2)
        assert not calls(svc, "attach_start")                     # Summary: no node is touched
        await pilot.press("right_square_bracket")
        await pilot.pause(0.2)
        assert calls(svc, "attach_start") == [(VEGA, None)]
        box = text(app, "#attach-box")
        assert "Live · ssh n-38-01 (job 723954)" in box and "every 15 s, next 15 s" in box and "clean ✓" in box
        assert "r now · a auto on/off" in box
        assert "89:00.0" in box and "C5:00.0" in box and "576/600W" in box
        assert "129392 VLLM::EngineCore 54.5G" in box
        assert "mem 134G / 256G (peak 188G, shmem 56G) · oom_kill 0 · CPU: next sample" in box
        assert "accounting (sstat, 30s ago)  Σ170% / 2 GPUs" in box   # a multi-GPU sum, never a mean
        assert "idle but holding" not in box                       # one sample is not a streak
        assert "v0.2" not in text(app, "#gpu-tab") + box
        await pilot.press("r")
        await pilot.pause(0.2)
        assert calls(svc, "attach_now") == [(VEGA, None, None, None)]
        box = text(app, "#attach-box")
        line = next(x for x in box.splitlines() if "C5:00.0" in x)
        assert "! idle but holding 82.7G" in line and "  0%" in line
        assert "CPU ≈ 3.6 / 8 cores" in box
        assert "GPU1 idle" not in box                              # the row says it; no second, differently rounded copy


async def test_k2_landed_in_another_job_says_whose_gpus(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, K2)
        box = text(app, "#attach-box")
        assert "ssh n-40-01 (job 2332863)" in box
        assert "ssh landed in job 2332863 on n-40-01; GPUs shown belong to 2332863" in box
        assert "(job 2332863)" in next(x for x in box.splitlines() if "88430" in x)
        assert "also here 2332863" in box
        assert "S reads" not in box                                # attach = "ssh": no srun offer
        await pilot.press("S")
        await pilot.pause(0.1)
        assert not calls(svc, "attach_now")


async def test_k2_srun_offer_when_ssh_landed_elsewhere(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.attach_setting["k2"] = "ssh+srun"
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, K2)
        state = svc.attach_state(K2)
        assert state is not None
        offer = AttachOffer("n-40-01", "ssh landed in job 2332863", "Read job 2332861 with srun?",
                            "Runs one step in job 2332861; it leaves one permanent sacct row 2332861.N rfprobe.",
                            "srun --jobid=2332861 --overlap --job-name=rfprobe -N1 -n1 -w n-40-01 …")
        svc._attach_put(dataclasses.replace(state, offer=offer))     # the core offers srun when ssh lands elsewhere
        await pilot.pause(0.4)
        assert "S reads job 2332861's own GPUs with srun on n-40-01" in text(app, "#attach-box")
        await pilot.press("S")
        await pilot.pause(0.2)
        assert isinstance(app.screen, AttachConfirmScreen)
        await pilot.press("y")
        await pilot.pause(0.2)
        assert calls(svc, "attach_now")[-1] == (K2, "srun", "once", None)
        state = svc.attach_state(K2)
        assert state is not None and state.srun_rows == 1


async def test_srun_offer_is_shown_when_ssh_cannot_reach_the_node(make_app: MakeApp) -> None:
    """HOSTKEY / DNS / REFUSED: no sample at all, but the core offers srun; the panel must advertise `S`."""
    app, svc = make_app()
    svc.attach_setting["k2"] = "ssh+srun"
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, K2)
        state = svc.attach_state(K2)
        assert state is not None
        offer = AttachOffer("n-40-01", "n-40-01's host key changed (login host's known_hosts)",
                            "Read job 2332861 with srun?", "one permanent sacct row", "srun … -w n-40-01 …")
        err = HostError(ErrorKind.HOSTKEY, "n-40-01's host key changed (login host's known_hosts)")
        svc._attach_put(dataclasses.replace(state, status=AttachStatus.LIVE, live=None, error=err, offer=offer))
        await pilot.pause(0.4)
        box = text(app, "#attach-box")
        assert "No reading yet" in box and "host key changed" in box
        assert "S reads job 2332861's own GPUs with srun on n-40-01" in box


# ---------------------------------------------------------------- srun consent (lyra)
async def test_lyra_consent_cancel_once_and_allow_for_this_job(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.add_lyra_running_job()
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, LYRA_LIVE_KEY)
        assert calls(svc, "attach_start") and not calls(svc, "attach_now")   # opening a tab never runs srun
        box = text(app, "#attach-box")
        assert "srun needed · lyra refuses ssh to compute nodes" in box and "r read with srun (asks first)" in box
        hints = app.screen.query_one(HintBar).text.plain
        assert "r read now" in hints and "a auto" in hints

        await pilot.press("r")                                        # 1. cancel
        await pilot.pause(0.2)
        dialog = app.screen
        assert isinstance(dialog, AttachConfirmScreen)
        assert dialog.query_one("#attach-confirm-box").border_title == "Read job 31367626 with srun?"
        body = text(app, "#confirm-body")
        assert "permanent sacct row 31367626.N rfprobe" in body
        assert "will run on lyra (node s71-02):  srun --jobid=31367626 --overlap" in body
        assert dialog.focused is not None and dialog.focused.id == "cancel"
        await pilot.press("n")
        await pilot.pause(0.2)
        assert isinstance(app.screen, JobDetailScreen)
        assert [c[2] for c in calls(svc, "attach_now")] == [None]
        assert svc.attach_samples.get(LYRA_LIVE_KEY) is None
        assert calls(svc, "attach_stop") == []                        # a dialog on top keeps the attach

        await pilot.press("r")                                        # 2. once
        await pilot.pause(0.2)
        await pilot.press("y")
        await pilot.pause(0.3)
        assert calls(svc, "attach_now")[-1] == (LYRA_LIVE_KEY, None, "once", None)
        box = text(app, "#attach-box")
        assert "Live · srun s71-02 · 10:42:37 · 1.21 s · manual" in box
        assert "1 sacct row added this session" in box and "r read again (asks)" in box
        assert "RTX PRO 6000 BW" in box or "71:00.0" in box

        await pilot.press("r")                                        # once is not remembered: asks again
        await pilot.pause(0.2)
        assert isinstance(app.screen, AttachConfirmScreen)
        assert "1 sacct row added for this job this session" in text(app, "#confirm-body")
        await pilot.press("a")                                        # 3. allow for this job (this session)
        await pilot.pause(0.3)
        assert calls(svc, "attach_now")[-1] == (LYRA_LIVE_KEY, None, "job", None)
        await pilot.press("r")                                        # no dialog any more
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen)
        assert calls(svc, "attach_now")[-1] == (LYRA_LIVE_KEY, None, None, None)
        state = svc.attach_state(LYRA_LIVE_KEY)
        assert state is not None and state.srun_rows == 3 and state.consent == "job"
        box = text(app, "#attach-box")
        assert "3 sacct rows added this session" in box and "r read again (allowed)" in box


async def test_readonly_blocks_srun(make_app: MakeApp) -> None:
    app, svc = make_app(readonly=True)
    svc.add_lyra_running_job()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await open_tab(pilot, app, LYRA_LIVE_KEY)
        assert "Live readings unavailable · read-only mode: srun adds a permanent sacct row" in text(app, "#attach-box")
        await pilot.press("r")
        await pilot.pause(0.2)
        assert not isinstance(app.screen, AttachConfirmScreen)
        assert any("read-only mode" in m for m in toasts(app))
        assert svc.attach_samples.get(LYRA_LIVE_KEY) is None


async def test_blocked_when_attach_is_off_and_hint_when_ssh_is_refused(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.attach_setting["vega"] = "off"
    svc.add_lyra_running_job()
    svc.attach_setting["lyra"] = "ssh"
    svc.node_ssh["lyra"] = "denied"
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await open_tab(pilot, app, VEGA)
        box = text(app, "#attach-box")
        assert "Live readings unavailable · attach is off for vega" in box
        assert "accounting (sstat" in box and "util" in text(app, "#gpu-tab")   # the sstat view stays
        await pilot.press("r")
        await pilot.pause(0.2)
        assert any("attach is off for vega" in m for m in toasts(app))
        await pilot.press("escape")
        await pilot.pause(0.2)
        await open_tab(pilot, app, LYRA_LIVE_KEY)
        box = text(app, "#attach-box")
        assert "lyra refuses ssh to compute nodes" in box
        assert 'add attach = "ssh+srun" to [[host]] lyra to read via srun' in box
        await pilot.press("r")
        await pilot.pause(0.2)
        assert any('add attach = "ssh+srun"' in m for m in toasts(app))


# ---------------------------------------------------------------- lifecycle
async def test_tab_switches_start_and_stop_the_attach(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.open_job(VEGA)
        await pilot.pause(0.2)
        seen = []
        for _ in range(4):                                            # gpu, logs, steps, procs
            await pilot.press("right_square_bracket")
            await pilot.pause(0.1)
            seen.append((len(calls(svc, "attach_start")), len(calls(svc, "attach_stop"))))
        assert seen == [(1, 0), (1, 1), (1, 1), (2, 1)]
        assert svc.attach_open == VEGA
        await pilot.press("question_mark")                            # a dialog on top keeps it
        await pilot.pause(0.2)
        assert isinstance(app.screen, HelpScreen) and svc.attach_open == VEGA
        await pilot.press("escape")
        await pilot.pause(0.2)
        await pilot.press("l")                                        # a full screen on top ends it ...
        await pilot.pause(0.3)
        assert isinstance(app.screen, LogScreen) and svc.attach_open is None
        await pilot.press("escape")                                   # ... and coming back restarts it
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen) and svc.attach_open == VEGA


async def test_closing_job_detail_stops_the_attach(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, VEGA)
        assert svc.attach_open == VEGA
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert isinstance(app.screen, MainScreen)
        assert svc.attach_open is None
        state = svc.attach_state(VEGA)
        assert state is not None and state.status is AttachStatus.STOPPED and state.stopped == "tab closed"


async def test_a_toggles_auto_refresh(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        app.open_job(VEGA)
        await pilot.pause(0.2)
        await pilot.press("a")                                        # Summary: says where it works
        await pilot.pause(0.1)
        assert not calls(svc, "attach_set_auto")
        assert any("GPU / Procs tab" in m for m in toasts(app))
        await pilot.press("right_square_bracket")
        await pilot.pause(0.2)
        await pilot.press("a")
        await pilot.pause(0.2)
        assert calls(svc, "attach_set_auto") == [(VEGA, False)]
        box = text(app, "#attach-box")
        assert "auto off" in box and "every 15 s" not in box
        await pilot.press("a")
        await pilot.pause(0.2)
        assert calls(svc, "attach_set_auto")[-1] == (VEGA, True)
        assert "every 15 s" in text(app, "#attach-box")


async def test_live_keys_rerender_and_countdown(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        screen = await open_tab(pilot, app, VEGA)
        state = svc.attach_state(VEGA)
        assert state is not None
        svc._attach_put(dataclasses.replace(state, status=AttachStatus.PAUSED, stopped="app in background",
                                            next_at=None))
        await pilot.pause(0.4)
        assert "Paused (app in background)" in text(app, "#attach-box")
        svc._attach_put(dataclasses.replace(state, status=AttachStatus.LIVE))
        await pilot.pause(0.4)
        assert "next 15 s" in text(app, "#attach-box")
        svc.clock.advance(6)
        screen.refresh_time()                                         # the 1 Hz ClockTick
        assert "next 9 s" in text(app, "#attach-box")


# ---------------------------------------------------------------- Procs tab
async def test_procs_tab_has_a_step_column(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, VEGA, "procs")
        assert calls(svc, "attach_start") == [(VEGA, None)]
        table = app.screen.query_one(ProcTable)
        assert table.display and [k for k, _l, _w in table.cols][:2] == ["gpu", "pid"]
        keys = [k for k, _l, _w in table.cols]
        assert "step" in keys and "task" not in keys
        assert table.order[0] == 129361                               # most GPU memory first
        row = {k: str(c) for (k, _l, _w), c in zip(table.cols, table._cells[129361], strict=True)}
        assert row["step"] == "723954.batch" and row["gpu"] == "0,1"
        note = text(app, "#procs-note")
        assert "Live · ssh n-38-01" in note and "3 processes of yours on n-38-01" in note
        assert "v0.2" not in note
        hints = app.screen.query_one(HintBar).text.plain
        assert "r read now" in hints


async def test_procs_tab_k2_note_and_steps_of_other_jobs(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await open_tab(pilot, app, K2, "procs")
        note = text(app, "#procs-note")
        assert "ssh landed in job 2332863 on n-40-01: the GPU columns are that job's GPUs" in note
        table = app.screen.query_one(ProcTable)
        steps = {pid: str(cells[[k for k, _l, _w in table.cols].index("step")]) for pid, cells in
                 table._cells.items()}
        assert steps == {88410: "2332861.batch", 88420: "2332862.batch", 88430: "2332863.batch"}


def test_d_state_and_zombies_are_highlighted() -> None:
    assert stat_style("D") is not None and stat_style("Zs") is not None
    assert stat_style("Sl") is None and stat_style("R+") is None
    assert stat_style("Z") != stat_style("D")


async def test_workstation_tabs_unchanged_and_pending_jobs_not_attached(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        task = next(j for j in svc.store.jobs() if j.key.host == "labstation")
        await open_tab(pilot, app, task.key, "procs")
        await pilot.pause(0.2)
        table = app.screen.query_one(ProcTable)
        keys = [k for k, _l, _w in table.cols]
        assert "task" in keys and "step" not in keys
        app.screen.query_one(TabbedContent).active = "gpu"
        await pilot.pause(0.2)
        assert not app.screen.query_one("#attach-box").display
        assert "r read now" not in app.screen.query_one(HintBar).text.plain
        assert not calls(svc, "attach_start")
        await pilot.press("escape")
        await pilot.pause(0.2)
        await open_tab(pilot, app, JobKey("vega", "724479"))         # PENDING
        assert not calls(svc, "attach_start")
        assert "only while the job runs" in text(app, "#attach-box") or \
            "start when this tab is open" in text(app, "#attach-box")


# ---------------------------------------------------------------- multi-node samples (pids repeat across nodes)
def _two_node_live() -> Any:
    from datetime import UTC, datetime

    from runfleet.models import GPU, GPUProcess, JobLive, NodeInspect, Proc

    key = JobKey("vega", "7")
    t = datetime(2026, 9, 7, 7, 42, 37, tzinfo=UTC)

    def proc(pid: int, args: str, stat: str = "Rl") -> Proc:
        return Proc(pid, 1, pid, "me", 600, 99.0, 1.0, 2**30, stat, args)

    def gpu(idx: int, uuid: str, pid: int, mem: float) -> GPU:
        return GPU("vega", 0, uuid, "NVIDIA H200", 90.0, mem, 143771.0, None, 50.0, 300.0, 600.0, None, "P0",
                   "nvidia-smi", (GPUProcess(pid, uuid, 0, mem, "me", "python", 600, 99.0, key),), t,
                   bus_id="00000000:89:00.0", slurm_idx=idx)

    n1 = NodeInspect("n-37-01", "7", "7", True, (gpu(0, "GPU-a", 4242, 30000.0),),
                     (proc(4242, "python train.py --rank 0"), proc(4200, "/bin/bash slurm_script", "S")),
                     {4242: "7.batch", 4200: "7.batch"}, cost_ms=600)
    n2 = NodeInspect("n-37-02", "7", "7", True, (gpu(3, "GPU-b", 4242, 50000.0),),
                     (proc(4242, "python train.py --rank 1"),), {4242: "7.0"}, cost_ms=650)
    return JobLive(key, "ssh", t, (n1, n2), 700, 0, True, "clean")


def test_live_procs_keep_every_node_apart() -> None:
    from runfleet.ui.widgets.attach_panel import live_procs

    rows = live_procs(_two_node_live())
    assert [(r.node, [p.pid for p in r.procs], dict(r.steps)) for r in rows] == [
        ("n-37-01", [4242, 4200], {4242: "7.batch", 4200: "7.batch"}), ("n-37-02", [4242], {4242: "7.0"})]
    assert [[g.slurm_idx for g in r.gpus] for r in rows] == [[0], [3]]


async def test_proc_table_keys_multi_node_rows_by_node_and_pid() -> None:
    from textual.app import App, ComposeResult

    from runfleet.ui.widgets.attach_panel import live_procs

    class TableApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ProcTable(id="t")

    app = TableApp()
    async with app.run_test(size=BIG) as pilot:
        table = app.query_one(ProcTable)
        table.show_live(live_procs(_two_node_live()), sort="gpu_mem", reverse=False)
        await pilot.pause()
        keys = [k for k, _l, _w in table.cols]
        assert keys[:3] == ["gpu", "node", "pid"] and "step" in keys
        assert table.order == [("n-37-02", 4242), ("n-37-01", 4242), ("n-37-01", 4200)]   # most GPU memory first

        def cell(row: object, col: str) -> str:
            return str(table._cells[row][keys.index(col)])                              # type: ignore[index]
        assert [cell(r, "node") for r in table.order] == ["n-37-02", "n-37-01", "n-37-01"]
        assert [cell(r, "gpu") for r in table.order] == ["3", "0", "-"]                 # each node's own GPU
        assert [cell(r, "gmem") for r in table.order] == ["48.8G", "29.3G", humanize.NONE]
        assert [cell(r, "step") for r in table.order] == ["7.0", "7.batch", "7.batch"]
        assert "rank 1" in cell(("n-37-02", 4242), "cmd") and "rank 0" in cell(("n-37-01", 4242), "cmd")
        table.move_cursor(row=1)
        current = table.current_proc()
        assert current is not None and "rank 0" in current.args
        # one node: rows keyed by pid as before, no NODE column
        one = dataclasses.replace(_two_node_live(), nodes=_two_node_live().nodes[:1])
        table.show_live(live_procs(one), sort="gpu_mem", reverse=False)
        await pilot.pause()
        assert table.order == [4242, 4200] and "node" not in [k for k, _l, _w in table.cols]


def test_procs_note_counts_processes_on_every_node() -> None:
    from datetime import UTC, datetime

    from runfleet.models import AttachState, JobState
    from runfleet.ui.widgets.attach_panel import procs_note
    from tests.ui.fakes import slurm_job

    live = _two_node_live()
    state = AttachState(key=live.key, status=AttachStatus.LIVE, setting="ssh", mode="ssh", live=live)
    job = slurm_job("vega", "7", "x", JobState.RUNNING)
    note = "\n".join(t.plain for t in procs_note(state, job, datetime(2026, 9, 7, 7, 42, 42, tzinfo=UTC), 160))
    assert "3 processes of yours on n-37-01, n-37-02" in note


def test_other_nvidia_smi_is_informational_not_a_leftover() -> None:
    from datetime import UTC, datetime

    from runfleet.models import AttachState, JobState
    from runfleet.ui import theme
    from runfleet.ui.widgets.attach_panel import live_lines
    from tests.ui.fakes import slurm_job

    base = _two_node_live()
    n1, n2 = base.nodes
    live = dataclasses.replace(base, nodes=(dataclasses.replace(n1, stray_nvidia_smi=0, other_nvidia_smi=2),
                                            dataclasses.replace(n2, stray_nvidia_smi=1, other_nvidia_smi=0)))
    state = AttachState(key=live.key, status=AttachStatus.LIVE, setting="ssh", mode="ssh", live=live)
    job = slurm_job("vega", "7", "x", JobState.RUNNING)
    lines = live_lines(state, job, None, datetime(2026, 9, 7, 7, 42, 42, tzinfo=UTC), 160, (50, 80), (50, 80))
    other = next(t for t in lines if "nvidia-smi" in t.plain and "n-37-01" in t.plain)
    assert other.plain.rstrip() == "2 other nvidia-smi of yours on n-37-01 (not RunFleet's, not hung)"
    assert other.style == theme.dim()
    hung = next(t for t in lines if "nvidia-smi" in t.plain and "n-37-02" in t.plain)
    assert hung.plain.rstrip() == "! 1 hung nvidia-smi on n-37-02"
