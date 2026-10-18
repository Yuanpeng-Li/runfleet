"""The simulated fleet: default scenario shape, lifecycle, determinism, actions, task signals, faults (§10.6)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from runfleet.demo import render
from runfleet.demo.world import (FAULTS, TASK_EPOCH_GRID_S, DemoWorld, SimLog, Trainer, compress_ranges,
                                 expand_pattern, expand_ranges, task_epoch, tqdm_bar)
from tests.demo.conftest import StepClock


def states(world: DemoWorld, host: str) -> dict[str, str]:
    return {j.display_id: j.state for j in world.cluster(host).jobs.values()}


# ---------------------------------------------------------------------------------------------------- scenario
def test_default_fleet_mirrors_the_real_one(world: DemoWorld) -> None:
    assert list(world.hosts) == ["k2", "vega", "lyra", "labstation", "lynx"]
    cfg = world.fleet_config()
    assert [h.name for h in cfg.hosts] == list(world.hosts) and cfg.source is None
    k2 = world.configs["k2"]
    assert k2.slurm.protected_qos == ("labpi",) and k2.slurm.log_templates
    assert world.configs["vega"].connect_timeout == 120
    assert world.faults == {"lynx": {"auth"}}           # ws-duo shape: NEEDS_AUTH until connect
    assert [world.cluster(n).slurm_version for n in ("k2", "vega", "lyra")] == ["25.05.8", "25.11.4", "26.05.3"]
    assert (world.cluster("k2").gpu_tres, world.cluster("lyra").gpu_tres) == (True, False)
    assert world.cluster("lyra").preempt_mode == "CANCEL" and world.cluster("k2").preempt_mode == "REQUEUE"
    assert len(world.workstation("labstation").gpus) == 4 and len(world.workstation("lynx").gpus) == 2


def test_scenario_process_starts_do_not_depend_on_the_moment_the_world_is_built() -> None:
    """Task ids are p<pid>-<start>: two `--demo` calls minutes apart build two worlds; their tasks must agree."""
    first = StepClock()
    later = StepClock()
    later.advance(TASK_EPOCH_GRID_S - 1 - first.t.timestamp() % TASK_EPOCH_GRID_S)    # same grid hour, its end
    a = DemoWorld(seed=7, now=first, latency_scale=0.0)
    b = DemoWorld(seed=7, now=later, latency_scale=0.0)
    assert b.t0 > a.t0
    for host in ("labstation", "lynx"):
        starts_a = {pid: p.start for pid, p in a.workstation(host).procs.items()}
        assert starts_a == {pid: p.start for pid, p in b.workstation(host).procs.items()}
        assert all(start <= a.t0 for start in starts_a.values())
    assert task_epoch(a.t0) % TASK_EPOCH_GRID_S == 0 and a.t0 - TASK_EPOCH_GRID_S < task_epoch(a.t0) <= a.t0


def test_default_scenario_covers_every_state_and_shape(world: DemoWorld) -> None:
    k2 = states(world, "k2")
    assert {"RUNNING", "PENDING", "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"} <= set(k2.values())
    lyra = states(world, "lyra")
    assert lyra["32354789"] == "PREEMPTED" and lyra["32377076_[5-12%2]"] == "PENDING"
    vega = world.cluster("vega").jobs
    assert vega[824479].reason == "QOSMaxJobsPerUserLimit" and vega[824507].reason == "DependencyNeverSatisfied"
    multi = world.cluster("k2").jobs[3372094]
    assert multi.nodes == ("n-37-01", "n-37-02") and multi.gpus == 8 and multi.step0 is not None
    stalled = world.cluster("k2").jobs[3370392]
    assert stalled.stalled and world.t0 - world.cluster("k2").logs[stalled.log_paths[0]].mtime == 47 * 60
    failed = world.cluster("k2").jobs[3368404]
    assert "Traceback (most recent call last)" in world.cluster("k2").logs[failed.log_paths[0]].text()
    others = [p for p in world.workstation("labstation").procs.values() if p.user != "alice" and p.gpu is not None]
    assert others and others[0].user == "carol"


# ---------------------------------------------------------------------------------------------------- lifecycle
def test_jobs_move_through_their_lifecycle(world: DemoWorld, clock: StepClock) -> None:
    k2 = world.cluster("k2")
    clock.advance(120)
    world.sync()
    assert k2.jobs[3371232].state == "COMPLETING"
    clock.advance(10)
    world.sync()
    assert k2.jobs[3371232].state == "COMPLETED" and k2.jobs[3371232].end is not None
    assert k2.jobs[3372066].state == "PENDING"
    clock.advance(36 * 60)
    world.sync()
    smoke = k2.jobs[3372066]
    assert smoke.state == "RUNNING" and smoke.nodes and smoke.log_paths[0] in k2.logs
    assert smoke.alloc_gpu_type == "nvidia_geforce_rtx_3090"


def test_logs_grow_with_tqdm_carriage_returns(world: DemoWorld, clock: StepClock) -> None:
    k2 = world.cluster("k2")
    lg = k2.logs[k2.jobs[3370390].log_paths[0]]
    size = lg.size
    clock.advance(60)
    world.sync()
    new = lg.read_from(size).decode()
    assert lg.size > size and "\r" in new and "it/s]" in new
    stalled = k2.logs[k2.jobs[3370392].log_paths[0]]
    before = stalled.size
    clock.advance(600)
    world.sync()
    assert stalled.size == before                     # a stalled job writes nothing


def test_qos_cap_holds_the_fifth_job_until_a_slot_frees(world: DemoWorld, clock: StepClock) -> None:
    vega = world.cluster("vega")
    clock.advance(60)
    world.sync()
    assert vega.jobs[824479].state == "PENDING" and vega.jobs[824479].reason == "QOSMaxJobsPerUserLimit"
    rc, err = world.slurm_action("vega", "cancel", ["824245"])
    assert (rc, err) == (0, "")
    clock.advance(10)
    world.sync()
    assert vega.jobs[824479].state == "RUNNING"
    assert vega.jobs[824507].state == "PENDING"       # DependencyNeverSatisfied never runs


def test_array_respects_its_throttle_and_drains(world: DemoWorld, clock: StepClock) -> None:
    lyra = world.cluster("lyra")
    arr = 32377076
    for _ in range(40):
        clock.advance(600)
        world.sync()
        running = [j for j in lyra.jobs.values() if j.array_job_id == arr and not j.summary
                   and j.state in ("RUNNING", "COMPLETING")]
        assert len(running) <= 2
    elements = sorted(int(j.array_task or 0) for j in lyra.jobs.values() if j.array_job_id == arr and not j.summary)
    assert elements == list(range(1, 13))
    assert not any(j.summary for j in lyra.jobs.values())
    last = next(j for j in lyra.jobs.values() if j.array_job_id == arr and j.array_task == "12")
    assert last.job_id == arr                           # the last task inherits the array job id


def test_queue_runner_starts_the_next_item(world: DemoWorld, clock: StepClock) -> None:
    ws = world.workstation("lynx")
    clock.advance(8 * 60 + 20)
    world.sync()
    assert 786345 not in ws.procs
    children = ws.children(642229)
    assert len(children) == 2 and {c.gpu for c in children} == {0, 1}
    new = next(c for c in children if c.pid != 642234)
    assert new.pgid == 642229 and new.item == "cifar/baseline-warmup/s44"
    assert "launch: item 5/7" in ws.logs[ws.procs[642229].fd1].text()


def test_crashing_task_leaves_a_traceback(world: DemoWorld, clock: StepClock) -> None:
    ws = world.workstation("labstation")
    clock.advance(105)
    world.sync()
    assert 590112 not in ws.procs
    text = ws.logs["/home/alice/projects/orb/logs/eval_ckpt_voc_s44.log"].text()
    assert "Traceback (most recent call last)" in text and text.rstrip().endswith("best.pt'")


def test_same_seed_same_world(make_world: Callable[..., tuple[DemoWorld, StepClock]]) -> None:
    def snapshot(seed: int) -> str:
        w, clk = make_world(seed)
        for step in (30, 300, 900, 1800):
            clk.advance(step)
            w.sync()
        now = w.now()
        return (render.squeue(w.cluster("k2"), now).stdout + render.sacct_window(w.cluster("vega"), now,
                since_s=86400).stdout + render.nvidia_gpu(w.workstation("lynx")).stdout
                + render.process_ps(w.workstation("labstation"), now).stdout)

    assert snapshot(7) == snapshot(7)
    assert snapshot(7) != snapshot(8)


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError):
        DemoWorld(scenario="chaos")


def test_fleet_config_subset_and_unknown(world: DemoWorld) -> None:
    assert [h.name for h in world.fleet_config(["lynx", "k2"]).hosts] == ["lynx", "k2"]
    with pytest.raises(KeyError, match="demo hosts: k2"):
        world.fleet_config(["atlas"])


# ---------------------------------------------------------------------------------------------------- slurm actions
def test_cancel_running_and_pending_jobs(world: DemoWorld) -> None:
    k2 = world.cluster("k2")
    assert world.slurm_action("k2", "cancel", ["3370390", "3372066"]) == (0, "")
    run, pd = k2.jobs[3370390], k2.jobs[3372066]
    assert run.state == "CANCELLED" and run.cancelled_by == k2.uid and run.nodes and run.exit_signal == 15
    assert "CANCELLED AT" in k2.logs[run.log_paths[0]].text().splitlines()[-1]
    assert pd.state == "CANCELLED" and pd.start is None and not pd.nodes
    rc, err = world.slurm_action("k2", "cancel", ["3370390", "999"])
    assert rc == 1
    assert "Kill job error on job id 3370390: Job/step already completing or completed" in err
    assert "Kill job error on job id 999: Invalid job id specified" in err


def test_hold_release_requeue(world: DemoWorld) -> None:
    k2 = world.cluster("k2")
    assert world.slurm_action("k2", "hold", ["3372066"]) == (0, "")
    assert k2.jobs[3372066].held and k2.jobs[3372066].reason == "JobHeldUser"
    assert world.slurm_action("k2", "release", ["3372066"]) == (0, "")
    assert not k2.jobs[3372066].held and k2.jobs[3372066].reason == "Priority"
    rc, err = world.slurm_action("k2", "hold", ["3370391"])
    assert rc == 1 and "no longer pending" in err
    assert world.slurm_action("k2", "requeue", ["3370391"]) == (0, "")
    job = k2.jobs[3370391]
    assert job.state == "PENDING" and job.restarts == 1 and job.reason == "BeginTime" and not job.nodes
    assert world.slurm_action("k2", "requeue", ["3371310"], hold=True) == (0, "")
    assert k2.jobs[3371310].held and k2.jobs[3371310].reason == "JobHeldUser"
    rc, err = world.slurm_action("k2", "requeue", ["3368404"])     # ended hours ago: gone from the controller
    assert rc == 1 and "Invalid job id specified" in err


def test_cancel_array_whole_and_partial(world: DemoWorld) -> None:
    lyra = world.cluster("lyra")
    assert world.slurm_action("lyra", "cancel", ["32377076_[7-8]"]) == (0, "")
    summary = lyra.jobs[32377076]
    assert summary.array_pending == [5, 6, 9, 10, 11, 12] and summary.array_task == "5-6,9-12%2"
    split = sorted(j.array_task or "" for j in lyra.jobs.values() if j.array_job_id == 32377076
                   and j.state == "CANCELLED")
    assert split == ["7", "8"]
    assert world.slurm_action("lyra", "cancel", ["32377076_[5-12]"]) == (0, "")
    assert summary.state == "CANCELLED" and summary.display_id.startswith("32377076_[")
    assert world.slurm_action("lyra", "cancel", ["32377076_3"]) == (0, "")
    assert lyra.jobs[32377079].state == "CANCELLED"


# ---------------------------------------------------------------------------------------------------- task signals
def test_terminate_task_reports_like_the_signal_script(world: DemoWorld, clock: StepClock) -> None:
    ws = world.workstation("lynx")
    root = ws.procs[642229]
    lines = world.signal_task("lynx", 642229, root.start, "TERM", 10, True)
    assert lines == ["members 642229 642234 786345 ", "sent 642234", "sent 786345", "sent 642229", "done"]
    assert not {642229, 642234, 786345} & set(ws.procs) and 642229 not in ws.runners
    clock.advance(600)
    world.sync()
    assert not any(p.ppid == 642229 for p in ws.procs.values())     # nothing respawned
    assert world.signal_task("lynx", 642229, root.start, "TERM", 10, True) == ["gone 642229", "done"]


def test_interrupt_writes_keyboard_interrupt_and_reused_pid_is_refused(world: DemoWorld) -> None:
    ws = world.workstation("labstation")
    root = ws.procs[363942]
    assert world.signal_task("labstation", 363942, root.start + 60, "INT", 0, False) == ["reused 363942", "done"]
    lines = world.signal_task("labstation", 363942, root.start, "INT", 0, False)
    assert lines[0] == "members 363942 363948 " and lines[-1] == "done"
    assert ws.logs[root.fd1].text().rstrip().endswith("KeyboardInterrupt")


# ---------------------------------------------------------------------------------------------------- faults
def test_inject_and_clear_faults(world: DemoWorld) -> None:
    for fault in FAULTS:
        world.inject("k2", fault)
    assert world.faults["k2"] == set(FAULTS)
    world.clear("k2", "garbage")
    assert not world.has_fault("k2", "garbage") and world.has_fault("k2", "auth")
    world.clear("k2")
    assert not world.faults["k2"]
    with pytest.raises(ValueError):
        world.inject("k2", "meteor")
    with pytest.raises(KeyError):
        world.inject("atlas", "auth")


# ---------------------------------------------------------------------------------------------------- helpers
def test_simlog_trims_but_keeps_true_size() -> None:
    lg = SimLog("/x", 0.0)
    chunk = "y" * 1_000_000 + "\n"
    for i in range(6):
        lg.write(chunk, float(i))
    assert lg.size == 6 * len(chunk) and lg.mtime == 5.0
    assert lg.tail(5) == b"yyyy\n"
    assert lg.read_from(lg.size - 2) == b"y\n"
    assert len(lg.read_from(0)) < lg.size              # head was dropped from memory


def test_ranges_patterns_and_bars(world: DemoWorld) -> None:
    assert expand_ranges("1-3,7") == [1, 2, 3, 7] and compress_ranges([7, 1, 2, 3]) == "1-3,7"
    job = world.cluster("lyra").jobs[32377079]
    assert expand_pattern("logs/slurm-%A_%a.out", job, "alice") == "/vol/alice/orb/logs/slurm-32377076_3.out"
    summary = world.cluster("lyra").jobs[32377076]
    assert expand_pattern("slurm-%A_%a-%%-%u.out", summary, "u").endswith("slurm-32377076_%a-%-u.out")
    assert tqdm_bar(0, 10) == " " * 10 and tqdm_bar(10, 10) == "█" * 10 and tqdm_bar(3, 8) == "███▊      "
    tr = Trainer(epochs=2, epoch_s=10.0, iters=100)
    text = tr.advance(25.0, 1_000_000.0, world.cluster("k2").tz, world.rng)
    assert text.count("\n") == 4 and tr.epoch == 3   # two epochs done -> bar + summary each, then nothing left


def test_notification_showcase_transitions(world: DemoWorld, clock: StepClock) -> None:
    """§7.14: the demo changes state in its first minutes (a failure, a requeue with restarts, a timeout)."""
    from runfleet.demo.world import DEMO_WATCHED, demo_ui_state

    lyra, k2 = world.cluster("lyra"), world.cluster("k2")
    assert k2.jobs[3372115].restarts == 1 and lyra.jobs[32378151].state == "RUNNING"
    clock.advance(90)
    world.sync()
    assert lyra.jobs[32378152].state == "FAILED"
    clock.advance(80)
    world.sync()
    assert k2.jobs[3372115].state == "PENDING" and k2.jobs[3372115].restarts == 2
    clock.advance(12 * 60)
    world.sync()
    assert lyra.jobs[32378151].state == "TIMEOUT"
    state = demo_ui_state(clock())
    assert list(state["watch"]) == [DEMO_WATCHED] and state["last_seen"] < clock().isoformat()
