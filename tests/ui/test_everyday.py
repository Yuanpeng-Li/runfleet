"""[WP3] Everyday actions (§7.15): copy menu + support bundle, port forwards, log in to all, free-GPU picker,
whole-array marks and the busy-controller note — pure helpers first, then pilot flows on FakeService."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from textual import events
from textual.widgets import Input, OptionList

from runfleet.backends.base import ActionSpec
from runfleet.models import JobDetail, JobKey, JobState
from runfleet.ui import everyday
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.action_menu import ActionMenuScreen
from runfleet.ui.screens.forwards import ForwardsScreen, PortPromptScreen
from runfleet.ui.screens.host_error import HostErrorScreen
from runfleet.ui.widgets.gpus_view import GPUsView
from runfleet.ui.widgets.host_card import HostCard, busy_note
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from tests.ui.fakes import ME, T0, FakeService, slurm_job, task_job

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (120, 35)
FLUSH = 0.3
JOB = "k2:2332861"
TB = "TensorBoard 2.20.0 at http://n-40-01:6006/ (Press CTRL+C to quit)"
WANDB = "wandb: Appending key for api.wandb.ai to your netrc: 0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _no_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMUX", raising=False)        # never touch the developer's real tmux buffer


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


def menu_ids(app: RunFleetApp) -> list[str]:
    menu = app.screen.query_one(OptionList)
    return [str(menu.get_option_at_index(i).id) for i in range(menu.option_count)]


async def choose(pilot: Any, app: RunFleetApp, option_id: str) -> None:
    menu = app.screen.query_one(OptionList)
    menu.highlighted = menu_ids(app).index(option_id)
    await pilot.press("enter")
    await pilot.pause()


async def to_job(pilot: Any, app: RunFleetApp, key: str = JOB) -> None:
    await pilot.press("tab")
    await pilot.pause()
    view = app.screen.query_one(JobsView)
    while view.preset != "all":
        await pilot.press("f")
    app.screen.query_one(JobTable).move_to_key(key)
    await pilot.pause()


# ====================================================================================================== pure helpers
def test_scrub_removes_common_secrets() -> None:
    text = "\n".join([WANDB, "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12", "OPENAI sk-proj-abcdefghijklmnop1234",
                      "git ghp_0123456789abcdefghijABCDEFGHIJ", "AKIAABCDEFGHIJKLMNOP",
                      "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG", "http://n-40-01:8888/lab?token=s3cr3t&x=1"])
    out = everyday.scrub(text)
    for secret in ("0123456789abcdef0123456789abcdef01234567", "hf_abcdefghij", "sk-proj-abc", "ghp_0123",
                   "AKIAABCD", "wJalrXUtnFEMI", "s3cr3t"):
        assert secret not in out
    assert "token=<redacted>&x=1" in out and "aws_secret_access_key = <redacted>" in out


def test_scrub_removes_assignments_bearer_userinfo_and_private_keys() -> None:
    text = "\n".join(["PASSWORD=hunter2", "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload",
                      "git clone https://alice:pw0rd@host/x.git", '{"api_key": "abcd1234", "max_tokens": 512}',
                      "export WANDB_API_KEY='q1w2e3r4t5'", "num_tokens=2048", "```text",
                      "-----BEGIN OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXktdjE", "```"])
    out = everyday.scrub(text)
    for secret in ("hunter2", "eyJhbGci", "alice", "pw0rd", "abcd1234", "q1w2e3r4t5", "b3BlbnNzaC1rZXktdjE"):
        assert secret not in out
    assert "PASSWORD=<redacted>" in out and "Bearer <redacted>" in out and "https://<redacted>@host/x.git" in out
    assert '"max_tokens": 512' in out and "num_tokens=2048" in out            # not every *token* is a secret
    assert "<redacted-private-key>\n```" in out                                   # a cut key keeps the fence


def test_copy_items_paths_and_shell_command() -> None:
    job = slurm_job("k2", "2332861", "train", JobState.RUNNING, node="n-40-01", idx=(2,))
    items = {i.id: i for i in everyday.copy_items(job, alias="k2", shell_cmd="srun --jobid=2332861 --pty bash")}
    assert items["job_id"].text == "2332861" and items["host_job"].text == "k2:2332861"
    assert items["stderr"].what == "stderr path (= stdout)" and items["stderr"].text == items["stdout"].text
    assert all(len(i.label) + 2 <= everyday.LABEL_WIDTH for i in items.values())  # one line each, key in front
    assert list(items)[0] == "stdout" and set(everyday.COPY_KEYS.values()) == set(items)
    assert items["shell"].what == "node shell command"
    assert items["shell"].text == "ssh -t k2 'srun --jobid=2332861 --pty bash'"
    bare = dataclasses.replace(job, logs=(), workdir=None, state=JobState.PENDING, nodes=())
    items = {i.id: i for i in everyday.copy_items(bare, alias="user@h", shell_cmd=None)}
    assert items["stdout"].disabled and items["stderr"].disabled and items["workdir"].disabled
    assert items["shell"].what == "shell command" and items["shell"].text == "ssh -t user@h"
    assert everyday.paste_shell_command("local", "cd /w") == "cd /w"


def test_support_bundle_markdown_is_complete_and_scrubbed() -> None:
    job = slurm_job("k2", "2330539", "lmft", JobState.FAILED, exit_code=1, ended_ago=60, node="n-38-01", idx=(0, 1),
                    gpus=2)
    job = dataclasses.replace(job, restarts=2, hints=("exit 1",))

    detail = JobDetail(job=job, fields={}, per_node=(), steps=(), logs=(), gpu=None, procs=(), procs_note="", env={},
                       explanation=("batch step exited with code 1",), fetched_at=T0,
                       efficiency=("GPU util 12% of 2 GPUs — mostly idle",))     # the §5.2.8 card lines

    lines = [f"line {i}" for i in range(40)] + [WANDB, "torch.OutOfMemoryError: CUDA out of memory."]
    md = everyday.support_bundle(job, detail, cluster="k2 (Slurm 25.05.8)", log_lines=lines, now=T0,
                                 version="0.1")
    assert md.startswith("# RunFleet support bundle · k2:2330539")
    for part in ("**cluster:** k2 (Slurm 25.05.8)", f"**user:** {ME}", "gpu-preempt / normal / campus01",
                 "2× GPU", "FAILED (F)", "exit 1:0", "n-38-01 (GPU IDX 0,1)", "**restarts:** 2",
                 "batch step exited with code 1", "## Efficiency", "mostly idle", "## Cause", "OOM-GPU",
                 "## Last 30 log lines", "secrets scrubbed"):
        assert part in md, part
    assert "line 11" not in md and "line 12" in md                  # the last 30 lines only
    assert "0123456789abcdef0123456789abcdef01234567" not in md
    assert everyday.bundle_path(Path("/s/bundles"), JobKey("k2", "500_[1-3]")) == Path("/s/bundles/k2-500__1-3_.md")


def test_support_bundle_cause_without_log_or_detail() -> None:
    """No log text and no detail: the cause still comes from the scheduler state; completed jobs get none."""
    timeout = slurm_job("k2", "2331790", "train", JobState.TIMEOUT, ended_ago=60)
    md = everyday.support_bundle(timeout, None, cluster="k2", log_lines=[], now=T0)
    assert "## Cause" in md and "TIMEOUT: hit the time limit" in md and "## Efficiency" not in md
    done = slurm_job("k2", "2331791", "train", JobState.COMPLETED, exit_code=0, ended_ago=60)
    assert "## Cause" not in everyday.support_bundle(done, None, cluster="k2", log_lines=[], now=T0)


def test_free_gpus_rank_and_holders() -> None:
    svc = FakeService()
    snap = svc.store.hosts["labstation"].snapshot
    assert snap is not None
    free, held = everyday.free_gpus(snap.gpus, {j.key: j for j in snap.jobs.values()})
    assert [g.index for g in free] == [0, 2]                       # GPU 1: alice's eval; GPU 3: my training
    assert held == ["GPU 1 alice", "GPU 3 you (p363942)"]
    busy = dataclasses.replace(snap.gpus[0], util_pct=40.0, mem_used_mib=500.0)
    assert everyday.free_gpus([busy])[1][0].startswith("GPU 0 busy (40%")
    lighter = dataclasses.replace(snap.gpus[2], mem_used_mib=0.0)
    heavier = dataclasses.replace(snap.gpus[0], mem_used_mib=800.0)
    assert [g.index for g in everyday.free_gpus([heavier, lighter])[0]] == [2, 0]   # least memory first
    assert everyday.cuda_visible(free) == "CUDA_VISIBLE_DEVICES=0,2"


def test_forward_node() -> None:
    running = slurm_job("k2", "1", "x", JobState.RUNNING, node="n-40-01")
    assert everyday.forward_node(running) == ("n-40-01", None)
    assert everyday.forward_node(dataclasses.replace(running, state=JobState.PENDING))[0] is None
    assert everyday.forward_node(None) == ("localhost", None)
    task = task_job("lynx", 1, "t", elapsed=T0 - T0, gpus=(), gpu_model="", util=0, mem_mib=0, mem_total_mib=1,
                    pids=(1,))
    assert everyday.forward_node(task) == ("localhost", None)


# ====================================================================================================== pilot flows
async def test_copy_menu_shell_command_and_bundle(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    app, svc = make_app()
    app._state_path = tmp_path / "state" / "ui.json"               # bundles go to <state dir>/bundles
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None and job.logs[0].path is not None
    svc.logs[job.logs[0].path] = [f"step {i}" for i in range(50)] + [WANDB]
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app)
        await pilot.press("Y")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        assert menu_ids(app) == ["stdout", "stderr", "job_id", "host_job", "workdir", "shell", "bundle"]
        await choose(pilot, app, "shell")
        assert copied[-1] == "ssh -t k2 'srun --jobid=2332861 --overlap --pty -w n-40-01 bash -l'"
        assert "copied node shell command" in toasts(app)[-1]
        await pilot.press("Y")
        await pilot.pause()
        await choose(pilot, app, "bundle")
        await app.workers.wait_for_complete()
        await pilot.pause()
        bundle = copied[-1]
        assert bundle.startswith("# RunFleet support bundle · k2:2332861") and "step 49" in bundle
        assert "n-40-01 (GPU IDX 2)" in bundle and "0123456789abcdef0123456789abcdef01234567" not in bundle
        saved = tmp_path / "state" / "bundles" / "k2-2332861.md"
        assert saved.read_text() == bundle and (saved.stat().st_mode & 0o777) == 0o600
        assert "saved" in toasts(app)[-1]
        assert svc.called("job_detail")                            # on demand, on the keypress


async def test_copy_menu_keys_and_y_y_copies_the_stdout_path(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch,
                                                            tmp_path: Path) -> None:
    app, svc = make_app()
    app._state_path = tmp_path / "state" / "ui.json"
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None and job.logs[0].path is not None
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app)
        await pilot.press("Y")
        await pilot.pause()
        menu = app.screen.query_one(OptionList)
        assert menu.highlighted == 0 and menu_ids(app)[0] == "stdout"             # ↵ alone: the stdout path too
        assert str(menu.get_option_at_index(0).prompt).startswith("o stdout path · ")
        await pilot.press("Y")                                                    # Y Y: the old one-key copy
        await pilot.pause()
        assert copied == [job.logs[0].path] and not isinstance(app.screen, ActionMenuScreen)
        for key, want in (("o", job.logs[0].path), ("i", "2332861"), ("h", "k2:2332861"),
                          ("s", "ssh -t k2 'srun --jobid=2332861 --overlap --pty -w n-40-01 bash -l'")):
            await pilot.press("Y", key)
            await pilot.pause()
            assert copied[-1] == want and not isinstance(app.screen, ActionMenuScreen), key
        await pilot.press("Y", "b")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert copied[-1].startswith("# RunFleet support bundle · k2:2332861")


async def test_copy_menu_keys_queued_behind_a_close_are_dropped(make_app: MakeApp,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """``↵ Y`` / ``esc i`` arriving in one burst: the first key decides (↵ used to only post OptionSelected, which the
    accelerator overtook: stdout copied, and the stray ↵ then opened the job detail underneath)."""
    app, svc = make_app()
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app)
        screens = len(app.screen_stack)
        for burst, want in ((("enter", "Y"), ["2332861"]), (("escape", "i"), ["2332861"])):
            await pilot.press("Y")
            await pilot.pause()
            menu = app.screen.query_one(OptionList)
            menu.highlighted = menu_ids(app).index("job_id")
            for key in burst:
                app.post_message(events.Key(key, None))
            await pilot.pause()
            await pilot.pause()
            assert copied == want and len(app.screen_stack) == screens, burst
        await pilot.press("Y")                                                    # ↓ j ↵ in one burst: moves first
        await pilot.pause()
        assert menu_ids(app)[:3] == ["stdout", "stderr", "job_id"]
        for key in ("down", "j", "enter"):
            app.post_message(events.Key(key, None))
        await pilot.pause()
        await pilot.pause()
        assert copied == ["2332861", "2332861"] and len(app.screen_stack) == screens


async def test_copy_menu_without_a_stdout_path(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None
    monkeypatch.setattr(app, "_current_job", lambda: dataclasses.replace(job, logs=()))
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.action_copy_menu()
        await pilot.pause()
        menu = app.screen.query_one(OptionList)
        assert menu_ids(app)[menu.highlighted or 0] == "job_id"                   # first enabled entry
        await pilot.press("Y")                                                    # disabled: says why, stays open
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen) and not copied
        assert "no stdout path known" in toasts(app)[-1]
        await pilot.press("enter")
        await pilot.pause()
        assert copied == ["2332861"]


async def test_action_menu_accelerators_are_optional(make_app: MakeApp) -> None:
    app, _svc = make_app()
    specs = [ActionSpec("a.run", "Run", False, False), ActionSpec("a.off", "Off", False, False, disabled_reason="no"),
             ActionSpec("a.two", "Two", False, False)]
    results: list[str | None] = []
    with pytest.raises(ValueError, match="keys of the menu itself"):
        ActionMenuScreen(specs, 1, keys={"j": "a.run"})
    with pytest.raises(ValueError, match="name no entry"):
        ActionMenuScreen(specs, 1, keys={"x": "a.gone"})
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.push_screen(ActionMenuScreen(specs, 1, keys={"r": "a.run", "R": "a.run", "f": "a.off", "t": "a.two"}),
                        results.append)
        await pilot.pause()
        prompts = [str(o.prompt) for o in app.screen.query_one(OptionList).options]
        assert prompts == ["r Run", "t Two", "f Off  — no"]                       # shown: the first key of each
        await pilot.press("f")                                                    # disabled: toast, still open
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen) and results == [] and "Off — no" in toasts(app)[-1]
        await pilot.press("t")
        await pilot.pause()
        assert results == ["a.two"] and not isinstance(app.screen, ActionMenuScreen)
        app.push_screen(ActionMenuScreen(specs, 1, keys={"r": "a.run"}), results.append)
        await pilot.pause()
        await pilot.press("R")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)                           # not this menu's key
        await pilot.press("escape")
        await pilot.pause()
        app.push_screen(ActionMenuScreen(specs, 1), results.append)               # no keys: as before
        await pilot.pause()
        assert [str(o.prompt) for o in app.screen.query_one(OptionList).options] == ["Run", "Two", "Off  — no"]
        assert app.screen.query_one("#menu-box").border_subtitle == "↵ run · esc close"
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen) and results == ["a.two", None]
        await pilot.press("enter")
        await pilot.pause()
        assert results == ["a.two", None, "a.run"]


async def test_copy_uses_tmux_buffer_under_tmux(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _svc = make_app()
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    buffers: list[str] = []
    monkeypatch.setattr(everyday, "tmux_load_buffer", buffers.append)
    monkeypatch.setattr(app, "copy_to_clipboard", lambda text: None)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        everyday.copy_text(app, "abc", "job id")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert buffers == ["abc"] and "OSC 52 + tmux buffer" in toasts(app)[-1]


async def test_forward_a_port_from_the_log_and_cancel_it(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None and job.logs[0].path is not None
    svc.logs[job.logs[0].path] = [TB] + [f"epoch {i}" for i in range(20)]   # the URL is only near the top
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app)
        await pilot.press("P")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        assert menu_ids(app) == ["url:6006", "other"]               # found by the on-demand log fetch
        await choose(pilot, app, "url:6006")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert svc.called("forward_port") == [("k2", "n-40-01", 6006)]
        assert copied[-1] == "http://localhost:6006/" and "localhost:6006 → k2 n-40-01:6006" in toasts(app)[-1]
        await pilot.press("P")                                      # "Other port…": a typed number
        await app.workers.wait_for_complete()
        await pilot.pause()
        await choose(pilot, app, "other")
        assert isinstance(app.screen, PortPromptScreen)
        await pilot.press("0", "enter")
        assert isinstance(app.screen, PortPromptScreen)             # 0 is no port: stays open with a note
        app.screen.query_one(Input).value = "8888"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert svc.called("forward_port")[-1] == ("k2", "n-40-01", 8888)
        app.show_forwards()
        await pilot.pause()
        assert isinstance(app.screen, ForwardsScreen) and app.screen.query_one(OptionList).option_count == 2
        await pilot.press("ctrl+k")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert svc.called("cancel_forward") == [(6006,)] and len(svc.forwards()) == 1
        svc.cancel_error = "k2: port cancel failed: timed out"
        await pilot.press("ctrl+k")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "could not stop forwarding localhost:8888" in toasts(app)[-1] and len(svc.forwards()) == 1
        await pilot.press("escape")
        await pilot.pause()


async def test_forward_refused_for_pending_job_and_host_needing_auth(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app, "k2:2334564")                      # pending
        await pilot.press("P")
        await pilot.pause()
        assert "is not running" in toasts(app)[-1] and not isinstance(app.screen, ActionMenuScreen)
        lynx = next(j for j in svc.store.jobs(["lynx"]))
        app.screen.query_one(JobTable).move_to_key(str(lynx.key))
        await pilot.pause()
        await pilot.press("P")
        await pilot.pause()
        assert "port forward unavailable: lynx is not connected" in toasts(app)[-1]
        assert not svc.called("forward_port")


async def test_log_in_to_all_in_one_suspend(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    svc.inject("vega", "unreachable")
    batches: list[list[tuple[list[str], str, str | None]]] = []

    def run_seq(steps: list[tuple[list[str], str, str | None]]) -> list[int]:
        batches.append(list(steps))
        return [0, 255]

    monkeypatch.setattr(app, "run_external_sequence", run_seq)
    monkeypatch.setattr(app, "run_external", lambda *a, **kw: pytest.fail("one suspend for all hosts"))
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("C")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(batches) == 1 and [argv[-1] for argv, _b, _l in batches[0]] == ["vega", "lynx"]
        assert batches[0][0][1].startswith("RunFleet → 1/2 vega — approve the Duo push")
        assert batches[0][1][2] == "/tmp/rf-ssh-lynx.log"
        assert svc.called("after_connect") == [("vega", 0), ("lynx", 255)]
        assert "logged in: vega · failed: lynx (rc 255)" in toasts(app)
        svc.set_ok("lynx")
        await pilot.press("C")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(batches) == 1 and toasts(app)[-1] == "no host needs a login"


async def test_free_gpu_picker_in_the_gpus_view(make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    app, svc = make_app()
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    svc.tick(0)                                                    # fresh readings
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        view.table.move_cursor(row=next(i for i, r in enumerate(view.rows) if r.key == "gpu:labstation:1"))
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen) and menu_ids(app) == ["1", "2"]
        await choose(pilot, app, "2")
        assert copied == ["CUDA_VISIBLE_DEVICES=0,2"]
        assert "in use: GPU 1 alice; GPU 3 you (p363942)" in toasts(app)[-1]
        view.table.move_cursor(row=next(i for i, r in enumerate(view.rows) if r.host == "k2"))
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        assert "pick a workstation GPU" in toasts(app)[-1] and not isinstance(app.screen, ActionMenuScreen)


async def test_free_gpu_picker_refuses_stale_gpu_sections_and_rechecks_on_copy(
        make_app: MakeApp, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.models import SectionStatus

    app, svc = make_app()
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    svc.tick(0)
    hs = svc.store.hosts["labstation"]
    now = svc.clock.now()
    healthy = hs.status
    hs.status = dataclasses.replace(healthy, sections={
        "nvidia.gpu": SectionStatus("nvidia.gpu", "timeout", False, None, None, now - timedelta(minutes=5))})
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab", "tab")
        await pilot.pause()
        view = app.screen.query_one(GPUsView)
        view.table.move_cursor(row=next(i for i, r in enumerate(view.rows) if r.key == "gpu:labstation:1"))
        await pilot.pause()
        await pilot.press("a")                     # host fresh (ps, metrics answer), nvidia-smi hangs
        await pilot.pause()
        assert "GPU readings are stale" in toasts(app)[-1] and not isinstance(app.screen, ActionMenuScreen)
        hs.status = healthy
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ActionMenuScreen)
        snap = hs.snapshot
        assert snap is not None
        hs.snapshot = dataclasses.replace(snap, gpus=tuple(dataclasses.replace(g, util_pct=90.0) if g.index == 0
                                                           else g for g in snap.gpus))   # a poll: GPU 0 taken
        await choose(pilot, app, "1")
        assert copied == [] and "GPU 0 no longer free" in toasts(app)[-1]


async def test_mark_whole_array_then_cancel_targets_it(make_app: MakeApp) -> None:
    app, svc = make_app()
    for jid, state in (("31377076_1", JobState.RUNNING), ("31377076_2", JobState.RUNNING),
                       ("31377076_[3-9]", JobState.PENDING)):
        svc.put_job(dataclasses.replace(slurm_job("k2", jid, "sweep", state, node="n-31-01" if state is
                                                  JobState.RUNNING else None), array_job_id="31377076"))
    async with app.run_test(size=BIG) as pilot:
        await to_job(pilot, app, "k2:31377076_2")
        await pilot.press("A")
        await pilot.pause()
        view = app.screen.query_one(JobsView)
        assert view.marks == {"k2:31377076_1", "k2:31377076_2", "k2:31377076_[3-9]"}
        assert "marked 3 rows of array 31377076 on k2" in toasts(app)[-1]
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert svc.called("plan")[-1][1] == tuple(JobKey("k2", k.split(":")[1]) for k in sorted(view.marks))
        await pilot.press("n")
        await pilot.pause()
        await pilot.press("A")
        await pilot.pause()
        assert view.marks == set() and "unmarked 3 rows" in toasts(app)[-1]
        app.screen.query_one(JobTable).move_to_key(JOB)
        await pilot.pause()
        await pilot.press("A")
        await pilot.pause()
        assert "not part of an array" in toasts(app)[-1]


async def test_busy_controller_note_in_card_and_error_screen(make_app: MakeApp) -> None:
    app, svc = make_app()
    st = svc.store.hosts["k2"].status
    async with app.run_test(size=BIG) as pilot:
        svc.store.set_status("k2", dataclasses.replace(st, controller_slowdown=4.0))
        await pilot.pause(FLUSH)
        card = next(c for c in app.screen.query(HostCard) if c.host == "k2")
        assert "controller busy — slowed 4×" in str(card.render())
        app.push_screen(HostErrorScreen("k2"))
        await pilot.pause()
        assert "controller busy — slowed 4×" in str(app.screen.query_one("#err-body").render())
    assert busy_note(dataclasses.replace(st, controller_slowdown=1.0)) is None
