"""[WP2] The §6.2 task signal script against real throwaway processes: a queue runner that respawns its child when it
dies is frozen first, its re-enumerated tree is signalled, and nothing survives (DESIGN.md §10.4).

The runners are double-forked so they are re-parented away from pytest (otherwise a dead runner would stay a zombie
of the test process and `kill -0` would still see it).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from runfleet.backends.process.actions import interpret_lines, render_script
from runfleet.backends.process.collector import ProcessCollector
from runfleet.models import Job, JobKey, JobKind, JobState
from runfleet.probe.section import Section
from tests.backends.factories import ctx
from tests.integration.harness import MockEnv, run_script, script_for, shells

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

POLITE = "while :; do sleep 300 & wait $!; done\n"
# ignores TERM, and so does its child (ignored dispositions survive exec)
STUBBORN = "trap '' TERM INT\nwhile :; do sleep 300 & wait $!; done\n"
# survives TERM through a no-op handler; its child gets the default action, dies, and is respawned
RESPAWNING = "trap ':' TERM INT\nwhile :; do sleep 300 & wait $!; done\n"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _children(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    return [int(a) for a, b in (ln.split() for ln in out.splitlines()) if int(b) == pid]


def _start_epoch(pid: int) -> int:
    et = int(subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True).stdout)
    return int(time.time()) - et


class Runner:
    """A double-forked queue runner (`sh runner.sh`) with one `sleep` child at a time."""

    def __init__(self, tmp: Path, body: str, *, own_group: bool) -> None:
        script = tmp / f"runner{time.monotonic_ns()}.sh"
        script.write_text(body)
        out = subprocess.run(["sh", "-c", f"sh {script} >/dev/null 2>&1 </dev/null & echo $!"], capture_output=True,
                             text=True, start_new_session=own_group, check=True)
        self.pid = int(out.stdout.strip())
        deadline = time.monotonic() + 5
        while not _children(self.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.child = _children(self.pid)[0]

    def tree(self) -> list[int]:
        return [self.pid, *_children(self.pid)]

    def kill(self) -> None:
        for pid in self.tree():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def runners(tmp_path: Path) -> Iterator[list[Runner]]:
    made: list[Runner] = []
    yield made
    for r in made:
        r.kill()
        try:
            os.kill(r.child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run(tmp_path: Path, runner: Runner, sig: str, grace: int, escalate: bool, *, shell: str = "/bin/sh",
         start: int | None = None) -> list[str]:
    env = MockEnv(tmp_path, "lynx", mocks=(), sh=shell)
    st = _start_epoch(runner.pid) if start is None else start
    sec = Section(f"process.kill.{runner.pid}", render_script(runner.pid, st, sig, grace, escalate),
                  timeout=grace + 15)
    res = run_script(script_for([sec], "action"), env.env(), shell=shell)
    return res[sec.name].stdout.splitlines()


def _wait_gone(pids: list[int], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(_alive(p) for p in pids)


@pytest.mark.parametrize("shell", shells())
@pytest.mark.parametrize("own_group", [False, True])
def test_terminate_polite_queue_runner(tmp_path: Path, runners: list[Runner], shell: str, own_group: bool) -> None:
    r = Runner(tmp_path, POLITE, own_group=own_group)
    runners.append(r)
    before = r.tree()
    lines = _run(tmp_path, r, "TERM", 3, True, shell=shell)
    ok, message = interpret_lines(lines)
    assert ok, lines
    assert lines[0].split() == ["members", str(r.pid), str(r.child)]
    assert f"sent {r.child}" in lines and f"sent {r.pid}" in lines
    assert not any(ln.startswith(("alive", "new")) for ln in lines)
    assert _wait_gone(before)
    assert lines[-1] == "done"


def test_terminate_stubborn_runner_escalates(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, STUBBORN, own_group=True)
    runners.append(r)
    lines = _run(tmp_path, r, "TERM", 2, True)
    ok, message = interpret_lines(lines)
    assert ok, lines
    assert f"killed {r.pid}" in lines                    # TERM ignored → SIGKILL after the grace period
    assert not any(ln.startswith(("alive", "new")) for ln in lines)
    assert _wait_gone([r.pid, r.child])


def test_stubborn_without_escalation_reports_survivors(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, STUBBORN, own_group=False)
    runners.append(r)
    lines = _run(tmp_path, r, "TERM", 1, False)
    ok, message = interpret_lines(lines)
    assert not ok and f"alive {r.pid}" in lines and f"alive {r.child}" in lines
    assert message == "2 processes survived"


def test_respawned_child_is_reported_and_escalation_kills_it(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, RESPAWNING, own_group=False)
    runners.append(r)
    lines = _run(tmp_path, r, "TERM", 1, False)
    ok, message = interpret_lines(lines)
    # the root survived its TERM handler and, once continued, started the next item: reported, not hidden
    assert not ok and f"alive {r.pid}" in lines and any(ln.startswith("new ") for ln in lines)
    assert "started during termination" in message
    lines = _run(tmp_path, r, "TERM", 1, True)
    assert interpret_lines(lines)[0], lines
    assert _wait_gone([r.pid])


def test_kill_and_interrupt(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, POLITE, own_group=False)
    runners.append(r)
    lines = _run(tmp_path, r, "KILL", 0, True)
    assert interpret_lines(lines)[0] and _wait_gone([r.pid, r.child])
    r2 = Runner(tmp_path, POLITE, own_group=False)
    runners.append(r2)
    lines = _run(tmp_path, r2, "INT", 0, False)
    # sh ignores SIGINT in background jobs it starts (`sleep &`), so the child may survive an interrupt: the script
    # must say so instead of claiming success
    ok, _ = interpret_lines(lines)
    assert ok == (not any(ln.startswith(("alive", "new")) for ln in lines))


def test_reused_and_gone(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, POLITE, own_group=False)
    runners.append(r)
    start = _start_epoch(r.pid)
    lines = _run(tmp_path, r, "TERM", 1, True, start=start - 3600)
    assert lines == [f"reused {r.pid}", "done"] and interpret_lines(lines) == (False, "pid reused — refresh")
    assert _alive(r.pid)                                 # a reused pid is never signalled
    r.kill()
    assert _wait_gone([r.pid])
    lines = _run(tmp_path, r, "TERM", 1, True, start=start)
    assert lines == [f"gone {r.pid}", "done"] and interpret_lines(lines) == (True, "already exited")


def test_plan_through_collector_and_interpret(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, POLITE, own_group=True)
    runners.append(r)
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    job = Job(key=JobKey("lynx", f"p{r.pid}-{_start_epoch(r.pid)}"), kind=JobKind.TASK, name="runner",
              user="alice07", state=JobState.RUNNING, raw_state="running", job_id_raw=str(r.pid),
              pids=(r.pid, r.child), raw={"root_pid": str(r.pid), "root_start": str(_start_epoch(r.pid))})
    plan = c.plan_action("task.terminate", [job], {"grace_s": 2}, cx)
    env = MockEnv(tmp_path, "lynx", mocks=())
    res = run_script(script_for(plan.sections, "action"), env.env())
    (result,) = c.interpret_action(plan, res, cx)
    assert result.ok and result.message.startswith("signalled") and result.target == job.key
    assert _wait_gone([r.pid, r.child])


@pytest.mark.skipif(not (Path("/usr/bin/script").exists() and Path("/bin/bash").exists()),
                    reason="needs util-linux script and bash")
def test_foreground_job_of_an_interactive_shell_is_not_frozen(tmp_path: Path) -> None:
    """Review: SIGSTOP on a tty's foreground job makes bash print `[1]+ Stopped` and take the terminal back; the
    later SIGCONT leaves a survivor (INT handler) running as a background job, detached from its tmux pane."""
    target = tmp_path / "fg.sh"
    target.write_text("trap ':' INT\nwhile :; do sleep 1; done\n")
    shell = subprocess.Popen(["script", "-qfc", "bash --norc --noprofile -i", "/dev/null"], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    assert shell.stdin is not None
    try:
        shell.stdin.write(f"sh {target}\n".encode())
        shell.stdin.flush()
        root = None
        deadline = time.monotonic() + 10
        while root is None and time.monotonic() < deadline:
            out = subprocess.run(["ps", "-ww", "-e", "-o", "pid=,stat=,args="], capture_output=True, text=True).stdout
            root = next((int(ln.split()[0]) for ln in out.splitlines() if f"sh {target}" in ln and "+" in ln.split()[1]),
                        None)
            time.sleep(0.1)
        assert root is not None, "foreground target did not start"
        env = MockEnv(tmp_path, "lynx", mocks=())
        sec = Section(f"process.kill.{root}", render_script(root, _start_epoch(root), "INT", 1, False), timeout=20)
        lines = run_script(script_for([sec], "action"), env.env())[sec.name].stdout.splitlines()
        assert f"alive {root}" in lines                      # survived its INT handler, as intended
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(root)], capture_output=True, text=True).stdout.strip()
        assert "+" in stat and "T" not in stat, stat        # still the terminal's foreground job, not stopped
    finally:
        os.killpg(shell.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-KILL", "-f", f"sh {target}"], check=False)
        shell.wait()
