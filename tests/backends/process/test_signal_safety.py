"""[WP2] Review 2026-10 (CORE-02/03, BACK-02/03): the §6.2 signal script never leaves the launcher stopped when it
dies while the root is frozen, finishes its escalation when the channel is gone, and never signals a pid whose start
time changed since the members were recorded (pid reuse). Real throwaway processes, as in tests/integration.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from runfleet.backends.process.actions import render_script
from tests.integration.test_signal_script import (POLITE, STUBBORN, Runner, _alive, _start_epoch, _wait_gone,
                                                  runners)

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
__all__ = ["runners"]                                   # the fixture, re-exported for pytest


def _state(pid: int) -> str:
    with open(f"/proc/{pid}/stat") as f:
        return f.read().rsplit(")", 1)[1].split()[0]


def _slow_ps_path(tmp_path: Path) -> str:
    """PATH whose `ps -e …` (the tree walk, the first call after SIGSTOP) takes 2 s."""
    real = shutil.which("ps")
    assert real is not None
    bindir = tmp_path / "slowbin"
    bindir.mkdir()
    ps = bindir / "ps"
    ps.write_text(f'#!/bin/sh\ncase " $* " in *" -e "*) sleep 2 ;; esac\nexec {real} "$@"\n')
    ps.chmod(0o755)
    return f"{bindir}{os.pathsep}{os.environ['PATH']}"


def test_killed_while_frozen_continues_the_root(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, POLITE, own_group=True)
    runners.append(r)
    script = render_script(r.pid, _start_epoch(r.pid), "INT", 0, False)
    proc = subprocess.Popen(["sh", "-c", script], env={**os.environ, "PATH": _slow_ps_path(tmp_path)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 5
    while _state(r.pid) != "T" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _state(r.pid) == "T"                          # inside the STOP…CONT window
    proc.send_signal(signal.SIGTERM)                     # the section timeout / a dropped session
    assert proc.wait(timeout=10) == 143
    assert _state(r.pid) != "T"                          # the EXIT trap continued it
    assert _alive(r.child)                               # and nothing was signalled


def test_closed_channel_does_not_abort_the_escalation(tmp_path: Path, runners: list[Runner]) -> None:
    r = Runner(tmp_path, STUBBORN, own_group=True)
    runners.append(r)
    script = render_script(r.pid, _start_epoch(r.pid), "TERM", 1, True)
    proc = subprocess.Popen(["sh", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    proc.stdout.close()                                  # every `echo` now hits a broken pipe
    proc.wait(timeout=20)
    assert _wait_gone([r.pid, r.child])                  # KILL still reached the TERM-ignoring tree


def test_reused_root_pid_is_neither_killed_nor_walked(tmp_path: Path, runners: list[Runner]) -> None:
    """The root's pid now names a different process (recorded start time differs): it gets no SIGKILL, its tree is
    not walked for newcomers, it is not reported; the recorded child (still the same process) is killed."""
    r = Runner(tmp_path, STUBBORN, own_group=False)
    runners.append(r)
    script = render_script(r.pid, _start_epoch(r.pid), "TERM", 1, True)
    marker = """IDS=$(for p in $R $SET; do printf '%s:%s ' "$p" "$(st "$p")"; done)\n"""
    assert marker in script
    script = script.replace(marker, marker + 'IDS=$(echo "$IDS" | sed "s/^$R:[0-9]*/$R:1/")\n')
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=20).stdout.splitlines()
    assert f"killed {r.child}" in out and f"killed {r.pid}" not in out, out
    assert not any(ln.startswith(("alive", "new")) for ln in out), out
    assert _alive(r.pid) and _wait_gone([r.child])


def test_reused_child_pid_gets_no_initial_signal(tmp_path: Path, runners: list[Runner]) -> None:
    """A member whose pid was reused between the tree walk and the first signal (recorded start time differs) is not
    sent the initial signal either; the root still is."""
    r = Runner(tmp_path, STUBBORN, own_group=False)
    runners.append(r)
    script = render_script(r.pid, _start_epoch(r.pid), "INT", 0, False)
    marker = """IDS=$(for p in $R $SET; do printf '%s:%s ' "$p" "$(st "$p")"; done)\n"""
    assert marker in script
    forged = f'IDS=$(echo "$IDS" | sed "s/ {r.child}:[0-9]*/ {r.child}:1/")\n'
    out = subprocess.run(["sh", "-c", script.replace(marker, marker + forged)], capture_output=True, text=True,
                         timeout=20).stdout.splitlines()
    assert f"sent {r.pid}" in out and f"sent {r.child}" not in out, out
    assert f"alive {r.child}" not in out, out
