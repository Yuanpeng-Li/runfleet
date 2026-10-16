"""install.sh (§8.6): fresh install (uv missing / present), update through `runfleet update`, older versions.

The script runs under /bin/sh with a PATH of stub programs (uv, curl, runfleet, ssh) that log their arguments and a
few real tools; HOME is a temp dir, so nothing real is installed or downloaded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.posix                     # runs install.sh under /bin/sh with stub programs

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"
REPO = "git+https://github.com/Yuanpeng-Li/runfleet"

RUNFLEET = """#!/bin/sh
echo "runfleet $*" >> "$LOG"
case "$1 ${2:-}" in
  "--version ") echo "runfleet 9.9.9" ;;
  "update --help") if [ "${OLD:-}" = 1 ]; then echo "usage: runfleet [HOST ...]"           # `update` = a host name
                   else echo "usage: runfleet update [--check] [--force]"; fi ;;
esac
"""

UV = """#!/bin/sh
echo "uv $*" >> "$LOG"
case "$1 $2" in
  "tool install") mkdir -p "$HOME/.local/bin"; cp "$STUBS/runfleet.tmpl" "$HOME/.local/bin/runfleet"
                  chmod +x "$HOME/.local/bin/runfleet" ;;
  "tool dir") echo "$HOME/.local/bin" ;;
  "tool list") [ -n "${UV_HAS_RUNFLEET:-}" ] && printf 'runfleet v0.3.0\\n- runfleet\\n' ;;
esac
exit 0
"""

CURL = """#!/bin/sh
echo "curl $*" >> "$LOG"
printf 'mkdir -p "$HOME/.local/bin"; cp "$STUBS/uv.tmpl" "$HOME/.local/bin/uv"; chmod +x "$HOME/.local/bin/uv"\\n'
"""


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    stubs, tools, home = tmp_path / "stubs", tmp_path / "tools", tmp_path / "home"
    for d in (stubs, tools, home):
        d.mkdir()
    for name in ("sh", "grep", "cat", "cp", "chmod", "mkdir"):
        real = shutil.which(name)
        assert real is not None, name
        (tools / name).symlink_to(real)
    for name, text in (("runfleet.tmpl", RUNFLEET), ("uv.tmpl", UV), ("curl", CURL)):
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    (stubs / "ssh").write_text("#!/bin/sh\n")
    (stubs / "ssh").chmod(0o755)
    return {"PATH": f"{stubs}:{tools}", "HOME": str(home), "LOG": str(tmp_path / "log"), "STUBS": str(stubs)}


def run(env: dict[str, str]) -> tuple[str, list[str]]:
    done = subprocess.run(["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30,
                          executable=shutil.which("sh"))
    assert done.returncode == 0, done.stdout + done.stderr
    log = Path(env["LOG"])
    return done.stdout + done.stderr, log.read_text().splitlines() if log.exists() else []


def put(env: dict[str, str], name: str, where: str = "stubs") -> None:
    src = Path(env["STUBS"]) / f"{name}.tmpl"
    dst = Path(env["STUBS"] if where == "stubs" else env["HOME"]) / name
    shutil.copy(src, dst)
    dst.chmod(0o755)


def test_fresh_machine_gets_uv_then_runfleet(env: dict[str, str]) -> None:
    os.remove(Path(env["STUBS"]) / "ssh")
    out, calls = run(env)
    assert calls[0] == "curl -LsSf https://astral.sh/uv/install.sh"
    assert f"uv tool install {REPO}" in calls
    assert "uv tool update-shell" in calls                     # ~/.local/bin is not on this PATH
    assert "no ssh client" in out and "Open a new terminal" in out and "runfleet 9.9.9" in out


def test_uv_present_installs_runfleet_only(env: dict[str, str]) -> None:
    put(env, "uv")
    env["PATH"] = f"{env['HOME']}/.local/bin:{env['PATH']}"
    out, calls = run(env)
    assert not any(c.startswith("curl") for c in calls)
    assert calls[0] == f"uv tool install {REPO}"
    assert "uv tool update-shell" not in calls and "Start it with: runfleet" in out


def test_installed_runfleet_updates_itself(env: dict[str, str]) -> None:
    put(env, "runfleet")
    _out, calls = run(env)
    assert calls == ["runfleet update --help", "runfleet update"]


def test_older_runfleet_without_update_is_upgraded_by_uv(env: dict[str, str]) -> None:
    put(env, "runfleet")
    put(env, "uv")
    env |= {"OLD": "1", "UV_HAS_RUNFLEET": "1"}
    out, calls = run(env)
    assert "uv tool upgrade runfleet" in calls and "done: runfleet 9.9.9" in out
    assert not any(c.startswith("uv tool install") for c in calls)
