"""install.sh (§8.6): fresh install (uv missing / present), update through `runfleet update`, older versions, the
agent skill refresh after each of them (§8.7) and `--uninstall`.

The script runs under /bin/sh with a PATH of stub programs (uv, curl, runfleet, ssh) that log their arguments and a
few real tools; HOME is a temp dir, so nothing real is installed or downloaded.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.posix                     # runs install.sh under /bin/sh with stub programs

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"
REPO = "git+https://github.com/Yuanpeng-Li/runfleet"

RUNFLEET = """#!/bin/sh
echo "runfleet $*" >> "$LOG"
[ -f "$STUBS/upgraded" ] && OLD=""                                      # `uv tool upgrade` installed a new version
case "$*" in
  "--version") echo "runfleet 9.9.9" ;;
  "update --help") if [ "${OLD:-}" = 1 ]; then echo "usage: runfleet [HOST ...]"           # `update` = a host name
                   else echo "usage: runfleet update [--check] [--force]"; fi ;;
  "skill update --help") if [ -n "${OLD:-}" ]; then echo "usage: runfleet [HOST ...]"      # no `skill update` yet
                         else echo "usage: runfleet skill update [--if-installed] [--quiet]"; fi ;;
  "skill uninstall --help") [ -z "${OLD:-}" ] && echo "usage: runfleet skill uninstall [--dry-run]" ;;
  "skill update --if-installed") echo "Skill: refreshed to 9.9.9"; exit "${SKILL_RC:-0}" ;;
  "skill uninstall") echo "Removed the runfleet skill"; exit "${SKILL_RC:-0}" ;;
  "update") exit "${UPDATE_RC:-0}" ;;
esac
exit 0
"""

UV = """#!/bin/sh
echo "uv $*" >> "$LOG"
case "$1 $2" in
  "tool install") mkdir -p "$HOME/.local/bin"; cp "$STUBS/runfleet.tmpl" "$HOME/.local/bin/runfleet"
                  chmod +x "$HOME/.local/bin/runfleet" ;;
  "tool dir") echo "$HOME/.local/bin" ;;
  "tool list") [ -n "${UV_HAS_RUNFLEET:-}" ] && printf 'runfleet v0.3.0\\n- runfleet\\n' ;;
  "tool upgrade") touch "$STUBS/upgraded" ;;
  "tool uninstall") rm -f "$HOME/.local/bin/runfleet" "$STUBS/runfleet" ;;
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
    for name in ("sh", "grep", "cat", "cp", "chmod", "mkdir", "rm", "touch", "uname", "sed"):
        real = shutil.which(name)
        assert real is not None, name
        (tools / name).symlink_to(real)
    for name, text in (("runfleet.tmpl", RUNFLEET), ("uv.tmpl", UV), ("curl", CURL)):
        (stubs / name).write_text(text)
        (stubs / name).chmod(0o755)
    (stubs / "ssh").write_text("#!/bin/sh\n")
    (stubs / "ssh").chmod(0o755)
    return {"PATH": f"{stubs}:{tools}", "HOME": str(home), "LOG": str(tmp_path / "log"), "STUBS": str(stubs)}


def run(env: dict[str, str], *args: str, rc: int = 0) -> tuple[str, list[str]]:
    done = subprocess.run(["sh", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30,
                          executable=shutil.which("sh"))
    assert done.returncode == rc, done.stdout + done.stderr
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


SKILL_REFRESH = ["runfleet skill update --help", "runfleet skill update --if-installed"]


def test_installed_runfleet_updates_itself_then_refreshes_the_skill(env: dict[str, str]) -> None:
    put(env, "runfleet")
    out, calls = run(env)
    assert calls == ["runfleet update --help", "runfleet update", *SKILL_REFRESH]
    assert "Skill: refreshed to 9.9.9" in out


def test_failed_update_still_refreshes_and_keeps_its_exit_code(env: dict[str, str]) -> None:
    put(env, "runfleet")
    env |= {"UPDATE_RC": "3", "SKILL_RC": "1"}                   # a failing skill refresh never fails the script
    _out, calls = run(env, rc=3)
    assert calls == ["runfleet update --help", "runfleet update", *SKILL_REFRESH]


def test_older_runfleet_without_update_is_upgraded_by_uv_then_refreshes(env: dict[str, str]) -> None:
    put(env, "runfleet")
    put(env, "uv")
    env |= {"OLD": "1", "UV_HAS_RUNFLEET": "1"}
    out, calls = run(env)
    assert "uv tool upgrade runfleet" in calls and "done: runfleet 9.9.9" in out
    assert not any(c.startswith("uv tool install") for c in calls)
    assert calls.index("uv tool upgrade runfleet") < calls.index("runfleet skill update --if-installed")


def test_a_new_version_without_skill_update_is_not_asked_to_refresh(env: dict[str, str]) -> None:
    put(env, "runfleet")
    env["OLD"] = "2"                                     # has `update`, not `skill update` (0.7.x)
    _out, calls = run(env)
    assert calls == ["runfleet update --help", "runfleet update", "runfleet skill update --help"]


def test_fresh_install_refreshes_a_skill_left_from_before(env: dict[str, str]) -> None:
    put(env, "uv")
    env["PATH"] = f"{env['HOME']}/.local/bin:{env['PATH']}"
    _out, calls = run(env)
    assert calls[0] == f"uv tool install {REPO}" and calls[-3:] == [*SKILL_REFRESH, "runfleet --version"]


# ---------------------------------------------------------------- --uninstall
def test_uninstall_removes_the_skill_then_the_tool_and_keeps_user_dirs(env: dict[str, str], tmp_path: Path) -> None:
    put(env, "runfleet")
    put(env, "uv")
    config, state = tmp_path / "o'brien config" / "runfleet", tmp_path / "state" / "runfleet"
    config.mkdir(parents=True)
    state.mkdir(parents=True)
    env |= {"UV_HAS_RUNFLEET": "1", "XDG_CONFIG_HOME": str(config.parent), "XDG_STATE_HOME": str(state.parent),
            "XDG_CACHE_HOME": str(tmp_path / "no-cache")}
    out, calls = run(env, "--uninstall")
    assert calls == ["runfleet skill uninstall --help", "runfleet skill uninstall", "uv tool list",
                     "uv tool uninstall runfleet"]
    assert "Removed the runfleet skill" in out and not (Path(env["STUBS"]) / "runfleet").exists()
    quoted = "'" + str(config).replace("'", "'\\''") + f"' '{state}'"
    assert f"kept your config and RunFleet's state (hosts, cache, log): {quoted}" in out
    assert f"to delete them too: rm -rf {quoted}" in out and "no-cache" not in out
    line = out.split("to delete them too: ", 1)[1].splitlines()[0]
    assert shlex.split(line) == ["rm", "-rf", str(config), str(state)]        # pastes into a shell as is


def test_uninstall_without_skill_support_or_runfleet(env: dict[str, str]) -> None:
    put(env, "runfleet")
    put(env, "uv")
    env |= {"OLD": "1", "UV_HAS_RUNFLEET": "1"}                            # a version without `skill uninstall`
    _out, calls = run(env, "--uninstall")
    assert calls == ["runfleet skill uninstall --help", "uv tool list", "uv tool uninstall runfleet"]
    del env["UV_HAS_RUNFLEET"]
    out, calls = run(env, "--uninstall")                                   # nothing left: says so, exit 0
    assert "RunFleet is not installed" in out and "kept" not in out


def test_uninstall_of_a_copy_uv_and_pipx_do_not_know_fails(env: dict[str, str]) -> None:
    put(env, "runfleet")
    out, calls = run(env, "--uninstall", rc=1)
    assert "runfleet skill uninstall" in calls and "remove it the way you installed it" in out


def test_unknown_option_is_refused(env: dict[str, str]) -> None:
    out, calls = run(env, "--bogus", rc=1)
    assert "unknown option: --bogus" in out and calls == []
