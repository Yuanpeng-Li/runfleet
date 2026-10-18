"""`runfleet update` and the TUI's "update and restart" (§8.6) with fake installs, checks and commands, and the agent
skill refresh after an update (`update.refresh_skill`, with a fake new executable)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet import __version__, cli, update
from runfleet.util import platform
from tests.cli.test_cli import Harness, _drop_log_handlers, run


@pytest.fixture(autouse=True)
def _posix_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests pin the POSIX behaviour (pure UI/CLI logic, valid on any OS); the Windows branches are tested in
    test_windows_cli.py / test_windows_ui.py / tests/unit/test_platform.py."""
    monkeypatch.setattr(platform, "IS_WINDOWS", False)


T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
UV = update.Install("uv", "https://github.com/Yuanpeng-Li/runfleet", (("/bin/uv", "tool", "upgrade", "runfleet"),),
                    "uv tool upgrade runfleet", env={"UV_TOOL_DIR": "/tools"})


class Updates:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.install = UV
        self.status = update.Status(__version__, "9.0.0", T0)
        self.steps: list[tuple[list[str], dict[str, str]]] = []
        self.step_rc = 0
        self.new_version: str | None = "9.0.0"
        self.restarted: list[list[str]] = []
        self.forced: list[bool] = []
        self.skill_line: str | None = "runfleet skill updated (0.7.2 → 9.0.0) in ~/.agents/skills/runfleet"
        self.skill_refreshes = 0
        monkeypatch.setattr(update, "detect_install", lambda: self.install)
        monkeypatch.setattr(update, "check", self._check)
        monkeypatch.setattr(update, "fetch_text", lambda name: "## [9.0.0] - 2027-01-01\n- shiny\n")
        monkeypatch.setattr(cli, "run_step", self._step)
        monkeypatch.setattr(cli, "installed_version", lambda: self.new_version)
        monkeypatch.setattr(cli, "restart", self._restart)
        monkeypatch.setattr(cli, "refresh_skill", self._refresh_skill)

    def _restart(self, argv: Any) -> int:
        self.restarted.append(list(argv))
        return 0

    def _refresh_skill(self) -> str | None:
        assert not self.restarted                          # before the new RunFleet starts
        self.skill_refreshes += 1
        return self.skill_line

    def _check(self, now: datetime, *, force: bool = False) -> update.Status:
        self.forced.append(force)
        return self.status

    def _step(self, argv: Any, env: Any) -> int:
        self.steps.append((list(argv), dict(env)))
        return self.step_rc


@pytest.fixture
def u(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Updates]:
    _drop_log_handlers()
    monkeypatch.setattr(cli.paths, "log_file", lambda: tmp_path / "runfleet.log")
    yield Updates(monkeypatch)
    _drop_log_handlers()


def test_update_runs_the_install_command_and_shows_what_changed(u: Updates, capsys: pytest.CaptureFixture[str]
                                                                 ) -> None:
    rc, out, _ = run(["update"], capsys)
    assert rc == 0 and u.forced == [True]
    assert u.steps == [(["/bin/uv", "tool", "upgrade", "runfleet"], {"UV_TOOL_DIR": "/tools"})]
    assert "Version 9.0.0 is available." in out and "  - shiny" in out
    assert "$ uv tool upgrade runfleet" in out and f"Updated: {__version__} → 9.0.0." in out
    assert "quit it (q) and start it again" in out


def test_up_to_date_does_nothing_unless_forced(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    u.status = update.Status(__version__, __version__, T0)
    rc, out, _ = run(["update"], capsys)
    assert rc == 0 and u.steps == [] and "This is the newest version" in out
    u.new_version = __version__
    rc, out, _ = run(["update", "--force"], capsys)
    assert rc == 0 and len(u.steps) == 1 and f"still {__version__}" in out


def test_check_only_looks(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["update", "--check"], capsys)
    assert rc == 0 and u.steps == [] and "Update with `runfleet update`." in out
    u.status = update.Status(__version__, None, None, error="Network is unreachable")
    rc, out, _ = run(["update", "--check"], capsys)
    assert rc == 1 and "Network is unreachable" in out


def test_offline_needs_force(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    u.status = update.Status(__version__, None, None, error="timed out")
    rc, out, _ = run(["update"], capsys)
    assert rc == 1 and u.steps == [] and "--force" in out
    rc, _out, _ = run(["update", "--force"], capsys)
    assert rc == 0 and len(u.steps) == 1


def test_failed_step_and_installs_that_cannot_update_themselves(u: Updates, capsys: pytest.CaptureFixture[str]
                                                                 ) -> None:
    u.step_rc = 2
    rc, out, _ = run(["update"], capsys)
    assert rc == 1 and "uv exited with 2" in out and "by hand: uv tool upgrade runfleet" in out
    u.install = update.Install("uvx", "", (), "uvx --refresh --from git+https://x runfleet")
    rc, out, _ = run(["update"], capsys)
    assert rc == 1 and "cannot update itself" in out and "uvx --refresh" in out
    rc, out, _ = run(["update", "--check"], capsys)
    assert rc == 0 and "(or by hand: uvx --refresh" in out


def test_tui_update_request_updates_then_restarts_with_the_same_arguments(
        u: Updates, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h = Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_tui", lambda service, clock, **kw: h.tui.append(service) or update.RESTART_CODE)
    rc, out, _ = run(["--demo", "--readonly"], capsys)
    assert rc == 0 and len(u.steps) == 1 and "it starts again when done" in out
    assert u.restarted == [update.relaunch_argv(["--demo", "--readonly"])]
    u.restarted.clear()
    u.step_rc = 1
    rc, _out, _ = run(["--demo"], capsys)
    assert rc == 1 and u.restarted == []                  # a failed update leaves the terminal with its output


# ---------------------------------------------------------------- the agent skill after an update (§8.7)
def test_successful_update_refreshes_the_skill_with_the_new_code(u: Updates, capsys: pytest.CaptureFixture[str]
                                                                 ) -> None:
    rc, out, _ = run(["update"], capsys)
    assert rc == 0 and u.skill_refreshes == 1
    assert out.index("Updated: ") < out.index("runfleet skill updated (0.7.2 → 9.0.0)")
    u.skill_line = None                                     # nothing installed: nothing said
    rc, out, _ = run(["update", "--force"], capsys)
    assert rc == 0 and u.skill_refreshes == 2 and "skill" not in out


def test_no_skill_refresh_when_nothing_was_updated(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    u.step_rc = 1
    run(["update"], capsys)
    run(["update", "--check"], capsys)
    u.status = update.Status(__version__, __version__, T0)
    run(["update"], capsys)
    assert u.skill_refreshes == 0


def test_tui_update_refreshes_the_skill_before_restarting(
        u: Updates, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h = Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_tui", lambda service, clock, **kw: h.tui.append(service) or update.RESTART_CODE)
    rc, out, _ = run(["--demo"], capsys)
    assert rc == 0 and u.skill_refreshes == 1 and len(u.restarted) == 1 and "runfleet skill updated" in out


FAKE_RUNFLEET = """#!/bin/sh
echo "$*" >> "{log}"
case "${{FAKE:-ok}}" in
  ok) echo "some noise"; echo "runfleet skill updated (0.7.2 → 9.0.0) in ~/.agents/skills/runfleet" ;;
  silent) ;;
  fail) echo "runfleet: could not write ~/.agents/skills/runfleet" >&2; exit 1 ;;
  old) echo "usage: runfleet [-h] ..." >&2; echo "runfleet: error: argument COMMAND: invalid choice: 'skill'" >&2
       exit 2 ;;
  slow) sleep 5 ;;
esac
"""


@pytest.fixture
def fake_runfleet(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "calls.log"
    exe = tmp_path / "bin" / "runfleet"
    exe.parent.mkdir()
    exe.write_text(FAKE_RUNFLEET.format(log=log), encoding="utf-8")
    exe.chmod(0o755)
    return exe, log


@pytest.mark.posix
@pytest.mark.parametrize(("mode", "line"), [
    ("ok", "runfleet skill updated (0.7.2 → 9.0.0) in ~/.agents/skills/runfleet"),
    ("silent", None),
    ("fail", "Agent skill not refreshed (exit 1); run `runfleet skill update`."),
    ("old", None),                                          # a version without `skill update`: nothing to refresh
])
def test_refresh_skill_runs_the_new_executable(fake_runfleet: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                               mode: str, line: str | None) -> None:
    exe, log = fake_runfleet
    monkeypatch.setenv("FAKE", mode)
    assert update.refresh_skill(argv0=str(exe), executable="/no/python") == line
    assert log.read_text(encoding="utf-8").splitlines() == ["skill update --if-installed --quiet"]


@pytest.mark.posix
def test_refresh_skill_finds_runfleet_on_path_and_gives_up_after_its_timeout(
        fake_runfleet: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    exe, log = fake_runfleet
    found = update.refresh_skill(argv0="runfleet", executable="/no/python", which=lambda name: str(exe))
    assert found is not None and found.startswith("runfleet skill updated") and log.exists()
    monkeypatch.setenv("FAKE", "slow")
    assert update.refresh_skill(argv0=str(exe), timeout=0.3) == \
        "Agent skill not refreshed (timed out); run `runfleet skill update`."


def test_refresh_skill_never_raises() -> None:
    assert update.refresh_skill(argv0="", executable=os.path.join(os.sep, "no", "such", "python")) is None


def test_refresh_skill_of_a_source_checkout_uses_python_m_runfleet(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kw: Any) -> Any:
        seen.append(argv)
        assert kw["timeout"] == update.SKILL_REFRESH_TIMEOUT and kw["stdin"] is not None
        return type("Done", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(update.subprocess, "run", fake_run)
    assert update.refresh_skill(argv0="/x/runfleet/__main__.py", executable=sys.executable) is None
    assert seen == [[sys.executable, "-m", "runfleet", "skill", "update", "--if-installed", "--quiet"]]
