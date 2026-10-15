"""`runfleet update` and the TUI's "update and restart" (§8.6) with fake installs, checks and commands."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet import __version__, cli, update
from tests.cli.test_cli import Harness, _drop_log_handlers, run

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
        monkeypatch.setattr(update, "detect_install", lambda: self.install)
        monkeypatch.setattr(update, "check", self._check)
        monkeypatch.setattr(update, "fetch_text", lambda name: "## [9.0.0] - 2027-01-01\n- shiny\n")
        monkeypatch.setattr(cli, "run_step", self._step)
        monkeypatch.setattr(cli, "installed_version", lambda: self.new_version)
        monkeypatch.setattr(cli, "restart", lambda argv: self.restarted.append(list(argv)) or 0)

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
