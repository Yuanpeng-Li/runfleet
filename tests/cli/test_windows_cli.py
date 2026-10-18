"""cli.py on native Windows (preview, DESIGN §1.2): terminal check, restart without exec, UTF-8 output, update by hand,
the doctor block and the connect wording. Runs anywhere: ``platform.IS_WINDOWS`` is patched, nothing is executed."""

from __future__ import annotations

import io
import signal
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet import __version__, cli, paths, update
from runfleet.util import platform
from tests.cli.test_cli import Harness, _drop_log_handlers, run

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
UV = update.Install("uv", "https://github.com/Yuanpeng-Li/runfleet", (("uv.exe", "tool", "upgrade", "runfleet"),),
                    "uv tool upgrade runfleet")


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    monkeypatch.setenv("WIN_PD_OVERRIDE_APPDATA", r"C:\Users\u\AppData\Roaming")       # platformdirs off Windows
    monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", r"C:\Users\u\AppData\Local")


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Harness]:
    _drop_log_handlers()
    yield Harness(monkeypatch, tmp_path)
    _drop_log_handlers()


class Tty(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self.tty = tty
        self.reconfigured: dict[str, Any] = {}

    def isatty(self) -> bool:
        return self.tty

    def reconfigure(self, **kw: Any) -> None:             # type: ignore[override]
        self.reconfigured = kw


# ---------------------------------------------------------------- terminal
def test_has_terminal_on_windows_checks_stdin_and_stdout_only(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def setup(stdin: bool, stdout: bool, stderr: bool) -> bool:
        monkeypatch.setattr(sys, "stdin", Tty(stdin))
        monkeypatch.setattr(sys, "stdout", Tty(stdout))
        monkeypatch.setattr(sys, "stderr", Tty(stderr))
        monkeypatch.delenv("TERM", raising=False)          # Windows consoles have no TERM
        return cli.has_terminal()

    assert setup(True, True, False)                        # `runfleet 2> log`: Textual draws on stdout there
    assert not setup(True, False, True)                    # `runfleet > file`
    assert not setup(False, True, True)                    # no keyboard


def test_utf8_output_only_for_redirected_streams_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    out, err = Tty(False), Tty(True)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    cli.utf8_output()
    assert out.reconfigured == {} and err.reconfigured == {}              # POSIX: nothing to do
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    cli.utf8_output()
    assert out.reconfigured == {"encoding": "utf-8", "errors": "replace"}  # `runfleet snapshot > x.txt`: ✓ → ·
    assert err.reconfigured == {}                                           # the console is UTF-8 already


# ---------------------------------------------------------------- restart
def test_restart_on_windows_runs_a_child_and_ignores_ctrl_c(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    handlers: list[tuple[int, Any]] = []
    calls: list[list[str]] = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.append((sig, handler)))
    monkeypatch.setattr(subprocess, "call", lambda argv: calls.append(list(argv)) or 3)
    monkeypatch.setattr(cli.os, "execv", lambda *a: pytest.fail("no exec on Windows"))
    assert cli.restart([r"C:\Users\u\.local\bin\runfleet.exe", "--demo"]) == 3       # its exit code is ours
    assert calls == [[r"C:\Users\u\.local\bin\runfleet.exe", "--demo"]]
    assert handlers == [(signal.SIGINT, signal.SIG_IGN)]

    def missing(argv: Any) -> int:
        raise FileNotFoundError(2, "not found")

    monkeypatch.setattr(subprocess, "call", missing)
    assert cli.restart(["runfleet.exe"]) == cli.EXIT_ERROR


def test_restart_on_posix_execs(monkeypatch: pytest.MonkeyPatch) -> None:
    execs: list[list[str]] = []

    def execv(path: str, argv: list[str]) -> None:
        execs.append(argv)
        raise OSError(2, "gone")

    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    monkeypatch.setattr(cli.os, "execv", execv)
    monkeypatch.setattr(subprocess, "call", lambda argv: pytest.fail("POSIX replaces the process"))
    assert cli.restart(["/x/runfleet", "k2"]) == cli.EXIT_ERROR and execs == [["/x/runfleet", "k2"]]


# ---------------------------------------------------------------- update by hand (§8.6)
class Updates:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, available: bool = True) -> None:
        self.steps: list[list[str]] = []
        self.restarted: list[list[str]] = []
        latest = "9.0.0" if available else __version__
        monkeypatch.setattr(update, "detect_install", lambda: UV)
        monkeypatch.setattr(update, "check", lambda now, *, force=False: update.Status(__version__, latest, T0))
        monkeypatch.setattr(update, "fetch_text", lambda name: "## [9.0.0] - 2027-01-01\n- shiny\n")
        monkeypatch.setattr(cli, "run_step", lambda argv, env: self.steps.append(list(argv)) or 0)
        monkeypatch.setattr(cli, "installed_version", lambda: "9.0.0")
        monkeypatch.setattr(cli, "restart", lambda argv: self.restarted.append(list(argv)) or 0)


@pytest.fixture
def u(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, windows: None) -> Iterator[Updates]:
    _drop_log_handlers()
    monkeypatch.setattr(cli.paths, "log_file", lambda: tmp_path / "runfleet.log")
    yield Updates(monkeypatch)
    _drop_log_handlers()


def test_update_on_windows_prints_the_steps_and_runs_nothing(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["update"], capsys)
    assert rc == cli.EXIT_ERROR and u.steps == []                     # nothing was updated
    assert "Version 9.0.0 is available." in out
    assert "1. quit RunFleet (q) in every window" in out and "2. run: uv tool upgrade runfleet" in out
    assert "Updated" not in out


def test_update_check_on_windows_says_by_hand(u: Updates, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["update", "--check"], capsys)
    assert rc == 0 and "Update by hand: quit RunFleet, then run uv tool upgrade runfleet" in out
    assert "Update with `runfleet update`" not in out


def test_tui_update_request_on_windows_never_restarts(u: Updates, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    h = Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_tui", lambda service, clock, **kw: h.tui.append(service) or update.RESTART_CODE)
    rc, out, _ = run(["--demo"], capsys)
    assert rc == cli.EXIT_ERROR and u.steps == [] and u.restarted == []
    assert "it starts again" not in out and "2. run: uv tool upgrade runfleet" in out


# ---------------------------------------------------------------- doctor, connect, help
def test_windows_report(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "ssh_version", lambda: "OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2")
    monkeypatch.setenv("WT_SESSION", "a1b2")
    lines = cli.windows_report()
    assert lines[0] == "windows (preview)"
    assert "  ✓ ssh -V  OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2" in lines
    assert "  ✓ console  Windows Terminal" in lines
    assert any("RUNFLEET_THEME=light" in x for x in lines)
    assert any("no ControlMaster" in x for x in lines) and any("mux: session" in x for x in lines)
    assert any("press c" in x for x in lines) and not any("WSL" in x for x in lines)
    monkeypatch.setattr(cli, "ssh_version", lambda: "OpenSSH_for_Windows_7.7p1, LibreSSL 2.6.5")
    monkeypatch.delenv("WT_SESSION")
    lines = cli.windows_report()
    assert any(x.startswith("  ✗ ssh -V") and "older than 8.1" in x for x in lines)
    assert any(x.startswith("  ! console  not Windows Terminal") for x in lines)
    monkeypatch.setattr(cli, "ssh_version", lambda: "OpenSSH_9.6p1, OpenSSL 3.2.1")
    assert any("not Windows OpenSSH" in x for x in cli.windows_report())
    monkeypatch.setattr(cli, "ssh_version", lambda: None)
    assert any("no ssh client" in x for x in cli.windows_report())


def test_doctor_shows_the_windows_block(h: Harness, windows: None, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "windows_report", lambda: ["windows (preview)", "  (test)"])
    rc, out, _ = run(["doctor", "--demo", "k2"], capsys)
    assert rc == 0 and out.splitlines()[:4] == ["terminal", "  (test)", "windows (preview)", "  (test)"]


def test_doctor_has_no_windows_block_elsewhere(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    monkeypatch.setattr(cli, "windows_report", lambda: pytest.fail("POSIX: no Windows block"))
    rc, out, _ = run(["doctor", "--demo", "k2"], capsys)
    assert rc == 0 and "windows" not in out


def test_ssh_version_reads_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert argv == ["ssh", "-V"]
        return subprocess.CompletedProcess(argv, 0, "", "OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2\n")

    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.ssh_version() == "OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2"

    def missing(argv: list[str], **kw: Any) -> None:
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(subprocess, "run", missing)
    assert cli.ssh_version() is None


def test_connect_on_windows_is_an_auth_check(h: Harness, windows: None, capsys: pytest.CaptureFixture[str]) -> None:
    """A session transport's login lives as long as the process that made it: `runfleet connect` only checks the
    login and says to press c in the TUI (§4.8); with mux = "off" it says what that means."""
    def session_host(svc: Any) -> None:
        svc.login = lambda: 0

    h.script = session_host
    rc, out, _ = run(["connect", "--demo", "lynx"], capsys)
    assert rc == 0 and "login works" in out and "press c" in out and "reuses" not in out
    h.script = None
    rc, out, _ = run(["connect", "--demo", "lynx"], capsys)
    assert rc == 0 and "login works" in out and 'mux = "off"' in out and "reuses" not in out


def test_probe_report_for_a_session_host_points_at_the_tui() -> None:
    """`hosts import` / first run: a session host that needs a login is logged in with c in RunFleet; `runfleet
    connect` would only check its auth (review: integration 5)."""
    from runfleet.models import ConnState, DetectReport

    session = DetectReport("lynx", ConnState.NEEDS_AUTH, None, None, "session", {}, None)
    assert "runfleet connect" not in cli.report_line(session, saved=True)
    assert "press c on it in RunFleet" in cli.report_line(session, saved=True)
    assert "save it, then press c" in cli.report_line(session, saved=False)
    managed = DetectReport("lynx", ConnState.NEEDS_AUTH, None, None, "managed", {}, None)
    assert "`runfleet connect lynx`" in cli.report_line(managed, saved=True)


def test_config_help_and_paths_name_the_real_location(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(paths, "config_file", lambda: tmp_path / "AppData" / "Roaming" / "runfleet" / "config.toml")
    text = cli.main_parser().format_help()
    assert "~\\AppData\\Roaming\\runfleet\\config.toml" in " ".join(text.split())
    assert cli.display_path(tmp_path / "x") == "~\\x"
    assert cli.shell_word("a b") == '"a b"' and cli.shell_word("k2") == "k2"
