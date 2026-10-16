"""cli.py: argument parsing, exit codes, first-run prompt, hosts/doctor/snapshot/connect against a fake FleetAPI."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli
from runfleet.config import FleetConfig, HostConfig
from runfleet.demo.transport import DemoTransport
from runfleet.errors import ConfigError
from runfleet.models import ConnState, DoctorCheck, DoctorReport, JobState
from runfleet.util import platform
from tests.cli.fakes import FakeService, host_state, job, report

DEMO_HOSTS = ["k2", "vega", "lyra", "labstation", "lynx"]


class Harness:
    """Patches every collaborator of cli.py; records the services it builds and the TUI launches."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.services: list[FakeService] = []
        self.tui: list[FakeService] = []
        self.answers: list[str] = []
        self.commands: list[list[str]] = []
        self.command_rc = 0
        self.script: Any = None              # callable(FakeService) to configure each new service
        self.config_path = tmp_path / "config.toml"
        self.log_path = tmp_path / "state" / "runfleet.log"
        monkeypatch.setattr(cli, "build_service", self._build)
        monkeypatch.setattr(cli, "run_tui", self._run_tui)
        monkeypatch.setattr(cli, "ssh_available", lambda: True)
        monkeypatch.setattr(cli, "is_interactive", lambda: True)
        monkeypatch.setattr(cli, "has_terminal", lambda: True)
        monkeypatch.setattr(cli, "ask", self._ask)
        monkeypatch.setattr(cli, "run_command", self._run_command)
        monkeypatch.setattr(cli, "terminal_report", lambda **kw: ["terminal", "  (test)"])   # never queries /dev/tty
        monkeypatch.setattr(cli, "windows_report", lambda: [])       # Windows: never runs `ssh -V`
        monkeypatch.setattr(cli.paths, "config_file", lambda: self.config_path)
        monkeypatch.setattr(cli.paths, "log_file", lambda: self.log_path)
        monkeypatch.setattr(cli.paths, "cache_file", lambda: tmp_path / "cache.sqlite")
        monkeypatch.setattr(cli.paths, "debug_dir", lambda: tmp_path / "debug")
        monkeypatch.setattr(cli.debugdump, "_dir", None)     # restored (disabled) after each test

    def _build(self, config: FleetConfig, transports: Any, clock: Any, cache: Any) -> FakeService:
        svc = FakeService(config, transports)
        if self.script is not None:
            self.script(svc)
        self.services.append(svc)
        return svc

    def _run_tui(self, service: FakeService, clock: Any, *, demo: bool = False, settings: bool = False) -> int:
        service.demo_mode = demo
        service.settings_open = settings
        self.tui.append(service)
        return 0

    def _ask(self, prompt: str) -> str:
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def _run_command(self, argv: Any) -> int:
        self.commands.append(list(argv))
        return self.command_rc


def _drop_log_handlers() -> None:
    root = logging.getLogger("runfleet")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Harness]:
    _drop_log_handlers()                    # setup_logging() is once-per-process; isolate each test's log file
    yield Harness(monkeypatch, tmp_path)
    _drop_log_handlers()


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = cli.main(args)
    out, err = capsys.readouterr()
    return rc, out, err


# ---------------------------------------------------------------------------------------------------- parsing
@pytest.mark.parametrize(("argv", "command"), [
    ([], None), (["k2", "lynx"], None), (["--demo"], None), (["snapshot", "--json"], "snapshot"),
    (["-c", "x.toml", "doctor", "k2"], "doctor"), (["--demo-seed", "3", "hosts", "list"], "hosts"),
    (["--", "doctor"], None), (["-c", "doctor"], None), (["--config=doctor.toml", "connect", "k2"], "connect"),
])
def test_find_command(argv: list[str], command: str | None) -> None:
    assert cli.find_command(argv) == command


def test_global_options_work_before_and_after_the_subcommand() -> None:
    before = cli.parse_args(["--demo", "--readonly", "snapshot", "--json", "k2"])
    after = cli.parse_args(["snapshot", "k2", "--demo", "--json"])
    for ns in (before, after):
        assert (ns.command, ns.demo, ns.json, ns.hosts) == ("snapshot", True, True, ["k2"])
    assert before.readonly and not after.readonly
    ns = cli.parse_args(["--", "doctor"])
    assert ns.command is None and ns.hosts == ["doctor"]
    ns = cli.parse_args(["hosts", "import", "--all", "--write"])
    assert (ns.hosts_command, ns.all, ns.write) == ("import", True, True)


def test_version_and_bad_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out.strip() == "runfleet 0.5.0"
    with pytest.raises(SystemExit) as exc:
        cli.main(["--frobnicate"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["hosts"])                               # missing list|import
    assert exc.value.code == 2


def test_python_m_runfleet_entry_point() -> None:
    out = subprocess.run([sys.executable, "-m", "runfleet", "--version"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "runfleet 0.5.0"


# ---------------------------------------------------------------------------------------------------- TUI start
def test_demo_tui_builds_the_demo_fleet(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, _ = run(["--demo", "--readonly", "--no-mouse"], capsys)
    assert rc == 0 and len(h.tui) == 1
    svc = h.tui[0]
    assert [x.name for x in svc.config.hosts] == DEMO_HOSTS and svc.config.source is None
    assert svc.config.general.readonly and not svc.config.general.mouse
    transport = svc.transports(svc.config.hosts[0]) if svc.transports else None
    assert isinstance(transport, DemoTransport) and transport.host == "k2"
    # leftover #1: the demo keeps UI state in memory (never the real ui.json) and takes no instance lock
    assert svc.demo_mode is True and svc.lock_path is False


def test_demo_subset_and_unknown_demo_host(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--demo", "lynx", "k2"], capsys)[0] == 0
    assert [x.name for x in h.tui[-1].config.hosts] == ["lynx", "k2"]
    rc, _, err = run(["--demo", "atlas"], capsys)
    assert rc == 2 and "not a demo host: atlas" in err


def test_demo_never_needs_ssh(h: Harness, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "ssh_available", lambda: False)
    assert run(["--demo"], capsys)[0] == 0
    rc, _, err = run(["k2"], capsys)
    assert rc == 3 and "no `ssh` client" in err


def test_config_warnings_become_one_notice(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    h.config_path.write_text('[[host]]\nname = "k2"\npoll_jobs = "2s"\ncolour = "red"\n', encoding="utf-8")
    assert run([], capsys)[0] == 0
    (notice,) = h.tui[-1].notices                     # §8.3: log + ONE toast
    assert notice.startswith("config: host[0].") and "(+1 more in the log)" in notice
    h.config_path.write_text('[[host]]\nname = "k2"\n', encoding="utf-8")
    assert run([], capsys)[0] == 0 and h.tui[-1].notices == []


def test_tui_needs_a_terminal(h: Harness, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "has_terminal", lambda: False)
    rc, out, _ = run(["--demo"], capsys)
    assert rc == 2 and "snapshot --json" in out and not h.tui


def test_no_config_non_interactive_is_a_usage_error(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "is_interactive", lambda: False)
    rc, out, _ = run([], capsys)
    assert rc == 2 and "runfleet hosts import" in out and not h.services


def test_missing_explicit_config_and_invalid_config(h: Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = run(["-c", str(tmp_path / "nope.toml")], capsys)
    assert rc == 2 and "no such config file" in err
    h.config_path.write_text("[[host]]\n", encoding="utf-8")

    def bad(path: Path | None, *, hosts: Any = ()) -> tuple[FleetConfig, tuple[str, ...]]:
        raise ConfigError("host[2].poll_jobs", 'expected duration like "15s", got 5')

    monkeypatch.setattr(cli, "load_config_with_warnings", bad)
    rc, _, err = run([], capsys)
    assert rc == 2 and 'host[2].poll_jobs: expected duration like "15s", got 5' in err


def test_configured_fleet_starts_with_flags_applied(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    h.config_path.write_text('[[host]]\nname = "k2"\n', encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake_load(path: Path | None, *, hosts: Any = ()) -> tuple[FleetConfig, tuple[str, ...]]:
        seen["args"] = (path, tuple(hosts))
        return FleetConfig(hosts=(HostConfig(name="k2", ssh="k2"), HostConfig(name="atlas", ssh="local"))), ()

    monkeypatch.setattr(cli, "load_config_with_warnings", fake_load)
    assert run(["--readonly"], capsys)[0] == 0
    assert seen["args"] == (None, ()) and h.tui[0].config.general.readonly
    assert run(["vega", "--no-mouse"], capsys)[0] == 0
    assert seen["args"] == (None, ("vega",)) and not h.tui[1].config.general.mouse


def test_config_without_hosts(h: Harness, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    h.config_path.write_text("[general]\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config_with_warnings", lambda path, hosts=(): (FleetConfig(hosts=()), ()))
    monkeypatch.setattr(cli, "is_interactive", lambda: False)
    rc, out, _ = run([], capsys)
    assert rc == 2 and "no hosts in" in out and not h.tui


def test_config_without_hosts_opens_settings(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """Every host unchecked in Settings: the next `runfleet` starts in Settings › Hosts instead of failing."""
    h.config_path.write_text("[general]\n", encoding="utf-8")
    rc, _out, _ = run([], capsys)
    assert rc == 0 and len(h.tui) == 1
    svc = h.tui[0]
    assert svc.settings_open is True and svc.config_path == h.config_path and not svc.remembered


# ---------------------------------------------------------------------------------------------------- first run
ALIASES = ["k2", "vega", "lyra", "labstation", "lynx", "jumpvm", "owl"]


def _first_run_setup(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sshconfig, "list_aliases", lambda: list(ALIASES))

    def script(svc: FakeService) -> None:
        svc.written_path = h.config_path
        svc.reports = {"k2": report("k2"), "vega": report("vega", slurm="25.11.4"),
                       "lyra": report("lyra", slurm="26.05.3", mux="managed", latency=1.02),
                       "labstation": report("labstation", slurm=None, driver="610.43.02", latency=0.08),
                       "lynx": report("lynx", ConnState.NEEDS_AUTH, error="permission denied")}
    h.script = script

    def fake_load(path: Path | None, *, hosts: Any = ()) -> tuple[FleetConfig, tuple[str, ...]]:
        names = [line.split('"')[1] for line in (path or h.config_path).read_text().splitlines() if "name" in line]
        return FleetConfig(hosts=tuple(HostConfig(name=n, ssh=n) for n in names), source=path), ()

    monkeypatch.setattr(cli, "load_config_with_warnings", fake_load)


def test_first_run_probes_writes_and_starts(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    _first_run_setup(h, monkeypatch)
    h.answers = ["1-5"]
    rc, out, _ = run([], capsys)
    assert rc == 0
    assert "No config yet" in out and "1 k2" in out and "7 owl" in out
    assert "Probing 5 hosts (read-only, BatchMode)" in out
    assert "slurm 25.05.8" in out and "mux managed" in out and "workstation (driver 610.43.02)" in out
    assert "lynx        needs auth  (saved; press c in RunFleet or run `runfleet connect lynx`)" in out
    assert "(5 hosts).\nFrom now on `runfleet` alone starts this fleet; add or remove hosts in Settings (press , in " \
        "RunFleet).\nStarting RunFleet" in out and "Not saved" not in out
    probe_svc = h.services[0]
    assert [a for a in probe_svc.called("probe_alias")] == ALIASES[:5]
    written, path = probe_svc.called("write_config")[0]
    assert [x.name for x in written] == ALIASES[:5] and path is None
    assert [x.name for x in h.tui[0].config.hosts] == ALIASES[:5]


def test_first_run_retries_bad_input_and_offers_demo(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    _first_run_setup(h, monkeypatch)
    h.answers = ["9", "nope", "d"]
    rc, out, _ = run([], capsys)
    assert rc == 0 and "no host number 9 (1-7)" in out and "unknown host 'nope'" in out
    assert [x.name for x in h.tui[0].config.hosts] == DEMO_HOSTS


def test_first_run_quit_and_eof(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    _first_run_setup(h, monkeypatch)
    h.answers = [""]
    assert run([], capsys)[0] == 0 and not h.services and not h.tui
    h.answers = []                                         # EOF (Ctrl-D)
    assert run([], capsys)[0] == 0 and not h.tui


def test_first_run_without_aliases_or_answers(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli.sshconfig, "list_aliases", lambda: [])
    rc, out, _ = run([], capsys)
    assert rc == 2 and "no Host aliases" in out
    _first_run_setup(h, monkeypatch)
    h.script = None                                        # every probe comes back unreachable
    h.answers = ["jumpvm,owl"]
    rc, out, _ = run([], capsys)
    assert rc == 1 and "Nothing to save" in out and "connection timed out (not saved)" in out
    assert "Add later with `runfleet hosts import jumpvm owl --write`" in out
    assert not h.services[0].called("write_config") and not h.tui


def test_parse_selection() -> None:
    aliases = ["a", "b", "c", "d"]
    assert cli.parse_selection("all", aliases) == aliases
    assert cli.parse_selection("1-2, d", aliases) == ["a", "b", "d"]
    assert cli.parse_selection("c 1 c", aliases) == ["c", "a"]
    for bad in ("0", "5", "x", " , "):
        with pytest.raises(cli.UsageError):
            cli.parse_selection(bad, aliases)


# ---------------------------------------------------------------------------------------------------- hosts
def test_hosts_import_writes_only_reachable(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    def script(svc: FakeService) -> None:
        svc.written_path = h.config_path
        svc.reports = {"k2": report("k2"), "odd alias": report("odd alias"),
                       "lynx": report("lynx", ConnState.HOSTKEY, error="host key unknown")}
    h.script = script
    rc, out, _ = run(["hosts", "import", "k2", "lynx", "github"], capsys)
    assert rc == 0 and "2 hosts usable; run again with --write" in out and not h.services[0].called("write_config")
    rc, out, _ = run(["hosts", "import", "k2", "lynx", "github", "odd alias", "--write"], capsys)
    written, _ = h.services[-1].called("write_config")[0]
    assert rc == 0 and [(x.name, x.ssh) for x in written] == [("k2", "k2"), ("lynx", "lynx"),
                                                               ("odd-alias", "odd alias")]
    assert "github      unreachable connection timed out (not saved)" in out and "Wrote " in out
    assert "Not saved (no answer now): github. Add later with `runfleet hosts import github --write`" in out


def test_hosts_import_reports_already_configured_and_long_aliases(
        h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    """Re-importing a configured alias must not claim it was written; long aliases keep a gap before the state."""
    def script(svc: FakeService) -> None:
        svc.written_path = h.config_path
        svc.reports = {"k2": report("k2"), "rf-blackholeunreachable": report("rf-blackholeunreachable"),
                       "vega": report("vega")}
    h.script = script
    h.config_path.write_text('[[host]]\nname = "k2"\nssh = "k2"\n', encoding="utf-8")
    rc, out, _ = run(["hosts", "import", "k2", "rf-blackholeunreachable", "vega", "--write",
                      "--config", str(h.config_path)], capsys)
    assert rc == 0
    assert "  rf-blackholeunreachable  ok" in out
    assert "(2 hosts: rf-blackholeunreachable, vega)" in out
    assert "Already configured in" in out and "(left unchanged): k2." in out
    h.config_path.write_text('[[host]]\nname = "k2"\nssh = "k2"\n', encoding="utf-8")
    rc, out, _ = run(["hosts", "import", "k2", "--write", "--config", str(h.config_path)], capsys)
    assert rc == 0 and "Wrote" not in out and "(left unchanged): k2." in out


def test_probe_aliases_reports_progress_in_completion_order() -> None:
    import asyncio

    class Svc:
        async def probe_alias(self, alias: str):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.02 if alias == "slow" else 0)
            return report(alias)

    seen: list[str] = []
    reports = asyncio.run(cli.probe_aliases(Svc(), ["slow", "fast"], lambda r: seen.append(r.alias)))  # type: ignore[arg-type]
    assert seen == ["fast", "slow"] and [r.alias for r in reports] == ["slow", "fast"]


def test_hosts_import_selection_errors(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["hosts", "import"], capsys)[0] == 2
    assert run(["hosts", "import", "--all", "k2"], capsys)[0] == 2
    monkeypatch.setattr(cli.sshconfig, "list_aliases", lambda: ["k2"])
    assert run(["hosts", "import", "--all"], capsys)[0] == 0
    assert h.services[-1].called("probe_alias") == ["k2"]
    monkeypatch.setattr(cli, "ssh_available", lambda: False)
    assert run(["hosts", "import", "k2"], capsys)[0] == 3


def test_hosts_import_demo_never_writes(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["hosts", "import", "--demo", "--write"], capsys)
    assert rc == 0 and "nothing written" in out and not h.services[0].called("write_config")
    assert h.services[0].called("probe_alias") == DEMO_HOSTS


def test_hosts_list(h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["hosts", "list"], capsys)
    assert rc == 0 and "No config at" in out and "`runfleet hosts import --all --write`" in out
    h.config_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda path, hosts=(): FleetConfig(
        hosts=(HostConfig(name="k2", ssh="k2", group="clusters"), HostConfig(name="box", ssh="local", enabled=False)),
        source=h.config_path))
    rc, out, _ = run(["hosts", "list"], capsys)
    lines = out.splitlines()
    assert rc == 0 and lines[0].startswith("2 hosts in ")
    assert lines[1].split() == ["#", "NAME", "SSH", "BACKEND", "MUX", "GROUP", "ENABLED"]
    assert lines[3].split() == ["2", "box", "local", "auto", "auto", "—", "no"]
    rc, out, _ = run(["hosts", "list", "--demo"], capsys)
    assert rc == 0 and out.startswith("5 hosts in demo fleet")


# ---------------------------------------------------------------------------------------------------- doctor
def test_doctor_output_and_exit_codes(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    def script(svc: FakeService) -> None:
        svc.doctor_reports = {
            "k2": DoctorReport("k2", (DoctorCheck("ssh -G", True, "user alice0001"), DoctorCheck("mux", None, "user"))),
            "lyra": DoctorReport("lyra", (DoctorCheck("gpu_tres", False, "no gres/gpuutil"),),
                                 warnings=("lyra: no GPU accounting — job GPU util unavailable",)),
            "vega": RuntimeError("boom"),
        }
    h.script = script
    rc, out, _ = run(["doctor", "--demo", "k2"], capsys)
    assert rc == 0
    assert out.splitlines() == ["terminal", "  (test)", "k2", "  ✓ ssh -G  user alice0001", "  · mux     user"]
    rc, out, _ = run(["doctor", "--demo", "--reset", "k2", "lyra", "vega"], capsys)
    assert rc == 4
    assert "  ✗ gpu_tres  no gres/gpuutil" in out and "  ! lyra: no GPU accounting" in out
    assert "vega\n  ✗ doctor  internal error: RuntimeError('boom')" in out
    assert h.services[-1].called("doctor") == [("k2", True), ("lyra", True), ("vega", True)]


# ---------------------------------------------------------------------------------------------------- snapshot
def test_snapshot_json_and_failed_host_exit_code(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    def script(svc: FakeService) -> None:
        svc.collected = {"k2": host_state("k2", jobs=[job("k2", "2332861"), job("k2", "1", JobState.FAILED)]),
                         "vega": host_state("vega")}
    h.script = script
    rc, out, _ = run(["snapshot", "--demo", "--json", "k2", "vega", "--timeout", "12"], capsys)
    doc = json.loads(out)
    assert rc == 0 and doc["schema"] == "runfleet.snapshot/1"
    assert [x["name"] for x in doc["hosts"]] == ["k2", "vega"]
    assert [j["key"]["id"] for j in doc["hosts"][0]["jobs"]] == ["1", "2332861"]
    assert h.services[0].called("collect_once") == [(["k2", "vega"], 12.0)]

    def failing(svc: FakeService) -> None:
        svc.collected = {"k2": host_state("k2"), "lyra": host_state("lyra", ConnState.UNREACHABLE,
                                                                    error="connection timed out")}
    h.script = failing
    rc, out, _ = run(["snapshot", "--demo", "k2", "lyra"], capsys)
    assert rc == 4
    lines = out.splitlines()
    assert lines[0].split() == ["HOST", "STATE", "MY", "JOBS", "MY", "GPUS", "NODES", "LATENCY"]
    assert lines[2].startswith("lyra") and "unreachable (connection timed out)" in lines[2]


# ---------------------------------------------------------------------------------------------------- connect
def test_connect(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["connect", "--demo", "lynx"], capsys)
    assert rc == 0 and h.commands == [["ssh", "-fN", "x"]] and "connecting to lynx" in out
    # the mux plan is resolved (and a stale socket cleared) BEFORE the argv is built (review core: an unresolved
    # transport always built the user form, so managed hosts like lyra got no socket and a leaked session)
    assert [n for n, _ in h.services[-1].calls][:2] == ["prepare_connect", "connect_argv"]

    def connected(svc: FakeService) -> None:
        svc.already_connected = True
    h.script = connected
    rc, out, _ = run(["connect", "--demo", "lynx"], capsys)
    assert rc == 0 and "already connected" in out and len(h.commands) == 1
    h.script = None
    h.command_rc = 255
    rc, out, _ = run(["connect", "--demo", "lynx"], capsys)
    assert rc == 1 and "ssh failed (exit 255)" in out

    def local(svc: FakeService) -> None:
        svc.connect = None
    h.script = local
    rc, out, _ = run(["connect", "--demo", "k2"], capsys)
    assert rc == 0 and "nothing to connect" in out


# ---------------------------------------------------------------------------------------------------- robustness
def test_unexpected_errors_become_exit_1(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    def broken(service: Any, clock: Any, **kw: Any) -> int:
        raise ValueError("bug")

    monkeypatch.setattr(cli, "run_tui", broken)
    rc, _, err = run(["--demo"], capsys)
    assert rc == 1 and "internal error: ValueError('bug')" in err
    with pytest.raises(ValueError):
        cli.main(["--demo", "--debug"])
    assert h.log_path.is_file() and "unexpected error" in h.log_path.read_text()


def test_demo_log_records_are_tagged(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    """M3: a simulated cancel in --demo must not read like a real one in runfleet.log."""
    run(["hosts", "list", "--demo"], capsys)
    logging.getLogger("runfleet.core.actions").info("action slurm.cancel k2:3370390 on k2: ok")
    _drop_log_handlers()
    run(["hosts", "list"], capsys)
    logging.getLogger("runfleet.core.actions").info("action real")
    text = h.log_path.read_text()
    assert "INFO [demo] runfleet.core.actions: action slurm.cancel k2:3370390" in text
    assert "INFO runfleet.core.actions: action real" in text


def test_keyboard_interrupt_is_130(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    def interrupted(service: Any, clock: Any, **kw: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_tui", interrupted)
    assert run(["--demo"], capsys)[0] == 130


def test_open_cache_never_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert cli.open_cache(True) is None

    class Broken:
        def __init__(self, path: Path) -> None:
            raise RuntimeError("corrupt")

    monkeypatch.setattr(cli, "Cache", Broken)
    monkeypatch.setattr(cli.paths, "cache_file", lambda: tmp_path / "c.sqlite")
    assert cli.open_cache(False) is None


# ---------------------------------------------------------------------------------------------------- review 2026-10
def test_options_between_host_positionals() -> None:
    """BACK-42: `runfleet k2 --readonly vega` / `snapshot k2 --json vega` exited 2 (unrecognized arguments: vega)."""
    ns = cli.parse_args(["k2", "--readonly", "vega"])
    assert ns.command is None and ns.hosts == ["k2", "vega"] and ns.readonly
    ns = cli.parse_args(["snapshot", "k2", "--json", "vega"])
    assert (ns.command, ns.hosts, ns.json) == ("snapshot", ["k2", "vega"], True)
    ns = cli.parse_args(["hosts", "import", "k2", "--write", "vega"])
    assert (ns.aliases, ns.write) == (["k2", "vega"], True)
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["connect", "k2", "vega"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        cli.parse_args(["snapshot", "k2", "--frobnicate"])


def test_abbreviated_options_are_rejected() -> None:
    """BACK-43: `--conf x.toml snapshot --json` made x.toml the command position and rejected --json."""
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--conf", "x.toml", "snapshot", "--json"])
    assert exc.value.code == 2
    ns = cli.parse_args(["--config", "x.toml", "snapshot", "--json"])
    assert ns.command == "snapshot" and ns.json and ns.config == Path("x.toml")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "soon"])
def test_snapshot_timeout_must_be_a_positive_number(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """BACK-40: `--timeout 0` failed every host at once; `inf` removed the bound."""
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["snapshot", "--timeout", value])
    assert exc.value.code == 2 and "positive number of seconds" in capsys.readouterr().err
    assert cli.parse_args(["snapshot", "--timeout", "2.5"]).timeout == 2.5


def test_terminal_check_uses_the_streams_textual_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    """BACK-37: Textual reads stdin and draws on stderr; the check looked at stdout only and ignored TERM=dumb."""
    monkeypatch.setattr(platform, "IS_WINDOWS", False)       # the POSIX behaviour (Windows: test_windows_*.py)
    class Tty:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    def setup(stdin: bool, stdout: bool, stderr: bool, term: str | None) -> bool:
        monkeypatch.setattr(sys, "stdin", Tty(stdin))
        monkeypatch.setattr(sys, "stdout", Tty(stdout))
        monkeypatch.setattr(sys, "stderr", Tty(stderr))
        if term is None:
            monkeypatch.delenv("TERM", raising=False)
        else:
            monkeypatch.setenv("TERM", term)
        return cli.has_terminal()

    assert setup(True, False, True, "xterm-256color")              # `runfleet > file` works
    assert not setup(False, True, True, "xterm-256color")          # `runfleet < /dev/null`: no keyboard
    assert not setup(True, True, False, "xterm-256color")          # `runfleet 2> log`: escapes into the file
    assert not setup(True, True, True, "dumb") and not setup(True, True, True, None)


@pytest.mark.parametrize(("stdout_tty", "terminal", "queried"), [
    (True, True, True),
    (False, True, False),            # `runfleet doctor | less`: the pager would read the reply
    (True, False, False),            # no keyboard / no terminal for Textual's streams
])
def test_terminal_report_asks_the_terminal_only_when_both_ends_are_it(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout_tty: bool, terminal: bool, queried: bool) -> None:
    """The real terminal_report (the CLI harness replaces it): the OSC 11 query, the saved choice and the result."""
    import runfleet.ui.app as app_module
    from runfleet.ui import termbg, theme

    asked: list[float] = []

    def query(timeout: float = termbg.TIMEOUT, **_kw: Any) -> str:
        asked.append(timeout)
        return "#fdf6e3"

    class Out:
        def isatty(self) -> bool:
            return stdout_tty

    state = tmp_path / "ui.json"
    state.write_text(json.dumps({"theme": "auto"}), encoding="utf-8")
    monkeypatch.setattr(termbg, "query_background", query)
    monkeypatch.setattr(cli, "has_terminal", lambda: terminal)
    monkeypatch.setattr(sys, "stdout", Out())
    monkeypatch.setattr(app_module, "ui_state_path", lambda: state)
    monkeypatch.setattr(theme, "color_system", lambda env=None: "truecolor")
    for var in ("RUNFLEET_THEME", "COLORFGBG", "TERM_PROGRAM", "LC_TERMINAL"):
        monkeypatch.delenv(var, raising=False)
    lines = cli.terminal_report()
    assert len(asked) == int(queried) and lines[0] == "terminal"
    if queried:
        assert "background #fdf6e3 · theme auto → light (terminal background)" in lines[2]
    else:
        assert "background not reported · theme auto → dark (default)" in lines[2]
    state.write_text(json.dumps({"theme": "ansi"}), encoding="utf-8")
    assert "theme ansi → ansi (setting)" in cli.terminal_report()[2]
    assert "theme auto → " in cli.terminal_report(demo=True)[2]         # --demo keeps no UI state


def test_connect_by_ssh_alias_uses_the_configured_name(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    """BACK-07: `runfleet connect k2` with `name = "research"`, `ssh = "k2"` failed with unknown host 'k2'."""
    h.config_path.write_text('[[host]]\nname = "research"\nssh = "k2"\n', encoding="utf-8")
    rc, out, _ = run(["connect", "k2"], capsys)
    assert rc == 0 and "connecting to k2" in out
    assert [c for c in h.services[-1].calls if c[0] in ("prepare_connect", "connect_argv")] == [
        ("prepare_connect", "research"), ("connect_argv", "research")]
    rc, _, _ = run(["connect", "usr@box.example.org"], capsys)              # ad-hoc target: sanitized name
    assert rc == 0 and h.services[-1].calls[0] == ("prepare_connect", "usr_box.example.org")


def test_snapshot_and_doctor_skip_disabled_hosts(h: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    """BACK-36: a disabled ("listed but not polled") host was still probed by snapshot/doctor and failed them."""
    h.config_path.write_text('[[host]]\nname = "k2"\n\n[[host]]\nname = "old"\nenabled = false\n', encoding="utf-8")

    def script(svc: FakeService) -> None:
        svc.collected = {"k2": host_state("k2")}
    h.script = script
    rc, _, _ = run(["snapshot"], capsys)
    assert rc == 0 and h.services[-1].called("collect_once") == [(["k2"], 60.0)]
    run(["snapshot", "old"], capsys)                                        # named explicitly: probed
    assert h.services[-1].called("collect_once") == [(["old"], 60.0)]
    rc, _, _ = run(["doctor"], capsys)
    assert rc == 0 and h.services[-1].called("doctor") == [("k2", False)]


def test_hosts_list_with_a_missing_explicit_config(h: Harness, tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """BACK-44: `hosts list -c /typo.toml` printed an empty listing and exited 0."""
    rc, _, err = run(["hosts", "list", "-c", str(tmp_path / "typo.toml")], capsys)
    assert rc == 2 and "no such config file" in err


def test_broken_pipe_is_not_an_internal_error(h: Harness, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """BACK-45: `runfleet snapshot --json | head -1` reported an internal error."""
    def closed(*args: Any, **kw: Any) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    dups: list[tuple[int, int]] = []
    monkeypatch.setitem(cli.COMMANDS, "snapshot", closed)
    monkeypatch.setattr(cli.os, "dup2", lambda a, b: dups.append((a, b)))     # never touch the real fd 1 (`-s`)
    monkeypatch.setattr(sys.stdout, "fileno", lambda: 1, raising=False)
    rc, _, err = run(["snapshot", "--demo"], capsys)
    assert rc == 141 and "internal error" not in err and [b for _, b in dups] == [1]
