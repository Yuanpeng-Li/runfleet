"""[WP4] argparse CLI, subcommands, first-run prompt; builds FleetService and runs the app (DESIGN.md §8.4-8.5).

    runfleet [HOST ...] [-c PATH] [--demo] [--demo-seed N] [--readonly] [--no-mouse] [--no-cache] [--debug]
    runfleet connect HOST
    runfleet hosts list | import [--all | ALIAS ...] [--write]
    runfleet doctor [HOST ...] [--reset]
    runfleet snapshot [HOST ...] [--json] [--timeout SEC]
    runfleet wait HOST:JOBID ... [--timeout SEC] [--json]
    runfleet update [--check] [--force]

Exit codes: 0 ok, 1 runtime error, 2 config/usage error, 3 no `ssh` client (never with --demo), 4 `snapshot` /
`doctor`: at least one host failed. `wait` has its own (runfleet.wait): 0 completed, 1 ended otherwise, 2 timeout,
3 error.

Collaborators are module-level functions (`build_service`, `run_tui`, `open_cache`, `ssh_available`,
`is_interactive`, `run_command`, `run_step`, `installed_version`, `restart`) so tests replace them with fakes; nothing
here touches the network itself (`update` asks GitHub through `runfleet.update`).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import logging.handlers
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from runfleet import __version__, debugdump, export, paths, sshconfig, update
from runfleet.config import FleetConfig, HostConfig, host_name_for, load_config, load_config_with_warnings
from runfleet.core.cache import Cache
from runfleet.core.clock import Clock, RealClock
from runfleet.core.service import FleetAPI, FleetService, TransportFactory
from runfleet.errors import ConfigError, RunFleetError
from runfleet.models import ConnState, DetectReport, DoctorReport

log = logging.getLogger("runfleet.cli")

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NO_SSH, EXIT_HOST_FAILED = 0, 1, 2, 3, 4
SUBCOMMANDS = ("connect", "hosts", "doctor", "snapshot", "wait", "update")
_VALUE_OPTIONS = frozenset({"-c", "--config", "--demo-seed", "--timeout"})
_WRITABLE = frozenset({ConnState.OK, ConnState.DEGRADED, ConnState.NEEDS_AUTH, ConnState.HOSTKEY})


class UsageError(RunFleetError):
    """Bad command line or selection (exit code 2)."""


# ====================================================================================================== collaborators
def build_service(config: FleetConfig, transports: TransportFactory, clock: Clock, cache: Cache | None) -> FleetAPI:
    return FleetService(config, transports, clock, cache)


def run_tui(service: FleetAPI, clock: Clock, *, demo: bool = False, settings: bool = False) -> int:
    """`settings`: open the Settings overlay on its Hosts tab at once (a config without hosts)."""
    from runfleet.ui.app import RunFleetApp     # Textual is imported only when the TUI really starts

    initial = None
    if demo:                                    # a watched job and a last visit, so alerts and `D` show at once
        from runfleet.demo.world import demo_ui_state
        initial = demo_ui_state(clock.now())
    app = RunFleetApp(service, clock=clock, persist_state=not demo, open_settings=settings, initial_state=initial,
                      check_updates=not demo and service.config.general.update_check)
    app.run()
    return int(app.return_code or 0)     # update.RESTART_CODE: "update and restart" (Settings › About)


def open_cache(disabled: bool) -> Cache | None:
    if disabled:
        return None
    try:
        return Cache(paths.cache_file())
    except Exception as exc:                    # a broken cache never prevents a start (§9 #27)
        log.warning("cache unavailable, running without it: %s", exc)
        return None


def ssh_available() -> bool:
    return shutil.which("ssh") is not None


def is_interactive() -> bool:
    """Both ends are a terminal: the first-run prompt may ask questions."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def has_terminal() -> bool:
    """The TUI can run: Textual reads keys from stdin and draws on stderr (not stdout), and needs a real TERM."""
    return sys.stdin.isatty() and sys.stderr.isatty() and os.environ.get("TERM", "") not in ("", "dumb")


def run_command(argv: Sequence[str]) -> int:
    """Run an interactive command in this terminal (connect)."""
    return subprocess.run(list(argv), check=False).returncode


def ask(prompt: str) -> str:
    return input(prompt)


def run_step(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """One update command in this terminal (its output shows as it runs)."""
    try:
        return subprocess.run(list(argv), check=False, env={**os.environ, **env}).returncode
    except OSError as exc:
        print(f"runfleet: cannot run {argv[0]}: {exc}", file=sys.stderr, flush=True)
        return 127


def installed_version() -> str | None:
    """The version a fresh start would run (after an update this process still is the old one)."""
    try:
        done = subprocess.run(update.relaunch_argv(["--version"]), capture_output=True, text=True, timeout=60,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.fullmatch(r"runfleet (\S+)", done.stdout.strip())
    return m.group(1) if done.returncode == 0 and m else None


def restart(argv: Sequence[str]) -> int:
    """Replace this process with a fresh RunFleet (returns only when that is impossible)."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execv(argv[0], list(argv))
    except OSError as exc:
        print(f"runfleet: cannot start {argv[0]}: {exc}", file=sys.stderr, flush=True)
    return EXIT_ERROR


# ====================================================================================================== parsing
def _common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Global options, accepted before and after a subcommand. Subcommand copies default to SUPPRESS so they do
    not clobber values given before the subcommand (argparse sub-parser defaults would)."""
    def d(value: Any) -> Any:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument("-c", "--config", type=Path, metavar="PATH", default=d(None),
                        help="config file (default: ~/.config/runfleet/config.toml)")
    parser.add_argument("--demo", action="store_true", default=d(False), help="simulated fleet, no network")
    parser.add_argument("--demo-seed", type=int, metavar="N", default=d(7), help="seed of the demo world (default 7)")
    parser.add_argument("--readonly", action="store_true", default=d(False), help="disable every mutating action")
    parser.add_argument("--no-mouse", action="store_true", default=d(False), help="no mouse capture (text selection)")
    parser.add_argument("--no-cache", action="store_true", default=d(False), help="do not read or write the cache")
    parser.add_argument("--debug", action="store_true", default=d(False),
                        help="debug logging and raw dumps of every probe (state dir: debug/)")


def _positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number of seconds, got {text!r}")
    return value


# allow_abbrev=False everywhere: find_command() skips the values of exactly these option spellings (`--conf x.toml
# snapshot` must not make `x.toml` the command position)
def main_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runfleet", description="Jobs, GPUs and hosts of your Slurm clusters and GPU "
                                "workstations in one terminal UI.",
                                epilog="subcommands: connect, hosts, doctor, snapshot, wait, update (runfleet <cmd> -h)",
                                allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"runfleet {__version__}")
    _common(p, suppress=False)
    p.add_argument("hosts", nargs="*", metavar="HOST", help="ssh aliases or config names (default: all configured)")
    return p


def command_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runfleet", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"runfleet {__version__}")
    _common(p, suppress=False)
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    c = sub.add_parser("connect", allow_abbrev=False, help="log in once (Duo/password prompts here) so RunFleet can reuse it")
    _common(c, suppress=True)
    c.add_argument("host", metavar="HOST")
    h = sub.add_parser("hosts", allow_abbrev=False, help="list configured hosts or import them from ~/.ssh/config")
    _common(h, suppress=True)
    hsub = h.add_subparsers(dest="hosts_command", required=True, metavar="list|import")
    hl = hsub.add_parser("list", allow_abbrev=False, help="configured hosts")
    _common(hl, suppress=True)
    hi = hsub.add_parser("import", allow_abbrev=False, help="probe ssh aliases (read-only) and optionally write them to the config")
    _common(hi, suppress=True)
    hi.add_argument("aliases", nargs="*", metavar="ALIAS")
    hi.add_argument("--all", action="store_true", help="every Host alias in ~/.ssh/config")
    hi.add_argument("--write", action="store_true", help="write/append [[host]] tables to the config")
    d = sub.add_parser("doctor", allow_abbrev=False, help="diagnose ssh, multiplexing, detection and capabilities per host")
    _common(d, suppress=True)
    d.add_argument("hosts", nargs="*", metavar="HOST")
    d.add_argument("--reset", action="store_true", help="close RunFleet-managed ssh masters first")
    s = sub.add_parser("snapshot", allow_abbrev=False, help="poll once and print the fleet (table, or JSON with --json)")
    _common(s, suppress=True)
    s.add_argument("hosts", nargs="*", metavar="HOST")
    s.add_argument("--json", action="store_true", help='JSON document, schema "runfleet.snapshot/1"')
    s.add_argument("--timeout", type=_positive_seconds, default=60.0, metavar="SEC",
                   help="per-host budget (default 60)")
    w = sub.add_parser("wait", allow_abbrev=False, help="wait until jobs end (exit 0 all completed, 1 otherwise, "
                       "2 timeout, 3 error); read-only")
    _common(w, suppress=True)
    w.add_argument("hosts", nargs="+", metavar="HOST:JOBID", help="k2:2332861, k2:31270818 (whole array), lynx:p642229")
    w.add_argument("--timeout", type=_positive_seconds, default=None, metavar="SEC", help="give up after SEC (exit 2)")
    w.add_argument("--json", action="store_true", help='one JSON object per line, schema "runfleet.wait/1"')
    u = sub.add_parser("update", allow_abbrev=False, help="update RunFleet to the newest version on GitHub")
    _common(u, suppress=True)
    u.add_argument("--check", action="store_true", help="only say whether a newer version exists")
    u.add_argument("--force", action="store_true", help="update even when no newer version is known")
    return p


def find_command(argv: Sequence[str]) -> str | None:
    """The subcommand named by the first positional token, if any (`--` ends options: `runfleet -- doctor` monitors
    a host called doctor)."""
    skip = False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok == "--":
            return None
        if tok.startswith("-"):
            skip = tok in _VALUE_OPTIONS
            continue
        return tok if tok in SUBCOMMANDS else None
    return None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Options may sit between host positionals (`runfleet k2 --readonly vega`, `snapshot k2 --json vega`)."""
    command = find_command(argv)
    if command is None:
        ns = main_parser().parse_intermixed_args(list(argv))
        ns.command = None
        return ns
    parser = command_parser()         # parse_intermixed_args does not support sub-parsers: collect the leftovers
    ns, extra = parser.parse_known_args(list(argv))
    dest = "aliases" if getattr(ns, "hosts_command", None) == "import" else "hosts"
    if extra and isinstance(getattr(ns, dest, None), list) and not any(t.startswith("-") for t in extra):
        getattr(ns, dest).extend(extra)
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return ns


# ====================================================================================================== helpers
class PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """runfleet.log and its rotations are 0600 (job names, paths and command lines), whatever the umask."""

    def _open(self):
        paths.private_file(Path(self.baseFilename))
        return super()._open()


def log_format(demo: bool) -> logging.Formatter:
    """`--demo` records carry a `[demo]` marker: a simulated `action slurm.cancel k2:…` must never read like a real
    one in runfleet.log (M3)."""
    tag = "[demo] " if demo else ""
    return logging.Formatter(f"%(asctime)s %(levelname)s {tag}%(name)s: %(message)s")


def setup_logging(debug: bool, demo: bool = False) -> None:
    """Rotating file log (5 MB x 3); never the terminal, which belongs to the TUI."""
    root = logging.getLogger("runfleet")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    existing = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    if existing:
        for h in existing:
            h.setFormatter(log_format(demo))
        return
    try:
        path = paths.log_file()
        paths.ensure_private_dir(path.parent)          # the state dir: 0700 whatever the umask
        handler = PrivateRotatingFileHandler(path, maxBytes=5 << 20, backupCount=3, encoding="utf-8")
    except (OSError, NotImplementedError) as exc:
        root.addHandler(logging.NullHandler())
        log.debug("no log file: %s", exc)
        return
    handler.setFormatter(log_format(demo))
    root.addHandler(handler)
    if debug:                                   # raw probe dumps (wire format) under the state dir, rotated per host
        debugdump.enable(paths.debug_dir())
        log.info("debug: raw probe dumps in %s", paths.debug_dir())


def display_path(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def default_config_path(args: argparse.Namespace) -> Path:
    return args.config if args.config is not None else paths.config_file()


@dataclasses.dataclass
class Fleet:
    """What a command runs against: the config, the transport factory and (for --demo) the world."""

    config: FleetConfig
    transports: TransportFactory
    world: Any = None
    warnings: tuple[str, ...] = ()     # config validation warnings (already logged); the TUI shows one toast


def apply_flags(config: FleetConfig, args: argparse.Namespace) -> FleetConfig:
    general = dataclasses.replace(config.general, readonly=config.general.readonly or args.readonly,
                                  mouse=config.general.mouse and not args.no_mouse)
    return dataclasses.replace(config, general=general)


def demo_fleet(args: argparse.Namespace, hosts: Sequence[str], clock: Clock) -> Fleet:
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld

    world = DemoWorld(seed=args.demo_seed, now=clock.now)
    try:
        config = world.fleet_config(hosts)
    except KeyError as exc:
        raise UsageError(str(exc.args[0])) from None
    return Fleet(apply_flags(config, args), lambda h: DemoTransport(h, world), world)


def real_fleet(args: argparse.Namespace, hosts: Sequence[str]) -> Fleet:
    from runfleet.transport import make_transport

    if args.config is not None and not args.config.exists():
        raise UsageError(f"no such config file: {args.config}")
    config, warnings = load_config_with_warnings(args.config, hosts=hosts)
    for w in warnings:
        log.warning("config: %s", w)
    return Fleet(apply_flags(config, args), make_transport, warnings=warnings)


def config_notice(warnings: Sequence[str]) -> str | None:
    """§8.3: config warnings are logged and shown as ONE toast."""
    if not warnings:
        return None
    more = f" (+{len(warnings) - 1} more in the log)" if len(warnings) > 1 else ""
    return f"config: {warnings[0]}{more}"


def needs_ssh(config: FleetConfig) -> bool:
    return any(h.ssh != "local" for h in config.hosts)


def no_ssh_message() -> str:
    return "runfleet: no `ssh` client found on PATH (install OpenSSH, or try `runfleet --demo`)"


class NoSSH(RunFleetError):
    """No local OpenSSH client (exit code 3)."""


def ensure_ssh(fleet: Fleet, args: argparse.Namespace) -> None:
    if not args.demo and needs_ssh(fleet.config) and not ssh_available():
        raise NoSSH()


# ====================================================================================================== probing
def kind_text(report: DetectReport) -> str:
    caps = report.caps
    if caps is None:
        return ""
    if "slurm" in caps.profile:
        return f"slurm {caps.slurm_version or '?'}"
    return f"workstation (driver {caps.driver_version})" if caps.driver_version else "workstation"


def alias_width(aliases: Sequence[str]) -> int:
    """Alias column width: the longest alias (at least 10) plus a 2-space gap, so long aliases never run into the
    state column."""
    return max([10] + [len(a) for a in aliases]) + 2


def report_line(report: DetectReport, *, saved: bool | None, width: int = 12) -> str:
    state = report.state.value.replace("_", " ")
    name = report.alias.ljust(max(width, len(report.alias) + 2))
    if report.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
        note = "saved; press c in RunFleet or run" if saved else "log in with"
        return f"  {name}{state:<12}({note} `runfleet connect {report.alias}`)"
    if report.state in (ConnState.OK, ConnState.DEGRADED):
        latency = f"{report.latency_s:.2f}s" if report.latency_s is not None else "—"
        return f"  {name}{state:<12}{kind_text(report):<28}{latency:>6}  mux {report.mux}"
    reason = report.error.summary if report.error else "no answer"
    return f"  {name}{state:<12}{reason}" + (" (not saved)" if saved is not None else "")


async def probe_aliases(service: FleetAPI, aliases: Sequence[str],
                        on_report: Callable[[DetectReport], None] | None = None) -> list[DetectReport]:
    """Probe every alias (the service bounds concurrency by max_parallel_ssh); `on_report` sees each report as it
    finishes, so a long `--all` run prints progress instead of staying silent. Returns reports in alias order."""
    async def one(alias: str) -> DetectReport:
        report = await service.probe_alias(alias)
        if on_report is not None:
            on_report(report)
        return report
    return list(await asyncio.gather(*(one(a) for a in aliases)))


def probe_and_report(service: FleetAPI, aliases: Sequence[str], out: Callable[[str], None], *,
                     saving: bool | None) -> list[DetectReport]:
    """Run the probes, printing one line per host as it finishes (`saving`: None = a dry run, else whether
    writable hosts are being saved)."""
    width = alias_width(aliases)

    def show(r: DetectReport) -> None:
        out(report_line(r, saved=None if saving is None else r.state in _WRITABLE, width=width))
    return asyncio.run(probe_aliases(service, aliases, show))


def configured_names(path: Path) -> set[str]:
    """Host names already in the config at `path` (empty when missing or unreadable — write_config reports a
    broken file itself)."""
    import tomllib

    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    return {str(h.get("name")) for h in doc.get("host", []) if isinstance(h, dict)}


def unsaved_hint(reports: Sequence[DetectReport]) -> str | None:
    """Hosts that did not answer right now (VPN off, a slow cold login) are not saved; say how to add them later."""
    missing = [r.alias for r in reports if r.state not in _WRITABLE]
    if not missing:
        return None
    args = " ".join(shell_word(a) for a in missing)
    return (f"Not saved (no answer now): {', '.join(missing)}. Add later with "
            f"`runfleet hosts import {args} --write` (slow logins: raise connect_timeout in the config).")


def shell_word(text: str) -> str:
    import shlex

    return shlex.quote(text)


def write_entries(service: FleetAPI, entries: Sequence[HostConfig], path: Path | None,
                  default: Path) -> tuple[Path, list[str], list[str]]:
    """write_config + which names were really appended vs already configured (write_config never rewrites)."""
    present = configured_names(path or default)
    written = service.write_config(entries, path)
    new = [e.name for e in entries if e.name not in present]
    return written, new, [e.name for e in entries if e.name in present]


def plural(n: int, word: str = "host") -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def host_entry(alias: str) -> HostConfig:
    return HostConfig(name=host_name_for(alias, "-"), ssh=alias)


def writable(reports: Sequence[DetectReport]) -> list[HostConfig]:
    return [host_entry(r.alias) for r in reports if r.state in _WRITABLE]


def parse_selection(text: str, aliases: Sequence[str]) -> list[str]:
    """`1-5`, `k2,lynx`, `all`, `2 4` -> aliases in list order; raises UsageError on anything unknown."""
    text = text.strip()
    if text.lower() == "all":
        return list(aliases)
    chosen: list[str] = []
    for tok in re.split(r"[,\s]+", text):
        if not tok:
            continue
        if re.fullmatch(r"\d+(-\d+)?", tok):
            lo, _, hi = tok.partition("-")
            for i in range(int(lo), int(hi or lo) + 1):
                if not 1 <= i <= len(aliases):
                    raise UsageError(f"no host number {i} (1-{len(aliases)})")
                chosen.append(aliases[i - 1])
        elif tok in aliases:
            chosen.append(tok)
        else:
            raise UsageError(f"unknown host {tok!r}")
    if not chosen:
        raise UsageError("nothing selected")
    return list(dict.fromkeys(chosen))


def alias_grid(aliases: Sequence[str], width: int = 100) -> str:
    cells = [f"{i} {a}" for i, a in enumerate(aliases, 1)]
    lines: list[str] = []
    line = " "
    for cell in cells:
        if len(line) + len(cell) + 4 > width and line.strip():
            lines.append(line.rstrip())
            line = " "
        line += " " + cell + "   "
    lines.append(line.rstrip())
    return "\n".join(lines)


# ====================================================================================================== commands
def first_run(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int | Path | str:
    """Mockup N. Returns the written config path, "demo", or an exit code."""
    from runfleet.transport import make_transport

    path = default_config_path(args)
    aliases = sshconfig.list_aliases()
    if not aliases:
        out(f"No config yet ({display_path(path)}) and no Host aliases in ~/.ssh/config.\n"
            "Run `runfleet <ssh-target> …` (it is saved once it answers; manage hosts later in Settings, key ,),\n"
            "`runfleet --demo`, or write a config (see docs/CONFIG.md).")
        return EXIT_USAGE
    out(f"No config yet ({display_path(path)}). Hosts in ~/.ssh/config:\n{alias_grid(aliases)}")
    while True:
        try:
            answer = ask("Monitor which? (e.g. 1-5, k2,lynx, all; empty = quit, d = demo): ")
        except EOFError:
            return EXIT_OK
        if not answer.strip():
            return EXIT_OK
        if answer.strip().lower() == "d":
            return "demo"
        try:
            selected = parse_selection(answer, aliases)
            break
        except UsageError as exc:
            out(f"  {exc}")
    if not ssh_available():
        raise NoSSH()
    out(f"Probing {len(selected)} host{'s' if len(selected) != 1 else ''} (read-only, BatchMode) …")
    service = build_service(FleetConfig(hosts=()), make_transport, clock, None)
    reports = probe_and_report(service, selected, out, saving=True)
    entries = writable(reports)
    hint = unsaved_hint(reports)
    if not entries:
        out("Nothing to save: no host answered. Check `ssh <alias>` in a terminal, or try `runfleet --demo`.")
        if hint:
            out(hint)
        return EXIT_ERROR
    written = service.write_config(entries, args.config)
    out(f"Wrote {display_path(written)} ({plural(len(entries))}).")
    if hint:
        out(hint)
    out("From now on `runfleet` alone starts this fleet; add or remove hosts in Settings (press , in RunFleet).")
    out("Starting RunFleet …")
    return written


def cmd_tui(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    if not has_terminal():
        out("runfleet: the TUI needs a terminal; use `runfleet snapshot --json` in scripts")
        return EXIT_USAGE
    if args.demo:
        fleet = demo_fleet(args, args.hosts, clock)
    else:
        path = default_config_path(args)
        if not args.hosts and not path.exists():
            if args.config is not None:
                raise UsageError(f"no such config file: {args.config}")
            if not is_interactive():
                out("runfleet: no config; run `runfleet hosts import` (or `runfleet --demo`)")
                return EXIT_USAGE
            result = first_run(args, clock, out)
            if isinstance(result, int):
                return result
            if result == "demo":
                args.demo = True
                return cmd_tui(args, clock, out)
            args.config = result
        fleet = real_fleet(args, args.hosts)
        if not fleet.config.hosts and not is_interactive():
            out(f"runfleet: no hosts in {display_path(default_config_path(args))}; add [[host]] tables or run "
                "`runfleet hosts import --write`")
            return EXIT_USAGE
        ensure_ssh(fleet, args)
    service = build_service(fleet.config, fleet.transports, clock, open_cache(args.no_cache or args.demo))
    if args.demo:
        if hasattr(service, "lock_path"):
            service.lock_path = False   # the demo never takes (or blocks) the real instance lock
        service.config_path = None      # Settings works in memory only: --demo never touches the config
        world = fleet.world
        if world is not None and hasattr(service, "alias_source"):    # Settings lists the simulated hosts
            service.alias_source = lambda: [c.name for c in world.host_configs()]
    else:
        service.config_path = default_config_path(args)
        if args.hosts:                  # §8.4: hosts named here are saved once they answer (general.remember_hosts)
            service.remember_hosts([h.name for h in fleet.config.hosts])
    notice = config_notice(fleet.warnings)
    if notice is not None:
        service.notices.append(notice)
    rc = run_tui(service, clock, demo=args.demo, settings=not fleet.config.hosts)
    return update_and_restart(args, out) if rc == update.RESTART_CODE else rc


def cmd_connect(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    fleet = demo_fleet(args, [args.host], clock) if args.demo else real_fleet(args, [args.host])
    if not fleet.config.hosts:
        raise UsageError(f"unknown host {args.host!r}")
    ensure_ssh(fleet, args)
    service = build_service(fleet.config, fleet.transports, clock, None)
    name = fleet.config.hosts[0].name   # HOST may be an ssh alias (`ssh = "k2"`) or `user@host`: the service uses names
    if asyncio.run(service.prepare_connect(name)):     # resolves the mux plan; clears a stale socket
        out(f"{args.host}: already connected; RunFleet reuses this connection")
        return EXIT_OK
    argv = service.connect_argv(name)
    if argv is None:
        out(f"{args.host}: nothing to connect (local host)")
        return EXIT_OK
    out(f"RunFleet → connecting to {args.host} (Duo/password/host-key prompts appear here; Ctrl-C aborts)")
    rc = run_command(argv)
    if rc == 0:
        out(f"{args.host}: connected; RunFleet reuses this connection while it lasts")
        return EXIT_OK
    out(f"{args.host}: ssh failed (exit {rc}); `runfleet doctor {args.host}` shows details")
    return EXIT_ERROR


def cmd_hosts_list(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    if args.demo:
        config = demo_fleet(args, [], clock).config
        source = "demo fleet"
    else:
        path = default_config_path(args)
        if args.config is not None and not path.exists():
            raise UsageError(f"no such config file: {args.config}")
        if not path.exists():
            out(f"No config at {display_path(path)}; run `runfleet` (picker) or "
                "`runfleet hosts import --all --write` to create one.")
            return EXIT_OK
        config = load_config(args.config)
        source = display_path(config.source or path)
    rows: list[tuple[str, ...]] = [("#", "NAME", "SSH", "BACKEND", "MUX", "GROUP", "ENABLED")]
    for i, h in enumerate(config.hosts, 1):
        rows.append((str(i), h.name, h.ssh, h.backend, h.mux, h.group or "—", "yes" if h.enabled else "no"))
    out(f"{len(config.hosts)} host{'s' if len(config.hosts) != 1 else ''} in {source}")
    out(export.format_table(rows).rstrip("\n"))
    return EXIT_OK


def cmd_hosts_import(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    if args.all and args.aliases:
        raise UsageError("give ALIAS … or --all, not both")
    if args.demo:
        fleet = demo_fleet(args, [], clock)
        aliases = list(args.aliases) or [h.name for h in fleet.config.hosts]
        transports = fleet.transports
    else:
        from runfleet.transport import make_transport

        aliases = sshconfig.list_aliases() if args.all else list(args.aliases)
        transports = make_transport
        if not ssh_available():
            raise NoSSH()
    if not aliases:
        raise UsageError("nothing to import: give ALIAS … or --all")
    out(f"Probing {len(aliases)} host{'s' if len(aliases) != 1 else ''} (read-only, BatchMode) …")
    service = build_service(FleetConfig(hosts=()), transports, clock, None)
    reports = probe_and_report(service, aliases, out, saving=True if args.write and not args.demo else None)
    entries = writable(reports)
    if not args.write:
        if entries:
            out(f"{len(entries)} host{'s' if len(entries) != 1 else ''} usable; run again with --write to save.")
        return EXIT_OK
    if args.demo:
        out("demo: nothing written (--demo never touches your config)")
        return EXIT_OK
    hint = unsaved_hint(reports)
    if not entries:
        out("Nothing written: no host answered.")
        if hint:
            out(hint)
        return EXIT_ERROR
    written, new, present = write_entries(service, entries, args.config, default_config_path(args))
    if new:
        out(f"Wrote {display_path(written)} ({plural(len(new))}: {', '.join(new)}).")
    if present:
        out(f"Already configured in {display_path(written)} (left unchanged): {', '.join(present)}.")
    if hint:
        out(hint)
    return EXIT_OK


_GLYPH = {True: "✓", False: "✗", None: "·"}


def probed_names(config: FleetConfig, args: argparse.Namespace) -> list[str]:
    """Hosts `doctor`/`snapshot` probe: every enabled one, or exactly those named (disabled ones included); a disabled
    host is "listed but not polled" and must not cost an ssh login (or a Duo prompt) by default."""
    names = [h.name for h in config.hosts if h.enabled or args.hosts]
    if not names:
        raise UsageError("every configured host is disabled; name one (HOST …) to probe it anyway")
    return names


def format_doctor(report: DoctorReport) -> str:
    width = max([len(c.name) for c in report.checks] + [6])
    lines = [report.host]
    lines += [f"  {_GLYPH[c.ok]} {c.name:<{width}}  {c.detail}" for c in report.checks]
    lines += [f"  ! {w}" for w in report.warnings]
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    fleet = demo_fleet(args, args.hosts, clock) if args.demo else real_fleet(args, args.hosts)
    if not fleet.config.hosts:
        raise UsageError("no hosts: give HOST … or configure some (`runfleet hosts import --write`)")
    ensure_ssh(fleet, args)
    # the cache keeps the capabilities doctor detects, so a following `snapshot` needs no detect probe (M3 budget)
    service = build_service(fleet.config, fleet.transports, clock, open_cache(args.no_cache or args.demo))
    names = probed_names(fleet.config, args)

    async def run_all() -> list[DoctorReport | BaseException]:
        return list(await asyncio.gather(*(service.doctor(n, reset=args.reset) for n in names),
                                         return_exceptions=True))

    failed = False
    for name, result in zip(names, asyncio.run(run_all())):
        if isinstance(result, BaseException):
            log.error("doctor %s failed", name, exc_info=result)
            out(f"{name}\n  ✗ doctor  internal error: {result!r} (see runfleet.log)")
            failed = True
            continue
        failed = failed or any(c.ok is False for c in result.checks)
        out(format_doctor(result))
    return EXIT_HOST_FAILED if failed else EXIT_OK


def cmd_snapshot(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    fleet = demo_fleet(args, args.hosts, clock) if args.demo else real_fleet(args, args.hosts)
    if not fleet.config.hosts:
        raise UsageError("no hosts: give HOST … or configure some (`runfleet hosts import --write`)")
    ensure_ssh(fleet, args)
    # cached capabilities (< 1 h) replace the detect probe; cached job rows are never printed (collect_once)
    service = build_service(fleet.config, fleet.transports, clock, open_cache(args.no_cache or args.demo))
    names = probed_names(fleet.config, args)
    asyncio.run(service.collect_once(names, timeout=args.timeout))
    if args.json:
        out(export.dumps(export.build_snapshot(service.store, generated_at=clock.now(), hosts=names)).rstrip("\n"))
    else:
        out(export.format_table(export.summary_rows(service.store, names)).rstrip("\n"))
    return EXIT_HOST_FAILED if export.failed_hosts(service.store, names) else EXIT_OK


def cmd_wait(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    """Poll just the named hosts (normal cadence, read-only, no cache: cached rows are not news) until decided."""
    from runfleet import wait

    def failed(message: str) -> int:
        """Exit 3 for every error, so a script never mistakes one for a timeout (2) or a failed job (1)."""
        if args.json:
            return wait.Waiter([], emit=out, as_json=True, now=clock.now).error(message)
        print(f"runfleet: {message}", file=sys.stderr, flush=True)
        return wait.EXIT_ERROR

    try:
        hosts = list(dict.fromkeys(wait.split_target(t)[0] for t in args.hosts))
        fleet = demo_fleet(args, hosts, clock) if args.demo else real_fleet(args, hosts)
        targets = wait.resolve_targets(args.hosts, fleet.config.hosts)
    except wait.TargetError as exc:
        return failed(str(exc))
    except ConfigError as exc:
        return failed(f"{display_path(default_config_path(args))}: {exc}")
    except (UsageError, RunFleetError) as exc:
        return failed(str(exc))
    if not args.demo and needs_ssh(fleet.config) and not ssh_available():
        return failed(no_ssh_message().removeprefix("runfleet: "))
    general = dataclasses.replace(fleet.config.general, readonly=True, idle_pause_after=wait.NEVER)
    config = dataclasses.replace(fleet.config, general=general,
                                 hosts=tuple(dataclasses.replace(h, enabled=True) for h in fleet.config.hosts))
    try:
        service = build_service(config, fleet.transports, clock, None)
        if args.demo and hasattr(service, "lock_path"):
            service.lock_path = False       # the demo never takes (or blocks) the real instance lock
        return asyncio.run(wait.wait_for(service, targets, clock=clock, timeout=args.timeout, emit=out,
                                         as_json=args.json))
    except RunFleetError as exc:            # "another RunFleet is polling", a service that cannot start
        return failed(str(exc))
    except (KeyboardInterrupt, BrokenPipeError):
        raise
    except Exception as exc:
        if args.debug:
            raise
        log.exception("wait: unexpected error")
        return failed(f"internal error: {exc!r} (details in runfleet.log)")


# ====================================================================================================== update
def run_update(install: update.Install, out: Callable[[str], None]) -> int:
    """Run the install's update commands, then say which version a fresh start runs."""
    if install.note:
        out(f"Note: {install.note}.")
    if not install.steps:
        out(f"This copy cannot update itself; run:\n  {install.manual}")
        return EXIT_ERROR
    for argv in install.steps:
        out("$ " + shlex.join([Path(argv[0]).name, *argv[1:]]))
        rc = run_step(argv, install.env)
        if rc != 0:
            out(f"Update failed ({Path(argv[0]).name} exited with {rc}). To update by hand: {install.manual}")
            return EXIT_ERROR
    new = installed_version()
    if new is None:
        out("Updated (could not read the new version: `runfleet --version`).")
    elif new == __version__:
        out(f"Done: still {new} (no newer code).")
    else:
        out(f"Updated: {__version__} → {new}.")
    return EXIT_OK


def cmd_update(args: argparse.Namespace, clock: Clock, out: Callable[[str], None]) -> int:
    install = update.detect_install()
    out(f"RunFleet {__version__}, {install.describe()}.")
    status = update.check(clock.now(), force=True)
    if status.error is not None:
        out(f"Could not ask GitHub for the newest version: {status.error}")
    elif status.available:
        out(f"Version {status.latest} is available.")
        try:
            notes = update.changes_since(update.fetch_text("CHANGELOG.md"), __version__)
        except Exception:                      # what changed is a nicety: the update itself does not need it
            notes = []
        if notes:
            out("\n" + "\n".join(f"  {line}" if line else "" for line in notes) + "\n")
    else:
        out(f"This is the newest version (GitHub main: {status.latest}).")
    if args.check:
        if status.available:
            out("Update with `runfleet update`" + (f" (or by hand: {install.manual})" if not install.steps else "")
                + ".")
        return EXIT_ERROR if status.error is not None else EXIT_OK
    if not status.available and not args.force:
        if status.error is not None:
            out("`runfleet update --force` runs the update anyway.")
            return EXIT_ERROR
        return EXIT_OK
    rc = run_update(install, out)
    if rc == EXIT_OK:
        out("If RunFleet is open in another terminal, quit it (q) and start it again.")
    return rc


def update_and_restart(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    """Settings › About › update: the TUI has quit; update this install, then start RunFleet again as it was."""
    out("RunFleet → updating; it starts again when done …")
    rc = run_update(update.detect_install(), out)
    if rc != EXIT_OK:
        return rc
    argv = list(getattr(args, "argv", []))
    if args.demo and "--demo" not in argv:     # the first-run prompt's `d`
        argv.append("--demo")
    return restart(update.relaunch_argv(argv))


COMMANDS: dict[str, Callable[[argparse.Namespace, Clock, Callable[[str], None]], int]] = {
    "connect": cmd_connect, "doctor": cmd_doctor, "snapshot": cmd_snapshot, "wait": cmd_wait, "update": cmd_update,
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    args.argv = argv                                 # an update started in the TUI restarts with the same arguments
    setup_logging(args.debug, demo=bool(getattr(args, "demo", False)))
    clock: Clock = RealClock()

    def out(text: str) -> None:
        print(text, flush=True)

    def err(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    if args.debug and args.command is not None:      # the TUI shows it in the log instead of on the terminal
        err(f"runfleet: --debug: raw probe dumps in {paths.debug_dir()}")
    try:
        if args.command is None:
            return cmd_tui(args, clock, out)
        if args.command == "hosts":
            handler = cmd_hosts_list if args.hosts_command == "list" else cmd_hosts_import
            return handler(args, clock, out)
        return COMMANDS[args.command](args, clock, out)
    except ConfigError as exc:
        err(f"{display_path(default_config_path(args))}: {exc}")
        return EXIT_USAGE
    except UsageError as exc:
        err(f"runfleet: {exc}")
        return EXIT_USAGE
    except NoSSH:
        err(no_ssh_message())
        return EXIT_NO_SSH
    except KeyboardInterrupt:
        err("interrupted")
        return 130
    except BrokenPipeError:                      # `runfleet snapshot | head`: the reader left; not an internal error
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())    # no second BrokenPipeError when Python flushes stdout at exit
        except (OSError, ValueError):
            pass
        return 141
    except RunFleetError as exc:
        err(f"runfleet: {exc}")
        return EXIT_ERROR
    except Exception as exc:
        if args.debug:
            raise
        log.exception("unexpected error")
        err(f"runfleet: internal error: {exc!r} (details in runfleet.log; --debug shows the traceback)")
        return EXIT_ERROR
