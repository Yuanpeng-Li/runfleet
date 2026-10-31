"""[WP1] FROZEN settings dataclasses (DESIGN.md §3.7); load/validate config.toml; TOML emitter."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from runfleet.errors import ConfigError

log = logging.getLogger("runfleet.config")

# Attach (§7.16): "ssh" = node ssh via pam_slurm_adopt only (no accounting trace); "ssh+srun" = ssh, and srun (with
# confirmation) when ssh is refused or lands in another job; "srun" = srun only (with confirmation; lyra); "off".
AttachSetting = Literal["ssh", "ssh+srun", "srun", "off"]
ATTACH_SETTINGS: tuple[str, ...] = ("ssh", "ssh+srun", "srun", "off")

@dataclass(frozen=True, slots=True)
class SlurmSettings:
    poll_jobs: float = 15            # floor 10
    poll_history: float = 120        # floor 60
    poll_nodes: float = 120          # floor 60
    poll_usage: float = 60           # floor 30, never below JobAcctGatherFrequency
    poll_logstat: float = 120        # floor 60
    sacct_window: float = 3 * 86400
    users: tuple[str, ...] = ()               # () -> `squeue --me`
    partitions: tuple[str, ...] = ()          # () -> derived (§5.2.1)
    squeue_args: tuple[str, ...] = ()
    log_templates: tuple[str, ...] = ()
    protected_partitions: tuple[str, ...] = ()
    protected_qos: tuple[str, ...] = ()
    attach: AttachSetting = "ssh+srun"   # Attach (§7.16): how JobDetail's GPU/Procs tab may read a running job's node
    attach_refresh: float = 15       # floor 10; ssh auto-refresh period (srun never auto-refreshes)
    account_usage: bool = True       # Account (§5.2.11): False = no `slurm.account` group and no on-demand report
    poll_account: float = 1800       # floor 900; the account summary period (×2 hidden)
    allocation_period: float = 365 * 86400   # the "period" window starts at next reallocation minus this (sbank);
                                             # 0 = unknown: no period window from sbank dates

@dataclass(frozen=True, slots=True)
class WorkstationSettings:
    poll_gpu: float = 5              # floor 2
    poll_procs: float = 10           # floor 5
    poll_logstat: float = 60         # floor 30
    task_min_age: float = 60
    task_min_cpu: float = 5.0        # percent
    ignore: tuple[str, ...] = ()     # regexes appended to the built-in noise list (§5.3.2)
    kill_grace: float = 10

@dataclass(frozen=True, slots=True)
class HostConfig:
    name: str
    ssh: str                         # alias, or "local"
    enabled: bool = True
    backend: Literal["auto", "slurm", "workstation"] = "auto"
    color: str | None = None
    group: str = ""
    mux: Literal["auto", "user", "managed", "off", "session"] = "auto"
    connect_timeout: float = 45
    ssh_options: tuple[str, ...] = ()
    remote_shell: Literal["sh", "login"] = "sh"
    forward_agent: bool = False
    slurm: SlurmSettings = field(default_factory=SlurmSettings)
    workstation: WorkstationSettings = field(default_factory=WorkstationSettings)

@dataclass(frozen=True, slots=True)
class GeneralSettings:
    readonly: bool = False
    mouse: bool = True
    default_view: Literal["fleet", "jobs", "gpus"] = "fleet"
    recent_window: float = 86400     # ended jobs/tasks stay in the default "fresh" filter
    stall_after: float = 1800        # "log idle" hint threshold
    idle_after: float = 600          # no input -> polling x4 slower
    idle_pause_after: float = 7200   # no input -> all polling paused until the next key/mouse event
    max_parallel_ssh: int = 8
    refresh_multiplier: float = 1.0
    pager: str = ""                  # "" -> $PAGER, else "less -R +G"
    remember_hosts: bool = True      # hosts named on the command line are saved here once they answer (§8.4)
    update_check: bool = True        # once a day, ask GitHub for the newest RunFleet version (§8.6)
    usage_window: Literal["7d", "30d", "90d", "period"] = "30d"   # Account (§5.2.11): default window of the Account
                                     # screen, the periodic summary ("period" -> 30d there) and `agent usage`

@dataclass(frozen=True, slots=True)
class UISettings:
    gpu_util_thresholds: tuple[float, float] = (10, 75)
    gpu_mem_thresholds: tuple[float, float] = (10, 80)
    jobs_columns: tuple[str, ...] = ("host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util",
                                     "node_reason")

NOTIFY_EVENTS = ("started", "ended_bad", "ended_ok", "ended_other", "restarted", "time_low", "gpu_idle")

@dataclass(frozen=True, slots=True)
class NotifySettings:
    enabled: bool = True             # false: events still go to the Notifications history, nothing pops up
    terminal: Literal["auto", "osc9", "osc777", "bell", "off"] = "auto"
    command: tuple[str, ...] | str = ()      # local hook: argv list, or a string run by /bin/sh -c; () = none
    command_timeout: float = 10
    events: tuple[str, ...] = ("ended_bad", "time_low", "restarted")          # for every own job
    watched_events: tuple[str, ...] = ("started", "ended_ok", "ended_other", "gpu_idle")   # watched jobs too

@dataclass(frozen=True, slots=True)
class FleetConfig:
    hosts: tuple[HostConfig, ...]
    general: GeneralSettings = field(default_factory=GeneralSettings)
    ui: UISettings = field(default_factory=UISettings)
    notify: NotifySettings = field(default_factory=NotifySettings)
    source: Path | None = None       # None for ad-hoc / demo

def load_config(path: Path | None, *, hosts: Sequence[str] = ()) -> FleetConfig:
    """Read config.toml (None -> default path; missing file -> no hosts). `hosts` selects/creates ad-hoc hosts
    (§8.4). Raises errors.ConfigError with a key path ("host[2].poll_jobs: expected duration ...")."""
    cfg, warnings = load_config_with_warnings(path, hosts=hosts)
    for w in warnings:
        log.warning("config: %s", w)
    return cfg


def dump_hosts_toml(hosts: Sequence[HostConfig]) -> str:
    """TOML text with one [[host]] table per host (non-default keys only) plus a commented defaults block."""
    out = [_DEFAULTS_BLOCK]
    for h in hosts:
        out.append(host_table_toml(h))
    return "\n".join(out)


# ======================================================================================= implementation
# Durations: "500ms", "15s", "2m", "3h", "3d" (§8.3). Bare numbers are rejected so a unit is never guessed.
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\s*$")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_HOST_NAME_RE = re.compile(r"^(?!\.+$)[A-Za-z0-9_.\-]+$")     # not "." / "..": names become file names
_SSH_OPTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*=[^\n\r\0]*$")
_SQUEUE_ARG_RE = re.compile(r"^[A-Za-z0-9_.=,:%/+\-]+$")
LOCAL_NAMES = frozenset({"local", "localhost"})

SLURM_FLOORS: Mapping[str, float] = {"poll_jobs": 10, "poll_history": 60, "poll_nodes": 60, "poll_usage": 30,
                                     "poll_logstat": 60, "attach_refresh": 10, "poll_account": 900}
WORKSTATION_FLOORS: Mapping[str, float] = {"poll_gpu": 2, "poll_procs": 5, "poll_logstat": 30}


def parse_duration(value: object, key: str) -> float:
    """Seconds from a duration string; raises ConfigError(key) for anything else (including bare numbers)."""
    m = _DURATION_RE.match(value) if isinstance(value, str) else None
    if m is None:
        raise ConfigError(key, f'expected duration like "15s", got {_show(value)}')
    return float(m.group(1)) * _UNIT_S[m.group(2)]


def parse_period(value: object, key: str) -> float:
    """A duration that may also be "0" (unknown / off): `allocation_period` (Account, §5.2.11)."""
    if value in ("0", 0) and not isinstance(value, bool):
        return 0.0
    try:
        return parse_duration(value, key)
    except ConfigError:
        raise ConfigError(key, f'expected duration like "365d" or "0", got {_show(value)}') from None


def format_duration(seconds: float) -> str:
    """Shortest exact duration string for `seconds` (inverse of parse_duration)."""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        n = seconds / size
        if n >= 1 and math.isclose(n, round(n), abs_tol=1e-9):
            return f"{round(n)}{unit}"
    ms = seconds * 1000
    if math.isclose(ms, round(ms), abs_tol=1e-6):
        return f"{round(ms)}ms"
    return f"{seconds!r}s"


def _show(value: object) -> str:
    return f'"{value}"' if isinstance(value, str) else repr(value)


# ---------------------------------------------------------------- typed value readers (key path for errors)
Reader = Callable[[object, str], Any]


def _bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(key, f"expected true or false, got {_show(value)}")
    return value


def _int(minimum: int) -> Reader:
    def read(value: object, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(key, f"expected an integer >= {minimum}, got {_show(value)}")
        return value
    return read


def _float(minimum: float, *, exclusive: bool = False) -> Reader:
    def read(value: object, key: str) -> float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            f = float(value)    # type: ignore[arg-type]
            ok = math.isfinite(f) and (f > minimum if exclusive else f >= minimum)     # TOML has `inf`/`nan`
        if not ok:
            bound = f"> {minimum:g}" if exclusive else f">= {minimum:g}"
            raise ConfigError(key, f"expected a number {bound}, got {_show(value)}")
        return float(value)     # type: ignore[arg-type]
    return read


def _str(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(key, f"expected a string, got {_show(value)}")
    return value


def _opt_str(value: object, key: str) -> str | None:
    return _str(value, key) or None


def _choice(*choices: str) -> Reader:
    def read(value: object, key: str) -> str:
        if value not in choices:
            raise ConfigError(key, f"expected one of {', '.join(choices)}, got {_show(value)}")
        return str(value)
    return read


def _str_list(item: Callable[[str, str], str] | None = None) -> Reader:
    def read(value: object, key: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(key, f"expected a list of strings, got {_show(value)}")
        return tuple(item(v, f"{key}[{i}]") if item else v for i, v in enumerate(value))
    return read


def _matching(regex: re.Pattern[str], what: str) -> Callable[[str, str], str]:
    def check(value: str, key: str) -> str:
        if not isinstance(value, str):      # `name = 123` used as a top-level reader, not only inside _str_list
            raise ConfigError(key, f"expected a string, got {_show(value)}")
        if not regex.match(value):
            raise ConfigError(key, f"invalid {what} {_show(value)}")
        return value
    return check


def _regex(value: str, key: str) -> str:
    try:
        re.compile(value)
    except re.error as e:
        raise ConfigError(key, f"invalid regex {_show(value)}: {e}") from None
    return value


def _pair(value: object, key: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ConfigError(key, f"expected a list of two numbers, got {_show(value)}")
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ConfigError(key, f"expected finite numbers, got {_show(value)}")
    if lo > hi:
        raise ConfigError(key, f"expected [low, high], got {_show(value)}")
    return lo, hi


_names = _str_list(_matching(_HOST_NAME_RE, "name"))


def _host_name(value: str, key: str) -> str:
    """A host name: `_HOST_NAME_RE`, and on Windows not a name Windows refuses for a file or directory (`con`,
    `nul`, `aux`, `com1`, a trailing dot): host names become file and directory names (§8.1)."""
    name = _matching(_HOST_NAME_RE, "host name")(value, key)
    from runfleet.util import platform

    if platform.IS_WINDOWS and platform.windows_reserved_name(name):
        raise ConfigError(key, f"host name {_show(name)} is reserved on Windows (a device name or a trailing dot); "
                               "pick another name")
    return name

GENERAL_SCHEMA: Mapping[str, Reader] = {
    "readonly": _bool, "mouse": _bool, "default_view": _choice("fleet", "jobs", "gpus"),
    "recent_window": parse_duration, "stall_after": parse_duration, "idle_after": parse_duration,
    "idle_pause_after": parse_duration, "max_parallel_ssh": _int(1),
    "refresh_multiplier": _float(0.0, exclusive=True), "pager": _str, "remember_hosts": _bool,
    "update_check": _bool, "usage_window": _choice("7d", "30d", "90d", "period"),
}
UI_SCHEMA: Mapping[str, Reader] = {
    "gpu_util_thresholds": _pair, "gpu_mem_thresholds": _pair, "jobs_columns": _str_list(),
}


def _command(value: object, key: str) -> tuple[str, ...] | str:
    """`[notify] command`: an argv list (non-empty first item) or a shell string."""
    if isinstance(value, str):
        return value
    argv = _str_list()(value, key)
    if argv and not argv[0]:
        raise ConfigError(f"{key}[0]", "the program must not be empty")
    return argv     # type: ignore[no-any-return]


_events = _str_list(_choice(*NOTIFY_EVENTS))
NOTIFY_SCHEMA: Mapping[str, Reader] = {
    "enabled": _bool, "terminal": _choice("auto", "osc9", "osc777", "bell", "off"), "command": _command,
    "command_timeout": parse_duration, "events": _events, "watched_events": _events,
}
SLURM_SCHEMA: Mapping[str, Reader] = {
    "poll_jobs": parse_duration, "poll_history": parse_duration, "poll_nodes": parse_duration,
    "poll_usage": parse_duration, "poll_logstat": parse_duration, "sacct_window": parse_duration,
    "users": _names, "partitions": _names, "squeue_args": _str_list(_matching(_SQUEUE_ARG_RE, "squeue argument")),
    "log_templates": _str_list(), "protected_partitions": _names, "protected_qos": _names,
    "attach": _choice(*ATTACH_SETTINGS), "attach_refresh": parse_duration,
    "account_usage": _bool, "poll_account": parse_duration, "allocation_period": parse_period,
}
WORKSTATION_SCHEMA: Mapping[str, Reader] = {
    "poll_gpu": parse_duration, "poll_procs": parse_duration, "poll_logstat": parse_duration,
    "task_min_age": parse_duration, "task_min_cpu": _float(0.0), "ignore": _str_list(_regex),
    "kill_grace": parse_duration,
}
HOST_SCHEMA: Mapping[str, Reader] = {
    "name": _host_name,                                 # type: ignore[dict-item]
    "ssh": _str, "enabled": _bool, "backend": _choice("auto", "slurm", "workstation"), "color": _opt_str,
    "group": _str, "mux": _choice("auto", "user", "managed", "off", "session"), "connect_timeout": parse_duration,
    "ssh_options": _str_list(_matching(_SSH_OPTION_RE, "ssh option (expected Key=Value)")),
    "remote_shell": _choice("sh", "login"), "forward_agent": _bool,
}


def _read_table(table: object, schema: Mapping[str, Reader], key: str, warnings: list[str],
                *, extra: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Validate the known keys of one TOML table; unknown keys (not in `extra` either) become warnings."""
    if not isinstance(table, dict):
        raise ConfigError(key, f"expected a table, got {_show(table)}")
    values: dict[str, Any] = {}
    for k, v in table.items():
        path = f"{key}.{k}" if key else k
        if k in schema:
            values[k] = schema[k](v, path)
        elif k not in extra:
            warnings.append(f"{path}: unknown key (ignored)")
    return values


def _clamp(values: dict[str, Any], floors: Mapping[str, float], key: str, warnings: list[str],
           *, quiet: frozenset[str] = frozenset()) -> dict[str, Any]:
    for k, floor in floors.items():
        if k in values and values[k] < floor:
            if k not in quiet:
                warnings.append(f"{key}.{k}: {format_duration(values[k])} is below the floor "
                                f"{format_duration(floor)}; using {format_duration(floor)}")
            values[k] = float(floor)
    return values


def load_config_with_warnings(path: Path | None, *,
                              hosts: Sequence[str] = ()) -> tuple[FleetConfig, tuple[str, ...]]:
    """`load_config` that also returns the validation warnings (unknown keys, clamped intervals) for a UI toast."""
    from runfleet import paths   # local import: paths pulls platformdirs, config must stay importable everywhere

    file = path if path is not None else paths.config_file()
    data: dict[str, Any] = {}
    if file.is_file():
        try:
            data = tomllib.loads(file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("", f"{file}: invalid TOML: {e}") from None
        except OSError as e:
            raise ConfigError("", f"{file}: {e.strerror}") from None
    cfg, warnings = parse_config(data, source=file if file.is_file() else None)
    if hosts:
        cfg = dataclasses.replace(cfg, hosts=select_hosts(cfg.hosts, hosts, data, warnings), source=None)
    return cfg, tuple(warnings)


def parse_config(data: Mapping[str, Any], *, source: Path | None = None) -> tuple[FleetConfig, list[str]]:
    """Validate an already-parsed TOML document."""
    warnings: list[str] = []
    for k in data:
        if k not in ("general", "ui", "defaults", "host", "notify"):
            warnings.append(f"{k}: unknown key (ignored)")
    general = GeneralSettings(**_read_table(data.get("general", {}), GENERAL_SCHEMA, "general", warnings))
    ui = UISettings(**_read_table(data.get("ui", {}), UI_SCHEMA, "ui", warnings))
    notify = NotifySettings(**_read_table(data.get("notify", {}), NOTIFY_SCHEMA, "notify", warnings))
    slurm_d, ws_d = _read_defaults(data.get("defaults", {}), warnings)
    raw_hosts = data.get("host", [])
    if not isinstance(raw_hosts, list):
        raise ConfigError("host", "expected [[host]] tables")
    out: list[HostConfig] = []
    seen: set[str] = set()
    for i, table in enumerate(raw_hosts):
        h = _parse_host(table, f"host[{i}]", slurm_d, ws_d, warnings)
        if h.name in seen:
            raise ConfigError(f"host[{i}].name", f"duplicate host name {_show(h.name)}")
        seen.add(h.name)
        out.append(h)
    return FleetConfig(hosts=tuple(out), general=general, ui=ui, source=source, notify=notify), warnings


def _read_defaults(table: object, warnings: list[str]) -> tuple[SlurmSettings, WorkstationSettings]:
    if not isinstance(table, dict):
        raise ConfigError("defaults", f"expected a table, got {_show(table)}")
    for k in table:
        if k not in ("slurm", "workstation"):
            warnings.append(f"defaults.{k}: unknown key (ignored)")
    s = _clamp(_read_table(table.get("slurm", {}), SLURM_SCHEMA, "defaults.slurm", warnings), SLURM_FLOORS,
               "defaults.slurm", warnings)
    w = _clamp(_read_table(table.get("workstation", {}), WORKSTATION_SCHEMA, "defaults.workstation", warnings),
               WORKSTATION_FLOORS, "defaults.workstation", warnings)
    return SlurmSettings(**s), WorkstationSettings(**w)


def _parse_host(table: object, key: str, slurm_d: SlurmSettings, ws_d: WorkstationSettings,
                warnings: list[str]) -> HostConfig:
    if not isinstance(table, dict):
        raise ConfigError(key, f"expected a table, got {_show(table)}")
    if "name" not in table:
        raise ConfigError(f"{key}.name", "missing (every [[host]] needs a name)")
    flat = frozenset(SLURM_SCHEMA) | frozenset(WORKSTATION_SCHEMA)
    base = _read_table(table, HOST_SCHEMA, key, warnings, extra=flat)
    # Flat Slurm/workstation keys override the matching [defaults.*] field. `poll_logstat` exists in both and sets
    # both, each clamped to its own floor; only the workstation clamp (the lower floor) warns.
    shared = frozenset(SLURM_SCHEMA) & frozenset(WORKSTATION_SCHEMA)
    s = _clamp(_read_table({k: v for k, v in table.items() if k in SLURM_SCHEMA}, SLURM_SCHEMA, key, []),
               SLURM_FLOORS, key, warnings, quiet=shared)
    w = _clamp(_read_table({k: v for k, v in table.items() if k in WORKSTATION_SCHEMA}, WORKSTATION_SCHEMA, key,
                           []), WORKSTATION_FLOORS, key, warnings)
    if "ssh" not in base:
        base["ssh"] = "local" if base["name"] in LOCAL_NAMES else base["name"]
    elif not base["ssh"]:
        raise ConfigError(f"{key}.ssh", "must not be empty")
    return HostConfig(**base, slurm=dataclasses.replace(slurm_d, **s),
                      workstation=dataclasses.replace(ws_d, **w))


def host_name_for(alias: str, fill: str = "_") -> str:
    """A valid host name for an ssh alias or destination: other characters, and a name of only dots, become `fill`;
    on Windows a reserved name gets a `_` (`con` -> `con_`)."""
    from runfleet.util import platform

    name = re.sub(r"[^A-Za-z0-9_.\-]", fill, alias)
    return fill * len(name) if not name.strip(".") else platform.safe_file_name(name)


def select_hosts(configured: Sequence[HostConfig], wanted: Sequence[str], data: Mapping[str, Any],
                 warnings: list[str]) -> tuple[HostConfig, ...]:
    """Ad-hoc host selection (§8.4): config names or ssh aliases, merged with matching [[host]] settings."""
    slurm_d, ws_d = _read_defaults(data.get("defaults", {}), [])
    by_name = {h.name: h for h in configured}
    by_ssh = {h.ssh: h for h in configured}
    out: list[HostConfig] = []
    for alias in dict.fromkeys(wanted):       # keep order, drop duplicates
        h = by_name.get(alias) or by_ssh.get(alias)
        if h is None:
            if not alias or alias.startswith("-") or any(c.isspace() for c in alias):
                raise ConfigError("hosts", f"invalid host alias {_show(alias)}")
            h = HostConfig(name=host_name_for(alias), ssh="local" if alias in LOCAL_NAMES else alias, slurm=slurm_d,
                           workstation=ws_d)
        if any(o.name == h.name for o in out):
            warnings.append(f"hosts: {alias} duplicates {h.name} (ignored)")
            continue
        out.append(h)
    return tuple(out)


# ---------------------------------------------------------------- TOML emitter (our own schema only)
def toml_value(value: object) -> str:
    """One TOML value: str, bool, int, float, or a list of those."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("TOML cannot hold nan/inf here")
        return repr(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot emit {type(value).__name__} as TOML")


def _toml_str(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _emit(key: str, value: object, reader: Reader) -> str:
    if reader is parse_period and value == 0:
        return f'{key} = "0"'
    if reader in (parse_duration, parse_period) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{key} = {_toml_str(format_duration(value))}"
    return f"{key} = {toml_value(value)}"


def host_table_toml(h: HostConfig) -> str:
    """One `[[host]]` table with only the keys that differ from the defaults (flat Slurm/workstation keys)."""
    lines = ["[[host]]", f"name = {_toml_str(h.name)}"]
    default_ssh = "local" if h.name in LOCAL_NAMES else h.name
    if h.ssh != default_ssh:
        lines.append(f"ssh = {_toml_str(h.ssh)}")
    proto = HostConfig(name=h.name, ssh=h.ssh)
    for k, reader in HOST_SCHEMA.items():
        if k in ("name", "ssh"):
            continue
        v = getattr(h, k)
        if v != getattr(proto, k):
            lines.append(_emit(k, "" if v is None else v, reader))
    # Flat keys; a key in both sections (`poll_logstat`) is written once, the Slurm value winning (lossy only when
    # both differ from their defaults in different ways, which the flat schema cannot express).
    emitted: set[str] = set()
    for settings, defaults, schema in ((h.slurm, SlurmSettings(), SLURM_SCHEMA),
                                       (h.workstation, WorkstationSettings(), WORKSTATION_SCHEMA)):
        for k, reader in schema.items():
            v = getattr(settings, k)
            if v != getattr(defaults, k) and k not in emitted:
                lines.append(_emit(k, list(v) if isinstance(v, tuple) else v, reader))
                emitted.add(k)
    return "\n".join(lines) + "\n"


def _commented_defaults() -> str:
    g, u, s, w = GeneralSettings(), UISettings(), SlurmSettings(), WorkstationSettings()
    lines = ["# RunFleet configuration (docs/CONFIG.md). Every key is optional except host.name.",
             "# Defaults, commented out — uncomment to change:", "#", "# [general]"]
    for k, r in GENERAL_SCHEMA.items():
        lines.append("# " + _emit(k, getattr(g, k), r))
    lines += ["#", "# [ui]"]
    for k, r in UI_SCHEMA.items():
        lines.append("# " + _emit(k, list(getattr(u, k)), r))
    lines += ["#", "# [defaults.slurm]"]
    for k, r in SLURM_SCHEMA.items():
        v = getattr(s, k)
        lines.append("# " + _emit(k, list(v) if isinstance(v, tuple) else v, r))
    lines += ["#", "# [defaults.workstation]"]
    for k, r in WORKSTATION_SCHEMA.items():
        v = getattr(w, k)
        lines.append("# " + _emit(k, list(v) if isinstance(v, tuple) else v, r))
    n = NotifySettings()
    lines += ["#", "# [notify]"]
    for k, r in NOTIFY_SCHEMA.items():
        v = getattr(n, k)
        lines.append("# " + _emit(k, list(v) if isinstance(v, tuple) else v, r))
    return "\n".join(lines) + "\n"


_DEFAULTS_BLOCK = _commented_defaults()
