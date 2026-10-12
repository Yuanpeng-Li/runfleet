"""[WP1] config.toml loading, validation, ad-hoc hosts and the TOML emitter (§8)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from runfleet.config import (FleetConfig, GeneralSettings, HostConfig, SlurmSettings, UISettings, WorkstationSettings,
                             dump_hosts_toml, format_duration, load_config, load_config_with_warnings,
                             parse_duration)
from runfleet.errors import ConfigError

EXAMPLE = '''
[general]
readonly = false
mouse = true
default_view = "fleet"
recent_window = "24h"
stall_after = "30m"
idle_after = "10m"
idle_pause_after = "2h"
max_parallel_ssh = 8
refresh_multiplier = 1.0
pager = ""

[ui]
gpu_util_thresholds = [10, 75]
gpu_mem_thresholds = [10, 80]
jobs_columns = ["host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "node_reason"]

[defaults.slurm]
poll_jobs = "15s"
poll_history = "2m"
poll_nodes = "2m"
poll_usage = "60s"
poll_logstat = "2m"
sacct_window = "3d"
log_templates = []

[defaults.workstation]
poll_gpu = "5s"
poll_procs = "10s"
poll_logstat = "60s"
task_min_age = "60s"
task_min_cpu = 5.0
ignore = []
kill_grace = "10s"

[[host]]
name = "k2"
protected_partitions = ["labpi"]
protected_qos = ["labpi"]
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]

[[host]]
name = "vega"
connect_timeout = "120s"
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]

[[host]]
name = "lyra"

[[host]]
name = "labstation"

[[host]]
name = "lynx"

[[host]]
name = "atlas"
ssh = "local"
'''


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_design_example_loads(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, EXAMPLE))
    assert [h.name for h in cfg.hosts] == ["k2", "vega", "lyra", "labstation", "lynx", "atlas"]
    k2, vega = cfg.hosts[0], cfg.hosts[1]
    assert k2.ssh == "k2" and k2.slurm.protected_partitions == ("labpi",)
    assert k2.slurm.log_templates == ("{workdir}/logs/{name}_{jobid}.out",)
    assert vega.connect_timeout == 120
    assert cfg.hosts[5].ssh == "local"
    assert cfg.general == GeneralSettings()
    assert cfg.ui == UISettings()
    assert cfg.source is not None


def test_missing_file_means_no_hosts(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg == FleetConfig(hosts=())


def test_flat_host_keys_override_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, '''
[defaults.slurm]
poll_jobs = "20s"
[defaults.workstation]
kill_grace = "30s"
[[host]]
name = "a"
users = ["bob", "alice"]
partitions = ["lab"]
poll_history = "5m"
poll_logstat = "90s"
ignore = ["^jupyter"]
task_min_cpu = 2
'''))
    h = cfg.hosts[0]
    assert h.slurm == SlurmSettings(poll_jobs=20, poll_history=300, poll_logstat=90, users=("bob", "alice"),
                                    partitions=("lab",))
    assert h.workstation == WorkstationSettings(kill_grace=30, poll_logstat=90, ignore=("^jupyter",), task_min_cpu=2)


def test_floors_clamp_with_warning(tmp_path: Path) -> None:
    cfg, warnings = load_config_with_warnings(write(tmp_path, '''
[defaults.workstation]
poll_gpu = "500ms"
[[host]]
name = "a"
poll_jobs = "5s"
poll_logstat = "45s"
'''))
    h = cfg.hosts[0]
    assert h.slurm.poll_jobs == 10 and h.workstation.poll_gpu == 2
    assert h.slurm.poll_logstat == 60 and h.workstation.poll_logstat == 45   # shared key, per-section floors
    assert any("host[0].poll_jobs" in w and "floor" in w for w in warnings)
    assert any("defaults.workstation.poll_gpu" in w for w in warnings)
    assert not any("poll_logstat" in w for w in warnings)


def test_unknown_keys_warn(tmp_path: Path) -> None:
    _, warnings = load_config_with_warnings(write(tmp_path, '''
color_scheme = "x"
[general]
fancy = true
[[host]]
name = "a"
frobnicate = 1
'''))
    assert sorted(warnings) == ["color_scheme: unknown key (ignored)", "general.fancy: unknown key (ignored)",
                                "host[0].frobnicate: unknown key (ignored)"]


@pytest.mark.parametrize(("text", "key", "msg"), [
    ('[[host]]\nname = "a"\n[[host]]\nname = "b"\n[[host]]\nname = "c"\npoll_jobs = 5\n', "host[2].poll_jobs",
     'expected duration like "15s", got 5'),
    ('[[host]]\nname = "a b"\n', "host[0].name", "invalid host name"),
    ('[[host]]\nssh = "x"\n', "host[0].name", "missing"),
    ('[[host]]\nname = "a"\n[[host]]\nname = "a"\n', "host[1].name", "duplicate"),
    ('[general]\nmax_parallel_ssh = 0\n', "general.max_parallel_ssh", "integer >= 1"),
    ('[general]\ndefault_view = "tree"\n', "general.default_view", "one of fleet, jobs, gpus"),
    ('[ui]\ngpu_util_thresholds = [80, 10]\n', "ui.gpu_util_thresholds", "[low, high]"),
    ('[[host]]\nname = "a"\nmux = "maybe"\n', "host[0].mux", "one of"),
    ('[[host]]\nname = "a"\nusers = ["bad user"]\n', "host[0].users[0]", "invalid name"),
    ('[[host]]\nname = "a"\nsqueue_args = ["--qos=x;rm"]\n', "host[0].squeue_args[0]", "invalid squeue argument"),
    ('[[host]]\nname = "a"\nignore = ["("]\n', "host[0].ignore[0]", "invalid regex"),
    ('[[host]]\nname = "a"\nssh_options = ["-oFoo"]\n', "host[0].ssh_options[0]", "Key=Value"),
    ('[[host]]\nname = "a"\nenabled = "yes"\n', "host[0].enabled", "true or false"),
    # review 2026-10: TOML `inf`/`nan` (BACK-25) and a non-string host name (BACK-39, was an internal TypeError)
    ('[general]\nrefresh_multiplier = inf\n', "general.refresh_multiplier", "expected a number > 0"),
    ('[general]\nrefresh_multiplier = nan\n', "general.refresh_multiplier", "expected a number > 0"),
    ('[ui]\ngpu_util_thresholds = [nan, 50]\n', "ui.gpu_util_thresholds", "finite numbers"),
    ('[ui]\ngpu_mem_thresholds = [10, inf]\n', "ui.gpu_mem_thresholds", "finite numbers"),
    ('[[host]]\nname = 123\n', "host[0].name", "expected a string, got 123"),
    # CORE-43: a name of only dots would become a path component outside the debug dir
    ('[[host]]\nname = ".."\n', "host[0].name", "invalid host name"),
    ('[[host]]\nname = "."\n', "host[0].name", "invalid host name"),
])
def test_validation_errors_carry_key_path(tmp_path: Path, text: str, key: str, msg: str) -> None:
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, text))
    assert e.value.key_path == key
    assert msg in str(e.value)


def test_host_name_for_never_yields_dots_only() -> None:
    """CORE-43: ad-hoc aliases and `hosts import` names are sanitised; "." / ".." never become host names."""
    from runfleet.config import host_name_for
    assert host_name_for("..") == "__" and host_name_for(".", "-") == "-"
    assert host_name_for("alice@box.lan") == "alice_box.lan" and host_name_for("a..b") == "a..b"
    assert host_name_for("") == ""


def test_invalid_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[[host]\nname="))


def test_durations() -> None:
    assert parse_duration("500ms", "k") == 0.5
    assert parse_duration("15s", "k") == 15
    assert parse_duration("2m", "k") == 120
    assert parse_duration("3h", "k") == 10800
    assert parse_duration("3d", "k") == 259200
    assert parse_duration("1.5m", "k") == 90
    for bad in (5, "5", "5x", "", True, "-1s"):
        with pytest.raises(ConfigError):
            parse_duration(bad, "k")
    for secs in (0.5, 1, 15, 90, 120, 3600, 86400, 259200, 2.25):
        assert parse_duration(format_duration(secs), "k") == secs


def test_ad_hoc_hosts(tmp_path: Path) -> None:
    path = write(tmp_path, EXAMPLE)
    cfg = load_config(path, hosts=["vega", "newbox", "localhost", "vega", "user@1.2.3.4"])
    assert [h.name for h in cfg.hosts] == ["vega", "newbox", "localhost", "user_1.2.3.4"]
    assert cfg.hosts[0].connect_timeout == 120                         # merged from [[host]]
    assert cfg.hosts[1].ssh == "newbox" and cfg.hosts[2].ssh == "local"
    assert cfg.hosts[3].ssh == "user@1.2.3.4"
    assert cfg.source is None
    with pytest.raises(ConfigError):
        load_config(path, hosts=["-oProxyCommand=x"])


def test_ad_hoc_uses_defaults_section(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, '[defaults.slurm]\npoll_jobs = "30s"\n'), hosts=["x"])
    assert cfg.hosts[0].slurm.poll_jobs == 30


def test_dump_round_trip(tmp_path: Path) -> None:
    hosts = (
        HostConfig(name="k2", ssh="k2", slurm=SlurmSettings(protected_partitions=("labpi",), poll_jobs=20.0,
                                                           log_templates=('{workdir}/logs/{name}_{jobid}.out',))),
        HostConfig(name="atlas", ssh="local", connect_timeout=120.0, enabled=False, mux="managed"),
        HostConfig(name="x", ssh='we"ird\\alias', color="red", group="workstations", ssh_options=("Compression=yes",),
                   workstation=WorkstationSettings(ignore=("^foo\\s",), task_min_cpu=2.5, kill_grace=0.5)),
        HostConfig(name="local", ssh="local"),
    )
    text = dump_hosts_toml(hosts)
    doc = tomllib.loads(text)
    assert [h["name"] for h in doc["host"]] == ["k2", "atlas", "x", "local"]
    assert doc["host"][3] == {"name": "local"}                         # non-default keys only
    assert doc["host"][0]["poll_jobs"] == "20s"
    assert "general" not in doc and "# [general]" in text              # defaults block is commented
    cfg = load_config(write(tmp_path, text))
    assert cfg.hosts == hosts


def test_commented_defaults_block_is_valid_when_uncommented(tmp_path: Path) -> None:
    text = dump_hosts_toml([])
    body = "\n".join(line[2:] for line in text.splitlines() if line.startswith("# ") and "=" in line or
                     line.startswith("# ["))
    cfg = load_config(write(tmp_path, body))
    assert cfg.general == GeneralSettings() and cfg.ui == UISettings()


def test_notify_section_parses_and_validates(tmp_path: Path) -> None:
    from runfleet.config import NotifySettings

    cfg = load_config(write(tmp_path, '[notify]\nterminal = "osc777"\ncommand = ["notify-send", "RunFleet"]\n'
                                      'command_timeout = "5s"\nevents = ["ended_bad"]\nwatched_events = []\n'))
    assert cfg.notify == NotifySettings(terminal="osc777", command=("notify-send", "RunFleet"), command_timeout=5,
                                        events=("ended_bad",), watched_events=())
    assert load_config(write(tmp_path, '[notify]\ncommand = "notify-send x"\n')).notify.command == "notify-send x"
    assert load_config(write(tmp_path, "")).notify == NotifySettings()
    for body, key in (('events = ["failed"]', "notify.events[0]"), ('terminal = "growl"', "notify.terminal"),
                      ('command = [""]', "notify.command[0]"), ("enabled = 1", "notify.enabled"),
                      ("command_timeout = 5", "notify.command_timeout")):
        with pytest.raises(ConfigError) as exc:
            load_config(write(tmp_path, f"[notify]\n{body}\n"))
        assert str(exc.value).startswith(key), str(exc.value)
