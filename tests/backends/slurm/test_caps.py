"""[WP2] Slurm capability parsing (§5.1) from slurm_config / sacctmgr_assoc / sinfo_version fixtures, plus the
stat and env fixtures that inform the logstat parser and the prelude's env unsetting."""

from __future__ import annotations

import pytest

from runfleet.backends.common import parse_stat_lines
from runfleet.backends.slurm.collector import SlurmCollector, parse_capabilities
from tests.backends.factories import fixture_result, result
from tests.conftest import sample


def caps_of(host: str) -> dict[str, object]:
    return parse_capabilities({
        "slurm.version": fixture_result("slurm.version", host, "sinfo_version"),
        "slurm.config": fixture_result("slurm.config", host, "slurm_config"),
        "slurm.assoc": fixture_result("slurm.assoc", host, "sacctmgr_assoc"),
        "slurm.gputres": fixture_result("slurm.gputres", host, "sacct_completed_gpu_tres"),
    })


@pytest.mark.parametrize(("host", "version", "preempt"), [("k2", "25.05.8", "REQUEUE"), ("vega", "25.11.4", "REQUEUE"),
                                                          ("lyra", "26.05.3", "CANCEL")])
def test_capabilities(host: str, version: str, preempt: str) -> None:
    c = caps_of(host)
    assert c["slurm_version"] == version
    assert c["cluster_name"] == host
    cfg = c["slurm_config"]
    assert isinstance(cfg, dict)
    assert cfg["PreemptMode"] == preempt and cfg["MinJobAge"] == "300 sec"
    assert cfg["AccountingStorageType"] == "accounting_storage/slurmdbd"
    assert c["acct_gather_freq_s"] == 30
    assert c["gpu_tres"] is (host != "lyra")


def test_associations() -> None:
    assert caps_of("k2")["associations"] == (("k2", "campus01", "", "gpu-priority,normal,labpi"),)
    assert caps_of("vega")["associations"] == (("vega", "dept", "", "normal"), ("vega", "pilab", "", "normal"))
    assert len(caps_of("lyra")["associations"]) == 3


def test_sacct_helpformat_and_missing_sections() -> None:
    helpformat = "Account             AdminComment        AllocCPUS           AllocNodes\nStdErr StdOut WorkDir"
    c = SlurmCollector.ingest_capabilities({"slurm.sacctfmt": result("slurm.sacctfmt", helpformat),
                                            "slurm.version": result("slurm.version", "", 127, "missing:sinfo")})
    assert c["sacct_fields"] == frozenset({"Account", "AdminComment", "AllocCPUS", "AllocNodes", "StdErr", "StdOut",
                                           "WorkDir"})
    assert "slurm_version" not in c


def test_gputres_seed_empty_is_unknown() -> None:
    c = parse_capabilities({"slurm.gputres": result("slurm.gputres", "")})
    assert "gpu_tres" not in c


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_slurm_env(host: str) -> None:
    """lyra sets SQUEUE_FORMAT site-wide: the probe prelude must unset it (§4.3); k2/vega set nothing."""
    lines = sample(host, "slurm_env").splitlines()
    assert lines[-1] == "(end)"
    if host == "lyra":
        assert any(ln.startswith("SQUEUE_FORMAT=") for ln in lines)
    else:
        assert lines == ["(end)"]


def test_stdout_path_stat_and_logstat_parser() -> None:
    # fixture: `echo "$p"; stat -c '%s bytes mtime=%Y' "$p"` → rebuild our `%n|%s|%Y` form and parse it
    path, stat_line = sample("k2", "stdout_path_stat").splitlines()
    size, mtime = stat_line.replace(" bytes mtime=", " ").split()
    stats = parse_stat_lines(f"{path}|{size}|{mtime}")
    assert stats[path].size == 663 and stats[path].mtime is not None
    assert int(stats[path].mtime.timestamp()) == 1788763833
    vega_path, vega_stat = sample("vega", "stdout_path_stat").splitlines()
    assert vega_stat.startswith("107038 bytes")
    # lyra: another user's log is unreadable — stat prints only an error
    (lyra_path,) = sample("lyra", "stdout_path_stat").splitlines()
    err = parse_stat_lines(f"stat: cannot statx '{lyra_path}': Permission denied")
    assert err[lyra_path].error == "Permission denied" and err[lyra_path].size is None
    both = parse_stat_lines("/a|b|c.out|10|1788763833\nstat: cannot stat '/x': No such file or directory\njunk")
    assert both["/a|b|c.out"].size == 10 and both["/x"].error == "No such file or directory"
