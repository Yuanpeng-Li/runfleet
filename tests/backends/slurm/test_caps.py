"""[WP2] Slurm capability parsing (§5.1) from slurm_config / sacctmgr_assoc / sinfo_version fixtures, the charging
rules of Account (§5.2.11: sacctmgr_qos, scontrol_partitions_charges), plus the stat and env fixtures that inform the
logstat parser and the prelude's env unsetting."""

from __future__ import annotations

import pytest

from runfleet.backends.common import parse_stat_lines
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.collector import (SlurmCollector, gpu_partitions, parse_capabilities, parse_qos,
                                               partition_charges)
from runfleet.models import PartitionCharge
from tests.backends.factories import fixture_result, result
from tests.conftest import sample


def caps_of(host: str) -> dict[str, object]:
    return parse_capabilities({
        "slurm.version": fixture_result("slurm.version", host, "sinfo_version"),
        "slurm.config": fixture_result("slurm.config", host, "slurm_config"),
        "slurm.assoc": fixture_result("slurm.assoc", host, "sacctmgr_assoc"),
        "slurm.parts": fixture_result("slurm.parts", host, "scontrol_partitions"),
        "slurm.gputres": fixture_result("slurm.gputres", host, "sacct_completed_gpu_tres"),
        "slurm.qos": fixture_result("slurm.qos", host, "sacctmgr_qos"),
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


@pytest.mark.parametrize(("host", "parts"), [
    ("k2", ("gpu-preempt", "lab", "labpi")),          # lab/labpi: AllowGroups has one of yours; labpi: QOS labpi
    ("vega", ("a30", "h100", "h200")),               # every GPU partition is open; normal/dev/vdi have no GPUs
    ("lyra", ("free-gpu", "free-gpuxl", "gpu", "gpu-highmem", "gpuxl")),   # paid gpu/gpuxl via QOS normal/high
])
def test_gpu_partitions_you_may_submit_to(host: str, parts: tuple[str, ...]) -> None:
    assert caps_of(host)["gpu_partitions"] == parts


def _part(name: str, **f: str) -> str:
    fields = {"AllowGroups": "ALL", "AllowAccounts": "ALL", "AllowQos": "ALL", "Hidden": "NO", "State": "UP",
              "TRES": "cpu=64,mem=500G,node=1,gres/gpu=4"} | f
    return f"PartitionName={name} " + " ".join(f"{k}={v}" for k, v in fields.items())


def test_gpu_partitions_access_rules() -> None:
    me = (("c", "acct", "", "normal,high"),)
    lines = "\n".join([
        "groups=users,lab", "parents=complete",
        _part("open"), _part("cpu", TRES="cpu=64,mem=500G,node=1"), _part("zero", TRES="cpu=1,gres/gpu=0"),
        _part("hidden", Hidden="YES"), _part("inactive", State="INACTIVE"), _part("down", State="DOWN"),
        _part("grp-ok", AllowGroups="lab,other"), _part("grp-no", AllowGroups="other"),
        _part("acct-no", AllowAccounts="other"), _part("acct-deny", DenyAccounts="acct"),
        _part("qos-ok", AllowQos="high"), _part("qos-no", AllowQos="low"), _part("qos-deny", DenyQos="normal,high"),
        _part("bad;name"), _part("drain", State="DRAIN"), _part("root", RootOnly="YES"),
    ])
    assert gpu_partitions(lines, me) == ("down", "grp-ok", "open", "qos-ok")
    # an association bound to other partitions does not count; an empty QOS list passes only unrestricted QOS
    assert gpu_partitions(lines, (("c", "acct", "elsewhere", "normal"),)) == ()
    assert gpu_partitions(lines, (("c", "acct", "", ""),)) == ("down", "grp-ok", "open")
    # associations of another cluster sharing the slurmdbd do not count: none here = no access, not "no data"
    assert gpu_partitions(lines, (("other", "acct", "", "normal"),), "c") == ()
    assert gpu_partitions(lines, me, "c") == ("down", "grp-ok", "open", "qos-ok")
    # without associations (no sacctmgr) only partitions open to every account and QOS; without the groups line,
    # only AllowGroups=ALL
    assert gpu_partitions(lines, ()) == ("down", "grp-ok", "open")
    assert "grp-ok" not in gpu_partitions(lines.split("\n", 1)[1], me)


def test_gpu_partitions_follow_the_account_hierarchy() -> None:
    """Allow/DenyAccounts include sub-accounts: `parent=` lines give the chain above the user's accounts."""
    lines = "\n".join(["groups=users", "parent=c|lab|dept", "parent=c|dept|college", "parent=x|lab|elsewhere",
                        "parents=complete", _part("college-only", AllowAccounts="college"),
                        _part("dept-denied", DenyAccounts="dept"), _part("other-denied", DenyAccounts="nobody"),
                        _part("other", AllowAccounts="elsewhere")])
    me = (("c", "lab", "", "normal"),)
    assert gpu_partitions(lines, me, "c") == ("college-only", "other-denied")
    assert gpu_partitions(lines, me, "x") == ()                        # no association on cluster x
    # a chain that may be cut short (a sacctmgr call failed, > 6 levels): a DenyAccounts list cannot be checked
    partial = lines.replace("parents=complete\n", "")
    assert gpu_partitions(partial, me, "c") == ("college-only",)


def test_gpu_partitions_names_with_spaces_and_partition_associations() -> None:
    lines = "\n".join(["groups=domain users,lab", "parents=complete",
                        _part("team", AllowAccounts="research lab", AllowGroups="domain users"),
                        _part("closed", DenyAccounts="research lab"), _part("hi", AllowQos="high")])
    assert gpu_partitions(lines, (("c", "research lab", "", "normal"),), "c") == ("team",)   # hi wants QOS high
    # an association for the partition itself wins over the account's association for all partitions
    both = (("c", "lab", "", "high"), ("c", "lab", "hi", "low"))
    assert "hi" not in gpu_partitions(lines, both, "c")
    assert "hi" in gpu_partitions(lines, (("c", "lab", "", "high"), ("c", "lab", "other", "low")), "c")


# ---------------------------------------------------------------- charging rules (Account, §5.2.11)
@pytest.mark.parametrize(("host", "count", "free"), [
    ("lyra", 28, {"low", "guest", "free-part", "free-gpu-part"}),   # high ×2; free-gpuxl-part is ×1 (not decisive)
    ("k2", 7, set()), ("vega", 9, set()),                           # usage only: every factor 1
])
def test_qos_usage_factor(host: str, count: int, free: set[str]) -> None:
    f = caps_of(host)["qos_usage_factor"]
    assert isinstance(f, dict) and len(f) == count and f["normal"] == 1.0
    assert {q for q, v in f.items() if v == 0} == free
    if host == "lyra":
        assert f["high"] == 2.0 and f["free-gpuxl-part"] == 1.0
    else:
        assert set(f.values()) == {1.0}


def test_qos_bad_rows() -> None:
    pr = parse_qos("normal|1.000000\nlow|0.000000\nodd|x\nneg|-1\nbad name|1\n\nwide|0.5|DenyOnLimit\n")
    assert pr.items == [("normal", 1.0), ("low", 0.0), ("wide", 0.5)] and len(pr.bad_lines) == 3
    assert not pr.mostly_bad
    # one bad row is skipped; mostly bad rows leave the field unset (the cost shows the factor as unknown)
    assert parse_capabilities({"slurm.qos": result("slurm.qos", "normal|1\nodd|x")})["qos_usage_factor"] == \
        {"normal": 1.0}
    assert "qos_usage_factor" not in parse_capabilities({"slurm.qos": result("slurm.qos", "a|x\nb|y\nc|1")})
    assert "qos_usage_factor" not in parse_capabilities(
        {"slurm.qos": result("slurm.qos", "", 1, "sacctmgr: error: Problem talking to the database")})
    assert parse_capabilities({"slurm.qos": result("slurm.qos", "")})["qos_usage_factor"] == {}


def test_partition_charges_from_lyra() -> None:
    """lyra prints QoS and TRESBillingWeights on every line: free partitions admit only QOS low/guest and weight
    nothing, the charged ones weight GPUs (gpuxl per model, with the capitalised types scontrol prints)."""
    c = parse_capabilities({"slurm.assoc": fixture_result("slurm.assoc", "lyra", "sacctmgr_assoc"),
                            "slurm.parts": fixture_result("slurm.parts", "lyra", "scontrol_partitions_charges")})
    charges = {p.partition: p for p in c["partition_charges"]}  # type: ignore[attr-defined]
    assert list(charges) == ["free", "free-gpu", "free-gpuxl", "gpu", "gpu-highmem", "gpuxl", "standard",
                             "standard-mem"]
    assert charges["free-gpuxl"] == PartitionCharge("free-gpuxl", "", "free-gpuxl-part", ("low", "guest"))
    assert charges["gpu"] == PartitionCharge("gpu", "cpu=1,gres/gpu=32", "gpu-part", ("normal", "high"))
    assert charges["gpuxl"].billing_weights == "cpu=1,gres/gpu:L40S=32,gres/gpu:RTX6000=64"
    assert charges["standard"].qos == "normal" and charges["standard-mem"].allow_qos == ("standard-mem",
                                                                                         "standard-mem-pi02")
    assert all(p.def_cpu_per_gpu is None and p.def_mem_per_cpu_mb is None for p in charges.values())
    # the extra fields change nothing about where the user may submit
    assert c["gpu_partitions"] == caps_of("lyra")["gpu_partitions"]


@pytest.mark.parametrize(("host", "count"), [("k2", 5), ("vega", 6)])
def test_partition_charges_without_charging_fields(host: str, count: int) -> None:
    """The trimmed k2 / vega lines carry no QoS and no weights: billing = CPUs, every partition QOS unknown."""
    charges = caps_of(host)["partition_charges"]
    assert isinstance(charges, tuple) and len(charges) == count
    assert all(p.billing_weights == "" and p.qos is None for p in charges)
    assert [p.allow_qos for p in charges if p.allow_qos] == ([("labpi",)] if host == "k2" else [])


def test_partition_charges_synthetic() -> None:
    lines = "\n".join([
        "groups=lab", "parents=complete",
        _part("a", QoS="N/A", DefCpuPerGPU="8", DefMemPerCPU="4000", TRESBillingWeights="CPU=1.0,Mem=0.25G"),
        _part("b", AllowQos="low", DefMemPerCPU="UNLIMITED", TresBillingWeights="cpu=2"),
        _part("bad;name"), _part("c", AllowQos=""), "PartitionName= AllowQos=ALL",
    ])
    assert partition_charges(lines) == (
        PartitionCharge("a", "CPU=1.0,Mem=0.25G", None, None, 8, 4000),
        PartitionCharge("b", "cpu=2", None, ("low",)),
        PartitionCharge("c"))


def test_qos_section_is_cheap_and_guarded() -> None:
    (sec,) = [s for s in SlurmCollector.capability_sections(None) if s.name == "slurm.qos"]  # type: ignore[arg-type]
    assert sec.cmd == cmds.QOS_CMD == "sacctmgr -n -P show qos format=Name,UsageFactor"
    assert sec.requires == ("sacctmgr",) and sec.timeout == 15 and sec.max_stdout == 64_000


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
