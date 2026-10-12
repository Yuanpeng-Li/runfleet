"""[WP2] Registry (builtins + entry points) and detection: detect sections, capability merge, profile choice (§5.1)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from runfleet.backends import registry
from runfleet.backends.base import Collector
from runfleet.backends.common.nvidia import NvidiaMonitor
from runfleet.backends.common.procfs import ProcfsMonitor
from runfleet.backends.detect import build_capabilities, choose_profile, detect_sections, parse_tz
from runfleet.backends.process.collector import ProcessCollector
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.errors import RunFleetError
from runfleet.probe.section import SectionResult
from tests.backends.factories import NOW, caps, fixture_result, host_config, result
from tests.conftest import sample


def tools_from_commands_fixture(host: str) -> str:
    """`commands` fixture (`name /path` or `name MISSING`) → what detect.tools prints (present names only)."""
    out = []
    for line in sample(host, "commands").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("/"):
            out.append(parts[0])
    return "\n".join(out)


def id_result(user: str, tz: str = "-0500", epoch: int | None = None) -> SectionResult:
    e = int(NOW.timestamp()) + 3 if epoch is None else epoch
    return result("detect.id", f"user={user}\nuid=610001\nhost=login02\ntz={tz}\nepoch={e}")


def test_builtin_registry() -> None:
    reg = registry.collectors()
    assert reg["slurm"] is SlurmCollector and reg["process"] is ProcessCollector
    assert reg["nvidia"] is NvidiaMonitor and reg["procfs"] is ProcfsMonitor
    assert all(cls.name == name for name, cls in reg.items())


class _EP:
    def __init__(self, name: str, target: Any, fail: bool = False) -> None:
        self.name, self.value, self._target, self._fail = name, f"pkg:{name}", target, fail

    def load(self) -> Any:
        if self._fail:
            raise ImportError("boom")
        return self._target


class PbsCollector(ProcessCollector):
    name = "pbs"


def test_entry_points_and_broken_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [_EP("pbs", PbsCollector), _EP("broken", None, fail=True), _EP("notacollector", object),
           _EP("wrongname", PbsCollector), _EP("slurm", PbsCollector)]
    monkeypatch.setattr(registry, "entry_points", lambda group: eps if group == registry.ENTRY_POINT_GROUP else [])
    reg = registry.collectors()
    assert reg["pbs"] is PbsCollector
    assert "broken" not in reg and "notacollector" not in reg and "wrongname" not in reg
    assert reg["slurm"] is SlurmCollector                 # builtins cannot be shadowed


def test_detect_sections() -> None:
    secs = detect_sections(host_config("k2"), registry.collectors())
    names = [s.name for s in secs]
    assert names[:2] == ["detect.id", "detect.tools"]
    assert set(names) == {"detect.id", "detect.tools", "slurm.version", "slurm.config", "slurm.assoc",
                          "slurm.sacctfmt", "slurm.gputres", "nvidia.version"}
    by = {s.name: s for s in secs}
    assert by["slurm.version"].requires == ("sinfo",) and by["nvidia.version"].requires == ("nvidia-smi",)
    assert "date +%z" in by["detect.id"].cmd and "command -v" in by["detect.tools"].cmd
    assert "-E now" in by["slurm.gputres"].cmd


class _BadCaps(Collector):
    name = "bad"

    @classmethod
    def applicable(cls, caps: Any, host: Any) -> bool:
        return False

    @classmethod
    def capability_sections(cls, host: Any) -> list[Any]:
        raise RuntimeError("plugin bug")

    @classmethod
    def ingest_capabilities(cls, results: Any) -> Any:
        raise RuntimeError("plugin bug")

    def cadences(self, ctx: Any) -> list[Any]:
        return []

    def sections(self, due: Any, ctx: Any) -> list[Any]:
        return []

    def ingest(self, results: Any, ctx: Any, monitors: Any) -> Any:
        raise NotImplementedError


def test_k2_capabilities_and_profile() -> None:
    reg = {**registry.collectors(), "bad": _BadCaps}
    assert [s.name for s in detect_sections(host_config("k2"), reg)][:2] == ["detect.id", "detect.tools"]
    results = {
        "detect.id": id_result("alice0001"),
        "detect.tools": result("detect.tools", tools_from_commands_fixture("k2")),
        "slurm.version": fixture_result("slurm.version", "k2", "sinfo_version"),
        "slurm.config": fixture_result("slurm.config", "k2", "slurm_config"),
        "slurm.assoc": fixture_result("slurm.assoc", "k2", "sacctmgr_assoc"),
        "slurm.gputres": fixture_result("slurm.gputres", "k2", "sacct_completed_gpu_tres"),
        "nvidia.version": result("nvidia.version", "", 127, "missing:nvidia-smi"),
    }
    c = build_capabilities(results, host_config("k2"), NOW, reg)
    assert c.remote_user == "alice0001" and c.uid == 610001 and c.login_host == "login02"
    assert c.tz_offset == timedelta(hours=-5) and c.clock_skew == timedelta(seconds=3)
    assert {"squeue", "sacct", "sstat", "timeout"} <= c.tools and "nvidia-smi" not in c.tools
    assert c.profile == ("slurm", "procfs")
    assert c.slurm_version == "25.05.8" and c.cluster_name == "k2" and c.gpu_tres is True
    assert c.slurm_config["PreemptMode"] == "REQUEUE" and c.acct_gather_freq_s == 30
    assert c.driver_version is None


def test_workstation_capabilities() -> None:
    results = {"detect.id": id_result("alice07", "-0700"),
               "detect.tools": result("detect.tools", tools_from_commands_fixture("lynx")),
               "nvidia.version": result("nvidia.version", sample("lynx", "nvidia_smi_version"))}
    c = build_capabilities(results, host_config("lynx"), NOW, registry.collectors())
    assert c.profile == ("process", "nvidia", "procfs") and c.driver_version == "610.43.02"
    assert c.tz_offset == timedelta(hours=-7) and c.gpu_tres is None


def test_detect_id_failures() -> None:
    reg = registry.collectors()
    with pytest.raises(RunFleetError):
        build_capabilities({}, host_config("k2"), NOW, reg)
    with pytest.raises(RunFleetError):
        build_capabilities({"detect.id": result("detect.id", "", 1, "sh: id: not found")}, host_config("k2"), NOW, reg)
    with pytest.raises(RunFleetError):
        build_capabilities({"detect.id": result("detect.id", "user=x\ntz=CEST\nepoch=1")}, host_config("k2"), NOW,
                           reg)


def test_choose_profile_rules() -> None:
    slurm_tools = frozenset({"squeue", "sinfo"})
    assert choose_profile(caps("k2", tools=slurm_tools), host_config("k2")) == ("slurm", "procfs")
    assert choose_profile(caps("k2", tools=slurm_tools | {"nvidia-smi"}), host_config("k2")) == \
        ("slurm", "procfs", "nvidia")
    assert choose_profile(caps("lynx", tools=frozenset({"nvidia-smi"})), host_config("lynx")) == \
        ("process", "nvidia", "procfs")
    assert choose_profile(caps("lynx", tools=frozenset({"ps"})), host_config("lynx")) == ("process", "procfs")
    assert choose_profile(caps("lynx", tools=frozenset()), host_config("lynx", backend="slurm")) == \
        ("slurm", "procfs")
    assert choose_profile(caps("k2", tools=slurm_tools), host_config("k2", backend="workstation")) == \
        ("process", "procfs")
    assert SlurmCollector.applicable(caps("k2", tools=slurm_tools), host_config("k2"))
    assert not ProcessCollector.applicable(caps("k2", tools=slurm_tools), host_config("k2"))
    assert NvidiaMonitor.applicable(caps("lynx"), host_config("lynx"))
    assert ProcfsMonitor.applicable(caps("lynx"), host_config("lynx"))


def test_parse_tz() -> None:
    assert parse_tz("+0530") == timedelta(hours=5, minutes=30)
    assert parse_tz("-07:00") == timedelta(hours=-7)
    with pytest.raises(ValueError):
        parse_tz("EST")
