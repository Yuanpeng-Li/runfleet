"""[WP2] Builders for collector tests: capabilities per fixture host, HostContext, SectionResults (§4.3 outcomes)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from runfleet.backends.base import HostContext
from runfleet.config import GeneralSettings, HostConfig, SlurmSettings, WorkstationSettings
from runfleet.models import DETECT_REV, Capabilities
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.conftest import meta, sample

# Field-probe day: 2026-09-07 ~05:58 (UTC-5) for k2/vega, ~03:59 (UTC-7) for lyra and the workstations.
NOW = datetime(2026, 9, 7, 11, 2, 37, tzinfo=UTC)

TZ = {"k2": timedelta(hours=-5), "vega": timedelta(hours=-5), "lyra": timedelta(hours=-7),
      "labstation": timedelta(hours=-7), "lynx": timedelta(hours=-7)}
USER = {"k2": "alice0001", "vega": "alice0001", "lyra": "alice07", "labstation": "alice07", "lynx": "alice07"}
UID = {"k2": 610001, "vega": 610001, "lyra": 2100007, "labstation": 1042, "lynx": 1042}
SLURM_TOOLS = frozenset({"squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel", "sacctmgr", "timeout", "less",
                         "stat", "ps", "pgrep"})
WS_TOOLS = frozenset({"nvidia-smi", "timeout", "less", "stat", "ps", "pgrep"})
SLURM_CONFIG = {
    "k2": {"ClusterName": "k2", "PreemptMode": "REQUEUE", "MinJobAge": "300 sec", "JobAcctGatherFrequency": "30"},
    "vega": {"ClusterName": "vega", "PreemptMode": "REQUEUE", "MinJobAge": "300 sec", "JobAcctGatherFrequency": "30"},
    "lyra": {"ClusterName": "lyra", "PreemptMode": "CANCEL", "MinJobAge": "300 sec", "JobAcctGatherFrequency": "30"},
}
GPU_TRES = {"k2": True, "vega": True, "lyra": False}


def outcome_for(stdout: str, rc: int | None, stderr: str = "") -> SectionOutcome:
    """§4.3 outcome derivation (the transport does this for real results)."""
    if rc is None:
        return SectionOutcome.INCOMPLETE
    if rc == 127 and stderr.startswith("missing:"):
        return SectionOutcome.MISSING
    if rc in (124, 137):
        return SectionOutcome.TIMEOUT
    if rc == 0:
        if stdout:
            return SectionOutcome.OK
        return SectionOutcome.STDERR_ONLY if stderr else SectionOutcome.EMPTY
    return SectionOutcome.FAILED


def result(name: str, stdout: str = "", rc: int | None = 0, stderr: str = "", *,
           outcome: SectionOutcome | None = None, ms: int | None = 5) -> SectionResult:
    return SectionResult(name=name, outcome=outcome or outcome_for(stdout, rc, stderr), rc=rc, stdout=stdout,
                         stderr=stderr, ms=ms)


def fixture_result(name: str, host: str, fixture: str, *, fold_stderr: bool = False) -> SectionResult:
    """SectionResult from a fixture + its meta rc/stderr; `fold_stderr` appends stderr to stdout (`2>&1` loops)."""
    m = meta(host, fixture)
    out = sample(host, fixture)
    if out.endswith("\n"):
        out = out[:-1]                               # the framing parser removes exactly one trailing newline
    err = m["stderr"]
    if fold_stderr and err:
        out = (out + "\n" + err) if out else err
        err = ""
    return result(name, out, m["rc"], err)


def caps(host: str = "k2", **overrides: Any) -> Capabilities:
    slurm = host in SLURM_CONFIG
    base = Capabilities(
        remote_user=USER[host], uid=UID[host], login_host=f"{host}-login", tz_offset=TZ[host],
        clock_skew=timedelta(0), tools=SLURM_TOOLS if slurm else WS_TOOLS,
        profile=("slurm", "procfs") if slurm else ("process", "nvidia", "procfs"),
        slurm_version={"k2": "25.05.8", "vega": "25.11.4", "lyra": "26.05.3"}.get(host),
        cluster_name=host if slurm else None, slurm_config=SLURM_CONFIG.get(host, {}),
        acct_gather_freq_s=30 if slurm else None, gpu_tres=GPU_TRES.get(host), detect_rev=DETECT_REV,
    )
    return dataclasses.replace(base, **overrides)


def host_config(host: str = "k2", *, slurm: Mapping[str, Any] | None = None,
                workstation: Mapping[str, Any] | None = None, **kw: Any) -> HostConfig:
    return HostConfig(name=host, ssh=host, slurm=SlurmSettings(**(slurm or {})),
                      workstation=WorkstationSettings(**(workstation or {})), **kw)


def ctx(host: str = "k2", *, now: datetime = NOW, cfg: HostConfig | None = None, readonly: bool = False,
        **cap_overrides: Any) -> HostContext:
    return HostContext(host=cfg or host_config(host), general=GeneralSettings(readonly=readonly),
                       caps=caps(host, **cap_overrides), now=now)


def at(c: HostContext, seconds: float) -> HostContext:
    return dataclasses.replace(c, now=c.now + timedelta(seconds=seconds))


def to_sep(text: str, sep: str) -> str:
    """Fixture rows use `|`; live commands use the unit separator."""
    return text.replace("|", sep)


_FORMAT = re.compile(r"--format=(\S+)")


def format_fields(cmd: str) -> list[str]:
    """`--format=JobID,…` list of a fixture's command (from _meta); the last one when the command has several."""
    found = _FORMAT.findall(cmd)
    assert found, cmd
    return str(found[-1]).split(",")
