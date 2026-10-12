"""[WP2] Detect sections, capability merge and profile selection (DESIGN.md §5.1)."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping
from datetime import datetime, timedelta

from runfleet.backends.base import Collector
from runfleet.config import HostConfig
from runfleet.errors import RunFleetError
from runfleet.models import Capabilities
from runfleet.probe.section import Section, SectionResult

log = logging.getLogger("runfleet.backends.detect")

DETECT_TOOLS = ("squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel", "sacctmgr", "nvidia-smi", "timeout",
                "less", "stat", "ps", "pgrep", "srun", "ssh")   # srun/ssh: Attach's login-side tools (§7.16)

ID_CMD = ("printf 'user=%s\\nuid=%s\\nhost=%s\\ntz=%s\\nepoch=%s\\n' \"$(id -un)\" \"$(id -u)\" \"$(hostname)\" "
          "\"$(date +%z)\" \"$(date +%s)\"")
TOOLS_CMD = ("for c in " + " ".join(DETECT_TOOLS)
             + "; do command -v \"$c\" >/dev/null 2>&1 && echo \"$c\"; done; true")

_TZ = re.compile(r"^([+-])(\d{2}):?(\d{2})$")

# Capabilities fields a collector may set through ingest_capabilities (anything else is ignored with a warning).
_CAP_FIELDS = frozenset({"slurm_version", "cluster_name", "slurm_config", "acct_gather_freq_s", "gpu_tres",
                         "sacct_fields", "associations", "driver_version"})


def detect_sections(host: HostConfig, collectors: Mapping[str, type[Collector]]) -> list[Section]:
    # detect.id + detect.tools + every collector's capability_sections(host); core renders them as one probe
    # (purpose="detect")
    out = [Section("detect.id", ID_CMD, timeout=10), Section("detect.tools", TOOLS_CMD, timeout=10)]
    names = {s.name for s in out}
    for cname, cls in collectors.items():
        try:
            sections = cls.capability_sections(host)
        except Exception:                           # a broken plugin must not break detection
            log.exception("collector %s: capability_sections failed", cname)
            continue
        for sec in sections:
            if sec.owner != cname or sec.name in names:
                log.warning("collector %s: skipping capability section %s (wrong owner or duplicate)", cname, sec.name)
                continue
            names.add(sec.name)
            out.append(sec)
    return out


def parse_tz(value: str) -> timedelta:
    m = _TZ.match(value.strip())
    if m is None:
        raise ValueError(f"bad timezone offset {value!r}")
    sign = -1 if m.group(1) == "-" else 1
    return sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))


def build_capabilities(results: Mapping[str, SectionResult], host: HostConfig, now: datetime,
                       collectors: Mapping[str, type[Collector]]) -> Capabilities:
    # pure: parses detect.id (tz_offset from %z; clock_skew = remote epoch - now) and detect.tools; merges each
    # collector's ingest_capabilities(results whose owner is that collector) in registry order; sets profile via
    # choose_profile. Raises errors.RunFleetError if detect.id is not OK or unparseable (core reports it as a
    # REMOTE/PARSE host error and retries per §4.4)
    rid = results.get("detect.id")
    if rid is None or not rid.ok:
        why = "missing" if rid is None else f"{rid.outcome.value}: {rid.stderr.strip()[:200]}"
        raise RunFleetError(f"detect.id failed ({why})")
    values: dict[str, str] = {}
    for line in rid.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    try:
        user = values["user"]
        if not user:
            raise ValueError("empty user")
        tz = parse_tz(values["tz"])
        epoch = int(values["epoch"])
    except (KeyError, ValueError) as e:
        raise RunFleetError(f"detect.id unparseable: {e}") from e
    uid = int(values["uid"]) if values.get("uid", "").isdigit() else None
    rtools = results.get("detect.tools")
    tools = frozenset(line.strip() for line in (rtools.stdout if rtools is not None else "").splitlines()
                      if line.strip())
    fields: dict[str, object] = {}
    for cname, cls in collectors.items():
        own = {name: r for name, r in results.items() if r.name.split(".", 1)[0] == cname}
        try:
            patch = cls.ingest_capabilities(own)
        except Exception:
            log.exception("collector %s: ingest_capabilities failed", cname)
            continue
        for field_name, field_value in patch.items():
            if field_name in _CAP_FIELDS:
                fields[field_name] = field_value
            else:
                log.warning("collector %s: ignoring unknown capability %s", cname, field_name)
    caps = Capabilities(remote_user=user, uid=uid, login_host=values.get("host", ""), tz_offset=tz,
                        clock_skew=timedelta(seconds=epoch - now.timestamp()), tools=tools)
    caps = dataclasses.replace(caps, **fields)  # type: ignore[arg-type]  # values are checked per field below
    _check_types(caps)
    return dataclasses.replace(caps, profile=choose_profile(caps, host))


def _check_types(caps: Capabilities) -> None:
    """Reject a plugin's capability values of the wrong type early (they are untyped `object`s)."""
    expected: dict[str, tuple[type, ...]] = {
        "slurm_version": (str, type(None)), "cluster_name": (str, type(None)), "slurm_config": (Mapping,),
        "acct_gather_freq_s": (int, type(None)), "gpu_tres": (bool, type(None)), "sacct_fields": (frozenset,),
        "associations": (tuple,), "driver_version": (str, type(None)),
    }
    for name, types in expected.items():
        if not isinstance(getattr(caps, name), types):
            raise RunFleetError(f"capability {name} has type {type(getattr(caps, name)).__name__}")


def choose_profile(caps: Capabilities, host: HostConfig) -> tuple[str, ...]:
    """§5.1: Slurm (`backend = "slurm"`, or auto with squeue + sinfo) → slurm + procfs (+ nvidia on a GPU login node);
    workstation (`backend = "workstation"`, or auto with nvidia-smi) → process + nvidia + procfs; else a CPU-only box →
    process + procfs."""
    nvidia = "nvidia-smi" in caps.tools
    if host.backend == "slurm" or (host.backend == "auto" and {"squeue", "sinfo"} <= caps.tools):
        return ("slurm", "procfs", "nvidia") if nvidia else ("slurm", "procfs")
    if host.backend == "workstation":
        return ("process", "nvidia", "procfs") if nvidia else ("process", "procfs")
    if nvidia:
        return ("process", "nvidia", "procfs")
    return ("process", "procfs")
