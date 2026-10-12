"""[WP1] Per-host diagnostics -> DoctorReport.

`runfleet doctor` (and the palette's "Doctor <host>"): ssh -G summary, mux mode and jump rewrite, master check, one
detect probe (latency, profile, tools, clock, gpu_tres). At most two ssh round trips: master creation if needed and
the detect probe; `ssh -G` and `-O check` are local.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from runfleet.config import HostConfig
from runfleet.core.clock import Clock
from runfleet.core.poller import BackendHooks, HostLanes
from runfleet.errors import RunFleetError
from runfleet.models import Capabilities, DoctorCheck, DoctorReport
from runfleet.probe import framing
from runfleet.transport import mux as muxpolicy
from runfleet.transport.base import Transport
from runfleet.transport.openssh import OpenSSHTransport

log = logging.getLogger("runfleet.core.doctor")

SKEW_WARN_S = 60.0
EXPECTED_TOOLS: dict[str, tuple[str, ...]] = {
    "slurm": ("squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel"),
    "process": ("ps", "pgrep"),
    "nvidia": ("nvidia-smi",),
}
NICE_TOOLS = ("timeout", "less", "stat")


async def run_doctor(host: HostConfig, transport: Transport, *, hooks: BackendHooks, clock: Clock,
                     reset: bool = False, on_caps: Callable[[Capabilities], None] | None = None,
                     lanes: HostLanes | None = None) -> DoctorReport:
    """`lanes` (the service's): master creation and the probe share the poller's session budget (§4.2)."""
    checks: list[DoctorCheck] = []
    warnings: list[str] = []

    if isinstance(transport, OpenSSHTransport):
        err = await transport.resolve()
        if err is not None:
            checks.append(DoctorCheck("ssh -G", False, err.summary))
            return DoctorReport(host=host.name, checks=tuple(checks), warnings=tuple(warnings))
        assert transport.settings is not None and transport.plan is not None
        s = transport.settings
        checks.append(DoctorCheck("ssh -G", True, " ".join(f"{k}={v}" for k, v in s.summary().items())))
        rewrite = ("ProxyJump rewritten to a BatchMode ProxyCommand" if muxpolicy.needs_rewrite(s)
                   else "no jump rewrite")
        checks.append(DoctorCheck("mux", None, f"{transport.plan.mode} ({transport.plan.control_path or 'no socket'});"
                                               f" {rewrite}"))
    else:
        checks.append(DoctorCheck("ssh -G", None, f"{transport.kind} transport (no ssh)"))
        checks.append(DoctorCheck("mux", None, transport.mux))

    if reset:
        await transport.reset_master()
        checks.append(DoctorCheck("reset", None, "managed master closed" if transport.mux == "managed"
                                  else "nothing to reset (only managed masters are closed)"))

    async with lanes.poll() if lanes is not None else contextlib.nullcontext():
        err = await transport.prepare()
    if err is not None:
        checks.append(DoctorCheck("master", False, f"{err.kind.value}: {err.summary}"))
        if transport.ssh_log:
            warnings.append(f"see {transport.ssh_log}")
        return DoctorReport(host=host.name, checks=tuple(checks), warnings=tuple(warnings))
    up = await transport.check_master()
    checks.append(DoctorCheck("master", up, "master up" if transport.mux in ("user", "managed") else "not used"))

    regs = hooks.collectors()
    script = framing.make_script(hooks.detect_sections(host, regs), "doctor")
    async with lanes.demand() if lanes is not None else contextlib.nullcontext():
        res = await transport.probe(script, timeout=framing.local_timeout(script))
    if res.error is not None:
        checks.append(DoctorCheck("probe", False, f"{res.error.kind.value}: {res.error.summary}"))
        return DoctorReport(host=host.name, checks=tuple(checks), warnings=tuple(warnings))
    checks.append(DoctorCheck("probe", True, f"{res.run.elapsed_s:.2f} s round trip"))
    try:
        caps = hooks.build_capabilities(res.sections, host, clock.now(), regs)
    except RunFleetError as e:
        checks.append(DoctorCheck("profile", False, f"detection failed: {e}"))
        return DoctorReport(host=host.name, checks=tuple(checks), warnings=tuple(warnings))
    if on_caps is not None:
        on_caps(caps)
    checks.append(DoctorCheck("profile", bool(caps.profile), " + ".join(caps.profile) or "no profile"))
    expected = [t for name in caps.profile for t in EXPECTED_TOOLS.get(name, ())]
    missing = [t for t in expected if t not in caps.tools]
    nice_missing = [t for t in NICE_TOOLS if t not in caps.tools]
    detail = "all expected tools present" if not missing else f"missing: {', '.join(missing)}"
    if nice_missing:
        detail += f"; optional missing: {', '.join(nice_missing)}"
    checks.append(DoctorCheck("tools", not missing, detail))
    if "timeout" in nice_missing:
        warnings.append(f"{host.name}: no remote `timeout`; only the local probe timeout bounds commands")
    skew = caps.clock_skew.total_seconds()
    tz_h = caps.tz_offset.total_seconds() / 3600
    checks.append(DoctorCheck("clock", abs(skew) <= SKEW_WARN_S, f"UTC{tz_h:+g}, skew {skew:+.0f} s"))
    if abs(skew) > SKEW_WARN_S:
        warnings.append(f"{host.name}: clock skew {skew:+.0f} s")
    if "slurm" in caps.profile:
        text = {True: "records gres/gpuutil", False: "no GPU accounting", None: "unknown (no recent GPU job)"}
        checks.append(DoctorCheck("gpu_tres", True if caps.gpu_tres else None, text[caps.gpu_tres]))
        if caps.gpu_tres is False:
            warnings.append(f"{host.name}: no GPU accounting — job GPU util unavailable")
    return DoctorReport(host=host.name, checks=tuple(checks), warnings=tuple(warnings))
