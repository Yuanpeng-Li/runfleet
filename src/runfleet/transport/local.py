"""[WP1] LocalTransport: runs the rendered probe with `sh -s` locally (§4.5).

POSIX only: the probes are POSIX sh reading /proc, `ps` and Slurm on this machine. On native Windows every call
fails with one clear HostError (`WINDOWS_ERROR`) and nothing is spawned.
"""

from __future__ import annotations

import os
from typing import Literal

from runfleet.config import HostConfig
from runfleet.errors import RunFleetError
from runfleet.models import ErrorKind, HostError, MuxMode
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, StreamSpec
from runfleet.transport import classify, spawn
from runfleet.transport.base import ProbeResult, RunResult, StreamHandle
from runfleet.util import platform

WINDOWS_SUMMARY = "local host monitoring needs Linux/macOS; run RunFleet in WSL"


def windows_error() -> HostError:
    return HostError(ErrorKind.MISSING, WINDOWS_SUMMARY,
                     'ssh = "local" runs POSIX sh probes on this machine; native Windows has no sh, /proc or ps. '
                     "Run RunFleet inside WSL to monitor that machine, or monitor it over ssh from elsewhere.")


class LocalTransport:
    """This machine: same script, same framing, no ssh. Useful on a login node or for the dev box itself."""

    kind: Literal["openssh", "local", "fake", "demo"] = "local"

    def __init__(self, host: HostConfig, *, shell: str = "sh") -> None:
        self.host = host.name
        self.config = host
        self._sh = shell

    @property
    def mux(self) -> MuxMode:
        return "local"

    @property
    def ssh_log(self) -> str | None:
        return None

    async def prepare(self, *, force: bool = False) -> HostError | None:
        return windows_error() if platform.IS_WINDOWS else None

    async def check_master(self) -> bool:
        return True

    def _classify(self, timeout: float) -> spawn.ProbeClassifier:
        def run_classify(res: RunResult, parsed: framing.ParsedProbe) -> HostError | None:
            stderr = res.stderr.decode("utf-8", errors="replace")
            if parsed.ended > 0:
                return None
            if res.timed_out:
                return classify.classify_ssh(None, stderr, timed_out=True, timeout_s=timeout)
            if res.rc is not None and res.rc < 0:
                return classify.killed_by_signal(res.rc, stderr, what="sh")
            if parsed.markers == 0:
                return classify.remote_shell_error(res.rc, stderr)
            return None
        return run_classify

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        if platform.IS_WINDOWS:
            return spawn.failed_probe(script, windows_error())
        return await spawn.run_probe([self._sh, "-s"], script, timeout=timeout, classify=self._classify(timeout),
                                     label=self.host)

    async def stream(self, spec: StreamSpec) -> StreamHandle:
        if platform.IS_WINDOWS:
            raise RunFleetError(WINDOWS_SUMMARY)
        return await spawn.ProcessStream.start([self._sh, "-c", spec.cmd])

    def interactive_argv(self, remote_cmd: str | None) -> list[str]:
        if remote_cmd is None:
            return [os.environ.get("SHELL") or "sh"]
        return ["sh", "-c", remote_cmd]

    async def before_connect(self) -> bool:
        return False

    def connect_argv(self) -> list[str] | None:
        return None

    async def reset_master(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def forward_blocker(self) -> str | None:
        if platform.IS_WINDOWS:
            return WINDOWS_SUMMARY
        return "this machine: open the port directly, no forward needed"

    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return HostError(ErrorKind.SSH, self.forward_blocker() or "")

    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return None
