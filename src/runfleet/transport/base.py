"""[WP1] FROZEN Transport protocol, RunResult, ProbeResult, StreamHandle (DESIGN.md §3.5)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from runfleet.models import HostError, MuxMode
from runfleet.probe.section import ProbeScript, SectionResult, StreamSpec

@dataclass(frozen=True, slots=True)
class RunResult:
    rc: int | None               # None if killed by our local timeout
    stdout: bytes
    stderr: bytes
    elapsed_s: float
    timed_out: bool

@dataclass(frozen=True, slots=True)
class ProbeResult:
    run: RunResult
    sections: Mapping[str, SectionResult]   # every requested section present (unfinished -> TIMEOUT/INCOMPLETE/NOT_RUN)
    error: HostError | None                 # transport-level failure, classified (§9). None if ssh ran, including a
                                            # local timeout after >= 1 section end marker (§4.2 partial rule)

class StreamHandle(Protocol):
    def chunks(self) -> AsyncIterator[bytes]: ...   # raw stdout blocks (read(65536)), never line-split; ends at EOF
    async def close(self) -> None: ...              # close stdin (remote watchdog kills tail), then terminate
    @property
    def stderr_text(self) -> str: ...               # last <= 4 KiB of stderr
    @property
    def returncode(self) -> int | None: ...

class Transport(Protocol):
    host: str                                       # RunFleet host name
    kind: Literal["openssh", "local", "fake", "demo"]
    @property
    def mux(self) -> MuxMode: ...                   # resolved by prepare(); "none" before
    @property
    def ssh_log(self) -> str | None: ...            # `-E` log path for masters/connect (None: local/fake/demo)
    async def prepare(self, *, force: bool = False) -> HostError | None: ...  # ssh -G, mux policy, ensure master
                                                    # (BatchMode); idempotent; force: user retry, skip 30 s throttle
    async def check_master(self) -> bool: ...       # `ssh -O check`; True when no master is involved
    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult: ...   # never raises
    async def stream(self, spec: StreamSpec) -> StreamHandle: ...
    def interactive_argv(self, remote_cmd: str | None) -> list[str]: ...  # `ssh -t …` run under App.suspend
    async def before_connect(self) -> bool: ...     # resolve + clear a stale socket; True = master already up
    def connect_argv(self) -> list[str] | None: ...  # `ssh -fN … -E <log>`; None if not applicable (local/fake)
    async def reset_master(self) -> None: ...       # managed masters only: `ssh -O exit`, unlink socket
    async def close(self) -> None: ...
    # local port forwards through the existing master (`ssh -O forward|cancel -L 127.0.0.1:lport:node:rport`);
    # nothing runs remotely. forward_blocker: why this transport cannot forward (no master), None if it can
    def forward_blocker(self) -> str | None: ...
    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None: ...
    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None: ...
