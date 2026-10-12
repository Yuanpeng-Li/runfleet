"""[WP1] FakeTransport for tests: scripted SectionResults per section name.

Results are looked up per section, in order: a one-shot queued result, the `responder` callable, a persistent
scripted result, else an EMPTY result. Transport-level failures are queued with `fail_next`. Every probe is recorded
(`probes`, with the purpose and the monotonic time from the optional clock) so tests can count round trips.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from runfleet.models import ErrorKind, HostError, MuxMode
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, SectionResult, StreamSpec
from runfleet.transport.base import ProbeResult, RunResult, StreamHandle

Responder = Callable[[Section], "SectionResult | None"]


class _Clock(Protocol):
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


def result(name: str, stdout: str = "", rc: int | None = 0, stderr: str = "", *, ms: int | None = 1,
           outcome: SectionOutcome | None = None) -> SectionResult:
    """A SectionResult with the outcome derived exactly like the framing parser does (unless given)."""
    if outcome is None:
        outcome = framing.outcome_for(rc, stdout, stderr, missing=rc == 127 and stderr.startswith("missing:"))
    return SectionResult(name=name, outcome=outcome, rc=rc, stdout=stdout, stderr=stderr, ms=ms)


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    script: ProbeScript
    timeout: float
    at: float | None                     # clock.monotonic() at the call, when a clock was given

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.script.sections)


class FakeStream:
    """StreamHandle that yields scripted chunks, then (if `hold_open`) waits until closed, like `tail -F`."""

    def __init__(self, chunks: Iterable[bytes], *, stderr: str = "", hold_open: bool = False,
                 returncode: int | None = 0) -> None:
        self._chunks = list(chunks)
        self._stderr = stderr
        self._hold = hold_open
        self._rc = returncode
        self._closed = asyncio.Event()
        self.closed = False

    async def chunks(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            if self.closed:
                return
            yield c
            await asyncio.sleep(0)
        if self._hold:
            await self._closed.wait()

    async def close(self) -> None:
        self.closed = True
        self._closed.set()

    @property
    def stderr_text(self) -> str:
        return self._stderr

    @property
    def returncode(self) -> int | None:
        return self._rc if (self.closed or not self._hold) else None


@dataclass
class FakeTransport:
    host: str
    mux_mode: MuxMode = "none"
    responder: Responder | None = None
    clock: _Clock | None = None
    latency: float = 0.0                 # seconds of clock.sleep per probe (needs `clock`)
    kind: Literal["openssh", "local", "fake", "demo"] = "fake"
    master_ok: bool = True
    results: dict[str, SectionResult] = field(default_factory=dict)
    queued: dict[str, deque[SectionResult]] = field(default_factory=dict)
    errors: deque[HostError] = field(default_factory=deque)
    prepare_errors: deque[HostError | None] = field(default_factory=deque)
    streams: dict[str, Callable[[StreamSpec], FakeStream]] = field(default_factory=dict)
    probes: list[ProbeRecord] = field(default_factory=list)
    opened_streams: list[FakeStream] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    forced_prepares: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    forward_block: str | None = None     # forward_blocker() answer (a host without a usable master)
    forward_errors: deque[HostError] = field(default_factory=deque)   # next forward() calls fail with these
    forwarded: set[tuple[int, str, int]] = field(default_factory=set)  # active (local port, node, remote port)
    cancel_errors: deque[HostError] = field(default_factory=deque)    # next cancel_forward() calls fail with these

    # ------------------------------------------------------------ scripting
    def set_result(self, name: str, stdout: str = "", rc: int | None = 0, stderr: str = "", *,
                   outcome: SectionOutcome | None = None) -> None:
        self.results[name] = result(name, stdout, rc, stderr, outcome=outcome)

    def queue(self, name: str, res: SectionResult) -> None:
        self.queued.setdefault(name, deque()).append(replace(res, name=name))

    def fail_next(self, error: HostError | ErrorKind, times: int = 1, summary: str = "") -> None:
        err = error if isinstance(error, HostError) else HostError(error, summary or f"fake {error.value}")
        self.errors.extend([err] * times)

    def set_stream(self, name: str, factory: Callable[[StreamSpec], FakeStream]) -> None:
        self.streams[name] = factory

    def count(self, purpose: str | None = None) -> int:
        return sum(1 for p in self.probes if purpose is None or p.script.purpose == purpose)

    # ------------------------------------------------------------ Transport protocol
    @property
    def mux(self) -> MuxMode:
        return self.mux_mode

    @property
    def ssh_log(self) -> str | None:
        return None

    async def prepare(self, *, force: bool = False) -> HostError | None:
        self.calls.append("prepare")
        if force:
            self.forced_prepares += 1
        return self.prepare_errors.popleft() if self.prepare_errors else None

    async def check_master(self) -> bool:
        self.calls.append("check_master")
        return self.master_ok

    def _answer(self, s: Section) -> SectionResult:
        q = self.queued.get(s.name)
        if q:
            return q.popleft()
        if self.responder is not None:
            r = self.responder(s)
            if r is not None:
                return replace(r, name=s.name)
        if s.name in self.results:
            return self.results[s.name]
        return result(s.name, "", 0, "")

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        self.probes.append(ProbeRecord(script, timeout, self.clock.monotonic() if self.clock else None))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.clock is not None and self.latency > 0:
                await self.clock.sleep(self.latency)
            else:
                await asyncio.sleep(0)
            run = RunResult(rc=0, stdout=b"", stderr=b"", elapsed_s=self.latency, timed_out=False)
            if self.errors:
                err = self.errors.popleft()
                parsed = framing.parse(b"", script)
                return ProbeResult(run=replace(run, rc=255), sections=parsed.sections, error=err)
            sections: Mapping[str, SectionResult] = {s.name: self._answer(s) for s in script.sections}
            return ProbeResult(run=run, sections=sections, error=None)
        finally:
            self.in_flight -= 1

    async def stream(self, spec: StreamSpec) -> StreamHandle:
        self.calls.append(f"stream:{spec.name}")
        factory = self.streams.get(spec.name)
        s = factory(spec) if factory is not None else FakeStream([])
        self.opened_streams.append(s)
        return s

    def interactive_argv(self, remote_cmd: str | None) -> list[str]:
        return ["fake-ssh", "-t", self.host] + ([remote_cmd] if remote_cmd else [])

    async def before_connect(self) -> bool:
        self.calls.append("before_connect")
        return False

    def connect_argv(self) -> list[str] | None:
        return ["fake-ssh", "-fN", self.host]

    async def reset_master(self) -> None:
        self.calls.append("reset_master")

    async def close(self) -> None:
        self.calls.append("close")

    def forward_blocker(self) -> str | None:
        return self.forward_block

    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        self.calls.append(f"forward:{local_port}:{node}:{remote_port}")
        if self.forward_errors:
            return self.forward_errors.popleft()
        self.forwarded.add((local_port, node, remote_port))
        return None

    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        self.calls.append(f"cancel_forward:{local_port}:{node}:{remote_port}")
        if self.cancel_errors:
            return self.cancel_errors.popleft()
        self.forwarded.discard((local_port, node, remote_port))
        return None
