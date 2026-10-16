"""A scriptable stand-in for `FleetAPI` in CLI tests (no transports, no network)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from runfleet.config import FleetConfig, HostConfig
from runfleet.core.store import HostState
from runfleet.models import (Capabilities, ConnState, DetectReport, DoctorCheck, DoctorReport, ErrorKind, HostError,
                             HostSnapshot, HostStatus, Job, JobKey, JobKind, JobState)
from tests.conftest import CLOCK_START


@dataclass
class SimpleStore:
    """Just `hosts` (what export.py and cli.py read)."""

    hosts: dict[str, HostState] = field(default_factory=dict)


def caps(user: str = "u", *, slurm: str | None = None, driver: str | None = None) -> Capabilities:
    profile = ("slurm", "procfs") if slurm else ("process", "nvidia", "procfs")
    return Capabilities(remote_user=user, uid=1000, login_host="login", tz_offset=timedelta(hours=-5),
                        clock_skew=timedelta(0), tools=frozenset(), profile=profile, slurm_version=slurm,
                        driver_version=driver)


def report(alias: str, state: ConnState = ConnState.OK, *, slurm: str | None = "25.05.8",
           driver: str | None = None, mux: str = "user", latency: float | None = 0.21,
           error: str | None = None) -> DetectReport:
    err = HostError(kind=ErrorKind.AUTH, summary=error) if error else None
    ok = state in (ConnState.OK, ConnState.DEGRADED)
    return DetectReport(alias=alias, state=state, error=err, caps=caps(slurm=slurm, driver=driver) if ok else None,
                        mux=mux, ssh_summary={}, latency_s=latency if ok else None)  # type: ignore[arg-type]


def job(host: str, jid: str, state: JobState = JobState.RUNNING) -> Job:
    return Job(key=JobKey(host, jid), kind=JobKind.SLURM, name=f"job-{jid}", user="u", state=state,
               raw_state=state.name, job_id_raw=jid)


def host_state(name: str, state: ConnState = ConnState.OK, *, jobs: Sequence[Job] = (),
               with_snapshot: bool = True, error: str | None = None) -> HostState:
    status = HostStatus(name=name, state=state, latency_s=0.2,
                        error=HostError(ErrorKind.TIMEOUT, error) if error else None)
    snap = HostSnapshot(host=name, taken_at=CLOCK_START, caps=caps(slurm="25.05.8"),
                        jobs={j.key.id: j for j in jobs}, gpus=(), procs=(), nodes=(), partitions=(),
                        metrics=None) if with_snapshot else None
    return HostState(config=HostConfig(name=name, ssh=name), status=status, snapshot=snap)


class FakeService:
    """Records calls; answers from the scripted fields."""

    def __init__(self, config: FleetConfig, transports: Callable[[HostConfig], Any] | None = None) -> None:
        self.config = config
        self.transports = transports
        self.readonly = config.general.readonly
        self.notices: list[str] = []
        self.store = SimpleStore()
        self.calls: list[tuple[str, Any]] = []
        self.reports: dict[str, DetectReport] = {}
        self.doctor_reports: dict[str, DoctorReport | Exception] = {}
        self.collected: dict[str, HostState] = {}
        self.written_path = Path("/tmp/runfleet-test-config.toml")
        self.connect: list[str] | None = ["ssh", "-fN", "x"]
        self.already_connected = False
        self.login: Callable[[], int] | None = None      # connect_login: set for a session-transport host (§4.8)
        self.lock_path: object = None
        self.demo_mode = False
        self.settings_open = False
        self.config_path: Path | None = config.source
        self.alias_source: Any = None
        self.remembered: list[str] = []

    async def probe_alias(self, alias: str) -> DetectReport:
        self.calls.append(("probe_alias", alias))
        return self.reports.get(alias, report(alias, ConnState.UNREACHABLE, error="connection timed out"))

    def write_config(self, hosts: Sequence[HostConfig], path: Path | None = None) -> Path:
        self.calls.append(("write_config", (list(hosts), path)))
        target = path or self.written_path
        target.write_text("".join(f'[[host]]\nname = "{h.name}"\n' for h in hosts), encoding="utf-8")
        return target

    async def doctor(self, host: str, *, reset: bool = False) -> DoctorReport:
        self.calls.append(("doctor", (host, reset)))
        rep = self.doctor_reports.get(host, DoctorReport(host=host, checks=(DoctorCheck("probe", True, "ok"),)))
        if isinstance(rep, Exception):
            raise rep
        return rep

    async def collect_once(self, hosts: Iterable[str] | None = None, *, timeout: float = 60) -> None:
        self.calls.append(("collect_once", (list(hosts or []), timeout)))
        self.store.hosts.update(self.collected)

    async def prepare_connect(self, host: str) -> bool:
        self.calls.append(("prepare_connect", host))
        return self.already_connected

    def connect_argv(self, host: str) -> list[str] | None:
        self.calls.append(("connect_argv", host))
        return self.connect

    def connect_login(self, host: str) -> Callable[[], int] | None:
        return self.login

    def remember_hosts(self, names: Sequence[str]) -> None:
        self.calls.append(("remember_hosts", list(names)))
        self.remembered.extend(names)

    def called(self, name: str) -> list[Any]:
        return [arg for n, arg in self.calls if n == name]
