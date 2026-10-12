"""[WP4] `runfleet wait` (§8.5): the real FleetService + pollers over FakeTransport and stub collectors.

Exit codes: 0 all COMPLETED, 1 ended otherwise, 2 timeout, 3 error (unknown host / job, needs a login, unreachable).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from runfleet import wait
from runfleet.config import FleetConfig, GeneralSettings, HostConfig
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.models import ConnState, ErrorKind, HostSnapshot, HostStatus, Job, JobKey, JobKind, JobState
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, run_for

K2 = HostConfig(name="k2", ssh="k2-alias", backend="slurm")


class World:
    def __init__(self, tmp: Path, squeue: str = "101 R\n102 PD") -> None:
        self.clock = ManualClock(CLOCK_START)
        self.t = FakeTransport("k2", clock=self.clock)
        self.t.set_result("detect.id", "user=me")
        self.t.set_result("slurm.squeue", squeue)
        general = GeneralSettings(readonly=True, idle_pause_after=wait.NEVER)
        self.svc = FleetService(FleetConfig(hosts=(K2,), general=general), lambda _h: self.t, self.clock, None)
        self.svc.hooks = STUB_HOOKS
        self.svc.lock_path = tmp / "instance.lock"
        self.lines: list[str] = []

    def start(self, *targets: str, timeout: float | None = None, as_json: bool = False) -> asyncio.Task[int]:
        resolved = wait.resolve_targets(targets, (K2,))
        return asyncio.ensure_future(wait.wait_for(self.svc, resolved, clock=self.clock, timeout=timeout,
                                                   emit=self.lines.append, as_json=as_json))


@pytest.fixture
def world(tmp_path: Path) -> Callable[..., World]:
    return lambda **kw: World(tmp_path, **kw)


async def test_completed_jobs_exit_0_and_print_transitions(world: Callable[..., World]) -> None:
    w = world()
    task = w.start("k2:101", "k2-alias:102")                      # the ssh alias names the host too
    await run_for(w.clock, 20)
    assert not task.done()
    assert any("k2:101  R" in ln for ln in w.lines) and any("k2:102  PD" in ln for ln in w.lines)
    w.t.set_result("slurm.squeue", "101 CD\n102 R")
    await run_for(w.clock, 20)
    assert any("k2:102  PD → R" in ln for ln in w.lines) and not task.done()
    w.t.set_result("slurm.squeue", "101 CD\n102 CD")
    await run_for(w.clock, 20)
    assert task.done() and task.result() == wait.EXIT_COMPLETED
    assert "all 2 jobs completed" in w.lines[-1]
    assert w.svc._tasks == {}                                      # pollers stopped
    assert all(p.script.purpose in ("detect", "poll") for p in w.t.probes)   # read-only: no action probe


async def test_failed_job_exits_1_with_json_lines(world: Callable[..., World]) -> None:
    w = world()
    task = w.start("k2:101", as_json=True)
    await run_for(w.clock, 20)
    w.t.set_result("slurm.squeue", "101 F")
    await run_for(w.clock, 20)
    assert task.result() == wait.EXIT_FAILED
    docs = [json.loads(ln) for ln in w.lines]
    assert all(d["schema"] == "runfleet.wait/1" for d in docs)
    assert [d["state"] for d in docs if d["event"] == "state"] == ["R", "F"]
    assert docs[-1]["event"] == "done" and docs[-1]["result"] == "failed" and docs[-1]["jobs"] == {"k2:101": "F"}


async def test_timeout_exits_2(world: Callable[..., World]) -> None:
    w = world()
    task = w.start("k2:101", timeout=60)
    await run_for(w.clock, 70)
    assert task.done() and task.result() == wait.EXIT_TIMEOUT and "still waiting for k2:101" in w.lines[-1]


async def test_unknown_job_and_needs_auth_exit_3(world: Callable[..., World]) -> None:
    w = world()
    task = w.start("k2:999")
    await run_for(w.clock, 40)
    assert task.result() == wait.EXIT_ERROR and "no such job" in w.lines[-1]

    w = world()
    w.t.prepare_errors.append(None)
    w.t.fail_next(ErrorKind.AUTH)
    task = w.start("k2:101")
    await run_for(w.clock, 20)
    assert task.result() == wait.EXIT_ERROR and "runfleet connect k2" in w.lines[-1]


def test_targets_resolve_and_reject() -> None:
    assert wait.resolve_targets(["k2:31270818_3"], (K2,)) == [wait.Target("k2", "31270818_3", "k2:31270818_3")]
    for bad in ("k2", ":1", "k2:", "nohost:1"):
        with pytest.raises(wait.TargetError):
            wait.resolve_targets([bad], (K2,))


# ---------------------------------------------------------------- Waiter on a hand-made store (arrays, tasks)
@dataclass
class _HS:
    status: HostStatus
    snapshot: HostSnapshot | None


@dataclass
class _Store:
    hosts: dict[str, Any] = field(default_factory=dict)


def _snap(*jobs: Job, n: int = 0) -> HostSnapshot:
    from datetime import timedelta
    return HostSnapshot("k2", CLOCK_START + timedelta(seconds=n), None, {j.key.id: j for j in jobs}, (), (), (), (),
                        None)


def _job(jid: str, state: JobState, *, array: str | None = None, kind: JobKind = JobKind.SLURM,
         raw: str | None = None, source: str = "squeue") -> Job:
    return Job(key=JobKey("k2", jid), kind=kind, name=f"n{jid}", user="me", state=state, raw_state=state.name,
               job_id_raw=raw or jid, array_job_id=array, source=source)   # type: ignore[arg-type]


def test_array_target_waits_for_every_task() -> None:
    lines: list[str] = []
    waiter = wait.Waiter([wait.Target("k2", "500", "k2:500")], emit=lines.append, now=lambda: CLOCK_START)
    ok = HostStatus("k2", ConnState.OK)
    store = _Store({"k2": _HS(ok, _snap(_job("500_[2-3]", JobState.PENDING, array="500"),
                                        _job("500_1", JobState.RUNNING, array="500"), n=1))})
    assert waiter.check(store, 0) is None
    store.hosts["k2"].snapshot = _snap(_job("500_1", JobState.COMPLETED, array="500"),
                                       _job("500_2", JobState.COMPLETED, array="500"),
                                       _job("500_3", JobState.RUNNING, array="500"), n=2)
    assert waiter.check(store, 1) is None
    store.hosts["k2"].snapshot = _snap(*(_job(f"500_{i}", JobState.COMPLETED, array="500") for i in (1, 2, 3)), n=3)
    assert waiter.check(store, 2) == wait.EXIT_COMPLETED


def test_task_label_vanished_unknown_and_unreachable() -> None:
    lines: list[str] = []
    task = _job("p642229-1788696459", JobState.EXITED, kind=JobKind.TASK, raw="642229")
    t = wait.Target("k2", "p642229", "k2:p642229")
    assert t.matches({task.key.id: task}) == [task]
    assert wait.Target("k2", "642229", "x").matches({task.key.id: task}) == [task]
    lost = _job("7", JobState.UNKNOWN, source="vanished")
    assert wait.is_final(lost) and not wait.is_final(_job("8", JobState.RUNNING, source="vanished"))
    waiter = wait.Waiter([t], emit=lines.append, now=lambda: CLOCK_START)
    down = HostStatus("k2", ConnState.UNREACHABLE)
    store = _Store({"k2": _HS(down, None)})
    assert waiter.check(store, 0) is None
    assert waiter.check(store, wait.UNREACHABLE_GIVE_UP) == wait.EXIT_ERROR and "unreachable" in lines[-1]
    waiter = wait.Waiter([t], emit=lines.append, now=lambda: CLOCK_START)
    store = _Store({"k2": _HS(HostStatus("k2", ConnState.OK), _snap(task, n=1))})
    assert waiter.check(store, 0) == wait.EXIT_FAILED          # a task's exit status is unknown: not "completed"



def test_array_task_that_failed_and_aged_out_still_fails_the_wait() -> None:
    waiter = wait.Waiter([wait.Target("k2", "500", "k2:500")], emit=[].append, now=lambda: CLOCK_START)
    store = _Store({"k2": _HS(HostStatus("k2", ConnState.OK),
                              _snap(_job("500_1", JobState.FAILED, array="500"),
                                    _job("500_2", JobState.RUNNING, array="500"), n=1))})
    assert waiter.check(store, 0) is None
    store.hosts["k2"].snapshot = _snap(_job("500_2", JobState.RUNNING, array="500"), n=2)   # 500_1 left sacct
    assert waiter.check(store, 1) is None
    store.hosts["k2"].snapshot = _snap(_job("500_2", JobState.COMPLETED, array="500"), n=3)
    assert waiter.check(store, 2) == wait.EXIT_FAILED


def test_duplicate_targets_finish() -> None:
    t = wait.Target("k2", "101", "k2:101")
    waiter = wait.Waiter([t, t], emit=[].append, now=lambda: CLOCK_START)
    store = _Store({"k2": _HS(HostStatus("k2", ConnState.OK), _snap(_job("101", JobState.COMPLETED), n=1))})
    assert waiter.check(store, 0) == wait.EXIT_COMPLETED


def test_unknown_job_needs_answered_job_listings() -> None:
    from datetime import timedelta

    from runfleet.models import SectionStatus

    def status(ok_at: int | None) -> HostStatus:
        at = CLOCK_START + timedelta(seconds=ok_at) if ok_at is not None else None
        sq = SectionStatus("slurm.squeue", "ok" if ok_at else "timeout", ok_at is not None, 0, 10, at)
        return HostStatus("k2", ConnState.DEGRADED, sections={"slurm.squeue": sq})

    waiter = wait.Waiter([wait.Target("k2", "999", "k2:999")], emit=[].append, now=lambda: CLOCK_START)
    store = _Store({"k2": _HS(status(None), _snap(n=1))})
    for n in range(2, 6):                       # squeue keeps failing; metrics-only ticks renew the snapshot
        store.hosts["k2"].snapshot = _snap(n=n)
        assert waiter.check(store, n) is None
    store.hosts["k2"].status = status(10)
    assert waiter.check(store, 10) is None      # one answer
    store.hosts["k2"].snapshot = _snap(n=11)
    assert waiter.check(store, 11) is None      # same squeue answer, newer snapshot: still one
    store.hosts["k2"].status = status(20)
    assert waiter.check(store, 20) == wait.EXIT_ERROR
