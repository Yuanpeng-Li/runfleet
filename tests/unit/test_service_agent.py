"""FleetService additions for `runfleet agent` (DESIGN.md §3.7, §5.4, §8.7): collect_once(keep_cached=True) loads
only the polled hosts' cached snapshots and keeps the cached jobs when the listing fails; a TUI takes the instance
locks it could not get at start once they are free, without a second notice."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from runfleet.config import FleetConfig, HostConfig
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService, instance_lock_file, lock_holder, release_instance_lock, try_instance_lock
from runfleet.demo.transport import DemoTransport, Out
from runfleet.demo.world import DemoWorld
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, run_for

K2 = HostConfig(name="k2", ssh="k2", backend="slurm")
LYNX = HostConfig(name="lynx", ssh="lynx", backend="workstation")


# ---------------------------------------------------------------- instance lock retry (D12)
async def test_a_lock_held_at_start_is_taken_once_free_without_a_second_notice(tmp_path: Path) -> None:
    clock = ManualClock(CLOCK_START)
    base = tmp_path / "instance.lock"
    k2_lock = instance_lock_file(base, "k2")
    held, _ = try_instance_lock(k2_lock)                  # an `agent --refresh` holds k2 while the TUI starts
    assert held is not None
    svc = FleetService(FleetConfig(hosts=(K2, LYNX)), lambda h: FakeTransport(h.name), clock, None)
    svc.hooks = STUB_HOOKS
    svc.lock_path = base
    try:
        await svc.start()
        notice = f"another RunFleet (pid {os.getpid()}) is polling k2 — remote load doubles there"
        assert svc.notices == [notice]
        assert set(svc._host_locks) == {"lynx"}
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)   # still held: retried, not taken
        assert set(svc._host_locks) == {"lynx"} and svc.notices == [notice]
        release_instance_lock(held)
        held = None
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)
        assert set(svc._host_locks) == {"k2", "lynx"}
        assert try_instance_lock(k2_lock) == (None, str(os.getpid()))    # ours now: a later refresh skips k2
        assert svc.notices == [notice]                                    # said once
        await svc.stop()
        assert lock_holder(k2_lock) is None
    finally:
        if held is not None:
            release_instance_lock(held)
        await svc.stop()


async def test_lock_retry_skips_removed_and_disabled_hosts(tmp_path: Path) -> None:
    clock = ManualClock(CLOCK_START)
    base = tmp_path / "instance.lock"
    held = [try_instance_lock(instance_lock_file(base, h))[0] for h in ("k2", "lynx")]
    svc = FleetService(FleetConfig(hosts=(K2, LYNX)), lambda h: FakeTransport(h.name), clock, None)
    svc.hooks = STUB_HOOKS
    svc.lock_path = base
    try:
        await svc.start()
        assert svc._lock_missing == {"k2", "lynx"} and len(svc.notices) == 1
        await svc.remove_host("k2", save=False)
        svc.set_host_enabled("lynx", False)
        for fd in held:
            assert fd is not None
            release_instance_lock(fd)
        held = []
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)
        assert svc._host_locks == {} and svc._lock_missing == {"lynx"}    # disabled: kept for when re-enabled
        svc.set_host_enabled("lynx", True)
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)
        assert set(svc._host_locks) == {"lynx"} and not svc._lock_missing
    finally:
        for fd in held:
            if fd is not None:
                release_instance_lock(fd)
        await svc.stop()


# ---------------------------------------------------------------- collect_once(keep_cached=True) (D1)
class SqueueDown(DemoTransport):
    def _answer(self, sim, section, now, purpose):  # type: ignore[no-untyped-def]
        if section.name == "slurm.squeue":
            return Out("", 1, "slurm_load_jobs error: Unable to contact slurm controller (connect failure)")
        return super()._answer(sim, section, now, purpose)


async def test_keep_cached_loads_only_the_polled_hosts_and_keeps_jobs_of_a_failing_listing(tmp_path: Path) -> None:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config(["k2", "vega"])
    path = tmp_path / "cache.sqlite"
    cache = Cache(path)
    first = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    first.lock_path = False
    await first.collect_once(["k2", "vega"], timeout=30)
    await first.stop()
    before = cache.load_snapshots()
    assert before["k2"].jobs and before["vega"].jobs
    clock.advance(300)
    plain = FleetService(config, lambda h: SqueueDown(h, world), clock, cache)
    plain.lock_path = False
    keep = FleetService(config, lambda h: SqueueDown(h, world), clock, cache)
    keep.lock_path = False
    await keep.collect_once(["k2"], timeout=30, keep_cached=True)
    assert keep.store.hosts["vega"].snapshot is None                     # not polled: not loaded either
    k2 = keep.store.hosts["k2"].snapshot
    assert k2 is not None and k2.jobs.keys() == before["k2"].jobs.keys() and k2.taken_at == clock.now()
    assert cache.load_snapshots()["k2"].jobs.keys() == before["k2"].jobs.keys()
    await keep.stop()
    await plain.collect_once(["k2"], timeout=30)                         # `snapshot`: no cached rows, ever
    lost = plain.store.hosts["k2"].snapshot
    assert lost is not None and lost.jobs.keys() < before["k2"].jobs.keys()        # printed as polled …
    await plain.stop()
    assert cache.load_snapshots()["k2"].jobs.keys() == before["k2"].jobs.keys()     # … never cached (C2)
    await cache.close()


async def test_lock_of_a_shared_destination_passes_to_the_other_name_on_removal(tmp_path: Path) -> None:
    """Codex: two names for one ssh destination share one lock; removing the owner must not leave the other
    polling without it."""
    clock = ManualClock(CLOCK_START)
    base = tmp_path / "instance.lock"
    a, b = HostConfig("lab-a", "shared.example"), HostConfig("lab-b", "shared.example")
    svc = FleetService(FleetConfig(hosts=(a, b)), lambda h: FakeTransport(h.name), clock, None)
    svc.hooks = STUB_HOOKS
    svc.lock_path = base
    try:
        await svc.start()
        assert set(svc._host_locks) == {"lab-a"}
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)
        assert set(svc._host_locks) == {"lab-a"} and not svc._lock_missing      # covered by lab-a: no retrying
        await svc.remove_host("lab-a", save=False)
        await run_for(clock, FleetService.LOCK_RETRY_EVERY + 5, step=5)
        assert set(svc._host_locks) == {"lab-b"}
        assert lock_holder(instance_lock_file(base, "shared.example")) == str(os.getpid())
    finally:
        await svc.stop()


class Hangs(DemoTransport):
    async def probe(self, script, *, timeout):  # type: ignore[no-untyped-def]
        await asyncio.sleep(3600)


async def test_keep_cached_restores_jobs_even_when_a_deadline_cancels_the_other_hosts(tmp_path: Path) -> None:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config(["k2", "vega"])
    cache = Cache(tmp_path / "cache.sqlite")
    first = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    first.lock_path = False
    await first.collect_once(["k2", "vega"], timeout=30)
    await first.stop()
    before = cache.load_snapshots()["k2"].jobs.keys()
    clock.advance(300)
    svc = FleetService(config, lambda h: SqueueDown(h, world) if h.name == "k2" else Hangs(h, world), clock, cache)
    svc.lock_path = False
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(svc.collect_once(["k2", "vega"], timeout=30, keep_cached=True), 0.5)
    await svc.stop()                                     # flushes what the poll queued
    assert cache.load_snapshots()["k2"].jobs.keys() == before
    await cache.close()


# ---------------------------------------------------------------- accounting down (C1), `snapshot` (C2)
class SacctDown(DemoTransport):
    def _answer(self, sim, section, now, purpose):  # type: ignore[no-untyped-def]
        if section.name.startswith("slurm.sacct"):
            return Out("", 1, "sacct: error: slurmdbd: connection refused")
        return super()._answer(sim, section, now, purpose)


async def _cached_fleet(tmp_path: Path, hosts: list[str]) -> tuple[ManualClock, DemoWorld, FleetConfig, Cache]:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config(["k2", "vega"])
    cache = Cache(tmp_path / "cache.sqlite")
    first = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    first.lock_path = False
    await first.collect_once(hosts, timeout=30)
    await first.stop()
    clock.advance(300)
    return clock, world, config, cache


async def test_keep_cached_with_accounting_down_keeps_the_cached_finished_jobs(tmp_path: Path) -> None:
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"].jobs
    finished = {k for k, j in before.items() if j.state.is_terminal}
    assert finished
    svc = FleetService(config, lambda h: SacctDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30, keep_cached=True)
    snap = svc.store.hosts["k2"].snapshot
    assert snap is not None and finished <= snap.jobs.keys()
    assert all(snap.jobs[k] == before[k] for k in finished)
    assert all(j.source == "squeue" for j in snap.jobs.values() if not j.state.is_terminal)
    assert finished <= cache.load_snapshots()["k2"].jobs.keys()
    await svc.stop()
    await cache.close()


def _row(cache: Cache, table: str, host: str) -> Any:
    db = sqlite3.connect(cache.path)
    try:
        return db.execute(f"SELECT * FROM {table} WHERE host = ?", (host,)).fetchone()
    finally:
        db.close()


@pytest.mark.parametrize("down", [SqueueDown, SacctDown])
async def test_snapshot_with_a_failing_listing_or_accounting_never_drops_the_cached_jobs(
        tmp_path: Path, down: type[DemoTransport]) -> None:
    """C2: `snapshot` (no keep_cached) prints what this poll saw, but a partial job list must not replace the
    cached one: the row written is the poll merged with the cached jobs (as keep_cached), so the rest of it
    (partitions, GPUs) matches the host_health written with it; the failure itself is still written."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"].jobs
    squeue_ok = json.loads(_row(cache, "host_health", "k2")[2])["sections"]["slurm.squeue"]["last_ok"]
    svc = FleetService(config, lambda h: down(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    printed = svc.store.hosts["k2"].snapshot
    assert printed is not None and printed.taken_at == clock.now()      # the output is this poll, as before
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert row.taken_at == clock.now()
    if down is SqueueDown:
        assert row.jobs.keys() == before.keys() | printed.jobs.keys()   # every cached job, fresh rows winning
    else:
        finished = {k for k, j in before.items() if j.state.is_terminal}
        assert finished and finished <= row.jobs.keys()                # the cached finished jobs …
        assert all(row.jobs[k].source == "squeue" for k, j in row.jobs.items() if not j.state.is_terminal)
    assert all(row.jobs[k] == j for k, j in printed.jobs.items())       # … and every fresh row as polled
    status = _row(cache, "host_status", "k2")
    assert status[1] == clock.now().timestamp() and status[2] == "degraded"
    health = json.loads(_row(cache, "host_health", "k2")[2])["sections"]
    if down is SqueueDown:                                              # the kept jobs' age: the old success
        assert health["slurm.squeue"]["ok"] is False and health["slurm.squeue"]["last_ok"] == squeue_ok
    else:
        assert health["slurm.sacct"]["ok"] is False and health["slurm.squeue"]["ok"] is True
    await cache.close()


async def test_snapshot_never_merges_the_rows_of_another_destination(tmp_path: Path) -> None:
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"].jobs
    moved = FleetConfig(hosts=tuple(dataclasses.replace(h, ssh="k2-other.example") if h.name == "k2" else h
                                    for h in config.hosts), general=config.general)
    svc = FleetService(moved, lambda h: SqueueDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    printed = svc.store.hosts["k2"].snapshot
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert printed is not None and row.jobs.keys() == printed.jobs.keys()     # the other machine's jobs: not ours
    assert before.keys() - row.jobs.keys()
    await cache.close()


async def test_snapshot_writes_a_partial_poll_of_a_host_never_cached(tmp_path: Path) -> None:
    clock, world, config, cache = await _cached_fleet(tmp_path, ["vega"])
    assert _row(cache, "snapshots", "k2") is None
    svc = FleetService(config, lambda h: SacctDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    await svc.stop()
    row = _row(cache, "snapshots", "k2")
    assert row is not None and row[1] == clock.now().timestamp()        # nothing good to lose: what it saw
    await cache.close()


async def test_a_cancelled_snapshot_still_keeps_the_cached_jobs(tmp_path: Path) -> None:
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2", "vega"])
    before = cache.load_snapshots()["k2"].jobs.keys()
    svc = FleetService(config, lambda h: SqueueDown(h, world) if h.name == "k2" else Hangs(h, world), clock, cache)
    svc.lock_path = False
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(svc.collect_once(["k2", "vega"], timeout=30), 0.5)
    await svc.stop()                                     # flushes what was queued: k2 merged, not k2 wiped
    assert before <= cache.load_snapshots()["k2"].jobs.keys()
    await cache.close()


# ---------------------------------------------------------------- stop() waits for a flush in flight (C6)
async def test_stop_keeps_the_locks_until_a_flush_in_flight_has_committed(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """C6: the flush loop is cancelled while its worker thread writes; stop()'s own flush found nothing pending
    and returned at once, releasing the instance locks mid-write. Cache.flush now waits for it."""
    clock = ManualClock(CLOCK_START)
    base = tmp_path / "instance.lock"
    cache = Cache(tmp_path / "cache.sqlite")
    svc = FleetService(FleetConfig(hosts=(K2,)), lambda h: FakeTransport(h.name), clock, cache)
    svc.hooks = STUB_HOOKS
    svc.lock_path = base
    lock = instance_lock_file(base, "k2")
    writing, seen = threading.Event(), []
    real = Cache._write

    def slow_write(self: Cache, *a: Any) -> None:
        writing.set()
        time.sleep(0.3)
        seen.append(lock_holder(lock))
        real(self, *a)
    monkeypatch.setattr(Cache, "_write", slow_write)
    await svc.start()
    try:
        cache.put_status("k2", svc.store.hosts["k2"].status, clock.now())
        inflight = threading.Thread(target=cache.flush_sync)          # what a cancelled flush loop leaves running
        inflight.start()
        assert await asyncio.to_thread(writing.wait, 5)
        await svc.stop()
        assert seen and set(seen) == {str(os.getpid())}                 # ours for the whole write
        assert lock_holder(lock) is None
        inflight.join(5)
    finally:
        await svc.stop()
        await cache.close()


@pytest.mark.parametrize(("squeue_ok", "sacct_ok", "expected"), [
    (True, False, {"1", "2", "4"}),            # accounting down: finished 4 back; R 3 left the listing: not revived
    (False, False, {"1", "2", "3", "4"}),      # listing down: every cached job back
    (False, True, {"1", "2", "3", "4"}),
    (True, True, {"1", "2"}),                  # both fine: the poll as is
])
def test_merged_jobs_per_failed_group(squeue_ok: bool, sacct_ok: bool, expected: set[str]) -> None:
    """C1: which cached jobs come back per failed group; a fresh row always wins over the cached one."""
    from runfleet.models import HostSnapshot, JobState, SectionStatus
    from tests.cli.fakes import job

    svc = FleetService(FleetConfig(hosts=(K2,)), lambda h: FakeTransport(h.name), ManualClock(CLOCK_START), None)

    def snap(*jobs: Any) -> HostSnapshot:
        return HostSnapshot("k2", CLOCK_START, None, {j.key.id: j for j in jobs}, (), (), (), (), None)
    old = snap(job("k2", "1", JobState.FAILED), job("k2", "3"), job("k2", "4", JobState.COMPLETED))
    fresh = snap(job("k2", "1", JobState.COMPLETED), job("k2", "2"))
    svc.store.set_status("k2", dataclasses.replace(svc.store.hosts["k2"].status, sections={
        "slurm.squeue": SectionStatus("slurm.squeue", "ok" if squeue_ok else "failed", squeue_ok, 0, 1, None),
        "slurm.sacct": SectionStatus("slurm.sacct", "ok" if sacct_ok else "failed", sacct_ok, 0, 1, None)}))
    merged = svc._merged_jobs("k2", fresh, old)
    got = fresh if merged is None else merged
    assert set(got.jobs) == expected
    assert got.jobs["1"].state is JobState.COMPLETED                                  # the fresh row wins


# ---------------------------------------------------------------- partitions / GPUs / metrics of a failed group (K1)
class NodesDown(DemoTransport):
    """sinfo, nvidia-smi and /proc all fail; the job listing answers."""

    def _answer(self, sim, section, now, purpose):  # type: ignore[no-untyped-def]
        if section.name == "slurm.sinfo":
            return Out("", 1, "slurm_load_node error: Unable to contact slurm controller (connect failure)")
        if section.name in ("nvidia.gpu", "nvidia.apps"):
            return Out("", 255, "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error")
        if section.name == "procfs.host":
            return Out("", 1, "cat: /proc/meminfo: Input/output error")
        return super()._answer(sim, section, now, purpose)


async def test_snapshot_with_sinfo_down_keeps_the_cached_partitions_and_nodes(tmp_path: Path) -> None:
    """K1: `snapshot` (no keep_cached) of a host whose sinfo fails printed no partitions (right: this poll saw
    none) but cached them as none, wiping the last good reading the agent shows with its age."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"]
    assert before.partitions and before.nodes and before.metrics is not None
    svc = FleetService(config, lambda h: NodesDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    printed = svc.store.hosts["k2"].snapshot
    assert printed is not None and printed.partitions == () and printed.nodes == ()   # printed as polled …
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert row.taken_at == clock.now()
    assert row.partitions == before.partitions and row.nodes == before.nodes          # … cached as before
    assert row.metrics == before.metrics
    assert row.jobs.keys() == printed.jobs.keys()                                     # the listing answered
    health = json.loads(_row(cache, "host_health", "k2")[2])["sections"]
    assert health["slurm.sinfo"]["ok"] is False and health["slurm.sinfo"]["last_ok"] is not None
    await cache.close()


async def test_refresh_with_sinfo_down_keeps_the_cached_partitions_and_nodes(tmp_path: Path) -> None:
    """K1, keep_cached: the store starts from the cached row and a failing sinfo answers None (not an empty
    tuple), so the partitions stay, in the store and in the cache."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"]
    svc = FleetService(config, lambda h: NodesDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30, keep_cached=True)
    snap = svc.store.hosts["k2"].snapshot
    assert snap is not None and snap.taken_at == clock.now()
    assert snap.partitions == before.partitions and snap.nodes == before.nodes and snap.metrics == before.metrics
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert row.partitions == before.partitions and row.nodes == before.nodes and row.metrics == before.metrics
    await cache.close()


@pytest.mark.parametrize("keep_cached", [False, True])
async def test_gpus_and_metrics_of_a_workstation_survive_a_failing_nvidia_smi(tmp_path: Path,
                                                                             keep_cached: bool) -> None:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config(["labstation"])
    cache = Cache(tmp_path / "cache.sqlite")
    first = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    first.lock_path = False
    await first.collect_once(["labstation"], timeout=30)
    await first.stop()
    before = cache.load_snapshots()["labstation"]
    assert before.gpus and before.metrics is not None
    clock.advance(300)
    svc = FleetService(config, lambda h: NodesDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["labstation"], timeout=30, keep_cached=keep_cached)
    await svc.stop()
    row = cache.load_snapshots()["labstation"]
    assert row.taken_at == clock.now()
    assert [g.index for g in row.gpus] == [g.index for g in before.gpus] and row.metrics == before.metrics
    await cache.close()


@pytest.mark.parametrize("rc", [6, 9])
def test_merged_keeps_an_empty_gpu_list_nvidia_smi_reported_as_no_devices(rc: int) -> None:
    """Codex: nvidia-smi rc 6 / "No devices were found" is an answer (NvidiaMonitor reports gpus=()), though the
    section counts as failed: the cached GPUs of a box whose GPUs went away must not come back."""
    from runfleet.models import ErrorKind, HostError, HostSnapshot, SectionStatus

    svc = FleetService(FleetConfig(hosts=(K2,)), lambda h: FakeTransport(h.name), ManualClock(CLOCK_START), None)
    gpu = object()
    old = HostSnapshot("k2", CLOCK_START, None, {}, (gpu,), (), (), (), None)  # type: ignore[arg-type]
    fresh = HostSnapshot("k2", CLOCK_START, None, {}, (), (), (), (), None)
    err = HostError(ErrorKind.REMOTE, "nvidia.gpu: No devices were found", detail="No devices were found",
                    section="nvidia.gpu")
    svc.store.set_status("k2", dataclasses.replace(svc.store.hosts["k2"].status, sections={
        "nvidia.gpu": SectionStatus("nvidia.gpu", "failed", False, rc, 1, None, error=err)}))
    got = svc._merged("k2", fresh, old) or fresh
    assert got.gpus == ()


class GpuLogin(DemoTransport):
    """k2 with nvidia-smi on its login node (profile slurm + procfs + nvidia): the nvidia sections are answered
    from the labstation simulation; `sinfo_down` makes only sinfo fail."""

    sinfo_down = False

    def _answer(self, sim, section, now, purpose):  # type: ignore[no-untyped-def]
        if section.name == "slurm.sinfo" and self.sinfo_down:
            return Out("", 1, "slurm_load_node error: Unable to contact slurm controller (connect failure)")
        if section.name.startswith("nvidia."):
            sim = self.world.hosts["labstation"]
        return super()._answer(sim, section, now, purpose)


class GpuLoginSinfoDown(GpuLogin):
    sinfo_down = True


async def test_sinfo_down_with_login_node_gpus_answering_keeps_partitions_and_nodes(tmp_path: Path) -> None:
    """F1: on a Slurm host whose profile also runs nvidia (login-node GPUs), a working nvidia.gpu made the "nodes"
    group look alive, so a failing sinfo wiped the cached partitions and nodes."""
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    world.hosts["k2"].tools = world.hosts["k2"].tools | {"nvidia-smi"}
    config = world.fleet_config(["k2"])
    cache = Cache(tmp_path / "cache.sqlite")
    first = FleetService(config, lambda h: GpuLogin(h, world), clock, cache)
    first.lock_path = False
    await first.collect_once(["k2"], timeout=30)
    assert first.store.hosts["k2"].status.sections["nvidia.gpu"].ok
    await first.stop()
    before = cache.load_snapshots()["k2"]
    assert before.partitions and before.nodes and before.gpus
    clock.advance(300)
    svc = FleetService(config, lambda h: GpuLoginSinfoDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    sections = svc.store.hosts["k2"].status.sections
    assert not sections["slurm.sinfo"].ok and sections["nvidia.gpu"].ok and sections["procfs.host"].ok
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert row.taken_at == clock.now()
    assert row.partitions == before.partitions and row.nodes == before.nodes
    assert row.gpus                                                       # this poll's nvidia-smi answer
    await cache.close()
    # final review 2: the agent dates the carried-over partitions by sinfo (5 min old, failing), never by the
    # login node's working nvidia-smi (age 0); the login GPUs by their own probe
    from runfleet.agent import view
    from runfleet.agent.source import load_view

    fv = load_view(config, ["k2"], cache_path=tmp_path / "cache.sqlite", now=clock.now(), lock_base=None)
    hv = fv.hosts[0]
    assert hv.kind == "slurm" and hv.data_at("nodes") == CLOCK_START
    assert view.nodes_at(hv) == view.nodes_at(hv, "partitions") == CLOCK_START
    assert view.nodes_at(hv, "gpus") == clock.now()
    assert view.host_json(hv, fv)["data"]["nodes_at"] == view.iso(CLOCK_START)
    host = view.host(fv, "k2")
    assert host.data["host"]["data"]["nodes_at"] == view.iso(CLOCK_START) and "nodes 5m" in host.text
    assert view.nodes_problem(hv, clock.now()) is not None                # sinfo failing: the counts are old
    free = view.brief(fv).data["free_gpus"]
    parts = [f for f in free if f["kind"] == "partition"]
    login = [f for f in free if f["kind"] == "gpu_model"]
    assert parts and all(f["nodes_age_s"] == 300 and f["stale"] for f in parts)
    assert login and all(f["nodes_age_s"] == 0 and not f["stale"] for f in login)
    gpus = view.gpus(fv).data
    assert gpus["partitions"] and all(p["nodes_age_s"] == 300 and p["stale"] for p in gpus["partitions"])


_FIELDS = ("partitions", "nodes", "gpus", "metrics", "procs")


@pytest.mark.parametrize(("failed", "ok", "carried"), [
    ({"slurm.sinfo"}, {"procfs.host", "nvidia.gpu"}, {"partitions", "nodes"}),
    ({"slurm.sinfo"}, {"procfs.host"}, {"partitions", "nodes"}),
    ({"procfs.host"}, {"slurm.sinfo"}, {"metrics"}),
    ({"slurm.sinfo", "procfs.host"}, set(), {"partitions", "nodes", "metrics"}),
    ({"nvidia.gpu"}, {"slurm.sinfo", "procfs.host"}, {"gpus"}),                   # login-node / workstation GPUs
    ({"nvidia.gpu", "procfs.host"}, {"process.ps"}, {"gpus", "metrics"}),         # workstation
    ({"process.ps"}, {"nvidia.gpu", "procfs.host"}, {"procs"}),
    ({"slurm.squeue"}, {"slurm.sinfo"}, set()),                # a failing squeue is no reason to revive procs
    (set(), {"slurm.sinfo", "procfs.host", "nvidia.gpu", "process.ps"}, set()),
])
def test_merged_carries_each_field_whose_own_section_failed_and_the_poll_left_empty(
        failed: set[str], ok: set[str], carried: set[str]) -> None:
    from runfleet.models import HostSnapshot, SectionStatus

    svc = FleetService(FleetConfig(hosts=(K2,)), lambda h: FakeTransport(h.name), ManualClock(CLOCK_START), None)
    part, node, gpu, metrics, proc = (object() for _ in range(5))     # identity only: never inspected
    old = HostSnapshot("k2", CLOCK_START, None, {}, (gpu,), (proc,), (node,), (part,),  # type: ignore[arg-type]
                       metrics)  # type: ignore[arg-type]
    fresh = HostSnapshot("k2", CLOCK_START, None, {}, (), (), (), (), None)
    sections = {n: SectionStatus(n, "failed", False, 0, 1, None) for n in failed}
    sections.update({n: SectionStatus(n, "ok", True, 0, 1, None) for n in ok})
    svc.store.set_status("k2", dataclasses.replace(svc.store.hosts["k2"].status, sections=sections))
    merged = svc._merged("k2", fresh, old)
    got = fresh if merged is None else merged
    assert {f for f in _FIELDS if getattr(got, f) == getattr(old, f)} == carried
    partial = dataclasses.replace(fresh, partitions=(part, part))  # type: ignore[arg-type]  # partial output
    got = svc._merged("k2", partial, old) or partial
    assert got.partitions == (part, part)                                  # the poll's, as in the TUI


# ---------------------------------------------------------------- another destination's rows (K2)
def _moved(config: FleetConfig, name: str = "k2") -> FleetConfig:
    return FleetConfig(hosts=tuple(dataclasses.replace(h, ssh=f"{name}-other.example") if h.name == name else h
                                   for h in config.hosts), general=config.general)


async def test_refresh_never_merges_the_rows_of_another_destination(tmp_path: Path) -> None:
    """K2: keep_cached loaded the cached row of a display name another config's machine wrote, then merged its
    jobs (and partitions) into this machine's."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    before = cache.load_snapshots()["k2"]
    svc = FleetService(_moved(config), lambda h: SqueueDown(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30, keep_cached=True)
    snap = svc.store.hosts["k2"].snapshot
    await svc.stop()
    row = cache.load_snapshots()["k2"]
    assert snap is not None and row.jobs.keys() == snap.jobs.keys()
    assert before.jobs.keys() - row.jobs.keys()                       # the other machine's jobs: not ours
    await cache.close()


@pytest.mark.parametrize("keep_cached", [False, True])
async def test_a_reused_display_name_never_skips_detection_with_foreign_caps(tmp_path: Path,
                                                                            keep_cached: bool) -> None:
    """N4: the caps, jobmeta and ended tasks cached for another destination under this name are not used."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2", "vega"])
    detects: list[str] = []

    class Counting(DemoTransport):
        def _answer(self, sim, section, now, purpose):  # type: ignore[no-untyped-def]
            if section.name.startswith("detect"):
                detects.append(self.host)
            return super()._answer(sim, section, now, purpose)

    own = FleetService(config, lambda h: Counting(h, world), clock, cache)
    own.lock_path = False
    await own._load_cache(snapshots=False, strict=True)
    assert {"k2", "vega"} <= own._cached_caps.keys() and "k2" in own._jobmeta
    svc = FleetService(_moved(config), lambda h: Counting(h, world), clock, cache)
    svc.lock_path = False
    stored = await svc._load_cache(snapshots=["k2", "vega"] if keep_cached else False, strict=True)
    assert "k2" not in stored and "vega" in stored
    assert "k2" not in svc._cached_caps and "vega" in svc._cached_caps
    assert "k2" not in svc._jobmeta and "k2" not in svc._ended
    assert svc.store.hosts["k2"].snapshot is None
    await svc.collect_once(["k2", "vega"], timeout=30, keep_cached=keep_cached)
    await svc.stop()
    assert "k2" in detects and "vega" not in detects                  # k2 detected afresh; vega's caps are its own
    await cache.close()


async def test_unreadable_health_loads_no_rows_for_collect_once_but_the_tui_starts_with_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])

    def broken(self: Cache) -> Any:
        raise sqlite3.DatabaseError("database disk image is malformed")
    monkeypatch.setattr(Cache, "load_health", broken)
    strict = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    assert await strict._load_cache(snapshots=["k2"], strict=True) == {}
    assert strict.store.hosts["k2"].snapshot is None and "k2" not in strict._cached_caps
    tui = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    assert "k2" in await tui._load_cache()                            # identity unknown: as before
    assert tui.store.hosts["k2"].snapshot is not None
    await cache.close()


async def test_two_refreshes_after_a_destination_change_never_trust_the_old_metadata(tmp_path: Path) -> None:
    """K2 (Codex): the first refresh of the moved host wrote its own health; the old destination's jobmeta and
    ended tasks, still in the cache, then passed for its own on the second."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2"])
    assert cache.load_jobmeta("k2")
    moved = _moved(config)
    for _ in range(2):
        svc = FleetService(moved, lambda h: SqueueDown(h, world), clock, cache)
        svc.lock_path = False
        await svc.collect_once(["k2"], timeout=30, keep_cached=True)
        await svc.stop()
        clock.advance(120)
    svc = FleetService(moved, lambda h: SqueueDown(h, world), clock, cache)
    await svc._load_cache(snapshots=False, strict=True)
    assert not svc._jobmeta.get("k2")                                 # the old machine's log paths: gone
    await cache.close()


async def test_strict_load_treats_a_damaged_health_row_as_unknown_identity(tmp_path: Path) -> None:
    """K2 (Codex): load_health drops a damaged row silently, so its host looked identity-less (= ours)."""
    clock, world, config, cache = await _cached_fleet(tmp_path, ["k2", "vega"])
    with cache._db:
        cache._db.execute("UPDATE host_health SET json = '{\"v\": 1, \"state\": ' WHERE host = 'k2'")
    strict = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    stored = await strict._load_cache(snapshots=["k2", "vega"], strict=True)
    assert "k2" not in stored and "k2" not in strict._cached_caps and "vega" in stored
    tui = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    assert "k2" in await tui._load_cache()                            # the TUI: as before
    await cache.close()
