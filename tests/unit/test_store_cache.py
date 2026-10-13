"""[WP1] FleetStore diffs/staleness (§3.7) and the SQLite cache (§5.5)."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runfleet.backends.base import CollectorOutput
from runfleet.config import HostConfig
from runfleet.core.cache import Cache, JobMeta, enrich_from_meta
from runfleet.core.store import ChangeSet, FleetStore, HostState, job_material_eq
from runfleet.models import (DETECT_REV, Capabilities, ConnState, HostMetrics, HostSnapshot, HostStatus, Job, JobKey,
                             JobKind, JobState, LogRef, NodeAlloc, ResourceAlloc, StateGroup)

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)
CAPS = Capabilities(remote_user="me", uid=1, login_host="l", tz_offset=timedelta(0), clock_skew=timedelta(0),
                    tools=frozenset(), profile=("slurm", "procfs"), detect_rev=DETECT_REV)


def job(jid: str, state: JobState = JobState.RUNNING, host: str = "k2", **kw: object) -> Job:
    return Job(key=JobKey(host, jid), kind=JobKind.SLURM, name=f"j{jid}", user="me", state=state, raw_state="",
               job_id_raw=jid, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- store
def test_material_eq_ignores_elapsed_and_observed_at() -> None:
    a = job("1", elapsed=timedelta(seconds=5), observed_at=T0)
    b = replace(a, elapsed=timedelta(seconds=20), observed_at=T0 + timedelta(seconds=15))
    assert job_material_eq(a, b)
    assert not job_material_eq(a, replace(a, state=JobState.COMPLETING))
    # leftover #6 (measured: every running squeue row differed on every poll via raw TimeUsed/TimeLeft)
    r1 = replace(a, raw={"TimeUsed": "1:00", "TimeLeft": "2:59:00", "Partition": "p"})
    r2 = replace(b, raw={"TimeUsed": "1:15", "TimeLeft": "2:58:45", "Partition": "p"})
    assert job_material_eq(r1, r2)
    assert not job_material_eq(r1, replace(r2, raw={**r2.raw, "Partition": "q"}))
    assert not job_material_eq(r1, replace(r2, raw={"TimeUsed": "1:15"}))


def test_apply_diffs_and_notifies() -> None:
    store = FleetStore([HostConfig("k2", "k2"), HostConfig("lynx", "lynx")])
    seen: list[ChangeSet] = []
    unsub = store.subscribe(seen.append)
    store.set_caps("k2", CAPS, T0)
    c1 = store.apply("k2", CollectorOutput(jobs={"1": job("1"), "2": job("2", JobState.PENDING)}), T0)
    assert c1.jobs_added == {"1", "2"} and not c1.jobs_changed
    c2 = store.apply("k2", CollectorOutput(jobs={"1": replace(job("1"), elapsed=timedelta(1)),
                                                 "3": job("3", JobState.FAILED)}), T0 + timedelta(seconds=15))
    assert c2.jobs_added == {"3"} and c2.jobs_removed == {"2"} and not c2.jobs_changed
    c3 = store.apply("k2", CollectorOutput(jobs={"1": job("1", JobState.COMPLETED), "3": job("3", JobState.FAILED)}),
                     T0)
    assert c3.jobs_changed == {"1"}
    quiet = store.apply("k2", CollectorOutput(), T0)                   # None fields keep the last value
    assert quiet == ChangeSet("k2") and len(store.hosts["k2"].snapshot.jobs) == 2   # type: ignore[union-attr]
    assert store.job(JobKey("k2", "3")) is not None and store.job(JobKey("lynx", "3")) is None
    assert [j.key.id for j in store.jobs(["k2"])] == ["1", "3"] and list(store.jobs(["nope"])) == []
    counts = store.counts()
    assert counts[StateGroup.DONE] == 1 and counts[StateGroup.FAILED] == 1 and counts[StateGroup.QUEUED] == 0
    m = HostMetrics(ncpu=1, load=None, cpu_pct=None, mem_total=None, mem_available=None, swap_total=None,
                    swap_free=None, uptime_s=None, sampled_at=T0)
    c4 = store.apply("k2", CollectorOutput(metrics=m, gpus=(), caps_patch={"gpu_tres": True, "bogus": 1}), T0)
    assert c4.metrics and not c4.gpus                                 # () == () -> unchanged
    assert store.hosts["k2"].snapshot.caps.gpu_tres is True           # type: ignore[union-attr]
    assert seen[0].status and len(seen) == 5
    unsub()
    store.apply("k2", CollectorOutput(jobs={}), T0)
    assert len(seen) == 5


def test_subscriber_exception_does_not_break_writes() -> None:
    store = FleetStore([HostConfig("k2", "k2")])

    def boom(_: ChangeSet) -> None:
        raise RuntimeError("ui bug")

    store.subscribe(boom)
    store.set_status("k2", HostStatus("k2", ConnState.OK))
    assert store.hosts["k2"].status.state is ConnState.OK


def test_status_and_idle_pause_changesets() -> None:
    store = FleetStore([HostConfig("k2", "k2"), HostConfig("off", "off", enabled=False)])
    assert store.hosts["off"].status.state is ConnState.DISABLED
    st = HostStatus("k2", ConnState.OK)
    assert store.set_status("k2", st).status
    assert not store.set_status("k2", st).status                       # unchanged -> no notification
    assert store.set_idle_paused(True) == ChangeSet("*", paused=True) and store.idle_paused
    assert store.set_idle_paused(True) == ChangeSet("*")


def test_stale_uses_effective_tick() -> None:
    snap = HostSnapshot("k2", T0, CAPS, {}, (), (), (), (), None)
    hs = HostState(HostConfig("k2", "k2"), HostStatus("k2", ConnState.OK, last_ok=T0, tick_interval=45.0), snap)
    assert not hs.stale(T0 + timedelta(seconds=94))                    # hidden ×3: 2 × 45 + 5 = 95 s
    assert hs.stale(T0 + timedelta(seconds=96))
    hs.status = replace(hs.status, state=ConnState.RETRYING)
    assert hs.stale(T0)
    hs.status = replace(hs.status, state=ConnState.DEGRADED)
    assert not hs.stale(T0)
    hs.snapshot = None
    assert hs.stale(T0)


# ---------------------------------------------------------------- cache
def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "cache.sqlite"
    c = Cache(path)
    snap = HostSnapshot("k2", T0, CAPS, {"1": job("1")}, (), (), (), (), None)
    c.put_snapshot("k2", snap)
    c.put_caps("k2", CAPS, T0)
    meta = JobMeta.from_job(job("1", logs=(LogRef("stdout", "/w/1.out"),), workdir="/w",
                                alloc=ResourceAlloc(1, 1, None, 1, "a30", {},
                                                    per_node=(NodeAlloc("n1", None, None, (0,), None),))), T0.timestamp())
    c.put_jobmeta("k2", [meta])
    ended = Job(key=JobKey("lynx", "p1-2"), kind=JobKind.TASK, name="t", user="me", state=JobState.EXITED,
                raw_state="exited", job_id_raw="1", end_time=T0)
    c.put_ended_task("lynx", ended, T0)
    c.put_ended_task("lynx", job("9"), T0)                             # Slurm jobs are not ended *tasks*
    c.flush_sync()
    c2 = Cache(path)
    assert c2.load_snapshots() == {"k2": snap}
    assert c2.load_caps("k2", T0 + timedelta(hours=1)) == (CAPS, T0)
    assert c2.load_caps("k2", T0 + timedelta(hours=25)) is None        # 24 h
    c2.put_caps("vega", replace(CAPS, detect_rev=0), T0)   # detected by an older version: detect again
    c2.flush_sync()
    assert c2.load_caps("vega", T0 + timedelta(minutes=1)) is None
    assert c2.load_jobmeta("k2") == {"1": meta}
    assert c2.load_ended_tasks("lynx", T0 - timedelta(hours=1)) == {"p1-2": ended}
    c2.prune(T0 + timedelta(days=8))
    assert c2.load_ended_tasks("lynx", T0 - timedelta(days=30)) == {}
    assert c2.load_jobmeta("k2") == {"1": meta}                        # jobmeta keeps 30 days
    c2.prune(T0 + timedelta(days=31))
    assert c2.load_jobmeta("k2") == {}


async def test_cache_async_flush_and_close(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c.sqlite")
    assert not c.has_pending()
    c.put_caps("k2", CAPS, T0)
    assert c.has_pending()
    await c.flush()
    assert not c.has_pending()
    await c.close()
    assert Cache(tmp_path / "c.sqlite").load_caps("k2", T0) is not None


def test_failed_flush_keeps_the_batch_for_retry(tmp_path: Path) -> None:
    """A sqlite error (disk full, read-only fs) must not drop ended tasks / job log paths queued only on change."""
    path = tmp_path / "c.sqlite"
    c = Cache(path)
    ended = Job(key=JobKey("lynx", "p1-2"), kind=JobKind.TASK, name="t", user="me", state=JobState.EXITED,
                raw_state="exited", job_id_raw="1", end_time=T0)
    c.put_ended_task("lynx", ended, T0)
    real, calls = c._write, []

    def broken(*a: object) -> None:
        calls.append(1)
        raise sqlite3.OperationalError("database or disk is full")
    c._write = broken  # type: ignore[method-assign]
    c.flush_sync()
    assert calls and c.has_pending()
    newer = replace(ended, name="t2")
    c.put_ended_task("lynx", newer, T0)                                # a newer entry wins over the kept one
    c._write = real  # type: ignore[method-assign]
    c.flush_sync()
    assert not c.has_pending()
    assert Cache(path).load_ended_tasks("lynx", T0 - timedelta(hours=1)) == {"p1-2": newer}


def test_state_files_are_private_whatever_the_umask(tmp_path: Path) -> None:
    import os
    import stat

    old = os.umask(0o002)
    try:
        path = tmp_path / "state" / "cache.sqlite"
        c = Cache(path)
        c.put_caps("k2", CAPS, T0)
        c.flush_sync()
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_cache_is_moved_aside(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    path.write_bytes(b"this is not a database" * 100)
    c = Cache(path)
    assert c.warnings and "moved to cache.sqlite.corrupt-" in c.warnings[0]
    assert list(tmp_path.glob("cache.sqlite.corrupt-*"))
    c.put_caps("k2", CAPS, T0)
    c.flush_sync()
    assert Cache(path).load_caps("k2", T0) is not None


def test_unreadable_rows_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    Cache(path)
    db = sqlite3.connect(path)
    db.execute("INSERT INTO snapshots VALUES ('k2', 0, '{\"host\": 1}')")
    db.commit()
    db.close()
    assert Cache(path).load_snapshots() == {}


def test_memory_cache() -> None:
    c = Cache(None)
    c.put_caps("k2", CAPS, T0)
    c.flush_sync()
    assert c.load_caps("k2", T0) == (CAPS, T0)


def test_enrich_history_job_from_jobmeta() -> None:
    live = job("7", logs=(LogRef("stdout", "/w/7.out", merged=True),), workdir="/w",
               alloc=ResourceAlloc(1, 1, None, 2, "nvidia_h200_nvl", {}))
    meta = JobMeta.from_job(live, 10.0)
    hist = job("7", JobState.COMPLETED, source="sacct", alloc=ResourceAlloc(1, 1, None, 2, None, {}),
               logs=(LogRef("stdout", "/w/slurm-7.out", source="template"),))
    got = enrich_from_meta(hist, meta)
    assert got.logs == (LogRef("stdout", "/w/7.out", merged=True, source="cache"),)
    assert got.workdir == "/w" and got.alloc.gpu_type == "nvidia_h200_nvl" and got.alloc.gpu_type_source == "cache"
    real = replace(hist, logs=(LogRef("stdout", "/x/7.out", source="sacct"),))
    assert enrich_from_meta(real, meta).logs == real.logs              # sacct StdOut wins over jobmeta
    later = JobMeta.from_job(job("7"), 20.0, meta)                      # a later sighting keeps what we knew
    assert later.stdout == "/w/7.out" and later.first_seen == 10.0 and later.last_seen == 20.0


def test_damaged_duration_row_is_skipped_not_the_whole_cache(tmp_path: Path) -> None:
    import json
    path = tmp_path / "c.sqlite"
    c = Cache(path)
    good = HostSnapshot("lynx", T0, CAPS, {"1": job("1", host="lynx")}, (), (), (), (), None)
    bad = HostSnapshot("k2", T0, CAPS, {"1": job("1", elapsed=timedelta(seconds=5))}, (), (), (), (), None)
    c.put_snapshot("lynx", good)
    c.put_snapshot("k2", bad)
    c.flush_sync()
    db = sqlite3.connect(path)
    text = db.execute("SELECT json FROM snapshots WHERE host = 'k2'").fetchone()[0]
    db.execute("UPDATE snapshots SET json = ? WHERE host = 'k2'", (text.replace('"elapsed": 5.0', '"elapsed": 1e30'),))
    db.commit()
    db.close()
    assert '"elapsed": 5.0' in text and json.loads(text)
    assert Cache(path).load_snapshots() == {"lynx": good}


def test_locked_database_is_not_moved_aside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cache.sqlite"
    Cache(path).flush_sync()
    calls = []

    def locked(self: Cache) -> sqlite3.Connection:
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(Cache, "_connect", locked)
    monkeypatch.setattr("runfleet.core.cache.time.sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        Cache(path)
    assert len(calls) == 3 and not list(tmp_path.glob("cache.sqlite.corrupt-*")) and path.exists()


def test_slow_flush_does_not_block_queued_writes(tmp_path: Path) -> None:
    import threading
    import time
    c = Cache(tmp_path / "c.sqlite")
    started, release = threading.Event(), threading.Event()
    real = c._write

    def slow(*a: object) -> None:
        started.set()
        release.wait(5)
        real(*a)  # type: ignore[arg-type]
    c._write = slow  # type: ignore[method-assign]
    c.put_caps("k2", CAPS, T0)
    th = threading.Thread(target=c.flush_sync)
    th.start()
    assert started.wait(5)
    t0 = time.monotonic()
    c.put_snapshot("k2", HostSnapshot("k2", T0, CAPS, {}, (), (), (), (), None))   # the event loop's call
    assert c.has_pending()
    assert time.monotonic() - t0 < 0.5
    release.set()
    th.join(5)
    c._write = real  # type: ignore[method-assign]
    c.flush_sync()
    assert not c.has_pending() and Cache(tmp_path / "c.sqlite").load_caps("k2", T0) is not None
