"""[WP4] cache.host_health (§5.5): one row per host with per-section freshness and the destination's identity, never
an error text; Cache.flush reports whether the batch committed; the poller rewrites the row with every snapshot."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from runfleet.agent.data import dest_hash
from runfleet.core.cache import ENDED_TASKS_RETENTION, Cache, health_from_json, health_json, merge_health
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.models import ConnState, ErrorKind, HostError, HostStatus, SectionStatus

T0 = datetime(2026, 10, 18, 14, 2, 31, tzinfo=UTC)
SECRET = "ssh: connect to host jump.lab.example port 22: Connection refused"


def _status() -> HostStatus:
    err = HostError(ErrorKind.TIMEOUT, SECRET, detail=SECRET, section="slurm.sacct", at=T0)
    return HostStatus(
        name="k2", state=ConnState.DEGRADED, error=err, last_ok=T0, latency_s=0.42, consecutive_failures=0,
        tick_interval=15.0, mux="managed", ssh_log="/home/alice/.local/state/runfleet/ssh-k2.log",
        controller_slowdown=2.0,
        sections={
            "slurm.squeue": SectionStatus("slurm.squeue", "ok", True, 0, 120, T0),
            "slurm.sacct": SectionStatus("slurm.sacct", "timeout", False, 124, None, T0 - timedelta(minutes=25),
                                         error=err),
            "slurm.sinfo": SectionStatus("slurm.sinfo", "failed", False, 1, 30, None),
        })


def test_health_document_holds_kinds_never_texts() -> None:
    doc = health_json(_status(), dest_hash=dest_hash("k2"), remote_user="alice")
    assert doc["v"] == 1 and doc["state"] == "degraded" and doc["dest_hash"] == dest_hash("k2")
    assert doc["last_ok"] == "2026-10-18T14:02:31+00:00" and doc["mux"] == "managed"
    assert doc["sections"]["slurm.sacct"] == {"ok": False, "outcome": "timeout",
                                              "last_ok": "2026-10-18T13:37:31+00:00", "error_kind": "timeout"}
    text = json.dumps(doc)
    assert "jump" not in text and "ssh-k2.log" not in text and "refused" not in text     # privacy
    assert set(doc) == {"v", "state", "dest_hash", "remote_user", "mux", "last_ok", "latency_s",
                        "consecutive_failures", "controller_slowdown", "tick_interval", "sections"}


def test_health_round_trip_and_prune(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.put_health("vega", HostStatus(name="vega", state=ConnState.OK), dest_hash=dest_hash("vega"),
                 remote_user=None, at=T0 - timedelta(seconds=ENDED_TASKS_RETENTION + 60))
    assert c.has_pending() and c.flush_sync() is True and not c.has_pending()
    loaded = Cache(path).load_health()
    k2 = loaded["k2"]
    assert k2.at == T0 and k2.state is ConnState.DEGRADED and k2.remote_user == "alice"
    assert k2.dest_hash == dest_hash("k2") and k2.latency_s == 0.42 and k2.controller_slowdown == 2.0
    assert k2.tick_interval == 15.0 and k2.last_ok == T0 and k2.mux == "managed"
    assert k2.data_at("jobs") == T0 and k2.data_at("history") == T0 - timedelta(minutes=25)
    assert k2.data_at("nodes") is None                                 # sinfo never succeeded
    assert [s.name for s in k2.failing()] == ["slurm.sacct", "slurm.sinfo"]
    assert k2.sections["slurm.sacct"].error_kind == "timeout"
    assert loaded["vega"].sections == {} and loaded["vega"].remote_user is None
    c.prune(T0)
    assert set(Cache(path).load_health()) == {"k2"}                    # older than 7 days: gone


def test_unreadable_or_newer_rows_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    Cache(path).flush_sync()
    db = sqlite3.connect(path)
    with db:
        db.executemany("INSERT INTO host_health VALUES (?, ?, ?)", [
            ("k2", T0.timestamp(), "not json"), ("vega", T0.timestamp(), json.dumps({"v": 2, "state": "ok"})),
            ("lyra", T0.timestamp(), json.dumps({"v": 1, "state": "bogus"})),
            ("lynx", T0.timestamp(), json.dumps({"v": 1, "state": "needs_auth"}))])
    db.close()
    loaded = Cache(path).load_health()
    assert set(loaded) == {"lynx"} and loaded["lynx"].state is ConnState.NEEDS_AUTH
    assert loaded["lynx"].consecutive_failures == 0 and loaded["lynx"].controller_slowdown == 1.0
    assert health_from_json([], T0) is None


async def test_flush_reports_failure_and_keeps_the_batch(tmp_path: Path) -> None:
    c = Cache(tmp_path / "cache.sqlite")
    assert c.flush_sync() is True and await c.flush() is True          # nothing to write
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    real = c._write

    def broken(*a: object) -> None:
        raise sqlite3.OperationalError("database or disk is full")
    c._write = broken  # type: ignore[method-assign]
    assert c.flush_sync() is False and c.has_pending()
    assert await c.flush() is False and c.has_pending()
    c._write = real  # type: ignore[method-assign]
    assert await c.flush() is True and not c.has_pending()
    assert "k2" in c.load_health()


def test_older_versions_positional_inserts_still_work(tmp_path: Path) -> None:
    """The new table never changes the existing ones: older RunFleet versions insert positional VALUES."""
    path = tmp_path / "cache.sqlite"
    Cache(path).flush_sync()
    db = sqlite3.connect(path)
    with db:
        db.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)", ("k2", 1.0, "{}"))
        db.execute("INSERT OR REPLACE INTO host_status VALUES (?, ?, ?, ?, ?)", ("k2", 1.0, "ok", None, None))
    db.close()


# ---------------------------------------------------------------- the poller writes it
@pytest.fixture
def demo_service(tmp_path: Path) -> tuple[FleetService, Cache]:
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    cache = Cache(tmp_path / "cache.sqlite")
    svc = FleetService(world.fleet_config(["k2", "labstation"]), lambda h: DemoTransport(h, world), clock, cache)
    svc.lock_path = False
    return svc, cache


async def test_poller_writes_host_health_with_every_snapshot(demo_service: tuple[FleetService, Cache]) -> None:
    svc, cache = demo_service
    await svc.collect_once(["k2", "labstation"], timeout=30)
    health = cache.load_health()
    assert set(health) == {"k2", "labstation"}
    for name in ("k2", "labstation"):
        st = svc.store.hosts[name].status
        h = health[name]
        assert h.dest_hash == dest_hash(svc.store.hosts[name].config.ssh)
        assert h.state is st.state and h.at <= T0 + timedelta(seconds=60)
        assert set(h.sections) == set(st.sections) and h.sections         # the sections _update_sections set
        assert all(h.sections[n].ok == s.ok for n, s in st.sections.items())
        caps = svc.store.hosts[name].snapshot.caps                        # type: ignore[union-attr]
        assert caps is not None and h.remote_user == caps.remote_user
    assert health["k2"].data_at("jobs") is not None and "slurm.squeue" in health["k2"].sections
    assert "process.ps" in health["labstation"].sections


def test_a_damaged_timestamp_skips_only_that_row(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.flush_sync()
    db = sqlite3.connect(path)
    with db:
        db.execute("INSERT INTO host_health VALUES (?, ?, ?)", ("vega", None, json.dumps({"v": 1, "state": "ok"})))
    db.close()
    assert set(Cache(path).load_health()) == {"k2"}


# ---------------------------------------------------------------- a failure never erases the last success (D1)
def _failing(at: datetime) -> HostStatus:
    """A fresh process (agent --refresh, a TUI just started) whose squeue fails: no last_ok known in memory."""
    err = HostError(ErrorKind.REMOTE, "slurmctld unreachable", section="slurm.squeue", at=at)
    return HostStatus(name="k2", state=ConnState.DEGRADED, error=err, last_ok=None, sections={
        "slurm.squeue": SectionStatus("slurm.squeue", "failed", False, 1, 40, None, error=err),
        "slurm.sinfo": SectionStatus("slurm.sinfo", "ok", True, 0, 30, at),
    })


def test_write_keeps_the_stored_last_ok_where_the_new_row_has_none(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    assert c.flush_sync()
    later = T0 + timedelta(minutes=5)
    c.put_health("k2", _failing(later), dest_hash=dest_hash("k2"), remote_user="alice", at=later)
    assert c.flush_sync()
    k2 = Cache(path).load_health()["k2"]
    assert k2.at == later and k2.state is ConnState.DEGRADED
    assert k2.sections["slurm.squeue"].ok is False and k2.sections["slurm.squeue"].last_ok == T0     # kept
    assert k2.sections["slurm.squeue"].error_kind == "remote"
    assert k2.sections["slurm.sinfo"].last_ok == later                 # a newer success replaces
    assert k2.last_ok == T0 and k2.data_at("jobs") == T0
    assert set(k2.sections) == {"slurm.squeue", "slurm.sinfo"}         # sections not run now are not revived


def test_another_destination_or_version_inherits_nothing(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    assert c.flush_sync()
    later = T0 + timedelta(minutes=5)
    c.put_health("k2", _failing(later), dest_hash=dest_hash("k2-other.example"), remote_user="bob", at=later)
    assert c.flush_sync()
    k2 = Cache(path).load_health()["k2"]
    assert k2.sections["slurm.squeue"].last_ok is None and k2.last_ok is None
    assert merge_health({"v": 1, "last_ok": None, "sections": {}}, {"v": 2, "last_ok": "x"})["last_ok"] is None
    assert merge_health({"v": 1, "dest_hash": "a", "last_ok": None},
                        "not a mapping")["last_ok"] is None
    db = sqlite3.connect(path)
    with db:
        db.execute("UPDATE host_health SET json = 'garbage' WHERE host = 'k2'")
    db.close()
    c.put_health("k2", _failing(later), dest_hash=dest_hash("k2"), remote_user="alice", at=later)
    assert c.flush_sync()                                              # a damaged stored row: written as is
    assert Cache(path).load_health()["k2"].last_ok is None


def test_close_sync_flushes_then_closes(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    assert c.close_sync() is True
    with pytest.raises(sqlite3.ProgrammingError):
        c._db.execute("SELECT 1")
    assert "k2" in Cache(path).load_health()


async def test_flush_waits_for_a_flush_in_flight_on_another_thread(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """C6: nothing is pending (the batch was swapped out by a flush still writing on a worker thread, e.g. the
    one a cancelled flush loop left running): flush() is a barrier all the same, never "done" mid-write."""
    import threading
    import time

    c = Cache(tmp_path / "cache.sqlite")
    writing, done = threading.Event(), threading.Event()
    real = Cache._write

    def slow_write(self: Cache, *a: Any) -> None:
        writing.set()
        time.sleep(0.3)
        real(self, *a)
        done.set()
    monkeypatch.setattr(Cache, "_write", slow_write)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    inflight = threading.Thread(target=c.flush_sync)
    inflight.start()
    assert writing.wait(5) and not c.has_pending()
    assert await c.flush() is True
    assert done.is_set()                                               # returned only once the write committed
    inflight.join(5)
    assert await c.flush() is True                                     # idle: no thread hop needed, still True
    c.close_sync()


def _meta(job_id: str) -> Any:
    from runfleet.core.cache import JobMeta
    return JobMeta(job_id, job_id, "orbit-train", "/home/alice/orbit", "/home/alice/orbit/out.log", None, True,
                   None, (), T0.timestamp(), T0.timestamp(), "RUNNING")


def test_health_of_another_destination_drops_the_old_ones_metadata(tmp_path: Path) -> None:
    """K2 (Codex): a reused display name skipped the foreign caps / jobmeta / ended tasks once, but they stayed in
    the cache and passed for the new destination's as soon as its health was stored. Writing a health row of
    another destination drops them, before the batch's own rows (which stay)."""
    path = tmp_path / "cache.sqlite"
    c = Cache(path)

    def rows(table: str, host: str) -> int:
        db = sqlite3.connect(path)
        try:
            return int(db.execute(f"SELECT COUNT(*) FROM {table} WHERE host = ?", (host,)).fetchone()[0])
        finally:
            db.close()

    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.put_jobmeta("k2", [_meta("3301")])
    c.put_jobmeta("vega", [_meta("4401")])
    assert c.flush_sync()
    with c._db:                                    # caps and an ended task of the first destination
        c._db.execute("INSERT INTO caps VALUES ('k2', ?, '{}')", (T0.timestamp(),))
        c._db.execute("INSERT INTO ended_tasks VALUES ('k2', 'p77', ?, '{}')", (T0.timestamp(),))
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)      # same: kept
    assert c.flush_sync() and set(c.load_jobmeta("k2")) == {"3301"}
    assert rows("caps", "k2") == 1 and rows("ended_tasks", "k2") == 1
    later = T0 + timedelta(minutes=5)
    c.put_health("k2", _failing(later), dest_hash=dest_hash("k2-other.example"), remote_user="bob", at=later)
    c.put_jobmeta("k2", [_meta("9901")])                       # the new destination's own, in the same batch
    assert c.flush_sync()
    assert set(c.load_jobmeta("k2")) == {"9901"} and set(c.load_jobmeta("vega")) == {"4401"}
    assert rows("caps", "k2") == 0 and rows("ended_tasks", "k2") == 0


def test_load_health_names_the_hosts_of_unreadable_rows(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.put_health("vega", _status(), dest_hash=dest_hash("vega"), remote_user="alice", at=T0)
    assert c.flush_sync()
    with c._db:
        c._db.execute("UPDATE host_health SET json = 'garbage' WHERE host = 'vega'")
    bad: set[str] = set()
    assert set(c.load_health(unreadable=bad)) == {"k2"} and bad == {"vega"}


def test_a_damaged_stored_health_row_counts_as_another_destination(tmp_path: Path) -> None:
    """Codex: a strict load rejected the rows of a host whose health row is damaged, but the next health write kept
    them, and the following load took them for the new destination's."""
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.put_jobmeta("k2", [_meta("3301")])
    assert c.flush_sync()
    with c._db:
        c._db.execute("UPDATE host_health SET json = '{\"v\": 1, \"dest' WHERE host = 'k2'")
    c.put_health("k2", _status(), dest_hash=dest_hash("k2-other.example"), remote_user="bob", at=T0)
    assert c.flush_sync()
    assert c.load_jobmeta("k2") == {}


@pytest.mark.parametrize(("column", "value", "forgotten"), [
    ("json", lambda d: {**d, "v": 99}, True),                 # another version: strict loads reject it
    ("json", lambda d: {**d, "state": "sleepy"}, True),       # a bad field
    ("json", lambda d: {**d, "sections": {"x": {}}}, True),   # a damaged section
    ("json", lambda d: [d], True),                            # not a document
    ("json", lambda d: {**d, "dest_hash": 12345}, True),      # an identity that is not a string
    ("json", lambda d: {**d, "dest_hash": [d["dest_hash"]]}, True),
    ("at", "yesterday", True),                                # a damaged time
    ("at", 1e300, True),                                      # a time out of range
    ("json", lambda d: {k: v for k, v in d.items() if k != "dest_hash"}, False),   # readable, no identity
    ("json", lambda d: d, False),                             # readable, the same destination
])
def test_forgetting_uses_the_strict_loading_health_check(tmp_path: Path, column: str, value: Any,
                                                         forgotten: bool) -> None:
    """F5: a stored health row strict loading cannot read (health_from_json: another version, a bad field, a
    damaged time) named the same dest_hash in valid JSON, so the next write kept the rows strict loading had
    rejected as of unknown identity. One check for both now."""
    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)
    c.put_jobmeta("k2", [_meta("3301")])
    assert c.flush_sync()
    with c._db:
        if column == "json":
            doc = json.loads(c._db.execute("SELECT json FROM host_health WHERE host = 'k2'").fetchone()[0])
            c._db.execute("UPDATE host_health SET json = ? WHERE host = 'k2'", (json.dumps(value(doc)),))
        else:
            c._db.execute("UPDATE host_health SET at = ? WHERE host = 'k2'", (value,))
    unreadable: set[str] = set()
    c.load_health(unreadable=unreadable)
    assert ("k2" in unreadable) is forgotten                           # what strict loading decides …
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=T0)   # same destination
    assert c.flush_sync()
    assert (c.load_jobmeta("k2") == {}) is forgotten                   # … decides forgetting too


def test_prune_keeps_the_identity_of_rows_still_cached(tmp_path: Path) -> None:
    """Codex: host_health went after 7 days while the snapshot (and 30 days of jobmeta) stayed: a name reused for
    another machine after a quiet week passed for the same one."""
    from runfleet.models import HostSnapshot

    path = tmp_path / "cache.sqlite"
    c = Cache(path)
    old = T0 - timedelta(seconds=ENDED_TASKS_RETENTION, days=1)
    c.put_health("k2", _status(), dest_hash=dest_hash("k2"), remote_user="alice", at=old)
    c.put_health("vega", _status(), dest_hash=dest_hash("vega"), remote_user="alice", at=old)
    c.put_snapshot("k2", HostSnapshot("k2", old, None, {}, (), (), (), (), None))
    assert c.flush_sync()
    c.prune(T0)
    assert set(c.load_health()) == {"k2"}                              # vega: nothing left it could identify
