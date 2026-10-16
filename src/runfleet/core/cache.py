"""[WP1] SQLite cache of capabilities, profiles and last snapshots (DESIGN.md §5.5).

Reads happen at startup; writes are queued in memory and flushed in one transaction (every 10 s by the service and at
exit) on a worker thread via `asyncio.to_thread`. A corrupt or unreadable database is renamed to
`cache.sqlite.corrupt-<ts>` and replaced by a fresh one (§9 row 27); `Cache(None)` is an in-memory cache.
`host_status` (each host's connection state) is written for `runfleet status`, which reads the file without this class
(read-only, `runfleet.status`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runfleet import paths
from runfleet.models import (DETECT_REV, Capabilities, HostSnapshot, HostStatus, Job, JobKind, LogRef, NodeAlloc,
                             from_json, to_json)

log = logging.getLogger("runfleet.core.cache")

CAPS_MAX_AGE = 24 * 3600.0
JOBMETA_RETENTION = 30 * 86400.0
ENDED_TASKS_RETENTION = 7 * 86400.0
OPEN_RETRIES = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots   (host TEXT PRIMARY KEY, taken_at REAL, json TEXT);
CREATE TABLE IF NOT EXISTS caps        (host TEXT PRIMARY KEY, detected_at REAL, json TEXT);
CREATE TABLE IF NOT EXISTS jobmeta     (host TEXT, job_id TEXT, job_id_raw TEXT, name TEXT, workdir TEXT, stdout TEXT,
                                        stderr TEXT, merged INTEGER, gpu_type TEXT, per_node_json TEXT,
                                        first_seen REAL, last_seen REAL, last_state TEXT,
                                        PRIMARY KEY (host, job_id));
CREATE TABLE IF NOT EXISTS ended_tasks (host TEXT, job_id TEXT, ended_at REAL, json TEXT, PRIMARY KEY (host, job_id));
CREATE TABLE IF NOT EXISTS host_status (host TEXT PRIMARY KEY, at REAL, state TEXT, error_kind TEXT, error TEXT);
"""


@dataclass(frozen=True, slots=True)
class JobMeta:
    """What a Slurm job looked like while it was live (scontrol forgets it MinJobAge after it ends, §5.2.4)."""
    job_id: str
    job_id_raw: str
    name: str
    workdir: str | None
    stdout: str | None
    stderr: str | None
    merged: bool
    gpu_type: str | None
    per_node: tuple[NodeAlloc, ...]
    first_seen: float
    last_seen: float
    last_state: str

    @classmethod
    def from_job(cls, job: Job, now: float, previous: JobMeta | None = None) -> JobMeta:
        # a guessed path (LogRef.source="template") is never recorded: it would come back as a "cache" (real) path
        out = next((r for r in job.logs if r.stream == "stdout" and r.path and r.source != "template"), None)
        err = next((r for r in job.logs if r.stream == "stderr" and r.path and r.source != "template"), None)
        merged = any(r.merged for r in job.logs)
        prev = previous
        return cls(
            job_id=job.key.id, job_id_raw=job.job_id_raw, name=job.name,
            workdir=job.workdir or (prev.workdir if prev else None),
            stdout=out.path if out else (prev.stdout if prev else None),
            stderr=err.path if err else (prev.stderr if prev else None),
            merged=merged or (prev.merged if prev else False),
            gpu_type=job.alloc.gpu_type or (prev.gpu_type if prev else None),
            per_node=job.alloc.per_node or (prev.per_node if prev else ()),
            first_seen=prev.first_seen if prev else now, last_seen=now, last_state=job.state.value,
        )

    def worth_keeping(self) -> bool:
        return bool(self.stdout or self.workdir or self.gpu_type or self.per_node)


def enrich_from_meta(job: Job, meta: JobMeta) -> Job:
    """Fill a history-only job's missing log paths / GPU type / per-node allocation from jobmeta (source="cache")."""
    updates: dict[str, Any] = {}
    has_real_log = any(r.path and r.source not in ("template",) for r in job.logs)
    if not has_real_log and meta.stdout:
        refs = [LogRef(stream="stdout", path=meta.stdout, merged=meta.merged, source="cache")]
        if meta.stderr and not meta.merged and meta.stderr != meta.stdout:
            refs.append(LogRef(stream="stderr", path=meta.stderr, source="cache"))
        updates["logs"] = tuple(refs)
    if job.workdir is None and meta.workdir:
        updates["workdir"] = meta.workdir
    alloc = job.alloc
    if alloc.gpu_type is None and meta.gpu_type and alloc.gpus > 0:
        alloc = replace(alloc, gpu_type=meta.gpu_type, gpu_type_source="cache")
    if not alloc.per_node and meta.per_node:
        alloc = replace(alloc, per_node=meta.per_node)
    if alloc is not job.alloc:
        updates["alloc"] = alloc
    return replace(job, **updates) if updates else job


class Cache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.warnings: list[str] = []            # surfaced once as a toast by the UI (FleetService.notices)
        self._lock = threading.Lock()            # the pending dicts only: the event loop takes it, so never held long
        self._db_lock = threading.Lock()         # the connection (reads, flushes, prune, close)
        self._pending_snapshots: dict[str, HostSnapshot] = {}
        self._pending_caps: dict[str, tuple[Capabilities, float]] = {}
        self._pending_meta: dict[tuple[str, str], JobMeta] = {}
        self._pending_ended: dict[tuple[str, str], tuple[Job, float]] = {}
        self._pending_status: dict[str, tuple[float, str, str | None, str | None]] = {}
        self._db = self._open()

    # ------------------------------------------------------------ connection
    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            db = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            paths.ensure_private_dir(self.path.parent)
            paths.private_file(self.path)              # 0600 before sqlite creates it (-wal/-shm copy its mode)
            db = sqlite3.connect(self.path, check_same_thread=False, timeout=2.0)
        try:
            if self.path is not None:
                db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("SELECT count(*) FROM snapshots").fetchone()    # forces a read: detects garbage files
        except BaseException:
            db.close()
            raise
        return db

    def _open(self) -> sqlite3.Connection:
        for attempt in range(OPEN_RETRIES):
            try:
                return self._connect()
            except sqlite3.OperationalError as e:
                # locked / busy (another instance converting to WAL), disk full, read-only, I/O error: not corruption,
                # so a live database is never moved aside; a lock is retried briefly, else no cache this run (cli)
                if attempt == OPEN_RETRIES - 1 or not re.search(r"locked|busy", str(e)):
                    raise
                time.sleep(0.2 * (attempt + 1))
            except sqlite3.DatabaseError as e:          # "file is not a database", "malformed"
                if self.path is None:
                    raise
                stamp = time.strftime("%Y%m%d-%H%M%S")
                moved = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
                for suffix in ("", "-wal", "-shm"):
                    p = Path(str(self.path) + suffix)
                    if p.exists():
                        p.rename(Path(str(moved) + suffix))
                msg = f"cache was unreadable ({e}); moved to {moved.name}, starting fresh"
                log.warning(msg)
                self.warnings.append(msg)
                return self._connect()
        raise AssertionError("unreachable")

    # ------------------------------------------------------------ reads (sync; call via to_thread from the loop)
    def load_snapshots(self) -> dict[str, HostSnapshot]:
        out: dict[str, HostSnapshot] = {}
        with self._db_lock:
            rows = self._db.execute("SELECT host, json FROM snapshots").fetchall()
        for host, text in rows:
            try:
                out[host] = from_json(HostSnapshot, json.loads(text))
            except (ValueError, TypeError, KeyError) as e:      # schema drift: ignore that row
                log.info("cache: dropping unreadable snapshot of %s (%s)", host, e)
        return out

    def load_caps(self, host: str, now: datetime, max_age: float = CAPS_MAX_AGE) -> tuple[Capabilities, datetime] | None:
        with self._db_lock:
            row = self._db.execute("SELECT detected_at, json FROM caps WHERE host = ?", (host,)).fetchone()
        if row is None or now.timestamp() - row[0] > max_age:
            return None
        try:
            caps = from_json(Capabilities, json.loads(row[1]))
        except (ValueError, TypeError, KeyError):
            return None
        if caps.detect_rev < DETECT_REV:          # detected by an older version: lacks newer fields, detect again
            return None
        return caps, datetime.fromtimestamp(row[0], tz=UTC)

    def load_jobmeta(self, host: str) -> dict[str, JobMeta]:
        with self._db_lock:
            rows = self._db.execute(
                "SELECT job_id, job_id_raw, name, workdir, stdout, stderr, merged, gpu_type, per_node_json, "
                "first_seen, last_seen, last_state FROM jobmeta WHERE host = ?", (host,)).fetchall()
        out: dict[str, JobMeta] = {}
        for r in rows:
            try:
                per_node = from_json(tuple[NodeAlloc, ...], json.loads(r[8] or "[]"))
            except (ValueError, TypeError, KeyError):
                per_node = ()
            out[r[0]] = JobMeta(job_id=r[0], job_id_raw=r[1], name=r[2], workdir=r[3], stdout=r[4], stderr=r[5],
                                merged=bool(r[6]), gpu_type=r[7], per_node=per_node, first_seen=r[9],
                                last_seen=r[10], last_state=r[11])
        return out

    def load_ended_tasks(self, host: str, since: datetime) -> dict[str, Job]:
        with self._db_lock:
            rows = self._db.execute("SELECT job_id, json FROM ended_tasks WHERE host = ? AND ended_at >= ?",
                                    (host, since.timestamp())).fetchall()
        out: dict[str, Job] = {}
        for job_id, text in rows:
            try:
                out[job_id] = from_json(Job, json.loads(text))
            except (ValueError, TypeError, KeyError):
                continue
        return out

    def prune(self, now: datetime) -> None:
        t = now.timestamp()
        with self._db_lock, self._db:
            self._db.execute("DELETE FROM jobmeta WHERE last_seen < ?", (t - JOBMETA_RETENTION,))
            self._db.execute("DELETE FROM ended_tasks WHERE ended_at < ?", (t - ENDED_TASKS_RETENTION,))
            self._db.execute("DELETE FROM caps WHERE detected_at < ?", (t - CAPS_MAX_AGE,))

    # ------------------------------------------------------------ queued writes
    def put_snapshot(self, host: str, snap: HostSnapshot) -> None:
        with self._lock:
            self._pending_snapshots[host] = snap

    def put_caps(self, host: str, caps: Capabilities, detected_at: datetime) -> None:
        with self._lock:
            self._pending_caps[host] = (caps, detected_at.timestamp())

    def put_jobmeta(self, host: str, metas: Iterable[JobMeta]) -> None:
        with self._lock:
            for m in metas:
                self._pending_meta[(host, m.job_id)] = m

    def put_ended_task(self, host: str, job: Job, ended_at: datetime) -> None:
        if job.kind is not JobKind.TASK:
            return
        with self._lock:
            self._pending_ended[(host, job.key.id)] = (job, ended_at.timestamp())

    def put_status(self, host: str, status: HostStatus, at: datetime) -> None:
        """The host's connection state since `at`, for `runfleet status` (§8.5); the TUI never reads it back."""
        err = status.error
        with self._lock:
            self._pending_status[host] = (at.timestamp(), status.state.value, err.kind.value if err else None,
                                          err.summary if err else None)

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending_snapshots or self._pending_caps or self._pending_meta or self._pending_ended
                        or self._pending_status)

    def flush_sync(self) -> None:
        # _db_lock first (flushes stay ordered: an older batch never overwrites a newer one), then _lock only for
        # the swap: a slow commit (busy timeout, slow home dir) must not block put_* on the event loop
        with self._db_lock:
            with self._lock:
                snaps, self._pending_snapshots = self._pending_snapshots, {}
                caps, self._pending_caps = self._pending_caps, {}
                metas, self._pending_meta = self._pending_meta, {}
                ended, self._pending_ended = self._pending_ended, {}
                status, self._pending_status = self._pending_status, {}
            if not (snaps or caps or metas or ended or status):
                return
            try:
                with self._db:
                    self._write(snaps, caps, metas, ended, status)
            except sqlite3.Error as e:
                # Keep the batch for the next flush (disk full, read-only fs, a locked db): ended workstation tasks
                # and job log paths are queued only on change, so dropping them would lose them for good. Newer
                # pending entries win; the size stays bounded (keys are per host / job).
                log.warning("cache write failed (kept for retry): %s", e)
                with self._lock:
                    self._pending_snapshots = {**snaps, **self._pending_snapshots}
                    self._pending_caps = {**caps, **self._pending_caps}
                    self._pending_meta = {**metas, **self._pending_meta}
                    self._pending_ended = {**ended, **self._pending_ended}
                    self._pending_status = {**status, **self._pending_status}

    def _write(self, snaps: Mapping[str, HostSnapshot], caps: Mapping[str, tuple[Capabilities, float]],
               metas: Mapping[tuple[str, str], JobMeta], ended: Mapping[tuple[str, str], tuple[Job, float]],
               status: Mapping[str, tuple[float, str, str | None, str | None]]) -> None:
        db = self._db
        db.executemany("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)",
                       [(h, s.taken_at.timestamp(), json.dumps(to_json(s))) for h, s in snaps.items()])
        db.executemany("INSERT OR REPLACE INTO caps VALUES (?, ?, ?)",
                       [(h, t, json.dumps(to_json(c))) for h, (c, t) in caps.items()])
        db.executemany("INSERT OR REPLACE INTO jobmeta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [(h, m.job_id, m.job_id_raw, m.name, m.workdir, m.stdout, m.stderr, int(m.merged),
                         m.gpu_type, json.dumps(to_json(m.per_node)), m.first_seen, m.last_seen, m.last_state)
                        for (h, _), m in metas.items()])
        db.executemany("INSERT OR REPLACE INTO ended_tasks VALUES (?, ?, ?, ?)",
                       [(h, j.key.id, t, json.dumps(to_json(j))) for (h, _), (j, t) in ended.items()])
        db.executemany("INSERT OR REPLACE INTO host_status VALUES (?, ?, ?, ?, ?)",
                       [(h, *row) for h, row in status.items()])

    async def flush(self) -> None:
        if self.has_pending():
            await asyncio.to_thread(self.flush_sync)

    async def close(self) -> None:
        await self.flush()
        with self._db_lock:
            self._db.close()
