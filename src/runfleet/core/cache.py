"""[WP1] SQLite cache of capabilities, profiles and last snapshots (DESIGN.md §5.5).

Reads happen at startup; writes are queued in memory and flushed in one transaction (every 10 s by the service and at
exit) on a worker thread via `asyncio.to_thread`. A corrupt or unreadable database is renamed to
`cache.sqlite.corrupt-<ts>` and replaced by a fresh one (§9 row 27); `Cache(None)` is an in-memory cache.
`host_status` (each host's connection state) is written for `runfleet status`, which reads the file without this class
(read-only, `runfleet.status`); `host_health` (per-section freshness, the ssh destination's hash) for `runfleet agent`.
`account_reports` (Account, §5.2.11) keeps the last on-demand `AccountReport` per (host, window).
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
from runfleet.agent.data import HostHealth, SectionHealth
from runfleet.models import (DETECT_REV, AccountReport, Capabilities, ConnState, HostSnapshot, HostStatus, Job,
                             JobKind, LogRef, NodeAlloc, from_json, to_json)

log = logging.getLogger("runfleet.core.cache")

CAPS_MAX_AGE = 24 * 3600.0
JOBMETA_RETENTION = 30 * 86400.0
ENDED_TASKS_RETENTION = 7 * 86400.0
ACCOUNT_REPORTS_RETENTION = 30 * 86400.0
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
CREATE TABLE IF NOT EXISTS host_health (host TEXT PRIMARY KEY, at REAL, json TEXT);
CREATE TABLE IF NOT EXISTS account_reports (host TEXT, window TEXT, at REAL, json TEXT, PRIMARY KEY (host, window));
"""
HEALTH_V = 1
# one account report row (§5.2.11): the newest `at` of a (host, window) wins, whoever writes it (Cache, `agent usage`)
UPSERT_REPORT = ("INSERT INTO account_reports VALUES (?, ?, ?, ?) ON CONFLICT (host, window) DO UPDATE SET "
                 "at = excluded.at, json = excluded.json WHERE excluded.at >= account_reports.at")


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else to_json(dt)


def _dt(value: Any) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def health_json(status: HostStatus, *, dest_hash: str, remote_user: str | None) -> dict[str, Any]:
    """The `host_health.json` document of one host (version 1). Error kinds only: summaries, details and the ssh
    log never go in (they can name a jump host or carry stderr)."""
    return {
        "v": HEALTH_V, "state": status.state.value, "dest_hash": dest_hash, "remote_user": remote_user,
        "mux": status.mux, "last_ok": _iso(status.last_ok), "latency_s": status.latency_s,
        "consecutive_failures": status.consecutive_failures, "controller_slowdown": status.controller_slowdown,
        "tick_interval": status.tick_interval,
        "sections": {name: {"ok": sec.ok, "outcome": sec.outcome, "last_ok": _iso(sec.last_ok),
                            "error_kind": sec.error.kind.value if sec.error is not None else None}
                     for name, sec in status.sections.items()},
    }


def merge_health(new: Mapping[str, Any], old: Any) -> dict[str, Any]:
    """`new` (a :func:`health_json` document) with every `last_ok` it lacks (None: no success in this process,
    e.g. a failing section right after start or in `agent --refresh`) taken from `old`, the stored document of
    the same destination: a failure never erases when the data was last good. Another version or destination:
    `new` as it is."""
    out = dict(new)
    if (not isinstance(old, Mapping) or old.get("v") != HEALTH_V or new.get("v") != HEALTH_V
            or old.get("dest_hash") != new.get("dest_hash")):
        return out
    if out.get("last_ok") is None and isinstance(old.get("last_ok"), str):
        out["last_ok"] = old["last_ok"]
    old_secs = old.get("sections")
    if isinstance(old_secs, Mapping) and isinstance(out.get("sections"), Mapping):
        secs: dict[str, Any] = {}
        for name, sec in out["sections"].items():
            prev = old_secs.get(name)
            if (isinstance(sec, Mapping) and sec.get("last_ok") is None and isinstance(prev, Mapping)
                    and isinstance(prev.get("last_ok"), str)):
                sec = {**sec, "last_ok": prev["last_ok"]}
            secs[name] = sec
        out["sections"] = secs
    return out


def health_from_json(data: Any, at: datetime) -> HostHealth | None:
    """Inverse of :func:`health_json` (`at`: the row's time); None for another version or a damaged document.
    Shared with readers that open the database read-only (`runfleet agent`)."""
    try:
        if not isinstance(data, Mapping) or data.get("v") != HEALTH_V:
            return None
        dest = data.get("dest_hash")
        if dest is not None and not isinstance(dest, str):    # a bad identity: unreadable, as strict loading says
            return None
        sections = {
            str(name): SectionHealth(name=str(name), ok=bool(sec["ok"]), outcome=str(sec["outcome"]),
                                     last_ok=_dt(sec.get("last_ok")), error_kind=sec.get("error_kind"))
            for name, sec in (data.get("sections") or {}).items()
        }
        latency, tick = data.get("latency_s"), data.get("tick_interval")
        return HostHealth(
            at=at, state=ConnState(data["state"]), dest_hash=dest,
            remote_user=data.get("remote_user"), mux=data.get("mux"), last_ok=_dt(data.get("last_ok")),
            latency_s=float(latency) if latency is not None else None,
            consecutive_failures=int(data.get("consecutive_failures") or 0),
            controller_slowdown=float(data.get("controller_slowdown") or 1.0),
            tick_interval=float(tick) if tick is not None else None, sections=sections)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as e:    # int(1e309): OverflowError
        log.info("cache: unreadable host health (%s)", e)
        return None


def _stored_health(at: Any, text: Any) -> HostHealth | None:
    """A host_health row (`at`, `json` columns) as strict loading reads it; None when damaged (JSON, time) or
    unreadable by :func:`health_from_json` (another version, a bad field)."""
    try:
        data, when = json.loads(text), datetime.fromtimestamp(at, tz=UTC)
    except (ValueError, TypeError, OverflowError, OSError, RecursionError):
        return None
    return health_from_json(data, when)


def upsert_account_report(path: Path, report: AccountReport, *, window: str | None = None,
                          timeout: float = 2.0) -> None:
    """`runfleet agent usage --refresh` (§5.5, §8.7): one row of `account_reports` written straight into an existing
    cache file, without this class (no create, migrate or repair); a TUI writing the same key next to it is
    harmless (newest `at` wins). `window`: the key when it is not the report's own ("period" answered as "30d").
    Raises sqlite3.Error (a missing file: OperationalError)."""
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, timeout=timeout)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS account_reports (host TEXT, window TEXT, at REAL, json TEXT, "
                       "PRIMARY KEY (host, window))")              # a cache of 0.8.0 and older lacks it
            db.execute(UPSERT_REPORT, (report.host, window or report.window, report.fetched_at.timestamp(),
                                       json.dumps(to_json(report))))
    finally:
        db.close()


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
        self._pending_health: dict[str, tuple[float, str]] = {}
        self._pending_reports: dict[tuple[str, str], AccountReport] = {}
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

    def load_health(self, *, unreadable: set[str] | None = None) -> dict[str, HostHealth]:
        """Every host's last recorded health (unreadable rows and other versions left out; their hosts are added to
        `unreadable` when given: a row exists, its identity is unknown)."""
        with self._db_lock:
            rows = self._db.execute("SELECT host, at, json FROM host_health").fetchall()
        out: dict[str, HostHealth] = {}
        for host, at, text in rows:
            health = _stored_health(at, text)
            if health is not None:
                out[host] = health
            elif unreadable is not None:
                unreadable.add(host)
        return out

    def load_account_reports(self, host: str) -> dict[str, AccountReport]:
        """The host's last on-demand account report per window (§5.5; unreadable rows left out), any age."""
        with self._db_lock:
            rows = self._db.execute("SELECT window, json FROM account_reports WHERE host = ?", (host,)).fetchall()
        out: dict[str, AccountReport] = {}
        for window, text in rows:
            try:
                out[window] = from_json(AccountReport, json.loads(text))
            except (ValueError, TypeError, KeyError) as e:      # schema drift: ignore that row
                log.info("cache: dropping unreadable account report of %s (%s)", host, e)
        with self._lock:                                         # queued, not yet written: newer
            out.update({w: r for (h, w), r in self._pending_reports.items() if h == host})
        return out

    def prune(self, now: datetime) -> None:
        t = now.timestamp()
        with self._db_lock, self._db:
            self._db.execute("DELETE FROM jobmeta WHERE last_seen < ?", (t - JOBMETA_RETENTION,))
            self._db.execute("DELETE FROM ended_tasks WHERE ended_at < ?", (t - ENDED_TASKS_RETENTION,))
            self._db.execute("DELETE FROM caps WHERE detected_at < ?", (t - CAPS_MAX_AGE,))
            self._db.execute("DELETE FROM account_reports WHERE at < ?", (t - ACCOUNT_REPORTS_RETENTION,))
            # a host's health is its rows' identity (which ssh destination wrote them): it goes only with the last
            # of them, never before (a name reused for another machine after a week would pass for the same one)
            self._db.execute("DELETE FROM host_health WHERE at < ? AND host NOT IN (SELECT host FROM snapshots "
                             "UNION SELECT host FROM caps UNION SELECT host FROM jobmeta "
                             "UNION SELECT host FROM ended_tasks UNION SELECT host FROM account_reports)",
                             (t - ENDED_TASKS_RETENTION,))

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

    def put_health(self, host: str, status: HostStatus, *, dest_hash: str, remote_user: str | None,
                   at: datetime) -> None:
        """The host's health as of `at` (sections, freshness, identity), rewritten with every snapshot by the poller
        for `runfleet agent` (§5.5). The write keeps the stored row's `last_ok`s where this one has none
        (:func:`merge_health`)."""
        text = json.dumps(health_json(status, dest_hash=dest_hash, remote_user=remote_user))
        with self._lock:
            self._pending_health[host] = (at.timestamp(), text)

    def put_account_report(self, report: AccountReport, window: str | None = None) -> None:
        """The last on-demand account report of (host, window) (§5.2.11), written with the next batch; `window`: the
        key when it is not the report's own ("period" answered as "30d")."""
        with self._lock:
            self._pending_reports[(report.host, window or report.window)] = report

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending_snapshots or self._pending_caps or self._pending_meta or self._pending_ended
                        or self._pending_status or self._pending_health or self._pending_reports)

    def flush_sync(self) -> bool:
        """Write the queued batch in one transaction. True when there was nothing to write or it committed; False
        when the write failed and the batch was kept for the next flush."""
        # _db_lock first (flushes stay ordered: an older batch never overwrites a newer one), then _lock only for
        # the swap: a slow commit (busy timeout, slow home dir) must not block put_* on the event loop
        with self._db_lock:
            with self._lock:
                snaps, self._pending_snapshots = self._pending_snapshots, {}
                caps, self._pending_caps = self._pending_caps, {}
                metas, self._pending_meta = self._pending_meta, {}
                ended, self._pending_ended = self._pending_ended, {}
                status, self._pending_status = self._pending_status, {}
                health, self._pending_health = self._pending_health, {}
                reports, self._pending_reports = self._pending_reports, {}
            if not (snaps or caps or metas or ended or status or health or reports):
                return True
            try:
                with self._db:
                    self._write(snaps, caps, metas, ended, status, health, reports)
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
                    self._pending_health = {**health, **self._pending_health}
                    self._pending_reports = {**reports, **self._pending_reports}
                return False
            return True

    def _write(self, snaps: Mapping[str, HostSnapshot], caps: Mapping[str, tuple[Capabilities, float]],
               metas: Mapping[tuple[str, str], JobMeta], ended: Mapping[tuple[str, str], tuple[Job, float]],
               status: Mapping[str, tuple[float, str, str | None, str | None]],
               health: Mapping[str, tuple[float, str]],
               reports: Mapping[tuple[str, str], AccountReport] | None = None) -> None:
        db = self._db
        self._forget_other_destinations(health)
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
        # newest `at` wins: `agent usage --refresh` may have upserted a newer row of the same key meanwhile
        for (h, w), r in (reports or {}).items():
            db.execute(UPSERT_REPORT, (h, w, r.fetched_at.timestamp(), json.dumps(to_json(r))))
        if health:
            db.executemany("INSERT OR REPLACE INTO host_health (host, at, json) VALUES (?, ?, ?)",
                           [(h, t, self._merged_health(h, text)) for h, (t, text) in health.items()])

    def _forget_other_destinations(self, health: Mapping[str, tuple[float, str]]) -> None:
        """Inside the write transaction, before this batch's rows: a host whose stored health names another ssh
        destination than the one written now (a second config reusing the display name took it over) loses that
        destination's capabilities, job metadata, ended tasks and account reports, which would otherwise pass for the new one's
        as soon as its health is stored (§8.7). A stored row strict loading cannot read (damaged JSON or time,
        another version, a bad field: the same :func:`_stored_health` check) counts as another destination
        (`strict` loads rejected its rows already); a readable one without `dest_hash` (no identity recorded)
        does not."""
        for host, (_, text) in health.items():
            row = self._db.execute("SELECT at, json FROM host_health WHERE host = ?", (host,)).fetchone()
            if row is None:
                continue
            try:
                new = json.loads(text)
            except (ValueError, TypeError, RecursionError):
                continue
            old = _stored_health(row[0], row[1])
            # unreadable: whose rows they are is unknown, so not the new one's
            was = "" if old is None else old.dest_hash
            now = new.get("dest_hash") if isinstance(new, Mapping) else None
            if isinstance(was, str) and isinstance(now, str) and was != now:
                self._db.execute("DELETE FROM caps WHERE host = ?", (host,))
                self._db.execute("DELETE FROM jobmeta WHERE host = ?", (host,))
                self._db.execute("DELETE FROM ended_tasks WHERE host = ?", (host,))
                self._db.execute("DELETE FROM account_reports WHERE host = ?", (host,))
                log.info("cache: %s now names another ssh destination; its old metadata dropped", host)

    def _merged_health(self, host: str, text: str) -> str:
        """Inside the write transaction (no other writer between this read and the replace)."""
        row = self._db.execute("SELECT json FROM host_health WHERE host = ?", (host,)).fetchone()
        if row is None:
            return text
        try:
            old, new = json.loads(row[0]), json.loads(text)
        except (ValueError, TypeError, RecursionError):
            return text
        return json.dumps(merge_health(new, old)) if isinstance(new, Mapping) else text

    async def flush(self) -> bool:
        """:meth:`flush_sync` on a worker thread; same result (True: nothing pending or committed). Also a barrier:
        a flush in flight on another thread (e.g. the one a cancelled flush loop left running) has finished when
        this returns, so the TUI's stop() never releases its instance locks while a batch is still being written.
        (The batch is swapped out of the pending maps under _db_lock: pending or _db_lock held, never neither.)"""
        if self.has_pending() or self._db_lock.locked():
            return await asyncio.to_thread(self.flush_sync)
        return True

    async def close(self) -> None:
        await self.flush()
        with self._db_lock:
            self._db.close()

    def close_sync(self) -> bool:
        """Flush, then close, on the calling thread (a flush in flight elsewhere finishes first: _db_lock); returns
        the flush's result. `agent --refresh` releases its instance locks only after this returned, so a lock is
        never free while its rows are half written; each write waits at most the 2 s busy timeout."""
        try:
            ok = self.flush_sync()
        finally:
            with self._db_lock:
                self._db.close()
        return ok
