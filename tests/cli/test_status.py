"""`runfleet status`: a cache-only summary for prompts and tmux (runfleet.status, DESIGN.md §8.5).

The cache is written with the real `Cache` (as the TUI does) and read back by the command; nothing here may run ssh,
build a FleetService or open the cache through `Cache` (its constructor creates and repairs the database).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli, status
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.models import ConnState, ErrorKind, HostError, HostSnapshot, HostStatus, Job, JobState, to_json
from tests.cli.fakes import job
from tests.conftest import CLOCK_START

NOW = CLOCK_START + timedelta(minutes=3)


def _drop_log_handlers() -> None:
    root = logging.getLogger("runfleet")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


class Env:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.cache_path = tmp_path / "state" / "cache.sqlite"
        self.config_path = tmp_path / "config.toml"
        self.clock = ManualClock(NOW)
        monkeypatch.setattr(cli.paths, "config_file", lambda: self.config_path)
        monkeypatch.setattr(cli.paths, "log_file", lambda: tmp_path / "state" / "runfleet.log")
        monkeypatch.setattr(cli.paths, "cache_file", lambda: self.cache_path)
        monkeypatch.setattr(cli, "RealClock", lambda: self.clock)

        def forbidden(*a: Any, **kw: Any) -> Any:
            raise AssertionError("runfleet status must not build a service, run ssh or open the cache with Cache()")
        for name in ("build_service", "ssh_available", "run_command", "open_cache", "Cache"):
            monkeypatch.setattr(cli, name, forbidden)

    def write(self, snapshots: dict[str, tuple[datetime, list[Job]]],
              states: dict[str, tuple[ConnState, HostError | None]] | None = None) -> None:
        c = Cache(self.cache_path)
        for host, (taken, jobs) in snapshots.items():
            c.put_snapshot(host, HostSnapshot(host, taken, None, {j.key.id: j for j in jobs}, (), (), (), (), None))
        for host, (state, err) in (states or {}).items():
            c.put_status(host, HostStatus(name=host, state=state, error=err), CLOCK_START - timedelta(hours=1))
        c.flush_sync()
        c._db.close()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Env]:
    _drop_log_handlers()
    yield Env(monkeypatch, tmp_path)
    _drop_log_handlers()


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = cli.main(args)
    out, err = capsys.readouterr()
    return rc, out, err


AUTH = (ConnState.NEEDS_AUTH, HostError(ErrorKind.AUTH, "permission denied (publickey)"))


def fleet(env: Env) -> None:
    """k2: 2 running, 1 pending, a fresh failure, an old failure, someone else's job; lynx needs a login."""
    mine_old = replace(job("k2", "7", JobState.FAILED), end_time=NOW - timedelta(days=2))
    theirs = replace(job("k2", "8", JobState.RUNNING), is_mine=False)
    fresh = replace(job("k2", "6", JobState.OUT_OF_MEMORY), end_time=NOW - timedelta(minutes=20))
    env.write({"k2": (CLOCK_START, [job("k2", "1"), job("k2", "2"), job("k2", "3", JobState.PENDING), fresh,
                                    mine_old, theirs]),
               "vega": (CLOCK_START - timedelta(minutes=1), [job("vega", "4", JobState.PENDING)]),
               "lynx": (CLOCK_START - timedelta(hours=2), [job("lynx", "5")])},
              {"k2": (ConnState.OK, None), "vega": (ConnState.OK, None), "lynx": AUTH})


# ---------------------------------------------------------------------------------------------------- parsing
def test_status_is_a_subcommand() -> None:
    assert cli.find_command(["status", "--short"]) == "status"
    assert cli.find_command(["-c", "x.toml", "status", "k2"]) == "status"
    ns = cli.parse_args(["status", "k2", "--short", "vega"])
    assert (ns.command, ns.hosts, ns.short, ns.json) == ("status", ["k2", "vega"], True, False)
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["status", "--short", "--json"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------------------------------- output
def test_short_line_counts_my_jobs_and_names_problem_hosts(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    rc, out, _ = run(["status", "--short"], capsys)
    # R: k2 1+2 and lynx 5 (theirs excluded) · PD: k2 3, vega 4 · F: the OOM 20 min ago (the 2-day-old one is out
    # of the 24 h window) · newest data: k2, 3 min old
    assert rc == 0 and out == "R3 PD2 F1 · lynx:auth · 3m\n"


def test_short_line_marks_stale_data(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    env.clock.advance(timedelta(minutes=8).total_seconds())
    assert run(["status", "--short"], capsys)[1] == "R3 PD2 F1 · lynx:auth · 11m stale\n"


def test_table_has_a_row_per_host_and_a_total(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    rc, out, _ = run(["status"], capsys)
    lines = [line.split() for line in out.splitlines()]
    assert rc == 0 and lines[0] == ["HOST", "STATE", "R", "PD", "F", "AGE"]
    assert lines[1] == ["k2", "ok", "2", "1", "1", "3m"]                     # no config: every cached host by name
    assert lines[2][:2] == ["lynx", "needs_auth"] and lines[2][-4:] == ["0", "0", "2h", "stale"]
    assert "needs_auth (permission denied (publicke…" in out             # long errors are cut (40 columns)
    assert lines[3] == ["vega", "ok", "0", "1", "0", "4m"]
    assert lines[4] == ["total", "3", "2", "1", "3m"]


def test_json_document(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    rc, out, _ = run(["status", "--json", "lynx", "k2"], capsys)
    doc = json.loads(out)
    assert rc == 0 and doc["schema"] == "runfleet.status/1" and doc["error"] is None
    assert doc["totals"] == {"running": 3, "pending": 1, "failed": 1}
    assert doc["problems"] == [{"host": "lynx", "problem": "auth"}]
    assert (doc["age_s"], doc["stale"], doc["stale_after_s"], doc["recent_window_s"]) == (180.0, False, 600.0, 86400.0)
    assert [h["name"] for h in doc["hosts"]] == ["lynx", "k2"]               # the order asked for
    lynx = doc["hosts"][0]
    assert (lynx["state"], lynx["problem"], lynx["stale"], lynx["running"]) == ("needs_auth", "auth", True, 1)
    assert lynx["state_since"] == (CLOCK_START - timedelta(hours=1)).isoformat()
    assert doc["data_at"] == CLOCK_START.isoformat()


def test_config_decides_hosts_order_and_window(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    env.config_path.write_text('[general]\nrecent_window = "3d"\n\n[[host]]\nname = "vega"\n\n[[host]]\nname = "k2"\n'
                               '\n[[host]]\nname = "lynx"\nenabled = false\n\n[[host]]\nname = "lyra"\n',
                               encoding="utf-8")
    rc, out, _ = run(["status"], capsys)
    names = [line.split()[0] for line in out.splitlines()[1:]]
    # config order, the disabled lynx left out, lyra (configured, never cached) listed without data
    assert rc == 0 and names == ["vega", "k2", "lyra", "total"]
    assert "no data" in out.splitlines()[3]
    assert run(["status", "--short"], capsys)[1] == "R2 PD2 F2 · 3m\n"      # 3 d window: the old failure counts
    assert run(["status", "--short", "lynx"], capsys)[1] == "R1 PD0 F0 · lynx:auth · 2h stale\n"   # named: shown
    env.config_path.write_text("[[host]\n", encoding="utf-8")                # broken: every cached host instead
    assert run(["status", "--short"], capsys)[1] == "R3 PD2 F1 · lynx:auth · 3m\n"


def test_a_config_with_every_host_disabled_shows_none_of_them(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """Review fix: an empty list of enabled hosts is not "no config" — the cached hosts it disabled stay out."""
    fleet(env)
    env.config_path.write_text('[[host]]\nname = "k2"\nenabled = false\n\n[[host]]\nname = "vega"\nenabled = false\n',
                               encoding="utf-8")
    assert run(["status", "--short"], capsys)[:2] == (1, "no data\n")
    rc, out, _ = run(["status"], capsys)
    assert rc == 1 and out == "no data: no enabled hosts in the config\n"
    assert run(["status", "--short", "k2"], capsys)[:2] == (0, "R2 PD1 F1 · 3m\n")     # named: still shown


def test_unknown_host_and_demo_are_usage_errors(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    fleet(env)
    rc, _, err = run(["status", "nope"], capsys)
    assert rc == 2 and "unknown host 'nope'" in err
    rc, _, err = run(["status", "--demo"], capsys)
    assert rc == 2 and "--demo" in err


# ---------------------------------------------------------------------------------------------------- no data
def test_missing_cache_is_no_data(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["status", "--short"], capsys)[:2] == (1, "no data\n")
    rc, out, _ = run(["status"], capsys)
    assert rc == 1 and out.startswith("no data: no cache yet")
    rc, out, _ = run(["status", "--json"], capsys)
    doc = json.loads(out)
    assert rc == 1 and doc["error"].startswith("no cache yet") and doc["hosts"] == [] and doc["stale"] is True
    assert doc["totals"] == {"running": 0, "pending": 0, "failed": 0}
    assert not env.cache_path.exists()                                     # never created


def test_garbage_file_is_no_data_and_left_alone(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.cache_path.parent.mkdir(parents=True)
    env.cache_path.write_bytes(b"not a database at all" * 100)
    assert run(["status", "--short"], capsys)[:2] == (1, "no data\n")
    assert env.cache_path.read_bytes() == b"not a database at all" * 100
    assert sorted(p.name for p in env.cache_path.parent.iterdir() if "cache" in p.name) == ["cache.sqlite"]


def test_old_schema_is_read_without_migrating_it(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """A cache written by RunFleet <= 0.5.0 has no host_status: counts and age still show, the file is not
    changed (Cache() would have added the table)."""
    env.cache_path.parent.mkdir(parents=True)
    db = sqlite3.connect(env.cache_path)
    db.execute("CREATE TABLE snapshots (host TEXT PRIMARY KEY, taken_at REAL, json TEXT)")
    snap = HostSnapshot("k2", CLOCK_START, None, {"1": job("k2", "1")}, (), (), (), (), None)
    db.execute("INSERT INTO snapshots VALUES (?, ?, ?)", ("k2", CLOCK_START.timestamp(), json.dumps(to_json(snap))))
    db.commit()
    db.close()
    before = env.cache_path.read_bytes()
    assert run(["status", "--short"], capsys)[:2] == (0, "R1 PD0 F0 · 3m\n")
    assert env.cache_path.read_bytes() == before
    db = sqlite3.connect(env.cache_path)
    assert [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")] == ["snapshots"]
    db.close()
    empty = env.cache_path.parent / "empty.sqlite"
    sqlite3.connect(empty).close()                                           # a database without our tables
    with pytest.raises(status.NoData, match="no snapshots"):
        status.read_cache(empty)


def test_locked_cache_is_no_data_not_an_error(env: Env, capsys: pytest.CaptureFixture[str],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    env.cache_path.parent.mkdir(parents=True)
    holder = sqlite3.connect(env.cache_path, isolation_level=None)           # rollback journal: readers wait
    holder.execute("CREATE TABLE snapshots (host TEXT PRIMARY KEY, taken_at REAL, json TEXT)")
    holder.execute("BEGIN EXCLUSIVE")
    monkeypatch.setattr(status, "BUSY_TIMEOUT", 0.05)
    try:
        rc, out, _ = run(["status", "--json"], capsys)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert rc == 1 and json.loads(out)["error"] == "cache busy"


def test_a_live_writer_does_not_block_status(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """WAL (as the TUI runs it): a write transaction in progress never blocks or breaks the reader."""
    fleet(env)
    writer = Cache(env.cache_path)
    writer._db.execute("BEGIN IMMEDIATE")
    writer._db.execute("DELETE FROM snapshots")
    try:
        assert run(["status", "--short"], capsys)[:2] == (0, "R3 PD2 F1 · lynx:auth · 3m\n")
    finally:
        writer._db.rollback()
        writer._db.close()


def test_count_jobs_skips_rows_it_cannot_read() -> None:
    doc = {"jobs": {"1": {"state": "R"}, "2": {"state": "??"}, "3": "junk", "4": {"state": "F", "end_time": "x"},
                    "5": {"state": "PD", "is_mine": False}, "6": {"state": "EX", "observed_at": NOW.isoformat()}}}
    assert status.count_jobs(json.dumps(doc), NOW, 3600) == (1, 0, 1)
    assert status.count_jobs("{broken", NOW, 3600) == (0, 0, 0)
