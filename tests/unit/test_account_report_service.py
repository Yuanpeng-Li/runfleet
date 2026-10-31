"""[WP5] FleetService.account_report / account_cached (§3.7, §5.2.11) over FakeTransport + StubSlurm, and the
`account_reports` cache table (§5.5): cache hit under max_age_s, one probe (purpose "account") on the on-demand lane,
in-flight dedup, never raises, read-only, workstations and `account_usage = false`, persistence and pruning."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from runfleet.agent.data import dest_hash
from runfleet.config import FleetConfig, GeneralSettings, HostConfig, SlurmSettings
from runfleet.core.cache import ACCOUNT_REPORTS_RETENTION, Cache
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService, merged_report, report_failures, report_read_nothing
from runfleet.core.store import ChangeSet
from runfleet.errors import RunFleetError
from runfleet.models import (AccountBalance, AccountReport, AccountSummary, AccountUse, ConnState, ErrorKind, HostError,
                             HostSnapshot, HostStatus, UsageCell, to_json)
from runfleet.transport.base import Transport
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, StubSlurm, run_for, settle

K2 = HostConfig(name="k2", ssh="k2", backend="slurm")
LYNX = HostConfig(name="lynx", ssh="lynx", backend="workstation")


class World:
    def __init__(self, hosts: tuple[HostConfig, ...], *, cache: Cache | None = None, latency: float = 0.0,
                 general: GeneralSettings | None = None) -> None:
        self.clock = ManualClock(CLOCK_START)
        self.transports: dict[str, FakeTransport] = {}
        self.latency = latency
        self.svc = FleetService(FleetConfig(hosts=hosts, general=general or GeneralSettings()), self.factory,
                                self.clock, cache)
        self.svc.hooks = STUB_HOOKS
        self.svc.lock_path = False

    def factory(self, host: HostConfig) -> Transport:
        t = self.transports.get(host.name)
        if t is None:
            t = FakeTransport(host.name, mux_mode="none", clock=self.clock, latency=self.latency)
            t.set_result("detect.id", "user=me")
            t.set_result("slurm.squeue", "101 R")
            t.set_result("process.ps", "642229 0")
            t.set_result("slurm.acct_share", "lab 298")
            self.transports[host.name] = t
        return t


@pytest.fixture
async def world() -> AsyncIterator[Callable[..., World]]:
    made: list[World] = []

    def make(*hosts: HostConfig, **kw: object) -> World:
        w = World(hosts or (K2,), **kw)   # type: ignore[arg-type]
        made.append(w)
        return w

    yield make
    for w in made:
        await w.svc.stop()


async def _started(w: World) -> World:
    await w.svc.start()
    await settle()
    return w


# ---------------------------------------------------------------- FleetService
async def test_one_probe_on_the_demand_lane_then_the_cache_answers(world: Callable[..., World]) -> None:
    w = await _started(world())
    t = w.transports["k2"]
    changes: list[ChangeSet] = []
    w.svc.store.subscribe(changes.append)
    assert w.svc.account_cached("k2", "30d") is None
    r1 = await w.svc.account_report("k2")                               # window None: general.usage_window
    assert r1.window == "30d" and r1.summary.balances[0].used_su == 298 and r1.errors == ()
    assert t.count("account") == 1 and w.svc.account_cached("k2", "30d") is r1
    assert any(c.host == "k2" and c.account for c in changes)            # store.notify_account
    w.clock.advance(599)
    assert await w.svc.account_report("k2", "30d") is r1 and t.count("account") == 1
    w.clock.advance(2)
    r2 = await w.svc.account_report("k2", "30d")
    assert r2 is not r1 and t.count("account") == 2
    r3 = await w.svc.account_report("k2", "30d", max_age_s=0)            # `r`
    assert r3.fetched_at == w.clock.now() and t.count("account") == 3
    (probe,) = [p for p in t.probes if p.script.purpose == "account"][:1]
    assert [s.name for s in probe.script.sections] == ["slurm.acct_share"]


async def test_default_window_and_period_fallback(world: Callable[..., World]) -> None:
    w = await _started(world(general=GeneralSettings(usage_window="7d")))
    assert (await w.svc.account_report("k2")).window == "7d"
    period = await w.svc.account_report("k2", "period")                  # no start: answered as 30d
    assert period.window == "30d" and w.svc.account_cached("k2", "period") is period
    assert w.svc.account_cached("k2", "30d") is period
    before = w.transports["k2"].count("account")
    assert await w.svc.account_report("k2", "period") is period and w.transports["k2"].count("account") == before


async def test_the_persisted_summary_is_handed_to_the_collector_first(world: Callable[..., World]) -> None:
    """Warm start (§5.2.11): the collector gets the snapshot's account summary before it plans a report, so the period
    window is known before its first `slurm.account` tick."""
    w = await _started(world())
    summary = AccountSummary(window="30d", read_at=w.clock.now(), mode="limits")
    snap = w.svc.store.hosts["k2"].snapshot
    assert snap is not None
    w.svc.store.set_snapshot("k2", replace(snap, account=summary))
    StubSlurm.restored.clear()
    await w.svc.account_report("k2", "period")
    assert StubSlurm.restored == [summary]


async def test_the_poller_hands_the_warm_start_summary_to_a_new_collector(world: Callable[..., World]) -> None:
    """Review: the periodic path (TUI restart, `agent --refresh`) restores the persisted readings too, not only an
    on-demand report, so a fresh collector neither re-reads a fresh summary nor overwrites it."""
    w = world()
    summary = AccountSummary(window="30d", read_at=w.clock.now(), mode="limits")
    w.svc.store.set_snapshot("k2", HostSnapshot(host="k2", taken_at=w.clock.now(), caps=None, jobs={}, gpus=(),
                                                procs=(), nodes=(), partitions=(), metrics=None, account=summary))
    StubSlurm.restored.clear()
    await _started(w)
    assert summary in StubSlurm.restored


async def test_an_identical_request_in_flight_is_awaited_not_duplicated(world: Callable[..., World]) -> None:
    w = await _started(world(latency=1.0))
    await run_for(w.clock, 2)
    t = w.transports["k2"]
    a = asyncio.ensure_future(w.svc.account_report("k2", "30d"))
    b = asyncio.ensure_future(w.svc.account_report("k2", "30d"))
    other = asyncio.ensure_future(w.svc.account_report("k2", "7d"))
    await run_for(w.clock, 4, step=0.25)
    ra, rb, ro = await asyncio.gather(a, b, other)
    assert ra is rb and ro.window == "7d" and t.count("account") == 2


async def test_a_cancelled_caller_does_not_cancel_the_fetch(world: Callable[..., World]) -> None:
    w = await _started(world(latency=1.0))
    await run_for(w.clock, 2)
    first = asyncio.ensure_future(w.svc.account_report("k2", "30d"))
    await settle()
    first.cancel()                                                      # a Textual worker replaced by `exclusive`
    second = asyncio.ensure_future(w.svc.account_report("k2", "30d"))
    await run_for(w.clock, 3, step=0.25)
    report = await second
    assert report.errors == () and w.transports["k2"].count("account") == 1


async def test_never_raises_a_transport_error_keeps_the_last_report(world: Callable[..., World]) -> None:
    w = await _started(world())
    t = w.transports["k2"]
    t.fail_next(HostError(ErrorKind.TIMEOUT, "probe timed out"))
    first = await w.svc.account_report("k2", "30d")
    assert first.errors[0].kind is ErrorKind.TIMEOUT and first.summary.balances == ()
    assert w.svc.account_cached("k2", "30d") is None                     # a failure is not kept
    good = await w.svc.account_report("k2", "30d")
    t.fail_next(HostError(ErrorKind.TIMEOUT, "probe timed out"))
    failed = await w.svc.account_report("k2", "30d", max_age_s=0)
    assert failed.errors[0].kind is ErrorKind.TIMEOUT and failed.summary == good.summary
    assert failed.fetched_at == good.fetched_at and w.svc.account_cached("k2", "30d") is good
    t.set_result("slurm.acct_share", "", 1, "sshare: error: Problem talking to the database")
    remote = await w.svc.account_report("k2", "30d", max_age_s=0)
    assert remote.errors[0].summary == "slurmdbd unreachable"            # section errors are a report like any


async def test_a_failing_sshare_keeps_the_last_good_balances_of_the_window(world: Callable[..., World]) -> None:
    """Review: a report whose sshare failed is merged with the last good report of that window (balances kept with
    their own time), never shown as 'usage only, balances 0s ago'."""
    w = await _started(world())
    good = await w.svc.account_report("k2", "30d")
    assert good.summary.balances
    w.clock.advance(120)
    w.transports["k2"].set_result("slurm.acct_share", "", 1, "sshare: error: Problem talking to the database")
    failed = await w.svc.account_report("k2", "30d", max_age_s=0)
    assert failed.errors and failed.summary.balances == good.summary.balances
    assert failed.summary.read_at == good.summary.read_at and w.svc.account_cached("k2", "30d") is failed


async def test_a_failing_sshare_keeps_the_snapshot_balances_without_an_earlier_report(
        world: Callable[..., World]) -> None:
    """Review: after a restart there is no report of the window yet, only the persisted summary: a report whose
    sshare failed takes its balances from that summary, with their own time."""
    w = await _started(world())
    first = await w.svc.account_report("k2", "30d")
    snap = w.svc.store.hosts["k2"].snapshot
    assert snap is not None and first.summary.balances
    w.svc.store.set_snapshot("k2", replace(snap, account=first.summary))
    w.svc._reports.clear()                                              # a restarted process: no report in memory
    w.clock.advance(120)
    w.transports["k2"].set_result("slurm.acct_share", "", 1, "sshare: error: Problem talking to the database")
    failed = await w.svc.account_report("k2", "7d", max_age_s=0)
    assert failed.errors and failed.summary.balances == first.summary.balances
    assert failed.summary.read_at == first.summary.read_at


async def test_a_period_report_answered_as_30d_persists_under_both_windows(tmp_path: Path,
                                                                           world: Callable[..., World]) -> None:
    w = await _started(world(cache=Cache(tmp_path / "cache.sqlite")))
    period = await w.svc.account_report("k2", "period")
    assert period.window == "30d"
    await w.svc.stop()
    again = await _started(world(cache=Cache(tmp_path / "cache.sqlite")))
    assert again.svc.account_cached("k2", "period") == period == again.svc.account_cached("k2", "30d")


def _r(at: float, *, errors: tuple[str, ...] = (), cells: bool = True, totals: bool = True) -> AccountReport:
    t = CLOCK_START + timedelta(seconds=at)
    cell = UsageCell("x", "lab", "gpu", "normal", "", 1, 0, 2.0, 0.0, 2.0, 2.0)
    use = AccountUse("lab", "account", 2.0, 0.0, 2.0)
    bal = AccountBalance("lab", 100.0, 10.0, read_at=t) if "slurm.acct_share" not in errors else AccountBalance("lab")
    no_sreport = any(e.startswith("slurm.acct_sreport") for e in errors)
    summary = AccountSummary(window="30d", read_at=t, mode="limits", balances=(bal,),
                             usage=() if no_sreport else (replace(use, scope="me"),), usage_at=None if no_sreport else t,
                             errors=tuple(HostError(ErrorKind.REMOTE, "x", section=e, at=t) for e in errors
                                          if not e.startswith("slurm.acct_jobs")))
    return AccountReport(host="k2", window="30d", fetched_at=t, summary=summary,
                         account_totals=(use,) if totals else (), cells=(cell,) if cells else (),
                         jobs_seen=1 if cells else 0,
                         errors=tuple(HostError(ErrorKind.REMOTE, "x", section=e, at=t) for e in errors))


def test_a_report_that_read_nothing_is_never_kept_and_a_partial_one_is_merged() -> None:
    """Review: an all-failing report (sshare, sreport, every chunk) is not stored over the last good one; a partial
    failure takes the failed parts (with their times) from the last good report of the window."""
    old = _r(0)
    nothing = _r(600, errors=("slurm.acct_share", "slurm.acct_sreport", "slurm.acct_jobs.c0"), cells=False,
                 totals=False)
    assert report_read_nothing(nothing) and not report_read_nothing(old)
    assert report_failures(nothing) == {"share", "sreport", "rollup"}
    rollup_only = _r(600, errors=("slurm.acct_jobs.c0",), cells=False)
    merged = merged_report(rollup_only, old)
    assert merged.cells == old.cells and merged.jobs_seen == 1 and merged.fetched_at == old.fetched_at
    assert merged.summary == rollup_only.summary                         # fresh balances and usage stay
    sreport = merged_report(_r(600, errors=("slurm.acct_sreport.p0",), totals=False), old)
    assert sreport.account_totals == old.account_totals and sreport.summary.usage == old.summary.usage
    assert sreport.summary.usage_at == old.summary.usage_at and sreport.fetched_at == CLOCK_START + timedelta(seconds=600)
    assert merged_report(rollup_only, None) is rollup_only and merged_report(old, rollup_only) is old


async def test_not_connected_yet_and_unknown_hosts(world: Callable[..., World]) -> None:
    w = world()                                                         # not started: no poller, no caps
    r = await w.svc.account_report("k2", "30d")
    assert r.errors and "not connected" in r.errors[0].summary and "k2" not in w.transports   # no ssh at all
    with pytest.raises(RunFleetError):
        await w.svc.account_report("nowhere", "30d")
    assert w.svc.account_cached("nowhere", "30d") is None


async def test_allowed_read_only(world: Callable[..., World]) -> None:
    w = await _started(world(general=GeneralSettings(readonly=True)))
    r = await w.svc.account_report("k2", "30d")
    assert r.errors == () and r.summary.balances and w.transports["k2"].count("account") == 1


async def test_hosts_without_scheduler_accounting_get_an_empty_report(world: Callable[..., World]) -> None:
    off = replace(K2, name="k2off", ssh="k2off", slurm=SlurmSettings(account_usage=False))
    w = await _started(world(K2, off, LYNX))
    ws = await w.svc.account_report("lynx", "30d")
    assert ws.summary.balances == () and ws.errors == () and ws.summary.notes == (
        "workstation: no scheduler accounting",)
    disabled = await w.svc.account_report("k2off", "period")
    assert disabled.window == "30d" and disabled.errors == () and "off for k2off" in disabled.summary.notes[0]
    assert w.transports["lynx"].count("account") == w.transports["k2off"].count("account") == 0


async def test_reports_persist_and_reload_from_the_cache(tmp_path: Path, world: Callable[..., World]) -> None:
    w = await _started(world(cache=Cache(tmp_path / "cache.sqlite")))
    report = await w.svc.account_report("k2", "90d")
    await w.svc.stop()                                                  # flushes
    again = world(cache=Cache(tmp_path / "cache.sqlite"))
    again.clock.advance(3 * 86400)
    assert again.svc.account_cached("k2", "90d") is None                # no I/O: start() loads them, off-loop
    await _started(again)
    cached = again.svc.account_cached("k2", "90d")                      # any age, no probe
    assert cached == report and again.svc.account_cached("k2", "30d") is None
    assert await again.svc.account_report("k2", "90d", max_age_s=4 * 86400) == report
    assert again.transports["k2"].count("account") == 0


# ---------------------------------------------------------------- cache table
def _report(at_offset: float = 0.0, host: str = "k2", window: str = "30d") -> AccountReport:
    at = CLOCK_START + timedelta(seconds=at_offset)
    return AccountReport(host=host, window=window, fetched_at=at,             # type: ignore[arg-type]
                         summary=AccountSummary(window=window, read_at=at, mode="usage_only"))  # type: ignore[arg-type]


def test_account_reports_table_round_trip_newest_wins_and_prune(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c.sqlite")
    c.put_account_report(_report(0))
    c.put_account_report(_report(0, window="7d"))
    assert c.load_account_reports("k2")["30d"] == _report(0)            # queued, not yet written: still seen
    assert c.flush_sync()
    other = Cache(tmp_path / "c.sqlite")                                # `agent usage --refresh` wrote a newer one
    other.put_account_report(_report(100))
    assert other.flush_sync()
    c.put_account_report(_report(50))                                   # older: does not overwrite
    assert c.flush_sync()
    assert Cache(tmp_path / "c.sqlite").load_account_reports("k2") == {"30d": _report(100), "7d": _report(0, window="7d")}
    c.prune(CLOCK_START + timedelta(seconds=ACCOUNT_REPORTS_RETENTION + 60))
    assert c.load_account_reports("k2") == {"30d": _report(100)}
    with sqlite3.connect(tmp_path / "c.sqlite") as db:
        db.execute("UPDATE account_reports SET json = 'garbage'")
    assert c.load_account_reports("k2") == {}                           # unreadable rows are left out


def test_account_reports_go_with_the_ssh_destination(tmp_path: Path) -> None:
    c = Cache(tmp_path / "c.sqlite")
    st = HostStatus(name="k2", state=ConnState.OK)
    c.put_health("k2", st, dest_hash=dest_hash("k2"), remote_user="alice", at=CLOCK_START)
    c.put_account_report(_report())
    assert c.flush_sync() and c.load_account_reports("k2")
    c.put_health("k2", st, dest_hash=dest_hash("k2-other"), remote_user="alice", at=CLOCK_START)
    assert c.flush_sync() and c.load_account_reports("k2") == {}


def test_the_table_is_created_on_an_old_cache(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    with sqlite3.connect(path) as db:                                   # a 0.8.0 cache: no account_reports
        db.execute("CREATE TABLE snapshots (host TEXT PRIMARY KEY, taken_at REAL, json TEXT)")
        snap = HostSnapshot("k2", CLOCK_START, None, {}, (), (), (), (), None)
        old = to_json(snap)
        del old["account"]
        db.execute("INSERT INTO snapshots VALUES (?, ?, ?)", ("k2", 0.0, json.dumps(old)))
    c = Cache(path)
    assert c.load_account_reports("k2") == {} and c.load_snapshots()["k2"].account is None
    c.put_account_report(_report())
    assert c.flush_sync() and c.load_account_reports("k2") == {"30d": _report()}
