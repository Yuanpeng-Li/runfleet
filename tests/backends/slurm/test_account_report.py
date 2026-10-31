"""[WP5] The on-demand account report (§5.2.11): SlurmCollector.account_sections / ingest_account — the windows (7d /
30d / 90d / period), sacct in ≤ 30-day chunks with dedup, "Too wide of a date range" as an error, truncation, no
sreport — and `account.period_starts`, from the anonymised account field probes."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.process.collector import ProcessCollector
from runfleet.backends.slurm import account as acct
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.collector import NO_SREPORT_REPORT, SBANK_PERIOD, SlurmCollector
from runfleet.models import AccountBalance, AccountSummary, ErrorKind
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.backends.factories import NOW, TZ, at, caps, fixture_result, host_config, result
from tests.backends.slurm.test_account_summary import _ctx, _results
from tests.conftest import sample

GROUP = frozenset({"slurm.account"})


def _report_results(host: str, sections: list[str], **override: SectionResult) -> dict[str, SectionResult]:
    """Fixture answers for a planned report: sshare, sreport (the combined totals output), sbank and, as the first
    sacct chunk, the field probe's jobs; later chunks answer empty."""
    fixture = {"slurm.acct_share": "acct_share", "slurm.acct_sbank": "acct_sbank", "slurm.acct_jobs.c0": "acct_jobs"}
    out: dict[str, SectionResult] = {}
    for name in sections:
        if name in override:
            out[name] = override[name]
        elif name.startswith("slurm.acct_sreport"):
            out[name] = fixture_result(name, host, "acct_sreport_totals")
        elif name in fixture:
            out[name] = fixture_result(name, host, fixture[name])
        else:
            out[name] = result(name, "")
    return out


def _names(sections: list) -> list[str]:  # type: ignore[type-arg]
    return [s.name for s in sections]


# ---------------------------------------------------------------- planning
def test_k2_30d_plans_share_sreport_totals_and_one_chunk() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    share, sreport, jobs = c.account_sections("30d", cx)
    assert share.cmd == cmds.acct_share_cmd(["campus01"]) and share.timeout == 15
    assert sreport.name == "slurm.acct_sreport"
    assert sreport.cmd == cmds.acct_sreport_cmd(["campus01"], "now-30days", totals=True)
    assert (sreport.timeout, sreport.max_stdout) == (20, cmds.ACCT_SREPORT_MAX_STDOUT)
    assert jobs.name == "slurm.acct_jobs.c0" and jobs.cmd == cmds.acct_jobs_cmd(30, 0)
    assert (jobs.timeout, jobs.max_stdout, jobs.requires) == (60, cmds.ACCT_JOBS_MAX_STDOUT, ("sacct",))
    week = c.account_sections("7d", cx)
    assert "start=now-7days" in week[1].cmd and week[2].cmd == cmds.acct_jobs_cmd(7, 0)


def test_90d_is_three_chunks_oldest_first() -> None:
    cx = _ctx("k2")
    chunks = [s for s in SlurmCollector(cx.host).account_sections("90d", cx) if s.name.startswith("slurm.acct_jobs")]
    assert _names(chunks) == ["slurm.acct_jobs.c0", "slurm.acct_jobs.c1", "slurm.acct_jobs.c2"]
    assert [c.cmd for c in chunks] == [cmds.acct_jobs_cmd(90, 60), cmds.acct_jobs_cmd(60, 30), cmds.acct_jobs_cmd(30, 0)]


def test_nothing_to_read_without_accounting() -> None:
    off = _ctx("k2", slurm={"account_usage": False})
    c = SlurmCollector(off.host)
    assert c.account_sections("30d", off) == [] and c.ingest_account("30d", {}, off) is None
    no_assoc = _ctx("k2", associations=())
    assert SlurmCollector(no_assoc.host).account_sections("30d", no_assoc) == []
    ws = ProcessCollector(host_config("labstation"))                  # the default hooks: no scheduler accounting
    assert ws.account_sections("30d", off) == [] and ws.ingest_account("30d", {}, off) is None


def test_sbank_is_read_again_after_an_hour() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    assert "slurm.acct_sbank" in _names(c.account_sections("30d", cx))       # never read
    c.sections(GROUP, cx)
    c.ingest(_results("lyra", "slurm.acct_sbank"), cx, MonitorView())
    assert "slurm.acct_sbank" not in _names(c.account_sections("30d", at(cx, 1800)))
    assert "slurm.acct_sbank" in _names(c.account_sections("30d", at(cx, 3600)))
    assert "slurm.acct_sbank" not in _names(c.account_sections("30d", _ctx("k2")))   # not installed


def test_lyra_period_starts_one_year_before_the_next_reallocation() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    c.ingest(_results("lyra", "slurm.acct_share", "slurm.acct_sbank"), cx, MonitorView())
    secs = c.account_sections("period", cx)
    p0, p1 = [s for s in secs if s.name.startswith("slurm.acct_sreport")]
    assert (p0.name, p0.meta["accounts"], p0.meta["start"]) == ("slurm.acct_sreport.p0", ["pilab_02"], "2026-01-14")
    assert (p1.meta["accounts"], p1.meta["start"]) == (["pilab_02_gpu", "pilab_03"], "2026-04-02")
    assert p0.cmd == cmds.acct_sreport_cmd(["pilab_02"], "2026-01-14", totals=True)
    start = datetime(2026, 1, 14, tzinfo=timezone(TZ["lyra"]))                  # cluster-local midnight
    days = -(-(NOW - start) // timedelta(days=1))
    chunks = [s for s in secs if s.name.startswith("slurm.acct_jobs")]
    assert len(chunks) == len(cmds.acct_chunks(days)) == 8 and chunks[0].cmd == cmds.acct_jobs_cmd(days, 210)
    report = c.ingest_account("period", _report_results("lyra", _names(secs)), cx)
    assert report is not None and report.window == "period" and report.window_start == start
    use = {u.account: u for u in report.summary.usage}
    assert set(use) == {"pilab_02", "pilab_02_gpu", "pilab_03"}                 # alice07: "No data found."
    assert use["pilab_02"].start == start and use["pilab_02"].start_inferred
    assert all(u.start_inferred for u in report.account_totals)


def test_period_without_a_start_is_answered_as_30d() -> None:
    k2 = _ctx("k2")                                                     # no sbank, no reset period
    c = SlurmCollector(k2.host)
    secs = c.account_sections("period", k2)
    assert "start=now-30days" in secs[1].cmd
    report = c.ingest_account("period", _report_results("k2", _names(secs)), k2)
    assert report is not None and report.window == "30d" and report.summary.window == "30d"
    lyra = _ctx()                                                       # sbank not read yet
    assert "slurm.acct_sreport" in _names(SlurmCollector(lyra.host).account_sections("period", lyra))


def test_a_restart_takes_the_period_start_from_the_persisted_summary() -> None:
    """The sbank state is forgotten on restart; the persisted summary's reallocation dates (and its sbank time) give
    the period at once, sbank is read again only once that reading is older than REPORT_SBANK_PERIOD, and a later
    sbank reading is never overwritten by them."""
    cx = _ctx()
    first = SlurmCollector(cx.host)
    first.sections(GROUP, cx)
    out = first.ingest(_results("lyra", "slurm.acct_share", "slurm.acct_sbank"), cx, MonitorView())
    assert out.account is not None
    restarted = SlurmCollector(cx.host)
    restarted.restore_account(out.account)
    secs = restarted.account_sections("period", cx)
    assert [s.meta["start"] for s in secs if s.name.startswith("slurm.acct_sreport")] == ["2026-01-14", "2026-04-02"]
    assert "slurm.acct_sbank" not in _names(secs)                        # read just now, by the previous process
    later = dataclasses.replace(cx, now=cx.now + timedelta(hours=2))
    assert "slurm.acct_sbank" in _names(restarted.account_sections("period", later))
    moved = AccountSummary("30d", NOW, "limits", balances=(AccountBalance(
        "pilab_02", next_realloc=date(2030, 1, 1), realloc_known=True),))
    first.restore_account(moved)                                         # read sbank this run: ignored
    assert [s.meta["start"] for s in first.account_sections("period", cx)
            if s.name.startswith("slurm.acct_sreport")] == ["2026-01-14", "2026-04-02"]
    fresh = SlurmCollector(cx.host)
    fresh.restore_account(None)
    assert "slurm.acct_sreport" in _names(fresh.account_sections("period", cx))     # nothing to restore: 30d


def test_vega_period_is_since_the_monthly_reset() -> None:
    cx = _ctx("vega")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("period", cx)
    (p0,) = [s for s in secs if s.name.startswith("slurm.acct_sreport")]
    assert p0.meta["start"] == "2026-09-01" and _names(secs)[-1] == "slurm.acct_jobs.c0"
    report = c.ingest_account("period", _report_results("vega", _names(secs)), cx)
    assert report is not None and report.window == "period"
    assert report.window_start == datetime(2026, 9, 1, tzinfo=timezone(TZ["vega"]))
    assert not any(u.start_inferred for u in report.summary.usage)


# ---------------------------------------------------------------- ingest
def test_k2_report_from_the_field_probes() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    report = c.ingest_account("30d", _report_results("k2", _names(secs)), cx)
    assert report is not None and report.host == "k2" and report.fetched_at == cx.now and report.errors == ()
    assert report.window_start == cx.now - timedelta(days=30) and not report.truncated
    assert report.jobs_seen == 6 and len(report.cells) == 6 and {x.account for x in report.cells} == {"campus01"}
    (total,) = report.account_totals
    assert total.scope == "account" and total.cpu_h == pytest.approx(446176398 / 60)
    (mine,) = report.summary.usage
    assert mine.scope == "me" and mine.cpu_h == pytest.approx(937961 / 60) and mine.source == "sreport"
    s = report.summary
    assert s.window == "30d" and s.mode == "usage_only" and s.usage_at == cx.now and s.read_at == cx.now
    assert [b.account for b in s.balances] == ["campus01"] and s.notes == () and s.errors == ()


def test_a_report_refreshes_the_summary_the_next_tick_publishes() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    c.ingest_account("30d", _report_results("k2", _names(secs)), cx)
    out = c.ingest({}, at(cx, 5), MonitorView()).account
    assert out is not None and out.read_at == cx.now and out.usage_at == cx.now and len(out.usage) == 1
    assert c.ingest({}, at(cx, 10), MonitorView()).account is None              # once
    week = c.account_sections("7d", cx)                                         # another window: balances only
    c.ingest_account("7d", _report_results("k2", _names(week)), at(cx, 60))
    out = c.ingest({}, at(cx, 65), MonitorView()).account
    assert out is not None and out.read_at == at(cx, 60).now and out.usage_at == cx.now


def test_lyra_rollup_is_free_and_sbank_refreshes_the_dates() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    assert "slurm.acct_sbank" in _names(secs)
    report = c.ingest_account("30d", _report_results("lyra", _names(secs)), cx)
    assert report is not None and report.jobs_seen == 7 and report.errors == ()
    assert all(x.charged_su == 0 and x.qos == "low" for x in report.cells)
    bal = {b.account: b for b in report.summary.balances}
    assert bal["pilab_02"].next_realloc == date(2027, 1, 14) and report.summary.sbank_at == cx.now
    assert report.summary.mode == "limits"
    assert "slurm.acct_sbank" not in {s.name for s in c.sections(GROUP, at(cx, 60))}   # the summary's 6 h too
    assert "slurm.acct_sbank" in {s.name for s in c.sections(GROUP, at(cx, SBANK_PERIOD.total_seconds()))}


def test_too_wide_a_range_is_an_error_never_no_jobs() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    refused = fixture_result("slurm.acct_jobs.c0", "k2", "acct_jobs_too_wide")
    report = c.ingest_account("30d", _report_results("k2", _names(secs), **{"slurm.acct_jobs.c0": refused}), cx)
    assert report is not None and report.cells == () and report.jobs_seen == 0
    assert report.truncated                                     # a refused chunk: the rollup is a lower bound
    (e,) = report.errors
    assert (e.section, e.kind, e.summary) == ("slurm.acct_jobs.c0", ErrorKind.REMOTE, "sacct refused the date range")
    assert report.summary.errors == () and len(report.summary.usage) == 1         # sreport still answered


def test_chunks_dedup_and_a_cut_chunk_marks_the_report_truncated() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("90d", cx)
    rows = sample("k2", "acct_jobs").rstrip("\n")
    first = rows.split("\n")[0]
    cut = result("slurm.acct_jobs.c0", first + "\n" + first[:20], None, outcome=SectionOutcome.INCOMPLETE)
    report = c.ingest_account("90d", _report_results("k2", _names(secs), **{
        "slurm.acct_jobs.c0": cut, "slurm.acct_jobs.c1": result("slurm.acct_jobs.c1", first),
        "slurm.acct_jobs.c2": result("slurm.acct_jobs.c2", rows)}), cx)
    assert report is not None and report.truncated and report.jobs_seen == 6       # the boundary job counted once
    assert [e.section for e in report.errors] == ["slurm.acct_jobs.c0"]


def test_a_chunk_not_run_or_failing_marks_the_report_truncated() -> None:
    """Review: a chunk the probe never reached (NOT_RUN: cut at the local timeout) or one slurmdbd failed leaves the
    rollup a lower bound: `truncated`, never a complete-looking partial count."""
    cx = _ctx("k2")
    for bad in (result("slurm.acct_jobs.c2", "", None, outcome=SectionOutcome.NOT_RUN),
                result("slurm.acct_jobs.c2", "", 1, "sacct: error: Problem talking to the database",
                       outcome=SectionOutcome.FAILED)):
        c = SlurmCollector(cx.host)
        secs = c.account_sections("90d", cx)
        report = c.ingest_account("90d", _report_results("k2", _names(secs), **{"slurm.acct_jobs.c2": bad}), cx)
        assert report is not None and report.truncated, bad.outcome
    c = SlurmCollector(cx.host)
    secs = c.account_sections("90d", cx)
    whole = c.ingest_account("90d", _report_results("k2", _names(secs)), cx)
    assert whole is not None and not whole.truncated                    # empty chunks are answers


def test_a_chunk_losing_rows_to_parsing_marks_the_report_truncated() -> None:
    """Review: rows that fail to parse (a few, or a whole mostly-bad chunk) are jobs missing from the rollup."""
    cx = _ctx("k2")
    rows = sample("k2", "acct_jobs").rstrip("\n")
    for c0 in (rows + "\ngarbage", "garbage\ngarbage\n" + rows.split("\n")[0]):
        c = SlurmCollector(cx.host)
        secs = c.account_sections("90d", cx)
        report = c.ingest_account("90d", _report_results("k2", _names(secs), **{
            "slurm.acct_jobs.c0": result("slurm.acct_jobs.c0", c0)}), cx)
        assert report is not None and report.truncated, c0[:20]


def test_an_account_sreport_left_out_used_nothing_for_the_total_too() -> None:
    """Review (lyra acct_sreport_totals: no rows at all for an account without usage): when sreport answered,
    such an account's total is 0 like the user's own usage, never 'account unknown'."""
    cx = _ctx()
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    report = c.ingest_account("30d", _report_results("lyra", _names(secs)), cx)
    assert report is not None
    totals = {u.account: u for u in report.account_totals}
    mine = {u.account: u for u in report.summary.usage}
    assert set(totals) == set(mine) == {"alice07", "pilab_02", "pilab_02_gpu", "pilab_03"}
    assert (totals["pilab_02_gpu"].cpu_h, totals["pilab_02_gpu"].gpu_h, totals["pilab_02_gpu"].scope) == (
        0.0, 0.0, "account")
    assert totals["pilab_02"].cpu_h == pytest.approx(78980 / 60)


def test_without_sreport_my_usage_is_summed_from_the_rollup() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    missing = result("slurm.acct_sreport", "", 127, "missing:sreport", outcome=SectionOutcome.MISSING)
    report = c.ingest_account("30d", _report_results("k2", _names(secs), **{"slurm.acct_sreport": missing}), cx)
    assert report is not None and report.errors == () and report.account_totals == ()
    (use,) = report.summary.usage
    assert use.source == "sacct" and use.gpu_h == pytest.approx(sum(x.gpu_h for x in report.cells))
    assert report.summary.notes == (NO_SREPORT_REPORT,)
    known = _ctx("k2", tools=cx.caps.tools - {"sreport"})               # detection knows: not even planned
    secs = c.account_sections("30d", known)
    assert "slurm.acct_sreport" not in _names(secs)
    report = c.ingest_account("30d", _report_results("k2", _names(secs)), known)
    assert report is not None and report.summary.notes == (NO_SREPORT_REPORT,)


def test_without_sreport_a_failing_sacct_chunk_leaves_the_usage_unknown_not_zero() -> None:
    """Review: no sreport and slurmdbd down for sacct: no fresh 0 CPU-h / GPU-h totals, the usage stays unknown."""
    cx = _ctx("k2", tools=_ctx("k2").caps.tools - {"sreport"})
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    down = result("slurm.acct_jobs.c0", "", 1, "sacct: error: Problem talking to the database",
                  outcome=SectionOutcome.FAILED)
    report = c.ingest_account("30d", _report_results("k2", _names(secs), **{"slurm.acct_jobs.c0": down}), cx)
    assert report is not None and report.truncated
    assert report.summary.usage == () and report.summary.usage_at is None


def test_without_sreport_a_cut_or_partly_bad_chunk_gives_no_account_sums() -> None:
    """Review: sums of an incomplete set of sacct rows (cut at max_stdout, some rows unreadable) are never published
    as the account's usage: unknown, while the rollup shows as a lower bound."""
    cx = _ctx("k2", tools=_ctx("k2").caps.tools - {"sreport"})
    rows = sample("k2", "acct_jobs").rstrip("\n")
    cut = result("slurm.acct_jobs.c0", rows + "\n" + rows.split("\n")[0][:20], None,
                 outcome=SectionOutcome.INCOMPLETE)
    for c0 in (cut, result("slurm.acct_jobs.c0", rows + "\ngarbage")):
        c = SlurmCollector(cx.host)
        secs = c.account_sections("30d", cx)
        report = c.ingest_account("30d", _report_results("k2", _names(secs), **{"slurm.acct_jobs.c0": c0}), cx)
        assert report is not None and report.truncated and report.cells
        assert report.summary.usage == () and report.summary.usage_at is None


def test_a_warm_started_collector_keeps_the_cached_usage_across_a_report_of_another_window() -> None:
    """Review: after a restart, a 7d report published at the next tick must not blank the cached 30d usage."""
    cx = _ctx()
    first = SlurmCollector(cx.host)
    first.sections(GROUP, cx)
    cached = first.ingest(_results("lyra", "slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank"), cx,
                          MonitorView()).account
    assert cached is not None and cached.window == "30d" and cached.usage
    c = SlurmCollector(cx.host)
    c.restore_account(cached)
    secs = c.account_sections("7d", cx)
    assert c.ingest_account("7d", _report_results("lyra", _names(secs)), cx) is not None
    out = c.ingest({}, cx, MonitorView()).account                       # the report's state, published
    assert out is not None and out.window == "30d"
    assert out.usage == cached.usage and out.usage_at == cached.usage_at


def test_a_failing_sreport_is_a_classified_error_in_report_and_summary() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    secs = c.account_sections("30d", cx)
    down = result("slurm.acct_sreport", "", 1, "sreport: error: Problem talking to the database: Connection refused")
    report = c.ingest_account("30d", _report_results("k2", _names(secs), **{"slurm.acct_sreport": down}), cx)
    assert report is not None and report.summary.usage == () and report.jobs_seen == 6
    (e,) = report.errors
    assert e.summary == "slurmdbd unreachable" and report.summary.errors == (e,)


def test_ingest_without_a_plan_is_none() -> None:
    cx = _ctx("k2")
    assert SlurmCollector(cx.host).ingest_account("30d", {}, cx) is None


# ---------------------------------------------------------------- period_starts
def _summary(*balances: AccountBalance) -> AccountSummary:
    return AccountSummary(window="30d", read_at=NOW, mode="limits", balances=balances)


@pytest.mark.parametrize(("reset", "start"), [
    ("DAILY", datetime(2026, 9, 7)), ("WEEKLY", datetime(2026, 9, 6)), ("MONTHLY", datetime(2026, 9, 1)),
    ("QUARTERLY", datetime(2026, 7, 1)), ("YEARLY", datetime(2026, 1, 1)), ("NONE", None), ("NOW", None)])
def test_reset_boundaries_in_the_cluster_offset(reset: str, start: datetime | None) -> None:
    tz = timedelta(hours=-5)
    got = acct.reset_start(reset, NOW, tz)                              # NOW is a Monday, 06:02 at -05:00
    assert got == (None if start is None else start.replace(tzinfo=timezone(tz)))
    assert acct.reset_start("DAILY", datetime(2026, 9, 7, 3, tzinfo=UTC), tz) == datetime(
        2026, 9, 6, tzinfo=timezone(tz))                                # still the 6th over there


def test_period_rules() -> None:
    c = dataclasses.replace(caps("k2"), slurm_config={"PriorityUsageResetPeriod": "MONTHLY"})
    cfg = host_config("k2")
    sept = datetime(2026, 9, 1, tzinfo=timezone(TZ["k2"]))
    lab = AccountBalance("lab", next_realloc=date(2027, 1, 14), realloc_known=True)
    none = AccountBalance("none", realloc_known=True)                   # "No data found."
    plain = AccountBalance("plain")                                     # no sbank reading: the reset rule
    pw = acct.period_starts(_summary(lab, none, plain), c, cfg, NOW)
    assert pw.starts == {"lab": datetime(2026, 1, 14, tzinfo=timezone(TZ["k2"])), "plain": sept}
    assert pw.inferred == {"lab"} and pw.reset == "MONTHLY" and not pw.clipped and pw.earliest == pw.starts["lab"]
    off = host_config("k2", slurm={"allocation_period": 0.0})           # unknown period: dates are not used
    assert acct.period_starts(_summary(lab), c, off, NOW).starts == {"lab": sept}
    future = AccountBalance("lab", next_realloc=date(2028, 1, 1), realloc_known=True)
    assert acct.period_starts(_summary(future), c, cfg, NOW).starts == {}
    old = acct.period_starts(_summary(lab), c, host_config("k2", slurm={"allocation_period": 900 * 86400.0}), NOW)
    assert old.clipped and NOW - old.starts["lab"] <= timedelta(days=acct.MAX_PERIOD_DAYS)
    assert len(cmds.acct_chunks(-(-(NOW - old.starts["lab"]) // timedelta(days=1)))) == 13
    many = [AccountBalance(f"a{i}", next_realloc=date(2027, 1, 1 + i), realloc_known=True) for i in range(6)]
    pw = acct.period_starts(_summary(*many), c, cfg, NOW)
    assert sorted(pw.starts) == ["a0", "a1", "a2", "a3"] and len(set(pw.starts.values())) == acct.MAX_PERIOD_STARTS
    bare = dataclasses.replace(c, slurm_config={})                      # k2: no dates, no reset
    assert acct.period_starts(_summary(plain), bare, cfg, NOW).starts == {}
    assert acct.period_starts(None, None, cfg, NOW).starts == {}
