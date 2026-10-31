"""[WP4] The periodic account summary (§5.2.11): SlurmCollector's `slurm.account` piggyback group — when it exists,
what it runs (sshare + sreport, sbank at most every 6 h), and how it ingests, keeps the last reading on failure and
treats a missing tool as an answer — from the anonymised account field probes."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import Any

import pytest

from runfleet.backends.base import HostContext, MonitorView
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.collector import (ACCOUNT_FLOOR, ACCOUNT_HIDDEN, ACCOUNT_MAX_BACKOFF, NO_SREPORT,
                                               SBANK_PERIOD, SLURMDBD, SlurmCollector, parse_qos, partition_charges)
from runfleet.config import GeneralSettings
from runfleet.models import AccountSummary, ErrorKind
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import SLURM_TOOLS, at, ctx, fixture_result, host_config, result
from tests.conftest import sample

ACCT_TOOLS = SLURM_TOOLS | {"sshare", "sreport"}
GROUP = frozenset({"slurm.account"})
ASSOC = {
    "lyra": (("lyra", "pilab_02", "", "normal"), ("lyra", "pilab_02_gpu", "gpu", "normal"),
             ("lyra", "pilab_03", "", "normal"), ("lyra", "alice07", "", "normal,low")),
    "k2": (("k2", "campus01", "", "normal"), ("far", "elsewhere", "", "normal")),
    "vega": (("vega", "pilab", "", "normal"),),
}
DECAY = {"lyra": ("00:00:00", "NONE"), "k2": ("7-00:00:00", "NONE"), "vega": ("14-00:00:00", "MONTHLY")}


def _ctx(host: str = "lyra", *, slurm: dict[str, Any] | None = None, window: str = "30d", **kw: Any) -> HostContext:
    """The host's context with the account field probe's accounting setup (slurmdbd, decay, QOS, associations)."""
    half, reset = DECAY[host]
    kw.setdefault("tools", ACCT_TOOLS | ({"sbank"} if host == "lyra" else set()))
    kw.setdefault("associations", ASSOC[host])
    kw.setdefault("qos_usage_factor", dict(parse_qos(sample(host, "sacctmgr_qos")).items))
    if host == "lyra":
        kw.setdefault("partition_charges", partition_charges(sample("lyra", "scontrol_partitions_charges")))
    c = ctx(host, cfg=host_config(host, slurm=slurm), **kw)
    config = {**c.caps.slurm_config, "AccountingStorageType": SLURMDBD, "PriorityDecayHalfLife": half,
              "PriorityUsageResetPeriod": reset}
    return dataclasses.replace(c, caps=dataclasses.replace(c.caps, slurm_config=config),
                               general=GeneralSettings(usage_window=window))  # type: ignore[arg-type]


def _results(host: str, *names: str) -> dict[str, Any]:
    fixture = {"slurm.acct_share": "acct_share", "slurm.acct_sreport": "acct_sreport",
               "slurm.acct_sbank": "acct_sbank"}
    return {n: fixture_result(n, host, fixture[n]) for n in names}


def _summary(c: SlurmCollector, results: dict[str, Any], cx: HostContext) -> AccountSummary:
    out = c.ingest(results, cx, MonitorView())
    assert out.account is not None
    return out.account


# ---------------------------------------------------------------- when the group exists
def test_group_is_a_piggyback_half_hour_with_hidden_and_backoff_factors() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    (cad,) = [cad for cad in c.cadences(cx) if cad.group == "slurm.account"]
    assert (cad.mode, cad.interval, cad.floor) == ("piggyback", 1800.0, ACCOUNT_FLOOR)
    assert cad.hidden_factor == ACCOUNT_HIDDEN == 2 and cad.focused_factor == 1
    assert cad.max_backoff == ACCOUNT_MAX_BACKOFF == 7200 and not cad.focused_only
    slow = _ctx(slurm={"poll_account": 3600.0})
    assert [cad.interval for cad in c.cadences(slow) if cad.group == "slurm.account"] == [3600.0]
    fast = _ctx(slurm={"poll_account": 60.0})                   # config clamps; the collector never goes below
    assert [cad.interval for cad in c.cadences(fast) if cad.group == "slurm.account"] == [ACCOUNT_FLOOR]


def test_group_needs_the_setting_slurmdbd_an_account_of_this_cluster_and_a_tool() -> None:
    c = SlurmCollector(_ctx().host)
    base = _ctx()
    off = _ctx(slurm={"account_usage": False})
    no_dbd = dataclasses.replace(base, caps=dataclasses.replace(
        base.caps, slurm_config={**base.caps.slurm_config, "AccountingStorageType": "accounting_storage/none"}))
    for blocked in (off, no_dbd, _ctx(associations=()), _ctx("k2", associations=(("far", "x", "", ""),)),
                    _ctx(tools=SLURM_TOOLS)):
        assert "slurm.account" not in {cad.group for cad in c.cadences(blocked)}
        assert c.sections(GROUP, blocked) == []
    assert "slurm.account" in {cad.group for cad in c.cadences(_ctx(tools=frozenset()))}   # tools unknown: assumed


# ---------------------------------------------------------------- what it runs
def test_sections_share_sreport_and_sbank_with_the_builders_commands() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    names = ["alice07", "pilab_02", "pilab_02_gpu", "pilab_03"]
    share, sreport, sbank = c.sections(GROUP, cx)
    assert share.name == "slurm.acct_share" and share.cmd == cmds.acct_share_cmd(names)
    assert sreport.name == "slurm.acct_sreport" and sreport.cmd == cmds.acct_sreport_cmd(names, "now-30days",
                                                                                        totals=False)
    assert sbank.name == "slurm.acct_sbank" and sbank.cmd == cmds.acct_sbank_cmd(names)
    assert [(s.timeout, s.max_stdout, s.requires) for s in (share, sreport, sbank)] == [
        (15, cmds.ACCT_SHARE_MAX_STDOUT, ("sshare",)), (20, cmds.ACCT_SREPORT_MAX_STDOUT, ("sreport",)),
        (20, cmds.ACCT_SBANK_MAX_STDOUT, ("sbank",))]
    week = _ctx(window="7d")
    assert "start=now-7days" in c.sections(GROUP, week)[1].cmd
    assert "start=now-30days" in c.sections(GROUP, _ctx(window="period"))[1].cmd    # no period in the summary
    k2 = _ctx("k2")
    assert [s.name for s in SlurmCollector(k2.host).sections(GROUP, k2)] == ["slurm.acct_share", "slurm.acct_sreport"]
    assert "campus01" in c.sections(GROUP, k2)[0].cmd and "elsewhere" not in c.sections(GROUP, k2)[0].cmd


def test_sbank_runs_again_only_after_six_hours_of_a_good_reading() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    c.ingest(_results("lyra", "slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank"), cx, MonitorView())
    later = at(cx, 1800)
    assert "slurm.acct_sbank" not in {s.name for s in c.sections(GROUP, later)}
    c.ingest({"slurm.acct_sbank": result("slurm.acct_sbank", "", None)}, at(cx, 1800), MonitorView())
    assert "slurm.acct_sbank" not in {s.name for s in c.sections(GROUP, at(cx, 3600))}   # failed: the 6 h stand
    due = at(cx, SBANK_PERIOD.total_seconds())
    assert "slurm.acct_sbank" in {s.name for s in c.sections(GROUP, due)}


# ---------------------------------------------------------------- ingest
def test_lyra_summary_from_sshare_sreport_and_sbank() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    s = _summary(c, _results("lyra", "slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank"), cx)
    assert s.window == "30d" and s.mode == "limits" and s.read_at == cx.now and s.usage_at == cx.now
    assert s.sbank_at == cx.now and s.half_life_s == 0 and s.reset_period is None and s.notes == () and s.errors == ()
    bal = {b.account: b for b in s.balances}
    assert list(bal) == ["alice07", "pilab_02", "pilab_02_gpu", "pilab_03"]
    p = bal["pilab_02"]
    assert (p.limit_su, p.limit_source, p.used_source, p.used_basis) == (97_885, "sshare", "sshare", "since_reset")
    assert p.used_su == pytest.approx(298.12, abs=0.01) and p.read_at == cx.now
    assert p.next_realloc == date(2027, 1, 14) and p.realloc_known
    assert bal["alice07"].next_realloc is None and bal["alice07"].realloc_known      # "No data found."
    use = {u.account: u for u in s.usage}
    assert list(use) == ["alice07", "pilab_02", "pilab_02_gpu", "pilab_03"]
    assert use["pilab_02"].cpu_h == pytest.approx(298.13, abs=0.01) and use["pilab_02"].start == cx.now - timedelta(30)
    assert (use["pilab_02_gpu"].cpu_h, use["pilab_02_gpu"].gpu_h) == (0, 0)          # sreport printed no rows: 0
    assert all(u.scope == "me" and u.source == "sreport" for u in s.usage)


@pytest.mark.parametrize(("host", "account", "half", "reset"), [("k2", "campus01", 7 * 86400, None),
                                                                ("vega", "pilab", 14 * 86400, "MONTHLY")])
def test_usage_only_clusters(host: str, account: str, half: float, reset: str | None) -> None:
    cx = _ctx(host)
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    s = _summary(c, _results(host, "slurm.acct_share", "slurm.acct_sreport"), cx)
    assert s.mode == "usage_only" and s.half_life_s == half and s.reset_period == reset and s.sbank_at is None
    (b,) = s.balances
    assert b.account == account and b.limit_su is None and b.used_basis == "decayed" and b.used_su is not None
    (u,) = s.usage
    assert u.account == account and u.gpu_h is not None and u.gpu_h > 0


def test_no_account_section_ran_means_unchanged() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    assert c.ingest({}, cx, MonitorView()).account is None
    assert c.ingest({"slurm.squeue": result("slurm.squeue", "")}, cx, MonitorView()).account is None


def test_a_failing_section_keeps_the_last_values_and_reports_a_classified_error() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    first = _summary(c, _results("lyra", "slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank"), cx)
    later = at(cx, 1800)
    down = "sreport: error: Problem talking to the database: Connection refused"
    out = c.ingest({"slurm.acct_share": fixture_result("slurm.acct_share", "lyra", "acct_share"),
                    "slurm.acct_sreport": result("slurm.acct_sreport", "", 1, down)}, later, MonitorView())
    s = out.account
    assert s is not None and s.read_at == later.now
    assert s.usage == first.usage and s.usage_at == cx.now                          # the last reading, with its age
    (e,) = s.errors
    assert (e.section, e.kind, e.summary) == ("slurm.acct_sreport", ErrorKind.REMOTE, "slurmdbd unreachable")
    assert e in out.errors                                                           # section status + backoff
    garbage = c.ingest({"slurm.acct_share": result("slurm.acct_share", "nonsense\nmore nonsense\n")}, at(cx, 3600),
                       MonitorView()).account
    assert garbage is not None and garbage.balances == s.balances and garbage.read_at == later.now
    assert {e.section: e.kind for e in garbage.errors} == {"slurm.acct_share": ErrorKind.PARSE,
                                                          "slurm.acct_sreport": ErrorKind.REMOTE}
    healed = _summary(c, _results("lyra", "slurm.acct_share", "slurm.acct_sreport"), at(cx, 5400))
    assert healed.errors == () and healed.usage_at == at(cx, 5400).now


def test_a_missing_sreport_is_an_answer_with_a_note() -> None:
    cx = _ctx("k2")
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    missing = result("slurm.acct_sreport", "", 127, "missing:sreport", outcome=SectionOutcome.MISSING)
    s = _summary(c, {**_results("k2", "slurm.acct_share"), "slurm.acct_sreport": missing}, cx)
    assert s.usage == () and s.notes == (NO_SREPORT,) and s.errors == () and len(s.balances) == 1
    known = _ctx("k2", tools=SLURM_TOOLS | {"sshare"})                   # detection knows: not even planned
    assert [x.name for x in c.sections(GROUP, known)] == ["slurm.acct_share"]
    s = _summary(c, _results("k2", "slurm.acct_share"), known)
    assert s.notes == (NO_SREPORT,)


def test_sbank_layout_errors_stay_with_their_account() -> None:
    cx = _ctx()
    c = SlurmCollector(cx.host)
    c.sections(GROUP, cx)
    good = sample("lyra", "acct_sbank")
    _summary(c, _results("lyra", "slurm.acct_sbank"), cx)
    broken = good.replace("PILAB_03            2,643 |       100,000          97,357", "something else entirely")
    s = _summary(c, {"slurm.acct_sbank": result("slurm.acct_sbank", broken)}, at(cx, SBANK_PERIOD.total_seconds()))
    (e,) = s.errors
    assert e.kind is ErrorKind.PARSE and e.section == "slurm.acct_sbank" and "pilab_03" in e.detail
    bal = {b.account: b for b in s.balances}
    assert bal["pilab_03"].next_realloc == date(2027, 4, 2)                     # the last good reading stays
    assert bal["pilab_02"].limit_su == 97_885 and bal["pilab_02"].limit_source == "sbank"   # no sshare read yet
