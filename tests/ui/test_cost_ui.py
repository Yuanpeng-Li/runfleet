"""Job cost in the UI (Account, §5.2.11, §7.5, §7.6, §7.17): the filter terms `acct:` / `part:` / `su:` / `gpuh:`, the
SU / GPU-h sort keys and columns, the preview's cost part, and the job detail's `cost` line, compact one-liner and
Cost box (charged, free, usage only, pending, unknown QOS, account balance)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Any, Literal

import pytest
from rich.text import Text
from textual.widgets import DataTable, Static
from textual.widgets.data_table import ColumnKey

from runfleet.models import AccountBalance, AccountSummary, Capabilities, Job, JobKey, JobState, PartitionCharge
from runfleet.ui import query
from runfleet.ui.commands import RunFleetCommands
from runfleet.ui.query import CostFn
from runfleet.ui.screens import job_detail as jd
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.widgets import job_table
from runfleet.ui.widgets.job_preview import cost_meta, meta_line
from runfleet.ui.widgets.job_table import JobTable
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.util import cost as costs
from tests.ui.fakes import T0, build_world, slurm_job

H = timedelta(hours=1)


def _caps(factors: dict[str, float], charges: tuple[PartitionCharge, ...] = ()) -> Capabilities:
    return Capabilities(remote_user="alice07", uid=1000, login_host="login1", tz_offset=timedelta(0),
                        clock_skew=timedelta(0), tools=frozenset(), cluster_name="lyra",
                        qos_usage_factor=factors, partition_charges=charges)


# lyra-shaped: a charged `gpu` partition with weights, a free one allowing only ×0 QOS
LYRA = _caps({"normal": 1.0, "low": 0.0, "guest": 0.0},
             (PartitionCharge("gpu", "cpu=1,gres/gpu=32"), PartitionCharge("free-gpu", allow_qos=("low", "guest"))))
K2 = _caps({"normal": 1.0, "gpu-priority": 1.0}, (PartitionCharge("gpu-preempt"),))     # usage only


def _job(state: JobState = JobState.RUNNING, *, qos: str = "normal", partition: str = "gpu",
         billing: str | None = "40", elapsed: timedelta | None = 4 * H, limit: timedelta | None = 72 * H,
         gpus: int = 1, account: str = "pilab_02", jid: str = "100") -> Job:
    job = slurm_job("lyra", jid, "mytask--train--lr--scratch--s0--a01", state, elapsed=elapsed, limit=limit,
                    partition=partition, qos=qos, gpus=gpus, gpu_type="rtx6000" if gpus else None)
    tres = dict(job.alloc.tres)
    if billing is None:
        tres.pop("billing")
    else:
        tres["billing"] = billing
    return dataclasses.replace(job, account=account, alloc=dataclasses.replace(job.alloc, tres=tres))


def _cost_fn(caps: Capabilities | None) -> CostFn:
    return lambda job, now: costs.job_cost(job, now, caps)


# ---------------------------------------------------------------- filter terms (§7.6)
def test_parse_cost_and_account_terms() -> None:
    q = query.parse("acct:pilab_02,Lab part:free-gpu p:gpu su:>10 !gpuh:<=2.5 su:3 su:>x gpuh:>=")
    kinds = [(t.kind, t.values, t.negate, t.number) for t in q.terms]
    assert kinds == [("account", ("pilab_02", "lab"), False, 0.0), ("partition", ("free-gpu",), False, 0.0),
                     ("partition", ("gpu",), False, 0.0), ("su", (">",), False, 10.0),
                     ("gpuh", ("<=",), True, 2.5), ("su", ("=",), False, 3.0),
                     ("text", ("su:>x",), False, 0.0), ("text", ("gpuh:>=",), False, 0.0)]   # malformed -> text
    assert [t.kind for t in query.parse("account:a partition:b").terms] == ["account", "partition"]


def test_account_and_partition_terms() -> None:
    job = _job()
    assert query.matches(query.parse("acct:PILAB_02"), job)
    assert not query.matches(query.parse("acct:lab"), job)
    pending = dataclasses.replace(_job(JobState.PENDING), partition="gpu,free-gpu")
    assert query.matches(query.parse("part:free-gpu"), pending)       # any partition of a pending job's list
    assert not query.matches(query.parse("p:gpuxl"), pending)


def test_cost_terms_compare_and_unknown_never_matches() -> None:
    fn = _cost_fn(LYRA)
    charged = _job()                                   # 40/h × 4 h × normal ×1 = 160 SU, 4 GPU-h
    free = _job(qos="low", partition="free-gpu")       # 0 SU
    assert query.matches(query.parse("su:>100"), charged, T0, fn)
    assert not query.matches(query.parse("su:>100"), free, T0, fn)
    assert query.matches(query.parse("su:<=0"), free, T0, fn)
    assert query.matches(query.parse("su:=160 gpuh:>=4"), charged, T0, fn)
    assert not query.matches(query.parse("gpuh:>4"), charged, T0, fn)
    # pending: SU is the projection at the limit (40 × 72 = 2,880)
    assert query.matches(query.parse("su:>2000"), _job(JobState.PENDING, elapsed=None), T0, fn)
    # unknown cost (no callback, or a workstation task): never matches, a negated term keeps it
    assert not query.matches(query.parse("su:>10"), charged)
    assert query.matches(query.parse("!su:>10"), charged)
    ws = dataclasses.replace(charged, kind=job_table.JobKind.TASK)
    assert not query.matches(query.parse("su:>=0"), ws, T0, fn)
    assert query.matches(query.parse("!su:>=0"), ws, T0, fn)


def test_su_equals_matches_what_the_column_shows() -> None:
    """Review: the column shows 33 for 33.4 SU, so `su:=33` must find it (the old 0.05 tolerance did not)."""
    fn = _cost_fn(LYRA)
    job = _job(billing="8.35")                         # 8.35/h × 4 h = 33.4 SU, shown `33`
    assert job_table.job_cells(job, [("su", 8)], _ctx(LYRA), cursor=False)[0].plain.strip() == "33"
    assert query.matches(query.parse("su:=33"), job, T0, fn)
    assert not query.matches(query.parse("su:=34"), job, T0, fn)
    small = _job(billing="0.725")                      # 2.9 SU: one decimal below 10
    assert query.matches(query.parse("su:=2.9"), small, T0, fn) and not query.matches(query.parse("su:=3"), small, T0, fn)


# ---------------------------------------------------------------- sorting (§7.5)
def test_sort_by_su_and_gpuh_largest_first_unknowns_last() -> None:
    fn = _cost_fn(LYRA)
    small = _job(jid="1", elapsed=1 * H)
    big = _job(jid="2", elapsed=10 * H)
    free = _job(jid="3", qos="low", partition="free-gpu", elapsed=20 * H)
    task = dataclasses.replace(_job(jid="4"), kind=job_table.JobKind.TASK)
    jobs = [small, task, free, big]
    out = query.sort_jobs(jobs, "su", False, host_order=["lyra"], now=T0, cost=fn)
    assert [j.key.id for j in out] == ["2", "1", "3", "4"]
    out = query.sort_jobs(jobs, "su", True, host_order=["lyra"], now=T0, cost=fn)
    assert [j.key.id for j in out] == ["3", "1", "2", "4"]          # ascending, the unknown still last
    out = query.sort_jobs(jobs, "gpuh", False, host_order=["lyra"], now=T0, cost=fn)
    assert [j.key.id for j in out] == ["3", "2", "1", "4"]
    # without a cost callback every value is unknown: the default order, nothing lost
    assert len(query.sort_jobs(jobs, "su", False, host_order=["lyra"], now=T0)) == 4
    assert "su" not in query.SORT_CYCLE and "gpuh" not in query.SORT_CYCLE
    assert query.SORT_LABELS["su"] == "SU" and query.sort_arrow("gpuh", False) == "▼"


def test_cost_sorts_are_total_on_the_fake_world() -> None:
    _config, states = build_world()
    caps = {s.config.name: s.snapshot.caps for s in states if s.snapshot is not None}
    jobs = [j for s in states if s.snapshot is not None for j in s.snapshot.jobs.values()]
    for column in query.COST_SORTS:
        out = query.sort_jobs(jobs, column, False, host_order=[], now=T0,
                              cost=lambda j, now: costs.job_cost(j, now, caps.get(j.key.host)))
        assert sorted(map(str, (j.key for j in out))) == sorted(map(str, (j.key for j in jobs)))


# ---------------------------------------------------------------- table cells and preview
def _ctx(caps: Capabilities | None, now: datetime = T0) -> job_table.RowContext:
    return job_table.RowContext(now, {}, {}, frozenset(), frozenset(), (10, 75), cost=_cost_fn(caps))


def test_cost_cells() -> None:
    cols = [("su", 8), ("gpuh", 8)]
    su, gpuh = job_table.job_cells(_job(), cols, _ctx(LYRA), cursor=False)
    assert su.plain == "    160" and gpuh.plain == "    4.0"              # right-aligned in 7 cells (+ the gap)
    free = job_table.job_cells(_job(qos="low", partition="free-gpu"), cols, _ctx(LYRA), cursor=False)[0]
    assert free.plain.strip() == "0" and free.spans                     # dim
    pending = job_table.job_cells(_job(JobState.PENDING, elapsed=None), cols, _ctx(LYRA), cursor=False)[0]
    assert pending.plain.strip() == "≈2,880"
    task = dataclasses.replace(_job(), kind=job_table.JobKind.TASK)
    assert job_table.job_cells(task, cols, _ctx(LYRA), cursor=False)[0].plain.strip() == "—"
    # a running row ticks with the clock (the 1 Hz refresh re-renders it)
    later = job_table.job_cells(_job(), cols, _ctx(LYRA, T0 + H), cursor=False)[0]
    assert later.plain.strip() == "200"
    assert job_table.COLUMNS["su"].priority == 4 and "su" not in job_table.layout(120, ("host",), show_user=False)


def test_su_cells_of_a_usage_only_cluster_are_billing_hours() -> None:
    """Review: one SU column mixed charged SU (lyra) and free billing-h (k2); the latter now carry an `h`, muted."""
    cols = [("su", 8)]
    ctx = dataclasses.replace(_ctx(LYRA), usage_only_hosts=frozenset({"lyra"}))
    cell = job_table.job_cells(_job(), cols, ctx, cursor=False)[0]
    assert cell.plain.strip() == "160h" and cell.spans
    pending = job_table.job_cells(_job(JobState.PENDING, elapsed=None), cols, ctx, cursor=False)[0]
    assert pending.plain.strip() == "≈2,880h"
    assert job_table.job_cells(_job(), cols, _ctx(LYRA), cursor=False)[0].plain.strip() == "160"


def test_preview_cost_part() -> None:
    assert cost_meta(_job(), costs.job_cost(_job(), T0, LYRA), False) == "160 SU"
    usage = _job(partition="gpu-preempt", qos="gpu-priority", billing="8")
    assert cost_meta(usage, costs.job_cost(usage, T0, K2), True) == "32 billing-h"
    free = _job(qos="low", partition="free-gpu")
    assert cost_meta(free, costs.job_cost(free, T0, LYRA), False) == "free (qos low ×0)"
    pending = _job(JobState.PENDING, elapsed=None)
    assert cost_meta(pending, costs.job_cost(pending, T0, LYRA), False) == "≈ 2,880 SU at limit"
    assert cost_meta(_job(), None, False) is None
    assert meta_line(_job(), "160 SU").plain.endswith(" · 160 SU")


# ---------------------------------------------------------------- job detail: Resources `cost` line
def _line(job: Job, caps: Capabilities | None, width: int = 120, mode: str | None = None) -> str:
    line = jd.cost_line(job, costs.job_cost(job, T0, caps), caps, width, mode=mode)
    assert line is not None
    return line.plain.rstrip()


def test_cost_line_variants() -> None:
    assert _line(_job(), LYRA) == "cost    32.0 CPU-h · 4.0 GPU-h · 40/h × normal ×1 = 160 SU · ≈ 2,880 SU at limit"
    assert _line(_job(qos="low", partition="free-gpu", billing="8"), LYRA) == \
        "cost    4.0 GPU-h RTX6000 · free (qos low ×0) · 32 billing-h not charged"
    usage = _job(partition="gpu-preempt", qos="gpu-priority", billing="8")
    assert _line(usage, K2) == "cost    32.0 CPU-h · 4.0 GPU-h · 32 billing-h (= CPU-h here)"
    pending = _job(JobState.PENDING, billing=None, elapsed=None)                      # billing from the weights
    assert _line(pending, LYRA) == "cost    ≈ 2,880 SU if it runs to its limit (billing from weights: estimate)"
    assert _line(_job(qos="weird"), LYRA).startswith("cost    32.0 CPU-h · 4.0 GPU-h · 40/h × weird ×1? = 160 SU")
    assert "× qos? ×1? =" in _line(dataclasses.replace(_job(), qos=None), LYRA)
    ended = _job(JobState.COMPLETED)
    assert _line(ended, LYRA) == "cost    32.0 CPU-h · 4.0 GPU-h · 40/h × normal ×1 = 160 SU"   # no projection
    # the account summary's mode decides SU or billing-h once read
    assert "40/h × normal ×1 = 160 SU" in _line(_job(), K2, mode="limits")
    workstation = dataclasses.replace(_job(), kind=job_table.JobKind.TASK)
    assert jd.cost_line(workstation, costs.job_cost(workstation, T0, LYRA), LYRA, 120) is None


def test_cost_line_drops_the_least_useful_parts_first() -> None:
    assert _line(_job(), LYRA, width=58) == "cost    32.0 CPU-h · 4.0 GPU-h · 40/h × normal ×1 = 160 SU"
    assert _line(_job(), LYRA, width=46) == "cost    4.0 GPU-h · 40/h × normal ×1 = 160 SU"
    usage = _job(partition="gpu-preempt", qos="gpu-priority", billing="8")
    assert _line(usage, K2, width=50) == "cost    32.0 CPU-h · 4.0 GPU-h · 32 billing-h"
    assert len(_line(_job(), LYRA, width=30)) <= 30                                    # never wider than the box


def test_usage_only_billing_h_are_never_scaled_by_a_qos_factor() -> None:
    """Review: a usage-only cluster whose QOS has factor 2: 8/h over 4 h is 32 billing-h (not 64), 576 at its 72 h
    limit, in the preview, the Summary line, the Resources line and the agent's job card alike."""
    from runfleet.agent.view import cost_text
    k2x2 = _caps({"normal": 1.0, "double": 2.0}, (PartitionCharge("gpu-preempt"),))
    run = _job(partition="gpu-preempt", qos="double", billing="8")
    pend = _job(JobState.PENDING, partition="gpu-preempt", qos="double", billing="8", elapsed=None)
    assert cost_meta(run, costs.job_cost(run, T0, k2x2), True) == "32 billing-h"
    assert cost_meta(pend, costs.job_cost(pend, T0, k2x2), True) == "≈ 576 billing-h at limit"
    summary = jd.cost_summary(run, costs.job_cost(run, T0, k2x2), k2x2, mode="usage_only")   # no SU limits read
    assert summary is not None and summary.plain == "cost 4.0 GPU-h · 32 billing-h (≈ 576 by limit)"
    line = jd.cost_line(run, costs.job_cost(run, T0, k2x2), k2x2, 200, mode="usage_only")
    assert line is not None and "32 billing-h" in line.plain and "64" not in line.plain
    card = cost_text(run, T0, k2x2, "usage_only")
    assert card is not None and "= 32 billing-h so far" in card and "≈ 576 at limit" in card


# ---------------------------------------------------------------- compact Summary one-liner
def _summary(job: Job, caps: Capabilities | None) -> str | None:
    line = jd.cost_summary(job, costs.job_cost(job, T0, caps), caps)
    return line.plain if line is not None else None


def test_compact_cost_summary() -> None:
    assert _summary(_job(), LYRA) == "cost 4.0 GPU-h · 160 SU (≈ 2,880 by limit)"
    usage = _job(partition="gpu-preempt", qos="gpu-priority", billing="8")
    assert _summary(usage, K2) == "cost 4.0 GPU-h · 32 billing-h (≈ 576 by limit)"
    assert _summary(_job(qos="low", partition="free-gpu"), LYRA) == "cost 4.0 GPU-h · free (qos low ×0)"
    assert _summary(_job(gpus=0, qos="weird"), LYRA) == "cost 32.0 CPU-h · 160 SU ×1? (≈ 2,880 by limit)"
    assert _summary(_job(JobState.PENDING, elapsed=None), LYRA) == "cost ≈ 2,880 SU if it runs to its limit"
    assert _summary(_job(JobState.PENDING, elapsed=None, limit=None), LYRA) is None
    task = dataclasses.replace(_job(), kind=job_table.JobKind.TASK)
    assert _summary(task, LYRA) is None


# ---------------------------------------------------------------- GPU tab Cost box (mockup P)
def _acct(mode: Literal["limits", "usage_only"], balances: tuple[AccountBalance, ...]) -> AccountSummary:
    return AccountSummary(window="30d", read_at=T0, mode=mode, balances=balances)


def _box(job: Job, caps: Capabilities | None, summary: AccountSummary | None = None) -> list[str]:
    lines = jd.cost_box_lines(job, costs.job_cost(job, T0, caps), caps, summary, host="lyra", now=T0, width=118)
    assert all(line.cell_len == 118 for line in lines)
    return [line.plain.rstrip() for line in lines]


def test_cost_box_charged_with_account_balance() -> None:
    summary = _acct(mode="limits", balances=(
        AccountBalance("pilab_02", limit_su=97_885, used_su=298),))
    lines = _box(_job(), LYRA, summary)
    assert lines[0].startswith("elapsed   4:00:00 of 3-00:00:00") and "account  pilab_02" in lines[0] \
        and "partition  gpu" in lines[0]
    assert lines[1] == "cpu       8 × 4.00 h = 32.0 CPU-h"
    assert lines[2] == "gpu       1 × RTX6000 × 4.00 h = 4.0 GPU-h"
    assert lines[3].startswith("billing   40 per hour (Slurm AllocTRES)") and \
        lines[3].endswith("weights of gpu: cpu=1,gres/gpu=32")
    assert lines[4] == "factor    qos normal ×1 (UsageFactor)"
    assert lines[5].startswith("charged   160 SU so far") and \
        lines[5].endswith("≈ 2,880 SU if it runs to its limit (3-00:00:00)")
    assert lines[6] == "account   pilab_02: this job ≈ 3.0% of what is left · 298 / 97,885 SU used"   # verdict first


def test_cost_box_says_exceeds_and_no_su_left_first() -> None:
    """Review: `≈ 1608% of what is left` was muted and cut at 80 columns; an exhausted account said nothing."""
    pending = _job(JobState.PENDING, elapsed=None)                        # ≈ 2,880 SU at its limit
    small = _acct(mode="limits", balances=(AccountBalance("pilab_02", limit_su=97, used_su=0),))
    lines = jd.cost_box_lines(pending, costs.job_cost(pending, T0, LYRA), LYRA, small, host="lyra", now=T0, width=76)
    acct = next(t for t in lines if t.plain.startswith("account "))
    assert acct.plain.startswith("account   pilab_02: this job exceeds what is left (97 SU)")
    assert any(sp.style for sp in acct.spans if "exceeds" in acct.plain[sp.start:sp.end])
    empty = _acct(mode="limits", balances=(AccountBalance("pilab_02", limit_su=97, used_su=97),))
    assert any(line.startswith("account   pilab_02: no SU left · 97 / 97 SU used") for line in _box(_job(), LYRA, empty))


def test_cost_box_account_line_only_when_the_summary_knows_the_account() -> None:
    assert not any(line.startswith("account ") for line in _box(_job(), LYRA))
    other = _acct(mode="limits", balances=(AccountBalance("pilab_03", limit_su=10, used_su=1),))
    assert not any(line.startswith("account ") for line in _box(_job(), LYRA, other))
    usage_only = _acct(mode="usage_only", balances=(AccountBalance("PILAB_02"),))
    lines = _box(_job(), LYRA, usage_only)
    assert lines[-1] == "account   PILAB_02: usage only, no SU limit on lyra"
    assert lines[-2].startswith("usage     160 billing-h so far")             # the summary's mode decides the unit


def test_cost_box_free_pending_ended_and_unknown_factor() -> None:
    free = _box(_job(qos="low", partition="free-gpu", billing="8"), LYRA)
    assert free[3].endswith("weights of free-gpu: none (billing = CPUs)")
    assert free[4] == "factor    qos low ×0 (UsageFactor) · free"
    assert free[5] == "charged   free (qos low ×0) · 32 billing-h not charged"
    pending = _box(_job(JobState.PENDING, billing=None, elapsed=None), LYRA)
    assert pending[0].startswith("elapsed   pending · limit 3-00:00:00")
    assert pending[1] == "cpu       8 (not started)" and pending[2] == "gpu       1 × RTX6000 (not started)"
    assert pending[3].startswith("billing   40 per hour (from the partition weights: estimate)")
    assert pending[5] == "charged   ≈ 2,880 SU if it runs to its limit (3-00:00:00)"
    ended = _box(_job(JobState.COMPLETED), LYRA,
                 _acct(mode="limits", balances=(AccountBalance("pilab_02", 97_885, 298),)))
    assert ended[5] == "charged   160 SU"
    assert ended[6] == "account   pilab_02: 298 / 97,885 SU used · this job 160 SU"
    unknown = _box(_job(qos="weird"), LYRA)
    assert unknown[4] == "factor    qos weird ×1? (UsageFactor unknown: counted ×1)"
    assert jd.cost_box_title(_job(), costs.job_cost(_job(), T0, LYRA), T0).startswith("Cost · as of now")
    assert jd.cost_box_lines(dataclasses.replace(_job(), kind=job_table.JobKind.TASK), None, LYRA, None,
                             host="lyra", now=T0, width=80)[0].plain.startswith("no cost")


@pytest.mark.parametrize("width", [80, 120])
def test_cost_box_lines_never_overflow(width: int) -> None:
    for job in (_job(), _job(JobState.PENDING, elapsed=None), _job(qos="low", partition="free-gpu")):
        lines = jd.cost_box_lines(job, costs.job_cost(job, T0, LYRA), LYRA, None, host="lyra", now=T0,
                                  width=width - 6)
        assert all(line.cell_len == width - 6 for line in lines)


# ---------------------------------------------------------------- screens
K2_JOB = JobKey("k2", "2332861")


async def test_job_detail_cost_accrues_every_second(make_app: Any) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        app.open_job(K2_JOB)
        for _ in range(6):
            await pilot.pause(0.05)
        assert isinstance(app.screen, JobDetailScreen)
        box = app.screen.query_one("#cost-box", Static)
        assert str(box.border_title) == "Cost · as of now (running, accrues every second)"
        assert "33 billing-h so far (= CPU-h here)" in str(box.render())
        assert "cost    33.0 CPU-h · 4.1 GPU-h · 33 billing-h" in str(app.screen.query_one("#res-box").render())
        svc.clock.advance(3600)
        app.screen.refresh_time()                       # the 1 Hz clock: no poll needed for the hours to grow
        await pilot.pause()
        assert "41 billing-h so far" in str(box.render())
        assert "41 billing-h" in str(app.screen.query_one("#res-box").render())


async def test_compact_detail_has_one_cost_line_at_80x24(make_app: Any) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.open_job(K2_JOB)
        for _ in range(6):
            await pilot.pause(0.05)
        compact = str(app.screen.query_one("#summary-compact", Static).render())
        assert "cost 4.1 GPU-h · 33 billing-h (≈ 576 by limit)" in compact
        assert app.screen.query_one("#summary-log").size.height >= 6        # the log keeps its minimum height


async def test_jobs_cost_columns_header_click_and_filter(make_app: Any) -> None:
    app, svc = make_app()
    ui = svc.config.ui
    svc.config = dataclasses.replace(svc.config, ui=dataclasses.replace(ui, jobs_columns=(*ui.jobs_columns, "su")))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = app.screen.query_one(JobsView)
        table = app.screen.query_one(JobTable)
        assert "su" in [k for k, _w in table.cols]
        table.post_message(DataTable.HeaderSelected(table, ColumnKey("su"), 0, Text("SU")))
        await pilot.pause()
        # largest first: the pending jobs' projections (≈ 576, ≈ 384), then 2333760's 72 billing-h so far
        assert view.sort == "su" and table.order[:3] == ["k2:2331790", "vega:724479", "k2:2333760"]
        view._set_query("su:>40 h:k2 st:r")
        await pilot.pause()
        assert table.order == ["k2:2333760"]
        names = [name for name, _cb, _help in RunFleetCommands(app.screen)._commands()]  # type: ignore[arg-type]
        assert "Sort by SU" in names and "Sort by GPU-h" in names
