"""The Account screen (`$`, §7.17, mockups O / O2) and the cluster host detail's `account` line: the pure renderers
(sorting, error wording, partitions, layout plan, box frame, the host-detail line) and the screen over FakeService —
`$` from anywhere, window switching with period skipped where no cluster has one (and remembered), `u` / `z` / `o`,
`enter` into the Jobs view, `r` at most once a minute, loading and slurmdbd failures (never zeros), scope changes,
workstations, 80×24."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from rich.text import Text
from textual.widgets import Static

from runfleet.models import AccountBalance, AccountSummary, AccountUse, ErrorKind, HostError
from runfleet.ui.app import RunFleetApp
from runfleet.ui.commands import RunFleetCommands
from runfleet.ui.screens import account as acs
from runfleet.ui.screens.account import AccountScreen, BoxData, Placement, ViewOpts
from runfleet.ui.screens.host_detail import HostDetailScreen, capability_lines
from runfleet.ui.screens.main import MainScreen
from runfleet.util.usage import RollupRow
from tests.ui.fakes import ACCOUNT_DOWN, T0, FakeService, account_report
from tests.ui.test_cost_ui import K2, LYRA

Factory = Callable[..., tuple[RunFleetApp, FakeService]]


def _row(key: str, gpu: float, su: float, cpu: float = 1.0, jobs: int = 1, billing: float = 0.0) -> RollupRow:
    return RollupRow(key=key, jobs=jobs, running=0, cpu_h=cpu, gpu_h=gpu, gpu_h_by_model={}, billing_h=billing,
                     charged_su=su)


def _text(app: RunFleetApp) -> str:
    return str(app.screen.query_one("#acct-boxes", Static).render())


async def _open(app: RunFleetApp, pilot: Any, *, scope: str | None = None) -> AccountScreen:
    await pilot.pause(0.1)
    if scope is not None:
        app.set_scope(scope)
        await pilot.pause(0.1)
    await pilot.press("dollar_sign")
    await pilot.pause(0.3)
    screen = app.screen
    assert isinstance(screen, AccountScreen)
    return screen


# ---------------------------------------------------------------- pure helpers
def test_rollup_sorts_and_their_arrows() -> None:
    rows = [_row("b", 5, 10, cpu=50, jobs=3, billing=10), _row("a", 9, 0, cpu=20, jobs=9, billing=40),
            _row("c", 1, 99, cpu=10, jobs=1, billing=99)]
    assert [r.key for r in acs.sort_rollup(rows, "gpu_h")] == ["a", "b", "c"]
    assert [r.key for r in acs.sort_rollup(rows, "su")] == ["c", "b", "a"]
    assert [r.key for r in acs.sort_rollup(rows, "su", usage_only=True)] == ["c", "a", "b"]   # billing-h
    assert [r.key for r in acs.sort_rollup(rows, "cpu_h")] == ["b", "a", "c"]
    assert [r.key for r in acs.sort_rollup(rows, "jobs")] == ["a", "b", "c"]
    assert [r.key for r in acs.sort_rollup(rows, "name", reverse=True)] == ["c", "b", "a"]
    assert acs.sort_label("gpu_h", False) == "GPU-h ▼" and acs.sort_label("name", False) == "name ▲"
    assert acs.sort_label("su", True, usage_only=True) == "BILL-h ▲"


def test_errors_are_worded_once_per_kind_never_raw_stderr() -> None:
    now = T0
    errs = [dataclasses.replace(ACCOUNT_DOWN, at=now - timedelta(minutes=12)),
            HostError(ErrorKind.REMOTE, "sacct refused the date range", "sacct: error: Too wide", "slurm.acct_jobs.c0"),
            HostError(ErrorKind.REMOTE, "sacct refused the date range", "sacct: error: Too wide", "slurm.acct_jobs.c1")]
    lines = [t.plain.rstrip() for t in acs.error_lines(errs, now, 100)]
    assert lines == ["✖ sreport: slurmdbd unreachable · 12m ago · r retries",
                     "✖ sacct refused the date range · 12m ago · r retries"]
    assert all("Problem talking" not in x for x in lines)


def test_partition_lines_charged_free_and_usage_only() -> None:
    lyra = dataclasses.replace(LYRA, associations=(("lyra", "pilab_02", "", "normal"),))
    wide = [t.plain.rstrip() for t in acs.partition_lines(lyra, (), 110, usage_only=False)]
    assert wide == ["charged  gpu  cpu=1 · gpu=32 per hour  (qos normal ×1)",
                    "free     free-gpu  (AllowQos low ×0, guest ×0: billing-h are recorded but not charged)"]
    (narrow,) = acs.partition_lines(LYRA, (), 76, usage_only=False, narrow=True)
    assert narrow.plain.rstrip() == "charged gpu · free free-gpu (qos low ×0, guest ×0)"
    (k2,) = acs.partition_lines(K2, (), 100, usage_only=True)
    assert k2.plain.rstrip() == "partitions  gpu-preempt (qos ×1 everywhere)"
    assert acs.partition_lines(None, (), 100, usage_only=True) == []
    assert acs.weights_text("cpu=1,gres/gpu:L40S=32") == "cpu=1 · gpu:L40S=32"


def _summary(**kw: Any) -> AccountSummary:
    base = dict(window="30d", read_at=T0, mode="limits",
                balances=(AccountBalance("pilab_02", 97_885, 298, "since_reset", "sshare", "sshare"),
                          AccountBalance("pilab_02_gpu", 97, 90, "since_reset", "sshare", "sshare"),
                          AccountBalance("alice", 1_000, 0, "since_reset", "sshare", "sshare")),
                usage=(AccountUse("alice", "me", 27_554, 3_256, 27_554), AccountUse("pilab_02", "me", 205, 2.9, 298)))
    base.update(kw)
    return AccountSummary(**base)    # type: ignore[arg-type]


def test_host_detail_account_line() -> None:
    line = acs.account_summary_line(_summary(), 120)
    assert line.plain == ("account  pilab_02 298/97,885 SU · pilab_02_gpu 90/97 · alice 0/1,000 · "
                          "30d 27,759 CPU-h 3,259 GPU-h · $ more")
    warn = next(s for s in line.spans if line.plain[s.start:s.end].startswith("pilab_02_gpu"))
    assert warn.style                                                    # 93 %: warning-coloured
    usage = _summary(mode="usage_only", half_life_s=7 * 86400.0,
                     balances=(AccountBalance("lab", None, 2_503_304, "decayed", None, "sshare", my_used_su=5_859),),
                     usage=(AccountUse("lab", "me", 171_881, 15_434, 171_881),))
    assert acs.account_summary_line(usage, 120).plain == (
        "account  usage only (no SU limits) · 30d 171,881 CPU-h · 15,434 GPU-h · recent 5,859 billing-h (decays) "
        "· $ more")
    assert acs.account_summary_line(None, 120).plain == "account  not read yet (every 30 min; $ reads it now)"
    many = _summary(balances=tuple(AccountBalance(f"a{i}", 10, 1) for i in range(6)), usage=())
    assert "a3 1.0/10 …+2" in acs.account_summary_line(many, 120).plain


def test_plan_layout_focus_first_then_collapse_then_fold() -> None:
    def h(base: int, total: int) -> dict[int, int]:
        return {r: base + r - (1 if total and r >= total else 0) for r in range(min(total, 12) + 1)}
    heights, totals = [h(10, 11), h(8, 19), h(8, 2)], [11, 19, 2]
    plan = acs.plan_layout(heights, totals, 32, focus=0)
    assert plan[0].mode == "full" and plan[1].mode == "full" and plan[2] == Placement("compact")
    assert plan[0].rows + plan[1].rows == 32 - 4 - 10 - 8 and plan[1].rows == 3   # the rows left: focused first
    tight = acs.plan_layout(heights, totals, 15, focus=2)
    assert tight[2] == Placement("full", 2) and tight[0] == Placement("compact") and tight[1].mode == "fold"
    assert acs.plan_layout(heights[:1], totals[:1], 5, focus=0)[0].mode == "full"    # never folds the focused box


def test_layout_rank_puts_the_limits_cluster_right_after_the_focused_box() -> None:
    """Review: in config order (k2, vega, lyra) the only cluster with SU limits was collapsed or folded while the
    usage-only ones stayed full. Limits clusters get room right after the focused box, the most used first; the
    boxes are still drawn in config order."""
    datas = [BoxData(h, LYRA if h == "lyra" else K2, r, r.summary, "30d")
             for h in ("k2", "vega", "lyra") if (r := account_report(h, "30d", T0)) is not None]
    assert acs.layout_rank(datas) == [2, 0, 1]
    def h(base: int, total: int) -> dict[int, int]:
        return {r: base + r - (1 if total and r >= total else 0) for r in range(min(total, 12) + 1)}
    heights, totals = [h(10, 11), h(8, 19), h(8, 2)], [11, 19, 2]
    plan = acs.plan_layout(heights, totals, 17, focus=0, rank=acs.layout_rank(datas))
    assert plan[0].mode == "full" and plan[2] == Placement("compact") and plan[1].mode == "fold"
    assert acs.fold_text(datas[2]) == "lyra: pilab_02 97,587 SU left"           # the most used (0.3 %)
    tight = dataclasses.replace(datas[2], summary=dataclasses.replace(datas[2].summary, balances=(  # type: ignore[arg-type]
        AccountBalance("pilab_02_gpu", 97, 96, "since_reset", "sshare", "sshare"),)))
    assert acs.fold_text(tight) == "lyra: pilab_02_gpu 1.0 SU left"
    (box,) = acs.compact_sections(tight, ViewOpts(narrow=False), T0, 100)
    used = box.lines[0]
    assert any(sp.style for sp in used.spans if used.plain[sp.start:sp.end].startswith("pilab_02_gpu"))  # ≥ 95 %
    assert "usage" in box.title.plain                                       # the wide title names the second age
    assert box.lines[1].plain.startswith("  top SU: ")                     # a limits cluster: what spent SU


def test_my_recent_is_my_row_and_u_shows_the_accounts() -> None:
    r = account_report("k2", "30d", T0)
    assert r is not None
    d = BoxData("k2", K2, r, r.summary, "30d")
    mine = "\n".join(t.plain for t in acs.account_lines(d, ViewOpts(), 120, None))
    acct = "\n".join(t.plain for t in acs.account_lines(d, ViewOpts(scope="account"), 120, None))
    assert "MY RECENT" in mine and "5,859 billing-h" in mine and "2,503,304" not in mine
    assert "ACCT RECENT" in acct and "2,503,304 billing-h" in acct


def test_draw_frames_sections_to_the_width() -> None:
    secs = [acs.Section(Text("lyra"), [Text("  a")]), acs.Section(Text("Where"), [Text("  b")], Text("z group"))]
    lines = [t.plain for t in acs.draw(secs, 40, focused=True)]
    assert all(len(x) == 40 for x in lines)
    assert lines[0].startswith("┌ lyra ─") and lines[0].endswith("┐")
    assert lines[2].startswith("├ Where ─") and lines[2].endswith(" z group ─┤") and lines[-1].startswith("└")


def test_box_never_shows_a_failed_fetch_as_zero() -> None:
    failed = account_report("vega", "30d", T0)
    assert failed is not None
    empty = dataclasses.replace(failed, summary=AccountSummary("30d", T0, "usage_only"), cells=(), jobs_seen=0,
                                errors=(HostError(ErrorKind.SSH, "vega: not connected yet"),))
    d = BoxData("vega", None, empty, empty.summary, "30d")
    text = "\n".join(t.plain for s in acs.box_sections(d, ViewOpts(), T0, 100, rows=3, cursor=0, offset=0)
                     for t in s.lines)
    assert "not connected yet" in text and " 0 " not in text and "billing-h" not in text


def test_period_meaning_and_since() -> None:
    lyra = account_report("lyra", "period", T0)
    vega = account_report("vega", "period", T0)
    assert acs.period_text(lyra, 365 * 86400) == "since 2026-01-14, inferred: next reallocation − 365d"
    assert acs.period_text(vega, 365 * 86400) == "since the monthly reset"
    assert acs.period_text(account_report("k2", "period", T0), 365 * 86400) is None     # answered as 30d
    assert acs.window_since("30d", T0) == f"(since {(T0 - timedelta(days=30)).strftime('%m-%d')})"


# ---------------------------------------------------------------- screen
async def test_dollar_opens_it_from_anywhere_and_does_nothing_on_it(make_app: Factory) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        screen = await _open(app, pilot)
        assert screen.hosts == ["k2", "vega", "lyra"] and screen.others == ["labstation", "lynx"]
        assert sorted(c[1][0] for c in svc.calls if c[0] == "account_report") == ["k2", "lyra", "vega"]
        assert all(c[1][1:] == ("30d", 600.0) for c in svc.calls if c[0] == "account_report")
        await pilot.press("dollar_sign")
        await pilot.pause(0.1)
        assert app.screen is screen and len([s for s in app.screen_stack if isinstance(s, AccountScreen)]) == 1
        text = _text(app)
        assert "pilab_02" in text and "campus01" in text and "labstation, lynx: workstations" in text
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert isinstance(app.screen, MainScreen)


async def test_from_a_cluster_host_detail_that_cluster_comes_first(make_app: Factory) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.1)
        app.action_open_host("lyra")
        await pilot.pause(0.2)
        assert isinstance(app.screen, HostDetailScreen)
        await pilot.press("dollar_sign")
        await pilot.pause(0.3)
        screen = app.screen
        assert isinstance(screen, AccountScreen) and screen.hosts[0] == "lyra" and screen.focus_idx == 0
        assert screen.current_host() == "lyra"


async def test_window_switching_skips_period_where_none_and_is_remembered(make_app: Factory) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        screen = await _open(app, pilot, scope="k2")                      # k2: no period window
        assert screen.available_windows() == ["7d", "30d", "90d"]
        await pilot.press("right_square_bracket")
        await pilot.press("right_square_bracket")
        await pilot.pause(0.3)
        assert screen.window == "7d" and app.ui_state["account_window"] == "7d"    # 90d, then round to 7d
        assert ("account_report", ("k2", "90d", 600.0)) in svc.calls
        assert ("account_report", ("k2", "7d", 600.0)) in svc.calls
        app.set_scope(None)                                              # lyra / vega have one
        await pilot.pause(0.3)
        assert screen.available_windows()[-1] == "period"
        await pilot.press("left_square_bracket")
        await pilot.pause(0.3)
        assert screen.window == "period"
        # lyra (SU limits) has the focus, so its period is in the header; k2 says it has none
        assert screen.focus_idx == 2 and "(since 2026-01-14, inferred: next reallocation − 365d)" in str(
            screen.query_one(acs.AccountHeader).render())
        assert "no period window on k2" in _text(app)
        await pilot.press("tab")                                         # k2 focused: no inferred period
        await pilot.pause(0.1)
        header = str(screen.query_one(acs.AccountHeader).render())
        assert "◀ period ▶" in header and "inferred: next reallocation − 365d" not in header
    app2, _ = make_app()
    app2.ui_state["account_window"] = "90d"
    async with app2.run_test(size=(120, 35)) as pilot:
        assert (await _open(app2, pilot)).window == "90d"


async def test_u_z_o_and_enter_into_the_jobs_view(make_app: Factory) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        screen = await _open(app, pilot, scope="lyra")
        assert "MY GPU-h" in _text(app)
        await pilot.press("u")
        await pilot.pause(0.1)
        assert "ACCT GPU-h" in _text(app) and "1,316" in _text(app)      # pilab_02's whole-account CPU-h
        await pilot.press("z")
        await pilot.press("o")
        await pilot.pause(0.1)
        assert screen.opts.group == "task" and screen.opts.sort == "su" and "by task" in _text(app)
        await pilot.press("O")
        assert screen.opts.reverse
        await pilot.press("z", "z")                                      # → name → account
        await pilot.press("G")                                           # last row: a rollup row (by account)
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)
        main = app.screen
        assert isinstance(main, MainScreen) and main.view == "jobs"
        assert main.jobs.query_text.startswith("host:lyra acct:") and main.jobs.preset == "all"


async def test_enter_on_an_experiment_filters_by_its_name(make_app: Factory) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot, scope="lyra")
        await pilot.press("j", "j", "j")                                 # past the 3 accounts: the top experiment
        await pilot.press("enter")
        await pilot.pause(0.3)
        main = app.screen
        assert isinstance(main, MainScreen)
        assert main.jobs.query_text == "host:lyra orb-swp--train--cifar-warmup--scratch"


async def test_r_rereads_the_focused_cluster_at_most_once_a_minute(make_app: Factory) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot, scope="vega")
        svc.calls.clear()
        await pilot.press("r")
        await pilot.pause(0.2)
        assert [c for c in svc.calls if c[0] == "account_report"] == []          # fetched 0 s ago: a toast
        svc.clock.advance(61)
        await pilot.press("r")
        await pilot.pause(0.2)
        assert ("account_report", ("vega", "30d", 0)) in svc.calls
        svc.calls.clear()
        await pilot.press("ctrl+r")
        await pilot.pause(0.2)
        assert [c for c in svc.calls if c[0] == "account_report"] == []


async def test_a_cluster_that_connects_while_the_screen_is_open_gets_its_report(make_app: Factory) -> None:
    """Review: opened before vega's first probe, vega is not a cluster yet; once detection makes it one, its report
    is fetched (once), not left unread until `r`."""
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.1)
        snap = app.store.hosts["vega"].snapshot
        assert snap is not None
        app.store.replace_snapshot("vega", caps=None, partitions=(), nodes=())
        screen = await _open(app, pilot)
        assert "vega" not in screen.hosts and "vega" in screen.others
        svc.calls.clear()
        app.store.replace_snapshot("vega", caps=snap.caps, partitions=snap.partitions, nodes=snap.nodes)
        app.store.set_status("vega", app.store.hosts["vega"].status)
        await pilot.pause(0.3)
        assert "vega" in screen.hosts
        assert [c for c in svc.calls if c[0] == "account_report"] == [("account_report", ("vega", "30d", 600.0))]
        app.store.set_status("vega", app.store.hosts["vega"].status)              # a later status: no refetch
        await pilot.pause(0.3)
        assert len([c for c in svc.calls if c[0] == "account_report"]) == 1


async def test_loading_then_a_slurmdbd_failure_keeps_the_last_data_with_a_red_line(make_app: Factory) -> None:
    app, svc = make_app()
    svc.account_gate = asyncio.Event()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot, scope="vega")
        assert "fetching…" in _text(app) and "reading account usage …" in _text(app)
        svc.account_gate.set()
        await pilot.pause(0.3)
        assert "fetching…" not in _text(app) and "3,410" in _text(app)
        svc.account_faults["vega"] = ACCOUNT_DOWN
        svc.clock.advance(700)
        await pilot.press("r")
        await pilot.pause(0.3)
        text = _text(app)
        assert "✖ sreport: slurmdbd unreachable · 12m ago · r retries" in text and "3,410" in text
        assert "Problem talking" not in text


async def test_a_raising_fetch_is_an_error_line_not_zero_usage(make_app: Factory) -> None:
    app, svc = make_app()

    async def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")
    svc.account_report = broken                                          # type: ignore[method-assign]
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot, scope="k2")
        await pilot.pause(0.2)
        text = _text(app)
        assert "account report failed: boom" in text and "171,881" not in text


async def test_off_in_config_and_the_periodic_summary_before_a_report(make_app: Factory) -> None:
    app, svc = make_app()
    k2 = svc.store.hosts["k2"]
    k2.config = dataclasses.replace(k2.config, slurm=dataclasses.replace(k2.config.slurm, account_usage=False))
    vega = svc.store.hosts["vega"]
    assert vega.snapshot is not None
    vega.snapshot = dataclasses.replace(vega.snapshot, account=AccountSummary(
        "30d", T0, "usage_only", balances=(AccountBalance("pilab", None, 1_234, "decayed", None, "sshare"),),
        usage=(AccountUse("pilab", "me", 3_000, 700, 3_000),), usage_at=T0, half_life_s=14 * 86400.0))
    svc.account_gate = asyncio.Event()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot)
        text = _text(app)
        assert "account usage is off for k2 in config" in text
        assert ("account_report", ("k2", "30d", 600.0)) not in svc.calls
        assert "summary 0s ago" in text and "3,000" in text                 # vega: the periodic summary at once
        svc.account_gate.set()


async def test_80x24_fits_and_collapses(make_app: Factory) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _open(app, pilot)
        lines = _text(app).splitlines()
        assert all(len(x) <= 80 for x in lines) and len(lines) <= 22
        # the screen opens on the cluster with SU limits: lyra is full, the usage-only ones are collapsed
        assert screen.focus_idx == 2 and "USED SU" in _text(app) and "lyra:" not in lines[-1]
        assert "no accounting" in lines[-1]
        await pilot.press("tab")                                         # k2 focused: it opens, lyra still in view
        await pilot.pause(0.1)
        assert screen.focus_idx == 0 and "RECENT" in _text(app) and "pilab_02" in _text(app)


async def test_opens_on_the_cluster_with_su_limits_until_the_user_moves(make_app: Factory) -> None:
    """Review: config order put k2 first, so lyra (the only cluster that can run out of SU) was collapsed at 80x24."""
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        screen = await _open(app, pilot)
        assert screen.hosts[screen.focus_idx] == "lyra"
        await pilot.press("tab")
        await pilot.pause(0.1)
        screen.paint()                                                   # new data never takes the focus back
        assert screen.hosts[screen.focus_idx] == "k2"


async def test_only_the_focused_box_draws_its_cursor(make_app: Factory) -> None:
    """Review: every full box drew `▶`, so j/k/enter's target was only told by the border colour (none in NO_COLOR)."""
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await _open(app, pilot)
        assert _text(app).count("│▶") == 1
        await pilot.press("tab")
        await pilot.pause(0.1)
        assert _text(app).count("│▶") == 1


async def test_j_past_the_last_row_moves_to_the_next_box(make_app: Factory) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        screen = await _open(app, pilot)
        await pilot.press("tab")                                         # from lyra (SU limits) round to k2
        await pilot.press("G")
        await pilot.press("j")
        await pilot.pause(0.1)
        assert screen.focus_idx == 1 and screen.box_cursor["vega"] == 0
        await pilot.press("k")
        await pilot.pause(0.1)
        assert screen.focus_idx == 0


async def test_palette_entries_and_scope_change(make_app: Factory) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(0.1)
        names = [c[0] for c in RunFleetCommands(app.screen)._commands()]
        assert "Account usage" in names and "Account: lyra" in names and "Account: labstation" not in names
        app.action_account("vega")
        await pilot.pause(0.3)
        screen = app.screen
        assert isinstance(screen, AccountScreen) and screen.hosts[0] == "vega"
        app.action_account("lyra")                                         # on it: focus that cluster
        await pilot.pause(0.1)
        assert screen.current_host() == "lyra"
        svc.calls.clear()
        await pilot.press("2")                                             # scope vega
        await pilot.pause(0.3)
        assert screen.hosts == ["vega"] and ("account_report", ("vega", "30d", 600.0)) in svc.calls


async def test_host_detail_shows_the_account_line(make_app: Factory) -> None:
    app, svc = make_app()
    lyra = svc.store.hosts["lyra"]
    assert lyra.snapshot is not None
    summary = dataclasses.replace(_summary(), balances=(
        AccountBalance("pilab_02", 97_885, 298, "since_reset", "sshare", "sshare", next_realloc=date(2027, 1, 14),
                       realloc_known=True),))
    lyra.snapshot = dataclasses.replace(lyra.snapshot, account=summary)
    lines = [t.plain for t in capability_lines(lyra, 116)]
    assert any(x.startswith("account  pilab_02 298/97,885 SU") for x in lines)
    k2 = svc.store.hosts["k2"]
    assert any(t.plain.startswith("account  not read yet") for t in capability_lines(k2, 116))
    k2.config = dataclasses.replace(k2.config, slurm=dataclasses.replace(k2.config.slurm, account_usage=False))
    assert not any(t.plain.startswith("account  ") for t in capability_lines(k2, 116))
    vega = svc.store.hosts["vega"]                                          # no slurmdbd known: nothing
    assert not any(t.plain.startswith("account  ") for t in capability_lines(vega, 116))
