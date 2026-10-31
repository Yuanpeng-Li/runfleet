"""[WP6] Account usage in the demo world (§5.2.11, §10.6): lyra is shaped like the field-probed cluster (SU limits on
the account rows, sbank with the next reallocation, no decay, a charged `gpu` partition next to free ones), k2 and
vega are usage only (k2 decays and refuses wide sacct ranges, vega resets monthly). The renderers print the real
formats, the real parsers read them, and the real service turns them into the periodic summary and the on-demand
report for every window."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from runfleet.backends.slurm import account as acct
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.demo import render
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import HISTORY_GAP_S, DemoWorld, next_annual
from runfleet.models import USAGE_WINDOWS, AccountReport, AccountSummary
from tests.conftest import CLOCK_START, sample
from tests.demo.test_roundtrip import Driver

LYRA_ACCOUNTS = ["alice", "pilab_02", "pilab_02_gpu"]


def _service(world_seed: int = 7) -> tuple[FleetService, DemoWorld]:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=world_seed, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    svc = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
    svc.lock_path = False
    return svc, world


# ---------------------------------------------------------------- renderers -> parsers
def test_sbank_prints_the_real_table(world: DemoWorld) -> None:
    """Byte-exact against the field sample for an account the demo shapes the same (pilab_02_gpu: 0 of 97 SU, next
    reallocation 2027-04-02); `alice` has no reallocation date."""
    c = world.cluster("lyra")
    out = render.acct_sbank(c, world.now(), accounts=LYRA_ACCOUNTS)
    blocks = out.stdout.split("@@acct ")
    field = sample("lyra", "acct_sbank").split("@@acct ")
    assert "@@acct " + blocks[3] == "@@acct " + field[2]                  # pilab_02_gpu
    balances = acct.parse_sbank(out.stdout)
    assert balances["pilab_02"].limit_su == 97885 and balances["pilab_02"].next_realloc == date(2027, 1, 14)
    assert balances["alice"].limit_su == 1000 and balances["alice"].realloc_known
    assert balances["alice"].next_realloc is None and not any(b.error for b in balances.values())


def test_next_reallocation_stays_ahead_of_the_clock() -> None:
    assert next_annual("01-14", CLOCK_START.timestamp(), timedelta(hours=-7)) == "2027-01-14"
    later = datetime(2027, 1, 14, 12, tzinfo=UTC).timestamp()
    assert next_annual("01-14", later, timedelta(hours=-7)) == "2028-01-14"


def test_sshare_account_rows_carry_limits_and_charged_usage(world: DemoWorld) -> None:
    c = world.cluster("lyra")
    pr = acct.parse_share(render.acct_share(c, world.now(), accounts=LYRA_ACCOUNTS).stdout)
    assert not pr.bad_lines
    rows = {(r.account, r.user): r for r in pr.items}
    assert set(rows) == {(a, u) for a in LYRA_ACCOUNTS for u in ("", "alice")}
    assert rows[("pilab_02", "")].grp_tres_mins == {"billing": 5873100.0}
    assert rows[("alice", "")].grp_tres_raw["billing"] == 0          # every own job on `alice` ran free (qos low ×0)
    own = rows[("pilab_02", "alice")].grp_tres_raw
    assert own["billing"] > own["cpu"] > 0 and own["gres/gpu:v100"] > 0  # the charged gpu job: billing cpu + 32/GPU
    assert rows[("pilab_02", "")].grp_tres_raw["billing"] > own["billing"]   # other members charge it too
    k2 = acct.parse_share(render.acct_share(world.cluster("k2"), world.now()).stdout)
    assert not k2.bad_lines and all(not r.grp_tres_mins for r in k2.items)    # usage only: no limits


def test_vega_sshare_since_the_reset_equals_sreport_since_then(world: DemoWorld) -> None:
    """Review: vega never decays, every factor is 1 and billing = CPUs, so (as lyra shows: sshare 158,803 min ≈
    sreport 159,912) the account's sshare usage since the monthly reset is sreport's account total over that span."""
    c = world.cluster("vega")
    now = world.now()
    reset = render.reset_start(c, now)
    assert reset is not None
    share = {r.account: r for r in acct.parse_share(render.acct_share(c, now).stdout).items if not r.user}
    _me, totals = acct.sreport_usage(acct.parse_sreport(render.acct_sreport(
        c, now, accounts=["dept", "lab"], start=reset, totals=True).stdout).items, cluster="vega")
    for total in totals:
        assert share[total.account].grp_tres_raw["billing"] / 60 == pytest.approx(total.billing_h, abs=1.0)


def test_sreport_rows_and_totals(world: DemoWorld) -> None:
    c = world.cluster("lyra")
    now = world.now()
    mine = acct.parse_sreport(render.acct_sreport(c, now, accounts=LYRA_ACCOUNTS, start=now - 30 * 86400,
                                                  totals=False).stdout)
    assert not mine.bad_lines and {r.login for r in mine.items} == {"alice"}
    assert {r.account for r in mine.items} == {"alice", "pilab_02"}            # pilab_02_gpu: no usage, no rows
    me, totals = acct.sreport_usage(acct.parse_sreport(render.acct_sreport(
        c, now, accounts=LYRA_ACCOUNTS, start=now - 30 * 86400, totals=True).stdout).items, cluster="lyra")
    by = {u.account: u for u in totals}
    for u in me:
        assert by[u.account].cpu_h is not None and u.cpu_h is not None and by[u.account].cpu_h >= u.cpu_h
    assert by["pilab_02"].billing_h > next(u for u in me if u.account == "pilab_02").billing_h   # type: ignore[operator]


def test_sacct_chunks_parse_and_agree_with_sreport(world: DemoWorld) -> None:
    """The rollup and sreport read the same allocations: the user's 30-day CPU-h / GPU-h / billing-h agree."""
    for host in ("k2", "vega", "lyra"):
        c = world.cluster(host)
        now = world.now()
        when = datetime.fromtimestamp(now, UTC)
        pr = acct.parse_acct_jobs(render.acct_jobs(c, now, older_days=30, newer_days=0).stdout, tz=c.tz)
        assert not pr.bad_lines and len(pr.items) > 20
        cells, seen = acct.cells(pr.items, None, now=when, start=when - timedelta(days=30))
        assert seen == len(pr.items)
        from_sacct = {u.account: u for u in acct.usage_from_cells(cells)}
        me, _ = acct.sreport_usage(acct.parse_sreport(render.acct_sreport(
            c, now, accounts=[], start=now - 30 * 86400, totals=False).stdout).items)
        for u in me:
            s = from_sacct[u.account]
            assert u.cpu_h == pytest.approx(s.cpu_h, abs=0.5) and u.gpu_h == pytest.approx(s.gpu_h, abs=0.5)
            assert u.billing_h == pytest.approx(s.billing_h, abs=0.5)


def test_history_is_older_than_the_jobs_table_and_follows_the_naming_convention(world: DemoWorld) -> None:
    for host in ("k2", "vega", "lyra"):
        c = world.cluster(host)
        hist = c.history()
        assert hist and all(h.start + h.elapsed_s <= world.t0 - HISTORY_GAP_S for h in hist)
        assert max(h.job_id for h in hist) < min(c.jobs) and len({h.job_id for h in hist}) == len(hist)
        assert all(h.name == "jupyter" or h.name.split("--")[-1].startswith("a") for h in hist)
    again = DemoWorld(seed=7, now=lambda: CLOCK_START, latency_scale=0.0)
    assert again.cluster("k2").history() == DemoWorld(seed=7, now=lambda: CLOCK_START).cluster("k2").history()


def test_k2_refuses_wide_sacct_ranges(world: DemoWorld) -> None:
    c = world.cluster("k2")
    wide = render.acct_jobs(c, world.now(), older_days=60, newer_days=0)
    assert (wide.stdout, wide.rc, wide.stderr) == ("", 0, "sacct: error: Too wide of a date range in query")
    assert render.acct_jobs(c, world.now(), older_days=45, newer_days=0).stdout
    assert render.acct_jobs(world.cluster("vega"), world.now(), older_days=90, newer_days=0).stdout


def test_lyra_charging_rules_reach_the_jobs(world: DemoWorld, clock: Any) -> None:
    """squeue/sacct billing follows the partition weights: the charged V100 job bills 8 + 32 = 40 per hour, the free
    jobs their CPUs."""
    c = world.cluster("lyra")
    job = c.jobs[32371502]
    row = render.sacct_ids(c, world.now(), ids_raw=["32371502"], fields=["JobIDRaw", "AllocTRES"], sep="|").stdout
    assert job.partition == "gpu" and "billing=40," in row
    row = render.sacct_ids(c, world.now(), ids_raw=["32303059"], fields=["JobIDRaw", "AllocTRES"], sep="|").stdout
    assert "billing=8," in row


# ---------------------------------------------------------------- detection and the periodic summary
async def test_detect_finds_the_accounting_tools(make_world: Any) -> None:
    world, clock = make_world()
    caps = await Driver(world, clock, "lyra").run_detect()
    assert {"sreport", "sbank"} <= caps.tools and caps.slurm_config["PriorityDecayHalfLife"] == "00:00:00"
    world, clock = make_world()
    caps = await Driver(world, clock, "k2").run_detect()
    assert "sreport" in caps.tools and "sbank" not in caps.tools


async def test_periodic_summary_lands_on_the_snapshot() -> None:
    svc, _world = _service()
    await svc.collect_once(["k2", "vega", "lyra", "labstation"])
    snaps = {h: svc.store.hosts[h].snapshot for h in ("k2", "vega", "lyra", "labstation")}
    assert snaps["labstation"] is not None and snaps["labstation"].account is None
    summaries: dict[str, AccountSummary] = {}
    for host in ("k2", "vega", "lyra"):
        snap = snaps[host]
        assert snap is not None and snap.account is not None, host
        summaries[host] = snap.account
        assert not snap.account.errors and snap.account.window == "30d" and snap.account.usage
    lyra = summaries["lyra"]
    assert lyra.mode == "limits" and lyra.half_life_s == 0 and lyra.sbank_at is not None
    bal = {b.account: b for b in lyra.balances}
    assert (bal["pilab_02"].limit_su, bal["pilab_02_gpu"].limit_su, bal["alice"].limit_su) == (97885, 97, 1000)
    assert bal["pilab_02"].used_basis == "since_reset" and 100 < (bal["pilab_02"].used_su or 0) < 1000
    assert bal["pilab_02"].next_realloc == date(2027, 1, 14) and bal["pilab_02_gpu"].next_realloc == date(2027, 4, 2)
    assert bal["alice"].realloc_known and bal["alice"].next_realloc is None and bal["alice"].used_su == 0
    k2 = summaries["k2"]
    assert k2.mode == "usage_only" and k2.half_life_s == 14 * 86400 and k2.balances[0].used_basis == "decayed"
    assert k2.sbank_at is None and k2.usage[0].gpu_h and k2.usage[0].gpu_h > 1000
    vega = summaries["vega"]
    assert vega.mode == "usage_only" and vega.reset_period == "MONTHLY"
    assert [b.account for b in vega.balances] == ["dept", "lab"] and vega.balances[1].used_basis == "since_reset"


# ---------------------------------------------------------------- the on-demand report
async def test_account_report_for_every_window_on_every_cluster() -> None:
    svc, world = _service()
    world.clear("lynx", "auth")                                           # logged in (it starts NEEDS_AUTH)
    await svc.collect_once(["k2", "vega", "lyra", "labstation", "lynx"])
    reports: dict[tuple[str, str], AccountReport] = {}
    for host in ("k2", "vega", "lyra"):
        for window in USAGE_WINDOWS:
            r = reports[(host, window)] = await svc.account_report(host, window)
            assert not r.errors and not r.truncated, (host, window, r.errors)
            assert r.cells and r.jobs_seen and r.summary.balances and r.account_totals, (host, window)
            assert r.window == ("30d" if (host, window) == ("k2", "period") else window)
    for host in ("k2", "vega", "lyra"):                                   # wider windows see more
        assert reports[(host, "7d")].jobs_seen < reports[(host, "30d")].jobs_seen < reports[(host, "90d")].jobs_seen
    k2 = reports[("k2", "90d")]
    assert any(c.running for c in k2.cells) and all(c.charged_su == c.billing_h for c in k2.cells)  # factor 1
    lyra = reports[("lyra", "30d")]
    charged = [c for c in lyra.cells if c.charged_su]
    assert any(c.partition == "gpu" and c.gpu_model == "v100" for c in charged)
    assert all(c.charged_su == 0 for c in lyra.cells if c.qos == "low") and any(c.qos == "low" for c in lyra.cells)
    period = reports[("lyra", "period")]
    starts = {u.account: u.start for u in period.summary.usage}
    assert starts["pilab_02"] is not None and starts["pilab_02"].date() == date(2026, 1, 14)
    assert all(u.start_inferred for u in period.summary.usage) and "alice" not in starts   # no reallocation date
    vega = reports[("vega", "period")]
    assert vega.window_start is not None and vega.window_start.day == 1                     # the monthly reset
    for host in ("labstation", "lynx"):
        r = await svc.account_report(host, "30d")
        assert not r.cells and not r.errors and r.summary.notes == ("workstation: no scheduler accounting",)


async def test_slurmdbd_down_fails_the_dbd_sections_only() -> None:
    svc, world = _service()
    await svc.collect_once(["lyra"])
    world.inject("lyra", "slurmdbd_down")
    r = await svc.account_report("lyra", "30d", max_age_s=0)
    assert r.errors and all(e.summary == "slurmdbd unreachable" for e in r.errors)
    assert {b.account for b in r.summary.balances} == set(LYRA_ACCOUNTS)       # sshare asks slurmctld: still there
