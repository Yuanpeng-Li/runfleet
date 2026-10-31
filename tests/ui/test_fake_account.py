"""[WP5] FakeService.account_report / account_cached (§5.2.11): the fake world's reports the Account screen tests and
snapshots build on — lyra with limits, reallocation dates and a free and a charged rollup, k2 / vega usage only, a
slurmdbd failure keeping the last report, workstations without scheduler accounting."""

from __future__ import annotations

import inspect

from runfleet.core import service
from runfleet.util.cost import classify_partitions
from runfleet.util.usage import group
from tests.ui.fakes import ACCOUNT_DOWN, FakeService


def test_fake_service_has_every_account_method_of_fleet_api() -> None:
    for name in ("account_report", "account_cached"):
        assert inspect.signature(getattr(FakeService, name)).parameters.keys() == \
            inspect.signature(getattr(service.FleetAPI, name)).parameters.keys()


async def test_lyra_limits_free_and_charged() -> None:
    svc = FakeService()
    assert svc.account_cached("lyra", "30d") is None
    r = await svc.account_report("lyra")
    assert r.window == "30d" and r.summary.mode == "limits" and r.errors == ()
    bal = {b.account: b for b in r.summary.balances}
    assert bal["pilab_02"].limit_su == 97_885 and bal["pilab_02"].available_su == 97_885 - 298.12
    assert bal["alice07"].realloc_known and bal["alice07"].next_realloc is None
    rows = group(r.cells)
    assert rows[0].key == "orb-swp--train--cifar-warmup--scratch" and rows[0].gpu_h == 1190 and rows[0].charged_su == 0
    charged = {x.key: x for x in rows}["lmft-x0--train--lr1e-5--finetune-base"]
    assert round(charged.charged_su) == 116 and len(rows) == 11
    caps = svc.store.hosts["lyra"].snapshot.caps                         # type: ignore[union-attr]
    verdict = {p.partition: p.charged for p in classify_partitions(caps, r.cells)}   # type: ignore[arg-type]
    assert verdict["gpu"] and not verdict["free-gpu"]
    assert svc.account_cached("lyra", "30d") is r and await svc.account_report("lyra") is r   # 10 min cache
    period = await svc.account_report("lyra", "period")
    assert period.window == "period" and all(u.start_inferred for u in period.summary.usage)


async def test_usage_only_hosts_period_and_workstations() -> None:
    svc = FakeService()
    k2 = await svc.account_report("k2", "period")                        # k2 has no period window
    assert k2.window == "30d" and k2.summary.mode == "usage_only" and k2.summary.half_life_s == 7 * 86400
    assert sum(c.jobs for c in k2.cells) == 1212 and len(group(k2.cells)) == 19
    vega = await svc.account_report("vega", "period")
    assert vega.window == "period" and vega.summary.reset_period == "MONTHLY"
    ws = await svc.account_report("labstation", "30d")
    assert ws.summary.balances == () and ws.summary.notes == ("workstation: no scheduler accounting",)


async def test_a_slurmdbd_failure_keeps_the_last_report_with_a_red_line() -> None:
    svc = FakeService()
    good = await svc.account_report("vega", "30d")
    svc.account_faults["vega"] = ACCOUNT_DOWN
    svc.clock.advance(700)
    bad = await svc.account_report("vega", "30d")
    assert bad.cells == good.cells and bad.fetched_at == good.fetched_at
    assert [e.summary for e in bad.errors] == ["slurmdbd unreachable"] and bad.summary.errors == bad.errors
    fresh = FakeService()
    fresh.account_faults["k2"] = ACCOUNT_DOWN
    old = await fresh.account_report("k2", "30d")                        # no last report: one 12 minutes old
    assert (fresh.clock.now() - old.fetched_at).total_seconds() == 12 * 60 and old.errors
