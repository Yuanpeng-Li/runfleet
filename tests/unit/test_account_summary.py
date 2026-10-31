"""[WP4] The periodic account summary in core (§5.2.11, §3.6, §3.7, §5.4, §5.5): `merge_outputs` and `FleetStore`
keep the last non-None `account` and say so in `ChangeSet.account`; `notify_account`; the `slurm.account` group under
the real HostPoller (load budget: piggyback, no extra probe; hidden ×2; backoff on a failing sreport; the controller
slowdown); the config keys; warm start and the `snapshot` merge."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import timedelta

import pytest

from runfleet.backends.base import CollectorOutput, MonitorView
from runfleet.backends.slurm.collector import SLURMDBD, SlurmCollector
from runfleet.config import GeneralSettings, HostConfig, SlurmSettings, format_duration, parse_config
from runfleet.core.clock import ManualClock
from runfleet.core.poller import BackendHooks, HostLanes, HostPoller, merge_outputs
from runfleet.core.service import merged_account
from runfleet.core.store import ChangeSet, FleetStore
from runfleet.errors import ConfigError
from runfleet.models import (AccountBalance, AccountSummary, AccountUse, ErrorKind, HostError, HostSnapshot,
                             Visibility, from_json, to_json)
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import StubProcfs, run_for, settle, stub_build_capabilities, stub_caps, stub_detect_sections

T0 = CLOCK_START


def _summary(used: float = 298.0, **kw: object) -> AccountSummary:
    bal = AccountBalance("pilab_02", limit_su=97_885.0, used_su=used, used_basis="since_reset",
                         limit_source="sshare", used_source="sshare", read_at=T0)
    use = AccountUse("pilab_02", "me", cpu_h=298.1, gpu_h=0.0, billing_h=298.1, start=T0 - timedelta(30))
    return replace(AccountSummary(window="30d", read_at=T0, mode="limits", balances=(bal,), usage=(use,),
                                  usage_at=T0), **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- merge rule and store (§3.6, §3.7)
def test_merge_outputs_keeps_the_last_non_none_account() -> None:
    s = _summary()
    assert merge_outputs([CollectorOutput(account=s), CollectorOutput()]).account == s
    assert merge_outputs([CollectorOutput(), CollectorOutput()]).account is None
    newer = _summary(300.0)
    assert merge_outputs([CollectorOutput(account=s), CollectorOutput(account=newer)]).account == newer


def test_store_apply_keeps_the_summary_and_flags_changes_only() -> None:
    store = FleetStore([HostConfig("lyra", "lyra")])
    seen: list[ChangeSet] = []
    store.subscribe(seen.append)
    change = store.apply("lyra", CollectorOutput(account=_summary()), T0)
    assert change.account and seen == [change]
    snap = store.hosts["lyra"].snapshot
    assert snap is not None and snap.account == _summary()
    again = store.apply("lyra", CollectorOutput(account=_summary()), T0 + timedelta(seconds=15))
    assert not again.account and len(seen) == 1                         # equal summary: no repaint
    store.apply("lyra", CollectorOutput(), T0 + timedelta(seconds=30))  # None = unchanged
    assert store.hosts["lyra"].snapshot.account == _summary()           # type: ignore[union-attr]
    assert store.apply("lyra", CollectorOutput(account=_summary(310.0)), T0).account


def test_notify_account_and_warm_start() -> None:
    store = FleetStore([HostConfig("lyra", "lyra")])
    seen: list[ChangeSet] = []
    store.subscribe(seen.append)
    assert store.notify_account("lyra") == ChangeSet("lyra", account=True)
    assert store.notify_account("gone") == ChangeSet("gone")              # unknown host: no-op
    snap = HostSnapshot(host="lyra", taken_at=T0, caps=None, jobs={}, gpus=(), procs=(), nodes=(), partitions=(),
                        metrics=None, account=_summary())
    cached = from_json(HostSnapshot, to_json(snap))                       # the cache's snapshots row
    assert store.set_snapshot("lyra", cached).account
    assert seen[0] == ChangeSet("lyra", account=True) and len(seen) == 2


def test_snapshot_merge_takes_what_a_failing_section_could_not_read() -> None:
    old = _summary()
    err = HostError(ErrorKind.REMOTE, "slurmctld unreachable", section="slurm.acct_share", at=T0)
    unread = replace(old.balances[0], used_su=None, limit_su=None, read_at=None, limit_source=None, used_source=None)
    new = _summary(read_at=T0 + timedelta(hours=1), balances=(unread,), mode="usage_only", errors=(err,),
                   usage_at=T0 + timedelta(hours=1))
    merged = merged_account(new, old)
    assert merged is not None and merged.balances == old.balances and merged.read_at == T0
    assert merged.mode == "limits" and merged.errors == (err,) and merged.usage_at == T0 + timedelta(hours=1)
    assert merged_account(None, old) is old and merged_account(old, None) is old
    assert merged_account(_summary(300.0), old) == _summary(300.0)          # a good reading wins
    # review: the cached usage of another window (general.usage_window changed) is never shown under the new label
    sre = HostError(ErrorKind.REMOTE, "slurmdbd down", section="slurm.acct_sreport", at=T0)
    same = merged_account(replace(new, usage=(), usage_at=None, errors=(err, sre)), old)
    assert same is not None and same.usage == old.usage and same.usage_at == T0
    other = merged_account(replace(new, window="7d", usage=(), usage_at=None, errors=(err, sre)), old)
    assert other is not None and other.usage == () and other.usage_at is None and other.balances == old.balances


def test_store_apply_keeps_the_cached_balances_when_the_first_tick_after_a_restart_fails() -> None:
    """Review: a restarted TUI whose first `slurm.account` tick has sshare failing must not replace the warm-start
    balances with unknown ones stamped "now" (mode usage_only) and persist them: the store merges like `snapshot`."""
    store = FleetStore([HostConfig("lyra", "lyra")])
    snap = HostSnapshot(host="lyra", taken_at=T0, caps=None, jobs={}, gpus=(), procs=(), nodes=(), partitions=(),
                        metrics=None, account=_summary())
    store.set_snapshot("lyra", snap)
    err = HostError(ErrorKind.REMOTE, "slurmctld unreachable", section="slurm.acct_share", at=T0)
    unread = replace(_summary().balances[0], used_su=None, limit_su=None, read_at=None, limit_source=None,
                     used_source=None)
    fresh = _summary(read_at=T0 + timedelta(hours=1), balances=(unread,), mode="usage_only", errors=(err,))
    store.apply("lyra", CollectorOutput(account=fresh), T0 + timedelta(hours=1))
    kept = store.hosts["lyra"].snapshot.account                          # type: ignore[union-attr]
    assert kept is not None and kept.balances == _summary().balances and kept.read_at == T0
    assert kept.mode == "limits" and kept.errors == (err,)


def test_a_restored_summary_younger_than_the_period_skips_the_periodic_group() -> None:
    """Review: every `agent --refresh` forced sshare + sreport + sbank; a new collector now takes the cached
    summary's reading time and sbank time (restore_account) and plans nothing while they are fresh."""
    from tests.backends.slurm.test_account_summary import GROUP, _ctx, _results
    cx = _ctx()
    first = SlurmCollector(cx.host)
    first.sections(GROUP, cx)
    out = first.ingest(_results("lyra", "slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank"), cx,
                       MonitorView())
    assert out.account is not None
    fresh = SlurmCollector(cx.host)
    fresh.restore_account(out.account)
    assert fresh.sections(GROUP, cx) == []                              # read just now by the previous process
    later = replace(cx, now=cx.now + timedelta(minutes=31))
    names = [x.name for x in fresh.sections(GROUP, later)]
    assert names == ["slurm.acct_share", "slurm.acct_sreport"]          # sbank: its 6 h period still holds
    assert "slurm.acct_sbank" in [x.name for x in SlurmCollector(cx.host).sections(GROUP, later)]


# ---------------------------------------------------------------- config (§3.7, §8.3)
def test_config_keys_defaults_floor_and_validation() -> None:
    s = SlurmSettings()
    assert (s.account_usage, s.poll_account, s.allocation_period) == (True, 1800, 365 * 86400)
    assert GeneralSettings().usage_window == "30d"
    cfg, warnings = parse_config({"general": {"usage_window": "period"},
                                  "defaults": {"slurm": {"poll_account": "1h", "allocation_period": "0"}},
                                  "host": [{"name": "k2", "account_usage": False, "poll_account": "5m"},
                                           {"name": "lyra", "allocation_period": "180d"}]})
    k2, lyra = cfg.hosts
    assert cfg.general.usage_window == "period"
    assert (k2.slurm.account_usage, k2.slurm.poll_account, k2.slurm.allocation_period) == (False, 900, 0)
    assert (lyra.slurm.poll_account, lyra.slurm.allocation_period) == (3600, 180 * 86400)
    assert any("poll_account" in w and "floor" in w for w in warnings)
    for bad in ({"general": {"usage_window": "60d"}}, {"host": [{"name": "k2", "account_usage": "yes"}]},
                {"host": [{"name": "k2", "allocation_period": 365}]},
                {"host": [{"name": "k2", "poll_account": "30"}]}):
        with pytest.raises(ConfigError):
            parse_config(bad)
    assert format_duration(SlurmSettings().allocation_period) == "365d"


def test_host_table_writes_allocation_period_zero_as_a_string() -> None:
    from runfleet.config import host_table_toml
    h = HostConfig("k2", "k2", slurm=SlurmSettings(allocation_period=0.0, account_usage=False))
    text = host_table_toml(h)
    assert 'allocation_period = "0"' in text and "account_usage = false" in text
    import tomllib
    (back,) = parse_config({"host": [tomllib.loads(text)["host"][0]]})[0].hosts
    assert back.slurm == h.slurm


# ---------------------------------------------------------------- the group under the poller (§5.4 load budget)
HOOKS = BackendHooks(collectors=lambda: {"slurm": SlurmCollector, "procfs": StubProcfs},
                     detect_sections=stub_detect_sections, build_capabilities=stub_build_capabilities)


class Rig:
    def __init__(self, *, account: bool = True, seed: int = 1) -> None:
        host = HostConfig("lyra", "lyra", backend="slurm", slurm=SlurmSettings(account_usage=account))
        self.clock = ManualClock(CLOCK_START)
        self.store = FleetStore([host])
        self.t = FakeTransport(host.name, mux_mode="none", clock=self.clock)
        caps = replace(stub_caps(host, cluster_name="lyra", slurm_config={"AccountingStorageType": SLURMDBD},
                                 associations=(("lyra", "pilab_02", "", "normal"),)),
                       tools=frozenset({"squeue", "sinfo", "sacct", "sshare", "sreport", "timeout"}))
        self.p = HostPoller(host, self.t, self.store, self.clock, random.Random(seed),
                            HostLanes(asyncio.Semaphore(8), self.t.mux), general=GeneralSettings(), hooks=HOOKS,
                            caps=caps, caps_at=CLOCK_START)
        self.task: asyncio.Task[None] | None = None

    def polls(self) -> list[tuple[float, tuple[str, ...]]]:
        return [(p.at, p.names) for p in self.t.probes if p.script.purpose == "poll" and p.at is not None]

    def account_times(self) -> list[float]:
        return [at for at, names in self.polls() if "slurm.acct_share" in names]


@pytest.fixture
async def rigs() -> AsyncIterator[Callable[..., Rig]]:
    made: list[Rig] = []

    def make(**kw: object) -> Rig:
        r = Rig(**kw)  # type: ignore[arg-type]
        r.task = asyncio.ensure_future(r.p.run())
        made.append(r)
        return r

    yield make
    for r in made:
        assert r.task is not None
        r.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await r.task


async def test_account_rides_regular_ticks_and_adds_no_probe(rigs: Callable[..., Rig]) -> None:
    """§5.4: with `slurm.account` the hour has exactly the probes it has without it (same seed, same jitter)."""
    on, off = rigs(), rigs(account=False)
    await run_for(on.clock, 3600, step=1.0)
    await run_for(off.clock, 3600, step=1.0)
    assert len(on.polls()) == len(off.polls()) and 225 <= len(on.polls()) <= 240
    assert all("slurm.squeue" in names for _at, names in on.polls())        # never a probe of its own
    times = on.account_times()
    assert len(times) == 2 and 1790 <= times[1] - times[0] <= 1830
    assert all("slurm.acct_sreport" in names for at, names in on.polls() if at in times)
    assert not any(n.startswith("slurm.acct") for _at, names in off.polls() for n in names)
    snap = on.store.hosts["lyra"].snapshot
    assert snap is not None and snap.account is not None and snap.account.balances[0].account == "pilab_02"
    assert off.store.hosts["lyra"].snapshot.account is None                 # type: ignore[union-attr]


async def test_hidden_host_reads_the_account_hourly(rigs: Callable[..., Rig]) -> None:
    rig = rigs()
    await settle()
    rig.p.set_visibility(Visibility.HIDDEN)
    await run_for(rig.clock, 7300, step=1.0)
    times = rig.account_times()
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(times) == 3 and all(3550 <= g <= 3700 for g in gaps), gaps


async def test_failing_sreport_backs_off_up_to_two_hours_and_keeps_the_summary(rigs: Callable[..., Rig]) -> None:
    rig = rigs()
    rig.t.set_result("slurm.acct_sreport", "", 1, "sreport: error: Problem talking to the database: Connection refused")
    await run_for(rig.clock, 5 * 3600 + 600, step=2.0)
    times = rig.account_times()
    gaps = [round((b - a) / 60) for a, b in zip(times, times[1:], strict=False)]
    assert gaps[:3] == [60, 120, 120], gaps                                  # 30 min ×2, ×4 capped at 2 h
    snap = rig.store.hosts["lyra"].snapshot
    assert snap is not None and snap.account is not None
    (e,) = snap.account.errors
    assert e.summary == "slurmdbd unreachable" and snap.account.balances[0].account == "pilab_02"
    st = rig.store.hosts["lyra"].status.sections["slurm.acct_sreport"]
    assert not st.ok and st.error is not None and st.error.summary == "slurmdbd unreachable"


async def test_controller_slowdown_slows_the_account_group(rigs: Callable[..., Rig]) -> None:
    rig = rigs()
    await settle()
    (cad,) = [c for cs in rig.p._cadences(rig.p.context()).values() for c in cs if c.group == "slurm.account"]
    assert rig.p.effective_period(cad) == 1800
    rig.p.controller.factor = 4.0
    assert rig.p.effective_period(cad) == 7200
