"""[WP1] Attach (§7.16) through FleetService with FakeTransport + StubSlurm: policy per setting, consent, read-only,
node_ssh learn/apply, refresh timing on the ManualClock, autostop, failure backoff, pauses, single flight, stop on job
end, srun row counting, job_detail live, and no tasks / probes left behind by stop, host removal or shutdown."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from runfleet import debugdump
from runfleet.config import FleetConfig, GeneralSettings, HostConfig, SlurmSettings
from runfleet.core import attach as A
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.core.store import ChangeSet
from runfleet.errors import AttachBlocked, AttachConsentRequired
from runfleet.models import AttachStatus, JobKey
from runfleet.transport.base import Transport
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, run_for, settle

K = JobKey("k2", "101")
K2 = JobKey("k2", "103")
NODE_OK = "landed=101\ngpu 0 50 1000"


def k2(attach: str = "ssh", refresh: float = 15) -> HostConfig:
    return HostConfig(name="k2", ssh="k2", backend="slurm",
                      slurm=SlurmSettings(attach=attach, attach_refresh=refresh))  # type: ignore[arg-type]


class World:
    def __init__(self, host: HostConfig, tmp: Path, *, general: GeneralSettings | None = None,
                 latency: float = 0.0, cache: Cache | None = None) -> None:
        self.clock = ManualClock(CLOCK_START)
        self.latency = latency
        self.transports: dict[str, FakeTransport] = {}
        self.svc = FleetService(FleetConfig(hosts=(host,), general=general or GeneralSettings()), self.factory,
                                self.clock, cache)
        self.svc.hooks = STUB_HOOKS
        self.svc.lock_path = tmp / "instance.lock"
        self.changes: list[ChangeSet] = []
        self.svc.store.subscribe(self.changes.append)

    def factory(self, host: HostConfig) -> Transport:
        t = self.transports.get(host.name)
        if t is None:
            t = FakeTransport(host.name, clock=self.clock, latency=self.latency)
            t.set_result("detect.id", "user=me")
            t.set_result("slurm.squeue", "101 R\n103 R\n102 PD")
            t.set_result("slurm.attach.node0", NODE_OK)
            t.set_result("slurm.attach.audit", "clean")
            self.transports[host.name] = t
        return t

    @property
    def t(self) -> FakeTransport:
        return self.transports["k2"]

    def attach_probes(self) -> list[float | None]:
        return [p.at for p in self.t.probes if p.script.purpose == "attach"]

    def live_notes(self, key: JobKey = K) -> int:
        return sum(1 for c in self.changes if c.host == key.host and key.id in c.live)

    def state(self, key: JobKey = K):  # type: ignore[no-untyped-def]
        st = self.svc.attach_state(key)
        assert st is not None
        return st

    async def started(self) -> World:
        await self.svc.start()
        await settle()
        if self.latency:                                               # detect + first poll take clock time
            await run_for(self.clock, 3 * self.latency + 1, step=0.5)
        assert self.svc.store.job(K) is not None
        return self


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[Callable[..., World]]:
    made: list[World] = []

    def make(host: HostConfig | None = None, **kw: object) -> World:
        w = World(host or k2(), tmp_path, **kw)   # type: ignore[arg-type]
        made.append(w)
        return w

    yield make
    for w in made:
        await w.svc.stop()
        assert not w.svc.attach._tasks


# ---------------------------------------------------------------- ssh: start, refresh, stop
async def test_ssh_samples_at_once_then_every_attach_refresh(world: Callable[..., World]) -> None:
    w = await world().started()
    st = w.svc.attach_start(K)
    assert st.mode == "ssh" and st.status is AttachStatus.LIVE and st.refresh_s == 15
    await settle()
    assert len(w.attach_probes()) == 1
    st = w.state()
    assert st.status is AttachStatus.LIVE and st.live is not None and st.live.has_data and st.live.clean is True
    assert st.next_at == w.clock.now() + timedelta(seconds=15)
    assert st.error is None and st.failures == 0 and st.srun_rows == 0
    assert w.live_notes() >= 2                                         # SAMPLING, then LIVE with data
    await run_for(w.clock, 46, step=1.0)
    ats = w.attach_probes()
    assert len(ats) == 4 and [b - a for a, b in zip(ats, ats[1:], strict=False)] == [15.0, 15.0, 15.0]  # type: ignore[operator]
    assert all(p.names == ("slurm.attach.node0", "slurm.attach.audit") for p in w.t.probes
               if p.script.purpose == "attach")


async def test_attach_start_is_idempotent_and_stop_keeps_the_state(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    w.svc.attach_start(K)
    w.svc.attach_start(K)
    await settle()
    assert len(w.attach_probes()) == 1
    w.svc.attach_stop(K)
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == A.TAB_CLOSED and st.live is not None
    assert st.next_at is None
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == 1
    w.svc.attach_stop()                                                # nothing open: no-op
    assert w.svc.attach_state(K2) is None


async def test_reopening_right_after_a_sample_keeps_its_schedule(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await run_for(w.clock, 5, step=1.0)
    w.svc.attach_stop()                                                # e.g. a dialog over the screen
    w.svc.attach_start(K)
    await settle()
    assert len(w.attach_probes()) == 1                                 # no second sample 5 s after the first
    await run_for(w.clock, 11, step=1.0)
    assert len(w.attach_probes()) == 2


async def test_one_attach_per_app(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    w.svc.attach_start(K2)
    st = w.state(K)
    assert st.status is AttachStatus.STOPPED and st.stopped == A.ANOTHER_JOB
    await settle()
    names = [p.script.sections[0].cmd for p in w.t.probes if p.script.purpose == "attach"]
    assert names == ["ssh n1 101", "ssh n1 103"]
    await run_for(w.clock, 20, step=1.0)
    names = [p.script.sections[0].cmd for p in w.t.probes if p.script.purpose == "attach"]
    assert names.count("ssh n1 101") == 1 and names.count("ssh n1 103") == 2


async def test_auto_off_and_on(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    st = w.svc.attach_set_auto(K, False)
    assert st.status is AttachStatus.IDLE and not st.auto and st.next_at is None
    await run_for(w.clock, 40, step=1.0)
    assert len(w.attach_probes()) == 1
    st = w.svc.attach_set_auto(K, True)                                # overdue: samples at once
    await settle()
    assert len(w.attach_probes()) == 2 and w.state().status is AttachStatus.LIVE


async def test_first_sample_runs_even_with_auto_off(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_set_auto(K, False)
    w.svc.attach_start(K)
    await run_for(w.clock, 40, step=1.0)
    assert len(w.attach_probes()) == 1 and w.state().status is AttachStatus.IDLE


# ---------------------------------------------------------------- `r` (attach_now)
async def test_attach_now_single_flight_and_three_second_floor(world: Callable[..., World]) -> None:
    w = await world(latency=2.0).started()
    w.svc.attach_start(K)
    await settle()
    assert w.state().status is AttachStatus.SAMPLING
    a = asyncio.ensure_future(w.svc.attach_now(K))
    b = asyncio.ensure_future(w.svc.attach_now(K))
    await run_for(w.clock, 2.5)
    la, lb = a.result(), b.result()
    assert la is lb and len(w.attach_probes()) == 1                   # in-flight sample awaited, not duplicated
    assert await w.svc.attach_now(K) is la                             # < 3 s old: returned as is
    await run_for(w.clock, 0.5)
    assert await w.svc.attach_now(K) is la                             # still < 3 s after it ended
    await run_for(w.clock, 2.5)
    c = asyncio.ensure_future(w.svc.attach_now(K))
    await run_for(w.clock, 2.5)
    assert c.result() is not la and len(w.attach_probes()) == 2
    st = w.state()
    assert st.status is AttachStatus.LIVE                              # next auto sample refresh after this one
    assert st.next_at == st.live.sampled_at + timedelta(seconds=15)  # type: ignore[union-attr]


async def test_attach_now_without_start_never_arms_auto(world: Callable[..., World]) -> None:
    w = await world().started()
    live = await w.svc.attach_now(K)
    assert live.has_data and w.state().status is AttachStatus.IDLE
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == 1


async def test_attach_now_blocked_reasons(world: Callable[..., World]) -> None:
    w = await world().started()
    with pytest.raises(AttachBlocked, match="job is not running"):
        await w.svc.attach_now(JobKey("k2", "102"))
    with pytest.raises(AttachBlocked, match='srun is not enabled for k2'):
        await w.svc.attach_now(K, mode="srun")
    assert w.attach_probes() == []
    st = w.svc.attach_start(JobKey("k2", "102"))
    assert st.status is AttachStatus.BLOCKED and st.blocked == "job is not running"


async def test_not_connected_is_blocked() -> None:
    w = World(k2(), Path("/nonexistent"))
    st = w.svc.attach_start(K)                                         # no service started, no job known
    assert st.status is AttachStatus.BLOCKED
    w.svc.attach_stop()
    await w.svc.stop()


# ---------------------------------------------------------------- settings, consent, read-only
async def test_setting_off(world: Callable[..., World]) -> None:
    w = await world(k2("off")).started()
    st = w.svc.attach_start(K)
    assert st.status is AttachStatus.BLOCKED and st.blocked == "attach is off for k2" and st.setting == "off"
    with pytest.raises(AttachBlocked, match="attach is off"):
        await w.svc.attach_now(K)
    await run_for(w.clock, 20)
    assert w.attach_probes() == []


async def test_srun_consent_once_and_for_this_job(world: Callable[..., World]) -> None:
    w = await world(k2("srun")).started()
    w.t.set_result("slurm.attach.node0", "landed=101\ngpu 0 99 2000")
    st = w.svc.attach_start(K)
    assert st.status is AttachStatus.NEEDS_CONSENT and st.mode == "srun" and st.refresh_s is None
    assert st.offer is not None and st.offer.preview == "srun n1 101" and st.offer.node == "n1"
    assert "permanent sacct row 101.N rfprobe" in st.offer.body and st.offer.title == "Read job 101 with srun?"
    await run_for(w.clock, 60, step=1.0)
    assert w.attach_probes() == []                                     # opening the tab never runs srun
    with pytest.raises(AttachConsentRequired) as exc:
        await w.svc.attach_now(K)
    assert exc.value.offer == st.offer and exc.value.key == K
    live = await w.svc.attach_now(K, consent="once")
    assert live.mode == "srun" and live.sacct_rows_added == 1
    st = w.state()
    assert st.status is AttachStatus.IDLE and st.srun_rows == 1 and st.consent == "none" and st.live is live
    with pytest.raises(AttachConsentRequired):                         # "once" asks again next time
        await w.svc.attach_now(K)
    await w.svc.attach_now(K, consent="job")
    await w.svc.attach_now(K)                                          # remembered for this job
    st = w.state()
    assert st.srun_rows == 3 and st.consent == "job"
    w.svc.attach_stop()
    assert w.svc.attach_start(K).consent == "job"                      # kept for the session
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == 3                                 # srun never auto-refreshes
    with pytest.raises(AttachConsentRequired):                         # another job asks on its own
        await w.svc.attach_now(K2)


async def test_srun_rows_count_failed_steps_but_not_unrun_ones(world: Callable[..., World]) -> None:
    w = await world(k2("srun")).started()
    w.t.set_result("slurm.attach.node0", "", rc=1, stderr="srun: error: step creation temporarily disabled")
    live = await w.svc.attach_now(K, consent="job")
    st = w.state()
    assert not live.has_data and live.errors and st.srun_rows == 1     # the stub counts a launched step
    assert st.error is not None and st.live is None and st.failures == 1
    w.svc.set_general(save=False, readonly=True)
    with pytest.raises(AttachBlocked, match="read-only"):
        await w.svc.attach_now(K)
    assert w.state().srun_rows == 1


async def test_readonly_blocks_srun_but_not_ssh(world: Callable[..., World]) -> None:
    w = await world(k2("srun"), general=GeneralSettings(readonly=True)).started()
    st = w.svc.attach_start(K)
    assert st.status is AttachStatus.BLOCKED and st.blocked == A.READONLY_REASON and st.offer is None
    with pytest.raises(AttachBlocked, match="read-only"):
        await w.svc.attach_now(K, consent="job")
    assert w.attach_probes() == []
    w2 = await world(k2("ssh+srun"), general=GeneralSettings(readonly=True)).started()
    st = w2.svc.attach_start(K)
    assert st.mode == "ssh" and st.status is AttachStatus.LIVE
    await settle()
    assert len(w2.attach_probes()) == 1
    with pytest.raises(AttachBlocked, match="read-only"):
        await w2.svc.attach_now(K, mode="srun", consent="once")


async def test_readonly_switched_on_while_waiting_for_the_lane(world: Callable[..., World]) -> None:
    w = await world(k2("srun")).started()
    orig = w.svc._on_demand
    rechecks: list[object] = []

    async def flip_then_probe(host, script, *, recheck=None):  # type: ignore[no-untyped-def]
        rechecks.append(recheck)
        w.svc.readonly = True                                          # Settings flipped it while queued
        return await orig(host, script, recheck=recheck)

    w.svc.attach._probe = flip_then_probe
    live = await w.svc.attach_now(K, consent="once")
    assert rechecks and callable(rechecks[0])
    assert not live.has_data and live.sacct_rows_added == 0 and w.attach_probes() == []
    assert any("read-only" in e.summary for e in live.errors)


# ---------------------------------------------------------------- node_ssh learn / apply
async def test_ssh_denied_is_learned_and_blocks_with_the_hint(tmp_path: Path, world: Callable[..., World]) -> None:
    cache = Cache(tmp_path / "cache.sqlite")
    w = await world(k2("ssh"), cache=cache).started()
    w.t.set_result("slurm.attach.node0", "", rc=255, stderr="Permission denied (publickey,hostbased).")
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.status is AttachStatus.BLOCKED and st.blocked == "k2 refuses ssh to compute nodes"
    assert st.hint == A.srun_hint("k2") and "permanent sacct row" in st.hint
    assert st.error is not None and st.live is None
    snap = w.svc.store.hosts["k2"].snapshot
    assert snap is not None and snap.caps is not None and snap.caps.node_ssh == "denied"
    assert w.svc._pollers["k2"].caps.node_ssh == "denied"              # type: ignore[union-attr]
    await cache.flush()
    loaded = cache.load_caps("k2", w.clock.now())
    assert loaded is not None and loaded[0].node_ssh == "denied"
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == 1                                 # no more ssh attempts
    with pytest.raises(AttachBlocked) as exc:
        await w.svc.attach_now(K)
    assert exc.value.hint == A.srun_hint("k2")
    await w.svc.stop()
    await cache.close()


async def test_ssh_denied_with_ssh_srun_goes_to_the_srun_offer(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun")).started()
    w.t.set_result("slurm.attach.node0", "denied")
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.status is AttachStatus.NEEDS_CONSENT and st.mode == "srun"
    assert st.offer is not None and st.offer.reason == "k2 refuses ssh to compute nodes"
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == 1
    w.t.set_result("slurm.attach.node0", NODE_OK)
    live = await w.svc.attach_now(K, consent="once")
    assert live.mode == "srun" and w.state().srun_rows == 1
    caps = w.svc.store.hosts["k2"].snapshot.caps                       # type: ignore[union-attr]
    assert caps is not None and caps.node_ssh == "denied"              # srun landings do not re-learn "adopt"


async def test_ssh_landing_learns_adopt(world: Callable[..., World]) -> None:
    w = await world().started()
    await w.svc.attach_now(K)
    caps = w.svc.store.hosts["k2"].snapshot.caps                       # type: ignore[union-attr]
    assert caps is not None and caps.node_ssh == "adopt"
    n = sum(1 for c in w.changes if c.status)
    await w.svc.attach_now(K)                                          # < 3 s: no probe, nothing re-applied
    await run_for(w.clock, 4)
    await w.svc.attach_now(K)                                          # same patch again: caps unchanged
    assert sum(1 for c in w.changes if c.status) <= n + 2              # (poll ticks may publish status)


async def test_ssh_landed_in_another_job_offers_srun(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun")).started()
    w.t.set_result("slurm.attach.node0", "landed=999\ngpu 0 1 100")
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.status is AttachStatus.LIVE and st.mode == "ssh" and st.live is not None
    assert st.offer is not None and st.offer.reason == "ssh landed in job 999" and "newest job" in st.offer.body
    with pytest.raises(AttachConsentRequired) as exc:
        await w.svc.attach_now(K, mode="srun")
    assert exc.value.offer == st.offer
    live = await w.svc.attach_now(K, mode="srun", consent="once")
    assert live.mode == "srun" and w.state().srun_rows == 1 and w.state().status is AttachStatus.LIVE
    w2 = await world(k2("ssh")).started()                              # plain "ssh": no srun offer
    w2.t.set_result("slurm.attach.node0", "landed=999\ngpu 0 1 100")
    w2.svc.attach_start(K)
    await settle()
    assert w2.state().offer is None and w2.state().live is not None


async def test_explicit_srun_read_from_ssh_mode(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun")).started()
    with pytest.raises(AttachConsentRequired) as exc:
        await w.svc.attach_now(K, mode="srun", node="n7")
    assert exc.value.offer.node == "n7" and exc.value.offer.preview == "srun n7 101"
    await w.svc.attach_now(K, mode="srun", node="n7", consent="once")
    cmds = [p.script.sections[0].cmd for p in w.t.probes if p.script.purpose == "attach"]
    assert cmds == ["srun n7 101"]


# ---------------------------------------------------------------- lifecycle guards
async def test_autostop_after_ten_minutes_without_input(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await run_for(w.clock, 590, step=5.0)
    assert w.state().status is AttachStatus.LIVE
    n = len(w.attach_probes())
    await run_for(w.clock, 15, step=5.0)
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == A.AUTOSTOP_REASON
    await run_for(w.clock, 120, step=5.0)
    assert len(w.attach_probes()) <= n + 1
    w.svc.note_user_activity()                                         # input alone does not restart …
    await run_for(w.clock, 30, step=5.0)
    assert w.state().status is AttachStatus.STOPPED
    await w.svc.attach_now(K)                                          # … `r` does
    assert w.state().status is AttachStatus.LIVE
    m = len(w.attach_probes())
    await run_for(w.clock, 31, step=1.0)
    assert len(w.attach_probes()) == m + 2


async def test_user_input_keeps_auto_refresh_going(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    for _ in range(14):
        await run_for(w.clock, 60, step=5.0)
        w.svc.note_user_activity()
    assert w.state().status is AttachStatus.LIVE and len(w.attach_probes()) >= 55


async def test_failures_back_off_then_stop_after_three(world: Callable[..., World]) -> None:
    w = await world().started()
    w.t.set_result("slurm.attach.node0", "", rc=255, stderr="ssh: connect to host n1 port 22: Connection timed out")
    w.svc.attach_start(K)
    await run_for(w.clock, 100, step=1.0)
    ats = w.attach_probes()
    assert len(ats) == 3 and [b - a for a, b in zip(ats, ats[1:], strict=False)] == [30.0, 30.0]  # type: ignore[operator]
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == A.FAILURES_REASON and st.failures == 3
    assert st.error is not None and st.live is None
    w.t.set_result("slurm.attach.node0", NODE_OK)
    await w.svc.attach_now(K)                                          # `r` re-arms
    st = w.state()
    assert st.status is AttachStatus.LIVE and st.failures == 0 and st.error is None


async def test_failure_backoff_is_capped_at_60s(world: Callable[..., World]) -> None:
    w = await world(k2(refresh=45)).started()
    w.t.set_result("slurm.attach.node0", "", rc=1, stderr="boom")
    w.svc.attach_start(K)
    await run_for(w.clock, 70, step=1.0)
    ats = w.attach_probes()
    assert len(ats) == 2 and ats[1] - ats[0] == 60.0                   # type: ignore[operator]


async def test_a_good_sample_keeps_live_across_a_failed_one(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    good = w.state().live
    w.t.set_result("slurm.attach.node0", "", rc=1, stderr="boom")
    await run_for(w.clock, 15, step=1.0)
    st = w.state()
    assert st.live is good and st.error is not None and st.failures == 1
    w.t.set_result("slurm.attach.node0", NODE_OK)
    await run_for(w.clock, 30, step=1.0)
    st = w.state()
    assert st.live is not good and st.error is None and st.failures == 0


async def test_transport_failure_still_ingests_and_appends_the_error(world: Callable[..., World]) -> None:
    from runfleet.models import ErrorKind
    w = await world(k2("srun")).started()
    w.t.fail_next(ErrorKind.TIMEOUT, summary="probe timed out")
    live = await w.svc.attach_now(K, consent="once")
    assert not live.has_data and live.errors[-1].summary == "probe timed out"
    assert live.sacct_rows_added == 0 and w.state().srun_rows == 0     # NOT_RUN sections: no row


async def test_blur_pauses_and_focus_resumes_at_once(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    w.svc.set_app_focus(False)
    await settle()
    st = w.state()
    assert st.status is AttachStatus.PAUSED and st.stopped == A.BLURRED
    await run_for(w.clock, 40, step=1.0)
    assert len(w.attach_probes()) == 1
    w.svc.set_app_focus(True)
    await settle()
    assert len(w.attach_probes()) == 2 and w.state().status is AttachStatus.LIVE


async def test_idle_pause_pauses_attach(world: Callable[..., World]) -> None:
    w = await world(general=GeneralSettings(idle_after=60, idle_pause_after=120)).started()
    w.svc.attach_start(K)
    await run_for(w.clock, 125, step=1.0)
    st = w.state()
    assert w.svc.store.idle_paused and st.status is AttachStatus.PAUSED and st.stopped == A.IDLE_PAUSED
    n = len(w.attach_probes())
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == n
    w.svc.note_user_activity()
    await settle()
    assert len(w.attach_probes()) == n + 1 and w.state().status is AttachStatus.LIVE


async def test_stop_when_the_job_ends_or_disappears(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    w.t.set_result("slurm.squeue", "101 CD\n103 R")
    await run_for(w.clock, 20, step=1.0)
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == A.JOB_ENDED_REASON and st.live is not None
    n = len(w.attach_probes())
    await run_for(w.clock, 60, step=1.0)
    assert len(w.attach_probes()) == n
    w.svc.attach_start(K2)
    await settle()
    w.t.set_result("slurm.squeue", "")
    await run_for(w.clock, 20, step=1.0)
    st = w.state(K2)
    assert st.status is AttachStatus.STOPPED and st.stopped == A.JOB_ENDED_REASON


async def test_tab_on_a_pending_job_attaches_once_it_runs(world: Callable[..., World]) -> None:
    w = await world().started()
    pend = JobKey("k2", "102")
    assert w.svc.attach_start(pend).status is AttachStatus.BLOCKED
    w.t.set_result("slurm.squeue", "101 R\n102 R")
    await run_for(w.clock, 20, step=1.0)
    assert w.state(pend).status is AttachStatus.LIVE and len(w.attach_probes()) == 1


async def test_disconnected_host_pauses_then_resumes(world: Callable[..., World]) -> None:
    from dataclasses import replace

    from runfleet.models import ConnState
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    hs = w.svc.store.hosts["k2"]
    good = hs.status
    w.svc.store.set_status("k2", replace(good, state=ConnState.RETRYING))
    await settle()
    st = w.state()
    assert st.status is AttachStatus.PAUSED and st.stopped == "k2: not connected"
    w.svc.store.set_status("k2", good)
    await settle()
    assert w.state().status is AttachStatus.LIVE


async def test_disabling_the_host_stops_attach(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    w.svc.set_host_enabled("k2", False)
    await settle()
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == "k2 is disabled"


# ---------------------------------------------------------------- detail, cleanup
async def test_job_detail_carries_the_live_sample(world: Callable[..., World]) -> None:
    w = await world().started()
    d = await w.svc.job_detail(K)
    assert d.live is None
    live = await w.svc.attach_now(K)
    assert (await w.svc.job_detail(K)).live is live                    # cached detail, live filled in
    assert w.svc._details[K][1].live is None                           # the cache keeps the bare detail


async def test_stop_cancels_an_in_flight_auto_sample(world: Callable[..., World]) -> None:
    w = await world(latency=5.0).started()
    w.svc.attach_start(K)
    await settle()
    assert w.t.in_flight == 1 and w.state().status is AttachStatus.SAMPLING
    w.svc.attach_stop()
    assert w.state().status is AttachStatus.STOPPED
    await settle()
    assert w.t.in_flight == 0 and not w.svc.attach._tasks
    await run_for(w.clock, 30, step=1.0)
    assert len(w.attach_probes()) == 1 and w.state().live is None


async def test_user_sample_survives_the_tab_closing(world: Callable[..., World]) -> None:
    w = await world(k2("srun"), latency=2.0).started()
    w.svc.attach_start(K)
    task = asyncio.ensure_future(w.svc.attach_now(K, consent="once"))
    await settle()
    w.svc.attach_stop()                                                # the srun step is already on its way
    await run_for(w.clock, 3.0)
    assert task.result().sacct_rows_added == 1 and w.state().srun_rows == 1


async def test_service_stop_leaves_no_tasks_or_probes(world: Callable[..., World]) -> None:
    w = await world(latency=5.0).started()
    w.svc.attach_start(K)
    user = asyncio.ensure_future(w.svc.attach_now(K))
    await settle()
    assert w.t.in_flight >= 1
    await w.svc.stop()
    assert w.t.in_flight == 0 and not w.svc.attach._tasks
    await settle()
    assert user.done()
    with pytest.raises((AttachBlocked, asyncio.CancelledError)):
        user.result()
    assert w.svc.attach_state(K).status is not AttachStatus.SAMPLING   # type: ignore[union-attr]


async def test_removing_the_host_forgets_its_attach(world: Callable[..., World]) -> None:
    w = await world(latency=5.0).started()
    w.svc.attach_start(K)
    await settle()
    await w.svc.remove_host("k2", save=False)
    assert w.svc.attach_state(K) is None and not w.svc.attach._tasks and w.t.in_flight == 0


def test_debug_dump_names_attach_probes(tmp_path: Path) -> None:
    debugdump.enable(tmp_path)
    try:
        path = debugdump.dump_probe("k2", "attach", "script", b"out", b"", "meta")
    finally:
        debugdump.disable()
    assert path is not None and path.name.endswith("-attach.in") and path.parent.name == "k2"


async def test_a_bug_in_the_sample_path_backs_off_instead_of_looping(world: Callable[..., World]) -> None:
    w = await world().started()
    calls: list[float] = []

    async def broken(host, script, *, recheck=None):  # type: ignore[no-untyped-def]
        calls.append(w.clock.monotonic())
        raise RuntimeError("bug")

    w.svc.attach._probe = broken
    w.svc.attach_start(K)
    await run_for(w.clock, 100, step=1.0)
    st = w.state()
    assert len(calls) == 3 and st.status is AttachStatus.STOPPED and st.stopped == A.FAILURES_REASON
    assert st.error is not None and st.error.kind.value == "internal"


# ---------------------------------------------------------------- requests waiting behind another sample
def _attach_cmds(w: World) -> list[str]:
    return [p.script.sections[0].cmd for p in w.t.probes if p.script.purpose == "attach"]


async def test_a_waiting_srun_read_never_starts_after_shutdown(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun"), latency=5.0).started()
    w.svc.attach_start(K)
    await settle()
    assert w.state().status is AttachStatus.SAMPLING                   # the auto ssh sample holds the entry
    user = asyncio.ensure_future(w.svc.attach_now(K, mode="srun", consent="once"))
    await settle()
    assert not user.done()                                             # waits behind the ssh sample
    await w.svc.stop()
    await run_for(w.clock, 10, step=1.0)
    assert user.done()
    with pytest.raises(AttachBlocked):
        user.result()
    assert not any(c.startswith("srun") for c in _attach_cmds(w))      # no srun step after shutdown
    assert w.t.in_flight == 0 and not w.svc.attach._tasks
    with pytest.raises(AttachBlocked):                                 # and none for a late caller either
        await w.svc.attach_now(K, mode="srun", consent="once")


async def test_a_waiting_srun_read_never_starts_after_host_removal(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun"), latency=5.0).started()
    w.svc.attach_start(K)
    await settle()
    user = asyncio.ensure_future(w.svc.attach_now(K, mode="srun", consent="once"))
    await settle()
    t = w.t
    await w.svc.remove_host("k2", save=False)
    await run_for(w.clock, 10, step=1.0)
    assert user.done()
    with pytest.raises(AttachBlocked):
        user.result()
    assert not any(p.script.sections[0].cmd.startswith("srun") for p in t.probes if p.script.purpose == "attach")
    assert t.in_flight == 0 and not w.svc.attach._tasks and w.svc.attach_state(K) is None


# ---------------------------------------------------------------- dispatch-time recheck of automatic samples
async def test_a_queued_auto_sample_is_dropped_when_the_app_blurs(world: Callable[..., World]) -> None:
    w = await world().started()
    orig = w.svc.attach._probe

    async def blur_then_probe(host, script, *, recheck=None):  # type: ignore[no-untyped-def]
        w.svc.set_app_focus(False)                                     # blurred while queued for the lane
        return await orig(host, script, recheck=recheck)

    w.svc.attach._probe = blur_then_probe
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert w.attach_probes() == []
    assert st.status is AttachStatus.PAUSED and st.stopped == A.BLURRED
    assert st.failures == 0 and st.error is None                       # a dropped sample is not a failed read
    w.svc.attach._probe = orig
    w.svc.set_app_focus(True)
    await settle()
    assert len(w.attach_probes()) == 1 and w.state().status is AttachStatus.LIVE


async def test_a_queued_auto_sample_is_dropped_when_auto_is_switched_off(world: Callable[..., World]) -> None:
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    assert len(w.attach_probes()) == 1
    orig = w.svc.attach._probe

    async def auto_off_then_probe(host, script, *, recheck=None):  # type: ignore[no-untyped-def]
        w.svc.attach_set_auto(K, False)
        return await orig(host, script, recheck=recheck)

    w.svc.attach._probe = auto_off_then_probe
    await run_for(w.clock, 16, step=1.0)
    st = w.state()
    assert len(w.attach_probes()) == 1 and st.status is AttachStatus.IDLE and st.failures == 0
    w.svc.attach._probe = orig
    await w.svc.attach_now(K)                                          # `r` keeps its own rules
    assert len(w.attach_probes()) == 2


async def test_a_queued_auto_sample_is_dropped_when_the_job_ends(world: Callable[..., World]) -> None:
    from dataclasses import replace

    from runfleet.models import JobState
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    orig = w.svc.attach._probe

    async def end_then_probe(host, script, *, recheck=None):  # type: ignore[no-untyped-def]
        snap = w.svc.store.hosts["k2"].snapshot
        assert snap is not None
        jobs = dict(snap.jobs)
        jobs[K.id] = replace(jobs[K.id], state=JobState.COMPLETED)
        w.svc.store.hosts["k2"].snapshot = replace(snap, jobs=jobs)
        return await orig(host, script, recheck=recheck)

    w.svc.attach._probe = end_then_probe
    await run_for(w.clock, 16, step=1.0)
    assert len(w.attach_probes()) == 1


# ---------------------------------------------------------------- node pins: cache and single flight per node
async def test_explicit_node_reads_never_reuse_another_nodes_sample(world: Callable[..., World]) -> None:
    w = await world().started()
    l1 = await w.svc.attach_now(K)
    l2 = await w.svc.attach_now(K, node="n2")                          # < 3 s later, but another node
    assert l2 is not l1 and l2.nodes[0].node == "n2"
    assert _attach_cmds(w) == ["ssh n1 101", "ssh n2 101"]
    assert await w.svc.attach_now(K, node="n2") is l2                  # same node < 3 s: reused
    assert len(w.attach_probes()) == 2


async def test_concurrent_reads_of_two_nodes_do_not_share_a_sample(world: Callable[..., World]) -> None:
    w = await world(latency=2.0).started()
    a = asyncio.ensure_future(w.svc.attach_now(K))
    b = asyncio.ensure_future(w.svc.attach_now(K, node="n2"))
    c = asyncio.ensure_future(w.svc.attach_now(K, node="n2"))
    await run_for(w.clock, 6)
    assert a.result().nodes[0].node == "n1" and b.result().nodes[0].node == "n2"
    assert c.result() is b.result()                                    # same node: joined
    assert _attach_cmds(w) == ["ssh n1 101", "ssh n2 101"]


async def test_changing_the_pin_drops_the_old_nodes_auto_sample(world: Callable[..., World]) -> None:
    w = await world(latency=2.0).started()
    w.svc.attach_start(K)
    await settle()
    assert w.state().status is AttachStatus.SAMPLING
    w.svc.attach_start(K, node="n2")                                   # pin changed while n1 is in flight
    await run_for(w.clock, 3)
    st = w.state()
    assert st.node == "n2" and st.live is not None and [n.node for n in st.live.nodes] == ["n2"]


async def test_changing_the_pin_wakes_a_sleeping_runner(world: Callable[..., World]) -> None:
    """After an n1 sample the runner sleeps until the next refresh; pinning n2 must sample n2 right away."""
    w = await world().started()
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.live is not None and [n.node for n in st.live.nodes] == ["n1"]
    w.svc.attach_start(K, node="n2")
    await settle()
    st = w.state()
    assert st.live is not None and [n.node for n in st.live.nodes] == ["n2"]


# ---------------------------------------------------------------- ssh cannot reach the node: srun offer
async def test_unknown_node_host_key_offers_srun_behind_consent(world: Callable[..., World]) -> None:
    w = await world(k2("ssh+srun")).started()
    w.t.set_result("slurm.attach.node0", "", rc=255, stderr="Host key verification failed.")
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.mode == "ssh" and st.error is not None and st.error.kind.value == "hostkey"
    assert st.offer is not None and st.offer.node == "n1" and "host key" in st.offer.reason
    assert st.offer.preview == "srun n1 101" and "permanent sacct row" in st.offer.body
    await run_for(w.clock, 100, step=1.0)
    st = w.state()
    assert st.status is AttachStatus.STOPPED and st.stopped == A.FAILURES_REASON and st.offer is not None
    assert all(c.startswith("ssh") for c in _attach_cmds(w))           # srun never runs by itself
    with pytest.raises(AttachConsentRequired) as exc:
        await w.svc.attach_now(K, mode="srun")
    assert exc.value.offer == st.offer
    w.t.set_result("slurm.attach.node0", NODE_OK)
    live = await w.svc.attach_now(K, mode="srun", consent="once")
    assert live.mode == "srun" and w.state().srun_rows == 1
    await w.svc.attach_now(K)                                          # ssh works again: the offer goes
    assert w.state().offer is None


async def test_unknown_node_host_key_without_srun_permission_offers_nothing(world: Callable[..., World]) -> None:
    w = await world(k2("ssh")).started()
    w.t.set_result("slurm.attach.node0", "", rc=255, stderr="Host key verification failed.")
    w.svc.attach_start(K)
    await settle()
    st = w.state()
    assert st.error is not None and st.offer is None


# ---------------------------------------------------------------- idle hint streak
async def test_a_failed_read_breaks_the_idle_streak(world: Callable[..., World]) -> None:
    w = await world().started()
    sched = w.svc._scheduler("k2")
    assert sched is not None
    seen: list[object] = []
    orig = sched.ingest_attach

    def spy(plan, results, ctx, *, previous=None):  # type: ignore[no-untyped-def]
        seen.append(previous)
        return orig(plan, results, ctx, previous=previous)

    sched.ingest_attach = spy  # type: ignore[method-assign]
    idle = "landed=101\ngpu 0 1 4000"
    w.t.set_result("slurm.attach.node0", idle)
    first = await w.svc.attach_now(K)
    assert first.idle_streak == {"GPU-0": 1}
    await run_for(w.clock, 4)
    w.t.set_result("slurm.attach.node0", "", rc=1, stderr="boom")
    assert not (await w.svc.attach_now(K)).has_data
    await run_for(w.clock, 4)
    w.t.set_result("slurm.attach.node0", idle)
    third = await w.svc.attach_now(K)
    assert third.idle_streak == {"GPU-0": 1}                           # idle, fail, idle: no "2 in a row"
    prev = seen[-1]
    assert isinstance(prev, type(first)) and prev.nodes == first.nodes  # counters kept for the cpu delta
    assert w.state().live is third
