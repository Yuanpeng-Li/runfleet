"""Resubmissions in the demo world (§5.2.12, §10.6): lyra's PREEMPTED job (PreemptMode CANCEL) has a same-name twin
submitted 13 minutes after it ended, which COMPLETED; through the real collector the two link, so `--demo` shows
"resubmitted as …" on the old one and "resumes …" on the new one."""

from __future__ import annotations

from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.models import JobKey, JobState
from runfleet.util.lineage import link
from tests.conftest import CLOCK_START

OLD = JobKey("lyra", "32354789")
NEW = JobKey("lyra", "32355120")


def test_the_twin_is_submitted_after_the_preemption_and_completes(world: DemoWorld) -> None:
    jobs = world.cluster("lyra").jobs
    old, new = jobs[int(OLD.id)], jobs[int(NEW.id)]
    assert (old.state, new.state) == ("PREEMPTED", "COMPLETED") and old.name == new.name
    assert old.end is not None and new.start is not None and new.submit > old.end and new.start >= old.end
    assert new.submit - old.end == 13 * 60


async def test_the_collected_snapshot_links_them() -> None:
    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    svc = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
    svc.lock_path = False
    await svc.collect_once(["lyra"])
    snap = svc.store.hosts["lyra"].snapshot
    assert snap is not None
    assert snap.jobs[OLD.id].state is JobState.PREEMPTED and snap.jobs[NEW.id].state is JobState.COMPLETED
    links = link(snap.jobs.values())
    nxt = links.next_of(OLD)
    prev = links.prev_of(NEW)
    assert nxt is not None and nxt.key == NEW and prev is not None and prev.key == OLD
    assert [k.id for k in links.successor] == [OLD.id]          # the scenario's only resubmission
