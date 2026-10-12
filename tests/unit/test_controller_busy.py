"""[WP1] Busy Slurm controller backoff (§5.4): stderr / slow-section signals, ×2 … ×8, decay, host status."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from runfleet.core.poller import BUSY_DECAY, BUSY_MAX, ControllerBackoff
from runfleet.probe.section import SectionOutcome, SectionResult
from runfleet.transport.fake import result
from tests.unit.stubs import run_for, settle
from tests.unit.test_poller import SLURM, Rig

BUSY_ERR = "squeue: error: slurm_load_jobs error: Socket timed out on send/recv operation"


def r(name: str, ms: int | None = 100, stderr: str = "") -> SectionResult:
    return SectionResult(name, SectionOutcome.OK, 0, "x", stderr, ms)


def test_stderr_doubles_up_to_the_cap_and_decays() -> None:
    b = ControllerBackoff()
    assert b.observe({"slurm.squeue": r("slurm.squeue", stderr=BUSY_ERR)}) and b.factor == 2
    for _ in range(5):
        b.observe({"slurm.sinfo": r("slurm.sinfo", stderr="sinfo: error: Rate limit exceeded")})
    assert b.factor == BUSY_MAX
    for _ in range(BUSY_DECAY - 1):
        assert not b.observe({"slurm.squeue": r("slurm.squeue")})
    assert b.observe({"slurm.squeue": r("slurm.squeue")}) and b.factor == BUSY_MAX / 2
    assert not b.observe({"procfs.host": r("procfs.host", stderr=BUSY_ERR)})   # not a controller section
    assert not b.observe({})


def test_failed_rounds_do_not_count_as_recovery() -> None:
    b = ControllerBackoff()
    b.observe({"slurm.squeue": r("slurm.squeue", stderr=BUSY_ERR)})
    failed = SectionResult("slurm.squeue", SectionOutcome.TIMEOUT, None, "", "", None)
    for _ in range(BUSY_DECAY * 3):
        assert not b.observe({"slurm.squeue": failed})              # no answer: neither busier nor recovered
    assert b.factor == 2
    for _ in range(BUSY_DECAY):
        b.observe({"slurm.squeue": r("slurm.squeue")})
    assert b.factor == 1


def test_slow_section_counts_only_against_its_own_median() -> None:
    b = ControllerBackoff()
    b.observe({"slurm.sacct": r("slurm.sacct", ms=20000)})          # no history yet: never busy
    assert b.factor == 1
    for ms in (900, 1100, 1000):
        b.observe({"slurm.squeue": r("slurm.squeue", ms=ms)})
    b.observe({"slurm.squeue": r("slurm.squeue", ms=4000)})         # 4× the median but under 5 s
    assert b.factor == 1
    b.observe({"slurm.squeue": r("slurm.squeue", ms=6000)})         # > 3× median and > 5 s
    assert b.factor == 2
    b.observe({"slurm.squeue": r("slurm.squeue", ms=None)})         # no duration (no GNU date): no verdict
    assert b.factor == 2


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    rig = Rig(SLURM)
    rig.task = asyncio.ensure_future(rig.p.run())
    yield rig
    rig.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rig.task


async def test_poller_slows_the_slurm_cadence_and_reports_it(rig: Rig) -> None:
    await settle()
    assert rig.status.tick_interval == 15 and rig.status.controller_slowdown == 1
    rig.t.queue("slurm.squeue", result("slurm.squeue", "101 R", 0, BUSY_ERR))
    await run_for(rig.clock, 19)
    st = rig.status
    assert st.controller_slowdown == 2 and st.tick_interval == 30            # every Slurm group ×2
    assert st.state.value == "ok"                                            # answered: busy, not failing
    before = len(rig.polls())
    await run_for(rig.clock, 50)
    assert len(rig.polls()) - before <= 2                                    # ~30 s apart now, not 15
    await run_for(rig.clock, 30 * (BUSY_DECAY + 1))
    assert rig.status.controller_slowdown == 1 and rig.status.tick_interval == 15
