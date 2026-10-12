"""[WP3] FakeService attach (§7.16) behaves like the contract: vega ssh live data, lyra srun consent, readonly."""

from __future__ import annotations

import pytest

from runfleet.errors import AttachBlocked, AttachConsentRequired
from runfleet.models import AttachStatus, JobKey
from tests.ui.fakes import FakeService

VEGA = JobKey("vega", "723954")


async def test_vega_ssh_attach_shows_idle_gpu_after_two_samples() -> None:
    svc = FakeService()
    seen: list[frozenset[str]] = []
    svc.store.subscribe(lambda cs: seen.append(cs.live))
    st = svc.attach_start(VEGA)
    assert st.status is AttachStatus.LIVE and st.mode == "ssh" and st.refresh_s == 15
    assert st.live is not None and st.live.hints == ()
    live = await svc.attach_now(VEGA)
    gpu1 = live.nodes[0].gpus[1]
    assert gpu1.util_pct == 0 and gpu1.mem_used_mib == 84725 and gpu1.slurm_idx == 1
    assert live.hints and live.nodes[0].cgroups["723954"].cpu_cores == 3.6
    assert frozenset({"723954"}) in seen
    svc.attach_stop(VEGA)
    state = svc.attach_state(VEGA)
    assert state is not None and state.status is AttachStatus.STOPPED and state.live is live


async def test_lyra_srun_needs_consent_and_counts_rows() -> None:
    svc = FakeService()
    key = svc.add_lyra_running_job()
    st = svc.attach_start(key)
    assert st.status is AttachStatus.NEEDS_CONSENT and st.offer is not None and "sacct" in st.offer.body
    with pytest.raises(AttachConsentRequired) as e:
        await svc.attach_now(key)
    assert e.value.offer.node == "s71-02"
    live = await svc.attach_now(key, consent="job")
    assert live.mode == "srun" and live.sacct_rows_added == 1
    await svc.attach_now(key)                    # allowed for this job for the session now
    state = svc.attach_state(key)
    assert state is not None and state.consent == "job" and state.srun_rows == 2


async def test_readonly_blocks_srun_but_not_ssh() -> None:
    svc = FakeService(readonly=True)
    key = svc.add_lyra_running_job()
    st = svc.attach_start(key)
    assert st.status is AttachStatus.BLOCKED and "read-only" in (st.blocked or "")
    with pytest.raises(AttachBlocked):
        await svc.attach_now(key, consent="once")
    assert svc.attach_start(VEGA).status is AttachStatus.LIVE


def test_k2_ssh_lands_in_the_newest_job() -> None:
    svc = FakeService()
    st = svc.attach_start(JobKey("k2", "2332861"))
    assert st.live is not None
    node = st.live.nodes[0]
    assert node.landed_job == "2332863" and node.gpus_of_target is False and "2332861" in node.cgroups
