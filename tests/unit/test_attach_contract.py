"""[WP1] Attach (§7.16) contract: config keys, model defaults and JSON, collector defaults, store notify, stubs."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from runfleet import models as m
from runfleet.backends.base import AttachPlan, HostContext
from runfleet.config import ATTACH_SETTINGS, GeneralSettings, HostConfig, SlurmSettings, parse_config
from runfleet.core.store import ChangeSet, FleetStore
from runfleet.errors import AttachBlocked, AttachConsentRequired, AttachError, ConfigError, RunFleetError
from runfleet.probe import framing
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.unit.stubs import StubProcess, StubSlurm, stub_caps

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


# ---------------------------------------------------------------- config
def test_attach_defaults() -> None:
    s = SlurmSettings()
    assert s.attach == "ssh+srun" and s.attach_refresh == 15
    assert ATTACH_SETTINGS == ("ssh", "ssh+srun", "srun", "off")


def test_attach_defaults_table_and_per_host_override() -> None:
    cfg, warnings = parse_config({
        "defaults": {"slurm": {"attach": "ssh+srun", "attach_refresh": "30s"}},
        "host": [{"name": "lyra", "attach": "srun"}, {"name": "k2"}, {"name": "vega", "attach": "off"}],
    })
    got = {h.name: (h.slurm.attach, h.slurm.attach_refresh) for h in cfg.hosts}
    assert got == {"lyra": ("srun", 30.0), "k2": ("ssh+srun", 30.0), "vega": ("off", 30.0)}
    assert warnings == []


def test_attach_refresh_floor_is_clamped_with_a_warning() -> None:
    cfg, warnings = parse_config({"host": [{"name": "k2", "attach_refresh": "5s"}]})
    assert cfg.hosts[0].slurm.attach_refresh == 10.0
    assert any("attach_refresh" in w and "floor" in w for w in warnings)


@pytest.mark.parametrize("table", [{"attach": "exporter"}, {"attach": "auto"}, {"attach": True},
                                   {"attach_refresh": 15}])
def test_bad_attach_values_are_errors(table: dict[str, object]) -> None:
    with pytest.raises(ConfigError) as e:
        parse_config({"host": [{"name": "k2", **table}]})
    assert e.value.key_path.startswith("host[0].attach")


# ---------------------------------------------------------------- models
def _gpu() -> m.GPU:
    return m.GPU(host="vega", index=1, uuid="GPU-b1", name="NVIDIA H200", util_pct=0.0, mem_used_mib=84725.0,
                 mem_total_mib=143771.0, mem_util_pct=None, temp_c=34.0, power_w=94.0, power_limit_w=600.0,
                 fan_pct=None, pstate="P2", source="nvidia-smi")


def test_existing_constructors_keep_working_with_new_defaults() -> None:
    gpu = _gpu()
    assert gpu.bus_id is None and gpu.slurm_idx is None
    caps = stub_caps(HostConfig("k2", "k2"))
    assert caps.node_ssh is None
    job = m.Job(key=m.JobKey("k2", "1"), kind=m.JobKind.SLURM, name="n", user="u", state=m.JobState.RUNNING,
                raw_state="RUNNING", job_id_raw="1")
    detail = m.JobDetail(job=job, fields={}, per_node=(), steps=(), logs=(), gpu=None, procs=(), procs_note="",
                         env={}, explanation=(), fetched_at=T0)
    assert detail.live is None
    assert ChangeSet("k2").live == frozenset()


def test_job_live_round_trips_through_json() -> None:
    key = m.JobKey("vega", "723954")
    gpu = dataclasses.replace(_gpu(), bus_id="00000000:C5:00.0", slurm_idx=1)
    node = m.NodeInspect(node="n-38-01", target_job="723954", landed_job="723954", gpus_of_target=True,
                         gpus=(gpu,), step_of_pid={129361: "723954.batch"},
                         cgroups={"723954": m.CgroupStats(mem_current=1, mem_max=None, shmem=2, oom_kill=0,
                                                          cpu_usage_usec=3, cpu_cores=3.6)},
                         stray_nvidia_smi=0, cost_ms=610)
    live = m.JobLive(key=key, mode="ssh", sampled_at=T0, nodes=(node,), cost_ms=670, clean=True,
                     audit_detail="clean", hints=("GPU1 idle (0%) but holding 84.7G",), idle_streak={"GPU-b1": 2})
    assert live.has_data and not m.JobLive(key=key, mode="srun", sampled_at=T0).has_data
    assert m.from_json(m.JobLive, m.to_json(live)) == live
    state = m.AttachState(key=key, status=m.AttachStatus.LIVE, setting="ssh", mode="ssh", live=live,
                          refresh_s=15.0, next_at=T0 + timedelta(seconds=15))
    assert m.from_json(m.AttachState, m.to_json(state)) == state
    caps = dataclasses.replace(stub_caps(HostConfig("k2", "k2")), node_ssh="denied")
    assert m.from_json(m.Capabilities, m.to_json(caps)).node_ssh == "denied"


def test_attach_errors() -> None:
    assert issubclass(AttachBlocked, AttachError) and issubclass(AttachConsentRequired, AttachError)
    assert issubclass(AttachError, RunFleetError)
    e = AttachBlocked("lyra refuses ssh to compute nodes", hint='attach = "ssh+srun"')
    assert str(e) == e.reason and e.hint
    offer = m.AttachOffer(node="s71-02", reason="lyra refuses ssh to compute nodes", title="t", body="b",
                          preview="srun …")
    c = AttachConsentRequired(m.JobKey("lyra", "31367626"), offer)
    assert c.offer is offer and str(c) == offer.reason


# ---------------------------------------------------------------- collectors, store, framing
def _ctx(host: HostConfig, **caps: object) -> HostContext:
    return HostContext(host=host, general=GeneralSettings(), caps=stub_caps(host, **caps), now=T0)


def _job(state: m.JobState = m.JobState.RUNNING) -> m.Job:
    return m.Job(key=m.JobKey("k2", "7"), kind=m.JobKind.SLURM, name="n", user="me", state=state, raw_state="",
                 job_id_raw="7", nodes=("n-40-01",))


def test_non_slurm_scheduler_cannot_attach_by_default() -> None:
    host = HostConfig("lynx", "lynx", backend="workstation")
    coll = StubProcess(host)
    assert coll.attach_modes(_job(), _ctx(host)) == ()
    with pytest.raises(AttachBlocked):
        coll.plan_attach(_job(), "ssh", _ctx(host))


def test_stub_slurm_attach_round_trip_and_caps_patch() -> None:
    host = HostConfig("k2", "k2")
    coll = StubSlurm(host)
    ctx = _ctx(host)
    assert coll.attach_modes(_job(), ctx) == ("ssh", "srun")
    assert coll.attach_modes(_job(), _ctx(host, node_ssh="denied")) == ("srun",)
    assert coll.attach_modes(_job(m.JobState.PENDING), ctx) == ()
    plan = coll.plan_attach(_job(), "ssh", ctx)
    assert isinstance(plan, AttachPlan) and plan.nodes == ("n-40-01",) and plan.sacct_rows == 0
    assert [s.name for s in plan.sections] == ["slurm.attach.node0", "slurm.attach.audit"]
    assert set(plan.inner) == {"slurm.attach.node0"}
    framing.validate(framing.make_script(plan.sections, "attach"))
    ok = SectionResult("slurm.attach.node0", SectionOutcome.OK, 0, "landed=7\ngpu 0 0 84725", "", 610)
    audit = SectionResult("slurm.attach.audit", SectionOutcome.OK, 0, "clean", "", 20)
    live, patch = coll.ingest_attach(plan, {ok.name: ok, audit.name: audit}, ctx)
    assert live.has_data and live.clean is True and live.nodes[0].gpus_of_target is True
    assert patch == {"node_ssh": "adopt"}
    denied = SectionResult("slurm.attach.node0", SectionOutcome.FAILED, 255, "",
                           "Permission denied (publickey,hostbased).", 90)
    live, patch = coll.ingest_attach(plan, {denied.name: denied}, ctx)
    assert not live.has_data and live.errors and patch == {"node_ssh": "denied"}
    srun = coll.plan_attach(_job(), "srun", ctx)
    assert srun.sacct_rows == 1


def test_store_notify_live() -> None:
    store = FleetStore([HostConfig("k2", "k2")])
    seen: list[ChangeSet] = []
    store.subscribe(seen.append)
    store.notify_live("k2", ["2332861"])
    store.notify_live("k2", [])                 # nothing changed: no notification
    store.notify_live("gone", ["1"])            # unknown host: ignored
    assert seen == [ChangeSet("k2", live=frozenset({"2332861"}))]


def test_framing_renders_nested_node_script_without_a_temp_dir() -> None:
    from runfleet.probe.section import Section
    script = framing.make_script([Section("node.id", "hostname -s", timeout=5)], "attach")
    text = framing.render(script, tmp=False)
    assert "mktemp" not in text and "\nRF_T=\n" in text
    assert framing.render(script) == text                     # purpose "attach": no temp dir by default (§7.16.4)
    assert "mktemp" in framing.render(script, tmp=True)
    assert "mktemp" in framing.render(framing.make_script(script.sections, "poll"))
    assert script.purpose == "attach"


async def test_fleet_service_attach_placeholder_until_phase_2() -> None:
    from runfleet.config import FleetConfig
    from runfleet.core.clock import ManualClock
    from runfleet.core.service import FleetService

    def no_transport(h: HostConfig) -> object:
        raise AssertionError("attach placeholder must not open a transport")

    svc = FleetService(FleetConfig(hosts=(HostConfig("k2", "k2"),)), no_transport,   # type: ignore[arg-type]
                       ManualClock(T0), None)
    key = m.JobKey("k2", "1")
    assert svc.attach_state(key) is None
    st = svc.attach_start(key)
    assert st.status is m.AttachStatus.BLOCKED and st.setting == "ssh+srun" and st.blocked
    svc.attach_stop()
    with pytest.raises(AttachBlocked):
        await svc.attach_now(key)
