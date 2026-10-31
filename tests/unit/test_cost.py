"""Job cost, billing rates, the pre-submit estimate, free / charged partitions and the display rules (§5.2.11).

The lyra charging rules come from the anonymised field-probe fixtures (`scontrol_partitions_charges`, `sacctmgr_qos`): free
partitions allow only QOS low / guest (×0), `gpu` weighs `cpu=1,gres/gpu=32`, `gpuxl` `cpu=1,gres/gpu:L40S=32,
gres/gpu:RTX6000=64` (capitalised, while jobs carry `gres/gpu:rtx6000`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runfleet.backends.slurm.collector import parse_qos, partition_charges
from runfleet.models import (Capabilities, Job, JobKey, JobKind, JobState, PartitionCharge, ResourceAlloc,
                             UsageCell)
from runfleet.util import cost
from runfleet.util.slurmfmt import parse_tres
from tests.conftest import sample

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)
FREE_TRES = "billing=8,cpu=8,gres/gpu:rtx6000=1,gres/gpu=1,mem=48G,node=1"


def _caps(charges: tuple[PartitionCharge, ...] = (), factors: dict[str, float] | None = None,
          flags: str = "", assoc: tuple[tuple[str, str, str, str], ...] = ()) -> Capabilities:
    return Capabilities(remote_user="alice", uid=1000, login_host="login1", tz_offset=timedelta(0),
                        clock_skew=timedelta(0), tools=frozenset(), cluster_name="lyra",
                        slurm_config={"PriorityFlags": flags}, associations=assoc,
                        qos_usage_factor=factors or {}, partition_charges=charges)


@pytest.fixture(scope="module")
def lyra() -> Capabilities:
    factors = dict(parse_qos(sample("lyra", "sacctmgr_qos")).items)
    return _caps(partition_charges(sample("lyra", "scontrol_partitions_charges")), factors,
                 assoc=(("lyra", "pilab_02", "", "high,normal"), ("lyra", "alice07", "", "low,normal")))


@pytest.fixture(scope="module")
def k2() -> Capabilities:
    factors = dict(parse_qos(sample("k2", "sacctmgr_qos")).items)
    return _caps(partition_charges(sample("k2", "scontrol_partitions")), factors)


def _job(state: JobState = JobState.RUNNING, *, tres: str = FREE_TRES, partition: str = "free-gpu", qos: str = "low",
         elapsed: timedelta | None = timedelta(hours=4), limit: timedelta | None = timedelta(days=3),
         gpu_type: str | None = "rtx6000", cpus: int | None = 8, gpus: int = 1, jid: str = "31367626",
         array_task: str | None = None, kind: JobKind = JobKind.SLURM, observed: datetime | None = NOW,
         start: datetime | None = None, end: datetime | None = None) -> Job:
    alloc = ResourceAlloc(nodes=1, cpus=cpus, mem_bytes=None, gpus=gpus, gpu_type=gpu_type, tres=parse_tres(tres))
    return Job(key=JobKey("lyra", jid), kind=kind, name="mytask--train--s0--a01", user="alice07", state=state,
               raw_state=state.name, job_id_raw=jid, partition=partition, qos=qos, account="alice07", alloc=alloc,
               elapsed=elapsed, time_limit=limit, observed_at=observed, array_task=array_task, start_time=start,
               end_time=end)


# ---------------------------------------------------------------- job_cost: the real lyra / k2 numbers
def test_free_partition_job_on_qos_low_costs_nothing(lyra: Capabilities) -> None:
    c = cost.job_cost(_job(), NOW, lyra)
    assert c is not None
    assert c.billing == 8 and c.billing_basis == "tres"
    assert c.usage_factor == 0 and c.free and c.live
    assert c.cpu_h == pytest.approx(32) and c.gpu_h == pytest.approx(4)
    assert c.gpu_h_by_model == {"rtx6000": pytest.approx(4)}
    assert c.billing_h == pytest.approx(32) and c.charged_su == 0
    assert c.projected_su == 0                      # ×0 at the limit too


def test_gpuxl_rtx6000_is_cpus_plus_64_per_hour(lyra: Capabilities) -> None:
    # pending, no billing= in the TRES: the partition weights decide (typed weight spelled RTX6000, job rtx6000)
    tres = "cpu=8,gres/gpu:rtx6000=1,gres/gpu=1,mem=48G,node=1"
    job = _job(JobState.PENDING, tres=tres, partition="gpuxl", qos="normal", elapsed=None, limit=timedelta(hours=10))
    c = cost.job_cost(job, NOW, lyra)
    assert c is not None
    assert c.billing == 8 + 64 and c.billing_basis == "weights"
    assert c.billing_h == 0 and c.charged_su == 0 and c.cpu_h == 0 and not c.live and not c.free
    assert c.projected_su == pytest.approx(720)
    # the typed key missing (squeue sometimes prints only gres/gpu=1): the GPU model comes from alloc.gpu_type
    untyped = _job(JobState.PENDING, tres="cpu=8,gres/gpu=1,mem=48G,node=1", partition="gpuxl", qos="normal",
                   elapsed=None)
    assert cost.job_cost(untyped, NOW, lyra).billing == 72        # type: ignore[union-attr]
    l40s = _job(JobState.PENDING, tres="cpu=8,gres/gpu=1", partition="gpuxl", qos="high", gpu_type="l40s",
                elapsed=None, limit=timedelta(hours=1))
    c2 = cost.job_cost(l40s, NOW, lyra)
    assert c2 is not None and c2.billing == 40 and c2.usage_factor == 2 and c2.projected_su == 80
    # review: a pending squeue row often has no TRES at all; CPUs / GPUs from the resolved allocation still count
    bare = _job(JobState.PENDING, tres="", partition="gpuxl", qos="normal", elapsed=None, limit=timedelta(hours=10))
    c3 = cost.job_cost(bare, NOW, lyra)
    assert c3 is not None and c3.billing == 72 and c3.billing_basis == "weights"
    assert c3.projected_su == pytest.approx(720)


def test_as_billing_h_puts_unscaled_hours_in_the_su_slots(lyra: Capabilities) -> None:
    """Review: a usage-only cluster's `su` column shows billing-h; a ×0 QOS job there is 32 billing-h, not 0."""
    c = cost.job_cost(_job(), NOW, lyra)
    assert c is not None and c.charged_su == 0 and c.projected_su == 0
    b = c.as_billing_h()
    assert b.charged_su == pytest.approx(32) and b.projected_su == pytest.approx(8 * 72)


def test_k2_billing_equals_cpus_and_factor_one(k2: Capabilities) -> None:
    job = _job(tres="billing=16,cpu=16,gres/gpu=2,mem=256G,node=1", partition="gpu-preempt", qos="gpu-preempt",
               gpu_type="nvidia_a100-sxm4-40gb", gpus=2, cpus=16, elapsed=timedelta(hours=2))
    c = cost.job_cost(job, NOW, k2)
    assert c is not None
    assert c.billing == c.cpu_h / 2 == 16 and c.billing_h == c.cpu_h == 32       # billing-h = CPU-h here
    assert c.usage_factor == 1 and c.charged_su == 32
    assert c.gpu_h_by_model == {"nvidia_a100-sxm4-40gb": pytest.approx(4)}      # untyped TRES: alloc.gpu_type


def test_running_job_accrues_from_observed_at(lyra: Capabilities) -> None:
    job = _job(partition="gpu", qos="normal", tres="billing=40,cpu=8,gres/gpu=1,node=1",
               observed=NOW - timedelta(minutes=30))
    live = cost.job_cost(job, NOW, lyra)
    frozen = cost.job_cost(job, NOW, lyra, extrapolate=False)
    assert live is not None and frozen is not None
    assert live.billing_h == pytest.approx(40 * 4.5) and live.live
    assert frozen.billing_h == pytest.approx(40 * 4) and not frozen.live
    assert live.projected_su == pytest.approx(40 * 72)
    later = cost.job_cost(job, NOW + timedelta(hours=1), lyra)
    assert later is not None and later.charged_su == pytest.approx(40 * 5.5)


def test_running_job_without_elapsed_uses_start_time(lyra: Capabilities) -> None:
    job = _job(elapsed=None, start=NOW - timedelta(hours=2), tres="billing=4,cpu=4", gpus=0, gpu_type=None,
               partition="standard", qos="normal")
    c = cost.job_cost(job, NOW, lyra)
    assert c is not None and c.billing_h == pytest.approx(8) and c.gpu_h == 0 and c.gpu_h_by_model == {}


def test_ended_job_has_no_projection(lyra: Capabilities) -> None:
    job = _job(JobState.COMPLETED, partition="gpu", qos="high", tres="billing=40,cpu=8,gres/gpu=1",
               elapsed=timedelta(hours=1, minutes=30))
    c = cost.job_cost(job, NOW + timedelta(days=1), lyra)
    assert c is not None
    assert c.charged_su == pytest.approx(120) and c.billing_h == pytest.approx(60)   # high ×2
    assert c.projected_su is None and not c.live
    timed = _job(JobState.TIMEOUT, elapsed=None, start=NOW - timedelta(hours=3), end=NOW - timedelta(hours=1),
                 partition="gpu", qos="normal", tres="billing=40,cpu=8,gres/gpu=1")
    assert cost.job_cost(timed, NOW, lyra).billing_h == pytest.approx(80)   # type: ignore[union-attr]


def test_unknown_qos_counts_once_and_is_flagged(lyra: Capabilities) -> None:
    c = cost.job_cost(_job(qos="mystery", partition="gpu"), NOW, lyra)
    assert c is not None and c.usage_factor is None and not c.free
    assert c.charged_su == c.billing_h == pytest.approx(32)
    no_caps = cost.job_cost(_job(), NOW, None)
    assert no_caps is not None and no_caps.usage_factor is None and no_caps.billing_basis == "tres"


def test_billing_falls_back_to_cpus(lyra: Capabilities) -> None:
    # free-gpu has no weights: billing = CPUs (Slurm's default), labelled as an estimate
    c = cost.job_cost(_job(tres="cpu=4,gres/gpu=1", cpus=4), NOW, lyra)
    assert c is not None and c.billing == 4 and c.billing_basis == "cpus"
    nothing = cost.job_cost(_job(tres="", cpus=None, gpus=0), NOW, None)
    assert nothing is not None and nothing.billing is None and nothing.billing_basis is None
    assert nothing.cpu_h is None and nothing.billing_h is None and nothing.charged_su is None


def test_mig_slices_and_mixed_models_split_gpu_hours() -> None:
    tres = "billing=16,cpu=16,gres/gpu=3,gres/gpu:nvidia_a100_3g.20gb=1,gres/gpu:nvidia_a30_1g.6gb=1"
    c = cost.job_cost(_job(tres=tres, gpus=3, gpu_type=None, elapsed=timedelta(hours=2)), NOW, None)
    assert c is not None and c.gpu_h == 6
    assert c.gpu_h_by_model == {"nvidia_a100_3g.20gb": 2, "nvidia_a30_1g.6gb": 2, "": 2}   # one slot untyped
    assert cost.gpu_models(c)[-1] == ("", 2)


def test_multi_node_billing_is_never_multiplied_by_nodes() -> None:
    tres = "billing=64,cpu=64,gres/gpu:nvidia_h200_nvl=8,gres/gpu=8,mem=512G,node=2"
    c = cost.job_cost(_job(tres=tres, gpus=8, cpus=64, elapsed=timedelta(hours=1)), NOW, None)
    assert c is not None and c.billing_h == 64 and c.gpu_h == 8 and c.cpu_h == 64


def test_tasks_and_pending_array_ranges_have_no_cost() -> None:
    assert cost.job_cost(_job(kind=JobKind.TASK), NOW, None) is None
    rng = _job(JobState.PENDING, jid="31377076_[5-12%2]", array_task="[5-12%2]", elapsed=None)
    assert cost.job_cost(rng, NOW, None) is None
    element = _job(JobState.PENDING, jid="31377076_5", array_task="5", elapsed=None)
    assert cost.job_cost(element, NOW, None) is not None


# ---------------------------------------------------------------- billing_rate
def _charge(weights: str, name: str = "p", **kw: object) -> PartitionCharge:
    return PartitionCharge(partition=name, billing_weights=weights, **kw)   # type: ignore[arg-type]


def test_billing_rate_rules() -> None:
    gpu = _charge("cpu=1,gres/gpu=32")
    assert cost.billing_rate(gpu, {"cpu": 8, "gres/gpu": 1}, max_tres=False) == 40
    typed = _charge("cpu=1,gres/gpu:L40S=32,gres/gpu:RTX6000=64")
    assert cost.billing_rate(typed, {"cpu": 8, "gres/gpu": 1, "gres/gpu:rtx6000": 1}, max_tres=False) == 72
    assert cost.billing_rate(typed, {"cpu": 8, "gres/gpu": 2}, max_tres=False, gpu_type="L40S") == 72
    assert cost.billing_rate(typed, {"cpu": 8, "gres/gpu": 1}, max_tres=False) == 8      # model unknown: not counted
    assert cost.billing_rate(None, {"cpu": 6}, max_tres=False) == 6                       # no weights -> CPUs
    assert cost.billing_rate(_charge(""), {"cpu": 6, "gres/gpu": 1}, max_tres=True) == 6


def test_billing_rate_memory_units_and_case() -> None:
    w = _charge("CPU=1.0,Mem=0.25G,GRES/gpu=2.0")
    tres = cost.tres_counts({"cpu": "4", "mem": "64G", "gres/gpu": "1", "billing": "x"})
    assert tres == {"cpu": 4, "mem": 65536, "gres/gpu": 1}
    assert cost.billing_rate(w, tres, max_tres=False) == pytest.approx(4 + 16 + 2)
    assert cost.billing_rate(_charge("mem=2"), {"mem": 1024}, max_tres=False) == 2048     # bare: per MB
    assert cost.billing_rate(_charge("mem=1T"), {"mem": 1024 * 1024}, max_tres=False) == pytest.approx(1)
    assert cost.billing_rate(_charge("cpu=x,gres/gpu=4"), {"cpu": 4, "gres/gpu": 1}, max_tres=False) == 4


def test_billing_rate_max_tres() -> None:
    w = _charge("cpu=1,mem=0.25G,gres/gpu=32,license/matlab=5")
    tres = {"cpu": 8, "mem": 64 * 1024, "gres/gpu": 1, "license/matlab": 2}
    assert cost.billing_rate(w, tres, max_tres=True) == 32 + 10          # max node-level term + global ones
    assert cost.billing_rate(w, tres, max_tres=False) == 8 + 16 + 32 + 10
    assert cost.max_tres(_caps(flags="SMALL_RELATIVE_TO_TIME,MAX_TRES"))
    assert not cost.max_tres(_caps(flags="MAX_TRES_GRES")) and not cost.max_tres(None)


def test_partition_lookup_takes_the_first_of_a_list(lyra: Capabilities) -> None:
    assert cost.partition_charge(lyra, "gpuxl,free-gpuxl").partition == "gpuxl"   # type: ignore[union-attr]
    assert cost.partition_charge(lyra, "nope") is None and cost.partition_charge(None, "gpu") is None


# ---------------------------------------------------------------- classify_partitions
def test_lyra_free_and_charged_partitions(lyra: Capabilities) -> None:
    got = {c.partition: c for c in cost.classify_partitions(lyra)}
    for p in ("free", "free-gpu", "free-gpuxl"):
        assert not got[p].charged and got[p].basis == "allow_qos"
        assert got[p].qos == (("low", 0.0), ("guest", 0.0))
    assert got["gpu"].charged and got["gpu"].basis == "weights" and got["gpu"].weights == "cpu=1,gres/gpu=32"
    assert got["gpu"].qos == (("high", 2.0), ("normal", 1.0))
    assert got["gpuxl"].basis == "weights"
    assert got["standard"].charged and got["standard"].basis == "qos"


def test_partition_qos_alone_never_decides() -> None:
    # free-gpuxl's partition QOS is ×1 while its AllowQos are ×0: still free
    charge = _charge("", "free-gpuxl", qos="free-gpuxl-part", allow_qos=("low", "guest"))
    caps = _caps((charge,), {"low": 0, "guest": 0, "free-gpuxl-part": 1})
    (c,) = cost.classify_partitions(caps)
    assert not c.charged and c.basis == "allow_qos"


def test_free_observed_from_my_jobs_and_cpus_fallback() -> None:
    open_part = _charge("", "scavenge")                           # AllowQos ALL, no weights
    caps = _caps((open_part,), {"low": 0, "normal": 1}, assoc=(("lyra", "lab", "", "low"),))
    cell = UsageCell(name="x", account="lab", partition="scavenge", qos="low", gpu_model="", jobs=3, running=0,
                     cpu_h=1, gpu_h=0, billing_h=1, charged_su=0)
    (seen,) = cost.classify_partitions(caps, [cell])
    assert not seen.charged and seen.basis == "observed" and seen.qos == (("low", 0.0),)
    (blind,) = cost.classify_partitions(caps)
    assert blind.charged and blind.basis == "cpus"                # no evidence: billing = CPUs, charged
    paid = _caps((open_part,), {"low": 0, "normal": 1}, assoc=(("lyra", "lab", "", "low,normal"),
                                                               ("other", "lab", "", "high")))
    (q,) = cost.classify_partitions(paid)
    assert q.charged and q.basis == "qos" and q.qos == (("normal", 1.0),)


# ---------------------------------------------------------------- estimate
def test_estimate_gpuxl_l40s(lyra: Capabilities) -> None:
    recent = [_job(partition="gpuxl", qos="normal", jid="1"), _job(partition="gpu", qos="high", jid="2")]
    e = cost.estimate(lyra, partition="gpuxl", cpus=8, gpus=1, gpu_type="L40S", mem_gb=None, nodes=1, hours=24,
                      qos=None, recent=recent, partition_gpus={"free-gpuxl": ("l40s", "rtx6000"),
                                                               "free-gpu": ("v100",)})
    assert e.billing_per_h == 40 and e.billing_basis == "weights"
    assert e.billing_terms == (("cpu", 8, 1), ("gres/gpu:L40S", 1, 32))
    assert e.qos == "normal" and e.usage_factor == 1 and e.usage_factor_basis == "your last job on gpuxl"
    assert e.su == 960 and e.alternatives == ("free-gpuxl",)


def test_estimate_defaults_and_notes(lyra: Capabilities) -> None:
    e = cost.estimate(lyra, partition="gpu", cpus=None, gpus=2, gpu_type=None, mem_gb=None, nodes=1, hours=None,
                      qos=None)
    assert e.billing_per_h == 1 + 64                              # 1 cpu (no DefCpuPerGPU printed) + 2 × 32
    assert "1 cpu (1 per node: no --cpus, no DefCpuPerGPU)" in e.notes
    assert "no time: per hour only" in e.notes and e.su is None and e.hours is None
    assert e.qos == "normal" and e.usage_factor_basis == "default normal"
    assert e.alternatives == ()                                   # no partition GPU types given
    defcpu = _charge("cpu=1,gres/gpu=32", "g", def_cpu_per_gpu=8, def_mem_per_cpu_mb=4096, allow_qos=("gq",))
    caps = _caps((defcpu,), {"gq": 0.5})
    e2 = cost.estimate(caps, partition="g", cpus=None, gpus=1, gpu_type=None, mem_gb=None, nodes=1, hours=10,
                       qos=None)
    assert e2.billing_per_h == 40 and "8 cpus from DefCpuPerGPU 8" in e2.notes
    assert "memory from DefMemPerCPU 4096M" in e2.notes
    assert e2.qos == "gq" and e2.usage_factor_basis == "AllowQos" and e2.su == 200


def test_estimate_qos_flag_unknown_factor_and_no_weights(k2: Capabilities) -> None:
    e = cost.estimate(k2, partition="gpu-preempt", cpus=16, gpus=1, gpu_type=None, mem_gb=64, nodes=1, hours=2,
                      qos="nope")
    assert e.billing_basis == "cpus" and e.billing_per_h == 16 and e.billing_terms == (("cpu", 16, 1),)
    assert e.usage_factor is None and "UsageFactor of qos nope unknown: counted ×1" in e.notes and e.su == 32
    assert "no TRESBillingWeights on gpu-preempt: billing = CPUs" in e.notes
    bare = _caps((_charge("", "p"),))
    e3 = cost.estimate(bare, partition="p", cpus=2, gpus=0, gpu_type=None, mem_gb=None, nodes=1, hours=1, qos=None)
    assert e3.qos is None and e3.usage_factor_basis == "" and "QOS unknown: counted ×1" in e3.notes and e3.su == 2


def test_estimate_typed_weights_need_a_gpu_type(lyra: Capabilities) -> None:
    e = cost.estimate(lyra, partition="gpuxl", cpus=8, gpus=1, gpu_type=None, mem_gb=32, nodes=1, hours=1,
                      qos="normal")
    assert e.billing_per_h == 8 and "no --gpu-type: the typed GPU weights are not counted" in e.notes
    assert e.usage_factor_basis == "--qos"


def test_estimate_free_partition_and_unknown_partition(lyra: Capabilities) -> None:
    e = cost.estimate(lyra, partition="free-gpu", cpus=8, gpus=1, gpu_type="rtx6000", mem_gb=None, nodes=1,
                      hours=5, qos="low", partition_gpus={"free-gpuxl": ("rtx6000",)})
    assert e.su == 0 and e.alternatives == ()                     # already free: nothing cheaper
    with pytest.raises(ValueError):
        cost.estimate(lyra, partition="nope", cpus=1, gpus=0, gpu_type=None, mem_gb=None, nodes=1, hours=1, qos=None)
    with pytest.raises(ValueError):
        cost.estimate(lyra, partition="gpu,free", cpus=1, gpus=0, gpu_type=None, mem_gb=None, nodes=1, hours=1,
                      qos=None)


def test_estimate_mem_weights_and_max_tres() -> None:
    w = _charge("cpu=1,mem=0.25G,gres/gpu=32", "p")
    caps = _caps((w,), {"normal": 1}, flags="MAX_TRES")
    e = cost.estimate(caps, partition="p", cpus=4, gpus=1, gpu_type=None, mem_gb=256, nodes=1, hours=1, qos=None)
    assert e.billing_per_h == 64                                  # max(4, 64, 32)
    assert ("mem", 256, 0.25) in e.billing_terms
    assert any("MAX_TRES" in n for n in e.notes)
    e2 = cost.estimate(_caps((w,), {"normal": 1}), partition="p", cpus=4, gpus=0, gpu_type=None, mem_gb=None,
                       nodes=1, hours=1, qos=None)
    assert "no --mem: memory not counted" in e2.notes and e2.billing_per_h == 4


# ---------------------------------------------------------------- display (§5.2.11 "Display and precision")
@pytest.mark.parametrize(("value", "su", "hours"), [
    (None, "—", "—"), (0, "0", "0"), (0.01, "<0.1", "<0.1"), (0.05, "0.1", "0.1"), (2.94, "2.9", "2.9"),
    (9.94, "9.9", "9.9"), (9.96, "10", "10.0"), (10.5, "11", "10.5"), (99.96, "100", "100"), (298.4, "298", "298"),
    (97885.0, "97,885", "97,885"), (27554.4, "27,554", "27,554"), (float("nan"), "—", "—"),
])
def test_amount_formatting(value: float | None, su: str, hours: str) -> None:
    assert cost.fmt_su(value) == su
    assert cost.fmt_hours(value) == hours


def test_other_formatters() -> None:
    assert cost.fmt_su(None, dash="unknown") == "unknown"
    assert [cost.fmt_pct(v) for v in (0.3, 0.0, 9.96, 42.4, None)] == ["0.3%", "0.0%", "10%", "42%", "—"]
    assert [cost.fmt_rate(v) for v in (8, 40.0, 0.5, 0.25, 1234.5)] == ["8/h", "40/h", "0.5/h", "0.25/h", "1,234.5/h"]
    assert [cost.fmt_factor(v) for v in (0, 1.0, 2, 0.5, None)] == ["×0", "×1", "×2", "×0.5", "×1?"]
    assert cost.fmt_projection(576.2) == "≈ 576 SU at limit"
    assert cost.fmt_projection(576.2, "billing-h") == "≈ 576 billing-h at limit"
    assert cost.fmt_projection(None) == "—"
    assert cost.json_amount(2.345678) == 2.35 and cost.json_amount(0.333, 1) == 0.3
    assert cost.json_amount(None) is None and cost.json_amount(float("inf")) is None


# ---------------------------------------------------------------- usage-only clusters (SU shown as billing-h)
def test_usage_only_from_the_charging_rules_or_the_summary_mode(lyra: Capabilities, k2: Capabilities) -> None:
    assert not cost.usage_only(lyra)                     # weights and ×0 QOS: SU
    assert cost.usage_only(k2)                           # no weights, every factor 1: billing-h
    assert cost.has_weights(lyra) and not cost.has_weights(k2)
    assert not cost.usage_only(None)                     # unknown: SU (the factor shows ×1?)
    assert cost.usage_only(lyra, "usage_only") and not cost.usage_only(k2, "limits")   # the summary decides
    assert cost.su_unit(True) == "billing-h" and cost.su_unit(False) == "SU"
