"""Job cost (§5.2.11): CPU-h, GPU-h, billing and charged SU of one job, billing rates, the pre-submit estimate, free /
charged partitions, and the display rules for all of them.

Pure and shared by the UI and the agent. Inputs are only the cached charging rules (`Capabilities.partition_charges`,
`qos_usage_factor`, `slurm_config` PriorityFlags) and the job's own TRES: nothing here runs a command. The cost is
computed on demand and never stored on `Job` (it changes every second and would break `job_material_eq`).

Units are never mixed: **billing** is Slurm's per-job billing TRES per hour of wall time (already the job total, never
× nodes); **billing-h** = billing × elapsed_h; **SU** = billing-h × UsageFactor of the job's QOS (what sshare and sbank
count; 1 SU = 60 billing-minutes). A QOS whose factor is unknown counts ×1 and shows as `×1?`.
"""

from __future__ import annotations

import dataclasses

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from runfleet.models import Capabilities, Job, JobKind, PartitionCharge, StateGroup, UsageCell
from runfleet.util.slurmfmt import nullish, parse_float, parse_mem_mib

NONE = "—"                        # unknown in the UI (the agent passes "unknown"); never 0
_GPU = "gres/gpu"
_TYPED = "gres/gpu:"
_UNITS = "KMGTP"
_WEIGHT = re.compile(r"^(?P<n>\d+(?:\.\d*)?|\.\d+)(?P<u>[KMGTP]?)$", re.IGNORECASE)
_MEGA_BASED = frozenset({"mem", "bb"})   # TRES counted in MB: a weight suffix is relative to M (`mem=0.25G` = per GiB)


@dataclass(frozen=True, slots=True)
class JobCost:
    cpu_h: float | None
    gpu_h: float | None
    gpu_h_by_model: Mapping[str, float]       # normalised gres type -> hours; "" = model unknown
    billing: float | None                     # per hour
    billing_basis: Literal["tres", "weights", "cpus"] | None   # tres = Slurm's own; weights / cpus = estimate
    usage_factor: float | None                # None = QOS unknown (sums use ×1, shown ×1?)
    qos: str | None
    billing_h: float | None                   # billing × elapsed_h
    charged_su: float | None                  # billing_h × factor
    projected_su: float | None                # ACTIVE / PENDING with a time limit: billing × limit_h × factor
    free: bool                                # factor == 0
    live: bool                                # ACTIVE: accrues with now
    projected_billing_h: float | None = None  # billing × limit_h (unscaled, as projected_su before the factor)

    def as_billing_h(self) -> JobCost:
        """This cost with billing-h in the SU slots: a usage-only cluster's `su` column / filter / sort show billing
        hours, not charged SU (a ×0 QOS job there is still 32 billing-h, as its detail says)."""
        return dataclasses.replace(self, charged_su=self.billing_h, projected_su=self.projected_billing_h)


@dataclass(frozen=True, slots=True)
class Estimate:                               # `agent cost` (§8.7)
    partition: str
    billing_per_h: float
    billing_terms: tuple[tuple[str, float, float], ...]   # (tres key, count, weight) — `gres/gpu:L40S 1×32`
    billing_basis: Literal["weights", "cpus"]
    qos: str | None
    usage_factor: float | None
    usage_factor_basis: str                   # "--qos" | "AllowQos" | "your last job on gpuxl" | "default normal" |
                                              # "AllowQos (all alike)" | ""
    hours: float | None
    su: float | None                          # billing_per_h × hours × factor
    alternatives: tuple[str, ...] = ()        # ≤ 2 free partitions with the requested GPU type
    notes: tuple[str, ...] = ()               # every assumption ("8 cpus from DefCpuPerGPU", "no time: per hour only")


@dataclass(frozen=True, slots=True)
class PartitionClass:
    partition: str
    charged: bool
    basis: Literal["allow_qos", "observed", "weights", "qos", "cpus"]
    weights: str = ""                         # raw TRESBillingWeights, for display
    qos: tuple[tuple[str, float], ...] = ()   # the QOS (and factors) the verdict rests on


@dataclass(frozen=True, slots=True)
class _Weight:
    key: str                                  # as the cluster spells it ("gres/gpu:L40S"), for display
    number: float                             # as printed (0.25 of `mem=0.25G`)
    scale: float                              # TRES count / scale = count in the weight's unit (1024 for G on mem)

    @property
    def per_unit(self) -> float:              # weight per TRES base unit (per MB for mem)
        return self.number / self.scale


# ---------------------------------------------------------------- TRES and weights
def tres_counts(raw: Mapping[str, str]) -> dict[str, float]:
    """A raw TRES dict (`Job.alloc.tres`) as numbers in Slurm's own units, keys lower-cased: `mem` in MiB (`48G` ->
    49152), everything else as printed. Unparsable values are left out."""
    out: dict[str, float] = {}
    for key, value in raw.items():
        k = key.strip().lower()
        n = parse_mem_mib(value) if k == "mem" else parse_float(value)
        if k and n is not None:
            out[k] = n
    return out


def parse_weights(text: str) -> list[_Weight]:
    """`TRESBillingWeights` (`CPU=1.0,Mem=0.25G,GRES/gpu:L40S=32`) -> weights; a malformed item is skipped. A unit
    suffix scales from the TRES base unit as Slurm does (MB for mem, none otherwise)."""
    out: list[_Weight] = []
    for item in (nullish(text) or "").split(","):
        key, eq, value = item.strip().partition("=")
        m = _WEIGHT.match(value.strip())
        if not eq or not key.strip() or m is None:
            continue
        base = 2 if key.strip().lower() in _MEGA_BASED else 0
        unit = _UNITS.index(m.group("u").upper()) + 1 if m.group("u") else base
        out.append(_Weight(key.strip(), float(m.group("n")), 1024.0 ** (unit - base)))
    return out


def _with_gpus(counts: Mapping[str, float], gpu_type: str | None) -> dict[str, float]:
    """Fill in the GPU keys a weight may ask for: the typed key from `gpu_type` when the TRES has only `gres/gpu`
    (squeue sometimes omits it), and `gres/gpu` from the typed keys when it is missing."""
    out = dict(counts)
    typed = {k: v for k, v in out.items() if k.startswith(_TYPED)}
    if _GPU in out and not typed and gpu_type:
        out[_TYPED + gpu_type.lower()] = out[_GPU]
    elif _GPU not in out and typed:
        out[_GPU] = sum(typed.values())
    return out


def _is_node_level(key: str) -> bool:
    return key in ("cpu", "mem", "node") or key.startswith("gres/")


def _terms(weights: Sequence[_Weight], counts: Mapping[str, float]) -> list[tuple[_Weight, float]]:
    """(weight, TRES count) for every weighted key the request has (keys compare case-insensitively)."""
    return [(w, counts[w.key.lower()]) for w in weights if counts.get(w.key.lower())]


def _combine(terms: Sequence[tuple[_Weight, float]], max_tres: bool) -> float:
    """Slurm's sum of weight × TRES; with PriorityFlags MAX_TRES the largest node-level term plus the global ones
    (licenses, burst buffers)."""
    if not max_tres:
        return sum(w.per_unit * n for w, n in terms)
    node = max((w.per_unit * n for w, n in terms if _is_node_level(w.key.lower())), default=0.0)
    return node + sum(w.per_unit * n for w, n in terms if not _is_node_level(w.key.lower()))


def billing_rate(charge: PartitionCharge | None, tres: Mapping[str, float], *, max_tres: bool,
                 gpu_type: str | None = None) -> float:
    """Slurm's rule: sum of weight × TRES over the weighted keys (PriorityFlags MAX_TRES: the max over node-level
    TRES plus the global ones); keys case-insensitive; `mem=0.25G` = per GiB; a typed GPU weight with only an untyped
    `gres/gpu` count uses gpu_type; no weights -> CPUs."""
    counts = {k.lower(): v for k, v in tres.items()}
    weights = parse_weights(charge.billing_weights) if charge is not None else []
    if not weights:
        return counts.get("cpu", 0.0)
    return _combine(_terms(weights, _with_gpus(counts, gpu_type)), max_tres)


def max_tres(caps: Capabilities | None) -> bool:
    """PriorityFlags has MAX_TRES (billing = the largest node-level term)."""
    if caps is None:
        return False
    flags = caps.slurm_config.get("PriorityFlags") or ""
    return "MAX_TRES" in {f.strip().upper() for f in flags.split(",")}


def partition_charge(caps: Capabilities | None, partition: str | None) -> PartitionCharge | None:
    """The charge of a partition (a pending job's `gpu,free-gpu` list: the first one), else None."""
    if caps is None or not partition:
        return None
    name = partition.split(",", 1)[0].strip()
    return next((c for c in caps.partition_charges if c.partition == name), None)


def usage_factor(caps: Capabilities | None, qos: str | None) -> float | None:
    """UsageFactor of the job's QOS (§5.2.11: the job QOS decides, never the partition QOS); None = unknown."""
    if caps is None or not qos:
        return None
    return caps.qos_usage_factor.get(qos)


def has_weights(caps: Capabilities | None) -> bool:
    """Some partition of the cluster has `TRESBillingWeights` (else billing = CPUs everywhere)."""
    return caps is not None and any(parse_weights(c.billing_weights) for c in caps.partition_charges)


def usage_only(caps: Capabilities | None, mode: str | None = None) -> bool:
    """The cluster charges no SU worth naming (§5.2.11 "usage only", k2 / vega): SU are shown as billing-h. ``mode`` is
    `AccountSummary.mode` when the summary has been read (it decides); without it the charging rules do: no billing
    weights and every known UsageFactor 1. Unknown caps -> False (the UI says SU, the factor shows ×1?)."""
    if mode is not None:
        return mode == "usage_only"
    if caps is None:
        return False
    return not has_weights(caps) and all(f == 1 for f in caps.qos_usage_factor.values())


def su_unit(only_usage: bool) -> str:
    """The unit of the job-cost amount: `SU`, or `billing-h` on a usage-only cluster."""
    return "billing-h" if only_usage else "SU"


# ---------------------------------------------------------------- job cost
def _elapsed_h(job: Job, now: datetime, extrapolate: bool) -> float | None:
    """Ended: Elapsed; ACTIVE: Elapsed + (now − observed_at) like the ELAPSED column; pending: 0."""
    group = job.state.group
    if group is StateGroup.QUEUED:
        return 0.0
    active = group is StateGroup.ACTIVE
    if job.elapsed is not None:
        s = job.elapsed.total_seconds()
        if active and extrapolate and job.observed_at is not None:
            s += max(0.0, (now - job.observed_at).total_seconds())
        return s / 3600
    if job.start_time is None or job.est_start:
        return None
    end = (now if extrapolate else job.observed_at) if active else job.end_time
    return None if end is None else max(0.0, (end - job.start_time).total_seconds()) / 3600


def _gpu_split(counts: Mapping[str, float], job: Job) -> tuple[float, dict[str, float]]:
    """GPU count and per-model counts: the typed keys (lower-case; a MIG slice is a model of its own), the rest of
    `gres/gpu` under `alloc.gpu_type`, else under "" (model unknown)."""
    typed: dict[str, float] = {}
    for k, v in counts.items():
        if k.startswith(_TYPED) and v > 0:
            model = k[len(_TYPED):]
            typed[model] = typed.get(model, 0.0) + v
    total = counts.get(_GPU)
    if total is None:
        total = sum(typed.values()) if typed else float(job.alloc.gpus)
    rest = total - sum(typed.values())
    if rest > 1e-9:
        model = "" if typed else (job.alloc.gpu_type or "").lower()
        typed[model] = typed.get(model, 0.0) + rest
    return total, typed


def _with_alloc(counts: Mapping[str, float], job: Job) -> dict[str, float]:
    """The TRES counts with what the TRES string lacks taken from the resolved allocation (a pending job's squeue row
    often has no `tres-alloc`, yet CPUs / GPUs / memory / nodes are known)."""
    out = dict(counts)
    a = job.alloc
    if "cpu" not in out and a.cpus is not None:
        out["cpu"] = float(a.cpus)
    if "mem" not in out and a.mem_bytes is not None:
        out["mem"] = a.mem_bytes / 2**20
    if "node" not in out and a.nodes is not None:
        out["node"] = float(a.nodes)
    if a.gpus and _GPU not in out and not any(k.startswith(_TYPED) for k in out):
        out[_GPU] = float(a.gpus)
    return out


def job_cost(job: Job, now: datetime, caps: Capabilities | None, *, extrapolate: bool = True) -> JobCost | None:
    """None for workstation tasks and pending array ranges."""
    if job.kind is not JobKind.SLURM:
        return None
    group = job.state.group
    if group is StateGroup.QUEUED and "[" in (job.array_task or job.key.id):
        return None
    counts = tres_counts(job.alloc.tres)
    cpus = counts.get("cpu", float(job.alloc.cpus) if job.alloc.cpus is not None else None)
    gpus, by_model = _gpu_split(counts, job)
    billing: float | None = counts.get("billing")
    basis: Literal["tres", "weights", "cpus"] | None = "tres" if billing is not None else None
    if billing is None:
        charge = partition_charge(caps, job.partition)
        if charge is not None and parse_weights(charge.billing_weights):
            billing = billing_rate(charge, _with_alloc(counts, job), max_tres=max_tres(caps),
                                   gpu_type=job.alloc.gpu_type)
            basis = "weights"
        elif cpus is not None:
            billing, basis = cpus, "cpus"
    factor = usage_factor(caps, job.qos)
    mult = 1.0 if factor is None else factor
    hours = _elapsed_h(job, now, extrapolate)
    billing_h = billing * hours if billing is not None and hours is not None else None
    projected_bh = None
    if group in (StateGroup.ACTIVE, StateGroup.QUEUED) and job.time_limit is not None and billing is not None:
        projected_bh = billing * job.time_limit.total_seconds() / 3600
    projected = projected_bh * mult if projected_bh is not None else None
    return JobCost(
        cpu_h=cpus * hours if cpus is not None and hours is not None else None,
        gpu_h=gpus * hours if hours is not None else None,
        gpu_h_by_model={m: n * hours for m, n in by_model.items()} if hours is not None else {},
        billing=billing, billing_basis=basis, usage_factor=factor, qos=job.qos, billing_h=billing_h,
        charged_su=billing_h * mult if billing_h is not None else None, projected_su=projected,
        free=factor == 0, live=group is StateGroup.ACTIVE and extrapolate, projected_billing_h=projected_bh)


# ---------------------------------------------------------------- partitions
def _assoc_qos(caps: Capabilities, partition: str) -> set[str]:
    """QOS names the user's associations of this cluster allow on a partition (or on all partitions)."""
    out: set[str] = set()
    for cluster, _account, part, qos in caps.associations:
        if cluster and caps.cluster_name and cluster != caps.cluster_name:
            continue
        if part and partition not in part.split(","):
            continue
        out.update(q for q in qos.split(",") if q)
    return out


def classify_partitions(caps: Capabilities, cells: Sequence[UsageCell] = ()) -> list[PartitionClass]:
    """Free or charged, per `caps.partition_charges` in scontrol order (§5.2.11 rule 5), first match wins: AllowQos
    holds only ×0 QOS -> free; my jobs there in the window all ran under ×0 QOS -> free (observed); weights, or a QOS
    that may run there with factor > 0 -> charged; else charged with billing = CPUs. The partition QOS factor alone
    never decides (lyra `free-gpuxl-part` is ×1, yet its jobs are not charged)."""
    factors = caps.qos_usage_factor
    out: list[PartitionClass] = []
    for c in caps.partition_charges:
        p, w = c.partition, c.billing_weights
        allow = c.allow_qos or ()
        if allow and all(factors.get(q) == 0 for q in allow):
            out.append(PartitionClass(p, False, "allow_qos", w, tuple((q, 0.0) for q in allow)))
            continue
        seen = sorted({cell.qos for cell in cells if cell.partition == p and cell.jobs > 0})
        if seen and all(factors.get(q) == 0 for q in seen):
            out.append(PartitionClass(p, False, "observed", w, tuple((q, 0.0) for q in seen)))
            continue
        may = set(allow) if allow else _assoc_qos(caps, p)
        paid = tuple(sorted((q, f) for q in may if (f := factors.get(q)) is not None and f > 0))
        if parse_weights(w):
            out.append(PartitionClass(p, True, "weights", w, paid))
        elif paid:
            out.append(PartitionClass(p, True, "qos", w, paid))
        else:
            out.append(PartitionClass(p, True, "cpus", w))
    return out


# ---------------------------------------------------------------- pre-submit estimate
def _num(value: float) -> str:
    """Up to two decimals, trailing zeros dropped: 8, 0.5, 0.25, 1,234."""
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _when(job: Job) -> float:
    t = job.submit_time or job.start_time or job.end_time
    return t.timestamp() if t is not None else -math.inf


def _default_qos(caps: Capabilities, charge: PartitionCharge, qos: str | None,
                 recent: Sequence[Job]) -> tuple[str | None, str]:
    """`--qos`; the partition's AllowQos when it holds one QOS; the QOS of my most recent job there; `normal` if it
    exists (and AllowQos, when set, allows it); AllowQos's first QOS when all of its QOS charge alike (lyra `free*`:
    `low,guest`, both ×0); unknown."""
    if qos:
        return qos, "--qos"
    allow = charge.allow_qos
    if allow is not None and len(allow) == 1:
        return allow[0], "AllowQos"
    p = charge.partition
    mine = [j for j in recent if j.kind is JobKind.SLURM and j.is_mine and j.qos
            and p in (j.partition or "").split(",") and (allow is None or j.qos in allow)]
    if mine:
        last = max(mine, key=_when)
        return last.qos, f"your last job on {p}"
    if "normal" in caps.qos_usage_factor and (allow is None or "normal" in allow):
        return "normal", "default normal"
    factors = {caps.qos_usage_factor.get(q) for q in allow or ()}
    if allow and len(factors) == 1 and None not in factors:
        return allow[0], "AllowQos (all alike)"
    return None, ""


def estimate(caps: Capabilities, *, partition: str, cpus: int | None, gpus: int, gpu_type: str | None,
             mem_gb: float | None, nodes: int, hours: float | None, qos: str | None,
             recent: Sequence[Job] = (), partition_gpus: Mapping[str, Collection[str]] | None = None) -> Estimate:
    """Pre-submit calculator: billing per hour and its terms, SU for `hours`, the factor and its basis, free
    alternatives, every assumption as text. `partition_gpus` (partition -> GPU types, from the snapshot's
    partitions) is what finds the free alternatives; without it there are none. Raises ValueError for a partition
    the charging rules do not know (the agent answers `not_found` before calling)."""
    charge = partition_charge(caps, partition)
    if charge is None or charge.partition != partition:
        raise ValueError(f"unknown partition {partition!r}")
    notes: list[str] = []
    nodes = max(1, nodes)
    if cpus is None:
        if gpus and charge.def_cpu_per_gpu:
            cpus = gpus * charge.def_cpu_per_gpu
            notes.append(f"{cpus} cpus from DefCpuPerGPU {charge.def_cpu_per_gpu}")
        else:
            cpus = nodes
            notes.append(f"{cpus} cpu{'s' if cpus != 1 else ''} (1 per node: no --cpus"
                         + (", no DefCpuPerGPU)" if gpus else ")"))
    counts: dict[str, float] = {"cpu": float(cpus), "node": float(nodes)}
    if mem_gb is not None:
        counts["mem"] = mem_gb * 1024
    elif charge.def_mem_per_cpu_mb:
        counts["mem"] = float(cpus * charge.def_mem_per_cpu_mb)
        notes.append(f"memory from DefMemPerCPU {charge.def_mem_per_cpu_mb}M")
    if gpus:
        counts[_GPU] = float(gpus)
    weights = parse_weights(charge.billing_weights)
    basis: Literal["weights", "cpus"]
    if weights:
        terms = _terms(weights, _with_gpus(counts, gpu_type))
        mx = max_tres(caps)
        per_h = _combine(terms, mx)
        billing_terms = tuple((w.key, n / w.scale, w.number) for w, n in terms)
        basis = "weights"
        keys = {w.key.lower() for w in weights}
        if "mem" in keys and "mem" not in counts:
            notes.append("no --mem: memory not counted")
        if gpus and not gpu_type and _GPU not in keys and any(k.startswith(_TYPED) for k in keys):
            notes.append("no --gpu-type: the typed GPU weights are not counted")
        if mx:
            notes.append("PriorityFlags MAX_TRES: billing = the largest per-node term")
    else:
        per_h, billing_terms, basis = float(cpus), (("cpu", float(cpus), 1.0),), "cpus"
        notes.append(f"no TRESBillingWeights on {partition}: billing = CPUs")
    q, q_basis = _default_qos(caps, charge, qos, recent)
    factor = caps.qos_usage_factor.get(q) if q else None
    if q is None:
        notes.append("QOS unknown: counted ×1")
    elif factor is None:
        notes.append(f"UsageFactor of qos {q} unknown: counted ×1")
    su = per_h * hours * (1.0 if factor is None else factor) if hours is not None else None
    if hours is None:
        notes.append("no time: per hour only")
    alternatives: tuple[str, ...] = ()
    if factor != 0 and per_h > 0:
        want = gpu_type.lower() if gpu_type else None
        free = [c.partition for c in classify_partitions(caps) if not c.charged and c.partition != partition]
        if gpus:
            types = partition_gpus or {}
            free = [p for p in free if (want in {t.lower() for t in types.get(p, ())} if want else types.get(p))]
        alternatives = tuple(free[:2])
    return Estimate(partition=partition, billing_per_h=per_h, billing_terms=billing_terms, billing_basis=basis,
                    qos=q, usage_factor=factor, usage_factor_basis=q_basis, hours=hours, su=su,
                    alternatives=alternatives, notes=tuple(notes))


# ---------------------------------------------------------------- display (§5.2.11 "Display and precision")
def _half_up(value: float, decimals: int = 0) -> float:
    """Round half away from zero (`f"{2.5:.0f}"` would give 2)."""
    q = 10 ** decimals
    return math.copysign(math.floor(abs(value) * q + 0.5) / q, value)


def _amount(value: float | None, big: float, dash: str) -> str:
    if value is None or not math.isfinite(value):
        return dash
    if value < 0:
        return "-" + _amount(-value, big, dash)
    if value == 0:
        return "0"
    if value < 0.05:
        return "<0.1"
    one = _half_up(value, 1)
    if one >= big:
        return f"{int(_half_up(value)):,}"
    return f"{one:.1f}"


def fmt_su(value: float | None, *, dash: str = NONE) -> str:
    """SU and billing-h: `97,885` (≥ 10, thousands separators as sbank prints them), `2.9`, `<0.1`, `0`."""
    return _amount(value, 10, dash)


def fmt_hours(value: float | None, *, dash: str = NONE) -> str:
    """CPU-h and GPU-h: `27,554` (≥ 100), `23.2`, `<0.1`, `0`."""
    return _amount(value, 100, dash)


def fmt_pct(value: float | None, *, dash: str = NONE) -> str:
    """Used share of a limit: `0.3%` below 10 %, `42%` above."""
    if value is None or not math.isfinite(value):
        return dash
    one = _half_up(value, 1)
    return f"{one:.1f}%" if abs(one) < 10 else f"{int(_half_up(value))}%"


def fmt_rate(value: float | None, *, dash: str = NONE) -> str:
    """Billing per hour: `8/h`, `40/h`, `0.5/h`."""
    return dash if value is None or not math.isfinite(value) else f"{_num(value)}/h"


def fmt_factor(value: float | None) -> str:
    """UsageFactor: `×0`, `×1`, `×2`, `×0.5`; unknown `×1?` (counted ×1)."""
    return "×1?" if value is None or not math.isfinite(value) else f"×{_num(value)}"


def fmt_projection(value: float | None, unit: str = "SU", *, dash: str = NONE) -> str:
    """`≈ 576 SU at limit` (at most, if the job runs to its time limit)."""
    return dash if value is None else f"≈ {fmt_su(value)} {unit} at limit"


def json_amount(value: float | None, decimals: int = 2) -> float | None:
    """JSON number: SU / hours with 2 decimals, used % with 1; unknown stays null (never 0)."""
    if value is None or not math.isfinite(value):
        return None
    return round(value, decimals)


def gpu_models(cost: JobCost) -> list[tuple[str, float]]:
    """GPU-h by model, largest first ("" = model unknown last among equals)."""
    return sorted(cost.gpu_h_by_model.items(), key=lambda kv: (-kv[1], kv[0] == "", kv[0]))

