"""[WP2] Pending priority evidence (§5.2.10): "why isn't my job starting?" answered with what Slurm itself says.

Pure. The cluster's priority settings come from the `slurm.config` capability section (`PriorityType`,
`PriorityWeight*`, `PriorityFlags`, `PriorityDecayHalfLife`, `PriorityMaxAge`, `PriorityUsageResetPeriod`); the job's
weighted factors from `sprio` and the user's fair share from `sshare` (cadence group `slurm.priority`). The result is
evidence — the factors as the cluster weights them, which one dominates, the fair share per account and plain-language
hints — never a start time or a queue position.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from runfleet.backends.common import ParseResult
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.states import NEVER_RUNS, humanize_reason, reason_key
from runfleet.models import FairShare, Job, PendingPriority, PriorityFactor
from runfleet.util.slurmfmt import nullish, parse_duration, parse_float, parse_int, split_rows

# factor -> `scontrol show config` key of its weight, in sprio's column order
WEIGHT_KEYS: dict[str, str] = {
    "age": "PriorityWeightAge", "assoc": "PriorityWeightAssoc", "fairshare": "PriorityWeightFairShare",
    "jobsize": "PriorityWeightJobSize", "partition": "PriorityWeightPartition", "qos": "PriorityWeightQOS",
    "tres": "PriorityWeightTRES",
}
FACTORS = tuple(WEIGHT_KEYS)
MULTIFACTOR = "priority/multifactor"
# sprio -o (commands.SPRIO_FORMAT): id, user, partition, priority, then the weighted factors (age assoc fairshare
# jobsize partition qos tres) and the nice adjustment. One row per pending job × partition.
SPRIO_COLUMNS = cmds.SPRIO_FORMAT.count("|") + 1
SSHARE_FIELDS = tuple(cmds.SSHARE_FIELDS.split(","))
UNSUPPORTED = "supported priority plugin"      # sprio under priority/basic: "You are not running a supported …"
NO_JOBS = "Unable to find jobs"                  # sprio when none of the user's jobs is pending (any stream)

UNRANKED_HINT = ("this cluster does not rank by priority: pending jobs start in submit order as resources free up "
                 "(backfill can start short jobs earlier)")
NO_SPRIO_HINT = "sprio is not installed on this host: only your fair share is known"
# reasons under which the job competes on priority; any other reason (a QOS cap, a dependency, a hold) blocks first
PRIORITY_REASONS = frozenset({"Priority", "Resources", "None", ""})
FS_LOW = 0.25
FS_HIGH = 0.75
MAX_HINTS = 4

_DOMINANT: dict[str, str] = {
    "age": "it grows the longer the job waits",
    "assoc": "the account's association priority, set by the admins",
    "fairshare": "the fair-share standing of you and your account",
    "jobsize": "the size of the request",
    "partition": "the partition's priority tier: jobs in higher-tier partitions go first",
    "qos": "the QOS: a higher-priority QOS goes first",
    "tres": "the requested resources (TRES weights)",
}


@dataclass(frozen=True, slots=True)
class PrioritySettings:
    """The cluster's priority configuration (`slurm.config`); `type` None = not detected (older capabilities)."""
    type: str | None
    weights: Mapping[str, float] = field(default_factory=dict)   # factor -> weight (0 when absent or "(null)")
    flags: frozenset[str] = frozenset()
    half_life_s: float | None = None          # PriorityDecayHalfLife; 0 = usage never decays
    max_age_s: float | None = None            # PriorityMaxAge: the age factor is full after this long
    reset: str | None = None                  # PriorityUsageResetPeriod (NONE, NOW, DAILY, WEEKLY, MONTHLY, …)

    @property
    def multifactor(self) -> bool:
        return self.type == MULTIFACTOR

    @property
    def ranked(self) -> bool:
        """Pending jobs are ordered by a priority that varies: multifactor with at least one weight > 0."""
        return self.multifactor and any(w > 0 for w in self.weights.values())

    def weight(self, factor: str) -> float:
        return self.weights.get(factor, 0.0)


def _weight(value: str | None) -> float:
    """`1000` → 1000; `(null)` → 0; a TRES weight list `CPU=1000,Mem=2000,GRES/gpu=3000` → their sum (the most the
    TRES factor can add)."""
    v = nullish(value)
    if v is None:
        return 0.0
    single = parse_float(v)
    if single is not None:
        return max(0.0, single)
    total = 0.0
    for item in v.split(","):
        _key, eq, num = item.partition("=")
        f = parse_float(num) if eq else None
        total += max(0.0, f) if f is not None else 0.0
    return total


def settings(config: Mapping[str, str]) -> PrioritySettings:
    """`Capabilities.slurm_config` → PrioritySettings."""
    ptype = nullish(config.get("PriorityType"))
    if ptype is None:
        return PrioritySettings(type=None)
    weights = {f: _weight(config.get(key)) for f, key in WEIGHT_KEYS.items()}
    flags = frozenset(x.strip() for x in (nullish(config.get("PriorityFlags")) or "").split(",") if x.strip())
    half = parse_duration(config.get("PriorityDecayHalfLife"))
    max_age = parse_duration(config.get("PriorityMaxAge"))
    return PrioritySettings(type=ptype, weights=weights, flags=flags,
                            half_life_s=half.total_seconds() if half is not None else None,
                            max_age_s=max_age.total_seconds() if max_age is not None else None,
                            reset=nullish(config.get("PriorityUsageResetPeriod")))


# ---------------------------------------------------------------- parsers
@dataclass(frozen=True, slots=True)
class SprioRow:
    job_id: str                       # sprio %i (raw job id; an array's pending meta record is its array job id)
    user: str
    partition: str | None
    priority: float
    factors: Mapping[str, float]      # factor -> weighted value
    nice: float                       # %N, the nice adjustment (subtracted from the priority)


def _tres_value(value: str) -> float | None:
    """sprio %T: empty (no TRES weights), one number, or a per-TRES list `cpu=12,mem=3`."""
    v = value.strip()
    if not v:
        return 0.0
    single = parse_float(v)
    if single is not None:
        return single
    total = 0.0
    for item in v.split(","):
        _key, eq, num = item.partition("=")
        f = parse_float(num) if eq else None
        if f is None:
            return None
        total += f
    return total


def parse_sprio(text: str) -> ParseResult[SprioRow]:
    """`sprio -h -u <me> -o commands.SPRIO_FORMAT` rows. sprio's "Unable to find jobs matching user/id(s) specified" (no pending
    job) is an empty answer, not a bad row."""
    pr: ParseResult[SprioRow] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip() or line.startswith(NO_JOBS):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != SPRIO_COLUMNS or not parts[0]:
            pr.bad(n, line, f"expected {SPRIO_COLUMNS} fields, got {len(parts)}")
            continue
        job_id, user, part, prio = parts[0], parts[1], parts[2], parse_float(parts[3])
        values = [parse_float(p) if p else 0.0 for p in parts[4:10]]
        tres = _tres_value(parts[10])
        nice = parse_float(parts[11]) if parts[11] else 0.0
        if prio is None or any(v is None for v in values) or tres is None or nice is None:
            pr.bad(n, line, "non-numeric priority or factor", row_id=job_id)
            continue
        factors = dict(zip(FACTORS[:6], (float(v or 0.0) for v in values), strict=True))
        factors["tres"] = tres
        pr.items.append(SprioRow(job_id=job_id, user=user, partition=nullish(part), priority=prio,
                                 factors=factors, nice=nice))
    return pr


def _level_fs(value: str) -> float | None:
    v = value.strip().lower()
    return None if v in ("", "inf", "infinity", "nan") else parse_float(v)


def parse_sshare(text: str) -> ParseResult[FairShare]:
    """`sshare -h -U -u <me> -P -o SSHARE_FIELDS` rows (one per account of the user). `LevelFS` `inf` (no usage) →
    None; the account may be indented in tree output: stripped."""
    pr: ParseResult[FairShare] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != len(SSHARE_FIELDS) or not parts[0].strip():
            pr.bad(n, line, f"expected {len(SSHARE_FIELDS)} fields, got {len(parts)}")
            continue
        account, user = parts[0].strip(), parts[1].strip()
        fs = parse_float(parts[6])
        if fs is None and nullish(parts[6]) is not None:
            pr.bad(n, line, "non-numeric FairShare", row_id=account)
            continue
        pr.items.append(FairShare(account=account, user=user, raw_shares=nullish(parts[2]),
                                  norm_shares=parse_float(parts[3]), raw_usage=parse_int(parts[4]),
                                  effective_usage=parse_float(parts[5]), fairshare=fs,
                                  level_fs=_level_fs(parts[7])))
    return pr


def rows_for(job: Job, rows: Mapping[str, Sequence[SprioRow]]) -> list[SprioRow]:
    """The sprio rows of `job`: by its raw id; an array's pending summary also by the array job id, or by the rows of
    its tasks when sprio lists them as `<array job id>_<task>` (`index_rows` files those under the array job id too)."""
    found = list(rows.get(job.job_id_raw, ()))
    if not found and job.array_job_id:
        found = list(rows.get(job.array_job_id, ()))
    return found


def index_rows(items: Sequence[SprioRow]) -> dict[str, list[SprioRow]]:
    """sprio rows by job id; a `500_7`-style id is also filed under `500`, so an array's summary row finds them."""
    out: dict[str, list[SprioRow]] = {}
    for row in items:
        out.setdefault(row.job_id, []).append(row)
        base, sep, _task = row.job_id.partition("_")
        if sep and base.isdigit():
            out.setdefault(base, []).append(row)
    return out


# ---------------------------------------------------------------- evidence + hints
def unranked(st: PrioritySettings) -> PendingPriority:
    return PendingPriority(ranked=False, hints=(UNRANKED_HINT,), half_life_s=st.half_life_s)


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.2f}"


def _period(seconds: float) -> str:
    """`14 days`, `12 hours`, `90 min` — a half-life or a max age in words."""
    s = int(seconds)
    if s >= 86400 and s % 86400 == 0:
        d = s // 86400
        return f"{d} day{'s' if d != 1 else ''}"
    if s >= 3600 and s % 3600 == 0:
        h = s // 3600
        return f"{h} hour{'s' if h != 1 else ''}"
    if s >= 86400:
        return f"{s / 86400:.1f} days"
    return f"{max(1, s // 60)} min"


def _recovery(st: PrioritySettings) -> str:
    if st.half_life_s is None:
        return "it recovers as old usage decays"
    if st.half_life_s > 0:
        return f"it recovers with a half-life of {_period(st.half_life_s)}"
    reset = (st.reset or "NONE").upper()
    if reset in ("NONE", "NOW"):
        return "usage does not decay on this cluster (PriorityDecayHalfLife 0) and is reset only by the admins"
    return f"usage does not decay on this cluster; it is reset {reset.lower()}"


def _share(job: Job, shares: Sequence[FairShare]) -> FairShare | None:
    return next((s for s in shares if s.account == job.account), None) or (shares[0] if len(shares) == 1 else None)


def fairshare_hint(share: FairShare, st: PrioritySettings) -> str | None:
    fs = share.fairshare
    if fs is None:
        return None
    if fs < FS_LOW:
        if share.level_fs is not None and share.level_fs < 1:
            who = f"you used more than your share of {share.account} recently"
        else:
            who = f"your group ({share.account}) used much more than its share recently"
        return f"fairshare {fs:.2f}: {who}; {_recovery(st)}"
    if fs >= FS_HIGH:
        if not share.raw_usage:
            return f"fairshare {fs:.2f}: no recent usage on {share.account}, so fair share works for you"
        return f"fairshare {fs:.2f}: good standing on {share.account} — recent usage is within your share"
    return None


def evidence(job: Job, rows: Sequence[SprioRow], shares: Sequence[FairShare], st: PrioritySettings, *,
             sampled_at: datetime | None) -> PendingPriority:
    """PendingPriority of one own pending job from its sprio rows (the best partition's row carries the factors) and
    the user's sshare rows. Pure and independent of the clock, so it only changes with a new reading."""
    if not st.ranked:
        return unranked(st)
    own = sorted(shares, key=lambda s: (s.account != job.account, s.account))
    hints: list[str] = []
    reason = reason_key(job.reason) or ""
    if job.held:
        hints.append("held: it does not compete for a start until it is released")
    elif reason in NEVER_RUNS:                    # nothing else is worth saying: priority cannot help
        hints.append("priority does not matter here: as submitted, this job will never start")
    elif reason not in PRIORITY_REASONS:
        hints.append(f"held back by {humanize_reason(job.reason) or reason}, not by priority: these factors only "
                     "count once that clears")
    best = max(rows, key=lambda r: r.priority) if rows else None
    factors: list[PriorityFactor] = []
    dominant: str | None = None
    others: tuple[tuple[str, float], ...] = ()
    if best is not None:
        for name in FACTORS:
            value, weight = best.factors.get(name, 0.0), st.weight(name)
            if weight > 0 or value:
                factors.append(PriorityFactor(name=name, value=value, weight=weight or None))  # type: ignore[arg-type]
        if best.nice:
            factors.append(PriorityFactor(name="nice", value=-best.nice))
        factors.sort(key=lambda f: -f.value)
        top = factors[0] if factors else None
        dominant = top.name if top is not None and top.value > 0 else None
        others = tuple((r.partition or "?", r.priority) for r in rows if r is not best)
        if dominant is not None and top is not None:
            hints.append(f"{dominant} carries most of its priority ({_num(top.value)} of {_num(best.priority)}): "
                         f"{_DOMINANT.get(dominant, dominant)}")
        elif not job.held:
            hints.append(f"every factor is 0 for this job (priority {_num(best.priority)}): it starts when nothing "
                         "ranked higher fits — backfill can still start it early if it is short")
        if best.nice > 0:
            hints.append(f"nice {_num(best.nice)} lowers its priority (sbatch --nice)")
    elif sampled_at is not None:
        hints.append("not in the last priority reading yet (sprio runs at most every 10 min)")
    share = _share(job, own)
    if st.weight("fairshare") > 0 and share is not None and (h := fairshare_hint(share, st)):
        hints.append(h)
    age = next((f for f in factors if f.name == "age"), None)
    if (age is not None and age.weight and dominant != "age" and age.value < age.weight * 0.99
            and st.max_age_s):
        hints.append(f"waiting raises it: age adds up to {_num(age.weight)} after {_period(st.max_age_s)} in the "
                     "queue")
    if reason in NEVER_RUNS and not job.held:
        hints = hints[:1]
    return PendingPriority(ranked=True, priority=best.priority if best else None,
                           partition=best.partition if best else None, factors=tuple(factors), dominant=dominant,
                           others=others, fairshare=tuple(own), hints=tuple(hints[:MAX_HINTS]),
                           sampled_at=sampled_at, half_life_s=st.half_life_s)
