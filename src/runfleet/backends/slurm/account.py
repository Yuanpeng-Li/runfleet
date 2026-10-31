"""Account usage (§5.2.11): pure parsers of the `slurm.acct_*` sections and the derivation rules.

Nothing here runs a command or keeps state; the collector (periodic summary, on-demand report) owns both.

- `parse_share`: `sshare -A` rows. The account row (User empty) holds the account's SU limit (GrpTRESMins) and charged
  usage (GrpTRESRaw), both in billing-minutes; the user's own rows start with a space.
- `parse_sreport`: `sreport cluster AccountUtilizationByUser -t Minutes` with Proper Name cut on the host; a row with
  an empty Login is the account's total. sreport is NOT scaled by UsageFactor: billing-h, never SU.
- `parse_sbank`: `sbank balance statement -a <a>` blocks (lyra); names upper-cased by sbank, compared ignoring case.
- `parse_acct_jobs`: the user's allocations of one ≤ 30-day sacct chunk; `merge_jobs` deduplicates the chunks and
  `cells` clips them to the window and rolls them up per (name, account, partition, QOS, GPU model).
- `balances`: one `AccountBalance` per account from sshare, else sbank (rules 1-3).
- `period_starts`: where each account's "period" window starts (next reallocation − allocation_period, else the last
  usage reset), for the on-demand report and the Account screen's `[`/`]`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from runfleet.backends.common import ParseResult, section_error
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.states import STATE_WORD, map_state
from runfleet.config import HostConfig
from runfleet.models import (AccountBalance, AccountSummary, AccountUse, Capabilities, ErrorKind, HostError, JobState,
                             StateGroup, UsageCell)
from runfleet.probe.section import SectionOutcome, SectionResult
from runfleet.util import cost
from runfleet.util.slurmfmt import nullish, parse_duration, parse_float, parse_int, parse_timestamp, parse_tres
from runfleet.util.slurmfmt import split_rows

log = logging.getLogger("runfleet.backends.slurm.account")

MINUTES_PER_SU = 60.0              # 1 SU = 60 billing-minutes, the unit of GrpTRESMins / GrpTRESRaw
SREPORT_TRES = {"cpu": "cpu_h", "gres/gpu": "gpu_h", "billing": "billing_h"}
_TYPED = "gres/gpu:"
_TOO_WIDE = re.compile(r"Too wide of a date range", re.IGNORECASE)
_DBD_DOWN = re.compile(r"Problem talking to the database|slurmdbd|Connection refused", re.IGNORECASE)
_CTLD_DOWN = re.compile(r"Unable to contact slurm controller|Socket timed out", re.IGNORECASE)
_TRUNCATED = "runfleet: output truncated"
_NUM = r"-?[\d,]+(?:\.\d+)?"
_SBANK_ROW = re.compile(rf"^\s*(?P<name>\S+)\s+(?P<usage>{_NUM})\s*\|\s*(?P<limit>{_NUM})\s+(?P<avail>{_NUM})\s*$")
_REALLOC = re.compile(r"^\s*Next reallocation:\s*(?P<value>.*?)\s*$", re.IGNORECASE)
_NO_DATA = "no data found"
MAX_PERIOD_STARTS = 4              # distinct period starts per report (one sreport each); accounts beyond them: none
MAX_PERIOD_DAYS = 390              # a longer period is clipped here (13 sacct chunks) and the report marked truncated
RESET_PERIODS = ("DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY")


# ---------------------------------------------------------------- accounts
def accounts(caps: Capabilities | None) -> list[str]:
    """The user's accounts on this cluster: distinct accounts of `caps.associations` whose cluster is
    `caps.cluster_name` (or empty), sorted, at most MAX_ACCOUNTS; a name the validator refuses is skipped (logged)."""
    if caps is None:
        return []
    found: set[str] = set()
    for cluster, account, _part, _qos in caps.associations:
        if cluster and caps.cluster_name and cluster != caps.cluster_name:
            continue
        if not account:
            continue
        if not cmds.NAME_RE.match(account):
            log.warning("account %r skipped: not a valid name", account[:60])
            continue
        found.add(account)
    out = sorted(found)
    if len(out) > cmds.MAX_ACCOUNTS:
        log.warning("%d accounts: only the first %d are read", len(out), cmds.MAX_ACCOUNTS)
    return out[:cmds.MAX_ACCOUNTS]


# ---------------------------------------------------------------- sshare -A
@dataclass(frozen=True, slots=True)
class ShareRow:
    account: str
    user: str                                 # "" = the account row
    raw_usage: int | None
    grp_tres_mins: Mapping[str, float]        # limits, TRES-minutes (keys lower-case); {} = none
    grp_tres_raw: Mapping[str, float]         # charged usage, TRES-minutes (after UsageFactor; decayed when the
                                              # cluster has a half-life)


def _tres_numbers(value: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw in parse_tres(value).items():
        n = parse_float(raw)
        if n is not None:
            out[key.lower()] = n
    return out


def parse_share(text: str) -> ParseResult[ShareRow]:
    """`slurm.acct_share`: `Account|User|RawUsage|GrpTRESMins|GrpTRESRaw` rows; user rows start with a space."""
    pr: ParseResult[ShareRow] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 5:
            pr.bad(n, line, f"expected 5 fields, got {len(parts)}")
            continue
        account, user = parts[0].strip(), parts[1].strip()
        if not cmds.NAME_RE.match(account) or (user and not cmds.NAME_RE.match(user)):
            pr.bad(n, line, "not an account and user")
            continue
        pr.items.append(ShareRow(account=account, user=user, raw_usage=parse_int(parts[2]),
                                 grp_tres_mins=_tres_numbers(parts[3]), grp_tres_raw=_tres_numbers(parts[4])))
    return pr


# ---------------------------------------------------------------- sreport
@dataclass(frozen=True, slots=True)
class SreportRow:
    cluster: str
    account: str
    login: str                                # "" = the account's total
    tres: str                                 # lower-case: "cpu", "gres/gpu", "billing"
    minutes: float


def parse_sreport(text: str) -> ParseResult[SreportRow]:
    """`slurm.acct_sreport[.p<i>]`: `Cluster|Account|Login|TRES Name|Used` (minutes; Proper Name cut on the host)."""
    pr: ParseResult[SreportRow] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 5:
            pr.bad(n, line, f"expected 5 fields, got {len(parts)}")
            continue
        cluster, account, login, tres, used = parts
        minutes = parse_float(used)
        if (not cmds.NAME_RE.match(account) or (login and not cmds.NAME_RE.match(login)) or not tres
                or minutes is None or minutes < 0):
            pr.bad(n, line, "not an sreport TRES row")
            continue
        pr.items.append(SreportRow(cluster, account, login, tres.lower(), minutes))
    return pr


def sreport_usage(rows: Iterable[SreportRow], *, cluster: str | None = None,
                  starts: Mapping[str, datetime] | None = None, inferred: bool = False,
                  ) -> tuple[tuple[AccountUse, ...], tuple[AccountUse, ...]]:
    """(the user's usage, the accounts' totals) per account, in hours, sorted by account. Rows of another cluster
    are dropped when `cluster` is known; TRES other than cpu / gres/gpu / billing are ignored; a TRES without a row
    stays None. `starts` (account -> window start) only labels the result."""
    sums: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if cluster and r.cluster and r.cluster != cluster:
            continue
        field = SREPORT_TRES.get(r.tres)
        if field is None:
            continue
        scope = "account" if not r.login else "me"
        bucket = sums.setdefault((scope, r.account), {})
        bucket[field] = bucket.get(field, 0.0) + r.minutes / 60
    mine: list[AccountUse] = []
    totals: list[AccountUse] = []
    for (scope, account), v in sorted(sums.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        use = AccountUse(account=account, scope="me" if scope == "me" else "account", cpu_h=v.get("cpu_h"),
                         gpu_h=v.get("gpu_h"), billing_h=v.get("billing_h"),
                         start=(starts or {}).get(account), start_inferred=inferred)
        (mine if scope == "me" else totals).append(use)
    return tuple(mine), tuple(totals)


# ---------------------------------------------------------------- sbank
@dataclass(frozen=True, slots=True)
class SbankBalance:
    account: str                              # as asked (the `@@acct` name)
    usage_su: float | None = None             # "Usage" (charged SU)
    limit_su: float | None = None             # "Account Limit"
    next_realloc: date | None = None          # "Next reallocation: YYYY-MM-DD"
    realloc_known: bool = False               # True with next_realloc None: "No data found."
    error: str | None = None                  # an unknown layout or an sbank error: a PARSE error for this account only


def _number(value: str) -> float | None:
    return parse_float(value.replace(",", ""))


def _sbank_block(account: str, lines: Sequence[str]) -> SbankBalance:
    row: re.Match[str] | None = None
    next_realloc: date | None = None
    known = False
    no_data = False
    for line in lines:
        if (m := _SBANK_ROW.match(line)) is not None and m.group("name").lower() == account.lower():
            row = row or m
        elif (r := _REALLOC.match(line)) is not None:
            value = r.group("value")
            if value.rstrip(".").lower() == _NO_DATA:
                known = True
            else:
                try:
                    next_realloc, known = date.fromisoformat(value), True
                except ValueError:
                    pass
        elif line.strip().rstrip(".").lower() == _NO_DATA:
            no_data = True
    if row is None:
        if no_data:
            return SbankBalance(account, realloc_known=True)
        first = next((ln.strip() for ln in lines if ln.strip()), "no output")
        return SbankBalance(account, error=f"no balance row for {account}: {first[:120]}")
    return SbankBalance(account, usage_su=_number(row.group("usage")), limit_su=_number(row.group("limit")),
                        next_realloc=next_realloc, realloc_known=known)


def parse_sbank(text: str) -> dict[str, SbankBalance]:
    """`slurm.acct_sbank`: one `@@acct <a>` block per account -> balance keyed by the lower-case account name. The
    table row is the one whose first column equals the account ignoring case (thousands separators allowed); an
    unknown layout or an error line sets `error` for that account only."""
    out: dict[str, SbankBalance] = {}
    current: str | None = None
    lines: list[str] = []
    for line in [*split_rows(text), "@@acct "]:
        if line.startswith("@@acct "):
            if current:
                out[current.lower()] = _sbank_block(current, lines)
            current, lines = line[len("@@acct "):].strip(), []
        elif current is not None:
            lines.append(line)
    return out


# ---------------------------------------------------------------- balances (rules 1-3)
def half_life_s(caps: Capabilities | None) -> float | None:
    """PriorityDecayHalfLife in seconds (0 = usage never decays); None = not known."""
    if caps is None:
        return None
    half = parse_duration(caps.slurm_config.get("PriorityDecayHalfLife"))
    return half.total_seconds() if half is not None else None


def reset_period(caps: Capabilities | None) -> str | None:
    """PriorityUsageResetPeriod unless NONE: "MONTHLY", "WEEKLY", …"""
    value = nullish(caps.slurm_config.get("PriorityUsageResetPeriod")) if caps is not None else None
    return None if value is None or value.upper() == "NONE" else value.upper()


def used_basis(caps: Capabilities | None) -> Literal["since_reset", "decayed", "unknown"]:
    """What sshare's usage means: half-life 0 -> "since_reset" (lyra); > 0 -> "decayed" (k2, vega: "recent", never
    "used"); unknown -> "unknown"."""
    half = half_life_s(caps)
    if half is None:
        return "unknown"
    return "since_reset" if half == 0 else "decayed"


def balances(names: Sequence[str], share: Sequence[ShareRow] | None, sbank: Mapping[str, SbankBalance] | None,
             caps: Capabilities | None, *, read_at: datetime | None = None) -> tuple[AccountBalance, ...]:
    """One balance per account, sorted. Used SU: the sshare account row's GrpTRESRaw billing / 60, else sbank
    "Usage"; limit: that row's GrpTRESMins billing / 60, else sbank "Account Limit", else None ("no SU limit"). Other
    GrpTRESMins keys go to `other_limits` in hours, never SU. sbank also gives the next reallocation. `my_used_su`:
    the user's own rows' GrpTRESRaw billing / 60 (sshare only), what usage-only mode shows as "recent"."""
    rows = {r.account.lower(): r for r in share or () if not r.user}
    mine: dict[str, float] = {}
    for r in share or ():
        if r.user and "billing" in r.grp_tres_raw:     # without -a: only the user's own rows
            mine[r.account.lower()] = mine.get(r.account.lower(), 0.0) + r.grp_tres_raw["billing"] / MINUTES_PER_SU
    basis = used_basis(caps)
    out: list[AccountBalance] = []
    for name in sorted(set(names)):
        row = rows.get(name.lower())
        bank = (sbank or {}).get(name.lower())
        if bank is not None and bank.error is not None:
            bank = None
        used: float | None = None
        limit: float | None = None
        used_src: Literal["sshare", "sbank"] | None = None
        limit_src: Literal["sshare", "sbank"] | None = None
        other: dict[str, float] = {}
        if row is not None:
            if "billing" in row.grp_tres_raw:
                used, used_src = row.grp_tres_raw["billing"] / MINUTES_PER_SU, "sshare"
            if "billing" in row.grp_tres_mins:
                limit, limit_src = row.grp_tres_mins["billing"] / MINUTES_PER_SU, "sshare"
            other = {k: v / 60 for k, v in row.grp_tres_mins.items() if k != "billing"}
        if used is None and bank is not None and bank.usage_su is not None:
            used, used_src = bank.usage_su, "sbank"
        if limit is None and bank is not None and bank.limit_su is not None:
            limit, limit_src = bank.limit_su, "sbank"
        out.append(AccountBalance(
            account=name, limit_su=limit, used_su=used, used_basis=basis, limit_source=limit_src,
            used_source=used_src, other_limits=other,
            next_realloc=bank.next_realloc if bank is not None else None,
            realloc_known=bank.realloc_known if bank is not None else False,
            read_at=read_at if used is not None else None, my_used_su=mine.get(name.lower())))
    return tuple(out)


def mode(items: Iterable[AccountBalance]) -> Literal["limits", "usage_only"]:
    """"limits" when at least one account has an SU limit (rule 4)."""
    return "limits" if any(b.limit_su is not None for b in items) else "usage_only"


# ---------------------------------------------------------------- period window
@dataclass(frozen=True, slots=True)
class PeriodWindow:
    starts: Mapping[str, datetime]            # account -> its period start; an account without one is left out
    inferred: frozenset[str] = frozenset()    # accounts whose start is next_realloc - allocation_period (labelled)
    reset: str | None = None                  # the PriorityUsageResetPeriod the other starts come from
    clipped: bool = False                     # a start more than MAX_PERIOD_DAYS back was clipped there

    @property
    def earliest(self) -> datetime | None:
        return min(self.starts.values()) if self.starts else None


def reset_start(reset: str | None, now: datetime, tz: timedelta) -> datetime | None:
    """The last PriorityUsageResetPeriod boundary at or before `now`: 00:00 in the cluster's offset of today
    (DAILY), the last Sunday (WEEKLY), the 1st of the month (MONTHLY), of the quarter (QUARTERLY) or of January
    (YEARLY); None for NONE / NOW / unknown."""
    if reset not in RESET_PERIODS:
        return None
    local = now.astimezone(timezone(tz)).date()
    if reset == "WEEKLY":
        local -= timedelta(days=(local.weekday() + 1) % 7)            # Monday 0 … Sunday 6
    elif reset == "MONTHLY":
        local = local.replace(day=1)
    elif reset == "QUARTERLY":
        local = local.replace(month=(local.month - 1) // 3 * 3 + 1, day=1)
    elif reset == "YEARLY":
        local = local.replace(month=1, day=1)
    return datetime.combine(local, time(), tzinfo=timezone(tz))


def period_starts(summary: AccountSummary | None, caps: Capabilities | None, host_cfg: HostConfig,
                  now: datetime) -> PeriodWindow:
    """Each account's "period" window (§5.2.11). An account with a known sbank reallocation date (and
    `allocation_period` > 0) starts at that date's cluster-local midnight minus allocation_period (inferred); sbank's
    "No data found." gives it no period. Other accounts start at the last usage reset when the cluster has a reset
    period (vega: monthly), else have none (k2). A start in the future gives no period; one more than
    MAX_PERIOD_DAYS back is clipped there (`clipped`); at most MAX_PERIOD_STARTS distinct starts (accounts in name
    order; later ones with a new start get none). An empty `starts` = no period window on this host."""
    if caps is None:
        return PeriodWindow({})
    tz = caps.tz_offset
    period = host_cfg.slurm.allocation_period
    reset = reset_period(caps)
    by_reset = reset_start(reset, now, tz)
    oldest = datetime.combine((now.astimezone(timezone(tz)) - timedelta(days=MAX_PERIOD_DAYS - 1)).date(), time(),
                              tzinfo=timezone(tz))
    names = sorted({b.account for b in summary.balances} if summary is not None else accounts(caps))
    bank = {b.account: b for b in summary.balances} if summary is not None else {}
    starts: dict[str, datetime] = {}
    inferred: set[str] = set()
    clipped = used_reset = False
    for name in names:
        b = bank.get(name)
        start: datetime | None
        if period > 0 and b is not None and b.realloc_known:
            if b.next_realloc is None:                               # "No data found."
                continue
            start = datetime.combine(b.next_realloc, time(), tzinfo=timezone(tz)) - timedelta(seconds=period)
            is_inferred = True
        else:
            start, is_inferred = by_reset, False
        if start is None or start > now:
            continue
        if start < oldest:
            start, clipped = oldest, True
        if start not in starts.values() and len(set(starts.values())) >= MAX_PERIOD_STARTS:
            log.info("account %s: more than %d period starts, left out", name, MAX_PERIOD_STARTS)
            continue
        starts[name] = start
        if is_inferred:
            inferred.add(name)
        else:
            used_reset = True
    return PeriodWindow(starts, frozenset(inferred), reset if used_reset else None, clipped)


# ---------------------------------------------------------------- sacct chunks
@dataclass(frozen=True, slots=True)
class AcctJob:
    job_id_raw: str
    account: str
    partition: str
    qos: str
    state: JobState
    elapsed_s: int                            # ElapsedRaw: wall seconds so far (running) or in total
    start: datetime
    end: datetime | None
    tres: Mapping[str, str]                   # AllocTRES, raw
    name: str


def parse_acct_jobs(text: str, *, tz: timedelta, sep: str = cmds.SEP) -> ParseResult[AcctJob]:
    """`slurm.acct_jobs.c<i>`: `sacct -X` rows in ACCT_JOBS_FIELDS order (JobName last: split with maxsplit, so a
    separator in the name stays in it). Rows that never started (ElapsedRaw 0) are skipped, not bad."""
    names = cmds.ACCT_JOBS_FIELDS
    pr: ParseResult[AcctJob] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split(sep, len(names) - 1)
        if len(parts) != len(names):
            pr.bad(n, line, f"expected {len(names)} fields, got {len(parts)}")
            continue
        row = dict(zip(names, parts, strict=True))
        raw_id = row["JobIDRaw"].strip()
        raw_state = row["State"].strip()
        elapsed = parse_int(row["ElapsedRaw"])
        start = parse_timestamp(row["Start"], tz)
        if not raw_id.isdigit() or not STATE_WORD.match(raw_state) or elapsed is None or elapsed < 0:
            pr.bad(n, line, f"bad row {raw_id!r}", raw_id if raw_id.isdigit() else None)
            continue
        if elapsed == 0:
            continue
        if start is None:
            pr.bad(n, line, f"no start for {raw_id}", raw_id)
            continue
        state, _extra = map_state(raw_state)
        pr.items.append(AcctJob(
            job_id_raw=raw_id, account=row["Account"].strip(), partition=row["Partition"].strip(),
            qos=row["QOS"].strip(), state=state, elapsed_s=elapsed, start=start,
            end=parse_timestamp(row["End"], tz), tres=parse_tres(row["AllocTRES"]), name=row["JobName"]))
    return pr


def merge_jobs(chunks: Iterable[Iterable[AcctJob]]) -> list[AcctJob]:
    """Deduplicate the chunks by JobIDRaw (a job crossing a chunk boundary is listed by both; the larger ElapsedRaw
    wins), in first-seen order."""
    out: dict[str, AcctJob] = {}
    for chunk in chunks:
        for job in chunk:
            seen = out.get(job.job_id_raw)
            if seen is None or job.elapsed_s > seen.elapsed_s:
                out[job.job_id_raw] = job
    return list(out.values())


def account_error(result: SectionResult, at: datetime | None) -> HostError | None:
    """Classified failure of an account section (None when OK/EMPTY). sacct's `Too wide of a date range` (empty
    stdout, rc 0 through the pipe: STDERR_ONLY) is "sacct refused the date range", never "no jobs"."""
    if result.ok:
        return None
    summary: str | None = None
    if result.outcome in (SectionOutcome.FAILED, SectionOutcome.STDERR_ONLY):
        if _TOO_WIDE.search(result.stderr):
            summary = "sacct refused the date range"
        elif result.name.startswith("slurm.acct_share") and _CTLD_DOWN.search(result.stderr):
            summary = "slurmctld unreachable"
        elif _DBD_DOWN.search(result.stderr):
            summary = "slurmdbd unreachable"
        elif result.stderr.startswith(_TRUNCATED) and result.stdout:
            summary = f"{result.name}: output cut at its size limit"
    return section_error(result, at, summary=summary, kind=ErrorKind.REMOTE if summary else None)


def truncated(result: SectionResult) -> bool:
    """A chunk whose rows are only a part of the answer: cut at max_stdout, or the connection died mid-section."""
    if result.outcome in (SectionOutcome.INCOMPLETE, SectionOutcome.TIMEOUT):
        return True
    return result.outcome is SectionOutcome.FAILED and result.stderr.startswith(_TRUNCATED)


# ---------------------------------------------------------------- rollup cells
def clipped_hours(job: AcctJob, start: datetime | None, now: datetime) -> float:
    """Hours of `[Start, Start + ElapsedRaw]` inside `[start, now]` (start None: from the job's start)."""
    begin = job.start if start is None else max(job.start, start)
    end = min(job.start + timedelta(seconds=job.elapsed_s), now)
    return max(0.0, (end - begin).total_seconds()) / 3600


def _gpu_split(counts: Mapping[str, float]) -> tuple[float, dict[str, float], str]:
    """GPU count, GPUs per model and the job's main model: the typed keys (lower-case; a MIG slice is a model of its
    own), the rest of `gres/gpu` under "" (untyped), as `util/cost._gpu_split` splits a job's GPU-h; the main model
    is the one with most GPUs (then a typed one, then by name), "" without GPUs."""
    typed: dict[str, float] = {}
    for k, v in counts.items():
        if k.startswith(_TYPED) and v > 0:
            typed[k[len(_TYPED):]] = typed.get(k[len(_TYPED):], 0.0) + v
    total = counts.get("gres/gpu", sum(typed.values()))
    rest = total - sum(typed.values())
    if rest > 1e-9:
        typed[""] = typed.get("", 0.0) + rest
    main = min(typed, key=lambda m: (-typed[m], m == "", m)) if typed else ""
    return total, typed, main


def cells(jobs: Iterable[AcctJob], caps: Capabilities | None, *, now: datetime,
          start: datetime | Mapping[str, datetime] | None) -> tuple[tuple[UsageCell, ...], int]:
    """(cells, distinct allocations counted). Each job is clipped to the window — one start for all accounts, or one
    per account (the period window: an account without one is left out) — and adds CPU-h, GPU-h, billing-h (AllocTRES
    billing, else `billing_rate` over the partition's weights, else CPUs) and charged SU (× the UsageFactor of its
    QOS; None when that factor is unknown) to its (name, account, partition, QOS, GPU model) cell. A job with GPUs
    of several models (or a typed and an untyped part) adds each model's GPU-h to that model's cell; it counts (jobs,
    CPU-h, billing) once, on its main model."""
    acc: dict[tuple[str, str, str, str, str], list[float]] = {}
    unknown: set[tuple[str, str, str, str, str]] = set()
    seen = 0
    maxt = cost.max_tres(caps)
    for job in jobs:
        if isinstance(start, Mapping):
            if job.account not in start:
                continue
            begin: datetime | None = start[job.account]
        else:
            begin = start
        hours = clipped_hours(job, begin, now)
        if hours <= 0:
            continue
        counts = cost.tres_counts(job.tres)
        cpus = counts.get("cpu", 0.0)
        _gpus, by_model, model = _gpu_split(counts)
        billing = counts.get("billing")
        if billing is None:
            charge = cost.partition_charge(caps, job.partition)
            weighted = charge is not None and cost.parse_weights(charge.billing_weights)
            billing = cost.billing_rate(charge, counts, max_tres=maxt, gpu_type=model or None) if weighted else cpus
        factor = cost.usage_factor(caps, job.qos)
        key = (job.name, job.account, job.partition, job.qos, model)
        if factor is None:
            unknown.add(key)
        a = acc.setdefault(key, [0, 0, 0.0, 0.0, 0.0, 0.0])
        a[0] += 1                                  # the job, its CPU-h and billing count once: on its main model
        a[1] += job.state.group is StateGroup.ACTIVE
        a[2] += cpus * hours
        for other, n in by_model.items():          # GPU-h of every model (and the untyped rest) where they ran
            acc.setdefault((*key[:4], other), [0, 0, 0.0, 0.0, 0.0, 0.0])[3] += n * hours
        a[4] += billing * hours
        a[5] += billing * hours * (1.0 if factor is None else factor)
        seen += 1
    out = tuple(UsageCell(name=k[0], account=k[1], partition=k[2], qos=k[3], gpu_model=k[4], jobs=int(a[0]),
                          running=int(a[1]), cpu_h=a[2], gpu_h=a[3], billing_h=a[4],
                          charged_su=None if k in unknown else a[5])
                for k, a in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][0], kv[0][4])))
    return out, seen


def usage_from_cells(items: Iterable[UsageCell], starts: Mapping[str, datetime] | None = None,
                     inferred: bool = False) -> tuple[AccountUse, ...]:
    """The user's usage per account summed from the rollup (no sreport: `source="sacct"`)."""
    sums: dict[str, list[float]] = {}
    for c in items:
        s = sums.setdefault(c.account, [0.0, 0.0, 0.0])
        s[0] += c.cpu_h
        s[1] += c.gpu_h
        s[2] += c.billing_h
    return tuple(AccountUse(account=a, scope="me", cpu_h=s[0], gpu_h=s[1], billing_h=s[2], source="sacct",
                            start=(starts or {}).get(a), start_inferred=inferred)
                 for a, s in sorted(sums.items()))
