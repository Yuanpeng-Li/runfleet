"""[WP2] Slurm state mapping, reason humanizer, exit explanations, hints (§5.2.6).

The pending-reason text and the short exit summary are also copied into `Job.hints` / `Job.raw["reason_text"]` by the
collector, because the UI must not import backends (§3.1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from runfleet.models import Job, JobState, JobStep, StateGroup
from runfleet.util import humanize, platform
from runfleet.util.slurmfmt import parse_mem_bytes, seconds_prefix

log = logging.getLogger("runfleet.backends.slurm.states")

_STATES: dict[str, JobState] = {
    "PENDING": JobState.PENDING, "CONFIGURING": JobState.CONFIGURING,
    "REQUEUED": JobState.REQUEUED, "REQUEUE_FED": JobState.REQUEUED, "REQUEUE_HOLD": JobState.REQUEUED,
    # SPECIAL_EXIT: slurmctld requeues the job (PENDING, priority 0, JobHeldUser) — it is waiting, not ended.
    # EXPEDITING (25.11+): an expedited requeue in progress. Both stay cancellable / releasable.
    "SPECIAL_EXIT": JobState.REQUEUED, "EXPEDITING": JobState.REQUEUED,
    "RESV_DEL_HOLD": JobState.PENDING,
    "RUNNING": JobState.RUNNING, "RESIZING": JobState.RUNNING, "SIGNALING": JobState.RUNNING,
    "STAGE_OUT": JobState.RUNNING,
    "COMPLETING": JobState.COMPLETING, "SUSPENDED": JobState.SUSPENDED, "STOPPED": JobState.STOPPED,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "TIMEOUT": JobState.TIMEOUT, "OUT_OF_MEMORY": JobState.OUT_OF_MEMORY, "NODE_FAIL": JobState.NODE_FAIL,
    "PREEMPTED": JobState.PREEMPTED, "BOOT_FAIL": JobState.BOOT_FAIL, "DEADLINE": JobState.DEADLINE,
    "CANCELLED": JobState.CANCELLED, "REVOKED": JobState.CANCELLED,
}
# sacct abbreviations (squeue %t / StateCompact), used when only the compact form is available.
_COMPACT: dict[str, str] = {"PD": "PENDING", "CF": "CONFIGURING", "RQ": "REQUEUED", "RH": "REQUEUE_HOLD",
                            "RS": "RESIZING", "SI": "SIGNALING", "SO": "STAGE_OUT", "R": "RUNNING",
                            "CG": "COMPLETING", "S": "SUSPENDED", "ST": "STOPPED", "CD": "COMPLETED", "F": "FAILED",
                            "SE": "SPECIAL_EXIT", "TO": "TIMEOUT", "OOM": "OUT_OF_MEMORY", "NF": "NODE_FAIL",
                            "PR": "PREEMPTED", "BF": "BOOT_FAIL", "DL": "DEADLINE", "CA": "CANCELLED",
                            "RV": "REVOKED", "RD": "RESV_DEL_HOLD", "RF": "REQUEUE_FED"}
_CANCELLED_BY = re.compile(r"^CANCELLED\s+by\s+(\d+)$")
STATE_WORD = re.compile(r"^[A-Z_]+(?:\s+by\s+\d+)?$")
_unknown_logged: set[str] = set()

HELD_REASONS = frozenset({"JobHeldUser", "JobHeldAdmin"})


def map_state(raw: str) -> tuple[JobState, dict[str, str]]:
    """squeue/sacct state string → (JobState, extra raw fields). `CANCELLED by 10001` → CA +
    {"cancelled_by_uid": "10001"}; an unknown string → UNKNOWN (logged once per string)."""
    s = raw.strip()
    m = _CANCELLED_BY.match(s)
    if m:
        return JobState.CANCELLED, {"cancelled_by_uid": m.group(1)}
    word = s.split()[0] if s else ""
    state = _STATES.get(word) or _STATES.get(_COMPACT.get(word, ""))
    if state is None:
        if s not in _unknown_logged:
            _unknown_logged.add(s)
            log.warning("unknown Slurm state %r", s[:60])
        return JobState.UNKNOWN, {}
    return state, {}


def is_held(reason: str | None, raw_state: str) -> bool:
    return (reason in HELD_REASONS) or raw_state.strip() in ("REQUEUE_HOLD", "RESV_DEL_HOLD", "SPECIAL_EXIT")


# ---------------------------------------------------------------- pending reasons
REASONS: dict[str, str] = {
    "Priority": "waiting for priority",
    "Resources": "waiting for resources",
    "QOSMaxJobsPerUserLimit": "per-user job cap (QOS)",
    "QOSMaxGRESPerUser": "per-user GPU cap (QOS)",
    "QOSMaxCpuPerUserLimit": "per-user CPU cap (QOS)",
    "QOSMaxMemoryPerUser": "per-user memory cap (QOS)",
    "QOSMaxNodePerUserLimit": "per-user node cap (QOS)",
    "QOSGrpGRES": "QOS-wide GPU cap",
    "QOSGrpCpuLimit": "QOS-wide CPU cap",
    "QOSGrpMemLimit": "QOS-wide memory cap",
    "QOSGrpNodeLimit": "QOS-wide node cap",
    "QOSMaxWallDurationPerJobLimit": "time limit above QOS max — will never run",
    "AssocGrpGRES": "group GPU cap",
    "AssocGrpCPULimit": "group CPU cap",
    "AssocGrpCpuLimit": "group CPU cap",
    "AssocGrpMemLimit": "group memory cap",
    "AssocGrpNodeLimit": "group node cap",
    "AssocGrpJobsLimit": "group job cap",
    "AssocMaxJobsLimit": "per-user job cap (association)",
    "AssocGrpBillingMinutes": "group compute budget used up",
    "AssocGrpCPUMinutesLimit": "group CPU-minutes budget used up",
    "AssocGrpGRESMinutes": "group GPU-minutes budget used up",
    "MaxGRESPerAccount": "account GPU cap",
    "Dependency": "waiting for dependency",
    "DependencyNeverSatisfied": "will never run: dependency failed",
    "ReqNodeNotAvail": "requested node unavailable",
    "Nodes required for job are DOWN": "required nodes down / drained",
    "PartitionTimeLimit": "time limit above partition max — will never run",
    "PartitionNodeLimit": "node count outside partition limits — will never run",
    "BeginTime": "scheduled start later",
    "JobHeldUser": "held",
    "JobHeldAdmin": "held",
    "launch failed requeued held": "launch failed, requeued and held",
    "JobArrayTaskLimit": "array task limit",
    "Reservation": "waiting for reservation",
    "BadConstraints": "constraints can never be met — will never run",
    "InvalidAccount": "invalid account — will never run",
    "InvalidQOS": "invalid QOS — will never run",
}
NEVER_RUNS = frozenset({"DependencyNeverSatisfied", "PartitionTimeLimit", "QOSMaxWallDurationPerJobLimit",
                        "PartitionNodeLimit", "BadConstraints", "InvalidAccount", "InvalidQOS"})
_WAIT_GROUP = "starts when your group's running jobs free enough of it; or submit under another account"
_WAIT_QOS = "everyone on this QOS together is at its limit; wait, or use another QOS / partition"
_WAIT_MINE = "starts when one of your running jobs ends; nothing to fix"
# One actionable line per reason (§5.2.8): what it means for you and what, if anything, to do about it.
REASON_ADVICE: dict[str, str] = {
    "Priority": "other jobs are ahead of you (fair-share); a shorter --time lets backfill start it sooner",
    "Resources": "next in line, but the nodes / GPUs it asks for are busy; fewer GPUs, another GPU type or partition, "
                 "or a shorter --time can start it sooner",
    "QOSMaxJobsPerUserLimit": "you run as many jobs as this QOS allows; " + _WAIT_MINE,
    "QOSMaxGRESPerUser": "you hold as many GPUs as this QOS allows per user; " + _WAIT_MINE,
    "QOSMaxCpuPerUserLimit": "you hold as many CPUs as this QOS allows per user; " + _WAIT_MINE,
    "QOSMaxMemoryPerUser": "you hold as much memory as this QOS allows per user; " + _WAIT_MINE,
    "QOSMaxNodePerUserLimit": "you hold as many nodes as this QOS allows per user; " + _WAIT_MINE,
    "QOSGrpGRES": _WAIT_QOS,
    "QOSGrpCpuLimit": _WAIT_QOS,
    "QOSGrpMemLimit": _WAIT_QOS,
    "QOSGrpNodeLimit": _WAIT_QOS,
    "QOSMaxWallDurationPerJobLimit": "--time is above the QOS's MaxWall: lower it (scontrol update JobId=<id> "
                                     "TimeLimit=…) or cancel and resubmit",
    "AssocGrpGRES": "your account's GPU limit is in use; " + _WAIT_GROUP,
    "AssocGrpCPULimit": "your account's CPU limit is in use; " + _WAIT_GROUP,
    "AssocGrpCpuLimit": "your account's CPU limit is in use; " + _WAIT_GROUP,
    "AssocGrpMemLimit": "your account's memory limit is in use; " + _WAIT_GROUP,
    "AssocGrpNodeLimit": "your account's node limit is in use; " + _WAIT_GROUP,
    "AssocGrpJobsLimit": "your account runs as many jobs as it may; " + _WAIT_GROUP,
    "AssocMaxJobsLimit": "you run as many jobs as your association allows; " + _WAIT_MINE,
    "AssocGrpBillingMinutes": "the account's compute budget is spent: ask the PI / admins for more, or use another "
                              "account (--account)",
    "AssocGrpCPUMinutesLimit": "the account's CPU-minutes budget is spent: ask the PI / admins, or use another account",
    "AssocGrpGRESMinutes": "the account's GPU-minutes budget is spent: ask the PI / admins, or use another account",
    "MaxGRESPerAccount": "your account holds as many GPUs as the QOS allows per account; " + _WAIT_GROUP,
    "Dependency": "starts once the jobs in its --dependency end as required",
    "DependencyNeverSatisfied": "a job it depends on failed or was cancelled: cancel this one and resubmit (or "
                                "scontrol update JobId=<id> Dependency=…)",
    "ReqNodeNotAvail": "a node it needs is down, drained or reserved (often maintenance): wait, or drop "
                       "--nodelist / --exclude / --constraint",
    "Nodes required for job are DOWN": "nodes it needs are down, drained or reserved for higher-priority partitions: "
                                       "wait, or relax node constraints",
    "PartitionTimeLimit": "--time is above the partition's MaxTime: lower it (scontrol update JobId=<id> "
                          "TimeLimit=…) or cancel and resubmit",
    "PartitionNodeLimit": "--nodes is outside the partition's limits: cancel and resubmit with a node count it allows",
    "BeginTime": "its --begin time has not come yet",
    "JobHeldUser": "held by you (or by a requeue): release it (x › Release, scontrol release <id>)",
    "JobHeldAdmin": "held by an administrator: ask the cluster admins",
    "launch failed requeued held": "the node could not start it (bad node or prolog): read the log, then release it "
                                   "(x › Release); --exclude the node if it repeats",
    "JobArrayTaskLimit": "the array's %N throttle is reached: the next element starts when one finishes",
    "Reservation": "the reservation it uses has not started yet",
    "BadConstraints": "--constraint / --gres asks for something no node has: cancel and resubmit",
    "InvalidAccount": "the --account is not valid for you: cancel and resubmit with one of your accounts "
                      "(sacctmgr show assoc user=$USER)",
    "InvalidQOS": "the --qos is not valid for your account / partition: cancel and resubmit with an allowed QOS",
}


def reason_key(reason: str | None) -> str | None:
    """`(ReqNodeNotAvail, UnavailableNodes:n-31-01)` → `ReqNodeNotAvail`; `None`/empty → None."""
    if not reason or reason == "None":
        return None
    r = reason.strip()
    if r.startswith("(") and r.endswith(")"):
        r = r[1:-1]
    return r.split(",", 1)[0].split("(", 1)[0].strip()


def humanize_reason(reason: str | None) -> str | None:
    """Pending reason → short text; unknown reasons are shown raw; `None`/empty → None."""
    key = reason_key(reason)
    if key is None:
        return None
    if key not in REASONS and reason is not None:
        r = reason.strip()
        return r[1:-1] if r.startswith("(") and r.endswith(")") else r
    return REASONS[key]


def reason_advice(reason: str | None) -> str | None:
    """One actionable line for a pending reason; unknown `Assoc…` / `QOS…` limits get the generic advice."""
    key = reason_key(reason)
    if key is None:
        return None
    if key in REASON_ADVICE:
        return REASON_ADVICE[key]
    if key.startswith("Assoc"):
        return "an account (association) limit is reached; " + _WAIT_GROUP
    if key.startswith("QOS"):
        return "a QOS limit is reached; it starts when usage under that QOS drops"
    return None


def will_never_run(reason: str | None) -> bool:
    return reason_key(reason) in NEVER_RUNS


# ---------------------------------------------------------------- exit explanations
def signal_name(n: int) -> str:
    """Linux names whatever the local OS: the job ran on a Linux node."""
    return platform.linux_signal_name(n)


def preempt_mode(slurm_config: Mapping[str, str]) -> str:
    return (slurm_config.get("PreemptMode") or "").split(",", 1)[0].strip().upper()


def preempt_text(slurm_config: Mapping[str, str]) -> str:
    mode = preempt_mode(slurm_config)
    if mode == "REQUEUE":
        return "preempted; Slurm will requeue it (PreemptMode REQUEUE)"
    if mode == "CANCEL":
        return "preempted and killed (PreemptMode CANCEL: killed, resubmit)"
    return f"preempted (PreemptMode {mode or 'unknown'})"


def short_summary(job: Job, *, uid: int | None, slurm_config: Mapping[str, str]) -> str | None:
    """One-line reason a job ended (NODE/REASON column, Jobs view); None for success/active jobs."""
    st = job.state
    if st is JobState.CANCELLED:
        by = job.raw.get("cancelled_by_uid")
        if by is None:
            return "cancelled"
        return "cancelled by you" if uid is not None and by == str(uid) else f"cancelled by uid {by}"
    if st is JobState.PREEMPTED:
        return "preempted; will requeue" if preempt_mode(slurm_config) == "REQUEUE" else "preempted, killed"
    if st is JobState.TIMEOUT:
        return "hit time limit"
    if st is JobState.OUT_OF_MEMORY:
        return "out of memory"
    if st is JobState.NODE_FAIL:
        return "node failure"
    if st is JobState.BOOT_FAIL:
        return "node boot failure"
    if st is JobState.DEADLINE:
        return "deadline reached"
    if st is JobState.FAILED:
        code = job.exit_code
        if code is not None and code >= 128:
            return f"exit {code} = {signal_name(code - 128)}"
        return f"exit {code}" if code is not None else "failed"
    return None


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit, k in (("T", 4), ("G", 3), ("M", 2), ("K", 1)):
        if n >= 1024 ** k:
            return f"{n / 1024 ** k:.1f}{unit}"
    return f"{n}B"


def explain(job: Job, steps: Sequence[JobStep], *, uid: int | None, slurm_config: Mapping[str, str]) -> tuple[str, ...]:
    """Detail-screen "Why it ended" lines (§5.2.6): signals on steps, preemption per PreemptMode, OOM, timeout,
    cancellation. A queued job gets what to do about its pending reason and the scheduler's start estimate, if squeue
    gave one, with the caveat that it is a guess (§5.2.8)."""
    out: list[str] = []
    st = job.state
    if st.group is StateGroup.QUEUED:
        advice = reason_advice(job.reason)
        if advice:
            out.append(advice)
        if job.est_start and job.start_time is not None:
            out.append(f"est. start {humanize.local_dt(job.start_time)}: Slurm's backfill guess — it moves when "
                       "running jobs end early or higher-priority jobs arrive")
        return tuple(out)
    if st is JobState.PREEMPTED:
        out.append(preempt_text(slurm_config))
    elif st is JobState.OUT_OF_MEMORY:
        max_rss = max((s.max_rss_bytes for s in steps if s.max_rss_bytes is not None), default=None)
        req = job.raw.get("ReqMem") or job.raw.get("MinMemory") or "?"
        req_b = req_mem_bytes(job)
        if max_rss is not None and req_b is not None and max_rss < req_b:
            # MaxRSS is sampled (JobAcctGatherFrequency): a short spike past the limit falls between samples
            out.append(f"out of memory: killed above --mem {req}; MaxRSS (sampled) {_fmt_bytes(max_rss)} — the spike "
                       "was likely missed between samples")
        else:
            out.append(f"out of memory: exceeded --mem {req}; MaxRSS {_fmt_bytes(max_rss)}")
    elif st is JobState.TIMEOUT:
        limit = job.raw.get("Timelimit") or job.raw.get("TimeLimit") or "?"
        out.append(f"hit time limit {limit}")
    elif st is JobState.CANCELLED:
        summary = short_summary(job, uid=uid, slurm_config=slurm_config)
        if summary:
            out.append(summary)
    for step in steps:
        if step.step_id.endswith(".extern"):
            continue
        code = step.exit_code
        if code is not None and code >= 128:
            n = code - 128
            out.append(f"{step.step_id.rpartition('.')[2] or 'job'} step {step.raw_state} {code}:"
                       f"{step.exit_signal or 0} = {signal_name(n)} (128+{n})")
        elif step.exit_signal == 125 and (step.state is JobState.OUT_OF_MEMORY or st is JobState.OUT_OF_MEMORY):
            continue                    # `0:125` is Slurm's OOM marker, not signal 125 (said above)
        elif step.exit_signal:
            out.append(f"{step.step_id} killed by {signal_name(step.exit_signal)}")
        elif code and "." in step.step_id and step.state is JobState.FAILED:
            out.append(f"{step.step_id.rpartition('.')[2]} step exited with code {code} (the job script / program "
                       "failed; not a Slurm kill)")
    if not out and st is JobState.FAILED and job.exit_code:
        out.append(f"exited with code {job.exit_code}")
    return tuple(dict.fromkeys(out))


def min_job_age_s(slurm_config: Mapping[str, str]) -> int:
    """`MinJobAge = 300 sec` → 300 (Slurm's default when unknown)."""
    return seconds_prefix(slurm_config.get("MinJobAge")) or 300


def requeue_window_open(job: Job, now: datetime, slurm_config: Mapping[str, str]) -> bool:
    """A terminal job can still be requeued while the controller remembers it (ended < MinJobAge ago)."""
    return job.end_time is not None and (now - job.end_time).total_seconds() < min_job_age_s(slurm_config)


def req_mem_bytes(job: Job) -> int | None:
    return parse_mem_bytes(job.raw.get("ReqMem") or job.raw.get("MinMemory"))
