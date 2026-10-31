"""[WP2] FROZEN Collector ABCs, Cadence, MonitorView, CollectorOutput, Action* (DESIGN.md §3.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal

from runfleet.config import GeneralSettings, HostConfig
from runfleet.errors import AttachBlocked
from runfleet.models import (GPU, AccountReport, AccountSummary, ActionResult, AttachMode, Capabilities, HostError,
                             HostMetrics, Job, JobDetail, JobKey, JobLive, Node, PartitionSummary, Proc, UsageWindow)
from runfleet.probe.section import ProbeScript, Section, SectionResult

class Role(StrEnum):
    SCHEDULER = "scheduler"
    MONITOR = "monitor"

@dataclass(frozen=True, slots=True)
class Cadence:
    group: str                   # "<collector>.<name>", unique per host: "slurm.jobs", "procfs.host", ...
    interval: float              # interval/piggyback: period in seconds at VISIBLE; event: ignored (use 0)
    floor: float = 0.0           # never faster than this, whatever the factors
    mode: Literal["interval", "piggyback", "event"] = "interval"
    # interval : defines the host tick (tick = min effective interval over interval groups) and is due at a tick
    #            once its own period has elapsed
    # piggyback: never defines the tick; due at a tick once `interval` has elapsed (procfs.host on Slurm hosts)
    # event    : due only when requested via CollectorOutput.followups or HostPoller.wake(groups=...); served at
    #            the next tick
    hidden_factor: float = 3.0   # multiplier when the host is not on screen
    focused_factor: float = 1.0  # multiplier when the host is focused (detail screen)
    max_backoff: float = 600.0   # cap of this group's failure backoff
    focused_only: bool = False   # due only while the host is focused (a detail screen is open); served at once when
                                 # the host becomes focused and the last run is older than the period

@dataclass(frozen=True, slots=True)
class HostContext:
    host: HostConfig
    general: GeneralSettings
    caps: Capabilities           # caps.profile is the host's collector list
    now: datetime                # wall clock (UTC) at tick

@dataclass(frozen=True, slots=True)
class MonitorView:
    """Latest SUCCESSFUL monitor state of this host, kept across ticks by the poller (not only this tick)."""
    gpus: tuple[GPU, ...] = ()
    gpu_procs: tuple[Proc, ...] = ()     # ps rows of GPU-holding pids (all users), same sample as gpus
    gpus_at: datetime | None = None      # sample time of gpus/gpu_procs; None = never sampled or no GPU monitor
    metrics: HostMetrics | None = None   # metrics.sampled_at is its sample time

    def gpus_age(self, now: datetime) -> float | None:
        return None if self.gpus_at is None else (now - self.gpus_at).total_seconds()

@dataclass(frozen=True, slots=True)
class CollectorOutput:
    jobs: Mapping[str, Job] | None = None          # FULL current job set of this scheduler (None = unchanged)
    gpus: tuple[GPU, ...] | None = None            # scheduler output replaces the monitor's (annotated GPUProcess.job)
    gpu_procs: tuple[Proc, ...] | None = None      # monitors only; feeds MonitorView
    procs: tuple[Proc, ...] | None = None          # scheduler only; HostSnapshot.procs
    nodes: tuple[Node, ...] | None = None
    partitions: tuple[PartitionSummary, ...] | None = None
    metrics: HostMetrics | None = None
    caps_patch: Mapping[str, object] | None = None # Capabilities field -> value learned at runtime ({"gpu_tres": True})
    errors: tuple[HostError, ...] = ()             # section-level, classified
    followups: frozenset[str] = frozenset()        # cadence groups (event or interval) to run at the next tick
    handled: frozenset[str] = frozenset()          # sections whose non-zero exit the collector read as an answer
                                                   # (sprio: no job pending / not multifactor): ok, no backoff
    account: AccountSummary | None = None          # Account (§5.2.11): scheduler only; None = unchanged
# Merge rule (poller, one tick): EVERY collector of the profile has ingest() called on EVERY tick, with an empty
# results map when none of its sections ran. Outputs are applied monitors first (profile order), scheduler last; for
# every Optional field the last non-None value wins; errors are concatenated; followups and handled are unioned.
# MonitorView is updated from monitor outputs only, and only from fields that are not None. A scheduler that annotates
# GPUs (process) therefore re-annotates MonitorView.gpus from its last known tasks on every tick and returns
# gpus/procs whenever MonitorView.gpus_at is not None, so annotations never blink on ticks without its own sections.

class Collector(ABC):
    name: ClassVar[str]          # "slurm" | "process" | "nvidia" | "procfs" (== section-name prefix)
    role: ClassVar[Role]

    def __init__(self, host: HostConfig) -> None:
        self.host = host

    @classmethod
    @abstractmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool: ...

    @classmethod
    def capability_sections(cls, host: HostConfig) -> list[Section]:
        return []

    @classmethod
    def ingest_capabilities(cls, results: Mapping[str, SectionResult]) -> Mapping[str, object]:
        return {}                # Capabilities field name -> value

    @abstractmethod
    def cadences(self, ctx: HostContext) -> list[Cadence]: ...

    @abstractmethod
    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]: ...

    @abstractmethod
    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput: ...

class MonitorCollector(Collector):
    role: ClassVar[Role] = Role.MONITOR

@dataclass(frozen=True, slots=True)
class ActionParam:
    name: str                    # "escalate", "grace_s", "hold"
    kind: Literal["bool", "int"]
    default: bool | int
    label: str                   # "send SIGKILL to survivors after"
    min: int | None = None
    max: int | None = None

@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str                      # §6.1: slurm.cancel slurm.hold slurm.release slurm.requeue
                                 #       task.terminate task.interrupt task.kill
    label: str                   # "Cancel job"
    mutating: bool               # disabled under --readonly
    confirm: bool                # ConfirmScreen before execute (True for every v0.1 action)
    params: tuple[ActionParam, ...] = ()
    disabled_reason: str | None = None   # greyed with reason ("protected partition labpi", "read-only mode")

@dataclass(frozen=True, slots=True)
class ActionPlan:
    spec: ActionSpec
    host: str
    targets: tuple[JobKey, ...]          # eligible targets only
    skipped: tuple[tuple[JobKey, str], ...]   # (target, reason) listed in the confirm body
    params: Mapping[str, object]
    sections: tuple[Section, ...]        # executed as one probe (purpose="action")
    preview: str                         # exact remote command(s) shown in the confirm dialog
    confirm_title: str                   # "Cancel 3 jobs on k2?"
    confirm_body: str                    # targets with name/state, consequences

@dataclass(frozen=True, slots=True)
class AttachPlan:                        # one Attach sample (§7.16), executed as ONE login probe (purpose="attach")
    key: JobKey
    mode: AttachMode
    job_id_raw: str                      # the target job (squeue JobID / sacct JobIDRaw)
    nodes: tuple[str, ...]               # compute nodes sampled; nodes[i] is carried by section "slurm.attach.node<i>"
    sections: tuple[Section, ...]        # "slurm.attach.node0" … "slurm.attach.node<n-1>", then "slurm.attach.audit"
                                         # (late); every node section carries a nested node script (base64 in argv)
    inner: Mapping[str, ProbeScript]     # outer section name -> the nested node script it carries; its framed output
                                         # is that outer section's stdout (parse with probe.framing.parse)
    preview: str                         # the exact login-side command of node0 (shown in the srun confirmation)
    sacct_rows: int = 0                  # permanent sacct rows this plan adds if it runs (srun: len(nodes); ssh: 0)

class SchedulerCollector(Collector):
    role: ClassVar[Role] = Role.SCHEDULER

    # ---- Attach (§7.16). Not abstract: the defaults mean "this scheduler cannot attach" (workstations, third-party
    # collectors). Core calls these only on the scheduler of the job's host; all three are sans-IO.
    def attach_modes(self, job: Job, ctx: HostContext) -> tuple[AttachMode, ...]:
        """Modes that can technically read `job` now, in preference order ("ssh" before "srun"). Ignores the
        `attach` config setting, read-only mode and consent (core applies those policies) but honours what is known:
        () unless the job is RUNNING with known nodes; "ssh" dropped when ctx.caps.node_ssh == "denied"; "srun"
        dropped when srun is missing from ctx.caps.tools or the Slurm version lacks --overlap."""
        return ()

    def plan_attach(self, job: Job, mode: AttachMode, ctx: HostContext, *,
                    node: str | None = None) -> AttachPlan:
        """Plan one sample. `node` limits it to that node (must be one of job.nodes); default: ssh = up to 4 of
        job.nodes (batch host first), srun = the batch host only (one step = one sacct row). Pure and cheap; raises
        errors.AttachBlocked when the job cannot be read this way (not running, unknown node, unsafe value)."""
        raise AttachBlocked(f"{self.name}: attach is not supported for this job")

    def ingest_attach(self, plan: AttachPlan, results: Mapping[str, SectionResult], ctx: HostContext, *,
                      previous: JobLive | None = None) -> tuple[JobLive, Mapping[str, object]]:
        """Parse one sample. Returns the JobLive and a caps_patch ({"node_ssh": "adopt"} after a landing,
        {"node_ssh": "denied"} after an auth refusal; {} otherwise). `previous` = the last JobLive of the same job
        with data (any mode): CPU cores (Δcpu.stat), idle streaks and the "GPU idle" hint need it. Never raises for
        remote failures: they become JobLive.errors / NodeInspect.errors. May update the collector's own state (the
        GPU model from nvidia-smi becomes the job's alloc.gpu_type, gpu_type_source="nvidia-smi", on later ticks)."""
        raise NotImplementedError(f"{self.name}: attach is not supported")

    # ---- Account (§5.2.11). Not abstract: the defaults mean "no scheduler accounting" (workstations, third-party
    # collectors). Core calls these only for FleetAPI.account_report; both are sans-IO.
    def account_sections(self, window: UsageWindow, ctx: HostContext) -> list[Section]:
        """Sections of one on-demand account report for `window`, rendered as ONE probe (purpose="account");
        [] = nothing to read (no accounting, disabled in config, no account). A "period" without a known start is
        planned as "30d" (the report's `window` says so)."""
        return []

    def ingest_account(self, window: UsageWindow, results: Mapping[str, SectionResult],
                       ctx: HostContext) -> AccountReport | None:
        """Parse one report. Never raises for remote failures (they become AccountReport.errors); None only when
        account_sections gave []. May refresh the collector's own summary state (balances, sbank dates), so the next
        tick publishes them in CollectorOutput.account."""
        return None

    def restore_account(self, summary: AccountSummary | None) -> None:
        """Warm start: the persisted `HostSnapshot.account`, handed over before every on-demand report. A collector
        may take from it what it otherwise keeps in memory only (Slurm: the sbank reallocation dates the "period"
        window starts from, until its own first sbank reading of this run). Default: nothing."""
        return None

    @abstractmethod
    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]: ...
        # every action this backend offers for these jobs; disabled_reason from state/ownership/protection.
        # A spec is enabled if >= 1 job is eligible.

    @abstractmethod
    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan: ...
        # pure and cheap: ConfirmScreen calls it again whenever a param changes; raises ActionError if no target
        # is eligible

    @abstractmethod
    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]: ...

    @abstractmethod
    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]: ...

    @abstractmethod
    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail: ...

    @abstractmethod
    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None: ...
        # remote command for `ssh -t <alias> <cmd>`; None = plain login shell. Task: cd to its cwd; running Slurm job:
        # a shell inside the job on `node` (default: its batch host); other Slurm jobs: cd to WorkDir on the login node
