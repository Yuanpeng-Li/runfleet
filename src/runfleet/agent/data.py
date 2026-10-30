"""[WP4] Data shapes shared by `agent.source` (IO: cache, refresh, logs) and `agent.view` (pure: projections, text,
JSON): one host as the agent sees it, the fleet view, problem codes, the rendered document (DESIGN.md §8.7)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from runfleet.models import ConnState, HostSnapshot, Job

# Probe sections whose last success dates one kind of data (HostHealth.data_at). Jobs: the listing that would show a
# job (`wait.JOB_SECTIONS`); history: accounting; nodes: partitions / free GPUs (sinfo) or workstation GPUs — each
# dated by its own section (NODE_FIELDS): a Slurm host's nodes data is its partitions, never its login-node GPUs.
SECTION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "jobs": ("slurm.squeue", "process.ps"),
    "history": ("slurm.sacct",),
    "nodes": ("slurm.sinfo", "nvidia.gpu"),
    "metrics": ("procfs.host",),
}
# the section that reads each kind of node data (HostSnapshot.partitions / .gpus)
NODE_FIELDS: Mapping[str, str] = {"partitions": "slurm.sinfo", "gpus": "nvidia.gpu"}


def node_field(sections: Mapping[str, object]) -> str:
    """The host's main node data: its partitions on a Slurm host (a slurm.sinfo section), else its GPUs."""
    return "partitions" if NODE_FIELDS["partitions"] in sections else "gpus"

STALE_AFTER = 600.0          # seconds: job data older than this makes a host "stale" (exit 4), as `runfleet status`
OLD_AFTER = 300.0            # seconds: a job seen longer ago is shown "as of <age> ago", its times not extrapolated
REFRESH_FLOOR = 60.0         # seconds: --refresh skips a host whose job data is younger than this
EXIT_OK, EXIT_NO_DATA, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 4


def dest_hash(ssh: str) -> str:
    """Identity of a host's ssh destination, stored with its cached health: a cache row written for another machine
    under the same display name (a second `-c` config) is not this host's data."""
    return hashlib.sha1(ssh.encode()).hexdigest()[:12]


class Source(StrEnum):
    TUI = "tui"              # another RunFleet holds the host's instance lock and its job data is advancing
    CACHE = "cache"          # cached data nobody is refreshing
    LIVE = "live"            # this call refreshed it (--refresh)
    DEMO = "demo"            # --demo: the simulated fleet, polled in memory


class ProblemCode(StrEnum):
    AUTH_REQUIRED = "auth_required"
    HOSTKEY = "hostkey"
    DNS = "dns"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    JUMP = "jump"
    STALE = "stale"
    DISABLED = "disabled"
    NO_DATA = "no_data"
    NOT_FOUND = "not_found"
    OUTSIDE_COVERAGE = "outside_coverage"
    LISTING_FAILED = "listing_failed"
    NO_LOG_PATH = "no_log_path"
    NO_LOG_YET = "no_log_yet"         # the job has not started / its log file does not exist yet
    NOT_ALLOWED = "not_allowed"       # another user's log
    NOT_LISTED = "not_listed"         # `log --file`: not a log file found under the job's workdir (outside it, the
                                      # workdir itself, a directory, or not among the files the find of this call listed)
    PARTIAL = "partial"
    CACHE_BUSY = "cache_busy"
    CONFIG_MISMATCH = "config_mismatch"
    NO_SSH = "no_ssh"
    ERROR = "error"
    INTERNAL = "internal"


# why data is stale / a refresh did not happen (Problem.reason for ProblemCode.STALE)
StaleReason = Literal["tui_idle", "no_poller", "refresh_skipped", "refresh_failed", "rate_limited", "offline",
                      "age_unknown", "refresh_running"]   # refresh_running: another RunFleet's refresh holds the host


@dataclass(frozen=True, slots=True)
class Problem:
    """A stable code plus parameters; agent.view turns it into a fixed message and hint (never raw ssh stderr)."""
    code: ProblemCode
    host: str | None = None
    job: str | None = None            # "k2:2332861" for job-level problems
    since: datetime | None = None
    reason: str | None = None         # StaleReason for STALE; a short fixed word for others
    key: str | None = None            # the job's full key for commands ("labstation:p363942-1788696459": a task's
                                      # display id can name two tasks sharing a pid); None: `job` as it is


@dataclass(frozen=True, slots=True)
class SectionHealth:
    name: str
    ok: bool
    outcome: str                      # SectionOutcome value
    last_ok: datetime | None
    error_kind: str | None            # ErrorKind value; the summary is never stored (it may name a jump host)


@dataclass(frozen=True, slots=True)
class HostHealth:
    """Cache table `host_health` (one row per host, rewritten with every snapshot by the poller); None for caches
    written by RunFleet < 0.8."""
    at: datetime
    state: ConnState
    dest_hash: str | None
    remote_user: str | None
    mux: str | None
    last_ok: datetime | None
    latency_s: float | None
    consecutive_failures: int
    controller_slowdown: float
    tick_interval: float | None
    sections: Mapping[str, SectionHealth] = field(default_factory=dict)

    def data_at(self, group: str) -> datetime | None:
        """Newest success among the group's sections (SECTION_GROUPS); None = never / not this backend. "nodes":
        the main node data's own section only (:func:`node_field`: slurm.sinfo on a Slurm host — a working
        login-node nvidia-smi never dates carried-over partitions —, nvidia.gpu on a workstation)."""
        names = (NODE_FIELDS[node_field(self.sections)],) if group == "nodes" else SECTION_GROUPS.get(group, ())
        oks = [s.last_ok for n in names if (s := self.sections.get(n)) and s.last_ok]
        return max(oks) if oks else None

    def failing(self) -> list[SectionHealth]:
        return [s for s in self.sections.values() if not s.ok]


@dataclass(frozen=True, slots=True)
class HostView:
    """One configured host as the agent sees it (built by agent.source from the cache, a refresh or the demo)."""
    name: str
    kind: Literal["slurm", "workstation", "unknown"]
    enabled: bool
    state: ConnState | None                    # None: never recorded
    state_since: datetime | None
    error_kind: str | None                     # ErrorKind value of the current error, if any
    snapshot: HostSnapshot | None              # None: no data
    health: HostHealth | None
    ended_tasks: tuple[Job, ...] = ()          # workstation tasks from the cache's ended_tasks table (7 days)
    source: Source = Source.CACHE
    session_transport: bool = False            # auth recovery is "open the TUI and press c", not `runfleet connect`
    sacct_window_s: float | None = None        # Slurm accounting window (config), None for workstations
    problems: tuple[Problem, ...] = ()         # raised by agent.source (config_mismatch, refresh outcome, …)

    def data_at(self, group: str = "jobs") -> datetime | None:
        """When this kind of data was last observed: the section's last success, else (older caches, demo stubs)
        the snapshot's taken_at for jobs; None = unknown."""
        if self.health is not None:
            at = self.health.data_at(group)
            if at is not None:
                return at
            if any(n in self.health.sections for n in SECTION_GROUPS.get(group, ())):
                return None                    # the section exists but never succeeded
        if group == "jobs" and self.snapshot is not None:
            return self.snapshot.taken_at
        return None


@dataclass(frozen=True, slots=True)
class FleetView:
    generated_at: datetime
    hosts: tuple[HostView, ...]                # config order
    recent_window: float                       # general.recent_window (s)
    demo: bool = False
    stall_after: float = 1800.0                # general.stall_after (s): a running job's stdout this quiet is "quiet"
    error: str | None = None                   # no data at all (missing / busy / unreadable cache): exit 1
    problems: tuple[Problem, ...] = ()         # fleet-level (cache_busy, no_ssh, …)
    # how the call was made, repeated in every command the output suggests (runnable as printed): `-c PATH` (None:
    # the default config) and, with `demo`, `--demo-seed N` (None: the default seed, not repeated)
    config_path: str | None = None
    demo_seed: int | None = None

    def host(self, name: str) -> HostView | None:
        return next((h for h in self.hosts if h.name == name), None)


@dataclass(frozen=True, slots=True)
class Document:
    """What a verb returns: the JSON document (schema runfleet.agent.<verb>/1), the compact text and the exit code.
    CLI prints `text` (or the JSON), an MCP adapter returns both."""
    schema: str
    data: Mapping[str, Any]
    text: str
    exit_code: int = EXIT_OK
