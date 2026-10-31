"""[WP2] SlurmCollector: sections, cadences, live/history merge (§5.2).

State kept across ticks (sans-IO; time comes from `HostContext.now`):

- `_live`: the last complete squeue answer; `_history`: sacct rows of the window;
- `_jobinfo`: `scontrol show job -d -o` facts per display id (fetched once per job: first sight and PD→R);
- `_meta`: "jobmeta" — log paths / GPU type / per-node indices seen while a job was live, so history rows keep them
  after scontrol forgets the job (MinJobAge);
- `_vanished_at`: ids that left squeue but are not terminal in history yet (looked up with backoff: `_vanished_tries`);
- `_jobinfo_fail`: scontrol answered for an id without a usable record: re-asked only after a backoff;
- `_usage`, `_stats`, `_nodes`: latest sstat / logstat / sinfo results;
- `_smi_model`: the GPU model nvidia-smi reported for a job during Attach (§7.16.5): it overrides the job's gres type
  (`gpu_type_source="nvidia-smi"`) while the job is known, and so reaches jobmeta.
- `_sprio`, `_shares`: the last `sprio` rows (by raw job id) and `sshare` rows of the user (§5.2.10), read by the
  `slurm.priority` group only while an own job is pending.
- `_acct_*`: the last good readings of the `slurm.account` group (Account, §5.2.11): sshare account rows, the user's
  sreport usage, sbank balances (`_acct_sbank_at`: the last successful sbank reading, re-read after SBANK_PERIOD) and
  the classified error of each failing section; together they make `CollectorOutput.account`. An on-demand account
  report refreshes them too (`ingest_account`); `_acct_plans` holds what `account_sections` planned per window.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from runfleet.backends.base import (ActionPlan, ActionSpec, AttachPlan, Cadence, CollectorOutput, HostContext,
                                    MonitorView, SchedulerCollector)
from runfleet.backends.common import (GpuIdleTracker, ParseResult, StatInfo, complete, log_idle_hint, parse_error,
                                      parse_stat_lines, section_error, usable, with_stat)
from runfleet.backends.slurm import account as acct
from runfleet.backends.slurm import actions as slurm_actions
from runfleet.backends.slurm import attach
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm import priority as prio
from runfleet.backends.slurm.logpaths import JobNames, refs_from_paths, refs_from_templates
from runfleet.backends.slurm.parse_occupants import Occupant, nodes_with_jobs, parse_names, parse_occupants
from runfleet.backends.slurm.parse_sacct import gputres_seed, parse_sacct, parse_steps
from runfleet.backends.slurm.parse_scontrol import JobInfo, parse_scontrol
from runfleet.backends.slurm.parse_sinfo import gpu_types, parse_sinfo, summarize
from runfleet.backends.slurm.parse_squeue import GpuSource, parse_squeue, parse_squeue_legacy, resolve_gpu
from runfleet.backends.slurm.parse_sstat import SstatRow, StepUsage, job_usage, parse_sstat
from runfleet.backends.slurm.states import (explain, humanize_reason, min_job_age_s, reason_advice,
                                           short_summary)
from runfleet.config import HostConfig
from runfleet.errors import AttachBlocked
from runfleet.models import (WINDOW_DAYS, AccountReport, AccountSummary, AccountUse, ActionResult, AttachMode,
                             Capabilities, ErrorKind, FairShare, HostError, Job, JobDetail, JobKey, JobKind, JobLive,
                             JobState, JobStep, LogRef, NODE_JOBS_TTL, Node, NodeAlloc, PartitionCharge,
                             PartitionSummary, PendingPriority, StateGroup, UsageWindow)
from runfleet.probe.section import Section, SectionOutcome, SectionResult
from runfleet.util.efficiency import efficiency
from runfleet.util.efficiency import lines as efficiency_lines
from runfleet.util.slurmfmt import (expand_array_tasks, nullish, parse_float, parse_int, parse_tres, parse_tres_per,
                                   seconds_prefix, split_rows)

log = logging.getLogger("runfleet.backends.slurm.collector")

VANISHED_GRACE = timedelta(minutes=10)
MAX_JOBINFO_IDS = 8
MAX_SSTAT_IDS = 20
MAX_LOGSTAT_PATHS = 30
MAX_VANISHED_IDS = 50
MAX_VANISHED_TRIES = 6                                  # sacct -j lookups per vanished job (30 s … 10 min apart)
DETAIL_JOBINFO_MAX_AGE = timedelta(seconds=30)
OCCUPANTS_PERIOD = 30.0                                # every user's jobs on the nodes: while the detail is open
PRIORITY_PERIOD = 600.0                                # sprio + sshare (§5.2.10): at most every 10 min, own pending jobs
PRIORITY_HIDDEN = 6.0                                  # off screen: once an hour
PRIORITY_MAX_BACKOFF = 3600.0                          # failures: 20, 40, 60 min
ACCOUNT_FLOOR = 900.0                                  # `slurm.account` (§5.2.11): poll_account, never below 15 min
ACCOUNT_HIDDEN = 2.0                                   # off screen: hourly at the default 30 min
ACCOUNT_MAX_BACKOFF = 7200.0                           # failures: 1 h, 2 h
SBANK_PERIOD = timedelta(hours=6)                      # a reallocation date changes about once a year
SLURMDBD = "accounting_storage/slurmdbd"
NO_SREPORT = "sreport not installed"
NO_SREPORT_REPORT = "sreport not installed: totals from your job records"
REPORT_SBANK_PERIOD = timedelta(hours=1)               # the on-demand report re-reads sbank at most hourly

_CTLD_DOWN = re.compile(r"Unable to contact slurm controller|Socket timed out|slurm_load_jobs error|"
                        r"Zero Bytes were transmitted", re.IGNORECASE)
_DBD_DOWN = re.compile(r"Problem talking to the database|slurmdbd|Connection refused", re.IGNORECASE)
# Only errors about the output format itself downgrade the format: "Invalid user: ghost", "Invalid partition", a dbd
# "Unknown error" must not (a config typo would otherwise switch squeue to the lossy legacy -o for the session).
_BAD_FORMAT = re.compile(r"format specification|invalid (?:\w+ )?(?:format|field)|field requested|unrecognized field|"
                         r"unknown field|delimiter", re.IGNORECASE)
_PROCS_NOTE = "processes of a running job: live readings on the Procs tab (Attach, §7.16)"


@dataclass(frozen=True, slots=True)
class JobMeta:
    """Facts remembered from a job's live phase (the in-memory `jobmeta` of §5.2.4)."""
    workdir: str | None
    stdout: str | None
    stderr: str | None                 # None = merged with stdout
    gpu_type: str | None
    per_node: tuple[NodeAlloc, ...]


@dataclass(frozen=True, slots=True)
class _ReportPlan:
    """What `account_sections` planned for one on-demand account report (§5.2.11), read back by `ingest_account`."""
    window: UsageWindow                # the window answered ("period" without a start -> "30d")
    starts: Mapping[str, datetime]     # account -> its window start (period: accounts without one left out)
    inferred: frozenset[str]           # accounts whose period start is inferred (next reallocation − period)
    sreport: Mapping[str, tuple[str, ...]]   # sreport section -> its accounts
    chunks: tuple[str, ...]            # the sacct chunk sections, oldest first
    clipped: bool                      # the period was clipped at MAX_PERIOD_DAYS: the report is truncated
    sep: str                           # sacct separator of the chunks


class SlurmCollector(SchedulerCollector):
    name = "slurm"

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        self._live: dict[str, Job] = {}
        self._history: dict[str, Job] = {}
        self._have_jobs = False
        self._last_sacct_ok: datetime | None = None
        self._jobinfo: dict[str, JobInfo] = {}
        self._jobinfo_at: dict[str, datetime] = {}
        self._jobinfo_gone: set[str] = set()
        self._want_jobinfo: dict[str, None] = {}          # ordered set of display ids
        self._meta: dict[str, JobMeta] = {}
        self._vanished_at: dict[str, datetime] = {}
        self._vanished_last: dict[str, Job] = {}
        self._vanished_tries: dict[str, tuple[int, datetime]] = {}   # display id -> (lookups so far, next due)
        self._jobinfo_fail: dict[str, tuple[int, datetime]] = {}     # display id -> (failures so far, next due)
        self._usage: dict[str, StepUsage] = {}             # by job_id_raw
        self._sstat_rows: dict[str, list[SstatRow]] = {}   # every step row of the last sstat, by job_id_raw
        self._gpu_idle = GpuIdleTracker()
        self._stats: dict[str, StatInfo] = {}
        self._nodes: tuple[Node, ...] | None = None
        self._node_types: dict[str, frozenset[str]] = {}
        self._gpu_tres: bool | None = None                 # learned at runtime (None: use caps)
        self._gpu_tres_misses: set[str] = set()
        self._sep: dict[str, str] = {"squeue": cmds.SEP, "sacct": cmds.SEP, "sinfo": cmds.SEP}
        self._squeue_legacy = False
        self._sacct_fields: list[str] = []
        self._sinfo_parts: tuple[str, ...] = ()            # partitions of the last planned sinfo
        self._sstat_last: dict[str, datetime] = {}         # job id -> when its usage was last planned
        self._last_usage_at: datetime | None = None         # when the last background sstat was planned
        self._smi_model: dict[str, tuple[str, float | None]] = {}   # display id -> (gres-style type, MiB per GPU)
        self._occupants: tuple[Occupant, ...] = ()         # every user's running jobs in scope (§5.2.9)
        self._occupants_at: datetime | None = None
        self._real_names: dict[str, str | None] = {}       # user -> GECOS name (None: looked up, none known)
        self._occ_sep = cmds.SEP                           # separator of the last planned occupants squeue
        self._names_asked: list[str] = []                  # users of the last planned getent
        self._sprio: dict[str, list[prio.SprioRow]] = {}   # raw job id -> sprio rows (one per partition), §5.2.10
        self._sprio_at: datetime | None = None
        self._shares: tuple[FairShare, ...] = ()           # the user's sshare rows
        self._prio_unsupported = False                     # sprio said the cluster is not multifactor (stale caps)
        self._acct_names: list[str] = []                   # accounts of the last planned `slurm.account` sections
        self._acct_share: tuple[acct.ShareRow, ...] | None = None   # last good sshare -A rows (None: never read)
        self._acct_read_at: datetime | None = None
        self._acct_usage: tuple[AccountUse, ...] = ()      # the user's sreport usage per account
        self._acct_usage_at: datetime | None = None
        self._acct_usage_window: UsageWindow | None = None   # of _acct_usage (a restored one may be of another)
        self._acct_no_sreport = False                      # sreport is not installed: no usage, a note
        self._acct_sbank: dict[str, acct.SbankBalance] = {}   # lower-case account -> last good sbank balance
        self._acct_sbank_at: datetime | None = None
        self._acct_seed_at: datetime | None = None         # the restored summary's sshare time (warm start)
        self._acct_errors: dict[str, HostError] = {}       # section -> its classified failure, until it answers
        self._acct_plans: dict[str, _ReportPlan] = {}      # requested window -> the on-demand report planned for it
        self._acct_fresh = False                           # a report refreshed the state: publish at the next tick

    # ------------------------------------------------------------ detection
    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return host.backend == "slurm" or (host.backend == "auto" and {"squeue", "sinfo"} <= caps.tools)

    @classmethod
    def capability_sections(cls, host: HostConfig) -> list[Section]:
        return [
            Section("slurm.version", cmds.VERSION_CMD, timeout=10, requires=("sinfo",)),
            Section("slurm.config", cmds.CONFIG_CMD, timeout=10, requires=("scontrol",)),
            Section("slurm.assoc", cmds.ASSOC_CMD, timeout=15, requires=("sacctmgr",)),
            Section("slurm.parts", cmds.PARTS_CMD, timeout=15, requires=("scontrol",)),
            Section("slurm.qos", cmds.QOS_CMD, timeout=15, requires=("sacctmgr",), max_stdout=cmds.QOS_MAX_STDOUT),
            Section("slurm.sacctfmt", cmds.SACCTFMT_CMD, timeout=10, requires=("sacct",)),
            Section("slurm.gputres", cmds.GPUTRES_CMD, timeout=30, requires=("sacct",)),
        ]

    @classmethod
    def ingest_capabilities(cls, results: Mapping[str, SectionResult]) -> Mapping[str, object]:
        return parse_capabilities(results)

    # ------------------------------------------------------------ planning
    def cadences(self, ctx: HostContext) -> list[Cadence]:
        s = ctx.host.slurm
        usage_floor = float(max(30, ctx.caps.acct_gather_freq_s or 0))
        return [
            Cadence("slurm.jobs", max(s.poll_jobs, 10.0), floor=10.0, max_backoff=300.0),
            Cadence("slurm.jobinfo", 0.0, mode="event"),
            Cadence("slurm.usage", max(s.poll_usage, usage_floor), floor=usage_floor, focused_factor=0.5),
            Cadence("slurm.history", max(s.poll_history, 60.0), floor=60.0),
            Cadence("slurm.vanished", 0.0, mode="event"),
            Cadence("slurm.logstat", max(s.poll_logstat, 60.0), floor=60.0, focused_factor=0.5),
            Cadence("slurm.nodes", max(s.poll_nodes, 60.0), floor=60.0, hidden_factor=5.0),
            Cadence("slurm.occupants", OCCUPANTS_PERIOD, floor=OCCUPANTS_PERIOD, focused_only=True),
            Cadence("slurm.names", 0.0, mode="event"),
            *([Cadence("slurm.priority", PRIORITY_PERIOD, floor=PRIORITY_PERIOD, hidden_factor=PRIORITY_HIDDEN,
                       max_backoff=PRIORITY_MAX_BACKOFF)] if self._priority_wanted(ctx) else []),
            *([Cadence("slurm.account", max(s.poll_account, ACCOUNT_FLOOR), floor=ACCOUNT_FLOOR, mode="piggyback",
                       hidden_factor=ACCOUNT_HIDDEN, max_backoff=ACCOUNT_MAX_BACKOFF)]
              if self._account_wanted(ctx) else []),
        ]

    def _priority_pending(self, ctx: HostContext) -> list[Job]:
        """Own pending jobs that compete for a start (not held)."""
        me = ctx.caps.remote_user
        return [j for j in self._live.values()
                if j.state in (JobState.PENDING, JobState.REQUEUED) and not j.held and j.is_mine
                and (not me or j.user == me)]

    def _priority_wanted(self, ctx: HostContext) -> bool:
        """`slurm.priority` exists only while an own job is pending on a cluster that ranks by a multifactor priority
        with some weight > 0 (§5.2.10) and sprio or sshare is there."""
        st = prio.settings(ctx.caps.slurm_config)
        tools = ctx.caps.tools
        if not st.ranked or self._prio_unsupported or (tools and not {"sprio", "sshare"} & tools):
            return False
        return bool(self._priority_pending(ctx))

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        out: list[Section] = []
        if "slurm.jobs" in due:
            out.append(self._squeue_section(ctx))
        if "slurm.jobinfo" in due and (sec := self._scontrol_section(ctx.now)) is not None:
            out.append(sec)
        if "slurm.usage" in due and (sec := self._sstat_section(ctx)) is not None:
            out.append(sec)
        if "slurm.history" in due:
            out.append(self._sacct_section(ctx))
        if "slurm.vanished" in due and (sec := self._vanished_section(ctx)) is not None:
            out.append(sec)
        if "slurm.logstat" in due and (sec := self._logstat_section()) is not None:
            out.append(sec)
        if "slurm.nodes" in due and (sec := self._sinfo_section(ctx)) is not None:
            out.append(sec)
        if "slurm.occupants" in due:
            out += self._occupant_sections(ctx)       # with sinfo: the GPU counts and the jobs of one moment
        if "slurm.names" in due and (sec := self._names_section()) is not None:
            out.append(sec)
        if "slurm.priority" in due:
            out += self._priority_sections(ctx)
        if "slurm.account" in due:
            out += self._account_sections(ctx)
        return out

    def _account_wanted(self, ctx: HostContext) -> bool:
        """`slurm.account` (§5.2.11): `account_usage` on, slurmdbd accounting, at least one account of this cluster
        in the associations, and sshare, sreport or sbank there (tools unknown: assumed)."""
        caps = ctx.caps
        if not ctx.host.slurm.account_usage or caps.slurm_config.get("AccountingStorageType") != SLURMDBD:
            return False
        if caps.tools and not {"sshare", "sreport", "sbank"} & caps.tools:
            return False
        return bool(acct.accounts(caps))

    def _account_sections(self, ctx: HostContext) -> list[Section]:
        """The periodic summary, piggybacked on a regular tick: sshare -A (balances) + sreport (the user's usage over
        `general.usage_window`, "period" -> 30d) + sbank when the last good reading is older than SBANK_PERIOD."""
        if not self._account_wanted(ctx):
            return []
        names = self._acct_names = acct.accounts(ctx.caps)
        period = timedelta(seconds=max(ctx.host.slurm.poll_account, ACCOUNT_FLOOR))
        if self._acct_read_at is None and self._acct_seed_at is not None and ctx.now - self._acct_seed_at < period:
            return []           # warm start, `agent --refresh`: the restored summary is younger than its period
        self._acct_seed_at = None
        tools = ctx.caps.tools
        out: list[Section] = []
        if not tools or "sshare" in tools:
            out.append(Section("slurm.acct_share", cmds.acct_share_cmd(names), timeout=15, requires=("sshare",),
                               max_stdout=cmds.ACCT_SHARE_MAX_STDOUT, meta={"accounts": names}))
        self._acct_no_sreport = bool(tools) and "sreport" not in tools
        if not self._acct_no_sreport:
            days = WINDOW_DAYS.get(ctx.general.usage_window, 30)
            out.append(Section("slurm.acct_sreport", cmds.acct_sreport_cmd(names, f"now-{days}days", totals=False),
                               timeout=20, requires=("sreport",), max_stdout=cmds.ACCT_SREPORT_MAX_STDOUT,
                               meta={"accounts": names, "days": days}))
        if "sbank" in tools and (self._acct_sbank_at is None or ctx.now - self._acct_sbank_at >= SBANK_PERIOD):
            out.append(Section("slurm.acct_sbank", cmds.acct_sbank_cmd(names), timeout=20, requires=("sbank",),
                               max_stdout=cmds.ACCT_SBANK_MAX_STDOUT, meta={"accounts": names}))
        return out

    def _priority_sections(self, ctx: HostContext) -> list[Section]:
        """sprio (own jobs) + sshare (own user; only where fair share has a weight), one round trip (§5.2.10)."""
        if not self._priority_wanted(ctx):
            return []
        tools = ctx.caps.tools
        out = []
        if not tools or "sprio" in tools:
            out.append(Section("slurm.sprio", cmds.SPRIO_CMD, timeout=15, requires=("sprio",),
                               max_stdout=cmds.SPRIO_MAX_STDOUT, meta={"format": cmds.SPRIO_FORMAT}))
        if prio.settings(ctx.caps.slurm_config).weight("fairshare") > 0 and (not tools or "sshare" in tools):
            out.append(Section("slurm.sshare", cmds.SSHARE_CMD, timeout=15, requires=("sshare",),
                               max_stdout=cmds.SSHARE_MAX_STDOUT, meta={"fields": cmds.SSHARE_FIELDS.split(",")}))
        return out

    def _squeue_section(self, ctx: HostContext) -> Section:
        s = ctx.host.slurm
        users = list(s.users) or None
        if self._squeue_legacy:
            cmd = cmds.squeue_legacy_cmd(users, s.squeue_args)
            meta: dict[str, object] = {"users": list(s.users), "extra": list(s.squeue_args), "sep": "|",
                                       "format": "legacy"}
        else:
            sep = self._sep["squeue"]
            cmd = cmds.squeue_cmd(users, s.squeue_args, sep)
            meta = {"users": list(s.users), "extra": list(s.squeue_args), "sep": sep, "format": "O"}
        return Section("slurm.squeue", cmd, timeout=20, requires=("squeue",), meta=meta)

    def _jobinfo_due(self, d: str, now: datetime) -> bool:
        fail = self._jobinfo_fail.get(d)
        return fail is None or fail[1] <= now

    def _scontrol_section(self, now: datetime) -> Section | None:
        ids = [d for d in self._want_jobinfo if d in self._live and self._jobinfo_due(d, now)][:MAX_JOBINFO_IDS]
        raws = list(dict.fromkeys(self._live[d].job_id_raw for d in ids))
        if not raws:
            return None
        return Section("slurm.scontrol", cmds.scontrol_cmd(raws), timeout=15, requires=("scontrol",),
                       meta={"ids_raw": raws})

    def _own_running(self, ctx: HostContext) -> list[Job]:
        me = ctx.caps.remote_user
        jobs = [j for j in self._live.values() if j.state is JobState.RUNNING and (not me or j.user == me)]
        return sorted(jobs, key=lambda j: j.start_time.timestamp() if j.start_time else 0.0, reverse=True)

    def _sstat_ids(self, ctx: HostContext) -> list[str]:
        """At most MAX_SSTAT_IDS own running jobs: never sampled ones first (newest first), then the least recently
        sampled, so with more jobs than slots every job still gets a sample every few rounds."""
        raws = list(dict.fromkeys(j.job_id_raw for j in self._own_running(ctx)))
        if len(raws) <= MAX_SSTAT_IDS:
            return raws
        fresh = [r for r in raws if r not in self._sstat_last]
        rest = sorted((r for r in raws if r in self._sstat_last), key=lambda r: self._sstat_last[r])
        return (fresh + rest)[:MAX_SSTAT_IDS]

    def _sstat_section(self, ctx: HostContext) -> Section | None:
        if ctx.caps.tools and "sstat" not in ctx.caps.tools:
            return None
        raws = self._sstat_ids(ctx)
        if not raws:
            if self._last_usage_at is None:
                return None
            # A job can end between squeue and sstat. An empty success clears
            # the previous sstat error instead of leaving an idle host degraded.
            # The shell no-op makes no Slurm RPC.
            return Section("slurm.sstat", ":", timeout=1, meta={"ids_raw": []})
        self._last_usage_at = ctx.now
        self._sstat_last.update(dict.fromkeys(raws, ctx.now))
        total = len({j.job_id_raw for j in self._own_running(ctx)})
        if total > len(raws):
            log.debug("%s: usage sampled for %d of %d running jobs this round", self.host.name, len(raws), total)
        return Section("slurm.sstat", cmds.sstat_cmd(raws), timeout=15, requires=("sstat",), meta={"ids_raw": raws})

    def _since_s(self, ctx: HostContext) -> int:
        window = ctx.host.slurm.sacct_window
        start = ctx.now - timedelta(seconds=window)
        if self._last_sacct_ok is not None:
            start = max(start, self._last_sacct_ok - timedelta(seconds=300))
        return max(60, int((ctx.now - start).total_seconds()))

    def _sacct_section(self, ctx: HostContext) -> Section:
        self._sacct_fields = cmds.sacct_fields(ctx.caps.sacct_fields)
        sep = self._sep["sacct"]
        since = self._since_s(ctx)
        return Section("slurm.sacct", cmds.sacct_window_cmd(since, self._sacct_fields, sep), timeout=30,
                       requires=("sacct",), meta={"since_s": since, "fields": list(self._sacct_fields), "sep": sep})

    def _vanished_due(self, now: datetime) -> list[str]:
        """Vanished ids whose next accounting lookup is due, oldest first (bounded tries with backoff)."""
        out = []
        for d in self._vanished_at:
            tries, due_at = self._vanished_tries.get(d, (0, now))
            if d in self._vanished_last and tries < MAX_VANISHED_TRIES and due_at <= now:
                out.append(d)
        return out

    def _vanished_section(self, ctx: HostContext) -> Section | None:
        raws: list[str] = []
        for d in self._vanished_due(ctx.now):
            raw = self._vanished_last[d].job_id_raw
            if raw not in raws and len(raws) >= MAX_VANISHED_IDS:
                continue                                   # the rest waits for the next round
            if raw not in raws:
                raws.append(raw)
            tries = self._vanished_tries.get(d, (0, ctx.now))[0] + 1
            self._vanished_tries[d] = (tries, ctx.now + _backoff(tries))
        if not raws:
            return None
        fields = self._sacct_fields or cmds.sacct_fields(ctx.caps.sacct_fields)
        sep = self._sep["sacct"]
        return Section("slurm.sacct_ids", cmds.sacct_ids_cmd(raws, fields, sep), timeout=30, requires=("sacct",),
                       meta={"ids_raw": raws, "fields": list(fields), "sep": sep})

    def _logstat_section(self) -> Section | None:
        paths: dict[str, None] = {}
        for job in self._live.values():
            if job.state is not JobState.RUNNING:
                continue
            for ref in self._logs_for(job):
                if ref.path and ref.stream == "stdout":
                    paths[ref.path] = None
        valid = [p for p in paths if _valid_path(p)][:MAX_LOGSTAT_PATHS]
        if not valid:
            return None
        return Section("slurm.logstat", cmds.logstat_cmd(valid), timeout=10, requires=("stat",),
                       meta={"paths": valid})

    def partitions(self, ctx: HostContext) -> list[str]:
        """sinfo scope (§5.2.1): config `partitions`, else partitions of the user's jobs (live + history) ∪ non-empty
        association partitions ∪ GPU partitions the user may submit to (detect)."""
        if ctx.host.slurm.partitions:
            return [p for p in ctx.host.slurm.partitions if cmds.NAME_RE.match(p)]
        found: dict[str, None] = {}
        for job in [*self._live.values(), *self._history.values()]:
            if job.is_mine:
                for p in (job.partition or "").split(","):
                    found[p] = None
        for cluster, _account, part, _qos in ctx.caps.associations:
            if ctx.caps.cluster_name and cluster and cluster != ctx.caps.cluster_name:
                continue                                   # a slurmdbd shared by several clusters lists them all
            for p in part.split(","):
                found[p] = None
        for p in ctx.caps.gpu_partitions:
            found[p] = None
        return sorted(p for p in found if p and cmds.NAME_RE.match(p))

    def _sinfo_section(self, ctx: HostContext) -> Section | None:
        parts = self.partitions(ctx)
        if not parts:
            return None
        sep = self._sep["sinfo"]
        self._sinfo_parts = tuple(parts)
        return Section("slurm.sinfo", cmds.sinfo_cmd(parts, sep), timeout=20, requires=("sinfo",),
                       meta={"partitions": parts, "sep": sep})

    def _occupant_sections(self, ctx: HostContext) -> list[Section]:
        """`sinfo` + every user's running jobs in the partitions in scope (§5.2.9); none after the squeue -O fallback
        (the legacy -o has no tres-alloc)."""
        sinfo = self._sinfo_section(ctx)
        if sinfo is None or self._squeue_legacy:
            return [sinfo] if sinfo is not None else []
        parts = list(self._sinfo_parts)
        nodes = sorted({n.name for n in self._nodes or ()
                        if set(n.partitions) & set(parts) and cmds.NAME_RE.match(n.name)})
        if len(nodes) > cmds.MAX_OCCUPANT_NODES:
            nodes = []
        sep = self._occ_sep = self._sep["squeue"]
        return [sinfo, Section("slurm.occupants", cmds.occupants_cmd(nodes, parts, sep), timeout=20,
                               requires=("squeue",), meta={"nodes": nodes, "partitions": parts, "sep": sep})]

    def _unnamed(self) -> list[str]:
        users = dict.fromkeys(o.user for o in self._occupants)
        return [u for u in users if u not in self._real_names and cmds.NAME_RE.match(u)][:cmds.MAX_NAME_LOOKUPS]

    def _names_section(self) -> Section | None:
        users = self._names_asked = self._unnamed()
        if not users:
            return None
        return Section("slurm.names", cmds.names_cmd(users), timeout=10, requires=("getent",),
                       meta={"users": users})

    # ------------------------------------------------------------ ingest
    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        now = ctx.now
        errors: list[HostError] = []
        followups: set[str] = set()
        ran = {name for name in results if name.startswith("slurm.")}

        if (r := results.get("slurm.scontrol")) is not None:
            self._ingest_scontrol(r, now, errors)
        if (r := results.get("slurm.squeue")) is not None:
            self._ingest_squeue(r, ctx, errors, followups)
        for name in ("slurm.sacct", "slurm.sacct_ids"):
            if (r := results.get(name)) is not None:
                self._ingest_sacct(r, ctx, errors, window=name == "slurm.sacct")
        caps_patch = self._ingest_sstat(results.get("slurm.sstat"), ctx, errors)
        if (r := results.get("slurm.logstat")) is not None:
            if usable(r):
                self._stats.update(parse_stat_lines(r.stdout))
            if not r.ok:
                errors.append(_err(r, now))
        nodes: tuple[Node, ...] | None = None
        partitions: tuple[PartitionSummary, ...] | None = None
        if (r := results.get("slurm.sinfo")) is not None:
            nodes, partitions = self._ingest_sinfo(r, ctx, errors)
        if self._ingest_occupants(results, ctx, errors) and self._unnamed():
            followups.add("slurm.names")
        handled = self._ingest_priority(results, ctx, errors)
        account = self._ingest_account(results, ctx, errors)
        if self._nodes is not None and ({"slurm.sinfo", "slurm.occupants", "slurm.names"} & set(results)):
            nodes = self._nodes_now(ctx)

        if any(self._jobinfo_due(d, now) for d in self._want_jobinfo):
            followups.add("slurm.jobinfo")
        if self._vanished_due(now):
            followups.add("slurm.vanished")                # first lookup, or a retry while accounting lags
        if self._have_jobs and (parts := self.partitions(ctx)) and tuple(parts) != self._sinfo_parts:
            followups.add("slurm.nodes")                   # partitions first known / changed: nodes now, not in 2 min
        usage_floor = timedelta(seconds=max(30, ctx.caps.acct_gather_freq_s or 0))
        if (self._have_jobs and not (ctx.caps.tools and "sstat" not in ctx.caps.tools)
                and (self._last_usage_at is None or now - self._last_usage_at >= usage_floor)
                and any(j.job_id_raw not in self._sstat_last for j in self._own_running(ctx))):
            followups.add("slurm.usage")                   # a job started running: its usage now, not in 60 s
        jobs = self._merge(ctx) if (ran and self._have_jobs) else None
        return CollectorOutput(jobs=jobs, nodes=nodes, partitions=partitions,
                               caps_patch=caps_patch or None, errors=tuple(errors), followups=frozenset(followups),
                               handled=handled, account=account)

    def _ingest_scontrol(self, r: SectionResult, now: datetime, errors: list[HostError]) -> None:
        if not r.ok:
            errors.append(_err(r, now))
        if not usable(r):
            return
        failed: list[str] = []
        for raw_id, block in parse_scontrol(r.stdout).items():
            for job in [j for j in self._live.values() if raw_id in (j.job_id_raw, j.array_job_id)]:
                d = job.key.id
                rec = block.record_for(job.array_task)
                if rec is not None:
                    self._store_jobinfo(d, rec, now)
                elif block.left_controller:
                    self._jobinfo_gone.add(d)
                else:                # an error (the loop folds it into stdout) or no matching record: back off
                    tries = self._jobinfo_fail.get(d, (0, now))[0] + 1
                    self._jobinfo_fail[d] = (tries, now + _backoff(tries))
                    failed += block.errors[:1]
                    continue         # still wanted: a PD→R refresh with cached facts is not re-requested by squeue
                self._want_jobinfo.pop(d, None)
        if failed and r.ok:
            summary = "slurmctld unreachable" if _CTLD_DOWN.search(failed[0]) else f"slurm.scontrol: {failed[0]}"
            errors.append(HostError(ErrorKind.REMOTE, summary[:80], "\n".join(dict.fromkeys(failed))[:4096],
                                    "slurm.scontrol", now))

    def _store_jobinfo(self, display: str, rec: JobInfo, now: datetime) -> None:
        self._jobinfo[display] = rec
        self._jobinfo_at[display] = now
        self._jobinfo_gone.discard(display)
        self._jobinfo_fail.pop(display, None)

    def _forget_attempt(self, display: str, raw: str) -> None:
        """A requeued job restarted (RestartCnt or its nodes changed): the previous attempt's nodes, GPU indices and
        usage no longer apply."""
        for d in (self._jobinfo, self._jobinfo_at, self._meta, self._jobinfo_fail):
            d.pop(display, None)
        self._jobinfo_gone.discard(display)
        self._usage.pop(raw, None)
        self._sstat_rows.pop(raw, None)

    def _ingest_squeue(self, r: SectionResult, ctx: HostContext, errors: list[HostError],
                       followups: set[str]) -> None:
        now = ctx.now
        if not r.ok:
            errors.append(_err(r, now))
            self._maybe_fallback("squeue", r)
        if not usable(r):
            return
        me = ctx.caps.remote_user or None
        if self._squeue_legacy:
            pr = parse_squeue_legacy(r.stdout, cmds.SQUEUE_LEGACY, host=ctx.host.name, tz=ctx.caps.tz_offset,
                                     observed_at=now, me=me)
        else:
            pr = parse_squeue(r.stdout, host=ctx.host.name, tz=ctx.caps.tz_offset, observed_at=now, me=me,
                              sep=self._sep["squeue"])
        if pr.mostly_bad:
            errors.append(parse_error(r.name, pr, now, "squeue"))
            return
        if pr.bad_lines:
            log.info("%s: %d bad squeue row(s), first: %s", ctx.host.name, len(pr.bad_lines), pr.bad_lines[0][2])
        live = {j.key.id: j for j in pr.items}
        for prev_id, kept in self._live.items():          # bad rows: keep the previous object, never "vanished"
            if prev_id not in live and (kept.job_id_raw in pr.bad_ids or prev_id in pr.bad_ids):
                live[prev_id] = kept
        for d, job in live.items():
            prev = self._live.get(d)
            if prev is not None and (job.restarts != prev.restarts or (
                    job.state is JobState.RUNNING and prev.state is JobState.RUNNING and job.nodes != prev.nodes)):
                self._forget_attempt(d, job.job_id_raw)
            if d not in self._jobinfo and d not in self._jobinfo_gone:
                self._want_jobinfo.setdefault(d, None)
            elif prev is not None and prev.state is not JobState.RUNNING and job.state is JobState.RUNNING:
                self._jobinfo_gone.discard(d)
                self._want_jobinfo.setdefault(d, None)
            self._vanished_at.pop(d, None)
            self._vanished_last.pop(d, None)
        if complete(r):
            live_arrays = {j.array_job_id for j in live.values() if j.array_job_id}
            for gone_id in self._live.keys() - live.keys():
                gone = self._live[gone_id]
                if _is_array_summary(gone) and gone.array_job_id in live_arrays:
                    continue        # the pending range shrank because a task started (`500_[1-20]` → `500_[2-20]`)
                self._vanished_at[gone_id] = now
                self._vanished_last[gone_id] = self._enrich(self._live[gone_id], ctx)
                self._vanished_tries.pop(gone_id, None)
            self._live = live
        else:                                              # failed with partial output: update, never drop
            self._live.update(live)
        for d in [d for d in self._want_jobinfo if d not in self._live]:
            del self._want_jobinfo[d]
        self._have_jobs = True

    def _ingest_sacct(self, r: SectionResult, ctx: HostContext, errors: list[HostError], *, window: bool) -> None:
        now = ctx.now
        if not r.ok:
            errors.append(_err(r, now))
            self._maybe_fallback("sacct", r)
        if not usable(r):
            return
        fields = self._sacct_fields or cmds.sacct_fields(ctx.caps.sacct_fields)
        pr = parse_sacct(r.stdout, fields, host=ctx.host.name, tz=ctx.caps.tz_offset, observed_at=now,
                         me=ctx.caps.remote_user or None, sep=self._sep["sacct"])
        if pr.mostly_bad:
            errors.append(parse_error(r.name, pr, now, "sacct"))
            return
        for job in pr.items:
            self._history[job.key.id] = job
        if window and complete(r):
            self._last_sacct_ok = now
        self._have_jobs = True

    def _ingest_sstat(self, r: SectionResult | None, ctx: HostContext,
                      errors: list[HostError]) -> dict[str, object]:
        if r is None:
            return {}
        now = ctx.now
        if not r.ok:
            errors.append(_err(r, now))
        if not usable(r):
            return {}
        pr = parse_sstat(r.stdout)
        if pr.mostly_bad:
            errors.append(parse_error(r.name, pr, now, "sstat"))
            return {}
        by_raw: dict[str, list[SstatRow]] = {}
        for row in pr.items:
            by_raw.setdefault(row.job_id_raw, []).append(row)
        live_by_raw = {j.job_id_raw: j for j in self._live.values()}
        before = self._effective_gpu_tres(ctx)
        for raw, rows in by_raw.items():
            job = live_by_raw.get(raw)
            if job is None:
                continue
            self._sstat_rows[raw] = rows
            alloc = self._resolved_alloc(job)
            usage = job_usage(rows, gpus=alloc[0], gpu_type=alloc[1], gpu_tres=self._effective_gpu_tres(ctx),
                              sampled_at=now, interval_s=ctx.host.slurm.poll_usage)
            if usage is None:
                continue
            smi = self._smi_model.get(job.key.id)
            if usage.gpu is not None and usage.gpu.mem_total_mib is None and smi is not None and smi[1]:
                usage = dataclasses.replace(usage, gpu=dataclasses.replace(usage.gpu,
                                                                           mem_total_mib=smi[1] * alloc[0]))
            self._usage[raw] = usage
            if alloc[0] > 0:
                self._learn_gpu_tres(raw, usage.row.has_gpu_keys(), self._effective_gpu_tres(ctx))
            if usage.gpu is not None:
                self._gpu_idle.add(job.key.id, now, usage.gpu.util_mean_pct)
        after = self._effective_gpu_tres(ctx)
        return {"gpu_tres": after} if after != before and after is not None else {}

    def _effective_gpu_tres(self, ctx: HostContext) -> bool | None:
        return self._gpu_tres if self._gpu_tres is not None else ctx.caps.gpu_tres

    def _learn_gpu_tres(self, raw: str, has_key: bool, known: bool | None) -> None:
        """Only GPU jobs count: key present → True; False only after two distinct GPU jobs lacked it (§5.1), and never
        once the cluster has shown it records GPU usage (a job that just started has no sample yet)."""
        if has_key:
            self._gpu_tres = True
            self._gpu_tres_misses.clear()
            return
        if known:
            return
        self._gpu_tres_misses.add(raw)
        if len(self._gpu_tres_misses) >= 2:
            self._gpu_tres = False

    def _ingest_sinfo(self, r: SectionResult, ctx: HostContext, errors: list[HostError],
                      ) -> tuple[tuple[Node, ...] | None, tuple[PartitionSummary, ...] | None]:
        now = ctx.now
        if not r.ok:
            errors.append(_err(r, now))
            self._maybe_fallback("sinfo", r)
        if not usable(r):
            return None, None
        pr = parse_sinfo(r.stdout, sep=self._sep["sinfo"])
        if pr.mostly_bad:
            errors.append(parse_error(r.name, pr, now, "sinfo"))
            return None, None
        self._nodes = tuple(pr.items)
        self._node_types = {n.name: gpu_types(n) for n in self._nodes}
        return self._nodes, summarize(self._nodes, self.partitions(ctx))

    def _ingest_occupants(self, results: Mapping[str, SectionResult], ctx: HostContext,
                          errors: list[HostError]) -> bool:
        """Every user's running jobs and their names (§5.2.9); True when a new jobs reading arrived."""
        fresh = False
        if (r := results.get("slurm.occupants")) is not None:
            if not r.ok:
                errors.append(_err(r, ctx.now))
            if usable(r):
                pr = parse_occupants(r.stdout, sep=self._occ_sep)
                if pr.mostly_bad:
                    errors.append(parse_error(r.name, pr, ctx.now, "squeue"))
                elif r.ok or pr.items:
                    self._occupants, self._occupants_at, fresh = tuple(pr.items), ctx.now, True
        if (r := results.get("slurm.names")) is not None:
            if usable(r):
                found = parse_names(r.stdout)
                self._real_names.update({u: found.get(u) for u in self._names_asked})   # unknown users: not asked again
                self._real_names.update(found)
            elif not r.ok:
                errors.append(_err(r, ctx.now))
        return fresh

    def _ingest_priority(self, results: Mapping[str, SectionResult], ctx: HostContext,
                         errors: list[HostError]) -> frozenset[str]:
        """sprio / sshare (§5.2.10). A failure keeps the last reading (the group backs off); sprio refusing a
        non-multifactor cluster (capabilities older than a config change) stops the group for the session. Returns
        the sections whose non-zero exit was such an answer (CollectorOutput.handled: no failure, no backoff)."""
        now = ctx.now
        handled: set[str] = set()
        if (r := results.get("slurm.sprio")) is not None:
            if not r.ok and prio.NO_JOBS in r.stderr + r.stdout and r.outcome is not SectionOutcome.TIMEOUT:
                self._sprio, self._sprio_at = {}, now        # none pending any more (squeue is a tick behind)
                handled.add(r.name)
            elif not r.ok and prio.UNSUPPORTED in r.stderr:
                handled.add(r.name)
                log.info("%s: sprio: %s — no priority evidence this session", self.host.name,
                         r.stderr.strip().replace("\n", " ")[:200])
                self._prio_unsupported = True
                self._sprio.clear()
            else:
                if not r.ok:
                    errors.append(_err(r, now))
                if usable(r):
                    pr = prio.parse_sprio(r.stdout)
                    if pr.mostly_bad:
                        errors.append(parse_error(r.name, pr, now, "sprio"))
                    else:
                        rows = prio.index_rows(pr.items)
                        if complete(r):
                            self._sprio = rows
                        else:
                            self._sprio.update(rows)
                        self._sprio_at = now
        if (r := results.get("slurm.sshare")) is not None:
            if not r.ok:
                errors.append(_err(r, now))
            if usable(r):
                sp = prio.parse_sshare(r.stdout)
                if sp.mostly_bad:
                    errors.append(parse_error(r.name, sp, now, "sshare"))
                elif r.ok or sp.items:
                    self._shares = tuple(sp.items)
        return frozenset(handled)

    def _ingest_account(self, results: Mapping[str, SectionResult], ctx: HostContext,
                        errors: list[HostError]) -> AccountSummary | None:
        """`slurm.account` (§5.2.11): a failing section keeps its last values (with their ages) and reports a classified
        error, both in `errors` (section status, group backoff) and in the summary. MISSING is an answer: no sreport
        -> no usage and a note; no sbank -> no reallocation dates. None when none of the group's sections ran and no
        on-demand report refreshed the state since the last tick."""
        now = ctx.now
        ran = [n for n in ("slurm.acct_share", "slurm.acct_sreport", "slurm.acct_sbank") if n in results]
        if not ran and not self._acct_fresh:
            return None
        self._acct_fresh = False
        if (r := results.get("slurm.acct_share")) is not None:
            self._ingest_share(r, now, errors)
        if (r := results.get("slurm.acct_sreport")) is not None:
            if r.outcome is SectionOutcome.MISSING:
                self._acct_no_sreport, self._acct_usage, self._acct_usage_at = True, (), None
                self._acct_errors.pop(r.name, None)
            elif self._account_ok(r, now, errors):
                sp = acct.parse_sreport(r.stdout)
                if sp.mostly_bad:
                    self._account_fail(parse_error(r.name, sp, now, "sreport"), errors)
                else:
                    self._acct_no_sreport = False
                    self._acct_usage, self._acct_usage_at = self._sreport_usage(sp.items, ctx), now
                    self._acct_usage_window = self._summary_window(ctx)
        if (r := results.get("slurm.acct_sbank")) is not None:
            self._ingest_sbank_result(r, now, errors)
        return self._account_summary(ctx)

    def _ingest_share(self, r: SectionResult, now: datetime, errors: list[HostError]) -> None:
        """`slurm.acct_share` (periodic or on demand): the sshare account rows; MISSING forgets them (no sshare)."""
        if r.outcome is SectionOutcome.MISSING:
            self._acct_share, self._acct_read_at = None, None
            self._acct_errors.pop(r.name, None)
        elif self._account_ok(r, now, errors):
            pr = acct.parse_share(r.stdout)
            if pr.mostly_bad:
                self._account_fail(parse_error(r.name, pr, now, "sshare"), errors)
            else:
                self._acct_share, self._acct_read_at = tuple(pr.items), now

    def _ingest_sbank_result(self, r: SectionResult, now: datetime, errors: list[HostError]) -> None:
        """`slurm.acct_sbank` (periodic or on demand); MISSING: no reallocation dates, not asked again for 6 h."""
        if r.outcome is SectionOutcome.MISSING:
            self._acct_sbank, self._acct_sbank_at = {}, now
            self._acct_errors.pop(r.name, None)
        elif self._account_ok(r, now, errors):
            self._ingest_sbank(r, now, errors)

    def _account_ok(self, r: SectionResult, now: datetime, errors: list[HostError]) -> bool:
        """True when `r` answered; else its classified error is kept (and reported) and the last values stay."""
        if r.ok:
            self._acct_errors.pop(r.name, None)
            return True
        err = acct.account_error(r, now)
        if err is not None:
            self._account_fail(err, errors)
        return False

    def _account_fail(self, err: HostError, errors: list[HostError]) -> None:
        if err.section:
            self._acct_errors[err.section] = err
        errors.append(err)

    def _sreport_usage(self, rows: Sequence[acct.SreportRow], ctx: HostContext, *,
                       starts: Mapping[str, datetime] | None = None,
                       inferred: frozenset[str] = frozenset()) -> tuple[AccountUse, ...]:
        """The user's usage per planned account (`starts`: account -> window start; default the summary window for
        every account); an account sreport printed no rows for used nothing (0, not unknown: sreport answered)."""
        only = starts is not None                          # a period sreport: only the accounts of that start
        if starts is None:
            days = WINDOW_DAYS.get(ctx.general.usage_window, 30)
            starts = dict.fromkeys(self._acct_names, ctx.now - timedelta(days=days))
        mine, _totals = acct.sreport_usage(rows, cluster=ctx.caps.cluster_name, starts=starts)
        got = {u.account: u for u in mine if not only or u.account in starts}
        for name, start in starts.items():
            got.setdefault(name, AccountUse(account=name, scope="me", cpu_h=0.0, gpu_h=0.0, billing_h=0.0,
                                            start=start))
        return tuple(dataclasses.replace(got[a], start_inferred=a in inferred) for a in sorted(got))

    def _ingest_sbank(self, r: SectionResult, now: datetime, errors: list[HostError]) -> None:
        """One block per account; an unknown layout is a PARSE error for that account only (its last good balance
        stays)."""
        bad: list[str] = []
        for key, bank in acct.parse_sbank(r.stdout).items():
            if bank.error is not None:
                bad.append(bank.error)
            else:
                self._acct_sbank[key] = bank
        self._acct_sbank_at = now
        if bad:
            err = HostError(ErrorKind.PARSE, f"could not read sbank for {len(bad)} account(s) (see log)",
                            "\n".join(bad)[:4096], r.name, now)
            log.info("%s: sbank: %s", self.host.name, "; ".join(bad)[:400])
            self._account_fail(err, errors)

    def _account_summary(self, ctx: HostContext) -> AccountSummary:
        caps = ctx.caps
        names = self._acct_names
        sbank = {k: v for k, v in self._acct_sbank.items() if k in {n.lower() for n in names}}
        items = acct.balances(names, self._acct_share, sbank, caps, read_at=self._acct_read_at or self._acct_sbank_at)
        window = self._summary_window(ctx)
        same = self._acct_usage_window in (None, window)
        return AccountSummary(
            window=window, read_at=self._acct_read_at or ctx.now, mode=acct.mode(items), balances=items,
            usage=self._acct_usage if same else (), usage_at=self._acct_usage_at if same else None,
            sbank_at=self._acct_sbank_at if sbank else None, half_life_s=acct.half_life_s(caps),
            reset_period=acct.reset_period(caps), notes=(NO_SREPORT,) if self._acct_no_sreport else (),
            errors=tuple(self._acct_errors[k] for k in sorted(self._acct_errors)))

    def restore_account(self, summary: AccountSummary | None) -> None:
        """Warm start (§5.2.11): a new collector (a restart, or the throwaway one of `agent --refresh`) takes from the
        persisted summary what it otherwise keeps in memory: the time of its last sshare reading (the periodic group
        is skipped while that is younger than its period) and the sbank readings with their time (reallocation dates,
        sbank usage / limits), so SBANK_PERIOD holds across processes and the "period" window is known at once; and
        the usage with its window, so an on-demand report published before the next sreport never blanks it."""
        if summary is None:
            return
        if self._acct_read_at is None and any(b.read_at is not None for b in summary.balances):
            self._acct_seed_at = summary.read_at
        if self._acct_usage_at is None and summary.usage_at is not None:   # so a report's publication keeps it
            self._acct_usage, self._acct_usage_at = summary.usage, summary.usage_at
            self._acct_usage_window = summary.window
        if self._acct_sbank_at is not None or self._acct_sbank:
            return
        for b in summary.balances:
            if b.realloc_known or "sbank" in (b.used_source, b.limit_source):
                self._acct_sbank[b.account.lower()] = acct.SbankBalance(
                    b.account, usage_su=b.used_su if b.used_source == "sbank" else None,
                    limit_su=b.limit_su if b.limit_source == "sbank" else None, next_realloc=b.next_realloc,
                    realloc_known=b.realloc_known)
        if self._acct_sbank:
            self._acct_sbank_at = summary.sbank_at

    # ------------------------------------------------------------ on-demand account report (§5.2.11)
    def account_sections(self, window: UsageWindow, ctx: HostContext) -> list[Section]:
        """One on-demand report: fresh sshare balances, sreport from the window start (the accounts' totals and the
        user's rows; for "period" one `.p<i>` per distinct start), the user's allocations in ≤ 30-day sacct chunks
        (oldest first) and sbank when its last reading is older than an hour. A "period" without a known start is
        answered as "30d". [] where `slurm.account` would not run (no slurmdbd, off in config, no account)."""
        if not self._account_wanted(ctx):
            return []
        caps, now, tools = ctx.caps, ctx.now, ctx.caps.tools
        names = self._acct_names = acct.accounts(caps)
        answered: UsageWindow = window
        starts: dict[str, datetime] = {}
        inferred: frozenset[str] = frozenset()
        clipped = False
        if window == "period":
            pw = acct.period_starts(self._account_summary(ctx), caps, ctx.host, now)
            starts, inferred, clipped = dict(pw.starts), pw.inferred, pw.clipped
            if not starts:
                answered = "30d"
        if answered != "period":
            starts = dict.fromkeys(names, now - timedelta(days=WINDOW_DAYS[answered]))
        out: list[Section] = []
        if not tools or "sshare" in tools:
            out.append(Section("slurm.acct_share", cmds.acct_share_cmd(names), timeout=15, requires=("sshare",),
                               max_stdout=cmds.ACCT_SHARE_MAX_STDOUT, meta={"accounts": names}))
        sreport: dict[str, tuple[str, ...]] = {}
        if not tools or "sreport" in tools:
            if answered == "period":
                by_start: dict[datetime, list[str]] = {}
                for name, start in sorted(starts.items()):
                    by_start.setdefault(start, []).append(name)
                for i, (start, group) in enumerate(sorted(by_start.items())):
                    day = start.astimezone(timezone(caps.tz_offset)).date().isoformat()
                    sec = f"slurm.acct_sreport.p{i}"
                    out.append(Section(sec, cmds.acct_sreport_cmd(group, day, totals=True), timeout=20,
                                       requires=("sreport",), max_stdout=cmds.ACCT_SREPORT_MAX_STDOUT,
                                       meta={"accounts": group, "start": day, "totals": True}))
                    sreport[sec] = tuple(group)
            else:
                days = WINDOW_DAYS[answered]
                out.append(Section("slurm.acct_sreport", cmds.acct_sreport_cmd(names, f"now-{days}days", totals=True),
                                   timeout=20, requires=("sreport",), max_stdout=cmds.ACCT_SREPORT_MAX_STDOUT,
                                   meta={"accounts": names, "days": days, "totals": True}))
                sreport["slurm.acct_sreport"] = tuple(names)
        sep = self._sep["sacct"]
        days = max(1, math.ceil((now - min(starts.values())).total_seconds() / 86400))
        chunks: list[str] = []
        for i, (older, newer) in enumerate(cmds.acct_chunks(days)):
            sec = f"slurm.acct_jobs.c{i}"
            out.append(Section(sec, cmds.acct_jobs_cmd(older, newer, sep), timeout=60, requires=("sacct",),
                               max_stdout=cmds.ACCT_JOBS_MAX_STDOUT,
                               meta={"older_days": older, "newer_days": newer, "sep": sep}))
            chunks.append(sec)
        if "sbank" in tools and (self._acct_sbank_at is None or now - self._acct_sbank_at >= REPORT_SBANK_PERIOD):
            out.append(Section("slurm.acct_sbank", cmds.acct_sbank_cmd(names), timeout=20, requires=("sbank",),
                               max_stdout=cmds.ACCT_SBANK_MAX_STDOUT, meta={"accounts": names}))
        self._acct_plans[window] = _ReportPlan(window=answered, starts=starts, inferred=inferred, sreport=sreport,
                                               chunks=tuple(chunks), clipped=clipped, sep=sep)
        return out

    def ingest_account(self, window: UsageWindow, results: Mapping[str, SectionResult],
                       ctx: HostContext) -> AccountReport | None:
        """The report planned by the last `account_sections(window)`. sshare and sbank refresh the collector's own
        summary state (the next tick publishes it), and so does sreport when the window is the summary's. Chunks are
        deduplicated and clipped to each account's start; a cut chunk marks the rollup `truncated`, a refused range
        is an error (never "no jobs"). Without sreport the user's usage is summed from the rollup (with a note)."""
        plan = self._acct_plans.pop(window, None)
        if plan is None:
            return None
        caps, now = ctx.caps, ctx.now
        errors: list[HostError] = []
        if (r := results.get("slurm.acct_share")) is not None:
            self._ingest_share(r, now, errors)
        if (r := results.get("slurm.acct_sbank")) is not None:
            self._ingest_sbank_result(r, now, errors)

        no_sreport = bool(caps.tools) and "sreport" not in caps.tools
        sreport_ok = bool(plan.sreport)
        mine: list[AccountUse] = []
        totals: list[AccountUse] = []
        for sec, group in plan.sreport.items():
            r = results.get(sec)
            if r is None or r.outcome is SectionOutcome.MISSING:
                no_sreport = no_sreport or r is not None
                sreport_ok = False
                continue
            if not r.ok:
                if (err := acct.account_error(r, now)) is not None:
                    errors.append(err)
                sreport_ok = False
                continue
            sp = acct.parse_sreport(r.stdout)
            if sp.mostly_bad:
                errors.append(parse_error(sec, sp, now, "sreport"))
                sreport_ok = False
                continue
            starts = {a: plan.starts[a] for a in group if a in plan.starts}
            mine += self._sreport_usage(sp.items, ctx, starts=starts, inferred=plan.inferred)
            _mine, total = acct.sreport_usage(sp.items, cluster=caps.cluster_name, starts=starts)
            got = {u.account: u for u in total if u.account in starts}
            for name, since in starts.items():         # no account row: nobody used it (0, as for "me")
                got.setdefault(name, AccountUse(account=name, scope="account", cpu_h=0.0, gpu_h=0.0, billing_h=0.0,
                                                start=since))
            totals += [dataclasses.replace(got[a], start_inferred=a in plan.inferred) for a in sorted(got)]

        truncated = plan.clipped
        chunks: list[list[acct.AcctJob]] = []
        for sec in plan.chunks:
            r = results.get(sec)
            if r is None or r.outcome is SectionOutcome.NOT_RUN:
                truncated = True                       # the probe stopped before this chunk: its jobs are missing
                continue
            cut = acct.truncated(r)
            truncated = truncated or cut
            if not r.ok and (err := acct.account_error(r, now)) is not None:
                errors.append(err)
            if not r.stdout or not (r.ok or cut or usable(r)):
                truncated = truncated or not r.ok      # refused / failed chunk: the rollup is a lower bound
                continue
            pr = acct.parse_acct_jobs(r.stdout, tz=caps.tz_offset, sep=plan.sep)
            truncated = truncated or bool(pr.bad_lines)    # rows lost to parsing: the rollup is a lower bound
            if pr.mostly_bad:
                errors.append(parse_error(sec, pr, now, "sacct"))
                continue
            chunks.append(pr.items)
        # one start for every account (so jobs of an account outside the associations count too), or per account
        start: datetime | Mapping[str, datetime] = (plan.starts if plan.window == "period"
                                                    else min(plan.starts.values()))
        cells, seen = acct.cells(acct.merge_jobs(chunks), caps, now=now, start=start)

        notes: tuple[str, ...] = ()
        usage: tuple[AccountUse, ...] = ()
        usage_at: datetime | None = None
        if sreport_ok:
            usage, usage_at = tuple(sorted(mine, key=lambda u: u.account)), now
            if plan.window == self._summary_window(ctx):
                self._acct_usage, self._acct_usage_at, self._acct_no_sreport = usage, now, False
                self._acct_usage_window = plan.window
                self._acct_errors.pop("slurm.acct_sreport", None)
        elif no_sreport and not truncated:             # sums (and zeros) only from a complete set of chunks
            summed = {u.account: u for u in acct.usage_from_cells(cells, plan.starts)}
            usage = tuple(dataclasses.replace(
                summed.get(a) or AccountUse(account=a, scope="me", cpu_h=0.0, gpu_h=0.0, billing_h=0.0,
                                            source="sacct", start=s), start_inferred=a in plan.inferred)
                for a, s in sorted(plan.starts.items()))
            usage_at, notes = now, (NO_SREPORT_REPORT,)
        self._acct_fresh = True
        base = self._account_summary(ctx)
        summary = dataclasses.replace(
            base, window=plan.window, usage=usage, usage_at=usage_at, notes=notes,
            errors=tuple(e for e in errors if not (e.section or "").startswith("slurm.acct_jobs")))
        return AccountReport(
            host=self.host.name, window=plan.window, fetched_at=now, summary=summary,
            window_start=min(plan.starts.values()) if plan.starts else None,
            account_totals=tuple(sorted(totals, key=lambda u: u.account)), cells=cells, jobs_seen=seen,
            truncated=truncated, errors=tuple(errors))

    @staticmethod
    def _summary_window(ctx: HostContext) -> UsageWindow:
        """The periodic summary's window: `general.usage_window`, "period" read as 30d."""
        window = ctx.general.usage_window
        return "30d" if window == "period" else window

    def _priority_of(self, job: Job, ctx: HostContext) -> PendingPriority | None:
        """Priority evidence of an own pending job (§5.2.10); None where the cluster's priority setup is unknown."""
        if job.state not in (JobState.PENDING, JobState.REQUEUED) or not job.is_mine or self._prio_unsupported:
            return None
        st = prio.settings(ctx.caps.slurm_config)
        if st.type is None:
            return None
        if not st.ranked:
            return prio.unranked(st)
        p = prio.evidence(job, prio.rows_for(job, self._sprio), self._shares, st, sampled_at=self._sprio_at)
        if ctx.caps.tools and "sprio" not in ctx.caps.tools:
            p = dataclasses.replace(p, hints=(prio.NO_SPRIO_HINT, *p.hints)[:prio.MAX_HINTS])
        return p

    def _nodes_now(self, ctx: HostContext) -> tuple[Node, ...]:
        """The last sinfo nodes, with every user's jobs when they were read lately (§5.2.9)."""
        nodes = self._nodes or ()
        at = self._occupants_at
        if at is None or ctx.now - at > NODE_JOBS_TTL:
            return nodes
        with_jobs = nodes_with_jobs(nodes, self._occupants, self._real_names, ctx.caps.remote_user)
        return tuple(dataclasses.replace(n, jobs_at=at) for n in with_jobs)

    def _maybe_fallback(self, kind: str, r: SectionResult) -> None:
        """A cluster that rejects the unit separator falls back to `|` (then squeue to the legacy -o format)."""
        if (r.outcome is not SectionOutcome.FAILED or r.stdout or not _BAD_FORMAT.search(r.stderr)
                or _CTLD_DOWN.search(r.stderr) or _DBD_DOWN.search(r.stderr)):
            return
        if self._sep[kind] == cmds.SEP:
            log.warning("%s: %s rejected the unit-separator format (%s); falling back to '|'", self.host.name, kind,
                        r.stderr.strip()[:200])
            self._sep[kind] = cmds.FALLBACK_SEP
        elif kind == "squeue" and not self._squeue_legacy:
            log.warning("%s: squeue -O failed (%s); falling back to -o", self.host.name, r.stderr.strip()[:200])
            self._squeue_legacy = True

    # ------------------------------------------------------------ merge + enrichment (§5.2.4)
    def _merge(self, ctx: HostContext) -> dict[str, Job]:
        now = ctx.now
        window_start = now - timedelta(seconds=ctx.host.slurm.sacct_window)
        for d in [d for d, j in self._history.items() if j.end_time is not None and j.end_time < window_start]:
            del self._history[d]
            self._meta.pop(d, None)
        jobs: dict[str, Job] = {d: self._enrich(j, ctx) for d, j in self._history.items() if j.state.is_terminal}
        for d, job in self._live.items():
            jobs[d] = self._enrich(job, ctx)
        horizon = timedelta(seconds=ctx.host.slurm.sacct_window)
        for d, t in list(self._vanished_at.items()):
            last = self._vanished_last.get(d)
            hist = jobs.get(d)
            summary = last is not None and _is_array_summary(last)
            if (hist is not None and hist.state.is_terminal and d not in self._live) or last is None \
                    or now - t > horizon or (summary and (now - t >= VANISHED_GRACE or _summary_resolved(last, jobs))):
                # a summary (`500_[1-2]`) is no job: accounting answers with its tasks, never with its id
                self._vanished_at.pop(d, None)
                self._vanished_last.pop(d, None)
                continue
            if now - t < VANISHED_GRACE:
                jobs[d] = dataclasses.replace(last, source="vanished", hints=("ended; waiting for accounting",),
                                              priority=None)
            else:
                jobs[d] = dataclasses.replace(last, state=JobState.UNKNOWN, source="vanished", priority=None,
                                              hints=("ended; final state unknown (accounting has no record)",))
        self._forget_unreferenced()
        return jobs

    def _forget_unreferenced(self) -> None:
        """Bound the per-job state to jobs still known (live, history window, vanished)."""
        live_ids = set(self._live)
        known = live_ids | set(self._history) | set(self._vanished_at)
        self._gpu_idle.forget(live_ids)
        for d in [d for d in self._jobinfo if d not in live_ids and d not in self._history]:
            self._jobinfo.pop(d, None)
            self._jobinfo_at.pop(d, None)
        for d in [d for d in self._meta if d not in known]:
            del self._meta[d]
        for d in [d for d in self._smi_model if d not in known]:
            del self._smi_model[d]
        self._jobinfo_gone &= live_ids
        for d in [d for d in self._jobinfo_fail if d not in live_ids]:
            del self._jobinfo_fail[d]
        for d in [d for d in self._vanished_tries if d not in self._vanished_at]:
            del self._vanished_tries[d]
        live_raw = {j.job_id_raw for j in self._live.values()}
        for raw in [r for r in self._usage if r not in live_raw]:
            del self._usage[raw]
        for raw in [r for r in self._sstat_rows if r not in live_raw]:
            del self._sstat_rows[raw]
        for raw in [r for r in self._sstat_last if r not in live_raw]:
            del self._sstat_last[raw]
        live_arrays = {j.array_job_id for j in self._live.values() if j.array_job_id}
        for raw in [r for r in self._sprio if r not in live_raw and r not in live_arrays]:
            del self._sprio[raw]
        paths = {r.path for j in (*self._live.values(), *self._history.values()) for r in self._logs_for(j) if r.path}
        for p in [p for p in self._stats if p not in paths]:
            del self._stats[p]

    def _resolved_alloc(self, job: Job) -> tuple[int, str | None, GpuSource | None]:
        """GPU count/type/source (§5.2.3); the model nvidia-smi showed during Attach wins over every gres name (lyra's
        `rtx6000` is an RTX PRO 6000 Blackwell, Attach field probes)."""
        gpus, gtype, source = self._resolved_gres(job)
        smi = self._smi_model.get(job.key.id)
        if gpus and smi is not None:
            return gpus, smi[0], "nvidia-smi"
        return gpus, gtype, source

    def _resolved_gres(self, job: Job) -> tuple[int, str | None, GpuSource | None]:
        info = self._jobinfo.get(job.key.id)
        meta = self._meta.get(job.key.id)
        raw = job.raw
        node_types = [self._node_types[n] for n in job.nodes if n in self._node_types]
        if len(node_types) != len(job.nodes):
            node_types = []
        if job.source == "sacct" or "AllocTRES" in raw:
            return resolve_gpu(parse_tres(raw.get("AllocTRES")), (None, None), (None, None), job.alloc.nodes,
                               req_tres=parse_tres(raw.get("ReqTRES")),
                               scontrol_type=info.gpu_type if info else None, node_types=node_types,
                               cached_type=meta.gpu_type if meta else None)
        return resolve_gpu(parse_tres(raw.get("tres-alloc")), parse_tres_per(raw.get("tres-per-node")),
                           parse_tres_per(raw.get("tres-per-job")), parse_int(raw.get("NumNodes")),
                           scontrol_type=info.gpu_type if info else None, node_types=node_types,
                           cached_type=meta.gpu_type if meta else None)

    @staticmethod
    def _names(job: Job) -> JobNames:
        return JobNames(job_id_raw=job.job_id_raw, array_job_id=job.array_job_id, array_task=job.array_task,
                        user=job.user or None, name=job.name or None, first_node=job.nodes[0] if job.nodes else None,
                        workdir=job.workdir, restarts=job.restarts if job.source == "squeue" else None)

    def _logs_for(self, job: Job) -> tuple[LogRef, ...]:
        """Best known log refs: scontrol (expanded) → squeue pattern (live) → jobmeta → sacct StdOut → templates."""
        info = self._jobinfo.get(job.key.id)
        if info is not None and info.stdout:
            return refs_from_paths(info.stdout, info.stderr, "scontrol")
        if job.source != "sacct" and any(r.path for r in job.logs):
            return job.logs
        meta = self._meta.get(job.key.id)
        if meta is not None and meta.stdout:
            return refs_from_paths(meta.stdout, meta.stderr, "cache")
        if job.logs and any(r.path for r in job.logs):
            return job.logs
        if job.source == "sacct":
            return refs_from_templates(self.host.slurm.log_templates, self._names(job))
        return job.logs

    def _enrich(self, job: Job, ctx: HostContext) -> Job:
        d = job.key.id
        info = self._jobinfo.get(d)
        meta = self._meta.get(d)
        gpus, gtype, source = self._resolved_alloc(job)
        per_node = info.per_node if info and info.per_node else (meta.per_node if meta else job.alloc.per_node)
        alloc = dataclasses.replace(job.alloc, gpus=gpus, gpu_type=gtype, per_node=per_node, gpu_type_source=source)
        workdir = job.workdir or (info.workdir if info else None) or (meta.workdir if meta else None)
        logs = tuple(with_stat(ref, self._stats) for ref in self._logs_for(job))
        usage = self._usage.get(job.job_id_raw) if job.state is JobState.RUNNING else None
        raw = dict(job.raw)
        hints: list[str] = []
        if job.state.group is StateGroup.QUEUED:
            text = humanize_reason(job.reason)
            if text:                                   # "will never run: dependency failed" included
                raw["reason_text"] = text
                hints.append(text)
            advice = reason_advice(job.reason)
            if advice:                                 # what to do about it (detail cause line, preview; §5.2.8)
                raw["reason_advice"] = advice
        elif job.state is JobState.RUNNING:
            stdout = next((r for r in logs if r.stream == "stdout"), None)
            idle = log_idle_hint(stdout.mtime if stdout else None, ctx.now, ctx.general.stall_after)
            if idle:
                hints.append(idle)
            gpu_idle = self._gpu_idle.hint(d, ctx.now)
            if gpu_idle:
                hints.append(gpu_idle)
        elif job.state.is_terminal:
            summary = short_summary(job, uid=ctx.caps.uid, slurm_config=ctx.caps.slurm_config)
            if summary:
                raw["reason_text"] = summary
                hints.append(summary)
        cpu_pct, rss = job.cpu_pct, job.rss_bytes
        if usage is not None:            # sstat AveCPU/MaxRSS are used regardless of gpu_tres (§5.1)
            cpu_pct = _cpu_pct(usage, job.elapsed)
            rss = usage.max_rss_bytes
        enriched = dataclasses.replace(job, alloc=alloc, workdir=workdir, logs=logs,
                                       gpu=usage.gpu if usage else (job.gpu if job.source == "vanished" else None),
                                       hints=tuple(hints), raw=raw, cpu_pct=cpu_pct, rss_bytes=rss,
                                       priority=self._priority_of(job, ctx) if job.source == "squeue" else None)
        if job.source == "squeue":
            self._remember(enriched)
        return enriched

    def _remember(self, job: Job) -> None:
        stdout = next((r for r in job.logs if r.stream == "stdout" and r.path), None)
        stderr = next((r for r in job.logs if r.stream == "stderr" and r.path), None)
        if stdout is None and job.alloc.gpu_type is None and not job.alloc.per_node:
            return
        self._meta[job.key.id] = JobMeta(
            workdir=job.workdir, stdout=stdout.path if stdout else None,
            stderr=None if stdout is None or stderr is None or stderr.merged else stderr.path,
            gpu_type=job.alloc.gpu_type, per_node=job.alloc.per_node)

    # ------------------------------------------------------------ details
    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]:
        if job.kind is not JobKind.SLURM:
            return []
        out: list[Section] = []
        d = job.key.id
        at = self._jobinfo_at.get(d)
        in_controller = not job.state.is_terminal or (
            job.end_time is not None and (ctx.now - job.end_time).total_seconds() < min_job_age_s(
                ctx.caps.slurm_config))
        if in_controller and (at is None or ctx.now - at > DETAIL_JOBINFO_MAX_AGE):
            out.append(Section("slurm.scontrol", cmds.scontrol_cmd([job.job_id_raw]), timeout=15,
                               requires=("scontrol",), meta={"ids_raw": [job.job_id_raw]}))
        sep = self._sep["sacct"]
        fields = cmds.step_fields(ctx.caps.sacct_fields)     # + TotalCPU where known (efficiency card, on demand)
        out.append(Section("slurm.steps", cmds.sacct_steps_cmd(job.job_id_raw, sep, fields), timeout=30,
                           requires=("sacct",), meta={"job_id_raw": job.job_id_raw, "sep": sep,
                                                      "fields": fields.split(",")}))
        if job.state is JobState.RUNNING:
            out.append(Section("slurm.sstat", cmds.sstat_cmd([job.job_id_raw]), timeout=15, requires=("sstat",),
                               meta={"ids_raw": [job.job_id_raw]}))
        paths = [p for p in dict.fromkeys(r.path for r in self._logs_for(job) if r.path) if p and _valid_path(p)]
        if paths:
            out.append(Section("slurm.logstat", cmds.logstat_cmd(paths), timeout=10, requires=("stat",),
                               meta={"paths": paths}))
        return out

    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail:
        now = ctx.now
        errors: list[HostError] = []
        d = job.key.id
        if (r := results.get("slurm.scontrol")) is not None:
            if not r.ok:
                errors.append(_err(r, now))
            if usable(r):
                for block in parse_scontrol(r.stdout).values():
                    rec = block.record_for(job.array_task)
                    if rec is not None:
                        self._store_jobinfo(d, rec, now)
                    elif block.left_controller:
                        self._jobinfo_gone.add(d)
        steps: tuple[JobStep, ...] = ()
        if (r := results.get("slurm.steps")) is not None:
            if not r.ok:
                errors.append(_err(r, now))
            if usable(r):
                parsed = parse_steps(r.stdout, cmds.step_fields(ctx.caps.sacct_fields).split(","),
                                     tz=ctx.caps.tz_offset, sep=self._sep["sacct"])
                if parsed.mostly_bad:
                    errors.append(parse_error(r.name, parsed, now, "sacct steps"))
                steps = tuple(parsed.items)
        if (r := results.get("slurm.sstat")) is not None:
            self._ingest_sstat(r, ctx, errors)
        steps = self._steps_with_sstat(job, steps)
        if (r := results.get("slurm.logstat")) is not None:
            if usable(r):
                self._stats.update(parse_stat_lines(r.stdout))
            if not r.ok:
                errors.append(_err(r, now))
        enriched = self._enrich(job, ctx)
        if job.state is JobState.RUNNING and enriched.gpu is None and job.alloc.gpus:
            enriched = dataclasses.replace(enriched, gpu=job.gpu)
        info = self._jobinfo.get(d)
        fields: Mapping[str, str] = dict(info.fields) if info else dict(job.raw)
        explanation = explain(enriched, steps, uid=ctx.caps.uid, slurm_config=ctx.caps.slurm_config)
        if job.state.group is StateGroup.QUEUED and enriched.hints:
            explanation = (*enriched.hints, *explanation)
        return JobDetail(job=enriched, fields=fields, per_node=enriched.alloc.per_node, steps=steps,
                         logs=enriched.logs, gpu=enriched.gpu, procs=(), procs_note=_PROCS_NOTE, env={},
                         explanation=tuple(explanation), fetched_at=now, errors=tuple(errors),
                         efficiency=efficiency_lines(efficiency(enriched, steps)))

    def _steps_with_sstat(self, job: Job, steps: tuple[JobStep, ...]) -> tuple[JobStep, ...]:
        """sacct leaves AveCPU/MaxRSS/TRESUsageIn* of running steps empty; fill them from the job's last sstat rows
        (the Steps tab's MAXRSS and "GPU (max per task)" columns, the cores estimate)."""
        rows = {r.step: r for r in self._sstat_rows.get(job.job_id_raw, [])}
        if not rows or job.state is not JobState.RUNNING:
            return steps
        out = []
        for st in steps:
            row = rows.get(st.step_id.rpartition(".")[2]) if "." in st.step_id else None
            if row is not None and not st.state.is_terminal:
                st = dataclasses.replace(st, max_rss_bytes=st.max_rss_bytes if st.max_rss_bytes is not None
                                         else row.max_rss_bytes, ave_cpu=st.ave_cpu or row.ave_cpu,
                                         ntasks=st.ntasks if st.ntasks is not None else row.ntasks,
                                         tres_usage_max=st.tres_usage_max or row.tres_max,     # GPU (max per task)
                                         tres_usage_tot=st.tres_usage_tot or row.tres_tot)
            out.append(st)
        return tuple(out)

    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None:
        """A running job: a login shell on its node, inside the job (`srun --overlap --pty`: the job's cgroup and
        GPUs; works where sshd refuses compute-node logins, e.g. lyra). The step ends with the shell; it leaves a
        sacct step row. Otherwise (or for a node-less row): its WorkDir on the login node."""
        if job is None:
            return None
        target = _shell_node(job, node)
        if target is None:
            if not job.workdir or not _valid_path(job.workdir):
                return None
            return f'cd {shlex.quote(job.workdir)} && exec "$SHELL" -l'
        cd = f"cd {shlex.quote(job.workdir)} 2>/dev/null; " if job.workdir and _valid_path(job.workdir) else ""
        overlap = " --overlap" if _has_overlap(ctx.caps.slurm_version) else ""
        hello = f"runfleet: shell on {target} inside job {job.key.id} (exit leaves the job running)"
        script = (f"{cd}printf '%s\\n' {shlex.quote(hello)}; "
                  f'srun --jobid={job.job_id_raw}{overlap} --pty -N1 -n1 -w {target} "${{SHELL:-sh}}" -l; rc=$?; '
                  f'[ "$rc" -eq 0 ] || [ "$rc" -eq 130 ] || '
                  f"{{ printf 'runfleet: srun exited with %s - press Enter ' \"$rc\"; read _rf; }}")
        return f"exec sh -c {shlex.quote(script)}"      # the login shell may be csh/fish: run the script in sh

    # ------------------------------------------------------------ attach (§7.16)
    def attach_modes(self, job: Job, ctx: HostContext) -> tuple[AttachMode, ...]:
        return attach.modes(job, ctx.caps, overlap=_has_overlap(ctx.caps.slurm_version))

    def plan_attach(self, job: Job, mode: AttachMode, ctx: HostContext, *,
                    node: str | None = None) -> AttachPlan:
        nodes = attach.target_nodes(job, mode, node)
        if mode == "srun" and not _has_overlap(ctx.caps.slurm_version):
            raise AttachBlocked(f"srun --overlap needs Slurm 20.11 or newer ({ctx.host.name} runs "
                                f"{ctx.caps.slurm_version})")
        return attach.build_plan(job, mode, nodes, {n: self._attach_jobs(job, n, ctx) for n in nodes})

    def _attach_jobs(self, job: Job, node: str, ctx: HostContext) -> list[tuple[str, str]]:
        """(raw id, SLUID or "") of the jobs node.cg / node.pids read on `node`: the target, then the user's other
        running jobs there (newest first; k2 lets a session read every own job's cgroup on the node)."""
        out = [(job.job_id_raw, self._sluid(job))]
        for other in self._own_running(ctx):
            if node in other.nodes and other.job_id_raw.isdigit() and other.job_id_raw not in {j for j, _ in out}:
                out.append((other.job_id_raw, self._sluid(other)))
        return out[:attach.MAX_NODE_JOBS]

    def _sluid(self, job: Job) -> str:
        info = self._jobinfo.get(job.key.id)
        return (info.fields.get("SLUID") or "") if info is not None else ""

    def _key_of_raw(self, raw: str) -> JobKey | None:
        for j in self._live.values():
            if j.job_id_raw == raw:
                return j.key
        return JobKey(self.host.name, raw) if raw.isdigit() else None

    def _known_idx(self, raw: str, node: str) -> tuple[int, ...]:
        """scontrol -d GPU indices of job `raw` on `node` (() unknown): cross-checks the ssh minor → IDX mapping."""
        job = next((j for j in self._live.values() if j.job_id_raw == raw), None)
        if job is None:
            return ()
        info = self._jobinfo.get(job.key.id)
        per_node = info.per_node if info is not None and info.per_node else job.alloc.per_node
        return next((na.gpu_indices for na in per_node if na.node == node), ())

    def ingest_attach(self, plan: AttachPlan, results: Mapping[str, SectionResult], ctx: HostContext, *,
                      previous: JobLive | None = None) -> tuple[JobLive, Mapping[str, object]]:
        job = self._live.get(plan.key.id)
        ax = attach.IngestContext(host=ctx.host.name, user=ctx.caps.remote_user, now=ctx.now,
                                  job_key=self._key_of_raw, known_idx=self._known_idx,
                                  job_gpus=max(job.alloc.gpus, self._resolved_gres(job)[0]) if job else None)
        res = attach.ingest(plan, results, ax, previous=previous)
        if res.gpu_model is not None and res.gpu_model != self._smi_model.get(plan.key.id):
            log.info("%s: job %s runs on %s (nvidia-smi)", self.host.name, plan.key.id, res.gpu_model[0])
            self._smi_model[plan.key.id] = res.gpu_model
        return res.live, res.caps_patch

    # ------------------------------------------------------------ actions
    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
        return slurm_actions.specs(jobs, ctx)

    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan:
        return slurm_actions.plan(action_id, jobs, params, ctx)

    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]:
        return slurm_actions.interpret(plan, results, ctx)


# ---------------------------------------------------------------- helpers
def _cpu_pct(usage: StepUsage, elapsed: timedelta | None) -> float | None:
    """Lifetime CPU use of the job's main step in % of one core: AveCPU × NTasks / elapsed (sstat, §5.1)."""
    if usage.ave_cpu is None or elapsed is None or elapsed.total_seconds() <= 0:
        return None
    return usage.ave_cpu.total_seconds() * max(1, usage.row.ntasks) / elapsed.total_seconds() * 100


def _is_array_summary(job: Job) -> bool:
    """A pending array summary row (`500_[2-20%1]`): its id is a range, not one job; no sacct row ever matches it."""
    return job.array_task is not None and not job.array_task.isdigit()


def _summary_resolved(summary: Job, jobs: Mapping[str, Job]) -> bool:
    """A vanished summary is accounted for once a terminal row of one of its tasks shows up (any task of the array
    when Slurm truncated the list)."""
    tasks = expand_array_tasks(summary.array_task)
    return any(j.array_job_id == summary.array_job_id and j.state.is_terminal and j.array_task is not None
               and j.array_task.isdigit() and (tasks is None or int(j.array_task) in tasks) for j in jobs.values())


def _backoff(tries: int) -> timedelta:
    """30 s, 1 min, 2 min, … at most 10 min."""
    return timedelta(seconds=min(600, 30 * 2 ** (min(tries, 6) - 1)))


def _shell_node(job: Job, node: str | None) -> str | None:
    """The node a job shell attaches to: `node` if it is one of the job's, else the batch host / first node; None when
    the job is not running here (pending, finished, a multi-row summary) or its ids are not safe to put in a command."""
    if job.state is not JobState.RUNNING or not job.job_id_raw.isdigit():
        return None
    nodes = job.nodes or ((job.batch_host,) if job.batch_host else ())
    target = node if node is not None and node in nodes else (job.batch_host if job.batch_host in nodes else None)
    target = target or (nodes[0] if nodes else None)
    return target if target is not None and _NODE_RE.match(target) else None


def _has_overlap(version: str | None) -> bool:
    """`srun --overlap` exists since 20.11 (and is needed there to share the job's CPUs/GPUs); unknown = assume so."""
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return m is None or (int(m[1]), int(m[2])) >= (20, 11)


_NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _valid_path(p: str) -> bool:
    return p.startswith("/") and "\x00" not in p and "\n" not in p


def _err(r: SectionResult, at: datetime) -> HostError:
    """Section failure with the Slurm-specific one-liners of §9 rows 10-11."""
    summary: str | None = None
    if r.outcome in (SectionOutcome.FAILED, SectionOutcome.STDERR_ONLY):
        if r.name in ("slurm.squeue", "slurm.scontrol", "slurm.sinfo") and _CTLD_DOWN.search(r.stderr):
            summary = "slurmctld unreachable"
        elif r.name.startswith("slurm.sacct") and _DBD_DOWN.search(r.stderr):
            summary = "slurmdbd unreachable"
    err = section_error(r, at, summary=summary)
    assert err is not None
    return err


def _names(value: str | None) -> set[str] | None:
    """A partition's Allow*/Deny* list; None = no restriction (absent, empty or ALL)."""
    if not value or value == "ALL":
        return None
    return {v for v in value.split(",") if v}


_FIELD = re.compile(r"(?:^|\s)([A-Za-z]+)=")


def _fields(line: str) -> dict[str, str]:
    """`scontrol show partition -o` fields; a value runs to the next ` Key=`, so names with spaces stay whole."""
    marks = list(_FIELD.finditer(line))
    return {m.group(1): line[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(line)].strip()
            for i, m in enumerate(marks)}


def gpu_partitions(text: str, associations: Sequence[tuple[str, str, str, str]],
                   cluster: str | None = None) -> tuple[str, ...]:
    """GPU partitions (TRES gres/gpu > 0) the user may submit to, from `slurm.parts` and the user's associations (of
    `cluster` when known): not hidden, root-only, inactive or drained; a group in AllowGroups; and an account whose
    association for this partition (else its association for all partitions) passes AllowAccounts/DenyAccounts, the
    account or a parent account counting, with a QOS that passes AllowQos/DenyQos. A DenyAccounts list needs the
    complete parent chain. Without association data (no sacctmgr) only partitions open to every account and QOS
    count; with associations on other clusters only, none."""
    groups: set[str] | None = None
    parent: dict[str, str] = {}
    complete = False
    parts: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("groups="):
            groups = {g for g in line[len("groups="):].split(",") if g}
        elif line.startswith("parent="):
            c, _, rest = line[len("parent="):].partition("|")
            acct, _, up = rest.partition("|")
            if acct and up and (cluster is None or c == cluster):
                parent.setdefault(acct, up)
        elif line == "parents=complete":
            complete = True
        elif line.startswith("PartitionName="):
            parts.append(_fields(line))
    mine = [a for a in associations if cluster is None or not a[0] or a[0] == cluster]

    def lineage(account: str) -> set[str]:
        out = {account}
        while (account := parent.get(account, "")) and account not in out:
            out.add(account)
        return out

    found: list[str] = []
    for f in parts:
        name = f.get("PartitionName", "")
        if (not name or not cmds.NAME_RE.match(name) or f.get("Hidden") == "YES" or f.get("RootOnly") == "YES"
                or f.get("State") in ("INACTIVE", "DRAIN")):
            continue
        gpus = re.search(r"(?:^|,)gres/gpu=(\d+)", f.get("TRES", ""))
        if not gpus or int(gpus.group(1)) == 0:
            continue
        allow_groups = _names(f.get("AllowGroups"))
        if allow_groups is not None and not (groups and groups & allow_groups):
            continue
        allow_acc, deny_acc = _names(f.get("AllowAccounts")), _names(f.get("DenyAccounts")) or set()
        allow_qos, deny_qos = _names(f.get("AllowQos")), _names(f.get("DenyQos")) or set()
        if not associations:
            ok = allow_acc is None and not deny_acc and allow_qos is None and not deny_qos
        else:
            ok = False
            for _cluster, account, part, qos in _effective(mine, name):
                chain = lineage(account)
                if (allow_acc is not None and not chain & allow_acc) or (deny_acc and (not complete or chain & deny_acc)):
                    continue
                qs = {q for q in qos.split(",") if q}
                if (not qs and allow_qos is None and not deny_qos) or any(
                        (allow_qos is None or q in allow_qos) and q not in deny_qos for q in qs):
                    ok = True
                    break
        if ok:
            found.append(name)
    return tuple(sorted(set(found)))


def partition_charges(text: str) -> tuple[PartitionCharge, ...]:
    """How each partition charges (Account, §5.2.11), from the `PartitionName=` lines of `slurm.parts`: every line
    whose name passes the validator (hidden ones and those closed to the user's groups are not printed without `-a`).
    TRESBillingWeights stays raw (keys compare case-insensitively later); `QoS=N/A` and `AllowQos=ALL` -> None."""
    out: list[PartitionCharge] = []
    for line in text.splitlines():
        if not line.startswith("PartitionName="):
            continue
        f = _fields(line)
        name = f.get("PartitionName", "")
        if not name or not cmds.NAME_RE.match(name):
            continue
        weights = next((v for k, v in f.items() if k.lower() == "tresbillingweights"), "")
        allow = f.get("AllowQos", "")
        out.append(PartitionCharge(
            partition=name, billing_weights=nullish(weights) or "", qos=nullish(f.get("QoS")),
            allow_qos=None if not allow or allow == "ALL" else tuple(q for q in allow.split(",") if q),
            def_cpu_per_gpu=parse_int(f.get("DefCpuPerGPU")),            # UNLIMITED / absent -> None
            def_mem_per_cpu_mb=parse_int(f.get("DefMemPerCPU"))))
    return tuple(out)


def parse_qos(text: str) -> ParseResult[tuple[str, float]]:
    """`slurm.qos` (Account, §5.2.11): `name|UsageFactor` rows. A row without a valid name or a finite factor >= 0 is
    bad; the caller drops the whole field when most rows are (`mostly_bad`)."""
    pr: ParseResult[tuple[str, float]] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        name, _, rest = line.strip().partition("|")
        factor = parse_float(rest.split("|", 1)[0])
        if not cmds.NAME_RE.match(name) or factor is None or factor < 0:
            pr.bad(n, line, "not a QOS name and usage factor")
            continue
        pr.items.append((name, factor))
    return pr


def _effective(assocs: Sequence[tuple[str, str, str, str]], partition: str) -> list[tuple[str, str, str, str]]:
    """Per account, Slurm uses the association for this partition when there is one, else the one for all."""
    specific = [a for a in assocs if a[2] and partition in a[2].split(",")]
    have = {a[1] for a in specific}
    return specific + [a for a in assocs if not a[2] and a[1] not in have]


def parse_capabilities(results: Mapping[str, SectionResult]) -> dict[str, object]:
    """Capability sections (§5.1) → Capabilities fields."""
    out: dict[str, object] = {}
    r = results.get("slurm.version")
    if r is not None and usable(r):
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", r.stdout)
        if m:
            out["slurm_version"] = m.group(1)
    r = results.get("slurm.config")
    if r is not None and usable(r):
        cfg: dict[str, str] = {}
        for line in r.stdout.splitlines():
            key, eq, value = line.partition("=")
            if eq and key.strip():
                cfg[key.strip()] = value.strip()
        out["slurm_config"] = cfg
        if cfg.get("ClusterName"):
            out["cluster_name"] = cfg["ClusterName"]
        freq = seconds_prefix(cfg.get("JobAcctGatherFrequency"))
        if freq is not None:
            out["acct_gather_freq_s"] = freq
    r = results.get("slurm.assoc")
    if r is not None and usable(r):
        assoc: list[tuple[str, str, str, str]] = []
        for line in r.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[0]:
                assoc.append((parts[0], parts[1], parts[2], parts[3]))
        out["associations"] = tuple(assoc)
    r = results.get("slurm.parts")
    if r is not None and usable(r):
        assoc_now, cluster = out.get("associations"), out.get("cluster_name")
        out["gpu_partitions"] = gpu_partitions(r.stdout, assoc_now if isinstance(assoc_now, tuple) else (),
                                               cluster if isinstance(cluster, str) else None)
        out["partition_charges"] = partition_charges(r.stdout)
    r = results.get("slurm.qos")
    if r is not None and usable(r):
        qos = parse_qos(r.stdout)
        if qos.mostly_bad:              # a PARSE error: the field stays unset (cost shows `×1?`), the detection goes on
            lineno, line, why = qos.bad_lines[0]
            log.warning("slurm.qos: could not parse the QOS list (line %d: %s: %.200s)", lineno, why, line)
        else:
            out["qos_usage_factor"] = dict(qos.items)
    r = results.get("slurm.sacctfmt")
    if r is not None and usable(r):
        out["sacct_fields"] = frozenset(r.stdout.split())
    r = results.get("slurm.gputres")
    if r is not None and usable(r):
        seed = gputres_seed(r.stdout)
        if seed is not None:
            out["gpu_tres"] = seed
    return out
