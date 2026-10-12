"""[WP2] ProcessCollector: workstation task sections, cadences and ingest (§5.3).

Tasks change only on a successful `process.ps` result; on every tick (also those without `process.ps`) `ingest`
re-annotates the monitor's latest GPUs with the owning tasks and returns `gpus`/`procs`, so annotations never blink
(§3.6 merge rule).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shlex
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from runfleet.backends.base import (ActionPlan, ActionSpec, Cadence, CollectorOutput, HostContext, MonitorView,
                                    SchedulerCollector)
from runfleet.backends.common import (GpuIdleTracker, StatInfo, complete, log_idle_hint, parse_error,
                                      parse_stat_lines, section_error, usable, with_stat)
from runfleet.backends.process import actions as task_actions
from runfleet.backends.process.parse_ps import OWN_COLUMNS, FdInfo, parse_fds, parse_ps, parse_ps_rows
from runfleet.backends.process.tasks import (NoiseFilter, TaskRecord, classify_exit, group_tasks, other_user_jobs,
                                             task_job)
from runfleet.config import HostConfig
from runfleet.models import (GPU, ActionResult, Capabilities, HostError, Job, JobDetail, JobKey, JobKind, JobState,
                             Proc, StateGroup)
from runfleet.probe.section import Section, SectionResult
from runfleet.util.logscan import PENDING_EXIT_HINT
from runfleet.util.slurmfmt import split_rows

log = logging.getLogger("runfleet.backends.process.collector")

PS_CMD = 'date +%s; ps -ww -u "$(id -un)" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args='
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "OMP_NUM_THREADS", "WANDB_PROJECT",
            "WANDB_RUN_ID")
EXITSCAN_MARK = "@@rftask "
EXITSCAN_BYTES = 8192
MAX_LOGSTAT_PATHS = 30
START_TOLERANCE_S = 5


FDSTAT_MARK = "@@stat "
FDS_REFRESH = timedelta(minutes=5)              # a task root's fds are re-read this often (`exec >train.log` later)


def fds_cmd(pids: Sequence[int]) -> str:
    """`pid␟cwd␟fd1␟fd2` per pid (␟ = \\x1f: paths may contain `|`), each followed by `@@stat <path>|<size>|<mtime>`
    for its regular-file stdout/stderr: the log paths are only known from this section, so stat'ing them here gives
    `snapshot` (two rounds) log sizes without a third round of process.logstat (M3 F9 residual)."""
    plist = " ".join(str(int(p)) for p in pids)
    return (f"for p in {plist}; do c=$(readlink /proc/$p/cwd 2>/dev/null); o=$(readlink /proc/$p/fd/1 2>/dev/null); "
            "e=$(readlink /proc/$p/fd/2 2>/dev/null); printf '%s\\037%s\\037%s\\037%s\\n' \"$p\" \"$c\" \"$o\" \"$e\"; "
            "for t in \"$o\" \"$e\"; do case \"$t\" in /dev/*|*' (deleted)') ;; "
            f"/*) stat -L -c '{FDSTAT_MARK}%n|%s|%Y' -- \"$t\" 2>/dev/null ;; esac; done; done; true")


def parse_fd_stats(text: str) -> dict[str, StatInfo]:
    """The `@@stat` lines of process.fds (same `%n|%s|%Y` format as process.logstat)."""
    return parse_stat_lines("\n".join(line[len(FDSTAT_MARK):] for line in text.splitlines()
                                      if line.startswith(FDSTAT_MARK)))


def logstat_cmd(paths: Sequence[str]) -> str:
    return "stat -L -c '%n|%s|%Y' -- " + " ".join(shlex.quote(p) for p in paths) + " 2>&1; true"


def exitscan_cmd(tasks: Sequence[tuple[str, Sequence[str]]]) -> str:
    parts = [f"printf '%s\\n' {shlex.quote(EXITSCAN_MARK + tid)}; tail -q -c {EXITSCAN_BYTES} -- "
             + " ".join(shlex.quote(p) for p in paths) + " 2>/dev/null; echo" for tid, paths in tasks]
    return "; ".join(parts) + "; true"


def detail_cmd(pids: Sequence[int], root: int | None) -> str:
    cmd = ("ps -ww -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args= -p "
           + ",".join(str(int(p)) for p in pids) + "; echo '@@env'")
    if root is not None:
        keys = "|".join(ENV_KEYS)
        cmd += f"; tr '\\0' '\\n' < /proc/{int(root)}/environ 2>/dev/null | grep -E '^({keys})='"
    return cmd + "; true"


def _valid_path(p: str) -> bool:
    return p.startswith("/") and "\x00" not in p and "\n" not in p


def parse_exitscan(text: str) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in split_rows(text):
        if line.startswith(EXITSCAN_MARK):
            current = line[len(EXITSCAN_MARK):].strip()
            out.setdefault(current, [])
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


class ProcessCollector(SchedulerCollector):
    name = "process"

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        extra: list[str] = []
        for pattern in host.workstation.ignore:
            try:
                re.compile(pattern)
                extra.append(pattern)
            except re.error as e:
                log.warning("%s: ignoring invalid workstation.ignore regex %r: %s", host.name, pattern, e)
        self._noise = NoiseFilter(extra)
        self._tasks: dict[str, TaskRecord] = {}
        self._jobs: dict[str, Job] = {}               # active own tasks as of the last process.ps
        self._ended: dict[str, Job] = {}
        self._procs: tuple[Proc, ...] = ()
        self._by_pid: dict[int, Proc] = {}
        self._epoch: int | None = None
        self._have_ps = False
        self._fds: dict[int, tuple[int, FdInfo]] = {}  # pid -> (start epoch, fds)
        self._fds_at: dict[int, datetime] = {}
        self._need_fds: dict[int, None] = {}
        self._fds_planned: tuple[int, ...] = ()
        self._exitscan: dict[str, list[str]] = {}
        self._exitscan_planned: tuple[str, ...] = ()   # task ids of the last planned process.exitscan
        self._stats: dict[str, StatInfo] = {}
        self._gpu_idle = GpuIdleTracker()
        self._other_starts: dict[int, int] = {}       # other users' GPU pids → start fixed at first sight

    # ------------------------------------------------------------ detection
    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        if host.backend == "workstation":
            return True
        return host.backend == "auto" and not {"squeue", "sinfo"} <= caps.tools

    # ------------------------------------------------------------ planning
    def cadences(self, ctx: HostContext) -> list[Cadence]:
        w = ctx.host.workstation
        return [
            Cadence("process.ps", max(w.poll_procs, 5.0), floor=5.0, focused_factor=0.5),
            Cadence("process.logstat", max(w.poll_logstat, 30.0), floor=30.0, focused_factor=0.5),
            Cadence("process.exitscan", 0.0, mode="event"),
        ]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        out: list[Section] = []
        if "process.ps" in due:
            out.append(Section("process.ps", PS_CMD, timeout=5, requires=("ps",)))
            pids = list(self._need_fds)[:64]
            self._fds_planned = tuple(pids)
            if pids:
                out.append(Section("process.fds", fds_cmd(pids), timeout=5, meta={"pids": pids}))
        if "process.logstat" in due:
            paths = list(dict.fromkeys(r.path for j in self._jobs.values() for r in j.logs
                                       if r.path and _valid_path(r.path)))[:MAX_LOGSTAT_PATHS]
            if paths:
                out.append(Section("process.logstat", logstat_cmd(paths), timeout=5, requires=("stat",),
                                   meta={"paths": paths}))
        if "process.exitscan" in due and self._exitscan:
            tasks = [(tid, paths) for tid, paths in self._exitscan.items()][:16]
            self._exitscan_planned = tuple(tid for tid, _ in tasks)
            out.append(Section("process.exitscan", exitscan_cmd(tasks), timeout=5, requires=("tail",),
                               meta={"tasks": [(t, list(p)) for t, p in tasks]}))
        return out

    # ------------------------------------------------------------ ingest
    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        now = ctx.now
        errors: list[HostError] = []
        followups: set[str] = set()
        if (r := results.get("process.fds")) is not None:
            if usable(r):
                self._stats.update(parse_fd_stats(r.stdout))
                parsed = parse_fds(r.stdout)
                if r.ok:             # a planned pid without a row had a malformed one (a newline in a path): not again
                    parsed.update({p: FdInfo(None, None, None) for p in self._fds_planned if p not in parsed})
                for pid, info in parsed.items():
                    proc = self._by_pid.get(pid)
                    if proc is not None and self._epoch is not None:
                        self._fds[pid] = (self._epoch - proc.etime_s, info)
                        self._fds_at[pid] = now
                    self._need_fds.pop(pid, None)
            if not r.ok:
                errors.append(_err(r, now))
        if (r := results.get("process.ps")) is not None:
            self._ingest_ps(r, ctx, monitors, errors, followups)
        if (r := results.get("process.exitscan")) is not None:
            self._ingest_exitscan(r, now, errors)
        if (r := results.get("process.logstat")) is not None:
            if usable(r):
                self._stats.update(parse_stat_lines(r.stdout))
            if not r.ok:
                errors.append(_err(r, now))
        if self._exitscan:
            followups.add("process.exitscan")
        if not self._have_ps:
            return CollectorOutput(errors=tuple(errors), followups=frozenset(followups))
        jobs, gpus, procs = self._assemble(ctx, monitors)
        return CollectorOutput(jobs=jobs, gpus=gpus, procs=procs, errors=tuple(errors), followups=frozenset(followups))

    def _gpu_pids(self, ctx: HostContext, monitors: MonitorView) -> frozenset[int]:
        """Own pids holding a GPU, from a monitor sample no older than 3 × poll_gpu (else none)."""
        age = monitors.gpus_age(ctx.now)
        if age is None or age > 3 * ctx.host.workstation.poll_gpu:
            return frozenset()
        return frozenset(p.pid for g in monitors.gpus for p in g.processes if p.pid in self._by_pid)

    def _ingest_ps(self, r: SectionResult, ctx: HostContext, monitors: MonitorView, errors: list[HostError],
                   followups: set[str]) -> None:
        now = ctx.now
        if not r.ok:
            errors.append(_err(r, now))
        if not complete(r):
            return                                         # a failed sample never ends tasks
        user = ctx.caps.remote_user
        epoch, pr = parse_ps(r.stdout, user=user)
        if pr.mostly_bad:
            errors.append(parse_error(r.name, pr, now, "ps"))
            return
        self._epoch = epoch if epoch is not None else int(now.timestamp() + ctx.caps.clock_skew.total_seconds())
        self._procs = tuple(pr.items)
        self._by_pid = {p.pid: p for p in self._procs}
        w = ctx.host.workstation
        gpu_pids = self._gpu_pids(ctx, monitors)
        grouping = group_tasks(self._procs, self._epoch, gpu_pids=gpu_pids, prev=self._tasks, host=ctx.host.name,
                               noise=self._noise, min_age=w.task_min_age, min_cpu=w.task_min_cpu)
        for rec in grouping.ended:
            self._end_task(rec, now, followups)
        self._tasks = grouping.active
        for pid in [p for p in self._fds if p not in self._by_pid or
                    abs(self._fds[p][0] - (self._epoch - self._by_pid[p].etime_s)) > START_TOLERANCE_S]:
            del self._fds[pid]
            self._fds_at.pop(pid, None)
        for rec in self._tasks.values():
            for pid in (rec.root_pid, *[p for p in rec.members[1:] if p in gpu_pids]):
                fetched = self._fds_at.get(pid)
                if pid not in self._fds or (pid == rec.root_pid and fetched is not None
                                            and now - fetched > FDS_REFRESH):
                    self._need_fds.setdefault(pid, None)
        for pid in [p for p in self._need_fds if p not in self._by_pid]:
            del self._need_fds[pid]
        if any(rec.root_pid in self._need_fds for rec in grouping.new):
            followups.add("process.ps")                    # fetch the new task's cwd/logs at the next tick
        self._have_ps = True

    def _end_task(self, rec: TaskRecord, now: datetime, followups: set[str]) -> None:
        last = self._jobs.get(rec.key.id)
        if last is None:
            return
        paths = [r.path for r in last.logs if r.path and _valid_path(r.path)]
        ended = dataclasses.replace(last, state=JobState.EXITED, raw_state="exited", end_time=now,
                                    hints=(PENDING_EXIT_HINT,) if paths else ("ended, exit status unknown",),
                                    gpu=None, cpu_pct=None, observed_at=now)
        self._ended[rec.key.id] = ended
        if paths:
            self._exitscan[rec.key.id] = list(dict.fromkeys(paths))
            followups.add("process.exitscan")

    def _ingest_exitscan(self, r: SectionResult, now: datetime, errors: list[HostError]) -> None:
        if not r.ok:
            errors.append(_err(r, now))
        if not usable(r):
            return
        found = parse_exitscan(r.stdout)
        # only the tasks this scan was planned for: a task beyond the batch of 16, or one that ended in this very
        # ingest (process.ps runs first), stays pending and is scanned next
        for tid in [t for t in self._exitscan_planned if t in self._exitscan]:
            if tid not in found and r.ok:
                found[tid] = ""
            if tid not in found:
                continue
            state, raw_state, hint = classify_exit(found[tid])
            job = self._ended.get(tid)
            if job is not None:
                self._ended[tid] = dataclasses.replace(job, state=state, raw_state=raw_state, hints=(hint,))
            del self._exitscan[tid]

    def _assemble(self, ctx: HostContext, monitors: MonitorView,
                  ) -> tuple[dict[str, Job], tuple[GPU, ...] | None, tuple[Proc, ...]]:
        now = ctx.now
        assert self._epoch is not None
        gpus = monitors.gpus if monitors.gpus_at is not None else ()
        gpu_pids = frozenset(p.pid for g in gpus for p in g.processes if p.pid in self._by_pid)
        fds = {pid: info for pid, (_, info) in self._fds.items()}
        user = ctx.caps.remote_user
        jobs: dict[str, Job] = {}
        owner: dict[int, JobKey] = {}
        for tid, rec in self._tasks.items():
            job = task_job(rec, self._by_pid, self._epoch, user=user, gpus=gpus, gpus_at=monitors.gpus_at,
                           gpu_pids=gpu_pids, fds=fds, stats=self._stats, observed_at=now)
            if job.gpu is not None and monitors.gpus_at is not None:
                self._gpu_idle.add(tid, monitors.gpus_at, job.gpu.util_mean_pct)
            stdout = next((r for r in job.logs if r.stream == "stdout"), None)
            hints = [h for h in (log_idle_hint(stdout.mtime if stdout else None, now, ctx.general.stall_after),
                                 self._gpu_idle.hint(tid, now)) if h]
            jobs[tid] = dataclasses.replace(job, hints=tuple(hints)) if hints else job
            for pid in rec.members:
                owner[pid] = rec.key
        self._jobs = dict(jobs)
        self._gpu_idle.forget(jobs)
        if monitors.gpus_at is not None:
            remote_now = int(monitors.gpus_at.timestamp() + ctx.caps.clock_skew.total_seconds())
            others = other_user_jobs(monitors.gpu_procs, gpus, remote_now, host=ctx.host.name, me=user,
                                     observed_at=now)
            for key, job in self._stable_others(others).items():
                jobs[key] = job
                owner[job.pids[0]] = job.key
        horizon = now - timedelta(seconds=ctx.general.recent_window)
        for tid in [t for t, j in self._ended.items() if j.end_time is not None and j.end_time < horizon]:
            del self._ended[tid]
            self._exitscan.pop(tid, None)
        jobs.update({tid: j for tid, j in self._ended.items() if tid not in jobs})
        annotated: tuple[GPU, ...] | None = None
        if monitors.gpus_at is not None:
            annotated = tuple(dataclasses.replace(g, processes=tuple(
                dataclasses.replace(p, job=owner.get(p.pid)) for p in g.processes)) for g in monitors.gpus)
        procs: dict[int, Proc] = {}
        for rec in self._tasks.values():
            for pid in rec.members:
                if pid in self._by_pid:
                    procs[pid] = dataclasses.replace(self._by_pid[pid], task=rec.key)
        for p in monitors.gpu_procs:
            if p.pid not in procs:
                procs[p.pid] = dataclasses.replace(p, task=owner.get(p.pid))
        return jobs, annotated, tuple(procs.values())

    def _stable_others(self, others: dict[str, Job]) -> dict[str, Job]:
        """Fix other users' task ids at first sight (their start is derived from etime and would jitter)."""
        out: dict[str, Job] = {}
        seen: set[int] = set()
        for job in others.values():
            pid = job.pids[0]
            start = int(job.raw["root_start"])
            fixed = self._other_starts.get(pid)
            if fixed is None or abs(fixed - start) > START_TOLERANCE_S:
                self._other_starts[pid] = fixed = start
            seen.add(pid)
            key = JobKey(job.key.host, f"p{pid}-{fixed}")
            raw = dict(job.raw)
            raw["root_start"] = str(fixed)
            out[key.id] = dataclasses.replace(job, key=key, raw=raw)
        for pid in [p for p in self._other_starts if p not in seen]:
            del self._other_starts[pid]
        return out

    # ------------------------------------------------------------ details
    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]:
        if job.kind is not JobKind.TASK:
            return []
        out: list[Section] = []
        if job.state.group is StateGroup.ACTIVE and job.pids:
            root = int(job.raw["root_pid"]) if job.is_mine and "root_pid" in job.raw else None
            out.append(Section("process.detail", detail_cmd(job.pids, root), timeout=5, requires=("ps",),
                               meta={"pids": list(job.pids), "root": root}))
        paths = list(dict.fromkeys(r.path for r in job.logs if r.path and _valid_path(r.path)))
        if paths:
            out.append(Section("process.logstat", logstat_cmd(paths), timeout=5, requires=("stat",),
                               meta={"paths": paths}))
        return out

    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail:
        now = ctx.now
        errors: list[HostError] = []
        procs: tuple[Proc, ...] = ()
        env: dict[str, str] = {}
        if (r := results.get("process.detail")) is not None:
            if not r.ok:
                errors.append(_err(r, now))
            if usable(r):
                ps_text, _, env_text = r.stdout.partition("@@env")
                pr = parse_ps_rows(split_rows(ps_text), OWN_COLUMNS, user=job.user)
                procs = tuple(dataclasses.replace(p, task=job.key) for p in pr.items)
                for line in split_rows(env_text):
                    key, eq, value = line.partition("=")
                    if eq and key in ENV_KEYS:
                        env[key] = value
        if (r := results.get("process.logstat")) is not None:
            if usable(r):
                self._stats.update(parse_stat_lines(r.stdout))
            if not r.ok:
                errors.append(_err(r, now))
        logs = tuple(with_stat(ref, self._stats) for ref in job.logs)
        fields: dict[str, str] = {"root pid": job.job_id_raw}
        if job.start_time is not None:
            fields["started"] = job.start_time.isoformat()
        if job.workdir:
            fields["cwd"] = job.workdir
        if job.command:
            fields["command"] = job.command
        for ref in logs:
            if ref.path:
                fields[ref.stream] = ref.path
        explanation: tuple[str, ...] = ()
        if job.state.is_terminal:
            explanation = (job.raw_state, *job.hints)
        note = "process list unavailable" if not procs and job.state.group is StateGroup.ACTIVE else ""
        return JobDetail(job=dataclasses.replace(job, logs=logs), fields=fields, per_node=(), steps=(), logs=logs,
                         gpu=job.gpu, procs=procs, procs_note=note, env=env, explanation=explanation, fetched_at=now,
                         errors=tuple(errors))

    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None:
        if job is None or not job.workdir or not _valid_path(job.workdir):
            return None
        d = shlex.quote(job.workdir)                 # a deleted workdir opens a clearly labelled home shell (UI-41)
        note = shlex.quote(f"runfleet: cannot cd to {job.workdir}, opening home")
        return f'cd {d} 2>/dev/null || {{ echo {note}; cd; }}; exec "$SHELL" -l'

    # ------------------------------------------------------------ actions
    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
        return task_actions.specs(jobs, ctx)

    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan:
        return task_actions.plan(action_id, jobs, params, ctx)

    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]:
        return task_actions.interpret(plan, results, ctx)


def _err(r: SectionResult, at: datetime) -> HostError:
    err = section_error(r, at)
    assert err is not None
    return err
