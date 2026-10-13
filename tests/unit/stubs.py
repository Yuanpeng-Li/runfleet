"""[WP1] Stub collectors, backend hooks and clock helpers for core tests (until/independent of WP2's collectors).

The stubs follow the §3.6 contract with tiny line formats:
- `slurm.squeue` stdout: one job per line, "<id> <state-code>" (e.g. "101 R"); new ids request `slurm.jobinfo`.
- `procfs.host` stdout: "load=<float>".
- `nvidia.gpu` stdout: one GPU per line, "<index> <util>".
- `process.ps` stdout: one task per line, "<pid> <gpu index or ->".
- Attach (§7.16), StubSlurm only: `slurm.attach.node0` stdout = "landed=<raw job id>" then one GPU per line
  "gpu <index> <util> <mem used MiB>" (meta: mode, node, job_id_raw); stdout "denied" (or stderr containing
  "Permission denied") = ssh refused -> caps_patch {"node_ssh": "denied"}; any landing -> {"node_ssh": "adopt"} in ssh
  mode; ssh stderr "Host key verification failed" -> one HOSTKEY error, no patch. GPUs with util < 5 and mem > 1024
  extend `idle_streak` (by uuid "GPU-<index>") from `previous` like the backend. `slurm.attach.audit` stdout "clean"
  -> JobLive.clean True, anything else -> False (detail = the text).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

from runfleet.backends.base import (ActionPlan, ActionSpec, AttachPlan, Cadence, Collector, CollectorOutput,
                                    HostContext, MonitorCollector, MonitorView, SchedulerCollector)
from runfleet.config import HostConfig
from runfleet.core.clock import ManualClock
from runfleet.core.poller import BackendHooks
from runfleet.errors import ActionError, AttachBlocked, RunFleetError
from runfleet.models import (DETECT_REV, GPU, ActionResult, AttachMode, Capabilities, ErrorKind, HostError, HostMetrics,
                             Job, JobDetail, JobKey, JobKind, JobLive, JobState, LogRef, NodeInspect)
from runfleet.probe import framing
from runfleet.probe.section import Section, SectionOutcome, SectionResult

T0 = datetime(2026, 9, 7, 10, 42, 37).astimezone()   # replaced by CLOCK_START in tests that need exact values


def _ok(r: SectionResult | None) -> bool:
    return r is not None and r.outcome in (SectionOutcome.OK, SectionOutcome.EMPTY)


class StubSlurm(SchedulerCollector):
    name = "slurm"
    ingests: list[tuple[frozenset[str], MonitorView]] = []

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        self.jobs: dict[str, Job] = {}
        self.seen_info: set[str] = set()
        self.raise_on_ingest = False

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return True

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        s = ctx.host.slurm
        return [Cadence("slurm.jobs", s.poll_jobs, floor=10),
                Cadence("slurm.history", s.poll_history, floor=60),
                Cadence("slurm.nodes", s.poll_nodes, floor=60, hidden_factor=5),
                Cadence("slurm.jobinfo", 0, mode="event")]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        out = []
        if "slurm.jobs" in due:
            out.append(Section("slurm.squeue", "squeue --me", timeout=20, requires=("squeue",)))
        if "slurm.history" in due:
            out.append(Section("slurm.sacct", "sacct", timeout=30))
        if "slurm.nodes" in due:
            out.append(Section("slurm.sinfo", "sinfo", timeout=20))
        if "slurm.jobinfo" in due:
            ids = sorted(set(self.jobs) - self.seen_info)
            out.append(Section("slurm.scontrol", "scontrol " + " ".join(ids), meta={"ids_raw": ids}))
        return out

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        type(self).ingests.append((frozenset(results), monitors))
        if self.raise_on_ingest:
            raise ValueError("stub bug")
        followups: set[str] = set()
        jobs: dict[str, Job] | None = None
        if "slurm.scontrol" in results:
            self.seen_info |= set(results["slurm.scontrol"].stdout.split() or self.jobs)
        sq = results.get("slurm.squeue")
        errors: tuple[HostError, ...] = ()
        if sq is not None and sq.stdout == "garbage":
            errors = (HostError(ErrorKind.PARSE, "could not parse squeue", section="slurm.squeue"),)
        elif _ok(sq):
            assert sq is not None
            jobs = {}
            for line in sq.stdout.splitlines():
                jid, code = line.split()
                jobs[jid] = Job(key=JobKey(ctx.host.name, jid), kind=JobKind.SLURM, name=f"job{jid}", user="me",
                                state=JobState(code), raw_state=code, job_id_raw=jid, observed_at=ctx.now,
                                logs=(LogRef("stdout", f"/w/{jid}.out", source="squeue"),),
                                elapsed=timedelta(seconds=(ctx.now.timestamp() % 1000)))
            if set(jobs) - self.seen_info:
                followups.add("slurm.jobinfo")
            self.jobs = jobs
        return CollectorOutput(jobs=jobs, errors=errors, followups=frozenset(followups))

    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
        live = [j for j in jobs if not j.state.is_terminal]
        return [ActionSpec("slurm.cancel", "Cancel job", mutating=True, confirm=True,
                           disabled_reason=None if live else "job already ended")]

    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan:
        targets = tuple(j.key for j in jobs if not j.state.is_terminal)
        if action_id != "slurm.cancel" or not targets:
            raise ActionError("nothing to cancel")
        ids = [k.id for k in targets]
        spec = self.actions(jobs, ctx)[0]
        return ActionPlan(spec=spec, host=ctx.host.name, targets=targets, skipped=(), params=dict(params),
                          sections=(Section("slurm.act.cancel", "scancel " + " ".join(ids), meta={"ids": ids}),),
                          preview="scancel " + " ".join(ids), confirm_title=f"Cancel {len(ids)} job(s)?",
                          confirm_body="")

    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]:
        r = results["slurm.act.cancel"]
        return [ActionResult(plan.spec.id, t, r.ok, f"cancelled {t.id}" if r.ok else r.stderr, rc=r.rc,
                             stdout=r.stdout, stderr=r.stderr) for t in plan.targets]

    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]:
        return [Section("slurm.steps", f"sacct -j {job.job_id_raw}")]

    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail:
        r = results.get("slurm.steps")
        return JobDetail(job=job, fields={"steps": r.stdout if r else ""}, per_node=(), steps=(), logs=job.logs,
                         gpu=None, procs=(), procs_note="", env={}, explanation=(), fetched_at=ctx.now)

    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None:
        return None if job is None else f"cd /w/{job.key.id} && exec $SHELL -l"

    # ---- Attach (tiny formats, see the module docstring)
    def attach_modes(self, job: Job, ctx: HostContext) -> tuple[AttachMode, ...]:
        if job.state is not JobState.RUNNING:
            return ()
        return ("srun",) if ctx.caps.node_ssh == "denied" else ("ssh", "srun")

    def plan_attach(self, job: Job, mode: AttachMode, ctx: HostContext, *,
                    node: str | None = None) -> AttachPlan:
        if job.state is not JobState.RUNNING:
            raise AttachBlocked("job is not running")
        target = node or (job.nodes[0] if job.nodes else "n1")
        inner = framing.make_script([Section("node.id", "hostname -s", timeout=5)], "attach")
        cmd = f"{mode} {target} {job.job_id_raw}"
        sec = Section("slurm.attach.node0", cmd, timeout=14 if mode == "ssh" else 27,
                      meta={"mode": mode, "node": target, "job_id_raw": job.job_id_raw})
        audit = Section("slurm.attach.audit", f"audit {job.job_id_raw}", timeout=10, late=True,
                        meta={"mode": mode, "job_id_raw": job.job_id_raw, "nodes": [target]})
        return AttachPlan(key=job.key, mode=mode, job_id_raw=job.job_id_raw, nodes=(target,), sections=(sec, audit),
                          inner={sec.name: inner}, preview=cmd, sacct_rows=1 if mode == "srun" else 0)

    def ingest_attach(self, plan: AttachPlan, results: Mapping[str, SectionResult], ctx: HostContext, *,
                      previous: JobLive | None = None) -> tuple[JobLive, Mapping[str, object]]:
        r = results.get("slurm.attach.node0")
        nodes: tuple[NodeInspect, ...] = ()
        errors: tuple[HostError, ...] = ()
        patch: dict[str, object] = {}
        rows = 0
        streak: dict[str, int] = {}
        if r is not None and (r.stdout.strip() == "denied" or "Permission denied" in r.stderr):
            patch["node_ssh"] = "denied"
            errors = (HostError(ErrorKind.PERMISSION, "ssh to the node was refused", section=r.name),)
        elif r is not None and plan.mode == "ssh" and "Host key verification failed" in r.stderr:
            errors = (HostError(ErrorKind.HOSTKEY, f"{plan.nodes[0]}'s host key is not known on the login host",
                                r.stderr, section=r.name),)
        elif _ok(r):
            assert r is not None
            landed: str | None = None
            gpus = []
            for line in r.stdout.splitlines():
                if line.startswith("landed="):
                    landed = line.split("=", 1)[1]
                elif line.startswith("gpu "):
                    _, idx, util, mem = line.split()
                    gpus.append(GPU(host=ctx.host.name, index=int(idx), uuid=f"GPU-{idx}", name="Stub GPU",
                                    util_pct=float(util), mem_used_mib=float(mem), mem_total_mib=None,
                                    mem_util_pct=None, temp_c=None, power_w=None, power_limit_w=None, fan_pct=None,
                                    pstate=None, source="nvidia-smi", sampled_at=ctx.now))
                    if float(util) < 5 and float(mem) > 1024:
                        prev = previous.idle_streak if previous is not None else {}
                        streak[f"GPU-{idx}"] = prev.get(f"GPU-{idx}", 0) + 1
            nodes = (NodeInspect(node=plan.nodes[0], target_job=plan.job_id_raw, landed_job=landed,
                                 gpus_of_target=landed == plan.job_id_raw, gpus=tuple(gpus), cost_ms=r.ms),)
            if plan.mode == "ssh":
                patch["node_ssh"] = "adopt"
            rows = plan.sacct_rows
        elif r is not None:
            errors = (HostError(ErrorKind.REMOTE, f"attach failed: {r.outcome}", r.stderr, section=r.name),)
            rows = plan.sacct_rows if plan.mode == "srun" and r.outcome is not SectionOutcome.NOT_RUN else 0
        a = results.get("slurm.attach.audit")
        clean = None if a is None or not _ok(a) else a.stdout.strip() == "clean"
        detail = "" if a is None else a.stdout.strip()
        return JobLive(key=plan.key, mode=plan.mode, sampled_at=ctx.now, nodes=nodes, sacct_rows_added=rows,
                       clean=clean, audit_detail=detail, idle_streak=streak, errors=errors), patch


class StubProcfs(MonitorCollector):
    name = "procfs"
    piggyback_interval: float | None = None      # tests may shorten the Slurm piggyback period

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return True

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        if "slurm" in ctx.caps.profile:
            iv = self.piggyback_interval or ctx.host.slurm.poll_jobs
            return [Cadence("procfs.host", iv, floor=min(iv, 10), mode="piggyback")]
        return [Cadence("procfs.host", ctx.host.workstation.poll_gpu, floor=2)]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        return [Section("procfs.host", "cat /proc/loadavg", timeout=5)] if "procfs.host" in due else []

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        r = results.get("procfs.host")
        if r is None or r.outcome is not SectionOutcome.OK:
            return CollectorOutput()
        load = float(r.stdout.split("=", 1)[1]) if "=" in r.stdout else 0.0
        return CollectorOutput(metrics=HostMetrics(ncpu=8, load=(load, load, load), cpu_pct=None, mem_total=None,
                                                   mem_available=None, swap_total=None, swap_free=None,
                                                   uptime_s=None, sampled_at=ctx.now))


class StubNvidia(MonitorCollector):
    name = "nvidia"

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return True

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        return [Cadence("nvidia.gpu", ctx.host.workstation.poll_gpu, floor=2, focused_factor=0.6)]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        return [Section("nvidia.gpu", "nvidia-smi", timeout=8, late=True)] if "nvidia.gpu" in due else []

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        r = results.get("nvidia.gpu")
        if r is None or r.outcome is not SectionOutcome.OK:
            return CollectorOutput()
        gpus = []
        for line in r.stdout.splitlines():
            idx, util = line.split()
            gpus.append(GPU(host=ctx.host.name, index=int(idx), uuid=f"GPU-{idx}", name="Stub GPU",
                            util_pct=float(util), mem_used_mib=None, mem_total_mib=None, mem_util_pct=None,
                            temp_c=None, power_w=None, power_limit_w=None, fan_pct=None, pstate=None,
                            source="nvidia-smi", sampled_at=ctx.now))
        return CollectorOutput(gpus=tuple(gpus), gpu_procs=())


class StubProcess(SchedulerCollector):
    """Workstation scheduler: re-annotates MonitorView GPUs on every tick (marks them name="annotated")."""
    name = "process"
    views: list[MonitorView] = []

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        self.jobs: dict[str, Job] = {}

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return True

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        return [Cadence("process.ps", ctx.host.workstation.poll_procs, floor=5, focused_factor=0.5)]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        return [Section("process.ps", "ps", timeout=5)] if "process.ps" in due else []

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        type(self).views.append(monitors)
        r = results.get("process.ps")
        if _ok(r):
            assert r is not None
            self.jobs = {}
            for line in r.stdout.splitlines():
                pid = line.split()[0]
                jid = f"p{pid}-1000"
                self.jobs[jid] = Job(key=JobKey(ctx.host.name, jid), kind=JobKind.TASK, name=f"task{pid}",
                                     user="me", state=JobState.RUNNING, raw_state="R", job_id_raw=pid,
                                     source="ps", observed_at=ctx.now)
            ended = getattr(self, "ended", ())
            for jid in ended:
                self.jobs[jid] = Job(key=JobKey(ctx.host.name, jid), kind=JobKind.TASK, name="ended", user="me",
                                     state=JobState.EXITED, raw_state="exited", job_id_raw=jid,
                                     end_time=ctx.now, source="ps")
        gpus = None
        if monitors.gpus_at is not None:
            gpus = tuple(replace(g, name="annotated") for g in monitors.gpus)
        return CollectorOutput(jobs=dict(self.jobs) if _ok(r) else None, gpus=gpus)

    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
        return [ActionSpec("task.terminate", "Terminate", mutating=True, confirm=True)]

    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan:
        if not jobs:
            raise ActionError("no task")
        targets = tuple(j.key for j in jobs)
        grace = int(params.get("grace_s", 10))   # type: ignore[call-overload]
        secs = tuple(Section(f"process.kill.{j.job_id_raw}", f"kill {j.job_id_raw}", timeout=grace + 15)
                     for j in jobs)
        return ActionPlan(spec=self.actions(jobs, ctx)[0], host=ctx.host.name, targets=targets, skipped=(),
                          params=dict(params), sections=secs, preview=f"TERM, grace {grace}s",
                          confirm_title="Terminate?", confirm_body="")

    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]:
        return [ActionResult(plan.spec.id, t, True, "terminated") for t in plan.targets]

    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]:
        return []

    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail:
        return JobDetail(job=job, fields={}, per_node=(), steps=(), logs=(), gpu=None, procs=(), procs_note="",
                         env={}, explanation=(), fetched_at=ctx.now)

    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None:
        return None


REGISTRY: dict[str, type[Collector]] = {"slurm": StubSlurm, "process": StubProcess, "nvidia": StubNvidia,
                                        "procfs": StubProcfs}


def stub_detect_sections(host: HostConfig, collectors: Mapping[str, type[Collector]]) -> list[Section]:
    return [Section("detect.id", "id -un", timeout=10), Section("detect.tools", "command -v squeue", timeout=10)]


def stub_caps(host: HostConfig, profile: tuple[str, ...] | None = None, **kw: object) -> Capabilities:
    if profile is None:
        profile = ("process", "nvidia", "procfs") if host.backend == "workstation" else ("slurm", "procfs")
    return Capabilities(remote_user="me", uid=1000, login_host=host.name, tz_offset=timedelta(0),
                        clock_skew=timedelta(0), tools=frozenset({"squeue", "sinfo", "less", "timeout"}),
                        profile=profile, detect_rev=DETECT_REV, **kw)   # type: ignore[arg-type]


def stub_build_capabilities(results: Mapping[str, SectionResult], host: HostConfig, now: datetime,
                            collectors: Mapping[str, type[Collector]]) -> Capabilities:
    r = results.get("detect.id")
    if r is None or r.outcome is not SectionOutcome.OK and r.outcome is not SectionOutcome.EMPTY:
        raise RunFleetError("detect.id failed")
    return stub_caps(host)


STUB_HOOKS = BackendHooks(collectors=lambda: dict(REGISTRY), detect_sections=stub_detect_sections,
                          build_capabilities=stub_build_capabilities)


# ---------------------------------------------------------------- driving a ManualClock
async def settle(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def run_for(clock: ManualClock, seconds: float, step: float = 0.5) -> None:
    """Advance `clock` by `seconds` in `step`s, letting tasks run after every step."""
    await settle()
    steps = round(seconds / step)
    for _ in range(steps):
        clock.advance(step)
        await settle()
