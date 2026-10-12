"""[WP2] Slurm cancel/hold/release/requeue specs, plans, result interpretation (§6).

Plans are pure and cheap (ConfirmScreen re-plans on every param change). Remote ids are the display ids with any
array throttle stripped (`31270818_[1-24%1]` → `31270818_[1-24]`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from runfleet.backends.base import ActionParam, ActionPlan, ActionSpec, HostContext
from runfleet.backends.slurm.commands import scancel_cmd, scontrol_action_cmd, strip_throttle
from runfleet.backends.slurm.states import requeue_window_open
from runfleet.errors import ActionError
from runfleet.models import ActionResult, Job, JobKind, JobState
from runfleet.probe.section import Section, SectionOutcome, SectionResult
from runfleet.util.slurmfmt import compress_tasks, expand_array_tasks

CANCELLABLE = frozenset({JobState.PENDING, JobState.CONFIGURING, JobState.REQUEUED, JobState.RUNNING,
                         JobState.COMPLETING, JobState.SUSPENDED, JobState.STOPPED})

HOLD_PARAM = ActionParam(name="hold", kind="bool", default=False, label="hold the job after requeueing")

_LABELS = {"slurm.cancel": "Cancel job", "slurm.hold": "Hold job", "slurm.release": "Release job",
           "slurm.requeue": "Requeue job"}
_VERBS = {"slurm.cancel": ("Cancel", "cancelled"), "slurm.hold": ("Hold", "held"),
          "slurm.release": ("Release", "released"), "slurm.requeue": ("Requeue", "requeued")}
ACTION_IDS = tuple(_LABELS)


def _protection(job: Job, ctx: HostContext) -> str | None:
    s = ctx.host.slurm
    for p in (job.partition or "").split(","):
        if p and p in s.protected_partitions:
            return f"protected partition {p}"
    if job.qos and job.qos in s.protected_qos:
        return f"protected QOS {job.qos}"
    return None


def abbreviated_array(job: Job) -> bool:
    """A pending array summary whose task list Slurm printed in a form that is not a valid job id: a step range
    (`1-99:2`) or a list truncated with `...` (SLURM_BITSTR_LEN)."""
    task = job.array_task or ""
    return not task.isdigit() and (":" in task or "..." in task)


def _remote_id(job: Job) -> str:
    """The id an action names: the display id minus the throttle. An abbreviated summary names its tasks explicitly
    (`500_[1-99:2]` → `500_[1,3,…,99]`); only a truncated list (`...`) falls back to the array job id, and the cancel
    then carries `--state=PENDING` (it reaches every pending task of that array, said so in the confirm body)."""
    if abbreviated_array(job):
        tasks = expand_array_tasks(job.array_task)
        parent = job.array_job_id or job.job_id_raw
        return f"{parent}_[{compress_tasks(tasks)}]" if tasks else parent
    return strip_throttle(job.key.id)


def _widened(job: Job) -> bool:
    """A summary whose task list cannot be named: its cancel reaches every pending task of the array."""
    return abbreviated_array(job) and expand_array_tasks(job.array_task) is None


def _state_reason(action_id: str, job: Job, ctx: HostContext) -> str | None:
    if action_id in ("slurm.hold", "slurm.release") and abbreviated_array(job):
        return "Slurm abbreviated this array's task list; hold/release single tasks or from a shell"
    st = job.state
    if action_id == "slurm.cancel":
        return None if st in CANCELLABLE else ("job already ended" if st.is_terminal else f"cannot cancel in state "
                                                                                          f"{job.raw_state}")
    if action_id == "slurm.hold":
        if st is not JobState.PENDING:
            return "only pending jobs can be held"
        return "already held" if job.held else None
    if action_id == "slurm.release":
        return None if job.held else "not held"
    if action_id == "slurm.requeue":
        if st in (JobState.RUNNING, JobState.SUSPENDED):
            return None
        if st.is_terminal:
            return None if requeue_window_open(job, ctx.now, ctx.caps.slurm_config) else \
                "no longer in controller memory — resubmit"
        return "only running jobs can be requeued"
    return f"unknown action {action_id}"


def ineligible_reason(action_id: str, job: Job, ctx: HostContext) -> str | None:
    """Why `job` cannot get `action_id` (None = eligible): kind, ownership, protection, state."""
    if job.kind is not JobKind.SLURM or job.key.host != ctx.host.name:
        return "not a Slurm job on this host"
    if not job.is_mine:
        return f"owned by {job.user}"
    return _protection(job, ctx) or _state_reason(action_id, job, ctx)


def specs(jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
    """Every Slurm action for these jobs; enabled if >= 1 job is eligible, else greyed with the first reason."""
    out: list[ActionSpec] = []
    for aid, label in _LABELS.items():
        reasons = [ineligible_reason(aid, j, ctx) for j in jobs]
        disabled: str | None
        if ctx.general.readonly:
            disabled = "read-only mode"
        elif not jobs:
            disabled = "no job selected"
        elif any(r is None for r in reasons):
            disabled = None
        else:
            disabled = next(r for r in reasons if r is not None)
        out.append(ActionSpec(id=aid, label=label, mutating=True, confirm=True,
                              params=(HOLD_PARAM,) if aid == "slurm.requeue" else (), disabled_reason=disabled))
    return out


def _elapsed(job: Job) -> str:
    if job.elapsed is None:
        return "—"
    s = int(job.elapsed.total_seconds())
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return (f"{d}-" if d else "") + f"{h}:{m:02d}:{sec:02d}"


def plan(action_id: str, jobs: Sequence[Job], params: Mapping[str, object], ctx: HostContext) -> ActionPlan:
    if action_id not in _LABELS:
        raise ActionError(f"unknown Slurm action {action_id!r}")
    spec = next(s for s in specs(jobs, ctx) if s.id == action_id)
    eligible: list[Job] = []
    skipped: list[tuple[Job, str]] = []
    for job in jobs:
        reason = ineligible_reason(action_id, job, ctx)
        if reason is None:
            eligible.append(job)
        else:
            skipped.append((job, reason))
    if not eligible:
        raise ActionError(skipped[0][1] if skipped else "no job selected")
    ids = [_remote_id(j) for j in eligible]
    hold = bool(params.get("hold", False)) if action_id == "slurm.requeue" else False
    try:
        if action_id == "slurm.cancel":
            plain = [i for i, j in zip(ids, eligible, strict=True) if not _widened(j)]
            pending = [i for i, j in zip(ids, eligible, strict=True) if _widened(j)]
            parts = ([scancel_cmd(plain)] if plain else []) + ([scancel_cmd(pending, pending_only=True)]
                                                               if pending else [])
            cmd = parts[0] if len(parts) == 1 else f"{parts[0]}; a=$?; {parts[1]} && [ $a -eq 0 ]"
        else:
            verb = {"slurm.hold": "hold", "slurm.release": "release"}.get(action_id,
                                                                         "requeuehold" if hold else "requeue")
            cmd = scontrol_action_cmd(verb, ids)
    except ValueError as e:          # an id the validators refuse must never reach the UI as a crash
        raise ActionError(f"cannot build the command: {e}") from None
    meta: dict[str, object] = {"ids": ids}
    if action_id == "slurm.requeue":
        meta["hold"] = hold
    section = Section(name=f"slurm.act.{action_id.split('.', 1)[1]}", cmd=cmd, timeout=30.0,
                      requires=("scancel",) if action_id == "slurm.cancel" else ("scontrol",), meta=meta)
    n = len(eligible)
    title_verb = _VERBS[action_id][0]
    title = f"{title_verb} {n} job{'s' if n != 1 else ''} on {ctx.host.name}?"
    body = [f"{j.key.id}  {j.name}  {j.state.value}  {_elapsed(j)}" for j in eligible[:8]]
    if n > 8:
        body.append(f"+{n - 8} more")
    if action_id == "slurm.cancel":
        body += [f"{j.key.id}: Slurm truncated the task list; this cancels ALL pending tasks of array "
                 f"{j.array_job_id or j.job_id_raw}" for j in eligible if _widened(j)]
    body += [f"skipped {j.key.id}: {r}" for j, r in skipped]
    if action_id == "slurm.requeue":
        body.append("the job restarts from the beginning" + (" and stays held" if hold else ""))
    return ActionPlan(spec=spec, host=ctx.host.name, targets=tuple(j.key for j in eligible),
                      skipped=tuple((j.key, r) for j, r in skipped), params={"hold": hold} if
                      action_id == "slurm.requeue" else {}, sections=(section,), preview=cmd, confirm_title=title,
                      confirm_body="\n".join(body))


def _matcher(display_id: str) -> Callable[[str], bool]:
    ids = {display_id, strip_throttle(display_id), display_id.split("_", 1)[0]}
    pats = [re.compile(rf"(?<![\w\[]){re.escape(i)}(?![\w\]])") for i in ids]
    task_pat: re.Pattern[str] | None = None
    tasks: set[int] | None = None
    if "_[" in display_id:           # per-task lines of a summary target: `500_7: Job has already finished`
        parent, _, task = display_id.partition("_")
        task_pat = re.compile(rf"(?<![\w\[]){re.escape(parent)}_(\d+)(?![\w\]])")
        listed = expand_array_tasks(task)
        tasks = set(listed) if listed is not None else None   # truncated list: any task of the array may be ours

    def match(line: str) -> bool:
        if any(p.search(line) for p in pats):
            return True
        return task_pat is not None and any(tasks is None or int(m[1]) in tasks for m in task_pat.finditer(line))
    return match


_HAS_ID = re.compile(r"\b\d{2,}")


def _humanize_error(line: str) -> str:
    if "Invalid job id" in line:
        return "job already ended"
    if "Access/permission denied" in line or "Permission denied" in line:
        return "permission denied"
    tail = line.rsplit(": ", 1)[-1].strip()
    return tail or line.strip()


def interpret(plan_: ActionPlan, results: Mapping[str, SectionResult], ctx: HostContext) -> list[ActionResult]:
    """One ActionResult per target. scancel's `Kill job error on job id <id>: <reason>` (and scontrol's per-id
    errors) map to that id; `Invalid job id specified` → "job already ended"."""
    action_id = plan_.spec.id
    done = _VERBS[action_id][1]
    section = plan_.sections[0]
    res = results.get(section.name)
    if res is None or res.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.INCOMPLETE, SectionOutcome.TIMEOUT,
                                      SectionOutcome.MISSING):
        why = "not run" if res is None else {SectionOutcome.TIMEOUT: "timed out",
                                             SectionOutcome.MISSING: "command not installed"}.get(
            res.outcome, f"did not finish ({res.outcome.value})")
        return [ActionResult(action_id=action_id, target=k, ok=False, message=why, rc=None if res is None else res.rc,
                             stdout="" if res is None else res.stdout, stderr="" if res is None else res.stderr,
                             duration_s=None if res is None or res.ms is None else res.ms / 1000)
                for k in plan_.targets]
    err_lines = [ln.strip() for ln in (res.stderr + "\n" + res.stdout).splitlines()
                 if ln.strip() and re.search(r"(?i)error|invalid|denied", ln)]
    matched: dict[str, list[str]] = {k.id: [ln for ln in err_lines if _matcher(k.id)(ln)] for k in plan_.targets}
    any_matched = any(matched.values())
    # scontrol follows each per-id error with an id-less summary (`slurm_suspend error:Invalid job id specified`);
    # once per-id lines account for the failures, such lines say nothing about the other targets
    unmatched = [ln for ln in err_lines if not any(ln in v for v in matched.values())
                 and not (any_matched and not _HAS_ID.search(ln))]
    out: list[ActionResult] = []
    for key in plan_.targets:
        mine = matched[key.id]
        if mine:
            ok, message = False, _humanize_error(mine[0])
        elif res.rc == 0 or (err_lines and not unmatched):
            ok, message = True, f"{done} {key.id}"   # rc != 0 only because of other targets' errors
        else:
            ok, message = False, _humanize_error(unmatched[0]) if unmatched else f"rc {res.rc}"
        out.append(ActionResult(action_id=action_id, target=key, ok=ok, message=message,
                                details=tuple(mine or (unmatched if not ok else ())), rc=res.rc, stdout=res.stdout,
                                stderr=res.stderr, duration_s=None if res.ms is None else res.ms / 1000))
    return out
