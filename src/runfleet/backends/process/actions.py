"""[WP2] Task terminate/interrupt/kill plans and the remote signal script (§6.2).

The plan only carries each task's root `(pid, start epoch)`; the member list is re-enumerated on the host at
execution time, because the user's launcher is often a queue runner that starts the next item when a child dies.
The script freezes the root first (SIGSTOP) unless it is the foreground job of a terminal (ps stat `+`: an interactive
shell would take the terminal back and the root would continue as a background job), signals the current tree (and
the process group when it holds nothing else), continues the root, waits `grace_s`, escalates to SIGKILL if asked and
reports survivors and newcomers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from runfleet.backends.base import ActionParam, ActionPlan, ActionSpec, HostContext
from runfleet.errors import ActionError
from runfleet.models import ActionResult, Job, JobKind, StateGroup
from runfleet.probe.section import Section, SectionOutcome, SectionResult

# The §6.2 script; the first line is rendered from validated integers / a fixed signal name. Every member's identity
# (`pid:starttime`, /proc stat field 22) is recorded while the root is frozen; a pid whose start time changed since is
# a reused pid and is never signalled again (`ok`). Nothing is written while the root is stopped, and the EXIT trap
# continues it if the script dies in that window; SIGPIPE is ignored so a dropped channel cannot stop the escalation.
SIGNAL_SCRIPT = r"""R={root} RS={start} SIG={sig} G={grace} ESC={esc}
tree() {{
  ps -e -o pid=,ppid= | awk -v r="$1" '{{par[$1]=$2}} END {{q[1]=r; n=1; seen[r]=1
    for (i = 1; i <= n; i++) for (p in par) if (par[p] == q[i] && !(p in seen)) {{seen[p]=1; q[++n]=p; print p}}}}'
}}
st() {{ awk '{{sub(/.*\) /, ""); print $20}}' "/proc/$1/stat" 2>/dev/null; }}
ok() {{ case " $IDS " in *" $1:$(st "$1") "*) ;; *) return 1 ;; esac; }}
all() {{ L=; for p in $R $SET; do ok "$p" && L="$L $p"; done; ok "$R" && L="$L $(tree "$R")"
  printf '%s\n' $L | sort -un; }}
et=$(ps -o etimes= -p "$R" 2>/dev/null | tr -d ' ')
[ -n "$et" ] || {{ echo "gone $R"; echo done; exit 0; }}
d=$(( $(date +%s) - et - RS )); [ "${{d#-}}" -le 5 ] || {{ echo "reused $R"; echo done; exit 0; }}
FZ=; trap '[ -z "$FZ" ] || kill -s CONT "$R" 2>/dev/null' EXIT; trap 'exit 129' HUP; trap 'exit 130' INT
trap 'exit 143' TERM; trap '' PIPE
case "$(ps -o stat= -p "$R" 2>/dev/null)" in *+*) ;; *) kill -s STOP "$R" 2>/dev/null && FZ=1 ;; esac
SET=$(tree "$R" | tr '\n' ' ')
IDS=$(for p in $R $SET; do printf '%s:%s ' "$p" "$(st "$p")"; done)
PG=$(ps -o pgid= -p "$R" | tr -d ' ')
OUT=$(ps -e -o pid=,pgid= | awk -v g="$PG" -v s=" $R $SET " '$2 == g && index(s, " " $1 " ") == 0 {{print $1}}')
GRP=; [ -n "$PG" ] && [ -z "$OUT" ] && GRP=$PG
M=; for p in $SET $R; do ok "$p" && kill -s "$SIG" "$p" 2>/dev/null && M="$M $p"; done
[ -n "$GRP" ] && kill -s "$SIG" -- "-$GRP" 2>/dev/null
[ -n "$FZ" ] && kill -s CONT "$R" 2>/dev/null; FZ=
echo "members $R $SET"; for p in $M; do echo "sent $p"; done
i=0; while [ "$i" -lt "$G" ]; do
  A=; for p in $(all); do kill -0 "$p" 2>/dev/null && A="$A $p"; done
  [ -z "$A" ] && break; sleep 1; i=$((i + 1))
done
if [ "$ESC" = 1 ]; then
  for p in $(all); do kill -0 "$p" 2>/dev/null && kill -s KILL "$p" 2>/dev/null && echo "killed $p"; done
  [ -n "$GRP" ] && {{ ! kill -0 "$GRP" 2>/dev/null || ok "$GRP"; }} && kill -s KILL -- "-$GRP" 2>/dev/null; sleep 1
fi
for p in $R $SET; do ok "$p" && kill -0 "$p" 2>/dev/null && echo "alive $p"; done
if ok "$R"; then for p in $(tree "$R"); do case " $SET " in *" $p "*) ;; *) echo "new $p" ;; esac; done; fi
echo done"""

SIGNALS = ("TERM", "INT", "KILL")
MAX_GRACE_S = 600

_LABELS = {"task.terminate": "Terminate task", "task.interrupt": "Interrupt task (Ctrl-C)", "task.kill": "Kill task"}
ACTION_IDS = tuple(_LABELS)


def _params(ctx: HostContext, action_id: str) -> tuple[ActionParam, ...]:
    if action_id != "task.terminate":
        return ()
    grace = int(ctx.host.workstation.kill_grace)
    return (ActionParam(name="escalate", kind="bool", default=True, label="send SIGKILL to survivors after"),
            ActionParam(name="grace_s", kind="int", default=grace, label="grace seconds", min=0, max=MAX_GRACE_S))


def render_script(root: int, start: int, sig: str, grace_s: int, escalate: bool) -> str:
    """§6.2 script for one task. Values are integers or a fixed signal name, so no quoting is needed."""
    if sig not in SIGNALS or root <= 1 or start < 0 or not 0 <= grace_s <= MAX_GRACE_S:
        raise ValueError(f"invalid signal plan root={root} start={start} sig={sig} grace={grace_s}")
    return SIGNAL_SCRIPT.format(root=int(root), start=int(start), sig=sig, grace=int(grace_s), esc=1 if escalate else 0)


def ineligible_reason(job: Job, ctx: HostContext) -> str | None:
    if job.kind is not JobKind.TASK or job.key.host != ctx.host.name:
        return "not a task on this host"
    if not job.is_mine:
        return f"owned by {job.user}"
    if job.state.group is not StateGroup.ACTIVE:
        return "task already ended"
    if "root_pid" not in job.raw or "root_start" not in job.raw:
        return "task identity unknown — refresh"
    return None


def specs(jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]:
    reasons = [ineligible_reason(j, ctx) for j in jobs]
    if ctx.general.readonly:
        disabled: str | None = "read-only mode"
    elif not jobs:
        disabled = "no task selected"
    elif any(r is None for r in reasons):
        disabled = None
    else:
        disabled = next(r for r in reasons if r is not None)
    return [ActionSpec(id=aid, label=label, mutating=True, confirm=True, params=_params(ctx, aid),
                       disabled_reason=disabled) for aid, label in _LABELS.items()]


def _resolve(action_id: str, params: Mapping[str, object], ctx: HostContext) -> tuple[str, int, bool]:
    if action_id == "task.interrupt":
        return "INT", 0, False
    if action_id == "task.kill":
        return "KILL", 0, True
    default = int(ctx.host.workstation.kill_grace)
    raw = params.get("grace_s", default)
    grace = raw if isinstance(raw, int) and not isinstance(raw, bool) else default
    return "TERM", max(0, min(MAX_GRACE_S, grace)), bool(params.get("escalate", True))


def plan(action_id: str, jobs: Sequence[Job], params: Mapping[str, object], ctx: HostContext) -> ActionPlan:
    if action_id not in _LABELS:
        raise ActionError(f"unknown task action {action_id!r}")
    eligible: list[Job] = []
    skipped: list[tuple[Job, str]] = []
    for job in jobs:
        reason = ineligible_reason(job, ctx)
        if reason is None:
            eligible.append(job)
        else:
            skipped.append((job, reason))
    if not eligible:
        raise ActionError(skipped[0][1] if skipped else "no task selected")
    sig, grace, escalate = _resolve(action_id, params, ctx)
    sections: list[Section] = []
    previews: list[str] = []
    for job in eligible:
        root, start = int(job.raw["root_pid"]), int(job.raw["root_start"])
        sections.append(Section(
            name=f"process.kill.{root}", cmd=render_script(root, start, sig, grace, escalate),
            timeout=float(grace + 15), requires=("ps", "awk"),
            meta={"root": root, "start": start, "sig": sig, "grace_s": grace, "escalate": escalate}))
        tail = f" · KILL after {grace} s" if sig == "TERM" and escalate else ""
        previews.append(f"STOP {root} · {sig} its tree{tail}" if sig != "KILL" else f"STOP {root} · KILL its tree")
    n = len(eligible)
    verb = {"task.terminate": "Terminate", "task.interrupt": "Interrupt", "task.kill": "Kill"}[action_id]
    title = (f"{verb} task {eligible[0].key.label} on {ctx.host.name}?" if n == 1
             else f"{verb} {n} tasks on {ctx.host.name}?")
    body = []
    for job in eligible[:8]:
        gpus = job.raw.get("gpu_indices") or ""
        body.append(f"{job.key.label}  {job.name} · {len(job.pids)} processes"
                    + (f" · GPUs {gpus}" if gpus else ""))
    if n > 8:
        body.append(f"+{n - 8} more")
    body.append(f"members are re-read on {ctx.host.name} at confirm time; launcher paused first")
    body += [f"skipped {j.key.label}: {r}" for j, r in skipped]
    out_params: dict[str, object] = {"escalate": escalate, "grace_s": grace} if action_id == "task.terminate" else {}
    return ActionPlan(spec=next(s for s in specs(jobs, ctx) if s.id == action_id), host=ctx.host.name,
                      targets=tuple(j.key for j in eligible), skipped=tuple((j.key, r) for j, r in skipped),
                      params=out_params, sections=tuple(sections), preview="\n".join(previews),
                      confirm_title=title, confirm_body="\n".join(body))


def interpret_lines(lines: Sequence[str]) -> tuple[bool, str]:
    """Script output → (ok, message): `gone` → ok "already exited"; `reused` → not ok; any `alive`/`new` → not ok."""
    words = [ln.split() for ln in lines if ln.strip()]
    if any(w[0] == "gone" for w in words):
        return True, "already exited"
    if any(w[0] == "reused" for w in words):
        return False, "pid reused — refresh"
    alive = [w[1] for w in words if w[0] == "alive" and len(w) > 1]
    new = [w[1] for w in words if w[0] == "new" and len(w) > 1]
    if alive or new:
        parts = []
        if alive:
            parts.append(f"{len(alive)} process{'es' if len(alive) != 1 else ''} survived")
        if new:
            parts.append(f"{len(new)} process{'es' if len(new) != 1 else ''} started during termination: "
                         + " ".join(new))
        return False, "; ".join(parts)
    if not any(w[0] == "done" for w in words):
        return False, "signal script did not finish"
    sent = sum(1 for w in words if w[0] == "sent")
    killed = sum(1 for w in words if w[0] == "killed")
    msg = f"signalled {sent} process{'es' if sent != 1 else ''}"
    return True, msg + (f", killed {killed}" if killed else "")


def interpret(plan_: ActionPlan, results: Mapping[str, SectionResult], ctx: HostContext) -> list[ActionResult]:
    out: list[ActionResult] = []
    for key, section in zip(plan_.targets, plan_.sections, strict=True):
        res = results.get(section.name)
        if res is None or res.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.MISSING):
            out.append(ActionResult(action_id=plan_.spec.id, target=key, ok=False,
                                    message="not run" if res is None else f"not run ({res.outcome.value})",
                                    stderr="" if res is None else res.stderr))
            continue
        lines = res.stdout.splitlines()
        ok, message = interpret_lines(lines)
        if res.outcome is SectionOutcome.TIMEOUT and not any(ln.strip() == "done" for ln in lines):
            ok, message = False, "timed out" + (f" ({message})" if lines else "")
        out.append(ActionResult(action_id=plan_.spec.id, target=key, ok=ok, message=message,
                                details=tuple(ln for ln in lines if ln.strip()), rc=res.rc, stdout=res.stdout,
                                stderr=res.stderr, duration_s=None if res.ms is None else res.ms / 1000))
    return out
