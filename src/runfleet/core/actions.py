"""[WP1] ActionService: eligibility, planning, execution, action log (§6.3).

Backends own what an action means (specs, plans, interpretation); this service adds the host-level rules (read-only
mode, host not OK, stale data), runs one `purpose="action"` probe per host on the on-demand lane, keeps the session's
action log and wakes the affected hosts so the tables reflect the change within ~3 s.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from runfleet.backends.base import ActionPlan, ActionSpec, HostContext, SchedulerCollector
from runfleet.core.clock import Clock
from runfleet.core.store import FleetStore
from runfleet.errors import ActionError
from runfleet.models import ActionResult, ConnState, Job, JobKey
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult

log = logging.getLogger("runfleet.core.actions")

READONLY_REASON = "read-only mode"
ACTION_LOG_SIZE = 200
# Cadence groups to refresh after an action, per scheduler collector (§6.3 step 5).
WAKE_GROUPS: Mapping[str, frozenset[str]] = {
    "slurm": frozenset({"slurm.jobs", "slurm.history"}),
    "process": frozenset({"process.ps"}),
}
# The section whose rows actions act on, per scheduler collector: other sections answering keeps the host's last_ok
# fresh while this one fails, and its rows are then only as fresh as its own last success.
JOB_SECTIONS: Mapping[str, str] = {"slurm": "slurm.squeue", "process": "process.ps"}
_STATE_LABEL: Mapping[ConnState, str] = {
    ConnState.NEW: "not connected yet", ConnState.CONNECTING: "connecting", ConnState.RETRYING: "retrying",
    ConnState.UNREACHABLE: "unreachable", ConnState.NEEDS_AUTH: "needs auth", ConnState.HOSTKEY: "host key problem",
    ConnState.DISABLED: "disabled",
}

class OnDemandProbe(Protocol):
    def __call__(self, host: str, script: ProbeScript, *,
                 recheck: Callable[[], str | None] | None = None) -> Awaitable[ProbeResult]: ...


def short_age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h"


class ActionService:
    def __init__(self, store: FleetStore, clock: Clock, *, readonly: bool,
                 scheduler_for: Callable[[str], SchedulerCollector | None],
                 context_for: Callable[[str], HostContext | None], probe: OnDemandProbe,
                 wake: Callable[[str, frozenset[str]], None]) -> None:
        self.store = store
        self.clock = clock
        self.readonly = readonly
        self._scheduler_for = scheduler_for
        self._context_for = context_for
        self._probe = probe
        self._wake = wake
        self._log: deque[ActionResult] = deque(maxlen=ACTION_LOG_SIZE)

    # ------------------------------------------------------------ eligibility
    def host_reason(self, host: str) -> str | None:
        """Host-level reason that disables every action on `host` (§3.7 precedence after read-only)."""
        hs = self.store.hosts.get(host)
        if hs is None:
            return f"{host}: unknown host"
        if hs.status.state not in (ConnState.OK, ConnState.DEGRADED):
            return f"{host}: {_STATE_LABEL.get(hs.status.state, hs.status.state.value)}"
        now = self.clock.now()
        if hs.stale(now):
            age = (now - hs.status.last_ok).total_seconds() if hs.status.last_ok else 0.0
            return f"data is {short_age(age)} old — refresh first"
        sched = self._scheduler_for(host)
        sec = hs.status.sections.get(JOB_SECTIONS.get(sched.name, "")) if sched is not None else None
        if sec is not None and not sec.ok:
            sec_age = (now - sec.last_ok).total_seconds() if sec.last_ok else None
            if sec_age is None or sec_age > 2 * (hs.status.tick_interval or 0.0) + 5.0:
                shown = f"{short_age(sec_age)} old" if sec_age is not None else "not current"
                return f"data is {shown} — {sec.name} failing"
        return None

    def _still_allowed(self, plan: ActionPlan) -> str | None:
        """Re-checked right before dispatch: master preparation and the lane wait may take a while after the
        confirmation, and read-only mode or the host may have changed meanwhile."""
        if self.readonly and plan.spec.mutating:
            return READONLY_REASON
        hs = self.store.hosts.get(plan.host)
        if hs is None or not hs.config.enabled:
            return f"{plan.host}: host removed or disabled — not run"
        return None

    def _by_host(self, keys: Sequence[JobKey]) -> dict[str, list[Job]]:
        groups: dict[str, list[Job]] = {}
        for k in keys:
            job = self.store.job(k)
            if job is not None:
                groups.setdefault(k.host, []).append(job)
        return {h: groups[h] for h in self.store.hosts if h in groups}   # config order

    def actions_for(self, keys: Sequence[JobKey]) -> list[ActionSpec]:
        """Merged per action id across hosts; enabled if any host can run it (§3.7 actions_for)."""
        per_id: dict[str, list[tuple[ActionSpec, str | None]]] = {}
        for host, jobs in self._by_host(keys).items():
            sched, ctx = self._scheduler_for(host), self._context_for(host)
            if sched is None or ctx is None:
                continue
            try:
                specs = sched.actions(jobs, ctx)
            except Exception:
                log.exception("%s: actions() failed", host)
                continue
            reason = self.host_reason(host)
            for spec in specs:
                per_id.setdefault(spec.id, []).append((spec, reason))
        out: list[ActionSpec] = []
        for entries in per_id.values():
            base = entries[0][0]
            if self.readonly and base.mutating:
                out.append(dataclasses.replace(base, disabled_reason=READONLY_REASON))
                continue
            if any(r is None and s.disabled_reason is None for s, r in entries):
                enabled = next(s for s, r in entries if r is None and s.disabled_reason is None)
                out.append(dataclasses.replace(enabled, disabled_reason=None))
                continue
            host_state = next((r for _, r in entries if r and not r.startswith("data is")), None)
            stale = next((r for _, r in entries if r and r.startswith("data is")), None)
            backend = next((s.disabled_reason for s, _ in entries if s.disabled_reason), None)
            out.append(dataclasses.replace(base, disabled_reason=host_state or stale or backend or "not available"))
        return out

    # ------------------------------------------------------------ planning (pure)
    def plan(self, action_id: str, keys: Sequence[JobKey],
             params: Mapping[str, object] | None = None) -> list[ActionPlan]:
        plans: list[ActionPlan] = []
        reasons: list[str] = []
        for host, jobs in self._by_host(keys).items():
            reason = self.host_reason(host)
            sched, ctx = self._scheduler_for(host), self._context_for(host)
            if reason is not None or sched is None or ctx is None:
                reasons.append(reason or f"{host}: not detected yet")
                continue
            try:
                p = sched.plan_action(action_id, jobs, dict(params or {}), ctx)
            except ActionError as e:
                reasons.append(f"{host}: {e}")
                continue
            if self.readonly and p.spec.mutating:
                raise ActionError(READONLY_REASON)
            plans.append(p)
        if not plans:
            raise ActionError(reasons[0] if reasons else "nothing eligible")
        return plans

    # ------------------------------------------------------------ execution (never raises)
    async def execute(self, plans: Sequence[ActionPlan]) -> list[ActionResult]:
        batches = await asyncio.gather(*(self._execute_one(p) for p in plans))
        results = [r for batch in batches for r in batch]
        for host in dict.fromkeys(p.host for p in plans):
            sched = self._scheduler_for(host)
            groups = WAKE_GROUPS.get(sched.name if sched is not None else "", frozenset())
            self._wake(host, groups)
        return results

    async def _execute_one(self, plan: ActionPlan) -> list[ActionResult]:
        started = self.clock.now()
        t0 = self.clock.monotonic()

        def fail(message: str, stderr: str = "") -> list[ActionResult]:
            return [ActionResult(action_id=plan.spec.id, target=t, ok=False, message=message, stderr=stderr,
                                 started_at=started, duration_s=self.clock.monotonic() - t0) for t in plan.targets]

        if self.readonly and plan.spec.mutating:
            results = fail(READONLY_REASON)
        else:
            try:
                res = await self._probe(plan.host, framing.make_script(plan.sections, "action"),
                                        recheck=lambda: self._still_allowed(plan))
                if res.error is not None:
                    results = fail(res.error.summary, res.error.detail)
                else:
                    sched, ctx = self._scheduler_for(plan.host), self._context_for(plan.host)
                    if sched is None or ctx is None:
                        results = fail("host not detected")
                    else:
                        results = list(sched.interpret_action(plan, res.sections, ctx))
            except Exception as e:          # §9 row 18: report, never raise, never retry
                log.exception("%s: action %s failed", plan.host, plan.spec.id)
                results = fail(f"internal error: {e}"[:80])
        dur = self.clock.monotonic() - t0
        results = [r if r.started_at is not None else dataclasses.replace(r, started_at=started, duration_s=dur)
                   for r in results]
        for r in results:
            self._log.append(r)
            log.info("action %s %s on %s: %s %s", r.action_id, r.target, plan.host, "ok" if r.ok else "FAILED",
                     r.message)
        return results

    def action_log(self) -> Sequence[ActionResult]:
        return tuple(self._log)
