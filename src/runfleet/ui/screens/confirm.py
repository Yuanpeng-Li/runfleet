"""[WP3] Confirm dialog for actions, with live re-plan on param change (mockups J/K).

Shows, per host plan: the targets (<= 8, then "+N more"), skipped targets with their reasons, the backend's
consequences, the exact remote command preview, and the action's params (checkbox / number). Changing a param calls
the pure ``FleetAPI.plan`` again and replaces the preview. *Cancel* has the default focus; ``y`` confirms, ``n`` /
``esc`` cancels. Dismisses with the plans to execute, or ``None``.

:class:`AttachConfirmScreen` is the three-choice srun confirmation of Attach (§7.16.2): *Once* / *Allow for this job
(this session)* / *Cancel* — never a "don't ask again". It dismisses with ``"once"``, ``"job"`` or ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static

from runfleet.backends.base import ActionParam, ActionPlan
from runfleet.errors import ActionError
from runfleet.models import AttachOffer, Job, JobKey
from runfleet.ui import keymap, render, theme
from runfleet.ui.query import elapsed_s
from runfleet.util import humanize

MAX_TARGETS = 8


def target_line(job: Job | None, key: JobKey, now: Any, width: int) -> Text:
    out = Text(f"{key.label:<10}", style="bold")
    if job is None:
        out.append("(no longer known)", style=theme.dim())
        return out
    name_w = max(10, width - 30)
    out.append(humanize.truncate(job.name, name_w, keep_tail=True).ljust(name_w))
    out.append(" ")
    out.append_text(render.state_badge(job.state, job.held).copy())
    out.append(f"  {humanize.clock(elapsed_s(job, now))}", style=theme.muted())
    return out


def plan_body(plan: ActionPlan, jobs: Mapping[JobKey, Job | None], now: Any, width: int,
              multi_host: bool) -> Text:
    lines: list[Text] = []
    if multi_host:
        lines.append(Text(f"{plan.host} · {len(plan.targets)} target{'s' if len(plan.targets) != 1 else ''}",
                          style=f"bold {theme.ACCENT}"))
    for key in plan.targets[:MAX_TARGETS]:
        lines.append(target_line(jobs.get(key), key, now, width))
    if len(plan.targets) > MAX_TARGETS:
        lines.append(Text(f"+{len(plan.targets) - MAX_TARGETS} more", style=theme.muted()))
    for key, reason in plan.skipped:
        lines.append(Text(f"skipped {key.label}: {reason}", style=theme.warn()))
    ids = {k.id for k in plan.targets} | {k.label for k in plan.targets}
    for line in plan.confirm_body.splitlines():
        if line.strip() and not any(i in line for i in ids):
            lines.append(Text(line, style=theme.muted()))
    lines.append(Text(""))
    cmd = Text(f"will run on {plan.host}:  ", style=theme.muted())
    cmd.append(plan.preview, style="bold")
    lines.append(cmd)
    return Text("\n").join(lines)


def dropped_targets(service: Any, action_id: str, keys: Sequence[JobKey],
                    plans: Sequence[ActionPlan]) -> list[tuple[JobKey, str]]:
    """Requested targets that no plan covers, with the reason: the core leaves a host out of the plans when it needs
    auth, is stale or not detected yet, and says nothing about it as long as another host has a plan."""
    covered = {k for p in plans for k in p.targets} | {k for p in plans for k, _r in p.skipped}
    by_host: dict[str, list[JobKey]] = {}
    for k in keys:
        if k not in covered:
            by_host.setdefault(k.host, []).append(k)
    out: list[tuple[JobKey, str]] = []
    for host, host_keys in by_host.items():
        try:
            spec = next((s for s in service.actions_for(host_keys) if s.id == action_id), None)
        except Exception:                      # a reason is a nicety: never fail the dialog for it
            spec = None
        reason = spec.disabled_reason if spec is not None and spec.disabled_reason else "not eligible"
        if not reason.startswith(f"{host}:"):
            reason = f"{host}: {reason}"
        for k in host_keys:
            out.append((k, reason if service.store.job(k) is not None else f"{host}: no longer known"))
    return out


def dropped_body(dropped: Sequence[tuple[JobKey, str]]) -> Text:
    """One warning line per reason: "not acted on — vega: needs auth (2 jobs)"."""
    counts: dict[str, int] = {}
    for _k, reason in dropped:
        counts[reason] = counts.get(reason, 0) + 1
    return Text("\n").join(Text(f"not acted on — {reason} ({n} job{'s' if n != 1 else ''})", style=theme.warn())
                           for reason, n in counts.items())


def dialog_title(plans: Sequence[ActionPlan]) -> str:
    if len(plans) == 1:
        return plans[0].confirm_title
    n = sum(len(p.targets) for p in plans)
    verb = plans[0].spec.label.split(" ")[0]
    return f"{verb} {n} targets on {len(plans)} hosts?"


class ConfirmScreen(ModalScreen[list[ActionPlan] | None]):
    BINDINGS = keymap.bindings("confirm")
    HELP_CONTEXT = "confirm"               # F1 inside a dialog shows its own keys (app.action_help)
    DEFAULT_CSS = """
    ConfirmScreen #confirm-scroll { height: auto; max-height: 1fr; }
    ConfirmScreen .dialog-buttons { dock: bottom; }
    ConfirmScreen #params { height: auto; margin-top: 1; }
    ConfirmScreen .param-row { height: auto; }
    ConfirmScreen Checkbox { border: none; padding: 0; background: transparent; }
    ConfirmScreen .param-row Input { width: 8; height: 1; border: none; padding: 0 1; margin: 0 1; }
    ConfirmScreen .param-row Label { padding: 0 0; }
    ConfirmScreen #confirm-error { color: $error; height: auto; }
    """

    def __init__(self, action_id: str, keys: Sequence[JobKey], plans: Sequence[ActionPlan]) -> None:
        super().__init__()
        self.action_id = action_id
        self.keys = list(keys)
        self.plans = list(plans)
        self.params: dict[str, object] = dict(plans[0].params) if plans else {}
        self.spec = plans[0].spec if plans else None
        for p in (self.spec.params if self.spec else ()):
            self.params.setdefault(p.name, p.default)

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box", classes="dialog"):
            with VerticalScroll(id="confirm-scroll"):      # many targets: the body scrolls, the buttons stay
                yield Static("", id="confirm-body")
                with Vertical(id="params"):
                    yield from self._param_widgets(self.spec.params if self.spec else ())
                yield Static("", id="confirm-error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel  (n/esc)", id="cancel", variant="default", compact=True)
                yield Button("Confirm  (y)", id="confirm", variant="error", compact=True)

    def _param_widgets(self, params: Sequence[ActionParam]) -> Any:
        bools = [p for p in params if p.kind == "bool"]
        ints = [p for p in params if p.kind == "int"]
        if bools and ints and len(bools) == 1 and len(ints) == 1:
            # "[x] send SIGKILL to survivors after [10] s" (the int qualifies the bool)
            with Horizontal(classes="param-row"):
                yield Checkbox(bools[0].label, bool(self.params.get(bools[0].name)), id=f"p-{bools[0].name}")
                yield Input(str(self.params.get(ints[0].name)), type="integer", id=f"p-{ints[0].name}",
                            compact=True)
                yield Label(ints[0].label if ints[0].label != "seconds" else "s")
            return
        for p in params:
            with Horizontal(classes="param-row"):
                if p.kind == "bool":
                    yield Checkbox(p.label, bool(self.params.get(p.name)), id=f"p-{p.name}")
                else:
                    yield Label(p.label)
                    yield Input(str(self.params.get(p.name)), type="integer", id=f"p-{p.name}", compact=True)

    def on_mount(self) -> None:
        if not self.query("#confirm-box"):     # dismissed / app exiting before Mount ran (queued keys)
            return
        self.query_one("#params").display = bool(self.spec and self.spec.params)
        self._refresh_body()
        self.query_one("#cancel", Button).focus()

    def _refresh_body(self, error: str | None = None) -> None:
        if not self.query("#confirm-box"):
            return
        box = self.query_one("#confirm-box")
        box.border_title = dialog_title(self.plans) if self.plans else "Nothing to do"
        now = self.rf.now()
        jobs = {k: self.rf.store.job(k) for p in self.plans for k in p.targets}
        width = 70
        bodies = [plan_body(p, jobs, now, width, len(self.plans) > 1) for p in self.plans]
        dropped = dropped_targets(self.rf.service, self.action_id, self.keys, self.plans) if self.plans else []
        if dropped:
            bodies.append(dropped_body(dropped))
        self.query_one("#confirm-body", Static).update(Text("\n\n").join(bodies))
        err = self.query_one("#confirm-error", Static)
        err.update(error or "")
        err.display = bool(error)
        self.query_one("#confirm", Button).disabled = bool(error) or not self.plans

    def _replan(self) -> None:
        try:
            self.plans = self.rf.service.plan(self.action_id, self.keys, dict(self.params))
        except ActionError as exc:
            self._refresh_body(str(exc))
            return
        self._refresh_body()

    def _param(self, widget_id: str | None) -> ActionParam | None:
        if not widget_id or not widget_id.startswith("p-") or self.spec is None:
            return None
        name = widget_id[2:]
        return next((p for p in self.spec.params if p.name == name), None)

    def _read_numbers(self) -> str | None:
        """Every number field parsed and range-checked into ``params``; the first problem, or None. Checked as a
        whole each time, so changing another param never replans past a field that still holds an invalid value."""
        error: str | None = None
        for param in (self.spec.params if self.spec else ()):
            fields = self.query(f"Input#p-{param.name}")
            if param.kind != "int" or not fields:
                continue
            try:
                value = int(fields.first(Input).value)
            except ValueError:
                error = error or f"{param.label}: enter a whole number"
                continue
            if (param.min is not None and value < param.min) or (param.max is not None and value > param.max):
                error = error or f"{param.label}: must be between {param.min} and {param.max}"
                continue
            self.params[param.name] = value
        return error

    def _params_changed(self) -> None:
        error = self._read_numbers()
        if error:
            self._refresh_body(error)
            return
        self._replan()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        param = self._param(event.checkbox.id)
        if param is not None:
            self.params[param.name] = event.value
            self._params_changed()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._param(event.input.id) is not None:
            self._params_changed()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.action_confirm()
        else:
            self.action_cancel()

    def action_confirm(self) -> None:
        error = self._read_numbers()
        if error:
            self._refresh_body(error)
            return
        if self.plans and not self.query_one("#confirm", Button).disabled:
            self.dismiss(self.plans)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------- Attach: srun confirmation (§7.16.2)
AttachConsent = Literal["once", "job"]


def attach_offer_body(offer: AttachOffer, host: str, srun_rows: int, width: int = 70) -> Text:
    """Why srun (the offer's reason), what it costs (the backend's text), how many rows this session already added,
    and the exact command it runs on the login host."""
    lines = [Text(offer.reason, style=theme.warn()), Text("")]
    lines.append(Text(offer.body))
    if srun_rows:
        lines.append(Text(f"{srun_rows} sacct row{'s' if srun_rows != 1 else ''} added for this job this session",
                          style=theme.muted()))
    lines.append(Text(""))
    cmd = Text(f"will run on {host} (node {offer.node}):  ", style=theme.muted())
    cmd.append(offer.preview, style="bold")
    lines.append(cmd)
    lines.append(Text(""))
    lines.append(Text("Once: this read only · Allow for this job: no more questions for this job until RunFleet "
                      "exits (every read still adds one row)", style=theme.dim()))
    return Text("\n").join(lines)


class AttachConfirmScreen(ModalScreen[AttachConsent | None]):
    BINDINGS = keymap.bindings("attach_confirm")
    HELP_CONTEXT = "attach_confirm"
    DEFAULT_CSS = """
    AttachConfirmScreen #attach-confirm-box { width: 84; }
    AttachConfirmScreen #confirm-scroll { height: auto; max-height: 1fr; }
    AttachConfirmScreen .dialog-buttons { dock: bottom; }
    """

    def __init__(self, host: str, offer: AttachOffer, srun_rows: int = 0) -> None:
        super().__init__()
        self.host = host
        self.offer = offer
        self.srun_rows = srun_rows

    def compose(self) -> ComposeResult:
        with Vertical(id="attach-confirm-box", classes="dialog"):
            with VerticalScroll(id="confirm-scroll"):
                yield Static(attach_offer_body(self.offer, self.host, self.srun_rows), id="confirm-body")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel  (n/esc)", id="cancel", variant="default", compact=True)
                yield Button("Allow for this job (this session)  (a)", id="job", variant="warning", compact=True)
                yield Button("Once  (y)", id="once", variant="primary", compact=True)

    def on_mount(self) -> None:
        if not self.query("#attach-confirm-box"):
            return
        self.query_one("#attach-confirm-box").border_title = self.offer.title
        self.query_one("#cancel", Button).focus()              # like every confirm: the safe choice has the focus

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "once":
            self.action_once()
        elif event.button.id == "job":
            self.action_allow_job()
        else:
            self.action_cancel()

    def action_once(self) -> None:
        self.dismiss("once")

    def action_allow_job(self) -> None:
        self.dismiss("job")

    def action_cancel(self) -> None:
        self.dismiss(None)
