"""[WP3] Host error screen: classified error, section statuses, ssh log.

Also the other connection-related modals: ``DoctorScreen`` (palette › Doctor <host>) and ``ManualCommandScreen``
(shown when the terminal cannot be suspended: the exact command to run elsewhere, then "Done" re-prepares the host).
"""

from __future__ import annotations

import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from runfleet.core.store import HostState
from runfleet.models import ConnState, DoctorReport, ErrorKind
from runfleet.ui import keymap, render, theme
from runfleet.ui.widgets.host_card import busy_note
from runfleet.util import humanize, platform

LOG_TAIL = 20

_ADVICE: dict[ErrorKind, str] = {
    ErrorKind.AUTH: "Background probes cannot prompt. Press c (Connect) to log in once in this terminal "
                    "(Duo / password / passphrase prompts appear there); polling resumes afterwards.",
    ErrorKind.HOSTKEY: "The host key is unknown or changed. If it changed legitimately, remove the old key with "
                       "`ssh-keygen -R <host>`; for an unknown key, Connect lets ssh ask you.",
    ErrorKind.DNS: "The host name does not resolve. Check ~/.ssh/config (HostName) and your network / VPN.",
    ErrorKind.TIMEOUT: "The host or its jump host did not answer in time. RunFleet keeps retrying with backoff.",
    ErrorKind.REFUSED: "The connection was refused (sshd down or a firewall). RunFleet keeps retrying.",
    ErrorKind.JUMP: "The jump host (ProxyJump) failed. Check that you can reach it on its own.",
    ErrorKind.MUX: "The shared ssh master is unresponsive. Connect restarts it.",
    ErrorKind.REMOTE: "The remote shell could not run the probe (try remote_shell = \"login\" for this host).",
}


# P1: no shared connection on Windows, so a login done with Connect does not carry over to the background probes
WINDOWS_AUTH = ("Windows ssh cannot keep a login open: key/agent hosts work, but a Duo or password host stays here "
                "after Connect. For those hosts run RunFleet in WSL until the session transport.")


def read_tail(path: str | None, n: int = LOG_TAIL) -> list[str]:
    if not path:
        return []
    try:
        return Path(path).expanduser().read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def error_report(state: HostState, now: datetime, width: int) -> Text:
    st = state.status
    lines: list[Text] = []
    head = render.conn_text(st.state)
    if st.last_ok is not None:
        head.append(f" · last ok {humanize.ago(st.last_ok, now)}", style=theme.muted())
    if st.consecutive_failures:
        head.append(f" · {st.consecutive_failures} consecutive failure{'s' if st.consecutive_failures > 1 else ''}",
                    style=theme.muted())
    if st.next_poll_at is not None and st.state not in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY,
                                                        ConnState.DISABLED):
        head.append(f" · next retry in {humanize.age(max(0.0, (st.next_poll_at - now).total_seconds()))}",
                    style=theme.muted())
    lines.append(head)
    busy = busy_note(st)
    if busy:
        lines.append(Text(f"{busy}: the Slurm controller answers slowly or with timeouts; RunFleet polls it less "
                          "often until it recovers", style=theme.warn()))
    err = st.error
    if err is not None:
        lines.append(Text(""))
        k = Text(f"{err.kind.value.upper()}  ", style=theme.conn_style(st.state))
        k.append(err.summary, style="bold")
        lines.append(k)
        if err.section:
            lines.append(Text(f"section {err.section}", style=theme.muted()))
        advice = _ADVICE.get(err.kind)
        if advice:
            lines.append(Text(advice.replace("<host>", state.config.ssh), style=theme.muted()))
        if err.kind is ErrorKind.AUTH and platform.IS_WINDOWS:
            lines.append(Text(WINDOWS_AUTH, style=theme.warn()))
        if err.detail:
            lines.append(Text(""))
            lines.append(Text("stderr", style="bold"))
            for line in err.detail.splitlines()[-12:]:
                lines.append(Text("  " + line, style=theme.warn()))
    else:
        lines.append(Text("no transport error recorded", style=theme.dim()))
    bad = [s for s in st.sections.values() if not s.ok]
    if bad:
        lines.append(Text(""))
        lines.append(Text("failing sections", style="bold"))
        for s in bad:
            row = Text(f"  {s.name:<18} {s.outcome:<10}", style=theme.warn())
            if s.error is not None:
                row.append(s.error.summary)
            lines.append(row)
    tail = read_tail(st.ssh_log)
    if tail:
        lines.append(Text(""))
        lines.append(Text(f"ssh log · {st.ssh_log}", style="bold"))
        lines.extend(Text("  " + line, style=theme.muted()) for line in tail)
    lines.append(Text(""))
    lines.append(Text(f"ssh {state.config.ssh} · mux {st.mux}", style=theme.dim()))
    return Text("\n").join(render.fit(x, width) if x.cell_len > width else x for x in lines)


class HostErrorScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("host_error")
    HELP_CONTEXT = "host_error"
    WANTS_FLEET_CHANGES = True             # the state, countdown and log follow the host while the dialog is open
    DEFAULT_CSS = """
    HostErrorScreen VerticalScroll { height: auto; max-height: 1fr; }
    HostErrorScreen .dialog-buttons { dock: bottom; }
    """

    def __init__(self, host: str) -> None:
        super().__init__()
        self.host = host

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        with Vertical(id="err-box", classes="dialog"):
            with VerticalScroll():
                yield Static("", id="err-body")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Connect (c)", id="connect", variant="primary", compact=True)
                yield Button("Retry now (r)", id="retry", compact=True)
                yield Button("Disable host (t)", id="disable", compact=True)
                yield Button("Close (esc)", id="close", compact=True)

    def on_mount(self) -> None:
        if not self.query("#err-box"):         # dismissed / app exiting before Mount ran (queued keys)
            return
        state = self._redraw()
        if state is None:
            return
        self.query_one("#connect" if state.status.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY)
                       else "#close", Button).focus()
        if getattr(self.app, "animations", False):
            self.set_interval(1.0, self._redraw)      # the retry countdown

    def _redraw(self) -> HostState | None:
        state = self.rf.store.hosts.get(self.host)
        if state is None or not self.query("#err-box"):
            return None
        box = self.query_one("#err-box")
        box.border_title = f"{self.host} · {theme.CONN_LABEL[state.status.state]}"
        box.border_subtitle = "c connect · r retry · t disable · esc close"
        disabled = state.status.state is ConnState.DISABLED
        self.query_one("#disable", Button).label = "Enable host (t)" if disabled else "Disable host (t)"
        self.query_one("#err-body", Static).update(error_report(state, self.rf.now(), 92))
        if state.status.state in (ConnState.OK, ConnState.DEGRADED) and self.focused is self.query_one("#connect"):
            self.query_one("#close", Button).focus()  # recovered meanwhile: nothing left to log in to
        return state

    def on_fleet_changed(self, message: Any) -> None:
        if self.host in message.hosts or "*" in message.hosts:
            self._redraw()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {"connect": self.action_connect, "retry": self.action_retry, "disable": self.action_disable,
         "close": self.action_close}[event.button.id or "close"]()

    def action_connect(self) -> None:
        self.dismiss(None)
        self.rf.connect(self.host)

    def action_retry(self) -> None:
        self.rf.service.refresh([self.host])
        self.rf.toast(f"retrying {self.host}")
        self.dismiss(None)

    def action_disable(self) -> None:
        self.dismiss(None)
        self.rf.action_toggle_host(self.host)

    def action_close(self) -> None:
        self.dismiss(None)


def doctor_text(report: DoctorReport) -> Text:
    lines = []
    for check in report.checks:
        mark, style = ("✓", theme.ok()) if check.ok else ("i", theme.muted()) if check.ok is None \
            else ("✖", theme.error())
        line = Text(f"{mark} ", style=style)
        line.append(f"{check.name:<10}", style="bold")
        line.append(check.detail)
        lines.append(line)
    for w in report.warnings:
        lines.append(Text(f"! {w}", style=theme.warn()))
    return Text("\n").join(lines) if lines else Text("no checks", style=theme.dim())


class DoctorScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("modal")
    HELP_CONTEXT = "modal"
    DEFAULT_CSS = """
    DoctorScreen VerticalScroll { height: auto; max-height: 1fr; }
    """

    def __init__(self, report: DoctorReport) -> None:
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="doctor-box", classes="dialog"), VerticalScroll():
            yield Static(doctor_text(self.report))

    def on_mount(self) -> None:
        if not self.query("#doctor-box"):
            return
        box = self.query_one("#doctor-box")
        box.border_title = f"Doctor · {self.report.host}"
        box.border_subtitle = "esc close"

    def action_close(self) -> None:
        self.dismiss(None)


class ManualCommandScreen(ModalScreen[None]):
    """The terminal cannot be suspended: show the exact command; for a login (``host`` given) "Done" tells the core
    the user connected to that host."""

    BINDINGS = keymap.bindings("modal")
    HELP_CONTEXT = "modal"
    DEFAULT_CSS = """
    """

    def __init__(self, argv: list[str], host: str | None) -> None:
        super().__init__()
        self.argv = argv
        self.host = host

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        with Vertical(id="manual-box", classes="dialog"):
            yield Static(Text.assemble(
                ("This terminal cannot be suspended. Run this in another terminal:\n\n", theme.muted()),
                ("  " + shlex.join(self.argv), "bold"),
                ("\n\nthen press Done." if self.host is not None else "", theme.muted()),
            ))
            with Horizontal(classes="dialog-buttons"):
                if self.host is not None:      # only a login has something to re-prepare afterwards
                    yield Button("Done", id="done", variant="primary", compact=True)
                yield Button("Close (esc)", id="close", compact=True)

    def on_mount(self) -> None:
        if not self.query("#manual-box"):
            return
        self.query_one("#manual-box").border_title = "Run manually"
        self.query_one("#done" if self.host is not None else "#close", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done" and self.host is not None:
            self.rf.run_worker(self.rf.service.after_connect(self.host, 0), exit_on_error=False)
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
