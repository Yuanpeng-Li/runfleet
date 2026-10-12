"""[WP3] Port forwards: the list of active forwards (palette › Forwards) and the "Other port…" prompt.

``ForwardsScreen`` lists ``FleetAPI.forwards()``: ``↵`` copies the local URL again, ``^k`` cancels the forward
(``ssh -O cancel``, local only). ``PortPromptScreen`` asks for a port number and dismisses with it (None on esc).
Every forward is cancelled when RunFleet exits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from runfleet.core.forwards import PortForward
from runfleet.errors import RunFleetError
from runfleet.ui import keymap, theme
from runfleet.util import humanize


def forward_text(fwd: PortForward, now: datetime) -> Text:
    out = Text(f"localhost:{fwd.local_port}", style="bold")
    out.append(f" → {fwd.host} {fwd.node}:{fwd.remote_port}")
    if fwd.job is not None:
        out.append(f" · {fwd.job.label}", style=theme.muted())
    if fwd.label:
        out.append(f" · {fwd.label}", style=theme.muted())
    if fwd.started_at is not None:
        out.append(f" · {humanize.ago(fwd.started_at, now)}", style=theme.dim())
    out.append(f"\n  {fwd.url}", style=theme.accent())
    return out


class ForwardsScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("forwards")
    HELP_CONTEXT = "forwards"
    DEFAULT_CSS = """
    ForwardsScreen #fwd-box { width: 90; }
    ForwardsScreen OptionList { border: none; height: auto; max-height: 16; background: transparent; padding: 0; }
    """

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="fwd-box", classes="dialog")
        box.border_title = "Port forwards"
        box.border_subtitle = "↵ copy URL · ^k cancel · esc close"
        with box:
            yield OptionList(id="fwd-list")
            yield Static("", id="fwd-note")

    def on_mount(self) -> None:
        if self.query("#fwd-list"):
            self.refill()
            self.query_one(OptionList).focus()

    def refill(self) -> None:
        forwards = sorted(self.rf.service.forwards(), key=lambda f: f.local_port)
        menu = self.query_one(OptionList)
        keep = menu.highlighted
        menu.clear_options()
        now = self.rf.now()
        menu.add_options([Option(forward_text(f, now), id=str(f.local_port)) for f in forwards])
        if forwards:
            menu.highlighted = min(keep or 0, len(forwards) - 1)
        note = ("no active forwards — P on a running job (or palette › Forward a port…) starts one" if not forwards
                else "local only (ssh -O forward on the existing master) · cancelled when RunFleet exits")
        self.query_one("#fwd-note", Static).update(Text(note, style=theme.dim()))

    def _current(self) -> PortForward | None:
        menu = self.query_one(OptionList)
        if menu.highlighted is None:
            return None
        opt = menu.get_option_at_index(menu.highlighted)
        return next((f for f in self.rf.service.forwards() if str(f.local_port) == opt.id), None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_copy()

    def action_copy(self) -> None:
        from runfleet.ui.everyday import copy_text

        fwd = self._current()
        if fwd is not None:
            copy_text(self.rf, fwd.url, "URL")

    def action_cancel_forward(self) -> None:
        fwd = self._current()
        if fwd is not None:
            self.cancel(fwd)

    @work(exit_on_error=False, group="forward-cancel")
    async def cancel(self, fwd: PortForward) -> None:
        try:
            await self.rf.service.cancel_forward(fwd.local_port)
        except RunFleetError as exc:
            self.rf.toast(f"could not stop forwarding localhost:{fwd.local_port}: {exc}", severity="error")
        else:
            self.rf.toast(f"stopped forwarding localhost:{fwd.local_port} → {fwd.node}:{fwd.remote_port}")
        if self.is_mounted:
            self.refill()

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)


class PortPromptScreen(ModalScreen[int | None]):
    """One line: a TCP port (1-65535). ``↵`` dismisses with it, ``esc`` with None."""

    BINDINGS = keymap.bindings("prompt")
    HELP_CONTEXT = "prompt"
    DEFAULT_CSS = """
    PortPromptScreen #prompt-box { width: 56; }
    PortPromptScreen Input { margin: 1 0 0 0; }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_text = title

    def compose(self) -> ComposeResult:
        box = Vertical(id="prompt-box", classes="dialog")
        box.border_title = self.title_text
        box.border_subtitle = "↵ forward · esc cancel"
        with box:
            yield Input(placeholder="port, e.g. 6006 (TensorBoard) or 8888 (Jupyter)", id="port", compact=True,
                        restrict=r"[0-9]*", max_length=5)
            yield Static("", id="prompt-error")

    def on_mount(self) -> None:
        if self.query("#port"):
            self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        port = int(text) if text.isdigit() else 0
        if not 1 <= port <= 65535:
            self.query_one("#prompt-error", Static).update(Text("a port is a number from 1 to 65535",
                                                                style=theme.warn()))
            return
        self.dismiss(port)

    def action_close(self) -> None:
        self.dismiss(None)
