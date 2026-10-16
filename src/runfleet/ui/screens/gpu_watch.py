"""Palette › Watch free GPUs… (the prompt) and palette › Watches (the list), §7.14 (``ui/gpu_watch.py``).

:class:`WatchPromptScreen` is one line in the pattern of the port prompt (keymap context ``prompt``): it reads the
text leniently as the user types (``2 h200 k2``), says how it was read, what matches right now (per node, only fresh
readings, planned GPUs apart) and why it cannot be saved, and dismisses with the watch on ``enter`` (None on
``esc``). :class:`WatchesScreen` lists the watches with what each is doing (waiting, holding 1 of 2 readings, fired,
quiet until …, paused with polling) and removes one (``d``); ``n`` opens the prompt. Saving and evaluating are
:class:`~runfleet.ui.notify.Alerts`' job.
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from runfleet.ui import gpu_watch, keymap, theme
from runfleet.ui.gpu_watch import GpuWatch, Parsed
from runfleet.ui.messages import FleetChanged
from runfleet.ui.screens import dialog_list_room

EXAMPLES = "e.g. 2 h200 k2 · 4x a100 k2/gpu-preempt · 1 mig vega · add planned to count backfill-planned GPUs"
RULE = (f"per node, fresh readings only · fires after {gpu_watch.CONSECUTIVE} readings in a row, then rests "
        f"≥ {int(gpu_watch.COOLDOWN_S // 60)} min")


def _level_style(level: gpu_watch.Level) -> str:
    return {"ok": theme.ok(), "warn": theme.warn(), "accent": theme.accent()}.get(level, theme.dim())


class WatchPromptScreen(ModalScreen[GpuWatch | None]):
    """Ask what to watch for; ``↵`` dismisses with the watch once it is usable, ``esc`` with None."""

    BINDINGS = keymap.bindings("prompt")
    HELP_CONTEXT = "prompt"
    DEFAULT_CSS = """
    WatchPromptScreen #gw-prompt-box { width: 96; }
    WatchPromptScreen Input { margin: 1 0 0 0; }
    """

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.text = text

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="gw-prompt-box", classes="dialog")
        box.border_title = "Watch free GPUs…"
        box.border_subtitle = "↵ watch · esc cancel"
        with box:
            yield Static(Text("how many GPUs of which type on one node, and where (host, partition)",
                              style=theme.muted()), id="gw-help")
            yield Input(self.text, placeholder=EXAMPLES, id="gw-text", compact=True, max_length=120)
            yield Static("", id="gw-what")
            yield Static("", id="gw-now")
            yield Static("", id="gw-note")

    def on_mount(self) -> None:
        if self.query("#gw-text"):
            self.query_one(Input).focus()
            self.show(self.text, submitted=False)

    def forget_rendered(self) -> None:
        """Live theme switch: every line carries theme colours."""
        if self.query("#gw-text"):
            self.query_one("#gw-help", Static).update(Text("how many GPUs of which type on one node, and where "
                                                           "(host, partition)", style=theme.muted()))
            self.show(self.query_one(Input).value, submitted=False)

    def show(self, text: str, *, submitted: bool) -> tuple[Parsed, str | None]:
        """Read ``text`` and show how; returns the parse and why it cannot be saved (None: it can)."""
        hosts = self.rf.store.hosts
        parsed = gpu_watch.parse(text, gpu_watch.fleet_of(hosts))
        watches = self.rf.alerts.gpu_watches
        problem = parsed.problem
        if problem is None and parsed.watch is not None:
            problem = gpu_watch.add_problem(watches, parsed.watch)
        what = self.query_one("#gw-what", Static)
        now = self.query_one("#gw-now", Static)
        note = self.query_one("#gw-note", Static)
        if parsed.watch is None:
            what.update(Text(EXAMPLES, style=theme.dim()))
            now.update("")
        else:
            line = Text("watch for ", style=theme.muted())
            line.append(parsed.watch.describe(), style="bold")
            line.append(f"\n{RULE}", style=theme.dim())
            what.update(line)
            readings = self.rf.alerts.gpu_readings()
            scope = gpu_watch.in_scope(parsed.watch, readings)
            out = Text(gpu_watch.now_text(parsed.watch, readings), style=theme.muted())
            unknown = [h for h in scope if readings[h].at is None]
            if unknown:
                out.append(f"\nnot counted until they report again: {' '.join(f'{h} ?' for h in unknown)}",
                           style=theme.warn())
            now.update(out)
        if problem is not None and (text.strip() or submitted):
            note.update(Text(problem, style=theme.warn()))
        elif parsed.notes:
            note.update(Text(" · ".join(parsed.notes), style=theme.dim()))
        else:
            note.update(Text(f"{len(watches)} of {gpu_watch.MAX_WATCHES} watches · palette › Watches lists them",
                             style=theme.dim()))
        return parsed, problem

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.show(event.value, submitted=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        parsed, problem = self.show(event.value, submitted=True)
        if problem is None and parsed.watch is not None:
            self.dismiss(parsed.watch)

    def action_close(self) -> None:
        self.dismiss(None)


def watch_text(watch: GpuWatch, status: tuple[str, gpu_watch.Level]) -> Text:
    out = Text(watch.describe(), style="bold")
    out.append("\n  ")
    out.append(status[0], style=_level_style(status[1]))
    return out


BOX_W = 100                      # = #gw-box width
LIST_ROWS = 20                   # the Watches list at most (10 watches of 2 lines; less on a short terminal)


class WatchesScreen(ModalScreen[None]):
    """The GPU watches and what each is doing; ``d`` removes one, ``n`` adds one. Follows the store live."""

    BINDINGS = keymap.bindings("watches")
    HELP_CONTEXT = "watches"
    WANTS_FLEET_CHANGES = True
    REFRESH_S = 5.0              # "quiet until …" and the cooldown move without a store change
    DEFAULT_CSS = f"""
    WatchesScreen #gw-box {{ width: {BOX_W}; }}
    WatchesScreen OptionList, WatchesScreen OptionList:focus {{
        border: none; height: auto; max-height: {LIST_ROWS}; background: transparent; padding: 0;
    }}
    """

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="gw-box", classes="dialog")
        box.border_title = "GPU watches"
        box.border_subtitle = "n new · d remove · esc close"
        with box:
            yield OptionList(id="gw-list")
            yield Static("", id="gw-foot")

    def on_mount(self) -> None:
        if self.query("#gw-list"):
            self.refill()
            self.query_one(OptionList).focus()
            self.set_interval(self.REFRESH_S, self.refill)

    def on_screen_resume(self) -> None:
        if self.is_mounted and self.query("#gw-list"):
            self.refill()              # back from the prompt: a new watch

    def on_fleet_changed(self, _message: FleetChanged) -> None:
        self.refill()

    def forget_rendered(self) -> None:
        if self.query("#gw-list"):
            self.refill()

    def refill(self) -> None:
        if not self.query("#gw-list"):
            return
        alerts = self.rf.alerts
        watches = list(alerts.gpu_watches)
        menu = self.query_one(OptionList)
        keep = menu.highlighted
        menu.clear_options()
        menu.add_options([Option(watch_text(w, alerts.gpu_watch_status(w)), id=str(i))
                          for i, w in enumerate(watches)])
        if watches:
            menu.highlighted = min(keep or 0, len(watches) - 1)
        self.query_one("#gw-box").border_title = f"GPU watches · {len(watches)} of {gpu_watch.MAX_WATCHES}"
        foot = ("no watches yet — n (or palette › Watch free GPUs…) adds one, e.g. 2 h200 k2" if not watches
                else f"a watch notifies like a job alert: toast, terminal, [notify] command\n{RULE}")
        self.query_one("#gw-foot", Static).update(Text(foot, style=theme.dim()))
        self._fit_list(foot)

    def on_resize(self, _event: events.Resize) -> None:
        if self.query("#gw-list"):
            self.refill()

    def _fit_list(self, foot: str) -> None:
        """The list never taller than the dialog leaves it next to the foot (its wrapped lines), so the highlighted
        watch is never clipped below the box on a short terminal."""
        inner = max(20, min(BOX_W, self.app.size.width * 96 // 100) - 4)
        reserved = sum(max(1, -(-cell_len(line) // inner)) for line in foot.split("\n"))
        self.query_one(OptionList).styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS, reserved)

    def current(self) -> GpuWatch | None:
        menu = self.query_one(OptionList)
        watches = self.rf.alerts.gpu_watches
        if menu.highlighted is None or not 0 <= menu.highlighted < len(watches):
            return None
        return watches[menu.highlighted]

    def action_new_watch(self) -> None:
        self.rf.prompt_gpu_watch()

    def action_remove_watch(self) -> None:
        watch = self.current()
        if watch is not None and self.rf.alerts.remove_gpu_watch(watch):
            self.rf.toast(f"stopped watching for {watch.describe()}")
            self.refill()

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)
