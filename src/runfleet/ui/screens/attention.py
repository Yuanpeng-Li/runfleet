"""The "needs you" picker of the Fleet view (``a``, §7.4): what asks for the user across the fleet, in rank order.

The entries are the cards' own rankings (:func:`runfleet.ui.fleet_layout.attention`): failures of the last 4 h, running
jobs with a warning or low on time, pending jobs that will never start or are held, older failures in the recent
window. Each row says host, job id, full name and why; ``enter`` closes the dialog and opens the job, ``esc`` closes.
It reads the store only (nothing is polled to fill it) and works whatever the Fleet layout (roomy, compact, table).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from runfleet.models import JobKey
from runfleet.ui import keymap, render, theme
from runfleet.ui.fleet_layout import Attention, array_suffix, attention_why
from runfleet.ui.screens import dialog_list_room
from runfleet.util import humanize

BOX_W = 112                      # = #attention-box width
LIST_ROWS = 24                   # = #attention-list max-height (less on a short terminal, dialog_list_room)
HOST_W = 12                      # host column at most
ID_W = 18                        # job id column at most (array ids: 31377076_[5-12%2])


def entry_text(entry: Attention, now: datetime, width: int, *, host_w: int, id_w: int, host_style: str = "",
               util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> Text:
    """``k2    2334990  F   orb-eval--cifar--s43  exit 1 · 12m ago``: one line when the full name and the reason fit
    in ``width``, else the reason on a second line under the name (the name is cut only when it alone is wider)."""
    job = entry.item.job
    head = Text()
    head.append(humanize.truncate(entry.host, host_w).ljust(host_w), style=f"bold {host_style}".strip())
    head.append("  ")
    head.append(humanize.truncate(job.key.label, id_w).ljust(id_w), style=theme.muted())
    head.append("  ")
    badge = render.state_badge(job.state, job.held)
    head.append_text(badge)
    head.append(" " * max(1, 4 - badge.cell_len))
    indent = head.cell_len
    name = job.name + array_suffix(job)
    why = attention_why(entry, now, util_t)
    if indent + cell_len(name) + 2 + why.cell_len <= width:
        head.append(name)
        head.append("  ")
        head.append_text(why)
        return render.fit(head, width)
    head.append(humanize.truncate(name, max(8, width - indent)))
    second = Text(" " * indent)
    second.append_text(why)
    return Text("\n").join([render.fit(head, width), render.fit(second, width)])


def column_widths(entries: Sequence[Attention]) -> tuple[int, int]:
    """Host and id column widths for these entries (the longest, capped)."""
    host_w = min(HOST_W, max((cell_len(e.host) for e in entries), default=4))
    id_w = min(ID_W, max((cell_len(e.item.job.key.label) for e in entries), default=6))
    return host_w, id_w


class AttentionScreen(ModalScreen[JobKey | None]):
    """What needs the user across the fleet; dismisses with the job to open (None: closed)."""

    BINDINGS = keymap.bindings("attention")
    HELP_CONTEXT = "attention"
    DEFAULT_CSS = f"""
    #attention-box {{ width: {BOX_W}; }}
    #attention-list, #attention-list:focus {{
        border: none; background: transparent; padding: 0; height: auto; max-height: {LIST_ROWS};
    }}
    """

    def __init__(self, entries: Sequence[Attention], now: datetime, host_colors: Mapping[str, str] | None = None,
                 *, util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> None:
        super().__init__()
        self.entries = list(entries)
        self.now = now
        self.host_colors = dict(host_colors or {})
        self.util_t = util_t

    def text_width(self) -> int:
        """Cells for one row: the box (at most 96 % of the terminal) less border, padding and the scrollbar."""
        term = self.app.size.width or 120
        return max(40, min(BOX_W, term * 96 // 100) - 4 - 2)

    def compose(self) -> ComposeResult:
        box = Vertical(id="attention-box", classes="dialog")
        n = len(self.entries)
        box.border_title = f"Needs you · {n} across the fleet" if n else "Needs you"
        box.border_subtitle = "↵ open job · esc close"
        with box:
            if self.entries:
                width = self.text_width()
                host_w, id_w = column_widths(self.entries)
                yield OptionList(*[Option(entry_text(e, self.now, width, host_w=host_w, id_w=id_w,
                                                     host_style=self.host_colors.get(e.host, ""), util_t=self.util_t),
                                          id=str(i))
                                   for i, e in enumerate(self.entries)], id="attention-list")
            else:
                yield Static(Text("nothing needs you: no recent failures, nothing running low on time or with a "
                                  "warning, nothing pending that will never start", style=theme.dim()))

    def on_mount(self) -> None:
        found = self.query(OptionList)
        if found:
            self._fit_list()
            found.first().focus()

    def on_resize(self, _event: events.Resize) -> None:
        self._fit_list()

    def _fit_list(self) -> None:
        for menu in self.query(OptionList):
            menu.styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.entries[int(event.option.id or 0)].key)

    def action_cursor_down(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_down()

    def action_cursor_up(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)
