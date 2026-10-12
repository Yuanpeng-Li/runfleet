"""[WP3] Help overlay generated from the keymap (mockup L).

Contextual: only the keys of the screen/view underneath (plus the global keys it does not override), in three
columns. Typing filters by key or description.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from runfleet.ui import keymap, theme
from runfleet.util import humanize

COLUMN_GAP = 3


def help_columns(contexts: Sequence[str], needle: str = "") -> list[tuple[str, list[tuple[str, str]]]]:
    entries = keymap.help_entries(contexts)
    needle = needle.lower().strip()
    out = []
    for group, title in keymap.GROUP_TITLES.items():
        rows = [(k, d) for k, d in entries[group] if not needle or needle in k.lower() or needle in d.lower()]
        out.append((title, rows))
    return out


def _column_widths(columns: Sequence[tuple[str, list[tuple[str, str]]]], key_ws: Sequence[int],
                   total: int) -> list[int]:
    """Widths by content: every column gets what it needs when that fits; otherwise the widest columns give up
    cells first (so a column of short entries never truncates to pay for a long one). Spare room is shared out."""
    needs = [max([len(title)] + [key_ws[i] + 1 + len(desc) for _k, desc in rows])
             for i, (title, rows) in enumerate(columns)]
    widths = list(needs)
    while sum(widths) > total and max(widths) > 12:
        widest = max(range(len(widths)), key=lambda i: widths[i])
        widths[widest] -= 1
    spare = total - sum(widths)
    if spare > 0 and widths:
        for i in range(len(widths)):
            widths[i] += spare // len(widths) + (1 if i < spare % len(widths) else 0)
    return widths


LEGEND: tuple[tuple[str, str, str], ...] = (
    ("jobs", "▸ + ! ~", "cursor row · marked · host has a problem · host data is stale"),
    ("gpus", "*  ~", "per-job aggregate from Slurm accounting (not one GPU's reading) · stale"),
    ("global", "⟳ 15s", "tick interval of the hosts in scope (spinner while a probe is in flight)"),
    ("global", "F 5/{window}", "jobs that ended within the recent window (header)"),
    ("global", "● ◌ ◐ ✖ ○", "host ok · connecting · retrying · down / needs auth · disabled (! = degraded)"),
)


def legend(contexts: Sequence[str], width: int, window: str = "24h") -> Text:
    """Glyph legend for the overlay (rows relevant to these contexts plus the global ones); ``window`` is
    general.recent_window as the header shows it (``F 5/24h``)."""
    out = Text("Legend", style=f"bold {theme.muted()}")
    for ctx, glyphs, text in LEGEND:
        if ctx in contexts or ctx == "global":
            line = Text("\n")
            line.append(glyphs.format(window=window).ljust(9), style=theme.accent())
            line.append(" " + text)
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
    return out


def render_help(contexts: Sequence[str], needle: str, width: int) -> Text:
    columns = help_columns(contexts, needle)
    key_ws = [min(9, max((len(k) for k, _d in rows), default=4)) for _t, rows in columns]
    col_ws = _column_widths(columns, key_ws, width - COLUMN_GAP * (len(columns) - 1))
    height = max((len(rows) for _t, rows in columns), default=0)
    lines = []
    head = Text()
    for i, (title, _rows) in enumerate(columns):
        cell = Text(title, style=f"bold {theme.muted()}")
        cell.truncate(col_ws[i], overflow="ellipsis")
        cell.pad_right(col_ws[i] - cell.cell_len + (COLUMN_GAP if i < 2 else 0))
        head.append_text(cell)
    lines.append(head)
    for r in range(height):
        line = Text()
        for i, (_title, rows) in enumerate(columns):
            cell = Text()
            if r < len(rows):
                key, desc = rows[r]
                cell.append(key.ljust(key_ws[i]), style=theme.accent())
                cell.append(" " + desc)
            if cell.cell_len > col_ws[i]:
                cell.truncate(col_ws[i], overflow="ellipsis")
            cell.pad_right(col_ws[i] - cell.cell_len + (COLUMN_GAP if i < 2 else 0))
            line.append_text(cell)
        lines.append(line)
    if height == 0:
        lines.append(Text(f"no key matches “{needle}”", style=theme.dim()))
    return Text("\n").join(lines)


class HelpScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("help")
    DEFAULT_CSS = """
    HelpScreen Input { border: none; height: 1; padding: 0; margin-bottom: 1; background: $boost; dock: top; }
    HelpScreen #help-scroll { height: auto; max-height: 1fr; }
    """

    def __init__(self, contexts: Sequence[str], title: str) -> None:
        super().__init__()
        self.contexts = list(contexts)
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box", classes="dialog"):
            yield Input(placeholder="type to search keys …", id="help-filter", compact=True)
            with VerticalScroll(id="help-scroll"):     # a short terminal: the keys scroll (wheel) instead of clipping
                yield Static("", id="help-body")

    def on_mount(self) -> None:
        if not self.query("#help-box"):        # dismissed / app exiting before Mount ran (queued keys)
            return
        box = self.query_one("#help-box")
        box.border_title = f"Help · {self.title_text}"
        box.border_subtitle = "type to search · ? / esc close"
        search = self.query_one(Input)
        search.cursor_blink = bool(getattr(self.app, "animations", True))
        search.focus()
        self._refresh_body("")

    def on_input_changed(self, event: Input.Changed) -> None:
        # The focused search box consumes printable keys, so `?` cannot be a binding here: `?` (never part of a key
        # search) toggles the overlay closed. `q` does not: "qos" / "queue" are searches, and closing on their first
        # letter would send the rest of the word to the screen below as commands (`s` = shell).
        if "?" in event.value:
            self.dismiss(None)
            return
        self._refresh_body(event.value)

    def _refresh_body(self, needle: str) -> None:
        if not self.query("#help-body"):
            return
        width = min(116, (self.app.size.width * 96) // 100) - 4
        body = render_help(self.contexts, needle, width)
        if not needle:
            service = getattr(self.app, "service", None)
            recent = service.config.general.recent_window if service is not None else 86400
            body = Text("\n").join([body, Text(""), legend(self.contexts, width, humanize.window(recent))])
        self.query_one("#help-body", Static).update(body)

    def action_close(self) -> None:
        self.dismiss(None)

