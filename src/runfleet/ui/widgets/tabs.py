"""[WP3] View tabs (Fleet / Jobs / GPUs).

One line: the three main views (clickable, the active one framed ``▌Jobs▐``) and a right-aligned status text
supplied by the active view (scope, filter/sort summary, row counts).
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from runfleet.ui import theme

VIEWS: tuple[tuple[str, str], ...] = (("fleet", "Fleet"), ("jobs", "Jobs"), ("gpus", "GPUs"))


def tab_strip(items: Sequence[tuple[str, str]], active: str, action: str) -> Text:
    """`` Fleet  Jobs  GPUs `` (active one highlighted) with click actions ``<action>('<id>')``."""
    out = Text()
    for ident, label in items:
        if ident == active:
            part = Text(f" {label} ", style=f"bold {'#ffffff' if theme.LIGHT else '#121212'} on {theme.ACCENT}" if not theme.no_color() else "bold reverse")
        else:
            part = Text(f" {label} ", style=theme.muted())
        part.stylize(Style(meta={"@click": f"{action}({ident!r})"}))
        out.append_text(part)
        out.append(" ")
    return out


class ViewTabs(Static):
    DEFAULT_CSS = """
    ViewTabs { height: 1; width: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.active = "fleet"
        self.status: Text = Text()
        self.text = Text()

    def set_state(self, active: str, status: Text | str) -> None:
        self.active = active
        self.status = Text(status, style=theme.muted()) if isinstance(status, str) else status
        self.refresh_content()

    def on_mount(self) -> None:
        self.refresh_content()

    def on_resize(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        width = self.size.width or 120
        left = Text(" ")
        left.append_text(tab_strip(VIEWS, self.active, "screen.show_view"))
        status = self.status.copy()
        room = width - left.cell_len - 2
        if status.cell_len > room:
            status.truncate(max(0, room), overflow="ellipsis")
        gap = max(1, width - left.cell_len - status.cell_len - 1)
        self.text = Text.assemble(left, " " * gap, status, " ")
        self.update(self.text)
