"""Clickable lines of a Static (the Fleet cards' job rows, the fleet panel's rows): each shown line may carry what a
click on it opens, and the line under the mouse is underlined while it does (like the header chips and view tabs,
:func:`runfleet.ui.theme.link_style`), so a clickable row tells itself apart from the text around it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.text import Text
from textual import events
from textual.widgets import Static


class RowLinks(Static):
    """A Static whose content is ``lines`` (padded with blank lines to ``pad_to``); ``row_keys[i]`` is what line ``i``
    opens (None: not clickable). Subclasses call :meth:`show_lines` instead of ``update``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lines: list[Text] = []
        self.row_keys: list[Any] = []
        self._pad_to = 0
        self._hover: int | None = None             # the clickable line drawn underlined
        self._pointer: int | None = None           # the content line under the mouse, clickable or not

    def show_lines(self, lines: Sequence[Text], keys: Sequence[Any], pad_to: int) -> None:
        self.lines = list(lines)
        self.row_keys = list(keys)[:len(self.lines)]
        self._pad_to = pad_to
        self._hover = self._link(self._pointer)   # new lines under a resting mouse: underline what is clickable now
        self._paint()

    def _link(self, row: int | None) -> int | None:
        return row if row is not None and row < len(self.row_keys) and self.row_keys[row] is not None else None

    def _paint(self) -> None:
        shown = list(self.lines)
        if self._hover is not None and self._hover < len(shown):
            line = shown[self._hover].copy()
            line.stylize("underline")
            shown[self._hover] = line
        self.update(Text("\n").join(shown + [Text("")] * max(0, self._pad_to - len(shown))))

    def row_at(self, event: events.MouseEvent) -> int | None:
        """The content line under the mouse (None: the border, padding or past the lines)."""
        offset = event.get_content_offset(self)
        if offset is None or not 0 <= offset.y < len(self.row_keys):
            return None
        return offset.y

    def key_at(self, event: events.MouseEvent) -> Any:
        """What the line under the mouse opens (None: nothing)."""
        row = self.row_at(event)
        return None if row is None else self.row_keys[row]

    def on_mouse_move(self, event: events.MouseMove) -> None:
        offset = event.get_content_offset(self)
        self._pointer = offset.y if offset is not None and offset.y >= 0 else None
        hover = self._link(self._pointer)
        if hover != self._hover:
            self._hover = hover
            self._paint()

    def on_leave(self, _event: events.Leave) -> None:
        self._pointer = None
        if self._hover is not None:
            self._hover = None
            self._paint()
