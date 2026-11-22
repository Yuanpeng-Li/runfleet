"""Clickable lines of a Static (the Fleet cards' job rows, the fleet panel's rows): each shown line may carry what a
click on it opens, and the line under the mouse is underlined while it does (like the header chips and view tabs,
:func:`runfleet.ui.theme.link_style`), so a clickable row tells itself apart from the text around it.

Lines beyond the rows on screen scroll (mouse wheel, :meth:`RowLinks.scroll_lines`): the first ``pinned`` lines (a card's
summary) stay put, the rest moves under them, and the bottom border says how many lines are above and below."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.text import Text
from textual import events
from textual.widgets import Static

from runfleet.ui import theme


class RowLinks(Static):
    """A Static whose content is ``lines`` (padded with blank lines to ``pad_to``, the rows on screen); ``row_keys[i]``
    is what line ``i`` opens (None: not clickable). More lines than ``pad_to``: the lines after the first ``pinned``
    scroll. Subclasses call :meth:`show_lines` instead of ``update``."""

    SCROLL_STEP = 1                                # lines per wheel notch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lines: list[Text] = []
        self.row_keys: list[Any] = []
        self._pad_to = 0
        self._pinned = 0
        self._offset = 0                           # lines of the scrolling part above the screen
        self._hover: int | None = None             # the clickable screen row drawn underlined
        self._pointer: int | None = None           # the screen row under the mouse, clickable or not

    def show_lines(self, lines: Sequence[Text], keys: Sequence[Any], pad_to: int, pinned: int = 0) -> None:
        anchor = self._anchor() if self._offset else None
        self.lines = list(lines)
        self.row_keys = list(keys)[:len(self.lines)]
        self._pad_to = pad_to
        self._pinned = pinned
        if anchor is not None:                    # scrolled: keep the same entry on the same screen row
            key, text, row, was = anchor
            pin = self._pin()
            same = [i for i in range(pin, len(self.row_keys)) if self.row_keys[i] == key]
            exact = [i for i in same if self.lines[i].plain == text]
            if exact or same:                     # one job may have several rows (activity): its text, else the
                self._offset = min(exact or same, key=lambda i: abs(i - was)) - pin - row   # nearest, tells them apart
        self._clamp()
        self._hover = self._link(self._pointer)   # new lines under a resting mouse: underline what is clickable now
        self._paint()

    # ------------------------------------------------------------------ scrolling
    @property
    def scrollable(self) -> bool:
        return self._pad_to > 0 and len(self.lines) > self._pad_to

    def _pin(self) -> int:
        """Pinned lines on screen: none when they would leave fewer than two rows to scroll."""
        return self._pinned if self._pinned <= self._pad_to - 2 else 0

    def _max_offset(self) -> int:
        if not self.scrollable:
            return 0
        pin = self._pin()
        return len(self.lines) - pin - (self._pad_to - pin)

    def _clamp(self) -> None:
        self._offset = max(0, min(self._offset, self._max_offset()))

    def shown_lines(self) -> list[int]:
        """The index in ``lines`` of each row on screen."""
        if not self.scrollable:
            return list(range(len(self.lines)))
        pin = self._pin()
        start = pin + self._offset
        return [*range(pin), *range(start, start + self._pad_to - pin)]

    def _anchor(self) -> tuple[Any, str, int, int] | None:
        """The first entry with a key in the scrolling part on screen: its key, its text, its row there and its
        index in ``lines``."""
        pin = self._pin()
        for row, i in enumerate(self.shown_lines()[pin:]):
            key = self.row_keys[i] if i < len(self.row_keys) else None
            if key is not None:
                return key, self.lines[i].plain, row, i
        return None

    def scroll_lines(self, delta: int) -> bool:
        """Scroll the list ``delta`` lines (down > 0); False when it cannot move that way."""
        before = self._offset
        self._offset += delta
        self._clamp()
        if self._offset == before:
            return False
        self._hover = self._link(self._pointer)
        self._paint()
        return True

    def page_rows(self) -> int:
        """Rows of the scrolling part on screen."""
        return max(1, self._pad_to - self._pin())

    def list_home(self) -> None:
        if self._offset:
            self._offset = 0
            self._paint()

    def scroll_note(self) -> Text | None:
        """``↑3 ↓12``: lines above and below the screen (None: nothing hidden)."""
        if not self.scrollable:
            return None
        above, below = self._offset, self._max_offset() - self._offset
        parts = ([f"↑{above}"] if above else []) + ([f"↓{below}"] if below else [])
        return Text(" ".join(parts), style=theme.dim()) if parts else None

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self.scroll_lines(self.SCROLL_STEP):
            event.stop()
            event.prevent_default()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.scroll_lines(-self.SCROLL_STEP):
            event.stop()
            event.prevent_default()

    # ------------------------------------------------------------------ lines and links
    def _key(self, row: int | None) -> Any:
        """What the screen row ``row`` opens."""
        if row is None:
            return None
        shown = self.shown_lines()
        if not 0 <= row < len(shown):
            return None
        i = shown[row]
        return self.row_keys[i] if i < len(self.row_keys) else None

    def _link(self, row: int | None) -> int | None:
        return row if self._key(row) is not None else None

    def _paint(self) -> None:
        shown = [self.lines[i] for i in self.shown_lines()]
        if self._hover is not None and self._hover < len(shown):
            line = shown[self._hover].copy()
            line.stylize("underline")
            shown[self._hover] = line
        self.border_subtitle = self.scroll_note()
        self.update(Text("\n").join(shown + [Text("")] * max(0, self._pad_to - len(shown))))

    def row_at(self, event: events.MouseEvent) -> int | None:
        """The screen row (content line) under the mouse (None: the border, padding or past the lines)."""
        offset = event.get_content_offset(self)
        if offset is None or not 0 <= offset.y < len(self.shown_lines()):
            return None
        return offset.y

    def line_at(self, event: events.MouseEvent) -> int | None:
        """The index in ``lines`` of the line under the mouse."""
        row = self.row_at(event)
        return None if row is None else self.shown_lines()[row]

    def key_at(self, event: events.MouseEvent) -> Any:
        """What the line under the mouse opens (None: nothing)."""
        return self._key(self.row_at(event))

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
