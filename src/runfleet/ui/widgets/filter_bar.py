"""[WP3] Filter input bar.

Opened with ``/``; filters live (100 ms debounce). ``enter``/``tab``/``down`` return focus to the table and keep
the filter, ``esc`` clears it and closes the bar. The bar stays visible while a filter is active.
"""

from __future__ import annotations

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Input

DEBOUNCE = 0.1


class FilterBar(Input):
    DEFAULT_CSS = """
    FilterBar { height: 1; border: none; padding: 0 1; }
    """
    BINDINGS = [
        Binding("escape", "clear_filter", "clear filter", show=False),
        Binding("enter", "done", "back to the table", show=False),
        Binding("tab", "done", "back to the table", show=False),
        Binding("down", "done", "back to the table", show=False),
    ]

    class FilterChanged(Message):
        """Debounced filter text."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class FilterDone(Message):
        """Focus should go back to the table (``cleared`` when esc emptied the filter)."""

        def __init__(self, cleared: bool) -> None:
            super().__init__()
            self.cleared = cleared

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(placeholder="filter: words · host:k2 · state:R,pd · user:me · !negate   (esc clears)",
                         id=id, compact=True)
        self._timer: Timer | None = None

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_timer(DEBOUNCE, self._emit)

    def _emit(self) -> None:
        self._timer = None
        self.post_message(self.FilterChanged(self.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()

    def action_done(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._emit()
        self.post_message(self.FilterDone(False))

    def action_clear_filter(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.value = ""
        self.post_message(self.FilterChanged(""))
        self.post_message(self.FilterDone(True))

    def on_blur(self, event: events.Blur) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._emit()
