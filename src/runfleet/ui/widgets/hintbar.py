"""[WP3] Context-sensitive key hint bar.

Generated from :mod:`runfleet.ui.keymap` for the active contexts; mutating entries disappear in read-only mode, and
the app may relabel or hide entries for the current selection (``^k cancel`` vs ``^k terminate``, nothing on an
ended job). Each entry is clickable and behaves exactly like pressing its key. ``? help`` is reserved at the right
edge and never dropped; when space runs out the low-value globals (``tab view``, ``^p cmd``) go first, then entries
from the end.
"""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from runfleet.ui import keymap, theme


class HintBar(Static):
    @property
    def link_style(self) -> Any:      # chips keep their own colours (theme.link_style)
        return theme.link_style()

    @property
    def link_style_hover(self) -> Any:
        return theme.link_style(hover=True)

    DEFAULT_CSS = """
    HintBar { height: 1; width: 1fr; }
    """

    def __init__(self, contexts: Sequence[str] = ("global",), *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.contexts = tuple(contexts)
        self.text = Text()

    def set_contexts(self, contexts: Sequence[str]) -> None:
        if tuple(contexts) != self.contexts:
            self.contexts = tuple(contexts)
            self.refresh_content()

    def on_mount(self) -> None:
        self.refresh_content()
        self.set_interval(0.5, self.refresh_content)      # follows the selection (relabelled / hidden actions)

    def on_resize(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if self.is_mounted:
            readonly = bool(getattr(self.app, "readonly", False))
            overrides_of = getattr(self.app, "hint_overrides", None)
            overrides = overrides_of(self.contexts) if callable(overrides_of) else {}
            text = self.build(self.contexts, self.size.width or 120, readonly=readonly, overrides=overrides)
            if text != self.text:
                self.text = text
                self.update(text)

    @staticmethod
    def build(contexts: Sequence[str], width: int, *, readonly: bool = False,
              overrides: Mapping[str, str | None] | None = None) -> Text:
        """``overrides``: action -> new label, or None to hide the entry (e.g. no cancellable job selected)."""
        overrides = overrides or {}
        key_style = theme.accent()

        def item_for(key_label: str, label: str, action: str) -> Text:
            item = Text()
            item.append(key_label, style=key_style)
            item.append(" " + label, style=theme.muted())
            item.stylize(Style(meta={"@click": f"app.press({_textual_key(contexts, key_label, action)!r})"}))
            return item

        entries: list[tuple[str, Text]] = []
        help_item: Text | None = None
        for key_label, label, action in keymap.hints(contexts, readonly=readonly):
            if action in overrides:
                if overrides[action] is None:
                    continue
                label = overrides[action] or label
            if action == "app.help":
                help_item = item_for(key_label, label, action)
                continue
            entries.append((action, item_for(key_label, label, action)))
        reserve = (help_item.cell_len + 2) if help_item is not None else 0

        def total(items: list[tuple[str, Text]]) -> int:
            return 1 + sum(t.cell_len for _a, t in items) + 2 * max(0, len(items) - 1)

        for low_value in LOW_VALUE:          # drop these before anything view-specific
            if total(entries) + reserve <= width:
                break
            entries = [e for e in entries if e[0] != low_value]
        while entries and total(entries) + reserve > width:
            entries.pop()
        out = Text(" ")
        for i, (_action, item) in enumerate(entries):
            if i:
                out.append("  ")
            out.append_text(item)
        if help_item is not None and out.cell_len + help_item.cell_len + 1 <= width:
            out.append(" " * max(2, width - out.cell_len - help_item.cell_len - 1))
            out.append_text(help_item)
        return out


LOW_VALUE = ("screen.next_view", "app.command_palette", "app.back_or_quit")


def _textual_key(contexts: Sequence[str], label: str, action: str | None = None) -> str:
    """The Textual key whose hint label is ``label`` (and, when given, whose action is ``action``)."""
    for ctx in contexts:
        for k in keymap.keys_of(ctx):
            if (k.hint_key or keymap.display_key(k.keys[0])) == label and (action is None or k.action == action):
                return k.keys[0]
    return label
