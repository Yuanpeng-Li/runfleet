"""[WP3] Action menu with disabled reasons; also the session's action log.

``ActionMenuScreen`` lists ``FleetAPI.actions_for()``: enabled actions first, disabled ones greyed with their reason
(read-only mode, host needs auth, stale data, protected partition, ...). Dismisses with the chosen action id.
Optional single-key accelerators (``keys``, the copy menu's ``o`` / ``e`` / ``i`` …) pick an entry without ``↵``.
``ActionLogScreen`` (palette › Action log) shows the last results with their full output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from runfleet.backends.base import ActionSpec
from runfleet.models import ActionResult
from runfleet.ui import keymap, theme
from runfleet.util import humanize


# the menu's own keys (action_menu bindings + OptionList's cursor keys and ↵): never accelerators
RESERVED_KEYS = frozenset({k for key in keymap.keys_of("action_menu") if key.bind for k in key.keys}
                          | {"enter", "up", "down", "home", "end", "pageup", "pagedown", "tab", "shift+tab"})
_CLOSE_KEYS = frozenset(k for key in keymap.keys_of("action_menu") if key.action == "close" for k in key.keys)
_NAV_KEYS = {"down": "cursor_down", "up": "cursor_up", "home": "first", "end": "last", "pagedown": "page_down",
             "pageup": "page_up"} | {k: key.action for key in keymap.keys_of("action_menu")
                                    if key.action in ("cursor_down", "cursor_up") for k in key.keys}


def option_text(spec: ActionSpec, key: str | None = None, *, pad: bool = False) -> Text:
    """One menu line; ``key`` (an accelerator) leads it, ``pad`` keeps key-less lines aligned with keyed ones."""
    out = Text()
    if key is not None:               # bold, not accent: the highlight bar is accent-coloured and would hide it
        out.append(f"{keymap.display_key(key)} ", style="bold" if spec.disabled_reason is None else theme.dim())
    elif pad:
        out.append("  ")
    out.append(spec.label, style="" if spec.disabled_reason is None else theme.dim())
    if spec.disabled_reason:
        out.append(f"  — {spec.disabled_reason}", style=f"{theme.dim()} italic")
    elif spec.mutating:
        out.append("  (asks to confirm)", style=theme.dim())
    return out


class ActionMenuScreen(ModalScreen[str | None]):
    BINDINGS = keymap.bindings("action_menu")
    HELP_CONTEXT = "action_menu"
    DEFAULT_CSS = """
    ActionMenuScreen OptionList { border: none; height: auto; max-height: 16; background: transparent; padding: 0; }
    """

    def __init__(self, specs: Sequence[ActionSpec], n_targets: int, *, title: str | None = None,
                 keys: Mapping[str, str] | None = None, subtitle: str | None = None) -> None:
        """``keys``: accelerator key (Textual key name) -> action id; several keys may pick one entry, the first is
        the one shown on its line. Keys of the menu itself (``RESERVED_KEYS``) are refused."""
        super().__init__()
        self.specs = sorted(specs, key=lambda s: s.disabled_reason is not None)
        self.n_targets = n_targets
        self.title_text = title
        self.subtitle_text = subtitle
        self.keys = dict(keys or {})
        if clash := sorted(set(self.keys) & RESERVED_KEYS):
            raise ValueError(f"menu accelerators {clash} are keys of the menu itself")
        if unknown := sorted(set(self.keys.values()) - {s.id for s in self.specs}):
            raise ValueError(f"menu accelerators name no entry: {unknown}")
        self.shown_key: dict[str, str] = {}
        self.picked = False                     # the first close wins (keys queued behind it are dropped)
        for key, action_id in self.keys.items():
            self.shown_key.setdefault(action_id, key)

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="menu-box", classes="dialog")       # titles set here: on_mount may find no DOM (below)
        box.border_title = self.title_text or f"Actions · {self.n_targets} target{'s' if self.n_targets != 1 else ''}"
        box.border_subtitle = self.subtitle_text or ("↵ / key run · esc close" if self.keys else "↵ run · esc close")
        with box:
            yield OptionList(*[Option(option_text(s, self.shown_key.get(s.id), pad=bool(self.keys)), id=s.id,
                                      disabled=s.disabled_reason is not None) for s in self.specs], id="menu")

    def on_mount(self) -> None:
        # a burst of keys can dismiss the menu (or exit the app) before Mount is handled: nothing left to set up
        found = self.query(OptionList)
        if not found:
            return
        menu = found.first()
        menu.focus()
        enabled = [i for i, s in enumerate(self.specs) if s.disabled_reason is None]
        if enabled:
            menu.highlighted = enabled[0]

    def pick(self, action_id: str | None) -> None:
        if self.picked:
            return
        self.picked = True
        self.dismiss(action_id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.pick(event.option.id)

    def on_key(self, event: events.Key) -> None:
        """An accelerator picks its entry at once; a disabled one only says why (the menu stays open). Modal: the
        app's bindings are not reached anyway, and the menu's own keys are never accelerators.

        Moving, ↵ and esc / q are handled here too, so a burst of keys is taken in order: through the OptionList /
        the bindings they would only post a message, which the next key overtakes (``↵ Y``, ``esc i``, ``↓ ↵``)."""
        if self.picked or not self.is_current:
            event.stop()
            event.prevent_default()
            return
        if event.key in _NAV_KEYS:
            event.stop()
            event.prevent_default()
            found = self.query(OptionList)
            if found:
                getattr(found.first(), f"action_{_NAV_KEYS[event.key]}")()
            return
        if event.key in _CLOSE_KEYS:
            event.stop()
            event.prevent_default()
            self.pick(None)
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            found = self.query(OptionList)
            index = found.first().highlighted if found else None
            if index is not None and self.specs[index].disabled_reason is None:
                self.pick(self.specs[index].id)
            return
        action_id = self.keys.get(event.key)
        if action_id is None:
            return
        event.stop()
        event.prevent_default()
        spec = next(s for s in self.specs if s.id == action_id)
        if spec.disabled_reason is not None:
            self.rf.toast(f"{spec.label} — {spec.disabled_reason}", severity="warning")
            return
        self.pick(action_id)

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_close(self) -> None:
        self.pick(None)


def result_lines(results: Sequence[ActionResult]) -> Text:
    out: list[Text] = []
    for r in reversed(results):
        head = Text("✓ " if r.ok else "✖ ", style=theme.ok() if r.ok else theme.error())
        head.append(f"{r.action_id} {r.target}", style="bold")
        head.append(f"  {r.message}")
        if r.started_at is not None:
            head.append(f"  {humanize.local_hm(r.started_at)}", style=theme.muted())
        if r.rc is not None:
            head.append(f"  rc={r.rc}", style=theme.muted())
        out.append(head)
        for line in r.details:
            out.append(Text(f"    {line}", style=theme.muted()))
        for line in (r.stderr or "").splitlines()[:20]:
            out.append(Text(f"    {line}", style=theme.warn()))
    if not out:
        out.append(Text("no actions in this session", style=theme.dim()))
    return Text("\n").join(out)


class ActionLogScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("modal")
    HELP_CONTEXT = "modal"
    DEFAULT_CSS = """
    ActionLogScreen VerticalScroll { height: auto; max-height: 30; }
    """

    def __init__(self, results: Sequence[ActionResult]) -> None:
        super().__init__()
        self.results = list(results)

    def compose(self) -> ComposeResult:
        with Vertical(id="log-box", classes="dialog"):
            with VerticalScroll():
                yield Static(result_lines(self.results))

    def on_mount(self) -> None:
        box = self.query_one("#log-box")
        box.border_title = f"Action log · {len(self.results)} results (newest first)"
        box.border_subtitle = "esc close"

    def action_close(self) -> None:
        self.dismiss(None)
