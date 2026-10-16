"""Palette › Save view as…: the name prompt for a saved Jobs view (§7.6, ``ui/saved_views.py``).

One line, the same pattern as the port prompt (keymap context ``prompt``): it shows what will be saved, says live why a
name cannot be used (rules, :data:`~runfleet.ui.saved_views.MAX_VIEWS` reached) or that it replaces an existing view,
and dismisses with the name on ``enter`` (None on ``esc``). Saving is the caller's job.
"""

from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from runfleet.ui import keymap, saved_views, theme
from runfleet.ui.saved_views import NAME_MAX, SavedView


class SaveViewScreen(ModalScreen[str | None]):
    """Ask for a view name; ``↵`` dismisses with it once it is usable, ``esc`` with None."""

    BINDINGS = keymap.bindings("prompt")
    HELP_CONTEXT = "prompt"
    DEFAULT_CSS = """
    SaveViewScreen #save-view-box { width: 64; }
    SaveViewScreen Input { margin: 1 0 0 0; }
    """

    def __init__(self, view: SavedView, views: Mapping[str, SavedView], suggestion: str = "") -> None:
        super().__init__()
        self.view = view
        self.views = dict(views)
        self.suggestion = suggestion

    def compose(self) -> ComposeResult:
        box = Vertical(id="save-view-box", classes="dialog")
        box.border_title = "Save the Jobs view as…"
        box.border_subtitle = "↵ save · esc cancel"
        with box:
            yield Static(Text(saved_views.describe(self.view), style=theme.muted()), id="save-view-what")
            yield Input(self.suggestion, placeholder=f"name, e.g. failed today (≤ {NAME_MAX} characters)",
                        id="save-view-name", compact=True, max_length=NAME_MAX)
            yield Static("", id="save-view-note")

    def on_mount(self) -> None:
        if self.query("#save-view-name"):
            self.query_one(Input).focus()
            self._note(self.suggestion.strip(), submitted=False)

    def forget_rendered(self) -> None:
        """Live theme switch: the summary line and the note carry theme colours."""
        if self.query("#save-view-what"):
            self.query_one("#save-view-what", Static).update(Text(saved_views.describe(self.view),
                                                                 style=theme.muted()))
            self._note(self.query_one(Input).value.strip(), submitted=False)

    def _note(self, name: str, *, submitted: bool) -> str | None:
        """Show why ``name`` cannot be saved (only once submitted for an empty name), or that it replaces a view."""
        problem = saved_views.save_problem(self.views, name)
        note = self.query_one("#save-view-note", Static)
        if problem is not None and (name or submitted):
            note.update(Text(problem, style=theme.warn()))
        elif problem is None and (old := saved_views.find(self.views, name)) is not None:
            note.update(Text(f"replaces the saved view “{old}”", style=theme.accent()))
        else:
            note.update(Text(f"{len(self.views)} of {saved_views.MAX_VIEWS} saved · palette › View: <name> applies one",
                             style=theme.dim()))
        return problem

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._note(event.value.strip(), submitted=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        name = event.value.strip()
        if self._note(name, submitted=True) is None:
            self.dismiss(name)

    def action_close(self) -> None:
        self.dismiss(None)
