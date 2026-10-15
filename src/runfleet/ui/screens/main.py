"""[WP3] Main screen: Fleet / Jobs / GPUs views with header, tabs and hint bar.

``tab``/``shift+tab`` cycle the views (the only focusables are each view's table/cards and the filter bar). Also
home of :class:`TooSmallScreen`, the placeholder shown below 60×16.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import ContentSwitcher, Static

from runfleet.models import Job, JobKey, Visibility
from runfleet.ui import keymap, theme
from runfleet.ui.messages import FleetChanged, ScopeChanged
from runfleet.ui.screens import FleetScreen
from runfleet.ui.widgets.fleet_view import FleetView
from runfleet.ui.widgets.gpus_view import GPUsView
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.ui.widgets.tabs import VIEWS, ViewTabs

VIEW_IDS = tuple(v for v, _ in VIEWS)
TITLES = {"fleet": "Fleet view", "jobs": "Jobs view", "gpus": "GPUs view"}


class MainScreen(FleetScreen):
    BINDINGS = keymap.bindings("main")

    def __init__(self) -> None:
        super().__init__()
        self.view = "fleet"

    def compose(self) -> ComposeResult:
        yield FleetHeader(id="header")
        yield ViewTabs(id="tabs")
        with ContentSwitcher(id="views", initial="fleet"):
            yield FleetView(id="fleet")
            yield JobsView(id="jobs")
            yield GPUsView(id="gpus")
        yield HintBar(("fleet", "main", "global"), id="hints")

    def on_mount(self) -> None:
        default = self.rf.service.config.general.default_view
        self.show_view(default if default in VIEW_IDS else "fleet")

    # ------------------------------------------------------------------ views
    @property
    def fleet(self) -> FleetView:
        return self.query_one(FleetView)

    @property
    def jobs(self) -> JobsView:
        return self.query_one(JobsView)

    @property
    def gpus(self) -> GPUsView:
        return self.query_one(GPUsView)

    def action_show_view(self, view: str) -> None:
        self.show_view(view)

    def show_view(self, view: str) -> None:
        if view not in VIEW_IDS:
            return
        self.view = view
        self.query_one(ContentSwitcher).current = view
        self.query_one(HintBar).set_contexts(self.help_contexts())
        self._refresh_view(view)
        if view == "fleet":
            self.fleet.focus_selection()
        elif view == "jobs":
            self.jobs.table.focus()
        else:
            self.gpus.table.focus()
        self.update_tabs()
        self.rf.update_visibility()

    def action_next_view(self) -> None:
        self.show_view(VIEW_IDS[(VIEW_IDS.index(self.view) + 1) % len(VIEW_IDS)])

    def action_prev_view(self) -> None:
        self.show_view(VIEW_IDS[(VIEW_IDS.index(self.view) - 1) % len(VIEW_IDS)])

    def update_tabs(self) -> None:
        if not self.is_mounted:
            return
        width = self.size.width or self.app.size.width
        if self.view == "jobs":
            status = self.jobs.status_text(width)
        elif self.view == "gpus":
            status = self.gpus.status_text(width)
        else:
            scope = self.rf.scope
            status = Text(f"scope: {scope}" if scope else "scope: all hosts", style=theme.muted())
            if scope:
                status.append("   0 shows all", style=theme.dim())
        self.query_one(ViewTabs).set_state(self.view, status)

    def _refresh_view(self, view: str) -> None:
        if view == "fleet":
            self.fleet.refresh_data()
        elif view == "jobs":
            self.jobs.refresh_data()
        else:
            self.gpus.refresh_data()

    # ------------------------------------------------------------------ FleetScreen
    def help_contexts(self) -> list[str]:
        return [self.view, "main", "global"]

    def legend_contexts(self) -> list[str]:
        """The fleet panel's glyphs only while the Fleet view shows the panel (not in table mode or compact cards)."""
        plan = self.fleet.plan
        shown = self.view == "fleet" and self.fleet.mode == "cards" and plan is not None and plan.panel is not None
        return [*self.help_contexts(), *(["fleet_panel"] if shown else [])]

    def help_title(self) -> str:
        return TITLES[self.view]

    def current_host(self) -> str | None:
        if self.view == "fleet":
            return self.fleet.current_host()
        if self.view == "gpus":
            return self.gpus.current_host()
        job = self.jobs.current_job()
        return job.key.host if job is not None else self.rf.scope

    def current_job(self) -> Job | None:
        if self.view == "jobs":
            return self.jobs.current_job()
        if self.view == "gpus":
            return self.gpus.current_job()
        return None

    def action_targets(self) -> list[JobKey]:
        if self.view == "jobs":
            return self.jobs.action_targets()
        job = self.current_job()
        return [job.key] if job is not None else []

    def visibility(self) -> dict[str, Visibility]:
        hosts = self.rf.host_names() if self.view == "fleet" else self.rf.scope_hosts()
        return {h: Visibility.VISIBLE for h in hosts}

    def escape(self) -> bool:
        if self.view == "jobs":
            return self.jobs.clear_filter() or self.jobs.clear_marks()
        return False

    def open_filter(self) -> None:
        if self.view != "jobs":
            self.show_view("jobs")
        self.jobs.open_filter()

    def clear_marks(self) -> None:
        self.jobs.clear_marks()

    def unmark(self, keys: Sequence[JobKey]) -> None:
        """Drop the marks of these jobs (an action handled them); other marks stay."""
        done = {str(k) for k in keys}
        if self.jobs.marks & done:
            self.jobs.marks -= done
            self.jobs.refresh_data()

    def refresh_data(self, changes: FleetChanged | None) -> None:
        if not self.is_mounted:
            return
        self.query_one(FleetHeader).refresh_content()
        if changes is None or changes.jobs_changed or changes.status_changed or changes.gpus_changed:
            self._refresh_view(self.view)
        self.update_tabs()

    def refresh_time(self) -> None:
        self.query_one(FleetHeader).refresh_content()
        if self.view == "jobs":
            self.jobs.refresh_time()
        elif self.view == "fleet":
            self.fleet.refresh_data()

    def on_scope_changed(self, message: ScopeChanged) -> None:
        if message.scope is not None:        # `1`-`9` scope: the Fleet selection follows, so `c`/`s`/`↵` hit it
            fleet = self.query_one(FleetView)
            fleet.selected = message.scope
            if self.view == "fleet":
                fleet.focus_selection()
        self.query_one(FleetHeader).refresh_content()
        self._refresh_view(self.view)
        self.update_tabs()


class TooSmallScreen(Screen[Any]):
    """Placeholder below 60×16 (§7.1); the app pops it on resize."""

    DEFAULT_CSS = """
    TooSmallScreen { align: center middle; }
    TooSmallScreen > Static { width: auto; height: auto; text-align: center; }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text.assemble(("RunFleet", "bold"), "\nterminal too small\nneeds at least 60×16"))

    def on_resize(self) -> None:
        self.app.check_size()  # type: ignore[attr-defined]
