"""[WP3] Jobs view: filter bar, job table, preview pane.

Owns the Jobs-view state (filter text, state preset, sort, grouping, marks, other users) and turns the store into
table rows on every change: scope -> users -> preset -> filter -> sort -> grouping (§7.5, §7.6). The preview pane
sits below the table (right of it at >= 150 columns), collapses to one line below 30 rows and hides below 20.
A saved view (palette › View: <name>, ``ui/saved_views.py``) sets filter, preset, sort and grouping at once; the status
line names it until any of them changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from rich.text import Text
from textual import events, work
from textual.app import ComposeResult
from textual.containers import Container
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DataTable

from runfleet.core.service import LogPeek
from runfleet.models import ConnState, Job, JobKey, StateGroup
from runfleet.ui import keymap, query, saved_views, theme
from runfleet.ui.query import PRESET_LABELS, SORT_LABELS, Preset, SortColumn
from runfleet.ui.saved_views import GROUPS, SavedView
from runfleet.ui.widgets.filter_bar import FilterBar
from runfleet.ui.widgets.job_preview import JobPreview
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.job_table import COLUMNS, ID_CAP, GroupRow, JobTable, Row, RowContext, layout
from runfleet.util import humanize

SIDE_PREVIEW_WIDTH = 150
PREVIEW_FULL_HEIGHT = 30
PREVIEW_MIN_HEIGHT = 20
# Hosts whose rows get the "!" marker and a problem label in NODE / REASON; retrying hosts are only stale ("~").
PROBLEM_LABELS = {ConnState.NEEDS_AUTH: "auth", ConnState.HOSTKEY: "hostkey", ConnState.UNREACHABLE: "down"}


class JobsView(Widget):
    BINDINGS = keymap.bindings("jobs")
    DEFAULT_CSS = """
    JobsView { height: 1fr; }
    """
    PEEK_DWELL = 1.0             # seconds of cursor dwell before a log peek (§4.7); tests set 0
    PEEK_REFRESH = 15.0          # re-ask (the service decides whether to re-fetch) at most this often

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.query_text = ""
        self.parsed = query.parse("")
        self.preset: Preset = "fresh"
        self.sort: SortColumn = "host"
        self.reverse = False
        self.group = "none"
        self.marks: set[str] = set()
        self.visible_keys: set[str] = set()     # job rows on screen now (preset + filter, not in a folded group)
        self.show_others = False
        self.folded: set[str] = set()
        self.shown = 0
        self.total = 0
        self._peeks: dict[tuple[str, str], tuple[LogPeek, float]] = {}
        self._peek_pending: tuple[str, str] | None = None
        self._dwell: Timer | None = None
        self._dwell_key: str | None = None
        self.ended_since: datetime | None = None    # digest `f`: own jobs that ended after this, preset aside (esc)
        self.active_view: str | None = None         # saved view in effect; cleared once any part of it changes
        self._view_basis: SavedView | None = None

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        yield FilterBar(id="filter")
        with Container(id="jobs-body"):
            yield JobTable(id="jobs-table")
            yield JobPreview(id="preview")

    def on_mount(self) -> None:
        state = self.rf.ui_state
        self.query_text = str(state.get("filter", ""))
        self.parsed = query.parse(self.query_text)
        preset = state.get("preset", "fresh")
        self.preset = preset if preset in query.PRESETS else "fresh"
        sort = state.get("sort", "host")
        self.sort = sort if sort in SORT_LABELS else "host"
        self.reverse = bool(state.get("reverse", False))
        group = state.get("group", "none")
        self.group = group if group in GROUPS else "none"
        bar = self.query_one(FilterBar)
        bar.value = self.query_text
        bar.display = bool(self.query_text)
        table = self.table
        table.sort_column, table.sort_reverse = self.sort, self.reverse
        name = state.get("active_view")
        views = saved_views.load_views(state)
        stored = saved_views.find(views, name) if isinstance(name, str) else None
        if stored is not None and views[stored] == self.current_view():
            self.active_view, self._view_basis = stored, views[stored]
        self._apply_size()

    @property
    def table(self) -> JobTable:
        return self.query_one(JobTable)

    @property
    def preview(self) -> JobPreview:
        return self.query_one(JobPreview)

    # ------------------------------------------------------------------ size rules (§7.4)
    def on_resize(self, event: events.Resize) -> None:
        self._apply_size()
        self.refresh_data()

    def _apply_size(self) -> None:
        width, height = self.app.size
        side = width >= SIDE_PREVIEW_WIDTH
        self.set_class(side, "-side")
        preview = self.preview
        preview.side = side
        if height < PREVIEW_MIN_HEIGHT:
            preview.mode = "hidden"
        elif height < PREVIEW_FULL_HEIGHT and not side:
            preview.mode = "line"
        else:
            preview.mode = "full"
        preview.display = preview.mode != "hidden"
        preview.lines_budget = max(4, height - 6) if side else (2 if preview.mode == "line" else 8)
        preview.styles.height = None if side else preview.lines_budget

    def _table_width(self) -> int:
        width = self.size.width or self.app.size.width
        if self.has_class("-side"):
            width = int(width * 0.62)
        return max(40, width - 2)       # vertical scrollbar + 1 cell of air

    # ------------------------------------------------------------------ rows
    def visible_jobs(self) -> tuple[list[Job], int]:
        now = self.rf.now()
        window = self.rf.service.config.general.recent_window
        jobs = [j for j in self.rf.store.jobs(self.rf.scope_hosts()) if self.show_others or j.is_mine]
        total = len(jobs)
        if self.ended_since is not None:
            since = self.ended_since
            jobs = [j for j in jobs if (end := query.ended_at(j)) is not None and end > since and j.is_mine
                    and query.matches(self.parsed, j)]
            return query.sort_jobs(jobs, self.sort, self.reverse, host_order=self.rf.host_names(), now=now), total
        jobs = [j for j in jobs if query.preset_matches(self.preset, j, now, window) and query.matches(self.parsed, j)]
        return query.sort_jobs(jobs, self.sort, self.reverse, host_order=self.rf.host_names(), now=now), total

    def _group(self, jobs: list[Job]) -> list[Row]:
        if self.group == "none":
            return list(jobs)
        rows: list[Row] = []
        if self.group == "host":
            for host in self.rf.host_names():
                members = [j for j in jobs if j.key.host == host]
                if not members:
                    continue
                key = f"§host:{host}"
                rows.append(GroupRow(key, host, tuple(members), key in self.folded))
                if key not in self.folded:
                    rows.extend(members)
            return rows
        # array: tasks under their parent, at the position of the first member; other jobs unchanged
        arrays: dict[str, list[Job]] = {}
        for job in jobs:
            if job.array_job_id:
                arrays.setdefault(f"§array:{job.key.host}:{job.array_job_id}", []).append(job)
        seen: set[str] = set()
        for job in jobs:
            if not job.array_job_id:
                rows.append(job)
                continue
            key = f"§array:{job.key.host}:{job.array_job_id}"
            if key in seen:
                continue
            seen.add(key)
            members = arrays[key]
            rows.append(GroupRow(key, f"{job.key.host} array {job.array_job_id}", tuple(members),
                                 key in self.folded))
            if key not in self.folded:
                rows.extend(members)
        return rows

    def _row_context(self, now: datetime) -> RowContext:
        hosts = self.rf.store.hosts
        colors = {}
        problems = {}
        stale = set()
        for i, (name, state) in enumerate(hosts.items()):
            colors[name] = theme.host_color(state.config, i)
            label = PROBLEM_LABELS.get(state.status.state)
            if label:
                problems[name] = label
            if state.stale(now):
                stale.add(name)
        alerts = self.rf.alerts
        watched = frozenset(str(j.key) for j in self.rf.store.jobs(self.rf.scope_hosts()) if alerts.is_watched(j)) \
            if alerts.watch else frozenset()
        return RowContext(now, colors, problems, frozenset(stale), frozenset(self.marks),
                          self.rf.service.config.ui.gpu_util_thresholds,
                          name_starts=getattr(self, "_name_starts", {}), watched=watched)

    def refresh_data(self) -> None:
        if not self.is_mounted:
            return
        self._check_view()
        now = self.rf.now()
        jobs, total = self.visible_jobs()
        live = {str(j.key) for j in self.rf.store.jobs()}
        self.marks &= live
        self.shown, self.total = len(jobs), total
        self._name_starts = humanize.distinct_starts([j.name for j in jobs])   # sweep siblings stay distinguishable
        rows = self._group(jobs)
        # a folded group hides its members like a filter does: marked members count as hidden, never acted on
        self.visible_keys = {str(r.key) for r in rows if isinstance(r, Job)}
        names = self.rf.host_names()
        host_w = max([8] + [len(n) for n in names]) if names else 8
        id_w = max([COLUMNS["id"].width] + [len(j.key.label) for j in jobs])   # array ids: "31377076_3"
        cols = layout(self._table_width(), self.rf.service.config.ui.jobs_columns, show_user=self.show_others,
                      host_width=min(12, host_w), id_width=min(ID_CAP, id_w))
        table = self.table
        table.sort_column, table.sort_reverse = self.sort, self.reverse
        if not table.set_columns(cols):
            table.refresh_headers()
        table.set_rows(rows, self._row_context(now))
        self._update_preview()
        self._post_status()

    def refresh_time(self) -> None:
        """1 Hz: ELAPSED/LEFT of ACTIVE rows (cells only change when the text changes)."""
        if self.is_mounted:
            table = self.table
            if table.order:
                table.set_rows([table.rows_by_key[k] for k in table.order], self._row_context(self.rf.now()))
            self._update_preview()

    # ------------------------------------------------------------------ saved views (§7.6)
    def current_view(self) -> SavedView:
        """Filter, preset, sort and grouping as they are now (what palette › Save view as… stores)."""
        return SavedView(self.query_text, self.preset, self.sort, self.reverse, self.group)

    def apply_view(self, name: str, view: SavedView) -> None:
        """Palette › View: <name>: take over all five settings at once (the digest's `f` mode ends too)."""
        self.query_text, self.parsed = view.filter, query.parse(view.filter)
        bar = self.query_one(FilterBar)
        bar.value = view.filter
        bar.display = bool(view.filter)
        self.preset, self.sort, self.reverse, self.group = view.preset, view.sort, view.reverse, view.group
        self.folded.clear()
        self.ended_since = None
        self.active_view, self._view_basis = name, view
        self.rf.save_ui_state(filter=view.filter, preset=view.preset, sort=view.sort, reverse=view.reverse,
                              group=view.group, active_view=name)
        self.refresh_data()
        self.table.focus()

    def view_saved(self, name: str) -> None:
        """The current settings were just saved as ``name``: the status line names it from now on."""
        self.active_view, self._view_basis = name, self.current_view()
        self.rf.save_ui_state(active_view=name)
        self._post_status()

    def view_deleted(self, name: str) -> None:
        if self.active_view is not None and self.active_view.casefold() == name.casefold():
            self._drop_view()
            self._post_status()

    def _check_view(self) -> None:
        if self.active_view is not None and (self.ended_since is not None or self.current_view() != self._view_basis):
            self._drop_view()

    def _drop_view(self) -> None:
        self.active_view, self._view_basis = None, None
        self.rf.save_ui_state(active_view=None)

    def _post_status(self) -> None:
        main = self.screen
        if hasattr(main, "update_tabs"):
            main.update_tabs()

    def status_text(self, width: int) -> Text:
        window = humanize.window(self.rf.service.config.general.recent_window)
        sort = f"{SORT_LABELS[self.sort]} {query.sort_arrow(self.sort, self.reverse)}"
        if self.ended_since is not None:       # digest `f`: the preset is set aside until esc
            out = Text(f"ended since {humanize.local_hm(self.ended_since)} · {self.shown}/{self.total}",
                       style=theme.accent())
            if self.query_text:
                out.append(f" · /{self.query_text}", style=theme.accent())
            out.append("  esc clears", style=theme.dim())
            return out
        if width < 100:
            out = Text()
            if self.active_view:
                out.append(f"{self.active_view} · ", style=theme.accent())
            out.append(f"{self.preset} · {sort.replace(' ', '')} · {self.shown}/{self.total}", style=theme.muted())
            if self.query_text:
                out.append(f" · /{self.query_text}", style=theme.accent())
            return out
        room = width - 24                      # the tab strip takes the left ~24 cells
        for compact in (False, True):
            out = self._status_full(window, sort, compact=compact)
            if out.cell_len <= room:
                return out
        # still too long: keep the view name, the count and the filter visible, drop the labels
        out = Text()
        if self.active_view:
            out.append(f"{self.active_view} · ", style=theme.accent())
        out.append(f"{self.preset} · {sort} · ", style=theme.muted())
        if self.query_text:
            out.append(f"/{self.query_text} · ", style=theme.accent())
        if self.marks:
            out.append(f"{self._marks_label()} · ", style=theme.accent())
        out.append(f"{self.shown}/{self.total}", style=theme.muted())
        return out

    def _status_full(self, window: str, sort: str, *, compact: bool) -> Text:
        gap = "  " if compact else "   "
        out = Text()
        if self.active_view:
            out.append("view: ", style=theme.dim())
            out.append(self.active_view, style=theme.accent())
            out.append(gap)
        out.append("state: ", style=theme.dim())
        out.append(PRESET_LABELS[self.preset].format(window=window), style=theme.muted())
        out.append(f"{gap}sort: ", style=theme.dim())
        out.append(sort, style=theme.muted())
        if not compact or self.group != "none":
            out.append(f"{gap}group: ", style=theme.dim())
            out.append(self.group, style=theme.muted())
        if self.query_text:
            out.append(f"{gap}filter: ", style=theme.dim())
            out.append(self.query_text, style=theme.accent())
        if self.marks:
            out.append(f"{gap}{self._marks_label()}", style=theme.accent())
        if self.show_others:
            out.append(f"{gap}others shown", style=theme.accent())
        out.append(f"{gap if compact else '      '}{self.shown} of {self.total} jobs", style=theme.muted())
        return out

    # ------------------------------------------------------------------ preview
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_preview()
        self._schedule_peek()

    def _schedule_peek(self) -> None:
        job = self.current_job()
        key = str(job.key) if job is not None else None
        if key == self._dwell_key and self._dwell is not None:
            return
        if self._dwell is not None:
            self._dwell.stop()
            self._dwell = None
        self._dwell_key = key
        if job is None or self.preview.mode == "hidden":
            return
        if self.PEEK_DWELL <= 0:
            self._peek_now()
        else:
            self._dwell = self.set_timer(self.PEEK_DWELL, self._peek_now)

    def _peek_now(self) -> None:
        self._dwell = None
        job = self.current_job()
        if job is None or str(job.key) != self._dwell_key:
            return
        if not self.display or self.screen is not self.app.screen or self.preview.mode == "hidden":
            self._dwell_key = None     # the user left the Jobs view during the dwell: no remote peek for it
            return
        ref = self.rf.log_ref(job)
        if ref is None or ref.path is None:
            return
        cached = self._peeks.get((job.key.host, ref.path))
        if cached is not None and self.rf.clock.monotonic() - cached[1] < self.PEEK_REFRESH:
            return
        self._peek_pending = (job.key.host, ref.path)
        self._update_preview()
        self.fetch_peek(job.key.host, ref)

    @work(exit_on_error=False, group="peek", exclusive=True)
    async def fetch_peek(self, host: str, ref: Any) -> None:
        try:
            peek = await self.rf.service.peek_log(host, ref)
        except Exception as exc:   # a peek failure is a preview note, never an app error
            peek = LogPeek((), None, None, self.rf.now(), error=f"log preview failed: {exc}")
        self._peeks[(host, ref.path)] = (peek, self.rf.clock.monotonic())
        if self._peek_pending == (host, ref.path):
            self._peek_pending = None
        self._update_preview()

    def _update_preview(self) -> None:
        preview = self.preview
        if preview.mode == "hidden":
            return
        row = self.table.current_row()
        if row is None:
            text = "no jobs match" if self.total else "no jobs in scope"
            if self.total and self.query_text:          # the filter is the cause (and `f` would type into it)
                text = f"no jobs match /{self.query_text} — esc clears the filter"
                if self.preset != "all":
                    text += f" (preset: {self.preset})"
            elif self.total and self.preset != "all":
                text += f" — f cycles the state preset (now: {self.preset})"
            preview.show_empty(text)
            return
        if isinstance(row, GroupRow):
            preview.show_text(Text(row.label, style="bold"),
                              [Text(f"{len(row.jobs)} jobs · enter folds / unfolds", style=theme.muted())])
            return
        job = row
        ref = self.rf.log_ref(job)
        peek = None
        if ref is not None and ref.path is not None:
            cached = self._peeks.get((job.key.host, ref.path))
            peek = cached[0] if cached else None
        state = self.rf.store.hosts.get(job.key.host)
        caps = state.snapshot.caps if state is not None and state.snapshot is not None else None
        ui = self.rf.service.config.ui
        preview.show_job(job, caps=caps, now=self.rf.now(), peek=peek,
                         peek_pending=self._peek_pending is not None, thresholds=ui.gpu_util_thresholds,
                         mem_thresholds=ui.gpu_mem_thresholds)

    # ------------------------------------------------------------------ selection
    def current_job(self) -> Job | None:
        row = self.table.current_row() if self.is_mounted else None
        if isinstance(row, Job):
            fresh = self.rf.store.job(row.key)
            return cast(Job, fresh) if fresh is not None else row
        return None

    def _marks_label(self) -> str:
        hidden = len(self.marks - self.visible_keys)
        return f"{len(self.marks)} marked" + (f" ({hidden} hidden — not acted on)" if hidden else "")

    def action_targets(self) -> list[JobKey]:
        """Marked rows that are visible (a filter or preset that hides a marked job also keeps actions off it),
        else the cursor row."""
        marked = self.marks & self.visible_keys
        if marked:
            keys = []
            for k in sorted(marked):
                host, _, jid = k.partition(":")
                keys.append(JobKey(host, jid))
            return keys
        job = self.current_job()
        return [job.key] if job is not None else []

    def clear_marks(self) -> bool:
        if not self.marks:
            return False
        self.marks.clear()
        self.refresh_data()
        return True

    # ------------------------------------------------------------------ filter
    def open_filter(self) -> None:
        bar = self.query_one(FilterBar)
        bar.display = True
        bar.focus()

    def filter_active(self) -> bool:
        return bool(self.query_text) or self.query_one(FilterBar).has_focus

    def set_ended_since(self, since: datetime | None) -> None:
        """Digest `f`: show the own jobs that ended after ``since`` (None: back to the preset)."""
        self.ended_since = since
        self.refresh_data()
        self.table.focus()

    def clear_filter(self) -> bool:
        bar = self.query_one(FilterBar)
        if self.ended_since is not None and not self.query_text and not bar.display:
            self.set_ended_since(None)
            return True
        if not self.query_text and not bar.display:
            return False
        bar.value = ""
        self._set_query("")
        bar.display = False
        self.table.focus()
        return True

    def _set_query(self, text: str) -> None:
        if text == self.query_text:
            return
        self.query_text = text
        self.parsed = query.parse(text)
        self.rf.save_ui_state(filter=text)
        self.refresh_data()

    def on_filter_bar_filter_changed(self, event: FilterBar.FilterChanged) -> None:
        self._set_query(event.text)

    def on_filter_bar_filter_done(self, event: FilterBar.FilterDone) -> None:
        bar = self.query_one(FilterBar)
        if event.cleared or not bar.value:
            bar.display = False
        self.table.focus()

    # ------------------------------------------------------------------ table events
    def on_click_table_activated(self, event: JobTable.Activated) -> None:
        self.action_open()

    def on_click_table_context_menu(self, event: JobTable.ContextMenu) -> None:
        self.rf.action_action_menu()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        col = COLUMNS.get(str(event.column_key.value))
        if col is None or col.sort is None:
            return
        if col.sort == self.sort:
            self.reverse = not self.reverse
        else:
            self.sort, self.reverse = col.sort, False
        self._sort_changed()

    # ------------------------------------------------------------------ actions (keymap "jobs")
    def action_cursor_down(self) -> None:
        self.table.action_cursor_down()

    def action_cursor_up(self) -> None:
        self.table.action_cursor_up()

    def action_first(self) -> None:
        self.table.action_first_row()

    def action_last(self) -> None:
        self.table.action_last_row()

    def action_half_page(self, direction: int) -> None:
        self.table.action_half_page(direction)

    def action_open(self) -> None:
        row = self.table.current_row()
        if isinstance(row, GroupRow):
            self.folded ^= {row.key}
            self.refresh_data()
        elif isinstance(row, Job):
            self.rf.open_job(row.key)

    def action_mark(self) -> None:
        job = self.current_job()
        if job is None:
            return
        key = str(job.key)
        self.marks ^= {key}
        self.refresh_data()
        self.table.action_cursor_down()

    def action_mark_all(self) -> None:
        visible = {str(r.key) for r in self.table.rows_by_key.values() if isinstance(r, Job)}
        if visible and visible <= self.marks:
            self.marks -= visible
        else:
            self.marks |= visible
        self.refresh_data()

    def action_mark_array(self) -> None:
        """``A``: mark every visible row of the cursor row's array (summary rows included), so cancel / hold act on
        the whole array; again (all marked) clears them. Rows a filter or a folded group hides stay unmarked."""
        job = self.current_job()
        if job is None:
            return
        if not job.array_job_id:
            self.rf.toast(f"{job.key.label} is not part of an array", severity="warning")
            return
        members = {str(r.key) for r in self.table.rows_by_key.values()
                   if isinstance(r, Job) and r.key.host == job.key.host and r.array_job_id == job.array_job_id}
        members &= self.visible_keys
        if not members:
            return
        name = f"array {job.array_job_id} on {job.key.host}"
        if members <= self.marks:
            self.marks -= members
            self.rf.toast(f"unmarked {len(members)} rows of {name}")
        else:
            self.marks |= members
            self.rf.toast(f"marked {len(members)} rows of {name} — ^k / x act on all of them")
        self.refresh_data()

    def action_next_preset(self) -> None:
        self.preset = query.next_preset(self.preset)
        self.rf.save_ui_state(preset=self.preset)
        self.refresh_data()

    def set_preset(self, preset: Preset) -> None:
        self.preset = preset
        self.rf.save_ui_state(preset=preset)
        self.refresh_data()

    def action_next_sort(self) -> None:
        self.sort, self.reverse = query.next_sort(self.sort), False
        self._sort_changed()

    def action_reverse_sort(self) -> None:
        self.reverse = not self.reverse
        self._sort_changed()

    def set_sort(self, column: SortColumn) -> None:
        self.sort, self.reverse = column, False
        self._sort_changed()

    def _sort_changed(self) -> None:
        self.rf.save_ui_state(sort=self.sort, reverse=self.reverse)
        self.refresh_data()

    def action_next_group(self) -> None:
        self.set_group(GROUPS[(GROUPS.index(self.group) + 1) % len(GROUPS)])

    def set_group(self, group: str) -> None:
        self.group = group
        self.folded.clear()
        self.rf.save_ui_state(group=group)
        self.refresh_data()

    def action_toggle_users(self) -> None:
        self.show_others = not self.show_others
        self.rf.others_visible = self.show_others
        self.rf.toast("showing other users' jobs" if self.show_others else "showing only my jobs")
        self.refresh_data()
        for header in self.screen.query(FleetHeader):
            header.refresh_content()

    def counts_line(self) -> dict[StateGroup, int]:
        out = {g: 0 for g in StateGroup}
        for r in self.table.rows_by_key.values():
            if isinstance(r, Job):
                out[r.state.group] += 1
        return out
