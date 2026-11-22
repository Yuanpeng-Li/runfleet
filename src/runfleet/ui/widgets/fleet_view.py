"""[WP3] Fleet overview: one card per host.

Cards (mockup A) when the view is >= 100 columns wide (2 per row, 3 per row >= 180), a one-row-per-host table
(mockup B) below that. Selection survives refreshes and the cards/table switch; ``space``/click scopes the app to
the selected host, ``enter``/double-click opens its host screen.

A tall view gets roomy cards (tier T1, :mod:`runfleet.ui.fleet_layout`): the row heights follow from the view size and
each card's shape only (never from the data), each card lists the jobs that need the user, and the fleet panel fills
the empty grid cell. ui.json ``fleet_layout = "compact"`` keeps today's cards. A click on a job row of a card or on an
activity row of the panel opens that job, one on its free part (or ``g``, in any layout) says where the free GPUs are
(:class:`~runfleet.ui.screens.free_gpus.FreeGpusScreen`); ``a`` lists what needs the user across the fleet, from the same rankings, in
any layout (:class:`~runfleet.ui.screens.attention.AttentionScreen`). Nothing here polls anything.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.widget import Widget
from textual.widgets import DataTable, Static

from runfleet.core.store import HostState
from runfleet.models import ConnState, JobKind, StateGroup
from runfleet.ui import fleet_layout, keymap, render, theme
from runfleet.ui.fleet_layout import CARDS_MIN_WIDTH, HighWater, Ranked, columns_for, inner_width
from runfleet.ui.widgets.fleet_panel import FleetPanel, panel_body
from runfleet.ui.widgets.job_table import ClickTable
from runfleet.ui.widgets.host_card import HostCard, card_body, card_title, is_cluster, old_reading
from runfleet.ui.query import ended_at
from runfleet.util import humanize

__all__ = ["CARDS_MIN_WIDTH", "FleetView", "columns_for", "host_row", "fleet_layout_choice"]

LAYOUT_CHOICES = ("auto", "compact")
AUTO_SHRINK_S = 120.0            # Auto: a card's rows shrink only after it needed fewer lines for this long

TABLE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("mark", "", 1), ("num", "#", 1), ("host", "HOST", 11), ("kind", "KIND", 5), ("state", "STATE", 8),
    ("jobs", "JOBS", 10), ("gpus", "GPUS", 5), ("util", "UTIL", 16), ("age", "AGE", 3),
)


def fleet_layout_choice(ui_state: Any) -> str:
    """ui.json ``fleet_layout``: ``auto`` (roomy cards when the view has room) or ``compact`` (always today's)."""
    choice = ui_state.get("fleet_layout") if isinstance(ui_state, dict) else None
    return choice if choice in LAYOUT_CHOICES else "auto"


class FleetTable(ClickTable):
    """Narrow-terminal host table; enter / double-click opens the host."""


@dataclass(slots=True)
class _Drag:
    """A press on a border between two rows (``axis`` "y") or two columns ("x"): what it moves and from where."""
    axis: str
    boundary: int
    start: int                     # screen y (rows) or x (columns) of the press
    sizes: list[int]               # outer row heights / column widths when pressed
    widget: Widget                 # what was pressed (a press that never moves selects it, like a click)
    base: Any = None               # the Sizing when pressed (a drag in Auto starts from Auto's sizes, not older ones)
    plan: Any = None               # the plan when pressed
    moved: bool = False


class FleetCards(VerticalScroll):
    """The scrolling grid of cards. A click on what is not a card (the fleet panel, a gap) leaves the focus on the
    selected card: only the cards take focus from the mouse.

    A card's bottom border (or the panel's) moves the border between its row and the next, its right border (or the
    left border of the card beside it) the border between its column and the next: press, drag, release. The sizes
    are kept in ui.json (``fleet_sizes``); ``=`` goes back to the default ones."""

    FOCUS_ON_CLICK = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._drag: _Drag | None = None
        self._pressed: Widget | None = None              # a border pressed without a move: its click is the card's

    def _view(self) -> FleetView:
        return self.query_ancestor(FleetView)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1:
            return
        self._end_drag()                                  # a drag whose release never came
        drag = self._view().drag_at(event.screen_x, event.screen_y)
        if drag is None:
            return
        self._drag = drag
        self.capture_mouse()
        self.screen.clear_selection()                     # the press began a text selection: a drag is not one
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        drag = self._drag
        if drag is None:
            return
        event.stop()
        if not event.button:                              # the button is up: the release was lost (outside, a screen)
            self._end_drag()
            return
        self.screen.clear_selection()
        at = event.screen_y if drag.axis == "y" else event.screen_x
        delta = int(at - drag.start)
        if delta or drag.moved:
            drag.moved = True
            self._view().drag_to(drag, delta)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        drag = self._drag
        if drag is None:
            return
        event.stop()
        self._end_drag()
        if not drag.moved and isinstance(drag.widget, HostCard):   # a press on a border that never moved: selects
            drag.widget.focus()
            self._pressed = drag.widget

    def on_mouse_release(self, _event: events.MouseRelease) -> None:
        self._end_drag()                                  # the capture went elsewhere (a screen was pushed)

    def _end_drag(self) -> None:
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if self.app.mouse_captured is self:
            self.release_mouse()
        self.screen.clear_selection()
        if drag.moved:
            self._view().drag_done()

    def on_click(self, event: events.Click) -> None:
        pressed, self._pressed = self._pressed, None
        if isinstance(pressed, HostCard):                 # the click of a border press (the capture had it)
            event.stop()
            if event.chain >= 2:
                self.app.action_open_host(pressed.host)  # type: ignore[attr-defined]


def empty_text() -> Text:
    """The empty-fleet message, in the current palette (rebuilt after a theme switch)."""
    return Text.assemble(
        ("No hosts in the fleet.\n\n", "bold"),
        ("Press ", theme.muted()), (",", "bold"), (" (Settings) to pick hosts from ~/.ssh/config or add an ssh "
                                                  "target; they are saved and polled at once.", theme.muted()))


class FleetView(Widget, can_focus=False):
    BINDINGS = keymap.bindings("fleet")
    DEFAULT_CSS = """
    FleetView { height: 1fr; }
    FleetView > #fleet-empty { padding: 2 4; height: auto; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.selected: str | None = None
        self.mode = "cards"
        self._syncing = False
        self.plan: fleet_layout.Plan | None = None
        self._hw: dict[str, HighWater] = {}               # shape high-water marks, per host, this session
        # host -> (jobs, minute, window, expiry, ranked list, its distinct name starts)
        self._ranks: dict[str, tuple[object, int, float, Any, list[Ranked], dict[str, int]]] = {}
        self._memo = fleet_layout.Memo()                   # the fleet panel's fleet-wide scans
        self.sizing = fleet_layout.Sizing()                # Auto / dragged sizes (ui.json fleet_sizing, fleet_sizes)
        self._need: dict[str, int] = {}                    # content lines each card (and "*": the panel) last needed
        self._held: dict[str, tuple[int, float | None]] = {}   # Auto's view of them: (lines, lower since)
        self._grid: tuple[int, str] | None = None          # the grid columns last applied

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        yield Static(empty_text(), id="fleet-empty")
        with FleetCards(id="fleet-cards"):
            with Grid(id="fleet-grid"):
                for i, name in enumerate(self.rf.host_names(), start=1):
                    yield HostCard(name, i)
                yield FleetPanel()
        table = FleetTable(id="fleet-table")
        yield table

    def on_mount(self) -> None:
        table = self.query_one(FleetTable)
        for key, label, width in TABLE_COLUMNS:
            table.add_column(label, key=key, width=width)
        names = self.rf.host_names()
        self.selected = self.rf.scope or (names[0] if names else None)
        self.sizing = fleet_layout.sizing_from(self.rf.ui_state)
        self._apply_mode(force=True)
        self.refresh_data()

    def forget_rendered(self) -> None:
        """The palette changed: rebuild the empty-fleet message (the cards and the panel follow refresh_data)."""
        self.query_one("#fleet-empty", Static).update(empty_text())

    def on_resize(self, event: events.Resize) -> None:
        self._apply_mode()
        self.refresh_data()

    # ------------------------------------------------------------------ mode
    def _apply_mode(self, force: bool = False) -> None:
        width = self.size.width or self.app.size.width
        mode = "cards" if width >= CARDS_MIN_WIDTH else "table"
        empty = not self.rf.host_names()
        self.query_one("#fleet-empty").display = empty
        if empty:
            self.query_one("#fleet-cards").display = False
            self.query_one(FleetTable).display = False
            return
        if mode == self.mode and not force:
            return
        self.mode = mode
        self.query_one("#fleet-cards").display = mode == "cards"
        self.query_one(FleetTable).display = mode == "table"
        if mode == "cards":
            self._set_grid(columns_for(width), ())
        if self.screen.focused is None or self.has_focus_within:
            self.focus_selection()

    def _set_grid(self, cols: int, widths: tuple[int, ...]) -> None:
        """The grid's column count and widths (dragged widths in cells, else equal columns)."""
        spec = (cols, " ".join(str(w) for w in widths) if widths else " ".join(["1fr"] * cols))
        if spec == self._grid:
            return
        self._grid = spec
        grid = self.query_one("#fleet-grid", Grid)
        grid.styles.grid_size_columns = cols
        grid.styles.grid_columns = spec[1]

    def cols(self) -> int:
        """Cards per grid row now."""
        if self.plan is not None and self.plan.mode == "cards":
            return self.plan.cols
        return columns_for(self.size.width or 120)

    def focus_selection(self) -> None:
        if not self.display:
            return
        if self.mode == "cards":
            card = self._card(self.selected)
            if card is not None:
                card.focus(scroll_visible=True)
        else:
            table = self.query_one(FleetTable)
            table.focus()
            if self.selected is not None:
                try:
                    table.move_cursor(row=table.get_row_index(self.selected))
                except Exception:   # row not present yet
                    pass

    def _card(self, host: str | None) -> HostCard | None:
        for card in self.query(HostCard):
            if card.host == host:
                return card
        return None

    # ------------------------------------------------------------------ hosts added / removed / reordered
    def _hosts_changed(self) -> bool:
        cards = [(c.host, c.index) for c in self.query(HostCard)]
        return cards != [(n, i) for i, n in enumerate(self.rf.host_names(), start=1)]

    async def _sync(self) -> None:
        try:
            await self.sync_hosts()
        finally:
            self._syncing = False
        if self._hosts_changed():             # changed again while rebuilding
            self.refresh_data()

    async def sync_hosts(self) -> None:
        """Rebuild the cards / table rows after Settings changed the host set or order (number = position)."""
        if not self.is_mounted or not self._hosts_changed():
            return
        names = self.rf.host_names()
        grid = self.query_one("#fleet-grid", Grid)
        await grid.remove_children()
        await grid.mount_all([*(HostCard(n, i) for i, n in enumerate(names, start=1)), FleetPanel()])
        self._hw.clear()                         # shapes are re-learnt for the new host set / order
        self._ranks.clear()
        self._memo.clear()
        self._need.clear()
        self._held.clear()
        self.query_one(FleetTable).clear()
        if self.selected not in names:
            self.selected = names[0] if names else None
        self._apply_mode(force=True)
        self.refresh_data()

    # ------------------------------------------------------------------ data
    def refresh_data(self) -> None:
        if not self.is_mounted:
            return
        if self._hosts_changed():
            if not self._syncing:            # one rebuild at a time; it re-checks the host list when done
                self._syncing = True
                self.run_worker(self._sync(), group="fleet-sync", exit_on_error=False)
            return
        now = self.rf.now()
        hosts: dict[str, HostState] = self.rf.store.hosts
        if self.mode == "cards":
            self._refresh_cards(hosts, now)
        else:
            self._refresh_table(hosts, now)

    def roomy_enabled(self) -> bool:
        return fleet_layout_choice(self.rf.ui_state) != "compact"

    def _ranked(self, state: HostState, now: Any, window: float) -> tuple[list[Ranked], dict[str, int]] | None:
        """The card's ranked list and its names' distinct starts, recomputed when its jobs change, a minute passes
        (time cells move per minute) or a failure crosses the 4 h / ``window`` line (:func:`fleet_layout.rank_expiry`:
        to the second, like the card's ``my jobs`` counts)."""
        snap = state.snapshot
        if snap is None:
            return None
        minute = int(now.timestamp() // 60)
        hit = self._ranks.get(state.config.name)
        if (hit is not None and hit[0] is snap.jobs and hit[1] == minute and hit[2] == window
                and (hit[3] is None or now < hit[3])):
            return hit[4], hit[5]
        if is_cluster(state):        # resubmitted failures rank last, never "needs you" (links once per jobs mapping)
            lineage = fleet_layout.lineage_of(state.config.name, snap, self._memo)
            ranked = fleet_layout.rank_slurm(snap.jobs.values(), now, window, lineage)
        else:
            ranked = fleet_layout.rank_ws(snap.jobs.values(), now, window)
        starts = fleet_layout.ranked_starts(ranked)
        self._ranks[state.config.name] = (snap.jobs, minute, window, fleet_layout.rank_expiry(ranked, window), ranked,
                                          starts)
        return ranked, starts

    def _refresh_cards(self, hosts: dict[str, HostState], now: Any) -> None:
        width = self.size.width or 120
        general = self.rf.service.config.general
        ui = self.rf.service.config.ui
        cards = list(self.query(HostCard))
        plan, shapes = self._plan(hosts, cards, width, now)
        self.plan = plan
        cols = plan.cols
        self._set_grid(cols, plan.widths if plan.tier >= 1 else ())
        if plan.tier >= 1:
            self._refresh_roomy(hosts, now, cards, shapes, plan)
            return
        inner = plan.inner
        self._show_panel(None, hosts, now)
        rendered = []
        for card in cards:
            state = hosts[card.host]
            lines, _keys = card_body(state, now, width=inner, window=general.recent_window,
                                     util_t=ui.gpu_util_thresholds, mem_t=ui.gpu_mem_thresholds)
            rendered.append((card, state, lines))
        for row_start in range(0, len(rendered), cols):
            row = rendered[row_start:row_start + cols]
            height = max(len(lines) for _, _, lines in row)
            for card, state, lines in row:
                problem = state.status.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.UNREACHABLE)
                card.set_lines(card_title(state, card.index, now, width=inner), lines, height,
                               stale=state.stale(now), problem=problem)
        for card in cards:
            card.set_class(card.host == self.rf.scope, "-scoped")

    def _plan(self, hosts: dict[str, HostState], cards: list[HostCard], width: int,
              now: Any) -> tuple[fleet_layout.Plan, list[fleet_layout.Shape]]:
        """The plan for the current sizing (default, dragged, Auto) and each card's shape at its own width."""
        height = self.size.height
        enabled = self.roomy_enabled()
        sizing = self.sizing

        def shape_at(i: int, inner: int) -> fleet_layout.Shape:
            host = cards[i].host
            return fleet_layout.shape_of(hosts[host], inner, self._hw.setdefault(host, HighWater()))

        if sizing.mode == "auto" and enabled:
            demands = [self._held_need(c.host, self._need_of(hosts[c.host], i, shape_at, width, now), now)
                       for i, c in enumerate(cards)]
            panel = self._held_need("*", self._need.get("*", fleet_layout.PANEL_MIN), now)
            keep = self.plan.cols if self.plan is not None and self.plan.sizing == "auto" else None
            got = fleet_layout.auto_plan(width, self.size.height, shape_at, demands, panel, enabled, keep)
            if got is not None:
                return got, [shape_at(i, got.inner) for i in range(len(cards))]
        cols = columns_for(width)
        if enabled and sizing.ncols and inner_width(width, sizing.ncols) >= fleet_layout.ROOMY_MIN_INNER:
            cols = sizing.ncols
        widths = fleet_layout.col_widths(width, cols, sizing.cols.get(cols)) if enabled else None
        shapes = [shape_at(i, widths[i % cols] - 4 if widths else inner_width(width, cols)) for i in range(len(cards))]
        got = fleet_layout.plan(width, height, shapes, enabled, cols=cols, widths=widths, rows_custom=sizing.rows)
        if got.tier < 1 and cols != columns_for(width):     # the dragged column count is not roomy here: the width's
            cols = columns_for(width)
            widths = fleet_layout.col_widths(width, cols, sizing.cols.get(cols)) if enabled else None
            shapes = [shape_at(i, widths[i % cols] - 4 if widths else inner_width(width, cols))
                      for i in range(len(cards))]
            got = fleet_layout.plan(width, height, shapes, enabled, cols=cols, widths=widths, rows_custom=sizing.rows)
        return got, shapes

    def _need_of(self, state: HostState, i: int, shape_at: Any, width: int, now: Any) -> int:
        """Content lines a card needs to show everything: what it took last time, else its summary and its list."""
        got = self._need.get(state.config.name)
        if got is not None:
            return got
        shape = shape_at(i, inner_width(width, columns_for(width)))
        ranked = self._ranked(state, now, self.rf.service.config.general.recent_window)
        if ranked is None or not shape.cap:
            return shape.t0
        return shape.t1 + 1 + fleet_layout.list_need(ranked[0])

    def _held_need(self, key: str, value: int, now: Any) -> int:
        """Auto's demand: it grows at once and shrinks only after it stayed lower for :data:`AUTO_SHRINK_S`, so a job
        that ends does not make the rows jump back and forth."""
        t = now.timestamp()
        cur = self._held.get(key)
        if cur is None or value >= cur[0]:
            self._held[key] = (value, None)
            return value
        since = cur[1] if cur[1] is not None else t
        if t - since >= AUTO_SHRINK_S:
            self._held[key] = (value, None)
            return value
        self._held[key] = (cur[0], since)
        return cur[0]

    def _refresh_roomy(self, hosts: dict[str, HostState], now: Any, cards: list[HostCard],
                       shapes: list[fleet_layout.Shape], plan: fleet_layout.Plan) -> None:
        """T1: every card padded to its row height; the list region of each card holds every job it lists and
        scrolls under the summary when they do not fit."""
        general = self.rf.service.config.general
        ui = self.rf.service.config.ui
        watched = self.rf.alerts.is_watched
        for i, (card, shape) in enumerate(zip(cards, shapes, strict=True)):
            height = plan.heights[i // plan.cols]
            inner = plan.col_inner(i % plan.cols)
            state = hosts[card.host]
            ranked, starts = self._ranked(state, now, general.recent_window) or (None, None)
            list_rows = None
            if shape.cap:
                list_rows = fleet_layout.full_rows(ranked or [], max(0, height - shape.t1))
            lines, keys = card_body(state, now, width=inner, window=general.recent_window,
                                    util_t=ui.gpu_util_thresholds, mem_t=ui.gpu_mem_thresholds,
                                    list_rows=list_rows, wide=inner >= fleet_layout.WIDE_INNER,
                                    fixed=shape.fixed, ranked=ranked, starts=starts, watched=watched)
            self._need[card.host] = len(lines)
            problem = state.status.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.UNREACHABLE)
            card.set_lines(card_title(state, card.index, now, width=inner), lines, height,
                           stale=state.stale(now), problem=problem, keys=keys,
                           pinned=shape.t1 if shape.cap else 0)
            card.set_class(card.host == self.rf.scope, "-scoped")
        self._show_panel(plan, hosts, now)

    def _show_panel(self, plan: fleet_layout.Plan | None, hosts: dict[str, HostState], now: Any) -> None:
        try:
            panel = self.query_one(FleetPanel)
        except Exception:        # rebuilding (sync_hosts)
            return
        spec = plan.panel if plan is not None else None
        if spec is None or plan is None:
            panel.display = False
            return
        first = 0 if spec.below else plan.cols - spec.span
        inner = plan.span_inner(first, spec.span)
        alerts = self.rf.alerts
        lines, keys = panel_body(hosts, now, rows=spec.rows, width=inner, wide=inner >= fleet_layout.WIDE_INNER,
                                 window=self.rf.service.config.general.recent_window, history=alerts.history,
                                 digest_since=alerts.digest_since, memo=self._memo,
                                 rate=getattr(self.rf, "multiplier", 1.0), watch_events=alerts.watch_events,
                                 scroll=True)
        used = len(lines)
        while used and keys[used - 1] is None and not lines[used - 1].plain:   # the padding
            used -= 1
        self._need["*"] = used
        pinned = 0
        while pinned < len(keys) and isinstance(keys[pinned], str):              # the free part stays put
            pinned += 1
        panel.set_lines(lines[:max(used, spec.rows)], spec.span, spec.rows, keys, pinned=pinned)
        panel.display = True

    def _refresh_table(self, hosts: dict[str, HostState], now: Any) -> None:
        table = self.query_one(FleetTable)
        thresholds = self.rf.service.config.ui.gpu_util_thresholds
        present = {str(k.value) for k in table.rows}
        for i, (name, state) in enumerate(hosts.items(), start=1):
            row = host_row(state, i, now, thresholds, scoped=self.rf.scope == name,
                           window=self.rf.service.config.general.recent_window)
            if name in present:
                for (key, _l, _w), value in zip(TABLE_COLUMNS, row, strict=True):
                    table.update_cell(name, key, value)
            else:
                table.add_row(*row, key=name)
        if self.selected is not None and self.selected in hosts:
            idx = list(hosts).index(self.selected)
            if table.cursor_row != idx:
                table.move_cursor(row=idx)

    # ------------------------------------------------------------------ selection / actions
    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        widget = event.widget
        if isinstance(widget, HostCard):
            self.selected = widget.host

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.selected = str(event.row_key.value)

    def on_click_table_activated(self, event: ClickTable.Activated) -> None:
        self.selected = event.key
        self.action_open_host()

    def current_host(self) -> str | None:
        return self.selected

    def action_move(self, direction: str) -> None:
        names = self.rf.host_names()
        if not names:
            return
        idx = names.index(self.selected) if self.selected in names else 0
        if self.mode == "cards":
            cols = self.cols()
            step = {"left": -1, "right": 1, "up": -cols, "down": cols}[direction]
        else:
            step = {"left": -1, "right": 1, "up": -1, "down": 1}[direction]
        new = idx + step
        if 0 <= new < len(names):
            self.selected = names[new]
            self.focus_selection()

    def action_open_host(self) -> None:
        if self.selected is not None:
            self.rf.action_open_host(self.selected)

    def action_scope_host(self) -> None:
        if self.selected is not None:
            self.rf.set_scope(None if self.rf.scope == self.selected else self.selected)

    # ------------------------------------------------------------------ sizes: drag, reset, Auto
    def drag_at(self, x: int, y: int) -> _Drag | None:
        """The border under the screen cell ``x``, ``y`` that a drag would move (None: not one)."""
        plan = self.plan
        if plan is None or plan.tier < 1 or self.mode != "cards":
            return None
        cards: list[Widget] = list(self.query(HostCard))
        n = len(cards)
        try:
            panel: Widget | None = self.query_one(FleetPanel)
        except Exception:
            panel = None
        cols = plan.cols
        for i, w in enumerate([*cards, panel]):
            if w is None or not w.display:
                continue
            r = w.region
            if not r.contains(x, y):
                continue
            if i < n:
                row, first, last = i // cols, i % cols, i % cols
            elif plan.panel is not None and plan.panel.below:
                row, first, last = len(plan.heights), 0, cols - 1
            else:
                row, first, last = len(plan.heights) - 1, cols - (plan.panel.span if plan.panel else 1), cols - 1
            widths = list(plan.widths) or self._screen_widths(cards, cols, plan)
            if y == r.bottom - 1 and row < plan.row_count - 1:
                return _Drag("y", row, y, plan.outer_heights(), w, self.sizing, plan)
            if x == r.right - 1 and last < cols - 1:
                return _Drag("x", last, x, widths, w, self.sizing, plan)
            if x == r.x and first > 0:
                return _Drag("x", first - 1, x, widths, w, self.sizing, plan)
            return None
        return None

    @staticmethod
    def _screen_widths(cards: list[Widget], cols: int, plan: fleet_layout.Plan) -> list[int]:
        """Column widths as drawn (the first row's cards), so a drag starts from what is on screen."""
        got = [c.region.width for c in cards[:cols]]
        if len(got) == cols and all(got) and sum(got) == plan.width - 1:
            return got
        return list(fleet_layout.equal_widths(plan.width, cols))

    def drag_to(self, drag: _Drag, delta: int) -> None:
        """The pressed border moved ``delta`` from where it was pressed: resize now (kept on release). Starts from the
        sizes on screen when pressed: a drag in Auto takes Auto's rows and columns as the user's (older drags of this
        grid are replaced), and keeps Auto's column count only when it is not the one the width picks anyway."""
        plan: fleet_layout.Plan | None = drag.plan or self.plan
        base: fleet_layout.Sizing = drag.base or self.sizing
        if plan is None or plan.tier < 1:
            return
        cols = plan.cols
        height = max(1, self.size.height)
        key = fleet_layout.row_key(cols, len(self.query(HostCard)))
        rows = dict(base.rows)
        widths = dict(base.cols)
        if base.mode == "auto" and plan.sizing == "auto":
            rows[key] = tuple(h / height for h in plan.outer_heights())
            widths.pop(cols, None)                          # Auto's columns are equal
        if drag.axis == "y":
            outer = fleet_layout.drag_rows(drag.sizes, drag.boundary, delta)
            rows[key] = tuple(h / height for h in outer)
        else:
            new = fleet_layout.drag_cols(drag.sizes, drag.boundary, delta)
            total = sum(new) or 1
            widths[cols] = tuple(w / total for w in new)
            if key not in rows:                   # keep the rows as they are now
                rows[key] = tuple(h / height for h in plan.outer_heights())
        ncols = base.ncols if plan.sizing != "auto" else None
        if plan.sizing == "auto" and cols != columns_for(plan.width):
            ncols = cols
        self.sizing = fleet_layout.Sizing("default", ncols, widths, rows)
        self.refresh_data()

    def drag_done(self) -> None:
        self._save_sizing()

    def _save_sizing(self) -> None:
        """Keep the sizes in ui.json, and use them as saved (rounded), so the next start looks the same."""
        state = {"fleet_sizing": self.sizing.mode, "fleet_sizes": self.sizing.to_state()}
        self.rf.save_ui_state(**state)
        saved = fleet_layout.sizing_from(state)
        if saved != self.sizing:
            self.sizing = saved
            self.refresh_data()

    def set_sizing(self, sizing: fleet_layout.Sizing) -> None:
        """Palette / keys: switch Auto, forget the drags; applied now and kept in ui.json."""
        self.sizing = sizing
        self._held.clear()
        self._save_sizing()
        for card in self.query(HostCard):
            card.list_home()
        self.refresh_data()

    def action_reset_layout(self) -> None:
        """``=``: the default sizes (dragged borders forgotten, Auto off)."""
        self.set_sizing(fleet_layout.Sizing())
        self.app.toast("Fleet: default layout (dragged sizes forgotten)", timeout=3, reply=True)  # type: ignore[attr-defined]

    def action_toggle_auto_layout(self) -> None:
        """``L``: Auto on / off (off: the dragged sizes, or the default ones)."""
        auto = self.sizing.mode != "auto"
        self.set_sizing(dataclasses.replace(self.sizing, mode="auto" if auto else "default"))
        if auto and not self.roomy_enabled():
            msg = "Fleet: Auto layout is on, but compact cards are chosen (palette › Fleet: roomy cards)"
        elif auto:
            msg = "Fleet: Auto layout: rows follow what each card lists (drag a border or = to leave it)"
        else:
            msg = "Fleet: Auto layout off" + (" (your dragged sizes)" if self.sizing.dragged else "")
        self.app.toast(msg, timeout=3, reply=True)  # type: ignore[attr-defined]

    def action_scroll_card(self, pages: int) -> None:
        """pgup / pgdn: scroll the selected card's list by a screen (less one line)."""
        card = self._card(self.selected)
        if card is not None and self.mode == "cards":
            card.scroll_lines(pages * max(1, card.page_rows() - 1))

    def attention(self, now: Any) -> list[fleet_layout.Attention]:
        """What needs the user across the fleet (every host, whatever the scope or layout), from the cards' cached
        rankings (:meth:`_ranked`)."""
        window = self.rf.service.config.general.recent_window
        per_host = []
        for name, state in self.rf.store.hosts.items():
            got = self._ranked(state, now, window)
            if got is not None:
                per_host.append((name, is_cluster(state), got[0]))
        return fleet_layout.attention(per_host)

    def action_attention(self) -> None:
        """``a``: the "needs you" picker; enter opens the job (a toast when it is gone by then)."""
        from runfleet.ui.screens.attention import AttentionScreen

        now = self.rf.now()
        hosts = self.rf.store.hosts
        colors = {name: theme.host_color(state.config, i) for i, (name, state) in enumerate(hosts.items())}
        self.app.push_screen(
            AttentionScreen(self.attention(now), now, colors, util_t=self.rf.service.config.ui.gpu_util_thresholds),
            lambda key: self.rf.open_listed_job(key) if key is not None else None)

    def on_fleet_panel_free_clicked(self, event: FleetPanel.FreeClicked) -> None:
        self.open_free(None if event.key == fleet_layout.ALL_FREE else event.key)

    def action_where_free(self) -> None:
        """``g``: the free GPUs dialog, every type."""
        self.open_free(None)

    def open_free(self, only: str | None) -> None:
        """The free GPUs dialog for one type key of the panel (None: every type); enter opens the host it names."""
        from runfleet.ui.screens.free_gpus import FreeGpusScreen

        now = self.rf.now()
        hosts = self.rf.store.hosts
        rate = getattr(self.rf, "multiplier", 1.0)

        def old(state: HostState, gpu: Any) -> bool:
            return old_reading(state, gpu, now)

        free = fleet_layout.free_by_type(hosts, now, old, self._memo, rate)
        mine = fleet_layout.my_types(hosts, now, self.rf.service.config.general.recent_window, self._memo)
        yours, others = fleet_layout.free_split(free, mine)
        keyed = {id(t): k for k, t in free.types.items()}
        order = [keyed[id(t)] for t in (*yours, *others)]
        colors = {name: theme.host_color(state.config, i) for i, (name, state) in enumerate(hosts.items())}
        self.app.push_screen(
            FreeGpusScreen(fleet_layout.free_spots(hosts, now, old, rate), order, only=only, mine=mine,
                           unknown=free.unknown, sinfo_age_s=free.sinfo_age_s, host_order=list(hosts),
                           host_colors=colors),
            self._open_free_target)

    def _open_free_target(self, target: Any) -> None:
        if target is not None:
            self.rf.open_host_at(target.host, partition=target.partition, gpu=target.gpu)


_SHORT_STATE = {ConnState.RETRYING: "retry", ConnState.NEEDS_AUTH: "auth", ConnState.CONNECTING: "conn…",
                ConnState.DEGRADED: "degr.", ConnState.UNREACHABLE: "down", ConnState.HOSTKEY: "hostkey",
                ConnState.DISABLED: "off"}


def host_row(state: HostState, index: int, now: Any, thresholds: tuple[float, float], *,
             scoped: bool = False, window: float = 86400) -> tuple[Text, ...]:
    """One mockup-B row: marker, #, HOST, KIND, STATE, JOBS, GPUS, UTIL, AGE."""
    st = state.status
    snap = state.snapshot
    stale = state.stale(now)
    cluster = is_cluster(state)
    marker = Text("!" if st.state is ConnState.DEGRADED else "~" if stale else "",
                  style=theme.warn() if st.state is ConnState.DEGRADED else theme.dim())
    name = Text(st.name, style=f"bold {theme.host_color(state.config, index - 1)}" + (" underline" if scoped else ""))
    kind = Text("slurm" if cluster else "ws" if snap is not None else "?", style=theme.muted())
    short = _SHORT_STATE.get(st.state)        # the STATE column is 8 cells
    state_text = (Text(f"{theme.CONN_GLYPH[st.state]} {short}", style=theme.conn_style(st.state)) if short
                  else render.conn_text(st.state))
    jobs = Text("–", style=theme.dim())
    gpus = Text("–", style=theme.dim())
    util = Text("n/a", style=theme.dim())
    if snap is not None:
        mine = [j for j in snap.jobs.values() if j.is_mine]
        if cluster:
            r = sum(1 for j in mine if j.state.group is StateGroup.ACTIVE)
            pd = sum(1 for j in mine if j.state.group is StateGroup.QUEUED)
            f = sum(1 for j in mine if j.state.group in (StateGroup.FAILED, StateGroup.ENDED)
                    and (now - (ended_at(j) or now)).total_seconds() <= window)
            jobs = Text(" ".join(f"{p}{n}" for p, n in (("R", r), ("PD", pd), ("F", f)) if n) or "—")
            running = [j for j in mine if j.state.group is StateGroup.ACTIVE]
            gpus = Text(str(sum(j.alloc.gpus for j in running)))
            us = [j.gpu for j in running if j.gpu is not None and j.gpu.util_mean_pct is not None]
            if us:
                total = sum(u.n_gpus for u in us) or 1
                mean = sum((u.util_mean_pct or 0) * u.n_gpus for u in us) / total
                util = render.gpu_bar(mean, 11, thresholds) + Text(humanize.pct(mean, width=5))
        else:
            tasks = [j for j in mine if j.kind is JobKind.TASK and j.state.group is StateGroup.ACTIVE]
            jobs = Text(f"{len(tasks)} task{'s' if len(tasks) != 1 else ''}")
            old = [g for g in snap.gpus if old_reading(state, g, now)]     # a hung nvidia-smi: never free
            busy = [g for g in snap.gpus if render.gpu_in_use(g) or g in old]
            gpus = Text(f"{len(busy)}/{len(snap.gpus)}" + ("~" if old else ""), style=theme.dim() if old else "")
            if snap.gpus:
                vals = [g.util_pct for g in snap.gpus if g.util_pct is not None and ((g.util_pct or 0) >= 5
                                                                                      or g.processes)]
                mean = sum(vals) / len(vals) if vals else 0.0
                util = render.gpu_bar(mean, 11, thresholds) + Text(humanize.pct(mean, width=5))
    age = Text(humanize.age((now - st.last_ok).total_seconds()) if st.last_ok else "–", style=theme.muted())
    if stale:
        for t in (jobs, gpus, util):
            t.stylize(theme.dim())
    return (marker, Text(str(index), style=theme.muted()), name, kind, state_text, jobs, gpus, util, age)
