"""[WP3] Fleet overview: one card per host.

Cards (mockup A) when the view is >= 100 columns wide (2 per row, 3 per row >= 180), a one-row-per-host table
(mockup B) below that. Selection survives refreshes and the cards/table switch; ``space``/click scopes the app to
the selected host, ``enter``/double-click opens its host screen.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.widget import Widget
from textual.widgets import DataTable, Static

from runfleet.core.store import HostState
from runfleet.models import ConnState, JobKind, StateGroup
from runfleet.ui import keymap, render, theme
from runfleet.ui.widgets.job_table import ClickTable
from runfleet.ui.widgets.host_card import HostCard, card_lines, card_title, is_cluster, old_reading
from runfleet.ui.query import ended_at
from runfleet.util import humanize

CARDS_MIN_WIDTH = 100

TABLE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("mark", "", 1), ("num", "#", 1), ("host", "HOST", 11), ("kind", "KIND", 5), ("state", "STATE", 8),
    ("jobs", "JOBS", 10), ("gpus", "GPUS", 5), ("util", "UTIL", 16), ("age", "AGE", 3),
)


def columns_for(width: int) -> int:
    return 3 if width >= 180 else 2


class FleetTable(ClickTable):
    """Narrow-terminal host table; enter / double-click opens the host."""


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

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        yield Static(Text.assemble(
            ("No hosts in the fleet.\n\n", "bold"),
            ("Press ", theme.muted()), (",", "bold"), (" (Settings) to pick hosts from ~/.ssh/config or add an ssh "
                                                      "target; they are saved and polled at once.", theme.muted())),
            id="fleet-empty")
        with VerticalScroll(id="fleet-cards"):
            with Grid(id="fleet-grid"):
                for i, name in enumerate(self.rf.host_names(), start=1):
                    yield HostCard(name, i)
        table = FleetTable(id="fleet-table")
        yield table

    def on_mount(self) -> None:
        table = self.query_one(FleetTable)
        for key, label, width in TABLE_COLUMNS:
            table.add_column(label, key=key, width=width)
        names = self.rf.host_names()
        self.selected = self.rf.scope or (names[0] if names else None)
        self._apply_mode(force=True)
        self.refresh_data()

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
            if mode == "cards":
                self.query_one("#fleet-grid", Grid).styles.grid_size_columns = columns_for(width)
            return
        self.mode = mode
        self.query_one("#fleet-cards").display = mode == "cards"
        self.query_one(FleetTable).display = mode == "table"
        if mode == "cards":
            self.query_one("#fleet-grid", Grid).styles.grid_size_columns = columns_for(width)
        if self.screen.focused is None or self.has_focus_within:
            self.focus_selection()

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
        await grid.mount_all([HostCard(n, i) for i, n in enumerate(names, start=1)])
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

    def _refresh_cards(self, hosts: dict[str, HostState], now: Any) -> None:
        width = self.size.width or 120
        cols = columns_for(width)
        inner = max(30, width // cols - 4)
        general = self.rf.service.config.general
        ui = self.rf.service.config.ui
        cards = list(self.query(HostCard))
        rendered = []
        for card in cards:
            state = hosts[card.host]
            lines = card_lines(state, now, width=inner, window=general.recent_window,
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
            cols = columns_for(self.size.width or 120)
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
