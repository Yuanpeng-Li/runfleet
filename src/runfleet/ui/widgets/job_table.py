"""[WP3] Merged, sortable job table (DataTable, diff refresh).

Column layout follows §7.4: columns have a minimum width and a priority; the lowest-priority columns are dropped
until the rest fits, NAME absorbs the slack (up to a cap, the rest goes to NODE / REASON). Refreshes are keyed
diffs: when the row order is unchanged only changed cells are rewritten (``update_cell``), otherwise the table is
rebuilt in one go and the cursor is restored by row key, so it never jumps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import DataTable
from textual.coordinate import Coordinate
from textual.widgets.data_table import RowKey

from runfleet.models import Job, JobKind, JobState, StateGroup
from runfleet.ui import query, render, theme
from runfleet.ui.query import CostFn, SortColumn, elapsed_s, ended_at, gpuh_value, left_s, su_value
from runfleet.util import cost as costs
from runfleet.util import humanize


@dataclass(frozen=True, slots=True)
class Column:
    key: str
    label: str
    width: int                   # minimum content width
    priority: int                # 1 = never dropped; "auto" columns use 0 and are added on demand
    sort: SortColumn | None = None


COLUMNS: dict[str, Column] = {c.key: c for c in (
    Column("marker", "", 2, 1),
    Column("host", "HOST", 8, 1, "host"),
    Column("id", "ID", 8, 1, "id"),
    Column("name", "NAME", 16, 1, "name"),
    Column("state", "ST", 3, 1, "state"),
    Column("elapsed", "ELAPSED", 8, 1, "elapsed"),
    Column("left", "LEFT", 9, 3, "left"),
    Column("gpus", "GPUS", 9, 3, "gpus"),
    Column("gpu_util", "GPU%", 6, 2, "gpu_util"),
    Column("node_reason", "NODE / REASON", 12, 2),     # why it waits / why it ended: kept before GPUS at 80 cols
    Column("part", "PART", 10, 4),
    Column("gmem", "GMEM", 8, 4),
    Column("cpu", "CPU", 5, 5),
    Column("mem", "MEM", 6, 5),
    Column("user", "USER", 9, 0, "user"),
    Column("submit", "SUBMIT", 11, 5, "submit"),
    Column("su", "SU", 8, 4, "su"),                    # Account (§5.2.11): off by default, via ui.jobs_columns
    Column("gpuh", "GPU-h", 7, 4, "gpuh"),
)}
RIGHT_ALIGNED = frozenset({"su", "gpuh"})

NAME_CAP = 40
NAME_FIRST = 28                  # slack tiers: NAME up to 28, then NODE / REASON up to 26, then NAME up to NAME_CAP,
REASON_WANT = 26                 # then the rest to NODE / REASON ("never: dependency failed" must not be cut at 120)
GAP = 1                          # spaces between columns (cell_padding=0, widths include the gap)


ID_CAP = 10                      # one long array id (`…012001_[5-12%2]`) must not widen ID for every 7-digit id


def layout(width: int, wanted: Sequence[str], *, show_user: bool, host_width: int = 10,
           id_width: int = 8) -> list[tuple[str, int]]:
    """Visible columns with their widths for a table ``width`` cells wide (§7.4)."""
    keys = ["marker"] + [k for k in wanted if k in COLUMNS and k != "marker"]
    for must in ("host", "id", "name", "state", "elapsed"):
        if must not in keys:
            keys.insert(min(len(keys), list(COLUMNS).index(must)), must)
    if show_user and "user" not in keys:
        keys.insert(keys.index("name"), "user")

    def base(k: str) -> int:
        w = host_width if k == "host" else id_width if k == "id" else COLUMNS[k].width
        return w + (GAP if k != "marker" else 0)

    while sum(base(k) for k in keys) > width:
        droppable = [k for k in keys if COLUMNS[k].priority > 1]
        if not droppable:
            break
        worst = max(droppable, key=lambda k: (COLUMNS[k].priority, keys.index(k)))
        keys.remove(worst)
    widths = {k: base(k) for k in keys}
    slack = width - sum(widths.values())

    def grow(key: str, upto: int) -> None:
        nonlocal slack
        if key in widths and slack > 0 and widths[key] < upto:
            add = min(slack, upto - widths[key])
            widths[key] += add
            slack -= add

    grow("name", NAME_FIRST + GAP)
    grow("node_reason", REASON_WANT + GAP)
    grow("name", NAME_CAP + GAP)
    if slack > 0:
        widths["node_reason" if "node_reason" in widths else "name"] += slack
    return [(k, widths[k]) for k in keys]


def fit_columns(cols: Sequence[tuple[str, str, int]], width: int, *, drop: Sequence[str] = (),
                flex: str | None = None) -> list[tuple[str, str, int]]:
    """Generic §7.4-style fitting for the smaller tables: drop columns in ``drop`` order until the rest fits, then
    give the slack to ``flex``."""
    visible = list(cols)
    for key in drop:
        if sum(c[2] for c in visible) <= width:
            break
        visible = [c for c in visible if c[0] != key]
    slack = width - sum(c[2] for c in visible)
    if flex is not None and slack > 0:
        visible = [(k, label, w + slack if k == flex else w) for k, label, w in visible]
    return visible


# ---------------------------------------------------------------- cells
@dataclass
class RowContext:
    now: datetime
    host_colors: Mapping[str, str]
    problem_hosts: Mapping[str, str]          # host -> short problem label ("auth", "retry")
    stale_hosts: frozenset[str]
    marked: frozenset[str]
    thresholds: tuple[float, float]
    widths: Mapping[str, int] = field(default_factory=dict)
    name_starts: Mapping[str, int] = field(default_factory=dict)    # humanize.distinct_starts of the visible names
    watched: frozenset[str] = frozenset()     # `w` (§7.14): "◉" before the name
    cost: CostFn | None = None                # SU / GPU-h columns (Account, §5.2.11); None = unknown everywhere
    usage_only_hosts: frozenset[str] = frozenset()   # their `su` value is billing-h, not SU: shown `576h`, muted


def gpus_text(job: Job) -> Text:
    if not job.alloc.gpus:
        return Text(humanize.NONE, style=theme.dim())
    return Text(f"{job.alloc.gpus}×{humanize.gpu_model(job.alloc.gpu_type)}")


def node_reason(job: Job, ctx: RowContext, *, est: bool = True) -> Text:
    """NODE / REASON: nodes (running), humanized reason + est. start (pending), exit summary (ended).

    The backend puts the humanized pending reason / exit summary in ``hints[0]`` for queued and ended jobs; the
    table prefers its own short wording for known Slurm reasons and uses the hint otherwise, never both.
    """
    group = job.state.group
    hints = list(job.hints)
    if group is StateGroup.QUEUED:
        if job.held:
            out = Text("held", style=theme.warn())
        elif job.reason and render.known_reason(job.reason):
            out = render.reason_text(job.reason)
        elif hints:
            out = Text(hints[0], style=theme.hint_style(render.hint_level(hints[0]) == "red"))
        else:
            out = render.reason_text(job.reason) if job.reason else Text("pending", style=theme.muted())
        if hints and (job.held or (job.reason and render.known_reason(job.reason)) or out.plain == hints[0]):
            hints = hints[1:]
        if est and job.start_time is not None and job.est_start:      # else the LEFT column shows it
            out.append(f" · est {humanize.local_hm(job.start_time)}", style=theme.muted())
    elif group is StateGroup.ACTIVE:
        if job.kind is JobKind.TASK:
            idx = sorted({i for n in job.alloc.per_node for i in n.gpu_indices})
            if idx:
                out = Text(f"gpu {','.join(map(str, idx))}", style=theme.muted())
            elif job.alloc.gpus:              # GPUs held, indices unknown: never "cpu only" next to a GPU%
                out = Text(f"{job.alloc.gpus} GPU{'s' if job.alloc.gpus != 1 else ''}", style=theme.muted())
            else:
                out = Text("cpu only", style=theme.muted())
        else:
            out = Text(job.nodelist or ",".join(job.nodes) or "—")
        if hints:                        # a hint (log idle 49m, GPU idle) is the news: before the node, not cut off
            hint = hints.pop(0)
            out = Text(hint, style=render.hint_style(hint)) + Text(" · ", style=theme.dim()) + out
    else:
        summary = hints.pop(0) if hints else exit_summary(job)
        failed = job.state.group is StateGroup.FAILED
        out = Text(summary, style=theme.state_style(job.state) if failed else theme.muted())
        if job.nodelist and job.kind is JobKind.SLURM and job.state is JobState.FAILED:
            out.append(f" · {job.nodelist}", style=theme.muted())
    for hint in hints[:1]:
        out.append(" · ", style=theme.dim())
        out.append(hint, style=render.hint_style(hint))
    problem = ctx.problem_hosts.get(job.key.host)
    if problem:
        out.append(f" · {problem}", style=theme.conn_style_label(problem))
    return out


def exit_summary(job: Job) -> str:
    st = job.state
    if st is JobState.PREEMPTED:
        return "preempted"
    if st is JobState.TIMEOUT:
        return "hit time limit"
    if st is JobState.OUT_OF_MEMORY:
        return "out of memory"
    if st is JobState.NODE_FAIL:
        return "node failure"
    if st is JobState.CANCELLED:
        return job.raw_state.lower().replace("cancelled by", "cancelled by uid") if "by" in job.raw_state \
            else "cancelled"
    if st is JobState.EXITED:
        return "ended, exit status unknown"
    if job.kind is JobKind.TASK and st is JobState.FAILED:
        return job.raw_state
    if job.exit_signal:
        return f"signal {job.exit_signal}"
    if job.exit_code is not None:
        return f"exit {job.exit_code}"
    return st.name.lower()


def left_text(job: Job, now: datetime) -> Text:
    if job.state.is_terminal:
        end = ended_at(job)
        return Text(f"ended {humanize.age((now - end).total_seconds())}" if end else "ended", style=theme.dim())
    if job.state.group is StateGroup.QUEUED:
        # a queued job has no time left yet; its useful fact is the estimated start (the limit is in the preview)
        if job.est_start and job.start_time is not None:
            wait = (job.start_time - now).total_seconds()
            return Text(f"~in {humanize.span(wait)}" if wait > 0 else "~now", style=theme.muted())
        return Text(humanize.NONE, style=theme.dim())
    remaining = left_s(job, now)
    if remaining is None:
        return Text(humanize.NONE, style=theme.dim())
    level = runway_level(remaining, job.time_limit.total_seconds() if job.time_limit is not None else None)
    style = "" if job.state.group is not StateGroup.ACTIVE else theme.error() if level == "red" else \
        theme.warn() if level == "amber" else ""
    return Text(humanize.left(remaining), style=style)


def cost_cell(job: Job, key: str, ctx: RowContext) -> Text:
    """SU / GPU-h (§7.5): ACTIVE rows tick with ``ctx.now``; a pending job's SU is its projection, dim (`≈576`); a free
    job a dim `0`; unknown `—`. On a usage-only cluster the value is billing-h, unscaled by the QOS factor (the cost
    fn swaps it in, JobCost.as_billing_h), never SU: it carries an `h` and is muted (`576h`, `≈576h`), so it is not
    read as charged SU next to a limits cluster's."""
    c = ctx.cost(job, ctx.now) if ctx.cost is not None else None
    value = su_value(job, c) if key == "su" else gpuh_value(job, c)
    if value is None:
        return Text(humanize.NONE, style=theme.dim())
    if key == "su":
        bill_h = job.key.host in ctx.usage_only_hosts
        text = costs.fmt_su(value) + ("h" if bill_h else "")
        if job.state.group is StateGroup.QUEUED:
            return Text("≈" + text, style=theme.dim())
        return Text(text, style=theme.dim() if value == 0 else theme.muted() if bill_h else "")
    return Text(costs.fmt_hours(value), style=theme.dim() if value == 0 else "")


def runway_level(remaining: float, limit: float | None) -> str | None:
    """Time left before the limit: ``red`` below 2 % of it or 5 min, ``amber`` below 10 % or 30 min. The absolute
    floors are capped at 10 % / 25 % of the limit, so a 30-minute job is not amber for its whole run."""
    if not limit:
        return "red" if remaining < 300 else "amber" if remaining < 1800 else None
    if remaining < 0.02 * limit or remaining < min(300.0, 0.10 * limit):
        return "red"
    if remaining < 0.10 * limit or remaining < min(1800.0, 0.25 * limit):
        return "amber"
    return None


def name_badges(job: Job, watched: bool) -> Text:
    """``◉`` (watched) and ``↻N`` (restarted N times) in front of the name."""
    out = Text()
    if watched:
        out.append("◉ ", style=theme.accent())
    if job.restarts > 0:
        out.append(f"↻{job.restarts} ", style=theme.warn())
    return out


def job_cells(job: Job, cols: Sequence[tuple[str, int]], ctx: RowContext, *, cursor: bool) -> tuple[Text, ...]:
    host = job.key.host
    stale = host in ctx.stale_hosts
    problem = host in ctx.problem_hosts
    out: list[Text] = []
    for key, width in cols:
        w = width - (GAP if key != "marker" else 0)
        if key == "marker":
            mark = "+" if str(job.key) in ctx.marked else "!" if problem else "~" if stale else " "
            cell = Text("▸" if cursor else " ", style=theme.accent())
            cell.append(mark, style=theme.accent() if mark == "+" else theme.warn() if mark == "!" else theme.dim())
        elif key == "host":
            cell = Text(humanize.truncate(host, w), style=ctx.host_colors.get(host, ""))
        elif key == "id":
            # cut from the left: the array task suffix (`…012001_3`, `…_[1-10%2]`) is what tells rows apart
            cell = Text(humanize.truncate_left(job.key.label, w), style="" if job.is_mine else theme.muted())
        elif key == "name":
            cell = name_badges(job, str(job.key) in ctx.watched)
            cell.append(humanize.truncate_name(job.name, max(1, w - cell.cell_len), ctx.name_starts.get(job.name, 0)))
        elif key == "state":
            cell = render.state_badge(job.state, job.held)
        elif key == "elapsed":
            cell = Text(humanize.left(elapsed_s(job, ctx.now)))           # 1-00:15, not 1-00:15…
        elif key == "left":
            cell = left_text(job, ctx.now)
        elif key == "gpus":
            cell = gpus_text(job)
        elif key == "gpu_util":
            util = job.gpu.util_mean_pct if job.gpu is not None and job.state.group is StateGroup.ACTIVE else None
            cell = render.mini_util(util, ctx.thresholds)
        elif key == "node_reason":
            cell = node_reason(job, ctx, est=not any(k == "left" for k, _w in cols))
        elif key == "part":
            cell = Text(job.partition or humanize.NONE, style=theme.muted())
        elif key == "gmem":
            mem = job.gpu.mem_used_mib if job.gpu is not None else None
            cell = Text(humanize.gib(mem, dash=humanize.NA), style=theme.muted())
        elif key == "cpu":
            cell = Text(humanize.pct(job.cpu_pct) if job.kind is JobKind.TASK
                        else str(job.alloc.cpus or humanize.NA), style=theme.muted())
        elif key == "mem":
            cell = Text(humanize.size(job.rss_bytes if job.kind is JobKind.TASK else job.alloc.mem_bytes),
                        style=theme.muted())
        elif key == "user":
            cell = Text(humanize.truncate(job.user, w), style="" if job.is_mine else theme.muted())
        elif key == "submit":
            sub = job.submit_time or job.start_time
            cell = Text(humanize.ago(sub, ctx.now, dash=humanize.NONE), style=theme.muted())
        elif key in RIGHT_ALIGNED:
            cell = cost_cell(job, key, ctx)
        else:   # pragma: no cover - layout only yields known keys
            cell = Text("")
        if key != "marker" and cell.cell_len > w:
            cell.truncate(w, overflow="ellipsis")
        elif key in RIGHT_ALIGNED and cell.cell_len < w:
            cell = Text(" " * (w - cell.cell_len)) + cell
        if stale and key not in ("marker",):
            cell.stylize(theme.dim())
        out.append(cell)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class GroupRow:
    """A group header: ``▾ k2 · 12 jobs · R 4 · PD 1 · GPU 94%``."""
    key: str
    label: str
    jobs: tuple[Job, ...]
    folded: bool


def group_cells(group: GroupRow, cols: Sequence[tuple[str, int]], ctx: RowContext, *,
                cursor: bool) -> tuple[Text, ...]:
    running = [j for j in group.jobs if j.state.group is StateGroup.ACTIVE]
    queued = sum(1 for j in group.jobs if j.state.group is StateGroup.QUEUED)
    utils = [j.gpu.util_mean_pct for j in running if j.gpu is not None and j.gpu.util_mean_pct is not None]
    summary = Text(f"{'▸' if group.folded else '▾'} ", style=theme.accent())
    summary.append(group.label, style="bold")
    summary.append(f" · {len(group.jobs)} job{'s' if len(group.jobs) != 1 else ''}", style=theme.muted())
    if running:
        summary.append(f" · R {len(running)}", style=theme.state_style_group(StateGroup.ACTIVE))
    if queued:
        summary.append(f" · PD {queued}", style=theme.state_style_group(StateGroup.QUEUED))
    if utils:
        summary.append(f" · GPU {humanize.pct(sum(utils) / len(utils))}", style=theme.muted())
    marked = sum(1 for j in group.jobs if str(j.key) in ctx.marked)
    if marked:                 # a folded group's marked members are hidden, so not acted on: say it on the header
        summary.append(f" · {marked} marked" + (" (folded: not acted on)" if group.folded else ""),
                       style=theme.accent())
    # DataTable cannot span columns, but with cell_padding=0 adjacent cells form one line: slice the summary
    # across the data columns.
    out: list[Text] = []
    rest = summary
    for key, width in cols:
        if key == "marker":
            out.append(Text("▸" if cursor else " ", style=theme.accent()) + Text(" "))
            continue
        out.append(render.fit(rest[:width], width) if rest.cell_len else Text(""))
        rest = rest[width:]
    return tuple(out)


class ClickTable(DataTable[Any]):
    """Row-cursor DataTable with the §7.9 mouse model: click selects, double-click / enter activates (posts
    ``Activated``), right-click posts ``ContextMenu``. Subclasses provide ``cursor_key``."""

    class Activated(Message):
        """Enter / double-click on a row (``key`` = row key)."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    class ContextMenu(Message):
        """Right-click on a row."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("cursor_type", "row")
        kwargs.setdefault("show_row_labels", False)
        # the cursor row keeps each cell's own colours (bars, states, hosts); only its background marks the row
        kwargs.setdefault("cursor_foreground_priority", "renderable")
        super().__init__(**kwargs)
        self._from_click = False
        self._click_chain = 1

    def forget_rendered(self) -> None:
        """The palette changed (§7.11): drop every "unchanged, skip it" cache — Rich's Text == compares plain text and
        spans, not the base style, so a recoloured cell would look unchanged and keep its old colour."""
        for attr, empty in (("_cells", {}), ("_signature", ())):
            if hasattr(self, attr):
                setattr(self, attr, empty)

    @property
    def cursor_key(self) -> str | None:
        if self.row_count == 0 or not 0 <= self.cursor_row < self.row_count:
            return None
        key = self.coordinate_to_cell_key(Coordinate(self.cursor_row, 0)).row_key.value
        return None if key is None else str(key)

    # ------------------------------------------------------------------ row cursor (home/end/page keys, UI-51)
    def action_first_row(self) -> None:
        self.move_cursor(row=0)

    def action_last_row(self) -> None:
        self.move_cursor(row=max(0, self.row_count - 1))

    def action_half_page(self, direction: int) -> None:
        step = max(1, (self.size.height - 1) // 2)
        self.move_cursor(row=max(0, min(self.row_count - 1, self.cursor_row + direction * step)))

    # ------------------------------------------------------------------ mouse / activation
    def on_click(self, event: events.Click) -> None:
        self._from_click = True
        self._click_chain = event.chain
        if event.button == 3:
            key = self.cursor_key
            if key is not None:
                self.post_message(self.ContextMenu(key))

    def on_key(self, event: events.Key) -> None:
        self._from_click = False

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        from_click, chain = self._from_click, self._click_chain
        self._from_click = False
        if from_click and chain < 2:
            return          # single click on the highlighted row only selects (§7.9)
        key = event.row_key.value if isinstance(event.row_key, RowKey) else None
        if key is not None:
            self.post_message(self.Activated(str(key)))


Row = Job | GroupRow


def row_key(row: Row) -> str:
    return row.key if isinstance(row, GroupRow) else str(row.key)


class JobTable(ClickTable):
    """The Jobs view table. Rows are ``Job``s (key ``host:id``) or ``GroupRow`` headers (key ``§…``)."""

    # the focused table sees these before JobsView: same moves as keymap "jobs" (g/home, G/end, ^d/pgdn, ^u/pgup)
    BINDINGS = [Binding("home", "first_row", show=False), Binding("end", "last_row", show=False),
                Binding("pagedown", "half_page(1)", show=False), Binding("pageup", "half_page(-1)", show=False)]

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, cursor_type="row", show_row_labels=False, cell_padding=0, zebra_stripes=False)
        self.cols: list[tuple[str, int]] = []
        self.rows_by_key: dict[str, Row] = {}
        self.order: list[str] = []
        self._cells: dict[str, tuple[Text, ...]] = {}
        self._cursor_key: str | None = None
        self._ctx: RowContext | None = None
        self.sort_column: SortColumn = "host"
        self.sort_reverse = False

    # ------------------------------------------------------------------ public
    @property
    def cursor_key(self) -> str | None:
        if not self.order or self.cursor_row < 0 or self.cursor_row >= len(self.order):
            return None
        return self.order[self.cursor_row]

    def current_row(self) -> Row | None:
        key = self.cursor_key
        return self.rows_by_key.get(key) if key else None

    def set_columns(self, cols: list[tuple[str, int]]) -> bool:
        """Install a new column layout; returns True when it changed (the caller then re-sets rows)."""
        if cols == self.cols:
            return False
        keep = self.cursor_key
        self.cols = cols
        self.clear(columns=True)
        for key, width in cols:
            self.add_column(self._header_label(key, width), key=key, width=width)
        self.order = []
        self._cells = {}
        self._cursor_key = keep
        return True

    def _header_label(self, key: str, width: int) -> Text:
        label = self._header_text(key, width)
        room = max(1, width - GAP)
        if key in RIGHT_ALIGNED and label.cell_len < room:      # over its right-aligned numbers
            return Text(" " * (room - label.cell_len)) + label
        return label

    def _header_text(self, key: str, width: int) -> Text:
        col = COLUMNS[key]
        room = max(1, width - GAP)
        sorted_here = (col.sort is not None and col.sort == self.sort_column) or (key == "host" and
                                                                                 self.sort_column == "host")
        if not sorted_here:
            return Text(humanize.truncate(col.label, room), style="bold")
        arrow = query.sort_arrow(self.sort_column, self.sort_reverse)
        if room < 3:
            return Text(arrow, style="bold")
        # the arrow always stays visible; the space goes before the label is cut ("ST▼", "ELAPSED▼", "NODE / RE… ▼")
        width_label = cell_len(col.label)
        if width_label + 2 <= room:
            return Text(f"{col.label} {arrow}", style="bold")
        if width_label + 1 <= room:
            return Text(f"{col.label}{arrow}", style="bold")
        return Text(f"{humanize.truncate(col.label, room - 2)} {arrow}", style="bold")

    def refresh_headers(self) -> None:
        for key, width in self.cols:
            column = self.columns.get(key)  # type: ignore[call-overload]
            if column is not None:
                column.label = self._header_label(key, width)
        self._require_update_dimensions = True
        self.refresh()

    def set_rows(self, rows: Sequence[Row], ctx: RowContext) -> None:
        """Keyed diff against the current rows (see module docstring)."""
        self._ctx = ctx
        keys = [row_key(r) for r in rows]
        new_rows = {row_key(r): r for r in rows}
        cursor_key = self.cursor_key or self._cursor_key
        cells = {k: self._render_row(new_rows[k], ctx, cursor=k == cursor_key) for k in keys}
        if keys == self.order:
            for k in keys:
                old = self._cells.get(k)
                new = cells[k]
                if old == new:
                    continue
                for (col, _w), value, prev in zip(self.cols, new, old or (None,) * len(new), strict=True):
                    if value != prev:
                        self.update_cell(k, col, value)
        else:
            # order changed: rebuild, but keep the selected job on the same screen row (the table must not shift
            # under the user on every refresh of a GPU%/elapsed sort)
            offset = self.cursor_row - int(self.scroll_y) if self.order else None
            old_index = self.order.index(cursor_key) if cursor_key in self.order else self.cursor_row
            fallback = None if cursor_key in new_rows else self._nearest_survivor(old_index, new_rows)
            self.clear(columns=False)
            for k in keys:
                self.add_row(*cells[k], key=k)
            if cursor_key in new_rows:
                index = keys.index(cursor_key)
                self.move_cursor(row=index, animate=False, scroll=offset is None)
                if offset is not None:
                    target = max(0, index - offset)
                    self.call_after_refresh(self.scroll_to, y=target, animate=False, force=True)
            elif keys:         # the selected row went away: its nearest surviving neighbour, not row 0 (clear() reset it)
                self.move_cursor(row=keys.index(fallback) if fallback is not None and fallback in new_rows else
                                 min(max(0, old_index), len(keys) - 1), animate=False)
        self.order = keys
        self.rows_by_key = new_rows
        self._cells = cells
        self._cursor_key = self.cursor_key

    def _nearest_survivor(self, index: int, new_rows: Mapping[str, Row]) -> str | None:
        """The old row nearest to ``index`` (below first) that is still present."""
        for d in range(1, len(self.order)):
            for i in (index + d, index - d):
                if 0 <= i < len(self.order) and self.order[i] in new_rows:
                    return self.order[i]
        return None

    def _render_row(self, row: Row, ctx: RowContext, *, cursor: bool) -> tuple[Text, ...]:
        if isinstance(row, GroupRow):
            return group_cells(row, self.cols, ctx, cursor=cursor)
        return job_cells(row, self.cols, ctx, cursor=cursor)

    def move_to_key(self, key: str) -> None:
        if key in self.order:
            self.move_cursor(row=self.order.index(key))

    # ------------------------------------------------------------------ cursor marker
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        new = self.cursor_key
        old = self._cursor_key
        self._cursor_key = new
        if self._ctx is None:
            return
        for key in {old, new} - {None}:
            assert key is not None
            row = self.rows_by_key.get(key)
            if row is None or key not in self.order:
                continue
            cells = self._render_row(row, self._ctx, cursor=key == new)
            self._cells[key] = cells
            try:
                self.update_cell(key, "marker", cells[0])
            except Exception:   # row vanished between highlight and refresh
                pass
