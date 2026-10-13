"""[WP3] Node / partition table (clusters; mockup H).

``PartitionTable``: one row per partition in scope (node count, node states, free/total GPUs, per-type breakdown).
``NodeTable``: the nodes of the selected partition with CPU and memory allocation, a discrete GPU usage bar, the GPU
type and the jobs on each node: every user's while the host detail is open (§5.2.9), else the user's own.
``f`` keeps only nodes with free GPUs. ``occupant_lines``: the panel under the table listing the jobs on one node.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from rich.text import Text
from textual.widgets import DataTable

from runfleet.models import Job, Node, NodeJob, PartitionSummary, StateGroup
from runfleet.ui import render, theme
from runfleet.ui.widgets.job_table import ClickTable, fit_columns
from runfleet.util import humanize

NODE_MAX = 20
_STATE_ABBR = {"allocated": "alloc", "completing": "compl", "draining": "drain"}


def node_state_text(node: Node) -> Text:
    state = node.state + ("-" if "-" in node.flags else "") + ("*" if "*" in node.flags else "")
    if node.state in ("idle",):
        style = theme.ok()
    elif node.state in ("mixed", "planned", "allocated", "completing"):
        style = theme.warn() if node.state == "allocated" else ""
    elif node.state in ("down", "drained", "draining", "fail", "failing", "maint", "inval"):
        style = theme.error()
    else:
        style = theme.muted()
    return Text(state, style=style)


def gpu_totals(node: Node) -> tuple[int, int, str]:
    gpus = [g for g in node.gres if g.name == "gpu"]
    total = sum(g.total for g in gpus)
    used = sum(g.used for g in gpus)
    types = sorted({g.type for g in gpus if g.type})
    if len(types) > 1:
        label = f"{types[0].split('_')[0] if '_' in types[0] else types[0]} MIG ({len(types)} types)"
        if all("nvidia_" in t for t in types):
            label = f"{humanize.gpu_model(types[0])} MIG ({len(types)} types)"
    else:
        label = humanize.gpu_label(types[0]) if types else humanize.NONE    # "3090", like every other view
    return used, total, label


def states_text(states: Mapping[str, int]) -> str:
    items = sorted(states.items(), key=lambda kv: -kv[1])
    return " · ".join(f"{_STATE_ABBR.get(k, k)} {v}" for k, v in items)


def _gpus(n: int | None, exact: bool) -> str:
    if n is None:
        return "?"
    return f"{n}" if exact else f"~{n}"


def jobs_cell(node: Node, mine: Sequence[Job], width: int, now: datetime) -> Text:
    """JOBS: the user's jobs (id), then the other users with the GPUs they hold here (``bob×2``), most GPUs first;
    what does not fit ends in ``+N``. Without a fresh reading of every user's jobs: the user's own, as before."""
    seen = node.jobs_seen(now)
    if seen is None:
        out = Text(mine[0].key.label if mine else humanize.NONE, style="" if mine else theme.dim())
        if len(mine) > 1:
            out.append(f" +{len(mine) - 1}", style=theme.muted())
        return out
    if not seen:
        return Text(humanize.NONE, style=theme.dim())
    items: list[Text] = [Text(j.job_id, style=f"bold {theme.accent()}") for j in seen if j.mine]
    others: dict[str, list[NodeJob]] = {}
    for j in seen:
        if not j.mine:
            others.setdefault(j.user, []).append(j)
    ranked = sorted(others.items(), key=lambda kv: (-sum(j.gpus or 0 for j in kv[1]), kv[0]))
    for user, jobs in ranked:
        item = Text(user)
        if any(j.gpus != 0 for j in jobs):                  # GPUs held, or unknown (None)
            known = [j.gpus for j in jobs if j.gpus is not None]
            exact = all(j.gpus is not None and j.gpus_exact for j in jobs)
            item.append("×" + (_gpus(sum(known), exact) if known else "?"), style=theme.muted())
        items.append(item)
    out = Text()
    for i, item in enumerate(items):
        more = len(items) - i - 1
        if i and out.cell_len + 3 + item.cell_len + (len(f" +{more}") if more else 0) > width - 1:
            out.append(f" +{len(items) - i}", style=theme.muted())
            break
        if i:
            out.append(" · ", style=theme.dim())
        out.append_text(item)
    return out


_OCC_COLS = (("job", "JOB", 12), ("user", "USER", 11), ("real", "NAME", 19), ("account", "ACCOUNT", 15),
             ("part", "PARTITION", 13), ("gpu", "GPU", 5), ("time", "ELAPSED / LIMIT", 20), ("left", "LEFT", 9),
             ("name", "JOB NAME", 12))
OCC_MAX_ROWS = 8


def occupant_lines(node: Node | None, width: int, now: datetime, *, hidden: bool = False,
                   max_rows: int = OCC_MAX_ROWS) -> tuple[list[Text], str]:
    """The jobs on one node (every user, §5.2.9): column header + one row per job, own jobs first, then the job that
    ends soonest (its GPUs free up first). Returns (lines, panel subtitle)."""
    if node is None:
        return [Text("no node selected", style=theme.dim())], ""
    seen = node.jobs_seen(now)
    if seen is None:
        return [Text(f"reading who runs on {node.name}…", style=theme.dim())], ""
    shift = (now - node.jobs_at).total_seconds() if node.jobs_at is not None else 0.0
    cols = fit_columns(_OCC_COLS, width, drop=("account", "real", "part", "time"), flex="name")
    head = Text()
    for _k, label, w in cols:
        head.append(label.ljust(w), style=f"bold {theme.muted()}")
    lines = [render.fit(head, width)]

    def left_of(j: NodeJob) -> float:
        return j.left_s - shift if j.left_s is not None else float("inf")

    jobs = sorted(seen, key=lambda j: (not j.mine, left_of(j), j.job_id))
    for j in jobs[:max_rows]:
        elapsed = j.elapsed_s + shift if j.elapsed_s is not None else None
        left = max(0.0, left_of(j)) if j.left_s is not None else None
        gpus = Text(_gpus(j.gpus, j.gpus_exact) if j.gpus != 0 else humanize.NONE,
                    style="" if j.gpus else theme.dim())
        cells = {
            "job": Text(j.job_id, style=f"bold {theme.accent()}" if j.mine else ""),
            "user": Text(j.user, style=f"bold {theme.accent()}" if j.mine else ""),
            "real": Text(j.real_name or humanize.NONE, style=theme.muted() if j.real_name else theme.dim()),
            "account": Text(j.account or humanize.NONE, style=theme.muted()),
            "part": Text(j.partition or humanize.NONE, style=theme.muted()),
            "gpu": gpus,
            "time": Text(f"{humanize.left(elapsed)} / {humanize.left(j.limit_s, dash='∞')}", style=theme.muted()),
            "left": Text(humanize.left(left, dash="∞")),
            "name": Text(j.name),
        }
        row = Text()
        for key, _label, w in cols:
            row.append_text(render.fit(_cut(cells[key], w), w))
        lines.append(render.fit(row, width))
    if len(jobs) > max_rows:
        lines.append(Text(f"+{len(jobs) - max_rows} more", style=theme.muted()))
    if not jobs:
        lines = [Text(f"no running jobs on {node.name}", style=theme.dim())]
    if hidden:
        lines.append(Text("this cluster hides other users' jobs (PrivateData=jobs): only yours are listed",
                          style=theme.dim()))
    used, total, _label = gpu_totals(node)
    if total and used < total:
        sub = f"{total - used} GPU{'s' if total - used > 1 else ''} free now"
    elif total:
        holders = [j for j in jobs if j.gpus != 0]           # GPUs held, or a split not known (None)
        ends = [left_of(j) for j in holders if j.left_s is not None]
        sub = (f"next GPU frees in ≤ {humanize.span(max(0.0, min(ends)))}" if ends
               else "no time limit on its jobs" if holders else "")
    else:
        sub = ""
    if shift >= 90:
        sub = (sub + " · " if sub else "") + f"as of {humanize.age(shift)} ago"
    return lines, sub


def _cut(text: Text, width: int) -> Text:
    """Fit a cell into ``width - 1`` cells (one cell of gap; the tables use cell_padding=0)."""
    if text.cell_len > width - 1:
        text = text.copy()
        text.truncate(max(1, width - 1), overflow="ellipsis")
    return text


class PartitionTable(ClickTable):
    COLS: tuple[tuple[str, str, int], ...] = (
        ("name", "PARTITION", 14), ("nodes", "NODES", 7), ("states", "STATES", 30), ("gpus", "GPUS FREE/TOTAL", 23),
        ("types", "BY TYPE (free/total)", 24),
    )

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, cell_padding=0)
        self.names: list[str] = []
        self.cols: list[tuple[str, str, int]] = []
        self._signature: tuple[object, ...] = ()

    def _layout(self) -> None:
        width = (self.size.width or self.app.size.width - 4) - 1
        cols = fit_columns(self.COLS, width, drop=("types", "nodes"), flex="types")
        if cols != self.cols:
            self.cols = cols
            self.clear(columns=True)
            for key, label, w in cols:
                self.add_column(Text(label, style="bold"), key=key, width=w)

    def select(self, name: str | None) -> None:
        """Put the cursor on `name` without emitting RowHighlighted (programmatic, one-way)."""
        if name in self.names and self.cursor_row != self.names.index(name):
            with self.prevent(DataTable.RowHighlighted):
                self.move_cursor(row=self.names.index(name))

    def show(self, parts: Sequence[PartitionSummary], selected: str | None = None) -> None:
        """Rebuild only when the rows changed, then put the cursor on `selected` (the screen's partition). Never
        emits RowHighlighted: the screen reacts only to the user moving the cursor (a rebuild that restored the old
        cursor row used to re-select the previous partition, which rebuilt again — an endless loop)."""
        self._layout()
        widths = {k: w for k, _l, w in self.cols}
        signature = (tuple(self.cols), tuple(parts))
        if signature == self._signature:
            self.select(selected)
            return
        self._signature = signature
        with self.prevent(DataTable.RowHighlighted):
            self._rebuild(parts, widths)
        self.select(selected)

    def _rebuild(self, parts: Sequence[PartitionSummary], widths: dict[str, int]) -> None:
        self.clear()
        for p in parts:
            types = " · ".join(f"{label} {f}/{n}" for label, f, n in humanize.gpu_type_counts(p.gpu_types))
            free = render.free_gpus(p, sep=" / ")
            cells = {"name": Text(p.name, style="bold"), "nodes": Text(str(p.nodes)),
                     "states": Text(states_text(p.states), style=theme.muted()), "gpus": free,
                     "types": Text(types, style=theme.muted())}
            self.add_row(*[_cut(cells[k], widths[k]) for k, _l, _w in self.cols], key=p.name)
        self.names = [p.name for p in parts]


class NodeTable(ClickTable):
    COLS: tuple[tuple[str, str, int], ...] = (
        ("node", "NODE", 11), ("state", "STATE", 11), ("cpu", "CPU A/I/T", 13), ("mem", "MEM ALLOC/TOTAL", 17),
        ("gpu", "GPU USED/TOTAL", 21), ("type", "TYPE", 18), ("jobs", "JOBS", 24),
    )

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, cell_padding=0)
        self.cols: list[tuple[str, str, int]] = []
        self.order: list[str] = []

    def _layout(self, nodes: Sequence[Node] = ()) -> None:
        width = (self.size.width or self.app.size.width - 4) - 1
        # NODE fits the longest name in scope (lyra's `lyra-gpu-s71-02` read `lyra-gpu-…` on every row); a name
        # that still does not fit is cut from the left so the distinguishing suffix stays visible.
        node_w = max(11, min(NODE_MAX, max((len(n.name) for n in nodes), default=0) + 1))
        base = tuple((k, label, node_w if k == "node" else w) for k, label, w in self.COLS)
        cols = fit_columns(base, width, drop=("type", "mem", "cpu"), flex="jobs")
        if cols != self.cols:
            self.cols = cols
            self.clear(columns=True)
            for key, label, w in cols:
                self.add_column(Text(label, style="bold"), key=key, width=w)

    def show(self, nodes: Sequence[Node], jobs: Sequence[Job], *, free_only: bool, now: datetime) -> int:
        self._layout(nodes)
        widths = {k: w for k, _l, w in self.cols}
        by_node: dict[str, list[Job]] = {}
        for j in jobs:
            if j.is_mine and j.state.group is StateGroup.ACTIVE:    # MY JOBS, and what ↵ opens (current_job)
                for n in j.nodes:
                    by_node.setdefault(n, []).append(j)
        rows = []
        for node in sorted(nodes, key=lambda n: n.name):
            used, total, label = gpu_totals(node)
            if free_only and not (node.usable and total - used > 0):
                continue
            rows.append((node, used, total, label))
        keep = self.cursor_key
        self.clear()
        for node, used, total, label in rows:
            cpu = (f"{node.cpus_alloc}/{node.cpus_idle}/{node.cpus_total}"
                   if node.cpus_total is not None else humanize.NA)
            mem = (f"{(node.mem_alloc_mib or 0) // 1024}/{(node.mem_total_mib or 0) // 1024}G"
                   if node.mem_total_mib else humanize.NA)
            gpu = Text(f"{used}/{total}".ljust(6))
            gpu.append_text(render.count_bar(used, total, max_cells=14))
            my = jobs_cell(node, by_node.get(node.name, []), widths["jobs"], now)
            cells = {"node": Text(humanize.truncate_left(node.name, widths["node"] - 1)), "state": node_state_text(node), "cpu": Text(cpu, style=theme.muted()),
                     "mem": Text(mem, style=theme.muted()), "gpu": gpu, "type": Text(label, style=theme.muted()),
                     "jobs": my}
            self.add_row(*[_cut(cells[k], widths[k]) for k, _l, _w in self.cols], key=node.name)
        self.order = [r[0].name for r in rows]
        if keep in self.order:
            self.move_cursor(row=self.order.index(keep))
        return len(rows)
