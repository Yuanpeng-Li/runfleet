"""[WP3] Node / partition table (clusters; mockup H).

``PartitionTable``: one row per partition in scope (node count, node states, free/total GPUs, per-type breakdown).
``NodeTable``: the nodes of the selected partition with CPU and memory allocation, a discrete GPU usage bar, the GPU
type and the user's jobs on each node. ``f`` keeps only nodes with free GPUs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.text import Text
from textual.widgets import DataTable

from runfleet.models import Job, Node, PartitionSummary, StateGroup
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
        ("gpu", "GPU USED/TOTAL", 21), ("type", "TYPE", 22), ("jobs", "MY JOBS", 14),
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
        cols = fit_columns(base, width, drop=("type", "mem", "cpu"), flex="type")
        if cols != self.cols:
            self.cols = cols
            self.clear(columns=True)
            for key, label, w in cols:
                self.add_column(Text(label, style="bold"), key=key, width=w)

    def show(self, nodes: Sequence[Node], jobs: Sequence[Job], *, free_only: bool) -> int:
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
            mine = by_node.get(node.name, [])
            my = Text(mine[0].key.label if mine else humanize.NONE, style="" if mine else theme.dim())
            if len(mine) > 1:
                my.append(f" +{len(mine) - 1}", style=theme.muted())
            cells = {"node": Text(humanize.truncate_left(node.name, widths["node"] - 1)), "state": node_state_text(node), "cpu": Text(cpu, style=theme.muted()),
                     "mem": Text(mem, style=theme.muted()), "gpu": gpu, "type": Text(label, style=theme.muted()),
                     "jobs": my}
            self.add_row(*[_cut(cells[k], widths[k]) for k, _l, _w in self.cols], key=node.name)
        self.order = [r[0].name for r in rows]
        if keep in self.order:
            self.move_cursor(row=self.order.index(keep))
        return len(rows)
