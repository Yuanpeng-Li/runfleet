"""[WP3] Process table (workstations; Slurm jobs through Attach).

Rows are the host snapshot's processes: members of my tasks plus GPU-holding processes of all users (others hidden
until ``u``). GPU index and GPU memory come from the GPU sample; COMMAND absorbs the remaining width. Keyed by pid so
the cursor stays on the same process across refreshes.

A Slurm job's Procs tab (Attach, §7.16) calls :meth:`ProcTable.show_live` with one :class:`NodeRows` per node (its
``ps`` rows, GPUs and pid -> ``"<job>.<step>"`` from ``scontrol listpids``): a STEP column replaces TASK, the GPU column
shows the Slurm IDX (``?`` when unknown, never a guessed number), and D-state (uninterruptible, usually I/O) and zombie
processes are highlighted. Pids repeat across nodes, so with more than one node every row, GPU join and step is keyed
by (node, pid) and a NODE column appears; one node keeps the plain pid keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from runfleet.models import GPU, Proc
from runfleet.ui import theme
from runfleet.ui.widgets.job_table import ClickTable, fit_columns
from runfleet.util import humanize

SORTS: tuple[str, ...] = ("gpu_mem", "cpu", "mem", "time", "pid")
SORT_LABELS = {"gpu_mem": "GPU mem", "cpu": "%CPU", "mem": "%MEM", "time": "TIME", "pid": "PID"}
_COLS: tuple[tuple[str, str, int], ...] = (
    ("gpu", "GPU", 4), ("pid", "PID", 9), ("user", "USER", 10), ("gmem", "GPU-MEM", 9), ("cpu", "%CPU", 6),
    ("mem", "%MEM", 6), ("time", "TIME", 11), ("task", "TASK", 10),
)
_STEP_COL = ("step", "STEP", 16)            # "31367626.extern" + one cell gap
NODE_COL_MAX = 14                           # NODE column width cap ("lyra-gpu-s71-03" is cut to fit)

RowKey = int | tuple[str, int]               # pid; (node, pid) when the rows come from several nodes


@dataclass(frozen=True, slots=True)
class NodeRows:
    """One node's share of a Slurm job's Procs tab (Attach): its ``ps`` rows, its GPUs and pid -> step."""
    node: str
    procs: Sequence[Proc]
    gpus: Sequence[GPU]
    steps: Mapping[int, str]


def _gpu_label(g: GPU, slurm: bool) -> str:
    if not slurm:
        return str(g.index)
    return str(g.slurm_idx) if g.slurm_idx is not None else "?"


def gpu_labels(gpus: Sequence[GPU], *, slurm: bool = False) -> dict[int, tuple[list[str], float]]:
    """pid -> (GPU labels, total GPU memory MiB); ``slurm``: labels are Slurm IDX numbers (``?`` = unknown)."""
    out: dict[int, tuple[list[str], float]] = {}
    for g in gpus:
        for p in g.processes:
            idx, mem = out.get(p.pid, ([], 0.0))
            out[p.pid] = (idx + [_gpu_label(g, slurm)], mem + (p.used_mem_mib or 0.0))
    return out


def stat_style(stat: str) -> str | None:
    """D (uninterruptible sleep: stuck on I/O, a hung nvidia-smi) and Z (zombie) rows stand out; None = normal."""
    if stat.startswith("Z"):
        return theme.error()
    if stat.startswith("D"):
        return theme.warn()
    return None


def _cmd(p: Proc) -> str:
    label = humanize.command_label(p.args)
    if p.stat.startswith("Z"):
        return f"<defunct> {label}"
    if p.stat.startswith("D"):
        return f"[D] {label}"
    return label


def sort_procs(rows: Sequence[tuple[RowKey, Proc]], gpu: Mapping[RowKey, tuple[Sequence[object], float]], key: str,
               reverse: bool) -> list[tuple[RowKey, Proc]]:
    def value(row: tuple[RowKey, Proc]) -> float:
        k, p = row
        if key == "gpu_mem":
            return gpu.get(k, ([], -1.0))[1]
        if key == "cpu":
            return p.cpu_pct
        if key == "mem":
            return p.mem_pct
        if key == "time":
            return float(p.etime_s)
        return float(-p.pid)
    return sorted(rows, key=value, reverse=not reverse)


def row_id(key: RowKey) -> str:
    """The DataTable row key: ``"4242"``, or ``"n-37-02/4242"`` for a multi-node row."""
    return str(key) if isinstance(key, int) else f"{key[0]}/{key[1]}"


class ProcTable(ClickTable):
    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, cell_padding=0)
        self.cols: list[tuple[str, str, int]] = []
        self.order: list[RowKey] = []
        self.procs: dict[RowKey, Proc] = {}
        self._cells: dict[RowKey, tuple[Text, ...]] = {}

    def _setup(self, width: int, steps: bool = False, node_w: int = 0) -> None:
        base = tuple(c for c in _COLS if c[0] != "task") + (_STEP_COL,) if steps else _COLS
        if node_w:
            base = (base[0], ("node", "NODE", node_w), *base[1:])
        drop = ("user", "time", "mem") if steps else ("time", "mem", "user")   # a job's processes are all mine
        cols = fit_columns((*base, ("cmd", "COMMAND", 20)), width - 2, drop=drop, flex="cmd")
        if cols == self.cols:
            return
        self.cols = cols
        self.clear(columns=True)
        for key, label, w in cols:
            self.add_column(Text(label, style="bold"), key=key, width=w)
        self.order = []
        self._cells = {}

    def show(self, procs: Sequence[Proc], gpus: Sequence[GPU], *, me: str | None, show_others: bool,
             sort: str, reverse: bool, task_labels: Mapping[Any, str],
             steps: Mapping[int, str] | None = None, gpu: int | None = None) -> None:
        """``steps``: a Slurm job's pid -> step map (Attach): STEP column, Slurm IDX, D / Z rows highlighted.
        ``gpu``: only the processes on that GPU, whoever owns them (``show_others`` does not apply)."""
        slurm = steps is not None
        gm: dict[RowKey, tuple[list[str], float]] = {pid: v for pid, v in gpu_labels(gpus, slurm=slurm).items()}
        step_of: dict[RowKey, str] = {pid: s for pid, s in (steps or {}).items()}
        if gpu is not None:
            on_gpu = {p.pid for g in gpus if g.index == gpu for p in g.processes}
            rows: list[tuple[RowKey, Proc]] = [(p.pid, p) for p in procs if p.pid in on_gpu]
        else:
            rows = [(p.pid, p) for p in procs if show_others or me is None or p.user == me]
        self._fill(rows, gm, step_of, {}, me=me, slurm=slurm, sort=sort, reverse=reverse, task_labels=task_labels)

    def show_live(self, nodes: Sequence[NodeRows], *, sort: str, reverse: bool) -> None:
        """A Slurm job's processes from an Attach sample (STEP column, Slurm IDX, D / Z highlighted). Several nodes:
        rows, GPU joins and steps keyed by (node, pid), plus a NODE column."""
        multi = len(nodes) > 1
        rows: list[tuple[RowKey, Proc]] = []
        gm: dict[RowKey, tuple[list[str], float]] = {}
        step_of: dict[RowKey, str] = {}
        node_of: dict[RowKey, str] = {}
        for n in nodes:
            labels = gpu_labels(n.gpus, slurm=True)
            for p in n.procs:
                k: RowKey = (n.node, p.pid) if multi else p.pid
                if k in node_of:                 # the same pid twice in one node's ps: keep the first row
                    continue
                rows.append((k, p))
                node_of[k] = n.node
                if p.pid in labels:
                    gm[k] = labels[p.pid]
                if p.pid in n.steps:
                    step_of[k] = n.steps[p.pid]
        self._fill(rows, gm, step_of, node_of if multi else {}, me=None, slurm=True, sort=sort, reverse=reverse,
                   task_labels={})

    def _fill(self, rows: list[tuple[RowKey, Proc]], gm: Mapping[RowKey, tuple[list[str], float]],
              step_of: Mapping[RowKey, str], node_of: Mapping[RowKey, str], *, me: str | None, slurm: bool,
              sort: str, reverse: bool, task_labels: Mapping[Any, str]) -> None:
        keep = self.cursor_key                   # before _setup: a width change clears the columns and the cursor
        node_w = min(NODE_COL_MAX, max((len(n) for n in node_of.values()), default=0) + 1) if node_of else 0
        self._setup(self.size.width or self.app.size.width - 4, slurm, node_w)
        rows = sort_procs(rows, gm, sort, reverse)
        widths = {k: w for k, _l, w in self.cols}
        cells: dict[RowKey, tuple[Text, ...]] = {}
        for key, p in rows:
            idx, mem = gm.get(key, ([], 0.0))
            mine = me is None or p.user == me
            style = "" if mine else theme.muted()
            flagged = stat_style(p.stat) if slurm else None
            if flagged is not None:
                style = flagged
            row = {
                "gpu": Text(",".join(idx) or "-", style=theme.accent() if idx else theme.dim()),
                "node": Text(humanize.truncate(node_of.get(key, ""), max(1, widths.get("node", 1) - 1)),
                             style=theme.muted()),
                "pid": Text(str(p.pid), style=style),
                "user": Text(humanize.truncate(p.user, 9), style=style),
                "gmem": Text(humanize.gib(mem) if idx else humanize.NONE, style=style if idx else theme.dim()),
                "cpu": Text(f"{p.cpu_pct:.1f}" if p.cpu_pct < 100 else f"{p.cpu_pct:.0f}", style=style),
                "mem": Text(f"{p.mem_pct:.1f}", style=style),
                "time": Text(humanize.clock(p.etime_s), style=theme.muted()),
                "task": Text(task_labels.get(p.task, humanize.NONE) if p.task else humanize.NONE,
                             style=theme.accent() if p.task and mine else theme.dim()),
                "step": Text(step_of.get(key) or humanize.NONE, style=theme.muted() if key in step_of
                             else theme.dim()),
                "cmd": Text(humanize.truncate(_cmd(p) if slurm else humanize.command_label(p.args),
                                              widths["cmd"] - 1), style=style),
            }
            cells[key] = tuple(row[k] for k, _l, _w in self.cols)
        order = [key for key, _p in rows]
        if order == self.order:
            for key in order:
                if cells[key] == self._cells.get(key):
                    continue
                for (col, _l, _w), value in zip(self.cols, cells[key], strict=True):
                    self.update_cell(row_id(key), col, value)
        else:
            self.clear()
            for key in order:
                self.add_row(*cells[key], key=row_id(key))
            ids = [row_id(key) for key in order]
            if keep is not None and keep in ids:
                self.move_cursor(row=ids.index(keep))
        self.order = order
        self._cells = cells
        self.procs = dict(rows)

    def current_proc(self) -> Proc | None:
        key = self.cursor_key
        if not key:
            return None
        return next((p for k, p in self.procs.items() if row_id(k) == key), None)
