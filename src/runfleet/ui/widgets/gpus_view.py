"""[WP3] GPUs view across hosts (mockup M).

One row per workstation GPU (live nvidia-smi readings) plus one row per running Slurm job with GPUs (the job's
aggregate from Slurm accounting, marked ``*``: per-GPU Slurm readings need compute-node access, v0.2). Freshness is per
row: a host that answers while its GPU section fails keeps the last readings, which are shown stale (``~``, dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Static

from runfleet.models import GPU, ConnState, Job, JobKey, JobKind, StateGroup
from runfleet.ui import keymap, render, theme
from runfleet.ui.widgets.job_table import ClickTable
from runfleet.util import humanize

COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("mark", "", 2), ("host", "HOST", 11), ("node", "NODE", 9), ("idx", "IDX", 4), ("name", "NAME", 16),
    ("util", "UTIL", 19), ("mem", "MEM", 25), ("temp", "TEMP", 5), ("pwr", "PWR", 6), ("job", "JOB / PROCESS", 18),
)
NARROW_DROP = ("node", "temp", "pwr")
NODE_MAX = 18
SORTS = ("host", "util", "mem", "temp")
NOTE = "* per-job aggregate from Slurm accounting (sstat), not a per-GPU reading · per-GPU Slurm readings are v0.2"


@dataclass(frozen=True, slots=True)
class GpuRow:
    key: str
    host: str
    node: str
    idx: str
    name: str
    util: float | None
    mem_used: float | None
    mem_total: float | None
    temp: float | None
    power: float | None
    owner: JobKey | None
    owner_text: str
    aggregate: bool
    stale: bool
    age_s: float | None = None   # age of a reading kept from an older sample (the host itself answers), else None


def task_short_name(name: str) -> str:
    """``orb.launch ×2 (orb.train)`` -> ``train``; ``python (eval.py)`` -> ``eval``; ``train.py`` -> ``train``."""
    inner = name.rsplit("(", 1)[-1].rstrip(")") if "(" in name else name.split(" ")[0]
    for ext in (".py", ".sh"):
        if inner.endswith(ext):
            inner = inner[: -len(ext)]
    return inner.rsplit(".", 1)[-1].rsplit("/", 1)[-1] or name


def sample_age(sampled_at: datetime | None, now: datetime, interval: float) -> float | None:
    """Age of a reading that missed its refresh (older than 3 periods + 5 s), else None."""
    if sampled_at is None:
        return None
    age = (now - sampled_at).total_seconds()
    return age if age > 3 * interval + 5 else None


def ws_row(gpu: GPU, stale: bool, jobs: dict[JobKey, Job], age_s: float | None = None) -> GpuRow:
    owner: JobKey | None = None
    text = humanize.NONE
    procs = sorted(gpu.processes, key=lambda p: -(p.used_mem_mib or 0))
    if procs:
        p = procs[0]
        owner = p.job
        job = jobs.get(p.job) if p.job is not None else None
        if job is not None:
            short = task_short_name(job.name)
            text = f"{job.key.label} {short}" if job.is_mine else f"{job.user} {short}"
        else:
            text = f"{p.user or '?'}/{p.pid}"
        if len(procs) > 1:
            text += f" +{len(procs) - 1}"
    return GpuRow(f"gpu:{gpu.host}:{gpu.index}", gpu.host, humanize.NONE, str(gpu.index),
                  humanize.short_gpu_name(gpu.name), gpu.util_pct, gpu.mem_used_mib, gpu.mem_total_mib,
                  gpu.temp_c, gpu.power_w, owner, text, False, stale or age_s is not None, age_s)


def job_row(job: Job, stale: bool, age_s: float | None = None) -> GpuRow:
    idx = sorted({i for n in job.alloc.per_node for i in n.gpu_indices})
    if len(idx) > 1 and idx == list(range(idx[0], idx[-1] + 1)):
        idx_text = f"{idx[0]}-{idx[-1]}"
    else:
        idx_text = ",".join(map(str, idx)) or humanize.NONE
    name = humanize.short_gpu_name(job.alloc.gpu_type) or "GPU"
    if job.alloc.gpus > 1:
        name += f" ×{job.alloc.gpus}"
    usage = job.gpu
    return GpuRow(f"job:{job.key}", job.key.host, job.nodelist or humanize.NONE, idx_text, name,
                  usage.util_mean_pct if usage else None, usage.mem_used_mib if usage else None,
                  usage.mem_total_mib if usage else None, None, None, job.key, job.key.label, True,
                  stale or age_s is not None, age_s)


class GpuTable(ClickTable):
    """GPU rows; activation opens the owning job / task (or the host)."""

    # the focused table sees these before GPUsView: same moves as keymap "gpus" (g/home, G/end)
    BINDINGS = [Binding("home", "first_row", show=False), Binding("end", "last_row", show=False)]


class GPUsView(Widget):
    BINDINGS = keymap.bindings("gpus")
    DEFAULT_CSS = """
    GPUsView { height: 1fr; }
    GPUsView > #gpus-note { height: 1; color: $text-muted; padding: 0 1; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.sort = "host"
        self.reverse = False
        self.rows: list[GpuRow] = []
        self._cols: tuple[str, ...] = ()
        self._cols_sig: tuple[object, ...] = ()
        self._cells: dict[str, tuple[Text, ...]] = {}

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        yield GpuTable(id="gpus-table", cursor_type="row", show_row_labels=False, cell_padding=0)
        yield Static(Text(NOTE), id="gpus-note")

    @property
    def table(self) -> GpuTable:
        return self.query_one(GpuTable)

    def on_resize(self) -> None:
        self.refresh_data()

    def build_rows(self) -> list[GpuRow]:
        now = self.rf.now()
        rows: list[GpuRow] = []
        others = bool(getattr(self.rf, "others_visible", False))      # Jobs view `u` (UI-19: "my" jobs by default)
        for host in self.rf.scope_hosts():
            state = self.rf.store.hosts.get(host)
            if state is None or state.snapshot is None:
                continue
            stale = state.stale(now)
            tick = state.status.tick_interval or 0.0
            slower = max(1.0, float(getattr(self.rf, "multiplier", 1.0)))       # `-` slows every cadence
            gpu_every = max(tick, state.config.workstation.poll_gpu * slower)
            usage_every = max(tick, state.config.slurm.poll_usage * slower)
            jobs = {j.key: j for j in state.snapshot.jobs.values()}
            for gpu in sorted(state.snapshot.gpus, key=lambda g: g.index):
                age = None if stale else sample_age(gpu.sampled_at, now, gpu_every)
                rows.append(ws_row(gpu, stale, jobs, age))
            for job in state.snapshot.jobs.values():
                if (job.kind is JobKind.SLURM and job.state.group is StateGroup.ACTIVE and job.alloc.gpus
                        and (job.is_mine or others)):
                    age = None if stale or job.gpu is None else sample_age(job.gpu.sampled_at, now, usage_every)
                    rows.append(job_row(job, stale, age))
        order = {h: i for i, h in enumerate(self.rf.host_names())}

        def first_index(r: GpuRow) -> int:
            digits = ""
            for ch in r.idx:
                if not ch.isdigit():
                    break
                digits += ch
            return int(digits) if digits else 1 << 20

        # default: host order, then node, then the physical GPU index (numeric: 0, 1, 2, … 10)
        rows.sort(key=lambda r: (order.get(r.host, 99), r.aggregate is False, r.node, first_index(r), r.key))
        if self.sort != "host":
            def value(r: GpuRow) -> float:
                if self.sort == "util":
                    return r.util if r.util is not None else -1
                if self.sort == "mem":
                    return r.mem_used if r.mem_used is not None else -1
                return r.temp if r.temp is not None else -1
            rows.sort(key=value, reverse=not self.reverse)
        elif self.reverse:
            rows.reverse()
        return rows

    def refresh_data(self) -> None:
        if not self.is_mounted:
            return
        width = self.size.width or self.app.size.width
        cols = tuple(k for k, _l, _w in COLUMNS if width >= 110 or k not in NARROW_DROP)
        table = self.table
        widths = {k: w for k, _l, w in COLUMNS}
        if width < 110:
            widths |= {"util": 15, "mem": 21, "name": 14}
        rows = self.build_rows()
        if "node" in cols:      # widen NODE for long names (lyra-gpu-s71-02) as far as the width allows
            longest = max((len(r.node) for r in rows), default=0) + 1
            room = width - sum(widths[k] for k in cols) - 2
            widths["node"] = max(widths["node"], min(NODE_MAX, longest, widths["node"] + max(0, room)))
        if (cols, widths["node"]) != self._cols_sig:
            keep = self.current_key()
            table.clear(columns=True)
            for k, label, _w in COLUMNS:
                if k in cols:
                    table.add_column(Text(label, style="bold"), key=k, width=widths[k])
            self._cols = cols
            self._cols_sig = (cols, widths["node"])
            self._cells = {}
            self._restore = keep
        thresholds = self.rf.service.config.ui
        cells = {r.key: self._cells_for(r, cols, widths, thresholds.gpu_util_thresholds,
                                        thresholds.gpu_mem_thresholds) for r in rows}
        keys = [r.key for r in rows]
        if keys == [r.key for r in self.rows] and self._cells:
            for k in keys:
                if cells[k] != self._cells.get(k):
                    for col, value in zip(cols, cells[k], strict=True):
                        table.update_cell(k, col, value)
        else:
            keep = getattr(self, "_restore", None) or self.current_key()
            table.clear()
            for k in keys:
                table.add_row(*cells[k], key=k)
            if keep in keys:
                table.move_cursor(row=keys.index(keep))
            self._restore = None
        self.rows = rows
        self._cells = cells
        self.query_one("#gpus-note", Static).update(self._note(rows))
        screen = self.screen
        if hasattr(screen, "update_tabs"):
            screen.update_tabs()

    def refresh_time(self) -> None:
        pass

    def _cells_for(self, r: GpuRow, cols: tuple[str, ...], widths: dict[str, int],
                   util_t: tuple[float, float], mem_t: tuple[float, float]) -> tuple[Text, ...]:
        star = " *" if r.aggregate and (r.util is not None or r.mem_used is not None) else "  "
        out: list[Text] = []
        for k in cols:
            w = widths[k] - 1
            if k == "mark":
                cell = Text("~" if r.stale else " ", style=theme.dim())
            elif k == "host":
                idx = self.rf.host_index(r.host)
                cell = Text(r.host, style=theme.host_color(self.rf.store.hosts[r.host].config, idx))
            elif k == "node":           # cut from the left: s71-02 vs s71-03 must stay distinguishable
                cell = Text(humanize.truncate_left(r.node, w), style=theme.muted())
            elif k == "idx":
                cell = Text(r.idx)
            elif k == "name":
                cell = Text(humanize.truncate(r.name, w))
            elif k == "util":
                cell = render.gpu_bar(r.util, max(4, w - 7), util_t) + Text(humanize.pct(r.util, width=5)) + \
                    Text(star, style=theme.warn())
            elif k == "mem":
                pct = r.mem_used / r.mem_total * 100 if r.mem_used is not None and r.mem_total else None
                pair = humanize.gib_pair(r.mem_used, r.mem_total)
                cell = render.gpu_bar(pct, max(4, w - 13), mem_t) + Text(" " + pair.rjust(10)) + \
                    Text(star, style=theme.warn())
            elif k == "temp":
                cell = Text(f"{r.temp:.0f}°C" if r.temp is not None else humanize.NONE,
                            style=theme.temp_style(r.temp))
            elif k == "pwr":
                cell = Text(f"{r.power:.0f}W" if r.power is not None else humanize.NONE, style=theme.muted())
            else:
                cell = Text(r.owner_text, style="" if r.owner is not None else theme.dim())
            if cell.cell_len > w:
                cell.truncate(w, overflow="ellipsis")
            if r.stale and k != "mark":
                cell.stylize(theme.dim())
            out.append(cell)
        return tuple(out)

    def _note(self, rows: list[GpuRow]) -> Text:
        """Footnote; with no rows it says why (a host that is down / needs auth / has not answered yet)."""
        if rows:
            old: dict[str, float] = {}
            for r in rows:
                if r.age_s is not None:
                    old[r.host] = max(old.get(r.host, 0.0), r.age_s)
            if not old:
                return Text(NOTE)
            out = Text("~ last readings, not refreshing: ", style=theme.warn())
            out.append(", ".join(f"{h} {humanize.age(a)} old" for h, a in old.items()), style=theme.warn())
            out.append(f" · {NOTE}", style=theme.muted())
            return out
        reasons = []
        for host in self.rf.scope_hosts():
            state = self.rf.store.hosts.get(host)
            if state is None:
                continue
            st = state.status.state
            if theme.is_problem(st) or st is ConnState.DISABLED:
                hint = ""
                if st in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
                    # `c` here connects the scoped host only (no row is selected), and only where it is bound
                    here = keymap.find("gpus", "app.connect_host") is not None and self.rf.scope == host
                    hint = " — c to log in" if here else " — log in from Fleet (c)"
                reasons.append(f"{host}: {theme.CONN_LABEL[st]}{hint}")
            elif state.snapshot is None:
                reasons.append(f"{host}: waiting for the first probe")
        text = " · ".join(reasons) if reasons else "no GPUs in scope (no workstation GPU, no running GPU job)"
        return Text(text, style=theme.warn() if reasons else theme.dim())

    def status_text(self, width: int) -> Text:
        n_ws = sum(1 for r in self.rows if not r.aggregate)
        n_job = sum(1 for r in self.rows if r.aggregate)
        arrow = "▼" if (self.sort != "host") != self.reverse else "▲"     # util/mem/temp: largest first
        out = Text()
        if width >= 110:
            out.append("workstation GPUs (live) + GPUs of my running Slurm jobs (util via sstat)", style=theme.dim())
            out.append("      ")
        out.append(f"sort: {self.sort} {arrow}   ", style=theme.muted())
        out.append(f"{n_ws} GPUs + {n_job} jobs", style=theme.muted())
        return out

    # ------------------------------------------------------------------ selection / actions
    def current_key(self) -> str | None:
        table = self.table
        if not self.rows or table.cursor_row < 0 or table.cursor_row >= len(self.rows):
            return None
        return self.rows[table.cursor_row].key

    def current_row(self) -> GpuRow | None:
        key = self.current_key()
        return next((r for r in self.rows if r.key == key), None)

    def current_host(self) -> str | None:
        row = self.current_row()
        return row.host if row else None

    def current_job(self) -> Job | None:
        row = self.current_row()
        if row is None or row.owner is None:
            return None
        job = self.rf.store.job(row.owner)
        return job

    def on_click_table_activated(self, event: ClickTable.Activated) -> None:
        self.action_open()

    def on_click_table_context_menu(self, event: ClickTable.ContextMenu) -> None:
        self.rf.action_action_menu()

    def action_open(self) -> None:
        row = self.current_row()
        if row is None:
            return
        job = self.current_job()
        if job is not None:
            self.rf.open_job(job.key)
        else:
            self.rf.action_open_host(row.host)

    def action_cursor_down(self) -> None:
        self.table.action_cursor_down()

    def action_cursor_up(self) -> None:
        self.table.action_cursor_up()

    def action_first(self) -> None:
        self.table.move_cursor(row=0)

    def action_last(self) -> None:
        self.table.move_cursor(row=max(0, self.table.row_count - 1))

    def action_next_sort(self) -> None:
        self.sort = SORTS[(SORTS.index(self.sort) + 1) % len(SORTS)]
        self.reverse = False
        self.rf.toast(f"GPUs sorted by {self.sort}")
        self.refresh_data()

    def action_reverse_sort(self) -> None:
        self.reverse = not self.reverse
        self.refresh_data()
