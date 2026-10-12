"""[WP3] Attach panel (§7.16): status line + live per-GPU rows, cgroup line and hints for JobDetail's GPU tab.

Pure renderers of :class:`~runfleet.models.AttachState` / :class:`~runfleet.models.JobLive`; the screen re-reads
``FleetAPI.attach_state(key)`` on ``FleetChanged.live_keys`` and on the 1 Hz clock (the next-sample countdown). The
GPU rows reuse the nvitop-style bars of ``widgets/gpu_panel.py``; the Procs tab reuses ``widgets/proc_table.py`` with
a STEP column (:func:`live_procs`; rows keyed by (node, pid) when the sample has several nodes).

The status line, per state::

    Live · ssh n-38-01 (job 723954) · 12:00:00 · 0.67 s · every 15 s, next 12 s · clean ✓     r now · a auto on/off
    Live · srun s71-02 · 21:56:38 · 1.21 s · manual · 2 sacct rows added this session · clean ✓  r read again (asks)
    srun needed · lyra refuses ssh to compute nodes                       r read with srun (asks first)
    Live readings unavailable · attach is off for vega
"""

from __future__ import annotations

import re
from datetime import datetime

from rich.text import Text
from textual.widgets import Static

from runfleet.models import (GPU, AttachState, AttachStatus, CgroupStats, GPUProcess, Job, JobGpuUsage, JobLive,
                             NodeInspect, Proc)
from runfleet.ui import render, theme
from runfleet.ui.widgets.proc_table import NodeRows
from runfleet.util import humanize

IDLE_STREAK = 2                   # samples with util < 5 % and memory held before a GPU reads "idle" (§7.16.5)
MEM_HINT = 0.9                    # cgroup memory above this share of memory.max is amber


# ---------------------------------------------------------------- small helpers
def short_bus(bus: str | None) -> str:
    """``00000000:89:00.0`` -> ``89:00.0`` (the PCI domain is always 0 on these nodes)."""
    if not bus:
        return "?"
    parts = bus.split(":")
    return ":".join(parts[-2:]) if len(parts) >= 2 else bus


def hms(dt: datetime | None) -> str:
    return dt.astimezone().strftime("%H:%M:%S") if dt is not None else humanize.NA


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 'es' if word.endswith('s') else 's'}"


def idle_streak(live: JobLive, g: GPU) -> int:
    return live.idle_streak.get(g.uuid or "", 0) or live.idle_streak.get(g.bus_id or "", 0)


def landed_elsewhere(live: JobLive | None) -> NodeInspect | None:
    """The first node where ssh landed in another of the user's jobs (its GPU rows are that job's)."""
    if live is None:
        return None
    return next((n for n in live.nodes if n.gpus_of_target is False), None)


def _nodes_text(state: AttachState, live: JobLive | None) -> str:
    names = [n.node for n in live.nodes] if live is not None and live.nodes else ([state.node] if state.node else [])
    if not names:
        return ""
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


# ---------------------------------------------------------------- status line
def status_lines(state: AttachState | None, job: Job | None, now: datetime, width: int) -> list[Text]:
    """The Attach status line (fitted, keys right-aligned) plus at most two explanation lines (blocked hint, offer,
    last failure)."""
    if state is None:
        if job is not None and job.state.is_terminal:
            return [Text("live readings: only while the job runs", style=theme.dim())]
        return [Text("live readings start when this tab is open on a running job", style=theme.dim())]
    live = state.live
    st = state.status
    left = Text()
    extra: list[Text] = []
    keys = ""
    if st is AttachStatus.BLOCKED:
        left.append("Live readings unavailable", style=theme.warn())
        left.append(f" · {state.blocked or 'not possible now'}", style=theme.muted())
        if state.hint:
            extra.append(Text(state.hint, style=theme.muted()))
    elif st is AttachStatus.NEEDS_CONSENT and live is None:
        left.append("srun needed", style=theme.warn())
        if state.offer is not None:
            left.append(f" · {state.offer.reason}", style=theme.muted())
        keys = "r read with srun (asks first)"
        extra.append(Text("each srun read adds one permanent sacct row (rfprobe); nothing runs until you press r "
                          "and confirm", style=theme.dim()))
    else:
        word, style = {
            AttachStatus.SAMPLING: ("Reading…", theme.accent()),
            AttachStatus.PAUSED: ("Paused", theme.warn()),
            AttachStatus.STOPPED: ("Stopped", theme.muted()),
        }.get(st, ("Live", theme.ok()))
        if live is None and st in (AttachStatus.LIVE, AttachStatus.IDLE):
            word, style = ("No reading yet", theme.warn()) if state.error is not None else \
                ("Waiting for the first sample", theme.muted())
        left.append(word, style=f"bold {style}".strip() if style == theme.ok() else style)
        if st in (AttachStatus.PAUSED, AttachStatus.STOPPED) and state.stopped:
            left.append(f" ({state.stopped})", style=theme.muted())
        mode = (live.mode if live is not None else None) or state.mode
        parts: list[Text] = []
        where = _nodes_text(state, live)
        if mode == "srun":
            parts.append(Text(f"srun {where}".strip()))
        elif mode == "ssh":
            landed = live.nodes[0].landed_job if live is not None and live.nodes else None
            parts.append(Text(f"ssh {where}".strip() + (f" (job {landed})" if landed else "")))
        if live is not None:
            parts.append(Text(hms(live.sampled_at)))
            if width >= 100:
                parts.append(Text(f"{live.cost_ms / 1000:.2f} s", style=theme.muted()))
        if mode == "ssh":
            if not state.auto:
                parts.append(Text("auto off", style=theme.warn()))
            elif state.refresh_s:
                every = f"every {state.refresh_s:.0f} s"
                if st is AttachStatus.LIVE and state.next_at is not None:
                    nxt = f"next {max(0, round((state.next_at - now).total_seconds()))} s"
                    every = f"{every}, {nxt}" if width >= 100 else nxt
                parts.append(Text(every, style=theme.muted()))
            keys = "r now · a auto on/off"
        elif mode == "srun":
            parts.append(Text("manual", style=theme.muted()))
            if state.srun_rows:
                parts.append(Text(f"{_plural(state.srun_rows, 'sacct row')} added this session", style=theme.warn()))
            keys = "r read again" + (" (allowed)" if state.consent == "job" else " (asks)")
        if live is not None and live.clean is True:
            parts.append(Text("clean ✓", style=theme.ok()))
        elif live is not None and live.clean is False:
            parts.append(Text(" leftovers ", style=f"bold reverse {theme.RED}" if not theme.no_color() else "reverse"))
        for p in parts:
            left.append(" · ", style=theme.dim())
            left.append_text(p)
        if live is not None and live.clean is False and live.audit_detail:
            extra.append(Text(f"audit: {live.audit_detail}", style=theme.error()))
        if state.offer is not None and mode == "ssh":
            # ssh landed in another job, or cannot reach the node (host key, DNS, refused): srun is the way in
            why = "" if landed_elsewhere(live) is not None else f"{state.offer.reason} · "
            extra.append(Text(f"{why}S reads job {state.key.id}'s own GPUs with srun on {state.offer.node} (asks "
                              "first; one permanent sacct row)", style=theme.muted()))
    if state.error is not None and st is not AttachStatus.BLOCKED:
        extra.append(Text(f"last read failed: {state.error.summary}", style=theme.error()))
    return _with_keys(left, keys, width) + [render.fit(x, width) for x in extra[:3]]


def _with_keys(left: Text, keys: str, width: int) -> list[Text]:
    """``left`` and the key hint right-aligned on one line, or the hint on a line of its own when both do not fit."""
    if not keys:
        return [render.fit(left, width)]
    right = Text(keys, style=theme.accent())
    if left.cell_len + right.cell_len + 3 > width:
        return [render.fit(left, width), render.fit(right, width)]
    return [Text.assemble(left, " " * (width - left.cell_len - right.cell_len), right)]


# ---------------------------------------------------------------- body
def _gpu_procs(g: GPU, procs: dict[int, Proc], target: str) -> Text:
    out = Text("↳ ", style=theme.dim())
    if not g.processes:
        out.append("no compute processes", style=theme.dim())
        return out
    for i, p in enumerate(sorted(g.processes, key=lambda x: -(x.used_mem_mib or 0))):
        if i:
            out.append(" · ", style=theme.dim())
        out.append(str(p.pid), style=theme.muted())
        out.append(" " + humanize.truncate(_proc_cmd(p, procs), 28))
        out.append(f" {humanize.gib(p.used_mem_mib)}", style=theme.muted())
        if p.job is not None and p.job.id != target:
            out.append(f" (job {p.job.id})", style=theme.warn())
    return out


def _proc_cmd(p: GPUProcess, procs: dict[int, Proc]) -> str:
    row = procs.get(p.pid)
    return humanize.command_label(row.args if row is not None else (p.command or "?"))


def gpu_rows(live: JobLive, node: NodeInspect, width: int, util_t: tuple[float, float],
             mem_t: tuple[float, float]) -> list[Text]:
    """Header + per GPU one bar row (IDX, bus, util, memory, temp, power, model, idle flag) and its processes."""
    gpus = sorted(node.gpus, key=lambda g: (g.slurm_idx if g.slurm_idx is not None else 99, g.bus_id or ""))
    wide = width >= 100
    flag = "! idle but holding {}" if wide else "! idle, {} held"
    flags = {id(g): (flag.format(humanize.gib(g.mem_used_mib))
                     if idle_streak(live, g) >= IDLE_STREAK and node.gpus_of_target is not False else "")
             for g in gpus}
    flag_w = max((len(f) for f in flags.values()), default=0)
    name_w = 16 if wide else 0

    bus_w = 9

    def bar_for(name: int) -> int:
        fixed = 4 + 2 + bus_w + 5 + 2 + 1 + 12 + 2 + 5 + (2 + 9 + (2 + name if name else 0) if wide else 0) + 4
        return (width - fixed - (flag_w + 2 if flag_w else 0)) // 2

    if bar_for(name_w) < 12:          # the model is the same on every row (and in Resources): bars first
        name_w = 0
    if bar_for(name_w) < 6:           # 80 columns with an idle flag: the bus id goes before the bars shrink further
        bus_w = 0
    bar = max(6, min(20, bar_for(name_w)))
    head = Text(f"{'IDX':>4}  " + (f"{'bus':<8} " if bus_w else "") + f"{'util':<{bar + 7}}  {'mem':<{bar + 15}}  "
                f"{'temp':>5}", style=f"bold {theme.muted()}")
    if wide:
        head.append(f"  {'power':>9}" + ("  GPU" if name_w else ""))
    lines = [render.fit(head, width)]
    procs = {p.pid: p for p in node.procs}
    for g in gpus:
        out = Text(f"{g.slurm_idx if g.slurm_idx is not None else '?':>4}  ",
                   style="bold" if g.slurm_idx is not None else theme.dim())
        if bus_w:
            out.append(f"{short_bus(g.bus_id):<8} ", style=theme.muted())
        out.append_text(render.gpu_bar(g.util_pct, bar + 2, util_t))
        out.append(humanize.pct(g.util_pct, width=5))
        out.append("  ")
        mem_pct = g.mem_used_mib / g.mem_total_mib * 100 if g.mem_used_mib is not None and g.mem_total_mib else \
            g.mem_util_pct
        out.append_text(render.gpu_bar(mem_pct, bar + 2, mem_t))
        out.append(" " + humanize.gib_pair(g.mem_used_mib, g.mem_total_mib).rjust(12))
        temp = f"{g.temp_c:.0f}°C" if g.temp_c is not None else humanize.NA
        out.append("  " + temp.rjust(5), style=theme.temp_style(g.temp_c))
        if wide:
            pwr = (f"{g.power_w:.0f}/{g.power_limit_w:.0f}W" if g.power_limit_w else f"{g.power_w:.0f}W") \
                if g.power_w is not None else humanize.NA
            out.append("  " + pwr.rjust(9), style=theme.muted())
            if name_w:
                out.append("  " + humanize.truncate(humanize.short_gpu_name(g.name), name_w - 1).ljust(name_w))
        if flags[id(g)]:
            out.append("  " + flags[id(g)], style=theme.warn())
        lines.append(render.fit(out, width))
        lines.append(render.fit(Text("      ").append_text(_gpu_procs(g, procs, node.target_job)), width))
    if not gpus:
        lines.append(Text("  no GPU rows from this node (nvidia-smi missing, hung or no GPU in the job)",
                          style=theme.dim()))
    return lines


def cgroup_line(cg: CgroupStats, cpus: int | None) -> Text:
    """``mem 134G / 256G (peak 188G, shmem 56.3G) · oom_kill 0 · CPU ≈ 3.6 / 16 cores``."""
    out = Text()
    hot = cg.mem_current is not None and cg.mem_max and cg.mem_current > MEM_HINT * cg.mem_max
    out.append(f"mem {humanize.size(cg.mem_current)} / "
               f"{humanize.size(cg.mem_max) if cg.mem_max is not None else 'no limit'}",
               style=theme.warn() if hot else "")
    extra = []
    if cg.mem_peak is not None:
        extra.append(f"peak {humanize.size(cg.mem_peak)}")
    if cg.shmem:
        extra.append(f"shmem {humanize.size(cg.shmem)}")
    if extra:
        out.append(f" ({', '.join(extra)})", style=theme.muted())
    if cg.oom_kill is not None:
        out.append(" · ", style=theme.dim())
        out.append(f"oom_kill {cg.oom_kill}", style=theme.error() if cg.oom_kill > 0 else theme.muted())
    out.append(" · ", style=theme.dim())
    if cg.cpu_cores is not None:
        out.append(f"CPU ≈ {cg.cpu_cores:.1f}" + (f" / {cpus} cores" if cpus else " cores"))
    else:
        out.append("CPU: next sample", style=theme.dim())
    return out


def accounting_line(usage: JobGpuUsage | None, now: datetime, gpu_tres: bool | None = None) -> Text:
    """The Slurm accounting reading kept under the live rows; a multi-GPU sum is phrased as a sum (``Σ100% / 2
    GPUs``), never as a mean that reads like half the GPUs idle."""
    out = Text("accounting ", style=theme.label())
    if usage is None:
        out.append("n/a — " + ("no GPU accounting on this cluster" if gpu_tres is False else "no sstat reading yet"),
                   style=theme.dim())
        return out
    age = f", {humanize.ago(usage.sampled_at, now)}" if usage.sampled_at is not None else ""
    out.append(f"({usage.source}{age})  ", style=theme.muted())
    if usage.n_gpus > 1 and usage.util_sum_pct is not None:
        out.append(f"Σ{humanize.pct(usage.util_sum_pct)} / {usage.n_gpus} GPUs")
    else:
        out.append(humanize.pct(usage.util_mean_pct))
    if usage.mem_used_mib is not None:
        out.append(f" · gpumem {humanize.gib(usage.mem_used_mib)}", style=theme.muted())
    return out


_IDLE_HINT = re.compile(r"^GPU\S* idle\b")


def _hint_style(hint: str) -> str:
    return theme.error() if hint.startswith("oom_kill") else render.hint_style(hint)


def live_lines(state: AttachState | None, job: Job, usage: JobGpuUsage | None, now: datetime, width: int,
               util_t: tuple[float, float], mem_t: tuple[float, float], gpu_tres: bool | None = None) -> list[Text]:
    """The whole Live box of the GPU tab: status, one block per node (GPU rows, processes, cgroup), hints, errors,
    the accounting line."""
    lines = status_lines(state, job, now, width)
    live = state.live if state is not None else None
    if live is not None and live.nodes:
        multi = len(live.nodes) > 1
        for node in live.nodes:
            lines.append(Text(""))
            if multi or node.errors:
                head = Text(node.node, style="bold")
                if node.cost_ms is not None:
                    head.append(f" · {node.cost_ms / 1000:.2f} s", style=theme.muted())
                lines.append(render.fit(head, width))
            if node.gpus_of_target is False:
                lines.append(render.fit(Text(
                    f"ssh landed in job {node.landed_job} on {node.node}; GPUs shown belong to {node.landed_job} "
                    "(ssh can only enter your newest job on a node)", style=theme.warn()), width))
            lines += gpu_rows(live, node, width, util_t, mem_t)
            cg = node.cgroups.get(node.target_job)
            cpus = job.alloc.cpus if len(live.nodes) == 1 else None
            if cg is not None:
                lines.append(render.fit(render.label_value("cgroup", cgroup_line(cg, cpus), width=8), width))
            others = [(j, c) for j, c in node.cgroups.items() if j != node.target_job]
            if others:
                text = " · ".join(f"{j} mem {humanize.size(c.mem_current)}" for j, c in others)
                lines.append(render.fit(render.label_value("also here", Text(text, style=theme.dim()), width=10),
                                        width))
            for err in node.errors:
                lines.append(render.fit(Text(f"! {err.summary}", style=theme.error()), width))
            if node.stray_nvidia_smi:
                lines.append(render.fit(Text(f"! {_plural(node.stray_nvidia_smi, 'hung nvidia-smi')} on {node.node}",
                                             style=theme.error()), width))
            if node.other_nvidia_smi:            # a job's own `nvidia-smi -l` etc.: informational, not a leftover
                lines.append(render.fit(Text(f"{node.other_nvidia_smi} other nvidia-smi of yours on {node.node} "
                                             "(not RunFleet's, not hung)", style=theme.dim()), width))
        lines.append(Text(""))
        hints = Text()
        flagged = any(idle_streak(live, g) >= IDLE_STREAK for n in live.nodes if n.gpus_of_target is not False
                      for g in n.gpus)
        shown = [h for h in live.hints if not (flagged and _IDLE_HINT.match(h))]   # the GPU row says it already
        for i, h in enumerate(shown):
            hints.append((" · " if i else "") + h, style=_hint_style(h))
        if not live.hints:
            hints.append("none", style=theme.dim())
        if hints:
            lines.append(render.fit(render.label_value("hints", hints, width=8), width))
        for err in live.errors:
            lines.append(render.fit(Text(f"! {err.summary}", style=theme.error()), width))
    elif live is not None and live.errors:
        lines += [render.fit(Text(f"! {e.summary}", style=theme.error()), width) for e in live.errors]
    lines.append(render.fit(accounting_line(usage, now, gpu_tres), width))
    return lines


def box_title(state: AttachState | None) -> str:
    """The status line says the mode; the title says what the box is (and that the next box is accounting)."""
    return "Live readings (node)" if state is not None and state.live is not None else "Live readings"


# ---------------------------------------------------------------- Procs tab
def live_procs(live: JobLive | None) -> list[NodeRows]:
    """Rows for :meth:`ProcTable.show_live`, one entry per node: its ``ps`` rows, its GPUs (GPU / GPU-MEM columns)
    and pid -> step. Pids are per node — the same pid on two nodes is two processes — so nothing is merged here."""
    return [NodeRows(node=n.node, procs=n.procs, gpus=n.gpus, steps=n.step_of_pid)
            for n in (live.nodes if live is not None else ())]


def procs_note(state: AttachState | None, job: Job, now: datetime, width: int) -> list[Text]:
    """Status line (same as the GPU tab: r / a work here too), what the table lists, the landed-elsewhere note and
    D-state / zombie counts."""
    lines = status_lines(state, job, now, width)
    live = state.live if state is not None else None
    if live is None or not live.has_data:
        return lines
    procs: list[Proc] = [p for rows in live_procs(live) for p in rows.procs]
    nodes = ", ".join(n.node for n in live.nodes)
    what = Text(f"{_plural(len(procs), 'process')} of yours on {nodes} (all your jobs there; STEP from scontrol "
                "listpids)", style=theme.muted())
    stuck = sum(1 for p in procs if p.stat.startswith("D"))
    zombies = sum(1 for p in procs if p.stat.startswith("Z"))
    if stuck:
        what.append(f" · {stuck} in D state", style=theme.warn())
    if zombies:
        what.append(f" · {_plural(zombies, 'zombie')}", style=theme.error())
    lines.append(render.fit(what, width))
    node = landed_elsewhere(live)
    if node is not None:
        lines.append(render.fit(Text(f"ssh landed in job {node.landed_job} on {node.node}: the GPU columns are that "
                                     "job's GPUs", style=theme.warn()), width))
    return lines


class AttachPanel(Static):
    """The Live box of JobDetail's GPU tab (a Static: lines are pre-fitted to the box width)."""

    DEFAULT_CSS = """
    AttachPanel { height: auto; }
    """

    def show(self, state: AttachState | None, job: Job, usage: JobGpuUsage | None, now: datetime,
             util_t: tuple[float, float], mem_t: tuple[float, float], width: int | None = None,
             gpu_tres: bool | None = None) -> None:
        w = width or max(40, (self.size.width or self.app.size.width - 4) - 2)
        self.border_title = box_title(state)
        self.update(Text("\n").join(live_lines(state, job, usage, now, w, util_t, mem_t, gpu_tres)))
