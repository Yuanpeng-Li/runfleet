"""[WP3] Host card for the Fleet view (mockup A).

A card is a bordered box: the title carries number, name, kind and connection state; the body is six summary lines
for a Slurm host (my jobs, my GPUs, job util, free GPUs, next pending job, login node) or one line per GPU plus
host and task lines for a workstation. A host with a problem keeps its last data and swaps the bottom lines for the
classified error and the key that fixes it.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from datetime import datetime
from typing import Any

from rich.text import Text
from textual import events
from textual.widgets import Static

from runfleet.core.store import HostState
from runfleet.models import ConnState, ErrorKind, HostSnapshot, HostStatus, Job, JobKind, StateGroup
from runfleet.ui import render, theme
from runfleet.ui.query import ended_at
from runfleet.ui.widgets.gpus_view import sample_age
from runfleet.util import humanize

LABEL_W = 10
MAX_GPU_LINES = 8


def is_cluster(state: HostState) -> bool:
    snap = state.snapshot
    if snap is not None and snap.caps is not None and snap.caps.profile:
        return "slurm" in snap.caps.profile
    if state.config.backend != "auto":
        return state.config.backend == "slurm"
    return snap is not None and bool(snap.partitions or snap.nodes)


def kind_label(state: HostState) -> str:
    snap = state.snapshot
    if snap is None or snap.caps is None:
        return "…" if state.status.state in (ConnState.NEW, ConnState.CONNECTING) else ""
    if is_cluster(state):
        return f"slurm {snap.caps.slurm_version}" if snap.caps.slurm_version else "slurm"
    return "workstation"


def card_title(state: HostState, index: int, now: datetime, *, width: int = 58) -> Text:
    st = state.status
    out = Text(f"{index} ", style=theme.muted())
    out.append(st.name, style=f"bold {theme.host_color(state.config, index - 1)}")
    if state.config.group:
        out.append(f" [{state.config.group}]", style=theme.dim())
    kind = kind_label(state)
    out.append(f" · {kind} · " if kind else " · ", style=theme.muted())
    out.append_text(render.conn_text(st.state))
    ok = st.state in (ConnState.OK, ConnState.DEGRADED)
    if ok and st.latency_s is not None:
        out.append(f" {st.latency_s:.2f}s", style=theme.muted())
    if st.last_ok is not None:
        out.append(f" · {'' if ok else 'last ok '}{humanize.ago(st.last_ok, now)}", style=theme.muted())
    if out.cell_len > width:
        out.truncate(width, overflow="ellipsis")
    return out


# ---------------------------------------------------------------- shared pieces
def _counts_line(jobs: list[Job], now: datetime, window: float) -> Text:
    c: Counter[StateGroup] = Counter()
    for j in jobs:
        g = j.state.group
        if j.state.is_terminal:
            end = ended_at(j)
            if end is not None and (now - end).total_seconds() > window:
                continue
        c[g] += 1
    out = Text()
    for label, group in (("R", StateGroup.ACTIVE), ("PD", StateGroup.QUEUED), ("F", StateGroup.FAILED),
                         ("CD", StateGroup.DONE)):
        n = c[group] + (c[StateGroup.ENDED] if group is StateGroup.FAILED else 0)
        out.append(f"{label} ", style=theme.muted())
        out.append(str(n), style=theme.state_style_group(group) if n else theme.dim())
        out.append("   ")
    return out


def busy_note(status: HostStatus) -> str | None:
    """"controller busy — slowed 4×" while the poller backs off a busy Slurm controller (§5.4)."""
    factor = status.controller_slowdown
    return f"controller busy — slowed {factor:g}×" if factor > 1 else None


def _error_lines(state: HostState, now: datetime, width: int) -> list[Text]:
    """Two lines: the classified one-liner and what happens next / which key fixes it."""
    st = state.status
    err = st.error
    if err is None and st.state is ConnState.DEGRADED:
        bad = [s for s in st.sections.values() if not s.ok and s.error is not None]
        if bad:
            err = bad[0].error
    if err is None and st.state is ConnState.DISABLED:
        return [Text("○ polling disabled", style=theme.dim()), Text("  press t to enable", style=theme.dim())]
    if err is None:
        return []
    glyph = "✖" if st.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.UNREACHABLE) else "!"
    style = theme.conn_style(st.state) if st.state is not ConnState.OK else theme.warn()
    first = Text(f"{glyph} ", style=style)
    prefix = f"{err.section}: " if err.section else ""
    first.append(prefix + err.summary, style=style)
    if st.state is ConnState.NEEDS_AUTH or err.kind is ErrorKind.AUTH:
        second = "press c to log in (Duo / password prompts appear)"
    elif st.state is ConnState.HOSTKEY or err.kind is ErrorKind.HOSTKEY:
        second = "host key problem · e info · c lets ssh ask"
    else:
        bits = []
        if err.kind is ErrorKind.JUMP:
            bits.append("(via jump host)")
        if st.next_poll_at is not None and st.state in (ConnState.RETRYING, ConnState.UNREACHABLE):
            bits.append(f"next retry in {humanize.age(max(0.0, (st.next_poll_at - now).total_seconds()))}")
        bits.append("e info")
        second = " · ".join(bits)
    return [render.fit(first, width), render.fit(Text("  " + second, style=theme.muted()), width)]


# ---------------------------------------------------------------- Slurm
def _free_lines(snap: HostSnapshot, width: int) -> list[Text]:
    """Free GPUs per partition in `width` cells: one line in full, else one line without the "(N planned)" notes, else
    two lines (full, then short), the second ending in "+N more" when even that does not fit (host detail lists all)."""
    if not snap.partitions:
        return [Text(humanize.NA, style=theme.dim())]
    sep = Text(" · ", style=theme.dim())

    def item(p: Any, short: bool) -> Text:
        out = Text(p.name + " ", style=theme.muted())
        out.append_text(render.free_gpus(dataclasses.replace(p, gpus_free_planned=0) if short else p))
        return out

    def pack(items: list[Text], rows: int) -> list[Text] | None:
        out: list[Text] = []
        for it in items:
            if it.cell_len > width:
                return None
            if out and out[-1].cell_len + sep.cell_len + it.cell_len <= width:
                out[-1].append_text(sep)
                out[-1].append_text(it)
            elif len(out) < rows:
                out.append(it.copy())
            else:
                return None
        return out

    for rows, short in ((1, False), (1, True), (2, False), (2, True)):
        packed = pack([item(p, short) for p in snap.partitions], rows)
        if packed is not None:
            return packed
    items = [item(p, True) for p in snap.partitions]
    for keep in range(len(items) - 1, -1, -1):         # as many as fit, then "+N more"
        packed = pack([*items[:keep], Text(f"+{len(items) - keep} more", style=theme.dim())], 2)
        if packed is not None:
            return packed
    return [Text(f"{len(items)} partitions", style=theme.dim())]


def _slurm_lines(state: HostState, snap: HostSnapshot, now: datetime, width: int, window: float,
                 thresholds: tuple[float, float]) -> list[Text]:
    jobs = [j for j in snap.jobs.values() if j.is_mine]
    lines: list[Text] = []

    counts = render.label_value("my jobs", _counts_line(jobs, now, window), width=LABEL_W)
    tail = Text(f"(last {humanize.window(window)})", style=theme.dim())
    lines.append(_right(counts, tail, width))

    running = [j for j in jobs if j.state.group is StateGroup.ACTIVE]
    n_gpus = sum(j.alloc.gpus for j in running)
    types: Counter[str] = Counter()
    for j in running:
        if j.alloc.gpus:
            types[humanize.gpu_label(j.alloc.gpu_type)] += j.alloc.gpus
    gl = Text(str(n_gpus), style="bold" if n_gpus else theme.dim())
    if types:
        gl.append("   " + " · ".join(f"{t} ×{n}" for t, n in types.most_common()), style=theme.muted())
    lines.append(render.label_value("my GPUs", gl, width=LABEL_W))

    gpu_tres = snap.caps.gpu_tres if snap.caps is not None else None
    usages = [j.gpu for j in running if j.gpu is not None and j.gpu.util_mean_pct is not None]
    if gpu_tres is False:
        util = Text("n/a — no GPU accounting on this cluster", style=theme.dim())
    elif not usages:
        util = Text("— no running GPU jobs" if not n_gpus else "waiting for sstat", style=theme.dim())
    else:
        total_g = sum(u.n_gpus for u in usages) or 1
        mean = sum((u.util_mean_pct or 0) * u.n_gpus for u in usages) / total_g
        used = sum(u.mem_used_mib or 0 for u in usages)
        total = sum(u.mem_total_mib or 0 for u in usages)
        mem = f"   mem {humanize.gib_pair(used, total)}" if total else ""
        bar_w = max(8, width - LABEL_W - 6 - len(mem))
        util = render.gpu_bar(mean, bar_w, thresholds)
        util.append(" " + humanize.pct(mean, width=4))
        util.append(mem, style=theme.muted())
    lines.append(render.label_value("job util", util, width=LABEL_W))

    free = _free_lines(snap, width - LABEL_W)
    lines.append(render.label_value("free GPUs", free[0], width=LABEL_W))
    lines += [render.label_value("", more, width=LABEL_W) for more in free[1:]]

    errors = _error_lines(state, now, width)
    if errors:
        return [render.fit(x, width) for x in lines] + errors

    # jobs with a start estimate first (soonest first): "no estimate" must not rank as "starting now"
    pending = sorted((j for j in jobs if j.state.group is StateGroup.QUEUED),
                     key=lambda j: (j.start_time is None or not j.est_start,
                                    j.start_time.timestamp() if j.start_time is not None else 0.0))
    if pending:
        j = pending[0]
        nxt = render.state_badge(j.state, j.held)
        nxt.append(" " + j.name)
        reason = render.reason_text(j.reason)
        if j.reason and not render.known_reason(j.reason) and j.hints:
            reason = Text(j.hints[0], style=render.hint_style(j.hints[0]))
        if reason:
            nxt.append(" · ", style=theme.dim())
            nxt.append_text(reason)
        if j.start_time is not None and j.est_start:
            wait = (j.start_time - now).total_seconds()
            when = humanize.span(wait) if wait >= 0 else "overdue"        # a past estimate is not "starting now"
            nxt.append(f" · est. {humanize.local_hm(j.start_time)} ({when})", style=theme.muted())
        lines.append(render.label_value("next", nxt, width=LABEL_W))
    else:
        lines.append(render.label_value("next", Text("nothing queued", style=theme.dim()), width=LABEL_W))

    login = Text(snap.caps.login_host if snap.caps is not None else "?")
    m = snap.metrics
    note = busy_note(state.status)
    if note:                                   # the controller's trouble matters more than the login node's load
        login.append(f"  {note}", style=theme.warn())
    elif m is not None and m.load is not None:
        login.append(f"  load {m.load[0]:.1f}/{m.ncpu or '?'}", style=theme.muted())
    lines.append(render.label_value("login", login, width=LABEL_W))
    return [render.fit(x, width) for x in lines]


# ---------------------------------------------------------------- workstation
def gpu_line(gpu: Any, width: int, util_t: tuple[float, float], mem_t: tuple[float, float],
             name_w: int = 9) -> Text:
    """``0 A6000   ▕███▌  ▏ 70%  ▕██▍   ▏ 21.0/48G  78°C`` sized to ``width``."""
    name = humanize.gpu_model(gpu.name)
    mem = humanize.gib_pair(gpu.mem_used_mib, gpu.mem_total_mib)
    temp = f"{gpu.temp_c:.0f}°C" if gpu.temp_c is not None else humanize.NA
    fixed = 2 + name_w + 1 + 5 + 2 + 1 + 10 + 1 + 4
    bar_w = max(4, (width - fixed) // 2)
    out = Text(f"{gpu.index} ", style=theme.muted())
    out.append(humanize.truncate(name, name_w).ljust(name_w))
    out.append(" ")
    out.append_text(render.gpu_bar(gpu.util_pct, bar_w, util_t))
    out.append(humanize.pct(gpu.util_pct, width=5))
    out.append("  ")
    # capacity, like the text next to it and gpu_panel: nvidia-smi `utilization.memory` (mem_util_pct) is memory
    # controller busy time — 1 % next to "23.1/48.0G" read as an empty GPU (M3 F7)
    mem_pct = None
    if gpu.mem_used_mib is not None and gpu.mem_total_mib:
        mem_pct = gpu.mem_used_mib / gpu.mem_total_mib * 100
    out.append_text(render.gpu_bar(mem_pct, bar_w, mem_t))
    out.append(" " + mem.rjust(10), style=theme.muted())
    out.append(" ")
    out.append(temp.rjust(4), style=theme.temp_style(gpu.temp_c))
    return render.fit(out, width)


def host_line(snap: HostSnapshot, width: int, mem_t: tuple[float, float]) -> Text:
    m = snap.metrics
    if m is None:
        return Text("cpu / mem: waiting for procfs", style=theme.dim())
    mem_pct = None
    if m.mem_total and m.mem_available is not None:
        mem_pct = (m.mem_total - m.mem_available) / m.mem_total * 100
    out = Text("cpu ", style=theme.muted())
    out.append_text(render.gpu_bar(m.cpu_pct, 6, theme.DEFAULT_UTIL_THRESHOLDS))
    out.append(humanize.pct(m.cpu_pct, width=4))
    out.append("  mem ", style=theme.muted())
    out.append_text(render.gpu_bar(mem_pct, 8 if width < 62 else 12, mem_t))
    out.append(humanize.pct(mem_pct, width=4))
    out.append(f" of {humanize.size(m.mem_total)}", style=theme.muted())
    if m.load is not None:
        out.append(f"  load {m.load[0]:.1f}/{m.ncpu or '?'}", style=theme.muted())
    return render.fit(out, width)


def tasks_line(snap: HostSnapshot, width: int) -> Text:
    tasks = [j for j in snap.jobs.values() if j.kind is JobKind.TASK and j.state.group is StateGroup.ACTIVE]
    mine = [j for j in tasks if j.is_mine]
    others = [j for j in tasks if not j.is_mine]
    out = Text("tasks  ", style=theme.muted())
    out.append(f"mine {len(mine)}", style="bold" if mine else theme.dim())
    if mine:
        labels = ", ".join(j.name.split(" (")[0] for j in mine[:2])
        gpus = sum(j.alloc.gpus for j in mine)
        out.append(f" ({labels}{'…' if len(mine) > 2 else ''}, {gpus} GPU{'s' if gpus != 1 else ''})",
                   style=theme.muted())
    out.append(f" · others {len(others)}", style=theme.muted())
    return render.fit(out, width)


def old_reading(state: HostState, gpu: Any, now: datetime) -> bool:
    """A GPU reading that missed its refresh while the host itself answers (hung nvidia-smi, failing section): shown
    dim and never counted as free (same rule as the GPUs view's `~`)."""
    every = max(state.status.tick_interval or 0.0, state.config.workstation.poll_gpu)
    return not state.stale(now) and sample_age(gpu.sampled_at, now, every) is not None


def _ws_lines(state: HostState, snap: HostSnapshot, now: datetime, width: int,
              util_t: tuple[float, float], mem_t: tuple[float, float]) -> list[Text]:
    gpus = sorted(snap.gpus, key=lambda g: g.index)
    lines = [gpu_line(g, width, util_t, mem_t) for g in gpus[:MAX_GPU_LINES]]
    for g, line in zip(gpus, lines, strict=False):
        if old_reading(state, g, now):
            line.stylize(theme.dim())
    if len(gpus) > MAX_GPU_LINES:
        lines.append(Text(f"+{len(gpus) - MAX_GPU_LINES} more GPUs · ↵ host detail", style=theme.dim()))
    if not gpus:
        lines.append(Text("no GPUs reported (nvidia-smi missing or not sampled yet)", style=theme.dim()))
    errors = _error_lines(state, now, width)
    if not errors:
        lines.append(host_line(snap, width, mem_t))
    lines.append(tasks_line(snap, width))
    return lines + errors


def card_lines(state: HostState, now: datetime, *, width: int, window: float = 86400,
               util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS,
               mem_t: tuple[float, float] = theme.DEFAULT_MEM_THRESHOLDS) -> list[Text]:
    snap = state.snapshot
    if snap is None:
        if state.status.state is ConnState.DISABLED:
            return [Text("polling disabled", style=theme.dim())] + _error_lines(state, now, width)
        errors = _error_lines(state, now, width)
        if errors:                   # probed and failed: the error explains the empty card
            return [Text("no data yet", style=theme.dim())] + errors
        return [Text("waiting for the first probe…", style=theme.dim())]
    if is_cluster(state):
        return _slurm_lines(state, snap, now, width, window, util_t)
    return _ws_lines(state, snap, now, width, util_t, mem_t)


def _right(left: Text, right: Text, width: int) -> Text:
    gap = width - left.cell_len - right.cell_len
    if gap < 1:
        return render.fit(left, width)
    return Text.assemble(left, " " * gap, right)


class HostCard(Static, can_focus=True):
    """One host's card; focus = selection (arrows/hjkl move it, click selects, double-click opens; `space` or a click
    on the header chip scopes)."""

    DEFAULT_CSS = """
    HostCard { height: auto; }
    """

    def __init__(self, host: str, index: int) -> None:
        super().__init__("", id=f"card-{index}")
        self.host = host
        self.index = index
        self.lines: list[Text] = []

    def set_lines(self, title: Text, lines: list[Text], pad_to: int, *, stale: bool, problem: bool) -> None:
        """Replace the content; rows of a grid are padded to the same height so borders line up."""
        self.border_title = title
        self.update(Text("\n").join(lines + [Text("")] * max(0, pad_to - len(lines))))
        self.set_class(stale, "-stale")
        self.set_class(problem, "-problem")

    def on_click(self, event: events.Click) -> None:
        # select only: a click (or the first click of a double-click) must not silently scope the whole app
        self.focus()
        if event.chain >= 2:
            self.app.action_open_host(self.host)  # type: ignore[attr-defined]
