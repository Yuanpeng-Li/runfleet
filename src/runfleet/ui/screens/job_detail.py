"""[WP3] Job detail screen: metadata, allocation, GPUs, steps, logs, raw (mockups E, F).

The job itself is re-read from the store on every change (so state and elapsed stay live); the slower facts
(scontrol fields, steps, fresh log stats, task members) come from ``FleetAPI.job_detail`` in a worker and are
refreshed with ``r``. One followed log stream feeds both the Summary log pane (when the terminal is >= 110×30) and the
Logs tab; it starts once the log path is known and is reopened when it dies while the job may still be writing.

Attach (§7.16): while the GPU or Procs tab of a running Slurm job is visible (this screen on top, modals aside) the
screen holds ``FleetAPI.attach_start(key)``; hiding the tab, covering the screen or closing it calls ``attach_stop``.
``FleetChanged.live_keys`` / the 1 Hz clock re-read ``attach_state(key)``. On those tabs ``r`` also reads the node now
(``attach_now``; srun asks first in a three-choice dialog), ``a`` toggles auto-refresh, ``S`` reads with srun when ssh
landed in another job. Workstation tasks keep their own GPU / Procs views.
"""

from __future__ import annotations

import re
import textwrap
from collections import deque
from datetime import datetime
from typing import Any

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static, TabbedContent, TabPane

from runfleet.core.service import LogUpdate
from runfleet.errors import AttachBlocked, AttachConsentRequired, AttachError
from runfleet.models import (AttachMode, AttachState, ConnState, FairShare, Job, JobDetail, JobGpuUsage, JobKey,
                             JobKind, JobState, JobStep, LogRef, StateGroup, Visibility)
from runfleet.ui import keymap, render, theme
from runfleet.ui.messages import FleetChanged
from runfleet.ui.query import elapsed_s, ended_at, left_s
from runfleet.ui.screens import FleetScreen
from runfleet.ui.screens.confirm import AttachConfirmScreen, AttachConsent
from runfleet.ui.screens.found_logs import (file_label, find_problem, fresher_ref, own_paths, picker, quiet_hint,
                                            remember_pick)
from runfleet.ui.widgets.attach_panel import AttachPanel, live_procs, procs_note
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.job_table import name_badges, runway_level
from runfleet.ui.widgets.log_view import LogFollower, LogView
from runfleet.ui.widgets.proc_table import ProcTable
from runfleet.util import humanize
from runfleet.util.efficiency import CPU_LOW, GPU_LOW, MEM_LOW, TIME_LOW, Efficiency, efficiency
from runfleet.util.efficiency import rows as efficiency_rows
from runfleet.util.logscan import EXIT_SCAN_LINES, TRIAGE_LINES, Cause, cause_of, exit_cause

TABS = ("summary", "gpu", "logs", "steps", "procs", "raw")
TAB_TITLES = {"summary": "Summary", "gpu": "GPU", "logs": "Logs", "steps": "Steps", "procs": "Procs", "raw": "Raw"}
LIVE_TABS = frozenset({"gpu", "procs"})          # tabs whose visibility holds an Attach (§7.16)
SUMMARY_LOG_MIN = (110, 30)
DETAIL_MIN_GAP = 5.0                             # `r`: a detail probe younger than this is reused, not re-run
LW = 10                                          # label width inside the boxes


# ---------------------------------------------------------------- pure renderers
def _title_right(job: Job, now: datetime, short: bool) -> Text:
    right = render.state_word(job.state, job.raw_state)
    el = elapsed_s(job, now)
    if job.state.group is StateGroup.ACTIVE:
        if el is not None:
            right.append(f"  {humanize.clock(el)}")
        remaining = left_s(job, now) if job.time_limit is not None else None
        if job.time_limit is not None and not short:
            right.append(f" of {humanize.clock(job.time_limit)}", style=theme.muted())
        if remaining is not None:
            level = runway_level(remaining, job.time_limit.total_seconds() if job.time_limit is not None else None)
            right.append(f" ({humanize.left(remaining)} left)", style=theme.error() if level == "red" else
                         theme.warn() if level == "amber" else theme.muted())
    elif job.state.group is StateGroup.QUEUED:
        reason = render.reason_text(job.reason, short=short)
        if reason:
            right.append("  ")
            right.append_text(reason)
    else:
        bits = []
        if job.exit_code is not None:
            bits.append(f"exit {job.exit_code}:{job.exit_signal or 0}" if not short else f"exit {job.exit_code}")
        if el is not None and not short:
            bits.append(f"ran {humanize.clock(el)}")
        end = ended_at(job)
        if end is not None:
            bits.append(f"ended {humanize.ago(end, now)}")
        right.append("  " + " · ".join(bits), style=theme.muted())
    return right


def title_line(job: Job, now: datetime, width: int, *, watched: bool = False) -> Text:
    """`host:id  ◉ ↻1 name  STATE  times`; when narrow the times shorten before the name is cut (80 columns read
    `orb-swp--eval--plac…` with room to spare)."""
    left = Text(f"{job.key.host}:{job.key.label}", style="bold")
    left.append("  ")
    left.append_text(name_badges(job, watched))
    left.append(job.name)
    right = _title_right(job, now, short=False)
    if width - left.cell_len - right.cell_len - 2 < 2:
        right = _title_right(job, now, short=True)
    gap = width - left.cell_len - right.cell_len - 2
    if gap < 2:
        left = render.fit(left, max(10, width - right.cell_len - 4))
        gap = 2
    return Text.assemble(" ", left, " " * gap, right)


def _pair(l1: str, v1: str | Text, l2: str = "", v2: str | Text = "", width: int = 50) -> Text:
    out = render.label_value(l1, v1 if isinstance(v1, Text) else Text(v1), width=LW)
    if l2:
        pad = max(1, width // 2 + 4 - out.cell_len)
        out.append(" " * pad)
        out.append_text(render.label_value(l2, v2 if isinstance(v2, Text) else Text(v2), width=9))
    return out


def job_box_lines(job: Job, detail: JobDetail | None, now: datetime, width: int) -> list[Text]:
    lines: list[Text] = []
    if job.kind is JobKind.TASK:
        lines.append(_pair("user", job.user, "root pid", job.job_id_raw, width))
        lines.append(_pair("started", f"{humanize.local_dt(job.start_time)} ({humanize.ago(job.start_time, now)})"))
        lines.append(_pair("processes", str(len(job.pids)), "cpu", humanize.cpu_load(job.cpu_pct), width))
        lines.append(_pair("rss", humanize.size(job.rss_bytes)))
        if detail is not None and detail.env:
            env = " ".join(f"{k}={v}" for k, v in detail.env.items())
            lines.append(_pair("env", humanize.truncate(env, width - LW)))
    else:
        lines.append(_pair("user", job.user, "account", job.account or humanize.NONE, width))
        lines.append(_pair("partition", job.partition or humanize.NONE, "qos", job.qos or humanize.NONE, width))
        lines.append(_pair("submitted", f"{humanize.local_dt(job.submit_time)} ({humanize.ago(job.submit_time, now)})"))
        if job.start_time is not None:
            if job.est_start:
                lines.append(_pair("est start", f"{humanize.local_dt(job.start_time)} "
                                                f"(in {humanize.span((job.start_time - now).total_seconds())})"))
            else:
                waited = (job.start_time - job.submit_time).total_seconds() if job.submit_time else None
                extra = f", waited {humanize.span(waited)}" if waited is not None else ""
                lines.append(_pair("started", f"{humanize.local_dt(job.start_time)} "
                                              f"({humanize.span((now - job.start_time).total_seconds())} ago{extra})"))
        if job.end_time is not None:
            lines.append(_pair("ended", f"{humanize.local_dt(job.end_time)} ({humanize.ago(job.end_time, now)})"))
        elif job.start_time is not None and job.time_limit is not None and not job.est_start:
            lines.append(_pair("ends by", humanize.local_dt(job.start_time + job.time_limit)))
        requeue = (detail.fields.get("Requeue") if detail else None) or job.raw.get("Requeue")
        lines.append(_pair("restarts", str(job.restarts), "requeue",
                           {"1": "yes", "0": "no"}.get(requeue or "", requeue or humanize.NONE), width))
        if job.cpu_pct is not None or job.rss_bytes is not None:      # sstat, running jobs (§5.1)
            cpu = f"{humanize.cpu_load(job.cpu_pct)} avg" if job.cpu_pct is not None else humanize.NA
            lines.append(_pair("cpu", cpu, "peak rss", humanize.size(job.rss_bytes), width))
    lines.append(_pair("workdir", humanize.truncate_left(job.workdir or humanize.NONE, width - LW)))
    lines.append(_pair("command", humanize.truncate_left(job.command or humanize.NONE, width - LW)))
    return [render.fit(x, width) for x in lines]


def pick_usage(job: Job, detail: JobDetail | None) -> JobGpuUsage | None:
    """The newer GPU sample of the detail probe and the polled job (the detail is fetched once, the store keeps
    polling); without timestamps the polled one wins."""
    a = detail.gpu if detail is not None else None
    b = job.gpu
    if a is None or b is None:
        return b or a
    if a.sampled_at is not None and (b.sampled_at is None or a.sampled_at > b.sampled_at):
        return a
    return b


def compact_lines(job: Job, detail: JobDetail | None, caps_gpu_tres: bool | None, now: datetime,
                  width: int) -> list[Text]:
    """Summary below 110×30: three lines (who/when, where, how it is doing) so the log fits under them; the full
    Job / Resources / GPU boxes stay on the GPU, Steps and Raw tabs."""
    who = Text(job.user, style="bold")
    if job.kind is JobKind.SLURM:
        who.append(f" · {job.partition or humanize.NONE}/{job.qos or humanize.NONE}", style=theme.muted())
    if job.state.group is StateGroup.QUEUED:
        if job.start_time is not None and job.est_start:
            who.append(f" · est start {humanize.local_hm(job.start_time)}", style=theme.muted())
        elif job.submit_time is not None:
            who.append(f" · submitted {humanize.ago(job.submit_time, now)}", style=theme.muted())
    elif job.start_time is not None:
        who.append(f" · started {humanize.ago(job.start_time, now)}", style=theme.muted())
        if job.end_time is None and job.time_limit is not None and job.kind is JobKind.SLURM:
            who.append(f" · ends by {humanize.local_hm(job.start_time + job.time_limit)}", style=theme.muted())
    if job.workdir:
        who.append(f" · {job.workdir}", style=theme.dim())
    a = job.alloc
    per_node = detail.per_node if detail is not None and detail.per_node else a.per_node
    where = Text(job.nodelist or ",".join(job.nodes) or ("—" if job.kind is JobKind.SLURM else job.key.host))
    if a.cpus:
        where.append(f" · {a.cpus} cpu", style=theme.muted())
    if a.mem_bytes:
        where.append(f" · {humanize.size(a.mem_bytes)}", style=theme.muted())
    if a.gpus:
        where.append(f" · {a.gpus}× {humanize.short_gpu_name(a.gpu_type) or 'GPU'}")
        idx = [f"{n.node} IDX {','.join(map(str, n.gpu_indices))}" if len(per_node) > 1 else
               f"IDX {','.join(map(str, n.gpu_indices))}" for n in per_node if n.gpu_indices]
        if idx:
            where.append(f" ({'; '.join(idx)})", style=theme.muted())
    usage = pick_usage(job, detail)
    how = Text()
    if usage is not None and usage.util_mean_pct is not None:
        how.append(f"GPU {humanize.pct(usage.util_mean_pct)}")
        if usage.mem_used_mib is not None:
            how.append(f" · mem {humanize.gib_pair(usage.mem_used_mib, usage.mem_total_mib)}", style=theme.muted())
    elif a.gpus and job.state.group is StateGroup.ACTIVE:
        why = "no GPU accounting" if caps_gpu_tres is False else "not sampled yet"
        how.append(f"GPU n/a — {why}", style=theme.dim())
    if job.cpu_pct is not None:
        how.append((" · " if how else "") + f"cpu {humanize.cpu_load(job.cpu_pct)}", style=theme.muted())
    for hint in job.hints[:2]:
        how.append(" · " if how else "")
        how.append(hint, style=render.hint_style(hint))
    lines = [who, where] + ([how] if how else [])
    return [render.fit(x, width) for x in lines]


def resources_lines(job: Job, detail: JobDetail | None, width: int) -> list[Text]:
    a = job.alloc
    per_node = detail.per_node if detail is not None and detail.per_node else a.per_node
    lines: list[Text] = []
    nodes = job.nodelist or ",".join(job.nodes) or humanize.NONE
    cpus = f"{a.cpus}" if a.cpus else humanize.NA
    cpu_ids = per_node[0].cpu_ids if per_node and per_node[0].cpu_ids else None
    if cpu_ids:
        cpus += f" ({cpu_ids})"
    out = render.label_value("nodes", Text(f"{a.nodes or len(job.nodes) or humanize.NA}  {nodes}"), width=8)
    out.append(f"   cpus {cpus}   mem {humanize.size(a.mem_bytes)}", style=theme.muted())
    lines.append(out)
    if a.gpus:
        g = Text(f"{a.gpus}× {a.gpu_type or 'GPU'}")
        for n in per_node:
            if n.gpu_indices:
                g.append(f"   node {n.node}  IDX {','.join(map(str, n.gpu_indices))}", style=theme.muted())
        lines.append(render.label_value("gpus", g, width=8))
    else:
        lines.append(render.label_value("gpus", Text(humanize.NONE, style=theme.dim()), width=8))
    if a.tres:
        tres = ",".join(f"{k}={v}" for k, v in a.tres.items())
        lines.append(render.label_value("tres", Text(humanize.truncate(tres, width - 8), style=theme.muted()),
                                        width=8))
    return [render.fit(x, width) for x in lines]


def gpu_box_title(job: Job, detail: JobDetail | None = None, now: datetime | None = None) -> str:
    usage = pick_usage(job, detail)
    if usage is None:
        return "GPU"
    src = {"sstat": "sstat (Slurm accounting, every 60 s)", "nvidia-smi": "nvidia-smi (live)",
           "demo": "demo"}.get(usage.source, usage.source)
    # the sample age first: it is what tells a stale reading apart, and a narrow box cuts the title's end
    age = f" · {humanize.ago(usage.sampled_at, now)}" if now is not None and usage.sampled_at is not None else ""
    return f"GPU{age} · source: {src}"


def gpu_lines(job: Job, detail: JobDetail | None, caps_gpu_tres: bool | None, width: int,
              util_t: tuple[float, float], mem_t: tuple[float, float]) -> list[Text]:
    usage = pick_usage(job, detail)
    lines: list[Text] = []
    bar = max(8, min(20, width - 40))
    if job.state.group is StateGroup.QUEUED:
        return [Text("n/a — job pending", style=theme.dim())]
    if usage is None:
        why = "no GPU accounting on this cluster" if caps_gpu_tres is False else \
            "job ended" if job.state.is_terminal else "no GPUs" if not job.alloc.gpus else "not sampled yet"
        return [Text(f"n/a — {why}", style=theme.dim())]
    util = render.gpu_bar(usage.util_mean_pct, bar + 2, util_t)
    util.append(humanize.pct(usage.util_mean_pct, width=5))
    if usage.util_peak_pct is not None:
        util.append(f"  (peak {humanize.pct(usage.util_peak_pct)})", style=theme.muted())
    lines.append(render.label_value("util", util, width=6))
    mem_pct = usage.mem_used_mib / usage.mem_total_mib * 100 if usage.mem_used_mib is not None and \
        usage.mem_total_mib else None
    mem = render.gpu_bar(mem_pct, bar + 2, mem_t)
    mem.append(" " + humanize.gib_pair(usage.mem_used_mib, usage.mem_total_mib, sep=" / "))
    lines.append(render.label_value("mem", mem, width=6))
    batch = _cpu_step(detail)
    if (batch is None or (batch.ave_cpu is None and batch.max_rss_bytes is None)) and job.cpu_pct is not None:
        cpu = Text(f"{humanize.cpu_load(job.cpu_pct)} (avg)" if job.cpu_pct >= 999.5
                   else f"{humanize.pct(job.cpu_pct)} of one core (avg)")
        if job.rss_bytes is not None:
            cpu.append(f"   peak rss {humanize.size(job.rss_bytes)} / {humanize.size(job.alloc.mem_bytes)}",
                       style=theme.muted())
        lines.append(render.label_value("cpu", cpu, width=6))
    elif batch is not None and (batch.ave_cpu is not None or batch.max_rss_bytes is not None):
        cpu = Text()
        if batch.ave_cpu is not None and job.alloc.cpus:
            el = (batch.elapsed or job.elapsed)
            tasks = max(1, batch.ntasks or 1)
            cores = batch.ave_cpu.total_seconds() * tasks / el.total_seconds() if el and el.total_seconds() else None
            cpu.append(f"AveCPU {humanize.clock(batch.ave_cpu)}")
            if tasks > 1 or not batch.step_id.endswith(".batch"):
                cpu.append(f" × {tasks} task{'s' if tasks != 1 else ''} (step {batch.step_id.partition('.')[2]})",
                           style=theme.muted())
            if cores is not None:
                cpu.append(f" (≈ {cores:.2f} of {job.alloc.cpus} cores)", style=theme.muted())
        if batch.max_rss_bytes is not None:
            cpu.append(f"   rss {humanize.size(batch.max_rss_bytes)} / {humanize.size(job.alloc.mem_bytes)}",
                       style=theme.muted())
        lines.append(render.label_value("cpu", cpu, width=6))
    hints = Text()
    for i, hint in enumerate(job.hints):
        hints.append((" · " if i else "") + hint, style=render.hint_style(hint))
    if not job.hints:
        hints.append("none", style=theme.dim())
    lines.append(render.label_value("hints", hints, width=6))
    if usage.note:
        lines.append(Text(usage.note, style=theme.dim()))
    return [render.fit(x, width) for x in lines]


def priority_box_title(job: Job, now: datetime) -> str:
    p = job.priority
    if p is None or p.sampled_at is None:
        return "Priority"
    return f"Priority · sprio {humanize.ago(p.sampled_at, now)}"


def _wrapped(text: str, width: int, *, first: str = "• ", style: str = "") -> list[Text]:
    rows = textwrap.wrap(text, max(10, width - len(first)), break_on_hyphens=False) or [""]
    return [Text((first if i == 0 else " " * len(first)) + row, style=style) for i, row in enumerate(rows)]


def _share_row(share: FairShare, account: str | None, room: int) -> Text:
    """`lab  0.03 · usage 83.3% of share 100.0%` — shorter forms when the box is narrow."""
    fs = f"{share.fairshare:.2f}" if share.fairshare is not None else humanize.NA
    usage = share.effective_usage is not None and share.norm_shares is not None
    for long in (True, False):
        row = Text(share.account, style="bold" if share.account == account else "")
        row.append(f"  {fs}", style=render.fairshare_style(share.fairshare))
        if usage and long:
            row.append(f" · usage {share.effective_usage:.1%} of share {share.norm_shares:.1%}", style=theme.muted())
        elif usage:
            row.append(f" · {share.effective_usage:.0%} of {share.norm_shares:.0%}", style=theme.muted())
        if not share.raw_usage:
            row.append(" · no recent usage" if long else " · no usage", style=theme.dim())
        if row.cell_len <= room:
            break
    return row


def priority_box_lines(job: Job, width: int) -> list[Text]:
    """The Summary's Priority box of an own pending Slurm job (in place of its "n/a — job pending" GPU box, §5.2.10):
    the job's priority and each factor as the cluster weights it (bar = share of that factor's weight), the user's
    fair share per account, then the hints. Evidence only: no start time, no queue position."""
    p = job.priority
    if p is None:
        return [Text("n/a — job pending", style=theme.dim())]
    lines: list[Text] = []
    if not p.ranked:
        for hint in p.hints:
            lines += _wrapped(hint, width, style=theme.muted())
        return [render.fit(x, width) for x in lines]
    if p.priority is None:
        lines.append(render.label_value("priority", Text("not read yet — sprio runs at most every 10 min while a "
                                                         "job of yours waits", style=theme.dim()), width=LW))
    else:
        head = Text(render.prio_num(p.priority), style="bold")
        if p.partition:
            head.append(f" in {p.partition}", style=theme.muted())
        for part, value in p.others[:3]:
            head.append(f" · {part} {render.prio_num(value)}", style=theme.dim())
        lines.append(render.label_value("priority", head, width=LW))
        nums = max((len(render.prio_num(f.value)) for f in p.factors), default=1)
        tail = max((len(render.prio_num(f.weight)) for f in p.factors if f.weight), default=0)
        marker = "  ◀ dominates" if width - LW - nums - tail - 18 >= 14 else " ◀"
        bar = max(4, min(14, width - LW - nums - tail - 4 - len(marker)))
        for f in p.factors:
            row = Text()
            if f.weight:
                row.append_text(render.share_bar(f.value / f.weight, bar))
                row.append(f" {render.prio_num(f.value):>{nums}} / {render.prio_num(f.weight)}")
            else:
                row.append(" " * bar)
                row.append(f" {render.prio_num(f.value):>{nums}}")
            if f.name == p.dominant:
                row.append(marker, style=theme.accent())
            lines.append(render.label_value(render.PRIORITY_LABELS.get(f.name, f.name), row, width=LW))
    for share in p.fairshare:
        lines.append(render.label_value("share" if share is p.fairshare[0] else "",
                                        _share_row(share, job.account, width - LW), width=LW))
    for hint in p.hints:
        lines += _wrapped(hint, width, style=theme.muted())
    return [render.fit(x, width) for x in lines]


_EXIT_HINT = re.compile(r"^exit \d+")


def why_bullets(job: Job, detail: JobDetail, log_cause: str | None) -> list[Text]:
    """"Why it ended" / "Notes": the log's failure line first (Traceback / OOM / Killed in the log tail of an ended,
    not successful job), the scheduler's explanation, hints it does not already say (no second "exit 1"), then the
    log files — a merged stdout/stderr file once."""
    bullets: list[Text] = []
    if log_cause and job.state.is_terminal and job.state.group is not StateGroup.DONE:
        bullets.append(Text("• log: " + log_cause, style=theme.error()))
    bullets += [Text("• " + line) for line in detail.explanation]
    said = " ".join(detail.explanation).lower()
    for h in job.hints:
        if h.lower() in said or (_EXIT_HINT.match(h) and ("exit" in said or "code" in said)):
            continue
        if log_cause and h.lower().startswith("log: ") and h[5:].strip().lower() in log_cause.lower():
            continue
        bullets.append(Text("• " + h, style=render.hint_style(h)))
    shown: set[str] = set()
    for ref in detail.logs:
        if not ref.path or ref.path in shown:
            continue
        shown.add(ref.path)
        both = ref.merged or sum(1 for r in detail.logs if r.path == ref.path) > 1
        stream = "stdout+stderr" if both else ref.stream
        info = f" · {humanize.size(ref.size)}" if ref.size is not None else ""
        bullets.append(Text(f"• {stream}: {ref.path}{info}" + (" (merged)" if both else ""), style=theme.muted()))
        if ref.error:
            bullets.append(Text(f"  {ref.error}", style=theme.warn()))
    return bullets


def _cpu_step(detail: JobDetail | None) -> JobStep | None:
    """The step whose AveCPU describes the job's CPU use. In srun/DDP jobs the batch step is only the launcher, so
    the busiest non-batch, non-extern step (AveCPU × NTasks) wins when one has numbers; else the batch step."""
    if detail is None:
        return None
    work = [s for s in detail.steps if s.ave_cpu is not None and not s.step_id.endswith((".batch", ".extern"))]
    if work:
        return max(work, key=lambda s: s.ave_cpu.total_seconds() * max(1, s.ntasks or 1) if s.ave_cpu else 0.0)
    for s in detail.steps:
        if s.step_id.endswith(".batch"):
            return s
    return detail.steps[0] if detail.steps else None


STEP_COLUMNS: tuple[tuple[str, int], ...] = (("STEP", 19), ("NAME", 14), ("STATE", 15), ("EXIT", 7),
                                            ("ELAPSED", 10), ("NODE", 11), ("NTASKS", 8), ("MAXRSS", 9),
                                            ("GPU (max per task)", 19))


def step_row(step: JobStep) -> tuple[Text, ...]:
    """One Steps-tab row; every cell keeps a one-cell gap (cell_padding 0), long values end in an ellipsis."""
    exit_text = f"{step.exit_code}:{step.exit_signal or 0}" if step.exit_code is not None else humanize.NONE
    gpu = step.tres_usage_max.get("gres/gpuutil")
    cells = (
        Text(step.step_id), Text(step.name),
        Text(step.raw_state.split(" ")[0] or step.state.name, style=theme.state_style(step.state)),
        Text(exit_text), Text(humanize.clock(step.elapsed)), Text(step.nodelist or humanize.NONE),
        Text(str(step.ntasks) if step.ntasks is not None else humanize.NONE),
        Text(humanize.size(step.max_rss_bytes, dash=humanize.NONE)),
        Text(f"{gpu}%" if gpu else humanize.NONE, style=theme.muted()),
    )
    out = []
    for cell, (_label, width) in zip(cells, STEP_COLUMNS, strict=True):
        if cell.cell_len > width - 1:
            cell = cell.copy()
            cell.truncate(width - 1, overflow="ellipsis")
        out.append(cell)
    return tuple(out)


_TIME_KEYS = frozenset({"Submit", "Start", "End", "Eligible"})


def raw_text(job: Job, detail: JobDetail | None, now: datetime) -> Text:
    """Raw tab: scontrol fields when known, plus squeue/sacct fields they do not already have (case-insensitively:
    no `JobID` next to `JobId`). Epoch times (SLURM_TIME_FORMAT=%s) are shown as local time + age, raw value dim."""
    fields: dict[str, str] = dict(detail.fields) if detail is not None else {}
    seen = {k.lower() for k in fields}
    for k, v in job.raw.items():
        if k.lower() not in seen:
            fields[k] = v
            seen.add(k.lower())
    out = Text()
    width = max((len(k) for k in fields), default=10)
    for k in sorted(fields):
        value = str(fields[k])
        out.append(f"{k:<{width}}  ", style=theme.label())
        if (k.endswith("Time") or k in _TIME_KEYS) and value.isdigit() and len(value) >= 9:
            when = datetime.fromtimestamp(int(value), tz=now.tzinfo)
            out.append(f"{humanize.local_dt(when)} ({humanize.ago(when, now)})")
            out.append(f"  {value}\n", style=theme.dim())
        else:
            out.append(f"{value}\n")
    if not fields:
        out.append("no raw fields", style=theme.dim())
    return out


def stale_banner(state: Any, now: datetime, *, context: str | None = None) -> Text | None:
    """Why the data may be old. ``context``: the keymap context showing it, when ``c`` may not be bound there."""
    if state is None or not state.stale(now):
        return None
    st = state.status
    if st.last_ok is None and state.snapshot is None:     # never connected: no "data from never"
        what = "needs a login" if st.state is ConnState.NEEDS_AUTH else f"is {theme.CONN_LABEL[st.state]}"
        out = Text(f"no data yet — {st.name} {what}", style=theme.warn())
    else:
        age = humanize.ago(st.last_ok, now, dash="never") if st.last_ok else "never"
        out = Text(f"data from {age} — {st.name} is {theme.CONN_LABEL[st.state]}", style=theme.warn())
    if st.error is not None:
        out.append(f" ({st.error.summary})", style=theme.muted())
    if st.state in (ConnState.RETRYING, ConnState.UNREACHABLE) and st.next_poll_at is not None:
        out.append(f", next retry in {humanize.age(max(0.0, (st.next_poll_at - now).total_seconds()))}",
                   style=theme.muted())
    if st.state is ConnState.NEEDS_AUTH:
        here = context is None or keymap.find(context, "app.connect_host") is not None
        out.append(" — press c to log in" if here else " — log in from Fleet (c)", style=theme.muted())
    return out


def detail_problem(detail: JobDetail) -> str | None:
    """Banner text for a detail probe that came back incomplete: failed sections must not look like missing data."""
    if not detail.errors:
        return None
    return "detail incomplete: " + "; ".join(e.summary for e in detail.errors) + " — r retries"


# ---------------------------------------------------------------- triage (§5.2.8)
def cause_line(job: Job, cause: Cause | None, width: int) -> Text | None:
    """The line under the title: an ended job's cause chip with the first log line that shows it (else its one-line
    explanation) and the key that opens the log there; a pending job's reason chip with what to do about it and the
    start estimate's caveat. None = nothing to say (running, completed)."""
    if job.state.group is StateGroup.QUEUED:
        reason = render.reason_text(job.reason, short=True).plain
        advice = job.raw.get("reason_advice")
        if not reason and not advice:
            return None
        out = render.cause_chip(reason or "pending", pending=True)
        if advice:
            out.append(" " + advice, style=theme.muted())
        if job.est_start and job.start_time is not None:
            out.append(f" · est start {humanize.local_hm(job.start_time)} (Slurm's guess; it moves)", style=theme.dim())
        return render.fit(out, width)
    if cause is None:
        return None
    out = render.cause_chip(cause.code)
    out.append(" ")
    key = Text("  L: log at this line", style=theme.dim()) if cause.evidence and cause.pattern is not None else Text()
    body = Text(cause.evidence, style=theme.error()) if cause.evidence else Text(cause.text, style=theme.muted())
    room = width - out.cell_len - key.cell_len
    if room < 12:
        key, room = Text(), width - out.cell_len
    if body.cell_len > room:
        body.truncate(max(0, room), overflow="ellipsis")
    out.append_text(body)
    out.append(" " * max(0, width - out.cell_len - key.cell_len))
    out.append_text(key)
    return render.fit(out, width)


_EFF_STYLE = {"ok": theme.ok, "warn": theme.warn, "bad": theme.error, "none": lambda: ""}


def efficiency_box_lines(eff: Efficiency | None, width: int) -> list[Text]:
    """The Summary's Efficiency box of an ended Slurm job (in place of its GPU box): reportseff colours — an amber
    value is an oversized request — and the "next time" request."""
    if eff is None:
        return [Text("n/a — the job did not run", style=theme.dim())]
    out = []
    for label, text, level in efficiency_rows(eff):
        style = theme.accent() if label == "next" and text != humanize.NA else _EFF_STYLE[level]()
        out.append(render.fit(render.label_value(label, Text(text, style=style), width=6), width))
    return out


def efficiency_summary(eff: Efficiency | None) -> Text | None:
    """One line for the compact Summary: ``eff  time 22% · cpu 61% · mem 38% · gpu 92% · next --time=… --mem=…``."""
    if eff is None or eff.live:
        return None
    out = Text("eff ", style=theme.label())
    parts: list[tuple[str, float | None, float]] = [("time", eff.time_pct, TIME_LOW if eff.completed else 0.0),
                                                   ("cpu", eff.cpu_pct, CPU_LOW), ("mem", eff.mem_pct, MEM_LOW)]
    if eff.gpus:
        parts.append(("gpu", eff.gpu_pct, GPU_LOW))
    for i, (label, value, low) in enumerate(parts):
        out.append(" · " if i else " ", style=theme.dim())
        out.append(f"{label} ", style=theme.muted())
        out.append(humanize.pct(value), style=theme.warn() if value is not None and value < low else "")
    nxt = " ".join(x for x in (f"--time={eff.next_time}" if eff.next_time else "",
                               f"--mem={eff.next_mem}" if eff.next_mem else "") if x)
    if nxt:
        out.append(" · next ", style=theme.dim())
        out.append(nxt, style=theme.accent())
    return out


def headroom_line(eff: Efficiency | None, width: int) -> Text | None:
    """A running job's memory headroom (sstat MaxRSS against the requested memory) for the GPU box."""
    if eff is None or not eff.live or eff.mem_peak is None:
        return None
    row = next((r for r in efficiency_rows(eff) if r[0] == "mem"), None)
    if row is None:
        return None
    return render.fit(render.label_value("room", Text(row[1], style=_EFF_STYLE[row[2]]()), width=6), width)


# ---------------------------------------------------------------- screen
class JobDetailScreen(FleetScreen):
    BINDINGS = keymap.bindings("job_detail")
    HELP_CONTEXTS = ("job_detail", "global")
    HELP_TITLE = "Job detail"

    def __init__(self, key: JobKey) -> None:
        super().__init__()
        self.key = key
        self.job: Job | None = None
        self.detail: JobDetail | None = None
        self.detail_error: str | None = None
        self.stream = "stdout"
        self.follower: LogFollower | None = None
        self.summary_log = False
        self._tail: deque[str] = deque(maxlen=EXIT_SCAN_LINES)    # last log lines, for "Why it ended"
        self.log_cause: str | None = None
        self._triage: deque[str] = deque(maxlen=TRIAGE_LINES)      # log lines already streamed, for the cause chip
        self.cause: Cause | None = None
        self._steps_shown: tuple[tuple[Text, ...], ...] = ()     # Steps rows on screen (rebuilt only on change)
        self._step_keys: list[str] = []
        self._reload_pending = False
        self._attached = False                                   # attach_start called, attach_stop not yet

    def forget_rendered(self) -> None:
        """The palette changed: rebuild the Steps rows (see ClickTable.forget_rendered)."""
        self._steps_shown = ()

    def compose(self) -> ComposeResult:
        job = self.rf.store.job(self.key)
        self.job = job
        label = job.key.label if job is not None else self.key.id
        yield FleetHeader(crumbs=[("Jobs", ("view", "jobs")), label], host=self.key.host, id="header")
        yield Static("", id="title", classes="title-line")
        yield Static("", id="banner", classes="title-line")
        cause = Static("", id="cause", classes="title-line")      # cause chip / pending reason (§5.2.8)
        cause.display = False
        yield cause
        self.summary_log = self._wants_summary_log()
        with TabbedContent(id="tabs", initial="summary"):
            with TabPane("Summary", id="summary"):
                # the log pane (1fr) needs a fixed-height parent; without it the boxes may scroll instead. Both
                # layouts are composed; a resize across 110×30 switches them (the follower keeps its views):
                # small terminal = 3 summary lines, then the log (critic, 80x24); large = the boxes, then the log
                with Vertical():
                    yield Static("", id="summary-compact", classes="box")
                    with Horizontal(id="summary-top"):
                        yield Static("", id="job-box", classes="box")
                        with Vertical(id="right-col"):
                            yield Static("", id="res-box", classes="box")
                            yield Static("", id="gpu-box", classes="box")
                    yield LogView(id="summary-log", classes="box")
            with TabPane("GPU", id="gpu"):
                with VerticalScroll():
                    yield AttachPanel("", id="attach-box", classes="box")
                    yield Static("", id="gpu-tab", classes="box")
            with TabPane("Logs", id="logs"):
                with Vertical():
                    hint = Static("", id="logs-hint", classes="note")     # "stdout quiet for 13h; f finds …"
                    hint.display = False
                    yield hint
                    yield LogView(id="logs-log", classes="box")
            with TabPane("Steps", id="steps"):
                with Vertical():
                    yield DataTable(id="steps-table", cursor_type="row", show_row_labels=False, cell_padding=0,
                                    cursor_foreground_priority="renderable")
                    yield Static("", id="why", classes="box")
            with TabPane("Procs", id="procs"):
                with Vertical():
                    yield Static("", id="procs-note", classes="note")
                    yield ProcTable(id="procs-table")
            with TabPane("Raw", id="raw"):
                with VerticalScroll():
                    yield Static("", id="raw-text")
        yield HintBar(self.help_contexts(), id="hints")

    def _wants_summary_log(self) -> bool:
        w, h = self.app.size
        return w >= SUMMARY_LOG_MIN[0] and h >= SUMMARY_LOG_MIN[1]

    def _apply_summary_layout(self) -> None:
        self.set_class(not self.summary_log, "-compact")
        self.query_one("#summary-compact", Static).display = not self.summary_log

    def on_resize(self) -> None:
        super().on_resize()
        narrow = self.app.size.width < 100
        if narrow != self.has_class("-narrow"):
            self.set_class(narrow, "-narrow")
        if self._wants_summary_log() != self.summary_log:
            self.summary_log = not self.summary_log
            self._apply_summary_layout()
        self._render_all()

    def on_mount(self) -> None:
        self.set_class(self.app.size.width < 100, "-narrow")
        self._apply_summary_layout()
        table = self.query_one("#steps-table", DataTable)
        for label, w in STEP_COLUMNS:
            table.add_column(Text(label, style="bold"), width=w)
        views = [v for v in self.query(LogView)]
        self.follower = LogFollower(self, views, on_update=self._on_log_update)
        self._start_log()
        self.refresh_data(None)
        self.load_detail()

    def on_screen_resume(self) -> None:
        super().on_screen_resume()
        # the full log viewer may have taken this screen's stream slot (one stream per host on some hosts)
        job, follower = self.job, self.follower
        if follower is not None and follower.dead and job is not None and not job.state.is_terminal:
            self._start_log()
        self._sync_attach()

    def on_screen_suspend(self) -> None:
        self._sync_attach()            # a full screen on top (log viewer) ends the attach; a dialog does not

    def on_unmount(self) -> None:
        self._attach_stop()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._sync_attach()
        self._render_live()

    # ------------------------------------------------------------------ attach (§7.16)
    def _live_tab(self) -> bool:
        if not self.is_mounted:
            return False
        return self.query_one(TabbedContent).active in LIVE_TABS

    def _live_job(self) -> bool:
        job = self.job
        return job is not None and job.kind is JobKind.SLURM and job.state is JobState.RUNNING

    def _on_top(self) -> bool:
        """This screen is the active one, or only dialogs cover it (the srun confirmation, help)."""
        for screen in reversed(self.app.screen_stack):
            if screen is self:
                return True
            if not isinstance(screen, ModalScreen):
                return False
        return False

    def _sync_attach(self) -> None:
        """Start the attach when a live tab of a running Slurm job becomes visible; stop it when the tab or the screen
        goes away. A job that stops running keeps its attach until then (the core says why it stopped)."""
        if not self.is_mounted:
            return
        visible = self._live_tab() and self._on_top()
        if visible and not self._attached and self._live_job():
            self._attached = True
            try:
                self.rf.service.attach_start(self.key)
            except AttachError as exc:         # the state says why; a raising service must not break the screen
                self.rf.toast(f"live readings: {exc}", severity="warning")
        elif not visible and self._attached:
            self._attach_stop()
        self.query_one(HintBar).set_contexts(self.help_contexts())

    def _attach_stop(self) -> None:
        if self._attached:
            self._attached = False
            try:
                self.rf.service.attach_stop(self.key)
            except Exception:                  # closing a screen never fails on the attach
                pass

    def _attach_state(self) -> AttachState | None:
        try:
            return self.rf.service.attach_state(self.key)
        except Exception:
            return None

    def _render_live(self) -> None:
        """Re-render only what Attach drives (status countdown, live rows, the Procs tab)."""
        job = self.job
        if job is None or job.kind is not JobKind.SLURM or not self.is_mounted:
            return
        width = (self.size.width or self.app.size.width) - 6
        self._render_attach_box(job, self.detail, width)
        self._render_procs(job, self.detail)

    def _render_attach_box(self, job: Job, detail: JobDetail | None, width: int) -> None:
        ui = self.rf.service.config.ui
        box = self.query_one("#attach-box", AttachPanel)
        box.display = job.kind is JobKind.SLURM
        if box.display:
            state = self.rf.store.hosts.get(self.key.host)
            caps = state.snapshot.caps if state is not None and state.snapshot is not None else None
            box.show(self._attach_state(), job, pick_usage(job, detail), self.rf.now(), ui.gpu_util_thresholds,
                     ui.gpu_mem_thresholds, width=width, gpu_tres=caps.gpu_tres if caps else None)

    @work(exit_on_error=False, group="attach")
    async def read_live(self, mode: AttachMode | None = None, consent: AttachConsent | None = None) -> None:
        """`r` / `S` on a live tab: one sample now. srun without consent asks (Once / Allow for this job / Cancel);
        a blocked attach says why and how to enable it."""
        try:
            await self.rf.service.attach_now(self.key, mode=mode, consent=consent)
        except AttachConsentRequired as exc:
            state = self._attach_state()

            def chosen(choice: AttachConsent | None) -> None:
                if choice is not None and self.is_mounted:
                    self.read_live(mode, choice)

            self.app.push_screen(AttachConfirmScreen(self.key.host, exc.offer, state.srun_rows if state else 0),
                                 chosen)
        except AttachBlocked as exc:
            self.rf.toast(exc.reason + (f" — {exc.hint}" if exc.hint else ""), severity="warning",
                          timeout=8.0 if exc.hint else None)
        except Exception as exc:           # transport trouble is in the state; anything else is one toast
            self.rf.toast(f"live read failed: {exc}", severity="warning")
        if self.is_mounted:
            self._render_live()

    # ------------------------------------------------------------------ data
    @work(exit_on_error=False, group="detail", exclusive=True)
    async def load_detail(self, max_age_s: float = 30) -> None:
        try:
            self.detail = await self.rf.service.job_detail(self.key, max_age_s=max_age_s)
            self.detail_error = detail_problem(self.detail)
        except Exception as exc:   # detail problems are shown in the banner, never raised
            self.detail_error = f"detail unavailable: {exc}"
        self.refresh_data(None)
        if self._reload_pending:   # `r` while this probe ran: one more, served from it when it is recent enough
            self._reload_pending = False
            self.call_later(self.load_detail, DETAIL_MIN_GAP)

    def _detail_busy(self) -> bool:
        return any(w.node is self and w.group == "detail" and not w.is_finished for w in self.workers)

    def _on_log_update(self, update: Any) -> None:
        """Keep the log tail; an ended job's Traceback / OOM / Killed line becomes the first "Why it ended" bullet."""
        if not update.lines:
            return
        self._tail.extend(update.lines)
        self._triage.extend(update.lines)
        cause = exit_cause(self._tail)
        if cause != self.log_cause:
            self.log_cause = cause
            if self.job is not None and self.is_mounted:
                self._render_steps(self.job, self.detail)
        if self.job is not None and self.job.state.is_terminal and self.is_mounted:
            self._render_cause()

    def shown_ref(self) -> LogRef | None:
        """The log this view shows (the pager and `l` open the same one)."""
        return self._wanted_ref()

    def _wanted_ref(self) -> LogRef | None:
        """The polled ref when its path is known (it follows requeues), else the detail's resolved one."""
        job = self.job
        if job is None:
            return None
        if self.stream == "file":                      # the file picked with `f`
            return self.rf.stream_ref(job, "file")
        ref = self.rf.log_ref(job, self.stream)
        if (ref is None or ref.path is None) and self.detail is not None:
            ref = next((r for r in self.detail.logs if r.stream == self.stream and r.path), None) or ref
        return ref

    def _start_log(self, *, retry: bool = False) -> None:
        job = self.job
        if job is None or self.follower is None:
            return
        self._tail.clear()
        self._triage.clear()
        self.log_cause = None
        ref = self._wanted_ref()
        if ref is None:            # never leave the previous stream writing under the new label
            self.follower.stop()
            for view in self.query(LogView):
                view.reset()
                view.apply(LogUpdate(note="no log path known"))
            self._update_log_titles()
            return
        self.follower.start(job.key.host, ref, retry=retry)
        self._update_log_titles()

    def _keep_following(self) -> None:
        """Start the stream once its path becomes known (or changes: requeue); reopen a dead stream with backoff
        while this screen is on top, the host answers and the job may still be writing."""
        job, follower = self.job, self.follower
        if job is None or follower is None:
            return
        ref = self._wanted_ref()
        if ref is not None and ref.path is not None and (follower.ref is None or follower.ref.path != ref.path):
            self._start_log()
            return
        state = self.rf.store.hosts.get(self.key.host)
        fresh = state is not None and not theme.is_problem(state.status.state) and not state.stale(self.rf.now())
        if self.app.screen is self and fresh and not job.state.is_terminal and follower.reconnect_due():
            self._start_log(retry=True)

    def refresh_data(self, changes: FleetChanged | None) -> None:
        if not self.is_mounted:
            return
        if changes is not None and self.key.host not in changes.hosts:
            return
        if changes is not None and self._live_only(changes):
            self._render_live()            # an Attach state change alone: no full re-render
            return
        job = self.rf.store.job(self.key)
        if job is not None:
            self.job = job
        self._sync_attach()                # the job may have started running under an open GPU / Procs tab
        if self.job is not None:
            for view in self.query(LogView):           # an ended job's log is static, not "following"
                if view.ended != self.job.state.is_terminal:
                    view.ended = self.job.state.is_terminal
                    view._render_live()
        self.query_one(FleetHeader).refresh_content()
        self._render_all()
        self._keep_following()

    def _live_only(self, changes: FleetChanged) -> bool:
        """The batch only carries Attach states (of this job or another one on the host)."""
        return all(c.live and not (c.jobs_added or c.jobs_changed or c.jobs_removed or c.status or c.paused or c.gpus
                                   or c.procs or c.metrics or c.nodes) for c in changes.changes if c.host == self.key.host)

    def refresh_time(self) -> None:
        self.query_one(FleetHeader).refresh_content()
        self._render_title()
        self._render_find_hint()
        if self._attached and self._live_tab():
            self._render_live()            # the next-sample countdown
        if self.follower is not None and self.follower.dead:
            self._keep_following()

    def _render_title(self) -> None:
        job = self.job
        width = (self.size.width or 120) - 2
        now = self.rf.now()
        if job is None:
            self.query_one("#title", Static).update(Text(f" {self.key} — no longer known", style=theme.warn()))
            return
        self.query_one("#title", Static).update(title_line(job, now, width, watched=self.rf.alerts.is_watched(job)))
        banner = stale_banner(self.rf.store.hosts.get(self.key.host), now, context="job_detail")
        if self.detail_error:
            banner = Text(self.detail_error, style=theme.warn())
        widget = self.query_one("#banner", Static)
        widget.display = banner is not None
        widget.update(banner or "")

    def _render_all(self) -> None:
        self._render_title()
        job = self.job
        if job is None:
            return
        now = self.rf.now()
        detail = self.detail
        state = self.rf.store.hosts.get(self.key.host)
        caps = state.snapshot.caps if state is not None and state.snapshot is not None else None
        ui = self.rf.service.config.ui
        total = self.size.width or self.app.size.width
        eff = efficiency(job, detail.steps if detail is not None else ())
        if not self.summary_log:
            compact = self.query_one("#summary-compact", Static)
            compact.border_title = "Task" if job.kind is JobKind.TASK else "Job"
            clines = compact_lines(job, detail, caps.gpu_tres if caps else None, now, total - 4)
            if job.state.is_terminal and detail is not None and (summary := efficiency_summary(eff)) is not None:
                clines.append(render.fit(summary, total - 4))
            if job.priority is not None:                     # pending: the priority evidence in one line (§5.2.10)
                clines.append(render.fit(render.priority_summary(job.priority, job.account, now), total - 4))
            compact.update(Text("\n").join(clines))
        if self.has_class("-narrow"):
            left_w = right_w = total - 4
        else:
            left_w = max(30, total // 2 - 4)
            right_w = max(30, total - total // 2 - 4)
        job_box = self.query_one("#job-box", Static)
        job_box.border_title = "Task" if job.kind is JobKind.TASK else "Job"
        job_box.update(Text("\n").join(job_box_lines(job, detail, now, left_w)))
        res = self.query_one("#res-box", Static)
        res.border_title = "Resources"
        res.update(Text("\n").join(resources_lines(job, detail, right_w)))
        gpu_box = self.query_one("#gpu-box", Static)
        if job.kind is JobKind.SLURM and job.state.is_terminal and detail is not None:
            gpu_box.border_title = "Efficiency"              # an ended job's GPU box only said "n/a — job ended"
            lines = efficiency_box_lines(eff, right_w)
        elif job.priority is not None:                       # a pending job's GPU box only said "n/a — job pending"
            gpu_box.border_title = priority_box_title(job, now)
            lines = priority_box_lines(job, right_w)
        else:
            gpu_box.border_title = gpu_box_title(job, detail, now)
            lines = gpu_lines(job, detail, caps.gpu_tres if caps else None, right_w, ui.gpu_util_thresholds,
                              ui.gpu_mem_thresholds)
            room = headroom_line(eff, right_w)
            if room is not None:
                lines.append(room)
        gpu_box.update(Text("\n").join(lines))
        self._render_cause()
        self._render_gpu_tab(job, detail, caps.gpu_tres if caps else None, total - 6)
        self._render_steps(job, detail)
        self._render_procs(job, detail)
        self._render_raw(job, detail)
        self._update_log_titles()

    def _render_cause(self) -> None:
        """The cause chip line (ended jobs: from the state, the exit status and the streamed log; pending jobs: the
        reason and what to do). Its tooltip carries the full explanation."""
        job = self.job
        widget = self.query_one("#cause", Static)
        self.cause = cause_of(job, self.detail, self._triage) if job is not None else None
        line = cause_line(job, self.cause, (self.size.width or 120) - 2) if job is not None else None
        widget.display = line is not None
        widget.update(line or "")
        widget.tooltip = self.cause.plain() if self.cause is not None else None

    def _render_gpu_tab(self, job: Job, detail: JobDetail | None, gpu_tres: bool | None, width: int) -> None:
        ui = self.rf.service.config.ui
        self._render_attach_box(job, detail, width)
        box = self.query_one("#gpu-tab", Static)
        box.border_title = gpu_box_title(job, detail, self.rf.now())
        lines = gpu_lines(job, detail, gpu_tres, width, ui.gpu_util_thresholds, ui.gpu_mem_thresholds)
        per_node = detail.per_node if detail is not None and detail.per_node else job.alloc.per_node
        lines.append(Text(""))
        lines.append(Text("allocation per node", style="bold"))
        for n in per_node:
            row = Text(f"  {n.node:<12}", style="")
            row.append(f"IDX {','.join(map(str, n.gpu_indices)) or '?':<8}")
            if n.gres:
                row.append(f"  {n.gres}", style=theme.muted())
            lines.append(row)
        if not per_node:
            why = ("unknown until scontrol reports the allocation" if job.kind is JobKind.SLURM
                   else "this task holds no GPU" if not job.alloc.gpus else "GPU indices not sampled yet")
            lines.append(Text(f"  {why}", style=theme.dim()))
        box.update(Text("\n").join(lines))

    def _render_steps(self, job: Job, detail: JobDetail | None) -> None:
        table = self.query_one("#steps-table", DataTable)
        why = self.query_one("#why", Static)
        why.border_title = "Why it ended" if job.state.is_terminal else "Notes"
        if detail is None:
            table.clear()
            self._steps_shown, self._step_keys = (), []
            table.display = False
            why.update(Text("loading steps …" if self.detail_error is None else self.detail_error,
                            style=theme.dim()))
            return
        table.display = bool(detail.steps)
        rows = tuple(step_row(step) for step in detail.steps)
        if rows != self._steps_shown:      # every poll re-renders: rebuild only on change, keep the cursor on its step
            keep = self._step_keys[table.cursor_row] if 0 <= table.cursor_row < len(self._step_keys) else None
            old_row = table.cursor_row
            table.clear()
            keys = [step.step_id for step in detail.steps]
            for step, row in zip(detail.steps, rows, strict=True):
                table.add_row(*row, key=step.step_id)
            if keys:
                table.move_cursor(row=keys.index(keep) if keep in keys else min(old_row, len(keys) - 1),
                                  animate=False)
            self._steps_shown, self._step_keys = rows, keys
        table.styles.height = len(detail.steps) + 2
        bullets = why_bullets(job, detail, self.log_cause)
        if not detail.steps and job.kind is JobKind.TASK:
            bullets.insert(0, Text("workstation tasks have no scheduler steps; see Procs", style=theme.dim()))
        failed = next((e for e in detail.errors if e.section in (None, "slurm.steps")), None)
        if not detail.steps and failed is not None and job.kind is JobKind.SLURM:
            bullets.insert(0, Text(f"steps unavailable: {failed.summary} — r retries", style=theme.warn()))
        why.update(Text("\n").join(bullets) if bullets else Text("nothing to explain", style=theme.dim()))

    def _render_procs(self, job: Job, detail: JobDetail | None) -> None:
        note = self.query_one("#procs-note", Static)
        table = self.query_one(ProcTable)
        if job.kind is JobKind.SLURM:          # the node's processes through Attach (§7.16), with their step
            state = self._attach_state()
            width = (self.size.width or self.app.size.width) - 2
            note.update(Text("\n").join(procs_note(state, job, self.rf.now(), width)))
            live_rows = live_procs(state.live if state is not None else None)
            table.display = any(rows.procs for rows in live_rows)
            if table.display:
                table.show_live(live_rows, sort="gpu_mem", reverse=False)
            return
        procs = detail.procs if detail is not None else ()
        note.update(Text(f"{len(procs)} member processes of {job.key.label} (root first)", style=theme.muted()))
        table.display = True
        state = self.rf.store.hosts.get(self.key.host)
        gpus = state.snapshot.gpus if state is not None and state.snapshot is not None else ()
        table.show(procs, gpus, me=None, show_others=True, sort="pid", reverse=True,
                   task_labels={job.key: job.key.label})

    def _render_raw(self, job: Job, detail: JobDetail | None) -> None:
        self.query_one("#raw-text", Static).update(raw_text(job, detail, self.rf.now()))

    def _stdout_ref(self) -> LogRef | None:
        """The stdout ref with the freshest stat: the polled one, or the detail probe's when it is newer."""
        job = self.job
        if job is None:
            return None
        return fresher_ref(self.rf.log_ref(job, "stdout"), self.detail.logs if self.detail is not None else ())

    def _render_find_hint(self) -> None:
        """The Logs tab's one line when a running job's stdout went quiet (or holds only a launcher's few lines)."""
        job = self.job
        if job is None or not self.is_mounted:
            return
        text = None
        if self.stream != "file":
            text = quiet_hint(job, self._stdout_ref(), self.rf.now(), self.rf.service.config.general.stall_after)
        widget = self.query_one("#logs-hint", Static)
        widget.display = text is not None
        widget.update(Text(text or "", style=theme.warn()))

    def _update_log_titles(self) -> None:
        job = self.job
        if job is None:
            return
        self._render_find_hint()
        now = self.rf.now()
        if self.stream == "file":
            ref = self.rf.stream_ref(job, "file")
            title = Text(file_label(ref), style="bold")
            if ref is not None and ref.path:
                title.append(" · " + humanize.truncate_left(ref.path, 70))
                title.append(" · found in the workdir", style=theme.dim())
        else:
            ref = self.rf.log_ref(job, self.stream)
            detail_ref = None
            if self.detail is not None:
                detail_ref = next((r for r in self.detail.logs if r.stream == self.stream), None)
            ref = detail_ref or ref
            title = Text(self.stream, style="bold")
            if ref is not None:
                title.append(" · " + humanize.truncate_left(ref.path or ref.pattern or "unknown path", 70))
                if ref.size is not None:
                    title.append(f" · {humanize.size(ref.size)}")
                if ref.mtime is not None:
                    title.append(f" · {humanize.ago(ref.mtime, now)}")
                if ref.merged and self.stream == "stderr":
                    title.append(" · merged into stdout", style=theme.dim())
        tabs = Text("  ")
        picked = self.rf.stream_ref(job, "file")
        streams = [("stdout", "o"), ("stderr", "e")]
        if picked is not None:                         # o / e / f(<name>): the way back to each
            streams.append(("file", f"f({file_label(picked, 24)})"))
        for s, key in streams:
            tabs.append(f"▌{key}▐" if s == self.stream else f" {key} ",
                        style=f"bold {theme.ACCENT}" if s == self.stream else theme.muted())
        for view in self.query(LogView):
            view.border_title = title + tabs

    # ------------------------------------------------------------------ FleetScreen
    def current_host(self) -> str | None:
        return self.key.host

    def current_job(self) -> Job | None:
        return self.job

    def visibility(self) -> dict[str, Visibility]:
        return {self.key.host: Visibility.FOCUSED}

    # ------------------------------------------------------------------ actions (keymap "job_detail")
    def help_contexts(self) -> list[str]:
        """``job_live`` first on the GPU / Procs tab of a running Slurm job: there `r` reads the node now."""
        if self._live_tab() and self._live_job():
            return ["job_live", *self.HELP_CONTEXTS]
        return list(self.HELP_CONTEXTS)

    def _tab_ids(self) -> list[str]:
        return list(TABS)

    def action_next_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        ids = self._tab_ids()
        tabs.active = ids[(ids.index(tabs.active) + 1) % len(ids)] if tabs.active in ids else ids[0]

    def action_prev_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        ids = self._tab_ids()
        tabs.active = ids[(ids.index(tabs.active) - 1) % len(ids)] if tabs.active in ids else ids[0]

    def action_stream(self, stream: str) -> None:
        if stream == self.stream:
            return
        if stream == "file" and self.job is not None and self.rf.stream_ref(self.job, "file") is None:
            self._find_logs()              # nothing picked yet
            return
        self.stream = stream
        self._start_log()

    def action_find_logs(self) -> None:
        """`f`: back to the file picked for this job; on it (or none picked yet) the picker of the files written under
        the job's workdir while it ran (one probe, in the picker's worker). Either way on the Logs tab: from Summary
        or GPU the switch would happen out of sight."""
        job = self.job
        if job is not None and self.stream != "file" and self.rf.stream_ref(job, "file") is not None:
            self._show_logs_tab()
            self.action_stream("file")
            return
        self._find_logs()

    def _show_logs_tab(self) -> None:
        tabs = self.query_one(TabbedContent)
        if tabs.active != "logs":
            tabs.active = "logs"

    def _find_logs(self) -> None:
        job = self.job
        problem = find_problem(job)
        if problem is not None or job is None:
            self.rf.toast(problem or "no job", severity="warning")
            return
        current = self._wanted_ref()
        own = own_paths(job, self.detail.logs if self.detail is not None else ())    # the detail may resolve them

        def picked(found: Any) -> None:
            if found is None or not self.is_mounted or self.job is None:
                return
            self.stream = remember_pick(self.rf.found_logs, self.job, found, own)
            self._show_logs_tab()
            self._start_log()

        self.app.push_screen(picker(self.rf, job, current.path if current is not None else None, own), picked)

    def action_search_log(self) -> None:
        """`/` searches the log (in the log viewer, which keeps this screen underneath) — never the Jobs filter."""
        self.rf.action_log_viewer(search=True)

    def action_open_cause(self) -> None:
        """`L`: the log viewer, landed on the first loaded line of the cause chip (OOM-GPU, NCCL, TIMEOUT, …); without
        such a line it opens as `l` does and says why."""
        from runfleet.ui.screens.log_viewer import LogScreen

        cause = self.cause
        self.rf.action_log_viewer()
        screen = self.app.screen
        if not isinstance(screen, LogScreen) or screen.key != self.key:
            return
        if cause is not None and cause.evidence and cause.pattern is not None:
            # the evidence line itself (its head: a long line wraps over several display lines)
            screen.open_at = re.compile(re.escape(cause.evidence[:30]))
        else:
            self.rf.toast("no failure line in the loaded log" if cause is not None else "no cause: the job has not "
                          "failed", severity="warning")

    def action_toggle_follow(self) -> None:
        for view in self.query(LogView):
            view.set_follow(not view.follow)

    def action_reload(self) -> None:
        """`r`: poll the host (the service keeps its 3 s floor) and re-fetch the detail — never restarting a probe in
        flight (one refresh is queued instead) and reusing one younger than DETAIL_MIN_GAP; reopens a dead log. On the
        GPU / Procs tab of a running Slurm job it also reads the node now (Attach; srun asks first)."""
        live = self._live_tab() and self._live_job()
        if live:
            self.read_live()           # its status line shows the read; a blocked one toasts why
        self.rf.service.refresh([self.key.host])
        if self._detail_busy():
            self._reload_pending = True
        else:
            self.load_detail(DETAIL_MIN_GAP)
        job, follower = self.job, self.follower
        if follower is not None and follower.dead and job is not None and not job.state.is_terminal:
            self._start_log()
        if not live:
            self.rf.toast(f"refreshing {self.key}")

    def _live_ready(self, what: str) -> bool:
        if self._live_tab() and self._live_job():
            return True
        self.rf.toast(f"{what}: on the GPU / Procs tab of a running Slurm job", severity="warning")
        return False

    def action_attach_auto(self) -> None:
        """`a`: ssh auto-refresh on / off for this job (srun never refreshes by itself)."""
        if not self._live_ready("auto-refresh"):
            return
        state = self._attach_state()
        if state is None or state.mode != "ssh":
            why = state.blocked if state is not None and state.blocked else "srun reads only when you press r"
            self.rf.toast(f"no auto-refresh here: {why}", severity="warning")
            return
        new = self.rf.service.attach_set_auto(self.key, not state.auto)
        self.rf.toast(f"live auto-refresh {'on' if new.auto else 'off'}")
        self._render_live()

    def action_attach_srun(self) -> None:
        """`S`: read this job's own GPUs with srun (ssh landed in another job, or srun is the only path); asks first."""
        if not self._live_ready("srun read"):
            return
        state = self._attach_state()
        if state is None or state.offer is None:
            why = (state.blocked or "ssh reads this job's own GPUs") if state is not None else "not attached"
            self.rf.toast(f"no srun read offered: {why}", severity="warning")
            return
        self.read_live("srun")
