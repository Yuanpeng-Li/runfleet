"""[WP3] Job preview pane: summary and log peek.

Follows the Jobs-table cursor (lazygit style). Full form: a title rule, a metadata line, a GPU line with the stdout
path, then the last log lines from a cached one-shot ``peek_log`` (requested by the Jobs view after 1 s of cursor
dwell — never a stream). Compact form (< 30 rows): the title rule with the essentials plus one log line. An ended job
shows its cause chip (``OOM-GPU`` + the peeked line that shows it), a pending one what to do about its reason and,
for an own pending Slurm job, its priority evidence in one line (§5.2.10: polled, no extra remote command).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from rich.text import Text
from textual.widgets import Static

from runfleet.core.service import LogPeek
from runfleet.models import Capabilities, Job, JobKind, StateGroup
from runfleet.ui import render, theme
from runfleet.ui.query import elapsed_s
from runfleet.util import humanize
from runfleet.util.logscan import cause_of

Mode = Literal["full", "line", "hidden"]


def _rule(left: Text, right: Text, width: int) -> Text:
    """``─ left ───────── right ─`` exactly ``width`` cells."""
    out = Text("─ ", style=theme.dim())
    out.append_text(left)
    out.append(" ")
    tail = Text(" ")
    tail.append_text(right)
    tail.append(" ─", style=theme.dim())
    fill = width - out.cell_len - tail.cell_len
    if fill < 1:
        out = render.fit(out, max(0, width - tail.cell_len - 1))
        fill = 1
    out.append("─" * fill, style=theme.dim())
    out.append_text(tail)
    return out


def _runtime(job: Job, now: datetime) -> Text:
    out = render.state_word(job.state, job.raw_state)
    el = elapsed_s(job, now)
    if el is not None:
        out.append(f" · {humanize.clock(el)}", style=theme.muted())
        if job.time_limit is not None:
            out.append(f" / {humanize.clock(job.time_limit)}", style=theme.muted())
    elif job.time_limit is not None:
        out.append(f" · limit {humanize.clock(job.time_limit)}", style=theme.muted())
    return out


def gpu_indices(job: Job) -> str:
    idx = sorted({i for n in job.alloc.per_node for i in n.gpu_indices})
    return ",".join(map(str, idx))


def meta_line(job: Job) -> Text:
    parts: list[str] = []
    if job.kind is JobKind.TASK:
        parts.append(f"root pid {job.job_id_raw}")
        parts.append(f"{len(job.pids)} process{'es' if len(job.pids) != 1 else ''}")
        if job.cpu_pct is not None:
            parts.append(f"cpu {humanize.cpu_load(job.cpu_pct)}")
        if job.rss_bytes is not None:
            parts.append(f"rss {humanize.size(job.rss_bytes)}")
        if job.workdir:
            parts.append(f"cwd {job.workdir}")
        if not job.is_mine:
            parts.insert(0, f"user {job.user}")
    else:
        for label, value in (("partition", job.partition), ("qos", job.qos), ("account", job.account)):
            if value:
                parts.append(f"{label} {value}")
        if job.nodes:
            parts.append(("node " if len(job.nodes) == 1 else "nodes ") + (job.nodelist or ",".join(job.nodes)))
        if job.alloc.cpus:
            parts.append(f"{job.alloc.cpus} cpu")
        if job.alloc.mem_bytes:
            parts.append(humanize.size(job.alloc.mem_bytes))
        if job.alloc.gpus:
            gpu = humanize.short_gpu_name(job.alloc.gpu_type) or "GPU"
            idx = gpu_indices(job)
            parts.append(f"{job.alloc.gpus}× {gpu}" + (f" idx {idx}" if idx else ""))
        if job.state.group is StateGroup.QUEUED and job.reason:
            parts.append(f"reason {render.reason_text(job.reason, short=False).plain}")
    out = Text(" · ".join(parts), style=theme.muted())
    return out


def gpu_parts(job: Job, caps: Capabilities | None, now: datetime, thresholds: tuple[float, float],
              mem_thresholds: tuple[float, float], *, bar: int = 12) -> list[Text]:
    """GPU facts as pieces: ``[heading, util, mem]``, or one ``n/a — why`` piece when there is nothing to show."""
    usage = job.gpu
    if job.alloc.gpus == 0 and usage is None:
        return [Text("GPU  none allocated", style=theme.dim())]
    if job.state.group is StateGroup.QUEUED:
        return [Text("GPU  n/a — job pending", style=theme.dim())]
    if job.state.is_terminal:
        return [Text("GPU  n/a — job ended", style=theme.dim())]
    if usage is None:
        reason = "no GPU accounting on this cluster" if caps is not None and caps.gpu_tres is False \
            else "not sampled yet"
        return [Text(f"GPU  n/a — {reason}", style=theme.dim())]
    age = humanize.age((now - usage.sampled_at).total_seconds()) if usage.sampled_at else "?"
    head = Text("GPU ", style="bold")
    head.append(f"({usage.source} · {age})", style=theme.muted())
    util = Text("util ", style=theme.muted())
    util.append_text(render.gpu_bar(usage.util_mean_pct, bar, thresholds))
    util.append(humanize.pct(usage.util_mean_pct, width=5))
    mem_pct = None
    if usage.mem_used_mib is not None and usage.mem_total_mib:
        mem_pct = usage.mem_used_mib / usage.mem_total_mib * 100
    mem = Text("mem  ", style=theme.muted())
    mem.append_text(render.gpu_bar(mem_pct, bar, mem_thresholds))
    mem.append(" " + humanize.gib_pair(usage.mem_used_mib, usage.mem_total_mib))
    return [head, util, mem]


def gpu_line(job: Job, caps: Capabilities | None, now: datetime, thresholds: tuple[float, float],
             mem_thresholds: tuple[float, float]) -> Text:
    """One line: ``GPU (sstat · 30s)  util ▕███▊▏ 98%   mem ▕██  ▏ 24.3/40.0G``."""
    parts = gpu_parts(job, caps, now, thresholds, mem_thresholds)
    out = parts[0].copy()
    for i, part in enumerate(parts[1:]):
        out.append("  " if i == 0 else "   ")
        out.append_text(part)
    return out


def cause_text(job: Job, peek: LogPeek | None, width: int) -> Text | None:
    """§5.2.8: an ended job's cause chip with the first peeked log line that shows it (else its one-line
    explanation); a pending job's "what to do" line. Only the cached peek is read — no extra remote command."""
    if job.state.group is StateGroup.QUEUED:
        advice = job.raw.get("reason_advice")
        return render.fit(Text("→ " + advice, style=theme.muted()), width) if advice else None
    cause = cause_of(job, None, peek.lines if peek is not None and not peek.error else ())
    if cause is None:
        return None
    out = render.cause_chip(cause.code)
    out.append(" ")
    out.append(cause.evidence or cause.text, style=theme.error() if cause.evidence else theme.muted())
    return render.fit(out, width)


def log_lines(peek: LogPeek | None, pending: bool, n: int, width: int, now: datetime,
              has_log: bool) -> list[Text]:
    if n <= 0:
        return []
    bar = Text("│ ", style=theme.dim())
    if not has_log:
        return [bar + Text("no log path known", style=theme.dim())]
    if peek is None:
        return [bar + Text("loading log preview…" if pending else "log preview after 1 s on a row",
                           style=theme.dim())]
    if peek.error:
        return [bar + Text(peek.error, style=theme.warn())]
    lines = [bar + Text.from_ansi(line) for line in peek.lines[-n:]] or [bar + Text("(empty log)",
                                                                                    style=theme.dim())]
    info = f"log {humanize.size(peek.size)}"
    if peek.mtime is not None:
        info += f" · {humanize.ago(peek.mtime, now)}"
    last = lines[-1]
    room = width - last.cell_len - len(info) - 1
    if room >= 1:
        lines[-1] = Text.assemble(last, " " * room, Text(info, style=theme.dim()))
    else:
        lines[-1] = render.fit(last, width)
    return [render.fit(x, width) for x in lines]


class JobPreview(Static):
    DEFAULT_CSS = """
    JobPreview { height: auto; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.mode: Mode = "full"
        self.side = False
        self.lines_budget = 7

    def show_empty(self, text: str) -> None:
        width = self.size.width or 120
        self.update(Text("\n").join([_rule(Text(text, style=theme.dim()), Text(""), width)]))

    def show_text(self, title: Text, body: list[Text]) -> None:
        width = self.size.width or 120
        self.update(Text("\n").join([_rule(title, Text(""), width), *[render.fit(b, width) for b in body]]))

    def show_job(self, job: Job, *, caps: Capabilities | None, now: datetime, peek: LogPeek | None,
                 peek_pending: bool, thresholds: tuple[float, float], mem_thresholds: tuple[float, float]) -> None:
        width = self.size.width or 120
        title = Text(f"{job.key.host}:{job.key.label}", style="bold")
        title.append(" · ", style=theme.dim())
        title.append(job.name)
        stdout = next((r for r in job.logs if r.stream == "stdout"), None)
        has_log = stdout is not None and (stdout.path is not None or stdout.pattern is not None)
        if self.mode == "line":
            bits = Text(" ")
            bits.append_text(render.state_badge(job.state, job.held))
            el = elapsed_s(job, now)
            if el is not None:
                bits.append(f" {humanize.clock(el)}")
            node = job.nodelist or (f"gpu {gpu_indices(job)}" if job.kind is JobKind.TASK else "")
            if node:
                bits.append(f" · {node}", style=theme.muted())
            if job.gpu is not None and job.gpu.util_mean_pct is not None and job.state.group is StateGroup.ACTIVE:
                bits.append(f" · GPU {humanize.pct(job.gpu.util_mean_pct)}", style=theme.muted())
                if job.gpu.mem_used_mib is not None:
                    bits.append(f" · {humanize.gib_pair(job.gpu.mem_used_mib, job.gpu.mem_total_mib)}",
                                style=theme.muted())
            cause = cause_of(job, None, peek.lines if peek is not None and not peek.error else ())
            if cause is not None:
                bits.append(" ")
                bits.append_text(render.cause_chip(cause.code))
            head = Text(f"{job.key.host}:{job.key.label}", style="bold") + bits
            lines = [_rule(head, Text(""), width)]
            lines += log_lines(peek, peek_pending, 1, width, now, has_log)
            self.update(Text("\n").join(lines[:2]))
            return
        if self.side:
            self._show_side(job, title, caps, now, peek, peek_pending, thresholds, mem_thresholds, has_log)
            return
        lines = [_rule(title, _runtime(job, now), width), render.fit(meta_line(job), width)]
        gl = gpu_line(job, caps, now, thresholds, mem_thresholds)
        if stdout is not None and (stdout.path or stdout.pattern):
            path = stdout.path or stdout.pattern or ""
            room = width - gl.cell_len - 12
            if room > 12:
                gl.append(" " * 5)
                gl.append("stdout ", style=theme.muted())
                gl.append(humanize.truncate_left(path, room), style=theme.muted())
        lines.append(render.fit(gl, width))
        chip = cause_text(job, peek, width)
        if chip is not None:
            lines.append(chip)
        if job.priority is not None:
            lines.append(render.fit(render.priority_summary(job.priority, job.account, now), width))
        for hint in job.hints if chip is None or not job.state.is_terminal else ():    # the chip says it already
            level = render.hint_level(hint)
            lines.append(Text(("! " if level != "info" else "· ") + hint, style=render.hint_style(hint)))
        lines += log_lines(peek, peek_pending, max(1, self.lines_budget - len(lines)), width, now, has_log)
        self.update(Text("\n").join(lines[: self.lines_budget]))

    def _show_side(self, job: Job, title: Text, caps: Capabilities | None, now: datetime, peek: LogPeek | None,
                   peek_pending: bool, thresholds: tuple[float, float], mem_thresholds: tuple[float, float],
                   has_log: bool) -> None:
        """Right-hand preview (>= 150 columns): the same facts, wrapped into a narrow, tall column."""
        width = self.size.width or 60
        lines = [render.fit(title, width), render.fit(_runtime(job, now), width), Text("")]
        lines += wrap_parts(meta_line(job).plain.split(" · "), width, style=theme.muted())
        lines.append(Text(""))
        bar = max(8, min(24, width - 18))
        lines += [render.fit(x, width) for x in gpu_parts(job, caps, now, thresholds, mem_thresholds, bar=bar)]
        stdout = next((r for r in job.logs if r.stream == "stdout"), None)
        if stdout is not None and (stdout.path or stdout.pattern):
            lines.append(render.fit(Text("stdout " + (stdout.path or stdout.pattern or ""), style=theme.muted()),
                                    width) if len(stdout.path or "") < width - 7 else
                         Text("stdout " + humanize.truncate_left(stdout.path or "", width - 7), style=theme.muted()))
        chip = cause_text(job, peek, width)
        if chip is not None:
            lines.append(chip)
        if job.priority is not None:
            lines += wrap_parts(render.priority_summary(job.priority, job.account, now).plain.split(" · "), width,
                                style=theme.muted())
        for hint in job.hints if chip is None or not job.state.is_terminal else ():
            lines.append(render.fit(Text("! " + hint, style=render.hint_style(hint)), width))
        lines.append(Text(""))
        lines += log_lines(peek, peek_pending, max(1, self.lines_budget - len(lines)), width, now, has_log)
        self.update(Text("\n").join(lines[: self.lines_budget]))


def wrap_parts(parts: list[str], width: int, *, style: str = "") -> list[Text]:
    """Pack ``a · b · c`` parts into lines no wider than ``width`` (a part longer than a line is cut)."""
    lines: list[Text] = []
    current = ""
    for part in parts:
        candidate = f"{current} · {part}" if current else part
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(Text(current, style=style))
        current = humanize.truncate(part, width)
    if current:
        lines.append(Text(current, style=style))
    return lines
