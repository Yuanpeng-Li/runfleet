"""[WP3] Pure Rich renderables: gpu_bar, state_badge, host_chip, mini_util.

No Textual, no I/O, no clock: every function maps values to a ``rich.text.Text`` and accepts ``None`` for unknown
inputs (rendered as a dash or an empty bar, never as 0).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from rich.text import Text

from runfleet.models import (ConnState, ErrorKind, FairShare, HostStatus, JobState, PartitionSummary,
                             PendingPriority)
from runfleet.ui import theme
from runfleet.util import humanize
from runfleet.util.gpufree import (GPU_BUSY_MEM_MIB as GPU_BUSY_MEM_MIB, GPU_BUSY_UTIL as GPU_BUSY_UTIL,
                                   gpu_in_use as gpu_in_use)     # moved to util.gpufree (shared with `runfleet agent`)

_EIGHTHS = " ▏▎▍▌▋▊▉█"      # index = eighths of a cell
_SPARK = "▁▂▃▄▅▆▇█"
BAR_LEFT = "▕"
BAR_RIGHT = "▏"


def _fill(value: float, inner: int) -> str:
    """Bar body of ``inner`` cells for ``value`` percent, with eighth-block resolution."""
    value = max(0.0, min(100.0, value))
    eighths = round(value / 100 * inner * 8)
    full, part = divmod(eighths, 8)
    body = "█" * full
    if part and full < inner:
        body += _EIGHTHS[part]
    return body.ljust(inner)


def gpu_bar(value: float | None, width: int, thresholds: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS,
            *, frame: bool = True) -> Text:
    """nvitop-style bar ``▕██████▊   ▏`` of total ``width`` cells, coloured by threshold level.

    ``None`` renders an empty frame with a dim ``n/a`` so an unknown value never looks like 0 %.
    """
    inner = max(0, width - 2) if frame else max(0, width)
    out = Text()
    if frame:
        out.append(BAR_LEFT, style=theme.dim())
    if value is None:
        label = "n/a" if inner >= 3 else ""
        out.append(label.center(inner), style=f"{theme.dim()} {theme.track()}".strip())
    else:
        body = _fill(value, inner)
        filled = body.rstrip(" ")
        style = theme.level_style(value, thresholds)
        out.append(filled, style=f"{style} {theme.track()}".strip())
        out.append(" " * (inner - len(filled)), style=theme.track())
    if frame:
        out.append(BAR_RIGHT, style=theme.dim())
    return out


def count_bar(used: int | None, total: int | None, *, max_cells: int = 16) -> Text:
    """Discrete GPU usage ``███░`` (one cell per GPU; scaled down above ``max_cells``)."""
    if used is None or total is None or total <= 0:
        return Text("")
    cells = min(total, max_cells)
    filled = round(used / total * cells)
    out = Text("█" * filled, style=theme.level_style(used / total * 100, theme.DEFAULT_MEM_THRESHOLDS))
    out.append("░" * (cells - filled), style=theme.dim())
    return out


def mini_util(pct: float | None, thresholds: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> Text:
    """Jobs-table GPU% cell: one spark block + percentage (``▇ 98%``); ``–`` when unavailable."""
    if pct is None:
        return Text(humanize.NA, style=theme.dim())
    clamped = max(0.0, min(100.0, pct))
    block = _SPARK[min(len(_SPARK) - 1, int(clamped / 100 * len(_SPARK)))]
    out = Text(block, style=theme.level_style(clamped, thresholds))
    out.append(" " + humanize.pct(pct, width=4))
    if theme.no_color() and theme.level(clamped, thresholds) == "high":
        out.append("!", style="bold")            # NO_COLOR: the high level needs a mark of its own
    return out


def state_badge(state: JobState | None, held: bool = False) -> Text:
    """State code coloured by group; ``H`` suffix when held (``PDH``)."""
    if state is None:
        return Text("?", style=theme.state_style(JobState.UNKNOWN))
    return Text(state.value + ("H" if held else ""), style=theme.state_style(state))


def state_word(state: JobState | None, raw: str | None = None) -> Text:
    """Long state for detail headers: ``RUNNING``, ``FAILED`` (raw Slurm word when present)."""
    if state is None:
        return Text("UNKNOWN", style=theme.state_style(JobState.UNKNOWN))
    word = (raw or state.name).split(" ")[0].upper() or state.name
    return Text(word, style=theme.state_style(state))


ChipForm = Literal["full", "short", "name", "num"]


def chip_suffix(status: HostStatus, age_s: float | None, retry_in: float | None) -> str:
    """What follows the host name in a full chip: age, ``retry 38s``, ``auth``, ``dns``, ``off``."""
    st = status.state
    if st is ConnState.RETRYING:
        return f"retry {humanize.age(retry_in)}" if retry_in is not None else "retry"
    if st is ConnState.NEEDS_AUTH:
        return "auth"
    if st is ConnState.HOSTKEY:
        return "hostkey"
    if st is ConnState.UNREACHABLE:
        return "dns" if status.error is not None and status.error.kind is ErrorKind.DNS else "down"
    if st is ConnState.DISABLED:
        return "off"
    if st in (ConnState.NEW, ConnState.CONNECTING):
        return "…"
    return humanize.age(age_s, dash="")


def host_chip(status: HostStatus, age_s: float | None = None, *, retry_in: float | None = None,
              form: ChipForm = "full", number: int | None = None) -> Text:
    """Header chip: ``● k2 8s`` (full), ``● k2`` / ``✖ lynx auth`` (short: no ages), ``●k2`` (name), ``●1`` (num);
    the compact forms of a problem host end in ``:d`` (down), ``:a`` (needs auth) or ``:k`` (host key): ``✖lynx:a``.

    Degraded hosts show a yellow ``!`` instead of the dot.
    """
    style = theme.conn_style(status.state)
    glyph = "!" if status.state is ConnState.DEGRADED else theme.CONN_GLYPH[status.state]
    out = Text(glyph, style=style)
    # compact forms keep a one-letter code: down, needs auth and host key share ✖ (and, under NO_COLOR, one style)
    code = {ConnState.UNREACHABLE: ":d", ConnState.NEEDS_AUTH: ":a", ConnState.HOSTKEY: ":k"}.get(status.state, "")
    if form == "num":
        out.append(str(number if number is not None else "?") + code, style=style)
        return out
    if form == "name":
        out.append(status.name + code, style=style if theme.is_problem(status.state) else "")
        return out
    out.append(" ")
    out.append(status.name, style="bold" if status.state is ConnState.OK else style)
    suffix = chip_suffix(status, age_s if form == "full" else None, retry_in if form == "full" else None)
    if suffix:
        out.append(" " + suffix, style=theme.dim() if status.state is ConnState.OK else style)
    return out


def conn_text(state: ConnState) -> Text:
    """``● ok`` / ``◐ retrying`` / ``✖ needs auth`` for cards, tables and the host screen."""
    return Text(f"{theme.CONN_GLYPH[state]} {theme.CONN_LABEL[state]}", style=theme.conn_style(state))


def label_value(label: str, value: str | Text, *, width: int = 10, value_style: str = "") -> Text:
    """``label     value`` with a dim, padded label."""
    out = Text(label.ljust(width), style=theme.label())
    if isinstance(value, Text):
        out.append_text(value)
    else:
        out.append(value, style=value_style)
    return out


def fit(text: Text, width: int) -> Text:
    """Pad or crop a Text to exactly ``width`` cells (ellipsis when cropped)."""
    out = text.copy()
    if out.cell_len > width:
        out.truncate(width, overflow="ellipsis")
    else:
        out.pad_right(width - out.cell_len)
    return out


# Pending-reason wording (§5.2.6). Display only: Job.hints from the backend carry the authoritative hint text.
_REASONS: dict[str, tuple[str, str, bool]] = {   # reason -> (short, long, red)
    "Priority": ("priority", "waiting for priority", False),
    "Resources": ("resources", "waiting for resources", False),
    "QOSMaxJobsPerUserLimit": ("per-user job cap (QOS)", "per-user job cap (QOS)", False),
    "AssocGrpGRES": ("group GPU cap", "group GPU cap", False),
    "Dependency": ("dependency", "waiting for dependency", False),
    "DependencyNeverSatisfied": ("never: dependency failed", "will never run: dependency failed", True),
    "ReqNodeNotAvail": ("node unavailable", "requested node unavailable", False),
    "PartitionTimeLimit": ("never: over time limit", "time limit above partition max — will never run", True),
    "BeginTime": ("begins later", "scheduled start later", False),
    "JobHeldUser": ("held", "held", False),
    "JobHeldAdmin": ("held (admin)", "held", False),
    # the rest of the common set (§5.2.8); the backend's states.REASONS / REASON_ADVICE carry the same wording
    "QOSMaxGRESPerUser": ("per-user GPU cap (QOS)", "per-user GPU cap (QOS)", False),
    "QOSMaxCpuPerUserLimit": ("per-user CPU cap (QOS)", "per-user CPU cap (QOS)", False),
    "QOSMaxMemoryPerUser": ("per-user mem cap (QOS)", "per-user memory cap (QOS)", False),
    "QOSMaxNodePerUserLimit": ("per-user node cap (QOS)", "per-user node cap (QOS)", False),
    "QOSGrpGRES": ("QOS GPU cap", "QOS-wide GPU cap", False),
    "QOSGrpCpuLimit": ("QOS CPU cap", "QOS-wide CPU cap", False),
    "QOSGrpMemLimit": ("QOS mem cap", "QOS-wide memory cap", False),
    "QOSGrpNodeLimit": ("QOS node cap", "QOS-wide node cap", False),
    "QOSMaxWallDurationPerJobLimit": ("never: over QOS time limit", "time limit above QOS max — will never run",
                                      True),
    "AssocGrpCPULimit": ("group CPU cap", "group CPU cap", False),
    "AssocGrpCpuLimit": ("group CPU cap", "group CPU cap", False),
    "AssocGrpMemLimit": ("group mem cap", "group memory cap", False),
    "AssocGrpNodeLimit": ("group node cap", "group node cap", False),
    "AssocGrpJobsLimit": ("group job cap", "group job cap", False),
    "AssocMaxJobsLimit": ("per-user job cap", "per-user job cap (association)", False),
    "AssocGrpBillingMinutes": ("budget used up", "group compute budget used up", False),
    "AssocGrpCPUMinutesLimit": ("CPU budget used up", "group CPU-minutes budget used up", False),
    "AssocGrpGRESMinutes": ("GPU budget used up", "group GPU-minutes budget used up", False),
    "MaxGRESPerAccount": ("account GPU cap", "account GPU cap", False),
    "Nodes required for job are DOWN": ("nodes down", "required nodes down / drained", False),
    "PartitionNodeLimit": ("never: node count", "node count outside partition limits — will never run", True),
    "launch failed requeued held": ("launch failed, held", "launch failed, requeued and held", False),
    "JobArrayTaskLimit": ("array task limit", "array task limit", False),
    "Reservation": ("reservation", "waiting for reservation", False),
    "BadConstraints": ("never: bad constraints", "constraints can never be met — will never run", True),
    "InvalidAccount": ("never: invalid account", "invalid account — will never run", True),
    "InvalidQOS": ("never: invalid QOS", "invalid QOS — will never run", True),
}


def known_reason(reason: str) -> bool:
    return reason.split(",")[0].strip("()") in _REASONS


HintLevel = Literal["red", "warn", "info"]


def reason_red(reason: str | None) -> bool:
    """The pending reason means the job will never run (shown red)."""
    if not reason:
        return False
    entry = _REASONS.get(reason.split(",")[0].strip("()"))
    return bool(entry and entry[2])


def hint_level(hint: str) -> HintLevel:
    """Backend hints: "will never run …" and exit-scan "log: …" lines are errors, stall hints warnings."""
    if hint.startswith(("will never", "log:", "exited (log")) or "never run" in hint:
        return "red"
    if "idle" in hint or "waiting for accounting" in hint:
        return "warn"
    return "info"


def hint_style(hint: str) -> str:
    level = hint_level(hint)
    return theme.hint_style(True) if level == "red" else theme.warn() if level == "warn" else theme.muted()


def reason_text(reason: str | None, *, short: bool = True) -> Text:
    """Humanized Slurm pending reason; red when the job will never run; raw text for unknown reasons."""
    if not reason or reason == "None":
        return Text("")
    key = reason.split(",")[0].strip("()")
    entry = _REASONS.get(key)
    if entry is None:
        return Text(reason, style=theme.muted())
    text = entry[0] if short else entry[1]
    return Text(text, style=theme.hint_style(True) if entry[2] else theme.muted())


def _chip_color(code: str) -> str:
    return {"CANCELLED": theme.MAGENTA, "PREEMPTED": theme.ORANGE, "TIMEOUT": theme.ORANGE}.get(code, theme.RED)


def cause_chip(code: str, *, pending: bool = False) -> Text:
    """The cause chip of an ended job (``logscan.Cause.code``: `` OOM-GPU ``) — or, ``pending``, of a pending reason —
    in panel colour on red (orange for time-outs and preemption, magenta for cancellations, yellow when pending)."""
    if theme.no_color():
        return Text(f" {code} ", style="bold reverse")
    color = theme.YELLOW if pending else _chip_color(code)
    return Text(f" {code} ", style=theme.on_color(color))


def free_gpus(p: PartitionSummary, *, sep: str = "/") -> Text:
    """``5/121 +17 MIG``: whole free GPUs over whole GPUs, then free MIG slices apart (on k2 most "free GPUs" were
    1g.10gb slices). Partitions without MIG read ``22/138``. ``(4 planned)`` follows when some of the free GPUs sit on
    nodes backfill has planned for a pending job (usable only by jobs short enough to finish first)."""
    wf, wt, mf, mt = humanize.gpu_free_split(p.gpu_types)
    if not mt:
        wf, wt = p.gpus_free, p.gpus_total
    out = Text(str(wf), style=theme.ok() if wf else theme.dim())
    out.append(f"{sep}{wt}", style=theme.muted())
    if mt:
        out.append(f" +{mf} MIG", style=theme.ok() if mf else theme.dim())
    if p.gpus_free_planned:                # free right now, but backfill has a pending job planned on those nodes
        out.append(f" ({p.gpus_free_planned} planned)", style=theme.dim())
    return out


# ---------------------------------------------------------------- pending priority (§5.2.10)
PRIORITY_LABELS = {"age": "age", "assoc": "assoc", "fairshare": "fairshare", "jobsize": "job size",
                   "partition": "partition", "qos": "QOS", "tres": "TRES", "nice": "nice"}
FAIRSHARE_LOW = 0.25             # the collector's hint thresholds (backends/slurm/priority.py), for colour only
FAIRSHARE_HIGH = 0.75


def prio_num(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 10 or value == int(value) else f"{value:.2f}"


def share_bar(fraction: float | None, width: int) -> Text:
    """A neutral ``▕███  ▏`` of ``width`` cells: how much of a factor's weight a pending job gets."""
    if fraction is None:
        return gpu_bar(None, width)
    inner = max(0, width - 2)
    body = _fill(fraction * 100, inner)
    filled = body.rstrip(" ")
    out = Text(BAR_LEFT, style=theme.dim())
    out.append(filled, style=f"{theme.accent()} {theme.track()}".strip())
    out.append(" " * (inner - len(filled)), style=theme.track())
    out.append(BAR_RIGHT, style=theme.dim())
    return out


def job_share(p: PendingPriority, account: str | None) -> FairShare | None:
    """The sshare row of the job's account (the collector puts it first), else the only one."""
    return next((s for s in p.fairshare if s.account == account), None) or (
        p.fairshare[0] if len(p.fairshare) == 1 else None)


def fairshare_style(value: float | None) -> str:
    if value is None:
        return theme.dim()
    return theme.warn() if value < FAIRSHARE_LOW else theme.ok() if value >= FAIRSHARE_HIGH else ""


def priority_summary(p: PendingPriority, account: str | None, now: datetime) -> Text:
    """One line for the Jobs preview and the compact job detail: ``prio 3,017 · fairshare dominates · fairshare 0.03
    · 4m ago`` — evidence, never a start time or a queue position."""
    out = Text("prio ", style=theme.label())
    if not p.ranked:
        out.append("not ranked here: submit order as resources free up (+ backfill)", style=theme.muted())
        return out
    if p.priority is None:
        out.append("not read yet (sprio runs at most every 10 min)", style=theme.dim())
    else:
        out.append(prio_num(p.priority), style="bold")
        if p.dominant:
            out.append(f" · {PRIORITY_LABELS.get(p.dominant, p.dominant)} dominates", style=theme.muted())
    share = job_share(p, account)
    if share is not None and share.fairshare is not None:
        out.append(" · fairshare ", style=theme.muted())
        out.append(f"{share.fairshare:.2f}", style=fairshare_style(share.fairshare))
    if p.sampled_at is not None:
        out.append(f" · {humanize.ago(p.sampled_at, now)}", style=theme.dim())
    return out
