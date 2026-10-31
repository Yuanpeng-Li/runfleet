"""[Account] Account usage screen (`$`, §7.17, mockups O / O2): per Slurm cluster in scope the account balances
(used / limit / available SU, reallocation), my usage in the window, the charged and free partitions and "where it
went" — the rollup of my job records by experiment / task / name / account / partition / GPU model (§5.2.11).

Nothing here starts a probe on a timer. The screen paints at once from ``FleetAPI.account_cached`` (else the host
snapshot's periodic summary) and asks ``account_report`` only when it opens, on a scope change, on a window switch
(``[`` / ``]``) and on ``r`` / ``^r`` (forced, at most once per 60 s per host and window); the service answers from its
10-minute cache otherwise. A failed fetch keeps the last report, dimmed by nothing but its age, with one red line per
failing section worded from the error kind — never zeros. A stale host's box says "as of …".

The boxes are drawn here (one ``Static``) rather than as Textual borders: the rollup sits under a ``├ … ┤`` divider
inside its cluster's box, and boxes that do not fit the screen height collapse to their title and a one-line summary
(or, short of even that, to a mention in the line under the boxes), the focused one last. ``tab`` moves the focus
between boxes, ``j`` / ``k`` between the rows of the focused box (accounts, then the rollup; past the last row on to
the next box). The pure renderers below are tested on their own; ``account_summary_line`` is the cluster host
detail's `account` line (§7.17 "Host detail").
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from runfleet.config import format_duration
from runfleet.core.store import HostState
from runfleet.models import (USAGE_WINDOWS, WINDOW_DAYS, AccountBalance, AccountReport, AccountSummary, AccountUse,
                             Capabilities, ConnState, ErrorKind, HostError, UsageWindow, Visibility)
from runfleet.ui import keymap, render, theme
from runfleet.ui.messages import FleetChanged, ScopeChanged
from runfleet.ui.screens import FleetScreen
from runfleet.ui.widgets.header import FleetHeader, _click
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.host_card import is_cluster
from runfleet.ui.widgets.job_table import fit_columns
from runfleet.util import cost as costs
from runfleet.util import humanize
from runfleet.util.usage import GROUP_BYS, GroupBy, RollupRow, group

log = logging.getLogger("runfleet.ui.account")

Scope = Literal["me", "account"]
SortKey = Literal["gpu_h", "su", "cpu_h", "jobs", "name"]
SORTS: tuple[SortKey, ...] = ("gpu_h", "su", "cpu_h", "jobs", "name")   # `o` order (§7.8)
GROUP_LABELS: Mapping[str, str] = {"experiment": "experiment", "task": "task", "name": "name", "account": "account",
                                   "partition": "partition", "gpu": "GPU model"}
MAX_ROLLUP = 12                   # rollup rows shown at most, then `… N more (↓)`
MIN_ROLLUP = 3                    # a full box shows at least this many (else it collapses)
COMPACT_ROWS = 4                  # a collapsed box: borders + title summary + one more line
REFRESH_GAP_S = 60.0              # `r` at most once per (host, window) in this time
REPORT_MAX_AGE_S = 600.0          # what account_report answers from its cache (opening, window switch)
NARROW = 100                      # below: the O2 layout (fewer columns, one partitions line, no basis line)
_SEP = " · "
RESET_PERIODS = ("DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY")   # = backends.slurm.account (ui/ imports
                                                                        # only the backend contract)


# ---------------------------------------------------------------- pure helpers
def window_since(window: UsageWindow, now: datetime) -> str:
    """`(since 09-30)` for a fixed window (local month-day of its start)."""
    days = WINDOW_DAYS.get(window, 30)
    return f"(since {(now - timedelta(days=days)).astimezone().strftime('%m-%d')})"


def period_text(report: AccountReport | None, allocation_period: float) -> str | None:
    """What the period window means for one host's report: `since 2026-01-14, inferred: next reallocation − 365d`,
    `since the monthly reset`; None when the report is not a period report."""
    if report is None or report.window != "period":
        return None
    use = report.summary.usage or report.account_totals
    inferred = any(u.start_inferred for u in use)
    if inferred and report.window_start is not None:
        day = min((u.start for u in use if u.start_inferred and u.start is not None), default=report.window_start)
        return f"since {day.date().isoformat()}, inferred: next reallocation − {format_duration(allocation_period)}"
    reset = report.summary.reset_period
    if reset:
        return f"since the {reset.lower()} reset"
    if report.window_start is not None:
        return f"since {report.window_start.date().isoformat()}"
    return None


def sort_rollup(rows: Sequence[RollupRow], sort: SortKey, reverse: bool = False, *,
                usage_only: bool = False) -> list[RollupRow]:
    """GPU-h ▼ (then billing-h), SU ▼ (billing-h ▼ on a usage-only cluster), CPU-h ▼, jobs ▼, name ▲; `O` reverses."""
    keys = {
        "gpu_h": lambda r: (-r.gpu_h, -r.billing_h, r.key),
        "su": (lambda r: (-r.billing_h, -r.gpu_h, r.key)) if usage_only else (lambda r: (-r.charged_su, -r.gpu_h, r.key)),
        "cpu_h": lambda r: (-r.cpu_h, -r.gpu_h, r.key),
        "jobs": lambda r: (-r.jobs, -r.gpu_h, r.key),
        "name": lambda r: (r.key,),
    }
    out = sorted(rows, key=keys[sort])
    return out[::-1] if reverse else out


def sort_label(sort: SortKey, reverse: bool, *, usage_only: bool = False) -> str:
    """`GPU-h ▼`, `SU ▼` (`BILL-h ▼` on a usage-only cluster), `name ▲`; reversed: the other arrow."""
    label = {"gpu_h": "GPU-h", "su": "BILL-h" if usage_only else "SU", "cpu_h": "CPU-h", "jobs": "jobs",
             "name": "name"}[sort]
    down = sort != "name"
    return f"{label} {'▼' if down != reverse else '▲'}"


def _section_label(section: str | None) -> str:
    name = (section or "").removeprefix("slurm.")
    for prefix, label in (("acct_share", "sshare"), ("acct_sreport", "sreport"), ("acct_jobs", "sacct"),
                          ("acct_sbank", "sbank")):
        if name.startswith(prefix):
            return label
    return ""


def error_lines(errors: Sequence[HostError], now: datetime, width: int, *, limit: int = 3) -> list[Text]:
    """One red line per failing section, worded from the classified error (never raw stderr): `sreport: slurmdbd
    unreachable · 12m ago · r retries`, `sacct refused the date range · …`. Repeats (one per sacct chunk) once."""
    seen: list[str] = []
    for e in errors:
        label = _section_label(e.section)
        text = e.summary if not label or label in e.summary or e.summary.startswith(label) else f"{label}: {e.summary}"
        if text not in seen:
            seen.append(text)
    out: list[Text] = []
    for i, text in enumerate(seen[:limit]):
        at = next((e.at for e in errors if e.at is not None), None)
        line = Text("✖ " + text, style=theme.error())
        if at is not None:
            line.append(f"{_SEP}{humanize.ago(at, now)}", style=theme.muted())
        line.append(f"{_SEP}r retries" + (f" (+{len(seen) - limit} more)" if i == limit - 1 < len(seen) - 1 else ""),
                    style=theme.muted())
        out.append(render.fit(line, width))
    return out


def weights_text(raw: str) -> str:
    """`cpu=1,gres/gpu=32` -> `cpu=1 · gpu=32` (the gres prefix is noise here)."""
    return _SEP.join(w.strip().replace("gres/", "") for w in raw.split(",") if w.strip())


def _qos_list(items: Sequence[tuple[str, float]]) -> str:
    return ", ".join(f"{q} {costs.fmt_factor(f)}" for q, f in items)


def partition_lines(caps: Capabilities | None, cells: Sequence[Any], width: int, *, usage_only: bool,
                    narrow: bool = False) -> list[Text]:
    """Charged and free partitions (§5.2.11 rule 5): `charged  gpu  cpu=1 · gpu=32 per hour  (qos normal ×1, high
    ×1)` and `free     free-gpu · free-gpuxl  (AllowQos low ×0, guest ×0: billing-h are recorded but not charged)`;
    a usage-only cluster where every QOS counts ×1: `partitions  gpu-preempt · labpi (qos ×1 everywhere)`. Narrow:
    one line `charged gpu · free free-gpu, free-gpuxl (qos low ×0)`. [] without charging rules."""
    if caps is None or not caps.partition_charges:
        return []
    classes = costs.classify_partitions(caps, cells)
    charged = [c for c in classes if c.charged]
    free = [c for c in classes if not c.charged]
    factors = set(caps.qos_usage_factor.values())
    if usage_only and not free and factors <= {1.0}:
        names = _SEP.join(c.partition for c in classes)
        line = Text("partitions  ", style=theme.label())
        line.append(names)
        line.append(" (qos ×1 everywhere)" if factors else " (QOS factors unknown)", style=theme.muted())
        return [render.fit(line, width)]
    free_qos = list(dict.fromkeys(q for c in free for q, _f in c.qos))        # AllowQos order
    if narrow:
        line = Text()
        if charged:
            line.append("charged ", style=theme.label())
            line.append(", ".join(c.partition for c in charged))
        if free:
            if line:
                line.append(_SEP, style=theme.dim())
            line.append("free ", style=theme.label())
            line.append(", ".join(c.partition for c in free), style=theme.ok())
            if free_qos:
                line.append(f" (qos {', '.join(q + ' ×0' for q in free_qos)})", style=theme.muted())
        return [render.fit(line, width)] if line else []
    out: list[Text] = []
    if charged:
        line = Text("charged  ", style=theme.label())
        line.append(_SEP.join(c.partition for c in charged))
        weights = {c.weights for c in charged}
        if len(weights) == 1 and (w := next(iter(weights))):
            line.append(f"  {weights_text(w)} per hour", style=theme.muted())
        elif len(weights) == 1:
            line.append("  billing = CPUs (no weights)", style=theme.muted())
        paid = sorted({q for c in charged for q in c.qos})
        if paid:
            line.append(f"  (qos {_qos_list(paid)})", style=theme.muted())
        out.append(render.fit(line, width))
    if free:
        line = Text("free     ", style=theme.label())
        line.append(_SEP.join(c.partition for c in free), style=theme.ok())
        allow = [c for c in free if c.basis == "allow_qos"]
        basis = "AllowQos" if allow else "observed: qos"
        line.append(f"  ({basis} {', '.join(q + ' ×0' for q in free_qos)}: billing-h are recorded but not charged)",
                    style=theme.muted())
        out.append(render.fit(line, width))
    return out


def _sum(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def account_summary_line(summary: AccountSummary | None, width: int) -> Text:
    """The cluster host detail's `account` line, from `snapshot.account` only (no probe): `account  pilab_02
    298/97,885 SU · pilab_02_gpu 0/97 · alice 0/1,000 · 30d 27,759 CPU-h 3,259 GPU-h · $ more`; usage only:
    `account  usage only (no SU limits) · 30d 171,881 CPU-h · 15,434 GPU-h · recent 5,859 billing-h (decays) · $ more`;
    never read: `account  not read yet (every 30 min; $ reads it now)`. An account at ≥ 80 % of its limit is
    warning-coloured (≥ 95 %: error); more than four accounts end in `…+N`. Not cropped: the Capabilities box wraps."""
    out = Text("account  ", style=theme.label())
    if summary is None:
        out.append("not read yet (every 30 min; $ reads it now)", style=theme.dim())
        return out
    bits: list[Text] = []
    if summary.mode == "limits":
        limited = [b for b in summary.balances if b.limit_su is not None]
        for i, b in enumerate(limited[:4]):
            pct = b.used_su / b.limit_su * 100 if b.used_su is not None and b.limit_su else None
            style = theme.level_style(pct, theme.DEFAULT_MEM_THRESHOLDS) if pct is not None and pct >= 80 else ""
            bits.append(Text(f"{b.account} {costs.fmt_su(b.used_su)}/{costs.fmt_su(b.limit_su)}"
                             + (" SU" if i == 0 else ""), style=style))
        if len(limited) > 4:
            bits[-1].append(f" …+{len(limited) - 4}", style=theme.muted())
    else:
        bits.append(Text("usage only (no SU limits)", style=theme.muted()))
    if summary.usage:
        cpu = _sum([u.cpu_h for u in summary.usage])
        gpu = _sum([u.gpu_h for u in summary.usage])
        sep = " " if summary.mode == "limits" else _SEP
        bits.append(Text(f"{summary.window} {costs.fmt_hours(cpu)} CPU-h{sep}{costs.fmt_hours(gpu)} GPU-h"))
    if summary.mode == "usage_only":
        recent = _sum([b.my_used_su for b in summary.balances])     # the user's own sshare rows, like `30d me`
        if recent is not None:
            decays = " (decays)" if summary.half_life_s else ""
            bits.append(Text(f"recent {costs.fmt_su(recent)} billing-h{decays}", style=theme.muted()))
    if summary.errors:
        bits.append(Text(f"! {summary.errors[0].summary}", style=theme.warn()))
    bits.append(Text("$ more", style=theme.dim()))
    out.append_text(Text(_SEP, style=theme.dim()).join(bits))
    return out


# ---------------------------------------------------------------- one box
@dataclass(frozen=True, slots=True)
class BoxData:
    """Everything one cluster's box shows (assembled by the screen from the store and the service)."""
    host: str
    caps: Capabilities | None
    report: AccountReport | None              # the report of the requested window (or its 30d fallback)
    summary: AccountSummary | None            # report.summary, else the snapshot's periodic summary
    window: UsageWindow                       # requested
    loading: bool = False
    off: bool = False                         # `account_usage = false` in config
    stale_at: datetime | None = None          # host stale / retrying: the data's time ("as of 2h ago")
    has_period: bool = False                  # the host can answer "period" (its fallback is "not known yet")
    fetch_error: HostError | None = None      # the fetch itself failed (never zeros)

    @property
    def usage_only(self) -> bool:
        """No SU worth naming here: the summary's mode once read, else the charging rules (`cost.usage_only`)."""
        return costs.usage_only(self.caps, self.summary.mode if self.summary is not None else None)

    @property
    def errors(self) -> tuple[HostError, ...]:
        found = self.report.errors if self.report is not None else self.summary.errors if self.summary else ()
        return (*found, self.fetch_error) if self.fetch_error is not None else found

    def accounts(self) -> list[str]:
        s = self.summary
        if s is None:
            return []
        return sorted({b.account for b in s.balances} | {u.account for u in s.usage})

    def use(self, scope: Scope) -> dict[str, AccountUse] | None:
        """The MY (or whole-account) usage per account in the requested window; None = not known for it."""
        if scope == "account":
            return {u.account: u for u in self.report.account_totals} if self.report is not None else None
        if self.report is not None:
            return {u.account: u for u in self.report.summary.usage}
        s = self.summary
        if s is None or s.window != self.window:
            return None
        return {u.account: u for u in s.usage}


@dataclass(frozen=True, slots=True)
class ViewOpts:
    """Screen-wide settings every box follows."""
    group: GroupBy = "experiment"
    sort: SortKey = "gpu_h"
    reverse: bool = False
    scope: Scope = "me"
    narrow: bool = False
    thresholds: tuple[float, float] = theme.DEFAULT_MEM_THRESHOLDS


@dataclass(frozen=True, slots=True)
class Section:
    """One part of a drawn box: an optional title in its top border (┌ or ├), a right-hand note, its lines."""
    title: Text | None
    lines: list[Text]
    right: Text | None = None


def _ago(at: datetime | None, now: datetime) -> str:
    return humanize.ago(at, now, dash="never")


def box_title(d: BoxData, now: datetime, narrow: bool) -> Text:
    """`lyra · limits from sshare · sbank 3h ago · usage 2m ago`; `k2 · usage only: no SU limits on this cluster ·
    billing = CPUs (no weights) · usage 2m ago`; narrow `lyra · sbank 3h ago · 2m ago`. Then `fetching…` and, for a
    stale host, `as of 2h ago`."""
    out = Text(d.host, style="bold")
    s = d.summary
    parts: list[tuple[str, str]] = []
    if d.off:
        parts.append(("account usage is off in config", theme.muted()))
    elif s is not None and (s.balances or s.usage):
        if d.usage_only:
            parts.append(("usage only (no SU limits)" if narrow else "usage only: no SU limits on this cluster",
                          theme.muted()))
            if not narrow and d.caps is not None and d.caps.partition_charges and not costs.has_weights(d.caps):
                parts.append(("billing = CPUs (no weights)", theme.muted()))
        elif not narrow:
            sources = sorted({b.limit_source for b in s.balances if b.limit_source})
            if sources:
                parts.append((f"limits from {' + '.join(sources)}", theme.muted()))
        if s.sbank_at is not None:
            parts.append((f"sbank {_ago(s.sbank_at, now)}", theme.muted()))
        if d.report is not None:
            at = d.report.summary.usage_at or d.report.fetched_at
            parts.append((("" if narrow else "usage ") + _ago(at, now), theme.muted()))
        else:
            parts.append((f"summary {_ago(s.usage_at or s.read_at, now)}", theme.muted()))
    if d.loading:
        parts.append(("fetching…", theme.accent()))
    if d.stale_at is not None:
        parts.append((f"as of {_ago(d.stale_at, now)}", theme.warn()))
    for text, style in parts:
        out.append(_SEP, style=theme.dim())
        out.append(text, style=style)
    return out


def _basis_text(d: BoxData, scope: Scope) -> str:
    """What "used" and "MY" mean on this cluster (wide layout only)."""
    s = d.summary
    bits: list[str] = []
    if s is not None and s.balances and not d.usage_only:
        basis = s.balances[0].used_basis
        if basis == "since_reset":
            bits.append("used = since the last reset (no decay)")
        elif basis == "decayed":
            reset = f" · resets {s.reset_period.lower()}" if s.reset_period else ""
            bits.append(f"used = recent (decays, half-life {humanize.age(s.half_life_s)}{reset}), not a total")
        else:
            bits.append("used = sshare's usage (decay unknown)")
    source = (d.report.summary.usage[0].source if d.report is not None and d.report.summary.usage else "sreport")
    if scope == "account":
        bits.append("ACCT = the whole account in the window (sreport) · u: just you")
    elif source == "sacct":
        bits.append("MY = your job records in the window (sreport not installed) · u: whole account")
    else:
        bits.append("MY = your usage in the window (sreport, ≤ 1 h behind) · u: whole account")
    return _SEP.join(bits)


def _cell(text: str | Text, width: int, align: str) -> Text:
    t = text if isinstance(text, Text) else Text(text)
    if t.cell_len > width:
        t = t.copy()
        t.truncate(width, overflow="ellipsis")
    pad = " " * (width - t.cell_len)
    return Text.assemble(pad, t) if align == "r" else Text.assemble(t, pad)


def _row(cols: Sequence[tuple[str, str, int, str]], cells: Mapping[str, str | Text], gutter: str | Text = "  ") -> Text:
    """One table line: the 2-cell gutter (`▶ ` on the cursor row), then the columns joined by two spaces."""
    out = Text.assemble(gutter if isinstance(gutter, Text) else Text(gutter))
    for i, (key, _label, w, align) in enumerate(cols):
        if i:
            out.append("  ")
        out.append_text(_cell(cells.get(key, ""), w, align))
    return out


def _fit(cols: list[tuple[str, str, int, str]], width: int, drop: Sequence[str],
         flex: str) -> list[tuple[str, str, int, str]]:
    """`job_table.fit_columns` over (key, label, width, align) for `width` cells after the gutter; the 2-space gap
    before every column but the first counts."""
    meta = {k: (lab, a) for k, lab, _w, a in cols}
    fitted = fit_columns([(k, lab, w + 2) for k, lab, w, _a in cols], width + 2, drop=drop, flex=flex)
    return [(k, lab, w - 2, meta[k][1]) for k, lab, w in fitted]


def account_lines(d: BoxData, o: ViewOpts, width: int, cursor: int | None) -> list[Text]:
    """The account table (mockups O / O2). Limits: ACCOUNT · USED SU · LIMIT SU · AVAIL SU · USED (bar, ≥ 80 %
    warning, ≥ 95 % error) · REALLOC · MY CPU-h / GPU-h / BILL-h; at 80 columns the bar, MY BILL-h, MY CPU-h, then
    REALLOC go. Usage only: ACCOUNT · MY CPU-h / GPU-h / BILL-h · MY RECENT (decays, half-life 7d: the user's own
    sshare row). `u` swaps MY for the whole account (ACCT, the account row). Unknown values are `—`, never 0."""
    s = d.summary
    names = d.accounts()
    if s is None or not names:
        return []
    bal = {b.account: b for b in s.balances}
    use = d.use(o.scope)
    who = "ACCT" if o.scope == "account" else "MY"
    mine = [(k, f"{who} {u}", max(10, len(who) + 7), "r") for k, u in (("cpu", "CPU-h"), ("gpu", "GPU-h"),
                                                                       ("bill", "BILL-h"))]
    acc_w = max(12, min(20, max(len(n) for n in names)))
    room = width - 2                               # the gutter
    if d.usage_only:
        half = humanize.age(s.half_life_s) if s.half_life_s else None
        recent = ("RECENT (decays)" if o.narrow else f"RECENT (decays, half-life {half})") if half else "RECENT"
        recent = f"{who} {recent}"
        cols = [("acct", "ACCOUNT", acc_w, "l"), *mine, ("recent", recent, 20, "l")]
        cols = _fit(cols, room, drop=("recent", "bill", "cpu"), flex="recent")
    else:
        cols = [("acct", "ACCOUNT", acc_w, "l"), ("used", "USED SU", 9, "r"), ("limit", "LIMIT SU", 9, "r"),
                ("avail", "AVAIL SU", 9, "r"), ("bar", "USED", 18, "l"), ("realloc", "REALLOC", 10, "l"), *mine]
        cols = _fit(cols, room, drop=("bar", "bill", "cpu", "realloc"), flex="acct")
    out = [render.fit(_row(cols, {k: Text(lab, style=theme.label()) for k, lab, _w, _a in cols}), width)]
    for i, name in enumerate(names):
        b = bal.get(name)
        u = use.get(name) if use is not None else None
        cells: dict[str, str | Text] = {"acct": name}
        if u is None:                              # not read for this window (a period account without a start)
            cells.update(cpu=Text(humanize.NONE, style=theme.dim()), gpu=Text(humanize.NONE, style=theme.dim()),
                         bill=Text(humanize.NONE, style=theme.dim()))
        else:
            cells.update(cpu=costs.fmt_hours(u.cpu_h), gpu=costs.fmt_hours(u.gpu_h), bill=costs.fmt_su(u.billing_h))
        if d.usage_only:
            amount = (b.used_su if o.scope == "account" else b.my_used_su) if b is not None else None
            if amount is not None:                 # MY: the user's own sshare row; `u` (ACCT): the account row
                cells["recent"] = Text(f"{costs.fmt_su(amount)} billing-h", style=theme.muted())
            else:
                cells["recent"] = Text(humanize.NONE, style=theme.dim())
        else:
            cells["used"] = costs.fmt_su(b.used_su if b else None)
            cells["limit"] = costs.fmt_su(b.limit_su if b else None)
            cells["avail"] = costs.fmt_su(b.available_su if b else None)
            pct = (b.used_su / b.limit_su * 100 if b is not None and b.used_su is not None and b.limit_su
                   else None)
            if pct is not None:
                bar = render.gpu_bar(pct, 12, o.thresholds)
                bar.append(costs.fmt_pct(pct).rjust(6), style=theme.level_style(pct, o.thresholds)
                           if pct >= o.thresholds[0] else "")
                cells["bar"] = bar
                if pct >= o.thresholds[0]:
                    cells["acct"] = Text(name, style=theme.level_style(pct, o.thresholds))
            if b is not None and b.next_realloc is not None:
                cells["realloc"] = b.next_realloc.isoformat()
            elif b is not None and b.realloc_known:
                cells["realloc"] = Text("— none set", style=theme.dim())
            else:
                cells["realloc"] = Text(humanize.NONE, style=theme.dim())
        out.append(render.fit(_row(cols, cells, _gutter(cursor == i)), width))
    return out


def _gutter(on: bool) -> Text:
    return Text("▶ ", style=f"bold {theme.ACCENT}" if not theme.no_color() else "bold") if on else Text("  ")


def _models(row: RollupRow) -> str:
    return _SEP.join(f"{humanize.gpu_label(m) if m else 'GPU'} {costs.fmt_hours(h)}"
                     for m, h in row.gpu_h_by_model.items()) or humanize.NONE


def rollup_rows(d: BoxData, o: ViewOpts) -> list[RollupRow]:
    if d.report is None:
        return []
    return sort_rollup(group(d.report.cells, o.group), o.sort, o.reverse, usage_only=d.usage_only)


def _key_label(key: str, by: GroupBy) -> str:
    if key:
        return humanize.gpu_label(key) if by == "gpu" else key
    return {"gpu": "no GPU / model unknown", "account": "(no account)", "partition": "(no partition)"}.get(
        by, "(no name)")


def rollup_section(d: BoxData, o: ViewOpts, width: int, rows: int, cursor: int | None,
                   offset: int) -> Section:
    """"Where it went" (mockup O): `≤ rows` of the sorted rollup from `offset`, the cursor row marked `▶`, then
    `… N more (↓)`. Columns: the group key · JOBS · RUN (running at report time) · CPU-h · GPU-h · BY MODEL · SU
    (BILL-h on a usage-only cluster) · ACCOUNT (when more than one); narrow: key · GPU-h · SU."""
    unit = "BILL-h" if d.usage_only else "SU"
    report = d.report
    window = report.window if report is not None else d.window
    title = Text("Where it went" if not o.narrow else "", style="bold")
    bits = [window, f"by {GROUP_LABELS[o.group]}"]
    if report is not None and not o.narrow:
        bits.append(f"{report.jobs_seen:,} jobs")
    bits.append(sort_label(o.sort, o.reverse, usage_only=d.usage_only))
    for i, b in enumerate(bits):
        if i or title.plain:
            title.append(_SEP, style=theme.dim())
        title.append(b, style=theme.muted())
    if report is not None and report.truncated:
        title.append(f"{_SEP}lower bound: a chunk was cut or not read", style=theme.warn())
    right = None if o.narrow else Text("z group · o sort", style=theme.dim())
    if report is None:
        what = "reading your job records …" if d.loading else "not read yet (r reads it)"
        return Section(title, [Text(f"  {what}", style=theme.dim())], right)
    items = rollup_rows(d, o)
    if not items:
        failed = any(_section_label(e.section) == "sacct" for e in report.errors)
        what = "your job records could not be read (see above)" if failed else "no jobs of yours in this window"
        return Section(title, [Text(f"  {what}", style=theme.dim())], right)
    multi = len({a for r in items for a in r.accounts}) > 1
    key_label = GROUP_LABELS[o.group].upper()
    if o.narrow:
        cols = [("key", key_label, 30, "l"), ("gpu", "GPU-h", 9, "r"), ("su", unit, 9, "r")]
        cols = _fit(cols, width - 2, drop=(), flex="key")
    else:
        cols = [("key", key_label, 30, "l"), ("jobs", "JOBS", 5, "r"), ("run", "RUN", 4, "r"),
                ("cpu", "CPU-h", 8, "r"), ("gpu", "GPU-h", 8, "r"), ("models", "BY MODEL", 26, "l"),
                ("su", unit, 8, "r"), *([("acct", "ACCOUNT", 10, "l")] if multi else [])]
        cols = _fit(cols, width - 2, drop=("models", "acct", "run", "jobs", "cpu"), flex="key")
    lines = [render.fit(_row(cols, {k: Text(lab, style=theme.label()) for k, lab, _w, _a in cols}), width)]
    shown = items[offset:offset + rows]
    for i, r in enumerate(shown, start=offset):
        amount = r.billing_h if d.usage_only else r.charged_su
        su = Text(costs.fmt_su(amount) + ("?" if r.factor_unknown and not d.usage_only else ""),
                  style=theme.dim() if amount == 0 and r.billing_h > 0 else "")
        cells: dict[str, str | Text] = {
            "key": _key_label(r.key, o.group), "jobs": f"{r.jobs:,}", "run": str(r.running) if r.running else "0",
            "cpu": costs.fmt_hours(r.cpu_h),
            "gpu": costs.fmt_hours(r.gpu_h) if r.gpu_h else Text(humanize.NONE, style=theme.dim()),
            "models": Text(_models(r), style=theme.muted()) if r.gpu_h else Text(humanize.NONE, style=theme.dim()),
            "su": su, "acct": ", ".join(r.accounts)}
        lines.append(render.fit(_row(cols, cells, _gutter(cursor == i)), width))
    below = len(items) - offset - len(shown)
    if below > 0 or offset > 0:
        more = Text("  … ")
        bits = ([f"{offset} above"] if offset else []) + ([f"{below} more" + ("" if o.narrow else " (↓)")]
                                                         if below else [])
        more.append(_SEP.join(bits))
        lines.append(render.fit(Text(more.plain, style=theme.dim()), width))
    return Section(title, lines, right)


def box_sections(d: BoxData, o: ViewOpts, now: datetime, width: int, *, rows: int, cursor: int | None,
                 offset: int) -> list[Section]:
    """A full box: the period fallback note, errors, notes, the account table, the basis line (wide), the partitions,
    then "where it went" under its own divider. `width`: the cells of a line (`draw` frames them)."""
    title = box_title(d, now, o.narrow)
    lines: list[Text] = []
    if d.off:
        lines.append(Text(f"  account usage is off for {d.host} in config", style=theme.dim()))
        return [Section(title, lines)]
    if d.window == "period" and d.report is not None and d.report.window != "period":
        what = ("period not known yet: showing 30d (r reads again)" if d.has_period
                else f"no period window on {d.host} (no reallocation date or usage reset): showing 30d")
        lines.append(Text("  " + what, style=theme.warn() if d.has_period else theme.dim()))
    lines += [Text.assemble("  ", x) for x in error_lines(d.errors, now, width - 2)]
    s = d.summary
    for note in (s.notes if s is not None else ()):
        lines.append(render.fit(Text("  " + note, style=theme.dim()), width))
    table = account_lines(d, o, width, cursor)
    if not table and s is not None and not s.notes and not d.errors:
        lines.append(Text("  no account data yet" + (" — reading …" if d.loading else ""), style=theme.dim()))
    elif not table and s is None:
        lines.append(Text("  reading account usage …" if d.loading else "  not read yet (r reads it)",
                          style=theme.dim()))
    lines += table
    if table and not o.narrow:
        lines.append(render.fit(Text("  " + _basis_text(d, o.scope), style=theme.dim()), width))
    cells = d.report.cells if d.report is not None else ()
    lines += [Text.assemble("  ", x) for x in partition_lines(d.caps, cells, width - 2, usage_only=d.usage_only,
                                                                 narrow=o.narrow)]
    out = [Section(title, lines)]
    if s is None or not (s.balances or s.usage or (d.report is not None and d.report.cells)):
        if d.report is None or not (d.report.cells or d.report.jobs_seen):
            return out
    n_acc = len(d.accounts())
    out.append(rollup_section(d, o, width, rows, cursor - n_acc if cursor is not None else None, offset))
    return out


def compact_sections(d: BoxData, o: ViewOpts, now: datetime, width: int) -> list[Section]:
    """A collapsed box (mockup O2's k2): its title and two lines — the balances or my usage in one line, then the top
    row of the rollup (or the first error)."""
    title = box_title(d, now, o.narrow)
    s = d.summary
    first = Text("  ")
    if d.off:
        first.append(f"account usage is off for {d.host} in config", style=theme.dim())
    elif s is None or not (s.balances or s.usage):
        first.append("reading …" if d.loading else "not read yet", style=theme.dim())
    elif not d.usage_only:
        for i, b in enumerate(x for x in s.balances if x.limit_su is not None):
            pct = _used_pct(b)                     # ≥ 80 % warning, ≥ 95 % error, as on the host detail
            if i:
                first.append(_SEP, style=theme.dim())
            first.append(f"{b.account} {costs.fmt_su(b.used_su)}/{costs.fmt_su(b.limit_su)}" + (" SU" if not i else ""),
                         style=theme.level_style(pct, o.thresholds) if pct is not None and pct >= o.thresholds[0]
                         else "")
    else:
        use = d.use(o.scope) or {}
        name = d.accounts()[0]
        u = use.get(name)
        first.append(f"{name}   ")
        if u is None:
            first.append(f"{humanize.NONE} (window not read yet)", style=theme.dim())
        else:
            first.append(f"{costs.fmt_hours(u.cpu_h)} CPU-h · {costs.fmt_hours(u.gpu_h)} GPU-h · "
                         f"{costs.fmt_su(u.billing_h)} billing-h")
            if d.caps is not None and not costs.has_weights(d.caps):
                first.append(" (= CPU-h)", style=theme.muted())
        if len(d.accounts()) > 1:
            first.append(f" +{len(d.accounts()) - 1} accounts", style=theme.muted())
    errors = error_lines(d.errors, now, width - 2, limit=1)
    if errors:
        second = Text.assemble("  ", errors[0])
    else:
        top = rollup_rows(d, ViewOpts(group=o.group))
        charged = [r for r in top if r.charged_su] if not d.usage_only else []
        second = Text("  ")
        if charged:                                # a limits cluster: what spent the SU (free GPU-h spend none)
            r = max(charged, key=lambda x: x.charged_su or 0.0)
            second.append("top SU: ", style=theme.label())
            key = humanize.truncate(_key_label(r.key, o.group), max(10, width - 24))
            second.append(f"{key} {costs.fmt_su(r.charged_su)} SU")
        elif top:
            r = top[0]
            second.append("top: ", style=theme.label())
            key = humanize.truncate(_key_label(r.key, o.group), max(10, width - 24))
            amount = (f"{costs.fmt_hours(r.gpu_h)} GPU-h" if r.gpu_h
                      else f"{costs.fmt_su(r.billing_h)} billing-h")
            second.append(f"{key} {amount}" + (" (0 SU)" if not d.usage_only else ""))
        else:
            second.append("where it went: " + ("reading …" if d.loading else "not read yet"), style=theme.dim())
    return [Section(title, [render.fit(first, width), render.fit(second, width)])]


def _used_pct(b: AccountBalance) -> float | None:
    return b.used_su / b.limit_su * 100 if b.used_su is not None and b.limit_su else None


def tightest(summary: AccountSummary | None) -> AccountBalance | None:
    """The account with an SU limit closest to it (highest used %), None without one."""
    limited = [b for b in (summary.balances if summary is not None else ()) if b.limit_su is not None]
    return max(limited, key=lambda b: (_used_pct(b) or 0.0, -(b.available_su or 0.0))) if limited else None


def fold_text(d: BoxData) -> str:
    """A box with no room even for its collapsed form: one mention under the boxes (a limits cluster names its
    tightest account)."""
    if d.off:
        return f"{d.host}: account usage off"
    if d.summary is None or not d.summary.balances:
        return f"{d.host}: not read yet"
    if d.usage_only:
        return f"{d.host}: usage only"
    b = tightest(d.summary)
    if b is None or b.available_su is None:
        return f"{d.host}: limits"
    return f"{d.host}: {b.account} {costs.fmt_su(b.available_su)} SU left"


def layout_rank(datas: Sequence[BoxData]) -> list[int]:
    """The order boxes get their room in when the screen is short (§7.17), after the focused one: clusters with SU
    limits first (the most used account first), then the others in config order. The boxes stay drawn in config
    order; only who stays full, collapses or folds follows this."""
    def key(i: int) -> tuple[int, float, int]:
        d = datas[i]
        if d.off or d.summary is None or d.usage_only:
            return (1, 0.0, i)
        b = tightest(d.summary)
        return (0, -((_used_pct(b) if b is not None else None) or 0.0), i)
    return sorted(range(len(datas)), key=key)


def draw(sections: Sequence[Section], width: int, *, focused: bool) -> list[Text]:
    """Frame the sections: `┌ title ─…─┐`, `├ title ─… right ─┤` between them, `│line │` (lines are `width - 3`
    cells: the gutter is theirs, one blank before the right border), `└─…─┘`."""
    border = (f"bold {theme.ACCENT}" if not theme.no_color() else "bold") if focused else (theme.BORDER or "dim")
    inner = width - 2
    out: list[Text] = []
    for i, sec in enumerate(sections):
        left, right_corner = ("┌", "┐") if i == 0 else ("├", "┤")
        top = Text(left + " ", style=border)
        title = (sec.title or Text()).copy()
        right = sec.right.copy() if sec.right is not None else Text()
        room = inner - 2 - (right.cell_len + 3 if right.cell_len else 0)
        if title.cell_len > room:
            title.truncate(max(0, room), overflow="ellipsis")
        top.append_text(title)
        top.append(" ", style=border)
        fill = inner - 2 - title.cell_len - (right.cell_len + 3 if right.cell_len else 0)
        top.append("─" * max(0, fill), style=border)
        if right.cell_len:
            top.append(" ", style=border)
            top.append_text(right)
            top.append(" ─", style=border)
        top.append(right_corner, style=border)
        out.append(render.fit(top, width))
        for line in sec.lines:
            row = Text("│", style=border)
            row.append_text(render.fit(line, inner - 1))
            row.append(" │", style=border)
            out.append(row)
    out.append(Text("└" + "─" * inner + "┘", style=border))
    return out


@dataclass(frozen=True, slots=True)
class Placement:
    mode: Literal["full", "compact", "fold"]
    rows: int = 0                                  # rollup rows of a full box


def plan_layout(heights: Sequence[Mapping[int, int]], totals: Sequence[int], budget: int, focus: int,
                rank: Sequence[int] | None = None) -> list[Placement]:
    """Where each box goes within `budget` rows (§7.17: boxes beyond the screen height collapse, the focused one
    last). `heights[i]` maps rollup rows -> the full box's height; `totals[i]` its rollup rows. The focused box first,
    then the others in `rank` order (`layout_rank`; default config order): full with ≥ MIN_ROLLUP rows if it fits,
    else collapsed (COMPACT_ROWS + 2 borders), else folded into the line under the boxes; the rows left over go to the
    rollups, the focused one first, up to MAX_ROLLUP each. The focused box is never folded (it scrolls when even it
    does not fit)."""
    n = len(heights)
    rest = [i for i in (rank if rank is not None else range(n)) if i != focus]
    order = [focus, *rest] if n else []
    out: list[Placement] = [Placement("fold")] * n
    left = budget
    for i in order:
        want = min(totals[i], MIN_ROLLUP)
        if heights[i][want] <= left or i == focus:
            out[i] = Placement("full", want)
            left -= heights[i][want]
        elif COMPACT_ROWS <= left:
            out[i] = Placement("compact")
            left -= COMPACT_ROWS
    for i in order:
        p = out[i]
        if p.mode != "full":
            continue
        rows = p.rows
        while rows < min(totals[i], MAX_ROLLUP) and heights[i][rows + 1] - heights[i][rows] <= left:
            left -= heights[i][rows + 1] - heights[i][rows]
            rows += 1
        out[i] = Placement("full", rows)
    return out


# ---------------------------------------------------------------- screen
class AccountHeader(FleetHeader):
    """`RunFleet ▏ALL▕ › Account      window ◀ 30d ▶ (since 09-30)      ⟳ fetched 2m ago ⚙` (◀ / ▶ clickable)."""

    def build(self, width: int) -> Text:
        screen = self.screen
        left = Text(" ")
        left.append("RunFleet", style=f"bold {theme.ACCENT}" if not theme.no_color() else "bold")
        left.stylize(_click("app.home"), 1)
        left.append(" ")
        left.append_text(self._scope_chip(False))
        left.append(" › ", style=theme.dim())
        left.append("Account")
        if not isinstance(screen, AccountScreen):
            return self._join(left, Text(), width)
        narrow = width < NARROW
        middle = Text("" if narrow else "window ")
        middle.append("◀", style=Style.combine([Style.parse(theme.accent()), _click("screen.window(-1)")]))
        middle.append(f" {screen.window} ", style="bold")
        middle.append("▶", style=Style.combine([Style.parse(theme.accent()), _click("screen.window(1)")]))
        if not narrow:
            middle.append(" " + screen.window_meaning(), style=theme.muted())
        right = self._badges(True)
        fetched = screen.fetched_age()
        right.append(f"⟳ {'' if narrow else 'fetched '}{fetched}", style=theme.muted())
        right.append("  ")
        right.append_text(self._gear())
        room = width - left.cell_len - right.cell_len - 2
        if middle.cell_len > room:
            middle.truncate(max(0, room), overflow="ellipsis")
        start = max(left.cell_len + 2, (width - middle.cell_len) // 2)
        gap = max(2, start - left.cell_len)
        return self._join(Text.assemble(left, " " * gap, middle), right, width)


class _Body(VerticalScroll, can_focus=False):
    """The boxes scroll (a tiny terminal) but never take the keys: the screen's bindings move the cursor."""


class AccountScreen(FleetScreen):
    BINDINGS = keymap.bindings("account")
    HELP_CONTEXTS = ("account", "global")
    HELP_TITLE = keymap.CONTEXT_TITLES["account"]

    def __init__(self, first: str | None = None) -> None:
        super().__init__()
        self.first_host = first                   # `$` from a cluster's host detail: that cluster first and focused
        self.window: UsageWindow = "30d"
        self.opts = ViewOpts()
        self.hosts: list[str] = []                # Slurm hosts in scope (boxes), first_host first
        self.others: list[str] = []               # workstations / unknown hosts in scope (the line under the boxes)
        self.focus_idx = 0
        self.box_cursor: dict[str, int] = {}          # host -> row in its box (accounts, then rollup)
        self.box_offset: dict[str, int] = {}          # host -> first rollup row shown
        self._latest: dict[tuple[str, str], AccountReport] = {}   # (host, window asked) -> this screen's fetch
        self._failed: dict[tuple[str, str], HostError] = {}       # a fetch that raised (never shown as zeros)
        self._loading: set[tuple[str, str]] = set()
        self._forced: dict[tuple[str, str], datetime] = {}       # (host, window) -> the last `r` that asked
        self._placements: list[Placement] = []
        self.focus_chosen = False                 # `$` from a host detail or a focus move: paint() keeps the focus

    # ------------------------------------------------------------------ compose / mount
    def compose(self) -> ComposeResult:
        yield AccountHeader(crumbs=["Account"], id="header")
        with _Body(id="acct-body"):
            yield Static("", id="acct-boxes")
        yield HintBar(self.help_contexts(), id="hints")

    def on_mount(self) -> None:
        general = self.rf.service.config.general
        saved = self.rf.ui_state.get("account_window")
        self.window = saved if saved in USAGE_WINDOWS else general.usage_window
        self._set_hosts()
        self.paint()
        self._fetch_all()

    def _set_hosts(self) -> None:
        store = self.rf.store.hosts
        scope = [h for h in self.rf.scope_hosts() if h in store]
        if self.first_host is not None and self.first_host in store and self.first_host not in scope:
            scope.insert(0, self.first_host)
        hosts = [h for h in scope if is_cluster(store[h])]
        if self.first_host in hosts:
            hosts.remove(self.first_host)
            hosts.insert(0, self.first_host)
        focused = self.hosts[self.focus_idx] if self.hosts and self.focus_idx < len(self.hosts) else None
        self.hosts = hosts
        self.others = [h for h in scope if h not in hosts]
        self.focus_idx = hosts.index(focused) if focused in hosts else 0
        if self.first_host in hosts:
            self.focus_chosen = True

    # ------------------------------------------------------------------ data
    def _state(self, host: str) -> HostState | None:
        return self.rf.store.hosts.get(host)

    def _report(self, host: str) -> AccountReport | None:
        """This screen's last fetch of (host, window), or the service's cached one when newer (another fetch, the
        agent); a failed fetch's report keeps its errors."""
        mine = self._latest.get((host, self.window))
        try:
            cached = self.rf.service.account_cached(host, self.window)
        except Exception:                      # a cache problem is never the screen's problem
            log.exception("account_cached failed")
            cached = None
        if mine is None:
            return cached
        if cached is not None and cached.fetched_at > mine.fetched_at:
            return cached
        return mine

    def _has_period(self, host: str, summary: AccountSummary | None) -> bool:
        """The host can answer "period": a period report came back as one, or the summary has a reallocation date or
        a usage reset period (the rules of `account.period_starts`, which ui/ may not import)."""
        state = self._state(host)
        if state is None:
            return False
        cached = self._latest.get((host, "period")) or self.rf.service.account_cached(host, "period")
        if cached is not None and cached.window == "period":
            return True
        if summary is None:
            return False
        if summary.reset_period in RESET_PERIODS:
            return True
        realloc = state.config.slurm.allocation_period > 0       # §5.2.11 "Period window": a known reallocation
        return realloc and any(b.next_realloc is not None for b in summary.balances)

    def box_data(self, host: str) -> BoxData:
        state = self._state(host)
        snap = state.snapshot if state is not None else None
        report = self._report(host)
        summary = report.summary if report is not None else (snap.account if snap is not None else None)
        if report is not None and not (report.summary.balances or report.summary.usage) and snap is not None \
                and snap.account is not None and report.errors:
            summary = snap.account                 # a failed first fetch: the periodic summary, not nothing
        now = self.rf.now()
        stale_at = None                            # stale host, and no fresh report of this window to show
        if state is not None and state.stale(now) and (report is None or report.errors):
            stale_at = report.fetched_at if report is not None else summary.read_at if summary else None
        return BoxData(host=host, caps=snap.caps if snap is not None else None, report=report, summary=summary,
                       window=self.window, loading=(host, self.window) in self._loading,
                       off=state is not None and not state.config.slurm.account_usage, stale_at=stale_at,
                       has_period=self._has_period(host, summary), fetch_error=self._failed.get((host, self.window)))

    def _fetch_all(self, max_age_s: float = REPORT_MAX_AGE_S) -> None:
        for host in self.hosts:
            self._fetch(host, max_age_s)

    def _fetch(self, host: str, max_age_s: float = REPORT_MAX_AGE_S) -> None:
        state = self._state(host)
        if state is None or not state.config.slurm.account_usage:
            return
        key = (host, self.window)
        self._loading.add(key)
        self.run_worker(self._fetch_one(host, self.window, max_age_s), group=f"acct-{host}", exclusive=True,
                        exit_on_error=False)

    async def _fetch_one(self, host: str, window: UsageWindow, max_age_s: float) -> None:
        key = (host, window)
        try:
            report = await self.rf.service.account_report(host, window, max_age_s=max_age_s)
        except Exception as e:                 # account_report never raises for remote failures; anything else
            log.exception("%s: account report failed", host)      # is shown as an error, never as zero usage
            self._failed[key] = HostError(ErrorKind.INTERNAL, f"account report failed: {e}"[:80], at=self.rf.now())
        else:
            self._failed.pop(key, None)
            self._latest[key] = report
            if report.window != window:
                self._latest.setdefault((host, report.window), report)
        finally:
            self._loading.discard(key)
        if self.is_mounted:
            self.paint()

    # ------------------------------------------------------------------ rendering
    def _opts(self) -> ViewOpts:
        width = self.size.width or self.app.size.width
        ui = self.rf.service.config.ui
        return ViewOpts(group=self.opts.group, sort=self.opts.sort, reverse=self.opts.reverse,
                        scope=self.opts.scope, narrow=width < NARROW, thresholds=ui.gpu_mem_thresholds)

    def paint(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(AccountHeader).refresh_content()
        width = self.size.width or self.app.size.width
        height = (self.size.height or self.app.size.height) - 2
        now = self.rf.now()
        o = self._opts()
        datas = [self.box_data(h) for h in self.hosts]
        if not self.focus_chosen:                  # until the user moves: the first cluster with SU limits (§7.17)
            self.focus_idx = next((i for i, d in enumerate(datas)
                                   if d.summary is not None and d.summary.mode == "limits"), self.focus_idx)
        inner = width - 3                          # between the borders, less the blank before the right one
        heights: list[dict[int, int]] = []
        totals: list[int] = []
        for d in datas:
            total = len(rollup_rows(d, o))
            totals.append(total)
            base = sum(len(s.lines) + 1 for s in box_sections(d, o, now, inner, rows=0, cursor=None, offset=0)) + 1
            # each rollup row is one line; the `… N more` line goes once every row is shown
            heights.append({r: base + r - (1 if total and r >= total else 0) for r in range(min(total, MAX_ROLLUP) + 1)})
        foot = self._foot(datas, [Placement("full")] * len(datas), o)
        budget = height - (1 if foot.plain.strip() or len(datas) > 1 else 0)
        self._placements = plan_layout(heights, totals, budget, self.focus_idx, layout_rank(datas)) if datas else []
        lines: list[Text] = []
        for i, (d, p) in enumerate(zip(datas, self._placements, strict=True)):
            focused = i == self.focus_idx
            if p.mode == "fold":
                continue
            if p.mode == "compact":
                secs = compact_sections(d, o, now, inner)
            else:
                cur = self._clamp(d, o, p.rows)    # kept for every box; drawn (▶) in the focused one only
                secs = box_sections(d, o, now, inner, rows=p.rows, cursor=cur if focused or len(datas) == 1 else None,
                                    offset=self.box_offset.get(d.host, 0))
            box = draw(secs, width, focused=focused and len(datas) > 1)
            if d.stale_at is not None:
                for line in box[1:-1]:
                    line.stylize("dim")
            lines += box
        foot = self._foot(datas, self._placements, o)
        if foot.plain.strip():
            lines.append(render.fit(foot, width))
        if not datas and not foot.plain.strip():
            lines.append(Text(" no hosts in scope", style=theme.dim()))
        self.query_one("#acct-boxes", Static).update(Text("\n").join(lines))

    def _foot(self, datas: Sequence[BoxData], placements: Sequence[Placement], o: ViewOpts) -> Text:
        """`vega: usage only (collapsed, ↓) · labstation, lynx: workstations, no scheduler accounting`."""
        bits: list[str] = []
        for d, p in zip(datas, placements, strict=True):
            if p.mode == "fold":
                bits.append(fold_text(d) + (" (↓)" if o.narrow else " (collapsed, ↓)"))
        store = self.rf.store.hosts
        unknown = [h for h in self.others if _never_connected(store[h])]
        ws = [h for h in self.others if h not in unknown]
        if ws:
            bits.append(f"{', '.join(ws)}: " + ("no accounting" if o.narrow
                                                 else "workstations, no scheduler accounting"))
        if unknown:
            bits.append(f"{', '.join(unknown)}: not connected yet")
        if not self.hosts and self.others:
            bits.append("no Slurm cluster in scope")
        return Text(" " + _SEP.join(bits), style=theme.dim()) if bits else Text()

    def _clamp(self, d: BoxData, o: ViewOpts, rows: int) -> int:
        """Keep the box's cursor on a row and its rollup window around it."""
        n_acc = len(d.accounts())
        n_roll = len(rollup_rows(d, o))
        cur = max(0, min(self.box_cursor.get(d.host, 0), max(0, n_acc + n_roll - 1)))
        self.box_cursor[d.host] = cur
        off = self.box_offset.get(d.host, 0)
        if rows > 0 and cur >= n_acc:
            r = cur - n_acc
            if r < off:
                off = r
            elif r >= off + rows:
                off = r - rows + 1
        off = max(0, min(off, max(0, n_roll - rows)))
        self.box_offset[d.host] = off
        return cur

    # ------------------------------------------------------------------ header questions
    def window_meaning(self) -> str:
        """`(since 09-30)`; period: what the focused cluster's period means."""
        if self.window != "period":
            return window_since(self.window, self.rf.now())
        focused = self._focused_host()                 # the focused cluster's period, else the first one's
        for host in sorted(self.hosts, key=lambda h: h != focused):
            state = self._state(host)
            text = period_text(self._report(host), state.config.slurm.allocation_period if state is not None else 0)
            if text:
                return f"({text})"
        return "(not known yet)"

    def fetched_age(self) -> str:
        host = self._focused_host()
        report = self._report(host) if host else None
        if host is not None and (host, self.window) in self._loading and report is None:
            return "…"
        return humanize.ago(report.fetched_at, self.rf.now(), dash="never") if report is not None else "never"

    def _focused_host(self) -> str | None:
        return self.hosts[self.focus_idx] if self.hosts and self.focus_idx < len(self.hosts) else None

    # ------------------------------------------------------------------ FleetScreen
    def current_host(self) -> str | None:
        return self._focused_host()

    def visibility(self) -> dict[str, Visibility]:
        return {h: Visibility.VISIBLE for h in self.hosts}

    def refresh_data(self, changes: FleetChanged | None) -> None:
        if not self.is_mounted:
            return
        if changes is not None:
            mine = set(self.hosts) | set(self.others)
            if not any(c.host in mine and (c.account or c.status) for c in changes.changes):
                return
        if changes is None or any(c.status for c in changes.changes):
            before = set(self.hosts)
            self._set_hosts()                      # a host's first probe may make it a cluster
            for host in self.hosts if changes is not None else ():   # a cluster that just connected: its report
                if host not in before:                                 # too (once; the mount fetches the rest)
                    self._fetch(host)
        self.paint()

    def refresh_time(self) -> None:
        self.paint()                             # ages only: no fetch

    def on_scope_changed(self, message: ScopeChanged) -> None:
        self.first_host = None
        self._set_hosts()
        self.rf.update_visibility()
        self.paint()
        self._fetch_all()

    def on_resize(self) -> None:
        super().on_resize()
        self.call_after_refresh(self.paint)

    # ------------------------------------------------------------------ actions (keymap "account")
    def _box_rows(self, host: str) -> int:
        d = self.box_data(host)
        return len(d.accounts()) + len(rollup_rows(d, self._opts()))

    def action_cursor(self, step: int) -> None:
        """`j` / `k`: the next / previous row of the focused box; past its last (first) row on to the next
        (previous) box."""
        host = self._focused_host()
        if host is None:
            return
        self.focus_chosen = True
        n = self._box_rows(host)
        cur = self.box_cursor.get(host, 0) + step
        if cur >= n or cur < 0:
            nxt = self.focus_idx + (1 if cur >= n else -1)
            if 0 <= nxt < len(self.hosts):
                self.focus_idx = nxt
                other = self.hosts[nxt]
                self.box_cursor[other] = 0 if step > 0 else max(0, self._box_rows(other) - 1)
            self.paint()
            return
        self.box_cursor[host] = cur
        self.paint()

    def action_edge(self, end: int) -> None:
        host = self._focused_host()
        if host is None:
            return
        self.box_cursor[host] = max(0, self._box_rows(host) - 1) if end > 0 else 0
        self.paint()

    def action_box(self, step: int) -> None:
        if self.hosts:
            self.focus_chosen = True
            self.focus_idx = (self.focus_idx + step) % len(self.hosts)
            self.paint()

    def available_windows(self) -> list[UsageWindow]:
        """7d · 30d · 90d, and period when some cluster in scope has one (§7.17: unavailable ones are skipped)."""
        out: list[UsageWindow] = ["7d", "30d", "90d"]
        if any(self._has_period(h, self.box_data(h).summary) for h in self.hosts):
            out.append("period")
        return out

    def action_window(self, step: int) -> None:
        """`[` / `]`: the previous / next window, remembered in ui.json (`account_window`); paints from the cache, then
        asks for the reports (the service answers from its 10-minute cache)."""
        windows = self.available_windows()
        idx = windows.index(self.window) if self.window in windows else windows.index("30d")
        self.window = windows[(idx + step) % len(windows)]
        self.rf.save_ui_state(account_window=self.window)
        for host in self.hosts:
            self.box_offset[host] = 0
        self.paint()
        self._fetch_all()

    def action_next_group(self) -> None:
        self.opts = _replace(self.opts, group=GROUP_BYS[(GROUP_BYS.index(self.opts.group) + 1) % len(GROUP_BYS)])
        self._reset_rollup_cursors()

    def action_next_sort(self) -> None:
        self.opts = _replace(self.opts, sort=SORTS[(SORTS.index(self.opts.sort) + 1) % len(SORTS)], reverse=False)
        self._reset_rollup_cursors()

    def action_reverse_sort(self) -> None:
        self.opts = _replace(self.opts, reverse=not self.opts.reverse)
        self._reset_rollup_cursors()

    def _reset_rollup_cursors(self) -> None:
        for host in self.hosts:
            d = self.box_data(host)
            n_acc = len(d.accounts())
            if self.box_cursor.get(host, 0) >= n_acc:
                self.box_cursor[host] = n_acc
            self.box_offset[host] = 0
        self.paint()

    def action_toggle_scope(self) -> None:
        """`u`: MY columns ↔ the whole account's totals (the rollup stays mine)."""
        self.opts = _replace(self.opts, scope="account" if self.opts.scope == "me" else "me")
        self.rf.toast("account columns: the whole account" if self.opts.scope == "account"
                      else "account columns: your usage", reply=True)
        self.paint()

    def action_refresh(self) -> None:
        """`r`: read the focused cluster again (the report, not the cache), at most once per 60 s."""
        host = self._focused_host()
        if host is not None:
            self._refresh([host])

    def action_refresh_all(self) -> None:
        self._refresh(list(self.hosts))

    def _refresh(self, hosts: Sequence[str]) -> None:
        now = self.rf.now()
        asked: list[str] = []
        recent: list[tuple[str, float, bool]] = []
        for host in hosts:
            if (host, self.window) in self._loading:
                continue
            key = (host, self.window)
            report = self._report(host)
            age = (now - report.fetched_at).total_seconds() if report is not None and not report.errors else math.inf
            forced = (now - self._forced[key]).total_seconds() if key in self._forced else math.inf
            if min(age, forced) < REFRESH_GAP_S:
                recent.append((host, min(age, forced), forced < age))
                continue
            self._forced[key] = now
            self._fetch(host, 0)
            asked.append(host)
        if recent and not asked:
            host, age, by_r = recent[0]
            self.rf.toast(f"{host}: {'asked' if by_r else 'fetched'} {humanize.age(age)} ago", reply=True)
        elif asked:
            self.rf.toast(f"reading account usage of {', '.join(asked)}", reply=True)
        self.paint()

    def action_open(self) -> None:
        """`enter`: the Jobs view filtered to the row (an account: `acct:`; a rollup group: its name as a text term,
        `acct:` or `part:`), state preset `all`."""
        host = self._focused_host()
        if host is None:
            return
        d = self.box_data(host)
        o = self._opts()
        names = d.accounts()
        cur = self.box_cursor.get(host, 0)
        if cur < len(names):
            term = f"acct:{names[cur]}"
        else:
            rows = rollup_rows(d, o)
            i = cur - len(names)
            if not 0 <= i < len(rows):
                return
            key = rows[i].key
            if o.group == "gpu" or not key or " " in key:
                self.rf.toast("the Jobs filter cannot select this group: showing the host's jobs", reply=True)
                term = ""
            elif o.group == "account":
                term = f"acct:{key}"
            elif o.group == "partition":
                term = f"part:{key}"
            else:
                term = key
        self.rf.show_jobs_filtered(f"host:{host} {term}".strip(), preset="all")


def _never_connected(state: HostState) -> bool:
    snap = state.snapshot
    return (snap is None or snap.caps is None) and state.config.backend == "auto" \
        and state.status.state is not ConnState.OK


def _replace(opts: ViewOpts, **changes: Any) -> ViewOpts:
    return dataclasses.replace(opts, **changes)
