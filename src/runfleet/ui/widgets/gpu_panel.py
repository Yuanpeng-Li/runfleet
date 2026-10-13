"""[WP3] nvitop-style GPU bar panel (host detail, workstation; mockup G).

One header line plus one row per GPU: index, model, temperature, power/limit, a utilisation bar and a memory bar
sized to the panel width, and the processes on the GPU (``user/pid``, ``+N``). Pure rendering in
:func:`gpu_panel_lines`; the widget only re-renders on data or size changes. A click on a GPU row posts
:class:`GPUPanel.Picked` (host detail: show the processes on that GPU); ``selected`` marks one row.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static

from runfleet.models import GPU, GPUProcess
from runfleet.ui import render, theme
from runfleet.util import humanize


def gpu_panel_lines(gpus: Sequence[GPU], width: int, util_t: tuple[float, float],
                    mem_t: tuple[float, float], *, show_others: bool = True, me: str | None = None,
                    selected: int | None = None) -> list[Text]:
    """Header + one row per GPU, fitted to ``width``: below 100 columns the power and process columns are dropped
    (the process table under the panel lists them) so the bars keep a useful length."""
    wide = width >= 100
    procs_w = 18 if wide else 0
    name_w = 16 if wide else 12                 # "RTX PRO 6000 BW" fits when there is room
    fixed = 5 + name_w + 5 + (10 if wide else 0) + 3 + 5 + 3 + 15 + (3 + procs_w if wide else 0)
    bar = max(4, min(30, (width - fixed) // 2 - 2))
    head = Text(f"{'#':>3}  {'NAME':<{name_w}}{'TEMP':>5}", style=f"bold {theme.muted()}")
    if wide:
        head.append(f" {'PWR':>9}")
    head.append(f"   {'UTIL':<{bar + 7}}   {'MEM':<{bar + 17}}")
    if wide:
        head.append("   PROCS")
    lines = [render.fit(head, width)]
    for g in sorted(gpus, key=lambda x: x.index):
        if g.index == selected:
            out = Text(f"▸{g.index:>2}  ", style=theme.accent())
        else:
            out = Text(f"{g.index:>3}  ", style=theme.muted())
        out.append(humanize.truncate(humanize.short_gpu_name(g.name), name_w - 1).ljust(name_w))
        temp = f"{g.temp_c:.0f}°C" if g.temp_c is not None else humanize.NA
        out.append(temp.rjust(5), style=theme.temp_style(g.temp_c))
        if wide:
            if g.power_w is not None:
                pwr = f"{g.power_w:.0f}/{g.power_limit_w:.0f}W" if g.power_limit_w else f"{g.power_w:.0f}W"
            else:
                pwr = humanize.NA
            out.append(" " + pwr.rjust(9), style=theme.muted())
        out.append("   ")
        out.append_text(render.gpu_bar(g.util_pct, bar + 2, util_t))
        out.append(humanize.pct(g.util_pct, width=5))
        out.append("   ")
        mem_pct = g.mem_util_pct
        if g.mem_used_mib is not None and g.mem_total_mib:
            mem_pct = g.mem_used_mib / g.mem_total_mib * 100
        out.append_text(render.gpu_bar(mem_pct, bar + 2, mem_t))
        out.append(" " + humanize.gib_pair(g.mem_used_mib, g.mem_total_mib, sep=" / ").rjust(14))
        if wide:
            out.append("   ")
            out.append_text(_procs_text(g, show_others, me))
        lines.append(render.fit(out, width))
    if not gpus:
        lines.append(Text("  no GPUs reported (nvidia-smi missing, hung or not sampled yet)", style=theme.dim()))
    return lines


def _procs_text(g: GPU, show_others: bool, me: str | None) -> Text:
    procs = sorted(g.processes, key=lambda p: -(p.used_mem_mib or 0))
    hidden: list[GPUProcess] = []
    if not show_others and me is not None:
        hidden = [p for p in procs if p.user not in (None, me)]
        procs = [p for p in procs if p.user in (None, me)]
    out = Text()
    if procs:
        first = procs[0]
        out.append(f"{first.user or '?'}/{first.pid}", style="" if first.user == me or me is None else theme.muted())
        if len(procs) > 1:
            out.append(f" +{len(procs) - 1}", style=theme.muted())
    elif hidden:
        out.append(f"{len(hidden)} other user{'s' if len(hidden) > 1 else ''}", style=theme.dim())
    else:
        out.append(humanize.NONE, style=theme.dim())
    return out


class GPUPanel(Static):
    DEFAULT_CSS = """
    GPUPanel { height: auto; }
    """

    rows: list[int] = []                       # GPU index of each row under the header, as last rendered

    def show(self, gpus: Sequence[GPU], util_t: tuple[float, float], mem_t: tuple[float, float], *,
             show_others: bool, me: str | None, selected: int | None = None) -> None:
        width = self.size.width or (self.app.size.width - 4)
        self.rows = [g.index for g in sorted(gpus, key=lambda x: x.index)]
        self.update(Text("\n").join(gpu_panel_lines(gpus, width, util_t, mem_t, show_others=show_others, me=me,
                                                    selected=selected)))

    class Picked(Message):
        """A GPU row was clicked."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def on_click(self, event: events.Click) -> None:
        offset = event.get_content_offset(self)
        if offset is not None and 1 <= offset.y <= len(self.rows):      # line 0 is the column header
            self.post_message(self.Picked(self.rows[offset.y - 1]))
