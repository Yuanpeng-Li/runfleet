"""[WP3] Header: host chips, totals, refresh indicators (§7.7).

One line. The main screen shows the scope chip, one chip per host and the totals over the scope; detail screens show
a breadcrumb and the status of their host instead of the chips. Responsive per §7.4: full chips >= 110 columns, no
ages < 110, dots + names < 90, dots + numbers < 70. Chips are clickable (scope / error details); the `⚙` chip at
the right edge opens Settings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from runfleet.models import ConnState, StateGroup
from runfleet.ui import render, theme
from runfleet.ui.query import ended_at
from runfleet.util import humanize

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _click(action: str) -> Style:
    return Style(meta={"@click": action})


class FleetHeader(Static):
    """The first line of every full screen."""

    DEFAULT_CSS = """
    FleetHeader { height: 1; width: 1fr; }
    """

    def __init__(self, *, crumbs: Sequence[str] = (), host: str | None = None,
                 extra: Callable[[], str] | None = None, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.crumbs = tuple(crumbs)
        self.host = host
        self.extra = extra
        self._frame = 0
        self.text = Text()

    @property
    def rf(self) -> Any:
        return self.app

    def on_mount(self) -> None:
        self.refresh_content()

    def on_resize(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        self._frame += 1
        self.text = self.build(self.size.width or 120)
        self.update(self.text)

    # ------------------------------------------------------------------ pieces
    def _scope_chip(self, compact: bool) -> Text:
        scope = self.host or self.rf.scope
        label = (scope or "ALL") if not self.host else self.host
        if compact:
            return Text(label, style=theme.accent())
        out = Text("▏", style=theme.dim())
        out.append(label, style=theme.accent())
        out.append("▕", style=theme.dim())
        return out

    def _chips(self, form: str) -> Text:
        now = self.rf.now()
        out = Text()
        for i, (name, state) in enumerate(self.rf.store.hosts.items(), start=1):
            st = state.status
            age = (now - st.last_ok).total_seconds() if st.last_ok is not None else None
            retry = (st.next_poll_at - now).total_seconds() if st.next_poll_at is not None else None
            if form == "full":
                chip = render.host_chip(st, age, retry_in=retry)
            elif form == "short":
                chip = render.host_chip(st, None, form="short")
            elif form == "name":
                chip = render.host_chip(st, form="name")
            else:
                chip = render.host_chip(st, form="num", number=i)
            if self.rf.scope == name:
                chip.stylize("underline")
            chip.stylize(_click(f"app.chip({name!r})"))
            if out:
                out.append("  " if form in ("full", "short") else " ")
            out.append_text(chip)
        return out

    def _totals(self, compact: bool) -> Text:
        hosts = self.rf.scope_hosts()
        now = self.rf.now()
        window = self.rf.service.config.general.recent_window
        active = queued = failed = 0
        others = bool(getattr(self.rf, "others_visible", False))
        for job in self.rf.store.jobs(hosts):
            if not (job.is_mine or others):      # like the cards and the default Jobs view: mine only (`u` adds)
                continue
            g = job.state.group
            if g is StateGroup.ACTIVE:
                active += 1
            elif g is StateGroup.QUEUED:
                queued += 1
            elif g in (StateGroup.FAILED, StateGroup.ENDED):
                end = ended_at(job)
                if end is None or (now - end).total_seconds() <= window:
                    failed += 1
        sep = "" if compact else " "
        out = Text()
        out.append(f"R{sep}{active}", style=theme.state_style_group(StateGroup.ACTIVE) if active else theme.dim())
        out.append("  " if not compact else " ")
        out.append(f"PD{sep}{queued}", style=theme.state_style_group(StateGroup.QUEUED) if queued else theme.dim())
        out.append("  " if not compact else " ")
        window_label = "" if compact else f"/{humanize.window(window)}"
        out.append(f"F{sep}{failed}{window_label}",
                   style=theme.state_style_group(StateGroup.FAILED) if failed else theme.dim())
        return out

    def _badges(self, compact: bool) -> Text:
        out = Text()
        if self.rf.store.idle_paused:
            out.append("paused (idle)" if compact else "paused (idle) — any key resumes",
                       style=f"{theme.warn()} reverse")
            out.append("  ")
        elif self.rf.idle_throttled():
            out.append("idle ×4", style=theme.warn())
            out.append("  ")
        if self.rf.multiplier != 1.0:
            out.append(f"×{self.rf.multiplier:g}", style=theme.muted())
            out.append("  ")
        if self.rf.readonly:
            out.append("R/O", style=f"{theme.warn()} reverse")
            out.append("  ")
        return out

    def _refresh(self, compact: bool) -> Text:
        hosts = [self.host] if self.host else self.rf.scope_hosts()
        states = [self.rf.store.hosts[h] for h in hosts if h in self.rf.store.hosts]
        ticks = [s.status.tick_interval for s in states if s.status.tick_interval]
        busy = any(s.status.in_flight for s in states) and self.rf.animations
        glyph = SPINNER[self._frame % len(SPINNER)] if busy else "⟳"
        tick = humanize.age(min(ticks), dash="–") if ticks else "–"
        return Text(f"{glyph}{'' if compact else ' '}{tick}", style=theme.muted())

    def _host_status(self, compact: bool) -> Text:
        assert self.host is not None
        state = self.rf.store.hosts.get(self.host)
        if state is None:
            return Text()
        st = state.status
        now = self.rf.now()
        out = render.conn_text(st.state)
        if st.state in (ConnState.OK, ConnState.DEGRADED) and st.latency_s is not None:
            out.append(f" {st.latency_s:.2f}s", style=theme.muted())
        if st.last_ok is not None:
            prefix = " · " if st.state in (ConnState.OK, ConnState.DEGRADED) else " · last ok "
            out.append(f"{prefix}{humanize.ago(st.last_ok, now)}", style=theme.muted())
        if not compact and self.extra is not None:
            extra = self.extra()
            if extra:
                out.append(" · " + extra, style=theme.muted())
        out.stylize(_click(f"app.chip({self.host!r})"))
        return out

    @staticmethod
    def _gear() -> Text:
        """Clickable settings chip (same as `,`)."""
        out = Text("⚙", style=theme.accent())
        out.stylize(_click("app.settings"))
        return out

    # ------------------------------------------------------------------ assembly
    def build(self, width: int) -> Text:
        compact = width < 110
        left = Text(" ")
        left.append("RunFleet", style=f"bold {theme.ACCENT}" if not theme.no_color() else "bold")
        left.append(" ")
        left.append_text(self._scope_chip(compact and not self.crumbs))
        if self.crumbs:
            for c in self.crumbs:
                left.append(" › ", style=theme.dim())
                left.append(c)
            right = Text()
            right.append_text(self._host_status(width < 100) if self.host else Text())
            right.append("  ")
            right.append_text(self._badges(True))
            right.append_text(self._refresh(False))
            right.append("  ")
            right.append_text(self._gear())
            return self._join(left, right, width)

        right = self._badges(width < 90)
        right.append_text(self._totals(width < 90))
        right.append("   " if width >= 90 else "  ")
        right.append_text(self._refresh(width < 90))
        right.append("  " if width >= 90 else " ")
        right.append_text(self._gear())
        forms = ["full", "short", "name", "num"]
        start = 0 if width >= 110 else 1 if width >= 90 else 2 if width >= 70 else 3
        for form in forms[start:]:
            chips = self._chips(form)
            middle = Text("  ") + chips if form in ("full", "short") else Text(" │ ", style=theme.dim()) + chips
            if form not in ("full", "short"):
                middle.append(" │", style=theme.dim())
            if left.cell_len + middle.cell_len + right.cell_len + 2 <= width or form == "num":
                return self._join(left + middle, right, width)
        return self._join(left, right, width)   # pragma: no cover - loop always returns

    @staticmethod
    def _join(left: Text, right: Text, width: int) -> Text:
        right = right.copy()
        right.append(" ")
        gap = width - left.cell_len - right.cell_len
        if gap < 1:
            left = render.fit(left, max(0, width - right.cell_len - 1))
            gap = 1
        return Text.assemble(left, " " * gap, right)
