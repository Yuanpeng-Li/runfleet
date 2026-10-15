"""Fleet panel of the roomy Fleet view (§7.4): free GPUs by type across the fleet and what changed lately.

It fills the empty grid cell after the last card (or a full-width row under the grid) when the plan of
:mod:`runfleet.ui.fleet_layout` leaves room for it. It is not a HostCard: it never takes focus, arrows skip it and a
click does nothing. Everything it shows is already in the store or in this session's alert history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

from runfleet.core.events import JobEvent
from runfleet.ui import fleet_layout, theme
from runfleet.ui.notify import since_label
from runfleet.ui.widgets.host_card import old_reading
from runfleet.util import humanize

if TYPE_CHECKING:
    from runfleet.core.store import HostState


def panel_lines(hosts: Mapping[str, HostState], now: datetime, *, rows: int, width: int, wide: bool, window: float,
                history: Iterable[JobEvent] = (), digest_since: datetime | None = None,
                memo: fleet_layout.Memo | None = None, rate: float = 1.0) -> list[Text]:
    """The panel body in exactly ``rows`` lines: the free part (at most half), then the activity part.

    ``memo`` keeps the fleet-wide scans between calls: the node and job scans until a host's data changes, the
    activity rows until then, a new alert or the moment a row leaves the window (the 1 Hz tick only re-renders
    text). ``history``: this session's alerts; those of hosts no longer in the fleet are left out. ``rate``: the +/-
    poll rate (how old a sinfo sample may be, :func:`fleet_layout.sinfo_bound_s`)."""
    memo = memo if memo is not None else fleet_layout.Memo()
    history = [e for e in history if e.key.host in hosts]          # a removed host's alerts are not news any more
    names = list(hosts)
    ended = tuple(memo.get(name, "ended", (snap.jobs,), (),
                           lambda: fleet_layout.own_ended(snap))           # noqa: B023 (called at once)
                  for name, state in hosts.items() if (snap := state.snapshot) is not None)
    free = fleet_layout.free_by_type(hosts, now, lambda state, gpu: old_reading(state, gpu, now), memo, rate)
    mine = fleet_layout.my_types(hosts, now, window, memo)
    yours = {id(t) for t in free.types.values() if t.label in mine and t.host is None}
    others = any(t.free for t in free.types.values() if id(t) not in yours)
    n_free = fleet_layout.panel_split(rows, 1 + len(yours) + (1 if others else 0))
    out = fleet_layout.free_lines(free, mine, n_free, width, wide=wide, host_order=names)
    since = digest_since or (now - timedelta(seconds=window))
    away = " (you were away)" if wide else ""
    head = (f"since {since_label(since, now)}{away}: " if digest_since is not None
            else f"last {humanize.window(window)}: ")

    def activity() -> tuple[str, list[fleet_layout.Activity], datetime | None]:
        jobs = [j for part in ended for j in part]
        return (fleet_layout.count_text(jobs, history, since),
                fleet_layout.activity_rows(jobs, history, now, window, since),
                fleet_layout.activity_expiry(jobs, history, now, window))

    # the rolling window moves with ``now``: what it holds changes only when an end or an event leaves it (expiry),
    # so the rows and the count line agree with the cards' "my jobs" counts to the second
    since_key = since if digest_since is not None else None
    counts, acts, _until = memo.get("*", "activity", (*ended, history[-1] if history else None),
                                    (window, since_key, len(history)), activity, now=now, expires=lambda v: v[2])
    colors = {name: theme.host_color(state.config, i) for i, (name, state) in enumerate(hosts.items())}
    out += fleet_layout.activity_lines(acts, counts, head, rows - len(out), width, wide=wide,
                                       host_style=lambda h: colors.get(h, ""), now=now)
    return out + [Text("")] * max(0, rows - len(out))


class FleetPanel(Static, can_focus=False):
    """The fleet panel (see the module docstring); hidden while the plan has no room for it."""

    DEFAULT_CSS = """
    FleetPanel { height: auto; }
    """

    def __init__(self) -> None:
        super().__init__("", id="fleet-panel")
        self.border_title = Text(fleet_layout.PANEL_TITLE, style=theme.muted())
        self.lines: Sequence[Text] = ()

    def set_lines(self, lines: Sequence[Text], span: int, rows: int) -> None:
        self.border_title = Text(fleet_layout.PANEL_TITLE, style=theme.muted())
        self.lines = list(lines)
        if self.styles.column_span != span:
            self.styles.column_span = span
        if self.styles.height is None or self.styles.height.value != rows + 2:
            self.styles.height = rows + 2
        self.update(Text("\n").join(list(lines)[:rows]))
