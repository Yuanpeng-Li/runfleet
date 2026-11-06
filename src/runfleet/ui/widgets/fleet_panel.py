"""Fleet panel of the roomy Fleet view (§7.4): free GPUs by type across the fleet and what changed lately.

It fills the empty grid cell after the last card (or a full-width row under the grid) when the plan of
:mod:`runfleet.ui.fleet_layout` leaves room for it. It is not a HostCard: it never takes focus and arrows skip it; a
click on an activity row opens that job, one on the free part the free GPUs dialog. Everything it shows is already in the
store or in this session's alert history (GPU watches that fired, :mod:`runfleet.ui.gpu_watch`, included).

A click on a free-GPU row opens the free GPUs dialog for that type (on the header or ``also free``: every type), which
says which partitions and nodes or which workstation GPUs those are; the clickable row under the mouse is underlined.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from rich.text import Text
from textual import events
from textual.message import Message

from runfleet.core.events import JobEvent
from runfleet.models import JobKey
from runfleet.ui import fleet_layout, theme
from runfleet.ui.gpu_watch import WatchEvent
from runfleet.ui.notify import since_label
from runfleet.ui.widgets.host_card import old_reading
from runfleet.ui.widgets.row_links import RowLinks
from runfleet.util import humanize
from runfleet.util.lineage import link

if TYPE_CHECKING:
    from runfleet.core.store import HostState


def panel_lines(hosts: Mapping[str, HostState], now: datetime, *, rows: int, width: int, wide: bool, window: float,
                history: Iterable[JobEvent] = (), digest_since: datetime | None = None,
                memo: fleet_layout.Memo | None = None, rate: float = 1.0,
                watch_events: Iterable[WatchEvent] = ()) -> list[Text]:
    """The lines of :func:`panel_body`."""
    return panel_body(hosts, now, rows=rows, width=width, wide=wide, window=window, history=history,
                      digest_since=digest_since, memo=memo, rate=rate, watch_events=watch_events)[0]


def panel_body(hosts: Mapping[str, HostState], now: datetime, *, rows: int, width: int, wide: bool, window: float,
               history: Iterable[JobEvent] = (), digest_since: datetime | None = None,
               memo: fleet_layout.Memo | None = None, rate: float = 1.0,
               watch_events: Iterable[WatchEvent] = ()) -> tuple[list[Text], list[JobKey | str | None]]:
    """The panel body in exactly ``rows`` lines: the free part (at most half), then the activity part; and what each
    line opens on a click: a job (activity rows), a free GPU type key or :data:`fleet_layout.ALL_FREE` (the free
    part), None (the count line, padding).

    ``memo`` keeps the fleet-wide scans between calls: the node and job scans until a host's data changes, the
    activity rows until then, a new alert or the moment a row leaves the window (the 1 Hz tick only re-renders
    text). ``history``: this session's alerts; those of hosts no longer in the fleet are left out. ``rate``: the +/-
    poll rate (how old a sinfo sample may be, :func:`fleet_layout.sinfo_bound_s`). ``watch_events``: GPU watches that
    fired this session (``◎`` rows, no job to open)."""
    memo = memo if memo is not None else fleet_layout.Memo()
    history = [e for e in history if e.key.host in hosts]          # a removed host's alerts are not news any more
    fired = [e for e in watch_events if e.kind == "gpu_free" and e.host in hosts]
    names = list(hosts)
    ended = tuple(memo.get(name, "ended", (snap.jobs,), (),
                           lambda: fleet_layout.own_ended(snap))           # noqa: B023 (called at once)
                  for name, state in hosts.items() if (snap := state.snapshot) is not None)
    snaps = [(name, snap) for name, state in hosts.items() if (snap := state.snapshot) is not None]
    lineage = memo.get("*", "lineage", tuple(snap.jobs for _name, snap in snaps), (),
                       lambda: link(j for _name, snap in snaps for j in snap.jobs.values()))
    free = fleet_layout.free_by_type(hosts, now, lambda state, gpu: old_reading(state, gpu, now), memo, rate)
    mine = fleet_layout.my_types(hosts, now, window, memo)
    n_free = fleet_layout.panel_split(rows, fleet_layout.free_needed(free, mine))
    out, free_keys = fleet_layout.free_region(free, mine, n_free, width, wide=wide, host_order=names)
    since = digest_since or (now - timedelta(seconds=window))
    away = " (you were away)" if wide else ""
    head = (f"since {since_label(since, now)}{away}: " if digest_since is not None
            else f"last {humanize.window(window)}: ")

    def activity() -> tuple[str, list[fleet_layout.Activity], datetime | None]:
        jobs = [j for part in ended for j in part]
        extra = [fleet_layout.Activity(e.at, e.host, fleet_layout.GLYPHS["gpu_free"], e.watch, e.summary, False,
                                       e.at <= since) for e in fired]
        return (fleet_layout.count_text(jobs, history, since),
                fleet_layout.activity_rows(jobs, history, now, window, since, extra, lineage=lineage),
                fleet_layout.activity_expiry(jobs, history, now, window, extra))

    # the rolling window moves with ``now``: what it holds changes only when an end or an event leaves it (expiry),
    # so the rows and the count line agree with the cards' "my jobs" counts to the second
    since_key = since if digest_since is not None else None
    counts, acts, _until = memo.get("*", "activity", (*ended, lineage, history[-1] if history else None,
                                                      fired[-1] if fired else None),
                                    (window, since_key, len(history), len(fired)), activity, now=now,
                                    expires=lambda v: v[2])
    colors = {name: theme.host_color(state.config, i) for i, (name, state) in enumerate(hosts.items())}
    keys: list[JobKey | str | None] = list(free_keys)
    lines, act_keys = fleet_layout.activity_region(acts, counts, head, rows - len(out), width, wide=wide,
                                                   host_style=lambda h: colors.get(h, ""), now=now)
    out += lines
    keys += act_keys
    pad = max(0, rows - len(out))
    return out + [Text("")] * pad, keys + [None] * pad


class FleetPanel(RowLinks, can_focus=False):
    """The fleet panel (see the module docstring); hidden while the plan has no room for it. A click on an activity
    row opens its job, one on the free part posts :class:`FleetPanel.FreeClicked` (the Fleet view opens the dialog)."""

    DEFAULT_CSS = """
    FleetPanel { height: auto; }
    """

    class FreeClicked(Message):
        """A click on the free part: ``key`` is a type key of the free GPUs, or ALL_FREE."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self) -> None:
        super().__init__("", id="fleet-panel")
        self.border_title = Text(fleet_layout.PANEL_TITLE, style=theme.muted())

    def set_lines(self, lines: Sequence[Text], span: int, rows: int,
                  keys: Sequence[JobKey | str | None] = ()) -> None:
        self.border_title = Text(fleet_layout.PANEL_TITLE, style=theme.muted())
        if self.styles.column_span != span:
            self.styles.column_span = span
        if self.styles.height is None or self.styles.height.value != rows + 2:
            self.styles.height = rows + 2
        self.show_lines(list(lines)[:rows], list(keys)[:rows], rows)

    def on_click(self, event: events.Click) -> None:
        key = self.key_at(event)
        if isinstance(key, JobKey):
            event.stop()
            self.app.open_listed_job(key)  # type: ignore[attr-defined]
        elif isinstance(key, str):
            event.stop()
            self.post_message(self.FreeClicked(key))
