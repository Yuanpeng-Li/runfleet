"""Alert history (palette › Notifications) and the "since you were away" digest (``D``), §7.14.

Both are dialogs over the current screen; ``enter`` on a row closes the dialog and opens that job (a GPU watch that
fired: its host). The digest also offers ``f``: the Jobs view filtered to the jobs that ended since the last visit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from runfleet.core.events import JobEvent, count_line, end_summary, end_word, ended_at
from runfleet.models import Job, JobKey, StateGroup
from runfleet.ui import keymap, render, theme
from runfleet.ui.gpu_watch import WatchEvent
from runfleet.ui.notify import since_label
from runfleet.ui.screens import dialog_list_room
from runfleet.util import humanize

LIST_ROWS = 20                   # = the lists' max-height (less on a short terminal: dialog_list_room)
_DIALOG_CSS = """
    #notify-box, #digest-box { width: 104; }
    #notify-list, #notify-list:focus, #digest-list, #digest-list:focus {
        border: none; background: transparent; padding: 0; height: auto; max-height: 20;
    }
"""


def _parse_key(text: str | None) -> JobKey | None:
    if not text:
        return None
    host, _, jid = text.partition(":")
    return JobKey(host, jid)


def event_text(event: JobEvent | WatchEvent, width: int = 96) -> Text:
    """``17:23  k2:2332861 orb-swp--train… failed: exit 1`` (red for bad news; a GPU watch: ``free GPUs: …``)."""
    out = Text(humanize.local_hm(event.at) + "  ", style=theme.muted())
    out.append(event.text(), style=theme.warn() if event.bad else "")
    return render.fit(out, width)


def failure_text(job: Job, width: int = 96) -> Text:
    out = Text(f"{job.key.host}:{job.key.label}  ", style="bold")
    out.append(humanize.truncate(job.name, 40) + "  ")
    out.append_text(render.state_badge(job.state, job.held))
    why = end_summary(job)
    if why:
        out.append(f"  {why}", style=theme.state_style(job.state))
    end = ended_at(job)
    if end is not None:
        out.append(f"  ended {humanize.local_hm(end)}", style=theme.muted())
    return render.fit(out, width)


class NotificationsScreen(ModalScreen[JobKey | str | None]):
    """The session's alert history, newest first; dismisses with the job to open (a host name for a GPU watch)."""

    BINDINGS = keymap.bindings("notifications")
    HELP_CONTEXT = "notifications"
    DEFAULT_CSS = _DIALOG_CSS

    def __init__(self, events: Sequence[JobEvent | WatchEvent]) -> None:
        super().__init__()
        self.events = sorted(reversed(events), key=lambda e: e.at, reverse=True)    # equal times: newest added first

    def compose(self) -> ComposeResult:
        box = Vertical(id="notify-box", classes="dialog")
        box.border_title = f"Notifications · {len(self.events)} this session (newest first)"
        box.border_subtitle = "↵ open job · esc close"
        with box:
            if self.events:
                yield OptionList(*[Option(event_text(e), id=f"{i}") for i, e in enumerate(self.events)],
                                 id="notify-list")
            else:
                yield Static(Text("nothing yet: failures, timeouts, restarts and jobs low on time appear here "
                                  "(w watches a job for its start and end too; palette › Watch free GPUs… adds "
                                  "free GPUs)", style=theme.dim()))

    def on_mount(self) -> None:
        found = self.query(OptionList)
        if found:
            found.first().focus()
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        for menu in self.query(OptionList):
            menu.styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        picked = self.events[int(event.option.id or 0)]
        self.dismiss(picked.key if isinstance(picked, JobEvent) else picked.host or None)

    def action_cursor_down(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_down()

    def action_cursor_up(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_up()

    def action_close(self) -> None:
        self.dismiss(None)


def digest_header(since: datetime, jobs: Sequence[Job], hosts: Sequence[str], now: datetime) -> Text:
    lines = [Text.assemble(("since ", theme.muted()), (since_label(since, now), "bold"),
                           (f" ({humanize.ago(since, now)}) · {len(jobs)} ended", theme.muted()))]
    for host in hosts:
        mine = [j for j in jobs if j.key.host == host]
        if mine:
            row = Text(f"  {host:<12}", style="bold")
            row.append(count_line(end_word(j.state) for j in mine))
            lines.append(row)
    if not jobs:
        lines.append(Text("  nothing of yours ended in that time", style=theme.dim()))
    return Text("\n").join(lines)


class DigestScreen(ModalScreen[tuple[str, JobKey | None] | None]):
    """What ended since the last visit: counts per host, then the failures (``enter`` opens one, ``f`` filters)."""

    BINDINGS = keymap.bindings("digest")
    HELP_CONTEXT = "digest"
    DEFAULT_CSS = _DIALOG_CSS

    def __init__(self, since: datetime, jobs: Sequence[Job], hosts: Sequence[str], now: datetime) -> None:
        super().__init__()
        self.since = since
        self.jobs = list(jobs)
        self.hosts = list(hosts)
        self.now = now
        self.failures = [j for j in self.jobs if j.state.group in (StateGroup.FAILED, StateGroup.ENDED)]

    def compose(self) -> ComposeResult:
        box = Vertical(id="digest-box", classes="dialog")
        box.border_title = "Since you were away"
        box.border_subtitle = "↵ open job · f show in Jobs view · esc close"
        with box:
            yield Static(digest_header(self.since, self.jobs, self.hosts, self.now), id="digest-head")
            if self.failures:
                yield Static(Text(f"\nfailures ({len(self.failures)})", style="bold"))
                yield OptionList(*[Option(failure_text(j), id=str(j.key)) for j in self.failures],
                                 id="digest-list")

    def on_mount(self) -> None:
        found = self.query(OptionList)
        if found:
            found.first().focus()
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        """The header and the "failures (n)" lines sit above the list inside the same dialog."""
        head = digest_header(self.since, self.jobs, self.hosts, self.now)
        reserved = len(head.plain.splitlines()) + 2
        for menu in self.query(OptionList):
            menu.styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS, reserved)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(("open", _parse_key(event.option.id)))

    def action_cursor_down(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_down()

    def action_cursor_up(self) -> None:
        for menu in self.query(OptionList):
            menu.action_cursor_up()

    def action_filter(self) -> None:
        self.dismiss(("filter", None))

    def action_close(self) -> None:
        self.dismiss(None)
