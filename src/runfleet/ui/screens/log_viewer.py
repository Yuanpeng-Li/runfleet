"""[WP3] Log viewer screen: append + follow (mockup I).

Full-screen log of one job: header with the path, stream tabs (``o``/``e``) with load and follow state, the log
body (``RichLog``, 20 000 lines) and the live ``\\r`` line. ``/`` searches the buffer (``n``/``N`` next/previous);
``]``/``[`` jump between error and warning lines (highlighted); ``p`` opens the whole file in ``$PAGER``.
Opened from a log search hit (§4.7.1) it streams from the hit's search window (an absolute byte offset) and lands on
the hit's line, paused. ``f`` picks another file written under the job's workdir (``found_logs``) and follows it as the
third stream (``file``); ``o`` / ``e`` go back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Input, Static

from runfleet.core.logsearch import LogHit, LogQuery
from runfleet.core.service import LogUpdate
from runfleet.models import Job, JobKey, LogRef, Visibility
from runfleet.ui import keymap, theme
from runfleet.ui.messages import FleetChanged
from runfleet.ui.screens import FleetScreen
from runfleet.ui.screens.found_logs import (file_label, find_problem, fresher_ref, own_paths, picker, quiet_hint,
                                             remember_pick)
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.log_view import RECONNECT_DELAYS, LogFollower, LogView
from runfleet.ui.widgets.tabs import tab_strip
from runfleet.util import humanize

STREAM_TAIL_BYTES = 65536          # FleetAPI.stream_log(tail_bytes=...) default (§3.7)
HIT_LOAD_CAP = 8 << 20             # a hit's window this far behind the file's end: plain tail, land on its text
HIT_WAIT = 5.0                     # s after the first lines: the hit's line never came, land on its text instead


class LogScreen(FleetScreen):
    BINDINGS = keymap.bindings("log")
    HELP_CONTEXTS = ("log", "global")
    HELP_TITLE = "Log viewer"

    def __init__(self, key: JobKey, stream: str = "stdout", detail_logs: Iterable[LogRef] = ()) -> None:
        super().__init__()
        self.key = key
        self.stream = stream
        # the stdout / stderr refs the job detail's probe resolved (a polled ref may hold only a pattern): `f` marks
        # those rows as the job's own, as the detail's picker does
        self.detail_logs = tuple(detail_logs)
        self.loaded = 0
        self.ended = False
        self.follower: LogFollower | None = None

    def _crumbs(self, label: str) -> list[Any]:
        """Jobs › <job> (back to its detail) › <stream> (a picked file: its name)."""
        last = file_label(self._ref(self.job)) if self.stream == "file" else self.stream
        return [("Jobs", ("view", "jobs")), (label, ("job", self.key.host, self.key.id)), last]

    def shown_ref(self) -> LogRef | None:
        """The log this view shows (the pager opens the same one)."""
        ref = self._ref(self.job)
        return ref if isinstance(ref, LogRef) else None

    def _ref(self, job: Job | None) -> Any:
        """The shown log's ref: stdout / stderr as polled (its path known: it follows requeues), else as job detail
        resolved it (``detail_logs``; a polled ref may hold only a pattern), or the file picked with `f`."""
        if job is None:
            return None
        ref = self.rf.stream_ref(job, self.stream)
        if self.stream != "file" and (ref is None or ref.path is None):
            ref = next((r for r in self.detail_logs if r.stream == self.stream and r.path), None) or ref
        return ref

    def compose(self) -> ComposeResult:
        job = self.rf.store.job(self.key)
        label = job.key.label if job is not None else self.key.id
        yield FleetHeader(crumbs=self._crumbs(label), host=self.key.host, id="header")
        yield Static("", id="streams", classes="title-line")
        hint = Static("", id="find-hint", classes="title-line")      # "stdout quiet for 13h; f finds …"
        hint.display = False
        yield hint
        yield LogView(id="log-view")
        search = Input(placeholder="search the buffer (enter: find · esc: cancel)", id="log-search", compact=True)
        search.display = False
        yield search
        yield HintBar(self.help_contexts(), id="hints")

    def on_mount(self) -> None:
        self.follower = LogFollower(self, [self.query_one(LogView)], on_update=self.on_log_update)
        self._start()
        if self.open_search:
            self.action_search()

    open_search = False                     # set by `/` in job detail: open with the search box
    open_at: re.Pattern[str] | None = None  # set by `L` in job detail: land on the first line of the cause chip
    hit: LogHit | None = None               # set by a log search result: stream from its window, land on its line
    hit_query: LogQuery | None = None
    _hit_line: int | None = None            # the raw line to land on (None: its text, from the end)
    _hit_since: float | None = None         # monotonic time of the first lines while landing
    _stream_start: int | None = None        # absolute offset the current stream began at (None: the last 64 KiB)

    @property
    def job(self) -> Job | None:
        return self.rf.store.job(self.key)

    def _start(self, *, retry: bool = False) -> None:
        job = self.job
        if job is None or self.follower is None:
            return
        ref = self._ref(job)
        self.loaded = 0
        self.ended = False
        view = self.query_one(LogView)
        view.ended = job.state.is_terminal
        header = self.query_one(FleetHeader)
        header.crumbs = tuple(self._crumbs(job.key.label))
        header.refresh_content()
        if ref is not None:
            start = self._hit_start(ref)
            self._stream_start = start
            if self.hit is not None:
                view.set_follow(False)
            self.follower.start(job.key.host, ref, retry=retry, start=start)
        else:                      # never leave the previous stream writing under the new label
            self.follower.stop()
            view.reset()
            view.apply(LogUpdate(note="no log path known"))
        self._render_streams()

    def _keep_following(self) -> None:
        """Start the stream once its path becomes known (or changes: requeue); reopen a dead stream with backoff
        while this screen is on top, the host answers and the job may still be writing."""
        job, follower = self.job, self.follower
        if job is None or follower is None:
            return
        ref = self._ref(job)
        if ref is not None and ref.path is not None and (follower.ref is None or follower.ref.path != ref.path):
            self._start()
            return
        state = self.rf.store.hosts.get(self.key.host)
        fresh = state is not None and not theme.is_problem(state.status.state) and not state.stale(self.rf.now())
        if self.app.screen is self and fresh and not job.state.is_terminal and follower.reconnect_due():
            self._start(retry=True)

    def on_log_update(self, update: LogUpdate) -> None:
        self.loaded += sum(len(line.encode("utf-8", "replace")) + 1 for line in update.lines)
        if update.eof:
            self.ended = True
        if self.open_at is not None and update.lines:
            self.call_after_refresh(self._land_on_cause)
        if self.hit is not None and (update.lines or update.eof):
            if self._hit_since is None:
                self._hit_since = self.rf.clock.monotonic()
            self.call_after_refresh(self._land_on_hit, update.eof)
        self._render_streams()

    def _hit_start(self, ref: Any) -> int | None:
        """Where the stream begins: the hit's search window when this is the hit's file and it did not grow by more
        than HIT_LOAD_CAP since (then the hit's line number is exact), else the usual last 64 KiB."""
        hit = self.hit
        if hit is None or ref.path != hit.file.path or self.stream != hit.file.stream:
            return None
        size = max(ref.size or 0, hit.size or 0)
        if size - hit.start > HIT_LOAD_CAP:
            self._hit_line = None
            return None
        self._hit_line = hit.line - 1
        return hit.start

    def _land_on_hit(self, final: bool = False) -> None:
        """Opened from a search hit: land on its line once loaded and drawn (then never again: following and jumping
        afterwards are the user's). Its text is the fallback when the line does not come (the file was truncated)."""
        hit, view = self.hit, self.query_one(LogView)
        if hit is None:
            return
        q = self.hit_query
        label = f"search hit “{humanize.truncate(q.pattern, 40)}”" if q is not None else "search hit"
        again = q.pattern if q is not None and not q.regex else ""
        line = self._hit_line if self._stream_start is not None else None
        state = view.land(line, hit.needle, label, again=again)
        waited = self._hit_since is not None and self.rf.clock.monotonic() - self._hit_since >= HIT_WAIT
        if state == "pending" and (final or waited) and line is not None:
            state = view.land(None, hit.needle, label, again=again)
        if state != "pending" or final or waited:
            self.hit = None
        self._render_streams()

    def _land_on_cause(self) -> None:
        """Opened from the cause chip: the first loaded line of the cause, once the buffer has it (then never again,
        so following / jumping afterwards is the user's)."""
        if self.open_at is not None and self.query_one(LogView).jump_error(pattern=self.open_at, first=True) \
                is not None:
            self.open_at = None
            self._render_streams()

    def _render_streams(self) -> None:
        job = self.job
        streams = self.query_one("#streams", Static)
        width = streams.size.width or (self.size.width or 120) - 2
        ref = self._ref(job)
        merged = ref is not None and ref.merged and self.stream == "stderr"
        items = [("stdout", "stdout"), ("stderr", "stderr (merged)" if merged else "stderr")]
        picked = self.rf.stream_ref(job, "file") if job is not None else None
        if picked is not None:                     # o / e / f(<name>): the way back to each
            items.append(("file", f"f({file_label(picked, 24)})"))
        line = Text(" ")
        line.append_text(tab_strip(items, self.stream, "screen.stream"))
        view = self.query_one(LogView)
        info = Text(f"   {humanize.size(self.loaded)} loaded", style=theme.muted())
        if ref is not None and ref.size is not None:
            # `loaded` counts lines after "\r" collapsing, so it undercounts tqdm logs; the stream starts with the
            # last STREAM_TAIL_BYTES (FleetAPI.stream_log default), so a file at most that big was read whole
            if self._stream_start is not None:       # from a search hit's window
                whole = self._stream_start == 0
            else:
                whole = self.loaded >= ref.size or ref.size <= STREAM_TAIL_BYTES
            # a picked file's size is from the find: it may have grown since (no "of" then)
            if whole or self.stream != "file":
                info.append(" (whole file)" if whole else f" of {humanize.size(ref.size)}", style=theme.muted())
        info.append(" · tail -F via ssh · ", style=theme.dim())
        state = self.rf.store.hosts[self.key.host].status.state if self.key.host in self.rf.store.hosts else None
        auto = self.follower is not None and self.follower.attempts < len(RECONNECT_DELAYS)
        if self.ended and state is not None and theme.is_problem(state):
            again = f"reconnects when {self.key.host} is back" if auto else "stream closed"
            info.append(f"connection lost ({theme.CONN_LABEL[state]}) — {again}", style=theme.error())
        elif self.ended and job is not None and job.state.is_terminal:
            info.append("ended · static", style=theme.muted())
        elif self.ended:
            info.append("stream ended — reconnecting shortly" if auto else "stream ended", style=theme.warn())
        elif self.follower is not None and self.follower.state in ("waiting", "idle"):
            info.append("waiting for the log path", style=theme.warn())
        else:
            info.append("following ●" if view.follow else "paused", style=theme.ok() if view.follow else theme.warn())
        line.append_text(info)
        path = (ref.path or ref.pattern or "") if ref is not None else "no log path known"
        room = width - line.cell_len - 1
        if room > 12:
            p = Text(humanize.truncate_left(path, room), style=theme.muted())
            line.append(" " * max(1, room - p.cell_len))
            line.append_text(p)
        streams.update(line)
        self._render_find_hint()

    def _render_find_hint(self) -> None:
        """One line under the streams when a running job's stdout went quiet (or holds only a launcher's lines)."""
        job = self.job
        text = None
        if job is not None and self.stream != "file":
            text = quiet_hint(job, fresher_ref(self.rf.log_ref(job, "stdout"), self.detail_logs), self.rf.now(),
                              self.rf.service.config.general.stall_after)
        widget = self.query_one("#find-hint", Static)
        widget.display = text is not None
        widget.update(Text(f" {text}" if text else "", style=theme.warn()))

    def refresh_data(self, changes: FleetChanged | None) -> None:
        if self.is_mounted:
            self.query_one(FleetHeader).refresh_content()
            self._keep_following()
            self._render_streams()

    def refresh_time(self) -> None:
        if self.is_mounted:
            self._render_find_hint()
        if self.is_mounted and self.follower is not None and self.follower.dead:
            self._keep_following()
        if self.is_mounted and self.hit is not None and self._hit_since is not None:
            self._land_on_hit()

    def current_host(self) -> str | None:
        return self.key.host

    def current_job(self) -> Job | None:
        return self.job

    def visibility(self) -> dict[str, Visibility]:
        return {self.key.host: Visibility.FOCUSED}

    # ------------------------------------------------------------------ actions (keymap "log")
    def action_stream(self, stream: str) -> None:
        job = self.job
        if stream == "file" and job is not None and self.rf.stream_ref(job, "file") is None:
            self._find_logs()                # nothing picked yet
            return
        if stream != self.stream:
            self.stream = stream
            self.hit = None                  # the other stream: no hit to land on
            self._start()

    def action_find_logs(self) -> None:
        """`f`: back to the file picked for this job; on it (or none picked yet) the picker of the files written under
        the job's workdir while it ran."""
        job = self.job
        if job is not None and self.stream != "file" and self.rf.stream_ref(job, "file") is not None:
            self.action_stream("file")
            return
        self._find_logs()

    def _find_logs(self) -> None:
        job = self.job
        problem = find_problem(job)
        if problem is not None or job is None:
            self.rf.toast(problem or "no job", severity="warning")
            return
        current = self._ref(job)
        own = own_paths(job, self.detail_logs)

        def picked(found: Any) -> None:
            job = self.job
            if found is None or not self.is_mounted or job is None:
                return
            self.stream = remember_pick(self.rf.found_logs, job, found, own)
            self.hit = None
            self._start()

        self.app.push_screen(picker(self.rf, job, current.path if current is not None else None, own), picked)

    def action_reconnect(self) -> None:
        """Reopen the stream now (also resets the automatic reconnect budget)."""
        self._start()

    def action_toggle_follow(self) -> None:
        view = self.query_one(LogView)
        view.set_follow(not view.follow)
        self._render_streams()

    def action_bottom(self) -> None:
        self.query_one(LogView).bottom()
        self._render_streams()

    def action_top(self) -> None:
        self.query_one(LogView).top()
        self._render_streams()

    # ------------------------------------------------------------------ search
    def action_search(self) -> None:
        box = self.query_one("#log-search", Input)
        box.display = True
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "log-search":
            return
        self._close_search()
        if event.value.strip():
            self.query_one(LogView).search(event.value.strip())
        self._render_streams()

    def _close_search(self) -> None:
        box = self.query_one("#log-search", Input)
        box.display = False
        self.query_one(LogView).richlog.focus()

    def action_search_next(self) -> None:
        self.query_one(LogView).search("", again=True)

    def action_search_prev(self) -> None:
        self.query_one(LogView).search("", forward=False, again=True)

    def action_next_error(self) -> None:
        self.query_one(LogView).jump_error()
        self._render_streams()

    def action_prev_error(self) -> None:
        self.query_one(LogView).jump_error(forward=False)
        self._render_streams()

    def escape(self) -> bool:
        box = self.query_one("#log-search", Input)
        if box.display:
            box.value = ""
            self._close_search()
            return True
        view = self.query_one(LogView)
        if view.search_note:
            view.clear_search()
            return True
        return False

    def action_copy_path(self) -> None:
        ref = self._ref(self.job)
        if ref is None or ref.path is None:
            self.rf.toast("no log path known", severity="warning")
            return
        self.app.copy_to_clipboard(ref.path)
        self.rf.toast(f"sent to clipboard (OSC 52): {ref.path}", timeout=8)   # no acknowledgement, like app's y/Y
