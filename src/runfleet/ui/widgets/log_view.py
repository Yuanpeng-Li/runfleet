"""[WP3] Log view widget (RichLog with live line).

``LogView`` = ``RichLog(max_lines=20000, wrap=True, markup=False)`` fed ``Text.from_ansi`` per completed line, plus a
one-line footer showing the current unterminated line (tqdm ``\\r`` progress, §4.7) and the follow state.
``LogFollower`` owns one ``LogSession`` and fans its updates out to one or more views (the job detail Summary pane
and its Logs tab share a single stream). It tracks whether the stream is alive, so a dead stream never reads as
"following" and the owning screen can reopen it (:meth:`LogFollower.reconnect_due`, bounded backoff).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog, Static

from runfleet.core.service import LogUpdate
from runfleet.models import LogRef
from runfleet.ui import theme
from runfleet.util.logscan import line_level

MAX_LINES = 20_000
RECONNECT_DELAYS = (5.0, 15.0, 60.0, 60.0, 60.0)   # automatic reopen of a dead stream, then only on request
LIVE_RESETS_BACKOFF = 60.0                        # a stream that stayed up this long starts the backoff afresh


class LogView(Widget):
    DEFAULT_CSS = """
    LogView { height: 1fr; }
    LogView.box { height: 1fr; }
    LogView > RichLog { height: 1fr; background: ansi_default; scrollbar-size-vertical: 1; }
    LogView > #live { height: 1; color: $text-muted; }
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self.follow = True
        self.live: str | None = None
        self.note: str | None = None
        self.count = 0
        self.ended = False                 # the job has ended: the log is static (badge "ended · static")
        self.stream_ended = False          # the stream died (eof / error) while the job may still be writing
        self.search_note: str | None = None
        self._needle = ""
        self._hit = -1
        self._scroll_paused = False        # follow was paused by scrolling up (scrolling back down resumes it)
        self._err_hit = -1                 # display line of the last `]` / `[` jump

    def compose(self) -> ComposeResult:
        yield RichLog(max_lines=MAX_LINES, wrap=True, markup=False, highlight=False, min_width=20,
                      auto_scroll=True, id="log")
        yield Static("", id="live")

    def on_mount(self) -> None:
        self.watch(self.richlog, "scroll_y", self._scrolled, init=False)

    @property
    def richlog(self) -> RichLog:
        return self.query_one(RichLog)

    def reset(self) -> None:
        self.richlog.clear()
        self.count = 0
        self._err_hit = -1
        self.live = None
        self.note = None
        self.stream_ended = False
        self._render_live()

    def apply(self, update: LogUpdate) -> None:
        log = self.richlog
        for line in update.lines:
            log.write(_highlight(line), scroll_end=self.follow)
        self.count += len(update.lines)
        if update.lines or update.live is not None:
            self.live = update.live
        if update.note is not None:
            self.note = update.note
        if update.eof:
            self.note = update.note or "stream ended"
            self.stream_ended = True
        self._render_live()

    def _scrolled(self, old: float, new: float) -> None:
        """Scrolling up while following pauses (a new line must not yank the view back to the bottom); scrolling
        back to the bottom resumes a pause that scrolling caused. Auto-scroll only ever moves down."""
        at_bottom = new >= self.richlog.max_scroll_y - 1
        if self.follow and new < old and not at_bottom:
            self.set_follow(False)
            self._scroll_paused = True
        elif not self.follow and self._scroll_paused and new > old and at_bottom:
            self.set_follow(True)

    def set_follow(self, follow: bool) -> None:
        self._scroll_paused = False
        self.follow = follow
        self.richlog.auto_scroll = follow
        if follow:
            self.richlog.scroll_end(animate=False)
        self._render_live()

    def top(self) -> None:
        self.set_follow(False)
        self.richlog.scroll_home(animate=False)

    # ------------------------------------------------------------------ in-buffer search (`/`, `n`, `N`)
    def search(self, needle: str, *, forward: bool = True, again: bool = False) -> int | None:
        """Jump to the next (previous) display line containing ``needle`` (case-insensitive) and pause following.
        Returns the line index, or None (no match). Lines are the wrapped display lines of the buffer; a match that a
        wrap splits over two display lines counts on the first."""
        needle = needle if not again else self._needle
        if not needle:
            return None
        if needle != self._needle:
            self._needle, self._hit = needle, -1
        low = needle.lower()
        texts = [strip.text.lower() for strip in self.richlog.lines]

        def found(i: int) -> bool:
            if low in texts[i]:
                return True
            if i + 1 >= len(texts) or low in texts[i + 1]:
                return False
            # Rich wraps at a space (dropped, the line padded) or, for a word longer than the width, mid-word
            head, tail = texts[i].rstrip(), texts[i + 1].lstrip()
            return low in head + " " + tail or low in head + tail

        hits = [i for i in range(len(texts)) if found(i)]
        if not hits:
            self.search_note = f"/{needle}: no match"
            self._render_live()
            return None
        start = self._hit if self._hit >= 0 else int(self.richlog.scroll_y) - (1 if forward else -1)
        if forward:
            hit = next((h for h in hits if h > start), hits[0])
        else:
            hit = next((h for h in reversed(hits) if h < start), hits[-1])
        self._hit = hit
        self.set_follow(False)
        self.richlog.scroll_to(y=max(0, hit - 2), animate=False)
        self.search_note = f"/{needle}: {hits.index(hit) + 1}/{len(hits)} (line {hit + 1}) · n next · N previous"
        self._render_live()
        return hit

    def clear_search(self) -> None:
        self._needle, self._hit, self.search_note = "", -1, None
        self._render_live()

    # ------------------------------------------------------------------ error navigation (`]`, `[`, §5.2.8)
    def jump_error(self, *, forward: bool = True, pattern: re.Pattern[str] | None = None,
                   first: bool = False) -> int | None:
        """Jump to the next (previous) error / warning line (``logscan.line_level``) — or, with ``pattern``, the next
        line matching it; ``first`` = the first such line of the buffer (opening the log from the cause chip). Pauses
        following like a search hit; returns the display line index or None."""
        texts = [strip.text for strip in self.richlog.lines]

        def match(text: str) -> bool:
            return bool(pattern.search(text) if pattern is not None else line_level(text))

        def found(i: int) -> bool:
            if match(texts[i]):
                return True
            if i + 1 >= len(texts) or match(texts[i + 1]):
                return False
            head, tail = texts[i].rstrip(), texts[i + 1].lstrip()     # a match a wrap splits counts on its head
            return match(head + " " + tail) or match(head + tail)

        marks = [found(i) for i in range(len(texts))]
        # a run of matching display lines (one wrapped line, or a burst of errors) is one stop
        hits = [i for i, m in enumerate(marks) if m and not (i and marks[i - 1])]
        what = "cause" if pattern is not None else "error / warning"
        if not hits:
            self.search_note = f"no {what} line in the loaded log"
            self._render_live()
            return None
        if first:
            hit = hits[0]
        else:
            top = int(self.richlog.scroll_y)       # no jump yet: from the screen (`[` includes all of it)
            start = self._err_hit if self._err_hit >= 0 else top - 1 if forward else top + self.richlog.size.height
            if forward:
                hit = next((h for h in hits if h > start), hits[0])
            else:
                hit = next((h for h in reversed(hits) if h < start), hits[-1])
        self._err_hit = hit
        self.set_follow(False)
        self.richlog.scroll_to(y=max(0, hit - 2), animate=False)
        self.search_note = (f"{what} {hits.index(hit) + 1}/{len(hits)} (line {hit + 1}) · ] next · [ previous: "
                            f"{texts[hit].strip()}")
        self._render_live()
        return hit

    def bottom(self) -> None:
        self.set_follow(True)

    def _render_live(self) -> None:
        if not self.is_mounted:
            return
        width = self.size.width or 80
        if self.ended:
            right = Text("── ended · static ──", style=theme.muted())
        elif self.stream_ended:
            right = Text("── stream ended ──", style=theme.warn())
        else:
            right = Text("── following ● ──" if self.follow else "── paused (F follows) ──",
                         style=theme.ok() if self.follow else theme.warn())
        left = Text()
        if self.search_note:
            left.append(self.search_note, style=theme.accent())
        elif self.live:
            left.append("↻ ", style=theme.accent())
            left.append_text(Text.from_ansi(self.live))
        elif self.note:
            left.append(self.note, style=theme.warn())
        room = width - right.cell_len - 1
        if left.cell_len > room:
            left.truncate(max(0, room), overflow="ellipsis")
        self.query_one("#live", Static).update(Text.assemble(left, " " * max(1, room - left.cell_len + 1), right))

    def on_resize(self) -> None:
        self._render_live()


class LogFollower:
    """One followed log (``FleetAPI.stream_log``) feeding ``views``; restart with :meth:`start`, end with :meth:`stop`.

    ``state``: ``idle`` (nothing followed), ``waiting`` (the path is not known yet), ``connecting``, ``live``,
    ``ended`` (eof / the connection dropped) or ``failed`` (could not open)."""

    def __init__(self, owner: Widget, views: Sequence[LogView],
                 on_update: Callable[[LogUpdate], None] | None = None) -> None:
        self.owner = owner
        self.views = list(views)
        self.on_update = on_update
        self.ref: LogRef | None = None
        self.host: str | None = None
        self.session: Any = None
        self.state = "idle"
        self.attempts = 0                  # automatic reopens since the last explicit start
        self._gen = 0                      # bumped by start/stop: a superseded worker never touches the views
        self._live_at = 0.0
        self._died_at = 0.0

    @property
    def dead(self) -> bool:
        return self.state in ("ended", "failed")

    def _now(self) -> float:
        return float(self.owner.app.clock.monotonic())   # type: ignore[attr-defined]

    def stop(self) -> None:
        """Stop following: cancel the worker (its ``finally`` closes the session) before anything else is shown."""
        self._gen += 1
        self.owner.workers.cancel_group(self.owner, "log-follow")
        self.session = None
        self.state = "idle"

    def start(self, host: str, ref: LogRef, *, retry: bool = False) -> None:
        self.stop()
        self.attempts = self.attempts + 1 if retry else 0
        self.host, self.ref = host, ref
        for view in self.views:
            view.reset()
            if ref.path is None:
                view.apply(LogUpdate(note=f"log path unknown yet ({ref.pattern or 'no pattern'})"))
        if ref.path is None:
            self.state = "waiting"
            return
        self.state = "connecting"
        self.owner.run_worker(self._run(host, ref, self._gen), group="log-follow", exclusive=True,
                              exit_on_error=False)

    def reconnect_due(self) -> bool:
        """A dead stream whose next automatic reopen is due (5 s, 15 s, then 60 s apart; at most 5 in a row)."""
        if not self.dead or self.ref is None or self.ref.path is None or self.attempts >= len(RECONNECT_DELAYS):
            return False
        return self._now() - self._died_at >= RECONNECT_DELAYS[self.attempts]

    def _died(self, gen: int, state: str) -> None:
        if gen != self._gen:
            return
        now = self._now()
        if self.state == "live" and now - self._live_at >= LIVE_RESETS_BACKOFF:
            self.attempts = 0
        self.state, self._died_at = state, now

    def _fan_out(self, gen: int, update: LogUpdate) -> None:
        if gen != self._gen:
            return
        for view in self.views:
            if view.is_attached:
                view.apply(update)
        if self.on_update is not None:
            self.on_update(update)

    async def _run(self, host: str, ref: LogRef, gen: int) -> None:
        service = self.owner.app.service   # type: ignore[attr-defined]
        try:
            session = await service.stream_log(host, ref)
        except Exception as exc:   # a broken stream is a note in the pane, never an app error
            self._died(gen, "failed")
            self._fan_out(gen, LogUpdate(note=f"cannot follow log: {exc}", eof=True))
            return
        if gen != self._gen:       # superseded while opening
            await _close(session)
            return
        self.session = session
        self.state, self._live_at = "live", self._now()
        eof = False
        try:
            async for update in session.updates():
                eof = eof or update.eof
                if update.eof:
                    self._died(gen, "ended")
                self._fan_out(gen, update)
            if not eof:            # the session just stopped (connection dropped): say so, never "following"
                self._died(gen, "ended")
                self._fan_out(gen, LogUpdate(note="stream ended", eof=True))
        except Exception as exc:   # same as a failed open: a note in the pane
            self._died(gen, "ended")
            self._fan_out(gen, LogUpdate(note=f"stream failed: {exc}", eof=True))
        finally:
            if gen == self._gen:
                self.session = None
            await _close(session)


def _highlight(line: str) -> Text:
    """A log line as shown: ANSI colours kept, error lines in the error colour, warnings in the warning colour."""
    text = Text.from_ansi(line)
    level = line_level(line)
    if level is not None:
        text.stylize(theme.error() if level == "error" else theme.warn())
    return text


async def _close(session: Any) -> None:
    try:
        await session.close()
    except Exception:   # closing a dead stream must not raise
        pass
