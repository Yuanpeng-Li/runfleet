"""[WP3] Other log files of a job (``f`` in the job detail and the log viewer): pick one and follow it.

A job's stdout often holds only a launcher's few lines while the program writes its real log somewhere under the
workdir (``runs/<task>/<run>/train.log``). :class:`FoundLogsScreen` asks ``FleetAPI.find_logs`` once (one bounded
``log.find`` probe on the login node / workstation, in a worker) for the files written there while the job ran and
lists them newest first: path relative to the workdir, size, age. The job's own stdout / stderr are marked (picking
one goes back to that stream), and so are other jobs' (``core.logs.known_logs``: a shared workdir holds every job's
``logs/<name>_<id>.out``); the cursor starts on the picked file shown, else ``core.logs.suggest`` — never one of
those while another row is left (:meth:`FoundLogsScreen.start_option`). ``enter`` dismisses with the file; the
caller follows it as ``LogRef(stream="file", source="found")`` and remembers it per job for the session
(``RunFleetApp.found_logs``). Rows are laid out for the list's real width (they never wrap).

The pure helpers decide when to offer it: :func:`find_problem` (own jobs with a workdir only), :func:`find_window`
(``core.logs.find_window``, the agent's window too: the job's start, and its end once it ended; the core adds its
slack to both) and :func:`quiet_hint` (a running job's stdout quiet by ``core.logs.stdout_quiet_for``, the agent's
rule too).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from runfleet.core.logs import (FIND_MAX_FILES, FIND_UNREADABLE, FoundLog, LogFiles, known_logs, norm_log_path,
                                stdout_quiet_for, suggest)
from runfleet.core.logs import find_window as logs_find_window
from runfleet.models import Job, JobState, LogRef
from runfleet.ui import keymap, theme
from runfleet.ui.screens import dialog_list_room
from runfleet.util import humanize

BOX_W = 104                      # = #fl-box width (runfleet.tcss)
LIST_ROWS = 20                   # = #fl-list max-height (less on a short terminal, dialog_list_room)
SUMMARY_ROWS = 3                 # the summary's lines and its margin, kept off the list's height (until rendered)
SIZE_W = 6                       # "1023 B", "3.1M"
AGE_W = 8                        # "123d ago"
MIN_PATH = 12                    # a row keeps this much of its path; the marks give way first
NAMES = "*.log *.out *.err *.stdout *.stderr"
SHOWN = "  ◀ shown"              # after the row of the file the view shows


# ---------------------------------------------------------------- pure helpers
def find_problem(job: Job | None) -> str | None:
    """Why `f` cannot look for this job's files (None: it can): own jobs with a known workdir only."""
    if job is None:
        return "no job selected"
    if not job.is_mine:
        return f"{job.key.label}: other log files are looked for in your own jobs' workdirs only"
    if not job.workdir:
        return f"{job.key.label}: no workdir known — nowhere to look for other log files"
    return None


def own_paths(job: Job, extra: Iterable[LogRef] = ()) -> dict[str, str]:
    """The job's own stdout / stderr paths (normalised) -> how a row marks them (``stdout``, ``stderr``,
    ``stdout+stderr``); ``extra``: refs the detail probe resolved (a polled ref may hold only a pattern)."""
    out: dict[str, str] = {}
    for ref in (*job.logs, *extra):
        if ref.stream not in ("stdout", "stderr") or not ref.path:
            continue
        path = norm_log_path(ref.path)
        if path not in out:
            out[path] = ref.stream
        elif out[path] != ref.stream:
            out[path] = "stdout+stderr"
    return out


def other_paths(jobs: Iterable[Job], job: Job) -> dict[str, str]:
    """Other jobs' stdout / stderr on the job's host (normalised) -> ``job <label> stdout``: marked, never suggested."""
    return known_logs(jobs, job.key)


def found_ref(found: FoundLog) -> LogRef:
    """A picked file as the ref the log views follow."""
    return LogRef(stream="file", path=found.path, source="found", size=found.size, mtime=found.mtime)


def rel_path(path: str, workdir: str) -> str:
    """``runs/a/s42/train.log`` for a file under the workdir; the absolute path otherwise."""
    base = workdir.rstrip("/") or "/"
    prefix = base if base.endswith("/") else base + "/"
    return path[len(prefix):] if path.startswith(prefix) and len(path) > len(prefix) else path


def file_label(ref: LogRef | None, width: int = 28) -> str:
    """The short name of a picked file for tabs and crumbs (its base name)."""
    if ref is None or not ref.path:
        return "file"
    return humanize.truncate(ref.path.rstrip("/").rsplit("/", 1)[-1] or ref.path, width)


def fresher_ref(polled: LogRef | None, resolved: Iterable[LogRef], stream: str = "stdout") -> LogRef | None:
    """The ref with the freshest stat: the polled one, or the job detail's resolved one when it is newer."""
    d = next((r for r in resolved if r.stream == stream), None)
    if d is not None and (polled is None or (d.mtime is not None and (polled.mtime is None or d.mtime > polled.mtime))):
        return d
    return polled


def quiet_hint(job: Job, ref: LogRef | None, now: datetime, stall_after: float) -> str | None:
    """One line for a running job of the user's whose stdout is quiet (``core.logs.stdout_quiet_for``: unchanged past
    ``stall_after``, or a launcher's few bytes unchanged for 10 min): the real log is probably elsewhere."""
    if job.state is not JobState.RUNNING or find_problem(job) is not None:
        return None
    quiet = stdout_quiet_for(ref, now, stall_after)
    if quiet is None:
        return None
    return f"stdout quiet for {humanize.age(quiet)}; f finds other log files in the workdir"


def find_window(job: Job) -> tuple[datetime | None, datetime | None, str]:
    """(since, until, how the summary says it) for the job: ``core.logs.find_window`` (the agent's window too: since
    its start, until its end once it ended; an ended job with no known start, the 24 h before its end)."""
    since, until = logs_find_window(job)
    started = job.start_time is not None and not job.est_start
    return since, until, window_text(since if started else None, until)


def window_text(since: datetime | None, until: datetime | None = None) -> str:
    """When the listed files were written: ``while the job ran (09-06 11:14 – 09-06 15:02)`` for an ended job."""
    if since is not None and until is not None:
        return f"while the job ran ({humanize.local_dt(since)} – {humanize.local_dt(until)})"
    if since is not None:
        return f"since the job started ({humanize.local_dt(since)})"
    if until is not None:
        return f"in the 24 h up to the job's end ({humanize.local_dt(until)})"
    return "in the last 24 h"


def incomplete_text(res: LogFiles) -> str | None:
    """Why the walk may have missed files (``LogFiles.incomplete``), and the cap (``truncated``: more matched than
    are listed) when both hold. Unreadable directories: the walk finished, so nothing "stopped early" or "so far"."""
    if res.incomplete is None:
        return None
    if res.incomplete == FIND_UNREADABLE:
        cap = f"\nonly the newest {FIND_MAX_FILES}: more were written" if res.truncated else ""
        return "some directories were unreadable: their files are not listed" + cap
    if res.truncated:
        return f"search stopped early ({res.incomplete}): over {FIND_MAX_FILES} found, the newest listed"
    return f"search stopped early ({res.incomplete}): {len(res.files)} found so far"


def trailer_width(mark: str, current: bool) -> int:
    """Cells the marks after a row's age take (``  · stdout``, ``  ◀ shown``): rows pad to the widest, so the size
    and age columns line up."""
    return (cell_len(f"  · {mark}") if mark else 0) + (cell_len(SHOWN) if current else 0)


def row_text(found: FoundLog, workdir: str, now: datetime, width: int, *, mark: str = "",
             current: bool = False, trailer: int = 0) -> Text:
    """``runs/cifar/s42/train.log    3.1M   12s ago  · stdout``, at most ``width`` cells: the path is cut from the
    left (it keeps MIN_PATH cells; a long mark gives way first), size and age always two spaces apart; the marks are
    padded to ``trailer`` cells (:func:`trailer_width` of the widest row) while that leaves the path its room."""
    tail = Text()
    tail.append(f"  {humanize.size(found.size):>{SIZE_W}}", style=theme.muted())
    tail.append(f"  {humanize.ago(found.mtime, now):>{AGE_W}}", style=theme.muted())
    shown = SHOWN if current else ""
    marks = Text()
    if mark:
        room = width - MIN_PATH - tail.cell_len - cell_len(shown) - 4
        if room >= 6:
            marks.append(f"  · {humanize.truncate(mark, room)}", style=theme.dim())
    if shown:
        marks.append(shown, style=theme.accent())
    pad = min(trailer, width - MIN_PATH - tail.cell_len) - marks.cell_len
    if pad > 0:
        marks.append(" " * pad)
    tail.append_text(marks)
    room = max(1, width - tail.cell_len)
    rel = humanize.truncate_left(rel_path(found.path, workdir), room)
    out = Text(rel, style=theme.dim() if mark else "")
    out.append(" " * max(0, room - cell_len(rel)))
    out.append_text(tail)
    out.truncate(max(1, width))                  # never wider than the list: a wrapped row runs size into age
    return out


# ---------------------------------------------------------------- screen
class FoundLogsScreen(ModalScreen[FoundLog | None]):
    """The files written under one job's workdir; dismisses with the one to follow (None: closed)."""

    BINDINGS = keymap.bindings("found_logs")
    HELP_CONTEXT = "found_logs"
    # the box's width is in runfleet.tcss (#fl-box): a DEFAULT_CSS width loses to the app's `.dialog { width: 80 }`.
    # A stable 1-cell scrollbar gutter: the rows' width does not depend on how many there are.
    DEFAULT_CSS = f"""
    #fl-summary {{ height: auto; margin: 0 0 1 0; }}
    #fl-list, #fl-list:focus {{
        border: none; background: transparent; padding: 0; height: auto; max-height: {LIST_ROWS};
        scrollbar-gutter: stable; scrollbar-size-vertical: 1;
    }}
    """

    def __init__(self, job: Job, *, current: str | None = None, since: datetime | None = None,
                 until: datetime | None = None, window: str | None = None, own: Mapping[str, str] | None = None,
                 others: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.job = job
        self.workdir = job.workdir or ""
        self.since = since
        self.until = until
        self.window = window or window_text(since, until)
        self.current = norm_log_path(current) if current else None
        self.own = dict(own if own is not None else own_paths(job))          # normalised path -> stdout / stderr
        self.others = dict(others or {})                                     # normalised path -> "job … stdout"
        self.result: LogFiles | None = None
        self.scanning = False                        # not `loading`: Widget.loading would cover the box with a spinner
        self.files: dict[str, FoundLog] = {}         # option id -> file
        self.row_width = 0                           # what the rows were laid out for (refit when the list differs)
        self.summary_rows = SUMMARY_ROWS             # the summary's lines as wrapped, and its margin

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="fl-box", classes="dialog")
        box.border_title = Text(f"Other log files · {self.job.key.host}:{self.job.key.label} · "   # Text: no markup
                                f"{humanize.truncate(self.job.name, 40)}")
        box.border_subtitle = "↵ follow · r look again · esc close"
        with box:
            yield Static("", id="fl-summary")
            yield OptionList(id="fl-list")

    def on_mount(self) -> None:
        self._fit_list()
        self.query_one(OptionList).focus()
        self.scan()

    def scan(self) -> None:
        if self.scanning:
            return
        self.scanning = True
        self._fill()
        self.run_worker(self._scan(), group="find-logs", exclusive=True, exit_on_error=False)

    async def _scan(self) -> None:
        host = self.job.key.host
        try:
            result = await self.rf.service.find_logs(host, self.workdir, since=self.since, until=self.until)
        except Exception as exc:                 # a failure is the summary line, never an app error
            result = LogFiles(self.workdir, (), self.rf.now(), file_error=(str(exc) or type(exc).__name__)[:120])
        self.result = result
        self.scanning = False
        if self.is_mounted:
            self._fill()

    def on_resize(self, _event: events.Resize) -> None:
        self._fit_list()
        self._fill()

    def _fit_list(self) -> None:
        self.query_one(OptionList).styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS,
                                                                        self.summary_rows)

    def forget_rendered(self) -> None:
        """Live theme switch: the summary and the rows carry theme colours."""
        if self.is_mounted:
            self._fill()

    def text_width(self) -> int:
        """Cells a row may take: the list's content width (its scrollbar gutter is stable), less the options' padding;
        before the list is laid out (or while hidden) the box's width by the stylesheet. A row one cell too wide
        wraps, and the size runs into the age on the next line."""
        menu = self.query_one(OptionList)
        pad = menu.get_component_styles("option-list--option").padding.width
        inner = menu.scrollable_content_region.width if menu.display else 0
        if inner <= 0:
            term = self.app.size.width or 120
            inner = min(BOX_W, term * 96 // 100) - 4 - menu.styles.scrollbar_size_vertical
        return max(20, inner - pad)

    def _refit(self) -> None:
        """After a layout: rows laid out for an estimate are laid out again for the list's real width."""
        if self.is_mounted and self.files and self.text_width() != self.row_width:
            self._fill()

    # ------------------------------------------------------------------ rendering
    def summary_width(self) -> int:
        static = self.query_one("#fl-summary", Static)
        return static.content_region.width if static.content_region.width > 0 else self.text_width() + 1

    def summary(self) -> Text:
        """The window searched and how it went: looking / error / nothing / N files; a second line when the walk
        stopped early (``incomplete``) or only the newest FIND_MAX_FILES are listed (``truncated``). Fitted to the
        box: the workdir is cut from the left."""
        width = self.summary_width()
        when = self.window

        def fit(before: str, after: str, short: str | None = None) -> str:     # the workdir between them
            if short is not None and width - cell_len(before + after) < 16:
                after = short
            return before + humanize.truncate_left(self.workdir, max(12, width - cell_len(before + after))) + after

        res = self.result
        if self.scanning:
            return Text(fit("looking for log files under ", f" written {when} …", " …"), style=theme.accent())
        if res is None:
            return Text("")
        if res.error is not None:
            return Text(f"{self.job.key.host}: {res.error.summary}", style=theme.error())
        if res.file_error is not None:
            return Text(fit("cannot list the files under ", f": {res.file_error}"), style=theme.error())
        stopped = incomplete_text(res)
        if not res.files:
            if stopped is None:
                names = f" ({NAMES})" if width >= 100 else ""
                return Text(fit(f"no log file{names} written under ", f" {when}"), style=theme.dim())
            if res.incomplete == FIND_UNREADABLE:      # the walk finished, in the directories it could read
                out = Text(fit("no log file found under ", f" {when}"), style=theme.dim())
            else:                                      # not "no log file written": the walk did not finish
                out = Text(fit("no log file found under ", f" {when} so far", " so far"), style=theme.dim())
            out.append(f"\n{stopped}", style=theme.warn())
            return out
        n = len(res.files)
        out = Text(fit(f"{n} file{'s' if n != 1 else ''} written under ", f" {when}, newest first", f" {when}"),
                   style="bold")
        if stopped is not None:
            out.append(f"\n{stopped}", style=theme.warn())
        elif res.truncated:
            out.append(f"\nonly the newest {FIND_MAX_FILES}: more were written", style=theme.warn())
        return out

    def mark(self, found: FoundLog) -> str:
        """``stdout`` / ``stderr`` for the job's own, ``job 3372094 stdout`` for another job's, else ""."""
        path = norm_log_path(found.path)
        return self.own.get(path) or self.others.get(path, "")

    def start_option(self) -> str | None:
        """The row the cursor starts on: the picked file being shown, else ``core.logs.suggest`` (never the job's own
        or another job's stdout / stderr), else the ``◀ shown`` row (the job's own stream the view shows), else the
        first row that is not another job's log, else the first row: ``enter`` follows another job's log only when
        nothing else is listed."""
        ids = list(self.files)
        if not ids:
            return None
        shown = next((i for i, f in self.files.items() if norm_log_path(f.path) == self.current), None)
        if shown is not None and self.current not in self.own:
            return shown
        best = suggest(list(self.files.values()), [*self.own, *self.others])
        if best is not None:
            return next(i for i, f in self.files.items() if f is best)
        if shown is not None:
            return shown
        return next((i for i, f in self.files.items() if norm_log_path(f.path) not in self.others), ids[0])

    def _fill(self) -> None:
        menu = self.query_one(OptionList)
        keep = menu.highlighted_option.id if menu.highlighted_option is not None else None
        res = self.result
        width = self.row_width = self.text_width()
        now = self.rf.now()
        options: list[Option] = []
        self.files = {}
        listed = res.files if res is not None and not self.scanning else ()
        rows = [(f, self.mark(f), norm_log_path(f.path) == self.current) for f in listed]
        trailer = max((trailer_width(mark, cur) for _f, mark, cur in rows), default=0)
        for found, mark, cur in rows:
            oid = f"f{len(self.files)}"
            self.files[oid] = found
            options.append(Option(row_text(found, self.workdir, now, width, mark=mark, current=cur,
                                           trailer=trailer), id=oid))
        summary = self.summary()
        self.query_one("#fl-summary", Static).update(summary)
        # a narrow terminal wraps it: the list gives up those rows, or its last ones would sit below the box
        sw = max(1, self.summary_width())
        lines = sum(max(1, -(-cell_len(line) // sw)) for line in summary.plain.split("\n")) + 1
        if lines != self.summary_rows:
            self.summary_rows = lines
            self._fit_list()
        menu.clear_options()
        menu.add_options(options)
        menu.display = bool(options)
        if not self.files:
            return
        if self.focused is not menu:         # hiding it while loading took the focus (enter must select again)
            menu.focus()
        start = keep if keep in self.files else self.start_option()
        if start is not None:
            menu.highlighted = menu.get_option_index(start)
        self.call_after_refresh(self._refit)

    # ------------------------------------------------------------------ actions (keymap "found_logs")
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        found = self.files.get(event.option.id or "")
        if found is not None:
            self.dismiss(found)

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_rescan(self) -> None:
        self.scan()

    def action_close(self) -> None:
        self.dismiss(None)


def picker(app: Any, job: Job, current: str | None, own: Mapping[str, str] | None = None) -> FoundLogsScreen:
    """The picker for `f` in a view showing ``current``: the job's window (start, and end once it ended), its own
    stdout / stderr (``own``, default from ``job.logs``) and the other jobs' on its host (the store, no probe)."""
    since, until, window = find_window(job)
    return FoundLogsScreen(job, current=current, since=since, until=until, window=window,
                           own=own if own is not None else own_paths(job),
                           others=other_paths(app.store.jobs([job.key.host]), job))


def remember_pick(found_logs: dict[Any, LogRef], job: Job, found: FoundLog,
                  own: Mapping[str, str] | None = None) -> str:
    """Remember a picked file for the job (this session) and return the stream to show: ``file``, or the job's own
    ``stdout`` / ``stderr`` (``own``, default from ``job.logs``) when that is what was picked (nothing remembered)."""
    mark = (own if own is not None else own_paths(job)).get(norm_log_path(found.path))
    if mark is not None:
        return "stderr" if mark == "stderr" else "stdout"
    found_logs[job.key] = found_ref(found)
    return "file"
