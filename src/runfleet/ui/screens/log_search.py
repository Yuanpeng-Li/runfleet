"""Palette › Search logs… (``^f`` in the Jobs view): the prompt and the results, §4.7.1 / §7.10 (``core/logsearch.py``).

:class:`SearchPromptScreen` asks for the pattern (literal by default; ``tab`` reaches "regular expression" and
"match case") and says what will be read: the user's own jobs of the hosts in scope that run or ended within
``general.recent_window``, the last 256 KiB of each log, at most 20 files per host. :class:`LogSearchScreen` runs one
``log.search`` probe per host (only now, never again by itself), lists the matches per log file — job, stdout /
stderr, the line excerpt with the match highlighted, cut to the width in cells (CJK-safe) — and says honestly what was
read: files, KiB, and every cap that cut the search short. ``enter`` opens the log viewer on the hit's line, above this
list (``esc`` there comes back here); ``/`` edits the search.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.cells import cell_len, get_character_cell_size
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Input, OptionList, Static
from textual.widgets.option_list import Option

from runfleet.core import logsearch
from runfleet.core.logsearch import FileOutcome, LogHit, LogQuery, LogSearchResult, SearchPlan
from runfleet.models import ConnState
from runfleet.ui import keymap, theme
from runfleet.ui.screens import dialog_list_room
from runfleet.util import humanize

BOX_W = 120                      # = #ls-box width (runfleet.tcss)
LIST_ROWS = 26                   # = #ls-list max-height (less on a short terminal, dialog_list_room)
SKIP_STATES = {ConnState.DISABLED: "disabled", ConnState.UNREACHABLE: "unreachable",
               ConnState.NEEDS_AUTH: "needs auth", ConnState.HOSTKEY: "host key"}
LIMITS = (f"the last {logsearch.PER_FILE // 1024} KiB of each log · ≤ {logsearch.MAX_FILES} files and "
          f"{logsearch.READ_BUDGET // (1 << 20)} MiB per host · ≤ {logsearch.MAX_MATCHES} matches")


def fit_excerpt(text: str, spans: Sequence[tuple[int, int]], width: int, *, cut_before: bool = False,
                cut_after: bool = False, style: str = "", hit_style: str = "") -> Text:
    """One excerpt in at most ``width`` cells: the first match a third in when the text is wider, ``…`` where text
    was left out (also where the search's context window cut the line), matches in ``hit_style``. Widths are cells,
    so wide (CJK) characters count twice and are never split."""
    width = max(4, width)
    first = spans[0][0] if spans else 0
    start, end = 0, len(text)
    lead = "…" if cut_before else ""
    tail = "…" if cut_after else ""
    if cell_len(lead + text + tail) > width:
        room = width - 2                                         # an ellipsis may go on either side
        start, used = first, 0
        while start > 0 and used + get_character_cell_size(text[start - 1]) <= room // 3:
            start -= 1
            used += get_character_cell_size(text[start])
        end = first                                              # text[start:first] is counted in `used`
        while end < len(text) and used + get_character_cell_size(text[end]) <= room:
            used += get_character_cell_size(text[end])
            end += 1
        if start > 0:
            lead = "…"
        if end < len(text):
            tail = "…"
        if not lead and end < len(text) and used + get_character_cell_size(text[end]) <= room + 1:
            used += get_character_cell_size(text[end])
            end += 1
    out = Text(lead, style=theme.dim())
    body = Text(text[start:end], style=style)
    for s, e in spans:
        s, e = max(s, start), min(e, end)
        if e > s:
            body.stylize(hit_style, s - start, e - start)
    out.append_text(body)
    out.append(tail, style=theme.dim())
    if out.cell_len > width:                                    # never wider than asked (a wide char at the edge)
        out.truncate(width)
    return out


def scope_text(plans: Sequence[SearchPlan], skipped: dict[str, str], window: float) -> Text:
    """What a search would read now, per host: jobs, files, and what is left out."""
    out = Text(f"your jobs running or ended in the last {humanize.window(window)}: ", style=theme.muted())
    parts: list[Text] = []
    for p in plans:
        part = Text(f"{p.host} ", style="bold")
        if not p.jobs:
            part.append("no such jobs", style=theme.dim())
        else:
            part.append(f"{p.jobs} job{'s' if p.jobs != 1 else ''} · {len(p.files)} file"
                        f"{'s' if len(p.files) != 1 else ''}", style=theme.muted())
            if p.left_out:
                part.append(f" (+{p.left_out} left out: {logsearch.MAX_FILES}-file limit)", style=theme.warn())
            if p.no_path and not p.files:
                part.append(" (no log path known)", style=theme.dim())
        parts.append(part)
    for host, why in skipped.items():
        parts.append(Text(f"{host} {why}: not searched", style=theme.warn()))
    out.append_text(Text("  ·  ").join(parts) if parts else Text("no hosts in scope", style=theme.dim()))
    return out


class SearchPromptScreen(ModalScreen[LogQuery | None]):
    """Ask for the pattern; ``↵`` in the pattern dismisses with the query once it is usable, ``esc`` with None."""

    BINDINGS = keymap.bindings("search_prompt")
    HELP_CONTEXT = "search_prompt"
    DEFAULT_CSS = """
    SearchPromptScreen Input { margin: 1 0 0 0; }
    SearchPromptScreen #ls-options { height: auto; margin: 1 0 0 0; }
    SearchPromptScreen Checkbox { border: none; padding: 0 2 0 0; background: transparent; width: auto; }
    """

    def __init__(self, query: LogQuery | None, hosts: Sequence[str]) -> None:
        super().__init__()
        self.query_ = query or LogQuery("")
        self.hosts = list(hosts)

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="ls-prompt-box", classes="dialog")
        box.border_title = "Search logs…"
        box.border_subtitle = "↵ search · tab options · esc cancel"
        with box:
            yield Static("", id="ls-scope")
            yield Input(self.query_.pattern, placeholder="text to find, e.g. CUDA out of memory", id="ls-pattern",
                        compact=True, max_length=logsearch.PATTERN_MAX)
            with Vertical(id="ls-options"):
                yield Checkbox("regular expression (grep -E)", self.query_.regex, id="ls-regex", compact=True)
                yield Checkbox("match case", self.query_.match_case, id="ls-case", compact=True)
            yield Static("", id="ls-note")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._render_scope()
        self._note(submitted=False)

    def forget_rendered(self) -> None:
        """Live theme switch: the scope line and the note carry theme colours."""
        if self.is_mounted:
            self._render_scope()
            self._note(submitted=False)

    def plans(self) -> tuple[list[SearchPlan], dict[str, str]]:
        """The search plan of every host in scope that can be searched now, and why the others are not."""
        store = self.rf.store
        now, window = self.rf.now(), self.rf.service.config.general.recent_window
        plans, skipped = [], {}
        for host in self.hosts:
            state = store.hosts[host].status.state if host in store.hosts else None
            if state in SKIP_STATES:
                skipped[host] = SKIP_STATES[state]
                continue
            plans.append(logsearch.plan(host, store.jobs([host]), now, window))
        return plans, skipped

    def _render_scope(self) -> None:
        plans, skipped = self.plans()
        line = scope_text(plans, skipped, self.rf.service.config.general.recent_window)
        line.append(f"\n{LIMITS}", style=theme.dim())
        self.query_one("#ls-scope", Static).update(line)

    def current(self) -> LogQuery:
        return LogQuery(self.query_one(Input).value, self.query_one("#ls-regex", Checkbox).value,
                        self.query_one("#ls-case", Checkbox).value)

    def _note(self, *, submitted: bool) -> str | None:
        query = self.current()
        problem = logsearch.pattern_problem(query)
        if problem is None and not any(p.files for p in self.plans()[0]):
            problem = "nothing to search: no log of a recent job of yours in scope (0 / 1-9 change the scope)"
        note = self.query_one("#ls-note", Static)
        if problem is not None and (query.pattern or submitted):
            note.update(Text(problem, style=theme.warn()))
        else:
            note.update(Text(f"{query.describe()} · runs once, now: one read-only probe per host",
                             style=theme.dim()))
        return problem

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._note(submitted=False)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._note(submitted=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._note(submitted=True) is None:
            self.dismiss(self.current())

    def action_close(self) -> None:
        self.dismiss(None)


class LogSearchScreen(ModalScreen[str | None]):
    """The matches of one search, per log file; ``enter`` opens a hit in the log viewer (pushed above this list).
    Dismisses with ``"again"`` (``/``: edit the search) or None."""

    BINDINGS = keymap.bindings("log_hits")
    HELP_CONTEXT = "log_hits"
    DEFAULT_CSS = f"""
    #ls-summary {{ height: auto; margin: 0 0 1 0; }}
    #ls-list, #ls-list:focus {{
        border: none; background: transparent; padding: 0; height: auto; max-height: {LIST_ROWS};
    }}
    """

    def __init__(self, query: LogQuery, hosts: Sequence[str]) -> None:
        super().__init__()
        self.query_ = query
        self.hosts = list(hosts)
        self.skipped: dict[str, str] = {}
        self.pending: list[str] = []
        self.results: dict[str, LogSearchResult] = {}
        self.hits: dict[str, LogHit] = {}        # option id -> hit

    @property
    def rf(self) -> Any:
        return self.app

    def compose(self) -> ComposeResult:
        box = Vertical(id="ls-box", classes="dialog")
        box.border_title = Text(f"Search logs · “{humanize.truncate(self.query_.pattern, 60)}” · "    # Text: never markup
                                f"{self.query_.describe()}")
        box.border_subtitle = "↵ open in the log viewer · / edit search · esc close"
        with box:
            yield Static("", id="ls-summary")
            yield OptionList(id="ls-list")

    def on_mount(self) -> None:
        store = self.rf.store
        for host in self.hosts:
            state = store.hosts[host].status.state if host in store.hosts else None
            if state in SKIP_STATES:
                self.skipped[host] = SKIP_STATES[state]
            else:
                self.pending.append(host)
        now, window = self.rf.now(), self.rf.service.config.general.recent_window
        probed = sum(1 for h in self.pending if logsearch.plan(h, store.jobs([h]), now, window).files)
        share = max(1, logsearch.MAX_MATCHES // max(1, probed))     # 200 matches overall, split between the probes
        for host in list(self.pending):
            self.run_worker(self._search(host, share), group="log-search", exit_on_error=False)
        self._fit_list()
        self.query_one(OptionList).focus()
        self._fill()

    async def _search(self, host: str, max_matches: int) -> None:
        try:
            result = await self.rf.service.search_logs(host, self.query_, max_matches=max_matches)
        except Exception as exc:                  # a host's failure is a line in the summary, never an app error
            result = LogSearchResult(host, self.query_, SearchPlan(host), max_matches=max_matches,
                                     error=str(exc)[:120] or "failed")
        self.results[host] = result
        if host in self.pending:
            self.pending.remove(host)
        if self.is_mounted:
            self._fill()

    def on_resize(self, _event: events.Resize) -> None:
        self._fit_list()
        self._fill()

    def _fit_list(self) -> None:
        self.query_one(OptionList).styles.max_height = dialog_list_room(self.app.size.height, LIST_ROWS, 5)

    def forget_rendered(self) -> None:
        """Live theme switch: the summary and every row carry theme colours."""
        if self.is_mounted:
            self._fill()

    def text_width(self) -> int:
        term = self.app.size.width or 120
        return max(40, min(BOX_W, term * 96 // 100) - 4 - 2)

    # ------------------------------------------------------------------ rendering
    def ordered(self) -> list[LogSearchResult]:
        return [self.results[h] for h in self.hosts if h in self.results]

    def summary(self) -> Text:
        """Searched N files, M KiB, matches — and every cap or failure that cut the search short."""
        done = self.ordered()
        files = sum(len(r.searched) for r in done)
        kib = sum(r.bytes_read for r in done) / 1024
        hits = sum(len(r.hits) for r in done)
        out = Text()
        if self.pending:
            out.append(f"searching {', '.join(self.pending)} …", style=theme.accent())
            if done:
                out.append("  ·  ", style=theme.dim())
        if done or not self.pending:
            hosts = [r.host for r in done if r.searched]
            out.append(f"searched {files} file{'s' if files != 1 else ''}"
                       f"{' on ' + ', '.join(hosts) if hosts else ''} · {kib:,.0f} KiB · {hits} "
                       f"match{'es' if hits != 1 else ''}", style="bold")
        caps = self.caps(done)
        if caps:
            out.append("\ncapped: " + " · ".join(caps), style=theme.warn())
        elif done and not self.pending and not any(r.error for r in done):
            out.append("\nnothing capped: every file in scope was searched in full", style=theme.ok())
        for r in done:
            if r.error:
                out.append(f"\n{r.host}: {r.error}", style=theme.error())
        for host, why in self.skipped.items():
            out.append(f"\n{host}: {why} — not searched", style=theme.warn())
        problems = [(r.host, f) for r in done for f in r.problems]
        if problems:
            kinds: dict[str, int] = {}
            for _host, f in problems:
                kinds[f.status] = kinds.get(f.status, 0) + 1
            out.append("\nnot readable: " + " · ".join(f"{n} {k}" for k, n in kinds.items()), style=theme.dim())
        empty = [r.host for r in done if not r.plan.files and not r.error]
        if empty:
            out.append(f"\nno log of a recent job of yours: {', '.join(empty)}", style=theme.dim())
        return out

    @staticmethod
    def caps(done: Sequence[LogSearchResult]) -> list[str]:
        out: list[str] = []
        window = sum(1 for r in done for f in r.files if f.window_cut and not f.budget_cut)
        if window:
            out.append(f"only the last {logsearch.PER_FILE // 1024} KiB of {window} file{'s' if window != 1 else ''}")
        budget_cut = sum(1 for r in done for f in r.files if f.budget_cut)
        budget_skip = sum(1 for r in done for f in r.files if f.status == "skipped (budget)")
        if budget_cut or budget_skip:
            what = [f"{budget_cut} cut short"] if budget_cut else []
            what += [f"{budget_skip} not searched"] if budget_skip else []
            out.append(f"{logsearch.READ_BUDGET // (1 << 20)} MiB read limit: " + ", ".join(what))
        left_out = sum(r.plan.left_out for r in done)
        if left_out:
            out.append(f"{logsearch.MAX_FILES}-file limit: {left_out} older file{'s' if left_out != 1 else ''} "
                       "left out")
        capped = [r for r in done if r.match_capped]
        if capped:
            out.append("match limit reached on " + ", ".join(f"{r.host} ({r.max_matches})" for r in capped)
                       + ": later matches not shown")
        late = sum(1 for r in done for f in r.files if f.status == "not reached" and not r.error)
        if late:
            out.append(f"{late} file{'s' if late != 1 else ''} not reached")
        return out

    def file_header(self, result: LogSearchResult, outcome: FileOutcome, width: int) -> Text:
        f = outcome.file
        head = Text()
        if len(self.hosts) > 1:
            head.append(f"{f.key.host}  ", style="bold")
        head.append(f.key.label, style=theme.accent())
        head.append(f"  {f.name}  ", style="bold")
        head.append(f"{f.stream}{'+stderr' if f.merged else ''}", style=theme.muted())
        if f.others:
            head.append(f" (+{f.others} job{'s' if f.others != 1 else ''})", style=theme.dim())
        count = f"  {outcome.hits} match{'es' if outcome.hits != 1 else ''}"
        if outcome.window_cut or outcome.budget_cut:
            count += f" in the last {humanize.size(outcome.searched)} of {humanize.size(outcome.size)}"
        room = width - head.cell_len - cell_len(count) - 2
        if room > 12:
            head.append("  ")
            head.append(humanize.truncate_left(f.path, room), style=theme.dim())
        head.append(count, style=theme.muted())
        return head

    def _fill(self) -> None:
        self.query_one("#ls-summary", Static).update(self.summary())
        menu = self.query_one(OptionList)
        keep = menu.highlighted_option.id if menu.highlighted_option is not None else None
        width = self.text_width()
        options: list[Option] = []
        self.hits = {}
        for result in self.ordered():
            by_file: dict[str, list[LogHit]] = {}
            for hit in result.hits:
                by_file.setdefault(hit.file.path, []).append(hit)
            for outcome in result.files:
                hits = by_file.get(outcome.file.path, [])
                if not hits:
                    continue
                options.append(Option(self.file_header(result, outcome, width), disabled=True))
                for hit in hits:
                    text, spans, before, after = logsearch.display_excerpt(hit, self.query_)
                    row = Text("  ")
                    row.append_text(fit_excerpt(text, spans, width - 2, cut_before=before, cut_after=after,
                                                hit_style=theme.pill()))
                    oid = f"h{len(self.hits)}"
                    self.hits[oid] = hit
                    options.append(Option(row, id=oid))
        if not options and not self.pending:
            options.append(Option(Text("no match", style=theme.dim()), disabled=True))
        menu.clear_options()
        menu.add_options(options)
        if keep is not None and keep in self.hits:
            menu.highlighted = menu.get_option_index(keep)
        elif self.hits and menu.highlighted is None:
            menu.highlighted = menu.get_option_index(next(iter(self.hits)))

    # ------------------------------------------------------------------ actions
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        hit = self.hits.get(event.option.id or "")
        if hit is not None:
            self.rf.open_log_hit(hit, self.query_)

    def action_cursor_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_again(self) -> None:
        self.dismiss("again")

    def action_close(self) -> None:
        self.dismiss(None)
