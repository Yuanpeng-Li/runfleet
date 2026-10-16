"""[WP3] Search logs… (§4.7.1, §7.10): the prompt, the results with honest caps, CJK-safe excerpts, and the log
viewer opened on the hit's line from the hit's absolute window (FakeService runs the real plan, the demo's play of the
`log.search` script and the real parser)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from rich.cells import cell_len
from textual.content import Content
from textual.widgets import Checkbox, Input, OptionList, Static

from runfleet.core import logsearch
from runfleet.core.logsearch import LogQuery
from runfleet.models import JobKey
from runfleet.ui import theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.commands import RunFleetCommands
from runfleet.ui.screens.log_search import LogSearchScreen, SearchPromptScreen, fit_excerpt
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.widgets.log_view import LogView
from tests.ui.fakes import FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (140, 40)
K2_JOB = JobKey("k2", "2332861")
K2_LOG = "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"


def text(app: RunFleetApp, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).render())


def rows(app: RunFleetApp) -> list[str]:
    menu = app.screen.query_one(OptionList)
    return [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]


async def settle(pilot: Any, n: int = 6) -> None:
    for _ in range(n):
        await pilot.pause()


async def search(app: RunFleetApp, pilot: Any, pattern: str, *, regex: bool = False) -> None:
    app.action_search_logs()
    await pilot.pause()
    assert isinstance(app.screen, SearchPromptScreen)
    app.screen.query_one(Input).value = pattern
    if regex:
        app.screen.query_one("#ls-regex", Checkbox).value = True
    await pilot.pause()
    app.screen.query_one(Input).focus()
    await pilot.press("enter")
    await settle(pilot)


def big_log(service: FakeService, n: int = 6000) -> list[str]:
    """A ~380 KiB log: only its last 256 KiB are searched, so the window starts mid-file (mid-line)."""
    lines = [f"[2026-09-07 05:{i // 60 % 60:02d}:{i % 60:02d}] step {i:05d}  train_loss 0.{i % 9973:04d}  "
             f"tokens/s 41{i % 97:02d}" for i in range(n)]
    lines[5000] = "[2026-09-07 06:23:20] step 05000  损失 is NaN — CUDA error: an illegal memory access"
    service.logs[K2_LOG] = lines
    return lines


def landed(view: LogView) -> str:
    """The display line a landing names in its note ("· line N") and the next one (a wrap); it must be on screen."""
    m = re.search(r"· line (\d+)", view.search_note or "")
    assert m is not None, view.search_note
    y = int(m.group(1)) - 1
    top = int(view.richlog.scroll_y)
    assert top <= y < top + view.richlog.scrollable_content_region.height, (top, y)
    return "".join(strip.text.rstrip() + " " for strip in view.richlog.lines[y:y + 2])


# ============================================================================================ pure
def test_fit_excerpt_keeps_the_match_in_view_in_cells() -> None:
    plain = "x" * 300 + "needle" + "y" * 300
    out = fit_excerpt(plain, [(300, 306)], 50, hit_style="bold")
    assert out.cell_len <= 50 and "needle" in out.plain and out.plain.startswith("…") and out.plain.endswith("…")
    assert any(s.style == "bold" and out.plain[s.start:s.end] == "needle" for s in out.spans)
    assert out.cell_len == 50 and 12 <= out.plain.index("needle") <= 18   # the row is filled; the match a third in
    cjk = "训练损失" * 30 + "爆炸" + "日志" * 40
    at = cjk.index("爆炸")
    for width in (20, 21, 37, 80):
        got = fit_excerpt(cjk, [(at, at + 2)], width, cut_before=True, cut_after=True)
        assert got.cell_len <= width and "爆炸" in got.plain, width
        assert got.cell_len >= width - 1, width                           # only a wide char at the edge is lost
        assert cell_len(got.plain) == got.cell_len                        # no half wide character anywhere
    short = fit_excerpt("loss=nan", [(0, 8)], 40, cut_before=True)
    assert short.plain == "…loss=nan"                                     # the context window cut the line there
    assert fit_excerpt("a" * 10, [], 4).cell_len <= 4


# ============================================================================================ prompt
async def test_ctrl_f_in_jobs_opens_the_prompt_with_the_scope(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.go_main("jobs")
        await pilot.pause()
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert isinstance(app.screen, SearchPromptScreen)
        scope = text(app, "#ls-scope")
        assert "k2 11 jobs · 11 files" in scope and "lynx needs auth: not searched" in scope
        assert "the last 256 KiB of each log · ≤ 20 files and 1 MiB per host · ≤ 200 matches" in scope
        await pilot.press("enter")                                         # nothing typed: refused, stays open
        await pilot.pause()
        assert isinstance(app.screen, SearchPromptScreen) and "type a pattern" in text(app, "#ls-note")
        await pilot.press(*"err(")
        app.screen.query_one("#ls-regex", Checkbox).value = True
        await pilot.pause()
        assert "unbalanced parentheses" in text(app, "#ls-note")
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SearchPromptScreen) and svc.called("search_logs") == []
        app.set_scope("k2")
        app.action_search_logs()
        await pilot.pause()
        assert "vega" not in text(app, "#ls-scope")                        # the host scope (key 1) narrows it


async def test_palette_offers_search_logs(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=BIG):
        names = {name: help_text for name, _cb, help_text in RunFleetCommands(app.screen)._commands()}  # type: ignore[arg-type]
        assert "Search logs…" in names and "256 KiB" in names["Search logs…"]


# ============================================================================================ results
async def test_results_are_honest_about_what_was_read(make_app: MakeApp) -> None:
    app, svc = make_app()
    svc.search_errors["vega"] = "ssh: connect to host timed out"
    svc.missing_logs.add(K2_LOG)
    async with app.run_test(size=BIG) as pilot:
        await search(app, pilot, "EPOCH 4")
        assert isinstance(app.screen, LogSearchScreen)
        calls = svc.called("search_logs")
        assert sorted(c[0] for c in calls) == ["k2", "labstation", "lyra", "vega"]   # lynx needs auth: no probe
        assert {c[2] for c in calls} == {50} and calls[0][1] == LogQuery("EPOCH 4")    # 200 split over 4 hosts
        summary = text(app, "#ls-summary")
        assert "searched 16 files on k2, lyra, labstation" in summary and "matches" in summary
        assert "capped: match limit reached on k2 (50): later matches not shown" in summary
        assert "vega: ssh: connect to host timed out" in summary and "lynx: needs auth — not searched" in summary
        assert "not readable: 1 missing" in summary
        shown = rows(app)
        assert shown[0].startswith("k2  2332862  orb-swp--train--cifar-const--s43--a02  stdout+stderr")
        assert "5 matches" in shown[0] and "epoch 40/300" in shown[1]
        menu = app.screen.query_one(OptionList)
        first = menu.get_option_at_index(1).prompt
        assert any(first.plain[s.start:s.end].lower() == "epoch 4" and s.style == theme.pill()   # type: ignore[union-attr]
                   for s in first.spans)                                   # type: ignore[union-attr]
        assert menu.highlighted == 1                                       # the first hit, not the file header
        app.set_theme_choice("light")                                     # cached rows are rebuilt in new colours
        await pilot.pause()
        again = app.screen.query_one(OptionList).get_option_at_index(1).prompt
        assert any(s.style == theme.pill() for s in again.spans)           # type: ignore[union-attr]


async def test_searching_state_and_nothing_found(make_app: MakeApp) -> None:
    import asyncio

    app, svc = make_app()
    svc.search_gate = asyncio.Event()
    async with app.run_test(size=BIG) as pilot:
        app.set_scope("labstation")
        await search(app, pilot, "no such text anywhere")
        assert "searching labstation …" in text(app, "#ls-summary")
        svc.search_gate.set()
        await settle(pilot)
        summary = text(app, "#ls-summary")
        assert "searched 1 file on labstation" in summary and "0 matches" in summary
        assert "nothing capped: every file in scope was searched in full" in summary
        assert rows(app) == ["no match"]
        await pilot.press("slash")                                        # edit the search: the prompt, prefilled
        await pilot.pause()
        assert isinstance(app.screen, SearchPromptScreen)
        assert app.screen.query_one(Input).value == "no such text anywhere"


async def test_a_pattern_that_looks_like_markup_is_shown_as_typed(make_app: MakeApp) -> None:
    """`[/x]`, `[bold]` are plain text to search for: the results' border title shows them as typed (Textual parses a
    str title as markup: a stray closing tag raised MarkupError and took the app down)."""
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        app.set_scope("labstation")
        await search(app, pilot, "[/x] [bold]loss[/]")
        await settle(pilot)
        assert isinstance(app.screen, LogSearchScreen)
        title = Content.from_markup(str(app.screen.query_one("#ls-box").border_title))
        assert "“[/x] [bold]loss[/]”" in title.plain and not title.spans


async def test_enter_lands_on_the_hit_line_from_its_window(make_app: MakeApp) -> None:
    app, svc = make_app()
    lines = big_log(svc)
    async with app.run_test(size=BIG) as pilot:
        app.set_scope("k2")
        await search(app, pilot, "illegal memory access")
        results = app.screen
        assert isinstance(results, LogSearchScreen)
        hit = next(iter(results.hits.values()))
        size = len(svc.log_bytes(K2_LOG))
        assert hit.file.path == K2_LOG and (hit.start, hit.size) == (size - logsearch.PER_FILE, size)
        assert "searched in full" not in text(app, "#ls-summary")
        assert "only the last 256 KiB of 1 file" in text(app, "#ls-summary")
        assert "in the last 256.0K of" in rows(app)[0] or "in the last 256" in rows(app)[0]
        await pilot.press("enter")
        await settle(pilot, 10)
        viewer = app.screen
        assert isinstance(viewer, LogScreen) and svc.stream_starts == [hit.start]
        view = viewer.query_one(LogView)
        assert view.search_note is not None and view.search_note.startswith("search hit “illegal memory access”")
        assert "illegal memory access" in landed(view)
        assert not view.follow                                            # paused on the hit
        assert lines[5000] in view._raw
        await pilot.press("escape", "escape")                              # note, then back to the results
        await pilot.pause()
        assert app.screen is results and rows(app)


async def test_landing_survives_growth_and_says_when_the_file_changed(make_app: MakeApp) -> None:
    app, svc = make_app()
    lines = big_log(svc)
    async with app.run_test(size=BIG) as pilot:
        app.set_scope("k2")
        await search(app, pilot, "illegal memory access")
        lines += [f"later line {i}" for i in range(3000)]                 # the job wrote on: the line number holds
        await pilot.press("enter")
        await settle(pilot, 10)
        view = app.screen.query_one(LogView)
        assert view.search_note is not None and "changed" not in view.search_note
        assert "illegal memory access" in landed(view)
        await pilot.press("escape", "escape")
        await pilot.pause()
        lines.insert(4990, "an extra line")                                # the file was rewritten: line moved
        await pilot.press("enter")
        await settle(pilot, 10)
        view = app.screen.query_one(LogView)
        assert view.search_note is not None and "the file changed since the search" in view.search_note
        assert "illegal memory access" in landed(view)
