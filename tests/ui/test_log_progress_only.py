"""[WP3] A log whose tail is only a progress line rewritten with "\\r" (tqdm, no newline in the last 64 KiB): the pane
showed nothing. The follower now reads the log's start once (`log.head`) and shows it with a note pointing at the ↻
line; never when the tail has complete lines or the view starts at a search hit."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from runfleet.core.logs import HEAD_LINES, head_section, parse_head
from runfleet.models import JobKey
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.widgets.log_view import LogView

BIG = (140, 40)
KEY = JobKey("k2", "2332861")
BAR = " 34%|███▍      | 681934/2000000 [2:36:17<2:39:12, 138.22it/s]"


def _path(svc: Any) -> str:
    job = svc.store.job(KEY)
    assert job is not None and job.logs and job.logs[0].path
    return str(job.logs[0].path)


async def _open(app: Any, pilot: Any, **kw: Any) -> LogView:
    await pilot.pause()
    app.push_screen(LogScreen(KEY, **kw))
    for _ in range(20):
        await pilot.pause(0.05)
    return app.screen.query_one(LogView)


async def test_progress_only_tail_shows_the_start_of_the_log(make_app: Any) -> None:
    app, svc = make_app()
    path = _path(svc)
    svc.logs[path] = []                                  # no complete line in the tail
    svc.log_live[path] = BAR
    svc.log_head[path] = ["host n-40-01  CUDA_VISIBLE_DEVICES=0", "config: mt10  seed=0"]
    async with app.run_test(size=BIG) as pilot:
        view = await _open(app, pilot)
        assert svc.called("head_log") == [("k2", path)]                     # read once
        raw = list(view._raw)
        assert raw[:2] == svc.log_head[path]
        assert "only rewrites one progress line" in raw[2] and "↻" in raw[2]
        assert view.live == BAR and "↻" in str(view.query_one("#live").render())


async def test_a_tail_with_complete_lines_reads_no_head(make_app: Any) -> None:
    app, svc = make_app()
    path = _path(svc)
    svc.logs[path] = ["epoch 1 loss 0.31", "epoch 2 loss 0.29"]
    svc.log_live[path] = BAR
    async with app.run_test(size=BIG) as pilot:
        view = await _open(app, pilot)
        assert svc.called("head_log") == [] and list(view._raw) == svc.logs[path]


async def test_a_search_hit_view_never_adds_the_head(make_app: Any) -> None:
    app, svc = make_app()
    path = _path(svc)
    svc.logs[path] = []
    svc.log_live[path] = BAR
    async with app.run_test(size=BIG) as pilot:
        await _open(app, pilot)
        screen: Any = app.screen
        ref = svc.store.job(KEY).logs[0]
        before = len(svc.called("head_log"))
        screen.follower.start("k2", ref, start=0)          # what a search hit does: stream from its window
        for _ in range(20):
            await pilot.pause(0.05)
        assert len(svc.called("head_log")) == before


async def test_no_complete_line_anywhere_says_so(make_app: Any) -> None:
    app, svc = make_app()
    path = _path(svc)
    svc.logs[path] = []
    svc.log_live[path] = BAR
    svc.log_head[path] = []
    async with app.run_test(size=BIG) as pilot:
        view = await _open(app, pilot)
        raw = list(view._raw)
        assert len(raw) == 1 and raw[0].startswith("── no complete line")


def test_head_section_and_parse() -> None:
    s = head_section("/scratch/a b/run.out", 32768)
    assert s.name == "log.head" and "head -c 32768 -- '/scratch/a b/run.out'" in s.cmd
    now = datetime(2026, 10, 16, tzinfo=UTC)
    out = "3145728|1760000000\nstart\nconfig a=1\n" + "".join(f"\r {i}%|bar|" for i in range(30))
    peek = parse_head(out, "", 32768, now)
    assert peek.lines == ("start", "config a=1") and peek.size == 3145728 and peek.error is None
    many = "10|1760000000\n" + "".join(f"line {i}\n" for i in range(HEAD_LINES + 50))
    assert len(parse_head(many, "", 32768, now).lines) == HEAD_LINES
    missing = parse_head("", "stat: cannot statx '/x': No such file or directory", 32768, now)
    assert missing.lines == () and missing.error == "No such file"


def test_demo_plays_log_head() -> None:
    from types import SimpleNamespace

    from runfleet.demo import render
    from runfleet.demo.world import SimLog

    lg = SimLog("/p/run.out", 0.0)
    lg.write("start\nconfig\n" + "".join(f"\r{i}%" for i in range(100)), 5.0)
    out = render.log_head(SimpleNamespace(logs={"/p/run.out": lg}), path="/p/run.out", nbytes=12)  # type: ignore[arg-type]
    assert out.stdout == f"{lg.size}|5\nstart\nconfig"
