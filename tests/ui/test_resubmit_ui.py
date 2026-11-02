"""Resubmissions in the TUI job surfaces (§5.2.12, ``util.lineage``): the `is:` filter term and the per-host link
cache, the superseded mark in the Jobs table, the "resubmitted as … / resumes …" lines of the job detail and the
preview, `n` / `N` between attempts with a crumb back, the support bundle and PreemptMode in the cause wording."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

from rich.text import Text
from textual.widgets import Static

from runfleet.models import JobKey, JobState
from runfleet.ui import everyday, query, render, theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.job_detail import JobDetailScreen, cause_line, job_box_lines, link_rows, title_line
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.job_preview import cause_text
from runfleet.ui.widgets.job_table import JobTable, RowContext, name_badges, node_reason
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.util.lineage import EMPTY, link
from runfleet.util.logscan import cause_of
from tests.ui.fakes import T0, FakeService, slurm_job

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (140, 40)
NAME = "orb-swp--train--places-cosinewarm--s44--a01"
H = 3600


def _chain() -> list[Any]:
    """lyra-like: PREEMPTED, resubmitted 13 min later, which COMPLETED (the PreemptMode CANCEL case)."""
    old = slurm_job("lyra", "31354789", NAME, JobState.PREEMPTED, elapsed=timedelta(seconds=64), ended_ago=18 * H,
                    submit_ago=19 * H, partition="free-gpuxl")
    new = slurm_job("lyra", "31358335", NAME, JobState.COMPLETED, elapsed=timedelta(hours=10), ended_ago=7 * H,
                    submit_ago=18 * H - 13 * 60, partition="free-gpuxl")
    return [old, new]


def _fff_cd() -> list[Any]:
    """F → F → F → CD (lyra 31327454 …)."""
    jobs = []
    for i, (jid, state) in enumerate((("31327454", JobState.FAILED), ("31327620", JobState.FAILED),
                                      ("31327915", JobState.FAILED), ("31327999", JobState.COMPLETED))):
        jobs.append(slurm_job("lyra", jid, "orb-swp--train--warmup--s49--a01", state, elapsed=timedelta(minutes=2),
                              ended_ago=(10 - 2 * i) * H, submit_ago=(10 - 2 * i) * H + 300, exit_code=2))
    return jobs


def _ctx(**kw: Any) -> RowContext:
    base: dict[str, Any] = dict(now=T0, host_colors={}, problem_hosts={}, stale_hosts=frozenset(),
                                marked=frozenset(), thresholds=(10, 75))
    base.update(kw)
    return RowContext(**base)


# ---------------------------------------------------------------- query: `is:` and the link cache
def test_is_term_parses_known_words_only() -> None:
    q = query.parse("is:resubmitted !is:resumes is:superseded is:foo")
    kinds = [(t.kind, t.values, t.negate) for t in q.terms]
    assert kinds[:3] == [("is", ("resubmitted",), False), ("is", ("resumes",), True), ("is", ("resubmitted",), False)]
    assert kinds[3] == ("text", ("is:foo",), False)          # an unknown word is plain text, as an unknown key is


def test_is_term_matches_the_links() -> None:
    old, new = _chain()
    links = link([old, new])
    assert query.matches(query.parse("is:resubmitted"), old, links=links)
    assert not query.matches(query.parse("is:resubmitted"), new, links=links)
    assert query.matches(query.parse("is:resumes"), new, links=links)
    assert query.matches(query.parse("!is:resubmitted"), new, links=links)
    assert not query.matches(query.parse("is:resubmitted"), old)        # no links given: nothing is linked
    assert query.matches(query.parse("!is:resubmitted"), old)


def test_lineage_cache_relinks_only_a_changed_host() -> None:
    old, new = _chain()
    k2 = slurm_job("k2", "2330539", "lmft", JobState.FAILED, elapsed=timedelta(minutes=5), ended_ago=H)
    hosts = {"lyra": SimpleNamespace(snapshot=SimpleNamespace(jobs={old.key.id: old, new.key.id: new})),
             "k2": SimpleNamespace(snapshot=SimpleNamespace(jobs={k2.key.id: k2})),
             "lynx": SimpleNamespace(snapshot=None)}
    cache = query.LineageCache()
    first = cache.get(hosts)
    assert first.next_of(old.key) == new and first.prev_of(new.key) == old
    assert cache.get(hosts) is first                            # nothing replaced: the same links, no relink
    hosts["k2"] = SimpleNamespace(snapshot=SimpleNamespace(jobs={k2.key.id: k2}))
    again = cache.get(hosts)
    assert again.next_of(old.key) == new
    del hosts["lyra"]                                           # a host leaves the fleet: its links go
    assert cache.get(hosts) is EMPTY


# ---------------------------------------------------------------- render helpers
def test_resubmitted_and_resumes_text() -> None:
    old, new = _chain()
    links = link([old, new])
    text = render.resubmitted_text(old, links, T0)
    assert text is not None and text.plain == "resubmitted as 31358335 · done 7h ago"
    assert theme.error() not in {str(s.style) for s in text.spans if s.style}            # reads as resolved
    clicks = [s.style.meta.get("@click") for s in text.spans if hasattr(s.style, "meta") and s.style.meta]
    assert clicks == ["app.open_job('lyra:31358335')"]
    resumes = render.resumes_text(new, links, T0)
    assert resumes is not None and resumes.plain == "resumes 31354789 preempted 18h ago"
    assert render.resubmitted_text(new, links, T0) is None and render.resumes_text(old, links, T0) is None
    assert render.chain_plain(new, links) == "31354789 PR → 31358335 CD"


def test_a_longer_chain_names_the_latest() -> None:
    jobs = _fff_cd()
    links = link(jobs)
    text = render.resubmitted_text(jobs[0], links, T0)
    assert text is not None
    assert text.plain == "resubmitted as 31327620 · failed 8h ago → latest 31327999 done"
    middle = render.lineage_line(jobs[1], links, T0)
    assert middle is not None and "resubmitted as 31327915" in middle.plain and "resumes 31327454" in middle.plain
    assert render.chain_plain(jobs[2], links) == "31327454 F → 31327620 F → 31327915 F → 31327999 CD"


def test_link_when_of_running_and_pending() -> None:
    running = slurm_job("lyra", "1", NAME, JobState.RUNNING, elapsed=timedelta(hours=2, minutes=3))
    pending = slurm_job("lyra", "2", NAME, JobState.PENDING, submit_ago=13 * 60)
    assert render.link_when(running, T0) == "for 2h03m"
    assert render.link_when(pending, T0) == "for 13m"
    assert render.link_state(pending) == "pending"


# ---------------------------------------------------------------- jobs table
def test_superseded_row_is_marked_not_red() -> None:
    old, new = _chain()
    links = link([old, new])
    assert name_badges(old, False, superseded=True).plain == "⤳ "
    assert name_badges(old, True, superseded=True).plain == "◉ ⤳ "
    cell = node_reason(old, _ctx(lineage=links))
    assert cell.plain == "preempted → 31358335 CD"
    assert all(str(s.style) in (theme.muted(), "") for s in cell.spans)
    jobs = _fff_cd()
    failed = node_reason(jobs[0], _ctx(lineage=link(jobs)))
    assert failed.plain == "exit 2 → 31327999 CD"                 # the newest of the chain, not the next failure
    assert theme.state_style(JobState.FAILED) not in {str(s.style) for s in failed.spans}
    plain = node_reason(jobs[3], _ctx(lineage=link(jobs)))
    assert "→" not in plain.plain
    unlinked = node_reason(jobs[0], _ctx())
    assert "→" not in unlinked.plain and theme.state_style(JobState.FAILED) in str(unlinked.style) + "".join(
        str(s.style) for s in unlinked.spans)


# ---------------------------------------------------------------- job detail renderers
def test_cause_line_of_a_superseded_job_keeps_the_chip() -> None:
    old, new = _chain()
    links = link([old, new])
    cause = cause_of(old, None, (), preempt_mode="CANCEL")
    assert cause is not None and "killed (PreemptMode CANCEL)" in cause.text
    plain = cause_line(old, cause, 120)
    assert plain is not None and "killed (PreemptMode CANCEL)" in plain.plain
    line = cause_line(old, cause, 120, links=links, now=T0)
    assert line is not None
    assert line.plain.rstrip() == " PREEMPTED  resubmitted as 31358335 · done 7h ago"
    assert "PreemptMode" not in line.plain and "L: log" not in line.plain
    assert cause_line(new, None, 120, links=links, now=T0) is None              # boxes: the Job box says it
    compact = cause_line(new, None, 120, links=links, now=T0, resumes=True)
    assert compact is not None and compact.plain.rstrip() == "resumes 31354789 preempted 18h ago"


def test_job_box_rows_and_title_badge() -> None:
    jobs = _fff_cd()
    links = link(jobs)
    rows = [r.plain for r in link_rows(jobs[1], links, T0)]
    assert rows == ["resubmit  31327915 failed · 6h ago → latest 31327999 done",
                    "resumes   31327454 failed · 10h ago"]
    box = [r.plain for r in job_box_lines(jobs[0], None, T0, 70, links)]
    i = next(i for i, r in enumerate(box) if r.startswith("restarts"))
    assert box[i + 1].startswith("resubmit  31327620 failed · 8h ago → latest 31327999 done")
    assert not any(r.plain.startswith(("resubmit", "resumes")) for r in job_box_lines(jobs[0], None, T0, 70))
    assert "⤳ orb-swp" in title_line(jobs[0], T0, 120, superseded=True).plain


# ---------------------------------------------------------------- preview, bundle, PreemptMode
def test_preview_cause_text_with_links() -> None:
    old, new = _chain()
    links = link([old, new])
    text = cause_text(old, None, 100, preempt_mode="REQUEUE", links=links, now=T0)
    assert text is not None and text.plain.rstrip() == " PREEMPTED  resubmitted as 31358335 · done 7h ago"
    alone = cause_text(old, None, 100, preempt_mode="REQUEUE")
    assert alone is not None and "not requeued" in alone.plain


def test_support_bundle_names_the_links() -> None:
    old, new = _chain()
    links = link([old, new])
    md = everyday.support_bundle(old, None, cluster="lyra", log_lines=[], now=T0, preempt_mode="CANCEL", links=links)
    assert "- **resubmitted as:** 31358335 · COMPLETED" in md and "PreemptMode CANCEL" in md
    md = everyday.support_bundle(new, None, cluster="lyra", log_lines=[], now=T0, links=links)
    assert "- **resumes:** 31354789 · PREEMPTED" in md


# ---------------------------------------------------------------- the app
FAILED = JobKey("k2", "2330539")
NEXT = JobKey("k2", "2331001")


def _resubmit(svc: FakeService, state: JobState = JobState.COMPLETED) -> None:
    """The k2 failure of the fake world, resubmitted under the same name 10 minutes after it ended."""
    svc.put_job(slurm_job("k2", NEXT.id, "lmft-x0-lr1e-5", state, elapsed=timedelta(hours=1),
                          ended_ago=2 * 86400 - 2 * H if state.is_terminal else None, submit_ago=2 * 86400 - 600,
                          node="n-38-01", gpu_type="nvidia_h200_nvl"))


async def _settle(pilot: Any, rounds: int = 6) -> None:
    for _ in range(rounds):
        await pilot.pause(0.05)


def _plain(app: RunFleetApp, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).render())


async def test_detail_says_resubmitted_and_n_opens_the_next_attempt(make_app: MakeApp) -> None:
    app, svc = make_app()
    _resubmit(svc, JobState.RUNNING)
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        app.open_job(FAILED)
        await _settle(pilot)
        old_screen = app.screen
        assert isinstance(old_screen, JobDetailScreen)
        cause = _plain(app, "#cause")
        assert "EXIT 1" in cause and "resubmitted as 2331001 · running for 1h" in cause
        tooltip = old_screen.query_one("#cause", Static).tooltip
        assert isinstance(tooltip, str) and "resubmissions: 2330539 F → 2331001 R" in tooltip
        assert "resubmit  2331001 running" in _plain(app, "#job-box")
        assert "⤳ lmft" in _plain(app, "#title")
        _resubmit(svc, JobState.COMPLETED)                     # the successor ends: the line follows it
        await _settle(pilot)
        assert "resubmitted as 2331001 · done" in _plain(app, "#cause")
        await pilot.press("n")
        await _settle(pilot)
        new_screen = app.screen
        assert isinstance(new_screen, JobDetailScreen) and new_screen.key == NEXT and new_screen.via == FAILED
        assert "resumes   2330539 failed" in _plain(app, "#job-box")
        header = new_screen.query_one(FleetHeader)
        assert ("2330539", ("job", "k2", "2330539")) in header.crumbs
        await pilot.press("N")                                 # back to the attempt it resumes (a new screen on top)
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == FAILED
        app.action_crumb("job", "k2", NEXT.id)
        await _settle(pilot)
        assert app.screen is new_screen
        app.action_crumb("job", "k2", FAILED.id)              # the crumb leads back through the old one
        await _settle(pilot)
        assert app.screen is old_screen


async def test_n_without_a_link_says_so_honestly(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG, notifications=True) as pilot:
        await pilot.pause()
        app.open_job(FAILED)
        await _settle(pilot)
        await pilot.press("n")
        await pilot.press("N")
        await _settle(pilot)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == FAILED
        messages = [n.message for n in app._notifications]
        assert any("not resubmitted as far as RunFleet sees (history covers the last 3d)" in m for m in messages)
        assert any("resumes no earlier job as far as RunFleet sees" in m for m in messages)
        app.open_job(JobKey("k2", "2332861"))                  # running: nothing to carry on
        await _settle(pilot)
        await pilot.press("n")
        await _settle(pilot)
        assert any("did not end badly" in n.message for n in app._notifications)


async def test_jobs_table_marks_and_filters_superseded_rows(make_app: MakeApp) -> None:
    app, svc = make_app()
    _resubmit(svc)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.go_main("jobs")
        await _settle(pilot)
        view = app.screen.query_one(JobsView)
        view.set_preset("failed")
        await _settle(pilot)
        table = view.query_one(JobTable)
        assert str(FAILED) in table.order                      # the failed preset keeps it, marked
        cells = {k: "".join(c.plain for c in v) for k, v in table._cells.items()}
        assert "⤳ lmft" in cells[str(FAILED)] and "→ 2331001 CD" in cells[str(FAILED)]
        view.set_filter("!is:resubmitted")
        await _settle(pilot)
        assert str(FAILED) not in table.order
        view.set_filter("is:resumes")
        view.set_preset("all")
        await _settle(pilot)
        assert table.order == [str(NEXT)]


async def test_preview_line_mode_and_full_mode_show_the_link(make_app: MakeApp) -> None:
    app, svc = make_app()
    _resubmit(svc)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.go_main("jobs")
        await _settle(pilot)
        view = app.screen.query_one(JobsView)
        view.set_preset("all")
        view.set_filter("lmft")
        await _settle(pilot)
        table = view.query_one(JobTable)
        table.move_cursor(row=table.order.index(str(FAILED)))
        await _settle(pilot)
        assert "resubmitted as 2331001 · done" in str(view.preview.render())
        table.move_cursor(row=table.order.index(str(NEXT)))
        await _settle(pilot)
        assert "resumes 2330539 failed" in str(view.preview.render())
        view.preview.mode = "line"
        view._update_preview()
        assert "resumes 2330539" in str(view.preview.render())


def test_unrelated_render_is_unchanged_without_links() -> None:
    job = dataclasses.replace(_chain()[0])
    assert render.lineage_line(job, EMPTY, T0) is None
    assert isinstance(cause_line(job, cause_of(job), 100), Text)


async def test_palette_only_resubmitted_shows_every_state(make_app: MakeApp) -> None:
    """The bad endings `is:resubmitted` keeps are ended jobs: the palette command switches the preset to all, so it is
    never an empty table on `active` / `queued` (or once they left `fresh`)."""
    app, svc = make_app()
    _resubmit(svc)
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.go_main("jobs")
        await _settle(pilot)
        view = app.screen.query_one(JobsView)
        view.set_preset("active")
        await _settle(pilot)
        app.jobs_setting("filter_all", "is:resubmitted")
        await _settle(pilot)
        assert view.preset == "all" and view.query_one(JobTable).order == [str(FAILED)]
        app.jobs_setting("filter", "!is:resubmitted")              # hide: keeps the preset
        await _settle(pilot)
        assert view.preset == "all" and str(FAILED) not in view.query_one(JobTable).order
