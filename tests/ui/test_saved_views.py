"""[WP3] Saved Jobs views (§7.6): name rules, tolerant ui.json reading, palette save / apply / delete, status line."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Input

import runfleet.ui.app as app_module
from runfleet.ui import saved_views
from runfleet.ui.app import RunFleetApp
from runfleet.ui.commands import RunFleetCommands
from runfleet.ui.saved_views import MAX_VIEWS, NAME_MAX, SavedView
from runfleet.ui.screens.save_view import SaveViewScreen
from runfleet.ui.widgets.filter_bar import FilterBar
from runfleet.ui.widgets.jobs_view import JobsView
from tests.ui.fakes import FakeService

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
BIG = (140, 35)


def palette(app: RunFleetApp) -> dict[str, tuple[Callable[[], Any], str]]:
    return {name: (cb, help_text) for name, cb, help_text in RunFleetCommands(app.screen)._commands()}  # type: ignore[arg-type]


def jobs(app: RunFleetApp) -> JobsView:
    return app.main_screen().jobs  # type: ignore[no-any-return]


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


# ---------------------------------------------------------------- pure rules
def test_name_rules() -> None:
    assert saved_views.name_problem("failed today") is None
    assert saved_views.name_problem("k2.gpu_sweep-3") is None
    assert saved_views.name_problem("x" * NAME_MAX) is None
    assert saved_views.name_problem("") == "type a name"
    assert "at most" in (saved_views.name_problem("x" * (NAME_MAX + 1)) or "")
    for bad in (" lead", "-dash", "semi;colon", "slash/x", "über", "tab\tx", "a\nb", "emoji ✓"):
        assert saved_views.name_problem(bad) is not None, bad
    assert saved_views.name_problem("two  spaces") is not None
    # review fix: a trailing space (ui.json by hand) made "gpu " a second view next to "gpu", unreachable from the
    # prompt (which strips the name)
    assert saved_views.name_problem("gpu ") is not None
    views = saved_views.load_views({"views": {"gpu": {}, "gpu ": {"preset": "failed"}}})
    assert list(views) == ["gpu"]


def test_load_ignores_bad_entries_and_keeps_good_ones() -> None:
    state: dict[str, Any] = {"views": {
        "failed": {"filter": "st:f", "preset": "failed", "sort": "submit", "reverse": True, "group": "host"},
        "partial": {"filter": "orb"},                                  # missing fields: Jobs view defaults
        "bad preset": {"preset": "weird"},
        "bad sort": {"sort": "colour"},
        "bad group": {"group": "user"},
        "bad reverse": {"reverse": "yes"},
        "bad filter": {"filter": 3},
        "multi line": {"filter": "a\nb"},
        "not a dict": ["fresh"],
        "bad/name": {"preset": "all"},
        "Partial": {"preset": "all"},                                  # same name ignoring case: one of them only
    }}
    views = saved_views.load_views(state)
    assert list(views) == ["failed", "Partial"]
    assert views["failed"] == SavedView("st:f", "failed", "submit", True, "host")
    del state["views"]["Partial"]
    views = saved_views.load_views(state)
    assert views["partial"] == SavedView("orb", "fresh", "host", False, "none")
    for junk in ({}, {"views": None}, {"views": []}, {"views": "x"}, {"views": {1: {}}}):
        assert saved_views.load_views(junk) == {}


def test_count_is_limited() -> None:
    state = {"views": {f"v{i:02d}": {} for i in range(MAX_VIEWS + 5)}}
    views = saved_views.load_views(state)
    assert len(views) == MAX_VIEWS
    assert saved_views.save_problem(views, "one more") is not None
    assert saved_views.save_problem(views, "V03") is None              # replacing one is fine at the limit
    assert saved_views.save_problem({}, "fine") is None


def test_writes_touch_only_the_named_entry() -> None:
    state = {"views": {"Gpu": {"preset": "active"}, "future": {"preset": "brand-new"}}}
    after = saved_views.with_view(state, "gpu", SavedView(preset="all"))
    assert after == {"future": {"preset": "brand-new"},
                     "gpu": {"filter": "", "preset": "all", "sort": "host", "reverse": False, "group": "none"}}
    assert saved_views.without_view(state, "GPU") == {"future": {"preset": "brand-new"}}
    assert saved_views.with_view({"views": "junk"}, "a", SavedView()) == {"a": SavedView().to_json()}


# ---------------------------------------------------------------- palette flows
async def test_save_apply_and_delete_from_the_palette(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = jobs(app)
        await pilot.press("f", "f", "f")                               # failed
        await pilot.press("o")                                         # next sort
        await pilot.press("z")                                         # group by host
        await pilot.press("slash", *"k2", "enter")
        await pilot.pause(0.3)
        saved = view.current_view()
        assert saved == SavedView("k2", "failed", "state", False, "host")
        assert "view:" not in view.status_text(140).plain

        names = palette(app)
        assert "Save view as…" in names and not any(n.startswith("View: ") for n in names)
        names["Save view as…"][0]()
        await pilot.pause()
        assert isinstance(app.screen, SaveViewScreen)
        assert "failed · sort state" in str(app.screen.query_one("#save-view-what").render())
        await pilot.press(*"k2 fails", "enter")
        await pilot.pause()
        assert not isinstance(app.screen, SaveViewScreen)
        assert app.ui_state["views"] == {"k2 fails": saved.to_json()}
        assert "view: k2 fails" in view.status_text(140).plain
        assert view.status_text(90).plain.startswith("k2 fails · ")

        # any change ends the name; going back does not bring it back
        await pilot.press("f")
        await pilot.pause()
        assert "k2 fails" not in view.status_text(140).plain
        await pilot.press("f", "f", "f", "f")
        await pilot.pause()
        assert view.current_view() == saved and "k2 fails" not in view.status_text(140).plain

        # reset everything, go to the Fleet view, apply the view from the palette
        view.clear_filter()
        view.set_preset("all")
        view.set_group("none")
        view.set_sort("name")
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.screen.view != "jobs"
        names = palette(app)
        assert names["View: k2 fails"][1] == "failed · sort state ▲ · group host · /k2"
        names["View: k2 fails"][0]()
        await pilot.pause(0.3)
        assert app.screen.view == "jobs"
        assert view.current_view() == saved and view.parsed.terms
        assert app.screen.query_one(FilterBar).value == "k2" and app.screen.query_one(FilterBar).display
        assert "view: k2 fails" in view.status_text(140).plain
        assert app.ui_state["active_view"] == "k2 fails" and app.ui_state["preset"] == "failed"
        rows = [view.table.rows_by_key[k] for k in view.table.order]
        assert rows and rows[0].key.startswith("§host:")               # grouped by host again

        palette(app)["Delete view: k2 fails"][0]()
        await pilot.pause()
        assert app.ui_state["views"] == {} and "k2 fails" not in view.status_text(140).plain
        assert not any("k2 fails" in n for n in palette(app))
        assert any("deleted view" in t for t in toasts(app))


async def test_palette_typing_applies_a_view(make_app: MakeApp) -> None:
    app, svc = make_app()
    app.ui_state["views"] = {"queued only": {"preset": "queued"}}
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press(*"View: queued")
        await pilot.pause(0.5)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert app.screen.view == "jobs" and jobs(app).preset == "queued"
        assert "view: queued only" in jobs(app).status_text(140).plain


async def test_prompt_refuses_bad_names_and_the_21st_view(make_app: MakeApp) -> None:
    app, svc = make_app()
    app.ui_state["views"] = {f"v{i:02d}": {} for i in range(MAX_VIEWS)}
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        app.prompt_save_view()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SaveViewScreen)
        await pilot.press("enter")                                     # empty
        await pilot.pause()
        assert app.screen is screen and "type a name" in str(screen.query_one("#save-view-note").render())
        screen.query_one(Input).value = "no/slash"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen and "letters, digits" in str(screen.query_one("#save-view-note").render())
        screen.query_one(Input).value = "brand new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen and f"{MAX_VIEWS} views saved already" in \
            str(screen.query_one("#save-view-note").render())
        screen.query_one(Input).value = "V05"                          # replacing one is allowed
        await pilot.pause()
        assert "replaces the saved view “v05”" in str(screen.query_one("#save-view-note").render())
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not screen
        assert "V05" in app.ui_state["views"] and "v05" not in app.ui_state["views"]
        assert len(app.ui_state["views"]) == MAX_VIEWS
        app.prompt_save_view()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SaveViewScreen)
        assert not app.save_view("one too many") and len(app.ui_state["views"]) == MAX_VIEWS


async def test_ui_json_round_trip_and_bad_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Views live in ui.json; a broken entry is ignored (and kept), the active name comes back after a restart."""
    target = tmp_path / "ui.json"
    target.write_text(json.dumps({"preset": "active", "sort": "host", "group": "none", "filter": "",
                                  "views": {"broken": {"preset": 7}, "Busy": {"preset": "active"}},
                                  "active_view": "busy"}), encoding="utf-8")
    monkeypatch.setattr(app_module, "ui_state_path", lambda: target)
    app = RunFleetApp(FakeService(), animations=False)  # type: ignore[arg-type]
    async with app.run_test(size=BIG) as pilot:
        await pilot.pause()
        view = jobs(app)
        assert view.active_view == "Busy"
        assert "view: Busy" in view.status_text(140).plain
        names = palette(app)
        assert "View: Busy" in names and "View: broken" not in names and "Delete view: broken" not in names
        await pilot.press("tab", "o")                                  # change the sort: the name goes
        await pilot.pause()
        assert view.active_view is None
        view.set_sort("host")
        assert app.save_view("sorted")
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["views"]["broken"] == {"preset": 7}                  # unreadable entries are left alone
    assert stored["views"]["sorted"]["preset"] == "active" and stored["active_view"] == "sorted"

    stored["preset"] = "all"                                           # settings no longer match: no name at start
    target.write_text(json.dumps(stored), encoding="utf-8")
    again = RunFleetApp(FakeService(), animations=False)  # type: ignore[arg-type]
    async with again.run_test(size=BIG) as pilot:
        await pilot.pause()
        assert jobs(again).active_view is None


async def test_applying_a_view_ends_the_digest_mode(make_app: MakeApp) -> None:
    app, svc = make_app()
    app.ui_state["views"] = {"all": {"preset": "all"}}
    async with app.run_test(size=BIG) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        view = jobs(app)
        view.set_ended_since(svc.clock.now())
        app.apply_saved_view("ALL")
        await pilot.pause()
        assert view.ended_since is None and view.active_view == "all"
        view.set_ended_since(svc.clock.now())                          # the digest's `f` is a change too
        await pilot.pause()
        assert view.active_view is None
        app.apply_saved_view("gone")
        await pilot.pause()
        assert any("no saved view" in t for t in toasts(app))
