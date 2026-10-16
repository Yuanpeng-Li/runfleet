"""[WP3] ui/keymap.py: single source of bindings, documented, unique, no chords; generated help/hints/docs."""

from __future__ import annotations

import inspect

import pytest

from runfleet.ui import keymap
from runfleet.ui.app import RunFleetApp
from runfleet.ui.screens.action_menu import ActionLogScreen, ActionMenuScreen
from runfleet.ui.screens.attention import AttentionScreen
from runfleet.ui.screens.confirm import AttachConfirmScreen, ConfirmScreen
from runfleet.ui.screens.forwards import ForwardsScreen, PortPromptScreen
from runfleet.ui.screens.gpu_watch import WatchesScreen, WatchPromptScreen
from runfleet.ui.screens.help import HelpScreen
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.host_error import HostErrorScreen
from runfleet.ui.screens.job_detail import JobDetailScreen
from runfleet.ui.screens.log_search import LogSearchScreen, SearchPromptScreen
from runfleet.ui.screens.log_viewer import LogScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.screens.save_view import SaveViewScreen
from runfleet.ui.screens.notifications import DigestScreen, NotificationsScreen
from runfleet.ui.screens.settings import SettingsScreen
from runfleet.ui.widgets.fleet_view import FleetView
from runfleet.ui.widgets.gpus_view import GPUsView
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.util import platform

# context -> classes whose actions resolve the context's un-prefixed action names
OWNERS: dict[str, tuple[type, ...]] = {
    "global": (RunFleetApp,), "main": (MainScreen,), "fleet": (FleetView,), "jobs": (JobsView,),
    "gpus": (GPUsView,), "job_detail": (JobDetailScreen,), "host_detail": (HostDetailScreen,),
    "host_ws": (HostDetailScreen,), "host_cluster": (HostDetailScreen,), "log": (LogScreen,),
    "confirm": (ConfirmScreen,), "attach_confirm": (AttachConfirmScreen,), "action_menu": (ActionMenuScreen,), "help": (HelpScreen,),
    "host_error": (HostErrorScreen,), "modal": (ActionLogScreen,), "settings": (SettingsScreen,),
    "notifications": (NotificationsScreen,), "digest": (DigestScreen,), "attention": (AttentionScreen,),
    "forwards": (ForwardsScreen,), "prompt": (PortPromptScreen, SaveViewScreen, WatchPromptScreen),
    "watches": (WatchesScreen,), "search_prompt": (SearchPromptScreen,), "log_hits": (LogSearchScreen,),
}


def _action_name(action: str) -> tuple[str, str]:
    target, _, rest = action.partition(".") if action.split("(")[0].count(".") else ("", "", action)
    return target, rest.split("(")[0]


@pytest.mark.parametrize("context", list(keymap.KEYMAP))
def test_every_binding_is_documented(context: str) -> None:
    for k in keymap.keys_of(context):
        assert k.description.strip(), f"{context}: {k.keys} has no description"
        assert k.group in keymap.GROUP_TITLES


@pytest.mark.parametrize("context", list(keymap.KEYMAP))
def test_no_duplicate_key_per_context(context: str) -> None:
    seen: dict[str, str] = {}
    for k in keymap.keys_of(context):
        for key in k.keys:
            assert key not in seen, f"{context}: {key!r} bound to {seen[key]} and {k.action}"
            seen[key] = k.action


def test_no_prefix_chords() -> None:
    for context, keys in keymap.KEYMAP.items():
        for k in keys:
            for key in k.keys:
                assert " " not in key and "," not in key, f"{context}: chord {key!r}"


@pytest.mark.parametrize("context", list(keymap.KEYMAP))
def test_actions_exist_on_their_owner(context: str) -> None:
    for k in keymap.keys_of(context):
        if not k.bind:
            continue
        target, name = _action_name(k.action)
        owners = (RunFleetApp,) if target == "app" else (MainScreen,) if target == "screen" and context == "main" \
            else OWNERS[context]
        if name in ("command_palette", "quit", "suspend_process"):
            continue                                       # Textual built-ins on App
        assert any(hasattr(o, f"action_{name}") for o in owners), f"{context}: no action_{name} for {k.action}"


def test_bindings_cover_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)       # the POSIX behaviour (Windows: test_windows_*.py)
    for context, keys in keymap.KEYMAP.items():
        bound = {b.key for b in keymap.bindings(context)}   # type: ignore[union-attr]
        for k in keys:
            if k.bind:
                assert set(k.keys) <= bound, context


def test_ctrl_c_and_ctrl_z_are_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)       # the POSIX behaviour (Windows: test_windows_*.py)
    prio = {b.key for b in keymap.bindings("global") if b.priority}  # type: ignore[union-attr]
    assert {"ctrl+c", "ctrl+z"} <= prio


def test_help_entries_contextual_and_overriding() -> None:
    entries = keymap.help_entries(["jobs", "main", "global"])
    flat = [k for rows in entries.values() for k, _d in rows]
    assert "j/k ↑/↓" in flat and "space" in flat and "?/F1" in flat
    assert "k" not in flat                                # summarised by "j/k"
    detail = keymap.help_entries(["job_detail", "global"])
    descs = {k: d for rows in detail.values() for k, d in rows}
    assert descs.get("r") == "refresh detail"
    assert all(d != "refresh hosts in scope (>= 3 s apart)" for d in descs.values())   # r overridden


def test_hints_hide_mutating_in_readonly() -> None:
    normal = [h[0] for h in keymap.hints(["jobs", "main", "global"])]
    ro = [h[0] for h in keymap.hints(["jobs", "main", "global"], readonly=True)]
    assert "^k" in normal and "^k" not in ro


def test_display_key() -> None:
    assert keymap.display_key("ctrl+k") == "^k"
    assert keymap.display_key("question_mark") == "?"
    assert keymap.display_key("x") == "x"


def test_markdown_lists_every_context() -> None:
    md = keymap.markdown()
    for title in keymap.CONTEXT_TITLES.values():
        assert f"| {title} |" in md
    assert "`^k`" in md


def test_widgets_take_bindings_from_keymap() -> None:
    for cls, ctx in ((JobsView, "jobs"), (FleetView, "fleet"), (GPUsView, "gpus"), (MainScreen, "main"),
                     (LogScreen, "log"), (JobDetailScreen, "job_detail"), (HelpScreen, "help"),
                     (ConfirmScreen, "confirm"), (AttachConfirmScreen, "attach_confirm"), (ActionMenuScreen, "action_menu"), (HostErrorScreen, "host_error"),
                     (SettingsScreen, "settings"), (NotificationsScreen, "notifications"), (DigestScreen, "digest"),
                     (AttentionScreen, "attention"), (WatchesScreen, "watches"), (WatchPromptScreen, "prompt")):
        assert cls.BINDINGS == keymap.bindings(ctx), cls
    assert inspect.getsource(RunFleetApp).count('keymap.bindings("global")') == 1


def test_hint_bar_shows_key_pairs_and_clicks_the_first_key() -> None:
    """Critic: footers read `[ tab`, `[ partition` and `G top/bottom` (g is top, G bottom + follow)."""
    from runfleet.ui.widgets.hintbar import HintBar, _textual_key

    jd = {(k, t) for k, t, _a in keymap.hints(["job_detail", "global"])}
    assert ("[/]", "tab") in jd and ("o/e", "stream") in jd
    assert ("[/]", "partition") in {(k, t) for k, t, _a in keymap.hints(["host_cluster", "global"])}
    assert ("g/G", "top/bottom") in {(k, t) for k, t, _a in keymap.hints(["log", "global"])}
    assert _textual_key(["job_detail", "global"], "[/]", "prev_tab") == "left_square_bracket"
    assert "[/] tab" in HintBar.build(["job_detail", "global"], 160).plain


def test_help_legend_uses_the_recent_window() -> None:
    from runfleet.ui.screens.help import legend

    assert "F 5/24h" in legend(["jobs", "global"], 100).plain
    assert "F 5/36h" in legend(["jobs", "global"], 100, "36h").plain and "3d" not in legend(["global"], 100).plain
