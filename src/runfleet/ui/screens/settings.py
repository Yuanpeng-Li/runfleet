"""[WP3] Settings overlay (herdr-style): Hosts / General / About, saved to config.toml (§7.13).

A centered panel drawn as one Rich text, like herdr's settings popup: a bold `settings` title, a row of section tabs
(the active one an accent pill), a thin rule, a section heading with a one-line description, the choice list (the
selected row an accent bar with `▸`), a message line, a key hint line and two buttons. Tabs, rows, checkboxes, value
pills and buttons are clickable; the wheel moves the selection. Keys come from ``keymap`` context ``settings``.

* **Hosts** — every ``~/.ssh/config`` alias merged with the configured and the current ad-hoc hosts. ``[x]`` = saved
  in config.toml; checking saves the host and starts polling it at once, unchecking stops polling and removes it (a
  one-line confirm when it runs jobs of mine, has hand-tuned keys, or is not in this run; checking it again in the
  same session restores the removed table). ``t`` probes the alias read-only and shows the result inline, ``a`` adds
  a custom ssh target, ``p`` polls a saved host this run left out (``runfleet k2``: watch vega too), ``J``/``K``
  reorder (= card order and number keys), ``/`` filters. ``/`` and ``a`` switch to
  the input line inside the key handler itself, so keys that arrive in the same burst (fast typing over a slow link)
  are typed, never run as commands.
* **General** — the [general] settings that make sense to flip here, then a *notifications* group ([notify]:
  enabled, terminal channel, an events preset; the command hook stays config.toml-only and is only reported);
  applied live where possible, persisted.
* **About** — config file (``↵`` opens it in ``$EDITOR``), log, state dir, version, theme (``↵`` cycles
  auto / dark / light / ansi, live, kept in ui.json). On the version row ``↵`` asks
  GitHub for the newest version; when one is newer, ``↵`` and ``↵`` again quit, update this install and start
  RunFleet again (palette › Update RunFleet lands here, §8.6).

Everything goes through :class:`~runfleet.core.service.FleetAPI`; with ``config_path`` None (``--demo``) changes
last for this run only and the panel says so.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from runfleet.config import NOTIFY_EVENTS, HostConfig
from runfleet.errors import ConfigError, RunFleetError
from runfleet.models import ConnState, DetectReport, StateGroup
from runfleet.ui import keymap, theme
from runfleet.util import platform

SECTIONS: tuple[tuple[str, str], ...] = (("hosts", "Hosts"), ("general", "General"), ("about", "About"))
PANEL_W, PANEL_H = 76, 22          # herdr's popup size; shrinks to the terminal (width - 4, height - 2)
_NAME_SUB = re.compile(r"[^A-Za-z0-9_.\-]")


def _click(action: str) -> Style:
    """Hit area for a mouse click (an action of the screen). Not Textual's `@click`: that restyles the segment as a
    link (link colour over the accent pill), and herdr's panel has no link look either."""
    return Style(meta={"settings": action})


def display_path(path: Path | None) -> str:
    from runfleet import paths

    return "" if path is None else paths.display_path(path)


def host_name_for(alias: str) -> str:
    """Config name for an ssh target (`me@box` -> `me_box`), like ad-hoc hosts on the command line (a name Windows
    cannot use as a file name gets a `_`)."""
    return platform.safe_file_name(_NAME_SUB.sub("_", alias) or "host")


def editor_argv(path: Path) -> list[str]:
    """`$VISUAL`, `$EDITOR`, else `vi` (Windows: `notepad`)."""
    default = "notepad" if platform.IS_WINDOWS else "vi"
    spec = os.environ.get("VISUAL") or os.environ.get("EDITOR") or default
    try:
        argv = platform.split_command(spec)
    except ValueError:
        argv = [default]
    return [*(argv or [default]), str(path)]


# ======================================================================================= rows
@dataclass
class HostRow:
    name: str
    ssh: str
    saved: bool          # a [[host]] table in config.toml (or the in-memory list in --demo)
    running: bool        # in the store this run (polled unless disabled)


@dataclass(frozen=True)
class Setting:
    key: str                                   # field of the table's settings (GeneralSettings, NotifySettings)
    label: str
    options: tuple[tuple[str, object], ...]    # (label, value)
    note: str                                  # right-hand note
    help: str                                  # description line while selected
    table: str = "general"                     # config.toml table: "general" | "notify"


def _events(*names: str) -> tuple[str, ...]:
    """An events preset in NOTIFY_EVENTS order (the row compares the configured list in the same order)."""
    return tuple(e for e in NOTIFY_EVENTS if e in names)


FAILURES = _events("ended_bad", "restarted", "time_low")         # = the NotifySettings default
GROUPS = {"notify": "notifications"}           # heading above the rows of a table other than [general]

GENERAL: tuple[Setting, ...] = (
    Setting("default_view", "default view", (("fleet", "fleet"), ("jobs", "jobs"), ("gpus", "gpus")),
            "at start", "the view RunFleet opens with (this run keeps its view)"),
    Setting("refresh_multiplier", "refresh speed", (("slow", 2.0), ("normal", 1.0), ("fast", 0.5)),
            "floors apply", "poll intervals ×2 / ×1 / ×0.5, never below each command's floor"),
    Setting("readonly", "read-only", (("on", True), ("off", False)),
            "", "disable cancel / hold / terminate and every other mutating action"),
    Setting("mouse", "mouse", (("on", True), ("off", False)),
            "", "off: the terminal selects text natively (Shift-drag works either way in most terminals)"),
    Setting("idle_pause_after", "idle pause", (("30m", 1800.0), ("2h", 7200.0), ("8h", 28800.0), ("24h", 86400.0)),
            "", "pause all polling after this long without a key press or click"),
    Setting("remember_hosts", "remember hosts", (("on", True), ("off", False)),
            "", "`runfleet k2` saves k2 to config.toml once it answers (it still shows only k2 that run)"),
    Setting("enabled", "enabled", (("on", True), ("off", False)),
            "", "off: events still go to the Notifications history, nothing pops up", "notify"),
    Setting("terminal", "terminal", (("auto", "auto"), ("osc9", "osc9"), ("osc777", "osc777"), ("bell", "bell"),
                                     ("off", "off")),
            "", "auto detects OSC 9 / 777 or bell · in tmux: set -g allow-passthrough on", "notify"),
    Setting("events", "events", (("failures", FAILURES), ("+gpu idle", _events(*FAILURES, "gpu_idle")),
                                 ("all", NOTIFY_EVENTS)),
            "", "every job of yours; failures = failed, restarted, low on time (w: more)", "notify"),
)


def _line_of(index: int) -> int:
    """Line of GENERAL[index] in the General list: each new table adds its group heading."""
    return index + sum(1 for i in range(1, index + 1) if GENERAL[i].table != GENERAL[i - 1].table)


def _keys_for(*actions: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for action in actions:
        k = keymap.find("settings", action)
        out |= dict.fromkeys(k.keys if k is not None else (), action)
    return out


_BEGIN_EDIT = _keys_for("filter", "add")     # handled in on_key, synchronously (see the module docstring)

ABOUT_ROWS: tuple[tuple[str, str], ...] = (("config", "config file"), ("log", "log"), ("state", "state dir"),
                                           ("ssh", "ssh config"), ("version", "version"), ("theme", "theme"))
_ABOUT_ACTIONS = frozenset({"config", "log", "version", "theme"})
_UPDATE = ":update"                # `confirm` while ↵ again updates (no host name has a colon)


class SettingsPanel(Static, can_focus=True):
    """The whole panel as one renderable; the screen holds the state."""

    DEFAULT_CSS = """
    SettingsPanel { width: 1fr; height: 1fr; background: $rf-surface; color: $rf-foreground; padding: 0; }
    """

    def on_resize(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        screen = self.screen
        if self.is_mounted and isinstance(screen, SettingsScreen):
            self.update(screen.render_panel(self.size.width or PANEL_W - 2, self.size.height or PANEL_H - 2))

    async def on_click(self, event: events.Click) -> None:
        action = event.style.meta.get("settings") if event.style is not None else None
        screen = self.screen
        if action and isinstance(screen, SettingsScreen):
            event.stop()
            await screen.run_action(action)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        screen = self.screen
        if isinstance(screen, SettingsScreen):
            screen.action_cursor(1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        screen = self.screen
        if isinstance(screen, SettingsScreen):
            screen.action_cursor(-1)


class SettingsScreen(ModalScreen[None]):
    BINDINGS = keymap.bindings("settings")
    WANTS_FLEET_CHANGES = True             # the app forwards FleetChanged: live state chips
    DEFAULT_CSS = """
    SettingsScreen { align: center middle; }
    #settings-box {
        width: 76; height: 22;
        border: solid $accent;
        background: $rf-surface;
        padding: 0;
    }
    """

    def __init__(self, section: str = "hosts", *, update: bool = False) -> None:
        super().__init__()
        self.section = "about" if update else section if section in dict(SECTIONS) else "hosts"
        self.selected: dict[str, int] = {"hosts": 0, "general": 0, "about": 0}
        if update:
            self.selected["about"] = [k for k, _l in ABOUT_ROWS].index("version")
        self._open_update = update             # palette › Update RunFleet: start on the version row's ↵
        self._asked = False                    # this panel asked GitHub: say the answer on the message line
        self.scroll = 0
        self.rows: list[HostRow] = []
        self.aliases: list[str] = []
        self.load_error: str | None = None
        self.filter = ""
        self.editing: str | None = None        # "filter" | "add" while typing into the input line
        self.buffer = ""
        self.message: Text = Text()
        self.confirm: str | None = None        # host waiting for a second space (it runs jobs of mine)
        self.tests: dict[str, DetectReport | str] = {}
        self._test_gen: dict[str, int] = {}    # per ssh target: the latest `t`, so an older answer cannot overwrite it

    # ------------------------------------------------------------------ plumbing
    @property
    def rf(self) -> Any:
        return self.app

    @property
    def service(self) -> Any:
        return self.rf.service

    @property
    def in_memory(self) -> bool:
        return self.service.config_path is None

    def on_screen_resume(self) -> None:
        self.app.add_class("-settings-open")      # the panel has its own hint line: hide the base screen's HintBar

    def on_screen_suspend(self) -> None:
        self.app.remove_class("-settings-open")

    def on_unmount(self) -> None:
        self.app.remove_class("-settings-open")

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="settings-box"):
            yield SettingsPanel(id="settings-panel")

    def on_mount(self) -> None:
        if not self.query("#settings-panel"):    # dismissed before Mount ran (key burst)
            return
        self.aliases = self.service.ssh_aliases()
        self.reload()
        self._fit()
        self.query_one(SettingsPanel).focus()
        if self._open_update:
            self.open_about("version")

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        if not self.query("#settings-box"):
            return
        box = self.query_one("#settings-box")
        width, height = self.app.size
        box.styles.width = max(20, min(PANEL_W, width - 4))
        box.styles.height = max(8, min(PANEL_H, height - 2))
        self.redraw()

    def redraw(self) -> None:
        for panel in self.query(SettingsPanel):
            panel.refresh_content()

    def on_fleet_changed(self, _message: Any) -> None:
        names = {r.name for r in self.rows}
        if set(self.rf.store.hosts) - names or any(r.running != (r.name in self.rf.store.hosts) for r in self.rows):
            self.reload()
        self.redraw()

    # ------------------------------------------------------------------ model
    def reload(self) -> None:
        """Rows = saved hosts (file order) + ad-hoc hosts of this run + the other ~/.ssh/config aliases."""
        store = self.rf.store.hosts
        try:
            saved = list(self.service.saved_hosts())
            self.load_error = None
        except ConfigError as e:
            saved = []
            self.load_error = str(e)
        current = self._current_row()
        rows = host_rows_for(saved, [(n, hs.config.ssh) for n, hs in store.items()], self.aliases)
        self.rows = rows
        if current is not None:
            self._select_name(current.name)
        self._clamp()
        self.redraw()

    def visible_rows(self) -> list[HostRow]:
        needle = self.filter.lower()
        if not needle:
            return list(self.rows)
        return [r for r in self.rows if needle in r.name.lower() or needle in r.ssh.lower()]

    def _count(self) -> int:
        if self.section == "hosts":
            return len(self.visible_rows())
        return len(GENERAL) if self.section == "general" else len(ABOUT_ROWS)

    def _clamp(self) -> None:
        n = self._count()
        self.selected[self.section] = max(0, min(self.selected[self.section], n - 1)) if n else 0

    def _current_row(self) -> HostRow | None:
        rows = self.visible_rows()
        i = self.selected["hosts"]
        return rows[i] if 0 <= i < len(rows) else None

    def _select_name(self, name: str) -> None:
        for i, r in enumerate(self.visible_rows()):
            if r.name == name:
                self.selected["hosts"] = i
                return

    def _say(self, text: str, style: str = "") -> None:
        self.message = Text(text, style=style or theme.muted())
        self.redraw()

    def _error(self, text: str) -> None:
        path = self.service.config_path
        if path is not None:                   # the title shows the path: keep the reason inside the line
            text = text.replace(f"{path}: ", f"{path.name}: ").replace(str(path), path.name)
        self._say(text, theme.error())

    # ------------------------------------------------------------------ actions (keymap "settings")
    def action_section(self, step: int) -> None:
        names = [s for s, _ in SECTIONS]
        self.switch(names[(names.index(self.section) + step) % len(names)])

    def switch(self, section: str) -> None:
        if section not in dict(SECTIONS):
            return
        self._end_edit()
        self.section = section
        self.confirm = None
        self.message = Text()
        self.scroll = 0
        self._clamp()
        self.redraw()

    def action_cursor(self, step: int) -> None:
        if self.editing:
            self._end_edit(keep=True)
        self.confirm = None
        n = self._count()
        if n:
            self.selected[self.section] = max(0, min(n - 1, self.selected[self.section] + step))
        self.redraw()

    def action_activate(self) -> None:
        if self.section == "hosts":
            row = self._current_row()
            if row is not None:
                self.toggle(row)
        elif self.section == "general":
            setting = GENERAL[self.selected["general"]]
            values = [v for _l, v in setting.options]
            current = self._current(setting)
            nxt = values[(values.index(current) + 1) % len(values)] if current in values else values[0]
            self.set_value(setting, nxt)
        else:
            self.open_about(ABOUT_ROWS[self.selected["about"]][0])

    def action_test(self) -> None:
        row = self._current_row() if self.section == "hosts" else None
        if row is None:
            return
        self.tests[row.ssh] = "testing"
        self._test_gen[row.ssh] = gen = self._test_gen.get(row.ssh, 0) + 1
        self._say(f"testing {row.ssh} (read-only probe, BatchMode) …")
        self.run_worker(self._test(row, gen), group="settings-test", exit_on_error=False)

    async def _test(self, row: HostRow, gen: int) -> None:
        try:
            report = await self.service.probe_alias(row.ssh)
        except Exception as e:                 # a probe problem is a message, never a crash
            if self._test_gen.get(row.ssh, 0) != gen:
                return
            self.tests.pop(row.ssh, None)
            self._error(f"{row.ssh}: test failed: {e}")
            return
        if self._test_gen.get(row.ssh, 0) != gen:   # a newer `t` on this host is running: its answer wins
            return
        self.tests[row.ssh] = report
        self._say(f"{row.ssh}: {probe_summary(report)}", theme.ok() if report.state in (
            ConnState.OK, ConnState.DEGRADED) else theme.warn() if report.state in (
            ConnState.NEEDS_AUTH, ConnState.HOSTKEY) else theme.error())

    def action_poll(self) -> None:
        """``p``: poll a saved host this (ad-hoc) run left out, with its saved settings; the file is not touched."""
        row = self._current_row() if self.section == "hosts" else None
        if row is None:
            return
        if row.running:
            self._say(f"{row.name} is already polled in this run")
            return
        if not row.saved:
            self._say(f"{row.name} is not saved — space saves it and starts polling")
            return
        self.confirm = None
        try:                                   # already in the file: add_host writes nothing, runs the saved table
            self.service.add_host(HostConfig(name=row.name, ssh=row.ssh))
        except (ConfigError, RunFleetError) as e:
            self._error(str(e))
            return
        self.reload()
        self._select_name(row.name)
        self._say(f"{row.name}: polling started{self._where()}", theme.ok())

    def action_add(self) -> None:
        if self.section != "hosts":
            self.switch("hosts")
        self._begin_edit("add")
        self._say("ssh target — an alias, host or user@host; ↵ saves and starts polling, esc cancels")

    def action_filter(self) -> None:
        if self.section != "hosts":
            self.switch("hosts")
        self._begin_edit("filter", self.filter)

    def action_move_host(self, step: int) -> None:
        if self.section != "hosts":
            return
        row = self._current_row()
        if row is None:
            return
        if not row.saved:
            self._say("save the host first (space) — only saved hosts have an order", theme.warn())
            return
        if self.filter:
            self._say("clear the filter (/ then esc) to reorder", theme.warn())
            return
        saved = [r.name for r in self.rows if r.saved]
        i = saved.index(row.name)
        j = i + step
        if not 0 <= j < len(saved):
            return
        saved[i], saved[j] = saved[j], saved[i]
        try:
            self.service.reorder_hosts(saved)
        except (ConfigError, RunFleetError) as e:
            self._error(str(e))
            return
        self.reload()
        self._select_name(row.name)
        self._say(f"{row.name} is now number {saved.index(row.name) + 1}" + self._where())

    def action_edit_config(self) -> None:
        self.open_about("config")

    def action_close(self) -> None:
        if self.editing:
            self._end_edit()
            return
        if self.confirm:
            self.confirm = None
            self.message = Text()
            self.redraw()
            return
        self.dismiss(None)

    # ------------------------------------------------------------------ hosts: toggle
    def _where(self) -> str:
        return " (this run only — demo)" if self.in_memory else ""

    def toggle(self, row: HostRow) -> None:
        if self.load_error:
            self._error("config.toml cannot be read — fix it first (e opens it)")
            return
        if row.saved:
            warning = self._removal_warning(row)
            if warning and self.confirm != row.name:
                self.confirm = row.name
                self._say(warning, theme.warn())
                return
            self.confirm = None
            group = f"settings-remove-{row.name}"
            if any(w.group == group and w.is_running for w in self.app.workers):
                self._say(f"still removing {row.name} …")
                return
            self._say(f"removing {row.name} …")
            # an app worker: closing Settings must not cancel a removal halfway (config edited, cleanup not done)
            self.app.run_worker(self._remove(row), group=group, exit_on_error=False)
            return
        try:
            self.service.add_host(HostConfig(name=row.name, ssh=row.ssh))
        except (ConfigError, RunFleetError) as e:
            self._error(str(e))
            return
        self.reload()
        self._select_name(row.name)
        what = "saved" if row.running else "saved · polling started"
        self._say(f"{row.name} {what}{self._where()}", theme.ok())

    async def _remove(self, row: HostRow) -> None:
        try:
            await self.service.remove_host(row.name)
        except Exception as e:                 # a worker: anything uncaught would vanish and leave "removing …"
            if self not in self.app.screen_stack:   # Settings closed meanwhile: the error still has to be seen
                self.rf.toast(f"removing {row.name} failed: {str(e) or type(e).__name__}", severity="error")
                return
            self._error(str(e) or type(e).__name__)
            return
        if self not in self.app.screen_stack:
            return
        self.reload()
        self._select_name(row.name)
        self._say(f"{row.name} removed{self._where()}")

    def _removal_warning(self, row: HostRow) -> str | None:
        """Why removing `row` deserves a second press: jobs of mine, hand-tuned keys, or not in this run (then space
        cannot mean "show it", only "forget it")."""
        mine = self._running_jobs(row.name)
        if mine:
            return (f"{row.name} runs {mine} job{'s' if mine != 1 else ''} of yours — space again removes it from the "
                    "fleet (the jobs keep running)")
        where = "the config" if self.in_memory else "config.toml"
        keys = self.service.custom_host_keys(row.name)
        if keys:
            shown = ", ".join(keys[:3]) + (", …" if len(keys) > 3 else "")
            return f"{row.name} has its own settings ({shown}) — space again removes its table from {where}"
        if not row.running:
            return f"{row.name} is saved but not in this run — space again removes it from {where} (p polls it)"
        return None

    def _running_jobs(self, host: str) -> int:
        if host not in self.rf.store.hosts:
            return 0
        return sum(1 for j in self.rf.store.jobs([host]) if j.is_mine and j.state.group in (
            StateGroup.ACTIVE, StateGroup.QUEUED))

    # ------------------------------------------------------------------ general
    def _current(self, setting: Setting) -> object:
        """The value in effect: refresh speed follows `+`/`-` (the app's live multiplier), the rest the config; an
        events list in preset order (a list that is no preset shows as `custom` and stays until the row changes)."""
        if setting.table == "notify":
            value = getattr(self.service.config.notify, setting.key)
            return _events(*value) if setting.key == "events" else value
        if setting.key == "refresh_multiplier":
            return float(self.rf.multiplier)
        return getattr(self.service.config.general, setting.key)

    def set_value(self, setting: Setting, value: object) -> None:
        """Save one row. [notify] is live by itself: Alerts reads the service's config.notify on every batch."""
        try:
            if setting.table == "notify":
                self.service.set_notify(**{setting.key: value})
            else:
                self.service.set_general(**{setting.key: value})
        except (ConfigError, RunFleetError) as e:
            self._error(str(e))
            return
        app = self.rf
        if setting.key == "refresh_multiplier":
            app.multiplier = float(value)        # type: ignore[arg-type]
            app.refresh_headers()
        elif setting.key == "readonly":
            app.refresh_headers()
        elif setting.key == "mouse":
            app.set_mouse(bool(value))
        label = next(lbl for lbl, v in setting.options if v == value)
        when = " — from the next start" if setting.note == "at start" else ""
        group = f"{GROUPS[setting.table]} · " if setting.table in GROUPS else ""
        self._say(f"{group}{setting.label}: {label}{when}{self._where()}", theme.ok())

    # ------------------------------------------------------------------ about
    def open_about(self, key: str) -> None:
        if key == "config":
            path = self.service.config_path
            if path is None:
                self._say("demo — nothing is saved, so there is no file to edit", theme.warn())
                return
            rc = self.rf.run_external(editor_argv(path))
            self.reload()
            if rc is not None:
                self._say(f"edited {display_path(path)} — changes made by hand apply at the next start")
        elif key == "log":
            self.rf.open_runfleet_log()
        elif key == "version":
            self._version_enter()
        elif key == "theme":
            choices = list(theme.CHOICES)
            current = self.rf.ui_state.get("theme", "auto")
            nxt = choices[(choices.index(current) + 1) % len(choices)] if current in choices else "auto"
            resolved = self.rf.set_theme_choice(nxt)
            kept = " — saved for the next start" if self.rf._state_path is not None else " (this run only — demo)"
            self._say(f"theme: {nxt} → {resolved.mode} ({resolved.source}){kept}", theme.ok())

    def _version_enter(self) -> None:
        """↵ on the version row: check (no newer version known) → arm → ↵ again quits, updates and restarts."""
        app = self.rf
        status = app.update_status
        if self.confirm == _UPDATE:
            app.request_update()
            return
        if self.in_memory:
            self._say("demo — no network; `runfleet update` updates RunFleet", theme.warn())
            return
        if app.update_checking:
            self._asked = True
            self._say("asking GitHub for the newest version …")
            return
        if status is None or not status.available:
            self._asked = True
            app.check_for_update(force=True)
            self._say("asking GitHub for the newest version …")
            return
        from runfleet import update

        manual = update.detect_install().manual
        if platform.IS_WINDOWS:          # the running runfleet.exe cannot be replaced: by hand, after quitting
            self._say(f"{status.latest}: quit RunFleet (q), then run {manual}", theme.warn())
            return
        self.confirm = _UPDATE
        self._say(f"↵ again: quit, update to {status.latest} ({manual}) and start again · esc: not now", theme.warn())

    def update_changed(self, answer: Any) -> None:
        """The app's update check started or ended (`answer`: the Status of a check this panel may have asked)."""
        if answer is not None and self._asked:
            self._asked = False
            if answer.available:
                self._version_enter() if self._open_update else self._say(
                    f"RunFleet {answer.latest} is available — ↵ " + ("shows how to update" if platform.IS_WINDOWS
                                                                     else "updates and restarts"), theme.ok())
            elif answer.error is not None:
                self._error(f"cannot ask GitHub: {answer.error}")
            else:
                self._say(f"this is the newest version ({answer.latest})", theme.ok())
            self._open_update = False
        self.redraw()

    # ------------------------------------------------------------------ input line (filter / add)
    def _begin_edit(self, mode: str, text: str = "") -> None:
        self.editing = mode
        self.buffer = text
        self.redraw()

    def _end_edit(self, keep: bool = False) -> None:
        if self.editing == "filter" and not keep:
            self.filter = ""
        self.editing = None
        self.buffer = ""
        self._clamp()
        self.redraw()

    def _commit_edit(self) -> None:
        if self.editing == "filter":
            self._end_edit(keep=True)
            return
        target = self.buffer.strip()
        self.editing = None
        self.buffer = ""
        if not target:
            self.message = Text()
            self.redraw()
            return
        if target.startswith("-") or any(c.isspace() for c in target):
            self._error(f"not an ssh target: {target!r}")
            return
        row = next((r for r in self.rows if target in (r.ssh, r.name)), None)
        if row is None:
            # `alice@box` -> `alice_box` may already name another endpoint: the core would keep that one, so suffix
            name = base = host_name_for(target)
            taken = {r.name for r in self.rows} | set(self.rf.store.hosts)
            n = 2
            while name in taken:
                name, n = f"{base}_{n}", n + 1
            row = HostRow(name, target, False, False)
            self.rows.append(row)
        self.filter = ""
        self._select_name(row.name)
        if row.saved:
            self._say(f"{row.name} is already saved")
            return
        self.toggle(row)

    def on_key(self, event: events.Key) -> None:
        if not self.editing:
            action = _BEGIN_EDIT.get(event.key)
            if action is not None:          # start the input now, not via the binding: that runs only after keys
                event.stop()                # already queued behind this one have passed through here as commands
                event.prevent_default()
                if action == "filter":
                    self.action_filter()
                else:
                    self.action_add()
            return
        key = event.key
        handled = True
        if key == "enter":
            self._commit_edit()
        elif key == "escape":
            self._end_edit(keep=False)
            if not self.editing:
                self.message = Text()
        elif key in ("down", "tab"):
            self._end_edit(keep=True)
        elif key == "backspace":
            self.buffer = self.buffer[:-1]
        elif key in ("ctrl+u",):
            self.buffer = ""
        elif key == "ctrl+w":
            self.buffer = re.sub(r"\S*\s*$", "", self.buffer)
        elif event.is_printable and event.character:
            self.buffer += event.character
        else:
            handled = False
        if handled:
            event.stop()
            event.prevent_default()
            if self.editing == "filter":
                self.filter = self.buffer
                self.selected["hosts"] = 0
                self.scroll = 0
            self.redraw()

    def on_paste(self, event: events.Paste) -> None:
        if self.editing:
            event.stop()
            self.buffer += event.text.splitlines()[0] if event.text else ""
            if self.editing == "filter":
                self.filter = self.buffer
            self.redraw()

    # ------------------------------------------------------------------ mouse (@click actions)
    def action_click_tab(self, section: str) -> None:
        self.switch(section)

    def action_click_row(self, index: int) -> None:
        if self.editing:
            self._end_edit(keep=True)
        self.confirm = None if self.selected[self.section] != index else self.confirm
        self.selected[self.section] = index
        self._clamp()
        self.redraw()

    def action_click_check(self, index: int) -> None:
        self.action_click_row(index)
        row = self._current_row()
        if row is not None:
            self.toggle(row)

    def action_click_value(self, index: int, option: int) -> None:
        self.selected["general"] = index
        setting = GENERAL[index]
        self.set_value(setting, setting.options[option][1])

    def action_click_button(self, which: str) -> None:
        if which == "close":
            self.action_close()
        else:
            self.action_activate()

    def action_click_filter(self) -> None:
        self.action_filter()

    # ================================================================== rendering
    def render_panel(self, width: int, height: int) -> Text:
        compact = height < 18
        lines: list[Text] = [self._title(width), self._tabs(width), Text("─" * width, style=theme.rule())]
        if not compact:
            lines.append(Text(""))
        heading, desc = self._heading(width)
        heading_at = len(lines)
        lines.append(heading)
        if not compact:
            lines += [desc, Text("")]
        footer = [self._message(width), self._hint(width), self._buttons(width)]
        list_h = max(1, height - len(lines) - len(footer))
        if self.section == "hosts":
            body = self._host_lines(width, list_h)
        else:                                  # a short panel: the list follows the selection, like Hosts
            body = self._general_lines(width) if self.section == "general" else self._about_lines(width)
            if len(body) > list_h:
                sel = self.selected[self.section]
                start = max(0, (_line_of(sel) if self.section == "general" else sel) - list_h + 1)
                body = body[start:start + list_h]
                lines[heading_at] = _spread(heading, Text(f"{sel + 1}/{self._count()} ", style=theme.dim()), width)
        body = body[:list_h] + [Text("")] * max(0, list_h - len(body))
        out = Text("\n").join([*lines, *body, *footer])
        out.no_wrap = True
        return out

    # ---- chrome
    def _title(self, width: int) -> Text:
        left = Text(" settings", style="bold")
        if self.in_memory:
            right = Text("demo — not saved ", style=theme.warn())
        else:
            path = display_path(self.service.config_path)
            room = max(0, width - left.cell_len - 3)
            if len(path) > room:                       # keep the file name: cut the path from the left
                path = "…" + path[len(path) - room + 1:] if room > 1 else ""
            right = Text(path + " ", style=theme.dim())
        return _spread(left, right, width)

    def _tabs(self, width: int) -> Text:
        out = Text(" ")
        for ident, label in SECTIONS:
            part = Text(f" {label} ", style=theme.pill() if ident == self.section else theme.muted())
            part.stylize(_click(f"click_tab({ident!r})"))
            out.append_text(part)
            out.append(" ")
        out.truncate(width)
        return out

    def _heading(self, width: int) -> tuple[Text, Text]:
        bold = "bold"
        if self.section == "hosts":
            n_saved = sum(1 for r in self.rows if r.saved)
            title = Text(" fleet hosts", style=bold)
            where = "demo hosts" if self.in_memory else "in ~/.ssh/config"
            counts = Text(f"{n_saved} saved · {len(self.aliases)} {where} ", style=theme.dim())
            desc = ("checked = saved and polled; uncheck to stop and forget"
                    + (" — this run only" if self.in_memory else ""))
            return _spread(title, counts, width), _fit(Text(" " + desc, style=theme.muted()), width)
        if self.section == "general":
            setting = GENERAL[self.selected["general"]]
            return (Text(" general", style=bold),
                    _fit(Text(" " + setting.help, style=theme.muted()), width))
        return (Text(" about", style=bold),
                _fit(Text(" where RunFleet keeps its files", style=theme.muted()), width))

    def _message(self, width: int) -> Text:
        if self.load_error and not self.message:
            return _fit(Text(f" {self.load_error}", style=theme.error()), width)
        return _fit(Text(" ") + self.message, width)

    def _hint(self, width: int) -> Text:
        if self.editing:
            text = " ↵ done  esc cancel" if self.editing == "add" else " ↵ keep filter  esc clear"
        elif self.section == "hosts":
            text = " ↑↓ select  space toggle  t test  a add  J/K move  / filter  tab section"
        elif self.section == "general":
            text = " ↑↓ select  space next value  click a value  tab section"
        else:
            text = " ↑↓ select  ↵ open  e edit config  tab section"
        return _fit(Text(text, style=theme.dim()), width)

    def _primary_label(self) -> str:
        if self.section == "hosts":
            row = self._current_row()
            return " ↵ remove " if row is not None and row.saved else " ↵ save "
        if self.section == "general":
            return " ↵ change "
        key = ABOUT_ROWS[self.selected["about"]][0]
        if key == "version":
            status = self.rf.update_status
            return (" ↵ update now " if self.confirm == _UPDATE else
                    (" ↵ how " if platform.IS_WINDOWS else " ↵ update ") if status is not None and status.available
                    else " ↵ check ")
        if key == "theme":
            return " ↵ change "
        return " ↵ open " if key in _ABOUT_ACTIONS else ""

    def _buttons(self, width: int) -> Text:
        label = self._primary_label()
        close = Text(" esc close ", style=theme.pill_quiet())
        close.stylize(_click("click_button('close')"))
        if label:
            primary = Text(label, style=theme.pill())
            primary.stylize(_click("click_button('primary')"))
            row = Text.assemble(primary, "  ", close)
        else:
            row = close
        pad = max(0, (width - row.cell_len) // 2)
        return Text(" " * pad) + row

    # ---- sections
    def _row_style(self, selected: bool) -> str:
        return theme.pill() if selected else ""

    def _host_lines(self, width: int, height: int) -> list[Text]:
        rows = self.visible_rows()
        out: list[Text] = []
        show_filter = self.editing == "filter" or bool(self.filter) or len(self.rows) > height
        if self.editing == "add":
            line = Text("  + ssh target: ", style=theme.accent())
            line.append(self.buffer)
            line.append("▏", style=theme.accent())
            out.append(_fit(line, width))
            height -= 1
        elif show_filter:
            line = Text("  / ", style=theme.accent())
            if self.editing == "filter":
                line.append(self.buffer)
                line.append("▏", style=theme.accent())
            elif self.filter:
                line.append(self.filter)
                line.append(f"   {len(rows)} of {len(self.rows)}", style=theme.dim())
            else:
                line.append("type / to filter", style=theme.dim())
            line.stylize(_click("click_filter()"))
            pos = Text(f"{min(self.selected['hosts'] + 1, len(rows))}/{len(rows)} ", style=theme.dim())
            out.append(_spread(line, pos, width))
            height -= 1
        if not rows:
            msg = (f"no host matches “{self.filter}”" if self.filter else
                   "no hosts yet — press a to add an ssh target")
            out.append(Text(f"   {msg}", style=theme.dim()))
            return out
        sel = self.selected["hosts"]
        if sel < self.scroll:
            self.scroll = sel
        elif sel >= self.scroll + height:
            self.scroll = sel - height + 1
        self.scroll = max(0, min(self.scroll, max(0, len(rows) - height)))
        name_w = max(10, min(22, max(len(r.name) for r in rows) + 2))
        for i in range(self.scroll, min(len(rows), self.scroll + height)):
            out.append(self._host_line(rows[i], i, i == sel, width, name_w))
        return out

    def _host_line(self, row: HostRow, index: int, selected: bool, width: int, name_w: int) -> Text:
        hs = self.rf.store.hosts.get(row.name) if row.running else None
        line = Text(" ▸ " if selected else "   ")
        box = Text("[x]" if row.saved else "[ ]", style=theme.accent() if row.saved else theme.muted())
        box.stylize(_click(f"click_check({index})"))
        line.append_text(box)
        line.append(" ")
        name = Text(row.name if len(row.name) < name_w else row.name[: name_w - 2] + "…",
                    style="bold" if row.saved or row.running else "")
        name.pad_right(name_w - name.cell_len)
        line.append_text(name)
        line.append_text(_cell(self._kind(row, hs), 22))
        line.append_text(self._state(row, hs))
        line = _fit(line, width)
        line.pad_right(width - line.cell_len)
        if selected:
            line.stylize(theme.pill())
        line.stylize(_click(f"click_row({index})"), 0, 3)
        line.stylize(_click(f"click_row({index})"), 7)
        return line

    def _kind(self, row: HostRow, hs: Any) -> Text:
        snap = hs.snapshot if hs is not None else None
        caps = snap.caps if snap is not None else None
        test = self.tests.get(row.ssh)
        if caps is None and isinstance(test, DetectReport):
            caps = test.caps
        if caps is None:
            return Text("?" if row.ssh != "local" else "local", style=theme.dim())
        if "slurm" in caps.profile:
            return Text(f"slurm {caps.slurm_version or ''}".strip(), style=theme.muted())
        if snap is not None and snap.gpus:
            return Text(f"workstation · {len(snap.gpus)} GPU{'s' if len(snap.gpus) != 1 else ''}",
                        style=theme.muted())
        return Text("workstation", style=theme.muted())

    def _state(self, row: HostRow, hs: Any) -> Text:
        test = self.tests.get(row.ssh)
        out = Text()
        if test == "testing":
            out.append("testing …", style=theme.muted())
        elif isinstance(test, DetectReport) and hs is None:
            out.append_text(probe_chip(test))
        elif hs is not None:
            st = hs.status.state
            out.append(f"{theme.CONN_GLYPH[st]} {theme.CONN_LABEL[st]}", style=theme.conn_style(st))
            if isinstance(test, DetectReport):
                out.append("  test: ", style=theme.dim())
                out.append_text(probe_chip(test, short=True))
        elif row.saved:
            out.append("not in this run · p polls it", style=theme.dim())
        if row.running and not row.saved:
            out.append("  unsaved", style=theme.warn())
        return out

    def _general_lines(self, width: int) -> list[Text]:
        out: list[Text] = []
        sel = self.selected["general"]
        for i, s in enumerate(GENERAL):
            if i and s.table != GENERAL[i - 1].table:
                out.append(self._group_line(s.table, width))
            current = self._current(s)
            line = Text(" ▸ " if i == sel else "   ")
            line.append(s.label.ljust(18))
            known = any(v == current for _l, v in s.options)
            for j, (label, value) in enumerate(s.options):
                on = value == current
                part = Text(f" {label} ", style=(theme.accent() + " reverse" if on and i != sel else
                                                 "bold underline" if on else theme.muted()))
                part.stylize(_click(f"click_value({i}, {j})"))
                line.append_text(part)
                line.append(" ")
            if not known:
                line.append(f" custom {_custom(s.key, current)} ", style=theme.warn())
            note = Text(s.note + " ", style=theme.dim())
            line = _spread(line, note, width)
            if i == sel:
                line.stylize(theme.pill())
            line.stylize(_click(f"click_row({i})"), 0, 21)
            out.append(line)
        return out

    def _group_line(self, table: str, width: int) -> Text:
        """Heading of a group of rows; for [notify] it says whether the command hook (config.toml only) is set."""
        note = ""
        if table == "notify":
            from runfleet.ui.notify import hook_argv

            hook = hook_argv(self.service.config.notify.command) is not None
            note = "command hook: set in config.toml" if hook else "command hook: none (config.toml only)"
        return _spread(Text("   " + GROUPS.get(table, table), style="bold"), Text(note + " ", style=theme.dim()), width)

    def _theme_value(self) -> tuple[str, str]:
        choice = self.rf.ui_state.get("theme", "auto")
        choice = choice if choice in theme.CHOICES else "auto"      # a hand-edited ui.json: what is in effect
        # the mode in effect whenever it is not the choice (auto, or RUNFLEET_THEME overriding the setting)
        value = f"{choice} → {theme.MODE}" if choice == "auto" or theme.MODE != choice else choice
        where = ("RUNFLEET_THEME" if theme.SOURCE == "RUNFLEET_THEME" else
                 f"background {theme.BACKGROUND}" if theme.BACKGROUND else
                 "background unknown" if theme.SOURCE in ("default", "COLORFGBG") else theme.SOURCE)
        return value, f"{where} · ↵ auto/dark/light/ansi"

    def _version_note(self) -> str:
        app = self.rf
        status = app.update_status
        if app.update_checking:
            return "checking …"
        if status is None:
            return "↵ check for updates"
        if status.available:
            return f"{status.latest} available · ↵ " + ("how to update" if platform.IS_WINDOWS else "update")
        if status.error is not None and status.latest is None:
            return "check failed · ↵ retry"
        return "newest · ↵ check again"

    def _about_lines(self, width: int) -> list[Text]:
        from runfleet import __version__, paths

        cfg = self.service.config_path
        try:
            log_file, state_dir = paths.log_file(), paths.state_dir()
        except Exception:                      # paths unavailable: show nothing rather than fail
            log_file, state_dir = None, None
        values = {
            "config": (display_path(cfg) if cfg is not None else "in memory — demo, nothing is saved",
                       "↵ edit in $EDITOR" if cfg is not None else ""),
            "log": (display_path(log_file), "↵ view"),
            "state": (display_path(state_dir) + "/" if state_dir else "", "cache, ui.json, ssh logs"),
            "ssh": ("~/.ssh/config", f"{len(self.aliases)} host alias{'es' if len(self.aliases) != 1 else ''}"),
            "version": (f"runfleet {__version__}", self._version_note()),
            "theme": self._theme_value(),
        }
        out: list[Text] = []
        sel = self.selected["about"]
        for i, (key, label) in enumerate(ABOUT_ROWS):
            value, note = values[key]
            line = Text(" ▸ " if i == sel else "   ")
            line.append(label.ljust(14), style=theme.muted() if i != sel else "")
            line.append(value)
            line = _spread(line, Text(note + " ", style=theme.dim()), width)
            if i == sel:
                line.stylize(theme.pill())
            line.stylize(_click(f"click_row({i})"))
            out.append(line)
        return out


# ======================================================================================= helpers
def probe_summary(report: DetectReport) -> str:
    """One line for the message bar: kind + version / driver + latency, or the error."""
    if report.state in (ConnState.OK, ConnState.DEGRADED):
        caps = report.caps
        kind = ("" if caps is None else f" — slurm {caps.slurm_version or '?'}" if "slurm" in caps.profile else
                f" — workstation, driver {caps.driver_version}" if caps.driver_version else " — workstation")
        latency = f" · {report.latency_s:.2f}s" if report.latency_s is not None else ""
        return f"ok{kind}{latency} · mux {report.mux}"
    if report.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
        what = "needs a login" if report.state is ConnState.NEEDS_AUTH else "host key problem"
        return f"{what} — save it, then press c on its card (or `runfleet connect {report.alias}`)"
    return report.error.summary if report.error is not None else "no answer"


def probe_chip(report: DetectReport, *, short: bool = False) -> Text:
    if report.state in (ConnState.OK, ConnState.DEGRADED):
        latency = f" {report.latency_s:.2f}s" if report.latency_s is not None else ""
        return Text(f"✓ ok{latency}", style=theme.ok())
    if report.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
        return Text(f"✖ {theme.CONN_LABEL[report.state]}", style=theme.conn_style(report.state))
    summary = report.error.summary if report.error is not None else "no answer"
    return Text(f"✖ {'failed' if short else summary}", style=theme.error())


def _custom(key: str, value: object) -> str:
    from runfleet.config import format_duration
    if key == "idle_pause_after" and isinstance(value, (int, float)):
        return format_duration(float(value))
    if key == "events" and isinstance(value, tuple):
        return f"{len(value)} event{'s' if len(value) != 1 else ''}"
    if isinstance(value, float):
        return f"×{value:g}"
    return str(value)


def _fit(text: Text, width: int) -> Text:
    if text.cell_len > width:
        text = text.copy()
        text.truncate(width, overflow="ellipsis")
    return text


def _cell(text: Text, width: int) -> Text:
    text = _fit(text, width - 1)
    text.pad_right(width - text.cell_len)
    return text


def _spread(left: Text, right: Text, width: int) -> Text:
    """`left` … `right` on one line of `width` cells, at least one space apart (left truncated first)."""
    gap = 1 if left.cell_len and right.cell_len else 0
    if left.cell_len + right.cell_len + gap > width:
        left = _fit(left, max(0, width - right.cell_len - gap))
    return Text.assemble(left, " " * max(0, width - left.cell_len - right.cell_len), right)


def host_rows_for(saved: Sequence[HostConfig], running: Sequence[tuple[str, str]],
                  aliases: Sequence[str]) -> list[HostRow]:
    """Hosts-tab rows: saved hosts (file order), then this run's ad-hoc hosts, then the other ssh aliases."""
    rows = [HostRow(h.name, h.ssh, True, any(n == h.name for n, _ in running)) for h in saved]
    taken = {r.name for r in rows} | {r.ssh for r in rows}
    for name, ssh in running:
        if name not in taken:
            rows.append(HostRow(name, ssh, False, True))
            taken |= {name, ssh}
    for alias in aliases:
        if alias not in taken and host_name_for(alias) not in taken:
            rows.append(HostRow(host_name_for(alias), alias, False, False))
            taken |= {alias, host_name_for(alias)}
    return rows
