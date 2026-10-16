"""[WP3] RunFleetApp: screen stack, global bindings, store->message bridge (<= 4 Hz).

The app owns the cross-screen state (host scope, rate multiplier, persisted Jobs-view settings), the global
actions of §7.8 and the action flow of §6.3 (action menu -> confirm -> execute -> toast). It talks to the core only
through :class:`~runfleet.core.service.FleetAPI`; :meth:`run_external` (and :meth:`run_external_sequence`, log in to
all) are the only places that suspend the TUI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from textual import events, work
from textual.app import App, SuspendNotSupported
from textual.binding import BindingType
from textual.driver import Driver
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.worker import Worker, WorkerState

from runfleet.backends.base import ActionSpec
from runfleet.core.clock import Clock, RealClock
from runfleet.core.service import FleetAPI
from runfleet.core.store import ChangeSet, HostState
from runfleet.errors import ActionError
from runfleet.transport import spawn
from runfleet.models import ActionResult, ConnState, Job, JobKey, JobKind, JobState, LogRef, Visibility
from runfleet.ui import keymap, saved_views, theme
from runfleet.ui.commands import RunFleetCommands
from runfleet.ui.messages import ClockTick, FleetChanged, ScopeChanged
from runfleet.ui.notify import Alerts
from runfleet.ui.saved_views import SavedView
from runfleet.ui.screens import FleetScreen
from runfleet.util import platform as rfplatform

log = logging.getLogger("runfleet.ui.app")

PROBLEM_STATES = frozenset({ConnState.UNREACHABLE, ConnState.NEEDS_AUTH, ConnState.HOSTKEY})
MIN_SIZE = (60, 16)
FLUSH_INTERVAL = 0.25          # store -> UI at most 4 Hz (§7.12)
TOAST_WINDOW = 10.0            # at most 3 toasts per 10 s (§7.7)
TOAST_MAX = 3
PUSH_GRACE = 0.15              # a `q` this soon after a push was meant for the new screen
QUICK_EXIT = 10.0              # a shell that fails sooner than this was not used: keep its error on screen
DOCTOR_HOLD = 300.0            # a doctor report that finished under a dialog is shown, not re-run, for 5 min

_PAST = {"cancel": "Cancelled", "hold": "Held", "release": "Released", "requeue": "Requeued",
         "terminate": "Terminated", "interrupt": "Interrupted", "kill": "Killed"}


def saved_theme_choice() -> str:
    """The theme picked in Settings › About (ui.json `theme`): auto | dark | light | ansi."""
    path = ui_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else {}
    except (OSError, ValueError):
        return "auto"
    choice = data.get("theme") if isinstance(data, dict) else None
    return choice if choice in theme.CHOICES else "auto"


def ui_state_path() -> Path | None:
    """``~/.local/state/runfleet/ui.json`` (filter, preset, sort, grouping); None if paths are unavailable."""
    try:
        from runfleet import paths
        return paths.log_file().parent / "ui.json"
    except Exception:   # paths not implemented yet / no state dir: persistence is best-effort
        return None


def write_crash_report(exc: BaseException, directory: Path | None = None) -> Path | None:
    """``crash-<ts>.txt`` next to runfleet.log: traceback + versions, no fleet data (§7.12). Never raises."""
    import platform
    import tempfile
    import traceback

    import rich
    import textual

    from runfleet import __version__

    try:
        if directory is None:
            try:
                from runfleet import paths
                directory = paths.log_file().parent
            except Exception:   # paths unavailable: fall back to the temp dir
                directory = Path(tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"crash-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        header = (f"RunFleet {__version__} · Python {platform.python_version()} · Textual {textual.__version__}"
                  f" · Rich {getattr(rich, '__version__', '?')} · {platform.platform()}\n\n")
        path.write_text(header + "".join(traceback.format_exception(exc)), encoding="utf-8")
        path.chmod(0o600)
        logging.getLogger("runfleet.ui").critical("crash (report %s)", path, exc_info=exc)   # runfleet.log too
        return path
    except Exception:   # a crash report must never mask the crash
        return None


def _cell_mouse_driver(base: type[Driver]) -> type[Driver]:
    """`base` with SGR-pixel mouse reporting (1016) and in-band resize reports (2048) left off.

    Textual turns pixel mouse coordinates on as soon as a terminal answers the 2048 query, and from the
    first in-band resize report on it divides every mouse coordinate by the cell size. Multiplexers such
    as herdr support 2048 but keep sending cell coordinates, so every click lands near the top-left corner
    and the mouse appears dead. RunFleet needs nothing finer than a cell, so it stays in cell mode
    everywhere and relies on SIGWINCH for resizes.
    """
    try:
        from textual.drivers.linux_driver import LinuxDriver
        from textual.messages import InBandWindowResize
    except ImportError:          # Windows: no termios, its driver never enables pixel mouse
        return base
    if not issubclass(base, LinuxDriver):
        return base

    class CellMouseDriver(base):  # type: ignore[valid-type,misc]
        def _enable_mouse_pixels(self) -> None:
            pass

        def _enable_in_band_window_resize(self) -> None:
            pass

        def process_message(self, message: Message) -> None:
            if isinstance(message, InBandWindowResize):
                if message.enabled:          # already on in this terminal: switch it off before reports arrive
                    self.write("\x1b[?2048l")
                    self.flush()
                return
            super().process_message(message)

    return CellMouseDriver


class RunFleetApp(App[None]):
    """The RunFleet TUI."""

    CSS_PATH = "runfleet.tcss"
    TITLE = "RunFleet"
    BINDINGS: ClassVar[list[BindingType]] = keymap.bindings("global")
    COMMANDS = {RunFleetCommands}       # RunFleet commands only: Keys/Maximize/Screenshot/Theme competed with ? and did nothing useful
    ENABLE_COMMAND_PALETTE = True

    def __init__(self, service: FleetAPI, *, clock: Clock | None = None, animations: bool = True,
                 persist_state: bool = True, open_settings: bool = False,
                 initial_state: dict[str, Any] | None = None, check_updates: bool = False) -> None:
        # mouse comes from service.config.general.mouse (cli applies --no-mouse/--readonly to the FleetConfig
        # with dataclasses.replace before building FleetService); clock None -> RealClock
        if os.environ.get("NO_COLOR") == "":
            # no-color.org: empty = unset (theme.no_color); Textual strips colour when the variable merely exists,
            # which would leave colourless rows without the -no-color attribute fallbacks (invisible cursor row)
            del os.environ["NO_COLOR"]
        super().__init__(ansi_color=True)    # lets the stylesheet use the terminal's default background
        self.register_theme(theme.TEXTUAL_THEME)
        self.theme = theme.TEXTUAL_THEME.name
        self.service = service
        self.clock: Clock = clock or RealClock()
        self.animations = animations
        if not animations:
            self.animation_level = "none"
        self.scope: str | None = None
        self.multiplier = min(4.0, max(0.5, service.config.general.refresh_multiplier))   # = the core's rate
        self.open_settings = open_settings          # a config without hosts: start in Settings › Hosts
        self.others_visible = False      # Jobs view `u`: header totals then count other users' jobs too
        # --demo keeps UI state in memory only: it must never touch the user's real ui.json
        self._state_path: Path | None = ui_state_path() if persist_state else None
        self.ui_state: dict[str, Any] = self._load_ui_state()
        self.ui_state.update(initial_state or {})   # --demo: a watched job and a last visit to show off (§7.14)
        self._pending: list[ChangeSet] = []
        self._flush_scheduled = False
        self._last_flush = 0.0
        self._host_states: dict[str, ConnState] = {}
        self._toasts: deque[float] = deque()
        self._suppressed = 0
        self._last_input = self.clock.monotonic()
        self._focused = True
        self._unsubscribe: Any = None
        self.crash_report: Path | None = None
        self._notices_seen = 0
        self._last_push = float("-inf")
        self._doctor_held: dict[str, tuple[float, Any]] = {}   # host -> (monotonic, DoctorReport) held back
        self._connecting: set[str] = set()
        self.alerts = Alerts(self)              # notifications, watch list, "since you were away" (§7.14)
        self.check_updates = check_updates      # ask GitHub for a newer RunFleet at start (§8.6; never --demo/tests)
        self.update_status: Any = None          # update.Status of the last check this run (None: not asked yet)
        self.update_checking = False
        self.log_search_query: Any = None       # the last Search logs… query this run (the prompt starts with it)

    # ------------------------------------------------------------------ lifecycle
    def get_theme_variable_defaults(self) -> dict[str, str]:
        """$rf-* exist under every theme (the stylesheet is parsed before ours is set)."""
        return {**super().get_theme_variable_defaults(), **theme.TEXTUAL_THEME.variables}

    def get_driver_class(self) -> type[Driver]:
        return _cell_mouse_driver(super().get_driver_class())

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """App.run with the configured mouse mode; prints the crash report path after the terminal is restored."""
        kwargs.setdefault("mouse", self.service.config.general.mouse)
        previous = sys.excepthook

        def hook(kind: type[BaseException], exc: BaseException, tb: Any) -> None:
            self.crash_report = self.crash_report or write_crash_report(exc)
            previous(kind, exc, tb)

        sys.excepthook = hook
        try:
            return super().run(*args, **kwargs)
        finally:
            sys.excepthook = previous
            if self.crash_report is not None:
                print(f"RunFleet crashed — report written to {self.crash_report}", file=sys.stderr)

    def _handle_exception(self, error: Exception) -> None:
        self.crash_report = self.crash_report or write_crash_report(error)
        super()._handle_exception(error)

    def on_mount(self) -> None:
        from runfleet.ui.screens.main import MainScreen

        self.set_class(theme.no_color(), "-no-color")     # tcss: attribute-only cursor rows (§7.11)
        self.set_class(theme.PLAIN_CURSOR, "-plain-cursor")   # tcss: reverse cursor rows (ansi / 16 colours)
        self._host_states = {h: s.status.state for h, s in self.store.hosts.items()}
        self._unsubscribe = self.store.subscribe(self._on_store_change)
        self.alerts.mount()
        self.push_screen(MainScreen())
        if self.open_settings:
            self.action_settings()
        self.run_worker(self.service.start(), name="service", exit_on_error=False)
        if self.animations:
            self.set_interval(1.0, self._tick)
        self.check_size()
        if self.check_updates:
            self.check_for_update()

    async def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        self.alerts.unmount()
        try:
            await self.service.stop()
        except Exception:   # never let shutdown raise
            log.exception("service.stop failed")

    @property
    def store(self) -> Any:
        return self.service.store

    @property
    def readonly(self) -> bool:
        return bool(self.service.readonly)

    # ------------------------------------------------------------------ store bridge
    def _on_store_change(self, cs: ChangeSet) -> None:
        self._pending.append(cs)
        if self._flush_scheduled:
            return
        self._flush_scheduled = True
        delay = FLUSH_INTERVAL - (time.monotonic() - self._last_flush)
        if delay <= 0:
            self.call_later(self._flush_changes)
        else:
            self.set_timer(delay, self._flush_changes)

    def _flush_changes(self) -> None:
        self._flush_scheduled = False
        if not self._pending:
            return
        changes, self._pending = self._pending, []
        self._last_flush = time.monotonic()
        self._check_transitions(changes)
        self.alerts.on_changes(changes)
        self._forget_removed_hosts()
        self._show_notices()
        for screen in self.screen_stack:
            if self._host_gone(screen):
                continue
            if isinstance(screen, FleetScreen) or getattr(screen, "WANTS_FLEET_CHANGES", False):
                screen.post_message(FleetChanged(changes))

    def _forget_removed_hosts(self) -> None:
        """A host left the fleet (Settings): drop its scope and close detail screens that show it."""
        hosts = self.store.hosts
        for name in [h for h in self._host_states if h not in hosts]:
            del self._host_states[name]
        if self.scope is not None and self.scope not in hosts:
            self.set_scope(None)
        # detail screens of a removed host close once they are on top (Settings may still be open above them)
        while len(self.screen_stack) > 2 and isinstance(self.screen, FleetScreen) and self._host_gone(self.screen):
            self.pop_screen()

    def _host_gone(self, screen: Screen[Any]) -> bool:
        host = _screen_host(screen)
        return host is not None and host not in self.store.hosts

    def _check_transitions(self, changes: Sequence[ChangeSet]) -> None:
        for cs in changes:
            if not cs.status or cs.host not in self.store.hosts:
                continue
            new = self.store.hosts[cs.host].status.state
            old = self._host_states.get(cs.host)
            self._host_states[cs.host] = new
            if old is None or old is new:
                continue
            # between problem states only a change of what the user has to do is news (unreachable -> needs auth:
            # waiting no longer helps); unreachable <-> retrying flapping stays quiet
            if (old in PROBLEM_STATES) == (new in PROBLEM_STATES) and new not in (ConnState.NEEDS_AUTH,
                                                                                 ConnState.HOSTKEY):
                continue
            if new is ConnState.NEEDS_AUTH:
                self.toast(f"{cs.host}: needs auth — press c to log in", severity="warning")
            elif new is ConnState.HOSTKEY:
                self.toast(f"{cs.host}: host key problem — press e for details", severity="error")
            elif new is ConnState.UNREACHABLE:
                self.toast(f"{cs.host}: unreachable — retrying in the background", severity="error")
            else:
                self.toast(f"{cs.host} is back ({new.value})")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """A failed worker is a toast and a log line, never a crash (§7.12)."""
        if event.state is WorkerState.SUCCESS and event.worker.name == "service":
            self._show_notices()
        if event.state is WorkerState.ERROR:
            error = event.worker.error
            log.error("worker %s failed", event.worker.name, exc_info=error)
            what = event.worker.name or event.worker.group or "task"
            self.toast(f"{what} failed: {error}", severity="error")

    def _show_notices(self) -> None:
        """One-shot warnings from ``FleetAPI.notices`` (config warnings, second instance, corrupt cache) as toasts."""
        notices = self.service.notices
        for message in notices[self._notices_seen:]:
            self.toast(str(message), severity="warning", timeout=10)
        self._notices_seen = len(notices)

    def _tick(self) -> None:
        screen = self.base_screen()
        if screen is not None and not self._host_gone(screen):
            screen.post_message(ClockTick())

    def now(self) -> Any:
        return self.clock.now()

    # ------------------------------------------------------------------ activity / idle
    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, (events.Key, events.MouseDown, events.MouseScrollDown, events.MouseScrollUp)):
            self.alerts.note_input(self.clock.monotonic() - self._last_input)
            self._last_input = self.clock.monotonic()
            try:
                self.service.note_user_activity()
            except Exception:
                log.exception("note_user_activity failed")
        await super().on_event(event)

    def on_app_focus(self) -> None:
        self._focused = True
        self.service.set_app_focus(True)

    def on_app_blur(self) -> None:
        self._focused = False
        self.service.set_app_focus(False)

    def idle_throttled(self) -> bool:
        """True while the core polls ×4 slower (no input for ``idle_after`` or terminal unfocused)."""
        idle = self.clock.monotonic() - self._last_input
        return (not self._focused) or idle >= self.service.config.general.idle_after

    def on_resize(self, event: events.Resize) -> None:
        self.check_size()

    def check_size(self) -> None:
        """Push / pop the "terminal too small" placeholder (screens call this on every resize)."""
        from runfleet.ui.screens.main import TooSmallScreen

        small = self.size.width < MIN_SIZE[0] or self.size.height < MIN_SIZE[1]
        top_small = isinstance(self.screen, TooSmallScreen)
        if small and not top_small:
            self.push_screen(TooSmallScreen())
        elif not small:
            # push_screen keeps the placeholder on top, so every one of them sits at the top of the stack
            while isinstance(self.screen, TooSmallScreen) and len(self.screen_stack) > 1:
                self.pop_screen()

    # ------------------------------------------------------------------ screens and scope
    def base_screen(self) -> FleetScreen | None:
        """Topmost full RunFleet screen (modals excluded)."""
        for screen in reversed(self.screen_stack):
            if isinstance(screen, FleetScreen):
                return screen
        return None

    def host_names(self) -> list[str]:
        return list(self.store.hosts)

    def host_index(self, host: str) -> int:
        names = self.host_names()
        return names.index(host) if host in names else 0

    def scope_hosts(self) -> list[str]:
        return [self.scope] if self.scope is not None else self.host_names()

    def set_scope(self, host: str | None) -> None:
        if host is not None and host not in self.store.hosts:
            return
        self.scope = host
        from runfleet.ui.screens.main import MainScreen

        for screen in self.screen_stack:
            if isinstance(screen, MainScreen):
                screen.post_message(ScopeChanged(host))
        self.update_visibility()

    def update_visibility(self) -> None:
        screen = self.base_screen()
        shown = screen.visibility() if screen is not None else {}
        levels = {h: shown.get(h, Visibility.HIDDEN) for h in self.host_names()}
        try:
            self.service.set_visibility(levels)
        except Exception:
            log.exception("set_visibility failed")

    def main_screen(self) -> Any:
        from runfleet.ui.screens.main import MainScreen

        for screen in self.screen_stack:
            if isinstance(screen, MainScreen):
                return screen
        return None

    def go_main(self, view: str | None = None) -> Any:
        """Pop back to the main screen, optionally switching its view."""
        main = self.main_screen()
        while self.screen is not main and len(self.screen_stack) > 2:
            self.pop_screen()
        if main is not None and view is not None:
            main.show_view(view)
        return main

    # ------------------------------------------------------------------ toasts
    def toast(self, message: str, *, severity: str = "information", timeout: float | None = None) -> None:
        """App.notify with the §7.7 rate limit (3 per 10 s, extras coalesced into the next toast)."""
        now = time.monotonic()
        while self._toasts and now - self._toasts[0] > TOAST_WINDOW:
            self._toasts.popleft()
        if len(self._toasts) >= TOAST_MAX and severity != "error":
            self._suppressed += 1
            return
        self._toasts.append(now)
        if self._suppressed:
            message = f"{message}  (+{self._suppressed} more)"
            self._suppressed = 0
        if timeout is None:
            timeout = 10.0 if severity == "error" else 4.0
        self.notify(message, severity=severity, timeout=timeout, markup=False)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ helpers for the current context
    def _current_host(self) -> str | None:
        screen = self.base_screen()
        host = screen.current_host() if screen is not None else None
        return host or self.scope

    def _current_job(self) -> Job | None:
        screen = self.base_screen()
        return screen.current_job() if screen is not None else None

    def host_state(self, host: str) -> HostState | None:
        return self.store.hosts.get(host)

    # ------------------------------------------------------------------ global actions (keymap "global")
    def action_help(self) -> None:
        from runfleet.ui.screens.help import HelpScreen

        if isinstance(self.screen, HelpScreen):
            return
        from runfleet.ui.screens.settings import SettingsScreen

        screen = self.base_screen()
        contexts = screen.help_contexts() if screen is not None else ["global"]
        glyphs = screen.legend_contexts() if screen is not None else contexts
        title = screen.help_title() if screen is not None else "RunFleet"
        dialog = getattr(self.screen, "HELP_CONTEXT", None)
        if isinstance(self.screen, SettingsScreen):
            contexts, title = ["settings", "global"], "Settings"
            glyphs = contexts
        elif isinstance(self.screen, ModalScreen) and isinstance(dialog, str):
            # a dialog's keys only: app bindings (palette, scope, refresh …) stop at a modal screen
            contexts, title = [dialog], keymap.CONTEXT_TITLES[dialog]
            glyphs = contexts
        self.push_screen(HelpScreen(contexts, title, glyphs))

    def action_back_or_quit(self) -> None:
        """``q``: back one screen; quit on the main screen. A `q` that arrives right after a screen was pushed was
        typed for that screen, and so was one right after a screen closed (keys arrive in bursts over slow links): it
        never quits."""
        from runfleet.ui.screens.main import MainScreen

        if isinstance(self.screen, MainScreen):
            if time.monotonic() - self._last_push < PUSH_GRACE:
                return
            self.exit()
        elif len(self.screen_stack) > 2:
            self.pop_screen()

    def action_escape(self) -> None:
        """``esc``: close overlay -> clear filter -> clear marks -> back (never quits)."""
        from runfleet.ui.screens.main import MainScreen

        screen = self.screen
        if isinstance(screen, FleetScreen) and screen.escape():
            return
        if not isinstance(screen, MainScreen) and len(self.screen_stack) > 2:
            self.pop_screen()

    def action_home(self) -> None:
        """Header logo click: close every detail screen and show the Fleet view."""
        self.go_main("fleet")

    def action_settings(self, section: str = "hosts") -> None:
        """``,`` / palette › Settings / header chip: the settings overlay (``,`` again closes it)."""
        from runfleet.ui.screens.settings import SettingsScreen

        if isinstance(self.screen, SettingsScreen):
            self.screen.dismiss(None)
            return
        self.push_screen(SettingsScreen(section), lambda _result: self._forget_removed_hosts())

    def set_theme_choice(self, choice: str) -> Any:
        """Settings › About › theme / palette › Theme: switch the palette now and keep the choice (ui.json)."""
        resolved = theme.setup(choice, theme.BACKGROUND)
        self.register_theme(theme.TEXTUAL_THEME)
        self.theme = theme.TEXTUAL_THEME.name
        self.set_class(theme.PLAIN_CURSOR, "-plain-cursor")
        self.save_ui_state(theme=choice)
        for screen in self.screen_stack:          # every cached Rich text is rebuilt with the new colours
            for node in (screen, *screen.walk_children()):
                forget = getattr(node, "forget_rendered", None)
                if callable(forget):
                    forget()
            if isinstance(screen, FleetScreen) and not self._host_gone(screen):   # a removed host's detail screen
                screen.refresh_data(None)                                         # under Settings has no state
            screen.refresh(repaint=True)
        self.refresh_headers()
        return resolved

    def set_fleet_layout(self, choice: str) -> None:
        """Palette › Fleet: roomy / compact cards: ``auto`` grows the Fleet cards into a tall terminal (§7.4),
        ``compact`` keeps them at their summary size. Applied now and kept in ui.json ``fleet_layout``."""
        self.save_ui_state(fleet_layout=choice if choice in ("auto", "compact") else "auto")
        for screen in self.screen_stack:
            if isinstance(screen, FleetScreen) and not self._host_gone(screen):
                screen.refresh_data(None)
        self.toast("Fleet: roomy cards when the terminal has room" if choice != "compact" else "Fleet: compact cards",
                   timeout=3)

    def open_update(self) -> None:
        """Palette › Update RunFleet: Settings › About on the version row, ready to update (or to check first)."""
        from runfleet.ui.screens.settings import SettingsScreen

        if isinstance(self.screen, SettingsScreen):
            self.screen.dismiss(None)
        self.push_screen(SettingsScreen("about", update=True), lambda _result: self._forget_removed_hosts())

    # ------------------------------------------------------------------ updates (§8.6)
    def check_for_update(self, *, force: bool = False) -> None:
        """Ask GitHub for the newest version in a thread (the saved answer while younger than a day, unless
        `force`); a newer one is announced once per version. Settings › About shows the result either way."""
        if self.update_checking:
            return
        self.update_checking = True
        self._update_changed()
        self.run_worker(lambda: self._check_update_thread(force), thread=True, group="update-check",
                        exit_on_error=False)

    def _check_update_thread(self, force: bool) -> None:
        from runfleet import update

        status = update.check(self.clock.now(), force=force)
        self.call_from_thread(self._update_checked, status, force)

    def _update_checked(self, status: Any, force: bool) -> None:
        from runfleet import update

        self.update_checking = False
        self.update_status = status
        self._update_changed(status if force else None)
        if not force and update.should_notify(status):
            update.mark_notified(status.latest)
            how = ("quit it, then run " + update.detect_install().manual if rfplatform.IS_WINDOWS
                   else "Update RunFleet in the palette (ctrl+p) updates and restarts")
            self.toast(f"RunFleet {status.latest} is out (this is {status.current}) — {how}", timeout=12.0)

    def _update_changed(self, answer: Any = None) -> None:
        from runfleet.ui.screens.settings import SettingsScreen

        for screen in self.screen_stack:
            if isinstance(screen, SettingsScreen):
                screen.update_changed(answer)

    def request_update(self) -> None:
        """Quit; the CLI updates this install in the plain terminal and starts RunFleet again (same arguments).
        Never on Windows (Settings › About shows the manual steps there instead)."""
        from runfleet import update

        if rfplatform.IS_WINDOWS:
            self.toast(f"quit RunFleet, then run {update.detect_install().manual}", severity="warning")
            return
        self.exit(return_code=update.RESTART_CODE)

    def set_mouse(self, enabled: bool) -> None:
        """Settings › mouse: switch mouse reporting now (the driver keeps it across suspend / resume)."""
        driver: Any = self._driver
        if driver is None or not hasattr(driver, "_enable_mouse_support"):
            return
        if enabled:
            driver._mouse = True
            driver._enable_mouse_support()
        else:
            driver._disable_mouse_support()
            driver._mouse = False

    def action_press(self, key: str) -> None:
        """Hint-bar click: behave exactly like pressing ``key`` in the view. A focused input (the Jobs filter) would
        take the key as text, so focus first goes back to the view's table (the filter itself stays)."""
        from textual.widgets import DataTable, Input

        focused = self.focused
        if isinstance(focused, Input):
            parent = focused.parent
            table = next((t for t in parent.query(DataTable) if t.focusable), None) if parent is not None else None
            if table is not None:
                table.focus()
            else:
                self.screen.set_focus(None)
        self.simulate_key(key)

    def action_scope_all(self) -> None:
        self.set_scope(None)

    def action_scope_index(self, n: int) -> None:
        names = self.host_names()
        if 1 <= n <= len(names):
            self.set_scope(names[n - 1])

    def action_filter(self) -> None:
        main = self.go_main("jobs")
        if main is not None:
            main.open_filter()

    def action_refresh_scope(self) -> None:
        screen = self.base_screen()
        host = screen.current_host() if screen is not None and not self._is_main() else None
        hosts = [host] if host else self.scope_hosts()
        self.service.refresh(hosts)
        self.toast(f"refreshing {', '.join(hosts) if len(hosts) <= 3 else f'{len(hosts)} hosts'}")

    def action_refresh_all(self) -> None:
        self.service.refresh(None)
        self.toast("refreshing all hosts")

    def action_rate(self, factor: float) -> None:
        self.multiplier = max(0.5, min(4.0, self.multiplier * factor))
        self.service.set_rate_multiplier(self.multiplier)
        self.toast(f"poll interval ×{self.multiplier:g}" + (" (floors apply)" if self.multiplier < 1 else ""))
        self.refresh_headers()

    def _is_main(self) -> bool:
        from runfleet.ui.screens.main import MainScreen

        return isinstance(self.base_screen(), MainScreen)

    def refresh_headers(self) -> None:
        from runfleet.ui.widgets.header import FleetHeader

        for screen in self.screen_stack:
            for header in screen.query(FleetHeader):
                header.refresh_content()

    # ------------------------------------------------------------------ host actions
    def action_connect_host(self, host: str | None = None) -> None:
        host = host or self._current_host()
        if host is None:
            self.toast("no host selected", severity="warning")
            return
        self.connect(host)

    @work(exit_on_error=False, group="connect")
    async def connect(self, host: str) -> None:
        if host in self._connecting:           # a second `c` during a slow master check: one login, not two
            self.toast(f"{host}: already connecting")
            return
        self._connecting.add(host)
        try:
            await self._connect(host)
        finally:
            self._connecting.discard(host)

    async def _connect(self, host: str) -> None:
        if await self.service.prepare_connect(host):
            await self.service.after_connect(host, 0)
            self.toast(f"{host}: already connected")
            return
        argv = self.service.connect_argv(host)
        if argv is None:
            self.toast(f"{host}: nothing to connect (local host)")
            return
        state = self.host_state(host)
        rc = self.run_external(argv, banner=f"RunFleet → connecting to {host} (Duo/password/host-key prompts appear "
                               "here; Ctrl-C aborts)", fail_log=state.status.ssh_log if state else None,
                               connect_host=host)
        if rc is not None:
            await self.service.after_connect(host, rc)
            if rc == 0:
                self.toast(f"{host}: connected")
            else:
                self.toast(f"{host}: ssh exited with {rc}", severity="error")

    def action_shell_host(self, host: str | None = None) -> None:
        host = host or self._current_host()
        if host is None:
            return
        rc = self.run_external(self.service.shell_argv(host), pause_on_error=True)
        if rc is not None:
            self._after_interactive(host)

    def action_shell_job(self) -> None:
        """A running Slurm job opens a shell on its node inside the job (a multi-node job asks which node); anything
        else opens the host shell in the job's directory."""
        job = self._current_job()
        if job is None:
            self.action_shell_host()
            return
        if job.kind is JobKind.SLURM and job.state is JobState.RUNNING and len(job.nodes) > 1:
            from runfleet.ui.screens.action_menu import ActionMenuScreen

            first = job.batch_host if job.batch_host in job.nodes else job.nodes[0]
            nodes = [first, *(n for n in job.nodes if n != first)]
            specs = [ActionSpec(n, n + ("  (batch host)" if n == job.batch_host else ""), mutating=False,
                                confirm=False) for n in nodes]

            def chosen(node: str | None) -> None:
                if node:
                    self._shell_job_on(job, node)

            self.push_screen(ActionMenuScreen(specs, len(nodes), title=f"Shell into job {job.key.id} on…"), chosen)
            return
        self._shell_job_on(job, None)

    def _shell_job_on(self, job: Job, node: str | None) -> None:
        if node is not None:                   # picked from a list that may be stale: the job can requeue meanwhile
            fresh = self.store.job(job.key)
            if fresh is None or fresh.state is not JobState.RUNNING or node not in fresh.nodes:
                self.toast(f"{node} is no longer part of job {job.key.label} — press s again", severity="warning")
                return
        rc = self.run_external(self.service.shell_argv(job.key.host, job.key, node), pause_on_error=True)
        if rc is not None:
            self._after_interactive(job.key.host)

    def _after_interactive(self, host: str) -> None:
        """After a shell or pager: on a host paused for a login, that session may have logged in (managed mode makes
        it the master), so re-prepare and poll now as after a connect; otherwise just refresh."""
        state = self.host_state(host)
        if state is not None and state.status.state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
            self._reconnect_after(host)
        else:
            self.service.refresh([host])

    @work(exit_on_error=False, group="connect")
    async def _reconnect_after(self, host: str) -> None:
        await self.service.after_connect(host, 0)

    def action_toggle_host(self, host: str | None = None) -> None:
        host = host or self._current_host()
        state = self.host_state(host) if host else None
        if host is None or state is None:
            return
        enable = state.status.state is ConnState.DISABLED
        self.service.set_host_enabled(host, enable)
        self.toast(f"{host}: polling {'enabled' if enable else 'disabled'}")

    def action_host_error(self, host: str | None = None) -> None:
        from runfleet.ui.screens.host_error import HostErrorScreen

        host = host or self._current_host()
        if host is not None:
            self.push_screen(HostErrorScreen(host))

    def action_chip(self, host: str) -> None:
        """Header chip click: problem chips open the error screen, others scope to the host."""
        state = self.host_state(host)
        if state is not None and state.status.state in PROBLEM_STATES | {ConnState.RETRYING}:
            self.action_host_error(host)
        else:
            self.set_scope(None if self.scope == host else host)

    def action_open_host(self, host: str) -> None:
        from runfleet.ui.screens.host_detail import HostDetailScreen

        if host in self.store.hosts:
            self.push_screen(HostDetailScreen(host))

    def action_open_job(self, key: str) -> None:
        host, _, jid = key.partition(":")
        self.open_job(JobKey(host, jid))

    def open_job(self, key: JobKey) -> None:
        from runfleet.ui.screens.job_detail import JobDetailScreen

        if self.store.job(key) is not None:
            self.push_screen(JobDetailScreen(key))

    # ------------------------------------------------------------------ logs
    def log_ref(self, job: Job, stream: str = "stdout") -> LogRef | None:
        refs = [r for r in job.logs if r.stream == stream]
        if not refs and stream == "stderr":
            refs = [r for r in job.logs if r.stream == "stdout"]
        return refs[0] if refs else None

    def action_log_viewer(self, stream: str = "stdout", search: bool = False) -> None:
        from runfleet.ui.screens.log_viewer import LogScreen

        job = self._current_job()
        if job is None:
            self.toast("no job selected", severity="warning")
            return
        if not job.logs:
            self.toast(f"{job.key.label}: no log path known", severity="warning")
            return
        screen = self.base_screen()
        if isinstance(screen, LogScreen):
            return
        cur = getattr(screen, "stream", stream)
        log_screen = LogScreen(job.key, cur if isinstance(cur, str) else stream)
        log_screen.open_search = search
        self.push_screen(log_screen)

    def action_pager(self) -> None:
        job = self._current_job()
        if job is None:
            return
        screen = self.base_screen()
        stream = getattr(screen, "stream", "stdout")
        ref = self.log_ref(job, stream if isinstance(stream, str) else "stdout")
        if ref is None or ref.path is None:
            self.toast(f"{job.key.label}: no log path known", severity="warning")
            return
        self.open_pager(job.key.host, ref)

    @work(exit_on_error=False, group="pager")
    async def open_pager(self, host: str, ref: LogRef) -> None:
        argv = self.service.pager_argv(host, ref)
        if argv is None:
            path = await self.service.fetch_log(host, ref)
            argv = self.service.local_pager_argv(path)
            self.run_external(argv)
            return
        if self.run_external(argv) is not None:
            self._after_interactive(host)

    # ------------------------------------------------------------------ copy
    def action_copy_id(self) -> None:
        job = self._current_job()
        if job is None:
            return
        text = job.key.id if job.kind is JobKind.SLURM else job.job_id_raw   # tasks: the root pid
        self.copy_to_clipboard(text)
        self.toast(f"sent to clipboard (OSC 52): {text}", timeout=8)   # no acknowledgement: the text stays readable

    def action_copy_path(self) -> None:
        job = self._current_job()
        ref = self.log_ref(job) if job is not None else None
        if ref is None or ref.path is None:
            self.toast("no stdout path known", severity="warning")
            return
        self.copy_to_clipboard(ref.path)
        self.toast(f"sent to clipboard (OSC 52): {ref.path}", timeout=8)

    # ------------------------------------------------------------------ everyday actions (ui/everyday.py)
    def action_copy_menu(self) -> None:
        """``Y``: copy id, host:jobid, workdir, log paths, the node-shell command or a support bundle."""
        from runfleet.ui import everyday
        everyday.copy_menu(self)

    def action_forward_port(self) -> None:
        """``P`` / palette / action menu: forward a port of the current job (or host) to localhost."""
        from runfleet.ui import everyday
        everyday.forward_port(self)

    def show_forwards(self) -> None:
        from runfleet.ui import everyday
        everyday.show_forwards(self)

    def action_login_all(self) -> None:
        """``C`` (Fleet) / palette: log in to every host that needs it, in one suspend."""
        from runfleet.ui import everyday
        self.run_worker(everyday.login_all(self), group="connect", exclusive=False, exit_on_error=False)

    def action_free_gpus(self) -> None:
        """``a`` (GPUs view, workstation host detail): CUDA_VISIBLE_DEVICES for N free GPUs."""
        from runfleet.ui import everyday
        everyday.free_gpu_menu(self)

    def bundle_dir(self) -> Path | None:
        """Where support bundles are kept (``<state dir>/bundles``); None without persisted state (--demo)."""
        return self._state_path.parent / "bundles" if self._state_path is not None else None

    # ------------------------------------------------------------------ actions (§6.3)
    def _targets(self) -> list[JobKey]:
        screen = self.base_screen()
        return screen.action_targets() if screen is not None else []

    def hint_overrides(self, contexts: Sequence[str]) -> dict[str, str | None]:
        """Hint-bar labels for the current selection: `^k` says what it would run (cancel / terminate) and is hidden
        when nothing selected can get it (no selection, only ended jobs); the action menu likewise."""
        try:
            jobs = [j for j in (self.store.job(k) for k in self._targets()) if j is not None]
        except Exception:                      # hints must never break the UI
            return {}
        live = [j for j in jobs if not j.state.is_terminal]
        out: dict[str, str | None] = {}
        if not live:
            out["app.quick_kill"] = None
        else:
            out["app.quick_kill"] = "terminate" if all(j.kind is JobKind.TASK for j in live) else "cancel"
        if not jobs:
            out["app.action_menu"] = None
        return out

    def action_action_menu(self) -> None:
        from runfleet.ui.screens.action_menu import ActionMenuScreen

        keys = self._targets()
        if not keys:
            self.toast("nothing selected", severity="warning")
            return
        specs = self.service.actions_for(keys)
        if not specs:
            self.toast("no actions for this selection", severity="warning")
            return

        def chosen(action_id: str | None) -> None:
            if action_id:
                self.start_action(action_id, keys)

        self.push_screen(ActionMenuScreen(specs, len(keys)), chosen)

    def action_quick_kill(self) -> None:
        """``ctrl+k``: cancel (Slurm) / terminate (task) the targets, through the confirm dialog."""
        keys = self._targets()
        if not keys:
            self.toast("nothing selected", severity="warning")
            return
        if self.readonly:
            self.toast("read-only mode — actions are disabled", severity="warning")
            return
        jobs = [j for j in (self.store.job(k) for k in keys) if j is not None]
        action_id = "task.terminate" if jobs and all(j.kind is JobKind.TASK for j in jobs) else "slurm.cancel"
        spec = next((s for s in self.service.actions_for(keys) if s.id == action_id), None)
        if spec is None:
            self.toast("cancel / terminate does not apply here", severity="warning")
            return
        if spec.disabled_reason:
            self.toast(f"{spec.label}: {spec.disabled_reason}", severity="warning")
            return
        self.start_action(action_id, keys)

    def start_action(self, action_id: str, keys: Sequence[JobKey]) -> None:
        from runfleet.ui.screens.confirm import ConfirmScreen, dropped_targets

        try:
            plans = self.service.plan(action_id, keys)
        except ActionError as exc:
            self.toast(str(exc), severity="warning")
            return

        def confirmed(result: Any) -> None:
            if result:
                plans = list(result)
                skipped = [s for p in plans for s in p.skipped] + dropped_targets(self.service, action_id, keys, plans)
                self.execute_plans(plans, skipped)

        self.push_screen(ConfirmScreen(action_id, list(keys), plans), confirmed)

    @work(exit_on_error=False, group="actions")
    async def execute_plans(self, plans: list[Any], skipped: Sequence[tuple[JobKey, str]] = ()) -> None:
        results = await self.service.execute(plans)
        self.report_results(results, skipped)
        screen = self.main_screen()
        if screen is not None:                 # skipped and failed targets stay marked: they were not handled
            screen.unmark([r.target for r in results if r.ok])

    def report_results(self, results: Sequence[ActionResult], skipped: Sequence[tuple[JobKey, str]] = ()) -> None:
        note = f" · {len(skipped)} skipped ({skipped[0][1]})" if skipped else ""
        if not results:
            self.toast("action produced no result" + note, severity="warning")
            return
        failed = [r for r in results if not r.ok]
        if failed:
            first = failed[0]
            line = next((x for x in (first.stderr.strip(), *first.details) if x), first.message)
            self.toast(f"{first.action_id} failed on {first.target}: {line.splitlines()[0] if line else ''}"
                       f" — details: palette → Action log{note}", severity="error", timeout=10)
            return
        by: dict[tuple[str, str], int] = {}
        for r in results:
            by[(r.action_id, r.target.host)] = by.get((r.action_id, r.target.host), 0) + 1
        parts = []
        for (aid, host), n in by.items():
            kind, verb = aid.split(".", 1)
            noun = "task" if kind == "task" else "job"
            parts.append(f"{_PAST.get(verb, verb.capitalize())} {n} {noun}{'s' if n != 1 else ''} on {host}")
        self.toast("; ".join(parts) + note, severity="warning" if skipped else "information")

    # ------------------------------------------------------------------ palette commands
    def jobs_setting(self, kind: str, value: Any) -> None:
        """Palette: state preset / sort / grouping of the Jobs view (switches to it)."""
        main = self.go_main("jobs")
        if main is None:
            return
        if kind == "preset":
            main.jobs.set_preset(value)
        elif kind == "sort":
            main.jobs.set_sort(value)
        elif kind == "group":
            main.jobs.set_group(value)

    # ------------------------------------------------------------------ log search (§4.7.1)
    def action_search_logs(self) -> None:
        """Palette › Search logs… / ``^f`` in the Jobs view: the prompt, then one probe per host in scope (on demand
        only) and the results list."""
        from runfleet.ui.screens.log_search import LogSearchScreen, SearchPromptScreen

        if isinstance(self.screen, (SearchPromptScreen, LogSearchScreen)):
            return
        hosts = self.scope_hosts()

        def chosen(query: Any) -> None:
            if query is None:
                return
            self.log_search_query = query

            def done(result: Any) -> None:
                if result == "again":
                    self.action_search_logs()

            self.push_screen(LogSearchScreen(query, hosts), done)

        self.push_screen(SearchPromptScreen(self.log_search_query, hosts), chosen)

    def open_log_hit(self, hit: Any, query: Any = None) -> None:
        """A search result's ``enter``: the log viewer above the results, on the hit's line (§4.7.1)."""
        from runfleet.ui.screens.log_viewer import LogScreen

        job = self.store.job(hit.key)
        if job is None:
            self.toast(f"{hit.key} is no longer listed", severity="warning")
            return
        ref = self.log_ref(job, hit.file.stream)
        screen = LogScreen(hit.key, hit.file.stream)
        if ref is not None and ref.path == hit.file.path:
            screen.hit, screen.hit_query = hit, query
        else:
            self.toast(f"{hit.key.label}: its {hit.file.stream} path changed since the search — showing the current "
                       "log", severity="warning")
        self.push_screen(screen)

    # ------------------------------------------------------------------ saved Jobs views (§7.6)
    def saved_jobs_views(self) -> dict[str, SavedView]:
        """The usable saved Jobs views of ui.json, by name (§7.6)."""
        return saved_views.load_views(self.ui_state)

    def prompt_save_view(self) -> None:
        """Palette › Save view as…: name the Jobs view's filter, preset, sort and grouping (kept in ui.json)."""
        from runfleet.ui.screens.save_view import SaveViewScreen

        main = self.main_screen()
        if main is None:
            return
        jobs = main.jobs

        def chosen(name: str | None) -> None:
            if name:
                self.save_view(name)

        self.push_screen(SaveViewScreen(jobs.current_view(), self.saved_jobs_views(), jobs.active_view or ""), chosen)

    def save_view(self, name: str) -> bool:
        """Store the Jobs view's current settings as ``name`` (replacing a view of that name in any case)."""
        main = self.main_screen()
        problem = saved_views.save_problem(self.saved_jobs_views(), name)
        if main is None or problem is not None:
            if problem is not None:
                self.toast(f"view not saved: {problem}", severity="warning")
            return False
        view = main.jobs.current_view()
        self.save_ui_state(views=saved_views.with_view(self.ui_state, name, view))
        main.jobs.view_saved(name)
        self.toast(f"saved view “{name}” — palette › View: {name} applies it")
        return True

    def apply_saved_view(self, name: str) -> None:
        """Palette › View: <name>: switch to the Jobs view and apply the saved filter, preset, sort and grouping."""
        views = self.saved_jobs_views()
        stored = saved_views.find(views, name)
        if stored is None:
            self.toast(f"no saved view “{name}” any more", severity="warning")
            return
        main = self.go_main("jobs")
        if main is not None:
            main.jobs.apply_view(stored, views[stored])

    def delete_saved_view(self, name: str) -> None:
        """Palette › Delete view: <name>."""
        stored = saved_views.find(self.saved_jobs_views(), name)
        if stored is None:
            return
        self.save_ui_state(views=saved_views.without_view(self.ui_state, stored))
        main = self.main_screen()
        if main is not None:
            main.jobs.view_deleted(stored)
        self.toast(f"deleted view “{stored}”")

    @work(exit_on_error=False, group="doctor")
    async def show_doctor(self, host: str) -> None:
        from runfleet.ui.screens.host_error import DoctorScreen

        held = self._doctor_held.pop(host, None)
        if held is not None and self.clock.monotonic() - held[0] < DOCTOR_HOLD:
            self.push_screen(DoctorScreen(held[1]))
            return
        self.toast(f"running doctor on {host} …")
        try:
            report = await self.service.doctor(host)
        except Exception as exc:   # doctor problems are shown, never raised
            self.toast(f"doctor {host} failed: {exc}", severity="error")
            return
        if not isinstance(self.screen, FleetScreen):   # a dialog opened meanwhile: never pop up on top of it
            self._doctor_held[host] = (self.clock.monotonic(), report)
            self.toast(f"doctor {host} finished — palette › Doctor {host} shows it")
            return
        self.push_screen(DoctorScreen(report))

    def show_action_log(self) -> None:
        from runfleet.ui.screens.action_menu import ActionLogScreen

        self.push_screen(ActionLogScreen(list(self.service.action_log())))

    def open_runfleet_log(self) -> None:
        self.run_external(self.service.local_pager_argv(None))

    # ------------------------------------------------------------------ notifications, watch, digest (§7.14)
    def action_toggle_watch(self) -> None:
        """``w``: watch the selected / marked jobs (start, end, GPU idle notify too); unwatch if all are watched."""
        keys = self._targets()
        on = self.alerts.toggle_watch(keys)
        if on is None:
            self.toast("nothing selected", severity="warning")
            return
        what = keys[0].label if len(keys) == 1 else f"{len(keys)} jobs"
        self.toast(f"watching {what} — start, end and GPU idle notify" if on else f"stopped watching {what}")
        self.refresh_watch_marks()

    def refresh_watch_marks(self) -> None:
        for screen in self.screen_stack:
            if isinstance(screen, FleetScreen) and screen.is_mounted:
                screen.refresh_data(None)

    def action_digest(self) -> None:
        """``D``: what ended since the last visit (counts per host, failures; ``f`` filters the Jobs view)."""
        from runfleet.ui.screens.notifications import DigestScreen

        if isinstance(self.screen, DigestScreen):
            return
        since, jobs = self.alerts.digest()

        def chosen(result: Any) -> None:
            if not result:
                return
            what, key = result
            if what == "open" and key is not None:
                self.open_listed_job(key)
            elif what == "filter":
                main = self.go_main("jobs")
                if main is not None:
                    main.jobs.set_ended_since(since)

        self.push_screen(DigestScreen(since, jobs, self.host_names(), self.now()), chosen)

    def show_notifications(self) -> None:
        """Palette › Notifications: this session's alert history and GPU watches that fired (``enter`` opens the job,
        or the host of a watch)."""
        from runfleet.ui.screens.notifications import NotificationsScreen

        def chosen(picked: JobKey | str | None) -> None:
            if isinstance(picked, JobKey):
                self.open_listed_job(picked)
            elif picked and picked not in self.store.hosts:     # a watch's host removed in Settings since
                self.toast(f"{picked} is no longer in the fleet", severity="warning")
            elif picked:                       # a GPU watch that fired: its host
                self.action_open_host(picked)

        self.push_screen(NotificationsScreen([*self.alerts.history, *self.alerts.watch_events]), chosen)

    def prompt_gpu_watch(self) -> None:
        """Palette › Watch free GPUs…: the prompt; a usable watch is kept (ui.json ``watches``) and evaluated."""
        from runfleet.ui.screens.gpu_watch import WatchPromptScreen

        def chosen(watch: Any) -> None:
            if watch is None:
                return
            problem = self.alerts.add_gpu_watch(watch)
            if problem is not None:
                self.toast(f"watch not added: {problem}", severity="warning")
                return
            self.toast(f"watching for {watch.describe()} — palette › Watches lists it")

        self.push_screen(WatchPromptScreen(), chosen)

    def show_gpu_watches(self) -> None:
        """Palette › Watches: the GPU watches and what each is doing (``d`` removes one)."""
        from runfleet.ui.screens.gpu_watch import WatchesScreen

        if not isinstance(self.screen, WatchesScreen):
            self.push_screen(WatchesScreen())

    def open_listed_job(self, key: JobKey) -> None:
        """Open a job picked from a list that may be older than the store (alerts, the Fleet cards and panel, the
        "needs you" picker): a job that is gone by now gets a toast instead of nothing."""
        if self.store.job(key) is None:
            self.toast(f"{key} is no longer listed", severity="warning")
            return
        self.open_job(key)

    # ------------------------------------------------------------------ external programs (§4.6)
    def action_suspend_process(self) -> None:
        """`^z`: the whole process stops until `fg`; probes in flight then get a fresh timeout budget."""
        with spawn.loop_frozen():
            super().action_suspend_process()

    def run_external(self, argv: list[str], *, banner: str | None = None, fail_log: str | None = None,
                     connect_host: str | None = None, pause_on_error: bool = False) -> int | None:
        """Run ``argv`` in the real terminal with the TUI suspended; the ONLY place that suspends (tests patch it).

        Returns the exit code, or None when suspending is impossible (a modal then shows the command to run; its
        *Done* re-prepares ``connect_host``, which only a login passes). ``pause_on_error``: a command that fails
        within seconds (a shell whose `cd` or ssh failed) waits for Enter, so its error is not hidden by the redraw.
        """
        try:
            with self.suspend(), spawn.loop_frozen():     # the loop is blocked: in-flight probe timers must not count
                if banner:
                    print(banner, flush=True)
                started = time.monotonic()
                try:
                    # Windows: `more` is more.com, `code` code.cmd (CreateProcess only adds .exe)
                    rc = subprocess.run(rfplatform.resolve_program(argv), check=False).returncode
                except OSError as exc:
                    print(f"runfleet: cannot run {argv[0]}: {exc}", flush=True)
                    rc = 127
                except KeyboardInterrupt:
                    rc = 130
                prompt = None
                if rc != 0 and fail_log:
                    _print_log_tail(fail_log)
                    prompt = "ssh failed — press Enter to return to RunFleet "
                elif pause_on_error and rc not in (0, 130) and time.monotonic() - started < QUICK_EXIT:
                    prompt = f"runfleet: {argv[0]} exited with {rc} — press Enter to return to RunFleet "
                if prompt is not None:
                    try:
                        input(prompt)
                    except (EOFError, KeyboardInterrupt):
                        pass
        except SuspendNotSupported:
            from runfleet.ui.screens.host_error import ManualCommandScreen

            self.push_screen(ManualCommandScreen(argv, connect_host))
            return None
        self.refresh(layout=True, repaint=True)
        return rc

    def run_external_sequence(self, steps: Sequence[tuple[list[str], str, str | None]]) -> list[int] | None:
        """Several commands in ONE suspend (log in to all), each after its banner; besides :meth:`run_external` the
        only place that suspends (tests patch it). ``steps``: (argv, banner, ssh log shown when it fails). Ctrl-C
        skips to the next one (rc 130); Enter is asked once at the end when one failed. None: cannot suspend."""
        rcs: list[int] = []
        try:
            with self.suspend(), spawn.loop_frozen():
                for argv, banner, fail_log in steps:
                    print(banner, flush=True)
                    try:
                        rc = subprocess.run(rfplatform.resolve_program(argv), check=False).returncode
                    except OSError as exc:
                        print(f"runfleet: cannot run {argv[0]}: {exc}", flush=True)
                        rc = 127
                    except KeyboardInterrupt:
                        rc = 130
                    if rc not in (0, 130) and fail_log:
                        _print_log_tail(fail_log)
                    rcs.append(rc)
                if any(rc not in (0, 130) for rc in rcs):
                    try:
                        input("some logins failed — press Enter to return to RunFleet ")
                    except (EOFError, KeyboardInterrupt):
                        pass
        except SuspendNotSupported:
            return None
        self.refresh(layout=True, repaint=True)
        return rcs

    # ------------------------------------------------------------------ persisted UI state
    def _load_ui_state(self) -> dict[str, Any]:
        path = self._state_path
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def save_ui_state(self, **values: Any) -> None:
        self.ui_state.update(values)
        path = self._state_path
        if path is None:
            return
        try:
            from runfleet import paths

            paths.ensure_private_dir(path.parent)
            paths.write_private(path, json.dumps(self.ui_state, indent=1, sort_keys=True))
        except OSError:
            log.warning("cannot write %s", path)

    # ------------------------------------------------------------------ misc
    def push_screen(self, screen: Screen[Any] | str, *args: Any, **kwargs: Any) -> Any:
        """Never once exiting (a queued key must not open a screen into a tearing-down DOM), and never above the
        "terminal too small" placeholder: a screen opened while small goes below it, so the placeholder stays on top
        and the resize that pops it reveals the new screen."""
        from runfleet.ui.screens.main import TooSmallScreen

        if self._exit:
            done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        self._last_push = time.monotonic()
        if isinstance(self.screen, TooSmallScreen) and not isinstance(screen, TooSmallScreen):
            self.pop_screen()
            result = super().push_screen(screen, *args, **kwargs)
            super().push_screen(TooSmallScreen())
        else:
            result = super().push_screen(screen, *args, **kwargs)
        self.call_later(self.update_visibility)
        return result

    def pop_screen(self) -> Any:
        self._last_push = time.monotonic()      # any screen change: a `q` right after it was typed for the old one
        return super().pop_screen()

    async def run_action(self, action: Any, default_namespace: Any = None, namespaces: Any = None) -> bool:
        """Drop key actions once exiting, and every action but quit/suspend while the terminal is too small."""
        from runfleet.ui.screens.main import TooSmallScreen

        if self._exit:
            return False
        name = action if isinstance(action, str) else ""
        if isinstance(self.screen, TooSmallScreen) and name not in ("app.quit", "quit", "app.suspend_process",
                                                                    "suspend_process"):
            return False
        return await super().run_action(action, default_namespace, namespaces)


def _screen_host(screen: Screen[Any]) -> str | None:
    """The host a detail screen is about (host detail, job detail, log viewer), None for the main screen."""
    host = getattr(screen, "host", None)
    if isinstance(host, str):
        return host
    key = getattr(screen, "key", None)
    return key.host if isinstance(key, JobKey) else None


def _print_log_tail(path: str, n: int = 20) -> None:
    try:
        lines = Path(path).expanduser().read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return
    if lines:
        print(f"--- last lines of {path} ---")
        print("\n".join(lines), flush=True)
