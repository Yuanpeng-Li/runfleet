"""[WP3] Theme (§7.11): mode resolution, colours mixed from the terminal background (readable on common profiles),
the ansi mode, the OSC 11 query (over a pty), clickable bars keeping their colours, live switching."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from runfleet.ui import termbg, theme
from runfleet.ui.widgets.header import FleetHeader
from runfleet.ui.widgets.hintbar import HintBar
from runfleet.ui.widgets.job_table import ClickTable
from runfleet.ui.widgets.tabs import ViewTabs

DARK = ("#000000", "#1f1f1f", "#15191f", "#1e1e2e", "#002b36", "#282a36", "#2e3440")
LIGHT = ("#ffffff", "#fafafa", "#fdf6e3", "#eff1f5")


@pytest.fixture(autouse=True)
def restore() -> Iterator[None]:
    yield
    theme.configure("dark")


def test_mode_precedence() -> None:
    assert theme.resolve("auto", env={"RUNFLEET_THEME": "ansi"}, background="#ffffff").mode == "ansi"
    assert theme.resolve("light", env={}, background="#000000") == theme.Resolved("light", "#000000", "setting")
    assert theme.resolve("auto", env={}, color_system="standard").source == "16 colours"
    assert theme.resolve("auto", env={}, background="#fafafa").mode == "light"
    assert theme.resolve("auto", env={}, background="#1e1e2e") == theme.Resolved("dark", "#1e1e2e",
                                                                                 "terminal background")
    assert theme.resolve("auto", env={"COLORFGBG": "0;15"}) == theme.Resolved("light", None, "COLORFGBG")
    assert theme.resolve("auto", env={}) == theme.Resolved("dark", None, "default")
    assert theme.resolve("auto", env={}, background="nonsense").background is None


def test_black_background_keeps_the_classic_palette() -> None:
    theme.configure("dark", "#000000")
    assert (theme.SURFACE, theme.TRACK, theme.RULE, theme.MUTED, theme.PANEL) == (
        "#1c1c1c", "#262626", "#3b3b3b", "#a8a8a8", "#262626")   # = the old fixed greys (±1)
    assert theme.CURSOR == "#172a3d"                                # the accent at 24 %


@pytest.mark.parametrize("bg", DARK + LIGHT)
def test_derived_colours_are_readable_on_common_backgrounds(bg: str) -> None:
    theme.configure(theme.resolve("auto", env={}, background=bg).mode, bg)
    for name in ("MUTED", "GREY", "FOREGROUND", "GREEN", "YELLOW", "RED", "ACCENT"):
        assert theme.contrast(getattr(theme, name), bg) >= 4, name
    for name in ("DIM", "BLUE_DIM", "CANCELLED"):
        assert theme.contrast(getattr(theme, name), bg) >= 3, name
    assert theme.contrast(theme.BORDER, bg) >= 2                  # frames: visible, not loud
    assert theme.contrast(theme.TRACK, bg) >= 1.15                 # a bar's empty part still reads as a bar
    assert theme.contrast(theme.MUTED, theme.CURSOR) >= 3.5        # text on the cursor row


def test_fills_take_the_backgrounds_hue() -> None:
    theme.configure("dark", "#002b36")                             # Solarized: teal, not neutral grey boxes
    r, g, b = theme.parse_hex(theme.TRACK) or (0, 0, 0)
    assert b > r and g > r
    theme.configure("light", "#000000")                            # a forced mode ignores a background it contradicts
    assert theme.SURFACE == "#f2f2f2"


def test_256_colours_snap_fills_to_xterm_entries() -> None:
    from rich._palettes import EIGHT_BIT_PALETTE

    theme.configure("dark", "#1e1e2e", color_system="256")
    entries = {EIGHT_BIT_PALETTE[i].hex for i in range(256)}
    assert {theme.TRACK, theme.SURFACE, theme.PANEL, theme.CURSOR} <= entries


@pytest.mark.parametrize("bg", DARK + LIGHT)
def test_256_colour_cursor_rows_stay_readable(bg: str) -> None:
    """Snapped one by one, the cursor family fell to grey / teal / navy: MUTED below 3.5:1 and done jobs (BLUE_DIM)
    at 1.6:1 on the cursor row, the blurred cursor louder than the focused one."""
    from rich._palettes import EIGHT_BIT_PALETTE
    from rich.color import Color, ColorSystem

    def drawn(color: str) -> str:                                  # what a 256-colour terminal shows for a hex
        return Color.parse(color).downgrade(ColorSystem.EIGHT_BIT).get_truecolor().hex

    theme.configure(theme.resolve("auto", env={}, background=bg).mode, bg, color_system="256")
    entries = {EIGHT_BIT_PALETTE[i].hex for i in range(16, 256)}  # never the terminal's own first 16
    family = (theme.CURSOR, theme.CURSOR_BLUR, theme.HOVER)
    assert set(family) <= entries and theme.CURSOR not in (theme.CURSOR_BLUR, theme.HOVER)
    for fill in family:                       # grey, or the accent's blue side: never a pink / cyan / yellow tint
        r, g, b = theme.parse_hex(fill) or (0, 0, 0)
        assert max(r, g, b) - min(r, g, b) <= 16 or (b > r and b >= g), fill
    for fill in family:
        assert theme.contrast(drawn(theme.MUTED), fill) >= 3.5, fill
    assert theme.contrast(drawn(theme.BLUE_DIM), theme.CURSOR) >= 2
    loud = [theme.contrast(fill, bg) for fill in family]          # against the background the terminal draws
    # near white a 256-colour terminal has few greys: the blurred cursor and hover may share one, never louder
    assert loud[0] > loud[1] >= loud[2] > 1.0 and loud[0] >= 1.2


def test_mid_grey_backgrounds_take_the_ink_that_reads_better() -> None:
    assert theme.resolve("auto", env={}, background="#999999").mode == "light"    # luminance 0.32
    assert theme.resolve("auto", env={}, background="#a0a0a0").mode == "light"
    assert theme.resolve("auto", env={}, background="#666666").mode == "dark"
    theme.configure("light", "#999999")
    assert theme.FOREGROUND != "#000000" and theme.contrast(theme.FOREGROUND, "#999999") >= 4.5
    assert theme.SURFACE != theme.mix("#ffffff", "#000000", 0.05)   # the reported grey is the base, not white


def test_ansi_mode_uses_only_the_terminal_palette() -> None:
    theme.configure("ansi")
    assert theme.ANSI and theme.TEXTUAL_THEME.ansi and theme.track() == ""
    assert theme.state_style_group(theme.StateGroup.ACTIVE) == "bold green"
    assert theme.host_color(None, 0) == "blue"
    assert all(not v.startswith("#") for v in theme.TEXTUAL_THEME.variables.values())


def test_state_styles_follow_configure() -> None:
    theme.configure("light")
    assert theme.GREEN in theme.state_style_group(theme.StateGroup.ACTIVE) and theme.GREEN == "#008700"
    assert "dim" not in theme.state_style_group(theme.StateGroup.CANCELLED)  # faint is drawn half-transparent


def test_clickable_bars_keep_their_own_colours() -> None:
    for widget in (FleetHeader(), HintBar(), ViewTabs()):
        assert widget.link_style.color is None and widget.link_style_hover.color is None
    assert ClickTable().cursor_foreground_priority == "renderable"


@pytest.mark.parametrize(("reply", "expected"), [
    (b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\\x1b[?62;22c", "#1e1e2e"),
    (b"\x1b]11;rgb:ff/00/80\x07", "#ff0080"),
    (b"\x1b]11;rgba:0000/2b2b/3636/ffff\x1b\\", "#002b36"),
    (b"\x1b[?1;2c", None),
])
def test_parse_reply(reply: bytes, expected: str | None) -> None:
    assert termbg.parse_reply(reply) == expected


class Terminal:
    """The far end of a pty: answers the query (after `delay`) with `answer`, or never."""

    def __init__(self, answer: bytes | None, delay: float = 0.0) -> None:
        self.master, self.slave = os.openpty()
        self.path = os.ttyname(self.slave)
        self.answer, self.delay = answer, delay
        self.thread = threading.Thread(target=self._respond, daemon=True)
        self.thread.start()

    def _respond(self) -> None:
        seen = b""
        while b"\x1b[c" not in seen:
            try:
                seen += os.read(self.master, 1024)
            except OSError:
                return
        time.sleep(self.delay)
        if self.answer is not None:
            os.write(self.master, self.answer)

    def close(self) -> None:
        self.thread.join(2)
        os.close(self.slave)
        os.close(self.master)


_CHILD = """
import os, sys
from runfleet.ui import termbg
path, timeout = sys.argv[1], float(sys.argv[2])
os.close(os.open(path, os.O_RDWR))      # a new session's first tty open makes it the controlling tty (foreground)
fd = os.open(path, os.O_RDWR)
print(termbg.query_background(timeout, tty=path) or "-")
"""


def _query(term: Terminal, timeout: float) -> tuple[str | None, float]:
    """Query from a child whose controlling tty (and foreground group) is the pty."""
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", _CHILD, term.path, str(timeout)], capture_output=True, text=True,
                          start_new_session=True, timeout=30, check=True)
    out = done.stdout.strip()
    return (None if out == "-" else out), time.monotonic() - started


@pytest.mark.posix                # a pty and termios
@pytest.mark.parametrize(("answer", "expected"), [
    (b"\x1b]11;rgb:2828/2a2a/3636\x1b\\\x1b[?62c", "#282a36"),
    (b"\x1b]11;rgb:2828/2a2a/3636\x07\x1b[?62c", "#282a36"),     # BEL-terminated (Terminal.app)
    (b"\x1b[?62c", None),                                          # OSC 11 ignored
])
def test_query_over_a_pty(answer: bytes, expected: str | None) -> None:
    term = Terminal(answer)
    try:
        result, took = _query(term, 5.0)
        assert result == expected and took < 4.0                   # the DA reply ends the wait, not the 5 s timeout
    finally:
        term.close()


@pytest.mark.posix                # a pty and termios
def test_query_without_any_answer_gives_up() -> None:
    term = Terminal(None)
    try:
        assert _query(term, 0.2)[0] is None
    finally:
        term.close()


def test_typeahead_is_what_remains_after_the_replies() -> None:
    data = b"2\x1b]11;rgb:0000/0000/0000\x1b\\j\x1b[?62;22cq"
    assert termbg.typeahead(data) == b"2jq" and termbg.parse_reply(data) == "#000000"


@pytest.mark.parametrize("data", [
    b"\x1b]11;#1e1e2e\x07",                       # a payload the colour regex does not know
    b"\x1b]11;rgb:1e1e/",                          # cut short (the drain path)
    b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b",             # cut inside the ST
    b"\x1b]10;rgb:ffff/ffff/ffff\x07",             # another OSC
    b"\x1b[?62;2",                                 # a DA reply cut short
])
def test_typeahead_strips_every_reply_whole_or_partial(data: bytes) -> None:
    assert termbg.typeahead(b"j" + data) == b"j"


def test_a_reply_without_its_terminator_gives_no_colour() -> None:
    assert termbg.parse_reply(b"\x1b]11;rgb:1e1e/1e1e/2e2e") is None
    assert termbg.parse_reply(b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b") is None
    assert termbg.parse_reply(b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\") == "#1e1e2e"


class ScriptedTerminal(Terminal):
    """The far end of a pty: once asked, writes each chunk after its delay (a reply split across packets, or late)."""

    def __init__(self, chunks: list[tuple[float, bytes]]) -> None:
        self.chunks = chunks
        super().__init__(None)

    def _respond(self) -> None:
        seen = b""
        while b"\x1b[c" not in seen:
            try:
                seen += os.read(self.master, 1024)
            except OSError:
                return
        for delay, chunk in self.chunks:
            time.sleep(delay)
            os.write(self.master, chunk)


_CHILD_REPORT = """
import json, os, sys, termios, time
from runfleet.ui import termbg
path, timeout = sys.argv[1], float(sys.argv[2])
os.close(os.open(path, os.O_RDWR))
fd = os.open(path, os.O_RDWR)
pushed = []
termbg._push_back = lambda _fd, keys: pushed.append(keys)     # what would go back to the tty as keys
before = termios.tcgetattr(fd)
colour = termbg.query_background(timeout, tty=path)
restored = termios.tcgetattr(fd) == before
time.sleep(0.3)
raw = termios.tcgetattr(fd)
raw[3] &= ~(termios.ICANON | termios.ECHO)
raw[6][termios.VMIN], raw[6][termios.VTIME] = 0, 0
termios.tcsetattr(fd, termios.TCSANOW, raw)
left = os.read(fd, 4096)
termios.tcsetattr(fd, termios.TCSANOW, before)
print(json.dumps({"colour": colour, "pushed": [k.decode("latin-1") for k in pushed],
                  "left": left.decode("latin-1"), "restored": restored}))
"""


def _query_report(term: Terminal, timeout: float) -> dict[str, object]:
    import json

    done = subprocess.run([sys.executable, "-c", _CHILD_REPORT, term.path, str(timeout)], capture_output=True,
                          text=True, start_new_session=True, timeout=30, check=True)
    return dict(json.loads(done.stdout))


@pytest.mark.posix                # a pty and termios
@pytest.mark.parametrize(("chunks", "expected", "pushed"), [
    # split across packets, inside the payload and inside the ST; the user typed "j" while waiting
    ([(0.0, b"j\x1b]11;rgb:2828/2a"), (0.05, b"2a/3636\x1b"), (0.05, b"\\\x1b[?62c")], "#282a36", ["j"]),
    ([(0.0, b"\x1b]11;rgb:2828/2a2a/3636\x07\x1b[?6"), (0.05, b"2c")], "#282a36", [""]),
])
def test_query_reassembles_a_split_reply(chunks: list[tuple[float, bytes]], expected: str,
                                         pushed: list[str]) -> None:
    term = ScriptedTerminal(chunks)
    try:
        report = _query_report(term, 2.0)
        assert report["colour"] == expected and report["restored"]
        assert report["pushed"] == pushed and report["left"] == ""
    finally:
        term.close()


@pytest.mark.posix                # a pty and termios
@pytest.mark.parametrize(("chunks", "expected"), [
    ([(0.4, b"\x1b]11;rgb:2828/2a2a/3636\x1b\\\x1b[?62c")], "#282a36"),     # late but whole: the colour counts
    ([(0.4, b"\x1b]11;rgb:2828/2a2a/3636")], None),                         # late and never terminated
    ([(0.4, b"\x1b]11;rgb:2828/")], None),                                   # late and cut short
    ([(0.4, b"\x1b]11;#282a36\x07")], None),                                # late in a form we cannot read
])
def test_a_reply_after_the_timeout_never_comes_back_as_keys(chunks: list[tuple[float, bytes]],
                                                             expected: str | None) -> None:
    term = ScriptedTerminal(chunks)
    try:
        report = _query_report(term, 0.2)                          # the reply lands in the drain after the timeout
        assert report["colour"] == expected and report["restored"]
        assert report["pushed"] == [] and report["left"] == ""
    finally:
        term.close()


_CHILD_BACKGROUND = """
import os, sys
from runfleet.ui import termbg
path = sys.argv[1]
os.close(os.open(path, os.O_RDWR))      # the controlling tty of this session; we are its foreground group …
pid = os.fork()
if pid == 0:
    os.setpgid(0, 0)                    # … this child is not (`runfleet doctor &`)
    before = len(os.listdir("/proc/self/fd"))
    results = {termbg.query_background(0.1, tty=path) for _ in range(20)}
    print(before, len(os.listdir("/proc/self/fd")), results, flush=True)
    os._exit(0)
os.waitpid(pid, 0)
"""


@pytest.mark.posix                # a pty, process groups, termios
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="counts descriptors in /proc")
def test_query_from_a_background_job_closes_the_tty() -> None:
    master, slave = os.openpty()
    try:
        done = subprocess.run([sys.executable, "-c", _CHILD_BACKGROUND, os.ttyname(slave)], capture_output=True,
                              text=True, start_new_session=True, timeout=30, check=True)
        before, after, results = done.stdout.split(" ", 2)
        assert results.strip() == "{None}" and int(after) == int(before)
    finally:
        os.close(slave)
        os.close(master)


async def test_live_switch_restyles_and_is_remembered(clock: object, tmp_path: object) -> None:
    from runfleet.ui.app import RunFleetApp
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        resolved = app.set_theme_choice("ansi")
        await pilot.pause()
        assert resolved.mode == "ansi" and app.has_class("-plain-cursor") and app.theme == "runfleet-ansi"
        assert app.ui_state["theme"] == "ansi"
        app.set_theme_choice("light")
        await pilot.pause()
        assert not app.has_class("-plain-cursor") and app.theme.startswith("runfleet-light")


async def test_theme_choice_survives_a_restart(clock: object, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_state=True: set_theme_choice writes ui.json, and saved_theme_choice() (what run_tui and doctor read at
    the next start) returns it."""
    import json

    import runfleet.ui.app as app_module
    from runfleet.ui.app import RunFleetApp, saved_theme_choice
    from tests.ui.fakes import FakeService

    state = tmp_path / "ui.json"
    monkeypatch.setattr(app_module, "ui_state_path", lambda: state)
    assert saved_theme_choice() == "auto"
    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=True)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        app.set_theme_choice("light")
        await pilot.pause()
        assert saved_theme_choice() == "light"                    # written at once, not only when the app quits
    assert json.loads(state.read_text(encoding="utf-8"))["theme"] == "light" and saved_theme_choice() == "light"
    again = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=True)  # type: ignore[arg-type]
    assert again.ui_state["theme"] == "light"
    state.write_text('{"theme": "neon"}', encoding="utf-8")                    # a hand-edited file
    assert saved_theme_choice() == "auto"


async def test_live_switch_recolours_cached_table_cells(clock: object) -> None:
    """Rich's Text == ignores the base style: without forget_rendered the Jobs table kept its old colours."""
    from runfleet.ui.app import RunFleetApp
    from runfleet.ui.widgets.job_table import JobTable
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause(0.3)
        table = app.screen.query_one(JobTable)
        key = table.order[0]
        before = str(table.get_cell(key, "host").style)
        app.set_theme_choice("light")
        await pilot.pause(0.3)
        after = str(table.get_cell(key, "host").style)
        assert before in theme.HOST_PALETTE_DARK and after in theme.HOST_PALETTE_LIGHT


@pytest.mark.parametrize("switch", ["theme", "fleet_layout"])
async def test_switch_skips_the_detail_screen_of_a_removed_host(clock: object, switch: str) -> None:
    """Settings removed k2 while HostDetailScreen(k2) is under it: a theme / layout switch must not refresh it (its
    host has no state any more — KeyError)."""
    from runfleet.ui.app import RunFleetApp
    from runfleet.ui.screens.host_detail import HostDetailScreen
    from runfleet.ui.screens.settings import SettingsScreen
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        await app.push_screen(HostDetailScreen("k2"))
        await app.push_screen(SettingsScreen("hosts"))
        await pilot.pause()
        await svc.remove_host("k2")
        await pilot.pause(0.3)
        assert [type(s) for s in app.screen_stack][-2:] == [HostDetailScreen, SettingsScreen]
        if switch == "theme":
            app.screen.open_about("theme")                         # type: ignore[attr-defined]
        else:
            app.set_fleet_layout("compact")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


def _colours(strips: object) -> set[str]:
    return {seg.style.color.get_truecolor().hex for strip in strips for seg in strip  # type: ignore[attr-defined]
            if seg.style is not None and seg.style.color is not None and not seg.style.color.is_default}


@pytest.mark.parametrize("follow", [True, False])
async def test_live_switch_recolours_the_log_and_keeps_its_place(clock: object, follow: bool) -> None:
    """The RichLog stores rendered strips: LogView.forget_rendered draws the kept lines and the status again in the
    new palette, at the same scroll position and follow state."""
    from runfleet.core.service import LogUpdate
    from runfleet.ui.app import RunFleetApp
    from runfleet.ui.screens.log_viewer import LogScreen
    from runfleet.ui.widgets.log_view import LogView
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause(0.3)
        await pilot.press("l")
        await pilot.pause(0.3)
        assert isinstance(app.screen, LogScreen)
        view = app.screen.query_one(LogView)
        lines = tuple(f"step {i}" for i in range(60)) + ("RuntimeError: CUDA out of memory",) + tuple(
            f"step {i}" for i in range(60, 120)) + ("UserWarning: deprecated thing",)
        svc.push_log(LogUpdate(lines=lines))
        await pilot.pause(0.3)
        if not follow:
            view.set_follow(False)
            view.richlog.scroll_to(y=40, animate=False, immediate=True)
            await pilot.pause()
        y, count = view.richlog.scroll_y, len(view.richlog.lines)
        dark = (theme.RED, theme.YELLOW, theme.GREEN)
        assert {theme.RED, theme.YELLOW} <= _colours(view.richlog.lines)
        app.set_theme_choice("light")
        await pilot.pause(0.3)
        shown = _colours(view.richlog.lines)
        assert {theme.RED, theme.YELLOW} <= shown and not shown & set(dark)
        assert len(view.richlog.lines) == count and view.follow is follow and view.richlog.scroll_y == y
        status = _colours(view.query_one("#live").render_lines(view.query_one("#live").region.reset_offset))
        assert (theme.GREEN if follow else theme.YELLOW) in status and not status & set(dark)


def test_ansi_pills_and_chips_draw_the_terminal_background_on_the_colour() -> None:
    """"default on yellow" is the terminal's foreground on its yellow (1.2:1 on VS Code dark): reverse video draws the
    terminal's own background on the colour."""
    from runfleet.ui import render

    theme.configure("ansi")
    assert theme.pill() == "bold reverse blue"
    assert str(render.cause_chip("OOM-GPU").style) == "bold reverse red"
    assert str(render.cause_chip("Priority", pending=True).style) == "bold reverse yellow"
    theme.configure("dark")
    assert theme.pill() == f"bold {theme.SURFACE} on {theme.ACCENT}" and "reverse" not in theme.pill()
    assert str(render.cause_chip("OOM-GPU").style) == f"bold {theme.SURFACE} on {theme.RED}"


async def test_ansi_focused_detail_tab_is_reverse_video(clock: object) -> None:
    from textual.widgets import Tab, Tabs

    from runfleet.ui.app import RunFleetApp
    from runfleet.ui.screens.job_detail import JobDetailScreen
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        app.set_theme_choice("ansi")
        await pilot.press("tab")
        await pilot.pause(0.3)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen)
        tabs = app.screen.query_one(Tabs)
        tabs.focus()
        await pilot.pause()
        active = tabs.query_one("Tab.-active", Tab)
        style = active.rich_style
        assert style.reverse and style.bold and style.color is not None and style.color.number == 4   # ansi blue
        app.set_theme_choice("dark")
        await pilot.pause()
        assert not active.rich_style.reverse and active.rich_style.bgcolor is not None


async def test_dialog_menus_highlight_on_the_cursor_tint(clock: object) -> None:
    """The highlighted option of a dialog menu sat on the accent, under prompts in their own colours (1.0-1.6:1)."""
    from textual.widgets import OptionList

    from runfleet.ui.app import RunFleetApp
    from runfleet.ui.screens.action_menu import ActionMenuScreen
    from tests.ui.fakes import FakeService

    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=svc.clock, animations=False, persist_state=False)  # type: ignore[arg-type]
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause(0.3)
        await pilot.press("x")
        await pilot.pause(0.3)
        assert isinstance(app.screen, ActionMenuScreen)
        menu = app.screen.query_one(OptionList)
        style = menu.get_component_rich_style("option-list--option-highlighted")
        assert style.bgcolor is not None and style.bgcolor.get_truecolor().hex == theme.CURSOR and not style.reverse
        assert theme.contrast(theme.DIM, theme.CURSOR) > 2 and theme.contrast(theme.ORANGE, theme.CURSOR) > 3
        app.set_theme_choice("ansi")
        await pilot.pause()
        assert menu.get_component_rich_style("option-list--option-highlighted").reverse
