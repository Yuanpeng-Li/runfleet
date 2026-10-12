"""[WP3] Colours, thresholds and NO_COLOR handling (§7.11).

Every style the UI uses comes from here, as a Rich style string. Under ``NO_COLOR`` the colour parts are dropped
(Textual strips them anyway) and replaced by attributes (bold / reverse / dim) so states stay distinguishable.
"""

from __future__ import annotations

import os
from typing import Literal

from textual.theme import Theme

from runfleet.config import HostConfig
from runfleet.models import ConnState, JobState, StateGroup

Level = Literal["low", "mid", "high"]

DEFAULT_UTIL_THRESHOLDS: tuple[float, float] = (10, 75)
DEFAULT_MEM_THRESHOLDS: tuple[float, float] = (10, 80)

# Host colours, cycled in config order unless `color` is set (§8.3).
HOST_PALETTE: tuple[str, ...] = (
    "#5fafff", "#d787ff", "#5fd7af", "#ffaf5f", "#ff87af", "#87d7ff", "#d7d75f", "#af87ff", "#87ffaf",
)
# The same hues darkened for a light background (contrast >= 4.5:1 on white; the pastels above reach ~1.8:1).
HOST_PALETTE_LIGHT: tuple[str, ...] = (
    "#0060c0", "#8a2be2", "#007a5e", "#a85400", "#c2185b", "#00739e", "#6b6200", "#6a3dc8", "#00804a",
)

def light_background(env: dict[str, str] | None = None) -> bool:
    """The terminal has a light background: ``RUNFLEET_THEME=light|dark`` wins, else ``COLORFGBG`` ("fg;bg", set by
    rxvt, Konsole, iTerm2 and others: bg 7 or 9-15 is light). Unknown -> dark (the common case)."""
    e = os.environ if env is None else env
    forced = e.get("RUNFLEET_THEME", "").strip().lower()
    if forced in ("light", "dark"):
        return forced == "light"
    bg = e.get("COLORFGBG", "").split(";")[-1].strip()
    return bg.isdigit() and (int(bg) == 7 or 9 <= int(bg) <= 15)


LIGHT = light_background()

# Colour constants for the detected background: the dark palette's light greys and yellow are near-invisible on a
# white terminal profile (macOS Terminal default), so a light background gets darker variants.
GREEN = "#008700" if LIGHT else "#5fd75f"
YELLOW = "#875f00" if LIGHT else "#ffd75f"
RED = "#d70000" if LIGHT else "#ff5f5f"
ORANGE = "#d75f00" if LIGHT else "#ff8700"
MAGENTA = "#af00af" if LIGHT else "#d75fd7"
BLUE_DIM = "#005f87" if LIGHT else "#5f87af"
GREY = "#626262" if LIGHT else "#8a8a8a"
DIM = "#8a8a8a" if LIGHT else "#6c6c6c"
ACCENT = "#005fd7" if LIGHT else "#5fafff"
MUTED = "#4e4e4e" if LIGHT else "#a8a8a8"
TRACK = "#e4e4e4" if LIGHT else "#262626"
SURFACE = "#f2f2f2" if LIGHT else "#1c1c1c"    # dialog panel background (= TEXTUAL_THEME.surface)
RULE = "#c6c6c6" if LIGHT else "#3a3a3a"       # thin separators inside dialogs


# Textual theme: nvitop-ish blue accent on a neutral base. Widget backgrounds use ``ansi_default`` in the stylesheet,
# so the terminal's own background shows through (§7.11 "transparent default background"); body text uses the
# terminal's default foreground (tcss ``color: ansi_default``), so it is readable on light and dark profiles.
TEXTUAL_THEME = Theme(
    name="runfleet",
    primary="#5fafff",
    secondary="#87afd7",
    accent="#5fafff",
    warning=YELLOW,
    error=RED,
    success=GREEN,
    foreground="#d0d0d0",
    background="#121212",
    surface="#1c1c1c",
    panel="#262626",
    boost="#5fafff14",
    dark=True,
) if not LIGHT else Theme(
    name="runfleet",
    primary=ACCENT,
    secondary="#5f87af",
    accent=ACCENT,
    warning=YELLOW,
    error=RED,
    success=GREEN,
    foreground="#262626",
    background="#ffffff",
    surface="#f2f2f2",
    panel="#e4e4e4",
    boost="#005fd714",
    dark=False,
)


def no_color() -> bool:
    """True when the user asked for no colour (https://no-color.org: any non-empty value)."""
    return bool(os.environ.get("NO_COLOR"))


def _pick(color: str, mono: str) -> str:
    return mono if no_color() else color


# ---------------------------------------------------------------- thresholds
def level(value: float | None, thresholds: tuple[float, float]) -> Level:
    """nvitop levels: < low -> "low", < high -> "mid", else "high"."""
    if value is None or value < thresholds[0]:
        return "low"
    return "mid" if value < thresholds[1] else "high"


def level_style(value: float | None, thresholds: tuple[float, float]) -> str:
    lv = level(value, thresholds)
    if lv == "low":
        return _pick(GREEN, "")
    if lv == "mid":
        return _pick(YELLOW, "bold")
    return _pick(RED, "bold")      # never reverse: a reverse-video █ is drawn in the background colour (looks empty)


def temp_style(temp_c: float | None) -> str:
    if temp_c is None or temp_c < 60:
        return ""
    return _pick(YELLOW if temp_c < 80 else RED, "bold")


# ---------------------------------------------------------------- job states
_GROUP_STYLE: dict[StateGroup, tuple[str, str]] = {
    StateGroup.ACTIVE: (f"bold {GREEN}", "bold"),
    StateGroup.QUEUED: (YELLOW, ""),
    StateGroup.DONE: (BLUE_DIM, "dim"),
    StateGroup.FAILED: (f"bold {RED}", "reverse"),
    StateGroup.CANCELLED: (f"{MAGENTA} dim", "dim italic"),
    StateGroup.ENDED: (GREY, "dim"),
    StateGroup.UNKNOWN: (f"{GREY} italic", "italic"),
}


def state_style(state: JobState) -> str:
    if state is JobState.PREEMPTED:
        return _pick(f"bold {ORANGE}", "reverse")
    color, mono = _GROUP_STYLE[state.group]
    return _pick(color, mono)


def state_style_group(group: StateGroup) -> str:
    color, mono = _GROUP_STYLE[group]
    return _pick(color, mono)


def hint_style(red: bool) -> str:
    """Style of a hint line; red hints are the "will never run" / exit-scan error kind."""
    return _pick(RED, "bold") if red else _pick(YELLOW, "")


# ---------------------------------------------------------------- connection states
CONN_GLYPH: dict[ConnState, str] = {
    ConnState.NEW: "◌",
    ConnState.CONNECTING: "◌",
    ConnState.OK: "●",
    ConnState.DEGRADED: "●",
    ConnState.RETRYING: "◐",
    ConnState.UNREACHABLE: "✖",
    ConnState.NEEDS_AUTH: "✖",
    ConnState.HOSTKEY: "✖",
    ConnState.DISABLED: "○",
}

CONN_LABEL: dict[ConnState, str] = {
    ConnState.NEW: "new",
    ConnState.CONNECTING: "connecting",
    ConnState.OK: "ok",
    ConnState.DEGRADED: "degraded",
    ConnState.RETRYING: "retrying",
    ConnState.UNREACHABLE: "unreachable",
    ConnState.NEEDS_AUTH: "needs auth",
    ConnState.HOSTKEY: "host key",
    ConnState.DISABLED: "disabled",
}


def conn_style(state: ConnState) -> str:
    if state is ConnState.OK:
        return _pick(GREEN, "bold")
    if state in (ConnState.DEGRADED, ConnState.RETRYING):
        return _pick(YELLOW, "bold")
    if state is ConnState.UNREACHABLE:
        return _pick(f"bold {RED}", "bold reverse")
    if state in (ConnState.NEEDS_AUTH, ConnState.HOSTKEY):
        return _pick(f"bold {MAGENTA}", "bold reverse")
    if state is ConnState.DISABLED:
        return _pick(DIM, "dim")
    return _pick(MUTED, "italic")   # NEW / CONNECTING


def conn_style_label(label: str) -> str:
    """Style of a short host-problem label in a job row (``auth``, ``retry``, ``down``)."""
    if label in ("auth", "hostkey"):
        return conn_style(ConnState.NEEDS_AUTH)
    if label in ("down", "dns"):
        return conn_style(ConnState.UNREACHABLE)
    return conn_style(ConnState.RETRYING)


def is_problem(state: ConnState) -> bool:
    """States that get the red/magenta treatment (chip click opens HostErrorScreen)."""
    return state in (ConnState.UNREACHABLE, ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.RETRYING)


# ---------------------------------------------------------------- hosts and misc
def host_color(cfg: HostConfig | None, index: int) -> str:
    """Host colour: config ``color`` wins, else the palette by config position."""
    if no_color():
        return "bold"
    if cfg is not None and cfg.color:
        return cfg.color
    palette = HOST_PALETTE_LIGHT if LIGHT else HOST_PALETTE
    return palette[index % len(palette)]


def track() -> str:
    """Background of the empty part of a bar (a faint track, so 0 % still reads as a bar)."""
    return "" if no_color() else f"on {TRACK}"


def dim() -> str:
    return _pick(DIM, "dim")


def muted() -> str:
    return _pick(MUTED, "")


def accent() -> str:
    return _pick(f"bold {ACCENT}", "bold")


def label() -> str:
    """Field labels in detail boxes."""
    return _pick(GREY, "dim")


def error() -> str:
    return _pick(RED, "bold")


def warn() -> str:
    return _pick(YELLOW, "bold")


def ok() -> str:
    return _pick(GREEN, "")


def pill() -> str:
    """Active tab / selected row / primary button in dialogs: panel colour on the accent (herdr's settings look)."""
    return _pick(f"bold {SURFACE} on {ACCENT}", "bold reverse")


def pill_quiet() -> str:
    """Secondary button (`esc close`)."""
    return _pick(f"bold {MUTED} on {RULE}", "bold")


def rule() -> str:
    return _pick(RULE, "dim")
