"""[WP3] Colours, thresholds and NO_COLOR handling (§7.11).

Every style the UI uses comes from here, as a Rich style string; the stylesheet reads the background-derived colours
as ``$rf-*`` theme variables. Under ``NO_COLOR`` the colour parts are dropped (Textual strips them anyway) and replaced
by attributes (bold / reverse / dim) so states stay distinguishable.

The palette is set by :func:`configure` (``cli.run_tui`` calls it once before the app starts; at import it follows
the environment only): a **mode** — ``dark``, ``light`` or ``ansi`` (only the terminal's 16 colours, so the terminal's
colour scheme decides how everything looks) — and the terminal's **background** when known (``termbg``, OSC 11).
Hues are fixed per mode; every neutral (greys for text, fills, frames) and every tint (cursor row, hover, problem
frame) is mixed from the real background, so panels and frames sit in the terminal's own hue instead of being grey
boxes on a tinted background. Textual blends translucent colours against black (it cannot know the background), so
the stylesheet uses these solid colours instead of ``$x N%``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from textual.theme import Theme

if TYPE_CHECKING:
    from rich.style import Style as RichStyle

from runfleet.config import HostConfig
from runfleet.models import ConnState, JobState, StateGroup

Level = Literal["low", "mid", "high"]
Mode = Literal["dark", "light", "ansi"]
CHOICES: tuple[str, ...] = ("auto", "dark", "light", "ansi")

DEFAULT_UTIL_THRESHOLDS: tuple[float, float] = (10, 75)
DEFAULT_MEM_THRESHOLDS: tuple[float, float] = (10, 80)

# Host colours, cycled in config order unless `color` is set (§8.3).
HOST_PALETTE_DARK: tuple[str, ...] = (
    "#5fafff", "#d787ff", "#5fd7af", "#ffaf5f", "#ff87af", "#87d7ff", "#d7d75f", "#af87ff", "#87ffaf",
)
# The same hues darkened for a light background (contrast >= 4.5:1 on white; the pastels above reach ~1.8:1).
HOST_PALETTE_LIGHT: tuple[str, ...] = (
    "#0060c0", "#8a2be2", "#007a5e", "#a85400", "#c2185b", "#00739e", "#6b6200", "#6a3dc8", "#00804a",
)
HOST_PALETTE_ANSI: tuple[str, ...] = (
    "blue", "magenta", "cyan", "yellow", "green", "bright_blue", "bright_magenta", "bright_cyan", "bright_green",
)
HOST_PALETTE = HOST_PALETTE_DARK


# ---------------------------------------------------------------- colour arithmetic
def parse_hex(value: str) -> tuple[int, int, int] | None:
    v = value.strip().lstrip("#")
    if len(v) != 6:
        return None
    try:
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except ValueError:
        return None


def mix(base: str, other: str, t: float) -> str:
    """`base` moved a fraction `t` towards `other` (both #rrggbb)."""
    a, b = parse_hex(base) or (0, 0, 0), parse_hex(other) or (0, 0, 0)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def luminance(color: str) -> float:
    """WCAG relative luminance of #rrggbb (0 black … 1 white)."""
    def lin(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
    r, g, b = parse_hex(color) or (0, 0, 0)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def is_light(background: str) -> bool:
    """A light background: black ink reads better on it than white (a mid grey like #999999 is light — judged by
    luminance > 0.4 it got the dark palette, its hues at 1.0-2.4:1)."""
    return contrast(background, "#000000") > contrast(background, "#ffffff")


# ---------------------------------------------------------------- mode
def light_background(env: dict[str, str] | None = None) -> bool:
    """The terminal has a light background: ``RUNFLEET_THEME=light|dark`` wins, else ``COLORFGBG`` ("fg;bg", set by
    rxvt, Konsole, iTerm2 and others: bg 7 or 9-15 is light). Unknown -> dark (the common case)."""
    e = os.environ if env is None else env
    forced = e.get("RUNFLEET_THEME", "").strip().lower()
    if forced in ("light", "dark"):
        return forced == "light"
    return _colorfgbg_light(e) is True


def _colorfgbg_light(env: Mapping[str, str]) -> bool | None:
    bg = env.get("COLORFGBG", "").split(";")[-1].strip()
    if not bg.isdigit():
        return None
    return int(bg) == 7 or 9 <= int(bg) <= 15


@dataclass(frozen=True, slots=True)
class Resolved:
    mode: Mode
    background: str | None       # #rrggbb the terminal reported (OSC 11), else None
    source: str                  # why: "RUNFLEET_THEME", "setting", "terminal background", "COLORFGBG", "16 colours",
                                 # "default"


def resolve(choice: str = "auto", *, env: dict[str, str] | None = None, background: str | None = None,
            color_system: str | None = None) -> Resolved:
    """Mode by precedence: ``RUNFLEET_THEME`` (dark|light|ansi), then `choice` (the UI setting) unless "auto", then
    a 16-colour terminal (ansi: hex colours would land on arbitrary palette slots), then the reported background
    (:func:`is_light`), then COLORFGBG, else dark."""
    e = os.environ if env is None else env
    bg = background if background and parse_hex(background) else None
    forced = e.get("RUNFLEET_THEME", "").strip().lower()
    if forced in ("dark", "light", "ansi"):
        return Resolved(forced, bg, "RUNFLEET_THEME")             # type: ignore[arg-type]
    if choice in ("dark", "light", "ansi"):
        return Resolved(choice, bg, "setting")                    # type: ignore[arg-type]
    if color_system == "standard":
        return Resolved("ansi", bg, "16 colours")
    if bg is not None:
        return Resolved("light" if is_light(bg) else "dark", bg, "terminal background")
    light = _colorfgbg_light(e)
    if light is not None:
        return Resolved("light" if light else "dark", None, "COLORFGBG")
    return Resolved("dark", None, "default")


# Textual's ansi-dark theme defines these itself (its ANSI widget CSS needs them, $ansi-background above all)
_TEXTUAL_ANSI_DARK_VARIABLES = {
    "ansi-background": "ansi_black", "ansi-foreground": "ansi_white", "border-blurred": "ansi_black",
    "block-cursor-foreground": "ansi_black", "block-cursor-background": "ansi_white",
    "input-cursor-background": "ansi_black", "input-cursor-foreground": "ansi_bright_white",
    "input-cursor-text-style": "none", "input-selection-background": "ansi_bright_blue",
    "input-selection-foreground": "ansi_black", "screen-selection-background": "ansi_bright_blue",
    "screen-selection-foreground": "ansi_black",
}

# ---------------------------------------------------------------- the palette (module globals, set by configure)
MODE: Mode = "dark"
LIGHT = False
ANSI = False
PLAIN_CURSOR = False             # ansi or 16 colours: no tint to spare, cursor rows are reverse / underline
BACKGROUND: str | None = None
SOURCE = "default"
GREEN = YELLOW = RED = ORANGE = MAGENTA = CANCELLED = BLUE_DIM = ""
GREY = DIM = ACCENT = MUTED = FOREGROUND = ""
TRACK = SURFACE = RULE = PANEL = BORDER = BORDER_PROBLEM = CURSOR = CURSOR_BLUR = HOVER = ""
TEXTUAL_THEME: Theme


def configure(mode: Mode, background: str | None = None, *, color_system: str | None = None,
              source: str = "") -> None:
    """Set every colour for `mode` on `background` (None: black for dark, white for light). In 256-colour mode the
    mixed fills snap to xterm-256 entries the way the terminal will show them."""
    global MODE, LIGHT, ANSI, PLAIN_CURSOR, BACKGROUND, SOURCE, HOST_PALETTE, TEXTUAL_THEME
    global GREEN, YELLOW, RED, ORANGE, MAGENTA, CANCELLED, BLUE_DIM, GREY, DIM, ACCENT, MUTED, FOREGROUND
    global TRACK, SURFACE, RULE, PANEL, BORDER, BORDER_PROBLEM, CURSOR, CURSOR_BLUR, HOVER
    MODE, LIGHT, ANSI = mode, mode == "light", mode == "ansi"
    PLAIN_CURSOR = ANSI or color_system == "standard"
    BACKGROUND, SOURCE = (background if background and parse_hex(background) else None), source or SOURCE
    if ANSI:
        GREEN, YELLOW, RED, ORANGE, MAGENTA, CANCELLED = "green", "yellow", "red", "bright_red", "magenta", "magenta"
        BLUE_DIM, GREY, DIM, ACCENT, MUTED, FOREGROUND = "blue", "bright_black", "bright_black", "blue", "default", \
            "default"
        TRACK, SURFACE, RULE, PANEL = "", "default", "bright_black", "default"
        BORDER, BORDER_PROBLEM, CURSOR, CURSOR_BLUR, HOVER = "bright_black", "red", "blue", "bright_black", "default"
        HOST_PALETTE = HOST_PALETTE_ANSI
        TEXTUAL_THEME = Theme(
            name="runfleet-ansi", primary="ansi_blue", secondary="ansi_cyan", accent="ansi_blue", warning="ansi_yellow",
            error="ansi_red", success="ansi_green", foreground="ansi_default", background="ansi_default",
            surface="ansi_default", panel="ansi_default", boost="ansi_default", dark=True, ansi=True,
            variables={**_TEXTUAL_ANSI_DARK_VARIABLES, "rf-border": "ansi_bright_black", "rf-border-problem": "ansi_red", "rf-cursor": "ansi_default",
                       "rf-cursor-blur": "ansi_default", "rf-hover": "ansi_default", "rf-panel": "ansi_default",
                       "rf-surface": "ansi_default", "rf-muted": "ansi_default", "rf-rule": "ansi_bright_black",
                       "rf-foreground": "ansi_default", "rf-scrollbar": "ansi_bright_black"})
        return
    if LIGHT:
        GREEN, YELLOW, RED, ORANGE, MAGENTA = "#008700", "#875f00", "#d70000", "#d75f00", "#af00af"
        BLUE_DIM, ACCENT = "#005f87", "#005fd7"
        HOST_PALETTE = HOST_PALETTE_LIGHT
    else:
        GREEN, YELLOW, RED, ORANGE, MAGENTA = "#5fd75f", "#ffd75f", "#ff5f5f", "#ff8700", "#d75fd7"
        BLUE_DIM, ACCENT = "#5f87af", "#5fafff"
        HOST_PALETTE = HOST_PALETTE_DARK
    # a reported background of the other brightness (a forced mode) cannot be the base: mixing light-mode inks from a
    # dark background would give dark text on dark fills
    fits = BACKGROUND is not None and is_light(BACKGROUND) == LIGHT
    base = BACKGROUND if fits and BACKGROUND else ("#ffffff" if LIGHT else "#000000")
    ink = "#000000" if LIGHT else "#ffffff"

    def n(t: float) -> str:                 # a neutral: the background moved towards the ink
        return mix(base, ink, t)


    if LIGHT:
        DIM, GREY, MUTED, FOREGROUND = n(0.46), n(0.62), n(0.69), n(0.85)
        TRACK, SURFACE, RULE, PANEL, BORDER = n(0.11), n(0.05), n(0.22), n(0.10), n(0.30)
        CURSOR, CURSOR_BLUR, HOVER = mix(base, ACCENT, 0.15), mix(base, ACCENT, 0.09), mix(base, ACCENT, 0.05)
    else:
        DIM, GREY, MUTED, FOREGROUND = n(0.42), n(0.54), n(0.66), n(0.82)
        TRACK, SURFACE, RULE, PANEL, BORDER = n(0.15), n(0.11), n(0.23), n(0.15), n(0.28)
        CURSOR, CURSOR_BLUR, HOVER = mix(base, ACCENT, 0.24), mix(base, ACCENT, 0.14), mix(base, ACCENT, 0.08)
    CANCELLED = mix(base, MAGENTA, 0.82)    # was "magenta dim": faint text is drawn half-transparent (iTerm2)
    BORDER_PROBLEM = mix(base, RED, 0.65)
    if color_system == "256":                # draw the fills as the terminal will, each still distinct
        bg = _snap256(base, ink, set())
        SURFACE = _snap256(SURFACE, ink, {bg}, grey=True)
        TRACK = _snap256(TRACK, ink, {bg, SURFACE}, grey=True)
        PANEL = _snap256(PANEL, ink, {bg}, grey=True)
        # a reported background is what the terminal draws: only a guessed one is avoided as its snapped entry
        CURSOR, CURSOR_BLUR, HOVER = _cursor256((CURSOR, CURSOR_BLUR, HOVER), BACKGROUND or base,
                                                set() if BACKGROUND else {bg}, MUTED, darker=LIGHT)
    TEXTUAL_THEME = Theme(
        name=f"runfleet-{MODE}-{base[1:]}", primary=ACCENT, secondary=BLUE_DIM, accent=ACCENT, warning=YELLOW, error=RED,
        success=GREEN, foreground=FOREGROUND, background=base, surface=SURFACE, panel=PANEL, dark=not LIGHT,
        variables={"rf-border": BORDER, "rf-border-problem": BORDER_PROBLEM, "rf-cursor": CURSOR,
                   "rf-cursor-blur": CURSOR_BLUR, "rf-hover": HOVER, "rf-panel": PANEL, "rf-surface": SURFACE,
                   "rf-muted": MUTED, "rf-rule": RULE, "rf-foreground": FOREGROUND,
                   "rf-scrollbar": mix(PANEL, ACCENT, 0.45), "boost": HOVER},
    )


def _snap256(color: str, ink: str, avoid: set[str], *, grey: bool = False) -> str:
    """The xterm-256 entry the terminal draws for `color` (the grey ramp for neutrals: the colour cube would turn a
    near-white into pink), stepped towards `ink` along the grey ramp while it equals one of `avoid` (the background
    or the previous fill: on 256 colours a light tint would otherwise vanish into the background)."""
    from rich._palettes import EIGHT_BIT_PALETTE
    from rich.color import Color, ColorSystem

    ramp = sorted({"#000000", "#ffffff", *(EIGHT_BIT_PALETTE[i].hex for i in range(232, 256))}, key=luminance)
    if grey:
        snapped = min(ramp, key=lambda g: abs(luminance(g) - luminance(color)))
    else:
        snapped = Color.parse(color).downgrade(ColorSystem.EIGHT_BIT).get_truecolor().hex
    if snapped not in avoid:
        return snapped
    i = min(range(len(ramp)), key=lambda k: abs(luminance(ramp[k]) - luminance(snapped)))
    step = 1 if luminance(ink) > luminance(snapped) else -1
    while 0 <= i < len(ramp) and ramp[i] in avoid:
        i += step
    return ramp[max(0, min(len(ramp) - 1, i))]


def _drawn256(color: str) -> str:
    """The xterm-256 entry Rich draws for #rrggbb on a 256-colour terminal."""
    from rich.color import Color, ColorSystem

    return Color.parse(color).downgrade(ColorSystem.EIGHT_BIT).get_truecolor().hex


def _distance(a: str, b: str) -> float:
    """"Redmean" colour distance (what Rich uses to match a palette): close to how different two colours look."""
    (r1, g1, b1), (r2, g2, b2) = parse_hex(a) or (0, 0, 0), parse_hex(b) or (0, 0, 0)
    rm = (r1 + r2) / 2
    return (2 + rm / 256) * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + (2 + (255 - rm) / 256) * (b1 - b2) ** 2


CURSOR256_VISIBLE = (1.2, 1.1, 1.02)     # least contrast against the background: cursor, blurred cursor, hover
CURSOR256_TEXT = 3.5                     # least contrast of MUTED text on any of them (as on truecolor)


def _cursor256(targets: tuple[str, str, str], background: str, avoid: set[str], muted: str, *,
               darker: bool) -> tuple[str, str, str]:
    """CURSOR, CURSOR_BLUR, HOVER on a 256-colour terminal: for each, the xterm-256 entry (16-255: the first 16 are
    the terminal's own) nearest to its truecolor tint such that MUTED text stays readable on it, it shows against the
    background the terminal really draws (the reported one, not a snapped base), it lies on the ink's side of the
    background, and the family keeps its order (cursor louder than the blurred cursor, louder than hover). Snapping
    each tint on its own sent them to grey / teal / navy, the blurred cursor louder than the focused one and MUTED
    below 3:1 on the cursor row."""
    from rich._palettes import EIGHT_BIT_PALETTE

    # what the terminal shows for each entry as Rich addresses it (Rich maps some greys one step off), deduplicated
    entries = sorted({_drawn256(EIGHT_BIT_PALETTE[i].hex) for i in range(16, 256)})
    text = _drawn256(muted)
    lum = luminance(background)
    taken = {background, *avoid}
    picked: list[str] = []
    louder = float("inf")
    for i, (target, visible) in enumerate(zip(targets, CURSOR256_VISIBLE)):
        fits = [c for c in entries if c not in taken and visible <= contrast(c, background) < louder
                and (luminance(c) < lum if darker else luminance(c) > lum) and contrast(text, c) >= CURSOR256_TEXT
                and _tint_ok(c, cursor=i == 0)]
        if not fits and picked:                 # nothing quieter fits: share the louder neighbour's (never louder)
            best = picked[-1]
        else:
            fallback = [c for c in entries if c not in taken and _tint_ok(c, cursor=True)
                        and (luminance(c) < lum if darker else luminance(c) > lum)]
            best = min(fits or fallback or entries, key=lambda c: _distance(c, target))
        picked.append(best)
        taken.add(best)
        louder = contrast(best, background)
    return picked[0], picked[1], picked[2]


def _tint_ok(color: str, *, cursor: bool) -> bool:
    """Hues a 256-colour cursor family may take: grey, and for the focused cursor also the accent's blue side. The
    nearest cube entry to a pale blue tint is often pink or cyan, which reads as a different meaning."""
    r, g, b = parse_hex(color) or (0, 0, 0)
    if max(r, g, b) - min(r, g, b) <= 16:
        return True
    return cursor and b > r and b >= g


def color_system(env: dict[str, str] | None = None) -> str | None:
    """The colour system Textual will use (Rich's detection with TEXTUAL_COLOR_SYSTEM; None: not a terminal)."""
    from rich.console import Console

    e = dict(os.environ if env is None else env)
    wanted = e.get("TEXTUAL_COLOR_SYSTEM", "auto")
    try:
        console = Console(force_terminal=True, color_system=wanted if wanted != "auto" else "auto",  # type: ignore
                          _environ=e, file=sys.__stdout__)
    except Exception:
        return None
    return console.color_system


def setup(choice: str = "auto", background: str | None = None, *, env: dict[str, str] | None = None) -> Resolved:
    """Resolve and configure; returns what was chosen and why (Settings › About, `runfleet doctor`)."""
    system = color_system(env)
    r = resolve(choice, env=env, background=background, color_system=system)
    configure(r.mode, r.background, color_system=system, source=r.source)
    return r


configure(resolve("auto").mode, None, source=resolve("auto").source)    # environment only; run_tui refines it


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
def _group_style(group: StateGroup) -> tuple[str, str]:
    """(colour style, NO_COLOR style); read at call time: configure() may change the palette."""
    return {
        StateGroup.ACTIVE: (f"bold {GREEN}", "bold"),
        StateGroup.QUEUED: (YELLOW, ""),
        StateGroup.DONE: (BLUE_DIM, "dim"),
        StateGroup.FAILED: (f"bold {RED}", "reverse"),
        StateGroup.CANCELLED: (CANCELLED, "dim italic"),
        StateGroup.ENDED: (GREY, "dim"),
        StateGroup.UNKNOWN: (f"{GREY} italic", "italic"),
    }[group]


def state_style(state: JobState) -> str:
    if state is JobState.PREEMPTED:
        return _pick(f"bold {ORANGE}", "reverse")
    color, mono = _group_style(state.group)
    return _pick(color, mono)


def state_style_group(group: StateGroup) -> str:
    color, mono = _group_style(group)
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
    palette = HOST_PALETTE
    return palette[index % len(palette)]


def track() -> str:
    """Background of the empty part of a bar (a faint track, so 0 % still reads as a bar)."""
    return "" if no_color() or not TRACK else f"on {TRACK}"


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
    """Active tab / selected row / primary button in dialogs: panel colour on the accent (herdr's settings look). In
    ansi mode the panel is the terminal's default background, and "default on blue" would draw the terminal's
    foreground on its blue (1.5:1 on some profiles): reverse video draws the terminal's own background on it."""
    return _pick(on_color(ACCENT), "bold reverse")


def on_color(color: str) -> str:
    """Bold panel-colour text on `color` (pills, cause chips); ansi: `color` in reverse video, so the text is drawn
    in the terminal's own background colour whatever its foreground."""
    return f"bold reverse {color}" if ANSI else f"bold {SURFACE} on {color}"


def pill_quiet() -> str:
    """Secondary button (`esc close`)."""
    return _pick(f"bold {MUTED} on {RULE}", "bold")


def link_style(*, hover: bool = False) -> RichStyle:
    """Widget.link_style of clickable bars (header chips, hints, view tabs): no colour of its own, so every chip keeps
    the colours of its text — Textual's link-color would repaint them all in one colour (the active tab's dark text on
    the accent became light grey on blue, 1.5:1). Hover: underlined, no background of its own (a tint behind the
    active tab's pill would replace the accent it is drawn on)."""
    from rich.style import Style

    return Style(underline=True) if hover else Style()


def rule() -> str:
    return _pick(RULE, "dim")
