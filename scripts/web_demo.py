#!/usr/bin/env python3
"""Record the web demo of RunFleet: the TUI on the demo fleet, as replayable styled text (headless, no network).

    uv run python scripts/web_demo.py [--deny PATH] [--world-hook FILE] [--out PATH] [--size 114x36] [--svg-dir DIR]
                                      [--settle 70]

Starts the full stack (FleetService + collectors + DemoTransport + RunFleetApp) in Textual's headless test driver,
lets the first polls land, then walks a scripted tour: the Fleet view host by host (each host's detail, the help /
free GPUs / needs-you overlays), the Jobs view row by row (each job's detail: every tab for a curated set, Summary
and Logs for the rest) and the GPUs view row by row. Every state's screen is captured as styled text, together with
the keys and clicks that lead from state to state, and written as one JSON file (format v1) that the website's
player replays like a live terminal:

    {"v": 1, "cols": C, "rows": R, "fg": "#rrggbb", "bg": "#rrggbb",
     "styles": [[fg, bg, flags], …],        reverse and dim applied; flags 1 bold, 2 italic, 4 underline, 8 strike
     "lines": [[[text, style], …], …],      de-duplicated screen rows, each exactly C cells wide
     "states": {id: {"title": str, "frames": [[R line indices], …], "ms": cycle interval of several frames,
                     "keys": {Textual key name: id}, "hot": [[x, y, w, h, id, "click" | "dbl"] | [x, y, w, h, key,
                     "key"], …]}},
     "start": id, "legend": [[key, label], …], "meta": {"runfleet": version, "generated": date, "size": "CxR"}}

A key missing from "keys" is not wired; a move that goes nowhere in the app (up on the first row) leads to the same
state. A "key" area (an item of the hint bar) is a click that presses that key. Clickable text of the app (Textual
@click actions: the logo, breadcrumbs, view labels, hint bar, job links) is wired where it leads to a recorded state,
and so are the job rows of the Fleet cards and panel. Where the app's screen depends on the way it was reached, the demo keeps one capture per place: a table row is
the capture made walking down (an up move in the scrolled part of a table shows the rows one line off from the app),
and a view keeps its cursor for one Tab / Shift+Tab round trip only.

Times are shown in UTC. --world-hook names a Python file, loaded before the DemoWorld is built, whose adjust(world) is
called with it after it is built and before its first sync; it lets a recording relabel or reshape the simulated
fleet. --deny names a list of terms the output must never contain; it is read at run time only and its terms are
never written anywhere. Plain entries (whole tokens of letters, digits, _ . -, any case) and "re:" regexes (any case)
are searched in every captured line, title and reference SVG; "struct:" entries must not be a partition / QOS /
account / host / node / user name of the world ("struct:<name>-partition" etc.: <name> must not be a name of that
kind). On any hit nothing is written: the offending states are listed and the exit code is 1.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import html
import importlib.util
import io
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rich.cells import cell_len
from rich.color import blend_rgb
from rich.color_triplet import ColorTriplet
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from textual.actions import parse as parse_action
from textual.widgets import DataTable, TabbedContent

import runfleet
from runfleet.core.clock import RealClock
from runfleet.core.service import FleetService
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld, SimCluster, SimWorkstation
from runfleet.models import Job, JobKey, JobKind, JobState
from runfleet.ui import fleet_layout, theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.messages import ClockTick
from runfleet.ui.screens.host_detail import HostDetailScreen
from runfleet.ui.screens.job_detail import TABS, JobDetailScreen
from runfleet.ui.screens.main import MainScreen
from runfleet.ui.widgets.fleet_panel import FleetPanel
from runfleet.ui.widgets.host_card import HostCard

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "site" / "runfleet-demo.json"

SEED = 7
BACKGROUND = "#14161b"            # the demo terminal's background (the palette's neutrals are mixed from it)
# The world starts this long before the app: the scripted job changes of its first half hour (an eval completes, a
# job is preempted and requeued, an array moves on, a time limit hits) have happened, and nothing changes for the
# next ~18 minutes (no new submissions either), so every capture of a job agrees with every other one.
HEAD_START = timedelta(minutes=36)
LIVE_FRAMES, LIVE_EVERY = 5, 3.0   # states that show the fleet is live: frames, seconds between them
PEEK_WAIT = 1.6                    # the Jobs view peeks at the log after a 1 s cursor dwell
GPU_TAB_WAIT = 20.0                # live readings of a running Slurm job: two samples, so the idle-GPU hint shows
TAB_TITLES = {"summary": "Summary", "gpu": "GPU", "logs": "Logs", "steps": "Steps", "procs": "Procs", "raw": "Raw"}

# Jobs whose every detail tab is captured (host, ID column); the others get Summary and Logs.
CURATED: tuple[tuple[str, str], ...] = (
    ("k2", "3370390"),         # running, busy GPU
    ("k2", "3372094"),         # running on two nodes (an srun step)
    ("vega", "823954"),        # running on two GPUs, the second one idle while holding memory
    ("k2", "3372115"),         # preempted and requeued once, running again
    ("lyra", "32377076_5"),    # a running element of a throttled array
    ("k2", "3372143"),         # pending on resources, with priority factors
    ("vega", "824479"),        # pending on the QOS running-job cap
    ("k2", "3368404"),         # failed: CUDA out of memory (cause chip OOM-GPU)
    ("k2", "3371400"),         # killed by the cgroup OOM killer
    ("vega", "824373"),        # failed: NCCL watchdog
    ("vega", "824295"),        # failed: exit 1 after a Traceback
    ("k2", "3371193"),         # hit its time limit
    ("k2", "3371311"),         # cancelled before it started
    ("k2", "3370737"),         # completed, its GPU idle the whole run
    ("labstation", "p363942"),  # a workstation task: a queue runner and its child
)
LIVE_LOG = ("lyra", "32377076_5")   # its Logs tab follows the growing log (an epoch every 30 s) over several frames

# Reference SVGs (fidelity checks): state -> file name; fleet.svg (the website teaser) is drawn at the end.
SVGS = {"fleet/k2": "start.svg", "jobs/0": "jobs.svg", "job/k2:3368404/summary": "job_summary.svg",
        "job/vega:823954/gpu": "job_gpu.svg", "host/k2": "host_detail.svg"}
LEGEND = [["tab", "next view"], ["shift+tab", "previous view"], ["left", "left"], ["up", "up"], ["down", "down"],
          ["right", "right"], ["enter", "open"], ["escape", "back"], ["left_square_bracket", "previous tab"],
          ["right_square_bracket", "next tab"], ["question_mark", "help"]]   # the player's key row, in this order

# ====================================================================================================== checks
CATEGORIES = ("partition", "qos", "account", "host", "node", "user")


def world_names(world: DemoWorld) -> dict[str, set[str]]:
    """Every partition / QOS / account / host / node / user name of the world, by kind."""
    out: dict[str, set[str]] = {k: set() for k in CATEGORIES}

    def items(text: str) -> list[str]:
        return [x for x in text.split(",") if x]

    for name, host in world.hosts.items():
        out["host"].add(name)
        out["user"].add(host.user)
        if isinstance(host, SimWorkstation):
            out["host"].add(host.hostname)
            out["user"].update(p.user for p in host.procs.values())
            continue
        out["host"].add(host.login_host)
        out["qos"].update(q for q, _factor in host.qos)
        for _cluster, account, partition, qos in host.associations:
            out["account"].add(account)
            out["partition"].update(items(partition))
            out["qos"].update(items(qos))
        for partition, fields in host.partition_fields.items():
            out["partition"].add(partition)
            out["qos"].update(items(fields.get("QoS", "")) + items(fields.get("AllowQos", "")))
        out["partition"].update(host.partition_tier)
        out["partition"].update(t[1] for t in host.submit_templates)
        out["account"].update(row[0] for row in host.shares)
        for table in (host.others_share, host.others_30d, host.account_limits, host.realloc):
            out["account"].update(table)
        for node in host.nodes:
            out["node"].add(node.name)
            out["partition"].update(node.partitions)
        for job in host.jobs.values():
            out["partition"].update(items(job.partition))
            out["qos"].add(job.qos)
            out["account"].add(job.account)
            out["user"].add(job.user)
            out["node"].update(job.nodes)
        for t in host.history_spec.templates if host.history_spec else ():
            out["account"].add(t.account)
            out["partition"].add(t.partition)
            out["qos"].add(t.qos)
    for cfg in world.configs.values():
        out["host"].update((cfg.name, cfg.ssh))
        out["partition"].update(cfg.slurm.protected_partitions)
        out["qos"].update(cfg.slurm.protected_qos)
    return out


def check_names(world: DemoWorld, deny: Deny) -> list[str]:
    """The kinds of the world's names the deny list forbids (the names themselves stay out of the output)."""
    return [f"{kind}: a name on the deny list" for kind, values in world_names(world).items()
            if any(deny.forbids(kind, v) for v in values)]


@dataclass(frozen=True)
class Deny:
    """The deny list (format in the module docstring). Empty without --deny."""

    tokens: re.Pattern[str] | None = None
    patterns: tuple[re.Pattern[str], ...] = ()
    names: frozenset[str] = frozenset()                 # struct: forbidden as any name
    scoped: frozenset[tuple[str, str]] = frozenset()    # struct: <name>-<kind>: forbidden as a name of that kind

    @classmethod
    def load(cls, path: Path) -> Deny:
        tokens: list[str] = []
        patterns: list[re.Pattern[str]] = []
        names: set[str] = set()
        scoped: set[tuple[str, str]] = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("re:"):
                patterns.append(re.compile(line[3:].strip(), re.IGNORECASE))
            elif line.startswith("struct:"):
                name = line[7:].strip().lower()
                names.add(name)
                base, _, kind = name.rpartition("-")
                if base and kind in CATEGORIES:
                    scoped.add((kind, base))
            else:
                tokens.append(line)
        alternatives = "|".join(map(re.escape, sorted(tokens, key=len, reverse=True)))
        token_re = re.compile(rf"(?<![\w.-])(?:{alternatives})(?![\w.-])", re.IGNORECASE) if tokens else None
        return cls(token_re, tuple(patterns), frozenset(names), frozenset(scoped))

    def forbids(self, kind: str, name: str) -> bool:
        return name.lower() in self.names or (kind, name.lower()) in self.scoped

    def hits(self, text: str) -> list[str]:
        found = [m[0] for m in self.tokens.finditer(text)] if self.tokens is not None else []
        return found + [m[0] for p in self.patterns for m in p.finditer(text)]


def load_hook(path: Path) -> Callable[[DemoWorld], None]:
    """The adjust(world) of a --world-hook file."""
    spec = importlib.util.spec_from_file_location("web_demo_world_hook", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: not a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adjust = getattr(module, "adjust", None)
    if not callable(adjust):
        raise SystemExit(f"{path}: defines no adjust(world)")
    return adjust


# ====================================================================================================== capture
Hot = tuple[int, int, int, int, str, str]


@dataclass
class State:
    title: str
    frames: list[list[int]]
    ms: int = 0
    keys: dict[str, str] = field(default_factory=dict)
    hot: list[Hot] = field(default_factory=list)


class Recorder:
    """Screens as de-duplicated rows of (text, style) runs: what App.export_screenshot draws, before the SVG."""

    def __init__(self, app: RunFleetApp, size: tuple[int, int]) -> None:
        self.cols, self.rows = size
        self.fg, self.bg = theme.FOREGROUND, BACKGROUND
        ansi = app.ansi_theme_dark              # what Textual maps ANSI colours to
        self.palette = TerminalTheme(_rgb(self.bg), _rgb(self.fg), [ansi.ansi_colors[i] for i in range(8)],
                                     [ansi.ansi_colors[i] for i in range(8, 16)])
        self.styles: list[tuple[str, str, int]] = []
        self.lines: list[list[tuple[str, int]]] = []
        self._style_ids: dict[Style, int] = {}
        self._entry_ids: dict[tuple[str, str, int], int] = {}
        self._line_ids: dict[tuple[tuple[str, int], ...], int] = {}

    def frame(self, app: RunFleetApp) -> list[int]:
        rows = [self._line(line) for line in screen_rows(app)]
        if len(rows) != self.rows:
            raise RuntimeError(f"captured {len(rows)} rows, expected {self.rows}")
        return rows

    def text(self, index: int) -> str:
        return "".join(text for text, _style in self.lines[index])

    def _line(self, segments: Iterable[Segment]) -> int:
        runs: list[tuple[str, int]] = []
        for seg in segments:
            if not seg.text:
                continue
            sid = self._style(seg.style)
            if runs and runs[-1][1] == sid:
                runs[-1] = (runs[-1][0] + seg.text, sid)
            else:
                runs.append((seg.text, sid))
        width = sum(cell_len(text) for text, _sid in runs)
        if width != self.cols:
            raise RuntimeError(f"row of {width} cells, expected {self.cols}: {''.join(t for t, _ in runs)!r}")
        key = tuple(runs)
        if key not in self._line_ids:
            self._line_ids[key] = len(self.lines)
            self.lines.append(runs)
        return self._line_ids[key]

    def _style(self, style: Style | None) -> int:
        style = style or Style.null()
        sid = self._style_ids.get(style)
        if sid is not None:
            return sid
        color, bgcolor = style.color, style.bgcolor
        fg = _rgb(self.fg) if color is None or color.is_default else color.get_truecolor(self.palette)
        bg = _rgb(self.bg) if bgcolor is None or bgcolor.is_default else bgcolor.get_truecolor(self.palette, False)
        if style.reverse:
            fg, bg = bg, fg
        if style.dim:
            fg = blend_rgb(fg, bg, 0.4)
        flags = sum(bit for bit, on in ((1, style.bold), (2, style.italic), (4, style.underline or style.underline2),
                                        (8, style.strike)) if on)
        entry = (fg.hex, bg.hex, flags)
        if entry not in self._entry_ids:
            self._entry_ids[entry] = len(self.styles)
            self.styles.append(entry)
        sid = self._style_ids[style] = self._entry_ids[entry]
        return sid


def screen_rows(app: RunFleetApp) -> list[list[Segment]]:
    """The screen as rows of segments, the way App.export_screenshot renders it."""
    cols, rows = app.size
    console = Console(width=cols, height=rows, file=io.StringIO(), force_terminal=True, color_system="truecolor",
                      legacy_windows=False, safe_box=False)
    update = app.screen._compositor.render_update(full=True, screen_stack=app._background_screens, simplify=True)
    segments = Segment.filter_control(console.render(update, console.options))
    return list(Segment.split_and_crop_lines(segments, cols, pad=False, include_new_lines=False))


def click_spans(app: RunFleetApp) -> list[tuple[int, int, int, str]]:
    """Runs of cells whose style carries a Textual @click action, as (x, y, width, action): what a click there runs."""
    cols, rows = app.size
    out: list[tuple[int, int, int, str]] = []
    for y in range(rows):
        run: list[Any] | None = None
        for x in range(cols):
            style = app.screen.get_style_at(x, y)
            action = style.meta.get("@click") if style is not None else None
            if not action:
                run = None
            elif run is not None and run[3] == action:
                run[2] += 1
            else:
                run = [x, y, 1, action]
                out.append(run)              # type: ignore[arg-type]
    return [(x, y, w, a) for x, y, w, a in out]


def plain_rows(app: RunFleetApp) -> list[str]:
    return ["".join(seg.text for seg in row) for row in screen_rows(app)]


def _rgb(hex_color: str) -> ColorTriplet:
    value = hex_color.lstrip("#")
    return ColorTriplet(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def job_id(key: JobKey, tab: str) -> str:
    return f"job/{key.host}:{key.label}/{tab}"


# ====================================================================================================== the tour
class Tour:
    def __init__(self, app: RunFleetApp, pilot: Any, size: tuple[int, int], svgs: bool) -> None:
        self.app, self.pilot = app, pilot
        self.rec = Recorder(app, size)
        self.states: dict[str, State] = {}
        self.want_svgs = svgs
        self.svgs: dict[str, str] = {}
        self.svg_text: dict[str, list[str]] = {}     # SVG name -> its screen rows (the scan covers both)
        self.rows: list[JobKey] = []                  # Jobs view rows, top to bottom
        self.walked: dict[JobKey, JobState] = {}      # each job's state when its row was captured
        self.opens: dict[str, tuple[str, str]] = {}   # state -> (what enter opens there, where esc goes from it)
        self.hosts: list[str] = []
        self.home: dict[str, str] = {}                # view -> its canonical state (Tab arrives there)
        self.checks: list[str] = []
        self.raw_hot: dict[str, list[Hot]] = {}       # graph(): each state's click targets before "@…" resolution

    @property
    def main(self) -> MainScreen:
        main = self.app.main_screen()
        assert main is not None
        return main

    # ------------------------------------------------------------------ driving and capturing
    async def press(self, *keys: str, wait: float = 0.15) -> None:
        await self.pilot.press(*keys)
        await asyncio.sleep(wait)

    async def until(self, ok: Callable[[], bool], what: str, timeout: float = 15.0) -> None:
        end = time.monotonic() + timeout
        while not ok():
            if time.monotonic() > end:
                raise RuntimeError(f"timed out waiting for {what}")
            await asyncio.sleep(0.05)

    async def shot(self, frames: int = 1) -> list[list[int]]:
        out = []
        for k in range(frames):
            if k:
                await asyncio.sleep(LIVE_EVERY)
            # what the 1 Hz clock tick would show now (the app runs without animations), and no toasts
            base = self.app.base_screen()
            if base is not None:
                base.post_message(ClockTick())
            self.app.clear_notifications()
            await self.pilot.pause()
            await asyncio.sleep(0.05)
            out.append(self.rec.frame(self.app))
        return out

    async def capture(self, sid: str, title: str, frames: int = 1) -> State:
        shots = await self.shot(frames)
        state = self.states[sid] = State(title, shots, int(LIVE_EVERY * 1000) if frames > 1 else 0)
        state.hot = self.screen_clicks()
        if self.want_svgs and sid in SVGS:
            self.svg(SVGS[sid])
        return state

    def svg(self, name: str) -> None:
        self.svgs[name] = self.app.export_screenshot(title="RunFleet")
        self.svg_text[name] = plain_rows(self.app)

    # ------------------------------------------------------------------ geometry of clickable things
    def screen_clicks(self) -> list[Hot]:
        """The screen's clickable text (Textual @click actions) that the demo can follow. Targets "@view:<view>",
        "@host:<host>" and "@job:<host>:<id>" depend on where the state is reached from; graph() resolves them."""
        main = self.app.screen is self.main
        out: list[Hot] = []
        for x, y, w, action in click_spans(self.app):
            dest, name, params = parse_action(action)
            call, args = f"{dest}.{name}", tuple(str(p) for p in params)
            view = "fleet" if call == "app.home" else args[0] if call == "screen.show_view" and args else \
                args[1] if call == "app.crumb" and args[:1] == ("view",) and len(args) == 2 else ""
            target, kind = "", "click"
            if view and not main:
                target = f"@view:{view}"                  # back to that view (as it was left)
            elif view:
                target = self.home[view] if view != self.main.view else ""
            elif call == "app.crumb" and args[:1] == ("host",) and len(args) == 2:
                target = f"@host:{args[1]}"
            elif call == "app.crumb" and args[:1] == ("job",) and len(args) == 3:
                target = f"@job:{args[1]}:{args[2]}"
            elif call == "app.open_job" and args:
                target = f"@job:{args[0]}"
            elif call == "app.press" and args:
                target, kind = args[0], "key"             # a hint bar item: the same as pressing its key
            if target:                                    # (header chips change the scope, ⚙ opens settings: off)
                out.append((x, y, w, 1, target, kind))
        return out

    def fleet_links(self, sid: str) -> list[Hot]:
        """The job rows of the Fleet cards and panel (a click opens that job), and the panel's free-GPUs header."""
        out: list[Hot] = []
        for card in self.main.fleet.query(HostCard):
            r = card.content_region
            for i, key in enumerate(card.row_keys):
                if key is not None:                       # the card is focused first: esc comes back to its host
                    target = f"@job:{key}|fleet/{card.host}"
                    out += [(r.x, r.y + i, r.width, 1, target, "click"), (r.x, r.y + i, r.width, 1, target, "dbl")]
        for panel in self.main.fleet.query(FleetPanel):
            r = panel.content_region
            for i, key in enumerate(panel.row_keys):
                target = f"@job:{key}" if isinstance(key, JobKey) else f"{sid}/free" if key == fleet_layout.ALL_FREE \
                    else ""
                if target:
                    out += [(r.x, r.y + i, r.width, 1, target, "click"), (r.x, r.y + i, r.width, 1, target, "dbl")]
        return out

    def table_hot(self, table: DataTable[Any], view: str) -> list[Hot]:
        """Visible rows of a view's table (click = move the cursor there, double-click = what enter opens there)."""
        region = table.scrollable_content_region
        top = table.header_height if table.show_header else 0
        out: list[Hot] = []
        y = top
        for index, row in enumerate(table.ordered_rows):
            screen_y = region.y + y - round(table.scroll_y)
            if region.y + top <= screen_y and screen_y + row.height <= region.bottom:
                out.append((region.x, screen_y, region.width, row.height, f"{view}/{index}", "click"))
                out.append((region.x, screen_y, region.width, row.height, f"open:{view}/{index}", "dbl"))
            y += row.height
        return out

    # ------------------------------------------------------------------ Fleet view, host details, overlays
    async def fleet(self) -> None:
        main = self.main
        fleet = main.fleet
        self.hosts = self.app.host_names()
        self.home = {"fleet": f"fleet/{self.hosts[0]}", "jobs": "jobs/0", "gpus": "gpus/0"}
        cards = {card.host: card.region for card in fleet.query(HostCard)}
        hot: list[Hot] = []
        for host, r in cards.items():
            hot += [(r.x, r.y, r.width, r.height, f"fleet/{host}", "click"),
                    (r.x, r.y, r.width, r.height, f"open:fleet/{host}", "dbl")]
        for i, host in enumerate(self.hosts):
            if i:
                await self.press("l")
            assert fleet.selected == host, (fleet.selected, host)
            state = await self.capture(f"fleet/{host}", f"Fleet view · {host}", LIVE_FRAMES if i == 0 else 1)
            state.hot += self.fleet_links(f"fleet/{host}") + hot
        for host in self.hosts:                       # where each move leads, by pressing it
            for key in ("up", "down", "left", "right", "k", "j", "h", "l"):
                await self.select(host)
                focused = self.app.focused
                await self.press(key, wait=0.05)
                if fleet.selected != host:
                    self.states[f"fleet/{host}"].keys[key] = f"fleet/{fleet.selected}"
                elif self.app.screen is main and main.view == "fleet" and self.app.focused is focused:
                    self.states[f"fleet/{host}"].keys[key] = f"fleet/{host}"     # at the edge: stays (no-op)
        for host in self.hosts:
            sid = f"fleet/{host}"
            await self.select(host)
            await self.press("enter")
            await self.until(lambda: isinstance(self.app.screen, HostDetailScreen), "host detail")
            await asyncio.sleep(1.5)
            await self.capture(f"host/{host}", f"Host detail · {host}")
            await self.back(sid)
            self.states[sid].keys["enter"] = f"host/{host}"
            self.states[f"host/{host}"].keys.update(escape=sid, q=sid)
            for key, name, title, closers in (("question_mark", "help", "Help", ("question_mark", "escape", "f1")),
                                              ("g", "free", "Free GPUs", ("escape", "q")),
                                              ("a", "attention", "Needs you", ("escape", "q"))):
                await self.press(key, wait=0.6)
                overlay = await self.capture(f"{sid}/{name}", f"{title} · over the Fleet view")
                overlay.keys.update(dict.fromkeys(closers, sid))
                self.states[sid].keys[key] = f"{sid}/{name}"
                if name == "attention":            # enter opens the highlighted job; esc there comes back here
                    await self.press("enter", wait=0.5)
                    if isinstance(self.app.screen, JobDetailScreen):
                        self.opens[f"{sid}/{name}"] = (job_id(self.app.screen.key, "summary"), sid)
                await self.back(sid, closers[0])

    async def select(self, host: str) -> None:
        fleet = self.main.fleet
        fleet.selected = host
        fleet.focus_selection()
        await self.pilot.pause()

    async def back(self, sid: str, key: str = "escape") -> None:
        """Close what is open over the main screen and check that the main screen shows `sid` again."""
        await self.press(key, wait=0.3)
        await self.until(lambda: self.app.screen is self.main, f"back to {sid}", timeout=5)
        main = self.main
        where = {"fleet": f"fleet/{main.fleet.selected}", "jobs": f"jobs/{main.jobs.table.cursor_row}",
                 "gpus": f"gpus/{main.gpus.table.cursor_row}"}[main.view]
        if where != sid:
            raise RuntimeError(f"expected {sid} after {key}, found {where}")

    # ------------------------------------------------------------------ Jobs view and job details
    async def jobs(self) -> None:
        await self.select(self.hosts[0])
        await self.press("tab")
        main = self.main
        table = main.jobs.table
        assert main.view == "jobs" and table.cursor_row == 0
        n = table.row_count
        for i in range(n):
            if i:
                await self.press("j")
            assert table.cursor_row == i
            job = main.jobs.current_job()
            assert job is not None, f"row {i} is not a job"
            self.rows.append(job.key)
            self.walked[job.key] = job.state
            await asyncio.sleep(PEEK_WAIT)
            state = await self.capture(f"jobs/{i}", f"Jobs view · row {i + 1} of {n} · {job.key.host}:"
                                       f"{job.key.label}", LIVE_FRAMES if i == 0 else 1)
            state.hot += self.table_hot(table, "jobs")
        if [self.row_key(r) for r in range(n)] != self.rows:
            raise RuntimeError("the Jobs view changed while it was walked")
        step = max(1, (table.size.height - 1) // 2)      # ClickTable.action_half_page
        moves = {"g": 0, "home": 0, "G": n - 1, "end": n - 1}
        for key, row in (("g", 0), ("end", n - 1), ("home", 0), ("pagedown", min(n - 1, step)), ("G", n - 1),
                         ("pageup", max(0, n - 1 - step)), ("ctrl+u", max(0, n - 1 - 2 * step)),
                         ("ctrl+d", max(0, n - 1 - step))):
            await self.press(key)
            self.check(table.cursor_row == row, f"Jobs view {key} -> row {row}")
        for i in range(n):
            keys = self.states[f"jobs/{i}"].keys
            keys.update({k: f"jobs/{r}" for k, r in moves.items()})       # at the edge a move stays (no-op)
            for k, r in (("j", i + 1), ("down", i + 1), ("k", i - 1), ("up", i - 1), ("pagedown", i + step),
                         ("ctrl+d", i + step), ("pageup", i - step), ("ctrl+u", i - step)):
                keys[k] = f"jobs/{max(0, min(n - 1, r))}"
            keys["enter"] = job_id(self.rows[i], "summary")

    def row_key(self, row: int) -> JobKey:
        table = self.main.jobs.table
        job = table.rows_by_key[table.order[row]]
        assert isinstance(job, Job), f"row {row} is not a job"
        return job.key

    async def details(self) -> None:
        main = self.main
        table = main.jobs.table
        await self.press("g")
        curated = set(CURATED)
        found = {(k.host, k.label) for k in self.rows} & curated
        missing = curated - found
        if missing:
            print(f"curated jobs not in the Jobs view (skipped): {sorted(missing)}", file=sys.stderr)
        if len(found) < 8:
            raise RuntimeError(f"only {len(found)} curated jobs in the Jobs view")
        changed = []
        for i, key in enumerate(self.rows):
            if i:
                await self.press("j", wait=0.05)
            assert table.cursor_row == i and self.row_key(i) == key
            job = main.jobs.current_job()
            if job is None or job.state is not self.walked[key]:
                changed.append(str(key))
            await self.press("enter")
            await self.until(lambda: isinstance(self.app.screen, JobDetailScreen), "job detail")
            screen = self.app.screen
            assert isinstance(screen, JobDetailScreen) and screen.key == key
            await self.until(lambda: screen.detail is not None or screen.detail_error is not None, "detail")
            await asyncio.sleep(1.0)
            tabs = TABS if (key.host, key.label) in curated else ("summary", "logs")
            content = screen.query_one(TabbedContent)
            for tab in tabs:
                while content.active != tab:
                    await self.press("right_square_bracket")
                live = screen.job is not None and screen.job.kind is JobKind.SLURM and \
                    screen.job.state is JobState.RUNNING
                await asyncio.sleep(GPU_TAB_WAIT if tab == "gpu" and live else 0.8)
                frames = LIVE_FRAMES if tab == "logs" and (key.host, key.label) == LIVE_LOG else 1
                state = await self.capture(job_id(key, tab), f"Job {key.host}:{key.label} · {TAB_TITLES[tab]}",
                                           frames)
                state.keys.update(escape=f"jobs/{i}", q=f"jobs/{i}")
                for pane in TABS:
                    r = content.get_tab(pane).region
                    if pane != tab:
                        state.hot.append((r.x, r.y, r.width, r.height, job_id(key, pane), "click"))
            for k, tab in enumerate(TABS):              # ] was pressed; [ is its inverse (the keymap's prev_tab)
                if job_id(key, tab) in self.states:
                    self.states[job_id(key, tab)].keys.update(right_square_bracket=job_id(key, TABS[(k + 1) % 6]),
                                                              left_square_bracket=job_id(key, TABS[k - 1]))
            await self.back(f"jobs/{i}")
        self.check(not changed, f"no job changed state during the tour {changed or ''}")

    # ------------------------------------------------------------------ GPUs view
    async def gpus(self) -> None:
        await self.press("tab")
        main = self.main
        table = main.gpus.table
        assert main.view == "gpus"
        await self.press("g")
        n = table.row_count
        for g in range(n):
            if g:
                await self.press("j")
            assert table.cursor_row == g
            row = main.gpus.rows[g]
            node = f" {row.node}" if row.node not in ("", "—") else ""
            state = await self.capture(f"gpus/{g}", f"GPUs view · row {g + 1} of {n} · {row.host}{node} · GPU "
                                       f"{row.idx}")
            state.hot += self.table_hot(table, "gpus")
            moves = (("j", g + 1), ("down", g + 1), ("k", g - 1), ("up", g - 1), ("g", 0), ("home", 0), ("G", n - 1),
                     ("end", n - 1))
            state.keys.update({k: f"gpus/{max(0, min(n - 1, r))}" for k, r in moves})   # at the edge: stays (no-op)
            await self.press("enter", wait=0.5)            # what enter opens here
            screen = self.app.screen
            if isinstance(screen, JobDetailScreen):
                self.opens[f"gpus/{g}"] = (job_id(screen.key, "summary"), f"gpus/{g}")
            elif isinstance(screen, HostDetailScreen):
                self.opens[f"gpus/{g}"] = (f"host/{screen.host}", f"gpus/{g}")
            await self.back(f"gpus/{g}")

    # ------------------------------------------------------------------ the website teaser and click checks
    async def teaser(self) -> None:
        """fleet.svg: the Fleet view cut to the cards, the first host selected."""
        main = self.main
        await self.to_fleet()
        cols, rows = self.app.size
        height = max(card.region.bottom for card in main.fleet.query(HostCard)) + 1      # + the hint bar
        await self.pilot.resize_terminal(cols, height)
        await asyncio.sleep(0.5)
        await self.pilot.pause()
        self.svg("fleet.svg")
        await self.pilot.resize_terminal(cols, rows)
        await asyncio.sleep(0.5)

    async def to_fleet(self) -> None:
        while self.main.view != "fleet":
            await self.press("tab")
        await self.select(self.hosts[0])

    def check(self, ok: bool, what: str) -> None:
        self.checks.append(f"{'ok  ' if ok else 'FAIL'} {what}")

    async def click(self, sid: str, target: str, kind: str = "click") -> None:
        """Click the hot area of `sid` that leads to `target` in the app, where no area listed before it lies."""
        hot = [h for h in self.states[sid].hot if h[5] == kind]
        area = next(h for h in hot if h[4] == target)
        x, y, w, h, *_ = area
        points = [(px, py) for py in range(y, y + h) for px in range(x, x + w)]
        mid = (x + w // 2, y + h // 2)
        points.sort(key=lambda p: abs(p[0] - mid[0]) + abs(p[1] - mid[1]))
        point = next(p for p in points if next(a for a in hot if a[0] <= p[0] < a[0] + a[2]
                                               and a[1] <= p[1] < a[1] + a[3]) is area)
        await self.pilot.click(offset=point, times=2 if kind == "dbl" else 1)
        await asyncio.sleep(0.6)

    async def verify_clicks(self) -> None:
        main = self.main
        await self.to_fleet()
        start, second, third = (f"fleet/{h}" for h in self.hosts[:3])
        await self.click(start, second)
        self.check(main.fleet.selected == self.hosts[1], f"click a card -> {second}")
        await self.click(start, f"open:{third}", "dbl")
        screen = self.app.screen
        self.check(isinstance(screen, HostDetailScreen) and screen.host == self.hosts[2],
                   f"double-click a card -> host/{self.hosts[2]}")
        await self.back(f"fleet/{self.hosts[2]}")
        await self.click(start, "jobs/0")
        self.check(main.view == "jobs", "click the Jobs label -> Jobs view")
        await self.press("g")
        await self.click("jobs/0", "jobs/2")
        self.check(main.jobs.table.cursor_row == 2, "click a job row -> jobs/2")
        await self.click("jobs/0", "open:jobs/2", "dbl")
        screen = self.app.screen
        summary = job_id(self.rows[2], "summary")
        self.check(isinstance(screen, JobDetailScreen) and screen.key == self.rows[2],
                   f"double-click a job row -> {summary}")
        if isinstance(screen, JobDetailScreen):
            await self.until(lambda: screen.detail is not None or screen.detail_error is not None, "detail")
            await self.click(summary, job_id(self.rows[2], "logs"))
            self.check(screen.query_one(TabbedContent).active == "logs", "click the Logs tab label -> Logs tab")
            await self.back("jobs/2")
        await self.click("jobs/0", "gpus/0")
        self.check(main.view == "gpus", "click the GPUs label -> GPUs view")
        await self.click("gpus/0", "gpus/1")
        self.check(main.gpus.table.cursor_row == 1, "click a GPU row -> gpus/1")
        await self.verify_links()

    async def verify_links(self) -> None:
        """The logo, breadcrumbs, host chip, hint bar and the Fleet cards' job rows, clicked in the app."""
        main = self.main
        await self.to_fleet()
        start, first = f"fleet/{self.hosts[0]}", self.hosts[0]

        def on_main(view: str) -> bool:
            return self.app.screen is main and main.view == view

        row = next((h[4] for h in self.states[start].hot if h[5] == "click" and h[4].startswith("@job:")
                    and "|" in h[4] and not h[4].endswith(f"|{start}")), "")
        self.check(bool(row), "a job row in another host's Fleet card")
        if row:
            key, _, via = row[5:].partition("|")
            for then in ("escape", "@view:fleet", "@host"):
                await self.click(start, row)
                screen = self.app.screen
                ok = isinstance(screen, JobDetailScreen) and str(screen.key) == key
                self.check(ok, f"click {key} in a Fleet card -> its job detail")
                if not isinstance(screen, JobDetailScreen):
                    await self.to_fleet()
                    continue
                sid = job_id(screen.key, "summary")
                await self.until(lambda: screen.detail is not None or screen.detail_error is not None, "detail")
                if then == "escape":
                    await self.press("escape", wait=0.5)
                    self.check(on_main("fleet") and f"fleet/{main.fleet.selected}" == via,
                               f"esc from a job opened in a Fleet card -> {via}")
                elif then == "@view:fleet":
                    await self.click(sid, then)
                    self.check(on_main("fleet"), "click the RunFleet logo in a job detail -> Fleet view")
                else:
                    await self.click(sid, f"@host:{screen.key.host}")
                    host = self.app.screen
                    self.check(isinstance(host, HostDetailScreen) and host.host == screen.key.host,
                               f"click the host chip of a job detail -> host/{screen.key.host}")
                    await self.click(f"host/{screen.key.host}", "@view:fleet")
                    self.check(on_main("fleet"), "click the RunFleet logo in a host detail -> Fleet view")
                await self.to_fleet()
        await self.click(start, "enter", "key")
        host = self.app.screen
        self.check(isinstance(host, HostDetailScreen) and host.host == first, f"click ↵ in the hint bar -> host/{first}")
        if isinstance(host, HostDetailScreen):
            await self.click(f"host/{first}", "@view:fleet")
            self.check(on_main("fleet"), "click Hosts in a host detail's breadcrumb -> Fleet view")
        await self.to_fleet()
        await self.press("tab")
        await self.press("g")
        await self.press("j", "j")
        await self.press("enter")
        await self.until(lambda: isinstance(self.app.screen, JobDetailScreen), "job detail")
        await self.click(job_id(self.rows[2], "summary"), "@view:jobs")
        self.check(on_main("jobs") and main.jobs.table.cursor_row == 2,
                   "click Jobs in a job detail's breadcrumb -> jobs/2")

    # ------------------------------------------------------------------ the graph
    def alias(self, base: str, old: str, new: str, family: str = "") -> str:
        """`base` reached another way: a copy whose references to `old` lead to `new` (and so do the copies of the
        states of its family it leads to), e.g. a job opened from the GPUs view, where esc goes back there."""
        sid = f"{base}@{new}"
        if sid not in self.states:
            src = self.states[base]
            copy = self.states[sid] = State(src.title, src.frames, src.ms)

            def to(target: str) -> str:
                if target == old:
                    return new
                if target == base:                  # a key that stays (no-op) stays in the copy
                    return sid
                if family and target.startswith(family) and target in self.states:
                    return self.alias(target, old, new, family)
                return target

            copy.keys = {k: to(t) for k, t in src.keys.items()}
            copy.hot = [(*h[:4], to(h[4]), h[5]) for h in self.raw_hot.get(base, src.hot)]
        return sid

    def resolve(self, sid: str, target: str) -> str:
        """Where a "@…" click target of `sid` leads ("": nowhere recorded), as the app goes there from that state."""
        states = self.states
        back = states[sid].keys.get("escape", "")
        if target.startswith("@view:"):               # the logo, "Jobs" / "Hosts": the main screen on that view,
            view = target[6:]                          # with its cursor where it was left
            return back if back.split("/", 1)[0] == view else self.home[view]
        if target.startswith("@host:"):               # the host chip of a detail screen: its host detail
            base = f"host/{target[6:]}"
            if base not in states or sid.split("@", 1)[0] == base:
                return sid if base in states else ""
            old = states[base].keys["escape"]          # opened over the main screen: esc goes back there
            return base if back in ("", old) else self.alias(base, old, back)
        key, _, via = target[5:].partition("|")        # "@job:<host>:<id>[|<state esc returns to>]"
        host, _, jid = key.partition(":")
        base = job_id(JobKey(host, jid), "summary")
        if base not in states or (not via and sid.startswith("job/") and "@" in sid):
            return ""                                  # a job's link from a job opened another way: not followed
        old, new = states[base].keys["escape"], via or sid
        return base if new == old else self.alias(base, old, new, base.rsplit("/", 1)[0] + "/")

    def opened(self, target: str) -> str:
        """Hot targets "open:<state>" stand for what enter opens in <state>."""
        return self.states[target[5:]].keys.get("enter", "") if target.startswith("open:") else target

    def graph(self) -> dict[str, State]:
        states = self.states
        for sid, (target, back) in self.opens.items():   # opened from elsewhere: esc goes back there
            if target in states:
                family = target.rsplit("/", 1)[0] + "/" if target.startswith("job/") else ""
                states[sid].keys["enter"] = self.alias(target, states[target].keys["escape"], back, family)
        views = ("fleet", "jobs", "gpus")
        for v, view in enumerate(views):                  # Tab / Shift+Tab (and the view labels)
            nxt, prev = self.home[views[(v + 1) % 3]], self.home[views[v - 1]]
            for sid in [s for s in states if s.startswith(f"{view}/") and s.count("/") == 1]:
                states[sid].keys.update({"tab": nxt, "shift+tab": prev})
        for v, view in enumerate(views):                  # the view left keeps its cursor for the round trip
            home = self.home[view]
            for sid in [s for s in states if s.startswith(f"{view}/") and s.count("/") == 1 and s != home]:
                swap = {t: self.alias(t, home, sid) for t in (self.home[views[(v + 1) % 3]],
                                                              self.home[views[v - 1]])}
                states[sid].keys = {k: swap.get(t, t) for k, t in states[sid].keys.items()}
                states[sid].hot = [(*h[:4], swap.get(h[4], h[4]), h[5]) for h in states[sid].hot]
        self.raw_hot = {sid: list(state.hot) for sid, state in states.items()}
        todo = list(states)                              # "@…" targets; resolving one may add aliases to resolve
        while todo:
            sid = todo.pop()
            known = set(states)
            states[sid].hot = [(*h[:4], self.resolve(sid, h[4]), h[5]) if h[4].startswith("@") else h
                               for h in states[sid].hot]
            todo += [s for s in states if s not in known]
        for state in states.values():                    # double-click = enter on that row / card
            state.hot = [(*h[:4], self.opened(h[4]), h[5]) for h in state.hot]
        for state in states.values():                    # only what was captured is wired
            state.keys = {k: t for k, t in state.keys.items() if t in states}
            state.hot = [h for h in state.hot if h[5] == "key" or h[4] in states]
        return states


# ====================================================================================================== main
def demo_world(adjust: Callable[[DemoWorld], None] | None) -> DemoWorld:
    """The demo fleet HEAD_START into its scenario, without new submissions; `adjust` (the world hook) runs first."""
    shift = HEAD_START
    clock = RealClock()

    def now() -> datetime:
        return clock.now() - shift

    world = DemoWorld(seed=SEED, now=now)
    if adjust is not None:
        adjust(world)
    for faults in world.faults.values():          # a healthy fleet: every host answers
        faults.clear()
    for host in world.hosts.values():
        if isinstance(host, SimCluster):
            host.next_submit = float("inf")
    shift = timedelta(0)
    world.sync()
    return world


async def record(size: tuple[int, int], settle: float, svgs: bool, world: DemoWorld) -> Tour:
    clock = RealClock()
    service = FleetService(world.fleet_config(), lambda h: DemoTransport(h, world), clock, None)
    service.lock_path = False                       # like --demo: no instance lock
    service.config_path = None                      # like --demo: no config file
    service.alias_source = lambda: [c.name for c in world.host_configs()]   # like --demo: never ~/.ssh/config
    app = RunFleetApp(service, clock=clock, animations=False, persist_state=False)   # no ui.json
    async with app.run_test(size=size) as pilot:
        tour = Tour(app, pilot, size, svgs)
        await asyncio.sleep(settle)                 # squeue/sacct, then the sstat round for running jobs
        started = time.monotonic()
        for step in (tour.fleet, tour.jobs, tour.details, tour.gpus):
            await step()
            print(f"{step.__name__}: {len(tour.states)} states, {time.monotonic() - started:.0f}s", file=sys.stderr)
        if svgs:
            await tour.teaser()
        await tour.verify_clicks()
        await pilot.press("ctrl+c")
    return tour


def document(tour: Tour, size: tuple[int, int]) -> dict[str, Any]:
    states = tour.graph()
    start = tour.home["fleet"]
    for sid, state in states.items():
        for target in [*state.keys.values(), *(h[4] for h in state.hot if h[5] != "key")]:
            assert target in states, (sid, target)
        assert state.frames and all(len(f) == size[1] for f in state.frames), sid
    rec = tour.rec
    return {
        "v": 1, "cols": size[0], "rows": size[1], "fg": rec.fg, "bg": rec.bg,
        "styles": [list(s) for s in rec.styles],
        "lines": [[list(run) for run in line] for line in rec.lines],
        "states": {sid: {"title": s.title, "frames": s.frames, "ms": s.ms, "keys": s.keys,
                         "hot": [list(h) for h in s.hot]} for sid, s in states.items()},
        "start": start,
        "legend": LEGEND,
        "meta": {"runfleet": runfleet.__version__, "generated": datetime.now(UTC).date().isoformat(),
                 "size": f"{size[0]}x{size[1]}"},
    }


def scan(deny: Deny, doc: dict[str, Any], tour: Tour) -> list[str]:
    """Where everything that would be written has deny-list hits, as "<where>: N hits". The terms themselves are never
    printed, so a log of the run cannot leak them either."""
    problems: list[str] = []
    users: dict[int, list[str]] = {}
    for sid, state in doc["states"].items():
        for frame in state["frames"]:
            for index in frame:
                users.setdefault(index, []).append(sid)

    def found(where: str, text: str) -> None:
        if n := len(deny.hits(text)):
            problems.append(f"{where}: {n} hit{'s' if n > 1 else ''}")

    for index in range(len(doc["lines"])):
        found(f"states {', '.join(sorted(set(users[index])))}", tour.rec.text(index))
    for sid, state in doc["states"].items():
        found(f"state id / title {sid}", f"{sid}\n{state['title']}")
    found("legend / meta", json.dumps([doc["legend"], doc["meta"]]))
    found("JSON", json.dumps(doc, ensure_ascii=False))
    for name, svg in tour.svgs.items():
        for text in ("\n".join(tour.svg_text[name]), html.unescape(re.sub(r"<[^>]+>", "", svg)),
                     html.unescape(re.sub(r"<[^>]+>", " ", svg))):
            found(name, text)
    return sorted(set(problems))


def main() -> int:
    os.environ["TZ"] = "UTC"                        # the recording shows UTC, whatever zone it is made in
    time.tzset()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deny", type=Path, help="deny list: terms the output must never contain")
    p.add_argument("--world-hook", type=Path, help="Python file whose adjust(world) changes the demo world first")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON to write (default: site/runfleet-demo.json)")
    p.add_argument("--size", default="114x36", help="terminal size COLSxROWS (default 114x36, the smallest that "
                                                     "gives the Fleet view its roomy cards)")
    p.add_argument("--svg-dir", type=Path, help="also write reference SVGs (start, jobs, job summary, job GPU, host "
                                                "detail, fleet teaser) here")
    p.add_argument("--settle", type=float, default=70.0,
                   help="seconds to let the first polls and the first sstat round land")
    args = p.parse_args()
    cols, rows = (int(x) for x in args.size.lower().split("x"))
    if args.deny is None:
        print("WARNING: no --deny list: the output is not scanned for terms it must not contain.", file=sys.stderr)
    if args.world_hook is None:
        print("WARNING: no --world-hook: this records the stock demo world, not the one on the project page.",
              file=sys.stderr)
    deny = Deny.load(args.deny) if args.deny is not None else Deny()
    began = time.monotonic()
    theme.configure("dark", BACKGROUND, color_system="truecolor", source="web demo")
    world = demo_world(load_hook(args.world_hook) if args.world_hook is not None else None)
    problems = check_names(world, deny)
    if problems:
        print("names of the demo world:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    tour = asyncio.run(record((cols, rows), args.settle, args.svg_dir is not None, world))
    doc = document(tour, (cols, rows))
    problems = scan(deny, doc, tour)
    if problems:
        print("deny-list hits (nothing written):\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    failed = [c for c in tour.checks if c.startswith("FAIL")]
    print("\n".join(tour.checks), file=sys.stderr)
    if failed:
        print(f"{len(failed)} checks failed (nothing written)", file=sys.stderr)
        return 1
    raw = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    if args.svg_dir is not None:
        args.svg_dir.mkdir(parents=True, exist_ok=True)
        for name, svg in tour.svgs.items():
            (args.svg_dir / name).write_text(svg, encoding="utf-8")
    frames = sum(len(s["frames"]) for s in doc["states"].values())
    print(f"{args.out}: {len(doc['states'])} states, {frames} frames, {len(doc['lines'])} lines, "
          f"{len(raw):,} bytes ({len(gzip.compress(raw, 9)):,} gzipped), {time.monotonic() - began:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
