"""Roomy Fleet layout (§7.4): row heights, ranked job lists, row templates and the fleet panel, as pure functions.

The Fleet cards grow into the empty space of a tall terminal (tier T1) without ever changing height on their own:
:func:`plan` is a function of the view size and each card's *shape* (how many lines its fixed part can take, kept as a
per-session high-water mark), so data only changes what fills the fixed rows. Below the gate (:data:`ROOMY_MIN_H`
rows, :data:`ROOMY_MIN_INNER` cells per card, :data:`LIST_MIN` spare rows per grid row) or with ui.json
``fleet_layout = "compact"`` the plan is tier T0: today's cards, unchanged.

At T1 a card lists the jobs that need the user first (:func:`rank_slurm`, :func:`rank_ws`), one row per job in a
fixed template (:func:`row_text`), and an orphan grid cell (or a full-width row under the grid) holds the fleet panel:
free GPUs by type counted once per node (:func:`free_by_type`) and what changed lately (:func:`activity_rows`). All
of it reads the store; nothing here asks a host for anything.

Every list row and activity row keeps the :class:`JobKey` it shows (:func:`list_region`, :func:`activity_region`), so a
click opens that job; :func:`attention` merges the cards' rankings into the fleet-wide "needs you" list (``a``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, TypeVar

from rich.cells import cell_len
from rich.text import Text

from runfleet.core.events import BAD_END, JobEvent, count_line, end_summary, end_word, ended_since, time_low
from runfleet.models import GPU, ConnState, HostSnapshot, Job, JobKey, JobKind, JobState, StateGroup
from runfleet.ui import render, theme
from runfleet.ui.query import _id_key, elapsed_s, ended_at, left_s
from runfleet.ui.widgets.job_table import exit_summary, gpus_text, name_badges, runway_level
from runfleet.util import humanize
from runfleet.util.humanize import gpu_type_from_name, task_label
from runfleet.util.slurmfmt import node_free

if TYPE_CHECKING:
    from runfleet.core.store import HostState

CARDS_MIN_WIDTH = 100            # narrower: the host table (mockup B)
ROOMY_MIN_H = 32                 # FleetView height for T1
ROOMY_MIN_INNER = 52             # card inner width for T1
LIST_MIN = 3                     # spare rows needed per grid row
WIDE_INNER = 72                  # wide row template / owner column
SLURM_LIST_CAP = 12
WS_LIST_CAP = 8
PANEL_CAP = 24                   # panel content rows (orphan cell)
PANEL_MIN = 6                    # orphan cell
PANEL_ROW_MIN = 8                # full-width row below the grid
HOST2_MIN_LIST = 8               # list rows needed before the workstation "host2" line appears
FRESH_FAIL_S = 4 * 3600          # a failure this recent ranks first
LABEL_W = 10                     # = host_card.LABEL_W (the label gutter)
MAX_GPU_LINES = 8                # = host_card.MAX_GPU_LINES
OWNER_WIDE = 22                  # owner column of a workstation GPU line (wide cards)
OWNER_NARROW = 6                 # owner tag (narrow cards)
PANEL_TITLE = "fleet · free GPUs · what changed"

_FRESH = (ConnState.OK, ConnState.DEGRADED)
_T = TypeVar("_T")
Group = Literal["failed", "running", "pending", "others"]
_GROUP_ORDER: dict[str, int] = {"failed": 0, "running": 1, "pending": 2, "others": 3}


def columns_for(width: int) -> int:
    return 3 if width >= 180 else 2


def inner_width(width: int, cols: int) -> int:
    """Text width inside a card: the scrollbar gutter takes one column, border and padding 4 cells per card."""
    return max(30, (width - 1) // cols - 4)


class Memo:
    """Results of the fleet-wide scans (every job, every node), kept while their inputs are the same objects: the
    store replaces a host's ``jobs`` / ``nodes`` / ``gpus`` when a poll changes them, so the 1 Hz tick only re-renders
    text. ``same`` is compared by identity (the memo holds a reference, so an id is never reused), ``eq`` by value.
    ``expires(value)`` (optional) gives the moment a value goes out of date on its own (a job leaving the recent
    window): from then on (``now`` at or past it) it is computed again."""

    def __init__(self) -> None:
        self._slots: dict[tuple[str, str],
                          tuple[tuple[object, ...], tuple[object, ...], object, datetime | None]] = {}

    def get(self, host: str, what: str, same: tuple[object, ...], eq: tuple[object, ...],
            compute: Callable[[], _T], *, now: datetime | None = None,
            expires: Callable[[_T], datetime | None] | None = None) -> _T:
        hit = self._slots.get((host, what))
        if (hit is not None and len(hit[0]) == len(same) and all(a is b for a, b in zip(hit[0], same, strict=True))
                and hit[1] == eq and (hit[3] is None or now is None or now < hit[3])):
            return hit[2]     # type: ignore[return-value]
        value = compute()
        self._slots[(host, what)] = (same, eq, value, expires(value) if expires is not None else None)
        return value

    def clear(self) -> None:
        self._slots.clear()


def type_label(name: str | None) -> str:
    """One label per GPU model for Slurm gres types and nvidia-smi names alike (``nvidia_a100-sxm4-40gb`` and
    ``NVIDIA A100-SXM4-40GB`` -> ``A100-40G``; ``NVIDIA H100 80GB HBM3`` -> ``H100-80G``), so the fleet panel never
    splits one model in two: a glued ``rtx3090`` / ``rtx6000`` reads like ``GeForce RTX 3090`` / ``Quadro RTX 6000``
    (``3090``, ``6000``), and an Ada card keeps its generation (``RTX 6000 Ada Generation`` -> ``6000Ada``) so it is
    never counted with the Turing ``RTX 6000``. A ``Ti`` / ``SUPER`` suffix stays with the number (``rtx2080ti`` and
    ``GeForce RTX 2080 Ti`` -> ``2080TI``), and ``GTX`` is a prefix like ``RTX`` (``gtx1080ti`` -> ``1080TI``, not
    one ``GTX`` row for every GTX card)."""
    if not name:
        return humanize.gpu_label(None)
    gtype = _GLUED_RTX.sub(r"\1\2_", gpu_type_from_name(name))
    gtype = _GTX.sub(r"\1", _SUFFIX.sub(r"\1\2", gtype))
    label = humanize.gpu_label(gtype)
    if _ADA.search(gtype) and "ADA" not in label.upper():
        model = humanize.gpu_model(gtype)
        label = label.replace(model, f"{model}Ada", 1)
    return label


_GLUED_RTX = re.compile(r"(^|_)(rtx|gtx)(?=a?\d)")      # rtx3090 / rtxa6000 -> rtx_3090 / rtx_a6000
_ADA = re.compile(r"(?:^|[_-])ada(?:[_-]|$)")
_SUFFIX = re.compile(r"(\d)[_-](ti|super)(?=[_-]|$)")    # rtx_2080_ti -> rtx_2080ti (as the glued gres spelling)
_GTX = re.compile(r"(^|_)gtx_")                           # gtx_1080ti -> 1080ti (gpu_model skips RTX, not GTX)


# ---------------------------------------------------------------- shapes and the plan
@dataclass(frozen=True, slots=True)
class Shape:
    """Content lines of one card: ``t0`` today (T0), ``t1`` the fixed part at T1, ``cap`` list rows it can use;
    ``fixed`` = the padded block (free-GPU lines F of a Slurm card, GPU lines G of a workstation)."""
    t0: int
    t1: int
    cap: int
    kind: Literal["none", "slurm", "ws"] = "none"
    fixed: int = 0


NO_DATA = Shape(3, 3, 0)


@dataclass(slots=True)
class HighWater:
    """Per-host, per-session high-water marks of the shape inputs (reset only when the host set changes)."""
    free: dict[int, int] = field(default_factory=dict)      # free-block width -> most lines seen
    gpus: int = 0


def worst_partitions(snap: HostSnapshot) -> HostSnapshot:
    """The snapshot with every partition entirely free and planned: the widest free-GPU text its names and totals
    can produce, so the free block's height never depends on the free counts."""
    parts = tuple(replace(p, gpus_free=p.gpus_total, gpus_free_planned=p.gpus_total,
                          gpu_types={k: (v[1], v[1]) for k, v in p.gpu_types.items()}) for p in snap.partitions)
    return replace(snap, partitions=parts)


def shape_of(state: HostState, inner: int, hw: HighWater) -> Shape:
    """A card's shape (updates ``hw``): Slurm (5 + F, 4 + F, 12), workstation (G + 3, G + 1, 8), no data (3, 3, 0)."""
    from runfleet.ui.widgets.host_card import _free_lines, is_cluster   # host_card renders the T1 lists from here

    snap = state.snapshot
    if snap is None:
        return NO_DATA
    if is_cluster(state):
        width = inner - LABEL_W
        # the worst case is not always the tallest: a partition too wide for the block even alone collapses it into
        # "+N more", while today's smaller numbers fit on a second line
        f = max(len(_free_lines(worst_partitions(snap), width)), len(_free_lines(snap, width)))
        f = hw.free[width] = max(hw.free.get(width, 0), f)
        return Shape(5 + f, 4 + f, SLURM_LIST_CAP, "slurm", f)
    n = len(snap.gpus)
    g = max(1, min(n, MAX_GPU_LINES) + (1 if n > MAX_GPU_LINES else 0))
    g = hw.gpus = max(hw.gpus, g)
    return Shape(g + 3, g + 1, WS_LIST_CAP, "ws", g)


@dataclass(frozen=True, slots=True)
class Panel:
    span: int                    # grid columns (the orphan cells, or every column for a row under the grid)
    rows: int                    # content rows (border excluded)
    below: bool = False          # a full-width row under the grid rather than the orphan cell


@dataclass(frozen=True, slots=True)
class Plan:
    mode: Literal["table", "cards"]
    tier: int = 0
    cols: int = 2
    inner: int = 30
    wide: bool = False
    heights: tuple[int, ...] = ()            # content rows per grid row (T1)
    panel: Panel | None = None


def plan(width: int, height: int, shapes: Sequence[Shape], enabled: bool = True) -> Plan:
    """Row heights for a view of ``width`` × ``height`` (see the module docstring)."""
    if width < CARDS_MIN_WIDTH:
        return Plan("table")
    cols = columns_for(width)
    inner = inner_width(width, cols)
    rows = [list(shapes[i:i + cols]) for i in range(0, len(shapes), cols)]
    n_rows = len(rows)
    base = [max(s.t0 for s in row) for row in rows]
    spare = height - sum(b + 2 for b in base)
    if not (enabled and rows and height >= ROOMY_MIN_H and inner >= ROOMY_MIN_INNER and spare >= LIST_MIN * n_rows):
        return Plan("cards", 0, cols, inner)
    orphan = len(shapes) % cols != 0
    cap = [max(s.t1 + s.cap for s in row) for row in rows]
    if orphan:
        cap[-1] = max(cap[-1], PANEL_CAP)
    h = base[:]
    while spare and any(h[r] < cap[r] for r in range(n_rows)):     # round-robin +1, top row first
        for r in range(n_rows):
            if spare and h[r] < cap[r]:
                h[r] += 1
                spare -= 1
    panel = None
    if orphan and h[-1] >= PANEL_MIN:
        h[-1] += spare
        panel = Panel(cols - len(shapes) % cols, h[-1])
    elif not orphan and spare >= PANEL_ROW_MIN + 2:
        panel = Panel(cols, spare - 2, below=True)
    return Plan("cards", 1, cols, inner, inner >= WIDE_INNER, tuple(h), panel)


# ---------------------------------------------------------------- ranking
@dataclass(frozen=True, slots=True)
class Ranked:
    rank: int
    job: Job
    group: Group
    members: tuple[Job, ...] = ()            # array siblings folded into this row (job first); () = just the job

    @property
    def count(self) -> int:
        return len(self.members) or 1


class RankedList(list[Ranked]):
    """A card's ranked list (:func:`rank_slurm`, :func:`rank_ws`) with its selections by row count, so the 1 Hz tick
    folds and selects once per ranking, not once per render. Never mutated after ranking."""

    def __init__(self, items: Iterable[Ranked] = ()) -> None:
        super().__init__(items)
        self.picks: dict[int, tuple[list[Ranked], list[Ranked]]] = {}


def _age_s(job: Job, now: datetime) -> float | None:
    end = ended_at(job)
    return None if end is None else (now - end).total_seconds()


def _newest(job: Job) -> float:
    end = ended_at(job)
    return -end.timestamp() if end is not None else 0.0


def _warn_hint(job: Job) -> str | None:
    return next((h for h in job.hints if render.hint_level(h) in ("warn", "red")), None)


def _red_hint(job: Job) -> str | None:
    return next((h for h in job.hints if render.hint_level(h) == "red"), None)


def _runway(job: Job, now: datetime) -> str | None:
    left = left_s(job, now)
    if left is None or job.state.group is not StateGroup.ACTIVE:
        return None
    return runway_level(left, job.time_limit.total_seconds() if job.time_limit is not None else None)


def _est_key(job: Job, now: datetime) -> tuple[int, float]:
    if job.start_time is not None and job.est_start:
        return (0, job.start_time.timestamp())
    return (1, 0.0)


def rank_slurm(jobs: Iterable[Job], now: datetime, window: float) -> RankedList:
    """Own Slurm jobs by how much they need the user: 0 fresh failures (newest first), 1 running with a warn/red hint
    or low on time, 2 pending that will never run or held, 3 older failures within ``window``, 4 other running
    (soonest end first, no limit last), 5 other pending (soonest estimate first). Ties by job id. CD/CA never."""
    out: list[tuple[tuple[object, ...], Ranked]] = []
    for j in jobs:
        if not j.is_mine:
            continue
        group = j.state.group
        if group is StateGroup.FAILED:
            age = _age_s(j, now)
            if age is None or age > window:
                continue
            rank = 0 if age <= FRESH_FAIL_S else 3
            out.append(((rank, _newest(j), _id_key(j)), Ranked(rank, j, "failed")))
        elif group is StateGroup.ACTIVE:
            if _warn_hint(j) or time_low(j, now) or _runway(j, now) in ("red", "amber"):
                out.append(((1, 0.0, _id_key(j)), Ranked(1, j, "running")))
            else:
                left = left_s(j, now)
                out.append(((4, (left is None, left or 0.0), _id_key(j)), Ranked(4, j, "running")))
        elif group is StateGroup.QUEUED:
            red = _red_hint(j) or render.reason_red(j.reason)
            if red or j.held:
                out.append(((2, 0.0, _id_key(j)), Ranked(2, j, "pending")))
            else:
                out.append(((5, _est_key(j, now), _id_key(j)), Ranked(5, j, "pending")))
    out.sort(key=lambda p: p[0])
    return RankedList(r for _k, r in out)


def ws_failed(job: Job) -> bool:
    """A task failure: FAILED group, or EXITED with a red hint (a plain EXITED task, status unknown, is not one)."""
    if job.state.group is StateGroup.FAILED:
        return True
    return job.state is JobState.EXITED and any(render.hint_level(h) == "red" for h in job.hints)


def rank_ws(jobs: Iterable[Job], now: datetime, window: float) -> RankedList:
    """Workstation tasks: 0 fresh failures, 1 own running with a warn/red hint, 2 own running (most GPUs first),
    3 other users' tasks, 4 older failures within ``window``; ties by id."""
    out: list[tuple[tuple[object, ...], Ranked]] = []
    for j in jobs:
        if j.kind is not JobKind.TASK:
            continue
        if j.state.is_terminal:
            if not j.is_mine or not ws_failed(j):
                continue
            age = _age_s(j, now)
            if age is None or age > window:
                continue
            rank = 0 if age <= FRESH_FAIL_S else 4
            out.append(((rank, _newest(j), _id_key(j)), Ranked(rank, j, "failed")))
        elif j.state.group is StateGroup.ACTIVE:
            if not j.is_mine:
                out.append(((3, 0, _id_key(j)), Ranked(3, j, "others")))
            elif _warn_hint(j):
                out.append(((1, 0, _id_key(j)), Ranked(1, j, "running")))
            else:
                out.append(((2, -j.alloc.gpus, _id_key(j)), Ranked(2, j, "running")))
    out.sort(key=lambda p: p[0])
    return RankedList(r for _k, r in out)


def rank_expiry(ranked: Iterable[Ranked], window: float) -> datetime | None:
    """When a ranking goes out of date on its own: the first failure that turns 4 h old (it leaves rank 0) or leaves
    ``window`` (it leaves the list, as it leaves the card's ``my jobs`` count). None: never."""
    out: datetime | None = None
    for r in ranked:
        end = ended_at(r.job) if r.group == "failed" else None
        if end is None:
            continue
        for at in ((end + timedelta(seconds=FRESH_FAIL_S)) if r.rank == 0 else None, end + timedelta(seconds=window)):
            if at is not None and (out is None or at < out):
                out = at
    return out


def _collapse(items: Sequence[Ranked]) -> list[Ranked]:
    """Rank 4/5 elements of one array (same ``array_job_id`` and group) fold into the row of the first (members
    gathered in lists, each folded row built once: linear in the number of siblings)."""
    out: list[Ranked] = []
    members: dict[tuple[str, str], list[Job]] = {}
    at: dict[tuple[str, str], int] = {}
    for it in items:
        aid = it.job.array_job_id
        if it.rank not in (4, 5) or not aid:
            out.append(it)
            continue
        key = (aid, it.group)
        if key in members:
            members[key].append(it.job)
        else:
            members[key] = [it.job]
            at[key] = len(out)
            out.append(it)
    for k, jobs in members.items():
        if len(jobs) > 1:
            out[at[k]] = replace(out[at[k]], members=tuple(jobs))
    return out


def select(ranked: Sequence[Ranked], rows: int) -> tuple[list[Ranked], list[Ranked]]:
    """The rows to show (grouped failed, running, pending / others, each in rank order) and the hidden rest. With more
    candidates than ``rows``, array siblings fold first, then ``rows - 1`` rows leave one for the overflow line.
    A :class:`RankedList` keeps the answer per ``rows``."""
    picks = ranked.picks if isinstance(ranked, RankedList) else None
    if picks is not None and rows in picks:
        return picks[rows]
    if rows <= 0:
        return [], list(ranked)
    items = list(ranked)
    if len(items) > rows:
        items = _collapse(items)
    hidden: list[Ranked] = []
    if len(items) > rows:
        items, hidden = items[:rows - 1], items[rows - 1:]
    shown = sorted(items, key=lambda r: _GROUP_ORDER[r.group])     # stable: rank order inside each group
    if picks is not None:
        picks[rows] = (shown, hidden)
    return shown, hidden


def overflow_line(hidden: Sequence[Ranked], now: datetime, width: int) -> Text:
    """``+7 more · R 5, next ends in 20h38m · PD 2 · tab: Jobs`` (dim)."""
    jobs = [j for r in hidden for j in (r.members or (r.job,))]
    running = [j for j in jobs if j.state.group is StateGroup.ACTIVE]
    pending = [j for j in jobs if j.state.group is StateGroup.QUEUED]
    failed = [j for j in jobs if j.state.is_terminal]
    parts = [f"+{len(jobs)} more"]
    if failed:
        parts.append(f"F {len(failed)}")
    if running:
        lefts = [x for j in running if (x := left_s(j, now)) is not None]
        parts.append(f"R {len(running)}" + (f", next ends in {span_min(min(lefts))}" if lefts else ""))
    if pending:
        parts.append(f"PD {len(pending)}")
    parts.append("tab: Jobs")
    return render.fit(Text(" · ".join(parts), style=theme.dim()), width)


# ---------------------------------------------------------------- row templates
def span_min(seconds: float | None) -> str:
    """:func:`humanize.span` that changes at most once a minute (``<1m`` under 60 s)."""
    if seconds is None:
        return humanize.NA
    return "<1m" if seconds < 60 else humanize.span(seconds)


def _cell(text: Text, width: int, *, right: bool = False) -> Text:
    out = text.copy()
    if out.cell_len > width:
        out.truncate(width, overflow="ellipsis")
    elif right:
        out = Text(" " * (width - out.cell_len)) + out
    else:
        out.pad_right(width - out.cell_len)
    return out


def array_suffix(job: Job, members: Sequence[Job] = ()) -> str:
    """`` [3]`` for an array element, `` [3-4] ×2`` for folded siblings, `` [5-12%2]`` for a pending summary."""
    if len(members) > 1:
        tasks = [(m.array_task or "").strip("[]") for m in members]
        nums = sorted(int(t) for t in tasks if t.isdigit())
        if len(nums) == len(tasks) and nums == list(range(nums[0], nums[0] + len(nums))):
            rng = f"{nums[0]}-{nums[-1]}"
        elif len(nums) == len(tasks):             # task order, not rank order: [3,5,12]
            rng = ",".join(map(str, nums))
        else:
            rng = ",".join(tasks)
        return f" [{humanize.truncate(rng, 12)}] ×{len(members)}"
    if job.array_task:
        return f" [{job.array_task.strip('[]')}]"
    return ""


def _name(item: Ranked, width: int, starts: Mapping[str, int], watched: bool) -> Text:
    job = item.job
    out = Text()
    out.append_text(name_badges(job, watched))
    if job.kind is JobKind.TASK and not job.is_mine:
        out.append(f"{job.user} ", style=theme.muted())
    suffix = array_suffix(job, item.members)
    room = max(1, width - out.cell_len - len(suffix))
    out.append(humanize.truncate_name(job.name, room, starts.get(job.name, 0)))
    out.append(suffix, style=theme.muted())
    return _cell(out, width)


def _task_gpus(job: Job) -> str:
    idx = sorted({i for n in job.alloc.per_node for i in n.gpu_indices})
    if idx:
        return f"gpu {','.join(map(str, idx))}"
    return f"{job.alloc.gpus} GPU{'s' if job.alloc.gpus != 1 else ''}" if job.alloc.gpus else "cpu only"


@dataclass(frozen=True, slots=True)
class Cells:
    detail: Text
    util: Text | None            # None: the detail spans the util column
    time: Text | None            # None: the detail runs to the end of the row
    gpus: Text | None = None     # narrow running rows: f"{gpus:<10}{util:>6}" as the detail


def row_cells(item: Ranked, now: datetime, util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> Cells:
    """Detail / util / time of one list row (§7.4 cell contents)."""
    j = item.job
    if item.group == "failed":
        summary = end_summary(j) or exit_summary(j)
        age = _age_s(j, now)
        when = "<1m ago" if age is not None and age < 60 else f"{humanize.age(age)} ago"
        return Cells(Text(summary, style=theme.state_style(j.state) if j.state.group is StateGroup.FAILED
                          else theme.error()), None, Text(when, style=theme.muted()))
    if j.state.group is StateGroup.ACTIVE:
        util = render.mini_util(j.gpu.util_mean_pct if j.gpu is not None else None, util_t)
        hint = _warn_hint(j)
        gpus: Text | None = None
        if hint:
            detail = Text(hint, style=render.hint_style(hint))
        elif j.kind is not JobKind.TASK and time_low(j, now):
            detail = Text("low on time", style=theme.warn())
        elif j.kind is JobKind.TASK:
            where = _task_gpus(j)
            if j.is_mine:
                extra = f" · cpu {j.cpu_pct:.0f}%" if j.cpu_pct is not None else ""
            else:
                mem = j.gpu.mem_used_mib if j.gpu is not None else None
                extra = f" · {humanize.gib(mem)}" if mem is not None else ""
            detail = Text(where + extra, style=theme.muted())
            gpus = Text(where, style=theme.muted())
        else:
            detail = gpus_text(j)
            gpus = detail
        left = left_s(j, now)
        if j.kind is not JobKind.TASK and left is not None:
            level = runway_level(left, j.time_limit.total_seconds() if j.time_limit is not None else None)
            style = theme.error() if level == "red" else theme.warn() if level == "amber" else ""
            time = Text(f"{span_min(left)} left", style=style)
        else:
            time = Text(f"{span_min(elapsed_s(j, now))} up", style=theme.muted())
        return Cells(detail, util, time, gpus)
    # pending: a known Slurm reason in the table's short wording ("never: dependency failed"), else the red hint
    red = _red_hint(j)
    if j.reason and render.known_reason(j.reason) and render.reason_text(j.reason).plain:
        detail = render.reason_text(j.reason)
    elif red:
        detail = Text(red, style=render.hint_style(red))
    elif j.reason and render.reason_text(j.reason).plain:
        detail = render.reason_text(j.reason)
    elif j.held:
        detail = Text("held", style=theme.warn())
    else:
        detail = Text("pending", style=theme.muted())
    if j.start_time is not None and j.est_start:
        wait = (j.start_time - now).total_seconds()
        time = Text(f"in {span_min(wait)}" if wait >= 0 else "overdue", style=theme.muted())
    else:
        time = Text("no est.", style=theme.dim())
    return Cells(detail, None, time)


def row_text(item: Ranked, now: datetime, *, inner: int, wide: bool, label: str = "",
             starts: Mapping[str, int] | None = None, watched: bool = False,
             util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> Text:
    """One list row, exactly ``inner`` cells.

    Wide (inner >= 72): ``label 10 | badge 4 | name inner-55 | detail 20 | util 6 | time 11``; narrow:
    ``badge 4 | name inner-34 | detail 16 | time 11`` (running: ``gpus 9 + util 6`` as the detail, a space between).
    The time column fits ``20h39m left``, so the util column never moves; a missing util lets the detail take its
    cells."""
    starts = starts or {}
    c = row_cells(item, now, util_t)
    out = Text()
    if wide:
        out.append(label.ljust(LABEL_W)[:LABEL_W], style=theme.label())
    out.append_text(_cell(render.state_badge(item.job.state, item.job.held), 4))
    name_w = inner - (55 if wide else 34)
    out.append_text(_name(item, name_w, starts, watched))
    out.append(" ")
    rest = inner - out.cell_len - 1                  # one spare cell at the end of the row
    if c.time is None:
        out.append_text(_cell(c.detail, rest))
        return render.fit(out, inner)
    time_w = max(11, c.time.cell_len)
    detail_w = rest - time_w - 1
    if wide and c.util is not None:
        out.append_text(_cell(c.detail, detail_w - 7))
        out.append(" ")
        out.append_text(_cell(c.util, 6, right=True))
    elif not wide and c.util is not None and c.gpus is not None:     # running: "1×H200" / "gpu 3" + util
        out.append_text(_cell(c.gpus, detail_w - 7))
        out.append(" ")
        out.append_text(_cell(c.util, 6, right=True))
    else:
        out.append_text(_cell(c.detail, detail_w))
    out.append(" ")
    out.append_text(_cell(c.time, time_w, right=True))
    return render.fit(out, inner)


def list_region(ranked: Sequence[Ranked], rows: int, now: datetime, *, inner: int, wide: bool,
                watched: Callable[[Job], bool] | None = None,
                util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS,
                starts: Mapping[str, int] | None = None) -> tuple[list[Text], list[JobKey | None]]:
    """The list region of a T1 card: at most ``rows`` lines (shown rows, then the overflow line), and the job each
    line opens on a click (a folded array row: its first job; the overflow line: None). ``starts``: the cached
    :func:`humanize.distinct_starts` of the whole ranked list (computed here when None)."""
    shown, hidden = select(ranked, rows)
    if starts is None:
        starts = ranked_starts(ranked)
    out: list[Text] = []
    keys: list[JobKey | None] = []
    last: str | None = None
    for item in shown:
        label = item.group if item.group != last else ""
        last = item.group
        out.append(row_text(item, now, inner=inner, wide=wide, label=label, starts=starts,
                            watched=bool(watched and watched(item.job)), util_t=util_t))
        keys.append(item.job.key)
    if hidden:
        line = overflow_line(hidden, now, inner - (LABEL_W if wide else 0))
        out.append(Text(" " * LABEL_W) + line if wide else line)
        keys.append(None)
    return out, keys


def list_lines(ranked: Sequence[Ranked], rows: int, now: datetime, *, inner: int, wide: bool,
               watched: Callable[[Job], bool] | None = None,
               util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS,
               starts: Mapping[str, int] | None = None) -> list[Text]:
    """The lines of :func:`list_region`."""
    return list_region(ranked, rows, now, inner=inner, wide=wide, watched=watched, util_t=util_t, starts=starts)[0]


def ranked_starts(ranked: Sequence[Ranked]) -> dict[str, int]:
    """Where each name of a card's ranked list starts to differ from its siblings (hidden rows included)."""
    return humanize.distinct_starts([r.job.name for r in ranked])


# ---------------------------------------------------------------- fleet-wide: what needs you (``a``)
TIER_WHY = ("failed lately", "running, needs you", "pending, will not start", "failed earlier")


@dataclass(frozen=True, slots=True)
class Attention:
    """One entry of the "needs you" picker: a card's ranked row that asks for the user."""
    host: str
    item: Ranked
    tier: int                    # 0 fresh failure, 1 running with a hint / low on time, 2 never runs / held, 3 older

    @property
    def key(self) -> JobKey:
        return self.item.job.key


def attention_tier(item: Ranked, cluster: bool) -> int | None:
    """The tier of a ranked row that needs the user, None for the rest (normal running / pending, others' tasks):
    Slurm ranks 0-3 as they are; a workstation's fresh failure 0, task with a hint 1, older failure (rank 4) 3."""
    if item.group == "failed":
        return 0 if item.rank == 0 else 3
    if cluster:
        return item.rank if item.rank in (1, 2) else None
    return 1 if item.rank == 1 and item.group == "running" else None


def attention(ranked: Iterable[tuple[str, bool, Sequence[Ranked]]]) -> list[Attention]:
    """Merge the cards' rankings ``(host, is cluster, ranked list)`` (in host order) into one list: by tier, failures
    newest first across the fleet, the rest in host order, then in each card's rank order. Array siblings are never
    folded here: each is its own entry."""
    keyed: list[tuple[tuple[object, ...], Attention]] = []
    for h, (host, cluster, items) in enumerate(ranked):
        for pos, item in enumerate(items):
            tier = attention_tier(item, cluster)
            if tier is None:
                continue
            newest = _newest(item.job) if item.group == "failed" else 0.0
            keyed.append(((tier, newest, h, pos), Attention(host, item, tier)))
    keyed.sort(key=lambda p: p[0])
    return [a for _k, a in keyed]


def attention_why(entry: Attention, now: datetime,
                  util_t: tuple[float, float] = theme.DEFAULT_UTIL_THRESHOLDS) -> Text:
    """Why an entry needs the user, from the card row's cells: ``exit 1 · 3h ago``, ``log idle 49m · 20m left``,
    ``low on time · 12m left``, ``never: dependency failed · no est.``, ``held · no est.``."""
    item = entry.item
    cells = row_cells(item, now, util_t)
    detail = cells.detail
    if item.group == "running" and not _warn_hint(item.job):
        detail = Text("low on time", style=theme.warn())      # ranked for its runway: the detail is just its GPUs
    out = detail.copy()
    if cells.time is not None:
        out.append(" · ", style=theme.dim())
        out.append_text(cells.time)
    return out


# ---------------------------------------------------------------- workstation owner column
_INTERPRETER = re.compile(r"^(?:python|pypy|bash|sh|zsh|node|ruby|perl|julia|Rscript)[\d.]*$")


def _short_label(job: Job) -> str:
    """``orb.launch (orb.train)`` -> ``orb.launch``; ``python (eval.py)`` -> ``eval.py``."""
    head, _, rest = job.name.partition(" (")
    return rest.rstrip(")") if rest and _INTERPRETER.match(head) else head


def owner_text(snap: HostSnapshot, gpu: GPU, *, old: bool, wide: bool) -> Text:
    """Who holds a workstation GPU: ``?`` (old reading, or ``no reading`` when nvidia-smi gave neither util nor
    memory), ``mine orb.launch`` / ``carol sample.py`` (+N more owners), ``in use · no process`` (busy without a
    visible process), ``free``. Wide: 22 cells; narrow: a 6-cell tag."""
    width = OWNER_WIDE if wide else OWNER_NARROW
    if old:
        return _cell(Text("?", style=theme.dim()), width)
    if gpu.util_pct is None and gpu.mem_used_mib is None and not gpu.processes:    # unknown, not "in use"
        return _cell(Text("no reading" if wide else "?", style=theme.dim()), width)
    me = snap.caps.remote_user if snap.caps is not None else None
    owners: dict[str, tuple[str, bool]] = {}    # who -> (text, mine), first process first
    for p in gpu.processes:
        job = snap.jobs.get(p.job.id) if p.job is not None else None
        if job is None:
            job = next((j for j in snap.jobs.values() if p.pid in j.pids and not j.state.is_terminal), None)
        if job is not None:
            mine = job.is_mine
            who, what = ("mine" if mine else job.user), _short_label(job)
        else:
            mine = p.user is not None and p.user == me
            who, what = ("mine" if mine else (p.user or "?")), (task_label(p.command) if p.command else "")
        owners.setdefault(who, (f"{who} {what}".strip() if wide else who, mine))
    if owners:
        text, mine = next(iter(owners.values()))
        more = Text(f"{' ' if wide else ''}+{len(owners) - 1}", style=theme.warn()) if len(owners) > 1 else Text()
        head = _cell(Text(text, style=theme.accent() if mine else ""), max(1, width - more.cell_len))
        head.rstrip()
        return _cell(head + more, width)
    if render.gpu_in_use(gpu):
        return _cell(Text("in use · no process" if wide else "busy", style=theme.warn()), width)
    return _cell(Text("free", style=theme.ok()), width)


def host2_line(snap: HostSnapshot, width: int) -> Text | None:
    """``swap 2% of 2.0G · up 117d · load 3.4 3.1 2.9`` (None without procfs)."""
    m = snap.metrics
    if m is None:
        return None
    bits: list[str] = []
    if m.swap_total:
        used = (m.swap_total - (m.swap_free or 0)) / m.swap_total * 100
        bits.append(f"swap {used:.0f}% of {humanize.size(m.swap_total)}")
    elif m.swap_total == 0:
        bits.append("no swap")
    if m.uptime_s is not None:
        bits.append(f"up {humanize.age(m.uptime_s)}")
    if m.load is not None:
        bits.append("load " + " ".join(f"{x:.1f}" for x in m.load))
    if not bits:
        return None
    return render.fit(Text(" · ".join(bits), style=theme.muted()), width)


# ---------------------------------------------------------------- fleet panel: free GPUs by type
@dataclass(slots=True)
class TypeFree:
    label: str                   # "H200", "A100-80G", "MIG slices"
    free: int = 0
    total: int = 0
    planned: int = 0
    host: str | None = None      # MIG buckets are per host
    where: dict[tuple[str, str], list[int]] = field(default_factory=dict)   # (host, partition|"gpu") -> [n, planned]
    gpus: dict[str, list[int]] = field(default_factory=dict)               # workstation host -> free indices


@dataclass(frozen=True, slots=True)
class FreeGPUs:
    types: dict[str, TypeFree]                   # key: label, or "MIG slices (host)"
    unknown: tuple[str, ...]                     # problem / stale hosts: not counted ("host ?")
    sinfo_age_s: float | None                    # the oldest sinfo reading among the counted Slurm hosts


_NodeFree = tuple[str, str, bool, str, int, int, int]       # key, label, MIG, partition, free, total, planned

# The Slurm collector's ``slurm.nodes`` cadence (backends/slurm/collector.py, §5.4; the UI never imports a backend):
NODES_FLOOR_S = 60.0             # never faster
NODES_HIDDEN_FACTOR = 5.0        # the host is off screen
NODES_MAX_BACKOFF_S = 600.0      # Cadence.max_backoff: a failing group's period grows up to this
IDLE_FACTOR = 4.0                # FleetService: no input for a while, or the terminal is not focused


def sinfo_bound_s(state: HostState, rate: float = 1.0) -> float:
    """How old the last good ``slurm.sinfo`` sample may be before the fleet panel stops counting the host's nodes:
    two periods of the slowest cadence the poller may legitimately run ``slurm.nodes`` at (``poll_nodes`` × hidden ×5
    × idle ×4 × the +/- ``rate``, × a busy controller's slowdown, at least one backed-off period) plus 5 s, like
    :meth:`HostState.stale`."""
    period = max(max(state.config.slurm.poll_nodes, NODES_FLOOR_S) * NODES_HIDDEN_FACTOR * IDLE_FACTOR * rate,
                 NODES_FLOOR_S)
    period *= max(1.0, state.status.controller_slowdown)
    return 2 * max(period, NODES_MAX_BACKOFF_S) + 5.0


FreeSource = Literal["skip", "unknown", "cluster", "ws"]


def free_source(state: HostState, now: datetime, rate: float = 1.0) -> tuple[FreeSource, float | None]:
    """Whether a host's free GPUs may be counted, and the age of its ``slurm.sinfo`` sample (clusters): ``skip``
    (disabled, or a cluster with no partition in scope: nothing to count, nothing unknown), ``unknown`` (a problem or
    stale host, or a fresh cluster whose last sinfo failed, is missing while nodes come from the cache, or is older than
    :func:`sinfo_bound_s`: the panel's ``host ?``), else ``cluster`` / ``ws``. The fleet panel and the GPU watches
    (:mod:`runfleet.ui.gpu_watch`) share this rule."""
    from runfleet.ui.widgets.host_card import is_cluster

    st = state.status
    if st.state is ConnState.DISABLED:
        return "skip", None
    snap = state.snapshot
    if snap is None or st.state not in _FRESH or state.stale(now):
        return "unknown", None
    if not is_cluster(state):
        return "ws", None
    sec = st.sections.get("slurm.sinfo")
    if sec is None and not snap.nodes:
        return "skip", None
    age = (now - sec.last_ok).total_seconds() if sec is not None and sec.last_ok is not None else None
    if sec is None or not sec.ok or age is None or age > sinfo_bound_s(state, rate):
        return "unknown", None                # free counts we cannot vouch for: "host ?", not stale numbers
    return "cluster", age


def _cluster_free(name: str, snap: HostSnapshot) -> list[_NodeFree]:
    """Per (type, first listed partition) sums over a cluster's nodes (the node scan :func:`free_by_type` caches)."""
    listed = [p.name for p in snap.partitions]
    labels: dict[str, tuple[str, str, bool]] = {}           # gres type -> key, label, MIG
    sums: dict[tuple[str, str], list[int]] = {}             # (gres type, partition) -> free, total, planned
    for node in snap.nodes:
        inside = [p for p in listed if p in node.partitions]
        if not inside:
            continue
        for gtype, free, total, planned in node_free(node):
            if gtype not in labels:
                mig = humanize.is_mig(gtype)
                label = "MIG slices" if mig else type_label(gtype)
                labels[gtype] = (f"MIG slices ({name})" if mig else label, label, mig)
            acc = sums.setdefault((gtype, inside[0]), [0, 0, 0])
            acc[0] += free
            acc[1] += total
            acc[2] += planned
    return [(*labels[gtype], part, acc[0], acc[1], acc[2]) for (gtype, part), acc in sums.items()]


def free_by_type(hosts: Mapping[str, HostState], now: datetime,
                 old: Callable[[HostState, GPU], bool] | None = None, memo: Memo | None = None,
                 rate: float = 1.0) -> FreeGPUs:
    """Free GPUs per type across the fleet, each node counted once (:func:`slurmfmt.node_free`: an allocated node
    with an idle GPU counts 0), only nodes in the host's listed partitions; workstation GPUs that are not in use and
    not an old reading. Problem or stale hosts are listed apart, never counted; so is a fresh cluster whose last
    ``slurm.sinfo`` failed or is older than :func:`sinfo_bound_s` (``rate``: the +/- poll rate), or that has nodes
    from the cache but no sinfo sample yet. MIG slices stay apart per host. ``memo``: the node scan is redone only
    when a host's nodes or partitions change."""
    memo = memo or Memo()
    types: dict[str, TypeFree] = {}
    unknown: list[str] = []
    oldest: float | None = None
    for name, state in hosts.items():
        source, age = free_source(state, now, rate)
        if source == "skip":
            continue
        if source == "unknown":
            unknown.append(name)
            continue
        snap = state.snapshot
        assert snap is not None
        if source == "cluster":
            assert age is not None
            oldest = age if oldest is None else max(oldest, age)
            sums = memo.get(name, "free", (snap.nodes, snap.partitions), (),
                            lambda: _cluster_free(name, snap))      # noqa: B023 (called at once)
            for key, label, mig, part, free, total, planned in sums:
                t = types.setdefault(key, TypeFree(label, host=name if mig else None))
                t.free += free
                t.total += total
                t.planned += planned
                w = t.where.setdefault((name, part), [0, 0])
                w[0] += free
                w[1] += planned
        else:
            for g in snap.gpus:
                label = type_label(g.name)
                t = types.setdefault(label, TypeFree(label))
                t.total += 1
                if render.gpu_in_use(g) or (old is not None and old(state, g)):
                    continue
                t.free += 1
                t.gpus.setdefault(name, []).append(g.index)
    return FreeGPUs(types, tuple(unknown), oldest)


def _host_types(snap: HostSnapshot) -> tuple[set[str], dict[str, datetime]]:
    """One host's GPU labels of own jobs: those of live jobs, and label -> newest end of the ended ones."""
    live: set[str] = set()
    ended: dict[str, datetime] = {}
    by_index = {g.index: g for g in snap.gpus}
    labels: dict[str | None, str] = {}           # a few distinct types among thousands of jobs: label each once
    for j in snap.jobs.values():
        if not j.is_mine:
            continue
        found: set[str] = set()
        if j.kind is JobKind.TASK:
            for n in j.alloc.per_node:
                for i in n.gpu_indices:
                    if i in by_index:
                        found.add(type_label(by_index[i].name))
            if j.alloc.gpu_type and not any(n.gpu_indices for n in j.alloc.per_node):
                found.add(type_label(j.alloc.gpu_type))
        elif j.alloc.gpus and j.alloc.gpu_type and not humanize.is_mig(j.alloc.gpu_type):
            gt = j.alloc.gpu_type
            found.add(labels[gt] if gt in labels else labels.setdefault(gt, type_label(gt)))
        if not found:
            continue
        if not j.state.is_terminal:
            live |= found
            continue
        end = ended_at(j)
        if end is not None:
            for label in found:
                if label not in ended or end > ended[label]:
                    ended[label] = end
    return live, ended


def my_types(hosts: Mapping[str, HostState], now: datetime, window: float, memo: Memo | None = None) -> set[str]:
    """GPU labels the user runs on: own Slurm jobs active, queued or ended within ``window``, and the GPUs of own
    workstation tasks. ``memo``: each host's jobs are scanned again only when they change."""
    memo = memo or Memo()
    out: set[str] = set()
    for name, state in hosts.items():
        snap = state.snapshot
        if snap is None:
            continue
        live, ended = memo.get(name, "types", (snap.jobs, snap.gpus), (),
                               lambda: _host_types(snap))     # noqa: B023 (called at once)
        out |= live
        out.update(label for label, end in ended.items() if (now - end).total_seconds() <= window)
    return out


def free_lines(free: FreeGPUs, mine: set[str], rows: int, width: int, *, wide: bool,
               host_order: Sequence[str] = ()) -> list[Text]:
    """The panel's free part in at most ``rows`` lines: header (``sinfo Nm ago`` on the right), your types (fleet total
    descending, then name; 0 free shown dim), then ``also free …`` for the other types with free GPUs."""
    order = {h: i for i, h in enumerate(host_order)}
    head = Text("free GPUs now · your types first", style=theme.label())
    if free.unknown:
        head.append("  " + " ".join(f"{h} ?" for h in free.unknown), style=theme.warn())
    if free.sinfo_age_s is not None:
        age = free.sinfo_age_s
        tail = Text(f"sinfo {'<1m' if age < 60 else f'{int(age // 60)}m'} ago", style=theme.dim())
        gap = width - head.cell_len - tail.cell_len
        head = head + Text(" " * gap) + tail if gap >= 1 else render.fit(head, width)
    lines = [render.fit(head, width)]
    yours = sorted((t for t in free.types.values() if t.label in mine and t.host is None),
                   key=lambda t: (-t.total, t.label))
    picked = {id(t) for t in yours}
    others = sorted((t for t in free.types.values() if id(t) not in picked and t.free),
                    key=lambda t: (-t.free, t.label, t.host or ""))
    room = rows - 1
    if len(yours) > room - (1 if others else 0):
        keep = max(0, room - 1)
        others = sorted([*(t for t in yours[keep:] if t.free), *others], key=lambda t: (-t.free, t.label))
        yours = yours[:keep]
    lab_w = max(10 if wide else 9, max((len(t.label) + 1 for t in yours), default=0))   # GH200-480G: counts align
    for t in yours:
        line = Text("  " if wide else " ")
        line.append(t.label.ljust(lab_w), style=theme.muted() if t.free else theme.dim())
        line.append(f"{t.free:>3}" if wide else f"{t.free:>3}", style=theme.ok() if t.free else theme.dim())
        if t.free:
            line.append("   " if wide else "  ")
            line.append_text(_where(t, wide, order))
        lines.append(render.fit(line, width))
    if others and len(lines) < rows:
        line = Text("  " if wide else " ")
        line.append("also free".ljust(lab_w + (2 if wide else 1)), style=theme.muted())
        sep = Text(" · ", style=theme.dim())
        for i, t in enumerate(others):
            if i:
                line.append_text(sep)
            line.append(t.label + " ", style=theme.muted())
            line.append(str(t.free), style=theme.ok())
            if t.host:
                line.append(f" ({t.host})", style=theme.dim())
        lines.append(render.fit(line, width))
    return lines[:rows]


def _where(t: TypeFree, wide: bool, order: Mapping[str, int]) -> Text:
    """``vega h200 6 (5 planned) · k2 gpu-preempt 3`` (narrow: per host) / ``labstation gpu 0, 2``."""
    out = Text()
    sep = Text(" · ", style=theme.dim())
    items: list[tuple[int, int, str, Text]] = []
    if wide:
        for (host, part), (n, planned) in t.where.items():
            if n:
                txt = Text(f"{host} {part} {n}")
                if planned:
                    txt.append(f" ({planned} planned)", style=theme.dim())
                items.append((-n, order.get(host, 99), host, txt))
    else:
        per: dict[str, list[int]] = {}
        for (host, _part), (n, planned) in t.where.items():
            acc = per.setdefault(host, [0, 0])
            acc[0] += n
            acc[1] += planned
        for host, (n, planned) in per.items():
            if n:
                txt = Text(f"{host} {n}")
                if planned:
                    txt.append(f" ({planned} planned)", style=theme.dim())
                items.append((-n, order.get(host, 99), host, txt))
    for host, idx in t.gpus.items():
        txt = Text(f"{host} gpu {', '.join(map(str, sorted(idx)))}" if wide else f"{host} {len(idx)}")
        items.append((-len(idx), order.get(host, 99), host, txt))
    for i, (_n, _o, _h, txt) in enumerate(sorted(items, key=lambda x: (x[0], x[1], x[2]))):
        if i:
            out.append_text(sep)
        out.append_text(txt)
    return out


# ---------------------------------------------------------------- fleet panel: what changed
GLYPHS = {"bad": "✖", "ok": "✓", "end": "■", "time_low": "◷", "restarted": "↻", "gpu_idle": "◌", "gpu_free": "◎"}


@dataclass(frozen=True, slots=True)
class Activity:
    at: datetime
    host: str
    glyph: str
    name: str
    summary: str
    bad: bool
    old: bool                     # before ``since``: shown dim
    key: JobKey | None = None     # the job a click on the row opens


def activity_rows(jobs: Iterable[Job], history: Iterable[JobEvent], now: datetime, window: float,
                  since: datetime, extra: Iterable[Activity] = ()) -> list[Activity]:
    """What changed within ``window``, newest first: own job ends from the store (complete, survive a restart),
    this session's ``time_low`` / ``restarted`` / ``gpu_idle`` events (one row per job and kind) and ``extra`` rows
    (GPU watches that fired, :mod:`runfleet.ui.gpu_watch`) inside the window."""
    start = now - timedelta(seconds=window)
    out: list[Activity] = [a for a in extra if a.at >= start]
    for j in ended_since(jobs, start):
        end = ended_at(j)
        assert end is not None
        if j.state is JobState.COMPLETED:
            glyph, summary, bad = GLYPHS["ok"], f"ran {humanize.clock(_ran_s(j))}", False
        elif j.state in BAD_END or (j.kind is JobKind.TASK and ws_failed(j)):
            glyph, summary, bad = GLYPHS["bad"], end_summary(j) or j.state.name.lower(), True
        else:
            glyph, summary, bad = GLYPHS["end"], end_summary(j) or exit_summary(j), False
        out.append(Activity(end, j.key.host, glyph, j.name + array_suffix(j), summary, bad, end <= since, j.key))
    seen: set[tuple[str, str, str]] = set()
    for e in sorted(history, key=lambda e: e.at, reverse=True):
        if e.kind not in ("time_low", "restarted", "gpu_idle") or e.at < start:
            continue
        key = (e.key.host, e.key.id, e.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(Activity(e.at, e.key.host, GLYPHS[e.kind], e.name, e.summary, True, e.at <= since, e.key))
    out.sort(key=lambda a: a.at, reverse=True)
    return out


def own_ended(snap: HostSnapshot) -> list[Job]:
    """A host's own ended jobs (with an end time): all that :func:`activity_rows` and :func:`count_text` read."""
    return [j for j in snap.jobs.values() if j.is_mine and j.state.is_terminal and ended_at(j) is not None]


def _ran_s(job: Job) -> float | None:
    if job.elapsed is not None:
        return job.elapsed.total_seconds()
    if job.start_time is not None and job.end_time is not None:
        return (job.end_time - job.start_time).total_seconds()
    return None


def count_text(jobs: Iterable[Job], history: Iterable[JobEvent], since: datetime) -> str:
    """``2 failed · 1 completed · 1 low on time``: own ends after ``since`` plus this session's time-low warnings."""
    words = [end_word(j.state) for j in ended_since(jobs, since)]
    low = {(e.key.host, e.key.id) for e in history if e.kind == "time_low" and e.at > since}
    words += ["low on time"] * len(low)
    return count_line(words)


DAY_LABEL_S = 20 * 3600         # = notify.since_label: older than this, a time gets its day


def when_label(at: datetime, now: datetime) -> str:
    """An activity row's time: ``11:00`` within 20 h (like :func:`notify.since_label`), else with its day
    (``Mon 11:00``; ``09-06 11:00`` from 6 days back, where a weekday would repeat)."""
    age = (now - at).total_seconds()
    if age < DAY_LABEL_S:
        return humanize.local_hm(at)
    if age < 6 * 86400:
        return at.astimezone().strftime("%a %H:%M")
    return humanize.local_dt(at)


def activity_lines(rows: Sequence[Activity], counts: str, head: str, n: int, width: int, *, wide: bool,
                   host_style: Callable[[str], str] | None = None, now: datetime | None = None) -> list[Text]:
    """The panel's activity part in ``n`` lines: the count line, then rows newest first (older than ``since`` dim).
    Columns are padded by terminal cells (CJK names keep the summary column in place); with ``now``, a row older than
    20 h shows its day (:func:`when_label`)."""
    return activity_region(rows, counts, head, n, width, wide=wide, host_style=host_style, now=now)[0]


def activity_region(rows: Sequence[Activity], counts: str, head: str, n: int, width: int, *, wide: bool,
                    host_style: Callable[[str], str] | None = None,
                    now: datetime | None = None) -> tuple[list[Text], list[JobKey | None]]:
    """:func:`activity_lines` and the job each line opens on a click (the count line: None)."""
    if n <= 0:
        return [], []
    first = Text(head, style=theme.label())
    first.append(counts or "nothing ended", style="" if counts else theme.dim())
    if any(r.old for r in rows[:n - 1]):
        tail = Text("· older dim", style=theme.dim())
        gap = width - first.cell_len - tail.cell_len
        if gap >= 2:
            first = first + Text(" " * gap) + tail
    out = [render.fit(first, width)]
    shown = rows[:n - 1]
    keys: list[JobKey | None] = [None, *(r.key for r in shown)]
    whens = [when_label(r.at, now) if now is not None else humanize.local_hm(r.at) for r in shown]
    when_w = max((cell_len(w) for w in whens), default=5)
    host_w = min(10, max((cell_len(r.host) for r in rows), default=4))
    name_w = 28 if wide else max(10, width - 16 - host_w - 12 - (when_w - 5))
    glyph_style = {GLYPHS["bad"]: theme.error(), GLYPHS["ok"]: theme.ok(), GLYPHS["end"]: theme.muted(),
                   GLYPHS["gpu_free"]: theme.ok()}
    for r, when in zip(shown, whens, strict=True):
        line = Text("  " if wide else " ")
        line.append_text(_cell(Text(when, style=theme.muted()), when_w))
        line.append("  " if wide else " ")
        line.append_text(_cell(Text(humanize.truncate(r.host, host_w), style=host_style(r.host) if host_style else ""),
                               host_w))
        line.append(" ")
        line.append(r.glyph + " ", style=glyph_style.get(r.glyph, theme.warn()))
        line.append_text(_cell(Text(humanize.truncate(r.name, name_w)), name_w))
        line.append(" ")
        line.append(r.summary, style=theme.error() if r.bad and r.glyph == GLYPHS["bad"] else theme.muted())
        if r.old:
            line.stylize(theme.dim())
        out.append(render.fit(line, width))
    return out, keys


def activity_expiry(jobs: Iterable[Job], history: Iterable[JobEvent], now: datetime,
                    window: float, extra: Iterable[Activity] = ()) -> datetime | None:
    """When :func:`activity_rows` / :func:`count_text` of a rolling ``window`` change on their own: the first own end,
    event or ``extra`` row (still inside the window) that leaves it. None: never."""
    span = timedelta(seconds=window)
    ats = [end + span for j in jobs if (end := ended_at(j)) is not None and end + span > now]
    ats += [e.at + span for e in history if e.at + span > now]
    ats += [a.at + span for a in extra if a.at + span > now]
    return min(ats, default=None)


def panel_split(rows: int, free_needed: int) -> int:
    """Rows for the free part: ``min(needed, ceil(rows / 2))``; the activity part takes the rest."""
    return min(free_needed, math.ceil(rows / 2))
