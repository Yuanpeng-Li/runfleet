"""GPU availability watches (§7.14): "tell me when N free GPUs of type T appear (on host H / partition P)" (pure).

A :class:`GpuWatch` is matched per node (a job needs its GPUs on one node), never against a fleet-wide sum, and only
with what the poller already has: a cluster's nodes from ``slurm.sinfo`` counted like the fleet panel
(:func:`~runfleet.util.slurmfmt.node_free`: an allocated node's idle GPU counts 0; only nodes of the listed
partitions) and a workstation's GPUs that are neither in use nor an old reading. Only fresh, successful readings
count, by the fleet panel's rule (:func:`~runfleet.ui.fleet_layout.free_source`: a ``host ?`` is not counted); a
workstation also needs its last ``nvidia.gpu`` sample to have succeeded.

:class:`Evaluator` keeps each watch's state. A host's reading counts once (by its sample time); a watch fires once the
condition held on :data:`CONSECUTIVE` readings of one host in a row, then is quiet for :data:`COOLDOWN_S` and re-arms
only after fresh readings show it no longer holds on any host. A host without a fresh reading breaks its run, and so
does an idle pause (polling stopped): :meth:`Evaluator.status` then says the watch is paused instead of staying
silent. GPUs that backfill has planned for a pending job count only for a watch that says ``planned``; MIG slices
only for a ``mig`` watch (a whole-GPU watch never counts slices, nor the other way round).

:func:`parse` reads the prompt leniently (``2 h200 k2``, ``4x a100 k2/gpu-preempt planned``, ``1 mig vega``);
:func:`load_watches` / :func:`with_watch` / :func:`without_watch` keep the watches in ``ui.json`` ``watches`` (an entry
this version cannot read is skipped, never an error, and kept in the file).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from runfleet.models import GPU, ConnState
from runfleet.ui import fleet_layout, render
from runfleet.util import humanize
from runfleet.util.slurmfmt import node_free

if TYPE_CHECKING:
    from runfleet.core.store import HostState

CONSECUTIVE = 2                  # readings in a row that must hold before a watch fires
COOLDOWN_S = 15 * 60.0           # after firing: quiet at least this long (and until it stopped holding)
MAX_WATCHES = 10
MAX_COUNT = 64
GPU_SECTION = "nvidia.gpu"       # a workstation's GPU sample (backends/common/nvidia.py; the UI imports no backend)
SINFO_SECTION = "slurm.sinfo"

_MIG_PROFILE = re.compile(r"(\d+g\.\d+gb)$")
_COUNT = re.compile(r"^(?:(\d+)[x×]?|[x×](\d+))$")
_COUNT_GLUED = re.compile(r"^(\d+)[x×](.+)$")
_MODEL = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_PLANNED = frozenset({"planned", "+planned", "backfill", "+backfill"})
_PARTITION_WORDS = frozenset({"partition", "part", "-p", "--partition"})
_FILLER = frozenset({"on", "in", "at", "of", "for", "free", "gpu", "gpus", "x", "×", "with", "when", "notify", "me",
                     "watch", "type", "host", "any", "anywhere", "whole", "node", "nodes", "the", "a", "an", "are",
                     "is", "appear", "appears", "available", "idle", "or", "more", "least", "+"})
_SETTINGS = {"p": "partition", "part": "partition", "partition": "partition", "host": "host", "h": "host",
             "on": "host", "type": "gpu", "gpu": "gpu", "gres": "gpu", "n": "count", "count": "count"}

Level = Literal["ok", "warn", "dim", "accent"]


# ---------------------------------------------------------------- the watch
def norm_gpu(token: str) -> str:
    """A typed GPU model as the fleet panel labels read: ``a100-80gb`` / ``A100_80G`` -> ``A100-80G``, ``h200`` ->
    ``H200``."""
    s = token.strip().upper().replace("_", "-")
    for prefix in ("NVIDIA-", "TESLA-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return re.sub(r"(\d+)GB(?=-|$)", r"\1G", s)


@dataclass(frozen=True, slots=True)
class GpuWatch:
    count: int = 1
    gpu: str | None = None        # model, normalised (:func:`norm_gpu`): H200, A100-80G; None = any model
    mig: str | None = None        # None = whole GPUs; "" = any MIG slice; "3g.40gb" = that profile
    host: str | None = None
    partition: str | None = None
    planned: bool = False         # count GPUs that backfill has planned for a pending job too

    def to_json(self) -> dict[str, Any]:
        return {"count": self.count, "gpu": self.gpu, "mig": self.mig, "host": self.host,
                "partition": self.partition, "planned": self.planned}

    def describe(self) -> str:
        """``2× H200 on k2/gpu-preempt``, ``1× GPU anywhere``, ``2× A100 MIG 3g.40gb on k2 · planned too``."""
        what = self.gpu or ("GPU" if self.mig is None else "")
        if self.mig is not None:
            what = " ".join(x for x in (what, "MIG", self.mig) if x)
        if self.host and self.partition:
            where = f" on {self.host}/{self.partition}"
        elif self.host:
            where = f" on {self.host}"
        elif self.partition:
            where = f" in partition {self.partition}"
        else:
            where = " anywhere"
        return f"{self.count}× {what}{where}" + (" · planned too" if self.planned else "")


def _spellings(gpu: str) -> tuple[str, ...]:
    """Label spellings a typed model accepts: itself, and the panel's label of it when it names no variant
    (``RTX3090`` -> ``3090``)."""
    out = [gpu]
    if "-" not in gpu:
        label = fleet_layout.type_label(gpu).upper()
        if label not in out:
            out.append(label)
    return tuple(out)


def gpu_matches(watch: GpuWatch, gtype: str) -> bool:
    """Whether a gres type (or a workstation's nvidia-smi name) is what ``watch`` waits for. A model matches its
    variants (``A100`` matches ``A100-80G`` and ``A100-40G``, never ``GH200``); MIG slices only match a MIG watch."""
    mig = humanize.is_mig(gtype)
    if mig != (watch.mig is not None):
        return False
    if mig and watch.mig:
        found = _MIG_PROFILE.search(gtype.lower())
        if found is None or found.group(1) != watch.mig:
            return False
    if watch.gpu is None:
        return True
    label = (humanize.gpu_model(gtype) if mig else fleet_layout.type_label(gtype)).upper()
    return any(label == s or label.startswith(s + "-") or (mig and s.startswith(label + "-"))
               for s in _spellings(watch.gpu))


# ---------------------------------------------------------------- readings (what the poller already has)
@dataclass(frozen=True, slots=True)
class NodeGPUs:
    node: str                                    # node name; a workstation's host name
    partitions: tuple[str, ...]                  # the listed partitions the node is in (workstation: ())
    types: tuple[tuple[str, int, int], ...]      # (gres type / nvidia-smi name, free, of them planned by backfill)


@dataclass(frozen=True, slots=True)
class Reading:
    host: str
    at: datetime | None                          # sample time (sinfo / nvidia-smi); None: nothing fresh to count
    why: str = ""                                # at None: why ("no fresh reading", "disabled", "no GPUs")
    nodes: tuple[NodeGPUs, ...] = ()
    partitions: tuple[str, ...] = ()             # the host's listed partitions (from its last snapshot)
    cluster: bool = False


def _cluster_nodes(snap: Any) -> tuple[NodeGPUs, ...]:
    listed = [p.name for p in snap.partitions]
    out: list[NodeGPUs] = []
    for node in snap.nodes:
        inside = tuple(p for p in listed if p in node.partitions)
        types = tuple((gtype, free, planned) for gtype, free, _total, planned in node_free(node))
        if inside and types:
            out.append(NodeGPUs(node.name, inside, types))
    return tuple(out)


def _ws_node(name: str, state: HostState, gpus: Sequence[GPU], old: Callable[[HostState, GPU], bool] | None
             ) -> NodeGPUs:
    free: dict[str, int] = {}
    for g in gpus:
        n = free.setdefault(g.name, 0)
        if not render.gpu_in_use(g) and not (old is not None and old(state, g)):
            free[g.name] = n + 1
    return NodeGPUs(name, (), tuple((t, n, 0) for t, n in free.items()))


def readings(hosts: Mapping[str, HostState], now: datetime, *, old: Callable[[HostState, GPU], bool] | None = None,
             rate: float = 1.0, memo: dict[str, tuple[Any, ...]] | None = None) -> dict[str, Reading]:
    """Every host's current reading. ``old``: a workstation GPU reading that missed its refresh (never free);
    ``rate``: the +/- poll rate (how old a sinfo sample may be); ``memo``: node scans are kept per host until its
    sample time or its nodes / partitions / GPUs change."""
    from runfleet.ui.widgets.host_card import is_cluster

    out: dict[str, Reading] = {}
    for name, state in hosts.items():
        snap = state.snapshot
        parts = tuple(p.name for p in snap.partitions) if snap is not None else ()
        cluster = is_cluster(state)
        source, _age = fleet_layout.free_source(state, now, rate)
        if source == "skip":
            why = "disabled" if state.status.state is ConnState.DISABLED else "no GPU partition in scope"
            out[name] = Reading(name, None, why, partitions=parts, cluster=cluster)
            continue
        if source == "unknown" or snap is None:
            out[name] = Reading(name, None, "no fresh reading", partitions=parts, cluster=cluster)
            continue
        section = state.status.sections.get(SINFO_SECTION if source == "cluster" else GPU_SECTION)
        if source == "ws" and not snap.gpus:
            out[name] = Reading(name, None, "no GPUs", partitions=parts)
            continue
        if section is None or not section.ok or section.last_ok is None:
            out[name] = Reading(name, None, "no fresh GPU reading", partitions=parts, cluster=cluster)
            continue
        at = section.last_ok
        key = (at, snap.nodes, snap.partitions, snap.gpus)
        held = memo.get(name) if memo is not None else None
        if held is not None and held[0] == at and all(a is b for a, b in zip(held[1:-1], key[1:], strict=True)):
            nodes = held[-1]
        else:
            nodes = _cluster_nodes(snap) if source == "cluster" else (_ws_node(name, state, snap.gpus, old),)
            if memo is not None:
                memo[name] = (*key, nodes)
        out[name] = Reading(name, at, nodes=nodes, partitions=parts, cluster=source == "cluster")
    if memo is not None:
        for name in [n for n in memo if n not in hosts]:
            del memo[name]
    return out


# ---------------------------------------------------------------- matching, per node
@dataclass(frozen=True, slots=True)
class Match:
    host: str
    node: str
    free: int                    # GPUs the watch counts on this node
    planned: int = 0             # free but planned by backfill and not counted (the watch did not say planned)
    cluster: bool = True

    def text(self) -> str:
        """``k2 n-38-01 3 free`` / ``labstation 2 free`` (``+2 planned`` when planned GPUs were left out)."""
        where = f"{self.host} {self.node}" if self.cluster else self.host
        return f"{where} {self.free} free" + (f" (+{self.planned} planned)" if self.planned else "")


def in_scope(watch: GpuWatch, rds: Mapping[str, Reading]) -> list[str]:
    """The hosts a watch looks at: its host (if still in the fleet), else every host; with a partition, only
    clusters that list it."""
    names = [watch.host] if watch.host else list(rds)
    return [n for n in names if n in rds
            and (watch.partition is None or (rds[n].cluster and watch.partition in rds[n].partitions))]


def node_count(watch: GpuWatch, node: NodeGPUs) -> tuple[int, int]:
    """(GPUs the watch counts on one node, planned ones it leaves out). A model or a MIG profile is one request (the
    best matching type of the node); "any GPU" adds every whole-GPU type of the node, as an untyped request may."""
    counted: list[int] = []
    left_out: list[int] = []
    for gtype, free, planned in node.types:
        if gpu_matches(watch, gtype):
            counted.append(free if watch.planned else free - planned)
            left_out.append(0 if watch.planned else planned)
    if not counted:
        return 0, 0
    if watch.gpu is None and watch.mig is None:
        return sum(counted), sum(left_out)
    best = max(range(len(counted)), key=lambda i: (counted[i], left_out[i]))
    return counted[best], left_out[best]


def candidates(watch: GpuWatch, reading: Reading) -> list[Match]:
    """Nodes of one host with any GPU the watch could count (or planned ones it leaves out), most free first."""
    out: list[Match] = []
    for node in reading.nodes:
        if watch.partition is not None and watch.partition not in node.partitions:
            continue
        free, planned = node_count(watch, node)
        if free > 0 or planned > 0:
            out.append(Match(reading.host, node.node, free, planned, reading.cluster))
    out.sort(key=lambda m: (-m.free, -m.planned, m.node))
    return out


def matches(watch: GpuWatch, reading: Reading) -> list[Match]:
    """The nodes where the watch holds now (``count`` or more counted free GPUs); none without a fresh reading."""
    if reading.at is None:
        return []
    return [m for m in candidates(watch, reading) if m.free >= watch.count]


def summary(found: Sequence[Match], limit: int = 3) -> str:
    """``k2 n-38-01 4 free · vega n-38-06 2 free · +3 more nodes``."""
    text = " · ".join(m.text() for m in found[:limit])
    return text + (f" · +{len(found) - limit} more nodes" if len(found) > limit else "")


# ---------------------------------------------------------------- evaluation
@dataclass
class WatchState:
    streak: dict[str, int] = field(default_factory=dict)            # host -> fresh readings in a row that held
    seen: dict[str, datetime] = field(default_factory=dict)         # host -> sample time already counted
    holding: dict[str, tuple[Match, ...]] = field(default_factory=dict)   # hosts whose last fresh reading held
    armed: bool = True
    fired_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Fired:
    watch: GpuWatch
    at: datetime
    found: tuple[Match, ...]


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """A watch that fired (``gpu_free``) or the watches pausing with polling (``paused``): the Notifications
    history and, for ``gpu_free``, the fleet panel's *what changed*."""
    kind: Literal["gpu_free", "paused"]
    at: datetime
    watch: str                   # GpuWatch.describe(); "" for a pause
    host: str                    # the first host that matched; "" for a pause
    summary: str

    @property
    def bad(self) -> bool:
        return self.kind == "paused"

    def text(self) -> str:
        if self.kind == "paused":
            return f"GPU watches paused: {self.summary}"
        return f"free GPUs: {self.watch} — {self.summary}"


def fired_event(fired: Fired) -> WatchEvent:
    return WatchEvent("gpu_free", fired.at, fired.watch.describe(), fired.found[0].host if fired.found else "",
                      summary(fired.found))


class Evaluator:
    """Each watch's run of readings, cooldown and arming (see the module docstring)."""

    def __init__(self, consecutive: int = CONSECUTIVE, cooldown_s: float = COOLDOWN_S) -> None:
        self.consecutive = consecutive
        self.cooldown = timedelta(seconds=cooldown_s)
        self.states: dict[GpuWatch, WatchState] = {}

    def update(self, watches: Iterable[GpuWatch], rds: Mapping[str, Reading], now: datetime, *,
               paused: bool = False) -> list[Fired]:
        """Count every new reading for every watch; the watches that fire now."""
        watches = list(dict.fromkeys(watches))
        for gone in [w for w in self.states if w not in watches]:
            del self.states[gone]
        out: list[Fired] = []
        for w in watches:
            fired = self._step(w, self.states.setdefault(w, WatchState()), rds, now, paused)
            if fired is not None:
                out.append(fired)
        return out

    def _step(self, watch: GpuWatch, st: WatchState, rds: Mapping[str, Reading], now: datetime,
              paused: bool) -> Fired | None:
        scope = in_scope(watch, rds)
        for table in (st.streak, st.seen, st.holding):
            for host in [h for h in table if h not in scope]:
                del table[host]                       # removed from the fleet, or no longer lists the partition
        if paused:                                    # a reading before and one after a pause are not in a row
            st.streak.clear()
            return None
        for host in scope:
            r = rds[host]
            if r.at is None:
                st.streak[host] = 0                   # no fresh reading breaks the run; whether it holds is unknown
                continue
            if st.seen.get(host) == r.at:
                continue                              # this reading was counted already
            st.seen[host] = r.at
            found = matches(watch, r)
            if found:
                st.streak[host] = st.streak.get(host, 0) + 1
                st.holding[host] = tuple(found)
            else:
                st.streak[host] = 0
                st.holding.pop(host, None)
        if not st.armed and not st.holding:
            st.armed = True                           # it stopped holding everywhere: the next run may fire again
        if not st.armed or (st.fired_at is not None and now - st.fired_at < self.cooldown):
            return None
        ready = [h for h in scope if st.streak.get(h, 0) >= self.consecutive]
        if not ready:
            return None
        st.armed = False
        st.fired_at = now
        found = sorted((m for h in ready for m in st.holding[h]), key=lambda m: (-m.free, m.host, m.node))
        return Fired(watch, now, tuple(found))

    def status(self, watch: GpuWatch, rds: Mapping[str, Reading], now: datetime, *,
               paused: bool = False) -> tuple[str, Level]:
        """One line on what the watch is doing now, for the Watches list."""
        if paused:
            return "paused: polling is idle-paused (any key resumes it), nothing is counted", "warn"
        scope = in_scope(watch, rds)
        if not scope:
            return no_scope(watch, rds), "warn"
        st = self.states.get(watch) or WatchState()
        unknown = [h for h in scope if rds[h].at is None]
        tail = f" · not counted: {' '.join(f'{h} ?' for h in unknown)}" if unknown else ""
        if st.fired_at is not None and not st.armed:
            return f"fired {humanize.local_hm(st.fired_at)} · re-arms once it no longer holds{tail}", "accent"
        if st.fired_at is not None and now - st.fired_at < self.cooldown:
            until = humanize.local_hm(st.fired_at + self.cooldown)
            return f"fired {humanize.local_hm(st.fired_at)} · quiet until {until}{tail}", "dim"
        runs = {h: n for h in scope if (n := st.streak.get(h, 0)) > 0}
        if runs:
            best = max(runs.values())
            return (f"holds on {', '.join(runs)} ({min(best, self.consecutive)} of {self.consecutive} readings in "
                    f"a row){tail}", "ok")
        if len(unknown) == len(scope):
            return f"no fresh reading to count{tail}", "warn"
        return f"waiting: {now_text(watch, rds, short=True)}{tail}", "dim"


def no_scope(watch: GpuWatch, rds: Mapping[str, Reading]) -> str:
    if watch.host and watch.host not in rds:
        return f"{watch.host} is not in the fleet any more"
    if watch.partition is not None:
        if watch.host:
            return f"{watch.host} lists no partition {watch.partition} now"
        return f"no host lists partition {watch.partition} now"
    return "no host to look at"


def now_text(watch: GpuWatch, rds: Mapping[str, Reading], *, short: bool = False) -> str:
    """What the watch sees right now: ``matches now: k2 n-38-01 3 free`` / ``best now: k2 n-38-01 1 free`` /
    ``none free now``, plus planned GPUs it leaves out."""
    scope = in_scope(watch, rds)
    if not scope:
        return no_scope(watch, rds)
    found: list[Match] = []
    near: list[Match] = []
    for host in scope:
        if rds[host].at is not None:
            near += candidates(watch, rds[host])
            found += matches(watch, rds[host])
    near.sort(key=lambda m: (-m.free, -m.planned, m.host, m.node))
    if found:
        found.sort(key=lambda m: (-m.free, m.host, m.node))
        text = f"matches now: {summary(found)}"
    elif near and near[0].free:
        text = f"best now: {near[0].text()}"
    else:
        text = "none free now"
    if not short and not watch.planned:
        planned = [m for m in near if m.planned and m.free < watch.count <= m.free + m.planned]
        if planned:
            text += (f" · {summary(planned, 2)} only with planned GPUs (backfill keeps them for a pending job; "
                     "add “planned” to count them)")
    return text


# ---------------------------------------------------------------- the prompt
@dataclass(frozen=True, slots=True)
class Fleet:
    """What the prompt can name: hosts, their listed partitions, GPU types (gres types and nvidia-smi names)."""
    hosts: tuple[str, ...] = ()
    partitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    types: tuple[str, ...] = ()
    clusters: frozenset[str] = frozenset()


def fleet_of(hosts: Mapping[str, HostState]) -> Fleet:
    from runfleet.ui.widgets.host_card import is_cluster

    parts: dict[str, tuple[str, ...]] = {}
    types: dict[str, None] = {}
    for name, state in hosts.items():
        snap = state.snapshot
        if snap is None:
            continue
        parts[name] = tuple(p.name for p in snap.partitions)
        for node in snap.nodes:
            for gtype, *_rest in node_free(node):
                types.setdefault(gtype)
        for g in snap.gpus:
            types.setdefault(g.name)
    return Fleet(tuple(hosts), parts, tuple(types), frozenset(n for n, s in hosts.items() if is_cluster(s)))


@dataclass(frozen=True, slots=True)
class Parsed:
    watch: GpuWatch | None
    problem: str | None = None
    notes: tuple[str, ...] = ()


def _known_type(gpu: str, mig: str | None, fleet: Fleet) -> bool:
    probes = [GpuWatch(gpu=gpu, mig=mig)] if mig is not None else [GpuWatch(gpu=gpu), GpuWatch(gpu=gpu, mig="")]
    return any(gpu_matches(p, t) for p in probes for t in fleet.types)


def parse(text: str, fleet: Fleet) -> Parsed:
    """The prompt's text -> a watch, leniently: a count (``2``, ``2x``, ``x2``, glued ``2xh200``), a GPU model, ``mig``
    or a MIG profile (``3g.40gb``, ``a100-3g.40gb``), a host, a partition (``k2/gpu-preempt``, ``partition gpu``,
    ``p=gpu`` or a bare name a host lists), ``planned``; filler words (``free``, ``gpus``, ``on`` …) are skipped. A
    problem means nothing can be saved; notes say how the text was read."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        return Parsed(None, "type what to wait for, e.g. 2 h200 k2")
    hosts = {h.lower(): h for h in fleet.hosts}
    every_part = {p.lower(): p for ps in fleet.partitions.values() for p in ps}
    got: dict[str, Any] = {}
    notes: list[str] = []

    def put(key: str, value: Any, what: str) -> str | None:
        if key in got and got[key] != value:
            return f"one {what} per watch ({got[key]} or {value}?)"
        got[key] = value
        return None

    def put_type(token: str) -> str | None:
        low = token.lower()
        if low == "mig":
            if got.get("mig"):
                return None                           # a profile names a MIG watch already
            return put("mig", "", "MIG profile")
        prof = _MIG_PROFILE.search(low)
        if prof is not None:
            if got.get("mig") == "":
                del got["mig"]
            problem = put("mig", prof.group(1), "MIG profile")
            model = low[:prof.start()].rstrip("-_")
            return problem or (put("gpu", norm_gpu(model), "GPU type") if model else None)
        return put("gpu", norm_gpu(token), "GPU type")

    def put_host(token: str) -> str | None:
        name = hosts.get(token.lower())
        if name is None:
            return f"no host “{token}” in the fleet"
        return put("host", name, "host")

    queue = list(tokens)
    while queue:
        token = queue.pop(0)
        low = token.lower()
        problem: str | None = None
        if "=" in token and not token.startswith("="):
            key, _, value = token.partition("=")
            setting = _SETTINGS.get(key.lower().lstrip("-"))
            if setting is None or not value:
                return Parsed(None, f"cannot read “{token}”: try host=k2, partition=gpu, type=h200 or a count",
                              tuple(notes))
            if setting == "count":
                if not value.isdigit():
                    return Parsed(None, f"“{value}” is not a count", tuple(notes))
                problem = put("count", int(value), "count")
            elif setting == "host":
                problem = put_host(value)
            elif setting == "partition":
                problem = put("partition", every_part.get(value.lower(), value), "partition")
            else:
                problem = put_type(value)
        elif low in _PARTITION_WORDS:
            if not queue:
                return Parsed(None, "partition … which one? e.g. partition gpu-preempt", tuple(notes))
            value = queue.pop(0)
            problem = put("partition", every_part.get(value.lower(), value), "partition")
        elif (m := _COUNT.match(low)) is not None and not (low.isdigit() and ("count" in got or int(low) > MAX_COUNT)
                                                            and _known_type(low, None, fleet)):
            problem = put("count", int(m.group(1) or m.group(2)), "count")      # (a bare 3090 after a count: a model)
        elif (m := _COUNT_GLUED.match(low)) is not None and not low[len(m.group(1)) + 1:].isdigit():
            problem = put("count", int(m.group(1)), "count")
            queue.insert(0, token[len(m.group(1)) + 1:])
        elif low in _PLANNED:
            got["planned"] = True
        elif low in _FILLER:
            if low in every_part:
                notes.append(f"“{token}” read as a word; write partition {every_part[low]} (or host/{every_part[low]})"
                             " to watch that partition")
        elif low.startswith("@") and len(low) > 1:
            problem = put_host(token[1:])
        elif low == "mig" or _MIG_PROFILE.search(low):
            problem = put_type(token)
        elif "/" in token:
            named, _, slash_part = token.partition("/")
            problem = put_host(named) if named else None
            if problem is None and slash_part:
                problem = put("partition", every_part.get(slash_part.lower(), slash_part), "partition")
        elif low in hosts:
            problem = put_host(token)
        elif _known_type(norm_gpu(token), None, fleet):
            problem = put_type(token)
        elif low in every_part:
            problem = put("partition", every_part[low], "partition")
        elif any(c.isdigit() for c in low) and _MODEL.fullmatch(low):
            problem = put_type(token)
            notes.append(f"no {norm_gpu(token)} GPUs seen in the fleet yet")
        else:
            problem = f"“{token}” is not a host, partition or GPU type seen in the fleet"
        if problem is not None:
            return Parsed(None, problem, tuple(notes))
    count = got.get("count", 1)
    if not 1 <= count <= MAX_COUNT:
        return Parsed(None, f"a count from 1 to {MAX_COUNT}", tuple(notes))
    host = got.get("host")
    part = got.get("partition")
    if part is not None:
        if host is not None:
            listed = fleet.partitions.get(host, ())
            if host not in fleet.clusters and not listed:
                return Parsed(None, f"{host} has no partitions (a workstation)", tuple(notes))
            canon = {p.lower(): p for p in listed}
            if part.lower() not in canon:
                shown = ", ".join(listed[:6]) + (" …" if len(listed) > 6 else "")
                return Parsed(None, f"{host} lists no partition “{part}”" + (f" (it lists {shown})" if listed else ""),
                              tuple(notes))
            part = canon[part.lower()]
        elif part.lower() not in every_part:
            return Parsed(None, f"no host lists a partition “{part}”", tuple(notes))
    gpu, mig = got.get("gpu"), got.get("mig")
    if gpu is not None and not any(n.startswith("no ") for n in notes) and not _known_type(gpu, mig, fleet):
        notes.append(f"no {gpu}{' MIG' if mig is not None else ''} GPUs seen in the fleet yet")
    if mig is not None and host is not None and host not in fleet.clusters:
        notes.append("a workstation's nvidia-smi lists whole GPUs only: this watch can never match there")
    if got.get("planned"):
        notes.append("counts GPUs backfill has planned for a pending job too (it may take them any moment)")
    return Parsed(GpuWatch(count, gpu, mig, host, part, bool(got.get("planned"))), None, tuple(notes))


def add_problem(watches: Sequence[GpuWatch], watch: GpuWatch) -> str | None:
    """Why ``watch`` cannot be added: a duplicate, or :data:`MAX_WATCHES` reached."""
    if watch in watches:
        return "you watch exactly that already"
    if len(watches) >= MAX_WATCHES:
        return f"at most {MAX_WATCHES} watches: remove one first (palette › Watches)"
    return None


# ---------------------------------------------------------------- ui.json ``watches``
def parse_json(raw: Any) -> GpuWatch | None:
    """One stored entry -> GpuWatch, None when any present field is unusable (a missing one takes its default)."""
    if not isinstance(raw, Mapping):
        return None
    count = raw.get("count", 1)
    gpu = raw.get("gpu")
    mig = raw.get("mig")
    host = raw.get("host")
    part = raw.get("partition")
    planned = raw.get("planned", False)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
        return None
    for value in (gpu, host, part):
        if value is not None and (not isinstance(value, str) or not value or len(value) > 64
                                  or any(c.isspace() or not c.isprintable() for c in value)):
            return None
    if mig is not None and (not isinstance(mig, str) or (mig and not _MIG_PROFILE.fullmatch(mig))):
        return None
    if not isinstance(planned, bool):
        return None
    return GpuWatch(count, norm_gpu(gpu) if gpu else None, mig, host, part, planned)


def load_watches(ui_state: Mapping[str, Any]) -> list[GpuWatch]:
    """The usable watches of ui.json, in order, without duplicates."""
    raw = ui_state.get("watches")
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(w for r in raw if (w := parse_json(r)) is not None))


def with_watch(ui_state: Mapping[str, Any], watch: GpuWatch) -> list[Any]:
    """The ``watches`` list with ``watch`` appended (entries this version cannot read are kept)."""
    raw = ui_state.get("watches")
    kept = list(raw) if isinstance(raw, list) else []
    if any(parse_json(r) == watch for r in kept):
        return kept
    return [*kept, watch.to_json()]


def without_watch(ui_state: Mapping[str, Any], watch: GpuWatch) -> list[Any]:
    raw = ui_state.get("watches")
    return [r for r in raw if parse_json(r) != watch] if isinstance(raw, list) else []
