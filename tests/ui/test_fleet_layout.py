"""Roomy Fleet layout (ui/fleet_layout.py, DESIGN.md §7.4): the plan, shapes, ranking, row templates, owner column,
free GPUs by type and the activity rows; plus the T0 identity of the cards."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import pytest
from rich.cells import cell_len

from runfleet.core.events import JobEvent
from runfleet.models import (GPU, ConnState, ErrorKind, GPUProcess, GresCount, HostError, JobKey, JobState, Node, PartitionSummary,
                             SectionStatus)
from runfleet.ui import fleet_layout as fl
from runfleet.ui.fleet_layout import NO_DATA, HighWater, Shape, plan
from runfleet.ui.widgets.host_card import card_body, card_lines
from tests.ui.fakes import T0, build_world, slurm_job, task_job

SLURM = Shape(6, 5, fl.SLURM_LIST_CAP, "slurm", 1)
WS4 = Shape(7, 5, fl.WS_LIST_CAP, "ws", 4)
DEMO = [SLURM, SLURM, SLURM, WS4, NO_DATA]           # k2 / vega / lyra / labstation / lynx (no cache), seed 7
DAY = 86400.0


def view_h(term_h: int) -> int:
    """FleetView height: the terminal minus header, tabs and hint bar."""
    return term_h - 3


# ---------------------------------------------------------------- plan
def test_plan_sizes() -> None:
    assert plan(80, view_h(24), DEMO).mode == "table"
    small = plan(100, view_h(30), DEMO)
    assert (small.mode, small.tier, small.inner) == ("cards", 0, 45)
    p35 = plan(120, view_h(35), DEMO)
    assert (p35.tier, p35.heights, p35.wide) == (1, (10, 10, 6), False)
    assert p35.panel == fl.Panel(1, 6)
    p40 = plan(120, view_h(40), DEMO)
    assert (p40.tier, p40.heights, p40.inner) == (1, (11, 12, 8), 55)
    assert p40.panel == fl.Panel(1, 8)
    p172 = plan(172, view_h(62), DEMO)
    assert (p172.tier, p172.heights, p172.inner, p172.wide) == (1, (17, 17, 19), 81, True)
    assert p172.panel == fl.Panel(1, 19)
    assert sum(h + 2 for h in p172.heights) <= view_h(62)          # no scrollbar


def test_plan_gate() -> None:
    assert plan(172, fl.ROOMY_MIN_H - 1, [SLURM, SLURM]).tier == 0          # too short
    assert plan(172, view_h(62), DEMO, enabled=False).tier == 0             # ui.json fleet_layout = "compact"
    base = 2 * (SLURM.t0 + 2)
    assert plan(120, base + fl.LIST_MIN - 1, [SLURM, SLURM]).tier == 0     # fewer than 3 spare rows per grid row
    assert plan(120, max(fl.ROOMY_MIN_H, base + fl.LIST_MIN), [SLURM, SLURM]).tier == 1


def test_full_width_panel_needs_ten_spare_rows() -> None:
    four = [SLURM] * 4
    full = 2 * (SLURM.t1 + SLURM.cap + 2)             # both rows grown to their cap
    assert plan(120, full + 9, four).panel is None
    p = plan(120, full + 10, four)
    assert p.heights == (17, 17) and p.panel == fl.Panel(2, 8, below=True)


def test_no_data_row_does_not_grow() -> None:
    p = plan(120, 60, [SLURM, SLURM, NO_DATA, NO_DATA])
    assert p.tier == 1 and p.heights[1] == NO_DATA.t0 and p.heights[0] == SLURM.t1 + SLURM.cap


def test_list_rows_at_least_three() -> None:
    p = plan(120, view_h(35), DEMO)
    for i, s in enumerate(DEMO):
        if s.cap:
            assert p.heights[i // p.cols] - s.t1 >= fl.LIST_MIN


# ---------------------------------------------------------------- shapes
def _states():
    _cfg, states = build_world()
    return {s.config.name: s for s in states}


def test_shapes_of_the_fixture_world() -> None:
    st = _states()
    hw: dict[str, HighWater] = {}
    shapes = {n: fl.shape_of(s, 55, hw.setdefault(n, HighWater())) for n, s in st.items()}
    assert shapes["k2"] == Shape(6, 5, 12, "slurm", 1)
    assert shapes["labstation"] == Shape(7, 5, 8, "ws", 4)
    assert shapes["lynx"] == Shape(5, 3, 8, "ws", 2)


def test_shape_high_water() -> None:
    st = _states()
    hw = HighWater()
    ws = st["labstation"]
    snap = ws.snapshot
    assert snap is not None
    assert fl.shape_of(ws, 55, hw).fixed == 4
    ws.snapshot = dataclasses.replace(snap, gpus=snap.gpus[:2])           # a GPU stops answering
    assert fl.shape_of(ws, 55, hw).fixed == 4

    k2 = st["vega"]
    hw2 = HighWater()
    ksnap = k2.snapshot
    assert ksnap is not None
    first = fl.shape_of(k2, 55, hw2)
    k2.snapshot = dataclasses.replace(ksnap, partitions=ksnap.partitions[:1])   # the free block shrinks
    assert fl.shape_of(k2, 55, hw2) == first
    extra = tuple(PartitionSummary(f"long-partition-name-{i}", 4, 400, 0, 0, {"h100": (0, 400)}, {})
                  for i in range(3))
    k2.snapshot = dataclasses.replace(ksnap, partitions=ksnap.partitions + extra)   # new partitions: grows once
    grown = fl.shape_of(k2, 55, hw2)
    assert grown.fixed > first.fixed
    k2.snapshot = ksnap
    assert fl.shape_of(k2, 55, hw2) == grown


def test_free_block_depends_on_names_and_totals_only() -> None:
    st = _states()
    k2 = st["k2"]
    snap = k2.snapshot
    assert snap is not None
    a = fl.shape_of(k2, 55, HighWater())
    busy = tuple(dataclasses.replace(p, gpus_free=0, gpus_free_planned=0) for p in snap.partitions)
    k2.snapshot = dataclasses.replace(snap, partitions=busy)
    assert fl.shape_of(k2, 55, HighWater()) == a


# ---------------------------------------------------------------- ranking
def _ranks(jobs, now=T0, window=DAY):
    return [(r.rank, r.job.key.id) for r in fl.rank_slurm(jobs, now, window)]


def test_rank_slurm_boundaries() -> None:
    jobs = [
        slurm_job("k2", "101", "fresh", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=3600, exit_code=1),
        slurm_job("k2", "102", "edge", JobState.TIMEOUT, elapsed=timedelta(hours=1), ended_ago=fl.FRESH_FAIL_S),
        slurm_job("k2", "103", "older", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=fl.FRESH_FAIL_S + 1),
        slurm_job("k2", "104", "too-old", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=DAY + 60),
        slurm_job("k2", "105", "stalled", JobState.RUNNING, elapsed=timedelta(hours=1), hints=("log idle 51m",)),
        slurm_job("k2", "106", "short", JobState.RUNNING, elapsed=timedelta(hours=1, minutes=55),
                  limit=timedelta(hours=2)),
        slurm_job("k2", "107", "never", JobState.PENDING, reason="DependencyNeverSatisfied"),
        slurm_job("k2", "108", "held", JobState.PENDING, reason="JobHeldUser", held=True),
        slurm_job("k2", "109", "long", JobState.RUNNING, elapsed=timedelta(hours=1), limit=timedelta(days=2)),
        slurm_job("k2", "110", "soon", JobState.RUNNING, elapsed=timedelta(hours=1), limit=timedelta(days=1)),
        slurm_job("k2", "111", "nolimit", JobState.RUNNING, elapsed=timedelta(hours=1), limit=None),
        slurm_job("k2", "112", "queued", JobState.PENDING, reason="Priority"),
        slurm_job("k2", "113", "done", JobState.COMPLETED, elapsed=timedelta(hours=1), ended_ago=60, exit_code=0),
        slurm_job("k2", "114", "cancelled", JobState.CANCELLED, elapsed=timedelta(hours=1), ended_ago=60),
        slurm_job("k2", "115", "theirs", JobState.RUNNING, elapsed=timedelta(hours=1), user="bob"),
    ]
    est = dataclasses.replace(slurm_job("k2", "116", "estimated", JobState.PENDING, reason="Resources"),
                              start_time=T0 + timedelta(minutes=30), est_start=True)
    assert _ranks([*jobs, est]) == [
        (0, "101"), (0, "102"), (1, "105"), (1, "106"), (2, "107"), (2, "108"), (3, "103"),
        (4, "110"), (4, "109"), (4, "111"), (5, "116"), (5, "112"),
    ]


def test_rank_ties_by_id() -> None:
    jobs = [slurm_job("k2", jid, f"j{jid}", JobState.RUNNING, elapsed=timedelta(hours=1)) for jid in ("30", "4", "200")]
    assert [k for _r, k in _ranks(jobs)] == ["4", "30", "200"]


def test_rank_ws() -> None:
    def task(pid: int, *, mine: bool = True, gpus: tuple[int, ...] = (0,), state: JobState = JobState.RUNNING,
             hints: tuple[str, ...] = (), ended: float | None = None):
        j = task_job("ws", pid, f"t{pid}", elapsed=timedelta(hours=1), gpus=gpus, gpu_model="NVIDIA RTX A6000",
                     util=50, mem_mib=1024, mem_total_mib=49140, pids=(pid,), mine=mine, state=state,
                     user="alice07" if mine else "carol")
        if ended is not None:
            j = dataclasses.replace(j, end_time=T0 - timedelta(seconds=ended))
        return dataclasses.replace(j, hints=hints)

    jobs = [
        task(1, state=JobState.FAILED, ended=600),
        task(2, state=JobState.EXITED, ended=600),                                   # status unknown: not a failure
        task(3, state=JobState.EXITED, ended=300, hints=("log: Traceback (most recent call last)",)),
        task(4, state=JobState.FAILED, ended=fl.FRESH_FAIL_S + 60),
        task(5, hints=("GPU idle 20m",)),
        task(6, gpus=(0, 1)),
        task(7),
        task(8, mine=False),
    ]
    got = [(r.rank, r.job.job_id_raw, r.group) for r in fl.rank_ws(jobs, T0, DAY)]
    assert got == [(0, "3", "failed"), (0, "1", "failed"), (1, "5", "running"), (2, "6", "running"),
                   (2, "7", "running"), (3, "8", "others"), (4, "4", "failed")]


def test_select_groups_and_array_collapse() -> None:
    fails = [slurm_job("k2", f"5{i}", f"f{i}", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=600 * (i + 1),
                       exit_code=1) for i in range(2)]
    arr = [dataclasses.replace(slurm_job("k2", f"900_{t}", "orb-sweep", JobState.RUNNING, elapsed=timedelta(hours=1)),
                               array_job_id="900", array_task=str(t)) for t in (3, 4, 5)]
    pend = [slurm_job("k2", "77", "queued", JobState.PENDING, reason="Priority")]
    ranked = fl.rank_slurm([*pend, *arr, *fails], T0, DAY)
    shown, hidden = fl.select(ranked, 4)                     # 6 candidates, the array folds into one row: fits
    assert not hidden
    assert [r.group for r in shown] == ["failed", "failed", "running", "pending"]
    assert shown[2].count == 3 and fl.array_suffix(shown[2].job, shown[2].members) == " [3-5] ×3"
    lines = fl.list_lines(ranked, 4, T0, inner=60, wide=False)
    assert "orb-sweep [3-5] ×3" in lines[2].plain
    shown, hidden = fl.select(ranked, 3)                     # still too many: 2 rows + the overflow line
    assert len(shown) == 2 and sum(r.count for r in hidden) == 4
    over = fl.overflow_line(hidden, T0, 60).plain
    assert over.startswith("+4 more · R 3, next ends in ") and "PD 1 · tab: Jobs" in over
    assert len(fl.list_lines(ranked, 3, T0, inner=60, wide=False)) == 3


# ---------------------------------------------------------------- row templates
@pytest.mark.parametrize("inner", [52, 55, 71, 72, 81, 120])
def test_rows_are_exactly_inner_wide(inner: int) -> None:
    jobs = [
        slurm_job("k2", "1", "a-very-long-job-name-that-will-not-fit-anywhere--s43--a02", JobState.RUNNING,
                  elapsed=timedelta(hours=3), util=93, hints=("log idle 51m",)),
        slurm_job("k2", "2", "ok", JobState.RUNNING, elapsed=timedelta(hours=3), limit=timedelta(hours=3, minutes=5)),
        slurm_job("k2", "3", "f", JobState.OUT_OF_MEMORY, elapsed=timedelta(hours=1), ended_ago=30),
        slurm_job("k2", "4", "p", JobState.PENDING, reason="DependencyNeverSatisfied"),
    ]
    for line in fl.list_lines(fl.rank_slurm(jobs, T0, DAY), 4, T0, inner=inner, wide=inner >= fl.WIDE_INNER):
        assert line.cell_len == inner
    wide = fl.list_lines(fl.rank_slurm(jobs, T0, DAY), 4, T0, inner=81, wide=True)
    assert wide[0].plain.startswith("failed    OOM") and "<1m ago" in wide[0].plain
    assert "never: dependency failed" in wide[-1].plain and "no est." in wide[-1].plain


def test_time_cells_move_per_minute() -> None:
    assert fl.span_min(59) == "<1m" and fl.span_min(61) == "1m" and fl.span_min(3 * 3600 + 7 * 60 + 59) == "3h07m"


# ---------------------------------------------------------------- owner column
def _owner(procs: tuple[GPUProcess, ...] = (), *, util: float = 0, mem: float = 0, old: bool = False,
           wide: bool = True) -> str:
    st = _states()["labstation"]
    snap = st.snapshot
    assert snap is not None
    g = GPU("labstation", 0, "GPU-x", "NVIDIA RTX A6000", util, mem, 49140, None, 30, None, None, None, None,
            "nvidia-smi", procs, T0)
    return fl.owner_text(snap, g, old=old, wide=wide).plain.rstrip()


def test_owner_column() -> None:
    st = _states()["labstation"]
    snap = st.snapshot
    assert snap is not None
    mine = next(j for j in snap.jobs.values() if j.is_mine)
    other = next(j for j in snap.jobs.values() if not j.is_mine)
    p_mine = GPUProcess(1, "GPU-x", 0, 1024, mine.user, "python -m orb.train", 60, 90.0, mine.key)
    p_other = GPUProcess(2, "GPU-x", 0, 1024, other.user, "python eval.py", 60, 90.0, other.key)
    p_loose = GPUProcess(3, "GPU-x", 0, 1024, "dave", "/opt/venv/bin/python sample.py --x", 60, 9.0, None)
    assert _owner() == "free"
    assert _owner((p_mine,)) == "mine orb.launch"
    assert _owner((p_other,)) == f"{other.user} eval.py"
    assert _owner((p_loose,)) == "dave sample.py"
    assert _owner((p_mine, p_other)) == "mine orb.launch +1"
    assert _owner((p_mine, p_other), wide=False) == "mine+1"
    assert _owner(util=40) == "in use · no process" and _owner(util=40, wide=False) == "busy"
    assert _owner((p_mine,), old=True) == "?"
    for wide, width in ((True, fl.OWNER_WIDE), (False, fl.OWNER_NARROW)):
        g = snap.gpus[0]
        assert fl.owner_text(snap, g, old=False, wide=wide).cell_len == width


# ---------------------------------------------------------------- free GPUs by type
def _node(name: str, parts: tuple[str, ...], state: str, gtype: str, total: int, used: int, *, idle_cpus: int = 4,
          flags: str = "") -> Node:
    return Node(name, parts, state, frozenset(flags), state in ("idle", "mixed", "allocated", "planned"), 0,
                idle_cpus, 0, 32, None, None, None, (GresCount("gpu", gtype, total, used, None),), None)


def _cluster(nodes: tuple[Node, ...], parts: tuple[str, ...], *, stale: bool = False):
    st = _states()["k2"]
    snap = st.snapshot
    assert snap is not None
    summaries = tuple(PartitionSummary(p, 1, 1, 0, 0, {}, {}) for p in parts)
    st.snapshot = dataclasses.replace(snap, nodes=nodes, partitions=summaries)
    sections = {"slurm.sinfo": SectionStatus("slurm.sinfo", "ok", True, 0, 100, T0 - timedelta(minutes=3))}
    st.status = dataclasses.replace(st.status, sections=sections, last_ok=T0 - timedelta(hours=1) if stale else T0)
    return st


def test_free_by_type_counts_each_node_once() -> None:
    nodes = (
        _node("lyra-gpu-36-04", ("free-gpu",), "allocated", "v100", 4, 3, idle_cpus=0),   # 1 idle V100: not free
        _node("n-both", ("free-gpu", "free-gpuxl"), "mixed", "v100", 4, 1),              # listed twice, counted once
        _node("n-outside", ("labpi",), "idle", "v100", 8, 0),                            # not a listed partition
        _node("n-mig", ("free-gpu",), "mixed", "nvidia_a100_80gb_pcie_1g.10gb", 7, 2, flags="-"),   # planned
    )
    free = fl.free_by_type({"k2": _cluster(nodes, ("free-gpu", "free-gpuxl"))}, T0)
    v100 = free.types["V100"]
    assert (v100.free, v100.total, v100.where) == (3, 8, {("k2", "free-gpu"): [3, 0]})
    mig = free.types["MIG slices (k2)"]
    assert (mig.label, mig.host, mig.free, mig.planned) == ("MIG slices", "k2", 5, 5)
    assert free.unknown == () and free.sinfo_age_s == 180
    stale = fl.free_by_type({"k2": _cluster(nodes, ("free-gpu",), stale=True)}, T0)
    assert stale.types == {} and stale.unknown == ("k2",)


def test_free_lines_your_types_first() -> None:
    st = _states()
    hosts = {"k2": st["k2"], "labstation": st["labstation"]}
    free = fl.free_by_type(hosts, T0)
    mine = fl.my_types(hosts, T0, DAY)
    assert "A6000" in mine
    lines = [x.plain for x in fl.free_lines(free, mine, 6, 81, wide=True, host_order=list(hosts))]
    assert lines[0].startswith("free GPUs now · your types first") and lines[0].rstrip().endswith("sinfo <1m ago")
    assert any(x.strip().startswith("A6000") and "labstation gpu 0, 2" in x for x in lines)
    assert all(len(x) <= 81 for x in lines)


# ---------------------------------------------------------------- activity
def test_activity_rows_dedup_and_dim() -> None:
    ended = slurm_job("k2", "7", "crashed", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=600, exit_code=1)
    done = slurm_job("k2", "8", "finished", JobState.COMPLETED, elapsed=timedelta(hours=2, minutes=3),
                     ended_ago=7200, exit_code=0)
    key = JobKey("k2", "9")
    hist = [JobEvent("time_low", key, "short", JobState.RUNNING, T0 - timedelta(minutes=5), "8m left of 2:00:00"),
            JobEvent("time_low", key, "short", JobState.RUNNING, T0 - timedelta(minutes=9), "9m left of 2:00:00"),
            JobEvent("restarted", key, "short", JobState.PENDING, T0 - timedelta(hours=3), "↻1"),
            JobEvent("started", key, "short", JobState.RUNNING, T0 - timedelta(hours=3))]
    since = T0 - timedelta(hours=1)
    rows = fl.activity_rows([ended, done], hist, T0, DAY, since)
    assert [(r.glyph, r.name, r.old) for r in rows] == [
        ("◷", "short", False), ("✖", "crashed", False), ("✓", "finished", True), ("↻", "short", True)]
    assert rows[2].summary == "ran 2:03:00"
    assert fl.count_text([ended, done], hist, since) == "1 failed · 1 low on time"
    lines = fl.activity_lines(rows, "1 failed", "since 09:42: ", 4, 60, wide=False)
    assert len(lines) == 4 and all(x.cell_len == 60 for x in lines)


# ---------------------------------------------------------------- T0 identity
GOLDEN = Path(__file__).parent / "golden" / "fleet_t0_cards.json"


def test_t0_cards_unchanged() -> None:
    """card_lines(list_rows=None) is today's card, line for line (golden: the cards before the roomy layout)."""
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for state in build_world()[1]:
        for width in (45, 55, 81):
            got = [x.plain for x in card_lines(state, T0, width=width)]
            assert got == expected[f"{state.config.name}@{width}"], (state.config.name, width)


def test_t1_cards_fit_their_rows() -> None:
    st = _states()
    for name, state in st.items():
        shape = fl.shape_of(state, 55, HighWater())
        if not shape.cap:
            continue
        for rows in (3, 8, 12):
            lines = card_lines(state, T0, width=55, list_rows=rows, wide=False, fixed=shape.fixed)
            assert shape.t1 <= len(lines) <= shape.t1 + rows, name        # set_lines pads to the row height
            assert all(x.cell_len <= 55 for x in lines), name
    auth = HostError(ErrorKind.AUTH, "Permission denied (publickey)")
    vega = dataclasses.replace(st["vega"], status=dataclasses.replace(st["vega"].status, state=ConnState.NEEDS_AUTH,
                                                                      error=auth))
    shape = fl.shape_of(vega, 81, HighWater())
    lines = [x.plain for x in card_lines(vega, T0, width=81, list_rows=6, wide=True, fixed=shape.fixed)]
    assert lines[shape.t1 - 1].startswith("✖") and lines[shape.t1].startswith("  press c to log in")


# ---------------------------------------------------------------- review fixes
def test_narrow_util_column_never_moves() -> None:
    """Review: an 11-cell time (`20h39m left`) shifted the GPU% cell one column left, and a 3-GPU task's `gpu 0,1,2`
    ran straight into it."""
    jobs = [slurm_job("k2", "1", "alpha", JobState.RUNNING, elapsed=timedelta(hours=1), util=50,
                      limit=timedelta(hours=1, minutes=34)),
            slurm_job("k2", "2", "beta", JobState.RUNNING, elapsed=timedelta(hours=1), util=50,
                      limit=timedelta(hours=21, minutes=40))]
    lines = [x.plain for x in fl.list_lines(fl.rank_slurm(jobs, T0, DAY), 2, T0, inner=55, wide=False)]
    assert "20h40m left" in lines[1] and "34m left" in lines[0]
    assert len({x.index("50%") for x in lines}) == 1
    task = task_job("labstation", 41, "python (train.py)", elapsed=timedelta(hours=2), gpus=(0, 1, 2),
                    gpu_model="NVIDIA RTX A6000", util=50, mem_mib=1000, mem_total_mib=49140, pids=(41,))
    row = fl.row_text(fl.Ranked(2, task, "running"), T0, inner=55, wide=False).plain
    assert "gpu 0,1,2 " in row and row.index("50%") == lines[0].index("50%")


def test_folded_array_tasks_in_task_order() -> None:
    """Review: siblings 3, 5, 12 read `[12,3,5]` (rank order)."""
    arr = [dataclasses.replace(slurm_job("k2", f"900_{t}", "orb-sweep", JobState.RUNNING,
                                         elapsed=timedelta(hours=1), limit=timedelta(hours=lim)),
                               array_job_id="900", array_task=str(t)) for t, lim in ((12, 2), (3, 3), (5, 4))]
    assert fl.array_suffix(arr[0], arr) == " [3,5,12] ×3"


def test_unsampled_gpu_is_not_in_use() -> None:
    """Review: a GPU without any reading read `in use · no process`; it is unknown."""
    st = _states()["labstation"]
    assert st.snapshot is not None
    g = GPU("labstation", 0, "GPU-x", "NVIDIA RTX A6000", None, None, 49140, None, None, None, None, None, None,
            "nvidia-smi", (), T0)
    assert fl.owner_text(st.snapshot, g, old=False, wide=True).plain.rstrip() == "no reading"
    assert fl.owner_text(st.snapshot, g, old=False, wide=False).plain.rstrip() == "?"


def test_one_label_per_gpu_model_across_clusters_and_workstations() -> None:
    """Review: `nvidia_a100-sxm4-40gb` (k2) and `NVIDIA A100-SXM4-40GB` (a workstation) were two panel rows."""
    for gres, smi, label in (("nvidia_a100-sxm4-40gb", "NVIDIA A100-SXM4-40GB", "A100-40G"),
                             ("nvidia_h100_80gb_hbm3", "NVIDIA H100 80GB HBM3", "H100-80G"),
                             ("nvidia_a100_80gb", "NVIDIA A100 80GB PCIe", "A100-80G"),
                             ("nvidia_geforce_rtx_3090", "NVIDIA GeForce RTX 3090", "3090"),
                             ("nvidia_rtx_a6000", "NVIDIA RTX A6000", "A6000")):
        assert fl.type_label(gres) == fl.type_label(smi) == label
    st = _states()
    lab = st["labstation"]
    assert lab.snapshot is not None
    lab.snapshot = dataclasses.replace(lab.snapshot, gpus=tuple(dataclasses.replace(g, name="NVIDIA A100-SXM4-40GB")
                                                                  for g in lab.snapshot.gpus))
    hosts = {"k2": st["k2"], "labstation": lab}
    free = fl.free_by_type(hosts, T0)
    assert "A100" not in free.types and free.types["A100-40G"].gpus.get("labstation")
    assert "A100" not in fl.my_types(hosts, T0, DAY)


def test_panel_labels_wider_than_the_column_keep_counts_aligned() -> None:
    """Review: `GH200-480G` (10 cells) pushed its count one column right of the others in a narrow panel."""
    types = {"GH200-480G": fl.TypeFree("GH200-480G", 2, 4), "H200": fl.TypeFree("H200", 3, 8)}
    lines = [x.plain for x in fl.free_lines(fl.FreeGPUs(types, (), None), {"GH200-480G", "H200"}, 4, 50,
                                            wide=False)]
    ends = {x.rstrip().split()[0]: len(x.rstrip()) for x in lines[1:3]}     # where each count ends
    assert ends["GH200-480G"] == ends["H200"], lines


def test_memo_skips_the_scans_until_the_data_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: the 1 Hz tick redid every fleet-wide scan (my_types, free_by_type, activity rows, distinct_starts)."""
    from runfleet.ui.widgets.fleet_panel import panel_lines

    hosts = _states()
    memo = fl.Memo()
    calls: dict[str, int] = {}
    for name in ("_cluster_free", "_host_types", "own_ended", "activity_rows"):
        real = getattr(fl, name)

        def spy(*a: object, _real: object = real, _name: str = name, **k: object) -> object:
            calls[_name] = calls.get(_name, 0) + 1
            return _real(*a, **k)       # type: ignore[operator]
        monkeypatch.setattr(fl, name, spy)
    first = [x.plain for x in panel_lines(hosts, T0, rows=12, width=60, wide=False, window=DAY, memo=memo)]
    before = dict(calls)
    assert before["activity_rows"] == 1 and before["_cluster_free"] >= 1
    again = [x.plain for x in panel_lines(hosts, T0 + timedelta(seconds=1), rows=12, width=60, wide=False,
                                          window=DAY, memo=memo)]
    assert calls == before and again == first                    # same data, same minute: nothing rescanned
    k2 = hosts["k2"]
    assert k2.snapshot is not None
    k2.snapshot = dataclasses.replace(k2.snapshot, jobs=dict(k2.snapshot.jobs))     # a poll: a new jobs dict
    panel_lines(hosts, T0 + timedelta(seconds=2), rows=12, width=60, wide=False, window=DAY, memo=memo)
    assert calls["_host_types"] == before["_host_types"] + 1 and calls["activity_rows"] == 2
    assert calls["_cluster_free"] == before["_cluster_free"]       # nodes unchanged: no node scan


def test_legend_shows_panel_glyphs_only_with_the_panel() -> None:
    """Review: the fleet panel's legend rows showed in table mode (80×24), cut short and with a misaligned key."""
    from runfleet.ui.screens.help import legend

    table = legend(["fleet", "main", "global"], 72).plain
    assert "what changed" not in table and "free GPUs" not in table
    roomy = legend(["fleet", "main", "global", "fleet_panel"], 104).plain.splitlines()[1:]
    assert "what changed" in "\n".join(roomy) and not any(x.endswith("…") for x in roomy)
    starts = {x.index(" ", len("✖ ✓ ■ ◷ ↻ ◌")) for x in roomy}
    assert len(starts) == 1, roomy                                  # every description starts in the same cell
    assert any(x.startswith("◎ ") and "GPU watch fired" in x for x in roomy)   # review fix: the watches' glyph too


# ---------------------------------------------------------------- deep review (B E F G J K L U V)
def _with_sinfo(st, *, ok: bool = True, ago: float = 180.0, slow: float = 1.0, poll_nodes: float = 120.0):
    sec = SectionStatus("slurm.sinfo", "ok" if ok else "error", ok, 0 if ok else 1, 100,
                        T0 - timedelta(seconds=ago))
    st.status = dataclasses.replace(st.status, sections={"slurm.sinfo": sec}, controller_slowdown=slow)
    st.config = dataclasses.replace(st.config, slurm=dataclasses.replace(st.config.slurm, poll_nodes=poll_nodes))
    return st


def test_free_by_type_needs_a_good_recent_sinfo() -> None:
    """Deep review B: free GPUs were counted from a failed or long-stale `slurm.sinfo` section while the host itself was
    fresh (OK / DEGRADED); such a host is listed as "host ?" now."""
    nodes = (_node("n1", ("free-gpu",), "idle", "v100", 4, 0),)

    def seen(st, rate: float = 1.0) -> tuple[int | None, tuple[str, ...]]:
        free = fl.free_by_type({"k2": st}, T0, rate=rate)
        t = free.types.get("V100")
        return (t.free if t is not None else None), free.unknown

    st = _with_sinfo(_cluster(nodes, ("free-gpu",)))
    assert seen(st) == (4, ())
    bound = fl.sinfo_bound_s(st)
    assert bound == 2 * 120 * fl.NODES_HIDDEN_FACTOR * fl.IDLE_FACTOR + 5            # 80 min at the defaults
    st.status = dataclasses.replace(st.status, state=ConnState.DEGRADED)
    assert seen(_with_sinfo(st, ok=False)) == (None, ("k2",))                         # sinfo failed: "k2 ?"
    assert seen(_with_sinfo(st, ago=bound - 1)) == (4, ())
    assert seen(_with_sinfo(st, ago=bound + 1)) == (None, ("k2",))                     # too old for the cadence
    assert seen(_with_sinfo(st, ago=bound + 1, slow=4.0)) == (4, ())                   # a busy controller: slower
    assert seen(_with_sinfo(st, ago=bound + 1), rate=2.0) == (4, ())                   # polling slowed with +/-
    st.status = dataclasses.replace(st.status, sections={})
    assert seen(st) == (None, ("k2",))                                                 # cached nodes, no sample yet
    assert st.snapshot is not None
    st.snapshot = dataclasses.replace(st.snapshot, nodes=())
    assert seen(st) == (None, ())                                                      # nothing in scope: nothing


def test_sinfo_bound_follows_the_nodes_cadence() -> None:
    """The bound's constants are the Slurm collector's `slurm.nodes` cadence (the UI cannot import it)."""
    from runfleet.backends.slurm.collector import SlurmCollector
    from tests.backends.factories import ctx

    cx = ctx("k2")
    nodes = {c.group: c for c in SlurmCollector(cx.host).cadences(cx)}["slurm.nodes"]
    assert (nodes.floor, nodes.hidden_factor, nodes.max_backoff) == (
        fl.NODES_FLOOR_S, fl.NODES_HIDDEN_FACTOR, fl.NODES_MAX_BACKOFF_S)


def _sweep(n: int, state: JobState = JobState.PENDING) -> list:
    return [dataclasses.replace(slurm_job("k2", f"900_{t}", "orb-sweep", state), array_job_id="900", array_task=str(t))
            for t in range(n)]


def test_array_fold_builds_each_row_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deep review E: every folded sibling copied the members tuple so far (quadratic in the array size)."""
    ranked = fl.rank_slurm(_sweep(400), T0, DAY)
    real = fl.replace
    calls: list[int] = []

    def spy(*a: object, **k: object) -> object:
        calls.append(1)
        return real(*a, **k)       # type: ignore[call-overload]
    monkeypatch.setattr(fl, "replace", spy)
    shown, hidden = fl.select(ranked, 5)
    assert len(calls) == 1 and hidden == []
    assert shown[0].count == 400 and {j.array_task for j in shown[0].members} == {str(t) for t in range(400)}


def test_selection_is_kept_per_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deep review E: the 1 Hz tick folded and selected again although neither the ranking nor the rows changed."""
    ranked = fl.rank_slurm(_sweep(30) + [slurm_job("k2", "1", "solo", JobState.PENDING)], T0, DAY)
    real = fl._collapse
    calls: list[int] = []

    def spy(items: object) -> object:
        calls.append(1)
        return real(items)         # type: ignore[arg-type]
    monkeypatch.setattr(fl, "_collapse", spy)
    first = [x.plain for x in fl.list_lines(ranked, 4, T0, inner=55, wide=False)]
    again = [x.plain for x in fl.list_lines(ranked, 4, T0 + timedelta(seconds=1), inner=55, wide=False)]
    assert first == again and len(calls) == 1
    fl.list_lines(ranked, 3, T0, inner=55, wide=False)                   # other rows: selected anew
    fl.list_lines(list(ranked), 4, T0, inner=55, wide=False)             # a plain list is never cached
    assert len(calls) == 3


def test_free_block_reserves_the_lines_the_data_takes() -> None:
    """Deep review F: a partition too wide for the block even alone folded the worst case into one "+N more" line,
    today's smaller numbers took two, and the T1 card dropped the long partition without a word."""
    st = _states()["k2"]
    assert st.snapshot is not None
    long = "x" * 37
    parts = (PartitionSummary("gpu", 1, 8, 0, 0, {}, {}), PartitionSummary(long, 10, 1000, 0, 0, {}, {}))
    st.snapshot = dataclasses.replace(st.snapshot, partitions=parts)
    shape = fl.shape_of(st, 55, HighWater())
    assert shape.fixed == 2 and shape.t0 == 7
    lines = [x.plain for x in card_lines(st, T0, width=55, list_rows=4, wide=False, fixed=shape.fixed)]
    assert lines[3].startswith("free GPUs gpu 0/8") and long in lines[4]
    squeezed = [x.plain for x in card_lines(st, T0, width=55, list_rows=4, wide=False, fixed=1)]
    assert squeezed[3].rstrip().endswith("+1 more") and not squeezed[4].startswith(" ")    # cut, but said so


def test_activity_rows_pad_by_cells() -> None:
    """Deep review G: names were padded by characters, so a CJK name (2 cells each) or one with combining accents
    (0 cells) moved the summary column."""
    when = [T0 - timedelta(minutes=m) for m in (5, 7, 9)]
    rows = [fl.Activity(when[0], "k2", "✖", "训练模型-第一阶段-学习率", "exit 1", True, False),
            fl.Activity(when[1], "k2", "✖", "café-run-été-série", "exit 1", True, False),
            fl.Activity(when[2], "k2", "✖", "train-stage-one-lr", "exit 1", True, False)]
    for wide, width in ((True, 82), (False, 56)):
        lines = fl.activity_lines(rows, "3 failed", "last 24h: ", 4, width, wide=wide, now=T0)[1:]
        assert all(x.cell_len == width for x in lines)
        starts = {cell_len(x.plain[:x.plain.index("exit 1")]) for x in lines}
        assert len(starts) == 1, [x.plain for x in lines]


def test_type_label_one_label_per_model() -> None:
    """Deep review J: `rtx3090` / `GeForce RTX 3090` and `rtx6000` / `Quadro RTX 6000` were two rows, while the
    Turing `Quadro RTX 6000` and the `RTX 6000 Ada` were one."""
    for a, b in (("rtx3090", "NVIDIA GeForce RTX 3090"), ("rtx6000", "Quadro RTX 6000"),
                 ("quadro_rtx_6000", "Quadro RTX 6000"), ("rtxa6000", "NVIDIA RTX A6000"),
                 ("nvidia_rtx_6000_ada_generation", "NVIDIA RTX 6000 Ada Generation")):
        assert fl.type_label(a) == fl.type_label(b), (a, b)
    assert fl.type_label("rtx3090") == "3090" and fl.type_label("Quadro RTX 6000") == "6000"
    assert fl.type_label("NVIDIA RTX 6000 Ada Generation") == "6000Ada"
    assert fl.type_label("rtx6000") != fl.type_label("rtx_6000_ada")


def test_type_label_keeps_ti_and_gtx_models_apart() -> None:
    """Verify J: `rtx2080ti` and `GeForce RTX 2080 Ti` were two rows (`2080TI`, `2080`), and every GTX card was one
    `GTX` row (a 1080 Ti's free GPUs added to a 1070's)."""
    for a, b in (("rtx2080ti", "NVIDIA GeForce RTX 2080 Ti"), ("gtx1080ti", "NVIDIA GeForce GTX 1080 Ti"),
                 ("rtx_3080_ti", "NVIDIA GeForce RTX 3080 Ti"), ("rtx2080super", "NVIDIA GeForce RTX 2080 SUPER"),
                 ("gtx1070", "NVIDIA GeForce GTX 1070")):
        assert fl.type_label(a) == fl.type_label(b), (a, b)
    assert fl.type_label("gtx1080ti") == "1080TI" and fl.type_label("gtx1070") == "1070"
    assert len({fl.type_label(n) for n in ("rtx2080", "rtx2080ti", "rtx2080super")}) == 3


def test_panel_leaves_out_alerts_of_removed_hosts() -> None:
    """Deep review K: a host removed in Settings kept its ◷ / ↻ rows and its "low on time" count in the panel."""
    from runfleet.ui.widgets.fleet_panel import panel_lines

    hosts = _states()
    ev = [JobEvent("time_low", JobKey("oldbox", "123"), "gone-host-job", JobState.RUNNING, T0 - timedelta(minutes=5),
                   "9m left of 2:00:00"),
          JobEvent("time_low", JobKey("k2", "124"), "kept-job", JobState.RUNNING, T0 - timedelta(minutes=6),
                   "9m left of 2:00:00")]
    text = "\n".join(x.plain for x in panel_lines(hosts, T0, rows=14, width=82, wide=True, window=DAY, history=ev))
    assert "gone-host-job" not in text and "kept-job" in text and "1 low on time" in text


def test_activity_rows_show_the_day_after_20h() -> None:
    """Deep review L: rows showed only HH:MM, so yesterday's 11:00 read as later than today's 09:30."""
    ats = [T0 - timedelta(minutes=5), T0 - timedelta(hours=21), T0 - timedelta(days=7)]
    rows = [fl.Activity(at, "k2", "✓", f"job-{i}", "ran 1:00:00", False, False) for i, at in enumerate(ats)]
    lines = [x.plain for x in fl.activity_lines(rows, "3 completed", "last 8d: ", 4, 70, wide=False, now=T0)[1:]]
    assert lines[0].split()[0] == ats[0].astimezone().strftime("%H:%M")
    assert ats[1].astimezone().strftime("%a %H:%M") in lines[1]
    assert ats[2].astimezone().strftime("%m-%d %H:%M") in lines[2]
    assert len({x.index("✓") for x in lines}) == 1                       # the time column has one width
    assert fl.when_label(T0 - timedelta(hours=19, minutes=59), T0) == fl.humanize.local_hm(T0 - timedelta(hours=19,
                                                                                                     minutes=59))


def test_rank_cache_expires_when_a_failure_crosses_a_line() -> None:
    """Deep review V: the card list kept a failure for up to a minute after it left the window (or 4 h) while the
    card's `my jobs` count, computed fresh, had already dropped it."""
    from types import SimpleNamespace

    from runfleet.ui.widgets.fleet_view import FleetView

    st = _states()["k2"]
    assert st.snapshot is not None
    crash = slurm_job("k2", "77", "crashed", JobState.FAILED, elapsed=timedelta(minutes=5), ended_ago=50, exit_code=1)
    st.snapshot = dataclasses.replace(st.snapshot, jobs={crash.key.id: crash})
    view = SimpleNamespace(_ranks={})
    ranked, _ = FleetView._ranked(view, st, T0, 60.0)                 # type: ignore[misc]
    assert [r.job.key.id for r in ranked] == ["77"]
    ranked, _ = FleetView._ranked(view, st, T0 + timedelta(seconds=15), 60.0)    # same minute, out of the window
    assert ranked == []
    fresh = dataclasses.replace(crash, end_time=T0 - timedelta(seconds=fl.FRESH_FAIL_S - 5))
    st.snapshot = dataclasses.replace(st.snapshot, jobs={fresh.key.id: fresh})
    assert FleetView._ranked(view, st, T0, DAY)[0][0].rank == 0       # type: ignore[misc]
    assert FleetView._ranked(view, st, T0 + timedelta(seconds=10), DAY)[0][0].rank == 3    # type: ignore[misc]


def test_activity_memo_expires_when_an_end_leaves_the_window() -> None:
    """Deep review V: the panel's activity part said "1 failed" for up to a minute after the end left the window."""
    from runfleet.ui.widgets.fleet_panel import panel_lines

    hosts = _states()
    k2 = hosts["k2"]
    assert k2.snapshot is not None
    crash = slurm_job("k2", "77", "crashed", JobState.FAILED, elapsed=timedelta(minutes=5), ended_ago=50, exit_code=1)
    k2.snapshot = dataclasses.replace(k2.snapshot, jobs={**k2.snapshot.jobs, crash.key.id: crash})
    memo = fl.Memo()

    def text(at: object) -> str:
        return "\n".join(x.plain for x in panel_lines(hosts, at, rows=12, width=60, wide=False, window=60.0,  # type: ignore[arg-type]
                                                      memo=memo))
    assert "1 failed" in text(T0) and "crashed" in text(T0)
    later = text(T0 + timedelta(seconds=15))                           # same minute, same jobs dict
    assert "1 failed" not in later and "crashed" not in later


def test_free_by_type_adds_up_to_the_partition_summaries() -> None:
    """Deep review U: for nodes in one partition each, the panel's free GPUs per host (whole and MIG) are the sum of
    `summarize()`'s gpus_free, partition by partition."""
    from runfleet.backends.slurm.parse_sinfo import summarize

    nodes = (
        _node("a1", ("gpu",), "idle", "nvidia_a100-sxm4-40gb", 4, 0),
        _node("a2", ("gpu",), "mixed", "nvidia_a100-sxm4-40gb", 4, 3),
        _node("a3", ("gpu",), "allocated", "nvidia_a100-sxm4-40gb", 4, 3, idle_cpus=0),     # idle GPU, no CPU
        _node("a4", ("gpu",), "down", "nvidia_a100-sxm4-40gb", 4, 0),                       # unusable
        _node("h1", ("h200",), "mixed", "h200", 8, 2, flags="-"),                           # planned
        _node("m1", ("mig",), "mixed", "nvidia_a100_80gb_pcie_1g.10gb", 7, 2),
        _node("m2", ("mig",), "planned", "nvidia_a100_80gb_pcie_3g.40gb", 2, 0),
        _node("g1", ("mig",), "idle", "a30", 4, 1),
    )
    names = ("gpu", "h200", "mig")
    hosts = {}
    for host, nn in (("k2", nodes), ("k3", nodes[:4])):
        st = _with_sinfo(_cluster(nn, names))
        assert st.snapshot is not None
        st.snapshot = dataclasses.replace(st.snapshot, partitions=summarize(nn, names))
        hosts[host] = st
    free = fl.free_by_type(hosts, T0)
    for host, st in hosts.items():
        assert st.snapshot is not None
        per: dict[str, int] = {}
        for t in free.types.values():
            for (h, part), (n, _planned) in t.where.items():
                if h == host:
                    per[part] = per.get(part, 0) + n
        want = {p.name: p.gpus_free for p in st.snapshot.partitions}
        assert sum(per.values()) == sum(want.values()) and all(per.get(p, 0) == n for p, n in want.items()), host


# ---------------------------------------------------------------- opening entries (job keys, "needs you")
def test_list_region_keeps_the_job_of_each_row() -> None:
    """Every shown list row carries its JobKey (a folded array row: its first job), the overflow line None."""
    fails = [slurm_job("k2", f"5{i}", f"f{i}", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=600 * (i + 1),
                       exit_code=1) for i in range(2)]
    arr = [dataclasses.replace(slurm_job("k2", f"900_{t}", "orb-sweep", JobState.RUNNING, elapsed=timedelta(hours=1)),
                               array_job_id="900", array_task=str(t)) for t in (3, 4, 5)]
    pend = [slurm_job("k2", "77", "queued", JobState.PENDING, reason="Priority")]
    ranked = fl.rank_slurm([*pend, *arr, *fails], T0, DAY)
    lines, keys = fl.list_region(ranked, 4, T0, inner=60, wide=False)
    assert keys == [JobKey("k2", "50"), JobKey("k2", "51"), JobKey("k2", "900_3"), JobKey("k2", "77")]
    assert len(lines) == len(keys) and "orb-sweep [3-5] ×3" in lines[2].plain
    lines, keys = fl.list_region(ranked, 3, T0, inner=60, wide=False)
    assert keys[-1] is None and lines[-1].plain.startswith("+4 more") and len(lines) == len(keys) == 3
    assert [x.plain for x in fl.list_lines(ranked, 3, T0, inner=60, wide=False)] == [x.plain for x in lines]


def test_activity_rows_keep_their_job() -> None:
    ended = slurm_job("k2", "7", "crashed", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=600, exit_code=1)
    key = JobKey("vega", "9")
    hist = [JobEvent("time_low", key, "short", JobState.RUNNING, T0 - timedelta(minutes=5), "8m left of 2:00:00")]
    rows = fl.activity_rows([ended], hist, T0, DAY, T0 - timedelta(hours=1))
    assert [r.key for r in rows] == [key, JobKey("k2", "7")]
    lines, keys = fl.activity_region(rows, "1 failed", "last 24h: ", 4, 60, wide=False)
    assert keys == [None, key, JobKey("k2", "7")] and len(lines) == 3
    assert "short" in lines[1].plain and "crashed" in lines[2].plain
    assert fl.activity_region(rows, "", "", 0, 60, wide=False) == ([], [])


def test_card_body_keys_match_the_list_rows() -> None:
    """A T1 card's keys sit on the lines that show those jobs; the summary lines and a T0 card have none."""
    _cfg, states = build_world()
    for st in states:
        if st.snapshot is None:
            continue
        shape = fl.shape_of(st, 55, HighWater())
        lines, keys = card_body(st, T0, width=55, list_rows=6, wide=False, fixed=shape.fixed)
        assert len(lines) == len(keys)
        assert all(k is None for k in keys[:shape.t1])
        listed = [(line.plain, k) for line, k in zip(lines, keys, strict=True) if k is not None]
        assert listed, st.config.name
        for plain, k in listed:
            job = st.snapshot.jobs[k.id]
            stem = job.name.split(" (")[0]
            assert stem[:3] in plain or stem[-3:] in plain, (plain, job.name)        # a cut name keeps one end
        t0_lines, t0_keys = card_body(st, T0, width=55)
        assert t0_keys == [None] * len(t0_lines)


def _needs() -> tuple[list[tuple[str, bool, fl.RankedList]], dict[str, object]]:
    k2 = [
        slurm_job("k2", "10", "fresh-old", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=3000, exit_code=1),
        slurm_job("k2", "11", "earlier", JobState.FAILED, elapsed=timedelta(hours=1), ended_ago=fl.FRESH_FAIL_S + 60,
                  exit_code=2),
        slurm_job("k2", "12", "short", JobState.RUNNING, elapsed=timedelta(minutes=55), limit=timedelta(hours=1)),
        slurm_job("k2", "13", "fine", JobState.RUNNING, elapsed=timedelta(hours=1)),
        slurm_job("k2", "14", "never", JobState.PENDING, reason="DependencyNeverSatisfied"),
        slurm_job("k2", "15", "queued", JobState.PENDING, reason="Priority"),
    ]
    vega = [
        slurm_job("vega", "20", "fresh-new", JobState.OUT_OF_MEMORY, elapsed=timedelta(hours=1), ended_ago=60),
        slurm_job("vega", "21", "stalled", JobState.RUNNING, elapsed=timedelta(hours=3), hints=("log idle 51m",)),
    ]

    def task(pid: int, *, mine: bool = True, hints: tuple[str, ...] = (), ended: float | None = None,
             state: JobState = JobState.RUNNING):
        j = task_job("ws", pid, f"t{pid}", elapsed=timedelta(hours=1), gpus=(0,), gpu_model="NVIDIA RTX A6000",
                     util=50, mem_mib=1024, mem_total_mib=49140, pids=(pid,), mine=mine, state=state,
                     user="alice07" if mine else "carol")
        if ended is not None:
            j = dataclasses.replace(j, end_time=T0 - timedelta(seconds=ended))
        return dataclasses.replace(j, hints=hints)

    ws = [task(1, hints=("GPU idle 20m",)), task(2), task(3, mine=False),
          task(4, state=JobState.FAILED, ended=fl.FRESH_FAIL_S + 600)]
    per_host = [("k2", True, fl.rank_slurm(k2, T0, DAY)), ("vega", True, fl.rank_slurm(vega, T0, DAY)),
                ("ws", False, fl.rank_ws(ws, T0, DAY))]
    return per_host, {j.key.id: j for j in [*k2, *vega, *ws]}


def test_attention_merges_the_rankings_in_rank_order() -> None:
    """Fresh failures newest first across hosts, then running jobs that need you, pending ones that will never run,
    older failures (newest first); normal running / pending and other users' tasks are not listed."""
    per_host, jobs = _needs()
    got = fl.attention(per_host)
    names = [(a.host, a.item.job.name, a.tier) for a in got]
    assert names == [("vega", "fresh-new", 0), ("k2", "fresh-old", 0),
                     ("k2", "short", 1), ("vega", "stalled", 1), ("ws", "t1", 1),
                     ("k2", "never", 2),
                     ("k2", "earlier", 3), ("ws", "t4", 3)]
    assert got[0].key == JobKey("vega", "20")
    assert fl.attention([]) == []


def test_attention_says_why() -> None:
    per_host, _jobs = _needs()
    why = {a.item.job.name: fl.attention_why(a, T0).plain for a in fl.attention(per_host)}
    assert why["fresh-new"].endswith(" · 1m ago") and why["earlier"].startswith("exit 2")
    assert why["short"] == "low on time · 5m left"
    assert why["stalled"].startswith("log idle 51m · ")
    assert why["never"] == "never: dependency failed · no est."
    assert why["t1"].startswith("GPU idle 20m")
