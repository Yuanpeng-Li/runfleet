"""[WP3] Pure renderers behind the widgets: cards, table layout and cells, preview, panels, dialogs."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

from rich.text import Text

from runfleet.backends.base import ActionPlan, ActionSpec
from runfleet.core.service import LogPeek
from runfleet.models import (
    GPU, ConnState, DoctorCheck, DoctorReport, ErrorKind, HostError, JobKey, JobState, SectionStatus,
)
from runfleet.ui import theme
from runfleet.ui.screens.confirm import dialog_title, plan_body
from runfleet.ui.screens.help import render_help
from runfleet.ui.screens.host_detail import capability_lines, connection_lines, host_metrics_line, my_jobs_line
from runfleet.ui.screens.host_error import doctor_text, error_report
from runfleet.ui.screens.job_detail import gpu_lines, job_box_lines, resources_lines, stale_banner, title_line
from runfleet.ui.widgets.fleet_view import columns_for, host_row
from runfleet.ui.widgets.gpu_panel import gpu_panel_lines
from runfleet.ui.widgets.gpus_view import job_row, task_short_name, ws_row
from runfleet.ui.widgets.host_card import MAX_GPU_LINES, card_lines, card_title, is_cluster
from runfleet.ui.widgets.job_preview import gpu_line, log_lines, meta_line, wrap_parts
from runfleet.ui.widgets.job_table import RowContext, fit_columns, job_cells, layout, node_reason
from runfleet.ui.widgets.node_table import gpu_totals, states_text
from tests.ui.fakes import T0, FakeHostState, build_world, slurm_job

T = (10.0, 75.0)
M = (10.0, 80.0)


def _states() -> dict[str, FakeHostState]:
    _config, states = build_world()
    return {s.config.name: s for s in states}   # type: ignore[misc]


def _plain(lines: list[Text]) -> str:
    return "\n".join(t.plain for t in lines)


def _ctx(**kw: object) -> RowContext:
    base = dict(now=T0, host_colors={}, problem_hosts={}, stale_hosts=frozenset(), marked=frozenset(),
                thresholds=T)
    base.update(kw)
    return RowContext(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- host cards
def test_slurm_card_lines() -> None:
    k2 = _states()["k2"]
    lines = card_lines(k2, T0, width=56)
    text = _plain(lines)
    assert len(lines) == 6 and all(t.cell_len <= 56 for t in lines)
    assert "my jobs" in text and "R 4" in text and "CD 7" in text and "(last 24h)" in text
    assert "A100-40G ×3" in text and "H200 ×2" in text
    assert "gpu-preempt 31/164" in text
    assert "smoke-test" in text and "priority" in text
    assert "login02" in text and "load 2.3/32" in text


def test_card_without_gpu_accounting_and_with_error() -> None:
    lyra = _states()["lyra"]
    text = _plain(card_lines(lyra, T0, width=56))
    assert "n/a — no GPU accounting on this cluster" in text
    assert "! ssh: Connection timed out during banner exchange" in text
    assert "next retry in 38s" in text and "e info" in text
    title = card_title(lyra, 3, T0).plain
    assert title.startswith("3 lyra · slurm 26.05.3 · ◐ retrying") and "last ok 4m ago" in title


def test_workstation_card_lines_and_auth_error() -> None:
    states = _states()
    qs = _plain(card_lines(states["labstation"], T0, width=56))
    assert qs.count("A6000") == 4 and "21.0/48.0G" in qs and "78°C" in qs
    assert "mine 1 (orb.launch, 1 GPU) · others 1" in qs and "cpu" in qs
    uc = card_lines(states["lynx"], T0, width=56)
    text = _plain(uc)
    assert "Permission denied" in text and "press c to log in" in text
    assert "cpu" not in text                                        # host line gives way to the error


def test_workstation_card_caps_gpu_lines() -> None:
    qs = _states()["labstation"]
    gpus = tuple(dataclasses.replace(qs.snapshot.gpus[0], index=i) for i in range(10))   # type: ignore[union-attr]
    qs.snapshot = dataclasses.replace(qs.snapshot, gpus=gpus)                              # type: ignore[arg-type]
    text = _plain(card_lines(qs, T0, width=56))
    assert f"+{10 - MAX_GPU_LINES} more GPUs" in text


def test_card_waiting_and_disabled() -> None:
    k2 = _states()["k2"]
    k2.snapshot = None
    k2.status = dataclasses.replace(k2.status, state=ConnState.CONNECTING)
    assert "waiting for the first probe" in _plain(card_lines(k2, T0, width=56))
    assert not is_cluster(k2)
    k2.status = dataclasses.replace(k2.status, state=ConnState.DISABLED, error=None)
    assert "polling disabled" in _plain(card_lines(k2, T0, width=56))


def test_degraded_card_shows_section_error() -> None:
    k2 = _states()["k2"]
    err = HostError(ErrorKind.REMOTE, "Problem talking to the database", section="slurm.sacct")
    sections = dict(k2.status.sections) | {"slurm.sacct": SectionStatus("slurm.sacct", "failed", False, 1, 30,
                                                                        None, err)}
    k2.status = dataclasses.replace(k2.status, state=ConnState.DEGRADED, sections=sections)
    text = _plain(card_lines(k2, T0, width=60))
    assert "! slurm.sacct: Problem talking to the database" in text


def test_fleet_table_rows() -> None:
    states = _states()
    row = [t.plain for t in host_row(states["labstation"], 4, T0, T)]
    assert row[2] == "labstation" and row[3] == "ws" and row[5] == "1 task" and row[6] == "2/4"
    k2 = [t.plain for t in host_row(states["k2"], 1, T0, T)]
    assert k2[5] == "R4 PD2" and k2[6] == "5"                      # the 2-day-old failure is outside 24 h
    uc = [t.plain for t in host_row(states["lynx"], 5, T0, T)]
    assert uc[0] == "~" and uc[4] == "✖ auth"
    assert columns_for(120) == 2 and columns_for(200) == 3


def test_old_gpu_reading_is_never_counted_free() -> None:
    """UI-04: a host that answers while nvidia-smi hangs keeps its last GPU readings; an idle-looking but old reading
    must not count as a free GPU on the Fleet row, and its card line is dimmed."""
    qs = _states()["labstation"]
    snap = qs.snapshot
    assert snap is not None and not qs.stale(T0)
    assert [t.plain for t in host_row(qs, 4, T0, T)][6] == "2/4"
    idle = next(g.index for g in snap.gpus if not g.processes and (g.util_pct or 0) < 5)
    old = T0 - timedelta(minutes=10)
    gpus = tuple(dataclasses.replace(g, sampled_at=old if g.index == idle else T0) for g in snap.gpus)
    qs.snapshot = dataclasses.replace(snap, gpus=gpus)
    assert [t.plain for t in host_row(qs, 4, T0, T)][6] == "3/4~"
    line = next(t for t in card_lines(qs, T0, width=56) if t.plain.startswith(f"{idle} "))
    assert any(theme.dim() in str(span.style) for span in line.spans)


# ---------------------------------------------------------------- jobs table
def test_layout_drops_by_priority_and_caps_name() -> None:
    wanted = ("host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "node_reason")
    wide = dict(layout(118, wanted, show_user=False))
    assert set(wide) == {"marker", *wanted}
    assert wide["name"] <= 41 and wide["node_reason"] >= 25           # "never: dependency failed" fits at 120
    narrow = dict(layout(78, wanted, show_user=False))
    # critic: at 80 columns the reason / hint column outranks LEFT and the GPU type
    assert "left" not in narrow and "gpus" not in narrow and {"node_reason", "gpu_util"} <= set(narrow)
    assert sum(narrow.values()) <= 78 and narrow["name"] >= 17
    tiny = dict(layout(50, wanted, show_user=False))
    assert {"host", "id", "name", "state", "elapsed"} <= set(tiny)
    assert "user" in dict(layout(118, wanted, show_user=True))


def test_fit_columns() -> None:
    cols = (("a", "A", 10), ("b", "B", 10), ("c", "C", 10))
    assert fit_columns(cols, 25, drop=("b",), flex="c") == [("a", "A", 10), ("c", "C", 15)]
    assert fit_columns(cols, 40, flex="a")[0][2] == 20


def test_task_node_reason_never_says_cpu_only_when_gpus_are_held() -> None:
    from runfleet.models import JobKind, NodeAlloc
    base = slurm_job("lynx", "p1-2", "train", JobState.RUNNING)
    task = dataclasses.replace(base, kind=JobKind.TASK, alloc=dataclasses.replace(base.alloc, gpus=2, per_node=()))
    assert node_reason(task, _ctx()).plain == "2 GPUs"
    idx = dataclasses.replace(task, alloc=dataclasses.replace(task.alloc, per_node=(
        NodeAlloc("lynx", None, None, (0, 1), None),)))
    assert node_reason(idx, _ctx()).plain == "gpu 0,1"
    cpu = dataclasses.replace(task, alloc=dataclasses.replace(task.alloc, gpus=0))
    assert node_reason(cpu, _ctx()).plain == "cpu only"


def test_node_reason_uses_hints_once() -> None:
    pd = slurm_job("k2", "1", "x", JobState.PENDING, reason="Priority")
    assert node_reason(dataclasses.replace(pd, hints=("waiting for priority",)), _ctx()).plain == "priority"
    odd = dataclasses.replace(pd, reason="JobArrayTaskLimit", hints=("array task limit",))
    assert node_reason(odd, _ctx()).plain == "array task limit"
    never = dataclasses.replace(pd, reason="DependencyNeverSatisfied", hints=("will never run: dependency failed",))
    assert node_reason(never, _ctx()).plain == "never: dependency failed"
    ended = slurm_job("k2", "2", "y", JobState.FAILED, ended_ago=60, exit_code=2, node="n1",
                      hints=("exit 2",))
    assert node_reason(ended, _ctx()).plain == "exit 2 · n1"
    running = slurm_job("k2", "3", "z", JobState.RUNNING, node="n-40-01", hints=("log idle 47m",))
    assert node_reason(running, _ctx(problem_hosts={"k2": "auth"})).plain == "log idle 47m · n-40-01 · auth"
    held = dataclasses.replace(pd, held=True, est_start=True, start_time=T0 + timedelta(minutes=30))
    assert node_reason(held, _ctx()).plain == "held · est 11:12"


def test_job_cells_markers_and_stale_dimming() -> None:
    job = slurm_job("k2", "2332861", "orb-swp--train--cifar-warmup--s43--a02", JobState.RUNNING,
                    elapsed=timedelta(hours=4, minutes=7, seconds=12), util=98, node="n-40-01")
    cols = layout(118, ("host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "node_reason"),
                  show_user=False)
    cells = {k: c for (k, _w), c in zip(cols, job_cells(job, cols, _ctx(), cursor=True), strict=True)}
    assert cells["marker"].plain == "▸ " and cells["state"].plain == "R" and cells["elapsed"].plain == "4:07:12"
    assert cells["gpus"].plain == "1×A100" and cells["gpu_util"].plain.endswith("98%")
    marked = job_cells(job, cols, _ctx(marked=frozenset({"k2:2332861"})), cursor=False)
    assert marked[0].plain == " +"
    stale = job_cells(job, cols, _ctx(stale_hosts=frozenset({"k2"})), cursor=False)
    assert stale[0].plain == " ~"
    assert any(theme.DIM in str(span.style) for span in stale[2].spans)


# ---------------------------------------------------------------- preview
def test_preview_pieces() -> None:
    job = _states()["k2"].snapshot.jobs["2332861"]                   # type: ignore[union-attr]
    meta = meta_line(job).plain
    assert "partition gpu-preempt" in meta and "A100-SXM4-40GB idx 2" in meta and "8 cpu" in meta
    gl = gpu_line(job, None, T0, T, M).plain
    assert gl.startswith("GPU (sstat · 30s)") and "98%" in gl and "24.3/40.0G" in gl
    pending = slurm_job("k2", "9", "p", JobState.PENDING)
    assert gpu_line(pending, None, T0, T, M).plain == "GPU  n/a — job pending"
    peek = LogPeek(("a", "b", "c"), 663, T0 - timedelta(seconds=12), T0)
    lines = log_lines(peek, False, 2, 60, T0, True)
    assert [x.plain.split()[1] for x in lines] == ["b", "c"] and lines[-1].plain.rstrip().endswith("log 663 B · 12s ago")
    assert "after 1 s" in log_lines(None, False, 3, 60, T0, True)[0].plain
    assert "loading" in log_lines(None, True, 3, 60, T0, True)[0].plain
    assert "no log path" in log_lines(None, False, 3, 60, T0, False)[0].plain
    err = LogPeek((), None, None, T0, error="permission denied")
    assert "permission denied" in log_lines(err, False, 3, 60, T0, True)[0].plain


def test_wrap_parts() -> None:
    out = [t.plain for t in wrap_parts(["partition gpu-preempt", "qos normal", "account campus01"], 34)]
    assert out == ["partition gpu-preempt · qos normal", "account campus01"]


# ---------------------------------------------------------------- GPU panel / GPUs view
def test_gpu_panel_wide_and_narrow() -> None:
    gpus = _states()["labstation"].snapshot.gpus                     # type: ignore[union-attr]
    wide = gpu_panel_lines(gpus, 116, T, M, me="alice07")
    assert "PROCS" in wide[0].plain and "alice07/363948" in wide[4].plain and "287/300W" in wide[4].plain
    assert all(t.cell_len <= 116 for t in wide)
    narrow = gpu_panel_lines(gpus, 76, T, M)
    assert "PROCS" not in narrow[0].plain and "PWR" not in narrow[0].plain
    assert all(t.cell_len <= 76 for t in narrow) and "21.0 / 48.0G" in narrow[4].plain
    hidden = gpu_panel_lines(gpus, 116, T, M, show_others=False, me="alice07")
    assert "1 other user" in hidden[2].plain
    assert "no GPUs reported" in gpu_panel_lines((), 100, T, M)[1].plain


def test_gpus_view_rows() -> None:
    states = _states()
    jobs = {j.key: j for j in states["labstation"].snapshot.jobs.values()}   # type: ignore[union-attr]
    gpu3 = ws_row(states["labstation"].snapshot.gpus[3], False, jobs)        # type: ignore[union-attr]
    assert gpu3.owner_text == "p363942 train" and gpu3.owner is not None
    gpu1 = ws_row(states["labstation"].snapshot.gpus[1], False, jobs)        # type: ignore[union-attr]
    assert gpu1.owner_text == "alice eval"
    idle = ws_row(states["labstation"].snapshot.gpus[0], False, jobs)        # type: ignore[union-attr]
    assert idle.owner is None and idle.owner_text == "—"
    agg = job_row(states["k2"].snapshot.jobs["2333760"], False)              # type: ignore[union-attr]
    assert agg.idx == "0-1" and agg.name == "H200 NVL ×2" and agg.aggregate
    assert task_short_name("orb.launch ×2 (orb.train)") == "train"
    assert task_short_name("train.py") == "train"


# ---------------------------------------------------------------- nodes
def test_node_helpers() -> None:
    k2 = _states()["k2"]
    planned = next(n for n in k2.snapshot.nodes if n.name == "n-31-05")      # type: ignore[union-attr]
    assert gpu_totals(planned) == (0, 12, "A100-3g.40gb")          # humanized like every other view
    mig = dataclasses.replace(planned, gres=planned.gres + (dataclasses.replace(planned.gres[0],
                                                                                 type="nvidia_a100_1g.10gb"),))
    assert "MIG (2 types)" in gpu_totals(mig)[2]
    assert states_text({"mixed": 19, "allocated": 3}) == "mixed 19 · alloc 3"


# ---------------------------------------------------------------- dialogs
def _plan(n: int, skipped: int = 0) -> ActionPlan:
    spec = ActionSpec("slurm.cancel", "Cancel job", True, True)
    targets = tuple(JobKey("k2", str(2332452 + i)) for i in range(n))
    skip = tuple((JobKey("k2", f"9{i}"), "job already ended") for i in range(skipped))
    return ActionPlan(spec, "k2", targets, skip, {}, (), "scancel " + " ".join(k.id for k in targets),
                      f"Cancel {n} jobs on k2?", "2332452  a  R\nthis cannot be undone")


def test_plan_body_lists_at_most_eight() -> None:
    body = plan_body(_plan(11, skipped=1), {}, T0, 70, multi_host=False).plain
    assert body.count("(no longer known)") == 8 and "+3 more" in body
    assert "skipped 90: job already ended" in body
    assert "this cannot be undone" in body and "2332452  a  R" not in body   # target lines come from the UI
    assert "will run on k2:  scancel 2332452" in body
    assert dialog_title([_plan(2)]) == "Cancel 2 jobs on k2?"
    other = dataclasses.replace(_plan(1), host="vega")
    assert dialog_title([_plan(2), other]) == "Cancel 3 targets on 2 hosts?"


def test_help_render_filters() -> None:
    text = render_help(["jobs", "main", "global"], "", 100).plain
    assert "NAVIGATION" in text and "SELECTION & ACTIONS" in text and "VIEW" in text and "mark / unmark" in text
    only = render_help(["jobs", "main", "global"], "palette", 100).plain
    assert "command palette" in only and "mark / unmark" not in only
    assert "no key matches" in render_help(["help"], "zzz", 100).plain


def test_error_report_and_doctor(tmp_path: Path) -> None:
    uc = _states()["lynx"]
    log = tmp_path / "ssh.log"
    log.write_text("debug1: line\nPermission denied (publickey).\n")
    uc.status = dataclasses.replace(uc.status, ssh_log=str(log))
    text = error_report(uc, T0, 90).plain
    assert "AUTH  Permission denied" in text and "Press c (Connect)" in text
    assert "ssh log · " in text and "Permission denied (publickey)." in text
    report = DoctorReport("k2", (DoctorCheck("mux", True, "user"), DoctorCheck("probe", False, "timeout"),
                                 DoctorCheck("gpu_tres", None, "unknown")), ("no GPU accounting",))
    doc = doctor_text(report).plain
    assert "✓ mux" in doc and "✖ probe" in doc and "i gpu_tres" in doc and "! no GPU accounting" in doc


def test_host_detail_lines() -> None:
    states = _states()
    qs = states["labstation"]
    line = host_metrics_line(qs, 116, M).plain
    assert "CPU" in line and "of 64" in line and "45/503G" in line and "up 117d" in line
    conn = _plain(connection_lines(qs, T0, 116))
    assert "ssh labstation" in conn and "sections ok 5/5" in conn and "process+nvidia+procfs" in conn
    caps = _plain(capability_lines(states["k2"], 116))
    assert "GPU accounting ✓" in caps and "PreemptMode REQUEUE" in caps and "labpi*" in caps
    lyra = _plain(connection_lines(states["lyra"], T0, 116))
    assert "✖ jump: ssh: Connection timed out" in lyra
    jobs = list(states["k2"].snapshot.jobs.values())                         # type: ignore[union-attr]
    assert "2334564 PD (priority)" in my_jobs_line(jobs, 116).plain


def test_job_detail_lines() -> None:
    states = _states()
    job = states["k2"].snapshot.jobs["2332861"]                             # type: ignore[union-attr]
    title = title_line(job, T0, 118).plain
    assert "k2:2332861" in title and "RUNNING  4:07:12 of 3-00:00:00 (2-19:52 left)" in title
    failed = states["k2"].snapshot.jobs["2330539"]                          # type: ignore[union-attr]
    assert "FAILED  exit 1:0 · ran 0:59:05 · ended 2d ago" in title_line(failed, T0, 118).plain
    box = _plain(job_box_lines(job, None, T0, 56))
    assert "gpu-priority" in box and "waited" in box and "ends by" in box
    res = _plain(resources_lines(job, None, 60))
    assert "IDX 2" in res and "cpus 8 (64-71)" in res
    gl = _plain(gpu_lines(job, None, True, 60, T, M))
    assert "98%" in gl and "(peak 99%)" in gl and "24.3 / 40.0G" in gl
    lyra = states["lyra"].snapshot.jobs["31327454"]                         # type: ignore[union-attr]
    running = dataclasses.replace(lyra, state=JobState.RUNNING, end_time=None, gpu=None)
    assert "no GPU accounting" in _plain(gpu_lines(running, None, False, 60, T, M))
    assert stale_banner(states["k2"], T0) is None
    banner = stale_banner(states["lyra"], T0)
    assert banner is not None and "data from 4m ago — lyra is retrying" in banner.plain
    assert "next retry in 38s" in banner.plain


def test_gpu_dataclass_sanity() -> None:
    # guard: the fake world keeps the shapes the renderers expect
    g: GPU = _states()["lynx"].snapshot.gpus[0]                             # type: ignore[union-attr]
    assert g.processes and g.processes[0].job is not None


def test_raw_tab_dedupes_keys_and_humanizes_epochs() -> None:
    """Leftover #5 / review: Raw showed both JobID and JobId, and times as epoch seconds."""
    from runfleet.models import JobDetail
    from runfleet.ui.screens.job_detail import raw_text

    job = dataclasses.replace(slurm_job("k2", "2332863", "x", JobState.RUNNING),
                              raw={"JobID": "2332863", "SubmitTime": str(int(T0.timestamp()) - 3600), "QOS": "n"})
    detail = JobDetail(job=job, fields={"JobId": "2332863", "EligibleTime": str(int(T0.timestamp()) - 60)},
                       per_node=(), steps=(), logs=(), gpu=None, procs=(), procs_note="", env={}, explanation=(),
                       fetched_at=T0, errors=())
    text = raw_text(job, detail, T0).plain
    assert "JobId" in text and "JobID" not in text and "QOS" in text
    assert "(1h" in text or "1h ago" in text or "60m" in text
    assert str(int(T0.timestamp()) - 60) in text                       # raw epoch kept (dim)


def test_card_memory_bar_is_capacity_not_bandwidth() -> None:
    """M3 F7: utilization.memory (1 %) drew an empty bar next to '23.1/48.0G'."""
    from runfleet.ui.widgets.host_card import gpu_line
    gpu = GPU(host="lbs", index=0, uuid="u", name="NVIDIA RTX A6000", util_pct=88.0, mem_used_mib=23654.0,
              mem_total_mib=49140.0, mem_util_pct=1.0, temp_c=60.0, power_w=None, power_limit_w=None, fan_pct=None,
              pstate=None, source="nvidia-smi")
    line = gpu_line(gpu, 70, T, M).plain
    bars = line.split("▕")
    assert len(bars) >= 3 and bars[2].count("█") >= 2
