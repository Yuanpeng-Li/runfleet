"""[WP3] Every user's jobs on a node (§5.2.9): the node table's JOBS cell and the host detail panel."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

from runfleet.models import JobState, Node, NodeJob
from runfleet.ui.widgets.node_table import jobs_cell, occupant_lines
from tests.ui.fakes import ME, T0, build_world, slurm_job


def _k2_nodes() -> dict[str, Node]:
    _cfg, states = build_world()
    k2 = next(s for s in states if s.config.name == "k2")
    assert k2.snapshot is not None
    return {n.name: n for n in k2.snapshot.nodes}


def _job(jid: str, user: str, gpus: int | None, left_h: float, *, exact: bool = True) -> NodeJob:
    return NodeJob(jid, user, None, "lab", "gpu-preempt", "train", "R", gpus, exact, None, 3600,
                   3600 + int(left_h * 3600), user == ME)


def test_jobs_cell_lists_users_by_gpus_and_keeps_own_first() -> None:
    nodes = _k2_nodes()
    assert jobs_cell(nodes["n-31-01"], [], 60, T0).plain == "bob×4 · carol×3 · dave×1"
    assert jobs_cell(nodes["n-32-03"], [], 60, T0).plain == "bob×4 · grace×~4"           # multi-node split: ~
    assert jobs_cell(nodes["n-38-01"], [], 60, T0).plain == "2333760 · heidi×2"           # own job id first
    assert jobs_cell(nodes["n-31-05"], [], 60, T0).plain == "—"
    assert jobs_cell(nodes["n-31-01"], [], 20, T0).plain == "bob×4 · carol×3 +1"         # what does not fit: +N
    node = dataclasses.replace(nodes["n-31-01"], jobs=(_job("1", "zed", 0, 1), _job("2", "amy", None, 1),
                                                       _job("3", "bob", 2, 1), _job("4", "bob", 1, 2)))
    assert jobs_cell(node, [], 60, T0).plain == "bob×3 · amy×? · zed"                     # per user; CPU job: name only


def test_jobs_cell_without_a_reading_shows_own_jobs() -> None:
    node = dataclasses.replace(_k2_nodes()["n-40-01"], jobs=None, jobs_at=None)
    mine = [slurm_job("k2", j, "t", JobState.RUNNING, node="n-40-01") for j in ("2332861", "2332862")]
    assert jobs_cell(node, mine, 40, T0).plain == "2332861 +1"
    assert jobs_cell(node, [], 40, T0).plain == "—"


def test_occupant_panel_rows_sorted_by_time_left_and_subtitle() -> None:
    nodes = _k2_nodes()
    lines, sub = occupant_lines(nodes["n-31-01"], 120, T0)
    text = [t.plain for t in lines]
    assert text[0].split()[:3] == ["JOB", "USER", "NAME"]
    assert [row.split()[1] for row in text[1:]] == ["carol", "dave", "bob"]           # ends soonest first
    assert "Carol Example" in text[1] and "2:17:30" in text[1]
    assert sub == "next GPU frees in ≤ 2h17m"                                         # full node
    _lines, sub = occupant_lines(nodes["n-40-01"], 120, T0)
    assert sub == "1 GPU free now"
    later, sub = occupant_lines(nodes["n-31-01"], 120, T0 + timedelta(minutes=2))   # aged reading: times move on
    assert "2:15:30" in later[1].plain and sub.endswith("as of 2m ago")
    expired = T0 + timedelta(minutes=3, seconds=1)                                    # past NODE_JOBS_TTL: not shown
    assert occupant_lines(nodes["n-31-01"], 120, expired)[0][0].plain == "reading who runs on n-31-01…"
    assert jobs_cell(nodes["n-31-01"], [], 60, expired).plain == "—"
    split = dataclasses.replace(nodes["n-31-01"], jobs=(_job("9", "zed", None, 1, exact=False),))
    assert occupant_lines(split, 120, T0)[1] == "next GPU frees in ≤ 1h"              # unknown split still holds GPUs
    cpu = dataclasses.replace(nodes["n-31-01"], jobs=(_job("9", "zed", 0, 1),))
    assert occupant_lines(cpu, 120, T0)[1] == ""


def test_occupant_panel_states_and_narrow_widths() -> None:
    nodes = _k2_nodes()
    assert occupant_lines(None, 100, T0)[0][0].plain == "no node selected"
    unread = dataclasses.replace(nodes["n-31-01"], jobs=None)
    assert occupant_lines(unread, 100, T0)[0][0].plain == "reading who runs on n-31-01…"
    assert occupant_lines(nodes["n-31-05"], 100, T0)[0][0].plain == "no running jobs on n-31-05"
    hidden, _ = occupant_lines(nodes["n-38-01"], 100, T0, hidden=True)
    assert "PrivateData=jobs" in hidden[-1].plain
    narrow, _ = occupant_lines(nodes["n-31-01"], 60, T0)
    assert all(t.cell_len <= 60 for t in narrow)
    assert "ACCOUNT" not in narrow[0].plain and "JOB" in narrow[0].plain and "LEFT" in narrow[0].plain
    many = dataclasses.replace(nodes["n-31-01"], jobs=tuple(_job(str(i), f"u{i}", 1, i) for i in range(11)))
    lines, _ = occupant_lines(many, 120, T0)
    assert len(lines) == 1 + 8 + 1 and lines[-1].plain == "+3 more"
