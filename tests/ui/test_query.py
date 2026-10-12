"""[WP3] ui/query.py: filter language, presets, sorting, derived times."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from runfleet.models import Job, JobState
from runfleet.ui import query
from tests.ui.fakes import T0, build_world, slurm_job


@pytest.fixture(scope="module")
def jobs() -> list[Job]:
    _config, states = build_world()
    return [j for s in states if s.snapshot is not None for j in s.snapshot.jobs.values()]


def _ids(jobs: list[Job]) -> set[str]:
    return {j.key.label for j in jobs}


def test_parse_terms() -> None:
    q = query.parse("lmft host:k2,VEGA !st:pd user:me foo:bar")
    kinds = [(t.kind, t.values, t.negate) for t in q.terms]
    assert kinds == [("text", ("lmft",), False), ("host", ("k2", "vega"), False), ("state", ("pd",), True),
                     ("user", ("me",), False), ("text", ("foo:bar",), False)]
    assert query.parse("   ").empty
    assert query.parse("!").empty
    assert query.parse("h:").terms[0].kind == "text"     # key without value stays plain text


def test_fuzzy_word(jobs: list[Job]) -> None:
    hit = [j for j in jobs if query.matches(query.parse("lmft"), j)]
    assert hit and all("lmft" in j.name for j in hit)
    # separators are ignored in names, but there is no subsequence fuzz (review: `/fail` matched 'orb-swp--train…')
    assert any(query.matches(query.parse("orbswp"), j) for j in jobs)
    assert not any(query.matches(query.parse("lmfttch"), j) for j in jobs)
    assert not any(query.matches(query.parse("zzqx"), j) for j in jobs)
    fail = [j for j in jobs if query.matches(query.parse("fail"), j)]
    assert all("fail" in " ".join([j.name, j.reason or "", j.state.name]).lower() for j in fail)


def test_state_names_and_aliases(jobs: list[Job]) -> None:
    running = [j for j in jobs if query.matches(query.parse("state:running"), j)]
    assert running and all(j.state is JobState.RUNNING for j in running)
    pending = [j for j in jobs if query.matches(query.parse("st:pending"), j)]
    assert pending and all(j.state is JobState.PENDING for j in pending)
    assert query.matches(query.parse("st:oom"), next(j for j in jobs if j.state is JobState.OUT_OF_MEMORY)) \
        if any(j.state is JobState.OUT_OF_MEMORY for j in jobs) else True


def test_host_and_state_terms(jobs: list[Job]) -> None:
    k2 = [j for j in jobs if query.matches(query.parse("h:k2"), j)]
    assert k2 and {j.key.host for j in k2} == {"k2"}
    running = [j for j in jobs if query.matches(query.parse("state:R"), j)]
    assert running and all(j.state is JobState.RUNNING for j in running)
    queued = [j for j in jobs if query.matches(query.parse("st:queued"), j)]
    assert queued and all(j.state.group.value == "queued" for j in queued)
    pd_f = [j for j in jobs if query.matches(query.parse("st:pd,f"), j)]
    assert {j.state for j in pd_f} == {JobState.PENDING, JobState.FAILED}


def test_held_suffix() -> None:
    held = slurm_job("k2", "1", "x", JobState.PENDING, held=True)
    assert query.matches(query.parse("st:pdh"), held)
    assert not query.matches(query.parse("st:pdh"), dataclasses.replace(held, held=False))


def test_negation(jobs: list[Job]) -> None:
    not_running = [j for j in jobs if query.matches(query.parse("!state:R"), j)]
    assert not_running and all(j.state is not JobState.RUNNING for j in not_running)
    assert len(not_running) + sum(1 for j in jobs if j.state is JobState.RUNNING) == len(jobs)


def test_user_terms(jobs: list[Job]) -> None:
    mine = [j for j in jobs if query.matches(query.parse("user:me"), j)]
    assert mine and all(j.is_mine for j in mine)
    alice = [j for j in jobs if query.matches(query.parse("u:alice"), j)]
    assert [j.user for j in alice] == ["alice"]


def test_presets(jobs: list[Job]) -> None:
    window = 86400.0
    fresh = [j for j in jobs if query.preset_matches("fresh", j, T0, window)]
    assert "2330539" not in _ids(fresh)              # failed 2 days ago: outside the 24 h window
    assert "31327454" in _ids(fresh)                 # failed 20 h ago: inside
    failed = [j for j in jobs if query.preset_matches("failed", j, T0, window)]
    assert {"2330539", "31327454", "31354789"} <= _ids(failed)
    ended = dataclasses.replace(jobs[0], state=JobState.EXITED, end_time=T0)
    assert query.preset_matches("failed", ended, T0, window)          # failed preset includes ENDED
    assert all(query.preset_matches("all", j, T0, window) for j in jobs)
    assert all(j.state.group.value == "active" for j in jobs if query.preset_matches("active", j, T0, window))
    assert all(j.state.group.value == "queued" for j in jobs if query.preset_matches("queued", j, T0, window))


def test_next_preset_cycles() -> None:
    seen = ["fresh"]
    for _ in range(5):
        seen.append(query.next_preset(seen[-1]))  # type: ignore[arg-type]
    assert seen == ["fresh", "active", "queued", "failed", "all", "fresh"]


def test_elapsed_extrapolates_active_only() -> None:
    job = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(hours=1))
    assert query.elapsed_s(job, T0 + timedelta(seconds=30)) == 3630
    done = dataclasses.replace(job, state=JobState.COMPLETED, end_time=T0)
    assert query.elapsed_s(done, T0 + timedelta(hours=5)) == 3600
    pend = slurm_job("k2", "2", "y", JobState.PENDING)
    assert query.elapsed_s(pend, T0) is None


def test_left() -> None:
    job = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(hours=1), limit=timedelta(hours=3))
    assert query.left_s(job, T0) == 2 * 3600
    assert query.left_s(job, T0 + timedelta(hours=5)) == 0
    pend = slurm_job("k2", "2", "y", JobState.PENDING, limit=timedelta(hours=2))
    assert query.left_s(pend, T0) == 7200
    assert query.left_s(dataclasses.replace(job, time_limit=None), T0) is None


def test_default_sort_host_then_state_then_submit(jobs: list[Job]) -> None:
    order = ["k2", "vega", "lyra", "labstation", "lynx"]
    out = query.sort_jobs(jobs, "host", False, host_order=order, now=T0)
    hosts = [j.key.host for j in out]
    assert hosts == sorted(hosts, key=order.index)
    k2 = [j for j in out if j.key.host == "k2"]
    groups = [query.GROUP_ORDER[j.state.group] for j in k2]
    assert groups == sorted(groups)
    running = [j for j in k2 if j.state is JobState.RUNNING]
    submits = [j.submit_time for j in running]
    assert submits == sorted(submits, reverse=True)       # newest first
    assert query.sort_jobs(jobs, "host", True, host_order=order, now=T0) == list(reversed(out))


def test_sort_by_gpu_util_puts_unknown_last(jobs: list[Job]) -> None:
    out = query.sort_jobs(jobs, "gpu_util", False, host_order=[], now=T0)
    utils = [j.gpu.util_mean_pct if j.gpu else None for j in out]
    known = [u for u in utils if u is not None]
    assert known == sorted(known, reverse=True)
    assert utils[len(known):] == [None] * (len(utils) - len(known))
    rev = query.sort_jobs(jobs, "gpu_util", True, host_order=[], now=T0)
    rev_known = [j.gpu.util_mean_pct for j in rev if j.gpu and j.gpu.util_mean_pct is not None]
    assert rev_known == sorted(rev_known)
    assert rev[-1].gpu is None or rev[-1].gpu.util_mean_pct is None


@pytest.mark.parametrize("column", ["id", "name", "state", "elapsed", "left", "gpus", "submit", "user"])
def test_every_sort_column_is_total(jobs: list[Job], column: query.SortColumn) -> None:
    out = query.sort_jobs(jobs, column, False, host_order=[], now=T0)
    assert sorted(map(str, (j.key for j in out))) == sorted(map(str, (j.key for j in jobs)))


def test_next_sort() -> None:
    assert query.next_sort("host") == "state"
    assert query.next_sort("submit") == "host"
    assert query.next_sort("user") == "host"
