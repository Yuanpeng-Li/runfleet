"""Resubmission links (§5.2.12): golden pairs from the field fixtures, and the cases that must never link."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from runfleet.models import Capabilities, Job, JobKey, JobKind, JobState, PartitionCharge
from runfleet.util.lineage import CONCURRENT_SLACK, EMPTY, link, split_attempt
from runfleet.util.logscan import cause_of, preempt_mode_for
from tests.backends.slurm.test_states import _history

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)
NAME = "mtl-fgk9--train--gt-block--s45--a01"


def job(jid: str, state: JobState, *, name: str = NAME, host: str = "lyra", submit: float = 0.0,
        start: float | None = None, end: float | None = None, **kw: object) -> Job:
    """Minutes after T0; a terminal state with no `end` ends one minute after it started."""
    at = lambda m: None if m is None else T0 + timedelta(minutes=m)   # noqa: E731
    if state.is_terminal and end is None:
        end = (start if start is not None else submit) + 1
    base = Job(key=JobKey(host, jid), kind=JobKind.SLURM, name=name, user="alice07", state=state,
               raw_state=state.name, job_id_raw=jid, submit_time=at(submit), start_time=at(start),
               end_time=at(end) if state.is_terminal else None, workdir="/pub/alice07/mtl", source="sacct")
    return dataclasses.replace(base, **kw)  # type: ignore[arg-type]


def keys(*ids: str, host: str = "lyra") -> list[JobKey]:
    return [JobKey(host, i) for i in ids]


# ---------------------------------------------------------------- field fixtures
def test_lyra_fixture_preempted_then_resubmitted() -> None:
    hist = _history("lyra")
    lin = link(hist.values())
    pr = JobKey("lyra", "31354789")
    assert lin.next_of(pr) is hist["31358335"] and lin.superseded(pr)
    assert lin.prev_of(JobKey("lyra", "31358335")) is hist["31354789"]
    assert not lin.superseded(JobKey("lyra", "31358335"))               # completed: nothing after it


def test_lyra_fixture_failure_chain_ends_in_the_completed_run() -> None:
    hist = _history("lyra")
    lin = link(hist.values())
    first = JobKey("lyra", "31327454")
    assert [j.key.id for j in lin.chain_after(first)] == ["31327620", "31327915", "31327999"]
    assert lin.latest_of(first) is hist["31327999"]
    # seeds never mix: s49 and s50 of the same variant are different runs
    assert lin.latest_of(JobKey("lyra", "31327456")).key.id == "31328001"   # type: ignore[union-attr]


def test_lyra_fixture_cancelled_attempt_resubmitted() -> None:
    lin = link(_history("lyra").values())
    assert lin.next_of(JobKey("lyra", "31301780")).key.id == "31303091"     # type: ignore[union-attr]


def test_fixture_hosts_never_link_completed_jobs() -> None:
    for host in ("lyra", "k2", "vega"):
        hist = _history(host)
        lin = link(hist.values())
        for k in lin.successor:
            old = hist[k.id]
            assert old.state.is_terminal and old.state is not JobState.COMPLETED          # type: ignore[attr-defined]


# ---------------------------------------------------------------- the rule
def test_preempted_then_same_name_resubmit() -> None:
    old = job("58717472", JobState.PREEMPTED, start=0, end=533)
    new = job("58721477", JobState.COMPLETED, submit=546, start=548, end=983)
    lin = link([new, old])
    assert lin.next_of(old.key) is new and lin.prev_of(new.key) is old and lin.latest_of(old.key) is new


@pytest.mark.parametrize("state", [JobState.RUNNING, JobState.PENDING])
def test_successor_may_still_be_queued_or_running(state: JobState) -> None:
    old = job("1", JobState.PREEMPTED, start=0, end=60)
    new = job("2", state, submit=70, start=71 if state is JobState.RUNNING else None)
    assert link([old, new]).next_of(old.key) is new


def test_singleton_segment_submitted_before_the_end_links() -> None:
    old = job("1", JobState.TIMEOUT, submit=0, start=1, end=600)
    nxt = job("2", JobState.RUNNING, submit=5, start=600.5)              # queued behind it (singleton), then ran
    assert link([old, nxt]).next_of(old.key) is nxt


def test_concurrent_duplicate_is_not_a_resubmission() -> None:
    old = job("1", JobState.FAILED, submit=0, start=1, end=600)
    dup = job("2", JobState.COMPLETED, submit=2, start=3, end=900)       # ran alongside it
    assert link([old, dup]) is EMPTY
    edge = job("3", JobState.RUNNING, submit=2, start=600 - CONCURRENT_SLACK.total_seconds() / 60)
    assert link([old, edge]).next_of(old.key) is edge


def test_pending_estimate_does_not_count_as_a_start() -> None:
    old = job("1", JobState.FAILED, submit=0, start=1, end=600)
    new = job("2", JobState.PENDING, submit=2, start=30, est_start=True)
    assert link([old, new]).next_of(old.key) is new


def test_attempt_bump_links_but_seed_or_variant_change_does_not() -> None:
    old = job("1", JobState.CANCELLED, name="t--train--v--s43--a01", start=0, end=10)
    a02 = job("2", JobState.RUNNING, name="t--train--v--s43--a02", submit=20, start=21)
    assert link([old, a02]).next_of(old.key) is a02
    other_seed = job("3", JobState.RUNNING, name="t--train--v--s44--a02", submit=20, start=21)
    other_var = job("4", JobState.RUNNING, name="t--train--w--s43--a02", submit=20, start=21)
    lower = job("5", JobState.RUNNING, name="t--train--v--s43--a00", submit=20, start=21)
    assert link([old, other_seed, other_var, lower]) is EMPTY


def test_plain_name_needs_an_exact_match() -> None:
    old = job("1", JobState.FAILED, name="smoke-test", start=0, end=1)
    assert link([old, job("2", JobState.RUNNING, name="smoke-test--a02", submit=5, start=6)]) is EMPTY
    assert link([old, job("3", JobState.RUNNING, name="smoke-test", submit=5, start=6)]).superseded(old.key)


@pytest.mark.parametrize("state", [JobState.COMPLETED, JobState.RUNNING, JobState.PENDING])
def test_only_bad_endings_are_superseded(state: JobState) -> None:
    old = job("1", state, start=0 if state is not JobState.PENDING else None, end=10 if state.is_terminal else None)
    assert link([old, job("2", JobState.RUNNING, submit=20, start=21)]) is EMPTY


def test_other_users_hosts_workdirs_and_kinds_never_link() -> None:
    old = job("1", JobState.PREEMPTED, start=0, end=10)
    new = dict(submit=20, start=21)
    assert link([old, job("2", JobState.RUNNING, user="bob", **new)]) is EMPTY                      # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, is_mine=False, **new)]) is EMPTY                   # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, host="k2", **new)]) is EMPTY                       # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, workdir="/elsewhere", **new)]) is EMPTY            # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, workdir=None, **new)]).superseded(old.key)         # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, workdir="/pub/alice07/mtl/", **new)]).superseded(old.key)  # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, kind=JobKind.TASK, **new)]) is EMPTY              # type: ignore[arg-type]
    assert link([old, job("2", JobState.RUNNING, submit=0, start=21)]) is EMPTY                     # same submit time
    assert link([dataclasses.replace(old, submit_time=None), job("2", JobState.RUNNING, **new)]) is EMPTY  # type: ignore[arg-type]


def test_arrays_link_task_to_same_task_of_another_array_only() -> None:
    t3 = job("10_3", JobState.FAILED, start=0, end=5, array_job_id="10", array_task="3")
    t4 = job("10_4", JobState.COMPLETED, submit=1, start=6, end=9, array_job_id="10", array_task="4")
    assert link([t3, t4]) is EMPTY                                                   # siblings of one array
    again = job("20_3", JobState.RUNNING, submit=10, start=11, array_job_id="20", array_task="3")
    other = job("20_4", JobState.RUNNING, submit=10, start=11, array_job_id="20", array_task="4")
    assert link([t3, other, again]).next_of(t3.key) is again
    single = job("30", JobState.RUNNING, submit=10, start=11)
    assert link([t3, single]) is EMPTY
    summary = job("40_[1-9%2]", JobState.PENDING, submit=10, array_job_id="40", array_task="[1-9%2]")
    assert link([job("41_[1-9]", JobState.FAILED, start=0, end=1, array_job_id="41", array_task="[1-9]"),
                 summary]) is EMPTY


def test_two_failed_duplicates_point_at_one_resubmit() -> None:
    a = job("1", JobState.FAILED, submit=0, start=1, end=50)
    b = job("2", JobState.FAILED, submit=1, start=2, end=50)              # ran alongside a: not a's successor
    c = job("3", JobState.COMPLETED, submit=60, start=61, end=90)
    lin = link([a, b, c])
    assert lin.next_of(a.key) is c and lin.next_of(b.key) is c
    assert lin.prev_of(c.key) is b                                       # the most recently submitted one


def test_earliest_later_job_is_the_successor() -> None:
    old = job("1", JobState.NODE_FAIL, start=0, end=5)
    first = job("2", JobState.FAILED, submit=10, start=11, end=12)
    second = job("3", JobState.RUNNING, submit=20, start=21)
    lin = link([second, old, first])
    assert lin.next_of(old.key) is first and lin.latest_of(old.key) is second
    assert [j.key for j in lin.chain_after(old.key)] == keys("2", "3")


def test_split_attempt() -> None:
    assert split_attempt(NAME) == ("mtl-fgk9--train--gt-block--s45", 1)
    assert split_attempt("x--a12") == ("x", 12)
    assert split_attempt("smoke-test") == ("smoke-test", None)
    assert split_attempt("--a01") == ("--a01", None)                    # nothing before the field


# ---------------------------------------------------------------- PreemptMode-aware advice
def _caps(mode: str | None, parts: tuple[PartitionCharge, ...] = ()) -> Capabilities:
    cfg = {"PreemptMode": mode} if mode else {}
    return Capabilities(remote_user="alice07", uid=1, login_host="l", tz_offset=timedelta(0),
                        clock_skew=timedelta(0), tools=frozenset(), slurm_config=cfg, partition_charges=parts)


def test_preempt_mode_for_prefers_the_partition() -> None:
    assert preempt_mode_for(None, "free-gpu") is None
    assert preempt_mode_for(_caps(None), "free-gpu") is None
    assert preempt_mode_for(_caps("cancel"), "free-gpu") == "CANCEL"
    assert preempt_mode_for(_caps("REQUEUE,GANG"), None) == "REQUEUE"
    parts = (PartitionCharge("free-gpu", preempt_mode="REQUEUE"), PartitionCharge("std"))
    assert preempt_mode_for(_caps("CANCEL", parts), "free-gpu") == "REQUEUE"
    assert preempt_mode_for(_caps("CANCEL", parts), "std") == "CANCEL"


def test_preempted_advice_follows_the_mode() -> None:
    pr = job("1", JobState.PREEMPTED, start=0, end=5)
    assert "resubmit" in cause_of(pr, preempt_mode="CANCEL").text                  # type: ignore[union-attr]
    assert "requeue" not in cause_of(pr, preempt_mode="CANCEL").text.lower()       # type: ignore[union-attr]
    assert "not requeued" in cause_of(pr, preempt_mode="REQUEUE").text             # type: ignore[union-attr]
    assert cause_of(pr).text == "preempted for a higher-priority job — resubmit or requeue; checkpoint often"  # type: ignore[union-attr]
