"""core/events.py: job transitions -> events (baseline, started, ended, restarted, time_low, gpu_idle), selection,
coalescing and the away digest helpers (§7.14)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

from runfleet.config import NotifySettings
from runfleet.core.events import (JobEvent, TransitionDetector, coalesce, count_line, end_summary, ended_since,
                                  time_left_s, time_low, wanted)
from runfleet.models import Job, JobKey, JobKind, JobState

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


def job(jid: str, state: JobState = JobState.RUNNING, *, host: str = "k2", elapsed: float | None = 3600,
        limit: float | None = 86400, restarts: int = 0, mine: bool = True, end: datetime | None = None,
        start: datetime | None = None, hints: tuple[str, ...] = (), kind: JobKind = JobKind.SLURM,
        exit_code: int | None = None, raw: dict[str, str] | None = None, raw_state: str = "") -> Job:
    return Job(key=JobKey(host, jid), kind=kind, name=f"run-{jid}", user="me" if mine else "bob", state=state,
               raw_state=raw_state or state.name, job_id_raw=jid,
               elapsed=timedelta(seconds=elapsed) if elapsed is not None else None,
               time_limit=timedelta(seconds=limit) if limit is not None else None,
               start_time=start if start is not None else (T0 - timedelta(seconds=elapsed or 0)),
               end_time=end, restarts=restarts, is_mine=mine, hints=hints, observed_at=T0, exit_code=exit_code,
               nodelist="n-31-01", raw=raw or {})


def jobs(*items: Job) -> dict[str, Job]:
    return {j.key.id: j for j in items}


def baseline(*items: Job, now: datetime = T0) -> TransitionDetector:
    det = TransitionDetector()
    assert det.observe("k2", jobs(*items), now) == []
    return det


def kinds(events: list[JobEvent]) -> list[tuple[str, str]]:
    return [(e.kind, e.key.id) for e in events]


# ---------------------------------------------------------------------------------------------------- baseline
def test_first_observation_is_a_baseline_even_with_ended_and_low_jobs() -> None:
    low = job("1", elapsed=86000)                                   # 400 s left: low already
    idle = job("2", hints=("GPU idle 20m",))
    done = job("3", JobState.FAILED, end=T0 - timedelta(minutes=5))
    det = baseline(low, idle, done)
    assert det.known("k2") and not det.known("vega")
    assert det.observe("k2", jobs(low, idle, done), T0 + timedelta(seconds=30)) == []   # nothing re-fires


def test_reset_makes_the_next_observation_a_baseline_again() -> None:
    det = baseline(job("1"))
    det.reset("k2")
    assert det.observe("k2", jobs(job("1", JobState.FAILED, end=T0)), T0) == []


# ---------------------------------------------------------------------------------------------------- transitions
def test_started_ended_and_others_jobs_ignored() -> None:
    pd = job("1", JobState.PENDING, elapsed=None, start=T0 + timedelta(hours=1))
    run = job("2")
    other = job("9", mine=False)
    det = baseline(pd, run, other)
    later = T0 + timedelta(minutes=1)
    out = det.observe("k2", jobs(job("1", start=later, elapsed=0), job("2", JobState.FAILED, end=later, exit_code=1),
                                 job("9", JobState.FAILED, mine=False, end=later)), later)
    assert kinds(out) == [("started", "1"), ("ended", "2")]
    assert out[1].category == "ended_bad" and out[1].summary == "exit 1" and out[0].summary == "on n-31-01"


def test_requeue_is_a_restart_and_restart_count_up_too() -> None:
    det = baseline(job("1"), job("2", restarts=1))
    later = T0 + timedelta(minutes=1)
    requeued = job("1", JobState.PENDING, elapsed=None, start=later + timedelta(minutes=2))
    out = det.observe("k2", jobs(requeued, job("2", restarts=2)), later)
    assert kinds(out) == [("restarted", "1"), ("restarted", "2")]
    assert out[1].summary == "↻2" and out[0].summary == "requeued"


def test_new_jobs_count_only_when_they_happened_after_the_baseline() -> None:
    det = baseline()
    later = T0 + timedelta(minutes=10)
    history = job("1", JobState.COMPLETED, end=T0 - timedelta(hours=3))           # sacct arriving a tick late
    fresh = job("2", JobState.TIMEOUT, end=T0 + timedelta(minutes=5), raw={"reason_text": "hit time limit"})
    started = job("3", start=T0 + timedelta(minutes=8), elapsed=120)
    old_run = job("4", start=T0 - timedelta(hours=5), elapsed=5 * 3600)
    out = det.observe("k2", jobs(history, fresh, started, old_run), later)
    assert kinds(out) == [("ended", "2"), ("started", "3")]
    assert out[0].summary == "hit time limit" and out[0].word == "timeout"


def test_time_low_fires_once_per_run() -> None:
    limit = 2 * 3600.0                                              # threshold max(12 min, 10 min)
    j = job("1", elapsed=limit - 13 * 60, limit=limit)
    det = baseline(j)
    assert det.observe("k2", jobs(j), T0 + timedelta(seconds=30)) == []
    out = det.observe("k2", jobs(j), T0 + timedelta(minutes=2))
    assert kinds(out) == [("time_low", "1")] and out[0].summary.endswith("left of 2:00:00")
    assert det.observe("k2", jobs(j), T0 + timedelta(minutes=3)) == []
    rerun = dataclasses.replace(j, restarts=1, observed_at=T0 + timedelta(minutes=3))   # requeued: a new run
    out = det.observe("k2", jobs(rerun), T0 + timedelta(minutes=5))
    assert ("time_low", "1") in kinds(out)


def test_gpu_idle_fires_when_first_raised() -> None:
    det = baseline(job("1"))
    out = det.observe("k2", jobs(job("1", hints=("log idle 31m", "GPU idle 15m"))), T0)
    assert kinds(out) == [("gpu_idle", "1")] and out[0].summary == "GPU idle 15m"
    assert det.observe("k2", jobs(job("1", hints=("GPU idle 16m",))), T0) == []


def test_task_ends_with_its_log_line() -> None:
    task = job("p1-2", kind=JobKind.TASK, limit=None)
    det = baseline(task)
    ended = job("p1-2", JobState.FAILED, kind=JobKind.TASK, limit=None, end=T0,
                raw_state="exited (log: torch.OutOfMemoryError: CUDA out of memory)")
    out = det.observe("k2", jobs(ended), T0)
    assert out[0].summary.startswith("exited (log: torch.OutOfMemoryError")


def test_task_end_waits_for_the_log_scan_verdict() -> None:
    from runfleet.util.logscan import PENDING_EXIT_HINT

    end = T0 + timedelta(seconds=5)

    def task(jid: str, state: JobState = JobState.RUNNING, hint: str = "") -> Job:
        return job(jid, state, kind=JobKind.TASK, limit=None, end=None if hint == "" else end,
                   hints=(hint,) if hint else ())

    det = baseline(task("p1-2"), task("p3-4"))
    assert det.observe("k2", jobs(task("p1-2", JobState.EXITED, PENDING_EXIT_HINT),
                                  task("p3-4", JobState.EXITED, PENDING_EXIT_HINT)), end) == []
    failed = task("p1-2", JobState.FAILED, "log: Traceback")
    unknown = task("p3-4", JobState.EXITED, "ended, exit status unknown")
    out = det.observe("k2", jobs(failed, unknown), T0 + timedelta(seconds=10))
    assert [(e.kind, e.key.id, e.category) for e in out] == [("ended", "p1-2", "ended_bad"),
                                                             ("ended", "p3-4", "ended_other")]
    assert det.observe("k2", jobs(failed, unknown), T0 + timedelta(seconds=15)) == []   # told once


# ---------------------------------------------------------------------------------------------------- helpers
def test_time_left_and_low_threshold() -> None:
    assert time_left_s(job("1", elapsed=3000, limit=3600), T0 + timedelta(seconds=100)) == 500
    assert time_left_s(job("1", JobState.PENDING), T0) is None
    assert time_low(job("1", elapsed=3600 - 599, limit=3600), T0)                  # < 10 min
    assert not time_low(job("1", elapsed=3600 - 601, limit=3600), T0)
    assert time_low(job("1", elapsed=86400 - 8000, limit=86400), T0)               # < 10 % of a day
    assert not time_low(job("1", limit=None), T0)


def test_end_summary_prefers_the_collectors_reason() -> None:
    assert end_summary(job("1", JobState.COMPLETED, exit_code=0)) == ""
    assert end_summary(job("1", JobState.FAILED, exit_code=2)) == "exit 2"
    assert end_summary(job("1", JobState.OUT_OF_MEMORY, hints=("out of memory",))) == "out of memory"


def test_selection_by_category_and_watch() -> None:
    s = NotifySettings()
    failed = JobEvent("ended", JobKey("k2", "1"), "x", JobState.FAILED, T0)
    done = JobEvent("ended", JobKey("k2", "1"), "x", JobState.COMPLETED, T0)
    started = JobEvent("started", JobKey("k2", "1"), "x", JobState.RUNNING, T0)
    cancelled = JobEvent("ended", JobKey("k2", "1"), "x", JobState.CANCELLED, T0)
    assert wanted(failed, s, watched=False) and not wanted(done, s, watched=False)
    assert wanted(done, s, watched=True) and wanted(started, s, watched=True) and not wanted(started, s,
                                                                                               watched=False)
    assert cancelled.category == "ended_other" and wanted(cancelled, s, watched=True)
    assert not wanted(failed, NotifySettings(events=()), watched=False)


def test_bursts_coalesce_per_host() -> None:
    evs = [JobEvent("ended", JobKey("k2", str(i)), f"run-{i}", JobState.FAILED, T0) for i in range(12)]
    evs += [JobEvent("ended", JobKey("k2", f"c{i}"), "ok", JobState.COMPLETED, T0) for i in range(3)]
    evs.append(JobEvent("time_low", JobKey("vega", "7"), "long", JobState.RUNNING, T0, "8m left of 2:00:00"))
    notices = coalesce(evs)
    assert [n.text for n in notices] == ["k2: 12 failed, 3 completed", "vega:7 long low on time: 8m left of 2:00:00"]
    assert notices[0].summary and notices[0].bad and not notices[1].summary


def test_count_line_puts_bad_news_first() -> None:
    assert count_line(["completed", "failed", "timeout", "completed"]) == "1 failed · 1 timeout · 2 completed"


def test_ended_since_newest_first_own_only() -> None:
    a = job("1", JobState.COMPLETED, end=T0 - timedelta(hours=1))
    b = job("2", JobState.FAILED, end=T0 - timedelta(minutes=5))
    c = job("3", JobState.FAILED, end=T0 - timedelta(hours=9))
    d = job("4", JobState.FAILED, end=T0, mine=False)
    e = job("5")
    assert [j.key.id for j in ended_since([a, b, c, d, e], T0 - timedelta(hours=2))] == ["2", "1"]
