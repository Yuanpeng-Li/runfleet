"""core/events.py: job transitions -> events (baseline, started, ended, restarted, time_low, gpu_idle, resubmitted),
selection, coalescing and the away digest helpers (§7.14)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

from runfleet.config import NotifySettings
from runfleet.core.events import (CATEGORIES, JobEvent, TransitionDetector, coalesce, count_line, end_summary,
                                  ended_since, resubmitted_note, resumes_note, time_left_s, time_low, wanted)
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


# ---------------------------------------------------------------------------------------------------- resubmitted
def seg(jid: str, state: JobState, *, submit: float, name: str = "orb-swp--train--s47--a01",
        end: datetime | None = None, elapsed: float | None = 600) -> Job:
    """One run of a resubmitted job (same name and workdir), submitted ``submit`` seconds before T0."""
    return dataclasses.replace(job(jid, state, end=end, elapsed=elapsed, raw_state=state.name), name=name,
                               submit_time=T0 - timedelta(seconds=submit), workdir="/scratch/me/orb")


PREEMPTED = seg("58717472", JobState.PREEMPTED, submit=9 * 3600, end=T0 - timedelta(minutes=13))


def test_a_new_job_that_resumes_a_bad_ending_fires_once() -> None:
    det = baseline(PREEMPTED)
    later = T0 + timedelta(minutes=1)
    new = seg("58721477", JobState.PENDING, submit=-30, elapsed=None)
    out = det.observe("k2", jobs(PREEMPTED, new), later)
    assert kinds(out) == [("resubmitted", "58721477")]
    e = out[0]
    assert (e.category, e.word, e.bad, e.summary) == ("resubmitted", "resubmitted", False,
                                                      "resumes 58717472 (PREEMPTED)")
    assert e.text() == "k2:58721477 orb-swp--train--s47--a01 resubmitted: resumes 58717472 (PREEMPTED)"
    started = dataclasses.replace(new, state=JobState.RUNNING, start_time=later)
    assert kinds(det.observe("k2", jobs(PREEMPTED, started), later + timedelta(minutes=1))) == [
        ("started", "58721477")]                                         # once: never again while it is listed


def test_resubmissions_already_there_at_the_baseline_are_not_news() -> None:
    new = seg("58721477", JobState.RUNNING, submit=600, elapsed=300)
    det = baseline(PREEMPTED, new)
    assert det.observe("k2", jobs(PREEMPTED, new), T0 + timedelta(minutes=1)) == []


def test_a_bad_ending_seen_after_its_resubmission_still_fires_and_says_so() -> None:
    """Late sacct (or a chained segment queued ahead): the failure is never hidden; its successor, already listed,
    is told to resume it in the same batch."""
    new = seg("58721477", JobState.RUNNING, submit=30, elapsed=10)
    det = baseline(new)
    late = dataclasses.replace(PREEMPTED, end_time=T0 - timedelta(seconds=60))
    out = det.observe("k2", jobs(late, new), T0 + timedelta(seconds=30))
    assert kinds(out) == [("ended", "58717472"), ("resubmitted", "58721477")]
    assert out[0].category == "ended_bad" and out[0].summary == "(already resubmitted as 58721477)"
    failed = seg("9", JobState.RUNNING, submit=7200, name="orb-eval--s45--a01", elapsed=3600)
    queued = seg("10", JobState.PENDING, submit=600, name="orb-eval--s45--a01", elapsed=None)   # singleton, ahead
    det = baseline(failed, queued)
    crash = dataclasses.replace(failed, state=JobState.FAILED, end_time=T0, exit_code=1)
    out = det.observe("k2", jobs(crash, queued), T0 + timedelta(seconds=30))
    assert kinds(out) == [("ended", "9"), ("resubmitted", "10")]
    assert out[0].summary == "exit 1 (already resubmitted as 10)"


def test_old_history_that_links_up_late_is_not_news() -> None:
    det = baseline(job("1"))
    old = dataclasses.replace(PREEMPTED, end_time=T0 - timedelta(days=2))
    done = seg("58721477", JobState.COMPLETED, submit=2 * 86400 - 600, end=T0 - timedelta(days=1))
    assert det.observe("k2", jobs(job("1"), old, done), T0 + timedelta(minutes=1)) == []


def test_resubmitted_is_selectable_but_not_a_default() -> None:
    e = JobEvent("resubmitted", JobKey("k2", "2"), "x", JobState.PENDING, T0, "resumes 1 (FAILED)")
    assert "resubmitted" in CATEGORIES and not wanted(e, NotifySettings(), watched=True)
    assert wanted(e, NotifySettings(events=("ended_bad", "resubmitted")), watched=False)
    assert count_line(["started", "resubmitted", "failed"]) == "1 failed · 1 resubmitted · 1 started"


def test_resubmission_notes() -> None:
    a = seg("2", JobState.FAILED, submit=100, end=T0)
    b = seg("3", JobState.COMPLETED, submit=50, end=T0)
    assert resubmitted_note(()) == "" and resumes_note(None) == ""
    assert resubmitted_note((a,)) == "resubmitted as 2 (FAILED)"
    assert resubmitted_note((a, b)) == "resubmitted as 2 (FAILED) → latest 3 COMPLETED"
    assert resumes_note(PREEMPTED) == "resumes 58717472 (PREEMPTED)"


def test_a_link_older_than_the_baseline_is_not_news_when_sacct_arrives_late() -> None:
    """Baseline from squeue only (sacct one tick late, or after a reconnect): the running resubmission, hours old,
    meets its old bad ending in the next tick — no `resubmitted` row for a link that predates the baseline."""
    new = seg("200", JobState.RUNNING, submit=5 * 3600, elapsed=4 * 3600)
    det = baseline(new)
    old = seg("100", JobState.PREEMPTED, submit=7 * 3600, end=T0 - timedelta(hours=6))
    assert det.observe("k2", jobs(old, new), T0 + timedelta(seconds=30)) == []
    assert det.observe("k2", jobs(old, new), T0 + timedelta(minutes=1)) == []      # and never later either
