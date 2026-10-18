"""[WP4] agent.events (docs/AGENT_INTERFACE.md §3.8): stateless event rebuild with stable ids, late accounting and
overlapping windows, cached ended tasks beyond recent_window, "now" items, ordering, coverage and gaps."""

from __future__ import annotations

from datetime import timedelta

from runfleet.agent import events as ev
from runfleet.agent import view
from runfleet.agent.data import EXIT_OK, EXIT_PARTIAL
from runfleet.models import ConnState, JobKind, JobState
from runfleet.util.logscan import PENDING_EXIT_HINT
from tests.agent.test_view import H, T0, fleet, mk_host, mk_job

M = timedelta(minutes=1)
S = timedelta(seconds=15)


def ids(events: list[ev.Event]) -> list[str]:
    return [e.id for e in events]


def test_late_accounting_needs_an_overlapping_window() -> None:
    # sacct reports at 14:02 a job that ended at 13:59: a window from 14:00 cannot show it ...
    late = mk_job("k2", "2331977", JobState.OUT_OF_MEMORY, submit_time=T0 - 5 * H, start_time=T0 - 4 * H,
                  end_time=T0 - 3 * M - 31 * timedelta(seconds=1), source="sacct")
    fv = fleet(mk_host("k2", [late]))
    since_last_look = T0 - 2 * M - timedelta(seconds=31)          # 14:00:00
    assert "k2:2331977:ended_bad" not in ids(ev.events_since(fv, since_last_look)[0])
    # ... the skill's overlapping window (last look minus 30 min) does, and the id is the same in every window
    a = ev.events_since(fv, since_last_look - 30 * M)[0]
    b = ev.events_since(fv, since_last_look - 45 * M)[0]
    end_a = next(e for e in a if e.id == "k2:2331977:ended_bad")
    end_b = next(e for e in b if e.id == "k2:2331977:ended_bad")
    assert end_a == end_b and end_a.kind == "ended_bad" and end_a.at == late.end_time
    assert end_a.text == "OUT_OF_MEMORY" and end_a.job == "k2:2331977"
    assert len(ev.dedupe([a, b])) == len({e.id for e in [*a, *b]})


def test_kinds_and_estimates() -> None:
    jobs = [
        mk_job("k2", "1", JobState.COMPLETED, submit_time=T0 - 5 * H, start_time=T0 - 4 * H, end_time=T0 - 3 * H),
        mk_job("k2", "2", JobState.CANCELLED, submit_time=T0 - 9 * H, start_time=T0 - 4 * H, end_time=T0 - 2 * H),
        mk_job("k2", "3", JobState.PENDING, submit_time=T0 - 1 * H, start_time=T0 + H, est_start=True),
        mk_job("k2", "4", submit_time=T0 - 3 * H, start_time=T0 - 2 * H, restarts=1),
        mk_job("k2", "5", user="bob", is_mine=False, submit_time=T0 - H, start_time=T0 - H),
    ]
    out, _cov = ev.events_since(fleet(mk_host("k2", jobs)), T0 - 6 * H)
    by_id = {e.id: e for e in out}
    assert by_id["k2:1:ended_ok"].kind == "ended_ok" and by_id["k2:2:ended_other"].kind == "ended_other"
    assert by_id["k2:1:started"].at == T0 - 4 * H and by_id["k2:1:submitted"].at == T0 - 5 * H
    assert "k2:2:submitted" not in by_id                          # submitted before the window
    assert "k2:3:submitted" in by_id and "k2:3:started" not in by_id    # an estimated start is no event
    assert "k2:4:started:1" in by_id                              # a requeued run starts anew
    assert not any(e.host == "k2" and e.job == "k2:5" for e in out)     # other users' jobs never count
    assert [e.at for e in out] == sorted(e.at for e in out if e.at is not None)


def test_ended_tasks_beyond_recent_window_and_dedup_with_the_snapshot() -> None:
    old = mk_job("labstation", "p1-100", JobState.FAILED, kind=JobKind.TASK, submit_time=None,
                 start_time=T0 - 80 * H, end_time=T0 - 72 * H, time_limit=None)
    both = mk_job("labstation", "p2-200", JobState.EXITED, kind=JobKind.TASK, submit_time=None,
                  start_time=T0 - 5 * H, end_time=T0 - 2 * H, time_limit=None)
    stale_copy = mk_job("labstation", "p2-200", JobState.FAILED, kind=JobKind.TASK, submit_time=None,
                        start_time=T0 - 5 * H, end_time=T0 - 2 * H, time_limit=None)
    hv = mk_host("labstation", [both], kind="workstation", ended_tasks=(old, stale_copy))
    out, cov = ev.events_since(fleet(hv), T0 - 96 * H)
    ends = [e for e in out if e.kind.startswith("ended")]
    assert ids(ends) == ["labstation:p1-100:ended_bad", "labstation:p2-200:ended_other"]  # task ids: the full key (V5)
    assert ends[0].kind == "ended_bad" and ends[0].job == "labstation:p1"
    assert ends[1].kind == "ended_other"                          # the snapshot's copy wins over the cached one
    assert cov[0].tasks_from == T0 - timedelta(days=7) and not cov[0].gap


def test_end_still_being_classified_waits() -> None:
    pending = mk_job("labstation", "p3-300", JobState.EXITED, kind=JobKind.TASK, end_time=T0 - M,
                     hints=(PENDING_EXIT_HINT,), time_limit=None)
    out, _ = ev.events_since(fleet(mk_host("labstation", [pending], kind="workstation")), T0 - H)
    assert not any(i.startswith("labstation:p3:ended") for i in ids(out))


def test_now_items_come_last() -> None:
    low = mk_job("lyra", "5512", start_time=T0 - 23 * H - 46 * M, elapsed=23 * H + 46 * M, observed_at=T0)
    fresh = mk_job("lyra", "5513", submit_time=T0 - 10 * M, start_time=T0 - 5 * M, elapsed=5 * M, observed_at=T0)
    lynx = mk_host("lynx", [], state=ConnState.NEEDS_AUTH, state_since=T0 - 2 * H)
    off = mk_host("off", [], state=ConnState.DISABLED, enabled=False)
    out, cov = ev.events_since(fleet(mk_host("lyra", [low, fresh]), lynx, off), T0 - H)
    kinds = [e.kind for e in out]
    assert kinds[-2:] == ["host", "time_low"] and out[-1].at is None and out[-2].at is None
    host_ev = out[-2]
    assert host_ev.id == f"lynx:host:needs_auth:{int((T0 - 2 * H).timestamp())}" and host_ev.since == T0 - 2 * H
    assert out[-1].id == "lyra:5512:time_low" and out[-1].text == "14m left of 1d"
    assert not any(e.host == "off" for e in out)
    assert {c.host: c.gap for c in cov}["off"] is False


def test_coverage_and_gaps() -> None:
    k2 = mk_host("k2", [mk_job("k2", "1")])
    old = mk_host("vega", [mk_job("vega", "1")], data_age=7200)
    none = mk_host("lynx", snapshot=False, state=ConnState.NEEDS_AUTH)
    fv = fleet(k2, old, none)
    _, cov = ev.events_since(fv, T0 - 2 * H)
    c = {x.host: x for x in cov}
    assert not c["k2"].gap and c["k2"].history_from == T0 - timedelta(seconds=11) - timedelta(days=3)
    assert c["vega"].gap and not c["vega"].no_data
    assert c["lynx"].gap and c["lynx"].no_data
    _, wide = ev.events_since(fleet(k2), T0 - timedelta(days=4))      # older than the accounting window
    assert wide[0].gap
    doc = view.events(fv, T0 - 2 * H)
    head = doc.text.splitlines()[1]
    assert head.startswith("# events 2026-10-18T12:02:31Z → 2026-10-18T14:02:31Z (coverage: k2; vega no data after "
                           "12:02Z; lynx no data")
    assert doc.exit_code == EXIT_PARTIAL
    assert view.events(fleet(k2), T0 - 2 * H).exit_code == EXIT_OK
    wide_doc = view.events(fleet(k2), T0 - timedelta(days=4))
    assert "k2 accounting from 10-15" in wide_doc.text and wide_doc.exit_code == EXIT_PARTIAL


def test_events_budget_and_priority() -> None:
    jobs = [mk_job("k2", str(1000 + i), JobState.COMPLETED, submit_time=T0 - 20 * H + i * S,
                   start_time=T0 - 19 * H + i * S, end_time=T0 - 10 * H + i * S) for i in range(2000)]
    bad = mk_job("k2", "9999", JobState.FAILED, submit_time=T0 - 22 * H, start_time=T0 - 21 * H,
                 end_time=T0 - 20 * H)
    doc = view.events(fleet(mk_host("k2", [*jobs, bad])), T0 - 24 * H, limit=100)
    shown = doc.data["events"]
    assert len(shown) <= 100 and doc.data["truncated"]["total"] == 6003
    assert "k2:9999:ended_bad" in [e["id"] for e in shown]           # a bad end outranks routine events
    assert [e["at"] for e in shown] == sorted(e["at"] for e in shown)
    assert len(doc.text.encode()) <= 6144 and len(view.dumps(doc.data).encode()) <= 24576
    # cut by the byte budget: a bigger --limit would be cut again, so the footer says so and offers JSON / narrowing
    assert "more (older / routine, output budget) → runfleet agent events --since" in doc.text
    assert "--json  (or narrow: a shorter --since, or HOST)" in doc.text


def test_events_limit_footer_asks_for_the_rest_on_the_same_verb() -> None:
    jobs = [mk_job("k2", str(1000 + i), JobState.COMPLETED, submit_time=T0 - 5 * H, start_time=T0 - 4 * H,
                   end_time=T0 - 3 * H + i * S) for i in range(5)]
    fv = fleet(mk_host("k2", jobs))
    doc = view.events(fv, T0 - 6 * H, ("k2",), limit=3, since_text="6h")
    total = doc.data["truncated"]["total"]
    nxt = f"runfleet agent events k2 --since 6h --limit {total}"
    assert doc.text.splitlines()[-1] == f"# +{total - 3} more (older / routine) → {nxt}"
    assert doc.data["truncated"]["next"] == nxt + " --json" and len(doc.data["events"]) == 3   # JSON asks JSON
    big = view.events(fv, T0 - 6 * H, limit=3, since_text="6h")
    assert big.data["truncated"]["next"].endswith(f"--since 6h --limit {total} --json")


def test_failing_accounting_is_a_gap() -> None:
    hv = mk_host("k2", [mk_job("k2", "1")], failing={"slurm.sacct": "timeout"})
    _, cov = ev.events_since(fleet(hv), T0 - H)
    assert cov[0].gap
    doc = view.events(fleet(hv), T0 - H)
    assert doc.exit_code == EXIT_PARTIAL and "k2 accounting" in doc.text.splitlines()[1]


def test_time_low_of_an_old_sighting_is_not_run_down() -> None:
    """Codex review: a job seen an hour ago with an hour left showed "0s left" as a now-item."""
    old = mk_job("lyra", "1", start_time=T0 - 24 * H, elapsed=22 * H + 50 * M, observed_at=T0 - H,
                 time_limit=timedelta(days=1))
    out, _ = ev.events_since(fleet(mk_host("lyra", [old], data_age=3600)), T0 - 2 * H)
    low = next(e for e in out if e.kind == "time_low")
    assert low.text == "1h10m left of 1d (as of 1h ago)"


def test_tasks_sharing_a_pid_never_share_an_event_id() -> None:
    """V5: two workstation tasks with the same display id (pid reused) in one view get distinct ids: tasks use the
    full key; `job` stays the display id; Slurm ids are unchanged."""
    first = mk_job("labstation", "p4242-1000", JobState.FAILED, kind=JobKind.TASK, submit_time=None,
                   start_time=T0 - 6 * H, end_time=T0 - 5 * H, time_limit=None)
    again = mk_job("labstation", "p4242-2000", JobState.RUNNING, kind=JobKind.TASK, submit_time=None,
                   start_time=T0 - 2 * H, time_limit=None)
    hv = mk_host("labstation", [again], kind="workstation", ended_tasks=(first,))
    out, _ = ev.events_since(fleet(hv, mk_host("k2", [mk_job("k2", "77", start_time=T0 - M)])), T0 - 24 * H)
    got = ids(out)
    assert len(got) == len(set(got))
    assert "labstation:p4242-1000:ended_bad" in got and "labstation:p4242-2000:started" in got
    assert {e.job for e in out if e.host == "labstation"} == {"labstation:p4242"}
    assert "k2:77:started" in got
    # the host event id carries the state and, when known, its start (docs: lynx:host:needs_auth[:<epoch>])
    lynx = mk_host("lynx", snapshot=False, state=ConnState.NEEDS_AUTH)
    hev = [e for e in ev.events_since(fleet(lynx), T0 - H)[0] if e.kind == "host"]
    assert hev[0].id == f"lynx:host:needs_auth:{int((T0 - 5 * H).timestamp())}"
    no_since = mk_host("lynx", snapshot=False, state=ConnState.NEEDS_AUTH, state_since=None)
    assert [e.id for e in ev.events_since(fleet(no_since), T0 - H)[0]] == ["lynx:host:needs_auth"]
