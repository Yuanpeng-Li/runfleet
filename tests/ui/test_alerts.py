"""Notifications, watch list, "since you were away" digest, runway colours and restart badges (§7.14).

Pure helpers (terminal detection, escapes, hook argv/env, the hook runner) plus pilot flows on FakeService.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import OptionList

from runfleet.config import NotifySettings
from runfleet.core.events import JobEvent, Notice
from runfleet.models import ConnState, JobKey, JobState
from runfleet.ui import theme
from runfleet.ui.app import RunFleetApp
from runfleet.ui.notify import escape_sequence, hook_argv, hook_env, run_hook, terminal_mode
from runfleet.ui.screens.job_detail import JobDetailScreen, title_line
from runfleet.ui.screens.notifications import DigestScreen, NotificationsScreen
from runfleet.ui.widgets.job_table import JobTable, left_text, runway_level
from runfleet.ui.widgets.jobs_view import JobsView
from runfleet.util import platform
from tests.ui.fakes import T0, FakeService, slurm_job

MakeApp = Callable[..., tuple[RunFleetApp, FakeService]]
FLUSH = 0.3


def toasts(app: RunFleetApp) -> list[str]:
    return [n.message for n in app._notifications]


def with_notify(svc: FakeService, **changes: Any) -> FakeService:
    svc.config = dataclasses.replace(svc.config, notify=dataclasses.replace(NotifySettings(), **changes))
    return svc


def ended(svc: FakeService, host: str, jid: str, state: JobState, **kw: Any) -> None:
    job = svc.store.job(JobKey(host, jid))
    assert job is not None
    svc.put_job(dataclasses.replace(job, state=state, end_time=svc.clock.now(), **kw))


# ---------------------------------------------------------------------------------------------------- pure
@pytest.mark.parametrize(("env", "mode"), [
    ({"TERM_PROGRAM": "iTerm.app"}, "osc9"), ({"TERM_PROGRAM": "WezTerm"}, "osc9"), ({"WT_SESSION": "x"}, "osc9"),
    ({"TERM": "xterm-kitty"}, "osc9"), ({"TERM_PROGRAM": "tmux", "KITTY_WINDOW_ID": "3"}, "osc9"),
    ({"TERM_PROGRAM": "tmux", "LC_TERMINAL": "iTerm2"}, "osc9"), ({"TERM_PROGRAM": "ghostty"}, "osc9"),
    ({"TERM": "foot"}, "osc777"), ({"TERM": "rxvt-unicode-256color"}, "osc777"), ({"VTE_VERSION": "7600"}, "osc777"),
    ({"TERM": "xterm-256color"}, "bell"), ({}, "bell"),
])
def test_terminal_mode_detection(env: dict[str, str], mode: str) -> None:
    assert terminal_mode(env) == mode


def test_escape_sequences_and_tmux_passthrough() -> None:
    assert escape_sequence("osc9", "RunFleet · k2", "k2:1 x failed") == "\x1b]9;RunFleet · k2: k2:1 x failed\x07"
    assert escape_sequence("osc777", "t;1", "a;b\x1b[31m\nc") == "\x1b]777;notify;t,1;a,b [31m c\x07"
    assert escape_sequence("bell", "t", "b", tmux=True) == "\a"
    assert escape_sequence("off", "t", "b") == ""
    wrapped = escape_sequence("osc9", "t", "b", tmux=True)
    assert wrapped == "\x1bPtmux;\x1b\x1b]9;t: b\x07\x1b\\"


def test_hook_argv_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)       # the POSIX behaviour (Windows: test_windows_*.py)
    assert hook_argv("notify-send RunFleet \"$RUNFLEET_SUMMARY\"") == ["/bin/sh", "-c",
                                                                       "notify-send RunFleet \"$RUNFLEET_SUMMARY\""]
    assert hook_argv(("notify-send", "x")) == ["notify-send", "x"]
    assert hook_argv(()) is None and hook_argv("  ") is None
    e = JobEvent("ended", JobKey("k2", "2332861"), "orb", JobState.OUT_OF_MEMORY, T0, "out of memory")
    env = hook_env(Notice("k2", e.text(), (e,)))
    assert env == {"RUNFLEET_EVENT": "ended_bad", "RUNFLEET_HOST": "k2", "RUNFLEET_JOB": "2332861",
                   "RUNFLEET_NAME": "orb", "RUNFLEET_STATE": "OUT_OF_MEMORY",
                   "RUNFLEET_SUMMARY": "k2:2332861 orb out of memory: out of memory"}
    burst = hook_env(Notice("k2", "k2: 5 failed", (e, e, e, e, e)))
    assert burst["RUNFLEET_EVENT"] == "summary" and burst["RUNFLEET_JOB"] == ""


async def test_run_hook_gets_the_env_and_a_slow_hook_is_killed(tmp_path: Path, caplog: Any) -> None:
    out = tmp_path / "out"
    await run_hook([sys.executable, "-c", f"import os; open({str(out)!r}, 'w').write(os.environ['RUNFLEET_HOST'])"],
                   {"RUNFLEET_HOST": "k2"}, 10)
    assert out.read_text() == "k2"
    started = asyncio.get_running_loop().time()
    await run_hook([sys.executable, "-c", "import time; time.sleep(30)"], {}, 0.3)
    assert asyncio.get_running_loop().time() - started < 10
    assert "killed" in caplog.text
    await run_hook(["/nonexistent/runfleet-hook"], {}, 1)          # cannot start: logged, never raised
    await run_hook([sys.executable, "-c", "import sys; sys.exit(3)"], {}, 5)
    assert "exited 3" in caplog.text


@pytest.mark.posix                # /bin/sh, process groups, /proc
async def test_run_hook_timeout_kills_a_background_child_of_an_exited_shell(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    await run_hook(["/bin/sh", "-c", f"sleep 30 & echo $! > {pidfile}"], {}, 0.3)   # sh exits, sleep keeps stderr

    def alive(pid: int) -> bool:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        except OSError:                          # no /proc: signal 0
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
        return stat.rsplit(")", 1)[1].split()[0] != "Z"

    pid = int(pidfile.read_text())
    for _ in range(40):
        if not alive(pid):
            break
        await asyncio.sleep(0.05)
    assert not alive(pid)


def test_runway_levels_and_left_text() -> None:
    assert runway_level(4 * 3600, 3 * 86400) == "amber"            # < 10 % of 3 days
    assert runway_level(20 * 3600, 3 * 86400) is None
    assert runway_level(25 * 60, 3 * 3600) == "amber" and runway_level(4 * 60, 3600) == "red"
    assert runway_level(60, 3600) == "red" and runway_level(3000, None) is None
    assert runway_level(20 * 60, 1800) is None and runway_level(6 * 60, 1800) == "amber"   # short limits: no all-run amber
    assert runway_level(2 * 60, 1800) == "red" and runway_level(1000, None) == "amber"
    low = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(minutes=57), limit=timedelta(hours=1))
    assert left_text(low, T0).style == theme.error()
    amber = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(hours=2, minutes=40), limit=timedelta(hours=3))
    assert left_text(amber, T0).style == theme.warn()
    calm = slurm_job("k2", "1", "x", JobState.RUNNING, elapsed=timedelta(hours=1), limit=timedelta(days=3))
    assert left_text(calm, T0).style == ""


def test_title_line_shows_watch_and_restarts() -> None:
    job = dataclasses.replace(slurm_job("k2", "1", "train", JobState.RUNNING, elapsed=timedelta(hours=1)),
                              restarts=2)
    assert "◉ ↻2 train" in title_line(job, T0, 120, watched=True).plain
    assert "◉" not in title_line(job, T0, 120).plain


# ---------------------------------------------------------------------------------------------------- pilot
async def test_no_events_at_start_then_a_failure_notifies_and_lands_in_history(make_app: MakeApp, clock: Any,
                                                                               monkeypatch: Any) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    app, svc = make_app(with_notify(FakeService(clock), terminal="osc9"))
    sent: list[str] = []
    async with app.run_test(size=(120, 35)) as pilot:
        app.alerts.write_terminal = sent.append        # type: ignore[method-assign]
        await pilot.pause(FLUSH)
        assert not app.alerts.history and not sent     # the first observation is a baseline
        ended(svc, "k2", "2332861", JobState.FAILED, exit_code=1, raw={"reason_text": "exit 1"})
        await pilot.pause(FLUSH)
        assert any("k2:2332861" in m and "failed: exit 1" in m for m in toasts(app))
        assert [e.key.id for e in app.alerts.history] == ["2332861"]
        assert sent == ["\x1b]9;RunFleet · k2: k2:2332861 orb-swp--train--cifar-warmup--s43--a02 failed: exit 1\x07"]
        app.show_notifications()
        await pilot.pause(0.2)
        assert isinstance(app.screen, NotificationsScreen)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen) and app.screen.key == JobKey("k2", "2332861")


async def test_a_burst_on_one_host_is_one_toast(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        for jid in ("2332861", "2332862", "2332863", "2333760"):
            ended(svc, "k2", jid, JobState.FAILED, exit_code=1)
        await pilot.pause(FLUSH)
        assert "k2: 4 failed" in toasts(app)
        assert len(app.alerts.history) == 4


async def test_watch_toggles_marks_persists_and_widens_what_notifies(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause(FLUSH)
        jobs = app.screen.query_one(JobsView)
        jobs.table.move_to_key("k2:2332861")
        await pilot.pause(0.1)
        first = jobs.current_job()
        assert first is not None and first.key == JobKey("k2", "2332861")
        await pilot.press("w")
        await pilot.pause(FLUSH)
        assert str(first.key) in app.ui_state["watch"]
        table = app.screen.query_one(JobTable)
        name_col = [k for k, _w in table.cols].index("name")
        assert table._cells[str(first.key)][name_col].plain.startswith("◉ ")
        other = next(j for j in svc.store.jobs(["k2"]) if j.state is JobState.RUNNING and j.key != first.key)
        ended(svc, "k2", other.key.id, JobState.COMPLETED, exit_code=0)          # not watched: quiet
        await pilot.pause(FLUSH)
        assert not app.alerts.history
        ended(svc, "k2", "2332861", JobState.COMPLETED, exit_code=0)           # watched: its end is news
        await pilot.pause(FLUSH)
        assert [e.category for e in app.alerts.history] == ["ended_ok"]
        await pilot.press("w")
        await pilot.pause(FLUSH)
        assert str(first.key) not in app.ui_state["watch"]


async def test_watch_from_job_detail_marks_the_title(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        app.open_job(JobKey("k2", "2332861"))
        await pilot.pause(0.3)
        await pilot.press("w")
        await pilot.pause(FLUSH)
        assert "◉" in str(app.screen.query_one("#title").render())


async def test_time_low_on_the_periodic_scan(make_app: MakeApp) -> None:
    app, svc = make_app()
    job = slurm_job("vega", "777", "short-eval", JobState.RUNNING, elapsed=timedelta(minutes=48),
                    limit=timedelta(hours=1))
    svc.store.put_job(job)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        assert not app.alerts.history                      # 12 min left at the baseline: not yet low
        svc.clock.advance(150)
        app.alerts.scan()
        await pilot.pause(0.1)
        assert [(e.category, e.key.id) for e in app.alerts.history] == [("time_low", "777")]
        app.alerts.scan()
        assert len(app.alerts.history) == 1


async def test_disabled_notify_keeps_history_but_stays_silent(make_app: MakeApp, clock: Any) -> None:
    svc = with_notify(FakeService(clock), enabled=False, command=("never-run",))
    app, _ = make_app(svc)
    started: list[Any] = []
    async with app.run_test(size=(120, 35)) as pilot:
        app.alerts._start_hook = lambda *a: started.append(a)   # type: ignore[method-assign]
        await pilot.pause(FLUSH)
        ended(svc, "k2", "2332861", JobState.TIMEOUT)
        await pilot.pause(FLUSH)
        assert len(app.alerts.history) == 1 and not started
        assert not any("2332861" in m for m in toasts(app))


async def test_hook_gets_one_call_per_notice(monkeypatch: pytest.MonkeyPatch, make_app: MakeApp, clock: Any) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)       # the POSIX behaviour (Windows: test_windows_*.py)
    svc = with_notify(FakeService(clock), command="true", terminal="off")
    app, _ = make_app(svc)
    calls: list[tuple[Any, ...]] = []
    async with app.run_test(size=(120, 35)) as pilot:
        app.alerts._start_hook = lambda *a: calls.append(a)   # type: ignore[method-assign]
        await pilot.pause(FLUSH)
        ended(svc, "k2", "2332861", JobState.OUT_OF_MEMORY)
        await pilot.pause(FLUSH)
        assert len(calls) == 1
        argv, env, timeout = calls[0]
        assert argv == ["/bin/sh", "-c", "true"] and env["RUNFLEET_STATE"] == "OUT_OF_MEMORY" and timeout == 10


async def test_since_you_were_away_toast_digest_and_filter(make_app: MakeApp, clock: Any) -> None:
    svc = FakeService(clock)
    app = RunFleetApp(svc, clock=clock, animations=False,               # type: ignore[arg-type]
                      initial_state={"last_seen": (T0 - timedelta(hours=3)).isoformat()})
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        assert not any("since" in m for m in toasts(app))      # lyra has not reported yet
        clock.advance(5)
        svc.set_ok("lyra")
        svc.tick()
        await pilot.pause(FLUSH)
        away = [m for m in toasts(app) if m.startswith("since ")]
        assert len(away) == 1 and "3 completed" in away[0] and away[0].endswith("D to review")
        await pilot.press("D")
        await pilot.pause(0.2)
        assert isinstance(app.screen, DigestScreen)
        head = str(app.screen.query_one("#digest-head").render())
        assert "3 ended" in head and "k2" in head
        assert not app.screen.query(OptionList)                # nothing failed in those 3 hours
        await pilot.press("f")
        await pilot.pause(FLUSH)
        jobs = app.screen.query_one(JobsView)
        assert jobs.ended_since == T0 - timedelta(hours=3) and jobs.shown == 3
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert jobs.ended_since is None


async def test_digest_lists_failures_and_opens_them(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        ended(svc, "k2", "2332861", JobState.FAILED, exit_code=1, raw={"reason_text": "exit 1"})
        await pilot.pause(FLUSH)
        await pilot.press("D")                               # no last visit: the recent window
        await pilot.pause(0.2)
        menu = app.screen.query_one(OptionList)
        assert menu.option_count == 3          # the new failure and lyra's two of the last 24 h (not 2 days ago)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert isinstance(app.screen, JobDetailScreen)


async def test_return_after_idle_summarises_what_ended(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        svc.clock.advance(3600)
        ended(svc, "vega", "723954", JobState.COMPLETED, exit_code=0)
        await pilot.pause(FLUSH)
        await pilot.press("tab")                            # the first key after an hour away
        await pilot.pause(FLUSH)
        assert any(m.startswith("since ") and "1 completed" in m for m in toasts(app))


async def test_restart_badge_in_the_table(make_app: MakeApp) -> None:
    app, svc = make_app()
    job = svc.store.job(JobKey("k2", "2332861"))
    assert job is not None
    svc.store.put_job(dataclasses.replace(job, restarts=2))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.press("tab")
        await pilot.pause(FLUSH)
        table = app.screen.query_one(JobTable)
        name_col = [k for k, _w in table.cols].index("name")
        assert table._cells["k2:2332861"][name_col].plain.startswith("↻2 ")


async def test_last_seen_and_watch_are_saved_on_exit(make_app: MakeApp) -> None:
    app, svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        app.alerts.toggle_watch([JobKey("k2", "2332861"), JobKey("k2", "gone-long-ago")])
        app.alerts.watch["k2:gone-long-ago"] = (T0 - timedelta(days=8)).isoformat()
    assert app.ui_state["last_seen"] == T0.isoformat(timespec="seconds")
    assert list(app.ui_state["watch"]) == ["k2:2332861"]                  # pruned: gone for > 7 days


async def test_no_baseline_from_a_poll_whose_job_list_failed(make_app: MakeApp) -> None:
    """A DEGRADED host whose squeue failed still shows (cached) jobs: they must not become the baseline."""
    app, svc = make_app()
    st = svc.store.hosts["vega"].status
    bad = dict(st.sections, **{"slurm.squeue": dataclasses.replace(st.sections["slurm.squeue"], ok=False)})
    svc.store.hosts["vega"].status = dataclasses.replace(st, state=ConnState.DEGRADED, sections=bad)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause(FLUSH)
        assert app.alerts.detector.known("k2") and not app.alerts.detector.known("vega")
        svc.store.set_status("vega", dataclasses.replace(st, last_ok=svc.clock.now()))
        await pilot.pause(FLUSH)
        assert app.alerts.detector.known("vega") and not app.alerts.history
        svc.inject("vega", "unreachable")
        await pilot.pause(FLUSH)
        assert not app.alerts.detector.known("vega")              # back from an outage: a fresh baseline


async def test_notifications_screen_when_empty(make_app: MakeApp) -> None:
    app, _svc = make_app()
    async with app.run_test(size=(120, 35)) as pilot:
        app.show_notifications()
        await pilot.pause(0.2)
        assert isinstance(app.screen, NotificationsScreen) and not app.screen.query(OptionList)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, NotificationsScreen)


async def test_notifications_list_fits_a_short_terminal(make_app: Any) -> None:
    """At 80×24 a 20-row list sat in a 19-row dialog: its last rows (and the highlighted one) were clipped unseen."""
    from datetime import timedelta

    from textual.widgets import OptionList

    from runfleet.core.events import JobEvent
    from runfleet.models import JobKey, JobState
    from runfleet.ui.screens.notifications import NotificationsScreen
    from tests.ui.fakes import T0

    app, _svc = make_app()
    events = [JobEvent("ended", JobKey("k2", f"23300{i:02d}"), f"run-{i}", JobState.FAILED,
                       T0 - timedelta(minutes=i), "exit 1", 1) for i in range(30)]
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.push_screen(NotificationsScreen(events))
        await pilot.pause(0.3)
        menu = app.screen.query_one(OptionList)
        box = app.screen.query_one("#notify-box")
        assert menu.outer_size.height <= box.size.height              # size: inside the border
        for _ in range(29):
            await pilot.press("down")
        await pilot.pause()
        assert menu.highlighted == 29 and menu.region.bottom <= box.region.bottom
