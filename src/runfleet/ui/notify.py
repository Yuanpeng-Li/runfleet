"""Notifications, the watch list and the "since you were away" digest (§7.14).

:class:`Alerts` is the app-side owner: it feeds every coalesced store batch to the pure
:class:`~runfleet.core.events.TransitionDetector`, keeps the alert history (palette › Notifications), delivers what
``[notify]`` selects as a toast, a terminal notification and the optional local hook, keeps the watch list and the
last-seen time in ``ui.json``, and after a start (or the first key after ``idle_after`` without input) shows one toast
with what ended meanwhile (``D`` opens the digest). It also runs the GPU availability watches (``ui.json``
``watches``, evaluated by :mod:`runfleet.ui.gpu_watch` on the same batches): a watch that fires, or the watches pausing
with an idle pause, is delivered the same way and kept in its own history (:attr:`Alerts.watch_events`).

Terminal notifications: OSC 9 (iTerm2, WezTerm, Windows Terminal, ghostty, kitty), OSC 777 (foot, urxvt, VTE), else
BEL; inside tmux wrapped in DCS passthrough (needs ``set -g allow-passthrough on``). They are written through the
Textual driver from the event loop between frames (``call_later``), never from a worker thread. The hook runs
locally in its own session with the event in ``RUNFLEET_*`` variables, under a timeout; failures are logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from runfleet.config import NotifySettings
from runfleet.core.events import (JobEvent, Notice, TransitionDetector, coalesce, count_line, end_word, ended_since,
                                  wanted)
from runfleet.models import ConnState, Job, JobKey, StateGroup
from runfleet.ui import gpu_watch
from runfleet.ui.gpu_watch import GpuWatch, Reading, WatchEvent
from runfleet.util import humanize, platform

if TYPE_CHECKING:
    from runfleet.ui.app import RunFleetApp

log = logging.getLogger("runfleet.ui.notify")

TerminalMode = Literal["osc9", "osc777", "bell", "off"]
_FRESH = (ConnState.OK, ConnState.DEGRADED)
_GONE = (ConnState.UNREACHABLE, ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.DISABLED)   # baseline again after
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


# ---------------------------------------------------------------- terminal notifications (pure)
def terminal_mode(env: Mapping[str, str]) -> TerminalMode:
    """The notification escape the terminal understands, from its environment (tmux keeps the outer terminal's
    ``KITTY_WINDOW_ID`` / ``WEZTERM_PANE`` / ``WT_SESSION`` / ``LC_TERMINAL`` …, not its ``TERM_PROGRAM``)."""
    program = env.get("TERM_PROGRAM", "").lower()
    term = env.get("TERM", "").lower()
    if (program in ("iterm.app", "wezterm", "ghostty") or env.get("LC_TERMINAL", "").lower() == "iterm2"
            or term in ("xterm-kitty", "xterm-ghostty")
            or any(k in env for k in ("WT_SESSION", "KITTY_WINDOW_ID", "WEZTERM_PANE", "GHOSTTY_RESOURCES_DIR",
                                      "ITERM_SESSION_ID"))):
        return "osc9"
    if term.startswith(("foot", "rxvt")) or "VTE_VERSION" in env:
        return "osc777"
    return "bell"


def _clean(text: str, limit: int = 240) -> str:
    """One line without control characters (they would end or corrupt the escape sequence)."""
    return humanize.truncate(" ".join(_CONTROL.sub(" ", text).split()), limit)


def escape_sequence(mode: TerminalMode, title: str, body: str, *, tmux: bool = False) -> str:
    """The bytes to write for one notification ("" for ``off``)."""
    if mode == "off":
        return ""
    if mode == "bell":
        return "\a"                                       # tmux forwards a bell by itself
    if mode == "osc9":
        seq = f"\x1b]9;{_clean(f'{title}: {body}')}\x07"   # never starts with a digit (ConEmu's 9;4 progress)
    else:
        seq = f"\x1b]777;notify;{_clean(title).replace(';', ',')};{_clean(body).replace(';', ',')}\x07"
    if tmux:
        seq = "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    return seq


# ---------------------------------------------------------------- local hook (pure parts)
WINDOWS_SHELL = "cmd.exe"            # a string hook on Windows: `cmd.exe /c <string>` (through the shell, unquoted)

def hook_argv(command: Sequence[str] | str) -> list[str] | None:
    """`[notify] command`: an argv list as given, a string through ``/bin/sh -c`` (Windows: ``cmd.exe /c``, run by
    :func:`run_hook` through the shell so cmd.exe gets the string unquoted); None when unset."""
    if isinstance(command, str):
        if not command.strip():
            return None
        return [WINDOWS_SHELL, "/c", command] if platform.IS_WINDOWS else ["/bin/sh", "-c", command]
    return list(command) or None


def hook_env(notice: Notice) -> dict[str, str]:
    """``RUNFLEET_EVENT`` (category, or ``summary`` for a burst), ``_HOST``, ``_JOB``, ``_NAME``, ``_STATE``,
    ``_SUMMARY`` (the one-line text)."""
    e = notice.events[0]
    one = not notice.summary
    return {"RUNFLEET_EVENT": e.category if one else "summary", "RUNFLEET_HOST": notice.host,
            "RUNFLEET_JOB": e.key.id if one else "", "RUNFLEET_NAME": e.name if one else "",
            "RUNFLEET_STATE": e.state.name if one else "", "RUNFLEET_SUMMARY": notice.text}


def watch_hook_env(event: WatchEvent) -> dict[str, str]:
    """A GPU watch for the hook: ``RUNFLEET_EVENT`` ``gpu_free`` (or ``gpu_watch_paused``), ``_HOST`` the first host
    that matched, ``_NAME`` the watch (``2× H200 on k2``), ``_SUMMARY`` the one-line text; no job or state."""
    return {"RUNFLEET_EVENT": "gpu_free" if event.kind == "gpu_free" else "gpu_watch_paused",
            "RUNFLEET_HOST": event.host, "RUNFLEET_JOB": "", "RUNFLEET_NAME": event.watch, "RUNFLEET_STATE": "",
            "RUNFLEET_SUMMARY": event.text()}


async def run_hook(argv: Sequence[str], env: Mapping[str, str], timeout: float) -> None:
    """Run the hook in its own session (no terminal), kill its tree after ``timeout``; log, never raise."""
    from runfleet.transport import spawn

    streams: dict[str, Any] = {"stdin": asyncio.subprocess.DEVNULL, "stdout": asyncio.subprocess.DEVNULL,
                               "stderr": asyncio.subprocess.PIPE, "env": {**os.environ, **env},
                               **platform.background_spawn_kwargs()}
    try:
        if platform.IS_WINDOWS and len(argv) == 3 and argv[:2] == [WINDOWS_SHELL, "/c"]:
            proc = await asyncio.create_subprocess_shell(argv[2], **streams)    # `%COMSPEC% /c "<string>"`
        else:
            proc = await asyncio.create_subprocess_exec(*platform.resolve_program(argv), **streams)
    except OSError as exc:
        log.warning("notify command %s cannot start: %s", argv[0], exc)
        return
    finished = False
    try:
        _out, err = await asyncio.wait_for(proc.communicate(), timeout)
        finished = True
        if proc.returncode:
            log.warning("notify command %s exited %s: %s", argv[0], proc.returncode,
                        err.decode(errors="replace").strip()[-300:])
    except TimeoutError:
        log.warning("notify command %s still running after %gs: killed", argv[0], timeout)
        await spawn.terminate(proc)
    finally:
        if not finished:        # timed out or cancelled (RunFleet quits): the whole group, even when the shell itself
            with contextlib.suppress(ProcessLookupError, PermissionError):   # exited and left `cmd &` holding stderr
                platform.kill_group(proc.pid)


# ---------------------------------------------------------------- the app-side owner
def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


def _jobs_ok(sections: Mapping[str, Any]) -> bool:
    """The job list of the last poll is fresh: no job-producing section (squeue, ps) failed."""
    return not any(name in sections and not sections[name].ok for name in ("slurm.squeue", "process.ps"))


def since_label(since: datetime, now: datetime) -> str:
    """``17:23`` today-ish, ``09-06 17:23`` further back."""
    return humanize.local_hm(since) if now - since < timedelta(hours=20) else humanize.local_dt(since)


class Alerts:
    """Notifications, watch list and away digest of one app (see the module docstring)."""

    SCAN_EVERY = 30.0            # time-based events (time_low) between store changes
    SEEN_EVERY = 300.0           # ui.json last_seen refresh
    AWAY_WAIT = 60.0             # the digest waits at most this long for every host to report
    WATCH_KEEP = 7 * 86400.0     # a watched job that is gone this long is forgotten
    HISTORY = 200
    MAX_HOOKS = 4                # concurrent hook processes; more are dropped (logged)

    def __init__(self, app: RunFleetApp) -> None:
        self.app = app
        self.detector = TransitionDetector()
        self.history: deque[JobEvent] = deque(maxlen=self.HISTORY)
        state = app.ui_state
        raw = state.get("watch")
        self.watch: dict[str, str] = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        self.last_seen = _parse_iso(state.get("last_seen"))
        self.digest_since: datetime | None = None            # `D` shows what ended after this
        self._away: tuple[datetime, datetime | None, float] | None = None   # (since, fresh after, asked at mono)
        self._hooks = 0
        self.gpu_watches: list[GpuWatch] = gpu_watch.load_watches(state)    # ui.json `watches`
        self.gpu_eval = gpu_watch.Evaluator()
        self.watch_events: deque[WatchEvent] = deque(maxlen=self.HISTORY)
        self._gpu_memo: dict[str, tuple[Any, ...]] = {}
        self._paused_said = False

    @property
    def settings(self) -> NotifySettings:
        return self.app.service.config.notify

    def now(self) -> datetime:
        return self.app.clock.now()

    # ------------------------------------------------------------------ lifecycle
    def mount(self) -> None:
        if self.last_seen is not None:
            self._away = (self.last_seen, self.now(), self.app.clock.monotonic())
        self.on_changes(())
        self.app.set_interval(self.SCAN_EVERY, self.scan)
        self.app.set_interval(self.SEEN_EVERY, self.save_seen)
        self.app.set_timer(self.AWAY_WAIT + 0.5, self.scan)    # the digest's deadline, should no batch arrive

    def unmount(self) -> None:
        self.save_seen(at_exit=True)

    def save_seen(self, *, at_exit: bool = False) -> None:
        """last_seen = when the user last looked: now when quitting, else their last key / click."""
        now = self.now()
        idle = 0.0 if at_exit else max(0.0, self.app.clock.monotonic() - self.app._last_input)
        self._prune_watch(now)
        self.app.save_ui_state(last_seen=_iso(now - timedelta(seconds=idle)), watch=dict(self.watch))

    # ------------------------------------------------------------------ store batches
    def on_changes(self, changes: Sequence[Any]) -> None:
        """Every coalesced store batch (and the periodic scan): detect, select, deliver; then the away check."""
        try:
            self._observe({c.host for c in changes if c.host != "*"})
            self.check_away()
        except Exception:                      # notifications must never break the store bridge
            log.exception("notification processing failed")
        try:
            self.check_gpu_watches()
        except Exception:                      # neither may a GPU watch
            log.exception("GPU watch evaluation failed")

    def scan(self) -> None:
        self.on_changes(())

    def _observe(self, touched: set[str]) -> None:
        store = self.app.store
        now = self.now()
        for host in [h for h in self.detector.hosts() if h not in store.hosts]:
            self.detector.reset(host)                      # removed in Settings
        events: list[JobEvent] = []
        for host, state in store.hosts.items():
            st = state.status.state
            if st in _GONE:
                self.detector.reset(host)
            elif st in _FRESH and state.snapshot is not None and state.status.last_ok is not None:
                if self.detector.known(host):
                    if host in touched or not touched:
                        events += self.detector.observe(host, state.snapshot.jobs, now)
                elif _jobs_ok(state.status.sections):     # never a baseline of cached jobs the poll could not renew
                    self.detector.observe(host, state.snapshot.jobs, now)
        if events:
            self.deliver(events)

    def deliver(self, events: Sequence[JobEvent]) -> None:
        settings = self.settings
        picked = [e for e in events if wanted(e, settings, watched=self.is_watched_key(e.key))]
        if not picked:
            return
        self.history.extend(picked)
        for e in picked:
            log.info("event %s %s %s", e.category, e.key, e.summary)
        if not settings.enabled:
            return
        mode: TerminalMode = settings.terminal if settings.terminal != "auto" else terminal_mode(os.environ)
        argv = hook_argv(settings.command)
        sent: set[str] = set()
        for notice in coalesce(picked):
            self.app.toast(notice.text, severity="warning" if notice.bad else "information",
                           timeout=12.0 if notice.summary else 8.0)
            seq = escape_sequence(mode, f"RunFleet · {notice.host}", notice.text, tmux=bool(os.environ.get("TMUX")))
            if seq and seq not in sent:               # one bell per batch, not one per notice
                sent.add(seq)
                self.app.call_later(self.write_terminal, seq)
            if argv is not None:
                self._start_hook(argv, hook_env(notice), settings.command_timeout)

    def _deliver_watch(self, event: WatchEvent) -> None:
        """A GPU watch fired or paused: history always; with ``[notify] enabled`` a toast, a terminal notification
        and the hook (``watch_hook_env``). The ``[notify] events`` selection does not apply: the user asked."""
        self.watch_events.append(event)
        log.info("gpu watch %s: %s", event.kind, event.text())
        settings = self.settings
        if not settings.enabled:
            return
        self.app.toast(event.text(), severity="warning" if event.bad else "information", timeout=12.0)
        mode: TerminalMode = settings.terminal if settings.terminal != "auto" else terminal_mode(os.environ)
        title = f"RunFleet · {event.host}" if event.host else "RunFleet"
        seq = escape_sequence(mode, title, event.text(), tmux=bool(os.environ.get("TMUX")))
        if seq:
            self.app.call_later(self.write_terminal, seq)
        argv = hook_argv(settings.command)
        if argv is not None:
            self._start_hook(argv, watch_hook_env(event), settings.command_timeout)

    def write_terminal(self, seq: str) -> None:
        """Write a notification escape through the driver (the event loop, between frames; never headless)."""
        app: Any = self.app
        driver = app._driver
        if driver is None or app.is_headless:
            return
        try:
            driver.write(seq)
        except Exception:                      # suspended / shutting down: a missed beep, never an error
            log.debug("terminal notification not written", exc_info=True)

    def _start_hook(self, argv: list[str], env: dict[str, str], timeout: float) -> None:
        if self._hooks >= self.MAX_HOOKS:
            log.warning("notify command skipped: %d still running", self._hooks)
            return
        self._hooks += 1

        async def run() -> None:
            try:
                await run_hook(argv, env, timeout)
            finally:
                self._hooks -= 1

        self.app.run_worker(run(), group="notify-hook", exit_on_error=False)

    # ------------------------------------------------------------------ watch list
    def is_watched_key(self, key: JobKey) -> bool:
        if str(key) in self.watch:
            return True
        job = self.app.store.job(key)
        return job is not None and self.is_watched(job)

    def is_watched(self, job: Job) -> bool:
        """Watched itself, or an element of a watched array summary (``500_[1-20%2]`` covers ``500_7``)."""
        if str(job.key) in self.watch:
            return True
        if job.array_job_id:
            prefix = f"{job.key.host}:{job.array_job_id}_["
            return any(k.startswith(prefix) for k in self.watch)
        return False

    def watched_keys(self) -> frozenset[str]:
        return frozenset(self.watch)

    def toggle_watch(self, keys: Iterable[JobKey]) -> bool | None:
        """Watch every key (unwatch them all when all are watched already); True = now watched, None = no keys."""
        names = [str(k) for k in keys]
        if not names:
            return None
        if all(n in self.watch for n in names):
            for n in names:
                self.watch.pop(n, None)
            on = False
        else:
            stamp = _iso(self.now())
            for n in names:
                self.watch.setdefault(n, stamp)
            on = True
        self.app.save_ui_state(watch=dict(self.watch))
        return on

    def _prune_watch(self, now: datetime) -> None:
        """Present jobs refresh their stamp; one gone for WATCH_KEEP is dropped."""
        store = self.app.store
        stamp = _iso(now)
        for name in list(self.watch):
            host, _, jid = name.partition(":")
            if store.job(JobKey(host, jid)) is not None:
                self.watch[name] = stamp
                continue
            seen = _parse_iso(self.watch[name])
            if seen is None or (now - seen).total_seconds() > self.WATCH_KEEP:
                del self.watch[name]

    # ------------------------------------------------------------------ GPU availability watches
    def gpu_readings(self) -> dict[str, Reading]:
        """Every host's current free-GPU reading (:func:`gpu_watch.readings`, the fleet panel's freshness rule)."""
        from runfleet.ui.widgets.host_card import old_reading

        now = self.now()
        return gpu_watch.readings(self.app.store.hosts, now, old=lambda state, gpu: old_reading(state, gpu, now),
                                  rate=getattr(self.app, "multiplier", 1.0), memo=self._gpu_memo)

    def check_gpu_watches(self) -> None:
        """Count new readings for every watch and deliver those that fire; once per idle pause, say that the watches
        are paused with polling (a silence the user could mistake for "no GPUs")."""
        if not self.gpu_watches:
            self._paused_said = False
            return
        now = self.now()
        paused = bool(self.app.store.idle_paused)
        for fired in self.gpu_eval.update(self.gpu_watches, self.gpu_readings(), now, paused=paused):
            self._deliver_watch(gpu_watch.fired_event(fired))
        if paused and not self._paused_said:
            self._paused_said = True
            n = len(self.gpu_watches)
            idle = humanize.age(self.app.service.config.general.idle_pause_after)
            self._deliver_watch(WatchEvent(
                "paused", now, "", "", f"polling stopped after {idle} without input, so {n} "
                f"watch{'es' if n != 1 else ''} cannot fire — any key in RunFleet resumes"))
        elif not paused:
            self._paused_said = False

    def gpu_watch_status(self, watch: GpuWatch) -> tuple[str, gpu_watch.Level]:
        return self.gpu_eval.status(watch, self.gpu_readings(), self.now(), paused=bool(self.app.store.idle_paused))

    def add_gpu_watch(self, watch: GpuWatch) -> str | None:
        """Keep a new watch (ui.json) and count the current reading at once; the problem when it cannot be added."""
        problem = gpu_watch.add_problem(self.gpu_watches, watch)
        if problem is not None:
            return problem
        self.gpu_watches.append(watch)
        self.app.save_ui_state(watches=gpu_watch.with_watch(self.app.ui_state, watch))
        self.check_gpu_watches()
        return None

    def remove_gpu_watch(self, watch: GpuWatch) -> bool:
        if watch not in self.gpu_watches:
            return False
        self.gpu_watches.remove(watch)
        self.gpu_eval.states.pop(watch, None)
        self.app.save_ui_state(watches=gpu_watch.without_watch(self.app.ui_state, watch))
        return True

    # ------------------------------------------------------------------ since you were away
    def note_input(self, idle_s: float) -> None:
        """A key / click after ``idle_after`` without input: the user is back, summarise what ended meanwhile."""
        if idle_s < self.app.service.config.general.idle_after or self._away is not None:
            return
        now = self.now()
        fresh_after = now if self.app.store.idle_paused else None     # polling was paused: wait for new data
        self._away = (now - timedelta(seconds=idle_s), fresh_after, self.app.clock.monotonic())
        if fresh_after is None:                # the data is current: say it now, after this key is handled
            self.app.call_later(self.check_away)

    def check_away(self) -> None:
        if self._away is None:
            return
        since, fresh_after, asked = self._away
        waited = self.app.clock.monotonic() - asked
        if fresh_after is not None and waited < self.AWAY_WAIT and not self._all_reported(fresh_after):
            return
        self._away = None
        self.digest_since = since
        jobs = ended_since(self.app.store.jobs(), since)
        if not jobs:
            return
        bad = any(j.state.group is StateGroup.FAILED for j in jobs)
        self.app.toast(f"since {since_label(since, self.now())}: {count_line(end_word(j.state) for j in jobs)}"
                       " — D to review", severity="warning" if bad else "information", timeout=20.0)

    def _all_reported(self, after: datetime) -> bool:
        """Every enabled host polled successfully after ``after``, or is in a state that needs the user."""
        for state in self.app.store.hosts.values():
            st = state.status
            if st.state in _GONE:
                continue
            if st.last_ok is None or st.last_ok < after:
                return False
        return True

    def digest(self) -> tuple[datetime, list[Job]]:
        """What `D` shows: own jobs ended since the last away point (else within ``recent_window``)."""
        since = self.digest_since or (self.now() - timedelta(seconds=self.app.service.config.general.recent_window))
        return since, ended_since(self.app.store.jobs(), since)
