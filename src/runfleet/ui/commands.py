"""[WP3] Command palette Provider with RunFleet commands (§7.10).

Fuzzy-matched commands only, no grammar: views, scope, host detail / connect / refresh / enable / disable per host,
state presets, sort, grouping, doctor, settings, action log, RunFleet log, help, quit. ``:`` opens the same palette.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from textual.command import DiscoveryHit, Hit, Hits, Provider

from runfleet.models import ConnState
from runfleet.ui.query import PRESETS, SORT_CYCLE, SORT_LABELS

Command = tuple[str, Callable[[], Any], str]


class RunFleetCommands(Provider):
    """Yields every RunFleet command; Textual's matcher ranks them against the typed query."""

    def _commands(self) -> Iterator[Command]:
        app: Any = self.app
        if not hasattr(app, "service"):
            return
        yield "Fleet view", partial(app.go_main, "fleet"), "show all hosts as cards"
        yield "Jobs view", partial(app.go_main, "jobs"), "merged job table"
        yield "GPUs view", partial(app.go_main, "gpus"), "every GPU across the fleet"
        yield "Scope: all", partial(app.set_scope, None), "show every host (key 0)"
        for i, host in enumerate(app.host_names(), start=1):
            state = app.store.hosts[host].status.state
            yield f"Scope: {host}", partial(app.set_scope, host), f"only {host} (key {i})" if i < 10 else ""
            yield f"Host detail: {host}", partial(app.action_open_host, host), "GPUs, processes, nodes"
            yield f"Connect {host}", partial(app.action_connect_host, host), "log in (Duo / password prompts)"
            yield f"Refresh {host}", partial(app.service.refresh, [host]), "poll now (>= 3 s apart)"
            if state is ConnState.DISABLED:
                yield f"Enable {host}", partial(app.service.set_host_enabled, host, True), "resume polling"
            else:
                yield f"Disable {host}", partial(app.service.set_host_enabled, host, False), "stop polling"
            yield f"Doctor {host}", partial(app.show_doctor, host), "connection diagnostics"
        yield "Refresh all", app.action_refresh_all, "poll every host now"
        yield "Log in to all hosts needing auth", app.action_login_all, "one terminal session, host after host (key C)"
        yield "Forward a port…", app.action_forward_port, "TensorBoard / Jupyter of the selected job → localhost (key P)"
        yield "Forwards", app.show_forwards, "active port forwards: copy URL, cancel"
        for preset in PRESETS:
            yield f"State preset: {preset}", partial(app.jobs_setting, "preset", preset), "jobs filter preset"
        for column in SORT_CYCLE:
            yield f"Sort by {SORT_LABELS[column]}", partial(app.jobs_setting, "sort", column), "jobs table"
        for group in ("none", "host", "array"):
            yield f"Group by {group}", partial(app.jobs_setting, "group", group), "jobs table"
        yield "Settings", app.action_settings, "hosts: save / remove / test / reorder (key ,)"
        yield "Settings: general", partial(app.action_settings, "general"), "default view, refresh speed, read-only"
        yield "Settings: about", partial(app.action_settings, "about"), "config file, log, version"
        yield "Action log", app.show_action_log, "results of this session's actions"
        yield "Notifications", app.show_notifications, "this session's job alerts (failures, restarts, time low …)"
        yield "Since you were away", app.action_digest, "what ended since your last visit (key D)"
        yield "RunFleet log", app.open_runfleet_log, "runfleet.log in the local pager"
        yield "Help", app.action_help, "keys for the current view"
        yield "Quit", app.exit, "leave RunFleet"

    async def discover(self) -> Hits:
        for name, callback, help_text in self._commands():
            yield DiscoveryHit(name, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, callback, help_text in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), callback, text=name, help=help_text)
