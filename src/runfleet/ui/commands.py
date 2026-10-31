"""[WP3] Command palette Provider with RunFleet commands (§7.10).

Fuzzy-matched commands only, no grammar: views, scope, host detail / connect / refresh / enable / disable per host,
account usage (all / one cluster first, §7.17), state presets, sort, grouping, search logs, saved Jobs views (save as / apply / delete), GPU watches (watch free GPUs… / the list),
doctor, settings, update, theme, Fleet card size, action log, RunFleet log, help, quit. ``:`` opens the same palette.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from textual.command import DiscoveryHit, Hit, Hits, Provider

from runfleet.models import ConnState
from runfleet.util import platform
from runfleet.ui.query import PRESETS, SORT_CYCLE, SORT_LABELS
from runfleet.ui.saved_views import GROUPS, describe
from runfleet.ui.widgets.host_card import is_cluster

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
        yield "Account usage", app.action_account, \
            "Slurm account balances, your usage per window and where it went, per cluster in scope (key $)"
        for host, state in app.store.hosts.items():
            if is_cluster(state):
                yield f"Account: {host}", partial(app.action_account, host), "account usage, this cluster first"
        yield "Log in to all hosts needing auth", app.action_login_all, "one terminal session, host after host (key C)"
        yield "Search logs…", app.action_search_logs, \
            "find text in the logs of your recent jobs in scope: last 256 KiB of each, ≤ 20 files per host (^f in Jobs)"
        yield "Forward a port…", app.action_forward_port, "TensorBoard / Jupyter of the selected job → localhost (key P)"
        yield "Forwards", app.show_forwards, "active port forwards: copy URL, cancel"
        for preset in PRESETS:
            yield f"State preset: {preset}", partial(app.jobs_setting, "preset", preset), "jobs filter preset"
        for column in SORT_CYCLE:
            yield f"Sort by {SORT_LABELS[column]}", partial(app.jobs_setting, "sort", column), "jobs table"
        yield "Sort by SU", partial(app.jobs_setting, "sort", "su"), \
            "jobs table: SU = charged billing-hours (= CPU-h where a cluster has no billing weights)"
        yield "Sort by GPU-h", partial(app.jobs_setting, "sort", "gpuh"), "jobs table: GPU-hours so far"
        for group in GROUPS:
            yield f"Group by {group}", partial(app.jobs_setting, "group", group), "jobs table"
        yield "Save view as…", app.prompt_save_view, "name the Jobs view's filter, preset, sort and grouping (ui.json)"
        views = app.saved_jobs_views()
        for name, view in views.items():
            yield f"View: {name}", partial(app.apply_saved_view, name), describe(view)
        for name in views:
            yield f"Delete view: {name}", partial(app.delete_saved_view, name), "forget this saved Jobs view"
        yield "Settings", app.action_settings, "hosts: save / remove / test / reorder (key ,)"
        yield "Settings: general", partial(app.action_settings, "general"), "default view, refresh speed, read-only"
        yield "Settings: about", partial(app.action_settings, "about"), "config file, log, version"
        yield ("Update RunFleet", app.open_update, "newest version from GitHub: how to update (by hand on Windows)"
               if platform.IS_WINDOWS else "newest version from GitHub: update and restart")
        for choice, what in (("auto", "follow the terminal's background"), ("dark", "dark palette"),
                             ("light", "light palette"), ("ansi", "the terminal's own 16 colours")):
            yield f"Theme: {choice}", partial(app.set_theme_choice, choice), what
        yield "Fleet: roomy cards", partial(app.set_fleet_layout, "auto"), \
            "cards grow into a tall terminal: ranked jobs, GPU owners, free GPUs (ui.json)"
        yield "Fleet: compact cards", partial(app.set_fleet_layout, "compact"), "cards keep their summary size (ui.json)"
        yield "Action log", app.show_action_log, "results of this session's actions"
        yield "Notifications", app.show_notifications, "this session's alerts: failures, restarts, time low …, GPU watches"
        yield "Since you were away", app.action_digest, "what ended since your last visit (key D)"
        yield "Watch free GPUs…", app.prompt_gpu_watch, \
            "notify when N free GPUs of a type appear on one node, e.g. 2 h200 k2 (ui.json)"
        yield "Watches", app.show_gpu_watches, "your free-GPU watches: what each is waiting for, remove one"
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
