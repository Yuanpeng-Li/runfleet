"""[WP3] Textual messages: FleetChanged, HostStatusChanged, ActionFinished, ...

The app turns store callbacks into these (coalesced to <= 4 Hz) and posts them to the active screen, which forwards
them to its data widgets. Widgets always re-read the store; the messages only say *what* may have changed.
"""

from __future__ import annotations

from collections.abc import Sequence

from textual.message import Message

from runfleet.core.store import ChangeSet
from runfleet.models import ActionResult, JobKey


class FleetChanged(Message):
    """One coalesced batch of store changes."""

    def __init__(self, changes: Sequence[ChangeSet]) -> None:
        super().__init__()
        self.changes = tuple(changes)

    @property
    def hosts(self) -> frozenset[str]:
        return frozenset(c.host for c in self.changes)

    @property
    def jobs_changed(self) -> bool:
        return any(c.jobs_added or c.jobs_changed or c.jobs_removed for c in self.changes)

    @property
    def status_changed(self) -> bool:
        return any(c.status or c.paused for c in self.changes)

    @property
    def gpus_changed(self) -> bool:
        return any(c.gpus or c.procs or c.metrics or c.nodes for c in self.changes)

    @property
    def live_keys(self) -> frozenset[JobKey]:
        """Jobs whose Attach state changed (§7.16): re-read `service.attach_state(key)`."""
        return frozenset(JobKey(c.host, i) for c in self.changes for i in c.live)


class ClockTick(Message):
    """1 Hz while animations are on: refresh time-derived text (ages, ELAPSED, LEFT) without a store change."""


class ScopeChanged(Message):
    """Host scope changed (``None`` = all hosts)."""

    def __init__(self, scope: str | None) -> None:
        super().__init__()
        self.scope = scope


class HostStatusChanged(Message):
    """A host entered or left a problem state (UNREACHABLE / NEEDS_AUTH / HOSTKEY); drives the transition toast."""

    def __init__(self, host: str, old: str, new: str) -> None:
        super().__init__()
        self.host = host
        self.old = old
        self.new = new


class ActionFinished(Message):
    """Results of one executed action (all plans of one confirm dialog)."""

    def __init__(self, results: Sequence[ActionResult]) -> None:
        super().__init__()
        self.results = tuple(results)
