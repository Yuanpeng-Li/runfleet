"""[WP3] Textual screens.

``FleetScreen`` is the base of every full screen: it receives store changes from the app, answers the questions the
app's global actions ask ("which host / job / targets are current?", "which help contexts apply?") and reports
which hosts it shows so the app can publish visibility levels to the poller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.screen import Screen

from runfleet.models import Job, JobKey, Visibility
from runfleet.ui.messages import ClockTick, FleetChanged

if TYPE_CHECKING:
    from runfleet.ui.app import RunFleetApp


def dialog_list_room(screen_height: int, cap: int, reserved: int = 0) -> int:
    """Rows a list in a dialog may take: at most ``cap``, and no more than the dialog has (``.dialog``: 90 % of the
    screen, less its border and ``reserved`` rows of other content). A fixed ``max-height`` alone lets a short terminal
    clip the list's bottom rows inside the box, so the highlighted row could sit there unseen (OptionList scrolls only
    within its own height)."""
    return max(3, min(cap, screen_height * 9 // 10 - 2 - reserved))


class FleetScreen(Screen[Any]):
    """Base class for RunFleet's full screens (not modals)."""

    HELP_CONTEXTS: tuple[str, ...] = ("global",)
    HELP_TITLE = "RunFleet"

    @property
    def rf(self) -> RunFleetApp:
        return self.app  # type: ignore[return-value]

    # ---- questions the app asks
    def help_contexts(self) -> list[str]:
        return list(self.HELP_CONTEXTS)

    def legend_contexts(self) -> list[str]:
        """Contexts of the help overlay's glyph legend (the key contexts, unless a screen shows more glyphs)."""
        return self.help_contexts()

    def help_title(self) -> str:
        return self.HELP_TITLE

    def current_host(self) -> str | None:
        return None

    def current_job(self) -> Job | None:
        return None

    def action_targets(self) -> list[JobKey]:
        job = self.current_job()
        return [job.key] if job is not None else []

    def visibility(self) -> dict[str, Visibility]:
        """Hosts this screen shows (others are HIDDEN)."""
        return {}

    def escape(self) -> bool:
        """Handle ``esc`` locally (close overlay, clear filter, clear marks); True if consumed."""
        return False

    # ---- data flow
    def refresh_data(self, changes: FleetChanged | None) -> None:
        """Re-read the store; ``None`` = full refresh (mount, resume, scope change)."""

    def on_fleet_changed(self, message: FleetChanged) -> None:
        self.refresh_data(message)

    def on_clock_tick(self, message: ClockTick) -> None:
        self.refresh_time()

    def refresh_time(self) -> None:
        """1 Hz: time-derived text only."""

    def on_resize(self) -> None:
        self.rf.check_size()

    def on_screen_resume(self) -> None:
        if self.rf._host_gone(self):         # its host left the fleet (Settings): close instead of refreshing
            self.app.call_later(self.rf._forget_removed_hosts)
            return
        self.rf.update_visibility()
        self.refresh_data(None)
