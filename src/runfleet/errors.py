"""[WP1] Exception classes only: RunFleetError, ConfigError, ActionError, Attach* (ErrorKind lives in models)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runfleet.models import AttachOffer, JobKey


class RunFleetError(Exception):
    """Base class of every RunFleet exception."""


class ConfigError(RunFleetError):
    """Invalid configuration; `key_path` locates the offending key, e.g. "host[2].poll_jobs"."""

    def __init__(self, key_path: str, message: str) -> None:
        super().__init__(f"{key_path}: {message}" if key_path else message)
        self.key_path = key_path


class ActionError(RunFleetError):
    """An action cannot be planned or executed (e.g. no eligible target)."""


class AttachError(RunFleetError):
    """Attach (§7.16) cannot take a sample now; the message is one line for the GPU/Procs tab."""


class AttachBlocked(AttachError):
    """Nothing can read this job now: attach off, read-only mode with srun the only path, job not running, host not
    connected, a non-Slurm job. `reason` is the user-facing line (also the message); `hint` optionally says how to
    enable a path (the exact config line and its cost)."""

    def __init__(self, reason: str, *, hint: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


class AttachConsentRequired(AttachError):
    """srun is the path and the user has not confirmed it for this job: show `offer` (a models.AttachOffer: title,
    body, preview) with the choices "once" / "allow for this job (this session)" / "cancel", then call
    `FleetAPI.attach_now(key, consent=...)` again. There is never a "don't ask again" choice."""

    def __init__(self, key: JobKey, offer: AttachOffer) -> None:
        super().__init__(offer.reason)
        self.key = key
        self.offer = offer
