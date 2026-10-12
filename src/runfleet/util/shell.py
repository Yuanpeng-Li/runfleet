"""[WP1] shlex helpers and id/pid/path validators for building remote commands.

Every value that comes from data (job ids, paths, partitions, user names) is validated here and then quoted with
`quote` *inside* a section command, before the framing quotes the whole command again (§4.3).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable

JOBID_RE = re.compile(r"^\d+(_(\d+|\[[\d,\-%]+\]))?(\.(batch|extern|\d+))?$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")          # partition, user, QOS, account, host names
SECTION_NAME_RE = re.compile(r"^[a-z]+\.[a-z0-9_.]+$")


def quote(value: str) -> str:
    """POSIX-shell quote one word."""
    return shlex.quote(value)


def quote_join(values: Iterable[str]) -> str:
    """Quote each word and join with spaces."""
    return " ".join(shlex.quote(v) for v in values)


def is_job_id(value: str) -> bool:
    return bool(JOBID_RE.match(value))


def job_id(value: str) -> str:
    """Return `value` if it is a valid Slurm job id (display or raw form), else raise ValueError."""
    if not is_job_id(value):
        raise ValueError(f"invalid job id {value!r}")
    return value


def name(value: str) -> str:
    """Validate a partition/user/QOS/account token."""
    if not NAME_RE.match(value):
        raise ValueError(f"invalid name {value!r}")
    return value


def pid(value: int | str) -> int:
    """Validate a process id (positive int, not 1)."""
    if isinstance(value, bool):
        raise ValueError(f"invalid pid {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid pid {value!r}") from None
    if n <= 1 or (isinstance(value, str) and not value.isdigit()):
        raise ValueError(f"invalid pid {value!r}")
    return n


def path(value: str) -> str:
    """Validate a remote path: absolute, no NUL, no newline."""
    if not value.startswith("/") or "\0" in value or "\n" in value or "\r" in value:
        raise ValueError(f"invalid path {value!r}")
    return value


def section_name(value: str) -> str:
    if not SECTION_NAME_RE.match(value):
        raise ValueError(f"invalid section name {value!r}")
    return value
