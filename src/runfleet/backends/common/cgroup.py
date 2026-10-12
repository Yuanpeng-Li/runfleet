"""[WP2] cgroup v2 parsing for Attach (§7.16): `parse_cgroup_block(text) -> models.CgroupStats`, pure.

One block is what the nested `node.cg` section prints for one job directory: `<file> <value>` lines for the
single-value files (`memory.current 10693746688`, `memory.max max`), then `memory.events` as is (`oom 0`,
`oom_kill 0`), the `anon`/`file`/`shmem` lines of `memory.stat` and the `usage_usec` line of `cpu.stat`. Missing
files (memory.peak before Linux 5.19) and unreadable values stay None; `memory.max` "max" (unlimited) is None too.
"""

from __future__ import annotations

from runfleet.models import CgroupStats
from runfleet.util.slurmfmt import split_rows

# key in the block -> CgroupStats field
_KEYS = {"memory.current": "mem_current", "memory.peak": "mem_peak", "memory.max": "mem_max", "anon": "anon",
         "file": "file", "shmem": "shmem", "oom": "oom", "oom_kill": "oom_kill", "usage_usec": "cpu_usage_usec"}


def parse_cgroup_block(text: str) -> CgroupStats:
    """`key value` lines → CgroupStats; unknown keys (memory.events `low`/`high`/`max`, `oom_group_kill`) and
    non-numeric values are ignored. The first occurrence of a key wins."""
    values: dict[str, int | None] = {}
    for line in split_rows(text):
        key, _, value = line.strip().partition(" ")
        field = _KEYS.get(key)
        if field is None or field in values:
            continue
        value = value.strip()
        values[field] = int(value) if value.isdigit() else None    # "max" -> None (unlimited)
    return CgroupStats(**values)


def is_empty(stats: CgroupStats) -> bool:
    """No value at all (the directory existed but nothing was readable)."""
    return all(getattr(stats, f) is None for f in _KEYS.values())
