"""[WP4] Which GPUs count as free: the in-use rule of the GPU views and the stricter workstation pick (free GPUs
picker, `runfleet agent gpus`).

Pure: imports only runfleet.models (no UI, no Rich), shared by ui/render.py, ui/everyday.py and agent/.
"""

from __future__ import annotations

from collections.abc import Iterable

from runfleet.models import GPU

GPU_BUSY_UTIL = 5.0             # % utilisation from which a GPU counts as in use
GPU_BUSY_MEM_MIB = 1024.0       # memory held from which it counts as in use (an idle CUDA context holds a few 100 MiB)
FREE_MEM_MIB = 1024.0           # a GPU is free below 1 GiB used …
FREE_UTIL_PCT = 5.0             # … and 5 % utilisation, with no process on it


def gpu_in_use(g: GPU) -> bool:
    """Whether a GPU should not be counted as free: busy, holding memory (another user's container or PID namespace
    shows 0 % and no processes, yet launching there would OOM), or not sampled at all (unknown is never free)."""
    if g.util_pct is None and g.mem_used_mib is None:
        return True
    return ((g.util_pct or 0) >= GPU_BUSY_UTIL or bool(g.processes)
            or (g.mem_used_mib or 0) >= GPU_BUSY_MEM_MIB)


def is_free(g: GPU) -> bool:
    """The workstation pick: no process, < 1 GiB used and < 5 % utilisation, both readings known."""
    return (not g.processes and g.mem_used_mib is not None and g.mem_used_mib < FREE_MEM_MIB
            and g.util_pct is not None and g.util_pct < FREE_UTIL_PCT)


def split_free(gpus: Iterable[GPU]) -> tuple[list[GPU], list[GPU]]:
    """(free GPUs idlest first, the others by index). Free = :func:`is_free`; ranked by memory used, then
    utilisation, then index."""
    free: list[GPU] = []
    held: list[GPU] = []
    for g in sorted(gpus, key=lambda g: g.index):
        (free if is_free(g) else held).append(g)
    free.sort(key=lambda g: (g.mem_used_mib or 0.0, g.util_pct or 0.0, g.index))
    return free, held
