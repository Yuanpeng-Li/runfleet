"""[WP3] Display formatting: durations, ages, bytes, MiB->G, percentages.

Pure string helpers shared by the UI. Every function accepts ``None`` for "unknown" and then returns the dash the
caller asks for (``–`` = unavailable metric, ``—`` = not applicable, §7.7), never ``0``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from rich.cells import cell_len

NA = "–"          # metric unavailable (never render 0 instead)
NONE = "—"        # not applicable (PD elapsed, no GPUs, no limit)
ELLIPSIS = "…"

_Seconds = float | int | timedelta | None


def _secs(value: _Seconds) -> float | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


def clock(value: _Seconds, *, dash: str = NONE) -> str:
    """Slurm-style elapsed time: ``0:01:04``, ``18:54:04``, ``2-19:52:10`` (days only when >= 1 day)."""
    s = _secs(value)
    if s is None:
        return dash
    total = max(0, int(s))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def left(value: _Seconds, *, dash: str = NONE) -> str:
    """Compact duration for 8-cell columns (LEFT, ELAPSED): ``2-19:52`` (>= 1 day, minutes precision) else
    ``4:07:12`` — a 1-day elapsed time must not end in ``…``."""
    s = _secs(value)
    if s is None:
        return dash
    total = max(0, int(s))
    days, rem = divmod(total, 86400)
    if days:
        hours, rem = divmod(rem, 3600)
        return f"{days}-{hours:02d}:{rem // 60:02d}"
    return clock(total)


def age(value: _Seconds, *, dash: str = NA) -> str:
    """One-unit age for chips and columns: ``8s``, ``4m``, ``14h``, ``2d``."""
    s = _secs(value)
    if s is None:
        return dash
    total = max(0, int(s))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h"
    return f"{total // 86400}d"


def span(value: _Seconds, *, dash: str = NA) -> str:
    """Two-unit duration for prose: ``45s``, ``52m``, ``4h07m``, ``2d03h``."""
    s = _secs(value)
    if s is None:
        return dash
    total = max(0, int(s))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        h, m = divmod(total // 60, 60)
        return f"{h}h{m:02d}m" if m else f"{h}h"
    d, h = divmod(total // 3600, 24)
    return f"{d}d{h:02d}h" if h else f"{d}d"


def window(value: _Seconds, *, dash: str = NA) -> str:
    """A look-back window: whole hours up to two days (``24h``, ``36h``), else :func:`age` (``3d``, ``30m``)."""
    s = _secs(value)
    if s is None:
        return dash
    if 3600 <= s <= 2 * 86400 and int(s) % 3600 == 0:
        return f"{int(s) // 3600}h"
    return age(s)


def ago(then: datetime | None, now: datetime, *, dash: str = NA) -> str:
    """``8s ago`` / ``4m ago``; dash when unknown."""
    if then is None:
        return dash
    return f"{age((now - then).total_seconds())} ago"


def local_dt(dt: datetime | None, *, dash: str = NONE) -> str:
    """Local wall time as ``09-06 11:14`` (month-day, the year is implied)."""
    if dt is None:
        return dash
    return dt.astimezone().strftime("%m-%d %H:%M")


def local_hm(dt: datetime | None, *, dash: str = NONE) -> str:
    """Local wall time as ``06:31``."""
    if dt is None:
        return dash
    return dt.astimezone().strftime("%H:%M")


def size(n_bytes: int | float | None, *, dash: str = NA) -> str:
    """Byte counts: ``663 B``, ``12K``, ``9.9G``, ``503G``, ``1.2T`` (binary units, short suffixes)."""
    if n_bytes is None:
        return dash
    value = float(n_bytes)
    if value < 1024:
        return f"{int(value)} B"
    for unit in ("K", "M", "G", "T", "P"):
        value /= 1024
        if value < 1024 or unit == "P":
            return _num(value) + unit
    return dash   # pragma: no cover - loop always returns


def _num(value: float) -> str:
    """One decimal below 10 (``9.9``), integers above (``48``, ``503``)."""
    return f"{value:.1f}" if value < 9.95 else f"{value:.0f}"


def gib(mib: float | None, *, decimals: int | None = None, dash: str = NA) -> str:
    """MiB -> ``24.3G``; ``decimals=None`` picks 1 decimal below 100 G and none above (``281G``)."""
    if mib is None:
        return dash
    g = mib / 1024
    if decimals is None:
        decimals = 1 if g < 99.95 else 0
    return f"{g:.{decimals}f}G"


def gib_pair(used_mib: float | None, total_mib: float | None, *, sep: str = "/", dash: str = NA) -> str:
    """``24.3/40.0G`` (unit once, at the end); ``–/48.0G`` when only the total is known."""
    if used_mib is None and total_mib is None:
        return dash
    total = gib(total_mib, dash=dash)
    if used_mib is None:
        return f"{dash}{sep}{total}"
    return f"{gib(used_mib)[:-1]}{sep}{total}"


def pct(value: float | None, *, dash: str = NA, width: int = 0) -> str:
    """``98%`` (rounded, clamped to 0..999); dash when unknown; right-aligned to ``width``."""
    if value is None:
        text = dash
    else:
        text = f"{max(0, min(999, round(value)))}%"
    return text.rjust(width)


def cpu_load(value: float | None, *, dash: str = NA) -> str:
    """CPU use given in % of one core: ``98%`` up to 999 %, above that in cores (``≈ 61 cores``) — a busy multi-node
    job must not read as a clamped ``999%``."""
    if value is None:
        return dash
    if value >= 999.5:
        return f"≈ {value / 100:.0f} cores"
    return pct(value)


def truncate(text: str, width: int, *, keep_tail: bool = False) -> str:
    """Fit ``text`` into ``width`` cells with ``…``.

    ``keep_tail`` cuts in the middle-right so the suffix stays visible (job names end in seed/attempt markers:
    ``orb-swp--train--cifar-warmup--s43--a02`` in 26 cells is ``orb-swp--train--c…s43--a02``); used from
    ``width >= 16`` (§7.4), else a plain right cut. Widths are terminal cells (CJK characters count 2), so Rich never
    re-truncates the result and undoes the kept tail.
    """
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width == 1:
        return ELLIPSIS
    if keep_tail and width >= 16:
        tail = min(10, width // 3)
        return _head(text, width - tail - 1) + ELLIPSIS + _tail(text, tail)
    return _head(text, width - 1) + ELLIPSIS


_SEPS = "-_.:/ "


def distinct_starts(names: Sequence[str]) -> dict[str, int]:
    """For each name, where the part that tells it apart from its most similar sibling begins: the longest common
    prefix with its neighbours in sorted order, backed up to the start of that segment (after a ``-_.:/`` or space).
    Sweep siblings ``orb-swp--train--cifar-{warmup,const,cosinewarm}--s43--a02`` -> the index of ``warmup`` /
    ``const`` / ``cosinewarm``. Names without a sibling sharing a prefix map to 0."""
    uniq = sorted(set(names))
    out: dict[str, int] = {}
    for i, name in enumerate(uniq):
        lcp = 0
        for j in (i - 1, i + 1):
            if 0 <= j < len(uniq):
                other = uniq[j]
                n = 0
                while n < min(len(name), len(other)) and name[n] == other[n]:
                    n += 1
                lcp = max(lcp, n)
        while lcp > 0 and name[lcp - 1] not in _SEPS:
            lcp -= 1
        out[name] = lcp if lcp < len(name) else 0
    return out


def truncate_name(name: str, width: int, start: int = 0) -> str:
    """A job name in ``width`` cells that keeps what distinguishes it (``start``: see :func:`distinct_starts`):
    ``<head>…<distinct part>`` with the distinct part keeping its seed/attempt tail when it is long too. Without a
    shared prefix (``start`` 0 or tiny) this is ``truncate(keep_tail=True)``."""
    if cell_len(name) <= width or width < 8:
        return truncate(name, width, keep_tail=True)
    if start <= 3 or start >= len(name):
        return truncate(name, width, keep_tail=True)
    region = name[start:]
    room = width - 1 - cell_len(region)
    if room >= 3:                                   # whole distinct part fits: as much head as the rest allows
        return _head(name, min(room, start)) + ELLIPSIS + region
    head = min(start, max(3, width // 4))
    rest = width - head - 1
    return _head(name, head) + ELLIPSIS + truncate(region, rest, keep_tail=rest >= 16)


def _head(text: str, cells: int) -> str:
    out, used = [], 0
    for ch in text:
        w = cell_len(ch)
        if used + w > cells:
            break
        out.append(ch)
        used += w
    return "".join(out)


def _tail(text: str, cells: int) -> str:
    """The last `cells` cells; never starting with a zero-width mark (a combining accent, ZWJ) cut off its base."""
    out = _head(text[::-1], cells)[::-1]
    i = 0
    while i < len(out) and cell_len(out[i]) == 0:
        i += 1
    return out[i:]


def truncate_left(text: str, width: int) -> str:
    """Cut from the left (paths: ``…/logs/x_2332861.out``)."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width == 1:
        return ELLIPSIS
    return ELLIPSIS + _tail(text, width - 1)


_INTERPRETERS = re.compile(r"^(?:\S*/)?(?:python[\d.]*|pypy[\d.]*|bash|sh|zsh|node|ruby|perl|Rscript|julia)$")


def command_label(args: str) -> str:
    """A process command line for a narrow column, nvitop-style: the interpreter's path is dropped so the script and
    its arguments stay visible (``/home/u/.venv/bin/python -m orb.train --lr 3`` -> ``python -m orb.train --lr 3``)."""
    words = args.split(" ")
    if words and words[0].startswith("/"):
        exe = words[0].rsplit("/", 1)[-1]
        if _INTERPRETERS.match(exe) or len(words) == 1:
            words[0] = exe
    return " ".join(words)


def short_gpu_name(name: str | None) -> str:
    """Compact GPU model for narrow columns: ``NVIDIA RTX A6000`` -> ``RTX A6000``, ``nvidia_a100-sxm4-40gb`` ->
    ``A100-SXM4-40GB``, ``nvidia_h200_nvl`` -> ``H200 NVL``, ``NVIDIA RTX PRO 6000 Blackwell ...`` -> ``RTX PRO 6000 BW``.
    """
    if not name:
        return ""
    s = name.strip()
    for prefix in ("NVIDIA ", "nvidia_", "Nvidia ", "Tesla ", "tesla_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s == s.lower():            # Slurm gres type: lower-case with underscores
        s = s.replace("_", " ").upper()
        s = s.replace("GEFORCE ", "")
    s = s.replace(" Blackwell Server Edition", " BW").replace(" Blackwell Workstation Edition", " BW")
    s = s.replace(" Blackwell", " BW").replace(" Max-Q Workstation Edition", " MaxQ")
    return s


def gpu_model(name: str | None) -> str:
    """Shortest recognisable model for ``2×H200`` style labels: ``A100``, ``H200``, ``A6000``, ``PRO6000``, ``3090``."""
    if not name:
        return "GPU"
    s = short_gpu_name(name).upper().replace("-", " ").replace("_", " ")
    words = s.split()
    if "PRO" in words:
        i = words.index("PRO")
        if i + 1 < len(words):
            return "PRO" + words[i + 1]
    for w in words:
        if w in ("RTX", "GEFORCE", "TESLA", "NVIDIA", "QUADRO"):
            continue
        return w
    return words[0] if words else "GPU"


def gpu_label(gpu_type: str | None) -> str:
    """GPU label for counts (``A100-40G``, ``H200``, ``A6000``, ``3090``): model plus memory when the type names it."""
    if not gpu_type:
        return "GPU"
    model = gpu_model(gpu_type)
    lowered = gpu_type.lower()
    mig = re.search(r"(\d+g\.\d+gb)$", lowered)
    if mig:                      # a MIG slice is not the parent GPU: `nvidia_a100_80gb_pcie_1g.10gb` -> A100-1g.10gb
        return f"{model}-{mig.group(1)}"
    digits = ""
    for token in lowered.replace("_", "-").split("-"):
        if token.endswith("gb") and token[:-2].isdigit():
            digits = token[:-2]
    if digits and digits not in model:
        return f"{model}-{digits}G"
    return model


_MIG_TYPE = re.compile(r"\d+g\.\d+gb$")


def is_mig(gpu_type: str | None) -> bool:
    """A MIG slice type (``nvidia_a100_80gb_pcie_1g.10gb``, ``a100_3g.40gb``), not a whole GPU."""
    return bool(gpu_type and _MIG_TYPE.search(gpu_type.lower()))


def gpu_type_counts(types: Mapping[str, tuple[int, int]]) -> list[tuple[str, int, int]]:
    """Per-type ``(free, total)`` merged by display label (``nvidia_a100-sxm4-80gb`` and ``nvidia_a100_80gb_pcie``
    both read ``A100-80G``: one entry, summed), largest total first."""
    merged: dict[str, list[int]] = {}
    for t, (free, total) in types.items():
        acc = merged.setdefault(gpu_label(t), [0, 0])
        acc[0] += free
        acc[1] += total
    return [(label, f, n) for label, (f, n) in sorted(merged.items(), key=lambda kv: (-kv[1][1], kv[0]))]


def gpu_free_split(types: Mapping[str, tuple[int, int]]) -> tuple[int, int, int, int]:
    """``(whole free, whole total, MIG free, MIG total)``: a free MIG slice is not a free GPU."""
    wf = wt = mf = mt = 0
    for t, (free, total) in types.items():
        if is_mig(t):
            mf, mt = mf + free, mt + total
        else:
            wf, wt = wf + free, wt + total
    return wf, wt, mf, mt

