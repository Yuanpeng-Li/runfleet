"""[WP2] Slurm durations, timestamps, memory, TRES, hostlist, state-suffix parsing; nullish().

Pure helpers shared by the Slurm parsers (DESIGN.md §5, §5.2.3). Every function accepts the raw text Slurm prints and
returns `None` for placeholders instead of raising.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta, timezone
from itertools import islice

NULLISH = frozenset({"", "N/A", "n/a", "(null)", "None", "None assigned", "Unknown", "[N/A]", "[Not Supported]",
                     "[Unknown Error]"})

# Duration words Slurm prints instead of a time (TimeLimit/TimeLeft/Elapsed).
_NO_DURATION = frozenset({"UNLIMITED", "INVALID", "NOT_SET", "PARTITION_LIMIT", "INFINITE"})

_DURATION = re.compile(r"^(?:(?P<d>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+)(?:\.(?P<f>\d+))?$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
_MEM = re.compile(r"^(?P<n>\d+(?:\.\d+)?)(?P<u>[KMGTP]?)(?:[nc])?$", re.IGNORECASE)
_UNITS = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}

STATE_FLAG_CHARS = "*~#!%$@^-+"

HOSTLIST_CAP = 4096


def nullish(value: str | None) -> str | None:
    """Placeholder normalization (§5): `N/A`, `(null)`, `None assigned`, `[Not Supported]`, ... → None."""
    if value is None:
        return None
    v = value.strip()
    return None if v in NULLISH else v


def parse_int(value: str | None) -> int | None:
    v = nullish(value)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    """A finite float, else None: `nan`/`inf` never reach the models or `snapshot --json` (not valid JSON)."""
    v = nullish(value)
    if v is None:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def parse_duration(value: str | None) -> timedelta | None:
    """`[D-]HH:MM:SS`, `MM:SS`, `M:SS(.fff)` → timedelta; UNLIMITED/INVALID/NOT_SET/garbage → None.

    The `.extern` step's garbage AveCPU (`213503982334-14:25:51`) overflows timedelta and is reported as None.
    """
    v = nullish(value)
    if v is None or v.upper() in _NO_DURATION:
        return None
    m = _DURATION.match(v)
    if m is None:
        return None
    try:
        return timedelta(days=int(m.group("d") or 0), hours=int(m.group("h") or 0), minutes=int(m.group("m")),
                         seconds=int(m.group("s")), microseconds=int((m.group("f") or "0")[:6].ljust(6, "0")))
    except OverflowError:
        return None


def parse_timestamp(value: str | None, tz_offset: timedelta) -> datetime | None:
    """Slurm timestamp → aware UTC datetime.

    All digits = epoch seconds (live path, `SLURM_TIME_FORMAT=%s`); `0` means unset. A naive ISO value (fixtures, a
    cluster that ignored the variable) is localized with the host's `date +%z` offset.
    """
    v = nullish(value)
    if v is None:
        return None
    if v.isdigit():
        n = int(v)
        return datetime.fromtimestamp(n, tz=UTC) if n > 0 else None
    if _ISO.match(v):
        local = datetime.fromisoformat(v).replace(tzinfo=timezone(tz_offset))
        return local.astimezone(UTC)
    return None


def parse_mem_bytes(value: str | None, default_unit: str = "M") -> int | None:
    """Slurm memory (`48G`, `10398988K`, `24870M`, `4000Mn`) → bytes; a bare number uses `default_unit`."""
    v = nullish(value)
    if v is None:
        return None
    m = _MEM.match(v)
    if m is None:
        return None
    unit = (m.group("u") or default_unit).upper()
    if unit not in _UNITS:
        return None
    return int(float(m.group("n")) * (1024 ** _UNITS[unit]))


def parse_mem_mib(value: str | None, default_unit: str = "M") -> float | None:
    b = parse_mem_bytes(value, default_unit)
    return None if b is None else b / (1024 * 1024)


def parse_exit(value: str | None) -> tuple[int | None, int | None]:
    """sacct/scontrol `ExitCode` `2:0` → (2, 0)."""
    v = nullish(value)
    if v is None or ":" not in v:
        return (parse_int(v), None)
    code, _, sig = v.partition(":")
    return (parse_int(code), parse_int(sig))


def split_outside(text: str, sep: str = ",", open_ch: str = "(", close_ch: str = ")") -> list[str]:
    """Split on `sep` outside parentheses/brackets: `gpu:a:3(IDX:0,2-3),gpu:b:1` → two items."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in open_ch:
            depth += 1
        elif ch in close_ch and depth > 0:
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s for s in out if s != ""]


def parse_tres(value: str | None) -> dict[str, str]:
    """`cpu=8,mem=48G,gres/gpu:a100=1` → {"cpu": "8", "mem": "48G", "gres/gpu:a100": "1"}; `(null)` → {}."""
    v = nullish(value)
    if v is None:
        return {}
    out: dict[str, str] = {}
    for item in v.split(","):
        key, eq, val = item.partition("=")
        if eq and key:
            out[key.strip()] = val.strip()
    return out


def tres_gpu(tres: dict[str, str]) -> tuple[int | None, str | None]:
    """GPU count and the *single* typed GPU key of a TRES dict (lower-cased); several types → type None (the count
    is then the aggregate `gres/gpu`, else the sum of the typed values)."""
    count = parse_int(tres.get("gres/gpu"))
    typed = {k.split(":", 1)[1].lower(): v for k, v in tres.items() if k.startswith("gres/gpu:")}
    if len(typed) != 1:
        counts = [parse_int(v) for v in typed.values()]
        if count is None and counts and all(c is not None for c in counts):
            count = sum(c for c in counts if c is not None)
        return count, None
    (gtype, n), = typed.items()
    return (count if count is not None else parse_int(n)), gtype


def parse_tres_per(value: str | None) -> tuple[int | None, str | None]:
    """squeue `tres-per-node`/`tres-per-job`: `gres/gpu:nvidia_h200_nvl:2`, `gres/gpu:1`, `gres:gpu:2` → (n, type);
    several GPU entries (`gres/gpu:a100:1,gres/gpu:v100:1`) → (their sum, the type only if they all share it)."""
    v = nullish(value)
    if v is None:
        return None, None
    found: list[tuple[int, str | None]] = []
    for item in v.split(","):
        parts = re.split(r"[:/]", item.strip())
        if len(parts) < 2 or parts[0] != "gres" or parts[1] != "gpu":
            continue
        rest = parts[2:]
        if not rest:
            found.append((1, None))
        elif len(rest) == 1:
            found.append((int(rest[0]), None) if rest[0].isdigit() else (1, rest[0].lower()))
        else:
            count = int(rest[-1]) if rest[-1].isdigit() else 1
            found.append((count, ":".join(rest[:-1] if rest[-1].isdigit() else rest).lower()))
    if not found:
        return None, None
    types = {t for _, t in found}
    return sum(n for n, _ in found), types.pop() if len(types) == 1 else None


def split_state(value: str) -> tuple[str, frozenset[str]]:
    """sinfo/scontrol node state → (base lower-case, flags): `mixed-` → ("mixed", {"-"}); `MIXED+PLANNED` → ("mixed",
    {"+"}) with the extra states dropped; `drained*` → ("drained", {"*"})."""
    v = value.strip().lower()
    flags: set[str] = set()
    if "+" in v:
        v = v.split("+", 1)[0]
        flags.add("+")
    while v and v[-1] in STATE_FLAG_CHARS:
        flags.add(v[-1])
        v = v[:-1]
    return v, frozenset(flags)


def parse_index_list(value: str | None) -> tuple[int, ...]:
    """`0-1`, `0,2-3`, `N/A` → physical GPU indices."""
    v = nullish(value)
    if v is None:
        return ()
    out: list[int] = []
    for part in v.split(","):
        a, dash, b = part.partition("-")
        if not a.strip().isdigit() or (dash and not b.strip().isdigit()):
            return ()
        lo = int(a)
        hi = int(b) if dash else lo
        out.extend(range(lo, hi + 1))
    return tuple(out)


ARRAY_TASK_CAP = 5000


def expand_array_tasks(task: str | None, cap: int = ARRAY_TASK_CAP) -> list[int] | None:
    """Array task list of a summary row → sorted task ids: `1-99:2%5` → [1, 3, …, 99], `1,3,7-9` → [1, 3, 7, 8, 9].
    None if Slurm truncated the list (`...`), it is malformed, or it holds more than `cap` tasks."""
    v = re.sub(r"%\d+$", "", (task or "").strip().strip("[]"))
    if not v or "..." in v:
        return None
    out: set[int] = set()
    for part in v.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part.strip())
        if m is None:
            return None
        lo = int(m[1])
        hi = int(m[2]) if m[2] else lo
        step = int(m[3]) if m[3] else 1
        if hi < lo or step < 1 or len(out) + (hi - lo) // step + 1 > cap:
            return None
        out.update(range(lo, hi + 1, step))
    return sorted(out)


def compress_tasks(tasks: list[int]) -> str:
    """Sorted task ids → Slurm's bracket body: [1, 2, 3, 5, 7] → `1-3,5,7`."""
    parts: list[str] = []
    i = 0
    while i < len(tasks):
        j = i
        while j + 1 < len(tasks) and tasks[j + 1] == tasks[j] + 1:
            j += 1
        parts.append(str(tasks[i]) if j == i else f"{tasks[i]}-{tasks[j]}")
        i = j + 1
    return ",".join(parts)


def _expand_ranges(spec: str, cap: int = HOSTLIST_CAP) -> list[str] | None:
    """`01-03,05` → ["01", "02", "03", "05"] keeping zero padding (at most `cap` items); None if malformed."""
    out: list[str] = []
    for part in spec.split(","):
        a, dash, b = part.partition("-")
        if not a.isdigit() or (dash and not b.isdigit()):
            return None
        if not dash:
            out.append(a)
        else:
            width = len(a) if a.startswith("0") or len(a) == len(b) else 0
            stop = min(int(b), int(a) + cap - len(out) - 1)          # never materialize `n[0-999999999]`
            out.extend(str(i).zfill(width) for i in range(int(a), stop + 1))
        if len(out) >= cap:
            return out[:cap]
    return out


def expand_hostlist(value: str | None, cap: int = HOSTLIST_CAP) -> list[str]:
    """Slurm hostlist → names: `n-[01-03,05]`, `lyra-gpu-s71-[00-03]`, `a[1-2]-b[3-4]`, `x,y`; at most `cap` names."""
    v = nullish(value)
    if v is None:
        return []
    out: list[str] = []
    for item in split_outside(v, ",", "[", "]"):
        names = [""]
        for literal, ranges in re.findall(r"([^\[]*)(?:\[([^\]]*)\])?", item):
            if literal == "" and ranges == "":
                continue
            expanded = _expand_ranges(ranges, cap) if ranges else [""]
            if expanded is None:
                expanded = [f"[{ranges}]"]
            names = list(islice((n + literal + e for n in names for e in expanded), cap))
        out.extend(names)
        if len(out) >= cap:
            return out[:cap]
    return out


def seconds_prefix(value: str | None) -> int | None:
    """`300 sec`, `30`, `task=30,energy=0` → leading integer seconds (scontrol show config values)."""
    v = nullish(value)
    if v is None:
        return None
    m = re.search(r"(?:^|task=)(\d+)", v)
    return int(m.group(1)) if m else None


def split_rows(text: str) -> list[str]:
    """Rows of command output: split on "\\n" only (a trailing "\\r" dropped). `str.splitlines()` also breaks on
    \\x0b, \\x0c, \\x1c-\\x1e, \\x85, U+2028/2029, which job names, commands and paths may contain."""
    return [line.removesuffix("\r") for line in text.split("\n")]
