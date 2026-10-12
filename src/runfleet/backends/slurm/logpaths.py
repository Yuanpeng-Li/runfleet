"""[WP2] Slurm log path expansion (%x/%j/%A/%a...), WorkDir resolution, merged stderr (§5.2.4-§5.2.5).

squeue prints the *unexpanded* `--output` pattern (`…/logs/%x_%j.out`, lyra `slurm-%j.out` relative to WorkDir);
scontrol prints the expanded `StdOut=`/`StdErr=` but forgets the job `MinJobAge` after it ends. These helpers turn
either source into `LogRef`s.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from runfleet.models import LogRef
from runfleet.util.slurmfmt import nullish

_TOKEN = re.compile(r"%(?P<w>\d*)(?P<c>[%AajJNnstuxbrS])")
NO_VAL = "4294967294"            # Slurm's NO_VAL: what %a expands to for a job that is not an array
MAX_WIDTH = 10                   # expand_stdio_fields caps the zero-padding width

LogSource = Literal["scontrol", "squeue", "sacct", "template", "procfd", "cache", "demo"]


@dataclass(frozen=True, slots=True)
class JobNames:
    """Everything a pattern can refer to."""
    job_id_raw: str
    array_job_id: str | None
    array_task: str | None           # "3", or a range "[1-20%1]" / "1-20%1" for a pending summary
    user: str | None
    name: str | None
    first_node: str | None
    workdir: str | None
    restarts: int | None = None      # RestartCnt (%r); None = unknown

    @property
    def task_index(self) -> str | None:
        return self.array_task if self.array_task and self.array_task.isdigit() else None


def expand(pattern: str, names: JobNames) -> str | None:
    """Expand a Slurm filename pattern like slurmstepd's `expand_stdio_fields`; None when a token cannot be expanded
    (yet): a PD job's `%N`, an array summary's `%a`, `%r` without RestartCnt, `%S` (SLUID), unknown user/name; or a
    relative result has no WorkDir to join.

    Slurm semantics: `%a` is NO_VAL (4294967294) and `%b` 4 for a job that is not an array; padding is capped at 10;
    a pattern containing `\\` is not expanded at all, only its backslashes are removed."""
    if "\\" in pattern:
        out = pattern.replace("\\", "")
        return _absolute(out, names.workdir)
    failed = False

    def sub(m: re.Match[str]) -> str:
        nonlocal failed
        c, width = m.group("c"), m.group("w")
        value: str | None
        if c == "%":
            return "%"
        if c == "A":
            value = names.array_job_id or names.job_id_raw
        elif c == "a":
            value = names.task_index if names.array_task else NO_VAL
        elif c in "jJ":
            value = names.job_id_raw
        elif c == "N":
            value = names.first_node.split(".", 1)[0] if names.first_node else None
        elif c in "nt":
            value = "0"
        elif c == "s":
            value = "batch"
        elif c == "u":
            value = names.user
        elif c == "x":
            value = names.name
        elif c == "r":
            value = str(names.restarts) if names.restarts is not None else None
        elif c == "S":
            value = None                             # SLUID (25.11+): not known to RunFleet
        else:                                        # %b: array task id modulo 10
            idx = names.task_index if names.array_task else NO_VAL
            value = str(int(idx) % 10) if idx is not None else None
        if value is None:
            failed = True
            return ""
        return value.zfill(min(int(width), MAX_WIDTH)) if width and value.isdigit() else value

    out = _TOKEN.sub(sub, pattern)
    if failed:
        return None
    return _absolute(out, names.workdir)


def _absolute(out: str, workdir: str | None) -> str | None:
    if not out.startswith("/"):
        if not workdir:
            return None
        out = posixpath.join(workdir, out)
    return posixpath.normpath(out)


def default_pattern(names: JobNames) -> str:
    """Slurm's default `--output` when none was given."""
    return "slurm-%A_%a.out" if names.array_task else "slurm-%j.out"


def _absolute_pattern(pattern: str, workdir: str | None) -> str:
    return pattern if pattern.startswith("/") or not workdir else posixpath.join(workdir, pattern)


def refs_from_patterns(stdout_pat: str | None, stderr_pat: str | None, names: JobNames,
                       source: LogSource = "squeue") -> tuple[LogRef, ...]:
    """squeue STDOUT/STDERR patterns → (stdout, stderr) LogRefs. STDERR `(null)` or equal to STDOUT → merged."""
    out_pat = nullish(stdout_pat) or default_pattern(names)
    err_raw = nullish(stderr_pat)
    merged = err_raw is None or err_raw == out_pat
    out_path = expand(out_pat, names)
    stdout = LogRef(stream="stdout", path=out_path,
                    pattern=None if out_path else _absolute_pattern(out_pat, names.workdir), merged=merged,
                    source=source)
    if merged:
        stderr = LogRef(stream="stderr", path=out_path, pattern=stdout.pattern, merged=True, source=source)
    else:
        assert err_raw is not None
        err_path = expand(err_raw, names)
        stderr = LogRef(stream="stderr", path=err_path,
                        pattern=None if err_path else _absolute_pattern(err_raw, names.workdir), source=source)
    return (stdout, stderr)


def refs_from_paths(stdout: str | None, stderr: str | None, source: LogSource) -> tuple[LogRef, ...]:
    """Expanded paths (scontrol `StdOut=`/`StdErr=`, sacct StdOut/StdErr, jobmeta). Empty StdErr (vega) or equal to
    StdOut (k2) → merged."""
    out = nullish(stdout)
    err = nullish(stderr)
    if out is None:
        return ()
    merged = err is None or err == out
    return (LogRef(stream="stdout", path=out, merged=merged, source=source),
            LogRef(stream="stderr", path=out if merged else err, merged=merged, source=source))


def refs_from_templates(templates: Sequence[str], names: JobNames) -> tuple[LogRef, ...]:
    """First host `log_templates` entry that formats (`{workdir}/logs/{name}_{jobid}.out`), else Slurm's default
    `{workdir}/slurm-{jobid}.out`. Marked `source="template"`: verified by stat when the detail screen opens."""
    values = {"workdir": names.workdir, "name": names.name, "jobid": names.job_id_raw,
              "array_job": names.array_job_id or names.job_id_raw, "task": names.task_index or "",
              "user": names.user}
    for tpl in templates:
        try:
            path = tpl.format(**{k: v for k, v in values.items() if v is not None})
        except (KeyError, IndexError, ValueError):
            continue
        if path.startswith("/"):
            return (LogRef(stream="stdout", path=posixpath.normpath(path), merged=True, source="template"),
                    LogRef(stream="stderr", path=posixpath.normpath(path), merged=True, source="template"))
    if not names.workdir:
        return ()
    base = (f"slurm-{names.array_job_id}_{names.task_index}.out" if names.array_job_id and names.task_index
            else f"slurm-{names.job_id_raw}.out")
    path = posixpath.join(names.workdir, base)
    return (LogRef(stream="stdout", path=path, merged=True, source="template"),
            LogRef(stream="stderr", path=path, merged=True, source="template"))
