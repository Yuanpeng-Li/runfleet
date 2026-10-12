"""[WP2] sacct parsers (history window, vanished ids, steps).

All take the field list they asked for (fixtures and live commands differ) and the delimiter. Field names may carry
a width suffix (`TRESUsageInMax%400`), which is ignored. A row with the wrong field count is bad (§5.2.3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from runfleet.backends.common import ParseResult
from runfleet.backends.slurm.commands import SEP
from runfleet.backends.slurm.logpaths import JobNames, refs_from_patterns
from runfleet.backends.slurm.parse_squeue import resolve_gpu
from runfleet.backends.slurm.states import STATE_WORD, is_held, map_state
from runfleet.models import Job, JobKey, JobKind, JobStep, LogRef, ResourceAlloc
from runfleet.util.slurmfmt import (expand_hostlist, nullish, parse_duration, parse_exit, parse_int, parse_mem_bytes,
                                    parse_timestamp, parse_tres)
from runfleet.util.slurmfmt import split_rows

_ARRAY = re.compile(r"^(?P<job>\d+)_(?P<task>\d+|\[[^\]]+\])$")
# heterogeneous-job components print as `<het_job_id>+<offset>` (`600+1`, steps `600+1.batch`)
_STEP_ID = re.compile(r"^\d+(?:_(?:\d+|\[[^\]]+\]))?(?:\+\d+)?(?:\.[A-Za-z0-9_+\-]+)?$")


_FREE_TEXT = frozenset({"StdOut", "StdErr", "WorkDir"})
_EXIT = re.compile(r"^\d+:\d+$")


def field_names(fields: Sequence[str]) -> list[str]:
    """`TRESUsageInMax%400` → `TRESUsageInMax`."""
    return [f.split("%", 1)[0] for f in fields]


def _aligned(names: Sequence[str], parts: Sequence[str]) -> bool:
    """A `|` in JobName shifts every later field just like one in a trailing path: only accept the row when the
    shaped fields after it are where they belong (ExitCode `n:n`, State a state word)."""
    if "JobName" not in names:
        return True
    checks = {"ExitCode": _EXIT, "State": STATE_WORD}
    after = names.index("JobName") + 1
    shaped = [(i, checks[nm]) for i, nm in enumerate(names) if i >= after and nm in checks]
    return bool(shaped) and all(rx.match(parts[i].strip()) for i, rx in shaped)


def _rows(text: str, fields: Sequence[str], sep: str, pr: ParseResult[Any]) -> list[tuple[int, str, dict[str, str]]]:
    names = field_names(fields)
    out: list[tuple[int, str, dict[str, str]]] = []
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split(sep)
        if len(parts) > len(names) and sep != SEP and names[-1] == "WorkDir":
            # the `|` fallback with a `|` inside a trailing path (StdOut/StdErr/WorkDir): none of them is trusted
            k = min(i for i in range(len(names)) if all(nm in _FREE_TEXT for nm in names[i:]))
            if _aligned(names[:k], parts[:k]):
                parts = parts[:k] + [""] * (len(names) - k)
        if len(parts) != len(names):
            first = parts[0].strip()
            pr.bad(n, line, f"expected {len(names)} fields, got {len(parts)}", first if _STEP_ID.match(first) else None)
            continue
        out.append((n, line, dict(zip(names, parts, strict=True))))
    return out


def job_from_sacct(row: dict[str, str], *, host: str, tz: timedelta, observed_at: datetime | None,
                   me: str | None) -> Job:
    """One `sacct -X` allocation row → Job (source "sacct"). Raises ValueError on an invalid row."""
    jid = row.get("JobID", "").strip()
    raw_id = (row.get("JobIDRaw") or jid).strip()
    if not _STEP_ID.match(jid) or "." in jid or not raw_id.isdigit():
        raise ValueError(f"bad JobID {jid!r}/{raw_id!r}")
    raw_state = row.get("State", "").strip()
    if not STATE_WORD.match(raw_state):
        raise ValueError(f"bad State {raw_state!r}")
    state, extra = map_state(raw_state)
    m = _ARRAY.match(jid)
    array_job, array_task = (m.group("job"), m.group("task").strip("[]")) if m else (None, None)
    # a het component is keyed like squeue keys it: by its own unique job id (JobIDRaw), not `600+1`
    display = raw_id if "+" in jid else jid
    nodelist = nullish(row.get("NodeList"))
    nodes = expand_hostlist(nodelist)
    alloc_tres = parse_tres(row.get("AllocTRES"))
    req_tres = parse_tres(row.get("ReqTRES"))
    nnodes = parse_int(row.get("NNodes"))
    gpus, gtype, source = resolve_gpu(alloc_tres, (None, None), (None, None), nnodes, req_tres=req_tres)
    mem = parse_mem_bytes(alloc_tres.get("mem") or req_tres.get("mem"))
    code, sig = parse_exit(row.get("ExitCode"))
    user = row.get("User", "").strip()
    reason = nullish(row.get("Reason"))
    raw = dict(row)
    raw.update(extra)
    # sacct prints the *unexpanded* --output pattern (verified on 25.05, 25.11, 26.05 at M3): expand it like squeue's
    logs: tuple[LogRef, ...] = ()
    if nullish(row.get("StdOut")) is not None:
        names = JobNames(job_id_raw=raw_id, array_job_id=array_job, array_task=array_task, user=user or None,
                         name=row.get("JobName") or None, first_node=nodes[0] if nodes else None,
                         workdir=nullish(row.get("WorkDir")))
        logs = refs_from_patterns(row.get("StdOut"), row.get("StdErr"), names, source="sacct")
    return Job(
        key=JobKey(host, display), kind=JobKind.SLURM, name=row.get("JobName", ""), user=user, state=state,
        raw_state=raw_state, job_id_raw=raw_id, reason=reason, held=is_held(reason, raw_state),
        submit_time=parse_timestamp(row.get("Submit"), tz), start_time=parse_timestamp(row.get("Start"), tz),
        end_time=parse_timestamp(row.get("End"), tz) if state.is_terminal else None,
        elapsed=parse_duration(row.get("Elapsed")), time_limit=parse_duration(row.get("Timelimit")),
        exit_code=code, exit_signal=sig, derived_exit=nullish(row.get("DerivedExitCode")),
        partition=nullish(row.get("Partition")), qos=nullish(row.get("QOS")), account=nullish(row.get("Account")),
        alloc=ResourceAlloc(nodes=nnodes, cpus=parse_int(row.get("AllocCPUS")), mem_bytes=mem, gpus=gpus,
                            gpu_type=gtype, tres=alloc_tres or req_tres, gpu_type_source=source),
        nodes=tuple(nodes), nodelist=nodelist, workdir=nullish(row.get("WorkDir")), logs=logs,
        array_job_id=array_job, array_task=array_task, is_mine=me is None or user == me, source="sacct",
        observed_at=observed_at, raw=raw,
    )


def parse_sacct(text: str, fields: Sequence[str], *, host: str, tz: timedelta, observed_at: datetime | None = None,
                me: str | None = None, sep: str = SEP) -> ParseResult[Job]:
    """`sacct -X -P --delimiter=<sep> -n --format=<fields>` (window or `-j ids`) → Jobs keyed by display JobID."""
    pr: ParseResult[Job] = ParseResult()
    for n, line, row in _rows(text, fields, sep, pr):
        try:
            pr.items.append(job_from_sacct(row, host=host, tz=tz, observed_at=observed_at, me=me))
        except ValueError as e:
            first = row.get("JobID", "").strip()
            pr.bad(n, line, str(e), first if _STEP_ID.match(first) else None)
    return pr


def parse_steps(text: str, fields: Sequence[str], *, tz: timedelta, sep: str = SEP) -> ParseResult[JobStep]:
    """`sacct -j <id>` without -X: the allocation row plus `.batch`/`.extern`/`.N` steps → JobSteps."""
    pr: ParseResult[JobStep] = ParseResult()
    for n, line, row in _rows(text, fields, sep, pr):
        sid = row.get("JobID", "").strip()
        raw_state = row.get("State", "").strip()
        if not _STEP_ID.match(sid) or not STATE_WORD.match(raw_state):
            pr.bad(n, line, f"bad step row {sid!r}", sid if _STEP_ID.match(sid) else None)
            continue
        state, _ = map_state(raw_state)
        code, sig = parse_exit(row.get("ExitCode"))
        pr.items.append(JobStep(
            step_id=sid, name=row.get("JobName", ""), state=state, raw_state=raw_state, exit_code=code,
            exit_signal=sig, elapsed=parse_duration(row.get("Elapsed")),
            start=parse_timestamp(row.get("Start"), tz), end=parse_timestamp(row.get("End"), tz),
            nodelist=nullish(row.get("NodeList")), ntasks=parse_int(row.get("NTasks")),
            max_rss_bytes=parse_mem_bytes(row.get("MaxRSS"), default_unit=""),
            ave_cpu=parse_duration(row.get("AveCPU")),
            tres_usage_max=parse_tres(row.get("TRESUsageInMax")), tres_usage_tot=parse_tres(row.get("TRESUsageInTot")),
            total_cpu=parse_duration(row.get("TotalCPU")),
        ))
    return pr


def gputres_seed(text: str) -> bool | None:
    """`slurm.gputres` (§5.1): step rows `JobID|TRESUsageInMax` of the user's last ended GPU job. Any row with
    `gres/gpuutil=` → True (k2, vega); rows without it → False (lyra); no rows (no recent GPU job) → None."""
    rows = [line for line in split_rows(text) if line.strip()]
    if not rows:
        return None
    return any("gres/gpuutil=" in line for line in rows)
