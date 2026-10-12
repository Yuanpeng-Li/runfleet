"""[WP2] squeue (-O and legacy -o) parsers.

`parse_squeue` reads `squeue -h -O 'F1:<sep>,F2:<sep>,…'` (every field followed by the separator, Command last) into
`Job`s with everything squeue alone can tell. `resolve_gpu` implements the GPU count/type resolution order of §5.2.3
and is reused by the collector once scontrol / sinfo / cached facts are known.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal

from runfleet.backends.common import ParseResult
from runfleet.backends.slurm.commands import SEP, SQUEUE_FIELDS
from runfleet.backends.slurm.logpaths import JobNames, refs_from_patterns
from runfleet.backends.slurm.states import STATE_WORD, is_held, map_state
from runfleet.models import Job, JobKey, JobKind, JobState, ResourceAlloc
from runfleet.util.slurmfmt import (expand_hostlist, nullish, parse_duration, parse_int, parse_mem_bytes,
                                    parse_timestamp, parse_tres, parse_tres_per, tres_gpu)
from runfleet.util.slurmfmt import split_rows

GpuSource = Literal["tres-alloc", "scontrol", "tres-req", "sinfo", "cache", "nvidia-smi"]

# Legacy `-o` codes → the -O field names used by the row builder. `%i` (display id, `123_[1-5%1]`) and `%R` (reason or
# nodelist) have no -O twin and are split here.
LEGACY_CODES: dict[str, str] = {
    "%A": "JobID", "%F": "ArrayJobID", "%K": "ArrayTaskID", "%j": "Name", "%u": "UserName", "%T": "State",
    "%t": "StateCompact", "%r": "Reason", "%M": "TimeUsed", "%l": "TimeLimit", "%L": "TimeLeft", "%D": "NumNodes",
    "%C": "NumCPUs", "%m": "MinMemory", "%P": "Partition", "%q": "QOS", "%a": "Account", "%V": "SubmitTime",
    "%S": "StartTime", "%e": "EndTime", "%N": "NodeList", "%B": "BatchHost", "%b": "tres-per-node", "%Z": "WorkDir",
    "%o": "Command", "%i": "%i", "%R": "%R",
}
_DISPLAY_ID = re.compile(r"^(?P<job>\d+)(?:_(?P<task>\d+|\[[^\]]+\]))?$")


def display_id(job_id: str, array_job_id: str | None, array_task: str | None) -> str:
    """`JobID` for plain jobs, else `ArrayJobID_ArrayTaskID` with a range in brackets (`31270818_[1-24%1]`)."""
    if array_task is None:
        return job_id
    task = array_task if array_task.isdigit() or array_task.startswith("[") else f"[{array_task}]"
    return f"{array_job_id or job_id}_{task}"


def resolve_gpu(tres_alloc: Mapping[str, str], req_node: tuple[int | None, str | None],
                req_job: tuple[int | None, str | None], nnodes: int | None, *,
                req_tres: Mapping[str, str] | None = None, scontrol_type: str | None = None,
                node_types: Sequence[frozenset[str]] = (), cached_type: str | None = None,
                ) -> tuple[int, str | None, GpuSource | None]:
    """GPU count and type, first hit wins (§5.2.3): (1) the single typed key in tres-alloc; (2) scontrol JOB_GRES /
    per-node GRES; (3) tres-per-node / tres-per-job (or sacct ReqTRES); (4) the single GPU type in the sinfo Gres of
    all the job's nodes; (5) the jobmeta cache. Types are lower-cased."""
    alloc_n, alloc_type = tres_gpu(dict(tres_alloc))
    req_n, req_tres_type = tres_gpu(dict(req_tres or {}))
    count = alloc_n
    if count is None and req_node[0] is not None:
        count = req_node[0] * (nnodes or 1)
    if count is None:
        count = req_job[0] if req_job[0] is not None else req_n
    if not count:
        return 0, None, None
    if alloc_type:
        return count, alloc_type, "tres-alloc"
    if scontrol_type:
        return count, scontrol_type.lower(), "scontrol"
    req_type = req_node[1] or req_job[1] or req_tres_type
    if req_type:
        return count, req_type.lower(), "tres-req"
    if node_types and all(node_types):
        union = frozenset().union(*node_types)
        if len(union) == 1:
            return count, next(iter(union)).lower(), "sinfo"
    if cached_type:
        return count, cached_type.lower(), "cache"
    return count, None, None


def _alloc_from_row(row: Mapping[str, str], nodes: Sequence[str]) -> ResourceAlloc:
    tres_alloc = parse_tres(row.get("tres-alloc"))
    req_node = parse_tres_per(row.get("tres-per-node"))
    req_job = parse_tres_per(row.get("tres-per-job"))
    nnodes = parse_int(row.get("NumNodes"))
    gpus, gtype, source = resolve_gpu(tres_alloc, req_node, req_job, nnodes)
    mem = parse_mem_bytes(tres_alloc.get("mem"))
    if mem is None:
        per_node = parse_mem_bytes(row.get("MinMemory"))
        mem = per_node * (nnodes or 1) if per_node else None
    return ResourceAlloc(nodes=nnodes, cpus=parse_int(row.get("NumCPUs")), mem_bytes=mem, gpus=gpus,
                         gpu_type=gtype, tres=tres_alloc, gpu_type_source=source)


def job_from_row(row: Mapping[str, str], *, host: str, tz: timedelta, observed_at: datetime | None,
                 me: str | None, display: str | None = None) -> Job:
    """One validated squeue row (field name → raw value) → Job. Raises ValueError on an invalid row."""
    job_id = row.get("JobID", "").strip()
    if not job_id.isdigit():
        raise ValueError(f"bad JobID {job_id!r}")
    raw_state = row.get("State", "").strip()
    if not STATE_WORD.match(raw_state):
        raise ValueError(f"bad State {raw_state!r}")
    state, extra = map_state(raw_state)
    elapsed = parse_duration(row.get("TimeUsed"))
    if nullish(row.get("TimeUsed")) is not None and elapsed is None:
        raise ValueError(f"bad TimeUsed {row.get('TimeUsed')!r}")
    array_job = nullish(row.get("ArrayJobID"))
    array_task = nullish(row.get("ArrayTaskID"))
    if array_task is None:
        array_job = None
    did = display or display_id(job_id, array_job, array_task)
    nodelist = nullish(row.get("NodeList"))
    nodes = expand_hostlist(nodelist)
    reason = nullish(row.get("Reason"))
    user = row.get("UserName", "").strip()
    name = row.get("Name", "")
    workdir = nullish(row.get("WorkDir"))
    start = parse_timestamp(row.get("StartTime"), tz)
    end = parse_timestamp(row.get("EndTime"), tz)
    names = JobNames(job_id_raw=job_id, array_job_id=array_job, array_task=array_task, user=user or None,
                     name=name or None, first_node=nodes[0] if nodes else None, workdir=workdir,
                     restarts=parse_int(row.get("RestartCnt")))
    raw = {k: v for k, v in row.items()}
    raw.update(extra)
    return Job(
        key=JobKey(host, did), kind=JobKind.SLURM, name=name, user=user, state=state, raw_state=raw_state,
        job_id_raw=job_id, reason=reason, held=is_held(reason, raw_state),
        submit_time=parse_timestamp(row.get("SubmitTime"), tz),
        start_time=start, est_start=state is JobState.PENDING and start is not None,
        end_time=end if state.is_terminal else None,
        elapsed=elapsed, time_limit=parse_duration(row.get("TimeLimit")),
        partition=nullish(row.get("Partition")), qos=nullish(row.get("QOS")), account=nullish(row.get("Account")),
        alloc=_alloc_from_row(row, nodes), nodes=tuple(nodes), nodelist=nodelist,
        batch_host=nullish(row.get("BatchHost")), workdir=workdir, command=nullish(row.get("Command")),
        logs=refs_from_patterns(row.get("STDOUT"), row.get("STDERR"), names) if "STDOUT" in row else (),
        array_job_id=array_job, array_task=array_task, restarts=parse_int(row.get("RestartCnt")) or 0,
        is_mine=me is None or user == me, source="squeue", observed_at=observed_at, raw=raw,
    )


_FREE_TEXT = ("STDOUT", "STDERR", "WorkDir", "Command")


def split_row(line: str, sep: str, nfields: int) -> list[str] | None:
    """`line.removesuffix(sep).split(sep)`; extras beyond the field count are re-joined into the last field."""
    parts = line.removesuffix(sep).split(sep)
    if len(parts) < nfields:
        return None
    if len(parts) > nfields:
        parts = parts[:nfields - 1] + [sep.join(parts[nfields - 1:])]
    return parts


def parse_squeue(text: str, *, host: str, tz: timedelta, observed_at: datetime | None = None, me: str | None = None,
                 sep: str = SEP, fields: Sequence[str] = SQUEUE_FIELDS) -> ParseResult[Job]:
    """`squeue -h -O` output → Jobs. Invalid rows (a `|` in Name with sep `|` shifts State) go to bad_lines and,
    when the first field is a job id, to bad_ids — never "vanished" (§5.2.3)."""
    pr: ParseResult[Job] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = split_row(line, sep, len(fields))
        first = line.split(sep, 1)[0].strip()
        if parts is None:
            pr.bad(n, line, f"expected {len(fields)} fields", first if first.isdigit() else None)
            continue
        row = dict(zip(fields, parts, strict=True))
        if sep != SEP and line.removesuffix(sep).count(sep) >= len(fields):
            # the `|` fallback with a `|` inside a free-text path: which one holds it is unknowable, so none is
            # trusted (scontrol supplies WorkDir/StdOut) instead of shifting text from one field into the next
            row.update((f, "") for f in _FREE_TEXT if f in row)
        try:
            pr.items.append(job_from_row(row, host=host, tz=tz, observed_at=observed_at, me=me))
        except ValueError as e:
            pr.bad(n, line, str(e), first if first.isdigit() else None)
    return pr


def _legacy_row(codes: Sequence[str], parts: Sequence[str]) -> tuple[dict[str, str], str | None]:
    row: dict[str, str] = {}
    display: str | None = None
    for code, value in zip(codes, parts, strict=True):
        name = LEGACY_CODES.get(code)
        if name is None:
            continue
        if name == "%i":
            m = _DISPLAY_ID.match(value.strip())
            if m is None:
                raise ValueError(f"bad %i {value!r}")
            display = value.strip()
            row.setdefault("JobID", m.group("job"))
            if m.group("task"):
                row.setdefault("ArrayJobID", m.group("job"))
                row.setdefault("ArrayTaskID", m.group("task").strip("[]"))
        elif name == "%R":
            v = value.strip()
            if v.startswith("(") and v.endswith(")"):
                row["Reason"] = v[1:-1]
            else:
                row["NodeList"] = v
        else:
            row[name] = value
    return row, display


def parse_squeue_legacy(text: str, spec: str, *, host: str, tz: timedelta, observed_at: datetime | None = None,
                        me: str | None = None) -> ParseResult[Job]:
    """Legacy `squeue -h -o <spec>` output (fields joined by `|`, Command/`%o` last) → Jobs. The spec is a parameter
    because the fixtures and the fallback command use different code lists."""
    codes = spec.split("|")
    pr: ParseResult[Job] = ParseResult()
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = split_row(line, "|", len(codes))
        first = line.split("|", 1)[0].strip()
        bad_id = first if _DISPLAY_ID.match(first) else None
        if parts is None:
            pr.bad(n, line, f"expected {len(codes)} fields", bad_id)
            continue
        try:
            row, display = _legacy_row(codes, parts)
            pr.items.append(job_from_row(row, host=host, tz=tz, observed_at=observed_at, me=me, display=display))
        except ValueError as e:
            pr.bad(n, line, str(e), bad_id)
    return pr
