"""[WP2] Every user's running jobs on the nodes in scope (§5.2.9): `squeue -a -t R,CG -O …` and `getent passwd`.

`parse_occupants` reads the rows of `commands.occupants_cmd`; `nodes_with_jobs` hangs them on the sinfo nodes with
the GPUs each job holds there; `parse_names` reads `user:GECOS` lines.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from runfleet.backends.common import ParseResult
from runfleet.backends.slurm.commands import OCCUPANT_FIELDS, SEP
from runfleet.backends.slurm.parse_squeue import display_id, split_row
from runfleet.models import Node, NodeJob
from runfleet.util.slurmfmt import expand_hostlist, nullish, parse_duration, parse_tres, split_rows, tres_gpu


@dataclass(frozen=True, slots=True)
class Occupant:
    job_id: str                  # display id
    user: str
    account: str | None
    partition: str | None
    state: str
    elapsed_s: int | None
    limit_s: int | None
    nodes: tuple[str, ...]
    gpus: int | None             # the whole job (all nodes)
    gpu_type: str | None
    name: str


def _seconds(value: str | None) -> int | None:
    d = parse_duration(value)
    return int(d.total_seconds()) if d is not None else None


def parse_occupants(text: str, *, sep: str = SEP) -> ParseResult[Occupant]:
    pr: ParseResult[Occupant] = ParseResult()
    fields = OCCUPANT_FIELDS
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = split_row(line, sep, len(fields))
        if parts is None:
            pr.bad(n, line, f"expected {len(fields)} fields")
            continue
        row = dict(zip(fields, parts, strict=True))
        job_id = row["JobID"].strip()
        user = row["UserName"].strip()
        nodes = expand_hostlist(nullish(row["NodeList"]))
        if not job_id.isdigit() or not user or not nodes:
            pr.bad(n, line, "bad JobID, UserName or NodeList")
            continue
        task = nullish(row["ArrayTaskID"])
        tres = parse_tres(row["tres-alloc"])
        gpus, gtype = tres_gpu(tres)
        if gpus is None and tres and "gres/gpu" not in row["tres-alloc"]:
            gpus = 0                                     # no gres/gpu at all: a CPU job (a bad one stays unknown)
        pr.items.append(Occupant(
            job_id=display_id(job_id, nullish(row["ArrayJobID"]), task), user=user,
            account=nullish(row["Account"]), partition=nullish(row["Partition"]),
            state=row["StateCompact"].strip() or "R", elapsed_s=_seconds(row["TimeUsed"]),
            limit_s=_seconds(row["TimeLimit"]), nodes=tuple(nodes), gpus=gpus, gpu_type=gtype,
            name=row["Name"].strip()))
    return pr


def parse_names(text: str) -> dict[str, str | None]:
    """`user:GECOS` lines → user → full name (the first comma field of GECOS; None when empty)."""
    out: dict[str, str | None] = {}
    for line in split_rows(text):
        user, colon, gecos = line.partition(":")
        if colon and user.strip():
            out[user.strip()] = gecos.split(",", 1)[0].strip() or None
    return out


def _per_node(occ: Occupant) -> tuple[int | None, bool]:
    """GPUs the job holds on each of its nodes: exact for one node, an even split otherwise (squeue has no per-node
    split; `scontrol show job -d` has, but one RPC per job is too dear for a whole partition)."""
    if occ.gpus is None:
        return None, False
    if len(occ.nodes) == 1:
        return occ.gpus, True
    if occ.gpus % len(occ.nodes) == 0:
        return occ.gpus // len(occ.nodes), False
    return None, False


def nodes_with_jobs(nodes: Iterable[Node], occupants: Sequence[Occupant], names: Mapping[str, str | None],
                    me: str | None) -> tuple[Node, ...]:
    """Each node with its running jobs (every user; own jobs first, then by GPUs held, then by job id)."""
    by_node: dict[str, list[NodeJob]] = {}
    for occ in occupants:
        gpus, exact = _per_node(occ)
        for name in occ.nodes:
            by_node.setdefault(name, []).append(NodeJob(
                job_id=occ.job_id, user=occ.user, real_name=names.get(occ.user), account=occ.account,
                partition=occ.partition, name=occ.name, state=occ.state, gpus=gpus, gpus_exact=exact,
                gpu_type=occ.gpu_type, elapsed_s=occ.elapsed_s, limit_s=occ.limit_s,
                mine=me is not None and occ.user == me))
    out = []
    for node in nodes:
        jobs = by_node.get(node.name, [])
        jobs.sort(key=lambda j: (not j.mine, -(j.gpus or 0), j.job_id))
        out.append(replace(node, jobs=tuple(jobs)))
    return tuple(out)
