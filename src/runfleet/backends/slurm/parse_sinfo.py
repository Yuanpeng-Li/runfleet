"""[WP2] sinfo node parser and partition summaries (free GPUs).

`sinfo -h -N -p <parts> -O 'NodeList:<sep>,Partition:<sep>,StateLong:<sep>,CPUsState:<sep>,Memory:<sep>,
AllocMem:<sep>,FreeMem:<sep>,Gres:<sep>,GresUsed:<sep>,Reason:<sep>'` prints one row per (node, partition); rows are
merged by node name. Usability and the "planned" bookkeeping follow §5.2.3.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from runfleet.backends.common import ParseResult
from runfleet.backends.slurm.commands import SEP, SINFO_FIELDS
from runfleet.backends.slurm.parse_squeue import split_row
from runfleet.models import GresCount, Node, PartitionSummary
from runfleet.util.slurmfmt import is_planned as is_planned
from runfleet.util.slurmfmt import node_free, nullish, parse_int, split_outside, split_state
from runfleet.util.slurmfmt import split_rows

USABLE_STATES = frozenset({"idle", "mixed", "allocated", "completing", "planned"})
UNUSABLE_FLAGS = frozenset("*~#!%$@^")


@dataclass(frozen=True, slots=True)
class GresItem:
    name: str
    type: str | None
    count: int
    detail: str | None           # text inside the parentheses: "S:0-7", "IDX:0,2-3", "IDX:N/A"


def parse_gres(value: str | None) -> list[GresItem]:
    """`gpu:tesla_v100-pcie-16gb:3(IDX:0,2-3),gpu:a30_1g.6gb:4(S:4-7)` / lyra `gpu:V100:4` → items. Commas inside
    parentheses do not split."""
    v = nullish(value)
    if v is None or v.lower() == "(null)":
        return []
    out: list[GresItem] = []
    for item in split_outside(v):
        body, _, rest = item.partition("(")
        detail = rest[:-1] if rest.endswith(")") else None
        parts = body.split(":")
        if len(parts) < 2 or not parts[-1].isdigit():
            if len(parts) == 1 or (len(parts) == 2 and not parts[1].isdigit()):
                # `gpu` / `gpu:type` without a count = 1 (rare spelling)
                out.append(GresItem(parts[0], parts[1].lower() if len(parts) == 2 else None, 1, detail))
            continue
        gtype = ":".join(parts[1:-1]).lower() or None
        out.append(GresItem(parts[0], gtype, int(parts[-1]), detail))
    return out


def _idx_text(detail: str | None) -> str | None:
    if not detail or not detail.startswith("IDX:"):
        return None
    v = detail[4:]
    return None if v in ("N/A", "") else v


def gres_counts(gres: str | None, used: str | None) -> tuple[GresCount, ...]:
    """Pair configured and used GRES per (name, type): free = total − used."""
    totals: dict[tuple[str, str | None], int] = {}
    for g in parse_gres(gres):
        totals[(g.name, g.type)] = totals.get((g.name, g.type), 0) + g.count
    used_map: dict[tuple[str, str | None], tuple[int, str | None]] = {}
    for g in parse_gres(used):
        n, idx = used_map.get((g.name, g.type), (0, None))
        used_map[(g.name, g.type)] = (n + g.count, _idx_text(g.detail) or idx)
    out: list[GresCount] = []
    for key, total in totals.items():
        n_used, idx = used_map.get(key, (0, None))
        out.append(GresCount(name=key[0], type=key[1], total=total, used=min(n_used, total), used_indices=idx))
    return tuple(out)


def is_usable(base: str, flags: Iterable[str]) -> bool:
    """Counts free GPUs: base in {idle, mixed, allocated, completing, planned} and none of `* ~ # ! % $ @ ^`. The `-`
    flag (planned by backfill, vega `mixed-`) stays usable."""
    return base in USABLE_STATES and not (set(flags) & UNUSABLE_FLAGS)


def parse_sinfo(text: str, *, sep: str = SEP, fields: Sequence[str] = SINFO_FIELDS) -> ParseResult[Node]:
    """Rows → Nodes merged by name (partitions accumulate, first row's numbers win)."""
    pr: ParseResult[Node] = ParseResult()
    by_name: dict[str, Node] = {}
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = split_row(line, sep, len(fields))
        if parts is None:
            pr.bad(n, line, f"expected {len(fields)} fields")
            continue
        row = dict(zip(fields, parts, strict=True))
        name = row["NodeList"].strip()
        state_raw = row["StateLong"].strip()
        if not name or not state_raw:
            pr.bad(n, line, "missing node name or state")
            continue
        partition = row["Partition"].strip().rstrip("*")
        prev = by_name.get(name)
        if prev is not None:
            if partition and partition not in prev.partitions:
                by_name[name] = dataclasses.replace(prev, partitions=(*prev.partitions, partition))
            continue
        base, flags = split_state(state_raw)
        cpus = [parse_int(x) for x in row["CPUsState"].split("/")] if "/" in row["CPUsState"] else [None] * 4
        if len(cpus) != 4:
            cpus = [None] * 4
        mem_total = parse_int(row["Memory"])
        mem_alloc = parse_int(row["AllocMem"])
        reason = nullish(row["Reason"])
        by_name[name] = Node(
            name=name, partitions=(partition,) if partition else (), state=base, flags=flags,
            usable=is_usable(base, flags), cpus_alloc=cpus[0], cpus_idle=cpus[1], cpus_other=cpus[2],
            cpus_total=cpus[3], mem_total_mib=mem_total, mem_alloc_mib=mem_alloc,
            mem_free_mib=parse_int(row["FreeMem"]), gres=gres_counts(row["Gres"], row["GresUsed"]),
            reason=None if reason is None or reason.lower() == "none" else reason,
        )
    pr.items.extend(by_name.values())
    return pr


def gpu_types(node: Node) -> frozenset[str]:
    """Distinct GPU types configured on a node (MIG profiles are distinct types)."""
    return frozenset(g.type for g in node.gres if g.name == "gpu" and g.type)


def summarize(nodes: Iterable[Node], partitions: Sequence[str] | None = None) -> tuple[PartitionSummary, ...]:
    """Per partition: node count, GPU totals, free GPUs on usable nodes with an idle CPU (planned ones also in
    gpus_free_planned),
    per-type (free, total) and node counts per base state."""
    per_part: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        for p in node.partitions:
            per_part[p].append(node)
    order = list(partitions) if partitions else sorted(per_part)
    out: list[PartitionSummary] = []
    for p in order:
        members = per_part.get(p, [])
        if not members and partitions:
            continue
        total = free = planned = 0
        types: dict[str, list[int]] = {}
        states: Counter[str] = Counter()
        for node in members:
            states[node.state] += 1
            for gtype, node_free_n, node_total, node_planned in node_free(node):
                total += node_total
                free += node_free_n
                planned += node_planned
                t = types.setdefault(gtype, [0, 0])
                t[0] += node_free_n
                t[1] += node_total
        out.append(PartitionSummary(name=p, nodes=len(members), gpus_total=total, gpus_free=free,
                                    gpus_free_planned=planned, gpu_types={k: (v[0], v[1]) for k, v in types.items()},
                                    states=dict(states)))
    return tuple(out)
