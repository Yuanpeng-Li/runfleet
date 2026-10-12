"""[WP4] Snapshot -> JSON export (schema "runfleet.snapshot/1", DESIGN.md §8.5).

Shape: `{"schema", "generated_at", "hosts": [{"name", "status", "caps", "jobs", "gpus", "procs", "nodes",
"partitions"}]}`; every value is `models.to_json` of the §3.3 type. Hosts keep config order; jobs are sorted by id.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from runfleet.core.store import FleetStore, HostState
from runfleet.models import ConnState, HostSnapshot, Job, StateGroup, to_json

SCHEMA = "runfleet.snapshot/1"
# DEGRADED (some section failed) is a successful snapshot with errors; DISABLED is the user's choice, not a failure.
FAILED_STATES = frozenset({ConnState.UNREACHABLE, ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.RETRYING,
                           ConnState.NEW, ConnState.CONNECTING})

Encoder = Callable[[Any], Any]


def host_failed(state: HostState) -> bool:
    """A host "failed" for `snapshot` (exit code 4): its last attempt failed, it needs a login/host key, or it never
    answered (still NEW/CONNECTING after collect_once)."""
    status = state.status
    return status.consecutive_failures > 0 or status.state in FAILED_STATES


def _selected(store: FleetStore, hosts: Iterable[str] | None) -> list[tuple[str, HostState]]:
    if hosts is None:
        return list(store.hosts.items())
    wanted = list(dict.fromkeys(hosts))
    return [(name, store.hosts[name]) for name in wanted if name in store.hosts]


def failed_hosts(store: FleetStore, hosts: Iterable[str] | None = None) -> list[str]:
    return [name for name, state in _selected(store, hosts) if host_failed(state)]


def host_document(name: str, state: HostState, encode: Encoder) -> dict[str, Any]:
    snap = state.snapshot
    return {
        "name": name,
        "status": encode(state.status),
        "caps": encode(snap.caps) if snap is not None and snap.caps is not None else None,
        "jobs": [encode(job) for _, job in sorted(snap.jobs.items())] if snap else [],
        "gpus": [encode(gpu) for gpu in snap.gpus] if snap else [],
        "procs": [encode(proc) for proc in snap.procs] if snap else [],
        "nodes": [encode(node) for node in snap.nodes] if snap else [],
        "partitions": [encode(part) for part in snap.partitions] if snap else [],
    }


def build_snapshot(store: FleetStore, *, generated_at: datetime, hosts: Iterable[str] | None = None,
                   encode: Encoder | None = None) -> dict[str, Any]:
    """The `runfleet snapshot --json` document for `hosts` (default: every host of the store)."""
    enc = encode or to_json
    return {"schema": SCHEMA, "generated_at": enc(generated_at),
            "hosts": [host_document(name, state, enc) for name, state in _selected(store, hosts)]}


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


STATE_WIDTH = 40


def summary_rows(store: FleetStore, hosts: Iterable[str] | None = None) -> list[tuple[str, ...]]:
    """Plain-text table for `snapshot` without --json: host, state, job counts, GPUs, nodes, latency."""
    order = (StateGroup.ACTIVE, StateGroup.QUEUED, StateGroup.FAILED, StateGroup.ENDED, StateGroup.CANCELLED,
             StateGroup.DONE, StateGroup.UNKNOWN)
    short = {StateGroup.ACTIVE: "active", StateGroup.QUEUED: "queued", StateGroup.FAILED: "failed",
             StateGroup.ENDED: "ended", StateGroup.CANCELLED: "cancelled", StateGroup.DONE: "done",
             StateGroup.UNKNOWN: "unknown"}
    rows: list[tuple[str, ...]] = [("HOST", "STATE", "MY JOBS", "MY GPUS", "NODES", "LATENCY")]
    for name, state in _selected(store, hosts):
        snap = state.snapshot
        counts: dict[StateGroup, int] = {}
        mine = [j for j in (snap.jobs.values() if snap else ()) if j.is_mine]    # like the Fleet cards
        for job in mine:
            counts[job.state.group] = counts.get(job.state.group, 0) + 1
        jobs = " · ".join(f"{counts[g]} {short[g]}" for g in order if counts.get(g)) or "—"
        status = state.status
        state_text = status.state.value + (f" ({status.error.summary})" if status.error else "")
        if len(state_text) > STATE_WIDTH:          # a long error must not push the other columns off screen
            state_text = state_text[: STATE_WIDTH - 1] + "…"
        latency = f"{status.latency_s:.2f}s" if status.latency_s is not None else "—"
        rows.append((name, state_text, jobs, my_gpus(snap, mine) if snap else "—",
                     str(len(snap.nodes)) if snap else "—", latency))
    return rows


def my_gpus(snap: HostSnapshot, mine: Iterable[Job]) -> str:
    """GPUs held by my running jobs: the allocation on a Slurm host (`14`), `in use/total` on a workstation
    (`2/4`, distinct indices of my tasks; the GPU count is never the number of nvidia-smi rows of a cluster)."""
    active = [j for j in mine if j.state.group is StateGroup.ACTIVE]
    if snap.partitions or snap.nodes or (snap.caps is not None and "slurm" in snap.caps.profile):
        return str(sum(j.alloc.gpus or 0 for j in active))
    used = {i for j in active for n in j.alloc.per_node for i in n.gpu_indices}
    held = len(used) or sum(j.alloc.gpus or 0 for j in active)
    return f"{held}/{len(snap.gpus)}" if snap.gpus else str(held)


def format_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() + "\n" for r in rows)
