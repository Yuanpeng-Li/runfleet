"""Usage rollup (§5.2.11 "Rollup grouping"): grouping keys over `UsageCell`s and `group()`, shared by the Account
screen and `runfleet agent usage`.

Pure. Keys come from the job name, following the run skills' convention
`<task>--<stage>--<variant>--<origin>--s<seed>--a<attempt>`: **experiment** (the default) strips the trailing seed and
attempt fields, so the seeds and attempts of one hypothesis add up; **task** is the first `--` field; **name** is the
name as is. Array elements share a name and collapse on their own. `account`, `partition` and `gpu` group by the
cell's own field.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from runfleet.models import UsageCell

GroupBy = Literal["experiment", "task", "name", "account", "partition", "gpu"]
GROUP_BYS: tuple[GroupBy, ...] = ("experiment", "task", "name", "account", "partition", "gpu")   # `z` cycle order

_SEED_ATTEMPT = re.compile(r"(?:--[sa]\d+)+$")


@dataclass(frozen=True, slots=True)
class RollupRow:
    key: str                                  # the group ("" = no name / no GPU model / untyped)
    jobs: int
    running: int                              # running at report time
    cpu_h: float
    gpu_h: float
    gpu_h_by_model: Mapping[str, float]       # normalised gres type -> hours, largest first; "" = model unknown
    billing_h: float
    charged_su: float                         # cells with an unknown factor count ×1 (and set factor_unknown)
    factor_unknown: bool = False
    accounts: tuple[str, ...] = ()            # sorted


def experiment_key(name: str) -> str:
    """`mytask-v3--train--lr-cosine--scratch--s0--a01` -> `mytask-v3--train--lr-cosine--scratch`; other names as
    they are (a name that is nothing but seed / attempt fields stays whole)."""
    return _SEED_ATTEMPT.sub("", name) or name


def task_key(name: str) -> str:
    """The first `--` field: `mytask-v3--train--…` -> `mytask-v3`; `jupyter` -> `jupyter`."""
    return name.split("--", 1)[0] or name


def group_key(cell: UsageCell, by: GroupBy) -> str:
    if by == "experiment":
        return experiment_key(cell.name)
    if by == "task":
        return task_key(cell.name)
    if by == "name":
        return cell.name
    if by == "account":
        return cell.account
    if by == "partition":
        return cell.partition
    return cell.gpu_model


def group(cells: Iterable[UsageCell], by: GroupBy = "experiment") -> list[RollupRow]:
    """Sum the cells per key; rows by GPU-h, then billing-h, largest first, then by key."""
    sums: dict[str, list[float]] = {}             # jobs, running, cpu_h, gpu_h, billing_h, charged_su
    models: dict[str, dict[str, float]] = {}
    unknown: set[str] = set()
    accounts: dict[str, set[str]] = {}
    for c in cells:
        k = group_key(c, by)
        s = sums.setdefault(k, [0.0] * 6)
        s[0] += c.jobs
        s[1] += c.running
        s[2] += c.cpu_h
        s[3] += c.gpu_h
        s[4] += c.billing_h
        s[5] += c.billing_h if c.charged_su is None else c.charged_su
        if c.charged_su is None:
            unknown.add(k)
        if c.gpu_h > 0:
            m = models.setdefault(k, {})
            m[c.gpu_model] = m.get(c.gpu_model, 0.0) + c.gpu_h
        if c.account:
            accounts.setdefault(k, set()).add(c.account)
    rows = [RollupRow(key=k, jobs=int(s[0]), running=int(s[1]), cpu_h=s[2], gpu_h=s[3],
                      gpu_h_by_model=dict(sorted(models.get(k, {}).items(), key=lambda kv: (-kv[1], kv[0]))),
                      billing_h=s[4], charged_su=s[5], factor_unknown=k in unknown,
                      accounts=tuple(sorted(accounts.get(k, ()))))
            for k, s in sums.items()]
    rows.sort(key=lambda r: (-r.gpu_h, -r.billing_h, r.key))
    return rows
