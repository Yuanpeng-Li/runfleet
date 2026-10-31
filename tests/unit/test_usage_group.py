"""Usage rollup grouping keys and group() (§5.2.11 "Rollup grouping")."""

from __future__ import annotations

import pytest

from runfleet.models import UsageCell
from runfleet.util import usage


def _cell(name: str, *, account: str = "pilab_02", partition: str = "gpu", qos: str = "normal",
          gpu_model: str = "rtx6000", jobs: int = 1, running: int = 0, cpu_h: float = 8.0, gpu_h: float = 1.0,
          billing_h: float = 72.0, charged_su: float | None = 72.0) -> UsageCell:
    return UsageCell(name=name, account=account, partition=partition, qos=qos, gpu_model=gpu_model, jobs=jobs,
                     running=running, cpu_h=cpu_h, gpu_h=gpu_h, billing_h=billing_h, charged_su=charged_su)


@pytest.mark.parametrize(("name", "experiment", "task"), [
    ("mytask-v3--train--lr-cosine--scratch--s0--a01", "mytask-v3--train--lr-cosine--scratch", "mytask-v3"),
    ("mytask-v3--train--lr-cosine--scratch--s12", "mytask-v3--train--lr-cosine--scratch", "mytask-v3"),
    ("mytask-v3--eval--base--a02", "mytask-v3--eval--base", "mytask-v3"),
    ("jupyter", "jupyter", "jupyter"),
    ("sweep-s0", "sweep-s0", "sweep-s0"),                          # a single dash is not a field separator
    ("run--seed0", "run--seed0", "run"),
    ("--s0--a01", "--s0--a01", "--s0--a01"),                       # nothing left: the name stays whole
    ("", "", ""),
])
def test_grouping_keys(name: str, experiment: str, task: str) -> None:
    assert usage.experiment_key(name) == experiment
    assert usage.task_key(name) == task
    assert usage.group_key(_cell(name), "name") == name


def test_seeds_and_attempts_of_one_experiment_add_up() -> None:
    cells = [
        _cell("orb-swp--train--cifar--scratch--s0--a01", jobs=2, running=1, gpu_model="rtx6000", gpu_h=10,
              qos="low", charged_su=0.0, billing_h=80, account="alice07", partition="free-gpu"),
        _cell("orb-swp--train--cifar--scratch--s1--a01", gpu_model="l40s", gpu_h=4, account="alice07"),
        _cell("orb-swp--train--cifar--scratch--s1--a02", gpu_model="rtx6000", gpu_h=2, charged_su=None,
              qos="mystery"),
        _cell("orb-swp--eval--cifar--fork", gpu_h=1),
        _cell("jupyter", gpu_model="", gpu_h=0, cpu_h=30, billing_h=30, charged_su=30),
    ]
    rows = usage.group(cells)
    assert [r.key for r in rows] == ["orb-swp--train--cifar--scratch", "orb-swp--eval--cifar--fork", "jupyter"]
    top = rows[0]
    assert top.jobs == 4 and top.running == 1
    assert top.gpu_h == 16 and dict(top.gpu_h_by_model) == {"rtx6000": 12, "l40s": 4}
    assert list(top.gpu_h_by_model) == ["rtx6000", "l40s"]          # largest first
    assert top.billing_h == 80 + 72 + 72
    assert top.charged_su == 0 + 72 + 72                           # the unknown factor counts ×1 ...
    assert top.factor_unknown                                      # ... and says so
    assert top.accounts == ("alice07", "pilab_02")
    assert rows[2].gpu_h_by_model == {} and not rows[2].factor_unknown


def test_group_by_task_and_fields() -> None:
    cells = [_cell("a--x--s0", account="lab", partition="gpu", gpu_model="v100", gpu_h=3),
             _cell("a--y--s0", account="lab2", partition="gpuxl", gpu_model="l40s", gpu_h=5),
             _cell("b", account="lab", partition="gpu", gpu_model="", gpu_h=1)]
    assert [(r.key, r.gpu_h) for r in usage.group(cells, "task")] == [("a", 8), ("b", 1)]
    assert [(r.key, r.jobs) for r in usage.group(cells, "account")] == [("lab2", 1), ("lab", 2)]
    assert [r.key for r in usage.group(cells, "partition")] == ["gpuxl", "gpu"]
    by_gpu = usage.group(cells, "gpu")
    assert [r.key for r in by_gpu] == ["l40s", "v100", ""]          # "" = untyped / unknown model
    assert usage.group([], "name") == []
    assert set(usage.GROUP_BYS) == {"experiment", "task", "name", "account", "partition", "gpu"}


def test_array_elements_collapse_and_ties_sort_by_key() -> None:
    cells = [_cell("arr", gpu_h=0, billing_h=1, jobs=10), _cell("arr", gpu_h=0, billing_h=1, jobs=5),
             _cell("zz", gpu_h=0, billing_h=2), _cell("aa", gpu_h=0, billing_h=2)]
    rows = usage.group(cells, "name")
    assert [(r.key, r.jobs) for r in rows] == [("aa", 1), ("arr", 15), ("zz", 1)]
