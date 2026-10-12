"""[WP3] util/humanize.py: display formatting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runfleet.util import humanize as h


@pytest.mark.parametrize(("secs", "out"), [
    (0, "0:00:00"), (64, "0:01:04"), (4 * 3600 + 7 * 60 + 12, "4:07:12"), (18 * 3600 + 54 * 60 + 4, "18:54:04"),
    (3 * 86400, "3-00:00:00"), (2 * 86400 + 19 * 3600 + 52 * 60 + 10, "2-19:52:10"), (-5, "0:00:00"),
])
def test_clock(secs: int, out: str) -> None:
    assert h.clock(secs) == out
    assert h.clock(timedelta(seconds=secs)) == out


def test_clock_none_uses_dash() -> None:
    assert h.clock(None) == "—"
    assert h.clock(None, dash="?") == "?"


@pytest.mark.parametrize(("secs", "out"), [
    (3600, "1:00:00"), (2 * 86400 + 19 * 3600 + 52 * 60 + 59, "2-19:52"), (86400, "1-00:00"), (None, "—"),
])
def test_left(secs: int | None, out: str) -> None:
    assert h.left(secs) == out


@pytest.mark.parametrize(("secs", "out"), [
    (8, "8s"), (59.9, "59s"), (60, "1m"), (240, "4m"), (14 * 3600, "14h"), (2 * 86400 + 5, "2d"), (None, "–"),
])
def test_age(secs: float | None, out: str) -> None:
    assert h.age(secs) == out


@pytest.mark.parametrize(("secs", "out"), [
    (45, "45s"), (52 * 60, "52m"), (4 * 3600 + 7 * 60, "4h07m"), (18 * 3600, "18h"), (2 * 86400 + 3 * 3600, "2d03h"),
    (3 * 86400, "3d"),
])
def test_span(secs: int, out: str) -> None:
    assert h.span(secs) == out


def test_window() -> None:
    assert h.window(86400) == "24h"
    assert h.window(36 * 3600) == "36h"
    assert h.window(3 * 86400) == "3d"
    assert h.window(1800) == "30m"
    assert h.window(None) == "–"


def test_ago_and_local_times() -> None:
    now = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)
    assert h.ago(now - timedelta(minutes=4), now) == "4m ago"
    assert h.ago(None, now) == "–"
    assert h.local_dt(now) == "09-07 10:42"      # conftest pins TZ=UTC
    assert h.local_hm(now) == "10:42"
    assert h.local_dt(None) == "—"


@pytest.mark.parametrize(("n", "out"), [
    (663, "663 B"), (12 * 1024, "12K"), (int(9.9 * 2**30), "9.9G"), (503 * 2**30, "503G"), (None, "–"),
    (2 * 2**40, "2.0T"),
])
def test_size(n: int | None, out: str) -> None:
    assert h.size(n) == out


def test_gib_and_pairs() -> None:
    assert h.gib(24.3 * 1024) == "24.3G"
    assert h.gib(281 * 1024) == "281G"
    assert h.gib(None) == "–"
    assert h.gib_pair(24.3 * 1024, 40 * 1024) == "24.3/40.0G"
    assert h.gib_pair(96.1 * 1024, 281 * 1024) == "96.1/281G"
    assert h.gib_pair(None, 48 * 1024) == "–/48.0G"
    assert h.gib_pair(None, None) == "–"
    assert h.gib_pair(0, 48 * 1024, sep=" / ") == "0.0 / 48.0G"


def test_pct() -> None:
    assert h.pct(97.6) == "98%"
    assert h.pct(None) == "–"
    assert h.pct(5, width=4) == "  5%"
    assert h.pct(-3) == "0%"


def test_truncate_keeps_tail_when_wide() -> None:
    name = "orb-swp--train--cifar-cosinewarm--s43--a02"
    out = h.truncate(name, 34, keep_tail=True)
    assert len(out) == 34 and out.endswith("s43--a02") and "…" in out
    # review: at 80 columns NAME is ~26 cells and seeds/attempts were cut away; the tail is kept from 16 cells
    narrow = h.truncate(name, 26, keep_tail=True)
    assert len(narrow) == 26 and narrow.endswith("s43--a02")
    assert h.truncate(name, 15, keep_tail=True) == name[:14] + "…"   # < 16 wide: plain right cut
    cjk = h.truncate("实验名称很长的作业--s43", 16, keep_tail=True)        # widths in cells, tail kept
    from rich.cells import cell_len
    assert cell_len(cjk) <= 16 and cjk.endswith("s43")
    assert h.truncate("short", 10) == "short"
    assert h.truncate("abc", 1) == "…"
    assert h.truncate("abc", 0) == ""


def test_truncate_left() -> None:
    assert h.truncate_left("/a/x.out", 20) == "/a/x.out"
    assert h.truncate_left("/a/b/c/logs/x.out", 10) == "…ogs/x.out"
    out = h.truncate_left("/shared/pilab/orb/logs/run_2332861.out", 16)
    assert len(out) == 16 and out.startswith("…") and out.endswith("2332861.out")


@pytest.mark.parametrize(("name", "label", "model", "short"), [
    ("nvidia_a100-sxm4-40gb", "A100-40G", "A100", "A100-SXM4-40GB"),
    ("nvidia_h200_nvl", "H200", "H200", "H200 NVL"),
    ("NVIDIA RTX A6000", "A6000", "A6000", "RTX A6000"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "PRO6000", "PRO6000", "RTX PRO 6000 BW"),
    ("nvidia_geforce_rtx_3090", "3090", "3090", "RTX 3090"),
    ("tesla_v100-pcie-16gb", "V100-16G", "V100", "V100-PCIE-16GB"),
])
def test_gpu_names(name: str, label: str, model: str, short: str) -> None:
    assert h.gpu_label(name) == label
    assert h.gpu_model(name) == model
    assert h.short_gpu_name(name) == short


def test_gpu_names_empty() -> None:
    assert h.gpu_label(None) == "GPU"
    assert h.gpu_model("") == "GPU"
    assert h.short_gpu_name(None) == ""


def test_command_label_drops_interpreter_paths() -> None:
    """Review: the process COMMAND lost the script name ('/data/home/u/projects/orb/.venv/bin/pyth…')."""
    assert h.command_label("/data/home/u/projects/orb/.venv/bin/python -m orb.train --lr 3") == \
        "python -m orb.train --lr 3"
    assert h.command_label("/usr/bin/bash run.sh") == "bash run.sh"
    assert h.command_label("/opt/tool/bin/mytool --x") == "/opt/tool/bin/mytool --x"   # not an interpreter
    assert h.command_label("python train.py") == "python train.py"
    assert h.gpu_label("nvidia_geforce_rtx_3090") == "3090"


def test_gpu_type_counts_merge_same_label_and_mig_split() -> None:
    """M3 F10 residual: SXM4 and PCIe A100-80G read the same label -> one summed entry; MIG slices apart."""
    types = {"nvidia_a100-sxm4-80gb": (0, 8), "nvidia_a100_80gb_pcie": (2, 2), "nvidia_geforce_rtx_3090": (0, 76),
             "nvidia_a100_80gb_pcie_1g.10gb": (7, 7), "nvidia_h100_80gb_hbm3_1g.12gb": (3, 7), "a30": (1, 3)}
    counts = h.gpu_type_counts(types)
    labels = [c[0] for c in counts]
    assert len(labels) == len(set(labels)) and ("A100-80G", 2, 10) in counts and counts[0] == ("3090", 0, 76)
    assert h.is_mig("nvidia_a100_80gb_pcie_3g.40gb") and not h.is_mig("nvidia_a100_80gb_pcie")
    assert h.gpu_free_split(types) == (3, 89, 10, 14)


def test_free_gpus_text_separates_mig() -> None:
    from runfleet.models import PartitionSummary
    from runfleet.ui import render

    mig = PartitionSummary("gpu-preempt", 10, 103, 13, 0, {"nvidia_a100_80gb_pcie": (2, 2), "a30": (1, 1),
                                                           "nvidia_a100_80gb_pcie_1g.10gb": (10, 100)}, {})
    assert render.free_gpus(mig).plain == "3/3 +10 MIG"
    plain = PartitionSummary("h200", 13, 52, 4, 0, {"nvidia_h200_nvl": (4, 52)}, {})
    assert render.free_gpus(plain).plain == "4/52" and render.free_gpus(plain, sep=" / ").plain == "4 / 52"
