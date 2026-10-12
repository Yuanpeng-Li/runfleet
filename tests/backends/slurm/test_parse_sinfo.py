"""[WP2] sinfo node parser, GRES pairing, usability and partition summaries against the real fixtures."""

from __future__ import annotations

import re

import pytest

from runfleet.backends.slurm.commands import SEP
from runfleet.backends.slurm.parse_sinfo import gres_counts, gpu_types, is_planned, parse_gres, parse_sinfo, summarize
from runfleet.models import Node
from runfleet.util.slurmfmt import expand_hostlist
from tests.backends.factories import to_sep
from tests.conftest import sample


def nodes(host: str, sep: str = "|") -> dict[str, Node]:
    text = sample(host, "sinfo_nodes_O")
    pr = parse_sinfo(text if sep == "|" else to_sep(text, sep), sep=sep)
    assert not pr.bad_lines, pr.bad_lines[:2]
    return {n.name: n for n in pr.items}


def test_k2_v100_used_indices() -> None:
    n = nodes("k2")["n-31-04"]
    (g,) = n.gres
    assert (g.type, g.total, g.used, g.used_indices) == ("tesla_v100-pcie-16gb", 4, 3, "0,2-3")
    assert n.state == "mixed" and n.usable
    assert (n.cpus_alloc, n.cpus_idle, n.cpus_other, n.cpus_total) == (6, 50, 0, 56)
    assert n.mem_total_mib == 128000 and n.reason is None


def test_k2_mig_node_planned_is_usable_and_counted() -> None:
    by = nodes("k2")
    n = by["n-31-05"]
    assert n.state == "planned" and n.usable and is_planned(n)
    assert len(gpu_types(n)) == 4
    assert all(g.used == 0 and g.used_indices is None for g in n.gres)
    (part,) = summarize(by.values(), ["gpu-preempt"])
    free_planned_nodes = sum(g.total - g.used for nm in ("n-31-05", "n-31-07", "n-32-03", "n-37-06")
                             for g in by[nm].gres)
    assert part.gpus_free_planned == free_planned_nodes
    assert part.gpus_free >= part.gpus_free_planned > 0
    assert part.gpu_types["nvidia_a100_80gb_pcie_3g.40gb"] == (2, 2)


def test_lyra_gres_without_parens() -> None:
    by = nodes("lyra")
    n = by["lyra-gpu-36-04"]
    (g,) = n.gres
    assert (g.type, g.total, g.used, g.used_indices) == ("v100", 4, 3, "0-2")
    assert by["lyra-gpu-p71-08"].gres[0].used == 0 and by["lyra-gpu-p71-08"].state == "idle"


def test_vega_mixed_dash_usable_and_reserved_not() -> None:
    by = nodes("vega")
    n = by["n-37-01"]
    assert n.state == "mixed" and n.flags == frozenset({"-"}) and n.usable and is_planned(n)
    r = by["n-37-09"]
    assert r.state == "reserved" and not r.usable and r.gres[0].used == 0
    parts = {p.name: p for p in summarize(by.values())}
    assert parts["h200"].states["reserved"] == 1 and parts["h200"].gpus_total == 52
    # Field-probe fact (DESIGN §10.2 says "vega free GPUs > 0", the fixture disagrees and the fact wins): every
    # usable vega node had all its GPUs in use that day; the only idle GPUs sit on the reserved node, which is
    # excluded. So the fixture's free count is 0 ...
    assert sum(p.gpus_free for p in parts.values()) == 0
    # ... and a `mixed-` node with an idle GPU does count, also as "planned":
    text = sample("vega", "sinfo_nodes_O").replace(
        "n-37-01|h200|mixed-|20/44/0/64|384000|163840|29206|gpu:nvidia_h200_nvl:2(S:3-4)|gpu:nvidia_h200_nvl:2(IDX:0-1)",
        "n-37-01|h200|mixed-|20/44/0/64|384000|163840|29206|gpu:nvidia_h200_nvl:2(S:3-4)|gpu:nvidia_h200_nvl:1(IDX:0)")
    h200 = {p.name: p for p in summarize(parse_sinfo(text, sep="|").items)}["h200"]
    assert h200.gpus_free == 1 and h200.gpus_free_planned == 1
    assert h200.gpu_types["nvidia_h200_nvl"] == (1, 52)


def test_unit_separator() -> None:
    assert nodes("k2", SEP).keys() == nodes("k2").keys()


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_sinfo_parts_agrees_with_nodes(host: str) -> None:
    """`sinfo -o '%P|%a|%l|%D|%T|%G|%N'` groups nodes by (partition, state, gres): same counts as the -N rows."""
    by = nodes(host)
    for line in sample(host, "sinfo_parts").splitlines():
        part, _avail, _limit, count, state, _gres, nodelist = line.split("|")
        names = expand_hostlist(nodelist)
        assert len(names) == int(count)
        for nm in names:
            assert part in by[nm].partitions
            assert state.rstrip("-") == by[nm].state


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_sinfo_summary_totals(host: str) -> None:
    """`sinfo -s` A/I/O/T totals of each sampled partition equal the number of -N rows in that partition."""
    by = nodes(host)
    seen = {p for n in by.values() for p in n.partitions}
    for line in sample(host, "sinfo_summary").splitlines()[1:]:
        cols = line.split()
        part = cols[0].rstrip("*")
        if part not in seen:
            continue
        m = re.match(r"(\d+)/(\d+)/(\d+)/(\d+)", cols[3])
        assert m is not None
        assert int(m.group(4)) == sum(1 for n in by.values() if part in n.partitions)


def test_merges_rows_per_node() -> None:
    text = ("n1|a|idle|0/4/0/4|100|0|90|gpu:x:2(S:0)|gpu:x:0(IDX:N/A)|none|\n"
            "n1|b|idle|0/4/0/4|100|0|90|gpu:x:2(S:0)|gpu:x:0(IDX:N/A)|none|\n")
    (n,) = parse_sinfo(text, sep="|").items
    assert n.partitions == ("a", "b")
    parts = summarize([n])
    assert [p.gpus_free for p in parts] == [2, 2]


def test_gres_parsing_edge_cases() -> None:
    assert [(g.name, g.type, g.count) for g in parse_gres("gpu:a30_1g.6gb:4(S:4-7),shard:8")] == \
        [("gpu", "a30_1g.6gb", 4), ("shard", None, 8)]
    assert parse_gres("(null)") == []
    counts = gres_counts("gpu:a:2(S:0),gpu:b:1(S:1)", "gpu:a:1(IDX:0),gpu:b:0(IDX:N/A)")
    assert [(c.type, c.total, c.used, c.used_indices) for c in counts] == [("a", 2, 1, "0"), ("b", 1, 0, None)]


def test_down_and_drain_flags_not_usable() -> None:
    text = ("n1|p|idle*|0/4/0/4|100|0|90|gpu:x:2|gpu:x:0(IDX:N/A)|not responding|\n"
            "n2|p|drained|0/4/0/4|100|0|90|gpu:x:2|gpu:x:0(IDX:N/A)|maint|\n"
            "n3|p|mixed$|0/4/0/4|100|0|90|gpu:x:2|gpu:x:0(IDX:N/A)|none|\n")
    by = {n.name: n for n in parse_sinfo(text, sep="|").items}
    assert not any(n.usable for n in by.values())
    assert by["n1"].reason == "not responding"
    (part,) = summarize(by.values())
    assert part.gpus_free == 0 and part.gpus_total == 6


def test_m3_k2_free_gpus_exclude_cpu_full_nodes_and_mig_is_labelled() -> None:
    """M3 F10/F11 (real k2 sinfo, 2026-09-07): 4 of the 25 'free' GPUs sat on `allocated` nodes (no idle CPU, not
    schedulable); MIG slices were labelled like the parent GPU ('A100-80G 7/7' for seven 1g.10gb slices)."""
    from runfleet.backends.slurm.commands import SEP
    from runfleet.backends.slurm.parse_sinfo import summarize
    from runfleet.util.humanize import gpu_label

    pr = parse_sinfo(sample("k2", "sinfo_nodes_m3"), sep=SEP)
    assert not pr.bad_lines
    gpu = {p.name: p for p in summarize(pr.items)}["gpu-preempt"]
    assert gpu.gpus_total == 138 and gpu.gpus_free == 21
    blocked = [n.name for n in pr.items if n.state == "allocated" and any(g.name == "gpu" and g.total > g.used
                                                                             for g in n.gres)]
    assert set(blocked) >= {"n-33-04", "n-33-05", "n-36-01", "n-36-03"}
    assert gpu_label("nvidia_a100_80gb_pcie_1g.10gb") == "A100-1g.10gb"
    assert gpu_label("nvidia_a100_80gb_pcie") == "A100-80G"
    assert gpu_label("nvidia_h100_nvl_1g.12gb") == "H100-1g.12gb"
