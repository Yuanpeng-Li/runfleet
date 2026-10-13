"""[WP2] Every user's running jobs per node (§5.2.9): command, parsers, collector planning and ingest."""

from __future__ import annotations

import shlex

import pytest

from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.commands import SEP
from runfleet.backends.slurm.parse_occupants import nodes_with_jobs, parse_names, parse_occupants
from runfleet.models import Node
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.backends.factories import at, result, to_sep
from tests.backends.slurm.test_collector import started, tick
from tests.conftest import sample

ROWS = [   # JobID|ArrayJobID|ArrayTaskID|UserName|Account|Partition|StateCompact|TimeUsed|TimeLimit|NumNodes|NodeList|
           # tres-alloc|Name|
    "4014136|4014136|N/A|bob|lab|gpu-preempt|R|4:05:30|2-00:00:00|1|n-31-01|"
    "cpu=8,mem=64G,node=1,billing=8,gres/gpu=5,gres/gpu:nvidia_geforce_rtx_3090=5|ddp|train|",
    "4014495|4014493|22|carol|lab|gpu-preempt|R|12:15|UNLIMITED|1|n-31-04|cpu=4,mem=16G,node=1,billing=4|cpu job|",
    "58106563|58106563|N/A|dave|physics|gpu|R|1:40:36|4:00:00|3|n-31-[01,04],n-32-03|"
    "cpu=6,mem=12G,node=3,billing=6,gres/gpu=3|benz|",
    "58106544|58106544|N/A|erin|chem|gpu|CG|1:45:42|8:00:00|2|n-31-[01,04]|cpu=6,node=2,gres/gpu=3|odd|",
    "4014841|4014841|N/A|alice0001|campus01|gpu-preempt|R|0:10|1-00:00:00|1|n-31-04|cpu=8,node=1,gres/gpu=1|mine|",
]


def text(rows: list[str] = ROWS, sep: str = SEP) -> str:
    return to_sep("\n".join(rows), sep) + "\n"


def sinfo_result() -> SectionResult:
    return result("slurm.sinfo", to_sep(sample("k2", "sinfo_nodes_O").strip(), SEP))


# ------------------------------------------------------------------ commands
def test_commands_quote_and_validate() -> None:
    argv = shlex.split(cmds.occupants_cmd(["n-31-01", "n-31-04"], ["gpu-preempt"], SEP))
    assert argv[:7] == ["squeue", "-a", "-h", "-t", "R,CG", "-w", "n-31-01,n-31-04"]    # by node: shared nodes too
    assert argv[8].split(",")[-1] == f"Name:{SEP}" and len(argv[8].split(",")) == len(cmds.OCCUPANT_FIELDS)
    argv = shlex.split(cmds.occupants_cmd([], ["gpu-preempt", "labpi"], SEP))
    assert argv[5:7] == ["-p", "gpu-preempt,labpi"]                                     # nodes not known yet
    with pytest.raises(ValueError):
        cmds.occupants_cmd([], ["gpu;rm -rf"], SEP)
    with pytest.raises(ValueError):
        cmds.occupants_cmd(["n-1,$(id)"], [], SEP)
    assert shlex.split(cmds.names_cmd(["bob", "carol"]))[:4] == ["getent", "passwd", "bob", "carol"]
    with pytest.raises(ValueError):
        cmds.names_cmd(["$(id)"])


# ------------------------------------------------------------------ parsers
@pytest.mark.parametrize("sep", [SEP, "|"])
def test_parse_occupants(sep: str) -> None:
    pr = parse_occupants(text(sep=sep) + "garbage\n", sep=sep)
    assert len(pr.items) == 5 and len(pr.bad_lines) == 1
    bob, carol, dave, erin, mine = pr.items
    assert (bob.job_id, bob.user, bob.gpus, bob.gpu_type, bob.elapsed_s, bob.limit_s) == \
        ("4014136", "bob", 5, "nvidia_geforce_rtx_3090", 4 * 3600 + 5 * 60 + 30, 2 * 86400)
    assert bob.name == ("ddp|train" if sep == "|" else "ddp|train".replace("|", SEP))   # free text: last field
    assert (carol.job_id, carol.gpus, carol.limit_s, carol.name) == ("4014493_22", 0, None, "cpu job")   # CPU job
    assert dave.nodes == ("n-31-01", "n-31-04", "n-32-03") and dave.gpus == 3 and dave.gpu_type is None
    assert erin.state == "CG" and mine.user == "alice0001"


def test_parse_occupants_keeps_bad_gpu_tres_unknown_and_names_whole() -> None:
    rows = ["1|1|N/A|bob|lab|gpu|R|1:00|2:00|1|n-31-01|cpu=8,gres/gpu=oops|a|",
            "2|2|N/A|bob|lab|gpu|R|1:00|2:00|1|n-31-01|cpu=8,gres/gpu:h100:2|b|",
            "3|3|N/A|bob|lab|gpu|R|1:00|2:00|1|n-31-01|cpu=8,mem=4G|train||"]
    a, b, c = parse_occupants(text(rows, "|"), sep="|").items
    assert a.gpus is None and b.gpus is None and c.gpus == 0                  # unknown vs. a CPU job
    assert c.name == "train|"                                                 # `|` fallback: a trailing `|` is name


def test_parse_names() -> None:
    names = parse_names("bob:Bob Example,Room 1,555\ncarol:\ndave:Dave Sample\n\nnoise\n")
    assert names == {"bob": "Bob Example", "carol": None, "dave": "Dave Sample"}


def test_nodes_with_jobs_splits_gpus_per_node() -> None:
    nodes = tuple(n for n in _sinfo_nodes() if n.name in ("n-31-01", "n-31-04", "n-31-05"))
    out = {n.name: n for n in nodes_with_jobs(nodes, parse_occupants(text()).items, {"bob": "Bob Example"},
                                               "alice0001")}
    n01 = out["n-31-01"].jobs
    assert n01 is not None
    assert [(j.user, j.gpus, j.gpus_exact) for j in n01] == [("bob", 5, True), ("dave", 1, False), ("erin", None, False)]
    assert n01[0].real_name == "Bob Example" and n01[1].real_name is None
    n04 = out["n-31-04"].jobs
    assert n04 is not None and n04[0].mine and n04[0].user == "alice0001"      # own jobs first
    assert out["n-31-05"].jobs == ()                                           # sampled: nobody there


def _sinfo_nodes() -> tuple[Node, ...]:
    c, cx, _ = started()
    out = tick(c, cx, sinfo_result())
    assert out.nodes is not None
    return out.nodes


# ------------------------------------------------------------------ collector
def test_occupants_cadence_is_focused_only_and_brings_sinfo() -> None:
    c, cx, _ = started()
    cad = {k.group: k for k in c.cadences(cx)}
    assert cad["slurm.occupants"].focused_only and cad["slurm.occupants"].interval == 30
    assert cad["slurm.names"].mode == "event"
    secs = {s.name: s for s in c.sections(frozenset({"slurm.occupants"}), cx)}
    assert set(secs) == {"slurm.sinfo", "slurm.occupants"}                    # GPU counts and jobs of one moment
    assert "-p gpu-preempt" in secs["slurm.occupants"].cmd                    # no nodes yet
    tick(c, cx, sinfo_result())
    occ = next(s for s in c.sections(frozenset({"slurm.occupants"}), cx) if s.name == "slurm.occupants")
    nodes = shlex.split(occ.cmd)[6].split(",")
    assert len(nodes) == 30 and nodes[0] == "n-31-01" and occ.meta["nodes"] == nodes
    c._squeue_legacy = True                                                    # legacy -o: no tres-alloc
    assert {s.name for s in c.sections(frozenset({"slurm.occupants"}), cx)} == {"slurm.sinfo"}


def test_ingest_attaches_jobs_then_names_then_expires() -> None:
    c, cx, _ = started()
    c.sections(frozenset({"slurm.occupants"}), cx)
    out = tick(c, cx, sinfo_result(), result("slurm.occupants", text()))
    assert out.nodes is not None and "slurm.names" in out.followups
    n01 = next(n for n in out.nodes if n.name == "n-31-01")
    assert n01.jobs is not None and [j.user for j in n01.jobs] == ["bob", "dave", "erin"] and n01.jobs_at == cx.now
    sec = c.sections(frozenset({"slurm.names"}), cx)[0]
    assert shlex.split(sec.cmd)[2:7] == ["bob", "carol", "dave", "erin", "alice0001"]
    named = tick(c, cx, result("slurm.names", "bob:Bob Example,,,\nalice0001:Alice Example\n"))
    assert named.nodes is not None and "slurm.names" not in named.followups
    n01 = next(n for n in named.nodes if n.name == "n-31-01")
    assert n01.jobs is not None and n01.jobs[0].real_name == "Bob Example"
    assert c.sections(frozenset({"slurm.names"}), cx) == []                    # carol/dave/erin: asked, unknown
    later = tick(c, at(cx, 181), sinfo_result())                               # host left: readings expire
    assert later.nodes is not None and all(n.jobs is None for n in later.nodes)


def test_failed_occupants_keep_the_last_reading_and_report() -> None:
    c, cx, _ = started()
    c.sections(frozenset({"slurm.occupants"}), cx)
    tick(c, cx, sinfo_result(), result("slurm.occupants", text()))
    bad = result("slurm.occupants", "", rc=1, stderr="slurm_load_jobs error: Socket timed out",
                 outcome=SectionOutcome.FAILED)
    out = tick(c, at(cx, 30), bad)
    assert out.nodes is not None and any(n.jobs for n in out.nodes)           # still within 3 min
    assert out.errors and out.errors[0].section == "slurm.occupants"
