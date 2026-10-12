"""[WP2] Log path expansion (%x/%j/%A/%a/%N/%u/%%/width), merged stderr, templates and defaults (§5.2.4-§5.2.5)."""

from __future__ import annotations

from runfleet.backends.slurm.logpaths import (JobNames, expand, refs_from_paths, refs_from_patterns,
                                              refs_from_templates)


def names(**kw: object) -> JobNames:
    base: dict[str, object] = {"job_id_raw": "31270824", "array_job_id": "31270818", "array_task": "3", "user": "u",
                               "name": "run", "first_node": "g-01.cluster", "workdir": "/w"}
    base.update(kw)
    return JobNames(**base)


def test_tokens() -> None:
    n = names()
    assert expand("%x_%j.out", n) == "/w/run_31270824.out"
    assert expand("/o/%A_%a.out", n) == "/o/31270818_3.out"
    assert expand("/o/%4a-%N-%u-%%-%s-%t-%n-%b.out", n) == "/o/0003-g-01-u-%-batch-0-0-3.out"
    assert expand("/o/%J", n) == "/o/31270824"


def test_unexpandable() -> None:
    assert expand("/o/%a.out", names(array_task="1-20%1")) is None      # array summary
    assert expand("/o/%N.out", names(first_node=None)) is None          # pending job
    assert expand("rel.out", names(workdir=None)) is None
    plain = names(array_job_id=None, array_task=None)
    # Slurm's expand_stdio_fields: %a is NO_VAL and %b is NO_VAL % 10 for a job that is not an array
    assert expand("/o/%A_%a.out", plain) == "/o/31270824_4294967294.out"
    assert expand("logs/%x_%A_%b.out", plain) == "/w/logs/run_31270824_4.out"
    assert expand("/l/%x_%j_r%r.out", names()) is None                   # %r without RestartCnt
    assert expand("/l/%x_%j_r%r.out", names(restarts=2)) == "/l/run_31270824_r2.out"
    assert expand("/l/%S.out", names()) is None                          # SLUID (25.11+)
    assert expand("/l/%12j.out", names()) == "/l/0031270824.out"         # padding capped at 10
    assert expand("/l/job\\%j.out", names()) == "/l/job%j.out"          # a backslash disables expansion


def test_merged_rules() -> None:
    n = names(array_job_id=None, array_task=None, job_id_raw="1")
    refs = refs_from_patterns("/l/%j.out", "(null)", n)
    assert refs[0].merged and refs[1].merged and refs[1].path == "/l/1.out"
    refs = refs_from_patterns("/l/%j.out", "/l/%j.out", n)
    assert refs[1].merged
    refs = refs_from_patterns("/l/%j.out", "/l/%j.err", n)
    assert not refs[1].merged and refs[1].path == "/l/1.err"
    refs = refs_from_patterns("(null)", "(null)", n)
    assert refs[0].path == "/w/slurm-1.out"                             # Slurm's default --output
    assert refs_from_paths("/o", "", "scontrol")[1].merged
    assert refs_from_paths("/o", "/o", "scontrol")[1].merged
    assert not refs_from_paths("/o", "/e", "scontrol")[1].merged
    assert refs_from_paths(None, "/e", "scontrol") == ()


def test_templates_and_default() -> None:
    n = names(array_job_id=None, array_task=None, job_id_raw="42")
    refs = refs_from_templates(["{workdir}/logs/{name}_{jobid}.out"], n)
    assert refs[0].path == "/w/logs/run_42.out" and refs[0].source == "template"
    assert refs_from_templates(["{nope}/x"], n)[0].path == "/w/slurm-42.out"
    arr = names()
    assert refs_from_templates([], arr)[0].path == "/w/slurm-31270818_3.out"
    assert refs_from_templates([], names(workdir=None)) == ()
