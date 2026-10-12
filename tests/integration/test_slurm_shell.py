"""[WP2] Shell integration for Slurm: real SlurmCollector sections → WP1 framing.render → /bin/sh|dash|bash with the
mock Slurm commands first on PATH → collector ingest. Asserts both the parsed jobs and the exact argv issued (§10.4),
plus the prelude's env scrubbing and SLURM_TIME_FORMAT=%s."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from runfleet.backends import registry
from runfleet.backends.base import MonitorView
from runfleet.backends.detect import build_capabilities, detect_sections
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.backends.slurm.commands import SEP, SQUEUE_FIELDS
from runfleet.models import JobState
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import NOW, ctx, host_config
from tests.conftest import sample
from tests.integration.harness import MockEnv, run_sections, shells

ALL_GROUPS = frozenset({"slurm.jobs", "slurm.usage", "slurm.history", "slurm.logstat", "slurm.nodes"})
SITE_ENV = {"SQUEUE_FORMAT": "%.12i %.10P %.10j", "SQUEUE_STATES": "PD", "SACCT_FORMAT": "JobID", "SINFO_FORMAT": "%P",
            "SLURM_CLUSTERS": "other", "SLURM_TIME_FORMAT": "relative"}


@pytest.fixture
def k2env(tmp_path: Path) -> MockEnv:
    return MockEnv(tmp_path, "k2", extra_env=SITE_ENV)


@pytest.mark.parametrize("shell", shells())
def test_poll_cycle_through_real_shell(tmp_path: Path, shell: str) -> None:
    k2env = MockEnv(tmp_path, "k2", extra_env=SITE_ENV, sh=shell)
    cx = ctx("k2", login_host="login02")
    c = SlurmCollector(cx.host)
    env = k2env.env()

    # tick 1: squeue + sacct window
    res = run_sections(c.sections(ALL_GROUPS, cx), env, shell=shell)
    assert {n: r.outcome for n, r in res.items()} == {"slurm.squeue": SectionOutcome.OK,
                                                      "slurm.sacct": SectionOutcome.OK}
    out = c.ingest(res, cx, MonitorView())
    assert out.errors == ()
    assert out.jobs is not None and {"2332861", "2334564", "2333760", "2333095"} <= set(out.jobs)
    assert out.jobs["2332861"].state is JobState.RUNNING and out.jobs["2333095"].state is JobState.COMPLETED
    assert out.jobs["2334564"].start_time is not None and out.jobs["2334564"].est_start

    (sq,) = k2env.calls("squeue")
    assert sq[1:5] == ["--me", "-h", "-O", ",".join(f"{f}:{SEP}" for f in SQUEUE_FIELDS)]
    (sa,) = k2env.calls("sacct")
    assert sa[1:3] == ["-u", sa[2]] and sa[2]                   # "$(id -un)" expanded remotely
    assert sa[3:10] == ["-S", "now-259200seconds", "-E", "now", "-X", "-P", f"--delimiter={SEP}"]
    assert sa[10] == "-n" and sa[11].startswith("--format=JobID,JobIDRaw,JobName,") and sa[11].endswith(",WorkDir")
    # the prelude unset the site's Slurm variables and forced epoch timestamps
    for e in k2env.env_lines():
        assert e["SLURM_TIME_FORMAT"] == "%s"
        assert e["SQUEUE_FORMAT"] == e["SQUEUE_STATES"] == e["SACCT_FORMAT"] == e["SINFO_FORMAT"] == ""
        assert e["SLURM_CLUSTERS"] == ""
    k2env.clear()

    # tick 2: the follow-up scontrol for new ids + usage, logstat and nodes (partitions now known)
    due = frozenset(out.followups | ALL_GROUPS - {"slurm.jobs", "slurm.history"})
    secs = c.sections(due, cx)
    assert {s.name for s in secs} == {"slurm.scontrol", "slurm.sstat", "slurm.logstat", "slurm.sinfo"}
    res = run_sections(secs, env, shell=shell)
    assert res["slurm.scontrol"].outcome is SectionOutcome.OK          # the loop ends with `true`
    assert res["slurm.logstat"].outcome is SectionOutcome.OK           # `stat … 2>&1; true`: errors folded in
    out = c.ingest(res, cx, MonitorView())
    assert out.jobs is not None
    j = out.jobs["2332861"]
    assert j.alloc.per_node[0].gpu_indices == (2,) and j.logs[0].source == "scontrol"
    assert j.gpu is not None and j.gpu.util_mean_pct == 98 and j.gpu.mem_used_mib == 24870
    assert j.logs[0].error == "No such file or directory"              # the fixture path does not exist here
    pd = out.jobs["2334564"]
    assert pd.logs[0].path == "/shared/pilab/pilab/rl-benchmarks/logs/smoke-test_2334564.out"
    assert out.nodes is not None and len(out.nodes) == 30
    assert out.partitions is not None and out.partitions[0].name == "gpu-preempt"
    sc_calls = k2env.calls("scontrol")
    assert sorted(call[-1] for call in sc_calls) == ["2332861", "2332862", "2332863", "2333760", "2334564"]
    assert all(call[1:5] == ["show", "job", "-d", "-o"] for call in sc_calls)
    (ss,) = k2env.calls("sstat")
    assert ss[1:4] == ["-a", "-j", "2332861,2332862,2332863,2333760"] and ss[-1].endswith("TRESUsageInTot%400")
    (si,) = k2env.calls("sinfo")
    assert si[1:5] == ["-h", "-N", "-p", "gpu-preempt"] and si[5] == "-O"


def test_vanished_job_follow_up(k2env: MockEnv, tmp_path: Path) -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    c.ingest(run_sections(c.sections(frozenset({"slurm.jobs"}), cx), k2env.env()), cx, MonitorView())
    # 2332862 leaves squeue; accounting already has it as COMPLETED
    sq = tmp_path / "squeue.txt"
    sq.write_text("\n".join(ln for ln in sample("k2", "squeue_me_O").splitlines() if not ln.startswith("2332862|")))
    acct = tmp_path / "sacct.txt"
    acct.write_text(sample("k2", "sacct_ext").replace(
        "2332863|2332863|orb-swp--train--cifar-cosinewarm--s43--a02", "2332862|2332862|orb-swp--train--cifar-const--s43"
        "--a02", 1).replace("RUNNING|0:0|0:0|None|2026-09-06T11:14:07|2026-09-07T01:50:27|Unknown",
                            "COMPLETED|0:0|0:0|None|2026-09-06T11:14:07|2026-09-07T01:50:27|2026-09-07T06:02:37", 1))
    env = k2env.env(RF_MOCK_FILE_SQUEUE=str(sq), RF_MOCK_FILE_SACCT=str(acct))
    out = c.ingest(run_sections(c.sections(frozenset({"slurm.jobs"}), cx), env), cx, MonitorView())
    assert out.jobs is not None and out.jobs["2332862"].source == "vanished"
    k2env.clear()
    assert "slurm.vanished" in out.followups
    secs = c.sections(frozenset({"slurm.vanished"}), cx)
    assert [s.name for s in secs] == ["slurm.sacct_ids"]
    out = c.ingest(run_sections(secs, env), cx, MonitorView())
    (call,) = k2env.calls("sacct")
    assert call[1:3] == ["-j", "2332862"] and "-X" in call and f"--delimiter={SEP}" in call
    assert out.jobs is not None and out.jobs["2332862"].state is JobState.COMPLETED
    assert out.jobs["2332862"].source == "sacct"


def test_separator_rejected_falls_back(k2env: MockEnv) -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    env = k2env.env(RF_MOCK_SQUEUE_ERR="squeue: error: Invalid field specifier: JobID:\x1f")
    res = run_sections(c.sections(frozenset({"slurm.jobs"}), cx), env)
    assert res["slurm.squeue"].outcome is SectionOutcome.FAILED
    c.ingest(res, cx, MonitorView())
    k2env.clear()
    out = c.ingest(run_sections(c.sections(frozenset({"slurm.jobs"}), cx), k2env.env()), cx, MonitorView())
    (call,) = k2env.calls("squeue")
    assert call[4].startswith("JobID:|,") and out.jobs is not None and len(out.jobs) == 5


def test_cancel_action_argv_and_errors(k2env: MockEnv) -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    out = c.ingest(run_sections(c.sections(frozenset({"slurm.jobs"}), cx), k2env.env()), cx, MonitorView())
    assert out.jobs is not None
    jobs = [out.jobs[i] for i in ("2332861", "2332862", "2332863")]
    plan = c.plan_action("slurm.cancel", jobs, {}, cx)
    res = run_sections(plan.sections, k2env.env(), purpose="action")
    assert k2env.calls("scancel") == [["scancel", "2332861", "2332862", "2332863"]]
    assert all(r.ok for r in c.interpret_action(plan, res, cx))
    err = "scancel: error: Kill job error on job id 2332862: Invalid job id specified"
    res = run_sections(plan.sections, k2env.env(RF_MOCK_SCANCEL_ERR=err), purpose="action")
    results = {r.target.id: r for r in c.interpret_action(plan, res, cx)}
    assert results["2332862"].message == "job already ended" and not results["2332862"].ok
    assert results["2332861"].ok


def test_array_summary_cancel_argv(tmp_path: Path) -> None:
    env = MockEnv(tmp_path, "lyra")
    sq = tmp_path / "parts.txt"
    sq.write_text(sample("lyra", "squeue_parts_O"))
    cx = ctx("lyra")
    c = SlurmCollector(cx.host)
    out = c.ingest(run_sections(c.sections(frozenset({"slurm.jobs"}), cx), env.env(RF_MOCK_FILE_SQUEUE=str(sq))), cx,
                   MonitorView())
    assert out.jobs is not None
    job = dataclasses.replace(out.jobs["31270818_[1-24%1]"], is_mine=True)
    plan = c.plan_action("slurm.cancel", [job], {}, cx)
    run_sections(plan.sections, env.env(), purpose="action")
    assert env.calls("scancel") == [["scancel", "31270818_[1-24]"]]     # throttle stripped, no glob expansion


def test_detect_probe_through_shell(tmp_path: Path) -> None:
    env = MockEnv(tmp_path, "k2")
    reg = registry.collectors()
    host = host_config("k2")
    res = run_sections(detect_sections(host, reg), env.env(), purpose="detect")
    assert res["nvidia.version"].outcome is SectionOutcome.MISSING      # no nvidia-smi on the login node
    caps = build_capabilities(res, host, NOW, reg)
    assert caps.profile == ("slurm", "procfs")
    assert {"squeue", "sacct", "sinfo", "scontrol", "sstat", "scancel", "sacctmgr"} <= caps.tools
    assert caps.slurm_version == "25.05.8" and caps.cluster_name == "k2" and caps.gpu_tres is True
    assert caps.associations == (("k2", "campus01", "", "gpu-priority,normal,labpi"),)
    assert "JobIDRaw" in caps.sacct_fields and "StdOut" not in caps.sacct_fields
    assert abs(caps.clock_skew.total_seconds()) < 10 ** 9


def test_lyra_empty_queue_and_missing_sstat(tmp_path: Path) -> None:
    env = MockEnv(tmp_path, "lyra", mocks=("squeue", "sacct"))
    cx = ctx("lyra")
    c = SlurmCollector(cx.host)
    res = run_sections(c.sections(frozenset({"slurm.jobs", "slurm.history"}), cx), env.env())
    assert res["slurm.squeue"].outcome is SectionOutcome.EMPTY
    out = c.ingest(res, cx, MonitorView())
    assert out.jobs is not None and all(j.source == "sacct" for j in out.jobs.values())
    assert out.jobs["31327454"].state is JobState.FAILED and out.jobs["31327454"].exit_code == 2
