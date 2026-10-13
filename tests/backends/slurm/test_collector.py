"""[WP2] SlurmCollector: planning, live/history merge (§5.2.4), vanished jobs, jobinfo, sstat, sinfo, hints, details."""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from runfleet.backends.base import CollectorOutput, HostContext, MonitorView
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.backends.slurm.commands import SEP
from runfleet.models import ErrorKind, JobState
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.backends.factories import at, ctx, host_config, result, to_sep
from tests.conftest import sample

INTERVAL_GROUPS = frozenset({"slurm.jobs", "slurm.usage", "slurm.history", "slurm.logstat", "slurm.nodes"})


def squeue(host: str = "k2", name: str = "squeue_me_O", *, drop: tuple[str, ...] = (), sep: str = SEP,
           edit: tuple[str, str] | None = None) -> SectionResult:
    lines = [ln for ln in sample(host, name).splitlines() if ln.split("|", 1)[0] not in drop]
    text = "\n".join(lines)
    if edit:
        text = text.replace(*edit)
    return result("slurm.squeue", to_sep(text, sep))


def sacct_rows(host: str = "k2", *, edit: tuple[str, str] | None = None, name: str = "slurm.sacct") -> SectionResult:
    text = sample(host, "sacct_ext").rstrip("\n")
    if edit:
        text = text.replace(*edit)
    return result(name, to_sep(text, SEP))


def with_tot(text: str) -> str:
    """Fixture sstat rows lack TRESUsageInTot: append Tot = Ave (NTasks is 1 in every fixture row)."""
    out = []
    for line in text.splitlines():
        cols = line.split("|")
        out.append("|".join([*cols, cols[5]]))
    return "\n".join(out)


def tick(c: SlurmCollector, cx: HostContext, *results: SectionResult) -> CollectorOutput:
    return c.ingest({r.name: r for r in results}, cx, MonitorView())


def started(host: str = "k2", **kw: object) -> tuple[SlurmCollector, HostContext, CollectorOutput]:
    cx = ctx(host, **kw)
    c = SlurmCollector(cx.host)
    out = tick(c, cx, squeue(host))
    return c, cx, out


# ------------------------------------------------------------------ planning
def test_cadences() -> None:
    cx = ctx("k2")
    cad = {c.group: c for c in SlurmCollector(cx.host).cadences(cx)}
    assert cad["slurm.jobs"].interval == 15 and cad["slurm.jobs"].floor == 10 and cad["slurm.jobs"].mode == "interval"
    assert cad["slurm.jobinfo"].mode == "event" and cad["slurm.vanished"].mode == "event"
    assert cad["slurm.usage"].interval == 60 and cad["slurm.usage"].floor == 30
    assert cad["slurm.usage"].focused_factor == 0.5 and cad["slurm.logstat"].focused_factor == 0.5
    assert cad["slurm.history"].interval == 120 and cad["slurm.nodes"].hidden_factor == 5.0
    slow = ctx("k2", acct_gather_freq_s=45)
    assert {c.group: c for c in SlurmCollector(slow.host).cadences(slow)}["slurm.usage"].floor == 45
    fast = ctx("k2", cfg=host_config("k2", slurm={"poll_jobs": 2.0}))
    assert {c.group: c for c in SlurmCollector(fast.host).cadences(fast)}["slurm.jobs"].interval == 10


def test_first_tick_sections_and_commands() -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    secs = {s.name: s for s in c.sections(INTERVAL_GROUPS, cx)}
    assert set(secs) == {"slurm.squeue", "slurm.sacct"}      # no jobs yet: nothing for sstat/logstat/sinfo
    sq = secs["slurm.squeue"]
    argv = shlex.split(sq.cmd)
    assert argv[:4] == ["squeue", "--me", "-h", "-O"]
    assert argv[4].startswith(f"JobID:{SEP},ArrayJobID:{SEP}") and argv[4].endswith(f"Command:{SEP}")
    assert sq.meta == {"users": [], "extra": [], "sep": SEP, "format": "O"} and sq.timeout == 20
    sa = secs["slurm.sacct"]
    assert "-S now-259200seconds -E now -X -P" in sa.cmd and f"--delimiter={shlex.quote(SEP)}" in sa.cmd
    assert sa.cmd.endswith(",NNodes,NodeList,WorkDir") and sa.meta["since_s"] == 259200
    assert all(s.owner == "slurm" for s in secs.values())


def test_sacct_fields_include_stdout_when_supported() -> None:
    cx = ctx("k2", sacct_fields=frozenset({"StdOut", "StdErr", "JobID"}))
    sa = next(s for s in SlurmCollector(cx.host).sections(frozenset({"slurm.history"}), cx))
    assert sa.cmd.endswith(",NodeList,StdOut,StdErr,WorkDir")


def test_users_and_squeue_args() -> None:
    cx = ctx("k2", cfg=host_config("k2", slurm={"users": ("alice", "bob"), "squeue_args": ("--array",)}))
    sq = SlurmCollector(cx.host).sections(frozenset({"slurm.jobs"}), cx)[0]
    argv = shlex.split(sq.cmd)
    assert argv[1:3] == ["-u", "alice,bob"] and argv[-1] == "--array"
    bad = ctx("k2", cfg=host_config("k2", slurm={"users": ("a;rm -rf /",)}))
    with pytest.raises(ValueError):
        SlurmCollector(bad.host).sections(frozenset({"slurm.jobs"}), bad)


def test_incremental_history_window() -> None:
    c, cx, _ = started()
    tick(c, cx, sacct_rows())
    later = at(cx, 120)
    sa = next(s for s in c.sections(frozenset({"slurm.history"}), later))
    assert sa.meta["since_s"] == 420                       # since last success − 300 s


# ------------------------------------------------------------------ live jobs
def test_live_jobs_and_jobinfo_followup() -> None:
    c, cx, out = started()
    assert out.jobs is not None and len(out.jobs) == 5
    assert "slurm.jobinfo" in out.followups
    sc = c.sections(frozenset({"slurm.jobinfo"}), cx)[0]
    assert sc.name == "slurm.scontrol" and len(sc.meta["ids_raw"]) == 5
    assert shlex.split(sc.cmd)[:2] == ["for", "j"] and "scontrol show job -d -o" in sc.cmd


def _scontrol_loop(host: str, items: dict[str, str]) -> SectionResult:
    """Loop output: `@@job <id>` + the fixture (or an error line) per id."""
    parts = []
    for jid, fixture in items.items():
        parts.append(f"@@job {jid}")
        parts.append(sample(host, fixture).strip() if fixture else "slurm_load_jobs error: Invalid job id specified")
    return result("slurm.scontrol", "\n".join(parts))


def test_jobinfo_enrichment_and_no_refetch() -> None:
    c, cx, _ = started()
    out = tick(c, cx, _scontrol_loop("k2", {"2332861": "scontrol_show_job_running_d", "2334564": "scontrol_show_job_d",
                                            "2332862": "", "2332863": "", "2333760": ""}))
    assert out.jobs is not None
    j = out.jobs["2332861"]
    assert j.alloc.per_node[0].gpu_indices == (2,) and j.alloc.per_node[0].cpu_ids == "64-71"
    assert j.alloc.gpu_type_source == "tres-alloc"         # (1) still wins over scontrol
    assert j.logs[0].source == "scontrol" and j.logs[0].merged
    assert "slurm.jobinfo" not in out.followups            # answered (or left the controller): never re-asked
    assert c.sections(frozenset({"slurm.jobinfo"}), cx) == []


def test_k2_2333760_is_h200_nvl() -> None:
    _, _, out = started()
    assert out.jobs is not None
    j = out.jobs["2333760"]
    assert j.alloc.gpus == 2 and j.alloc.gpu_type == "nvidia_h200_nvl" and j.alloc.gpu_type_source == "tres-req"


def test_lyra_running_job_type_from_scontrol() -> None:
    row = ["31367493", "31367493", "N/A", "train_a01", "alice07", "RUNNING", "R", "None", "31:07", "12:00:00",
           "11:28:53", "1", "4", "16G", "free-gpu", "low", "victor", "2026-09-07T03:23:39", "2026-09-07T03:27:22",
           "2026-09-07T15:27:22", "lyra-gpu-q71-01", "lyra-gpu-q71-01", "cpu=4,mem=16G,node=1,billing=4,gres/gpu=1",
           "gres/gpu:1", "N/A", "0", "/home/victor/vision/logs/%x_%j.out", "(null)",
           "/home/victor/vision", "/home/victor/vision/train.sbatch"]
    cx = ctx("lyra")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, result("slurm.squeue", SEP.join(row) + SEP))
    assert out.jobs is not None and out.jobs["31367493"].alloc.gpu_type is None
    out = tick(c, cx, _scontrol_loop("lyra", {"31367493": "scontrol_show_job_running_d"}))
    assert out.jobs is not None
    j = out.jobs["31367493"]
    assert j.alloc.gpu_type == "a30" and j.alloc.gpu_type_source == "scontrol"
    assert j.alloc.per_node[0].gpu_indices == (0,)


def test_pd_to_r_refetches_jobinfo() -> None:
    c, cx, _ = started()
    tick(c, cx, _scontrol_loop("k2", {"2334564": "scontrol_show_job_d"}))
    edit = ("2334564|2334564|N/A|smoke-test|alice0001|PENDING|PD|Priority|0:00",
            "2334564|2334564|N/A|smoke-test|alice0001|RUNNING|R|None|0:05")
    out = tick(c, at(cx, 15), squeue(edit=edit))
    assert "slurm.jobinfo" in out.followups
    assert "2334564" in c.sections(frozenset({"slurm.jobinfo"}), cx)[0].meta["ids_raw"]


def test_pending_hints_and_reason_text() -> None:
    _, _, out = started()
    assert out.jobs is not None
    pd = out.jobs["2334564"]
    assert pd.hints == ("waiting for priority",) and pd.raw["reason_text"] == "waiting for priority"
    cx = ctx("vega")
    out = tick(SlurmCollector(cx.host), cx, squeue("vega", "squeue_parts_O"))
    assert out.jobs is not None
    assert out.jobs["712655_[32-63%1]"].hints == ("will never run: dependency failed",)
    assert not out.jobs["712655_[32-63%1]"].is_mine


# ------------------------------------------------------------------ vanished
def test_vanished_then_accounting() -> None:
    c, cx, _ = started()
    out = tick(c, at(cx, 15), squeue(drop=("2332862",)))
    assert out.jobs is not None
    v = out.jobs["2332862"]
    assert v.source == "vanished" and v.state is JobState.RUNNING
    assert v.hints == ("ended; waiting for accounting",)
    assert "slurm.vanished" in out.followups
    sec = c.sections(frozenset({"slurm.vanished"}), cx)[0]
    assert sec.name == "slurm.sacct_ids" and sec.meta["ids_raw"] == ["2332862"]
    assert shlex.split(sec.cmd)[:3] == ["sacct", "-j", "2332862"]
    done = sacct_rows(name="slurm.sacct_ids", edit=(
        "2332863|2332863|orb-swp--train--cifar-cosinewarm--s43--a02|alice0001|campus01|gpu-priority|gpu-preempt|"
        "RUNNING|0:0|0:0|None|2026-09-06T11:14:07|2026-09-07T01:50:27|Unknown",
        "2332862|2332862|orb-swp--train--cifar-const--s43--a02|alice0001|campus01|gpu-priority|gpu-preempt|"
        "COMPLETED|0:0|0:0|None|2026-09-06T11:14:07|2026-09-07T01:50:27|2026-09-07T06:02:37"))
    out = tick(c, at(cx, 30), squeue(drop=("2332862",)), done)
    assert out.jobs is not None
    j = out.jobs["2332862"]
    assert j.source == "sacct" and j.state is JobState.COMPLETED
    # the history row inherits what was seen live (jobmeta): log path and GPU type
    assert j.logs[0].path == "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-const--s43--a02_2332862.out"
    assert c.sections(frozenset({"slurm.vanished"}), cx) == []


def test_vanished_without_accounting_becomes_unknown() -> None:
    c, cx, _ = started()
    tick(c, at(cx, 15), squeue(drop=("2332862",)))
    out = tick(c, at(cx, 15 + 11 * 60), squeue(drop=("2332862",)))
    assert out.jobs is not None
    assert out.jobs["2332862"].state is JobState.UNKNOWN and out.jobs["2332862"].source == "vanished"


def test_reappearing_job_is_not_vanished() -> None:
    c, cx, _ = started()
    tick(c, at(cx, 15), squeue(drop=("2332862",)))
    out = tick(c, at(cx, 30), squeue())
    assert out.jobs is not None and out.jobs["2332862"].source == "squeue"
    assert c.sections(frozenset({"slurm.vanished"}), cx) == []


@pytest.mark.parametrize("bad", [
    result("slurm.squeue", "", 1, "squeue: error: Unable to contact slurm controller (connect failure)"),
    result("slurm.squeue", "", 0, "squeue: error: slurm_load_jobs error: Socket timed out on send/recv operation"),
    result("slurm.squeue", "", None, outcome=SectionOutcome.TIMEOUT),
    result("slurm.squeue", "", None, outcome=SectionOutcome.INCOMPLETE),
])
def test_failed_squeue_never_computes_vanished(bad: SectionResult) -> None:
    c, cx, _ = started()
    out = tick(c, at(cx, 15), bad)
    assert out.jobs is not None and len(out.jobs) == 5
    assert all(j.source == "squeue" for j in out.jobs.values())
    assert out.errors and out.errors[0].section == "slurm.squeue"
    if bad.outcome in (SectionOutcome.FAILED, SectionOutcome.STDERR_ONLY):
        assert out.errors[0].summary == "slurmctld unreachable" and out.errors[0].kind is ErrorKind.REMOTE
    assert "slurm.vanished" not in out.followups


def test_partial_failed_output_updates_but_never_drops() -> None:
    c, cx, _ = started()
    partial = squeue(drop=("2332862", "2332863"))
    partial = SectionResult(name=partial.name, outcome=SectionOutcome.FAILED, rc=1, stdout=partial.stdout,
                            stderr="squeue: error: partial", ms=1)
    out = tick(c, at(cx, 15), partial)
    assert out.jobs is not None and {j.source for j in out.jobs.values()} == {"squeue"} and len(out.jobs) == 5


def test_bad_row_keeps_previous_job() -> None:
    c, cx, first = started()
    assert first.jobs is not None
    evil = squeue(edit=("|alice0001|RUNNING|R|None|4:07:00|3-00:00:00|2-19:53:00|1|8|48G|gpu-preempt|gpu-priority|"
                        "campus01|2026-09-06T11:14:06|",
                        "|alice0001|garbled state|R|None|4:07:00|3-00:00:00|2-19:53:00|1|8|48G|gpu-preempt|"
                        "gpu-priority|campus01|2026-09-06T11:14:06|"))
    out = tick(c, at(cx, 15), evil)
    assert out.jobs is not None
    assert out.jobs["2332861"] == first.jobs["2332861"]      # kept as is, stale, never "vanished"
    assert "slurm.vanished" not in out.followups


def test_mostly_bad_output_is_a_parse_error() -> None:
    c, cx, _ = started()
    out = tick(c, at(cx, 15), result("slurm.squeue", "garbage\nmore garbage\n1|x"))
    assert out.errors[0].kind is ErrorKind.PARSE and out.errors[0].section == "slurm.squeue"
    assert out.jobs is not None and len(out.jobs) == 5


def test_empty_squeue_is_a_valid_answer() -> None:
    c, cx, _ = started()
    out = tick(c, at(cx, 15), result("slurm.squeue", ""))
    assert out.jobs is not None and {j.source for j in out.jobs.values()} == {"vanished"}


def test_separator_fallback() -> None:
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    tick(c, cx, result("slurm.squeue", "", 1, "squeue: error: Invalid field specifier: JobID:\x1f"))
    sq = c.sections(frozenset({"slurm.jobs"}), cx)[0]
    assert sq.meta["sep"] == "|" and "JobID:|" in sq.cmd
    out = tick(c, cx, squeue(sep="|"))
    assert out.jobs is not None and len(out.jobs) == 5
    tick(c, cx, result("slurm.squeue", "", 1, "squeue: error: Invalid field specifier"))
    sq = c.sections(frozenset({"slurm.jobs"}), cx)[0]
    assert sq.meta["format"] == "legacy" and "-o" in shlex.split(sq.cmd)


@pytest.mark.parametrize("stderr", [
    "squeue: error: Invalid user: ghost", "squeue: error: Invalid partition name specified",
    "sacct: error: Problem talking to the database: Unknown error",
    "squeue: error: Unable to contact slurm controller (connect failure)",
])
def test_unrelated_errors_never_downgrade_the_format(stderr: str) -> None:
    """Review: a config typo (`users = ["ghost"]`) switched squeue to `|` and then to the lossy legacy -o for good."""
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    for i in range(3):
        tick(c, at(cx, 15 * i), result("slurm.squeue", "", 1, stderr), result("slurm.sacct", "", 1, stderr))
    assert c._sep == {"squeue": SEP, "sacct": SEP, "sinfo": SEP} and not c._squeue_legacy


def test_pending_array_summary_does_not_turn_into_a_ghost() -> None:
    """Review: `500_[1-20%1]` → `500_[2-20%1]` + `500_1` when a task starts; the old summary key is no job that
    ended and no sacct row ever resolves it, so it stayed 'final state unknown' for the whole window."""
    def row(jid: str, task: str, state: str = "PENDING", reason: str = "JobArrayTaskLimit") -> str:
        fields = {"JobID": jid, "ArrayJobID": "500", "ArrayTaskID": task, "Name": "arr", "UserName": "alice0001",
                  "State": state, "StateCompact": state[:2], "Reason": reason, "TimeUsed": "0:00",
                  "TimeLimit": "1:00:00", "TimeLeft": "1:00:00", "NumNodes": "1", "NumCPUs": "1", "MinMemory": "1G",
                  "Partition": "normal", "QOS": "normal", "Account": "a", "SubmitTime": "1788723357",
                  "StartTime": "N/A", "EndTime": "N/A", "NodeList": "", "BatchHost": "n/a", "tres-alloc": "cpu=1",
                  "tres-per-node": "N/A", "tres-per-job": "N/A", "RestartCnt": "0", "STDOUT": "/x/o",
                  "STDERR": "/x/o", "WorkDir": "/x", "Command": "/x/run.sh"}
        from runfleet.backends.slurm.commands import SQUEUE_FIELDS
        return "".join(fields[f] + SEP for f in SQUEUE_FIELDS)

    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    tick(c, cx, result("slurm.squeue", row("500", "1-20%1")))
    out = tick(c, at(cx, 60), result("slurm.squeue", "\n".join([row("500", "2-20%1"),
                                                                  row("501", "1", "RUNNING", "None")])))
    assert out.jobs is not None and set(out.jobs) == {"500_[2-20%1]", "500_1"}
    assert "slurm.vanished" not in out.followups and not c._vanished_at
    # a cancelled array (no task left in squeue) still goes through the vanished lookup
    out = tick(c, at(cx, 120), result("slurm.squeue", ""))
    assert set(c._vanished_at) == {"500_[2-20%1]", "500_1"}


# ------------------------------------------------------------------ history
def test_history_terminal_rows_and_live_wins() -> None:
    c, cx, _ = started()
    out = tick(c, cx, sacct_rows())
    assert out.jobs is not None
    assert out.jobs["2332863"].source == "squeue"            # live always wins for active jobs
    done = out.jobs["2333095"]
    assert done.source == "sacct" and done.state is JobState.COMPLETED
    assert done.alloc.gpu_type == "nvidia_rtx_pro_6000_blackwell_server_edition"
    assert done.logs[0].source == "template" and done.logs[0].path == "/shared/pilab/pilab/orb/slurm-2333095.out"


def test_history_log_templates_and_pruning() -> None:
    cfg = host_config("k2", slurm={"log_templates": ("{workdir}/logs/{name}_{jobid}.out",)})
    cx = ctx("k2", cfg=cfg)
    c = SlurmCollector(cx.host)
    out = tick(c, cx, sacct_rows())
    assert out.jobs is not None
    assert out.jobs["2333095"].logs[0].path == \
        "/shared/pilab/pilab/orb/logs/orb-ab--train--resnet18-bs512--s42--a02_2333095.out"
    out = tick(c, at(cx, 4 * 86400), result("slurm.squeue", ""))
    assert out.jobs is not None and "2333095" not in out.jobs   # ended > sacct_window ago


def test_slurmdbd_down_keeps_history() -> None:
    c, cx, _ = started()
    tick(c, cx, sacct_rows())
    out = tick(c, at(cx, 120), result("slurm.sacct", "", 1, "sacct: error: Problem talking to the database"))
    assert out.errors[0].summary == "slurmdbd unreachable"
    assert out.jobs is not None and "2333095" in out.jobs


# ------------------------------------------------------------------ sstat / gpu_tres
def test_sstat_usage_and_sections() -> None:
    c, cx, _ = started()
    sec = c.sections(frozenset({"slurm.usage"}), cx)[0]
    # most recently started first: 2332861-71 started 01:50:27, 2333760 at 20:55:13 the day before
    assert sec.meta["ids_raw"] == ["2332861", "2332862", "2332863", "2333760"]
    out = tick(c, cx, result("slurm.sstat", with_tot(sample("k2", "sstat_running").strip())))
    assert out.jobs is not None
    g = out.jobs["2332861"].gpu
    assert g is not None and g.util_mean_pct == 98 and g.util_peak_pct == 99 and g.mem_used_mib == 24870
    assert out.caps_patch is None                           # already known True


def test_gpu_tres_learning() -> None:
    c, cx, _ = started(gpu_tres=None)
    out = tick(c, cx, result("slurm.sstat", with_tot(sample("k2", "sstat_running").strip())))
    assert out.caps_patch == {"gpu_tres": True}
    # GPU jobs without the key: False only after two distinct GPU jobs lacked it
    c2, cx2, _ = started(gpu_tres=None)
    no_gpu = "2332861.batch|1|01:00:00|1K|1K|cpu=01:00:00,mem=1K|cpu=01:00:00,mem=1K|cpu=01:00:00,mem=1K"
    assert tick(c2, cx2, result("slurm.sstat", no_gpu)).caps_patch is None
    assert tick(c2, cx2, result("slurm.sstat", no_gpu)).caps_patch is None    # same job twice: still unknown
    other = no_gpu.replace("2332861", "2333760")
    assert tick(c2, cx2, result("slurm.sstat", other)).caps_patch == {"gpu_tres": False}


def test_cpu_only_job_never_sets_gpu_tres_false() -> None:
    cx = ctx("vega", gpu_tres=None)
    c = SlurmCollector(cx.host)
    row = ["9", "9", "N/A", "cpu", "alice0001", "RUNNING", "R", "None", "1:00", "30:00", "29:00", "1", "4", "8G", "dev",
           "normal", "pilab", "2026-09-07T05:42:37", "2026-09-07T05:43:37", "2026-09-07T06:13:37", "m-31-02",
           "m-31-02", "cpu=4,mem=8G,node=1", "N/A", "N/A", "0", "/w/o.out", "(null)", "/w", "/w/run.sh"]
    rows = [SEP.join(row) + SEP, SEP.join([*row[:1], "9", *row[2:]]).replace("9", "8", 1) + SEP]
    tick(c, cx, result("slurm.squeue", "\n".join(rows)))
    for jid in ("9", "8"):
        out = tick(c, cx, result("slurm.sstat", f"{jid}.batch|1|00:01:00|1K|1K|cpu=1,mem=1K|cpu=1,mem=1K|cpu=1"))
        assert out.caps_patch is None


def test_gpu_idle_hint() -> None:
    c, cx, _ = started()
    idle = with_tot(sample("k2", "sstat_running").strip()).replace("gres/gpuutil=98", "gres/gpuutil=0")
    for minute in range(0, 17):
        out = tick(c, at(cx, minute * 60), result("slurm.sstat", idle))
    assert out.jobs is not None and "GPU idle 16m" in out.jobs["2332861"].hints


# ------------------------------------------------------------------ logstat / sinfo
def test_logstat_and_log_idle_hint() -> None:
    c, cx, _ = started()
    sec = c.sections(frozenset({"slurm.logstat"}), cx)[0]
    paths = sec.meta["paths"]
    p = "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"
    assert p in paths and len(paths) == 4   # RUNNING jobs only
    mtime = int((cx.now - timedelta(minutes=47)).timestamp())
    out = tick(c, cx, result("slurm.logstat", f"{p}|663|{mtime}"))
    assert out.jobs is not None
    j = out.jobs["2332861"]
    assert j.logs[0].size == 663 and "log idle 47m" in j.hints


def test_sinfo_partitions_nodes_and_type_from_sinfo() -> None:
    c, cx, first = started()
    assert c.partitions(cx) == ["gpu-preempt"]
    assert "slurm.nodes" in first.followups               # partitions just became known: fetch nodes next tick
    sec = c.sections(frozenset({"slurm.nodes"}), cx)[0]
    assert "slurm.nodes" not in tick(c, cx, squeue()).followups   # planned for this set already
    assert shlex.split(sec.cmd)[:5] == ["sinfo", "-h", "-N", "-p", "gpu-preempt"]
    out = tick(c, cx, result("slurm.sinfo", to_sep(sample("k2", "sinfo_nodes_O").strip(), SEP)))
    assert out.nodes is not None and len(out.nodes) == 30
    assert out.partitions is not None and out.partitions[0].name == "gpu-preempt"
    cfg = ctx("k2", cfg=host_config("k2", slurm={"partitions": ("lab", "bad;part")}))
    assert SlurmCollector(cfg.host).partitions(cfg) == ["lab"]
    assoc = ctx("k2", associations=(("k2", "campus01", "gpu,cpu", "normal"),))
    assert SlurmCollector(assoc.host).partitions(assoc) == ["cpu", "gpu"]
    paid = ctx("k2", gpu_partitions=("labpi", "gpu-preempt"))           # GPU partitions you may use, never used yet
    assert SlurmCollector(paid.host).partitions(paid) == ["gpu-preempt", "labpi"]
    assert SlurmCollector(cfg.host).partitions(dataclasses.replace(cfg, caps=paid.caps)) == ["lab"]   # config wins


def test_new_running_jobs_request_usage_followup() -> None:
    c, cx, first = started()
    assert "slurm.usage" in first.followups               # running jobs just became known: sstat next tick
    assert c.sections(frozenset({"slurm.usage"}), cx)[0].name == "slurm.sstat"
    assert "slurm.usage" not in tick(c, cx, squeue()).followups   # planned for these jobs already
    nosstat = ctx("k2", tools=("squeue", "sacct", "scontrol", "sinfo"))
    assert "slurm.usage" not in tick(SlurmCollector(nosstat.host), nosstat, squeue()).followups


def test_gpu_type_from_sinfo_for_untyped_job() -> None:
    row = ["77", "77", "N/A", "u", "alice0001", "RUNNING", "R", "None", "1:00", "1:00:00", "59:00", "1", "8", "48G",
           "gpu-preempt", "normal", "campus01", "2026-09-07T05:42:37", "2026-09-07T05:43:37", "2026-09-07T06:43:37",
           "n-40-01", "n-40-01", "cpu=8,mem=48G,node=1,gres/gpu=1", "gres/gpu:1", "N/A", "0", "/w/o.out", "(null)",
           "/w", "/w/r.sh"]
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    tick(c, cx, result("slurm.squeue", SEP.join(row) + SEP))
    out = tick(c, cx, result("slurm.sinfo", to_sep(sample("k2", "sinfo_nodes_O").strip(), SEP)))
    assert out.jobs is not None
    j = out.jobs["77"]
    assert j.alloc.gpu_type == "nvidia_a100-sxm4-40gb" and j.alloc.gpu_type_source == "sinfo"


def test_no_results_returns_unchanged() -> None:
    c, cx, _ = started()
    out = c.ingest({}, cx, MonitorView())
    assert out.jobs is None and out.errors == ()


# ------------------------------------------------------------------ details / shell
def test_detail_running_job() -> None:
    c, cx, out = started()
    assert out.jobs is not None
    job = out.jobs["2332861"]
    names = [s.name for s in c.detail_sections(job, cx)]
    assert names == ["slurm.scontrol", "slurm.steps", "slurm.sstat", "slurm.logstat"]
    detail = c.ingest_detail(job, {
        "slurm.scontrol": _scontrol_loop("k2", {"2332861": "scontrol_show_job_running_d"}),
        "slurm.sstat": result("slurm.sstat", with_tot(sample("k2", "sstat_running").strip())),
        "slurm.steps": result("slurm.steps", "", 1, "sacct: error: x"),
    }, cx)
    assert detail.per_node[0].gpu_indices == (2,)
    assert detail.fields["SubmitTime"] == "2026-09-06T11:14:06"
    assert detail.gpu is not None and detail.gpu.util_mean_pct == 98
    assert detail.procs == () and "Procs tab" in detail.procs_note
    assert detail.errors and detail.errors[0].section == "slurm.steps"
    # jobinfo is now fresh: the next detail within 30 s skips scontrol
    assert "slurm.scontrol" not in [s.name for s in c.detail_sections(job, at(cx, 10))]


def test_detail_failed_history_job_explains() -> None:
    cx = ctx("lyra")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, result("slurm.sacct", to_sep(sample("lyra", "sacct_ext").strip(), SEP)))
    assert out.jobs is not None
    job = out.jobs["31327454"]
    secs = [s.name for s in c.detail_sections(job, cx)]
    assert secs[0] == "slurm.steps" and "slurm.scontrol" not in secs   # long gone from the controller
    steps = sample("lyra", "sacct_failed_job_steps").strip()
    # live step fields: JobID,JobName,State,ExitCode,DerivedExitCode,Reason,Elapsed,Start,End,NodeList,NTasks,MaxRSS,
    # AveCPU,TRESUsageInMax,TRESUsageInTot — rebuild the fixture rows (11 columns) in that order
    rows = []
    for line in steps.splitlines():
        f = line.split("|")
        rows.append(SEP.join([*f[:10], "1", f[10], "", "", ""]))
    preempted = c.ingest_detail(job, {"slurm.steps": result("slurm.steps", "\n".join(rows))}, cx)
    assert len(preempted.steps) == 3
    assert any("SIGTERM (128+15)" in e for e in preempted.explanation)


def _inner(cmd: str) -> str:
    """The sh script of `exec sh -c '<script>'`."""
    words = shlex.split(cmd)
    assert words[:3] == ["exec", "sh", "-c"] and len(words) == 4
    return words[3]


def test_shell_command_running_job_attaches_to_its_node() -> None:
    """A running job's shell lands on its compute node inside the job (lyra refuses ssh to nodes), not the login node."""
    c, cx, out = started()
    assert out.jobs is not None
    job = out.jobs["2332861"]
    assert job.state is JobState.RUNNING and job.nodes == ("n-40-01",)
    script = _inner(c.shell_command(job, cx) or "")
    assert script.startswith("cd /shared/pilab/pilab/orb 2>/dev/null; ")
    assert 'srun --jobid=2332861 --overlap --pty -N1 -n1 -w n-40-01 "${SHELL:-sh}" -l' in script
    assert c.shell_command(job, cx, "n-29-99") == c.shell_command(job, cx)      # not one of the job's nodes
    assert c.shell_command(None, cx) is None


def test_shell_command_node_choice_and_old_slurm() -> None:
    c, cx, out = started()
    assert out.jobs is not None
    job = dataclasses.replace(out.jobs["2332861"], nodes=("n-40-01", "n-40-02"), batch_host="n-40-01")
    assert "-w n-40-02 " in _inner(c.shell_command(job, cx, "n-40-02") or "")
    assert "-w n-40-01 " in _inner(c.shell_command(job, cx) or "")
    bad = dataclasses.replace(job, nodes=("n-40-01;rm",), batch_host=None)
    assert c.shell_command(bad, cx) == 'cd /shared/pilab/pilab/orb && exec "$SHELL" -l'
    old = dataclasses.replace(cx, caps=dataclasses.replace(cx.caps, slurm_version="20.02.7"))
    assert "--overlap" not in _inner(c.shell_command(job, old) or "")


def test_shell_command_not_running_stays_on_login_node() -> None:
    c, cx, out = started()
    assert out.jobs is not None
    job = dataclasses.replace(out.jobs["2332861"], state=JobState.PENDING, nodes=(), batch_host=None)
    assert c.shell_command(job, cx) == 'cd /shared/pilab/pilab/orb && exec "$SHELL" -l'
    assert c.shell_command(dataclasses.replace(job, workdir=None), cx) is None


@pytest.mark.parametrize("rc, pauses", [(0, False), (130, False), (1, True)])
def test_shell_command_script_runs_under_sh(tmp_path: Path, rc: int, pauses: bool) -> None:
    """The script really runs in POSIX sh: srun gets the job/node, a failed srun waits for Enter before returning."""
    c, cx, out = started()
    assert out.jobs is not None
    script = _inner(c.shell_command(out.jobs["2332861"], cx) or "").replace("cd /shared/pilab/pilab/orb", f"cd {tmp_path}")
    (tmp_path / "srun").write_text(f'#!/bin/sh\necho "srun $*" > "{tmp_path}/args"\nexit {rc}\n')
    (tmp_path / "srun").chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "SHELL": "/bin/sh"}
    res = subprocess.run(["sh", "-c", script], input="\n", capture_output=True, text=True, env=env, timeout=10)
    assert res.returncode == 0
    assert (tmp_path / "args").read_text().split() == ["srun", "--jobid=2332861", "--overlap", "--pty", "-N1", "-n1",
                                                       "-w", "n-40-01", "/bin/sh", "-l"]
    assert "shell on n-40-01 inside job 2332861" in res.stdout
    assert ("press Enter" in res.stdout) is pauses


def test_sstat_cpu_and_rss_reach_running_jobs_and_steps() -> None:
    """M3 F8: AveCPU/MaxRSS were parsed every 60 s and never shown (Job.cpu_pct/rss_bytes stayed null)."""
    from datetime import timedelta as td

    from runfleet.backends.slurm.parse_sstat import parse_sstat, job_usage
    from runfleet.backends.slurm.collector import _cpu_pct

    rows = parse_sstat("31367725.batch|1|01:04:41|60G|69G|cpu=01:04:41|cpu=01:04:41|cpu=01:04:41\n").items
    usage = job_usage(rows, gpus=0, gpu_type=None, gpu_tres=False, sampled_at=None, interval_s=60)
    assert usage is not None and usage.max_rss_bytes == 69 * 1024 ** 3
    assert _cpu_pct(usage, td(hours=1, minutes=4, seconds=41)) == pytest.approx(100.0)
    assert _cpu_pct(usage, None) is None
    c, cx, _ = started()
    job = next(j for j in c._live.values() if j.state is JobState.RUNNING)
    raw = job.job_id_raw
    text = (f"{raw}.batch|1|00:30:00|1G|2G|cpu=00:30:00|cpu=00:30:00,gres/gpuutil=184|cpu=00:30:00\n"
            f"{raw}.extern|1|00:00:00|0|0|||\n")
    out = tick(c, at(cx, 60), squeue(), result("slurm.sstat", text))
    assert out.jobs is not None
    enriched = out.jobs[job.key.id]
    assert enriched.rss_bytes == 2 * 1024 ** 3 and enriched.cpu_pct is not None and enriched.cpu_pct > 0
    from runfleet.models import JobStep
    step = JobStep(step_id=f"{raw}.batch", name="batch", state=JobState.RUNNING, raw_state="RUNNING", exit_code=None,
                   exit_signal=None, elapsed=None, start=None, end=None, nodelist=None, ntasks=None,
                   max_rss_bytes=None, ave_cpu=None, tres_usage_max={}, tres_usage_tot={})
    (filled,) = c._steps_with_sstat(enriched, (step,))
    assert filled.max_rss_bytes == 2 * 1024 ** 3 and filled.ave_cpu == td(minutes=30)
    # M3 F8 residual: the Steps tab's "GPU (max per task)" of a running step comes from the same sstat row
    assert filled.tres_usage_max.get("gres/gpuutil") == "184" and filled.ntasks == 1


# ------------------------------------------------------------------ review 2026-10 (BACK-08..38)
def _row(jid: str, task: str = "N/A", state: str = "RUNNING", *, array: str | None = None, nodes: str = "n1",
         restarts: str = "0", start: str = "1788723457") -> str:
    from runfleet.backends.slurm.commands import SQUEUE_FIELDS
    fields = {"JobID": jid, "ArrayJobID": array or jid, "ArrayTaskID": task, "Name": "j", "UserName": "alice0001",
              "State": state, "StateCompact": state[:2], "Reason": "None", "TimeUsed": "1:00",
              "TimeLimit": "1:00:00", "TimeLeft": "59:00", "NumNodes": "1", "NumCPUs": "1", "MinMemory": "1G",
              "Partition": "normal", "QOS": "normal", "Account": "a", "SubmitTime": "1788723357",
              "StartTime": start if state == "RUNNING" else "N/A", "EndTime": "N/A",
              "NodeList": nodes if state == "RUNNING" else "", "BatchHost": nodes if state == "RUNNING" else "n/a",
              "tres-alloc": "cpu=1,gres/gpu=1", "tres-per-node": "gres/gpu:1", "tres-per-job": "N/A",
              "RestartCnt": restarts, "STDOUT": "/x/o", "STDERR": "/x/o", "WorkDir": "/x", "Command": "/x/run.sh"}
    return "".join(fields[f] + SEP for f in SQUEUE_FIELDS)


def _squeue_rows(*rows: str) -> SectionResult:
    return result("slurm.squeue", "\n".join(rows))


def test_cancelled_pending_array_summary_leaves_no_ghost() -> None:
    """BACK-10: sacct answers a cancelled `500_[1-2]` with 500_1/500_2; the summary key stayed UNKNOWN for 3 days."""
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    tick(c, cx, _squeue_rows(_row("500", "1-2", "PENDING")))
    tick(c, at(cx, 15), _squeue_rows())
    fields = cmds_sacct_fields()
    rows = []
    for t in ("1", "2"):
        vals = {"JobID": f"500_{t}", "JobIDRaw": f"50{t}", "JobName": "j", "User": "alice0001", "Account": "a",
                "QOS": "normal", "Partition": "normal", "State": "CANCELLED by 1", "ExitCode": "0:0",
                "DerivedExitCode": "0:0", "Reason": "None", "Submit": "1788723357", "Start": "None",
                "End": "1788723557", "Elapsed": "00:00:00", "Timelimit": "01:00:00", "AllocCPUS": "0",
                "ReqMem": "1G", "ReqTRES": "cpu=1", "AllocTRES": "", "NNodes": "1", "NodeList": "None",
                "WorkDir": "/x"}
        rows.append(SEP.join(vals[f] for f in fields))
    out = tick(c, at(cx, 30), _squeue_rows(), result("slurm.sacct_ids", "\n".join(rows)))
    assert out.jobs is not None and "500_[1-2]" not in out.jobs and {"500_1", "500_2"} <= set(out.jobs)
    # and without any accounting it is dropped after the grace period instead of turning UNKNOWN
    c2 = SlurmCollector(cx.host)
    tick(c2, cx, _squeue_rows(_row("600", "1-2", "PENDING", array="600")))
    tick(c2, at(cx, 15), _squeue_rows())
    out = tick(c2, at(cx, 15 + 11 * 60), _squeue_rows())
    assert out.jobs is not None and "600_[1-2]" not in out.jobs


def cmds_sacct_fields() -> list[str]:
    from runfleet.backends.slurm.commands import sacct_fields
    return sacct_fields(frozenset())


def test_scontrol_errors_back_off_and_surface() -> None:
    """BACK-11: `scontrol … 2>&1; true` turned controller errors into stdout; the ids were re-asked every tick."""
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, _squeue_rows(_row("700")))
    assert "slurm.jobinfo" in out.followups
    c.sections(frozenset({"slurm.jobinfo"}), cx)
    err = result("slurm.scontrol", "@@job 700\nslurm_load_jobs error: Unable to contact slurm controller")
    out = tick(c, cx, err)
    assert out.errors and out.errors[0].summary == "slurmctld unreachable" and out.errors[0].kind is ErrorKind.REMOTE
    out = tick(c, at(cx, 15), _squeue_rows(_row("700")))
    assert "slurm.jobinfo" not in out.followups and c.sections(frozenset({"slurm.jobinfo"}), at(cx, 15)) == []
    out = tick(c, at(cx, 31), _squeue_rows(_row("700")))
    assert "slurm.jobinfo" in out.followups               # due again after the 30 s backoff
    assert c.sections(frozenset({"slurm.jobinfo"}), at(cx, 31))[0].meta["ids_raw"] == ["700"]


def test_failed_pd_to_r_refresh_is_retried_after_the_backoff() -> None:
    """A PD→R refresh that scontrol fails keeps its retry even though the pending-time facts are cached (squeue
    alone would never ask again: the job is not new and no longer changes state)."""
    c, cx, _ = started()
    tick(c, cx, _scontrol_loop("k2", {"2332861": "scontrol_show_job_running_d", "2334564": "scontrol_show_job_d",
                                      "2332862": "", "2332863": "", "2333760": ""}))
    edit = ("2334564|2334564|N/A|smoke-test|alice0001|PENDING|PD|Priority|0:00",
            "2334564|2334564|N/A|smoke-test|alice0001|RUNNING|R|None|0:05")
    tick(c, at(cx, 15), squeue(edit=edit))
    err = result("slurm.scontrol", "@@job 2334564\nslurm_load_jobs error: Unable to contact slurm controller")
    tick(c, at(cx, 16), err)
    out = tick(c, at(cx, 30), squeue(edit=edit))
    assert "slurm.jobinfo" not in out.followups           # inside the 30 s backoff
    out = tick(c, at(cx, 47), squeue(edit=edit))
    assert "slurm.jobinfo" in out.followups
    assert c.sections(frozenset({"slurm.jobinfo"}), at(cx, 47))[0].meta["ids_raw"] == ["2334564"]
    tick(c, at(cx, 47), _scontrol_loop("k2", {"2334564": "scontrol_show_job_running_d"}))
    assert "2334564" not in c._want_jobinfo               # answered: never asked again

def test_new_job_usage_followup_respects_the_floor() -> None:
    """BACK-32: jobs starting on consecutive ticks sent the full sstat batch every 15 s."""
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    out = tick(c, cx, _squeue_rows(_row("801")))
    assert "slurm.usage" in out.followups
    c.sections(frozenset({"slurm.usage"}), cx)
    out = tick(c, at(cx, 15), _squeue_rows(_row("801"), _row("802")))
    assert "slurm.usage" not in out.followups             # 15 s after the last sstat: wait for the 30 s floor
    out = tick(c, at(cx, 30), _squeue_rows(_row("801"), _row("802")))
    assert "slurm.usage" in out.followups


def test_sstat_rotates_through_more_jobs_than_slots() -> None:
    """BACK-38: with 25 running jobs the 5 oldest never got a usage sample."""
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    tick(c, cx, _squeue_rows(*[_row(str(900 + i), start=str(1788723457 + i)) for i in range(25)]))
    seen: set[str] = set()
    for k in range(2):
        (sec,) = c.sections(frozenset({"slurm.usage"}), at(cx, 60 * k))
        assert len(sec.meta["ids_raw"]) == 20
        seen.update(sec.meta["ids_raw"])
    assert seen == {str(900 + i) for i in range(25)}


def test_requeued_job_drops_the_previous_attempts_facts() -> None:
    """BACK-08: a requeue within one poll (RUNNING → RUNNING, RestartCnt 0 → 1, new node) kept the old GPU IDX."""
    c, cx, _ = started()
    tick(c, cx, _scontrol_loop("k2", {"2332861": "scontrol_show_job_running_d"}))
    assert c._jobinfo["2332861"].per_node[0].gpu_indices == (2,)
    edit = ("|0|/shared/pilab/pilab/orb/logs/%x_%j.out|/shared/pilab/pilab/orb/logs/%x_%j.out|/shared/pilab/pilab/orb|"
            "/shared/pilab/pilab/orb/jobs/train.sbatch|")
    text = sample("k2", "squeue_me_O").replace("n-40-01|n-40-01", "n-41-01|n-41-01", 1)
    first = text.splitlines()[1]
    text = text.replace(first, first.replace(edit, edit.replace("|0|", "|1|", 1)))
    out = tick(c, at(cx, 15), result("slurm.squeue", to_sep(text, SEP)))
    assert "2332861" not in c._jobinfo and "slurm.jobinfo" in out.followups
    assert out.jobs is not None and all(n.node != "n-40-01" for n in out.jobs["2332861"].alloc.per_node)


def test_vanished_lookup_is_retried_while_accounting_lags() -> None:
    """BACK-09: the sacct -j lookup ran once; a slurmdbd lag left the job UNKNOWN for the whole window."""
    c, cx, _ = started()
    tick(c, at(cx, 15), squeue(drop=("2332862",)))
    assert c.sections(frozenset({"slurm.vanished"}), at(cx, 15))
    out = tick(c, at(cx, 20), squeue(drop=("2332862",)), result("slurm.sacct_ids", ""))
    assert "slurm.vanished" not in out.followups          # next try only after the backoff
    out = tick(c, at(cx, 50), squeue(drop=("2332862",)))
    assert "slurm.vanished" in out.followups
    t = 50
    for _ in range(10):
        c.sections(frozenset({"slurm.vanished"}), at(cx, t))
        t += 700
    assert c.sections(frozenset({"slurm.vanished"}), at(cx, t)) == []   # bounded number of tries


def test_gpu_tres_is_never_downgraded_after_a_positive_sample() -> None:
    """BACK-17: two freshly started jobs without a first GPU sample flipped the cluster to 'no GPU accounting'."""
    c, cx, _ = started(gpu_tres=None)
    good = with_tot(sample("k2", "sstat_running").strip())
    assert tick(c, cx, result("slurm.sstat", good)).caps_patch == {"gpu_tres": True}
    no_gpu = "2332861.batch|1|01:00:00|1K|1K|cpu=01:00:00,mem=1K|cpu=01:00:00,mem=1K|cpu=01:00:00,mem=1K"
    for jid in ("2332861", "2333760"):
        assert tick(c, cx, result("slurm.sstat", no_gpu.replace("2332861", jid))).caps_patch is None
    assert c._gpu_tres is True
