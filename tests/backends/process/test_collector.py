"""[WP2] ProcessCollector: sticky tasks across ticks, fds/logs, ended tasks + exit scan, annotations, details."""

from __future__ import annotations

import dataclasses
import shlex
from datetime import timedelta
from pathlib import Path

from runfleet.backends.base import MonitorView
from runfleet.backends.process.collector import ProcessCollector, parse_exitscan
from runfleet.models import JobState, Proc
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import NOW, at, ctx, host_config, result
from tests.backends.process.test_tasks import EPOCH, monitor_view, own_ps

ROOT = "p642229-" + str(EPOCH - 67514)


def ps(shift: int = 0, drop: tuple[int, ...] = ()) -> object:
    text = own_ps("lynx", EPOCH + shift)
    lines = [ln for ln in text.splitlines() if not any(ln.startswith(f"{p} ") for p in drop)]
    return result("process.ps", "\n".join(lines))


def started() -> tuple[ProcessCollector, object, MonitorView, object]:
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    mv = monitor_view("lynx")
    out = c.ingest({"process.ps": ps()}, cx, mv)
    return c, cx, mv, out


def test_cadences_and_first_sections() -> None:
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    cad = {x.group: x for x in c.cadences(cx)}
    assert cad["process.ps"].interval == 10 and cad["process.ps"].focused_factor == 0.5
    assert cad["process.logstat"].interval == 60 and cad["process.exitscan"].mode == "event"
    (sec,) = c.sections(frozenset({"process.ps", "process.logstat", "process.exitscan"}), cx)
    assert sec.name == "process.ps" and sec.cmd.startswith("date +%s; ps -ww -u \"$(id -un)\" -o pid=,ppid=,pgid=")
    out = c.ingest({}, cx, MonitorView())
    assert out.jobs is None and out.gpus is None


def test_task_annotation_and_fds_followup() -> None:
    c, cx, mv, out = started()
    assert out.jobs is not None and list(out.jobs) == [ROOT]
    job = out.jobs[ROOT]
    assert job.name == "orb.launch ×2 (orb.train)" and job.key.label == "p642229"
    assert out.gpus is not None
    assert {p.pid: p.job for g in out.gpus for p in g.processes} == {786345: job.key, 642234: job.key}
    assert out.procs is not None and {p.pid for p in out.procs} == {642229, 642234, 786345}
    assert all(p.task == job.key for p in out.procs)
    assert "process.ps" in out.followups                  # fetch cwd/logs of the new task at the next tick
    secs = {s.name: s for s in c.sections(frozenset({"process.ps"}), cx)}
    assert secs["process.fds"].meta["pids"] == [642229, 642234, 786345]
    assert "readlink /proc/$p/fd/1" in secs["process.fds"].cmd


def test_fds_logs_and_logstat() -> None:
    c, cx, mv, _ = started()
    fds = ("642229|/data/home/alice07/projects/orb|/data/home/alice07/projects/orb/logs/l.out|"
           "/data/home/alice07/projects/orb/logs/l.out\n642234|/x|pipe:[1]|pipe:[1]\n786345|/x|/dev/pts/1|/dev/pts/1")
    out = c.ingest({"process.fds": result("process.fds", fds), "process.ps": ps(10)}, at(cx, 10), mv)
    assert out.jobs is not None
    job = out.jobs[ROOT]
    assert job.workdir == "/data/home/alice07/projects/orb"
    assert [(r.stream, r.path, r.merged, r.source) for r in job.logs] == [
        ("stdout", "/data/home/alice07/projects/orb/logs/l.out", True, "procfd"),
        ("stderr", "/data/home/alice07/projects/orb/logs/l.out", True, "procfd")]
    assert "process.fds" not in {s.name for s in c.sections(frozenset({"process.ps"}), cx)}
    (ls,) = c.sections(frozenset({"process.logstat"}), cx)
    assert ls.meta["paths"] == ["/data/home/alice07/projects/orb/logs/l.out"]
    mtime = int((cx.now - timedelta(minutes=45)).timestamp())
    out = c.ingest({"process.logstat": result("process.logstat",
                                              f"/data/home/alice07/projects/orb/logs/l.out|5000|{mtime}")},
                   at(cx, 12), mv)
    assert out.jobs is not None
    assert out.jobs[ROOT].logs[0].size == 5000 and "log idle 45m" in out.jobs[ROOT].hints
    assert c.shell_command(out.jobs[ROOT], cx) == ("cd /data/home/alice07/projects/orb 2>/dev/null || "
                                                   "{ echo 'runfleet: cannot cd to /data/home/alice07/projects/orb, "
                                                   "opening home'; cd; }; exec \"$SHELL\" -l")


def test_task_shell_falls_back_to_home_when_the_workdir_is_gone(tmp_path: Path) -> None:
    """UI-41: `cd X && exec $SHELL` exited at once for a deleted workdir; now it opens a labelled home shell."""
    import os
    import subprocess
    c, cx, mv, _ = started()
    jobs = c.ingest({"process.ps": ps(10)}, at(cx, 10), mv).jobs
    assert jobs is not None
    job = dataclasses.replace(jobs[ROOT], workdir=str(tmp_path / "gone"))
    shell = tmp_path / "fake-shell"
    shell.write_text('#!/bin/sh\npwd\n')
    shell.chmod(0o755)
    cmd = c.shell_command(job, cx)
    assert cmd is not None
    env = {**os.environ, "SHELL": str(shell), "HOME": str(tmp_path)}
    out = subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, timeout=10, check=True).stdout
    assert out.splitlines() == [f"runfleet: cannot cd to {tmp_path / 'gone'}, opening home", str(tmp_path)]


def test_fds_section_stats_its_logs() -> None:
    """M3 F9 residual: `snapshot` (2 rounds) learns log paths from process.fds only, so fds stats them itself."""
    c, cx, mv, _ = started()
    mtime = int((cx.now - timedelta(minutes=3)).timestamp())
    fds = ("642229|/w|/w/a b.out|/w/a b.out\n"
           f"@@stat /w/a b.out|3891|{mtime}\n642234|/x|pipe:[1]|pipe:[1]\n786345|/x|/dev/pts/1|/dev/pts/1")
    out = c.ingest({"process.fds": result("process.fds", fds), "process.ps": ps(10)}, at(cx, 10), mv)
    assert out.jobs is not None
    log = out.jobs[ROOT].logs[0]
    assert log.path == "/w/a b.out" and log.size == 3891 and log.mtime is not None
    assert int(log.mtime.timestamp()) == mtime
    secs = {s.name: s for s in c.sections(frozenset({"process.ps"}), cx)}
    assert "process.fds" not in secs


def test_failed_ps_never_ends_a_task() -> None:
    c, cx, mv, _ = started()
    for bad in (result("process.ps", "", 0, "ps: weird"), result("process.ps", "", None,
                                                                  outcome=SectionOutcome.TIMEOUT),
                result("process.ps", "garbage\nmore\nworse")):
        out = c.ingest({"process.ps": bad}, at(cx, 10), mv)
        assert out.jobs is not None and out.jobs[ROOT].state is JobState.RUNNING
        assert out.errors


def test_ticks_without_ps_reannotate() -> None:
    c, cx, mv, _ = started()
    out = c.ingest({}, at(cx, 5), mv)
    assert out.gpus is not None and all(p.job is not None for g in out.gpus for p in g.processes)
    assert out.jobs is not None and ROOT in out.jobs


def _with_logs(c: ProcessCollector, cx: object, mv: MonitorView) -> None:
    fds = "642229|/w|/w/run.log|/w/run.err"
    c.ingest({"process.fds": result("process.fds", fds), "process.ps": ps(10)}, at(cx, 10), mv)


def test_ended_task_exit_scan_failed() -> None:
    c, cx, mv, _ = started()
    _with_logs(c, cx, mv)
    out = c.ingest({"process.ps": ps(20, drop=(642229, 642234, 786345))}, at(cx, 20), MonitorView())
    assert out.jobs is not None
    ended = out.jobs[ROOT]
    assert ended.state is JobState.EXITED and ended.end_time == at(cx, 20).now and ended.gpu is None
    assert "process.exitscan" in out.followups
    (sec,) = c.sections(frozenset({"process.exitscan"}), cx)
    assert "tail -q -c 8192 -- /w/run.log /w/run.err" in sec.cmd and sec.meta["tasks"] == [(ROOT, ["/w/run.log",
                                                                                                   "/w/run.err"])]
    scan = f"@@rftask {ROOT}\nepoch 3\nTraceback (most recent call last):\n  File t.py\nRuntimeError: boom\n"
    out = c.ingest({"process.exitscan": result("process.exitscan", scan)}, at(cx, 25), MonitorView())
    assert out.jobs is not None
    failed = out.jobs[ROOT]
    assert failed.state is JobState.FAILED and failed.raw_state.startswith("exited (log: Traceback")
    assert "process.exitscan" not in out.followups
    # the ended task stays recent_window (24 h), then disappears
    out = c.ingest({"process.ps": ps(30, drop=(642229, 642234, 786345))}, at(cx, 86400 + 60), MonitorView())
    assert out.jobs is not None and ROOT not in out.jobs


def test_ended_task_clean_log_and_no_log() -> None:
    c, cx, mv, _ = started()
    _with_logs(c, cx, mv)
    c.ingest({"process.ps": ps(20, drop=(642229,))}, at(cx, 20), MonitorView())
    out = c.ingest({"process.exitscan": result("process.exitscan", f"@@rftask {ROOT}\nall done\n")}, at(cx, 25),
                   MonitorView())
    assert out.jobs is not None and out.jobs[ROOT].state is JobState.EXITED
    c2, cx2, mv2, _ = started()
    out = c2.ingest({"process.ps": ps(20, drop=(642229,))}, at(cx2, 20), MonitorView())
    assert out.jobs is not None and out.jobs[ROOT].state is JobState.EXITED
    assert out.jobs[ROOT].hints == ("ended, exit status unknown",) and "process.exitscan" not in out.followups


def test_children_of_ended_root_may_start_new_tasks() -> None:
    c, cx, mv, _ = started()
    out = c.ingest({"process.ps": ps(20, drop=(642229,))}, at(cx, 20), mv)
    assert out.jobs is not None
    active = [j for j in out.jobs.values() if j.state is JobState.RUNNING]
    # the orphaned GPU children (re-parented, still holding GPUs) are rediscovered as their own tasks
    assert {j.key.label for j in active} == {"p642234", "p786345"}


def test_other_users_gpu_processes() -> None:
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    mv = monitor_view("lynx")
    alice = Proc(pid=900, ppid=1, pgid=None, user="alice", etime_s=100, cpu_pct=50.0, mem_pct=1.0, rss_bytes=0,
                 stat="R", args="python -m their.job")
    g0 = mv.gpus[0]
    gpus = (dataclasses.replace(g0, processes=(*g0.processes, dataclasses.replace(
        g0.processes[0], pid=900, user="alice", used_mem_mib=100.0))), *mv.gpus[1:])
    mv = MonitorView(gpus=gpus, gpu_procs=(*mv.gpu_procs, alice), gpus_at=NOW)
    out = c.ingest({"process.ps": ps()}, cx, mv)
    assert out.jobs is not None
    theirs = [j for j in out.jobs.values() if not j.is_mine]
    assert len(theirs) == 1 and theirs[0].user == "alice" and theirs[0].name == "their.job"
    first_key = theirs[0].key
    # one second later etime grew by 2 (sampling jitter): the id stays the same
    mv2 = MonitorView(gpus=gpus, gpu_procs=(*mv.gpu_procs[:-1], dataclasses.replace(alice, etime_s=102)),
                      gpus_at=NOW + timedelta(seconds=1))
    out2 = c.ingest({}, at(cx, 1), mv2)
    assert out2.jobs is not None and first_key.id in out2.jobs
    assert out2.gpus is not None and any(p.job == first_key for g in out2.gpus for p in g.processes)


def test_stale_gpu_sample_is_not_used_for_discovery() -> None:
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    mv = dataclasses.replace(monitor_view("lynx"), gpus_at=NOW - timedelta(seconds=60))
    out = c.ingest({"process.ps": ps()}, cx, mv)
    # discovery still finds the task through CPU use (children at >100 % CPU, running > 60 s)
    assert out.jobs is not None and ROOT in out.jobs


def test_detail_sections_and_ingest() -> None:
    c, cx, mv, out = started()
    assert out.jobs is not None
    job = out.jobs[ROOT]
    (sec,) = c.detail_sections(job, cx)
    assert sec.name == "process.detail" and sec.meta == {"pids": [642229, 642234, 786345], "root": 642229}
    assert "/proc/642229/environ" in sec.cmd and "CUDA_VISIBLE_DEVICES" in sec.cmd
    assert shlex.split(sec.cmd)                         # valid sh word list
    stdout = (" 642229 1 642229 67600 0.0 0.0 1000 SN python -m orb.launch\n"
              " 642234 642229 642229 67600 121 1.9 2000 RNl python -m orb.train\n@@env\nCUDA_VISIBLE_DEVICES=1\n"
              "SECRET=nope\nWANDB_PROJECT=orb")
    d = c.ingest_detail(job, {"process.detail": result("process.detail", stdout)}, cx)
    assert [p.pid for p in d.procs] == [642229, 642234] and all(p.task == job.key for p in d.procs)
    assert d.env == {"CUDA_VISIBLE_DEVICES": "1", "WANDB_PROJECT": "orb"}
    assert d.fields["root pid"] == "642229" and d.steps == () and d.per_node == ()


def test_invalid_ignore_regex_is_skipped() -> None:
    cfg = host_config("lynx", workstation={"ignore": ("([", "^python -m orb.launch")})
    cx = ctx("lynx", cfg=cfg)
    c = ProcessCollector(cx.host)
    out = c.ingest({"process.ps": ps()}, cx, monitor_view("lynx"))
    # the launcher is now noise: the GPU children become two separate tasks
    assert out.jobs is not None and {j.key.label for j in out.jobs.values()} == {"p642234", "p786345"}


def test_parse_exitscan() -> None:
    assert parse_exitscan("@@rftask a\nx\ny\n@@rftask b\n") == {"a": "x\ny", "b": ""}


def test_exitscan_only_resolves_the_planned_batch() -> None:
    """Review BACK-14: one scan result cleared every pending task, also those beyond the batch of 16 and those that
    ended in the same ingest, so their logs were never read."""
    cx = ctx("lynx")
    c = ProcessCollector(cx.host)
    c._exitscan = {f"t{i}": [f"/w/{i}.log"] for i in range(17)}
    (sec,) = c.sections(frozenset({"process.exitscan"}), cx)
    assert len(sec.meta["tasks"]) == 16
    c._exitscan["late"] = ["/w/late.log"]                 # ended after the scan was planned
    out = c.ingest({"process.exitscan": result("process.exitscan", "@@rftask t0\nall done\n")}, cx, MonitorView())
    assert set(c._exitscan) == {"t16", "late"} and "process.exitscan" in out.followups
    (sec,) = c.sections(frozenset({"process.exitscan"}), cx)
    assert [t for t, _ in sec.meta["tasks"]] == ["t16", "late"]


def test_fds_rows_with_pipes_in_paths_and_malformed_rows() -> None:
    """Review BACK-23: `|` in a cwd or log path broke the row and its pid was re-read on every ps tick."""
    from runfleet.backends.process.parse_ps import parse_fds
    assert parse_fds("7\x1f/work/a|b\x1f/work/a|b/o.log\x1fpipe:[1]")[7].cwd == "/work/a|b"
    assert parse_fds("7|/w|/w/o.log|/w/o.log")[7].stdout == "/w/o.log"            # the demo's legacy rows
    c, cx, mv, _ = started()
    c.sections(frozenset({"process.ps"}), cx)
    c.ingest({"process.fds": result("process.fds", "642229\x1f/w\x1f/w/a\nb.log\x1f/w/a\nb.log"),
              "process.ps": ps(10)}, at(cx, 10), mv)
    assert 642229 not in c._need_fds                      # malformed row: not retried on every tick


def test_root_fds_are_refreshed() -> None:
    """Review BACK-24: a root's fds were read once per pid; a later `exec >train.log` was never seen."""
    c, cx, mv, _ = started()
    _with_logs(c, cx, mv)
    assert 642229 not in c._need_fds
    c.ingest({"process.ps": ps(10 + 6 * 60)}, at(cx, 10 + 6 * 60), mv)
    secs = {s.name: s for s in c.sections(frozenset({"process.ps"}), cx)}
    assert 642229 in secs["process.fds"].meta["pids"]
