"""[WP2] Task discovery/grouping on the real workstation fixtures (§10.2) + synthetic sticky/reuse/exit-scan cases.

The `my_procs` / `gpu_pids_ps` fixtures have headers and fewer columns than RunFleet's `ps -o …=` commands; the tiny
adapters below rewrite them into RunFleet's format.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.common.nvidia import attach_processes, parse_apps, parse_gpu_query
from runfleet.backends.process.parse_ps import APPS_COLUMNS, OWN_COLUMNS, parse_ps, parse_ps_rows
from runfleet.backends.process.tasks import (NoiseFilter, classify_exit, group_tasks, label)
from runfleet.models import JobState
from tests.backends.factories import NOW
from tests.conftest import sample

EPOCH = int(NOW.timestamp())
NINE = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
        "power.limit")


def own_ps(host: str, epoch: int = EPOCH) -> str:
    """my_procs (`PID PPID ELAPSED %CPU %MEM STAT COMMAND`) → `date +%s` + `pid ppid pgid etimes pcpu pmem rss stat
    args` (pgid unknown → pid, rss unknown → 0)."""
    header, *rows = sample(host, "my_procs").splitlines()
    assert header.split() == ["PID", "PPID", "ELAPSED", "%CPU", "%MEM", "STAT", "COMMAND"]
    out = [str(epoch)]
    for row in rows:
        pid, ppid, et, cpu, mem, stat, args = row.split(None, 6)
        etime = int(et) + (epoch - EPOCH)                 # later samples: processes aged as much as the clock
        out.append(f"{pid} {ppid} {pid} {etime} {cpu} {mem} 0 {stat} {args}")
    return "\n".join(out)


def gpu_ps_rows(host: str) -> str:
    """gpu_pids_ps (`PID PPID USER ELAPSED %CPU %MEM COMMAND`) → `pid ppid user etimes pcpu pmem rss stat args`."""
    header, *rows = sample(host, "gpu_pids_ps").splitlines()
    assert header.split() == ["PID", "PPID", "USER", "ELAPSED", "%CPU", "%MEM", "COMMAND"]
    out = []
    for row in rows:
        pid, ppid, user, et, cpu, mem, args = row.split(None, 6)
        out.append(f"{pid} {ppid} {user} {et} {cpu} {mem} 0 Rl {args}")
    return "\n".join(out)


def monitor_view(host: str) -> MonitorView:
    gpus = parse_gpu_query(sample(host, "gpu_query"), NINE, host=host, sampled_at=NOW).items
    apps, procs = parse_apps(sample(host, "compute_apps").rstrip("\n") + "\n@@ps\n" + gpu_ps_rows(host))
    return MonitorView(gpus=attach_processes(gpus, apps, procs.items), gpu_procs=tuple(procs.items), gpus_at=NOW)


def group(host: str, prev: dict[str, object] | None = None, text: str | None = None, mv: MonitorView | None = None):
    epoch, pr = parse_ps(text if text is not None else own_ps(host), user="alice07")
    assert epoch is not None and not pr.bad_lines
    view = mv or monitor_view(host)
    own = {p.pid for p in pr.items}
    gpu_pids = frozenset(p.pid for g in view.gpus for p in g.processes if p.pid in own)
    return pr.items, group_tasks(pr.items, epoch, gpu_pids=gpu_pids, prev=prev or {}, host=host,
                                 noise=NoiseFilter(), min_age=60, min_cpu=5.0)


def test_lynx_one_task() -> None:
    procs, g = group("lynx")
    (rec,) = g.active.values()
    assert rec.key.label == "p642229"
    assert rec.key.id == f"p642229-{EPOCH - 67514}"
    assert set(rec.members) == {642229, 642234, 786345} and rec.members[0] == 642229
    assert g.new == (rec,) and g.ended == ()


def test_lynx_job_fields() -> None:
    from runfleet.backends.process.tasks import task_job
    procs, g = group("lynx")
    (rec,) = g.active.values()
    view = monitor_view("lynx")
    job = task_job(rec, {p.pid: p for p in procs}, EPOCH, user="alice07", gpus=view.gpus, gpus_at=NOW,
                   gpu_pids=frozenset({642234, 786345}), fds={}, stats={}, observed_at=NOW)
    assert job.name == "orb.launch ×2 (orb.train)"
    assert job.raw["gpu_indices"] == "0,1" and job.alloc.gpus == 2
    # M3 F6: the held GPUs are the task's one-node allocation (the table read per_node and said "cpu only")
    assert [(n.node, n.gpu_indices) for n in job.alloc.per_node] == [("lynx", (0, 1))]
    assert job.alloc.gpu_type == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    assert job.alloc.gpu_type_source == "nvidia-smi"
    assert job.state is JobState.RUNNING and job.elapsed == timedelta(seconds=67514)
    assert job.pids[0] == 642229 and job.cpu_pct == pytest.approx(235.0)
    assert job.gpu is not None and job.gpu.util_mean_pct == pytest.approx(98.5) and job.gpu.util_peak_pct == 100
    assert job.gpu.mem_used_mib == 2 * 15234 and job.gpu.mem_total_mib == 2 * 97887
    assert job.key.label == "p642229" and job.is_mine and job.source == "ps"


@pytest.mark.parametrize("pid", [639192, 639193, 639209, 641936, 641995, 801325, 801456, 801457])
def test_lynx_noise_is_not_a_task(pid: int) -> None:
    # systemd --user, (sd-pam), sshd-session, -zsh, nvitop, `bash -s` (our probe), ps, head
    _, g = group("lynx")
    assert all(pid not in r.members for r in g.active.values())


def test_labstation_one_task_without_xvfb_and_vscode() -> None:
    procs, g = group("labstation")
    (rec,) = g.active.values()
    assert rec.key.label == "p363942" and set(rec.members) == {363942, 363948}
    from runfleet.backends.process.tasks import task_job
    view = monitor_view("labstation")
    job = task_job(rec, {p.pid: p for p in procs}, EPOCH, user="alice07", gpus=view.gpus, gpus_at=NOW,
                   gpu_pids=frozenset({363948}), fds={}, stats={}, observed_at=NOW)
    assert job.name == "orb.launch (orb.train)" and job.raw["gpu_indices"] == "3"
    xvfb = [p for p in procs if p.args.startswith("Xvfb")]
    assert len(xvfb) == 17 and not any(p.pid in rec.members for p in xvfb)


def test_sticky_without_gpu_or_cpu() -> None:
    procs, g = group("lynx")
    # next sample: children gone, the launcher idles at 0 % between manifest items, no GPU sample at all
    text = own_ps("lynx", EPOCH + 10)
    lines = [ln for ln in text.splitlines() if not ln.startswith(("642234 ", "786345 "))]
    _, k2 = group("lynx", prev=g.active, text="\n".join(lines), mv=MonitorView())
    (rec,) = k2.active.values()
    assert rec.key == next(iter(g.active.values())).key and rec.members == (642229,) and k2.ended == ()


def test_reused_root_pid_is_a_new_task() -> None:
    _, g = group("lynx")
    old = next(iter(g.active.values()))
    text = own_ps("lynx", EPOCH + 20).replace("642229 1 642229 67534", "642229 1 642229 30")
    _, k2 = group("lynx", prev=g.active, text=text)
    assert k2.ended == (old,)
    (new,) = k2.active.values()
    assert new.key != old.key and new.key.label == "p642229" and new.root_start == EPOCH + 20 - 30


def test_climb_rules() -> None:
    # tmux → zsh → python train.py ⇒ root python (shells are noise); bash run.sh is a runner, not noise
    text = "\n".join([str(EPOCH),
                      "100 1 100 9000 0.1 0 0 Ss tmux new -s x",
                      "101 100 101 9000 0 0 0 Ss -zsh",
                      "102 101 102 9000 99 1 0 R python train.py --lr 1",
                      "200 101 200 9000 0 0 0 S bash run.sh",
                      "201 200 200 9000 50 1 0 R python -m pkg.worker",
                      "300 1 300 10 99 0 0 R python young.py",
                      "400 1 400 9000 1.0 0 0 S python idle.py"])
    _, g = group("x", text=text, mv=MonitorView())
    roots = {r.root_pid for r in g.active.values()}
    assert roots == {102, 200}                            # young.py too young, idle.py below min cpu
    assert {r.root_pid: set(r.members) for r in g.active.values()}[200] == {200, 201}


def test_labels() -> None:
    assert label("python -m orb.launch --manifest m.txt") == "orb.launch"
    assert label("/x/.venv/bin/python /x/train.py --a 1") == "train.py"
    assert label("torchrun --nproc 4 train.py") == "torchrun train.py"
    assert label("accelerate launch -m pkg.main") == "accelerate pkg.main"
    assert label("bash run.sh") == "run.sh"
    assert label("/usr/bin/Xvfb :99") == "Xvfb"
    assert label("") == "?"


def test_noise_filter() -> None:
    noise = NoiseFilter(["^Xvfb"])
    for args in ("-zsh", "zsh", "/usr/bin/bash", "sshd: u@pts/3", "sshd-session: u@pts/0", "/usr/lib/systemd/systemd "
                 "--user", "(sd-pam)", "tmux new", "/home/u/.vscode-server/code-x --cli", "htop", "watch -n1 x",
                 "/u/.local/share/uv/tools/nvitop/bin/python /u/.local/bin/nvitop", "python3 /u/bin/gpustat --watch",
                 "python3 -m gpustat -i", "Xvfb :99"):
        assert noise(args), args
    for args in ("bash run.sh", "python train.py", "python -m orb.launch"):
        assert not noise(args), args


def test_exit_scan() -> None:
    tb = "epoch 1\nTraceback (most recent call last):\n  File x\ntorch.OutOfMemoryError: CUDA out of memory."
    assert classify_exit(tb) == (JobState.FAILED, "exited (log: Traceback (most recent call last):)",
                                 "log: Traceback (most recent call last):")
    st, raw, _ = classify_exit("ok\nKilled\n")
    assert st is JobState.FAILED and raw == "exited (log: Killed)"
    assert classify_exit("ValueError: bad\n")[0] is JobState.FAILED
    assert classify_exit("epoch 300/300 done\n") == (JobState.EXITED, "exited", "ended, exit status unknown")
    assert classify_exit(None)[0] is JobState.EXITED
    old_error = "Traceback (most recent call last):\n" + "fine\n" * 60
    assert classify_exit(old_error)[0] is JobState.EXITED   # only the last 50 lines count


def test_parse_ps_rows_columns() -> None:
    epoch, pr = parse_ps("1788763357\n  10   1  10  5 1.5 0.1 2048 Sl python a b  c\nbad", user="me")
    assert epoch == 1788763357
    (p,) = pr.items
    assert p.args == "python a b  c" and p.rss_bytes == 2048 * 1024 and p.user == "me" and p.pgid == 10
    assert pr.bad_lines[0][1] == "bad"
    apps = parse_ps_rows([" 5 1 alice 9 0 0 1 S x"], APPS_COLUMNS)
    assert apps.items[0].user == "alice" and apps.items[0].pgid is None
    assert len(OWN_COLUMNS) == 9


def test_non_finite_numbers_never_reach_models() -> None:
    """CORE-44: `nan`/`inf` would end up in `snapshot --json` (invalid JSON); parse_float drops them, ps keeps the row."""
    from runfleet.util.slurmfmt import parse_float
    assert parse_float("nan") is None and parse_float("inf") is None and parse_float("-Infinity") is None
    assert parse_float("1.5") == 1.5
    (p,) = parse_ps_rows([" 5 1 alice 9 nan inf 1 S x"], APPS_COLUMNS).items
    assert p.cpu_pct == 0.0 and p.mem_pct == 0.0


def test_truncated_ps_user_is_still_me() -> None:
    """Review: procps prints `alice000+` for long names in a default-width column; own GPU processes then showed up
    a second time as someone else's read-only job. The command asks for `user:64=`; truncated rows still match."""
    from datetime import UTC, datetime

    from runfleet.backends.common.nvidia import APPS_PAYLOAD
    from runfleet.backends.process.tasks import other_user_jobs
    from runfleet.models import Proc

    assert "user:64=" in APPS_PAYLOAD
    procs = [Proc(pid=4242, ppid=1, pgid=None, user="alice000+", etime_s=60, cpu_pct=1.0, mem_pct=0.1,
                  rss_bytes=0, stat="R", args="python train.py"),
             Proc(pid=4343, ppid=1, pgid=None, user="yara", etime_s=60, cpu_pct=1.0, mem_pct=0.1,
                  rss_bytes=0, stat="R", args="python other.py")]
    out = other_user_jobs(procs, [], 1788733357, host="lynx", me="alice0001", observed_at=datetime.now(UTC))
    assert [j.user for j in out.values()] == ["yara"]


def test_zombie_root_ends_the_task() -> None:
    """Review BACK-16: a root in stat Z (its parent does not reap it) stayed RUNNING forever."""
    _, g = group("lynx")
    old = next(iter(g.active.values()))
    text = own_ps("lynx", EPOCH + 10)
    lines = [ln for ln in text.splitlines() if not ln.startswith(("642234 ", "786345 "))]
    lines = [" ".join(ln.split(None, 8)[:7] + ["Z", ln.split(None, 8)[8]]) if ln.startswith("642229 ") else ln
             for ln in lines]
    _, k2 = group("lynx", prev=g.active, text="\n".join(lines), mv=MonitorView())
    assert k2.ended == (old,) and not k2.active


def test_unknown_gpu_memory_is_not_zero() -> None:
    """Review BACK-21: `[N/A]` per-process memory (None) was summed as a measured 0 MiB."""
    from runfleet.backends.process.tasks import gpu_usage
    view = monitor_view("lynx")
    unknown = tuple(dataclasses.replace(g, processes=tuple(dataclasses.replace(p, used_mem_mib=None)
                                                            for p in g.processes)) for g in view.gpus)
    _, usage = gpu_usage([642234, 786345], unknown, NOW)
    assert usage is not None and usage.mem_used_mib is None
    partial = (unknown[0], *view.gpus[1:])
    _, usage = gpu_usage([642234, 786345], partial, NOW)
    assert usage is not None and usage.mem_used_mib == 15234 and "lower bound" in usage.note
