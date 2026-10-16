"""[WP2] Attach end to end through real shells (§7.16.3/§7.16.4): the planned login probe is rendered by WP1's framing
and run with `<shell> -s`; fake `ssh`/`srun` run the remote command locally, so the node script really is decoded from
argv and executed by `sh`. Stub `nvidia-smi`, `ps`, `scontrol`, `hostname`, `pgrep`, `squeue`, a fake
/proc/self/cgroup and a fake cgroup v2 tree stand in for the compute node. Nothing may be left in TMPDIR."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from runfleet.backends.slurm import attach
from runfleet.backends.slurm import parse_attach as pa
from runfleet.models import JobKey
from runfleet.probe import framing
from tests.backends.factories import ctx
from tests.backends.slurm.test_attach_parse import HEALTHY, STUCK
from tests.backends.slurm.test_attach import collector_with, make_job
from tests.integration.harness import MockEnv, run_script, shells

pytestmark = [pytest.mark.posix,                  # real shells and cgroup files
              pytest.mark.skipif(not Path("/sys/fs/cgroup").is_dir(), reason="needs Linux /sys/fs/cgroup")]

UUID0 = "GPU-11111111-2222-3333-4444-555555555555"
UUID1 = "GPU-66666666-7777-8888-9999-000000000000"

STUBS = {
    # ssh: log argv, drop the options up to "--", then run the remote command like sshd does (`$SHELL -c`)
    "ssh": '''printf 'ssh %s\\n' "$*" >> "$RF_LOG"
[ -n "$RF_DENY" ] && { echo "me@n1: Permission denied (publickey,hostbased)." >&2; exit 255; }
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; break; }; shift; done
shift
exec sh -c "$1"''',
    # srun: log argv, drop options (-w takes a value), run the command with the step's GPUs in the environment
    "srun": '''printf 'srun %s\\n' "$*" >> "$RF_LOG"
while [ $# -gt 0 ]; do case $1 in -w) shift 2 ;; -*) shift ;; *) break ;; esac; done
SLURM_JOB_ID=4242 SLURM_STEP_GPUS=1 exec "$@"''',
    "hostname": "echo n1",
    "cat": '''for a in "$@"; do [ "$a" = /proc/self/cgroup ] && { echo "$RF_CGLINE"; exit 0; }; done
exec "$RF_SYS/cat" "$@"''',
    # the scope S of the node script, pointed at the fake tree below /sys/fs/cgroup/../../..
    "sed": '''case "$*" in *slurmstepd*/proc/self/cgroup*) echo "/../../..$RF_CG/system.slice/n1_slurmstepd.scope"; exit 0 ;; esac
exec "$RF_SYS/sed" "$@"''',
    "grep": '''case "$*" in *proc/driver/nvidia*) cat "$RF_MAP"; exit 0 ;; esac
exec "$RF_SYS/grep" "$@"''',
    # ps: the calling shell ($PPID, parent outside the list = like sshd/slurmstepd), itself, and the job's processes
    "ps": '''case "$*" in
*pid=,ppid=,stat=,etimes=,args=*) [ -z "$RF_PS_SELF" ] || cat "$RF_PS_SELF"; exit 0 ;;   # node.self: nvidia-smi rows
*pid=,ppid=,stat=,args=*) exit 0 ;;                                                        # the nvidia-smi guard
esac
printf '%7d %7d %7d %7d %4s %4s %5d %-4s %s\\n' "$PPID" 1 "$PPID" 0 0.0 0.0 2000 S "sh -c echo"
printf '%7d %7d %7d %7d %4s %4s %5d %-4s %s\\n' "$$" "$PPID" "$PPID" 0 0.0 0.0 3000 R "ps -ww -u me"
printf '%7d %7d %7d %7d %4s %4s %5d %-4s %s\\n' 4242001 4241999 4242001 3600 0.0 0.0 3400 S "/bin/bash slurm_script"
printf '%7d %7d %7d %7d %4s %4s %5d %-4s %s\\n' 4242040 4242001 4242001 3590 98.5 2.0 10778856 Rl "python train.py"''',
    "scontrol": '''[ "$1" = listpids ] || exit 1
echo "PID      JOBID    STEPID   LOCALID GLOBALID"
[ "$2" = 4242 ] && { echo "4242001  4242     batch    0       0"; echo "4242040  4242     batch    -       -"; }
true''',
    "nvidia-smi": f'''case "$*" in
*query-gpu*) echo "0, {UUID1}, NVIDIA A100-SXM4-40GB, 3, 24881, 40960, 33, 61.20, 400.00, [N/A], P0, 0, 00000000:41:00.0" ;;
*query-compute-apps*) echo "{UUID1}, 4242040, 24870" ;;
*) exit 1 ;;
esac''',
    "pgrep": "exit 1",
    "squeue": 'echo "4242.batch batch"; echo "4242.extern extern"',
    # nothing may be created on any server (§7.16.4): every mktemp call is logged
    "mktemp": '''printf 'mktemp %s\\n' "$*" >> "$RF_LOG"
exec "$RF_SYS/mktemp" "$@"''',
}


def node_env(tmp: Path, shell: str) -> tuple[MockEnv, dict[str, str]]:
    env = MockEnv(tmp, "k2", mocks=(), sh=shell)
    for tool in ("base64", "sed", "grep", "cat"):
        link = env.sysbin / tool
        real = os.path.realpath(shutil.which(tool) or "")
        if real and not link.exists():
            link.symlink_to(real)
    for name, body in STUBS.items():
        p = env.bin / name
        p.write_text("#!/bin/sh\n" + body + "\n")
        p.chmod(0o755)
    job_dir = tmp / "cg" / "system.slice" / "n1_slurmstepd.scope" / "job_4242"
    job_dir.mkdir(parents=True)
    (job_dir / "memory.current").write_text("49000000000\n")
    (job_dir / "memory.max").write_text("51539607552\n")
    (job_dir / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 1\noom_kill 1\n")
    (job_dir / "memory.stat").write_text("anon 30000000000\nfile 900000000\nkernel 1\nshmem 18000000000\n")
    (job_dir / "cpu.stat").write_text("usage_usec 60723561547\nuser_usec 1\n")
    (tmp / "map.txt").write_text("".join(
        f"/proc/driver/nvidia/gpus/{bus}/information:{k}: \t {v}\n"
        for bus, uuid, minor in (("0000:01:00.0", UUID0, 0), ("0000:41:00.0", UUID1, 1))
        for k, v in (("GPU UUID", uuid), ("Bus Location", bus), ("Device Minor", str(minor)))))
    tmpdir = tmp / "tmpdir"
    tmpdir.mkdir()
    extra = {"RF_LOG": str(tmp / "argv.log"), "RF_SYS": str(env.sysbin), "RF_CG": str(tmp / "cg"),
             "RF_MAP": str(tmp / "map.txt"), "TMPDIR": str(tmpdir),
             "RF_CGLINE": "0::/system.slice/n1_slurmstepd.scope/job_4242/step_extern/user/task_0"}
    return env, env.env(**extra)


@pytest.mark.parametrize("shell", shells())
@pytest.mark.parametrize("mode", ["ssh", "srun"])
def test_attach_probe_through_real_shells(tmp_path: Path, shell: str, mode: str) -> None:
    env, environ = node_env(tmp_path, shell)
    job = make_job("k2", "4242", ["n1"], gpus=1)
    col = collector_with("k2", job)
    cx = ctx("k2")
    plan = col.plan_attach(job, mode, cx)                                # type: ignore[arg-type]
    results = run_script(framing.make_script(plan.sections, "attach"), environ, shell=shell)
    live, patch = col.ingest_attach(plan, results, cx)

    assert not live.errors, live.errors
    (n,) = live.nodes
    assert n.errors == (), n.errors
    assert (n.landed_job, n.gpus_of_target) == ("4242", True)
    (g,) = n.gpus
    assert (g.uuid, g.bus_id, g.slurm_idx, g.util_pct) == (UUID1, "00000000:41:00.0", 1, 3.0)
    (p,) = g.processes
    assert (p.pid, p.job, p.command) == (4242040, JobKey("k2", "4242"), "python train.py")
    assert [pr.pid for pr in n.procs] == [4242001, 4242040]               # the probe's own shell and ps dropped
    c = n.cgroups["4242"]
    assert (c.mem_current, c.mem_max, c.oom_kill, c.shmem, c.mem_peak) == (49000000000, 51539607552, 1, 18000000000,
                                                                           None)
    assert n.step_of_pid == {4242001: "4242.batch", 4242040: "4242.batch"} and n.stray_nvidia_smi == 0
    assert live.hints == ("oom_kill 1 — a process of this job was killed for memory",
                          "memory 95% of the job's limit (incl. shmem 16.8G)")
    assert (live.clean, live.audit_detail) == (True, "clean")
    assert patch == ({"node_ssh": "adopt"} if mode == "ssh" else {})
    assert live.sacct_rows_added == (1 if mode == "srun" else 0)

    argv = (tmp_path / "argv.log").read_text()
    if mode == "ssh":
        assert ("-o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=accept-new "
                "-o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts "
                f"/etc/ssh/ssh_known_hosts2 {os.environ['HOME']}/.ssh/known_hosts") in argv   # $HOME expanded there
        assert "-- n1 timeout -k 1 10 sh -c 'echo " in argv
    else:
        assert "--jobid=4242 --overlap --job-name=rfprobe -N1 -n1 -w n1 --time=1 --quiet --input=none timeout" in argv
    assert "mktemp" not in argv                                           # no temp dir on the login host or node
    assert list((tmp_path / "tmpdir").iterdir()) == []


@pytest.mark.parametrize("shell", shells())
def test_refused_hop_keeps_its_stderr_without_a_temp_dir(tmp_path: Path, shell: str) -> None:
    """The login probe captures each hop's stderr in a variable (no temp dir): ssh's refusal still classifies."""
    env, environ = node_env(tmp_path, shell)
    job = make_job("k2", "4242", ["n1"], gpus=1)
    col = collector_with("k2", job)
    plan = col.plan_attach(job, "ssh", ctx("k2"))
    results = run_script(framing.make_script(plan.sections, "attach"), dict(environ, RF_DENY="1"), shell=shell)
    r = results["slurm.attach.node0"]
    assert r.rc == 255 and "Permission denied (publickey,hostbased)." in r.stderr
    live, patch = col.ingest_attach(plan, results, ctx("k2"))
    assert patch == {"node_ssh": "denied"} and live.errors[0].summary == "ssh to n1 refused (Permission denied)"
    assert "mktemp" not in (tmp_path / "argv.log").read_text()
    assert list((tmp_path / "tmpdir").iterdir()) == []


@pytest.mark.parametrize("shell", shells())
def test_node_script_runs_alone_and_stays_within_its_timeouts(tmp_path: Path, shell: str) -> None:
    """The exact remote command of the ssh hop, run by `sh -c` the way sshd would, with no outer framing."""
    env, environ = node_env(tmp_path, shell)
    script = attach.node_script(["4242", "999"], ["", "sNOPE00"])
    b64 = attach.encode(script)
    remote = attach.ssh_cmd("n1", b64).split(" -- n1 ", 1)[1][1:-1]     # the double-quoted remote command
    out = subprocess.run([shell, "-c", remote], capture_output=True, env=environ, timeout=30, stdin=subprocess.DEVNULL)
    parsed = framing.parse(out.stdout, script)
    assert parsed.ended == len(attach.NODE_SECTIONS) and out.returncode == 0
    assert parsed.sections["node.cg"].stdout.splitlines()[-1] == "@@cg 999 -"   # neither job_999 nor the SLUID
    assert list((tmp_path / "tmpdir").iterdir()) == []


@pytest.mark.parametrize("shell", shells())
@pytest.mark.parametrize("rows,stuck,other,slow", [
    ("", (), 0, False),
    (f"{HEALTHY}\n", (), 2, False),                     # a job's own nvidia-smi: no second read, no delay
    (f"{HEALTHY}\n{STUCK}\n", (2102390,), 2, True),     # D state on both reads 1 s apart: hung
])
def test_node_self_tells_stuck_queries_from_healthy_ones(tmp_path: Path, shell: str, rows: str,
                                                          stuck: tuple[int, ...], other: int, slow: bool) -> None:
    env, environ = node_env(tmp_path, shell)
    (tmp_path / "self.txt").write_text(rows)
    t0 = time.monotonic()
    out = subprocess.run([shell, "-c", attach.NODE_SELF_CMD], capture_output=True, text=True, timeout=10,
                         env=dict(environ, RF_PS_SELF=str(tmp_path / "self.txt")), stdin=subprocess.DEVNULL)
    elapsed = time.monotonic() - t0
    chk = pa.parse_self(out.stdout)
    assert out.returncode == 0 and chk is not None, out
    assert (chk.stuck, chk.other) == (stuck, other)
    assert (elapsed >= 1.0) is slow
