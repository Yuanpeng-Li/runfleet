"""[WP2] Self-tests of the conftest helpers: sample(), meta() (last entry wins), frame(), no-ssh guard."""

from __future__ import annotations

import shutil

import pytest

from tests.conftest import _argv_of, _spawns_real_ssh, _ssh_stem, frame, meta, sample, sample_path


def test_sample_is_byte_exact() -> None:
    raw = sample_path("k2", "squeue_me_O").read_bytes()
    assert sample("k2", "squeue_me_O").encode("utf-8") == raw
    with pytest.raises(FileNotFoundError):
        sample("k2", "does_not_exist")


def test_meta_last_entry_wins() -> None:
    # rc 1 `sacct -j NONE` in _meta_followup, the real rc 0 run in _meta_followup2 (§10.2)
    m = meta("k2", "sacct_completed_gpu_tres")
    assert m["rc"] == 0
    assert "2333102" in m["cmd"]


def test_meta_stderr() -> None:
    m = meta("k2", "scontrol_show_job_finished")
    assert m["rc"] == 1
    assert m["stderr"] == "slurm_load_jobs error: Invalid job id specified"
    assert meta("k2", "sinfo_version") == {"cmd": "sinfo --version", "rc": 0, "ms": 26, "stderr": ""}
    with pytest.raises(KeyError):
        meta("k2", "nope")


def test_meta_fixture(meta: object) -> None:
    assert callable(meta)


def test_frame_layout() -> None:
    out = frame({"slurm.squeue": ("a\nb", 0, ""), "slurm.sacct": ("", 1, "boom"),
                 "nvidia.gpu": ("", 127, "missing:nvidia-smi")}, "abcdefabcdef", ms=7)
    rf = "@@RFabcdefabcdef"
    assert out == (
        f"\n{rf} B slurm.squeue\na\nb\n{rf} E slurm.squeue 0 7\n"         # the prelude's order (CORE-27/32)
        f"\n{rf} B slurm.sacct\n\n{rf} X slurm.sacct\nboom\n{rf} Y slurm.sacct\n\n{rf} E slurm.sacct 1 7\n"
        f"\n{rf} B nvidia.gpu\n\n{rf} E nvidia.gpu 127 0 missing:nvidia-smi\n"
    )


@pytest.mark.skipif(shutil.which("ssh") is None, reason="no system ssh")
def test_no_ssh_guard_policy(tmp_path: object) -> None:
    assert _spawns_real_ssh(["ssh", "k2", "exec sh -s"], None, None)
    assert _spawns_real_ssh(_argv_of("ssh -T k2 true", shell=True), None, None)
    assert not _spawns_real_ssh(["ssh", "-G", "k2"], None, None)
    assert not _spawns_real_ssh(["sh", "-s"], None, None)
    assert not _spawns_real_ssh(["ssh", "k2"], None, {"PATH": "/nonexistent"})
    real = shutil.which("ssh")
    assert real is not None and _spawns_real_ssh([real, "k2"], None, None)


def test_no_ssh_guard_knows_windows_names() -> None:
    assert _ssh_stem("ssh") == _ssh_stem("ssh.exe") == _ssh_stem(r"C:\Windows\System32\OpenSSH\SSH.EXE") == "ssh"
    assert _ssh_stem("C:/Program Files/Git/usr/bin/ssh.exe") == "ssh"
    assert _ssh_stem("sshd") == "sshd" and _ssh_stem("/usr/bin/ssh-agent") == "ssh-agent"


def test_gpu_idle_run_breaks_across_a_telemetry_gap() -> None:
    """Review BACK-22: two idle samples, then an hour without telemetry, then one idle sample: not 'GPU idle 1h'."""
    from datetime import timedelta

    from runfleet.backends.common import GpuIdleTracker
    from tests.backends.factories import NOW
    t = GpuIdleTracker()
    t.add("j", NOW, 0.0)
    t.add("j", NOW + timedelta(minutes=1), 0.0)
    t.add("j", NOW + timedelta(hours=1), 0.0)
    assert t.hint("j", NOW + timedelta(hours=1)) is None
    for m in range(1, 17):
        t.add("j", NOW + timedelta(hours=1, minutes=m), 0.0)
    assert t.hint("j", NOW + timedelta(hours=1, minutes=16)) == "GPU idle 16m"
