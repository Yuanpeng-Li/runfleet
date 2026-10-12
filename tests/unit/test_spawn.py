"""[WP1] transport/spawn.py: tty-less spawning (§4.2), timeouts, drains, streams."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from runfleet.transport import classify, spawn

FAKE_PROMPTING_SSH = """#!/bin/sh
# Behaves like ssh asking for a passphrase: it needs the controlling terminal.
if { printf 'Enter passphrase for key: ' > /dev/tty; } 2>/dev/null; then
  read -r pw < /dev/tty
  echo "got a password" >&2
  exit 0
fi
echo "read_passphrase: can't open /dev/tty: No such device or address" >&2
echo "alice07@lyra: Permission denied (publickey,gssapi-with-mic)." >&2
exit 255
"""

HARNESS = textwrap.dedent("""
    import asyncio, json, os, sys
    sys.path.insert(0, {src!r})
    from runfleet.transport import spawn
    # We are a session leader without a controlling terminal: opening the pty slave makes it ours.
    fd = os.open({slave!r}, os.O_RDWR)
    tty_ok = True
    try:
        os.close(os.open("/dev/tty", os.O_RDWR))
    except OSError:
        tty_ok = False
    res = asyncio.run(spawn.run([{fake!r}], timeout=10))
    with open({out!r}, "w") as f:
        json.dump({{"tty_ok": tty_ok, "rc": res.rc, "stderr": res.stderr.decode()}}, f)
""")


def test_spawned_ssh_cannot_prompt_on_the_terminal(tmp_path: Path) -> None:
    """§10.3 no-tty: a prompting fake ssh spawned via spawn.py inside a pty harness fails (ENXIO), writes nothing
    to the pty, and classifies as AUTH."""
    fake = tmp_path / "ssh"
    fake.write_text(FAKE_PROMPTING_SSH, encoding="utf-8")
    fake.chmod(0o755)
    out = tmp_path / "result.json"
    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)
    src = str(Path(spawn.__file__).resolve().parents[2])
    code = HARNESS.format(src=src, slave=slave_name, fake=str(fake), out=str(out))
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    os.close(slave)
    written = b""
    deadline = time.monotonic() + 20
    while proc.poll() is None and time.monotonic() < deadline:
        r, _, _ = select.select([master], [], [], 0.1)
        if r:
            try:
                written += os.read(master, 1024)
            except OSError:
                break
    proc.wait(timeout=5)
    os.close(master)
    data = json.loads(out.read_text())
    assert data["tty_ok"] is True                    # the harness really had a controlling terminal
    assert data["rc"] == 255
    assert b"passphrase" not in written and written == b""
    err = classify.classify_ssh(data["rc"], data["stderr"])
    assert err is not None and err.kind.value == "auth"


def test_scrubbed_env() -> None:
    env = spawn.scrubbed_env({"DISPLAY": ":0", "SSH_ASKPASS": "/usr/bin/x", "PATH": "/bin", "SSH_AUTH_SOCK": "/s"})
    assert env == {"PATH": "/bin", "SSH_AUTH_SOCK": "/s", "SSH_ASKPASS_REQUIRE": "never"}


async def test_run_sees_scrubbed_env_and_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    res = await spawn.run(["sh", "-c", 'echo "${DISPLAY-unset} $SSH_ASKPASS_REQUIRE"; ps -o sid= -p $$; '
                                       'ps -o pid= -p $$'], timeout=5)
    lines = res.stdout.decode().split()
    assert lines[:2] == ["unset", "never"]
    assert lines[2] == lines[3]                       # session id == own pid: a new session
    assert res.rc == 0 and not res.timed_out


async def test_run_feeds_stdin_and_times_out() -> None:
    res = await spawn.run(["sh", "-c", "cat; echo end"], timeout=5, input=b"hello\n")
    assert res.stdout == b"hello\nend\n"
    t0 = time.monotonic()
    res = await spawn.run(["sh", "-c", "echo partial; sleep 30"], timeout=0.5)
    assert res.timed_out and res.rc is None and res.stdout == b"partial\n"
    assert time.monotonic() - t0 < 3


async def test_run_does_not_wait_for_a_descendant_holding_the_pipe() -> None:
    t0 = time.monotonic()
    res = await spawn.run(["sh", "-c", "(sleep 20 &) ; echo quick"], timeout=10)
    assert res.rc == 0 and res.stdout.startswith(b"quick")
    assert time.monotonic() - t0 < 3                  # exit + 0.5 s drain, not 20 s


async def test_run_input_larger_than_pipe_buffer() -> None:
    data = b"x" * (1 << 20)
    res = await spawn.run(["sh", "-c", "wc -c"], timeout=10, input=data)
    assert res.stdout.strip() == str(len(data)).encode()


async def test_spawn_error() -> None:
    with pytest.raises(spawn.SpawnError):
        await spawn.run(["/nonexistent/ssh"], timeout=1)


async def test_cancellation_kills_the_process(tmp_path: Path) -> None:
    marker = tmp_path / "pid"
    task = asyncio.ensure_future(spawn.run(["sh", "-c", f"echo $$ > {marker}; sleep 30"], timeout=60))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if marker.exists() and marker.read_text().strip():
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(marker.read_text())
    await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_process_stream_chunks_and_watchdog() -> None:
    cmd = "printf 'a\\nb'; echo oops >&2; cat >/dev/null; echo never"
    stream = await spawn.ProcessStream.start(["sh", "-c", cmd])
    chunks = stream.chunks()
    first = await asyncio.wait_for(chunks.__anext__(), 5)
    assert first == b"a\nb"
    await stream.close()                              # closing stdin ends `cat`, the process exits
    assert stream.returncode is not None
    assert "oops" in stream.stderr_text


def test_failed_probe_marks_everything_not_run() -> None:
    from runfleet.models import ErrorKind, HostError
    from runfleet.probe.section import ProbeScript, Section, SectionOutcome

    s = ProbeScript((Section("a.b", "x"),), "0123456789ab")
    res = spawn.failed_probe(s, HostError(ErrorKind.SSH, "boom"))
    assert res.error is not None and res.sections["a.b"].outcome is SectionOutcome.NOT_RUN


async def test_probe_timer_expiring_while_the_loop_was_frozen_is_not_a_timeout() -> None:
    """Critic: after `s`/`c`/`p` (synchronous App.suspend) or `^z`, in-flight probes were killed as TIMEOUT and the
    host flipped to retry; a freeze gives them one fresh budget instead."""
    import time as _time

    # still running when the loop freezes, then 300 KB of output: the child blocks on the full pipe nobody reads, so
    # it cannot exit during the freeze and the (expired) timer fires first on resume
    task = asyncio.ensure_future(spawn.run(["sh", "-c", "sleep 0.3; head -c 300000 /dev/zero; echo done >&2"],
                                           timeout=0.4))
    await asyncio.sleep(0.15)
    with spawn.loop_frozen():
        _time.sleep(0.6)                              # the TUI is suspended: the loop does not run
    res = await task
    assert not res.timed_out and res.rc == 0 and res.stderr == b"done\n"
    hung = asyncio.ensure_future(spawn.run(["sleep", "30"], timeout=0.3))
    await asyncio.sleep(0.05)
    with spawn.loop_frozen():
        _time.sleep(0.5)
    res = await hung                                  # still a real timeout after the fresh budget
    assert res.timed_out and res.rc is None


# ---------------------------------------------------------------- review regressions
def _procs_with(arg: str) -> list[int]:
    out = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    words = f.read().split(b"\0")
            except OSError:
                continue
            if words[:1] == [b"sleep"] and arg.encode() in words:
                out.append(int(entry))
    return out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc")
async def test_cancellation_also_kills_descendants_in_their_own_group() -> None:
    """GNU timeout setpgid()s itself: the group SIGKILL alone left it and its section running."""
    task = asyncio.ensure_future(spawn.run(["sh", "-c", "timeout 30 sleep 47.25; :"], timeout=60))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if _procs_with("47.25"):
            break
    assert _procs_with("47.25")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    assert _procs_with("47.25") == []


async def test_process_stream_close_survives_a_cancelled_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spawn, "KILL_GRACE", 0.3)
    s = await spawn.ProcessStream.start(["sh", "-c", "exec sleep 30"])          # ignores stdin EOF
    proc = s._proc
    first = asyncio.ensure_future(s.close())
    await asyncio.sleep(0.05)
    first.cancel()                                                            # during the exit wait
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(s.close(), 5)                                      # not a no-op: waits for the shutdown
    assert proc.returncode is not None and proc not in spawn._CHILDREN


async def test_attach_probe_is_fed_without_a_temp_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """§7.16.4: the login probe of an Attach sample (purpose "attach") writes nothing on the login host; every other
    probe keeps its private temp dir."""
    from runfleet.probe import framing
    from runfleet.probe.section import Section

    fed: dict[str, bytes] = {}

    async def fake_run(argv: object, *, timeout: float, input: bytes | None = None, **kw: object) -> spawn.RunResult:
        fed[s.purpose] = input or b""
        return spawn.RunResult(rc=0, stdout=b"", stderr=b"", elapsed_s=0.0, timed_out=False)

    monkeypatch.setattr(spawn, "run", fake_run)
    for purpose in ("attach", "poll", "detail"):
        s = framing.make_script([Section("slurm.attach.audit", "true")], purpose)
        await spawn.run_probe(["sh", "-s"], s, timeout=5, classify=lambda r, p: None)
    assert b"mktemp" not in fed["attach"] and b"RF_T=\n" in fed["attach"]
    assert b"mktemp" in fed["poll"] and b"mktemp" in fed["detail"]
