"""[WP1] SessionTransport (§4.8) through a fake ssh: a Python script that answers `ssh -G` and otherwise runs the
remote command locally under `sh -c`, relaying stdin/stdout/stderr like ssh does (so killing the fake ends the
session's pipes, as a dropped connection would). The agent itself is the real one (tests/unit/test_agent.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from runfleet.config import HostConfig
from runfleet.core.logs import tail_spec
from runfleet.errors import RunFleetError
from runfleet.models import ErrorKind
from runfleet.probe import framing
from runfleet.probe.section import Section, SectionOutcome
from runfleet.sshconfig import parse_ssh_g
from runfleet.transport import agent, make_transport, mux, session
from runfleet.transport.local import LocalTransport
from runfleet.transport.openssh import OpenSSHTransport
from runfleet.transport.session import InteractiveLogin, SessionTransport
from runfleet.util import platform

pytestmark = pytest.mark.posix          # the fake ssh is a #! script; the agent needs sh

G_K2 = "user alice0001\nhostname k2.example.edu\nport 22\ncontrolmaster auto\n" \
       "controlpath /home/usr/.ssh/cm-0123\ncontrolpersist 600\n"
G_LYRA = "user alice07\nhostname lyra.example.edu\nport 22\ncontrolmaster false\ncontrolpersist no\n" \
         "proxyjump jumpvm\n"

FAKE_SSH = r'''#!{python}
"""Fake ssh: records argv; -G prints $RF_FAKE_DIR/g.txt; modes in $RF_FAKE_DIR/mode; else relays `sh -c CMD`."""
import os, subprocess, sys, threading, time
d = os.environ["RF_FAKE_DIR"]
args = sys.argv[1:]
with open(os.path.join(d, "argv.log"), "a") as f:
    f.write("\x1f".join(args) + "\n")
if "-G" in args:
    with open(os.path.join(d, "g.txt")) as f:
        sys.stdout.write(f.read())
    sys.exit(0)
log = args[args.index("-E") + 1] if "-E" in args else None
try:
    mode = open(os.path.join(d, "mode")).read().strip()
except OSError:
    mode = ""
def fail(text, rc=255):
    if log:
        with open(log, "a") as f:
            f.write(text + "\n")
    else:
        sys.stderr.write(text + "\n")
    sys.exit(rc)
if mode == "auth":
    fail("alice07@lyra: Permission denied (publickey,keyboard-interactive).")
if mode == "refused":
    fail("ssh: connect to host lyra.example.edu port 22: Connection refused")
if mode == "hang":
    time.sleep(60)
if mode == "prompt":                # an ssh at its password prompt: SIGTERM restores the terminal (readpassphrase)
    import signal
    def restore(signum, frame):
        open(os.path.join(d, "restored"), "w").close()
        os._exit(255)
    signal.signal(signal.SIGTERM, restore)
    open(os.path.join(d, "prompting"), "w").close()
    time.sleep(60)
    sys.exit(255)
cmd = args[-1] if mode != "noagent" else "echo 'sh: 1: dd: not found' >&2; exit 127"
def drop(signum, frame):            # SIGUSR1: the connection drops (ssh logs why and exits 255)
    with open(log, "a") as f:
        f.write("client_loop: send disconnect: Broken pipe\n")
    os._exit(255)
import signal
signal.signal(signal.SIGUSR1, drop)
p = subprocess.Popen(["sh", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def pump(src, dst, close):
    slow = mode == "slow" and dst == 1      # a slow link: about 34 KB/s towards RunFleet
    try:
        while True:
            b = os.read(src, 1024 if slow else 65536)
            if not b:
                break
            os.write(dst, b)
            if slow:
                time.sleep(0.03)
    except OSError:
        pass
    if close:
        try: os.close(dst)
        except OSError: pass
threading.Thread(target=pump, args=(0, p.stdin.fileno(), True), daemon=True).start()
ts = [threading.Thread(target=pump, args=(p.stdout.fileno(), 1, False)),
      threading.Thread(target=pump, args=(p.stderr.fileno(), 2, False))]
for t in ts: t.start()
for t in ts: t.join()
sys.exit(p.wait())
'''


@pytest.fixture(autouse=True)
def _posix_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "ssh"
    exe.write_text(FAKE_SSH.replace("{python}", sys.executable), encoding="utf-8")
    exe.chmod(0o755)
    (tmp_path / "g.txt").write_text(G_K2, encoding="utf-8")
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    monkeypatch.setenv("RF_FAKE_DIR", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(remote_tmp))
    return exe


def mode(tmp_path: Path, value: str) -> None:
    (tmp_path / "mode").write_text(value, encoding="utf-8")


def calls(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "argv.log"
    lines = path.read_text().splitlines() if path.exists() else []
    return [line.split("\x1f") for line in lines if not line.startswith("-G")]


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make(fake: Path, tmp_path: Path, host: HostConfig | None = None, clock: Clock | None = None) -> SessionTransport:
    return SessionTransport(host or HostConfig("k2", "k2", mux="session"), ssh=str(fake), mux_dir=tmp_path / "mux",
                            log_path=tmp_path / "ssh-k2.log", monotonic=clock or time.monotonic)


def leftovers(marker: str) -> list[str]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,args="], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if marker in line and "ps -A" not in line]


async def gone(marker: str, timeout: float = 5) -> list[str]:
    deadline = time.monotonic() + timeout
    while (left := leftovers(marker)) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return left


async def until(cond: object, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not cond():                                       # type: ignore[operator]
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------- policy (pure)
@pytest.mark.parametrize(("windows", "cfg", "expected"), [
    (False, "auto", False), (False, "managed", False), (False, "off", False), (False, "session", True),
    (True, "auto", True), (True, "managed", True), (True, "user", True), (True, "session", True), (True, "off", False),
])
def test_who_gets_the_session_transport(monkeypatch: pytest.MonkeyPatch, windows: bool, cfg: str,
                                        expected: bool) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", windows)
    monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", r"C:\Users\u\AppData\Local")
    host = HostConfig("k2", "k2", mux=cfg)                                   # type: ignore[arg-type]
    assert mux.wants_session(host) is expected
    plan = mux.decide(parse_ssh_g(G_K2, "k2"), host, Path("/run/user/1000/runfleet"))
    if expected:
        assert plan.mode == "session" and not plan.uses_master and plan.control_path is None
        assert plan.probe_opts == ("-o", "ControlMaster=no", "-o", "ControlPath=none")
    t = make_transport(host)
    assert isinstance(t, SessionTransport) is expected and t.kind == "openssh"
    assert isinstance(t, InteractiveLogin) is expected
    assert isinstance(make_transport(HostConfig("atlas", "local", mux=cfg)), LocalTransport)   # type: ignore[arg-type]


def test_config_accepts_session() -> None:
    from runfleet.config import parse_config
    from runfleet.errors import ConfigError

    assert parse_config({"host": [{"name": "k2", "mux": "session"}]})[0].hosts[0].mux == "session"
    with pytest.raises(ConfigError):
        parse_config({"host": [{"name": "k2", "mux": "sessions"}]})


async def test_session_argv(fake: Path, tmp_path: Path) -> None:
    (tmp_path / "g.txt").write_text(G_LYRA, encoding="utf-8")
    t = make(fake, tmp_path, HostConfig("lyra", "lyra", mux="session", connect_timeout=60,
                                        ssh_options=("ServerAliveInterval=5",), remote_shell="login"))
    assert await t.resolve() is None and t.mux == "session"
    log = str(tmp_path / "ssh-k2.log")
    jump_b = f"ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -E {log} -W '[%h]:%p' jumpvm"
    jump_i = f"ProxyCommand=ssh -T -o ConnectTimeout=10 -o LogLevel=ERROR -E {log} -W '[%h]:%p' jumpvm"
    tail = ["-o", "ServerAliveInterval=5", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=15", "-o",
            "ServerAliveCountMax=4", "-o", "LogLevel=ERROR", "-E", log, "--", "lyra",
            "exec bash -l -c 'IFS= read -r n; eval \"$(dd bs=1 count=$n 2>/dev/null)\"'"]
    assert t.session_argv(batch=True) == [str(fake), "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o",
                                          "ControlPath=none", "-o", jump_b, *tail]
    login = t.session_argv(batch=False)
    assert "BatchMode=yes" not in login and jump_i in login and "ConnectTimeout=30" in login
    assert t.connect_argv() == [str(fake), "-o", "ControlPath=none", "-o", jump_i, "-o", "ServerAliveInterval=5",
                                "-o", "ConnectTimeout=30", "-o", "LogLevel=ERROR", "-E", log, "--", "lyra", "true"]
    assert t.interactive_argv(None)[:4] == [str(fake), "-t", "-o", "ControlPath=none"]
    assert t.ssh_log == log
    assert "session" in (t.forward_blocker() or "") and await t.forward(8888, "n1", 8888) is not None


# ---------------------------------------------------------------- lifecycle
async def test_prepare_ping_probe_and_close(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    assert await t.check_master() is False                                # nothing up yet
    assert await t.prepare() is None and t.mux == "session"
    assert await t.check_master() is True and await t.before_connect() is True
    assert await t.prepare() is None                                      # idempotent: a ping, no second ssh
    assert len(calls(tmp_path)) == 1 and "BatchMode=yes" in calls(tmp_path)[0]
    s = framing.make_script([Section("a.one", "echo hello"), Section("a.two", "exit 4"),
                             Section("a.err", "echo e >&2")])
    res = await t.probe(s, timeout=10)
    assert res.error is None and res.run.rc == 0
    assert res.sections["a.one"].stdout == "hello\n" and res.sections["a.two"].outcome is SectionOutcome.FAILED
    assert res.sections["a.err"].outcome is SectionOutcome.STDERR_ONLY
    assert len(list((tmp_path / "remote-tmp").iterdir())) >= 1            # the agent's private dir (+ the probe's)
    await t.close()
    assert await t.check_master() is False
    await until(lambda: list((tmp_path / "remote-tmp").iterdir()) == [])  # stdin EOF: the agent cleaned up
    assert await gone(str(tmp_path / "ssh")) == []
    await until(lambda: not [th for th in threading.enumerate() if th.name.startswith("rf-session-k2-")])


async def test_probe_without_a_session_starts_one(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    res = await t.probe(framing.make_script([Section("a.one", "echo hi")]), timeout=10)
    assert res.error is None and res.sections["a.one"].stdout == "hi\n"
    await t.close()


async def test_framing_matches_plain_sh(fake: Path, tmp_path: Path) -> None:
    """Byte-exact: the same script through the session parses to what LocalTransport (plain `sh -s`) gives."""
    t = make(fake, tmp_path)
    sections = [Section("t.nonl", "printf abc"), Section("t.blank", "printf 'abc\\n\\n'"),
                Section("t.binary", "printf '\\001\\377\\033[31mred\\r\\n'"), Section("t.stdin", "cat; echo a"),
                Section("t.big", "printf '%9000s\\n' x"), Section("t.rc", "echo x; exit 3"),
                Section("t.missing", "x", requires=("nosuchtool_rf",)), Section("t.late", "echo l", late=True)]
    s = framing.make_script(sections)
    via = await t.probe(s, timeout=20)
    plain = await LocalTransport(HostConfig("atlas", "local")).probe(s, timeout=20)
    assert via.error is None
    for name, b in plain.sections.items():
        a = via.sections[name]
        assert (a.outcome, a.rc, a.stdout, a.stderr) == (b.outcome, b.rc, b.stdout, b.stderr), name
    await t.close()


async def test_concurrent_probes_beyond_the_agent_cap(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    scripts = [framing.make_script([Section("c.x", f"sleep 0.5; echo {i}")]) for i in range(12)]
    t0 = time.monotonic()
    results = await asyncio.gather(*(t.probe(s, timeout=20) for s in scripts))
    assert [r.sections["c.x"].stdout for r in results] == [f"{i}\n" for i in range(12)]
    assert all(r.error is None for r in results)                          # never "busy": the client keeps to 8
    assert time.monotonic() - t0 < 5
    await t.close()


async def test_a_timeout_kills_only_that_job(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    slow = framing.make_script([Section("s.slow", "sleep 38.7; echo never", timeout=60)])
    quick = framing.make_script([Section("s.quick", "sleep 1.5; echo quick")])
    r_slow, r_quick = await asyncio.gather(t.probe(slow, timeout=1.0), t.probe(quick, timeout=10))
    assert r_slow.run.timed_out and r_slow.error is not None and r_slow.error.kind is ErrorKind.TIMEOUT
    assert r_slow.sections["s.slow"].outcome is SectionOutcome.TIMEOUT
    assert r_quick.error is None and r_quick.sections["s.quick"].stdout == "quick\n"
    assert await gone("sleep 38.7") == []                                  # the job's command is gone …
    assert await t.check_master() is True                                  # … the session is not
    assert len(calls(tmp_path)) == 1
    await t.close()


async def test_stream_through_the_session(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    log = tmp_path / "job.out"
    log.write_text("line1\nline2\n")
    handle = await t.stream(tail_spec(str(log), 65536))
    got = b""
    async for chunk in handle.chunks():
        got += chunk
        if b"line2\n" in got:
            break
    assert handle.returncode is None
    with log.open("a") as f:
        f.write("progress 1\rprogress 2\r")
    async for chunk in handle.chunks():
        got += chunk
        if got.endswith(b"progress 2\r"):
            break
    await handle.close()
    assert got == b"line1\nline2\nprogress 1\rprogress 2\r"
    assert handle.returncode is not None
    missing = await t.stream(tail_spec(str(tmp_path / "nope.out"), 100))    # tail -F waits for it to appear
    await asyncio.sleep(0.5)
    await missing.close()                                                   # stderr comes back with the job's end
    assert missing.returncode is not None and "nope.out" in missing.stderr_text
    assert await t.check_master() is True
    await t.close()


async def test_auth_failure_is_classified_and_throttled(fake: Path, tmp_path: Path) -> None:
    clock = Clock()
    t = make(fake, tmp_path, clock=clock)
    mode(tmp_path, "auth")
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.AUTH
    assert "Permission denied" in err.detail
    clock.t += 10
    assert await t.prepare() == err and len(calls(tmp_path)) == 1           # throttled: no second login
    res = await t.probe(framing.make_script([Section("a.b", "true")]), timeout=5)
    assert res.error == err and res.sections["a.b"].outcome is SectionOutcome.NOT_RUN
    with pytest.raises(RunFleetError, match="authentication"):
        await t.stream(tail_spec(str(tmp_path / "x.out"), 100))
    mode(tmp_path, "")
    assert await t.prepare(force=True) is None and len(calls(tmp_path)) == 2   # a user retry bypasses it
    await t.close()


async def test_remote_without_the_agent_tools(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    mode(tmp_path, "noagent")
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.REMOTE and "dd: not found" in err.detail


async def test_agent_that_never_answers_times_out(fake: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "READY_GRACE", 0.0)
    t = make(fake, tmp_path, HostConfig("k2", "k2", mux="session", connect_timeout=1))
    mode(tmp_path, "hang")
    t0 = time.monotonic()
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.TIMEOUT and time.monotonic() - t0 < 6
    assert await gone(str(tmp_path / "ssh")) == []                          # the stuck ssh was killed


async def test_session_death_fails_jobs_and_reconnects_throttled(fake: Path, tmp_path: Path) -> None:
    clock = Clock()
    t = make(fake, tmp_path, clock=clock)
    assert await t.prepare() is None
    s = t._session
    assert s is not None
    probe = asyncio.ensure_future(t.probe(framing.make_script([Section("d.x", "sleep 39.1")]), timeout=30))
    await until(lambda: leftovers("sleep 39.1"))
    os.kill(s.pid, signal.SIGUSR1)                                          # the connection drops
    res = await asyncio.wait_for(probe, 5)
    assert res.error is not None and res.error.kind is ErrorKind.MUX and res.error.summary == "ssh session ended"
    assert "Broken pipe" in res.error.detail
    assert res.sections["d.x"].outcome is SectionOutcome.INCOMPLETE        # its begin marker had arrived
    assert await t.check_master() is False
    assert await gone("sleep 39.1") == []                                   # stdin EOF reached the agent
    clock.t += 5                                                            # < 30 s after the last start
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.MUX and len(calls(tmp_path)) == 1
    clock.t += 30
    assert await t.prepare() is None and len(calls(tmp_path)) == 2           # reconnected
    assert (await t.probe(framing.make_script([Section("d.y", "echo back")]), timeout=10)).error is None
    await t.close()


async def test_killed_session_ssh_is_a_transient_ssh_error(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    s = t._session
    assert s is not None
    probe = asyncio.ensure_future(t.probe(framing.make_script([Section("d.x", "sleep 39.6")]), timeout=30))
    await until(lambda: leftovers("sleep 39.6"))
    os.kill(s.pid, signal.SIGKILL)                                          # `pkill ssh`, the OOM killer
    res = await asyncio.wait_for(probe, 5)
    assert res.error is not None and res.error.kind is ErrorKind.SSH and res.error.summary == "ssh killed by SIGKILL"
    assert await gone("sleep 39.6") == []
    await t.close()


async def test_reset_master_ends_the_session(fake: Path, tmp_path: Path) -> None:
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    await t.reset_master()
    assert await t.check_master() is False
    assert await t.prepare() is None and len(calls(tmp_path)) == 2           # no throttle after a reset
    await t.close()


async def test_interactive_login(fake: Path, tmp_path: Path) -> None:
    """`login()` blocks (the TUI is suspended); its session then serves the polls."""
    t = make(fake, tmp_path)
    assert await t.before_connect() is False
    assert t.login() == 0
    argv = calls(tmp_path)[-1]
    assert "BatchMode=yes" not in argv and "ConnectTimeout=30" in argv
    assert await t.check_master() is True and await t.prepare() is None and len(calls(tmp_path)) == 1
    res = await t.probe(framing.make_script([Section("l.x", "echo logged-in")]), timeout=10)
    assert res.error is None and res.sections["l.x"].stdout == "logged-in\n"
    assert await t.before_connect() is True                                 # `c` again: already connected
    mode(tmp_path, "auth")
    await t.reset_master()
    assert await t.before_connect() is False
    assert t.login() == 255 and t._session is None
    await t.close()


def test_login_argv_keeps_the_terminal_and_ignores_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    assert platform.login_argv(["ssh", "-T", "h"]) == ["/bin/sh", "-c", 'trap "" INT QUIT; exec "$@"',
                                                       "runfleet-login", "ssh", "-T", "h"]
    assert platform.login_spawn_kwargs() == {}
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.login_argv(["ssh", "h"]) == ["ssh", "h"]
    assert platform.login_spawn_kwargs() == {"creationflags": platform.CREATE_NEW_PROCESS_GROUP}
    out = subprocess.run(["/bin/sh", "-c", 'trap "" INT QUIT; exec "$@"', "x", "sh", "-c",
                          "kill -INT $$; echo survived"], capture_output=True, text=True)
    assert out.stdout == "survived\n"


# ---------------------------------------------------------------- the transport contract, both transports
@pytest.mark.parametrize("cls", [OpenSSHTransport, SessionTransport])
async def test_contract_probe_and_stream(fake: Path, tmp_path: Path, cls: type[OpenSSHTransport]) -> None:
    host = HostConfig("k2", "k2", mux="off" if cls is OpenSSHTransport else "session")
    t = cls(host, ssh=str(fake), mux_dir=tmp_path / "mux", log_path=tmp_path / "ssh-k2.log")
    assert await t.prepare() is None
    s = framing.make_script([Section("a.one", "echo hello"), Section("a.two", "exit 4")])
    res = await t.probe(s, timeout=10)
    assert res.error is None
    assert res.sections["a.one"].stdout == "hello\n" and res.sections["a.two"].outcome is SectionOutcome.FAILED
    log = tmp_path / "job.out"
    log.write_text("line1\nline2\n")
    handle = await t.stream(tail_spec(str(log), 65536))
    got = b""
    async for chunk in handle.chunks():
        got += chunk
        if b"line2\n" in got:
            break
    await handle.close()
    assert got == b"line1\nline2\n" and handle.returncode is not None
    mode(tmp_path, "auth")
    await t.reset_master()
    t2 = cls(host, ssh=str(fake), mux_dir=tmp_path / "mux", log_path=tmp_path / "ssh-k2.log")
    res = await t2.probe(s, timeout=10)
    assert res.error is not None and res.error.kind is ErrorKind.AUTH
    assert all(r.outcome is SectionOutcome.NOT_RUN for r in res.sections.values())
    await t.close()
    await t2.close()


# ---------------------------------------------------------------- review regressions (P2)
@pytest.mark.skipif(not shutil.which("setsid"), reason="needs setsid")
async def test_probes_whose_jobs_cannot_be_killed(fake: Path, tmp_path: Path) -> None:
    """Eight probes whose late section leaves an unkillable stdout holder (a stand-in for nvidia-smi in D state):
    each keeps the sections that finished (mawk used to hold every line until EOF; review: protocol 1), and the
    session is not left "busy" for good afterwards (the agent counted the jobs forever; protocol 2)."""
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    script = framing.make_script([Section("a.ok", "echo ok"),
                                  Section("b.stuck", "(setsid sh -c 'exec sleep 45.6' &); echo started; sleep 45.5",
                                          late=True)])
    try:
        results = await asyncio.gather(*(t.probe(script, timeout=1.5) for _ in range(agent.MAX_JOBS)))
        for r in results:
            assert r.run.timed_out and r.error is None                     # §4.2 partial rule: a.ok ended
            assert r.sections["a.ok"].outcome is SectionOutcome.OK and r.sections["a.ok"].stdout == "ok\n"
            assert r.sections["b.stuck"].outcome is SectionOutcome.TIMEOUT
        r = await t.probe(framing.make_script([Section("c.after", "echo free")]), timeout=10)
        assert r.error is None and r.sections["c.after"].stdout == "free\n"
        assert await t.check_master() is True
    finally:
        for line in leftovers("sleep 45.6"):
            with contextlib.suppress(OSError):
                os.kill(int(line.split()[0]), signal.SIGKILL)
        await t.close()
    assert await gone("sleep 45.") == []


async def test_a_ping_counts_output_that_is_still_arriving(fake: Path, tmp_path: Path) -> None:
    """Everything shares one stream, so on a slow link a pong waits behind a big fetch: the session is alive while
    its bytes keep arriving (review: protocol 4). A silent agent still fails the ping."""
    mode(tmp_path, "slow")
    t = make(fake, tmp_path)
    assert await t.prepare() is None
    s = t._session
    assert s is not None and s.ready is not None
    big = framing.make_script([Section("b.big", "head -c 100000 /dev/zero | tr '\\0' x; echo", timeout=60)])
    fetch = asyncio.ensure_future(t.probe(big, timeout=30))
    await asyncio.sleep(0.6)
    assert not fetch.done()
    assert await s.ping(0.3) is True                      # the pong itself is about 2 s behind the output
    res = await fetch
    assert res.error is None and len(res.sections["b.big"].stdout) == 100001
    os.kill(s.ready.pid, signal.SIGSTOP)                   # the agent stops answering; nothing arrives
    try:
        await asyncio.sleep(0.4)
        assert await s.ping(0.3) is False
    finally:
        os.kill(s.ready.pid, signal.SIGCONT)
    await t.close()


def _spy_sessions(monkeypatch: pytest.MonkeyPatch) -> list[session.AgentSession]:
    made: list[session.AgentSession] = []
    real = session.AgentSession.__init__

    def spy(self: session.AgentSession, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(session.AgentSession, "__init__", spy)
    return made


async def test_a_cancelled_start_ends_its_ssh(fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The poller is cancelled (host removed or disabled) while a session starts: its ssh must not stay logged in
    until RunFleet exits (review: protocol 6)."""
    made = _spy_sessions(monkeypatch)
    t = make(fake, tmp_path)
    assert await t.resolve() is None
    task = asyncio.ensure_future(t.prepare())
    try:
        await until(lambda: made, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await t.close()
        await until(lambda: made[0].proc.poll() is not None, 5)
        await until(lambda: not list((tmp_path / "remote-tmp").iterdir()), 5)
    finally:
        for s in made:
            if s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait(5)


async def test_close_during_a_start_does_not_install_it(fake: Path, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """`close()` while a start is in flight (an on-demand probe during remove_host): the new session used to be
    installed on the closed transport (review: protocol 6)."""
    made = _spy_sessions(monkeypatch)
    t = make(fake, tmp_path)
    assert await t.resolve() is None
    task = asyncio.ensure_future(t.prepare())
    try:
        await until(lambda: made, 5)
        await t.close()
        err = await task
        assert err is not None and err.kind is ErrorKind.MUX and t._session is None
        await until(lambda: made[0].proc.poll() is not None, 5)
    finally:
        for s in made:
            if s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait(5)


async def test_an_aborted_login_gets_sigterm_before_sigkill(fake: Path, tmp_path: Path) -> None:
    """Ctrl-C during the login: ssh may still be at its password prompt with echo off. OpenSSH's readpassphrase
    restores the terminal on SIGTERM, never on SIGKILL (review: integration 3)."""
    mode(tmp_path, "prompt")
    t = make(fake, tmp_path)
    assert await t.resolve() is None
    s = session.AgentSession(t.session_argv(batch=False), env=dict(os.environ), interactive=True, label="k2")
    try:
        await until(lambda: (tmp_path / "prompting").exists(), 5)
        await asyncio.to_thread(s.kill_now)
        assert (tmp_path / "restored").exists() and s.proc.poll() is not None
    finally:
        if s.proc.poll() is None:
            s.proc.kill()
            s.proc.wait(5)
