"""[WP1] The session agent (§4.8): the pure codec, and the REAL agent text run under sh / dash / bash / BusyBox ash.

Each live test starts the agent exactly as the session ssh does (`<shell> -c BOOTSTRAP`, the agent's bytes on stdin)
with TMPDIR pointing into the test's tmp_path, so its private temp dir and every leftover can be checked. Commands
carry unique `sleep` durations (`sleep 31.4`) to find leftover processes with `ps`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from runfleet.config import HostConfig
from runfleet.probe import framing
from runfleet.probe.section import Section
from runfleet.transport import agent
from runfleet.transport.local import LocalTransport

pytestmark = pytest.mark.posix          # runs POSIX shells, ps, signals

SHELLS = [s for s in (["sh"], ["dash"], ["bash"], ["busybox", "sh"]) if shutil.which(s[0])
          and (s[0] != "busybox" or subprocess.run(["busybox", "sh", "-c", "true"], capture_output=True).returncode == 0)]


# ---------------------------------------------------------------- pure codec
def test_agent_text_is_ascii_and_hello_counts_bytes() -> None:
    raw = agent.hello()
    n, _, body = raw.partition(b"\n")
    assert int(n) == len(body) == len(agent.AGENT_SH.encode("ascii"))
    assert f"R %s {agent.VERSION} " in agent.AGENT_SH            # the ready line carries the client's version


def test_remote_command_quoting() -> None:
    assert agent.remote_command("sh") == "exec sh -c 'IFS= read -r n; eval \"$(dd bs=1 count=$n 2>/dev/null)\"'"
    assert agent.remote_command("login").startswith("exec bash -l -c 'IFS= read -r n;")


def test_payload_escaping_rules() -> None:
    assert agent.encode_payload(b"echo a%b\\c d\n") == "echo\\040a%%b\\134c\\040d\\012"
    assert agent.encode_payload(b"-n x") == "\\055n\\040x"              # never an option to printf
    assert agent.encode_payload(b"\x00\xff\xc3\xa9") == "\\000\\377\\303\\251"
    assert agent.encode_payload(b"") == ""
    assert " " not in agent.encode_payload(bytes(range(256)))           # one word on the request line


def test_requests_reject_bad_ids() -> None:
    assert agent.ping("p1") == b"P p1\n" and agent.kill("j7") == b"K j7\n" and agent.close_stdin("j7") == b"W j7\n"
    assert agent.probe_request("j1", b"true\n") == b"Q j1 true\\012\n"
    assert agent.stream_request("j2", "tail -F x") == b"S j2 tail\\040-F\\040x\n"
    for bad in ("", "a b", "../x", "j-1", "é"):
        with pytest.raises(ValueError):
            agent.ping(bad)


def test_decoder_across_chunk_boundaries() -> None:
    d = agent.ReplyDecoder()
    stream = (b"Welcome to k2\nlast login: x\nL j1 early\nR 4242 1 setsid 4096\nL j1 hello\nC j1 abc\nL j1 def\n"
              b"D j2 0d70726f\nE j1 oops\nF j1 tail\nZ j1 3\nP p9\nZ j2 -1\nQ j1 bogus\nL bad/id x\nD j2 zz\n")
    out: list[agent.Reply] = []
    for i in range(len(stream)):                  # one byte at a time: every split point
        out += d.feed(stream[i:i + 1])
    assert out[0] == agent.Noise(b"Welcome to k2") and out[1] == agent.Noise(b"last login: x")
    assert out[2] == agent.Noise(b"L j1 early")                        # nothing counts before `R`
    assert out[3] == agent.Ready(4242, 1, True, 4096) and d.ready
    assert out[4:13] == [agent.Output("j1", b"hello\n"), agent.Output("j1", b"abc"), agent.Output("j1", b"def\n"),
                         agent.Output("j2", b"\rpro"), agent.Output("j1", b"oops\n", stderr=True),
                         agent.Output("j1", b"tail", stderr=True), agent.Done("j1", 3), agent.Pong("p9"),
                         agent.Done("j2", -1)]
    assert all(isinstance(r, agent.Noise) for r in out[13:]) and len(out) == 16
    assert d.feed(b"L j1 no newline") == [] and d.eof() == [agent.Noise(b"L j1 no newline")]


def test_decoder_fatal_no_setsid_and_runaway_lines() -> None:
    d = agent.ReplyDecoder()
    assert d.feed(b"X cannot create a private temp dir\n") == [agent.Fatal("cannot create a private temp dir")]
    assert d.feed(b"R 1 1 - 512\n") == [agent.Ready(1, 1, False, 512)]
    big = d.feed(b"x" * (agent.LINE_LIMIT + 1))
    assert len(big) == 1 and isinstance(big[0], agent.Noise) and d.feed(b"P a\n") == [agent.Pong("a")]


# ---------------------------------------------------------------- the real agent
class Agent:
    """The agent under `shell`, driven like the session transport drives it (one stdout reader)."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self.dec = agent.ReplyDecoder()
        self.out: dict[str, bytearray] = {}
        self.err: dict[str, bytearray] = {}
        self.done: dict[str, int] = {}
        self.pongs: list[str] = []
        self.ready: agent.Ready | None = None
        self.raw_lines: list[bytes] = []
        self.events: list[tuple[float, agent.Reply]] = []
        self._task = asyncio.ensure_future(self._read())

    async def _read(self) -> None:
        assert self.proc.stdout is not None
        buf = b""
        while data := await self.proc.stdout.read(65536):
            buf += data
            *lines, buf = buf.split(b"\n")
            self.raw_lines += lines
            for r in self.dec.feed(data):
                self.events.append((time.monotonic(), r))
                if isinstance(r, agent.Ready):
                    self.ready = r
                elif isinstance(r, agent.Output):
                    (self.err if r.stderr else self.out).setdefault(r.job, bytearray()).extend(r.data)
                elif isinstance(r, agent.Done):
                    self.done[r.job] = r.rc
                elif isinstance(r, agent.Pong):
                    self.pongs.append(r.token)

    async def send(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def until(self, cond: object, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        while not cond():                                   # type: ignore[operator]
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out; done={self.done} out={ {k: bytes(v[:80]) for k, v in self.out.items()} }")
            await asyncio.sleep(0.02)

    async def wait_done(self, job: str, timeout: float = 10) -> int:
        await self.until(lambda: job in self.done, timeout)
        return self.done[job]

    async def eof(self, timeout: float = 10) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        await asyncio.wait_for(self._task, timeout)
        return await asyncio.wait_for(self.proc.wait(), timeout)


@pytest.fixture(params=SHELLS, ids=[" ".join(s) for s in SHELLS])
async def live(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Agent]:
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(remote_tmp))
    proc = await asyncio.create_subprocess_exec(*request.param, "-c", agent.BOOTSTRAP, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, limit=1 << 22)
    a = Agent(proc)
    await a.send(agent.hello())
    await a.until(lambda: a.ready is not None, 10)
    yield a
    if proc.returncode is None:
        proc.kill()
        await proc.wait()


def leftovers(marker: str) -> list[str]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,args="], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if marker in line and "ps -A" not in line]


async def gone(marker: str, timeout: float = 5) -> list[str]:
    deadline = time.monotonic() + timeout
    while (left := leftovers(marker)) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    return left


async def test_ready_and_ping(live: Agent, tmp_path: Path) -> None:
    assert live.ready is not None and live.ready.version == agent.VERSION and live.ready.pipe_buf >= 512
    dirs = list((tmp_path / "remote-tmp").iterdir())
    assert len(dirs) == 1 and dirs[0].name.startswith("rfa.") and dirs[0].stat().st_mode & 0o077 == 0
    await live.send(agent.ping("hello1"))
    await live.until(lambda: "hello1" in live.pongs, 5)


async def test_probes_run_concurrently_up_to_the_cap(live: Agent) -> None:
    t0 = time.monotonic()
    for i in range(agent.MAX_JOBS):
        await live.send(agent.probe_request(f"c{i}", f"sleep 1; echo job{i}; exit {i}".encode()))
    await live.send(agent.probe_request("over", b"echo never"))       # a 9th while 8 run: refused at once
    assert await live.wait_done("over", 5) == 125 and b"busy" in live.err["over"]
    for i in range(agent.MAX_JOBS):
        assert await live.wait_done(f"c{i}", 10) == i
        assert bytes(live.out[f"c{i}"]) == f"job{i}\n".encode()
    assert time.monotonic() - t0 < 4                                     # in parallel, not 8 s in a row
    await live.send(agent.probe_request("after", b"echo free again"))
    assert await live.wait_done("after") == 0 and bytes(live.out["after"]) == b"free again\n"


async def test_long_line_is_byte_exact_and_every_write_fits_pipe_buf(live: Agent) -> None:
    assert live.ready is not None
    script = b"printf '%s\\n' \"$(printf '%5000s' '' | tr ' ' x)\"; printf 'a\\tb\\r\\n\\033[1m\\001'; echo"
    await live.send(agent.probe_request("big", script))
    assert await live.wait_done("big") == 0
    assert bytes(live.out["big"]) == b"x" * 5000 + b"\na\tb\r\n\x1b[1m\x01\n"
    assert max(len(line) + 1 for line in live.raw_lines) < min(4096, live.ready.pipe_buf)


async def test_stream_carriage_returns_arrive_while_running(live: Agent) -> None:
    cmd = ("(i=0; while :; do i=$((i+1)); printf '\\rprogress %s' $i; sleep 0.2; done) & k=$!; "
           "cat >/dev/null; kill $k")
    await live.send(agent.stream_request("s1", cmd))
    await live.until(lambda: b"\rprogress 3" in live.out.get("s1", b""), 10)
    assert "s1" not in live.done and b"\n" not in live.out["s1"]       # partial lines, no newline added
    await live.send(agent.close_stdin("s1"))                             # stdin EOF: its watchdog ends it
    assert await live.wait_done("s1", 5) == 0


async def test_kill_ends_only_that_job(live: Agent) -> None:
    await live.send(agent.probe_request("slow", b"sleep 31.4; echo never"))
    await live.send(agent.stream_request("tail", "sleep 32.6 & w=$!; cat >/dev/null; kill $w"))
    await live.until(lambda: leftovers("sleep 31.4") and leftovers("sleep 32.6"), 5)
    t0 = time.monotonic()
    await live.send(agent.kill("slow"))
    rc = await live.wait_done("slow", 5)
    assert rc in (143, 128 + 15, 137) and time.monotonic() - t0 < 4
    assert await gone("sleep 31.4") == []
    assert "tail" not in live.done and leftovers("sleep 32.6")          # the other job runs on
    await live.send(agent.ping("still"))
    await live.until(lambda: "still" in live.pongs, 5)
    await live.send(agent.close_stdin("tail"))
    assert await live.wait_done("tail", 5) == 0
    assert await gone("sleep 32.6") == []


async def test_stderr_and_exit_status_come_back(live: Agent) -> None:
    await live.send(agent.probe_request("e", b"echo out; echo err1 >&2; printf 'no-nl' >&2; exit 7"))
    assert await live.wait_done("e") == 7
    assert bytes(live.out["e"]) == b"out\n" and bytes(live.err["e"]) == b"err1\nno-nl\n"
    await live.send(agent.stream_request("bad", "tail -n 1 -- /nonexistent/rf-agent-test"))
    assert await live.wait_done("bad") != 0 and b"nonexistent" in live.err["bad"]


async def test_stdin_eof_ends_every_job_and_removes_the_temp_dir(live: Agent, tmp_path: Path) -> None:
    await live.send(agent.probe_request("p", b"sleep 33.7"))
    await live.send(agent.stream_request("s", "sleep 34.1"))             # ignores its stdin: needs the kill
    await live.send(agent.probe_request("q", b"trap '' TERM; sleep 35.3"))   # ignores SIGTERM: needs SIGKILL
    await live.until(lambda: all(leftovers(m) for m in ("sleep 33.7", "sleep 34.1", "sleep 35.3")), 5)
    t0 = time.monotonic()
    assert await live.eof(10) == 0
    assert time.monotonic() - t0 < 5
    for m in ("sleep 33.7", "sleep 34.1", "sleep 35.3"):
        assert await gone(m) == [], m
    assert list((tmp_path / "remote-tmp").iterdir()) == []
    assert await gone("rfa_hold") == [] and await gone(str(tmp_path)) == []


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
async def test_signal_to_the_agent_cleans_up(live: Agent, tmp_path: Path, sig: signal.Signals) -> None:
    assert live.ready is not None
    await live.send(agent.probe_request("p", b"sleep 36.2"))
    await live.until(lambda: leftovers("sleep 36.2"), 5)
    os.kill(live.ready.pid, sig)
    assert await asyncio.wait_for(live.proc.wait(), 10) in (129, 143, -sig)
    assert await gone("sleep 36.2") == []
    assert list((tmp_path / "remote-tmp").iterdir()) == []


async def test_framing_is_byte_exact_through_the_agent(live: Agent) -> None:
    """A rendered ProbeScript through the agent parses to exactly what the same script gives under plain sh."""
    sections = [Section("t.empty", "true"), Section("t.nonl", "printf abc"), Section("t.blank", "printf 'abc\\n\\n'"),
                Section("t.binary", "printf '\\001\\377\\033[31mred\\r\\n'"), Section("t.err", "echo o; echo e >&2"),
                Section("t.rc", "echo x; exit 3"), Section("t.missing", "x", requires=("nosuchtool_rf",)),
                Section("t.stdin", "cat; echo after-cat"), Section("t.big", "printf '%7000s' '' | tr ' ' y"),
                Section("t.late", "echo late", late=True)]
    s = framing.make_script(sections)
    await live.send(agent.probe_request("fr", framing.render(s).encode()))
    assert await live.wait_done("fr") == 0
    via_agent = framing.parse(bytes(live.out["fr"]), s).sections
    local = (await LocalTransport(HostConfig("atlas", "local")).probe(s, timeout=20)).sections
    for name in local:
        a, b = via_agent[name], local[name]
        assert (a.outcome, a.rc, a.stdout, a.stderr) == (b.outcome, b.rc, b.stdout, b.stderr), name


async def test_payload_bytes_survive_printf(live: Agent) -> None:
    every = bytes(b for b in range(1, 256))                              # NUL cannot be part of a shell script
    script = b"cat <<'RF_EOF' | od -A n -t x1 -v | tr -d ' \\n'\n" + every.replace(b"\n", b"") + b"\nRF_EOF\n"
    await live.send(agent.probe_request("bytes", script))
    assert await live.wait_done("bytes") == 0
    assert bytes.fromhex(bytes(live.out["bytes"]).decode().strip()) == every.replace(b"\n", b"") + b"\n"


@pytest.mark.skipif(not shutil.which("setsid") or not shutil.which("ps"), reason="needs setsid and ps to hide")
async def test_kill_without_setsid_reaches_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote without util-linux `setsid` (macOS, BusyBox without it): the agent kills the job's descendants
    found with `ps` instead of a process group."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("sh", "dd", "awk", "od", "tr", "tail", "sleep", "mktemp", "cat", "rm", "ps", "basename",
                 "getconf", "date", "mkdir", "echo", "printf"):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(remote_tmp))
    proc = await asyncio.create_subprocess_exec(str(bindir / "sh"), "-c", agent.BOOTSTRAP, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, env={**os.environ, "PATH": str(bindir)})
    a = Agent(proc)
    await a.send(agent.hello())
    await a.until(lambda: a.ready is not None, 10)
    assert a.ready is not None and a.ready.setsid is False
    await a.send(agent.probe_request("slow", b"(sleep 37.9; echo never) & wait"))
    await a.until(lambda: leftovers("sleep 37.9"), 5)
    await a.send(agent.kill("slow"))
    await a.wait_done("slow", 5)
    assert await gone("sleep 37.9") == []
    assert await a.eof() == 0 and list(remote_tmp.iterdir()) == []


# ---------------------------------------------------------------- review regressions (P2)
def _awk_variants() -> list[tuple[str, str]]:
    """(id, how to run it) for every awk on this machine: the agent must behave the same under each."""
    out, seen = [], set()
    for name in ("mawk", "gawk", "original-awk", "nawk"):
        path = shutil.which(name)
        if path and os.path.realpath(path) not in seen:
            seen.add(os.path.realpath(path))
            out.append((name, path))
    if shutil.which("busybox") and subprocess.run(["busybox", "awk", "BEGIN{}"], capture_output=True).returncode == 0:
        out.append(("busybox", "busybox"))
    return out


AWKS = _awk_variants()


async def start_agent(tmp_path: Path, argv: list[str], *, awk: str | None = None,
                      prefix: str = "") -> Agent:
    """The agent under `argv` (a shell), TMPDIR in tmp_path; `awk`: the awk it finds first on PATH; `prefix`: shell
    text run before the bootstrap (what a login shell prints)."""
    env = dict(os.environ)
    if awk is not None:
        bindir = tmp_path / "awkbin"
        bindir.mkdir()
        wrapper = bindir / "awk"
        wrapper.write_text(f'#!/bin/sh\nexec {"busybox awk" if awk == "busybox" else awk} "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(remote_tmp)
    proc = await asyncio.create_subprocess_exec(*argv, "-c", prefix + agent.BOOTSTRAP, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, env=env, limit=1 << 22)
    a = Agent(proc)
    await a.send(agent.hello())
    await a.until(lambda: a.ready is not None, 10)
    return a


async def stop_agent(a: Agent) -> None:
    if a.proc.returncode is None:
        with contextlib.suppress(Exception):
            await a.eof(10)
    if a.proc.returncode is None:
        a.proc.kill()
        await a.proc.wait()


@pytest.mark.skipif(not AWKS, reason="no awk to choose")
@pytest.mark.parametrize("awk", [p for _, p in AWKS], ids=[n for n, _ in AWKS])
async def test_probe_lines_arrive_while_the_job_still_runs(tmp_path: Path, awk: str) -> None:
    """mawk reads a pipe in blocks until EOF: without `-W interactive` a probe's finished sections only arrived once
    the job ended, so a probe that then timed out lost all of them (review: protocol 1)."""
    a = await start_agent(tmp_path, ["sh"], awk=awk)
    try:
        await a.send(agent.probe_request("early", b"echo first; sleep 4.31; echo never"))
        await a.until(lambda: bytes(a.out.get("early", b"")) == b"first\n", 2.5)
        assert "early" not in a.done
        await a.send(agent.kill("early"))
        assert await a.wait_done("early", 5) != 0
    finally:
        await stop_agent(a)


@pytest.mark.skipif(not AWKS or not hasattr(socket, "SOCK_SEQPACKET"), reason="needs an awk and SOCK_SEQPACKET")
@pytest.mark.parametrize("awk", [p for _, p in AWKS], ids=[n for n, _ in AWKS])
def test_every_agent_write_is_whole_lines(tmp_path: Path, awk: str) -> None:
    """A SOCK_SEQPACKET stdout keeps write(2) boundaries: every write the agent makes must be complete lines, or
    lines of concurrent jobs could interleave (mawk's unbuffered mode writes each printf conversion on its own)."""
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    bindir = tmp_path / "awkbin"
    bindir.mkdir()
    (bindir / "awk").write_text(f'#!/bin/sh\nexec {"busybox awk" if awk == "busybox" else awk} "$@"\n',
                                encoding="utf-8")
    (bindir / "awk").chmod(0o755)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    proc = subprocess.Popen(["sh", "-c", agent.BOOTSTRAP], stdin=subprocess.PIPE, stdout=theirs, env=env)
    theirs.close()
    assert proc.stdin is not None
    script = b"echo hello; printf '%5000s\\n' '' | tr ' ' x; echo bye >&2"
    proc.stdin.write(agent.hello() + agent.probe_request("w1", script))
    proc.stdin.flush()
    writes: list[bytes] = []
    ours.settimeout(10)
    try:
        while not any(w.startswith(b"Z w1 ") for w in writes):
            writes.append(ours.recv(1 << 16))
    finally:
        proc.stdin.close()
        proc.wait(10)
        ours.close()
    assert writes and all(w.endswith(b"\n") for w in writes), writes[:8]
    lines = [line for w in writes for line in w.split(b"\n")[:-1]]
    assert b"L w1 hello" in lines and b"E w1 bye" in lines


async def test_login_output_without_a_newline_does_not_hide_ready(tmp_path: Path) -> None:
    """A `.bashrc` that sets the terminal title (no newline) used to glue onto `R`, so the session never became
    ready (review: protocol 3)."""
    a = await start_agent(tmp_path, ["sh"], prefix="printf '\\033]0;k2\\007'; ")
    try:
        assert a.ready is not None
        await a.send(agent.ping("after"))
        await a.until(lambda: "after" in a.pongs, 5)
    finally:
        await stop_agent(a)


def _ancestors(pid: int) -> list[int]:
    table = {}
    for line in subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = int(parts[1])
    chain = []
    while pid in table and pid not in chain and pid > 1:
        chain.append(pid)
        pid = table[pid]
    return chain


@pytest.mark.skipif(not shutil.which("ps"), reason="needs ps")
async def test_jobs_are_not_children_of_the_agent_shell(live: Agent) -> None:
    """The agent never `wait`s, so a job started with a plain `&` stayed in its shell's job table for good (bash:
    memory and per-job latency grew with every probe; review: protocol 5). Jobs run detached from it instead."""
    assert live.ready is not None
    await live.send(agent.probe_request("d", b"sleep 43.7"))
    await live.until(lambda: leftovers("sleep 43.7"), 5)
    pid = int(leftovers("sleep 43.7")[0].split()[0])
    assert live.ready.pid not in _ancestors(pid)
    await live.send(agent.kill("d"))
    await live.wait_done("d", 5)


@pytest.mark.skipif(not shutil.which("setsid") or not shutil.which("ps"), reason="needs setsid and ps")
async def test_a_job_that_cannot_be_killed_frees_its_slot(tmp_path: Path) -> None:
    """A killed job whose stdout stays open (a child in D state, or one that escaped) never sends `Z`; it used to
    count toward MAX_JOBS forever, so after eight of them every request got `Z 125` (review: protocol 2)."""
    a = await start_agent(tmp_path, ["sh"])
    try:
        stuck = b"(setsid sh -c 'exec sleep 44.9' &); echo started; sleep 44.8"
        for i in range(agent.MAX_JOBS):
            await a.send(agent.probe_request(f"s{i}", stuck))
        await a.until(lambda: all(bytes(a.out.get(f"s{i}", b"")) == b"started\n" for i in range(agent.MAX_JOBS)), 10)
        await a.send(agent.probe_request("full", b"echo no room"))
        assert await a.wait_done("full", 5) == 125
        for i in range(agent.MAX_JOBS):
            await a.send(agent.kill(f"s{i}"))
        await asyncio.sleep(3.6)                     # SIGTERM, SIGKILL 2 s later, then 1 s for the output to close
        assert not any(f"s{i}" in a.done for i in range(agent.MAX_JOBS))    # still held open: no `Z`
        await a.send(agent.probe_request("room", b"echo room again"))
        assert await a.wait_done("room", 5) == 0 and bytes(a.out["room"]) == b"room again\n"
    finally:
        for line in leftovers("sleep 44.9"):
            with contextlib.suppress(OSError):
                os.kill(int(line.split()[0]), signal.SIGKILL)
        await stop_agent(a)
        assert await gone("sleep 44.") == []
