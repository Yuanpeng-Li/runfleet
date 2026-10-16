"""[WP1] util/platform.py and the Windows branches it feeds (DESIGN §1.2, §4.1, §8.1), tested on any OS.

Windows code paths run on Linux by patching ``platform.IS_WINDOWS`` (and, for locks, a fake ``msvcrt`` in
``sys.modules``); nothing here starts a real ssh, and the POSIX-only checks are marked ``posix``.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from runfleet import paths
from runfleet.config import HostConfig
from runfleet.models import ErrorKind
from runfleet.probe import framing
from runfleet.probe.section import Section, SectionOutcome
from runfleet.sshconfig import parse_ssh_g
from runfleet.transport import mux, spawn
from runfleet.transport.local import LocalTransport
from runfleet.transport.openssh import OpenSSHTransport
from runfleet.util import platform

G_K2 = "user alice0001\nhostname k2.example.edu\nport 22\ncontrolmaster auto\n" \
       "controlpath /home/usr/.ssh/cm-0123\ncontrolpersist 600\n"
G_LYRA = "user alice07\nhostname lyra.example.edu\nport 22\ncontrolmaster false\ncontrolpersist no\n" \
         "proxyjump jumpvm\n"
G_MULTI = "user u\nhostname t\nport 22\ncontrolpersist no\nproxyjump usr@jump1:2222,ssh://bob@jump2:2200\n"
SSH_EXE = r"C:\Windows\System32\OpenSSH\ssh.exe"


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the Windows branches (only what reads platform.IS_WINDOWS at call time changes)."""
    monkeypatch.setattr(platform, "IS_WINDOWS", True)


class FakeMsvcrt(types.ModuleType):
    """`msvcrt.locking` over real files: byte-range locks per (file, offset), held per descriptor, so two
    descriptors of one file contend like two Windows processes. Records every call."""

    def __init__(self) -> None:
        super().__init__("msvcrt")
        self.held: dict[tuple[int, int, int], int] = {}
        self.calls: list[tuple[int, int, int, int]] = []        # (fd, mode, nbytes, position)

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        pos = os.lseek(fd, 0, os.SEEK_CUR)
        st = os.fstat(fd)
        key = (st.st_dev, st.st_ino, pos)
        self.calls.append((fd, mode, nbytes, pos))
        if mode == 0:                                           # LK_UNLCK
            if self.held.get(key) != fd:
                raise PermissionError(errno.EACCES, "not locked")
            del self.held[key]
            return
        owner = self.held.get(key)
        if owner is not None and owner != fd:
            raise PermissionError(errno.EACCES, "Permission denied")
        self.held[key] = fd


@pytest.fixture
def msvcrt(monkeypatch: pytest.MonkeyPatch, windows: None) -> FakeMsvcrt:
    fake = FakeMsvcrt()
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    return fake


# ---------------------------------------------------------------- spawn keywords, kill
def test_background_spawn_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert platform.background_spawn_kwargs() == {"start_new_session": True}
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    flags = platform.background_spawn_kwargs()["creationflags"]
    assert flags == 0x08000000 | 0x00000200                     # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    assert set(platform.background_spawn_kwargs()) == {"creationflags"}


class FakeProc:
    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        self.pid, self.returncode = pid, returncode
        self.events: list[str] = []

    def send_signal(self, sig: int) -> None:
        self.events.append(f"signal {sig}")

    def kill(self) -> None:
        self.events.append("kill")


def test_kill_tree_windows_runs_taskkill_before_terminating(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(platform, "taskkill_tree", lambda pid: order.append(f"taskkill {pid}") or 0)
    proc = FakeProc()
    proc.kill = lambda: order.append("kill")                    # type: ignore[method-assign]
    platform.kill_tree(proc, hard=False)                        # no soft kill on Windows: the tree ends at once
    assert order == ["taskkill 4242", "kill"]
    order.clear()
    platform.kill_tree(FakeProc(returncode=0), hard=True)       # already gone: nothing
    assert order == []

    def gone() -> None:
        raise ProcessLookupError

    proc2 = FakeProc()
    proc2.kill = gone                                           # type: ignore[method-assign]
    platform.kill_tree(proc2, hard=True)                        # never raises for a vanished process
    platform.kill_group(77)
    assert order == ["taskkill 4242", "taskkill 77"]


def test_taskkill_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        assert kw["timeout"] <= 5 and kw["stdin"] is subprocess.DEVNULL
        return subprocess.CompletedProcess(argv, 128)

    monkeypatch.setattr(subprocess, "run", run)
    assert platform.taskkill_tree(31) == 128
    assert seen == [["taskkill", "/F", "/T", "/PID", "31"]]

    def missing(argv: list[str], **kw: Any) -> None:
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(subprocess, "run", missing)
    assert platform.taskkill_tree(31) is None


@pytest.mark.posix
def test_kill_tree_posix_signals_the_group() -> None:
    proc = subprocess.Popen(["sh", "-c", "sleep 30 & wait"], **platform.background_spawn_kwargs())
    try:
        time.sleep(0.2)
        platform.kill_tree(proc, hard=False)                   # SIGTERM to the group: sh and its sleep
        assert proc.wait(timeout=10) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    platform.kill_tree(proc, hard=True)                         # exited: a no-op


def test_kill_signal_and_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    assert platform.KILL_SIGNAL == getattr(signal, "SIGKILL", signal.SIGTERM)
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.current_uid() is None and platform.is_root() is False


@pytest.mark.posix
def test_uid_on_posix() -> None:
    assert platform.current_uid() == os.getuid()


# ---------------------------------------------------------------- locks
@pytest.mark.posix
def test_posix_lock_contends_between_descriptors(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    a = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    b = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert platform.try_lock(a) is True
        assert platform.try_lock(b) is False
        platform.unlock(a)
        assert platform.try_lock(b) is True
    finally:
        os.close(a)
        os.close(b)


def test_windows_lock_uses_a_sentinel_byte_and_keeps_the_pid_readable(tmp_path: Path, msvcrt: FakeMsvcrt) -> None:
    path = tmp_path / "lock"
    a = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    b = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert platform.try_lock(a) is True
        assert msvcrt.calls == [(a, 2, 1, platform.LOCK_OFFSET)]          # LK_NBLCK on one byte at 1 MiB
        assert os.lseek(a, 0, os.SEEK_CUR) == 0                            # back at the start for the pid
        os.write(a, b"4242")
        assert platform.try_lock(b) is False                               # held by the other descriptor
        assert os.lseek(b, 0, os.SEEK_CUR) == 0 and os.read(b, 32) == b"4242"
        platform.unlock(a)
        assert msvcrt.calls[-1] == (a, 0, 1, platform.LOCK_OFFSET)         # LK_UNLCK of the same byte
        assert platform.try_lock(b) is True
        platform.unlock(b)
        platform.unlock(b)                                                 # not held any more: never raises
    finally:
        os.close(a)
        os.close(b)


def test_windows_lock_other_errors_raise(tmp_path: Path, msvcrt: FakeMsvcrt) -> None:
    def broken(fd: int, mode: int, nbytes: int) -> None:
        raise OSError(errno.EBADF, "bad file descriptor")

    msvcrt.locking = broken                                                # type: ignore[method-assign]
    fd = os.open(tmp_path / "lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(OSError):
            platform.try_lock(fd)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
    finally:
        os.close(fd)


async def test_instance_lock_on_windows_names_the_other_pid(tmp_path: Path, msvcrt: FakeMsvcrt) -> None:
    from runfleet.config import FleetConfig
    from runfleet.core.clock import ManualClock
    from runfleet.core.service import FleetService
    from runfleet.transport.fake import FakeTransport
    from tests.conftest import CLOCK_START

    clock = ManualClock(CLOCK_START)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    a = FleetService(cfg, lambda h: FakeTransport(h.name), clock, None)
    b = FleetService(cfg, lambda h: FakeTransport(h.name), clock, None)
    a.lock_path = b.lock_path = tmp_path / "instance.lock"
    await a.start()
    await b.start()
    try:
        assert not a.notices
        assert b.notices == [f"another RunFleet (pid {os.getpid()}) is polling k2 — remote load doubles there"]
        assert msvcrt.held                                                 # the msvcrt path took the lock
    finally:
        await a.stop()
        await b.stop()


def test_config_edit_lock_on_windows(tmp_path: Path, msvcrt: FakeMsvcrt, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet import configedit
    from runfleet.errors import ConfigError

    path = tmp_path / "config.toml"
    path.write_text("[general]\n", encoding="utf-8")
    editor = configedit.ConfigFile(path)
    real_unlink = os.unlink

    def no_unlink_while_open(p: Any, *a: Any, **kw: Any) -> None:
        if str(p).endswith(".lock"):
            raise PermissionError(errno.EACCES, "in use")                   # what Windows says for an open file
        real_unlink(p, *a, **kw)

    monkeypatch.setattr(os, "unlink", no_unlink_while_open)
    with editor._lock():
        other = os.open(tmp_path / ".config.toml.lock", os.O_RDWR)
        try:
            assert platform.try_lock(other) is False                        # a second RunFleet would wait
        finally:
            os.close(other)
    assert (tmp_path / ".config.toml.lock").exists()                        # left behind on Windows: harmless
    assert not msvcrt.held                                                  # but released
    monkeypatch.setattr(configedit, "LOCK_TIMEOUT", 0.05)
    with editor._lock():                                                    # the stale file is reused
        pass
    holder = os.open(tmp_path / ".config.toml.lock", os.O_RDWR)
    try:
        assert platform.try_lock(holder)
        with pytest.raises(ConfigError, match="another RunFleet is saving it"):
            with editor._lock():
                pass
    finally:
        os.close(holder)


# ---------------------------------------------------------------- command lines, names, signals
def test_split_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert platform.split_command("less -R '+G x'") == ["less", "-R", "+G x"]
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.split_command(r'"C:\Program Files\Notepad++\notepad++.exe" -multiInst') == [
        r"C:\Program Files\Notepad++\notepad++.exe", "-multiInst"]
    assert platform.split_command(r"C:\tools\less.exe -R +G") == [r"C:\tools\less.exe", "-R", "+G"]
    assert platform.split_command(r"IdentityFile C:\Users\me\.ssh\id  # mine", comments=True) == [
        "IdentityFile", r"C:\Users\me\.ssh\id"]
    with pytest.raises(ValueError):
        platform.split_command('"C:\\unterminated')


def test_resolve_program(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    found = {"more": r"C:\Windows\System32\more.com", "code": r"C:\Users\u\AppData\Local\Programs\VS Code\bin\code.cmd"}
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: found.get(name))
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert platform.resolve_program(["more", "x.log"]) == ["more", "x.log"]               # POSIX: execvp's own PATH
    monkeypatch.setattr(platform, "IS_WINDOWS", True)                       # CreateProcess appends only .exe
    assert platform.resolve_program(["more", "x.log"]) == [found["more"], "x.log"]
    assert platform.resolve_program(("code", "--wait", "c.toml")) == [found["code"], "--wait", "c.toml"]
    assert platform.resolve_program(["vim", "c.toml"]) == ["vim", "c.toml"]               # not found: start reports it
    assert platform.resolve_program([r"C:\x\more", "y"]) == [r"C:\x\more", "y"]          # a path is the user's
    assert platform.resolve_program(["./more"]) == ["./more"]
    assert platform.resolve_program([]) == []


def test_display_argv_and_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    argv = ["uv", "tool", "install", "/a b/c"]
    assert platform.display_argv(argv) == "uv tool install '/a b/c'"
    assert platform.chain(["git pull", "uv tool install ."]) == "git pull && uv tool install ."
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.display_argv(["uv", "tool", "install", r"C:\My Tools\rf"]) == r'uv tool install "C:\My Tools\rf"'
    assert platform.chain(["git pull", "uv tool install ."]) == "git pull; uv tool install ."


@pytest.mark.parametrize(("name", "reserved"), [
    ("con", True), ("CON", True), ("nul", True), ("aux", True), ("prn", True), ("com1", True), ("LPT9", True),
    ("nul.log", True), ("web.", True), ("k2", False), ("console", False), ("com10", False), ("my.con", False),
])
def test_windows_reserved_names(name: str, reserved: bool) -> None:
    assert platform.windows_reserved_name(name) is reserved


def test_safe_file_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert platform.safe_file_name("con") == "con"                         # only Windows refuses it
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.safe_file_name("con") == "con_" and platform.safe_file_name("web..") == "web_"
    assert platform.safe_file_name("k2") == "k2"


@pytest.mark.parametrize(("text", "ok"), [
    ("OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2", True), ("OpenSSH_for_Windows_8.1p1, LibreSSL 3.0.2", True),
    ("OpenSSH_for_Windows_7.7p1, LibreSSL 2.6.5", False), ("OpenSSH_9.6p1, OpenSSL 3.2.1 30 Jan 2024", None),
    ("", None),
])
def test_ssh_version_floor(text: str, ok: bool | None) -> None:
    assert platform.ssh_version_ok(text) is ok


def test_linux_signal_names_whatever_the_local_os(monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.backends.slurm.states import signal_name
    from runfleet.util import logscan

    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert platform.linux_signal_name(9) == "SIGKILL" and platform.linux_signal_name(10) == "SIGUSR1"
    assert platform.linux_signal_name(35) == "signal 35"
    assert signal_name(15) == "SIGTERM" and signal_name(29) == "SIGIO" and signal_name(99) == "signal 99"
    assert logscan._signal_name(7) == "SIGBUS"
    assert logscan._by_signal(11).code == "SEGV"


@pytest.mark.posix
def test_linux_signal_table_matches_python_on_linux() -> None:
    import signal

    if not sys.platform.startswith("linux"):
        pytest.skip("the local signal numbering is Linux's only on Linux")
    for n, name in platform.LINUX_SIGNALS.items():
        assert signal.Signals(n).name == name


# ---------------------------------------------------------------- paths (§8.1)
@pytest.fixture
def win_dirs(monkeypatch: pytest.MonkeyPatch, windows: None) -> tuple[str, str]:
    roaming, local = r"C:\Users\u\AppData\Roaming", r"C:\Users\u\AppData\Local"
    monkeypatch.setenv("WIN_PD_OVERRIDE_APPDATA", roaming)
    monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", local)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")                # ignored on Windows
    return roaming, local


def _win(p: Path) -> str:
    return str(p).replace("/", "\\")                       # platformdirs joins with os.sep: "\\" on real Windows


def test_windows_locations(win_dirs: tuple[str, str]) -> None:
    roaming, local = win_dirs
    assert _win(paths.config_file()) == roaming + r"\runfleet\config.toml"     # appauthor=False: one level
    assert _win(paths.state_dir()) == local + r"\runfleet"
    assert _win(paths.cache_file()) == local + r"\runfleet\cache.sqlite"
    assert _win(paths.log_file()) == local + r"\runfleet\runfleet.log"
    assert _win(paths.cache_dir()) == local + r"\runfleet\Cache"
    assert _win(paths.mux_dir()) == local + r"\runfleet\Cache\cm"             # no XDG, no length check, no /tmp


def test_windows_mux_dir_never_falls_back_to_tmp(monkeypatch: pytest.MonkeyPatch, windows: None) -> None:
    monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", "C:\\" + "x" * 120)
    monkeypatch.setattr(platform, "current_uid", lambda: pytest.fail("no uid on Windows"))
    assert "x" * 120 in str(paths.mux_dir()) and not str(paths.mux_dir()).startswith("/tmp")


def test_windows_private_dirs_skip_modes(tmp_path: Path, windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "chmod", lambda *a, **k: pytest.fail("no chmod on Windows"))
    monkeypatch.setattr(platform, "current_uid", lambda: pytest.fail("no uid on Windows"))
    d = paths.ensure_private_dir(tmp_path / "a" / "b")
    assert d.is_dir()
    f = paths.private_file(d / "log")
    paths.write_private(d / "x.json", "{}")
    assert f.exists() and (d / "x.json").read_text() == "{}"


def test_display_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    inside = tmp_path / ".config" / "runfleet" / "config.toml"
    assert paths.display_path(inside) == "~/" + str(Path(".config", "runfleet", "config.toml"))
    assert paths.display_path(Path("/etc/x")) == str(Path("/etc/x"))
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert paths.display_path(inside).startswith("~\\")


# ---------------------------------------------------------------- mux policy and the jump rewrite (§4.1-4.2)
@pytest.mark.parametrize("cfg_mux", ["auto", "user", "managed", "off"])
@pytest.mark.parametrize("g", [G_K2, G_LYRA])
def test_windows_forces_sharing_off(windows: None, g: str, cfg_mux: str) -> None:
    plan = mux.decide(parse_ssh_g(g, "h"), HostConfig("h", "h", mux=cfg_mux), Path("/x"))  # type: ignore[arg-type]
    assert plan.mode == "off" and not plan.uses_master and plan.control_path is None
    assert plan.probe_opts == ("-o", "ControlMaster=no", "-o", "ControlPath=none")


def test_windows_jump_rewrite_quoting(windows: None) -> None:
    opts = mux.jump_rewrite(parse_ssh_g(G_LYRA, "lyra"), batch=True, ssh=SSH_EXE)
    assert opts == ("-o", f'ProxyCommand="{SSH_EXE}" -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR '
                          '-W "[%h]:%p" jumpvm')
    multi = mux.jump_rewrite(parse_ssh_g(G_MULTI, "t"), batch=True, ssh=SSH_EXE)
    assert multi[1].endswith('-p 2200 -l bob -J usr@jump1:2222 -W "[%h]:%p" jump2')
    log = Path(r"C:\Users\Jane Doe\AppData\Local\runfleet\ssh-50%.log")
    inter = mux.jump_rewrite(parse_ssh_g(G_LYRA, "lyra"), batch=False, log=log,
                             ssh=r"C:\Program Files\OpenSSH\ssh.exe")
    cmd = inter[1].removeprefix("ProxyCommand=")
    assert cmd.startswith('"C:\\Program Files\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=10')
    assert f'-E "{str(log).replace("%", "%%")}"' in cmd                     # quoted for the MSVC runtime, % escaped
    assert "BatchMode" not in cmd and "'" not in cmd                        # no POSIX quoting anywhere


def test_windows_jump_rewrite_round_trips_through_the_c_runtime(windows: None) -> None:
    """What the hop's C runtime parses back (CommandLineToArgvW rules, as subprocess.list2cmdline writes them)."""
    log = Path(r"C:\a b\x.log")
    cmd = mux.jump_rewrite(parse_ssh_g(G_LYRA, "lyra"), batch=False, log=log, ssh=SSH_EXE)[1]
    args = cmd.removeprefix(f'ProxyCommand="{SSH_EXE}" ')
    words = _msvc_split(args.replace("%%", "%"))
    assert words == ["-T", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR", "-E", str(log), "-W", "[%h]:%p",
                     "jumpvm"]


def _msvc_split(text: str) -> list[str]:
    """Minimal CommandLineToArgvW: whitespace splits, double quotes group, backslashes are literal before
    anything but a quote (enough for the words the rewrite produces)."""
    out, cur, quoted, have = [], "", False, False
    for ch in text:
        if ch == '"':
            quoted, have = not quoted, True
        elif ch in " \t" and not quoted:
            if cur or have:
                out.append(cur)
            cur, have = "", False
        else:
            cur += ch
    if cur or have:
        out.append(cur)
    return out


def test_posix_jump_rewrite_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    opts = mux.jump_rewrite(parse_ssh_g(G_LYRA, "lyra"), batch=True)
    assert opts[1] == "ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -W '[%h]:%p' jumpvm"


async def test_openssh_transport_on_windows(windows: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    t = OpenSSHTransport(HostConfig("lyra", "lyra", mux="managed"), ssh="ssh", mux_dir=tmp_path / "cm",
                         log_path=tmp_path / "ssh-lyra.log")
    t.settings = parse_ssh_g(G_LYRA, "lyra")
    t.plan = mux.decide(t.settings, t.config, tmp_path / "cm")
    monkeypatch.setattr(t, "_ssh_exe", lambda: SSH_EXE)
    argv = t.base_argv()
    assert argv[0] == "ssh" and "ControlPath=none" in argv
    proxy = next(a for a in argv if a.startswith("ProxyCommand="))
    assert proxy.startswith(f'ProxyCommand="{SSH_EXE}" -T -o BatchMode=yes')
    connect = t.connect_argv()
    assert connect is not None and "-fN" not in connect and connect[-1] == "true"    # just an auth check
    assert t.check_argv()[1] == "-O"            # never run: the poller skips check_master when sharing is off
    assert "Windows ssh lacks" in (t.forward_blocker() or "")
    assert await t.check_master() is True
    assert await t.forward(8888, "n1", 8888) is not None

    async def no_spawn(*a: Any, **k: Any) -> None:
        pytest.fail("no `ps` on Windows")

    monkeypatch.setattr(spawn, "run", no_spawn)
    from runfleet.transport import openssh

    assert await openssh._pids_with_args(["ssh", "-fN", "x"]) == []
    assert await openssh._detached_ssh_with("-o ControlPath=x") == []


def test_ssh_exe_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ssh"
    exe.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: "ssh" if name == "ssh" else None)
    t = OpenSSHTransport(HostConfig("k2", "k2"), log_path=tmp_path / "log")
    assert t._ssh_exe() == str(exe)                                        # a relative PATH entry made absolute
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    assert t._ssh_exe() == "ssh"


async def test_openssh_resolve_on_windows_never_needs_a_uid(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    from runfleet.transport import openssh

    async def fake_resolve(alias: str, **kw: Any) -> Any:
        return parse_ssh_g(G_K2, alias)                                     # a user ControlPath in ~/.ssh/config

    monkeypatch.setattr(openssh, "resolve", fake_resolve)
    monkeypatch.setattr(platform, "current_uid", lambda: pytest.fail("no uid on Windows"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", r"C:\Users\u\AppData\Local")
    t = OpenSSHTransport(HostConfig("k2", "k2"), log_path=tmp_path / "ssh-k2.log")
    assert await t.resolve() is None
    assert t.mux == "off" and t.rendered_control_path is None
    assert await t.prepare() is None                                        # no master to create


async def test_local_host_is_refused_on_windows(windows: None) -> None:
    from runfleet.errors import RunFleetError
    from runfleet.probe.section import StreamSpec

    t = LocalTransport(HostConfig("localhost", "local"))
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.MISSING and "WSL" in err.summary and len(err.summary) <= 80
    script = framing.make_script([Section("t.x", "echo hi")], "poll")
    res = await t.probe(script, timeout=5)
    assert res.error == err and res.sections["t.x"].outcome is SectionOutcome.NOT_RUN
    with pytest.raises(RunFleetError, match="WSL"):
        await t.stream(StreamSpec("log.tail", "tail -f x"))
    assert "WSL" in (t.forward_blocker() or "")


# ---------------------------------------------------------------- spawn on Windows (§4.2)
async def test_spawn_gives_windows_a_closed_stdin_pipe(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    real = platform.background_spawn_kwargs

    def kwargs() -> dict[str, Any]:
        calls.append({})
        monkeypatch.setattr(platform, "IS_WINDOWS", False)                  # real creationflags only work on Windows
        try:
            return real()
        finally:
            monkeypatch.setattr(platform, "IS_WINDOWS", True)

    monkeypatch.setattr(platform, "background_spawn_kwargs", kwargs)
    res = await spawn.run([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], timeout=20)
    assert res.rc == 0 and res.stdout.strip() == b"''" and calls                # a pipe at EOF, never NUL


async def test_terminate_on_windows_runs_taskkill_off_the_event_loop(windows: None,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """taskkill is a process of its own: when the network drops, every probe times out at once, and the TUI must not
    stall for each kill in turn."""
    import threading

    threads: list[int] = []
    monkeypatch.setattr(platform, "taskkill_tree", lambda pid: threads.append(threading.get_ident()) or 0)
    monkeypatch.setattr(platform, "background_spawn_kwargs", lambda: {})
    proc = await spawn.spawn([sys.executable, "-c", "import time; time.sleep(60)"], stdin_pipe=True)
    try:
        await spawn.terminate(proc, grace=5)
        assert proc.returncode is not None                  # TerminateProcess after taskkill
        assert threads and threading.get_ident() not in threads
    finally:
        spawn.release(proc)


def test_descendants_never_reads_proc_on_windows(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "listdir", lambda p: pytest.fail(f"listdir({p!r}) on Windows"))
    assert spawn.descendants(os.getpid()) == []


async def test_terminate_on_windows_uses_taskkill(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(platform, "taskkill_tree", lambda pid: killed.append(pid) or 0)
    monkeypatch.setattr(platform, "background_spawn_kwargs", lambda: {})
    proc = await spawn.spawn([sys.executable, "-c", "import time; time.sleep(60)"], stdin_pipe=True)
    try:
        await spawn.terminate(proc, grace=5)
        assert proc.returncode is not None and killed == [proc.pid]
    finally:
        spawn.release(proc)


# ---------------------------------------------------------------- logging over the TUI
def test_failed_log_rotation_keeps_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet import cli

    path = tmp_path / "runfleet.log"
    handler = cli.PrivateRotatingFileHandler(path, maxBytes=50, backupCount=2, encoding="utf-8")
    calls: list[str] = []

    def refuse(src: str, dst: str) -> None:
        calls.append(dst)
        raise PermissionError(errno.EACCES, "used by another process")       # Windows: another RunFleet has it

    monkeypatch.setattr(handler, "rotate", refuse)
    logger = logging.getLogger("runfleet.test.rotation")
    logger.propagate = False
    logger.addHandler(handler)
    errors: list[str] = []
    monkeypatch.setattr(handler, "handleError", lambda record: errors.append(record.getMessage()))
    try:
        for i in range(10):
            logger.warning("line %d %s", i, "x" * 40)
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert not errors and len(calls) == 1                                    # tried once, then stopped trying
    assert path.read_text(encoding="utf-8").count("line ") == 10


async def test_doctor_on_windows_says_why_sharing_is_off(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    from runfleet.core.clock import ManualClock
    from runfleet.core.doctor import run_doctor
    from runfleet.models import HostError
    from runfleet.transport import openssh
    from tests.conftest import CLOCK_START
    from tests.unit.stubs import STUB_HOOKS

    async def fake_resolve(alias: str, **kw: Any) -> Any:
        return parse_ssh_g(G_LYRA, alias)

    async def no_probe(script: Any, *, timeout: float) -> Any:
        return spawn.failed_probe(script, HostError(ErrorKind.AUTH, "authentication needed"))

    monkeypatch.setattr(openssh, "resolve", fake_resolve)
    host = HostConfig("lyra", "lyra", backend="slurm", mux="managed")
    t = OpenSSHTransport(host, mux_dir=tmp_path / "cm", log_path=tmp_path / "ssh.log")
    monkeypatch.setattr(t, "probe", no_probe)
    rep = await run_doctor(host, t, hooks=STUB_HOOKS, clock=ManualClock(CLOCK_START))
    by = {c.name: c for c in rep.checks}
    assert by["mux"].detail.startswith("off (Windows); ProxyJump rewritten")
    assert by["master"].detail == "not used" and by["probe"].ok is False
    assert mux.WINDOWS_NO_MUX in rep.warnings


# ---------------------------------------------------------------- config and ~/.ssh/config on Windows
def test_reserved_host_names_are_refused_on_windows(windows: None) -> None:
    from runfleet.config import host_name_for, parse_config
    from runfleet.errors import ConfigError

    for bad in ("con", "NUL", "com1", "lpt3.lab", "web."):
        with pytest.raises(ConfigError, match="reserved on Windows"):
            parse_config({"host": [{"name": bad, "ssh": "box"}]})
    assert parse_config({"host": [{"name": "console", "ssh": "box"}]})[0].hosts[0].name == "console"
    assert host_name_for("con") == "con_" and host_name_for("aux.example.org") == "aux.example.org_"
    assert host_name_for("web.") == "web_" and host_name_for("k2") == "k2"


def test_reserved_host_names_are_fine_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.config import host_name_for, parse_config

    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert parse_config({"host": [{"name": "con", "ssh": "box"}]})[0].hosts[0].name == "con"
    assert host_name_for("con") == "con"


def test_ssh_config_aliases_with_windows_paths(windows: None, tmp_path: Path) -> None:
    from runfleet import sshconfig

    inc = tmp_path / "conf.d"
    inc.mkdir()
    (inc / "lab").write_text("Host lab-gpu\n  HostName gpu.example.org\n", encoding="utf-8")
    cfg = tmp_path / "config"
    cfg.write_text(f'Host k2 "my box"\n  IdentityFile C:\\Users\\me\\.ssh\\id_k2\nInclude {inc}/*\n', encoding="utf-8")
    assert sshconfig.aliases_from(cfg) == ["k2", "my box", "lab-gpu"]
    assert sshconfig._tokens(r"IdentityFile C:\Users\me\.ssh\id_k2") == ["IdentityFile", r"C:\Users\me\.ssh\id_k2"]
