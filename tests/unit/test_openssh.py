"""[WP1] OpenSSH transport: mux policy, jump rewrite, exact argv (§4.1-4.2), ensure_master, probes via a fake ssh."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from runfleet.config import HostConfig
from runfleet.models import ErrorKind
from runfleet.probe import framing
from runfleet.probe.section import Section, SectionOutcome
from runfleet.sshconfig import SshSettings, parse_ssh_g
from runfleet.transport import make_transport, mux
from runfleet.transport.base import RunResult
from runfleet.transport.local import LocalTransport
from runfleet.transport.openssh import OpenSSHTransport

MUX_DIR = Path("/run/user/1000/runfleet")

G_K2 = "user alice0001\nhostname k2.example.edu\nport 22\ncontrolmaster auto\n" \
       "controlpath /home/usr/.ssh/cm-0123\ncontrolpersist 600\n"
G_LYRA = "user alice07\nhostname lyra.example.edu\nport 22\ncontrolmaster false\ncontrolpersist no\n" \
         "proxyjump jumpvm\n"
G_LBS = "user alice07\nhostname labstation.example.edu\nport 22\ncontrolmaster auto\ncontrolpath /home/usr/.ssh/cm-9\n" \
       "controlpersist 600\nproxyjump jumpvm\n"
G_MULTI = "user u\nhostname t\nport 22\ncontrolpersist no\nproxyjump usr@jump1:2222,ssh://bob@jump2:2200\n"
G_OWNPC = "user u\nhostname t\nport 22\ncontrolpersist no\nproxyjump jumpvm\nproxycommand nc %h %p\n"
G_PERSIST_NO = "user u\nhostname t\nport 22\ncontrolpath /home/u/.ssh/cm-%C\ncontrolpersist no\n"


def settings(text: str, alias: str = "h") -> SshSettings:
    return parse_ssh_g(text, alias)


# ---------------------------------------------------------------- mux policy (§4.1 table)
@pytest.mark.parametrize(("g", "cfg_mux", "mode"), [
    (G_K2, "auto", "user"), (G_LYRA, "auto", "managed"), (G_LBS, "auto", "user"), (G_PERSIST_NO, "auto", "managed"),
    (G_K2, "off", "off"), (G_K2, "managed", "managed"), (G_LYRA, "user", "user"),
])
def test_mux_decision(g: str, cfg_mux: str, mode: str) -> None:
    plan = mux.decide(settings(g), HostConfig(name="h", ssh="h", mux=cfg_mux), MUX_DIR)  # type: ignore[arg-type]
    assert plan.mode == mode


def test_mux_option_lists() -> None:
    user = mux.decide(settings(G_K2), HostConfig("k2", "k2"), MUX_DIR)
    # the user's socket as `ssh -G` expanded it is named literally everywhere (the jump rewrite changes %C)
    cp = ("-o", "ControlPath=/home/usr/.ssh/cm-0123")
    assert user.probe_opts == ("-o", "ControlMaster=no", *cp) and user.imux_opts == cp and user.path_opts == cp
    assert user.socket == "/home/usr/.ssh/cm-0123"
    managed = mux.decide(settings(G_LYRA), HostConfig("lyra", "lyra"), MUX_DIR)
    assert managed.probe_opts == ("-o", "ControlMaster=no", "-o", f"ControlPath={MUX_DIR}/cm-%C")
    assert managed.imux_opts == ("-o", "ControlMaster=auto", "-o", f"ControlPath={MUX_DIR}/cm-%C",
                                 "-o", "ControlPersist=15m") and managed.socket is None    # a shell may be the master
    lit = mux.decide(settings(G_LYRA), HostConfig("lyra", "lyra"), MUX_DIR, socket=f"{MUX_DIR}/cm-50%x")
    assert lit.probe_opts == ("-o", "ControlMaster=no", "-o", f"ControlPath={MUX_DIR}/cm-50%%x")   # % escaped
    assert mux.control_path_option("/a b/c", literal=True) == 'ControlPath="/a b/c"'
    unexpanded = mux.decide(settings(G_PERSIST_NO), HostConfig("h", "h", mux="user"), MUX_DIR)
    assert unexpanded.probe_opts[-1] == "ControlPath=/home/u/.ssh/cm-%C" and unexpanded.socket is None
    off = mux.decide(settings(G_K2), HostConfig("k2", "k2", mux="off"), MUX_DIR)
    assert off.probe_opts == ("-o", "ControlMaster=no", "-o", "ControlPath=none")
    assert off.imux_opts == ("-o", "ControlPath=none") and not off.uses_master


# ---------------------------------------------------------------- jump rewrite (§4.2)
def test_jump_rewrite_single_hop() -> None:
    assert mux.jump_rewrite(settings(G_LYRA), batch=True) == (
        "-o", "ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -W '[%h]:%p' jumpvm")
    inter = mux.jump_rewrite(settings(G_LYRA), batch=False, log=Path("/home/u/.local/state/runfleet/ssh-lyra.log"))
    assert inter == ("-o", "ProxyCommand=ssh -T -o ConnectTimeout=10 -o LogLevel=ERROR -E "
                           "/home/u/.local/state/runfleet/ssh-lyra.log -W '[%h]:%p' jumpvm")


def test_jump_rewrite_multi_hop_and_escaping() -> None:
    opts = mux.jump_rewrite(settings(G_MULTI), batch=True, log=None)
    assert opts == ("-o", "ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -p 2200 "
                          "-l bob -J usr@jump1:2222 -W '[%h]:%p' jump2")
    weird = mux.jump_rewrite(settings(G_LYRA), batch=False, log=Path("/tmp/a b/50%.log"))
    assert "-E '/tmp/a b/50%%.log'" in weird[1]


def test_no_rewrite_without_jump_or_with_own_proxycommand() -> None:
    assert mux.jump_rewrite(settings(G_K2), batch=True) == ()
    assert mux.jump_rewrite(settings(G_OWNPC), batch=True) == ()


def test_parse_jump() -> None:
    hops = mux.parse_jump("usr@jump1:2222,jump2,ssh://bob@[::1]:22")
    assert [(h.user, h.host, h.port) for h in hops] == [("usr", "jump1", 2222), (None, "jump2", None),
                                                         ("bob", "::1", 22)]
    assert hops[2].spec() == "bob@[::1]:22"
    with pytest.raises(ValueError):
        mux.parse_jump("a@@b:x:y")
    # OpenSSH splits at the LAST "@": e-mail style user names are legal
    (hop,) = mux.parse_jump("first.last@corp.com@bastion:2200")
    assert (hop.user, hop.host, hop.port) == ("first.last@corp.com", "bastion", 2200)
    assert hop.spec() == "first.last@corp.com@bastion:2200"


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs the local OpenSSH client for ssh -G")
def test_rewritten_argv_through_real_ssh_g(tmp_path: Path) -> None:
    """§10.3 golden: the real local `ssh -G` sees a BatchMode ProxyCommand and no ProxyJump."""
    cfg = tmp_path / "config"
    cfg.write_text("Host lyra\n  HostName lyra.example.edu\n  User alice07\n  ProxyJump jumpvm\n"
                   "  ControlMaster no\nHost multi\n  ProxyJump usr@jump1:2222,jump2\n", encoding="utf-8")
    for alias, g in (("lyra", G_LYRA), ("multi", "user u\nhostname multi\nport 22\nproxyjump usr@jump1:2222,jump2\n")):
        opts = mux.jump_rewrite(settings(g, alias), batch=True)
        out = subprocess.run(["ssh", "-G", "-F", str(cfg), *opts, alias], capture_output=True, text=True,
                             check=True).stdout
        keys = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
        assert "BatchMode=yes" in keys["proxycommand"] and "ConnectTimeout=10" in keys["proxycommand"]
        assert "proxyjump" not in keys


# ---------------------------------------------------------------- exact command lines
FAKE_SSH = r"""#!/bin/sh
# Fake ssh for tests: records argv, answers -G from $RF_FAKE_G, emulates -O check / -fN, runs probes locally.
printf '%s\037' "$@" >> "$RF_FAKE_LOG"; printf '\n' >> "$RF_FAKE_LOG"
case "$1" in
  -G) cat "$RF_FAKE_G"; for a in "$@"; do case "$a" in ControlPath=*) echo "controlpath ${a#ControlPath=}" | sed 's/%C/abc123/';; esac; done; exit 0 ;;
  -O)
    cp=; for a in "$@"; do case "$a" in ControlPath=*) cp=${a#ControlPath=};; esac; done
    if [ -f "$RF_FAKE_DIR/master" ]; then echo "Master running (pid=4242)" >&2; exit 0; fi
    if [ -n "$cp" ] && [ -e "$cp" ]; then
      [ -f "$RF_FAKE_DIR/busy" ] && { echo "mux_client_hello_exchange: read packet: Broken pipe" >&2; exit 255; }
      echo "Control socket connect($cp): Connection refused" >&2; exit 255
    fi
    echo "Control socket connect(x): No such file or directory" >&2; exit 255 ;;
  -fN)
    log=; prev=; cp=; for a in "$@"; do [ "$prev" = -E ] && log=$a; prev=$a
      case "$a" in ControlPath=*) cp=${a#ControlPath=};; esac; done
    case "$(cat "$RF_FAKE_DIR/mode" 2>/dev/null)" in
      auth) echo "alice07@lyra: Permission denied (publickey,gssapi-with-mic)." >> "$log"; exit 255 ;;
      squat) echo "ControlSocket $cp already exists, disabling multiplexing" >> "$log"; exit 0 ;;
      race) echo "ControlSocket $cp already exists, disabling multiplexing" >> "$log"; : > "$RF_FAKE_DIR/master"
            exit 0 ;;
      hop) case "$*" in *"-E $log -W"*) echo "alice07@jumpvm: Permission denied (publickey)." >> "$log";; esac
           echo "Connection closed by UNKNOWN port 65535" >> "$log"; exit 255 ;;
      *) : > "$RF_FAKE_DIR/master"; exit 0 ;;
    esac ;;
esac
for last; do :; done
exec sh -c "$last"
"""


@pytest.fixture
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "ssh"
    exe.write_text(FAKE_SSH, encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("RF_FAKE_LOG", str(tmp_path / "argv.log"))
    monkeypatch.setenv("RF_FAKE_DIR", str(tmp_path))
    return exe


def use_g(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    (tmp_path / "g.txt").write_text(text, encoding="utf-8")
    monkeypatch.setenv("RF_FAKE_G", str(tmp_path / "g.txt"))


def calls(tmp_path: Path) -> list[list[str]]:
    lines = (tmp_path / "argv.log").read_text().splitlines() if (tmp_path / "argv.log").exists() else []
    return [line.split("\x1f")[:-1] for line in lines]


def transport(fake: Path, tmp_path: Path, host: HostConfig) -> OpenSSHTransport:
    return OpenSSHTransport(host, ssh=str(fake), mux_dir=tmp_path / "mux", log_path=tmp_path / "ssh-h.log")


async def test_argv_user_mux(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_K2)
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2", ssh_options=("Compression=yes",)))
    assert await t.resolve() is None and t.mux == "user"
    ssh, log = str(fake_ssh), str(tmp_path / "ssh-h.log")
    cp = ["-o", "ControlPath=/home/usr/.ssh/cm-0123"]
    user = ["-o", "Compression=yes"]
    base = [ssh, "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no", *cp, *user, "-o", "ConnectTimeout=10", "-o",
            "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "-o", "LogLevel=ERROR", "--", "k2"]
    assert t.probe_argv() == base + ["exec sh -s"]
    assert t.stream_argv("tail -F -- '/x y'") == base + ["exec sh -c 'tail -F -- '\"'\"'/x y'\"'\"''"]
    assert t.check_argv() == [ssh, "-O", "check", "-o", "ControlMaster=no", *cp, *user, "--", "k2"]
    master = t.master_argv()
    assert master == [ssh, "-fN", "-o", "BatchMode=yes", "-o", "ControlMaster=yes", *cp, *user,
                      "-o", "ConnectTimeout=45", "-o", "LogLevel=ERROR", "-E", log, "--", "k2"]
    assert not any("ServerAlive" in a or "ControlPersist" in a for a in master)   # only the user's own options
    assert t.interactive_argv(None) == [ssh, "-t", *cp, *user, "--", "k2"]
    assert t.interactive_argv("cd /w && exec $SHELL -l") == [ssh, "-t", *cp, *user, "--", "k2",
                                                             "cd /w && exec $SHELL -l"]
    assert t.connect_argv() == [ssh, "-fN", "-o", "ControlMaster=yes", *cp, *user, "-o", "LogLevel=ERROR", "-E",
                                log, "--", "k2"]
    assert calls(tmp_path)[0] == ["-G", "-o", "Compression=yes", "--", "k2"]     # ssh -G sees ssh_options too
    login = transport(fake_ssh, tmp_path, HostConfig("k2", "k2", remote_shell="login", forward_agent=True))
    await login.resolve()
    assert login.probe_argv()[-1] == 'exec bash -l -c "exec sh -s"'
    assert login.interactive_argv(None) == [ssh, "-t", *cp, "-A", "--", "k2"]


async def test_argv_managed_mux_with_jump(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_LYRA)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra", connect_timeout=120))
    assert await t.resolve() is None and t.mux == "managed"
    # the socket is expanded once by `ssh -G` and then named literally by every command (%C cannot differ)
    ssh, log, cp = str(fake_ssh), str(tmp_path / "ssh-h.log"), f"{tmp_path / 'mux'}/cm-abc123"
    jump_b = "ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -W '[%h]:%p' jumpvm"
    jump_m = f"ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR -E {log} -W '[%h]:%p' " \
             "jumpvm"                       # master creation: the hop logs its own auth/host-key failures
    jump_i = f"ProxyCommand=ssh -T -o ConnectTimeout=10 -o LogLevel=ERROR -E {log} -W '[%h]:%p' jumpvm"
    assert t.probe_argv() == [ssh, "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", f"ControlPath={cp}",
                              "-o", jump_b, "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o",
                              "ServerAliveCountMax=2", "-o", "LogLevel=ERROR", "--", "lyra", "exec sh -s"]
    assert t.master_argv() == [ssh, "-fN", "-o", "BatchMode=yes", "-o", "ControlMaster=yes", "-o",
                               f"ControlPath={cp}", "-o", "ControlPersist=15m", "-o", jump_m, "-o",
                               "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=120",
                               "-o", "LogLevel=ERROR", "-E", log, "--", "lyra"]
    assert t.connect_argv() == [ssh, "-fN", "-o", "ControlMaster=yes", "-o", f"ControlPath={cp}", "-o",
                                "ControlPersist=15m", "-o", jump_i, "-o", "ServerAliveInterval=15", "-o",
                                "ServerAliveCountMax=4", "-o", "LogLevel=ERROR", "-E", log, "--", "lyra"]
    assert t.interactive_argv(None) == [ssh, "-t", "-o", "ControlMaster=auto", "-o", f"ControlPath={cp}", "-o",
                                        "ControlPersist=15m", "-o", jump_i, "--", "lyra"]
    assert t.check_argv("exit") == [ssh, "-O", "exit", "-o", "ControlMaster=no", "-o", f"ControlPath={cp}", "-o",
                                    jump_b, "--", "lyra"]
    assert t.rendered_control_path == cp
    assert (tmp_path / "mux").stat().st_mode & 0o777 == 0o700


async def test_argv_off(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_K2)
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2", mux="off"))
    await t.resolve()
    assert t.mux == "off" and await t.check_master() is True
    assert ["-o", "ControlMaster=no", "-o", "ControlPath=none"] == t.probe_argv()[4:8]
    assert await t.prepare() is None
    assert not any(c[0] == "-O" for c in calls(tmp_path))


async def test_ensure_master_creates_once_and_checks(fake_ssh: Path, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_K2)
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2"))
    assert await t.prepare() is None
    kinds = [c[0] for c in calls(tmp_path)]
    assert kinds == ["-G", "-O", "-fN", "-O"]                    # check failed -> create -> verify it took the socket
    assert await t.check_master() is True
    assert await t.prepare() is None                                   # idempotent: only a check now
    assert [c[0] for c in calls(tmp_path)] == ["-G", "-O", "-fN", "-O", "-O", "-O"]


async def test_ensure_master_auth_failure_is_classified_and_throttled(
        fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_LYRA)
    (tmp_path / "mode").write_text("auth")
    now = [1000.0]
    t = OpenSSHTransport(HostConfig("lyra", "lyra"), ssh=str(fake_ssh), mux_dir=tmp_path / "mux",
                         log_path=tmp_path / "ssh-h.log", monotonic=lambda: now[0])
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.AUTH
    assert "Permission denied" in (tmp_path / "ssh-h.log").read_text()
    n = len(calls(tmp_path))
    now[0] += 10
    again = await t.prepare()                                          # < 30 s: no new master attempt
    assert again == err and [c[0] for c in calls(tmp_path)[n:]] == ["-O"]
    now[0] += 30
    (tmp_path / "mode").write_text("ok")
    assert await t.prepare() is None


async def test_manual_retry_forces_a_master_attempt(fake_ssh: Path, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """A user retry (r / after connect) must not be swallowed by the 30 s throttle."""
    use_g(tmp_path, monkeypatch, G_LYRA)
    (tmp_path / "mode").write_text("auth")
    now = [1000.0]
    t = OpenSSHTransport(HostConfig("lyra", "lyra"), ssh=str(fake_ssh), mux_dir=tmp_path / "mux",
                         log_path=tmp_path / "ssh-h.log", monotonic=lambda: now[0])
    assert (await t.prepare()) is not None
    now[0] += 5
    (tmp_path / "mode").write_text("ok")                               # the user ran ssh-add
    assert await t.prepare() is not None                               # throttled
    assert await t.prepare(force=True) is None
    assert [c[0] for c in calls(tmp_path)].count("-fN") == 2


async def test_stale_managed_socket_is_removed_before_creating(fake_ssh: Path, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A master killed with -9 leaves its socket; ControlMaster=yes would then run as a plain, leaked login."""
    use_g(tmp_path, monkeypatch, G_LYRA)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    await t.resolve()
    sock = Path(t.rendered_control_path or "")
    sock.write_text("")                                                 # stale: check says Connection refused
    assert await t.prepare() is None
    assert not sock.exists() and [c[0] for c in calls(tmp_path)][-3:] == ["-O", "-fN", "-O"]


async def test_busy_socket_blocks_creation(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_LYRA)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    await t.resolve()
    sock = Path(t.rendered_control_path or "")
    sock.write_text("")
    (tmp_path / "busy").write_text("")                                  # something answers, but not as a master
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.MUX and "not answering" in err.summary
    assert sock.exists() and "-fN" not in [c[0] for c in calls(tmp_path)]


@pytest.mark.parametrize(("cm", "unlinked"), [("auto", True), ("yes", False)])
async def test_stale_user_socket_only_removed_like_openssh_would(
        fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cm: str, unlinked: bool) -> None:
    sock = tmp_path / "um-sock"
    use_g(tmp_path, monkeypatch, f"user u\nhostname h\nport 22\ncontrolmaster {cm}\ncontrolpath {sock}\n"
                                 "controlpersist 600\n")
    sock.write_text("")
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2"))
    err = await t.prepare()
    assert (err is None) is unlinked and sock.exists() is not unlinked
    if not unlinked:
        assert err is not None and err.kind is ErrorKind.MUX and "stale" in err.summary


async def test_master_that_did_not_take_the_socket_is_an_error_and_reaped(
        fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ssh -fN -o ControlMaster=yes` exits 0 even when it could not bind; the stray login must not survive."""
    import subprocess as sp

    from runfleet.transport import openssh

    use_g(tmp_path, monkeypatch, G_LYRA)
    (tmp_path / "mode").write_text("squat")
    stray = sp.Popen(["sleep", "60"], start_new_session=True)
    seen: list[str] = []

    async def fake_pids(argv: list[str]) -> list[int]:
        seen.append(" ".join(argv))
        return [stray.pid]

    monkeypatch.setattr(openssh, "_pids_with_args", fake_pids)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.MUX and "another process" in err.summary
    assert seen == [" ".join(t.master_argv())]
    assert stray.wait(timeout=5) == -15


async def test_pids_with_args_finds_exact_command_lines() -> None:
    import subprocess as sp

    from runfleet.transport.openssh import _pids_with_args

    fake = Path(shutil.which("sleep") or "/bin/sleep")
    link = Path(__file__).parent / "__pycache__" / "fake-ssh-sleep"     # a program whose name contains "ssh"
    link.parent.mkdir(exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(fake)
    proc = sp.Popen([str(link), "61.25"], start_new_session=True)
    try:
        assert await _pids_with_args(["ssh", "61.25"]) == [proc.pid]      # wrapper/full path: matched on the tail
        assert await _pids_with_args(["ssh", "61.2"]) == []
        assert await _pids_with_args(["sleep", "61.25"]) == [proc.pid]
    finally:
        link.unlink(missing_ok=True)
        proc.kill()
        proc.wait()


async def test_before_connect_resolves_and_reports_live_master(fake_ssh: Path, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_LYRA)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    assert await t.before_connect() is False and t.mux == "managed"
    assert any("ControlPath=" in a for a in (t.connect_argv() or []))     # the managed form, not the user guess
    (tmp_path / "master").write_text("")
    assert await t.before_connect() is True


async def test_hop_auth_failure_during_master_creation_pauses(fake_ssh: Path, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The backgrounded hop's stderr is lost (capture=False); its own `-E` puts the failure into the master log."""
    use_g(tmp_path, monkeypatch, G_LYRA)
    (tmp_path / "mode").write_text("hop")
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.AUTH and "via jumpvm" in err.summary


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs the local OpenSSH client for ssh -G")
async def test_every_command_names_the_same_socket_through_real_ssh_g(tmp_path: Path) -> None:
    """Regression (M3 lyra, review core #1/#3/#4): ProxyJump + %C and identity-changing ssh_options gave the master,
    `-O check` and the probes different sockets. Evaluate each argv's options with the real `ssh -G`."""
    cfg = tmp_path / "config"
    cfg.write_text("Host lyra\n  HostName lyra.example.org\n  User bob\n  ProxyJump bastion\n"
                   "Host lbs\n  HostName lbs.example.org\n  ProxyJump bastion\n  ControlMaster auto\n"
                   f"  ControlPath {tmp_path}/um-%C\n  ControlPersist 600\n", encoding="utf-8")
    wrapper = tmp_path / "ssh"
    wrapper.write_text(f"#!/bin/sh\nexec ssh -F {cfg} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)

    def socket_of(argv: list[str]) -> str:
        i = argv.index("--")
        opts = [w for k, w in zip(argv[1:i], argv[2:i + 1], strict=False) if k == "-o"]
        out = subprocess.run([str(wrapper), "-G", *[x for o in opts for x in ("-o", o)], "--", argv[i + 1]],
                             capture_output=True, text=True, check=True).stdout
        return next(line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("controlpath "))

    for alias, ssh_options in (("lyra", ("User=alice", "Port=2222")), ("lbs", ())):
        t = OpenSSHTransport(HostConfig(alias, alias, ssh_options=ssh_options), ssh=str(wrapper),
                             mux_dir=tmp_path / "m", log_path=tmp_path / "l")
        assert await t.resolve() is None
        argvs = [t.master_argv(), t.check_argv(), t.check_argv("exit"), t.probe_argv(), t.stream_argv("true"),
                 t.connect_argv() or [], t.interactive_argv(None)]
        assert {socket_of(a) for a in argvs} == {t.rendered_control_path}
    # the user-mode socket is the one the user's own `ssh lbs` (config ProxyJump, no rewrite) uses
    own = subprocess.run([str(wrapper), "-G", "lbs"], capture_output=True, text=True, check=True).stdout
    assert f"controlpath {t.rendered_control_path}" in own


def test_ssh_options_can_override_tunable_defaults(tmp_path: Path) -> None:
    """OpenSSH keeps the first value: ssh_options must precede RunFleet's tunable defaults (but not BatchMode)."""
    t = OpenSSHTransport(HostConfig("h", "h", ssh_options=("ConnectTimeout=30",)), mux_dir=tmp_path,
                         log_path=tmp_path / "l")
    t.settings = settings(G_PERSIST_NO)
    t.plan = mux.decide(t.settings, t.config, tmp_path)
    argv = t.probe_argv()
    assert argv.index("ConnectTimeout=30") < argv.index("ConnectTimeout=10")
    assert argv.index("BatchMode=yes") < argv.index("ConnectTimeout=30")


async def test_probe_through_fake_ssh(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_K2)
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2"))
    s = framing.make_script([Section("a.one", "echo hello"), Section("a.two", "exit 4")])
    res = await t.probe(s, timeout=10)
    assert res.error is None
    assert res.sections["a.one"].stdout == "hello\n" and res.sections["a.two"].outcome is SectionOutcome.FAILED


async def test_probe_auth_failure_and_no_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ssh"
    exe.write_text("#!/bin/sh\ncase \"$1\" in -G) echo 'user u'; echo 'hostname h'; exit 0;; esac\n"
                   "if [ -f \"$RF_FAKE_DIR/quiet\" ]; then cat >/dev/null; exit 0; fi\n"
                   "echo 'u@h: Permission denied (publickey).' >&2; exit 255\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("RF_FAKE_DIR", str(tmp_path))
    t = OpenSSHTransport(HostConfig("h", "h", mux="off"), ssh=str(exe), mux_dir=tmp_path, log_path=tmp_path / "l")
    s = framing.make_script([Section("a.one", "echo hi")])
    res = await t.probe(s, timeout=5)
    assert res.error is not None and res.error.kind is ErrorKind.AUTH
    assert res.sections["a.one"].outcome is SectionOutcome.NOT_RUN
    (tmp_path / "quiet").write_text("")
    res = await t.probe(s, timeout=5)
    assert res.error is not None and res.error.kind is ErrorKind.REMOTE    # rc 0, no markers at all (§9 row 8)


async def test_stream_through_fake_ssh(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.core.logs import tail_spec

    use_g(tmp_path, monkeypatch, G_K2)
    log = tmp_path / "job.out"
    log.write_text("line1\nline2\n")
    t = transport(fake_ssh, tmp_path, HostConfig("k2", "k2"))
    handle = await t.stream(tail_spec(str(log), 65536))
    got = b""
    async for chunk in handle.chunks():
        got += chunk
        if b"line2\n" in got:
            break
    await handle.close()
    assert got == b"line1\nline2\n"
    assert handle.returncode is not None


async def test_reset_master_only_for_managed(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_K2)
    user = transport(fake_ssh, tmp_path, HostConfig("k2", "k2"))
    await user.resolve()
    await user.reset_master()
    assert not any(c[:2] == ["-O", "exit"] for c in calls(tmp_path))
    use_g(tmp_path, monkeypatch, G_LYRA)
    managed = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    await managed.resolve()
    sock = Path(managed.rendered_control_path or "")
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")
    await managed.reset_master()
    assert any(c[:2] == ["-O", "exit"] for c in calls(tmp_path))
    assert not sock.exists()


async def test_ssh_g_failure_is_a_host_error(tmp_path: Path) -> None:
    t = OpenSSHTransport(HostConfig("h", "h"), ssh=str(tmp_path / "nope"), mux_dir=tmp_path, log_path=tmp_path / "l")
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.SSH
    res = await t.probe(framing.make_script([Section("a.b", "true")]), timeout=1)
    assert res.error is not None


def test_unresolved_interactive_argv_falls_back_to_user_config(tmp_path: Path) -> None:
    t = OpenSSHTransport(HostConfig("h", "alias"), ssh="ssh", mux_dir=tmp_path, log_path=tmp_path / "l")
    assert t.interactive_argv(None) == ["ssh", "-t", "--", "alias"]
    assert t.connect_argv() == ["ssh", "-fN", "-o", "ControlMaster=yes", "-o", "LogLevel=ERROR", "-E",
                                str(tmp_path / "l"), "--", "alias"]
    assert t.mux == "none" and t.ssh_log == str(tmp_path / "l")


def test_make_transport() -> None:
    assert isinstance(make_transport(HostConfig("atlas", "local")), LocalTransport)
    t = make_transport(HostConfig("k2", "k2"))
    assert isinstance(t, OpenSSHTransport) and t.kind == "openssh"


async def test_local_transport_basics() -> None:
    t = LocalTransport(HostConfig("atlas", "local"))
    assert t.mux == "local" and t.ssh_log is None and t.connect_argv() is None
    assert await t.prepare() is None and await t.check_master() is True
    assert t.interactive_argv("echo hi") == ["sh", "-c", "echo hi"]
    assert len(t.interactive_argv(None)) == 1


# ---------------------------------------------------------------- review regressions
G_NOPATH = "user u\nhostname t\nport 22\ncontrolmaster auto\ncontrolpersist 600\n"


async def test_user_mux_without_controlpath_never_spawns_a_master(fake_ssh: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    plan = mux.decide(settings(G_NOPATH), HostConfig("h", "h", mux="user"), MUX_DIR)
    assert plan.mode == "user" and not plan.uses_master
    use_g(tmp_path, monkeypatch, G_NOPATH)
    t = transport(fake_ssh, tmp_path, HostConfig("h", "h", mux="user"))
    assert await t.prepare() is None and await t.check_master() is True
    assert await t.before_connect() is False
    assert not any(c[0] in ("-fN", "-O") for c in calls(tmp_path))
    connect = t.connect_argv() or []
    assert "-fN" not in connect and connect[-1] == "true"                 # an auth check, not a stray login


async def test_lost_master_race_reaps_our_stray_login(fake_ssh: Path, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    from runfleet.transport import openssh

    use_g(tmp_path, monkeypatch, G_LYRA)
    (tmp_path / "mode").write_text("race")              # another creator binds the socket; ours stays a login
    stray = sp.Popen(["sleep", "60"], start_new_session=True)

    async def fake_pids(argv: list[str]) -> list[int]:
        return [stray.pid, 4242]                        # 4242 answers -O check: the real master, spared

    monkeypatch.setattr(openssh, "_pids_with_args", fake_pids)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    assert await t.prepare() is None
    assert stray.wait(timeout=5) == -15


async def test_reset_master_stops_a_master_that_ignores_exit(fake_ssh: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    from runfleet.transport import openssh

    use_g(tmp_path, monkeypatch, G_LYRA)
    wedged = sp.Popen(["sleep", "60"], start_new_session=True)
    seen: list[str] = []

    async def fake_pids(argv: list[str]) -> list[int]:
        seen.append(argv[1])
        return [wedged.pid] if argv[1] == "-fN" and "BatchMode=yes" in argv else []

    monkeypatch.setattr(openssh, "_pids_with_args", fake_pids)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    await t.resolve()
    sock = Path(t.rendered_control_path or "")
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.write_text("")                                 # -O exit: "Connection refused", rc 255
    await t.reset_master()
    assert wedged.wait(timeout=5) == -15 and not sock.exists()


def test_managed_path_escapes_percent_in_the_directory() -> None:
    assert mux.managed_control_path(Path("/run/a%b")) == "/run/a%%b/cm-%C"


async def test_managed_resolve_quotes_a_mux_dir_with_spaces(fake_ssh: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    use_g(tmp_path, monkeypatch, G_LYRA)
    t = OpenSSHTransport(HostConfig("lyra", "lyra"), ssh=str(fake_ssh), mux_dir=tmp_path / "a b",
                         log_path=tmp_path / "l")
    assert await t.resolve() is None
    g_calls = [c for c in calls(tmp_path) if c[0] == "-G"]
    assert f'ControlPath="{tmp_path}/a b/cm-%C"' in g_calls[1]


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs the local OpenSSH client for ssh -G")
async def test_user_controlpath_with_literal_percent_is_not_expanded_twice(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.write_text(f"Host lbs\n  HostName lbs.example.org\n  ControlMaster auto\n  ControlPath {tmp_path}/cm-%%C-%h\n"
                   "  ControlPersist 600\n", encoding="utf-8")
    wrapper = tmp_path / "ssh"
    wrapper.write_text(f"#!/bin/sh\nexec ssh -F {cfg} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    t = OpenSSHTransport(HostConfig("lbs", "lbs"), ssh=str(wrapper), mux_dir=tmp_path / "m", log_path=tmp_path / "l")
    assert await t.resolve() is None and t.plan is not None
    assert t.rendered_control_path == f"{tmp_path}/cm-%C-lbs.example.org"
    assert f"ControlPath={tmp_path}/cm-%%C-lbs.example.org" in t.probe_argv()


async def test_reset_master_stops_a_wedged_master_left_by_an_interactive_session(
        fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shells/pagers use `ControlMaster=auto` + `ControlPersist`: the master they fork runs with the interactive
    command line and no tty, so a failed `-O exit` must find it by its ControlPath options, not the `-fN` forms."""
    import subprocess as sp
    import sys

    from runfleet.transport import openssh

    use_g(tmp_path, monkeypatch, G_LYRA)

    async def no_pids(argv: list[str]) -> list[int]:
        return []

    monkeypatch.setattr(openssh, "_pids_with_args", no_pids)
    t = transport(fake_ssh, tmp_path, HostConfig("lyra", "lyra"))
    await t.resolve()
    link = tmp_path / "fake-ssh-py"                     # a program whose name contains "ssh"
    link.symlink_to(sys.executable)
    argv = t.interactive_argv("htop")
    wedged = sp.Popen([str(link), "-c", "import time; time.sleep(60)", *argv[1:]], start_new_session=True)
    try:
        sock = Path(t.rendered_control_path or "")
        sock.parent.mkdir(parents=True, exist_ok=True)
        sock.write_text("")                             # -O exit: "Connection refused", rc 255
        await t.reset_master()
        assert wedged.wait(timeout=5) == -15 and not sock.exists()
    finally:
        wedged.kill()


async def test_detached_ssh_with_skips_processes_that_keep_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.transport import openssh

    out = (b"  101 ?        ssh -t -o ControlMaster=auto -o ControlPath=/m/cm -- k2\n"
           b"  102 pts/3    ssh -t -o ControlMaster=auto -o ControlPath=/m/cm -- k2\n"      # the live session
           b"  103 ??       /usr/bin/ssh -t -o ControlMaster=auto -o ControlPath=/m/cm -- k2 htop\n"
           b"  104 ?        ssh -t -o ControlMaster=auto -o ControlPath=/m/cmX -- k2\n"
           b"  105 ?        vim -o ControlMaster=auto -o ControlPath=/m/cm -- k2\n")

    async def fake_run(argv: list[str], **_: object) -> RunResult:
        return RunResult(0, out, b"", 0.0, False)

    monkeypatch.setattr(openssh.spawn, "run", fake_run)
    assert await openssh._detached_ssh_with("-o ControlMaster=auto -o ControlPath=/m/cm") == [101, 103]
