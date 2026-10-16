"""[WP1] Mux policy from `ssh -G`, ProxyJump rewrite, managed socket dir (§4.1-4.2).

Pure functions: given the resolved `SshSettings` and the host config, decide the mux mode and produce the exact option
lists the OpenSSH transport splices into its command lines.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from runfleet.config import HostConfig
from runfleet.models import MuxMode
from runfleet.sshconfig import SshSettings
from runfleet.util import platform

MANAGED_PERSIST = "15m"
# Why sharing is off on Windows whatever the config says (doctor, host detail; DESIGN §4.1)
WINDOWS_NO_MUX = ("Windows OpenSSH has no ControlMaster; every poll logs in again — keys/agent work, Duo hosts: "
                  "use WSL until the session transport")


@dataclass(frozen=True, slots=True)
class MuxPlan:
    """Resolved mux mode and the option lists derived from it (§4.1 table, §4.2 command lines).

    `socket` is the ControlPath with every `%` token already expanded by `ssh -G` (None if that failed or for "off").
    When it is known, every command that touches the master (create, check/exit, probe, stream, connect, shells)
    names it literally, so options that change OpenSSH's `%C` hash (ProxyJump vs. the ProxyCommand rewrite, User,
    Port, HostName from `ssh_options`) can never make two commands look at different sockets.
    """
    mode: MuxMode                        # "user" | "managed" | "off"
    control_path: str | None             # the socket path in use ("…/cm-%C" pattern or the literal socket)
    probe_opts: tuple[str, ...]          # added to BASE (probe, stream)
    imux_opts: tuple[str, ...]           # interactive sessions
    check_opts: tuple[str, ...]          # `-O check|exit`
    path_opts: tuple[str, ...] = ()      # `-o ControlPath=…` alone (master creation, connect)
    socket: str | None = None            # literal socket path, `%` tokens expanded

    @property
    def uses_master(self) -> bool:
        # `mux = "user"` without a ControlPath cannot multiplex: a `-fN -o ControlMaster=yes` there would only stay up
        # as a plain background login, so nothing creates or checks a master and the user's config applies as is
        return self.mode == "managed" or (self.mode == "user" and self.control_path is not None)


def managed_control_path(mux_dir: Path) -> str:
    """`<mux_dir>/cm-%C` as a ControlPath pattern: a `%` in the directory itself is escaped."""
    return str(mux_dir).replace("%", "%%") + "/cm-%C"


_TOKEN = re.compile(r"%[%A-Za-z]")


def is_expanded(path: str) -> bool:
    """True if `path` carries no OpenSSH `%` token any more (as printed by `ssh -G`)."""
    return _TOKEN.search(path) is None


def control_path_option(path: str, *, literal: bool) -> str:
    """`ControlPath=<path>` for `-o`: a literal path has `%` escaped as `%%`; whitespace is double-quoted (OpenSSH
    splits `-o` values like config lines)."""
    value = path.replace("%", "%%") if literal else path
    if any(c.isspace() for c in value):
        value = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return f"ControlPath={value}"


def decide(settings: SshSettings, host: HostConfig, mux_dir: Path, *, socket: str | None = None) -> MuxPlan:
    """§4.1: user ControlPath (with ControlPersist) -> "user"; none/ControlPersist=no -> "managed"; config overrides.

    `socket`: the expanded socket path (`OpenSSHTransport.resolve` gets it from `ssh -G`); without it the managed
    plan names the `cm-%C` pattern and the user plan the path `ssh -G` printed.

    Windows: always "off", whatever `mux` or ~/.ssh/config say (`WINDOWS_NO_MUX`): its ssh has no ControlMaster, and
    a master started with `-fN` never backgrounds there (the command would hang).
    """
    mode = "off" if platform.IS_WINDOWS else host.mux
    if mode == "auto":
        persist_no = settings.controlpersist.strip().lower() in ("no", "false", "0")
        mode = "user" if settings.controlpath is not None and not persist_no else "managed"
    if mode == "off":
        return MuxPlan("off", None, ("-o", "ControlMaster=no", "-o", "ControlPath=none"),
                       ("-o", "ControlPath=none"), ())
    if mode == "user":
        path = socket or settings.controlpath
        if path is None:          # `mux = "user"` forced without a ControlPath: the user's config applies as is
            return MuxPlan("user", None, ("-o", "ControlMaster=no"), (), ("-o", "ControlMaster=no"))
        literal = socket is not None or is_expanded(path)
        cp = ("-o", control_path_option(path, literal=literal))
        return MuxPlan("user", path, ("-o", "ControlMaster=no", *cp), cp, ("-o", "ControlMaster=no", *cp), cp,
                       path if literal else None)
    path = socket or managed_control_path(mux_dir)
    cp = ("-o", control_path_option(path, literal=socket is not None))
    opts = ("-o", "ControlMaster=no", *cp)
    # Interactive sessions (shell, pager) become the master when none is up: on a Duo host the login the user just
    # completed then also serves the BatchMode probes, instead of leaving them at "needs auth".
    imux = ("-o", "ControlMaster=auto", *cp, "-o", f"ControlPersist={MANAGED_PERSIST}")
    return MuxPlan("managed", path, opts, imux, opts, cp, socket)


# ---------------------------------------------------------------- ProxyJump rewrite
@dataclass(frozen=True, slots=True)
class Hop:
    host: str
    user: str | None = None
    port: int | None = None

    def spec(self) -> str:
        """`[user@]host[:port]` for `-J`."""
        h = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.user + '@' if self.user else ''}{h}{':' + str(self.port) if self.port else ''}"


# OpenSSH splits a hop at its LAST "@" (users like "first.last@corp.com" are legal), hence the greedy user group.
_HOP_RE = re.compile(r"^(?:ssh://)?(?:(?P<user>.+)@)?(?:\[(?P<v6>[^\]]+)\]|(?P<host>[^:@\[\]]+))(?::(?P<port>\d+))?$")


def parse_jump(proxyjump: str) -> list[Hop]:
    """`ssh -G` proxyjump value ("u@h1:2222,h2", "ssh://u@h:22") -> hops in order. Raises ValueError."""
    hops = []
    for part in proxyjump.split(","):
        m = _HOP_RE.match(part.strip())
        if m is None:
            raise ValueError(f"cannot parse ProxyJump hop {part!r}")
        hops.append(Hop(host=m.group("v6") or m.group("host"), user=m.group("user"),
                        port=int(m.group("port")) if m.group("port") else None))
    return hops


def needs_rewrite(settings: SshSettings) -> bool:
    return settings.proxyjump is not None and settings.proxycommand is None


def jump_hosts(settings: SshSettings) -> tuple[str, ...]:
    """Names to look for in ssh stderr when classifying jump failures (§9 row 5)."""
    if settings.proxyjump is None:
        return ()
    try:
        return tuple(h.host for h in parse_jump(settings.proxyjump))
    except ValueError:
        return ()


def _pct(s: str) -> str:
    """Escape `%` so ssh does not expand it as a ProxyCommand token."""
    return s.replace("%", "%%")


def jump_rewrite(settings: SshSettings, *, batch: bool, log: Path | None = None,
                 ssh: str = "ssh") -> tuple[str, ...]:
    """`-o ProxyCommand=ssh -T [-o BatchMode=yes] -o ConnectTimeout=10 -o LogLevel=ERROR [-E log] [-p P] [-l U]
    [-J H1,…,Hn-1] -W '[%h]:%p' Hn` when ssh -G reports a ProxyJump and no ProxyCommand; else ().

    OpenSSH does not pass command-line `-o` options to the ProxyJump hop, so BatchMode would not reach it. `batch=False`
    is the interactive form (connect, shells): the hop may prompt in the real terminal, but logs to `log`.

    Windows: OpenSSH there starts the ProxyCommand without a shell (CreateProcess), so the command is
    `"<absolute ssh.exe>" <args quoted for the MSVC runtime> -W "[%h]:%p" Hn`: `ssh` is the client's absolute path
    (the caller resolves it); a leading quoted program is passed through as is, and its arguments are split by the
    hop's C runtime, which `subprocess.list2cmdline` quotes for.
    """
    if not needs_rewrite(settings):
        return ()
    assert settings.proxyjump is not None
    hops = parse_jump(settings.proxyjump)
    last, earlier = hops[-1], hops[:-1]
    words = ["-T"]
    if batch:
        words += ["-o", "BatchMode=yes"]
    words += ["-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR"]
    if log is not None:
        words += ["-E", str(log)]
    if last.port:
        words += ["-p", str(last.port)]
    if last.user:
        words += ["-l", last.user]
    if earlier:
        words += ["-J", ",".join(h.spec() for h in earlier)]
    if platform.IS_WINDOWS:
        program = '"' + _pct(ssh) + '"'
        cmd = " ".join([program, *(_pct(subprocess.list2cmdline([w])) for w in words)])
        cmd += ' -W "[%h]:%p" ' + _pct(subprocess.list2cmdline([last.host]))
        return ("-o", f"ProxyCommand={cmd}")
    cmd = " ".join(_pct(shlex.quote(w)) for w in [ssh, *words]) + " -W '[%h]:%p' " + _pct(shlex.quote(last.host))
    return ("-o", f"ProxyCommand={cmd}")
