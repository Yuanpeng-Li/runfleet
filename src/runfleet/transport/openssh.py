"""[WP1] OpenSSHTransport: probe, stream, argv builders, ensure_master (§4.1-4.2)."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from runfleet import paths
from runfleet.config import HostConfig
from runfleet.errors import RunFleetError
from runfleet.models import ErrorKind, HostError, MuxMode
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, StreamSpec
from runfleet.sshconfig import SshSettings, resolve
from runfleet.transport import classify, mux, spawn
from runfleet.transport.mux import MuxPlan
from runfleet.transport.base import ProbeResult, RunResult, StreamHandle

log = logging.getLogger("runfleet.transport.openssh")

CHECK_TIMEOUT = 3.0
ENSURE_INTERVAL = 30.0          # at most one master attempt per 30 s per host (§4.1)
# Option order on every command line (OpenSSH keeps the FIRST value of an option): the structural options RunFleet
# depends on (BatchMode, ControlMaster/ControlPath/ControlPersist, the ProxyJump rewrite) come first, then the config's
# `ssh_options`, then RunFleet's tunable defaults, which `ssh_options` may therefore override (e.g. ConnectTimeout=30).
BASE_LEAD = ("-T", "-o", "BatchMode=yes")
BASE_DEFAULTS = ("-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                 "-o", "LogLevel=ERROR")
BASE_OPTS = BASE_LEAD + BASE_DEFAULTS
MANAGED_PERSIST_OPTS = ("-o", f"ControlPersist={mux.MANAGED_PERSIST}")
MANAGED_KEEPALIVE_OPTS = ("-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
MANAGED_MASTER_OPTS = MANAGED_PERSIST_OPTS + MANAGED_KEEPALIVE_OPTS
ALREADY_EXISTS = "already exists, disabling multiplexing"
PATH_TOO_LONG = "too long for Unix domain socket"


class OpenSSHTransport:
    """System OpenSSH as asyncio subprocesses, multiplexed over a user or RunFleet-managed ControlMaster."""

    kind: Literal["openssh", "local", "fake", "demo"] = "openssh"

    def __init__(self, host: HostConfig, *, ssh: str = "ssh", mux_dir: Path | None = None,
                 log_path: Path | None = None, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.host = host.name
        self.config = host
        self._ssh = ssh
        self._mux_dir = mux_dir
        self._log_path = log_path if log_path is not None else paths.ssh_log(host.name)
        self._monotonic = monotonic        # only throttles master attempts; injectable for tests
        self.settings: SshSettings | None = None
        self.plan: MuxPlan | None = None
        self.rendered_control_path: str | None = None   # the socket with every %-token expanded (all commands)
        self._last_ensure: float | None = None
        self._last_ensure_error: HostError | None = None
        self._connect_pending = False
        self._last_check_err = ""

    # ------------------------------------------------------------ properties
    @property
    def mux(self) -> MuxMode:
        return self.plan.mode if self.plan is not None else "none"

    @property
    def ssh_log(self) -> str | None:
        return str(self._log_path)

    @property
    def alias(self) -> str:
        return self.config.ssh

    def mux_dir(self) -> Path:
        return self._mux_dir if self._mux_dir is not None else paths.mux_dir()

    # ------------------------------------------------------------ resolution
    async def resolve(self) -> HostError | None:
        """`ssh -G` once per session (with the config's `ssh_options`, so user/host/port are the ones RunFleet will
        use), then the mux decision, then the socket path with every `%` token expanded. Idempotent."""
        if self.settings is not None and self.plan is not None:
            return None
        user_opts = self._user_opts()
        try:
            settings = await resolve(self.alias, ssh=self._ssh, extra=user_opts)
        except RunFleetError as e:
            return HostError(ErrorKind.SSH, str(e)[: classify.SUMMARY_LIMIT], str(e))
        try:
            mux.parse_jump(settings.proxyjump) if settings.proxyjump else None
        except ValueError as e:
            return HostError(ErrorKind.JUMP, str(e)[: classify.SUMMARY_LIMIT], str(e))
        plan = mux.decide(settings, self.config, self.mux_dir())
        socket: str | None = None
        if plan.mode == "managed":
            try:
                paths.ensure_private_dir(self.mux_dir())
            except OSError as e:
                return HostError(ErrorKind.MUX, "managed socket directory unusable", f"{self.mux_dir()}: {e}")
            with contextlib.suppress(RunFleetError):
                assert plan.control_path is not None
                expanded = await resolve(self.alias, ssh=self._ssh, extra=(
                    "-o", mux.control_path_option(plan.control_path, literal=False), *user_opts))
                if expanded.controlpath and mux.is_expanded(expanded.controlpath):
                    socket = expanded.controlpath
        elif plan.mode == "user" and plan.socket is not None:
            socket = plan.socket            # `ssh -G` already expanded the user's own ControlPath (no rewrite)
        elif plan.mode == "user" and plan.control_path is not None:
            # A `%` left in the path `ssh -G` printed: an unexpanded pattern (an older ssh) or a literal `%%` in the
            # user's config. A probe pattern tells which: an ssh that expands prints `%%` as `%`.
            with contextlib.suppress(RunFleetError):
                probe = await resolve(self.alias, ssh=self._ssh, extra=("-o", "ControlPath=/rf-%%h", *user_opts))
                if probe.controlpath == "/rf-%h":
                    socket = plan.control_path
        if socket is not None:
            plan = mux.decide(settings, self.config, self.mux_dir(), socket=socket)
        self.settings, self.plan, self.rendered_control_path = settings, plan, socket
        return None

    def _require(self) -> tuple[SshSettings, MuxPlan]:
        if self.settings is None or self.plan is None:
            raise RunFleetError(f"{self.host}: transport not prepared")
        return self.settings, self.plan

    # ------------------------------------------------------------ argv builders (§4.2, exact)
    def _jump(self, *, batch: bool, hop_log: bool = False) -> tuple[str, ...]:
        """The ProxyJump rewrite. Interactive forms and master creation give the hop `-E <log>`, so its own
        auth/host-key failures land in the log `ensure_master`/connect classify (a backgrounded hop's stderr is
        otherwise lost)."""
        if self.settings is None:          # interactive argv before resolution: the user's config applies as is
            return ()
        log = self._log_path if (hop_log or not batch) else None
        return mux.jump_rewrite(self.settings, batch=batch, log=log)

    def _user_opts(self) -> tuple[str, ...]:
        return tuple(w for opt in self.config.ssh_options for w in ("-o", opt))

    def base_argv(self) -> list[str]:
        _, plan = self._require()
        return [self._ssh, *BASE_LEAD, *plan.probe_opts, *self._jump(batch=True), *self._user_opts(),
                *BASE_DEFAULTS, "--", self.alias]

    def probe_argv(self) -> list[str]:
        remote = "exec sh -s" if self.config.remote_shell == "sh" else 'exec bash -l -c "exec sh -s"'
        return [*self.base_argv(), remote]

    def stream_argv(self, cmd: str) -> list[str]:
        return [*self.base_argv(), "exec sh -c " + shlex.quote(cmd)]

    def check_argv(self, op: Literal["check", "exit"] = "check") -> list[str]:
        _, plan = self._require()
        return [self._ssh, "-O", op, *plan.check_opts, *self._jump(batch=True), *self._user_opts(), "--",
                self.alias]

    def master_argv(self) -> list[str]:
        """BatchMode master creation. User mode carries only the user's own options plus the pinned socket path
        (§4.1, Appendix D #1)."""
        _, plan = self._require()
        managed = plan.mode == "managed"
        return [self._ssh, "-fN", "-o", "BatchMode=yes", "-o", "ControlMaster=yes", *plan.path_opts,
                *(MANAGED_PERSIST_OPTS if managed else ()), *self._jump(batch=True, hop_log=True),
                *self._user_opts(), *(MANAGED_KEEPALIVE_OPTS if managed else ()),
                "-o", f"ConnectTimeout={int(self.config.connect_timeout)}", "-o", "LogLevel=ERROR",
                "-E", str(self._log_path), "--", self.alias]

    def interactive_argv(self, remote_cmd: str | None) -> list[str]:
        argv = [self._ssh, "-t", *(self.plan.imux_opts if self.plan is not None else ())]
        if self.config.forward_agent:
            argv.append("-A")
        argv += [*self._jump(batch=False), *self._user_opts(), "--", self.alias]
        return argv + ([remote_cmd] if remote_cmd else [])

    def connect_argv(self) -> list[str] | None:
        """`ssh -fN` (interactive: prompts in the real terminal). Call `before_connect()` first: it resolves the
        mux plan (without it the user form is the only safe guess) and clears a stale socket."""
        mode = self.plan.mode if self.plan is not None else "user"
        log = ("-o", "LogLevel=ERROR", "-E", str(self._log_path))
        if mode == "off" or (self.plan is not None and not self.plan.uses_master):
            return [self._ssh, "-o", "ControlPath=none", *self._jump(batch=False), *self._user_opts(),
                    "-o", "ConnectTimeout=30", *log, "--", self.alias, "true"]   # just an auth check
        managed = mode == "managed"
        path = self.plan.path_opts if self.plan is not None else ()
        return [self._ssh, "-fN", "-o", "ControlMaster=yes", *path, *(MANAGED_PERSIST_OPTS if managed else ()),
                *self._jump(batch=False), *self._user_opts(), *(MANAGED_KEEPALIVE_OPTS if managed else ()),
                *log, "--", self.alias]

    # ------------------------------------------------------------ master handling
    async def prepare(self, *, force: bool = False) -> HostError | None:
        """Resolve and ensure the master. `force`: a user-initiated retry (manual refresh, after connect) bypasses
        the 30 s master-attempt throttle."""
        err = await self.resolve()
        if err is not None:
            return err
        _, plan = self._require()
        if self._connect_pending:
            self._connect_pending = False
            await self._reap_strays(self.connect_argv())
        if not plan.uses_master:
            return None
        return await self.ensure_master(force=force)

    async def before_connect(self) -> bool:
        """Before the interactive `connect_argv()`: resolve, and clear a stale socket so `ControlMaster=yes` can bind.
        Returns True if a working master is already up (connecting again would only start a second, non-mux login)."""
        if await self.resolve() is not None or self.plan is None:
            return False
        self._connect_pending = self.plan.uses_master
        if not self.plan.uses_master:
            return False
        if await self.check_master():
            self._connect_pending = False
            return True
        if self.plan.mode == "managed" and self._socket_exists() and not self._stale_check():
            await self.reset_master()          # a wedged managed master would make the new one fail to bind
        else:
            self._clear_stale_socket()
        self._truncate_log()                     # truncated by each connect/ensure (§4.2)
        return False

    def _truncate_log(self) -> None:
        """Empty the `-E` log (0600 in a 0700 dir: it names hosts, users and jump paths)."""
        paths.ensure_private_dir(self._log_path.parent)
        paths.write_private(self._log_path, b"")

    async def _check(self) -> tuple[bool, str]:
        if self.plan is None or not self.plan.uses_master:
            return True, ""
        try:
            res = await spawn.run(self.check_argv(), timeout=CHECK_TIMEOUT)
        except spawn.SpawnError as e:
            return False, str(e)
        text = res.stderr.decode("utf-8", errors="replace")
        return res.rc == 0, "timed out" if res.timed_out else text

    async def check_master(self) -> bool:
        ok, self._last_check_err = await self._check()
        return ok

    def _socket_exists(self) -> bool:
        return self.rendered_control_path is not None and os.path.lexists(self.rendered_control_path)

    def _stale_check(self) -> bool:
        """The last `-O check` found a socket file nobody listens on (a master killed with -9, OOM, a crash)."""
        return "Connection refused" in self._last_check_err

    def _may_unlink_stale(self) -> bool:
        """Managed sockets are ours. At a user ControlPath we only do what the user's own `ssh` would do: OpenSSH
        unlinks a refused socket itself when ControlMaster is auto/autoask (never with yes/no/ask)."""
        if self.plan is None:
            return False
        if self.plan.mode == "managed":
            return True
        cm = (self.settings.controlmaster if self.settings is not None else "").strip().lower()
        return self.plan.mode == "user" and cm in ("auto", "autoask")

    def _clear_stale_socket(self) -> HostError | None:
        """Called after a failed check. Returns an error if something sits at the socket that we must not replace."""
        if not self._socket_exists():
            return None
        path = self.rendered_control_path
        assert path is not None
        if self._stale_check() and self._may_unlink_stale():
            log.info("%s: removing stale control socket %s", self.host, path)
            with contextlib.suppress(OSError):
                os.unlink(path)
            return None
        if self._stale_check():
            return HostError(ErrorKind.MUX, "stale ssh control socket", f"{path} refuses connections; remove it "
                             "(or press c to log in again)")
        return HostError(ErrorKind.MUX, "ssh master not answering", f"{path} exists but `ssh -O check` failed: "
                         f"{self._last_check_err.strip()}"[: classify.DETAIL_LIMIT])

    async def ensure_master(self, *, force: bool = False) -> HostError | None:
        """§4.1: `-O check`, else create the master (BatchMode, waiting for the `-fN` process to exit), then check
        again: `ControlMaster=yes` that could not bind the socket still exits 0 and stays up as a plain login."""
        if await self.check_master():
            self._last_ensure_error = None
            return None
        now = self._monotonic()
        if not force and self._last_ensure is not None and now - self._last_ensure < ENSURE_INTERVAL:
            return self._last_ensure_error or HostError(ErrorKind.MUX, "ssh master not up yet")
        self._last_ensure = now
        blocked = self._clear_stale_socket()
        if blocked is not None:
            self._last_ensure_error = blocked
            return blocked
        self._truncate_log()                     # truncated by each connect/ensure (§4.2)
        argv = self.master_argv()
        try:
            res = await spawn.run(argv, timeout=self.config.connect_timeout + 5, capture=False)
        except spawn.SpawnError as e:
            self._last_ensure_error = HostError(ErrorKind.SSH, str(e)[: classify.SUMMARY_LIMIT], str(e))
            return self._last_ensure_error
        text = _read_tail(self._log_path)
        if res.rc == 0:
            if await self.check_master():
                if ALREADY_EXISTS in text:       # another creator won the race; ours stays up as a plain login
                    await self._reap_strays(argv)
                self._last_ensure_error = None
                return None
            await self._reap_strays(argv)
            why = ("another process holds the control socket" if ALREADY_EXISTS in text
                   else "ssh master started but does not answer")
            self._last_ensure_error = HostError(ErrorKind.MUX, why, text or self._last_check_err)
            return self._last_ensure_error
        if PATH_TOO_LONG in text:
            err: HostError | None = HostError(ErrorKind.MUX, "control socket path too long",
                                              f"{text}\nset XDG_RUNTIME_DIR to a short directory or use mux = "
                                              "\"managed\" (RunFleet keeps its own sockets short)")
        else:
            err = classify.classify_ssh(res.rc, text, timed_out=res.timed_out,
                                        timeout_s=self.config.connect_timeout, jumps=self._jumps())
        if err is None:
            err = HostError(ErrorKind.SSH, f"ssh master failed (rc {res.rc})", text)
        self._last_ensure_error = err
        return err

    async def _reap_strays(self, argv: list[str] | None) -> None:
        """Terminate backgrounded `ssh -fN` processes started with exactly `argv` that did not become the master
        (their log says "ControlSocket … already exists, disabling multiplexing"). The forked child runs in a new
        session (daemon(3)), so it is found by its command line, not by our process group."""
        if argv is None or ALREADY_EXISTS not in _read_tail(self._log_path):
            return
        if self.plan is not None and self.plan.uses_master and await self.check_master():
            master = _master_pid(self._last_check_err)
        else:
            master = None
        for pid in await _pids_with_args(argv):
            if pid != master:
                log.warning("%s: terminating stray ssh %d (control socket already taken)", self.host, pid)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGTERM)

    async def reset_master(self) -> None:
        if self.plan is None or self.plan.mode != "managed":
            return
        res: RunResult | None = None
        with contextlib.suppress(spawn.SpawnError):
            res = await spawn.run(self.check_argv("exit"), timeout=CHECK_TIMEOUT)
        if res is None or res.rc != 0:
            # a hung master did not take `-O exit`: once its socket is gone nothing could reach (or stop) it
            pids = await _pids_with_args(self.master_argv()) + await _pids_with_args(self.connect_argv() or [])
            # a ControlPersist interactive session (shell, pager) may have forked the master: its daemon has no tty
            pids += await _detached_ssh_with(" ".join(self.plan.imux_opts))
            for pid in dict.fromkeys(pids):
                log.warning("%s: terminating unresponsive ssh master %d", self.host, pid)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGTERM)
        if self.rendered_control_path:
            with contextlib.suppress(OSError):
                os.unlink(self.rendered_control_path)
        self._last_ensure = None

    # ------------------------------------------------------------ probes and streams
    def _jumps(self) -> tuple[str, ...]:
        return mux.jump_hosts(self.settings) if self.settings is not None else ()

    def _classify(self, timeout: float) -> spawn.ProbeClassifier:
        def run_classify(res: RunResult, parsed: framing.ParsedProbe) -> HostError | None:
            stderr = res.stderr.decode("utf-8", errors="replace")
            err = classify.classify_ssh(res.rc, stderr, timed_out=res.timed_out, timeout_s=timeout,
                                        jumps=self._jumps(), section_ended=parsed.ended > 0)
            if err is None and parsed.markers == 0 and not res.timed_out:
                err = classify.remote_shell_error(res.rc, stderr)
            return err
        return run_classify

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        err = await self.resolve()
        if err is not None:
            return spawn.failed_probe(script, err)
        return await spawn.run_probe(self.probe_argv(), script, timeout=timeout, classify=self._classify(timeout),
                                     label=self.host)

    async def stream(self, spec: StreamSpec) -> StreamHandle:
        err = await self.resolve()
        if err is not None:
            raise RunFleetError(err.summary)
        return await spawn.ProcessStream.start(self.stream_argv(spec.cmd))

    async def close(self) -> None:
        """Masters are left running; ControlPersist expires them (§4.1)."""

    # ------------------------------------------------------------ port forwards (local only, through the master)
    def forward_argv(self, op: Literal["forward", "cancel"], local_port: int, node: str,
                     remote_port: int) -> list[str] | None:
        """`ssh -O forward|cancel -L 127.0.0.1:<lport>:<node>:<rport>` against the master; None without one."""
        if self.plan is None or not self.plan.uses_master:
            return None
        spec = f"127.0.0.1:{local_port}:{node}:{remote_port}"
        return [self._ssh, "-O", op, "-L", spec, *self.plan.check_opts, *self._jump(batch=True), *self._user_opts(),
                "--", self.alias]

    def forward_blocker(self) -> str | None:
        if self.plan is None:
            return "not connected yet"
        if not self.plan.uses_master:
            return 'no ssh master to forward through (mux = "user" without a ControlPath, or mux = "off")'
        return None

    async def _forward_op(self, op: Literal["forward", "cancel"], local_port: int, node: str,
                          remote_port: int) -> HostError | None:
        argv = self.forward_argv(op, local_port, node, remote_port)
        if argv is None:
            return HostError(ErrorKind.MUX, self.forward_blocker() or "no ssh master")
        try:
            res = await spawn.run(argv, timeout=CHECK_TIMEOUT)
        except spawn.SpawnError as e:
            return HostError(ErrorKind.SSH, str(e)[: classify.SUMMARY_LIMIT], str(e))
        if res.rc == 0:
            return None
        text = res.stderr.decode("utf-8", errors="replace").strip()
        # "mux_client_forward: forwarding request failed: Port forwarding failed" -> its last part
        first = "timed out" if res.timed_out else (text.splitlines()[0].rsplit(": ", 1)[-1] if text else f"rc {res.rc}")
        return HostError(ErrorKind.MUX, f"port {op} failed: {first}"[: classify.SUMMARY_LIMIT], text)

    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return await self._forward_op("forward", local_port, node, remote_port)

    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return await self._forward_op("cancel", local_port, node, remote_port)


def _master_pid(check_stderr: str) -> int | None:
    m = re.search(r"pid=(\d+)", check_stderr)
    return int(m.group(1)) if m else None


async def _pids_with_args(argv: list[str]) -> list[int]:
    """Our own ssh processes started with `argv` (`ps -ww`, Linux and macOS). Matched on everything after argv[0]
    (it carries `-fN` and the literal ControlPath), so an `ssh` wrapper that execs the real binary still matches."""
    tail = " ".join(argv[1:])
    try:
        res = await spawn.run(["ps", "-ww", "-o", "pid=,args=", "-U", str(os.getuid())], timeout=CHECK_TIMEOUT)
    except spawn.SpawnError:
        return []
    out = []
    for line in res.stdout.decode("utf-8", errors="replace").splitlines():
        pid, _, args = line.strip().partition(" ")
        args = args.strip()
        prog = args.partition(" ")[0]
        if (pid.isdigit() and int(pid) != os.getpid() and "ssh" in os.path.basename(prog)
                and (args == " ".join(argv) or args.endswith(" " + tail))):
            out.append(int(pid))
    return out


async def _detached_ssh_with(fragment: str) -> list[int]:
    """Our own ssh processes without a controlling terminal whose command line contains `fragment`: the background
    master an interactive `ControlMaster=auto` + `ControlPersist` session forked (the session itself keeps its tty)."""
    try:
        res = await spawn.run(["ps", "-ww", "-o", "pid=,tty=,args=", "-U", str(os.getuid())], timeout=CHECK_TIMEOUT)
    except spawn.SpawnError:
        return []
    out = []
    for line in res.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or parts[1].strip("?"):
            continue
        pid, args = int(parts[0]), parts[2]
        if pid != os.getpid() and "ssh" in os.path.basename(args.partition(" ")[0]) and f" {fragment} " in args:
            out.append(pid)
    return out


def _read_tail(path: Path, limit: int = 4096) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")
