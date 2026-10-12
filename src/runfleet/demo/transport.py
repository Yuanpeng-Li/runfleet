"""[WP4] DemoTransport: the `Transport` protocol over the simulated world (DESIGN.md §3.5, §10.6).

It plays a remote host: every `Section` of a probe is answered by name and `meta` (Appendix A) with the exact output
the real command would print (`render.py`), framed per §4.3 into `RunResult.stdout`, with the outcome derived the
same way the real framing parser derives it. Actions are applied to the world; `log.tail` streams follow the
simulated files. Faults injected with `DemoWorld.inject` turn into the same transport and section failures the real
fleet produces.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import Literal

from runfleet.config import HostConfig
from runfleet.demo import render
from runfleet.demo.render import Out, meta_list
from runfleet.demo.world import DemoWorld, SimCluster, SimHost, SimWorkstation, stable_hash
from runfleet.models import ErrorKind, HostError, MuxMode
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, SectionResult, StreamSpec
from runfleet.transport.base import ProbeResult, RunResult, StreamHandle

log = logging.getLogger("runfleet.demo.transport")

STDERR_CAP = 4096
CHUNK = 65536
_DBD_SECTIONS = frozenset({"slurm.sacct", "slurm.sacct_ids", "slurm.steps", "slurm.gputres", "slurm.assoc"})
_SHELL_SCRIPT = (
    "import sys\n"
    "cmd = ' '.join(a for a in sys.argv[2:] if a)\n"
    "print(f'demo: this would run `ssh -t {sys.argv[1]}' + (f' {cmd}' if cmd else '') + '` (simulated host)')\n"
    "try:\n    input('press Enter to return to RunFleet ')\nexcept EOFError:\n    pass\n")

Handler = Callable[["DemoTransport", SimHost, Section, float], Out]


def derive_outcome(rc: int | None, stdout: str, stderr: str) -> SectionOutcome:
    """§4.3 outcome rules (the missing-tool case is decided before a command runs)."""
    if rc is None:
        return SectionOutcome.NOT_RUN
    if rc in (124, 137):
        return SectionOutcome.TIMEOUT
    if rc == 0:
        if stdout:
            return SectionOutcome.OK
        return SectionOutcome.STDERR_ONLY if stderr else SectionOutcome.EMPTY
    return SectionOutcome.FAILED


def _meta_int(meta: Mapping[str, object], key: str, default: int) -> int:
    value = meta.get(key, default)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return default


def _meta_str(meta: Mapping[str, object], key: str, default: str) -> str:
    value = meta.get(key, default)
    return value if isinstance(value, str) else default


def _not_slurm(name: str) -> Out:
    return Out("", 127, f"sh: 1: {name}: not found")


# ====================================================================================================== handlers
def _cluster(h: SimHost, tool: str) -> SimCluster | Out:
    return h if isinstance(h, SimCluster) else _not_slurm(tool)


def _ws(h: SimHost, tool: str) -> SimWorkstation | Out:
    return h if isinstance(h, SimWorkstation) else _not_slurm(tool)


def _h_squeue(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "squeue")
    if isinstance(c, Out):
        return c
    users = meta_list(s.meta, "users")
    if _meta_str(s.meta, "format", "O") == "legacy":
        return render.squeue_legacy(c, now, users=users)
    return render.squeue(c, now, users=users, sep=_meta_str(s.meta, "sep", "\x1f"))


def _h_sacct(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sacct")
    if isinstance(c, Out):
        return c
    fields = meta_list(s.meta, "fields") or render.SACCT_FIELDS
    return render.sacct_window(c, now, since_s=_meta_int(s.meta, "since_s", 3 * 86400), fields=fields,
                               sep=_meta_str(s.meta, "sep", "\x1f"))


def _h_sacct_ids(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sacct")
    if isinstance(c, Out):
        return c
    fields = meta_list(s.meta, "fields") or render.SACCT_FIELDS
    return render.sacct_ids(c, now, ids_raw=meta_list(s.meta, "ids_raw"), fields=fields,
                            sep=_meta_str(s.meta, "sep", "\x1f"))


def _h_scontrol(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "scontrol")
    return c if isinstance(c, Out) else render.scontrol_loop(c, now, ids_raw=meta_list(s.meta, "ids_raw"))


def _h_sstat(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sstat")
    return c if isinstance(c, Out) else render.sstat(c, now, ids_raw=meta_list(s.meta, "ids_raw"))


def _h_steps(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sacct")
    if isinstance(c, Out):
        return c
    fields = [f.split("%", 1)[0] for f in meta_list(s.meta, "fields")] or render.STEP_FIELDS   # + TotalCPU
    return render.sacct_steps(c, now, job_id_raw=_meta_str(s.meta, "job_id_raw", ""),
                              sep=_meta_str(s.meta, "sep", "\x1f"), fields=fields)


def _h_sinfo(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sinfo")
    if isinstance(c, Out):
        return c
    return render.sinfo_nodes(c, partitions=meta_list(s.meta, "partitions"), sep=_meta_str(s.meta, "sep", "\x1f"))


def _h_slurm_act(verb: str) -> Handler:
    def handler(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
        c = _cluster(h, "scancel" if verb == "cancel" else "scontrol")
        if isinstance(c, Out):
            return c
        rc, err = t.world.slurm_action(c.name, verb, meta_list(s.meta, "ids"), hold=bool(s.meta.get("hold")))
        return Out("", rc, err)
    return handler


def _h_logstat(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    return render.logstat(h, paths=meta_list(s.meta, "paths"))


def _h_procfs(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    return render.procfs_host(h, now)


def _h_nvidia_gpu(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "nvidia-smi")
    if isinstance(ws, Out):
        return ws
    return render.nvidia_gpu(ws, fields=meta_list(s.meta, "fields") or render.NVIDIA_GPU_FIELDS)


def _h_nvidia_apps(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "nvidia-smi")
    return ws if isinstance(ws, Out) else render.nvidia_apps(ws, now)


def _h_ps(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "ps")
    return ws if isinstance(ws, Out) else render.process_ps(ws, now)


def _h_fds(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "readlink")
    if isinstance(ws, Out):
        return ws
    return render.process_fds(ws, pids=[int(p) for p in meta_list(s.meta, "pids") if p.isdigit()])


def _h_exitscan(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    raw = s.meta.get("tasks", ())
    tasks: list[tuple[str, list[str]]] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                tasks.append((str(item[0]), [str(p) for p in item[1]]))
    return render.exitscan(h, tasks=tasks)


def _h_detail(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "ps")
    if isinstance(ws, Out):
        return ws
    pids = [int(p) for p in meta_list(s.meta, "pids") if p.isdigit()]
    return render.process_detail(ws, now, pids=pids, root=_meta_int(s.meta, "root", pids[0] if pids else 0))


def _h_kill(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "ps")
    if isinstance(ws, Out):
        return ws
    m = s.meta
    sig = _meta_str(m, "sig", "TERM")
    lines = t.world.signal_task(ws.name, _meta_int(m, "root", -1), _meta_int(m, "start", -1), sig,
                                _meta_int(m, "grace_s", 10), bool(m.get("escalate", sig != "INT")))
    return Out("".join(line + "\n" for line in lines))


def _h_peek(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    return render.log_peek(h, path=_meta_str(s.meta, "path", ""), nbytes=_meta_int(s.meta, "bytes", 4096))


def _h_fetch(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    return render.log_fetch(h, path=_meta_str(s.meta, "path", ""), nbytes=_meta_int(s.meta, "bytes", 4 << 20))


def _h_attach_node(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "ssh" if s.meta.get("mode") == "ssh" else "srun")
    return c if isinstance(c, Out) else render.attach_node(c, now, s.meta)


def _h_attach_audit(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "squeue")
    return c if isinstance(c, Out) else render.attach_audit(c, now, s.meta)


def _simple(fn: Callable[[SimCluster], Out], tool: str) -> Handler:
    def handler(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
        c = _cluster(h, tool)
        return c if isinstance(c, Out) else fn(c)
    return handler


def _h_gputres(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    c = _cluster(h, "sacct")
    return c if isinstance(c, Out) else render.sacct_gputres(c, now)


def _h_nvidia_version(t: DemoTransport, h: SimHost, s: Section, now: float) -> Out:
    ws = _ws(h, "nvidia-smi")
    return ws if isinstance(ws, Out) else render.nvidia_version(ws)


HANDLERS: dict[str, Handler] = {
    "detect.id": lambda t, h, s, now: render.detect_id(h, now),
    "detect.tools": lambda t, h, s, now: render.detect_tools(h),
    "slurm.version": _simple(render.sinfo_version, "sinfo"),
    "slurm.config": _simple(render.slurm_config, "scontrol"),
    "slurm.assoc": _simple(render.sacctmgr_assoc, "sacctmgr"),
    "slurm.sacctfmt": _simple(render.sacct_helpformat, "sacct"),
    "slurm.gputres": _h_gputres,
    "nvidia.version": _h_nvidia_version,
    "slurm.squeue": _h_squeue,
    "slurm.sacct": _h_sacct,
    "slurm.sacct_ids": _h_sacct_ids,
    "slurm.scontrol": _h_scontrol,
    "slurm.sstat": _h_sstat,
    "slurm.steps": _h_steps,
    "slurm.logstat": _h_logstat,
    "slurm.sinfo": _h_sinfo,
    "slurm.act.cancel": _h_slurm_act("cancel"),
    "slurm.act.hold": _h_slurm_act("hold"),
    "slurm.act.release": _h_slurm_act("release"),
    "slurm.act.requeue": _h_slurm_act("requeue"),
    "procfs.host": _h_procfs,
    "nvidia.gpu": _h_nvidia_gpu,
    "nvidia.apps": _h_nvidia_apps,
    "process.ps": _h_ps,
    "process.fds": _h_fds,
    "process.logstat": _h_logstat,
    "process.exitscan": _h_exitscan,
    "process.detail": _h_detail,
    "log.peek": _h_peek,
    "log.fetch": _h_fetch,
    "slurm.attach.audit": _h_attach_audit,
}


def handler_for(name: str) -> Handler | None:
    if name.startswith("process.kill."):
        return _h_kill
    if name.startswith("slurm.attach.node"):
        return _h_attach_node
    return HANDLERS.get(name)


# ====================================================================================================== transport
class DemoTransport:
    """`Transport` for one demo host. Construct as `lambda h: DemoTransport(h, world)` (§3.7)."""

    kind: Literal["openssh", "local", "fake", "demo"]

    def __init__(self, host: HostConfig, world: DemoWorld) -> None:
        self.host = host.name
        self.kind = "demo"
        self.config = host
        self.world = world
        self._rng = random.Random(world.seed + stable_hash(host.name))
        self._connect_requested = False

    # ------------------------------------------------------------------ protocol properties
    @property
    def mux(self) -> MuxMode:
        return "none"

    @property
    def ssh_log(self) -> str | None:
        return None

    # ------------------------------------------------------------------ connection
    def _sim(self) -> SimHost | None:
        return self.world.host(self.host)

    def _transport_error(self) -> tuple[HostError, str] | None:
        """(classified error, ssh stderr) when this host cannot be reached right now."""
        now = datetime.now(UTC)
        alias = self.config.ssh
        if self._sim() is None:
            msg = f"ssh: Could not resolve hostname {alias}: Name or service not known"
            return HostError(ErrorKind.DNS, f"could not resolve hostname {alias}", msg, None, now), msg
        if self.world.has_fault(self.host, "unreachable"):
            msg = f"ssh: connect to host {alias} port 22: Connection timed out"
            return HostError(ErrorKind.TIMEOUT, "connection timed out", msg, None, now), msg
        if self.world.has_fault(self.host, "auth"):
            sim = self._sim()
            user = sim.user if sim is not None else "user"
            msg = f"{user}@{alias}: Permission denied (publickey,gssapi-with-mic,keyboard-interactive)."
            return HostError(ErrorKind.AUTH, "permission denied (publickey,keyboard-interactive)", msg, None,
                             now), msg
        return None

    async def _latency(self, scale: float = 1.0) -> None:
        sim = self._sim()
        lo, hi = sim.latency if sim is not None else (0.05, 0.1)
        delay = self._rng.uniform(lo, hi) * scale * self.world.latency_scale
        if delay > 0:
            await asyncio.sleep(delay)

    async def prepare(self, *, force: bool = False) -> HostError | None:
        await self._latency()
        if self._connect_requested:
            # the user ran connect_argv() (the UI's `c`) just before after_connect() re-prepares: "logged in"
            self._connect_requested = False
            if self.world.has_fault(self.host, "auth"):
                self.world.clear(self.host, "auth")
        err = self._transport_error()
        return err[0] if err else None

    async def check_master(self) -> bool:
        return self._transport_error() is None

    # ------------------------------------------------------------------ probes
    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        started = asyncio.get_running_loop().time()
        err = self._transport_error()
        if err is not None:
            await self._latency(scale=10.0 if err[0].kind is ErrorKind.TIMEOUT else 1.0)
            sections = {s.name: SectionResult(s.name, SectionOutcome.NOT_RUN, None, "", "", None)
                        for s in script.sections}
            run = RunResult(255, b"", err[1].encode() + b"\n", asyncio.get_running_loop().time() - started, False)
            return ProbeResult(run=run, sections=sections, error=err[0])
        await self._latency()
        sim = self._sim()
        assert sim is not None
        now = self.world.sync()
        ordered = [s for s in script.sections if not s.late] + [s for s in script.sections if s.late]
        results: dict[str, SectionResult] = {}
        for section in ordered:
            results[section.name] = self._run_section(sim, section, now, script.purpose)
        framed = _frame(ordered, results, script.nonce)
        run = RunResult(0, framed.encode("utf-8"), b"", asyncio.get_running_loop().time() - started, False)
        return ProbeResult(run=run, sections={s.name: results[s.name] for s in script.sections}, error=None)

    def _run_section(self, sim: SimHost, section: Section, now: float, purpose: str) -> SectionResult:
        for tool in section.requires:
            if tool not in sim.tools:
                return SectionResult(section.name, SectionOutcome.MISSING, 127, "", f"missing:{tool}", 0)
        try:
            out = self._answer(sim, section, now, purpose)
        except Exception as exc:                 # a demo bug must look like a failing section, never crash a probe
            log.exception("demo: section %s on %s failed", section.name, self.host)
            out = Out("", 1, f"demo: internal error in {section.name}: {exc!r}")
        stdout = out.stdout[:section.max_stdout]
        stderr = out.stderr[:STDERR_CAP]
        ms = 0 if out.rc == 127 else self._rng.randint(8, 90)
        return SectionResult(section.name, derive_outcome(out.rc, stdout, stderr), out.rc, stdout, stderr, ms)

    def _answer(self, sim: SimHost, section: Section, now: float, purpose: str) -> Out:
        name = section.name
        world = self.world
        if world.has_fault(self.host, "slurmdbd_down") and name in _DBD_SECTIONS:
            tool = "sacctmgr" if name == "slurm.assoc" else "sacct"
            return Out("", 1, f"{tool}: error: Problem talking to the database: Connection refused")
        if world.has_fault(self.host, "nvidia_hang") and name in ("nvidia.gpu", "nvidia.apps"):
            hangs = world.nvidia_hangs.get(self.host, 0)
            world.nvidia_hangs[self.host] = hangs + 1
            if hangs == 0:
                return Out("", 124, "nvidia-smi did not answer within 6 s (abandoned)")
            return Out("", 125, "nvidia-smi already running (hung?)")
        handler = handler_for(name)
        if handler is None:
            log.warning("demo: no simulation for section %s", name)
            return Out("", 1, f"demo: no simulation for section {name!r}")
        out = handler(self, sim, section, now)
        if (world.has_fault(self.host, "garbage") and purpose not in ("detect", "action")
                and section.owner in ("slurm", "process", "nvidia", "procfs") and out.rc == 0):
            return Out(render.garbage(self._rng))
        return out

    # ------------------------------------------------------------------ streams
    async def stream(self, spec: StreamSpec) -> StreamHandle:
        return DemoStream(self, spec)

    # ------------------------------------------------------------------ interactive
    def interactive_argv(self, remote_cmd: str | None) -> list[str]:
        return [sys.executable, "-c", _SHELL_SCRIPT, self.config.ssh, remote_cmd or ""]

    async def before_connect(self) -> bool:
        return False

    def connect_argv(self) -> list[str] | None:
        self._connect_requested = True
        return [sys.executable, "-c", "print('demo: login ok')"]

    async def reset_master(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ port forwards (simulated: nothing listens)
    def forward_blocker(self) -> str | None:
        err = self._transport_error()
        return f"{self.host} is not connected ({err[0].summary})" if err else None

    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        await self._latency()
        err = self._transport_error()
        if err is not None:
            return err[0]
        self.world.note(f"{self.host}: demo forward localhost:{local_port} -> {node}:{remote_port} (simulated)")
        return None

    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None:
        return None


def _frame(sections: list[Section], results: Mapping[str, SectionResult], nonce: str) -> str:
    """The §4.3 framed stdout for these results (what the real prelude prints)."""
    rf = f"@@RF{nonce}"
    out: list[str] = []
    for s in sections:
        r = results[s.name]
        out.append(f"{rf} B {s.name}\n")
        if r.outcome is SectionOutcome.MISSING:
            out.append(f"\n{rf} E {s.name} 127 0 {r.stderr}\n")
            continue
        out.append(f"{r.stdout}\n{rf} E {s.name} {r.rc} {r.ms}\n")
        if r.stderr:
            out.append(f"{rf} X {s.name}\n{r.stderr}\n{rf} Y {s.name}\n")
    return "".join(out)


class DemoStream:
    """`log.tail`: the last `bytes` of the file, then whatever the world appends, until `close()` (tail -F
    semantics, including "cannot open … / has appeared" on stderr for a file that does not exist yet)."""

    def __init__(self, transport: DemoTransport, spec: StreamSpec) -> None:
        self._t = transport
        self._spec = spec
        self._closed = asyncio.Event()
        self._stderr = ""
        self._rc: int | None = None

    @property
    def stderr_text(self) -> str:
        return self._stderr[-STDERR_CAP:]

    @property
    def returncode(self) -> int | None:
        return self._rc

    async def close(self) -> None:
        self._closed.set()
        if self._rc is None:
            self._rc = 0

    def _fail(self, rc: int, msg: str) -> None:
        self._stderr += msg + "\n"
        self._rc = rc

    async def chunks(self) -> AsyncIterator[bytes]:
        world = self._t.world
        if self._spec.name != "log.tail":
            self._fail(1, f"demo: no simulation for stream {self._spec.name!r}")
            return
        err = self._t._transport_error()
        if err is not None:
            self._fail(255, err[1])
            return
        await self._t._latency()
        path = _meta_str(self._spec.meta, "path", "")
        nbytes = _meta_int(self._spec.meta, "bytes", 65536)
        sim = self._t._sim()
        assert sim is not None
        world.sync()
        offset: int | None = None
        lg = sim.logs.get(path)
        if lg is None:
            self._stderr += f"tail: cannot open '{path}' for reading: No such file or directory\n"
        else:
            offset = lg.size
            for block in _blocks(lg.tail(nbytes)):
                yield block
        while not self._closed.is_set():
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=world.stream_poll_s)
                break
            except TimeoutError:
                pass
            world.sync()
            lg = sim.logs.get(path)
            if lg is None:
                continue
            if offset is None:
                self._stderr += f"tail: '{path}' has appeared;  following new file\n"
                offset = 0
            data = lg.read_from(offset)
            offset = lg.size
            for block in _blocks(data):
                yield block
        if self._rc is None:
            self._rc = 0


def _blocks(data: bytes) -> list[bytes]:
    return [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
