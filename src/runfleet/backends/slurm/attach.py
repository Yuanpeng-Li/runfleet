"""[WP2] Attach for Slurm (§7.16): technical modes, the login probe (node hops + audit), the nested node script, and
ingest of one sample into a `JobLive`.

One sample is ONE login probe: `slurm.attach.node<i>` sections (ssh: up to 4 nodes, batch host first; srun: exactly
one node) and the late `slurm.attach.audit`. Each node section carries a nested ProbeScript, rendered without a temp
dir (`framing.render(tmp=False)`), base64-encoded into argv and run under `timeout` on the node itself; its framed
output (own nonce) is the outer section's stdout. The login probe itself (purpose "attach") is rendered without a temp
dir too (framing's default for that purpose), so nothing is written on any server. Every no-leftover layer of §7.16.4
lives in the command lines below; the audit is clean only with an affirmative node.self from every node that may have
run the script.

The constants are part of the contract (DESIGN.md §7.16, Appendix A) and shared with the demo transport and the tests.
Everything here is pure: the collector supplies its state (job ids → keys, scontrol GPU indices) via `IngestContext`.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from runfleet.backends.base import AttachPlan
from runfleet.backends.common import section_error, usable
from runfleet.backends.common.nvidia import GPU_FIELDS, attach_processes, parse_apps, parse_gpu_query, wrap
from runfleet.backends.slurm import parse_attach as pa
from runfleet.errors import AttachBlocked
from runfleet.models import (GPU, AttachMode, Capabilities, CgroupStats, ErrorKind, HostError, Job, JobKey, JobKind,
                             JobLive, JobState, NodeInspect, Proc)
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, SectionResult
from runfleet.util.humanize import gib          # hint amounts match the UI's GPU rows (MiB / 1024)

log = logging.getLogger("runfleet.backends.slurm.attach")

# ---- section names (login side; owner "slurm") ----
NODE_SECTION_PREFIX = "slurm.attach.node"     # + str(i): carries AttachPlan.nodes[i]
AUDIT_SECTION = "slurm.attach.audit"          # late; always the last section of an attach probe

# ---- nested node script sections (owner "node"; rendered with framing.render(tmp=False), order fixed) ----
NODE_SECTIONS: tuple[str, ...] = ("node.id", "node.cg", "node.ps", "node.pids", "node.map",   # early
                                  "node.gpu", "node.apps", "node.self")                        # late, in this order

# ---- limits and thresholds ----
MAX_SSH_NODES = 4               # ssh mode samples at most this many nodes per sample (batch host first)
NODE_SECTION_TIMEOUT = 5.0      # each nested section (framing renders `timeout -k 2 5`)
NODE_SCRIPT_TIMEOUT = 10        # node side: `timeout -k 1 10 sh -c '…'` around the whole nested script
SSH_HOP_TIMEOUT = 12            # login side: `timeout -k 2 12 ssh …`; outer Section.timeout = 14
SRUN_INNER_TIMEOUT = 15         # srun … `timeout -k 1 15 sh -c '…'`
SRUN_HOP_TIMEOUT = 25           # login side: `timeout -k 2 25 srun …`; outer Section.timeout = 27
AUDIT_TIMEOUT = 10.0

IDLE_UTIL_PCT = 5.0             # "GPU idle" hint: util < 5 % …
IDLE_MEM_MIB = 1024.0           # … and memory used > 1 GiB …
IDLE_SAMPLES = 2                # … on >= 2 consecutive samples
MEM_HINT_FRACTION = 0.9         # cgroup memory.current (incl. shmem) > 90 % of memory.max

MAX_NODE_JOBS = 8               # own running jobs per node read by node.cg / node.pids (target first)

# nvidia-smi fields of node.gpu: the workstation list + the bus id (pairs GPUs with /proc/driver/nvidia, §7.16.5)
ATTACH_GPU_FIELDS: tuple[str, ...] = (*GPU_FIELDS, "pci.bus_id")

# login → node hop (§7.16.4 layer 2): no mux socket or ControlPersist process on the login host, never a key written
# to its known_hosts, no agent/X11/port forwarding, bounded connect and liveness
SSH_OPTIONS: tuple[str, ...] = ("BatchMode=yes", "ControlMaster=no", "ControlPath=none", "StrictHostKeyChecking=yes",
                                "UpdateHostKeys=no", "ClearAllForwardings=yes", "ConnectTimeout=5",
                                "ServerAliveInterval=5", "ServerAliveCountMax=1", "LogLevel=ERROR")

_NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_SLUID_RE = re.compile(r"^[A-Za-z0-9]+$")
_B64_RE = re.compile(r"^[A-Za-z0-9+/=]+$")

# ---- nested node section commands (§7.16.3) ----
NODE_ID_CMD = 'hostname -s; cat /proc/self/cgroup; echo "J=${SLURM_JOB_ID:-} G=${SLURM_STEP_GPUS:-}"'
# `echo $$` first: the parser drops the probe's own process tree (the chain of own ancestors of this shell)
NODE_PS_CMD = 'echo "$$"; ps -ww -u "$(id -un)" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args='
NODE_MAP_CMD = "grep -H -E 'GPU UUID|Device Minor|Bus Location' /proc/driver/nvidia/gpus/*/information"
# node.self (§7.16.4 layer 5): every own nvidia-smi as `pid ppid stat etimes args`; when one is in D state a second
# read 1 s later tells a hung query (still D: survives every signal) from one in flight; `@@done` = complete answer
NODE_SELF_CMD = (
    'rf_s() { p=$(ps -ww -u "$(id -un)" -o pid=,ppid=,stat=,etimes=,args=) || return 1\n'
    "  printf '%s\\n' \"$p\" | awk '{c=$5; sub(/.*\\//, \"\", c)} c == \"nvidia-smi\"'; }\n"
    "a=$(rf_s) || exit 1\n"
    "[ -z \"$a\" ] || printf '%s\\n' \"$a\"\n"
    "if printf '%s\\n' \"$a\" | awk '$3 ~ /^D/ {d=1} END {exit !d}'; then\n"
    "  sleep 1; b=$(rf_s) || exit 1; echo @@again; [ -z \"$b\" ] || printf '%s\\n' \"$b\"\n"
    "fi\n"
    "echo @@done")
NODE_GPU_CMD = wrap(f"nvidia-smi --query-gpu={','.join(ATTACH_GPU_FIELDS)} --format=csv,noheader,nounits")
NODE_APPS_CMD = wrap("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")

# The job's cgroup directory sits right below the slurmstepd scope of the probe's own cgroup: `<node>_slurmstepd.scope`
# on k2/vega, `slurmstepd.scope` on lyra; the job directory is `job_<J>`, or the SLUID on Slurm 26.05 (lyra).
_CG_HEAD = (
    "S=$(sed -n -E 's#^0::(/system\\.slice/[^/]*slurmstepd\\.scope)(/.*)?$#\\1#p' /proc/self/cgroup | head -n 1)\n"
    '[ -n "$S" ] || S=/system.slice/$(hostname -s)_slurmstepd.scope\n'
    '[ -d "/sys/fs/cgroup$S" ] || S=/system.slice/slurmstepd.scope\n')
_CG_BODY = (
    '  j=${p%%:*} u=${p#*:} d=\n'
    '  for c in "job_$j" "$u"; do\n'
    '    [ -n "$c" ] && [ -d "/sys/fs/cgroup$S/$c" ] && { d="/sys/fs/cgroup$S/$c"; break; }\n'
    '  done\n'
    '  echo "@@cg $j ${d:--}"\n'
    '  [ -n "$d" ] || continue\n'
    '  for f in memory.current memory.peak memory.max; do [ -r "$d/$f" ] && echo "$f $(cat "$d/$f")"; done\n'
    '  cat "$d/memory.events" 2>/dev/null\n'
    "  grep -E '^(anon|file|shmem) ' \"$d/memory.stat\" 2>/dev/null\n"
    "  grep '^usage_usec ' \"$d/cpu.stat\" 2>/dev/null\n"
    "done; true")


def cg_cmd(jobs_raw: Sequence[str], sluids: Sequence[str]) -> str:
    """node.cg: one `@@cg <job> <dir|->` block per job (values validated by the caller)."""
    pairs = " ".join(shlex.quote(f"{j}:{u}") for j, u in zip(jobs_raw, sluids, strict=True))
    return f"{_CG_HEAD}for p in {pairs}; do\n{_CG_BODY}"


def pids_cmd(jobs_raw: Sequence[str]) -> str:
    """node.pids: `scontrol listpids` of every listed job (pid → step, §7.16.3)."""
    return f'for j in {" ".join(shlex.quote(j) for j in jobs_raw)}; do scontrol listpids "$j"; done'


def node_script(jobs_raw: Sequence[str], sluids: Sequence[str]) -> ProbeScript:
    """The nested node script of one node hop (purpose "attach", fresh nonce)."""
    t = NODE_SECTION_TIMEOUT
    return framing.make_script([
        Section("node.id", NODE_ID_CMD, timeout=t),
        Section("node.cg", cg_cmd(jobs_raw, sluids), timeout=t),
        Section("node.ps", NODE_PS_CMD, timeout=t),
        Section("node.pids", pids_cmd(jobs_raw), timeout=t, requires=("scontrol",)),
        Section("node.map", NODE_MAP_CMD, timeout=t),
        Section("node.gpu", NODE_GPU_CMD, timeout=t, requires=("nvidia-smi",), late=True),
        Section("node.apps", NODE_APPS_CMD, timeout=t, requires=("nvidia-smi",), late=True),
        Section("node.self", NODE_SELF_CMD, timeout=t, requires=("ps", "awk"), late=True),
    ], "attach")


def encode(script: ProbeScript) -> str:
    """base64 (one line) of the rendered nested script; it travels in argv, never as a file or on stdin."""
    return base64.b64encode(framing.render(script, tmp=False).encode()).decode("ascii")


def _decoder(b64: str) -> str:
    assert _B64_RE.match(b64)                      # safe inside single and double quotes
    return f"echo {b64} | base64 -d | sh"


def ssh_cmd(node: str, b64: str) -> str:
    """Login side of one ssh hop (§7.16.3/§7.16.4 layers 2-3)."""
    opts = " ".join(f"-o {o}" for o in SSH_OPTIONS)
    remote = f"timeout -k 1 {NODE_SCRIPT_TIMEOUT} sh -c '{_decoder(b64)}'"
    return f'timeout -k 2 {SSH_HOP_TIMEOUT} ssh -T -a -x {opts} -- {shlex.quote(node)} "{remote}"'


def srun_cmd(job_id_raw: str, node: str, b64: str) -> str:
    """Login side of one srun step (§7.16.4 layer 4): never --gres=none (no GPUs), never --pty, never --exclusive."""
    return (f"timeout -k 2 {SRUN_HOP_TIMEOUT} srun --jobid={shlex.quote(job_id_raw)} --overlap --job-name=rfprobe "
            f"-N1 -n1 -w {shlex.quote(node)} --time=1 --quiet --input=none "
            f"timeout -k 1 {SRUN_INNER_TIMEOUT} sh -c '{_decoder(b64)}'")


def audit_cmd(job_id_raw: str) -> str:
    """§7.16.4 layer 5 (`%T` is invalid with `-s` on all three clusters; `[h]` keeps pgrep off this very shell)."""
    return (f"echo @@steps; squeue -h -s -j {shlex.quote(job_id_raw)} -o '%i %j' 2>&1; echo @@ssh; "
            "pgrep -u \"$(id -un)\" -f 'ControlPat[h]=none' | wc -l; true")


def preview_of(cmd: str, b64: str) -> str:
    """The confirmation preview: the exact command with the base64 blob abbreviated (it is ~6 KB)."""
    return cmd.replace(b64, f"<read-only node script, {len(b64)} bytes base64>")


# ---------------------------------------------------------------- modes and plans
def job_nodes(job: Job) -> tuple[str, ...]:
    """The job's nodes, batch host first."""
    nodes = list(job.nodes) or ([job.batch_host] if job.batch_host else [])
    if job.batch_host and job.batch_host in nodes:
        nodes.remove(job.batch_host)
        nodes.insert(0, job.batch_host)
    return tuple(nodes)


def srun_known(caps: Capabilities) -> bool:
    """srun on the login host: listed by detect, or unknown (a tool list detected before v0.2 has neither `srun` nor
    `ssh`; a missing srun then shows up as the section's MISSING outcome)."""
    return "srun" in caps.tools or "ssh" not in caps.tools


def modes(job: Job, caps: Capabilities, *, overlap: bool) -> tuple[AttachMode, ...]:
    """Technically possible modes for `job`, "ssh" first (SchedulerCollector.attach_modes)."""
    if job.kind is not JobKind.SLURM or job.state is not JobState.RUNNING or not job.job_id_raw.isdigit():
        return ()
    if not job_nodes(job):
        return ()
    out: list[AttachMode] = []
    if caps.node_ssh != "denied":
        out.append("ssh")
    if overlap and srun_known(caps):
        out.append("srun")
    return tuple(out)


def target_nodes(job: Job, mode: AttachMode, node: str | None) -> tuple[str, ...]:
    """Nodes one sample reads: `node` alone, else ssh = up to MAX_SSH_NODES (batch host first), srun = the batch host
    (one step = one permanent sacct row). Raises AttachBlocked when the job cannot be read."""
    if job.kind is not JobKind.SLURM:
        raise AttachBlocked("attach reads Slurm jobs only")
    if job.state is not JobState.RUNNING:
        raise AttachBlocked("job is not running")
    if not job.job_id_raw.isdigit():
        raise AttachBlocked(f"job id {job.job_id_raw!r} cannot be attached")
    nodes = job_nodes(job)
    if not nodes:
        raise AttachBlocked(f"job {job.key.id} has no node yet")
    if node is not None:
        if node not in nodes:
            raise AttachBlocked(f"{node} is not a node of job {job.key.id}")
        nodes = (node,)
    picked = nodes[:MAX_SSH_NODES] if mode == "ssh" else nodes[:1]
    bad = [n for n in picked if not _NODE_RE.match(n)]
    if bad:
        raise AttachBlocked(f"unsafe node name {bad[0]!r}")
    return picked


def build_plan(job: Job, mode: AttachMode, nodes: Sequence[str],
               jobs_by_node: Mapping[str, Sequence[tuple[str, str]]]) -> AttachPlan:
    """The AttachPlan for these nodes. `jobs_by_node[node]` = [(raw job id, SLUID or "")], target first: the jobs
    node.cg / node.pids read on that node."""
    sections: list[Section] = []
    inner: dict[str, ProbeScript] = {}
    preview = ""
    for i, node in enumerate(nodes):
        pairs = [(j, u if _SLUID_RE.match(u or "") else "") for j, u in jobs_by_node.get(node, ())
                 if j.isdigit() and j != job.job_id_raw][: MAX_NODE_JOBS - 1]
        target_sluid = next((u for j, u in jobs_by_node.get(node, ()) if j == job.job_id_raw), "")
        pairs.insert(0, (job.job_id_raw, target_sluid if _SLUID_RE.match(target_sluid or "") else ""))
        jobs_raw = [j for j, _ in pairs]
        sluids = [u for _, u in pairs]
        script = node_script(jobs_raw, sluids)
        b64 = encode(script)
        name = f"{NODE_SECTION_PREFIX}{i}"
        if mode == "ssh":
            cmd, timeout, requires = ssh_cmd(node, b64), SSH_HOP_TIMEOUT + 2.0, ("timeout", "ssh")
        else:
            cmd, timeout, requires = srun_cmd(job.job_id_raw, node, b64), SRUN_HOP_TIMEOUT + 2.0, ("timeout", "srun")
        sections.append(Section(name, cmd, timeout=timeout, requires=requires, meta={
            "mode": mode, "node": node, "job_id_raw": job.job_id_raw, "jobs_raw": jobs_raw, "sluids": sluids,
            "inner_nonce": script.nonce, "inner": [s.name for s in framing.ordered(script)]}))
        inner[name] = script
        if i == 0:
            preview = preview_of(cmd, b64)
    sections.append(Section(AUDIT_SECTION, audit_cmd(job.job_id_raw), timeout=AUDIT_TIMEOUT, late=True,
                            meta={"mode": mode, "job_id_raw": job.job_id_raw, "nodes": list(nodes)}))
    return AttachPlan(key=job.key, mode=mode, job_id_raw=job.job_id_raw, nodes=tuple(nodes), sections=tuple(sections),
                      inner=inner, preview=preview, sacct_rows=len(nodes) if mode == "srun" else 0)


# ---------------------------------------------------------------- ingest
@dataclass(frozen=True, slots=True)
class IngestContext:
    host: str                                            # RunFleet host name (GPU.host, JobKey.host)
    user: str                                            # caps.remote_user (ps rows)
    now: datetime
    job_key: Callable[[str], JobKey | None]              # raw job id -> its JobKey (None: unknown here)
    known_idx: Callable[[str, str], tuple[int, ...]]     # (raw job id, node) -> scontrol -d IDX; () unknown
    job_gpus: int | None = None                          # GPUs allocated to the target (None: unknown)


@dataclass(frozen=True, slots=True)
class AttachIngest:
    live: JobLive
    caps_patch: dict[str, object]
    gpu_model: tuple[str, float | None] | None = None    # nvidia-smi (gres-style type, memory.total MiB) of the target


_DENIED = re.compile(r"Permission denied \(")
_ADOPT_DENY = re.compile(r"pam_slurm_adopt|no active jobs on this node", re.IGNORECASE)
_HOSTKEY = re.compile(r"Host key verification failed|host key is known|REMOTE HOST IDENTIFICATION HAS CHANGED|"
                      r"host key for \S+ has changed", re.IGNORECASE)
_DNS = re.compile(r"Could not resolve hostname|Name or service not known", re.IGNORECASE)
_CONN_TIMEOUT = re.compile(r"Connection timed out|Operation timed out|timed out during banner", re.IGNORECASE)
_REFUSED = re.compile(r"Connection refused|No route to host", re.IGNORECASE)
# failed before a session existed (a "Connection timed out" read error comes after the command started)
_PRECONNECT = re.compile(r"connect to host \S+ port \d+: |timed out during banner", re.IGNORECASE)
_DRIVER = re.compile(r"couldn't communicate with the NVIDIA driver", re.IGNORECASE)
# srun did not create a step (no sacct row): its own error lines, before any output of the step
_NO_STEP = re.compile(r"Unable to create step|Invalid job id|already (?:completed|finished)|Invalid node name|"
                      r"Requested node configuration is not available|Unable to confirm allocation|"
                      r"Access/permission denied|unrecognized option|invalid option|Invalid user", re.IGNORECASE)


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _err(kind: ErrorKind, summary: str, detail: str, section: str, at: datetime) -> HostError:
    return HostError(kind=kind, summary=summary[:80], detail=detail[:4096], section=section, at=at)


def hop_error(mode: AttachMode, node: str, r: SectionResult, at: datetime) -> HostError:
    """Why the node never ran the script (no inner marker): classified for the GPU tab (§7.16.6)."""
    text = f"{r.stderr}\n{r.stdout}"
    if r.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.MISSING):
        err = section_error(r, at)
        assert err is not None
        return err
    if mode == "ssh":
        if _DENIED.search(text):
            return _err(ErrorKind.PERMISSION, f"ssh to {node} refused (Permission denied)", r.stderr, r.name, at)
        if _ADOPT_DENY.search(text):
            return _err(ErrorKind.PERMISSION, f"no job of yours on {node} — job moved or ended?", r.stderr,
                        r.name, at)
        if _HOSTKEY.search(text):
            advice = (f"{node}'s host key is not in the login host's known_hosts; the attach hop never adds keys "
                      f"(StrictHostKeyChecking=yes). Run `ssh {node} true` there once and check the fingerprint.")
            return _err(ErrorKind.HOSTKEY, f"{node}'s host key is not known on the login host",
                        f"{advice}\n{r.stderr}", r.name, at)
        if _DNS.search(text):
            return _err(ErrorKind.DNS, f"login host cannot resolve {node}", r.stderr, r.name, at)
        if _CONN_TIMEOUT.search(text):
            return _err(ErrorKind.TIMEOUT, f"ssh to {node} timed out", r.stderr, r.name, at)
        if _REFUSED.search(text):
            return _err(ErrorKind.REFUSED, f"ssh to {node}: {_first_line(r.stderr)}", r.stderr, r.name, at)
    if r.outcome is SectionOutcome.TIMEOUT:
        what = "ssh to" if mode == "ssh" else "srun on"
        return _err(ErrorKind.TIMEOUT, f"{what} {node} timed out", r.stderr, r.name, at)
    line = _first_line(r.stderr) or _first_line(r.stdout)
    if mode == "srun":
        return _err(ErrorKind.REMOTE, f"srun on {node}: {line or f'rc {r.rc}'}", r.stderr, r.name, at)
    if r.rc == 255 and not line:
        # LogLevel=ERROR hides the banner pam_slurm_adopt sends and "Connection closed by …": a bare rc 255 is
        # most often a node without a job of the user
        return _err(ErrorKind.SSH, f"ssh to {node} closed without a message — no job of yours there?",
                    "ssh exited 255 with no message (LogLevel=ERROR hides pam_slurm_adopt's banner)", r.name, at)
    return _err(ErrorKind.SSH, f"ssh to {node}: {line or f'rc {r.rc}'}", r.stderr, r.name, at)


def srun_rows(plan: AttachPlan, results: Mapping[str, SectionResult]) -> int:
    """Permanent sacct rows this srun sample created: one per node section that launched srun, unless the section
    never ran or srun's own error says no step was created (§7.16.5)."""
    if plan.mode != "srun":
        return 0
    rows = 0
    for i in range(len(plan.nodes)):
        r = results.get(f"{NODE_SECTION_PREFIX}{i}")
        if r is None or r.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.MISSING):
            continue
        if f"@@RF{plan.inner[r.name].nonce} " not in r.stdout and _NO_STEP.search(r.stderr):
            continue
        rows += 1
    return rows


def _bytes_gib(n: int) -> str:
    return gib(n / 2**20)


def gpu_label(g: GPU) -> str:
    """`GPU1` (Slurm IDX), else the bus id (`GPU C5:00.0`); never a guessed index (§7.16.5)."""
    if g.slurm_idx is not None:
        return f"GPU{g.slurm_idx}"
    bus = pa.canon_bus(g.bus_id)
    if bus:
        return f"GPU {bus.removeprefix('0000:').upper()}"
    return f"GPU {g.uuid[:12]}" if g.uuid else f"GPU#{g.index}"


def _idx_map(gpus: Sequence[GPU], gmap: Mapping[str, pa.GpuMapEntry], mode: AttachMode,
             step_gpus: tuple[int, ...] | None, known: tuple[int, ...]) -> tuple[GPU, ...]:
    """bus_id + slurm_idx (§7.16.5 order): srun = SLURM_STEP_GPUS paired by uuid (through the minors when the step
    has several GPUs); ssh = Device Minor, unless scontrol's IDX for the landed job disagree; else None."""
    entries = [pa.map_by_uuid_or_bus(gmap, g.uuid, g.bus_id) for g in gpus]
    minors = [e.minor if e is not None else None for e in entries]
    idx: list[int | None] = [None] * len(gpus)
    if mode == "srun":
        if step_gpus is not None and len(gpus) == len(step_gpus):
            if len(gpus) == 1:
                idx = [step_gpus[0]]
            elif None not in minors and sorted(m for m in minors if m is not None) == sorted(step_gpus):
                idx = list(minors)
    elif gpus and None not in minors and (not known or set(minors) == set(known)):
        idx = list(minors)
    out = []
    for g, e, i in zip(gpus, entries, idx, strict=True):
        bus = g.bus_id or (e.bus if e is not None else None)
        out.append(dataclasses.replace(g, bus_id=bus, slurm_idx=i))
    return tuple(out)


def _cpu_cores(node: str, job: str, stats: CgroupStats, previous: JobLive | None, now: datetime) -> float | None:
    """Δcpu.stat usage_usec / Δwall against the previous sample of the same (node, job); None on the first sample
    or when Δt < 1 s."""
    if previous is None or stats.cpu_usage_usec is None:
        return None
    dt = (now - previous.sampled_at).total_seconds()
    if dt < 1.0:
        return None
    prev_node = next((n for n in previous.nodes if n.node == node), None)
    prev = prev_node.cgroups.get(job) if prev_node is not None else None
    if prev is None or prev.cpu_usage_usec is None or stats.cpu_usage_usec < prev.cpu_usage_usec:
        return None
    return (stats.cpu_usage_usec - prev.cpu_usage_usec) / 1e6 / dt


def _inner_problem(node: str, outer: str, r: SectionResult, at: datetime) -> HostError | None:
    err = section_error(r, at)
    if err is None:
        return None
    return dataclasses.replace(err, summary=f"{node}: {err.summary}"[:80], section=outer)


def _nvidia_problem(node: str, outer: str, r: SectionResult, job_gpus: int | None, at: datetime) -> HostError | None:
    """node.gpu / node.apps failures: hung (rc 124/125, timeout), no driver, no devices, missing nvidia-smi."""
    text = r.stdout + r.stderr
    if pa.no_devices(text):
        if job_gpus == 0:
            return None
        return _err(ErrorKind.REMOTE, f"no GPU visible on {node} (nvidia-smi: No devices found)", text, outer, at)
    if r.ok:
        return None
    if r.rc in (124, 125) or r.outcome in (SectionOutcome.TIMEOUT, SectionOutcome.INCOMPLETE):
        return _err(ErrorKind.TIMEOUT, f"nvidia-smi hung on {node}", r.stderr, outer, at)
    if r.rc == 9 or _DRIVER.search(text):
        return _err(ErrorKind.REMOTE, f"NVIDIA driver not loaded on {node}", r.stderr, outer, at)
    if r.outcome is SectionOutcome.MISSING:
        return None if job_gpus == 0 else _err(ErrorKind.MISSING, f"nvidia-smi not found on {node}", "", outer, at)
    if r.outcome is SectionOutcome.NOT_RUN:
        return None                                     # reported once as "node script cut off"
    return _inner_problem(node, outer, r, at)


def _hop_unchecked(mode: AttachMode, r: SectionResult) -> str | None:
    """A hop without any inner marker: None only when it provably never started anything on the node (refused before
    login, no step created, never ran), else why the node may still have run (part of) the script — a timeout, a
    connection reset after the remote command started, any other unexplained failure."""
    if r.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.MISSING):
        return None
    if r.outcome in (SectionOutcome.TIMEOUT, SectionOutcome.INCOMPLETE) or r.rc in (124, 137):
        return f"{mode} timed out"
    text = f"{r.stderr}\n{r.stdout}"
    if mode == "srun":
        return None if _NO_STEP.search(r.stderr) else f"srun failed (rc {r.rc})"
    if any(p.search(text) for p in (_DENIED, _ADOPT_DENY, _HOSTKEY, _DNS, _PRECONNECT)):
        return None
    # a bare rc 255 is most often pam_slurm_adopt's refusal (hop_error), but ssh also exits 255 silently when it is
    # killed after the remote command started: not a proof that nothing ran
    return f"ssh failed (rc {r.rc})"


def _self_unchecked(r: SectionResult, chk: pa.SelfCheck | None) -> str | None:
    """Why node.self gave no affirmative cleanup result (None: it did)."""
    if chk is not None:
        return None
    if r.outcome in (SectionOutcome.NOT_RUN, SectionOutcome.INCOMPLETE, SectionOutcome.TIMEOUT):
        return "node.self did not finish"
    if r.outcome is SectionOutcome.MISSING:
        return f"node.self could not run ({r.stderr.removeprefix('missing:')} not found)"
    if not r.ok:
        return f"node.self failed (rc {r.rc})"
    return "node.self output unreadable"


def ingest_node(plan: AttachPlan, i: int, results: Mapping[str, SectionResult], ax: IngestContext,
                previous: JobLive | None) -> tuple[NodeInspect | None, list[HostError], bool, str | None]:
    """One node section → (NodeInspect or None when the node never ran the script, errors, ssh landed, why the node
    gave no affirmative cleanup result — None when it did or nothing can have run there)."""
    name = f"{NODE_SECTION_PREFIX}{i}"
    node = plan.nodes[i]
    now = ax.now
    r = results.get(name) or SectionResult(name, SectionOutcome.NOT_RUN, None, "", "", None)
    inner = plan.inner.get(name)
    sec = next((s for s in plan.sections if s.name == name), None)
    meta: Mapping[str, object] = sec.meta if sec is not None else {}
    jobs_raw = [str(j) for j in meta.get("jobs_raw", [plan.job_id_raw])]   # type: ignore[attr-defined]
    sluids = [str(u) for u in meta.get("sluids", [""] * len(jobs_raw))]    # type: ignore[attr-defined]
    parsed = framing.parse(r.stdout, inner) if inner is not None and r.stdout else None
    if parsed is None or parsed.markers == 0:
        return None, [hop_error(plan.mode, node, r, now)], False, _hop_unchecked(plan.mode, r)
    ir = parsed.sections
    errors: list[HostError] = []

    def got(sname: str) -> SectionResult:
        return ir.get(sname) or SectionResult(sname, SectionOutcome.NOT_RUN, None, "", "", None)

    rid = got("node.id")
    nid = pa.parse_node_id(rid.stdout) if usable(rid) else pa.NodeId(None, None, None, None, None)
    landed = plan.job_id_raw if plan.mode == "srun" else pa.landed_job(nid, jobs_raw, sluids)
    of_target = None if landed is None else landed == plan.job_id_raw

    # processes (the probe's own tree removed) and their steps
    rps = got("node.ps")
    all_procs: list[Proc] = []
    procs: list[Proc] = []
    if usable(rps):
        self_pid, pr = pa.parse_node_ps(rps.stdout, user=ax.user)
        all_procs = pr.items
        mine = pa.probe_pids(all_procs, self_pid)
        procs = [p for p in all_procs if p.pid not in mine]
    rpids = got("node.pids")
    step_of_pid = pa.parse_listpids(rpids.stdout) if usable(rpids) else {}

    # cgroups (target first), CPU cores from the previous sample
    rcg = got("node.cg")
    cgroups: dict[str, CgroupStats] = {}
    if usable(rcg):
        for job, stats in pa.parse_cgroups(rcg.stdout).items():
            if stats is not None:
                cgroups[job] = dataclasses.replace(stats, cpu_cores=_cpu_cores(node, job, stats, previous, now))
        if rcg.ok and plan.job_id_raw not in cgroups:
            errors.append(_err(ErrorKind.REMOTE, f"{node}: cgroup of job {plan.job_id_raw} not found", rcg.stdout,
                               name, now))

    # GPUs: nvidia-smi rows + bus/IDX from /proc/driver/nvidia + compute apps joined with ps and listpids
    rgpu, rapps, rmap = got("node.gpu"), got("node.apps"), got("node.map")
    gpus: tuple[GPU, ...] = ()
    if usable(rgpu) and not pa.no_devices(rgpu.stdout + rgpu.stderr):
        gq = parse_gpu_query(rgpu.stdout, ATTACH_GPU_FIELDS, host=ax.host, sampled_at=now)
        if gq.mostly_bad:
            line = gq.bad_lines[0] if gq.bad_lines else (0, "", "")
            errors.append(_err(ErrorKind.PARSE, f"{node}: could not parse nvidia-smi output",
                               f"line {line[0]}: {line[2]}: {line[1][:200]}", name, now))
        else:
            gpus = tuple(gq.items)
    gmap = pa.parse_gpu_map(rmap.stdout) if usable(rmap) else {}
    known = ax.known_idx(landed, node) if landed is not None else ()
    gpus = _idx_map(gpus, gmap, plan.mode, nid.step_gpus, known)
    if gpus and usable(rapps):
        apps, _ = parse_apps(rapps.stdout)
        gpus = attach_processes(gpus, apps, all_procs)
        gpus = tuple(dataclasses.replace(g, processes=tuple(
            dataclasses.replace(p, job=_job_of(p.pid, step_of_pid, ax)) for p in g.processes)) for g in gpus)
    for problem in (_nvidia_problem(node, name, rgpu, ax.job_gpus if of_target is not False else None, now),
                    _nvidia_problem(node, name, rapps, ax.job_gpus if of_target is not False else None, now)):
        if problem is not None and not any(e.summary == problem.summary for e in errors):
            errors.append(problem)
    if gpus and not rmap.ok:
        if (e := _inner_problem(node, name, rmap, now)) is not None and rmap.outcome is not SectionOutcome.NOT_RUN:
            errors.append(e)
    for rr in (rid, rps, rpids):
        if not rr.ok and rr.outcome not in (SectionOutcome.NOT_RUN, SectionOutcome.INCOMPLETE):
            if (e := _inner_problem(node, name, rr, now)) is not None:
                errors.append(e)
    cut = [s for s in NODE_SECTIONS if got(s).outcome in (SectionOutcome.NOT_RUN, SectionOutcome.INCOMPLETE)]
    if cut:
        errors.append(_err(ErrorKind.TIMEOUT, f"node script on {node} was cut off before {cut[0]}",
                           r.stderr, name, now))
    # the cleanup check (§7.16.4 layer 5): stuck = D state on both reads; other nvidia-smi are informational
    rself = got("node.self")
    chk = pa.parse_self(rself.stdout) if rself.ok else None
    ni = NodeInspect(node=node, target_job=plan.job_id_raw, landed_job=landed, gpus_of_target=of_target, gpus=gpus,
                     procs=tuple(procs), step_of_pid=step_of_pid, cgroups=cgroups,
                     stray_nvidia_smi=len(chk.stuck) if chk is not None else None, cost_ms=r.ms, errors=tuple(errors),
                     other_nvidia_smi=chk.other if chk is not None else None)
    return ni, [], plan.mode == "ssh" and rid.ok, _self_unchecked(rself, chk)


def _job_of(pid: int, step_of_pid: Mapping[int, str], ax: IngestContext) -> JobKey | None:
    step = step_of_pid.get(pid)
    return ax.job_key(step.partition(".")[0]) if step else None


def _hints(plan: AttachPlan, nodes: Sequence[NodeInspect], previous: JobLive | None,
           ) -> tuple[tuple[str, ...], dict[str, int]]:
    """§7.16.5 hints for the target job (GPU idle on ≥ 2 consecutive samples, oom_kill, memory near the limit) and the
    idle streaks carried to the next sample."""
    prev = previous.idle_streak if previous is not None else {}
    multi = len(nodes) > 1
    hints: list[str] = []
    streak: dict[str, int] = {}
    for n in nodes:
        where = f" on {n.node}" if multi else ""
        if n.gpus_of_target:
            for g in n.gpus:
                k = g.uuid or g.bus_id or f"{n.node}:{g.index}"
                if (g.util_pct is not None and g.util_pct < IDLE_UTIL_PCT and g.mem_used_mib is not None
                        and g.mem_used_mib > IDLE_MEM_MIB):
                    streak[k] = prev.get(k, 0) + 1
                    if streak[k] >= IDLE_SAMPLES:
                        hints.append(f"{gpu_label(g)} idle ({g.util_pct:.0f}%) but holding "
                                     f"{gib(g.mem_used_mib)}{where}")
        c = n.cgroups.get(plan.job_id_raw)
        if c is None:
            continue
        if c.oom_kill:
            hints.append(f"oom_kill {c.oom_kill} — a process of this job was killed for memory{where}")
        if c.mem_current is not None and c.mem_max and c.mem_current > MEM_HINT_FRACTION * c.mem_max:
            shmem = f" (incl. shmem {_bytes_gib(c.shmem)})" if c.shmem else ""
            hints.append(f"memory {c.mem_current * 100 / c.mem_max:.0f}% of the job's limit{shmem}{where}")
    return tuple(hints), streak


def _audit(plan: AttachPlan, r: SectionResult | None, nodes: Sequence[NodeInspect], at: datetime,
           unchecked: Sequence[str] = ()) -> tuple[bool | None, str, HostError | None]:
    """§7.16.4 layer 5 → (clean, detail, error). Clean needs an affirmative answer from every layer: the audit
    section AND node.self of every node the script may have run on (`unchecked`: "<node> (<why not>)")."""
    leftovers = [f"{n.stray_nvidia_smi} hung nvidia-smi on {n.node}" for n in nodes if n.stray_nvidia_smi]
    ran = r is not None and r.outcome is not SectionOutcome.NOT_RUN
    info = pa.parse_audit(r.stdout) if r is not None and ran and usable(r) else None
    if info is not None and info.steps:
        leftovers += [f"rfprobe step {sid} still running" for sid, name in info.steps if name == "rfprobe"]
    if info is not None and info.ssh_clients:
        k = info.ssh_clients
        leftovers.append(f"{k} node ssh client{'s' if k != 1 else ''} still running on the login node")
    if leftovers:
        detail = "; ".join(leftovers)
        log.warning("attach %s: leftovers after the sample: %s", plan.key, detail)
        return False, detail, None                      # a leftover found anywhere wins over a missing answer
    if r is None or not ran:
        return None, "audit did not run", None
    err = section_error(r, at) if not r.ok else None
    if info is None or info.steps is None or info.ssh_clients is None:
        why = info.note if info is not None and info.note else (err.summary if err else "no audit output")
        return None, f"audit inconclusive: {why}", err
    if unchecked:
        return None, f"audit inconclusive: no leftover check from {'; '.join(unchecked)}", None
    return True, "clean", None


def ingest(plan: AttachPlan, results: Mapping[str, SectionResult], ax: IngestContext, *,
           previous: JobLive | None = None) -> AttachIngest:
    """One sample → JobLive + caps_patch (+ the GPU model nvidia-smi reported for the target). Never raises for
    remote failures."""
    nodes: list[NodeInspect] = []
    errors: list[HostError] = []
    landed = denied = False
    unchecked: list[str] = []
    for i in range(len(plan.nodes)):
        ni, errs, hop_landed, why = ingest_node(plan, i, results, ax, previous)
        if why is not None:
            unchecked.append(f"{plan.nodes[i]} ({why})")
        errors += errs
        landed = landed or hop_landed
        denied = denied or any(e.kind is ErrorKind.PERMISSION and "Permission denied" in e.summary for e in errs)
        if ni is not None:
            nodes.append(ni)
    patch: dict[str, object] = {}
    if plan.mode == "ssh" and landed:
        patch["node_ssh"] = "adopt"
    elif plan.mode == "ssh" and denied:
        patch["node_ssh"] = "denied"
    clean, detail, audit_err = _audit(plan, results.get(AUDIT_SECTION), nodes, ax.now, unchecked)
    if audit_err is not None:
        errors.append(audit_err)
    hints, streak = _hints(plan, nodes, previous)
    names = {g.name for n in nodes if n.gpus_of_target for g in n.gpus if g.name}
    model: tuple[str, float | None] | None = None
    if len(names) == 1:
        totals = [g.mem_total_mib for n in nodes if n.gpus_of_target for g in n.gpus if g.mem_total_mib]
        model = (pa.gpu_type_from_name(names.pop()), max(totals) if totals else None)
    cost = sum(r.ms or 0 for name, r in results.items() if name.startswith("slurm.attach."))
    live = JobLive(key=plan.key, mode=plan.mode, sampled_at=ax.now, nodes=tuple(nodes), cost_ms=cost,
                   sacct_rows_added=srun_rows(plan, results), clean=clean, audit_detail=detail, hints=hints,
                   idle_streak=streak, errors=tuple(errors))
    return AttachIngest(live=live, caps_patch=patch, gpu_model=model)
