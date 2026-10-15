"""[WP2] Attach parsers (§7.16): node.id, node.cg, node.ps, scontrol listpids, /proc/driver/nvidia information, audit.

All pure. nvidia-smi rows reuse `backends.common.nvidia.parse_gpu_query(fields=…)` / `parse_apps`, ps rows reuse
`backends.process.parse_ps.parse_ps_rows`; one cgroup block is `backends.common.cgroup.parse_cgroup_block`. The
formats are the nested node sections of `backends/slurm/attach.py` (field evidence: tests/fixtures/attach/).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from runfleet.backends.common import ParseResult
from runfleet.backends.common.cgroup import parse_cgroup_block
from runfleet.backends.process.parse_ps import OWN_COLUMNS, parse_ps_rows
from runfleet.models import CgroupStats, Proc
from runfleet.util.humanize import gpu_type_from_name as gpu_type_from_name   # lives in util: the UI reads it too
from runfleet.util.slurmfmt import split_rows

# `0::/system.slice/n-40-01_slurmstepd.scope/job_2332863/step_extern/user/task_0` (k2/vega) or
# `0::/system.slice/slurmstepd.scope/sQ4HR26FE80M00/step_0/user/task_0` (lyra, Slurm 26.05: SLUID directory)
_CG_LINE = re.compile(r"^0::(?P<path>/\S*)$")
_SCOPE_DIR = re.compile(r"/system\.slice/[^/]*slurmstepd\.scope/(?P<dir>[^/]+)")
_JOB_DIR = re.compile(r"^job_(?P<id>\d+)$")
_ENV_LINE = re.compile(r"^J=(?P<j>\S*) G=(?P<g>\S*)$")
_MAP_LINE = re.compile(r"^(?:.*/)?gpus/(?P<dir>[^/]+)/information:(?P<key>GPU UUID|Device Minor|Bus Location):"
                       r"\s*(?P<val>\S+)\s*$")
_BUS = re.compile(r"^(?:(?P<dom>[0-9A-Fa-f]{1,8}):)?(?P<bus>[0-9A-Fa-f]{2}):(?P<dev>[0-9A-Fa-f]{2})\.(?P<fn>[0-7])$")
_NO_DEVICES = re.compile(r"No devices (?:were )?found", re.IGNORECASE)


# ---------------------------------------------------------------- node.id
@dataclass(frozen=True, slots=True)
class NodeId:
    hostname: str | None
    cgroup: str | None                   # the cgroup v2 path of the probe ("0::" line)
    cgroup_dir: str | None               # the component right below *slurmstepd.scope: "job_2332863" | a SLUID
    env_job: str | None                  # $SLURM_JOB_ID ("" -> None)
    step_gpus: tuple[int, ...] | None    # $SLURM_STEP_GPUS (srun: the step's Slurm GPU IDX list); None unset/bad


def parse_node_id(text: str) -> NodeId:
    """`hostname -s`, `cat /proc/self/cgroup`, `J=<SLURM_JOB_ID> G=<SLURM_STEP_GPUS>`."""
    lines = [ln.strip() for ln in split_rows(text) if ln.strip()]
    hostname = lines[0] if lines and not _CG_LINE.match(lines[0]) and not _ENV_LINE.match(lines[0]) else None
    cgroup = cg_dir = env_job = None
    step_gpus: tuple[int, ...] | None = None
    for line in lines:
        if (m := _CG_LINE.match(line)) is not None and cgroup is None:
            cgroup = m.group("path")
            d = _SCOPE_DIR.search(cgroup)
            cg_dir = d.group("dir") if d else None
        elif (e := _ENV_LINE.match(line)) is not None:
            env_job = e.group("j") if e.group("j").isdigit() else None
            step_gpus = parse_idx_list(e.group("g"))
    return NodeId(hostname=hostname, cgroup=cgroup, cgroup_dir=cg_dir, env_job=env_job, step_gpus=step_gpus)


def parse_idx_list(text: str) -> tuple[int, ...] | None:
    """`0,1` / `2` / `0-3` (SLURM_STEP_GPUS) → indices; empty or malformed → None."""
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        lo, dash, hi = part.partition("-")
        if not lo.isdigit() or (dash and not hi.isdigit()):
            return None
        out.extend(range(int(lo), int(hi) + 1) if dash else [int(lo)])
    return tuple(out)


def landed_job(nid: NodeId, jobs_raw: Sequence[str], sluids: Sequence[str]) -> str | None:
    """The job the node session runs in: the `job_<N>` cgroup directory, or the SLUID directory mapped back through
    the plan's (jobs_raw, sluids), else $SLURM_JOB_ID (§7.16.5)."""
    if nid.cgroup_dir is not None:
        m = _JOB_DIR.match(nid.cgroup_dir)
        if m is not None:
            return m.group("id")
        for job, sluid in zip(jobs_raw, sluids, strict=False):
            if sluid and sluid == nid.cgroup_dir:
                return job
    return nid.env_job


# ---------------------------------------------------------------- node.cg
def parse_cgroups(text: str) -> dict[str, CgroupStats | None]:
    """`@@cg <job> <dir>` blocks (dir `-` = not found on this node) → raw job id -> stats (None = no directory)."""
    out: dict[str, CgroupStats | None] = {}
    job: str | None = None
    found = False
    block: list[str] = []

    def flush() -> None:
        if job is not None:
            out[job] = parse_cgroup_block("\n".join(block)) if found else None

    for line in split_rows(text):
        if line.startswith("@@cg "):
            flush()
            parts = line.split()
            job = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
            found = len(parts) > 2 and parts[2] != "-"
            block = []
        elif job is not None:
            block.append(line)
    flush()
    return out


# ---------------------------------------------------------------- node.ps
def parse_node_ps(text: str, *, user: str) -> tuple[int | None, ParseResult[Proc]]:
    """`echo $$` (the section's own shell) then headerless `ps -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,
    args=` rows → (self pid, rows)."""
    lines = split_rows(text)
    self_pid: int | None = None
    if lines and lines[0].strip().isdigit():
        self_pid = int(lines[0].strip())
        lines = lines[1:]
    return self_pid, parse_ps_rows(lines, OWN_COLUMNS, user=user)


def probe_pids(procs: Sequence[Proc], self_pid: int | None) -> frozenset[int]:
    """The probe's own processes: every own ancestor of the node.ps shell (the chain ends at a root-owned parent:
    sshd for ssh, slurmstepd for srun, neither in `ps -u`) and every descendant of the topmost one. A job's processes
    are never in that tree, because the session's parent is a root daemon."""
    by_pid = {p.pid: p for p in procs}
    if self_pid is None or self_pid not in by_pid:
        return frozenset()
    chain: list[int] = []
    cur: int | None = self_pid
    while cur is not None and cur in by_pid and cur not in chain:
        chain.append(cur)
        cur = by_pid[cur].ppid
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p.pid)
    out = {chain[-1]}
    todo = [chain[-1]]
    while todo:
        for child in children.get(todo.pop(), ()):
            if child not in out:
                out.add(child)
                todo.append(child)
    return frozenset(out)


# ---------------------------------------------------------------- node.pids
def parse_listpids(text: str) -> dict[int, str]:
    """`scontrol listpids <J>` tables (one per job, header repeated) → pid -> "<raw job id>.<step>"
    ("2332861.batch", "31367626.0", "2332863.extern"); the `-1` rows (step roots without a pid) are skipped."""
    out: dict[int, str] = {}
    for line in split_rows(text):
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue                                  # header, "-1" rows, error text
        out.setdefault(int(parts[0]), f"{parts[1]}.{parts[2]}")
    return out


# ---------------------------------------------------------------- node.map
@dataclass(frozen=True, slots=True)
class GpuMapEntry:
    bus: str                             # canonical "0000:41:00.0"
    uuid: str | None
    minor: int | None                    # /dev/nvidia<minor>; = Slurm IDX on k2 (§7.16.5)


def canon_bus(bus: str | None) -> str | None:
    """`00000000:41:00.0` (nvidia-smi), `0000:41:00.0` (/proc), `41:00.0` → `0000:41:00.0` (lower-case hex)."""
    if not bus:
        return None
    m = _BUS.match(bus.strip())
    if m is None:
        return None
    dom = int(m.group("dom") or "0", 16)
    return f"{dom:04x}:{m.group('bus').lower()}:{m.group('dev').lower()}.{m.group('fn')}"


def parse_gpu_map(text: str) -> dict[str, GpuMapEntry]:
    """`grep -H -E 'GPU UUID|Device Minor|Bus Location' /proc/driver/nvidia/gpus/*/information` → canonical bus ->
    entry (every GPU of the node, whatever the session's cgroup allows)."""
    raw: dict[str, dict[str, str]] = {}
    for line in split_rows(text):
        m = _MAP_LINE.match(line.strip())
        if m is not None:
            raw.setdefault(m.group("dir"), {})[m.group("key")] = m.group("val")
    out: dict[str, GpuMapEntry] = {}
    for d, vals in raw.items():
        bus = canon_bus(vals.get("Bus Location")) or canon_bus(d)
        if bus is None:
            continue
        minor = vals.get("Device Minor")
        out[bus] = GpuMapEntry(bus=bus, uuid=vals.get("GPU UUID"),
                               minor=int(minor) if minor and minor.isdigit() else None)
    return out


# ---------------------------------------------------------------- node.gpu / node.self
def no_devices(text: str) -> bool:
    """nvidia-smi saw no GPU: "No devices were found" (query) / "No devices found." (`-L`) — a step without GPUs
    (`srun --gres=none`, field probes) or a CPU-only job."""
    return bool(_NO_DEVICES.search(text))


def parse_count(text: str) -> int | None:
    """`… | wc -l` → int (None when the output is not a number)."""
    v = text.strip()
    return int(v) if v.isdigit() else None


@dataclass(frozen=True, slots=True)
class SelfCheck:
    """node.self: the user's nvidia-smi processes on the node after the sample (§7.16.4 layer 5)."""
    stuck: tuple[int, ...]               # pids in D state on both reads ~1 s apart: hung (they survive every signal;
                                         # a query our timeouts ended can only remain this way)
    other: int                           # the rest of the last read: a job's `nvidia-smi -l`, the user's `dmon`, a
                                         # query in flight — healthy, not ours to judge (informational)


def parse_self(text: str) -> SelfCheck | None:
    """`ps -o pid=,ppid=,stat=,etimes=,args=` rows of every own nvidia-smi, `@@again` + a second read 1 s later
    (only when a row was in D state), `@@done`. None = incomplete or unreadable (no `@@done`, a D-state row without
    its second read, a bad row): no affirmative cleanup result."""
    reads: list[list[tuple[int, str]]] = [[]]
    done = False
    for line in split_rows(text):
        s = line.strip()
        if not s:
            continue
        if done:
            return None
        if s == "@@again":
            if len(reads) > 1:
                return None
            reads.append([])
            continue
        if s == "@@done":
            done = True
            continue
        parts = s.split(None, 4)
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        reads[-1].append((int(parts[0]), parts[2]))
    if not done:
        return None
    first = {pid for pid, stat in reads[0] if stat.startswith("D")}
    if first and len(reads) < 2:
        return None
    last = reads[-1]
    stuck = tuple(sorted(first & {pid for pid, stat in last if stat.startswith("D")}))
    return SelfCheck(stuck=stuck, other=len({pid for pid, _ in last} - set(stuck)))


# ---------------------------------------------------------------- audit
@dataclass(frozen=True, slots=True)
class AuditInfo:
    steps: tuple[tuple[str, str], ...] | None   # (step id, step name) live in squeue -s; None = squeue failed
    ssh_clients: int | None                     # own `ssh … ControlPath=none` processes left on the login host
    note: str = ""                              # why `steps` is None (squeue's message)


def parse_audit(text: str) -> AuditInfo:
    """`@@steps` + `squeue -h -s -j J -o '%i %j'` (stderr folded in) + `@@ssh` + a count."""
    steps: list[tuple[str, str]] = []
    bad: list[str] = []
    ssh: int | None = None
    part = ""
    seen_steps = False
    for line in split_rows(text):
        s = line.strip()
        if s in ("@@steps", "@@ssh"):
            part = s
            seen_steps = seen_steps or s == "@@steps"
            continue
        if not s:
            continue
        if part == "@@steps":
            sid, _, name = s.partition(" ")
            if re.match(r"^\d+(?:_\d+)?\.\S+$", sid):
                steps.append((sid, name.strip()))
            else:
                bad.append(s)
        elif part == "@@ssh" and ssh is None:
            ssh = parse_count(s)
    if not seen_steps or (bad and not steps):
        return AuditInfo(steps=None, ssh_clients=ssh, note=bad[0] if bad else "no squeue output")
    return AuditInfo(steps=tuple(steps), ssh_clients=ssh)


def map_by_uuid_or_bus(entries: Mapping[str, GpuMapEntry], uuid: str | None, bus: str | None) -> GpuMapEntry | None:
    """The /proc/driver/nvidia entry of one nvidia-smi GPU: by uuid, else by bus id."""
    if uuid:
        for e in entries.values():
            if e.uuid == uuid:
                return e
    cb = canon_bus(bus)
    return entries.get(cb) if cb else None
