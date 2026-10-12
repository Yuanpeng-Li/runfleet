"""[WP2] `scontrol show job -d -o` parser (paths, GPU IDX/type, per-node blocks).

Values may contain spaces (`SubmitLine=sbatch … seed=44 …`, Hydra-style `key=value` arguments are common in this
fleet), so tokenizing follows §5.2.3:

1. split points are positions where whitespace is followed by `Key=` with `Key` matching `[A-Z][A-Za-z0-9_/:]*`;
2. *greedy* keys (`SubmitLine`, `Command`, `Comment`, `Reason`, `WorkDir`, `StdOut`, …) only end at the next key from
   `KNOWN_KEYS` (the closed set seen in all fixtures plus the documented ones);
3. the per-node block `Nodes=… CPU_IDs=… Mem=… GRES=…` may repeat (multi-node jobs) and is collected in order;
4. empty values are legal (`NodeList=`, vega `StdErr=`).

The multi-line form (`scontrol show job` without `-o`) is accepted too: indented continuation lines are joined.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from runfleet.models import NodeAlloc
from runfleet.util.slurmfmt import expand_hostlist, nullish, parse_index_list, parse_int, split_outside
from runfleet.util.slurmfmt import split_rows

KNOWN_KEYS = frozenset("""
Account AccrueTime AdminComment AllocNode:Sid AllocTRES ArrayJobId ArrayTaskId ArrayTaskThrottle BatchFlag BatchHost
BurstBuffer BurstBufferState CPU_IDs CPUs/Task Command Comment Container ContainerID Contiguous CoreSpec
CpusPerTres CronJob Deadline DelayBoot Dependency DerivedExitCode EligibleTime EndTime ExcNodeList Exclusive ExitCode
Features GRES GroupId HetJobId HetJobIdSet HetJobOffset JOB_GRES JobId JobName JobState KillOInInvalidDependent
KillOnInvalidDependent LastSchedEval Licenses LicensesAlloc MCS_label MailType MailUser Mem MemPerTres MinCPUsNode
MinMemoryCPU MinMemoryNode MinTmpDiskNode Network Nice NodeList Nodes NtasksPerN:B:S:C NtasksPerTRES NumCPUs NumNodes
NumTasks OverSubscribe OverTimeLimit Partition Power PreemptEligibleTime PreemptTime Prefer Priority QOS Reason Reboot
ReqB:S:C:T ReqNodeList ReqTRES Requeue Reservation ResizeTime Restarts RunTime SLUID SchedNodeList Scheduler
SecsPreSuspend Segment Socks/Node StdErr StdIn StdOut StepMgrEnabled SubmitLine SubmitTime SuspendTime
SystemComment TimeLimit TimeMin TresBind TresFreq TresPerJob TresPerNode TresPerSocket TresPerTask UserId WCKey
WorkDir
StartTime Extra SegmentSize SpreadSegments ConsolidateSegments Switches CpuFreqMin CpuFreqMax CpuFreqGov
GresEnforceBind GresAllowTaskSharing GresOneTaskPerSharing SpreadJob ExpeditedRequeue OOMKillStep Profile TRES
ClusterFeatures MemPerNode MemPerCPU ResvPorts NtasksPerNode Hold
""".split())                     # the last two lines: Slurm 25.x/26.x src/scontrol/info_job.c (review 2026-10)

GREEDY_KEYS = frozenset({"SubmitLine", "Command", "Comment", "AdminComment", "SystemComment", "Reason", "WorkDir",
                         "StdOut", "StdErr", "StdIn"})
# Paths almost never contain " Key=", so these also end at any CamelCase key: a key a newer Slurm added after StdOut
# (GresEnforceBind=, OOMKillStep=, …) must not become part of the log path.
PATH_KEYS = frozenset({"WorkDir", "StdOut", "StdErr", "StdIn"})
_CAMEL = re.compile(r"[A-Z][A-Za-z0-9]+")

_CANDIDATE = re.compile(r"(?:^|(?<=\s))([A-Z][A-Za-z0-9_/:]*)=")
_JOB_MARK = re.compile(r"^@@job (\S+)\s*$")
_GRES = re.compile(r"^(?P<name>[a-z]+):(?:(?P<type>[^:()]+):)?(?P<n>\d+)(?:\((?:IDX:)?(?P<idx>[^)]*)\))?$")


def tokenize(record: str) -> list[tuple[str, str]]:
    """One record → ordered (key, value) pairs (keys may repeat: the per-node block)."""
    marks = [(m.start(), m.end(), m.group(1)) for m in _CANDIDATE.finditer(record)]
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(marks):
        _, vstart, key = marks[i]
        j = i + 1
        while j < len(marks) and (key in GREEDY_KEYS and marks[j][2] not in KNOWN_KEYS
                                   and not (key in PATH_KEYS and _CAMEL.fullmatch(marks[j][2]))):
            j += 1
        vend = marks[j][0] if j < len(marks) else len(record)
        out.append((key, record[vstart:vend].strip()))
        i = j
    return out


@dataclass(frozen=True, slots=True)
class JobInfo:
    """What `scontrol show job -d -o` adds to a squeue row: expanded log paths, per-node GPU indices, GPU type."""
    job_id: str
    array_job_id: str | None
    array_task_id: str | None
    fields: Mapping[str, str]                  # first occurrence of every key
    per_node: tuple[NodeAlloc, ...]
    gpu_type: str | None                       # from JOB_GRES / per-node GRES, lower-cased
    stdout: str | None
    stderr: str | None                         # None = merged (empty StdErr, or equal to StdOut)
    workdir: str | None

    @property
    def merged(self) -> bool:
        return self.stderr is None


def _gres_type(gres: str | None) -> str | None:
    """`gpu:nvidia_a100-sxm4-40gb:1(IDX:2)` / `gpu:A30:1` → lower-cased type (None if untyped or not a GPU)."""
    v = nullish(gres)
    if v is None:
        return None
    for item in split_outside(v, ","):           # `gpu:h100:2(IDX:0,2)`: the IDX list has commas too
        m = _GRES.match(item.strip())
        if m and m.group("name") == "gpu" and m.group("type"):
            return m.group("type").lower()
    return None


def _node_allocs(pairs: list[tuple[str, str]]) -> tuple[NodeAlloc, ...]:
    out: list[NodeAlloc] = []
    block: dict[str, str] | None = None

    def flush() -> None:
        if block is None:
            return
        idx: tuple[int, ...] = ()
        gres = nullish(block.get("GRES"))
        if gres:      # several GPU types on one node (a full GPU plus a MIG slice): union of every gpu IDX list
            found: set[int] = set()
            for item in split_outside(gres, ","):
                if re.split(r"[:(]", item.strip(), maxsplit=1)[0] == "gpu":      # not `nic:1(IDX:0)`
                    found.update(i for grp in re.findall(r"\(IDX:([^)]*)\)", item) for i in parse_index_list(grp))
            idx = tuple(sorted(found))
        mem_mib = parse_int(block.get("Mem"))
        for node in expand_hostlist(block.get("Nodes")):
            out.append(NodeAlloc(node=node, cpu_ids=nullish(block.get("CPU_IDs")),
                                 mem_bytes=mem_mib * 1024 * 1024 if mem_mib is not None else None,
                                 gpu_indices=idx, gres=gres))

    for key, value in pairs:
        if key == "Nodes":
            flush()
            block = {"Nodes": value}
        elif block is not None and key in ("CPU_IDs", "Mem", "GRES"):
            block[key] = value
        elif block is not None:
            flush()
            block = None
    flush()
    return tuple(out)


def job_info(record: str) -> JobInfo | None:
    pairs = tokenize(record)
    fields: dict[str, str] = {}
    for k, v in pairs:
        fields.setdefault(k, v)
    for k in PATH_KEYS:          # `Comment=… StdOut=/x` spoofs a path: a repeated path key is ambiguous, use squeue's
        if sum(1 for key, _ in pairs if key == k) > 1:
            del fields[k]
    job_id = fields.get("JobId", "")
    if not job_id.isdigit():
        return None
    per_node = _node_allocs(pairs)
    gtype = _gres_type(fields.get("JOB_GRES")) or next((t for t in (_gres_type(n.gres) for n in per_node) if t), None)
    stdout = nullish(fields.get("StdOut"))
    stderr = nullish(fields.get("StdErr"))
    if stderr == stdout:
        stderr = None
    return JobInfo(job_id=job_id, array_job_id=nullish(fields.get("ArrayJobId")),
                   array_task_id=nullish(fields.get("ArrayTaskId")), fields=fields, per_node=per_node,
                   gpu_type=gtype, stdout=stdout, stderr=stderr, workdir=nullish(fields.get("WorkDir")))


def _records(lines: list[str]) -> list[str]:
    """Group lines into records: a record starts at a line beginning `JobId=`; indented lines continue it."""
    out: list[str] = []
    for line in lines:
        if line.startswith("JobId="):
            out.append(line.rstrip())
        elif out and line[:1].isspace() and line.strip():
            out[-1] += " " + line.strip()
    return out


@dataclass(slots=True)
class ScontrolBlock:
    """Output of one `scontrol show job -d -o <id>` call: its records and any other lines (the id's error)."""
    records: list[JobInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def left_controller(self) -> bool:
        """`slurm_load_jobs error: Invalid job id specified` → the job left the controller (MinJobAge)."""
        return not self.records and any("Invalid job id" in e for e in self.errors)

    def record_for(self, array_task: str | None) -> JobInfo | None:
        """The record whose ArrayTaskId equals the squeue row's ArrayTaskID (a pending range like `1-20%1` for a
        summary row, None for plain jobs); no match → None."""
        want = array_task.strip("[]") if array_task else None
        for r in self.records:
            if r.array_task_id == want:
                return r
        return None


def parse_scontrol(text: str) -> dict[str | None, ScontrolBlock]:
    """`@@job <id>`-delimited loop output (§5.2.2 `scontrol_cmd`) → {id: block}. Text without markers (fixtures, a
    single call) is one block under the key None."""
    blocks: dict[str | None, list[str]] = {}
    current: str | None = None
    for line in split_rows(text):
        m = _JOB_MARK.match(line)
        if m:
            current = m.group(1)
            blocks.setdefault(current, [])
            continue
        blocks.setdefault(current, []).append(line)
    out: dict[str | None, ScontrolBlock] = {}
    for jid, lines in blocks.items():
        if jid is None and not any(line.strip() for line in lines):
            continue
        block = ScontrolBlock()
        for rec in _records(lines):
            info = job_info(rec)
            if info is not None:
                block.records.append(info)
        block.errors = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("JobId=")
                        and not ln[:1].isspace()]
        out[jid] = block
    return out


def parse_node_record(text: str) -> dict[str, str]:
    """`scontrol show node -o` → key/value dict (first occurrence wins); values with spaces (`OS=Linux … 2026`) kept."""
    out: dict[str, str] = {}
    for k, v in tokenize(text.strip()):
        out.setdefault(k, v)
    return out
