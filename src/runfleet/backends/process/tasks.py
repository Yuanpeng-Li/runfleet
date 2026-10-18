"""[WP2] Process-tree grouping, sticky identity, naming, noise filter, exit scan (§5.3.2).

A workstation *task* is the user's top-level launcher plus all its current descendants:

- discovery starts from an *interesting* own process (holds a GPU, or non-noise with etime >= task_min_age and
  cpu >= task_min_cpu) and climbs `ppid` while the parent is an own, non-noise process other than pid 1;
- identity is `(root pid, root start epoch)`, id `p<pid>-<start>`; a task lives while its root appears in a
  successful `process.ps` sample (± 5 s on the start), whatever its CPU/GPU use;
- membership is recomputed from every sample and is not filtered by interest.

Everything here is pure; `ProcessCollector` owns the state between ticks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from runfleet.backends.common import StatInfo, with_stat
from runfleet.backends.process.parse_ps import FdInfo, is_regular_log
from runfleet.models import (GPU, Job, JobGpuUsage, JobKey, JobKind, JobState, LogRef, NodeAlloc, Proc, ResourceAlloc)
from runfleet.util.humanize import task_label as label
from runfleet.util.logscan import EXIT_PATTERN, EXIT_SCAN_LINES   # shared with job detail's "Why it ended"

START_TOLERANCE_S = 5

NOISE_DEFAULT: tuple[str, ...] = (
    r"^-?(\S*/)?(ba|z|da|k|tc|fi)?sh$",
    r"^sshd",
    r"^(\S*/)?systemd --user",
    r"^\(sd-pam\)",
    r"^(\S*/)?(tmux|screen|SCREEN)",
    r"sftp-server|ssh-agent|gpg-agent|dbus-daemon",
    r"\.vscode-server/|code-server|cursor-server",
    r"^(\S*/)?(nvitop|htop|top|btop|less|man|watch)( |$)",
    r"python\S* (-m )?\S*(nvitop|gpustat)",
)



GPU_NOTE = "GPU-level utilisation; shared GPUs count other processes too"


class NoiseFilter:
    """Built-in noise regexes (login shells, sshd, systemd --user, tmux, editors' servers, monitors) + config
    `ignore` regexes, matched against `args`."""

    def __init__(self, extra: Iterable[str] = ()) -> None:
        self._patterns = [re.compile(p) for p in (*NOISE_DEFAULT, *extra)]

    def __call__(self, args: str) -> bool:
        return any(p.search(args) for p in self._patterns)


def task_id(pid: int, start: int) -> str:
    return f"p{pid}-{start}"


_TASK_ID = re.compile(r"p([0-9]{1,10})-([0-9]{1,12})")


def parse_task_id(text: str) -> tuple[int, int] | None:
    """(root pid, start epoch) of a full task id `p<pid>-<start>`; None for anything else."""
    m = _TASK_ID.fullmatch(text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def nearest_task(job_id: str, jobs: Mapping[str, Job]) -> Job | None:
    """The task `job_id` (a full id `p<pid>-<start>`) names when no id matches exactly: same root pid, start within
    START_TOLERANCE_S. A task's start is `date - etimes`, recomputed by every collector (each `runfleet` call), so
    the id another call printed may be a second off; the closest start wins (then the smaller id). None: not a full
    task id, or no such task."""
    want = parse_task_id(job_id)
    if want is None:
        return None
    best: tuple[int, str, Job] | None = None
    for key, job in jobs.items():
        if job.kind is not JobKind.TASK or (got := parse_task_id(key)) is None or got[0] != want[0]:
            continue
        delta = abs(got[1] - want[1])
        if delta <= START_TOLERANCE_S and (best is None or (delta, key) < best[:2]):
            best = (delta, key, job)
    return best[2] if best is not None else None


@dataclass(frozen=True, slots=True)
class TaskRecord:
    key: JobKey
    root_pid: int
    root_start: int              # remote epoch, fixed at discovery
    members: tuple[int, ...]     # root first


@dataclass(frozen=True, slots=True)
class Grouping:
    active: dict[str, TaskRecord]
    ended: tuple[TaskRecord, ...]
    new: tuple[TaskRecord, ...]


def _descendants(root: int, children: Mapping[int, list[int]], claimed: set[int]) -> list[int]:
    out: list[int] = []
    queue = list(children.get(root, ()))
    seen = {root}
    while queue:
        pid = queue.pop(0)
        if pid in seen or pid in claimed:
            continue
        seen.add(pid)
        out.append(pid)
        queue.extend(children.get(pid, ()))
    return out


def group_tasks(procs: Sequence[Proc], epoch: int, *, gpu_pids: frozenset[int], prev: Mapping[str, TaskRecord],
                host: str, noise: NoiseFilter, min_age: float, min_cpu: float) -> Grouping:
    """One successful `process.ps` sample → active tasks (sticky ones first), ended tasks, newly discovered ones."""
    by_pid = {p.pid: p for p in procs}
    children: dict[int, list[int]] = {}
    for p in procs:
        children.setdefault(p.ppid, []).append(p.pid)

    def start_of(p: Proc) -> int:
        return epoch - p.etime_s

    active: dict[str, TaskRecord] = {}
    ended: list[TaskRecord] = []
    claimed: set[int] = set()
    for rec in sorted(prev.values(), key=lambda r: (r.root_start, r.root_pid)):
        root_proc = by_pid.get(rec.root_pid)
        if root_proc is None or abs(start_of(root_proc) - rec.root_start) > START_TOLERANCE_S \
                or rec.root_pid in claimed or _defunct(root_proc):
            ended.append(rec)
            continue
        claimed.add(rec.root_pid)
        members = (rec.root_pid, *_descendants(rec.root_pid, children, claimed))
        claimed.update(members)
        active[rec.key.id] = TaskRecord(rec.key, rec.root_pid, rec.root_start, members)

    def interesting(p: Proc) -> bool:
        if _defunct(p):
            return False
        if p.pid in gpu_pids:
            return True
        return p.etime_s >= min_age and p.cpu_pct >= min_cpu and not noise(p.args)

    def climb(p: Proc) -> Proc:
        cur, seen = p, {p.pid}
        while True:
            parent = by_pid.get(cur.ppid)
            if parent is None or parent.pid <= 1 or parent.pid in seen or noise(parent.args):
                return cur
            seen.add(parent.pid)
            cur = parent

    new: list[TaskRecord] = []
    for p in sorted(procs, key=lambda q: q.pid):
        if p.pid in claimed or not interesting(p):
            continue
        root = climb(p)
        if root.pid in claimed:
            continue
        claimed.add(root.pid)
        members = (root.pid, *_descendants(root.pid, children, claimed))
        claimed.update(members)
        start = start_of(root)
        rec = TaskRecord(JobKey(host, task_id(root.pid, start)), root.pid, start, members)
        active[rec.key.id] = rec
        new.append(rec)
    return Grouping(active=active, ended=tuple(ended), new=tuple(new))


def _defunct(p: Proc) -> bool:
    """A zombie (or dead) process: it has exited and only waits for a parent that does not reap it."""
    return p.stat[:1] in ("Z", "X") if p.stat else False


# ---------------------------------------------------------------- Job construction
def _state(members: Sequence[Proc]) -> JobState:
    codes = [m.stat[:1] for m in members if m.stat]
    if any(c in "RSDI" for c in codes):
        return JobState.RUNNING
    if codes and all(c in "Tt" for c in codes):
        return JobState.STOPPED
    return JobState.RUNNING


def gpu_usage(pids: Iterable[int], gpus: Sequence[GPU], sampled_at: datetime | None) -> tuple[list[GPU],
                                                                                             JobGpuUsage | None]:
    """GPUs holding any of `pids` and the task's GPU usage (mean/max of those GPUs, memory of these pids)."""
    pidset = set(pids)
    held = [g for g in gpus if any(p.pid in pidset for p in g.processes)]
    if not held:
        return [], None
    utils = [g.util_pct for g in held if g.util_pct is not None]
    mems = [p.used_mem_mib for g in held for p in g.processes if p.pid in pidset]
    known = [m for m in mems if m is not None]       # `[N/A]` (some drivers, MIG) is unknown, not 0 MiB
    mem_used = sum(known) if known else None
    note = GPU_NOTE + ("; memory of some processes unknown (lower bound)" if known and len(known) < len(mems) else "")
    totals = [g.mem_total_mib for g in held if g.mem_total_mib is not None]
    return held, JobGpuUsage(source="nvidia-smi", n_gpus=len(held),
                             util_mean_pct=sum(utils) / len(utils) if utils else None,
                             util_sum_pct=sum(utils) if utils else None,
                             util_peak_pct=max(utils) if utils else None, mem_used_mib=mem_used,
                             mem_total_mib=sum(totals) if totals else None, sampled_at=sampled_at, note=note)


def _name(root: Proc, members: Sequence[Proc], gpu_pids: frozenset[int]) -> str:
    base = label(root.args)
    child_labels = [label(m.args) for m in members[1:] if m.pid in gpu_pids]
    others = [c for c in child_labels if c != base]
    if not others:
        return base
    common = max(set(others), key=others.count)
    k = len(others)
    return f"{base} ×{k} ({common})" if k > 1 else f"{base} ({common})"


def _logs(rec: TaskRecord, fds: Mapping[int, FdInfo], gpu_pids: frozenset[int]) -> tuple[LogRef, ...]:
    """Root fd1/fd2 if they are regular files, else the first GPU-holding member's."""
    candidates = [rec.root_pid, *[p for p in rec.members[1:] if p in gpu_pids]]
    for pid in candidates:
        fd = fds.get(pid)
        if fd is None:
            continue
        out_ok, err_ok = is_regular_log(fd.stdout), is_regular_log(fd.stderr)
        if not out_ok and not err_ok:
            continue
        merged = out_ok and fd.stdout == fd.stderr
        refs = []
        if out_ok:
            refs.append(LogRef(stream="stdout", path=fd.stdout, merged=merged, source="procfd"))
        if err_ok:
            refs.append(LogRef(stream="stderr", path=fd.stderr, merged=merged, source="procfd"))
        return tuple(refs)
    return ()


def _held_alloc(host: str, held: Sequence[GPU]) -> tuple[NodeAlloc, ...]:
    """The GPUs a task holds as the one-node allocation the UI reads (NODE / REASON "gpu 0,1", GPU tab)."""
    if not held:
        return ()
    return (NodeAlloc(node=host, cpu_ids=None, mem_bytes=None, gpu_indices=tuple(sorted(g.index for g in held)),
                      gres=None),)


def task_job(rec: TaskRecord, by_pid: Mapping[int, Proc], epoch: int, *, user: str, gpus: Sequence[GPU],
             gpus_at: datetime | None, gpu_pids: frozenset[int], fds: Mapping[int, FdInfo],
             stats: Mapping[str, StatInfo], observed_at: datetime, hints: tuple[str, ...] = ()) -> Job:
    """An active own task → Job (kind TASK)."""
    members = [by_pid[p] for p in rec.members if p in by_pid]
    root = by_pid[rec.root_pid]
    held, usage = gpu_usage(rec.members, gpus, gpus_at)
    names = sorted({g.name for g in held})
    root_fd = fds.get(rec.root_pid)
    return Job(
        key=rec.key, kind=JobKind.TASK, name=_name(root, members, gpu_pids), user=user, state=_state(members),
        raw_state=_state(members).name.lower(), job_id_raw=str(rec.root_pid),
        start_time=datetime.fromtimestamp(rec.root_start, tz=UTC), elapsed=timedelta(seconds=root.etime_s),
        alloc=ResourceAlloc(nodes=None, cpus=None, mem_bytes=None, gpus=len(held),
                            gpu_type=names[0] if len(names) == 1 else (", ".join(names) or None), tres={},
                            gpu_type_source="nvidia-smi" if held else None, per_node=_held_alloc(rec.key.host, held)),
        workdir=root_fd.cwd if root_fd else None, command=root.args,
        logs=tuple(with_stat(r, dict(stats)) for r in _logs(rec, fds, gpu_pids)),
        pids=tuple(m.pid for m in members), cpu_pct=sum(m.cpu_pct for m in members),
        rss_bytes=sum(m.rss_bytes for m in members), gpu=usage, hints=hints, is_mine=True, source="ps",
        observed_at=observed_at,
        raw={"root_pid": str(rec.root_pid), "root_start": str(rec.root_start),
             "gpu_indices": ",".join(str(g.index) for g in held)},
    )


def other_user_jobs(gpu_procs: Sequence[Proc], gpus: Sequence[GPU], sample_epoch: int, *, host: str, me: str,
                    observed_at: datetime) -> dict[str, Job]:
    """Other users' GPU processes as single-process, read-only tasks (shown when the user filter includes others)."""
    out: dict[str, Job] = {}
    for p in gpu_procs:
        # procps truncates long names to 7 chars + "+" in a default-width column ("alice000+"); the command asks for
        # `user:64=`, but an old row (or another ps) may still be truncated
        if p.user == me or (p.user.endswith("+") and len(p.user) > 1 and me.startswith(p.user[:-1])):
            continue
        start = sample_epoch - p.etime_s
        key = JobKey(host, task_id(p.pid, start))
        held, usage = gpu_usage([p.pid], gpus, observed_at)
        names = sorted({g.name for g in held})
        out[key.id] = Job(
            key=key, kind=JobKind.TASK, name=label(p.args), user=p.user, state=JobState.RUNNING, raw_state="running",
            job_id_raw=str(p.pid), start_time=datetime.fromtimestamp(start, tz=UTC),
            elapsed=timedelta(seconds=p.etime_s),
            alloc=ResourceAlloc(nodes=None, cpus=None, mem_bytes=None, gpus=len(held),
                                gpu_type=names[0] if len(names) == 1 else None, tres={},
                                gpu_type_source="nvidia-smi" if held else None, per_node=_held_alloc(host, held)),
            command=p.args, pids=(p.pid,), cpu_pct=p.cpu_pct, rss_bytes=p.rss_bytes, gpu=usage, is_mine=False,
            source="ps", observed_at=observed_at, raw={"root_pid": str(p.pid), "root_start": str(start)},
        )
    return out


# ---------------------------------------------------------------- ended tasks
def classify_exit(text: str | None) -> tuple[JobState, str, str]:
    """Last 50 log lines → (state, raw_state, hint): a Traceback/OOM/Killed/segfault/`…Error:` line → FAILED,
    `exited (log: <first matching line, 60 chars>)`; nothing (or no log) → EXITED (exit status unknown)."""
    if text:
        for line in text.splitlines()[-EXIT_SCAN_LINES:]:
            if EXIT_PATTERN.search(line):
                excerpt = line.strip()[:60]
                return JobState.FAILED, f"exited (log: {excerpt})", f"log: {excerpt}"
    return JobState.EXITED, "exited", "ended, exit status unknown"
