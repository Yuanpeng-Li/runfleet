"""[WP3] FakeService(FleetAPI) over a static FleetStore built from model objects.

The world mirrors the DESIGN.md §7.3 mockups (k2 / vega / lyra Slurm, labstation / lynx workstations) so snapshots
can be compared with them by eye. Everything is in memory and driven by a ``ManualClock``; tests mutate the world
through the helpers at the bottom of :class:`FakeService` (``inject``, ``put_job``, ``push_log`` …), which notify
store subscribers exactly like the real core would.
"""

from __future__ import annotations

import asyncio
import dataclasses
import tempfile
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from runfleet.backends.base import ActionParam, ActionPlan, ActionSpec, CollectorOutput
from runfleet.config import (FleetConfig, GeneralSettings, HostConfig, NotifySettings, SlurmSettings,
                             UISettings)
from runfleet.core.clock import ManualClock
from runfleet.core import logsearch
from runfleet.core.forwards import PortForward
from runfleet.core.logsearch import LogQuery, LogSearchResult
from runfleet.core.service import LogPeek, LogUpdate
from runfleet.core.store import ChangeSet, FleetStore, HostState
from runfleet.errors import ActionError, AttachBlocked, AttachConsentRequired, RunFleetError
from runfleet.models import (
    GPU, ActionResult, AttachMode, AttachOffer, AttachState, AttachStatus, Capabilities, CgroupStats, ConnState,
    DetectReport, DoctorCheck, DoctorReport, ErrorKind, GPUProcess, GresCount, HostError, HostMetrics, HostSnapshot,
    HostStatus, Job, JobDetail, JobGpuUsage, JobKey, JobKind, JobLive, JobState, JobStep, LogRef, Node, NodeAlloc,
    NodeInspect, NodeJob, PartitionSummary, Proc, ResourceAlloc, SectionStatus, StateGroup, Visibility,
)
from runfleet.probe.section import SectionOutcome, SectionResult
from runfleet.util.efficiency import efficiency
from runfleet.util.efficiency import lines as efficiency_lines

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)
ME = "alice0001"
GiB = 1 << 30


# ---------------------------------------------------------------- store
class FakeHostState(HostState):
    """HostState with the §3.7 staleness rule implemented (the real one lands with WP1)."""

    def stale(self, now: datetime) -> bool:
        st = self.status
        if self.snapshot is None or st.state not in (ConnState.OK, ConnState.DEGRADED):
            return True
        if st.last_ok is None:
            return True
        tick = st.tick_interval or 15.0
        return (now - st.last_ok).total_seconds() > 2 * tick + 5


class FakeStore(FleetStore):
    """In-memory FleetStore: same read API as the real one, plus test mutators that emit ChangeSets."""

    def __init__(self, states: Iterable[HostState]) -> None:
        self.hosts = {s.config.name: s for s in states}
        self.idle_paused = False
        self._subs: list[Callable[[ChangeSet], None]] = []

    # read API (§3.7)
    def subscribe(self, cb: Callable[[ChangeSet], None]) -> Callable[[], None]:
        self._subs.append(cb)
        return lambda: self._subs.remove(cb) if cb in self._subs else None

    def _selected(self, hosts: Iterable[str] | None) -> list[HostState]:
        if hosts is None:
            return list(self.hosts.values())
        wanted = set(hosts)
        return [s for name, s in self.hosts.items() if name in wanted]

    def jobs(self, hosts: Iterable[str] | None = None) -> Iterator[Job]:
        for s in self._selected(hosts):
            if s.snapshot is not None:
                yield from s.snapshot.jobs.values()

    def job(self, key: JobKey) -> Job | None:
        s = self.hosts.get(key.host)
        if s is None or s.snapshot is None:
            return None
        return s.snapshot.jobs.get(key.id)

    def gpus(self, hosts: Iterable[str] | None = None) -> Iterator[GPU]:
        for s in self._selected(hosts):
            if s.snapshot is not None:
                yield from s.snapshot.gpus

    def counts(self, hosts: Iterable[str] | None = None) -> Mapping[StateGroup, int]:
        out = {g: 0 for g in StateGroup}
        for j in self.jobs(hosts):
            out[j.state.group] += 1
        return out

    def apply(self, host: str, out: CollectorOutput, taken_at: datetime) -> ChangeSet:   # pragma: no cover
        raise NotImplementedError("FakeStore is mutated through FakeService helpers")

    def set_status(self, host: str, status: HostStatus) -> ChangeSet:
        self.hosts[host].status = status
        return self.emit(ChangeSet(host, status=True))

    def set_idle_paused(self, paused: bool) -> ChangeSet:
        self.idle_paused = paused
        return self.emit(ChangeSet("*", paused=True))

    # test mutators
    def emit(self, cs: ChangeSet) -> ChangeSet:
        for cb in list(self._subs):
            cb(cs)
        return cs

    def replace_snapshot(self, host: str, **changes: Any) -> None:
        snap = self.hosts[host].snapshot
        assert snap is not None
        self.hosts[host].snapshot = dataclasses.replace(snap, **changes)

    def put_job(self, job: Job) -> ChangeSet:
        snap = self.hosts[job.key.host].snapshot
        assert snap is not None
        jobs = dict(snap.jobs)
        added = job.key.id not in jobs
        jobs[job.key.id] = job
        self.replace_snapshot(job.key.host, jobs=jobs)
        ids = frozenset({job.key.id})
        return self.emit(ChangeSet(job.key.host, jobs_added=ids if added else frozenset(),
                                   jobs_changed=frozenset() if added else ids))

    def remove_job(self, key: JobKey) -> ChangeSet:
        snap = self.hosts[key.host].snapshot
        assert snap is not None
        jobs = {k: v for k, v in snap.jobs.items() if k != key.id}
        self.replace_snapshot(key.host, jobs=jobs)
        return self.emit(ChangeSet(key.host, jobs_removed=frozenset({key.id})))


# ---------------------------------------------------------------- world builders
def _ago(seconds: float) -> datetime:
    return T0 - timedelta(seconds=seconds)


def _hms(h: int = 0, m: int = 0, s: int = 0, d: int = 0) -> timedelta:
    return timedelta(days=d, hours=h, minutes=m, seconds=s)


def slurm_job(host: str, jid: str, name: str, state: JobState, *, elapsed: timedelta | None = None,
              limit: timedelta | None = _hms(d=3), gpus: int = 1, gpu_type: str | None = "nvidia_a100-sxm4-40gb",
              util: float | None = None, mem_used_gib: float | None = None, mem_total_gib: float | None = None,
              node: str | None = None, reason: str | None = None, partition: str = "gpu-preempt",
              qos: str = "normal", ended_ago: float | None = None, exit_code: int | None = None,
              idx: tuple[int, ...] = (), cpus: int = 8, mem_gib: int = 48, held: bool = False,
              raw_state: str | None = None, hints: tuple[str, ...] = (), user: str = ME,
              submit_ago: float = 18 * 3600, workdir: str = "/shared/pilab/pilab/orb") -> Job:
    nodes = (node,) if node else ()
    start = None
    if elapsed is not None and state.group is not StateGroup.QUEUED:
        start = (T0 - elapsed) if ended_ago is None else _ago(ended_ago) - elapsed
    end = _ago(ended_ago) if ended_ago is not None else None
    usage = None
    if util is not None or mem_used_gib is not None:
        usage = JobGpuUsage(
            "sstat", gpus, util, util * gpus if util is not None else None, min(100.0, (util or 0) + 1),
            mem_used_gib * 1024 if mem_used_gib is not None else None,
            mem_total_gib * 1024 if mem_total_gib is not None else None, T0 - timedelta(seconds=30),
            "Slurm accounting · refreshed every 60 s",
        )
    per_node = (NodeAlloc(node, "64-71", mem_gib * GiB, idx, f"gpu:{gpu_type}:{gpus}"),) if node and gpus else ()
    log_path = f"{workdir}/logs/{name}_{jid}.out"
    return Job(
        key=JobKey(host, jid), kind=JobKind.SLURM, name=name, user=user, state=state,
        raw_state=raw_state or state.name, job_id_raw=jid.split("_")[0], reason=reason, held=held,
        submit_time=_ago(submit_ago), start_time=start, end_time=end, elapsed=elapsed, time_limit=limit,
        exit_code=exit_code, partition=partition, qos=qos, account="campus01",
        alloc=ResourceAlloc(len(nodes) or 1, cpus, mem_gib * GiB, gpus, gpu_type if gpus else None,
                            {"cpu": str(cpus), "mem": f"{mem_gib}G", "node": "1", "billing": str(cpus),
                             "gres/gpu": str(gpus)}, per_node, "tres-alloc"),
        nodes=nodes, nodelist=node, batch_host=node, workdir=workdir, command=f"{workdir}/jobs/train.sbatch",
        logs=(LogRef("stdout", log_path, source="scontrol", size=663, mtime=T0 - timedelta(seconds=12)),
              LogRef("stderr", log_path, merged=True, source="scontrol")),
        gpu=usage, hints=hints, is_mine=user == ME, source="squeue" if ended_ago is None else "sacct",
        observed_at=T0, raw={"JobId": jid, "JobName": name, "Partition": partition, "QOS": qos},
    )


def task_job(host: str, pid: int, name: str, *, elapsed: timedelta, gpus: tuple[int, ...], gpu_model: str,
             util: float, mem_mib: float, mem_total_mib: float, pids: tuple[int, ...], user: str = "alice07",
             mine: bool = True, cwd: str = "/home/alice07/orb", state: JobState = JobState.RUNNING) -> Job:
    start = int((T0 - elapsed).timestamp())
    return Job(
        key=JobKey(host, f"p{pid}-{start}"), kind=JobKind.TASK, name=name, user=user, state=state,
        raw_state="running", job_id_raw=str(pid), start_time=T0 - elapsed, elapsed=elapsed,
        alloc=ResourceAlloc(1, None, None, len(gpus), gpu_model, {}, (NodeAlloc(host, None, None, gpus, None),),
                            "nvidia-smi"),
        nodes=(host,), workdir=cwd, command="python -m orb.launch --manifest manifests/cifar.json --per-gpu 1",
        logs=(LogRef("stdout", f"{cwd}/logs/launch_{pid}.log", source="procfd", size=48213,
                     mtime=T0 - timedelta(seconds=4)),
              LogRef("stderr", f"{cwd}/logs/launch_{pid}.log", merged=True, source="procfd")),
        pids=pids, cpu_pct=100.8, rss_bytes=9 * GiB,
        gpu=JobGpuUsage("nvidia-smi", len(gpus), util, util * len(gpus), util, mem_mib, mem_total_mib, T0,
                        "GPU-level utilisation; shared GPUs count other processes too"),
        is_mine=mine, source="ps", observed_at=T0,
    )


def _gpu(host: str, index: int, name: str, util: float, used_gib: float, total_mib: float, temp: float,
         power: float, limit: float, procs: tuple[GPUProcess, ...] = ()) -> GPU:
    return GPU(host, index, f"GPU-{host}-{index}", name, util, used_gib * 1024, total_mib,
               used_gib * 1024 / total_mib * 100, temp, power, limit, 30.0, "P2" if util else "P8", "nvidia-smi",
               procs, T0)


def _node(name: str, state: str, used: int, total: int, gtype: str, cpus: tuple[int, int, int],
          mem: tuple[int, int], flags: str = "") -> Node:
    return Node(name, ("gpu-preempt",), state, frozenset(flags), True, cpus[0], cpus[1], 0, cpus[2], mem[1] * 1024,
                mem[0] * 1024, (mem[1] - mem[0]) * 1024, (GresCount("gpu", gtype, total, used, None),), None)


def _nj(jid: str, user: str, gpus: int | None, elapsed: timedelta, limit: timedelta | None, name: str, *,
        real: str | None = None, account: str = "lab", exact: bool = True) -> NodeJob:
    return NodeJob(jid, user, real, account, "gpu-preempt", name, "R", gpus, exact, None,
                   int(elapsed.total_seconds()), int(limit.total_seconds()) if limit is not None else None, user == ME)


def _caps(user: str, login: str, profile: tuple[str, ...], **kw: Any) -> Capabilities:
    return Capabilities(remote_user=user, uid=610001, login_host=login, tz_offset=timedelta(hours=-5),
                        clock_skew=timedelta(0), tools=frozenset(kw.pop("tools", ("squeue", "sacct", "sinfo",
                                                                                   "scontrol", "sstat", "less"))),
                        profile=profile, **kw)


def _completed(host: str, base: int, n: int, prefix: str) -> list[Job]:
    return [slurm_job(host, str(base + i), f"{prefix}--s{40 + i}--a01", JobState.COMPLETED, elapsed=_hms(2, 10 * i),
                      ended_ago=3600 * (2 + 3 * i), exit_code=0, util=None, node=f"n-3{i + 1}-01")
            for i in range(n)]


def _status(name: str, state: ConnState, *, ok_ago: float | None, latency: float | None = None,
            tick: float = 15, error: HostError | None = None, retry_in: float | None = None,
            mux: str = "user") -> HostStatus:
    sections = {
        s: SectionStatus(s, "ok", True, 0, 120, _ago(ok_ago or 0))
        for s in (("slurm.squeue", "slurm.sacct", "slurm.sinfo", "slurm.sstat") if tick == 15
                  else ("process.ps", "nvidia.gpu", "nvidia.apps", "procfs.host", "process.fds"))
    }
    return HostStatus(
        name=name, state=state, error=error, last_ok=_ago(ok_ago) if ok_ago is not None else None,
        last_attempt=T0, latency_s=latency, consecutive_failures=0 if state is ConnState.OK else 2,
        next_poll_at=T0 + timedelta(seconds=retry_in if retry_in is not None else tick / 2), tick_interval=tick,
        mux=mux,  # type: ignore[arg-type]
        ssh_log=f"/tmp/rf-ssh-{name}.log", sections=sections,
    )


def build_world() -> tuple[FleetConfig, list[HostState]]:
    """The five hosts of mockup A, at T0."""
    hosts = (
        HostConfig("k2", "k2", slurm=SlurmSettings(protected_partitions=("labpi",), protected_qos=("labpi",))),
        HostConfig("vega", "vega", connect_timeout=120),
        HostConfig("lyra", "lyra"),
        HostConfig("labstation", "labstation"),
        HostConfig("lynx", "lynx"),
    )
    config = FleetConfig(hosts=hosts, general=GeneralSettings(), ui=UISettings())

    # k2 ----------------------------------------------------------------------------------------------------------
    k2_jobs = [
        slurm_job("k2", "2332861", "orb-swp--train--cifar-warmup--s43--a02", JobState.RUNNING,
                  elapsed=_hms(4, 7, 12), util=98, mem_used_gib=24.3, mem_total_gib=40, node="n-40-01", idx=(2,),
                  qos="gpu-priority"),
        slurm_job("k2", "2332862", "orb-swp--train--cifar-const--s43--a02", JobState.RUNNING,
                  elapsed=_hms(4, 7, 12), util=97, mem_used_gib=22.8, mem_total_gib=40, node="n-40-01", idx=(1,),
                  qos="gpu-priority", submit_ago=18 * 3600 - 1),
        slurm_job("k2", "2332863", "orb-swp--train--cifar-cosinewarm--s43--a02", JobState.RUNNING,
                  elapsed=_hms(4, 7, 12), util=98, mem_used_gib=23.1, mem_total_gib=40, node="n-40-01", idx=(3,),
                  qos="gpu-priority", submit_ago=18 * 3600 - 2),
        slurm_job("k2", "2333760", "lmft-x07-sampler-s0", JobState.RUNNING, elapsed=_hms(9, 2, 26), gpus=2,
                  gpu_type="nvidia_h200_nvl", util=86, mem_used_gib=96.1, mem_total_gib=281, node="n-38-01",
                  idx=(0, 1), submit_ago=10 * 3600),
        slurm_job("k2", "2334564", "smoke-test", JobState.PENDING, limit=_hms(1), gpu_type="nvidia_geforce_rtx_3090",
                  reason="Priority", submit_ago=600),
        slurm_job("k2", "2330539", "lmft-x0-lr1e-5", JobState.FAILED, elapsed=_hms(0, 59, 5), exit_code=1,
                  ended_ago=2 * 86400, node="n-38-01", gpu_type="nvidia_h200_nvl", submit_ago=3 * 86400),
        slurm_job("k2", "2331790", "labpi-sweep", JobState.PENDING, partition="labpi", qos="labpi",
                  reason="Resources", submit_ago=900),
        *_completed("k2", 2331214, 7, "orb-swp--train--imagenette-warmup"),
    ]
    k2_parts = (PartitionSummary("gpu-preempt", 26, 164, 31, 4, {"nvidia_geforce_rtx_3090": (14, 68),
                                                                  "nvidia_a100_80gb": (0, 12),
                                                                  "nvidia_h200_nvl": (3, 4)},
                                 {"mixed": 19, "allocated": 3, "planned": 2, "idle": 2}),
                PartitionSummary("labpi", 4, 16, 2, 0, {"nvidia_a100-sxm4-40gb": (2, 16)}, {"mixed": 4}))
    occupants = {      # every user's jobs per node (§5.2.9): the host detail panel and its JOBS column
        "n-31-01": (_nj("2331907", "bob", 4, _hms(20, 11, 5), _hms(d=2), "ddp-train-r50", real="Bob Example"),
                    _nj("2333115", "carol", 3, _hms(1, 42, 30), _hms(4), "llama-ft-lora", real="Carol Example"),
                    _nj("2333301", "dave", 1, _hms(0, 18, 2), _hms(d=1), "jupyter", real="Dave Sample",
                        account="physics")),
        "n-31-04": (_nj("2330422", "erin", 2, _hms(2, 3, 0, d=1), _hms(d=3), "md-sim-04", real="Erin Sample",
                        account="chem"),
                    _nj("2333412", "frank", 1, _hms(3, 10, 44), None, "vasp-relax", account="bio")),
        "n-31-05": (),
        "n-32-03": (_nj("2329950", "grace", 4, _hms(5, 31, 0, d=2), _hms(d=7), "rl-ppo-atari",
                        real="Grace Placeholder", account="physics", exact=False),
                    _nj("2332001", "bob", 4, _hms(7, 2, 51), _hms(d=1), "eval-sweep", real="Bob Example")),
        "n-38-01": (_nj("2333760", ME, 2, _hms(9, 2, 26), _hms(d=3), "lmft-x07-sampler-s0", real="Alice Example"),
                    _nj("2333590", "heidi", 2, _hms(11, 40, 9), _hms(d=1), "diffusion-xl", real="Heidi Demo",
                        account="chem")),
        "n-40-01": tuple(_nj(j, ME, 1, _hms(18, 54, 4), _hms(d=3), n, real="Alice Example")
                         for j, n in (("2332861", "orb-swp--train--cifar-warmup--s43--a02"),
                                      ("2332862", "orb-swp--train--cifar-const--s43--a02"),
                                      ("2332863", "orb-swp--train--cifar-cosinewarm--s43--a02"))),
    }
    k2_nodes = tuple(dataclasses.replace(n, jobs=occupants[n.name], jobs_at=T0) for n in (
        _node("n-31-01", "mixed", 8, 8, "nvidia_geforce_rtx_3090", (34, 30, 64), (282, 500)),
        _node("n-31-04", "mixed", 3, 4, "tesla_v100-pcie-16gb", (6, 50, 56), (72, 125)),
        _node("n-31-05", "planned", 0, 12, "nvidia_a100_3g.40gb", (0, 64, 64), (0, 503)),
        _node("n-32-03", "mixed", 8, 8, "nvidia_geforce_rtx_3090", (36, 28, 64), (368, 503), flags="-"),
        _node("n-38-01", "mixed", 4, 4, "nvidia_h200_nvl", (40, 24, 64), (400, 1007)),
        _node("n-40-01", "mixed", 3, 4, "nvidia_a100-sxm4-40gb", (24, 40, 64), (144, 503)),
    ))
    k2 = HostSnapshot("k2", T0, _caps(ME, "login02", ("slurm", "procfs"), slurm_version="25.05.8",
                                      cluster_name="k2", gpu_tres=True,
                                      slurm_config={"PreemptMode": "REQUEUE", "MinJobAge": "300",
                                                    "AccountingStorageType": "accounting_storage/slurmdbd"},
                                      associations=(("k2", "campus01", "", "gpu-priority"),
                                                    ("k2", "campus01", "", "normal"),
                                                    ("k2", "campus01", "", "labpi"))),
                      {j.key.id: j for j in k2_jobs}, (), (), k2_nodes, k2_parts,
                      HostMetrics(32, (2.3, 2.1, 2.0), 7.0, 256 * GiB, 180 * GiB, 0, 0, 3e6, T0))

    # vega --------------------------------------------------------------------------------------------------------
    vega_jobs = [
        slurm_job("vega", "723954", "lmft-x07-sampler-s1", JobState.RUNNING, elapsed=_hms(5, 54, 27), gpus=2,
                  gpu_type="nvidia_h200", util=85, mem_used_gib=120, mem_total_gib=282, node="n-38-01",
                  limit=_hms(d=2), partition="h200"),
        slurm_job("vega", "723952", "lmft-x07-trainer-s1", JobState.RUNNING, elapsed=_hms(8, 26, 24), gpus=2,
                  gpu_type="nvidia_h200", util=91, mem_used_gib=131, mem_total_gib=282, node="n-38-12",
                  limit=_hms(d=2), partition="h200"),
        slurm_job("vega", "724246", "orb-swp--train--voc-cosinewarm--s48--a01", JobState.RUNNING,
                  elapsed=_hms(6, 4, 44), gpu_type="nvidia_h200", util=71, mem_used_gib=40, mem_total_gib=141,
                  node="n-38-06", limit=_hms(d=2), partition="h200"),
        slurm_job("vega", "724245", "orb-swp--train--voc-cosinewarm--s47--a01", JobState.RUNNING,
                  elapsed=_hms(6, 5, 44), gpu_type="nvidia_h200", util=80, mem_used_gib=41, mem_total_gib=141,
                  node="n-37-12", limit=_hms(d=2), partition="h200"),
        slurm_job("vega", "724479", "lmft-x07-trainer-s0", JobState.PENDING, gpus=2, gpu_type="nvidia_h200",
                  limit=_hms(d=2), reason="QOSMaxJobsPerUserLimit", partition="h200", submit_ago=1200),
        *_completed("vega", 723435, 3, "orb-swp--train--voc-warmup"),
    ]
    vega = HostSnapshot("vega", T0, _caps(ME, "login-v01", ("slurm", "procfs"), slurm_version="25.11.4",
                                          cluster_name="vega", gpu_tres=True,
                                          slurm_config={"PreemptMode": "REQUEUE", "MinJobAge": "300"}),
                        {j.key.id: j for j in vega_jobs}, (), (), (),
                        (PartitionSummary("h200", 5, 40, 2, 2, {"nvidia_h200": (2, 40)}, {"mixed": 5}),
                         PartitionSummary("h100", 2, 9, 0, 0, {"nvidia_h100": (0, 9)}, {"allocated": 2}),
                         PartitionSummary("a30", 6, 24, 3, 0, {"nvidia_a30": (3, 24)}, {"mixed": 6})),
                        HostMetrics(64, (5.1, 4.8, 4.6), 9.0, 512 * GiB, 300 * GiB, 0, 0, 9e6, T0))

    # lyra (retrying, stale) --------------------------------------------------------------------------------------
    lyra_jobs = [
        slurm_job("lyra", "31354789", "orb-swp--train--places-cosinewarm--s47--a01", JobState.PREEMPTED,
                  elapsed=_hms(0, 1, 4), ended_ago=18 * 3600, gpu_type="rtx6000", node="r71-00",
                  partition="free-gpu", raw_state="PREEMPTED", user="alice07"),
        slurm_job("lyra", "31327454", "orb-swp--train--warmup--s49--a01", JobState.FAILED, elapsed=_hms(0, 2, 30),
                  ended_ago=20 * 3600, exit_code=2, gpu_type="rtx6000", node="r71-00", partition="free-gpu",
                  user="alice07"),
        *_completed("lyra", 31308336, 3, "orb-swp--train--places-warmup"),
    ]
    lyra_jobs = [dataclasses.replace(j, user="alice07", is_mine=True) for j in lyra_jobs]
    lyra = HostSnapshot("lyra", _ago(240), _caps("alice07", "login-a01", ("slurm", "procfs"),
                                                 slurm_version="26.05.3", cluster_name="lyra", gpu_tres=False,
                                                 slurm_config={"PreemptMode": "CANCEL"}),
                        {j.key.id: j for j in lyra_jobs}, (), (), (),
                        (PartitionSummary("free-gpu", 30, 120, 9, 0, {"rtx6000": (9, 120)}, {"mixed": 30}),
                         PartitionSummary("free-gpuxl", 9, 36, 4, 0, {"a30": (4, 36)}, {"mixed": 9})),
                        None)

    # labstation --------------------------------------------------------------------------------------------------
    qs_task = task_job("labstation", 363942, "orb.launch (orb.train)", elapsed=_hms(18, 54, 4), gpus=(3,),
                       gpu_model="NVIDIA RTX A6000", util=70, mem_mib=21.0 * 1024, mem_total_mib=49140,
                       pids=(363942, 363948))
    other = task_job("labstation", 771002, "python (eval.py)", elapsed=_hms(0, 42, 0), gpus=(1,),
                     gpu_model="NVIDIA RTX A6000", util=0, mem_mib=2.1 * 1024, mem_total_mib=49140, pids=(771002,),
                     user="alice", mine=False)
    qs_gpus = (
        _gpu("labstation", 0, "NVIDIA RTX A6000", 0, 0.0, 49140, 30, 24, 300),
        _gpu("labstation", 1, "NVIDIA RTX A6000", 0, 2.1, 49140, 31, 28, 300,
             (GPUProcess(771002, "GPU-labstation-1", 1, 2.1 * 1024, "alice", "python eval.py --ckpt best.pt",
                         2520, 0.3, other.key),)),
        _gpu("labstation", 2, "NVIDIA RTX A6000", 0, 0.0, 49140, 29, 20, 300),
        _gpu("labstation", 3, "NVIDIA RTX A6000", 70, 21.0, 49140, 78, 287, 300,
             (GPUProcess(363948, "GPU-labstation-3", 3, 21.0 * 1024, "alice07",
                         "/home/alice07/orb/.venv/bin/python -m orb.train --config cfg/cifar.yaml", 68046, 100.0,
                         qs_task.key),)),
    )
    qs_procs = (
        Proc(363948, 363942, 363942, "alice07", 68046, 100.0, 0.8, 4 * GiB, "Rl",
             "/home/alice07/orb/.venv/bin/python -m orb.train --config cfg/cifar.yaml", qs_task.key),
        Proc(363942, 1, 363942, "alice07", 68047, 0.0, 0.0, 200 << 20, "Sl",
             "python -m orb.launch --manifest manifests/cifar.json --per-gpu 1", qs_task.key),
        Proc(771002, 770990, None, "alice", 2520, 0.3, 0.4, 3 * GiB, "Sl", "python eval.py --ckpt best.pt",
             other.key),
    )
    qs = HostSnapshot("labstation", T0, _caps("alice07", "labstation", ("process", "nvidia", "procfs"),
                                              tools=("nvidia-smi", "ps", "less", "timeout"),
                                              driver_version="610.43.02"),
                      {qs_task.key.id: qs_task, other.key.id: other}, qs_gpus, qs_procs, (), (),
                      HostMetrics(64, (2.10, 1.98, 2.04), 3.0, 503 * GiB, 458 * GiB, 8 * GiB, 8 * GiB,
                                  117 * 86400, T0))

    # lynx (needs auth, last data 14 min old) ---------------------------------------------------------------------
    uc_task = task_job("lynx", 642229, "orb.launch ×2 (orb.train)", elapsed=_hms(18, 45, 12), gpus=(0, 1),
                       gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition", util=98, mem_mib=29.8 * 1024,
                       mem_total_mib=2 * 97887, pids=(642229, 642234, 786345))
    uc_gpus = (
        _gpu("lynx", 0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", 97, 14.9, 97887, 55, 196, 600,
             (GPUProcess(786345, "GPU-lynx-0", 0, 14.9 * 1024, "alice07", "python -m orb.train", 3000, 99.0,
                         uc_task.key),)),
        _gpu("lynx", 1, "NVIDIA RTX PRO 6000 Blackwell Server Edition", 100, 14.9, 97887, 57, 210, 600,
             (GPUProcess(642234, "GPU-lynx-1", 1, 14.9 * 1024, "alice07", "python -m orb.train", 67000, 99.0,
                         uc_task.key),)),
    )
    uc_procs = (
        Proc(642229, 1, 642229, "alice07", 67512, 0.0, 0.0, 180 << 20, "Sl",
             "python -m orb.launch --manifest manifests/places.json --per-gpu 1", uc_task.key),
        Proc(642234, 642229, 642229, "alice07", 67000, 99.0, 1.1, 6 * GiB, "Rl", "python -m orb.train --gpu 1",
             uc_task.key),
        Proc(786345, 642229, 642229, "alice07", 3000, 99.0, 1.0, 6 * GiB, "Rl", "python -m orb.train --gpu 0",
             uc_task.key),
    )
    uc = HostSnapshot("lynx", _ago(840), _caps("alice07", "lynx-gpu", ("process", "nvidia", "procfs"),
                                               tools=("nvidia-smi", "ps", "timeout"), driver_version="610.43.02"),
                      {uc_task.key.id: uc_task}, uc_gpus, uc_procs, (), (),
                      HostMetrics(96, (4.0, 4.0, 4.0), 5.0, 1007 * GiB, 900 * GiB, 0, 0, 40 * 86400, _ago(840)))

    states = [
        FakeHostState(hosts[0], _status("k2", ConnState.OK, ok_ago=8, latency=0.21), k2),
        FakeHostState(hosts[1], _status("vega", ConnState.OK, ok_ago=3, latency=0.20), vega),
        FakeHostState(hosts[2], _status("lyra", ConnState.RETRYING, ok_ago=240, retry_in=38, mux="managed",
                                        error=HostError(ErrorKind.JUMP,
                                                        "ssh: Connection timed out during banner exchange",
                                                        "Connection timed out during banner exchange\n"
                                                        "Connection to UNKNOWN port 65535 timed out",
                                                        None, T0)), lyra),
        FakeHostState(hosts[3], _status("labstation", ConnState.OK, ok_ago=2, latency=0.08, tick=5), qs),
        FakeHostState(hosts[4], _status("lynx", ConnState.NEEDS_AUTH, ok_ago=840, tick=5,
                                        error=HostError(ErrorKind.AUTH,
                                                        "Permission denied (publickey,keyboard-interactive)",
                                                        "alice07@lynx: Permission denied "
                                                        "(publickey,keyboard-interactive).", None, T0)), uc),
    ]
    return config, states


# ---------------------------------------------------------------- attach (§7.16) sample data
# vega 723954 on n-38-01: GPU0 busy, GPU1 0 % holding 85 GB (the Attach field finding); k2 2332861/2332862 on
# n-40-01: ssh lands in the newest job 2332863 (GPUs of another job); lyra 31367626 (add_lyra_running_job): srun only.
LYRA_LIVE_KEY = JobKey("lyra", "31367626")
H200_MIB = 143771.0
RTXPRO_MIB = 97887.0
A100_MIB = 40960.0


def _live_gpu(host: str, index: int, uuid: str, name: str, bus: str, idx: int | None, util: float, used: float,
              total: float, temp: float, power: float, limit: float, procs: tuple[GPUProcess, ...] = ()) -> GPU:
    return GPU(host, index, uuid, name, util, used, total, used / total * 100, temp, power, limit, None,
               "P0" if util else "P2", "nvidia-smi", procs, T0, bus_id=bus, slurm_idx=idx)


def _live_proc(pid: int, ppid: int, etime: int, cpu: float, rss_gib: float, stat: str, args: str,
               user: str = ME) -> Proc:
    return Proc(pid, ppid, ppid, user, etime, cpu, rss_gib / 10, int(rss_gib * GiB), stat, args)


def attach_sample(key: JobKey, n: int, now: datetime) -> JobLive:
    """The n-th (1-based) deterministic Attach sample of `key` (ssh for vega/k2, srun for lyra)."""
    if key == JobKey("vega", "723954"):
        job = key
        ps: tuple[Proc, ...] = (
            _live_proc(129361, 129300, 21000, 362.0, 41.0, "Sl", "python3 -m lmft.train --cfg sampler_s1.yaml"),
            _live_proc(129392, 129361, 20990, 99.0, 6.5, "Rl", "VLLM::EngineCore"),
            _live_proc(129300, 129290, 21010, 0.0, 0.1, "S", "/bin/bash /var/spool/slurmd/job723954/slurm_script"))
        g0p = (GPUProcess(129392, "GPU-vega-a0", 0, 55788.0, ME, "VLLM::EngineCore", 20990, 99.0, job),
               GPUProcess(129361, "GPU-vega-a0", 0, 35738.0, ME, "python3 -m lmft.train", 21000, 362.0, job))
        g1p = (GPUProcess(129361, "GPU-vega-b1", 1, 84634.0, ME, "python3 -m lmft.train", 21000, 362.0, job),)
        gpus: tuple[GPU, ...] = (
            _live_gpu("vega", 0, "GPU-vega-a0", "NVIDIA H200", "00000000:89:00.0", 0, 100, 91625, H200_MIB, 73, 576,
                      600, g0p),
            _live_gpu("vega", 1, "GPU-vega-b1", "NVIDIA H200", "00000000:C5:00.0", 1, 0, 84725, H200_MIB, 34, 94,
                      600, g1p))
        cg = CgroupStats(mem_current=143_700_000_000, mem_peak=202_200_000_000, mem_max=256 * GiB,
                         anon=80_100_000_000, file=3_200_000_000, shmem=60_400_000_000, oom=0, oom_kill=0,
                         cpu_usage_usec=75_600_000_000 + n * 54_000_000, cpu_cores=3.6 if n > 1 else None)
        node = NodeInspect("n-38-01", "723954", "723954", True, gpus, ps,
                           {129361: "723954.batch", 129392: "723954.batch", 129300: "723954.batch"},
                           {"723954": cg}, 0, 610)
        streak = {"GPU-vega-b1": n}
        hints = ("GPU1 idle (0%) but holding 84.7G",) if n >= 2 else ()
        return JobLive(key, "ssh", now, (node,), 670, 0, True, "clean", hints, streak)
    if key.host == "k2" and key.id in ("2332861", "2332862", "2332863"):
        other = JobKey("k2", "2332863")
        ps = (_live_proc(88410, 88400, 14832, 99.0, 10.4, "Rl", "python -m orb.train --sched warmup"),
              _live_proc(88420, 88401, 14832, 98.0, 10.4, "Rl", "python -m orb.train --sched const"),
              _live_proc(88430, 88402, 14832, 98.0, 10.4, "Rl", "python -m orb.train --sched cosinewarm"))
        gp = (GPUProcess(88430, "GPU-k2-3", 0, 24870.0, ME, "python -m orb.train", 14832, 98.0, other),)
        gpus = (_live_gpu("k2", 0, "GPU-k2-3", "NVIDIA A100-SXM4-40GB", "00000000:41:00.0", 3, 100, 24881,
                          A100_MIB, 53, 250.31, 400, gp),)
        cgs = {key.id: CgroupStats(10_650_000_000, 11_780_000_000, 48 * GiB, 9_900_000_000, 700_000_000,
                                   50_000_000, 0, 0, 60_600_000_000, 1.0 if n > 1 else None),
               "2332863": CgroupStats(10_660_000_000, 13_720_000_000, 48 * GiB, 9_900_000_000, 700_000_000,
                                      60_000_000, 0, 0, 60_620_000_000, 1.0 if n > 1 else None)}
        steps = {88410: "2332861.batch", 88420: "2332862.batch", 88430: "2332863.batch"}
        node = NodeInspect("n-40-01", key.id, "2332863", key.id == "2332863", gpus, ps, steps, cgs, 0, 631)
        return JobLive(key, "ssh", now, (node,), 820, 0, True, "clean", (), {})
    if key != LYRA_LIVE_KEY:             # any other job: one busy GPU on "node-01"
        gp = (GPUProcess(4242, f"GPU-{key.id}", 0, 30000.0, ME, "python train.py", 3600, 99.0, key),)
        gpus = (_live_gpu(key.host, 0, f"GPU-{key.id}", "NVIDIA H200", "00000000:1A:00.0", 0, 88, 30000, H200_MIB,
                          60, 420, 600, gp),)
        ps = (_live_proc(4242, 4200, 3600, 99.0, 8.0, "Rl", "python train.py"),)
        node = NodeInspect("node-01", key.id, key.id, True, gpus, ps, {4242: f"{key.id}.batch"},
                           {key.id: CgroupStats(8 * GiB, 9 * GiB, 64 * GiB, 7 * GiB, GiB // 2, 0, 0, 0, 1000, None)},
                           0, 500)
        return JobLive(key, "ssh", now, (node,), 700, 0, True, "clean", (), {})
    # lyra (srun)
    ps = (_live_proc(3012, 3001, 5400, 101.0, 36.3, "Rl", "python -m orb.train --dataset places", user="alice07"),
          _live_proc(3001, 2990, 5402, 0.0, 0.1, "S", "/bin/bash slurm_script", user="alice07"))
    gp = (GPUProcess(3012, "GPU-lyra-71", 0, 48144.0, "alice07", "python -m orb.train", 5400, 101.0, key),)
    gpus = (_live_gpu("lyra", 0, "GPU-lyra-71", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "00000000:71:00.0",
                      0, 97, 48144, RTXPRO_MIB, 61, 412, 600, gp),)
    cg = CgroupStats(78_740_000_000, 79_100_000_000, 96 * GiB, 36_300_000_000, 1_800_000_000, 40_600_000_000, 0, 0,
                     5_450_000_000, None)
    node = NodeInspect("s71-02", "31367626", "31367626", True, gpus, ps,
                       {3012: "31367626.batch", 3001: "31367626.batch"}, {"31367626": cg}, 0, 180)
    return JobLive(key, "srun", now, (node,), 1210, 1, True, "clean", (), {})


def lyra_running_job() -> Job:
    job = slurm_job("lyra", LYRA_LIVE_KEY.id, "orb-swp--train--places-cosinewarm--s50--a01", JobState.RUNNING,
                    elapsed=_hms(1, 30), gpu_type="rtx6000", node="s71-02", partition="free-gpu", user="alice07",
                    cpus=16, mem_gib=96)
    return dataclasses.replace(job, is_mine=True)


LOG_LINES = (
    "[2026-09-07 01:50:39] host n-40-01  CUDA_VISIBLE_DEVICES=0  torch 2.9.1+cu130",
    "[2026-09-07 01:50:52] config: cifar  model: resnet18  sched=warmup  seed=43",
    "[2026-09-07 05:55:35] epoch 40/300  train_loss 0.3140  vs_base -2.38%  212 it/s",
    "[2026-09-07 05:56:18] epoch 41/300  train_loss 0.3121  vs_base -2.41%  213 it/s",
    "[2026-09-07 05:56:39] epoch 42/300  train_loss 0.3098  vs_base -2.44%  211 it/s",
    "[2026-09-07 05:57:01] epoch 43/300  train_loss 0.3087  vs_base -2.47%  212 it/s",
    "[2026-09-07 05:57:22] epoch 44/300  train_loss 0.3071  vs_base -2.50%  210 it/s",
)


# ---------------------------------------------------------------- logs
class FakeLogSession:
    """LogSession whose updates come from an initial tail plus whatever the test pushes."""

    def __init__(self, host: str, ref: LogRef, initial: Sequence[str]) -> None:
        self.host = host
        self.ref = ref
        self.queue: asyncio.Queue[LogUpdate | None] = asyncio.Queue()
        self.queue.put_nowait(LogUpdate(lines=tuple(initial)))
        self.closed = False

    async def updates(self) -> AsyncIterator[LogUpdate]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item
            if item.eof:
                return

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(None)


# ---------------------------------------------------------------- service
_SLURM_SPECS = (("slurm.cancel", "Cancel job"), ("slurm.hold", "Hold job"), ("slurm.release", "Release job"),
                ("slurm.requeue", "Requeue job"))
_TASK_SPECS = (("task.terminate", "Terminate task"), ("task.interrupt", "Interrupt task (SIGINT)"),
               ("task.kill", "Kill task (SIGKILL)"))
_TASK_PARAMS = (ActionParam("escalate", "bool", True, "send SIGKILL to survivors after"),
                ActionParam("grace_s", "int", 10, "seconds", 1, 600))


class FakeService:
    """In-memory FleetAPI for UI tests (structurally typed against ``runfleet.core.service.FleetAPI``)."""

    def __init__(self, clock: ManualClock | None = None, *, readonly: bool = False,
                 world: tuple[FleetConfig, list[HostState]] | None = None) -> None:
        config, states = world or build_world()
        if readonly:
            config = dataclasses.replace(config, general=dataclasses.replace(config.general, readonly=True))
        self.config = config
        self.readonly = config.general.readonly
        self.notices: list[str] = []
        self.store: FakeStore = FakeStore(states)
        self.clock = clock or ManualClock(T0)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.visibility: dict[str, Visibility] = {}
        self.sessions: list[FakeLogSession] = []
        self.results: list[ActionResult] = []
        self.multiplier = 1.0
        self.focused = True
        self.started = False
        self.last_activity = self.clock.monotonic()
        self.peeks: list[tuple[str, str | None]] = []
        self.logs: dict[str, list[str]] = {}
        self.stream_starts: list[int | None] = []          # stream_log(start=…) per stream, in order
        self.missing_logs: set[str] = set()                 # log.search answers "@@err I missing" for these
        self.search_errors: dict[str, str] = {}             # host -> search_logs error (no probe)
        self.search_gate: asyncio.Event | None = None       # set: search_logs waits for it (a slow host)
        self.already_connected: set[str] = set()
        self.logins: dict[str, Callable[[], int]] = {}      # connect_login answers (session transport hosts)
        # Settings (FleetAPI runtime host management): in memory like --demo unless a test sets config_path
        self.config_path: Path | None = None
        self.saved: list[HostConfig] = [HostConfig(name=h.name, ssh=h.ssh) for h in config.hosts]
        self.aliases: list[str] = ["k2", "vega", "lyra", "labstation", "lynx", "edge1", "edge2", "atlas"]
        self.config_error: str | None = None
        self.write_error: str | None = None         # set: every save raises ConfigError (an unwritable file)
        self.host_keys: dict[str, list[str]] = {}   # custom_host_keys: hand-tuned keys of saved tables
        self.probe_reports: dict[str, DetectReport] = {}
        self.port_forwards: dict[int, PortForward] = {}   # forward_port / cancel_forward (no ssh)
        self.forward_blocks: dict[str, str] = {}          # host -> forward_blocker() reason
        self.cancel_error: str | None = None              # set: cancel_forward raises it (the forward stays)
        # Attach (§7.16): per-host setting overrides (the fake world's config keeps the default "ssh+srun") and the
        # learned node_ssh capability; states/samples per job
        self.attach_setting: dict[str, Literal["ssh", "ssh+srun", "srun", "off"]] = {"lyra": "srun"}
        self.node_ssh: dict[str, str] = {"k2": "adopt", "vega": "adopt", "lyra": "denied"}
        self.attach_states: dict[JobKey, AttachState] = {}
        self.attach_samples: dict[JobKey, int] = {}
        self.attach_open: JobKey | None = None

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def called(self, name: str) -> list[tuple[Any, ...]]:
        return [a for n, a in self.calls if n == name]

    # lifecycle
    async def start(self) -> None:
        self.started = True
        self._record("start")

    async def stop(self) -> None:
        self._record("stop")

    async def collect_once(self, hosts: Iterable[str] | None = None, *, timeout: float = 60) -> None:
        self._record("collect_once", hosts)

    # scheduling hints
    def set_visibility(self, levels: Mapping[str, Visibility]) -> None:
        self.visibility = dict(levels)
        self._record("set_visibility", dict(levels))

    def note_user_activity(self) -> None:
        self.last_activity = self.clock.monotonic()
        if self.store.idle_paused:
            self.store.set_idle_paused(False)

    def set_app_focus(self, focused: bool) -> None:
        self.focused = focused

    def refresh(self, hosts: Iterable[str] | None = None) -> None:
        self._record("refresh", None if hosts is None else tuple(hosts))

    def set_rate_multiplier(self, m: float) -> None:
        self.multiplier = m
        self._record("set_rate_multiplier", m)

    def set_host_enabled(self, host: str, enabled: bool) -> None:
        self._record("set_host_enabled", host, enabled)
        st = self.store.hosts[host].status
        self.store.set_status(host, dataclasses.replace(st, state=ConnState.OK if enabled else ConnState.DISABLED))

    # details and logs
    async def job_detail(self, key: JobKey, *, max_age_s: float = 30) -> JobDetail:
        self._record("job_detail", key)
        job = self.store.job(key)
        if job is None:
            raise KeyError(str(key))
        steps: tuple[JobStep, ...] = ()
        explanation: tuple[str, ...] = ()
        procs: tuple[Proc, ...] = ()
        procs_note = ""
        if job.kind is JobKind.SLURM:
            procs_note = "per-process view needs compute-node access (v0.2)"
            steps = (
                JobStep(job.job_id_raw, job.name, job.state, job.raw_state, job.exit_code, 0, job.elapsed,
                        job.start_time, job.end_time, job.nodelist, None, None, None),
                JobStep(f"{job.job_id_raw}.batch", "batch", job.state, job.raw_state, job.exit_code, 0, job.elapsed,
                        job.start_time, job.end_time, job.nodelist, 1, int(9.9 * GiB), timedelta(hours=4, minutes=20),
                        {"gres/gpuutil": "99", "gres/gpumem": "24.3G"}),
                JobStep(f"{job.job_id_raw}.extern", "extern", JobState.COMPLETED if job.state.is_terminal
                        else job.state, "COMPLETED", 0, 0, job.elapsed, job.start_time, job.end_time, job.nodelist,
                        None, None, None),
            )
            if job.state is JobState.FAILED:
                explanation = ("batch step exited with code 1 (the job script / program failed; not a Slurm kill)",
                               'last log lines: "torch.OutOfMemoryError: CUDA out of memory."')
        else:
            snap = self.store.hosts[key.host].snapshot
            procs = tuple(p for p in (snap.procs if snap else ()) if p.task == key)
        fields = dict(job.raw) | {"WorkDir": job.workdir or "", "Command": job.command or ""}
        return JobDetail(job, fields, job.alloc.per_node, steps, job.logs, job.gpu, procs, procs_note,
                         {"CUDA_VISIBLE_DEVICES": "3"} if job.kind is JobKind.TASK else {}, explanation,
                         self.clock.now(), efficiency=efficiency_lines(efficiency(job, steps)))   # like the collector

    def log_lines(self, path: str | None) -> list[str]:
        return self.logs.setdefault(path or "", list(LOG_LINES))

    async def peek_log(self, host: str, ref: LogRef, *, nbytes: int = 4096) -> LogPeek:
        self.peeks.append((host, ref.path))
        return LogPeek(tuple(self.log_lines(ref.path)[-8:]), ref.size, ref.mtime, self.clock.now())

    def log_bytes(self, path: str | None) -> bytes:
        """The fake file behind ``log_lines``: the lines, each ended by a newline."""
        return "".join(line + "\n" for line in self.log_lines(path)).encode("utf-8")

    async def stream_log(self, host: str, ref: LogRef, *, tail_bytes: int = 65536,
                         start: int | None = None) -> FakeLogSession:
        lines: Sequence[str] = self.log_lines(ref.path)
        if start is not None:                      # `tail -c +K`: from that byte on (the first line may be partial)
            lines = self.log_bytes(ref.path)[start:].decode("utf-8", errors="replace").split("\n")[:-1]
        session = FakeLogSession(host, ref, lines)
        self.sessions.append(session)
        self.stream_starts.append(start)
        self._record("stream_log", host, ref.stream, ref.path)
        return session

    async def search_logs(self, host: str, query: LogQuery, *, max_matches: int = 200) -> LogSearchResult:
        """The real plan and parser over the fake files, through the demo's play of the `log.search` script."""
        from types import SimpleNamespace

        from runfleet.demo import render
        from runfleet.demo.world import SimLog

        self._record("search_logs", host, query, max_matches)
        if self.search_gate is not None:
            await self.search_gate.wait()
        search = logsearch.plan(host, self.store.jobs([host]), self.clock.now(), self.config.general.recent_window)
        if host in self.search_errors:
            return LogSearchResult(host, query, search, max_matches=max_matches, error=self.search_errors[host])
        problem = logsearch.pattern_problem(query)
        if problem is not None or not search.files:
            return LogSearchResult(host, query, search, max_matches=max_matches, error=problem)
        section = logsearch.search_section([f.path for f in search.files], query, max_matches=max_matches)
        files: dict[str, SimLog] = {}
        for f in search.files:
            if f.path not in self.missing_logs:
                files[f.path] = SimLog(f.path, 0.0)
                files[f.path].write(self.log_bytes(f.path).decode("utf-8"), 0.0)
        m = section.meta
        out = render.log_search(SimpleNamespace(logs=files), paths=list(m["paths"]), pattern=str(m["pattern"]),
                                icase=bool(m["icase"]), max_matches=int(m["max_matches"]),
                                per_file=int(m["per_file"]), budget=int(m["budget"]))
        res = SectionResult(section.name, SectionOutcome.OK if out.stdout else SectionOutcome.EMPTY, out.rc,
                            out.stdout, out.stderr, 7)
        return logsearch.parse(host, query, search, res, max_matches=max_matches)

    async def fetch_log(self, host: str, ref: LogRef, *, nbytes: int = 4 << 20) -> Path:
        fd, name = tempfile.mkstemp(prefix="rf-log-")
        with open(fd, "w") as fh:
            fh.write("\n".join(self.log_lines(ref.path)))
        return Path(name)

    # actions
    def _host_reason(self, host: str) -> str | None:
        state = self.store.hosts[host]
        if state.status.state not in (ConnState.OK, ConnState.DEGRADED):
            label = {ConnState.NEEDS_AUTH: "needs auth", ConnState.HOSTKEY: "host key"}.get(
                state.status.state, state.status.state.value)
            return f"{host}: {label}"
        if state.stale(self.clock.now()):
            return "data is stale — refresh first"
        return None

    def _eligible(self, action_id: str, job: Job) -> str | None:
        """None if eligible, else the reason."""
        if not job.is_mine:
            return f"owned by {job.user}"
        if action_id.startswith("task."):
            if job.kind is not JobKind.TASK:
                return "not a task"
            return None if job.state.group is StateGroup.ACTIVE else "task is not running"
        if job.kind is not JobKind.SLURM:
            return "not a Slurm job"
        host_cfg = self.store.hosts[job.key.host].config
        if job.partition in host_cfg.slurm.protected_partitions or job.qos in host_cfg.slurm.protected_qos:
            return f"protected partition {job.partition}"
        group = job.state.group
        if action_id == "slurm.cancel":
            return None if group in (StateGroup.QUEUED, StateGroup.ACTIVE) else "job already ended"
        if action_id == "slurm.hold":
            return None if group is StateGroup.QUEUED and not job.held else "only pending, not held jobs"
        if action_id == "slurm.release":
            return None if job.held else "job is not held"
        return None if group is StateGroup.ACTIVE else "no longer in controller memory — resubmit"

    def _jobs(self, keys: Sequence[JobKey]) -> list[Job]:
        return [j for j in (self.store.job(k) for k in keys) if j is not None]

    def actions_for(self, keys: Sequence[JobKey]) -> list[ActionSpec]:
        jobs = self._jobs(keys)
        kinds = {j.kind for j in jobs}
        catalogue = (_SLURM_SPECS if JobKind.SLURM in kinds else ()) + (_TASK_SPECS if JobKind.TASK in kinds else ())
        specs: list[ActionSpec] = []
        for action_id, label in catalogue:
            reason: str | None = None
            if self.readonly:
                reason = "read-only mode"
            else:
                host_reasons = [r for r in (self._host_reason(h) for h in {j.key.host for j in jobs}) if r]
                job_reasons = [self._eligible(action_id, j) for j in jobs]
                if host_reasons and len(host_reasons) == len({j.key.host for j in jobs}):
                    reason = host_reasons[0]
                elif all(r is not None for r in job_reasons):
                    reason = next(r for r in job_reasons if r is not None) if job_reasons else "nothing selected"
            params = _TASK_PARAMS if action_id == "task.terminate" else ()
            specs.append(ActionSpec(action_id, label, True, True, params, reason))
        return specs

    def plan(self, action_id: str, keys: Sequence[JobKey],
             params: Mapping[str, object] | None = None) -> list[ActionPlan]:
        self._record("plan", action_id, tuple(keys), dict(params or {}))
        spec = next((s for s in self.actions_for(keys) if s.id == action_id), None)
        if spec is None or self.readonly:
            raise ActionError("read-only mode" if self.readonly else f"{action_id} not applicable")
        params = dict(params or {})
        by_host: dict[str, list[Job]] = {}
        skipped: dict[str, list[tuple[JobKey, str]]] = {}
        for job in self._jobs(keys):
            reason = self._host_reason(job.key.host) or self._eligible(action_id, job)
            if reason is None:
                by_host.setdefault(job.key.host, []).append(job)
            else:
                skipped.setdefault(job.key.host, []).append((job.key, reason))
        if not by_host:
            raise ActionError("no eligible target")
        plans = []
        for host, jobs in by_host.items():
            ids = " ".join(j.key.id for j in jobs)
            if action_id.startswith("slurm."):
                verb = action_id.split(".")[1]
                cmd = {"cancel": "scancel", "hold": "scontrol hold", "release": "scontrol release",
                       "requeue": "scontrol requeue"}[verb]
                preview = f"{cmd} {ids}"
                n = len(jobs)
                title = f"{spec.label.split()[0]} {n} job{'s' if n != 1 else ''} on {host}?"
                body = "\n".join(f"{j.key.id}  {j.name}  {j.state.value}" for j in jobs)
            else:
                escalate = bool(params.get("escalate", True))
                grace = int(params.get("grace_s", 10))   # type: ignore[call-overload]
                sig = {"task.terminate": "TERM", "task.interrupt": "INT", "task.kill": "KILL"}[action_id]
                roots = " ".join(str(j.job_id_raw) for j in jobs)
                preview = f"STOP {roots} · {sig} its tree"
                if action_id == "task.terminate":
                    preview += f" · KILL after {grace} s" if escalate else " · no escalation"
                label = " ".join(j.key.label for j in jobs)
                title = f"{spec.label.split()[0]} task {label} on {host}?"
                body = "\n".join(f"{j.name} · {len(j.pids)} processes · GPUs "
                                 f"{','.join(str(i) for n in j.alloc.per_node for i in n.gpu_indices)}"
                                 for j in jobs) + f"\nmembers are re-read on {host} at confirm time; launcher paused first"
            plans.append(ActionPlan(spec, host, tuple(j.key for j in jobs), tuple(skipped.get(host, ())), params, (),
                                    preview, title, body))
        return plans

    async def execute(self, plans: Sequence[ActionPlan]) -> list[ActionResult]:
        self._record("execute", tuple(plans))
        out: list[ActionResult] = []
        now = self.clock.now()
        for plan in plans:
            for key in plan.targets:
                job = self.store.job(key)
                if job is None:
                    continue
                aid = plan.spec.id
                if aid == "slurm.cancel":
                    job = dataclasses.replace(job, state=JobState.CANCELLED, raw_state="CANCELLED by 610001",
                                              end_time=now)
                elif aid == "slurm.hold":
                    job = dataclasses.replace(job, held=True, reason="JobHeldUser")
                elif aid == "slurm.release":
                    job = dataclasses.replace(job, held=False, reason="Priority")
                elif aid == "slurm.requeue":
                    job = dataclasses.replace(job, state=JobState.PENDING, restarts=job.restarts + 1)
                else:
                    job = dataclasses.replace(job, state=JobState.EXITED, raw_state="exited", end_time=now)
                self.store.put_job(job)
                res = ActionResult(aid, key, True, f"{aid.split('.')[1]} {key.label}", rc=0, started_at=now,
                                   duration_s=0.2)
                out.append(res)
        self.results.extend(out)
        return out

    def action_log(self) -> Sequence[ActionResult]:
        return list(self.results)

    # interactive argv
    async def prepare_connect(self, host: str) -> bool:
        self._record("prepare_connect", host)
        return host in self.already_connected

    def connect_argv(self, host: str) -> list[str] | None:
        return ["ssh", "-fN", "-o", "ControlMaster=yes", host]

    def connect_login(self, host: str) -> Callable[[], int] | None:
        return self.logins.get(host)

    async def after_connect(self, host: str, rc: int) -> None:
        self._record("after_connect", host, rc)
        if rc == 0:
            self.set_ok(host)

    def shell_argv(self, host: str, key: JobKey | None = None, node: str | None = None) -> list[str]:
        return ["ssh", "-t", host] + ([f"cd /work && exec $SHELL -l  # {key}" + (f" on {node}" if node else "")]
                                      if key else [])

    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None:
        snap = self.store.hosts[host].snapshot
        if snap is None or snap.caps is None or "less" not in snap.caps.tools or ref.path is None:
            return None
        return ["ssh", "-t", host, "less", "-R", "+G", "--", ref.path]

    def local_pager_argv(self, path: Path | None = None) -> list[str]:
        return ["less", "-R", "+G", str(path or "/tmp/runfleet.log")]

    # import and diagnostics
    async def probe_alias(self, alias: str) -> DetectReport:
        self._record("probe_alias", alias)
        return self.probe_reports.get(alias, DetectReport(alias, ConnState.OK, None, None, "user", {}, 0.1))

    def write_config(self, hosts: Sequence[HostConfig], path: Path | None = None) -> Path:
        return path or Path("/tmp/config.toml")

    # runtime hosts and settings
    def ssh_aliases(self) -> list[str]:
        return list(self.aliases)

    def _check_config(self, write: bool = True) -> None:
        from runfleet.errors import ConfigError
        if self.config_error is not None:
            raise ConfigError("", self.config_error)
        if write and self.write_error is not None:
            raise ConfigError("", self.write_error)

    def saved_hosts(self) -> list[HostConfig]:
        self._check_config(write=False)
        return list(self.saved)

    def custom_host_keys(self, host: str) -> list[str]:
        return list(self.host_keys.get(host, []))

    def add_host(self, cfg: HostConfig, *, save: bool = True) -> HostConfig:
        self._record("add_host", cfg.name, cfg.ssh, save)
        if save:
            self._check_config()
            if not any(h.name == cfg.name for h in self.saved):
                self.saved.append(HostConfig(name=cfg.name, ssh=cfg.ssh))
        if cfg.name not in self.store.hosts:
            self.store.hosts[cfg.name] = FakeHostState(cfg, HostStatus(cfg.name, ConnState.CONNECTING), None)
            self.store.emit(ChangeSet(cfg.name, status=True))
        return self.store.hosts[cfg.name].config

    async def remove_host(self, host: str, *, save: bool = True) -> None:
        self._record("remove_host", host, save)
        if save:
            self._check_config()
            self.saved = [h for h in self.saved if h.name != host]
        state = self.store.hosts.pop(host, None)
        if state is not None:
            jobs = frozenset(state.snapshot.jobs) if state.snapshot is not None else frozenset()
            self.store.emit(ChangeSet(host, status=True, jobs_removed=jobs, gpus=True))

    def reorder_hosts(self, names: Sequence[str], *, save: bool = True) -> None:
        self._record("reorder_hosts", tuple(names), save)
        rank = {n: i for i, n in enumerate(names)}
        if save:
            self._check_config()
            self.saved.sort(key=lambda h: rank.get(h.name, len(rank)))
        items = sorted(self.store.hosts.items(), key=lambda kv: rank.get(kv[0], len(rank)))
        self.store.hosts.clear()
        self.store.hosts.update(items)
        self.store.emit(ChangeSet("*", status=True))

    def set_general(self, *, save: bool = True, **changes: Any) -> GeneralSettings:
        self._record("set_general", dict(changes), save)
        if save:
            self._check_config()
        general = dataclasses.replace(self.config.general, **changes)
        self.config = dataclasses.replace(self.config, general=general)
        if "readonly" in changes:
            self.readonly = general.readonly
        return general

    def set_notify(self, *, save: bool = True, **changes: Any) -> NotifySettings:
        self._record("set_notify", dict(changes), save)
        if save:
            self._check_config()
        notify = dataclasses.replace(self.config.notify, **changes)
        self.config = dataclasses.replace(self.config, notify=notify)
        return notify

    def remember_hosts(self, names: Sequence[str]) -> None:
        self._record("remember_hosts", tuple(names))

    async def doctor(self, host: str, *, reset: bool = False) -> DoctorReport:
        self._record("doctor", host)
        return DoctorReport(host, (DoctorCheck("ssh -G", True, f"user alice07 · hostname {host}.example.edu"),
                                   DoctorCheck("mux", True, "user ControlPath ~/.ssh/cm-%C"),
                                   DoctorCheck("probe", True, "0.21 s, 5/5 sections ok"),
                                   DoctorCheck("gpu_tres", None, "cluster records gres/gpuutil")),
                            ("no GPU accounting — job GPU util unavailable",) if host == "lyra" else ())

    # everyday actions
    def job_shell_command(self, host: str, key: JobKey | None = None, node: str | None = None) -> str | None:
        job = self.store.job(key) if key is not None else None
        if job is None:
            return None
        if job.kind is JobKind.SLURM and job.state is JobState.RUNNING and job.nodes:
            return f"srun --jobid={job.job_id_raw} --overlap --pty -w {node or job.nodes[0]} bash -l"
        return f"cd {job.workdir or '~'} && exec $SHELL -l"

    def forward_blocker(self, host: str) -> str | None:
        if host in self.forward_blocks:
            return self.forward_blocks[host]
        st = self.store.hosts[host].status.state
        return None if st in (ConnState.OK, ConnState.DEGRADED) else f"{host} is not connected ({st.value})"

    async def forward_port(self, host: str, node: str, remote_port: int, *, job: JobKey | None = None,
                           scheme: str = "http", path: str = "", label: str = "") -> PortForward:
        from runfleet.core.forwards import local_url
        self._record("forward_port", host, node, remote_port)
        for fwd in self.port_forwards.values():
            if (fwd.host, fwd.node, fwd.remote_port) == (host, node, remote_port):
                return fwd
        local = remote_port if remote_port not in self.port_forwards else 40000 + len(self.port_forwards)
        fwd = PortForward(host, node, remote_port, local, local_url(local, scheme=scheme, path=path), job, label,
                          self.clock.now())
        self.port_forwards[local] = fwd
        return fwd

    async def cancel_forward(self, local_port: int) -> None:
        self._record("cancel_forward", local_port)
        if self.cancel_error is not None:
            raise RunFleetError(self.cancel_error)
        self.port_forwards.pop(local_port, None)

    def forwards(self) -> list[PortForward]:
        return list(self.port_forwards.values())

    # ------------------------------------------------------------ attach (§7.16), deterministic
    def _attach_setting(self, host: str) -> Literal["ssh", "ssh+srun", "srun", "off"]:
        hs = self.store.hosts.get(host)
        return self.attach_setting.get(host, hs.config.slurm.attach if hs is not None else "off")

    def _attach_notify(self, key: JobKey) -> None:
        self.store.emit(ChangeSet(key.host, live=frozenset({key.id})))

    def _attach_put(self, state: AttachState) -> AttachState:
        self.attach_states[state.key] = state
        self._attach_notify(state.key)
        return state

    def _attach_offer(self, job: Job) -> AttachOffer:
        node = job.batch_host or (job.nodes[0] if job.nodes else "?")
        preview = (f"srun --jobid={job.job_id_raw} --overlap --job-name=rfprobe -N1 -n1 -w {node} --time=1 "
                   f"--quiet --input=none timeout -k 1 15 sh -c '…'")
        return AttachOffer(node, f"{job.key.host} refuses ssh to compute nodes",
                           f"Read job {job.key.id} with srun?",
                           f"{job.key.host} does not allow ssh to compute nodes. Reading job {job.key.id}'s GPUs runs "
                           f"a one-shot step in the job (srun --overlap, about 0.1-0.2 s, at most 1 minute). It "
                           f"leaves one permanent sacct row {job.job_id_raw}.N rfprobe that cannot be deleted.",
                           preview)

    def _attach_base(self, key: JobKey) -> AttachState:
        """Policy of §7.16 for `key` without sampling (status BLOCKED / NEEDS_CONSENT / IDLE)."""
        old = self.attach_states.get(key)
        setting = self._attach_setting(key.host)
        keep: dict[str, Any] = {}
        if old is not None:
            keep = {"live": old.live, "consent": old.consent, "srun_rows": old.srun_rows, "auto": old.auto}
        job = self.store.job(key)
        blocked: str | None = None
        hint: str | None = None
        if job is None or job.kind is not JobKind.SLURM:
            blocked = "live readings are for running Slurm jobs"
        elif job.state is not JobState.RUNNING:
            blocked = "job is not running"
        elif setting == "off":
            blocked = f"attach is off for {key.host}"
        if blocked is not None:
            return AttachState(key, AttachStatus.BLOCKED, setting, blocked=blocked, **keep)
        assert job is not None
        ssh_ok = setting in ("ssh", "ssh+srun") and self.node_ssh.get(key.host) != "denied"
        srun_ok = setting in ("ssh+srun", "srun") and not self.readonly
        if ssh_ok:
            hs = self.store.hosts[key.host]
            return AttachState(key, AttachStatus.IDLE, setting, "ssh", refresh_s=hs.config.slurm.attach_refresh,
                               **keep)
        if srun_ok:
            consent = keep.get("consent", "none")
            status = AttachStatus.IDLE if consent == "job" else AttachStatus.NEEDS_CONSENT
            return AttachState(key, status, setting, "srun", offer=self._attach_offer(job), **keep)
        if setting in ("ssh+srun", "srun") and self.readonly:
            blocked = "read-only mode: srun adds a permanent sacct row"
        else:
            blocked = f"{key.host} refuses ssh to compute nodes"
            hint = (f'use attach = "ssh+srun" (the default) for {key.host} to read via srun; '
                    "each read adds one permanent sacct row")
        return AttachState(key, AttachStatus.BLOCKED, setting, blocked=blocked, hint=hint, **keep)

    def attach_start(self, key: JobKey, *, node: str | None = None) -> AttachState:
        self._record("attach_start", key, node)
        if self.attach_open is not None and self.attach_open != key:
            self.attach_stop(self.attach_open, reason="another job attached")
        self.attach_open = key
        state = dataclasses.replace(self._attach_base(key), node=node)
        if state.mode == "ssh":            # the real core samples at once in the background; the fake does it now
            live = self._attach_take(key)
            return self._attach_put(dataclasses.replace(
                state, status=AttachStatus.LIVE if state.auto else AttachStatus.IDLE, live=live,
                next_at=self.clock.now() + timedelta(seconds=15) if state.auto else None))
        return self._attach_put(state)

    def attach_stop(self, key: JobKey | None = None, *, reason: str = "tab closed") -> None:
        self._record("attach_stop", key)
        key = key or self.attach_open
        if key is None:
            return
        if key == self.attach_open:
            self.attach_open = None
        old = self.attach_states.get(key)
        if old is not None and old.status is not AttachStatus.BLOCKED:
            self._attach_put(dataclasses.replace(old, status=AttachStatus.STOPPED, stopped=reason, next_at=None))

    def _attach_take(self, key: JobKey) -> JobLive:
        n = self.attach_samples[key] = self.attach_samples.get(key, 0) + 1
        return attach_sample(key, n, self.clock.now())

    async def attach_now(self, key: JobKey, *, mode: AttachMode | None = None,
                         consent: Literal["once", "job"] | None = None, node: str | None = None) -> JobLive:
        self._record("attach_now", key, mode, consent, node)
        base = self._attach_base(key)
        if base.status is AttachStatus.BLOCKED:
            raise AttachBlocked(base.blocked or "unavailable", hint=base.hint)
        use = mode or base.mode
        if use == "srun":
            job = self.store.job(key)
            assert job is not None
            if self.readonly or base.setting not in ("ssh+srun", "srun"):
                raise AttachBlocked("read-only mode: srun adds a permanent sacct row" if self.readonly
                                    else f"srun is not enabled for {key.host}")
            if consent is None and base.consent != "job":
                raise AttachConsentRequired(key, base.offer or self._attach_offer(job))
            live = dataclasses.replace(self._attach_take(key), mode="srun", sacct_rows_added=1)
            self._attach_put(dataclasses.replace(
                base, status=AttachStatus.IDLE, mode="srun", live=live, srun_rows=base.srun_rows + 1,
                consent="job" if consent == "job" else base.consent, offer=base.offer, next_at=None))
            return live
        live = self._attach_take(key)
        auto = base.auto and self.attach_open == key
        self._attach_put(dataclasses.replace(
            base, status=AttachStatus.LIVE if auto else AttachStatus.IDLE, live=live,
            next_at=self.clock.now() + timedelta(seconds=15) if auto else None))
        return live

    def attach_set_auto(self, key: JobKey, enabled: bool) -> AttachState:
        self._record("attach_set_auto", key, enabled)
        old = self.attach_states.get(key) or self._attach_base(key)
        status = old.status
        if old.mode == "ssh" and status in (AttachStatus.LIVE, AttachStatus.IDLE):
            status = AttachStatus.LIVE if enabled else AttachStatus.IDLE
        return self._attach_put(dataclasses.replace(
            old, auto=enabled, status=status,
            next_at=self.clock.now() + timedelta(seconds=15) if status is AttachStatus.LIVE else None))

    def attach_state(self, key: JobKey) -> AttachState | None:
        return self.attach_states.get(key)

    # ------------------------------------------------------------ test helpers (world mutation)
    def set_ok(self, host: str) -> None:
        st = self.store.hosts[host].status
        now = self.clock.now()
        self.store.set_status(host, dataclasses.replace(st, state=ConnState.OK, error=None, last_ok=now,
                                                        consecutive_failures=0, latency_s=st.latency_s or 0.2))

    def inject(self, host: str, fault: str) -> None:
        """Mimic demo ``world.inject``: ``unreachable`` / ``auth`` / ``ok``."""
        st = self.store.hosts[host].status
        if fault == "ok":
            self.set_ok(host)
        elif fault == "unreachable":
            self.store.set_status(host, dataclasses.replace(
                st, state=ConnState.UNREACHABLE, consecutive_failures=3,
                error=HostError(ErrorKind.TIMEOUT, "ssh: connect to host timed out", "", None, self.clock.now())))
        elif fault == "auth":
            self.store.set_status(host, dataclasses.replace(
                st, state=ConnState.NEEDS_AUTH,
                error=HostError(ErrorKind.AUTH, "Permission denied (publickey)", "", None, self.clock.now())))
        else:
            raise ValueError(fault)

    def check_idle(self) -> None:
        """What the poller does: pause everything after ``idle_pause_after`` without input."""
        idle = self.clock.monotonic() - self.last_activity
        if idle >= self.config.general.idle_pause_after and not self.store.idle_paused:
            self.store.set_idle_paused(True)

    def push_log(self, update: LogUpdate, index: int = -1) -> None:
        self.sessions[index].queue.put_nowait(update)

    def put_job(self, job: Job) -> None:
        self.store.put_job(job)

    def add_lyra_running_job(self) -> JobKey:
        """A RUNNING lyra job (srun-only attach, needs consent); not in the default world so snapshots stay put."""
        self.store.put_job(lyra_running_job())
        return LYRA_LIVE_KEY

    def tick(self, seconds: float = 0) -> None:
        """Advance time and re-stamp healthy hosts (a successful poll everywhere that is OK)."""
        if seconds:
            self.clock.advance(seconds)
        now = self.clock.now()
        for name, state in self.store.hosts.items():
            if state.status.state is ConnState.OK:
                state.status = dataclasses.replace(state.status, last_ok=now)
                self.store.emit(ChangeSet(name, status=True))
