"""[WP2] ProcfsMonitor: loadavg, meminfo, /proc/stat delta, uptime (§5.3).

On a Slurm profile the group is *piggyback* at the jobs interval, so it never adds round trips to a login node; on a
workstation it is an interval group at the GPU period. CPU % is the delta of the `cpu` line between two ticks
(no remote sleep); the first sample has none.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from runfleet.backends.base import Cadence, CollectorOutput, HostContext, MonitorCollector, MonitorView
from runfleet.backends.common import section_error, usable
from runfleet.config import HostConfig
from runfleet.models import Capabilities, HostError, HostMetrics
from runfleet.probe.section import Section, SectionResult
from runfleet.util.slurmfmt import parse_float, parse_int

PROCFS_CMD = "\n".join([
    "printf 'ncpu=%s\\nepoch=%s\\n' \"$(nproc 2>/dev/null)\" \"$(date +%s)\"; "
    "printf 'load=%s\\n' \"$(cat /proc/loadavg)\"",
    "grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo; head -1 /proc/stat",
    "printf 'uptime=%s\\n' \"$(cut -d' ' -f1 /proc/uptime)\"",
])


@dataclass(frozen=True, slots=True)
class CpuTimes:
    total: int
    idle: int        # idle + iowait


@dataclass(frozen=True, slots=True)
class ProcfsSample:
    ncpu: int | None
    epoch: int | None
    load: tuple[float, float, float] | None
    mem_kb: Mapping[str, int]
    cpu: CpuTimes | None
    uptime_s: float | None


def parse_cpu_line(line: str) -> CpuTimes | None:
    """`cpu  user nice system idle iowait irq softirq steal …` → totals (guest time is already in user/nice)."""
    parts = line.split()
    if not parts or parts[0] != "cpu" or len(parts) < 5:
        return None
    try:
        vals = [int(x) for x in parts[1:9]]
    except ValueError:
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return CpuTimes(total=sum(vals), idle=idle)


def cpu_pct(prev: CpuTimes | None, cur: CpuTimes | None) -> float | None:
    """(Δtotal − Δidle − Δiowait) / Δtotal in percent; None without a previous sample or a counter reset."""
    if prev is None or cur is None:
        return None
    dt = cur.total - prev.total
    if dt <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (dt - (cur.idle - prev.idle)) / dt))


def parse_procfs(text: str) -> ProcfsSample:
    ncpu = epoch = None
    load: tuple[float, float, float] | None = None
    mem: dict[str, int] = {}
    cpu: CpuTimes | None = None
    uptime: float | None = None
    for line in text.splitlines():
        if line.startswith("ncpu="):
            ncpu = parse_int(line[5:])
        elif line.startswith("epoch="):
            epoch = parse_int(line[6:])
        elif line.startswith("load="):
            vals = line[5:].split()
            try:
                load = (float(vals[0]), float(vals[1]), float(vals[2]))
            except (IndexError, ValueError):
                load = None
        elif line.startswith("uptime="):
            uptime = parse_float(line[7:])
        elif line.startswith("cpu "):
            cpu = parse_cpu_line(line)
        elif ":" in line:
            key, _, rest = line.partition(":")
            value = rest.split()
            if value and value[0].isdigit():
                mem[key.strip()] = int(value[0])
    return ProcfsSample(ncpu=ncpu, epoch=epoch, load=load, mem_kb=mem, cpu=cpu, uptime_s=uptime)


class ProcfsMonitor(MonitorCollector):
    name = "procfs"

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        self._prev_cpu: CpuTimes | None = None

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return True

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        if "slurm" in ctx.caps.profile:
            return [Cadence("procfs.host", max(ctx.host.slurm.poll_jobs, 10.0), floor=10.0, mode="piggyback")]
        return [Cadence("procfs.host", max(ctx.host.workstation.poll_gpu, 2.0), floor=2.0)]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        if "procfs.host" not in due:
            return []
        return [Section("procfs.host", PROCFS_CMD, timeout=5)]

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        r = results.get("procfs.host")
        if r is None:
            return CollectorOutput()
        errors: list[HostError] = []
        if not r.ok:
            err = section_error(r, ctx.now)
            if err is not None:
                errors.append(err)
        if not usable(r):
            return CollectorOutput(errors=tuple(errors))
        s = parse_procfs(r.stdout)
        pct = cpu_pct(self._prev_cpu, s.cpu)
        if s.cpu is not None:
            self._prev_cpu = s.cpu

        def kb(key: str) -> int | None:
            v = s.mem_kb.get(key)
            return None if v is None else v * 1024

        metrics = HostMetrics(ncpu=s.ncpu, load=s.load, cpu_pct=pct, mem_total=kb("MemTotal"),
                              mem_available=kb("MemAvailable"), swap_total=kb("SwapTotal"), swap_free=kb("SwapFree"),
                              uptime_s=s.uptime_s, sampled_at=ctx.now)
        return CollectorOutput(metrics=metrics, errors=tuple(errors))
