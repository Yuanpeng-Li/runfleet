"""[WP2] NvidiaMonitor: nvidia-smi sections and parsers (§5.3).

Both sections are `late` and wrapped: a hung nvidia-smi usually sits in uninterruptible sleep, where SIGKILL does not
end it and GNU `timeout` would wait for it. The wrapper therefore runs the payload in the background with detached
stdio, polls for completion for 6 s and abandons it (rc 124); a guard refuses to stack a third stuck instance (rc
125) — stuck = D state or still owned by an abandoned wrapper; healthy nvidia-smi a job or the user runs do not count.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from runfleet.backends.base import Cadence, CollectorOutput, HostContext, MonitorCollector, MonitorView
from runfleet.backends.common import ParseResult, parse_error, section_error, usable
from runfleet.backends.process.parse_ps import APPS_COLUMNS, parse_ps_rows
from runfleet.config import HostConfig
from runfleet.models import GPU, Capabilities, ErrorKind, GPUProcess, HostError, Proc
from runfleet.probe.section import Section, SectionOutcome, SectionResult
from runfleet.util.slurmfmt import nullish, parse_float, parse_int
from runfleet.util.slurmfmt import split_rows

log = logging.getLogger("runfleet.backends.common.nvidia")

GPU_FIELDS: tuple[str, ...] = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total",
                               "temperature.gpu", "power.draw", "power.limit", "fan.speed", "pstate",
                               "utilization.memory")
HUNG_RCS = frozenset({124, 125})
HUNG_SUSPEND = timedelta(minutes=5)
HUNG_LIMIT = 3
APPS_MAX_AGE = 1.5           # × the GPU poll interval: older process lists are not attached to a fresh GPU sample

# The stacking guard counts only the user's nvidia-smi that are stuck: in D state (a hung query survives SIGKILL), or
# still waited for by a RunFleet wrapper (`rf_nv` in the parent's argv: a query abandoned in the background). A job's
# own `nvidia-smi -l`, the user's `dmon` in tmux etc. are healthy and never block the probe. A `ps` without these
# options (BusyBox) falls back to counting every nvidia-smi of the user with pgrep (rc 1 = none); when processes
# cannot be listed at all the query is refused: never fail open.
_GUARD = "\n".join([
    "rf_g() {",
    "  if rf_p=$(ps -ww -u \"$(id -un)\" -o pid=,ppid=,stat=,args= 2>/dev/null); then",
    "    printf '%s\\n' \"$rf_p\" | awk '{c=$4; sub(/.*\\//, \"\", c); a[$1]=$0} c == \"nvidia-smi\" {n[$1]=$2; s[$1]=$3}"
    " END {k=0; for (p in n) if (s[p] ~ /^D/ || index(a[n[p]], \"rf_nv\")) k++; print k}' || echo x",
    "    return",
    "  fi",
    "  rf_p=$(pgrep -u \"$(id -un)\" -x nvidia-smi 2>/dev/null); [ $? -le 1 ] || { echo x; return; }",
    "  printf '%s\\n' \"$rf_p\" | grep -c .",
    "}",
    "rf_k=$(rf_g)",
    "case $rf_k in ''|*[!0-9]*) echo 'nvidia-smi guard: cannot list processes (refusing)' >&2; exit 125;; esac",
    "[ \"$rf_k\" -ge 2 ] && { echo 'nvidia-smi already running (hung?)' >&2; exit 125; }",
])


def wrap(payload: str) -> str:
    """Background + poll + abandon wrapper around a payload that runs nvidia-smi (§5.3.1)."""
    return "\n".join([
        _GUARD,
        "rf_nv() {",
        payload,
        "}",
        "[ -n \"$RF_T\" ] || { rf_nv; exit $?; }      # no temp dir: run in foreground",
        "o=\"$RF_T/nv.$$\"",
        # `(rf_nv)`: an `exit` inside the payload must end only that subshell, never skip the rc file
        "( (rf_nv) >\"$o\" 2>\"$o.e\"; echo $? >\"$o.rc\" ) </dev/null >/dev/null 2>&1 &",
        "i=0; while [ ! -s \"$o.rc\" ] && [ $i -lt 60 ]; do sleep 0.1 2>/dev/null || sleep 1; i=$((i+1)); done",
        "[ -s \"$o.rc\" ] || { echo 'nvidia-smi did not answer within 6 s (abandoned)' >&2; exit 124; }",
        "cat \"$o\"; cat \"$o.e\" >&2; exit \"$(cat \"$o.rc\")\"",
    ])


def gpu_query_cmd(fields: Sequence[str] = GPU_FIELDS) -> str:
    return wrap(f"nvidia-smi --query-gpu={','.join(fields)} --format=csv,noheader,nounits")


APPS_PAYLOAD = """a=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits) || return $?
printf '%s\\n' "$a"; echo '@@ps'
p=$(printf '%s\\n' "$a" | awk -F', ' 'NF>=2 {print $2}' | sort -u | paste -sd, -)
[ -n "$p" ] && ps -ww -o pid=,ppid=,user:64=,etimes=,pcpu=,pmem=,rss=,stat=,args= -p "$p"; true"""


def apps_cmd() -> str:
    return wrap(APPS_PAYLOAD)


def _num(value: str) -> float | None:
    v = nullish(value)
    return None if v is None else parse_float(v.split()[0] if v else v)


def parse_gpu_query(text: str, fields: Sequence[str] = GPU_FIELDS, *, host: str,
                    sampled_at: datetime | None = None) -> ParseResult[GPU]:
    """`--query-gpu=…,noheader,nounits` rows → GPUs. Index and uuid are the first two fields; the last
    `len(fields) − 3` values are taken from the right in query order; everything between is the name (it may contain
    commas or `, `)."""
    pr: ParseResult[GPU] = ParseResult()
    tail_fields = list(fields[3:])
    n_tail = len(tail_fields)
    for n, line in enumerate(split_rows(text), 1):
        if not line.strip():
            continue
        parts = line.split(", ")
        if len(parts) < 3 + n_tail:
            pr.bad(n, line, f"expected {len(fields)} fields")
            continue
        index = parse_int(parts[0])
        if index is None:
            pr.bad(n, line, "bad index")
            continue
        tail = parts[len(parts) - n_tail:] if n_tail else []
        name = ", ".join(parts[2:len(parts) - n_tail]).strip()
        values = dict(zip(tail_fields, tail, strict=True))

        def num(key: str, vals: Mapping[str, str] = values) -> float | None:
            return _num(vals[key]) if key in vals else None

        pstate = nullish(values.get("pstate"))
        pr.items.append(GPU(
            host=host, index=index, uuid=nullish(parts[1]), name=name, util_pct=num("utilization.gpu"),
            mem_used_mib=num("memory.used"), mem_total_mib=num("memory.total"),
            mem_util_pct=num("utilization.memory"), temp_c=num("temperature.gpu"), power_w=num("power.draw"),
            power_limit_w=num("power.limit"), fan_pct=num("fan.speed"), pstate=pstate, source="nvidia-smi",
            sampled_at=sampled_at, bus_id=nullish(values.get("pci.bus_id")),     # Attach queries pci.bus_id last
        ))
    return pr


def parse_apps(text: str) -> tuple[list[tuple[str, int, float | None]], ParseResult[Proc]]:
    """`nvidia.apps` output → ([(gpu_uuid, pid, used MiB)], ps rows of those pids as Proc(pgid=None))."""
    apps_text, _, ps_text = text.partition("@@ps")
    apps: list[tuple[str, int, float | None]] = []
    for line in split_rows(apps_text):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        apps.append((parts[0], int(parts[1]), _num(parts[2]) if len(parts) > 2 else None))
    return apps, parse_ps_rows(split_rows(ps_text), APPS_COLUMNS)


def attach_processes(gpus: Sequence[GPU], apps: Sequence[tuple[str, int, float | None]],
                     procs: Sequence[Proc]) -> tuple[GPU, ...]:
    """Join compute apps to GPUs by uuid (→ GPU index) and ps rows by pid."""
    by_uuid = {g.uuid: g for g in gpus if g.uuid}
    by_pid = {p.pid: p for p in procs}
    per_gpu: dict[str | None, list[GPUProcess]] = {}
    for uuid, pid, mem in apps:
        gpu = by_uuid.get(uuid)
        ps = by_pid.get(pid)
        per_gpu.setdefault(uuid, []).append(GPUProcess(
            pid=pid, gpu_uuid=uuid, gpu_index=gpu.index if gpu else None, used_mem_mib=mem,
            user=ps.user if ps else None, command=ps.args if ps else None, etime_s=ps.etime_s if ps else None,
            cpu_pct=ps.cpu_pct if ps else None, job=None))
    return tuple(dataclasses.replace(g, processes=tuple(per_gpu.get(g.uuid, ()))) for g in gpus)


def parse_driver_version(text: str) -> str | None:
    """`nvidia-smi --version`: `KMD version : 610.43.02`, falling back to `DRIVER version` (now "Deprecated")."""
    values: dict[str, str] = {}
    for line in split_rows(text):
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip().lower()] = value.strip()
    for key in ("kmd version", "driver version"):
        v = values.get(key)
        if v and v[0].isdigit():
            return v
    return None


class NvidiaMonitor(MonitorCollector):
    name = "nvidia"

    def __init__(self, host: HostConfig) -> None:
        super().__init__(host)
        self._hung = 0
        self._suspended_until: datetime | None = None
        self._last_apps: tuple[list[tuple[str, int, float | None]], list[Proc]] | None = None
        self._last_apps_at: datetime | None = None

    @classmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool:
        return "nvidia-smi" in caps.tools

    @classmethod
    def capability_sections(cls, host: HostConfig) -> list[Section]:
        return [Section("nvidia.version", "nvidia-smi --version", timeout=10, requires=("nvidia-smi",))]

    @classmethod
    def ingest_capabilities(cls, results: Mapping[str, SectionResult]) -> Mapping[str, object]:
        r = results.get("nvidia.version")
        if r is None or not usable(r):
            return {}
        version = parse_driver_version(r.stdout)
        return {"driver_version": version} if version else {}

    def cadences(self, ctx: HostContext) -> list[Cadence]:
        return [Cadence("nvidia.gpu", max(ctx.host.workstation.poll_gpu, 2.0), floor=2.0, focused_factor=0.6)]

    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]:
        if "nvidia.gpu" not in due:
            return []
        if self._suspended_until is not None and ctx.now < self._suspended_until:
            return []
        return [Section("nvidia.gpu", gpu_query_cmd(), timeout=8, requires=("nvidia-smi",), late=True,
                        meta={"fields": list(GPU_FIELDS)}),
                Section("nvidia.apps", apps_cmd(), timeout=8, requires=("nvidia-smi",), late=True)]

    def _classify(self, r: SectionResult, at: datetime) -> HostError | None:
        text = r.stderr + r.stdout
        if r.rc == 9 or "couldn't communicate with the NVIDIA driver" in text:
            return HostError(ErrorKind.REMOTE, "NVIDIA driver not loaded", r.stderr[:4096], r.name, at)
        if r.rc in HUNG_RCS or r.outcome is SectionOutcome.TIMEOUT:
            return HostError(ErrorKind.TIMEOUT, "nvidia-smi hung", r.stderr[:4096], r.name, at)
        return section_error(r, at)

    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput:
        now = ctx.now
        gr = results.get("nvidia.gpu")
        ar = results.get("nvidia.apps")
        if gr is None and ar is None:
            return CollectorOutput()
        errors: list[HostError] = []
        gpus: tuple[GPU, ...] | None = None
        gpu_procs: tuple[Proc, ...] | None = None
        hung = False
        if gr is not None:
            if gr.rc == 6 or "No devices were found" in gr.stdout + gr.stderr:
                gpus = ()
            elif usable(gr):
                pr = parse_gpu_query(gr.stdout, host=ctx.host.name, sampled_at=now)
                if pr.mostly_bad:
                    errors.append(parse_error(gr.name, pr, now, "nvidia-smi"))
                else:
                    gpus = tuple(pr.items)
            if not gr.ok and gpus is None:
                err = self._classify(gr, now)
                if err is not None:
                    errors.append(err)
                    hung = err.summary == "nvidia-smi hung"
        if ar is not None:
            if usable(ar):
                apps, procs = parse_apps(ar.stdout)
                self._last_apps, self._last_apps_at = (apps, procs.items), now
                gpu_procs = tuple(procs.items)
            elif not ar.ok:
                err = self._classify(ar, now)
                if err is not None and not any(e.summary == err.summary for e in errors):
                    errors.append(err)
                hung = hung or (err is not None and err.summary == "nvidia-smi hung")
        if gpus == ():
            self._last_apps = self._last_apps_at = None
        max_age = timedelta(seconds=max(ctx.host.workstation.poll_gpu, 2.0) * APPS_MAX_AGE)
        if (gpus is not None and self._last_apps is not None and self._last_apps_at is not None
                and now - self._last_apps_at <= max_age):     # one failed apps query is bridged, not more
            gpus = attach_processes(gpus, *self._last_apps)
        self._track_hangs(hung, now)
        return CollectorOutput(gpus=gpus, gpu_procs=gpu_procs, errors=tuple(errors))

    def _track_hangs(self, hung: bool, now: datetime) -> None:
        """After 3 consecutive hung outcomes the group is suspended for 5 min (§5.3.1)."""
        if not hung:
            self._hung = 0
            return
        self._hung += 1
        if self._hung >= HUNG_LIMIT:
            log.warning("%s: nvidia-smi hung %d times in a row; pausing GPU polling for 5 min", self.host.name,
                        self._hung)
            self._suspended_until = now + HUNG_SUSPEND
            self._hung = 0
