"""[WP4] "Find other log files" in the demo world (§4.7.2, §10.6): k2's launcher job writes 6 lines to stdout and goes
quiet while its train.log keeps growing; `--demo` answers `log.find` from the simulated files (clusters and
workstations), and the found file peeks, heads and streams like a stdout. An ended job lists nothing written after its
end + 5 min; other jobs' stdouts are marked, never suggested; the paths are the same in every CLI process."""

from __future__ import annotations

import posixpath
from datetime import UTC, datetime, timedelta
from typing import Any

from runfleet.core.clock import ManualClock
from runfleet.core.logs import (FIND_END_SLACK, FIND_MAX_DEPTH, FIND_MAX_FILES, find_section, known_logs, suggest,
                                tail_spec)
from runfleet.core.service import FleetService
from runfleet.demo import render
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import (LAUNCHER_JOB, LAUNCHER_TRAIN_LOG, LAUNCHER_WANDB_LOG, LAUNCHER_WORKDIR, DemoWorld,
                                 SimLog)
from runfleet.models import JobKey, LogRef
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult
from tests.demo.conftest import StepClock
from tests.ui.fakes import T0


class _Recording(DemoTransport):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.sections: list[str] = []

    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult:
        self.sections += [s.name for s in script.sections]
        return await super().probe(script, timeout=timeout)


def _service(at: datetime = T0) -> tuple[FleetService, DemoWorld, ManualClock, dict[str, _Recording]]:
    clock = ManualClock(at)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0, stream_poll_s=0.01)
    transports: dict[str, _Recording] = {}

    def make(h: Any) -> _Recording:
        transports[h.name] = _Recording(h, world)
        return transports[h.name]

    svc = FleetService(world.fleet_config(), make, clock, None)
    svc.lock_path = False
    return svc, world, clock, transports


def test_launcher_stdout_goes_quiet_while_train_log_grows(world: DemoWorld, clock: StepClock) -> None:
    c = world.cluster("k2")
    job = c.jobs[LAUNCHER_JOB]
    assert job.state == "RUNNING" and job.workdir == LAUNCHER_WORKDIR and job.start is not None
    out, train = c.logs[job.log_paths[0]], c.logs[LAUNCHER_TRAIN_LOG]
    assert out.size < 1024 and out.text().count("\n") == 6 and "launcher:" in out.text()
    assert LAUNCHER_TRAIN_LOG.removeprefix(LAUNCHER_WORKDIR + "/") in out.text()
    assert out.mtime < world.t0 - 12 * 3600                      # the Logs tab says "13h ago"
    assert train.size > 10 * out.size and "epoch" in train.text()
    sizes = (out.size, train.size)
    clock.advance(600)
    world.sync()
    assert out.size == sizes[0] and train.size > sizes[1] and train.mtime > world.t0
    assert len(job.log_paths) == 1                                # stdout only (merged): Slurm knows no other file


def test_render_log_find_rules(world: DemoWorld) -> None:
    c = world.cluster("k2")
    job = c.jobs[LAUNCHER_JOB]
    assert job.start is not None
    since = int(job.start) - 60
    out = render.log_find(c, workdir=LAUNCHER_WORKDIR, since=since, max_depth=FIND_MAX_DEPTH,
                          max_files=FIND_MAX_FILES)
    assert out.rc == 0 and out.stdout.endswith("@@end 0\n")
    lines = out.stdout.split("\n")[:-2]
    assert all(ln.endswith("\x1f") for ln in lines)
    paths = [ln.split(" ", 2)[2].removesuffix("\x1f") for ln in lines]
    rel = [p.removeprefix(LAUNCHER_WORKDIR + "/") for p in paths]
    assert rel[0] == "runs/in1k-b16/r0412/s42/train.log"
    assert set(rel) == {"runs/in1k-b16/r0412/s42/train.log", "runs/in1k-b16/r0412/s42/eval.err",
                        posixpath.relpath(LAUNCHER_WANDB_LOG, LAUNCHER_WORKDIR),
                        posixpath.relpath(job.log_paths[0], LAUNCHER_WORKDIR)}
    assert render.log_find(c, workdir="/nonexistent/dir", since=0, max_depth=6, max_files=30).stdout == "@@nodir\n"
    wide = render.log_find(c, workdir=LAUNCHER_WORKDIR, since=0, max_depth=FIND_MAX_DEPTH, max_files=30).stdout
    assert "r0398/s42/train.log" in wide and ".venv" not in wide and "deep" not in wide and "prep.log" not in wide
    early = render.log_find(c, workdir=LAUNCHER_WORKDIR, since=since, max_depth=FIND_MAX_DEPTH, max_files=30,
                            until=int(job.start) + 100).stdout       # `until`: nothing written after it
    assert "eval.err" in early and "output.log" in early and "train.log" not in early

def test_render_log_find_cap() -> None:
    from types import SimpleNamespace

    logs = {f"/w/r/{i:02d}.log": SimLog(f"/w/r/{i:02d}.log", 1000.0 + i) for i in range(40)}
    for lg in logs.values():
        lg.write("x\n", lg.created)
    out = render.log_find(SimpleNamespace(logs=logs), workdir="/w", since=0, max_depth=6,   # type: ignore[arg-type]
                          max_files=30).stdout
    lines = out.split("\n")
    assert len(lines) == 33 and lines[0].endswith("/w/r/39.log\x1f") and lines[-2:] == ["@@end 0", ""]


async def test_find_logs_round_trip_on_a_cluster() -> None:
    svc, world, _clock, transports = _service()
    await svc.collect_once()
    job = svc.store.job(JobKey("k2", str(LAUNCHER_JOB)))
    assert job is not None and job.workdir == LAUNCHER_WORKDIR and job.start_time is not None
    stdout = next(r for r in job.logs if r.stream == "stdout")
    before = transports["k2"].sections.count("log.find")
    r = await svc.find_logs("k2", job.workdir, since=job.start_time)
    assert transports["k2"].sections.count("log.find") == before + 1 == 1      # one probe, never from the poll
    assert r.error is None and r.file_error is None and not r.truncated and r.workdir == LAUNCHER_WORKDIR
    assert r.files[0].path == LAUNCHER_TRAIN_LOG and r.files[0].size == world.cluster("k2").logs[LAUNCHER_TRAIN_LOG].size
    assert stdout.path in [f.path for f in r.files]                 # the caller drops the job's own stdout
    others = [f for f in r.files if f.path != stdout.path]
    assert len(others) == 3 and all(f.mtime >= job.start_time for f in others)
    assert [f.mtime for f in r.files] == sorted((f.mtime for f in r.files), reverse=True)
    day = await svc.find_logs("k2", LAUNCHER_WORKDIR)               # no start time: the last 24 h
    assert {f.path for f in day.files} >= {f.path for f in r.files}
    gone = await svc.find_logs("k2", "/shared/lab/alice/nope")
    assert gone.file_error == "No such directory" and gone.files == ()
    down = await svc.find_logs("nohost", LAUNCHER_WORKDIR)
    assert down.error is not None and down.files == ()


async def test_find_logs_on_a_workstation() -> None:
    svc, world, _clock, _t = _service()
    await svc.collect_once()
    ws = world.workstation("labstation")
    task = next(j for j in svc.store.jobs(["labstation"]) if j.workdir and any(r.path for r in j.logs))
    own = next(r.path for r in task.logs if r.path)
    assert task.workdir is not None and own is not None
    r = await svc.find_logs("labstation", task.workdir)
    assert r.error is None and r.file_error is None and own in [f.path for f in r.files]
    assert all(f.path.startswith(task.workdir.rstrip("/") + "/") for f in r.files)
    assert all(f.path in ws.logs for f in r.files)


async def test_found_file_peeks_heads_and_streams() -> None:
    svc, world, clock, _t = _service()
    await svc.collect_once()
    found = (await svc.find_logs("k2", LAUNCHER_WORKDIR)).files[0]
    ref = LogRef(stream="file", path=found.path, source="found", size=found.size, mtime=found.mtime)
    peek = await svc.peek_log("k2", ref)
    assert peek.error is None and peek.size == found.size and any("epoch" in ln for ln in peek.lines)
    head = await svc.head_log("k2", ref)
    assert head.error is None and head.lines[0].startswith("[") and "workdir " + LAUNCHER_WORKDIR in head.lines[1]
    lg = world.cluster("k2").logs[found.path]
    t = DemoTransport(world.configs["k2"], world)
    handle = await t.stream(tail_spec(found.path, 4096))
    chunks = handle.chunks()
    first = await chunks.__anext__()
    assert first == lg.tail(4096)
    size = lg.size
    clock.advance(30)
    more = await chunks.__anext__()
    await handle.close()
    assert lg.size > size and more == lg.read_from(size)[:len(more)] and more


async def test_demo_section_through_the_framing() -> None:
    """The real section (built by core) answered by the demo transport, framed and parsed like a remote one."""
    world = DemoWorld(seed=7, now=lambda: T0, latency_scale=0.0)
    t = DemoTransport(world.configs["k2"], world)
    job = world.cluster("k2").jobs[LAUNCHER_JOB]
    assert job.start is not None
    sec = find_section(LAUNCHER_WORKDIR + "/", datetime.fromtimestamp(job.start, tz=UTC), T0)
    res = await t.probe(framing.make_script([sec], "log"), timeout=30)
    r = res.sections["log.find"]
    assert r.ok and r.stdout.startswith(f"{world.cluster('k2').logs[LAUNCHER_TRAIN_LOG].mtime:.10f} ")


async def test_found_paths_are_the_same_in_every_process() -> None:
    """Every CLI process re-anchors the demo world to its own now: `agent log --files` in one process, then
    `--file wandb/…/output.log` in the next (seconds or an hour later) must name a file the second one lists."""
    listed: list[set[str]] = []
    for at in (T0, T0 + timedelta(seconds=47), T0 + timedelta(hours=1, seconds=3)):
        svc, world, _clock, _t = _service(at)                         # a fresh world, as a new process builds
        await svc.collect_once()
        job = svc.store.job(JobKey("k2", str(LAUNCHER_JOB)))
        assert job is not None and job.start_time is not None
        r = await svc.find_logs("k2", LAUNCHER_WORKDIR, since=job.start_time)
        listed.append({f.path for f in r.files})
        assert LAUNCHER_WANDB_LOG in world.cluster("k2").logs
    assert LAUNCHER_WANDB_LOG in listed[0] and listed[0] == listed[1] == listed[2]


async def test_an_ended_job_lists_nothing_written_after_its_end() -> None:
    svc, _world, _clock, _t = _service()
    await svc.collect_once()
    job = svc.store.job(JobKey("k2", "3371193"))                       # timed out ~5 h before the demo starts
    assert job is not None and job.end_time is not None and job.workdir is not None
    open_ended = await svc.find_logs("k2", job.workdir, since=job.start_time)
    ended = await svc.find_logs("k2", job.workdir, since=job.start_time, until=job.end_time)
    limit = job.end_time + timedelta(seconds=FIND_END_SLACK)
    assert any(f.mtime > limit for f in open_ended.files)              # later jobs wrote here since
    assert ended.files and all(f.mtime <= limit for f in ended.files) and ended.incomplete is None
    own = {r.path for r in job.logs if r.path}
    assert own & {f.path for f in ended.files}


async def test_other_jobs_logs_are_marked_and_never_suggested() -> None:
    svc, _world, _clock, _t = _service()
    await svc.collect_once()
    async def best(key: JobKey) -> tuple[str | None, dict[str, str]]:
        job = svc.store.job(key)
        assert job is not None and job.workdir is not None
        r = await svc.find_logs(key.host, job.workdir, since=job.start_time,
                                until=job.end_time if job.state.is_terminal else None)
        known = known_logs(svc.store.jobs([key.host]), key)
        s = suggest(r.files, [*(x.path for x in job.logs if x.path), *known])
        return (s.path if s else None), {f.path: known[f.path] for f in r.files if f.path in known}

    path, marked = await best(JobKey("k2", str(LAUNCHER_JOB)))
    assert path == LAUNCHER_TRAIN_LOG and marked == {}
    tasks = [j for j in svc.store.jobs(["labstation"]) if j.workdir == "/home/alice/projects/orb"]
    assert len(tasks) == 2                                             # two tasks share a workdir and logs/
    for task in tasks:
        path, marked = await best(task.key)
        other = next(t for t in tasks if t is not task)
        assert path is None                                            # only the two stdouts: no suggestion
        assert list(marked.values()) == [f"job {other.key.label} stdout"]
    path, marked = await best(JobKey("k2", "3371193"))                  # ended: its own .out and older jobs' .outs
    assert path is None and marked and all(v.startswith("job 33707") for v in marked.values())
