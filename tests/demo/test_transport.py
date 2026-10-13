"""DemoTransport: section catalogue (Appendix A), outcomes, faults, actions, log streams, interactive argv."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from runfleet.config import HostConfig
from runfleet.demo.transport import HANDLERS, DemoTransport, derive_outcome, handler_for
from runfleet.demo.world import DemoWorld
from runfleet.models import ErrorKind
from runfleet.probe.section import ProbeScript, Section, SectionOutcome, StreamSpec
from runfleet.transport.base import Transport
from tests.demo.conftest import StepClock

NONCE = "0123456789ab"
K2_LOG = "/shared/lab/alice/orb/logs/orb-swp--train--cifar-warmup--s43--a02_3370390.out"
LBS_LOG = "/home/alice/projects/orb/logs/launch_voc_s44.log"

# Appendix A: every section name with representative meta, and the host that can answer it.
CATALOGUE: list[tuple[str, str, dict[str, object]]] = [
    ("k2", "detect.id", {}), ("lynx", "detect.tools", {}),
    ("k2", "slurm.version", {}), ("k2", "slurm.config", {}), ("k2", "slurm.assoc", {}), ("k2", "slurm.parts", {}),
    ("k2", "slurm.sacctfmt", {}), ("k2", "slurm.gputres", {}), ("labstation", "nvidia.version", {}),
    ("k2", "slurm.squeue", {"users": [], "extra": [], "sep": "\x1f", "format": "O"}),
    ("k2", "slurm.sacct", {"since_s": 86400, "fields": ["JobID", "JobIDRaw", "State"], "sep": "\x1f"}),
    ("lyra", "slurm.sacct_ids", {"ids_raw": ["32377076"], "fields": ["JobID", "State"], "sep": "\x1f"}),
    ("k2", "slurm.scontrol", {"ids_raw": ["3370390", "3372066"]}),
    ("vega", "slurm.sstat", {"ids_raw": ["823954"]}),
    ("lyra", "slurm.steps", {"job_id_raw": "32354789", "sep": "\x1f"}),
    ("k2", "slurm.logstat", {"paths": [K2_LOG]}),
    ("vega", "slurm.sinfo", {"partitions": ["h200"], "sep": "\x1f"}),
    ("vega", "slurm.occupants", {"partitions": ["h200"], "sep": "\x1f"}), ("k2", "slurm.names", {"users": ["bob"]}),
    ("k2", "slurm.act.hold", {"ids": ["3372066"]}), ("k2", "slurm.act.release", {"ids": ["3372066"]}),
    ("k2", "slurm.act.requeue", {"ids": ["3370391"], "hold": False}),
    ("k2", "slurm.act.cancel", {"ids": ["3370390"]}),
    ("k2", "procfs.host", {}), ("labstation", "procfs.host", {}),
    ("labstation", "nvidia.gpu", {"fields": ["index", "uuid", "name", "utilization.gpu"]}),
    ("labstation", "nvidia.apps", {}), ("labstation", "process.ps", {}),
    ("labstation", "process.fds", {"pids": [363942, 363948]}),
    ("labstation", "process.logstat", {"paths": [LBS_LOG]}),
    ("labstation", "process.exitscan", {"tasks": [("p590112-1788777037", [LBS_LOG])]}),
    ("labstation", "process.detail", {"pids": [363942, 363948], "root": 363942}),
    ("k2", "log.peek", {"path": K2_LOG, "bytes": 2048}), ("labstation", "log.fetch", {"path": LBS_LOG, "bytes": 4096}),
]


def tr(world: DemoWorld, host: str) -> DemoTransport:
    return DemoTransport(world.configs[host], world)


async def probe(t: DemoTransport, *sections: Section, purpose: str = "poll"):  # type: ignore[no-untyped-def]
    return await t.probe(ProbeScript(tuple(sections), NONCE, purpose), timeout=30)  # type: ignore[arg-type]


def test_satisfies_the_transport_protocol(world: DemoWorld) -> None:
    t: Transport = tr(world, "k2")       # structural typing (mypy); runtime facts below
    assert (t.host, t.kind, t.mux, t.ssh_log) == ("k2", "demo", "none", None)


def test_every_catalogue_section_has_a_handler() -> None:
    names = {name for _, name, _ in CATALOGUE}
    assert names <= set(HANDLERS)
    assert handler_for("process.kill.642229") is not None and handler_for("nope.x") is None


@pytest.mark.parametrize(("host", "name", "meta"), CATALOGUE, ids=[f"{h}:{n}" for h, n, _ in CATALOGUE])
async def test_catalogue_section_answers(world: DemoWorld, host: str, name: str, meta: dict[str, object]) -> None:
    world.clear("lynx")
    if name == "slurm.act.release":
        world.slurm_action("k2", "hold", ["3372066"])
    res = await probe(tr(world, host), Section(name, "true", meta=meta))
    r = res.sections[name]
    assert res.error is None
    assert r.ok, (r.outcome, r.rc, r.stderr)
    if r.outcome is SectionOutcome.OK and not name.startswith(("log.", "process.exitscan")):
        assert r.stdout.endswith("\n")             # table output; file tails end wherever the file does
    framed = res.run.stdout.decode()
    assert f"@@RF{NONCE} B {name}\n" in framed and f"@@RF{NONCE} E {name} {r.rc} " in framed


async def test_late_sections_run_last_and_results_cover_the_request(world: DemoWorld) -> None:
    res = await probe(tr(world, "labstation"), Section("nvidia.gpu", "x", late=True), Section("process.ps", "x"),
                      Section("procfs.host", "x"))
    framed = res.run.stdout.decode()
    assert framed.index("B process.ps") < framed.index("B procfs.host") < framed.index("B nvidia.gpu")
    assert list(res.sections) == ["nvidia.gpu", "process.ps", "procfs.host"]


async def test_missing_tool_unknown_section_and_wrong_host(world: DemoWorld) -> None:
    res = await probe(tr(world, "k2"), Section("nvidia.version", "x", requires=("nvidia-smi",)),
                      Section("bogus.section", "x"), Section("process.ps", "x"))
    missing, bogus, wrong = (res.sections[n] for n in ("nvidia.version", "bogus.section", "process.ps"))
    assert (missing.outcome, missing.rc, missing.stderr, missing.ms) == (SectionOutcome.MISSING, 127,
                                                                         "missing:nvidia-smi", 0)
    assert bogus.outcome is SectionOutcome.FAILED and "no simulation" in bogus.stderr
    assert wrong.outcome is SectionOutcome.FAILED and wrong.rc == 127
    assert f"@@RF{NONCE} E nvidia.version 127 0 missing:nvidia-smi\n" in res.run.stdout.decode()


def test_outcome_rules() -> None:
    assert derive_outcome(0, "x", "") is SectionOutcome.OK
    assert derive_outcome(0, "", "") is SectionOutcome.EMPTY
    assert derive_outcome(0, "", "fatal: serializer_required") is SectionOutcome.STDERR_ONLY
    assert derive_outcome(124, "", "") is SectionOutcome.TIMEOUT and derive_outcome(137, "", "") is SectionOutcome.TIMEOUT
    assert derive_outcome(1, "partial", "err") is SectionOutcome.FAILED
    assert derive_outcome(None, "", "") is SectionOutcome.NOT_RUN


async def test_empty_answers_are_empty(world: DemoWorld) -> None:
    res = await probe(tr(world, "lyra"), Section("slurm.squeue", "x", meta={"users": ["nobody"], "sep": "|"}))
    assert res.sections["slurm.squeue"].outcome is SectionOutcome.EMPTY


# ---------------------------------------------------------------------------------------------------- faults
async def test_needs_auth_until_connect(world: DemoWorld) -> None:
    t = tr(world, "lynx")
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.AUTH
    assert not await t.check_master()
    res = await probe(t, Section("process.ps", "x"))
    assert res.error is not None and res.error.kind is ErrorKind.AUTH and res.run.rc == 255
    assert res.sections["process.ps"].outcome is SectionOutcome.NOT_RUN
    assert b"Permission denied" in res.run.stderr
    assert await t.prepare() is not None                # a retry without connecting stays NEEDS_AUTH
    assert t.connect_argv() == [sys.executable, "-c", "print('demo: login ok')"]
    assert await t.prepare() is None and await t.check_master()
    assert (await probe(t, Section("process.ps", "x"))).sections["process.ps"].ok


async def test_unreachable_and_unknown_hosts(world: DemoWorld) -> None:
    world.inject("k2", "unreachable")
    t = tr(world, "k2")
    err = await t.prepare()
    assert err is not None and err.kind is ErrorKind.TIMEOUT
    res = await probe(t, Section("slurm.squeue", "x"))
    assert res.error is not None and res.error.kind is ErrorKind.TIMEOUT and not res.run.timed_out
    world.clear("k2", "unreachable")
    assert (await probe(t, Section("slurm.squeue", "x"))).error is None
    ghost = DemoTransport(HostConfig(name="atlas", ssh="atlas"), world)
    err = await ghost.prepare()
    assert err is not None and err.kind is ErrorKind.DNS and "atlas" in err.detail


async def test_nvidia_hang_then_guard(world: DemoWorld) -> None:
    world.inject("labstation", "nvidia_hang")
    t = tr(world, "labstation")
    first = (await probe(t, Section("nvidia.gpu", "x", late=True), Section("process.ps", "x"))).sections
    assert first["nvidia.gpu"].outcome is SectionOutcome.TIMEOUT and first["nvidia.gpu"].rc == 124
    assert first["process.ps"].ok                       # other sections keep working
    second = (await probe(t, Section("nvidia.gpu", "x", late=True))).sections["nvidia.gpu"]
    assert second.rc == 125 and second.outcome is SectionOutcome.FAILED and "already running" in second.stderr
    world.clear("labstation", "nvidia_hang")
    assert (await probe(t, Section("nvidia.gpu", "x", late=True))).sections["nvidia.gpu"].ok


async def test_garbage_and_slurmdbd_down(world: DemoWorld) -> None:
    t = tr(world, "k2")
    world.inject("k2", "garbage")
    res = await probe(t, Section("slurm.squeue", "x", meta={"sep": "|"}))
    assert res.sections["slurm.squeue"].ok and "3370390" not in res.sections["slurm.squeue"].stdout
    detect = await probe(t, Section("detect.id", "x"), purpose="detect")
    assert detect.sections["detect.id"].stdout.startswith("user=alice\n")
    world.clear("k2")
    world.inject("k2", "slurmdbd_down")
    res = await probe(t, Section("slurm.sacct", "x"), Section("slurm.squeue", "x"), Section("slurm.assoc", "x"))
    sacct = res.sections["slurm.sacct"]
    assert sacct.outcome is SectionOutcome.FAILED and "Problem talking to the database" in sacct.stderr
    assert res.sections["slurm.squeue"].ok and res.sections["slurm.assoc"].stderr.startswith("sacctmgr:")


# ---------------------------------------------------------------------------------------------------- actions
async def test_actions_change_the_world(world: DemoWorld) -> None:
    t = tr(world, "k2")
    res = await probe(t, Section("slurm.act.cancel", "x", meta={"ids": ["3370390", "123"]}), purpose="action")
    r = res.sections["slurm.act.cancel"]
    assert r.rc == 1 and r.outcome is SectionOutcome.FAILED
    assert r.stderr == "scancel: error: Kill job error on job id 123: Invalid job id specified"
    assert world.cluster("k2").jobs[3370390].state == "CANCELLED"
    sq = await probe(t, Section("slurm.squeue", "x", meta={"sep": "|"}))
    assert "\n3370390|" not in "\n" + sq.sections["slurm.squeue"].stdout
    world.clear("lynx")
    ws = world.workstation("lynx")
    kill = Section("process.kill.642229", "x", meta={"root": 642229, "start": ws.procs[642229].start, "sig": "TERM",
                                                     "grace_s": 10, "escalate": True})
    out = (await probe(tr(world, "lynx"), kill, purpose="action")).sections["process.kill.642229"]
    assert out.ok and out.stdout.splitlines()[-1] == "done" and "sent 642229" in out.stdout


# ---------------------------------------------------------------------------------------------------- streams
async def test_log_tail_follows_the_file(world: DemoWorld, clock: StepClock) -> None:
    t = tr(world, "k2")
    handle = await t.stream(StreamSpec("log.tail", "x", meta={"path": K2_LOG, "bytes": 512}))
    it = handle.chunks().__aiter__()
    first = await it.__anext__()
    assert len(first) == 512 and handle.returncode is None
    clock.advance(30)
    more = await it.__anext__()
    assert more and b"it/s" in more
    await handle.close()
    with pytest.raises(StopAsyncIteration):
        while True:
            await it.__anext__()
    assert handle.returncode == 0


async def test_log_tail_waits_for_a_missing_file(world: DemoWorld, clock: StepClock) -> None:
    k2 = world.cluster("k2")
    path = "/shared/lab/alice/rl-bench/logs/smoke-test_3372066.out"
    t = tr(world, "k2")
    handle = await t.stream(StreamSpec("log.tail", "x", meta={"path": path, "bytes": 65536}))
    it = handle.chunks().__aiter__()
    pending = asyncio.ensure_future(it.__anext__())
    await asyncio.sleep(0.05)                            # the stream looked once: no such file yet
    assert "cannot open" in handle.stderr_text
    clock.advance(36 * 60)                               # the pending job starts and creates its log
    data = await pending
    assert data.startswith(b"[") and path in k2.logs
    assert "cannot open" in handle.stderr_text and "has appeared" in handle.stderr_text
    await handle.close()


async def test_stream_failures(world: DemoWorld) -> None:
    other = await tr(world, "k2").stream(StreamSpec("log.other", "x"))
    assert [c async for c in other.chunks()] == [] and other.returncode == 1
    world.inject("k2", "unreachable")
    dead = await tr(world, "k2").stream(StreamSpec("log.tail", "x", meta={"path": K2_LOG, "bytes": 10}))
    assert [c async for c in dead.chunks()] == [] and dead.returncode == 255
    assert "Connection timed out" in dead.stderr_text


# ---------------------------------------------------------------------------------------------------- interactive
def test_interactive_and_connect_argv_run_locally(world: DemoWorld) -> None:
    t = tr(world, "labstation")
    argv = t.interactive_argv("cd /tmp && exec $SHELL -l")
    out = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    assert out.returncode == 0 and "ssh -t labstation cd /tmp" in out.stdout
    login = subprocess.run(t.connect_argv() or [], capture_output=True, text=True, timeout=20)
    assert login.stdout.strip() == "demo: login ok"


async def test_latency_is_simulated_when_enabled(clock: StepClock) -> None:
    import time
    world = DemoWorld(seed=1, now=clock, latency_scale=1.0)
    t = tr(world, "lyra")                                  # ProxyJump-shaped: 200-300 ms
    start = time.monotonic()
    await probe(t, Section("detect.id", "x"))
    assert 0.18 < time.monotonic() - start < 2.0


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_occupants_account_for_every_busy_gpu(world: DemoWorld, host: str) -> None:
    """The simulated other users' jobs hold exactly each node's background GPUs; the user's own jobs are in it too."""
    from runfleet.backends.slurm.parse_occupants import parse_names, parse_occupants
    from runfleet.demo import render
    from runfleet.demo.world import free_gpus

    c = world.cluster(host)
    parts = sorted({p for n in c.nodes for p in n.partitions})
    pr = parse_occupants(render.squeue_occupants(c, world.now(), partitions=parts).stdout)
    assert pr.items and not pr.bad_lines
    held: dict[str, int] = {}
    for o in pr.items:
        for n in o.nodes:
            held[n] = held.get(n, 0) + (o.gpus or 0) // len(o.nodes)
    for node in c.nodes:
        if node.state not in ("down", "drained"):
            assert held.get(node.name, 0) == node.gpu_total - len(free_gpus(c, node, None)), node.name
    assert any(o.user == c.user for o in pr.items)
    names = parse_names(render.getent_names(c, users=sorted({o.user for o in pr.items}) + ["ghost"]).stdout)
    assert names[c.user] == render.OWN_NAME and "ghost" not in names
