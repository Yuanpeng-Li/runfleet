"""[WP4] LogService.read (§4.7, `runfleet agent log`): one probe, every complete line of the window, the live
segment, typed transport errors, partial output, file errors, no path; parse_read on its own; the demo serves it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runfleet.core.clock import ManualClock
from runfleet.core.logs import HEAD_LINES, LogService, parse_read
from runfleet.models import ErrorKind, LogRef
from runfleet.probe.section import ProbeScript, SectionOutcome
from runfleet.transport.base import ProbeResult
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START

MTIME = 1788696459
REF = LogRef("stdout", "/shared/lab/alice/orb/logs/orb-eval_2331977.out")


class Probe:
    def __init__(self, t: FakeTransport) -> None:
        self.t = t
        self.scripts: list[ProbeScript] = []

    async def __call__(self, host: str, script: ProbeScript) -> ProbeResult:
        self.scripts.append(script)
        return await self.t.probe(script, timeout=10)


def make(t: FakeTransport) -> tuple[LogService, Probe]:
    probe = Probe(t)
    svc = LogService(ManualClock(CLOCK_START), transport_for=lambda h: t, stream_limit_for=lambda h: 1, probe=probe,
                     caps_for=lambda h: None)
    return svc, probe


def test_parse_read_keeps_every_line_and_drops_the_cut_fragment() -> None:
    body = "\n".join(f"line {i}" for i in range(50)) + "\nepoch 1 10%\repoch 1 20%"
    r = parse_read(f"100000|{MTIME}\nragment\n{body}", "", 4096, CLOCK_START, "/w/x.out")
    assert r.lines == tuple(f"line {i}" for i in range(50))          # all of them, not the last 8 of a peek
    assert r.live == "epoch 1 20%" and r.started_mid_file and r.window_bytes == 4096
    assert r.size == 100000 and r.mtime == datetime.fromtimestamp(MTIME, tz=UTC) and r.path == "/w/x.out"
    assert r.fetched_at == CLOCK_START and r.error is None and r.file_error is None and not r.partial
    small = parse_read(f"12|{MTIME}\nfirst\nlast\n", "", 4096, CLOCK_START, "/w/x.out")
    assert small.lines == ("first", "last") and small.live is None             # whole file: nothing dropped
    assert not small.started_mid_file and small.window_bytes == 12
    crlf = parse_read(f"9000|{MTIME}\ncut\r\na\r\nb\x1b[31mred\x1b[0m\n", "", 64, CLOCK_START, "/w/x.out")
    assert crlf.lines == ("a", "b\x1b[31mred\x1b[0m")                         # CRLF; ANSI left for the caller
    one = parse_read(f"9000|{MTIME}\n" + "x" * 64, "", 64, CLOCK_START, "/w/x.out")
    assert one.lines == () and one.live == "…" + "x" * 64                      # one huge line: marked as cut
    gone = parse_read("", "stat: cannot statx '/w/x.out': No such file or directory", 64, CLOCK_START, "/w/x.out")
    assert gone.file_error == "No such file" and gone.size is None and gone.lines == ()


async def test_tail_read_is_one_peek_probe() -> None:
    t = FakeTransport("k2")
    t.set_result("log.peek", f"30|{MTIME}\nTraceback\n  boom\nRuntimeError: x\n")
    svc, probe = make(t)
    r = await svc.read("k2", REF, nbytes=32768)
    assert r.lines == ("Traceback", "  boom", "RuntimeError: x") and r.live is None
    assert len(probe.scripts) == 1 and [s.name for s in probe.scripts[0].sections] == ["log.peek"]
    sec = probe.scripts[0].sections[0]
    assert sec.meta == {"path": REF.path, "bytes": 32768} and "tail -c 32768" in sec.cmd
    await svc.read("k2", REF, nbytes=32768)
    assert t.count("log") == 2                                       # never cached


async def test_head_read_uses_log_head() -> None:
    t = FakeTransport("k2")
    lines = "\n".join(f"h{i}" for i in range(HEAD_LINES + 50))
    t.set_result("log.head", f"999999|{MTIME}\n{lines}\npartial-unterminated")
    svc, probe = make(t)
    r = await svc.read("k2", REF, nbytes=8192, head=True)
    assert [s.name for s in probe.scripts[0].sections] == ["log.head"]
    assert r.lines == tuple(f"h{i}" for i in range(HEAD_LINES)) and r.live is None
    assert not r.started_mid_file and r.window_bytes == 8192 and r.size == 999999


async def test_transport_error_is_typed() -> None:
    t = FakeTransport("lynx")
    t.fail_next(ErrorKind.AUTH, summary="Permission denied (publickey,keyboard-interactive) via jump.lab")
    svc, _ = make(t)
    r = await svc.read("lynx", REF, nbytes=4096)
    assert r.error is not None and r.error.kind is ErrorKind.AUTH
    assert r.lines == () and r.file_error is None and not r.partial


async def test_partial_and_file_errors() -> None:
    t = FakeTransport("k2")
    t.set_result("log.peek", f"40|{MTIME}\nalpha\nbeta\nhalf", rc=124, outcome=SectionOutcome.TIMEOUT)
    svc, _ = make(t)
    r = await svc.read("k2", REF, nbytes=4096)
    assert r.partial and r.lines == ("alpha", "beta") and r.live == "half" and r.error is None
    t.set_result("log.peek", "", 1, "tail: cannot open '/w/x' for reading: Permission denied")
    denied = await svc.read("k2", REF, nbytes=4096)
    assert denied.file_error == "Permission denied" and not denied.partial and denied.lines == ()
    t.set_result("log.peek", "", rc=124, outcome=SectionOutcome.TIMEOUT)
    silent = await svc.read("k2", REF, nbytes=4096)            # nothing came: a typed timeout, not a file error
    assert silent.error is not None and silent.error.kind is ErrorKind.TIMEOUT and silent.error.section == "log.peek"
    assert silent.file_error is None and silent.lines == () and not silent.partial
    t.set_result("log.peek", "", rc=124, stderr="tail: cannot open '/w/x': No such file or directory",
                 outcome=SectionOutcome.TIMEOUT)
    assert (await svc.read("k2", REF, nbytes=4096)).file_error == "No such file"     # a known cause still wins


async def test_no_path_sends_nothing_and_a_crashing_probe_does_not_raise() -> None:
    t = FakeTransport("k2")
    svc, probe = make(t)
    r = await svc.read("k2", LogRef("stdout", None, pattern="%x_%j.out"), nbytes=4096)
    assert r.file_error == "log path unknown" and r.path == "" and not probe.scripts

    async def crash(host: str, script: ProbeScript) -> ProbeResult:
        raise RuntimeError("unknown host 'k3'")
    bad = LogService(ManualClock(CLOCK_START), transport_for=lambda h: t, stream_limit_for=lambda h: 1, probe=crash,
                     caps_for=lambda h: None)
    r = await bad.read("k3", REF, nbytes=4096)
    assert r.error is not None and r.error.kind is ErrorKind.INTERNAL


@pytest.mark.parametrize("head", [False, True])
async def test_demo_transport_serves_reads(head: bool, tmp_path: Path) -> None:
    from runfleet.core.service import FleetService
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld
    from runfleet.models import StateGroup

    clock = ManualClock(CLOCK_START)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    svc = FleetService(world.fleet_config(["k2"]), lambda h: DemoTransport(h, world), clock, None)
    svc.lock_path = False
    await svc.collect_once(["k2"], timeout=30)
    snap = svc.store.hosts["k2"].snapshot
    assert snap is not None
    job = next(j for j in snap.jobs.values() if j.state.group is StateGroup.ACTIVE
               and any(r.path and r.path in world.cluster("k2").logs for r in j.logs))
    ref = next(r for r in job.logs if r.path and r.path in world.cluster("k2").logs)
    r = await svc.logs.read("k2", ref, nbytes=8192, head=head)
    assert r.error is None and r.file_error is None and r.size is not None and (r.lines or r.live)


async def test_remote_oddities_never_raise() -> None:
    t = FakeTransport("k2")
    svc, probe = make(t)
    bad_path = await svc.read("k2", LogRef("stdout", "/w/a\nb.out"), nbytes=4096)
    assert bad_path.file_error and not probe.scripts                  # no probe for a path no shell can carry
    t.set_result("log.peek", f"40|{MTIME}\n", 1, "tail: cannot open '/w/x' for reading: Permission denied")
    denied = await svc.read("k2", REF, nbytes=4096)                   # stat fine, contents unreadable
    assert denied.partial and denied.file_error == "Permission denied" and denied.size == 40
    t.set_result("log.peek", "40|99999999999999999999\nx\n")
    assert (await svc.read("k2", REF, nbytes=4096)).file_error == "unreadable stat output"
    t.set_result("log.head", "40|99999999999999999999\nx\n")
    assert (await svc.read("k2", REF, nbytes=4096, head=True)).file_error == "unreadable stat output"
