"""[WP1] FakeTransport: scripted results, queues, responders, transport failures, streams."""

from __future__ import annotations

from runfleet.models import ErrorKind
from runfleet.probe import framing
from runfleet.probe.section import Section, SectionOutcome, StreamSpec
from runfleet.transport.fake import FakeStream, FakeTransport, result


async def test_lookup_order_and_recording() -> None:
    t = FakeTransport("k2")
    t.set_result("a.persist", "p")
    t.queue("a.persist", result("ignored-name", "once"))
    t.responder = lambda s: result(s.name, "resp") if s.name == "a.resp" else None
    s = framing.make_script([Section("a.persist", "x"), Section("a.resp", "x"), Section("a.none", "x")], "detail")
    r1 = await t.probe(s, timeout=5)
    assert r1.error is None
    assert (r1.sections["a.persist"].stdout, r1.sections["a.persist"].name) == ("once", "a.persist")
    assert r1.sections["a.resp"].stdout == "resp"
    assert r1.sections["a.none"].outcome is SectionOutcome.EMPTY
    r2 = await t.probe(s, timeout=5)
    assert r2.sections["a.persist"].stdout == "p"
    assert t.count() == 2 and t.count("detail") == 2 and t.probes[0].names == ("a.persist", "a.resp", "a.none")


async def test_transport_failures_and_outcome_derivation() -> None:
    t = FakeTransport("k2")
    t.fail_next(ErrorKind.AUTH, times=2)
    s = framing.make_script([Section("a.b", "x")])
    for _ in range(2):
        r = await t.probe(s, timeout=5)
        assert r.error is not None and r.error.kind is ErrorKind.AUTH
        assert r.sections["a.b"].outcome is SectionOutcome.NOT_RUN
    assert (await t.probe(s, timeout=5)).error is None
    assert result("x", "", 0, "err").outcome is SectionOutcome.STDERR_ONLY
    assert result("x", "", 127, "missing:sacct").outcome is SectionOutcome.MISSING
    assert result("x", "", 124).outcome is SectionOutcome.TIMEOUT


async def test_streams() -> None:
    t = FakeTransport("k2")
    t.set_stream("log.tail", lambda spec: FakeStream([b"a", b"b"]))
    h = await t.stream(StreamSpec("log.tail", "tail"))
    assert [c async for c in h.chunks()] == [b"a", b"b"]
    assert h.returncode == 0
    held = FakeStream([], hold_open=True)
    assert held.returncode is None
    await held.close()
    assert held.returncode == 0 and [c async for c in held.chunks()] == []
    assert t.connect_argv() == ["fake-ssh", "-fN", "k2"] and t.mux == "none" and t.ssh_log is None
