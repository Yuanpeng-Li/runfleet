"""[WP1] LogService: line assembly with CR semantics, streaming sessions, peek cache, fetch, pagers (§4.7)."""

from __future__ import annotations

import asyncio
import stat
import tracemalloc
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runfleet.core.clock import ManualClock
from runfleet.core.logs import (MAX_LINE, LineAssembler, LogService, StreamSession, parse_peek, stream_note,
                                tail_spec)
from runfleet.core.service import LogUpdate
from runfleet.errors import RunFleetError
from runfleet.models import Capabilities, ErrorKind, LogRef
from runfleet.probe.section import ProbeScript
from runfleet.transport.base import ProbeResult
from runfleet.transport.fake import FakeStream, FakeTransport
from tests.conftest import CLOCK_START

CAPS = Capabilities(remote_user="me", uid=1, login_host="l", tz_offset=timedelta(0), clock_skew=timedelta(0),
                    tools=frozenset({"less"}))


# ---------------------------------------------------------------- line assembly
def test_newlines_crs_and_crlf() -> None:
    a = LineAssembler()
    assert a.feed(b"epoch 1\nloss 0.5\r\n 10%|#\r 20%|##") == ["epoch 1", "loss 0.5"]
    assert a.live == " 20%|##"
    assert a.feed(b"\r 30%|###\n") == [" 30%|###"]
    assert a.live is None


def test_cr_at_chunk_end_is_decided_by_the_next_chunk() -> None:
    a = LineAssembler()
    assert a.feed(b"windows line\r") == []
    assert a.feed(b"\nnext") == ["windows line"]                       # CRLF split across chunks
    b = LineAssembler()
    b.feed(b"progress 1\r")
    assert b.feed(b"progress 2\n") == ["progress 2"]                   # a real carriage return


def test_long_line_is_cut_with_marker() -> None:
    a = LineAssembler()
    out = a.feed(b"x" * (20 * 1024) + b"\nnext\n")
    assert out[0] == "x" * MAX_LINE + "…[+4096 bytes]"
    assert out[1] == "next"


def test_multibyte_utf8_split_across_chunks() -> None:
    a = LineAssembler()
    data = "损失 0.5 ✓\n".encode()
    out: list[str] = []
    for i in range(len(data)):
        out += a.feed(data[i:i + 1])
    assert out == ["损失 0.5 ✓"]
    assert LineAssembler().feed(b"\xff\xfebad\n") == ["��bad"]


def test_five_mb_cr_only_output_stays_bounded() -> None:
    a = LineAssembler()
    chunk = b"".join(f"\r{i % 100:3d}%|{'#' * 50}".encode() for i in range(1100))[:65536]
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    lines: list[str] = []
    for _ in range(80):                                                # ~5 MB, no newline at all
        lines += a.feed(chunk)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert lines == []
    assert a.live is not None and len(a.live) <= MAX_LINE
    assert peak < 2 * 1024 * 1024


def test_stream_notes() -> None:
    assert stream_note("tail: cannot open '/w/x.out' for reading: No such file or directory", "/w/x.out") == \
        "waiting for /w/x.out to appear"
    assert stream_note("tail: '/w/x.out' has appeared;  following new file", "/w/x.out") == "/w/x.out appeared"
    assert stream_note("tail: cannot open '/w/x': Permission denied", "/w/x") == "permission denied"
    assert stream_note("", "/w/x") is None


def test_tail_spec_quotes_and_validates() -> None:
    spec = tail_spec("/w/a b.out", 65536)
    assert spec.name == "log.tail" and spec.meta == {"path": "/w/a b.out", "bytes": 65536}
    assert spec.cmd == "tail -c 65536 -F -- '/w/a b.out' & p=$!; cat >/dev/null; kill $p 2>/dev/null"
    with pytest.raises(ValueError):
        tail_spec("relative.out", 10)


# ---------------------------------------------------------------- streaming sessions
async def collect(session: StreamSession, limit: int = 50) -> list[LogUpdate]:
    out = []
    async for u in session.updates():
        out.append(u)
        if u.eof or len(out) >= limit:
            break
    return out


async def test_session_batches_lines_and_live_line() -> None:
    chunks = [b"line1\nline2\n", b" 10%\r", b" 20%\r 30%", b"\ndone\n"]
    s = StreamSession("lynx", LogRef("stdout", "/w/x.out"), FakeStream(chunks), batch_interval=0.01)
    ups = await collect(s)
    lines = [ln for u in ups for ln in u.lines]
    assert lines == ["line1", "line2", " 30%", "done"]
    assert ups[-1].eof and ups[-1].note == "stream ended"
    await s.close()


async def test_session_five_mb_cr_only_stream() -> None:
    chunk = (b"\r" + b"#" * 1023) * 64                                  # 64 KiB, CR-only
    stream = FakeStream([chunk] * 80)
    s = StreamSession("lynx", LogRef("stdout", "/w/x.out"), stream, batch_interval=0.0)
    ups = await collect(s, limit=1000)
    assert all(u.lines == () for u in ups)
    assert ups[-1].eof and ups[0].live == "#" * 1023


async def test_session_waiting_note_and_close() -> None:
    stream = FakeStream([], stderr="tail: cannot open '/w/p.out' for reading: No such file or directory",
                        hold_open=True)
    s = StreamSession("k2", LogRef("stdout", "/w/p.out"), stream, batch_interval=0.0)
    agen = s.updates()
    first = await agen.__anext__()
    assert first.note == "waiting for /w/p.out to appear" and not first.eof
    await s.close()
    assert stream.closed


# ---------------------------------------------------------------- the service
class Probe:
    def __init__(self, t: FakeTransport) -> None:
        self.t = t

    async def __call__(self, host: str, script: ProbeScript) -> ProbeResult:
        return await self.t.probe(script, timeout=10)


def make_service(t: FakeTransport, clock: ManualClock, *, limit: int = 1, pager: str = "") -> LogService:
    return LogService(clock, transport_for=lambda h: t, stream_limit_for=lambda h: limit, probe=Probe(t),
                      caps_for=lambda h: CAPS, pager=pager)


async def test_stream_lane_closes_the_oldest() -> None:
    t = FakeTransport("k2")
    t.set_stream("log.tail", lambda spec: FakeStream([b"x\n"], hold_open=True))
    svc = make_service(t, ManualClock(CLOCK_START), limit=1)
    s1 = await svc.stream("k2", LogRef("stdout", "/w/1.out"))
    s2 = await svc.stream("k2", LogRef("stdout", "/w/2.out"))
    assert t.opened_streams[0].closed and not t.opened_streams[1].closed
    assert svc.open_streams("k2") == 1
    await s2.close()
    assert svc.open_streams("k2") == 0
    await s1.close()
    with pytest.raises(RunFleetError):
        await svc.stream("k2", LogRef("stdout", None, pattern="%x.out"))


async def test_peek_cache_rules() -> None:
    clock = ManualClock(CLOCK_START)
    t = FakeTransport("k2")
    t.set_result("log.peek", "120|1788696459\npartial\nl1\nl2\nl3")
    svc = make_service(t, clock)
    ref = LogRef("stdout", "/w/1.out", size=120, mtime=datetime.fromtimestamp(1788696459, tz=UTC))
    p = await svc.peek("k2", ref, nbytes=20)
    assert p.lines == ("l1", "l2", "l3") and p.size == 120                # first line cut by tail -c
    assert t.count("log") == 1
    await svc.peek("k2", ref)
    clock.advance(20)
    await svc.peek("k2", ref)                                          # old but logstat unchanged: cached
    assert t.count("log") == 1
    await svc.peek("k2", replace(ref, size=130))                       # changed and >= 15 s old: refetch
    assert t.count("log") == 2
    await svc.peek("k2", replace(ref, size=140))                       # changed but fresh: cached
    assert t.count("log") == 2
    nopath = await svc.peek("k2", LogRef("stdout", None, error="No such file"))
    assert nopath.error == "No such file"


def test_parse_peek_errors_and_tail() -> None:
    now = CLOCK_START
    err = parse_peek("", "stat: cannot statx '/w/x': No such file or directory", 10, now)
    assert err.error == "No such file" and err.size is None
    p = parse_peek("5|1788696459\nab\ncd", "", 4096, now)
    assert p.lines == ("ab", "cd")
    assert parse_peek("5|1788696459\nab\ncd\n", "", 4096, now).lines == ("ab", "cd")      # exact stdout, no blank
    assert parse_peek("9|1788696459\nab\n10%\r20%", "", 4096, now).lines == ("ab", "20%")  # live line last
    many = parse_peek("100|1\n" + "\n".join(f"l{i}" for i in range(20)), "", 4096, now)
    assert many.lines == tuple(f"l{i}" for i in range(12, 20))


async def test_fetch_writes_private_temp_file() -> None:
    t = FakeTransport("k2")
    t.set_result("log.fetch", "a\nb")
    svc = make_service(t, ManualClock(CLOCK_START))
    path = await svc.fetch("k2", LogRef("stdout", "/w/1.out"))
    try:
        assert path.read_text() == "a\nb\n"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        path.unlink()
    t.set_result("log.fetch", "", 1, "tail: cannot open '/w/1.out' for reading: No such file or directory")
    with pytest.raises(RunFleetError, match="No such file"):
        await svc.fetch("k2", LogRef("stdout", "/w/1.out"))
    t.fail_next(ErrorKind.TIMEOUT)
    with pytest.raises(RunFleetError):
        await svc.fetch("k2", LogRef("stdout", "/w/1.out"))


def test_pager_argvs(monkeypatch: pytest.MonkeyPatch) -> None:
    t = FakeTransport("k2")
    svc = make_service(t, ManualClock(CLOCK_START))
    assert svc.pager_argv("k2", LogRef("stdout", "/w/a b.out")) == ["fake-ssh", "-t", "k2",
                                                                     "less -R +G -- '/w/a b.out'"]
    assert svc.pager_argv("k2", LogRef("stdout", None)) is None
    nolss = LogService(ManualClock(CLOCK_START), transport_for=lambda h: t, stream_limit_for=lambda h: 1,
                       probe=Probe(t), caps_for=lambda h: None)
    assert nolss.pager_argv("k2", LogRef("stdout", "/w/x")) is None
    monkeypatch.setenv("PAGER", "most -s")
    assert svc.local_pager_argv(Path("/tmp/x.log")) == ["most", "-s", "/tmp/x.log"]
    monkeypatch.delenv("PAGER")
    assert svc.local_pager_argv(Path("/tmp/x.log")) == ["less", "-R", "+G", "/tmp/x.log"]
    assert make_service(t, ManualClock(CLOCK_START), pager="bat --paging=always").local_pager_argv()[:2] == \
        ["bat", "--paging=always"]
    assert svc.local_pager_argv()[-1].endswith("runfleet.log")


# ---------------------------------------------------------------- review regressions
class SlowCloseStream(FakeStream):
    """`ssh` that takes a while to exit after stdin closes (ProcessStream.close waits up to KILL_GRACE)."""

    def __init__(self) -> None:
        super().__init__([b"x\n"], hold_open=True)
        self.release = asyncio.Event()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        await self.release.wait()
        await super().close()


@pytest.mark.parametrize("limit", [1, 2])
async def test_stream_while_the_oldest_is_still_closing_does_not_spin(limit: int) -> None:
    t = FakeTransport("k2")
    slow = SlowCloseStream()
    t.set_stream("log.tail", lambda spec: slow if not t.opened_streams else FakeStream([], hold_open=True))
    svc = make_service(t, ManualClock(CLOCK_START), limit=limit)
    a = await svc.stream("k2", LogRef("stdout", "/w/a.out"))
    for i in range(limit - 1):
        await svc.stream("k2", LogRef("stdout", f"/w/b{i}.out"))
    closing = asyncio.ensure_future(a.close())           # the viewer's worker finally block
    await asyncio.sleep(0)
    opening = asyncio.ensure_future(svc.stream("k2", LogRef("stdout", "/w/c.out")))
    await asyncio.sleep(0.05)                            # the loop is alive (a spin would never get here)
    assert not opening.done()
    slow.release.set()
    await asyncio.wait_for(opening, 1.0)
    await closing
    assert svc.open_streams("k2") == limit
    assert slow.close_calls == 1


async def test_concurrent_closes_share_one_shutdown() -> None:
    slow = SlowCloseStream()
    s = StreamSession("k2", LogRef("stdout", "/w/x.out"), slow)
    first, second = asyncio.ensure_future(s.close()), asyncio.ensure_future(s.close())
    await asyncio.sleep(0)
    assert not second.done()                             # waits for the real shutdown, not a no-op
    slow.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 1.0)
    assert slow.close_calls == 1 and slow.closed


async def test_concurrent_opens_respect_the_stream_limit() -> None:
    t = FakeTransport("k2")
    t.set_stream("log.tail", lambda spec: FakeStream([], hold_open=True))
    svc = make_service(t, ManualClock(CLOCK_START), limit=1)
    await asyncio.gather(svc.stream("k2", LogRef("stdout", "/w/1.out")), svc.stream("k2", LogRef("stdout", "/w/2.out")))
    assert svc.open_streams("k2") == 1
    assert t.opened_streams[0].closed and not t.opened_streams[1].closed


async def test_stream_opened_while_the_host_is_removed_is_closed() -> None:
    t = FakeTransport("k2")
    gate = asyncio.Event()
    handle = FakeStream([], hold_open=True)

    async def slow_master(host: str) -> None:
        await gate.wait()

    t.set_stream("log.tail", lambda spec: handle)
    svc = LogService(ManualClock(CLOCK_START), transport_for=lambda h: t, stream_limit_for=lambda h: 1,
                     probe=Probe(t), caps_for=lambda h: CAPS, ensure_master=slow_master)
    opening = asyncio.ensure_future(svc.stream("k2", LogRef("stdout", "/w/1.out")))
    await asyncio.sleep(0)
    await svc.close_host("k2")
    gate.set()
    with pytest.raises(RunFleetError, match="host removed"):
        await opening
    assert handle.closed and svc.open_streams("k2") == 0


async def test_rotation_breaks_the_partial_line_and_repeats_the_note() -> None:
    stream = FakeStream([b"epoch 3 loss"], hold_open=True)
    s = StreamSession("k2", LogRef("stdout", "/w/x.out"), stream, batch_interval=0.0)
    agen = s.updates()
    first = await agen.__anext__()
    assert first.live == "epoch 3 loss"
    stream._stderr = "tail: /w/x.out: file truncated\n"
    rotated = await agen.__anext__()
    assert rotated.lines == ("epoch 3 loss",) and rotated.live is None
    assert rotated.note == "file was replaced or truncated"
    stream._stderr += "tail: /w/x.out: file truncated\n"
    again = await agen.__anext__()
    assert again.note == "file was replaced or truncated"         # a second rotation is reported too
    await s.close()


def test_peek_stats_the_symlink_target() -> None:
    from runfleet.core.logs import peek_section
    assert peek_section("/w/latest.log", 4096).cmd.startswith("stat -L -c ")


def test_parse_peek_cut_fragment_keeps_cr_only_progress() -> None:
    now = CLOCK_START
    cr = parse_peek("99999|1788696459\nepoch 1: 10%\repoch 1: 20%\repoch 1: 30%", "", 4096, now)
    assert cr.lines == ("epoch 1: 30%",)
    assert parse_peek("99999|1\nrtial\nA\nB", "", 4096, now).lines == ("A", "B")
    assert parse_peek("99999|1\nrtial\r\nA", "", 4096, now).lines == ("A",)
    assert parse_peek("99999|1\npartial\rX", "", 4096, now).lines == ("X",)
    assert parse_peek("99999|1\nno boundary at all", "", 4096, now).lines == ("…no boundary at all",)


def test_line_cap_is_a_byte_budget() -> None:
    a = LineAssembler(max_line=10)
    out = a.feed("损失损失x\n".encode())                              # 3 bytes per CJK character
    assert out == ["损失损…[+4 bytes]"]                                # 9 bytes kept, never half a character


async def test_unchanged_logstat_does_not_keep_a_peek_forever() -> None:
    clock = ManualClock(CLOCK_START)
    t = FakeTransport("k2")
    t.set_result("log.peek", "120|1788696459\nl1\n")
    svc = make_service(t, clock)
    ref = LogRef("stdout", "/w/1.out", size=120, mtime=datetime.fromtimestamp(1788696459, tz=UTC))
    await svc.peek("k2", ref)
    clock.advance(200)
    await svc.peek("k2", ref)
    assert t.count("log") == 1
    clock.advance(200)
    await svc.peek("k2", ref)
    assert t.count("log") == 2
