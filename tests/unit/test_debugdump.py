"""--debug raw probe dumps (M3 F3 / leftover #3): wire-format files per probe, rotated per host, never raising."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from runfleet import debugdump
from runfleet.config import HostConfig
from runfleet.probe import framing
from runfleet.probe.section import Section
from runfleet.transport.local import LocalTransport


@pytest.fixture
def dumps(tmp_path: Path) -> Iterator[Path]:
    debugdump.enable(tmp_path / "debug")
    yield tmp_path / "debug"
    debugdump.disable()


@pytest.mark.posix                # LocalTransport runs sh
async def test_local_probe_is_dumped_in_wire_format(dumps: Path) -> None:
    t = LocalTransport(HostConfig("atlas", "local"))
    script = framing.make_script([Section("a.one", "echo hello"), Section("a.two", "exit 3")], "poll")
    res = await t.probe(script, timeout=10)
    assert res.error is None
    (inp,) = sorted((dumps / "atlas").glob("*.in"))
    assert inp.name.endswith("-poll.in") and "echo hello" in inp.read_text()
    out = inp.with_suffix(".out").read_text()
    assert f"@@RF{script.nonce} B a.one" in out and "hello" in out           # replayable as a fixture
    meta = inp.with_suffix(".meta").read_text()
    assert "atlas: probe poll 2 sections" in meta and "sections: a.one a.two" in meta


def test_rotation_and_caps(dumps: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(debugdump, "MAX_PROBES_PER_HOST", 3)
    monkeypatch.setattr(debugdump, "MAX_BYTES", 10)
    for i in range(5):
        assert debugdump.dump_probe("k2/../x", "poll", f"script {i}", b"x" * 50, b"", "meta") is not None
    d = dumps / "k2_.._x"                                                   # host names cannot escape the dir
    assert len(list(d.glob("*.in"))) == 3 and len(list(d.iterdir())) == 12
    newest = sorted(d.glob("*.out"))[-1].read_bytes()
    assert newest.startswith(b"x" * 10) and b"40 more bytes not dumped" in newest


def test_disabled_and_unwritable(tmp_path: Path) -> None:
    debugdump.disable()
    assert debugdump.dump_probe("h", "poll", "", b"", b"", "") is None
    blocker = tmp_path / "file"
    blocker.write_text("")
    debugdump.enable(blocker)                                               # a file, not a directory: never raises
    try:
        assert debugdump.dump_probe("h", "poll", "", b"", b"", "") is None
    finally:
        debugdump.disable()


@pytest.mark.posix                # umask and file modes
def test_dumps_and_logs_are_private_whatever_the_umask(dumps: Path, tmp_path: Path) -> None:
    """m3: with umask 002 the debug top dir, runfleet.log and the dumps were group-readable."""
    import logging
    import os
    import stat

    from runfleet.cli import PrivateRotatingFileHandler

    old = os.umask(0o002)
    try:
        assert debugdump.dump_probe("k2", "poll", "echo", b"out", b"err", "rc=0") is not None
        handler = PrivateRotatingFileHandler(tmp_path / "runfleet.log", maxBytes=64, backupCount=1)
        rec = logging.LogRecord("runfleet", logging.INFO, __file__, 1, "x" * 100, None, None)
        handler.emit(rec)
        handler.emit(rec)                                            # rolls over: the new file is private too
        handler.close()
    finally:
        os.umask(old)
    assert stat.S_IMODE(dumps.stat().st_mode) == 0o700 and stat.S_IMODE((dumps / "k2").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in (dumps / "k2").iterdir())
    logs = list(tmp_path.glob("runfleet.log*"))
    assert len(logs) == 2 and all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in logs)


def test_dot_host_names_stay_inside_the_dump_dir(dumps: Path) -> None:
    for name in (".", ".."):
        inp = debugdump.dump_probe(name, "poll", "s", b"", b"", "m")
        assert inp is not None and inp.parent.parent == dumps and inp.parent.name == "_" * len(name)
    assert not list(dumps.parent.glob("*.in"))
