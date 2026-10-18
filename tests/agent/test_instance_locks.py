"""[WP4] The instance-lock helpers of core.service (§5.4) shared by FleetService, `snapshot` and `agent --refresh`,
and `snapshot` taking the free locks of the hosts it polls (never one another RunFleet holds; none in --demo)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli
from runfleet.core import service as svc_mod
from runfleet.core.service import (default_lock_base, instance_lock_file, lock_attempt_at, lock_holder,
                                   release_instance_lock, try_instance_lock, try_instance_lock_at)
from tests.cli.fakes import FakeService, host_state
from tests.cli.test_cli import Harness, _drop_log_handlers, run


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Harness]:
    _drop_log_handlers()                    # as tests/cli: each test logs to its own file
    yield Harness(monkeypatch, tmp_path)
    _drop_log_handlers()


def test_lock_file_is_per_destination(tmp_path: Path) -> None:
    base = tmp_path / "instance.lock"
    a = instance_lock_file(base, "alice@k2.example")
    assert a.parent == tmp_path and a.name.startswith("instance.lock.alice_k2.example-")
    assert instance_lock_file(base, "alice@k2.example") == a
    assert instance_lock_file(base, "vega") != a
    long = instance_lock_file(base, "x" * 300)
    assert len(long.name) == len("instance.lock.") + 40 + 1 + 8


def test_second_descriptor_sees_the_holder_and_lock_holder_never_takes_it(tmp_path: Path) -> None:
    path = instance_lock_file(tmp_path / "instance.lock", "k2")
    assert lock_holder(path) is None and not path.exists()             # no file: free, and none created
    fd, other = try_instance_lock(path)
    assert fd is not None and other is None
    try:
        again, holder = try_instance_lock(path)                       # a second descriptor, same process
        assert again is None and holder == str(os.getpid())
        assert lock_holder(path) == str(os.getpid())
        assert path.read_text() == str(os.getpid())
    finally:
        release_instance_lock(fd)
    assert lock_holder(path) is None                                   # released at once (unlock, then close)
    fd2, _ = try_instance_lock(path)
    assert fd2 is not None
    release_instance_lock(fd2)
    release_instance_lock(fd2)                                         # closed already: never raises


def test_taking_the_lock_tells_when_it_was_last_taken_before(tmp_path: Path) -> None:
    """C3: the previous attempt time, read under the lock before our pid moves the mtime: `agent --refresh`
    re-checks its rate limit on it (another refresh may have come and gone since it looked)."""
    path = instance_lock_file(tmp_path / "instance.lock", "k2")
    fd, other, previous = try_instance_lock_at(path)
    assert fd is not None and other is None and previous is None       # created just now: no attempt before
    release_instance_lock(fd)
    os.utime(path, (1_700_000_000, 1_700_000_000))                     # that attempt, dated
    fd, _, previous = try_instance_lock_at(path)
    assert fd is not None and previous == 1_700_000_000
    try:
        at = lock_attempt_at(path)
        assert at is not None and at > 1_700_000_000                   # this attempt dates the file now
        assert try_instance_lock_at(path) == (None, str(os.getpid()), None)      # held: no time either
    finally:
        release_instance_lock(fd)
    path.write_bytes(b"")                                              # a pid never written (crash): no attempt
    fd, _, previous = try_instance_lock_at(path)
    assert fd is not None and previous is None
    release_instance_lock(fd)


def test_unopenable_lock_is_no_lock(tmp_path: Path) -> None:
    missing_dir = tmp_path / "nope" / "instance.lock.k2"
    assert try_instance_lock(missing_dir) == (None, None)
    assert try_instance_lock_at(missing_dir) == (None, None, None)
    assert lock_holder(missing_dir) is None


class _OsWithoutWrite:
    """`os` for core.service whose write() fails (a full disk, a read-only remount after the open)."""

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    @staticmethod
    def write(fd: int, data: bytes) -> int:
        raise OSError(28, "No space left on device")


def test_a_pid_that_cannot_be_written_lets_the_lock_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """K3: the lock was taken, then writing our pid raised: the lock stayed held (and the fd leaked) until exit."""
    path = instance_lock_file(tmp_path / "instance.lock", "k2")
    monkeypatch.setattr(svc_mod, "os", _OsWithoutWrite())
    assert try_instance_lock_at(path) == (None, None, None)           # as an unopenable file: carry on unlocked
    assert try_instance_lock(path) == (None, None)
    monkeypatch.undo()
    assert lock_holder(path) is None                                   # let go, not leaked
    fd, other = try_instance_lock(path)
    assert fd is not None and other is None
    release_instance_lock(fd)


def test_an_empty_lock_file_is_no_attempt(tmp_path: Path) -> None:
    """K4: lock_attempt_at agrees with try_instance_lock_at: a file no pid was written into dates nothing."""
    path = instance_lock_file(tmp_path / "instance.lock", "k2")
    assert lock_attempt_at(path) is None                               # no file
    path.write_bytes(b"")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert lock_attempt_at(path) is None
    path.write_text("4242")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert lock_attempt_at(path) == 1_700_000_000


def test_a_lock_taken_without_the_pid_leaves_the_file_untouched_until_claimed(tmp_path: Path) -> None:
    """K5: `agent --refresh` takes the lock, re-checks the previous attempt and only then writes its pid; letting go
    before must not date an attempt it never made."""
    path = instance_lock_file(tmp_path / "instance.lock", "k2")
    path.write_text("4242")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    fd, other, previous = try_instance_lock_at(path, write_pid=False)
    assert fd is not None and other is None and previous == 1_700_000_000
    release_instance_lock(fd)
    assert path.read_text() == "4242" and lock_attempt_at(path) == 1_700_000_000
    fd, _, _ = try_instance_lock_at(path, write_pid=False)
    assert fd is not None
    try:
        assert svc_mod.claim_instance_lock(fd)
        assert path.read_text() == str(os.getpid())
        at = lock_attempt_at(path)
        assert at is not None and at > 1_700_000_000
    finally:
        release_instance_lock(fd)


def test_default_lock_base_is_in_the_private_mux_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet import paths

    monkeypatch.setattr(paths, "mux_dir", lambda: tmp_path / "cm")
    assert default_lock_base() == tmp_path / "cm" / "instance.lock" and (tmp_path / "cm").is_dir()
    (tmp_path / "file").write_text("x")
    monkeypatch.setattr(paths, "mux_dir", lambda: tmp_path / "file" / "cm")
    assert default_lock_base() is None                                 # cannot be made: no locking, no crash


def test_default_locations_are_isolated_per_test(isolated_user_dirs: Path) -> None:
    """tests/conftest.py's autouse fixture: no test may reach the developer's real state, mux dir or locks."""
    from runfleet import paths

    root = isolated_user_dirs
    for path in (paths.config_file(), paths.cache_file(), paths.log_file(), paths.ui_state_file(), paths.debug_dir(),
                 paths.ssh_log("k2"), paths.cache_dir(), Path.home()):
        assert path.is_relative_to(root), path
    runtime = Path(os.environ["XDG_RUNTIME_DIR"])
    assert not runtime.is_relative_to(Path.home()) and runtime.is_dir()
    if not paths.platform.IS_WINDOWS:          # never the shared /tmp/runfleet-<uid> fallback
        assert paths.mux_dir() == runtime / "runfleet"
    base = default_lock_base()
    assert base is not None and base.parent == paths.mux_dir()


# ---------------------------------------------------------------- snapshot
def _config(harness: Harness) -> None:
    harness.config_path.write_text('[[host]]\nname = "k2"\n\n[[host]]\nname = "vega"\n', encoding="utf-8")


def test_snapshot_takes_free_locks_and_leaves_held_ones(harness: Harness, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    _config(harness)
    base = tmp_path / "cm" / "instance.lock"
    base.parent.mkdir()
    monkeypatch.setattr(cli, "default_lock_base", lambda: base)
    held, _ = try_instance_lock(instance_lock_file(base, "vega"))     # "another RunFleet" polls vega
    assert held is not None
    seen: dict[str, str | None] = {}

    def script(svc: FakeService) -> None:
        svc.collected = {"k2": host_state("k2"), "vega": host_state("vega")}
        real = svc.collect_once

        async def collect(hosts: Any = None, *, timeout: float = 60) -> None:
            for name in ("k2", "vega"):
                seen[name] = lock_holder(instance_lock_file(base, name))
            await real(hosts, timeout=timeout)
        svc.collect_once = collect  # type: ignore[method-assign]
    harness.script = script
    try:
        rc, out, _ = run(["snapshot", "--json", "k2", "vega"], capsys)
        assert rc == 0 and '"runfleet.snapshot/1"' in out                # output and exit code as before
        assert seen == {"k2": str(os.getpid()), "vega": str(os.getpid())}   # k2 ours while polling; vega theirs
        assert lock_holder(instance_lock_file(base, "k2")) is None          # released afterwards
        assert lock_holder(instance_lock_file(base, "vega")) == str(os.getpid())   # never touched
    finally:
        release_instance_lock(held)


def test_snapshot_demo_takes_no_lock(harness: Harness, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture[str]) -> None:
    def boom(*a: object) -> Any:
        raise AssertionError("--demo must not touch the real instance locks")
    monkeypatch.setattr(cli, "default_lock_base", boom)
    monkeypatch.setattr(svc_mod, "try_instance_lock", boom)
    harness.script = lambda svc: svc.collected.update({"k2": host_state("k2")})
    rc, _, _ = run(["snapshot", "--demo", "k2"], capsys)
    assert rc == 0
