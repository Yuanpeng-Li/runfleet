"""[WP2] Shared test helpers: sample()/meta()/frame(), ManualClock fixture, no-ssh guard (DESIGN.md §10.1-10.2).

Helpers are plain functions (``from tests.conftest import sample``) and also fixtures of the same name.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet.core.clock import ManualClock

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "fixtures" / "samples"

# Meta files in the order that decides "last entry wins" (§10.2).
META_FILES = ("_meta.txt", "_meta_followup.txt", "_meta_followup2.txt")

DEFAULT_NONCE = "0123456789ab"

# Fixed start for ManualClock: 2026-09-07 10:42:37 UTC (field-probe day; no DST change between it and any fixture time, the next one is 2026-11-01).
CLOCK_START = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


# ---------------------------------------------------------------- fixtures from tests/fixtures/samples
def sample_path(host: str, name: str) -> Path:
    return SAMPLES_DIR / host / f"{name}.txt"


def sample(host: str, name: str) -> str:
    """Byte-exact content of tests/fixtures/samples/<host>/<name>.txt (no copies, no newline translation).

    Decoded like the transport decodes remote output (utf-8, errors="replace").
    """
    path = sample_path(host, name)
    if not path.is_file():
        raise FileNotFoundError(f"no sample {host}/{name} ({path})")
    return path.read_bytes().decode("utf-8", errors="replace")


_META_HEADER = re.compile(r"^## (?P<name>\S+)\s*$")
_META_RC = re.compile(r"^rc=(?P<rc>-?\d+)(?:\s+ms=(?P<ms>-?\d+))?\s*$")


def _parse_meta_file(text: str) -> dict[str, dict[str, Any]]:
    """Parse one _meta*.txt: blocks of `## name`, `cmd: ...`, `rc=N ms=M`, optional `stderr:` + lines.

    Blocks are separated by one blank line (probes/split.py); stderr keeps inner blank lines and loses the trailing
    newline(s) that belong to the separator.
    """
    entries: dict[str, dict[str, Any]] = {}
    name: str | None = None
    entry: dict[str, Any] = {}
    stderr_lines: list[str] | None = None

    def flush() -> None:
        if name is None:
            return
        entry["stderr"] = "\n".join(stderr_lines).rstrip("\n") if stderr_lines is not None else ""
        entries[name] = entry

    for line in text.splitlines():
        m = _META_HEADER.match(line)
        if m:
            flush()
            name, entry, stderr_lines = m.group("name"), {"cmd": None, "rc": None, "ms": None}, None
            continue
        if name is None:
            continue
        if stderr_lines is not None:
            stderr_lines.append(line)
        elif line.startswith("cmd: "):
            entry["cmd"] = line[len("cmd: "):]
        elif (r := _META_RC.match(line)) is not None:
            entry["rc"] = int(r.group("rc"))
            entry["ms"] = int(r.group("ms")) if r.group("ms") is not None else None
        elif line == "stderr:":
            stderr_lines = []
    flush()
    return entries


def meta(host: str, probe: str) -> dict[str, Any]:
    """{"cmd", "rc", "ms", "stderr"} of `probe` on `host` from the _meta*.txt files.

    When a probe appears in several meta files, the last entry in file order (_meta, _meta_followup,
    _meta_followup2) wins (§10.2).
    """
    found: dict[str, Any] | None = None
    for fname in META_FILES:
        path = SAMPLES_DIR / host / fname
        if path.is_file():
            entries = _parse_meta_file(path.read_bytes().decode("utf-8", errors="replace"))
            if probe in entries:
                found = entries[probe]
    if found is None:
        raise KeyError(f"no meta entry for {host}/{probe}")
    return dict(found)


def frame(sections: Mapping[str, tuple[str, int, str]], nonce: str = DEFAULT_NONCE, *, ms: int = 0) -> str:
    """Framed probe stdout as the §4.3 prelude prints it, for {name: (stdout, rc, stderr)}.

    `\\n` + `B` marker, raw stdout, stderr between `\\n` + `X`/`Y` markers when non-empty, then `\\n` + `E name rc ms`
    (stderr before the end marker, CORE-27). A section with rc 127 and stderr "missing:<tool>" is framed like the
    prelude's missing-tool branch (`E name 127 0 missing:<tool>`).
    """
    rf = f"@@RF{nonce}"
    out: list[str] = []
    for name, (stdout, rc, stderr) in sections.items():
        out.append(f"\n{rf} B {name}\n")
        if rc == 127 and stderr.startswith("missing:") and not stdout:
            out.append(f"\n{rf} E {name} 127 0 {stderr}\n")
            continue
        out.append(stdout)
        if stderr:
            out.append(f"\n{rf} X {name}\n{stderr}\n{rf} Y {name}\n")
        out.append(f"\n{rf} E {name} {rc} {ms}\n")
    return "".join(out)


@pytest.fixture(name="sample")
def _sample_fixture() -> Callable[[str, str], str]:
    return sample


@pytest.fixture(name="meta")
def _meta_fixture() -> Callable[[str, str], dict[str, Any]]:
    return meta


@pytest.fixture(name="frame")
def _frame_fixture() -> Callable[..., str]:
    return frame


@pytest.fixture
def manual_clock() -> ManualClock:
    """A ManualClock at CLOCK_START (aware UTC)."""
    return ManualClock(CLOCK_START)


# ---------------------------------------------------------------- `real` marker: opt-in only
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked `real` touch real hosts: skipped unless selected explicitly with `-m real`."""
    markexpr = config.getoption("markexpr", default="") or ""
    if re.search(r"(?<!not )\breal\b", markexpr):
        return
    skip = pytest.mark.skip(reason="touches real hosts; opt in with -m real (only on hosts you may probe)")
    for item in items:
        if item.get_closest_marker("real") is not None:
            item.add_marker(skip)


# ---------------------------------------------------------------- no-ssh guard
def _real_ssh_binaries() -> frozenset[str]:
    found: set[str] = set()
    for d in os.environ.get("PATH", os.defpath).split(os.pathsep):
        cand = Path(d or ".") / "ssh"
        if cand.is_file() and os.access(cand, os.X_OK):
            found.add(os.path.realpath(cand))
    return frozenset(found)


_REAL_SSH = _real_ssh_binaries()   # computed at import, before any test edits PATH


def _argv_of(args: Any, shell: bool) -> list[str]:
    if isinstance(args, (str, bytes, os.PathLike)):
        text = os.fsdecode(args)
        if shell:
            try:
                return shlex.split(text)
            except ValueError:
                return text.split()
        return [text]
    return [os.fsdecode(a) for a in args]


def _spawns_real_ssh(argv: list[str], executable: Any, env: Mapping[str, str] | None) -> bool:
    if not argv:
        return False
    prog = os.fsdecode(executable) if executable is not None else argv[0]
    if os.path.basename(prog) != "ssh":
        return False
    if "-G" in argv[1:]:
        return False                 # local config evaluation, no network (§10.1 exception)
    path = (env or os.environ).get("PATH", os.defpath)
    resolved = prog if os.sep in prog else shutil.which(prog, path=path)
    if resolved is None:
        return False
    return os.path.realpath(resolved) in _REAL_SSH   # fake `ssh` scripts on a test PATH are allowed


@pytest.fixture(autouse=True)
def no_real_ssh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test that spawns the system `ssh` unless it is marked `real` (`ssh -G` is allowed).

    asyncio subprocesses go through subprocess.Popen, so one patch covers both. The violation raises at the spawn
    site and is also reported at teardown, in case the code under test swallows the exception.
    """
    if request.node.get_closest_marker("real") is not None:
        yield
        return
    violations: list[list[str]] = []
    original_init = subprocess.Popen.__init__

    def guarded_init(self: subprocess.Popen[Any], args: Any, *a: Any, **kw: Any) -> None:
        argv = _argv_of(args, bool(kw.get("shell", False)))
        if _spawns_real_ssh(argv, kw.get("executable"), kw.get("env")):
            violations.append(argv)
            raise RuntimeError(f"test spawned real ssh (mark it `real` to allow): {argv!r}")
        original_init(self, args, *a, **kw)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)
    yield
    if violations:
        pytest.fail(f"test spawned real ssh {len(violations)}x (mark it `real` to allow): {violations[0]!r}")
