"""[WP2] Shared test helpers: sample()/meta()/frame(), ManualClock fixture, no-ssh guard (DESIGN.md §10.1-10.2).

Helpers are plain functions (``from tests.conftest import sample``) and also fixtures of the same name. Tests marked
``posix`` (sh scripts, process groups, file modes, ptys, /proc) are skipped on native Windows; ``is_root()`` replaces
``os.geteuid() == 0`` in skip conditions (no euid on Windows).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runfleet.core.clock import ManualClock

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "fixtures" / "samples"

# Meta files in the order that decides "last entry wins" (§10.2).
META_FILES = ("_meta.txt", "_meta_followup.txt", "_meta_followup2.txt", "_meta_account.txt")

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
    _meta_followup2, _meta_account) wins (§10.2).
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


WINDOWS = sys.platform == "win32"


def is_root() -> bool:
    """Running as root (permission tests cannot fail then); never on Windows, which has no euid."""
    return not WINDOWS and getattr(os, "geteuid")() == 0


# ---------------------------------------------------------------- `real` marker: opt-in only; `posix`: not on Windows
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked `real` touch real hosts: skipped unless selected explicitly with `-m real`. Tests marked `posix`
    need a POSIX system and are skipped on native Windows."""
    if WINDOWS:
        posix = pytest.mark.skip(reason="POSIX only (sh, process groups, file modes, ptys); not on native Windows")
        for item in items:
            if item.get_closest_marker("posix") is not None:
                item.add_marker(posix)
    markexpr = config.getoption("markexpr", default="") or ""
    if re.search(r"(?<!not )\breal\b", markexpr):
        return
    skip = pytest.mark.skip(reason="touches real hosts; opt in with -m real (only on hosts you may probe)")
    for item in items:
        if item.get_closest_marker("real") is not None:
            item.add_marker(skip)


# ---------------------------------------------------------------- isolated user dirs
# Every place RunFleet could read or write user state: XDG dirs (Linux, BSD), HOME (macOS ~/Library, ~/.ssh,
# ~/.claude skill installs), USERPROFILE (Path.home() on Windows) and platformdirs' Windows overrides.
ISOLATED_ENV = ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
                "XDG_RUNTIME_DIR", "WIN_PD_OVERRIDE_APPDATA", "WIN_PD_OVERRIDE_LOCAL_APPDATA")
# Where coding agents keep their config and skills (`runfleet skill` reads them): unset, so the defaults under the
# isolated HOME apply and no test can reach the developer's real ~/.claude, $CODEX_HOME, … through them.
AGENT_ENV = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "PI_CODING_AGENT_DIR", "COPILOT_HOME", "OPENCODE_CONFIG_DIR",
             "OPENCODE_CONFIG", "GEMINI_CLI_HOME", "GOOSE_PATH_ROOT", "AMP_SETTINGS_FILE", "FACTORY_HOME")


def _short_tmp_dir() -> Path:
    """A fresh private dir with a short path: the mux dir is ``$XDG_RUNTIME_DIR/runfleet`` only while a socket path
    there fits 86 bytes, else the *shared* ``/tmp/runfleet-<uid>``, and pytest's tmp_path is often too long."""
    base = "/tmp" if not WINDOWS and os.path.isdir("/tmp") else None
    return Path(tempfile.mkdtemp(prefix="rft-", dir=base))


@pytest.fixture(autouse=True)
def isolated_user_dirs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point HOME and every XDG / platformdirs location at fresh per-test dirs, so no test (nor a subprocess it
    starts) ever reads or writes the developer's real config, cache, log, ui.json, mux sockets, instance locks or agent
    skill dirs (the AGENT_ENV variables are unset).

    ``runfleet.paths`` computes every location at call time from the environment (nothing is cached at import), so
    the environment is enough; tests that check path rules still override these variables themselves. Yields the
    root of the isolated tree. Its own MonkeyPatch (not the ``monkeypatch`` fixture): this autouse fixture is set up
    first and so torn down last, after a test's own patches (say of ``os.lstat``) are undone, before ``rmtree``.
    """
    root = tmp_path_factory.mktemp("userdirs")
    runtime = _short_tmp_dir()
    values = {
        "HOME": root / "home", "USERPROFILE": root / "home",
        "XDG_CONFIG_HOME": root / "config", "XDG_STATE_HOME": root / "state", "XDG_CACHE_HOME": root / "cache",
        "XDG_DATA_HOME": root / "data", "XDG_RUNTIME_DIR": runtime,
        "WIN_PD_OVERRIDE_APPDATA": root / "appdata", "WIN_PD_OVERRIDE_LOCAL_APPDATA": root / "localappdata",
    }
    assert set(values) == set(ISOLATED_ENV)
    try:
        with pytest.MonkeyPatch.context() as mp:
            for name, path in values.items():
                path.mkdir(parents=True, exist_ok=True)
                mp.setenv(name, str(path))
            for name in AGENT_ENV:
                mp.delenv(name, raising=False)
            yield root
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


# ---------------------------------------------------------------- no-ssh guard
def _real_ssh_binaries() -> frozenset[str]:
    found: set[str] = set()
    for d in os.environ.get("PATH", os.defpath).split(os.pathsep):
        for name in ("ssh", "ssh.exe"):
            cand = Path(d or ".") / name
            if cand.is_file() and os.access(cand, os.X_OK):
                found.add(os.path.normcase(os.path.realpath(cand)))
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
    if len(argv) > 4 and argv[3] == "runfleet-login" and executable is None:   # platform.login_argv's sh wrapper
        return _spawns_real_ssh(argv[4:], None, env)
    prog = os.fsdecode(executable) if executable is not None else argv[0]
    if _ssh_stem(prog) != "ssh":                 # `ssh`, `ssh.exe`, `C:\…\OpenSSH\SSH.EXE`
        return False
    if "-G" in argv[1:]:
        return False                 # local config evaluation, no network (§10.1 exception)
    path = (env or os.environ).get("PATH", os.defpath)
    resolved = prog if os.sep in prog else shutil.which(prog, path=path)
    if resolved is None:
        return False
    return os.path.normcase(os.path.realpath(resolved)) in _REAL_SSH   # fake `ssh` scripts on a test PATH are allowed


def _ssh_stem(prog: str) -> str:
    """The program name without directory and `.exe`, case-folded (Windows paths use either slash)."""
    name = prog.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.removesuffix(".exe")


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
