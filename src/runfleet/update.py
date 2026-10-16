"""[WP4] `runfleet update` and the update check (§8.6): how this copy was installed, the newest version on GitHub,
the commands that update it, and the relaunch after an update started from the TUI.

RunFleet is installed from GitHub main (no PyPI release), so "the newest version" is ``__version__`` in main's
``src/runfleet/__init__.py``, read over HTTPS from raw.githubusercontent.com (CHANGELOG.md too, for what changed).
Nothing else is sent: a plain GET with a ``runfleet/<version>`` User-Agent. The TUI checks at most once a day
(``general.update_check``; never in ``--demo``) and keeps the answer in ``<state>/update.json``.

The install method is read from the files the installers leave next to the interpreter: ``uv-receipt.toml``
(``uv tool install``), ``pipx_metadata.json`` (pipx); a package directory inside a git checkout is a source checkout.
Everything here is synchronous and quick (5 s timeouts); the TUI runs :func:`check` in a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from runfleet import __version__, paths
from runfleet.util import platform

log = logging.getLogger("runfleet.update")

REPO = "Yuanpeng-Li/runfleet"
GIT_URL = f"https://github.com/{REPO}"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/"
INSTALL_SH = RAW_URL + "install.sh"
CHECK_EVERY = timedelta(days=1)
FETCH_TIMEOUT = 5.0
MAX_BYTES = 1 << 20
RESTART_CODE = 75        # RunFleetApp's return code for "update, then start again" (never a process exit code)
_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)
_SECTION_RE = re.compile(r"^## \[([^\]]+)\]", re.M)
_LINK_RE = re.compile(r"^\[[^\]]+\]: https?://")

Kind = Literal["uv", "pipx", "source", "uvx", "pip"]


# ====================================================================================================== install
@dataclass(frozen=True)
class Install:
    kind: Kind
    source: str                                  # what it was installed from: git URL, directory, checkout, venv
    steps: tuple[tuple[str, ...], ...]           # commands that update it, run in order; () = only by hand
    manual: str                                  # the same as one line to run by hand (always set)
    note: str = ""                               # said before updating ("pinned to v0.2.1: …")
    env: Mapping[str, str] = field(default_factory=dict)   # extra environment of the steps

    def describe(self) -> str:
        how = {"uv": "uv tool", "pipx": "pipx", "source": "a git checkout", "uvx": "uvx (temporary)",
               "pip": "pip"}[self.kind]
        return f"installed with {how}" + (f" from {self.source}" if self.source and self.kind != "source" else
                                          f" ({self.source})" if self.source else "")


def find_uv(which: Callable[[str], str | None] = shutil.which, home: Path | None = None) -> str | None:
    """`uv` on PATH, else where its installer puts it (a shell started from a desktop may not have ~/.local/bin).
    Windows: `uv.exe` (`which` honours PATHEXT; the directory fallback names the file itself)."""
    found = which("uv")
    if found:
        return found
    home = home or Path.home()
    name = "uv.exe" if platform.IS_WINDOWS else "uv"
    dirs = [os.environ.get("UV_INSTALL_DIR"), os.environ.get("XDG_BIN_HOME"), str(home / ".local" / "bin"),
            str(home / ".cargo" / "bin")]
    for d in dirs:
        cand = Path(d) / name if d else None
        if cand is not None and cand.is_file() and (platform.IS_WINDOWS or os.access(cand, os.X_OK)):
            return str(cand)
    return None


def _git_pull(root: str, which: Callable[[str], str | None]) -> tuple[str, ...] | None:
    git = which("git")
    return (git, "-C", root, "pull", "--ff-only") if git else None


def _uv_install(prefix: Path, which: Callable[[str], str | None], home: Path | None) -> Install:
    try:
        tool = tomllib.loads((prefix / "uv-receipt.toml").read_text(encoding="utf-8")).get("tool", {})
        req = next((r for r in tool.get("requirements", []) if isinstance(r, dict) and r.get("name") == "runfleet"),
                   {})
        script = next((e.get("install-path") for e in tool.get("entrypoints", [])
                       if isinstance(e, dict) and e.get("name") == "runfleet"), None)
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        req, script = {}, None
    uv = find_uv(which, home)
    env = {"UV_TOOL_DIR": str(prefix.parent)}          # dirs chosen at install time may not be set in this shell
    if isinstance(script, str) and script:
        env["UV_TOOL_BIN_DIR"] = str(Path(script).parent)
    git = req.get("git")
    local = req.get("directory") or req.get("editable")
    if isinstance(git, str):
        url, _, query = git.partition("?")
        pin = next((v for k, _, v in (p.partition("=") for p in query.split("&")) if k in ("rev", "tag", "branch")),
                   "")
        cmd: tuple[str, ...]
        if pin:              # `uv tool upgrade` keeps the pin: reinstall from main instead
            cmd = ("tool", "install", "--force", f"git+{url}")
            note = f"pinned to {pin}; updating moves it to the newest main"
        else:
            cmd, note = ("tool", "upgrade", "runfleet"), ""
        return Install("uv", url + (f"@{pin}" if pin else ""), ((uv, *cmd),) if uv else (),
                       "uv " + " ".join(cmd), note, env)
    if isinstance(local, str):
        editable = "editable" in req
        pull = _git_pull(local, which) if (Path(local) / ".git").exists() else None
        steps: list[tuple[str, ...]] = [pull] if pull else []
        if not editable and uv:                         # --reinstall: uv caches the build of a directory by version
            steps.append((uv, "tool", "install", "--force", "--reinstall", local))
        manual = platform.chain([f"git -C {local} pull", *(() if editable else
                                                             (f"uv tool install --force --reinstall {local}",))])
        complete = bool(pull) and (editable or bool(uv))
        return Install("uv", local, tuple(steps) if complete else (), manual, "", env)
    return Install("uv", "PyPI", ((uv, "tool", "upgrade", "runfleet"),) if uv else (),
                   "uv tool upgrade runfleet", "", env)


def detect_install(prefix: Path | None = None, package_dir: Path | None = None, *,
                   which: Callable[[str], str | None] = shutil.which, home: Path | None = None,
                   executable: str | None = None) -> Install:
    prefix = Path(sys.prefix) if prefix is None else prefix
    package_dir = Path(__file__).resolve().parent if package_dir is None else package_dir
    executable = executable or sys.executable
    root = package_dir.parent.parent                    # <checkout>/src/runfleet
    if (root / "pyproject.toml").is_file() and (root / ".git").exists():
        pull = _git_pull(str(root), which)
        return Install("source", str(root), (pull,) if pull else (), f"git -C {root} pull")
    if (prefix / "uv-receipt.toml").is_file():
        return _uv_install(prefix, which, home)
    if (prefix / "pipx_metadata.json").is_file():
        pipx = which("pipx")
        return Install("pipx", "pipx", ((pipx, "reinstall", "runfleet"),) if pipx else (), "pipx reinstall runfleet")
    if any(part.startswith("archive-v") for part in prefix.parts):       # uvx: a cached, throw-away environment
        return Install("uvx", "", (), f"uvx --refresh --from git+{GIT_URL} runfleet")
    return Install("pip", str(prefix), (), f"{executable} -m pip install --upgrade git+{GIT_URL}")


# ====================================================================================================== versions
def parse_version(text: str) -> tuple[int, ...] | None:
    m = re.fullmatch(r"\s*v?(\d+)\.(\d+)\.(\d+)\s*", text)
    return tuple(int(g) for g in m.groups()) if m else None


def is_newer(latest: str, current: str) -> bool:
    a, b = parse_version(latest), parse_version(current)
    return a is not None and b is not None and a > b


def fetch_text(name: str, timeout: float = FETCH_TIMEOUT) -> str:
    """GET <main>/<name>; raises OSError (network, HTTP status) or ValueError (not text)."""
    req = urllib.request.Request(RAW_URL + name, headers={"User-Agent": f"runfleet/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (fixed https URL)
        return bytes(resp.read(MAX_BYTES)).decode("utf-8")


def fetch_latest(timeout: float = FETCH_TIMEOUT) -> str:
    m = _VERSION_RE.search(fetch_text("src/runfleet/__init__.py", timeout))
    if m is None or parse_version(m.group(1)) is None:
        raise ValueError("no version in src/runfleet/__init__.py on main")
    return m.group(1)


def changes_since(changelog: str, current: str, *, max_lines: int = 30) -> list[str]:
    """CHANGELOG.md sections newer than `current` (newest first), heading + body, at most `max_lines` lines."""
    out: list[str] = []
    marks = list(_SECTION_RE.finditer(changelog))
    for i, m in enumerate(marks):
        if not is_newer(m.group(1), current):
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(changelog)
        out += [line.rstrip() for line in changelog[m.start():end].strip().splitlines()
                if not _LINK_RE.match(line)]                             # the compare links at the end
        out.append("")
    while out and not out[-1]:
        out.pop()
    if len(out) > max_lines:
        out = out[:max_lines] + [f"… (more in {GIT_URL}/blob/main/CHANGELOG.md)"]
    return out


# ====================================================================================================== check
@dataclass(frozen=True)
class Status:
    current: str
    latest: str | None = None
    checked_at: datetime | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.latest is not None and is_newer(self.latest, self.current)


def state_file() -> Path:
    return paths.state_dir() / "update.json"


def _load(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: Mapping[str, object]) -> None:
    try:
        paths.ensure_private_dir(path.parent)
        paths.write_private(path, json.dumps(dict(data), indent=1, sort_keys=True))
    except OSError as exc:
        log.warning("cannot write %s: %s", path, exc)


def _when(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def check(now: datetime, *, force: bool = False, path: Path | None = None,
          fetch: Callable[[], str] = fetch_latest, current: str = __version__) -> Status:
    """The newest version: the saved answer while younger than CHECK_EVERY (and not `force`), else asked anew. A
    failed request keeps the last answer and says why; never raises."""
    path = path or state_file()
    data = _load(path)
    latest = data.get("latest") if isinstance(data.get("latest"), str) else None
    checked = _when(data.get("checked_at"))
    if not force and latest is not None and checked is not None and timedelta(0) <= now - checked < CHECK_EVERY:
        return Status(current, str(latest), checked)
    try:
        fresh = fetch()
    except Exception as exc:                 # network, TLS, HTTP status, a moved file: a message, never a crash
        log.info("update check failed: %s", exc)
        return Status(current, str(latest) if latest else None, checked, error=_short(exc))
    _save(path, {**data, "latest": fresh, "checked_at": now.isoformat()})
    return Status(current, fresh, now)


def should_notify(status: Status, *, path: Path | None = None) -> bool:
    """Announce each new version once (the About tab keeps showing it)."""
    if not status.available:
        return False
    return _load(path or state_file()).get("notified") != status.latest


def mark_notified(version: str, *, path: Path | None = None) -> None:
    path = path or state_file()
    _save(path, {**_load(path), "notified": version})


def _short(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)             # URLError wraps the socket / TLS error
    text = str(reason if reason is not None else exc) or type(exc).__name__
    return text if len(text) <= 80 else text[:79] + "…"


# ====================================================================================================== relaunch
def relaunch_argv(argv: Sequence[str], *, argv0: str | None = None, executable: str | None = None,
                  which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """The command that starts RunFleet again with the same arguments: the `runfleet` script this one was started
    as, else `python -m runfleet`."""
    argv0 = sys.argv[0] if argv0 is None else argv0
    executable = executable or sys.executable
    if argv0 and not argv0.endswith(("__main__.py", ".py")) and Path(argv0).name.startswith("runfleet"):
        if os.path.isfile(argv0) and os.access(argv0, os.X_OK):
            return [os.path.abspath(argv0), *argv]
        found = which("runfleet")
        if found:
            return [found, *argv]
    return [executable, "-m", "runfleet", *argv]
