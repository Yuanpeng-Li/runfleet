"""[WP4] The `runfleet` agent skill on disk: `runfleet skill install|update|uninstall|status` and the cheap refresh check
every invocation makes (docs/AGENT_INTERFACE.md §6).

One real copy of the packaged skill (agent/skill/: SKILL.md rendered with this machine's host names as trigger words,
references, Codex metadata) lives in <home>/.agents/skills/runfleet, which Codex, Pi, Copilot CLI, OpenCode, Gemini CLI,
Cursor, Amp, Goose, Factory Droid, Windsurf, Cline and Warp read directly. Agents that do not read it get a link (POSIX:
relative symlink; Windows: junction, then symlink, then a tracked copy): Claude Code always, Copilot CLI only with
$COPILOT_HOME, Antigravity only when asked. A private manifest (<state_dir>/skill.json) lists every target with the
hashes of the files written, so updates and removal touch only what RunFleet wrote; a file the user edited or added is
copied to <state_dir>/skill-backup/<UTC time>/ first, and a foreign directory is only ever moved there (--force).
Local files only; never edits any agent's settings."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import runfleet
from runfleet import paths
from runfleet.errors import RunFleetError
from runfleet.util import platform

log = logging.getLogger("runfleet.skill")
_DIAG = threading.local()                    # auto_refresh collects the diagnostics below into its one log line


def diagnostic(level: int, msg: str, *args: object) -> None:
    """A diagnostic of the file work: logged, or collected while auto_refresh runs (it logs exactly one line)."""
    sink: list[str] | None = getattr(_DIAG, "sink", None)
    if sink is None:
        log.log(level, msg, *args)
    else:
        sink.append(msg % args if args else msg)

SKILL_NAME = "runfleet"
STAMP_FILE = ".runfleet-skill.json"          # in every copy: {"version", "bundle_sha256", "install_id", "files"}
LEGACY_STAMP = ".runfleet-version"           # 0.8 pre-release installs: adopted
STAMPS = (STAMP_FILE, LEGACY_STAMP)
MANIFEST_FILE = "skill.json"
LOCK_FILE = "skill.lock"
BACKUP_DIR = "skill-backup"
SCHEMA = 1
RETRY = "retry"                              # manifest bundle_sha256 after a copy could not be refreshed
STAGING = ".runfleet-staging-"
# temporary names next to a target (prefix + random hex), swept on every install / update / uninstall
_LEFTOVER = re.compile(r"\.runfleet(?:-staging-|\.old-|\.tmp-)[0-9a-f]{8,32}")
MAX_DESCRIPTION = 1023                       # skill loaders reject descriptions of 1025+ characters; keep a margin
MAX_HOST_PREFIX = 200                        # host names prefixed to the description, at most
LOCK_WAIT = 30.0                             # install / update / uninstall wait this long for another one
BACKOFF_SECONDS = 3600.0                     # a failed automatic refresh is tried again at most this often
AUTO_OFF_NOTE = "automatic refresh is off; run `runfleet skill update`"
PERMISSION_RULE = "Bash(runfleet agent *)"


def _permission_hint(settings: str) -> str:
    return (f'Claude Code: add "{PERMISSION_RULE}" to permissions.allow in {settings} (or use /permissions) '
            "so the read-only agent verbs run without a prompt; `runfleet wait` and `runfleet status` still ask. "
            "Other agents: approve `runfleet agent` when asked, or allow that command prefix in their approval rules. "
            "RunFleet never edits these settings itself.")


def _follow_ups(settings: str, codex_config: str) -> tuple[str, ...]:
    return ("MCP server (if you added it): claude mcp remove runfleet -s user",
            f"Codex MCP (if you added it): delete the [mcp_servers.runfleet] block from {codex_config}",
            f'Claude Code permission (if you added it): remove "{PERMISSION_RULE}" from permissions.allow in '
            f"{settings}")


PERMISSION_HINT = _permission_hint("~/.claude/settings.json")
FOLLOW_UPS = _follow_ups("~/.claude/settings.json", "~/.codex/config.toml")

_DESC_KEY = "description:"
_WORD = re.compile(r"[\w+@-]+(?:\.[\w+@-]+)*")             # words; dots only inside (k2.lab, not "GPU.")
_YAML_UNSAFE = re.compile(r"[\s<>:#\"'`\\\[\]{},&*!|%]")     # a name that could break the plain YAML scalar
_YAML_LEAD = tuple("-?@")
_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[.-]?(dev|a|b|rc|post)(\d*))?(?:\+.*)?")
_PHASE = {"dev": 0, "a": 1, "b": 2, "rc": 3, None: 4, "post": 5}


class SkillInstallError(RunFleetError):
    """`skill install|update|uninstall` refused (a directory or link RunFleet did not write is in the way and --force
    was not given; another one is running) or could not write; the message says which path and what to do."""


# ---------------------------------------------------------------- environment and agent registry
@dataclass(frozen=True)
class Env:
    """Where agents look: the real home (never $CODEX_HOME & co.), the environment, the platform, PATH lookup."""

    home: Path
    environ: Mapping[str, str]
    is_windows: bool
    which: Callable[[str], str | None]

    @classmethod
    def current(cls) -> Env:
        return cls(home=Path.home(), environ=os.environ, is_windows=platform.IS_WINDOWS, which=shutil.which)

    def path_var(self, name: str) -> Path | None:
        """$name as a path (`~` is this Env's home); None when unset or empty."""
        value = self.environ.get(name, "").strip()
        if not value:
            return None
        if value == "~":
            return self.home
        if value.startswith(("~/", "~\\")):
            return self.home / value[2:]
        return Path(value)

    def config_home(self) -> Path:
        return self.path_var("XDG_CONFIG_HOME") or self.home / ".config"


@dataclass(frozen=True)
class AgentSpec:
    id: str
    label: str
    reads_shared: bool                                   # reads <home>/.agents/skills (when link_path gives None)
    link_path: Callable[[Env], Path | None]              # where it needs a link to the shared copy; None: no link
    detect: Callable[[Env], bool]
    legacy_dirs: Callable[[Env], tuple[Path, ...]]       # skills roots where a stray copy could shadow ours
    explicit_only: bool = False                          # never chosen by auto / all
    # legacy dirs it no longer reads because $home_var moved its home: a link to ours there is stale, not a duplicate
    unread_dirs: Callable[[Env], tuple[Path, ...]] = lambda e: ()
    home_var: str | None = None


def canonical_dir(env: Env) -> Path:
    return env.home / ".agents" / "skills" / SKILL_NAME


def _claude_root(env: Env) -> Path:
    return env.path_var("CLAUDE_CONFIG_DIR") or env.home / ".claude"


def _codex_root(env: Env) -> Path:
    return env.path_var("CODEX_HOME") or env.home / ".codex"


def permission_hint(env: Env) -> str:
    """Where to allow the agent verbs, with the settings file Claude Code reads here ($CLAUDE_CONFIG_DIR)."""
    return _permission_hint(_show(env, _claude_root(env) / "settings.json"))


def follow_ups(env: Env) -> tuple[str, ...]:
    """What `uninstall` never does, with the files the agents read here ($CLAUDE_CONFIG_DIR, $CODEX_HOME)."""
    return _follow_ups(_show(env, _claude_root(env) / "settings.json"), _show(env, _codex_root(env) / "config.toml"))


def _program_is(env: Env, program: str, marker: str) -> bool:
    """`program` is on PATH and its resolved path names `marker` (a bare `pi` could be any program)."""
    found = env.which(program)
    if not found:
        return False
    try:
        return marker in os.path.realpath(found).lower()
    except (OSError, ValueError):
        return False


def _found(env: Env, paths: Sequence[Path], programs: Sequence[str] = (), variables: Sequence[str] = ()) -> bool:
    if any(env.path_var(v) is not None for v in variables):
        return True
    try:
        if any(p.exists() for p in paths):
            return True
    except OSError:
        pass
    return any(env.which(p) for p in programs)


def _dirs(*paths: Path | None) -> tuple[Path, ...]:
    out: list[Path] = []
    for p in paths:
        if p is not None and p not in out:
            out.append(p)
    return tuple(out)


def _unread(env: Env, default_root: Path, root: Path) -> tuple[Path, ...]:
    """The default skills dir when a variable moved the agent's home elsewhere (it then reads only the new one)."""
    try:
        moved = not _same(default_root, root)
    except (OSError, ValueError):
        moved = False
    return (default_root / "skills",) if moved else ()


def _shared(agent_id: str, label: str, detect: Callable[[Env], bool],
            legacy: Callable[[Env], tuple[Path, ...]]) -> AgentSpec:
    return AgentSpec(agent_id, label, True, lambda e: None, detect, legacy)


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec("claude", "Claude Code", False, lambda e: _claude_root(e) / "skills" / SKILL_NAME,
              lambda e: _found(e, [e.home / ".claude"], ["claude"], ["CLAUDE_CONFIG_DIR"]),
              lambda e: _dirs(e.home / ".claude" / "skills") if e.path_var("CLAUDE_CONFIG_DIR") else (),
              unread_dirs=lambda e: _unread(e, e.home / ".claude", _claude_root(e)),
              home_var="CLAUDE_CONFIG_DIR"),
    # Codex and Pi: their dirs first; a program named `pi` counts only when it is the Pi coding agent
    AgentSpec("codex", "Codex", True, lambda e: None,
              lambda e: _found(e, [e.home / ".codex", *_dirs(e.path_var("CODEX_HOME"))], ["codex"]),
              lambda e: _dirs(_codex_root(e) / "skills", e.home / ".codex" / "skills"),
              unread_dirs=lambda e: _unread(e, e.home / ".codex", _codex_root(e)),
              home_var="CODEX_HOME"),
    _shared("pi", "Pi", lambda e: _found(e, [e.home / ".pi"], (), ["PI_CODING_AGENT_DIR"])
            or _program_is(e, "pi", "pi-coding-agent"),
            lambda e: _dirs((e.path_var("PI_CODING_AGENT_DIR") or e.home / ".pi" / "agent") / "skills")),
    AgentSpec("copilot", "Copilot CLI", True,
              lambda e: (h / "skills" / SKILL_NAME) if (h := e.path_var("COPILOT_HOME")) else None,
              lambda e: _found(e, [e.home / ".copilot"], ["copilot"], ["COPILOT_HOME"]),
              lambda e: _dirs(e.home / ".copilot" / "skills")),
    _shared("opencode", "OpenCode", lambda e: _found(e, [e.config_home() / "opencode"], ["opencode"],
                                                     ["OPENCODE_CONFIG_DIR"]),
            lambda e: _dirs(e.config_home() / "opencode" / "skills", e.config_home() / "opencode" / "skill",
                            (d / "skills") if (d := e.path_var("OPENCODE_CONFIG_DIR")) else None)),
    _shared("gemini", "Gemini CLI", lambda e: _found(e, [e.home / ".gemini"], ["gemini"]),
            lambda e: _dirs(e.home / ".gemini" / "skills")),
    _shared("cursor", "Cursor", lambda e: _found(e, [e.home / ".cursor"], ["cursor-agent", "cursor"]),
            lambda e: _dirs(e.home / ".cursor" / "skills")),
    _shared("amp", "Amp", lambda e: _found(e, [e.config_home() / "amp"], ["amp"]),
            lambda e: _dirs(e.config_home() / "amp" / "skills")),
    _shared("goose", "Goose", lambda e: _found(e, [e.config_home() / "goose"], ["goose"]),
            lambda e: _dirs(e.config_home() / "goose" / "skills")),
    _shared("droid", "Factory Droid", lambda e: _found(e, [e.home / ".factory"], ["droid"]),
            lambda e: _dirs(e.home / ".factory" / "skills")),
    _shared("windsurf", "Windsurf", lambda e: _found(e, [e.home / ".codeium" / "windsurf"], ["windsurf"]),
            lambda e: _dirs(e.home / ".codeium" / "windsurf" / "skills")),
    _shared("cline", "Cline", lambda e: _found(e, [e.home / ".cline"], ["cline"]),
            lambda e: _dirs(e.home / ".cline" / "skills")),
    _shared("warp", "Warp", lambda e: _found(e, [e.home / ".warp"], ["warp", "warp-terminal"]),
            lambda e: _dirs(e.home / ".warp" / "skills")),
    AgentSpec("antigravity", "Antigravity CLI", False,
              lambda e: e.home / ".gemini" / "antigravity-cli" / "skills" / SKILL_NAME,
              lambda e: _found(e, [e.home / ".gemini" / "antigravity-cli"]), lambda e: (), explicit_only=True),
)
AGENT_IDS: tuple[str, ...] = tuple(a.id for a in AGENTS)
_BY_ID = {a.id: a for a in AGENTS}


def _label(agent_id: str) -> str:
    return "shared copy" if agent_id == "shared" else _BY_ID[agent_id].label if agent_id in _BY_ID else agent_id


def _detected(env: Env, spec: AgentSpec) -> bool:
    try:
        return bool(spec.detect(env))
    except (OSError, ValueError):
        return False


def select_agents(env: Env, agents: str | Sequence[str]) -> tuple[str, ...]:
    """Agent ids for `--agent auto|all|id,id…` (auto: the detected ones; neither picks explicit-only agents);
    ValueError names an unknown id."""
    items = [a.strip().lower() for a in (agents.split(",") if isinstance(agents, str) else agents) if a.strip()]
    unknown = [a for a in items if a not in AGENT_IDS and a not in ("auto", "all", "shared")]
    if unknown:
        raise ValueError(f"unknown agent {', '.join(unknown)}; choose auto, all or any of: {', '.join(AGENT_IDS)}")
    chosen: set[str] = set()
    for item in items or ["auto"]:
        if item == "all":
            chosen.update(a.id for a in AGENTS if not a.explicit_only)
        elif item == "auto":
            chosen.update(a.id for a in AGENTS if not a.explicit_only and _detected(env, a))
        elif item != "shared":
            chosen.add(item)
    return tuple(a for a in AGENT_IDS if a in chosen)


def shared_readers(env: Env) -> tuple[str, ...]:
    """Agents served by the shared copy alone (no link) in this environment."""
    return tuple(a.id for a in AGENTS if a.reads_shared and a.link_path(env) is None)


# ---------------------------------------------------------------- packaged files
def _package_root() -> Traversable:
    return resources.files("runfleet.agent").joinpath("skill")


def _walk(node: Traversable, prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Traversable]]:
    for child in sorted(node.iterdir(), key=lambda t: t.name):
        if child.name == "__pycache__" or child.name.startswith("."):
            continue
        if child.is_dir():
            yield from _walk(child, (*prefix, child.name))
        else:
            yield (*prefix, child.name), child


def packaged_files() -> dict[str, bytes]:
    """Relative POSIX path → content of every packaged skill file (SKILL.md as shipped, before rendering)."""
    return {"/".join(parts): t.read_bytes() for parts, t in _walk(_package_root())}


def template() -> str:
    return _package_root().joinpath("SKILL.md").read_text(encoding="utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@cache
def bundle_sha256() -> str:
    """Hash of the packaged files (before rendering): a dev build with changed skill text at the same version differs.
    Computed once per process."""
    h = hashlib.sha256()
    for rel, data in sorted(packaged_files().items()):
        h.update(f"{rel}\0{_sha(data)}\n".encode())
    return h.hexdigest()


@cache
def install_identity() -> str:
    """Which RunFleet install is running: its resolved package dir. A dev checkout (`uv run`) and the installed tool
    can carry the same version with different skill text; only the one that wrote the skill refreshes it for that
    reason. cli.skill_install_identity computes the same without importing this module (a test pins the two)."""
    return os.path.normcase(str(Path(runfleet.__file__).resolve().parent))


def in_backoff(data: Mapping[str, Any], now: float) -> bool:
    """A refresh failed less than BACKOFF_SECONDS ago (`last_failure_at`, epoch seconds): no automatic retry yet."""
    at = data.get("last_failure_at")
    return isinstance(at, (int, float)) and not isinstance(at, bool) and 0 <= now - at < BACKOFF_SECONDS


def clean_hosts(host_names: Sequence[str]) -> list[str]:
    return [n.strip() for n in host_names if n.strip()]


def hosts_sha256(host_names: Sequence[str]) -> str:
    return _sha("\n".join(clean_hosts(host_names)).encode("utf-8"))


# ---------------------------------------------------------------- render
@dataclass(frozen=True, slots=True)
class RenderedSkill:
    text: str                                # the SKILL.md to install
    description: str
    hosts: tuple[str, ...]                   # host names prefixed to the description
    notes: tuple[str, ...]                   # names skipped (unsafe characters) or dropped (length cap)


def _usable(name: str) -> bool:
    return bool(name) and not _YAML_UNSAFE.search(name) and not name.startswith(_YAML_LEAD) and name.isprintable()


def render(host_names: Sequence[str], text: str | None = None) -> RenderedSkill:
    """SKILL.md with the host names prefixed to the frontmatter description as trigger words: names already among
    its words (case-insensitive) or repeated are left out, names with characters that could break the YAML or the
    loader (< > : # quotes, spaces, …) are skipped, and trailing names are dropped to keep the prefix within
    MAX_HOST_PREFIX and the description within MAX_DESCRIPTION characters."""
    text = (template() if text is None else text).replace("\r\n", "\n")
    lines = text.split("\n")
    end = lines.index("---", 1) if lines and lines[0] == "---" and "---" in lines[1:] else -1
    at = next((i for i in range(1, end) if lines[i].startswith(_DESC_KEY)), None) if end > 0 else None
    if at is None:
        raise SkillInstallError("packaged SKILL.md has no single-line `description:` in its frontmatter")
    base = lines[at][len(_DESC_KEY):].strip()
    seen = {w.lower() for w in _WORD.findall(base)}
    notes: list[str] = []
    hosts: list[str] = []
    for name in clean_hosts(host_names):
        if not _usable(name):
            notes.append(f"host name {name!r} not added to the skill description (unsupported characters)")
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        hosts.append(name)
    dropped: list[str] = []
    while hosts and (len(" ".join(hosts)) > MAX_HOST_PREFIX
                     or len(" ".join(hosts)) + 1 + len(base) > MAX_DESCRIPTION):
        dropped.insert(0, hosts.pop())
    if dropped:
        notes.append(f"{len(dropped)} host name(s) left out of the skill description to keep it short: "
                     f"{' '.join(dropped)}")
    description = " ".join([*hosts, base])
    lines[at] = f"{_DESC_KEY} {description}"
    return RenderedSkill(text="\n".join(lines), description=description, hosts=tuple(hosts), notes=tuple(notes))


# ---------------------------------------------------------------- versions
def _version_key(text: str) -> tuple[int, int, int, int, int] | None:
    m = _VERSION.fullmatch(text.strip())
    if m is None:
        return None
    major, minor, patch, phase, n = m.groups()
    return int(major), int(minor), int(patch or 0), _PHASE[phase], int(n or 0)


def version_newer(running: str, installed: str) -> bool | None:
    """True when `running` is newer than `installed`, False when older or the same release, None when either does
    not parse (X.Y.Z[.devN|aN|bN|rcN|.postN][+local]): then nothing is refreshed."""
    a, b = _version_key(running), _version_key(installed)
    if a is None or b is None:
        return None
    return a > b


# ---------------------------------------------------------------- small file-system helpers
def _real(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _same(a: Path | str, b: Path | str) -> bool:
    return _real(a) == _real(b)


def _is_link(path: Path) -> bool:
    """A symlink, or any Windows reparse point (a junction, also on Python 3.11 where os.path.isjunction is
    missing): never a copy of ours, and removed as a link, never by deleting what it points to."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove(path: Path) -> None:
    if _is_link(path):
        try:
            path.unlink()
        except (IsADirectoryError, PermissionError):
            os.rmdir(path)                   # a directory symlink / junction on Windows
    elif path.is_dir():
        shutil.rmtree(path)
    elif _exists(path):
        path.unlink()


def _tag() -> str:
    return uuid.uuid4().hex[:10]


def _show(env: Env, path: Path) -> str:
    try:
        rel = path.relative_to(env.home)
    except ValueError:
        return str(path)
    return ("~\\" if env.is_windows else "~/") + str(rel)


def _create_junction(target: Path, link: Path) -> None:
    """A Windows directory junction at `link` → `target` (absolute): no admin rights or developer mode needed."""
    try:
        winapi: Any = importlib.import_module("_winapi")
        winapi.CreateJunction(str(target), str(link))
        return
    except (ImportError, AttributeError):
        pass
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True,
                          check=False, timeout=30, **platform.background_spawn_kwargs())
    if proc.returncode != 0 or not _exists(link):
        raise OSError(f"mklink /J failed: {(proc.stderr or proc.stdout).strip()[:200]}")


def _link_text(p: Path) -> str:
    """Where the link `p` points (its text; a junction's target), never followed further."""
    try:
        return os.readlink(p)
    except (OSError, ValueError):
        return os.path.realpath(p)


def _tree(d: Path) -> dict[str, str]:
    """Relative POSIX path → sha256 of every file under `d` but the stamps (a link inside counts as `link:<dest>`;
    the tree is never followed through links: a directory link or Windows junction is pruned from the walk)."""
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(d, followlinks=False):
        base = Path(root)
        linked = [x for x in dirs if _is_link(base / x)]
        dirs[:] = [x for x in dirs if x not in linked]
        for name in [*files, *linked]:
            p = base / name
            rel = p.relative_to(d).as_posix()
            if rel in STAMPS:
                continue
            if _is_link(p):
                out[rel] = "link:" + _link_text(p)
            else:
                try:
                    out[rel] = _sha(p.read_bytes())
                except OSError:
                    out[rel] = "unreadable"
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_stamp(d: Path) -> dict[str, Any] | None:
    """The stamp of a copy RunFleet wrote (legacy .runfleet-version → {"version", "legacy": True}); None: not ours."""
    if _is_link(d) or not d.is_dir():
        return None
    stamp = _read_json(d / STAMP_FILE)
    if stamp is not None:
        return stamp
    try:
        version = (d / LEGACY_STAMP).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return {"version": version, "legacy": True}


def _manifest_path(state_dir: Path) -> Path:
    return state_dir / MANIFEST_FILE


def read_manifest(state_dir: Path) -> dict[str, Any] | None:
    data = _read_json(_manifest_path(state_dir))
    if data is None or data.get("schema") != SCHEMA or not isinstance(data.get("targets"), list):
        return None
    return data


def _write_manifest(state_dir: Path, data: dict[str, Any]) -> None:
    paths.ensure_private_dir(state_dir)
    tmp = state_dir / f".{MANIFEST_FILE}.tmp-{_tag()}"
    try:
        paths.write_private(tmp, json.dumps(data, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, _manifest_path(state_dir))
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _entry(manifest: Mapping[str, Any] | None, path: Path) -> dict[str, Any]:
    """The manifest's record of `path` ({} when it has none)."""
    for t in (manifest or {}).get("targets", []):
        if isinstance(t, dict) and t.get("path") == str(path):
            return t
    return {}


def _owned_copy(d: Path, manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The stamp when `d` is a copy of ours: its install id matches the manifest's (or, without a manifest, it has
    any stamp; or the manifest lists it as `pending`: a pre-manifest copy adoption could not rewrite yet)."""
    stamp = read_stamp(d)
    if stamp is None:
        return None
    if manifest is None or stamp.get("install_id") == manifest.get("install_id") or _entry(manifest, d).get("pending"):
        return stamp
    return None


def _shared_is_ours(env: Env, manifest: Mapping[str, Any] | None) -> bool:
    """The shared copy is ours or gone: only then can a link that resolves to its place be ours (a user's own link
    at ~/.agents/skills/runfleet and at ~/.claude/skills/runfleet to the same dir of theirs is not)."""
    canonical = canonical_dir(env)
    return not _exists(canonical) or _owned_copy(canonical, manifest) is not None


def _points_to(link: Path, target: Path) -> bool:
    """`link` is a link (also a dangling one) that resolves to `target`, realpath on both sides."""
    return _is_link(link) and _same(link, target)


def _leftovers(parent: Path) -> list[Path]:
    """Leftovers of an interrupted install next to a target: our temporary names (prefix + random hex) holding a
    link (only the link goes), a staging dir, or a stamped copy put aside (its user files are saved before it goes).
    Anything else is kept, whatever its name."""
    found: list[Path] = []
    try:
        entries = sorted(parent.iterdir()) if parent.is_dir() else []
    except OSError:
        return found
    for p in entries:
        if not _LEFTOVER.fullmatch(p.name):
            continue
        if _is_link(p) or (p.is_dir() and (p.name.startswith(STAGING) or read_stamp(p) is not None)):
            found.append(p)
        else:
            diagnostic(logging.INFO, "skill: %s looks like a leftover but is not RunFleet's; kept", p)
    return found


def _sweep(parent: Path, backup: _Backup) -> list[Path]:
    """Remove the leftovers next to a target (files the user edited or added in an old copy saved first); returns
    what was removed."""
    gone: list[Path] = []
    for p in _leftovers(parent):
        try:
            stamp = None if _is_link(p) or p.name.startswith(STAGING) else read_stamp(p)
            if stamp is not None:
                files = stamp.get("files")
                known = {str(k): str(v) for k, v in files.items()} if isinstance(files, dict) else None
                _save_files(backup, "leftover", p, known)
            _remove(p)
            gone.append(p)
        except OSError as exc:
            diagnostic(logging.WARNING, "skill: could not remove leftover %s: %s", p, exc)
    return gone


# ---------------------------------------------------------------- lock and backups
_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _locked(state_dir: Path, *, wait: float) -> Iterator[bool]:
    """The skill lock (<state_dir>/skill.lock, process-wide and across processes): yields True when held, False when
    still busy after `wait` seconds (0: one attempt)."""
    paths.ensure_private_dir(state_dir)
    deadline = time.monotonic() + wait
    got = _THREAD_LOCK.acquire(timeout=wait) if wait > 0 else _THREAD_LOCK.acquire(blocking=False)
    if not got:
        yield False
        return
    try:
        fd = _acquire(state_dir / LOCK_FILE, deadline)
        if fd is None:
            yield False
            return
        try:
            yield True
        finally:
            try:
                platform.unlock(fd)
            finally:
                os.close(fd)
    finally:
        _THREAD_LOCK.release()


def _acquire(path: Path, deadline: float) -> int | None:
    """An open, locked fd of the lock file at `path`, None when still busy at `deadline`. `uninstall` deletes the
    file while holding it (POSIX): a process that got the lock on the deleted file opens the new one and tries
    again, so two never hold "the" lock at once."""
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            while not platform.try_lock(fd):
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(0.05)
            if _still_named(fd, path):
                return fd
            platform.unlock(fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        os.close(fd)


def _still_named(fd: int, path: Path) -> bool:
    if platform.IS_WINDOWS:                  # an open file cannot be deleted there
        return True
    try:
        a, b = os.fstat(fd), os.stat(path)
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


@contextlib.contextmanager
def _lock_or_raise(state_dir: Path) -> Iterator[None]:
    with _locked(state_dir, wait=LOCK_WAIT) as held:
        if not held:
            raise SkillInstallError(f"another `runfleet skill` command is running (lock {state_dir / LOCK_FILE}); "
                                    "try again when it has finished")
        yield


class _Backup:
    """<state_dir>/skill-backup/<UTC time>/<agent>/…, created on first use (private)."""

    def __init__(self, state_dir: Path, now: datetime, *, dry_run: bool = False) -> None:
        self._base = state_dir / BACKUP_DIR
        self._stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._root: Path | None = None
        self.dry_run = dry_run
        self.saved: list[Path] = []
        self.older = 0                       # of these, files of an install that kept no record of what it wrote
        self._slots: dict[str, str] = {}
        self._used: set[str] = set()         # names written (lowercased)
        self._dirs: set[str] = set()         # directories the files need (a link record must not take one)

    def _free(self, dest: Path) -> Path:
        """`dest`, or `<stem>-2<suffix>`, -3 … when this backup already wrote that name (case-insensitively: a link
        record `a.link.txt` must never overwrite a file `A.link.txt` of the user, nor the other way round)."""
        candidate, n = dest, 1
        while candidate.as_posix().lower() in self._used | self._dirs or (not self.dry_run and _exists(candidate)):
            n += 1
            candidate = dest.with_name(f"{dest.stem}-{n}{dest.suffix}")
        self._used.add(candidate.as_posix().lower())
        return candidate

    def _slot(self, agent: str, src: Path) -> str:
        """<agent> for the first source of that agent, <agent>-2, -3 … for others (two stray copies of one agent
        must not overwrite each other's files)."""
        key = os.path.normcase(os.path.abspath(src))
        if key not in self._slots:
            name, n = agent, 1
            while name in self._slots.values():
                n += 1
                name = f"{agent}-{n}"
            self._slots[key] = name
        return self._slots[key]

    @property
    def root(self) -> Path:
        if self._root is None:
            root, n = self._base / self._stamp, 1
            while _exists(root) and not self.dry_run:
                n += 1
                root = self._base / f"{self._stamp}-{n}"
            if not self.dry_run:
                paths.ensure_private_dir(self._base)
                root.mkdir(parents=True)
            self._root = root
        return self._root

    def copy_files(self, agent: str, src: Path, rels: Sequence[str], *, older: bool = False) -> list[Path]:
        """Copy these files of `src`; a link inside (a symlink, a Windows junction) is saved as `<name>.link.txt`
        holding its target, never copied through."""
        out: list[Path] = []
        links = {rel for rel in rels if _is_link(src / Path(*rel.split("/")))}
        slot = self.root / self._slot(agent, src)
        for rel in rels:
            if rel not in links:
                self._dirs.update(p.as_posix().lower() for p in (slot / Path(*rel.split("/"))).parents)
        for rel in sorted(rels, key=lambda r: r in links):              # files first, then the link records
            path = src / Path(*rel.split("/"))
            dest = slot / Path(*rel.split("/"))
            link = rel in links
            dest = self._free(dest.with_name(dest.name + ".link.txt") if link else dest)
            if not self.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if link:
                    dest.write_text(f"link to {_link_text(path)}\n", encoding="utf-8")
                else:
                    shutil.copy2(path, dest, follow_symlinks=False)
            out.append(dest)
        self.saved += out
        if older:
            self.older += len(out)
        return out

    def move(self, agent: str, path: Path) -> Path:
        dest = self.root / self._slot(agent, path) / path.name
        if not self.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
        self.saved.append(dest)
        return dest


def _user_files(d: Path, known: Mapping[str, str] | None, baseline: Mapping[str, str] | None = None) -> list[str]:
    """Files in our copy `d` that RunFleet did not write as they are: edited (hash differs from `known`) or added
    (not listed). Nothing known (a legacy copy): every file but those identical to `baseline` (this version's)."""
    tree = _tree(d)
    if known is None:
        return sorted(rel for rel, h in tree.items() if baseline is None or baseline.get(rel) != h)
    return sorted(rel for rel, h in tree.items() if known.get(rel) != h)


def _packaged_hashes() -> dict[str, str]:
    return {rel: _sha(data) for rel, data in packaged_files().items()}


def _save_files(backup: _Backup, agent: str, d: Path, known: Mapping[str, str] | None,
                baseline: Mapping[str, str] | None = None) -> list[Path]:
    """Back up the user's files of our copy `d` before it is replaced or removed. Without a record of what was
    written (a pre-release copy), files identical to this version's are skipped and the rest are reported as files
    of an older install, not as the user's edits."""
    if known is None and baseline is None:
        baseline = _packaged_hashes()
    rels = _user_files(d, known, baseline)
    return backup.copy_files(agent, d, rels, older=known is None) if rels else []


# ---------------------------------------------------------------- results
@dataclass(frozen=True, slots=True)
class TargetResult:
    """One place the skill is (or was) installed: the shared copy, or an agent's link / copy."""

    agent: str                   # "shared" | an AGENT_IDS id
    path: Path
    kind: str                    # "copy" | "symlink" | "junction" | "" (nothing there)
    action: str                  # created updated unchanged replaced kept planned failed missing removed foreign …
    error: str | None = None
    link_target: str | None = None
    backups: tuple[Path, ...] = ()
    files: Mapping[str, str] | None = None               # copies: hashes of the files written

    def to_json(self) -> dict[str, Any]:
        return {"agent": self.agent, "path": str(self.path), "kind": self.kind or None, "action": self.action,
                "error": self.error, "link_target": self.link_target, "backups": [str(p) for p in self.backups]}

    def manifest_entry(self) -> dict[str, Any]:
        return {"agent": self.agent, "path": str(self.path), "kind": self.kind, "link_target": self.link_target,
                "files": dict(self.files) if self.files is not None else None}


def _target_line(env: Env, t: TargetResult) -> str:
    where = _show(env, t.path)
    if t.error:
        return f"{_label(t.agent)}: {where} {t.action}: {t.error}"
    if t.kind in ("symlink", "junction"):
        return f"{_label(t.agent)}: {where} → {_show(env, canonical_dir(env))} ({t.kind}, {t.action})"
    return f"{_label(t.agent)}: {where} ({t.kind or 'none'}, {t.action})"


def _backup_lines(env: Env, backups: Sequence[Path], root: Path | None, *, older: int = 0,
                  dry_run: bool = False) -> list[str]:
    if not backups or root is None:
        return []
    verb = "Would save" if dry_run else "Saved"
    out: list[str] = []
    if len(backups) > older:
        out.append(f"{verb} {len(backups) - older} file(s) or folder(s) you edited, added or had in the way to "
                   f"{_show(env, root)}")
    if older:
        out.append(f"{verb} {older} file(s) from an older install (it kept no record of what it wrote) to "
                   f"{_show(env, root)}")
    return out


@dataclass(frozen=True, slots=True)
class InstallResult:
    env: Env
    canonical: Path
    version: str
    targets: tuple[TargetResult, ...]
    rendered: RenderedSkill
    readers: tuple[str, ...]                 # agent ids served by the shared copy alone
    agents: tuple[str, ...]                  # agent ids chosen
    backup_dir: Path | None = None
    backups: tuple[Path, ...] = ()
    notes: tuple[str, ...] = ()
    auto_update: bool = True
    printed_only: bool = False
    manifest_path: Path | None = None
    swept: tuple[Path, ...] = ()
    permission_rule: str = PERMISSION_RULE
    permission_hint: str = PERMISSION_HINT
    older_backups: int = 0

    @property
    def exit_code(self) -> int:
        return 1 if any(t.action == "failed" for t in self.targets) else 0

    def lines(self) -> list[str]:
        if self.printed_only:
            return self.rendered.text.rstrip("\n").split("\n")
        env, shared = self.env, self.targets[0] if self.targets else None
        out: list[str] = []
        if shared is not None and shared.action == "failed":
            out.append(f"Could not install the runfleet skill in {_show(env, self.canonical)}: {shared.error}")
        else:
            out.append(f"Installed the runfleet skill {self.version} in {_show(env, self.canonical)} "
                       f"({shared.action if shared else 'unchanged'})")
            out.append("  read there by " + ", ".join(_label(a) for a in self.readers)
                       + " and other agents that read ~/.agents/skills")
        out += [_target_line(env, t) for t in self.targets[1:]]
        if self.rendered.hosts:
            out.append(f"Trigger words: {' '.join(self.rendered.hosts)}")
        out += _backup_lines(env, self.backups, self.backup_dir, older=self.older_backups)
        out += [f"Note: {n}" for n in self.notes]
        out.append("Kept up to date automatically when RunFleet is upgraded" if self.auto_update else
                   "Automatic refresh is off: run `runfleet skill update` after upgrading RunFleet")
        out.append(f"Permission rule: {self.permission_rule}")
        out.append(self.permission_hint)
        return out

    def to_json(self) -> dict[str, Any]:
        return {"printed_only": self.printed_only, "version": self.version, "canonical": str(self.canonical),
                "targets": [t.to_json() for t in self.targets], "covered_by_shared": list(self.readers),
                "agents": list(self.agents), "hosts": list(self.rendered.hosts),
                "backup_dir": str(self.backup_dir) if self.backup_dir else None,
                "backups": [str(p) for p in self.backups], "notes": list(self.notes),
                "auto_update": self.auto_update, "swept": [str(p) for p in self.swept],
                "permission_rule": self.permission_rule, "exit_code": self.exit_code}


@dataclass(frozen=True, slots=True)
class UpdateResult:
    env: Env
    # updated | unchanged | not_installed | canonical_missing | disabled (--if-installed, automatic refresh off) |
    # newer_installed (--if-installed, a newer RunFleet wrote it: never downgraded silently)
    state: str
    version: str
    previous_version: str | None
    targets: tuple[TargetResult, ...] = ()
    backup_dir: Path | None = None
    backups: tuple[Path, ...] = ()
    notes: tuple[str, ...] = ()
    if_installed: bool = False
    downgrade: bool = False
    older_backups: int = 0

    @property
    def exit_code(self) -> int:
        return 1 if any(t.action == "failed" for t in self.targets) else 0

    def failure(self) -> str | None:
        """What failed, in a few words (None: nothing)."""
        failed = [t for t in self.targets if t.action == "failed"]
        if not failed:
            return None
        return f"{len(failed)} target(s) failed, first {_show(self.env, failed[0].path)}: {failed[0].error}"

    def summary(self) -> str | None:
        """One line for `--quiet` and the post-upgrade hook; None when there is nothing worth saying."""
        canonical = _show(self.env, canonical_dir(self.env))
        if self.state in ("disabled", "newer_installed"):
            return f"runfleet skill: {self.notes[0]}" if self.notes else None
        if self.state == "updated":
            failed = [t for t in self.targets if t.action == "failed"]
            prev = f"{self.previous_version} → " if self.previous_version not in (None, self.version) else ""
            extra = f"; {len(failed)} target(s) failed (runfleet skill status)" if failed else ""
            verb = "downgraded" if self.downgrade else "updated"
            return f"runfleet skill {verb} ({prev}{self.version}) in {canonical}{extra}"
        if self.state == "canonical_missing":
            return (f"runfleet skill: {canonical} was removed, not recreated; `runfleet skill install` puts it back, "
                    "`runfleet skill uninstall` cleans up the rest")
        if self.state == "unchanged" and self.exit_code:
            return "runfleet skill: some targets could not be refreshed (runfleet skill status)"
        return None

    def lines(self) -> list[str]:
        if self.state == "not_installed":
            return [] if self.if_installed else ["The runfleet skill is not installed; run `runfleet skill install`"]
        out: list[str] = []
        if self.state in ("disabled", "newer_installed"):
            return [f"runfleet skill: {n}" for n in self.notes]
        if self.state == "unchanged":
            out.append(f"The runfleet skill {self.version} is up to date in {_show(self.env, canonical_dir(self.env))}")
        elif (line := self.summary()) is not None:
            out.append(line)
        out += [_target_line(self.env, t) for t in self.targets if t.action not in ("unchanged",)]
        out += _backup_lines(self.env, self.backups, self.backup_dir, older=self.older_backups)
        out += [f"Note: {n}" for n in self.notes]
        return out

    def to_json(self) -> dict[str, Any]:
        return {"state": self.state, "version": self.version, "previous_version": self.previous_version,
                "downgrade": self.downgrade, "targets": [t.to_json() for t in self.targets],
                "backup_dir": str(self.backup_dir) if self.backup_dir else None,
                "backups": [str(p) for p in self.backups], "notes": list(self.notes), "exit_code": self.exit_code}


@dataclass(frozen=True, slots=True)
class UninstallResult:
    env: Env
    dry_run: bool
    removed: tuple[TargetResult, ...]
    kept: tuple[TargetResult, ...] = ()          # not ours, or failed
    backup_dir: Path | None = None
    backups: tuple[Path, ...] = ()
    swept: tuple[Path, ...] = ()
    manifest_removed: bool = False
    follow_ups: tuple[str, ...] = FOLLOW_UPS
    older_backups: int = 0
    removed_dirs: tuple[Path, ...] = ()          # empty skills dirs RunFleet had created

    @property
    def nothing_installed(self) -> bool:
        return not self.removed and not self.manifest_removed and not any(t.action == "failed" for t in self.kept)

    @property
    def exit_code(self) -> int:
        return 1 if any(t.action == "failed" for t in self.kept) else 0

    def lines(self) -> list[str]:
        env = self.env
        out: list[str] = []
        if self.nothing_installed:
            out.append("The runfleet skill is not installed; nothing to remove")
        verb = "Would remove" if self.dry_run else "Removed"
        for t in self.removed:
            what = f"{t.kind} → {_show(env, canonical_dir(env))}" if t.kind in ("symlink", "junction") else t.kind
            out.append(f"{verb} {_label(t.agent)}: {_show(env, t.path)} ({what})")
        for t in self.kept:
            out.append(f"Kept {_show(env, t.path)}: {t.error}" if t.error else f"Kept {_show(env, t.path)}")
        if self.swept:
            out.append(f"{verb} {len(self.swept)} leftover(s) of interrupted installs")
        if self.manifest_removed:
            out.append(f"{verb} the install record (skill.json)")
        if self.removed_dirs:
            out.append(f"{verb} {len(self.removed_dirs)} empty folder(s) RunFleet had created: "
                       + ", ".join(_show(env, p) for p in self.removed_dirs))
        out += _backup_lines(env, self.backups, self.backup_dir, older=self.older_backups, dry_run=self.dry_run)
        out.append("Not touched (remove by hand if you set them up):")
        out += [f"  {f}" for f in self.follow_ups]
        return out

    def to_json(self) -> dict[str, Any]:
        return {"dry_run": self.dry_run, "nothing_installed": self.nothing_installed,
                "removed": [t.to_json() for t in self.removed], "kept": [t.to_json() for t in self.kept],
                "backup_dir": str(self.backup_dir) if self.backup_dir else None,
                "backups": [str(p) for p in self.backups], "swept": [str(p) for p in self.swept],
                "manifest_removed": self.manifest_removed, "removed_dirs": [str(p) for p in self.removed_dirs],
                "follow_ups": list(self.follow_ups), "exit_code": self.exit_code}


@dataclass(frozen=True, slots=True)
class AgentStatus:
    id: str
    label: str
    detected: bool
    covered_by: str              # shared | link | copy | none
    path: Path | None            # its link / copy path; None for shared readers
    state: str                   # ok | missing | broken | foreign | stale | n/a
    in_manifest: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "detected": self.detected, "covered_by": self.covered_by,
                "path": str(self.path) if self.path else None, "state": self.state, "in_manifest": self.in_manifest}


@dataclass(frozen=True, slots=True)
class Stray:
    agent: str
    path: Path
    kind: str                    # stamped_copy | link_to_ours | stale_link | foreign

    def to_json(self) -> dict[str, Any]:
        return {"agent": self.agent, "path": str(self.path), "kind": self.kind}


@dataclass(frozen=True, slots=True)
class StatusReport:
    env: Env
    installed: bool
    canonical: Path
    canonical_state: str         # ok | missing | foreign
    version: str | None          # of the installed copy
    running_version: str
    up_to_date: bool
    hosts: tuple[str, ...]
    auto_update: bool | None
    edited: tuple[str, ...]
    agents: tuple[AgentStatus, ...]
    strays: tuple[Stray, ...]
    broken_links: tuple[Path, ...]
    manifest_path: Path
    legacy: bool = False         # installed by a 0.8 pre-release (no manifest yet)
    exit_code: int = 0
    last_backup: Path | None = None              # where the last update saved the user's files
    last_failure: str | None = None              # the last refresh that failed: when and why

    def lines(self) -> list[str]:
        env = self.env
        where = _show(env, self.canonical)
        if not self.installed:
            out = [f"The runfleet skill is not installed ({where} {self.canonical_state}); "
                   "`runfleet skill install` adds it"]
        else:
            fresh = "up to date" if self.up_to_date else f"out of date (RunFleet {self.running_version}; " \
                                                          "`runfleet skill update`)"
            out = [f"runfleet skill {self.version or '?'} in {where}: {fresh}"
                   + (", installed by an older RunFleet (adopted on the next update)" if self.legacy else "")]
            out.append("  automatic refresh: " + ("on" if self.auto_update else "off"))
            if self.last_failure:
                out.append(f"  last refresh failed {self.last_failure} (retried automatically at most once an "
                           "hour; `runfleet skill update` tries now)")
            if self.hosts:
                out.append(f"  trigger words: {' '.join(self.hosts)}")
            if self.edited:
                out.append(f"  edited by you (saved before the next update): {', '.join(self.edited)}")
            if self.last_backup is not None:
                out.append(f"  your edited files from the last update are saved in {_show(env, self.last_backup)}")
        for a in self.agents:
            if not (a.detected or a.in_manifest):
                continue
            how = {"shared": "reads the shared copy", "link": f"link {_show(env, a.path) if a.path else ''}",
                   "copy": f"copy {_show(env, a.path) if a.path else ''}"}.get(a.covered_by, "not covered")
            state = "" if a.state in ("ok", "n/a") else f" ({a.state})"
            fix = (f"; `runfleet skill install --agent {a.id}` adds it"
                   if self.installed and a.detected and a.covered_by == "none" and a.state == "missing" else "")
            out.append(f"  {a.label}: {how}{state}{fix}")
        for s in self.strays:
            if s.kind == "stale_link":
                var = _BY_ID[s.agent].home_var if s.agent in _BY_ID else None
                rm = f"rmdir {s.path}" if env.is_windows else f"rm {_show(env, s.path)}"
                out.append(f"  stale link: {_show(env, s.path)} → the shared copy, unused ({_label(s.agent)} reads "
                           f"${var}/skills now); `{rm}` removes it")
                continue
            what = {"stamped_copy": "a stray RunFleet copy (`runfleet skill uninstall` removes it)",
                    "link_to_ours": "a second link to the shared copy (`runfleet skill uninstall` removes it)",
                    "foreign": "another skill named runfleet (not RunFleet's; may shadow it)"}[s.kind]
            out.append(f"  duplicate for {_label(s.agent)}: {_show(env, s.path)} is {what}")
        for p in self.broken_links:
            out.append(f"  broken link: {_show(env, p)}")
        return out

    def to_json(self) -> dict[str, Any]:
        return {"installed": self.installed, "canonical": str(self.canonical),
                "canonical_state": self.canonical_state, "version": self.version,
                "running_version": self.running_version, "up_to_date": self.up_to_date, "hosts": list(self.hosts),
                "auto_update": self.auto_update, "edited": list(self.edited), "legacy": self.legacy,
                "agents": [a.to_json() for a in self.agents], "strays": [s.to_json() for s in self.strays],
                "broken_links": [str(p) for p in self.broken_links], "manifest": str(self.manifest_path),
                "last_backup": str(self.last_backup) if self.last_backup else None,
                "last_failure": self.last_failure}


# ---------------------------------------------------------------- writing one target
@dataclass
class _Ctx:
    env: Env
    state_dir: Path
    files: dict[str, bytes]
    stamp: dict[str, Any]
    manifest: Mapping[str, Any] | None
    backup: _Backup
    notes: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)     # skills dirs (and parents) this run created

    @property
    def canonical(self) -> Path:
        return canonical_dir(self.env)

    def stamp_bytes(self) -> bytes:
        return (json.dumps(self.stamp, sort_keys=True) + "\n").encode()

    def hashes(self) -> dict[str, str]:
        return {rel: _sha(data) for rel, data in self.files.items()}


def _known_files(manifest: Mapping[str, Any] | None, path: Path, stamp: Mapping[str, Any]) -> dict[str, str] | None:
    """What RunFleet wrote into the copy at `path`: the manifest's file list, else the stamp's; None if unknown."""
    files = _entry(manifest, path).get("files")
    if isinstance(files, dict):
        return {str(k): str(v) for k, v in files.items()}
    files = stamp.get("files")
    return {str(k): str(v) for k, v in files.items()} if isinstance(files, dict) else None


def _save_user_files(ctx: _Ctx, agent: str, d: Path, stamp: Mapping[str, Any]) -> list[Path]:
    known = _known_files(ctx.manifest, d, stamp)
    return _save_files(ctx.backup, agent, d, known, ctx.hashes() if known is None else None)


def _ensure_dir(ctx: _Ctx, d: Path) -> None:
    """mkdir -p `d`, remembering which of its dirs did not exist (uninstall removes them again while empty)."""
    missing: list[Path] = []
    p = d
    while not _exists(p) and p != p.parent:
        missing.append(p)
        p = p.parent
    d.mkdir(parents=True, exist_ok=True)
    for m in missing:                        # by real path: uninstall then never removes one through a link
        real = os.path.realpath(m)
        if real not in ctx.created:
            ctx.created.append(real)


def _swap_in(new: Path, dest: Path, *, atomic: bool) -> Path | None:
    """Rename `new` to `dest`. `atomic` (POSIX, link over link or nothing): one os.replace. Otherwise whatever is at
    `dest` is renamed aside first, put back if the rename fails, and returned for the caller to remove."""
    if atomic or not _exists(dest):
        os.replace(new, dest)
        return None
    old = dest.parent / f".runfleet.old-{_tag()}"
    os.replace(dest, old)
    try:
        os.replace(new, dest)
    except BaseException:
        os.replace(old, dest)
        raise
    return old


def _drop_old(old: Path | None) -> None:
    if old is None:
        return
    try:
        _remove(old)
    except OSError as exc:                   # the new one is in place; a leftover is swept next time
        diagnostic(logging.WARNING, "skill: could not remove %s: %s", old, exc)


def _write_copy(ctx: _Ctx, agent: str, dest: Path) -> TargetResult:
    """Our copy at `dest` (nothing there, or a copy of ours whose user files are saved first): staged next to it, then
    swapped in. Unchanged when it already holds exactly these files."""
    files, hashes = ctx.files, ctx.hashes()
    existing = _owned_copy(dest, ctx.manifest) if _exists(dest) else None
    if existing is None and _exists(dest):
        raise OSError(f"{dest} is not RunFleet's; left alone")
    if existing is not None and _tree(dest) == hashes and existing == ctx.stamp:
        return TargetResult(agent, dest, "copy", "unchanged", files=hashes)
    backups = _save_user_files(ctx, agent, dest, existing) if existing is not None else []
    _ensure_dir(ctx, dest.parent)
    staging = dest.parent / f"{STAGING}{_tag()}"
    try:
        for rel, data in files.items():
            out = staging.joinpath(*rel.split("/"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        (staging / STAMP_FILE).write_bytes(ctx.stamp_bytes())
        old = _swap_in(staging, dest, atomic=False)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _drop_old(old)
    action = "created" if existing is None and old is None else "updated"
    return TargetResult(agent, dest, "copy", action, backups=tuple(backups), files=hashes)


def _make_link(ctx: _Ctx, at: Path) -> str:
    """A link at `at` (a new name next to the destination) to the shared copy; returns its kind. Raises OSError when
    no kind of link works here."""
    env, target = ctx.env, ctx.canonical
    errors: list[str] = []
    if env.is_windows:
        try:
            _create_junction(Path(os.path.realpath(target)), at)
            return "junction"
        except OSError as exc:
            errors.append(f"junction: {exc}")
    real_target, real_parent = os.path.realpath(target), os.path.realpath(at.parent)
    try:
        rel = os.path.relpath(real_target, real_parent)
    except ValueError:                       # another drive on Windows
        rel = real_target
    for dest in dict.fromkeys([real_target] if env.is_windows else [rel, real_target]):
        try:
            os.symlink(dest, at, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            errors.append(f"symlink: {exc}")
            continue
        if _same(at, target):
            return "symlink"
        at.unlink()
        errors.append(f"symlink {dest} does not resolve to {target}")
    raise OSError("; ".join(errors))


def _write_link(ctx: _Ctx, agent: str, dest: Path, *, prefer: str | None = None) -> TargetResult:
    """A link (or, when no link works, a tracked copy) at `dest` → the shared copy. `dest` is empty, a link to the
    shared copy (kept as is), or a copy of ours (its user files saved first). The new link is made under a temporary
    name and only then swapped in, so a working old one stays when anything fails."""
    env = ctx.env
    if _points_to(dest, ctx.canonical):
        kind = "junction" if env.is_windows and not os.path.islink(dest) else "symlink"
        return TargetResult(agent, dest, kind, "unchanged", link_target=os.path.realpath(dest))
    existing = _owned_copy(dest, ctx.manifest) if _exists(dest) else None
    if existing is None and _exists(dest):
        raise OSError(f"{dest} is not RunFleet's; left alone")
    _ensure_dir(ctx, dest.parent)
    tmp = dest.parent / f".runfleet.tmp-{_tag()}"
    if prefer != "copy":
        try:
            kind = _make_link(ctx, tmp)
        except OSError as exc:
            diagnostic(logging.INFO, "skill: no link possible at %s (%s); copying instead", dest, exc)
            ctx.notes.append(f"could not link {_show(env, dest)} ({exc}); copied the skill there instead "
                             "(refreshed with the shared copy)")
        else:
            try:
                backups = _save_user_files(ctx, agent, dest, existing) if existing is not None else []
                atomic = not env.is_windows and (not _exists(dest) or _is_link(dest))
                old = _swap_in(tmp, dest, atomic=atomic)
            except BaseException:
                with contextlib.suppress(OSError):
                    _remove(tmp)
                raise
            _drop_old(old)
            action = "replaced" if existing is not None else "created"
            return TargetResult(agent, dest, kind, action, link_target=os.path.realpath(ctx.canonical),
                                backups=tuple(backups))
    return _write_copy(ctx, agent, dest)


def _blocker(ctx: _Ctx, agent: str, dest: Path, *, link: bool, shared_ours: bool = True) -> str | None:
    """Why `dest` is not ours to replace (None: it is empty or ours). A link to the shared copy's place is ours only
    while the shared copy is (`shared_ours`)."""
    if not _exists(dest):
        return None
    if link and _points_to(dest, ctx.canonical):
        return None if shared_ours else "a link to the shared copy's place, which is not RunFleet's"
    if _owned_copy(dest, ctx.manifest) is not None:
        return None
    if _is_link(dest):
        return "a link to elsewhere" if os.path.exists(dest) else "a broken link"
    return "a directory" if dest.is_dir() else "a file"


# ---------------------------------------------------------------- install / update / uninstall
# called under the lock with the config path the manifest records (None: the default config) → (host names or None
# when that config cannot be read now, the config path to record, its mtime)
ConfigResolver = Callable[[str | None], tuple[Sequence[str] | None, str | None, float | None]]
_MODES = ("list", "auto", "all")             # --agent at install: ids only, auto (the detected ones), all


def _bundle(host_names: Sequence[str]) -> tuple[RenderedSkill, dict[str, bytes]]:
    rendered = render(host_names)
    files = packaged_files()
    files["SKILL.md"] = rendered.text.encode("utf-8")
    return rendered, files


def _plan_links(env: Env, agent_ids: Sequence[str]) -> list[tuple[str, Path]]:
    """(agent, link path) for the chosen agents that need a link; none when its skills dir is the shared one."""
    out: list[tuple[str, Path]] = []
    canonical = canonical_dir(env)
    for agent_id in agent_ids:
        p = _BY_ID[agent_id].link_path(env)
        if p is None or _same(p.parent, canonical.parent) or any(_same(p, q) for _, q in out):
            continue
        out.append((agent_id, p))
    return out


def _agents_mode(agents: str | Sequence[str], previous: Any) -> str:
    """How the agents were chosen, the widest of this install and the earlier ones (`install --agent claude` after
    an auto install stays auto): explicit `skill update` links agents found later only for auto and all."""
    items = [a.strip().lower() for a in (agents.split(",") if isinstance(agents, str) else agents) if a.strip()]
    mode = "all" if "all" in items else "auto" if "auto" in items or not items else "list"
    if previous in _MODES and _MODES.index(str(previous)) > _MODES.index(mode):
        return str(previous)
    return mode


def _str_list(value: Any) -> list[str]:
    return [str(v) for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _new_manifest(ctx: _Ctx, *, hosts: Sequence[str], config_path: str | None, config_mtime: float | None,
                  auto_update: bool, agents: Sequence[str], agents_mode: str | None,
                  targets: Sequence[Mapping[str, Any]], results: Sequence[TargetResult],
                  now: datetime) -> dict[str, Any]:
    """The manifest after an install / update. When a copy could not be rewritten it still holds the old text: the
    bundle hash is recorded as RETRY, so the next refresh at this version tries again (the version itself is this
    one, so an older RunFleet never takes that as a reason to downgrade); any failure records `last_failure_at`, so
    the automatic refresh retries at most once per BACKOFF_SECONDS. `writer`: install_identity() of this RunFleet."""
    previous = ctx.manifest or {}
    data: dict[str, Any] = {
        "schema": SCHEMA, "install_id": ctx.stamp["install_id"], "version": runfleet.__version__,
        "bundle_sha256": bundle_sha256(), "writer": install_identity(), "hosts": list(hosts),
        "hosts_sha256": hosts_sha256(hosts), "config_path": config_path, "config_mtime": config_mtime,
        "auto_update": auto_update, "agents": list(agents), "agents_mode": agents_mode,
        "updated_at": now.astimezone(UTC).isoformat(timespec="seconds"), "targets": [dict(t) for t in targets],
        "created_dirs": list(dict.fromkeys([*_str_list(previous.get("created_dirs")), *ctx.created])),
        # every agent link path ever recorded: an explicit update links a path not among them (an agent found since,
        # or moved by $CLAUDE_CONFIG_DIR / $COPILOT_HOME), never one the user removed
        "linked_paths": list(dict.fromkeys([
            *_str_list(previous.get("linked_paths")),
            *(str(t.get("path")) for t in previous.get("targets") or [] if isinstance(t, dict)
              and t.get("path") and t.get("agent") != "shared"),           # also those gone now: never recreated
            *(str(t.get("path")) for t in targets if t.get("agent") != "shared")]))}
    if ctx.backup.saved and not ctx.backup.dry_run:
        data["last_backup"] = str(ctx.backup.root)
    elif isinstance(previous.get("last_backup"), str):
        data["last_backup"] = previous["last_backup"]
    failed = [r for r in results if r.action == "failed"]
    if any(r.kind == "copy" for r in failed):
        data["bundle_sha256"] = RETRY
    if failed:
        data["last_failure_at"] = now.timestamp()
        data["last_error"] = f"{_show(ctx.env, failed[0].path)}: {failed[0].error}"[:300]
    return data


def _ctx(env: Env, state_dir: Path, files: dict[str, bytes], manifest: Mapping[str, Any] | None,
         install_id: str, now: datetime) -> _Ctx:
    stamp = {"version": runfleet.__version__, "bundle_sha256": bundle_sha256(), "install_id": install_id,
             "files": {rel: _sha(data) for rel, data in files.items()}}
    return _Ctx(env=env, state_dir=state_dir, files=files, stamp=stamp, manifest=manifest,
                backup=_Backup(state_dir, now))


def _sweep_all(paths_: Sequence[Path], backup: _Backup) -> list[Path]:
    swept: list[Path] = []
    for parent in dict.fromkeys(p.parent for p in paths_):
        swept += _sweep(parent, backup)
    return swept


def _aliases_canonical(p: Path, canonical: Path) -> bool:
    """`p` is the shared copy under another name (its skills dir links to the shared one, or an aliased home)."""
    return _same(p.parent, canonical.parent) or (not _is_link(p) and _same(p, canonical))


def _is_canonical_path(p: Path, canonical: Path) -> bool:
    """`p` names the shared copy's place (also through a symlinked home), whatever is there now."""
    return p.name == canonical.name and _same(p.parent, canonical.parent)


def install(env: Env, *, state_dir: Path, host_names: Sequence[str], config_path: str | None,
            config_mtime: float | None, agents: str | Sequence[str] = "auto", force: bool = False,
            auto_update: bool = True, print_only: bool = False, now: datetime) -> InstallResult:
    """Write the shared copy and the links the chosen agents need, record them in the manifest. Something RunFleet
    did not write in the way → SkillInstallError before anything is written, unless `force` (then it is moved to the
    backup dir). Per-target failures are reported in the result (exit_code 1). `print_only` writes nothing."""
    chosen = select_agents(env, agents)
    canonical = canonical_dir(env)
    links = _plan_links(env, chosen)
    readers = shared_readers(env)
    hosts = clean_hosts(host_names)
    rendered, files = _bundle(hosts)
    if print_only:
        planned = (TargetResult("shared", canonical, "copy", "planned"),
                   *(TargetResult(a, p, "junction" if env.is_windows else "symlink", "planned") for a, p in links))
        return InstallResult(env=env, canonical=canonical, version=runfleet.__version__, targets=planned,
                             rendered=rendered, readers=readers, agents=chosen, notes=rendered.notes,
                             auto_update=auto_update, printed_only=True, permission_hint=permission_hint(env))
    with _lock_or_raise(state_dir):
        manifest = read_manifest(state_dir)
        install_id = str(manifest["install_id"]) if manifest and manifest.get("install_id") else uuid.uuid4().hex
        ctx = _ctx(env, state_dir, files, manifest, install_id, now)
        ctx.notes += rendered.notes
        shared_ours = _shared_is_ours(env, manifest)
        for t in (manifest or {}).get("targets", []):       # keep (and refresh) what an earlier install linked
            agent, p = str(t.get("agent")), Path(str(t.get("path")))
            if agent != "shared" and t.get("path") and p not in {q for _, q in links} and (
                    not _aliases_canonical(p, canonical)) and (
                    (shared_ours and _points_to(p, canonical)) or _owned_copy(p, manifest) is not None):
                links.append((agent, p))
        recorded = tuple(dict.fromkeys([*chosen, *(a for a, _ in links)]))
        swept = _sweep_all([canonical, *(p for _, p in links)], ctx.backup)
        blocked = [(a, p, why) for a, p, why in
                   [("shared", canonical, _blocker(ctx, "shared", canonical, link=False)),
                    *((a, p, _blocker(ctx, a, p, link=True, shared_ours=shared_ours)) for a, p in links)] if why]
        if blocked and not force:
            raise SkillInstallError("; ".join(f"{p} already exists ({why}, not installed by RunFleet)"
                                              for _, p, why in blocked)
                                    + ": move it away, or re-run with --force to move it into the backup folder "
                                    f"{state_dir / BACKUP_DIR}")
        moved: dict[Path, Path] = {}
        for agent, p, _why in blocked:
            try:
                moved[p] = ctx.backup.move(agent, p)
            except OSError as exc:
                raise SkillInstallError(f"could not move {p} out of the way: {exc}") from exc
        results: list[TargetResult] = []
        try:
            shared = _write_copy(ctx, "shared", canonical)
        except OSError as exc:
            shared = TargetResult("shared", canonical, "copy", "failed", error=str(exc))
        if canonical in moved:
            shared = _replace(shared, action=shared.action if shared.action == "failed" else "replaced",
                              backups=(*shared.backups, moved[canonical]))
        results.append(shared)
        for agent, p in links:
            if shared.action == "failed" and not os.path.isdir(canonical):
                results.append(TargetResult(agent, p, "", "skipped", error="the shared copy is missing"))
                continue
            try:
                r = _write_link(ctx, agent, p)
            except OSError as exc:
                r = TargetResult(agent, p, _kind(p), "failed", error=str(exc))
            if p in moved:
                r = _replace(r, action=r.action if r.action == "failed" else "replaced", backups=(*r.backups, moved[p]))
            results.append(r)
        entries = _manifest_targets(results, manifest, env, adopting=manifest is None)
        # a failed shared copy without a manifest stays as it was (a legacy stamp stays adoptable)
        if manifest is not None or (results[0].action != "failed" and any(r.action != "failed" for r in results)):
            _write_manifest(state_dir, _new_manifest(
                ctx, hosts=hosts, config_path=config_path, config_mtime=config_mtime, auto_update=auto_update,
                agents=recorded, agents_mode=_agents_mode(agents, (manifest or {}).get("agents_mode")),
                targets=entries, results=results, now=now))
    log.info("skill: install %s", " ".join(f"{r.agent}={r.action}" for r in results))
    backup_dir = ctx.backup.root if ctx.backup.saved else None
    return InstallResult(env=env, canonical=canonical, version=runfleet.__version__, targets=tuple(results),
                         rendered=rendered, readers=readers, agents=recorded, backup_dir=backup_dir,
                         backups=tuple(ctx.backup.saved), notes=tuple(ctx.notes), auto_update=auto_update,
                         manifest_path=_manifest_path(state_dir), swept=tuple(swept),
                         permission_hint=permission_hint(env), older_backups=ctx.backup.older)


def _replace(t: TargetResult, **changes: Any) -> TargetResult:
    from dataclasses import replace

    return replace(t, **changes)


def _kind(p: Path) -> str:
    if not _exists(p):
        return ""
    if os.path.islink(p):
        return "symlink"
    if _is_link(p):
        return "junction"
    return "copy" if p.is_dir() else "file"


def _manifest_targets(results: Sequence[TargetResult], manifest: Mapping[str, Any] | None,
                      env: Env, *, adopting: bool = False) -> list[dict[str, Any]]:
    """The targets to record: every one that worked; for a failed one, the old entry while what it names is still
    ours (a working old link, an old copy). Adopting a pre-manifest install, a copy of it that could not be
    refreshed is recorded as `pending` (its stamp has no install id yet): later updates retry it, uninstall removes
    it."""
    old = {t.get("path"): t for t in (manifest or {}).get("targets", []) if isinstance(t, dict)}
    canonical = canonical_dir(env)
    out: list[dict[str, Any]] = []
    for r in results:
        if r.action not in ("failed", "skipped", "missing", "foreign"):
            out.append(r.manifest_entry())
        elif (prev := old.get(str(r.path))) is not None and _exists(r.path):
            if _points_to(r.path, canonical) or _owned_copy(r.path, manifest) is not None:
                out.append(dict(prev))
        elif adopting and r.action == "failed" and r.agent != "shared" and read_stamp(r.path) is not None:
            out.append({"agent": r.agent, "path": str(r.path), "kind": _kind(r.path), "link_target": None,
                        "files": None, "pending": True})
    return out


def _legacy_targets(env: Env) -> list[tuple[str, Path]]:
    """A pre-manifest install: the stamped shared copy plus every agent link path that is a link to it or a stamped
    copy."""
    canonical = canonical_dir(env)
    if read_stamp(canonical) is None:
        return []
    out: list[tuple[str, Path]] = [("shared", canonical)]
    for spec in AGENTS:
        p = spec.link_path(env)
        if p is None or _same(p.parent, canonical.parent):
            continue
        if _points_to(p, canonical) or read_stamp(p) is not None:
            out.append((spec.id, p))
    return out


def update(env: Env, *, state_dir: Path, host_names: Sequence[str] | None, config_path: str | None,
           config_mtime: float | None, if_installed: bool = False, now: datetime,
           config: ConfigResolver | None = None, add_new_agents: bool = False) -> UpdateResult:
    """Refresh exactly the targets the manifest lists with this version's skill (host names: `host_names`, or the
    recorded ones when None; `config`, when given, is asked under the lock instead, with the config path recorded
    then). A pre-manifest install is adopted. Never recreates what the user removed: a missing shared copy is
    reported (install puts it back), a missing link is left missing. `if_installed` (installers, `runfleet update`):
    nothing installed → silent; automatic refresh off → only a note; a newer RunFleet wrote it → left as is (an
    explicit update downgrades it and says so). `add_new_agents` (explicit `skill update`): an install made with
    --agent auto|all also links the agents detected since."""
    with _lock_or_raise(state_dir):
        return _update_locked(env, state_dir=state_dir, host_names=host_names, config_path=config_path,
                              config_mtime=config_mtime, if_installed=if_installed, now=now, config=config,
                              add_new_agents=add_new_agents)


def _update_locked(env: Env, *, state_dir: Path, host_names: Sequence[str] | None, config_path: str | None,
                   config_mtime: float | None, if_installed: bool, now: datetime, config: ConfigResolver | None = None,
                   add_new_agents: bool = False, log_result: bool = True) -> UpdateResult:
    version = runfleet.__version__
    manifest = read_manifest(state_dir)
    canonical = canonical_dir(env)
    entries_by_path: dict[str, Mapping[str, Any]] = {}
    if manifest is None:
        legacy = _legacy_targets(env)
        if not legacy:
            return UpdateResult(env=env, state="not_installed", version=version, previous_version=None,
                                if_installed=if_installed)
        stamp = read_stamp(canonical) or {}
        planned = [(a, p) for a, p in legacy if a != "shared"]
        prev_version = str(stamp.get("version")) if stamp.get("version") else None
        auto_update, agents, agents_mode = True, [a for a, _ in planned], None
        install_id = uuid.uuid4().hex
    else:
        planned = [(str(t.get("agent")), Path(str(t.get("path")))) for t in manifest["targets"]
                   if isinstance(t, dict) and t.get("path") and t.get("agent") != "shared"]
        prev_version = str(manifest.get("version")) if manifest.get("version") else None
        auto_update = bool(manifest.get("auto_update", True))
        if if_installed and not auto_update:
            return UpdateResult(env=env, state="disabled", version=version, previous_version=prev_version,
                                notes=(AUTO_OFF_NOTE,), if_installed=if_installed)
        agents = [str(a) for a in manifest.get("agents") or []]
        agents_mode = str(manifest["agents_mode"]) if manifest.get("agents_mode") in _MODES else None
        install_id = str(manifest.get("install_id") or uuid.uuid4().hex)
        entries_by_path = {str(t.get("path")): t for t in manifest["targets"] if isinstance(t, dict)}
    downgrade = prev_version is not None and version_newer(prev_version, version) is True
    if downgrade and if_installed:
        return UpdateResult(env=env, state="newer_installed", version=version, previous_version=prev_version,
                            notes=(f"left at {prev_version}, installed by a newer RunFleet "
                                   f"(`runfleet skill update` would downgrade it to {version})",),
                            if_installed=if_installed)
    if config is not None:                   # read now, under the lock: the recorded config cannot change meanwhile
        host_names, config_path, config_mtime = config(
            str(manifest["config_path"]) if manifest and isinstance(manifest.get("config_path"), str) else None)
    if manifest is None:
        hosts = clean_hosts(host_names or [])
    else:
        hosts = clean_hosts(host_names if host_names is not None else [str(h) for h in manifest.get("hosts") or []])
        if host_names is None and config_path is None:          # keep the recorded config with the recorded hosts
            config_path = manifest.get("config_path")
            config_mtime = manifest.get("config_mtime") if config_mtime is None else config_mtime
    if not os.path.isdir(canonical) or _owned_copy(canonical, manifest) is None:
        notes = (f"{_show(env, canonical)} is no longer RunFleet's copy; left alone",) if _exists(canonical) else ()
        return UpdateResult(env=env, state="canonical_missing", version=version, previous_version=prev_version,
                            notes=notes, if_installed=if_installed)
    rendered, files = _bundle(hosts)
    ctx = _ctx(env, state_dir, files, manifest, install_id, now)
    ctx.notes += rendered.notes
    if downgrade:
        ctx.notes.append(f"downgrading skill {prev_version} → {version}")
    planned = [(a, p) for a, p in planned if not _aliases_canonical(p, canonical)]
    swept = _sweep_all([canonical, *(p for _, p in planned)], ctx.backup)
    results: list[TargetResult] = []
    try:
        results.append(_write_copy(ctx, "shared", canonical))
    except OSError as exc:
        results.append(TargetResult("shared", canonical, "copy", "failed", error=str(exc)))
    for agent, p in planned:
        entry = entries_by_path.get(str(p), {})
        try:
            if not _exists(p):
                results.append(TargetResult(agent, p, "", "missing", error="removed; not recreated"))
            elif _points_to(p, canonical):
                results.append(TargetResult(agent, p, _kind(p), "unchanged", link_target=os.path.realpath(p)))
            elif _owned_copy(p, manifest) is not None:
                # a tracked copy (no link possible there) stays a copy; a legacy copy becomes a link where possible
                keep_copy = entry.get("kind") == "copy" and not entry.get("pending")
                results.append(_write_copy(ctx, agent, p) if keep_copy else _write_link(ctx, agent, p))
            else:
                results.append(TargetResult(agent, p, _kind(p), "foreign",
                                            error="no longer RunFleet's (changed by you); left alone"))
        except OSError as exc:
            results.append(TargetResult(agent, p, _kind(p), "failed", error=str(exc)))
    added = _link_new_agents(ctx, agents, agents_mode, planned, results) \
        if add_new_agents and os.path.isdir(canonical) else []
    agents += added
    changed = any(r.action in ("created", "updated", "replaced") for r in results) or manifest is None or (
        manifest.get("version") != version or manifest.get("bundle_sha256") != bundle_sha256()
        or manifest.get("hosts_sha256") != hosts_sha256(hosts)) or bool(added)
    entries = _manifest_targets(results, manifest, env, adopting=manifest is None)
    if manifest is not None or results[0].action != "failed":     # a failed adoption leaves the legacy install
        _write_manifest(state_dir, _new_manifest(ctx, hosts=hosts, config_path=config_path,
                                                 config_mtime=config_mtime, auto_update=auto_update, agents=agents,
                                                 agents_mode=agents_mode, targets=entries, results=results, now=now))
    if manifest is None and results[0].action != "failed":
        ctx.notes.append("adopted an install made by an older RunFleet")
    if swept:
        ctx.notes.append(f"removed {len(swept)} leftover(s) of interrupted installs")
    if log_result:
        log.info("skill: update %s", " ".join(f"{r.agent}={r.action}" for r in results))
    return UpdateResult(env=env, state="updated" if changed else "unchanged", version=version,
                        previous_version=prev_version, targets=tuple(results),
                        backup_dir=ctx.backup.root if ctx.backup.saved else None, backups=tuple(ctx.backup.saved),
                        notes=tuple(ctx.notes), if_installed=if_installed, downgrade=downgrade,
                        older_backups=ctx.backup.older)


def _link_new_agents(ctx: _Ctx, agents: Sequence[str], mode: str | None, planned: Sequence[tuple[str, Path]],
                     results: list[TargetResult]) -> list[str]:
    """Explicit `skill update` after an install with --agent auto|all: the agents chosen that way now get a link where
    they look for it now, unless that path was ever recorded (installed since, or $CLAUDE_CONFIG_DIR / $COPILOT_HOME
    set since; a link the user removed is not recreated). Results appended; returns the new agent ids to record."""
    if mode not in ("auto", "all"):
        return []
    env, canonical = ctx.env, ctx.canonical
    history = (ctx.manifest or {}).get("linked_paths")
    seen = set(_str_list(history)) if isinstance(history, list) else None
    chosen = select_agents(env, mode)
    have = {p for _, p in planned}
    failed: set[str] = set()
    linked: list[str] = []
    for agent, p in _plan_links(env, chosen):
        if p in have or _aliases_canonical(p, canonical):
            continue
        if (str(p) in seen) if seen is not None else (agent in agents):   # no history yet: by agent id
            continue
        why = _blocker(ctx, agent, p, link=True)
        if why:
            results.append(TargetResult(agent, p, _kind(p), "foreign", error=f"{why} is in the way, not RunFleet's; "
                                        f"left alone (`runfleet skill install --agent {agent} --force` moves it)"))
            failed.add(agent)
            continue
        try:
            results.append(_write_link(ctx, agent, p))
            linked.append(agent)
        except OSError as exc:
            results.append(TargetResult(agent, p, _kind(p), "failed", error=str(exc)))
            failed.add(agent)
    if linked:
        ctx.notes.append("linked the skill for agents found (or moved) since the install: "
                         + ", ".join(_label(a) for a in linked))
    return [a for a in chosen if a not in agents and a not in failed]


def _stray_candidates(env: Env, exclude: Sequence[Path]) -> list[tuple[str, Path, bool]]:
    """(agent, path, stale) for `runfleet` entries in the legacy skills roots of every agent, minus the shared copy
    itself (a root that is the shared skills dir through a link) and the paths in `exclude`; `stale`: a root the
    agent no longer reads (its home was moved by $CLAUDE_CONFIG_DIR / $CODEX_HOME)."""
    canonical = canonical_dir(env)
    out: list[tuple[str, Path, bool]] = []
    for spec in AGENTS:
        unread = spec.unread_dirs(env)
        for root in spec.legacy_dirs(env):
            p = root / SKILL_NAME
            if not _exists(p) or _same(root, canonical.parent) or p in exclude or any(p == q for _, q, _s in out):
                continue
            out.append((spec.id, p, root in unread))
    return out


def uninstall(env: Env, *, state_dir: Path, dry_run: bool = False, now: datetime) -> UninstallResult:
    """Remove what RunFleet wrote: the shared copy, links to it, tracked copies (user files saved first), stray
    stamped copies and links to ours in other agents' skills roots, leftovers, the manifest, the lock file and the
    skills dirs it created while they are empty. Anything else named runfleet is left alone; when the shared copy
    is not RunFleet's, no link to it is either. Never edits agent settings (the follow-ups say what to remove by
    hand)."""
    lock_file = state_dir / LOCK_FILE
    lock_existed = _exists(lock_file)
    with _lock_or_raise(state_dir):
        manifest = read_manifest(state_dir)
        canonical = canonical_dir(env)
        shared_ours = _shared_is_ours(env, manifest)
        backup = _Backup(state_dir, now, dry_run=dry_run)
        tracked: list[tuple[str, Path]] = []                 # ours by the manifest's install id (or legacy stamp)
        for t in (manifest or {}).get("targets", []):
            if isinstance(t, dict) and t.get("path") and Path(str(t["path"])) not in {q for _, q in tracked}:
                tracked.append((str(t.get("agent")), Path(str(t["path"]))))
        tracked += [(a, p) for a, p in _legacy_targets(env) if p not in {q for _, q in tracked}]
        if canonical not in {q for _, q in tracked} and _exists(canonical):
            tracked.append(("shared", canonical))
        if shared_ours:                      # links to ours wherever an agent looks (incl. explicit-only agents)
            for spec in AGENTS:
                p = spec.link_path(env)
                if p is not None and p not in {q for _, q in tracked} and _points_to(p, canonical):
                    tracked.append((spec.id, p))
        strays = _stray_candidates(env, [p for _, p in tracked])
        candidates = [(a, p, False) for a, p in tracked] + [(a, p, True) for a, p, _stale in strays]
        candidates.sort(key=lambda c: c[1] == canonical)     # links first: they resolve only while the copy is there
        baseline = _packaged_hashes()
        removed: list[TargetResult] = []
        kept: list[TargetResult] = []
        for agent, p, stray in candidates:
            if not _exists(p):
                continue
            try:
                if _is_link(p):
                    if _is_canonical_path(p, canonical) or not _points_to(p, canonical) or not shared_ours:
                        if not stray:
                            why = "a link to elsewhere" if _is_canonical_path(p, canonical) or not \
                                _points_to(p, canonical) else "a link to a shared copy that is not RunFleet's"
                            kept.append(TargetResult(agent, p, _kind(p), "foreign", error=why))
                        continue
                    kind = _kind(p)
                    if not dry_run:
                        _remove(p)
                    removed.append(TargetResult(agent, p, kind, "removed"))
                    continue
                if p != canonical and _same(p, canonical):
                    continue                 # the shared copy seen through a linked skills dir
                stamp = read_stamp(p) if stray else _owned_copy(p, manifest)
                if stamp is None:
                    if not stray:
                        kept.append(TargetResult(agent, p, _kind(p), "foreign", error="not installed by RunFleet"))
                    continue
                saved = _save_files(backup, agent, p, _known_files(manifest, p, stamp), baseline)
                if not dry_run:
                    _remove(p)
                removed.append(TargetResult(agent, p, "copy", "removed", backups=tuple(saved)))
            except OSError as exc:
                kept.append(TargetResult(agent, p, _kind(p), "failed", error=str(exc)))
        swept: list[Path] = []
        for parent in dict.fromkeys([canonical.parent, *(p.parent for _, p, _s in candidates)]):
            if not dry_run:
                swept += _sweep(parent, backup)
                continue
            swept += _leftovers(parent)
        manifest_file = _manifest_path(state_dir)
        failed = any(t.action == "failed" for t in kept)
        manifest_removed = _exists(manifest_file) and not failed       # a failed removal keeps the record for a retry
        removed_dirs: list[Path] = []
        if manifest_removed and not dry_run:
            manifest_file.unlink()
            removed_dirs = _remove_created_dirs(_str_list((manifest or {}).get("created_dirs")))
        drop_lock = not failed and (not dry_run or not lock_existed)
        if drop_lock and not platform.IS_WINDOWS:       # deleted while held: see _acquire
            with contextlib.suppress(OSError):
                lock_file.unlink()
    if drop_lock and platform.IS_WINDOWS:                # an open file cannot be deleted there: after closing it
        with contextlib.suppress(OSError):
            lock_file.unlink()
    log.info("skill: uninstall%s removed %d, kept %d", " (dry run)" if dry_run else "", len(removed), len(kept))
    return UninstallResult(env=env, dry_run=dry_run, removed=tuple(removed), kept=tuple(kept),
                           backup_dir=backup.root if backup.saved else None, backups=tuple(backup.saved),
                           swept=tuple(swept), manifest_removed=manifest_removed, follow_ups=follow_ups(env),
                           older_backups=backup.older, removed_dirs=tuple(removed_dirs))


def _remove_created_dirs(created: Sequence[str]) -> list[Path]:
    """The dirs an install created (their real paths, deepest first), each only while it is an empty directory
    reached without any link on the way (a parent replaced by a link since: kept)."""
    out: list[Path] = []
    for d in sorted(dict.fromkeys(created), key=lambda x: len(Path(x).parts), reverse=True):
        p = Path(d)
        try:
            if p.is_absolute() and _real(p) == os.path.normcase(os.path.normpath(p)) and p.is_dir() \
                    and not _is_link(p) and not any(p.iterdir()):
                p.rmdir()
                out.append(p)
        except OSError as exc:
            diagnostic(logging.INFO, "skill: kept %s: %s", p, exc)
    return out


# ---------------------------------------------------------------- status
def _failure_text(manifest: Mapping[str, Any] | None) -> str | None:
    at = (manifest or {}).get("last_failure_at")
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return None
    try:
        when = datetime.fromtimestamp(at, UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (OverflowError, OSError, ValueError):
        when = "?"
    error = (manifest or {}).get("last_error")
    return f"at {when}" + (f": {error}" if isinstance(error, str) and error else "")


def status(env: Env, *, state_dir: Path) -> StatusReport:
    """What is installed where, for every agent: detected, covered by the shared copy / a link / a copy, broken links,
    edited files, duplicates in other skills roots. Read-only."""
    manifest = read_manifest(state_dir)
    canonical = canonical_dir(env)
    stamp = read_stamp(canonical)
    if not _exists(canonical):
        canonical_state = "missing"
    elif _owned_copy(canonical, manifest) is not None:
        canonical_state = "ok"
    else:
        canonical_state = "foreign"
    installed = canonical_state == "ok"
    version = str(stamp.get("version")) if stamp and stamp.get("version") else None
    up_to_date = bool(installed and stamp and version == runfleet.__version__
                      and stamp.get("bundle_sha256") == bundle_sha256())
    edited: tuple[str, ...] = ()
    if installed and stamp is not None:
        known = _known_files(manifest, canonical, stamp)
        edited = tuple(_user_files(canonical, known)) if known is not None else ()
    in_manifest = {str(t.get("path")) for t in (manifest or {}).get("targets", []) if isinstance(t, dict)}
    agents: list[AgentStatus] = []
    broken: list[Path] = []
    for spec in AGENTS:
        detected = _detected(env, spec)
        p = spec.link_path(env)
        if p is None:
            covered = "shared" if spec.reads_shared and installed else "none"
            agents.append(AgentStatus(spec.id, spec.label, detected, covered, None, "ok" if installed else "missing"))
            continue
        tracked = str(p) in in_manifest
        if _same(p.parent, canonical.parent):
            covered, state = ("shared", "ok") if installed else ("none", "missing")
        elif _points_to(p, canonical):
            if canonical_state == "foreign":
                covered, state = "none", "foreign"
            else:
                covered, state = ("link", "ok") if os.path.isdir(p) else ("none", "broken")
        elif _is_link(p) and not os.path.exists(p):
            covered, state = "none", "broken"
        elif _exists(p) and read_stamp(p) is not None:
            s = read_stamp(p) or {}
            covered = "copy"
            state = "ok" if s.get("version") == runfleet.__version__ and \
                s.get("bundle_sha256") == bundle_sha256() else "stale"
        elif _exists(p):
            covered, state = "none", "foreign"
        else:
            covered, state = "none", "missing"
        if state == "broken":
            broken.append(p)
        agents.append(AgentStatus(spec.id, spec.label, detected, covered, p, state, tracked))
    strays: list[Stray] = []
    for agent, p, stale in _stray_candidates(env, [q for q in (s.path for s in agents) if q is not None]):
        if _is_link(p):
            ours = canonical_state != "foreign" and _points_to(p, canonical)
            kind = ("stale_link" if stale else "link_to_ours") if ours else "foreign"
        else:
            kind = "stamped_copy" if read_stamp(p) is not None else "foreign"
        strays.append(Stray(agent, p, kind))
    auto = bool(manifest.get("auto_update", True)) if manifest else (True if installed else None)
    hosts = tuple(str(h) for h in (manifest or {}).get("hosts") or [])
    last = (manifest or {}).get("last_backup")
    last_backup = Path(last) if isinstance(last, str) and os.path.isdir(last) else None
    return StatusReport(env=env, installed=installed, canonical=canonical, canonical_state=canonical_state,
                        version=version, running_version=runfleet.__version__, up_to_date=up_to_date, hosts=hosts,
                        auto_update=auto, edited=edited, agents=tuple(agents), strays=tuple(strays),
                        broken_links=tuple(broken), manifest_path=_manifest_path(state_dir),
                        legacy=installed and manifest is None, last_backup=last_backup,
                        last_failure=_failure_text(manifest) if installed else None)


# ---------------------------------------------------------------- automatic refresh
def refresh_reason(env: Env, *, state_dir: Path, config_mtime: float | None, config_path: str | None = None,
                   now: float | None = None) -> str | None:
    """Why the installed skill should be refreshed now, cheaply (one small JSON read and a stat): "version A → B"
    (never to an older or unparsable version), "bundle" (same version, different packaged text, and this install
    is the one that wrote it: computed only then), "config" (the recorded config file changed: the caller still
    compares the host names). None: nothing to do, not installed, automatic refresh off, a refresh failed less
    than BACKOFF_SECONDS before `now` (epoch seconds; default the current time), or the shared copy was removed.
    `config_path`: the config the caller loaded and stat'ed (None: the default one); any other than the recorded
    one is no reason. cli.skill_refresh_due mirrors this without importing the module (a test pins the two)."""
    try:
        with open(_manifest_path(state_dir), "rb") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("auto_update") is not True:
        return None
    if in_backoff(data, time.time() if now is None else now):
        return None
    if not os.path.isdir(canonical_dir(env)):
        return None
    installed, running = str(data.get("version") or ""), runfleet.__version__
    if installed != running:
        newer = version_newer(running, installed)
        if newer:
            return f"version {installed} → {running}"
        if newer is None or _version_key(running) != _version_key(installed):
            return None                       # older, or unsure: never downgrade
    if data.get("writer") in (None, install_identity()) and data.get("bundle_sha256") != bundle_sha256():
        return "bundle"
    if config_mtime != data.get("config_mtime") and config_path == data.get("config_path"):
        return "config"
    return None


def auto_refresh(env: Env, *, state_dir: Path, config_mtime: float | None, load_host_names: Callable[[], Sequence[str]],
                 now: datetime, config_path: str | None = None) -> str | None:
    """The per-invocation refresh: when refresh_reason says so, refresh the recorded targets under a non-blocking lock
    (skipped when another one holds it); host names are reloaded only when the recorded config (`config_path`, None:
    the default one; `config_mtime` and `load_host_names` are of it) changed. Never adds links for agents found
    since. Never raises, never prints; logs exactly one line when it did something (the diagnostics of the file work
    folded in) and returns it (None: nothing done). A failure is recorded in the manifest while the lock is still
    held, so the next automatic try waits BACKOFF_SECONDS."""
    diagnostics: list[str] = []
    _DIAG.sink = diagnostics
    try:
        line = _auto_refresh(env, state_dir=state_dir, config_mtime=config_mtime, load_host_names=load_host_names,
                             now=now, config_path=config_path)
    except Exception as exc:                 # noqa: BLE001 - a refresh must never break the command it rides on
        line = f"skill: refresh failed: {exc}"
    finally:
        _DIAG.sink = None
    if line is None:
        if diagnostics:
            log.info("skill: %s", "; ".join(diagnostics))
        return None
    full = line + ("" if not diagnostics else " [" + "; ".join(diagnostics) + "]")
    log.log(logging.WARNING if line.startswith("skill: refresh failed") else logging.INFO, "%s", full)
    # None: nothing was refreshed (busy, or it raised); a line: refreshed, or the targets that failed
    return None if line.startswith(("skill: refresh failed: ", "skill: refresh skipped")) else full


def _auto_refresh(env: Env, *, state_dir: Path, config_mtime: float | None,
                  load_host_names: Callable[[], Sequence[str]], now: datetime, config_path: str | None) -> str | None:
    at = now.timestamp()
    reason = refresh_reason(env, state_dir=state_dir, config_mtime=config_mtime, config_path=config_path, now=at)
    if reason is None:
        return None
    with _locked(state_dir, wait=0) as held:
        if not held:
            return "skill: refresh skipped, another skill command is running"
        try:
            # decided again under the lock: another process may have upgraded (or turned it off) meanwhile
            reason = refresh_reason(env, state_dir=state_dir, config_mtime=config_mtime, config_path=config_path,
                                    now=at)
            manifest = read_manifest(state_dir)
            if reason is None or manifest is None:
                return None
            same_config = config_path == manifest.get("config_path")
            hosts: list[str] | None = None
            if same_config and config_mtime != manifest.get("config_mtime"):
                loaded = clean_hosts(load_host_names())
                if hosts_sha256(loaded) != manifest.get("hosts_sha256"):
                    hosts = loaded
                elif reason == "config":                       # touched, same hosts: remember the new mtime
                    manifest["config_mtime"] = config_mtime
                    _write_manifest(state_dir, manifest)
                    return None
            cfg_mtime = config_mtime if same_config else manifest.get("config_mtime")
            result = _update_locked(env, state_dir=state_dir, host_names=hosts if hosts is not None else
                                    [str(h) for h in manifest.get("hosts") or []],
                                    config_path=manifest.get("config_path"), config_mtime=cfg_mtime,
                                    if_installed=True, now=now, log_result=False)
        except Exception as exc:
            _note_failure(state_dir, at, exc)            # still under the lock: no other install can be in between
            raise
    failure = result.failure()
    if failure is not None:
        return f"skill: refresh failed ({reason}): {failure}; next automatic try in an hour at the earliest"
    return f"skill: refreshed ({reason}): " + (result.summary() or result.state)


def _note_failure(state_dir: Path, at: float, exc: BaseException) -> None:
    """Record a refresh that raised (the caller holds the lock), so the next automatic try waits too (best effort)."""
    try:
        data = read_manifest(state_dir)
        if data is not None:
            data["last_failure_at"] = at
            data["last_error"] = str(exc)[:300]
            _write_manifest(state_dir, data)
    except Exception as exc2:                # noqa: BLE001
        diagnostic(logging.DEBUG, "skill: could not record the failed refresh: %s", exc2)
