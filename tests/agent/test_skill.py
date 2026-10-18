"""[WP4] agent.skill on disk: install / update / uninstall / status / automatic refresh into a temporary HOME via an
injected Env (never the real one): agent detection, shared copy + links (relative symlink, junction, copy), manifest
and stamps, user files saved before anything is replaced, foreign targets refused or moved aside, partial failures,
the lock, leftovers, legacy adoption, the refresh check and its cost. Content checks of the packaged SKILL.md live in
test_skill_content.py; the renders here use a synthetic template."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import stat
import statistics
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import runfleet
from runfleet.agent import skill
from runfleet.util import platform as rf_platform
from tests.conftest import WINDOWS, is_root

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ["k2", "vega"]
NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)
TEMPLATE = "---\nname: runfleet\ndescription: runfleet squeue GPU queue status.\nmetadata:\n  x: \"1\"\n---\n# body\n"


# ---------------------------------------------------------------- helpers
def _can_symlink(tmp: Path) -> bool:
    try:
        os.symlink(tmp, tmp / "probe-link", target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    (tmp / "probe-link").unlink()
    return True


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def symlinks(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("no symlink permission here")


def make_env(home: Path, *, environ: dict[str, str] | None = None, which: tuple[str, ...] = (),
             windows: bool = False) -> skill.Env:
    return skill.Env(home=home, environ=dict(environ or {}), is_windows=windows,
                     which=lambda name: f"/usr/bin/{name}" if name in which else None)


def claude_env(home: Path, **kw: Any) -> skill.Env:
    (home / ".claude").mkdir(exist_ok=True)
    return make_env(home, **kw)


def do_install(env: skill.Env, state: Path, **kw: Any) -> skill.InstallResult:
    kw.setdefault("host_names", HOSTS)
    kw.setdefault("config_path", None)
    kw.setdefault("config_mtime", None)
    return skill.install(env, state_dir=state, now=NOW, **kw)


def do_update(env: skill.Env, state: Path, **kw: Any) -> skill.UpdateResult:
    kw.setdefault("host_names", None)
    kw.setdefault("config_path", None)
    kw.setdefault("config_mtime", None)
    return skill.update(env, state_dir=state, now=NOW, **kw)


def manifest(state: Path) -> dict[str, Any]:
    data = json.loads((state / "skill.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def edit_manifest(state: Path, **changes: Any) -> None:
    data = manifest(state)
    data.update(changes)
    (state / "skill.json").write_text(json.dumps(data), encoding="utf-8")


def canonical(home: Path) -> Path:
    return home / ".agents" / "skills" / "runfleet"


def claude_link(home: Path) -> Path:
    return home / ".claude" / "skills" / "runfleet"


def files_of(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}


def backup_files(state: Path) -> dict[str, str]:
    root = state / "skill-backup"
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix().split("/", 1)[1]: p.read_text(encoding="utf-8")
            for p in root.rglob("*") if p.is_file()}


def fake_junction(calls: list[tuple[Path, Path]]) -> Callable[[Path, Path], None]:
    """Stands in for _create_junction on a POSIX test machine: an absolute symlink (as a junction is absolute)."""
    def create(target: Path, link: Path) -> None:
        calls.append((target, link))
        os.symlink(target, link, target_is_directory=True)
    return create


# ---------------------------------------------------------------- packaged files, render, versions
def test_packaged_files_are_found_and_hash_is_stable() -> None:
    files = skill.packaged_files()
    assert "SKILL.md" in files and all(not k.startswith(".") for k in files)
    assert skill.bundle_sha256() == skill.bundle_sha256() and len(skill.bundle_sha256()) == 64


def test_render_prefixes_host_names_and_keeps_the_body() -> None:
    r = skill.render(["k2", "vega"], TEMPLATE)
    assert r.description == "k2 vega runfleet squeue GPU queue status." and r.hosts == ("k2", "vega")
    assert r.text == TEMPLATE.replace("description: runfleet", "description: k2 vega runfleet")


def test_render_dedupes_and_skips_unsafe_names() -> None:
    r = skill.render(["k2", "GPU", "RunFleet", "K2", " lyra ", "<x>", "a:b", "-d", "q'x", "tab\tx", ""], TEMPLATE)
    assert r.hosts == ("k2", "lyra")
    assert len(r.notes) == 5 and all("unsupported characters" in n for n in r.notes)


def test_render_caps_the_host_prefix() -> None:
    many = [f"node{i:03d}x" for i in range(100)]
    r = skill.render(many, TEMPLATE)
    assert len(" ".join(r.hosts)) <= skill.MAX_HOST_PREFIX < len(" ".join([*r.hosts, many[len(r.hosts)]]))
    assert r.hosts == tuple(many[:len(r.hosts)]) and len(r.notes) == 1 and many[-1] in r.notes[0]
    long_base = TEMPLATE.replace("status.", "x" * 950)
    capped = skill.render(many, long_base)
    assert 0 < len(capped.hosts) < len(r.hosts) and len(capped.description) <= skill.MAX_DESCRIPTION


def test_render_handles_crlf_and_rejects_a_template_without_description() -> None:
    assert skill.render(["k2"], TEMPLATE.replace("\n", "\r\n")).description.startswith("k2 runfleet")
    with pytest.raises(skill.SkillInstallError):
        skill.render(["k2"], "---\nname: x\n---\nbody\n")


@pytest.mark.parametrize(("running", "installed", "newer"), [
    ("0.8.1", "0.8.0", True), ("0.8.0", "0.8.1", False), ("0.8.0", "0.8.0", False), ("0.9", "0.8.9", True),
    ("0.8.0", "0.8.0.dev3", True), ("0.8.0rc1", "0.8.0.dev9", True), ("0.8.0", "0.8.0rc2", True),
    ("0.8.0.post1", "0.8.0", True), ("0.8.0.dev2", "0.8.0.dev10", False), ("0.8.0+g12ab", "0.8.0", False),
    ("garbage", "0.8.0", None), ("0.8.0", "", None),
])
def test_version_newer(running: str, installed: str, newer: bool | None) -> None:
    assert skill.version_newer(running, installed) is newer


# ---------------------------------------------------------------- Env and detection
def test_env_current_and_path_vars(home: Path) -> None:
    cur = skill.Env.current()
    assert cur.home == Path.home() and cur.is_windows == rf_platform.IS_WINDOWS and cur.which("definitely-no-such")\
        is None
    env = make_env(home, environ={"CLAUDE_CONFIG_DIR": "~/cc", "CODEX_HOME": " ", "X": "/abs"})
    assert env.path_var("CLAUDE_CONFIG_DIR") == home / "cc"
    assert env.path_var("CODEX_HOME") is None and env.path_var("X") == Path("/abs") and env.path_var("NOPE") is None
    assert env.config_home() == home / ".config"


def test_detection_matrix(home: Path, tmp_path: Path) -> None:
    assert skill.select_agents(make_env(home), "auto") == ()
    assert skill.select_agents(make_env(home, environ={"CLAUDE_CONFIG_DIR": str(tmp_path / "cc")}), "auto") == \
        ("claude",)
    found = skill.Env(home=home, environ={}, is_windows=False,
                      which=lambda n: str(tmp_path / "no-such" / n) if n in ("claude", "codex", "pi") else None)
    assert skill.select_agents(found, "auto") == ("claude", "codex")         # a bare `pi` is not enough (E7)
    (home / ".copilot").mkdir()
    (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
    env = claude_env(home)
    auto = skill.select_agents(env, "auto")
    assert "claude" in auto and "copilot" in auto and "gemini" in auto and "antigravity" not in auto
    assert "antigravity" not in skill.select_agents(env, "all")
    assert skill.select_agents(env, "antigravity,claude") == ("claude", "antigravity")
    assert skill.select_agents(env, ["codex", " pi "]) == ("codex", "pi")
    with pytest.raises(ValueError, match="unknown agent nope"):
        skill.select_agents(env, "claude,nope")
    with pytest.raises(ValueError):
        do_install(env, tmp_path / "s", agents="vscode")


def test_link_paths_only_where_needed(home: Path, tmp_path: Path) -> None:
    env = make_env(home)
    links = {a.id: a.link_path(env) for a in skill.AGENTS}
    assert links["claude"] == home / ".claude" / "skills" / "runfleet"
    assert links["copilot"] is None and links["codex"] is None and links["pi"] is None
    assert links["antigravity"] == home / ".gemini" / "antigravity-cli" / "skills" / "runfleet"
    assert "copilot" in skill.shared_readers(env) and "claude" not in skill.shared_readers(env)
    env2 = make_env(home, environ={"COPILOT_HOME": str(tmp_path / "cp"), "CLAUDE_CONFIG_DIR": str(tmp_path / "cc")})
    assert skill._BY_ID["copilot"].link_path(env2) == tmp_path / "cp" / "skills" / "runfleet"
    assert skill._BY_ID["claude"].link_path(env2) == tmp_path / "cc" / "skills" / "runfleet"
    assert "copilot" not in skill.shared_readers(env2)
    assert home / ".claude" / "skills" in skill._BY_ID["claude"].legacy_dirs(env2)


# ---------------------------------------------------------------- install
def test_print_only_writes_nothing(home: Path, state: Path) -> None:
    r = do_install(claude_env(home), state, print_only=True)
    assert r.printed_only and r.exit_code == 0 and r.lines()[0] == "---"
    assert [t.action for t in r.targets] == ["planned", "planned"]
    assert sorted(p.name for p in home.iterdir()) == [".claude"] and not state.exists()


def test_fresh_install_layout_relative_symlink_manifest(home: Path, state: Path, symlinks: None) -> None:
    (home / ".codex").mkdir()
    env = claude_env(home)
    r = do_install(env, state, config_path="/c.toml", config_mtime=12.5)
    shared, link = canonical(home), claude_link(home)
    assert r.exit_code == 0 and [(t.agent, t.kind, t.action) for t in r.targets] == \
        [("shared", "copy", "created"), ("claude", "symlink", "created")]
    assert shared.is_dir() and not shared.is_symlink()
    assert files_of(shared) == set(skill.packaged_files()) | {skill.STAMP_FILE}
    assert "k2 vega" in (shared / "SKILL.md").read_text(encoding="utf-8")
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.resolve() == shared.resolve()
    stamp = json.loads((shared / skill.STAMP_FILE).read_text(encoding="utf-8"))
    m = manifest(state)
    assert stamp["install_id"] == m["install_id"] and stamp["version"] == runfleet.__version__
    assert stamp["bundle_sha256"] == m["bundle_sha256"] == skill.bundle_sha256()
    assert set(m) == {"schema", "install_id", "version", "bundle_sha256", "writer", "hosts", "hosts_sha256",
                      "config_path", "config_mtime", "auto_update", "agents", "agents_mode", "updated_at", "targets",
                      "created_dirs", "linked_paths"}
    assert m["linked_paths"] == [str(link)]
    assert m["writer"] == skill.install_identity() and m["agents_mode"] == "auto"
    assert m["created_dirs"] == [str(shared.parent), str(home / ".agents"), str(link.parent)]
    assert m["schema"] == 1 and m["hosts"] == HOSTS and m["config_path"] == "/c.toml" and m["config_mtime"] == 12.5
    assert m["auto_update"] is True and "claude" in m["agents"] and "codex" in m["agents"]
    assert m["updated_at"] == "2026-10-18T12:00:00+00:00"
    shared_entry, claude_entry = m["targets"]
    assert shared_entry["agent"] == "shared" and shared_entry["kind"] == "copy"
    assert set(shared_entry["files"]) == set(skill.packaged_files())
    assert claude_entry == {"agent": "claude", "path": str(link), "kind": "symlink",
                            "link_target": os.path.realpath(shared), "files": None}
    if not WINDOWS:
        assert stat.S_IMODE((state / "skill.json").stat().st_mode) == 0o600
    text = "\n".join(r.lines())
    assert "read there by Codex, Pi" in text and "Claude Code: ~/.claude/skills/runfleet → ~/.agents/skills/runfleet" \
        in text and "Trigger words: k2 vega" in text and "Kept up to date automatically" in text
    assert r.to_json()["targets"][1]["kind"] == "symlink"
    assert not [p for p in shared.parent.iterdir() if p.name.startswith(".")]


def test_claude_config_dir_is_honoured(home: Path, state: Path, tmp_path: Path, symlinks: None) -> None:
    cc = tmp_path / "cc"
    env = make_env(home, environ={"CLAUDE_CONFIG_DIR": str(cc)})
    r = do_install(env, state)
    assert [t.path for t in r.targets] == [canonical(home), cc / "skills" / "runfleet"]
    assert (cc / "skills" / "runfleet" / "SKILL.md").is_file() and not (home / ".claude").exists()


def test_copilot_link_only_with_copilot_home_and_antigravity_only_explicit(home: Path, state: Path, tmp_path: Path,
                                                                          symlinks: None) -> None:
    (home / ".copilot").mkdir()
    (home / ".gemini" / "antigravity-cli").mkdir(parents=True)
    r = do_install(make_env(home), state)
    assert [t.agent for t in r.targets] == ["shared"] and "copilot" in r.readers
    cp = tmp_path / "cp"
    r = do_install(make_env(home, environ={"COPILOT_HOME": str(cp)}), state, agents="auto")
    assert [t.agent for t in r.targets] == ["shared", "copilot"] and (cp / "skills" / "runfleet").is_symlink()
    r = do_install(make_env(home), state, agents="antigravity")
    assert r.targets[1].path == home / ".gemini" / "antigravity-cli" / "skills" / "runfleet"
    assert r.targets[1].path.is_symlink() and not (home / ".gemini" / "skills").exists()


def test_reinstall_is_idempotent(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    first = manifest(state)
    mtime = (canonical(home) / "SKILL.md").stat().st_mtime_ns
    r = do_install(env, state)
    assert [t.action for t in r.targets] == ["unchanged", "unchanged"] and r.backups == ()
    assert manifest(state)["install_id"] == first["install_id"]
    assert (canonical(home) / "SKILL.md").stat().st_mtime_ns == mtime


def test_junction_on_windows(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch, symlinks: None) -> None:
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(skill, "_create_junction", fake_junction(calls))
    env = claude_env(home, windows=True)
    r = do_install(env, state)
    assert r.targets[1].kind == "junction" and len(calls) == 1 and calls[0][1].name.startswith(".runfleet.tmp-")
    assert os.path.isabs(os.readlink(claude_link(home))) and manifest(state)["targets"][1]["kind"] == "junction"
    assert do_install(env, state).targets[1].action == "unchanged" and len(calls) == 1
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.exit_code == 0 and not os.path.lexists(claude_link(home)) and not canonical(home).exists()


def test_windows_falls_back_to_symlink_then_copy(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch,
                                                 symlinks: None) -> None:
    def no_junction(target: Path, link: Path) -> None:
        raise OSError("no junctions here")
    monkeypatch.setattr(skill, "_create_junction", no_junction)
    env = claude_env(home, windows=True)
    assert do_install(env, state).targets[1].kind == "symlink"
    skill.uninstall(env, state_dir=state, now=NOW)

    def no_symlink(*a: object, **k: object) -> None:
        raise OSError("symlinks not permitted")
    monkeypatch.setattr(skill.os, "symlink", no_symlink)
    r = do_install(env, state)
    assert r.targets[1].kind == "copy" and any("copied the skill there" in n for n in r.notes)


def test_copy_fallback_is_tracked_and_refreshed(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_symlink(*a: object, **k: object) -> None:
        raise OSError("symlinks not permitted")
    monkeypatch.setattr(skill.os, "symlink", no_symlink)
    env = claude_env(home)
    r = do_install(env, state)
    copy = claude_link(home)
    assert r.exit_code == 0 and r.targets[1].kind == "copy" and copy.is_dir() and not copy.is_symlink()
    entry = manifest(state)["targets"][1]
    assert entry["kind"] == "copy" and set(entry["files"]) == set(skill.packaged_files())
    assert json.loads((copy / skill.STAMP_FILE).read_text(encoding="utf-8"))["install_id"] == \
        manifest(state)["install_id"]
    u = do_update(env, state, host_names=["k2", "lyra"])
    assert [t.action for t in u.targets] == ["updated", "updated"] and u.state == "updated"
    assert "k2 lyra" in (copy / "SKILL.md").read_text(encoding="utf-8")
    assert not claude_link(home).is_symlink()
    skill.uninstall(env, state_dir=state, now=NOW)
    assert not copy.exists() and not canonical(home).exists()


def test_home_through_a_symlinked_parent(tmp_path: Path, state: Path, symlinks: None) -> None:
    real = tmp_path / "real-home"
    (real / ".claude").mkdir(parents=True)
    linked = tmp_path / "linked-home"
    os.symlink(real, linked, target_is_directory=True)
    env = make_env(linked)
    r = do_install(env, state)
    assert r.exit_code == 0 and claude_link(linked).resolve() == canonical(real).resolve()
    assert [t.action for t in do_install(env, state).targets] == ["unchanged", "unchanged"]
    st = skill.status(env, state_dir=state)
    assert st.installed and st.up_to_date and next(a for a in st.agents if a.id == "claude").covered_by == "link"
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert len(u.removed) == 2 and not os.path.lexists(claude_link(real)) and not canonical(real).exists()


def test_user_edited_and_added_files_are_saved_on_update_and_uninstall(home: Path, state: Path,
                                                                     symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    shared = canonical(home)
    (shared / "SKILL.md").write_text("my edits", encoding="utf-8")
    (shared / "notes.md").write_text("my notes", encoding="utf-8")
    r = do_install(env, state, host_names=["k2", "lyra"])
    assert r.targets[0].action == "updated" and len(r.backups) == 2 and r.backup_dir is not None
    assert backup_files(state) == {"shared/SKILL.md": "my edits", "shared/notes.md": "my notes"}
    assert not (shared / "notes.md").exists() and "k2 lyra" in (shared / "SKILL.md").read_text(encoding="utf-8")
    assert any("Saved 2 file(s)" in line for line in r.lines())
    assert str(r.backup_dir).startswith(str(state / "skill-backup" / "20261018T120000Z"))
    (shared / "references").mkdir(exist_ok=True)
    (shared / "references" / "mine.md").write_text("mine too", encoding="utf-8")
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.exit_code == 0 and not shared.exists()
    assert u.backup_dir is not None and u.backup_dir.name == "20261018T120000Z-2"
    assert (u.backup_dir / "shared" / "references" / "mine.md").read_text(encoding="utf-8") == "mine too"


def test_status_reports_edited_files(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    (canonical(home) / "SKILL.md").write_text("changed", encoding="utf-8")
    st = skill.status(env, state_dir=state)
    assert st.edited == ("SKILL.md",) and any("edited by you" in line for line in st.lines())


def test_foreign_shared_dir_is_refused_and_force_moves_it(home: Path, state: Path, symlinks: None) -> None:
    shared = canonical(home)
    shared.mkdir(parents=True)
    (shared / "SKILL.md").write_text("the user's own skill", encoding="utf-8")
    env = claude_env(home)
    with pytest.raises(skill.SkillInstallError, match="not installed by RunFleet"):
        do_install(env, state)
    assert (shared / "SKILL.md").read_text(encoding="utf-8") == "the user's own skill"
    assert not os.path.lexists(claude_link(home)) and not (state / "skill.json").exists()
    r = do_install(env, state, force=True)
    assert r.targets[0].action == "replaced" and r.exit_code == 0
    assert backup_files(state) == {"shared/runfleet/SKILL.md": "the user's own skill"}
    assert (shared / skill.STAMP_FILE).is_file()


def test_foreign_link_is_refused_and_force_moves_only_the_link(home: Path, state: Path, tmp_path: Path,
                                                              symlinks: None) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "keep.txt").write_text("x", encoding="utf-8")
    link = claude_link(home)
    link.parent.mkdir(parents=True)
    os.symlink(other, link, target_is_directory=True)
    env = make_env(home)
    with pytest.raises(skill.SkillInstallError, match="a link to elsewhere"):
        do_install(env, state)
    assert not canonical(home).exists()
    r = do_install(env, state, force=True)
    assert r.targets[1].action == "replaced" and link.resolve() == canonical(home).resolve()
    assert (other / "keep.txt").is_file()
    moved = state / "skill-backup" / "20261018T120000Z" / "claude" / "runfleet"
    assert moved.is_symlink() and moved.resolve() == other.resolve()


def test_other_install_id_is_not_ours(home: Path, state: Path, tmp_path: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    other_state = tmp_path / "other-state"
    do_install(env, other_state)                   # another state dir adopts the stamped copy under a new install id
    with pytest.raises(skill.SkillInstallError):
        do_install(env, state)


@pytest.mark.skipif(WINDOWS or is_root(), reason="POSIX permissions; root ignores them")
def test_partial_failure_keeps_the_shared_copy(home: Path, state: Path) -> None:
    env = claude_env(home)
    skills = home / ".claude" / "skills"
    skills.mkdir()
    skills.chmod(0o500)
    try:
        r = do_install(env, state)
    finally:
        skills.chmod(0o700)
    assert r.exit_code == 1 and [t.action for t in r.targets] == ["created", "failed"]
    assert r.targets[1].error and (canonical(home) / "SKILL.md").is_file()
    assert [t["agent"] for t in manifest(state)["targets"]] == ["shared"]
    assert any("failed" in line for line in r.lines())


def test_failed_swap_keeps_the_old_copy_and_the_working_link(home: Path, state: Path,
                                                             monkeypatch: pytest.MonkeyPatch, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    before = (canonical(home) / "SKILL.md").read_text(encoding="utf-8")
    real_replace = os.replace

    def broken(src: Any, dst: Any) -> None:
        if Path(src).name.startswith(skill.STAGING):
            raise OSError("disk full")
        real_replace(src, dst)
    monkeypatch.setattr(skill.os, "replace", broken)
    r = do_install(env, state, host_names=["k2", "lyra"])
    assert r.exit_code == 1 and r.targets[0].action == "failed" and "disk full" in (r.targets[0].error or "")
    assert (canonical(home) / "SKILL.md").read_text(encoding="utf-8") == before
    assert r.targets[1].action == "unchanged" and claude_link(home).resolve() == canonical(home).resolve()
    assert sorted(p.name for p in canonical(home).parent.iterdir()) == ["runfleet"]


def test_failed_link_swap_keeps_the_old_copy(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch,
                                             symlinks: None) -> None:
    """A copy at the Claude path (from an older Windows install) is replaced by a link; when that swap fails the
    copy stays as it was."""
    env = claude_env(home)
    do_install(env, state, agents="codex")
    old = claude_link(home)
    old.mkdir(parents=True)
    (old / "SKILL.md").write_text("old copy", encoding="utf-8")
    (old / skill.LEGACY_STAMP).write_text("0.7.0.dev1\n", encoding="utf-8")
    (state / "skill.json").unlink()                 # legacy: no manifest, stamped copies are ours

    def broken(new: Path, dest: Path, *, atomic: bool) -> Path | None:
        raise OSError("rename failed")
    monkeypatch.setattr(skill, "_swap_in", broken)
    r = do_install(env, state, agents="claude")
    assert r.exit_code == 1 and r.targets[1].action == "failed"
    assert (old / "SKILL.md").read_text(encoding="utf-8") == "old copy" and not old.is_symlink()
    assert not [p for p in old.parent.iterdir() if p.name.startswith(".")]


def test_install_and_update_are_serialised(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch,
                                           symlinks: None) -> None:
    env = claude_env(home)
    active, peak = [0], [0]
    real = skill._write_copy

    def slow(*a: Any, **k: Any) -> skill.TargetResult:
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        try:
            return real(*a, **k)
        finally:
            active[0] -= 1
    monkeypatch.setattr(skill, "_write_copy", slow)
    do_install(env, state)
    results: list[object] = []

    def run(i: int) -> None:
        if i % 2:
            results.append(do_install(env, state, host_names=[f"h{i}"]).exit_code)
        else:
            results.append(do_update(env, state, host_names=[f"h{i}"]).exit_code)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [0, 0, 0, 0] and peak[0] == 1


def test_lock_held_elsewhere_refuses_install_after_waiting(home: Path, state: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill, "LOCK_WAIT", 0.2)
    with held_lock(state):
        with pytest.raises(skill.SkillInstallError, match="another `runfleet skill` command is running"):
            do_install(make_env(home), state)


class held_lock:  # noqa: N801 - a context manager used like a function
    """Holds <state>/skill.lock through a second descriptor, as another process would."""

    def __init__(self, state: Path) -> None:
        state.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(state / skill.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)

    def __enter__(self) -> held_lock:
        assert rf_platform.try_lock(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        rf_platform.unlock(self.fd)
        os.close(self.fd)


def test_leftovers_are_swept(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    skills = canonical(home).parent
    (skills / ".runfleet-staging-0a1b2c3d4e" / "x").mkdir(parents=True)        # an interrupted write
    (skills / ".runfleet.old-0a1b2c3d").mkdir()                                  # an old copy put aside
    (skills / ".runfleet.old-0a1b2c3d" / skill.LEGACY_STAMP).write_text("0.8.0\n", encoding="utf-8")
    (home / ".claude" / "skills").mkdir(parents=True)
    os.symlink(skills, home / ".claude" / "skills" / ".runfleet.tmp-0a1b2c3d4e", target_is_directory=True)
    (skills / ".unrelated").mkdir()
    r = do_install(env, state)
    assert len(r.swept) == 3 and sorted(p.name for p in skills.iterdir()) == [".unrelated", "runfleet"]
    assert skills.is_dir()                                   # the tmp link was removed, not followed
    os.symlink(skills, skills / ".runfleet.tmp-ffffffff", target_is_directory=True)
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert any(p.name == ".runfleet.tmp-ffffffff" for p in u.swept) and sorted(p.name for p in skills.iterdir()) == \
        [".unrelated"]


# ---------------------------------------------------------------- update
def _legacy_install(home: Path) -> None:
    shared = canonical(home)
    (shared / "references").mkdir(parents=True)
    (shared / "SKILL.md").write_text("---\nname: runfleet\n---\nold\n", encoding="utf-8")
    (shared / "references" / "reading-output.md").write_text("old ref", encoding="utf-8")
    (shared / skill.LEGACY_STAMP).write_text("0.7.0.dev1\n", encoding="utf-8")
    claude_link(home).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(shared, claude_link(home), target_is_directory=True)


def test_update_adopts_a_legacy_install(home: Path, state: Path, symlinks: None) -> None:
    _legacy_install(home)
    env = claude_env(home)
    r = do_update(env, state, host_names=["k2"], if_installed=True)
    assert r.state == "updated" and r.previous_version == "0.7.0.dev1" and r.exit_code == 0
    assert [(t.agent, t.action) for t in r.targets] == [("shared", "updated"), ("claude", "unchanged")]
    shared = canonical(home)
    assert not (shared / skill.LEGACY_STAMP).exists() and (shared / skill.STAMP_FILE).is_file()
    assert set(backup_files(state)) == {"shared/SKILL.md", "shared/references/reading-output.md"}
    m = manifest(state)
    assert [t["agent"] for t in m["targets"]] == ["shared", "claude"] and m["hosts"] == ["k2"]
    assert any("adopted" in n for n in r.notes)
    assert "updated" in (r.summary() or "") and "0.7.0.dev1 →" in (r.summary() or "")


def test_update_turns_a_legacy_windows_copy_into_a_link(home: Path, state: Path, symlinks: None) -> None:
    _legacy_install(home)
    link = claude_link(home)
    link.unlink()
    link.mkdir()
    (link / "SKILL.md").write_text("copy", encoding="utf-8")
    (link / skill.LEGACY_STAMP).write_text("0.7.0.dev1\n", encoding="utf-8")
    r = do_update(claude_env(home), state)
    assert r.targets[1].action == "replaced" and link.is_symlink() and link.resolve() == canonical(home).resolve()
    assert "claude/SKILL.md" in backup_files(state)


def test_update_without_install(home: Path, state: Path) -> None:
    env = claude_env(home)
    r = do_update(env, state, if_installed=True)
    assert r.state == "not_installed" and r.exit_code == 0 and r.lines() == [] and r.summary() is None
    assert not canonical(home).exists() and not (state / "skill.json").exists()
    r = do_update(env, state)
    assert r.exit_code == 0 and "not installed" in r.lines()[0]


def test_update_does_not_recreate_what_the_user_removed(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    claude_link(home).unlink()
    r = do_update(env, state, host_names=["lyra"])
    assert [t.action for t in r.targets] == ["updated", "missing"] and not os.path.lexists(claude_link(home))
    assert [t["agent"] for t in manifest(state)["targets"]] == ["shared"]
    shutil.rmtree(canonical(home))
    r = do_update(env, state)
    assert r.state == "canonical_missing" and not canonical(home).exists() and r.exit_code == 0
    assert "runfleet skill install" in (r.summary() or "") and r.lines()


def test_update_keeps_recorded_hosts_and_reports_unchanged(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state, host_names=["k2", "lyra"], config_path="/c", config_mtime=5.0)
    r = do_update(env, state)
    assert r.state == "unchanged" and r.summary() is None and "up to date" in r.lines()[0]
    m = manifest(state)
    assert m["hosts"] == ["k2", "lyra"] and m["config_path"] == "/c" and m["config_mtime"] == 5.0
    assert "k2 lyra" in (canonical(home) / "SKILL.md").read_text(encoding="utf-8")


def test_update_leaves_a_target_the_user_replaced(home: Path, state: Path, tmp_path: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    claude_link(home).unlink()
    (tmp_path / "mine").mkdir()
    os.symlink(tmp_path / "mine", claude_link(home), target_is_directory=True)
    r = do_update(env, state, host_names=["x1"])
    assert r.targets[1].action == "foreign" and claude_link(home).resolve() == (tmp_path / "mine").resolve()


# ---------------------------------------------------------------- uninstall
def test_uninstall_removes_exactly_ours(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    shared = canonical(home)
    stray = home / ".codex" / "skills" / "runfleet"                  # an old copy RunFleet wrote somewhere else
    shutil.copytree(shared, stray)
    (stray / "added.md").write_text("mine", encoding="utf-8")
    gem = home / ".gemini" / "skills" / "runfleet"                   # a second link to ours
    gem.parent.mkdir(parents=True)
    os.symlink(shared, gem, target_is_directory=True)
    foreign = home / ".cursor" / "skills" / "runfleet"               # somebody else's skill of that name
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("theirs", encoding="utf-8")
    st = skill.status(env, state_dir=state)
    assert {(s.agent, s.kind) for s in st.strays} == {("codex", "stamped_copy"), ("gemini", "link_to_ours"),
                                                     ("cursor", "foreign")}
    dry = skill.uninstall(env, state_dir=state, dry_run=True, now=NOW)
    assert dry.dry_run and len(dry.removed) == 4 and shared.is_dir() and gem.is_symlink() and stray.is_dir()
    assert (state / "skill.json").is_file() and not (state / "skill-backup").exists()
    assert any(line.startswith("Would remove") for line in dry.lines())
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.exit_code == 0 and u.manifest_removed and len(u.removed) == 4
    assert not os.path.lexists(shared) and not os.path.lexists(claude_link(home)) and not os.path.lexists(gem)
    assert not stray.exists() and (foreign / "SKILL.md").read_text(encoding="utf-8") == "theirs"
    assert not (state / "skill.json").exists() and backup_files(state) == {"codex/added.md": "mine"}
    text = "\n".join(u.lines())
    assert "claude mcp remove runfleet -s user" in text and "[mcp_servers.runfleet]" in text
    assert "Bash(runfleet agent *)" in text
    again = skill.uninstall(env, state_dir=state, now=NOW)
    assert again.nothing_installed and again.exit_code == 0 and "nothing to remove" in again.lines()[0]
    assert again.to_json()["nothing_installed"] is True


def test_uninstall_leaves_a_linked_skills_dir_setup_intact(home: Path, state: Path, symlinks: None) -> None:
    (home / ".agents" / "skills").mkdir(parents=True)
    (home / ".claude").mkdir()
    os.symlink(home / ".agents" / "skills", home / ".claude" / "skills", target_is_directory=True)
    env = make_env(home)
    r = do_install(env, state)
    assert [t.agent for t in r.targets] == ["shared"]
    claude = next(a for a in skill.status(env, state_dir=state).agents if a.id == "claude")
    assert claude.covered_by == "shared" and claude.state == "ok"
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert [t.agent for t in u.removed] == ["shared"]
    assert (home / ".claude" / "skills").is_symlink() and (home / ".agents" / "skills").is_dir()


def test_uninstall_leaves_foreign_targets_named_in_the_manifest(home: Path, state: Path, tmp_path: Path,
                                                               symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    claude_link(home).unlink()
    claude_link(home).mkdir()
    (claude_link(home) / "SKILL.md").write_text("now mine", encoding="utf-8")
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert [t.agent for t in u.removed] == ["shared"] and [t.action for t in u.kept] == ["foreign"]
    assert (claude_link(home) / "SKILL.md").read_text(encoding="utf-8") == "now mine"


# ---------------------------------------------------------------- status
def test_status_json_shape(home: Path, state: Path, symlinks: None) -> None:
    (home / ".codex").mkdir()
    env = claude_env(home)
    st = skill.status(env, state_dir=state)
    assert not st.installed and st.canonical_state == "missing" and "not installed" in st.lines()[0]
    do_install(env, state)
    st = skill.status(env, state_dir=state)
    j = st.to_json()
    assert set(j) == {"installed", "canonical", "canonical_state", "version", "running_version", "up_to_date",
                      "hosts", "auto_update", "edited", "legacy", "agents", "strays", "broken_links", "manifest",
                      "last_backup", "last_failure"}
    assert j["installed"] and j["up_to_date"] and j["auto_update"] is True and j["hosts"] == HOSTS
    by_id = {a["id"]: a for a in j["agents"]}
    assert set(by_id) == set(skill.AGENT_IDS)
    assert set(by_id["claude"]) == {"id", "label", "detected", "covered_by", "path", "state", "in_manifest"}
    assert by_id["claude"]["covered_by"] == "link" and by_id["claude"]["in_manifest"]
    assert by_id["codex"] == {"id": "codex", "label": "Codex", "detected": True, "covered_by": "shared",
                              "path": None, "state": "ok", "in_manifest": False}
    assert by_id["antigravity"]["covered_by"] == "none" and not by_id["antigravity"]["detected"]
    text = "\n".join(st.lines())
    assert "up to date" in text and "Claude Code: link" in text and "Codex: reads the shared copy" in text
    assert "Antigravity" not in text
    claude_link(home).unlink()
    os.symlink(home / "nowhere", claude_link(home))
    st = skill.status(env, state_dir=state)
    assert st.broken_links == (claude_link(home),) and "broken link" in "\n".join(st.lines())


def test_status_of_a_legacy_install_and_an_old_version(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch,
                                                       symlinks: None) -> None:
    _legacy_install(home)
    st = skill.status(claude_env(home), state_dir=state)
    assert st.installed and st.legacy and not st.up_to_date and st.version == "0.7.0.dev1"
    assert "older RunFleet" in st.lines()[0]


# ---------------------------------------------------------------- refresh_reason / auto_refresh
@pytest.fixture
def installed(home: Path, state: Path) -> skill.Env:
    env = make_env(home)
    do_install(env, state, config_path="/c.toml", config_mtime=100.0)
    return env


def test_refresh_reason_matrix(installed: skill.Env, state: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reason(mtime: float | None = 100.0, path: str | None = None) -> str | None:
        return skill.refresh_reason(installed, state_dir=state, config_mtime=mtime, config_path=path)
    assert reason() is None
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    assert reason() == f"version {manifest(state)['version']} → 99.0.0"
    edit_manifest(state, version="100.0.0")
    assert reason() is None                                          # never downgrade
    edit_manifest(state, version="99.0.0+local")
    assert reason() is None                                          # same release
    edit_manifest(state, version="not-a-version")
    assert reason() is None                                          # unsure: nothing
    edit_manifest(state, version="99.0.0", bundle_sha256="0" * 64)
    assert reason() == "bundle"
    edit_manifest(state, bundle_sha256=skill.bundle_sha256())
    assert reason(101.0, "/c.toml") == "config" and reason(None, "/c.toml") == "config"
    assert reason(101.0, "/other.toml") is None
    assert reason(101.0) is None                                     # the default config is not the recorded one
    edit_manifest(state, auto_update=False, version="1.0")
    assert reason(101.0, "/c.toml") is None
    edit_manifest(state, auto_update=True)
    assert reason(101.0, "/c.toml") is not None
    shutil.rmtree(canonical(home))
    assert reason(101.0, "/c.toml") is None
    (state / "skill.json").unlink()
    assert reason(101.0, "/c.toml") is None
    (state / "skill.json").write_text("{not json", encoding="utf-8")
    assert reason(101.0, "/c.toml") is None


def test_no_auto_update_is_recorded(home: Path, state: Path) -> None:
    env = make_env(home)
    r = do_install(env, state, auto_update=False)
    assert manifest(state)["auto_update"] is False and "Automatic refresh is off" in "\n".join(r.lines())
    edit_manifest(state, version="0.0.1")
    assert skill.refresh_reason(env, state_dir=state, config_mtime=None) is None
    do_update(env, state)
    assert manifest(state)["auto_update"] is False


def test_auto_refresh_after_an_upgrade(installed: skill.Env, state: Path, home: Path,
                                       monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    line = skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: ["zz"], now=NOW)
    assert line is not None and "99.0.0" in line
    stamp = json.loads((canonical(home) / skill.STAMP_FILE).read_text(encoding="utf-8"))
    assert stamp["version"] == "99.0.0" and manifest(state)["version"] == "99.0.0"
    assert manifest(state)["hosts"] == HOSTS                         # config unchanged: hosts not reloaded
    assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0) is None
    assert capsys.readouterr() == ("", "")


def test_auto_refresh_on_config_change(installed: skill.Env, state: Path, home: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    before = (canonical(home) / "SKILL.md").stat().st_mtime_ns
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=200.0, load_host_names=lambda: HOSTS,
                              now=NOW, config_path="/c.toml") is None
    assert manifest(state)["config_mtime"] == 200.0 and (canonical(home) / "SKILL.md").stat().st_mtime_ns == before
    line = skill.auto_refresh(installed, state_dir=state, config_mtime=300.0, load_host_names=lambda: ["k2", "orion"],
                              now=NOW, config_path="/c.toml")
    assert line is not None and "config" in line
    assert "k2 orion" in (canonical(home) / "SKILL.md").read_text(encoding="utf-8")
    m = manifest(state)
    assert m["hosts"] == ["k2", "orion"] and m["config_mtime"] == 300.0 and m["config_path"] == "/c.toml"
    assert capsys.readouterr() == ("", "")


def test_auto_refresh_skips_when_busy_and_never_raises(installed: skill.Env, state: Path, home: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    with held_lock(state):
        t0 = time.monotonic()
        assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                                  now=NOW) is None
        assert time.monotonic() - t0 < 1.0
    assert manifest(state)["version"] != "99.0.0"

    def boom() -> list[str]:
        raise RuntimeError("config exploded")
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=5.0, load_host_names=boom, now=NOW,
                              config_path="/c.toml") is None
    assert manifest(state)["last_failure_at"] == NOW.timestamp() and "config exploded" in manifest(state)["last_error"]
    assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0, now=NOW.timestamp() + 60) is None

    def clear_backoff() -> None:
        data = manifest(state)
        data.pop("last_failure_at")
        (state / "skill.json").write_text(json.dumps(data), encoding="utf-8")

    def broken(*a: object, **k: object) -> None:
        raise PermissionError("read-only home")
    clear_backoff()
    monkeypatch.setattr(skill, "_write_copy", broken)
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                              now=NOW) is not None                   # per-target failure: reported in the line
    clear_backoff()
    monkeypatch.setattr(skill, "_update_locked", broken)
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                              now=NOW) is None
    monkeypatch.setattr(skill, "refresh_reason", broken)
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                              now=NOW) is None
    assert capsys.readouterr() == ("", "")


def test_auto_refresh_does_nothing_when_not_installed(home: Path, state: Path) -> None:
    assert skill.auto_refresh(make_env(home), state_dir=state, config_mtime=1.0, load_host_names=lambda: ["x"],
                              now=NOW) is None
    assert not state.exists()


def test_refresh_reason_is_cheap(installed: skill.Env, state: Path) -> None:
    skill.refresh_reason(installed, state_dir=state, config_mtime=100.0)          # warm the per-process caches
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0) is None
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.020, f"median {statistics.median(times) * 1000:.2f} ms (target < 5 ms)"


# ---------------------------------------------------------------- misc
def test_is_link_sees_windows_reparse_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    target = tmp_path / "d"
    target.mkdir()
    assert not skill._is_link(target)
    fake = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=0x400)
    monkeypatch.setattr(skill.os, "lstat", lambda p: fake)
    assert skill._is_link(target)


@pytest.mark.skipif(not WINDOWS, reason="real junctions need Windows")
def test_real_junction_on_windows(tmp_path: Path) -> None:
    target, link = tmp_path / "t", tmp_path / "j"
    target.mkdir()
    skill._create_junction(target, link)
    assert skill._is_link(link) and not os.path.islink(link) and os.path.samefile(link, target)
    skill._remove(link)
    assert target.is_dir() and not os.path.lexists(link)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_skill_files_ship_in_the_wheel(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    proc = subprocess.run(["uv", "build", "--offline", "--wheel", "--out-dir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=300, check=False)
    if proc.returncode != 0:
        pytest.skip(f"uv build unavailable here: {proc.stderr[-300:]}")
    names = set(zipfile.ZipFile(next(out.glob("*.whl"))).namelist())
    for f in skill.packaged_files():
        assert f"runfleet/agent/skill/{f}" in names
    assert "runfleet/agent/skill.py" in names


# ---------------------------------------------------------------- review fixes
def test_sweep_keeps_lookalikes_that_are_not_ours(home: Path, state: Path, symlinks: None) -> None:
    skills = canonical(home).parent
    (skills / ".runfleet.old-notes").mkdir(parents=True)
    (skills / ".runfleet.old-0123abcd").mkdir()
    (skills / ".runfleet.old-0123abcd" / "mine.txt").write_text("keep", encoding="utf-8")
    ours = skills / ".runfleet.old-89abcdef"
    ours.mkdir()
    (ours / skill.STAMP_FILE).write_text("{}", encoding="utf-8")
    env = make_env(home)
    r = do_install(env, state)
    assert r.swept == (ours,) and (skills / ".runfleet.old-notes").is_dir()
    assert (skills / ".runfleet.old-0123abcd" / "mine.txt").read_text(encoding="utf-8") == "keep"
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.swept == () and (skills / ".runfleet.old-0123abcd").is_dir()


def test_backups_of_two_copies_of_one_agent_do_not_collide(home: Path, state: Path, tmp_path: Path,
                                                           symlinks: None) -> None:
    codex_home = tmp_path / "codex-home"
    env = make_env(home, environ={"CODEX_HOME": str(codex_home)})
    do_install(env, state)
    for root, text in ((codex_home, "one"), (home / ".codex", "two")):
        stray = root / "skills" / "runfleet"
        shutil.copytree(canonical(home), stray)
        (stray / "notes.md").write_text(text, encoding="utf-8")
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.exit_code == 0 and sorted(backup_files(state).items()) == [("codex-2/notes.md", "two"),
                                                                        ("codex/notes.md", "one")]


def test_auto_refresh_rechecks_under_the_lock(installed: skill.Env, state: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Another process upgraded the skill between the check and the lock: no downgrade."""
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    real = skill._locked

    @contextlib.contextmanager
    def racing(state_dir: Path, *, wait: float) -> Iterator[bool]:
        edit_manifest(state, version="100.0.0")
        with real(state_dir, wait=wait) as held:
            yield held
    monkeypatch.setattr(skill, "_locked", racing)
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                              now=NOW) is None
    assert manifest(state)["version"] == "100.0.0"


def test_a_failed_refresh_is_retried(installed: skill.Env, state: Path, home: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    old_version = manifest(state)["version"]
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    real_replace = os.replace

    def broken(src: Any, dst: Any) -> None:
        if Path(src).name.startswith(skill.STAGING):
            raise OSError("disk full")
        real_replace(src, dst)
    monkeypatch.setattr(skill.os, "replace", broken)
    line = skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [], now=NOW)
    assert line is not None and line.startswith("skill: refresh failed (version") and "disk full" in line
    assert "updated" not in line and old_version != "99.0.0"
    assert manifest(state)["version"] == "99.0.0" and manifest(state)["bundle_sha256"] == skill.RETRY
    later = NOW.timestamp() + skill.BACKOFF_SECONDS + 1
    assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0, now=later) == "bundle"
    monkeypatch.setattr(runfleet, "__version__", "98.0.0")          # an older RunFleet: no downgrade for a retry
    assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0, now=later) is None
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    monkeypatch.setattr(skill.os, "replace", real_replace)
    skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=lambda: [],
                       now=datetime.fromtimestamp(later, UTC))
    assert manifest(state)["version"] == "99.0.0" and "last_failure_at" not in manifest(state)
    assert json.loads((canonical(home) / skill.STAMP_FILE).read_text(encoding="utf-8"))["version"] == "99.0.0"


def test_uninstall_keeps_a_foreign_link_at_the_shared_path(home: Path, state: Path, tmp_path: Path,
                                                           symlinks: None) -> None:
    mine = tmp_path / "my-skill"
    mine.mkdir()
    canonical(home).parent.mkdir(parents=True)
    os.symlink(mine, canonical(home), target_is_directory=True)
    u = skill.uninstall(make_env(home), state_dir=state, now=NOW)
    assert canonical(home).is_symlink() and [t.action for t in u.kept] == ["foreign"] and not u.removed


def test_reinstall_with_other_agents_keeps_earlier_targets(home: Path, state: Path, symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    r = do_install(env, state, agents="codex")
    assert [t.agent for t in r.targets] == ["shared", "claude"] and r.targets[1].action == "unchanged"
    m = manifest(state)
    assert [t["agent"] for t in m["targets"]] == ["shared", "claude"] and "claude" in m["agents"]
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert {t.agent for t in u.removed} == {"shared", "claude"}


def test_force_with_a_failed_write_is_still_a_failure(home: Path, state: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    shared = canonical(home)
    shared.mkdir(parents=True)
    (shared / "SKILL.md").write_text("theirs", encoding="utf-8")

    def broken(*a: object, **k: object) -> None:
        raise OSError("disk full")
    monkeypatch.setattr(skill, "_write_copy", broken)
    r = do_install(make_env(home), state, force=True)
    assert r.exit_code == 1 and r.targets[0].action == "failed" and len(r.targets[0].backups) == 1
    assert backup_files(state) == {"shared/runfleet/SKILL.md": "theirs"}


def test_refresh_reason_is_cheap_in_a_cold_process(installed: skill.Env, state: Path) -> None:
    """A CLI process starts with no cached bundle hash: the check then hashes the packaged files once."""
    times = []
    for _ in range(20):
        skill.bundle_sha256.cache_clear()
        t0 = time.perf_counter()
        assert skill.refresh_reason(installed, state_dir=state, config_mtime=100.0) is None
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.020, f"median {statistics.median(times) * 1000:.2f} ms"


def test_a_skills_dir_linked_after_install_is_never_replaced_by_a_self_link(home: Path, state: Path,
                                                                           symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    skills = home / ".claude" / "skills"
    claude_link(home).unlink()
    skills.rmdir()
    os.symlink(home / ".agents" / "skills", skills, target_is_directory=True)
    r = do_install(env, state, host_names=["k2", "lyra"])
    assert r.exit_code == 0 and [t.agent for t in r.targets] == ["shared"]
    u = do_update(env, state, host_names=["k2", "orion"])
    assert u.exit_code == 0 and [t.agent for t in u.targets] == ["shared"]
    shared = canonical(home)
    assert shared.is_dir() and not shared.is_symlink() and "k2 orion" in (shared / "SKILL.md").read_text("utf-8")
    skill.uninstall(env, state_dir=state, now=NOW)
    assert skills.is_symlink() and not os.path.lexists(shared)


def test_a_failed_adoption_leaves_the_legacy_install_adoptable(home: Path, state: Path,
                                                               monkeypatch: pytest.MonkeyPatch,
                                                               symlinks: None) -> None:
    _legacy_install(home)
    env = claude_env(home)

    def broken(*a: object, **k: object) -> None:
        raise OSError("disk full")
    monkeypatch.setattr(skill, "_write_copy", broken)
    assert do_update(env, state).exit_code == 1 and not (state / "skill.json").exists()
    assert do_install(env, state).exit_code == 1 and not (state / "skill.json").exists()
    monkeypatch.undo()
    r = do_update(env, state)
    assert r.exit_code == 0 and r.targets[0].action == "updated" and (state / "skill.json").is_file()


def test_a_stamped_leftover_keeps_the_users_files(home: Path, state: Path) -> None:
    env = make_env(home)
    do_install(env, state)
    old = canonical(home).parent / ".runfleet.old-0123456789"
    shutil.copytree(canonical(home), old)
    (old / "recovered.md").write_text("mine", encoding="utf-8")
    r = do_install(env, state)
    assert r.swept == (old,) and backup_files(state) == {"leftover/recovered.md": "mine"}


def test_uninstall_through_another_name_of_home_keeps_a_foreign_link(tmp_path: Path, state: Path,
                                                                     symlinks: None) -> None:
    real = tmp_path / "real-home"
    real.mkdir()
    alias = tmp_path / "alias-home"
    os.symlink(real, alias, target_is_directory=True)
    do_install(make_env(alias), state)
    shutil.rmtree(canonical(real))
    mine = tmp_path / "mine"
    mine.mkdir()
    os.symlink(mine, canonical(real), target_is_directory=True)
    u = skill.uninstall(make_env(real), state_dir=state, now=NOW)
    assert canonical(real).is_symlink() and not u.removed and u.kept


# ---------------------------------------------------------------- second review (F1-F6, E1-E7)
def test_f1_users_own_links_are_never_ours_when_the_shared_copy_is_not(home: Path, state: Path, tmp_path: Path,
                                                                        symlinks: None) -> None:
    """No manifest; the user's own links at ~/.agents/skills/runfleet and ~/.claude/skills/runfleet both point to
    a skill of theirs: uninstall removes nothing, status does not count the link as ours, install refuses both."""
    mine = tmp_path / "personal" / "my-skill"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("mine", encoding="utf-8")
    canonical(home).parent.mkdir(parents=True)
    os.symlink(mine, canonical(home), target_is_directory=True)
    claude_link(home).parent.mkdir(parents=True)
    os.symlink(mine, claude_link(home), target_is_directory=True)
    env = claude_env(home)
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert not u.removed and u.exit_code == 0 and not u.manifest_removed
    assert canonical(home).is_symlink() and claude_link(home).is_symlink()
    assert (mine / "SKILL.md").read_text(encoding="utf-8") == "mine"
    claude = next(a for a in skill.status(env, state_dir=state).agents if a.id == "claude")
    assert (claude.covered_by, claude.state) == ("none", "foreign")
    with pytest.raises(skill.SkillInstallError) as err:
        do_install(env, state)
    assert str(claude_link(home)) in str(err.value) and "not RunFleet's" in str(err.value)
    r = do_install(env, state, force=True)                           # both links moved aside, never their target
    assert r.exit_code == 0 and not canonical(home).is_symlink() and claude_link(home).resolve() == \
        canonical(home).resolve()
    assert (mine / "SKILL.md").read_text(encoding="utf-8") == "mine"


def test_f1_links_stay_when_the_user_replaced_the_shared_copy_after_install(home: Path, state: Path, tmp_path: Path,
                                                                            symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    mine = tmp_path / "mine"
    mine.mkdir()
    shutil.rmtree(canonical(home))
    os.symlink(mine, canonical(home), target_is_directory=True)     # our relative claude link now resolves to it
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert not u.removed and claude_link(home).is_symlink() and canonical(home).is_symlink()
    assert {t.path for t in u.kept} == {canonical(home), claude_link(home)}


def test_f2_no_auto_update_also_stops_the_installers_refresh(home: Path, state: Path) -> None:
    env = make_env(home)
    do_install(env, state, auto_update=False)
    edit_manifest(state, version="0.0.1")
    r = do_update(env, state, if_installed=True)
    assert r.state == "disabled" and r.exit_code == 0 and r.targets == ()
    assert r.summary() == "runfleet skill: automatic refresh is off; run `runfleet skill update`"
    assert r.lines() == [r.summary()] and manifest(state)["version"] == "0.0.1"
    r = do_update(env, state)                                        # explicit: refreshes
    assert r.state == "updated" and manifest(state)["version"] == runfleet.__version__
    assert manifest(state)["auto_update"] is False


def test_f3_a_legacy_copy_that_could_not_be_refreshed_stays_tracked(home: Path, state: Path,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     symlinks: None) -> None:
    _legacy_install(home)
    link = claude_link(home)
    link.unlink()
    link.mkdir()
    (link / "SKILL.md").write_text("copy", encoding="utf-8")
    (link / skill.LEGACY_STAMP).write_text("0.7.0.dev1\n", encoding="utf-8")
    env = claude_env(home)
    real = skill._write_link

    def broken(ctx: Any, agent: str, dest: Path, **kw: Any) -> skill.TargetResult:
        raise OSError("read-only skills dir")
    monkeypatch.setattr(skill, "_write_link", broken)
    r = do_update(env, state)
    assert r.exit_code == 1 and [t.action for t in r.targets] == ["updated", "failed"]
    m = manifest(state)
    entry = next(t for t in m["targets"] if t["agent"] == "claude")
    assert entry["pending"] is True and entry["path"] == str(link) and m["bundle_sha256"] == skill.RETRY
    later = NOW.timestamp() + skill.BACKOFF_SECONDS + 1
    assert skill.refresh_reason(env, state_dir=state, config_mtime=None, now=later) == "bundle"
    assert skill.status(env, state_dir=state).agents[0].in_manifest
    monkeypatch.setattr(skill, "_write_link", real)
    r = do_update(env, state)                                        # retried: now a link, no longer pending
    assert r.exit_code == 0 and r.targets[1].action == "replaced" and link.is_symlink()
    assert "pending" not in next(t for t in manifest(state)["targets"] if t["agent"] == "claude")


def test_f3_uninstall_removes_a_pending_legacy_copy(home: Path, state: Path, monkeypatch: pytest.MonkeyPatch,
                                                    symlinks: None) -> None:
    _legacy_install(home)
    link = claude_link(home)
    link.unlink()
    link.mkdir()
    (link / "SKILL.md").write_text("copy", encoding="utf-8")
    (link / skill.LEGACY_STAMP).write_text("0.7.0.dev1\n", encoding="utf-8")
    env = claude_env(home)

    def broken(ctx: Any, agent: str, dest: Path, **kw: Any) -> skill.TargetResult:
        raise OSError("busy")
    monkeypatch.setattr(skill, "_write_link", broken)
    do_update(env, state)
    monkeypatch.undo()
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert {t.agent for t in u.removed} == {"shared", "claude"} and not os.path.lexists(link)
    assert any("older install" in line for line in u.lines())


def test_f4_walks_never_go_through_a_junction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Windows junction is a directory os.walk may enter: faked here by a predicate that calls one dir a link."""
    d = tmp_path / "copy"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.md").write_text("a", encoding="utf-8")
    junction = d / "sub" / "junction"
    junction.mkdir()
    (junction / "secret.txt").write_text("not ours", encoding="utf-8")
    real = skill._is_link
    monkeypatch.setattr(skill, "_is_link", lambda p: Path(p) == junction or real(Path(p)))
    tree = skill._tree(d)
    assert set(tree) == {"sub/a.md", "sub/junction"} and tree["sub/junction"] == f"link:{junction}"
    b = skill._Backup(tmp_path / "state", NOW)
    saved = b.copy_files("shared", d, sorted(tree))
    names = {p.relative_to(b.root).as_posix() for p in saved}
    assert names == {"shared/sub/a.md", "shared/sub/junction.link.txt"}
    assert (b.root / "shared" / "sub" / "junction.link.txt").read_text(encoding="utf-8") == f"link to {junction}\n"
    assert not list((tmp_path / "state").rglob("secret.txt"))


def test_f4_a_symlink_inside_our_copy_is_backed_up_as_a_link_record(home: Path, state: Path, tmp_path: Path,
                                                                    symlinks: None) -> None:
    env = make_env(home)
    do_install(env, state)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_text("x" * 100, encoding="utf-8")
    os.symlink(outside, canonical(home) / "data", target_is_directory=True)
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert u.exit_code == 0 and not canonical(home).exists() and (outside / "big.bin").is_file()
    assert backup_files(state) == {"shared/data.link.txt": f"link to {outside}\n"}


def test_link_records_drop_the_windows_extended_length_prefix(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """os.readlink on Windows reports an absolute target as \\\\?\\C:\\…: the backup record and the tree name the
    plain path (a UNC target as \\\\server\\…); POSIX link text is kept byte for byte."""
    assert skill._strip_extended("\\\\?\\C:\\Users\\ana\\out") == "C:\\Users\\ana\\out"
    assert skill._strip_extended("\\\\?\\UNC\\srv\\share\\x") == "\\\\srv\\share\\x"
    assert skill._strip_extended("\\\\?\\Volume{1}\\x") == "\\\\?\\Volume{1}\\x"   # no plain form: kept
    assert skill._strip_extended("..\\rel") == "..\\rel"
    monkeypatch.setattr(skill.os, "readlink", lambda p: "\\\\?\\C:\\out")
    monkeypatch.setattr(rf_platform, "IS_WINDOWS", True)
    assert skill._link_text(tmp_path) == "C:\\out"
    monkeypatch.setattr(rf_platform, "IS_WINDOWS", False)
    assert skill._link_text(tmp_path) == "\\\\?\\C:\\out"


def test_f5_update_reads_the_recorded_config_under_the_lock(home: Path, state: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(home)
    do_install(env, state, config_path="/a.toml", config_mtime=1.0)
    real = skill._locked

    @contextlib.contextmanager
    def racing(state_dir: Path, *, wait: float) -> Iterator[bool]:
        edit_manifest(state, config_path="/b.toml")                 # another install -c /b.toml got in first
        with real(state_dir, wait=wait) as held:
            yield held
    monkeypatch.setattr(skill, "_locked", racing)
    seen: list[str | None] = []

    def config(recorded: str | None) -> tuple[list[str], str | None, float]:
        seen.append(recorded)
        return ["b1"], recorded, 2.0
    r = skill.update(env, state_dir=state, host_names=None, config_path=None, config_mtime=None, config=config,
                     now=NOW)
    m = manifest(state)
    assert r.exit_code == 0 and seen == ["/b.toml"] and (m["config_path"], m["config_mtime"], m["hosts"]) == \
        ("/b.toml", 2.0, ["b1"])


def test_f5_auto_refresh_never_loads_another_config_than_the_recorded_one(installed: skill.Env, state: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller stat'ed the default config (None) while the manifest records /c.toml: an upgrade still refreshes,
    with the recorded hosts and mtime; the default config's names are never loaded."""
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")

    def never() -> list[str]:
        raise AssertionError("loaded the wrong config")
    line = skill.auto_refresh(installed, state_dir=state, config_mtime=555.0, load_host_names=never, now=NOW,
                              config_path=None)
    m = manifest(state)
    assert line is not None and "refreshed" in line and (m["config_path"], m["config_mtime"], m["hosts"]) == \
        ("/c.toml", 100.0, HOSTS)
    assert skill.refresh_reason(installed, state_dir=state, config_mtime=555.0, config_path=None) is None


def test_f6_two_installs_of_one_version_do_not_overwrite_each_other(installed: skill.Env, state: Path) -> None:
    def reason() -> str | None:
        return skill.refresh_reason(installed, state_dir=state, config_mtime=100.0, config_path="/c.toml")
    assert manifest(state)["writer"] == skill.install_identity()
    edit_manifest(state, bundle_sha256="0" * 64, writer="/elsewhere/site-packages/runfleet")
    assert reason() is None                                          # the other install's text: left to it
    edit_manifest(state, writer=skill.install_identity())
    assert reason() == "bundle"                                      # ours, changed (a dev checkout edited it)
    data = manifest(state)
    data.pop("writer")
    (state / "skill.json").write_text(json.dumps(data), encoding="utf-8")
    assert reason() == "bundle"                                      # written before the writer was recorded
    edit_manifest(state, writer="/elsewhere/site-packages/runfleet")
    r = do_update(installed, state)                                  # explicit: always, and takes over the record
    assert r.state == "updated" and manifest(state)["writer"] == skill.install_identity() and reason() is None


def test_e1_a_permanently_failing_refresh_backs_off(installed: skill.Env, state: Path, home: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    if WINDOWS or is_root():
        pytest.skip("needs POSIX permissions that bind")
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    skills = canonical(home).parent
    skills.chmod(0o500)
    calls: list[int] = []
    real = skill._update_locked

    def counting(*a: Any, **k: Any) -> skill.UpdateResult:
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(skill, "_update_locked", counting)
    try:
        with caplog.at_level("INFO", logger="runfleet.skill"):
            line = skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list, now=NOW)
        assert line is not None and line.startswith("skill: refresh failed (version") and "updated" not in line
        assert [r.getMessage() for r in caplog.records] == [line]
        m = manifest(state)
        assert m["last_failure_at"] == NOW.timestamp() and m["last_error"]
        for minutes in (1, 30, 59):                                  # within the hour: not even tried
            soon = datetime.fromtimestamp(NOW.timestamp() + minutes * 60, UTC)
            assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list,
                                      now=soon) is None
        assert calls == [1]
        st = skill.status(installed, state_dir=state)
        assert st.last_failure is not None and "2026-10-18 12:00 UTC" in st.last_failure
        assert any("last refresh failed" in ln for ln in st.lines())
        later = datetime.fromtimestamp(NOW.timestamp() + skill.BACKOFF_SECONDS + 1, UTC)
        assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list,
                                  now=later) is not None
        assert calls == [1, 1]
    finally:
        skills.chmod(0o700)
    skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list,
                       now=datetime.fromtimestamp(NOW.timestamp() + 3 * skill.BACKOFF_SECONDS, UTC))
    assert "last_failure_at" not in manifest(state) and skill.status(installed, state_dir=state).last_failure is None


def test_e1_auto_refresh_logs_exactly_one_line(installed: skill.Env, state: Path, monkeypatch: pytest.MonkeyPatch,
                                               caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    with caplog.at_level("DEBUG", logger="runfleet"):
        line = skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list, now=NOW)
    assert line is not None and [r.getMessage() for r in caplog.records] == [line]


def test_e3_status_names_the_fix_for_an_agent_installed_later(home: Path, state: Path) -> None:
    env = make_env(home)
    do_install(env, state)
    (home / ".claude").mkdir()                                       # Claude Code installed after the skill
    st = skill.status(env, state_dir=state)
    line = next(ln for ln in st.lines() if ln.strip().startswith("Claude Code"))
    assert "not covered" in line and "`runfleet skill install --agent claude` adds it" in line


def test_e3_explicit_update_links_agents_found_since_an_auto_install(home: Path, state: Path,
                                                                     symlinks: None) -> None:
    env = make_env(home)
    do_install(env, state)
    assert manifest(state)["agents_mode"] == "auto"
    (home / ".claude").mkdir()
    (home / ".gemini").mkdir()
    r = do_update(env, state, if_installed=True)                     # automatic / installer: no new links
    assert not os.path.lexists(claude_link(home)) and r.state == "unchanged"
    skill.auto_refresh(env, state_dir=state, config_mtime=1.0, load_host_names=list, now=NOW)
    assert not os.path.lexists(claude_link(home))
    r = do_update(env, state, add_new_agents=True)                   # explicit `skill update`
    assert r.exit_code == 0 and r.state == "updated" and claude_link(home).is_symlink()
    assert any("found (or moved) since the install: Claude Code" in n for n in r.notes)
    m = manifest(state)
    assert "claude" in m["agents"] and "gemini" in m["agents"] and [t["agent"] for t in m["targets"]] == \
        ["shared", "claude"]
    claude_link(home).unlink()                                       # removed by the user: never recreated
    r = do_update(env, state, add_new_agents=True)
    assert not os.path.lexists(claude_link(home)) and [t.action for t in r.targets] == ["unchanged", "missing"]


def test_e3_explicit_agent_list_installs_do_not_grow(home: Path, state: Path, symlinks: None) -> None:
    env = make_env(home)
    do_install(env, state, agents="codex")
    assert manifest(state)["agents_mode"] == "list"
    (home / ".claude").mkdir()
    do_update(env, state, add_new_agents=True)
    assert not os.path.lexists(claude_link(home))
    do_install(env, state, agents="claude")                          # the fix status names; mode stays as widest
    assert claude_link(home).is_symlink() and manifest(state)["agents_mode"] == "list"
    do_install(env, state)
    do_install(env, state, agents="claude")
    assert manifest(state)["agents_mode"] == "auto"


def test_e3_a_foreign_dir_in_the_way_of_a_new_agent_is_left_alone(home: Path, state: Path) -> None:
    env = make_env(home)
    do_install(env, state)
    claude_link(home).mkdir(parents=True)
    (claude_link(home) / "SKILL.md").write_text("theirs", encoding="utf-8")
    r = do_update(env, state, add_new_agents=True)
    assert r.exit_code == 0 and r.targets[-1].action == "foreign" and "--force" in (r.targets[-1].error or "")
    assert (claude_link(home) / "SKILL.md").read_text(encoding="utf-8") == "theirs"
    assert "claude" not in manifest(state)["agents"]


def test_e4_no_silent_downgrade(home: Path, state: Path) -> None:
    env = make_env(home)
    do_install(env, state)
    edit_manifest(state, version="99.0.0")
    r = do_update(env, state, if_installed=True)
    assert r.state == "newer_installed" and r.exit_code == 0 and manifest(state)["version"] == "99.0.0"
    assert r.summary() is not None and "left at 99.0.0" in r.summary() and "downgrade" in r.summary()
    r = do_update(env, state)
    assert r.state == "updated" and r.downgrade and f"downgrading skill 99.0.0 → {runfleet.__version__}" in r.notes
    assert r.summary() == f"runfleet skill downgraded (99.0.0 → {runfleet.__version__}) in ~/.agents/skills/runfleet"
    assert manifest(state)["version"] == runfleet.__version__


def test_e5_status_points_to_the_backup_of_an_automatic_refresh(installed: skill.Env, state: Path, home: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    (canonical(home) / "SKILL.md").write_text("my notes\n", encoding="utf-8")
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list, now=NOW)
    root = state / "skill-backup" / "20261018T120000Z"
    assert manifest(state)["last_backup"] == str(root) and (root / "shared" / "SKILL.md").is_file()
    st = skill.status(installed, state_dir=state)
    assert st.last_backup == root and st.to_json()["last_backup"] == str(root)
    assert any("saved in" in ln and "20261018T120000Z" in ln for ln in st.lines())
    do_update(installed, state, host_names=["other"])               # an update without backups keeps the pointer
    assert manifest(state)["last_backup"] == str(root)


def test_e6_legacy_files_are_not_called_edits_and_unchanged_ones_are_not_saved(home: Path, state: Path,
                                                                               symlinks: None) -> None:
    _legacy_install(home)
    shared = canonical(home)
    ref = "references/reading-output.md"
    (shared / ref).write_bytes(skill.packaged_files()[ref])           # identical to this version's: nothing to save
    r = do_update(claude_env(home), state)
    assert set(backup_files(state)) == {"shared/SKILL.md"} and r.older_backups == 1
    text = "\n".join(r.lines())
    assert "1 file(s) from an older install" in text and "you edited" not in text


def test_e7_stale_claude_link_when_claude_config_dir_is_set(home: Path, state: Path, tmp_path: Path,
                                                            symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)                                           # ~/.claude/skills/runfleet → shared
    moved = make_env(home, environ={"CLAUDE_CONFIG_DIR": str(tmp_path / "cc")})
    st = skill.status(moved, state_dir=state)
    assert [(s.agent, s.kind) for s in st.strays] == [("claude", "stale_link")]
    line = next(ln for ln in st.lines() if "stale link" in ln)
    assert "~/.claude/skills/runfleet" in line and "$CLAUDE_CONFIG_DIR/skills" in line and \
        "`rm ~/.claude/skills/runfleet`" in line and "duplicate" not in line
    # Windows: native separators throughout, and a command that removes the link alone from cmd and PowerShell
    win = make_env(home, environ={"CLAUDE_CONFIG_DIR": str(tmp_path / "cc")}, windows=True)
    line = next(ln for ln in skill.status(win, state_dir=state).lines() if "stale link" in ln)
    assert "stale link: ~\\.claude\\skills\\runfleet →" in line and \
        f'`cmd /c rmdir "{claude_link(home)}"`' in line and "~/" not in line
    # a path either shell would expand inside double quotes: no command, never a wrong one
    st = skill.status(win, state_dir=state)
    odd = skill.Stray(st.strays[0].agent, Path("C:/Users/a%USERNAME%$x/.claude/skills/runfleet"), "stale_link")
    line = next(ln for ln in dataclasses.replace(st, strays=(odd,)).lines() if "stale link" in ln)
    assert "rmdir" not in line and "delete that link (only the link, not the folder it points to)" in line


def test_e7_follow_ups_name_the_files_the_agents_read_here(home: Path, state: Path, tmp_path: Path) -> None:
    env = make_env(home, environ={"CLAUDE_CONFIG_DIR": str(tmp_path / "cc"), "CODEX_HOME": "~/codex-home"})
    r = do_install(env, state)
    assert f"{tmp_path / 'cc' / 'settings.json'}" in r.permission_hint and "~/.claude" not in r.permission_hint
    u = skill.uninstall(env, state_dir=state, now=NOW)
    text = "\n".join(u.follow_ups)
    assert "~/codex-home/config.toml" in text and str(tmp_path / "cc" / "settings.json") in text
    assert "~/.codex" not in text and "~/.claude" not in text
    plain = skill.uninstall(make_env(home), state_dir=state, now=NOW)
    assert plain.follow_ups == skill.FOLLOW_UPS


def test_e7_pi_needs_its_dir_or_the_real_pi_program(home: Path, tmp_path: Path, symlinks: None) -> None:
    pi = skill._BY_ID["pi"]
    other = tmp_path / "other" / "pi"
    other.parent.mkdir()
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    some_pi = skill.Env(home=home, environ={}, is_windows=False, which=lambda n: str(other) if n == "pi" else None)
    assert not skill._detected(some_pi, pi)                          # some other `pi` on PATH
    real = tmp_path / "lib" / "node_modules" / "@earendil-works" / "pi-coding-agent" / "dist" / "cli.js"
    real.parent.mkdir(parents=True)
    real.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    os.symlink(real, tmp_path / "bin" / "pi")
    env = skill.Env(home=home, environ={}, is_windows=False,
                    which=lambda n: str(tmp_path / "bin" / "pi") if n == "pi" else None)
    assert skill._detected(env, pi)
    assert skill._detected(make_env(home, environ={"PI_CODING_AGENT_DIR": "/x"}), pi)
    (home / ".pi").mkdir()
    assert skill._detected(make_env(home), pi)


def test_e7_uninstall_removes_the_lock_and_the_empty_dirs_it_created(home: Path, state: Path,
                                                                     symlinks: None) -> None:
    env = claude_env(home)                                           # ~/.claude exists; its skills dir does not
    do_install(env, state)
    assert (state / skill.LOCK_FILE).exists()
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert sorted(p.name for p in home.iterdir()) == [".claude"] and not (home / ".claude" / "skills").exists()
    assert {p.name for p in u.removed_dirs} == {"skills", ".agents"} and not (state / skill.LOCK_FILE).exists()
    assert any("empty folder(s) RunFleet had created" in ln for ln in u.lines())
    # a dir that held something else, or that existed before, stays
    (home / ".agents" / "skills").mkdir(parents=True)
    do_install(env, state)
    (home / ".agents" / "skills" / "other-skill").mkdir()
    u = skill.uninstall(env, state_dir=state, now=NOW)
    assert (home / ".agents" / "skills" / "other-skill").is_dir() and [p.name for p in u.removed_dirs] == ["skills"]
    assert (home / ".claude").is_dir()
    u = skill.uninstall(env, state_dir=state, now=NOW, dry_run=True)
    assert not (state / skill.LOCK_FILE).exists()


def test_e7_a_lock_on_a_deleted_lock_file_is_taken_again(state: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uninstall deletes skill.lock while holding it: whoever then gets the lock on the deleted file must take the
    new one instead, or two commands would hold "the" lock at once."""
    if WINDOWS:
        pytest.skip("an open file cannot be deleted on Windows")
    state.mkdir()
    lock = state / skill.LOCK_FILE
    real = rf_platform.try_lock
    swapped: list[int] = []

    def racing(fd: int) -> bool:
        if not swapped:                       # the file is replaced between our open and our lock
            lock.unlink()
            lock.write_text("", encoding="utf-8")
            swapped.append(fd)
        return bool(real(fd))
    monkeypatch.setattr(rf_platform, "try_lock", racing)
    with skill._locked(state, wait=1.0) as held:
        assert held and len(swapped) == 1
    fd = os.open(lock, os.O_RDWR)
    try:
        assert skill._still_named(fd, lock)
        lock.unlink()
        assert not skill._still_named(fd, lock)
    finally:
        os.close(fd)


def test_codex_review_backup_names_never_collide(tmp_path: Path, symlinks: None) -> None:
    """A link `a` (saved as a.link.txt) next to a file the user named `A.link.txt`: both kept."""
    d = tmp_path / "copy"
    d.mkdir()
    os.symlink(tmp_path, d / "a", target_is_directory=True)
    (d / "A.link.txt").write_text("my file", encoding="utf-8")
    b = skill._Backup(tmp_path / "state", NOW)
    saved = b.copy_files("shared", d, sorted(skill._tree(d)))
    assert len({p.as_posix().lower() for p in saved}) == 2
    texts = sorted(p.read_text(encoding="utf-8") for p in saved)
    assert texts == [f"link to {tmp_path}\n", "my file"]


def test_codex_review_created_dirs_are_never_removed_through_a_link(home: Path, state: Path, tmp_path: Path,
                                                                    symlinks: None) -> None:
    env = claude_env(home)
    do_install(env, state)
    assert str(home / ".claude" / "skills") in manifest(state)["created_dirs"]
    claude_link(home).unlink()
    (home / ".claude" / "skills").rmdir()
    os.rename(home / ".claude", tmp_path / "claude-moved")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "skills").mkdir(parents=True)                       # someone else's empty skills dir
    os.symlink(elsewhere, home / ".claude", target_is_directory=True)
    skill.uninstall(env, state_dir=state, now=NOW)
    assert (elsewhere / "skills").is_dir() and not (home / ".agents").exists()


def test_codex_review_a_moved_agent_home_gets_its_link_on_explicit_update(home: Path, state: Path, tmp_path: Path,
                                                                         symlinks: None) -> None:
    (home / ".copilot").mkdir()
    env = claude_env(home)
    do_install(env, state)                                           # Copilot reads the shared copy then
    moved = make_env(home, environ={"COPILOT_HOME": str(tmp_path / "cp"), "CLAUDE_CONFIG_DIR": str(tmp_path / "cc")})
    do_update(moved, state, if_installed=True)
    assert not (tmp_path / "cp" / "skills" / "runfleet").exists()
    r = do_update(moved, state, add_new_agents=True)
    assert r.exit_code == 0
    for p in (tmp_path / "cp" / "skills" / "runfleet", tmp_path / "cc" / "skills" / "runfleet"):
        assert p.is_symlink() and p.resolve() == canonical(home).resolve()
    assert claude_link(home).is_symlink()                            # the old one: now a stale link, left in place
    st = skill.status(moved, state_dir=state)
    assert ("claude", "stale_link") in [(s.agent, s.kind) for s in st.strays]


def test_codex_review_a_raised_refresh_is_recorded_under_the_lock(installed: skill.Env, state: Path,
                                                                  monkeypatch: pytest.MonkeyPatch,
                                                                  caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    held: list[bool] = []
    real = skill._note_failure

    def note(*a: Any) -> None:
        held.append(skill._THREAD_LOCK.locked())
        real(*a)

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(skill, "_note_failure", note)
    monkeypatch.setattr(skill, "_update_locked", boom)
    with caplog.at_level("DEBUG", logger="runfleet"):
        assert skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list,
                                  now=NOW) is None
    assert held == [True] and manifest(state)["last_failure_at"] == NOW.timestamp()
    assert [r.getMessage() for r in caplog.records] == ["skill: refresh failed: disk on fire"]


def test_codex_review_file_diagnostics_fold_into_the_one_log_line(installed: skill.Env, state: Path, home: Path,
                                                                  monkeypatch: pytest.MonkeyPatch,
                                                                  caplog: pytest.LogCaptureFixture) -> None:
    lookalike = canonical(home).parent / ".runfleet.old-0123abcd"
    lookalike.mkdir()                                                # not ours: kept, and said so
    monkeypatch.setattr(runfleet, "__version__", "99.0.0")
    with caplog.at_level("DEBUG", logger="runfleet"):
        line = skill.auto_refresh(installed, state_dir=state, config_mtime=100.0, load_host_names=list, now=NOW)
    assert line is not None and "looks like a leftover" in line
    assert [r.getMessage() for r in caplog.records] == [line]


def test_codex_review_a_link_record_never_takes_a_directory_the_users_files_need(tmp_path: Path,
                                                                                  symlinks: None) -> None:
    d = tmp_path / "copy"
    (d / "a.link.txt").mkdir(parents=True)
    (d / "a.link.txt" / "notes.md").write_text("mine", encoding="utf-8")
    os.symlink(tmp_path, d / "a", target_is_directory=True)
    b = skill._Backup(tmp_path / "state", NOW)
    saved = b.copy_files("shared", d, sorted(skill._tree(d)))
    rels = sorted(p.relative_to(b.root).as_posix() for p in saved)
    assert rels == ["shared/a.link-2.txt", "shared/a.link.txt/notes.md"]
    assert (b.root / "shared" / "a.link.txt" / "notes.md").read_text(encoding="utf-8") == "mine"


def test_codex_review_links_removed_under_an_old_manifest_stay_removed(home: Path, state: Path,
                                                                       symlinks: None) -> None:
    """A manifest written before `linked_paths` existed: a link the user removed is not recreated, ever."""
    env = claude_env(home)
    do_install(env, state)
    data = manifest(state)
    data.pop("linked_paths")
    (state / "skill.json").write_text(json.dumps(data), encoding="utf-8")
    claude_link(home).unlink()
    for _ in range(3):
        do_update(env, state, add_new_agents=True)
        assert not os.path.lexists(claude_link(home))
    assert manifest(state)["linked_paths"] == [str(claude_link(home))]
