"""update.py (§8.6): install detection, version compare, CHANGELOG excerpt, the daily check and its state file,
relaunch command. No network: fetches are injected."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runfleet import update

T0 = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
GIT = "https://github.com/Yuanpeng-Li/runfleet"


def which_all(name: str) -> str | None:
    return f"/usr/bin/{name}"


def which_none(name: str) -> str | None:
    return None


def receipt(prefix: Path, requirement: str, bin_dir: str = "") -> Path:
    prefix.mkdir(parents=True)
    entry = f'entrypoints = [{{ name = "runfleet", install-path = "{bin_dir}/runfleet", from = "runfleet" }}]\n'
    (prefix / "uv-receipt.toml").write_text(f"[tool]\nrequirements = [{{ name = \"runfleet\", {requirement} }}]\n"
                                            + (entry if bin_dir else ""))
    return prefix


def detect(prefix: Path, which=which_all, package_dir: Path | None = None, home: Path | None = None
           ) -> update.Install:
    return update.detect_install(prefix, package_dir or prefix / "lib" / "site-packages" / "runfleet", which=which,
                                 home=home or prefix, executable="/venv/bin/python")


def test_uv_tool_from_git(tmp_path: Path) -> None:
    prefix = receipt(tmp_path / "tools" / "runfleet", f'git = "{GIT}"', bin_dir="/opt/bin")
    inst = detect(prefix)
    assert (inst.kind, inst.steps, inst.manual, inst.note) == (
        "uv", (("/usr/bin/uv", "tool", "upgrade", "runfleet"),), "uv tool upgrade runfleet", "")
    assert inst.env == {"UV_TOOL_DIR": str(tmp_path / "tools"), "UV_TOOL_BIN_DIR": "/opt/bin"}   # as installed
    assert inst.describe() == f"installed with uv tool from {GIT}"


def test_uv_tool_pinned_to_a_tag_moves_to_main(tmp_path: Path) -> None:
    inst = detect(receipt(tmp_path / "runfleet", f'git = "{GIT}?rev=v0.2.1"'))
    assert inst.steps == (("/usr/bin/uv", "tool", "install", "--force", f"git+{GIT}"),)
    assert inst.source == f"{GIT}@v0.2.1" and "pinned to v0.2.1" in inst.note


def test_uv_tool_from_a_clone_pulls_and_reinstalls(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    inst = detect(receipt(tmp_path / "a" / "runfleet", f'directory = "{clone}"'))
    assert inst.steps == (("/usr/bin/git", "-C", str(clone), "pull", "--ff-only"),
                          ("/usr/bin/uv", "tool", "install", "--force", "--reinstall", str(clone)))
    editable = detect(receipt(tmp_path / "b" / "runfleet", f'editable = "{clone}"'))
    assert editable.steps == (("/usr/bin/git", "-C", str(clone), "pull", "--ff-only"),)
    no_git = detect(receipt(tmp_path / "c" / "runfleet", f'directory = "{tmp_path}"'))
    assert no_git.steps == () and "pull" in no_git.manual            # not a checkout: by hand


def test_uv_missing_or_found_outside_path(tmp_path: Path) -> None:
    prefix = receipt(tmp_path / "runfleet", f'git = "{GIT}"')
    assert detect(prefix, which=which_none, home=tmp_path / "home").steps == ()
    uv = tmp_path / "home" / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    assert detect(prefix, which=which_none, home=tmp_path / "home").steps[0][0] == str(uv)


def test_pipx_source_uvx_and_pip(tmp_path: Path) -> None:
    pipx = tmp_path / "pipx" / "venvs" / "runfleet"
    pipx.mkdir(parents=True)
    (pipx / "pipx_metadata.json").write_text("{}")
    assert detect(pipx).steps == (("/usr/bin/pipx", "reinstall", "runfleet"),)
    checkout = tmp_path / "src-checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    src = detect(tmp_path / "venv", package_dir=checkout / "src" / "runfleet")
    assert (src.kind, src.steps) == ("source", (("/usr/bin/git", "-C", str(checkout), "pull", "--ff-only"),))
    uvx = detect(tmp_path / "cache" / "uv" / "archive-v0" / "abc")
    assert (uvx.kind, uvx.steps) == ("uvx", ()) and "--refresh" in uvx.manual
    pip = detect(tmp_path / "venv")
    assert (pip.kind, pip.steps, pip.manual) == ("pip", (), f"/venv/bin/python -m pip install --upgrade git+{GIT}")


@pytest.mark.parametrize(("latest", "current", "newer"), [
    ("0.4.0", "0.3.0", True), ("0.3.10", "0.3.9", True), ("0.3.0", "0.3.0", False), ("0.2.9", "0.3.0", False),
    ("1.0.0rc1", "0.3.0", False), ("v0.4.0", "0.3.0", True),
])
def test_is_newer(latest: str, current: str, newer: bool) -> None:
    assert update.is_newer(latest, current) is newer


CHANGELOG = """# Changelog

## [Unreleased]

## [0.5.0] - 2026-11-01

### Added
- five

## [0.4.0] - 2026-10-20

### Fixed
- four

## [0.3.0] - 2026-10-13
- three

[Unreleased]: https://github.com/x/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/x/compare/v0.4.0...v0.5.0
"""


def test_changes_since_lists_newer_sections_only() -> None:
    assert update.changes_since(CHANGELOG, "0.3.0") == [
        "## [0.5.0] - 2026-11-01", "", "### Added", "- five", "",
        "## [0.4.0] - 2026-10-20", "", "### Fixed", "- four"]
    assert update.changes_since(CHANGELOG, "0.5.0") == []
    assert update.changes_since(CHANGELOG, "0.2.0", max_lines=3)[-1].startswith("… (more in")


def test_check_asks_once_a_day_and_keeps_the_answer(tmp_path: Path) -> None:
    path = tmp_path / "update.json"
    asked: list[int] = []

    def fetch() -> str:
        asked.append(1)
        return "0.4.0"

    first = update.check(T0, path=path, fetch=fetch, current="0.3.0")
    assert (first.latest, first.available, first.checked_at, len(asked)) == ("0.4.0", True, T0, 1)
    again = update.check(T0 + timedelta(hours=23), path=path, fetch=fetch, current="0.3.0")
    assert (again.latest, again.checked_at, len(asked)) == ("0.4.0", T0, 1)          # the saved answer
    update.check(T0 + timedelta(hours=1), path=path, fetch=fetch, current="0.3.0", force=True)
    update.check(T0 + timedelta(days=2), path=path, fetch=fetch, current="0.3.0")
    assert len(asked) == 3
    assert json.loads(path.read_text())["latest"] == "0.4.0"
    assert path.stat().st_mode & 0o077 == 0


def test_a_failed_check_keeps_the_last_answer_and_says_why(tmp_path: Path) -> None:
    path = tmp_path / "update.json"
    update.check(T0, path=path, fetch=lambda: "0.4.0", current="0.3.0")

    def offline() -> str:
        raise OSError("Network is unreachable")

    status = update.check(T0 + timedelta(days=2), path=path, fetch=offline, current="0.3.0")
    assert (status.latest, status.checked_at, status.error) == ("0.4.0", T0, "Network is unreachable")
    assert update.check(T0, path=tmp_path / "none.json", fetch=offline, current="0.3.0").latest is None


def test_each_new_version_is_announced_once(tmp_path: Path) -> None:
    path = tmp_path / "update.json"
    status = update.check(T0, path=path, fetch=lambda: "0.4.0", current="0.3.0")
    assert update.should_notify(status, path=path)
    update.mark_notified("0.4.0", path=path)
    assert not update.should_notify(status, path=path)
    assert json.loads(path.read_text())["latest"] == "0.4.0"                       # kept beside the mark
    newer = update.check(T0, path=path, fetch=lambda: "0.5.0", current="0.3.0", force=True)
    assert update.should_notify(newer, path=path)
    assert not update.should_notify(update.Status("0.3.0", "0.3.0", T0), path=path)


def test_relaunch_argv(tmp_path: Path) -> None:
    script = tmp_path / "bin" / "runfleet"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    assert update.relaunch_argv(["--demo"], argv0=str(script), executable="/py") == [str(script), "--demo"]
    assert update.relaunch_argv(["k2"], argv0="runfleet", executable="/py",
                                which=lambda n: "/home/u/.local/bin/runfleet") == ["/home/u/.local/bin/runfleet", "k2"]
    assert update.relaunch_argv([], argv0="/x/runfleet/__main__.py", executable="/py") == ["/py", "-m", "runfleet"]
