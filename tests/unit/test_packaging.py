"""Packaging guard: the sdist is a whitelist and never ships local-only development notes."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ONLY = ("/docs/research", "/docs/integration-notes.md")     # gitignored dev notes, never published


def _sdist() -> dict[str, list[str]]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["sdist"]


def test_sdist_is_a_whitelist_without_local_notes() -> None:
    sdist = _sdist()
    include = sdist["include"]
    assert "/src/runfleet" in include and "/README.md" in include and "/LICENSE" in include
    assert not any(p in ("/docs", "/docs/", "docs", "*") or p.startswith(LOCAL_ONLY) for p in include)
    for path in LOCAL_ONLY:
        assert path in sdist["exclude"]


def test_local_notes_are_gitignored() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for path in LOCAL_ONLY:
        assert path in ignored or path + "/" in ignored


def test_sdist_does_not_ship_tests() -> None:
    """Review BACK-48: the test suite (and its fixtures) is for git checkouts; the sdist is package + docs only."""
    assert not any(p.rstrip("/") in ("/tests", "tests") for p in _sdist()["include"])


def test_field_fixtures_live_in_tests() -> None:
    samples = ROOT / "tests" / "fixtures" / "samples"
    assert samples.is_dir() and any(samples.glob("*/*.txt"))


def test_no_two_tracked_files_differ_only_in_case() -> None:
    """Windows and macOS file systems are case-insensitive: squeue_me_O.txt and squeue_me_o.txt became one file on the
    Windows CI runner, so every squeue test there read the wrong fixture."""
    import shutil
    import subprocess
    from collections import defaultdict

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    seen: dict[str, list[str]] = defaultdict(list)
    for f in files:
        seen[f.lower()].append(f)
    assert not [v for v in seen.values() if len(v) > 1]
