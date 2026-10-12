"""[WP2] Every field-probe fixture in tests/fixtures/samples is referenced by at least one test (§10.2), except the
explicit allowlist (identity, pam_adopt_login, *_source, *_json*, *_size, gpu_query_timing, nvidia_smi_full, os_release,
top_procs, commands, squeue_me_default, scontrol_listpids* — Slurm process lists are v0.2 per D7)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from tests.conftest import REPO_ROOT, SAMPLES_DIR

ALLOWLIST = ("identity", "pam_adopt_login", "*_source", "*_json*", "*_size", "gpu_query_timing", "nvidia_smi_full",
             "os_release", "top_procs", "commands", "squeue_me_default", "scontrol_listpids*", "_meta*")

TEST_DIRS = [REPO_ROOT / "tests"]


def _test_sources() -> str:
    texts = []
    for d in TEST_DIRS:
        for p in d.rglob("*.py"):
            if "__pycache__" not in p.parts and p.name != Path(__file__).name:
                texts.append(p.read_text(encoding="utf-8"))
    return "\n".join(texts)


def _fixture_names() -> set[str]:
    return {p.stem for p in SAMPLES_DIR.glob("*/*.txt")}


def _allowed(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in ALLOWLIST)


def test_every_fixture_is_referenced() -> None:
    src = _test_sources()
    missing = sorted(n for n in _fixture_names()
                     if not _allowed(n) and not re.search(rf"[\"']{re.escape(n)}[\"']", src))
    assert not missing, f"fixtures not referenced by any test: {missing}"


def test_allowlist_entries_exist() -> None:
    names = _fixture_names()
    for pat in ALLOWLIST:
        assert any(fnmatch.fnmatch(n, pat) for n in names), pat
