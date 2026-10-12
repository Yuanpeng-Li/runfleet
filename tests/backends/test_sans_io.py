"""[WP2] Collectors are sans-IO (§3.1, D4): no asyncio/subprocess/textual/socket imports, no clock reads, no file or
network access anywhere under `backends/` (time comes from HostContext.now)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import runfleet

BACKENDS = Path(runfleet.__file__).resolve().parent / "backends"
FILES = sorted(p for p in BACKENDS.rglob("*.py") if "__pycache__" not in p.parts)
FORBIDDEN_MODULES = {"asyncio", "subprocess", "textual", "socket", "threading", "multiprocessing", "time",
                     "sqlite3", "os", "shutil", "urllib", "http"}
FORBIDDEN_CALLS = {("datetime", "now"), ("datetime", "utcnow"), ("datetime", "today"), ("date", "today"),
                   ("time", "time"), ("time", "monotonic"), ("os", "system")}


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(BACKENDS)))
def test_no_io_modules(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not (_imports(tree) & FORBIDDEN_MODULES)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(BACKENDS)))
def test_no_clock_or_file_calls(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                assert f.id not in ("open", "print", "input", "exec", "eval"), f"{path.name}: {f.id}()"
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                assert (f.value.id, f.attr) not in FORBIDDEN_CALLS, f"{path.name}: {f.value.id}.{f.attr}()"


def test_checker_catches_violations(tmp_path: Path) -> None:
    bad = ast.parse("import subprocess\nfrom datetime import datetime\nx = datetime.now()\n")
    assert "subprocess" in _imports(bad)
    calls = [(n.func.value.id, n.func.attr) for n in ast.walk(bad) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)]
    assert ("datetime", "now") in calls
