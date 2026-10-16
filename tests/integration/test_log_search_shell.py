"""[WP1] Shell integration for the cross-job log search (§4.7.1): the real `log.search` section through sh / dash /
bash with GNU grep, tail, head and stat on real files — the 256 KiB window pinned to an absolute offset, the 1 MiB read
budget, the match cap, quoting of hostile paths and patterns, unreadable files — and the demo's play of the same
script, byte for byte."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from runfleet.core import logsearch
from runfleet.core.logsearch import LogQuery, LogSearchResult, SearchFile, SearchPlan
from runfleet.demo import render
from runfleet.demo.world import SimLog
from runfleet.models import JobKey
from runfleet.probe.section import SectionOutcome, SectionResult
from tests.integration.harness import MockEnv, run_sections, shells

pytestmark = pytest.mark.posix                     # real sh / dash / bash

KIB = 1024


def _plan(paths: Sequence[str]) -> SearchPlan:
    return SearchPlan("k2", tuple(SearchFile(p, JobKey("k2", str(100 + i)), f"job{i}", "stdout")
                                  for i, p in enumerate(paths)), len(paths))


def _run(env: MockEnv, paths: Sequence[str], query: LogQuery, *, shell: str = "/bin/sh",
         max_matches: int = logsearch.MAX_MATCHES) -> tuple[LogSearchResult, SectionResult]:
    sec = logsearch.search_section(paths, query, max_matches=max_matches)
    res = run_sections([sec], env.env(), shell=shell, purpose="log")["log.search"]
    return logsearch.parse("k2", query, _plan(paths), res, max_matches=max_matches), res


def _demo(paths: Sequence[str], query: LogQuery, *, max_matches: int = logsearch.MAX_MATCHES) -> SectionResult:
    files = {}
    for p in paths:
        if os.path.isfile(p) and os.access(p, os.R_OK):
            lg = files[p] = SimLog(p, 0.0)
            lg.write(Path(p).read_bytes().decode("utf-8"), 0.0)
    m = logsearch.search_section(paths, query, max_matches=max_matches).meta
    out = render.log_search(SimpleNamespace(logs=files), paths=list(m["paths"]), pattern=str(m["pattern"]),
                            icase=bool(m["icase"]), max_matches=int(m["max_matches"]), per_file=int(m["per_file"]),
                            budget=int(m["budget"]))
    return SectionResult("log.search", SectionOutcome.OK, 0, out.stdout, "", 1)


def _filler(n: int, tag: str) -> str:
    """``n`` bytes of plain log lines."""
    line = f"[{tag}] epoch 12/300  train_loss 0.3140  vs_base -2.38%  212 it/s\n"
    text = line * (n // len(line) + 1)
    return text[:n - 1] + "\n"


@pytest.fixture
def env(tmp_path: Path) -> MockEnv:
    return MockEnv(tmp_path / "env", "k2", mocks=())


def _assert_anchored(path: Path, r: LogSearchResult) -> None:
    """Every hit: its excerpt sits at its absolute offset, and its grep -n line of the window holds its needle."""
    data = path.read_bytes()
    for hit in r.hits:
        assert data[hit.offset:].startswith(hit.excerpt.encode("utf-8")), hit
        assert hit.needle in data[hit.start:].split(b"\n")[hit.line - 1].decode("utf-8", "replace"), hit


@pytest.mark.parametrize("shell", shells())
def test_window_is_the_last_256_kib_at_an_absolute_offset(env: MockEnv, tmp_path: Path, shell: str) -> None:
    log = tmp_path / "big.out"
    head = "early ERROR boom (outside the window)\n" + _filler(100 * KIB, "a")
    oom = "[2026-09-07 05:55:35] rank 0: torch.OutOfMemoryError: CUDA out of memory. ERROR boom"
    tail = _filler(200 * KIB, "b") + oom + "\n" + _filler(30 * KIB, "c")
    log.write_text(head + tail + "late error BOOM\n")
    r, raw = _run(env, [str(log)], LogQuery("error boom"), shell=shell)
    size = log.stat().st_size
    assert raw.outcome is SectionOutcome.OK and r.error is None
    f = r.files[0]
    assert (f.status, f.size, f.start, f.searched) == ("searched", size, size - 256 * KIB, 256 * KIB)
    assert f.window_cut and not f.budget_cut
    assert [h.needle for h in r.hits] == [oom[-70:], "late error BOOM"]   # 60 bytes of context before the match
    _assert_anchored(log, r)
    case, _ = _run(env, [str(log)], LogQuery("ERROR boom", match_case=True), shell=shell)
    assert [h.needle for h in case.hits] == [oom[-70:]]


@pytest.mark.parametrize("shell", shells())
def test_hostile_paths_and_literal_specials(env: MockEnv, tmp_path: Path, shell: str) -> None:
    odd = tmp_path / "it's $(touch pwned) `id` *.out"
    odd.write_text("aXb[c](d)\nresult a.b[c](d)*+?{1}|^$\\e] ok\n损失 nan at step 3\n")
    star = tmp_path / "x.out"
    star.write_text("glob must not expand\n")
    r, _ = _run(env, [str(odd), str(tmp_path / "*.out")], LogQuery("a.b[c](d)*+?{1}|^$\\e]"), shell=shell)
    assert [h.needle for h in r.hits] == ["result a.b[c](d)*+?{1}|^$\\e] ok"]
    assert [f.status for f in r.files] == ["searched", "missing"]       # "*.out" is a name, never a glob
    assert not (Path.cwd() / "pwned").exists() and not (tmp_path / "pwned").exists()
    cjk, _ = _run(env, [str(odd)], LogQuery("损失"), shell=shell)
    assert [h.needle for h in cjk.hits] == ["损失 nan at step 3"]
    _assert_anchored(odd, cjk)
    rx, _ = _run(env, [str(odd)], LogQuery(r"step [[:digit:]]+|a\.b", regex=True), shell=shell)
    assert [h.line for h in rx.hits] == [2, 3]


@pytest.mark.parametrize("shell", shells())
def test_read_budget_match_cap_and_unreadable_files(env: MockEnv, tmp_path: Path, shell: str) -> None:
    paths = []
    for i, size in enumerate((100 * KIB, 300 * KIB, 300 * KIB, 300 * KIB, 300 * KIB, 10 * KIB)):
        p = tmp_path / f"f{i}.out"
        last = f"needle in f{i}\n"
        p.write_text(_filler(size - len(last), f"f{i}") + last)
        paths.append(str(p))
    r, _ = _run(env, paths, LogQuery("needle"), shell=shell)
    assert [f.status for f in r.files] == ["searched"] * 5 + ["skipped (budget)"]
    assert [f.searched for f in r.files[:5]] == [100 * KIB, 256 * KIB, 256 * KIB, 256 * KIB, 156 * KIB]
    assert r.bytes_read == logsearch.READ_BUDGET and r.files[4].budget_cut
    assert len(r.hits) == 5                                             # f4's needle is at its end: still read
    capped, _ = _run(env, paths, LogQuery("epoch"), shell=shell, max_matches=7)
    assert len(capped.hits) == 7 and capped.match_capped
    assert [f.status for f in capped.files][1:] == ["skipped (matches)"] * 5
    locked = tmp_path / "locked.out"
    locked.write_text("needle\n")
    locked.chmod(0)
    others = [str(tmp_path / "gone.out"), "/dev/null", str(locked), str(tmp_path)]
    bad, _ = _run(env, others, LogQuery("needle"), shell=shell)
    statuses = [f.status for f in bad.files]
    assert statuses[:2] == ["missing", "not a file"] and statuses[3] == "not a file"
    assert statuses[2] == ("searched" if os.geteuid() == 0 else "denied")


@pytest.mark.parametrize("shell", shells())
def test_bad_regex_and_carriage_returns(env: MockEnv, tmp_path: Path, shell: str) -> None:
    log = tmp_path / "tqdm.out"
    log.write_text("start\n" + "".join(f" {i}%|{'#' * (i // 10)}| loss=0.{i:02d}\r" for i in range(0, 100, 7))
                   + " 100%| done loss=nan\nend\n")
    bad, raw = _run(env, [str(log)], LogQuery("loss=[0", regex=True), shell=shell)
    assert raw.stdout.startswith("@@bad grep: ") and (bad.error or "").startswith("grep: ")
    assert bad.hits == () and [f.status for f in bad.files] == ["not reached"]
    r, _ = _run(env, [str(log)], LogQuery("loss=nan"), shell=shell)
    assert [h.needle for h in r.hits] == [" 100%| done loss=nan"] and r.hits[0].line == 2
    _assert_anchored(log, r)


def test_the_demo_plays_the_script_byte_for_byte(env: MockEnv, tmp_path: Path) -> None:
    paths = []
    for i, size in enumerate((40 * KIB, 300 * KIB, 280 * KIB, 300 * KIB, 300 * KIB)):
        p = tmp_path / f"d{i}.out"
        p.write_text(_filler(size // 2, f"d{i}") + f"ERROR boom {i} 损失\n" + _filler(size // 2, f"e{i}"))
        paths.append(str(p))
    paths.append(str(tmp_path / "missing.out"))
    for query, cap in ((LogQuery("error BOOM"), 200), (LogQuery("epoch 1[0-9]/", regex=True), 9),
                       (LogQuery("损失", match_case=True), 200)):
        _r, real = _run(env, paths, query, max_matches=cap)
        assert _demo(paths, query, max_matches=cap).stdout == real.stdout, query
