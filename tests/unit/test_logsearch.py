"""[WP1] Bounded cross-job log search (§4.7.1): scope, pattern rules, the `log.search` command, parsing, excerpts,
and LogService.search over a FakeTransport."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from runfleet.core import logsearch
from runfleet.core.clock import ManualClock
from runfleet.core.logs import LogService, tail_spec
from runfleet.core.logsearch import LogQuery, SearchFile, SearchPlan
from runfleet.models import ErrorKind, HostError, JobKey, JobState, LogRef
from runfleet.probe.section import ProbeScript, SectionOutcome, SectionResult
from runfleet.transport.base import ProbeResult
from runfleet.transport.fake import FakeTransport
from runfleet.transport.spawn import failed_probe
from tests.ui.fakes import T0, slurm_job

DAY = 86400.0


def _job(jid: str, state: JobState = JobState.COMPLETED, *, ended_ago: float | None = 3600.0, name: str = "train",
         user: str = "alice0001", logs: tuple[LogRef, ...] | None = None, started_ago: float = 7200.0):
    job = slurm_job("k2", jid, name, state, elapsed=timedelta(seconds=600),
                    ended_ago=ended_ago if state.is_terminal else None, user=user)
    if not state.is_terminal:
        job = dataclasses.replace(job, start_time=T0 - timedelta(seconds=started_ago))
    if logs is not None:
        job = dataclasses.replace(job, logs=logs)
    return job


# ============================================================================================ scope
def test_scope_is_own_running_and_recently_ended_jobs_most_recent_first() -> None:
    jobs = [
        _job("100", ended_ago=2 * DAY),                                   # ended before the window
        _job("101", ended_ago=600),
        _job("102", JobState.RUNNING, started_ago=60),
        _job("103", JobState.RUNNING, started_ago=6000),
        _job("104", JobState.PENDING),                                    # no log yet
        _job("105", ended_ago=60, user="bob0002"),                        # not mine
        _job("106", JobState.FAILED, ended_ago=30),
    ]
    p = logsearch.plan("k2", jobs, T0, DAY)
    assert [f.key.id for f in p.files] == ["102", "103", "106", "101"]
    assert p.jobs == 4 and p.left_out == 0 and p.no_path == 0
    assert all(f.stream == "stdout" and f.merged for f in p.files)      # stderr merged: one file, not two
    assert logsearch.plan("vega", jobs, T0, DAY).files == ()            # other hosts' jobs never count


def test_scope_dedupes_paths_counts_sharers_and_caps_at_twenty_files() -> None:
    shared = (LogRef("stdout", "/w/train.log"), LogRef("stderr", "/w/train.err"))
    jobs = [_job("200", ended_ago=10, logs=shared), _job("201", ended_ago=20, logs=shared)]
    jobs += [_job(str(300 + i), ended_ago=100 + i, logs=(LogRef("stdout", f"/w/{i}.out"),)) for i in range(25)]
    jobs.append(_job("400", ended_ago=5, logs=(LogRef("stdout", None, pattern="/w/%j.out"),)))
    jobs.append(_job("401", ended_ago=6, logs=(LogRef("stdout", "relative/x.out"), LogRef("stderr", "/w/a\nb"))))
    p = logsearch.plan("k2", jobs, T0, DAY)
    assert len(p.files) == logsearch.MAX_FILES and p.left_out == 2 + 25 - logsearch.MAX_FILES
    first = p.files[0]
    assert (first.path, first.key.id, first.stream, first.others) == ("/w/train.log", "200", "stdout", 1)
    assert (p.files[1].path, p.files[1].stream, p.files[1].merged) == ("/w/train.err", "stderr", False)
    assert p.no_path == 1                                                 # 400: only a pattern
    assert all(f.path.startswith("/") and "\n" not in f.path for f in p.files)   # 401's paths never run


# ============================================================================================ patterns
@pytest.mark.parametrize(("query", "problem"), [
    (LogQuery(""), "type a pattern"),
    (LogQuery("x" * 201), "at most 200 characters"),
    (LogQuery("a\nb"), "one line only"),
    (LogQuery(r"(a)\1", regex=True), "back-references"),
    (LogQuery("a)(b", regex=True), "unbalanced parentheses"),
    (LogQuery("(a|b", regex=True), "unbalanced parentheses"),
    (LogQuery(r"\(x[)(]", regex=True), None),                           # escaped / bracketed parens do not count
    (LogQuery("a)(b"), None),                                            # literal: anything on one line goes
    (LogQuery("CUDA out of memory"), None),
    # counted repetitions grep expands when it compiles: nested ones took > 3 GB on the login node (C locale)
    (LogQuery("((a{1,255}){1,255}){1,255}", regex=True), "repetition"),
    (LogQuery("(.{1,100}){1,100}", regex=True), "repetition"),
    (LogQuery("x{1,32767}", regex=True), "repetition"),
    (LogQuery("x{300,}", regex=True), "repetition"),
    (LogQuery("x{1,32767}{1,32767}", regex=True), "repetition"),
    (LogQuery("(.{1,30}){1,30}", regex=True), None),                     # 900 positions: a few MB
    (LogQuery("[0-9]{4}-[0-9]{2}-[0-9]{2}", regex=True), None),
    (LogQuery("loss[ =]+[0-9.]+(e-?[0-9]+)?", regex=True), None),
    (LogQuery("x{1,32767}"), None),                                      # literal: braces are text
])
def test_pattern_problems(query: LogQuery, problem: str | None) -> None:
    got = logsearch.pattern_problem(query)
    assert (got is None) if problem is None else (got is not None and problem in got)


def test_literal_becomes_an_ere_without_stray_escapes() -> None:
    assert logsearch.literal_ere("a.b[c](d)*+?{1}|^$\\e]") == r"a[.]b[[]c[]][(]d[)][*][+][?][{]1[}][|]\^[$][\]e[]]"
    q = LogQuery("loss=nan (step 3)")
    assert logsearch.grep_pattern(q) == ".{0,60}(loss=nan [(]step 3[)]).{0,60}"
    assert logsearch.grep_pattern(LogQuery("err(or)?", regex=True)) == ".{0,60}(err(or)?).{0,60}"


def test_python_side_of_a_bracketed_literal_warns_nothing() -> None:
    """`[` becomes `[[]` for grep; Python's re warns "Possible nested set" (FutureWarning) on that, and a warning on
    stderr scribbles over the running TUI (--demo plays grep with Python, the results highlight with it)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = LogQuery("[/x] [bold]")
        rx = logsearch.ere_bytes(logsearch.grep_pattern(q), icase=True)
        assert not isinstance(rx, str) and rx.search(b"a [/X] [BOLD] b")
        assert logsearch.match_spans("a [x] b", LogQuery("[[]x[]]", regex=True)) == [(2, 5)]


def test_command_quotes_every_path_and_the_pattern() -> None:
    paths = ["/w/a b.out", "/w/it's $(rm -rf ~).out", "/w/*.out"]
    cmd = logsearch.search_command(paths, LogQuery("'; rm -rf / #"))
    assert "for f in '/w/a b.out' '/w/it'\"'\"'s $(rm -rf ~).out' '/w/*.out'; do" in cmd
    assert "p='.{0,60}('\"'\"'; rm -rf / #).{0,60}'" in cmd
    assert "grep -a -n -b -o -i -E -m \"$left\" -e \"$p\"" in cmd and "cut -b 1-600" in cmd
    assert "tail -c +$((b + 1)) -- \"$f\" | head -c \"$n\"" in cmd
    assert "-i" not in logsearch.search_command(["/w/x"], LogQuery("x", match_case=True)).split("grep -a")[1][:12]
    with pytest.raises(ValueError):
        logsearch.search_command(["relative.out"], LogQuery("x"))
    sec = logsearch.search_section(paths, LogQuery("x"), max_matches=40)
    assert (sec.name, sec.timeout, sec.max_stdout) == ("log.search", 30.0, logsearch.MAX_STDOUT)
    assert "left=40; budget=1048576" in sec.cmd and "-gt 262144" in sec.cmd
    assert sec.meta["paths"] == paths and sec.meta["max_matches"] == 40 and sec.meta["icase"] is True


# ============================================================================================ parse
def _plan(n: int = 3) -> SearchPlan:
    return SearchPlan("k2", tuple(SearchFile(f"/w/{i}.out", JobKey("k2", str(100 + i)), f"job{i}", "stdout")
                                  for i in range(1, n + 1)), n)


def _res(stdout: str, outcome: SectionOutcome = SectionOutcome.OK, stderr: str = "") -> SectionResult:
    return SectionResult("log.search", outcome, 0 if outcome is SectionOutcome.OK else 124, stdout, stderr, 40)


def test_parse_hits_offsets_and_file_outcomes() -> None:
    out = ("@@file 1 300000 37856 262144\n"
           "12:4096:[step 3] loss=nan at lr 3e-4\n"
           "40:9000:\x1b[31mERROR\x1b[0m loss=NaN\r 10%|# loss=nan again\n"
           "@@err 2 missing\n"
           "@@file 3 900 0 900\n"
           "@@end 197 0\n")
    r = logsearch.parse("k2", LogQuery("loss=nan"), _plan(), _res(out))
    assert r.error is None and not r.timed_out and len(r.hits) == 2
    a, b = r.hits
    assert (a.start, a.line, a.offset, a.size) == (37856, 12, 37856 + 4096, 300000)
    assert a.needle == "[step 3] loss=nan at lr 3e-4" and a.file.key.id == "101"
    assert b.needle == "\x1b[31mERROR\x1b[0m loss=NaN"                    # the \r segment with the first match
    assert [f.status for f in r.files] == ["searched", "missing", "searched"]
    assert r.files[0].window_cut and not r.files[0].budget_cut and r.files[0].hits == 2
    assert r.bytes_read == 262144 + 900 and not r.match_capped


def test_parse_budget_skip_match_cap_and_timeout() -> None:
    out = ("@@file 1 2000000 1737856 262144\n1:0:x\n2:5:x\n@@skip 2 matches\n@@skip 3 budget\n@@end 0 0\n")
    r = logsearch.parse("k2", LogQuery("x"), _plan(), _res(out), max_matches=2)
    assert [f.status for f in r.files] == ["searched", "skipped (matches)", "skipped (budget)"]
    assert r.match_capped and len(r.hits) == 2
    cut = logsearch.FileOutcome(_plan().files[0], "searched", 500_000, 400_000, 100_000)
    assert cut.budget_cut and not cut.window_cut
    partial = "@@file 1 50 0 50\n3:0:x one\n@@file 2 70 0 70\n1:0:x tw"
    t = logsearch.parse("k2", LogQuery("x"), _plan(), _res(partial, SectionOutcome.TIMEOUT))
    assert t.timed_out and "timed out after 30 s" in (t.error or "")
    assert [f.status for f in t.files] == ["searched", "searched", "not reached"] and len(t.hits) == 2


def test_parse_bad_pattern_and_missing_tool() -> None:
    r = logsearch.parse("k2", LogQuery("(", regex=True), _plan(), _res("@@bad grep: Unmatched ( or \\(\n"))
    assert r.error == "grep: Unmatched ( or \\(" and r.hits == () and {f.status for f in r.files} == {"not reached"}
    m = logsearch.parse("k2", LogQuery("x"), _plan(),
                        SectionResult("log.search", SectionOutcome.MISSING, 127, "", "missing:stat", 0))
    assert m.error == "cannot search here: stat is missing"


def test_display_excerpt_strips_ansi_and_marks_cuts() -> None:
    q = LogQuery("loss=nan")
    hit = logsearch.LogHit("k2", _plan().files[0], 0, 1, 0, "x" * 60 + "\x1b[1mloss=NaN\x1b[0m\tdone",
                           "x" * 60 + "\x1b[1mloss=NaN\x1b[0m\tdone")
    text, spans, before, after = logsearch.display_excerpt(hit, q)
    assert text == "x" * 60 + "loss=NaN done" and spans == [(60, 68)] and before and not after
    assert logsearch.match_spans("a1 b22 c333", LogQuery("[[:digit:]]+", regex=True)) == [(1, 2), (4, 6), (8, 11)]
    assert logsearch.match_spans("Err err", LogQuery("err", match_case=True)) == [(4, 7)]
    assert logsearch.match_spans("x", LogQuery("[", regex=True)) == []      # Python cannot read it: no highlight


def test_needle_drops_cut_characters_and_other_cr_segments() -> None:
    q = LogQuery("損失")
    assert logsearch.needle_of("\ufffd損失 0.5\r 20%|##", q) == "損失 0.5"
    assert logsearch.needle_of(" 10%|#\rstep 4 損失 nan\ufffd", q) == "step 4 損失 nan"


# ============================================================================================ the service
class Probe:
    def __init__(self, t: FakeTransport) -> None:
        self.t, self.scripts = t, []

    async def __call__(self, host: str, script: ProbeScript) -> ProbeResult:
        self.scripts.append(script)
        return await self.t.probe(script, timeout=10)


async def test_log_service_search_runs_one_log_probe() -> None:
    t = FakeTransport("k2")
    t.set_result("log.search", "@@file 1 10 0 10\n1:0:hello\n@@end 199 1048566\n")
    probe = Probe(t)
    svc = LogService(ManualClock(T0), transport_for=lambda h: t, stream_limit_for=lambda h: 1, probe=probe,
                     caps_for=lambda h: None)
    r = await svc.search("k2", _plan(1), LogQuery("hello"))
    assert [h.needle for h in r.hits] == ["hello"] and r.error is None
    assert len(probe.scripts) == 1 and probe.scripts[0].purpose == "log"
    assert [s.name for s in probe.scripts[0].sections] == ["log.search"]
    assert (await svc.search("k2", SearchPlan("k2"), LogQuery("x"))).hits == ()   # nothing in scope: no probe
    assert (await svc.search("k2", _plan(1), LogQuery(""))).error == "type a pattern"
    assert len(probe.scripts) == 1

    async def failing(host: str, script: ProbeScript) -> ProbeResult:
        return failed_probe(script, HostError(ErrorKind.AUTH, "needs auth"))

    svc2 = LogService(ManualClock(T0), transport_for=lambda h: t, stream_limit_for=lambda h: 1, probe=failing,
                      caps_for=lambda h: None)
    assert (await svc2.search("k2", _plan(1), LogQuery("x"))).error == "needs auth"


def test_tail_spec_from_an_absolute_offset() -> None:
    spec = tail_spec("/w/a b.out", 65536, 37856)
    assert spec.cmd.startswith("tail -c +37857 -F -- '/w/a b.out' & p=$!") and spec.meta["start"] == 37856
    assert tail_spec("/w/x", 65536).cmd.startswith("tail -c 65536 -F -- /w/x") and "start" not in tail_spec(
        "/w/x", 65536).meta
