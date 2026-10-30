"""[WP4] `runfleet agent log --files` / `--file PATH` in agent.view and agent.source (DESIGN.md §8.7, §4.7.2): the
workdir rule (lexical, POSIX, never outside it), the listing (relative paths, own stdout / stderr marked, newest
first, next command), find errors as fixed texts or typed problems, `--file` only for a listed file, the job card's
"stdout is quiet" line, budgets; agent.source.find_logs without network for a login-waiting session host."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from runfleet.agent import source, view
from runfleet.agent.data import EXIT_NO_DATA, EXIT_OK, EXIT_PARTIAL, FleetView, Problem, ProblemCode
from runfleet.core.logs import FIND_SLACK, FIND_UNREADABLE, FoundLog, LogFiles, LogRead, find_window
from runfleet.models import ConnState, ErrorKind, HostError, JobState, LogRef
from tests.agent.test_view import (ENVELOPE, H, T0, _posix_quoting, _printed_commands_round_trip,  # noqa: F401
                                   fleet, keys, mk_host, mk_job, size)

WD = "/shared/lab/alice/vitfm"
OUT = f"{WD}/logs/vitfm_7.out"
TRAIN = f"{WD}/runs/in1k/r0412/s42/train.log"


def launcher(**kw: Any) -> Any:
    """A running launcher job: stdout = 539 bytes written at its start, 13 h ago."""
    d: dict[str, Any] = dict(workdir=WD, start_time=T0 - 13 * H, elapsed=13 * H,
                             logs=(LogRef("stdout", OUT, merged=True, size=539, mtime=T0 - 13 * H),
                                   LogRef("stderr", OUT, merged=True, size=539, mtime=T0 - 13 * H)))
    d.update(kw)
    return mk_job("k2", "7", **d)


def found(path: str, age_s: float, size: int = 1000) -> FoundLog:
    return FoundLog(path=path, size=size, mtime=T0 - timedelta(seconds=age_s))


def listing(*files: FoundLog, **kw: Any) -> LogFiles:
    return LogFiles(workdir=WD, files=files, fetched_at=T0, **kw)


LISTED = listing(found(OUT, 13 * 3600, 539), found(TRAIN, 2, 3_100_000), found(f"{WD}/runs/in1k/r0412/s42/eval.err", 600))


def lfv(*jobs: Any, **kw: Any) -> FleetView:
    return fleet(mk_host("k2", jobs or [launcher()]), **kw)


def _read(path: str = TRAIN) -> LogRead:
    return LogRead(path=path, lines=("epoch 1", "epoch 2", "Traceback (most recent call last):"), live="epoch 3 40%",
                   size=3_100_000, mtime=T0, window_bytes=32768, started_mid_file=True, fetched_at=T0)


# ---------------------------------------------------------------- the workdir rule
@pytest.mark.parametrize(("path", "expected"), [
    ("runs/x/train.log", f"{WD}/runs/x/train.log"),
    (f"{WD}/runs/x/train.log", f"{WD}/runs/x/train.log"),
    ("./runs//x/../x/train.log", f"{WD}/runs/x/train.log"),
    (f"/{WD}/a.log", f"{WD}/a.log"),                                  # a leading "//" is "/"
    ("../vitfm2/a.log", None),                                       # a sibling sharing the prefix
    ("/shared/lab/alice/vitfm2/a.log", None),
    ("runs/../../../../etc/passwd", None),
    ("/etc/passwd", None),
    (".", None), ("", None), (WD, None), (WD + "/", None),          # the workdir itself is no file
    ("a\nb.log", None), ("a\x00.log", None), ("a\rb.log", None),
])
def test_under_workdir_is_lexical_and_strict(path: str, expected: str | None) -> None:
    assert view.under_workdir(WD, path) == expected
    assert view.under_workdir(WD + "/", path) == expected                 # a trailing slash changes nothing


def test_under_workdir_needs_an_absolute_workdir_other_than_root() -> None:
    assert view.under_workdir("relative/dir", "a.log") is None
    assert view.under_workdir("/", "etc/passwd") is None and view.under_workdir("//", "/etc/passwd") is None
    assert view.workdir_rel(WD, TRAIN) == "runs/in1k/r0412/s42/train.log"
    assert view.workdir_rel(WD, "/elsewhere/x.log") == "/elsewhere/x.log"


def test_find_window_is_the_start_and_matches_the_probe_slack() -> None:
    assert view.FIND_SLACK == FIND_SLACK and view.find_window is find_window          # one rule with the TUI
    assert view.find_window(launcher()) == (T0 - 13 * H, None)
    assert view.find_window(launcher(start_time=None)) == (None, None)
    assert view.find_window(launcher(start_time=T0 + H, est_start=True)) == (None, None)
    # round 2: a running job's end_time is its time limit, never an `until`
    assert view.find_window(launcher(end_time=T0 + 3 * H)) == (T0 - 13 * H, None)


# ---------------------------------------------------------------- --files
def test_files_lists_relative_paths_newest_first_with_own_streams_marked() -> None:
    doc = view.log_files(lfv(), ("k2", "7"), LISTED)
    lines = doc.text.splitlines()
    assert lines[1].startswith(f"# k2:7 log files under {WD} written since 01:01Z (job start - 60 s), newest first")
    rows = [ln.split() for ln in lines[2:5]]
    assert [r[0] for r in rows] == ["runs/in1k/r0412/s42/train.log", "runs/in1k/r0412/s42/eval.err",
                                    "logs/vitfm_7.out"]
    assert rows[2][-1] == "stdout" and rows[0][-1] == "ago"
    assert lines[-1] == "next: runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log"
    assert doc.exit_code == EXIT_OK
    d = doc.data
    assert keys(d) == sorted([*ENVELOPE, "file", "files", "find", "log"])
    assert d["log"] is None and d["file"] is None and not d["errors"]
    assert keys(d["files"][0]) == ["age_s", "mtime", "other_job", "path", "size", "stream"]
    assert [f["stream"] for f in d["files"]] == [None, None, "stdout"]
    assert d["files"][0] == {"path": "runs/in1k/r0412/s42/train.log", "size": 3_100_000, "mtime": "2026-10-18T14:02:29Z",
                             "age_s": 2, "stream": None, "other_job": None}
    assert keys(d["find"]) == ["fetched_at", "incomplete", "job", "key", "next", "path_too_long", "redacted", "since",
                               "truncated", "until", "workdir"]
    assert d["find"]["since"] == "2026-10-18T01:01:31Z" and d["find"]["workdir"] == WD
    assert d["find"]["until"] is None and d["find"]["incomplete"] is None and d["find"]["path_too_long"] == 0
    assert d["find"]["next"] == "runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log --json"


def test_files_marks_a_separate_stderr_and_offers_no_next_without_another_file() -> None:
    err = f"{WD}/logs/vitfm_7.err"
    job = launcher(logs=(LogRef("stdout", OUT, size=539), LogRef("stderr", err, size=10)))
    doc = view.log_files(lfv(job), ("k2", "7"), listing(found(err, 5), found(OUT, 50)))
    assert [f["stream"] for f in doc.data["files"]] == ["stderr", "stdout"]
    assert doc.data["find"]["next"] is None and "next:" not in doc.text


def test_other_jobs_logs_are_marked_and_never_suggested() -> None:
    neighbour = f"{WD}/logs/vitfm_8.out"
    eval_err = f"{WD}/logs/vitfm_8.err"
    other = mk_job("k2", "8", workdir=WD, logs=(LogRef("stdout", neighbour, size=10**6, mtime=T0),
                                                LogRef("stderr", eval_err, size=10, mtime=T0)))
    task = mk_job("k2", "p4242", JobState.EXITED, workdir=WD,
                  logs=(LogRef("stdout", f"{WD}/./runs//prep.log", size=10, mtime=T0),))
    fv = fleet(replace(mk_host("k2", [launcher(), other]), ended_tasks=(task,)))
    files = listing(found(neighbour, 1), found(f"{WD}/runs/prep.log", 2), found(TRAIN, 3), found(OUT, 13 * 3600, 539))
    doc = view.log_files(fv, ("k2", "7"), files)
    d = doc.data
    assert [(f["path"], f["stream"], f["other_job"]) for f in d["files"]] == [
        ("logs/vitfm_8.out", None, "job 8 stdout"), ("runs/prep.log", None, "job p4242 stdout"),
        ("runs/in1k/r0412/s42/train.log", None, None), ("logs/vitfm_7.out", "stdout", None)]
    rows = {ln.split()[0]: ln for ln in doc.text.splitlines() if ln.startswith("  ")}
    assert rows["logs/vitfm_8.out"].endswith("job 8 stdout") and rows["runs/prep.log"].endswith("job p4242 stdout")
    assert rows["logs/vitfm_7.out"].endswith("stdout") and rows["runs/in1k/r0412/s42/train.log"].endswith("ago")
    assert d["find"]["next"] == "runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log --json"  # not the newest
    assert doc.text.splitlines()[-1] == "next: runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log"


def test_no_next_when_every_file_is_this_or_another_jobs_log() -> None:
    other = mk_job("k2", "8", workdir=WD, logs=(LogRef("stdout", f"{WD}/logs/vitfm_8.out", size=10, mtime=T0),))
    elsewhere = mk_job("vega", "8", workdir=WD, logs=(LogRef("stdout", f"{WD}/logs/vega_8.out", size=10, mtime=T0),))
    fv = fleet(mk_host("k2", [launcher(), other]), mk_host("vega", [elsewhere]))
    doc = view.log_files(fv, ("k2", "7"), listing(found(f"{WD}/logs/vitfm_8.out", 1), found(OUT, 50)))
    assert doc.data["find"]["next"] is None and not doc.text.splitlines()[-1].startswith("next:")
    same_path = view.log_files(fv, ("k2", "7"), listing(found(f"{WD}/logs/vega_8.out", 1)))     # another host's job
    assert same_path.data["files"][0]["other_job"] is None and same_path.data["find"]["next"] is not None


def test_an_ended_job_lists_files_up_to_its_end() -> None:
    ended = launcher(state=JobState.FAILED, raw_state="FAILED", end_time=T0 - 5 * H, elapsed=8 * H)
    assert view.find_window(ended) == (T0 - 13 * H, T0 - 5 * H)
    assert view.find_window(replace(ended, end_time=None)) == (T0 - 13 * H, None)
    doc = view.log_files(lfv(ended), ("k2", "7"), listing(found(TRAIN, 5 * 3600)))
    assert doc.data["find"]["until"] == "2026-10-18T09:07:31Z"                   # end + FIND_END_SLACK
    assert " until 09:07Z (job end + 5 min), newest first" in doc.text.splitlines()[1]


def test_find_logs_passes_the_jobs_window_to_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.config import FleetConfig, HostConfig
    from runfleet.core.clock import ManualClock
    from runfleet.core.logs import LogService

    seen: list[tuple[Any, Any]] = []

    async def fake_find(self: Any, host: str, workdir: str, *, since: Any, until: Any = None) -> LogFiles:
        seen.append((since, until))
        return listing()

    monkeypatch.setattr(LogService, "find", fake_find)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    ended = launcher(state=JobState.COMPLETED, end_time=T0 - H)
    running = launcher(end_time=T0 + 3 * H)                       # a running job's end_time: its time limit
    old = launcher(state=JobState.COMPLETED, start_time=None, end_time=T0 - 72 * H)    # ended 3 days ago, no start
    for job in (running, ended, old):
        got = asyncio.run(source.find_logs(cfg, mk_host("k2", [job]), job, clock=ManualClock(T0), timeout=5,
                                           transports=lambda h: None))
        assert got == listing()
    assert seen == [(T0 - 13 * H, None), (T0 - 13 * H, T0 - H), (T0 - 96 * H, T0 - 72 * H)]


def test_an_ended_job_without_a_start_searches_the_day_before_its_end() -> None:
    """Round 2: since None was the core's "last 24 h" from now, inverted against `until` for a job that ended days
    ago; agent text and JSON now show the window searched (the TUI's, core.logs.find_window)."""
    old = launcher(state=JobState.COMPLETED, start_time=None, end_time=T0 - 72 * H, elapsed=8 * H)   # ran: no start
    doc = view.log_files(lfv(old), ("k2", "7"), listing(found(TRAIN, 74 * 3600)))
    assert doc.data["find"]["since"] == "2026-10-14T14:01:31Z"                  # end - 24 h - FIND_SLACK
    assert doc.data["find"]["until"] == "2026-10-15T14:07:31Z"                  # end + FIND_END_SLACK
    assert ("written since 10-14 14:01Z (job end - 24 h, start unknown) until 10-15 14:07Z (job end + 5 min)"
            in doc.text.splitlines()[1])
    assert "in the last 24 h" not in doc.text


def test_files_drops_anything_outside_the_workdir_and_notes_truncation() -> None:
    odd = listing(found("/etc/shadow.log", 1), found(f"{WD}/../x.log", 1), found("relative.log", 1),
                  found(TRAIN, 3), truncated=True)
    doc = view.log_files(lfv(), ("k2", "7"), odd)
    assert [f["path"] for f in doc.data["files"]] == ["runs/in1k/r0412/s42/train.log"]
    assert "shadow" not in doc.text and "only the 1 newest files are listed" in doc.text
    assert doc.data["find"]["truncated"] is True


def test_files_none_found_is_an_answer() -> None:
    doc = view.log_files(lfv(), ("k2", "7"), listing())
    assert doc.exit_code == EXIT_OK and doc.data["files"] == [] and "no log files" in doc.text
    since_unknown = view.log_files(lfv(launcher(start_time=None)), ("k2", "7"), listing())
    assert "in the last 24 h" in since_unknown.text and since_unknown.data["find"]["since"] == "2026-10-17T14:02:31Z"


@pytest.mark.parametrize(("file_error", "text"), [
    ("unsupported: needs GNU find (-printf, -newermt)", "log file search: not supported on this host (needs GNU find)"),
    ("No such directory", "log file search: the workdir does not exist"),
    ("Permission denied", "log file search: the workdir is not readable"),
    ("find: '/x/login7.hidden-cluster.example': weird", "log file search: failed"),
])
def test_find_file_errors_are_fixed_texts(file_error: str, text: str) -> None:
    doc = view.log_files(lfv(), ("k2", "7"), listing(file_error=file_error))
    assert doc.exit_code == EXIT_NO_DATA and doc.data["errors"][0]["code"] == "error"
    assert doc.data["errors"][0]["message"] == text and doc.data["errors"][0]["job"] == "k2:7"
    assert "hidden-cluster" not in doc.text + view.dumps(doc.data)


def test_an_incomplete_search_says_so_and_never_no_files() -> None:
    doc = view.log_files(lfv(), ("k2", "7"), listing(found(TRAIN, 3), incomplete="timed out"))
    assert doc.exit_code == EXIT_OK and doc.data["find"]["incomplete"] == "timed out"
    assert doc.data["find"]["truncated"] is False
    assert "# note: search stopped early (timed out): 1 found so far" in doc.text and "newest files" not in doc.text
    for why in ("timed out", "output cut", "sort failed", "find failed (rc 1)"):
        none = view.log_files(lfv(), ("k2", "7"), listing(incomplete=why))
        assert none.exit_code == EXIT_OK and none.data["find"]["incomplete"] == why and none.data["files"] == []
        assert f"# note: search stopped early ({why}): 0 found so far" in none.text
        assert "no log files" not in none.text and "only the 0 newest" not in none.text
    complete = view.log_files(lfv(), ("k2", "7"), listing(found(TRAIN, 3)))
    assert complete.data["find"]["incomplete"] is None and "stopped early" not in complete.text


def test_unreadable_directories_are_not_a_search_that_stopped_early() -> None:
    """Round 2: the walk finished (only some directories were unreadable): never "stopped early" or "so far"."""
    for files in ((), (found(TRAIN, 3),)):
        doc = view.log_files(lfv(), ("k2", "7"), listing(*files, incomplete=FIND_UNREADABLE))
        assert doc.exit_code == EXIT_OK and doc.data["find"]["incomplete"] == FIND_UNREADABLE
        assert "# note: some directories were unreadable: their files are not listed" in doc.text
        assert "stopped early" not in doc.text and "so far" not in doc.text
    empty = view.log_files(lfv(), ("k2", "7"), listing(incomplete=FIND_UNREADABLE))
    assert "in that time (in the readable directories)" in empty.text
    capped = listing(*(found(f"{WD}/{i:02d}.log", i) for i in range(30)), truncated=True, incomplete=FIND_UNREADABLE)
    both = view.log_files(lfv(), ("k2", "7"), capped).text
    assert "their files are not listed" in both and "# note: only the 30 newest files are listed" in both
    job = launcher()
    miss = view.file_choice(job, "runs/late.log", listing(found(TRAIN, 3), incomplete=FIND_UNREADABLE))[1]
    assert miss == Problem(ProblemCode.NOT_LISTED, reason="unreadable")
    assert view.message(miss) == "not among the log files found: some directories were unreadable (it may be in one)"


def test_a_capped_and_stopped_search_says_both() -> None:
    both = listing(*(found(f"{WD}/{i:02d}.log", i) for i in range(30)), truncated=True, incomplete="timed out")
    doc = view.log_files(lfv(), ("k2", "7"), both)
    assert "# note: search stopped early (timed out): 30 found so far" in doc.text
    assert "# note: only the 30 newest files are listed" in doc.text
    assert doc.data["find"]["truncated"] is True and doc.data["find"]["incomplete"] == "timed out"


def test_a_find_failure_keeps_its_fixed_text() -> None:
    doc = view.log_files(lfv(), ("k2", "7"), listing(file_error="find failed (rc 1)"))
    assert doc.exit_code == EXIT_NO_DATA and doc.data["errors"][0]["message"] == "log file search: find failed (rc 1)"


@pytest.mark.parametrize(("kind", "code"), [(ErrorKind.AUTH, "auth_required"), (ErrorKind.TIMEOUT, "timeout"),
                                            (ErrorKind.REFUSED, "unreachable"), (ErrorKind.INTERNAL, "error")])
def test_find_transport_errors_are_typed(kind: ErrorKind, code: str) -> None:
    err = HostError(kind, "Permission denied (publickey) on login7.hidden-cluster.example")
    doc = view.log_files(lfv(), ("k2", "7"), listing(error=err))
    assert doc.data["errors"][0]["code"] == code and doc.exit_code == EXIT_NO_DATA
    assert "hidden-cluster" not in doc.text + view.dumps(doc.data)


def test_files_refusals_before_any_probe() -> None:
    other = mk_job("k2", "9", user="bob", is_mine=False, workdir=WD)
    pending = mk_job("k2", "8", JobState.PENDING, workdir=WD)
    nowd = launcher(workdir=None)
    fv = fleet(mk_host("k2", [other, pending, nowd]))
    assert view.log_files(fv, ("k2", "9"), LISTED).data["errors"][0]["code"] == "not_allowed"
    assert "train.log" not in view.log_files(fv, ("k2", "9"), LISTED).text
    assert view.log_files(fv, ("k2", "8"), None).data["errors"][0]["code"] == "no_log_yet"
    gone = view.log_files(fv, ("k2", "7"), None)
    assert gone.data["errors"][0]["message"].startswith("the job's working directory is unknown")
    assert view.log_files(fv, ("k2", "404"), None).data["errors"][0]["code"] == "not_found"
    for doc in (gone, view.log_files(fv, ("k2", "9"), None)):
        assert keys(doc.data) == sorted([*ENVELOPE, "file", "files", "find", "log"])


def test_no_find_without_ssh_or_by_mistake() -> None:
    assert view.log_files(lfv(), ("k2", "7"), None).data["errors"][0]["code"] == "internal"
    no_ssh = lfv(problems=(Problem(ProblemCode.NO_SSH, host="k2"),))
    codes = [e["code"] for e in view.log_files(no_ssh, ("k2", "7"), None).data["errors"]]
    assert codes == ["no_ssh", "no_ssh"]


def test_files_budget_with_long_unicode_paths() -> None:
    many = listing(*(found(f"{WD}/{'日' * 400}/{i:02d}.log", i) for i in range(30)))
    doc = view.log_files(lfv(), ("k2", "7"), many)
    t, j = size(doc)
    assert t <= view.BUDGETS["log"][0] and j <= view.BUDGETS["log"][1]
    assert 0 < len(doc.data["files"]) < 30 and doc.data["truncated"]["total"] == 30
    assert all(f["path"] == f"{'日' * 400}/{i:02d}.log" for i, f in enumerate(doc.data["files"]))    # never cut
    assert "files not shown (output budget)" in doc.text


def test_long_listed_paths_stay_usable_selectors() -> None:
    from runfleet.agent import commands
    from runfleet.config import FleetConfig, HostConfig

    assert view.FOUND_PATH_CAP == commands.FILE_MAX == 4096                     # PATH_MAX
    deep = f"{WD}/{'/'.join(c * 250 for c in 'defghijk')}/train.log"           # ~2000 characters, 8 levels
    rel = deep[len(WD) + 1:]
    assert 2000 <= len(rel) <= 2100
    long_listing = listing(found(deep, 1), found(TRAIN, 5))
    doc = view.log_files(lfv(), ("k2", "7"), long_listing)
    assert doc.data["files"][0]["path"] == rel and f"  {rel}  " in doc.text            # listed whole, never cut
    assert doc.data["find"]["next"] == f"runfleet agent log k2:7 --file {rel} --json"
    assert doc.text.splitlines()[-1] == f"next: runfleet agent log k2:7 --file {rel}"
    t, j = size(doc)
    assert t <= view.BUDGETS["log"][0] and j <= view.BUDGETS["log"][1]
    req = commands.build_request("log", {"id": "k2:7", "file": rel}, FleetConfig(hosts=(HostConfig("k2", "k2"),)))
    assert commands._Params(req.params).text("file") == rel
    hit, why = view.file_choice(launcher(), rel, long_listing)
    assert why is None and hit is not None and hit.path == deep
    read = view.log(lfv(), ("k2", "7"), _read(deep), file=rel, files=long_listing)
    assert read.exit_code == EXIT_OK and read.data["file"] == rel and read.data["log"]["stream"] == "file"


def test_a_path_longer_than_path_max_is_dropped_with_a_note_not_cut() -> None:
    huge = f"{WD}/{'x' * 4100}.log"
    doc = view.log_files(lfv(), ("k2", "7"), listing(found(huge, 1), found(TRAIN, 5)))
    assert [f["path"] for f in doc.data["files"]] == ["runs/in1k/r0412/s42/train.log"]
    assert doc.data["find"]["path_too_long"] == 1 and "x" * 100 not in doc.text + view.dumps(doc.data)
    assert "# note: 1 files with a path over 4096 characters not listed" in doc.text
    assert doc.data["find"]["next"] == "runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log --json"


def test_a_path_that_would_print_changed_gets_no_command() -> None:
    secret = f"{WD}/token=hf_{'Q' * 30}.log"
    doc = view.log_files(lfv(), ("k2", "7"), listing(found(secret, 1)))
    assert "hf_QQQQ" not in doc.text + view.dumps(doc.data)
    assert doc.data["find"]["next"] is None and "next:" not in doc.text and doc.data["find"]["redacted"] == 1
    assert "# note: 1 paths are shown redacted (secret-like text): --file cannot name them" in doc.text
    commit = f"{WD}/runs/{'0123456789abcdef' * 2}01234567/train.log"            # a 40-hex run dir prints redacted
    both = view.log_files(lfv(), ("k2", "7"), listing(found(commit, 1), found(TRAIN, 9)))
    assert both.data["find"]["next"] == "runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log --json"


def test_a_next_command_too_long_to_print_is_advice_never_cut() -> None:
    deep = f"{WD}/{'/'.join('日' * 250 for _ in range(16))}/x.log"                 # 4000+ characters, 12 KB
    doc = view.log_files(lfv(), ("k2", "7"), listing(found(deep, 1)))
    t, j = size(doc)
    assert t <= view.BUDGETS["log"][0] and j <= view.BUDGETS["log"][1]
    assert doc.data["find"]["next"] is None and doc.data["files"][0]["path"] == deep[len(WD) + 1:]
    assert doc.text.splitlines()[-1].startswith("next: no command printed: too long to print here")


# ---------------------------------------------------------------- --file PATH
def test_file_reads_a_listed_file_like_stdout() -> None:
    doc = view.log(lfv(), ("k2", "7"), _read(), file="runs/in1k/r0412/s42/train.log", files=LISTED, lines=2)
    assert doc.exit_code == EXIT_OK
    lg = doc.data["log"]
    assert lg["stream"] == "file" and lg["path"] == TRAIN and lg["merged"] is False and lg["lines"][-1].startswith("Trace")
    assert doc.data["file"] == "runs/in1k/r0412/s42/train.log" and doc.data["files"] is None
    assert doc.text.splitlines()[1].startswith(f"# k2:7 file {TRAIN} · 3.0M")
    assert "| [live] epoch 3 40%" in doc.text
    grep = view.log(lfv(), ("k2", "7"), _read(), file=TRAIN, files=LISTED, grep="traceback")
    assert grep.data["log"]["grep"]["total"] == 1 and grep.data["file"] == "runs/in1k/r0412/s42/train.log"


def test_file_choice_refuses_what_the_find_did_not_list() -> None:
    job = launcher()
    hit, why = view.file_choice(job, f"{WD}/runs/in1k/r0412/s42/./train.log", LISTED)
    assert hit is not None and hit.path == TRAIN and why is None
    assert view.found_ref(hit) == LogRef("file", TRAIN, source="found", size=3_100_000, mtime=T0 - timedelta(seconds=2))
    assert view.file_choice(job, "/etc/passwd", LISTED)[1] == Problem(ProblemCode.NOT_LISTED, reason="outside")
    assert view.file_choice(job, "runs/metrics.jsonl", LISTED)[1] == Problem(ProblemCode.NOT_LISTED)
    stopped = replace(LISTED, incomplete="timed out")
    assert view.file_choice(job, "runs/late.log", stopped)[1] == Problem(ProblemCode.NOT_LISTED, reason="incomplete")
    assert view.file_choice(job, TRAIN, stopped)[0] is not None
    doc = view.log(lfv(), ("k2", "7"), None, file="runs/late.log", files=stopped)
    assert doc.data["errors"][0]["message"] == "not among the log files found: the search stopped early (it may exist)"
    assert view.file_choice(job, "x.log", None)[1] == Problem(ProblemCode.INTERNAL)
    assert view.file_choice(launcher(workdir=None), "x.log", LISTED)[1] is not None
    denied = view.file_choice(job, "x.log", listing(error=HostError(ErrorKind.AUTH, "x")))[1]
    assert denied is not None and denied.code is ProblemCode.AUTH_REQUIRED


def test_file_never_reads_another_users_log_in_a_shared_workdir() -> None:
    bob_out = f"{WD}/logs/bob_9.out"
    bob = mk_job("k2", "9", user="bob", is_mine=False, workdir=WD, logs=(LogRef("stdout", bob_out, size=10, mtime=T0),))
    hv = mk_host("k2", [launcher(), bob])
    files = listing(found(bob_out, 1), found(TRAIN, 5))
    assert view.file_precheck(launcher(), "logs/./bob_9.out", hv) == Problem(ProblemCode.NOT_ALLOWED)
    assert view.file_choice(launcher(), bob_out, files, hv)[1] == Problem(ProblemCode.NOT_ALLOWED)
    doc = view.log(fleet(hv), ("k2", "7"), None, file="logs/bob_9.out", files=None)
    assert doc.data["errors"][0]["code"] == "not_allowed" and doc.data["log"] is None
    listed = view.log_files(fleet(hv), ("k2", "7"), files)                       # listed and marked, never suggested
    assert listed.data["files"][0]["other_job"] == "job 9 stdout"
    assert listed.data["find"]["next"] == "runfleet agent log k2:7 --file runs/in1k/r0412/s42/train.log --json"
    assert view.file_precheck(launcher(), "runs/in1k/r0412/s42/train.log", hv) is None


@pytest.mark.parametrize(("path", "reason"), [
    (".", "workdir"), (WD, "workdir"), (WD + "/", "workdir"), ("runs/..", "workdir"), (f"/{WD}//.", "workdir"),
    ("runs/", "directory"), ("runs/in1k/.", "directory"), ("runs/in1k/r0412/..", "directory"),
    ("..", "outside"), ("/etc/passwd", "outside"), ("../vitfm2", "outside"),
])
def test_file_refusals_before_any_probe_say_why(path: str, reason: str) -> None:
    assert view.file_refusal(WD, path) == Problem(ProblemCode.NOT_LISTED, reason=reason)
    doc = view.log(lfv(), ("k2", "7"), None, file=path, files=None)
    err = doc.data["errors"][0]
    assert err["code"] == "not_listed" and doc.exit_code == EXIT_NO_DATA
    assert err["message"] == {"workdir": "the path is the job's workdir, not a file",
                              "directory": "the path names a directory, not a file",
                              "outside": "the path is outside the job's workdir"}[reason]
    assert view.file_refusal(WD, "runs/train.log") is None and view.file_refusal(WD, f"{WD}/a.log") is None


def test_file_problems_in_the_document() -> None:
    outside = view.log(lfv(), ("k2", "7"), None, file="../../../etc/passwd", files=None)
    err = outside.data["errors"][0]
    assert err["code"] == "not_listed" and err["message"] == "the path is outside the job's workdir"
    assert err["hint"] == "run runfleet agent log k2:7 --files and read a file it lists" and outside.exit_code == 1
    missing = view.log(lfv(), ("k2", "7"), None, file="runs/other.log", files=LISTED)
    assert missing.data["errors"][0]["message"] == "not a log file found under the job's workdir"
    assert missing.data["log"] is None and missing.data["file"] is None
    partial = view.log(lfv(), ("k2", "7"), replace(_read(), partial=True), file=TRAIN, files=LISTED)
    assert partial.exit_code == EXIT_PARTIAL


# ---------------------------------------------------------------- the job card's quiet-stdout line
def test_quiet_stdout_rule() -> None:
    assert view.quiet_stdout(launcher(), T0, 1800)                                       # 539 B, 13 h old
    just_written = launcher(start_time=T0 - 5 * H, elapsed=5 * H,
                            logs=(LogRef("stdout", OUT, size=202, mtime=T0),))
    assert not view.quiet_stdout(just_written, T0, 1800)            # 202 B written 0 s ago: not quiet (round 1 bug)
    tiny_idle = launcher(start_time=T0 - timedelta(minutes=20), elapsed=timedelta(minutes=20),
                         logs=(LogRef("stdout", OUT, size=4096, mtime=T0 - timedelta(minutes=10)),))
    assert view.quiet_stdout(tiny_idle, T0, 1800)                   # ≤ 4 KiB and unchanged 10 min
    assert not view.quiet_stdout(replace(tiny_idle, logs=(LogRef("stdout", OUT, size=4097,
                                                                  mtime=T0 - timedelta(minutes=10)),)), T0, 1800)
    assert not view.quiet_stdout(replace(tiny_idle, logs=(LogRef("stdout", OUT, size=10),)), T0, 1800)  # no mtime
    busy = launcher(logs=(LogRef("stdout", OUT, size=3_000_000, mtime=T0 - timedelta(seconds=60)),))
    assert not view.quiet_stdout(busy, T0, 1800)
    idle = launcher(logs=(LogRef("stdout", OUT, size=3_000_000, mtime=T0 - 13 * H),))
    assert view.quiet_stdout(idle, T0, 1800) and not view.quiet_stdout(idle, T0, 13 * 3600 + 60)
    assert not view.quiet_stdout(launcher(workdir=None), T0, 1800)
    assert not view.quiet_stdout(launcher(is_mine=False), T0, 1800)
    assert not view.quiet_stdout(launcher(logs=()), T0, 1800)
    ended = mk_job("k2", "5", JobState.FAILED, workdir=WD, logs=(LogRef("stdout", OUT, size=5, mtime=T0 - 13 * H),))
    assert not view.quiet_stdout(ended, T0, 1800)


def test_quiet_stdout_agrees_with_the_core_rule_at_the_last_sighting() -> None:
    from runfleet.core.logs import stdout_quiet_for

    seen_long_ago = launcher(observed_at=T0 - 2 * H, logs=(LogRef("stdout", OUT, size=202, mtime=T0 - 2 * H),))
    assert not view.quiet_stdout(seen_long_ago, T0, 1800)           # 0 s old when last seen: judged then, not now
    assert stdout_quiet_for(seen_long_ago.logs[0], T0, 1800) is not None
    for job in (launcher(), seen_long_ago, launcher(logs=(LogRef("stdout", OUT, size=202, mtime=T0),))):
        expect = stdout_quiet_for(job.logs[0], view._seen_now(job, T0), 1800) is not None
        assert view.quiet_stdout(job, T0, 1800) is expect


def test_job_card_points_at_other_logs_when_stdout_is_quiet() -> None:
    doc = view.job(lfv(launcher(), mk_job("k2", "8", logs=(LogRef("stdout", "/w/b.out", size=10**6, mtime=T0),))),
                   [("k2", "7"), ("k2", "8")])
    assert "  next     stdout is quiet; other logs: runfleet agent log k2:7 --files" in doc.text.splitlines()
    assert doc.text.count("stdout is quiet") == 1
    objs = {o["id"]: o for o in doc.data["jobs"]}
    assert objs["k2:7"]["other_logs"] == "runfleet agent log k2:7 --files --json" and objs["k2:8"]["other_logs"] is None
    idle = launcher(logs=(LogRef("stdout", OUT, size=10**6, mtime=T0 - 13 * H),))
    assert "stdout is quiet" in view.job(lfv(idle), [("k2", "7")]).text          # the fleet's own threshold
    assert "stdout is quiet" not in view.job(lfv(idle, stall_after=20 * 3600.0), [("k2", "7")]).text


def test_quiet_line_is_the_first_dropped_from_a_full_card() -> None:
    lines = ["k2:7  R  x", "  hints    " + "h" * 300, "  next     runfleet agent log k2:7 --lines 80",
             view.CARD_QUIET + "runfleet agent log k2:7 --files"]
    capped = sum(min(len(x), 200) + 1 for x in lines)              # what the 200-byte line cap leaves
    text = view._fit_card(lines, capped - 5)
    assert "stdout is quiet" not in text and "h" * 150 in text and "--lines 80" in text


# ---------------------------------------------------------------- agent.source.find_logs
def test_find_logs_without_network_for_a_session_host_waiting_for_a_login() -> None:
    from runfleet.config import FleetConfig, HostConfig
    from runfleet.core.clock import ManualClock

    hv = mk_host("k2", [launcher()], state=ConnState.NEEDS_AUTH, session_transport=True)
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))

    def no_transport(h: Any) -> Any:
        raise AssertionError("no transport for a host waiting for a login")

    got = asyncio.run(source.find_logs(cfg, hv, launcher(), clock=ManualClock(T0), timeout=5,
                                       transports=no_transport))
    assert got is not None and got.error is not None and got.error.kind is ErrorKind.AUTH and got.files == ()
    assert asyncio.run(source.find_logs(cfg, hv, launcher(workdir=None), clock=ManualClock(T0), timeout=5,
                                        transports=no_transport)) is None


def test_file_of_another_users_log_sends_no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.agent import commands
    from runfleet.config import FleetConfig, HostConfig
    from runfleet.core.clock import ManualClock

    async def no_find(*a: Any, **kw: Any) -> LogFiles:
        raise AssertionError("no find for another user's log")

    monkeypatch.setattr(source, "find_logs", no_find)
    bob_out = f"{WD}/logs/bob_9.out"
    bob = mk_job("k2", "9", user="bob", is_mine=False, workdir=WD, logs=(LogRef("stdout", bob_out, size=10, mtime=T0),))
    hv = mk_host("k2", [launcher(), bob])
    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    req = commands.build_request("log", {"id": "k2:7", "file": "logs/bob_9.out"}, cfg)
    doc = asyncio.run(commands._with_files(req, cfg, fleet(hv), hv, launcher(), "logs/bob_9.out",
                                           clock=ManualClock(T0), transports=lambda h: None, timeout=lambda: 5.0))
    assert doc.data["errors"][0]["code"] == "not_allowed"
