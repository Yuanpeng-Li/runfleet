"""[WP1] "Find other log files" (§4.7.2): the `log.find` section (quoting, bounds, an ended job's `until`), its parser
(sizes, mtimes, newest first, the cap, `incomplete` walks, find's failures, unsupported find), LogService.find's typed
errors, the real script through sh / dash / bash with GNU find on real files (symlinks, pruned and deep directories,
hostile workdir names), and the helpers both surfaces share (known_logs, suggest, stdout_quiet_for)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runfleet.core import logs
from runfleet.core.clock import ManualClock
from runfleet.core.logs import (FIND_END_SLACK, FIND_MAX_DEPTH, FIND_MAX_FILES, FIND_UNSUPPORTED, QUIET_TINY,
                                QUIET_TINY_AGE, FoundLog, LogFiles, LogService, find_section, known_logs, parse_find,
                                stdout_quiet_for, suggest)
from runfleet.models import ErrorKind, HostError, Job, JobKey, JobKind, JobState, LogRef
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, SectionOutcome
from runfleet.transport.base import ProbeResult
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START
from tests.integration.harness import MockEnv, run_sections, shells

NOW = CLOCK_START
WD = "/shared/lab/alice/vitfm"
T = int(NOW.timestamp())


# ---------------------------------------------------------------- the section
@pytest.mark.parametrize("wd", ["/data/my runs/x", "/data/alice's/x", "/data/$HOME/`id`/x", "/data/a\"b;rm -rf ~/x",
                                "/data/*/x"])
def test_section_quotes_the_workdir(wd: str) -> None:
    s = find_section(wd, None, NOW)
    first = s.cmd.split("\n", 1)[0]
    assert first == f"d={shlex.quote(wd)}" and shlex.split(first) == [f"d={wd}"]
    assert s.cmd.count(shlex.quote(wd)) == 1 and s.cmd.count('"$d"') == 7    # only through "$d" after that
    assert s.meta["workdir"] == wd and s.name == "log.find" and s.timeout == logs.FIND_TIMEOUT


def test_section_bounds_and_cutoff() -> None:
    since = NOW - timedelta(hours=13)
    s = find_section(WD + "/", since, NOW)
    assert s.meta == {"workdir": WD, "since": T - 13 * 3600 - 60, "until": None, "max_depth": FIND_MAX_DEPTH,
                      "max_files": FIND_MAX_FILES}
    assert f"-newermt @{T - 13 * 3600 - 60} " in s.cmd and f"-maxdepth {FIND_MAX_DEPTH} " in s.cmd
    assert "! -newermt" not in s.cmd                                       # a running job: no upper bound
    assert f"head -n {FIND_MAX_FILES + 1}" in s.cmd and "find -H" in s.cmd and " -L " not in s.cmd
    assert set(s.requires) == {"find", "sort", "head"} and s.max_stdout < 200_000
    assert find_section(WD, None, NOW).meta["since"] == T - 86400          # no start time: the last 24 h
    for name in logs.FIND_PRUNE:
        assert f"-name {shlex.quote(name)}" in s.cmd


def test_section_until_bounds_an_ended_job() -> None:
    end = NOW - timedelta(hours=5)
    s = find_section(WD, NOW - timedelta(hours=9), NOW, until=end)
    upper = T - 5 * 3600 + int(FIND_END_SLACK)
    assert s.meta["until"] == upper and f"-newermt @{T - 9 * 3600 - 60} ! -newermt @{upper} -printf" in s.cmd
    frac = find_section(WD, None, NOW, until=end + timedelta(seconds=0.25))      # rounded up, never earlier
    assert frac.meta["until"] == upper + 1


@pytest.mark.parametrize("wd", ["", "relative/dir", "/a\nb", "/a\rb", "/", "//", "/x/..", "/a/link/../job"])
def test_section_refuses_unusable_workdirs(wd: str) -> None:
    with pytest.raises(ValueError):
        find_section(wd, None, NOW)


def test_workdir_is_normalised() -> None:
    assert find_section("//shared/./lab//alice/", None, NOW).meta["workdir"] == "/shared/lab/alice"
    assert find_section("/shared/x..y", None, NOW).meta["workdir"] == "/shared/x..y"      # only a `..` component


# ---------------------------------------------------------------- the parser
def _line(ts: float, size: int, path: str) -> str:
    return f"{ts:.10f} {size} {path}\x1f\n"


def test_parse_sizes_mtimes_newest_first() -> None:
    out = (_line(T - 5.25, 21985, f"{WD}/runs/a/s42/train.log") + _line(T - 3600, 90, f"{WD}/runs/a/s42/eval.err")
           + _line(T - 1, 0, f"{WD}/wandb/run-1/files/output.log") + "@@end 0\n")
    r = parse_find(out, "", WD, NOW)
    assert r.error is None and r.file_error is None and not r.truncated and r.workdir == WD and r.fetched_at == NOW
    assert [f.path.rsplit("/", 1)[1] for f in r.files] == ["output.log", "train.log", "eval.err"]
    assert [f.size for f in r.files] == [0, 21985, 90]
    assert r.files[1].mtime == datetime.fromtimestamp(T - 5.25, tz=UTC)


def test_parse_cap_sets_truncated() -> None:
    out = "".join(_line(T - i, i, f"{WD}/r/{i:02d}.log") for i in range(FIND_MAX_FILES + 1)) + "@@end 0\n"
    r = parse_find(out, "", WD, NOW)
    assert r.truncated and len(r.files) == FIND_MAX_FILES and r.files[0].path == f"{WD}/r/00.log"
    exact = "".join(_line(T - i, i, f"{WD}/r/{i:02d}.log") for i in range(FIND_MAX_FILES)) + "@@end 0\n"
    assert not parse_find(exact, "", WD, NOW).truncated
    bad = _line(T + 1, 1, f"{WD}/bad\ufffd.log")                         # 31 records, one of them dropped
    dropped = parse_find(bad + exact, "", WD, NOW)
    assert len(dropped.files) == FIND_MAX_FILES and dropped.truncated      # the cap was hit all the same


def test_parse_permission_denied_is_an_incomplete_walk_not_an_error() -> None:
    err = "\n".join(f"find: '{WD}/private{i}': Permission denied" for i in range(50))
    out = (_line(T, 12, f"{WD}/a.log") + "garbage line\n" + _line(T, 5, "/etc/passwd.log")     # outside: dropped
           + _line(T, 5, f"{WD}x/sibling.log") + _line(T, 5, f"{WD}/bad\ufffd.log") + "@@end 1\n")
    r = parse_find(out, err, WD, NOW)
    assert r.file_error is None and not r.truncated and [f.path for f in r.files] == [f"{WD}/a.log"]
    assert r.incomplete == "some directories unreadable"
    none = parse_find("@@end 1\n", err, WD, NOW)                      # nothing found: no error, but say why
    assert none == LogFiles(WD, (), NOW, incomplete="some directories unreadable")


def test_parse_a_failed_find_is_never_an_empty_list() -> None:
    io = f"find: '{WD}/runs': Input/output error\n"
    r = parse_find("@@end 1\n", io, WD, NOW)                          # NFS hiccup: not "no log files"
    assert r.files == () and r.file_error == f"find: '{WD}/runs': Input/output error" and r.incomplete is None
    mixed = f"find: '{WD}/private': Permission denied\n" + io           # the real cause, not the private dir
    assert parse_find("@@end 1\n", mixed, WD, NOW).file_error == f"find: '{WD}/runs': Input/output error"
    assert parse_find("@@end 2\n", "", WD, NOW).file_error == "find failed (rc 2)"
    some = parse_find(_line(T, 3, f"{WD}/a.log") + "@@end 1\n", mixed, WD, NOW)
    assert [f.path for f in some.files] == [f"{WD}/a.log"] and some.file_error is None
    assert some.incomplete == "find failed (rc 1)" and not some.truncated
    gone = parse_find(_line(T, 3, f"{WD}/a.log") + "@@end 1\n", f"find: '{WD}/tmp/x1': No such file or directory\n",
                      WD, NOW)                                          # removed during the walk: nothing missed
    assert gone.incomplete is None and gone.file_error is None and len(gone.files) == 1
    long = parse_find("@@end 1\n", "find: " + "x" * 500 + ": Input/output error\n", WD, NOW)
    assert long.file_error is not None and len(long.file_error) <= 120


def _through_framing(stdout: str, stderr: str, rc: int = 1) -> tuple[str, str]:
    """(stdout, stderr) of a `log.find` section as the probe framing returns them: the remote ``head -c 4096`` keeps
    the first STDERR_LIMIT bytes of find's stderr, which may end mid-line."""
    script = framing.make_script([find_section(WD, None, NOW)], "log")
    rf = f"@@RF{script.nonce}"
    err = stderr.encode()[:framing.STDERR_LIMIT].decode("utf-8", "replace")
    framed = f"\n{rf} B log.find\n{stdout}\n{rf} X log.find\n{err}\n{rf} Y log.find\n\n{rf} E log.find {rc} 9\n"
    res = framing.parse(framed, script).sections["log.find"]
    return res.stdout, res.stderr


def test_parse_permission_denied_cut_at_the_stderr_limit() -> None:
    """Round 2: ~100 unreadable subdirectories fill find's 4096-byte stderr; its last line, cut mid-message, matched
    nothing and the walk read as `file_error` "fi" (nothing found) or "find failed (rc 1)" (files found)."""
    denied = "".join(f"find: '{WD}/runs/sweep/private{i:04d}': Permission denied\n" for i in range(120))
    assert len(denied.encode()) > framing.STDERR_LIMIT
    found = _line(T, 3, f"{WD}/a.log") + "@@end 1"
    for out, files in (("@@end 1", 0), (found, 1)):
        stdout, stderr = _through_framing(out, denied)
        assert len(stderr) == framing.STDERR_LIMIT and not stderr.endswith("\n")       # cut mid-line
        r = parse_find(stdout, stderr, WD, NOW)
        assert r.file_error is None and r.incomplete == "some directories unreadable" and len(r.files) == files
    whole = "".join(f"find: '{WD}/p{i:02d}': Permission denied\n" for i in range(10))   # under the cap: kept whole
    _, stderr = _through_framing("@@end 1", whole + "find: 'x': Input/output error")
    assert parse_find("@@end 1\n", stderr, WD, NOW).file_error == "find: 'x': Input/output error"
    io = "find: '" + "y" * 5000 + "': Input/output error\n"                         # one line over the cap: no guess
    _, stderr = _through_framing("@@end 1", io)
    assert parse_find("@@end 1\n", stderr, WD, NOW).file_error == "find failed (rc 1)"


def test_parse_the_workdir_itself_gone_is_no_such_directory() -> None:
    """Round 2: the workdir removed / renamed between the `[ -d ]` check and the walk read as a finished, empty walk
    ("no log files"); only a path below it, gone during the walk, is tolerated."""
    for quoted in (f"'{WD}'", f"`{WD}'", WD):                                     # GNU, old findutils, unquoted
        r = parse_find("@@end 1\n", f"find: {quoted}: No such file or directory\n", WD, NOW)
        assert r.files == () and r.file_error == "No such directory" and r.incomplete is None
    below = parse_find("@@end 1\n", f"find: '{WD}/tmp/x1': No such file or directory\n", WD, NOW)
    assert below.file_error is None and below.incomplete is None
    sibling = parse_find("@@end 1\n", f"find: '{WD}2': No such file or directory\n", WD, NOW)
    assert sibling.file_error is None


def test_a_workdir_gone_during_the_walk_is_no_such_directory_whatever_its_name() -> None:
    """GNU find escapes the name it prints under LC_ALL=C (`'/w/\\346…'`, `'/w/run\\''`): the probe itself answers
    @@nodir after a failed walk whose workdir is gone, so no stderr matching is needed."""
    cmd = logs.find_command(WD, 0)
    assert cmd.index('[ "$r" = 0 ] || [ -d "$d" ] || { echo @@nodir; exit 0; }') < cmd.index('echo "@@end $r"')
    r = parse_find(_line(T, 1, f"{WD}/a.log") + "\n@@nodir\n", "find: '/w/\\346\\225': No such file or directory\n",
                   WD, NOW)
    assert r.files == () and r.file_error == "No such directory"


def test_find_window_is_one_rule_for_both_surfaces() -> None:
    start, end = NOW - timedelta(days=3, hours=6), NOW - timedelta(days=3)
    running = replace(_job("k2:7"), start_time=start, end_time=NOW + timedelta(hours=3))   # end = its time limit
    assert logs.find_window(running) == (start, None)
    assert logs.find_window(replace(running, start_time=None)) == (None, None)            # the last 24 h
    assert logs.find_window(replace(running, est_start=True)) == (None, None)             # an estimate is no start
    ended = replace(running, state=JobState.COMPLETED, end_time=end)
    assert logs.find_window(ended) == (start, end)
    no_start = logs.find_window(replace(ended, start_time=None))                         # ended 3 days ago
    assert no_start == (end - timedelta(seconds=logs.FIND_WINDOW), end)                  # never "24 h from now"
    assert logs.find_window(replace(ended, est_start=True)) == no_start
    assert logs.find_window(replace(ended, end_time=None)) == (start, None)


@pytest.mark.parametrize(("marker", "error"), [("@@unsupported", FIND_UNSUPPORTED), ("@@nodir", "No such directory"),
                                               ("@@denied", "Permission denied")])
def test_parse_markers(marker: str, error: str) -> None:
    r = parse_find(marker + "\n", "", WD, NOW)
    assert r.file_error == error and r.files == () and not r.truncated


def test_parse_cut_or_stopped_walks_are_incomplete_not_truncated() -> None:
    one = _line(T, 1, f"{WD}/a.log")
    two = one + _line(T - 1, 2, f"{WD}/b.log")
    cut_out = parse_find(one, "", WD, NOW)                             # no @@end: the output was cut
    assert cut_out.incomplete == "output cut" and not cut_out.truncated and len(cut_out.files) == 1
    for rc in ("124", "137"):                                          # `timeout` stopped find
        r = parse_find(two + f"@@end {rc}\n", "", WD, NOW)
        assert r.incomplete == "timed out" and not r.truncated and len(r.files) == 2 and r.file_error is None
    zero = parse_find("@@end 124\n", "", WD, NOW)                      # stopped before finding anything
    assert zero.incomplete == "timed out" and zero.files == () and zero.file_error is None and not zero.truncated
    many = "".join(_line(T - i, i, f"{WD}/r/{i:02d}.log") for i in range(FIND_MAX_FILES + 1))
    both = parse_find(many + "@@end 124\n", "", WD, NOW)
    assert both.truncated and both.incomplete == "timed out" and len(both.files) == FIND_MAX_FILES
    cut = parse_find(one + f"{T}.5 77 {WD}/runs/tra", "", WD, NOW)   # the last record was cut mid-path
    assert [f.path for f in cut.files] == [f"{WD}/a.log"]
    cut = parse_find(one + f"{T}.5 77 {WD}/runs/tra\n@@end 124\n", "", WD, NOW)  # find killed, sort added "\n"
    assert cut.incomplete == "timed out" and [f.path for f in cut.files] == [f"{WD}/a.log"]
    sort = parse_find(one + "@@sort 2\n@@end 0\n", "", WD, NOW)        # sort died after some output
    assert sort.incomplete == "sort failed" and not sort.truncated and len(sort.files) == 1
    assert sort.file_error is None
    failed = parse_find("@@sort 2\n@@end 0\n", "sort: write failed: /tmp/sortX: No space left on device", WD, NOW)
    assert failed.files == () and failed.file_error == "sort: write failed: /tmp/sortX: No space left on device"
    assert parse_find("@@sort 2\n@@end 0\n", "", WD, NOW).file_error == "sort failed (rc 2)"
    empty = parse_find("", "sh: 1: find: Input/output error", WD, NOW)
    assert empty.file_error == "sh: 1: find: Input/output error" and empty.files == ()


# ---------------------------------------------------------------- LogService.find
class Probe:
    def __init__(self, t: FakeTransport) -> None:
        self.t = t

    async def __call__(self, host: str, script: ProbeScript) -> ProbeResult:
        return await self.t.probe(script, timeout=10)


def _service(t: FakeTransport, clock: ManualClock | None = None) -> LogService:
    return LogService(clock or ManualClock(NOW), transport_for=lambda h: t, stream_limit_for=lambda h: 1,
                      probe=Probe(t), caps_for=lambda h: None)


async def test_find_one_probe_and_result() -> None:
    t = FakeTransport("k2")
    t.set_result("log.find", _line(T, 7, f"{WD}/runs/s42/train.log") + "@@end 0")
    r = await _service(t).find("k2", WD + "/", since=NOW - timedelta(hours=1))
    assert [f.path for f in r.files] == [f"{WD}/runs/s42/train.log"] and r.workdir == WD
    assert t.count("log") == 1 and t.probes[0].names == ("log.find",)
    assert t.probes[0].script.sections[0].meta["since"] == T - 3600 - 60


async def test_find_errors_are_typed_and_never_raise() -> None:
    t = FakeTransport("k2")
    svc = _service(t)
    t.fail_next(ErrorKind.AUTH, summary="permission denied (publickey)")
    r = await svc.find("k2", WD, since=None)
    assert r.error is not None and r.error.kind is ErrorKind.AUTH and r.files == () and r.workdir == WD
    t.set_result("log.find", "", rc=124, outcome=SectionOutcome.TIMEOUT)
    r = await svc.find("k2", WD, since=None)
    assert r.error is not None and r.error.kind is ErrorKind.TIMEOUT and r.error.section == "log.find"
    t.set_result("log.find", "", rc=127, stderr="missing:find", outcome=SectionOutcome.MISSING)
    assert (await svc.find("k2", WD, since=None)).file_error == "unsupported: find not installed"
    t.set_result("log.find", "@@unsupported")
    assert (await svc.find("k2", WD, since=None)).file_error == FIND_UNSUPPORTED
    t.set_result("log.find", _line(T, 3, f"{WD}/a.log"), rc=124, outcome=SectionOutcome.TIMEOUT)
    partial = await svc.find("k2", WD, since=None)                     # some output, then the remote timeout
    assert partial.incomplete == "timed out" and not partial.truncated and len(partial.files) == 1
    assert partial.error is None
    t.set_result("log.find", f"{T}.5 77 {WD}/runs/tra", rc=124, outcome=SectionOutcome.TIMEOUT)
    nothing = await svc.find("k2", WD, since=None)                     # timed out before a whole record
    assert nothing.incomplete == "timed out" and nothing.files == () and nothing.file_error is None
    t.set_result("log.find", _line(T, 3, f"{WD}/a.log"), rc=0, outcome=SectionOutcome.INCOMPLETE)
    dropped = await svc.find("k2", WD, since=None)                     # the section ended early, no transport error
    assert dropped.incomplete == "output cut" and len(dropped.files) == 1
    t.fail_next(ErrorKind.SSH, summary="connection closed")       # records came, then the connection died
    lost = await svc.find("k2", WD, since=None)
    assert lost.error is not None and lost.error.kind is ErrorKind.SSH and lost.files == ()
    assert lost.incomplete is None
    before = t.count()
    bad = await svc.find("k2", "relative", since=None)                 # nothing is sent for a bad workdir
    assert bad.file_error and t.count() == before
    root = await svc.find("k2", "/", since=None)
    assert "refusing" in (root.file_error or "") and t.count() == before
    ancient = await svc.find("k2", WD, since=datetime.min.replace(tzinfo=UTC))   # the slack overflows: no raise
    assert ancient.file_error and t.count() == before


async def test_find_passes_until() -> None:
    t = FakeTransport("k2")
    t.set_result("log.find", "@@end 0")
    end = NOW - timedelta(hours=5)
    r = await _service(t).find("k2", WD, since=NOW - timedelta(hours=9), until=end)
    assert r.files == () and r.file_error is None and r.incomplete is None
    assert t.probes[0].script.sections[0].meta["until"] == int(end.timestamp()) + int(FIND_END_SLACK)


async def test_find_survives_a_raising_probe() -> None:
    async def boom(host: str, script: ProbeScript) -> ProbeResult:
        raise KeyError("lynx")

    svc = LogService(ManualClock(NOW), transport_for=lambda h: FakeTransport(h), stream_limit_for=lambda h: 1,
                     probe=boom, caps_for=lambda h: None)
    r = await svc.find("lynx", WD, since=None)
    assert isinstance(r.error, HostError) and r.error.kind is ErrorKind.INTERNAL


# ---------------------------------------------------------------- the real script (sh, dash, bash + GNU find)
def _gnu_find() -> str | None:
    path = shutil.which("find")
    if path is None:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return path if "GNU findutils" in out else None


GNU_FIND = _gnu_find()
real_find = pytest.mark.skipif(GNU_FIND is None, reason="needs GNU find")


def _env(tmp: Path, find: str, shell: str = "/bin/sh") -> MockEnv:
    env = MockEnv(tmp / "env", "k2", mocks=(), sh=shell)          # `sh -c` inside the probe is this shell too
    link = env.sysbin / "find"
    if not link.exists():
        link.symlink_to(find)
    return env


def _touch(path: Path, ago: float = 0.0, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    t = time.time() - ago
    os.utime(path, (t, t))
    return path


def _find(env: MockEnv, wd: str, since: datetime | None, shell: str = "/bin/sh",
          until: datetime | None = None) -> LogFiles:
    now = datetime.now(UTC)
    sec = find_section(wd, since, now, until)
    res = run_sections([sec], env.env(), shell=shell, purpose="log")["log.find"]
    assert res.ok, res
    return parse_find(res.stdout, res.stderr, str(sec.meta["workdir"]), now)


@pytest.mark.posix
@real_find
@pytest.mark.parametrize("shell", shells())
def test_real_find_rules(tmp_path: Path, shell: str) -> None:
    assert GNU_FIND is not None
    env = _env(tmp_path, GNU_FIND, shell)
    assert os.path.realpath(env.sysbin / "sh") == os.path.realpath(shell)
    wd = tmp_path / "it's a $HOME `id` dir"
    train = _touch(wd / "runs/in1k/r0412/s42/train.log", 5, "epoch 1\n" * 10)
    _touch(wd / "wandb/run-1/files/output.log", 60)
    _touch(wd / "logs/job_42.out", 120)
    _touch(wd / "runs/in1k/r0412/s42/eval.stderr", 30)
    _touch(wd / "runs/in1k/r0398/s42/train.log", 3 * 86400)                  # before the job started
    _touch(wd / "metrics.jsonl", 1)                                           # not a log name
    for pruned in (".git", ".venv-3.12/lib", "venv", "node_modules/x", "__pycache__", ".cache/pip",
                   "lib/python3.12/site-packages/pkg"):
        _touch(wd / pruned / "build.log", 1)
    _touch(wd / "a/b/c/d/e/f/deep.log", 1)                                    # depth 7
    _touch(wd / "a/b/c/d/e/ok.log", 2)                                        # depth 6
    _touch(wd / "new\nline.log", 1)
    outside = tmp_path / "outside"
    _touch(outside / "secret.log", 1)
    (wd / "linked-dir").symlink_to(outside)                                   # never entered
    (wd / "linked.log").symlink_to(outside / "secret.log")                    # never listed
    r = _find(env, str(wd), datetime.now(UTC) - timedelta(hours=1), shell)
    names = [f.path.removeprefix(str(wd) + "/") for f in r.files]
    assert names == ["a/b/c/d/e/ok.log", "runs/in1k/r0412/s42/train.log", "runs/in1k/r0412/s42/eval.stderr",
                     "wandb/run-1/files/output.log", "logs/job_42.out"]
    assert r.files[1].size == train.stat().st_size and not r.truncated and r.file_error is None
    assert r.incomplete is None
    assert abs(r.files[1].mtime.timestamp() - train.stat().st_mtime) < 1e-3
    assert not (tmp_path / "id").exists()                                     # nothing in the name ran
    old = _find(env, str(wd), None, shell)                                    # last 24 h: the 3-day-old run stays out
    assert "runs/in1k/r0398/s42/train.log" not in [f.path.removeprefix(str(wd) + "/") for f in old.files]
    # an ended job (ended 2 h ago): what was written after its end + 5 min belongs to later jobs
    _touch(wd / "logs/after_end.out", 2 * 3600 - FIND_END_SLACK + 60)          # 4 min after the end: kept
    _touch(wd / "logs/next_job.out", 2 * 3600 - FIND_END_SLACK - 60)           # 6 min after: left out
    ended = _find(env, str(wd), datetime.now(UTC) - timedelta(hours=10), shell,
                  until=datetime.now(UTC) - timedelta(hours=2))
    assert [f.path.removeprefix(str(wd) + "/") for f in ended.files] == ["logs/after_end.out"]


@pytest.mark.posix
@real_find
def test_real_find_cap_symlinked_workdir_and_missing(tmp_path: Path) -> None:
    assert GNU_FIND is not None
    env = _env(tmp_path, GNU_FIND)
    real = tmp_path / "real"
    for i in range(FIND_MAX_FILES + 5):
        _touch(real / f"r/{i:02d}.log", i)
    (tmp_path / "wd").symlink_to(real)                                        # the workdir itself may be a link
    r = _find(env, str(tmp_path / "wd"), None)
    assert r.truncated and len(r.files) == FIND_MAX_FILES and r.incomplete is None
    assert [f.path for f in r.files[:2]] == [f"{tmp_path}/wd/r/00.log", f"{tmp_path}/wd/r/01.log"]
    gone = _find(env, str(tmp_path / "nope"), None)
    assert gone.file_error == "No such directory" and gone.files == ()


@pytest.mark.posix
@real_find
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads every directory")
def test_real_find_tolerates_unreadable_subdirs(tmp_path: Path) -> None:
    assert GNU_FIND is not None
    env = _env(tmp_path, GNU_FIND)
    wd = tmp_path / "wd"
    _touch(wd / "ok.log", 1)
    _touch(wd / "private/hidden.log", 1)
    (wd / "private").chmod(0)
    try:
        r = _find(env, str(wd), None)
    finally:
        (wd / "private").chmod(0o700)
    assert [f.path for f in r.files] == [f"{wd}/ok.log"] and r.file_error is None
    assert r.incomplete == "some directories unreadable" and not r.truncated
    many = tmp_path / "shared"                       # find's stderr over the framing's 4096 bytes, cut mid-line
    private = [many / f"sweep-run-{i:04d}-private" for i in range(120)]
    for d in private:
        d.mkdir(parents=True)
        d.chmod(0)
    try:
        none = _find(env, str(many), None)
        _touch(many / "ok.log", 1)
        some = _find(env, str(many), None)
    finally:
        for d in private:
            d.chmod(0o700)
    assert none.files == () and none.file_error is None and none.incomplete == "some directories unreadable"
    assert len(some.files) == 1 and some.file_error is None and some.incomplete == "some directories unreadable"


@pytest.mark.posix
def test_a_workdir_gone_before_the_capability_check_is_no_such_directory(tmp_path: Path) -> None:
    """Codex round 3: the workdir removed between `[ -d ]` and the GNU-find check read as "unsupported"."""
    fake = tmp_path / "fake-find"
    fake.write_text('#!/bin/sh\nrm -rf "$2"\nexit 1\n')                  # `find -H "$d" -maxdepth 0 …`
    fake.chmod(0o755)
    env = _env(tmp_path, str(fake))
    _touch(tmp_path / "wd/a.log", 1)
    r = _find(env, str(tmp_path / "wd"), None)
    assert r.file_error == "No such directory" and r.files == ()


@pytest.mark.posix
@pytest.mark.skipif(shutil.which("busybox") is None, reason="needs busybox")
def test_busybox_find_is_reported_unsupported(tmp_path: Path) -> None:
    busybox = shutil.which("busybox")
    assert busybox is not None
    env = _env(tmp_path, busybox)
    _touch(tmp_path / "wd/a.log", 1)
    r = _find(env, str(tmp_path / "wd"), None)
    assert r.file_error == FIND_UNSUPPORTED and r.files == ()


# ---------------------------------------------------------------- shared by the TUI and the agent
def _job(key: str, *paths: tuple[str, str | None], state: JobState = JobState.RUNNING) -> Job:
    host, _, jid = key.partition(":")
    logs_ = tuple(LogRef(stream=stream, path=path) for stream, path in paths)    # type: ignore[arg-type]
    return Job(key=JobKey(host, jid), kind=JobKind.SLURM, name="x", user="alice", state=state, raw_state=state.value,
               job_id_raw=jid, workdir=WD, logs=logs_)


def test_known_logs_labels_other_jobs_stdout_and_stderr() -> None:
    me = _job("k2:3372131", ("stdout", f"{WD}/logs/me_3372131.out"))
    other = _job("k2:3372094", ("stdout", f"{WD}//logs/./ddp_3372094.out"), ("stderr", f"{WD}/logs/ddp_3372094.err"))
    task = Job(key=JobKey("k2", "p642229-1788696459"), kind=JobKind.TASK, name="t", user="alice",
               state=JobState.RUNNING, raw_state="running", job_id_raw="642229",
               logs=(LogRef(stream="stdout", path=f"{WD}/nohup.out"), LogRef(stream="stderr", path=None),
                     LogRef(stream="stdout", path="relative.out"), LogRef(stream="file", path=f"{WD}/x.log")))
    elsewhere = _job("vega:9", ("stdout", f"{WD}/logs/vega_9.out"))         # another host's filesystem
    got = known_logs([me, other, task, elsewhere], me.key)
    assert got == {f"{WD}/logs/ddp_3372094.out": "job 3372094 stdout", f"{WD}/logs/ddp_3372094.err": "job 3372094 stderr",
                   f"{WD}/nohup.out": "job p642229 stdout"}
    assert known_logs([me], me.key) == {}


def test_suggest_skips_own_and_known_logs() -> None:
    def f(name: str, ago: float) -> FoundLog:
        return FoundLog(path=f"{WD}/{name}", size=1, mtime=NOW - timedelta(seconds=ago))

    files = [f("logs/ddp_3372094.out", 1), f("logs/me_3372131.out", 2), f("runs/s42/train.log", 5), f("eval.err", 9)]
    skip = {f"{WD}/logs/me_3372131.out", *known_logs([_job("k2:3372094", ("stdout", f"{WD}/logs/ddp_3372094.out"))],
                                                    JobKey("k2", "3372131"))}
    assert suggest(files, skip) == files[2]
    assert suggest(list(reversed(files)), skip) == files[2]                       # newest, whatever the order
    assert suggest(files, {f"/{WD}/./logs/me_3372131.out", f"{WD}/logs//ddp_3372094.out"}) == files[2]   # normalised
    assert suggest(files[:2], skip) is None and suggest([], skip) is None        # nothing left: no suggestion
    assert suggest(files, ()) == files[0]


def test_stdout_quiet_for() -> None:
    def ref(size: int | None, ago: float | None) -> LogRef:
        return LogRef(stream="stdout", path=f"{WD}/o.out", size=size,
                      mtime=None if ago is None else NOW - timedelta(seconds=ago))

    stall = 1800.0
    assert stdout_quiet_for(None, NOW, stall) is None
    assert stdout_quiet_for(ref(10, None), NOW, stall) is None                   # mtime unknown
    assert stdout_quiet_for(ref(202, 0), NOW, stall) is None                     # tiny but just written (round 1 bug)
    assert stdout_quiet_for(ref(QUIET_TINY, QUIET_TINY_AGE - 1), NOW, stall) is None
    assert stdout_quiet_for(ref(QUIET_TINY, QUIET_TINY_AGE), NOW, stall) == QUIET_TINY_AGE
    assert stdout_quiet_for(ref(QUIET_TINY + 1, QUIET_TINY_AGE), NOW, stall) is None   # a real log, recently written
    assert stdout_quiet_for(ref(None, QUIET_TINY_AGE), NOW, stall) is None
    assert stdout_quiet_for(ref(10 << 20, stall), NOW, stall) is None            # exactly stall_after: not yet
    assert stdout_quiet_for(ref(10 << 20, stall + 1), NOW, stall) == stall + 1
    assert stdout_quiet_for(ref(539, 13 * 3600), NOW, stall) == 13 * 3600        # the field report
