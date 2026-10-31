"""`runfleet agent <verb>` and `runfleet skill install|update|uninstall|status` (and the automatic skill refresh)
through `cli.main` (DESIGN.md §8.7, docs/AGENT_INTERFACE.md).

Everything runs on `--demo` (simulated fleet, in memory) or on a config and cache under tmp_path; ~/.config,
~/.local/state, the real mux directory and HOME are never touched, and the real-config tests forbid any process.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import runfleet
from runfleet import cli
from runfleet.agent import commands, skill
from runfleet.core.cache import Cache
from runfleet.core.clock import ManualClock
from runfleet.agent.data import dest_hash
from runfleet.models import ConnState, HostSnapshot, HostStatus, JobState, SectionStatus
from tests.cli.fakes import job
from tests.conftest import CLOCK_START

NOW = CLOCK_START + timedelta(minutes=3)
ENVELOPE = {"schema", "runfleet", "generated_at", "hosts", "truncated", "errors"}
SKILL_MD = Path(skill.__file__).resolve().parent / "skill" / "SKILL.md"


def _drop_log_handlers() -> None:
    root = logging.getLogger("runfleet")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


class Env:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.cache_path = tmp_path / "state" / "cache.sqlite"
        self.config_path = tmp_path / "config.toml"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.clock = ManualClock(NOW)
        monkeypatch.setattr(cli.paths, "config_file", lambda: self.config_path)
        monkeypatch.setattr(cli.paths, "log_file", lambda: tmp_path / "state" / "runfleet.log")
        monkeypatch.setattr(cli.paths, "cache_file", lambda: self.cache_path)
        monkeypatch.setattr(cli.paths, "mux_dir", lambda: tmp_path / "mux")
        monkeypatch.setattr(cli.paths, "debug_dir", lambda: tmp_path / "state" / "debug")
        monkeypatch.setattr(cli, "RealClock", lambda: self.clock)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: self.home))
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("USERPROFILE", str(self.home))

    def config(self, text: str) -> None:
        self.config_path.write_text(text, encoding="utf-8")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Env]:
    _drop_log_handlers()
    yield Env(monkeypatch, tmp_path)
    _drop_log_handlers()


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = cli.main(args)
    out, err = capsys.readouterr()
    return rc, out, err


def run_json(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    rc, out, err = run([*args, "--json"], capsys)
    assert err == "" or "--debug" in err, err
    return rc, json.loads(out)


def demo_ids(capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    """(a failed job of mine with a log path, a running one) of the demo fleet."""
    _rc, doc = run_json(["--demo", "agent", "jobs", "--state", "all", "--limit", "400"], capsys)
    jobs = doc["jobs"]
    failed = next(j for j in jobs if j["group"] == "failed" and j["mine"] and j["logs"]["stdout"]
                  and j["kind"] == "slurm")
    running = next(j for j in jobs if j["group"] == "active" and j["mine"])
    return failed["id"], running["id"]


# ====================================================================================================== demo verbs
@pytest.mark.parametrize(("args", "rc"), [
    (["brief"], 4),                                   # demo lynx needs a login: partial
    (["brief", "k2", "vega"], 0),
    (["jobs"], 4),
    (["jobs", "k2", "--state", "failed"], 0),
    (["jobs", "--all-users", "--limit", "3"], 4),
    (["gpus"], 4),
    (["gpus", "k2", "--type", "h200", "--min", "1"], 0),
    (["host", "k2"], 0),
    (["host", "lynx"], 4),
    (["events", "--since", "2h"], 4),
    (["events", "--since", "90m", "k2"], 0),
    (["job", "lynx:1"], 1),                           # no data for lynx: never "not found"
])
def test_every_verb_text_and_json(env: Env, capsys: pytest.CaptureFixture[str], args: list[str], rc: int) -> None:
    code, out, err = run(["--demo", "agent", *args], capsys)
    assert (code, err) == (rc, "")
    assert out.startswith("# runfleet ") and " · demo" in out.splitlines()[0]
    code, doc = run_json(["--demo", "agent", *args], capsys)
    assert code == rc
    assert doc["schema"] == f"runfleet.agent.{args[0]}/1"
    assert ENVELOPE <= doc.keys()
    assert all(h["source"] == "demo" for h in doc["hosts"])


def test_job_and_log_of_a_failed_job(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    failed, running = demo_ids(capsys)
    rc, out, _ = run(["--demo", "agent", "job", failed, running], capsys)
    assert rc == 0 and failed in out and running in out and f"agent log {failed}" in out      # (--demo kept)
    rc, doc = run_json(["--demo", "agent", "job", failed], capsys)
    assert rc == 0 and [j["id"] for j in doc["jobs"]] == [failed] and "pending" in doc["jobs"][0]
    rc, out, _ = run(["--demo", "agent", "log", failed, "--lines", "5"], capsys)
    assert rc == 0 and "last 5 of" in out
    rc, doc = run_json(["--demo", "agent", "log", failed, "--lines", "5"], capsys)
    assert rc == 0 and doc["schema"] == "runfleet.agent.log/1" and len(doc["log"]["lines"]) == 5
    rc, out, _ = run(["--demo", "agent", "log", failed, "--path-only"], capsys)
    assert rc == 0 and len(out.splitlines()) == 2
    rc, doc = run_json(["--demo", "agent", "log", failed, "--grep", "error|Traceback", "--stream", "out"], capsys)
    assert rc == 0 and doc["log"]["grep"]["terms"] == ["error", "traceback"]


def test_a_task_key_one_demo_call_prints_names_the_task_in_the_next(env: Env,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """M2: every `--demo` call builds a fresh world at its own now; task ids (p<pid>-<start>) derived from that
    now changed between calls, so a printed `runfleet agent job labstation:p…` named nothing in the next one."""
    _rc, doc = run_json(["--demo", "agent", "jobs", "labstation", "--state", "all"], capsys)
    task = next(j["key"] for j in doc["jobs"] if j["kind"] == "task" and j["mine"])      # the full key
    env.clock.advance(7 * 60)                                  # a later call, a new world
    rc, doc = run_json(["--demo", "agent", "job", task], capsys)
    assert rc == 0 and [j["key"] for j in doc["jobs"]] == [task]
    host, _, key = task.partition(":")
    pid, start = key[1:].split("-")
    off_by_one = f"{host}:p{pid}-{int(start) + 1}"               # another collector's `date - etimes`
    rc, doc = run_json(["--demo", "agent", "job", off_by_one], capsys)
    assert rc == 0 and [j["key"] for j in doc["jobs"]] == [task]


def test_a_usage_error_echoing_input_is_scrubbed_escaped_and_capped(env: Env,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """F6: `runfleet agent` quotes a bad host as given on stderr (read by an agent): no secret, no escapes, capped."""
    token = "hf_" + "Q" * 30
    rc, out, err = run(["--demo", "agent", "host", f"{token}\x1b[2J\nforged" + "x" * 2000], capsys)
    assert rc == 2 and out == "" and token not in err and "\x1b" not in err
    assert err.count("\n") == 1 and len(err.encode()) <= len("runfleet: \n") + commands.ERROR_TEXT_MAX


def test_log_head_shows_the_first_lines(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    failed, _running = demo_ids(capsys)
    rc, doc = run_json(["--demo", "agent", "log", failed, "--head", "--lines", "3"], capsys)
    assert rc == 0 and doc["log"]["first_line"] == 1 and len(doc["log"]["lines"]) == 3
    rc, out, _ = run(["--demo", "agent", "log", failed, "--head", "--lines", "3"], capsys)
    assert rc == 0 and "# first 3 of" in out


def test_not_found_and_unknown_ids(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = run_json(["--demo", "agent", "job", "k2:999999999"], capsys)
    assert rc == 1 and doc["errors"][0]["code"] == "not_found"
    rc, doc = run_json(["--demo", "agent", "log", "k2:999999999"], capsys)
    assert rc == 1 and doc["log"] is None and doc["errors"][0]["code"] == "not_found"


def test_help(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["agent", "help"], capsys)        # no config, no cache, no demo: still answers
    assert rc == 0 and "problem codes" in out and len(out.encode()) <= 6144
    rc, out, _ = run(["agent", "help", "log"], capsys)
    assert rc == 0 and "--path-only" in out and "gpus [HOST…]" not in out
    rc, doc = run_json(["agent", "help"], capsys)
    assert rc == 0 and doc == dict(commands.run("help", {}).data)
    assert doc["schema"] == "runfleet.agent.help/1" and "brief" in doc["verbs"]


@pytest.mark.parametrize("args", [
    ["jobs", "--state", "bogus"],
    ["jobs", "nosuch"],
    ["host", "nosuch"],
    ["gpus", "k2", "nosuch"],
    ["job", "nosuch:1"],
    ["job", "k2"],                                    # not HOST:ID
    ["job", *[f"k2:{n}" for n in range(11)]],          # at most 10
    ["events", "--since", "yesterday"],
    ["jobs", "--since", "2 hours"],
    ["events", "--since", "0001-01-01T00:00:00+01:00"],     # no UTC time: out of range
    ["help", "nosuch"],
    ["log", "k2:1", "--grep", "|".join("abcdefghijklmnopqrstuvwxyz")],      # > 20 alternatives
])
def test_usage_errors_exit_2_on_stderr(env: Env, capsys: pytest.CaptureFixture[str], args: list[str]) -> None:
    rc, out, err = run(["--demo", "agent", *args], capsys)
    assert rc == 2 and out == "" and err.startswith("runfleet: ")


@pytest.mark.parametrize("args", [["frobnicate"], ["log", "k2:1", "--stream", "both"], ["log", "k2:1", "--lines", "0"],
                                  ["events"], ["jobs", "--limit", "-1"], ["gpus", "--min", "x"]])
def test_bad_flags_are_argparse_errors(env: Env, capsys: pytest.CaptureFixture[str], args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--demo", "agent", *args])
    assert exc.value.code == 2 and "usage:" in capsys.readouterr().err


TOKEN = "hf_" + "Q" * 30


@pytest.mark.parametrize("args", [
    ["agent", f"{TOKEN} \x1b[2J\u202eforged"],                               # invalid verb choice
    ["agent", "brief", f"--bogus={TOKEN}\x1b[2J"],                            # unrecognized argument
    ["agent", "jobs", "--limit", f"{TOKEN} \x1b[2J"],                         # a bad value, quoted back
    ["agent", "log", "k2:1", "--stream", f"{TOKEN} \x1b[2J"],                 # invalid choice of a verb flag
    ["mcp", f"--{TOKEN}\x1b[2J"],
])
def test_argparse_errors_of_agent_commands_are_scrubbed(env: Env, capsys: pytest.CaptureFixture[str],
                                                        args: list[str]) -> None:
    """Final review 4: argparse errors under `agent` / `mcp` echoed the input raw (a token, terminal escapes);
    they go through commands.safe_text like every other agent error."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--demo", *args])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "usage:" in err and "error:" in err
    assert TOKEN not in err and "\x1b" not in err and "\u202e" not in err


def test_argparse_errors_elsewhere_are_unchanged(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["status", "--bogus=plain"])
    assert "unrecognized arguments: --bogus=plain" in capsys.readouterr().err


@pytest.mark.parametrize("command", [["agent", "brief"], ["agent", "help"], ["mcp"]])
def test_a_config_error_under_agent_names_only_the_key(env: Env, capsys: pytest.CaptureFixture[str],
                                                       command: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Final review 4: a ConfigError under `agent` / `mcp` printed its message, which may quote config values
    (an ssh target); now only the key path, as the MCP server reports it."""
    from runfleet.errors import ConfigError

    def broken(*a: Any, **kw: Any) -> Any:
        raise ConfigError("host[0].ssh", "bad value 'secret-jump.example.org'")
    monkeypatch.setitem(cli.COMMANDS, "agent", broken)
    monkeypatch.setitem(cli.COMMANDS, "mcp", broken)
    rc, out, err = run(["-c", str(env.config_path), *command], capsys)
    assert rc == 2 and out == "" and "secret-jump" not in err
    assert err.strip() == ("runfleet: the RunFleet config is invalid at host[0].ssh; ask the user to run "
                           "`runfleet doctor` with the same -c option (details in runfleet.log)")
    rc, _out, err = run(command, capsys)
    assert rc == 2 and "run `runfleet doctor` (details" in err and "secret-jump" not in err


def test_a_real_invalid_config_under_agent(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.config('[[host]]\nname = "k2"\nssh = "k2"\nmux = "secret-jump.example.org"\n')
    rc, out, err = run(["agent", "brief"], capsys)
    assert rc == 2 and out == "" and "secret-jump" not in err and "invalid at host[0].mux" in err


def test_brief_budgets(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    _rc, out, _ = run(["--demo", "agent", "brief"], capsys)
    assert len(out.rstrip("\n").encode()) <= 2048
    _rc, out, _ = run(["--demo", "agent", "brief", "--json"], capsys)
    assert len(out.rstrip("\n").encode()) <= 6144


def test_global_flags_after_the_verb_and_hosts_around_options(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = run_json(["agent", "jobs", "k2", "--state", "all", "vega", "--demo", "--demo-seed", "7"], capsys)
    assert rc == 0 and [h["name"] for h in doc["hosts"]] == ["k2", "vega"]
    rc, doc = run_json(["agent", "job", "k2:1", "--demo", "vega:2"], capsys)
    assert {e["job"] for e in doc["errors"]} == {"k2:1", "vega:2"}


def test_limit_truncates_with_a_next_command(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, doc = run_json(["--demo", "agent", "jobs", "--state", "all", "--limit", "2"], capsys)
    assert len(doc["jobs"]) == 2 and doc["truncated"]["shown"] == 2
    assert doc["truncated"]["next"].startswith("runfleet --demo agent jobs --state all --limit ")


def test_printed_commands_carry_the_demo_seed_unless_default(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """C5: a next command without --demo-seed would show another fleet."""
    rc, doc = run_json(["--demo", "--demo-seed", "11", "agent", "jobs", "--state", "all", "--limit", "2"], capsys)
    assert doc["truncated"]["next"].startswith("runfleet --demo --demo-seed 11 agent jobs --state all --limit ")
    rc, out, _ = run(["--demo", "--demo-seed", "11", "agent", "brief"], capsys)
    nxt = [ln for ln in out.splitlines() if "runfleet " in ln and not ln.startswith("# runfleet")]
    assert nxt and all("runfleet --demo --demo-seed 11 " in ln for ln in nxt)
    rc, out, _ = run(["--demo", "--demo-seed", "7", "agent", "brief"], capsys)
    assert "--demo-seed" not in out


def test_all_users_and_state_help(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """C4: --all-users names every column other users' rows carry; --state explains finished and ended."""
    with pytest.raises(SystemExit):
        cli.main(["agent", "jobs", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "(user, id, state, GPUs, partition only)" in out
    assert "finished = done,failed,cancelled,ended" in out and "ended = workstation task gone, exit unknown" in out


def test_no_cache_flag_is_a_usage_error(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, err = run(["agent", "brief", "--no-cache"], capsys)
    assert rc == 2 and "--no-cache" in err


# ====================================================================================================== other log files
LAUNCHER = "k2:3372131"          # demo.world.LAUNCHER_JOB: stdout = a launcher's few lines, train.log under the workdir
TRAIN_REL = "runs/in1k-b16/r0412/s42/train.log"


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The section names every demo probe of the call ran (in order)."""
    from runfleet.demo.transport import DemoTransport

    seen: list[str] = []
    real = DemoTransport.probe

    async def spy(self: DemoTransport, script: Any, *, timeout: float) -> Any:
        seen.extend(s.name for s in script.sections)
        return await real(self, script, timeout=timeout)

    monkeypatch.setattr(DemoTransport, "probe", spy)
    return seen


def test_log_files_lists_the_launchers_real_log(env: Env, capsys: pytest.CaptureFixture[str],
                                               probes: list[str]) -> None:
    from runfleet.demo import world

    assert LAUNCHER == f"k2:{world.LAUNCHER_JOB}" and world.LAUNCHER_TRAIN_LOG.endswith("/" + TRAIN_REL)
    rc, out, err = run(["--demo", "agent", "log", LAUNCHER, "--files"], capsys)
    assert (rc, err) == (0, "") and probes.count("log.find") == 1 and "log.peek" not in probes
    lines = out.splitlines()
    assert lines[1].startswith(f"# {LAUNCHER} log files under {world.LAUNCHER_WORKDIR} written since ")
    rows = [ln.split() for ln in lines[2:] if ln.startswith("  ")]
    assert rows[0][0] == TRAIN_REL                                             # newest first
    assert any(r[0].startswith("logs/") and r[-1] == "stdout" for r in rows)  # the job's own stdout, marked
    assert {"eval.err", "output.log"} <= {r[0].rsplit("/", 1)[-1] for r in rows}
    assert not any(x in out for x in ("metrics.jsonl", ".venv", "r0398", "prep.log"))
    assert lines[-1] == f"next: runfleet --demo agent log {LAUNCHER} --file {TRAIN_REL}"
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--files"], capsys)
    assert rc == 0 and doc["log"] is None and doc["files"][0]["path"] == TRAIN_REL
    assert doc["find"]["next"] == f"runfleet --demo agent log {LAUNCHER} --file {TRAIN_REL} --json"
    assert len(out.encode()) <= 24 << 10 and len(json.dumps(doc).encode()) <= 32 << 10


def test_log_file_reads_a_listed_file_like_stdout(env: Env, capsys: pytest.CaptureFixture[str],
                                                 probes: list[str]) -> None:
    rc, out, _ = run(["--demo", "agent", "log", LAUNCHER, "--file", TRAIN_REL, "--lines", "5"], capsys)
    assert rc == 0 and [s for s in probes if s.startswith("log.")] == ["log.find", "log.peek"]
    assert "# last 5 of" in out and "epoch" in out and out.splitlines()[1].startswith(f"# {LAUNCHER} file /")
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--file", TRAIN_REL, "--grep", "epoch"], capsys)
    assert rc == 0 and doc["log"]["stream"] == "file" and doc["log"]["path"].endswith("/" + TRAIN_REL)
    assert doc["file"] == TRAIN_REL and doc["log"]["grep"]["total"] > 0
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--file", f"./runs/../{TRAIN_REL}", "--head",
                        "--lines", "3"], capsys)
    assert rc == 0 and doc["log"]["first_line"] == 1 and len(doc["log"]["lines"]) == 3      # normalised, then read
    absolute = doc["log"]["path"]
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--file", absolute, "--lines", "1"], capsys)
    assert rc == 0 and doc["file"] == TRAIN_REL


@pytest.mark.parametrize(("path", "outside"), [
    ("../../../../etc/passwd", True), ("/etc/passwd", True), ("/shared/lab/alice/vitfm2/x.log", True),
    ("runs/in1k-b16/r0412/s42/metrics.jsonl", False),           # not a log name: the find never lists it
    ("runs/in1k-b16/r0398/s42/train.log", False),               # written 3 days before the job started
    (".venv/build.log", False),                                 # pruned
])
def test_log_file_refuses_what_the_find_does_not_list(env: Env, capsys: pytest.CaptureFixture[str],
                                                     probes: list[str], path: str, outside: bool) -> None:
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--file", path], capsys)
    err = doc["errors"][0]
    assert rc == 1 and err["code"] == "not_listed" and doc["log"] is None and err["job"] == LAUNCHER
    assert "log.peek" not in probes and "log.head" not in probes                # never read
    assert ("log.find" not in probes) is outside                                  # outside: refused before ssh
    assert (err["message"] == "the path is outside the job's workdir") is outside
    assert err["hint"] == f"run runfleet --demo agent log {LAUNCHER} --files and read a file it lists"


@pytest.mark.parametrize(("path", "message"), [
    ("runs/..", "the path is the job's workdir, not a file"), (".", "the path is the job's workdir, not a file"),
    ("runs/in1k-b16/", "the path names a directory, not a file"),
])
def test_log_file_naming_the_workdir_or_a_directory(env: Env, capsys: pytest.CaptureFixture[str],
                                                   probes: list[str], path: str, message: str) -> None:
    rc, doc = run_json(["--demo", "agent", "log", LAUNCHER, "--file", path], capsys)
    assert rc == 1 and doc["errors"][0]["code"] == "not_listed" and doc["errors"][0]["message"] == message
    assert "log.find" not in probes


def test_log_files_of_an_ended_job_in_a_shared_workdir(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """An ended job's list stops at its end (+5 min), other jobs' stdout is marked and never suggested."""
    _rc, doc = run_json(["--demo", "agent", "jobs", "k2", "--state", "finished", "--limit", "400"], capsys)
    ended = [j["id"] for j in doc["jobs"] if j["mine"] and j.get("ended_at")]
    assert ended
    seen_other = False
    for jid in ended:
        rc, got = run_json(["--demo", "agent", "log", jid, "--files"], capsys)
        if rc != 0:
            continue
        find, files = got["find"], got["files"]
        assert find["until"] is not None and all(f["mtime"] <= find["until"] for f in files)
        others = {f["path"] for f in files if f["other_job"]}
        seen_other |= bool(others)
        assert all(f["other_job"].startswith("job ") for f in files if f["other_job"])
        nxt = find["next"]
        assert nxt is None or not any(nxt.split(" --file ", 1)[1].startswith(p + " ") for p in others)
        assert nxt is None or not any(f["stream"] and nxt.split(" --file ", 1)[1].startswith(f["path"] + " ")
                                      for f in files)
    assert seen_other


def test_job_card_quiet_line_follows_the_tui_rule(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """A tiny stdout written just now is not "quiet" (round 1 called the demo eval task's 202 B stdout, written 0 s
    ago, quiet); the launcher's 539 B stdout, unchanged for 13 h, is."""
    _rc, card = run_json(["--demo", "agent", "job", LAUNCHER], capsys)
    assert card["jobs"][0]["other_logs"] is not None and card["jobs"][0]["logs"]["stdout_size"] == 539
    _rc, evals = run_json(["--demo", "agent", "jobs", "labstation", "--name", "eval_ckpt*", "--state", "active"], capsys)
    fresh = [j["id"] for j in evals["jobs"]]
    assert fresh
    _rc, card = run_json(["--demo", "agent", "job", fresh[0]], capsys)
    assert card["jobs"][0]["logs"]["stdout_size"] == 202 and card["jobs"][0]["other_logs"] is None


def test_log_files_of_another_users_job_is_not_allowed(env: Env, capsys: pytest.CaptureFixture[str],
                                                      probes: list[str]) -> None:
    _rc, doc = run_json(["--demo", "agent", "jobs", "--all-users", "--state", "active", "--limit", "400"], capsys)
    other = next(j["id"] for j in doc["jobs"] if not j["mine"] and j["group"] == "active")
    probes.clear()
    for flags in (["--files"], ["--file", "x.log"]):
        rc, doc = run_json(["--demo", "agent", "log", other, *flags], capsys)
        assert rc == 1 and doc["errors"][0]["code"] == "not_allowed" and doc["files"] is None
    assert "log.find" not in probes


def test_job_card_of_the_launcher_points_at_its_other_logs(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["--demo", "agent", "job", LAUNCHER], capsys)
    quiet = [ln for ln in out.splitlines() if "stdout is quiet" in ln]
    assert rc == 0 and quiet == [f"  next     stdout is quiet; other logs: runfleet --demo agent log {LAUNCHER} --files"]
    rc, doc = run_json(["--demo", "agent", "job", LAUNCHER], capsys)
    assert doc["jobs"][0]["other_logs"] == f"runfleet --demo agent log {LAUNCHER} --files --json"
    printed = shlex.split(quiet[0].split("other logs: ", 1)[1])
    rc, out, _ = run(printed[1:], capsys)                                       # runnable as printed
    assert rc == 0 and TRAIN_REL in out


@pytest.mark.parametrize(("args", "clash"), [
    (["--files", "--grep", "x"], "--files does not combine with --grep"),
    (["--files", "--file", "a.log"], "--files does not combine with --file"),
    (["--files", "--path-only"], "--files does not combine with --path-only"),
    (["--file", "a.log", "--stream", "err"], "--file does not combine with --stream"),
    (["--file", "a.log", "--path-only"], "--file does not combine with --path-only"),
    (["--file", ""], "file: a path under the job's workdir"),
    (["--file", "a\nb.log"], "file: a path under the job's workdir"),
    (["--file", "x" * 4097], "at most 4096 characters"),
])
def test_files_flag_clashes_are_usage_errors(env: Env, capsys: pytest.CaptureFixture[str], args: list[str],
                                            clash: str) -> None:
    rc, out, err = run(["--demo", "agent", "log", LAUNCHER, *args], capsys)
    assert rc == 2 and out == "" and clash in err


def test_mcp_read_log_lists_and_reads_other_files(env: Env) -> None:
    from runfleet.agent import mcp

    runner = mcp.command_runner(config_path=None, demo=True, demo_seed=commands.DEMO_SEED, clock=env.clock)
    listed = runner("log", {"id": LAUNCHER, "list_files": True})
    assert listed.exit_code == 0 and listed.data["files"][0]["path"] == TRAIN_REL
    read = runner("log", {"id": LAUNCHER, "file": TRAIN_REL, "lines": 3})
    assert read.exit_code == 0 and read.data["log"]["stream"] == "file" and len(read.data["log"]["lines"]) == 3


def test_mcp_list_files_of_an_ended_job_without_a_start(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 2: an ended job with no start time searched "the last 24 h" from now, inverted against its end for a job
    that ended 3 days ago (nothing listed, `--file` refused); the TUI searched the 24 h before its end. Both now use
    core.logs.find_window, and the MCP result shows that window."""
    from runfleet.agent import mcp, source
    from runfleet.core.logs import LogService

    end = NOW - timedelta(days=3)
    real_view, real_find = source.demo_view, LogService.find
    asked: list[tuple[Any, Any]] = []

    async def demo_view(*a: Any, **kw: Any) -> Any:              # k2:3371193 ended 3 days ago, start unknown
        fv = await real_view(*a, **kw)
        hosts = []
        for hv in fv.hosts:
            if hv.name == "k2" and hv.snapshot is not None:
                old = dataclasses.replace(hv.snapshot.jobs["3371193"], start_time=None, end_time=end)
                hv = dataclasses.replace(hv, snapshot=dataclasses.replace(
                    hv.snapshot, jobs={**hv.snapshot.jobs, "3371193": old}))
            hosts.append(hv)
        return dataclasses.replace(fv, hosts=tuple(hosts))

    async def find(self: Any, host: str, workdir: str, *, since: Any, until: Any = None) -> Any:
        asked.append((since, until))
        return await real_find(self, host, workdir, since=since, until=until)

    monkeypatch.setattr(source, "demo_view", demo_view)
    monkeypatch.setattr(LogService, "find", find)
    server = mcp.Server(mcp.command_runner(config_path=None, demo=True, demo_seed=commands.DEMO_SEED,
                                           clock=env.clock))
    server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "orb"}}})
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "read_log", "arguments": {"id": "k2:3371193", "list_files": True}}})
    assert reply is not None
    res = reply["result"]
    assert res["isError"] is False and asked == [(end - timedelta(hours=24), end)]
    find_ = res["structuredContent"]["find"]
    assert find_["since"] == (end - timedelta(hours=24, seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert find_["until"] == (end + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert "(job end - 24 h, start unknown) until " in res["content"][0]["text"]


# ====================================================================================================== SKILL.md
def _skill_command_lines() -> list[str]:
    text = SKILL_MD.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:sh|bash)?\n(.*?)```", text, flags=re.S)
    lines = [ln.split("  #", 1)[0].strip() for b in blocks for ln in b.splitlines()]
    return [ln for ln in lines if ln.startswith("runfleet ")]


_PLACEHOLDERS = {"HOST…": "k2", "VERB": "jobs", "<cursor|2h>": "2h", "HOST": "k2", "HOST:ID": "k2:1"}


def _argv(line: str) -> list[str]:
    out: list[str] = []
    for tok in shlex.split(line)[1:]:
        if tok.startswith("[") and tok.endswith("]"):
            tok = tok[1:-1]
        out.append(_PLACEHOLDERS.get(tok, tok))
    return out


def test_every_command_line_of_the_skill_parses() -> None:
    lines = _skill_command_lines()
    assert len(lines) >= 8 and any(ln.startswith("runfleet agent brief") for ln in lines)
    verbs = set()
    for line in lines:
        ns = cli.parse_args(_argv(line))
        assert ns.command in ("agent", "wait", "status"), line
        if ns.command == "agent":
            verbs.add(ns.agent_verb)
    assert verbs == set(commands.VERBS)


def test_skill_allowed_tools_cover_its_commands() -> None:
    head = SKILL_MD.read_text(encoding="utf-8").split("---")[1]
    allowed = re.search(r"allowed-tools: (.*)", head)
    assert allowed is not None
    prefixes = re.findall(r"Bash\((runfleet [a-z]+) \*\)", allowed.group(1))
    for line in _skill_command_lines():
        assert any(line.startswith(p + " ") for p in prefixes), line


# ====================================================================================================== skill lifecycle
AGENT_VARS = ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "PI_CODING_AGENT_DIR", "COPILOT_HOME", "OPENCODE_CONFIG_DIR")


@pytest.fixture
def sk(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    """`env` plus a skill environment of its own: no agent variable from the real environment, no program found on
    PATH (agents are detected by their dirs under the temp HOME only), Claude Code present (~/.claude)."""
    for name in AGENT_VARS:
        monkeypatch.delenv(name, raising=False)
    real = skill.Env.current
    monkeypatch.setattr(skill.Env, "current", classmethod(lambda cls: dataclasses.replace(real(), which=lambda _: None)))
    (env.home / ".claude").mkdir()
    cli.skill_bundle_sha256.cache_clear()
    return env


def _shared(env: Env) -> Path:
    return env.home / ".agents" / "skills" / "runfleet"


def _claude(env: Env) -> Path:
    return env.home / ".claude" / "skills" / "runfleet"


def _manifest() -> dict[str, Any]:
    data = skill.read_manifest(cli.paths.state_dir())
    assert data is not None
    return data


def _patch_manifest(**changes: Any) -> None:
    path = cli.paths.state_dir() / skill.MANIFEST_FILE
    path.write_text(json.dumps({**_manifest(), **changes}), encoding="utf-8")


def _description(env: Env) -> str:
    md = (_shared(env) / "SKILL.md").read_text(encoding="utf-8")
    return next(ln for ln in md.splitlines() if ln.startswith("description:"))


TWO_HOSTS = '[[host]]\nname = "k2"\nssh = "k2"\n\n[[host]]\nname = "lyra"\nssh = "lyra"\nenabled = false\n'


def test_skill_install_shared_copy_claude_link_and_manifest(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    sk.config(TWO_HOSTS)
    rc, out, err = run(["skill", "install"], capsys)
    assert (rc, err) == (0, "")
    assert _description(sk).startswith("description: k2 ") and "lyra" not in _description(sk)
    assert (_shared(sk) / skill.STAMP_FILE).is_file() and (_shared(sk) / "references" / "reading-output.md").is_file()
    assert skill._is_link(_claude(sk)) and os.path.samefile(_claude(sk), _shared(sk))    # Windows: a junction
    kind = "junction" if sys.platform == "win32" else "symlink"
    assert re.search(rf"Claude Code: \S+ → \S+ \({kind}, created\)", out), out
    assert "read there by Codex" in out and "Trigger words: k2" in out and "Permission rule: Bash(runfleet agent *)" in out
    m = _manifest()
    assert m["hosts"] == ["k2"] and m["config_path"] is None and m["auto_update"] is True
    assert m["config_mtime"] == sk.config_path.stat().st_mtime
    assert {t["agent"] for t in m["targets"]} == {"shared", "claude"}
    rc, _out, _err = run(["skill", "install"], capsys)            # ours: replaced without --force
    assert rc == 0


def test_skill_install_print_and_demo_write_nothing(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["skill", "install", "--print"], capsys)      # no config: no host names
    assert rc == 0 and out.startswith("---\nname: runfleet\n")
    rc, out, _ = run(["--demo", "skill", "install", "--print"], capsys)
    assert rc == 0 and "description: k2 vega" in out
    rc, out, _ = run(["--demo", "skill", "status"], capsys)        # reads only
    assert rc == 0 and "not installed" in out
    for argv in (["install"], ["update"], ["uninstall"], ["uninstall", "--dry-run"]):
        rc, _out, err = run(["--demo", "skill", *argv], capsys)
        assert rc == 2 and "--print" in err, argv
    assert not (sk.home / ".agents").exists() and not (sk.home / ".claude" / "skills").exists()
    assert not (cli.paths.state_dir() / skill.MANIFEST_FILE).exists()


def test_skill_install_moves_a_foreign_directory_only_with_force(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    foreign = _shared(sk)
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("mine\n", encoding="utf-8")
    rc, _out, err = run(["skill", "install"], capsys)
    assert rc == 1 and "--force" in err
    assert (foreign / "SKILL.md").read_text(encoding="utf-8") == "mine\n"
    rc, out, _err = run(["skill", "install", "--force"], capsys)
    assert rc == 0 and (foreign / skill.STAMP_FILE).is_file()
    saved = list((cli.paths.state_dir() / skill.BACKUP_DIR).rglob("SKILL.md"))
    assert [p.read_text(encoding="utf-8") for p in saved] == ["mine\n"] and "Saved" in out   # moved, never deleted


def test_skill_install_agent_choice(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _out, err = run(["skill", "install", "--agent", "claude,bogus"], capsys)
    assert rc == 2 and "--agent: unknown agent bogus" in err and not _shared(sk).exists()
    rc, out, _ = run(["skill", "install", "--agent", "Antigravity"], capsys)     # explicit only, case forgiven
    gravity = sk.home / ".gemini" / "antigravity-cli" / "skills" / "runfleet"
    assert rc == 0 and skill._is_link(gravity) and not _claude(sk).exists() and "Antigravity" in out
    assert _manifest()["agents"] == ["antigravity"]


def test_skill_install_records_an_explicit_config_and_update_reuses_it(sk: Env, tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    other = tmp_path / "other.toml"
    other.write_text('[[host]]\nname = "vega"\nssh = "vega"\n', encoding="utf-8")
    sk.config(TWO_HOSTS)                                        # the default config: not used
    rc, _out, _ = run(["-c", str(other), "skill", "install"], capsys)
    assert rc == 0 and _description(sk).startswith("description: vega ")
    assert _manifest()["config_path"] == str(other.resolve())
    other.write_text('[[host]]\nname = "vega"\nssh = "vega"\n[[host]]\nname = "lynx"\nssh = "lynx"\n',
                     encoding="utf-8")
    rc, out, _ = run(["skill", "update"], capsys)               # the recorded config, not the default one
    assert rc == 0 and _description(sk).startswith("description: vega lynx ") and "updated" in out
    rc, _out, err = run(["-c", str(tmp_path / "missing.toml"), "skill", "install"], capsys)
    assert rc == 2 and "no such config file" in err


def test_skill_update_if_installed_and_quiet(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["skill", "update", "--if-installed"], capsys) == (0, "", "")
    rc, out, _ = run(["skill", "update"], capsys)
    assert rc == 0 and "not installed" in out and "runfleet skill install" in out
    run(["skill", "install"], capsys)
    assert run(["skill", "update", "--quiet"], capsys) == (0, "", "")          # nothing changed: silent
    rc, out, _ = run(["skill", "update"], capsys)
    assert rc == 0 and "is up to date" in out
    sk.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    rc, out, _ = run(["skill", "update", "--quiet"], capsys)
    assert rc == 0 and len(out.splitlines()) == 1 and "runfleet skill updated" in out
    assert _description(sk).startswith("description: k2 ")


def test_skill_update_never_recreates_a_removed_shared_copy(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    import shutil

    run(["skill", "install"], capsys)
    shutil.rmtree(_shared(sk))
    rc, out, _ = run(["skill", "update", "--if-installed", "--quiet"], capsys)
    assert rc == 0 and "was removed, not recreated" in out and "runfleet skill install" in out
    assert not _shared(sk).exists()


def test_skill_uninstall(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = run(["skill", "uninstall"], capsys)
    assert rc == 0 and "not installed; nothing to remove" in out
    run(["skill", "install"], capsys)
    (_shared(sk) / "SKILL.md").write_text("edited\n", encoding="utf-8")
    rc, out, _ = run(["skill", "uninstall", "--dry-run"], capsys)
    assert rc == 0 and "Would remove" in out and _shared(sk).is_dir() and skill._is_link(_claude(sk))
    rc, out, _ = run(["skill", "uninstall"], capsys)
    assert rc == 0 and not _shared(sk).exists() and not os.path.lexists(_claude(sk))
    assert not (cli.paths.state_dir() / skill.MANIFEST_FILE).exists()
    assert "claude mcp remove runfleet -s user" in out and "[mcp_servers.runfleet]" in out
    assert "Bash(runfleet agent *)" in out
    saved = list((cli.paths.state_dir() / skill.BACKUP_DIR).rglob("SKILL.md"))
    assert [p.read_text(encoding="utf-8") for p in saved] == ["edited\n"]


def test_skill_status_text_and_json(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    rc, out, _ = run(["skill", "status"], capsys)
    assert rc == 0 and "up to date" in out and "Claude Code: link " in out
    rc, doc = run_json(["skill", "status"], capsys)
    assert rc == 0 and doc["installed"] is True and doc["up_to_date"] is True and doc["auto_update"] is True
    claude = next(a for a in doc["agents"] if a["id"] == "claude")
    assert claude["covered_by"] == "link" and claude["state"] == "ok"


# ---------------------------------------------------------------- automatic refresh (cli.skill_auto_refresh)
def test_auto_refresh_after_an_upgrade_never_prints(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1")
    (_shared(sk) / "SKILL.md").write_text("old text\n", encoding="utf-8")       # ours to replace (backed up)
    rc, out, err = run(["status", "--short"], capsys)
    assert (rc, out, err) == (1, "no data\n", "")                             # the command's own output only
    assert _manifest()["version"] == runfleet.__version__
    assert (_shared(sk) / "SKILL.md").read_text(encoding="utf-8").startswith("---\nname: runfleet")


def test_auto_refresh_never_downgrades_and_respects_its_switch(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(version="99.0.0")                                          # a newer RunFleet installed it
    run(["status", "--short"], capsys)
    assert _manifest()["version"] == "99.0.0"
    run(["skill", "install", "--no-auto-update"], capsys)
    _patch_manifest(version="0.0.1")
    run(["status", "--short"], capsys)
    assert _manifest()["version"] == "0.0.1"


def test_auto_refresh_follows_host_changes_of_the_config(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    sk.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    run(["skill", "install"], capsys)
    stamp = (_shared(sk) / "SKILL.md").stat().st_mtime_ns
    sk.config('[[host]]\nname = "k2"\nssh = "k2"   # a comment: same hosts\n')
    os.utime(sk.config_path, (1, 1_000_000))
    run(["status", "--short"], capsys)
    assert _manifest()["config_mtime"] == 1_000_000 and (_shared(sk) / "SKILL.md").stat().st_mtime_ns == stamp
    sk.config('[[host]]\nname = "k2"\nssh = "k2"\n[[host]]\nname = "vega"\nssh = "vega"\n')
    run(["status", "--short"], capsys)
    assert _description(sk).startswith("description: k2 vega ") and _manifest()["hosts"] == ["k2", "vega"]


def test_auto_refresh_skips_skill_commands_and_demo(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1")
    for argv in (["skill", "status"], ["--demo", "agent", "brief"], ["--demo", "status", "--short"]):
        run(argv, capsys)
        assert _manifest()["version"] == "0.0.1", argv
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert _manifest()["version"] == "0.0.1"


def test_auto_refresh_runs_before_mcp_serves(sk: Env, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1")
    seen: list[str] = []
    monkeypatch.setitem(cli.COMMANDS, "mcp", lambda args, clock, out: seen.append(_manifest()["version"]) or 0)
    assert run(["mcp"], capsys) == (0, "", "")
    assert seen == [runfleet.__version__]


def test_auto_refresh_errors_never_reach_the_command(sk: Env, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1")

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(skill, "auto_refresh", boom)
    assert run(["status", "--short"], capsys) == (1, "no data\n", "")
    (cli.paths.state_dir() / skill.MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    assert run(["status", "--short"], capsys) == (1, "no data\n", "")


def test_auto_refresh_fast_path_is_cheap_and_does_not_import_the_skill_module(
        sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """Nothing to do: no import of runfleet.agent.skill, a JSON read, a stat and a cached bundle hash. Target < 5 ms
    per command; the bounds here are generous for slow CI machines."""
    import time

    run(["skill", "install"], capsys)
    assert cli.SKILL_MANIFEST == skill.MANIFEST_FILE
    assert cli.skill_bundle_sha256() == skill.bundle_sha256()               # the import-free copy matches
    args = cli.parse_args(["status", "--short"])
    clock = ManualClock(NOW)
    cli.skill_bundle_sha256.cache_clear()
    started = time.perf_counter()
    cli.skill_auto_refresh(args, clock)                                      # cold: hashes the bundle once
    cold = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(50):
        cli.skill_auto_refresh(args, clock)
    warm = (time.perf_counter() - started) / 50
    assert cold < 0.05 and warm < 0.005, (cold, warm)
    assert not _hook_imports_skill(sk)
    assert skill.canonical_dir(skill.Env.current()) == Path.home() / ".agents" / "skills" / "runfleet"


def _hook_imports_skill(env: Env) -> bool:
    """Run the hook in a fresh process: did it import runfleet.agent.skill?"""
    code = ("import sys\nfrom runfleet import cli\nfrom runfleet.core.clock import RealClock\n"
            "cli.skill_auto_refresh(cli.parse_args(['status', '--short']), RealClock())\n"
            "print('runfleet.agent.skill' in sys.modules)\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False,
                          env={**os.environ, "HOME": str(env.home), "USERPROFILE": str(env.home)})
    assert done.returncode == 0, done.stderr
    return done.stdout.strip() == "True"


def test_auto_refresh_does_no_repeated_work_for_a_newer_install_or_a_removed_copy(
        sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """Codex: neither case can ever refresh, so neither may import the skill module on every command."""
    import shutil

    run(["skill", "install"], capsys)
    _patch_manifest(version="99.0.0")
    assert not _hook_imports_skill(sk)
    _patch_manifest(version="0.0.1")
    assert _hook_imports_skill(sk)                          # an upgrade: refreshed (and then current)
    assert _manifest()["version"] == runfleet.__version__ and not _hook_imports_skill(sk)
    shutil.rmtree(_shared(sk))
    _patch_manifest(version="0.0.1")
    assert not _hook_imports_skill(sk) and not _shared(sk).exists()


def test_a_missing_recorded_config_keeps_the_host_names(sk: Env, tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Codex: a recorded -c config that is gone (an unmounted drive) must not erase the trigger words."""
    other = tmp_path / "other.toml"
    other.write_text('[[host]]\nname = "vega"\nssh = "vega"\n', encoding="utf-8")
    run(["-c", str(other), "skill", "install"], capsys)
    other.unlink()
    rc, _out, _ = run(["skill", "update"], capsys)
    assert rc == 0 and _manifest()["hosts"] == ["vega"] and _description(sk).startswith("description: vega ")
    run(["status", "--short"], capsys)                       # its mtime changed (gone): the hook keeps them too
    assert _manifest()["hosts"] == ["vega"] and _description(sk).startswith("description: vega ")


def test_skill_update_if_installed_respects_no_auto_update(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """F2: installers and `runfleet update` run `skill update --if-installed`: off means off."""
    run(["skill", "install", "--no-auto-update"], capsys)
    _patch_manifest(version="0.0.1")
    rc, out, _ = run(["skill", "update", "--if-installed", "--quiet"], capsys)
    assert rc == 0 and out == "runfleet skill: automatic refresh is off; run `runfleet skill update`\n"
    assert _manifest()["version"] == "0.0.1"
    rc, out, _ = run(["skill", "update", "--quiet"], capsys)
    assert rc == 0 and "runfleet skill updated" in out and _manifest()["version"] == runfleet.__version__


def test_skill_update_links_agents_found_later_only_when_explicit(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """E3: Claude Code installed after an auto install: status names the fix; `skill update` links it, the
    installers' `--if-installed` and the automatic refresh do not."""
    import shutil

    shutil.rmtree(sk.home / ".claude")
    run(["skill", "install"], capsys)
    (sk.home / ".claude").mkdir()
    _, out, _ = run(["skill", "status"], capsys)
    assert "Claude Code: not covered (missing); `runfleet skill install --agent claude` adds it" in out
    run(["skill", "update", "--if-installed"], capsys)
    _patch_manifest(version="0.0.1")
    run(["status", "--short"], capsys)                               # automatic refresh: refreshed, no new link
    assert _manifest()["version"] == runfleet.__version__ and not os.path.lexists(_claude(sk))
    rc, out, _ = run(["skill", "update"], capsys)
    assert rc == 0 and skill._is_link(_claude(sk)) and "found (or moved) since the install: Claude Code" in out


def test_skill_update_reads_the_config_recorded_at_lock_time(sk: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    """F5: the recorded config is read under the lock (another `skill install -c` may have changed it)."""
    sk.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    run(["skill", "install"], capsys)
    other = tmp_path / "other.toml"
    other.write_text('[[host]]\nname = "vega"\nssh = "vega"\n', encoding="utf-8")
    real = skill._locked

    @contextlib.contextmanager
    def racing(state_dir: Path, *, wait: float) -> Iterator[bool]:
        _patch_manifest(config_path=str(other))
        with real(state_dir, wait=wait) as held:
            yield held
    monkeypatch.setattr(skill, "_locked", racing)
    rc, _out, _ = run(["skill", "update"], capsys)
    assert rc == 0 and _manifest()["hosts"] == ["vega"] and _manifest()["config_path"] == str(other)


def test_auto_refresh_logs_one_line_also_with_an_invalid_config(sk: Env, capsys: pytest.CaptureFixture[str],
                                                                caplog: pytest.LogCaptureFixture) -> None:
    sk.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1")
    sk.config("[[host]\nthis is not toml")
    caplog.clear()
    with caplog.at_level("DEBUG", logger="runfleet.skill"):
        run(["status", "--short"], capsys)
    lines = [r.getMessage() for r in caplog.records if r.name == "runfleet.skill"]
    assert len(lines) == 1 and "refreshed" in lines[0] and "config not loaded" in lines[0], lines
    assert _manifest()["version"] == runfleet.__version__ and _manifest()["hosts"] == ["k2"]


def test_skill_uninstall_leaves_no_lock_file(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    run(["skill", "uninstall"], capsys)
    state = cli.paths.state_dir()
    assert not (state / skill.LOCK_FILE).exists() and not (state / skill.MANIFEST_FILE).exists()
    assert not (sk.home / ".agents").exists() and not (sk.home / ".claude" / "skills").exists()


def test_fast_path_identity_and_backoff_constants_match_the_skill_module() -> None:
    assert cli.skill_install_identity() == skill.install_identity()
    assert cli.SKILL_BACKOFF == skill.BACKOFF_SECONDS and cli.SKILL_SCHEMA == skill.SCHEMA
    for v in ("0.8.0", "0.8", "1.2.3rc1", "0.8.0.dev3", "0.8.0-post2", "0.8.0+g12ab", "99.0.0rc1", "x", "", " 0.8.1 ",
              "0.8.0.dev", "1.2.3.4"):
        assert cli.skill_version_key(v) == skill._version_key(v), v


@pytest.mark.parametrize("case", [
    {}, {"version": "0.0.1"}, {"version": "99.0.0"}, {"auto_update": False}, {"version": "0.0.1", "auto_update": False},
    {"config_mtime": 1.0}, {"bundle_sha256": "0" * 64}, {"bundle_sha256": "0" * 64, "writer": "/other/runfleet"},
    {"bundle_sha256": skill.RETRY, "writer": "/other/runfleet"}, {"bundle_sha256": skill.RETRY},
    {"bundle_sha256": "0" * 64, "writer": None}, {"config_mtime": 1.0, "writer": "/other/runfleet"},
    {"version": "0.0.1", "last_failure_at": "recent"}, {"version": "0.0.1", "last_failure_at": "old"},
    {"bundle_sha256": "0" * 64, "last_failure_at": "recent"}, {"config_mtime": 1.0, "last_failure_at": "future"},
    {"version": "0.0.1", "removed": True}, {"version": "99.0.0rc1"}, {"version": "garbage"},
    {"version": runfleet.__version__ + "+local"}, {"version": runfleet.__version__ + "+local", "config_mtime": 1.0},
    {"version": "0.0.1", "schema": 2}, {"version": "0.0.1rc1"},
])
def test_fast_path_agrees_with_refresh_reason(sk: Env, case: dict[str, Any],
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """F6 / E1: cli.skill_refresh_due (no import of the skill module) and skill.refresh_reason decide alike."""
    import shutil

    run(["skill", "install"], capsys)
    now = NOW.timestamp()
    changes = dict(case)
    removed = changes.pop("removed", False)
    at = {"recent": now - 60, "old": now - skill.BACKOFF_SECONDS - 1, "future": now + 600}
    if "last_failure_at" in changes:
        changes["last_failure_at"] = at[changes["last_failure_at"]]
    base = _manifest()
    data = {**base, **changes}
    if changes.get("writer", "") is None:
        data.pop("writer")
    (cli.paths.state_dir() / skill.MANIFEST_FILE).write_text(json.dumps(data), encoding="utf-8")
    if removed:
        shutil.rmtree(_shared(sk))
    mtime = base.get("config_mtime") if "config_mtime" not in case else 2.0
    env = skill.Env.current()
    due = cli.skill_refresh_due(data, config_mtime=mtime, now=now)
    reason = skill.refresh_reason(env, state_dir=cli.paths.state_dir(), config_mtime=mtime,
                                  config_path=data.get("config_path"), now=now)
    assert due == (reason is not None), (case, reason)


def test_fast_path_skips_during_backoff_without_importing_the_skill_module(
        sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    import time

    run(["skill", "install"], capsys)
    _patch_manifest(version="0.0.1", last_failure_at=time.time() - 60)
    assert not _hook_imports_skill(sk) and _manifest()["version"] == "0.0.1"
    _patch_manifest(last_failure_at=time.time() - skill.BACKOFF_SECONDS - 5)
    assert _hook_imports_skill(sk) and _manifest()["version"] == runfleet.__version__


def test_fast_path_skips_another_installs_bundle_without_importing(sk: Env, capsys: pytest.CaptureFixture[str]) -> None:
    run(["skill", "install"], capsys)
    _patch_manifest(bundle_sha256="0" * 64, writer="/some/other/install/runfleet")
    assert not _hook_imports_skill(sk) and _manifest()["bundle_sha256"] == "0" * 64
    _patch_manifest(writer=cli.skill_install_identity())
    assert _hook_imports_skill(sk) and _manifest()["bundle_sha256"] == skill.bundle_sha256()


# ====================================================================================================== real config
@pytest.fixture
def no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any process (ssh or not) or any Cache() (it creates and repairs the database) fails the test."""
    def forbidden(*a: Any, **kw: Any) -> Any:
        raise AssertionError("runfleet agent (cache path) must not start a process or construct Cache")
    monkeypatch.setattr(subprocess.Popen, "__init__", forbidden)
    monkeypatch.setattr("runfleet.agent.source.Cache", forbidden)
    for name in ("build_service", "run_command", "open_cache", "run_tui"):
        monkeypatch.setattr(cli, name, forbidden)


@pytest.mark.parametrize("args", [["brief"], ["jobs"], ["gpus"], ["host", "k2"], ["events", "--since", "1d"],
                                  ["job", "k2:1"], ["log", "k2:1"]])
def test_real_config_without_a_cache_is_no_data(env: Env, capsys: pytest.CaptureFixture[str], no_processes: None,
                                                args: list[str]) -> None:
    env.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    rc, out, err = run(["agent", *args], capsys)
    assert (rc, err) == (1, "")
    assert "no cache yet" in out
    rc, doc = run_json(["agent", *args], capsys)
    assert rc == 1 and doc["errors"][-1]["code"] == "no_data" and "no cache yet" in doc["errors"][-1]["message"]
    assert not env.cache_path.exists() and not (env.tmp / "mux").exists()


def test_no_config_at_all_is_no_data(env: Env, capsys: pytest.CaptureFixture[str], no_processes: None) -> None:
    rc, out, _err = run(["agent", "brief"], capsys)
    assert rc == 1 and "no hosts configured" in out and "hosts import" in out


def test_a_named_config_that_does_not_exist_is_a_usage_error(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, err = run(["-c", str(env.tmp / "nope.toml"), "agent", "brief"], capsys)
    assert rc == 2 and out == "" and "no such config file" in err


def test_real_cache_is_read_without_ssh(env: Env, capsys: pytest.CaptureFixture[str], no_processes: None) -> None:
    env.config('[[host]]\nname = "k2"\nssh = "k2"\n\n[[host]]\nname = "vega"\nssh = "vega"\n')
    c = Cache(env.cache_path)
    jobs = [job("k2", "101"), job("k2", "102", JobState.PENDING)]
    c.put_snapshot("k2", HostSnapshot("k2", NOW - timedelta(minutes=1), None, {j.key.id: j for j in jobs},
                                      (), (), (), (), None))
    c.put_status("k2", HostStatus(name="k2", state=ConnState.OK), NOW - timedelta(hours=1))
    c.flush_sync()
    rc, out, _ = run(["agent", "brief", "k2"], capsys)          # no health row (older RunFleet): age approximate
    assert rc == 4 and "mine: R 1 · PD 1" in out and "stale" in out
    ok = HostStatus(name="k2", state=ConnState.OK, sections={
        "slurm.squeue": SectionStatus("slurm.squeue", "ok", True, 0, 50, NOW - timedelta(minutes=1))})
    c.put_health("k2", ok, dest_hash=dest_hash("k2"), remote_user=None, at=NOW - timedelta(minutes=1))
    c.flush_sync()
    c._db.close()
    rc, out, _ = run(["agent", "brief", "k2"], capsys)
    assert rc == 0 and "mine: R 1 · PD 1" in out and "(cache)" in out.splitlines()[0]
    rc, doc = run_json(["agent", "jobs"], capsys)               # vega has no data: partial
    assert rc == 4 and {j["id"] for j in doc["jobs"]} == {"k2:101", "k2:102"}
    rc, doc = run_json(["agent", "job", "k2:101"], capsys)
    assert rc == 0 and doc["jobs"][0]["state"] == "R"
    rc, doc = run_json(["agent", "log", "k2:101"], capsys)      # no log path known: no probe at all
    assert rc == 1 and doc["errors"][0]["code"] == "no_log_path"


def test_printed_commands_carry_the_named_config(env: Env, capsys: pytest.CaptureFixture[str],
                                                no_processes: None) -> None:
    """C5: with -c PATH, every command RunFleet prints repeats it (else it would read the default config)."""
    other = env.tmp / "lab config.toml"
    other.write_text('[[host]]\nname = "k2"\nssh = "k2"\n', encoding="utf-8")
    c = Cache(env.cache_path)
    jobs = [job("k2", str(n)) for n in range(101, 106)]
    c.put_snapshot("k2", HostSnapshot("k2", NOW - timedelta(minutes=1), None, {j.key.id: j for j in jobs},
                                      (), (), (), (), None))
    ok = HostStatus(name="k2", state=ConnState.OK, sections={
        "slurm.squeue": SectionStatus("slurm.squeue", "ok", True, 0, 50, NOW - timedelta(minutes=1))})
    c.put_health("k2", ok, dest_hash=dest_hash("k2"), remote_user=None, at=NOW - timedelta(minutes=1))
    c.flush_sync()
    c._db.close()
    # quoted for this OS's shells: POSIX shlex; Windows double quotes (cmd and PowerShell; the path has a space)
    quoted = f'"{other}"' if sys.platform == "win32" else shlex.quote(str(other))
    rc, doc = run_json(["-c", str(other), "agent", "jobs", "--limit", "2"], capsys)
    assert doc["truncated"]["next"].startswith(f"runfleet -c {quoted} agent jobs ")
    if sys.platform != "win32":
        assert shlex.split(doc["truncated"]["next"])[1:3] == ["-c", str(other)]   # runnable as printed
    rc, out, _ = run(["-c", str(other), "agent", "brief"], capsys)
    nxt = [ln for ln in out.splitlines() if "runfleet " in ln and not ln.startswith("# runfleet")]
    assert nxt and all(f"runfleet -c {quoted} " in ln for ln in nxt)
    env.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    rc, doc = run_json(["agent", "jobs", "--limit", "2"], capsys)                  # the default config: no -c
    assert doc["truncated"]["next"].startswith("runfleet agent jobs ")


def test_a_slow_cache_read_cannot_outlast_the_deadline(env: Env, capsys: pytest.CaptureFixture[str],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from runfleet.agent import source

    env.config('[[host]]\nname = "k2"\nssh = "k2"\n')

    def slow(*a: Any, **kw: Any) -> Any:
        time.sleep(3)
        raise AssertionError("abandoned")
    monkeypatch.setattr(source, "load_view", slow)
    monkeypatch.setattr(commands, "GRACE", 0.0)
    started = time.monotonic()
    rc, out, _ = run(["agent", "brief", "--timeout", "0.3"], capsys)
    assert time.monotonic() - started < 2.0
    assert rc == 1 and "deadline" in out


# ====================================================================================================== startup
def test_agent_paths_never_import_textual(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import runfleet.paths as p\n"
        f"base = Path({str(tmp_path)!r})\n"
        "p.log_file = lambda: base / 'runfleet.log'\n"
        "p.cache_file = lambda: base / 'cache.sqlite'\n"
        "p.config_file = lambda: base / 'config.toml'\n"
        "p.mux_dir = lambda: base / 'mux'\n"
        "from runfleet import cli\n"
        "for argv in (['--demo', 'agent', 'brief'], ['agent', 'jobs'], ['agent', 'help'],\n"
        "             ['--demo', 'skill', 'install', '--print']):\n"
        "    cli.main(argv)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'textual' or m.startswith('runfleet.ui'))\n"
        "print('BAD', bad)\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False,
                          env=env, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "BAD []"


def test_log_err_of_a_job_with_only_stdout_is_no_log_path(env: Env, capsys: pytest.CaptureFixture[str],
                                                         no_processes: None) -> None:
    """Codex-5: the dispatcher reads only the ref agent.view picks; `err` of unmerged stdout is no stderr file."""
    import dataclasses

    from runfleet.models import LogRef

    env.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    c = Cache(env.cache_path)
    j = dataclasses.replace(job("k2", "101"), logs=(LogRef("stdout", "/shared/lab/alice/orb/logs/orb_101.out"),))
    c.put_snapshot("k2", HostSnapshot("k2", NOW - timedelta(minutes=1), None, {j.key.id: j}, (), (), (), (), None))
    c.flush_sync()
    c._db.close()
    rc, doc = run_json(["agent", "log", "k2:101", "--stream", "err"], capsys)
    assert rc == 1 and doc["errors"][0]["code"] == "no_log_path"


def test_since_is_validated_against_the_injected_clock() -> None:
    """Codex-14: build_request's --since check uses the caller's clock, not the wall clock."""
    from datetime import UTC, datetime

    from runfleet.config import FleetConfig, HostConfig

    cfg = FleetConfig(hosts=(HostConfig("k2", "k2"),))
    early = datetime(1, 3, 1, tzinfo=UTC)
    with pytest.raises(commands.UsageError, match="too far back"):
        commands.build_request("events", {"since": "100d"}, cfg, now=early)
    assert commands.build_request("events", {"since": "100d"}, cfg, now=NOW).verb == "events"


async def test_dispatch_passes_the_clock_to_validation(env: Env) -> None:
    from datetime import UTC, datetime

    env.config('[[host]]\nname = "k2"\nssh = "k2"\n')
    with pytest.raises(commands.UsageError, match="too far back"):
        await commands.run_async("events", {"since": "100d"}, config_path=env.config_path,
                                 clock=ManualClock(datetime(1, 3, 1, tzinfo=UTC)))
