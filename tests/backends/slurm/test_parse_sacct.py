"""[WP2] sacct parsers against the real fixtures (history rows, steps, gpu_tres seed, state-filter gotcha)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runfleet.backends.slurm.commands import SACCT_FIELDS, SEP
from runfleet.backends.slurm.parse_sacct import field_names, gputres_seed, parse_sacct, parse_steps
from runfleet.models import JobState
from tests.backends.factories import TZ, USER, format_fields, to_sep
from tests.conftest import meta, sample


def history(host: str, name: str = "sacct_ext", sep: str = "|") -> dict[str, object]:
    fields = format_fields(meta(host, name)["cmd"])
    text = sample(host, name) if sep == "|" else to_sep(sample(host, name), sep)
    pr = parse_sacct(text, fields, host=host, tz=TZ[host], me=USER[host], sep=sep)
    assert not pr.bad_lines, pr.bad_lines[:2]
    return {j.key.id: j for j in pr.items}


def test_fixture_field_list_is_ours() -> None:
    assert format_fields(meta("k2", "sacct_ext")["cmd"]) == [*SACCT_FIELDS, "WorkDir"]


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_sacct_ext_common(host: str) -> None:
    by_id = history(host)
    assert by_id
    for j in by_id.values():
        assert j.source == "sacct" and j.job_id_raw.isdigit()
        if j.state is JobState.RUNNING:
            assert j.end_time is None                     # End "Unknown" while running
        if j.raw_state.startswith("CANCELLED by"):
            assert j.state is JobState.CANCELLED and j.raw["cancelled_by_uid"].isdigit()


def test_lyra_sacct_ext_details() -> None:
    by_id = history("lyra")
    ca = by_id["31301780"]
    assert ca.state is JobState.CANCELLED and ca.raw["cancelled_by_uid"] == "2100007"
    assert ca.nodes == () and ca.nodelist is None         # "None assigned"
    assert ca.start_time is None                          # Start "None"
    assert ca.alloc.gpus == 1 and ca.alloc.gpu_type == "rtx6000" and ca.alloc.gpu_type_source == "tres-req"
    f = by_id["31327454"]
    assert f.state is JobState.FAILED and f.exit_code == 2 and f.exit_signal == 0
    assert f.alloc.gpu_type == "rtx6000" and f.alloc.gpu_type_source == "tres-alloc"   # lower-case gres in sacct
    assert f.workdir == "/vol/alice07/orb" and f.nodes == ("lyra-gpu-r71-00",)
    assert f.elapsed == timedelta(minutes=2, seconds=30)
    assert f.end_time == datetime(2026, 9, 5, 2, 8, 22, tzinfo=UTC)   # 19:08:22 local


def test_k2_running_row() -> None:
    by_id = history("k2")
    r = by_id["2332863"]
    assert r.state is JobState.RUNNING and r.end_time is None and r.raw["End"] == "Unknown"
    assert r.alloc.gpu_type == "nvidia_a100-sxm4-40gb"
    assert r.time_limit == timedelta(days=3)


def test_vega_sacct_ext_reason() -> None:
    by_id = history("vega")
    assert by_id["723086"].reason == "QOSMaxJobsPerUserLimit"
    assert by_id["723085"].alloc.gpus == 0 and by_id["723085"].alloc.gpu_type is None


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_sacct_basic(host: str) -> None:
    by_id = history(host, "sacct_basic")
    assert by_id
    cancelled = [j for j in by_id.values() if j.state is JobState.CANCELLED]
    for j in cancelled:
        assert j.start_time is None or j.elapsed is not None


def test_unit_separator_and_bad_field_count() -> None:
    by_id = history("k2", "sacct_ext", sep=SEP)
    assert "2332863" in by_id
    fields = format_fields(meta("k2", "sacct_ext")["cmd"])
    line = sample("k2", "sacct_ext").splitlines()[0].replace("orb-swp--train--cifar-cosinewarm--s43--a02", "x|y")
    pr = parse_sacct(line, fields, host="k2", tz=TZ["k2"], sep="|")
    assert pr.items == [] and pr.bad_ids == {"2332863"}


def test_failed_job_steps_lyra() -> None:
    m = meta("lyra", "sacct_failed_job_steps")
    pr = parse_steps(sample("lyra", "sacct_failed_job_steps"), format_fields(m["cmd"]), tz=TZ["lyra"], sep="|")
    steps = {s.step_id: s for s in pr.items}
    assert steps["31354789"].state is JobState.PREEMPTED
    batch = steps["31354789.batch"]
    assert batch.state is JobState.FAILED and batch.exit_code == 143 and batch.exit_signal == 0
    assert batch.max_rss_bytes == 26028 * 1024
    assert steps["31354789.extern"].state is JobState.COMPLETED


@pytest.mark.parametrize("host", ["k2", "vega"])
def test_failed_job_steps_exit_1(host: str) -> None:
    m = meta(host, "sacct_failed_job_steps")
    pr = parse_steps(sample(host, "sacct_failed_job_steps"), format_fields(m["cmd"]), tz=TZ[host], sep="|")
    batch = next(s for s in pr.items if s.step_id.endswith(".batch"))
    assert batch.state is JobState.FAILED and batch.exit_code == 1


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
@pytest.mark.parametrize("name", ["sacct_steps", "sacct_running_steps", "sacct_completed_gpu_tres"])
def test_step_fixtures(host: str, name: str) -> None:
    m = meta(host, name)
    pr = parse_steps(sample(host, name), format_fields(m["cmd"]), tz=TZ[host], sep="|")
    assert pr.items and not pr.bad_lines


def test_completed_gpu_tres_steps() -> None:
    m = meta("k2", "sacct_completed_gpu_tres")
    pr = parse_steps(sample("k2", "sacct_completed_gpu_tres"), format_fields(m["cmd"]), tz=TZ["k2"], sep="|")
    batch = next(s for s in pr.items if s.step_id == "2333102.batch")
    assert batch.tres_usage_max["gres/gpuutil"] == "101"
    assert batch.max_rss_bytes == 12167296 * 1024


def test_gputres_seed() -> None:
    def seed(host: str) -> bool | None:
        # the detect section prints `JobID|TRESUsageInMax`; the fixtures carry extra columns, the rule is the same
        return gputres_seed(sample(host, "sacct_completed_gpu_tres"))

    assert seed("vega") is True
    assert seed("k2") is True
    assert seed("lyra") is False
    assert gputres_seed("") is None


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_state_filter_needs_endtime(host: str) -> None:
    # field probes: `-s` without `-E` returns nothing; with `-E now` it does. Our window command passes -E now.
    text = sample(host, "sacct_state_filter_test")
    blocks = text.split("-s ")
    assert blocks[1].strip().endswith("without -E:")
    assert "|COMPLETED" in blocks[2]
    assert sample(host, "sacct_failed_states") == ""      # the probe that forgot -E
    assert "-E" not in meta(host, "sacct_failed_states")["cmd"]


def test_field_names() -> None:
    assert field_names(["JobID", "TRESUsageInMax%400"]) == ["JobID", "TRESUsageInMax"]


# ---------------------------------------------------------------- M3 (field validation, 2026-09-07)
M3_FIELDS = [*SACCT_FIELDS, "StdOut", "StdErr", "WorkDir"]


@pytest.mark.parametrize(("host", "jid", "path", "merged"), [
    ("k2", "2329557", "/shared/pilab/bob/rlsim/sim/logs/eval_b_r2_2329557.out", True),   # StdErr empty
    ("lyra", "31303059", None, True),                                                          # StdErr == StdOut
    ("vega", "723089", "/shared/pilab/pilab/orb/logs/orb-swp--train--warmup--s46--a01_723089.out", True),
])
def test_sacct_stdout_patterns_are_expanded(host: str, jid: str, path: str | None, merged: bool) -> None:
    """M3 F2: sacct StdOut/StdErr hold the UNEXPANDED --output pattern on 25.05/25.11/26.05; they were dropped and
    every ended job fell back to {workdir}/slurm-<id>.out."""
    pr = parse_sacct(sample(host, "sacct_window_m3"), M3_FIELDS, host=host, tz=TZ[host], me=USER[host], sep=SEP)
    assert not pr.bad_lines
    job = {j.key.id: j for j in pr.items}[jid]
    out = job.logs[0]
    assert out.source == "sacct" and out.path is not None and "%" not in out.path and out.merged is merged
    assert out.path == (path or f"/vol/alice07/orb/logs/{job.name}_{jid}.out")


def test_het_job_components_parse_and_key_by_raw_id() -> None:
    fields = ["JobID", "JobIDRaw", "JobName", "User", "State"]
    text = "600+0|600|het|u|COMPLETED\n600+1|601|het|u|COMPLETED\n"
    pr = parse_sacct(text, fields, host="k2", tz=timedelta(0), sep="|")
    assert not pr.bad_lines and [j.key.id for j in pr.items] == ["600", "601"]        # squeue keys them 600, 601
    steps = parse_steps("600+0.batch|batch|COMPLETED|0:0\n", ["JobID", "JobName", "State", "ExitCode"],
                        tz=timedelta(0), sep="|")
    assert [s.step_id for s in steps.items] == ["600+0.batch"]


def test_pipe_fallback_with_a_pipe_in_workdir_keeps_the_row() -> None:
    """Review BACK-27: a `|` in the trailing WorkDir dropped the row with the `|` fallback separator."""
    fields = format_fields(meta("k2", "sacct_ext")["cmd"])
    assert fields[-1] == "WorkDir"
    line = sample("k2", "sacct_ext").splitlines()[0]
    pr = parse_sacct(line.rsplit("|", 1)[0] + "|/w/a|b", fields, host="k2", tz=TZ["k2"], sep="|")
    (job,) = pr.items
    assert job.key.id == "2332863" and job.workdir is None


def test_pipe_fallback_with_a_pipe_in_jobname_stays_bad() -> None:
    """A `|` in JobName shifts User/State/… : the trailing-path recovery must not accept the shifted row."""
    fields = format_fields(meta("k2", "sacct_ext")["cmd"])
    line = sample("k2", "sacct_ext").splitlines()[0]
    jid, raw, name, rest = line.split("|", 3)
    rest = rest.replace("|gpu-preempt|", "|GPU|", 1)      # shifted into State it would pass as a state word
    pr = parse_sacct(f"{jid}|{raw}|{name}|run|{rest}", fields, host="k2", tz=TZ["k2"], sep="|", me="alice0001")
    assert not pr.items and pr.bad_lines and pr.bad_ids == {"2332863"}
