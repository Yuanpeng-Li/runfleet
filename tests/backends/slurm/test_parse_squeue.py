"""[WP2] squeue parsers against the real fixtures (§10.2 mandated assertions) + synthetic separator/epoch cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runfleet.backends.slurm.commands import SEP, SQUEUE_FIELDS, strip_throttle
from runfleet.backends.slurm.parse_squeue import display_id, parse_squeue, parse_squeue_legacy, resolve_gpu
from runfleet.models import JobState
from tests.backends.factories import TZ, USER, to_sep
from tests.conftest import sample

LEGACY_FIXTURE_SPEC = "%i|%j|%T|%M|%l|%D|%R|%b|%C|%m|%P|%V|%S|%Z|%o"


def jobs(host: str, name: str = "squeue_me_O", sep: str = "|") -> dict[str, object]:
    pr = parse_squeue(sample(host, name), host=host, tz=TZ[host], me=USER[host], sep=sep)
    assert not pr.bad_lines, pr.bad_lines
    return {j.key.id: j for j in pr.items}


def test_k2_squeue_me_O() -> None:
    pr = parse_squeue(sample("k2", "squeue_me_O"), host="k2", tz=TZ["k2"], me=USER["k2"], sep="|")
    assert not pr.bad_lines
    by_id = {j.key.id: j for j in pr.items}
    assert len(by_id) == 5

    j = by_id["2332861"]
    assert j.state is JobState.RUNNING and j.raw_state == "RUNNING"
    assert j.alloc.gpus == 1 and j.alloc.gpu_type == "nvidia_a100-sxm4-40gb"
    assert j.alloc.gpu_type_source == "tres-alloc"
    assert j.nodes == ("n-40-01",) and j.batch_host == "n-40-01"
    assert j.elapsed == timedelta(hours=4, minutes=7)
    assert j.time_limit == timedelta(days=3)
    stdout = next(r for r in j.logs if r.stream == "stdout")
    assert stdout.path == "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"
    assert stdout.merged                                  # k2 STDERR repeats the STDOUT pattern
    assert j.job_id_raw == "2332861" and j.is_mine
    assert j.alloc.cpus == 8 and j.alloc.mem_bytes == 48 << 30
    assert j.partition == "gpu-preempt" and j.qos == "gpu-priority" and j.account == "campus01"

    pd = by_id["2334564"]
    assert pd.state is JobState.PENDING and pd.reason == "Priority" and not pd.held
    assert pd.est_start and pd.start_time == datetime(2026, 9, 7, 11, 31, 41, tzinfo=UTC)   # 06:31 local (−05:00)
    assert pd.nodes == () and pd.batch_host is None
    assert pd.alloc.gpu_type == "nvidia_geforce_rtx_3090"

    h200 = by_id["2333760"]
    assert h200.state is JobState.RUNNING
    assert h200.alloc.gpus == 2 and h200.alloc.gpu_type == "nvidia_h200_nvl"
    assert h200.alloc.gpu_type_source == "tres-req"        # tres-alloc is untyped, tres-per-node is not
    err = next(r for r in h200.logs if r.stream == "stderr")
    assert err.merged and err.path == next(r for r in h200.logs if r.stream == "stdout").path   # STDERR (null)


def test_vega_squeue_me_O() -> None:
    by_id = jobs("vega")
    j = by_id["724479"]
    assert j.state is JobState.PENDING and j.reason == "QOSMaxJobsPerUserLimit"
    assert j.start_time is None and not j.est_start
    assert j.alloc.gpus == 2 and j.alloc.gpu_type == "nvidia_h200_nvl"
    assert by_id["723954"].nodes == ("n-38-01",)


def test_lyra_array_summary_and_relative_paths() -> None:
    by_id = jobs("lyra", "squeue_parts_O")
    j = by_id["31270818_[1-24%1]"]
    assert j.job_id_raw == "31270818" and j.array_job_id == "31270818" and j.array_task == "1-24%1"
    assert strip_throttle(j.key.id) == "31270818_[1-24]"
    assert j.command == "./train.sh"
    out = next(r for r in j.logs if r.stream == "stdout")
    assert out.path is None                               # %a of a pending summary cannot be expanded
    assert out.pattern == "/scratch/rita/md/run1/slurm-%A_%a.out"
    plain = by_id["31270819"]
    assert next(r for r in plain.logs if r.stream == "stdout").path == \
        "/scratch/rita/md/run1/slurm-31270819.out"
    assert plain.reason == "Dependency" and not plain.is_mine


def test_lyra_squeue_me_O_is_empty() -> None:
    assert sample("lyra", "squeue_me_O") == ""
    pr = parse_squeue("", host="lyra", tz=TZ["lyra"], sep="|")
    assert pr.items == [] and pr.bad_lines == []


def test_vega_parts_array_tasks() -> None:
    by_id = jobs("vega", "squeue_parts_O")
    summary = by_id["724196_[21-32%1]"]
    assert summary.state is JobState.PENDING and summary.reason == "Resources"
    never = by_id["712655_[32-63%1]"]
    assert never.reason == "DependencyNeverSatisfied"
    # (null) STDOUT → Slurm's default pattern relative to WorkDir, unexpandable for a summary
    assert next(r for r in never.logs if r.stream == "stdout").pattern == "/home/ken/slurm-%A_%a.out"


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_parts_fixtures_parse_cleanly(host: str) -> None:
    pr = parse_squeue(sample(host, "squeue_parts_O"), host=host, tz=TZ[host], sep="|")
    assert pr.items and not pr.bad_lines
    for j in pr.items:
        assert j.key.id.startswith(j.job_id_raw) or j.array_job_id is not None


def test_squeue_parts_states_matches_rows() -> None:
    # squeue_parts_states counts states over the whole partitions (not just the 25 sampled rows): the sampled rows'
    # states must all be among the counted ones
    for host in ("k2", "vega", "lyra"):
        counted = {line.split()[1] for line in sample(host, "squeue_parts_states").splitlines() if line.strip()}
        parsed = {j.raw_state for j in parse_squeue(sample(host, "squeue_parts_O"), host=host, tz=TZ[host],
                                                    sep="|").items}
        assert parsed <= counted


def test_squeue_array_expand_ids_match() -> None:
    # `squeue -r -o '%i|%A|%a|%F|%K|%T'`: %i == %A for non-array jobs, %K = N/A
    for host in ("k2", "vega"):
        rows = [ln.split("|") for ln in sample(host, "squeue_array_expand").splitlines()]
        ids = set(jobs(host))
        assert {r[0] for r in rows} == ids
        assert all(r[0] == r[1] == r[3] and r[4] == "N/A" for r in rows)
    assert sample("lyra", "squeue_array_expand") == ""


def test_name_with_pipe_is_bad_id_with_pipe_sep_and_parses_with_us() -> None:
    line = sample("k2", "squeue_me_O").splitlines()[1]
    evil = line.replace("orb-swp--train--cifar-warmup--s43--a02", "a|b", 1)
    pr = parse_squeue(evil, host="k2", tz=TZ["k2"], sep="|")
    assert pr.items == [] and pr.bad_ids == {"2332861"}
    us_line = to_sep(line, SEP).replace("orb-swp--train--cifar-warmup--s43--a02", "a|b", 1)
    pr2 = parse_squeue(us_line, host="k2", tz=TZ["k2"], sep=SEP)
    assert [j.name for j in pr2.items] == ["a|b"]


@pytest.mark.parametrize("odd", ["a\u2028b", "a\x85b", "a\x0cb", "a\x0bb", "a\x1cb", "a\x1eb", "a\rb"])
def test_names_with_unicode_line_breaks_stay_one_row(odd: str) -> None:
    """Review: str.splitlines() broke rows at \\x0b, \\x0c, \\x1c-\\x1e, \\x85, U+2028/9 — a running job vanished into
    bad lines. Rows are split on "\\n" only, in every Slurm and ps parser."""
    line = to_sep(sample("k2", "squeue_me_O").splitlines()[1], SEP).replace(
        "orb-swp--train--cifar-warmup--s43--a02", odd, 1)
    pr = parse_squeue(line + "\r\n", host="k2", tz=TZ["k2"], sep=SEP)
    assert not pr.bad_lines and [j.name for j in pr.items] == [odd]


def test_command_with_separator_is_rejoined() -> None:
    line = to_sep(sample("k2", "squeue_me_O").splitlines()[1], SEP)
    line = line.replace("/shared/pilab/pilab/orb/jobs/train.sbatch", f"run.sh a{SEP}b", 1)
    (job,) = parse_squeue(line, host="k2", tz=TZ["k2"], sep=SEP).items
    assert job.command == f"run.sh a{SEP}b"


def test_epoch_timestamps() -> None:
    row = ["123", "123", "N/A", "train", "me", "RUNNING", "R", "None", "1:00", "2:00:00", "1:59:00", "1", "4", "8G",
           "p", "normal", "acct", "1788696459", "1788696557", "1788703757", "n1", "n1", "cpu=4,gres/gpu=1", "N/A",
           "N/A", "0", "/w/out-%j.log", "(null)", "/w", "/w/run.sh"]
    assert len(row) == len(SQUEUE_FIELDS)
    text = SEP.join(row) + SEP
    (job,) = parse_squeue(text, host="h", tz=TZ["k2"], sep=SEP).items
    assert job.submit_time == datetime.fromtimestamp(1788696459, tz=UTC)
    assert job.start_time is not None and job.start_time.tzinfo is not None
    assert job.end_time is None                           # EndTime of a running job is only the limit
    assert job.logs[0].path == "/w/out-123.log"


def test_bad_rows_ratio() -> None:
    good = sample("k2", "squeue_me_O").splitlines()
    pr = parse_squeue("\n".join([*good, "garbage", "2338403|x"]), host="k2", tz=TZ["k2"], sep="|")
    assert len(pr.items) == 5 and len(pr.bad_lines) == 2 and not pr.mostly_bad
    assert pr.bad_ids == {"2338403"}
    assert parse_squeue("x\ny\n4001|z", host="k2", tz=TZ["k2"], sep="|").mostly_bad


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
@pytest.mark.parametrize("name", ["squeue_me_o", "squeue_parts_o"])
def test_legacy_fixtures(host: str, name: str) -> None:
    pr = parse_squeue_legacy(sample(host, name), LEGACY_FIXTURE_SPEC, host=host, tz=TZ[host], me=USER[host])
    assert not pr.bad_lines
    o = {j.key.id: j for j in parse_squeue(sample(host, name.replace("_o", "_O")), host=host, tz=TZ[host],
                                           sep="|").items}
    for j in pr.items:
        if j.array_task and not j.array_task.isdigit():
            assert j.key.id in o                          # 724196_[21-32%1] both ways
        if j.key.id in o:
            assert j.state is o[j.key.id].state and j.name == o[j.key.id].name
            assert j.reason == o[j.key.id].reason or (j.reason is None and o[j.key.id].reason in (None, "None"))


def test_legacy_reason_or_nodelist() -> None:
    pr = parse_squeue_legacy(sample("k2", "squeue_me_o"), LEGACY_FIXTURE_SPEC, host="k2", tz=TZ["k2"])
    by_id = {j.key.id: j for j in pr.items}
    assert by_id["2334564"].reason == "Priority" and by_id["2334564"].nodes == ()
    assert by_id["2332861"].nodes == ("n-40-01",)
    assert by_id["2332861"].alloc.gpu_type == "nvidia_a100-sxm4-40gb"


def test_display_id_and_resolve_order() -> None:
    assert display_id("31270824", "31270818", "3") == "31270818_3"
    assert display_id("31270818", "31270818", "1-24%1") == "31270818_[1-24%1]"
    assert display_id("2332861", None, None) == "2332861"
    # (2) scontrol beats tres-per-node, (4) sinfo, (5) cache
    assert resolve_gpu({"gres/gpu": "1"}, (1, "x"), (None, None), 1, scontrol_type="A30") == (1, "a30", "scontrol")
    assert resolve_gpu({"gres/gpu": "1"}, (1, None), (None, None), 1,
                       node_types=[frozenset({"rtx6000"})]) == (1, "rtx6000", "sinfo")
    assert resolve_gpu({"gres/gpu": "1"}, (1, None), (None, None), 1,
                       node_types=[frozenset({"a", "b"})], cached_type="a") == (1, "a", "cache")
    assert resolve_gpu({}, (2, None), (None, None), 2) == (4, None, None)
    assert resolve_gpu({"cpu": "4"}, (None, None), (None, None), 1) == (0, None, None)


def test_pipe_fallback_does_not_shift_a_path_into_command() -> None:
    """Review BACK-27: with the `|` fallback, `WorkDir=/w/a|b` became WorkDir /w/a and Command `b|/w/run.sh`."""
    first = sample("k2", "squeue_me_O").splitlines()[1]
    line = first.replace("|/shared/pilab/pilab/orb|", "|/shared/pilab/pilab/a|b|", 1)
    assert line != first
    pr = parse_squeue(line, host="k2", tz=TZ["k2"], me=USER["k2"], sep="|")
    (job,) = pr.items
    assert job.workdir is None and job.command is None and job.state is JobState.RUNNING
    (ok,) = parse_squeue(first, host="k2", tz=TZ["k2"], me=USER["k2"], sep="|").items
    assert ok.workdir == "/shared/pilab/pilab/orb"
