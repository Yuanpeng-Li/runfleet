"""[WP2] `scontrol show job -d -o` tokenizer and JobInfo extraction against the real fixtures."""

from __future__ import annotations

import pytest

from runfleet.backends.slurm.parse_scontrol import job_info, parse_node_record, parse_scontrol, tokenize
from tests.backends.factories import fixture_result
from tests.conftest import sample


def one(host: str, name: str) -> object:
    blocks = parse_scontrol(sample(host, name))
    (block,) = blocks.values()
    (rec,) = block.records
    return rec


def test_vega_running_gres_idx_and_merged() -> None:
    rec = one("vega", "scontrol_show_job_running_d")
    assert rec.job_id == "723954"
    (node,) = rec.per_node
    assert node.node == "n-38-01" and node.gpu_indices == (0, 1) and node.cpu_ids == "24-39"
    assert node.mem_bytes == 262144 << 20
    assert node.gres == "gpu:nvidia_h200_nvl:2(IDX:0-1)"
    assert rec.gpu_type == "nvidia_h200_nvl"
    assert rec.merged and rec.stderr is None              # vega prints an empty StdErr=
    assert rec.stdout == "/shared/pilab/alix/lmft/logs/lmft-x07-sampler-s1_723954.out"


def test_vega_submitline_with_spaces_then_workdir() -> None:
    rec = one("vega", "scontrol_show_job_d")
    sl = rec.fields["SubmitLine"]
    assert sl.startswith("sbatch --job-name=lmft-x07-trainer-s0 ")
    assert " --exclude=n-38-06 " in sl
    assert sl.endswith("jobs/train.sbatch --config configs/exp_b.yaml seed=0 lr=1e-5 steps=100")
    assert rec.workdir == "/shared/pilab/alix/lmft"
    assert rec.per_node == ()                             # pending: no per-node block
    assert rec.fields["NodeList"] == ""


def test_lyra_submitline_numeric_args_and_a30() -> None:
    rec = one("lyra", "scontrol_show_job_running_d")
    assert rec.fields["SubmitLine"].endswith("0.003 0.9 8")
    assert rec.workdir == "/home/victor/vision"
    assert rec.gpu_type == "a30"                          # JOB_GRES=gpu:A30:1, lower-cased
    assert rec.per_node[0].gpu_indices == (0,)
    assert rec.merged


def test_k2_running_stderr_equal_stdout_is_merged() -> None:
    rec = one("k2", "scontrol_show_job_running_d")
    assert rec.merged
    assert rec.per_node[0].gpu_indices == (2,) and rec.per_node[0].cpu_ids == "64-71"
    assert rec.gpu_type == "nvidia_a100-sxm4-40gb"
    assert rec.fields["ExcNodeList"] == "n-31-[04-05],n-33-[04-06],n-36-[01-03],n-37-02"


def test_k2_pending_has_no_per_node_block() -> None:
    rec = one("k2", "scontrol_show_job_d")
    assert rec.per_node == () and rec.gpu_type is None
    assert rec.fields["JobState"] == "PENDING" and rec.fields["SchedNodeList"] == "n-32-03"
    assert rec.stdout == "/shared/pilab/pilab/rl-benchmarks/logs/smoke-test_2334564.out" and rec.merged


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
@pytest.mark.parametrize("name", ["scontrol_show_job", "scontrol_show_job_d", "scontrol_show_job_running_d"])
def test_every_fixture_tokenizes(host: str, name: str) -> None:
    rec = one(host, name)
    assert rec.fields["JobId"].isdigit() and rec.fields["UserId"]
    assert rec.fields["Command"].startswith("/")


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
@pytest.mark.parametrize("pair", [("scontrol_show_job_multiline", "scontrol_show_job_d"),
                                  ("scontrol_show_job_running_multiline", "scontrol_show_job_running_d")])
def test_multiline_form_matches_one_line(host: str, pair: tuple[str, str]) -> None:
    a, b = one(host, pair[0]), one(host, pair[1])
    assert a.fields == b.fields and a.per_node == b.per_node and a.stdout == b.stdout


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_finished_job_left_controller(host: str) -> None:
    # the loop runs `scontrol … 2>&1`: stderr folds into stdout under the job's marker
    r = fixture_result("slurm.scontrol", host, "scontrol_show_job_finished", fold_stderr=True)
    blocks = parse_scontrol(f"@@job 2330539\n{r.stdout}")
    assert blocks["2330539"].left_controller
    assert blocks["2330539"].record_for(None) is None


def test_loop_output_and_array_records() -> None:
    running = sample("vega", "scontrol_show_job_running_d").strip()
    summary = ("JobId=724196 ArrayJobId=724196 ArrayTaskId=21-32%1 ArrayTaskThrottle=1 JobName=mcmc UserId=u(1) "
               "JobState=PENDING Reason=Resources WorkDir=/w StdErr= StdOut=/w/a_%a.out")
    element = ("JobId=724216 ArrayJobId=724196 ArrayTaskId=10 JobName=mcmc UserId=u(1) JobState=RUNNING "
               "Reason=None WorkDir=/w StdErr= StdOut=/w/a_10.out")
    text = f"@@job 723954\n{running}\n@@job 724196\n{summary}\n{element}\n@@job 1\nslurm_load_jobs error: x\n"
    blocks = parse_scontrol(text)
    assert blocks["723954"].record_for(None) is not None
    assert blocks["724196"].record_for("1-20%1") is None
    assert blocks["724196"].record_for("21-32%1") is not None
    assert blocks["724196"].record_for("[21-32%1]").array_task_id == "21-32%1"
    assert blocks["724196"].record_for("10").stdout == "/w/a_10.out"
    assert blocks["1"].errors == ["slurm_load_jobs error: x"] and not blocks["1"].left_controller


def test_tokenize_rules() -> None:
    pairs = tokenize("JobId=1 Comment=keep A=b going TRAIN.lr=3 WorkDir=/w x=1 StdOut=/o Reason=a B=c")
    d = dict(pairs)
    assert d["Comment"] == "keep A=b going TRAIN.lr=3"    # greedy: only known keys end it
    assert d["WorkDir"] == "/w x=1"
    assert d["Reason"] == "a B=c"
    multi = tokenize("JobId=2 Nodes=n-[01-02] CPU_IDs=0-3 Mem=100 GRES=gpu:a:1(IDX:0) Nodes=n-03 CPU_IDs=4 Mem=5 "
                     "GRES=gpu:a:2(IDX:N/A) MinCPUsNode=1")
    assert [k for k, _ in multi].count("Nodes") == 2
    info = job_info("JobId=2 Nodes=n-[01-02] CPU_IDs=0-3 Mem=100 GRES=gpu:a:1(IDX:0) Nodes=n-03 CPU_IDs=4 Mem=5 "
                    "GRES=gpu:a:2(IDX:N/A) MinCPUsNode=1")
    assert info is not None
    assert [n.node for n in info.per_node] == ["n-01", "n-02", "n-03"]
    assert info.per_node[2].gpu_indices == () and info.per_node[0].gpu_indices == (0,)


@pytest.mark.parametrize("host", ["k2", "vega", "lyra"])
def test_scontrol_show_node(host: str) -> None:
    rec = parse_node_record(sample(host, "scontrol_show_node"))
    assert rec["NodeName"] and rec["Gres"].startswith("gpu:")
    assert rec["OS"].startswith("Linux ") and rec["OS"].endswith("2026")   # value with spaces kept whole
    assert rec["RealMemory"].isdigit()
    if host == "vega":
        assert rec["State"] == "MIXED+PLANNED"


@pytest.mark.parametrize("inserted", ["GresEnforceBind=Yes", "OOMKillStep=1", "KillOnInvalidDependent=Yes",
                                      "SegmentSize=2", "CpuFreqMin=Low CpuFreqMax=High CpuFreqGov=OnDemand",
                                      "SpreadJob=Yes", "ExpeditedRequeue=Yes", "BrandNewKey26=x"])
def test_newer_slurm_keys_after_stdout_do_not_leak_into_the_path(inserted: str) -> None:
    """Review: keys newer Slurm prints after StdOut were absorbed into the (greedy) path value."""
    rec = sample("k2", "scontrol_show_job_running_d").replace(" TresPerNode=", f" {inserted} TresPerNode=", 1)
    info = job_info(rec.strip())
    assert info is not None
    assert info.stdout == "/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out"


def test_comment_keeps_key_value_text_but_stops_at_extra() -> None:
    pairs = dict(tokenize("JobId=1 Comment=lr=0.1 seed=3 Extra=foo StdIn=/dev/null"))
    assert pairs["Comment"] == "lr=0.1 seed=3" and pairs["Extra"] == "foo"


def test_per_node_gpu_indices_union_every_gres_type() -> None:
    """k2's mixed MIG node: a full GPU plus a MIG slice on one node gave only the first IDX group."""
    rec = sample("k2", "scontrol_show_job_running_d").replace(
        "GRES=gpu:nvidia_a100-sxm4-40gb:1(IDX:2)", "GRES=gpu:a100:1(IDX:0),gpu:a100_3g.20gb:1(IDX:1)", 1)
    info = job_info(rec.strip())
    assert info is not None and info.per_node[0].gpu_indices == (0, 1)


def test_non_gpu_gres_indices_are_not_gpu_indices() -> None:
    """Review BACK-19: `nic:1(IDX:0)` next to the GPU made the job hold GPUs 0 and 3."""
    rec = sample("k2", "scontrol_show_job_running_d").replace(
        "GRES=gpu:nvidia_a100-sxm4-40gb:1(IDX:2)", "GRES=gpu:h100:1(IDX:3),nic:1(IDX:0)", 1)
    info = job_info(rec.strip())
    assert info is not None and info.per_node[0].gpu_indices == (3,)


def test_gpu_type_with_a_comma_in_the_idx_list() -> None:
    """Review BACK-20: `gpu:h100:2(IDX:0,2)` was split inside the parentheses and lost its type."""
    from runfleet.backends.slurm.parse_scontrol import _gres_type
    assert _gres_type("gpu:h100:2(IDX:0,2)") == "h100"
    rec = sample("k2", "scontrol_show_job_running_d").replace(
        "GRES=gpu:nvidia_a100-sxm4-40gb:1(IDX:2)", "GRES=gpu:h100:2(IDX:0,2)", 1)
    info = job_info(rec.strip())
    assert info is not None and info.per_node[0].gpu_indices == (0, 2)


def test_repeated_path_key_from_a_comment_is_dropped() -> None:
    """Review BACK-28: `Comment=… StdOut=/wrong` before the real StdOut= spoofed the job's log path."""
    info = job_info("JobId=1 JobName=x Comment=please use StdOut=/wrong StdErr=/e StdIn=/dev/null StdOut=/real")
    assert info is not None and info.stdout is None and info.stderr == "/e"
