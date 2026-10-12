"""[WP1] models.to_json / from_json round-trip every §3.3 type (cache, snapshot --json)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from runfleet import models as m

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


def _snapshot() -> m.HostSnapshot:
    caps = m.Capabilities(remote_user="usr", uid=1000, login_host="login1", tz_offset=timedelta(hours=-5),
                          clock_skew=timedelta(seconds=2), tools=frozenset({"squeue", "sinfo"}),
                          profile=("slurm", "procfs"), slurm_version="25.05.8", slurm_config={"MinJobAge": "300"},
                          gpu_tres=None, sacct_fields=frozenset({"StdOut"}),
                          associations=(("k2", "pilab", "", "normal"),))
    alloc = m.ResourceAlloc(nodes=1, cpus=8, mem_bytes=48 << 30, gpus=2, gpu_type="nvidia_h200_nvl",
                            tres={"gres/gpu": "2"}, per_node=(m.NodeAlloc("n-40-01", "64-71", None, (0, 1), None),),
                            gpu_type_source="tres-req")
    gproc = m.GPUProcess(pid=7, gpu_uuid="GPU-1", gpu_index=1, used_mem_mib=1.5, user=None, command=None,
                         etime_s=None, cpu_pct=None, job=m.JobKey("k2", "p7-100"))
    gpu = m.GPU(host="k2", index=1, uuid="GPU-1", name="H200", util_pct=50.0, mem_used_mib=1.0, mem_total_mib=2.0,
                mem_util_pct=None, temp_c=None, power_w=None, power_limit_w=None, fan_pct=None, pstate="P0",
                source="nvidia-smi", processes=(gproc,), sampled_at=T0)
    job = m.Job(key=m.JobKey("k2", "31270818_[1-24%1]"), kind=m.JobKind.SLURM, name="n", user="u",
                state=m.JobState.PENDING, raw_state="PENDING", job_id_raw="31270818", held=True, submit_time=T0,
                elapsed=timedelta(seconds=90), time_limit=None, alloc=alloc,
                logs=(m.LogRef("stdout", None, pattern="%x_%j.out", source="squeue", mtime=T0),),
                gpu=m.JobGpuUsage("sstat", 2, 50.0, 100.0, 84.5, 24870.0, None, T0, "note"),
                hints=("waiting for priority",), raw={"Reason": "Priority"})
    node = m.Node(name="n-31-05", partitions=("lab",), state="planned", flags=frozenset({"-"}), usable=True,
                  cpus_alloc=1, cpus_idle=2, cpus_other=0, cpus_total=3, mem_total_mib=1, mem_alloc_mib=0,
                  mem_free_mib=1, gres=(m.GresCount("gpu", "a30", 4, 1, "0"),), reason=None)
    part = m.PartitionSummary(name="lab", nodes=1, gpus_total=4, gpus_free=3, gpus_free_planned=3,
                              gpu_types={"a30": (3, 4)}, states={"planned": 1})
    metrics = m.HostMetrics(ncpu=8, load=(0.1, 0.2, 0.3), cpu_pct=None, mem_total=1, mem_available=1,
                            swap_total=None, swap_free=None, uptime_s=1.5, sampled_at=T0)
    proc = m.Proc(pid=7, ppid=1, pgid=None, user="u", etime_s=5, cpu_pct=1.0, mem_pct=0.5, rss_bytes=10,
                  stat="R", args="python x.py", task=m.JobKey("k2", "p7-100"))
    return m.HostSnapshot(host="k2", taken_at=T0, caps=caps, jobs={job.key.id: job}, gpus=(gpu,), procs=(proc,),
                          nodes=(node,), partitions=(part,), metrics=metrics)


def test_snapshot_round_trip_through_json_text() -> None:
    snap = _snapshot()
    text = json.dumps(m.to_json(snap))
    back = m.from_json(m.HostSnapshot, json.loads(text))
    assert back == snap


def test_status_detail_report_round_trip() -> None:
    err = m.HostError(m.ErrorKind.AUTH, "auth", "detail", section=None, at=T0)
    st = m.HostStatus(name="k2", state=m.ConnState.NEEDS_AUTH, error=err, mux="user", tick_interval=15.0,
                      sections={"slurm.squeue": m.SectionStatus("slurm.squeue", "ok", True, 0, 3, T0)})
    for obj in (st, m.DoctorReport("k2", (m.DoctorCheck("probe", None, "x"),), ("w",)),
                m.ActionResult("slurm.cancel", m.JobKey("k2", "1"), False, "nope", ("a",), 1, "", "", T0, 0.5),
                m.DetectReport("k2", m.ConnState.OK, None, None, "managed", {"user": "u"}, 0.1)):
        assert m.from_json(type(obj), json.loads(json.dumps(m.to_json(obj)))) == obj


def test_json_shapes() -> None:
    out = m.to_json(_snapshot())
    assert out["taken_at"] == "2026-09-07T10:42:37+00:00"
    assert out["caps"]["tz_offset"] == -18000.0
    assert out["caps"]["tools"] == ["sinfo", "squeue"]              # frozensets sorted
    job = out["jobs"]["31270818_[1-24%1]"]
    assert job["state"] == "PD" and job["kind"] == "slurm" and job["key"] == {"host": "k2", "id": "31270818_[1-24%1]"}


def test_datetimes_are_utc_and_naive_is_assumed_utc() -> None:
    est = datetime(2026, 9, 7, 5, 42, 37, tzinfo=timezone(timedelta(hours=-5)))
    assert m.to_json(est) == "2026-09-07T10:42:37+00:00"
    assert m.to_json(datetime(2026, 1, 1)) == "2026-01-01T00:00:00+00:00"
    assert m.from_json(datetime, "2026-01-01T00:00:00") == datetime(2026, 1, 1, tzinfo=UTC)


def test_missing_keys_take_defaults_and_unknown_keys_are_ignored() -> None:
    st = m.from_json(m.HostStatus, {"name": "x", "state": "ok", "future_field": 1})
    assert st == m.HostStatus(name="x", state=m.ConnState.OK)


@pytest.mark.parametrize("bad", [{"name": "x", "state": "nope"}, {"name": 3, "state": "ok"},
                                 {"name": "x", "state": "ok", "consecutive_failures": True}])
def test_invalid_data_raises(bad: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        m.from_json(m.HostStatus, bad)


def test_to_json_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        m.to_json(object())


def test_out_of_range_duration_is_a_value_error() -> None:
    from datetime import timedelta

    import pytest

    from runfleet.models import from_json
    for bad in (1e30, float("inf")):
        with pytest.raises(ValueError):
            from_json(timedelta, bad)
