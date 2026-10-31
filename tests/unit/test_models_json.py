"""[WP1] models.to_json / from_json round-trip every §3.3 type (cache, snapshot --json)."""

from __future__ import annotations

import json
from dataclasses import replace
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


def test_account_models_round_trip_with_dates() -> None:
    from datetime import date
    bal = m.AccountBalance(account="pilab_02", limit_su=97885.0, used_su=298.12, used_basis="since_reset",
                           limit_source="sshare", used_source="sshare", other_limits={"gres/gpu": 2000.0},
                           next_realloc=date(2027, 1, 14), realloc_known=True, read_at=T0)
    out = m.to_json(bal)
    assert out["next_realloc"] == "2027-01-14" and out["read_at"] == "2026-09-07T10:42:37+00:00"
    assert m.from_json(m.AccountBalance, json.loads(json.dumps(out))) == bal
    assert bal.available_su == pytest.approx(97586.88)
    use = m.AccountUse(account="pilab_02", scope="account", cpu_h=1316.0, gpu_h=None, billing_h=1316.0, start=T0,
                       start_inferred=True)
    assert m.from_json(m.AccountUse, json.loads(json.dumps(m.to_json(use)))) == use
    with pytest.raises((TypeError, ValueError)):
        m.from_json(m.AccountBalance, {**out, "next_realloc": "2027-01-14T00:00:00"})


def test_snapshot_with_account_summary_round_trips_and_old_rows_decode() -> None:
    """Account (§5.2.11): `HostSnapshot.account` rides the snapshots JSON; a row of 0.8.0 (no key) decodes to None."""
    from datetime import date
    bal = m.AccountBalance(account="pilab_02", limit_su=97885.0, used_su=298.12, used_basis="since_reset",
                           limit_source="sshare", used_source="sshare", next_realloc=date(2027, 1, 14),
                           realloc_known=True, read_at=T0)
    use = m.AccountUse(account="pilab_02", scope="me", cpu_h=298.1, gpu_h=0.0, billing_h=298.1, start=T0)
    err = m.HostError(m.ErrorKind.REMOTE, "slurmdbd unreachable", "x", section="slurm.acct_sreport", at=T0)
    summary = m.AccountSummary(window="30d", read_at=T0, mode="limits", balances=(bal,), usage=(use,), usage_at=T0,
                               sbank_at=T0, half_life_s=0.0, reset_period=None, notes=("sreport not installed",),
                               errors=(err,))
    snap = replace(_snapshot(), account=summary)
    back = m.from_json(m.HostSnapshot, json.loads(json.dumps(m.to_json(snap))))
    assert back == snap and back.account is not None and back.account.balances[0].next_realloc == date(2027, 1, 14)
    old = m.to_json(_snapshot())
    del old["account"]
    assert m.from_json(m.HostSnapshot, old).account is None


def test_account_report_round_trips() -> None:
    """Account (§5.2.11): `AccountReport` as the `account_reports` cache stores it (to_json)."""
    use = m.AccountUse(account="pilab_02", scope="account", cpu_h=1316.0, gpu_h=171.0, billing_h=1316.0, start=T0,
                       start_inferred=True)
    cell = m.UsageCell(name="lmft--train--s0--a01", account="pilab_02", partition="gpu", qos="normal",
                       gpu_model="v100", jobs=3, running=0, cpu_h=23.2, gpu_h=2.9, billing_h=116.0, charged_su=None)
    summary = m.AccountSummary(window="period", read_at=T0, mode="usage_only", usage=(replace(use, scope="me"),))
    err = m.HostError(m.ErrorKind.REMOTE, "sacct refused the date range", "x", section="slurm.acct_jobs.c0", at=T0)
    report = m.AccountReport(host="lyra", window="period", fetched_at=T0, summary=summary, window_start=T0,
                             account_totals=(use,), cells=(cell,), jobs_seen=3, truncated=True, errors=(err,))
    assert m.from_json(m.AccountReport, json.loads(json.dumps(m.to_json(report)))) == report


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


def test_pending_priority_round_trip_and_old_rows() -> None:
    """§5.2.10: a pending job's priority evidence survives the cache / snapshot --json; rows written before it decode
    with priority None."""
    prio = m.PendingPriority(
        ranked=True, priority=3017.0, partition="gpu-preempt",
        factors=(m.PriorityFactor("fairshare", 3000.0, 100000.0), m.PriorityFactor("nice", -50.0)),
        dominant="fairshare", others=(("lab", 2017.0),),
        fairshare=(m.FairShare("lab", "u", "parent", 0.5, 123456, 0.01, 0.03, None),),
        hints=("fairshare 0.03: …",), sampled_at=T0, half_life_s=0.0)
    job = m.Job(key=m.JobKey("k2", "2334564"), kind=m.JobKind.SLURM, name="n", user="u", state=m.JobState.PENDING,
                raw_state="PENDING", job_id_raw="2334564", priority=prio)
    data = json.loads(json.dumps(m.to_json(job)))
    assert m.from_json(m.Job, data) == job
    assert data["priority"]["others"] == [["lab", 2017.0]] and data["priority"]["fairshare"][0]["level_fs"] is None
    del data["priority"]
    assert m.from_json(m.Job, data).priority is None
