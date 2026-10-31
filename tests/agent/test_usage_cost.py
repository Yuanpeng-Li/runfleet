"""`runfleet agent usage` / `cost` and the job cost of `job` / `jobs` / `host` (DESIGN.md §8.7, §5.2.11;
docs/AGENT_INTERFACE.md §3.14-§3.15): the pure documents over hand-built hosts, the cache path (summary vs cached
reports, `usage_window_not_cached`, `--refresh` upserting a report) on a temporary sqlite written by the real
core.cache.Cache from the demo world, the validation before any IO, and the MCP tools. No real host."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli
from runfleet.agent import commands, mcp, source, view
from runfleet.agent.data import EXIT_NO_DATA, EXIT_OK, EXIT_PARTIAL, FleetView, HostView
from runfleet.config import HostConfig
from runfleet.core.cache import Cache, upsert_account_report
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.demo.transport import DemoTransport
from runfleet.demo.world import DemoWorld
from runfleet.models import (AccountBalance, AccountReport, AccountSummary, AccountUse, Capabilities, ConnState,
                             ErrorKind, HostError, Job, JobState, PartitionCharge, PartitionSummary, UsageCell)
from runfleet.transport.base import Transport
from tests.agent.test_view import ENVELOPE, T0, H, fleet, keys, mk_host, mk_job

LYRA_PARTS = (
    PartitionCharge("free-gpu", "", "free-gpu-part", ("low", "guest")),
    PartitionCharge("gpu", "cpu=1,gres/gpu=32", "gpu-part", ("normal", "high")),
    PartitionCharge("free-gpuxl", "", "free-gpuxl-part", ("low", "guest")),
    PartitionCharge("gpuxl", "cpu=1,gres/gpu:L40S=32,gres/gpu:RTX6000=64", "gpuxl-part", ("normal", "high")),
)
FACTORS = {"normal": 1.0, "low": 0.0, "high": 2.0, "guest": 0.0, "free-gpuxl-part": 1.0}
SLURMDBD = {"AccountingStorageType": "accounting_storage/slurmdbd", "PriorityDecayHalfLife": "00:00:00"}
GPUXL = PartitionSummary("gpuxl", 8, 32, 4, 0, {"l40s": (2, 4), "rtx6000": (2, 28)}, {"mixed": 8})
FREE_GPUXL = PartitionSummary("free-gpuxl", 8, 32, 4, 0, {"l40s": (2, 4), "rtx6000": (2, 28)}, {"mixed": 8})


def lyra_caps(**kw: Any) -> Capabilities:
    d: dict[str, Any] = dict(remote_user="alice", uid=1000, login_host="login", tz_offset=timedelta(0),
                             clock_skew=timedelta(0), tools=frozenset({"sshare", "sreport", "sbank"}),
                             profile=("slurm",), cluster_name="lyra", slurm_config=SLURMDBD,
                             associations=(("lyra", "pilab_02", "", "normal,low"), ("lyra", "lab_gpu", "", "normal")),
                             qos_usage_factor=FACTORS, partition_charges=LYRA_PARTS)
    d.update(kw)
    return Capabilities(**d)


def k2_caps() -> Capabilities:
    return lyra_caps(cluster_name="k2", associations=(("k2", "lab", "", "normal"),),
                     slurm_config={**SLURMDBD, "PriorityDecayHalfLife": "7-00:00:00"},
                     qos_usage_factor={"normal": 1.0},
                     partition_charges=(PartitionCharge("lab"), PartitionCharge("gpu-preempt")))


def summary(**kw: Any) -> AccountSummary:
    d: dict[str, Any] = dict(
        window="30d", read_at=T0 - timedelta(minutes=2), mode="limits", usage_at=T0 - timedelta(minutes=2),
        sbank_at=T0 - 3 * H, half_life_s=0.0,
        balances=(AccountBalance("lab_gpu", 97.0, 0.0, "since_reset", "sshare", "sshare", next_realloc=date(2027, 4, 2),
                                 realloc_known=True),
                  AccountBalance("pilab_02", 97885.0, 298.0, "since_reset", "sshare", "sshare",
                                 next_realloc=date(2027, 1, 14), realloc_known=True)),
        usage=(AccountUse("lab_gpu", "me", 0.0, 0.0, 0.0), AccountUse("pilab_02", "me", 205.0, 2.9, 300.0)))
    d.update(kw)
    return AccountSummary(**d)


def k2_summary() -> AccountSummary:
    return AccountSummary(window="30d", read_at=T0 - timedelta(minutes=2), mode="usage_only",
                          usage_at=T0 - timedelta(minutes=2), half_life_s=14 * 86400.0,
                          balances=(AccountBalance("lab", None, 2_503_303.8, "decayed", None, "sshare",
                                                   my_used_su=5859.0),),
                          usage=(AccountUse("lab", "me", 171881.0, 15434.0, 171881.0),))


def cell(name: str, gpu_h: float, *, partition: str = "gpuxl", qos: str = "normal", su: float | None = None,
         account: str = "pilab_02") -> UsageCell:
    return UsageCell(name, account, partition, qos, "l40s", 2, 0, gpu_h * 8, gpu_h, gpu_h * 40,
                     gpu_h * 40 if su is None else su)


def report(window: str = "90d", cells: Iterable[UsageCell] = (), **kw: Any) -> AccountReport:
    d: dict[str, Any] = dict(host="lyra", window=window, fetched_at=T0 - timedelta(minutes=1),
                             summary=summary(window=window), window_start=T0 - timedelta(days=90),
                             account_totals=(AccountUse("pilab_02", "account", 1076.0, 174.0, 1162.0),),
                             cells=tuple(cells), jobs_seen=12)
    d.update(kw)
    return AccountReport(**d)


def lyra(jobs: Iterable[Job] = (), *, caps: Capabilities | None = None, account: AccountSummary | None = None,
         reports: dict[str, AccountReport] | None = None, **kw: Any) -> HostView:
    hv = mk_host("lyra", jobs, partitions=(GPUXL, FREE_GPUXL), **kw)
    assert hv.snapshot is not None
    snap = dataclasses.replace(hv.snapshot, caps=caps or lyra_caps(), account=account)
    return dataclasses.replace(hv, snapshot=snap, caps_at=T0 - H, account_reports=reports or {})


def k2(**kw: Any) -> HostView:
    hv = mk_host("k2")
    assert hv.snapshot is not None
    return dataclasses.replace(hv, snapshot=dataclasses.replace(hv.snapshot, caps=k2_caps(), account=k2_summary()),
                               **kw)


def gpuxl_job(jid: str = "500", **kw: Any) -> Job:
    """A running job of mine on gpuxl: 1 L40S, 8 CPUs, billing 40/h (Slurm's own), 2 h so far, 1 d limit."""
    from runfleet.models import ResourceAlloc
    tres = {"cpu": "8", "mem": "64G", "node": "1", "billing": "40", "gres/gpu": "1", "gres/gpu:l40s": "1"}
    d: dict[str, Any] = dict(partition="gpuxl", qos="normal", account="pilab_02",
                             alloc=ResourceAlloc(1, 8, 64 << 30, 1, "l40s", tres))
    d.update(kw)
    return mk_job("lyra", jid, **d)


# ====================================================================================================== job cost
def test_job_card_and_json_carry_the_cost() -> None:
    fv = fleet(lyra([gpuxl_job()]))
    doc = view.job(fv, [("lyra", "500")])
    cost = doc.data["jobs"][0]["cost"]
    assert cost["billing"] == 40 and cost["billing_basis"] == "tres" and cost["usage_factor"] == 1.0
    assert cost["qos"] == "normal" and cost["live"] is True and cost["free"] is False
    assert cost["gpu_h_by_model"] == {"l40s": cost["gpu_h"]} and cost["as_of"] == view.iso(T0)
    assert cost["charged_su"] == pytest.approx(80.0, abs=0.01) and cost["projected_su"] == 960.0
    line = next(ln for ln in doc.text.splitlines() if ln.startswith("  cost "))
    assert line == "  cost     2.0 GPU-h (L40S) · 16.0 CPU-h · 40/h × qos normal ×1 = 80 SU so far · ≈ 960 at limit"


def test_free_pending_old_and_other_users_jobs() -> None:
    free = gpuxl_job("501", partition="free-gpuxl", qos="low")
    pend = gpuxl_job("502", state=JobState.PENDING)
    old = gpuxl_job("503", observed_at=T0 - timedelta(minutes=20))
    other = gpuxl_job("504", user="bob", is_mine=False)
    fv = fleet(lyra([free, pend, old, other]))
    card = view.job(fv, [("lyra", "501")]).text
    assert "  cost     2.0 GPU-h (L40S) · 16.0 CPU-h · free (qos low ×0) · 80 billing-h not charged" in card
    assert "  cost     ≈ 480 SU if it runs to its limit (40/h × qos normal ×1)" in view.job(fv, [("lyra", "502")]).text
    old_doc = view.job(fv, [("lyra", "503")])
    oc = old_doc.data["jobs"][0]["cost"]
    assert oc["as_of"] == view.iso(old.observed_at) and oc["live"] is False
    assert "as of 20m ago" in next(ln for ln in old_doc.text.splitlines() if ln.startswith("  cost "))
    assert view.job(fv, [("lyra", "504")]).data["jobs"][0]["cost"] is None
    assert "cost" not in view.job(fv, [("lyra", "504")]).text


def test_cost_line_is_the_first_to_go_when_a_card_is_too_big() -> None:
    j = gpuxl_job(hints=tuple("h" * 110 + str(i) for i in range(4)), workdir="/w/" + "d" * 230)
    lines = view._job_lines(j, lyra([j]), T0, 86400.0, fleet(lyra([j])))
    capped = [ln if ln.startswith(view.CARD_NEXT) else view._bcap(ln, 200) for ln in lines]
    fitted = view._fit_card(lines, len("\n".join(capped).encode()) - 10)
    assert "  cost " not in fitted and "  hints " in fitted


def test_jobs_sort_su_and_gpuh() -> None:
    small = gpuxl_job("600", elapsed=H)
    big = gpuxl_job("601", elapsed=5 * H, alloc=dataclasses.replace(gpuxl_job().alloc, gpus=2, tres={
        "cpu": "8", "billing": "72", "gres/gpu": "2"}))
    free = gpuxl_job("602", partition="free-gpuxl", qos="low", elapsed=9 * H)
    fv = fleet(lyra([small, free, big]))
    su = view.jobs(fv, view.JobFilter(sort="su"))
    assert [j["id"] for j in su.data["jobs"]] == ["lyra:601", "lyra:600", "lyra:602"]
    assert su.data["filter"]["sort"] == "su"
    gpuh = view.jobs(fv, view.JobFilter(sort="gpuh"), limit=1)
    assert [j["id"] for j in gpuh.data["jobs"]] == ["lyra:601"]
    assert gpuh.data["truncated"]["next"] == "runfleet agent jobs --sort gpuh --limit 3 --json"


# ====================================================================================================== cost
def creq(**kw: Any) -> view.CostRequest:
    d: dict[str, Any] = dict(partition="gpuxl", gpus=1, gpu_type="L40S", cpus=8, hours=24.0, time_text="24h")
    d.update(kw)
    return view.CostRequest(**d)


def test_cost_matches_the_mockup() -> None:
    last = gpuxl_job("700", state=JobState.COMPLETED)
    fv = fleet(lyra([last], account=summary()))
    doc = view.cost(fv, "lyra", creq())
    assert doc.exit_code == EXIT_OK
    assert doc.text.splitlines()[1:] == [
        "lyra gpuxl · 1× L40S · 8 cpus (--cpus) · 24h · qos normal ×1 (your last job on gpuxl)",
        "  billing 40/h  = cpu 8×1 + gres/gpu:L40S 1×32   (TRESBillingWeights of gpuxl)",
        "  cost    960 SU for 24h · account pilab_02: 97,587 SU left → 1.0%",
        "  cheaper free-gpuxl has L40S, qos low ×0, guest ×0"]
    d = doc.data
    assert keys(d) == sorted([*ENVELOPE, "host", "partition", "request", "billing_per_h", "billing_terms",
                              "billing_basis", "qos", "usage_factor", "usage_factor_basis", "hours", "su", "su_min",
                              "su_max", "unit", "account", "candidate_accounts", "alternatives", "notes",
                              "caps_age_s", "known_partitions"])
    assert d["su"] == 960 and d["billing_terms"] == [{"tres": "cpu", "count": 8.0, "weight": 1.0},
                                                     {"tres": "gres/gpu:L40S", "count": 1.0, "weight": 32.0}]
    assert d["account"] == {"account": "pilab_02", "basis": "your last job on gpuxl", "available_su": 97587.0,
                            "share_pct": 1.0, "exceeds": False, "as_of": view.iso(T0 - timedelta(minutes=2))}
    assert [a["partition"] for a in d["alternatives"]] == ["free-gpuxl"] and d["caps_age_s"] == 3600
    assert len(doc.text.encode()) <= 2048 and len(view.dumps(d)) <= 4096


def test_cost_names_every_limited_account_and_says_exceeds() -> None:
    """Review: no --account and no job on the partition gave no account line; an estimate far above what is left
    was a bare `1608%`. Now every limited account that may run there is listed (tightest first) with `exceeds`."""
    fv = fleet(lyra(account=summary()))
    doc = view.cost(fv, "lyra", creq(gpu_type="RTX6000", cpus=None))
    d = doc.data
    assert d["account"] is None and d["su"] == 1560
    assert [(c["account"], c["available_su"], c["exceeds"]) for c in d["candidate_accounts"]] == [
        ("lab_gpu", 97.0, True), ("pilab_02", 97587.0, False)]
    assert "  accounts lab_gpu 97 left (EXCEEDS) · pilab_02 97,587 left" in doc.text
    picked = view.cost(fv, "lyra", creq(gpu_type="RTX6000", cpus=None, account="LAB_GPU"))   # sbank spells it so
    acct = picked.data["account"]
    assert acct["account"] == "lab_gpu" and acct["available_su"] == 97 and acct["exceeds"] is True
    assert acct["share_pct"] > 100 and "exceeds what is left on lab_gpu (97 SU)" in picked.data["notes"]
    assert "account lab_gpu: EXCEEDS what is left (1,560 > 97 SU left)" in picked.text
    assert picked.data["candidate_accounts"] == []


def test_cost_without_gpu_type_on_typed_only_weights_is_a_range() -> None:
    """Review: gpuxl weighs only gres/gpu:L40S and gres/gpu:RTX6000; without --gpu-type the estimate was 24 SU
    (CPUs only). Now it is the range of the typed models, `su` the highest."""
    doc = view.cost(fleet(lyra(account=summary())), "lyra", creq(gpu_type=None, cpus=None))
    d = doc.data
    assert (d["su_min"], d["su_max"], d["su"]) == (792, 1560, 1560)
    assert "  cost    792-1,560 SU for 24h (L40S…RTX6000; pass --gpu-type)" in doc.text.splitlines()
    assert d["notes"][0].startswith("no --gpu-type: 792 (L40S) to 1,560 (RTX6000) SU")
    one = view.cost(fleet(lyra(account=summary())), "lyra", creq(partition="gpu", gpu_type=None, cpus=None))
    assert one.data["su_min"] is None and one.data["su"] == 792                     # untyped gres/gpu weight


def test_cost_defaults_notes_and_old_rules() -> None:
    fv = fleet(lyra(caps_at=None))
    hv = dataclasses.replace(lyra(), caps_at=T0 - timedelta(days=2))
    doc = view.cost(fleet(hv), "lyra", creq(partition="gpu", gpu_type=None, cpus=None, hours=None, time_text=None))
    assert "1 cpu (1 per node: no --cpus, no DefCpuPerGPU)" in doc.data["notes"]
    assert "no time: per hour only" in doc.data["notes"] and "charging rules as of 2d ago" in doc.data["notes"]
    assert doc.data["su"] is None and "  cost    33/h SU (no --time)" in doc.text
    assert doc.data["qos"] == "normal" and doc.data["usage_factor_basis"] == "default normal"
    free = view.cost(fv, "lyra", creq(partition="free-gpuxl"))       # AllowQos low,guest: normal is not allowed
    assert free.data["su"] == 0 and free.data["alternatives"] == []
    assert (free.data["qos"], free.data["usage_factor_basis"]) == ("low", "AllowQos (all alike)")


def test_cost_unknown_partition_workstation_and_no_rules() -> None:
    doc = view.cost(fleet(lyra()), "lyra", creq(partition="nope"))
    assert doc.exit_code == EXIT_NO_DATA and doc.data["errors"][0]["code"] == "not_found"
    assert doc.data["known_partitions"] == ["free-gpu", "gpu", "free-gpuxl", "gpuxl"]
    assert "not_found  no partition nope → known: free-gpu, gpu, free-gpuxl, gpuxl" in doc.text
    ws = view.cost(fleet(mk_host("labstation", kind="workstation")), "labstation", creq())
    assert ws.exit_code == EXIT_NO_DATA and "workstation has no scheduler" in ws.text
    bare = view.cost(fleet(lyra(caps=lyra_caps(partition_charges=()))), "lyra", creq())
    assert bare.exit_code == EXIT_NO_DATA and bare.data["errors"][0]["code"] == "no_data"


def test_cost_on_a_usage_only_cluster_is_billing_hours() -> None:
    doc = view.cost(fleet(k2()), "k2", creq(partition="lab", gpus=0, gpu_type=None, cpus=16))
    assert doc.data["unit"] == "billing-h" and doc.data["su"] == 384
    assert "  cost    384 billing-h for 24h" in doc.text
    assert any("usage only" in n for n in doc.data["notes"])

    # review: a ×0 QOS on a usage-only cluster still runs 384 billing-h (unscaled, as the job cost shows them)
    free = dataclasses.replace(k2_caps(), qos_usage_factor={"normal": 1.0, "scavenger": 0.0})
    hv = k2()
    hv = dataclasses.replace(hv, snapshot=dataclasses.replace(hv.snapshot, caps=free))  # type: ignore[arg-type]
    doc = view.cost(fleet(hv), "k2", creq(partition="lab", gpus=0, gpu_type=None, cpus=16, qos="scavenger"))
    assert doc.data["su"] == 384 and "  cost    384 billing-h for 24h" in doc.text
    rate = view.cost(fleet(hv), "k2", creq(partition="lab", gpus=0, gpu_type=None, cpus=16, qos="scavenger",
                                           hours=None, time_text=None))
    assert "  cost    16/h billing-h (no --time)" in rate.text


# ====================================================================================================== usage
def test_usage_default_window_reads_the_summary() -> None:
    fv = fleet(lyra(account=summary()), k2(), mk_host("labstation", kind="workstation"))
    doc = view.usage(fv)
    assert doc.exit_code == EXIT_OK, doc.text
    lines = doc.text.splitlines()
    assert lines[1] == "lyra  limits (sbank + sshare)  usage 2m ago · balances 2m ago · sbank 3h ago"
    assert lines[3] == ("  pilab_02       298 / 97,885 SU used · 97,587 left · realloc 2027-01-14 · 30d me 205 CPU-h "
                        "2.9 GPU-h")
    assert lines[4] == ("  charged gpu (cpu=1,gres/gpu=32), gpuxl (cpu=1,gres/gpu:L40S=32,gres/gpu:RTX6000=64) · "
                        "free free-gpu, free-gpuxl (qos guest ×0, low ×0)")
    assert "k2  usage only (no SU limits; billing = CPUs)  usage 2m ago · balances 2m ago" in lines
    # usage only: the user's own sshare row and the whole account's, both labelled (k2: 5,859 vs 2,503,304)
    assert "  lab            recent me 5,859 · account 2,503,304 billing-h (decays, half-life 14d) · 30d me 171,881 " \
           "CPU-h 15,434 GPU-h 171,881 billing-h" in lines
    assert lines[-1] == "labstation  workstation · no scheduler accounting"
    hu = {h["host"]: h for h in doc.data["hosts_usage"]}
    assert keys(hu["lyra"]) == sorted(["host", "state", "mode", "window", "summary_at", "summary_age_s", "sbank_at",
                                       "report_at", "report_age_s", "window_start", "accounts", "partitions",
                                       "rollup", "next", "notes", "errors"])
    acct = hu["lyra"]["accounts"][1]
    assert acct["account"] == "pilab_02" and acct["available_su"] == 97587 and acct["account_total"] is None
    assert acct["me"] == {"cpu_h": 205.0, "gpu_h": 2.9, "billing_h": 300.0, "source": "sreport"}
    assert hu["lyra"]["rollup"] is None and hu["k2"]["accounts"][0]["limit_su"] is None
    assert hu["labstation"]["state"] == "workstation" and hu["labstation"]["mode"] == "none"


def test_usage_window_not_cached_names_the_exact_refresh() -> None:
    fv = fleet(lyra(account=summary()))
    doc = view.usage(fv, window="90d")
    assert doc.exit_code == EXIT_NO_DATA
    assert doc.text.splitlines()[1] == ("lyra  no_data: 90d never fetched → runfleet agent usage lyra --window 90d "
                                        "--refresh")
    h = doc.data["hosts_usage"][0]
    assert h["state"] == "usage_window_not_cached"
    assert h["next"] == "runfleet agent usage lyra --window 90d --refresh --json"
    assert doc.data["errors"][0]["code"] == "no_data" and "--window 90d --refresh" in doc.data["errors"][0]["hint"]
    by = view.usage(fv, by="task")
    assert "lyra  no_data: 30d by task never fetched → runfleet agent usage lyra --by task --refresh" in by.text
    mixed = view.usage(fleet(lyra(account=summary()), k2(account_reports={"90d": report(host="k2")})), window="90d")
    assert mixed.exit_code == EXIT_PARTIAL


def test_usage_with_a_cached_report_shows_the_rollup() -> None:
    cells = [cell(f"orb--train--lr{i}--s{s}--a01", 10.0 * (i + 1)) for i in range(7) for s in (0, 1)]
    fv = fleet(lyra(account=summary(), reports={"90d": report(cells=cells)}))
    doc = view.usage(fv, window="90d", scope="account", limit=3)
    assert doc.exit_code == EXIT_OK, doc.text
    text = doc.text.splitlines()
    assert "  top 90d by experiment (3 of 7 · 12 jobs):" in text
    assert text[text.index("  top 90d by experiment (3 of 7 · 12 jobs):") + 1] == \
        "    orb--train--lr6  140 GPU-h · 1,120 CPU-h · 5,600 SU (4 jobs)"
    assert "    # +4 more → runfleet agent usage lyra --window 90d --by experiment --limit 7" in text
    assert "· account 1,076 CPU-h 174 GPU-h" in doc.text
    roll = doc.data["hosts_usage"][0]["rollup"]
    assert roll["by"] == "experiment" and roll["rows_total"] == 7 and len(roll["rows"]) == 3
    assert keys(roll["rows"][0]) == sorted(["key", "jobs", "running", "cpu_h", "gpu_h", "gpu_h_by_model", "billing_h",
                                            "charged_su", "factor_unknown", "accounts"])
    by_task = view.usage(fv, window="90d", by="task")
    assert by_task.data["hosts_usage"][0]["rollup"]["rows"][0]["key"] == "orb"


def test_usage_period_fallbacks_and_old_reports() -> None:
    period_30 = report(window="30d", fetched_at=T0 - 2 * H)
    fv = fleet(lyra(account=summary(), reports={"period": period_30}), k2(account_reports={"period": report(
        window="30d", host="k2", summary=k2_summary())}))
    doc = view.usage(fv, window="period")
    assert "  note: period not known yet: showing 30d (--refresh reads again)" in doc.text
    assert "  note: report from 2h ago: --refresh reads it again" in doc.text
    assert "  note: no period window on k2 (no reallocation date or usage reset): showing 30d" in doc.text
    start = T0 - timedelta(days=290)
    inferred = report(window="period", summary=summary(window="period", usage=(
        AccountUse("pilab_02", "me", 900.0, 9.0, 950.0, start=start, start_inferred=True),)))
    acct = view.usage(fleet(lyra(account=summary(), reports={"period": inferred})), window="period")
    pil = next(a for a in acct.data["hosts_usage"][0]["accounts"] if a["account"] == "pilab_02")
    assert pil["period_start"] == view.iso(start) and pil["period_inferred"] is True
    assert f"since {view.iso(start)[:10]} (inferred)" in acct.text


def test_usage_failing_sections_are_fixed_texts_and_partial() -> None:
    err = HostError(ErrorKind.REMOTE, "sreport: secret.example.org refused", "stderr with secret", section="slurm.acct_sreport",
                    at=T0 - timedelta(minutes=12))
    doc = view.usage(fleet(lyra(account=summary(errors=(err,)))))
    assert doc.exit_code == EXIT_PARTIAL
    assert "  ! sreport failing (remote, 12m ago) (last good values kept)" in doc.text
    assert doc.data["hosts_usage"][0]["errors"] == [{"section": "sreport", "kind": "remote",
                                                     "at": view.iso(err.at), "age_s": 720}]
    assert "secret" not in doc.text and "secret" not in view.dumps(doc.data)


def test_usage_states_off_no_accounting_not_read_and_no_slurm() -> None:
    off = view.usage(fleet(lyra(account=summary(), account_usage=False)))
    assert off.data["hosts_usage"][0]["state"] == "off" and "account usage is off for lyra in config" in off.text
    nodbd = view.usage(fleet(lyra(caps=lyra_caps(slurm_config={}))))
    assert nodbd.data["hosts_usage"][0]["state"] == "no_accounting"
    unread = view.usage(fleet(lyra()))
    assert unread.data["hosts_usage"][0]["state"] == "not_read" and unread.exit_code == EXIT_NO_DATA
    assert "lyra  no_data: the account summary never fetched → runfleet agent usage lyra --refresh" in unread.text
    ws = view.usage(fleet(mk_host("labstation", kind="workstation")))
    assert ws.exit_code == EXIT_NO_DATA and ws.data["errors"][0]["code"] == "error"


def test_usage_budgets_hold_with_many_accounts_and_rows() -> None:
    bal = tuple(AccountBalance(f"acct{i:02d}-" + "x" * 40, 1e6, 1e3 * i, "since_reset", "sshare", "sshare")
                for i in range(12))
    use = tuple(AccountUse(b.account, "me", 1e4, 1e3, 1e4) for b in bal)
    cells = [cell("e" * 150 + str(i), 1.0 + i) for i in range(60)]
    hosts = [dataclasses.replace(lyra(account=summary(balances=bal, usage=use), reports={"30d": report(
        "30d", cells=cells, summary=summary(balances=bal, usage=use))}), name=f"lyra{i}") for i in range(6)]
    doc = view.usage(fleet(*hosts), limit=50)
    assert len(doc.text.encode()) <= 4096 and len(view.dumps(doc.data)) <= 12288
    assert doc.text.splitlines()[-1].startswith("# output cut at 4096 bytes")


def test_balance_wording_decayed_limits_and_realloc_none_set() -> None:
    """Review: a decayed number is never "used" (limits mode with a half-life > 0), and sbank's "No data found." is
    "realloc none set", not "unknown"; account names match ignoring case."""
    bal = (AccountBalance("alice", 1000.0, 0.0, "since_reset", "sshare", "sshare", realloc_known=True),
           AccountBalance("pilab_02", 97885.0, 298.0, "decayed", "sshare", "sshare"))
    doc = view.usage(fleet(lyra(account=summary(balances=bal, half_life_s=7 * 86400.0))))
    text = doc.text
    assert "  alice          0 / 1,000 SU used · 1,000 left · realloc none set" in text
    assert "298 SU recent (decays, half-life 7d) / 97,885 limit · 97,587 left" in text and "realloc unknown" not in text
    accts = {a["account"]: a for a in doc.data["hosts_usage"][0]["accounts"]}
    assert accts["alice"]["next_realloc"] is None and accts["alice"]["realloc_known"] is True
    assert view._find(bal, "PILAB_02") is bal[1]


def test_host_line_usage_only_recent_is_mine_not_the_accounts() -> None:
    doc = view.host(fleet(k2()), "k2")
    line = next(ln for ln in doc.text.splitlines() if ln.startswith("account  "))
    assert "30d me 171,881 CPU-h · 15,434 GPU-h · recent 5,859 billing-h (decays)" in line and "2,503" not in line
    assert doc.data["host"]["account"]["accounts"][0]["my_used_su"] == 5859


def test_a_failed_rollup_is_unknown_never_no_jobs() -> None:
    err = HostError(ErrorKind.REMOTE, "slurmdbd unreachable", section="slurm.acct_jobs.c0", at=T0)
    rep = report("90d", errors=(err,), jobs_seen=0)
    text = view.usage(fleet(lyra(account=summary(), reports={"90d": rep})), window="90d").text
    assert "  rollup unknown (sacct failing)" in text and "no jobs of yours" not in text
    assert "(last good values kept)" not in text
    # review: a transport failure (no section) or a cut rollup never reads as an empty history either
    ssh = HostError(ErrorKind.SSH, "connection refused", at=T0)
    rep = report("90d", errors=(ssh,), jobs_seen=0)
    text = view.usage(fleet(lyra(account=summary(), reports={"90d": rep})), window="90d").text
    assert "  rollup unknown (jobs not read)" in text and "no jobs of yours" not in text
    rep = report("90d", truncated=True, jobs_seen=0)
    text = view.usage(fleet(lyra(account=summary(), reports={"90d": rep})), window="90d").text
    assert "  rollup unknown (incomplete)" in text and "no jobs of yours" not in text


# ====================================================================================================== host
def test_host_account_line_and_json() -> None:
    doc = view.host(fleet(lyra(account=summary())), "lyra")
    line = next(ln for ln in doc.text.splitlines() if ln.startswith("account  "))
    assert line == ("account  lab_gpu 0/97 SU · pilab_02 298/97,885 · 30d me 205 CPU-h · 2.9 GPU-h · read 2m ago → "
                    "runfleet agent usage lyra")
    acct = doc.data["host"]["account"]
    assert acct["mode"] == "limits" and acct["accounts"][1]["limit_su"] == 97885.0
    unread = view.host(fleet(lyra()), "lyra")
    assert "account  not read yet (every 30 min)" in unread.text and unread.data["host"]["account"] is None
    assert "account " not in view.host(fleet(lyra(caps=lyra_caps(slurm_config={}))), "lyra").text


# ====================================================================================================== commands
@pytest.mark.parametrize(("verb", "params", "text"), [
    ("usage", {"window": "1y"}, "window: one of 7d 30d 90d period"),
    ("usage", {"by": "user"}, "by: one of experiment"),
    ("usage", {"scope": "lab"}, "scope: one of me, account"),
    ("usage", {"limit": 51}, "limit: at most 50"),
    ("cost", {"host": "lyra"}, "--partition is required"),
    ("cost", {"host": "lyra", "partition": "gpu", "time": "0h"}, "time: a positive duration"),
    ("cost", {"host": "lyra", "partition": "gpu", "time": "soon"}, "time: a positive duration"),
    ("cost", {"host": "lyra", "partition": "gpu", "mem": "lots"}, "mem: a positive size"),
    ("cost", {"host": "lyra", "partition": "gpu", "gpus": -1}, "gpus: expected an integer >= 0"),
    ("cost", {"host": "lyra", "partition": "gpu", "gpu_type": "l40s; rm"}, "gpu_type: a GPU type"),
    ("cost", {"host": "nosuch", "partition": "gpu"}, "unknown host"),
    ("jobs", {"sort": "cost"}, "sort: one of su, gpuh"),
])
def test_usage_errors_before_any_io(verb: str, params: dict[str, Any], text: str) -> None:
    with pytest.raises(commands.UsageError, match=text.replace("(", r"\(")):
        commands.run(verb, params, demo=True)


def test_time_and_mem_parsing() -> None:
    assert commands.parse_hours("24h") == 24 and commands.parse_hours("1d12h") == 36
    assert commands.parse_hours("2-00:00:00") == 48 and commands.parse_hours("90") == 1.5
    assert commands.parse_mem_gb("64G") == 64 and commands.parse_mem_gb("512M") == 0.5
    assert commands.parse_mem_gb("1T") == 1024


def test_demo_usage_cost_and_unknown_gpu_type() -> None:
    doc = commands.run("usage", {"hosts": ["lyra"], "window": "90d", "by": "task"}, demo=True)
    assert doc.exit_code == EXIT_OK and doc.data["hosts_usage"][0]["rollup"]["by"] == "task"
    cost = commands.run("cost", {"host": "lyra", "partition": "gpuxl", "gpus": 1, "gpu_type": "L40S", "time": "24h",
                                 "cpus": 8}, demo=True)
    assert cost.exit_code == EXIT_OK and cost.data["su"] == 960
    with pytest.raises(commands.UsageError, match="'h999' is not a GPU type of lyra"):
        commands.run("cost", {"host": "lyra", "partition": "gpuxl", "gpus": 1, "gpu_type": "h999"}, demo=True)


def test_cli_and_mcp(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["--demo", "agent", "cost", "lyra", "--partition", "gpu", "--gpus", "2", "--time", "12h"])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[1].startswith("lyra gpu · 2× GPU · 1 cpus · 12h · qos normal")
    rc = cli.main(["--demo", "agent", "usage", "lyra", "--window", "7d", "--json"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["window"] == "7d"
    rc = cli.main(["--demo", "agent", "cost", "lyra", "--partition", "gpu", "--time", "0"])
    assert rc == 2 and "time: a positive duration" in capsys.readouterr().err
    runner = mcp.command_runner(config_path=None, demo=True, demo_seed=commands.DEMO_SEED, clock=None)
    s = mcp.Server(runner)
    s.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "estimate_cost", "arguments": {"host": "lyra", "partition": "gpuxl", "gpus": 1, "gpu_type": "l40s",
                                               "time": "24h", "cpus": 8}}})
    assert r is not None and r["result"]["structuredContent"]["su"] == 960 and not r["result"]["isError"]
    r = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "account_usage", "arguments": {"hosts": ["lyra"], "by": "partition", "limit": 2}}})
    assert r is not None and r["result"]["structuredContent"]["by"] == "partition"
    assert "`usage|cost` = account_usage, estimate_cost" in mcp.INSTRUCTIONS


# ====================================================================================================== cache
@pytest.fixture
async def cached(tmp_path: Path) -> tuple[DemoWorld, ManualClock, Any, Path]:
    """The demo fleet polled once at T0 into tmp/cache.sqlite (its periodic summaries included)."""
    clock = ManualClock(T0)
    world = DemoWorld(seed=7, now=clock.now, latency_scale=0.0)
    config = world.fleet_config()
    path = tmp_path / "cache.sqlite"
    cache = Cache(path)
    svc = FleetService(config, lambda h: DemoTransport(h, world), clock, cache)
    svc.lock_path = False
    await svc.collect_once(["k2", "lyra", "labstation"], timeout=30)
    await svc.stop()
    await cache.close()
    return world, clock, config, path


def _load(config: Any, path: Path, clock: ManualClock) -> FleetView:
    return source.load_view(config, ["k2", "lyra", "labstation"], cache_path=path, now=clock.now(), lock_base=None,
                            reports=True)


async def test_cache_only_then_refresh_upserts_the_report(cached: tuple[DemoWorld, ManualClock, Any, Path]) -> None:
    world, clock, config, path = cached
    fv = _load(config, path, clock)
    lyra_v = fv.host("lyra")
    assert lyra_v is not None and lyra_v.snapshot is not None and lyra_v.snapshot.account is not None
    assert lyra_v.caps_at is not None and lyra_v.account_reports == {}
    assert view.usage(fv).exit_code == EXIT_OK
    missing = view.usage(fv, window="90d")
    assert missing.exit_code == EXIT_NO_DATA
    assert {h["state"] for h in missing.data["hosts_usage"] if h["host"] != "labstation"} == \
        {"usage_window_not_cached"}
    made: list[str] = []

    def transports(cfg: HostConfig) -> Transport:
        made.append(cfg.name)
        return DemoTransport(cfg, world)

    got = await source.read_usage(config, lyra_v, "90d", clock=clock, timeout=30, cache_path=path,
                                  transports=transports)
    assert got is not None and got.window == "90d" and got.cells and made == ["lyra"]
    assert "90d" in source.read_rows(path, ["lyra"], reports=True).reports["lyra"]
    again = _load(config, path, clock)
    doc = view.usage(again, window="90d")
    lyra_u = next(h for h in doc.data["hosts_usage"] if h["host"] == "lyra")
    assert lyra_u["state"] == "ok" and lyra_u["rollup"]["rows"]
    clock.advance(60)                                    # < 10 min: the cached report, no probe
    reused = await source.read_usage(config, again.host("lyra"), "90d", clock=clock, timeout=30,  # type: ignore[arg-type]
                                     cache_path=path, transports=transports)
    assert reused is not None and reused.fetched_at == got.fetched_at and made == ["lyra"]


async def test_a_period_refresh_answered_as_30d_is_found_by_the_period_lookup(
        cached: tuple[DemoWorld, ManualClock, Any, Path]) -> None:
    """Review: `--window period --refresh` answered as 30d (no start) is stored under "period" too, so the next
    cache-only call finds it and another refresh within REPORT_MAX_AGE does not probe again."""
    world, clock, config, path = cached
    made: list[str] = []

    def transports(cfg: HostConfig) -> Transport:
        made.append(cfg.name)
        return DemoTransport(cfg, world)

    k2_v = _load(config, path, clock).host("k2")
    got = await source.read_usage(config, k2_v, "period", clock=clock, timeout=30, cache_path=path,  # type: ignore[arg-type]
                                  transports=transports)
    assert got is not None and got.window == "30d" and made == ["k2"]
    stored = source.read_rows(path, ["k2"], reports=True).reports["k2"]
    assert {"30d", "period"} <= set(stored)
    again = await source.read_usage(config, _load(config, path, clock).host("k2"), "period",  # type: ignore[arg-type]
                                    clock=clock, timeout=30, cache_path=path, transports=transports)
    assert again is not None and again.fetched_at == got.fetched_at and made == ["k2"]


async def test_a_session_host_waiting_for_a_login_is_not_probed(
        cached: tuple[DemoWorld, ManualClock, Any, Path]) -> None:
    _world, clock, config, path = cached
    hv = dataclasses.replace(_load(config, path, clock).host("lyra"), session_transport=True,  # type: ignore[arg-type]
                             state=ConnState.NEEDS_AUTH)

    def boom(cfg: HostConfig) -> Transport:
        raise AssertionError("no transport for a login-waiting session host")

    got = await source.read_usage(config, hv, "90d", clock=clock, timeout=30, cache_path=path, transports=boom)
    assert got is not None and got.errors[0].kind is ErrorKind.AUTH and got.cells == ()


def test_upsert_keeps_the_newest_report(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    sqlite3.connect(path).close()
    newer, older = report(fetched_at=T0), report(fetched_at=T0 - H, jobs_seen=99)
    upsert_account_report(path, newer)
    upsert_account_report(path, older)
    db = sqlite3.connect(path)
    at, = db.execute("SELECT at FROM account_reports WHERE host = 'lyra' AND window = '90d'").fetchone()
    db.close()
    assert at == T0.timestamp()


def test_usage_balances_come_from_the_newer_reading() -> None:
    """A 90d report fetched an hour ago next to a summary read a minute ago: the window's usage from the report, the
    balances (window-independent) from the summary."""
    newer = summary(read_at=T0 - timedelta(minutes=1), balances=(
        AccountBalance("pilab_02", 97885.0, 400.0, "since_reset", "sshare", "sshare"),))
    old = report(fetched_at=T0 - H, summary=summary(window="90d", read_at=T0 - H))
    doc = view.usage(fleet(lyra(account=newer, reports={"90d": old})), window="90d")
    pil = next(a for a in doc.data["hosts_usage"][0]["accounts"] if a["account"] == "pilab_02")
    assert pil["used_su"] == 400 and pil["me"]["cpu_h"] == 205.0
    assert doc.data["hosts_usage"][0]["summary_age_s"] == 60
