"""export.py (schema runfleet.snapshot/1) and the end-to-end `snapshot --demo` path on the real stack."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from runfleet import cli, export
from runfleet.models import ConnState, JobState
from tests.cli.fakes import SimpleStore, host_state, job

NOW = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


def store() -> SimpleStore:
    return SimpleStore({"k2": host_state("k2", jobs=[job("k2", "2332861"), job("k2", "2330539", JobState.FAILED)]),
                        "lyra": host_state("lyra", ConnState.RETRYING, with_snapshot=False, error="timed out"),
                        "lynx": host_state("lynx", ConnState.DEGRADED)})


def plain(obj: Any) -> Any:
    """A trivial encoder so the document shape is testable independently of models.to_json."""
    return getattr(obj, "name", None) or str(obj)


def test_document_shape_and_order() -> None:
    doc = export.build_snapshot(store(), generated_at=NOW, encode=plain)   # type: ignore[arg-type]
    assert list(doc) == ["schema", "generated_at", "hosts"] and doc["schema"] == "runfleet.snapshot/1"
    assert [h["name"] for h in doc["hosts"]] == ["k2", "lyra", "lynx"]
    k2 = doc["hosts"][0]
    assert list(k2) == ["name", "status", "caps", "jobs", "gpus", "procs", "nodes", "partitions", "account"]
    assert len(k2["jobs"]) == 2
    lyra = doc["hosts"][1]
    assert lyra["caps"] is None and lyra["jobs"] == [] and lyra["nodes"] == []


def test_real_encoder_round_trips_through_json() -> None:
    try:
        doc = export.build_snapshot(store(), generated_at=NOW, hosts=["lynx", "k2"])   # type: ignore[arg-type]
    except NotImplementedError:                              # pragma: no cover - before WP1's to_json lands
        pytest.skip("models.to_json not implemented yet")
    text = export.dumps(doc)
    assert text.endswith("}\n")
    back = json.loads(text)
    assert [h["name"] for h in back["hosts"]] == ["lynx", "k2"]
    assert back["generated_at"].startswith("2026-09-07T10:42:37")
    assert [j["key"]["id"] for j in back["hosts"][1]["jobs"]] == ["2330539", "2332861"]   # sorted by id
    assert back["hosts"][1]["status"]["state"] == "ok"


def test_failed_hosts_and_summary() -> None:
    s = store()
    assert export.failed_hosts(s) == ["lyra"]                                   # type: ignore[arg-type]
    s.hosts["off"] = host_state("off", ConnState.DISABLED, with_snapshot=False)
    s.hosts["flaky"] = host_state("flaky", ConnState.DEGRADED)
    s.hosts["flaky"].status = dataclasses.replace(s.hosts["flaky"].status, consecutive_failures=1)
    assert export.failed_hosts(s) == ["lyra", "flaky"]                          # type: ignore[arg-type]
    assert export.failed_hosts(s, ["k2", "lynx", "nope"]) == []                 # type: ignore[arg-type]
    rows = export.summary_rows(s)                                               # type: ignore[arg-type]
    assert rows[1] == ("k2", "ok", "1 active · 1 failed", "0", "0", "0.20s")
    assert rows[2][1] == "retrying (timed out)" and rows[2][2:5] == ("—", "—", "—")
    table = export.format_table(rows)
    assert table.splitlines()[0].split() == ["HOST", "STATE", "MY", "JOBS", "MY", "GPUS", "NODES", "LATENCY"]
    # leftover #5: a long error summary is cut so the other columns stay readable
    long = dataclasses.replace(s.hosts["flaky"].status, error=dataclasses.replace(
        s.hosts["lyra"].status.error, summary="x" * 80))                       # type: ignore[arg-type]
    s.hosts["flaky"].status = long
    cell = [r for r in export.summary_rows(s) if r[0] == "flaky"][0][1]       # type: ignore[arg-type]
    assert len(cell) == export.STATE_WIDTH and cell.endswith("…")


def test_snapshot_demo_end_to_end(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`runfleet snapshot --demo --json` on the real FleetService + collectors + DemoTransport (M2 path)."""
    monkeypatch.setattr(cli, "setup_logging", lambda debug, demo=False: None)
    try:
        rc = cli.main(["snapshot", "--demo", "--json", "--timeout", "30", "k2", "lyra", "labstation", "lynx"])
    except NotImplementedError as exc:                       # pragma: no cover - before WP1/WP2 land
        pytest.skip(f"core not implemented yet: {exc!r}")
    doc = json.loads(capsys.readouterr().out)
    hosts = {h["name"]: h for h in doc["hosts"]}
    assert rc == 4                                           # lynx starts NEEDS_AUTH in the demo scenario
    assert hosts["lynx"]["status"]["state"] == "needs_auth"
    assert hosts["k2"]["status"]["state"] in ("ok", "degraded") and hosts["k2"]["caps"]["slurm_version"] == "25.05.8"
    k2_ids = {j["key"]["id"] for j in hosts["k2"]["jobs"]}
    assert {"3370390", "3372094", "3372066", "3368404"} <= k2_ids
    assert any(j["key"]["id"] == "32377076_[5-12%2]" for j in hosts["lyra"]["jobs"])
    assert len(hosts["labstation"]["gpus"]) == 4
    assert any(j["key"]["id"].startswith("p363942-") for j in hosts["labstation"]["jobs"])
    task_logs = [lg for j in hosts["labstation"]["jobs"] if j["kind"] == "task" for lg in j["logs"]]
    assert task_logs and all(lg["size"] is not None and lg["mtime"] for lg in task_logs)   # M3 F9 residual


def test_summary_counts_my_jobs_and_my_gpus() -> None:
    """Critic: `snapshot` showed GPUS 0 for Slurm hosts where I held 14 GPUs and counted other users' tasks."""
    from runfleet.models import GPU, NodeAlloc, PartitionSummary, ResourceAlloc

    def alloc(gpus: int, idx: tuple[int, ...] = ()) -> ResourceAlloc:
        per = (NodeAlloc("n1", None, None, idx, None),) if idx else ()
        return ResourceAlloc(1, 8, None, gpus, None, {}, per)

    k2 = host_state("k2", jobs=[dataclasses.replace(job("k2", "1"), alloc=alloc(8)),
                                dataclasses.replace(job("k2", "2"), alloc=alloc(6)),
                                dataclasses.replace(job("k2", "3", JobState.PENDING), alloc=alloc(2))])
    assert k2.snapshot is not None
    k2.snapshot = dataclasses.replace(k2.snapshot, partitions=(PartitionSummary("p", 1, 8, 0, 0, {}, {}),))
    ws = host_state("lbs", jobs=[dataclasses.replace(job("lbs", "p1"), alloc=alloc(2, (0, 1))),
                                dataclasses.replace(job("lbs", "p2"), alloc=alloc(1, (2,)), is_mine=False)])
    assert ws.snapshot is not None
    gpu = GPU(host="lbs", index=0, uuid="u", name="A6000", util_pct=1, mem_used_mib=1, mem_total_mib=2,
              mem_util_pct=None, temp_c=None, power_w=None, power_limit_w=None, fan_pct=None, pstate=None,
              processes=(), source="nvidia-smi")
    ws.snapshot = dataclasses.replace(ws.snapshot, gpus=(gpu,) * 4,
                                      caps=dataclasses.replace(ws.snapshot.caps, profile=("process", "nvidia")))
    rows = {r[0]: r for r in export.summary_rows(SimpleStore({"k2": k2, "lbs": ws}))}   # type: ignore[arg-type]
    assert rows["k2"][2:4] == ("2 active · 1 queued", "14")
    assert rows["lbs"][2:4] == ("1 active", "2/4")
