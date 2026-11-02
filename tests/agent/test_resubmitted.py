"""[WP4] Resubmissions in the agent interface (§5.2.12, docs/AGENT_INTERFACE.md §3): `agent job` lineage line, JSON
`resubmitted_as` / `resumes`, `next` follows the latest job, `jobs` notes and --attention, `brief` counts and
attention, the `resubmitted` event and the PreemptMode wording of causes (`snapshot --json`:
tests/cli/test_export.py)."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import Any

from runfleet.agent import events as ev
from runfleet.agent import view
from runfleet.agent.data import FleetView
from runfleet.models import Capabilities, ConnState, Job, JobState, PartitionCharge
from tests.agent.test_view import H, LINK, T0, fleet, mk_host, mk_job

M = timedelta(minutes=1)
NAME = "orb-swp--train--places-cosinewarm--s44--a01"


def lyra_job(jid: str, state: JobState, submit_ago: timedelta, end_ago: timedelta | None = None, **kw: Any) -> Job:
    d: dict[str, Any] = dict(name=NAME, submit_time=T0 - submit_ago, partition="free-gpu")
    if end_ago is not None:
        d.update(start_time=T0 - submit_ago + M, end_time=T0 - end_ago)
    d.update(kw)
    return mk_job("lyra", jid, state, **d)


def preempted_pair() -> list[Job]:
    """The golden pair: preempted (PreemptMode CANCEL), resubmitted 13 min later under the same name, completed."""
    return [lyra_job("31354789", JobState.PREEMPTED, 9 * H, 6 * H),
            lyra_job("31358335", JobState.COMPLETED, 6 * H - 13 * M, 2 * H)]


def failed_chain(last: JobState = JobState.COMPLETED) -> list[Job]:
    """F → F → F → `last`: lyra 31327454 … 31327999."""
    ids = ("31327454", "31327620", "31327915", "31327999")
    out = [lyra_job(jid, JobState.FAILED, (12 - 2 * i) * H, (11 - 2 * i) * H) for i, jid in enumerate(ids[:3])]
    end = None if last is JobState.RUNNING else 3 * H
    return [*out, lyra_job(ids[3], last, 6 * H, end)]


def lyra(jobs: list[Job], preempt_mode: str | None = "CANCEL") -> FleetView:
    hv = mk_host("lyra", jobs)
    assert hv.snapshot is not None and hv.snapshot.caps is not None
    charge = PartitionCharge("free-gpu", preempt_mode=preempt_mode)
    caps: Capabilities = dataclasses.replace(hv.snapshot.caps, partition_charges=(charge,))
    return fleet(dataclasses.replace(hv, snapshot=dataclasses.replace(hv.snapshot, caps=caps)))


# ---------------------------------------------------------------- agent job
def test_card_of_a_superseded_job_names_its_successor_and_follows_it() -> None:
    doc = view.job(lyra(preempted_pair()), [("lyra", "31354789")])
    lines = doc.text.splitlines()
    result = next(i for i, ln in enumerate(lines) if ln.startswith("  result"))
    assert "killed (PreemptMode CANCEL) — already resubmitted as lyra:31358335" in lines[result]
    assert "— resubmit" not in lines[result]                         # the diagnosis stays, the advice does not
    assert lines[result + 1] == "  lineage  resubmitted as lyra:31358335 (COMPLETED, submitted 08:15Z)"
    nxt = [ln for ln in lines if ln.startswith("  next")]
    assert nxt == ["  next     resubmitted → runfleet agent job lyra:31358335"]       # never "read the log / resubmit"
    j = doc.data["jobs"][0]
    assert j["resubmitted_as"] == {"key": "lyra:31358335", "id": "lyra:31358335", "state": "CD", "group": "done",
                                   "latest": None}
    assert j["resumes"] is None
    assert "PreemptMode CANCEL" in j["exit"]["cause"] and "already resubmitted as" in j["exit"]["cause"]


def test_card_of_the_successor_says_what_it_resumes() -> None:
    doc = view.job(lyra(preempted_pair()), [("lyra", "31358335")])
    assert "  lineage  resumes lyra:31354789 (PREEMPTED)" in doc.text.splitlines()
    j = doc.data["jobs"][0]
    assert j["resubmitted_as"] is None
    assert j["resumes"] == {"key": "lyra:31354789", "id": "lyra:31354789", "state": "PR", "group": "failed"}
    assert sorted(j["resumes"]) == LINK


def test_a_longer_chain_names_the_latest_job() -> None:
    fv = lyra(failed_chain())
    doc = view.job(fv, [("lyra", "31327454")])
    line = next(ln for ln in doc.text.splitlines() if ln.startswith("  lineage"))
    assert line.startswith("  lineage  resubmitted as lyra:31327620 (FAILED, submitted ")
    assert line.endswith(" · latest lyra:31327999 (COMPLETED)")
    assert "  next     resubmitted → runfleet agent job lyra:31327999" in doc.text
    rs = doc.data["jobs"][0]["resubmitted_as"]
    assert rs["id"] == "lyra:31327620" and rs["latest"]["id"] == "lyra:31327999" and rs["latest"]["state"] == "CD"
    middle = view.job(fv, [("lyra", "31327620")])
    assert "resumes lyra:31327454 (FAILED)" in middle.text and "latest lyra:31327999" in middle.text


def test_the_lineage_line_survives_the_card_budget() -> None:
    long = "x" * 150
    jobs = [lyra_job("31354789", JobState.PREEMPTED, 9 * H, 6 * H, hints=tuple(f"hint {i} {long}" for i in range(60)),
                     nodes=tuple(f"node-{i:03d}" for i in range(400)), workdir="/w/" + "d" * 230),
            lyra_job("31358335", JobState.RUNNING, 6 * H - 13 * M, workdir="/w/" + "d" * 230)]
    doc = view.job(lyra(jobs), [("lyra", "31354789")])
    assert len(doc.text.encode()) <= view.BUDGETS["job"][0]
    assert "  lineage  resubmitted as lyra:31358335 (RUNNING" in doc.text
    assert "runfleet agent job lyra:31358335" in doc.text
    assert doc.data["jobs"][0]["resubmitted_as"]["id"] == "lyra:31358335"         # _fit_job_json keeps it


def test_requeue_wording_and_no_link_without_a_later_job() -> None:
    doc = view.job(lyra(preempted_pair()[:1], preempt_mode="REQUEUE"), [("lyra", "31354789")])
    assert "not requeued (PreemptMode REQUEUE" in doc.text
    assert "lineage" not in doc.text and doc.data["jobs"][0]["resubmitted_as"] is None
    assert "runfleet agent log lyra:31354789" in doc.text                         # the usual next


# ---------------------------------------------------------------- agent jobs
def test_jobs_rows_note_the_resubmission_and_attention_skips_it() -> None:
    fv = lyra(preempted_pair())
    doc = view.jobs(fv, view.JobFilter(states=view.parse_states("finished")))
    row = next(ln for ln in doc.text.splitlines() if ln.startswith("lyra:31354789"))
    assert row.endswith("· resubmitted as 31358335 (COMPLETED)")
    succ = next(ln for ln in doc.text.splitlines() if ln.startswith("lyra:31358335"))
    assert succ.endswith("· resumes 31354789 (PREEMPTED)")
    by_id = {j["id"]: j for j in doc.data["jobs"]}
    assert by_id["lyra:31354789"]["resubmitted_as"]["id"] == "lyra:31358335"
    assert by_id["lyra:31358335"]["resumes"]["id"] == "lyra:31354789"
    att = view.jobs(fv, view.JobFilter(attention_only=True))
    assert att.data["jobs"] == []                                  # the failure is carried on: nothing to look at


def test_attention_moves_to_the_latest_job_when_it_failed_too() -> None:
    fv = lyra(failed_chain(JobState.TIMEOUT))
    att = view.jobs(fv, view.JobFilter(attention_only=True))
    assert [j["id"] for j in att.data["jobs"]] == ["lyra:31327999"]
    every = view.jobs(fv, view.JobFilter(states=view.parse_states("failed")))
    assert len(every.data["jobs"]) == 4                             # history is kept: all four rows


# ---------------------------------------------------------------- brief
def test_brief_counts_stay_and_attention_drops_resubmitted_failures() -> None:
    lone = mk_job("lyra", "31300001", JobState.OUT_OF_MEMORY, name="other-run", submit_time=T0 - 5 * H,
                  start_time=T0 - 5 * H, end_time=T0 - 4 * H)
    doc = view.brief(lyra([*preempted_pair(), lone]))
    counts = doc.data["counts"]
    assert counts["failed_recent"] == 2 and counts["resubmitted_recent"] == 1
    assert "failed/24h 2 (1 resubmitted) · done/24h 1" in doc.text
    assert [a["job"] for a in doc.data["attention"]] == ["lyra:31300001"]
    plain = view.brief(lyra([lone]))
    assert plain.data["counts"]["resubmitted_recent"] == 0 and "resubmitted" not in plain.text


def test_brief_resubmitted_recent_is_unknown_without_job_data() -> None:
    fv = fleet(mk_host("lyra", snapshot=False, state=ConnState.NEEDS_AUTH))
    assert view.brief(fv).data["counts"]["resubmitted_recent"] is None


# ---------------------------------------------------------------- events
def test_resubmitted_event_when_the_link_became_true() -> None:
    fv = lyra(preempted_pair())
    out, _cov = ev.events_since(fv, T0 - 12 * H)
    by_id = {e.id: e for e in out}
    e = by_id["lyra:31358335:resubmitted"]
    assert e.kind == "resubmitted" and e.at == T0 - 6 * H + 13 * M and e.job == "lyra:31358335"
    assert e.text == "resumes lyra:31354789 (PREEMPTED)" and e.state == "CD" and e.name == NAME
    assert by_id["lyra:31354789:ended_bad"].text == "PREEMPTED"              # the bad end is reported as before
    assert "resubmitted" in ev.KINDS and set(ev.KINDS) <= set(view._EVENT_PRIO)
    doc = view.events(fv, T0 - 12 * H)
    assert any(x["kind"] == "resubmitted" for x in doc.data["events"])
    assert "resubmitted  lyra:31358335" in doc.text
    # outside the window: no event (the later of the resubmit and the bad ending decides)
    assert not any(x.kind == "resubmitted" for x in ev.events_since(fv, T0 - 5 * H)[0])



def test_resubmitted_event_of_a_segment_queued_ahead_is_dated_at_the_bad_ending() -> None:
    """A singleton segment is submitted long before the run it carries on is preempted: the link becomes true at the
    preemption, so a rolling `--since 90m` poll sees it (dated then), never only an `--since 11h` one."""
    old = lyra_job("100", JobState.PREEMPTED, 10 * H, 1 * H, name="orb-chain--s44")
    new = lyra_job("101", JobState.RUNNING, 10 * H - M, None, name="orb-chain--s44", start_time=T0 - 59 * M)
    out, _cov = ev.events_since(lyra([old, new]), T0 - 90 * M)
    e = {x.id: x for x in out}["lyra:101:resubmitted"]
    assert e.at == T0 - 1 * H and e.text == "resumes lyra:100 (PREEMPTED)"
