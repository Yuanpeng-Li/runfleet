"""[WP2] Section outcomes of the real `_meta*.txt` entries (rc/stderr → §4.3 outcome) and how collectors treat them:
STDERR_ONLY (lyra `--json | head`, rc 0) is a failure that never clears data; FAILED with output is still parsed."""

from __future__ import annotations

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.common import STALE_OUTCOMES, ParseResult, complete, section_error, usable
from runfleet.backends.slurm.collector import SlurmCollector
from runfleet.models import ErrorKind
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import ctx, fixture_result, outcome_for, result
from tests.backends.slurm.test_collector import squeue
from tests.conftest import META_FILES, SAMPLES_DIR, _parse_meta_file, meta, sample


def all_meta() -> list[tuple[str, str]]:
    out = []
    for host_dir in sorted(SAMPLES_DIR.iterdir()):
        for fname in META_FILES:
            path = host_dir / fname
            if path.is_file():
                out.extend((host_dir.name, name) for name in _parse_meta_file(path.read_text()))
    return sorted(set(out))


@pytest.mark.parametrize(("host", "probe"), all_meta())
def test_every_meta_entry_has_a_defined_outcome(host: str, probe: str) -> None:
    m = meta(host, probe)
    if m["rc"] is None:                                   # *_source entries record no command
        return
    try:
        out = sample(host, probe)
    except FileNotFoundError:
        out = ""
    oc = outcome_for(out.rstrip("\n"), m["rc"], m["stderr"])
    assert oc in SectionOutcome


def test_lyra_json_is_stderr_only() -> None:
    m = meta("lyra", "squeue_me_json")
    assert m["rc"] == 0 and "serializer_required" in m["stderr"] and sample("lyra", "squeue_me_json") == ""
    r = fixture_result("slurm.squeue", "lyra", "squeue_me_json")
    assert r.outcome is SectionOutcome.STDERR_ONLY and r.outcome in STALE_OUTCOMES
    assert not usable(r) and not complete(r)
    err = section_error(r, None)
    assert err is not None and err.kind is ErrorKind.REMOTE and "serializer_required" in err.summary
    # the collector keeps the previous jobs: STDERR_ONLY never clears data
    cx = ctx("k2")
    c = SlurmCollector(cx.host)
    c.ingest({"slurm.squeue": squeue()}, cx, MonitorView())
    out = c.ingest({"slurm.squeue": result("slurm.squeue", "", 0, m["stderr"])}, cx, MonitorView())
    assert out.jobs is not None and len(out.jobs) == 5 and all(j.source == "squeue" for j in out.jobs.values())


def test_lyra_empty_squeue_is_empty() -> None:
    r = fixture_result("slurm.squeue", "lyra", "squeue_me_O")
    assert r.outcome is SectionOutcome.EMPTY and complete(r)
    cx = ctx("lyra")
    out = SlurmCollector(cx.host).ingest({"slurm.squeue": r}, cx, MonitorView())
    assert out.jobs == {}


def test_failed_with_output_is_parsed() -> None:
    r = fixture_result("x.y", "k2", "commands")               # rc 127 (a missing jq) but useful stdout
    assert r.outcome is SectionOutcome.FAILED and usable(r) and not complete(r)
    listpids = fixture_result("x.y", "k2", "scontrol_listpids_job")
    assert listpids.outcome is SectionOutcome.FAILED and not usable(listpids)
    timeout = result("x.y", "", 124)
    assert timeout.outcome is SectionOutcome.TIMEOUT and section_error(timeout, None).kind is ErrorKind.TIMEOUT
    missing = result("x.y", "", 127, "missing:sstat")
    err = section_error(missing, None)
    assert err is not None and err.kind is ErrorKind.MISSING and "sstat" in err.summary
    assert section_error(result("x.y", "ok"), None) is None


def test_parse_result_ratio() -> None:
    pr: ParseResult[int] = ParseResult()
    assert not pr.mostly_bad
    pr.items.append(1)
    pr.bad(1, "x", "why", "7")
    assert not pr.mostly_bad and pr.bad_ids == {"7"}
    pr.bad(2, "y", "why")
    assert pr.mostly_bad
