"""[WP2] Pending priority evidence (§5.2.10): settings from `slurm.config`, the sprio / sshare parsers, the hints, and
the `slurm.priority` cadence group of SlurmCollector (when it exists, what it runs, how it ingests and fails)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm import priority as prio
from runfleet.backends.slurm.collector import PRIORITY_PERIOD, SlurmCollector
from runfleet.models import FairShare, Job, JobKey, JobKind, JobState, PendingPriority
from tests.backends.factories import NOW, SLURM_CONFIG, SLURM_TOOLS, at, ctx, fixture_result, result, to_sep
from tests.conftest import meta, sample

PRIO_TOOLS = SLURM_TOOLS | {"sprio", "sshare"}


def _config(host: str) -> dict[str, str]:
    """`slurm.config` of the host as detection reads it: the base keys + the priority keys (§5.1, §5.2.10)."""
    out = dict(SLURM_CONFIG[host])
    for line in sample(host, "slurm_config_priority").splitlines():
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _settings(host: str) -> prio.PrioritySettings:
    return prio.settings(_config(host))


def _job(job_id: str = "2334564", *, reason: str = "Priority", account: str = "campus01", held: bool = False,
         state: JobState = JobState.PENDING, array_job_id: str | None = None) -> Job:
    return Job(key=JobKey("k2", job_id), kind=JobKind.SLURM, name="smoke-test", user="alice0001", state=state,
               raw_state=state.name, job_id_raw=job_id, reason=reason, held=held, account=account,
               partition="gpu-preempt", array_job_id=array_job_id)


def _row(job_id: str = "2334564", part: str = "gpu-preempt", prio_value: float = 0.0, **factors: float) -> str:
    f = {name: factors.get(name, 0.0) for name in ("age", "assoc", "fairshare", "jobsize", "partition", "qos")}
    total = prio_value or sum(f.values()) or 1
    return (f"{job_id}|alice0001|{part}|{total:.0f}|" + "|".join(f"{v:.0f}" for v in f.values())
            + f"|{factors.get('tres', '') or ''}|{factors.get('nice', 0):.0f}")


# ---------------------------------------------------------------- settings (detection, §5.1)
def test_config_command_greps_the_priority_keys() -> None:
    for key in ("PriorityType", "PriorityWeight[A-Za-z]*", "PriorityFlags", "PriorityDecayHalfLife",
                "PriorityMaxAge", "PriorityUsageResetPeriod", "ClusterName", "PrivateData"):
        assert key in cmds.CONFIG_CMD
    assert meta("k2", "slurm_config_priority")["cmd"].split("grep -E ")[1].strip("'").startswith("^(PriorityType|")


def test_settings_from_the_fixtures() -> None:
    k2 = _settings("k2")
    assert k2.multifactor and k2.ranked
    assert dict(k2.weights) == {"age": 1000, "assoc": 0, "fairshare": 100000, "jobsize": 1000, "partition": 1000,
                                "qos": 0, "tres": 0}
    assert k2.half_life_s == 14 * 86400 and k2.max_age_s == 7 * 86400 and k2.reset == "NONE"
    assert k2.flags == frozenset()
    vega = _settings("vega")
    assert vega.ranked and vega.half_life_s == 0 and vega.reset == "MONTHLY"
    lyra = _settings("lyra")                         # priority/multifactor with every weight 0: not ranked
    assert lyra.multifactor and not lyra.ranked
    assert prio.settings({}).type is None           # capabilities detected before §5.2.10
    basic = prio.settings({"PriorityType": "priority/basic"})
    assert basic.type == "priority/basic" and not basic.ranked


def test_tres_weights_and_flags() -> None:
    st = prio.settings({"PriorityType": "priority/multifactor", "PriorityWeightTRES": "CPU=1000,Mem=2000,GRES/gpu=3000",
                        "PriorityFlags": "FAIR_TREE,MAX_TRES"})
    assert st.weight("tres") == 6000 and st.ranked and st.flags == {"FAIR_TREE", "MAX_TRES"}


def test_capabilities_carry_the_priority_keys() -> None:
    from runfleet.backends.slurm.collector import parse_capabilities
    text = sample("k2", "slurm_config") + sample("k2", "slurm_config_priority")
    caps = parse_capabilities({"slurm.config": result("slurm.config", text)})
    cfg = caps["slurm_config"]
    assert isinstance(cfg, dict) and cfg["PriorityType"] == "priority/multifactor"
    assert cfg["PriorityWeightFairShare"] == "100000" and cfg["PriorityFlags"] == ""
    assert prio.settings(cfg).ranked


# ---------------------------------------------------------------- parsers
def test_parse_sprio_fixtures() -> None:
    pr = prio.parse_sprio(sample("k2", "sprio"))
    assert not pr.bad_lines
    (row,) = pr.items
    assert (row.job_id, row.user, row.partition, row.priority) == ("2334564", "alice0001", "gpu-preempt", 1)
    assert set(row.factors.values()) == {0.0} and row.nice == 0      # empty %T (no TRES weights) = 0
    (vrow,) = prio.parse_sprio(sample("vega", "sprio")).items
    assert vrow.factors["fairshare"] == 10000 and vrow.factors["qos"] == 5000 and vrow.factors["age"] == 476


def test_parse_sprio_synthetic_rows() -> None:
    text = "\n".join([
        "Unable to find jobs matching user/id(s) specified",            # not a bad row
        "5|alice0001|gpu|2034.5|34.5|0|1000|0|1000|0|cpu=10,mem=2|100",  # floats, a TRES list, a nice value
        "6|alice0001|gpu|x|0|0|0|0|0|0||0",                              # bad number
        "7|alice0001|gpu|1",                                             # wrong field count
    ])
    pr = prio.parse_sprio(text)
    (row,) = pr.items
    assert row.priority == 2034.5 and row.factors["tres"] == 12 and row.nice == 100
    assert [b[0] for b in pr.bad_lines] == [3, 4] and pr.bad_ids == {"6"}
    assert prio.parse_sprio("Unable to find jobs matching user/id(s) specified\n").items == []


def test_parse_sshare_fixtures() -> None:
    (k2,) = prio.parse_sshare(sample("k2", "sshare")).items
    assert (k2.account, k2.user, k2.raw_shares, k2.raw_usage) == ("campus01", "alice0001", "1", 0)
    assert k2.fairshare == 0.95 and k2.level_fs is None                  # LevelFS inf (no usage)
    dept, pilab = prio.parse_sshare(sample("vega", "sshare")).items
    assert pilab.norm_shares == 0.5 and pilab.effective_usage == 0.01 and pilab.level_fs == 1.5
    assert pilab.raw_usage == 123456 and dept.account == "dept"
    bad = prio.parse_sshare("  lab|alice0001|1|0.5|0|0.0|zz|inf\nshort|row\n")
    assert len(bad.bad_lines) == 2 and bad.mostly_bad


# ---------------------------------------------------------------- evidence + hints
def _share(account: str = "campus01", fs: float | None = 0.03, level: float | None = 1.2,
           usage: int | None = 8_000_000) -> FairShare:
    return FairShare(account=account, user="alice0001", raw_shares="1", norm_shares=0.5, raw_usage=usage,
                     effective_usage=0.8, fairshare=fs, level_fs=level)


def test_evidence_fairshare_dominates_with_recovery_hint() -> None:
    rows = prio.parse_sprio(_row(fairshare=3000, age=2, jobsize=16)).items
    p = prio.evidence(_job(), rows, [_share()], _settings("k2"), sampled_at=NOW)
    assert p.ranked and p.priority == 3018 and p.partition == "gpu-preempt" and p.dominant == "fairshare"
    assert [f.name for f in p.factors] == ["fairshare", "jobsize", "age", "partition"]   # weight > 0, largest first
    assert p.factors[0].weight == 100000 and p.sampled_at == NOW and p.half_life_s == 14 * 86400
    assert p.hints[0].startswith("fairshare carries most of its priority (3,000 of 3,018)")
    assert ("fairshare 0.03: your group (campus01) used much more than its share recently; it recovers with a "
            "half-life of 14 days") in p.hints
    assert "waiting raises it: age adds up to 1,000 after 7 days in the queue" in p.hints


def test_evidence_says_when_you_used_more_than_your_share_and_usage_does_not_decay() -> None:
    p = prio.evidence(_job(account="pilab"), prio.parse_sprio(_row(fairshare=300, age=400)).items,
                      [_share("pilab", fs=0.03, level=0.4)], _settings("vega"), sampled_at=NOW)
    assert p.dominant == "age"
    assert ("fairshare 0.03: you used more than your share of pilab recently; usage does not decay on this cluster; "
            "it is reset monthly") in p.hints
    never = dataclasses.replace(_settings("vega"), reset="NONE")
    p = prio.evidence(_job(account="pilab"), [], [_share("pilab", fs=0.1)], never, sampled_at=None)
    assert any("PriorityDecayHalfLife 0" in h and "only by the admins" in h for h in p.hints)


def test_evidence_all_zero_factors_and_good_standing() -> None:
    """The k2 fixtures: priority 1, every factor 0; fair share 0.95 without usage."""
    shares = prio.parse_sshare(sample("k2", "sshare")).items
    p = prio.evidence(_job(), prio.parse_sprio(sample("k2", "sprio")).items, shares, _settings("k2"), sampled_at=NOW)
    assert p.priority == 1 and p.dominant is None
    assert p.hints[0].startswith("every factor is 0 for this job (priority 1)")
    assert "fairshare 0.95: no recent usage on campus01, so fair share works for you" in p.hints
    assert not any("start at" in h or "position" in h for h in p.hints)       # evidence, never a promise


def test_unranked_cluster_says_submit_order_instead_of_zero_factors() -> None:
    p = prio.evidence(_job(), prio.parse_sprio(sample("k2", "sprio")).items, [], _settings("lyra"), sampled_at=NOW)
    assert p == PendingPriority(ranked=False, hints=(prio.UNRANKED_HINT,), half_life_s=7 * 86400)
    assert p.factors == () and "submit order" in p.hints[0] and "backfill" in p.hints[0]


def test_evidence_other_reasons_holds_and_never_runs() -> None:
    st = _settings("k2")
    rows = prio.parse_sprio(_row(fairshare=3000)).items
    capped = prio.evidence(_job(reason="QOSMaxJobsPerUserLimit"), rows, [], st, sampled_at=NOW)
    assert capped.hints[0] == ("held back by per-user job cap (QOS), not by priority: these factors only count once "
                               "that clears")
    held = prio.evidence(_job(reason="JobHeldUser", held=True), rows, [], st, sampled_at=NOW)
    assert held.hints[0].startswith("held: it does not compete")
    never = prio.evidence(_job(reason="DependencyNeverSatisfied"), rows, [_share()], st, sampled_at=NOW)
    assert never.hints == ("priority does not matter here: as submitted, this job will never start",)
    assert never.factors                                                  # the evidence itself stays


def test_evidence_partitions_nice_and_missing_rows() -> None:
    st = _settings("k2")
    rows = prio.parse_sprio("\n".join([_row(part="gpu-preempt", fairshare=3000),
                                       _row(part="lab", fairshare=3000, partition=1000, nice=50)])).items
    p = prio.evidence(_job(), rows, [], st, sampled_at=NOW)
    assert p.partition == "lab" and p.priority == 4000 and p.others == (("gpu-preempt", 3000.0),)
    assert ("nice", -50.0) in [(f.name, f.value) for f in p.factors]
    assert any(h.startswith("nice 50 lowers its priority") for h in p.hints)
    later = prio.evidence(_job("2334999"), prio.rows_for(_job("2334999"), {"2334564": list(rows)}), [], st,
                          sampled_at=NOW)
    assert later.priority is None and later.factors == ()
    assert "not in the last priority reading yet (sprio runs at most every 10 min)" in later.hints
    summary = _job("500", array_job_id="500")
    assert prio.rows_for(summary, {"500": list(rows)}) == list(rows)
    tasks = prio.parse_sprio("500_7|alice0001|gpu|10|10|0|0|0|0|0||0\n500_8|alice0001|gpu|12|12|0|0|0|0|0||0").items
    by_id = prio.index_rows(tasks)
    assert [r.job_id for r in prio.rows_for(summary, by_id)] == ["500_7", "500_8"] and len(by_id["500_7"]) == 1


# ---------------------------------------------------------------- the collector's slurm.priority group
def _pctx(host: str = "k2", **kw: Any) -> Any:
    kw.setdefault("slurm_config", _config(host))
    kw.setdefault("tools", PRIO_TOOLS)
    return ctx(host, **kw)


def _with_jobs(c: SlurmCollector, cx: Any, host: str = "k2") -> None:
    """One poll of the squeue fixture (k2: 2334564 PD Priority; vega: 724479 PD QOSMaxJobsPerUserLimit)."""
    sq = fixture_result("slurm.squeue", host, "squeue_me_O")
    c.ingest({"slurm.squeue": dataclasses.replace(sq, stdout=to_sep(sq.stdout, cmds.SEP))}, cx, MonitorView())


def test_group_exists_only_while_an_own_job_pends_on_a_ranked_cluster() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    assert "slurm.priority" not in {cad.group for cad in c.cadences(cx)}        # no jobs yet
    _with_jobs(c, cx)
    (cad,) = [cad for cad in c.cadences(cx) if cad.group == "slurm.priority"]
    assert (cad.mode, cad.interval, cad.floor) == ("interval", PRIORITY_PERIOD, 600.0)
    assert cad.hidden_factor >= 6 and cad.max_backoff > cad.interval and not cad.focused_only
    for blocked in (_pctx(slurm_config=_config("lyra")),                       # every weight 0
                    _pctx(slurm_config={**_config("k2"), "PriorityType": "priority/basic"}),
                    _pctx(slurm_config=SLURM_CONFIG["k2"]),                    # old capabilities: unknown
                    _pctx(tools=SLURM_TOOLS)):                                 # neither sprio nor sshare
        assert "slurm.priority" not in {cad.group for cad in c.cadences(blocked)}
        assert c.sections(frozenset({"slurm.priority"}), blocked) == []


def test_group_ignores_held_and_running_jobs() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    _with_jobs(c, cx)
    c._live = {d: dataclasses.replace(j, held=j.state is JobState.PENDING) for d, j in c._live.items()}
    assert "slurm.priority" not in {cad.group for cad in c.cadences(cx)}


def test_sections_are_sprio_and_sshare_in_one_probe() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    _with_jobs(c, cx)
    sprio, sshare = c.sections(frozenset({"slurm.priority"}), cx)
    assert sprio.name == "slurm.sprio" and sshare.name == "slurm.sshare"
    assert sprio.cmd == 'sprio -h -u "$(id -un)" -o \'%i|%u|%r|%Y|%A|%B|%F|%J|%P|%Q|%T|%N\''
    assert sshare.cmd == ('sshare -h -U -u "$(id -un)" -P -o '
                          'Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare,LevelFS')
    for s in (sprio, sshare):
        assert s.timeout == 15 and s.max_stdout <= 512_000 and s.requires == (s.name.split(".")[1],)
    no_fs = _pctx(slurm_config={**_config("k2"), "PriorityWeightFairShare": "0"})
    assert [s.name for s in c.sections(frozenset({"slurm.priority"}), no_fs)] == ["slurm.sprio"]
    only_sshare = _pctx(tools=SLURM_TOOLS | {"sshare"})
    assert [s.name for s in c.sections(frozenset({"slurm.priority"}), only_sshare)] == ["slurm.sshare"]
    out = c.ingest({"slurm.sshare": fixture_result("slurm.sshare", "k2", "sshare")}, only_sshare, MonitorView())
    assert out.jobs is not None
    p = out.jobs["2334564"].priority
    assert p is not None and p.priority is None and p.hints[0] == prio.NO_SPRIO_HINT
    assert [s.account for s in p.fairshare] == ["campus01"]


def test_ingest_puts_the_evidence_on_own_pending_jobs() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    _with_jobs(c, cx)
    sq = fixture_result("slurm.squeue", "k2", "squeue_me_O")
    out = c.ingest({"slurm.squeue": dataclasses.replace(sq, stdout=to_sep(sq.stdout, cmds.SEP)),
                    "slurm.sprio": fixture_result("slurm.sprio", "k2", "sprio"),
                    "slurm.sshare": fixture_result("slurm.sshare", "k2", "sshare")}, cx, MonitorView())
    assert out.errors == () and out.jobs is not None
    p = out.jobs["2334564"].priority
    assert p is not None and p.ranked and p.priority == 1 and p.sampled_at == NOW
    assert [s.account for s in p.fairshare] == ["campus01"]
    assert all(j.priority is None for d, j in out.jobs.items() if j.state is not JobState.PENDING)
    # a failed reading keeps the last one and reports the section; the group backs off in the poller
    later = at(cx, 600)
    out = c.ingest({"slurm.squeue": dataclasses.replace(sq, stdout=to_sep(sq.stdout, cmds.SEP)),
                    "slurm.sprio": result("slurm.sprio", "", 1, "sprio: error: Socket timed out on send/recv"),
                    "slurm.sshare": result("slurm.sshare", "", 124, "")}, later, MonitorView())
    assert {e.section for e in out.errors} == {"slurm.sprio", "slurm.sshare"}
    assert out.jobs is not None and out.jobs["2334564"].priority == p


def test_unranked_cluster_explains_without_running_anything() -> None:
    """k2's jobs under lyra's priority setup (multifactor, every weight 0)."""
    unranked = {k: v for k, v in _config("lyra").items() if k.startswith("Priority")}
    cx = _pctx(slurm_config={**SLURM_CONFIG["k2"], **unranked})
    c = SlurmCollector(cx.host)
    sq = fixture_result("slurm.squeue", "k2", "squeue_me_O")
    out = c.ingest({"slurm.squeue": dataclasses.replace(sq, stdout=to_sep(sq.stdout, cmds.SEP))}, cx, MonitorView())
    assert out.jobs is not None
    pending = [j for j in out.jobs.values() if j.state is JobState.PENDING and j.is_mine]
    assert pending and all(j.priority is not None and not j.priority.ranked for j in pending)
    assert all(j.priority.hints == (prio.UNRANKED_HINT,) for j in pending if j.priority)
    assert "slurm.priority" not in {cad.group for cad in c.cadences(cx)}


def test_sprio_refusing_a_basic_cluster_stops_the_group() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    _with_jobs(c, cx)
    err = ("sprio: error: You are not running a supported priority plugin\n(priority/basic).\n"
           "Only 'priority/multifactor' is supported.")
    out = c.ingest({"slurm.sprio": result("slurm.sprio", "", 1, err)}, cx, MonitorView())
    assert out.errors == ()
    assert "slurm.priority" not in {cad.group for cad in c.cadences(cx)}


def test_sprio_finding_no_job_is_an_empty_reading_wherever_it_says_so() -> None:
    """The job started (or was cancelled) between squeue and sprio: sprio's "Unable to find jobs" — on stdout or, with
    a non-zero exit, on stderr — is an empty reading, never an error badge or a backoff."""
    cx = _pctx()
    msg = "Unable to find jobs matching user/id(s) specified"
    for res in (result("slurm.sprio", msg, 0), result("slurm.sprio", "", 1, msg), result("slurm.sprio", msg, 1)):
        c = SlurmCollector(cx.host)
        _with_jobs(c, cx)
        c._sprio = {"2334564": prio.parse_sprio(_row(fairshare=900)).items}       # an older reading
        out = c.ingest({"slurm.sprio": res}, at(cx, 600), MonitorView())
        assert out.errors == (), res
        assert c._sprio == {} and c._sprio_at == at(cx, 600).now


def test_mostly_bad_sprio_is_a_parse_error() -> None:
    cx = _pctx()
    c = SlurmCollector(cx.host)
    _with_jobs(c, cx)
    out = c.ingest({"slurm.sprio": result("slurm.sprio", "garbage\nmore garbage\n")}, cx, MonitorView())
    (e,) = out.errors
    assert e.section == "slurm.sprio" and e.kind.value == "parse"


@pytest.mark.parametrize("host", ["k2", "vega"])
def test_priority_fixtures_parse_cleanly(host: str) -> None:
    assert not prio.parse_sprio(sample(host, "sprio")).bad_lines
    assert not prio.parse_sshare(sample(host, "sshare")).bad_lines
    assert meta(host, "sprio")["rc"] == 0 and meta(host, "sshare")["rc"] == 0
