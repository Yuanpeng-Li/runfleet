"""[WP2] Account usage (§5.2.11): the `slurm.acct_*` command builders and the pure parsers / derivation rules of
`backends/slurm/account.py`, from the anonymised account field probes (acct_share, acct_sreport, acct_sreport_totals,
lyra/acct_sbank, acct_jobs, k2/acct_jobs_too_wide)."""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from runfleet.backends.slurm import account as acct
from runfleet.backends.slurm import commands as cmds
from runfleet.backends.slurm.collector import parse_qos, partition_charges
from runfleet.models import AccountBalance, Capabilities, ErrorKind, JobState
from runfleet.probe.section import SectionOutcome
from runfleet.util.cost import classify_partitions
from tests.backends.factories import NOW, TZ, USER, caps, fixture_result, result
from tests.conftest import sample

SEP = cmds.SEP
LYRA_ACCOUNTS = ["alice07", "pilab_02", "pilab_02_gpu", "pilab_03"]
# PriorityDecayHalfLife / PriorityUsageResetPeriod of the account field probes (V1)
DECAY = {"lyra": ("00:00:00", "NONE"), "k2": ("7-00:00:00", "NONE"), "vega": ("14-00:00:00", "MONTHLY")}


def account_caps(host: str, **overrides: object) -> Capabilities:
    half, reset = DECAY[host]
    c = caps(host)
    qos = dict(parse_qos(sample(host, "sacctmgr_qos")).items)
    charges = partition_charges(sample("lyra", "scontrol_partitions_charges")) if host == "lyra" else ()
    return dataclasses.replace(c, slurm_config={**c.slurm_config, "PriorityDecayHalfLife": half,
                                                "PriorityUsageResetPeriod": reset},
                               qos_usage_factor=qos, partition_charges=charges, **overrides)  # type: ignore[arg-type]


def jobs_of(host: str) -> list[acct.AcctJob]:
    pr = acct.parse_acct_jobs(sample(host, "acct_jobs"), tz=TZ[host], sep=SEP)
    assert not pr.bad_lines
    return pr.items


# ---------------------------------------------------------------- commands
def test_share_command_quotes_one_account_list() -> None:
    assert cmds.acct_share_cmd(["pilab_02", "alice07"]) == \
        "sshare -h -P -A pilab_02,alice07 -o Account,User,RawUsage,GrpTRESMins,GrpTRESRaw"


@pytest.mark.parametrize("build", [
    lambda a: cmds.acct_share_cmd(a),
    lambda a: cmds.acct_sreport_cmd(a, "now-30days", totals=True),
    lambda a: cmds.acct_sbank_cmd(a),
])
@pytest.mark.parametrize("bad", [["lab;rm -rf ~"], ["a b"], ["$(id)"], ["lab", "x'y"], [""], [],
                                 [f"a{i}" for i in range(cmds.MAX_ACCOUNTS + 1)]])
def test_builders_reject_bad_account_lists(build: object, bad: list[str]) -> None:
    with pytest.raises(ValueError):
        build(bad)  # type: ignore[operator]


def test_sreport_commands() -> None:
    mine = cmds.acct_sreport_cmd(["pilab_02"], "now-30days", totals=False)
    assert mine == ('sreport -n -P cluster AccountUtilizationByUser user="$(id -un)" start=now-30days end=now '
                    "-t Minutes --tres=cpu,gres/gpu,billing | cut -d'|' -f1-3,5-")
    totals = cmds.acct_sreport_cmd(["pilab_02", "alice07"], "2026-01-14", totals=True)
    assert "accounts=pilab_02,alice07 start=2026-01-14 end=now" in totals
    assert totals.endswith("| awk -F'|' -v me=\"$(id -un)\" '$3 == \"\" || $3 == me' | cut -d'|' -f1-3,5-")
    for start in ("now-30days;id", "2026-1-14", "yesterday", "now-30 days", "$(date)"):
        with pytest.raises(ValueError):
            cmds.acct_sreport_cmd(["pilab_02"], start, totals=False)


def test_sbank_command_is_one_call_per_account() -> None:
    assert cmds.acct_sbank_cmd(["pilab_02", "alice07"]) == (
        'for a in pilab_02 alice07; do echo "@@acct $a"; sbank balance statement -a "$a" 2>&1; done; true')


def test_jobs_command_and_chunks() -> None:
    c = cmds.acct_jobs_cmd(60, 30)
    assert c.startswith('sacct -u "$(id -un)" -S now-60days -E now-30days -X -n -P --delimiter=')
    assert "--format=JobIDRaw,Account,Partition,QOS,State,ElapsedRaw,Start,End,AllocTRES,JobName" in c
    assert c.endswith(f"| awk -F{shlex.quote(SEP)} '$6 != \"0\"'")
    assert "-E now -X" in cmds.acct_jobs_cmd(7, 0)
    for older, newer in ((31, 0), (30, 30), (10, 20), (5, -1)):
        with pytest.raises(ValueError):
            cmds.acct_jobs_cmd(older, newer)
    assert cmds.acct_chunks(7) == [(7, 0)]
    assert cmds.acct_chunks(30) == [(30, 0)]
    assert cmds.acct_chunks(90) == [(90, 60), (60, 30), (30, 0)]
    assert cmds.acct_chunks(100) == [(100, 90), (90, 60), (60, 30), (30, 0)]
    assert len(cmds.acct_chunks(390)) == 13
    with pytest.raises(ValueError):
        cmds.acct_chunks(0)


@pytest.mark.posix
def test_remote_filters_keep_only_mine_and_drop_proper_name(tmp_path: Path) -> None:
    """The host-side awk/cut of the commands, run against stub sreport / sacct / id: other members' rows and the
    Proper Name column never leave the host; never-started rows are dropped."""
    bin_dir = os.fspath(tmp_path)
    stubs = {
        "id": "echo alice07",
        "sreport": ("printf '%s\\n' 'lyra|pilab_02|||cpu|100' 'lyra|pilab_02|alice07|Alice A|cpu|40' "
                    "'lyra|pilab_02|bob|Bob B|cpu|60'"),
        "sacct": "printf '1\\037a\\037p\\037q\\037RUNNING\\0370\\037x\\037y\\037z\\037n\\n"
                 "2\\037a\\037p\\037q\\037COMPLETED\\03712\\037x\\037y\\037z\\037n\\n'",
    }
    for name, body in stubs.items():
        path = os.path.join(bin_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"#!/bin/sh\n{body}\n")
        os.chmod(path, 0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}

    def run(cmd: str) -> str:
        return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, env=env, check=True).stdout

    assert run(cmds.acct_sreport_cmd(["pilab_02"], "now-30days", totals=True)).split("\n")[:-1] == [
        "lyra|pilab_02||cpu|100", "lyra|pilab_02|alice07|cpu|40"]
    assert run(cmds.acct_sreport_cmd(["pilab_02"], "now-30days", totals=False)).split("\n")[0] == \
        "lyra|pilab_02||cpu|100"
    assert run(cmds.acct_jobs_cmd(30, 0)).split("\n")[:-1] == [SEP.join(
        ["2", "a", "p", "q", "COMPLETED", "12", "x", "y", "z", "n"])]


# ---------------------------------------------------------------- accounts
def test_accounts_of_this_cluster() -> None:
    c = caps("lyra", associations=(("lyra", "pilab_02", "", "low"), ("lyra", "alice07", "", "low"),
                                   ("lyra", "pilab_02", "gpu", "normal"), ("other", "far", "", "normal"),
                                   ("", "plain", "", "normal"), ("lyra", "bad name", "", "low")))
    assert acct.accounts(c) == ["alice07", "pilab_02", "plain"]
    many = caps("k2", associations=tuple(("k2", f"acct{i:02d}", "", "normal") for i in range(15)))
    assert acct.accounts(many) == [f"acct{i:02d}" for i in range(cmds.MAX_ACCOUNTS)]
    assert acct.accounts(None) == []


# ---------------------------------------------------------------- sshare
def test_share_account_rows_and_own_rows() -> None:
    pr = acct.parse_share(sample("lyra", "acct_share"))
    assert not pr.bad_lines and len(pr.items) == 8
    rows = {(r.account, r.user): r for r in pr.items}
    assert set(rows) == {(a, u) for a in LYRA_ACCOUNTS for u in ("", "alice07")}   # leading space stripped
    head = rows[("pilab_02", "")]
    assert head.grp_tres_mins == {"billing": 5873100.0} and head.grp_tres_raw["billing"] == 17887.0
    assert head.raw_usage == 1073256 and head.grp_tres_raw["gres/gpu:rtx6000"] == 0.0
    assert rows[("pilab_02", "alice07")].grp_tres_mins == {}


@pytest.mark.parametrize("host", ["k2", "vega"])
def test_share_without_limits(host: str) -> None:
    pr = acct.parse_share(sample(host, "acct_share"))
    assert not pr.bad_lines and [r.user for r in pr.items] == ["", USER[host]]
    assert all(r.grp_tres_mins == {} for r in pr.items)


def test_share_bad_rows() -> None:
    pr = acct.parse_share("lab||1|billing=60|billing=6\nlab|u|1\nla b||1||\n|u|1||\n")
    assert len(pr.items) == 1 and len(pr.bad_lines) == 3 and pr.mostly_bad


# ---------------------------------------------------------------- balances
def test_lyra_balances_from_sshare_with_sbank_dates() -> None:
    c = account_caps("lyra")
    share = acct.parse_share(sample("lyra", "acct_share")).items
    bank = acct.parse_sbank(sample("lyra", "acct_sbank"))
    got = {b.account: b for b in acct.balances(LYRA_ACCOUNTS, share, bank, c, read_at=NOW)}
    lab = got["pilab_02"]
    assert lab.limit_su == 97885 and lab.used_su == pytest.approx(298.12, abs=0.01)   # = sbank 298 / 97,885
    assert (lab.limit_source, lab.used_source, lab.used_basis) == ("sshare", "sshare", "since_reset")
    assert lab.available_su == pytest.approx(97586.88, abs=0.01)
    assert lab.next_realloc == date(2027, 1, 14) and lab.realloc_known and lab.read_at == NOW
    assert got["pilab_02_gpu"].limit_su == 97 and got["pilab_02_gpu"].used_su == 0
    assert got["pilab_03"].limit_su == 100000 and got["pilab_03"].used_su == pytest.approx(2646.7, abs=0.1)
    own = got["alice07"]
    assert own.limit_su == 1000 and own.used_su == 0 and own.next_realloc is None and own.realloc_known
    assert [b.account for b in acct.balances(LYRA_ACCOUNTS, share, bank, c)] == sorted(LYRA_ACCOUNTS)
    assert acct.mode(got.values()) == "limits"


def test_balances_fall_back_to_sbank() -> None:
    bank = acct.parse_sbank(sample("lyra", "acct_sbank"))
    (lab,) = acct.balances(["pilab_02"], None, bank, account_caps("lyra"), read_at=NOW)
    assert (lab.used_su, lab.limit_su, lab.used_source, lab.limit_source) == (298, 97885, "sbank", "sbank")
    assert lab.next_realloc == date(2027, 1, 14)
    (none,) = acct.balances(["pilab_02"], None, None, None)
    assert none == AccountBalance("pilab_02")
    assert none.available_su is None


@pytest.mark.parametrize(("host", "account", "used", "basis", "half", "reset"), [
    ("k2", "campus01", 150198226 / 60, "decayed", 7 * 86400.0, None),
    ("vega", "pilab", 345165 / 60, "decayed", 14 * 86400.0, "MONTHLY"),
])
def test_usage_only_balances(host: str, account: str, used: float, basis: str, half: float,
                             reset: str | None) -> None:
    c = account_caps(host)
    share = acct.parse_share(sample(host, "acct_share")).items
    (b,) = acct.balances([account], share, None, c, read_at=NOW)
    assert b.limit_su is None and b.limit_source is None and b.used_su == pytest.approx(used)
    assert b.used_basis == basis and b.other_limits == {} and acct.mode([b]) == "usage_only"
    assert acct.half_life_s(c) == half and acct.reset_period(c) == reset


def test_usage_only_keeps_the_users_own_row_apart_from_the_accounts() -> None:
    """Review: on k2 the account row's GrpTRESRaw billing is the whole university account (2,503,304
    billing-h), the user's own row 5,844.7; `my_used_su` keeps the user's so "recent" next to `me` is the user's."""
    share = acct.parse_share(sample("k2", "acct_share")).items
    (b,) = acct.balances(["campus01"], share, None, account_caps("k2"), read_at=NOW)
    assert b.used_su == pytest.approx(150198226 / 60) and b.my_used_su == pytest.approx(350681 / 60)
    (no_rows,) = acct.balances(["campus01"], [r for r in share if not r.user], None, account_caps("k2"))
    assert no_rows.my_used_su is None


def test_basis_unknown_and_other_limits() -> None:
    c = caps("k2")                                          # no priority settings known
    assert acct.used_basis(c) == "unknown" and acct.half_life_s(None) is None and acct.reset_period(c) is None
    share = acct.parse_share("lab||0|cpu=600000,gres/gpu=120000|billing=30,gres/gpu=6\n").items
    (b,) = acct.balances(["lab"], share, None, c)
    assert b.limit_su is None and b.used_su == 0.5
    assert b.other_limits == {"cpu": 10000.0, "gres/gpu": 2000.0}              # hours, never SU
    assert acct.mode([b]) == "usage_only"


# ---------------------------------------------------------------- sbank
def test_sbank_fixture() -> None:
    bank = acct.parse_sbank(sample("lyra", "acct_sbank"))
    assert set(bank) == set(LYRA_ACCOUNTS)
    assert bank["pilab_03"] == acct.SbankBalance("pilab_03", 2643, 100000, date(2027, 4, 2), True)
    assert bank["alice07"] == acct.SbankBalance("alice07", 0, 1000, None, True)   # "No data found."
    assert all(b.error is None for b in bank.values())


def test_sbank_errors_stay_per_account() -> None:
    text = ("@@acct Lab_A\nAccount Usage | Account Limit Available (SUs)\nLAB_A  1,234.5 |  2,000  765.5\n"
            "@@acct lab_b\nbash: line 3: sbank: command not found\n"
            "@@acct lab_c\nsomething | new | layout\nNext reallocation: 2027-02-30\n"
            "@@acct lab_d\nNo data found.\n")
    bank = acct.parse_sbank(text)
    assert bank["lab_a"] == acct.SbankBalance("Lab_A", 1234.5, 2000)
    assert bank["lab_b"].error is not None and "command not found" in bank["lab_b"].error
    assert bank["lab_c"].error is not None and bank["lab_c"].next_realloc is None
    assert bank["lab_d"] == acct.SbankBalance("lab_d", realloc_known=True)
    (b,) = acct.balances(["lab_b"], None, bank, None)
    assert b.used_su is None and b.limit_su is None                       # an error row is not used
    assert acct.parse_sbank("") == {} and acct.parse_sbank("garbage before any block\n") == {}


# ---------------------------------------------------------------- sreport
def test_sreport_my_rows() -> None:
    pr = acct.parse_sreport(sample("lyra", "acct_sreport"))
    assert not pr.bad_lines and len(pr.items) == 9
    mine, totals = acct.sreport_usage(pr.items, cluster="lyra")
    assert totals == ()
    got = {u.account: u for u in mine}
    assert set(got) == {"alice07", "pilab_02", "pilab_03"}                  # pilab_02_gpu: no use, no rows
    assert got["pilab_02"].cpu_h == pytest.approx(298.13, abs=0.01) and got["pilab_02"].gpu_h == 0
    assert got["alice07"].gpu_h == pytest.approx(1154.92, abs=0.01)
    assert all(u.scope == "me" and u.source == "sreport" for u in mine)


@pytest.mark.parametrize(("host", "account", "cpu_h"), [("lyra", "pilab_02", 1316.33), ("k2", "campus01", 7436273.3),
                                                        ("vega", "pilab", 10726.48)])
def test_sreport_account_totals(host: str, account: str, cpu_h: float) -> None:
    pr = acct.parse_sreport(sample(host, "acct_sreport_totals"))
    assert not pr.bad_lines
    start = NOW - timedelta(days=30)
    mine, totals = acct.sreport_usage(pr.items, cluster=host, starts={account: start})
    tot = {u.account: u for u in totals}
    assert tot[account].scope == "account" and tot[account].cpu_h == pytest.approx(cpu_h, abs=0.01) and tot[account].start == start
    assert {u.account for u in mine} == set(tot)


def test_sreport_other_clusters_and_bad_rows() -> None:
    text = ("lyra|lab|alice07|cpu|60\nfar|lab|alice07|cpu|600\nlyra|lab|alice07|energy|5\n"
            "lyra|lab|alice07|Alice A|cpu|60\nlyra|lab|alice07|cpu|-1\n")
    pr = acct.parse_sreport(text)
    assert len(pr.items) == 3 and len(pr.bad_lines) == 2                    # a Proper Name column is never read
    (use,), () = acct.sreport_usage(pr.items, cluster="lyra")
    assert (use.cpu_h, use.gpu_h, use.billing_h) == (1.0, None, None)
    (both,), () = acct.sreport_usage(pr.items)                              # cluster unknown: every row counts
    assert both.cpu_h == 11.0


# ---------------------------------------------------------------- sacct chunks
def test_acct_jobs_fixtures() -> None:
    k2 = {j.job_id_raw: j for j in jobs_of("k2")}
    assert len(k2) == 6                                                     # the two never-started rows were dropped
    j = k2["2329557"]
    assert (j.account, j.partition, j.qos, j.state, j.elapsed_s) == ("campus01", "gpu-preempt", "gpu-priority",
                                                                     JobState.COMPLETED, 11606)
    assert j.start == datetime.fromtimestamp(1788537330, UTC) and j.name == "eval_b_r2"
    assert j.tres["gres/gpu:nvidia_h100_nvl_1g.12gb"] == "1"
    assert len(jobs_of("lyra")) == 7 and len(jobs_of("vega")) == 6


def test_acct_jobs_rows() -> None:
    row = SEP.join(["41", "lab", "gpu", "normal", "RUNNING", "7200", "1788770000", "Unknown",
                    "billing=40,cpu=8,gres/gpu=1", f"name{SEP}with{SEP}sep"])
    never = SEP.join(["42", "lab", "gpu", "normal", "CANCELLED by 1", "0", "None", "1788770000", "", "x"])
    short = SEP.join(["43", "lab", "gpu"])
    bad_state = SEP.join(["44", "lab", "gpu", "normal", "running?", "5", "1788770000", "", "", "x"])
    no_start = SEP.join(["45", "lab", "gpu", "normal", "COMPLETED", "5", "None", "", "", "x"])
    pr = acct.parse_acct_jobs("\n".join([row, never, short, bad_state, no_start]) + "\n", tz=TZ["k2"])
    (job,) = pr.items
    assert job.name == f"name{SEP}with{SEP}sep" and job.state is JobState.RUNNING and job.end is None
    assert len(pr.bad_lines) == 3 and pr.bad_ids == {"44", "45"} and pr.mostly_bad


def test_merge_jobs_dedups_chunk_boundaries() -> None:
    old = acct.AcctJob("7", "lab", "gpu", "normal", JobState.COMPLETED, 3600, NOW, NOW, {}, "a")
    new = dataclasses.replace(old, elapsed_s=7200)
    other = dataclasses.replace(old, job_id_raw="8")
    assert acct.merge_jobs([[old, other], [new]]) == [new, other]
    assert acct.merge_jobs([[new], [old]]) == [new]


def test_too_wide_date_range_is_an_error_not_no_jobs() -> None:
    r = fixture_result("slurm.acct_jobs.c0", "k2", "acct_jobs_too_wide")
    assert r.outcome is SectionOutcome.STDERR_ONLY
    err = acct.account_error(r, NOW)
    assert err is not None and err.kind is ErrorKind.REMOTE and err.summary == "sacct refused the date range"
    assert err.section == "slurm.acct_jobs.c0" and not acct.truncated(r)


def test_account_errors_and_truncation() -> None:
    assert acct.account_error(fixture_result("slurm.acct_jobs.c0", "k2", "acct_jobs"), NOW) is None
    dbd = result("slurm.acct_sreport", "", 1, "sreport: error: Problem talking to the database: Connection refused")
    assert acct.account_error(dbd, NOW).summary == "slurmdbd unreachable"   # type: ignore[union-attr]
    ctld = result("slurm.acct_share", "", 1, "sshare: error: Unable to contact slurm controller")
    assert acct.account_error(ctld, NOW).summary == "slurmctld unreachable"  # type: ignore[union-attr]
    missing = result("slurm.acct_sbank", "", 127, "missing:sbank")
    assert acct.account_error(missing, NOW).kind is ErrorKind.MISSING       # type: ignore[union-attr]
    cut = result("slurm.acct_jobs.c1", "1|x\n", 0, "runfleet: output truncated at 2097152 characters",
                 outcome=SectionOutcome.FAILED)
    assert acct.truncated(cut) and acct.account_error(cut, NOW).summary.endswith("size limit")  # type: ignore
    assert acct.truncated(result("slurm.acct_jobs.c1", "1|x", None))      # connection died mid-section
    assert not acct.truncated(dbd)


# ---------------------------------------------------------------- cells
def test_k2_cells() -> None:
    cells, seen = acct.cells(jobs_of("k2"), account_caps("k2"), now=NOW, start=NOW - timedelta(days=30))
    assert seen == 6
    by = {(c.name, c.partition, c.gpu_model): c for c in cells}
    mig = by[("eval_b_r2", "gpu-preempt", "nvidia_h100_nvl_1g.12gb")]
    assert (mig.jobs, mig.running, mig.qos) == (1, 0, "gpu-priority")
    assert mig.gpu_h == pytest.approx(11606 / 3600) and mig.cpu_h == pytest.approx(8 * 11606 / 3600)
    assert mig.billing_h == mig.cpu_h and mig.charged_su == mig.billing_h              # factor 1
    untyped = by[("lmft-x0-lr1e-6", "gpu-preempt", "")]                              # gres/gpu=2 without a type
    assert untyped.gpu_h == pytest.approx(2 * 3558 / 3600)
    cpu = by[("prep_env", "cpu-preempt", "")]
    assert cpu.gpu_h == 0 and cpu.billing_h == pytest.approx(4 * 69 / 3600)
    assert [c.account for c in cells] == ["campus01"] * 6


def test_lyra_cells_are_free_and_classify_observed() -> None:
    c = account_caps("lyra")
    cells, seen = acct.cells(jobs_of("lyra"), c, now=NOW, start=NOW - timedelta(days=30))
    assert seen == 7 and {(x.account, x.partition, x.qos, x.gpu_model) for x in cells} == {
        ("alice07", "free-gpuxl", "low", "rtx6000")}
    assert all(x.charged_su == 0 and x.billing_h > 0 for x in cells)                  # QOS low ×0
    verdict = {p.partition: p for p in classify_partitions(c, cells)}
    assert not verdict["free-gpuxl"].charged and verdict["free-gpuxl"].basis == "allow_qos"
    assert verdict["gpuxl"].charged and verdict["gpuxl"].basis == "weights"


def test_cells_clip_to_the_window_and_per_account_starts() -> None:
    t0 = NOW - timedelta(hours=10)
    a = acct.AcctJob("1", "lab", "gpu", "normal", JobState.COMPLETED, 4 * 3600, t0, t0 + timedelta(hours=4),
                     {"billing": "10", "cpu": "2"}, "exp--s1--a01")
    run = acct.AcctJob("2", "lab", "gpu", "normal", JobState.RUNNING, 3 * 3600, NOW - timedelta(hours=3), None,
                       {"billing": "10", "cpu": "2"}, "exp--s1--a01")
    other = dataclasses.replace(a, job_id_raw="3", account="other")
    old = dataclasses.replace(a, job_id_raw="4", start=NOW - timedelta(days=40))
    c = caps("k2", qos_usage_factor={"normal": 2.0})
    (cell,), seen = acct.cells([a, run, old], c, now=NOW, start=NOW - timedelta(hours=8))
    assert seen == 2 and (cell.jobs, cell.running) == (2, 1)
    assert cell.billing_h == pytest.approx(10 * (2 + 3)) and cell.charged_su == pytest.approx(100)  # 2 h of job 1
    cells, seen = acct.cells([a, other], c, now=NOW, start={"lab": NOW - timedelta(days=1)})
    assert seen == 1 and [x.account for x in cells] == ["lab"]                         # "other" has no period
    (whole,), _ = acct.cells([a], c, now=NOW, start=None)
    assert whole.cpu_h == pytest.approx(8.0)


def test_cells_billing_without_tres_and_unknown_factor() -> None:
    c = account_caps("lyra")
    t0 = NOW - timedelta(hours=2)
    weighted = acct.AcctJob("1", "pilab_02", "gpuxl", "normal", JobState.COMPLETED, 3600, t0, NOW,
                            {"cpu": "4", "gres/gpu:rtx6000": "1", "gres/gpu": "1"}, "x")
    cpus = dataclasses.replace(weighted, job_id_raw="2", partition="nowhere", tres={"cpu": "4"}, qos="mystery")
    cells, _ = acct.cells([weighted, cpus], c, now=NOW, start=None)
    by = {x.partition: x for x in cells}
    assert by["gpuxl"].billing_h == pytest.approx(4 + 64) and by["gpuxl"].charged_su == pytest.approx(68)
    assert by["gpuxl"].gpu_model == "rtx6000"
    assert by["nowhere"].billing_h == pytest.approx(4) and by["nowhere"].charged_su is None   # QOS factor unknown


def test_cells_split_gpu_h_by_model_like_job_cost() -> None:
    """Review: a job with GPUs of two models plus an untyped one puts each model's GPU-h under that model (as
    `util/cost._gpu_split` does for the Job detail), and counts the job, its CPU-h and billing once."""
    t0 = NOW - timedelta(hours=1)
    job = acct.AcctJob("1", "lab", "gpu", "normal", JobState.COMPLETED, 3600, t0, NOW,
                       {"billing": "8", "cpu": "8", "gres/gpu": "3", "gres/gpu:l40s": "1", "gres/gpu:rtx6000": "1"},
                       "x")
    cells, seen = acct.cells([job], caps("k2", qos_usage_factor={"normal": 1.0}), now=NOW, start=None)
    by = {c.gpu_model: c for c in cells}
    assert seen == 1 and {m: c.gpu_h for m, c in by.items()} == pytest.approx({"": 1.0, "l40s": 1.0, "rtx6000": 1.0})
    assert sum(c.jobs for c in cells) == 1 and sum(c.cpu_h for c in cells) == pytest.approx(8.0)
    assert sum(c.billing_h for c in cells) == pytest.approx(8.0) and by["l40s"].jobs == 1    # main: a tie, by name


def test_usage_from_cells() -> None:
    cells, _ = acct.cells(jobs_of("vega"), account_caps("vega"), now=NOW, start=None)
    start = NOW - timedelta(days=30)
    (use,) = acct.usage_from_cells(cells, {"pilab": start})
    assert use.source == "sacct" and use.scope == "me" and use.start == start
    assert use.gpu_h == pytest.approx(sum(x.gpu_h for x in cells)) and use.billing_h == use.cpu_h


def test_empty_outputs() -> None:
    assert acct.parse_share("").items == [] and acct.parse_sreport("\n").items == []
    assert acct.parse_acct_jobs("", tz=TZ["k2"]).items == []
    assert acct.sreport_usage([]) == ((), ()) and acct.cells([], None, now=NOW, start=None) == ((), 0)
