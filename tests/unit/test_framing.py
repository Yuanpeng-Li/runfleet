"""[WP1] Probe framing (§4.3): pure render/parse, and round trips through the real /bin/sh (LocalTransport)."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from runfleet.config import HostConfig
from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionOutcome
from runfleet.transport import spawn
from runfleet.transport.local import LocalTransport
from tests.conftest import DEFAULT_NONCE, frame

LOCAL = LocalTransport(HostConfig(name="atlas", ssh="local"))


def script(*sections: Section, purpose: str = "poll") -> ProbeScript:
    return ProbeScript(sections=sections, nonce=DEFAULT_NONCE, purpose=purpose)  # type: ignore[arg-type]


async def run(*sections: Section, timeout: float | None = None) -> dict[str, object]:
    s = framing.make_script(sections)
    res = await LOCAL.probe(s, timeout=timeout or framing.local_timeout(s))
    assert res.error is None, res.error
    return dict(res.sections)


# ---------------------------------------------------------------- pure
def test_render_orders_late_sections_last_and_quotes() -> None:
    s = script(Section("nvidia.gpu", "nvidia-smi", late=True, timeout=8, requires=("nvidia-smi",)),
               Section("slurm.squeue", "squeue --me -O 'JobID:|'", timeout=20, requires=("squeue",)),
               Section("procfs.host", "cat /proc/loadavg", timeout=2.5))
    text = framing.render(s)
    assert text.startswith(f"# runfleet probe {DEFAULT_NONCE} v2\n")
    assert f"RF=@@RF{DEFAULT_NONCE}" in text and "export SLURM_TIME_FORMAT=%s" in text
    body = text.splitlines()
    rf_lines = [ln for ln in body if ln.startswith("rf ")]
    assert rf_lines == ["rf slurm.squeue 20 squeue 'squeue --me -O '\"'\"'JobID:|'\"'\"''",
                        "rf procfs.host 2.5 '' 'cat /proc/loadavg'",
                        "rf nvidia.gpu 8 nvidia-smi nvidia-smi"]
    assert body[-1] == "exit 0"


@pytest.mark.parametrize("bad", [
    script(Section("Bad.name", "x")), script(Section("a.b", "x"), Section("a.b", "y")),
    script(Section("a.b", "x", requires=("rm -rf",))), script(Section("a.b", "x", timeout=0)),
    ProbeScript(sections=(), nonce="xyz"),
])
def test_render_rejects_bad_scripts(bad: ProbeScript) -> None:
    with pytest.raises(ValueError):
        framing.render(bad)


def test_local_timeout_formula() -> None:
    assert framing.local_timeout(script(Section("a.b", "x", timeout=5))) == 25            # max(20, 5) + 5
    s = script(Section("a.b", "x", timeout=20), Section("a.c", "x", timeout=30), Section("n.g", "x", timeout=8,
                                                                                            late=True))
    assert framing.local_timeout(s) == 63
    big = script(*[Section(f"a.s{i}", "x", timeout=30) for i in range(5)])
    assert framing.local_timeout(big) == 90


def test_parse_conftest_frames() -> None:
    s = script(Section("a.ok", "x"), Section("a.nl", "x"), Section("a.empty", "x"), Section("a.err", "x"),
               Section("a.miss", "x"), Section("a.to", "x"), Section("a.fail", "x"), Section("a.later", "x"))
    out = frame({"a.ok": ("abc", 0, ""), "a.nl": ("l1\nl2\n", 0, ""), "a.empty": ("", 0, ""),
                 "a.err": ("", 0, "oops\nmore"), "a.miss": ("", 127, "missing:sacct"), "a.to": ("", 124, ""),
                 "a.fail": ("partial", 1, "stat: cannot stat")})
    p = framing.parse(("login banner\n" + out).encode(), s)
    r = p.sections
    assert (r["a.ok"].outcome, r["a.ok"].stdout) == (SectionOutcome.OK, "abc")
    assert r["a.nl"].stdout == "l1\nl2\n"                    # only the framing "\n" is removed (§4.3 round-trip)
    assert r["a.empty"].outcome is SectionOutcome.EMPTY
    assert (r["a.err"].outcome, r["a.err"].stderr) == (SectionOutcome.STDERR_ONLY, "oops\nmore")
    assert (r["a.miss"].outcome, r["a.miss"].rc, r["a.miss"].stderr) == (SectionOutcome.MISSING, 127,
                                                                         "missing:sacct")
    assert r["a.to"].outcome is SectionOutcome.TIMEOUT
    assert (r["a.fail"].outcome, r["a.fail"].stdout, r["a.fail"].rc) == (SectionOutcome.FAILED, "partial", 1)
    assert r["a.later"].outcome is SectionOutcome.NOT_RUN
    assert p.ended == 7


def test_parse_incomplete_and_timed_out() -> None:
    s = script(Section("a.one", "x"), Section("a.two", "x"))
    text = frame({"a.one": ("1", 0, "")}) + f"@@RF{DEFAULT_NONCE} B a.two\nhalf"
    assert framing.parse(text, s).sections["a.two"].outcome is SectionOutcome.INCOMPLETE
    p = framing.parse(text, s, timed_out=True)
    assert p.sections["a.two"].outcome is SectionOutcome.TIMEOUT and p.sections["a.two"].stdout == "half"
    assert p.ended == 1


def test_parse_ignores_other_nonces_and_unknown_sections() -> None:
    s = script(Section("a.one", "x"))
    inner = "@@RFffffffffffff E a.one 0 1\n@@RF" + DEFAULT_NONCE + " E a.other 0 1"
    p = framing.parse(frame({"a.one": (inner, 0, "")}), s)
    assert p.sections["a.one"].stdout == inner


def test_max_stdout_truncates() -> None:
    s = script(Section("a.big", "x", max_stdout=10))
    assert len(framing.parse(frame({"a.big": ("y" * 100, 0, "")}), s).sections["a.big"].stdout) == 10


# ---------------------------------------------------------------- through the real /bin/sh
async def test_round_trip_through_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQUEUE_STATES", "all")
    monkeypatch.setenv("SQUEUE_FORMAT", "%i")
    monkeypatch.setenv("SLURM_TIME_FORMAT", "standard")
    r = await run(
        Section("t.empty", "true"),
        Section("t.nonl", "printf abc"),
        Section("t.nl", "printf 'abc\\n'"),
        Section("t.blank", "printf 'abc\\n\\n'"),
        Section("t.binary", "printf '\\001\\377\\033[31mred\\r\\n'"),
        Section("t.marker", f"echo '@@RF{DEFAULT_NONCE} E t.marker 0 0'; echo '@@RFxyz B other'"),
        Section("t.stderr", "echo out; echo err >&2"),
        Section("t.stderronly", "echo only-err >&2"),
        Section("t.rc", "echo x; exit 3"),
        Section("t.missing", "nosuchtool", requires=("nosuchtool_rf",)),
        Section("t.stdin", "cat; echo after-cat"),
        Section("t.env", "env | grep -c '^SQUEUE_' ; echo \"$SLURM_TIME_FORMAT\" ; echo \"$LC_ALL\""),
        Section("t.late", "echo late", late=True),
        Section("t.after", "echo after"),
    )
    get = lambda n: r[n]  # noqa: E731
    assert get("t.empty").outcome is SectionOutcome.EMPTY                         # type: ignore[attr-defined]
    assert get("t.nonl").stdout == "abc" and get("t.nl").stdout == "abc\n"         # type: ignore[attr-defined]
    assert get("t.blank").stdout == "abc\n\n"                                      # type: ignore[attr-defined]
    assert get("t.binary").stdout == "\x01�\x1b[31mred\r\n"                   # type: ignore[attr-defined]
    assert get("t.marker").stdout.startswith(f"@@RF{DEFAULT_NONCE} E t.marker")    # type: ignore[attr-defined]
    assert (get("t.stderr").stdout, get("t.stderr").stderr) == ("out\n", "err\n")  # type: ignore[attr-defined]
    assert get("t.stderronly").outcome is SectionOutcome.STDERR_ONLY               # type: ignore[attr-defined]
    assert (get("t.rc").outcome, get("t.rc").rc) == (SectionOutcome.FAILED, 3)     # type: ignore[attr-defined]
    assert get("t.missing").outcome is SectionOutcome.MISSING                      # type: ignore[attr-defined]
    assert get("t.stdin").stdout == "after-cat\n"                                  # type: ignore[attr-defined]
    assert get("t.after").stdout == "after\n"                                      # cat did not eat the script
    assert get("t.env").stdout == "0\n%s\nC\n"                                     # type: ignore[attr-defined]
    assert get("t.late").stdout == "late\n"                                        # type: ignore[attr-defined]
    assert all(get(n).ms is not None for n in r)                                   # type: ignore[attr-defined]


async def test_remote_timeout_gives_124() -> None:
    t0 = time.monotonic()
    r = await run(Section("t.slow", "sleep 10", timeout=1), Section("t.next", "echo next"))
    assert r["t.slow"].outcome is SectionOutcome.TIMEOUT and r["t.slow"].rc == 124  # type: ignore[attr-defined]
    assert r["t.next"].stdout == "next\n"                                            # type: ignore[attr-defined]
    assert time.monotonic() - t0 < 6


async def test_local_timeout_partial_rule() -> None:
    s = framing.make_script([Section("t.fast", "echo ok"), Section("t.hang", "sleep 30", timeout=60)])
    res = await LOCAL.probe(s, timeout=1.5)
    assert res.error is None and res.run.timed_out                     # ≥ 1 end marker: ssh evidently works
    assert res.sections["t.fast"].ok
    assert res.sections["t.hang"].outcome is SectionOutcome.TIMEOUT
    s2 = framing.make_script([Section("t.hang", "sleep 30", timeout=60)])
    res2 = await LOCAL.probe(s2, timeout=1.0)
    assert res2.error is not None and res2.error.kind.value == "timeout"


async def test_term_to_the_script_exits_143_and_cleans_up() -> None:
    s = framing.make_script([Section("t.tmp", 'echo "$RF_T"'), Section("t.sleep", "sleep 30", timeout=60)])
    proc = await spawn.spawn(["sh", "-s"], stdin_pipe=True)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(framing.render(s).encode())
    await proc.stdin.drain()
    proc.stdin.close()
    seen = b""
    while b"B t.sleep" not in seen:
        seen += await asyncio.wait_for(proc.stdout.read(4096), 5)
    tmp = framing.parse(seen + b"\n", s).sections["t.tmp"].stdout or seen.decode().split("B t.tmp\n")[1].split("\n")[0]
    os.killpg(proc.pid, signal.SIGTERM)
    rc = await asyncio.wait_for(proc.wait(), 5)
    assert rc == 143
    assert tmp.startswith("/") and not Path(tmp).exists()


NVIDIA_WRAPPER = r'''[ "$(pgrep -u "$(id -un)" -x nvidia-smi 2>/dev/null | wc -l)" -ge 2 ] && { echo 'nvidia-smi already running (hung?)' >&2; exit 125; }
[ -n "$RF_T" ] || exec nvidia-smi --query-gpu=index --format=csv,noheader,nounits
o="$RF_T/nv.$$"
( nvidia-smi --query-gpu=index --format=csv,noheader,nounits >"$o" 2>"$o.e"; echo $? >"$o.rc" ) </dev/null >/dev/null 2>&1 &
i=0; while [ ! -s "$o.rc" ] && [ $i -lt 60 ]; do sleep 0.1 2>/dev/null || sleep 1; i=$((i+1)); done
[ -s "$o.rc" ] || { echo 'nvidia-smi did not answer within 6 s (abandoned)' >&2; exit 124; }
cat "$o"; cat "$o.e" >&2; exit "$(cat "$o.rc")"'''


async def test_hung_nvidia_smi_is_abandoned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.3: a TERM-ignoring nvidia-smi that sleeps 30 s -> rc 124 within ~6 s, earlier sections done, no
    transport failure."""
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\ntrap '' TERM\nsleep 30\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    s = framing.make_script([Section("nvidia.gpu", NVIDIA_WRAPPER, timeout=8, late=True, requires=("nvidia-smi",)),
                             Section("procfs.host", "echo load=1", timeout=5)])
    t0 = time.monotonic()
    res = await LOCAL.probe(s, timeout=framing.local_timeout(s))
    elapsed = time.monotonic() - t0
    assert res.error is None
    assert res.sections["procfs.host"].ok
    gpu = res.sections["nvidia.gpu"]
    assert gpu.outcome is SectionOutcome.TIMEOUT and gpu.rc == 124 and "abandoned" in gpu.stderr
    assert elapsed < 9
    os.system(f"pkill -KILL -f {fake} >/dev/null 2>&1")          # the abandoned child is reparented; tidy up


# ---------------------------------------------------------------- review regressions
def test_action_budget_is_not_capped_at_90_s() -> None:
    kill = script(Section("process.kill", "x", timeout=135), purpose="action")
    assert framing.local_timeout(kill) >= 140
    many = script(*[Section(f"process.k{i}", "x", timeout=25) for i in range(7)], purpose="action")
    assert framing.local_timeout(many) >= 7 * 27
    assert framing.local_timeout(script(*[Section(f"a.s{i}", "x", timeout=30) for i in range(5)])) == 90


def test_truncated_output_is_not_authoritative() -> None:
    s = script(Section("a.big", "x", max_stdout=10))
    r = framing.parse(frame({"a.big": ("row1\nrow2\nrow33\n", 0, "")}), s).sections["a.big"]
    assert r.outcome is SectionOutcome.FAILED and r.stdout == "row1\nrow2\n"     # whole rows, still parseable
    assert r.stderr.startswith("runfleet: output truncated at 10 characters")


def test_cut_between_stderr_and_end_marker_is_incomplete() -> None:
    s = script(Section("slurm.squeue", "x"))
    rf = f"@@RF{DEFAULT_NONCE}"
    cut = f"\n{rf} B slurm.squeue\n\n{rf} X slurm.squeue\nslurmctld timeout\n{rf} Y slurm.squeue\n"
    assert framing.parse(cut, s).sections["slurm.squeue"].outcome is SectionOutcome.INCOMPLETE


async def test_stderr_block_comes_before_the_end_marker() -> None:
    s = framing.make_script([Section("t.e", "echo out; echo err >&2")])
    res = await LOCAL.probe(s, timeout=10)
    out = res.run.stdout.decode()
    assert out.index(f"@@RF{s.nonce} Y t.e") < out.index(f"@@RF{s.nonce} E t.e")
    assert (res.sections["t.e"].stdout, res.sections["t.e"].stderr) == ("out\n", "err\n")


async def test_unterminated_banner_does_not_swallow_the_first_marker() -> None:
    s = framing.make_script([Section("detect.id", "echo me"), Section("detect.tools", "echo t")])
    res = await spawn.run(["sh", "-s"], timeout=10, input=("printf Welcome\n" + framing.render(s)).encode())
    r = framing.parse(res.stdout, s).sections
    assert r["detect.id"].outcome is SectionOutcome.OK and r["detect.id"].stdout == "me\n"


async def test_stderr_survives_an_unusable_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMPDIR", "/nonexistent/rf-test")
    r = await run(Section("t.only", "echo only-err >&2"), Section("t.both", "echo out; echo e2 >&2"),
                  Section("t.rc", "exit 3"))
    assert r["t.only"].outcome is SectionOutcome.STDERR_ONLY                        # type: ignore[attr-defined]
    assert r["t.both"].stdout == "out\n" and "e2" in r["t.both"].stderr            # type: ignore[attr-defined]
    assert r["t.rc"].rc == 3                                                         # type: ignore[attr-defined]


@pytest.mark.skipif(not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(), reason="needs bash")
async def test_inherited_errexit_does_not_abort_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELLOPTS", "errexit")
    s = framing.make_script([Section("t.fail", "false"), Section("t.after", "echo after")])
    res = await spawn.run(["bash", "-s"], timeout=10, input=framing.render(s).encode())
    r = framing.parse(res.stdout, s).sections
    assert r["t.fail"].outcome is SectionOutcome.FAILED and r["t.after"].stdout == "after\n"


async def test_attach_probes_write_nothing_on_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§7.16.4: an Attach login probe (purpose "attach") never creates a temp dir — stderr goes through a variable —
    while stdout, stderr, rc, the remote timeout and partial output behave as with one."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    s = framing.make_script([
        Section("slurm.attach.node0", "echo out; echo 'Permission denied (publickey).' >&2; exit 255", timeout=5),
        Section("slurm.attach.node1", "echo partial; echo half >&2; sleep 30", timeout=1),
        Section("slurm.attach.audit", 'ls -A "$TMPDIR" | wc -l; echo "T=$RF_T"', late=True),
    ], "attach")
    assert "mktemp" not in framing.render(s)
    assert "mktemp" in framing.render(ProbeScript(s.sections, s.nonce, "poll"))       # other purposes keep it
    assert "mktemp" in framing.render(s, tmp=True)                                    # an explicit choice wins
    res = await LOCAL.probe(s, timeout=framing.local_timeout(s))
    assert res.error is None, res.error
    n0, n1, audit = (res.sections[n] for n in ("slurm.attach.node0", "slurm.attach.node1", "slurm.attach.audit"))
    assert (n0.outcome, n0.rc, n0.stdout, n0.stderr) == (SectionOutcome.FAILED, 255, "out\n",
                                                         "Permission denied (publickey).")
    assert (n1.outcome, n1.rc, n1.stdout, n1.stderr) == (SectionOutcome.TIMEOUT, 124, "partial\n", "half")
    assert audit.stdout.split() == ["0", "T="]                                        # nothing in TMPDIR meanwhile
    assert list(tmp_path.iterdir()) == []
