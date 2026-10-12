"""[WP2] Shell integration for workstations: nvidia (mock nvidia-smi, late + abandon wrapper), procfs (real /proc)
and process (real `ps`) sections rendered into one probe and run through real shells."""

from __future__ import annotations

import getpass
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.common.nvidia import NvidiaMonitor
from runfleet.backends.common.procfs import ProcfsMonitor
from runfleet.backends.process.collector import ProcessCollector
from runfleet.config import HostConfig
from runfleet.probe import framing
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import ctx
from tests.integration.harness import MockEnv, run_script, script_for, shells

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

ME = getpass.getuser()


def ws_ctx():
    return ctx("lynx", remote_user=ME)


def all_sections(cx):
    nv, pf, pc = NvidiaMonitor(cx.host), ProcfsMonitor(cx.host), ProcessCollector(cx.host)
    secs = [*pc.sections(frozenset({"process.ps"}), cx), *nv.sections(frozenset({"nvidia.gpu"}), cx),
            *pf.sections(frozenset({"procfs.host"}), cx)]
    return nv, pf, pc, secs


def _kill_pids(path: Path) -> None:
    if path.exists():
        for line in path.read_text().split():
            try:
                os.kill(int(line), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


@pytest.fixture
def lynx(tmp_path: Path) -> Iterator[MockEnv]:
    env = MockEnv(tmp_path, "lynx", mocks=("nvidia-smi",),
                  extra_env={"RF_MOCK_NVIDIA_PIDS": str(tmp_path / "nv.pids")})
    yield env
    _kill_pids(tmp_path / "nv.pids")


@pytest.mark.parametrize("shell", shells())
def test_workstation_probe(tmp_path: Path, shell: str) -> None:
    env = MockEnv(tmp_path, "lynx", mocks=("nvidia-smi",), sh=shell)
    cx = ws_ctx()
    nv, pf, pc, secs = all_sections(cx)
    script = script_for(secs)
    text = framing.render(script)
    assert text.index("\nrf process.ps ") < text.index("\nrf procfs.host ") < text.index("\nrf nvidia.gpu ")
    res = run_script(script, env.env(), shell=shell)
    assert {n: r.outcome for n, r in res.items()} == {n: SectionOutcome.OK for n in
                                                      ("process.ps", "nvidia.gpu", "nvidia.apps", "procfs.host")}
    nvo = nv.ingest(res, cx, MonitorView())
    assert nvo.errors == () and nvo.gpus is not None and len(nvo.gpus) == 2
    assert nvo.gpus[0].name == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    assert nvo.gpus[0].fan_pct is None and nvo.gpus[0].util_pct == 97      # padded [N/A] fields
    assert {p.pid for g in nvo.gpus for p in g.processes} == {642234, 786345}
    pfo = pf.ingest(res, cx, MonitorView())
    assert pfo.metrics is not None and pfo.metrics.ncpu == os.cpu_count() and pfo.metrics.mem_total
    view = MonitorView(gpus=nvo.gpus, gpu_procs=nvo.gpu_procs or (), gpus_at=cx.now, metrics=pfo.metrics)
    pco = pc.ingest(res, cx, view)
    assert pco.errors == ()
    # the real local process table parsed cleanly; this test process is in it with the right parent
    procs = {p.pid: p for p in pc._procs}
    assert procs[os.getpid()].ppid == os.getppid() and procs[os.getpid()].user == ME
    calls = env.calls("nvidia-smi")
    assert [c[1].split("=")[0] for c in calls] == ["--query-gpu", "--query-compute-apps"]
    assert calls[0][1].endswith(",fan.speed,pstate,utilization.memory") and calls[0][2] == "--format=csv,noheader,nounits"


def test_hung_nvidia_smi_is_abandoned(lynx: MockEnv) -> None:
    cx = ws_ctx()
    nv, _, _, secs = all_sections(cx)
    t0 = time.monotonic()
    res = run_script(script_for(secs), lynx.env(RF_MOCK_NVIDIA_HANG="20"))
    elapsed = time.monotonic() - t0
    assert 5 < elapsed < 15                               # abandoned after ~6 s, not the 20 s hang
    gpu = res["nvidia.gpu"]
    assert gpu.outcome is SectionOutcome.TIMEOUT and gpu.rc == 124 and "abandoned" in gpu.stderr
    assert res["process.ps"].outcome is SectionOutcome.OK and res["procfs.host"].outcome is SectionOutcome.OK
    assert res["nvidia.apps"].outcome is SectionOutcome.OK   # one hung instance does not trip the guard
    out = nv.ingest(res, cx, MonitorView())
    assert out.gpus is None and out.errors[0].summary == "nvidia-smi hung"


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    fake = tmp_path / "fakebin" / "nvidia-smi"
    fake.parent.mkdir()
    fake.symlink_to(subprocess.run(["sh", "-c", "command -v sleep"], capture_output=True, text=True).stdout.strip())
    return fake


def test_guard_refuses_to_stack(lynx: MockEnv, tmp_path: Path) -> None:
    fake = _fake_nvidia_smi(tmp_path)
    # two "hung" queries of earlier probes, still waited for by their abandoned wrapper subshells (`rf_nv`)
    stuck = [subprocess.Popen(["sh", "-c", f"rf_nv() {{ {fake} 30; }}; rf_nv; true"]) for _ in range(2)]
    time.sleep(0.2)
    try:
        cx = ws_ctx()
        nv, _, _, secs = all_sections(cx)
        res = run_script(script_for(secs), lynx.env())
        assert res["nvidia.gpu"].rc == 125 and "already running" in res["nvidia.gpu"].stderr
        out = nv.ingest(res, cx, MonitorView())
        assert out.errors[0].summary == "nvidia-smi hung"
        assert lynx.calls("nvidia-smi") == []             # never started a third one
    finally:
        for p in stuck:
            p.kill()
            p.wait()


def test_guard_ignores_healthy_nvidia_smi_of_the_user(lynx: MockEnv, tmp_path: Path) -> None:
    """Two `nvidia-smi -l` the user runs in a terminal (not stuck, not RunFleet's) never block GPU polling."""
    fake = _fake_nvidia_smi(tmp_path)
    healthy = [subprocess.Popen([str(fake), "30"]) for _ in range(2)]
    try:
        cx = ws_ctx()
        nv, _, _, secs = all_sections(cx)
        res = run_script(script_for(secs), lynx.env())
        assert res["nvidia.gpu"].outcome is SectionOutcome.OK and res["nvidia.apps"].outcome is SectionOutcome.OK
        assert [c[1].split("=")[0] for c in lynx.calls("nvidia-smi")] == ["--query-gpu", "--query-compute-apps"]
    finally:
        for p in healthy:
            p.kill()
            p.wait()


def test_broken_driver(lynx: MockEnv) -> None:
    cx = ws_ctx()
    nv, _, _, secs = all_sections(cx)
    res = run_script(script_for(secs), lynx.env(RF_MOCK_NVIDIA_RC="9"))
    assert res["nvidia.gpu"].outcome is SectionOutcome.FAILED and res["nvidia.gpu"].rc == 9
    out = nv.ingest(res, cx, MonitorView())
    assert out.errors[0].summary == "NVIDIA driver not loaded"


def test_fds_and_detail_sections_run(lynx: MockEnv) -> None:
    """process.fds / process.detail commands are valid sh and read this test process's cwd and environment."""
    from runfleet.backends.process.collector import detail_cmd, fds_cmd
    from runfleet.probe.section import Section
    me = os.getpid()
    res = run_script(script_for([Section("process.fds", fds_cmd([me])),
                                 Section("process.detail", detail_cmd([me], me))]), lynx.env())
    fds = res["process.fds"].stdout.split("\x1f")
    assert fds[0] == str(me) and fds[1] == os.getcwd()
    assert res["process.fds"].ok
    assert res["process.detail"].stdout.splitlines()[0].split()[0] == str(me)


def test_fds_section_stats_regular_file_stdout(lynx: MockEnv, tmp_path: Path) -> None:
    """A task writing to a file: process.fds prints `@@stat <path>|<size>|<mtime>` for it (M3 F9 residual)."""
    import subprocess

    from runfleet.backends.process.collector import fds_cmd, parse_fd_stats
    from runfleet.probe.section import Section
    log = tmp_path / "train.log"
    with log.open("w") as f:
        f.write("epoch 1\n")
        f.flush()
        child = subprocess.Popen(["sleep", "30"], stdout=f, stderr=subprocess.DEVNULL)
    try:
        res = run_script(script_for([Section("process.fds", fds_cmd([child.pid]))]), lynx.env())
    finally:
        child.kill()
        child.wait()
    assert res["process.fds"].ok
    stats = parse_fd_stats(res["process.fds"].stdout)
    assert stats[str(log)].size == 8 and stats[str(log)].mtime is not None and "/dev/null" not in stats


def test_logstat_follows_symlinks(lynx: MockEnv, tmp_path: Path) -> None:
    """process.logstat reports the size of the file `tail` reads, not of the symlink (latest.log -> run/train.log)."""
    from runfleet.backends.common import parse_stat_lines
    from runfleet.backends.process.collector import logstat_cmd
    from runfleet.probe.section import Section
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "train.log").write_text("x" * 1000)
    link = tmp_path / "latest.log"
    link.symlink_to("run/train.log")
    from runfleet.backends.slurm.commands import logstat_cmd as slurm_logstat_cmd
    res = run_script(script_for([Section("process.logstat", logstat_cmd([str(link)])),
                                 Section("slurm.logstat", slurm_logstat_cmd([str(link)]))]), lynx.env())
    assert parse_stat_lines(res["process.logstat"].stdout)[str(link)].size == 1000
    assert parse_stat_lines(res["slurm.logstat"].stdout)[str(link)].size == 1000


async def test_through_local_transport(lynx: MockEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same sections through WP1's LocalTransport (spawn hygiene, framing, partial rule)."""
    from runfleet.transport.local import LocalTransport
    for k, v in lynx.env().items():
        monkeypatch.setenv(k, v)
    cx = ws_ctx()
    nv, pf, pc, secs = all_sections(cx)
    script = script_for(secs)
    t = LocalTransport(HostConfig(name="lynx", ssh="local"))
    try:
        pr = await t.probe(script, timeout=framing.local_timeout(script))
    except NotImplementedError:
        pytest.skip("LocalTransport not implemented yet (WP1)")
    assert pr.error is None
    assert all(r.outcome is SectionOutcome.OK for r in pr.sections.values())
    out = nv.ingest(pr.sections, cx, MonitorView())
    assert out.gpus is not None and len(out.gpus) == 2
