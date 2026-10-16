"""[WP2] nvidia-smi parsing (query rows from the right, compute-app join, driver version) and NvidiaMonitor outcomes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.common.gpu_types import mem_mib
from runfleet.backends.common.nvidia import (GPU_FIELDS, NvidiaMonitor, apps_cmd, attach_processes, parse_apps,
                                             parse_driver_version, parse_gpu_query, wrap)
from runfleet.models import ErrorKind
from runfleet.probe.section import SectionOutcome
from tests.backends.factories import at, ctx, result
from tests.conftest import meta, sample

FIXTURE_FIELDS = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu",
                  "power.draw", "power.limit")


def gpus(host: str) -> list[object]:
    m = re.search(r"--query-gpu=(\S+)", meta(host, "gpu_query")["cmd"])
    assert m is not None and tuple(m.group(1).split(",")) == FIXTURE_FIELDS
    pr = parse_gpu_query(sample(host, "gpu_query"), FIXTURE_FIELDS, host=host)
    assert not pr.bad_lines
    return pr.items


def test_labstation_gpu_query() -> None:
    g = gpus("labstation")
    assert len(g) == 4
    g3 = g[3]
    assert g3.index == 3 and g3.util_pct == 70 and g3.mem_used_mib == 21464 and g3.mem_total_mib == 49140
    assert g3.name == "NVIDIA RTX A6000" and g3.temp_c == 78 and g3.power_w == pytest.approx(286.58)
    assert g3.uuid == "GPU-d5f3487b-34d5-9bc6-16d5-f3d609021850" and g3.fan_pct is None and g3.pstate is None


def test_lynx_long_name_intact() -> None:
    g = gpus("lynx")
    assert [x.name for x in g] == ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"] * 2
    assert g[1].util_pct == 100 and g[0].mem_total_mib == 97887


def test_synthetic_twelve_fields_with_comma_name_and_na() -> None:
    line = "0, GPU-abc, NVIDIA Foo, Bar Edition, 55, 100, 200, 40, 70.5, 300.00, [N/A], P0, 12"
    (g,) = parse_gpu_query(line, GPU_FIELDS, host="h").items
    assert g.name == "NVIDIA Foo, Bar Edition"
    assert g.fan_pct is None and g.pstate == "P0" and g.mem_util_pct == 12 and g.util_pct == 55
    assert parse_gpu_query("1, GPU-x, [Not Supported], 3", GPU_FIELDS, host="h").bad_lines


def test_header_fixtures_agree_with_nounits() -> None:
    # gpu_query_header has units and a 13-field list; the nounits fixture values match it row by row
    for host in ("labstation", "lynx"):
        header, *rows = sample(host, "gpu_query_header").splitlines()
        assert header.startswith("index, uuid, name, utilization.gpu [%]")
        plain = {x.index: x for x in gpus(host)}
        for row in rows:
            cols = row.split(", ")
            assert plain[int(cols[0])].util_pct == float(cols[3].split()[0])
            assert plain[int(cols[0])].mem_used_mib == float(cols[4].split()[0])


@pytest.mark.parametrize("host", ["labstation", "lynx"])
def test_compute_apps_join(host: str) -> None:
    text = sample(host, "compute_apps").rstrip("\n") + "\n@@ps\n"
    apps, procs = parse_apps(text)
    joined = attach_processes(gpus(host), apps, procs.items)
    holders = {p.pid: p.gpu_index for g in joined for p in g.processes}
    if host == "lynx":
        assert holders == {786345: 0, 642234: 1}
        assert all(p.used_mem_mib == 15234 for g in joined for p in g.processes)
    else:
        assert holders == {363948: 3}
    header = sample(host, "compute_apps_header").splitlines()
    assert header[0].startswith("gpu_uuid, pid, process_name, used_gpu_memory")
    assert {int(r.split(", ")[1]) for r in header[1:]} == set(holders)


def test_apps_with_ps_rows() -> None:
    text = ("GPU-c6faebeb-cf00-3f1a-1182-49cec7f3d256, 786345, 15234\n@@ps\n"
            " 786345  642229 alice07     3103  114  1.9 11425460 SNl /data/home/alice07/projects/orb/.venv/bin/python "
            "-m orb.train --config x.yaml\n")
    apps, procs = parse_apps(text)
    (p,) = procs.items
    assert p.pgid is None and p.user == "alice07" and p.rss_bytes == 11425460 * 1024 and p.args.endswith("x.yaml")
    joined = attach_processes(gpus("lynx"), apps, procs.items)
    gp = joined[0].processes[0]
    assert gp.user == "alice07" and gp.etime_s == 3103 and gp.command is not None
    assert parse_apps("\n@@ps\n") == ([], parse_apps("\n@@ps\n")[1]) and parse_apps("\n@@ps")[0] == []


@pytest.mark.parametrize("host", ["labstation", "lynx"])
def test_driver_version(host: str) -> None:
    assert parse_driver_version(sample(host, "nvidia_smi_version")) == "610.43.02"
    assert NvidiaMonitor.ingest_capabilities({"nvidia.version": result("nvidia.version",
                                                                       sample(host, "nvidia_smi_version"))}) == \
        {"driver_version": "610.43.02"}
    assert parse_driver_version("DRIVER version : 550.1\n") == "550.1"


def test_gpu_memory_table() -> None:
    assert mem_mib("nvidia_a100-sxm4-80gb") == 81920 and mem_mib("nvidia_a100_80gb_pcie") == 81920
    assert mem_mib("nvidia_a100_80gb_pcie_3g.40gb") == 40960
    assert mem_mib("v100") is None and mem_mib(None) is None
    assert mem_mib("nvidia_h200_nvl") == 143771
    assert mem_mib("RTX6000") is None          # lyra's RTX6000 = 96 GB RTX PRO 6000 Blackwell; Quadro ones 24 GB
    assert mem_mib("tesla_v100-pcie-16gb") == 16384


# ------------------------------------------------------------------ monitor
def _monitor() -> tuple[NvidiaMonitor, object]:
    cx = ctx("lynx")
    return NvidiaMonitor(cx.host), cx


def full_query(host: str) -> str:
    # live rows carry 12 fields: append fan, pstate, mem util to the 9-field fixture rows
    return "\n".join(f"{row}, 30, P1, 28" for row in sample(host, "gpu_query").splitlines())


def test_monitor_sections_and_ingest() -> None:
    m, cx = _monitor()
    assert m.cadences(cx)[0].group == "nvidia.gpu" and m.cadences(cx)[0].interval == 5
    secs = m.sections(frozenset({"nvidia.gpu"}), cx)
    assert [s.name for s in secs] == ["nvidia.gpu", "nvidia.apps"] and all(s.late for s in secs)
    assert all(s.timeout == 8 and s.requires == ("nvidia-smi",) for s in secs)
    apps = sample("lynx", "compute_apps").rstrip("\n") + "\n@@ps"
    out = m.ingest({"nvidia.gpu": result("nvidia.gpu", full_query("lynx")),
                    "nvidia.apps": result("nvidia.apps", apps)}, cx, MonitorView())
    assert out.gpus is not None and len(out.gpus) == 2 and out.gpus[0].pstate == "P1"
    assert {p.pid for g in out.gpus for p in g.processes} == {642234, 786345}
    assert out.gpu_procs == () and out.errors == ()
    assert m.ingest({}, cx, MonitorView()).gpus is None


def test_driver_not_loaded_and_no_devices() -> None:
    m, cx = _monitor()
    out = m.ingest({"nvidia.gpu": result("nvidia.gpu", "", 9, "NVIDIA-SMI has failed because it couldn't "
                                                              "communicate with the NVIDIA driver.")}, cx, MonitorView())
    assert out.gpus is None and out.errors[0].summary == "NVIDIA driver not loaded"
    assert out.errors[0].kind is ErrorKind.REMOTE
    out = m.ingest({"nvidia.gpu": result("nvidia.gpu", "No devices were found", 6)}, cx, MonitorView())
    assert out.gpus == () and out.errors == ()


def test_hung_three_times_suspends_five_minutes() -> None:
    m, cx = _monitor()
    hung = {"nvidia.gpu": result("nvidia.gpu", "", 124, "nvidia-smi did not answer within 6 s (abandoned)"),
            "nvidia.apps": result("nvidia.apps", "", 125, "nvidia-smi already running (hung?)")}
    for i in range(3):
        out = m.ingest(hung, at(cx, i * 5), MonitorView())
        assert out.gpus is None and out.errors[0].summary == "nvidia-smi hung" and len(out.errors) == 1
        assert out.errors[0].kind is ErrorKind.TIMEOUT
    assert m.sections(frozenset({"nvidia.gpu"}), at(cx, 20)) == []
    assert m.sections(frozenset({"nvidia.gpu"}), at(cx, 20 + 301))


def test_apps_failure_keeps_last_processes() -> None:
    m, cx = _monitor()
    apps = sample("lynx", "compute_apps").rstrip("\n") + "\n@@ps"
    m.ingest({"nvidia.gpu": result("nvidia.gpu", full_query("lynx")), "nvidia.apps": result("nvidia.apps", apps)},
             cx, MonitorView())
    out = m.ingest({"nvidia.gpu": result("nvidia.gpu", full_query("lynx")),
                    "nvidia.apps": result("nvidia.apps", "", None, outcome=SectionOutcome.INCOMPLETE)},
                   at(cx, 5), MonitorView())
    assert out.gpus is not None and {p.pid for g in out.gpus for p in g.processes} == {642234, 786345}
    assert out.gpu_procs is None


def test_capability_section() -> None:
    (s,) = NvidiaMonitor.capability_sections(ctx("lynx").host)
    assert s.name == "nvidia.version" and s.requires == ("nvidia-smi",)


def test_apps_failing_for_longer_drops_stale_processes() -> None:
    """Review BACK-13: a fresh GPU sample carried the process list of the last good apps query indefinitely."""
    m, cx = _monitor()
    apps = sample("lynx", "compute_apps").rstrip("\n") + "\n@@ps"
    good = {"nvidia.gpu": result("nvidia.gpu", full_query("lynx")), "nvidia.apps": result("nvidia.apps", apps)}
    bad = {"nvidia.gpu": result("nvidia.gpu", full_query("lynx")),
           "nvidia.apps": result("nvidia.apps", "", 1, "Failed to initialize NVML")}
    m.ingest(good, cx, MonitorView())
    m.ingest(bad, at(cx, 5), MonitorView())
    out = m.ingest(bad, at(cx, 10), MonitorView())
    assert out.gpus is not None and all(g.processes == () for g in out.gpus)
    out = m.ingest(good, at(cx, 15), MonitorView())
    assert out.gpus is not None and {p.pid for g in out.gpus for p in g.processes} == {642234, 786345}


@pytest.mark.posix                # runs the wrapper under sh with a fake nvidia-smi script
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_instant_payload_failure_keeps_its_rc(tmp_path: Path) -> None:
    """Review BACK-12: `|| exit $?` in the apps payload ended the wrapper's subshell before the rc file was written,
    so an instant nvidia-smi failure read as a 6 s hang (rc 124) and fed the hang suspension."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'Failed to initialize NVML: Driver/library version mismatch' >&2\nexit 18\n")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "RF_T": str(tmp_path)}
    t0 = time.monotonic()
    proc = subprocess.run(["sh", "-c", apps_cmd()], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 18 and "version mismatch" in proc.stderr
    assert time.monotonic() - t0 < 3


@pytest.mark.parametrize("gpu_type", ["nvidia_a100_80gb_pcie_1g.10gb", "nvidia_a100_80gb_pcie-1g.10gb", "1g.10gb"])
def test_mig_profile_wins_over_the_parent_card(gpu_type: str) -> None:
    """Review BACK-34: the hyphenated MIG name got the parent card's 80 GB."""
    assert mem_mib(gpu_type) == 10240


# `ps -o pid=,ppid=,stat=,args=` rows the guard reads (a fake ps prints them)
_WRAPPER = "sh -c [ \"$(ps -ww -u me -o pid=,ppid=,stat=,args= …)\" -ge 2 ] && { … }; rf_nv() { nvidia-smi --query-gpu=…"


@pytest.mark.posix                # sh with fake ps / nvidia-smi scripts
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
@pytest.mark.parametrize("rows,refused", [
    # two healthy instances of the user's own (`nvidia-smi -l` in a job, `nvidia-smi dmon` in tmux): not hung
    (["101 100 Sl nvidia-smi -l 30", "102 100 S /usr/bin/nvidia-smi dmon", "100 1 Ss bash"], False),
    # two stuck queries in D state (they survive SIGKILL): never stack a third
    (["201 1 D nvidia-smi --query-gpu=index", "202 1 D nvidia-smi --query-compute-apps=pid"], True),
    # two abandoned RunFleet queries still waited for by their wrapper subshells (workstations): never stack
    (["301 300 S nvidia-smi --query-gpu=index", "300 1 S " + _WRAPPER, "303 302 S nvidia-smi --query-gpu=index",
      "302 1 S " + _WRAPPER], True),
    # one stuck query and a healthy one: run
    (["401 1 D nvidia-smi --query-gpu=index", "402 400 S nvidia-smi -l 5", "400 1 S python train.py"], False),
])
def test_guard_counts_only_stuck_or_abandoned_instances(tmp_path: Path, rows: list[str], refused: bool) -> None:
    """The stacking guard refuses (rc 125) only when two of the user's nvidia-smi are stuck (D state) or RunFleet's
    own abandoned queries; healthy instances a job or the user started are not counted (Attach and workstations)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "rows").write_text("".join(f"{r}\n" for r in rows))
    fake_ps = bindir / "ps"
    fake_ps.write_text(f"#!/bin/sh\ncat {tmp_path / 'rows'}\n")
    fake_ps.chmod(0o755)
    fake_pgrep = bindir / "pgrep"                                      # the same process table for pgrep
    fake_pgrep.write_text(f"#!/bin/sh\nawk '{{c=$4; sub(/.*\\//, \"\", c)}} c == \"nvidia-smi\" {{print $1}}' "
                          f"{tmp_path / 'rows'}\n")
    fake_pgrep.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "RF_T": ""}
    proc = subprocess.run(["sh", "-c", wrap("echo ran")], env=env, capture_output=True, text=True, timeout=20)
    if refused:
        assert proc.returncode == 125 and "already running" in proc.stderr and proc.stdout == ""
    else:
        assert proc.returncode == 0 and proc.stdout == "ran\n"


@pytest.mark.posix                # sh with fake ps / nvidia-smi scripts
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
@pytest.mark.parametrize("pgrep,refused", [
    ("echo 101; echo 102", "already running"),       # two instances of the user's: refuse, as before
    ("echo 101", None),
    ("exit 1", None),                                # pgrep rc 1 = no match
    ("exit 3", "cannot list processes"),             # pgrep broken too: never fail open
])
def test_guard_without_a_capable_ps_counts_every_instance(tmp_path: Path, pgrep: str, refused: str | None) -> None:
    """BusyBox `ps` rejects `-u` / `-o`: the guard must not read that as "no nvidia-smi" (fail open) but fall back
    to counting every nvidia-smi of the user with pgrep, and refuse when it cannot list processes at all."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ps").write_text("#!/bin/sh\necho 'ps: invalid option -- u' >&2\nexit 1\n")
    (bindir / "pgrep").write_text(f"#!/bin/sh\n{pgrep}\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "RF_T": ""}
    proc = subprocess.run(["sh", "-c", wrap("echo ran")], env=env, capture_output=True, text=True, timeout=20)
    if refused:
        assert proc.returncode == 125 and refused in proc.stderr and proc.stdout == ""
    else:
        assert proc.returncode == 0 and proc.stdout == "ran\n"


@pytest.mark.posix                # sh with fake ps / awk scripts
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_guard_refuses_when_awk_fails(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ps").write_text("#!/bin/sh\necho '1 0 S bash'\n")
    (bindir / "awk").write_text("#!/bin/sh\nexit 2\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "RF_T": ""}
    proc = subprocess.run(["sh", "-c", wrap("echo ran")], env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 125 and "cannot list processes" in proc.stderr and proc.stdout == ""
