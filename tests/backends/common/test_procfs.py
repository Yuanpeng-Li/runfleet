"""[WP2] ProcfsMonitor: metrics from the resources/meminfo/proc_stat fixtures, CPU % from two `cpu` lines, cadence."""

from __future__ import annotations

import re

import pytest

from runfleet.backends.base import MonitorView
from runfleet.backends.common.procfs import ProcfsMonitor, cpu_pct, parse_cpu_line, parse_procfs
from tests.backends.factories import at, ctx, result
from tests.conftest import sample


def procfs_output(host: str, cpu_line: int) -> str:
    """Rebuild what our `procfs.host` section prints from the probe fixtures (resources = nproc; free -b;
    /proc/loadavg; uptime)."""
    res = sample(host, "resources").splitlines()
    m = re.search(r"up (\d+) days?, +(\d+):(\d+)", res[-1])
    assert m is not None
    uptime = int(m.group(1)) * 86400 + int(m.group(2)) * 3600 + int(m.group(3)) * 60
    cpu = sample(host, "proc_stat").splitlines()[cpu_line]
    return "\n".join([f"ncpu={res[0]}", "epoch=1788763357", f"load={res[4]}", sample(host, "meminfo").rstrip("\n"),
                      cpu, f"uptime={uptime}.00"])


@pytest.mark.parametrize(("host", "ncpu", "load1"), [("labstation", 64, 1.02), ("lynx", 64, 2.61)])
def test_metrics_and_cpu_delta(host: str, ncpu: int, load1: float) -> None:
    cx = ctx(host)
    m = ProcfsMonitor(cx.host)
    first = m.ingest({"procfs.host": result("procfs.host", procfs_output(host, 0))}, cx, MonitorView())
    met = first.metrics
    assert met is not None and met.ncpu == ncpu and met.load is not None and met.load[0] == load1
    assert met.cpu_pct is None                            # first sample: no delta yet
    mem = dict(line.split(":") for line in sample(host, "meminfo").splitlines())
    assert met.mem_total == int(mem["MemTotal"].split()[0]) * 1024
    assert met.mem_available == int(mem["MemAvailable"].split()[0]) * 1024
    assert met.swap_free is not None and met.uptime_s is not None and met.uptime_s > 100 * 86400
    second = m.ingest({"procfs.host": result("procfs.host", procfs_output(host, 1))}, at(cx, 5), MonitorView())
    assert second.metrics is not None and second.metrics.cpu_pct is not None
    assert 0 < second.metrics.cpu_pct < 100


def test_labstation_cpu_pct_value() -> None:
    a, b = (parse_cpu_line(x) for x in sample("labstation", "proc_stat").splitlines())
    # Δuser 4 + Δnice 79 + Δsystem 23 busy over a total Δ of 6428 jiffies
    assert cpu_pct(a, b) == pytest.approx(100 * 106 / 6428)
    assert cpu_pct(None, b) is None and cpu_pct(b, a) is None


def test_free_b_agrees_with_meminfo() -> None:
    for host in ("labstation", "lynx"):
        res = sample(host, "resources").splitlines()
        free_total = int(res[2].split()[1])
        meminfo_total = int(sample(host, "meminfo").splitlines()[0].split()[1]) * 1024
        assert abs(free_total - meminfo_total) / meminfo_total < 0.03


def test_parse_robustness() -> None:
    s = parse_procfs("ncpu=\nload=garbage\ncpu  x y\nMemTotal: 10 kB\n")
    assert s.ncpu is None and s.load is None and s.cpu is None and s.mem_kb == {"MemTotal": 10}


def test_cadence_piggyback_on_slurm_hosts() -> None:
    slurm = ctx("k2")
    (cad,) = ProcfsMonitor(slurm.host).cadences(slurm)
    assert cad.mode == "piggyback" and cad.interval == 15 and cad.group == "procfs.host"
    ws = ctx("lynx")
    (cad,) = ProcfsMonitor(ws.host).cadences(ws)
    assert cad.mode == "interval" and cad.interval == 5 and cad.floor == 2


def test_failures_keep_previous() -> None:
    cx = ctx("lynx")
    m = ProcfsMonitor(cx.host)
    out = m.ingest({"procfs.host": result("procfs.host", "", 1, "cat: /proc/loadavg: No such file")}, cx,
                   MonitorView())
    assert out.metrics is None and out.errors[0].section == "procfs.host"
    assert m.ingest({}, cx, MonitorView()).metrics is None
    assert m.sections(frozenset({"procfs.host"}), cx)[0].timeout == 5
