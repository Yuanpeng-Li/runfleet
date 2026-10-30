"""[WP1] M0 contract tests: every module imports, layer boundaries hold, every contract dataclass constructs.

DESIGN.md §3.2 (layout), §3.3-§3.7 (FROZEN contracts), §11.3 (M0), §11.5 (import rules).
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import importlib
import inspect
import pkgutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import get_args, get_type_hints

import pytest

import runfleet
from runfleet import config, errors, models
from runfleet.backends import base as bbase
from runfleet.core import clock as cclock
from runfleet.core import service, store
from runfleet.probe import section
from runfleet.transport import base as tbase

PKG_DIR = Path(runfleet.__file__).resolve().parent

# Every module of §3.2 (dotted, relative to `runfleet`).
EXPECTED_MODULES = """
__main__ cli export models errors config configedit sshconfig paths debugdump wait update status
util util.shell util.slurmfmt util.humanize util.logscan util.efficiency util.platform util.redact util.gpufree
probe probe.section probe.framing
transport transport.base transport.openssh transport.mux transport.spawn transport.classify transport.local
transport.fake transport.agent transport.session
backends backends.base backends.registry backends.detect
backends.common backends.common.nvidia backends.common.procfs backends.common.gpu_types backends.common.cgroup
backends.slurm backends.slurm.collector backends.slurm.commands backends.slurm.states backends.slurm.parse_squeue
backends.slurm.parse_sacct backends.slurm.parse_scontrol backends.slurm.parse_sinfo backends.slurm.parse_sstat
backends.slurm.logpaths backends.slurm.actions backends.slurm.attach backends.slurm.parse_attach
backends.slurm.parse_occupants backends.slurm.priority
backends.process backends.process.collector backends.process.parse_ps backends.process.tasks
backends.process.actions
core core.service core.store core.clock core.poller core.actions core.logs core.cache core.doctor
core.forwards core.events core.attach core.logsearch
ui ui.app ui.messages ui.keymap ui.render ui.theme ui.commands ui.query ui.everyday ui.fleet_layout
ui.screens ui.screens.main ui.screens.job_detail ui.screens.host_detail ui.screens.log_viewer ui.screens.help
ui.screens.confirm ui.screens.action_menu ui.screens.host_error ui.screens.settings ui.screens.forwards
ui.notify ui.screens.notifications ui.screens.attention ui.termbg ui.saved_views ui.screens.save_view
ui.gpu_watch ui.screens.gpu_watch ui.screens.log_search ui.screens.found_logs
ui.widgets ui.widgets.header ui.widgets.hintbar ui.widgets.tabs ui.widgets.fleet_view ui.widgets.host_card
ui.widgets.jobs_view ui.widgets.job_table ui.widgets.job_preview ui.widgets.gpus_view ui.widgets.gpu_panel
ui.widgets.proc_table ui.widgets.node_table ui.widgets.log_view ui.widgets.filter_bar ui.widgets.attach_panel
ui.widgets.fleet_panel
demo demo.world demo.render demo.transport
agent agent.data agent.view agent.events agent.source agent.skill agent.commands agent.mcp
""".split()

CONTRACT_MODULES: tuple[ModuleType, ...] = (models, section, tbase, bbase, config, cclock, store, service)

T0 = datetime(2026, 9, 7, 10, 42, 37, tzinfo=UTC)


def _module_path(dotted: str) -> Path:
    rel = Path(*dotted.split("."))
    pkg = PKG_DIR / rel / "__init__.py"
    return pkg if pkg.is_file() else PKG_DIR / rel.with_suffix(".py")


def _all_module_files() -> list[Path]:
    return sorted(p for p in PKG_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_top_names(path: Path) -> set[str]:
    """Top-level package names imported anywhere in the file (incl. function-local and TYPE_CHECKING imports)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "import_module" and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value.split(".")[0])
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__"
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value.split(".")[0])
    return names


# ---------------------------------------------------------------- layout and imports
@pytest.mark.parametrize("dotted", EXPECTED_MODULES)
def test_module_exists_imports_and_has_docstring(dotted: str) -> None:
    assert _module_path(dotted).is_file(), f"missing {_module_path(dotted)} (§3.2)"
    mod = importlib.import_module(f"runfleet.{dotted}")
    assert (mod.__doc__ or "").strip(), f"runfleet.{dotted} needs a module docstring (§11.3)"


def test_every_module_in_the_tree_imports() -> None:
    found = [m.name for m in pkgutil.walk_packages(runfleet.__path__, prefix="runfleet.")]
    assert len(found) >= len(EXPECTED_MODULES)
    for name in found:
        importlib.import_module(name)


def test_no_unexpected_modules() -> None:
    expected = {_module_path(m) for m in EXPECTED_MODULES} | {PKG_DIR / "__init__.py"}
    extra = [p.relative_to(PKG_DIR) for p in _all_module_files() if p not in expected]
    assert not extra, f"modules not in DESIGN.md §3.2 (add them there first): {extra}"


def test_tcss_present() -> None:
    assert (PKG_DIR / "ui" / "runfleet.tcss").is_file()


# ---------------------------------------------------------------- §11.5 boundary rules
def test_textual_only_in_ui_and_cli() -> None:
    offenders = []
    for path in _all_module_files():
        rel = path.relative_to(PKG_DIR)
        if rel.parts[0] == "ui" or rel == Path("cli.py"):
            continue
        if "textual" in _imported_top_names(path):
            offenders.append(str(rel))
    assert not offenders, f"textual imported outside ui/ and cli.py: {offenders}"


# docs/AGENT_INTERFACE.md §5.1: `runfleet agent` is read-only by construction. A maintenance guard, not the security
# boundary (that is: no mutating code path at all): agent/ imports no action, attach or UI code and calls none of the
# mutating or interactive FleetAPI entry points.
AGENT_BANNED_IMPORTS = ("runfleet.core.actions", "runfleet.core.attach", "runfleet.ui", "textual", "rich")
AGENT_BANNED_CALLS = frozenset({"plan", "execute", "attach_now", "attach_start", "forward_port", "write_config",
                                "add_host", "remove_host", "set_general", "connect_login", "doctor"})


def _dotted_imports(path: Path) -> set[str]:
    """Every module a file imports, dotted (``from a.b import c`` gives ``a.b`` and ``a.b.c``), incl. local ones."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def _agent_import_offences(path: Path) -> list[str]:
    bad = []
    for mod in sorted(_dotted_imports(path)):
        parts = mod.split(".")
        if any(mod == b or mod.startswith(b + ".") for b in AGENT_BANNED_IMPORTS):
            bad.append(mod)
        elif parts[:2] == ["runfleet", "backends"] and "actions" in parts:
            bad.append(mod)
    return bad


def _agent_call_offences(path: Path) -> list[str]:
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            sql = (name == "execute" and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == "db")
            if name in AGENT_BANNED_CALLS and not sql:          # db.execute: the read-only sqlite connection
                bad.append(f"{name}( (line {node.lineno})")
    return bad


def test_agent_is_read_only_by_construction() -> None:
    offenders = []
    for path in sorted((PKG_DIR / "agent").rglob("*.py")):
        bad = _agent_import_offences(path) + _agent_call_offences(path)
        if bad:
            offenders.append(f"{path.relative_to(PKG_DIR)}: {bad}")
    assert not offenders, f"agent/ must stay read-only (AGENT_INTERFACE.md §5.1): {offenders}"


def test_agent_checker_detects_violations(tmp_path: Path) -> None:
    probe = tmp_path / "x.py"
    probe.write_text("from runfleet.core import actions\nimport rich.cells\n"
                     "def f(svc):\n    from runfleet.backends.slurm.actions import cancel\n"
                     "    svc.actions.plan('x')\n    svc.doctor('k2')\n    svc.logs.read('k2')\n    db.execute('SELECT 1')\n"
                     "    svc.actions.execute(p)\n", encoding="utf-8")
    assert _agent_import_offences(probe) == ["rich.cells", "runfleet.backends.slurm.actions",
                                             "runfleet.backends.slurm.actions.cancel", "runfleet.core.actions"]
    assert _agent_call_offences(probe) == ["plan( (line 5)", "doctor( (line 6)", "execute( (line 9)"]


def test_backends_are_sans_io() -> None:
    offenders = []
    for path in (PKG_DIR / "backends").rglob("*.py"):
        bad = _imported_top_names(path) & {"asyncio", "subprocess", "textual"}
        if bad:
            offenders.append(f"{path.relative_to(PKG_DIR)}: {sorted(bad)}")
    assert not offenders, f"backends/ must not import asyncio/subprocess/textual: {offenders}"


def _backend_imports(path: Path) -> set[str]:
    """``runfleet.backends`` modules a file imports anywhere (incl. function-local imports), dotted."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("runfleet.backends"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "runfleet.backends":
                found.update(f"runfleet.backends.{a.name}" for a in node.names)
            elif node.module.startswith("runfleet.backends."):
                found.add(node.module)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str) and node.args[0].value.startswith("runfleet.backends")):
            found.add(node.args[0].value)          # importlib.import_module / __import__
    return found


def test_ui_imports_only_the_backend_contract() -> None:
    """§11.5 / DESIGN §5.2.5: the UI must not import backends; only the contract types of ``backends.base``."""
    offenders = []
    for path in (PKG_DIR / "ui").rglob("*.py"):
        bad = {m for m in _backend_imports(path) if m != "runfleet.backends.base"}
        if bad:
            offenders.append(f"{path.relative_to(PKG_DIR)}: {sorted(bad)}")
    assert not offenders, f"ui/ imports backends beyond backends.base: {offenders}"


# POSIX-only local APIs: the whole package must import and run on native Windows (DESIGN §1.2), so these live in
# util/platform.py behind IS_WINDOWS; ui/termbg.py keeps its own termios code behind `sys.platform` guards.
POSIX_ONLY_MODULES = frozenset({"fcntl", "termios", "pty", "tty", "pwd", "grp", "resource", "posix", "readline",
                                "syslog"})
POSIX_ONLY_ATTRS = frozenset({
    ("os", "killpg"), ("os", "getuid"), ("os", "geteuid"), ("os", "getgid"), ("os", "getpgrp"), ("os", "getpgid"),
    ("os", "setpgid"), ("os", "setsid"), ("os", "tcgetpgrp"), ("os", "fork"), ("os", "O_NOCTTY"),
    ("os", "setpgrp"), ("os", "getsid"), ("os", "mkfifo"), ("os", "chown"), ("os", "lchown"), ("os", "getgroups"),
    ("os", "wait3"), ("os", "wait4"), ("os", "WNOHANG"), ("os", "openpty"), ("os", "forkpty"), ("os", "uname"),
    ("os", "getloadavg"), ("os", "sysconf"), ("os", "statvfs"), ("os", "sched_getaffinity"),
    ("signal", "SIGKILL"), ("signal", "SIGTSTP"), ("signal", "SIGHUP"), ("signal", "SIGQUIT"), ("signal", "SIGSTOP"),
    ("signal", "SIGCONT"), ("signal", "SIGUSR1"), ("signal", "SIGUSR2"), ("signal", "SIGCHLD"), ("signal", "SIGALRM"),
    ("signal", "SIGWINCH"), ("signal", "SIGPIPE"), ("signal", "SIGTTOU"), ("signal", "SIGTTIN"), ("signal", "alarm"),
    ("signal", "setitimer"), ("signal", "pause"),
})
# subprocess / asyncio keywords Windows rejects or silently ignores (start_new_session: no new session there, so a
# child would share the TUI's console and get its Ctrl-C): spawn with `platform.background_spawn_kwargs()` instead
POSIX_ONLY_KWARGS = frozenset({"preexec_fn", "start_new_session", "process_group", "pass_fds", "user", "group",
                               "extra_groups", "umask"})
# event-loop methods the Windows (Proactor) loop does not implement
POSIX_ONLY_LOOP_METHODS = frozenset({"add_signal_handler", "remove_signal_handler", "add_reader", "add_writer"})
_SPAWNERS = frozenset({"Popen", "run", "call", "check_call", "check_output", "create_subprocess_exec",
                       "create_subprocess_shell", "subprocess_exec", "subprocess_shell"})
POSIX_ALLOWED = frozenset({Path("util/platform.py"), Path("ui/termbg.py")})


def _posix_only_uses(path: Path) -> list[str]:
    """`import fcntl`, `from os import killpg`, `os.getuid`, `signal.SIGKILL` … anywhere in the file."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in POSIX_ONLY_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in POSIX_ONLY_MODULES:
                found.append(f"from {node.module} import …")
            found += [f"from {node.module} import {a.name}" for a in node.names
                      if (node.module, a.name) in POSIX_ONLY_ATTRS]
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and (node.value.id, node.attr) in POSIX_ONLY_ATTRS):
            found.append(f"{node.value.id}.{node.attr} (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr in POSIX_ONLY_LOOP_METHODS:
            found.append(f".{node.attr} (line {node.lineno})")
        elif isinstance(node, ast.Call):
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if callee in _SPAWNERS:
                found += [f"{callee}({k.arg}=…) (line {node.lineno})" for k in node.keywords
                          if k.arg in POSIX_ONLY_KWARGS]
    return found


def test_posix_only_apis_stay_in_util_platform() -> None:
    offenders = []
    for path in _all_module_files():
        rel = path.relative_to(PKG_DIR)
        if rel in POSIX_ALLOWED:
            continue
        uses = _posix_only_uses(path)
        if uses:
            offenders.append(f"{rel}: {uses}")
    assert not offenders, f"POSIX-only APIs outside util/platform.py (use runfleet.util.platform): {offenders}"


def test_posix_only_checker_detects_violations(tmp_path: Path) -> None:
    probe = tmp_path / "x.py"
    probe.write_text("import os, signal\nfrom os import killpg\ndef f():\n    import fcntl\n"
                     "    os.killpg(os.getuid(), signal.SIGKILL)\n    os.kill(1, signal.SIGTERM)\n", encoding="utf-8")
    assert sorted(_posix_only_uses(probe)) == ["from os import killpg", "import fcntl", "os.getuid (line 5)",
                                               "os.killpg (line 5)", "signal.SIGKILL (line 5)"]


def test_posix_only_checker_detects_spawn_keywords_and_loop_methods(tmp_path: Path) -> None:
    probe = tmp_path / "x.py"
    probe.write_text("import asyncio, subprocess\n"
                     "subprocess.Popen(['a'], start_new_session=True)\n"
                     "subprocess.run(['a'], preexec_fn=None, check=False)\n"
                     "asyncio.create_subprocess_exec('a', stdin=None)\n"
                     "loop = asyncio.get_event_loop(); loop.add_signal_handler(2, print); loop.add_reader(0, print)\n"
                     "dict(start_new_session=True)\n", encoding="utf-8")
    assert sorted(_posix_only_uses(probe)) == [".add_reader (line 5)", ".add_signal_handler (line 5)",
                                               "Popen(start_new_session=…) (line 2)", "run(preexec_fn=…) (line 3)"]


def test_backend_import_checker_detects_violations(tmp_path: Path) -> None:
    probe = tmp_path / "x.py"
    probe.write_text("from runfleet.backends.base import Cadence\nfrom runfleet.backends import base, registry\n"
                     "def f():\n    from runfleet.backends.slurm.parse_sinfo import node_free\n", encoding="utf-8")
    assert _backend_imports(probe) == {"runfleet.backends.base", "runfleet.backends.registry",
                                       "runfleet.backends.slurm.parse_sinfo"}


def test_boundary_checker_detects_violations(tmp_path: Path) -> None:
    probe = tmp_path / "x.py"
    probe.write_text("def f():\n    import asyncio.subprocess\n    from textual.app import App\n", encoding="utf-8")
    assert {"asyncio", "textual"} <= _imported_top_names(probe)


# ---------------------------------------------------------------- one instance of every contract dataclass
def _instances() -> list[object]:
    m, s, t, b, c, st, sv = models, section, tbase, bbase, config, store, service
    err = m.HostError(m.ErrorKind.TIMEOUT, "timed out")
    caps = m.Capabilities(remote_user="usr", uid=1000, login_host="login1", tz_offset=timedelta(hours=-5),
                          clock_skew=timedelta(0), tools=frozenset({"squeue", "sinfo"}))
    metrics = m.HostMetrics(ncpu=8, load=(0.1, 0.2, 0.3), cpu_pct=None, mem_total=1 << 30, mem_available=1 << 29,
                            swap_total=None, swap_free=None, uptime_s=12.0, sampled_at=T0)
    status = m.HostStatus(name="k2", state=m.ConnState.OK)
    key = m.JobKey("k2", "2332861")
    node_alloc = m.NodeAlloc(node="n-40-01", cpu_ids="64-71", mem_bytes=None, gpu_indices=(0, 1), gres=None)
    alloc = m.ResourceAlloc(nodes=1, cpus=8, mem_bytes=None, gpus=2, gpu_type="nvidia_h200_nvl", tres={})
    logref = m.LogRef(stream="stdout", path="/tmp/x.out")
    proc = m.Proc(pid=1, ppid=0, pgid=None, user="usr", etime_s=5, cpu_pct=0.0, mem_pct=0.0, rss_bytes=0,
                  stat="S", args="python train.py")
    gproc = m.GPUProcess(pid=1, gpu_uuid="GPU-x", gpu_index=0, used_mem_mib=1.0, user="usr", command="python",
                         etime_s=5, cpu_pct=1.0, job=None)
    gpu = m.GPU(host="lynx", index=0, uuid=None, name="NVIDIA RTX A6000", util_pct=None, mem_used_mib=None,
                mem_total_mib=None, mem_util_pct=None, temp_c=None, power_w=None, power_limit_w=None,
                fan_pct=None, pstate=None, source="nvidia-smi")
    usage = m.JobGpuUsage(source="sstat", n_gpus=1, util_mean_pct=98.0, util_sum_pct=98.0, util_peak_pct=99.0,
                          mem_used_mib=24870.0, mem_total_mib=None)
    job = m.Job(key=key, kind=m.JobKind.SLURM, name="train", user="usr", state=m.JobState.RUNNING,
                raw_state="RUNNING", job_id_raw="2332861")
    factor = m.PriorityFactor(name="fairshare", value=950.0, weight=100000.0)
    share = m.FairShare(account="lab", user="usr", raw_shares="1", norm_shares=0.5, raw_usage=123456,
                        effective_usage=0.01, fairshare=0.95, level_fs=None)
    pprio = m.PendingPriority(ranked=True, priority=1234.0, partition="gpu", factors=(factor,), dominant="fairshare",
                              others=(("gpu-preempt", 1.0),), fairshare=(share,), hints=("fairshare 0.95",),
                              sampled_at=T0, half_life_s=14 * 86400.0)
    pending = m.Job(key=m.JobKey("k2", "2334564"), kind=m.JobKind.SLURM, name="smoke", user="usr",
                    state=m.JobState.PENDING, raw_state="PENDING", job_id_raw="2334564", priority=pprio)
    gres = m.GresCount(name="gpu", type="a30", total=4, used=1, used_indices="0")
    node = m.Node(name="n-31-04", partitions=("lab",), state="mixed", flags=frozenset(), usable=True,
                  cpus_alloc=None, cpus_idle=None, cpus_other=None, cpus_total=None, mem_total_mib=None,
                  mem_alloc_mib=None, mem_free_mib=None, gres=(gres,), reason=None,
                  jobs=(m.NodeJob(job_id="2332861", user="usr", real_name=None, account="lab", partition="lab",
                                  name="train", state="R", gpus=1, gpus_exact=True, gpu_type="a30", elapsed_s=60,
                                  limit_s=3600, mine=True),), jobs_at=T0)
    part = m.PartitionSummary(name="lab", nodes=1, gpus_total=4, gpus_free=1, gpus_free_planned=0,
                              gpu_types={"a30": (1, 4)}, states={"mixed": 1})
    step = m.JobStep(step_id="2332861.batch", name="batch", state=m.JobState.FAILED, raw_state="FAILED",
                     exit_code=143, exit_signal=0, elapsed=None, start=None, end=None, nodelist=None, ntasks=1,
                     max_rss_bytes=None, ave_cpu=None)
    detail = m.JobDetail(job=job, fields={}, per_node=(node_alloc,), steps=(step,), logs=(logref,), gpu=usage,
                         procs=(), procs_note="", env={}, explanation=(), fetched_at=T0)
    snap = m.HostSnapshot(host="k2", taken_at=T0, caps=caps, jobs={key.id: job}, gpus=(gpu,), procs=(proc,),
                          nodes=(node,), partitions=(part,), metrics=metrics)
    result = m.ActionResult(action_id="slurm.cancel", target=key, ok=True, message="cancelled 2332861")
    detect = m.DetectReport(alias="k2", state=m.ConnState.OK, error=None, caps=caps, mux="user",
                            ssh_summary={}, latency_s=0.2)
    check = m.DoctorCheck(name="probe", ok=True, detail="ok")
    doctor = m.DoctorReport(host="k2", checks=(check,))
    sstatus = m.SectionStatus(name="slurm.squeue", outcome="ok", ok=True, rc=0, ms=30, last_ok=T0, error=err)

    sec = s.Section(name="slurm.squeue", cmd="squeue --me")
    sres = s.SectionResult(name="slurm.squeue", outcome=s.SectionOutcome.OK, rc=0, stdout="x", stderr="", ms=1)
    script = s.ProbeScript(sections=(sec,), nonce="0123456789ab")
    spec = s.StreamSpec(name="log.tail", cmd="tail -f x")

    run = t.RunResult(rc=0, stdout=b"", stderr=b"", elapsed_s=0.1, timed_out=False)
    pres = t.ProbeResult(run=run, sections={sres.name: sres}, error=None)

    hcfg = c.HostConfig(name="k2", ssh="k2")
    general = c.GeneralSettings()
    fleet = c.FleetConfig(hosts=(hcfg,))
    cad = b.Cadence(group="slurm.jobs", interval=15.0)
    ctx = b.HostContext(host=hcfg, general=general, caps=caps, now=T0)
    mview = b.MonitorView()
    cout = b.CollectorOutput()
    param = b.ActionParam(name="escalate", kind="bool", default=True, label="send SIGKILL to survivors after")
    aspec = b.ActionSpec(id="slurm.cancel", label="Cancel job", mutating=True, confirm=True, params=(param,))
    plan = b.ActionPlan(spec=aspec, host="k2", targets=(key,), skipped=(), params={}, sections=(sec,),
                        preview="scancel 2332861", confirm_title="Cancel 1 job on k2?", confirm_body="")

    cg = m.CgroupStats(mem_current=1, mem_max=None, shmem=0, oom_kill=0, cpu_usage_usec=10)
    ninsp = m.NodeInspect(node="n-37-08", target_job="724655", landed_job="724655", gpus_of_target=True, gpus=(gpu,),
                          procs=(proc,), step_of_pid={1: "724655.batch"}, cgroups={"724655": cg})
    live = m.JobLive(key=key, mode="ssh", sampled_at=T0, nodes=(ninsp,), cost_ms=670, clean=True,
                     audit_detail="clean")
    offer = m.AttachOffer(node="s71-02", reason="lyra refuses ssh to compute nodes", title="Read with srun?",
                          body="leaves one sacct row", preview="srun --jobid=1 --overlap …")
    astate = m.AttachState(key=key, status=m.AttachStatus.LIVE, setting="ssh", mode="ssh", live=live, offer=offer)
    aplan = b.AttachPlan(key=key, mode="ssh", job_id_raw="2332861", nodes=("n-40-01",), sections=(sec,),
                         inner={sec.name: script}, preview="ssh n-40-01 …")
    hstate = st.HostState(config=hcfg, status=status, snapshot=snap)
    change = st.ChangeSet(host="k2")
    lupd = sv.LogUpdate()
    lpeek = sv.LogPeek(lines=(), size=None, mtime=None, fetched_at=T0)

    return [err, caps, metrics, status, key, node_alloc, alloc, logref, proc, gproc, gpu, usage, job, gres, node,
            part, step, detail, snap, result, detect, check, doctor, sstatus, sec, sres, script, spec, run, pres,
            c.SlurmSettings(), c.WorkstationSettings(), hcfg, general, c.UISettings(), fleet, cad, ctx, mview,
            cout, param, aspec, plan, hstate, change, lupd, lpeek, c.NotifySettings(), cg, ninsp, live, offer, astate,
            aplan, *(node.jobs or ()), factor, share, pprio, pending]


def test_every_contract_dataclass_constructs() -> None:
    built = {type(o) for o in _instances()}
    declared = {
        obj
        for mod in CONTRACT_MODULES
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if dataclasses.is_dataclass(obj) and obj.__module__ == mod.__name__
    }
    missing = sorted(cls.__qualname__ for cls in declared - built)
    assert not missing, f"no instance constructed for: {missing}"


def test_contract_dataclasses_are_frozen_and_slotted() -> None:
    mutable_ok = {store.HostState}   # §3.7: plain @dataclass (core mutates it)
    for obj in _instances():
        cls = type(obj)
        if cls in mutable_ok:
            continue
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, cls.__qualname__
        assert "__slots__" in cls.__dict__, cls.__qualname__
        first = dataclasses.fields(obj)[0].name   # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, first, getattr(obj, first))


# ---------------------------------------------------------------- small behavioural checks of the contracts
def test_job_state_groups_are_total() -> None:
    for st in models.JobState:
        assert isinstance(st.group, models.StateGroup)
    assert models.JobState.RUNNING.group is models.StateGroup.ACTIVE
    assert not models.JobState.PENDING.is_terminal
    assert not models.JobState.UNKNOWN.is_terminal
    assert models.JobState.EXITED.group is models.StateGroup.ENDED and models.JobState.EXITED.is_terminal
    assert models.JobState.CANCELLED.is_terminal


def test_job_key_label_and_order() -> None:
    task = models.JobKey("lynx", "p642229-1788696459")
    slurm = models.JobKey("k2", "31270818_[1-24%1]")
    assert task.label == "p642229"
    assert slurm.label == "31270818_[1-24%1]"
    assert str(task) == "lynx:p642229-1788696459"
    assert sorted([task, slurm]) == [slurm, task]


def test_section_owner_and_result_ok() -> None:
    assert section.Section(name="process.kill.642229", cmd="true").owner == "process"
    assert section.Section(name="log.tail", cmd="true").owner == "log"
    for outcome in section.SectionOutcome:
        r = section.SectionResult(name="x.y", outcome=outcome, rc=0, stdout="", stderr="", ms=None)
        assert r.ok == (outcome in (section.SectionOutcome.OK, section.SectionOutcome.EMPTY))


def test_monitor_view_gpus_age() -> None:
    assert bbase.MonitorView().gpus_age(T0) is None
    assert bbase.MonitorView(gpus_at=T0).gpus_age(T0 + timedelta(seconds=7)) == 7.0


def test_job_default_alloc_is_empty() -> None:
    job = models.Job(key=models.JobKey("h", "1"), kind=models.JobKind.TASK, name="n", user="u",
                     state=models.JobState.RUNNING, raw_state="", job_id_raw="1")
    assert job.alloc.gpus == 0 and job.alloc.tres == {}


def test_collector_abcs_are_abstract() -> None:
    assert inspect.isabstract(bbase.Collector)
    assert inspect.isabstract(bbase.SchedulerCollector)
    assert bbase.MonitorCollector.role is bbase.Role.MONITOR
    assert bbase.SchedulerCollector.role is bbase.Role.SCHEDULER


def test_fleet_service_implements_every_fleet_api_method() -> None:
    api = {n for n, v in vars(service.FleetAPI).items() if inspect.isfunction(v) and not n.startswith("_")}
    impl = {n for n, v in vars(service.FleetService).items() if inspect.isfunction(v)}
    assert api and api <= impl, sorted(api - impl)


# ---------------------------------------------------------------- P2 (session transport): exactly what was decided
def test_mux_values_are_frozen() -> None:
    """P2 decision: MuxMode and the config's `mux` gain "session", nothing else."""
    assert get_args(models.MuxMode) == ("user", "managed", "off", "session", "local", "none")
    hints = get_type_hints(config.HostConfig)
    assert get_args(hints["mux"]) == ("auto", "user", "managed", "off", "session")


def test_transport_contract_members_are_frozen() -> None:
    """§3.5 is unchanged by P2: the session login is an optional protocol of transport/session.py, not a member of
    the frozen Transport contract."""
    members = {n for n in vars(tbase.Transport) if not n.startswith("_")}
    assert members == {"mux", "ssh_log", "prepare", "check_master", "probe", "stream", "interactive_argv",
                       "before_connect", "connect_argv", "reset_master", "close", "forward_blocker", "forward",
                       "cancel_forward"}
    assert {n for n, v in vars(tbase).items() if isinstance(v, type) and v.__module__ == tbase.__name__} == {
        "RunResult", "ProbeResult", "StreamHandle", "Transport"}


def test_interactive_login_capability() -> None:
    """The session login (§4.8) and the one FleetAPI method the UI reaches it through (§3.7)."""
    from runfleet.transport.session import InteractiveLogin, SessionTransport

    assert {n for n in vars(InteractiveLogin) if not n.startswith("_")} == {"login"}
    assert str(inspect.signature(InteractiveLogin.login)) == "(self) -> 'int'"
    assert str(inspect.signature(SessionTransport.login)) == "(self) -> 'int'"
    assert str(inspect.signature(service.FleetAPI.connect_login)) == "(self, host: 'str') -> 'Callable[[], int] | None'"


def test_config_error_carries_key_path() -> None:
    e = errors.ConfigError("host[2].poll_jobs", "expected duration")
    assert isinstance(e, errors.RunFleetError)
    assert e.key_path == "host[2].poll_jobs"
    assert str(e) == "host[2].poll_jobs: expected duration"
    assert issubclass(errors.ActionError, errors.RunFleetError)


def test_errors_module_holds_exceptions_only() -> None:
    for name, obj in vars(errors).items():
        if inspect.isclass(obj) and obj.__module__ == errors.__name__:
            assert issubclass(obj, Exception), name


# ---------------------------------------------------------------- clock (FROZEN, implemented at M0)
def test_real_clock_is_aware_utc() -> None:
    now = cclock.RealClock().now()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)


def test_manual_clock_rejects_naive_start() -> None:
    with pytest.raises(ValueError):
        cclock.ManualClock(datetime(2026, 1, 1))


async def test_manual_clock_wakes_sleepers_in_deadline_order(manual_clock: cclock.ManualClock) -> None:
    start = manual_clock.now()
    woke: list[str] = []

    async def sleeper(tag: str, seconds: float) -> None:
        await manual_clock.sleep(seconds)
        woke.append(tag)

    tasks = [asyncio.create_task(sleeper("b", 10)), asyncio.create_task(sleeper("a", 5)),
             asyncio.create_task(sleeper("c", 30))]
    await asyncio.sleep(0)
    manual_clock.advance(4.9)
    await asyncio.sleep(0)
    assert woke == []
    manual_clock.advance(5.1)
    for _ in range(3):
        await asyncio.sleep(0)
    assert woke == ["a", "b"]
    assert manual_clock.monotonic() == pytest.approx(10.0)
    assert manual_clock.now() == start + timedelta(seconds=10)
    manual_clock.advance(20)
    await asyncio.gather(*tasks)
    assert woke == ["a", "b", "c"]
    with pytest.raises(ValueError):
        manual_clock.advance(-1)


def test_clocks_satisfy_protocol() -> None:
    def use(c: cclock.Clock) -> datetime:
        return c.now()

    assert use(cclock.RealClock()).tzinfo is not None
    assert use(cclock.ManualClock(T0)) == T0
