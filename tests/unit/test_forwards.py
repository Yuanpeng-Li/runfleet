"""[WP1] Port forwards (§7.15): URL discovery, local port choice, `ssh -O forward` argv and FleetService bookkeeping.

No network: the OpenSSH part runs a fake `ssh` script that only records its argv; the service part uses FakeTransport.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from runfleet.config import FleetConfig, GeneralSettings, HostConfig
from runfleet.core import forwards
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.errors import RunFleetError
from runfleet.models import ConnState, ErrorKind, HostError, HostStatus, JobKey
from runfleet.sshconfig import parse_ssh_g
from runfleet.transport import mux
from runfleet.transport.fake import FakeTransport
from runfleet.transport.local import LocalTransport
from runfleet.transport.openssh import OpenSSHTransport
from tests.conftest import CLOCK_START
from tests.unit.stubs import STUB_HOOKS, run_for

G_USER = "user u\nhostname k2.example.edu\nport 22\ncontrolmaster auto\ncontrolpath /home/u/.ssh/cm-1\n" \
         "controlpersist 600\n"
G_NOPATH = "user u\nhostname t\nport 22\ncontrolpersist 600\n"

# ---------------------------------------------------------------- URLs in logs
TB = "TensorBoard 2.20.0 at http://n-38-01:6006/ (Press CTRL+C to quit)"
JUPYTER = ["[I 10:00:01 ServerApp] Jupyter Server 2.14 is running at:",
           "[I 10:00:01 ServerApp] http://n-40-01:8888/lab?token=abc123def",
           "[I 10:00:01 ServerApp]     http://127.0.0.1:8888/lab?token=abc123def"]
GRADIO = "\x1b[32mRunning on local URL:  http://127.0.0.1:7860\x1b[0m"


def test_find_urls_tensorboard_jupyter_gradio() -> None:
    cands = forwards.find_urls([TB, *JUPYTER, GRADIO, "epoch 3 loss 0.1", "see https://pytorch.org/docs for help"])
    by_port = {c.port: c for c in cands}
    assert set(by_port) == {6006, 8888, 7860}                     # a URL without a port is no local service
    assert by_port[6006].label == "TensorBoard" and by_port[6006].target == "n-38-01" and by_port[6006].path == ""
    assert by_port[8888].label == "Jupyter" and by_port[8888].path == "/lab?token=abc123def"   # token kept
    assert by_port[7860].target == "127.0.0.1" and by_port[7860].label == "Gradio"            # ANSI stripped


def test_find_urls_prefers_the_url_with_a_path_and_ignores_bad_ports() -> None:
    cands = forwards.find_urls(["serving http://localhost:8000", "open http://localhost:8000/ui?x=1.",
                                "bad http://h:99999/x"])
    assert [(c.port, c.path) for c in cands] == [(8000, "/ui?x=1")]   # trailing "." is punctuation, not the URL


def test_local_url_keeps_scheme_path_and_query() -> None:
    assert forwards.local_url(6006) == "http://localhost:6006/"
    assert forwards.local_url(18888, path="/lab?token=t") == "http://localhost:18888/lab?token=t"
    assert forwards.local_url(443, scheme="https", path="?a=1") == "https://localhost:443?a=1"


def test_pick_local_port_prefers_the_remote_number() -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken_port = busy.getsockname()[1]
        other = forwards.pick_local_port(taken_port)
        assert other != taken_port and forwards.port_is_free(other)
    free = forwards.pick_local_port(None)
    assert forwards.pick_local_port(free) == free                    # free here: the same number
    assert forwards.pick_local_port(free, taken=[free]) != free      # used by another forward of ours


# ---------------------------------------------------------------- OpenSSH: `-O forward` on the master
FAKE_SSH = """#!/bin/sh
printf '%s\\n' "$*" >> "$RF_FAKE_LOG"
if [ -n "$RF_FAKE_FAIL" ]; then echo "mux_client_forward: forwarding request failed: Port forwarding failed" >&2; exit 255; fi
exit 0
"""


@pytest.fixture
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "ssh"
    exe.write_text(FAKE_SSH, encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("RF_FAKE_LOG", str(tmp_path / "argv.log"))
    return exe


def _prepared(exe: Path, tmp_path: Path, g: str, **kw: object) -> OpenSSHTransport:
    t = OpenSSHTransport(HostConfig("k2", "k2", **kw), ssh=str(exe), mux_dir=tmp_path,   # type: ignore[arg-type]
                         log_path=tmp_path / "l")
    t.settings = parse_ssh_g(g, "k2")
    t.plan = mux.decide(t.settings, t.config, tmp_path)
    return t


async def test_openssh_forward_argv_and_run(fake_ssh: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = _prepared(fake_ssh, tmp_path, G_USER, ssh_options=("Compression=yes",))
    assert t.forward_blocker() is None
    assert t.forward_argv("forward", 6006, "n-38-01", 6006) == [
        str(fake_ssh), "-O", "forward", "-L", "127.0.0.1:6006:n-38-01:6006", "-o", "ControlMaster=no",
        "-o", "ControlPath=/home/u/.ssh/cm-1", "-o", "Compression=yes", "--", "k2"]
    assert await t.forward(6006, "n-38-01", 6006) is None
    assert await t.cancel_forward(6006, "n-38-01", 6006) is None
    log = (tmp_path / "argv.log").read_text().splitlines()
    assert log[0].startswith("-O forward -L 127.0.0.1:6006:n-38-01:6006") and log[1].startswith("-O cancel -L")
    monkeypatch.setenv("RF_FAKE_FAIL", "1")
    err = await t.forward(6007, "n-38-01", 6006)
    assert err is not None and err.kind is ErrorKind.MUX and "Port forwarding failed" in err.summary


def test_openssh_without_a_master_cannot_forward(tmp_path: Path) -> None:
    t = _prepared(Path("ssh"), tmp_path, G_NOPATH, mux="user")       # user mode, no ControlPath: no master
    assert t.forward_argv("forward", 1, "n", 1) is None and "ControlPath" in (t.forward_blocker() or "")
    fresh = OpenSSHTransport(HostConfig("k2", "k2"), mux_dir=tmp_path, log_path=tmp_path / "l")
    assert fresh.forward_blocker() == "not connected yet"
    assert LocalTransport(HostConfig("me", "local")).forward_blocker() is not None


# ---------------------------------------------------------------- FleetService
@pytest.fixture
async def svc(tmp_path: Path) -> AsyncIterator[tuple[FleetService, dict[str, FakeTransport], ManualClock]]:
    clock = ManualClock(CLOCK_START)
    made: dict[str, FakeTransport] = {}

    def factory(cfg: HostConfig) -> FakeTransport:
        t = made.get(cfg.name)
        if t is None:
            t = made[cfg.name] = FakeTransport(cfg.name, mux_mode="managed", clock=clock)
            t.set_result("detect.id", "user=me")
            t.set_result("slurm.squeue", "101 R")
        return t

    hosts = (HostConfig("k2", "k2", backend="slurm"), HostConfig("vega", "vega", backend="slurm"))
    service = FleetService(FleetConfig(hosts=hosts, general=GeneralSettings()), factory, clock, None)
    service.hooks = STUB_HOOKS
    service.lock_path = tmp_path / "instance.lock"
    await service.start()
    await run_for(clock, 2)
    yield service, made, clock
    await service.stop()


async def test_forward_port_records_reuses_and_cancels(svc: tuple[FleetService, dict[str, FakeTransport],
                                                                  ManualClock]) -> None:
    service, made, _clock = svc
    assert service.store.hosts["k2"].status.state is ConnState.OK
    free = forwards.pick_local_port(None)
    fwd = await service.forward_port("k2", "n-38-01", free, job=JobKey("k2", "101"), path="/lab?token=t",
                                     label="Jupyter")
    assert fwd.local_port == free and fwd.url == f"http://localhost:{free}/lab?token=t"
    assert (free, "n-38-01", free) in made["k2"].forwarded
    assert await service.forward_port("k2", "n-38-01", free) is fwd           # same target: the same forward
    assert service.forwards() == [fwd]
    await service.cancel_forward(free)
    assert service.forwards() == [] and not made["k2"].forwarded


async def test_forward_retries_on_another_port_and_reports_failures(
        svc: tuple[FleetService, dict[str, FakeTransport], ManualClock]) -> None:
    service, made, _clock = svc
    t = made["k2"]
    t.forward_errors.append(HostError(ErrorKind.MUX, "port forward failed: Address already in use"))
    free = forwards.pick_local_port(None)
    fwd = await service.forward_port("k2", "n1", free)
    assert fwd.local_port != free                                              # second try: any free port
    t.forward_errors.extend([HostError(ErrorKind.MUX, "port forward failed: nope")] * 2)
    with pytest.raises(RunFleetError, match="nope"):
        await service.forward_port("k2", "n2", 9)
    t.forward_block = "no ssh master to forward through"
    assert service.forward_blocker("k2") == "no ssh master to forward through"
    with pytest.raises(RunFleetError, match="no ssh master"):
        await service.forward_port("k2", "n3", 9)


async def test_forward_blocked_when_disconnected_and_cancelled_on_stop(
        svc: tuple[FleetService, dict[str, FakeTransport], ManualClock]) -> None:
    service, made, _clock = svc
    service.store.set_status("vega", HostStatus(name="vega", state=ConnState.NEEDS_AUTH))
    assert "press c" in (service.forward_blocker("vega") or "")
    fwd = await service.forward_port("k2", "n1", forwards.pick_local_port(None))
    await service.stop()                                                       # exit: every forward cancelled
    assert service.forwards() == [] and f"cancel_forward:{fwd.local_port}:n1:{fwd.remote_port}" in made["k2"].calls


async def test_job_shell_command_is_the_shell_argv_command(
        svc: tuple[FleetService, dict[str, FakeTransport], ManualClock]) -> None:
    service, _made, _clock = svc
    assert service.job_shell_command("k2", JobKey("k2", "101")) == "cd /w/101 && exec $SHELL -l"   # stub collector
    assert service.shell_argv("k2", JobKey("k2", "101"))[-1] == "cd /w/101 && exec $SHELL -l"
    assert service.job_shell_command("k2", JobKey("k2", "nope")) is None


async def test_forward_survives_a_cancelled_caller_and_is_cancelled_on_stop(
        svc: tuple[FleetService, dict[str, FakeTransport], ManualClock]) -> None:
    import asyncio

    service, made, _clock = svc
    t = made["k2"]
    gate = asyncio.Event()
    real = t.forward

    async def slow(local: int, node: str, remote: int) -> HostError | None:
        await gate.wait()                       # the master installs the listener while the caller is gone
        return await real(local, node, remote)

    t.forward = slow  # type: ignore[method-assign]
    caller = asyncio.ensure_future(service.forward_port("k2", "n1", forwards.pick_local_port(None)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    caller.cancel()
    await asyncio.sleep(0)
    gate.set()
    await service.stop()                        # waits for the forward, records it, cancels it
    assert t.forwarded == set() and any(c.startswith("cancel_forward:") for c in t.calls)


async def test_cancel_failure_keeps_the_forward_and_reforward_restores_a_lost_one(
        svc: tuple[FleetService, dict[str, FakeTransport], ManualClock]) -> None:
    service, made, _clock = svc
    t = made["k2"]
    fwd = await service.forward_port("k2", "n1", forwards.pick_local_port(None))
    t.cancel_errors.append(HostError(ErrorKind.MUX, "port cancel failed: timed out"))
    with pytest.raises(RunFleetError, match="timed out"):
        await service.cancel_forward(fwd.local_port)
    assert service.forwards() == [fwd]          # maybe still listening: listed, and stop() tries again
    t.forwarded.clear()                          # the master was replaced (reconnect): its listener went with it
    assert await service.forward_port("k2", "n1", fwd.remote_port) is fwd
    assert (fwd.local_port, "n1", fwd.remote_port) in t.forwarded              # asked again: listening again
    t.cancel_errors.append(HostError(ErrorKind.MUX, "port cancel failed: No such file or directory"))
    await service.cancel_forward(fwd.local_port)                               # no master: the forward is gone
    assert service.forwards() == []
