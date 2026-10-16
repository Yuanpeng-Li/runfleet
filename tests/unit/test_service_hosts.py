"""FleetService runtime host management (Settings §7.13, auto-remember §8.4) over FakeTransports: add_host starts a
poller, remove_host cancels it without leaking tasks, reorder, set_general applied live, remember_hosts, and the
config file edits behind them."""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from runfleet.config import FleetConfig, GeneralSettings, HostConfig
from runfleet.core.clock import ManualClock
from runfleet.core.service import FleetService
from runfleet.errors import ConfigError
from runfleet.models import ConnState, ErrorKind, JobKey
from runfleet.transport.base import Transport
from runfleet.transport.fake import FakeTransport
from tests.conftest import CLOCK_START, is_root
from tests.unit.stubs import STUB_HOOKS, run_for, settle

K2 = HostConfig(name="k2", ssh="k2", backend="slurm")
LYNX = HostConfig(name="lynx", ssh="lynx", backend="workstation")


class World:
    def __init__(self, hosts: tuple[HostConfig, ...], tmp: Path, *, general: GeneralSettings | None = None,
                 config_path: Path | None = None) -> None:
        self.clock = ManualClock(CLOCK_START)
        self.transports: dict[str, FakeTransport] = {}
        self.made: list[str] = []
        self.svc = FleetService(FleetConfig(hosts=hosts, general=general or GeneralSettings()), self.factory,
                                self.clock, None)
        self.svc.hooks = STUB_HOOKS
        self.svc.lock_path = tmp / "instance.lock"
        self.svc.config_path = config_path
        self.svc.alias_source = lambda: ["k2", "lynx", "vega", "edge1"]

    def factory(self, host: HostConfig) -> Transport:
        self.made.append(host.name)
        t = self.transports.get(host.name)
        if t is None:
            t = FakeTransport(host.name, clock=self.clock)   # type: ignore[arg-type]
            t.set_result("detect.id", "user=me")
            t.set_result("slurm.squeue", "101 R\n102 PD")
            t.set_result("process.ps", "642229 0")
            t.set_result("nvidia.gpu", "0 90\n1 5")
            self.transports[host.name] = t
        return t

    def state(self, host: str) -> ConnState:
        return self.svc.store.hosts[host].status.state


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[Callable[..., World]]:
    made: list[World] = []

    def make(*hosts: HostConfig, **kw: object) -> World:
        w = World(hosts, tmp_path, **kw)        # type: ignore[arg-type]
        made.append(w)
        return w

    yield make
    for w in made:
        await w.svc.stop()


def _poller_tasks() -> list[asyncio.Task[object]]:
    return [t for t in asyncio.all_tasks() if not t.done() and "HostPoller.run" in repr(t.get_coro())]


# ---------------------------------------------------------------- add / remove / reorder (in memory, like --demo)
async def test_add_host_starts_polling_at_once(world: Callable[..., World]) -> None:
    w = world(K2)
    await w.svc.start()
    await settle()
    changes: list[str] = []
    w.svc.store.subscribe(lambda cs: changes.append(cs.host))
    running = w.svc.add_host(HostConfig(name="vega", ssh="vega"))
    assert running.name == "vega" and list(w.svc.store.hosts) == ["k2", "vega"] and "vega" in changes
    await run_for(w.clock, 2)
    assert w.state("vega") is ConnState.OK and w.transports["vega"].count("poll") >= 1
    assert [h.name for h in w.svc.saved_hosts()] == ["k2", "vega"]
    assert len(_poller_tasks()) == 2
    assert w.svc.add_host(HostConfig(name="vega", ssh="vega")).name == "vega"     # idempotent: no second poller
    assert len(_poller_tasks()) == 2 and [h.name for h in w.svc.saved_hosts()] == ["k2", "vega"]


async def test_add_host_before_start_and_save_false(world: Callable[..., World]) -> None:
    w = world(K2)
    w.svc.add_host(HostConfig(name="vega", ssh="vega"), save=False)
    assert "vega" in w.svc.store.hosts and [h.name for h in w.svc.saved_hosts()] == ["k2"]
    assert not _poller_tasks()                                       # pollers start with start()
    await w.svc.start()
    await run_for(w.clock, 2)
    assert w.state("vega") is ConnState.OK


async def test_add_host_rejects_bad_names_and_targets(world: Callable[..., World]) -> None:
    w = world(K2)
    for bad in (HostConfig(name="a b", ssh="ab"), HostConfig(name="x", ssh="-oProxyCommand=evil"),
                HostConfig(name="y", ssh="")):
        with pytest.raises(ConfigError):
            w.svc.add_host(bad)
    assert list(w.svc.store.hosts) == ["k2"]


async def test_add_host_refuses_a_name_taken_by_another_endpoint(world: Callable[..., World]) -> None:
    """UI-22: `name = "lynx"` with another ssh target used to return the existing host as if it had been added."""
    w = world(K2, LYNX)
    with pytest.raises(ConfigError, match="already exists"):
        w.svc.add_host(HostConfig(name="lynx", ssh="me@other-box"))
    assert w.svc.store.hosts["lynx"].config.ssh == "lynx"
    assert w.svc.add_host(HostConfig(name="lynx", ssh="lynx")).ssh == "lynx"          # same endpoint: idempotent


async def test_remove_host_survives_a_cancelled_caller(world: Callable[..., World],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """UI-03: Settings closing mid-removal cancels the caller; the teardown still finishes (no host left in the store
    without its poller), and adding the host back while it runs is refused instead of reporting a false success."""
    w = world(K2, LYNX)
    await w.svc.start()
    await run_for(w.clock, 2)
    gate = asyncio.Event()
    real_close_host = w.svc.logs.close_host

    async def slow_close_host(host: str) -> None:
        await gate.wait()
        await real_close_host(host)

    monkeypatch.setattr(w.svc.logs, "close_host", slow_close_host)
    caller = asyncio.ensure_future(w.svc.remove_host("lynx", save=False))
    await settle()
    caller.cancel()
    await settle()
    assert caller.cancelled() and "lynx" in w.svc.store.hosts
    with pytest.raises(ConfigError, match="still being removed"):
        w.svc.add_host(LYNX, save=False)
    gate.set()
    await settle()
    assert list(w.svc.store.hosts) == ["k2"] and len(_poller_tasks()) == 1 and not w.svc._removing
    w.svc.add_host(LYNX, save=False)
    assert "lynx" in w.svc.store.hosts


async def test_remove_host_cancels_its_poller_and_leaks_nothing(world: Callable[..., World],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    w = world(K2, LYNX)
    await w.svc.start()
    await run_for(w.clock, 2)
    assert len(_poller_tasks()) == 2 and w.svc._lock_fds and "lynx" in w.svc._host_locks
    closed: list[str] = []
    real_close_host = w.svc.logs.close_host

    async def close_host(host: str) -> None:
        closed.append(host)
        await real_close_host(host)

    monkeypatch.setattr(w.svc.logs, "close_host", close_host)
    changes = []
    w.svc.store.subscribe(changes.append)
    await w.svc.remove_host("lynx")
    assert list(w.svc.store.hosts) == ["k2"] and closed == ["lynx"]
    assert "close" in w.transports["lynx"].calls
    assert len(_poller_tasks()) == 1 and "lynx" not in w.svc._pollers and "lynx" not in w.svc._tasks
    assert "lynx" not in w.svc._host_locks and len(w.svc._lock_fds) == 1
    assert any(cs.host == "lynx" and cs.jobs_removed == frozenset({"p642229-1000"}) for cs in changes)
    n = len(w.transports["lynx"].probes)
    await run_for(w.clock, 60)
    assert len(w.transports["lynx"].probes) == n                       # nothing polls it any more
    assert [h.name for h in w.svc.saved_hosts()] == ["k2"]
    await w.svc.remove_host("nope")                                    # unknown: no error


async def test_removed_then_added_again_polls_with_a_fresh_transport_and_lock(world: Callable[..., World]) -> None:
    w = world(K2, LYNX)
    await w.svc.start()
    await run_for(w.clock, 2)
    await w.svc.remove_host("lynx")
    w.svc.add_host(HostConfig(name="lynx", ssh="lynx"))
    await run_for(w.clock, 2)
    assert w.state("lynx") is ConnState.OK and w.made.count("lynx") == 2 and "lynx" in w.svc._host_locks
    assert list(w.svc.store.hosts) == ["k2", "lynx"]


async def test_reorder_hosts(world: Callable[..., World]) -> None:
    w = world(K2, LYNX, HostConfig(name="vega", ssh="vega"))
    await w.svc.start()
    hosts_dict = w.svc.store.hosts
    w.svc.reorder_hosts(["vega", "k2"])
    assert list(w.svc.store.hosts) == ["vega", "k2", "lynx"] and w.svc.store.hosts is hosts_dict
    assert [h.name for h in w.svc.saved_hosts()] == ["vega", "k2", "lynx"]


# ---------------------------------------------------------------- general settings, live
async def test_set_general_applies_live(world: Callable[..., World]) -> None:
    w = world(K2, general=GeneralSettings(idle_after=600, idle_pause_after=7200))
    await w.svc.start()
    await run_for(w.clock, 2)
    w.svc.set_general(readonly=True)
    assert w.svc.readonly and w.svc.config.general.readonly
    spec = next(s for s in w.svc.actions_for([JobKey("k2", "101")]) if s.mutating)
    assert spec.disabled_reason == "read-only mode"
    w.svc.set_general(refresh_multiplier=2.0)
    await run_for(w.clock, 20)
    assert w.svc.store.hosts["k2"].status.tick_interval == 30
    w.svc.set_general(idle_pause_after=900.0, pager="most")
    assert w.svc.local_pager_argv(Path("/x"))[0] == "most"
    await run_for(w.clock, 960, step=5.0)
    assert w.svc.store.idle_paused                                     # the shorter pause took effect
    with pytest.raises(ConfigError):
        w.svc.set_general(readonly="yes")


# ---------------------------------------------------------------- file-backed
CONFIG = """# my fleet
[defaults.slurm]
poll_jobs = "30s"   # gentle

# the main cluster
[[host]]
name = "k2"
"""


async def test_file_backed_add_remove_reorder_keep_comments(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    w = world(K2, config_path=path)
    await w.svc.start()
    running = w.svc.add_host(HostConfig(name="vega", ssh="vega"))
    assert running.slurm.poll_jobs == 30                               # [defaults.slurm] of the file applies
    assert path.read_text() == CONFIG + '\n[[host]]\nname = "vega"\n'
    w.svc.add_host(HostConfig(name="me_box", ssh="me@box"))
    w.svc.reorder_hosts(["me_box", "k2", "vega"])
    assert [h["name"] for h in tomllib.loads(path.read_text())["host"]] == ["me_box", "k2", "vega"]
    assert "# the main cluster\n[[host]]\nname = \"k2\"" in path.read_text()
    assert list(w.svc.store.hosts) == ["me_box", "k2", "vega"]
    await w.svc.remove_host("k2")
    text = path.read_text()
    assert "k2" not in text and "# the main cluster" not in text and "poll_jobs = \"30s\"   # gentle" in text
    assert [h.name for h in w.svc.saved_hosts()] == ["me_box", "vega"]


async def test_broken_config_changes_nothing(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    broken = '[[host]]\nname = "k2"\n[oops\n'
    path.write_text(broken)
    w = world(K2, config_path=path)
    await w.svc.start()
    with pytest.raises(ConfigError, match="invalid TOML"):
        w.svc.add_host(HostConfig(name="vega", ssh="vega"))
    assert "vega" not in w.svc.store.hosts and len(_poller_tasks()) == 1
    with pytest.raises(ConfigError):
        await w.svc.remove_host("k2")
    assert "k2" in w.svc.store.hosts
    with pytest.raises(ConfigError):
        w.svc.set_general(mouse=False)
    assert w.svc.config.general.mouse is True and path.read_text() == broken


async def test_set_general_is_persisted(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    w = world(K2, config_path=path)
    w.svc.set_general(default_view="gpus", idle_pause_after=28800.0)
    text = path.read_text()
    assert text.startswith('# my fleet\n[general]\ndefault_view = "gpus"\nidle_pause_after = "8h"\n\n'
                           '[defaults.slurm]\npoll_jobs = "30s"   # gentle\n')


async def test_set_notify_is_persisted_and_live(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    w = world(K2, config_path=path)
    n = w.svc.set_notify(enabled=False, events=("ended_bad", "gpu_idle"))
    assert n is w.svc.config.notify and not n.enabled and n.events == ("ended_bad", "gpu_idle")
    assert path.read_text() == ('# my fleet\n[defaults.slurm]\npoll_jobs = "30s"   # gentle\n\n[notify]\n'
                                'enabled = false\nevents = ["ended_bad", "gpu_idle"]\n\n# the main cluster\n'
                                '[[host]]\nname = "k2"\n')
    with pytest.raises(ConfigError, match="notify.terminal"):
        w.svc.set_notify(terminal="beep")
    with pytest.raises(ConfigError, match="unknown setting"):
        w.svc.set_notify(sound=True)
    assert w.svc.config.notify.terminal == "auto"
    assert w.svc.set_notify(command_timeout="2s").command_timeout == 2.0          # live = what a reload gives
    assert 'command_timeout = "2s"' in path.read_text()
    w.svc.config_path = None                                          # --demo: this run only
    assert w.svc.set_notify(terminal="bell").terminal == "bell" and 'terminal' not in path.read_text()


# ---------------------------------------------------------------- remember hosts named on the command line
async def test_remember_hosts_saves_answering_hosts_once(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    vega, lynx, dead = (HostConfig(name=n, ssh=n) for n in ("vega", "lynx", "dead"))
    w = world(K2, vega, lynx, dead, config_path=path)
    w.factory(lynx).fail_next(ErrorKind.AUTH)
    w.factory(dead).fail_next(ErrorKind.DNS, times=50)
    w.svc.remember_hosts(["k2", "vega", "lynx", "dead"])
    await w.svc.start()
    await run_for(w.clock, 3)
    assert w.state("lynx") is ConnState.NEEDS_AUTH and w.state("dead") is ConnState.UNREACHABLE
    assert [h["name"] for h in tomllib.loads(path.read_text())["host"]] == ["k2", "vega", "lynx"]
    saved = [n for n in w.svc.notices if n.startswith("saved ")]
    assert saved == ["saved vega, lynx — manage in Settings (,)"]
    await run_for(w.clock, 60)
    assert [n for n in w.svc.notices if n.startswith("saved ")] == saved      # one notice, dead never saved
    assert "dead" not in path.read_text()


async def test_remember_hosts_creates_the_config(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "new" / "config.toml"
    w = world(HostConfig(name="me_box", ssh="me@box"), config_path=path)
    w.svc.remember_hosts(["me_box"])
    await w.svc.start()
    await run_for(w.clock, 2)
    assert tomllib.loads(path.read_text())["host"] == [{"name": "me_box", "ssh": "me@box"}]


async def test_remember_hosts_respects_the_setting_and_known_aliases(world: Callable[..., World],
                                                                     tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG + '\n[[host]]\nname = "box"\nssh = "me@box"\n')
    hosts = (HostConfig(name="me_box", ssh="me@box"), HostConfig(name="vega", ssh="vega"))
    known = world(*hosts, config_path=path)
    known.svc.remember_hosts(["me_box", "vega"])
    assert list(known.svc._remember) == ["vega"]                      # me@box is configured (as "box")
    off = world(*hosts, config_path=path, general=GeneralSettings(remember_hosts=False))
    before = path.read_text()
    off.svc.remember_hosts(["me_box", "vega"])
    assert not off.svc._remember
    await off.svc.start()
    await run_for(off.clock, 2)
    assert path.read_text() == before and not [n for n in off.svc.notices if n.startswith("saved")]


# ---------------------------------------------------------------- unwritable config, restore within the session
@pytest.mark.posix                              # directory modes
@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
async def test_an_unwritable_config_raises_config_error_and_changes_nothing(world: Callable[..., World],
                                                                            tmp_path: Path) -> None:
    ro = tmp_path / "dotfiles"
    ro.mkdir()
    path = ro / "config.toml"
    path.write_text(CONFIG)
    ro.chmod(0o555)                        # e.g. a config managed by a dotfiles / Nix store directory
    try:
        w = world(K2, config_path=path)
        await w.svc.start()
        with pytest.raises(ConfigError, match="cannot write: Permission denied"):
            w.svc.add_host(HostConfig(name="vega", ssh="vega"))
        assert "vega" not in w.svc.store.hosts
        with pytest.raises(ConfigError, match="cannot write"):
            await w.svc.remove_host("k2")
        assert "k2" in w.svc.store.hosts
        with pytest.raises(ConfigError, match="cannot write"):
            w.svc.set_general(mouse=False)
        assert w.svc.config.general.mouse is True and path.read_text() == CONFIG
    finally:
        ro.chmod(0o755)


@pytest.mark.posix                              # directory modes
@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
async def test_remember_reports_an_unwritable_config(world: Callable[..., World], tmp_path: Path) -> None:
    ro = tmp_path / "dotfiles"
    ro.mkdir()
    path = ro / "config.toml"
    path.write_text(CONFIG)
    ro.chmod(0o555)
    try:
        vega = HostConfig(name="vega", ssh="vega")
        w = world(K2, vega, config_path=path)
        w.svc.remember_hosts(["vega"])
        await w.svc.start()
        await run_for(w.clock, 3)          # vega answers: the save fails -> a notice, not an exception in a callback
        assert w.state("vega") is ConnState.OK and path.read_text() == CONFIG
        assert [n for n in w.svc.notices if "could not save vega" in n and "cannot write" in n]
        assert not [n for n in w.svc.notices if n.startswith("saved ")]
    finally:
        ro.chmod(0o755)


async def test_a_host_removed_and_added_again_keeps_its_settings(world: Callable[..., World], tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG.replace('name = "k2"\n', 'name = "k2"\nprotected_qos = ["labpi"]   # never touch\n'))
    before = path.read_text()
    w = world(K2, config_path=path)
    await w.svc.start()
    assert w.svc.custom_host_keys("k2") == ["protected_qos"] and w.svc.custom_host_keys("vega") == []
    await w.svc.remove_host("k2")
    assert "k2" not in path.read_text() and "k2" not in w.svc.store.hosts
    running = w.svc.add_host(HostConfig(name="k2", ssh="k2"))
    assert path.read_text() == before and running.slurm.protected_qos == ("labpi",)
    memory = world(K2)                          # --demo: nothing on disk, no keys
    assert memory.svc.custom_host_keys("k2") == []
