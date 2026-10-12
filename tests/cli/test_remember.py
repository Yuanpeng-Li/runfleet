"""`runfleet HOST …` remembers new hosts (§8.4) — end to end through cli.main with a real FleetService over
FakeTransports (no ssh, no network), and the Settings wiring of the CLI (config_path, demo in memory)."""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli
from runfleet.config import FleetConfig, HostConfig
from runfleet.core.service import FleetService
from runfleet.models import ErrorKind
from runfleet.transport.fake import FakeTransport
from tests.unit.stubs import STUB_HOOKS


class Run:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.config = tmp_path / "config" / "config.toml"
        self.services: list[FleetService] = []
        self.failing: set[str] = set()
        self.settings: list[bool] = []
        monkeypatch.setattr(cli.paths, "config_file", lambda: self.config)
        monkeypatch.setattr(cli.paths, "log_file", lambda: tmp_path / "state" / "runfleet.log")
        monkeypatch.setattr(cli.paths, "cache_file", lambda: tmp_path / "state" / "cache.sqlite")
        monkeypatch.setattr(cli, "has_terminal", lambda: True)
        monkeypatch.setattr(cli, "is_interactive", lambda: True)
        monkeypatch.setattr(cli, "ssh_available", lambda: True)
        monkeypatch.setattr("runfleet.transport.make_transport", self.transport)
        monkeypatch.setattr(cli, "build_service", self.build)
        monkeypatch.setattr(cli, "run_tui", self.run_tui)

    def transport(self, host: HostConfig) -> FakeTransport:
        t = FakeTransport(host.name)
        t.set_result("detect.id", "user=me")
        t.set_result("slurm.squeue", "101 R")
        if host.name in self.failing:
            t.fail_next(ErrorKind.DNS, times=100)
        return t

    def build(self, config: FleetConfig, transports: Any, clock: Any, cache: Any) -> FleetService:
        svc = FleetService(config, transports, clock, None)
        svc.hooks = STUB_HOOKS
        svc.lock_path = self.tmp / "instance.lock"
        self.services.append(svc)
        return svc

    def run_tui(self, service: FleetService, clock: Any, *, demo: bool = False, settings: bool = False) -> int:
        self.settings.append(settings)

        async def session() -> None:
            await service.start()
            for _ in range(200):                        # until the first polls answered (real clock, fake ssh)
                await asyncio.sleep(0.005)
                if all(hs.status.last_attempt is not None and not hs.status.in_flight
                       for hs in service.store.hosts.values()):
                    break
            await asyncio.sleep(0.02)
            await service.stop()

        asyncio.run(session())
        return 0


@pytest.fixture
def r(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Run]:
    yield Run(monkeypatch, tmp_path)
    import logging
    root = logging.getLogger("runfleet")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _saved(path: Path) -> list[dict[str, Any]]:
    return list(tomllib.loads(path.read_text(encoding="utf-8")).get("host", []))


def test_hosts_named_without_a_config_are_saved_once_they_answer(r: Run) -> None:
    r.failing.add("vpn-off")
    assert cli.main(["k2", "me@box", "vpn-off"]) == 0
    svc = r.services[0]
    assert svc.config_path == r.config
    assert _saved(r.config) == [{"name": "k2"}, {"name": "me_box", "ssh": "me@box"}]   # command-line order
    assert "saved k2, me_box — manage in Settings (,)" in svc.notices
    assert cli.main([]) == 0                                   # next time plain `runfleet` has them
    assert [h.name for h in r.services[1].config.hosts] == ["k2", "me_box"]


def test_an_existing_config_gains_the_new_host_but_the_run_shows_only_it(r: Run) -> None:
    r.config.parent.mkdir(parents=True)
    original = '# mine\n[[host]]\nname = "vega"   # slow login\nconnect_timeout = "120s"\n'
    r.config.write_text(original)
    assert cli.main(["k2"]) == 0
    assert list(r.services[0].store.hosts) == ["k2"]          # only shows k2 this run
    assert r.config.read_text() == original + '\n[[host]]\nname = "k2"\n'
    assert cli.main(["vega"]) == 0                             # already configured: nothing written
    assert r.config.read_text() == original + '\n[[host]]\nname = "k2"\n'


def test_remember_hosts_off_writes_nothing(r: Run) -> None:
    r.config.parent.mkdir(parents=True)
    r.config.write_text("[general]\nremember_hosts = false\n")
    assert cli.main(["k2"]) == 0
    assert r.config.read_text() == "[general]\nremember_hosts = false\n"
    assert r.settings == [False]


def test_demo_settings_live_in_memory(r: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.demo.transport import DemoTransport

    seen: list[FleetService] = []
    monkeypatch.setattr(cli, "run_tui", lambda svc, clock, **kw: seen.append(svc) or 0)
    assert cli.main(["--demo", "k2"]) == 0
    svc = seen[0]
    assert svc.config_path is None and svc.ssh_aliases() == ["k2", "vega", "lyra", "labstation", "lynx"]
    svc.add_host(HostConfig(name="vega", ssh="vega"))
    assert [h.name for h in svc.saved_hosts()] == ["k2", "vega"] and not r.config.exists()
    assert isinstance(svc._transport("vega"), DemoTransport)
