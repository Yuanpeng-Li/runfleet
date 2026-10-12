"""`runfleet wait` command line (§8.5): parsing, config handed to the service, exit codes. No network: the
service is the CLI FakeService and `wait.wait_for` is replaced (tests/unit/test_wait.py drives the real one)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli, wait
from tests.cli.test_cli import Harness, _drop_log_handlers, run


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Harness]:
    _drop_log_handlers()
    yield Harness(monkeypatch, tmp_path)
    _drop_log_handlers()


@pytest.fixture
def waited(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_wait_for(service: Any, targets: Sequence[wait.Target], *, clock: Any, timeout: float | None,
                            emit: Any, as_json: bool = False) -> int:
        calls.append({"service": service, "targets": list(targets), "timeout": timeout, "json": as_json})
        emit("all 1 job completed")
        return wait.EXIT_FAILED

    monkeypatch.setattr(wait, "wait_for", fake_wait_for)
    return calls


def test_wait_parses_targets_between_options(h: Harness, waited: list[dict[str, Any]],
                                             capsys: pytest.CaptureFixture[str]) -> None:
    ns = cli.parse_args(["wait", "k2:1", "--json", "vega:31270818", "--timeout", "90"])
    assert ns.command == "wait" and ns.hosts == ["k2:1", "vega:31270818"] and ns.json and ns.timeout == 90
    rc, out, _ = run(["wait", "--demo", "k2:2332861", "vega:723954", "--timeout", "30"], capsys)
    assert rc == wait.EXIT_FAILED and "all 1 job completed" in out          # the waiter's exit code is returned
    call = waited[0]
    assert [(t.host, t.job) for t in call["targets"]] == [("k2", "2332861"), ("vega", "723954")]
    assert call["timeout"] == 30.0 and not call["json"]
    config = call["service"].config
    assert [hc.name for hc in config.hosts] == ["k2", "vega"]              # just the named hosts
    assert config.general.readonly and config.general.idle_pause_after == wait.NEVER
    assert call["service"].lock_path is False                              # --demo: never the real instance locks


def test_wait_bad_targets_exit_3(h: Harness, waited: list[dict[str, Any]],
                                 capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = run(["wait", "--demo", "k2"], capsys)
    assert rc == wait.EXIT_ERROR and "HOST:JOBID" in err and not waited
    with pytest.raises(SystemExit) as exc:                                 # no target at all: usage error
        cli.parse_args(["wait"])
    assert exc.value.code == 2


def test_wait_setup_errors_exit_3_never_2_or_1(h: Harness, waited: list[dict[str, Any]],
                                                capsys: pytest.CaptureFixture[str],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    missing = str(h.config_path.parent / "nope.toml")
    rc, _, err = run(["wait", "-c", missing, "k2:1"], capsys)            # not 2: that is a timeout for `wait`
    assert rc == wait.EXIT_ERROR and "no such config file" in err and not waited
    h.config_path.write_text("[[host]]\nname = 'k2'\npoll_jobs = 'soon'\n")
    rc, out, _ = run(["wait", "k2:1", "--json"], capsys)
    doc = json.loads(out.strip().splitlines()[-1])
    assert rc == wait.EXIT_ERROR and doc["event"] == "error" and doc["exit"] == wait.EXIT_ERROR

    async def broken(*_a: Any, **_k: Any) -> int:
        raise cli.RunFleetError("another RunFleet (pid 7) is polling")

    monkeypatch.setattr(wait, "wait_for", broken)
    rc, _, err = run(["wait", "--demo", "k2:2332861"], capsys)           # not 1: that is a failed job
    assert rc == wait.EXIT_ERROR and "another RunFleet" in err


def test_wait_duplicate_targets_wait_once() -> None:
    from runfleet.config import HostConfig

    k2 = HostConfig(name="k2", ssh="k2-alias", backend="slurm")
    assert len(wait.resolve_targets(["k2:101", "k2:101", "k2-alias:101"], (k2,))) == 1
