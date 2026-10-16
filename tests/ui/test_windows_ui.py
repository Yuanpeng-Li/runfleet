"""The UI's Windows branches (preview, DESIGN §1.2, §4.1, §7.8, §8.6) with ``platform.IS_WINDOWS`` patched: no ^z,
the notify hook through cmd.exe, notepad and more as defaults, host detail / error wording, update by hand."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from runfleet import update
from runfleet.core.clock import ManualClock
from runfleet.core.logs import LogService
from runfleet.models import ErrorKind, HostError
from runfleet.transport.fake import FakeTransport
from runfleet.transport.mux import WINDOWS_NO_MUX
from runfleet.ui import keymap, notify
from runfleet.ui.screens.host_detail import connection_lines
from runfleet.ui.screens.host_error import WINDOWS_AUTH, error_report
from runfleet.ui.screens.settings import editor_argv, host_name_for
from runfleet.util import platform
from tests.conftest import CLOCK_START
from tests.ui.fakes import T0, build_world


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", True)


# ---------------------------------------------------------------- keys (§7.8)
def test_no_ctrl_z_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    def keys() -> set[str]:
        return {b.key for b in keymap.bindings("global")}           # type: ignore[union-attr]

    def help_rows() -> list[str]:
        return [label for rows in keymap.help_entries(["global"]).values() for label, _ in rows]

    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    doc = keymap.markdown()
    assert "ctrl+z" in keys() and any("suspend" in t for t in [d for rows in keymap.help_entries(["global"]).values()
                                                              for _, d in rows])
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    assert "ctrl+z" not in keys() and "ctrl+c" in keys()
    assert not any("^z" in label for label in help_rows())
    assert keymap.markdown() == doc                                 # USER_GUIDE stays the same everywhere


# ---------------------------------------------------------------- notify hook (§7.14)
def test_hook_argv_on_windows(windows: None) -> None:
    assert notify.hook_argv('msg * "%RUNFLEET_SUMMARY%"') == ["cmd.exe", "/c", 'msg * "%RUNFLEET_SUMMARY%"']
    assert notify.hook_argv(["toast.exe", "x"]) == ["toast.exe", "x"]
    assert notify.hook_argv("  ") is None


class FakeProc:
    def __init__(self, rc: int = 0, hang: bool = False) -> None:
        self.returncode: int | None = None if hang else rc
        self.pid = 4321
        self._rc, self._hang = rc, hang

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(60)
        return b"", b""

    def send_signal(self, sig: int) -> None:
        pass

    def kill(self) -> None:
        self.returncode = 1


async def test_run_hook_on_windows_goes_through_the_shell(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    shell: list[tuple[str, dict[str, Any]]] = []
    execs: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def create_shell(cmd: str, **kw: Any) -> FakeProc:
        shell.append((cmd, kw))
        return FakeProc()

    async def create_exec(*argv: str, **kw: Any) -> FakeProc:
        execs.append((argv, kw))
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create_shell)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_exec)
    await notify.run_hook(["cmd.exe", "/c", 'msg * "%RUNFLEET_SUMMARY%"'], {"RUNFLEET_SUMMARY": "k2 failed"}, 5)
    await notify.run_hook(["toast.exe", "--title", "RunFleet"], {}, 5)
    assert [c for c, _ in shell] == ['msg * "%RUNFLEET_SUMMARY%"']             # cmd.exe gets the string unquoted
    assert [a for a, _ in execs] == [("toast.exe", "--title", "RunFleet")]
    for kw in [shell[0][1], execs[0][1]]:
        assert kw["creationflags"] == platform.CREATE_NO_WINDOW | platform.CREATE_NEW_PROCESS_GROUP
        assert "start_new_session" not in kw
    assert shell[0][1]["env"]["RUNFLEET_SUMMARY"] == "k2 failed"


async def test_slow_hook_on_windows_is_killed_with_its_tree(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(platform, "taskkill_tree", lambda pid: killed.append(pid) or 0)

    async def create_exec(*argv: str, **kw: Any) -> FakeProc:
        return FakeProc(hang=True)

    async def terminate(proc: Any, grace: float = 2.0) -> None:
        platform.kill_tree(proc, hard=True)

    from runfleet.transport import spawn

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_exec)
    monkeypatch.setattr(spawn, "terminate", terminate)
    await notify.run_hook(["toast.exe"], {}, 0.05)
    assert killed and set(killed) == {4321}                   # taskkill /T on timeout (and the final group kill)


# ---------------------------------------------------------------- editor, pager, names
def test_editor_default_is_notepad_on_windows(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert editor_argv(Path("c.toml")) == ["notepad", "c.toml"]
    monkeypatch.setenv("EDITOR", r'"C:\Program Files\Notepad++\notepad++.exe" -multiInst')
    assert editor_argv(Path("c.toml")) == [r"C:\Program Files\Notepad++\notepad++.exe", "-multiInst", "c.toml"]
    monkeypatch.setenv("EDITOR", '"broken')
    assert editor_argv(Path("c.toml")) == ["notepad", "c.toml"]


def test_editor_default_is_vi_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert editor_argv(Path("c.toml")) == ["vi", "c.toml"]


def test_pager_default_on_windows(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    t = FakeTransport("k2")
    svc = LogService(ManualClock(CLOCK_START), transport_for=lambda h: t, stream_limit_for=lambda h: 1,
                     probe=lambda h, s: None, caps_for=lambda h: None)   # type: ignore[arg-type]
    monkeypatch.delenv("PAGER", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    assert svc.local_pager_argv(Path("x.log")) == ["more", "x.log"]            # always there on Windows
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: r"C:\Program Files\Git\usr\bin\less.exe")
    assert svc.local_pager_argv(Path("x.log")) == ["less", "-R", "+G", "x.log"]
    monkeypatch.setenv("PAGER", r'"C:\Program Files\Git\usr\bin\less.exe" -R')
    assert svc.local_pager_argv(Path("x.log")) == [r"C:\Program Files\Git\usr\bin\less.exe", "-R", "x.log"]


def test_settings_host_names_avoid_windows_device_names(windows: None) -> None:
    assert host_name_for("con") == "con_" and host_name_for("me@box") == "me_box" and host_name_for("") == "host"


# ---------------------------------------------------------------- host detail and error dialog
def _states() -> dict[str, Any]:
    _config, states = build_world()
    return {s.config.name: s for s in states}


def test_host_detail_says_why_sharing_is_off(windows: None) -> None:
    st = _states()["labstation"]
    st.status = dataclasses.replace(st.status, mux="off")
    text = "\n".join(t.plain for t in connection_lines(st, T0, 400))
    assert "mux: off (Windows)" in text and WINDOWS_NO_MUX in text


def test_host_detail_unchanged_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    st = _states()["labstation"]
    st.status = dataclasses.replace(st.status, mux="off")
    text = "\n".join(t.plain for t in connection_lines(st, T0, 400))
    assert "mux: off" in text and "(Windows)" not in text and WINDOWS_NO_MUX not in text


def test_auth_error_explains_the_windows_limit(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    lynx = _states()["lynx"]
    lynx.status = dataclasses.replace(lynx.status, error=HostError(ErrorKind.AUTH, "Permission denied"))
    text = error_report(lynx, T0, 400).plain
    assert WINDOWS_AUTH in text and "WSL" in text
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    assert WINDOWS_AUTH not in error_report(lynx, T0, 400).plain


# ---------------------------------------------------------------- update (§8.6)
async def test_update_on_windows_is_by_hand(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from runfleet.ui.app import RunFleetApp
    from tests.ui.fakes import FakeService

    install = update.Install("uv", "x", (("uv.exe", "tool", "upgrade", "runfleet"),), "uv tool upgrade runfleet")
    monkeypatch.setattr(update, "detect_install", lambda: install)
    app = RunFleetApp(FakeService(), clock=ManualClock(T0), persist_state=False)
    toasts: list[str] = []
    monkeypatch.setattr(app, "toast", lambda text, **kw: toasts.append(text))
    monkeypatch.setattr(app, "exit", lambda **kw: pytest.fail("never quits to update on Windows"))
    app.request_update()
    assert toasts == ["quit RunFleet, then run uv tool upgrade runfleet"]


# ---------------------------------------------------------------- external programs (pager, editor, ssh -t)
@pytest.mark.parametrize("sequence", [False, True])
def test_external_programs_are_found_with_pathext(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                  sequence: bool) -> None:
    """CreateProcess appends only `.exe`: the default pager `more` is more.com and VS Code's `code` is code.cmd, so
    the suspended run looks the program up on PATH with PATHEXT first."""
    import contextlib
    import shutil
    import subprocess

    from runfleet.ui.app import RunFleetApp
    from tests.ui.fakes import FakeService

    found = {"more": r"C:\Windows\System32\more.com",
             "code": r"C:\Users\u\AppData\Local\Programs\VS Code\bin\code.cmd"}
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: found.get(name))
    ran: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[bytes]:
        ran.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run)
    app = RunFleetApp(FakeService(), clock=ManualClock(T0), persist_state=False)
    monkeypatch.setattr(app, "suspend", contextlib.nullcontext)
    monkeypatch.setattr(app, "refresh", lambda **kw: app)
    if sequence:
        steps = [(["more", "x.log"], "banner", None), (["code", "c.toml"], "banner", None)]
        assert app.run_external_sequence(steps) == [0, 0]
    else:
        assert app.run_external(["more", "x.log"]) == 0
        assert app.run_external(["code", "c.toml"]) == 0
    assert ran == [[found["more"], "x.log"], [found["code"], "c.toml"]]
