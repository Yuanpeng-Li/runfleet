"""configedit.py: comment-preserving config.toml edits (golden files), validation, invalid TOML, atomic writes."""

from __future__ import annotations

import os
import sys
import stat
import tomllib
from pathlib import Path

import pytest
import tomlkit

from runfleet import configedit as ce
from runfleet.config import HostConfig, dump_hosts_toml, parse_config
from runfleet.errors import ConfigError
from tests.conftest import is_root


def _left(d: Path) -> list[str]:
    """Files in `d`; Windows refuses to unlink the open `.config.toml.lock`, so it stays there (harmless)."""
    return sorted(p.name for p in d.iterdir() if not (sys.platform == "win32" and p.name == ".config.toml.lock"))

GOLDEN = Path(__file__).parent / "golden" / "configedit"
INPUT = (GOLDEN / "input.toml").read_text(encoding="utf-8")


def _cfg(path: Path, text: str = INPUT) -> ce.ConfigFile:
    path.write_text(text, encoding="utf-8")
    return ce.ConfigFile(path)


def _names(path: Path) -> list[str]:
    return [h["name"] for h in tomllib.loads(path.read_text(encoding="utf-8")).get("host", [])]


# ---------------------------------------------------------------- golden files (comments and formatting survive)
@pytest.mark.parametrize(("golden", "op"), [
    ("remove_middle", lambda f: f.remove_host("vega")),
    ("remove_first", lambda f: f.remove_host("k2")),
    ("remove_last", lambda f: f.remove_host("lyra")),
    ("reorder", lambda f: f.reorder_hosts(["lyra", "k2"])),
    ("add", lambda f: f.add_hosts([HostConfig("lynx", "lynx"), HostConfig("me_box", "me@box")])),
    ("general", lambda f: f.set_general({"readonly": True, "idle_pause_after": 28800.0,
                                         "refresh_multiplier": 0.5})),
    ("notify", lambda f: f.set_table("notify", {"enabled": False, "terminal": "bell",
                                                "events": ("ended_bad", "restarted", "time_low", "gpu_idle")})),
])
def test_edits_match_golden_files(tmp_path: Path, golden: str, op) -> None:  # type: ignore[no-untyped-def]
    f = _cfg(tmp_path / "config.toml")
    assert op(f)
    assert f.path.read_text(encoding="utf-8") == (GOLDEN / f"{golden}.toml").read_text(encoding="utf-8")


def test_removing_every_host_keeps_the_rest_and_the_tail(tmp_path: Path) -> None:
    f = _cfg(tmp_path / "config.toml")
    for name in ("k2", "vega", "lyra"):
        assert f.remove_host(name)
    assert f.path.read_text(encoding="utf-8") == (GOLDEN / "remove_all.toml").read_text(encoding="utf-8")
    assert f.hosts() == []
    assert f.add_hosts([HostConfig("k2", "k2")]) == ["k2"]
    assert _names(f.path) == ["k2"]
    assert "# [[host]]\n# name = \"atlas\"" in f.path.read_text(encoding="utf-8")


def test_a_comment_block_moves_with_its_host_and_file_header_stays(tmp_path: Path) -> None:
    text = "# my fleet\n# notes\n\n# first host\n[[host]]\nname = \"a\"\n\n[[host]]\nname = \"b\"\n"
    f = _cfg(tmp_path / "c.toml", text)
    f.reorder_hosts(["b", "a"])
    assert f.path.read_text() == ("# my fleet\n# notes\n\n[[host]]\nname = \"b\"\n\n# first host\n[[host]]\n"
                                  "name = \"a\"\n")
    f.remove_host("a")
    assert f.path.read_text() == "# my fleet\n# notes\n\n[[host]]\nname = \"b\"\n"


def test_general_table_is_inserted_above_the_first_table_and_its_comment(tmp_path: Path) -> None:
    text = "# my fleet\n\n# first host\n[[host]]\nname = \"a\"\n"
    f = _cfg(tmp_path / "c.toml", text)
    assert f.set_general({"default_view": "gpus", "mouse": False})
    assert f.path.read_text() == ("# my fleet\n\n[general]\ndefault_view = \"gpus\"\nmouse = false\n\n"
                                  "# first host\n[[host]]\nname = \"a\"\n")
    assert not f.set_general({"mouse": False})                      # unchanged value: no write


@pytest.mark.parametrize(("text", "expected"), [
    # no table before the hosts: the file header stays on top, the first host keeps its comment
    ("# my fleet\n\n# first host\n[[host]]\nname = \"a\"\n",
     "# my fleet\n\n[notify]\nenabled = false\n\n# first host\n[[host]]\nname = \"a\"\n"),
    ("[[host]]\nname = \"a\"\n", "[notify]\nenabled = false\n\n[[host]]\nname = \"a\"\n"),
    # a table glued to the first host: a blank line on each side
    ("[general]\nmouse = false\n[[host]]\nname = \"a\"\n",
     "[general]\nmouse = false\n\n[notify]\nenabled = false\n\n[[host]]\nname = \"a\"\n"),
    # no hosts: at the end
    ("[general]\nmouse = false\n", "[general]\nmouse = false\n\n[notify]\nenabled = false\n"),
])
def test_a_missing_table_goes_above_the_hosts(tmp_path: Path, text: str, expected: str) -> None:
    f = _cfg(tmp_path / "c.toml", text)
    assert f.set_table("notify", {"enabled": False})
    assert f.path.read_text() == expected
    assert f.add_hosts([HostConfig("x", "x")]) == ["x"]               # the result is still editable


def test_table_keys_are_updated_in_place_and_round_trip(tmp_path: Path) -> None:
    from runfleet.config import load_config

    text = ('[general]\nmouse = false\n\n[notify]\nenabled = true   # pop-ups on\ncommand = "notify-send x"\n\n'
            '# first host\n[[host]]\nname = "a"\nssh = "me@a"   # mine\n')
    f = _cfg(tmp_path / "config.toml", text)
    hosts = f.hosts()
    assert f.set_table("notify", {"enabled": False, "terminal": "osc9", "events": ["time_low"]})
    assert f.path.read_text() == ('[general]\nmouse = false\n\n[notify]\nenabled = false   # pop-ups on\n'
                                  'command = "notify-send x"\nterminal = "osc9"\nevents = ["time_low"]\n\n'
                                  '# first host\n[[host]]\nname = "a"\nssh = "me@a"   # mine\n')
    mtime = f.path.stat().st_mtime_ns
    assert not f.set_table("notify", {"enabled": False, "events": ("time_low",)})    # unchanged: no write
    assert f.path.stat().st_mtime_ns == mtime and f.hosts() == hosts
    cfg = load_config(f.path)
    assert (cfg.notify.enabled, cfg.notify.terminal, cfg.notify.events) == (False, "osc9", ("time_low",))
    assert cfg.notify.command == "notify-send x" and not cfg.general.mouse and cfg.hosts == tuple(hosts)


def test_table_values_are_validated(tmp_path: Path) -> None:
    f = _cfg(tmp_path / "config.toml")
    before = f.path.read_text()
    with pytest.raises(ConfigError, match="notify.events"):
        f.set_table("notify", {"events": ["ended_bad", "exploded"]})
    with pytest.raises(ConfigError, match="notify.colour: unknown setting"):
        f.set_table("notify", {"colour": "blue"})
    with pytest.raises(ConfigError, match="unknown setting"):
        f.set_table("host", {"name": "x"})
    inline = _cfg(tmp_path / "inline.toml", "notify = 3\n")
    with pytest.raises(ConfigError, match="notify: expected a table"):
        inline.set_table("notify", {"enabled": False})
    assert f.path.read_text() == before and inline.path.read_text() == "notify = 3\n"


def test_add_is_idempotent_by_name_and_alias(tmp_path: Path) -> None:
    f = _cfg(tmp_path / "config.toml")
    assert f.add_hosts([HostConfig("k2", "k2"), HostConfig("vega2", "vega")]) == []
    assert f.add_hosts([HostConfig("x", "x"), HostConfig("x", "x")]) == ["x"]
    assert _names(f.path) == ["k2", "vega", "lyra", "x"]


def test_minimal_tables_only(tmp_path: Path) -> None:
    """A host from Settings never freezes defaults into its table (flat keys stay with [defaults.*])."""
    from runfleet.config import SlurmSettings

    f = _cfg(tmp_path / "config.toml")
    f.add_hosts([HostConfig("z", "me@z", connect_timeout=120, slurm=SlurmSettings(poll_jobs=30))])
    text = f.path.read_text()
    assert text.endswith('[[host]]\nname = "z"\nssh = "me@z"\n\n# [[host]]\n# name = "atlas"\n# ssh = "local"\n')
    assert next(h for h in f.hosts() if h.name == "z").slurm.poll_jobs == 20      # [defaults.slurm] applies


@pytest.mark.posix                # umask and file modes
def test_new_file_starts_from_the_commented_template(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "config.toml"
    f = ce.ConfigFile(path)
    assert f.hosts() == []
    assert f.add_hosts([HostConfig("k2", "k2"), HostConfig("vega", "vega")]) == ["k2", "vega"]
    text = path.read_text()
    assert text.startswith(dump_hosts_toml(()).rstrip("\n"))
    assert text.endswith('\n[[host]]\nname = "k2"\n\n[[host]]\nname = "vega"\n')
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


def test_general_values_are_validated_and_durations_written_as_strings(tmp_path: Path) -> None:
    f = _cfg(tmp_path / "config.toml")
    before = f.path.read_text()
    with pytest.raises(ConfigError, match="general.readonly"):
        f.set_general({"readonly": "yes"})
    with pytest.raises(ConfigError, match="unknown setting"):
        f.set_general({"colour": "blue"})
    assert f.path.read_text() == before
    f.set_general({"idle_after": 300.0})
    assert 'idle_after = "5m"' in f.path.read_text()


# ---------------------------------------------------------------- the file is never damaged
def test_invalid_toml_is_never_overwritten(tmp_path: Path) -> None:
    broken = '[general]\nreadonly = tru\n\n[[host]]\nname = "k2"\n'
    f = _cfg(tmp_path / "config.toml", broken)
    mtime = f.path.stat().st_mtime_ns
    for op in (lambda: f.add_hosts([HostConfig("x", "x")]), lambda: f.remove_host("k2"),
               lambda: f.reorder_hosts(["k2"]), lambda: f.set_general({"mouse": False}),
               lambda: f.set_table("notify", {"enabled": False}), f.hosts):
        with pytest.raises(ConfigError, match="invalid TOML"):
            op()
    assert f.path.read_text() == broken and f.path.stat().st_mtime_ns == mtime
    assert _left(tmp_path) == ["config.toml"]


def test_an_edit_that_would_not_validate_is_not_written(tmp_path: Path) -> None:
    text = '[[host]]\nname = "k2"\npoll_jobs = 15\n'             # bare number: invalid for RunFleet, valid TOML
    f = _cfg(tmp_path / "config.toml", text)
    with pytest.raises(ConfigError, match="poll_jobs"):
        f.add_hosts([HostConfig("x", "x")])
    assert f.path.read_text() == text


def test_split_or_inline_host_lists_are_refused(tmp_path: Path) -> None:
    split = '[[host]]\nname = "a"\n\n[ui]\njobs_columns = ["id"]\n\n[[host]]\nname = "b"\n'
    with pytest.raises(ConfigError):
        _cfg(tmp_path / "a.toml", split).remove_host("a")
    inline = 'host = [{name = "a"}]\n'
    with pytest.raises(ConfigError, match="by hand"):
        _cfg(tmp_path / "b.toml", inline).add_hosts([HostConfig("x", "x")])


@pytest.mark.posix                # file modes
def test_atomic_write_keeps_mode_and_follows_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "runfleet.toml"
    real.parent.mkdir()
    real.write_text(INPUT)
    real.chmod(0o640)
    link = tmp_path / "config.toml"
    link.symlink_to(real)
    ce.ConfigFile(link).remove_host("vega")
    assert link.is_symlink() and "vega" not in real.read_text()
    assert stat.S_IMODE(real.stat().st_mode) == 0o640
    assert sorted(p.name for p in real.parent.iterdir()) == ["runfleet.toml"]      # no temp file left


def test_a_failed_rename_leaves_the_original_and_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _cfg(tmp_path / "config.toml")

    def boom(src: str, dst: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ce.os, "replace", boom)
    with pytest.raises(ConfigError, match="cannot write: No space left on device"):     # an error, not a crash
        f.remove_host("k2")
    assert f.path.read_text() == INPUT
    assert _left(tmp_path) == ["config.toml"]


def test_every_golden_file_still_loads() -> None:
    for path in GOLDEN.glob("*.toml"):
        parse_config(tomllib.loads(path.read_text()), source=path)
        assert tomlkit.dumps(tomlkit.parse(path.read_text())) == path.read_text()


def test_crlf_files_stay_crlf_and_a_missing_final_newline_is_fine(tmp_path: Path) -> None:
    crlf = _cfg(tmp_path / "crlf.toml", '# fleet\r\n[[host]]\r\nname = "k2"\r\n')
    crlf.add_hosts([HostConfig("x", "x")])
    assert crlf.path.read_bytes() == b'# fleet\r\n[[host]]\r\nname = "k2"\r\n\r\n[[host]]\r\nname = "x"\r\n'
    bare = _cfg(tmp_path / "bare.toml", '[[host]]\nname = "k2"')
    bare.add_hosts([HostConfig("x", "x")])
    assert bare.path.read_text() == '[[host]]\nname = "k2"\n\n[[host]]\nname = "x"\n'


# ---------------------------------------------------------------- separators and headings stay in place
BARE = (GOLDEN / "bare_input.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(("golden", "op"), [
    ("bare_reorder", lambda f: f.reorder_hosts(["lyra", "k2"])),
    ("bare_remove_first", lambda f: f.remove_host("k2")),
])
def test_a_file_that_starts_with_a_host_keeps_its_layout(tmp_path: Path, golden: str, op) -> None:  # type: ignore[no-untyped-def]
    f = _cfg(tmp_path / "config.toml", BARE)
    assert op(f)
    assert f.path.read_text(encoding="utf-8") == (GOLDEN / f"{golden}.toml").read_text(encoding="utf-8")


def test_reorder_there_and_back_restores_the_file_byte_for_byte(tmp_path: Path) -> None:
    for text in (BARE, INPUT):
        f = _cfg(tmp_path / "config.toml", text)
        names = _names(f.path)
        f.reorder_hosts(list(reversed(names)))
        moved = f.path.read_text()
        assert moved.startswith(text[:9]) and moved.count("\n\n") == text.count("\n\n")    # no glued tables
        f.reorder_hosts(names)
        assert f.path.read_text() == text


def test_a_section_heading_above_the_first_host_stays(tmp_path: Path) -> None:
    text = ("[general]\nmouse = true\n\n# --- my fleet ---\n# a is my first box\n[[host]]\nname = \"a\"\n"
            "group = \"lab\"\n\n# b notes\n[[host]]\nname = \"b\"\n")
    f = _cfg(tmp_path / "c.toml", text)
    f.reorder_hosts(["b", "a"])
    assert f.path.read_text() == ("[general]\nmouse = true\n\n# --- my fleet ---\n# b notes\n[[host]]\nname = \"b\"\n"
                                  "\n# a is my first box\n[[host]]\nname = \"a\"\ngroup = \"lab\"\n")
    f.reorder_hosts(["a", "b"])
    assert f.path.read_text() == text
    f.remove_host("a")
    assert f.path.read_text() == "[general]\nmouse = true\n\n# --- my fleet ---\n# b notes\n[[host]]\nname = \"b\"\n"
    f.remove_host("b")
    assert f.path.read_text() == "[general]\nmouse = true\n\n# --- my fleet ---\n"


def test_a_removed_table_can_be_restored_whole(tmp_path: Path) -> None:
    f = _cfg(tmp_path / "config.toml")
    assert f.host_keys("vega") == ["connect_timeout"] and f.host_keys("nope") == []
    text = f.remove_host("vega")
    assert text == '# vega: slow cold logins (GSSAPI)\n# keep the long timeout\n[[host]]\nname = "vega"\n' \
                   'connect_timeout = "120s"\n'
    assert f.add_hosts([HostConfig("vega", "vega")], restore={"vega": text}) == ["vega"]
    assert 'name = "vega"\nconnect_timeout = "120s"\n' in f.path.read_text()
    assert "# keep the long timeout\n[[host]]\nname = \"vega\"" in f.path.read_text()
    f.remove_host("vega")                                        # another target under that name: minimal table
    f.add_hosts([HostConfig("vega", "me@vega")], restore={"vega": text})
    assert f.path.read_text().endswith('[[host]]\nname = "vega"\nssh = "me@vega"\n\n# [[host]]\n# name = "atlas"\n'
                                       '# ssh = "local"\n')


# ---------------------------------------------------------------- write failures and concurrent writers
@pytest.mark.posix                              # directory modes
@pytest.mark.skipif(is_root(), reason="root ignores directory permissions")
def test_an_unwritable_directory_is_a_config_error(tmp_path: Path) -> None:
    ro = tmp_path / "dotfiles"
    ro.mkdir()
    f = _cfg(ro / "config.toml")
    ro.chmod(0o555)
    try:
        for op in (lambda: f.add_hosts([HostConfig("x", "x")]), lambda: f.remove_host("k2"),
                   lambda: f.reorder_hosts(["lyra"]), lambda: f.set_general({"mouse": False}),
                   lambda: f.set_table("notify", {"enabled": False})):
            with pytest.raises(ConfigError, match="cannot write: Permission denied"):
                op()
        assert f.path.read_text() == INPUT and sorted(p.name for p in ro.iterdir()) == ["config.toml"]
    finally:
        ro.chmod(0o755)


@pytest.mark.posix                # holds the lock with fcntl.flock
def test_a_held_lock_waits_then_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    f = _cfg(tmp_path / "config.toml")
    monkeypatch.setattr(ce, "LOCK_TIMEOUT", 0.1)
    fd = os.open(tmp_path / ".config.toml.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)                  # another RunFleet in the middle of an edit
    try:
        with pytest.raises(ConfigError, match="another RunFleet is saving it"):
            f.remove_host("k2")
        assert f.path.read_text() == INPUT
    finally:
        os.close(fd)
    assert f.remove_host("k2")
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]       # the lock file lives only while held


def test_concurrent_processes_never_lose_an_edit(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = tmp_path / "config.toml"
    path.write_text(INPUT)
    code = ("import sys; from pathlib import Path; from runfleet.configedit import ConfigFile; "
            "from runfleet.config import HostConfig; f = ConfigFile(Path(sys.argv[1])); "
            "[f.add_hosts([HostConfig(f'{sys.argv[2]}{i}', f'{sys.argv[2]}{i}')]) for i in range(8)]")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path), f"h{n}x"]) for n in range(4)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0, 0, 0]
    names = _names(path)
    assert len(names) == 3 + 32 and {f"h{n}x{i}" for n in range(4) for i in range(8)} <= set(names)
    assert "# vega: slow cold logins (GSSAPI)" in path.read_text()
    assert _left(tmp_path) == ["config.toml"]


def test_a_lock_that_cannot_be_taken_in_a_writable_directory_is_an_error(tmp_path: Path) -> None:
    """Review BACK-06: the sidecar lock failed open, so two RunFleets could overwrite each other's edit."""
    f = _cfg(tmp_path / "config.toml")
    (tmp_path / ".config.toml.lock").mkdir()            # os.open fails (EISDIR) although the directory is writable
    with pytest.raises(ConfigError, match="cannot lock"):
        f.set_general({"mouse": False})
    assert f.path.read_text() == INPUT


def test_remove_and_restore_keeps_the_hosts_trailing_comments(tmp_path: Path) -> None:
    """Review BACK-46: a note under a host's keys was lost after remove + restore (undo / re-add in Settings)."""
    text = ('[[host]]\nname = "a"\n\n[[host]]\nname = "k2"\nconnect_timeout = "60s"\n# note about k2 settings\n\n'
            '[[host]]\nname = "z"\n')
    f = _cfg(tmp_path / "config.toml", text)
    removed = f.remove_host("k2")
    assert removed is not None and "# note about k2 settings" in removed
    assert "note about k2" not in f.path.read_text()
    f.add_hosts([HostConfig("k2", "k2")], restore={"k2": removed})
    assert f.path.read_text().endswith('[[host]]\nname = "k2"\nconnect_timeout = "60s"\n# note about k2 settings\n')
    # the last host's trailing comments are the file tail (a commented-out example): they never move with it
    removed = f.remove_host("k2")
    assert removed is not None and "# note about k2 settings" not in removed
    assert f.path.read_text().endswith('name = "z"\n# note about k2 settings\n')


def test_set_table_keys_on_a_table_split_over_the_file() -> None:
    """Dotted keys (`notify.a = …` twice) or `[notify]` plus a later `[notify.x]` load as tomlkit's proxy: keys are
    set where they already are, new ones join the table."""
    doc = ce.parse_document('notify.enabled = true   # on\nnotify.terminal = "auto"\n\n[[host]]\nname = "k2"\n')
    assert ce.set_table_keys(doc, "notify", {"terminal": "bell", "enabled": True})
    assert tomlkit.dumps(doc).startswith('notify.enabled = true   # on\nnotify.terminal = "bell"\n')
    assert ce.set_table_keys(doc, "notify", {"events": ["ended_bad"]})
    assert tomllib.loads(tomlkit.dumps(doc))["notify"] == {"enabled": True, "terminal": "bell",
                                                           "events": ["ended_bad"]}
    doc = ce.parse_document('[notify]\nterminal = "auto"  # mine\n\n[[host]]\nname = "k2"\n\n[notify.x]\n')
    assert ce.set_table_keys(doc, "notify", {"terminal": "osc9"})
    assert '[notify]\nterminal = "osc9"  # mine\n' in tomlkit.dumps(doc)
    doc = ce.parse_document('[notify]\nterminal = "auto"\n\n# first host\n[[host]]\nname = "k2"\n\n[notify.x]\n')
    assert ce.set_table_keys(doc, "notify", {"enabled": False})            # a new key stays above the host comment
    assert tomlkit.dumps(doc).startswith('[notify]\nterminal = "auto"\nenabled = false\n\n# first host\n[[host]]\n')
