"""[WP1] paths, util/shell validators, sshconfig alias listing and `ssh -G` parsing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runfleet import paths, sshconfig
from runfleet.errors import RunFleetError
from runfleet.util import shell


# ---------------------------------------------------------------- paths
@pytest.mark.posix                # XDG directories
def test_paths_are_under_platformdirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert paths.config_file() == tmp_path / "cfg" / "runfleet" / "config.toml"
    assert paths.cache_file() == tmp_path / "state" / "runfleet" / "cache.sqlite"
    assert paths.log_file() == tmp_path / "state" / "runfleet" / "runfleet.log"
    assert paths.ssh_log("lyra") == tmp_path / "state" / "runfleet" / "ssh-lyra.log"
    assert not (tmp_path / "state").exists()                       # computing paths creates nothing


@pytest.mark.posix                # XDG_RUNTIME_DIR, socket length, /tmp/runfleet-<uid>
def test_mux_dir_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert paths.mux_dir() == Path("/run/user/1000/runfleet")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/" + "x" * 80)
    assert paths.mux_dir() == Path(f"/tmp/runfleet-{os.getuid()}")   # socket path too long
    # OpenSSH binds "<socket>.<16 chars>" first: a 56-byte base (final socket 100 bytes) already fails on Linux,
    # and macOS' sun_path is 104 bytes -> the final path must stay <= 86 bytes
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/" + "y" * 46)                # base 56 bytes incl. "/runfleet"
    assert paths.mux_dir() == Path(f"/tmp/runfleet-{os.getuid()}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/" + "z" * 32)                # base 42 bytes -> socket 86 bytes: fits
    assert paths.mux_dir() == Path("/" + "z" * 32 + "/runfleet")
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", "/home/u/.cache")
    assert paths.mux_dir() == Path("/home/u/.cache/runfleet/cm")
    monkeypatch.setenv("XDG_CACHE_HOME", "/Users/alexandra/Library/Caches")   # macOS, username > 8 chars
    assert paths.mux_dir() == Path(f"/tmp/runfleet-{os.getuid()}")


@pytest.mark.posix                # directory modes
def test_ensure_private_dir(tmp_path: Path) -> None:
    d = paths.ensure_private_dir(tmp_path / "a" / "b")
    assert d.is_dir() and d.stat().st_mode & 0o777 == 0o700
    d.chmod(0o755)
    paths.ensure_private_dir(d)
    assert d.stat().st_mode & 0o777 == 0o700


# ---------------------------------------------------------------- util/shell
@pytest.mark.parametrize("jid", ["2332861", "31270818_3", "31270818_[1-24%1]", "31270818_[1,3-5]", "2332861.batch",
                                 "31354789.0", "2332861.extern"])
def test_valid_job_ids(jid: str) -> None:
    assert shell.job_id(jid) == jid


@pytest.mark.parametrize("jid", ["", "abc", "1;rm -rf /", "1 2", "1_", "1.foo", "$(id)", "1\n2"])
def test_invalid_job_ids(jid: str) -> None:
    with pytest.raises(ValueError):
        shell.job_id(jid)


def test_other_validators() -> None:
    assert shell.name("lab-pi_1.x") == "lab-pi_1.x"
    with pytest.raises(ValueError):
        shell.name("a b")
    assert shell.pid(642229) == 642229 and shell.pid("42") == 42
    for bad in (0, 1, -5, "x", "4 2", True):
        with pytest.raises(ValueError):
            shell.pid(bad)   # type: ignore[arg-type]
    assert shell.path("/shared/x y/a|b.out") == "/shared/x y/a|b.out"
    for bad in ("rel/path", "/a\nb", "/a\0b", ""):
        with pytest.raises(ValueError):
            shell.path(bad)
    assert shell.quote_join(["a b", "c"]) == "'a b' c"
    assert shell.section_name("process.kill.642229") == "process.kill.642229"
    with pytest.raises(ValueError):
        shell.section_name("Bad.name")


# ---------------------------------------------------------------- sshconfig
def test_list_aliases_include_aware(tmp_path: Path) -> None:
    ssh = tmp_path / ".ssh"
    (ssh / "conf.d").mkdir(parents=True)
    (ssh / "config").write_text(
        "# comment\nHost k2 vega\n  HostName x\nHost *.edu !bad\nHost=lyra\nInclude conf.d/*.conf\n"
        "Match host foo\n  Include extra\nHost \"quoted\" k2\n", encoding="utf-8")
    (ssh / "conf.d" / "a.conf").write_text("Host labstation lynx?\nInclude ../config\n", encoding="utf-8")
    (ssh / "conf.d" / "b.conf").write_text("HOST jumpvm\n", encoding="utf-8")
    (ssh / "extra").write_text("Host atlas\n", encoding="utf-8")
    assert sshconfig.aliases_from(ssh / "config") == ["k2", "vega", "lyra", "labstation", "jumpvm", "atlas",
                                                      "quoted"]


def test_list_aliases_missing_config(tmp_path: Path) -> None:
    assert sshconfig.aliases_from(tmp_path / "none") == []


def test_list_aliases_uses_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("Host a\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))       # what Path.home() reads on Windows
    assert sshconfig.list_aliases() == ["a"]


SSH_G_LYRA = """user alice07
hostname lyra.example.edu
port 22
batchmode no
controlmaster false
connecttimeout none
controlpersist no
proxyjump jumpvm
identityfile ~/.ssh/id_ed25519
identityfile ~/.ssh/id_rsa
"""


def test_parse_ssh_g() -> None:
    s = sshconfig.parse_ssh_g(SSH_G_LYRA, "lyra")
    assert (s.user, s.hostname, s.port, s.proxyjump, s.proxycommand) == ("alice07", "lyra.example.edu", 22,
                                                                         "jumpvm", None)
    assert s.controlpath is None and s.controlpersist == "no"
    assert s.raw["identityfile"] == "~/.ssh/id_ed25519"
    assert s.summary() == {"user": "alice07", "hostname": "lyra.example.edu", "port": "22", "proxyjump": "jumpvm",
                           "controlpath": "none", "controlpersist": "no"}
    with pytest.raises(RunFleetError):
        sshconfig.parse_ssh_g("garbage\n", "x")


@pytest.mark.posix                # fake ssh shell script
async def test_resolve_with_fake_ssh(tmp_path: Path) -> None:
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\n[ \"$1\" = -G ] || exit 9\necho 'user me'\necho 'hostname h.example'\n"
                    "echo 'controlpath /run/cm-abc'\necho 'controlpersist 600'\n", encoding="utf-8")
    fake.chmod(0o755)
    s = await sshconfig.resolve("box", ssh=str(fake))
    assert s.controlpath == "/run/cm-abc" and s.hostname == "h.example"
    bad = tmp_path / "ssh-bad"
    bad.write_text("#!/bin/sh\necho 'ssh: Could not resolve hostname x' >&2\nexit 255\n", encoding="utf-8")
    bad.chmod(0o755)
    with pytest.raises(RunFleetError, match="Could not resolve"):
        await sshconfig.resolve("box", ssh=str(bad))
    with pytest.raises(RunFleetError):
        await sshconfig.resolve("box", ssh=str(tmp_path / "missing"))


@pytest.mark.posix                # /tmp fallback, uids, symlinks
def test_tmp_fallback_mux_dir_rejects_planted_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "_SHARED_TMP", tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)
    victim.chmod(0o755)
    (tmp_path / "runfleet-1").symlink_to(victim)
    with pytest.raises(OSError, match="owned by you"):
        paths.ensure_private_dir(tmp_path / "runfleet-1")
    assert victim.stat().st_mode & 0o777 == 0o755                          # never chmod'ed through the link
    mine = paths.ensure_private_dir(tmp_path / "runfleet-2")
    monkeypatch.setattr(os, "getuid", lambda: mine.stat().st_uid + 1)     # someone else's directory
    with pytest.raises(OSError):
        paths.ensure_private_dir(mine)
