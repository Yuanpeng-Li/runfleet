"""[WP1] List Host aliases from ~/.ssh/config (Include-aware) and resolve hosts with `ssh -G`."""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from runfleet.errors import RunFleetError
from runfleet.util import platform

log = logging.getLogger("runfleet.sshconfig")

_MAX_INCLUDE_DEPTH = 16
_PATTERN_CHARS = frozenset("*?!")


def list_aliases() -> list[str]:   # ~/.ssh/config Host aliases, Include-aware, patterns with * ? ! skipped
    return aliases_from(Path.home() / ".ssh" / "config")


def aliases_from(path: Path, *, ssh_dir: Path | None = None) -> list[str]:
    """Concrete `Host` aliases of an ssh config file and everything it `Include`s, in file order, deduplicated.

    Relative Include paths are resolved against `ssh_dir` (default: the directory of `path`, i.e. ~/.ssh for the user
    config), globs are expanded, and Include inside a Host/Match block is followed too (we only list names).
    """
    base = ssh_dir if ssh_dir is not None else path.parent
    seen: dict[str, None] = {}
    _scan(path, base, seen, depth=0, visited=set())
    return list(seen)


def _tokens(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    # "Key=Value" and "Key Value" are both legal; only the first "=" separates.
    key, sep, rest = stripped.partition("=")
    if sep and " " not in key.strip() and "\t" not in key.strip():
        stripped = f"{key.strip()} {rest.strip()}"
    try:
        return platform.split_command(stripped, comments=True)    # Windows: backslashes in paths stay
    except ValueError:
        return stripped.split()


def _scan(path: Path, base: Path, seen: dict[str, None], *, depth: int, visited: set[Path]) -> None:
    if depth > _MAX_INCLUDE_DEPTH:
        log.warning("ssh config Include depth > %d at %s", _MAX_INCLUDE_DEPTH, path)
        return
    try:
        real = path.resolve()
        if real in visited:
            return
        visited.add(real)
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        toks = _tokens(line)
        if not toks:
            continue
        key = toks[0].lower()
        if key == "host":
            for name in toks[1:]:
                if not (_PATTERN_CHARS & set(name)):
                    seen.setdefault(name, None)
        elif key == "include":
            for pattern in toks[1:]:
                expanded = os.path.expanduser(pattern)
                full = expanded if os.path.isabs(expanded) else str(base / expanded)
                for match in sorted(glob.glob(full)):
                    _scan(Path(match), base, seen, depth=depth + 1, visited=visited)


# ---------------------------------------------------------------- `ssh -G`
@dataclass(frozen=True, slots=True)
class SshSettings:
    """The parts of `ssh -G <alias>` RunFleet cares about (§4.1). `None` = the option is "none"/absent."""
    alias: str
    user: str
    hostname: str
    port: int
    proxyjump: str | None = None
    proxycommand: str | None = None
    controlmaster: str = "false"
    controlpath: str | None = None
    controlpersist: str = "no"
    raw: Mapping[str, str] = field(default_factory=dict)   # first value of every key

    def summary(self) -> dict[str, str]:
        """DetectReport.ssh_summary / doctor: user, hostname, port, proxyjump, controlpath, controlpersist."""
        return {"user": self.user, "hostname": self.hostname, "port": str(self.port),
                "proxyjump": self.proxyjump or "none", "controlpath": self.controlpath or "none",
                "controlpersist": self.controlpersist}


def _none(value: str | None) -> str | None:
    return None if value is None or value.strip().lower() in ("", "none") else value.strip()


def parse_ssh_g(text: str, alias: str) -> SshSettings:
    """Parse `ssh -G` output ("key value" lines, lower-case keys). Raises RunFleetError if it is not that."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key and key.islower() and key not in raw:
            raw[key] = value.strip()
    if "hostname" not in raw:
        raise RunFleetError(f"ssh -G {alias}: unexpected output")
    try:
        port = int(raw.get("port", "22"))
    except ValueError:
        port = 22
    return SshSettings(
        alias=alias, user=raw.get("user", ""), hostname=raw["hostname"], port=port,
        proxyjump=_none(raw.get("proxyjump")), proxycommand=_none(raw.get("proxycommand")),
        controlmaster=raw.get("controlmaster", "false"), controlpath=_none(raw.get("controlpath")),
        controlpersist=raw.get("controlpersist", "no"), raw=raw,
    )


async def resolve(alias: str, *, ssh: str = "ssh", extra: Sequence[str] = (), timeout: float = 10.0) -> SshSettings:
    """Run `ssh -G [extra] -- <alias>` (local config evaluation, no network). Raises RunFleetError."""
    from runfleet.transport import spawn   # local: transport imports this module for SshSettings

    try:
        res = await spawn.run([ssh, "-G", *extra, "--", alias], timeout=timeout)
    except spawn.SpawnError as e:
        raise RunFleetError(f"ssh -G {alias}: {e}") from e
    if res.timed_out or res.rc != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        msg = err[-1] if err else ("timed out" if res.timed_out else f"rc {res.rc}")
        raise RunFleetError(f"ssh -G {alias}: {msg}")
    return parse_ssh_g(res.stdout.decode("utf-8", errors="replace"), alias)
