"""[WP1] platformdirs locations: config, state, cache, mux dir, ssh logs.

Functions only compute paths; directories are created by the ``ensure_*`` helpers (mode 0700) when a caller is about
to write there, so importing RunFleet or running ``--help`` never touches the filesystem.

Windows (§8.1): platformdirs with ``appauthor=False``: config in ``%APPDATA%\\runfleet`` (roaming), state in
``%LOCALAPPDATA%\\runfleet``, cache in ``%LOCALAPPDATA%\\runfleet\\Cache``. File modes mean nothing there (the
user profile's ACL keeps them private), so the 0700/0600 fixes and the ``/tmp`` owner check are POSIX only.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import platformdirs

from runfleet.util import platform

APP = "runfleet"
# A ControlPath is rendered as "<mux_dir>/cm-<40 hex>" (OpenSSH %C). OpenSSH first binds "<path>.<16 random chars>"
# and then links it into place, so that temporary name (17 bytes longer) must fit sun_path: 104 bytes on macOS/BSD,
# 108 on Linux, including the NUL. Budget for the portable limit: final path <= 103 - 17 = 86 bytes.
_SOCKET_NAME_LEN = len("cm-") + 40
_BIND_SUFFIX_LEN = 17
_SUN_PATH_MAX = 103
_MAX_SOCKET_PATH = _SUN_PATH_MAX - _BIND_SUFFIX_LEN
_SHARED_TMP = Path("/tmp")


def _dirs(*, roaming: bool = False) -> platformdirs.PlatformDirsABC:
    """`appauthor=False`: one `runfleet` level on Windows (not `runfleet\\runfleet`); `roaming` matters on Windows
    only (the config follows a roaming profile, state and cache stay on the machine)."""
    if platform.IS_WINDOWS:
        from platformdirs.windows import Windows

        return Windows(APP, appauthor=False, roaming=roaming)
    return platformdirs.PlatformDirs(APP, appauthor=False, roaming=roaming)


def config_dir() -> Path:
    return Path(_dirs(roaming=True).user_config_dir)


def state_dir() -> Path:
    return Path(_dirs().user_state_dir)


def cache_dir() -> Path:
    return Path(_dirs().user_cache_dir)


def config_file() -> Path:         # ~/.config/runfleet/config.toml
    return config_dir() / "config.toml"


def cache_file() -> Path:          # ~/.local/state/runfleet/cache.sqlite  (Cache(cache_file()))
    return state_dir() / "cache.sqlite"


def log_file() -> Path:            # ~/.local/state/runfleet/runfleet.log
    return state_dir() / "runfleet.log"


def ui_state_file() -> Path:
    return state_dir() / "ui.json"


def debug_dir() -> Path:
    """Raw section dumps written with ``--debug`` (§5)."""
    return state_dir() / "debug"


def ssh_log(host: str) -> Path:
    """`-E` log of master creation / connect for one host (§4.2)."""
    return state_dir() / f"ssh-{host}.log"


def mux_dir() -> Path:
    """Directory for managed ControlMaster sockets and ``instance.lock`` (§4.1).

    ``$XDG_RUNTIME_DIR/runfleet`` if set, else ``~/.cache/runfleet/cm``; if a socket path there would exceed 86 bytes
    (OpenSSH binds ``<path>.<16 chars>`` first, and macOS ``sun_path`` holds 104), ``/tmp/runfleet-<uid>``. Windows
    has no ControlMaster (sharing is always off there, §4.1): ``%LOCALAPPDATA%\\runfleet\\Cache\\cm`` holds only the
    instance locks, so no length limit and no shared fallback.
    """
    if platform.IS_WINDOWS:
        return cache_dir() / "cm"
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) / APP if runtime else cache_dir() / "cm"
    if len(os.fsencode(str(base))) + 1 + _SOCKET_NAME_LEN > _MAX_SOCKET_PATH:
        base = _SHARED_TMP / f"{APP}-{platform.current_uid()}"
    return base


def display_path(path: Path) -> str:
    """`path` for messages: `~/…` under the home directory (`~\\…` on Windows, which PowerShell understands too),
    else as is."""
    try:
        rel = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return ("~\\" if platform.IS_WINDOWS else "~/") + str(rel)


def ensure_private_dir(path: Path) -> Path:
    """Create `path` (and parents) if needed and make it 0700; returns `path`. Used for the mux dir and the state
    dir (log, cache, ui.json, ssh logs, debug dumps hold job names, paths and command lines: never readable by
    other users of a shared machine, whatever the umask)."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if platform.IS_WINDOWS:          # no modes, no shared /tmp: the profile's ACL keeps it private
        return path
    if path.parent == _SHARED_TMP:
        # the /tmp fallback name is predictable: another user may have planted a symlink or a directory of theirs
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid != platform.current_uid():
            raise OSError(errno.EPERM, "not a directory owned by you; remove it", str(path))
    if path.stat().st_mode & 0o777 != 0o700:
        path.chmod(0o700)
    return path


def private_file(path: Path) -> Path:
    """Create `path` empty with mode 0600 if missing, else tighten it to 0600; returns `path`."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.close(fd)
    if not platform.IS_WINDOWS and path.stat().st_mode & 0o077:
        path.chmod(0o600)
    return path


def write_private(path: Path, data: str | bytes) -> None:
    """Replace the content of `path` (created 0600, never readable by others in between)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
    if not platform.IS_WINDOWS and path.stat().st_mode & 0o077:
        path.chmod(0o600)
