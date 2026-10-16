"""[WP3] Ask the terminal for its background colour (OSC 11) before the TUI starts (§7.11).

``ESC ] 11 ; ? ST`` is answered with ``ESC ] 11 ; rgb:RRRR/GGGG/BBBB`` by xterm, iTerm2, Terminal.app, kitty, WezTerm,
Ghostty, VS Code, foot, Konsole, GNOME Terminal and tmux (its own idea of the outer terminal). It is followed by a
Primary Device Attributes request (``ESC [ c``), which every terminal answers: replies come back in order, so the DA
reply ends the wait at once — a terminal that ignores OSC 11 costs one round trip, not a timeout.

Runs on /dev/tty in non-canonical, no-echo mode and restores it. Only from the foreground process group (a background
job touching the tty would be stopped by SIGTTOU). A colour is taken only from a complete reply (terminated by BEL or
ST). Without a DA reply in time it keeps draining until the line has been quiet for a while, flushes the input and
pushes nothing back, so a slow or split reply never reaches Textual as keystrokes (keys typed in that window are
dropped with it). When the DA reply did come, keys typed meanwhile — everything left once every OSC sequence and DA
reply, complete or cut short, is removed — are pushed back with TIOCSTI.

Known limits: TIOCSTI is off on hardened kernels (Linux ``dev.tty.legacy_tiocsti=0``, OpenBSD), where keys typed
during the query (a round trip, at most :data:`TIMEOUT`) are lost; and a reply arriving after the drain cap reaches
Textual as keys.
"""

from __future__ import annotations

import os
import re
import sys
import time

TIMEOUT = 2.0                    # wait for the DA reply (slow ssh links answer well within this)
QUIET = 0.3                      # after a timeout: drain until nothing arrived for this long …
DRAIN_CAP = 3.0                  # … but never longer than this
# a complete OSC 11 reply: the colour counts only once the terminator (BEL or ST) is there
_OSC11 = re.compile(rb"\x1b\]11;rgba?:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})[^\x07\x1b]*"
                    rb"(?:\x07|\x1b\\)")
_DA1 = re.compile(rb"\x1b\[\?[0-9;]*c")
# what typeahead() removes: any OSC sequence (any payload; terminated, or cut short by the next ESC or the end, an ST
# cut after its ESC included) and any DA1 reply, complete or cut short
_ANY_OSC = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|\x1b\Z)?")
_ANY_DA1 = re.compile(rb"\x1b\[\?[0-9;]*c?")


def parse_reply(data: bytes) -> str | None:
    """#rrggbb from an OSC 11 reply (1-4 hex digits per channel, scaled), else None."""
    m = _OSC11.search(data)
    if m is None:
        return None
    out = "#"
    for h in m.groups():
        out += f"{round(int(h, 16) / (16 ** len(h) - 1) * 255):02x}"
    return out


def typeahead(data: bytes) -> bytes:
    """What the user typed meanwhile: `data` without any OSC sequence or DA reply, complete or partial."""
    return _ANY_DA1.sub(b"", _ANY_OSC.sub(b"", data))


def _read(fd: int, until: float, done: bool, data: bytes, quiet: float | None = None) -> bytes:
    import select

    last = time.monotonic()
    while not done:
        now = time.monotonic()
        if now >= until or (quiet is not None and now - last >= quiet):
            break
        wait = until - now if quiet is None else min(until - now, quiet - (now - last))
        ready, _, _ = select.select([fd], [], [], max(0.0, wait))
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        data += chunk
        last = time.monotonic()
        done = bool(_DA1.search(data))
    return data


def _push_back(fd: int, keys: bytes) -> None:
    if sys.platform == "win32":                         # never reached (no termios); keeps mypy --platform win32 honest
        return
    try:
        import fcntl
        import termios
        for i in range(len(keys)):
            fcntl.ioctl(fd, termios.TIOCSTI, keys[i:i + 1])
    except (ImportError, AttributeError, OSError):      # TIOCSTI is off on hardened kernels: the keys are lost
        pass


def query_background(timeout: float = TIMEOUT, *, tty: str = "/dev/tty") -> str | None:
    """The terminal's background as #rrggbb, or None (no tty, background job, not answered, not POSIX). Never
    raises. Native Windows: None (P1 asks nothing; `RUNFLEET_THEME=light` picks the light palette)."""
    if sys.platform == "win32":
        return None
    try:
        import termios
    except ImportError:
        return None
    try:
        fd = os.open(tty, os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return None
    try:                                                 # the descriptor is closed on every path
        return _query(fd, timeout)
    except (OSError, termios.error):
        return None
    finally:
        os.close(fd)


def _query(fd: int, timeout: float) -> str | None:
    if sys.platform == "win32":
        return None
    import termios

    if os.tcgetpgrp(fd) != os.getpgrp():                 # `runfleet doctor &`: SIGTTOU would stop us
        return None
    old = termios.tcgetattr(fd)
    data = b""
    try:
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN], new[6][termios.VTIME] = 0, 0
        termios.tcsetattr(fd, termios.TCSANOW, new)
        os.write(fd, b"\x1b]11;?\x1b\\\x1b[c")
        start = time.monotonic()
        data = _read(fd, start + timeout, False, data)
        if _DA1.search(data):
            _push_back(fd, typeahead(data))
        else:                                            # late or never: drain until quiet, then drop it all — a
            data = _read(fd, start + timeout + DRAIN_CAP, False, data, quiet=QUIET)    # reply split across the
            termios.tcflush(fd, termios.TCIFLUSH)        # timeout must not come back as keys
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, old)
        except termios.error:
            pass
    return parse_reply(data)
