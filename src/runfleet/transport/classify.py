"""[WP1] Classify ssh stderr/rc -> ErrorKind/ConnState (table §9).

`classify_ssh` turns a failed ssh invocation (rc 255, a local timeout, a non-running remote shell) into a `HostError`;
`state_for` maps a transport-level error plus the consecutive-failure count to the host's `ConnState` (§4.4).

Jump hops (§9 row 5): network-type failures that mention a jump host, or the hop's own `-W` errors, get kind JUMP and
"via <jump>" in the summary. Authentication and host-key failures on a hop keep kind AUTH/HOSTKEY (so the host still
pauses as NEEDS_AUTH/HOSTKEY) and name the hop in the summary.
"""

from __future__ import annotations

import re
import signal
from collections.abc import Sequence
from dataclasses import dataclass

from runfleet.models import ConnState, ErrorKind, HostError

DETAIL_LIMIT = 4096
SUMMARY_LIMIT = 80


@dataclass(frozen=True, slots=True)
class _Rule:
    kind: ErrorKind
    pattern: re.Pattern[str]
    summary: str


def _rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# Order matters: the first matching rule wins (host-key text can contain "Permission denied"-like words).
RULES: tuple[_Rule, ...] = (
    _Rule(ErrorKind.HOSTKEY, _rx(r"REMOTE HOST IDENTIFICATION HAS CHANGED"), "host key changed"),
    _Rule(ErrorKind.HOSTKEY, _rx(r"Host key verification failed", r"No \S+ host key is known"), "unknown host key"),
    # OpenSSH's own wording only: a bare "Permission denied" is also a remote exec failure (rc 126) or an unusable
    # control socket, neither of which a re-authentication fixes
    _Rule(ErrorKind.AUTH, _rx(r"Permission denied \(", r"Permission denied, please try again",
                              r"Authentication failed", r"No more authentication methods",
                              r"Too many authentication failures", r"read_passphrase: can't open /dev/tty"),
          "authentication needed"),
    _Rule(ErrorKind.DNS, _rx(r"Could not resolve hostname", r"Name or service not known"), "dns lookup failed"),
    _Rule(ErrorKind.MUX, _rx(r"Session open refused by peer", r"mux_client_\w+", r"Control socket connect"),
          "ssh master trouble"),
    _Rule(ErrorKind.TIMEOUT, _rx(r"Connection timed out", r"Operation timed out", r"timed out during banner exchange",
                                 r"banner exchange.*timed out"), "connection timed out"),
    _Rule(ErrorKind.REFUSED, _rx(r"Connection refused", r"No route to host", r"Network is unreachable",
                                 r"kex_exchange_identification", r"Connection closed by",
                                 r"Connection reset by", r"banner exchange: .*(Broken pipe|reset)"),
          "connection failed"),
)

# The `-W` hop's own failures; any of these means the jump leg broke.
_JUMP_MARKERS = _rx(r"channel \d+: open failed", r"stdio forwarding failed", r"Connection closed by UNKNOWN",
                    r"Connection to UNKNOWN port")
_NOISE = _rx(r"^Pseudo-terminal will not be allocated", r"^Warning: Permanently added", r"^\*\*", r"^\s*$")
SESSION_REFUSED = _rx(r"Session open refused by peer", r"mux_client_request_session: session request failed")


def _lines(stderr: str) -> list[str]:
    return [ln.strip() for ln in stderr.splitlines() if not _NOISE.search(ln)]


def _short(text: str) -> str:
    return text if len(text) <= SUMMARY_LIMIT else text[: SUMMARY_LIMIT - 1] + "…"


def _detail(stderr: str) -> str:
    return stderr[-DETAIL_LIMIT:]


def _jump_named(line: str, jumps: Sequence[str]) -> str | None:
    low = line.lower()
    for j in jumps:
        if j and re.search(rf"(^|[\s@'\"(]){re.escape(j.lower())}([\s:'\".)]|$)", low):
            return j
    return None


def classify_ssh(rc: int | None, stderr: str, *, timed_out: bool = False, timeout_s: float | None = None,
                 jumps: Sequence[str] = (), section_ended: bool = False) -> HostError | None:
    """Transport-level error of one ssh run, or None if ssh evidently worked.

    `jumps`: jump host names/aliases (for row 5). `section_ended`: at least one probe end marker arrived (§4.2
    partial rule: then neither a local timeout nor a dropped connection is a transport failure).
    """
    if section_ended:
        return None
    if timed_out:
        after = f" after {timeout_s:.0f} s" if timeout_s is not None else ""
        return HostError(ErrorKind.TIMEOUT, _short(f"probe timed out{after}"), _detail(stderr))
    if rc is not None and rc < 0:
        return killed_by_signal(rc, stderr)
    if rc == 255 or (rc is not None and rc != 0 and _matches_any(stderr)):
        return _classify_255(stderr, jumps)
    return None


def killed_by_signal(rc: int, stderr: str, *, what: str = "ssh") -> HostError:
    """The local client died from a signal (rc < 0: `pkill ssh`, the OOM killer, a logout). That says nothing about
    the remote shell, so it is a transient SSH error retried with backoff — never the pausing REMOTE row."""
    try:
        name = signal.Signals(-rc).name
    except ValueError:
        name = f"signal {-rc}"
    return HostError(ErrorKind.SSH, _short(f"{what} killed by {name}"), _detail(f"rc={rc}\n{stderr}"))


def _matches_any(stderr: str) -> bool:
    return any(rule.pattern.search(stderr) for rule in RULES)


def _classify_255(stderr: str, jumps: Sequence[str]) -> HostError:
    lines = _lines(stderr)
    hop = None
    for ln in lines:
        hop = hop or _jump_named(ln, jumps)
    hop_failed = bool(jumps) and (hop is not None or bool(_JUMP_MARKERS.search(stderr)))
    hop_name = hop or (jumps[0] if len(jumps) == 1 else "jump chain" if jumps else "")   # never blame a guessed hop
    for rule in RULES:
        line = next((ln for ln in lines if rule.pattern.search(ln)), None)
        if line is None:
            continue
        kind, summary = rule.kind, rule.summary
        if hop_failed and kind in (ErrorKind.AUTH, ErrorKind.HOSTKEY):
            summary = f"{summary} (via {hop_name})"
        elif hop_failed and kind in (ErrorKind.DNS, ErrorKind.TIMEOUT, ErrorKind.REFUSED):
            kind, summary = ErrorKind.JUMP, f"{summary} via {hop_name}"
        return HostError(kind, _short(summary), _detail(stderr))
    if hop_failed:
        return HostError(ErrorKind.JUMP, _short(f"jump host failed via {hop_name}"), _detail(stderr))
    first = lines[0] if lines else "ssh failed (rc 255, no message)"
    return HostError(ErrorKind.SSH, _short(first), _detail(stderr))


REMOTE_SHELL_SUMMARY = 'remote shell could not run sh -s (try remote_shell="login")'


def remote_shell_error(rc: int | None, stderr: str) -> HostError:
    """§9 row 8: the remote side could not run the probe (rc 127 for `sh`, or no markers at all)."""
    return HostError(ErrorKind.REMOTE, REMOTE_SHELL_SUMMARY, _detail(f"rc={rc}\n{stderr}"))


def state_for(error: HostError, consecutive_failures: int) -> ConnState:
    """Host state after a transport-level failure (§4.4, §9)."""
    if error.kind is ErrorKind.AUTH:
        return ConnState.NEEDS_AUTH
    if error.kind is ErrorKind.HOSTKEY:
        return ConnState.HOSTKEY
    if error.kind is ErrorKind.DNS:
        return ConnState.UNREACHABLE
    if error.kind is ErrorKind.REMOTE:
        return ConnState.DEGRADED
    return ConnState.RETRYING if consecutive_failures < 3 else ConnState.UNREACHABLE


def pauses_polling(error: HostError) -> bool:
    """AUTH / HOSTKEY / an unusable remote shell (§9 row 8) stop polling until a manual refresh or connect; every
    other transport-level error is retried with backoff."""
    if error.kind is ErrorKind.REMOTE:
        return error.summary == REMOTE_SHELL_SUMMARY
    return error.kind in (ErrorKind.AUTH, ErrorKind.HOSTKEY)


def session_refused(error: HostError) -> bool:
    """The shared master refused another session (MaxSessions): the host's on-demand lane drops to 1."""
    return bool(SESSION_REFUSED.search(error.detail))
