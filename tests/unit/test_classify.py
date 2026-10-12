"""[WP1] transport/classify.py: every detection rule of §9 rows 1-8."""

from __future__ import annotations

from pathlib import Path

import pytest

from runfleet.models import ConnState, ErrorKind, HostError
from runfleet.transport.classify import (classify_ssh, pauses_polling, remote_shell_error, session_refused,
                                         state_for)


@pytest.mark.parametrize(("stderr", "kind"), [
    ("alice07@lyra.example.edu: Permission denied (publickey,gssapi-with-mic).", ErrorKind.AUTH),
    ("Authentication failed.", ErrorKind.AUTH),
    ("Received disconnect from 1.2.3.4 port 22:2: Too many authentication failures", ErrorKind.AUTH),
    ("No more authentication methods to try.", ErrorKind.AUTH),
    ("read_passphrase: can't open /dev/tty: No such device or address", ErrorKind.AUTH),
    ("Host key verification failed.", ErrorKind.HOSTKEY),
    ("@@@@@@@@@@\n@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n@@@@@@@", ErrorKind.HOSTKEY),
    ("No ED25519 host key is known for k2 and you have requested strict checking.", ErrorKind.HOSTKEY),
    ("ssh: Could not resolve hostname k2x: Name or service not known", ErrorKind.DNS),
    ("ssh: connect to host k2 port 22: Connection timed out", ErrorKind.TIMEOUT),
    ("ssh: connect to host k2 port 22: Operation timed out", ErrorKind.TIMEOUT),
    ("Connection timed out during banner exchange", ErrorKind.TIMEOUT),
    ("ssh: connect to host k2 port 22: Connection refused", ErrorKind.REFUSED),
    ("ssh: connect to host k2 port 22: No route to host", ErrorKind.REFUSED),
    ("ssh: connect to host k2 port 22: Network is unreachable", ErrorKind.REFUSED),
    ("kex_exchange_identification: read: Connection reset by peer", ErrorKind.REFUSED),
    ("Connection closed by 10.0.0.1 port 22", ErrorKind.REFUSED),
    ("mux_client_request_session: session request failed: Session open refused by peer", ErrorKind.MUX),
    ("Control socket connect(/run/user/1/runfleet/cm-abc): Connection refused", ErrorKind.MUX),
    ("something new and strange", ErrorKind.SSH),
])
def test_rules(stderr: str, kind: ErrorKind) -> None:
    err = classify_ssh(255, "Pseudo-terminal will not be allocated because stdin is not a terminal.\n" + stderr)
    assert err is not None and err.kind is kind, err
    assert len(err.summary) <= 80
    assert stderr.splitlines()[-1] in err.detail


@pytest.mark.parametrize(("stderr", "kind", "via"), [
    ("ssh: connect to host jumpvm.example port 22: Connection timed out\n"
     "Connection closed by UNKNOWN port 65535", ErrorKind.JUMP, "via jumpvm"),
    ("channel 0: open failed: connect failed: No route to host\nstdio forwarding failed\n"
     "kex_exchange_identification: Connection closed by remote host", ErrorKind.JUMP, "via jumpvm"),
    ("ssh: Could not resolve hostname jumpvm: Name or service not known", ErrorKind.JUMP, "via jumpvm"),
    ("alice07@jumpvm: Permission denied (publickey).\nConnection closed by UNKNOWN port 65535", ErrorKind.AUTH,
     "(via jumpvm)"),
    ("Host key verification failed.\nConnection closed by UNKNOWN port 65535", ErrorKind.HOSTKEY, "(via jumpvm)"),
])
def test_jump_hop_failures(stderr: str, kind: ErrorKind, via: str) -> None:
    err = classify_ssh(255, stderr, jumps=("jumpvm",))
    assert err is not None and err.kind is kind and via in err.summary, err


def test_target_failure_behind_a_jump_is_not_a_jump_failure() -> None:
    err = classify_ssh(255, "alice07@lyra.example.edu: Permission denied (publickey).", jumps=("jumpvm",))
    assert err is not None and err.kind is ErrorKind.AUTH and "via" not in err.summary


def test_success_timeout_and_partial_rule() -> None:
    assert classify_ssh(0, "") is None
    assert classify_ssh(1, "squeue: error: whatever") is None           # remote command rc, not ssh
    t = classify_ssh(None, "", timed_out=True, timeout_s=25)
    assert t is not None and t.kind is ErrorKind.TIMEOUT and t.summary == "probe timed out after 25 s"
    assert classify_ssh(None, "", timed_out=True, section_ended=True) is None
    assert classify_ssh(255, "Connection reset", section_ended=True) is None


def test_remote_shell_error() -> None:
    e = remote_shell_error(127, "sh: not found")
    assert e.kind is ErrorKind.REMOTE and "remote_shell" in e.summary


@pytest.mark.parametrize(("kind", "n", "state"), [
    (ErrorKind.AUTH, 1, ConnState.NEEDS_AUTH), (ErrorKind.HOSTKEY, 1, ConnState.HOSTKEY),
    (ErrorKind.DNS, 1, ConnState.UNREACHABLE), (ErrorKind.TIMEOUT, 1, ConnState.RETRYING),
    (ErrorKind.REFUSED, 2, ConnState.RETRYING), (ErrorKind.JUMP, 3, ConnState.UNREACHABLE),
    (ErrorKind.MUX, 1, ConnState.RETRYING), (ErrorKind.REMOTE, 1, ConnState.DEGRADED),
])
def test_state_for(kind: ErrorKind, n: int, state: ConnState) -> None:
    assert state_for(HostError(kind, "x"), n) is state


def test_pause_and_session_refused() -> None:
    assert pauses_polling(HostError(ErrorKind.AUTH, "x")) and pauses_polling(HostError(ErrorKind.HOSTKEY, "x"))
    assert not pauses_polling(HostError(ErrorKind.TIMEOUT, "x"))
    assert pauses_polling(remote_shell_error(127, "sh: not found"))
    assert not pauses_polling(HostError(ErrorKind.REMOTE, "detection failed: detect.id"))
    assert session_refused(HostError(ErrorKind.MUX, "x", "channel 2: open failed: Session open refused by peer"))
    assert not session_refused(HostError(ErrorKind.MUX, "x", "Control socket connect: No such file"))


@pytest.mark.parametrize(("rc", "name"), [(-9, "SIGKILL"), (-15, "SIGTERM")])
def test_local_ssh_killed_by_signal_is_transient(rc: int, name: str) -> None:
    """`pkill ssh` / the OOM killer: never the pausing remote-shell row, always a retried SSH error."""
    e = classify_ssh(rc, "")
    assert e is not None and e.kind is ErrorKind.SSH and e.summary == f"ssh killed by {name}"
    assert not pauses_polling(e) and state_for(e, 1) is ConnState.RETRYING
    assert classify_ssh(rc, "", section_ended=True) is None            # partial rule still wins


@pytest.mark.parametrize("rc", [-9, -15])
def test_transport_classifiers_treat_signal_as_transient(rc: int, tmp_path: Path) -> None:
    from runfleet.config import HostConfig
    from runfleet.probe import framing
    from runfleet.transport.base import RunResult
    from runfleet.transport.local import LocalTransport
    from runfleet.transport.openssh import OpenSSHTransport

    res = RunResult(rc=rc, stdout=b"", stderr=b"", elapsed_s=0.1, timed_out=False)
    parsed = framing.ParsedProbe(sections={}, ended=0, markers=0)
    ssh = OpenSSHTransport(HostConfig("k2", "k2"), ssh="ssh", mux_dir=tmp_path, log_path=tmp_path / "l")
    for t in (ssh, LocalTransport(HostConfig("local", "localhost"))):
        err = t._classify(10.0)(res, parsed)
        assert err is not None and err.kind is ErrorKind.SSH and "killed by SIG" in err.summary
        assert not pauses_polling(err)


# ---------------------------------------------------------------- review regressions
def test_banner_exchange_broken_pipe_is_not_a_timeout() -> None:
    err = classify_ssh(255, "banner exchange: Connection to UNKNOWN port -1: Broken pipe")
    assert err is not None and err.kind is ErrorKind.REFUSED
    via = classify_ssh(255, "banner exchange: Connection to UNKNOWN port -1: Broken pipe", jumps=("jumpvm",))
    assert via is not None and via.kind is ErrorKind.JUMP
    timed = classify_ssh(255, "Connection timed out during banner exchange")
    assert timed is not None and timed.kind is ErrorKind.TIMEOUT


def test_unidentified_hop_of_a_chain_is_not_blamed_on_the_first() -> None:
    err = classify_ssh(255, "Host key verification failed.\nstdio forwarding failed", jumps=("bastion1", "bastion2"))
    assert err is not None and err.kind is ErrorKind.HOSTKEY and "(via jump chain)" in err.summary
    named = classify_ssh(255, "ssh: connect to host bastion2 port 22: Connection timed out",
                         jumps=("bastion1", "bastion2"))
    assert named is not None and "via bastion2" in named.summary


def test_bare_permission_denied_is_not_auth() -> None:
    assert classify_ssh(126, "bash: /bin/sh: Permission denied") is None          # remote exec: remote_shell_error
    sock = classify_ssh(255, "Control socket connect(/run/user/1/rf/cm-x): Permission denied")
    assert sock is not None and sock.kind is ErrorKind.MUX
    retry = classify_ssh(255, "Permission denied, please try again.")
    assert retry is not None and retry.kind is ErrorKind.AUTH
