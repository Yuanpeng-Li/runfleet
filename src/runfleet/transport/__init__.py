"""[WP1] Transports: OpenSSH, Session, Local, Fake (Demo lives in demo/); `make_transport` entry point for cli.py."""

from __future__ import annotations

from runfleet.config import HostConfig
from runfleet.transport.base import Transport


def make_transport(host: HostConfig) -> Transport:
    """LocalTransport if host.ssh == "local"; SessionTransport where `mux.wants_session` (Windows unless mux = "off",
    elsewhere mux = "session", §4.8); else OpenSSHTransport."""
    from runfleet.transport import mux
    from runfleet.transport.local import LocalTransport
    from runfleet.transport.openssh import OpenSSHTransport
    from runfleet.transport.session import SessionTransport

    if host.ssh == "local":
        return LocalTransport(host)
    if mux.wants_session(host):
        return SessionTransport(host)
    return OpenSSHTransport(host)
