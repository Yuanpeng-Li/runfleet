"""[WP1] Transports: OpenSSH, Local, Fake (Demo lives in demo/); `make_transport` entry point for cli.py."""

from __future__ import annotations

from runfleet.config import HostConfig
from runfleet.transport.base import Transport


def make_transport(host: HostConfig) -> Transport:   # LocalTransport if host.ssh == "local", else OpenSSHTransport
    from runfleet.transport.local import LocalTransport
    from runfleet.transport.openssh import OpenSSHTransport

    if host.ssh == "local":
        return LocalTransport(host)
    return OpenSSHTransport(host)
