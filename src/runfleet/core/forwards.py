"""[WP1] Local port forwards through a host's ssh master (`P`, palette › Forward a port…): URL discovery, local ports.

A forward is `ssh -O forward -L 127.0.0.1:<lport>:<node>:<rport>` against the master RunFleet already uses (the
transport builds it); nothing runs on the remote side. This module holds the pure parts: the `PortForward` record,
finding `http(s)://host:port/…` URLs in log text (TensorBoard, Jupyter with `?token=`, Gradio …) and picking a free
local port, preferring the remote port's number.
"""

from __future__ import annotations

import re
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

from runfleet.models import JobKey

# scheme://host:port[/path?query] in a log line; the port is required (a bare http://host is no service we can find)
_URL = re.compile(r"\bhttps?://[A-Za-z0-9_.\-\[\]]+:\d{2,5}(?:[/?#][^\s'\"<>`)\]]*)?")
_LABELS = (("token=", "Jupyter"), ("tensorboard", "TensorBoard"), ("gradio", "Gradio"), ("jupyter", "Jupyter"),
           ("streamlit", "Streamlit"), ("ray dashboard", "Ray dashboard"), ("mlflow", "MLflow"),
           ("wandb", "W&B local"), ("vllm", "vLLM"), ("uvicorn", "Uvicorn"))
_PORT_LABELS = {6006: "TensorBoard", 8888: "Jupyter", 7860: "Gradio", 8501: "Streamlit", 8265: "Ray dashboard",
                5000: "MLflow?", 8000: "HTTP server?"}


@dataclass(frozen=True, slots=True)
class UrlCandidate:
    """A service URL found in a log: what to forward and what to show afterwards."""
    port: int
    scheme: str                  # "http" / "https"
    target: str                  # host part as printed ("localhost", "0.0.0.0", "n-38-01")
    path: str                    # path + query + fragment, token kept ("/lab?token=…"); "" for "/"
    label: str                   # "TensorBoard", "Jupyter", "" when unknown
    line: str = ""               # the log line it came from (trimmed)


@dataclass(frozen=True, slots=True)
class PortForward:
    host: str                    # RunFleet host whose master carries the forward
    node: str                    # where the port listens, as seen from the login side ("localhost" on workstations)
    remote_port: int
    local_port: int
    url: str                     # what to open locally: "http://localhost:6006/" (token preserved)
    job: JobKey | None = None
    label: str = ""
    started_at: datetime | None = None


def find_urls(lines: Iterable[str]) -> list[UrlCandidate]:
    """Service URLs in log lines, one per port (the first one with a path / token wins over a bare one)."""
    out: dict[int, UrlCandidate] = {}
    for raw in lines:
        line = _strip_ansi(raw)
        for m in _URL.finditer(line):
            try:
                parts = urlsplit(m.group(0).rstrip(".,;:"))
                port = parts.port
            except ValueError:
                continue
            if port is None or not 1 <= port <= 65535:
                continue
            path = parts.path if parts.path not in ("", "/") else ""
            if parts.query:
                path = (path or "/") + "?" + parts.query
            if parts.fragment:
                path = (path or "/") + "#" + parts.fragment
            cand = UrlCandidate(port, parts.scheme, parts.hostname or "", path, _label(line, port),
                                line.strip()[:160])
            prev = out.get(port)
            if prev is None or (not prev.path and cand.path):
                out[port] = cand
    return list(out.values())


def _label(line: str, port: int) -> str:
    low = line.lower()
    for needle, label in _LABELS:
        if needle in low:
            return label
    return _PORT_LABELS.get(port, "")


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def local_url(local_port: int, *, scheme: str = "http", path: str = "") -> str:
    """The URL to open on this machine: same scheme, path and query (Jupyter's token keeps working)."""
    if path and not path.startswith(("/", "?", "#")):
        path = "/" + path
    return f"{scheme}://localhost:{local_port}{path or '/'}"


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_local_port(preferred: int | None, taken: Iterable[int] = ()) -> int:
    """`preferred` (the remote port's number) when it is free here and not used by another forward, else a free
    ephemeral port. The port is only probed, not held: `ssh -O forward` may still lose a race (the caller retries)."""
    busy = set(taken)
    if preferred and preferred not in busy and port_is_free(preferred):
        return preferred
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port not in busy:
            return port
    raise OSError("no free local port")
