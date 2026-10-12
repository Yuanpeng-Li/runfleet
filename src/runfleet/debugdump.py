"""[WP1] `--debug` raw probe dumps (§10.2): every probe's rendered script and raw output, in RunFleet wire format.

Off unless `enable()` was called (the CLI does it for `--debug`). Each probe writes `<dir>/<host>/<stamp>-<purpose>.in`
(the rendered script), `.out` (raw stdout, i.e. the framed sections), `.err` (ssh stderr) and `.meta` (rc, timing,
transport error). Files are capped in size and rotated per host, so a long `--debug` session cannot fill the disk.
These dumps use the same wire format as the field-probe fixtures and can be committed as fixtures. Attach probes (§7.16,
purpose "attach") are dumped the same way: each `slurm.attach.node<i>` section's stdout in `.out` is the nested node
script's own framed output, so one dump is a complete `tests/fixtures/attach/` case.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("runfleet.debugdump")

MAX_PROBES_PER_HOST = 200        # rotation: keep the newest N probes (4 files each) per host
MAX_BYTES = 4 << 20              # per file; longer output keeps its head and a marker

_dir: Path | None = None
_lock = threading.Lock()
_seq = 0


def enable(directory: Path) -> None:
    global _dir
    _dir = directory


def disable() -> None:
    global _dir
    _dir = None


def enabled() -> bool:
    return _dir is not None


def _safe(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "host"
    return out if out.strip(".") else out.replace(".", "_")      # "." / ".." would leave the dump dir


def _cap(data: bytes) -> bytes:
    if len(data) <= MAX_BYTES:
        return data
    return data[:MAX_BYTES] + f"\n[runfleet: {len(data) - MAX_BYTES} more bytes not dumped]\n".encode()


def dump_probe(host: str, purpose: str, script: str, stdout: bytes, stderr: bytes, meta: str) -> Path | None:
    """Write one probe; returns the `.in` path (None when disabled or on any I/O error — never raises)."""
    global _seq
    base_dir = _dir
    if base_dir is None:
        return None
    try:
        with _lock:
            _seq += 1
            seq = _seq
        from runfleet import paths

        paths.ensure_private_dir(base_dir)          # the top dir too, not only the per-host dirs (any umask)
        d = paths.ensure_private_dir(base_dir / _safe(host))
        stem = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{seq:05d}-{_safe(purpose)}"
        paths.write_private(stem.with_suffix(".in"), script)
        paths.write_private(stem.with_suffix(".out"), _cap(stdout))
        paths.write_private(stem.with_suffix(".err"), _cap(stderr))
        paths.write_private(stem.with_suffix(".meta"), meta.rstrip("\n") + "\n")
        _rotate(d)
        return stem.with_suffix(".in")
    except OSError as e:
        log.debug("debug dump for %s failed: %s", host, e)
        return None


def _rotate(d: Path) -> None:
    ins = sorted(d.glob("*.in"))
    for old in ins[: max(0, len(ins) - MAX_PROBES_PER_HOST)]:
        for suffix in (".in", ".out", ".err", ".meta"):
            try:
                old.with_suffix(suffix).unlink()
            except OSError:
                pass
