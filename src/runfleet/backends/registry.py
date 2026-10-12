"""[WP2] BUILTIN collectors + entry-point group "runfleet.collectors" (DESIGN.md §3.6, §5.1)."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from runfleet.backends.base import Collector
from runfleet.backends.common.nvidia import NvidiaMonitor
from runfleet.backends.common.procfs import ProcfsMonitor
from runfleet.backends.process.collector import ProcessCollector
from runfleet.backends.slurm.collector import SlurmCollector

log = logging.getLogger("runfleet.backends.registry")

ENTRY_POINT_GROUP = "runfleet.collectors"

BUILTIN: dict[str, type[Collector]] = {"slurm": SlurmCollector, "process": ProcessCollector, "nvidia": NvidiaMonitor,
                                       "procfs": ProcfsMonitor}


def collectors() -> dict[str, type[Collector]]:
    # BUILTIN plus entry-point group "runfleet.collectors"; a broken plugin is logged and skipped
    out = dict(BUILTIN)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception:
            log.exception("collector plugin %s (%s) failed to load; skipped", ep.name, ep.value)
            continue
        if not (isinstance(cls, type) and issubclass(cls, Collector)):
            log.error("collector plugin %s is not a Collector subclass; skipped", ep.name)
            continue
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or name != ep.name:
            log.error("collector plugin %s: class name attribute %r must equal the entry-point name; skipped",
                      ep.name, name)
            continue
        if name in out:
            log.error("collector plugin %s would shadow an existing collector; skipped", name)
            continue
        out[name] = cls
    return out
