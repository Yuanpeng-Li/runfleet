"""[WP1] FROZEN Section, SectionResult, SectionOutcome, ProbeScript, StreamSpec (DESIGN.md §3.4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

class SectionOutcome(StrEnum):
    OK = "ok"                    # rc == 0 and stdout non-empty
    EMPTY = "empty"              # rc == 0, no stdout, no stderr: a valid empty answer (no jobs, no GPU procs)
    FAILED = "failed"            # rc not in {0, 124, 127, 137}
    STDERR_ONLY = "stderr_only"  # rc == 0, stdout empty, stderr non-empty (lyra `--json | head` lesson) -> failure
    TIMEOUT = "timeout"          # rc 124/137 (remote timeout or abandoned), or unfinished at local timeout
    MISSING = "missing"          # a `requires` tool is absent (rc 127 + sentinel)
    INCOMPLETE = "incomplete"    # begin marker without end marker (connection died mid-section)
    NOT_RUN = "not_run"          # no marker at all (script died before it)

@dataclass(frozen=True, slots=True)
class Section:
    name: str                    # "<owner>.<part>", regex ^[a-z]+\.[a-z0-9_.]+$, unique in a probe
    cmd: str                     # POSIX sh snippet; every data value already validated + shlex-quoted
    timeout: float = 15.0        # remote `timeout -k 2 <t>` when available
    requires: tuple[str, ...] = ()           # tools that must exist (`command -v`), else MISSING
    late: bool = False           # rendered after all non-late sections (may hang in D state: nvidia.*)
    meta: Mapping[str, object] = field(default_factory=dict)   # structured args for Fake/Demo transports (Appendix A)
    max_stdout: int = 8_000_000  # client-side truncation guard

    @property
    def owner(self) -> str:      # routing key (§3.1): collector name, "detect" or "log"
        return self.name.split(".", 1)[0]

@dataclass(frozen=True, slots=True)
class SectionResult:
    name: str
    outcome: SectionOutcome
    rc: int | None
    stdout: str                  # utf-8, errors="replace"; exact command output (only the framing "\n" removed)
    stderr: str                  # <= 4 KiB
    ms: int | None               # remote wall time (None without GNU date)

    @property
    def ok(self) -> bool:
        return self.outcome in (SectionOutcome.OK, SectionOutcome.EMPTY)

@dataclass(frozen=True, slots=True)
class ProbeScript:
    sections: tuple[Section, ...]
    nonce: str                   # 12 hex chars, secrets.token_hex(6)
    purpose: Literal["poll", "detect", "detail", "action", "log", "doctor", "attach", "account"] = "poll"
                                 # "account": an on-demand account report (§5.2.11; Account, additive)

@dataclass(frozen=True, slots=True)
class StreamSpec:
    name: str                    # "log.tail"
    cmd: str                     # POSIX sh, runs until stdin closes (watchdog form, §4.7)
    meta: Mapping[str, object] = field(default_factory=dict)
