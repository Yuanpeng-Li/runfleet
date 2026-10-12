"""Exit log scan (§5.3, §7 "Why it ended") and failure triage (§5.2.8): why a program died, from its log.

Pure and shared: the workstation collector classifies ended tasks with `EXIT_PATTERN`; job detail adds the cause found
in the log tail of an ended Slurm job (Traceback / CUDA OOM / Killed) to "Why it ended". `cause_of` maps an ended job
to one cause chip (OOM-GPU, TIMEOUT, NCCL, …) from its state, exit status and log text RunFleet already fetched;
`line_level` / `ERROR_LINE` mark the error and warning lines the log viewer highlights and jumps between.
"""

from __future__ import annotations

import re
import signal
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from runfleet.models import Job, JobDetail, JobState, StateGroup

EXIT_PATTERN = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|MemoryError|"
                          r"Killed|Segmentation fault|core dumped|^\S*Error:")
EXIT_SCAN_LINES = 50
PENDING_EXIT_HINT = "ended; scanning log"   # a workstation task whose end is not classified yet (process.exitscan)
_TRACEBACK_HEADER = re.compile(r"^\s*Traceback \(most recent call last\):?\s*$")


def exit_cause(lines: Iterable[str], *, limit: int = 100) -> str | None:
    """The most telling failure line among the last EXIT_SCAN_LINES: the last matching line that is not a bare
    `Traceback (most recent call last):` header (that is the exception itself, e.g. `torch.OutOfMemoryError: CUDA
    out of memory …`), else the header; None when nothing matches."""
    tail = [ln.rstrip("\r\n") for ln in lines][-EXIT_SCAN_LINES:]
    hits = [ln.strip() for ln in tail if EXIT_PATTERN.search(ln)]
    if not hits:
        return None
    best = next((h for h in reversed(hits) if not _TRACEBACK_HEADER.match(h)), hits[0])
    return best if len(best) <= limit else best[: limit - 1] + "…"


# ---------------------------------------------------------------- failure triage (§5.2.8)
# Error / warning lines: what the log viewer's `]` / `[` jump between and what log panes highlight. NCCL INFO chatter
# (NCCL_DEBUG=INFO) is neither; `NCCL WARN` is a warning, an NCCL line naming an error / timeout / abort an error.
ERROR_LINE = re.compile(
    r"Traceback \(most recent call last\)|Error\b|\bERROR\b|\berror:|\bFATAL\b|\bFatal\b|"
    r"CUDA out of memory|(?i:out of memory)|\bNCCL\b.*(?i:error|timeout|timed out|abort|fail)|nccl\w*Error|"
    r"Watchdog caught collective|\bKilled\b|\boom\b|(?i:oom[_-]kill)|slurmstepd: error|DUE TO TIME LIMIT|PREEMPT|"
    r"Segmentation fault|core dumped|\bAborted\b|(?i:\bnan loss\b|\bloss\b[^\n]{0,24}?\b(?:nan|inf)\b)|"
    r"Disk quota exceeded|No space left on device")
WARN_LINE = re.compile(r"\bWARN(?:ING)?\b|Warning\b|\bwarning:")
# every ERROR_LINE / WARN_LINE alternative contains one of these (lower-case): a cheap prefilter for line_level
_LEVEL_KEYS = ("traceback", "error", "fatal", "memory", "nccl", "watchdog", "killed", "oom", "time limit", "preempt",
               "segmentation", "core dumped", "aborted", "nan", "inf", "quota", "no space", "warn")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# (code, pattern, one actionable line), most telling first: one rank's GPU OOM / segfault / NaN shows up as NCCL
# timeouts on the other ranks, so NCCL only wins when nothing more specific is in the log.
CAUSE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("OOM-GPU", re.compile(r"CUDA out of memory|torch\.(?:cuda\.)?OutOfMemoryError|CUDA error: out of memory|"
                           r"RESOURCE_EXHAUSTED: Out of memory|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|"
                           r"HIP out of memory|OOM when allocating tensor"),
     "GPU memory ran out — smaller batch, activation checkpointing / gradient accumulation, or a bigger GPU"),
    ("OOM-HOST", re.compile(r"(?i:oom[_-]kill)|OOM Killed|Out of memory: Kill(?:ed)? process|\bMemoryError\b|"
                            r"std::bad_alloc|Cannot allocate memory"),
     "host memory ran out (OOM killer) — request more --mem, or load less into RAM"),
    ("QUOTA", re.compile(r"Disk quota exceeded|No space left on device|\bEDQUOT\b|\bENOSPC\b"),
     "disk quota exceeded / disk full — clean up, or write checkpoints and caches to scratch"),
    ("IMPORT", re.compile(r"\bModuleNotFoundError\b|\bImportError\b|error while loading shared libraries"),
     "an import failed — wrong environment activated in the job, or a package missing from it"),
    ("SEGV", re.compile(r"Segmentation fault|\bSIGSEGV\b|\bsegfault\b"),
     "segmentation fault — a native crash (CUDA / driver / library mismatch, a bad extension build)"),
    ("NAN", re.compile(r"(?i:\bnan loss\b|\bloss\b[^\n]{0,24}?\b(?:nan|inf)\b|non-finite loss|NaN or Inf found|"
                       r"\bdetected (?:nan|inf)\b)"),
     "the loss went NaN / inf — lower the learning rate, clip gradients, check the data"),
    ("NCCL", re.compile(r"\bNCCL\b.*(?i:error|timeout|timed out|abort|fail)|nccl\w*Error|"
                        r"Watchdog caught collective operation timeout|\bDistBackendError\b"),
     "NCCL / distributed communication failed — often another rank crashed first (read its log), else a timeout"),
)
CAUSE_PATTERNS: dict[str, re.Pattern[str]] = {code: rx for code, rx, _ in CAUSE_RULES}
# Slurm's own last words, which also identify a cause when the state does not (a CANCELLED record of a time-out)
_TIMEOUT_LINE = re.compile(r"DUE TO TIME LIMIT")
_PREEMPT_LINE = re.compile(r"DUE TO PREEMPTION")
_NODE_FAIL_LINE = re.compile(r"DUE TO NODE FAIL")
CAUSE_PATTERNS.update(TIMEOUT=_TIMEOUT_LINE, PREEMPTED=_PREEMPT_LINE, NODE_FAIL=_NODE_FAIL_LINE)
TRIAGE_LINES = 2000                 # log lines job detail keeps for the cause chip (only text already fetched)
_SIGNAL_NOTES = {9: " — often the kernel OOM killer, or kill -9", 15: " — scancel, preemption or a time limit",
                 6: " — abort(): a failed assertion or a fatal library error", 7: " — bus error (bad memory access)"}


@dataclass(frozen=True, slots=True)
class Cause:
    """Why an ended job died, in one short code (the chip) plus one actionable line and, when the log shows it, the
    first log line that does."""
    code: str                       # "OOM-HOST" "OOM-GPU" "TIMEOUT" "PREEMPTED" "NODE_FAIL" "NCCL" "IMPORT" "QUOTA"
                                    # "SEGV" "NAN" "SIGNAL 9" "CANCELLED" "EXIT 1"
    text: str
    evidence: str | None = None

    @property
    def pattern(self) -> re.Pattern[str] | None:
        """The log pattern behind the code (log viewer: land on its first match)."""
        return CAUSE_PATTERNS.get(self.code)

    def plain(self) -> str:
        """``OOM-GPU: GPU memory ran out — … · log: torch.OutOfMemoryError: …`` (support bundle, export)."""
        return f"{self.code}: {self.text}" + (f" · log: {self.evidence}" if self.evidence else "")


def clean_line(line: str) -> str:
    """A log line as matched and shown: ANSI escapes removed, the last ``\\r`` segment (tqdm), stripped."""
    return _ANSI.sub("", line).rstrip("\r\n").rsplit("\r", 1)[-1].strip()


def line_level(line: str) -> Literal["error", "warn"] | None:
    """``error`` / ``warn`` for lines the log viewer highlights and jumps to with ``]`` / ``[``; None otherwise.
    Lines without any of the patterns' keywords (nearly all of a training log) skip the regexes."""
    low = line.lower()
    if not any(k in low for k in _LEVEL_KEYS):
        return None
    if ERROR_LINE.search(line) and not WARN_LINE.search(line):
        return "error"
    if WARN_LINE.search(line):
        return "warn"
    return None


def _first(lines: Sequence[str], rx: re.Pattern[str], limit: int) -> str | None:
    for ln in lines:
        if rx.search(ln):
            text = clean_line(ln)
            return text if len(text) <= limit else text[: limit - 1] + "…"
    return None


def _signal_name(n: int) -> str:
    try:
        return signal.Signals(n).name
    except ValueError:
        return f"signal {n}"


def _by_signal(n: int) -> Cause:
    if n == signal.SIGSEGV:
        return Cause("SEGV", f"killed by SIGSEGV (signal {n}) — a native crash")
    return Cause(f"SIGNAL {n}", f"killed by {_signal_name(n)} (signal {n}){_SIGNAL_NOTES.get(n, '')}")


def cause_of(job: Job, detail: JobDetail | None = None, lines: Iterable[str] = (), *,
             limit: int = 200) -> Cause | None:
    """The cause chip of an ended job that did not succeed (None for queued, running and completed jobs).

    Scheduler states decide first (OUT_OF_MEMORY, TIMEOUT, PREEMPTED, NODE_FAIL / BOOT_FAIL, DEADLINE), with the log
    line that shows them as evidence; a CANCELLED record whose log says ``DUE TO TIME LIMIT`` / ``PREEMPTION`` is that
    instead. Otherwise the log decides (CAUSE_RULES, most telling first; the evidence is the first matching line),
    then the exit status of the job or its steps (``rc >= 128`` = signal ``rc - 128``). ``lines`` is log text RunFleet
    already holds (detail stream, log peek) — this never fetches anything."""
    st = job.state
    if not st.is_terminal or st.group is StateGroup.DONE:
        return None
    text = [ln for ln in lines if ln]
    if st is JobState.OUT_OF_MEMORY:
        req = job.raw.get("ReqMem") or job.raw.get("MinMemory")
        what = f"exceeded --mem {req}" if req else "exceeded --mem"
        return Cause("OOM-HOST", f"{what}: Slurm's OOM killer ended it — request more --mem, or load less into RAM",
                     _first(text, CAUSE_PATTERNS["OOM-HOST"], limit))
    if st is JobState.TIMEOUT or st is JobState.DEADLINE or (
            st is JobState.CANCELLED and _first(text, _TIMEOUT_LINE, limit)):
        what = "reached its deadline" if st is JobState.DEADLINE else "hit the time limit"
        return Cause("TIMEOUT", f"{what} — request more --time, or checkpoint and resume",
                     _first(text, _TIMEOUT_LINE, limit))
    if st is JobState.PREEMPTED or (st is JobState.CANCELLED and _first(text, _PREEMPT_LINE, limit)):
        return Cause("PREEMPTED", "preempted for a higher-priority job — resubmit or requeue; checkpoint often",
                     _first(text, _PREEMPT_LINE, limit))
    if st in (JobState.NODE_FAIL, JobState.BOOT_FAIL) or (
            st is JobState.CANCELLED and _first(text, _NODE_FAIL_LINE, limit)):
        return Cause("NODE_FAIL", "the node failed, not your code — resubmit; --exclude the node if it repeats",
                     _first(text, _NODE_FAIL_LINE, limit))
    if st is JobState.CANCELLED:
        by = job.raw.get("cancelled_by_uid")
        return Cause("CANCELLED", "cancelled with scancel" + (f" (by uid {by})" if by else ""))
    for name, rx, advice in CAUSE_RULES:
        hit = _first(text, rx, limit)
        if hit is not None:
            return Cause(name, advice, hit)
    code, sig = job.exit_code, job.exit_signal
    if code is not None and code >= 128:
        return _by_signal(code - 128)
    if sig:
        return _by_signal(sig)
    for step in detail.steps if detail is not None else ():
        if step.step_id.endswith(".extern"):
            continue
        if step.exit_code is not None and step.exit_code >= 128:
            return _by_signal(step.exit_code - 128)
        if step.exit_signal and step.exit_signal != 125:          # 0:125 is Slurm's OOM marker, not a signal
            return _by_signal(step.exit_signal)
    if code:
        return Cause(f"EXIT {code}", f"the program exited with code {code} — read the end of the log")
    return None
