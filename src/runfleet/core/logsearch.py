"""[WP1] Bounded cross-job log search (§4.7.1): scope, the one `log.search` section per host, its parser, excerpts.

On demand only (palette › Search logs…, ``^f`` in the Jobs view), never from a timer. The scope is the user's own jobs
of one host that run or ended within ``general.recent_window`` and have a known log path: at most
:data:`MAX_FILES` distinct files (stdout and stderr, deduplicated, most recent job first). One probe reads the last
:data:`PER_FILE` bytes of each (``tail -c +K | head -c N``: the window is pinned to an absolute byte offset), at most
:data:`READ_BUDGET` bytes in all, through ``grep -a -n -b -o -E -m`` with a ±:data:`CONTEXT`-byte context window, at
most ``max_matches`` excerpts, under a hard timeout. Paths and the pattern are validated and quoted
(``util/shell.py``); nothing is globbed or walked.

A hit is anchored by its window start (absolute byte offset) and its line number inside that window (``grep -n``),
plus the excerpt text: the log viewer streams from the same offset (``tail -c +K -F``), so the line number is exact
however much the file grew since, and the excerpt confirms it (:class:`LogHit`).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AnyStr, Literal

from runfleet.models import Job, JobKey, StateGroup
from runfleet.probe.section import Section, SectionOutcome, SectionResult
from runfleet.util import shell

MAX_FILES = 20                 # distinct log files per host and search
PER_FILE = 256 * 1024          # the last 256 KiB of each file
READ_BUDGET = 1 << 20          # bytes read per probe, all files together
MAX_MATCHES = 200              # excerpts per search, all hosts together (split between the hosts searched)
CONTEXT = 60                   # bytes of context before and after a match (grep -o window)
EXCERPT_BYTES = 600            # one output line at most (cut -b), "LINE:OFFSET:" included
PATTERN_MAX = 200              # characters
REPEAT_MAX = 255               # largest {n,m} count of a regex (POSIX RE_DUP_MAX minimum)
EXPANDED_MAX = 1000            # positions after grep expands the counted repetitions: nested ones blow up its memory
TIMEOUT = 30.0                 # remote `timeout -k 2`; the local probe deadline is framing.local_timeout (35 s)
MAX_STDOUT = 256_000           # ≤ 200 excerpts × 600 bytes + ≤ 20 file lines, with room
SECTION = "log.search"

Stream = Literal["stdout", "stderr"]
FileStatus = Literal["searched", "missing", "denied", "not a file", "unreadable", "skipped (budget)",
                     "skipped (matches)", "not reached"]


@dataclass(frozen=True, slots=True)
class LogQuery:
    pattern: str
    regex: bool = False            # False: a literal string (the default)
    match_case: bool = False       # False: ASCII letters match either case (grep -i in the C locale)

    def describe(self) -> str:
        kind = "regex" if self.regex else "literal"
        return f"{kind}, {'match case' if self.match_case else 'ignore case'}"


@dataclass(frozen=True, slots=True)
class SearchFile:
    path: str
    key: JobKey                    # the most recent job in scope writing this file
    name: str                      # its job name
    stream: Stream
    merged: bool = False           # stderr goes to this (stdout) file too
    others: int = 0                # more jobs in scope writing the same file


@dataclass(frozen=True, slots=True)
class SearchPlan:
    host: str
    files: tuple[SearchFile, ...] = ()   # ≤ MAX_FILES, most recent job first
    jobs: int = 0                  # own jobs in scope (running, or ended within the window)
    no_path: int = 0               # of those, jobs without a known log path
    left_out: int = 0              # distinct files beyond MAX_FILES: not searched


@dataclass(frozen=True, slots=True)
class FileOutcome:
    file: SearchFile
    status: FileStatus
    size: int | None = None        # file size when it was searched
    start: int | None = None       # absolute offset of the searched window
    searched: int = 0              # bytes read
    hits: int = 0

    @property
    def window_cut(self) -> bool:
        """Only the last PER_FILE bytes were searched (the file is bigger)."""
        return self.size is not None and self.searched == PER_FILE and self.size > PER_FILE

    @property
    def budget_cut(self) -> bool:
        """The read budget ran out inside this file: less than its window was read."""
        return self.size is not None and self.searched < min(self.size, PER_FILE)


@dataclass(frozen=True, slots=True)
class LogHit:
    host: str
    file: SearchFile
    start: int                     # absolute offset of the searched window: the viewer streams from here
    line: int                      # 1-based line in the window (grep -n): the anchor, exact even if the file grew
    offset: int                    # absolute offset of the excerpt (window start + grep -b)
    excerpt: str                   # ≤ CONTEXT bytes + match + ≤ CONTEXT bytes of one line (may hold ANSI and \r)
    needle: str                    # the \r segment of the excerpt with the match: what the viewer looks for
    cut: bool = False              # the output line hit the EXCERPT_BYTES cap (the excerpt ends early)
    size: int | None = None        # the file's size when it was searched

    @property
    def key(self) -> JobKey:
        return self.file.key


@dataclass(frozen=True, slots=True)
class LogSearchResult:
    host: str
    query: LogQuery
    plan: SearchPlan
    hits: tuple[LogHit, ...] = ()
    files: tuple[FileOutcome, ...] = ()
    max_matches: int = MAX_MATCHES
    error: str | None = None       # not searched or cut short: transport error, timeout, a pattern grep refused
    timed_out: bool = False

    @property
    def searched(self) -> list[FileOutcome]:
        return [f for f in self.files if f.status == "searched"]

    @property
    def bytes_read(self) -> int:
        return sum(f.searched for f in self.files)

    @property
    def match_capped(self) -> bool:
        return len(self.hits) >= self.max_matches or any(f.status == "skipped (matches)" for f in self.files)

    @property
    def problems(self) -> list[FileOutcome]:
        return [f for f in self.files if f.status in ("missing", "denied", "not a file", "unreadable")]


# ============================================================================================ scope
def in_scope(job: Job, now: datetime, window_s: float) -> bool:
    """Own jobs that run now or ended within the recent window (pending jobs have no log yet)."""
    if not job.is_mine:
        return False
    group = job.state.group
    if group is StateGroup.ACTIVE:
        return True
    if not job.state.is_terminal:
        return False
    end = job.end_time or job.observed_at
    return end is not None and now - end <= timedelta(seconds=window_s)


def _recency(job: Job) -> tuple[int, float]:
    """Sort key, most recent first: running jobs (latest start first), then ended ones (latest end first)."""
    if job.state.group is StateGroup.ACTIVE:
        t = job.start_time or job.submit_time
        return (0, -(t.timestamp() if t else 0.0))
    t = job.end_time or job.observed_at
    return (1, -(t.timestamp() if t else 0.0))


def plan(host: str, jobs: Iterable[Job], now: datetime, window_s: float, *, max_files: int = MAX_FILES) -> SearchPlan:
    """Which files one host's search reads: distinct valid paths of the jobs in scope, most recent job first."""
    picked = sorted((j for j in jobs if j.key.host == host and in_scope(j, now, window_s)), key=_recency)
    files: dict[str, SearchFile] = {}
    no_path = 0
    for job in picked:
        refs = [r for r in job.logs if r.path]
        if not refs:
            no_path += 1
            continue
        for ref in sorted(refs, key=lambda r: r.stream != "stdout"):
            path = ref.path or ""
            try:
                shell.path(path)
            except ValueError:
                continue
            seen = files.get(path)
            if seen is None:
                merged = ref.stream == "stdout" and any(r.merged for r in job.logs if r.stream == "stderr")
                files[path] = SearchFile(path, job.key, job.name, ref.stream, merged)
            elif seen.key != job.key:
                files[path] = SearchFile(seen.path, seen.key, seen.name, seen.stream, seen.merged, seen.others + 1)
    ordered = list(files.values())
    return SearchPlan(host, tuple(ordered[:max_files]), len(picked), no_path, max(0, len(ordered) - max_files))


# ============================================================================================ pattern
_ERE_SPECIAL = set(".[]()*+?{}|$\\")


def pattern_problem(query: LogQuery) -> str | None:
    """Why this pattern cannot be searched (None: it can; grep may still refuse a regex: `@@bad`)."""
    p = query.pattern
    if not p:
        return "type a pattern"
    if len(p) > PATTERN_MAX:
        return f"at most {PATTERN_MAX} characters"
    if any(c in p for c in "\n\r\0"):
        return "one line only"
    if query.regex and re.search(r"\\[1-9]", p):
        return "back-references (\\1 …) are not supported"
    if query.regex and not _balanced(p):
        return "unbalanced parentheses"
    if query.regex:
        biggest, size = _repetition(p)
        if biggest > REPEAT_MAX:
            return f"repetition counts {{n,m}} above {REPEAT_MAX} are not supported"
        if size > EXPANDED_MAX:
            return f"repetitions too large: grep would expand them past {EXPANDED_MAX} positions (nested {{n,m}})"
    return None


def _bracket_end(p: str, i: int) -> int:
    """Index just past the bracket expression that starts at ``p[i] == "["`` (or the end of ``p``)."""
    n, j = len(p), i + 1
    if j < n and p[j] == "^":
        j += 1
    if j < n and p[j] == "]":
        j += 1
    while j < n and p[j] != "]":
        j = p.index(":]", j) + 2 if p.startswith("[:", j) and ":]" in p[j:] else j + 1
    return j + 1


_COUNT = re.compile(r"\{(\d*)(,?)(\d*)\}")


def _repetition(p: str) -> tuple[int, int]:
    """``(largest count, expanded size)`` of a (balanced) ERE: grep compiles ``x{n,m}`` as ``m`` copies of ``x``
    (``x{n,}``: ``n + 1``), so nested counts multiply — ``((a{1,255}){1,255}){1,255}`` exhausts gigabytes of
    memory on the login node before reading a byte. Alternatives add up (an upper bound)."""
    biggest = 0
    stack: list[list[int]] = [[0, 0]]          # per open group: [size so far, size of the last atom]
    i, n = 0, len(p)
    while i < n:
        c, atom = p[i], 1
        if c == "{" and (m := _COUNT.match(p, i)) is not None and (m.group(1) or m.group(3)):
            lo, comma, hi = int(m.group(1) or 0), m.group(2), m.group(3)
            count = int(hi) if hi else (lo + 1 if comma else lo)
            biggest = max(biggest, lo, int(hi or 0))
            frame = stack[-1]
            frame[0] += frame[1] * (max(count, 1) - 1)
            frame[1] *= max(count, 1)
            i = m.end()
            continue
        if c == "\\":
            i += 2
        elif c == "[":
            i = _bracket_end(p, i)
        elif c == "(":
            stack.append([0, 0])
            i += 1
            continue
        elif c == ")" and len(stack) > 1:
            atom = stack.pop()[0]
            i += 1
        elif c in "*+?|^$":
            stack[-1][1] = 0 if c == "|" else stack[-1][1]
            i += 1
            continue
        else:
            i += 1
        stack[-1][0] += atom
        stack[-1][1] = atom
    return biggest, sum(f[0] for f in stack)


def _balanced(p: str) -> bool:
    """Parentheses of an ERE pair up (escaped ones and those in bracket expressions do not count): the pattern is
    put in a group of its own, so a stray ``)`` would close that group instead of failing."""
    depth, i, n = 0, 0, len(p)
    while i < n:
        c = p[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i = _bracket_end(p, i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def literal_ere(text: str) -> str:
    """A literal string as a POSIX ERE: special characters in brackets (no "stray \\" warnings), ``^`` escaped."""
    out = []
    for c in text:
        if c == "^":
            out.append("\\^")
        elif c in _ERE_SPECIAL:
            out.append(f"[{c}]")
        else:
            out.append(c)
    return "".join(out)


def grep_pattern(query: LogQuery) -> str:
    """The ERE grep runs: the pattern in a context window, so `-o` prints an excerpt around each match."""
    core = query.pattern if query.regex else literal_ere(query.pattern)
    return f".{{0,{CONTEXT}}}({core}).{{0,{CONTEXT}}}"


_POSIX_CLASSES = {"alpha": "A-Za-z", "digit": "0-9", "alnum": "A-Za-z0-9", "upper": "A-Z", "lower": "a-z",
                  "space": r" \t\r\n\v\f", "blank": r" \t", "xdigit": "0-9A-Fa-f", "punct": r"!-/:-@\[-`{-~",
                  "word": "A-Za-z0-9_"}


def python_regex(query: LogQuery) -> re.Pattern[str] | None:
    """The pattern for Python (excerpt highlighting, the demo's grep): literal escaped, POSIX classes mapped, case
    folding as asked. None when Python cannot read the ERE (the excerpt is then shown without highlighting)."""
    if query.regex:
        src = re.sub(r"\[:(\w+):\]", lambda m: _POSIX_CLASSES.get(m.group(1), m.group(0)), query.pattern)
    else:
        src = re.escape(query.pattern)
    try:
        return _compile(src, 0 if query.match_case else re.IGNORECASE)
    except re.error:
        return None


def _compile(src: AnyStr, flags: int) -> re.Pattern[AnyStr]:
    """re.compile without the FutureWarning ("Possible nested set") a POSIX `[[]` draws: a warning on stderr would
    scribble over the running TUI."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return re.compile(src, flags)


def ere_bytes(ere: str, *, icase: bool) -> re.Pattern[bytes] | str:
    """A grep ERE (as :func:`grep_pattern` builds it) for Python on bytes, the way the demo plays grep: POSIX classes
    mapped, ASCII case folding like grep in the C locale. A message instead when Python cannot read it."""
    src = re.sub(r"\[:(\w+):\]", lambda m: _POSIX_CLASSES.get(m.group(1), m.group(0)), ere)
    try:
        return _compile(src.encode("utf-8"), re.IGNORECASE if icase else 0)
    except re.error as e:
        return str(e)


def match_spans(text: str, query: LogQuery) -> list[tuple[int, int]]:
    """Non-empty, non-overlapping matches of the query in ``text`` (character offsets)."""
    rx = python_regex(query)
    if rx is None:
        return []
    return [(m.start(), m.end()) for m in rx.finditer(text) if m.end() > m.start()]


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-Z\\-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def needle_of(excerpt: str, query: LogQuery) -> str:
    """The ``\\r`` segment of the excerpt that holds the (first) match, without the replacement characters a byte
    cut left at its ends: that text is on the line as the log viewer shows it (``\\r`` restarts a line)."""
    spans = match_spans(excerpt, query)
    at = spans[0][0] if spans else len(excerpt)
    begin = excerpt.rfind("\r", 0, at) + 1
    end = excerpt.find("\r", at)
    segment = excerpt[begin:end if end >= 0 else len(excerpt)]
    return segment.strip("�")


def display_excerpt(hit: LogHit, query: LogQuery) -> tuple[str, list[tuple[int, int]], bool, bool]:
    """``(text, match spans, cut before, cut after)`` for one row: the needle without ANSI codes, tabs as spaces,
    other control characters as ``·``; "cut" = the context window (or the byte cap) ended there, not the line."""
    text = _ANSI.sub("", hit.needle).replace("\t", " ")
    text = _CONTROL.sub("·", text)
    spans = match_spans(text, query)
    if not spans:
        return text, [], False, False
    before = len(text[:spans[0][0]].encode("utf-8")) >= CONTEXT
    after = len(text[spans[-1][1]:].encode("utf-8")) >= CONTEXT or hit.cut
    return text, spans, before, after


# ============================================================================================ command
def search_command(paths: Sequence[str], query: LogQuery, *, max_matches: int = MAX_MATCHES,
                   per_file: int = PER_FILE, budget: int = READ_BUDGET) -> str:
    """`log.search` (Appendix A): one POSIX sh snippet over the files, in order; output, one line each:
    ``@@bad <grep message>`` (grep refused the regex; nothing else follows), ``@@file I SIZE START N`` then the
    ``LINE:OFFSET:EXCERPT`` lines of file I, ``@@err I missing|notfile|denied|stat``, ``@@skip I budget|matches``,
    and ``@@end LEFT BUDGET`` last. Every path is validated (absolute, one line) and quoted; so is the pattern."""
    files = shell.quote_join(shell.path(p) for p in paths)
    icase = "" if query.match_case else " -i"
    per, cap, m = int(per_file), int(budget), max(1, int(max_matches))
    return "\n".join((
        f"p={shell.quote(grep_pattern(query))}; left={m}; budget={cap}; i=0",
        "printf 'x\\n' | grep -E -e \"$p\" >/dev/null 2>&1",
        "[ $? -lt 2 ] || { printf '@@bad %s\\n' \"$(printf 'x\\n' | grep -E -e \"$p\" 2>&1 >/dev/null | head -n 1 "
        "| cut -b 1-200)\"; exit 0; }",
        f"for f in {files}; do",
        "  i=$((i + 1))",
        "  if [ \"$left\" -le 0 ]; then echo \"@@skip $i matches\"; continue; fi",
        "  if [ \"$budget\" -le 0 ]; then echo \"@@skip $i budget\"; continue; fi",
        "  if [ ! -e \"$f\" ]; then echo \"@@err $i missing\"; continue; fi",
        "  if [ ! -f \"$f\" ]; then echo \"@@err $i notfile\"; continue; fi",
        "  if [ ! -r \"$f\" ]; then echo \"@@err $i denied\"; continue; fi",
        "  s=$(stat -L -c %s -- \"$f\" 2>/dev/null)",
        "  case $s in ''|*[!0-9]*) echo \"@@err $i stat\"; continue;; esac",
        f"  n=$s; [ \"$n\" -gt {per} ] && n={per}; [ \"$n\" -gt \"$budget\" ] && n=$budget",
        "  b=$((s - n)); budget=$((budget - n))",
        "  echo \"@@file $i $s $b $n\"",
        "  [ \"$n\" -gt 0 ] || continue",
        f"  r=$(tail -c +$((b + 1)) -- \"$f\" | head -c \"$n\" | grep -a -n -b -o{icase} -E -m \"$left\" -e \"$p\" "
        f"| head -n \"$left\" | cut -b 1-{EXCERPT_BYTES})",
        "  [ -z \"$r\" ] || { printf '%s\\n' \"$r\"; left=$((left - $(printf '%s\\n' \"$r\" | wc -l))); }",
        "done",
        "echo \"@@end $left $budget\"",
    ))


def search_section(paths: Sequence[str], query: LogQuery, *, max_matches: int = MAX_MATCHES) -> Section:
    """The one section of a host's search probe (purpose ``log``): `meta` carries the arguments for Demo / Fake."""
    m = max(1, int(max_matches))
    return Section(SECTION, cmd=search_command(paths, query, max_matches=m), timeout=TIMEOUT,
                   requires=("grep", "tail", "head", "stat"), max_stdout=MAX_STDOUT,
                   meta={"paths": list(paths), "pattern": grep_pattern(query), "icase": not query.match_case,
                         "regex": query.regex, "max_matches": m, "per_file": PER_FILE, "budget": READ_BUDGET})


# ============================================================================================ parse
_HIT = re.compile(r"^(\d+):(\d+):(.*)$", re.S)
_ERR_STATUS: dict[str, FileStatus] = {"missing": "missing", "notfile": "not a file", "denied": "denied",
                                      "stat": "unreadable"}


@dataclass
class _Open:
    index: int
    size: int
    start: int
    n: int
    hits: list[LogHit] = field(default_factory=list)


def parse(host: str, query: LogQuery, search: SearchPlan, result: SectionResult, *,
          max_matches: int = MAX_MATCHES) -> LogSearchResult:
    """`log.search` output -> hits and per-file outcomes. Partial output (timeout, a dropped connection) keeps what
    arrived; files the script never reached are "not reached"."""
    files = search.files
    outcomes: dict[int, FileOutcome] = {}
    hits: list[LogHit] = []
    cur: _Open | None = None
    bad: str | None = None
    ended = False

    def close() -> None:
        nonlocal cur
        if cur is not None and 1 <= cur.index <= len(files):
            outcomes[cur.index] = FileOutcome(files[cur.index - 1], "searched", cur.size, cur.start, cur.n,
                                              len(cur.hits))
            hits.extend(cur.hits)
        cur = None

    for line in result.stdout.split("\n"):
        if line.startswith("@@"):
            word, _, rest = line[2:].partition(" ")
            parts = rest.split()
            if word == "bad":
                bad = rest.strip() or "grep refused the pattern"
                break
            close()
            if word == "end":
                ended = True
                break
            if not parts or not parts[0].isdigit():
                continue
            i = int(parts[0])
            if not 1 <= i <= len(files):
                continue
            if word == "file" and len(parts) == 4 and all(p.isdigit() for p in parts[1:]):
                cur = _Open(i, int(parts[1]), int(parts[2]), int(parts[3]))
            elif word == "err" and len(parts) == 2:
                outcomes[i] = FileOutcome(files[i - 1], _ERR_STATUS.get(parts[1], "unreadable"))
            elif word == "skip" and len(parts) == 2:
                outcomes[i] = FileOutcome(files[i - 1], "skipped (matches)" if parts[1] == "matches"
                                          else "skipped (budget)")
            continue
        m = _HIT.match(line)
        if m is None or cur is None or len(hits) + len(cur.hits) >= max_matches:
            continue
        excerpt = m.group(3)
        cut = len(line.encode("utf-8", "replace")) >= EXCERPT_BYTES
        cur.hits.append(LogHit(host, files[cur.index - 1], cur.start, int(m.group(1)), cur.start + int(m.group(2)),
                               excerpt, needle_of(excerpt, query), cut, cur.size))
    if not ended:
        close()
    out = tuple(outcomes.get(i, FileOutcome(f, "not reached")) for i, f in enumerate(files, start=1))
    timed_out = result.outcome is SectionOutcome.TIMEOUT
    error = None
    if bad is not None:
        error = f"grep refused the pattern: {bad}" if not bad.startswith("grep") else bad
        out = tuple(FileOutcome(f, "not reached") for f in files)
        hits = []
    elif timed_out:
        error = f"timed out after {int(TIMEOUT)} s: the files below were not searched or only in part"
    elif not ended:
        error = _section_error(result)
    return LogSearchResult(host, query, search, tuple(hits), out, max_matches, error, timed_out)


def _section_error(result: SectionResult) -> str:
    if result.outcome is SectionOutcome.MISSING:
        return f"cannot search here: {result.stderr.removeprefix('missing:') or 'a tool'} is missing"
    first = next((ln.strip() for ln in result.stderr.splitlines() if ln.strip()), "")
    return (first or f"search ended early ({result.outcome.value})")[:120]
