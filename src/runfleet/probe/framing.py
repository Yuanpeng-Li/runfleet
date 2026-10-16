"""[WP1] Render ProbeScript -> POSIX sh and parse framed stdout -> SectionResults (§4.3).

The rendered script is one `sh -s` program: a prelude (Slurm env scrub, epoch time format, private temp dir, `rf`
runner) followed by one `rf NAME TIMEOUT REQUIRES CMD` line per section, late sections last. Each section's output is
framed by `@@RF<nonce> B|E|X|Y <name>` marker lines, so the parser can recover every section from one stdout stream.
"""

from __future__ import annotations

import logging
import re
import secrets
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from runfleet.probe.section import ProbeScript, Section, SectionOutcome, SectionResult

log = logging.getLogger("runfleet.probe.framing")

SECTION_NAME_RE = re.compile(r"^[a-z]+\.[a-z0-9_.]+$")
NONCE_RE = re.compile(r"^[0-9a-f]{12}$")
STDERR_LIMIT = 4096

_PRELUDE = r"""# runfleet probe {nonce} v2
set +e +u 2>/dev/null
export LC_ALL=C
for rf_v in $(env | awk -F= '/^(SQUEUE|SINFO|SACCT|SCONTROL|SSTAT)_|^SLURM_(TIME_FORMAT|CLUSTERS)=/ {{print $1}}'); do
  unset "$rf_v"; done
export SLURM_TIME_FORMAT=%s
RF=@@RF{nonce}
RF_T=$(mktemp -d "${{TMPDIR:-/tmp}}/rf.XXXXXX" 2>/dev/null) || RF_T=
export RF_T
rf_cleanup() {{ [ -n "$RF_T" ] && rm -rf "$RF_T"; }}
trap rf_cleanup EXIT; trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
if command -v timeout >/dev/null 2>&1; then RF_TO=1; else RF_TO=; fi
# uutils date 0.8 ignores %3N and can omit its leading zeroes. Request all
# nine nanosecond digits explicitly, then convert to milliseconds ourselves.
rf_ms() {{
  rf_d=$(date +%s%9N 2>/dev/null)
  case $rf_d in *[!0-9]*|'') echo 0 ;; *) echo "$((rf_d / 1000000))" ;; esac
}}
rf() {{
  rf_n=$1 rf_s=$2 rf_q=$3; shift 3
  printf '\n%s B %s\n' "$RF" "$rf_n"
  for rf_t in $rf_q; do
    command -v "$rf_t" >/dev/null 2>&1 || {{ printf '\n%s E %s 127 0 missing:%s\n' "$RF" "$rf_n" "$rf_t"; return 0; }}
  done
  rf_0=$(rf_ms)
  # stderr goes before the end marker, so a connection cut in between leaves the section INCOMPLETE, never a
  # falsely authoritative EMPTY; without a temp dir (full /tmp) it is captured in a variable instead of lost
  if [ -n "$RF_T" ]; then
    rf_c "$1" 2>"$RF_T/e"; rf_rc=$?; rf_1=$(rf_ms)
    if [ -s "$RF_T/e" ]; then
      printf '\n%s X %s\n' "$RF" "$rf_n"; head -c 4096 "$RF_T/e"; printf '\n%s Y %s\n' "$RF" "$rf_n"
    fi
  else
    {{ rf_err=$(rf_c "$1" 2>&1 1>&3 3>&-); rf_rc=$?; }} 3>&1
    rf_1=$(rf_ms)
    if [ -n "$rf_err" ]; then
      printf '\n%s X %s\n' "$RF" "$rf_n"; printf '%s' "$rf_err" | head -c 4096; printf '\n%s Y %s\n' "$RF" "$rf_n"
    fi
  fi
  printf '\n%s E %s %s %s\n' "$RF" "$rf_n" "$rf_rc" "$(( ${{rf_1:-0}} - ${{rf_0:-0}} ))"
  return 0
}}
rf_c() {{
  if [ -n "$RF_TO" ]; then timeout -k 2 "$rf_s" sh -c "$1" </dev/null; else sh -c "$1" </dev/null; fi
}}
"""


def new_nonce() -> str:
    """12 hex chars (`secrets.token_hex(6)`)."""
    return secrets.token_hex(6)


def make_script(sections: Sequence[Section], purpose: str = "poll") -> ProbeScript:
    """A ProbeScript with a fresh nonce."""
    return ProbeScript(sections=tuple(sections), nonce=new_nonce(), purpose=purpose)  # type: ignore[arg-type]


def ordered(script: ProbeScript) -> list[Section]:
    """Sections in execution order: non-late in the given order, then late ones."""
    return [s for s in script.sections if not s.late] + [s for s in script.sections if s.late]


def validate(script: ProbeScript) -> None:
    """Raise ValueError for a bad nonce, a bad or duplicate section name, or an unsafe `requires` entry."""
    if not NONCE_RE.match(script.nonce):
        raise ValueError(f"bad nonce {script.nonce!r}")
    seen: set[str] = set()
    for s in script.sections:
        if not SECTION_NAME_RE.match(s.name):
            raise ValueError(f"bad section name {s.name!r}")
        if s.name in seen:
            raise ValueError(f"duplicate section {s.name!r}")
        seen.add(s.name)
        for tool in s.requires:
            if not re.match(r"^[A-Za-z0-9_.+\-]+$", tool):
                raise ValueError(f"bad required tool {tool!r} in {s.name}")
        if not s.timeout > 0:
            raise ValueError(f"bad timeout {s.timeout!r} in {s.name}")


def _fmt_timeout(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else f"{t:g}"


_MKTEMP = 'RF_T=$(mktemp -d "${TMPDIR:-/tmp}/rf.XXXXXX" 2>/dev/null) || RF_T=\n'


def render(script: ProbeScript, *, tmp: bool | None = None) -> str:
    """The POSIX sh program for `script` (fed to `sh -s` on stdin).

    `tmp=False` renders a script that never creates a temp dir (`RF_T=` empty: stderr is captured in a variable and
    `nvidia.wrap` runs its payload in the foreground). The default (`None`) is False for purpose "attach" — an Attach
    sample writes nothing on any server: neither its login probe (§7.16 layer 1) nor the nested node script (layer 3)
    — and True for every other purpose."""
    validate(script)
    if tmp is None:
        tmp = script.purpose != "attach"
    prelude = _PRELUDE.format(nonce=script.nonce)
    if not tmp:
        assert _MKTEMP in prelude
        prelude = prelude.replace(_MKTEMP, "RF_T=\n")
    lines = [prelude]
    for s in ordered(script):
        lines.append(f"rf {shlex.quote(s.name)} {_fmt_timeout(s.timeout)} {shlex.quote(' '.join(s.requires))} "
                     f"{shlex.quote(s.cmd)}\n")
    lines.append("exit 0\n")
    return "".join(lines)


def local_timeout(script: ProbeScript) -> float:
    """§4.2: min(90, max(20, Σ non-late timeouts) + Σ late timeouts + 5) seconds.

    A confirmed action is not capped: its sections (a terminate grace of up to 10 min, several targets in turn) must
    run to their end, so the budget is every section's remote bound (`timeout -k 2`) in turn, plus a margin."""
    if script.purpose == "action":
        return max(20.0, sum(s.timeout + 2.0 for s in script.sections) + 5.0)
    early = sum(s.timeout for s in script.sections if not s.late)
    late = sum(s.timeout for s in script.sections if s.late)
    return min(90.0, max(20.0, early) + late + 5.0)


# ---------------------------------------------------------------- parsing
@dataclass(frozen=True, slots=True)
class ParsedProbe:
    sections: Mapping[str, SectionResult]   # every section of the script
    ended: int                              # number of end markers seen (partial rule, §4.2)
    markers: int                            # number of marker lines of any kind


@dataclass
class _Acc:
    out: list[str]
    err: list[str] | None = None
    rc: int | None = None
    ms: int | None = None
    ended: bool = False
    missing: bool = False
    in_err: bool = False


def outcome_for(rc: int | None, stdout: str, stderr: str, *, missing: bool = False) -> SectionOutcome:
    """Outcome rule of §4.3 for a section that has an end marker."""
    if missing:
        return SectionOutcome.MISSING
    if rc in (124, 137):
        return SectionOutcome.TIMEOUT
    if rc == 0:
        if stdout:
            return SectionOutcome.OK
        return SectionOutcome.STDERR_ONLY if stderr else SectionOutcome.EMPTY
    return SectionOutcome.FAILED


def parse(stdout: bytes | str, script: ProbeScript, *, timed_out: bool = False) -> ParsedProbe:
    """Recover every section of `script` from framed stdout.

    Unfinished sections: begin marker without end marker -> TIMEOUT if we hit the local timeout, else INCOMPLETE; no
    marker at all -> NOT_RUN. Text outside B...E blocks (login banners) is ignored.
    """
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
    prefix = f"@@RF{script.nonce} "
    wanted = {s.name: s for s in script.sections}
    acc: dict[str, _Acc] = {}
    current: _Acc | None = None        # section whose stdout we are collecting
    markers = ended = 0
    stray = 0
    for line in text.split("\n"):
        if line.startswith(prefix):
            parts = line[len(prefix):].split(" ")
            tag, name = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if name in wanted and tag in ("B", "E", "X", "Y"):
                markers += 1
                if tag == "B":
                    current = acc[name] = _Acc(out=[])
                    continue
                a = acc.get(name)
                if a is None:
                    continue
                if tag == "E":
                    a.ended, current = True, None
                    ended += 1
                    try:
                        a.rc = int(parts[2]) if len(parts) > 2 else None
                        a.ms = int(parts[3]) if len(parts) > 3 else None
                    except ValueError:
                        a.rc = a.ms = None
                    a.missing = len(parts) > 4 and parts[4].startswith("missing:") and a.rc == 127
                    if a.missing:
                        a.err = [parts[4]]
                elif tag == "X":
                    a.err, a.in_err = [], True
                    current = a
                elif tag == "Y":
                    a.in_err, current = False, None
                continue
        if current is None:
            if line:
                stray += 1
            continue
        if current.in_err:
            assert current.err is not None
            current.err.append(line)
        else:
            current.out.append(line)
    if stray:
        log.debug("probe %s: %d stray line(s) outside sections ignored", script.nonce, stray)

    results: dict[str, SectionResult] = {}
    for name, sec in wanted.items():
        a = acc.get(name)
        if a is None:
            results[name] = SectionResult(name, SectionOutcome.NOT_RUN, None, "", "", None)
            continue
        # The framing prints "\n" before the E/Y marker; splitting on "\n" already consumed exactly that one, so the
        # joined lines are the command's exact output (§3.4/§4.3: "a\nb" and "a\nb\n" both round-trip).
        stdout_text = "\n".join(a.out)
        stderr_text = "" if a.err is None else "\n".join(a.err)
        if a.missing:
            stdout_text = ""
        truncated = len(stdout_text) > sec.max_stdout
        if truncated:                  # whole lines only; and never a complete (authoritative) answer, see below
            cut = stdout_text.rfind("\n", 0, sec.max_stdout)
            stdout_text = stdout_text[: cut + 1 if cut >= 0 else sec.max_stdout]
            stderr_text = f"runfleet: output truncated at {sec.max_stdout} characters\n{stderr_text}"
        stderr_text = stderr_text[:STDERR_LIMIT]
        if not a.ended:
            outcome = SectionOutcome.TIMEOUT if timed_out else SectionOutcome.INCOMPLETE
            results[name] = SectionResult(name, outcome, None, stdout_text, stderr_text, None)
            continue
        outcome = outcome_for(a.rc, stdout_text, stderr_text, missing=a.missing)
        if truncated and outcome is SectionOutcome.OK:
            outcome = SectionOutcome.FAILED      # rows still parse (FAILED with output), but nothing counts as vanished
        results[name] = SectionResult(name, outcome, a.rc, stdout_text, stderr_text, a.ms)
    return ParsedProbe(sections=results, ended=ended, markers=markers)
