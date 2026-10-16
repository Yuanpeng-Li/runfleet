"""[WP1] The session agent (§4.8): its POSIX sh text, request encoding and an incremental reply decoder. Pure.

The session transport runs ONE `ssh -T <alias> <bootstrap>` per host. The bootstrap reads exactly the agent's bytes
from stdin (`dd bs=1`, so nothing past them is consumed) and evals them; the agent then serves requests on the same
stdin until EOF. Nothing is installed: the agent lives in the remote shell's memory, keeps a few small state files in
a private `mktemp -d` directory and removes it, and ends every job it started, on stdin EOF, exit or a signal.

Wire protocol (one line per message, `\\n`-terminated, ASCII; job ids and tokens match `[A-Za-z0-9_]+`):

client -> agent
  `P <token>`            ping; answered `P <token>`
  `Q <id> <payload>`     probe: payload (a rendered ProbeScript) is fed to `sh -s`; stdout comes back line-framed
  `S <id> <payload>`     stream: payload is a `sh -c` command whose stdin stays open until `W`; stdout comes back
                         as raw bytes (hex), one line per read(2), so `\\r` progress lines arrive at once
  `W <id>`               close the stream's stdin (its watchdog then ends it, §4.7)
  `K <id>`               kill the job's command (SIGTERM, SIGKILL 2 s later); the job still ends with `Z`. A job
                         still running 1 s after the SIGKILL (a child in D state or one that escaped, keeping its
                         stdout open) stops counting toward MAX_JOBS; its `Z` may come late or never

  A payload is the bytes as a printf(1) format: printable ASCII as is, `%` as `%%`, every other byte (space, `\\`,
  newline, non-ASCII) as a 3-digit octal escape (`\\040`); a leading `-` is escaped too. It needs no `base64`.

agent -> client
  `R <pid> <version> <setsid|-> <pipe_buf>`   ready (after the temp dir exists)
  `P <token>`            pong
  `L <id> <text>`        a stdout line of a probe (the newline is implied)
  `C <id> <text>`        a piece of a longer stdout line (no newline)
  `D <id> <hex>`         stdout bytes of a stream
  `E <id> <text>`        a stderr line (sent after the job's stdout ended; last 8 KiB)
  `F <id> <text>`        a piece of a longer stderr line
  `Z <id> <rc>`          the job ended (rc 125: refused, the agent already runs MAX_JOBS jobs)
  `X <text>`             the agent cannot run (no temp dir); it exits

Every agent write is a single `write(2)` shorter than PIPE_BUF (4 KiB on Linux, 512 B where `getconf` says so), so
lines from concurrent jobs never interleave (awk prints each line with ONE `%s` conversion: mawk's unbuffered mode
writes every conversion of a format separately). Probe lines leave as soon as the job prints them, also under mawk
(Debian/Ubuntu's awk), which otherwise reads a pipe in blocks until EOF: the agent runs it as `awk -W interactive`,
so the finished sections of a probe that later times out are not lost. Line mode cannot tell whether a probe's last
line had a newline; framed probe output always ends with one, so the framing parser (§4.3) sees exactly the bytes
`sh -s` printed. NUL bytes are not carried faithfully in line mode (mawk keeps them, BusyBox awk splits the line
there); probes print text. The ready line starts with a newline, so login-shell output without one (a terminal
title escape from `.bashrc`) cannot glue onto it.

Remote requirements: a POSIX `sh` (dash, bash, BusyBox ash, ksh), `dd`, `awk` with `fflush` (every awk since 2012:
gawk, mawk, BusyBox, BSD/macOS), `od`, `tr`, `tail`, `sleep`, `mktemp` (or a writable `$TMPDIR`//tmp). Optional:
`setsid` (util-linux; each job then runs in a process group of its own) and `ps -A -o pid= -o ppid=` (kills reach a
job's descendants: GNU `timeout` puts sections in groups of their own). Without either (rare: macOS has `ps`), a kill
reaches only the job's shell and the job's processes end at their own remote timeouts. No `mkfifo` is needed: a
stream's stdin is a pipe from a small holder loop that ends on `W`, at agent exit, or when the agent is gone.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

VERSION = 1
MAX_JOBS = 8                    # the agent refuses more concurrent jobs (rc 125); the client keeps below it too
STDERR_TAIL = 8192              # bytes of a job's stderr sent back (E/F lines)
LINE_LIMIT = 1 << 20            # a partial reply line longer than this is noise (a runaway login banner)
ID_RE = re.compile(r"^[A-Za-z0-9_]+$")

AGENT_SH = r"""# runfleet session agent v1 (DESIGN 4.8): eval'd once by `sh -c`; installs nothing
umask 077
RFA_PID=$$
RFA_SS=; command -v setsid >/dev/null 2>&1 && RFA_SS=setsid
RFA_PS=
if ps -A -o pid= -o ppid= >/dev/null 2>&1; then RFA_PS='ps -A -o pid= -o ppid='
elif ps -o pid= -o ppid= >/dev/null 2>&1; then RFA_PS='ps -o pid= -o ppid='; fi
RFA_PB=$(getconf PIPE_BUF / 2>/dev/null); case $RFA_PB in ''|*[!0-9]*) RFA_PB=512 ;; esac
RFA_W=$((RFA_PB - 128)); [ "$RFA_W" -gt 3968 ] && RFA_W=3968; [ "$RFA_W" -lt 384 ] && RFA_W=384
RFA_B=$((RFA_W / 2))
RFA_T=1; sleep 0.01 2>/dev/null && RFA_T=0.2
RFA_AWK=awk; case $(awk -W version </dev/null 2>&1) in *mawk*) RFA_AWK='awk -W interactive' ;; esac
RA=
for rfa_d in "${TMPDIR:-/tmp}" /tmp; do
  RA=$(mktemp -d "$rfa_d/rfa.XXXXXX" 2>/dev/null) && break
  RA="$rfa_d/rfa.$$.$(date +%s 2>/dev/null)"; mkdir "$RA" 2>/dev/null && break; RA=
done
[ -n "$RA" ] || { printf 'X cannot create a private temp dir\n'; exit 97; }
rfa_tree() {
  [ -n "$RFA_PS" ] || return 0
  $RFA_PS 2>/dev/null | LC_ALL=C awk -v r="$1" '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { pp[$1] = $2 }
    END { w[r] = 1; n = 1; while (n) { n = 0; for (p in pp) if (!(p in w) && (pp[p] in w)) { w[p] = 1; n = 1 } }
          for (p in w) if (p != r) print p }'
}
rfa_victims() {
  rfa_k="$RA/$1"
  [ -f "$rfa_k.job" ] && [ -s "$rfa_k.pg" ] || return 0
  rfa_g=$(cat "$rfa_k.pg" 2>/dev/null)
  rfa_tree "$rfa_g"
  if [ -n "$RFA_SS" ]; then echo "-$rfa_g"; else echo "$rfa_g"; fi
}
rfa_kill() {
  rfa_v=$(rfa_victims "$1")
  [ -n "$rfa_v" ] || return 0
  kill -TERM $rfa_v 2>/dev/null
  ( ( sleep 2; [ -e "$RA/$1.job" ] || exit 0; kill -KILL $rfa_v 2>/dev/null
      sleep 1; rm -f "$RA/$1.job" ) </dev/null >/dev/null 2>&1 & )
}
rfa_full() { set -- "$RA"/*.job; [ -e "$1" ] && [ "$#" -ge 8 ]; }
rfa_lines() {
  LC_ALL=C $RFA_AWK -v id="$1" -v l="$2" -v c="$3" -v w="$RFA_W" '{ s = $0
    while (length(s) > w) { printf "%s", c " " id " " substr(s, 1, w) "\n"; fflush(); s = substr(s, w + 1) }
    printf "%s", l " " id " " s "\n"; fflush() }'
}
rfa_bytes() {
  while :; do
    rfa_h=$(dd bs="$RFA_B" count=1 2>/dev/null | od -A n -t x1 -v | tr -d ' \t\n')
    [ -n "$rfa_h" ] || break
    printf 'D %s %s\n' "$1" "$rfa_h"
  done
}
rfa_hold() {
  while [ ! -e "$RA/$1.w" ] && [ -d "$RA" ] && kill -0 "$RFA_PID" 2>/dev/null; do
    rfa_hp=; { read -r rfa_hp <"$RA/$1.pg"; } 2>/dev/null
    [ -n "$rfa_hp" ] && { kill -0 "$rfa_hp" 2>/dev/null || break; }
    sleep "$RFA_T"
  done
}
rfa_run() {
  rfa_j="$RA/$1"
  if [ "$2" = Q ]; then
    { printf "$3" | $RFA_SS sh -c 'echo "$$" >"$1"; exec sh -s' rfa "$rfa_j.pg" 2>"$rfa_j.err"
      echo "$?" >"$rfa_j.rc"; } | rfa_lines "$1" L C
  else
    { rfa_hold "$1" | $RFA_SS sh -c 'echo "$$" >"$1"; exec sh -c "$2"' rfa "$rfa_j.pg" "$(printf "$3")" 2>"$rfa_j.err"
      echo "$?" >"$rfa_j.rc"; } | rfa_bytes "$1"
  fi
  [ -s "$rfa_j.err" ] && tail -c 8192 "$rfa_j.err" | rfa_lines "$1" E F
  rfa_rc=$(cat "$rfa_j.rc" 2>/dev/null)
  rm -f "$rfa_j.pg" "$rfa_j.err" "$rfa_j.rc" "$rfa_j.w" "$rfa_j.job"
  printf 'Z %s %s\n' "$1" "${rfa_rc:-143}"
}
rfa_end() {
  trap '' HUP INT TERM PIPE
  rfa_v=
  for rfa_f in "$RA"/*.job; do [ -f "$rfa_f" ] && rfa_v="$rfa_v $(rfa_victims "$(basename "$rfa_f" .job)")"; done
  if [ -n "${rfa_v# }" ]; then kill -TERM $rfa_v 2>/dev/null; sleep "$RFA_T"; kill -KILL $rfa_v 2>/dev/null; fi
  rm -rf "$RA"
}
trap rfa_end EXIT; trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 141' PIPE
printf '\nR %s 1 %s %s\n' "$RFA_PID" "${RFA_SS:--}" "$RFA_PB"
while IFS=' ' read -r rfa_op rfa_id rfa_a; do
  case $rfa_id in ''|*[!A-Za-z0-9_]*) continue ;; esac
  case $rfa_op in
    P) printf 'P %s\n' "$rfa_id" ;;
    Q|S)
      if [ -e "$RA/$rfa_id.job" ] || rfa_full; then printf 'F %s runfleet agent: busy\nZ %s 125\n' "$rfa_id" "$rfa_id"
      else : >"$RA/$rfa_id.job"; ( rfa_run "$rfa_id" "$rfa_op" "$rfa_a" </dev/null 2>/dev/null & ); fi ;;
    W) [ -e "$RA/$rfa_id.job" ] && : >"$RA/$rfa_id.w" ;;
    K) rfa_kill "$rfa_id" ;;
  esac
done
exit 0
"""

# Reads exactly the agent's byte count from stdin (`read` and `dd bs=1` never read ahead), then runs it.
BOOTSTRAP = 'IFS= read -r n; eval "$(dd bs=1 count=$n 2>/dev/null)"'


def remote_command(remote_shell: str = "sh") -> str:
    """The one remote command word of the session ssh (the login shell parses it: bash, zsh, tcsh, fish alike)."""
    if remote_shell == "sh":
        return "exec sh -c " + shlex.quote(BOOTSTRAP)
    return "exec bash -l -c " + shlex.quote(BOOTSTRAP)     # remote_shell="login": the agent runs in a login env


def hello() -> bytes:
    """The first bytes written to the session: `<n>\\n` and the agent's n bytes."""
    text = AGENT_SH.encode("ascii")
    return f"{len(text)}\n".encode("ascii") + text


_SAFE = frozenset(range(0x21, 0x7F)) - {ord("\\"), ord("%")}
_TABLE = [chr(b) if b in _SAFE else "%%" if b == ord("%") else f"\\{b:03o}" for b in range(256)]


def encode_payload(data: bytes) -> str:
    """`data` as a printf(1) format that prints exactly `data` (see the module docstring)."""
    out = "".join(_TABLE[b] for b in data)
    return "\\055" + out[1:] if out.startswith("-") else out


def _check_id(value: str) -> str:
    if not ID_RE.match(value):
        raise ValueError(f"bad agent id {value!r}")
    return value


def ping(token: str) -> bytes:
    return f"P {_check_id(token)}\n".encode("ascii")


def probe_request(job: str, script: bytes) -> bytes:
    return f"Q {_check_id(job)} {encode_payload(script)}\n".encode("ascii")


def stream_request(job: str, cmd: str) -> bytes:
    return f"S {_check_id(job)} {encode_payload(cmd.encode('utf-8'))}\n".encode("ascii")


def close_stdin(job: str) -> bytes:
    return f"W {_check_id(job)}\n".encode("ascii")


def kill(job: str) -> bytes:
    return f"K {_check_id(job)}\n".encode("ascii")


# ---------------------------------------------------------------- replies
@dataclass(frozen=True, slots=True)
class Ready:
    pid: int
    version: int
    setsid: bool
    pipe_buf: int


@dataclass(frozen=True, slots=True)
class Pong:
    token: str


@dataclass(frozen=True, slots=True)
class Output:
    job: str
    data: bytes
    stderr: bool = False


@dataclass(frozen=True, slots=True)
class Done:
    job: str
    rc: int


@dataclass(frozen=True, slots=True)
class Fatal:
    text: str


@dataclass(frozen=True, slots=True)
class Noise:
    line: bytes                  # anything else: a login banner before `R`, a line we cannot read


Reply = Ready | Pong | Output | Done | Fatal | Noise


class ReplyDecoder:
    """Bytes from the session's stdout -> replies, across arbitrary chunk boundaries. Lines before `R` are noise
    (`.bashrc` output); so is any line that does not parse."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.ready = False

    def feed(self, data: bytes) -> list[Reply]:
        self._buf += data
        out: list[Reply] = []
        start = 0
        while True:
            nl = self._buf.find(b"\n", start)
            if nl < 0:
                break
            out.append(self._line(bytes(self._buf[start:nl])))
            start = nl + 1
        del self._buf[:start]
        if len(self._buf) > LINE_LIMIT:
            out.append(Noise(bytes(self._buf)))
            self._buf.clear()
        return out

    def eof(self) -> list[Reply]:
        """End of the stream: a last line without a newline is noise (the agent always ends its lines)."""
        rest, self._buf = bytes(self._buf), bytearray()
        return [Noise(rest)] if rest else []

    def _line(self, line: bytes) -> Reply:
        op, _, rest = line.partition(b" ")
        if not self.ready:
            parts = rest.split(b" ")
            if op == b"R" and len(parts) == 4 and all(p.isdigit() for p in (parts[0], parts[1], parts[3])):
                self.ready = True
                return Ready(int(parts[0]), int(parts[1]), parts[2] != b"-", int(parts[3]))
            if op == b"X" and rest:
                return Fatal(rest.decode("utf-8", "replace"))
            return Noise(line)
        job_b, _, text = rest.partition(b" ")
        try:
            job = job_b.decode("ascii")
        except UnicodeDecodeError:
            return Noise(line)
        if not ID_RE.match(job):
            return Noise(line)
        if op == b"L":
            return Output(job, text + b"\n")
        if op == b"C":
            return Output(job, text)
        if op == b"D":
            try:
                return Output(job, bytes.fromhex(text.decode("ascii")))
            except (ValueError, UnicodeDecodeError):
                return Noise(line)
        if op == b"E":
            return Output(job, text + b"\n", stderr=True)
        if op == b"F":
            return Output(job, text, stderr=True)
        if op == b"Z" and text.lstrip(b"-").isdigit():
            return Done(job, int(text))
        if op == b"P" and not text:
            return Pong(job)
        return Noise(line)
