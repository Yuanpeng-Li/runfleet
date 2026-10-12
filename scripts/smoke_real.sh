#!/usr/bin/env bash
# Read-only smoke test of RunFleet against REAL hosts (DESIGN.md §11.4 M3).
#
#   !!! Touches real clusters and workstations. Run it only by hand, on hosts you are allowed to probe. !!!
#
# Per host it runs `runfleet doctor` and `runfleet snapshot --json --debug` (both read-only, --readonly on top),
# counts every `ssh` the two commands start through a PATH shim, and fails when a host needed more than the M3 budget
# of network ssh sessions (default 3). Creating a RunFleet-managed master is budgeted separately: at most
# --max-masters (default 1) `ssh -fN` logins, each with at most one ProxyCommand hop (`ssh -W`) — a cold managed host
# (behind a jump host, once ControlPersist expired) legitimately shows master + hop + 3 sessions. Local `ssh -G` (config evaluation)
# and `ssh -O check|exit` (talks to the local master socket) are counted separately and are not budgeted.
#
# usage: scripts/smoke_real.sh --yes [--max-calls N] [--max-masters N] [--out DIR] HOST ...   (ssh aliases or config
#        names; at least one). RUNFLEET=<command> overrides how runfleet is started (default: `runfleet` on PATH,
#        else `uv run runfleet` from this checkout).
set -euo pipefail

HOSTS=()
MAX_CALLS=3
MAX_MASTERS=1
OUT=""
APPROVED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) APPROVED=1 ;;
    --max-calls) MAX_CALLS="$2"; shift ;;
    --max-masters) MAX_MASTERS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) HOSTS+=("$1") ;;
  esac
  shift
done
if [ "$APPROVED" != 1 ]; then
  echo "This contacts real hosts: ${HOSTS[*]:-(none named yet)} (read-only)." >&2
  echo "Re-run with --yes once you may probe them." >&2
  exit 2
fi
[ ${#HOSTS[@]} -gt 0 ] || { echo "name at least one HOST to test (see --help)" >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${RUNFLEET:-}" ]; then
  read -r -a RF <<<"$RUNFLEET"
elif command -v runfleet >/dev/null 2>&1; then
  RF=(runfleet)
else
  RF=(uv run --project "$REPO" runfleet)
fi

REAL_SSH=$(command -v ssh) || { echo "no ssh on PATH" >&2; exit 3; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/rf-smoke.XXXXXX")
[ -n "$OUT" ] || OUT="$WORK/out"
mkdir -p "$WORK/bin" "$OUT"
trap 'rm -rf "$WORK/bin"' EXIT

# The shim logs one line per ssh start: "<host under test>\t<kind>\t<argv>" and then becomes the real ssh.
# kinds: config (-G), control (-O), master (-fN: a managed master login), hop (-W: a ProxyCommand jump leg; only a
# standalone argument counts — the ProxyCommand string inside "-o ProxyCommand=… -W …" is one argument), net (the rest).
cat >"$WORK/bin/ssh" <<EOF
#!/bin/sh
kind=net
f=0; n=0
for a in "\$@"; do
  case "\$a" in
    -G) kind=config ;;
    -O) kind=control ;;
    -W) [ "\$kind" = net ] && kind=hop ;;
    -fN|-Nf) [ "\$kind" = net ] && kind=master ;;
    -f) f=1 ;;
    -N) n=1 ;;
  esac
done
[ "\$kind" = net ] && [ "\$f\$n" = 11 ] && kind=master
printf '%s\t%s\t%s\n' "\${RF_SMOKE_HOST:-?}" "\$kind" "\$*" >>"$WORK/calls.log"
exec "$REAL_SSH" "\$@"
EOF
chmod +x "$WORK/bin/ssh"
: >"$WORK/calls.log"
export PATH="$WORK/bin:$PATH"

echo "runfleet: $("${RF[@]}" --version)"
echo "output:   $OUT"
fail=0
DUMPS="(not reported; see runfleet.log in the RunFleet state dir)"
for host in "${HOSTS[@]}"; do
  echo "== $host"
  export RF_SMOKE_HOST="$host"
  set +e
  "${RF[@]}" doctor --readonly "$host" >"$OUT/doctor-$host.txt" 2>&1
  doctor_rc=$?
  "${RF[@]}" snapshot --readonly --debug --json --timeout 90 "$host" >"$OUT/snapshot-$host.json" \
    2>"$OUT/snapshot-$host.err"
  snap_rc=$?
  set -e
  net=$(awk -F'\t' -v h="$host" '$1 == h && $2 == "net"' "$WORK/calls.log" | wc -l | tr -d ' ')
  cfg=$(awk -F'\t' -v h="$host" '$1 == h && $2 == "config"' "$WORK/calls.log" | wc -l | tr -d ' ')
  ctl=$(awk -F'\t' -v h="$host" '$1 == h && $2 == "control"' "$WORK/calls.log" | wc -l | tr -d ' ')
  mst=$(awk -F'\t' -v h="$host" '$1 == h && $2 == "master"' "$WORK/calls.log" | wc -l | tr -d ' ')
  hop=$(awk -F'\t' -v h="$host" '$1 == h && $2 == "hop"' "$WORK/calls.log" | wc -l | tr -d ' ')
  verdict=ok
  if [ "$net" -gt "$MAX_CALLS" ]; then verdict="TOO MANY ssh sessions (> $MAX_CALLS)"; fail=1; fi
  if [ "$mst" -gt "$MAX_MASTERS" ]; then verdict="TOO MANY master logins (> $MAX_MASTERS)"; fail=1; fi
  if [ "$hop" -gt "$mst" ]; then verdict="jump hops without a master login ($hop > $mst): a probe fell back"; fail=1; fi
  if [ "$doctor_rc" -ne 0 ] && [ "$doctor_rc" -ne 4 ]; then verdict="doctor exit $doctor_rc"; fail=1; fi
  if [ "$snap_rc" -ne 0 ] && [ "$snap_rc" -ne 4 ]; then verdict="snapshot exit $snap_rc"; fail=1; fi
  printf '   ssh sessions %s (budget %s) · master logins %s (+%s hop, budget %s) · ssh -G %s · ssh -O %s · doctor rc %s · snapshot rc %s · %s\n' \
    "$net" "$MAX_CALLS" "$mst" "$hop" "$MAX_MASTERS" "$cfg" "$ctl" "$doctor_rc" "$snap_rc" "$verdict"
  dumps=$(sed -n 's/^runfleet: --debug: raw probe dumps in //p' "$OUT/snapshot-$host.err" | tail -n 1)
  [ -z "$dumps" ] || DUMPS="$dumps"
  sed 's/^/   /' "$OUT/doctor-$host.txt"
done
cp "$WORK/calls.log" "$OUT/ssh-calls.tsv"
echo "every ssh invocation: $OUT/ssh-calls.tsv; --debug raw dumps: $DUMPS"
exit "$fail"
