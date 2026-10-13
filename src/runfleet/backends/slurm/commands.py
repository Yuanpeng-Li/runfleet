"""[WP2] Every Slurm command string and field list (single source, §5.2.2).

Every data value interpolated into a command is validated here first and `shlex.quote`d (§4.3 notes); anything that
fails validation is rejected before rendering (ValueError).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Sequence

from runfleet.util import shell

SEP = "\x1f"   # ASCII unit separator: -O field suffix and sacct --delimiter, so a "|" in a job name cannot split a row.
               # Verified at M3 on all three clusters; if a cluster rejects it, that host falls back to "|" (logged).
FALLBACK_SEP = "|"

SQUEUE_FIELDS = ["JobID", "ArrayJobID", "ArrayTaskID", "Name", "UserName", "State", "StateCompact", "Reason",
    "TimeUsed", "TimeLimit", "TimeLeft", "NumNodes", "NumCPUs", "MinMemory", "Partition", "QOS", "Account",
    "SubmitTime", "StartTime", "EndTime", "NodeList", "BatchHost", "tres-alloc", "tres-per-node", "tres-per-job",
    "RestartCnt", "STDOUT", "STDERR", "WorkDir", "Command"]          # Command MUST stay last

# Legacy fallback (only if -O fails with rc≠0; -O works on all three clusters today):
SQUEUE_LEGACY = "%A|%F|%K|%j|%u|%T|%t|%r|%M|%l|%L|%D|%C|%m|%P|%q|%a|%V|%S|%e|%N|%B|%b|%Z|%o"

SACCT_FIELDS = ["JobID", "JobIDRaw", "JobName", "User", "Account", "QOS", "Partition", "State", "ExitCode",
    "DerivedExitCode", "Reason", "Submit", "Start", "End", "Elapsed", "Timelimit", "AllocCPUS", "ReqMem",
    "ReqTRES", "AllocTRES", "NNodes", "NodeList"]  # + ["StdOut","StdErr"] if in caps.sacct_fields, + ["WorkDir"] last

STEP_FIELDS = "JobID,JobName,State,ExitCode,DerivedExitCode,Reason,Elapsed,Start,End,NodeList,NTasks,MaxRSS,AveCPU," \
              "TRESUsageInMax%400,TRESUsageInTot%400"
STEP_EXTRA = ("TotalCPU",)      # efficiency card (§5.2.8): appended when caps.sacct_fields lists them (detail only)

SSTAT_FMT = "JobID,NTasks,AveCPU,AveRSS,MaxRSS,TRESUsageInAve%400,TRESUsageInMax%400,TRESUsageInTot%400"

SINFO_FIELDS = ["NodeList", "Partition", "StateLong", "CPUsState", "Memory", "AllocMem", "FreeMem", "Gres", "GresUsed",
                "Reason"]                                # Reason last

# ---------------------------------------------------------------- validators (§5.2.2, util/shell.py)
JOBID_RE = shell.JOBID_RE
NAME_RE = shell.NAME_RE                              # partition, user, QOS, account
SQUEUE_ARG_RE = re.compile(r"^[A-Za-z0-9_.,:=%/+\-]+$")   # squeue_args tokens (config)


def check_job_id(job_id: str) -> str:
    return shell.job_id(job_id)


def check_name(value: str, what: str = "name") -> str:
    try:
        return shell.name(value)
    except ValueError:
        raise ValueError(f"invalid {what} {value!r}") from None


def check_path(path: str) -> str:
    return shell.path(path)


def check_squeue_arg(arg: str) -> str:
    if not SQUEUE_ARG_RE.match(arg):
        raise ValueError(f"invalid squeue argument {arg!r}")
    return arg


def strip_throttle(display_id: str) -> str:
    """Action id for an array summary: `31270818_[1-24%1]` → `31270818_[1-24]` (the `%N` throttle is not an id)."""
    return re.sub(r"%\d+(?=\])", "", display_id)


def _q(values: Iterable[str]) -> str:
    return " ".join(shlex.quote(v) for v in values)


# ---------------------------------------------------------------- commands
def squeue_cmd(users: Sequence[str] | None, extra: Sequence[str], sep: str = SEP) -> str:
    who = "--me" if not users else "-u " + shlex.quote(",".join(check_name(u, "user") for u in users))
    fmt = ",".join(f"{f}:{sep}" for f in SQUEUE_FIELDS)
    return f"squeue {who} -h -O {shlex.quote(fmt)} {_q(check_squeue_arg(a) for a in extra)}".rstrip()


def squeue_legacy_cmd(users: Sequence[str] | None, extra: Sequence[str]) -> str:
    who = "--me" if not users else "-u " + shlex.quote(",".join(check_name(u, "user") for u in users))
    return f"squeue {who} -h -o {shlex.quote(SQUEUE_LEGACY)} {_q(check_squeue_arg(a) for a in extra)}".rstrip()


def sacct_fields(caps_fields: frozenset[str]) -> list[str]:
    """SACCT_FIELDS + StdOut/StdErr when this sacct knows them + WorkDir (last: it is free text)."""
    extra = ["StdOut", "StdErr"] if {"StdOut", "StdErr"} <= caps_fields else []
    return [*SACCT_FIELDS, *extra, "WorkDir"]


def sacct_window_cmd(since_s: int, fields: Sequence[str], sep: str = SEP) -> str:
    # relative start avoids timezone parsing; -E now is REQUIRED for any state filtering (field probes)
    return (f'sacct -u "$(id -un)" -S now-{int(since_s)}seconds -E now -X -P --delimiter={shlex.quote(sep)} -n '
            f'--format={",".join(fields)}')


def sacct_ids_cmd(ids_raw: Sequence[str], fields: Sequence[str], sep: str = SEP) -> str:
    ids = ",".join(check_job_id(i) for i in ids_raw)
    return f"sacct -j {shlex.quote(ids)} -X -P --delimiter={shlex.quote(sep)} -n --format={','.join(fields)}"


def step_fields(caps_fields: frozenset[str] = frozenset()) -> str:
    """STEP_FIELDS + the STEP_EXTRA fields this sacct knows (on-demand detail query only, never the periodic one)."""
    return ",".join([STEP_FIELDS, *(f for f in STEP_EXTRA if f in caps_fields)])


def sacct_steps_cmd(job_id_raw: str, sep: str = SEP, fields: str = STEP_FIELDS) -> str:
    return f"sacct -j {shlex.quote(check_job_id(job_id_raw))} -P --delimiter={shlex.quote(sep)} -n --format={fields}"


def scontrol_cmd(ids_raw: Sequence[str]) -> str:          # one RPC per id; ids validated by JOBID_RE
    return "for j in " + _q(check_job_id(i) for i in ids_raw) + \
           '; do echo "@@job $j"; scontrol show job -d -o "$j" 2>&1; done; true'


def sstat_cmd(ids_raw: Sequence[str]) -> str:              # fields never contain "|"
    return f"sstat -a -j {shlex.quote(','.join(check_job_id(i) for i in ids_raw))} -P -n --format={SSTAT_FMT}"


def logstat_cmd(paths: Sequence[str]) -> str:              # parse with rsplit("|", 2): paths may contain "|"
    return "stat -L -c '%n|%s|%Y' -- " + _q(check_path(p) for p in paths) + " 2>&1; true"


def sinfo_cmd(partitions: Sequence[str], sep: str = SEP) -> str:
    fmt = ",".join(f"{f}:{sep}" for f in SINFO_FIELDS)
    parts = ",".join(check_name(p, "partition") for p in partitions)
    return f"sinfo -h -N -p {shlex.quote(parts)} -O {shlex.quote(fmt)}"


# ---------------------------------------------------------------- detect (capability) commands (§5.1)
VERSION_CMD = "sinfo --version"
CONFIG_CMD = ("scontrol show config | grep -E '^(ClusterName|AccountingStorageType|JobAcctGatherType|"
              "JobAcctGatherFrequency|PreemptMode|PrologFlags|LaunchParameters|MinJobAge|PrivateData) '")
ASSOC_CMD = 'sacctmgr -n -P show assoc where user="$(id -un)" format=Cluster,Account,Partition,QOS'
# who may submit where (§5.1): the user's groups (group database, comma-separated: names may hold spaces), the parent
# chain of the user's accounts (Allow/DenyAccounts include sub-accounts; `parent=cluster|account|parent`, up to 6
# levels, root left out; `parents=complete` only when every sacctmgr call worked and the chain ended), then one line
# per partition (AllowGroups/Accounts/Qos, Deny*, Hidden, RootOnly, State, TRES gres/gpu=N)
PARTS_CMD = (
    "u=$(id -un); g=$(for i in $(id -G \"$u\"); do getent group \"$i\" 2>/dev/null | cut -d: -f1; done "
    "| paste -sd, -); [ -n \"$g\" ] || g=$(id -Gn \"$u\" | tr ' ' ,); printf 'groups=%s\\n' \"$g\"; ok=1; "
    "r=$(sacctmgr -n -P show assoc where user=\"$u\" format=Account 2>/dev/null) || ok=0; "
    "a=$(printf '%s\\n' \"$r\" | sed '/^$/d' | sort -u | paste -sd, -); i=0; "
    "while [ -n \"$a\" ] && [ $i -lt 6 ]; do "
    "r=$(sacctmgr -n -P show assoc where account=\"$a\" format=Cluster,Account,User,ParentName 2>/dev/null) || ok=0; "
    "p=$(printf '%s\\n' \"$r\" | awk -F'|' '$3 == \"\" && $4 != \"\" && $4 != \"root\" {print $1 \"|\" $2 \"|\" $4}' "
    "| sort -u); [ -n \"$p\" ] && printf '%s\\n' \"$p\" | sed 's/^/parent=/'; "
    "a=$(printf '%s\\n' \"$p\" | sed '/^$/d' | cut -d'|' -f3 | sort -u | paste -sd, -); i=$((i + 1)); done; "
    "[ -z \"$a\" ] && [ $ok = 1 ] && echo parents=complete; "
    "scontrol show partition -o")
SACCTFMT_CMD = "sacct --helpformat"
GPUTRES_CMD = ("j=$(sacct -u \"$(id -un)\" -S now-14days -E now -X -n -P --format=JobIDRaw,AllocTRES,State "
               "| awk -F'|' '$2 ~ /gres\\/gpu/ && $3 !~ /^(RUNNING|PENDING)/ {j=$1} END {print j}'); "
               "[ -n \"$j\" ] && sacct -j \"$j\" -n -P --format=JobID,TRESUsageInMax%400; true")


# ---------------------------------------------------------------- actions (§6.1)
def scancel_cmd(ids: Sequence[str], *, pending_only: bool = False) -> str:
    return "scancel " + ("--state=PENDING " if pending_only else "") + _q(check_job_id(i) for i in ids)


def scontrol_action_cmd(verb: str, ids: Sequence[str]) -> str:
    if verb not in ("hold", "release", "requeue", "requeuehold"):
        raise ValueError(f"invalid scontrol verb {verb!r}")
    return f"scontrol {verb} " + _q(check_job_id(i) for i in ids)
