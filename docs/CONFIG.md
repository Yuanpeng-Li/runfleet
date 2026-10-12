# RunFleet configuration

RunFleet works without any configuration (`runfleet k2 labstation` monitors those ssh aliases), but a config file
gives your fleet an order, names, protection rules and tuned refresh rates. It lives at
`~/.config/runfleet/config.toml` (override with `-c PATH`). You rarely need to edit it by hand: the first `runfleet`
offers the hosts of `~/.ssh/config`, hosts you name on the command line are saved once they answer, and **Settings**
(press `,` in RunFleet) adds, removes, tests and reorders hosts and flips the `[general]` settings below.
`runfleet hosts import --all --write` does the same from a script.

RunFleet edits the file in place and keeps your comments, blank lines and formatting: only the touched `[[host]]`
table or `[general]` key changes, each change is one read-modify-write of the file as it is on disk (so an edit you
made in an editor meanwhile is kept), checked with the same validation as at startup, and written atomically (a temp
file next to it, then a rename; the file mode is kept and a symlinked config is written through to its target). Two
RunFleet windows saving at the same moment take turns (an flock on `.config.toml.lock`, which exists only during the
save), so neither edit is lost. A comment block directly above a `[[host]]` header belongs to that host: it moves when
you reorder and goes away when you remove the host (Settings asks twice before removing a table with settings of its
own, and checking the host again in the same session restores it). Blank lines between tables stay where they are,
and a section heading above the first host (comment lines up to a `# ---` / `# ===` style marker) stays above
whichever host is first. If the file is not valid TOML, RunFleet never rewrites it; Settings shows the error and `e`
opens the file in `$EDITOR`. If it cannot be written (read-only directory, disk full), Settings shows that error and
a remembered host gets a notice; nothing changes. `--demo` never writes anything.

Every key is optional except `host.name`. Durations are strings with a unit: `"500ms"`, `"15s"`, `"2m"`, `"3h"`,
`"3d"` (a bare number is an error, so a unit is never guessed). Unknown keys produce a warning and are ignored, so
newer configs keep working with older versions. A wrong type or a bad duration stops RunFleet with exit code 2 and the
exact key, for example `config.toml: host[2].poll_jobs: expected duration like "15s", got 5`.

## A complete example (three Slurm clusters and two GPU workstations)

```toml
[general]
readonly = false
mouse = true
default_view = "fleet"         # fleet | jobs | gpus
recent_window = "24h"          # ended jobs/tasks stay in the default "fresh" filter this long
stall_after = "30m"            # "log idle" hint threshold
idle_after = "10m"             # no input -> poll 4x slower
idle_pause_after = "2h"        # no input -> pause all polling until a key is pressed
max_parallel_ssh = 8
refresh_multiplier = 1.0
pager = ""                     # default: $PAGER, else "less -R +G"
remember_hosts = true          # `runfleet k2` saves k2 here once it answers

[ui]
gpu_util_thresholds = [10, 75]
gpu_mem_thresholds = [10, 80]
jobs_columns = ["host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "node_reason"]

[defaults.slurm]               # every host whose profile is Slurm
poll_jobs = "15s"              # floor 10s
poll_history = "2m"            # floor 60s
poll_nodes = "2m"              # floor 60s
poll_usage = "60s"             # floor 30s, never below the cluster's JobAcctGatherFrequency
poll_logstat = "2m"            # floor 60s
sacct_window = "3d"
log_templates = []
attach = "ssh"                 # live GPU/process readings: ssh | ssh+srun | srun | off (see "Live readings")
attach_refresh = "15s"         # floor 10s; ssh mode only

[defaults.workstation]         # every workstation
poll_gpu = "5s"                # floor 2s
poll_procs = "10s"             # floor 5s
poll_logstat = "60s"           # floor 30s
task_min_age = "60s"
task_min_cpu = 5.0             # percent
ignore = []
kill_grace = "10s"

[[host]]
name = "k2"                    # display name; also the ssh alias unless `ssh` is set
group = "clusters"
protected_partitions = ["labpi"]
protected_qos = ["labpi"]
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]

[[host]]
name = "vega"
connect_timeout = "120s"       # cold logins can take minutes (GSSAPI, MOTD, slow home directories)
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]

[[host]]
name = "lyra"                  # no ControlMaster in ~/.ssh/config -> RunFleet manages one ("managed")

[[host]]
name = "labstation"

[[host]]
name = "lynx"

# [[host]]
# name = "atlas"
# ssh = "local"                # this machine, no ssh
```

## `[general]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `readonly` | bool | `false` | disable every mutating action (same as `--readonly`) |
| `mouse` | bool | `true` | mouse support (`--no-mouse` turns it off for terminal-native selection) |
| `default_view` | `fleet` \| `jobs` \| `gpus` | `fleet` | the view RunFleet opens in |
| `recent_window` | duration | `24h` | how long ended jobs and tasks stay in the default *fresh* preset (and the `F n/24h` header count) |
| `stall_after` | duration | `30m` | `log idle` hint when a running job's log has not changed for this long |
| `idle_after` | duration | `10m` | no key/mouse input for this long → everything polls 4× slower |
| `idle_pause_after` | duration | `2h` | no input for this long → all polling pauses until the next key |
| `max_parallel_ssh` | int | `8` | global cap on concurrent ssh processes (polls, on-demand probes, and each `hosts import` / first-run probe including its master login; long-lived log streams are bounded per host instead) |
| `refresh_multiplier` | float | `1.0` | scales every interval (`+`/`-` change it at runtime, 0.5–4); Settings › refresh speed writes `2.0` (slow), `1.0` (normal) or `0.5` (fast) and applies it at once — per-command floors still apply |
| `pager` | string | `""` | pager for local log copies; empty = `$PAGER`, else `less -R +G` |
| `remember_hosts` | bool | `true` | hosts named on the command line (`runfleet k2 vega`) that are not configured yet are saved (a minimal `[[host]]` table) as soon as they answer or need a login; one notice "saved k2, vega — manage in Settings (,)". That run still shows only the named hosts. |

Settings › General changes `default_view` (used at the next start), `refresh_multiplier`, `readonly`, `mouse`,
`idle_pause_after` and `remember_hosts`; all but `default_view` take effect immediately.

## `[ui]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gpu_util_thresholds` | `[low, high]` | `[10, 75]` | GPU bar colours: below low green, below high yellow, else red |
| `gpu_mem_thresholds` | `[low, high]` | `[10, 80]` | the same for GPU memory |
| `jobs_columns` | list | see example | Jobs table columns, in order: `host id name state elapsed left gpus gpu_util node_reason part gmem cpu mem user submit` (narrow terminals drop low-priority ones first) |

## `[notify]`

What happens when one of your jobs changes state while RunFleet runs (a toast, a terminal notification, an optional
local command). Nothing new runs on the servers: events come from the polls RunFleet makes anyway. Events of the
first poll after a start or a reconnect are never reported (that is the baseline); what ended while RunFleet was not
running shows up once as "since 21:40: …" (`D` opens the list).

```toml
[notify]
enabled = true
terminal = "auto"              # auto | osc9 | osc777 | bell | off
command = ["notify-send", "RunFleet"]          # or a shell string: 'ntfy pub runs "$RUNFLEET_SUMMARY"'
command_timeout = "10s"
events = ["ended_bad", "time_low", "restarted"]                       # every job of yours
watched_events = ["started", "ended_ok", "ended_other", "gpu_idle"]   # jobs you watch (w)
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `true` | `false`: no toast, terminal notification or command; events still go to palette › *Notifications* |
| `terminal` | `auto` \| `osc9` \| `osc777` \| `bell` \| `off` | `auto` | terminal notification: `auto` picks OSC 9 for iTerm2, WezTerm, Windows Terminal, ghostty and kitty, OSC 777 for foot, urxvt and VTE terminals, else the bell. Inside tmux the escape is passed through, which needs `set -g allow-passthrough on` in `~/.tmux.conf` (tmux 3.3+); without it tmux drops it silently |
| `command` | list of strings \| string | `[]` | run locally for each notification (never on a server): a list is the argv, a string runs through `/bin/sh -c`. It gets `RUNFLEET_EVENT` (`ended_bad`, `time_low`, …, or `summary` for a burst), `RUNFLEET_HOST`, `RUNFLEET_JOB`, `RUNFLEET_NAME`, `RUNFLEET_STATE` (`FAILED`, `TIMEOUT`, …) and `RUNFLEET_SUMMARY` (the one-line text) in its environment, no terminal and no stdin; a failure is logged in `runfleet.log` |
| `command_timeout` | duration | `10s` | the command is killed (with its children) after this long |
| `events` | list | `["ended_bad", "time_low", "restarted"]` | events reported for every job of yours |
| `watched_events` | list | `["started", "ended_ok", "ended_other", "gpu_idle"]` | events reported in addition for watched jobs (`w`) |

Events: `started` (a job starts running, a new task appears), `ended_bad` (FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL,
PREEMPTED, BOOT_FAIL, DEADLINE), `ended_ok` (COMPLETED), `ended_other` (CANCELLED, a task that ended with unknown
status), `restarted` (requeued, the restart count went up), `time_low` (less than 10 % of the time limit or 10
minutes left, once per run), `gpu_idle` (the `GPU idle` hint appears). Many events on one host at once become one
notification ("k2: 12 failed, 3 completed"); each job is still listed in *Notifications*.

Settings › General › *notifications* changes `enabled`, `terminal` and `events` (presets: `failures` = `ended_bad`,
`restarted`, `time_low`, the default; `+gpu idle`; `all`; another list shows as `custom` and is left alone unless you
change that row) and applies them at once; a missing `[notify]` table is added above the `[[host]]` tables.
`command`, `command_timeout` and `watched_events` are set here only.

## `[[host]]`

The order of `[[host]]` tables is the order of the cards and the scope keys `1`–`9`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | unique, `[A-Za-z0-9_.-]+`; shown everywhere |
| `ssh` | string | `name` | the ssh alias (anything `ssh` accepts), or `"local"` for this machine |
| `enabled` | bool | `true` | `false` keeps the host listed but not polled (`t` toggles at runtime) |
| `backend` | `auto` \| `slurm` \| `workstation` | `auto` | `auto` detects: `squeue` + `sinfo` → Slurm; `nvidia-smi` → GPU workstation; otherwise a CPU workstation |
| `color` | string | palette | host colour in tables |
| `group` | string | `""` | label shown on cards (`"clusters"`, `"workstations"`) |
| `mux` | `auto` \| `user` \| `managed` \| `off` | `auto` | ssh connection sharing, see below |
| `connect_timeout` | duration | `45s` | how long making a connection may take |
| `ssh_options` | list of strings | `[]` | extra `-o Key=Value` for background probes and log streams (not for interactive sessions) |
| `remote_shell` | `sh` \| `login` | `sh` | `login` runs the probe through `bash -l` for sites whose tools need a login profile |
| `forward_agent` | bool | `false` | `-A` for interactive shells only |

Slurm keys may be written flat in a `[[host]]` table and override `[defaults.slurm]` for that host:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `users` | list | `[]` | whose jobs to list (`squeue -u`); empty = `squeue --me` |
| `partitions` | list | `[]` | partitions shown in the node view; empty = partitions of your jobs and associations |
| `squeue_args` | list | `[]` | extra squeue arguments (validated tokens) |
| `poll_jobs`, `poll_history`, `poll_nodes`, `poll_usage`, `poll_logstat` | duration | see `[defaults.slurm]` | refresh intervals while the host is on screen (×3 when hidden); values below the floor are raised to it with a warning |
| `sacct_window` | duration | `3d` | how far back finished jobs are listed |
| `log_templates` | list | `[]` | where logs of finished jobs live when Slurm has forgotten them; placeholders `{workdir} {name} {jobid} {array_job} {task} {user}` (`{jobid}` is the raw job id) |
| `protected_partitions`, `protected_qos` | list | `[]` | no action is ever offered on jobs in these partitions / QOS |
| `attach` | `ssh` \| `ssh+srun` \| `srun` \| `off` | `ssh` | live per-GPU / per-process readings of a running job while its GPU or Procs tab is open (see *Live readings* below) |
| `attach_refresh` | duration | `15s` | how often those readings refresh in `ssh` mode (floor 10 s); `srun` reads only when you press `r` |

### Live readings (`attach`)

The background refresh never logs in to compute nodes. When you open the **GPU** or **Procs** tab of one of your
running jobs, RunFleet can read that job's node: per-GPU utilisation, memory, temperature and power, which process
holds how much GPU memory, the job's processes (with their Slurm step), and the job's cgroup memory (incl. shared
memory), OOM kills and current CPU use. It stops when you leave the tab, when RunFleet is in the background, and after
10 minutes without a key press; only one job is read at a time.

- `ssh` (default): from the login node, `ssh <node>` lands inside your job (pam_slurm_adopt). Nothing is recorded in
  Slurm accounting. Works where the cluster runs pam_slurm_adopt (`k2` and `vega` above). If you have several jobs on
  one node, ssh can only enter the newest one; the GPU rows then belong to that job and say so.
- `srun`: `srun --jobid <job> --overlap` runs a one-second read-only step in the job. **Every read adds one permanent
  row `<job>.N rfprobe` to `sacct`, which cannot be deleted.** RunFleet therefore never refreshes it on its own: each
  `r` asks for confirmation ("once" or "allow for this job until RunFleet exits"; there is no "never ask"). The only
  way on clusters whose compute nodes refuse ssh (`lyra` above).
- `ssh+srun`: ssh, and the srun read is offered (with the same confirmation) when ssh is refused or lands in another
  job.
- `off`: never read compute nodes; the tabs show Slurm accounting (`sstat`) only.

`--readonly` allows `ssh` reads but never runs `srun` (it writes accounting). Example for the fleet above:

```toml
[[host]]
name = "lyra"
attach = "srun"                # sshd refuses node logins; each read asks first and adds one sacct row
```

Workstation keys, flat in `[[host]]` or in `[defaults.workstation]`:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `poll_gpu`, `poll_procs`, `poll_logstat` | duration | `5s`, `10s`, `60s` | refresh intervals (floors 2 s, 5 s, 30 s) |
| `task_min_age` | duration | `60s` | a process becomes a task only after running this long … |
| `task_min_cpu` | float | `5.0` | … and using at least this much CPU (%), or when it holds a GPU |
| `ignore` | list of regexes | `[]` | process command lines that never start a task (added to the built-in list: shells, sshd, tmux/screen, editors' servers, nvitop/htop/top, …) |
| `kill_grace` | duration | `10s` | default grace time between SIGTERM and SIGKILL for *Terminate* |

## Connection sharing (`mux`)

RunFleet sends one probe per host per refresh. Making a new ssh connection each time would be slow (and through a jump
host, impolite), so it shares one connection per host:

- `auto` (default) uses your own ControlMaster when `ssh -G <alias>` shows a `ControlPath` and a `ControlPersist`
  (`user` mode). If RunFleet has to start that master, it starts it with only your options, so your own terminals can
  share it afterwards. Otherwise RunFleet keeps its own master under `$XDG_RUNTIME_DIR/runfleet/` (`managed` mode,
  15 minutes after the last use).
- `user` / `managed` force one of the two; `off` makes every probe a fresh login (only for testing). `user` needs a
  `ControlPath` in your `~/.ssh/config`: without one RunFleet does not multiplex at all (no master is created or
  checked), so every probe is a fresh login as with `off`.

RunFleet holds at most three sessions on a shared master, so your own `ssh` sessions are never refused. It never
closes your master on its own; `runfleet doctor --reset HOST` closes managed ones.

## Hosts without a config

`runfleet HOST …` accepts ssh aliases and config names: a name found in the config keeps its settings, anything else
becomes an ad-hoc host with the defaults, and only the named hosts are shown that run. New hosts are saved to the
config once they answer (`remember_hosts`, see above; without a config file this creates it), so the next plain
`runfleet` includes them. `runfleet hosts import ALIAS … --write` adds hosts from a script (appending `[[host]]`
tables for new names only). Hosts added from Settings or remembered this way get a minimal table (`name`, plus `ssh`
when the target is not a plain alias, e.g. `name = "me_box"`, `ssh = "me@box"`); add per-host keys by hand.
