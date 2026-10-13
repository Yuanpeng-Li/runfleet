# RunFleet user guide

RunFleet watches the Slurm clusters and GPU workstations you reach with `ssh` and shows their jobs, GPUs and hosts in
one terminal UI. This guide covers starting it, the screens, filtering, actions and what to do when a host misbehaves.
Configuration keys are in [CONFIG.md](CONFIG.md); the design and its reasoning in [DESIGN.md](DESIGN.md).

## Starting

| Command | What it does |
|---|---|
| `runfleet` | Monitors the hosts in `~/.config/runfleet/config.toml`. With no config on a terminal it lists the Host aliases of `~/.ssh/config`, lets you pick (`1-5`, `k2,lynx`, `all`, `d` for the demo), probes them read-only and writes the config. Hosts that do not answer right now (VPN off, a slow first login) are not saved; it prints the `runfleet hosts import … --write` command that adds them later. After that, manage hosts in **Settings** (`,`). A config whose hosts were all removed starts in Settings. |
| `runfleet HOST …` | Exactly these hosts this run (ssh aliases or config names), merged with matching `[[host]]` settings. Hosts not in the config yet are saved as soon as they answer or need a login (`general.remember_hosts`, on by default), with one notice; the next plain `runfleet` includes them. |
| `runfleet --demo` | A simulated fleet (Slurm clusters `k2`, `vega`, `lyra`; workstations `labstation`, `lynx`) behind the real parsers and UI. No network. `lynx` starts as "needs auth": press `c` to "log in". `--demo-seed N` picks another world. |
| `runfleet connect HOST` | Logs in once in this terminal (Duo, passwords and host-key questions appear here) with RunFleet's multiplexing, then exits. Later RunFleet sessions reuse the connection. |
| `runfleet hosts list` | The configured hosts (with no config: how to create one). |
| `runfleet hosts import [--all \| ALIAS …] [--write]` | Probes aliases read-only (`BatchMode`, at most `max_parallel_ssh` at a time, one line per host as it answers) and prints what each one is (Slurm version, workstation, needs auth, unreachable). `--write` saves the usable ones (including hosts that only need a login) to the config; an existing file is appended to, never rewritten, and aliases already in it are reported as "already configured". Unreachable hosts are listed with the command to add them later. |
| `runfleet doctor [HOST …] [--reset]` | Per host: `ssh -G` summary, multiplexing mode, ProxyJump rewrite, master check, probe latency, detected profile, capabilities, missing tools, clock skew, GPU accounting. `--reset` closes RunFleet-managed ssh masters first. |
| `runfleet snapshot [HOST …] [--json] [--timeout SEC]` | Polls every host once and prints a table (your jobs by state; your GPUs: allocated GPUs on a cluster, `in use/total` on a workstation), or with `--json` a document with schema `runfleet.snapshot/1` (status, capabilities, jobs, GPUs, processes, nodes and partitions per host). |
| `runfleet wait HOST:JOBID … [--timeout SEC] [--json]` | Waits until the jobs end, for scripts and agents (`runfleet wait k2:2332861 && ./eval.sh`). Polls just those hosts like an unattended RunFleet (normal cadence, 4× slower after 10 minutes, never paused), read-only, and prints every state change (`--json`: one `runfleet.wait/1` object per line). `k2:31270818` waits for every task of that array; `lynx:p642229` names a workstation task. Exit `0` all ended COMPLETED · `1` one ended otherwise (failed, cancelled, timeout, a task's unknown exit) · `2` `--timeout` passed · `3` error (unknown host or job, a host that needs a login — run `runfleet connect` first — or is unreachable for 10 minutes). |

Global options (anywhere on the command line): `-c PATH` config file, `--readonly` (every action disabled),
`--no-mouse` (terminal-native text selection), `--no-cache` (no warm start from `~/.local/state/runfleet/cache.sqlite`),
`--debug` (verbose log and a raw dump of every probe under the state dir's `debug/`; non-TUI commands print the
directory on stderr), `--version`.

Exit codes: `0` ok · `1` runtime error · `2` configuration or usage error · `3` no `ssh` client found (never with
`--demo`) · `4` `snapshot`/`doctor`: at least one host failed (unreachable, needs auth, a failed check). `wait` has its
own, above: every error, a bad configuration included, is `3` (an argument error caught by the parser stays `2`).

## Screens

**Fleet view** (the start screen). One card per host: for clusters your job counts, your GPUs and the job
utilisation from Slurm accounting, free GPUs per partition (`13/108 +11 MIG`: whole GPUs free/total, then free MIG
slices, which are not whole GPUs; nodes planned by the backfill scheduler included and marked), the next pending job with its reason; for workstations one bar pair per GPU (utilisation, memory, temperature),
CPU/memory/load and your tasks. Below 100 columns the cards become one table row per host. A card's footer explains a
problem and the key that fixes it.

**Jobs view** (`tab`). Every job of every host in one table. By default it shows the *fresh* preset: active and queued
jobs plus anything that ended in the last 24 h (failed, cancelled and "ended" tasks stay visible). Long names keep the
part that tells sweep siblings apart (`orb-swp--tr…warmup--s43--a02` next to `orb-swp--tra…const--s43--a02`);
NODE / REASON puts a hint first (`log idle 47m · n-40-01`); at 80 columns it stays and LEFT and GPUS go first. The
header totals (`R 15  PD 5  F 5/24h`) count your jobs, like the cards; `u` in the Jobs view adds other users'. Under the table a
preview follows the cursor: metadata, resources, GPU usage, an ended job's cause chip and the last log lines (fetched
once after the cursor rests for a second; never a live stream).

**GPUs view** (`tab` again). Every workstation GPU (live from `nvidia-smi`) and one row per running Slurm job with
GPUs. Slurm values are per-job aggregates from accounting (`sstat`), refreshed every 60 s, and marked `*`.

**Job detail** (`enter`). Tabs: *Summary* (job, resources, GPU usage, live log; below 110×30 three summary lines and
the log), *GPU* (usage, per-node GPU indices; for a running Slurm job also live per-GPU readings, see *Live GPU
readings (Attach)* below), *Logs*, *Steps* (`sacct` steps with exit codes, MaxRSS and peak GPU per task, and the reason
the job ended — led by the failure line of the log tail, e.g. `torch.OutOfMemoryError`), *Procs* (task members; for a
running Slurm job the node's processes with their Slurm step) and *Raw* (every field as reported). Under
the title an ended job that did not succeed shows its **cause chip** and a pending job what to do about its reason;
the Summary's right column shows an ended job's **Efficiency** box instead of its GPU box (see *Why did it fail?*
below).

**Host detail** (`enter` on a host). Workstations: per-GPU bars, host CPU/memory, and processes (GPU holders
of all users plus the members of your tasks; `u` toggles other users). Click a GPU row (or step with `[` / `]`) to
list only the processes on that GPU, whoever owns them; click it again for all processes. Clusters: partitions with free GPUs by type,
nodes (state, CPU, memory, GPU use, which of your jobs run there) and the detected capabilities. A host that never
connected shows only its connection panel (`c` log in, `e` details, `t` disable).

**Log viewer** (`l`). The job's stdout or stderr (`o`/`e`), following the file as it grows (`F` toggles). Progress
bars that rewrite one line with carriage returns (tqdm) show up as a live line under the log instead of flooding it.
Search the log in your pager with `p` (`less` on the remote host when available, else a downloaded copy). `/` searches
the loaded buffer (`n`/`N`); `]`/`[` jump to the next / previous error or warning line (Traceback, `…Error`, CUDA out
of memory, NCCL errors, `Killed`, OOM kills, `slurmstepd: error`, `DUE TO TIME LIMIT`, preemption, segmentation
faults, NaN losses, disk quota), which are shown in red (warnings in yellow) here and in the job detail log panes.

**Command palette** (`ctrl+p` or `:`). Fuzzy commands: switch views and scope, open a host, connect, refresh,
enable/disable hosts, state presets, sort/group, *Doctor &lt;host&gt;*, *Settings*, *Action log*, *RunFleet log*,
*Log in to all hosts needing auth*, *Forward a port…*, *Forwards*.

**Settings** (`,`, the `⚙` at the top right, or the palette). A panel with three sections (`tab` or `←`/`→`, or click
a tab):

- *Hosts* lists every alias of `~/.ssh/config` together with the hosts in your config and the ones you named on the
  command line. `[x]` means saved in `config.toml`. `space` (or a click on the box) on an unchecked host saves it and
  starts polling it right away — a card appears; on a checked host it stops polling and removes it from the config.
  You press `space` twice when the host shows jobs of yours (the jobs keep running), when its table has settings of
  its own beyond `name`/`ssh`, or when it is saved but not part of this run (`runfleet k2` shows only k2); checking
  it again before you quit puts the removed table back unchanged. `p` on such a saved host that this run left out
  starts polling it now, with its saved settings, without touching the file. `t` tests the highlighted alias
  read-only and shows what it is (Slurm version, workstation, needs a login, the error). `a` adds a target that is
  not in your ssh config (`user@host`, a plain host name; if its name, `user_host`, is taken already, a `_2` suffix
  keeps the two apart). `J`/`K` move a saved host down/up: that is the card order
  and the number keys `1`–`9`. `/` filters a long list.
- *General*: default view, refresh speed (slow / normal / fast; shows the speed in effect, so after `+`/`-` it
  follows them), read-only, mouse, idle pause, remember hosts, and a *notifications* group: enabled (off: alerts
  still go to palette › *Notifications*, nothing pops up), terminal (auto / osc9 / osc777 / bell / off) and events
  for all your jobs (*failures* = failed, restarted, low on time — the default; *+gpu idle*; *all*; a list you wrote
  by hand shows as `custom` and stays until you change that row). `space` steps to the next value; click a value to
  pick it. Changes apply at once and are saved in `[general]` / `[notify]` (the default view takes effect at the next
  start). The notification command hook is set in config.toml only; the group heading says whether one is set.
- *About*: the config file (`enter` or `e` opens it in `$EDITOR`), the log, the state directory, the version.

Your comments and formatting in `config.toml` are kept. If the file cannot be written (a read-only dotfiles
directory, a full disk) the panel says so and nothing changes. With `--demo` nothing is saved (the panel says so).

## Filtering, sorting, grouping

`/` opens the filter. Terms are separated by spaces and must all match:

| Term | Meaning |
|---|---|
| `word` | fuzzy match against host, id, name, state, partition, nodes, reason, user |
| `host:k2`, `h:k2,vega` | host is one of these |
| `state:R`, `st:pd,f`, `state:failed` | state code (`R PD CG F TO OOM CA CD …`) or group (`queued active done failed cancelled ended`) |
| `user:me`, `user:alice` | owner |
| `!term` | negation of any term |

`f` cycles the state presets *fresh → active → queued → failed (failed + ended) → all*; `o` cycles the sort column and
`O` reverses it (or click a column header); `z` groups by nothing → host → array. Scope with `0` (all hosts) and `1`–`9`
(host N in config order). Filter text, preset, sort and grouping are remembered in `~/.local/state/runfleet/ui.json`.

## Live GPU readings (Attach)

Slurm accounting only knows a job's GPU *sum* (`sstat`, refreshed about once a minute): `Σ100% / 2 GPUs` can be one
busy GPU and one idle one. Open the **GPU** or **Procs** tab of a *running* Slurm job and RunFleet reads that job's
node directly — per-GPU utilisation and memory, temperature and power, the processes on each GPU, the job's cgroup
(memory, peak, shmem, `oom_kill`, CPU cores in use) and `ps` with each process's step. Nothing is read in the
background: only while one of those two tabs is on screen, for that one job. Leaving the tab, opening the log viewer
or closing the job stops it; dialogs on top (help, the srun question) do not.

```
 Live · ssh n-38-01 (job 723954) · 15:00:00 · 0.67 s · every 15 s, next 9 s · clean ✓       r now · a auto on/off
  IDX  bus      util                    mem                              temp      power
    0  89:00.0  ▕███████████████▏ 100%  ▕█████████▌     ▏    89.5/140G   73°C   576/600W
       ↳ 129392 VLLM::EngineCore 54.5G · 129361 python3 -m lmft.train --cfg… 34.9G
    1  C5:00.0  ▕               ▏   0%  ▕████████▉      ▏    82.7/140G   34°C    94/600W  ! idle but holding 82.7G
       ↳ 129361 python3 -m lmft.train --cfg… 82.7G
 cgroup  mem 134G / 256G (peak 188G, shmem 56G) · oom_kill 0 · CPU ≈ 3.6 / 8 cores
 accounting (sstat, 30s ago)  Σ170% / 2 GPUs · gpumem 120G
```

How it reads the node depends on the host's `attach` setting (`docs/CONFIG.md`, *Live readings*):

- **`ssh`** (default; clusters with pam_slurm_adopt, like `k2` and `vega` in the demo): `ssh <node>` from the login
  host. It leaves no trace in Slurm accounting and refreshes every `attach_refresh` (15 s) while the tab is open, the
  terminal has the focus and you have pressed a key in the last 10 minutes (`Paused` / `Stopped` say which rule held
  it; `r` resumes). `a` turns the auto-refresh off and on; `r` reads at once. `IDX` is Slurm's GPU number (`?` when it
  cannot be known for sure — never a guess).
- If the node puts the ssh session into **another of your jobs** (a node runs several of them; ssh can only enter the
  newest), the tab says so: `ssh landed in job 2332863 …; GPUs shown belong to 2332863`. The cgroup and processes of
  the job you opened are still right. With `attach = "ssh+srun"`, `S` reads the job's own GPUs with srun.
- **`srun`** (clusters that refuse ssh to compute nodes, like `lyra` in the demo; or `ssh+srun` as fallback): a
  one-shot `srun --overlap` step inside the job. Each read leaves **one permanent row** (`<job>.N rfprobe`) in the
  job's `sacct` history, so it never happens by itself: `r` asks first, showing the exact command, with **Once**,
  **Allow for this job (this session)** (no more questions for this job until RunFleet exits; every read still adds a
  row) and **Cancel**. The status line counts the rows added this session. There is no "don't ask again".
- **`off`**, or `--readonly` where srun is the only way: the tab says why (`Live readings unavailable · …`) and, when
  a setting would help, the exact line to add and what it costs. `--readonly` still allows ssh reads.

`clean ✓` means the read left nothing behind (no probe step, no ssh client, no stuck `nvidia-smi`); a red
`leftovers` badge names what remained. Hints: a GPU that held memory at < 5 % utilisation for two reads in a row
(`! idle but holding 82.7G`), `oom_kill > 0` (red), cgroup memory above 90 % of the job's limit (shmem counts).
On the Procs tab, processes in D state (stuck in I/O, often a hung `nvidia-smi`) are amber and zombies red.

## Keys

Every view lists its keys with `?`. The table below is generated from the same source as the help overlay
(`runfleet.ui.keymap`); `scripts/gen_docs.py` rewrites it.

<!-- keymap:begin -->
| Context | Keys |
|---|---|
| Global | `?`/`f1` help (this view) · `^p`/`:` command palette · `,` settings (hosts, defaults) · `q` back · quit on the main screen · `^c` quit now · `esc` close / clear filter, marks / back · `0` all hosts (1-9: host N) · `/` filter jobs · `r` refresh hosts in scope (>= 3 s apart) · `^r` refresh all hosts · `+` poll faster (down to 0.5×) · `-` poll slower (up to 4×) · `^z` suspend to the shell · `D` since you were away: what ended, failures first |
| Main screen | `tab` next view (Fleet→Jobs→GPUs) · `S-tab` previous view |
| Fleet view | `↑`/`k` host above · `↓`/`j` host below · `←`/`h` previous host · `→`/`l` next host · `↵` host detail · `space` scope to this host · `c` connect (log in: Duo / password prompts) · `C` log in to every host needing auth (one terminal session) · `s` shell on the host · `t` enable / disable polling · `e` error details |
| Jobs view | `j` move down · `k` move up · `g`/`home` first row · `G`/`end` last row · `^d`/`pgdn` half page down · `^u`/`pgup` half page up · `↵` job detail · `l` log viewer · `p` log in $PAGER · `x` action menu · `^k` cancel job / terminate task · `space` mark / unmark · `V` mark all / clear marks · `A` mark the whole array of this row (cancel / hold act on all of it) · `y` copy id · `Y` copy menu: o stdout · e stderr path · i id · h host:id · w workdir · s shell command · b support bundle · Y again = stdout path · `s` shell · `P` forward a port (TensorBoard, Jupyter …) to localhost · `c` connect host · `/` filter jobs · `f` next state preset · `o` next sort column · `O` reverse sort · `z` group by · `u` other users · `w` watch / unwatch (notify on start, end, GPU idle; marked rows too) |
| GPUs view | `j` move down · `k` move up · `g`/`home` first row · `G`/`end` last row · `↵` owning job / task (or host) · `x` actions of the owning task · `^k` terminate the owning task · `c` connect host · `a` copy CUDA_VISIBLE_DEVICES for N free GPUs of this workstation · `o` next sort column · `O` reverse sort · `/` filter jobs (Jobs view) |
| Job detail | `[` previous tab · `]` next tab · `o` stdout · `e` stderr · `F` follow on / off · `l` full log viewer · `L` log viewer at the failure line (cause chip) · `/` search the log (opens the log viewer) · `p` log in $PAGER · `r` refresh detail · `x` action menu · `^k` cancel / terminate · `s` shell · `c` connect host · `y` copy id · `Y` copy menu: o stdout · e stderr path · i id · h host:id · w workdir · s shell command · b support bundle · Y again = stdout path · `P` forward a port (TensorBoard, Jupyter …) to localhost · `w` watch / unwatch (notify on start, end, GPU idle) · `a` live readings: auto-refresh on / off (GPU / Procs tab) · `S` live readings: read this job's own GPUs with srun (asks first; one permanent sacct row) |
| Job detail · GPU / Procs tab (running Slurm job) | `r` read the node now (srun asks first) and refresh the detail · `a` auto-refresh on / off (ssh: every attach_refresh while the tab is open) · `S` read this job's own GPUs with srun when ssh landed in another job (asks first) |
| Host detail | `^k` terminate the selected task · `x` actions · `s` shell · `c` connect · `t` enable / disable polling · `e` error details |
| Host detail (workstation) | `↵` open the process' task · `l` log of the selected task · `u` other users' processes · `[` previous GPU: only its processes, all users (click a GPU too) · `]` next GPU · `a` copy CUDA_VISIBLE_DEVICES for N free GPUs · `o` next sort column · `O` reverse sort |
| Host detail (cluster) | `↵` my job on the selected node · `f` nodes with free GPUs only · `[` previous partition · `]` next partition |
| Log viewer | `F` follow on / off · `G` bottom + follow · `g` top of the buffer · `o` stdout · `e` stderr · `/` search the buffer · `n` next match · `N` previous match · `[` previous error / warning line · `]` next error / warning line · `p` log in $PAGER · `R` reopen the log stream · `y` copy the path of the shown stream (stdout / stderr) · `c` connect host |
| srun confirmation (live readings) | `y` read once (one srun step, one permanent sacct row) · `a` allow srun reads for this job until RunFleet exits (each read still adds a row) · `n`/`esc` cancel · `f1` help (this dialog) |
| Confirm dialog | `y` confirm · `n`/`esc` cancel · `tab` move between params and buttons · `f1` help (this dialog) |
| Action menu | `j` move down · `k` move up · `↵` run the action · `letter` pick the entry by its key (copy menu: Y again = stdout path) · `esc`/`q` close · `f1` help (this dialog) |
| Help | `esc` close · `?` close · `f1` close |
| Host error | `c` connect (log in) · `r` retry now · `t` enable / disable the host · `esc`/`q` close · `f1` help (this dialog) |
| Notifications | `j` move down · `k` move up · `↵` open the job · `esc`/`q` close · `f1` help (this dialog) |
| Since you were away | `j` move down · `k` move up · `↵` open the failed job · `f` show everything that ended since then in the Jobs view · `esc`/`q` close · `f1` help (this dialog) |
| Port forwards | `j` move down · `k` move up · `↵` copy the local URL again · `^k` cancel the forward (ssh -O cancel, local) · `esc`/`q` close · `f1` help (this dialog) |
| Prompt | `↵` use this value · `esc` cancel · `f1` help (this dialog) |
| Dialogs | `esc`/`q` close · `f1` help (this dialog) |
| Settings | `tab`/`→` next section (Hosts → General → About) · `S-tab`/`←` previous section · `↓`/`j` move down · `↑`/`k` move up · `space`/`↵` toggle host (save + poll / remove) · change value · open · `t` test the host now (read-only probe) · `a` add an ssh target that is not in ~/.ssh/config · `p` poll a saved host this run left out (its saved settings, file unchanged) · `J`/`S-↓` move host down (card order, number keys) · `K`/`S-↑` move host up · `/` filter hosts · `e` open config.toml in $EDITOR · `?`/`f1` these keys · `esc`/`q`/`,` close |
<!-- keymap:end -->

**Mouse.** Click selects rows, cards, tabs and header chips; the `RunFleet` logo at the top left goes back to the Fleet view
from any screen; double-click drills in; right-click opens the action menu;
clicking a column header sorts. To select text hold `Shift` while dragging (most terminals), or start with
`--no-mouse`; `y` copies the id and `Y` opens the copy menu (below) without selecting. Copying uses OSC 52, which the
terminal never acknowledges, so the toast says what was sent and shows the text for you to select if nothing arrived:
iTerm2, kitty, WezTerm, Alacritty, Windows Terminal and foot accept it (some only after enabling clipboard access); in
tmux set `set -g set-clipboard on` (and `set -g mouse on` for the mouse); inside tmux the copy menu also loads the tmux
paste buffer (`tmux load-buffer -w`). Clicking a hint at the bottom acts like its key in the view, also while the
filter has the focus.

Inside a dialog (confirm, action menu, copy menu, host error, doctor, notifications, since you were away, port
forwards) only its own keys work, plus `F1` for a list of them; the palette, scope and refresh keys wait until it is
closed. The help overlay searches whatever you type (`?`, `F1` or `esc` close it).

## Actions

| Action | Applies to | Remote command |
|---|---|---|
| Cancel (`ctrl+k`, menu) | your pending/running Slurm jobs, array summaries included | `scancel <ids>` |
| Hold / Release | your pending jobs / your held jobs | `scontrol hold` / `scontrol release` |
| Requeue (optionally held) | your running jobs; ended ones only within the controller's memory (5 min) | `scontrol requeue[hold]` |
| Terminate (`ctrl+k` on a task) | your workstation tasks | pause the launcher, SIGTERM the whole tree as it is *now*, SIGKILL survivors after the grace time (default 10 s) |
| Interrupt | your workstation tasks | SIGINT to the tree (like Ctrl-C), no escalation |
| Kill | your workstation tasks | SIGKILL to the tree |

Every action opens a confirmation that lists the targets, the jobs that will be skipped and why, and the exact command;
*Cancel* has the focus, `y` confirms. Marked rows (`space`, `V`) are acted on together, one dialog for all hosts.
Results appear as a toast and in the palette's *Action log*; every action is also written to `runfleet.log`.

Actions are never offered for other users' jobs or processes, for jobs in `protected_partitions`/`protected_qos`, or
when the data is stale ("data is 4m old — refresh first"). With `--readonly` (or `readonly = true`) they are listed
but disabled. Task signals re-read the process tree on the host at the moment you confirm and pause the launcher first,
so a queue runner cannot start its next item in between; the result names any process that survived or appeared.

Not actions (no confirmation): `s` opens a shell, `p` the log in a pager, `l` the log viewer, `c` connects.
Where `s` lands: a host → its login shell; a task → its working directory; a **running Slurm job → its compute node,
inside the job** (`srun --jobid=<id> --overlap --pty`, so `nvidia-smi` shows the job's GPUs; a multi-node job asks which
node, batch host first). It works on clusters whose sshd refuses compute-node logins (`lyra` in the demo). Leaving the shell ends only
that interactive step, never the job; the step stays visible in `sacct` as one extra row. Pending / finished jobs → the
login node in the job's `WorkDir`.

## Everyday shortcuts

None of these changes a job, so they work in read-only mode too.

- **Copy menu** (`Y`, Jobs view and job detail): stdout path (`o`, highlighted, so `↵` takes it too) and stderr path
  (`e`), job id (`i`), `host:jobid` (`h`), workdir (`w`), the shell command `s` would run as something to paste into
  another terminal (`s`; `ssh -t k2 'exec sh -c …srun --jobid=… --pty…'`: lands on the job's node), and a **support
  bundle** (`b`). Each entry shows its key in front; pressing it copies at once, and **`Y` again copies the stdout
  path**, so `Y Y` does what `Y` alone did before. An entry with nothing to copy stays greyed with the reason, and
  its key only repeats the reason. The **support bundle** is a Markdown report for Slack or a ticket with the cluster, job,
  user, partition / QOS / account, the request, state and exit with the explanation, the cause and efficiency lines,
  nodes with GPU indices, restarts and the last 30 log lines. Secrets are scrubbed (wandb keys, `hf_…`, `sk-…`,
  `ghp_…`, AWS keys, `token=` parameters, private keys, `Bearer …`, `user:password@` in URLs, values of `…PASSWORD`,
  `…SECRET`, `…TOKEN`, `…API_KEY` settings) — still read it before you paste it anywhere; a copy is kept in `~/.local/state/runfleet/bundles/HOST-ID.md`. Building
  it asks the host for the job detail and the log tail once.
- **Forward a port** (`P`, Jobs view and job detail; palette › *Forward a port…*): TensorBoard, Jupyter or Gradio
  running in a job, opened on this machine. RunFleet looks for `http://host:port/…` URLs in the log (what it already
  holds, else the last MiB of the log, fetched once) and offers them plus *Other port…*. It forwards
  `localhost:<port>` (the same number when free here) to the job's node through the existing ssh master
  (`ssh -O forward`; nothing runs on the cluster) and copies `http://localhost:<port>/…` with Jupyter's token kept.
  The service has to listen on the node's network address (`tensorboard --bind_all`, `jupyter lab --ip=0.0.0.0`);
  on a workstation the target is the workstation itself. Palette › *Forwards* lists them (`enter` copies the URL
  again, `^k` stops one; a forward ssh could not stop stays listed); all are stopped when RunFleet exits. After a
  reconnect `P` on the same port sets the forward up again on the new master. Not available without a master (a host that needs a
  login, `mux = "user"` without a ControlPath, `mux = "off"`).
- **Log in to all** (`C` in the Fleet view; palette › *Log in to all hosts needing auth*): after the overnight
  disconnect, one trip to the terminal logs in to every host that needs it or is unreachable, one after another
  (`2/3 vega — approve the Duo push`; `Ctrl-C` skips a host), then RunFleet resumes them all and says which worked.
- **Free GPUs** (`a` in the GPUs view on a workstation row, or in a workstation's host detail): a menu `1 / 2 / …
  free GPUs` copies `CUDA_VISIBLE_DEVICES=0,2` for the idlest GPUs without processes (under 1 GiB used and 5 %
  busy, least memory first); the toast says who holds the others. Stale readings are refused (the host, or just its
  `nvidia-smi` / `ps` answers), and GPUs taken while the menu was open are not copied.
- **Whole array** (`A`, Jobs view): marks every visible row of the cursor row's array (the pending summary included),
  so `^k` and the action menu act on the whole array; `A` again clears them.

## States and hints

`ST` shows the Slurm state code (`PD` pending, `R` running, `CG` completing, `CD` completed, `F` failed, `TO` timeout,
`OOM` out of memory, `PR` preempted, `CA` cancelled, …), `PDH` for held jobs. Workstation tasks are `R` while their
root process lives and become `F` (the log's last lines contain a Traceback, an OOM or a crash) or `EX` ("ended, exit
status unknown") when it is gone. Colours: active green, queued yellow, failed red (preempted orange), done dim blue,
cancelled magenta, ended grey.

Hints flag what needs a look: `log idle 47m` (no log output for `stall_after`, 30 min), `GPU idle 20m` (utilisation
below 5 % for 15 min), `will never run: dependency failed`, `per-user job cap (QOS)`, `time limit above partition max`.
The Steps tab explains endings: exit code 143 = SIGTERM, 137 = SIGKILL, "preempted; Slurm will requeue it" vs "preempted
and killed; resubmit", "exceeded --mem", "hit time limit".

Pending jobs explain themselves: every common Slurm reason (`Priority`, `Resources`, `Dependency`, the `Assoc…` and
`QOS…` limits, `PartitionTimeLimit`, `ReqNodeNotAvail`, held jobs, `BadConstraints`, `InvalidAccount`, …) has a short
form, a red "will never run" where it never will, and one actionable line under the job detail title and in the
preview (`→ you hold as many GPUs as this QOS allows per user; starts when one of your running jobs ends`). When
squeue has a start estimate it is shown as Slurm's backfill guess: it moves when running jobs end early or
higher-priority jobs arrive, and `Priority`-pending jobs often have none.

`LEFT` turns amber when less than 10 % of the time limit or 30 minutes are left and red below 2 % or 5 minutes (for short jobs the minute floors shrink to a quarter / a tenth of the limit, so a 30-minute job is not amber all along). `↻2`
before a name means Slurm restarted the job twice (requeue, preemption); `◉` marks a job you watch.

## Why did it fail? Was the request right?

**Cause chip.** An ended job that did not succeed gets one short cause: `OOM-HOST` (Slurm's OOM kill or `oom_kill`
in the log), `OOM-GPU` (CUDA / torch / JAX out of memory), `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`, `NCCL` (NCCL error,
timeout or watchdog), `IMPORT` (`ModuleNotFoundError` / `ImportError`), `QUOTA` (disk quota exceeded, no space
left), `SEGV`, `NAN` (loss is nan / inf), `SIGNAL n`, `CANCELLED` or `EXIT n`. The scheduler's state decides first,
then the log (the most telling match: one rank's GPU OOM shows up as NCCL timeouts on the others, so `NCCL` only wins
when nothing more specific is there), then the exit status. Job detail shows the chip under the title with the
first log line that shows it (hover for the full explanation and what to try); `L` opens the log viewer on that
line. The Jobs preview shows the chip too. Only log text RunFleet already has is read (the detail's log stream, the
preview's one-shot peek): nothing extra runs on the cluster.

**Efficiency box.** For an ended Slurm job the Summary shows, seff / reportseff style:

```
time  4:48:00 of 1-00:00:00 (20%)
cpu   50% of 8 cores (CPU time 19:12:00)
mem   peak 18G of 48G (38%)
gpu   92% mean over 1 GPU (sacct gres/gpuutil)
next  --time=6:00:00 --mem=24G
```

CPU efficiency is `TotalCPU / (Elapsed × AllocCPUS)` (TotalCPU comes from the detail's own `sacct` steps query when
the cluster's sacct knows the field, else AveCPU × NTasks); peak memory is the largest MaxRSS of a step times its tasks
on a single-node job (MaxRSS is per task and sampled); GPU is `gres/gpuutil` from accounting where the cluster records
it (`k2` and `vega` in the demo; `lyra` shows `–`). Amber means the request was oversized: CPU below 50 %, memory below 25 %, time below
25 % (time only for completed jobs). `next` suggests `--time` and `--mem` with a 25 % margin, rounded (never more than
what worked); a time-out gets a +25 % `--time` guess, an OOM kill a +50 % `--mem` guess, and a run that failed early
none. A note warns when the GPUs looked idle (mean below 5 %). Running jobs show live memory headroom (`room  rss 11G
of 256G (4%) · 245G headroom`, red above 90 %) in the GPU box instead. Below 110×30 the card is one `eff …` line.

## Notifications, watching, "since you were away"

Leave RunFleet running (a tmux pane is fine) and it tells you when one of your jobs **fails** (failed, timeout, out of
memory, node failure, preempted), **restarts**, or is about to **hit its time limit** (10 % or 10 minutes left): a
toast in RunFleet, a desktop notification through the terminal, and optionally a command of your own. Many changes on
one host at once become one line (`k2: 12 failed, 3 completed`). Palette › *Notifications* lists every alert of the
session; `enter` opens the job. Nothing extra runs on the clusters for this.

`w` (Jobs view, job detail) **watches** the job under the cursor, or all marked jobs; `w` again stops. For a watched
job RunFleet also tells you when it starts, completes or its GPUs sit idle. Watches are kept in `ui.json` and
forgotten a week after the job is gone. A watched pending array covers its tasks.

**Since you were away.** When you start RunFleet again (or press the first key after `idle_after`, 10 minutes, without
input), it waits until the hosts have answered and shows one note: `since 21:40: 3 failed · 12 completed · 1 timeout
— D to review`. `D` (anywhere) opens the list: counts per host and the failures with the reason, `enter` opens one,
`f` shows everything that ended since then in the Jobs view (`esc` returns to the normal preset).

Terminal notifications: iTerm2, WezTerm, Windows Terminal, ghostty and kitty get OSC 9, foot, urxvt and VTE terminals
(GNOME Terminal, Tilix) OSC 777, others the bell. Inside tmux add `set -g allow-passthrough on` to `~/.tmux.conf`
(tmux 3.3+), else tmux drops them. To get a phone or desktop notification another way, set `[notify] command`, e.g.
`command = 'ntfy pub my-runs "$RUNFLEET_SUMMARY"'` ([CONFIG.md](CONFIG.md#notify) lists the variables and the event
selection). Settings › General › *notifications* switches pop-ups off, picks the terminal channel and widens or narrows
the events without editing the file. `--demo` watches one job and pretends you were last here 8 hours ago, so all of
this shows within minutes.

## When a host is in trouble

| Header chip / card | Meaning | What to do |
|---|---|---|
| `✖ auth` (magenta) | the background `ssh` needs a login (key, Duo, password, expired Kerberos); polling is paused | press `c` (or run `runfleet connect HOST`) and log in; RunFleet resumes |
| `✖ hostkey` | unknown or changed host key | unknown: `c` lets ssh ask; changed: check it, then `ssh-keygen -R HOST` |
| `◐ retry 38s` (yellow) | 1–2 failed connection attempts, backing off | nothing; `r` retries now |
| `✖ unreachable` (red) | 3 failures in a row, or DNS failure; retries every ≤ 5 min | check the network/VPN; `e` shows the ssh error and log |
| `!` (yellow) | connected, but a part failed (`slurmdbd` down, `nvidia-smi` hung, unparseable output) | the host screen lists the failing parts; the rest keeps updating |
| rows marked `~`, dimmed | data older than two refreshes | it refreshes itself; the age is shown |

In a narrow terminal the header shortens chips to `✖lynx:a` or `✖5:a`: `:a` needs auth, `:k` host key, `:d` down, so
the state reads without colour too. A new notice appears whenever a host moves to a state that needs you (an
unreachable host that comes back asking for a login says so).

**Colours.** RunFleet picks a light palette when `COLORFGBG` reports a light background; terminals that do not set it
(macOS Terminal, many others) get the dark one, so on a white profile start it with `RUNFLEET_THEME=light` (or `dark`
to force the opposite). `NO_COLOR=1` drops colour for attributes (bold, reverse); an empty `NO_COLOR` counts as unset.

`runfleet doctor HOST` prints the full diagnosis in a plain terminal. `e` on a host opens the same information in the
TUI, with *Connect*, *Retry now* and *Disable host*.

## Load on shared machines

A visible Slurm host costs 4 SSH round trips and at most 7 Slurm controller/database calls per minute; hidden hosts
a third of that. Workstations are polled every 5 s while visible. After 10 minutes without input (or when the terminal
loses focus) everything polls 4× slower; after 2 hours without input polling stops until you press a key. `+`/`-`
change the pace (floors always apply). Background polling never runs anything on compute nodes and never starts job
steps; only the interactive `s` shell into a running Slurm job does (one `srun --overlap` step, visible in `sacct`).
When a Slurm controller is busy (`squeue` / `sacct` / `sinfo` report `Socket timed out`, `Rate limit`, … or answer
three times slower than usual and over 5 s) RunFleet polls that cluster 2× slower per busy round, up to 8×, and speeds
up again after a few healthy rounds; the host card and `e` say `controller busy — slowed 4×`.

## Files

| Path | Purpose |
|---|---|
| `~/.config/runfleet/config.toml` | your configuration (optional); Settings (`,`) edits it, keeping your comments |
| `~/.local/state/runfleet/cache.sqlite` | warm start: last snapshots, capabilities, job metadata (`--no-cache` to skip) |
| `~/.local/state/runfleet/ui.json` | view, sort, filter, preset, grouping; watched jobs and the last visit |
| `~/.local/state/runfleet/runfleet.log` | log (5 MB × 3); `--debug` adds `debug/` raw dumps; lines from `--demo` carry `[demo]` |
| `~/.local/state/runfleet/bundles/` | support bundles from the copy menu (`HOST-ID.md`, secrets scrubbed) |
| `~/.local/state/runfleet/ssh-HOST.log` | ssh's own messages when making a connection |
| `$XDG_RUNTIME_DIR/runfleet/` | RunFleet-managed ssh master sockets, per-host instance locks |

The state directory is private (0700, files 0600) whatever your umask: it holds job names, paths and command lines.
Two RunFleet instances polling the same host warn once about the doubled load; instances with different hosts do not.
