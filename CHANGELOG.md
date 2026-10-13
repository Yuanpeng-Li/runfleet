# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Clickable GPUs on workstations**: in a workstation's host detail, clicking a GPU row (or `[` / `]`) lists only
  the processes on that GPU, other users' included; the same GPU again shows all processes.
- **Header logo goes home**: clicking `RunFleet` at the top left closes detail screens and shows the Fleet view.
- **Notifications, watch list, "since you were away"**: your jobs that fail, restart or run low on time (and, for
  jobs watched with `w`, start, complete or sit GPU-idle) raise a toast, a terminal notification (OSC 9 / OSC 777 /
  bell, tmux passthrough) and an optional local `[notify] command`; bursts become one line per host, palette ›
  *Notifications* keeps the history. After a restart or a long idle spell one note says what ended meanwhile; `D`
  lists it and filters the Jobs view to it. LEFT turns amber/red as the time limit nears, `↻N` marks restarted jobs.
  No new remote commands; the first poll after a start or reconnect is a silent baseline. New config section
  `[notify]`; Settings › General has *notifications* rows (on/off, terminal channel, which events: failures /
  +gpu idle / all; a hand-written list shows as *custom* and is kept), applied live and saved to `config.toml`.
- **Failure triage and efficiency**: an ended job's **cause chip** (`OOM-HOST`, `OOM-GPU`, `TIMEOUT`, `PREEMPTED`,
  `NODE_FAIL`, `NCCL`, `IMPORT`, `QUOTA`, `SEGV`, `NAN`, `SIGNAL n`, `CANCELLED`, `EXIT n`) with the first log line that
  shows it, under the job detail title and in the Jobs preview; `L` opens the log viewer on that line. The log viewer
  highlights error / warning lines and `]` / `[` jump between them. Ended Slurm jobs get an **Efficiency** box (time,
  CPU = TotalCPU / (Elapsed × CPUs), peak memory, GPU util; reportseff colours) with a "next time" `--time` / `--mem`
  suggestion; running jobs show memory headroom. Every common pending reason has a short form and one actionable line,
  and squeue's start estimate is labelled a guess. No new periodic remote command (TotalCPU joins the on-demand detail
  steps query where sacct knows it).
- **Everyday actions**: `Y` copy menu (`y` still copies the id; `Y Y` copies the stdout path as `Y` used to, and
  each entry has a key: `o` stdout, `e` stderr, `i` id, `h` host:jobid, `w` workdir, `s` shell, `b` bundle): log
  paths, `host:jobid`, workdir, the node-shell command to paste, and a scrubbed Markdown support bundle (with the cause chip and efficiency card; also saved under
  `bundles/`), with a tmux buffer fallback. `P` / palette forwards a job's TensorBoard / Jupyter URL through the
  existing ssh master (palette › *Forwards*; all cancelled on exit). `C` logs in to every host needing auth in one
  terminal session, `a` picks free GPUs (`CUDA_VISIBLE_DEVICES=…`), `A` marks a whole array, and
  `runfleet wait HOST:JOBID … [--timeout] [--json]` serves scripts (exit 0/1/2/3). A busy Slurm controller (slurmctld
  socket timeouts, rate limits, answers 3× slower than usual) slows that host's polling ×2 … ×8 with a "controller
  busy" note.

- **Live GPU readings (Attach, v0.2)**: the job detail's GPU and Procs tabs of a running Slurm job read its compute
  node on demand: per-GPU util / memory / temperature / power with the Slurm IDX and bus, the processes holding each
  GPU, the job's cgroup memory (current / peak / limit, shmem, oom_kill) and CPU cores, every process of yours on the
  node with its step. Hints: a GPU idle but holding memory (two samples in a row), oom_kill > 0, memory above 90 % of
  the limit. Per-host `attach = "ssh" | "ssh+srun" | "srun" | "off"` (default `ssh`) and `attach_refresh` (15 s,
  floor 10 s). `ssh` enters the job through pam_slurm_adopt and leaves no record; `srun` (the only way where compute
  nodes refuse ssh) runs a one-shot `srun --overlap` step that leaves one permanent sacct row per read, so it only runs on `r`
  after a confirmation (once / this job for the session), never from a timer and never under `--readonly`. The
  background loop still never touches compute nodes; reads stop when the tab closes, the app is idle or in the
  background, the job ends, or after 10 minutes without input. Nothing is written or left running on any server:
  no temp files, no ControlMaster, no known_hosts writes, timeouts on both sides of the hop, and a final audit
  (`clean ✓`, a red `leftovers` badge, or inconclusive). The nvidia-smi stacking guard (workstations too) now
  counts only stuck or abandoned queries, not a job's own `nvidia-smi -l`.

### Fixed
- **Attach on nodes you never ssh'ed to**: the login → node hop required the node's host key to be in the login host's
  known_hosts already, so on clusters that ship no node keys (and run `StrictHostKeyChecking no` themselves) live
  readings failed with "host key is not known". A node key that is recorded must still match; an unknown node is now
  accepted without being written to any known_hosts file.
- `s` on a **running Slurm job** opened a login-node shell. It now lands on the job's compute node inside the job
  (`srun --jobid --overlap --pty`, the job's GPUs visible; also where sshd refuses node logins). Multi-node
  jobs ask which node. Pending / finished jobs keep the login-node shell in `WorkDir`.
- A sorted narrow column's header lost its label (`ST` became `… ▲`); it now reads `ST▲`.

### Fixed (code review, 2026-10)
- Nothing left behind on servers: a timed-out or cancelled local ssh is killed with its whole process tree; the task
  signal script resumes a stopped process if it dies halfway, and re-checks process identity (pid + start time)
  before each KILL so a recycled pid is never signalled; a losing duplicate `ssh -fN` and failed `-O exit` masters
  are terminated; `mux = "user"` without a ControlPath no longer starts stray `-fN` logins.
- Probes: an inherited `errexit` no longer aborts them; an rc-file banner without a newline cannot swallow the first
  marker; stderr is kept when `/tmp` is full; a connection cut mid-section reads as incomplete, never as an empty
  (authoritative) answer; truncated output is an error instead of making jobs vanish; action probes get their full
  timeout. ssh errors are classified more precisely (auth vs. remote exec failure vs. mux socket, banner timeouts,
  multi-hop jump chains).
- Slurm: cancelling a summarised array (`500_[1-99:2%5]`) cancels exactly its listed tasks (all pending tasks only
  when Slurm truncated the list, and the confirm dialog says so); requeued jobs and node changes refetch their
  details; vanished-job accounting and unreachable `scontrol` retry with backoff; sstat samples every running job
  within a few rounds; GPU GRES/TRES parsing handles typed, multi-entry and MIG strings; hostlist expansion is bounded.
- Workstations: a zombie task root ends the task; late `exec >train.log` redirects are picked up; paths with `|` no
  longer corrupt fd parsing; a hung `nvidia-smi` reports its real error and keeps old process lists only briefly;
  partial GPU memory is noted as a lower bound; a task shell whose directory is gone opens a labelled home shell.
- Logs: log sizes/mtimes follow symlinks (`latest.log -> run/train.log`); preview tails keep the last tqdm progress
  state; concurrent stream opens respect the per-host stream limit; streams that end or fail say so and reconnect with
  backoff (`R` in the log viewer reopens at once); follow pauses when you scroll up and resumes at the bottom; job
  detail and the log viewer pick up a log path that becomes known or changes on requeue.
- Core: stale data now disables actions when the job section keeps failing; actions re-check read-only mode and the
  host right before running; a removed host is fully torn down even if Settings closes mid-removal; `add_host` refuses
  a name already used by another ssh target; backoff arithmetic cannot overflow; a departed collector no longer keeps
  a host degraded; SQLite commits no longer block the UI, and a locked cache is retried instead of moved aside.
- Config/CLI: host names made only of dots are rejected; `nan`/`inf` are rejected in config and never reach
  `snapshot --json`; `connect` works with ssh aliases that differ from the name; `doctor`/`snapshot` skip disabled
  hosts unless named; `snapshot --timeout` validates its value; option abbreviations are no longer accepted; broken
  pipes exit quietly (141); removing a host keeps its trailing comments.
- TUI: bulk actions show and report the hosts they skip, and skipped/failed jobs stay marked; folded groups are never
  acted on; Confirm stays visible at 80×24 and validates its number fields; old GPU readings are marked (`~`) and never
  counted as free; the GPUs view shows only my Slurm jobs unless `u` is on; job detail shows the newer GPU sample
  with its age, partial-detail errors and failed steps; cursors survive refreshes and resizes; Home/End/PgUp/PgDn
  move the table cursor; `c` logs in from the GPUs view, job detail and the log viewer, and a shell or pager on a host
  waiting for a login resumes its polling; F1 works in dialogs; Help no longer closes on `q`; compact header chips
  read `:d`/`:a`/`:k`; a light-terminal host palette; clipboard toasts say OSC 52; Settings `p` polls a saved host
  this run left out.

### Added
- **Settings** overlay (`,`, the header `⚙` chip, palette › Settings), modelled on herdr's settings popup: section tabs,
  accent-pill selection, clickable tabs / rows / checkboxes / values / buttons, works down to 60×16.
  - *Hosts*: every `~/.ssh/config` alias with a checkbox, merged with the configured and ad-hoc hosts. Checking saves
    the host and starts polling it immediately (no restart); unchecking stops polling and removes it (a one-line
    confirm when it shows jobs of yours, has hand-tuned keys, or is not in this run; checking it again in the same
    session brings its table back whole). `t` tests an alias read-only with the result inline, `a` adds an ssh target
    that is not in `~/.ssh/config`, `J`/`K` reorder (card order and number keys), `/` filters long lists. Keys typed
    right after `/` or `a` go into the input even when they arrive in one burst over a slow link.
  - *General*: default view, refresh speed (slow / normal / fast → `refresh_multiplier`, floors kept), read-only,
    mouse, idle pause, remember hosts — applied live where possible and saved. *About*: config file (open in
    `$EDITOR`), log, state dir, version.
  - `--demo`: works in memory and says "demo — not saved".
- Hosts named on the command line are remembered: `runfleet k2 vega` saves the ones not yet configured once they
  answer (or need a login), with one notice "saved k2, vega — manage in Settings (,)" (`general.remember_hosts`,
  default on). That run still shows only k2 and vega; afterwards plain `runfleet` includes them.
- Config edits keep comments and formatting (tomlkit): add / remove / reorder `[[host]]` tables (a comment block above
  a host moves with it; blank-line separators and a `# ---` section heading above the first host stay in place) and
  set `[general]` keys; validated like at startup, written atomically (mode kept, symlinks written through) under an
  flock shared by every RunFleet process; an invalid file is never overwritten, and an unwritable one (read-only
  dotfiles dir, disk full) is an error message in Settings or a notice for remembered hosts, never a crash. New
  dependency: `tomlkit`.
- Core: `FleetAPI` runtime host management — `add_host` (starts the poller), `remove_host` (cancels the poller, closes
  its log streams and transport, releases the instance lock), `reorder_hosts`, `set_general`, `saved_hosts`,
  `custom_host_keys`, `ssh_aliases`, `remember_hosts`, `config_path`.
- A config that exists but has no hosts starts the TUI in Settings › Hosts (on a terminal) instead of exiting.

### Fixed
- Mouse in terminal multiplexers such as herdr: RunFleet stays in cell mouse mode (no SGR-pixel reporting, no in-band
  resize reports). These multiplexers answer the in-band resize query but keep sending cell coordinates, so Textual
  divided every click by the cell size and all clicks landed near the top-left corner.

### Fixed (M3 field validation and review, 2026-10)
- ssh: every command names one literal ControlPath expanded by `ssh -G` — the ProxyJump rewrite changed `%C`, so behind
  a jump host `-O check` never found the managed master and each ensure leaked a persistent login (M3 F1); stale sockets are
  removed (or reported), a master that could not bind its socket is detected and its stray `-fN` terminated; the hop
  of a jump rewrite logs into the master log (hop auth failures now pause as NEEDS_AUTH); `ssh_options` reach every
  command and may override RunFleet's tunable defaults; `runfleet connect` resolves the mux plan first; manual retries
  bypass the 30 s master throttle; socket paths respect OpenSSH's temporary bind name (86-byte budget).
- Polling: focus/visibility changes no longer collapse transport backoff; forced groups survive a failed probe.
- Slurm: sacct log patterns are expanded (wrong log paths for every ended job, M3 F2); `%a`/`%b`/`%r`/`%S`/padding/
  backslash follow Slurm; pending array summaries no longer turn into ghost "final state unknown" rows; abbreviated
  array ids plan `scancel --state=PENDING` instead of crashing; newer `scontrol` keys no longer leak into log paths;
  het-job components parse; `SPECIAL_EXIT`/`EXPEDITING`/`RF` map to requeued; scontrol summary lines no longer fail
  other targets; format fallback only on format errors; rows split on `\n` only; sstat CPU/RSS shown for running jobs;
  free GPUs skip CPU-full nodes; MIG slices are labelled as such.
- Workstations: long user names are no longer truncated by `ps`; a terminal's foreground job is not SIGSTOPped; GPU
  tasks show their GPU indices instead of "cpu only"; fleet-card memory bars show capacity.
- `doctor` + `snapshot` stay within 3 ssh sessions per host (cached capabilities); `snapshot` includes log sizes;
  `--debug` writes raw probe dumps (wire format, rotated).
- TUI: no more highlight loop / freeze on `[`/`]` in cluster host detail; no crashes from key bursts or keys after `q`;
  NO_COLOR cursor rows and bars; light-terminal palette; `/` searches logs; filter without subsequence fuzz and with
  state names; consistent sort arrows; readable array ids; hidden marks are not acted on; stable scroll on reorder;
  hint bar keeps `? help` and labels `^k` by what it would do; help closes on `?`/`q` and has a legend; demo mode no
  longer writes the real `ui.json` or takes the instance lock, and uses fictional users and paths.

### Fixed (M3 re-run and release review, 2026-10)
- Packaging: the sdist is a whitelist (`src`, `scripts`, `docs/*.md`, screenshots, README, CHANGELOG, LICENSE) and
  never ships local development notes or raw probe dumps; the test suite and its anonymised field fixtures
  (`tests/fixtures/`) live in the git repository only.
- ssh: a local ssh/sh killed by a signal (`pkill ssh`, OOM killer) is a transient error with backoff, not the pausing
  "remote shell could not run sh -s"; probes whose timer ran out while the TUI was suspended (`c`/`s`/`p`) or stopped
  (`^z`) get one fresh budget instead of flipping the host to retry; log streams revive a gone shared/managed master
  first instead of falling back to a full BatchMode login.
- `hosts import --all` / the first-run picker: master creation counts against `max_parallel_ssh` (dozens of
  simultaneous logins behind one bastion exceeded sshd MaxStartups); one line per host as it finishes; the alias
  column fits long aliases; aliases already in the config are reported as such instead of "Wrote … (1 host)";
  unreachable hosts are listed with the command that adds them later; `hosts list` without a config suggests
  `runfleet` or `hosts import --all --write`.
- `snapshot`: workstation task logs have size/mtime (process.fds stats the paths it discovers — M3 F9 residual); the
  plain table counts my jobs and my GPUs (allocated GPUs on clusters; it showed 0) instead of every row.
- Privacy: the state dir and `debug/` are 0700 and the log (and rotations), cache, `ui.json`, ssh logs and dumps 0600
  regardless of umask; `--demo` log records carry `[demo]`, and the demo world's Slurm job ids no longer equal real
  ones.
- Cache: a failed sqlite write keeps the batch for the next flush (ended tasks and log paths were lost for good).
- Second-instance notice: per-host locks; only hosts polled by both instances are named (disjoint fleets: no notice).
- Slurm: running steps show "GPU (max per task)" from sstat (M3 F8 residual); multi-node/srun jobs estimate cores from
  the busiest non-batch step and show `≈ 61 cores` instead of a clamped `999%`; a bare `rtx6000` GPU type has unknown
  memory (on one probed cluster they were 96 GB RTX PRO 6000 Blackwell boards, not 24 GB Quadros).
- TUI: header totals count my jobs (as the cards do; `u` adds others'); free GPUs show MIG slices apart
  (`13/108 +11 MIG`) and BY TYPE merges types with the same label (two `A100-80G`); node names are cut from the left
  and NODE columns widen to the longest name (long `<cluster>-gpu-…` names); Jobs table: ID capped at 10, NAME keeps the part
  that tells sweep siblings apart, NODE / REASON gets room for reasons at 120 columns, shows hints first and outranks
  LEFT/GPUS at 80 columns, ELAPSED over a day reads `1-00:15`; job detail below 110×30 shows three summary lines and
  the log, and the title shortens times before the name; "Why it ended" leads with the log's failure line
  (`torch.OutOfMemoryError: …`), drops the duplicate `exit 1` and lists a merged log once; a never-connected host shows
  only its connection panel ("no data yet — lynx needs a login"); footers show `[/] tab`, `[/] partition`, `o/e`,
  `g/G top/bottom`; the help legend uses the configured recent window.
- Docs/scripts: README install works from a checkout (no `<owner>` placeholder); `smoke_real.sh` budgets a managed
  master login (`-fN` + its `-W` hop) separately from the ≤ 3 sessions (a cold host behind a jump host no longer
  fails), prints the real `--debug` dump directory and needs the hosts named explicitly; `--debug` help says every
  probe is dumped.

### Added
- M0 skeleton: package layout, frozen contracts, test scaffolding (DESIGN.md §11.3).
- Command line (`runfleet`): the TUI on the configured fleet, ad-hoc hosts (`runfleet k2 labstation`), a first-run
  prompt that imports hosts from `~/.ssh/config`, `connect`, `hosts list|import`, `doctor`, `snapshot [--json]`,
  `--demo`, `--readonly`, `--no-mouse`, `--no-cache`, `--debug`, `--version`; documented exit codes 0-4.
- `snapshot --json` export with schema `runfleet.snapshot/1`.
- Demo mode: a simulated fleet (Slurm clusters k2/vega/lyra, GPU workstations labstation/lynx) whose transport
  answers every probe section with byte-exact real `squeue`/`sacct`/`scontrol`/`sstat`/`sinfo`/`nvidia-smi`/`ps`/procfs
  output, runs actions against the simulation, streams growing logs (with tqdm carriage returns), and injects faults
  (unreachable, auth, hung nvidia-smi, garbage output, slurmdbd down) for tests.
- Docs: README, user guide (key table generated from the keymap), configuration reference.
- `scripts/smoke_real.sh HOST …`: read-only smoke test against real hosts that counts ssh sessions per host (manual,
  opt-in with `--yes`); `scripts/gen_docs.py`; `scripts/screenshots.py`.
- CI: lint, types and tests on Python 3.11 and 3.13, plus a packaging job that runs the demo from the built wheel.
