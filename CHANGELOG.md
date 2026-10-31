# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.9.0] - 2026-10-31

### Added
- **Find other log files of a job.** When a job's stdout holds only a launcher's few lines (`Popen(stdout=fh)` into
  `runs/<task>/<run>/train.log`), RunFleet can now list the log-like files written under the job's workdir while it
  ran and follow the one you pick. One bounded probe on the login node or workstation, on demand only: files named
  `*.log *.out *.err *.stdout *.stderr` (wandb's `output.log` included) at most 5 directories below the workdir
  (`a/b/c/d/e/train.log`), written since the job started and, for a job that ended, no later than 5 minutes after its
  end (what later jobs wrote in a shared workdir is theirs; an ended job with no known start: the 24 h before its end), newest first, at most 30; `.git`, virtualenvs, `node_modules`, caches and `site-packages` are skipped,
  and symlinks are never followed out of the workdir. A search that could not finish says so and why (timed out after
  12 s, output cut, sort or find failed); a timeout or an output cap keeps what it found, a dropped connection is an
  error. Unreadable subdirectories are named as such (the files found elsewhere are listed). A failed search is never
  shown as "no log files". The job's own stdout/stderr are listed and marked, and so are other jobs' logs in the same workdir
  (`job 3370736 stdout`); neither is ever the suggested file. In the TUI, `f` in the job detail or the log viewer opens
  the list (your own jobs with a known workdir; the Logs tab of a running job whose stdout went quiet points to it),
  starting on the file shown or the newest file that is not a job's stdout/stderr (never on another job's log while
  something else is listed); `enter` follows it in the Logs tab like stdout, `r`
  looks again, and later `f` switches back to the picked file (on it: pick another), `o`/`e` back to stdout/stderr.
  For agents, `runfleet agent job` adds a `next` line when the stdout is quiet (the same rule as the TUI),
  `runfleet agent log JOB --files` lists the files with a `next:` command for the suggested one, and `--file PATH`
  reads one of the listed files (relative to the workdir or absolute; anything else is refused); MCP `read_log` takes
  `list_files` / `file`. Needs GNU find (Linux login nodes and workstations); elsewhere the list says "unsupported".
  Works with `--demo` (k2 job 3372131).
- **Account usage and job cost (Slurm).** How much your lab has spent, how much is left, what a job costs and which
  experiment used the GPU-hours, from Slurm's own accounting, read-only and cheapest first. Detection now also reads
  the charging rules: each QOS's UsageFactor (`sacctmgr show qos`) and each partition's `TRESBillingWeights`, QOS,
  AllowQos and CPU/memory defaults (cached hosts are detected again once). SU are counted the way sshare and sbank
  charge them, billing × hours × the UsageFactor of the job's QOS; billing-hours (sreport) are never called SU, a QOS
  with factor 0 is free, an unknown factor counts ×1 and shows as `×1?`, and on clusters without limits or weights
  ("usage only") amounts are billing-h. **Job cost** costs no remote command: a `cost` line in the job detail (CPU-h,
  GPU-h by model, the billing rate and where it comes from, the QOS factor, SU so far and at the time limit), a Cost
  box on the GPU tab that shows the arithmetic and the account's balance, `su` and `gpuh` columns and sort keys in the
  Jobs view, and filter terms `su:>10`, `gpuh:>=5`, `acct:`, `part:`. A **periodic summary** (sshare and sreport every
  30 min, sbank at most every 6 h; `[slurm] account_usage`, `poll_account`) keeps each account's used / limit / left
  SU, the next reallocation and your CPU-h / GPU-h; it survives restarts and shows as an `account` line in the cluster
  host detail. **`$` opens the Account screen**: one box per cluster with the balances and a usage bar, what "used"
  means there (since the last reset, or "recent (decays)"), the charged and free partitions, and **where it went**,
  grouped by experiment (the job name without its `--s<seed>--a<attempt>` tail), task, name, account, partition or
  GPU model (`z`), sorted with `o`/`O`; `u` shows the whole account instead of you, `enter` opens the matching jobs.
  Windows `7d`, `30d`, `90d` and `period` (since the inferred reallocation or the usage reset) switch with `[`/`]`
  (`[general] usage_window`, `[slurm] allocation_period`); each is one on-demand report (sshare, sreport and ≤ 30-day
  sacct chunks), cached for 10 minutes and in the cache file, `r` reads again. A failing section shows its last values
  with a fixed red line, never zeros. **For agents**, `runfleet agent usage [HOST…] [--window] [--by] [--scope
  me|account]` answers from the cache (another window or grouping says `usage_window_not_cached` with the exact
  `--refresh` command, which fetches one report and stores it), and `runfleet agent cost HOST --partition P [--gpus N]
  [--gpu-type T] [--time 24h] …` estimates a job before it is submitted from the cached rules: billing per hour and
  its terms, SU for the time, the share of the account's SU left and up to two free partitions with that GPU type,
  every assumption as a note (it never runs ssh and never submits). `agent job` gains a `cost` line and JSON `cost{}`,
  `agent jobs` `--sort su|gpuh`, `agent host` an `account` line, `runfleet snapshot --json` `hosts[].account`; MCP
  adds the tools `account_usage` and `estimate_cost`, and the skill (revision 3) asks for `agent cost` and `agent
  usage` before a large launch and offers a free partition when SU run low, while hpc-run still chooses and submits.
  The cost says `EXCEEDS` (`account.exceeds`) when a job needs more SU than are left, lists every limited account
  that may run there when none was picked (`candidate_accounts`), gives a range (`su_min` / `su_max`) when the
  partition charges by GPU model and no `--gpu-type` was given, and matches `--account` ignoring case. On usage-only
  clusters "recent" is your own decaying usage, the whole account's labelled beside it; a decaying balance is never
  called "used". A report whose every part failed never replaces the last good one, a failed part keeps its last
  good values, a missing job chunk marks the rollup a lower bound, and a restart or `agent --refresh` reuses a fresh
  cached summary instead of reading sshare / sreport / sbank again. When the screen is short, a cluster with SU
  limits keeps its room before usage-only ones.
  Works with `--demo` (lyra: limits and sbank; k2 and vega: usage only).

### Fixed
- Cancelling a probe on a session host (the `K` of the session agent) signals the job's process group first and
  its escaped descendants after, so the job's shell can no longer see its child die and run its next command before
  its own SIGTERM arrives.

## [0.8.0] - 2026-10-18

### Added
- **`runfleet agent`: RunFleet for AI agents** (Claude Code, Codex). A read-only command family — `brief`, `jobs`,
  `job`, `log`, `gpus`, `host`, `events`, `help` — with compact text or versioned JSON (`runfleet.agent.<verb>/1`),
  answered from the cache without ssh (only `log` reads one bounded log tail). Every answer says how old each kind of
  data is (job list, nodes, accounting) and where it came from; problems are stable codes with fixed messages and the
  next step (`runfleet connect HOST` or "press c in the TUI" for an expired login), never raw ssh errors; outputs stay
  within byte budgets (`brief` ≤ 2 KB) and end in the exact command for the rest, runnable as printed. Unknown values
  print as `unknown` (a host without current job data is never counted as 0 jobs; a host whose accounting fails makes
  the finished/failed counts lower bounds, `≥`), and data older than 5 minutes is marked `as of <age> ago`. A cut
  answer's `next` (JSON `truncated.next`) is one command, runnable as printed with the call's `-c PATH` / `--demo
  --demo-seed N`, narrowed to one host when one call cannot hold the rest; the prose goes to `truncated.advice`.
  `--state finished` selects every job that ended (done, failed, cancelled, ended). Events carry stable ids
  (`k2:2331977:ended_bad`; a workstation task's id has its full key, so a reused pid never collides) for
  overlapping-window dedupe. Log lines are printed after `| ` (the log line a failure cause quotes too: `cause (from
  log): | …`), so log content never passes for RunFleet's own lines. `--refresh` polls once first, skipping hosts
  another RunFleet is polling and hosts anyone tried to refresh in the last minute or is refreshing now
  (`rate_limited`); a probe that fails during the refresh keeps the last good data, cached finished jobs included.
  Only your own jobs by default; other users' rows show user, id, state, GPUs and partition only, and their logs are
  never read. Secrets are scrubbed from every string (API keys and tokens, AWS keys, PEM private keys even when cut
  off, `Bearer`/`Basic` credentials, `user:password@` in URLs, `…PASSWORD=` / `…TOKEN=` style values). Exit `0` ok,
  `4` partial, `1` no data, `2` usage. Works with `--demo`.
- **`runfleet skill install`**: installs the bundled `runfleet` skill for most coding agents at once: one copy in
  `~/.agents/skills/runfleet`, which Codex, GitHub Copilot CLI, Gemini CLI, OpenCode, Cursor, Amp, Goose, Factory
  Droid, Windsurf, Pi, Cline and Warp read, plus a link for Claude Code (`${CLAUDE_CONFIG_DIR:-~/.claude}/skills`; a
  junction on Windows), for Copilot when `COPILOT_HOME` is set, and for Antigravity CLI on request (`--agent
  auto|all|claude,…`). Your host names become trigger words; `--print` previews; it never edits agent settings and
  prints the permission rule `Bash(runfleet agent *)`. The SKILL.md is portable (any agent that can run commands).
- **The skill stays up to date**: `runfleet update`, the TUI's update, `install.sh` and `install.ps1` refresh it with
  the new version, and any `runfleet` command refreshes it after an upgrade done another way (`uv tool upgrade`) or a
  change of the config's hosts (one stat when nothing is due; `--no-auto-update` turns all of that off, the
  post-upgrade and installer refresh included). A refresh that keeps failing is retried at most once an hour; a
  skill written by a newer RunFleet is only downgraded by an explicit `runfleet skill update`, which says so; a
  development checkout and the installed tool at the same version do not overwrite each other's skill text.
  `runfleet skill update` does it by hand and also links agents installed since an `--agent auto` install (the
  automatic refresh never does); `runfleet skill status [--json]` shows which agents see it (an agent without its
  link: the command that adds it), its version, edited files, where the last update saved them, the last failed
  refresh, broken or stale links and stray copies.
- **Easy removal**: `runfleet skill uninstall [--dry-run]` removes only what RunFleet wrote (files you edited or added
  are backed up under the state dir first; a link inside is recorded, never followed), never a link to a
  `~/.agents/skills/runfleet` that is not RunFleet's, plus its lock file and the skills folders it created while
  empty, and prints the manual steps for an MCP registration or permission rule (with `$CODEX_HOME` /
  `$CLAUDE_CONFIG_DIR` paths when set). When `runfleet` is not found, the skill first checks whether it is only off
  the agent's PATH before suggesting removal (POSIX and Windows commands).
  `install.sh --uninstall` (`curl … | sh -s -- --uninstall`) and `install.ps1 -Uninstall` remove the skill, then
  RunFleet, and print where your config and state are kept.
- **`runfleet mcp`**: the same read-only verbs as MCP tools over stdio (`fleet_brief`, `list_jobs`, `job_detail`,
  `read_log`, `gpu_availability`, `host_health`, `fleet_events`, `runfleet_help`), with no extra dependency, for
  clients without a shell. Register with `claude mcp add --scope user runfleet -- runfleet mcp`, or
  `[mcp_servers.runfleet]` with `command = "runfleet"`, `args = ["mcp"]` in `~/.codex/config.toml`.

### Changed
- `runfleet snapshot` takes the instance locks of the hosts it polls that no other RunFleet holds, so a TUI started
  meanwhile knows the host is being polled (held hosts are polled as before).
- `runfleet snapshot` no longer replaces a host's cached jobs when its job listing or accounting failed during the
  poll, nor its cached partitions and nodes when `sinfo` failed (even while a login node's `nvidia-smi` answered),
  its GPUs when `nvidia-smi` failed, its host metrics when `/proc` failed or its process table when `ps` failed: the
  cache keeps them (merged with what the poll saw, still dated by their last success); the output still shows this
  poll.
- A full workstation task key (`labstation:p642229-1788696459`) given to `runfleet wait` or `runfleet agent` also
  finds the task when its start differs by up to 5 s (each call recomputes it from the process's elapsed time), and
  the demo's task keys stay the same between `--demo` calls within the hour.
- Error texts that quote what an agent or MCP client sent (unknown host, bad time, unknown tool or argument) are
  scrubbed of secrets, have control characters made visible and are capped at 300 bytes, in the reply and in
  runfleet.log.
- `runfleet snapshot` and `agent --refresh` ignore cached data (snapshot, capabilities, log paths, ended tasks) that
  another ssh destination wrote under the same host name (a second config reusing it): the host is detected afresh,
  and the old destination's capabilities, log paths and ended tasks are dropped from the cache once the new one's
  health is stored (also when the stored health row is unreadable: another version or a damaged field).
- The cache gains a `host_health` table: per host, the last success of each probe, latency and failure counts, and
  which ssh destination wrote it (never error texts). A failed probe keeps the last success already stored. Older
  RunFleet versions ignore it.
- The TUI retries every ~30 s to take the instance locks it could not get at start (another RunFleet, such as an
  agent `--refresh`, held them), so it ends up holding its hosts; the "another RunFleet is polling" notice is shown
  once.

### Fixed
- **False "controller busy" on hosts with uutils `date`** (Ubuntu's Rust coreutils ignores the width in `%3N`): a
  ~100 ms query was timed as ~100,000,000 ms and backed polling off. The probe now reads `%s%9N` and converts to
  milliseconds itself; a `date` without `%N` still times sections as 0. Thanks @ypd666 (#3).
- **A Slurm host stayed "degraded" after its last job ended**: a job that finished between `squeue` and `sstat` left
  an sstat error that nothing cleared once the queue was empty. An idle usage tick now records an empty success (a
  shell no-op, no Slurm RPC); real sstat errors with running jobs stay visible. Thanks @ypd666 (#3).

## [0.7.2] - 2026-10-17

### Changed
- Host detail of a workstation shows other users' processes by default (it used to need `u`); `u` still hides or
  shows them, and the next host detail opens the way you left it.

### Fixed
- **Empty Logs tab for tqdm-only output**: a job writing nothing but a `\r` progress bar (no newline in the last
  64 KiB) showed an empty log pane. RunFleet now reads the log's first 32 KiB once and shows its start, with a note
  that the program only rewrites a progress line and the latest is on the `↻` line below.
- The answer to your own key press (e.g. `C` with no host needing a login says "no host needs a login") could be
  dropped when three background notices had just been shown; such replies are never rate-limited now.
- Session transport (`mux = "session"`, Windows): when the login ended while jobs were being killed, a job's last
  write could race the helper's `rm -rf` and leave its private temp directory (`rfa.*`) on the login node; the
  helper now retries the removal.

## [0.7.1] - 2026-10-16

### Added
- **Clickable path at the top of detail screens**: in `RunFleet | k2 › Jobs › 4024018`, a click on `k2` goes to its
  host detail, `Jobs` to the Jobs view and, in the log viewer, the job id back to that job — returning to the screen
  you came through when it is still open (cursor and tab as you left them). The last part is the screen you are on.

## [0.7.0] - 2026-10-16

### Added
- **Duo and password hosts on Windows** (session transport): the Windows `ssh` has no ControlMaster, so RunFleet now
  keeps one ssh login per host open and runs every refresh, detail view and log stream over it, through a small
  temporary POSIX `sh` helper it sends on that login's stdin. Nothing is installed on the server; the helper keeps
  its few state files in a private temp directory and removes it, and ends everything it started, when the login
  ends. `c` (and `C`, log in to all) now logs in once in the console and that login serves the host until RunFleet
  exits; key and `ssh-agent` hosts need only one login per host instead of one per refresh. A probe that times out
  ends only itself; the session is restarted only when it stops answering too, or when it dropped (at most once per
  30 s; Duo hosts then ask for `c` again). Shells and the remote pager are still separate `ssh -t` logins, and port forwards are not available over a
  session yet. Host detail and `runfleet doctor` show `mux: session` and what it means; `runfleet connect` only checks
  the login there and says to press `c` in RunFleet.
- **`mux = "session"`** brings the same transport to Linux and macOS (opt-in; never chosen by `auto`). The server
  needs a POSIX `sh`, `dd`, `awk`, `od` and `mktemp`; `setsid` and `ps` make kills reach every process of a probe.

### Fixed
- Ctrl-C in a command RunFleet runs in the terminal (the local pager, a `c` login) no longer ends
  RunFleet once the command returns: the command gets the Ctrl-C (`less` stops following), and a session login is
  aborted at the first Ctrl-C.

### Changed
- Windows: every process RunFleet starts joins a Job Object that Windows ends with RunFleet, so not even a crash
  leaves an `ssh.exe` (or its ProxyJump hop) behind. `mux = "off"` keeps the old behaviour (a login per refresh).

## [0.6.0] - 2026-10-16

### Added
- **Open what needs you from the Fleet view**: a click on a job row of a roomy Fleet card, or on a row of the fleet
  panel's *what changed*, opens that job (a click elsewhere on a card still selects it). `a` opens *Needs you*: every
  failure of the last 4 h, running job with a warning or low on time, pending job that will never start or is held,
  and older failure across the fleet, most urgent first, with host, job id, full name and why; `enter` opens the job.
  It works in the compact cards and the narrow table too and polls nothing new.
- **`runfleet status [--short | --json] [HOST …]`** for shell prompts and tmux status bars: your running, pending and
  failed jobs across the fleet, the hosts that need you and how old the data is, e.g. `R4 PD2 F1 · lynx:auth · 3m`
  (`14m stale` past 10 minutes). It only reads RunFleet's cache — no ssh, no writes, about 70 ms — prints `no data`
  (exit 1) when there is none, and has a per-host table by default and a `runfleet.status/1` JSON document. The user
  guide has tmux, zsh and starship snippets. The cache now also keeps each host's connection state for it.
- **Watch for free GPUs**: palette › *Watch free GPUs…*, type e.g. `2 h200 k2`, `4x a100 k2/gpu-preempt` or `1 mig
  vega`, and RunFleet notifies you (toast, terminal notification, `[notify] command`) when that many GPUs are free
  **on one node**. It uses only what it already polls, counts only fresh readings (never a `host ?`), fires after two
  readings in a row agree, then rests at least 15 minutes and until the GPUs are taken again. Backfill-planned GPUs
  count only with `planned`, MIG slices only with `mig`. While you type it shows what matches right now; palette ›
  *Watches* lists what each watch is doing and removes one (`d`). Firings appear in *Notifications* and the fleet
  panel; when polling pauses after a long idle time the watches say so instead of going silent. Up to 10, kept in
  `ui.json`.
- **Saved Jobs views**: palette › *Save view as…* names the Jobs view's filter, state preset, sort and grouping;
  *View: \<name\>* switches to the Jobs view and applies all of it at once, *Delete view: \<name\>* forgets it.
  The status line shows `view: <name>` until you change any part of it. Up to 20 views, names up to 24 characters
  (letters, digits, space, `. _ -`), kept in `ui.json`; entries RunFleet cannot read are skipped.
- **Why isn't my job starting? — priority evidence**: a pending Slurm job's detail now shows a *Priority* box in place
  of the empty GPU box: its priority as Slurm computes it, each factor (age, fair share, job size, partition, QOS,
  TRES, association) against the most the cluster lets it add, which one dominates, your fair share per account
  (usage vs share) and plain hints such as "fairshare 0.03: your group used much more than its share recently; it
  recovers with a half-life of 14 days" or "held back by per-user job cap (QOS), not by priority". The Jobs preview and
  the compact detail show one line (`prio 3,017 · fairshare dominates · fairshare 0.03 · 4m ago`). On a cluster that
  does not rank by priority (every weight 0, or `priority/basic`) it says so — submit order plus backfill — instead
  of a row of zeros. It is evidence, not a promised start time or queue position. Cost: `sprio` and `sshare` (your
  own jobs and user only) in one round trip at most every 10 minutes (hourly when the host is off screen), only while
  a job of yours is pending on a cluster that ranks by priority; the priority settings come with the existing
  `scontrol show config` detection (cached capabilities are re-detected once).
- **Search logs across your recent jobs**: palette › *Search logs…* (or `ctrl+f` in the Jobs view) finds text in the
  logs of your own jobs that run or ended within the recent window, on the hosts in scope — literal by default, a
  regular expression (`grep -E`) or case-sensitive on request. The results list each match per job and file
  (stdout / stderr) with the matched text highlighted, and say exactly what was read: files, KiB, and every cap that
  applied. `enter` opens the log viewer right at that line — anchored to the search's byte offset, so it is the same
  line however much the job wrote since — and `esc` comes back to the results; `/` edits the search. It runs only when
  you ask, one read-only probe per host: at most 20 log files per host, the last 256 KiB of each, 1 MiB read and 200
  matches in all, 30 s at most; paths and the pattern are quoted, nothing is globbed or walked, and a regular
  expression whose `{n,m}` counts grep would expand into gigabytes of memory on the login node is refused up front.
- **Windows (preview)**: RunFleet now runs natively on Windows 10/11 with the built-in OpenSSH client (8.1 or newer),
  best in Windows Terminal. Install in PowerShell with
  `powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.ps1 | iex"`
  (Git for Windows is a prerequisite). Files live in `%APPDATA%\runfleet` (config) and `%LOCALAPPDATA%\runfleet`
  (cache, state, log). Preview limits, all shown by `runfleet doctor` and in the host screen: Windows `ssh` has no
  connection sharing, so every refresh is a full login — hosts with keys or `ssh-agent` work, hosts behind Duo or a
  password need WSL for now; no port forwards, no `ssh = "local"` host, no `ctrl+z`; `runfleet update` prints the
  command to run after quitting RunFleet instead of replacing the running `runfleet.exe`. CI now runs the test suite
  and a demo smoke test on Windows, and type-checks the Windows code paths on Linux.

### Fixed
- Job exit signals (`exit 138 = SIGUSR1`, "killed by SIGBUS") are named from the Linux table whatever OS RunFleet runs
  on; on macOS some numbers were named after macOS's own signals (10 read as SIGBUS).
- A log rotation that cannot rename `runfleet.log` (another RunFleet has it open) keeps logging instead of printing a
  traceback over the TUI on every line.

## [0.5.0] - 2026-10-15

### Added
- **Roomy Fleet view**: in a tall terminal (e.g. 120×40 or 172×62) the Fleet cards grow into the empty space instead
  of leaving it blank, and keep a fixed height so nothing jumps. Each card lists your jobs by how much they need you:
  fresh failures with why they ended, stalled or short-on-time jobs, pending jobs that will never run, then the rest
  by time left. Workstation GPU lines say who holds each GPU. The empty cell after the last card shows free GPUs by
  type across the fleet (your types first) and what changed since your last visit. Nothing new is polled. Palette ›
  *Fleet: compact cards* keeps today's cards (remembered).

### Changed
- **Colours follow your terminal**: RunFleet asks the terminal for its background colour at start and mixes every grey,
  panel, frame, track and cursor row from it, so they sit in your colour scheme's hue instead of being grey boxes on
  Solarized, Dracula, Catppuccin or a tinted iTerm2 profile; a light background gets the light palette even over ssh
  (where `COLORFGBG` is missing).
- **Theme setting**: Settings › About › *theme* (or palette › *Theme: …*) switches live between `auto`, `dark`,
  `light` and `ansi` (only the terminal's own 16 colours, chosen automatically on 16-colour terminals) and is
  remembered. `RUNFLEET_THEME=ansi` works too.
- `runfleet doctor` starts with a *terminal* block (colour depth, reported background, chosen theme, hints for ssh
  without `COLORTERM` and iTerm2's colour settings).

### Fixed
- At 100×30 the Fleet cards could wrap a long line and grow one card of a row taller than its neighbour: the cards'
  scrollbar column is now always reserved and lines are cut with `…`.
- The active view tab was light grey on blue (1.5:1) and the header's host chips lost their state colours: Textual's
  link colour repainted every clickable span. They keep their own colours now.
- Card frames, table cursor rows, hover and the problem-host frame were blended against black whatever the terminal
  showed: near-invisible frames on non-black backgrounds, a dark navy cursor bar on light ones. Opening a dialog no
  longer paints the whole screen near-black.
- The selected table row kept its background but lost the colours of its cells (GPU bars, states, host names).
- Table text and the colours of cancelled jobs and inactive tabs no longer rely on faint (SGR 2) text, which iTerm2
  draws half-transparent.
- The highlighted entry of the action menu, the digest and the forwards dialog sits on the cursor tint, so its dimmed
  and coloured text stays readable.
- Job names with emoji (`❤️`, ZWJ sequences) are cut to the right width instead of overflowing with a second `…`.

## [0.4.0] - 2026-10-15

### Added
- **One-line install and update**: `curl -LsSf https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.sh | sh`
  installs uv when missing (with its own Python if needed), then RunFleet; run again, it updates.
- **`runfleet update`** works out how RunFleet was installed (uv tool, pipx, a git clone), shows what changed since
  your version, runs the matching command and prints the new version. `--check` only looks; `--force` updates
  anyway.
- **Update from inside RunFleet**: at start it asks GitHub for the newest version at most once a day and says once
  when a new one is out. *Update RunFleet* in the palette (or `↵` on the version row in Settings › About) quits,
  updates and starts RunFleet again with the same arguments. Only the version number is fetched; turn the check off
  with `update_check = false` under `[general]`. `--demo` never checks.

## [0.3.0] - 2026-10-13

### Added
- **Who runs on a node**: a cluster's host detail now shows other users' jobs, not only yours. The JOBS column
  (formerly MY JOBS) lists your job ids, then each other user with the GPUs they hold there (`bob×4 · carol×3`).
  A new panel under the nodes lists every job on the selected node — user and real name (from `getent passwd`),
  account, partition, GPUs, elapsed / limit, time left — and, on a full node, when the next GPU frees up at the
  latest. Read with one `squeue -a` over the nodes shown (jobs of other partitions on shared nodes included), only
  while the host detail is open (every 30 s, with the node list so both show the same moment); the fleet view costs
  nothing more. Clusters with `PrivateData=jobs` show only your jobs, and the panel says so.

## [0.2.1] - 2026-10-13

### Fixed
- **GPU partitions you have not used lately disappeared**: the node view and the host card only covered partitions of
  your jobs from the last few days (plus association partitions, which most sites leave empty), so a lab's paid GPU
  partition vanished once you had run only in free ones for a while. Detection now also reads
  `scontrol show partition -o` (with your groups and the parent chain of your accounts) once and adds every GPU
  partition you may submit to: AllowGroups, AllowAccounts/DenyAccounts (sub-accounts included, as in Slurm) and
  AllowQos/DenyQos checked against your associations on that cluster; hidden, root-only, inactive and drained
  partitions stay out. Capabilities cached by an older version are detected again at start, so the change shows up
  right after the upgrade. A `partitions` list in the config still replaces the derived set.
- **Host card free GPUs**: with many partitions the line was cut off at the card edge. It now drops the
  "(N planned)" notes, then wraps to a second line, and only then ends with "+N more" (host detail lists all).

## [0.2.0] - 2026-10-13

### Added
- **Clickable GPUs on workstations**: in a workstation's host detail, clicking a GPU row (or `[` / `]`) lists only
  the processes on that GPU, other users' included; the same GPU again shows all processes.
- **Header logo goes home**: clicking `RunFleet` at the top left closes detail screens and shows the Fleet view.

### Changed
- **Live readings switch to srun by themselves where nodes refuse ssh**: `attach` now defaults to `ssh+srun`. On a
  cluster whose compute nodes refuse ssh even inside your job, the first refused read switches the GPU / Procs tab to
  srun (remembered per host) and waits for `r`; every srun read still asks first and adds one permanent sacct row.
  In 0.1.0 such hosts stopped at "refuses ssh to compute nodes" until you set `attach = "ssh+srun"` yourself. Set
  `attach = "ssh"` to keep that behaviour.

### Fixed
- **Attach on nodes you never ssh'ed to**: the login → node hop required the node's host key to be in the login host's
  known_hosts already, so on clusters that ship no node keys (and run `StrictHostKeyChecking no` themselves) live
  readings failed with "host key is not known". A node key that is recorded must still match; an unknown node is now
  accepted without being written to any known_hosts file.

## [0.1.0] - 2026-10-12

First public release.

### Added
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

[Unreleased]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Yuanpeng-Li/runfleet/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Yuanpeng-Li/runfleet/releases/tag/v0.1.0
