# RunFleet — Technical Design (v0.1)

Status: **revision 3**, 2026-10 — revision 3 adds the v0.2 **Attach** contract (D7 rewritten, §7.16, models /
Collector / FleetAPI / config changes marked "Attach"). Revision 2, 2026-10 — revision 1 plus an adversarial
critique (every finding resolved or answered in Appendix D) and the v0.1 scope cut.

**Evidence.** The design rests on read-only **field probes**: the exact Slurm, `nvidia-smi`, `ps` and procfs commands
RunFleet runs, captured on three production Slurm clusters (Slurm 25.05, 25.11 and 26.05) and two GPU workstations,
plus notes on prior art and on the stack. The raw dumps and notes are not published, because they name real users,
hosts, paths and jobs. Anonymised captures are the test fixtures under `tests/fixtures/samples/<host>/` and
`tests/fixtures/attach/`, and every behaviour this document relies on is pinned by a test against them (§10.2).
Short citations such as "field probes", "stack notes", "prior-art notes" or "Attach field probes" point at those
unpublished notes. "M3 F*n*" is finding *n* of the M3 read-only validation run (§11.4).

**Host names.** Examples use the demo fleet's names (§10.6): Slurm clusters `k2`, `vega` and `lyra`, workstations
`labstation` and `lynx`, and a jump host `jumpvm`. These fictional hosts are modelled on the probed machines (their
Slurm versions, features and quirks), but the captures are anonymised: host and node names, job and user ids, times,
paths and users (`alice` for the probing account) were changed, so a fictional host is not a 1:1 copy of any real
one. Where this document and a **field-probe fact** disagree, the fact wins and this document is a bug.

Read order for implementers: §1 (scope), §3 (architecture and the frozen contracts), then your work package in §11 and
the sections it references. Every file has exactly one owning work package (§11.2). Interfaces marked **FROZEN** may
only change by editing this document first. Anything marked **v0.2** is out of scope: do not build it, do not stub it — except **Attach** (§7.16), which
revision 3 brings into scope.

---

## 0. Summary of the key decisions

| # | Decision | Why (evidence) |
|---|---|---|
| D1 | **Python ≥ 3.11 + Textual 8.2.x**, no other runtime deps besides `rich` and `platformdirs` | stack notes: DataTable diff refresh < 50 ms/1000 rows, Pilot + snapshot testing works, `App.suspend()` exists; no Go/Rust toolchain on the dev box |
| D2 | **System OpenSSH as asyncio subprocesses**; no SSH library | 4/5 hosts already have ControlMaster sockets; asyncssh can't reuse them; vega GSSAPI/network-FS cold logins can take minutes |
| D3 | **One SSH round trip per host per tick**: a POSIX `sh` script on stdin, framed into named *sections* with a per-probe nonce, per-section `rc/ms/stderr`, each command under remote `timeout`. Sections that can hang uninterruptibly (`nvidia-smi`) run **last** and are abandoned, never waited for | slurm-avail pattern; field probes: 30 commands in 1.2–3.2 s; zsh login shells on workstations |
| D4 | **Collectors are sans-IO**: they only *plan* sections and *parse* results. All I/O lives in `core/` and `transport/` | parsers unit-test against real fixtures; demo mode is free; backends stay small |
| D5 | **Text formats only, never `--json`** for Slurm | lyra has no JSON serializer (and rc=0 when piped); k2 has Python 3.6 and no jq; JSON is 21 KB/job |
| D6 | `squeue --me -h -O` (Command last, unit-separator suffix); `sacct -X -P --delimiter=<US> -n -S now-… -E now`; `scontrol show job -d -o` **once per job**; `sstat … TRESUsageInTot` for job GPU util where the cluster records `gres/gpuutil`; `SLURM_TIME_FORMAT=%s` so every timestamp is an epoch | field probes; DST ends 2026-11-01 inside the sacct window |
| D7 | **The background loop never touches compute nodes; the user opening JobDetail's GPU or Procs tab on a running Slurm job may (Attach, §7.16).** HostPoller only ever talks to the login host (`sstat`, `scontrol -d`: zero-trace sources). Attach reads one job's node on demand, via `ssh <node>` from the login host (pam_slurm_adopt; no accounting trace) or, where that is refused and the user confirms, a one-shot `srun --jobid --overlap` step. The `rfprobe` srun step is a **user-initiated, one-shot, read-only probe** (nvidia-smi, cgroup files, `ps`; bounded by `--time=1` and timeouts) — not an experiment payload, and RunFleet never launches work of its own; it is never started by a timer | Attach field probes: ssh-adopt works on k2/vega (0.5–0.7 s, no sacct row); lyra refuses node ssh before PAM, `srun --overlap` works everywhere (0.1–0.3 s) but leaves one permanent sacct row per call; five no-leftover layers verified by audit on all three clusters |
| D8 | Mux: if the user's `ssh -G` has a ControlPath, use it (a master RunFleet starts there carries **only the user's own options**); otherwise a RunFleet-managed ControlPath (`$XDG_RUNTIME_DIR/runfleet/cm-%C`, lyra). Probes never become masters and never fall back to cold logins on mux hosts | stack.md connection policy; Appendix D #1, #22 |
| D9 | Background ssh processes **cannot prompt**: new session (no controlling tty), `BatchMode=yes`, `SSH_ASKPASS_REQUIRE=never`, and ProxyJump rewritten into an explicit BatchMode `ProxyCommand` (OpenSSH drops command-line `-o` for the jump hop). Auth problems become a `NEEDS_AUTH` badge; `c` suspends the TUI and runs `ssh -fN host` in the real terminal | OpenSSH 10.0p2 verified: `-o` options are not passed to the ProxyJump hop |
| D10 | **Unified `Job` model** for Slurm jobs and workstation "tasks" (a process tree rooted at the user's top-level launcher, identity = root pid + start time, sticky while the root lives), one normalized `JobState` enum with groups | PRD "consistent experience" |
| D11 | **Demo mode = a simulated world behind a `DemoTransport`** that emits byte-exact real formats, so `--demo` exercises parsers, poller, store and UI | no cluster and no GPU on the dev box |
| D12 | Licence **MIT**. No code copied from GPL projects (nvitop TUI, nvtop, jobstats, seff) | prior-art notes |

---

## 1. Goals, non-goals, scope

### 1.1 Goals
1. `runfleet` opens in < 300 ms on the **Fleet overview** of every configured machine (last-known data from cache,
   marked stale) and shows fresh data within ~1.5 s for hosts with a live ControlMaster.
2. One merged, filterable, sortable table of **all jobs on all machines**: Slurm jobs (running, pending, recently
   finished) and workstation tasks.
3. Per-job inspection: metadata, allocation, per-node GPU indices, GPU utilisation where obtainable (sstat /
   nvidia-smi), steps, processes where obtainable (workstation tasks), log tail with follow, failure forensics, stall
   hints.
4. Per-host inspection: nvitop-style GPU bars with processes on workstations; partitions/nodes/free GPUs on clusters.
5. Safe management: Slurm cancel / hold / release / requeue; task terminate / interrupt / kill; always confirmed;
   `--readonly`.
6. Light on shared infrastructure: ≤ 4 SSH round trips/min on average per visible Slurm host (test-enforced: ≤ 240/h), ≤ 7 controller/dbd
   RPCs/min, ≤ 3 concurrent RunFleet sessions on a user's shared ControlMaster, polling paused after 2 h without input.
7. Never crashes because of a host: every remote failure is a badge, a stale marker and a retry schedule.

### 1.2 Non-goals (v0.x)
- Job **submission**, templates or resubmission (the user's own submission tooling owns that).
- Installing anything on remote hosts, daemons, persistent agents, Prometheus, slurmrestd. **Allowed (P2): a
  session-scoped helper** — the session transport's `sh` agent (§4.8) is sent on stdin of RunFleet's own ssh login
  each time, lives only in that remote shell's memory, keeps its few state files in a private `mktemp -d` directory
  it removes, and ends with the login (stdin EOF, exit, a signal). Nothing is written to the home directory, nothing
  outlives the session, nothing listens; still no daemon, no install.
- Admin features (draining nodes, editing partitions), acting on other users' jobs or processes.
- A web UI.
- Long-term metrics storage.

**Native Windows is a goal, shipped as a preview** (WSL remains the full-featured way on a Windows machine). The
package imports and runs on Windows 10/11 with the built-in OpenSSH client (≥ 8.1); every local OS difference lives
in `util/platform.py` (§3.2; a contract test keeps `fcntl`, `os.killpg`, `os.getuid`, `signal.SIGKILL`, `termios`
out of every other module, `ui/termbg.py` excepted behind its own guards), and CI runs the suite on `windows-latest`
(tests marked `posix` skipped) plus `mypy --platform win32` on Linux. Windows OpenSSH has no ControlMaster, so P2
polls every host over the **session transport** (§4.8): one long-lived `ssh.exe` login per host running a temporary
`sh` agent sent on stdin. Duo and password hosts work: `c` logs in once in the real console and that login serves
every background poll until RunFleet exits (each restart needs `c` again). Limits, all said in `doctor` and the UI:
shells (`s`) and the remote pager (`p`) are separate `ssh -t` logins (a Duo push each); no port forwards (P3); no
`ssh = "local"` host; no `ctrl+z`; `runfleet update` prints manual steps (§8.6); the terminal's background is not
asked; `runfleet connect` is only an auth check (a session cannot outlive the CLI process). Planned: P3, port
forwards over the session and in-place updates.

### 1.3 v0.1 scope (polished core) vs later phases

| Area | v0.1 | v0.2 | v0.3+ |
|---|---|---|---|
| Hosts | `config.toml`; `runfleet hosts import` (non-interactive, plus a plain first-run prompt when no config exists); ad-hoc `runfleet k2 lynx`; local host | import wizard UI, node-ssh mode (pam_slurm_adopt), multiple login endpoints | port forwards over the session transport (Windows, §4.8) |
| Slurm | own jobs (squeue), history (sacct, bounded window), job detail (scontrol -d, sacct steps, sstat), job GPU util via sstat, sinfo nodes for the user's partitions | **Attach** (§7.16): per-GPU / per-process / cgroup readings of a running job via node ssh or a confirmed `srun --overlap`, on-demand `history <dur>`, other users' running jobs per node (host detail, §5.2.9), queue position, `scontrol update TimeLimit`, seff-style efficiency | PBS/Torque, LSF, HTCondor, Kubernetes |
| Workstation | NVIDIA GPUs, GPU processes of all users, own tasks (sticky process trees), CPU/mem/load, logs via `/proc/<pid>/fd/1`, exit log scan (Traceback/OOM/Killed), stall hints | AMD `rocm-smi`, Intel | Apple |
| Views | **Fleet (default)**, Jobs, GPUs, Job detail, Host detail, Log viewer (append + follow), Help, Confirm, Action menu, Host error, Textual command palette with RunFleet commands | log search / older-chunk paging / re-wrap, preview modes, themes / ASCII mode, pins, sparklines, experiment-regex grouping, column chooser, `:` command grammar, `:history` | LED grid for > 20 hosts |
| Actions | Slurm cancel, hold/release, requeue; task terminate (TERM→grace→KILL, re-enumerated remotely), interrupt (INT), kill; host/task shell via `App.suspend`; log in `$PAGER`; copy id/path; `--readonly` | job attach (`srun --pty` / node ssh), Slurm signal, signals beyond TERM/INT/KILL, custom commands, notifications | batch "rerun failed" hook |
| CLI | `runfleet`, `runfleet <hosts…>`, `connect`, `hosts list/import`, `doctor`, `snapshot --json`, `--demo` (one scenario), `--readonly` | `--record/--replay`, chaos/big demo scenarios, `watch <jobid>` | |
| Quality | unit, shell-integration, Pilot and SVG snapshot tests; performance budgets checked by hand at M2 | performance CI on a big scenario | |

---

## 2. Stack and packaging

### 2.1 Stack (decided)
- **Language:** Python, `requires-python = ">=3.11"` (`StrEnum`, `tomllib`, `asyncio.TaskGroup`). Developed on 3.13.
- **TUI:** Textual 8.2.x + Rich 15. Built-ins used: `DataTable`, `RichLog`, `TabbedContent`, `Input`, `OptionList`,
  `ModalScreen`, `ContentSwitcher`, the command palette (`App.COMMANDS` + `Provider`), `textual.fuzzy.Matcher`,
  `App.suspend()`, `App.copy_to_clipboard` (OSC 52), workers with `exit_on_error=False`.
- **Transport:** system OpenSSH client ≥ 7.3 (`ssh -G` ≥ 6.8; ProxyJump ≥ 7.3). The dev box has OpenSSH 10.0p2.
- **Remote footprint:** `/bin/sh` + coreutils; Slurm CLIs / `nvidia-smi` / `ps` when present. Nothing installed or
  written remotely except one private `mktemp -d` directory per probe, deleted by a `trap` (section stderr, nvidia-smi
  output files).
- **Local state:** stdlib `sqlite3` (cache) and JSON (UI state). `hosts import` writes TOML with a small in-house
  emitter for our own schema — no `tomli-w` dependency.

### 2.2 `pyproject.toml` (WP4 owns; exact content)

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "runfleet"
version = "0.1.0"
description = "All your Slurm jobs and GPU machines in one terminal: queues, GPUs, logs and failures over plain SSH"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Yuanpeng Li", email = "liyp1220@gmail.com" }]
dependencies = [
  "textual>=8.2.8,<9",
  "rich>=15.0",
  "platformdirs>=4.12",
  "tomlkit>=0.13",
]

[project.scripts]
runfleet = "runfleet.cli:main"

[dependency-groups]
dev = [
  "pytest>=9.1.1",
  "pytest-asyncio>=1.4.0",
  "pytest-textual-snapshot==1.0.0",   # 1.1.0 pins syrupy==4.8.0; 1.0.0 works with syrupy 6.x
  "syrupy>=6.1",
  "textual-dev>=1.8.0",
  "ruff>=0.6",
  "mypy>=1.11",
]

[tool.hatch.build.targets.wheel]
packages = ["src/runfleet"]

# Whitelist: the sdist carries the package, scripts and user docs only. Local-only development notes (gitignored
# /docs/research and /docs/integration-notes.md) are excluded explicitly as well; tests/unit/test_packaging.py
# enforces both. The test suite (and its fixtures under tests/fixtures) is for git checkouts, not the sdist.
[tool.hatch.build.targets.sdist]
include = [
  "/src/runfleet",
  "/scripts",
  "/docs/*.md",
  "/docs/screenshots",
  "/README.md",
  "/CHANGELOG.md",
  "/LICENSE",
  "/pyproject.toml",
]
exclude = ["__pycache__", "*.py[cod]", "/docs/research", "/docs/integration-notes.md"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "real: touches real hosts over ssh; never run in CI (opt-in with -m real)",
  "posix: needs a POSIX system (sh, process groups, file modes, ptys); skipped on native Windows",
]

[tool.ruff]
line-length = 110
target-version = "py311"
extend-exclude = ["docs/research"]   # local-only dev notes (gitignored), not package code

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]     # explicit: ruff's default set changes between versions (contracts pass this set)

[tool.mypy]
python_version = "3.11"
files = ["src/runfleet"]             # `uv run mypy` == `uv run mypy src/runfleet`
strict = true                        # §11.5: strict for models, config, probe, transport, backends, core

[[tool.mypy.overrides]]
module = ["runfleet.ui", "runfleet.ui.*", "runfleet.cli"]   # Textual code may be non-strict (§11.5)
disallow_untyped_defs = false
disallow_incomplete_defs = false
disallow_untyped_calls = false
disallow_subclassing_any = false
disallow_untyped_decorators = false
disallow_any_generics = false
warn_return_any = false
```

`uv.lock` is committed and is the real pin set. Floors above are the versions verified at the time of writing.
`.python-version` (`3.13`) pins the dev interpreter for `uv sync`; CI also runs the 3.11 floor. mypy's `strict` is a
global-only option, so it is on for the whole package and relaxed per module for the Textual code. The sdist is a
whitelist (package, scripts, user docs, screenshots); the test suite and its field fixtures ship only with the git
repository. Local, gitignored development notes are excluded from the sdist and from `ruff` explicitly, and
`tests/unit/test_packaging.py` checks both.

### 2.3 Install and launch
```
uv tool install git+https://github.com/Yuanpeng-Li/runfleet   # no PyPI release yet
pipx install git+https://github.com/Yuanpeng-Li/runfleet
uv tool install .           # from a checkout
uvx --from . runfleet --demo
runfleet                    # configured fleet (no config yet: prompts to import hosts from ~/.ssh/config)
runfleet k2 labstation      # ad-hoc: only these ssh aliases this run (new ones are saved once they answer, §8.4)
runfleet --demo             # synthetic fleet, no network
```
Windows (preview, §1.2): `powershell -ExecutionPolicy ByPass -c "irm …/main/install.ps1 | iex"` (§8.6), or the same
`uv tool install` line; Git for Windows and the OpenSSH Client feature are prerequisites.
Dev loop: `uv sync`, `uv run runfleet --demo`, `uv run pytest`, `uv run textual console` + `uv run textual run --dev
runfleet.ui.app:dev_app` for live CSS.

---

## 3. Architecture

### 3.1 Layers

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ UI (Textual)  runfleet/ui/**     screens, widgets, keymap, palette provider            │
│   reads FleetStore; calls FleetAPI (core/service.py); never touches transport/backends │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Core          runfleet/core/**   FleetService · HostPoller · FleetStore · ActionService│
│                                  LogService · Cache · Doctor · Clock                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Collectors    runfleet/backends/**  sans-IO: plan Sections, parse SectionResults       │
│   Scheduler role: slurm, process      Monitor role: nvidia, procfs                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Probe         runfleet/probe/**  Section / ProbeScript / framing (render + parse)      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Transport     runfleet/transport/**  OpenSSH · Local · Fake  (Demo lives in demo/)     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Rules:
- Dependencies point downward only. `models.py`, `config.py` and `errors.py` are importable from every layer. Nothing
  outside `ui/` and `cli.py` imports `textual`.
- Collectors never perform I/O, never sleep, never read the clock (time is passed in). They may keep per-host state.
- Everything runs on the single asyncio loop that Textual owns. Parsing is synchronous (< 5 ms for our sizes); the
  only `asyncio.to_thread` use is SQLite.
- A host is described by a **profile**: one Scheduler collector + zero or more Monitor collectors, chosen by detection
  (§5.1). `k2` = `slurm` + `procfs`; `labstation` = `process` + `nvidia` + `procfs`; a Slurm login node with
  `nvidia-smi` = `slurm` + `nvidia` + `procfs`.
- **Section routing**: a section's owner is the prefix of its name before the first `.` (`Section.owner`). Owners are
  the registered collector names plus `detect` (`backends/detect.py`) and `log` (`core/logs.py`). Results are routed
  by owner only.

### 3.2 Package layout (exact; owner in brackets, §11.2)

```
pyproject.toml uv.lock .python-version .gitignore LICENSE README.md CHANGELOG.md         [WP4]
docs/USER_GUIDE.md docs/CONFIG.md                                                       [WP4]
src/runfleet/
  __init__.py            __version__                                                    [WP1]
  py.typed               PEP 561 marker (third-party collectors type against backends.base) [WP1]
  __main__.py            `python -m runfleet` → cli.main()                              [WP4]
  cli.py                 argparse, subcommands, first-run prompt, builds FleetService, runs the app  [WP4]
  export.py              snapshot → JSON (schema "runfleet.snapshot/1")                 [WP4]
  models.py              FROZEN shared dataclasses + enums (§3.3)                       [WP1]
  errors.py              exception classes only: RunFleetError, ConfigError, ActionError [WP1]
  config.py              FROZEN settings dataclasses (§3.7); load/validate config.toml; TOML emitter  [WP1]
  configedit.py          comment-preserving config.toml edits (tomlkit): add/remove/reorder [[host]], set [general]; atomic write  [WP1]
  sshconfig.py           list Host aliases from ~/.ssh/config (Include-aware); `ssh -G` resolver  [WP1]
  paths.py               platformdirs locations (config, state, cache, mux dir, ssh logs) [WP1]
  debugdump.py           `--debug` raw probe dumps (wire format, rotated per host) under <state>/debug [WP1]
  wait.py                `runfleet wait HOST:JOBID …`: pollers for those hosts until the jobs end (§8.5)  [WP4]
  update.py              `runfleet update`, the daily update check, install detection, relaunch (§8.6)  [WP4]
  status.py              `runfleet status`: cache-only summary for prompts / tmux, read-only sqlite (§8.5)  [WP4]
  util/
    shell.py             shlex helpers, id/pid/path validators                          [WP1]
    slurmfmt.py          Slurm durations/timestamps/memory/TRES/hostlist/state-suffix parsing  [WP2]
    humanize.py          display formatting: durations, ages, bytes, MiB→G, percentages [WP3]
    logscan.py           exit log scan (Traceback / OOM / Killed): ended tasks + "Why it ended"; cause chip, error lines [WP2]
    efficiency.py        efficiency card: time / CPU / memory / GPU used vs requested, "next time" request (§5.2.8)
    platform.py          the only home of OS differences: IS_WINDOWS, spawn keywords (background, session login),
                         kill_tree, the Windows Job Object, file locks, command splitting/display, Windows-reserved
                         names, Linux signal names (§1.2, §4.2, §4.8, §8.1)  [WP1]
  probe/
    section.py           FROZEN Section, SectionResult, SectionOutcome, ProbeScript, StreamSpec  [WP1]
    framing.py           render ProbeScript → POSIX sh; parse framed stdout → SectionResults  [WP1]
  transport/
    base.py              FROZEN Transport protocol, RunResult, ProbeResult, StreamHandle [WP1]
    openssh.py           OpenSSHTransport (probe, stream, argv builders, ensure_master)  [WP1]
    session.py           SessionTransport: one long-lived ssh per host running the agent; jobs, pings, login (§4.8)  [WP1]
    agent.py             the session agent: its POSIX sh text, request encoding, incremental reply decoder (pure, §4.8)  [WP1]
    mux.py               mux policy from `ssh -G` (incl. who gets the session transport), jump rewrite, managed socket dir  [WP1]
    spawn.py             tty-less subprocess spawning (new session, scrubbed env)        [WP1]
    classify.py          ssh stderr/rc → ErrorKind/ConnState (table §9)                 [WP1]
    local.py             LocalTransport (`sh -s` locally)                               [WP1]
    fake.py              FakeTransport (tests): scripted SectionResults per section name [WP1]
  backends/
    __init__.py
    base.py              FROZEN Collector ABCs, Cadence, MonitorView, CollectorOutput, Action* (§3.6)  [WP2]
    registry.py          BUILTIN collectors + entry-point group "runfleet.collectors"   [WP2]
    detect.py            detect sections, capability merge, profile selection           [WP2]
    common/
      nvidia.py          NvidiaMonitor: nvidia-smi sections + parsers                   [WP2]
      procfs.py          ProcfsMonitor: loadavg, meminfo, /proc/stat delta, uptime      [WP2]
      gpu_types.py       GPU type → memory MiB                                          [WP2]
      cgroup.py          cgroup v2 file parsing → CgroupStats (Attach, §7.16)           [WP2]
    slurm/
      collector.py       SlurmCollector                                                 [WP2]
      commands.py        every Slurm command string / field list (single source)        [WP2]
      states.py          state mapping, reason humanizer, exit explanations, hints       [WP2]
      parse_squeue.py parse_sacct.py parse_scontrol.py parse_sinfo.py parse_sstat.py     [WP2]
      parse_occupants.py every user's running jobs per node + getent names (§5.2.9)    [WP2]
      priority.py        pending priority evidence: priority settings, sprio / sshare parsers, hints (§5.2.10)  [WP2]
      logpaths.py        %x/%j/%A/%a… expansion, WorkDir resolution, merged-stderr rules [WP2]
      actions.py         cancel / hold / release / requeue specs, plans, interpretation [WP2]
      attach.py          Attach plan/ingest, nested node script, section names + thresholds (§7.16)  [WP2]
      parse_attach.py    node.id / listpids / /proc/driver/nvidia / audit parsers (§7.16)  [WP2]
    process/
      collector.py       ProcessCollector (workstations)                                [WP2]
      parse_ps.py        ps parsers                                                     [WP2]
      tasks.py           process-tree grouping, sticky identity, naming, noise filter, exit scan  [WP2]
      actions.py         terminate / interrupt / kill plans (§6.2)                      [WP2]
  core/
    service.py           FROZEN FleetAPI protocol + FleetService (§3.7)                 [WP1]
    store.py             FROZEN-API FleetStore, HostState, ChangeSet                    [WP1]
    clock.py             FROZEN Clock protocol, RealClock, ManualClock                  [WP1]
    poller.py            HostPoller: ticks, cadence modes, backoff, lanes, MonitorView  [WP1]
    actions.py           ActionService: eligibility, plan, execute, action log          [WP1]
    logs.py              LogService: follow streams, line assembly, peek, fetch, search, pager argv  [WP1]
    logsearch.py         cross-job log search (§4.7.1): scope, the `log.search` command, its parser, excerpts (pure)  [WP1]
    cache.py             SQLite cache (§5.5)                                            [WP1]
    doctor.py            per-host diagnostics → DoctorReport                            [WP1]
    forwards.py          port forwards: PortForward, service URLs in logs, free local port (§7.15)  [WP1]
    events.py            job transitions → events (baseline, coalescing, [notify] selection), pure (§7.14)  [WP1]
    attach.py            AttachService: policy, consent, auto-refresh timer, autostop, srun rows (§7.16)  [WP1]
  ui/
    app.py               RunFleetApp; screen stack; global bindings; store→message bridge (≤ 4 Hz)  [WP3]
    messages.py          FleetChanged, HostStatusChanged, ActionFinished, …             [WP3]
    keymap.py            single source of bindings + help text (§7.8)                   [WP3]
    render.py            pure Rich renderables: gpu_bar, state_badge, host_chip, mini_util  [WP3]
    theme.py             colours mixed from the terminal background, dark/light/ansi modes, thresholds, NO_COLOR  [WP3]
    termbg.py            OSC 11 query of the terminal background before the TUI starts (§7.11)  [WP3]
    runfleet.tcss
    commands.py          command palette Provider (§7.10)                               [WP3]
    query.py             filter language (§7.6): parse + match Job → bool (pure)        [WP3]
    fleet_layout.py      roomy Fleet view (§7.4): row-height plan, ranked card lists, row templates, fleet panel data (pure)  [WP3]
    saved_views.py       saved Jobs views (§7.6): name rules, ≤ 20 views, tolerant ui.json `views` reader/writer (pure)  [WP3]
    gpu_watch.py         GPU availability watches (§7.14): per-node matching, readings, evaluator, prompt parser, ui.json `watches` (pure)  [WP3]
    everyday.py          copy menu + support bundle, forwards, log in to all, free GPUs (§7.15)  [WP3]
    notify.py            Alerts: notification delivery (toast, OSC 9/777, hook), watch list, away digest, GPU watches (§7.14)  [WP3]
    screens/  main.py job_detail.py host_detail.py log_viewer.py help.py confirm.py action_menu.py host_error.py
              settings.py forwards.py notifications.py
              attention.py ("needs you" picker of the Fleet view, `a`, §7.4)
              save_view.py (palette › Save view as…: the name prompt, §7.6)
              gpu_watch.py (palette › Watch free GPUs… prompt and the Watches list, §7.14)
              log_search.py (palette › Search logs… / ^f in Jobs: the prompt and the results, §4.7.1)  [WP3]
    widgets/  header.py hintbar.py tabs.py fleet_view.py host_card.py jobs_view.py job_table.py job_preview.py
              gpus_view.py gpu_panel.py proc_table.py node_table.py log_view.py filter_bar.py
              attach_panel.py (Attach status line + live GPU rows, §7.16)
              fleet_panel.py (roomy Fleet view: free GPUs by type + what changed, §7.4)  [WP3]
  demo/
    world.py             simulated fleet ("default" scenario) + fault-injection hooks for tests  [WP4]
    render.py            world → byte-exact command outputs                             [WP4]
    transport.py         DemoTransport                                                  [WP4]
tests/
  __init__.py            every test directory is a package (unique module names; `from tests.conftest import …`)
  conftest.py            sample()/meta()/frame() helpers, ManualClock fixture, no-ssh guard, `real` opt-in  [WP2]
  unit/ [WP1]   backends/ integration/ mockbin/ [WP2]   ui/ (incl. ui/fakes.py FakeService) [WP3]   demo/ cli/ [WP4]
scripts/smoke_real.sh    read-only smoke test against real hosts (manual, opt-in)       [WP4]
.github/workflows/ci.yml                                                                [WP4]
```

### 3.3 Core data model — `src/runfleet/models.py` (FROZEN)

Conventions: every dataclass is `frozen=True, slots=True`; collections are tuples or `Mapping`s treated as immutable.
Datetimes are timezone-aware UTC (display converts to local). Sizes are **bytes** (`int`) except where the source
reports MiB and the field name says so (`*_mib`: GPU memory from nvidia-smi/sstat, node memory from sinfo).
`to_json`/`from_json` round-trip every type here (cache, `snapshot --json`).

```python
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, TypeVar

T = TypeVar("T")

# ---------- hosts ----------
class ConnState(StrEnum):
    NEW = "new"                  # configured, never probed
    CONNECTING = "connecting"    # first probe / ensure_master in flight (no probe is sent while a master is being made)
    OK = "ok"
    DEGRADED = "degraded"        # transport fine, >= 1 section failing (slurmdbd down, nvidia-smi hung, parse errors)
    RETRYING = "retrying"        # 1-2 consecutive transport failures, backing off
    UNREACHABLE = "unreachable"  # >= 3 consecutive transport failures, or DNS failure
    NEEDS_AUTH = "needs_auth"    # BatchMode auth failure -> polling paused until connect
    HOSTKEY = "hostkey"          # unknown/changed host key -> paused until connect
    DISABLED = "disabled"        # user toggled off / enabled = false

class ErrorKind(StrEnum):        # defined ONLY here; errors.py holds exception classes only
    AUTH = "auth"
    HOSTKEY = "hostkey"
    DNS = "dns"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    JUMP = "jump"
    MUX = "mux"
    SSH = "ssh"
    REMOTE = "remote"
    MISSING = "missing"
    PARSE = "parse"
    PERMISSION = "permission"
    INTERNAL = "internal"

MuxMode = Literal["user", "managed", "off", "session", "local", "none"]   # "none" = fake/demo transports;
                                                                         # "session": one ssh + agent (§4.8)

class Visibility(StrEnum):
    HIDDEN = "hidden"
    VISIBLE = "visible"
    FOCUSED = "focused"

@dataclass(frozen=True, slots=True)
class HostError:
    kind: ErrorKind
    summary: str                 # one line for badges/toasts, <= 80 chars
    detail: str = ""             # stderr excerpt (<= 4 KiB)
    section: str | None = None   # None = transport level
    at: datetime | None = None

@dataclass(frozen=True, slots=True)
class SectionStatus:
    name: str
    outcome: str                 # SectionOutcome value (probe/section.py)
    ok: bool
    rc: int | None
    ms: int | None
    last_ok: datetime | None
    error: HostError | None = None

@dataclass(frozen=True, slots=True)
class Capabilities:
    remote_user: str
    uid: int | None
    login_host: str                          # `hostname` of the ssh target
    tz_offset: timedelta                     # `date +%z` at detect; ONLY for ISO timestamps (fixtures/fallback)
    clock_skew: timedelta                    # remote `date +%s` - local, informational
    tools: frozenset[str]                    # names found by `command -v`
    profile: tuple[str, ...] = ()            # collector names, e.g. ("slurm", "procfs")
    slurm_version: str | None = None
    cluster_name: str | None = None
    slurm_config: Mapping[str, str] = field(default_factory=dict)   # AccountingStorageType, PreemptMode, MinJobAge, ...
    acct_gather_freq_s: int | None = None
    gpu_tres: bool | None = None             # cluster records gres/gpuutil (None = not known yet), §5.1
    sacct_fields: frozenset[str] = frozenset()
    associations: tuple[tuple[str, str, str, str], ...] = ()       # (cluster, account, partition, qos)
    gpu_partitions: tuple[str, ...] = ()     # GPU partitions the user may submit to (scontrol show partition), §5.1
    detect_rev: int = 0                      # DETECT_REV of the detection (cache.load_caps drops older ones: detect again)
    driver_version: str | None = None        # nvidia "KMD version" (fallback "DRIVER version")
    node_ssh: Literal["adopt", "denied"] | None = None   # Attach (§7.16), learned via caps_patch by the first node
                                             # ssh: "adopt" = login→node ssh lands in a job (pam_slurm_adopt);
                                             # "denied" = refused before/at auth (lyra). None = not tried yet

@dataclass(frozen=True, slots=True)
class HostMetrics:
    ncpu: int | None
    load: tuple[float, float, float] | None
    cpu_pct: float | None        # /proc/stat delta between two samples; None on the first
    mem_total: int | None
    mem_available: int | None
    swap_total: int | None
    swap_free: int | None
    uptime_s: float | None
    sampled_at: datetime

@dataclass(frozen=True, slots=True)
class HostStatus:
    name: str
    state: ConnState
    error: HostError | None = None
    last_ok: datetime | None = None
    last_attempt: datetime | None = None
    latency_s: float | None = None           # wall time of the last successful probe
    consecutive_failures: int = 0
    next_poll_at: datetime | None = None
    tick_interval: float | None = None       # current effective tick period after all factors (§5.4)
    in_flight: bool = False
    mux: MuxMode = "none"
    ssh_log: str | None = None               # `-E` log of master creation / connect, shown in HostErrorScreen
    sections: Mapping[str, SectionStatus] = field(default_factory=dict)
    controller_slowdown: float = 1.0         # busy Slurm controller: Slurm cadence ×2 … ×8 (§5.4); 1 = normal

# ---------- jobs ----------
class JobKind(StrEnum):
    SLURM = "slurm"
    TASK = "task"                # workstation process tree

class StateGroup(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ENDED = "ended"              # workstation task gone, exit status unknown (shown next to failures)
    UNKNOWN = "unknown"

class JobState(StrEnum):         # value = short code shown in the ST column
    PENDING = "PD"
    CONFIGURING = "CF"
    REQUEUED = "RQ"
    RUNNING = "R"
    COMPLETING = "CG"
    SUSPENDED = "S"
    STOPPED = "ST"
    COMPLETED = "CD"
    EXITED = "EX"
    FAILED = "F"
    TIMEOUT = "TO"
    OUT_OF_MEMORY = "OOM"
    NODE_FAIL = "NF"
    PREEMPTED = "PR"
    BOOT_FAIL = "BF"
    DEADLINE = "DL"
    CANCELLED = "CA"
    UNKNOWN = "?"

    @property
    def group(self) -> StateGroup:
        return _STATE_GROUP[self]

    @property
    def is_terminal(self) -> bool:
        return _STATE_GROUP[self] not in (StateGroup.QUEUED, StateGroup.ACTIVE, StateGroup.UNKNOWN)

_STATE_GROUP: dict[JobState, StateGroup] = {
    JobState.PENDING: StateGroup.QUEUED, JobState.CONFIGURING: StateGroup.QUEUED, JobState.REQUEUED: StateGroup.QUEUED,
    JobState.RUNNING: StateGroup.ACTIVE, JobState.COMPLETING: StateGroup.ACTIVE, JobState.SUSPENDED: StateGroup.ACTIVE,
    JobState.STOPPED: StateGroup.ACTIVE,
    JobState.COMPLETED: StateGroup.DONE,
    JobState.EXITED: StateGroup.ENDED,
    JobState.FAILED: StateGroup.FAILED, JobState.TIMEOUT: StateGroup.FAILED, JobState.OUT_OF_MEMORY: StateGroup.FAILED,
    JobState.NODE_FAIL: StateGroup.FAILED, JobState.PREEMPTED: StateGroup.FAILED, JobState.BOOT_FAIL: StateGroup.FAILED,
    JobState.DEADLINE: StateGroup.FAILED,
    JobState.CANCELLED: StateGroup.CANCELLED,
    JobState.UNKNOWN: StateGroup.UNKNOWN,
}

_TASK_ID = re.compile(r"^p(\d+)-(\d+)$")

@dataclass(frozen=True, slots=True, order=True)
class JobKey:
    host: str                    # RunFleet host name (config `name`)
    id: str                      # Slurm display id: "2332861", "31270818_3", "31270818_[1-24%1]"
                                 # task: "p<root pid>-<root start epoch>", e.g. "p642229-1788696459"

    def __str__(self) -> str:
        return f"{self.host}:{self.id}"

    @property
    def label(self) -> str:      # what the ID column shows: task -> "p642229", Slurm unchanged
        m = _TASK_ID.match(self.id)
        return f"p{m.group(1)}" if m else self.id

@dataclass(frozen=True, slots=True)
class NodeAlloc:
    node: str
    cpu_ids: str | None          # "64-71"
    mem_bytes: int | None
    gpu_indices: tuple[int, ...] # physical IDX from scontrol -d GRES=...(IDX:0-1); () if unknown
    gres: str | None             # raw "gpu:nvidia_h200_nvl:2(IDX:0-1)"

@dataclass(frozen=True, slots=True)
class ResourceAlloc:
    nodes: int | None
    cpus: int | None
    mem_bytes: int | None        # total for the job
    gpus: int                    # 0 when none
    gpu_type: str | None         # normalized lower-case gres type: "nvidia_a100-sxm4-40gb", "rtx6000"
    tres: Mapping[str, str]      # raw TRES dict (alloc if running, req if pending)
    per_node: tuple[NodeAlloc, ...] = ()
    gpu_type_source: Literal["tres-alloc", "scontrol", "tres-req", "sinfo", "cache", "nvidia-smi"] | None = None

@dataclass(frozen=True, slots=True)
class LogRef:
    stream: Literal["stdout", "stderr"]
    path: str | None             # absolute, expanded; None if unknown
    pattern: str | None = None   # raw Slurm pattern ("…/logs/%x_%j.out") if that is all we have
    merged: bool = False         # stderr goes to the stdout file
    source: Literal["scontrol", "squeue", "sacct", "template", "procfd", "cache", "demo"] = "squeue"
    size: int | None = None      # from the last stat
    mtime: datetime | None = None
    error: str | None = None     # "No such file", "Permission denied"

@dataclass(frozen=True, slots=True)
class Proc:                      # one `ps` row (workstations)
    pid: int
    ppid: int
    pgid: int | None             # None when the source did not report it (nvidia.apps ps)
    user: str
    etime_s: int
    cpu_pct: float
    mem_pct: float
    rss_bytes: int
    stat: str
    args: str
    task: JobKey | None = None   # set by ProcessCollector for members of a task

@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    gpu_uuid: str
    gpu_index: int | None
    used_mem_mib: float | None
    user: str | None
    command: str | None
    etime_s: int | None
    cpu_pct: float | None
    job: JobKey | None           # task that owns the pid (set by ProcessCollector)

@dataclass(frozen=True, slots=True)
class GPU:
    host: str
    index: int
    uuid: str | None
    name: str                    # "NVIDIA RTX A6000"
    util_pct: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    mem_util_pct: float | None
    temp_c: float | None
    power_w: float | None
    power_limit_w: float | None
    fan_pct: float | None
    pstate: str | None
    source: Literal["nvidia-smi", "demo"]
    processes: tuple[GPUProcess, ...] = ()
    sampled_at: datetime | None = None
    bus_id: str | None = None    # "00000000:41:00.0" (nvidia-smi pci.bus_id); Attach pairs GPUs by uuid or bus id
    slurm_idx: int | None = None # Slurm GRES IDX (Attach, §7.16 IDX mapping); `index` stays nvidia-smi's own index,
                                 # which inside a job step is 0..n-1 and NOT the physical/Slurm number

@dataclass(frozen=True, slots=True)
class JobGpuUsage:
    source: Literal["sstat", "nvidia-smi", "demo"]
    n_gpus: int
    util_mean_pct: float | None  # sstat: Tot gpuutil / n_gpus clamped to 100; nvidia-smi: mean over the task's GPUs
    util_sum_pct: float | None   # sstat: TRESUsageInTot gres/gpuutil (sum over tasks)
    util_peak_pct: float | None  # sstat: min(100, Max gpuutil x NTasks / n_gpus) (peak per GPU, est.); nvidia-smi: max
    mem_used_mib: float | None   # total over the job's GPUs
    mem_total_mib: float | None  # nvidia-smi, or gpu_types table x n_gpus
    sampled_at: datetime | None = None
    note: str = ""               # "Slurm accounting · refreshed every 60 s", "GPU-level; shared GPUs count others"

@dataclass(frozen=True, slots=True)
class PriorityFactor:            # one term of a pending job's priority/multifactor priority (§5.2.10), as sprio weights it
    name: Literal["age", "assoc", "fairshare", "jobsize", "partition", "qos", "tres", "nice"]
    value: float                 # weighted contribution (sprio %A %B %F %J %P %Q %T; nice: minus the adjustment)
    weight: float | None = None  # the cluster's PriorityWeight<Factor>: the most this factor can add; None for nice

@dataclass(frozen=True, slots=True)
class FairShare:                 # one `sshare -U` row of the user: fair-share standing in one account (§5.2.10)
    account: str
    user: str
    raw_shares: str | None       # "1", "parent"
    norm_shares: float | None    # share among the siblings, 0..1
    raw_usage: int | None        # decayed usage (TRES-seconds)
    effective_usage: float | None
    fairshare: float | None      # the fair-share factor 0..1 (×PriorityWeightFairShare in the job's priority)
    level_fs: float | None       # NormShares / EffectvUsage (> 1: under-served); None: "inf" (no usage) or unknown

@dataclass(frozen=True, slots=True)
class PendingPriority:           # why a pending Slurm job waits, as evidence — never a start time or queue position
    ranked: bool                 # False: the cluster does not rank by priority (priority/basic, or every weight 0)
    priority: float | None = None            # sprio %Y of the job's best partition; None: not read yet
    partition: str | None = None             # partition of that sprio row
    factors: tuple[PriorityFactor, ...] = () # weighted factors (weight > 0), largest contribution first
    dominant: str | None = None              # the largest contribution (None: all 0 / not read)
    others: tuple[tuple[str, float], ...] = ()   # (partition, priority) of the job's other sprio rows
    fairshare: tuple[FairShare, ...] = ()    # the user's sshare rows; the job's account first
    hints: tuple[str, ...] = ()              # plain language, most telling first (≤ 4)
    sampled_at: datetime | None = None       # when sprio was read (None: never)
    half_life_s: float | None = None         # PriorityDecayHalfLife (0 = usage does not decay)

@dataclass(frozen=True, slots=True)
class Job:
    key: JobKey
    kind: JobKind
    name: str
    user: str
    state: JobState
    raw_state: str               # "CANCELLED by 10001", "RUNNING", "exited (log: Traceback ...)"
    job_id_raw: str              # Slurm: squeue JobID / sacct JobIDRaw (for scontrol, sstat, sacct -j, %j); task: root pid
    reason: str | None = None    # Slurm Reason, raw
    held: bool = False
    submit_time: datetime | None = None
    start_time: datetime | None = None   # PD: scheduler estimate (est_start=True)
    est_start: bool = False
    end_time: datetime | None = None
    elapsed: timedelta | None = None     # at observed_at; the UI extrapolates for ACTIVE jobs
    time_limit: timedelta | None = None  # None = unlimited/unknown
    exit_code: int | None = None
    exit_signal: int | None = None
    derived_exit: str | None = None
    partition: str | None = None
    qos: str | None = None
    account: str | None = None
    alloc: ResourceAlloc = field(default_factory=lambda: ResourceAlloc(None, None, None, 0, None, {}))
    nodes: tuple[str, ...] = ()          # expanded hostnames
    nodelist: str | None = None          # raw compressed
    batch_host: str | None = None
    workdir: str | None = None
    command: str | None = None
    logs: tuple[LogRef, ...] = ()
    array_job_id: str | None = None
    array_task: str | None = None        # "3" or "[1-20%1]"
    restarts: int = 0
    pids: tuple[int, ...] = ()           # task: all current members, root first
    cpu_pct: float | None = None         # task: sum over members
    rss_bytes: int | None = None         # task: sum over members
    gpu: JobGpuUsage | None = None
    hints: tuple[str, ...] = ()          # "log idle 47m", "GPU idle 20m", "will never run: dependency failed"
    is_mine: bool = True
    source: Literal["squeue", "sacct", "vanished", "ps", "cache", "demo"] = "squeue"
    observed_at: datetime | None = None
    raw: Mapping[str, str] = field(default_factory=dict)   # every raw field (detail "Raw" tab)
    priority: PendingPriority | None = None  # own pending Slurm jobs: priority evidence (§5.2.10); else None

# ---------- cluster ----------
@dataclass(frozen=True, slots=True)
class GresCount:
    name: str                    # "gpu" | "shard"
    type: str | None             # "nvidia_a100_80gb_pcie_3g.40gb" (MIG profiles are distinct types)
    total: int
    used: int
    used_indices: str | None     # "0,2-3" / None for N/A

@dataclass(frozen=True, slots=True)
class Node:
    name: str
    partitions: tuple[str, ...]
    state: str                   # base state, lower-case: idle mixed allocated completing planned reserved
                                 # drained draining down fail failing maint future inval unknown power_down ...
    flags: frozenset[str]        # "*" not responding, "~" powered off, "#" powering up, "!" pending power down,
                                 # "%" powering down, "$" maintenance, "@" pending reboot, "^" rebooting,
                                 # "-" planned by backfill (still usable), "+" multiple states
    usable: bool                 # contributes free GPUs (rule §5.2.3)
    cpus_alloc: int | None
    cpus_idle: int | None
    cpus_other: int | None
    cpus_total: int | None
    mem_total_mib: int | None
    mem_alloc_mib: int | None
    mem_free_mib: int | None
    gres: tuple[GresCount, ...]
    reason: str | None
    jobs: tuple[NodeJob, ...] | None = None   # running jobs of every user (§5.2.9); None = not sampled lately
    jobs_at: datetime | None = None           # when `jobs` was sampled
    # jobs_seen(now): `jobs` while younger than NODE_JOBS_TTL (3 min), else None

@dataclass(frozen=True, slots=True)
class NodeJob:
    """A running job on one node, any user (§5.2.9: sampled while the host's detail is open)."""
    job_id: str                  # display id (`123_4` for an array task)
    user: str
    real_name: str | None        # GECOS of `getent passwd` (None: unknown or not looked up yet)
    account: str | None
    partition: str | None
    name: str
    state: str                   # squeue compact state: "R" / "CG"
    gpus: int | None             # GPUs on this node (None: unknown); a multi-node job's split when not exact
    gpus_exact: bool             # False: an even split of a multi-node job's total
    gpu_type: str | None
    elapsed_s: int | None
    limit_s: int | None          # None: UNLIMITED / unknown
    mine: bool
    # left_s (property): limit_s - elapsed_s, ≥ 0; None without both

@dataclass(frozen=True, slots=True)
class PartitionSummary:
    name: str
    nodes: int
    gpus_total: int
    gpus_free: int               # sum over usable nodes of (total - used)
    gpus_free_planned: int       # part of gpus_free on nodes that are planned for a pending job ("-" / planned)
    gpu_types: Mapping[str, tuple[int, int]]   # type -> (free, total)
    states: Mapping[str, int]    # base state -> node count

# ---------- details ----------
@dataclass(frozen=True, slots=True)
class JobStep:
    step_id: str                 # "2332861.batch", "31354789.0"
    name: str
    state: JobState
    raw_state: str
    exit_code: int | None
    exit_signal: int | None
    elapsed: timedelta | None
    start: datetime | None
    end: datetime | None
    nodelist: str | None
    ntasks: int | None
    max_rss_bytes: int | None
    ave_cpu: timedelta | None
    tres_usage_max: Mapping[str, str] = field(default_factory=dict)
    tres_usage_tot: Mapping[str, str] = field(default_factory=dict)
    total_cpu: timedelta | None = None   # sacct TotalCPU (detail probe, when caps.sacct_fields has it; §5.2.8)

@dataclass(frozen=True, slots=True)
class JobDetail:
    job: Job
    fields: Mapping[str, str]            # scontrol key/values, or ps/procfs facts for tasks
    per_node: tuple[NodeAlloc, ...]
    steps: tuple[JobStep, ...]
    logs: tuple[LogRef, ...]             # with fresh stat info
    gpu: JobGpuUsage | None
    procs: tuple[Proc, ...]              # task members; () for Slurm in v0.1
    procs_note: str                      # e.g. "per-process view needs compute-node access (v0.2)"
    env: Mapping[str, str]               # allow-listed env of the task root (Appendix A process.detail)
    explanation: tuple[str, ...]         # "batch step FAILED 143:0 = SIGTERM (128+15): preempted"
    fetched_at: datetime
    errors: tuple[HostError, ...] = ()
    efficiency: tuple[str, ...] = ()     # plain efficiency card (util/efficiency.lines), Slurm jobs that ran
    live: JobLive | None = None          # Attach (§7.16): the job's latest live sample this session, filled in by
                                         # FleetService.job_detail from AttachState.live (collectors leave it None)

# ---------- attach (§7.16) ----------
AttachMode = Literal["ssh", "srun"]      # "ssh": login → `ssh <node>` landing via pam_slurm_adopt (no sacct row);
                                         # "srun": `srun --jobid --overlap` one-shot step (ONE permanent sacct row)

@dataclass(frozen=True, slots=True)
class CgroupStats:                       # one job's cgroup v2 directory on one node (bytes / counts)
    mem_current: int | None = None       # memory.current
    mem_peak: int | None = None          # memory.peak
    mem_max: int | None = None           # memory.max; "max" (unlimited) -> None
    anon: int | None = None              # memory.stat anon
    file: int | None = None              # memory.stat file
    shmem: int | None = None             # memory.stat shmem (counts against memory.max; part of sstat MaxRSS)
    oom: int | None = None               # memory.events oom
    oom_kill: int | None = None          # memory.events oom_kill
    cpu_usage_usec: int | None = None    # cpu.stat usage_usec (cumulative)
    cpu_cores: float | None = None       # Δcpu_usage_usec / Δwall between this and the previous sample of the same
                                         # (node, job); None on the first sample (computed by ingest_attach)

@dataclass(frozen=True, slots=True)
class NodeInspect:                       # one compute node of one Attach sample
    node: str                            # short hostname ("n-37-08")
    target_job: str                      # job_id_raw of the attached job
    landed_job: str | None = None        # raw job id the node session ran in: ssh = the job pam_slurm_adopt chose
                                         # (from /proc/self/cgroup, else $SLURM_JOB_ID); srun = target_job
    gpus_of_target: bool | None = None   # True: `gpus` are the target job's; False: they belong to landed_job
                                         # (ssh landed in another own job on this node); None: unknown
    gpus: tuple[GPU, ...] = ()           # GPU.host = RunFleet host name; bus_id/slurm_idx set when known;
                                         # GPU.processes = compute apps of this sample (GPUProcess.job set when the
                                         # pid's job is known from listpids)
    procs: tuple[Proc, ...] = ()         # `ps` rows of the user's processes on the node (all own jobs, task=None)
    step_of_pid: Mapping[int, str] = field(default_factory=dict)   # pid -> "<raw job id>.<step>" (scontrol listpids:
                                         # "724655.batch", "31367626.0", "2332863.extern")
    cgroups: Mapping[str, CgroupStats] = field(default_factory=dict)   # raw job id -> stats (target first; k2 can
                                         # read every own job's cgroup on the node)
    stray_nvidia_smi: int | None = None  # node.self: own nvidia-smi stuck after the sample (D state on two reads 1 s
                                         # apart: hung; >0 = leftover); None = no affirmative answer (cut, unreadable)
    cost_ms: int | None = None           # wall time of this node's hop (outer section ms)
    errors: tuple[HostError, ...] = ()   # node-level problems (section = outer section name)
    other_nvidia_smi: int | None = None  # node.self: the user's other, healthy nvidia-smi there (a job's
                                         # `nvidia-smi -l`, …): informational, never a leftover; None = unknown

@dataclass(frozen=True, slots=True)
class JobLive:                           # one Attach sample of one job (FleetAPI.attach_now, AttachState.live)
    key: JobKey
    mode: AttachMode
    sampled_at: datetime                 # when the login probe finished (local wall clock, UTC)
    nodes: tuple[NodeInspect, ...] = ()  # nodes that answered (possibly with node-level errors); () = no data
    cost_ms: int = 0                     # wall time of the whole login probe
    sacct_rows_added: int = 0            # permanent sacct rows THIS sample created (srun: steps launched; ssh: 0)
    clean: bool | None = None            # audit (§7.16 layer 5): True clean, False leftovers found, None not run or
                                         # inconclusive (e.g. a just-finished rfprobe step still listed)
    audit_detail: str = ""               # "clean" | "rfprobe step 31367626.3 still running" | "1 hung nvidia-smi on
                                         # s71-02" | "2 node ssh clients still running on the login node"
    hints: tuple[str, ...] = ()          # §7.16 hints, ready to show ("GPU1 idle (0%) but holding 85 GB")
    idle_streak: Mapping[str, int] = field(default_factory=dict)   # GPU uuid (else bus id) -> consecutive samples with
                                         # util < 5 % and mem > 1 GiB (carried over from `previous`)
    errors: tuple[HostError, ...] = ()   # sample-level problems (probe failed, audit failed, parse errors)

    @property
    def has_data(self) -> bool:
        return any(n.gpus or n.procs or n.cgroups for n in self.nodes)

class AttachStatus(StrEnum):
    BLOCKED = "blocked"                  # nothing can run now; AttachState.blocked says why (+ hint)
    NEEDS_CONSENT = "needs_consent"      # only srun can read this job: `r` asks for confirmation (AttachState.offer)
    SAMPLING = "sampling"                # a sample is in flight
    LIVE = "live"                        # ssh auto-refresh armed; next sample at AttachState.next_at
    IDLE = "idle"                        # no automatic sample pending (srun after a read, auto off): `r` reads once
    PAUSED = "paused"                    # auto-refresh suspended until the app is focused / input resumes polling
    STOPPED = "stopped"                  # auto-refresh ended (tab closed, 10 min without input, job ended, another job
                                         # attached, 3 failures); `r` or re-opening the tab starts again

@dataclass(frozen=True, slots=True)
class AttachOffer:                       # srun is possible behind a confirmation (texts ready for ConfirmScreen)
    node: str                            # node the srun step would run on
    reason: str                          # one line: "lyra refuses ssh to compute nodes" | "ssh landed in job 2332863"
    title: str                           # "Read job 31367626 with srun?"
    body: str                            # what runs + cost: "…leaves one permanent sacct row 31367626.N rfprobe…"
    preview: str                         # exact srun command line (AttachPlan.preview)

@dataclass(frozen=True, slots=True)
class AttachState:                       # FleetAPI.attach_state; one per job attached this session (kept after stop)
    key: JobKey
    status: AttachStatus
    setting: Literal["ssh", "ssh+srun", "srun", "off"]   # the host's effective `attach` config
    mode: AttachMode | None = None       # mode `r` uses without an explicit mode (None: blocked)
    live: JobLive | None = None          # last sample WITH data (kept across failed samples and stops)
    error: HostError | None = None       # last sample's failure; cleared by the next sample with data
    auto: bool = True                    # auto-refresh wanted (ssh only; `a` toggles); srun never auto-refreshes
    refresh_s: float | None = None       # the host's attach_refresh while mode == "ssh", else None
    next_at: datetime | None = None      # next automatic sample (None: none scheduled)
    consent: Literal["none", "job"] = "none"   # "job": srun allowed for this job for the rest of this session
    offer: AttachOffer | None = None     # srun available with confirmation (also while LIVE when ssh landed elsewhere)
    blocked: str | None = None           # why nothing can run (status BLOCKED): "attach is off for lyra", "read-only
                                         # mode: srun writes accounting", "job is not running", "lyra: not connected"
    hint: str | None = None              # how to enable more (attach="ssh" + node_ssh="denied": the config line + cost)
    stopped: str | None = None           # why PAUSED/STOPPED: "app in background", "polling paused (idle)", "no input
                                         # for 10 min", "tab closed", "job is no longer running", "another job attached"
    srun_rows: int = 0                   # sacct rows this session's srun samples added for this job
    node: str | None = None              # node pinned by attach_start(node=...); None = backend default
    failures: int = 0                    # consecutive failed samples (auto-refresh stops at 3)

# ---------- snapshot ----------
@dataclass(frozen=True, slots=True)
class HostSnapshot:
    host: str
    taken_at: datetime
    caps: Capabilities | None
    jobs: Mapping[str, Job]              # by JobKey.id
    gpus: tuple[GPU, ...]
    procs: tuple[Proc, ...]              # workstation process table: own task members + GPU pids of all users
    nodes: tuple[Node, ...]
    partitions: tuple[PartitionSummary, ...]
    metrics: HostMetrics | None

# ---------- actions, import, doctor ----------
@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str               # "slurm.cancel", "task.terminate", ...
    target: JobKey
    ok: bool
    message: str                 # "cancelled 2332861", "permission denied", "2 processes survived"
    details: tuple[str, ...] = ()   # per-pid / per-id lines ("sent 642234", "killed 786345", "new 790001")
    rc: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime | None = None
    duration_s: float | None = None

@dataclass(frozen=True, slots=True)
class DetectReport:              # FleetAPI.probe_alias (hosts import)
    alias: str
    state: ConnState             # OK, NEEDS_AUTH, HOSTKEY, UNREACHABLE, ...
    error: HostError | None
    caps: Capabilities | None    # caps.profile tells slurm vs workstation
    mux: MuxMode
    ssh_summary: Mapping[str, str]   # from `ssh -G`: user, hostname, port, proxyjump, controlpath, controlpersist
    latency_s: float | None

@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str                    # "ssh -G", "mux", "master", "probe", "profile", "tools", "clock", "gpu_tres"
    ok: bool | None              # None = informational
    detail: str

@dataclass(frozen=True, slots=True)
class DoctorReport:
    host: str
    checks: tuple[DoctorCheck, ...]
    warnings: tuple[str, ...] = ()

def to_json(obj: Any) -> Any:
    """Dataclass/enum/datetime/timedelta/frozenset -> JSON-compatible value (datetimes ISO 8601 UTC)."""
    ...

def from_json(cls: type[T], data: Any) -> T:
    """Inverse of to_json for every dataclass in this module."""
    ...
```

### 3.4 Probe contract — `src/runfleet/probe/section.py` (FROZEN)

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

class SectionOutcome(StrEnum):
    OK = "ok"                    # rc == 0 and stdout non-empty
    EMPTY = "empty"              # rc == 0, no stdout, no stderr: a valid empty answer (no jobs, no GPU procs)
    FAILED = "failed"            # rc not in {0, 124, 127, 137}
    STDERR_ONLY = "stderr_only"  # rc == 0, stdout empty, stderr non-empty (lyra `--json | head` lesson) -> failure
    TIMEOUT = "timeout"          # rc 124/137 (remote timeout or abandoned), or unfinished at local timeout
    MISSING = "missing"          # a `requires` tool is absent (rc 127 + sentinel)
    INCOMPLETE = "incomplete"    # begin marker without end marker (connection died mid-section)
    NOT_RUN = "not_run"          # no marker at all (script died before it)

@dataclass(frozen=True, slots=True)
class Section:
    name: str                    # "<owner>.<part>", regex ^[a-z]+\.[a-z0-9_.]+$, unique in a probe
    cmd: str                     # POSIX sh snippet; every data value already validated + shlex-quoted
    timeout: float = 15.0        # remote `timeout -k 2 <t>` when available
    requires: tuple[str, ...] = ()           # tools that must exist (`command -v`), else MISSING
    late: bool = False           # rendered after all non-late sections (may hang in D state: nvidia.*)
    meta: Mapping[str, object] = field(default_factory=dict)   # structured args for Fake/Demo transports (Appendix A)
    max_stdout: int = 8_000_000  # client-side truncation guard

    @property
    def owner(self) -> str:      # routing key (§3.1): collector name, "detect" or "log"
        return self.name.split(".", 1)[0]

@dataclass(frozen=True, slots=True)
class SectionResult:
    name: str
    outcome: SectionOutcome
    rc: int | None
    stdout: str                  # utf-8, errors="replace"; exact command output (only the framing "\n" removed)
    stderr: str                  # <= 4 KiB
    ms: int | None               # remote wall time (None without GNU date)

    @property
    def ok(self) -> bool:
        return self.outcome in (SectionOutcome.OK, SectionOutcome.EMPTY)

@dataclass(frozen=True, slots=True)
class ProbeScript:
    sections: tuple[Section, ...]
    nonce: str                   # 12 hex chars, secrets.token_hex(6)
    purpose: Literal["poll", "detect", "detail", "action", "log", "doctor", "attach"] = "poll"

@dataclass(frozen=True, slots=True)
class StreamSpec:
    name: str                    # "log.tail"
    cmd: str                     # POSIX sh, runs until stdin closes (watchdog form, §4.7)
    meta: Mapping[str, object] = field(default_factory=dict)
```

### 3.5 Transport contract — `src/runfleet/transport/base.py` (FROZEN)

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from runfleet.models import HostError, MuxMode
from runfleet.probe.section import ProbeScript, SectionResult, StreamSpec

@dataclass(frozen=True, slots=True)
class RunResult:
    rc: int | None               # None if killed by our local timeout
    stdout: bytes
    stderr: bytes
    elapsed_s: float
    timed_out: bool

@dataclass(frozen=True, slots=True)
class ProbeResult:
    run: RunResult
    sections: Mapping[str, SectionResult]   # every requested section present (unfinished -> TIMEOUT/INCOMPLETE/NOT_RUN)
    error: HostError | None                 # transport-level failure, classified (§9). None if ssh ran, including a
                                            # local timeout after >= 1 section end marker (§4.2 partial rule)

class StreamHandle(Protocol):
    def chunks(self) -> AsyncIterator[bytes]: ...   # raw stdout blocks (read(65536)), never line-split; ends at EOF
    async def close(self) -> None: ...              # close stdin (remote watchdog kills tail), then terminate
    @property
    def stderr_text(self) -> str: ...               # last <= 4 KiB of stderr
    @property
    def returncode(self) -> int | None: ...

class Transport(Protocol):
    host: str                                       # RunFleet host name
    kind: Literal["openssh", "local", "fake", "demo"]
    @property
    def mux(self) -> MuxMode: ...                   # resolved by prepare(); "none" before
    @property
    def ssh_log(self) -> str | None: ...            # `-E` log path for masters/connect (None: local/fake/demo)
    async def prepare(self, *, force: bool = False) -> HostError | None: ...  # ssh -G, mux policy, ensure master
                                                    # (BatchMode); idempotent; force = a user retry (manual `r`,
                                                    # after connect) that bypasses the 30 s master throttle
    async def check_master(self) -> bool: ...       # `ssh -O check` (session: alive + a ping answered within 3 s);
                                                    # True when no master is involved
    async def probe(self, script: ProbeScript, *, timeout: float) -> ProbeResult: ...   # never raises
    async def stream(self, spec: StreamSpec) -> StreamHandle: ...
    def interactive_argv(self, remote_cmd: str | None) -> list[str]: ...  # `ssh -t …` run under App.suspend
    async def before_connect(self) -> bool: ...     # resolve the mux plan + clear a stale socket before connect_argv;
                                                    # True = a working master is already up (nothing to log in to)
    def connect_argv(self) -> list[str] | None: ...  # `ssh -fN … -E <log>`; None if not applicable (local/fake);
                                                    # an `ssh … true` auth check without a master (off, session)
    async def reset_master(self) -> None: ...       # managed masters: `ssh -O exit`, unlink socket; session: end it
    async def close(self) -> None: ...
    # port forwards through the existing master (§7.15); nothing runs remotely
    def forward_blocker(self) -> str | None: ...    # why no forward is possible (no master, local host); None = can
    async def forward(self, local_port: int, node: str, remote_port: int) -> HostError | None: ...
                                                    # `ssh -O forward -L 127.0.0.1:<lport>:<node>:<rport>`
    async def cancel_forward(self, local_port: int, node: str, remote_port: int) -> HostError | None: ...  # -O cancel
```

Lane limits (§4.2) are enforced by core (poller, LogService, ActionService), not by transports. The session
transport's interactive login is NOT part of this contract: it is the optional `InteractiveLogin` protocol of
`transport/session.py` (§4.8), which the UI reaches through `FleetAPI.connect_login` (§3.7).

### 3.6 Collector contract — `src/runfleet/backends/base.py` (FROZEN)

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal

from runfleet.config import GeneralSettings, HostConfig
from runfleet.errors import AttachBlocked
from runfleet.models import (GPU, ActionResult, AttachMode, Capabilities, HostError, HostMetrics, Job, JobDetail,
                             JobKey, JobLive, Node, PartitionSummary, Proc)
from runfleet.probe.section import ProbeScript, Section, SectionResult

class Role(StrEnum):
    SCHEDULER = "scheduler"
    MONITOR = "monitor"

@dataclass(frozen=True, slots=True)
class Cadence:
    group: str                   # "<collector>.<name>", unique per host: "slurm.jobs", "procfs.host", ...
    interval: float              # interval/piggyback: period in seconds at VISIBLE; event: ignored (use 0)
    floor: float = 0.0           # never faster than this, whatever the factors
    mode: Literal["interval", "piggyback", "event"] = "interval"
    # interval : defines the host tick (tick = min effective interval over interval groups) and is due at a tick
    #            once its own period has elapsed
    # piggyback: never defines the tick; due at a tick once `interval` has elapsed (procfs.host on Slurm hosts)
    # event    : due only when requested via CollectorOutput.followups or HostPoller.wake(groups=...); served at
    #            the next tick
    hidden_factor: float = 3.0   # multiplier when the host is not on screen
    focused_factor: float = 1.0  # multiplier when the host is focused (detail screen)
    max_backoff: float = 600.0   # cap of this group's failure backoff
    focused_only: bool = False   # due only while the host is focused (a detail screen is open); served at once when
                                 # the host becomes focused and the last run is older than the period (no early
                                 # retry while the host backs off or is paused)

@dataclass(frozen=True, slots=True)
class HostContext:
    host: HostConfig
    general: GeneralSettings
    caps: Capabilities           # caps.profile is the host's collector list
    now: datetime                # wall clock (UTC) at tick

@dataclass(frozen=True, slots=True)
class MonitorView:
    """Latest SUCCESSFUL monitor state of this host, kept across ticks by the poller (not only this tick)."""
    gpus: tuple[GPU, ...] = ()
    gpu_procs: tuple[Proc, ...] = ()     # ps rows of GPU-holding pids (all users), same sample as gpus
    gpus_at: datetime | None = None      # sample time of gpus/gpu_procs; None = never sampled or no GPU monitor
    metrics: HostMetrics | None = None   # metrics.sampled_at is its sample time

    def gpus_age(self, now: datetime) -> float | None:
        return None if self.gpus_at is None else (now - self.gpus_at).total_seconds()

@dataclass(frozen=True, slots=True)
class CollectorOutput:
    jobs: Mapping[str, Job] | None = None          # FULL current job set of this scheduler (None = unchanged)
    gpus: tuple[GPU, ...] | None = None            # scheduler output replaces the monitor's (annotated GPUProcess.job)
    gpu_procs: tuple[Proc, ...] | None = None      # monitors only; feeds MonitorView
    procs: tuple[Proc, ...] | None = None          # scheduler only; HostSnapshot.procs
    nodes: tuple[Node, ...] | None = None
    partitions: tuple[PartitionSummary, ...] | None = None
    metrics: HostMetrics | None = None
    caps_patch: Mapping[str, object] | None = None # Capabilities field -> value learned at runtime ({"gpu_tres": True})
    errors: tuple[HostError, ...] = ()             # section-level, classified
    followups: frozenset[str] = frozenset()        # cadence groups (event or interval) to run at the next tick
    handled: frozenset[str] = frozenset()          # sections whose non-zero exit the collector read as an answer
                                                   # (sprio: no job pending / not multifactor): ok, no backoff
# Merge rule (poller, one tick): EVERY collector of the profile has ingest() called on EVERY tick, with an empty
# results map when none of its sections ran. Outputs are applied monitors first (profile order), scheduler last; for
# every Optional field the last non-None value wins; errors are concatenated; followups and handled are unioned.
# MonitorView is updated from monitor outputs only, and only from fields that are not None. A scheduler that annotates
# GPUs (process) therefore re-annotates MonitorView.gpus from its last known tasks on every tick and returns
# gpus/procs whenever MonitorView.gpus_at is not None, so annotations never blink on ticks without its own sections.

class Collector(ABC):
    name: ClassVar[str]          # "slurm" | "process" | "nvidia" | "procfs" (== section-name prefix)
    role: ClassVar[Role]

    def __init__(self, host: HostConfig) -> None:
        self.host = host

    @classmethod
    @abstractmethod
    def applicable(cls, caps: Capabilities, host: HostConfig) -> bool: ...

    @classmethod
    def capability_sections(cls, host: HostConfig) -> list[Section]:
        return []

    @classmethod
    def ingest_capabilities(cls, results: Mapping[str, SectionResult]) -> Mapping[str, object]:
        return {}                # Capabilities field name -> value

    @abstractmethod
    def cadences(self, ctx: HostContext) -> list[Cadence]: ...

    @abstractmethod
    def sections(self, due: frozenset[str], ctx: HostContext) -> list[Section]: ...

    @abstractmethod
    def ingest(self, results: Mapping[str, SectionResult], ctx: HostContext,
               monitors: MonitorView) -> CollectorOutput: ...

class MonitorCollector(Collector):
    role: ClassVar[Role] = Role.MONITOR

@dataclass(frozen=True, slots=True)
class ActionParam:
    name: str                    # "escalate", "grace_s", "hold"
    kind: Literal["bool", "int"]
    default: bool | int
    label: str                   # "send SIGKILL to survivors after"
    min: int | None = None
    max: int | None = None

@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str                      # §6.1: slurm.cancel slurm.hold slurm.release slurm.requeue
                                 #       task.terminate task.interrupt task.kill
    label: str                   # "Cancel job"
    mutating: bool               # disabled under --readonly
    confirm: bool                # ConfirmScreen before execute (True for every v0.1 action)
    params: tuple[ActionParam, ...] = ()
    disabled_reason: str | None = None   # greyed with reason ("protected partition labpi", "read-only mode")

@dataclass(frozen=True, slots=True)
class ActionPlan:
    spec: ActionSpec
    host: str
    targets: tuple[JobKey, ...]          # eligible targets only
    skipped: tuple[tuple[JobKey, str], ...]   # (target, reason) listed in the confirm body
    params: Mapping[str, object]
    sections: tuple[Section, ...]        # executed as one probe (purpose="action")
    preview: str                         # exact remote command(s) shown in the confirm dialog
    confirm_title: str                   # "Cancel 3 jobs on k2?"
    confirm_body: str                    # targets with name/state, consequences

@dataclass(frozen=True, slots=True)
class AttachPlan:                        # one Attach sample (§7.16), executed as ONE login probe (purpose="attach")
    key: JobKey
    mode: AttachMode
    job_id_raw: str                      # the target job (squeue JobID / sacct JobIDRaw)
    nodes: tuple[str, ...]               # compute nodes sampled; nodes[i] is carried by section "slurm.attach.node<i>"
    sections: tuple[Section, ...]        # "slurm.attach.node0" … "slurm.attach.node<n-1>", then "slurm.attach.audit"
                                         # (late); every node section carries a nested node script (base64 in argv)
    inner: Mapping[str, ProbeScript]     # outer section name -> the nested node script it carries; its framed output
                                         # is that outer section's stdout (parse with probe.framing.parse)
    preview: str                         # the exact login-side command of node0 (shown in the srun confirmation)
    sacct_rows: int = 0                  # permanent sacct rows this plan adds if it runs (srun: len(nodes); ssh: 0)

class SchedulerCollector(Collector):
    role: ClassVar[Role] = Role.SCHEDULER

    # ---- Attach (§7.16). Not abstract: the defaults mean "this scheduler cannot attach" (workstations, third-party
    # collectors). Core calls these only on the scheduler of the job's host; all three are sans-IO.
    def attach_modes(self, job: Job, ctx: HostContext) -> tuple[AttachMode, ...]:
        """Modes that can technically read `job` now, in preference order ("ssh" before "srun"). Ignores the
        `attach` config setting, read-only mode and consent (core applies those policies) but honours what is known:
        () unless the job is RUNNING with known nodes; "ssh" dropped when ctx.caps.node_ssh == "denied"; "srun"
        dropped when srun is missing from ctx.caps.tools or the Slurm version lacks --overlap."""
        return ()

    def plan_attach(self, job: Job, mode: AttachMode, ctx: HostContext, *,
                    node: str | None = None) -> AttachPlan:
        """Plan one sample. `node` limits it to that node (must be one of job.nodes); default: ssh = up to 4 of
        job.nodes (batch host first), srun = the batch host only (one step = one sacct row). Pure and cheap; raises
        errors.AttachBlocked when the job cannot be read this way (not running, unknown node, unsafe value)."""
        raise AttachBlocked(f"{self.name}: attach is not supported for this job")

    def ingest_attach(self, plan: AttachPlan, results: Mapping[str, SectionResult], ctx: HostContext, *,
                      previous: JobLive | None = None) -> tuple[JobLive, Mapping[str, object]]:
        """Parse one sample. Returns the JobLive and a caps_patch ({"node_ssh": "adopt"} after a landing,
        {"node_ssh": "denied"} after an auth refusal; {} otherwise). `previous` = the last JobLive of the same job
        with data (any mode): CPU cores (Δcpu.stat), idle streaks and the "GPU idle" hint need it. Never raises for
        remote failures: they become JobLive.errors / NodeInspect.errors. May update the collector's own state (the
        GPU model from nvidia-smi becomes the job's alloc.gpu_type, gpu_type_source="nvidia-smi", on later ticks)."""
        raise NotImplementedError(f"{self.name}: attach is not supported")

    @abstractmethod
    def actions(self, jobs: Sequence[Job], ctx: HostContext) -> list[ActionSpec]: ...
        # every action this backend offers for these jobs; disabled_reason from state/ownership/protection.
        # A spec is enabled if >= 1 job is eligible.

    @abstractmethod
    def plan_action(self, action_id: str, jobs: Sequence[Job], params: Mapping[str, object],
                    ctx: HostContext) -> ActionPlan: ...
        # pure and cheap: ConfirmScreen calls it again whenever a param changes; raises ActionError if no target
        # is eligible

    @abstractmethod
    def interpret_action(self, plan: ActionPlan, results: Mapping[str, SectionResult],
                         ctx: HostContext) -> list[ActionResult]: ...

    @abstractmethod
    def detail_sections(self, job: Job, ctx: HostContext) -> list[Section]: ...

    @abstractmethod
    def ingest_detail(self, job: Job, results: Mapping[str, SectionResult], ctx: HostContext) -> JobDetail: ...

    @abstractmethod
    def shell_command(self, job: Job | None, ctx: HostContext, node: str | None = None) -> str | None: ...
        # remote command for `ssh -t <alias> <cmd>`; None = plain login shell. Task: cd to its cwd; running Slurm job:
        # a shell inside the job on `node` (default: its batch host); other Slurm jobs: cd to WorkDir on the login node
```

Registration (`backends/registry.py`): `BUILTIN: dict[str, type[Collector]] = {"slurm": SlurmCollector, "process":
ProcessCollector, "nvidia": NvidiaMonitor, "procfs": ProcfsMonitor}`; third-party collectors via
`importlib.metadata.entry_points(group="runfleet.collectors")`. A new scheduler (PBS, LSF, k8s) is one
`SchedulerCollector` subclass + parsers + `applicable` — no change to core or UI, which render `Job`/`ActionSpec`
generically.

### 3.7 Settings, clock, store and the core facade (FROZEN)

`src/runfleet/config.py` — settings objects (durations are seconds as `float`; collectors read only these fields):

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

AttachSetting = Literal["ssh", "ssh+srun", "srun", "off"]   # §7.16; ATTACH_SETTINGS lists the values

@dataclass(frozen=True, slots=True)
class SlurmSettings:
    poll_jobs: float = 15            # floor 10
    poll_history: float = 120        # floor 60
    poll_nodes: float = 120          # floor 60
    poll_usage: float = 60           # floor 30, never below JobAcctGatherFrequency
    poll_logstat: float = 120        # floor 60
    sacct_window: float = 3 * 86400
    users: tuple[str, ...] = ()               # () -> `squeue --me`
    partitions: tuple[str, ...] = ()          # () -> derived (§5.2.1)
    squeue_args: tuple[str, ...] = ()
    log_templates: tuple[str, ...] = ()
    protected_partitions: tuple[str, ...] = ()
    protected_qos: tuple[str, ...] = ()
    attach: AttachSetting = "ssh+srun"   # Attach (§7.16): "ssh" | "ssh+srun" | "srun" | "off"
    attach_refresh: float = 15       # floor 10; ssh auto-refresh period (srun never auto-refreshes)

@dataclass(frozen=True, slots=True)
class WorkstationSettings:
    poll_gpu: float = 5              # floor 2
    poll_procs: float = 10           # floor 5
    poll_logstat: float = 60         # floor 30
    task_min_age: float = 60
    task_min_cpu: float = 5.0        # percent
    ignore: tuple[str, ...] = ()     # regexes appended to the built-in noise list (§5.3.2)
    kill_grace: float = 10

@dataclass(frozen=True, slots=True)
class HostConfig:
    name: str
    ssh: str                         # alias, or "local"
    enabled: bool = True
    backend: Literal["auto", "slurm", "workstation"] = "auto"
    color: str | None = None
    group: str = ""
    mux: Literal["auto", "user", "managed", "off", "session"] = "auto"
    connect_timeout: float = 45
    ssh_options: tuple[str, ...] = ()
    remote_shell: Literal["sh", "login"] = "sh"
    forward_agent: bool = False
    slurm: SlurmSettings = field(default_factory=SlurmSettings)
    workstation: WorkstationSettings = field(default_factory=WorkstationSettings)

@dataclass(frozen=True, slots=True)
class GeneralSettings:
    readonly: bool = False
    mouse: bool = True
    default_view: Literal["fleet", "jobs", "gpus"] = "fleet"
    recent_window: float = 86400     # ended jobs/tasks stay in the default "fresh" filter
    stall_after: float = 1800        # "log idle" hint threshold
    idle_after: float = 600          # no input -> polling x4 slower
    idle_pause_after: float = 7200   # no input -> all polling paused until the next key/mouse event
    max_parallel_ssh: int = 8
    refresh_multiplier: float = 1.0
    pager: str = ""                  # "" -> $PAGER, else "less -R +G"
    remember_hosts: bool = True      # hosts named on the command line are saved once they answer (§8.4)
    update_check: bool = True        # once a day, ask GitHub for the newest RunFleet version (§8.6)

@dataclass(frozen=True, slots=True)
class UISettings:
    gpu_util_thresholds: tuple[float, float] = (10, 75)
    gpu_mem_thresholds: tuple[float, float] = (10, 80)
    jobs_columns: tuple[str, ...] = ("host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util",
                                     "node_reason")

@dataclass(frozen=True, slots=True)
class FleetConfig:
    hosts: tuple[HostConfig, ...]
    general: GeneralSettings = field(default_factory=GeneralSettings)
    ui: UISettings = field(default_factory=UISettings)
    source: Path | None = None       # None for ad-hoc / demo

def load_config(path: Path | None, *, hosts: Sequence[str] = ()) -> FleetConfig:
    """Read config.toml (None -> default path; missing file -> no hosts). `hosts` selects/creates ad-hoc hosts
    (§8.4). Raises errors.ConfigError with a key path ("host[2].poll_jobs: expected duration ...")."""
    ...

def load_config_with_warnings(path: Path | None, *,
                              hosts: Sequence[str] = ()) -> tuple[FleetConfig, tuple[str, ...]]:
    """`load_config` that also returns the validation warnings (unknown keys, clamped intervals) instead of only
    logging them; cli.py hands them to `FleetAPI.notices` so the TUI shows them as the one toast of §8.3."""
    ...

def dump_hosts_toml(hosts: Sequence[HostConfig]) -> str:
    """TOML text with one [[host]] table per host (non-default keys only) plus a commented defaults block."""
    ...
```

`src/runfleet/core/clock.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Protocol

class Clock(Protocol):
    def now(self) -> datetime: ...               # aware UTC
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...

class RealClock:                                 # time.time / time.monotonic / asyncio.sleep
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...

class ManualClock:                               # tests + deterministic UI snapshots
    def __init__(self, start: datetime) -> None: ...
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...   # completes when advance() passes its deadline
    def advance(self, seconds: float) -> None: ...       # wakes due sleepers in deadline order
```

`src/runfleet/core/store.py` — pure Python, no Textual. The UI reads; only core writes.

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime

from runfleet.backends.base import CollectorOutput
from runfleet.config import HostConfig
from runfleet.models import GPU, HostSnapshot, HostStatus, Job, JobKey, StateGroup

@dataclass
class HostState:
    config: HostConfig
    status: HostStatus
    snapshot: HostSnapshot | None        # last good merged snapshot (never cleared by failures)

    def stale(self, now: datetime) -> bool:
        """True if snapshot is None, state not in {OK, DEGRADED}, or
        now - status.last_ok > 2 * status.tick_interval + 5 s (effective interval: hidden/idle throttling is normal)."""
        ...

@dataclass(frozen=True, slots=True)
class ChangeSet:
    host: str
    status: bool = False
    jobs_added: frozenset[str] = frozenset()
    jobs_changed: frozenset[str] = frozenset()   # material changes only (job_material_eq)
    jobs_removed: frozenset[str] = frozenset()
    gpus: bool = False
    procs: bool = False
    nodes: bool = False
    metrics: bool = False
    paused: bool = False                         # idle pause toggled (host == "*")
    live: frozenset[str] = frozenset()           # job ids whose AttachState changed (§7.16): re-read attach_state()

class FleetStore:
    hosts: dict[str, HostState]                  # insertion order = config order = scope numbers 1..9
    idle_paused: bool

    def subscribe(self, cb: Callable[[ChangeSet], None]) -> Callable[[], None]: ...   # returns unsubscribe
    def jobs(self, hosts: Iterable[str] | None = None) -> Iterator[Job]: ...
    def job(self, key: JobKey) -> Job | None: ...
    def gpus(self, hosts: Iterable[str] | None = None) -> Iterator[GPU]: ...
    def counts(self, hosts: Iterable[str] | None = None) -> Mapping[StateGroup, int]: ...   # header totals
    # core only:
    def apply(self, host: str, out: CollectorOutput, taken_at: datetime) -> ChangeSet: ...
    def set_status(self, host: str, status: HostStatus) -> ChangeSet: ...
    def set_idle_paused(self, paused: bool) -> ChangeSet: ...
    def add_host(self, cfg: HostConfig) -> ChangeSet: ...       # Settings / auto-remember (§7.13); notifies
    def remove_host(self, host: str) -> ChangeSet: ...          # ChangeSet(host, status, jobs_removed=all its jobs)
    def reorder(self, names: Iterable[str]) -> ChangeSet: ...   # in place; ChangeSet("*", status=True)
    def notify_live(self, host: str, job_ids: Iterable[str]) -> ChangeSet: ...   # Attach (§7.16): ChangeSet(host,
                                                 # live=ids); no-op for no ids / unknown host; the store holds no
                                                 # attach data (FleetAPI.attach_state does)

def job_material_eq(a: Job, b: Job) -> bool:
    """== ignoring observed_at and elapsed, so a pure time tick does not repaint rows."""
    ...
```

`src/runfleet/core/service.py` — the **only** API the UI calls, and the only one `cli.py` calls besides the entry
points frozen after it. The UI types against `FleetAPI`, so WP3's `tests/ui/fakes.py: FakeService(FleetAPI)` can stand
in before WP1 lands.

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from runfleet.backends.base import ActionPlan, ActionSpec
from runfleet.config import FleetConfig, HostConfig
from runfleet.core.cache import Cache
from runfleet.core.clock import Clock
from runfleet.core.store import FleetStore
from runfleet.models import (ActionResult, AttachMode, AttachState, DetectReport, DoctorReport, JobDetail, JobKey,
                             JobLive, LogRef, Visibility)
from runfleet.transport.base import Transport

TransportFactory = Callable[[HostConfig], Transport]     # cli.py picks OpenSSH / Local / Demo per host

@dataclass(frozen=True, slots=True)
class LogUpdate:
    lines: tuple[str, ...] = ()  # newly completed lines (ANSI kept, CR-collapsed, each <= 16 KiB)
    live: str | None = None      # current unterminated line after its last "\r" (tqdm); replaces the previous one
    note: str | None = None      # "waiting for …/x.out to appear", "stream ended", "permission denied"
    eof: bool = False

class LogSession(Protocol):
    host: str
    ref: LogRef
    def updates(self) -> AsyncIterator[LogUpdate]: ...   # initial tail, then follow; batched <= 10/s
    async def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class LogPeek:
    lines: tuple[str, ...]       # last <= 8 complete lines (live line appended if any)
    size: int | None
    mtime: datetime | None
    fetched_at: datetime
    error: str | None = None

class FleetAPI(Protocol):
    store: FleetStore
    config: FleetConfig
    readonly: bool
    notices: list[str]           # append-only one-shot warnings the UI toasts once each (config warnings from cli,
                                 # corrupt cache moved aside, "another RunFleet (pid N) is polling", §7.7/§8.3)
    config_path: Path | None     # the file Settings edits (cli: -c PATH or the default path, even when the run shows
                                 # ad-hoc hosts); None = in memory only (--demo: nothing is ever written)

    # lifecycle
    async def start(self) -> None: ...           # load cache -> store; start one HostPoller per enabled host
    async def stop(self) -> None: ...
    async def collect_once(self, hosts: Iterable[str] | None = None, *, timeout: float = 60) -> None: ...
        # `snapshot`: prepare + detect + one full poll per host, no pollers left running
    # scheduling hints from the UI
    def set_visibility(self, levels: Mapping[str, Visibility]) -> None: ...
    def note_user_activity(self) -> None: ...    # every key/mouse event: idle throttle reset, idle pause resume
    def set_app_focus(self, focused: bool) -> None: ...   # AppFocus / AppBlur -> x4 when blurred
    def refresh(self, hosts: Iterable[str] | None = None) -> None: ...   # manual; 3 s per-host floor
    def set_rate_multiplier(self, m: float) -> None: ...   # global +/- (0.5 ... 4.0)
    def set_host_enabled(self, host: str, enabled: bool) -> None: ...
    # details and logs
    async def job_detail(self, key: JobKey, *, max_age_s: float = 30) -> JobDetail: ...
    async def peek_log(self, host: str, ref: LogRef, *, nbytes: int = 4096) -> LogPeek: ...
        # preview pane; cached per (host, path); re-fetched only if >= 15 s old and logstat size/mtime changed
    async def stream_log(self, host: str, ref: LogRef, *, tail_bytes: int = 65536,
                         start: int | None = None) -> LogSession: ...
        # JobDetail log pane and LogScreen only; opening one beyond the host's stream lane closes the oldest.
        # start: from that absolute byte offset instead of the last tail_bytes (a log search hit, §4.7.1)
    async def fetch_log(self, host: str, ref: LogRef, *, nbytes: int = 4 << 20) -> Path: ...
        # last nbytes into a private temp file (local $PAGER fallback)
    async def search_logs(self, host: str, query: LogQuery, *, max_matches: int = 200) -> LogSearchResult: ...
        # palette › Search logs… only (never a timer): one `log.search` probe over the host's own recent log files
        # (logsearch.plan: ≤ 20 files, last 256 KiB each, ≤ 1 MiB, ≤ max_matches excerpts); never raises (§4.7.1)
    # actions
    def actions_for(self, keys: Sequence[JobKey]) -> list[ActionSpec]: ...
        # merged per action id across hosts; disabled_reason precedence: read-only mode -> host not OK/DEGRADED
        # ("lynx: needs auth") -> stale data ("data is 4m old — refresh first") -> backend reason
    def plan(self, action_id: str, keys: Sequence[JobKey],
             params: Mapping[str, object] | None = None) -> list[ActionPlan]: ...
        # one plan per host with eligible targets; pure (no I/O); raises errors.ActionError if nothing is eligible
    async def execute(self, plans: Sequence[ActionPlan]) -> list[ActionResult]: ...
        # never raises; then wakes the affected hosts ("slurm.jobs", "slurm.history" / "process.ps")
    def action_log(self) -> Sequence[ActionResult]: ...   # last 200 results of this session
    # interactive argv; the UI runs them inside App.suspend
    async def prepare_connect(self, host: str) -> bool: ...  # Transport.before_connect; True = already connected
    def connect_argv(self, host: str) -> list[str] | None: ...   # call prepare_connect first (UI `c`, `runfleet connect`)
    def connect_login(self, host: str) -> Callable[[], int] | None: ...
        # the transport's InteractiveLogin.login (session transport, §4.8): BLOCKING, the UI runs it inside
        # App.suspend instead of connect_argv (0 = logged in; the session then serves the polls); None otherwise
    async def after_connect(self, host: str, rc: int) -> None: ...   # reset backoff, re-prepare, poll now
    def shell_argv(self, host: str, key: JobKey | None = None) -> list[str]: ...
    def pager_argv(self, host: str, ref: LogRef) -> list[str] | None: ...
        # `ssh -t <alias> less -R +G -- <path>` if `less` is in caps.tools, else None (use fetch_log + local pager)
    def local_pager_argv(self, path: Path | None = None) -> list[str]: ...
        # general.pager / $PAGER / `less -R +G` on a local file; None = runfleet.log (palette "RunFleet log")
    # hosts import and diagnostics (aliases need not be in the config)
    async def probe_alias(self, alias: str) -> DetectReport: ...
    def write_config(self, hosts: Sequence[HostConfig], path: Path | None = None) -> Path: ...
        # new file: dump_hosts_toml; existing file: appends [[host]] tables for names not yet present, never rewrites
    async def doctor(self, host: str, *, reset: bool = False) -> DoctorReport: ...
    # runtime host management and settings (Settings §7.13, auto-remember §8.4); `save=False` changes only this run.
    # Every save is one comment-preserving read-modify-write of config_path (configedit.py) under an flock on the
    # sidecar `.config.toml.lock`; a file that is not valid TOML is never rewritten, and any failure (also OSError
    # while writing: read-only dir, disk full) raises ConfigError with nothing changed, neither file nor fleet.
    def ssh_aliases(self) -> list[str]: ...           # ~/.ssh/config Host aliases (demo: the simulated fleet's)
    def saved_hosts(self) -> list[HostConfig]: ...    # hosts in config_path, file order; raises ConfigError
    def custom_host_keys(self, host: str) -> list[str]: ...
        # keys of the saved [[host]] table beyond name/ssh ([] in memory or unreadable): Settings confirms first
    def add_host(self, cfg: HostConfig, *, save: bool = True) -> HostConfig: ...
        # save a minimal [[host]] table (name, ssh if it differs; unless the name or alias is present; a table that
        # remove_host took out this session, same ssh target, comes back whole with its comment block) and, if not in
        # the store yet, add it there and start its poller at once (instance lock taken); returns the running
        # HostConfig, [defaults.*] of the file applied
    async def remove_host(self, host: str, *, save: bool = True) -> None: ...
        # unsave; if running: cancel and await its poller, close its log streams and transport, release its instance
        # lock, drop it from the store (no task leaks)
    def reorder_hosts(self, names: Sequence[str], *, save: bool = True) -> None: ...
        # store order (= card order and scope numbers) and the order of the [[host]] tables (comments move along)
    def set_general(self, *, save: bool = True, **changes: Any) -> GeneralSettings: ...
        # [general] keys (validated like the file); live: readonly, refresh_multiplier, idle_after/idle_pause_after,
        # pager; at next start: default_view, max_parallel_ssh; the UI applies mouse
    def remember_hosts(self, names: Sequence[str]) -> None: ...
        # hosts named on the command line and not yet saved (by name or alias): saved once they answer (OK, DEGRADED,
        # NEEDS_AUTH, HOSTKEY) while general.remember_hosts; one notice per batch, once none is still connecting:
        # "saved k2, vega — manage in Settings (,)"
    # everyday actions (§7.15): no remote command
    def job_shell_command(self, host: str, key: JobKey | None = None, node: str | None = None) -> str | None: ...
    def forward_blocker(self, host: str) -> str | None: ...
    async def forward_port(self, host: str, node: str, remote_port: int, *, job: JobKey | None = None,
                           scheme: str = "http", path: str = "", label: str = "") -> PortForward: ...
    async def cancel_forward(self, local_port: int) -> None: ...   # stop() cancels every forward
    def forwards(self) -> list[PortForward]: ...
    # Attach (§7.16): live per-GPU / per-process / cgroup readings of ONE running Slurm job, read from its compute
    # node only while JobDetail's GPU or Procs tab shows it. Every state change notifies store subscribers with
    # ChangeSet(host, live={job id}); the UI then re-reads attach_state(key). Never touches the background poll.
    def attach_start(self, key: JobKey, *, node: str | None = None) -> AttachState: ...
        # the GPU/Procs tab of `key` became visible (idempotent while it stays visible). Stops any other attach
        # ("another job attached"). No I/O itself: ssh mode schedules the first sample at once (then every
        # attach_refresh while the rules of §7.16 hold); srun mode never samples here (status NEEDS_CONSENT or
        # IDLE until attach_now). Returns the state right after the call.
    def attach_stop(self, key: JobKey | None = None) -> None: ...
        # tab hidden / screen closed (None = whatever is attached). Cancels the timer and an in-flight AUTO sample;
        # the state (live data, consent, srun_rows) stays readable as STOPPED ("tab closed")
    async def attach_now(self, key: JobKey, *, mode: AttachMode | None = None,
                         consent: Literal["once", "job"] | None = None, node: str | None = None) -> JobLive: ...
        # `r`: one sample now (mode None = the state's mode; "srun" explicitly = the "read with srun" offer).
        # Raises errors.AttachConsentRequired (srun without consent: show e.offer, call again with consent),
        # errors.AttachBlocked (reason [+ hint]). Remote failures do not raise: the JobLive carries errors and
        # has no nodes. ssh: re-arms auto-refresh if the tab is open; a sample < 3 s old is returned as is; an
        # in-flight sample of the same job is awaited, not duplicated
    def attach_set_auto(self, key: JobKey, enabled: bool) -> AttachState: ...
        # `a`: ssh auto-refresh on/off for this job (session only); srun ignores it (never auto-refreshes)
    def attach_state(self, key: JobKey) -> AttachState | None: ...
        # None = never attached this session; otherwise the latest state (also after stop)

class FleetService(FleetAPI):
    def __init__(self, config: FleetConfig, transports: TransportFactory, clock: Clock,
                 cache: Cache | None) -> None: ...
```

Entry points `cli.py` uses besides `FleetAPI` (FROZEN signatures spanning four files; each file adds its own imports;
no other WP1/WP3 internals):

```python
# src/runfleet/transport/__init__.py
def make_transport(host: HostConfig) -> Transport: ...   # LocalTransport if host.ssh == "local", else OpenSSHTransport
# src/runfleet/sshconfig.py
def list_aliases() -> list[str]: ...   # ~/.ssh/config Host aliases, Include-aware, patterns with * ? ! skipped
# src/runfleet/paths.py
def config_file() -> Path: ...         # ~/.config/runfleet/config.toml
def cache_file() -> Path: ...          # ~/.local/state/runfleet/cache.sqlite  (Cache(cache_file()))
def log_file() -> Path: ...            # ~/.local/state/runfleet/runfleet.log
# src/runfleet/ui/app.py
class RunFleetApp(App[None]):
    def __init__(self, service: FleetAPI, *, clock: Clock | None = None, animations: bool = True,
                 persist_state: bool = True, open_settings: bool = False) -> None: ...
        # mouse comes from service.config.general.mouse (cli applies --no-mouse/--readonly to the FleetConfig
        # with dataclasses.replace before building FleetService); clock None -> RealClock
```
`--demo` uses `lambda h: DemoTransport(h, world)` instead of `make_transport`.

`Cache` (`core/cache.py`), `HostPoller`, `ActionService`, `LogService` and `Doctor` are WP1-internal; only the types
above cross work-package boundaries. The UI holds exactly one `FleetAPI` and reads `FleetAPI.store`.

---

## 4. Transport

### 4.1 Host resolution and the mux policy (`transport/mux.py`, `sshconfig.py`)
On `prepare()` (first poll and after every `connect`), once per session: `ssh -G {config ssh_options} <alias>` (local,
~5 ms; with the host's `ssh_options`, so user/host/port/ProxyCommand are the ones RunFleet will really use) → `user`,
`hostname`, `port`, `proxyjump`, `proxycommand`, `controlmaster`, `controlpath`, `controlpersist`.

**One literal socket (M3 F1).** OpenSSH's `%C` hashes local host, host, port, user **and the ProxyJump** — and the
ProxyJump rewrite below (a command-line ProxyCommand) disables the config's ProxyJump, so `%C` differed between a
command with the rewrite (master, probes) and one without (`-O check`, the old `ssh -G` render): lyra's check never
found its master and every ensure leaked one more plain login through jumpvm. Therefore the socket path is expanded
**once** by `ssh -G` and then named literally (`%` escaped as `%%`) by every command that touches the master: create,
`-O check|exit`, probe, stream, connect and interactive shells:
- `user`: the `controlpath` `ssh -G <alias>` prints (no rewrite) — exactly the socket the user's own `ssh <alias>` uses;
- `managed`: `ssh -G -o ControlPath=<mux_dir>/cm-%C {ssh_options} <alias>`.
If `ssh -G` cannot expand it, the pattern is used as before (and the rewrite then applies to every command alike).

Mux decision (config `mux = "auto" | "user" | "managed" | "off" | "session"`, default `auto`):

| `ssh -G` says | `auto` resolves to | Probe/stream options added |
|---|---|---|
| `controlpath` ≠ none **and** `controlpersist` ≠ no (k2, vega, labstation, lynx) | `user` | `-o ControlMaster=no` (use the master at the user's path, never become one) |
| `controlpath` = none (lyra: `ControlMaster false`, no path), or `controlpersist` = no (a `-fN` master would never expire) | `managed` | `-o ControlMaster=no -o ControlPath=<mux_dir>/cm-%C` |
| config `mux = "off"` | `off` | `-o ControlMaster=no -o ControlPath=none` (every probe is a cold login) |
| config `mux = "session"` (opt-in on Linux/macOS; never chosen by `auto` there) | `session` | none: probes and streams are jobs of the one session ssh (§4.8), which runs with `-o ControlMaster=no -o ControlPath=none` |
| `ssh = "local"` | `local` | no ssh (LocalTransport, §4.5) |

**Windows: `session`** (`decide()` / `mux.wants_session()`, whatever `mux` or a ControlPath in `~/.ssh/config` say,
except `mux = "off"`): Windows OpenSSH has no ControlMaster (Win32-OpenSSH#405), and a `-fN` master never
backgrounds there, so `user`/`managed` would hang; `auto`, `user`, `managed` and `session` all become the session
transport (§4.8; `make_transport` picks `SessionTransport` by the same rule). `doctor` (mux check `session (one ssh
login + a temporary sh agent)` plus what that means as a warning) and the host screen (`mux: session` and the same
note) say so; the auth dialog says that `c` logs in for as long as RunFleet runs. `mux = "off"` on Windows keeps the
P1 plan: `connect_argv` is an `ssh … true` auth check, the poller skips `check_master`, `reset_master` does nothing,
`forward_blocker` says why, and doctor / the host screen show `off (Windows)` with the reason. The ProxyJump rewrite
applies in Windows form either way (§4.2).

`<mux_dir>` = `$XDG_RUNTIME_DIR/runfleet` if set, else `~/.cache/runfleet/cm`; mode 0700; if a rendered socket path
would exceed 86 bytes, `/tmp/runfleet-<uid>/` (0700). (OpenSSH first binds `<path>.<16 random chars>`, 17 bytes longer,
and macOS' `sun_path` holds 104 bytes incl. NUL: 103 − 17 = 86. "too long for Unix domain socket" in the log becomes
a MUX error naming the fix.) RunFleet never creates or removes anything at a user ControlPath
except the user-mode master below, and never kills a user master without the user's confirmation.

**Ensure master** (in `prepare()`, and before a probe whose `-O check` failed; at most once per 30 s per host). While it
runs the host is `CONNECTING` and **no probe is sent**: mux hosts never fall back to direct probes (a second cold login
per tick through jumpvm or vega's slow login would be impolite and count as timeouts).
1. `ssh -O check <mux opts> <alias>` (3 s).
1a. **Stale or busy socket.** `ControlMaster=yes` never replaces an existing socket: it logs "ControlSocket … already
   exists, disabling multiplexing", exits 0 and stays up as an ordinary backgrounded login. So when the check failed
   and the socket file exists: "Connection refused" (a master killed with -9 / OOM) → unlink it if it is ours
   (managed) or if the user's own `ssh` would (user mode with ControlMaster auto/autoask; otherwise a MUX error "stale
   ssh control socket"); anything else (a master that does not answer) → MUX error, nothing spawned.
2. Create, spawned per §4.2 hygiene, waiting for the *process exit* (`connect_timeout`, default 45 s):
   - `user`: `ssh -fN -o BatchMode=yes -o ControlMaster=yes -o ConnectTimeout=<ct> -o LogLevel=ERROR -E <log>
     [jump rewrite] <alias>` — nothing else. ControlPath, ControlPersist, keepalives and forwarding come from the user's
     config, so the master their terminals later share behaves like one their own `ssh k2` would start. It does inherit
     RunFleet's environment (agent socket as RunFleet saw it; `DISPLAY` removed, so no X11 forwarding for later mux
     clients — set `mux = "managed"` if that matters).
   - `managed`: the same plus `-o ControlPath=<mux_dir>/cm-%C -o ControlPersist=15m -o ServerAliveInterval=15
     -o ServerAliveCountMax=4`.
   rc 0 → `-O check` again: master up only if it answers. Otherwise (it could not bind the socket) the backgrounded
   `-fN` is found by its command line (`ps -ww -o pid=,args= -U <uid>`; the forked child runs in its own session after
   daemon(3), so the spawn's process group does not reach it) and SIGTERMed — never the master `-O check` names — and
   the result is a MUX error. rc ≠ 0 → classify `<log>` (§9); the hop of a jump rewrite logs into the same file (see
   §4.2), so a hop auth / host-key failure pauses the host as NEEDS_AUTH / HOSTKEY instead of retrying forever.
   A user-initiated retry (manual `r` on a paused host, after `connect`) passes `prepare(force=True)` and bypasses the
   30 s throttle, so "exactly one more attempt" really attempts.
3. Every probe/stream on a mux host is preceded by `-O check` (local, ~5 ms). A master vanishing between check and probe
   turns that one probe into a direct BatchMode login — harmless; the next tick re-ensures.

On exit RunFleet leaves masters running (ControlPersist expires them). `runfleet doctor --reset <host>` runs
`ssh -O exit` for managed masters (a `session` lives only inside the running RunFleet: `--reset` says there is
nothing to reset). User masters are only restarted by `c` after the user confirmed "restart the shared
master?" in HostErrorScreen.

### 4.2 Process hygiene and command lines (exact)

**No background ssh can prompt.** Every non-interactive ssh (probe, stream, ensure master, `ssh -G`, `-O check|exit`) is
spawned by `transport/spawn.py` with `start_new_session=True` (no controlling terminal, so ssh cannot open `/dev/tty`),
`stdin=PIPE` (probe/stream) or `DEVNULL`, and an environment equal to `os.environ` minus `DISPLAY` and `SSH_ASKPASS`
plus `SSH_ASKPASS_REQUIRE=never`. Any prompt (host key, passphrase, password, Duo) therefore fails at once and is
classified (§9 rows 1–2). On Windows (`util/platform.py`) the same spawn uses `CREATE_NO_WINDOW |
CREATE_NEW_PROCESS_GROUP` instead of a new session, stdin is always a pipe (closed at once when there is no input:
some ssh.exe builds hang on a NUL stdin), and killing a probe, stream or notify hook ends its tree with
`taskkill /F /T` and TerminateProcess, and every process it starts joins RunFleet's one Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`platform.kill_on_exit`, ctypes), so not even a crash or a kill of RunFleet
leaves an ssh.exe (or its ProxyCommand hop, which joins its parent's job) behind; a timed-out
probe's taskkill runs in a worker thread, so a dropped network (every probe timing out at once) never stalls the
TUI. Programs the user names (pager, editor, a notify hook argv) are looked up on `PATH` with `PATHEXT` first
(`platform.resolve_program`): CreateProcess only appends `.exe`, and `more` is `more.com`, `code` is `code.cmd`.

**Jump rewrite.** OpenSSH builds the ProxyJump hop as `ssh -W '[%h]:%p' <jump>` and does **not** pass command-line `-o`
options to it (verified on OpenSSH 10.0p2), so BatchMode/ConnectTimeout would not reach `jumpvm`. When `ssh -G` reports
`proxyjump` ≠ none and `proxycommand` = none, non-interactive commands add one option (command-line options take
precedence over the config's ProxyJump):

```
-o ProxyCommand=ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR [-p P] [-l U] [-J H1,…,Hn-1] -W [%h]:%p Hn
```
where `[U@]Hn[:P]` is the last hop of the chain and `H1…Hn-1` the earlier ones (single-hop for lyra and labstation).
Earlier hops of a multi-hop chain do not get the options but are still tty-less, so prompts fail and the local timeout
bounds them. `connect` and interactive commands use the same rewrite **without** BatchMode but with `-E <log>`, so the
hop can prompt in the real terminal yet never writes onto the TUI later. A user-defined `ProxyCommand` is left alone.

**Windows form.** Windows OpenSSH starts a ProxyCommand without a shell (`posix_spawnp` → `CreateProcess`): a command
that does not start with a quoted program is quoted whole after its first `.exe`. So on Windows the rewrite is
`"<absolute ssh.exe>" -T … -W "[%h]:%p" Hn` — the client's absolute path (`shutil.which`) in double quotes, every
argument quoted for the hop's C runtime (`subprocess.list2cmdline`, so an `-E` log path with spaces stays one word),
`%` still escaped as `%%`. `mux.jump_rewrite` is pure and tested on Linux with `IS_WINDOWS` patched.

OpenSSH keeps the **first** value of an option, so every command line has the same order: the structural options
RunFleet depends on (BatchMode, ControlMaster / the literal ControlPath / ControlPersist, the jump rewrite) first, then
the config's `ssh_options`, then RunFleet's tunable defaults — which `ssh_options` may therefore override (a slow VPN's
`ConnectTimeout=30` works; BatchMode, the socket and the rewrite cannot be overridden).

```
BASE   = ssh -T -o BatchMode=yes {mux opts} {jump rewrite} {config ssh_options}
             -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o LogLevel=ERROR --
probe  : BASE <alias> 'exec sh -s'                    # stdin = rendered ProbeScript, then EOF
         remote_shell="login": BASE <alias> 'exec bash -l -c "exec sh -s"'
stream : BASE <alias> 'exec sh -c <quoted StreamSpec.cmd>'   # stdin kept open as the watchdog
check  : ssh -O check|exit {mux opts} {jump rewrite} {config ssh_options} -- <alias>
master : ssh -fN -o BatchMode=yes -o ControlMaster=yes -o ControlPath=<socket> {managed: -o ControlPersist=15m}
             {jump rewrite, batch form + -E <log> for the hop} {config ssh_options}
             {managed: -o ServerAliveInterval=15 -o ServerAliveCountMax=4} -o ConnectTimeout=<ct> -o LogLevel=ERROR
             -E <log> -- <alias>
interactive: ssh -t {imux} [-A if forward_agent] {jump rewrite, interactive form} {config ssh_options} -- <alias> [cmd]
connect: ssh -fN -o ControlMaster=yes -o ControlPath=<socket> {managed: -o ControlPersist=15m} {jump rewrite,
             interactive form} {config ssh_options} {managed: keepalives} -o LogLevel=ERROR -E <log> -- <alias>
```
`{mux opts}`: user and managed → `-o ControlMaster=no -o ControlPath=<socket>` (literal, §4.1); off →
`-o ControlMaster=no -o ControlPath=none`. `{imux}`: user → `-o ControlPath=<socket>` (ControlMaster etc. from their
config); managed → `-o ControlMaster=no -o ControlPath=<socket>`; off → `-o ControlPath=none`.
**Connect** (`c`, `runfleet connect`) first calls `FleetAPI.prepare_connect` → `Transport.before_connect()`: it resolves
the mux plan (an unresolved transport could only guess the user form, so lyra got no socket and a leaked session),
clears a stale socket (a wedged managed master is reset), truncates `<log>`, and returns True when a working master is
already up — then nothing is run ("already connected"). After the connect, `prepare()` reaps a stray `-fN` that logged
"already exists". `<log>` = `~/.local/state/runfleet/ssh-<host>.log`, truncated by each connect/ensure and shown in
HostErrorScreen. The ServerAlive options in BASE only matter for `mux = "off"` (mux clients use the master's
keepalive). The remote command is one word list for the login shell (zsh on labstation/lynx); `exec sh -s` is portable
across bash/zsh/tcsh, and the script itself is POSIX `sh`.

- **Local timeout** per probe = `min(90, max(20, Σ timeouts of non-late sections) + Σ late timeouts + 5)` s. On expiry:
  SIGTERM ssh, SIGKILL after 2 s. **Partial rule:** if ≥ 1 section end marker had arrived, ssh evidently works:
  `ProbeResult.error = None`, unfinished sections become `TIMEOUT`/`INCOMPLETE`, the host goes `DEGRADED`. Only a
  timeout before any end marker is a transport `TIMEOUT` (§4.4).
- Read stdout/stderr in reader tasks, `await proc.wait()` for exit, then give readers 0.5 s to drain and cancel them
  (guards against a descendant holding the pipe).
- **Lanes** (core-enforced): global `asyncio.Semaphore(max_parallel_ssh=8)`. Per host: ≤ 1 poll probe (a tick that
  finds one in flight is skipped, never queued); on-demand probes (detail, action, log peek/fetch, doctor) ≤ 1 on a
  `user` master, ≤ 2 otherwise; log streams ≤ 1 on a `user` master, ≤ 2 otherwise (opening another closes the oldest).
  So RunFleet holds ≤ 3 sessions on a user's shared master, whose sshd `MaxSessions` (default 10) also serves the
  user's own terminals. `Session open refused by peer` / `mux_client_request_session: session request failed` → the
  host's on-demand lane drops to 1 and streams to 0 extra for the session. `hosts import` / the first-run picker hold
  one global permit per alias for the whole probe (`ssh -G`, master creation, detect): 40 aliases behind one bastion
  must not exceed sshd `MaxStartups`. Log streams are long-lived and stay outside the global semaphore (they would
  starve polls); the per-host stream lane bounds them.
- **Frozen loop**: `App.suspend` for `c`/`s`/`p` runs the external command synchronously and `^z` stops the whole
  process; both are recorded (`spawn.loop_frozen()`). A probe whose local timeout expired while the loop was frozen
  gets one fresh timeout budget (it may have finished, or be blocked on a pipe nobody read) instead of being killed and
  counted as a TIMEOUT failure (critic).
- **Killed client**: a local ssh (or `sh`) that dies from a signal (rc < 0: `pkill ssh`, the OOM killer) is a transient
  SSH error ("ssh killed by SIGTERM") with backoff, never §9 row 8.

### 4.3 Framing: the probe script (`probe/framing.py`)

`render(script, *, tmp=None) -> str` emits the prelude, then every non-late section in the given order, then the late
ones. `tmp=False` replaces the `mktemp` line with `RF_T=`: nothing is created on the server, stderr is captured in a
variable (same framing, same outcomes: stderr block before the end marker, partial stdout, `timeout -k 2` rc 124) and
`nvidia.wrap` runs its payload in the foreground. The default `tmp=None` means False for purpose `"attach"` — both the
Attach login probe (§7.16 layer 1) and its nested node script (layer 3) write nothing on any server — and True for
every other purpose.

```sh
# runfleet probe {nonce} v2
export LC_ALL=C
for rf_v in $(env | awk -F= '/^(SQUEUE|SINFO|SACCT|SCONTROL|SSTAT)_|^SLURM_(TIME_FORMAT|CLUSTERS)=/ {print $1}'); do
  unset "$rf_v"; done                      # site env can change rows, not just formats (lyra sets SQUEUE_FORMAT)
export SLURM_TIME_FORMAT=%s                # every Slurm timestamp is epoch seconds (no DST/TZ guessing)
RF=@@RF{nonce}
RF_T=$(mktemp -d "${TMPDIR:-/tmp}/rf.XXXXXX" 2>/dev/null) || RF_T=
export RF_T                                # section commands (sh -c) use it
rf_cleanup() { [ -n "$RF_T" ] && rm -rf "$RF_T"; }
trap rf_cleanup EXIT; trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
if command -v timeout >/dev/null 2>&1; then RF_TO=1; else RF_TO=; fi
rf_ms() { rf_d=$(date +%s%3N 2>/dev/null); case $rf_d in *N|'') echo 0 ;; *) echo "$rf_d" ;; esac; }
rf() {  # rf NAME TIMEOUT REQUIRES CMD
  rf_n=$1 rf_s=$2 rf_q=$3; shift 3
  printf '%s B %s\n' "$RF" "$rf_n"
  for rf_t in $rf_q; do
    command -v "$rf_t" >/dev/null 2>&1 || { printf '\n%s E %s 127 0 missing:%s\n' "$RF" "$rf_n" "$rf_t"; return 0; }
  done
  rf_e=/dev/null; [ -n "$RF_T" ] && rf_e="$RF_T/e"
  rf_0=$(rf_ms)
  if [ -n "$RF_TO" ]; then timeout -k 2 "$rf_s" sh -c "$1" </dev/null 2>"$rf_e"
  else sh -c "$1" </dev/null 2>"$rf_e"; fi
  rf_rc=$?
  rf_1=$(rf_ms)
  printf '\n%s E %s %s %s\n' "$RF" "$rf_n" "$rf_rc" "$(( ${rf_1:-0} - ${rf_0:-0} ))"
  if [ -n "$RF_T" ] && [ -s "$rf_e" ]; then
    printf '%s X %s\n' "$RF" "$rf_n"; head -c 4096 "$rf_e"; printf '\n%s Y %s\n' "$RF" "$rf_n"
  fi
  return 0
}
rf 'slurm.squeue' 20 'squeue' '<cmd>'
rf 'slurm.sacct' 30 'sacct' '<cmd>'
...
rf 'nvidia.gpu' 8 'nvidia-smi' '<cmd>'     # late sections last
exit 0
```

Notes that matter:
- Every section runs with `</dev/null`, otherwise a command that reads stdin (`srun`, `ssh`, `cat`) would eat the rest of
  the script.
- Section commands are passed as one single-quoted argument (`shlex.quote(cmd)`), so they may use pipes, `$USER`,
  `$(id -un)` and `;` freely. Any value coming from data (job ids, paths, partitions) is validated and `shlex.quote`d
  **inside** `cmd` by the collector (`util/shell.py`), before the whole `cmd` is quoted again.
- `\n` is always printed before the end marker. The parser removes exactly one trailing `\n` from the captured stdout
  (that framing `\n`), so "no trailing newline" and "trailing newline" outputs both round-trip exactly: `printf abc`
  → `"abc"`, `echo abc` → `"abc\n"`.
- The parser is line-based: lines starting with `@@RF{nonce} ` are markers; anything outside `B…E` blocks (e.g. a login
  banner in `remote_shell="login"` mode) is ignored and logged at debug level. A nonce per probe makes collisions with
  log content impossible in practice.
- `outcome` derivation: missing end marker → `INCOMPLETE`; `E … 127 0 missing:<t>` → `MISSING`; rc 124/137 → `TIMEOUT`;
  rc 0 + empty stdout + non-empty stderr → `STDERR_ONLY` (treated as failure); rc 0 + empty both → `EMPTY`; rc 0 → `OK`;
  else `FAILED`.
- `date +%s%3N` is GNU; without it `rf_ms` prints `0`, so `ms` is `0` (the prelude above is verbatim; `None` only
  when the end marker carries no number).
- Section stderr sits between the `X` and `Y` markers (first 4 KiB) and loses exactly one trailing `\n`, like stdout.
  The prelude was smoke-tested with `/bin/sh` on the dev box (dash): ok, empty, stderr, missing tool (`127 … missing:<tool>`),
  timeout (`124`), a section running `cat` (does not eat the script), non-zero rc. Revision 2's prelude, the nvidia
  wrapper and the §6.2 script were smoke-tested the same way: Slurm env unset, a TERM-ignoring fake `nvidia-smi`
  abandoned after 6 s without holding the pipe, a TERM-ignoring queue runner and its child fully killed.
- A section that may hang in uninterruptible sleep is `late=True` **and** bounds itself (§5.3.1: run in the
  background, poll for completion, abandon); `timeout -k` cannot kill a D-state process and GNU `timeout` would wait
  for it.

### 4.4 Connection state machine (`core/poller.py` drives, `transport/classify.py` classifies)

```
            prepare ok / probe ok                         probe ok, some sections failing
  NEW ──► CONNECTING ───────────────────► OK ◄──────────────────────────► DEGRADED
               │                           │  ▲                               │
               │ auth/hostkey              │  │ probe ok                      │ transport failure
               ▼                           ▼  │                               ▼
     NEEDS_AUTH / HOSTKEY ◄────────── RETRYING (1–2 consecutive transport failures, backoff)
       (polling paused)                    │
          ▲     │ user presses c → connect │ 3rd consecutive failure, or DNS failure
          │     └──► CONNECTING            ▼
          └──────── auth failure ── UNREACHABLE (backoff continues, capped at 5 min)
  any state ── user toggles host off ──► DISABLED ── toggle on ──► CONNECTING
```

- Backoff (transport failures): `delay = min(base × 2^(n−1), 300 s) × U(0.8, 1.2)`, `base` = the host's tick interval.
  Visibility / multiplier / focus changes re-grid the tick only for a healthy host: while the host is in transport
  backoff (or paused) the backoff deadline stands (re-gridding from the last good tick retried an unreachable host at
  once on every alt-tab). Groups forced by `wake(groups=…)` (e.g. `slurm.history` after a cancel) stay forced when the
  probe fails at the transport level.
  Reset on success.
- A transport failure is: ssh rc 255 with a classified stderr, a spawn error, or a local timeout **before any section
  end marker** (§4.2 partial rule). Section failures and partial timeouts never touch the transport backoff; they back
  off their own cadence group (×2 up to `Cadence.max_backoff`) and set `DEGRADED` with the error per section.
- `NEEDS_AUTH`/`HOSTKEY` stop polling. A manual `r` makes exactly one more BatchMode attempt (`prepare(force=True)`:
  the 30 s master throttle does not swallow it). `c` runs the interactive
  connect (§4.6), then `after_connect()` → `CONNECTING`.
- Two consecutive transport timeouts with a `managed` master → `reset_master()`, then re-ensure. A `session` is reset
  only when `check_master()` fails as well (its ping is a real round trip, unlike `-O check`): a probe stuck remotely
  must not end a healthy session, and a reset session on a Duo host needs `c` again (BatchMode cannot log in). With a `user` master we only show "shared
  ssh master unresponsive — press c to restart it".

### 4.5 Local host
`ssh = "local"` (or `name` ∈ {`local`, `localhost`} with no `ssh` key) → `LocalTransport`: `create_subprocess_exec("sh",
"-s")` with the same rendered script and framing; streams are `sh -c <cmd>`; `interactive_argv(cmd)` = `[$SHELL]` or
`["sh", "-c", cmd]`; `connect_argv()` = None. Useful for running RunFleet on a login node or for the machine it runs on.
POSIX only: on native Windows `prepare()` / `probe()` return, and `stream()` raises, one HostError (MISSING, "local
host monitoring needs Linux/macOS; run RunFleet in WSL") and nothing is spawned.

### 4.6 Interactive connect and shells (UI side, `ui/app.py`)

```python
async def action_connect(self, host: str) -> None:
    argv = self.service.connect_argv(host)      # ['ssh','-fN','-o','ControlMaster=yes','-o','LogLevel=ERROR',
    if argv is None:                            #  '-E','~/.local/state/runfleet/ssh-lyra.log', …,'lyra']
        return
    rc = self.run_external(argv, banner=f"RunFleet → connecting to {host} (Duo/password/host-key prompts appear "
                           "here; Ctrl-C aborts)", fail_log=self.service.store.hosts[host].status.ssh_log)
    if rc is not None:                          # None: SuspendNotSupported, handled by the modal below
        await self.service.after_connect(host, rc)

def run_external(self, argv: list[str], *, banner: str | None = None, fail_log: str | None = None) -> int | None:
    # §7.12: the ONLY place that suspends; tests monkeypatch it. Inside App.suspend(): print banner,
    # rc = subprocess.run(argv).returncode (blocking is fine: the TUI is suspended); if rc != 0 and fail_log: print
    # its last 20 lines (ssh's own messages went to the -E log; private helper in ui/app.py) and wait for Enter
    # ("ssh failed — press Enter to return to RunFleet"). SuspendNotSupported -> modal (below), returns None.
```

- Jump hosts prompt in the same terminal (interactive rewrite keeps prompts, drops later chatter into the log).
- `ssh -fN` returns after authentication; the backgrounded master (and its hop) log only to `<log>`, never to the TUI.
- **Session hosts** (§4.8): `FleetAPI.connect_login(host)` returns the transport's blocking login, and `run_external`
  (`login=`) / `run_external_sequence` (`logins=`, log in to all) call it inside the same suspend instead of running
  the argv, which then only serves the manual-command modal. The login starts the session ssh WITHOUT BatchMode,
  the terminal inherited, so its prompts appear there as with `ssh -fN`; it returns 0 once the agent says ready and
  that ssh keeps running after the resume. `runfleet connect` stays an auth check for session hosts and says to press
  `c` in the TUI instead.
- `SuspendNotSupported` → modal (`ManualCommandScreen`) with the exact command to run in another terminal and a
  *Done* button that calls `FleetAPI.after_connect(host, 0)` (re-prepare + poll now). `check_master` is a Transport
  method, not on `FleetAPI`, so the UI does not poll it (M2 integration decision).
- Shell (`s`): host → `ssh -t <alias>`; workstation task → `ssh -t <alias> 'cd <cwd> && exec "$SHELL" -l'`; running
  Slurm job → `exec sh -c '<script>'` that cds to WorkDir and runs `srun --jobid=<raw> --overlap --pty -N1 -n1 -w <node>
  "$SHELL" -l` (the job's cgroup and GPUs; `--overlap` only for Slurm ≥ 20.11; a failed srun waits for Enter so its
  error stays readable); a multi-node job first asks for the node (batch host first). Other Slurm jobs → a login-node
  shell `cd`'d to WorkDir. srun (not ssh-to-node) because lyra's sshd refuses compute-node logins and pam_slurm_adopt
  lands in the *newest* job on a shared node (Attach field probes, §7.16).
- **Ctrl-C while suspended**: `asyncio.run` (Textual's `App.run`) installs a SIGINT handler that only cancels the
  main task on the first Ctrl-C, which would end RunFleet after the resume. The suspend therefore runs under
  `platform.ctrl_c_raises()` (Python's default handler: KeyboardInterrupt), and a command under
  `platform.ctrl_c_to_child()` (a Python no-op handler, so the command alone decides: `less` stops following, ssh
  ends); a session login is aborted by the KeyboardInterrupt (rc 130).
- After every suspend the app forces a full repaint and refreshes that host.

### 4.7 Logs (`core/logs.py`)
- **Master first**: before a stream opens, a `user`/`managed` master that is gone (ControlPersist expired, the user
  closed theirs) is revived exactly like for on-demand probes (`check_master` → `prepare`); a failed prepare is shown in
  the pane and no stream starts (no fallback BatchMode login through the jump host, no instant Duo failure). A
  session host's transport does the same itself (a dead session is restarted, throttled, before the stream, §4.8).
- **Follow** (JobDetail log pane and LogScreen only): `StreamSpec("log.tail", cmd="tail -c {n} -F -- {path} & p=$!;
  cat >/dev/null; kill $p 2>/dev/null", meta={"path": path, "bytes": n})`, `n = 65536`. Initial content and follow come
  from one command (no gap, no duplicate); the stdin watchdog kills the remote `tail` when we close stdin (verified on
  two workstations).
- **Line assembly**: `StreamHandle.chunks()` yields raw blocks (≤ 64 KiB, never line-split, so a multi-MB CR-only tqdm
  line cannot overflow a reader limit). An incremental UTF-8 decoder (`errors="replace"`) feeds a splitter: `\n` ends a
  line; within a line `\r` restarts it (terminal semantics). The text after the last `\r` of the unterminated tail is
  `LogUpdate.live`, rendered in a one-line footer under the log so progress shows before any newline. A line longer
  than 16 KiB is cut with `…[+N bytes]` and the rest dropped until the next `\n`. Updates are batched ≤ 10/s.
- **Missing file** (pending job): GNU `tail -F` reports `cannot open … No such file` on stderr and keeps retrying; the
  viewer shows "waiting for <path> to appear".
- **Peek** (Jobs preview): on-demand probe `log.peek` = `stat -c '%s|%Y' -- {path}; tail -c {nbytes} -- {path}`, sent
  after ≥ 1 s of cursor dwell, cached per (host, path), re-fetched only when ≥ 15 s old **and** the job's logstat
  size/mtime changed. Never a stream.
- **Pager** (`p`): `ssh -t <alias> less -R +G -- <path>` if `less` ∈ caps.tools; else `fetch_log` (`log.fetch` =
  `tail -c 4194304 -- {path}`) into a 0600 temp file and the local `general.pager` / `$PAGER` / `less -R +G`. In v0.1
  the pager is where log search happens.
- LogView keeps ≤ 20 000 lines (RichLog `max_lines`). Older-chunk paging, wrap toggle and save are **v0.2**; `/`
  searches the loaded buffer, and §4.7.1 searches several jobs' files on the host.

#### 4.7.1 Cross-job log search (`core/logsearch.py`, `ui/screens/log_search.py`)
On demand only — palette › **Search logs…** or `^f` in the Jobs view; never from a timer, never by the poller, and
read-only (allowed with `--readonly`). The prompt takes a pattern, **literal by default** (`tab` reaches two options:
regular expression = POSIX ERE via `grep -E`; match case — otherwise ASCII letters match either case, grep `-i` in the C
locale the probe prelude exports).
- **Scope** (`logsearch.plan`, pure): per host of the current host scope (header `ALL` or `0`/`1-9`), the user's own
  jobs (`is_mine`) that run now (`StateGroup.ACTIVE`) or ended within `general.recent_window` (end, else observed
  time), most recent first; their stdout and stderr paths, deduplicated (merged stderr = the stdout file; a file several
  jobs write is searched once, listed under the newest job), validated (`util.shell.path`: absolute, one line), at
  most **20 files per host** (`left_out` counts the rest). Hosts that are disabled, unreachable or need auth / a host
  key are skipped and listed as such; a host with no file gets no probe.
- **One probe per host** (purpose `log`, the on-demand lane, `_on_demand` like peek/fetch), one section `log.search`
  (timeout 30 s, the local deadline 35 s by `framing.local_timeout`, `max_stdout` 256 000), the hosts searched in
  parallel. The pattern goes into grep as `.{0,60}(PATTERN).{0,60}` (a literal one with every ERE special character
  in a bracket expression, `^` escaped), so `-o` prints an excerpt of ≤ 60 bytes either side; a regex must not use
  back-references and its parentheses must pair (else the group could be closed early), and no `{n,m}` count may exceed
  255 nor may the counted repetitions expand past 1000 positions (`logsearch.EXPANDED_MAX`): grep expands them when it
  compiles, so a nested `((a{1,255}){1,255}){1,255}` took over 3 GB on the login node before reading a byte (checked
  locally, before anything runs). Exact command (paths and the
  pattern `shlex`-quoted, nothing globbed, no recursion, no file is written):
  ```sh
  p='.{0,60}(PATTERN).{0,60}'; left=200; budget=1048576; i=0
  printf 'x\n' | grep -E -e "$p" >/dev/null 2>&1
  [ $? -lt 2 ] || { printf '@@bad %s\n' "$(printf 'x\n' | grep -E -e "$p" 2>&1 >/dev/null | head -n 1 | cut -b 1-200)"; exit 0; }
  for f in '/path/one.out' '/path/two.err'; do
    i=$((i + 1))
    if [ "$left" -le 0 ]; then echo "@@skip $i matches"; continue; fi
    if [ "$budget" -le 0 ]; then echo "@@skip $i budget"; continue; fi
    if [ ! -e "$f" ]; then echo "@@err $i missing"; continue; fi
    if [ ! -f "$f" ]; then echo "@@err $i notfile"; continue; fi
    if [ ! -r "$f" ]; then echo "@@err $i denied"; continue; fi
    s=$(stat -L -c %s -- "$f" 2>/dev/null)
    case $s in ''|*[!0-9]*) echo "@@err $i stat"; continue;; esac
    n=$s; [ "$n" -gt 262144 ] && n=262144; [ "$n" -gt "$budget" ] && n=$budget
    b=$((s - n)); budget=$((budget - n))
    echo "@@file $i $s $b $n"
    [ "$n" -gt 0 ] || continue
    r=$(tail -c +$((b + 1)) -- "$f" | head -c "$n" | grep -a -n -b -o -i -E -m "$left" -e "$p" | head -n "$left" | cut -b 1-600)
    [ -z "$r" ] || { printf '%s\n' "$r"; left=$((left - $(printf '%s\n' "$r" | wc -l))); }
  done
  echo "@@end $left $budget"
  ```
  Bounds: the **last 256 KiB** of each file, **1 MiB read** per probe in all (a file the budget reaches is read short,
  the rest are skipped and say so), **≤ 200 excerpts overall** (split evenly between the hosts searched:
  `max_matches = 200 // hosts`, `-m` and `head -n` per file), ≤ 600 bytes per output line, `/dev/null`, FIFOs and
  directories never opened (`-f`), the 30 s remote timeout (the 35 s local deadline also bounds a read stuck on a
  network filesystem). Cost: one ssh round trip and no Slurm RPC per host per search; the remote work is ≤ 1 MiB of
  sequential reads from the end of ≤ 20 files and one grep over them.
- **Output → `LogSearchResult`** (`logsearch.parse`): per file a `FileOutcome` (searched with size / window start / bytes
  read / hits; missing, denied, not a file, unreadable; skipped by the budget or the match cap; not reached when the
  probe timed out or died); per hit a `LogHit`: window start `b` (absolute), grep's line number `-n` inside the window,
  absolute offset `b + -b`, the excerpt, its **needle** (the `\r` segment holding the first match, cut characters
  dropped) and the file size. `@@bad` (grep refused the regex) is the host's error, as are a timeout and a transport
  failure; partial output is kept.
- **Results** (`LogSearchScreen`): the summary says exactly what was read — `searched N files on <hosts> · M KiB ·
  K matches`, then every cap that applied (`only the last 256 KiB of N files`, `1 MiB read limit: N cut short, N not
  searched`, `20-file limit: N older files left out`, `match limit reached on k2 (50)`, files not reached) or `nothing
  capped`; per-host errors and skipped hosts; unreadable files by kind. Rows are grouped per file (host, job id, name,
  stdout / stderr / `stdout+stderr`, the path cut from the left, `N matches in the last 256K of 3.1M`), each hit one
  row: the excerpt without ANSI codes, the match(es) highlighted, fitted to the width in cells with the first match a
  third in (`…` where text is left out, or where the context window cut the line; wide characters never split).
- **Open a hit** (`enter`): `LogScreen` is pushed **above** the results (`esc` comes back to them) and streams from
  the hit's window start, `tail -c +<b+1> -F` (`FleetAPI.stream_log(start=b)`), so grep's line number is the index of
  the line in the viewer however much the job wrote since: the anchor is the absolute offset plus the line number,
  confirmed by the needle — never a tail-relative line number. `LogView` maps the raw line to its first wrapped
  display line (its RichLog records where each write's display lines start), scrolls there, pauses following and
  notes `search hit "…" · line N` (a literal pattern also primes `n`/`N`). When that line lacks the needle (the file
  was rewritten, a `\r` update overwrote the match) the nearest line with it wins, with a note saying so; when the
  file grew by more than 8 MiB since the search, the usual last 64 KiB are streamed and the last line with the needle
  is used; the line never arriving within 5 s falls back the same way.

### 4.8 Session transport (`transport/session.py`, `transport/agent.py`)

**Why.** Windows OpenSSH has no ControlMaster, so without one every probe would be a full login (P1): fine for keys,
impossible for Duo or password hosts. The session transport keeps ONE `ssh` per host open and multiplexes every
probe and log stream over it through a small POSIX `sh` agent that RunFleet sends on that login's stdin. It still
uses only the system ssh (D2), and it installs nothing (§1.2): the agent lives in the remote shell's memory, keeps a
few state files in a private `mktemp -d` directory and removes it, and ends every job it started when its stdin
closes, when it exits, or on HUP/INT/TERM/PIPE. Who gets it: Windows unless `mux = "off"`; Linux/macOS only with
`mux = "session"` (opt-in, never the default: CI and developers exercise it there).

**Command line.** `SessionTransport` subclasses `OpenSSHTransport` and reuses `resolve` (`ssh -G`), the jump rewrite,
`ssh_options`, the `-E` log and the classification (§9):
```
session: ssh -T [-o BatchMode=yes] -o ControlMaster=no -o ControlPath=none {jump rewrite, + -E <log> for the hop}
             {config ssh_options} -o ConnectTimeout=<connect_timeout | 30 for the login> -o ServerAliveInterval=15
             -o ServerAliveCountMax=4 -o LogLevel=ERROR -E <log> -- <alias> <bootstrap>
bootstrap (one word for the login shell):
         exec sh -c 'IFS= read -r n; eval "$(dd bs=1 count=$n 2>/dev/null)"'
         remote_shell="login": exec bash -l -c '<the same>'
stdin  : "<n>\n" + the agent's n bytes, then one request per line
```
`read` and `dd bs=1` never read ahead, so the requests after the agent stay on stdin for its own `read` loop.

**Protocol** (`agent.py` docstring is the reference; ids `[A-Za-z0-9_]+`, one ASCII line per message):

| client → agent | meaning |
|---|---|
| `P <token>` | ping → `P <token>` |
| `Q <id> <payload>` | probe: the payload (a rendered ProbeScript, §4.3) is fed to `sh -s` |
| `S <id> <payload>` | stream: the payload runs as `sh -c`, its stdin open until `W` (§4.7 watchdog form) |
| `W <id>` | close the stream's stdin |
| `K <id>` | SIGTERM the job's command (its process group with `setsid`, and its descendants via `ps`), SIGKILL 2 s later; a job still running 1 s after that (output held open by a child in D state or one that escaped) stops counting toward the 8 |

| agent → client | meaning |
|---|---|
| `R <pid> <version> <setsid\|-> <pipe_buf>` | ready, sent after an empty line so login-shell output without a newline (a title escape) cannot glue onto it (lines before it — `.bashrc` chatter — are ignored) |
| `L <id> <text>` / `C <id> <text>` | a probe stdout line (newline implied) / a piece of a longer line |
| `D <id> <hex>` | stream stdout bytes, one line per `read(2)`: `\r` progress arrives at once |
| `E <id> <text>` / `F <id> <text>` | stderr line / piece, after the job's stdout ended (last 8 KiB) |
| `Z <id> <rc>` | the job ended (`125` + `F … busy`: refused, 8 jobs already run) |
| `X <text>` | the agent cannot start (no temp dir) |

A payload is the bytes as a `printf` format (printable ASCII as is, `%` → `%%`, everything else and a leading `-` as
`\ooo`): no `base64` needed remotely. **Every agent write is one `write(2)` below PIPE_BUF** (`getconf PIPE_BUF /`,
512 if unknown; lines ≤ PIPE_BUF − 128 bytes and never more than 4 KiB), so lines of concurrent jobs never
interleave on the shared stdout. Probe output goes through `awk` (`fflush` per line, each line printed with ONE `%s`
conversion, long lines cut into `C` pieces), so framing (§4.3) sees exactly the bytes `sh -s` printed: framed output
always ends in `\n`, the one thing line mode cannot carry (NUL bytes are not faithful either; probes print text).
mawk (Debian/Ubuntu's awk) reads a pipe in blocks until EOF, so the agent runs it as `awk -W interactive`: lines
leave as the job prints them, and a probe that times out keeps its finished sections (its unbuffered writes are why
each line is one conversion). Streams go through `dd bs=N count=1 | od -A n -t x1`.

**Jobs.** At most 8 at once (`agent.MAX_JOBS`): the client holds an `asyncio.Semaphore(8)` per host, the agent
refuses a 9th (rc 125). After a `K` the client keeps the job's slot until its `Z` or 3.5 s, when the agent has
stopped counting it too, so both sides count the same jobs. Jobs start detached from the agent's shell
(`( rfa_run … & )`): it never `wait`s, and a plain `&` would grow its job table with every probe. Each job runs `setsid sh …` when util-linux `setsid` exists (its own process group), writes
the command's pid to `<dir>/<id>.pg`, its stderr to `<id>.err`, its rc to `<id>.rc`; `K` and the exit cleanup kill
that group plus its descendants found with `ps -A -o pid= -o ppid=` (GNU `timeout` puts sections in groups of their
own), never the job's own plumbing, so the encoder flushes the partial output and `Z` still comes. A stream's stdin
is a pipe from a holder loop (no `mkfifo`) that ends on `W`, when the command ended, when the temp dir is gone, or
when the agent's pid is gone (even after a SIGKILL of the agent).

**Remote requirements.** POSIX `sh` (dash, bash, BusyBox ash, ksh), `dd`, `awk` with `fflush`, `od`, `tr`, `tail`,
`sleep`, `cat`, `rm`, `mktemp` or a writable `$TMPDIR` / `/tmp`. Optional: `setsid` (Linux; without it — macOS,
some BusyBox systems — kills go through the `ps` walk) and `ps` with `-o pid= -o ppid=` (without either, a kill
reaches only the job's shell and the job's processes end at their own remote timeouts). Tested on Linux CI under
dash, bash and BusyBox ash (`tests/unit/test_agent.py`); macOS and other BSD remotes are expected to work (no
`setsid`: the `ps` fallback, also tested) but are untested.

**Lifecycle** (`SessionTransport`):
- `prepare(force)`: resolve; a live session must answer a ping within 3 s, or keep delivering output meanwhile (a
  pong queues behind a big log fetch on a slow link) (alive but silent → TIMEOUT error, the session is kept:
  ServerAlive ends a dead link, and two timeouts in a row with a failing ping reset it, §4.4); else at most one new
  BatchMode session per 30 s unless `force`; ready = `R` within `connect_timeout` + 5 s. Failures are classified from
  `<log>` + the session's stderr (AUTH, HOSTKEY, REFUSED …; an agent that never started → §9 row 8).
- `check_master()`: process alive and a ping answered within 3 s (or output still arriving); the poller treats
  `session` like `managed` (check before a tick, `prepare` when it fails), except that two timeouts in a row reset it
  only when `check_master()` fails too: a probe stuck remotely (a hung NFS mount) must not end a healthy Duo login.
- `probe()`: starts a session if none is up (throttled), runs one `Q` job; its local timeout (§4.2, with the
  frozen-loop extra budget) sends `K` for that job only, waits up to 3.5 s for its `Z` and keeps whatever output
  arrived (partial rule). A session
  that ends under a job finishes it as lost: `ProbeResult.error` from the session's death (ssh's log line classified,
  else MUX "ssh session ended"), unless an end marker had arrived.
- `stream()`: one `S` job; `close()` sends `W`, then `K` after 2 s (and waits up to 3.5 s more).
- `login()` (`InteractiveLogin`, defined in `transport/session.py`): blocking, inside `App.suspend`; the session
  WITHOUT BatchMode, the terminal inherited (POSIX: same process group so `/dev/tty` prompts work, through `sh -c 'trap "" INT QUIT; exec "$@"'` so a
  later Ctrl-C in a pager or shell does not reach it; Windows: `CREATE_NEW_PROCESS_GROUP`, the console inherited:
  Win32 `read_passphrase` reads the console even with piped stdin). Waits for `R` on a `threading.Event` (Ctrl-C
  → 130), so the blocked event loop does not matter; the session then replaces any BatchMode attempt in flight.
  Ctrl-C works because `run_external` puts Python's default SIGINT handler back for the suspend (§4.6): asyncio's
  runner would otherwise only cancel the main task on the first Ctrl-C (ending RunFleet after the resume). An
  aborted login's ssh gets SIGTERM first (it restores the terminal's echo), SIGKILL 1 s later.
- A BatchMode start that is cancelled (the host was removed or disabled) or whose transport was closed meanwhile is
  closed, never installed.
- `reset_master()` / `close()`: close the session's stdin (the agent ends everything and removes its dir), kill
  after 2 s. A crash of RunFleet closes the pipe too (POSIX) or the Job Object kills ssh.exe (Windows, §4.2).
- I/O: three daemon threads per session on a plain `subprocess.Popen` (stdout decoder, stderr tail, stdin writer),
  replies handed to the loop with `call_soon_threadsafe`: the same code on the Proactor loop and under a suspended
  TUI.
- Not over the session: shells (`s`) and the remote pager (`p`) are their own `ssh -t` logins (a Duo push each), and
  port forwards (`forward_blocker` says so; P3 adds `-D` to the session).

---

## 5. Collectors

General rules for every collector:
- **Parse stdout whenever it is present**, even when the section outcome is `FAILED` (multi-id commands such as
  `sstat -j a,b`, the `scontrol` loop and `stat a b` return rc 1 with useful partial output). The outcome only decides
  error badges and backoff.
- A section whose outcome is `STDERR_ONLY`, `TIMEOUT`, `INCOMPLETE` or `NOT_RUN` **never** clears previous data: the
  collector returns `None` for that part of `CollectorOutput`, so the store keeps the last good value (marked stale).
- Parsers are pure functions `parse_x(text: str, ctx) -> ParseResult[T]` where `ParseResult(items: list[T],
  bad_lines: list[tuple[int, str, str]])`. A row that fails validation is skipped and counted; if > 50 % of rows fail the
  section is reported as `ErrorKind.PARSE` with the first bad line in `detail`. With `--debug` **every** probe is dumped
  in RunFleet wire format (`runfleet/debugdump.py`): `~/.local/state/runfleet/debug/<host>/<ts>-<pid>-<seq>-<purpose>`
  `.in` (rendered script), `.out` (framed stdout), `.err`, `.meta` (rc, timing, outcomes, transport error, argv);
  files capped at 4 MiB, the newest 200 probes per host kept. They replay through `framing.parse` and become fixtures;
  every probe also logs a DEBUG summary line, and a TUI crash is logged to `runfleet.log` besides its crash report.
- Placeholder normalization (`util/slurmfmt.py: nullish()`): `""`, `N/A`, `n/a`, `(null)`, `None`, `None assigned`,
  `Unknown`, `[N/A]`, `[Not Supported]`, `[Unknown Error]` → `None`.
- `ParseResult` also carries `bad_ids: set[str]`: ids of rows that failed validation but whose id field was readable.
  Collectors **never** treat a bad id as vanished/ended: they keep the previous object (stale) and count the error.

### 5.1 Detection and capabilities (`backends/detect.py`)
The **detect probe** (one round trip, purpose `detect`) holds the generic sections plus every registered collector's
`capability_sections()` (each guarded by `requires`, so absent tools come back `MISSING` at no cost):

| Section | Command | Parsed into |
|---|---|---|
| `detect.id` | `printf 'user=%s\nuid=%s\nhost=%s\ntz=%s\nepoch=%s\n' "$(id -un)" "$(id -u)" "$(hostname)" "$(date +%z)" "$(date +%s)"` | remote_user, uid, login_host, tz_offset, clock_skew |
| `detect.tools` | `for c in squeue sacct sinfo scontrol sstat scancel sacctmgr nvidia-smi timeout less stat ps pgrep srun ssh sprio sshare; do command -v "$c" >/dev/null 2>&1 && echo "$c"; done; true` | tools |
| `slurm.version` (req `sinfo`) | `sinfo --version` | `slurm 25.05.8` → "25.05.8" |
| `slurm.config` (req `scontrol`) | `scontrol show config \| grep -E '^(ClusterName\|AccountingStorageType\|JobAcctGatherType\|JobAcctGatherFrequency\|PreemptMode\|PrologFlags\|LaunchParameters\|MinJobAge\|PrivateData\|PriorityType\|PriorityWeight[A-Za-z]*\|PriorityFlags\|PriorityDecayHalfLife\|PriorityMaxAge\|PriorityUsageResetPeriod) '` | cluster_name, slurm_config (incl. the priority settings of §5.2.10), acct_gather_freq_s |
| `slurm.assoc` (req `sacctmgr`) | `sacctmgr -n -P show assoc where user="$(id -un)" format=Cluster,Account,Partition,QOS` | associations |
| `slurm.parts` (req `scontrol`) | `commands.PARTS_CMD`: `groups=` (comma-separated names from the group database via `getent`, not the session's groups), `parent=cluster\|account\|parent` lines walking up from the user's accounts (account-level `sacctmgr show assoc` rows, ≤ 6 levels, `root` left out), `parents=complete` when every sacctmgr call worked and the chain ended, then `scontrol show partition -o` (a value runs to the next ` Key=`, so names with spaces stay whole) | gpu_partitions: partitions with `gres/gpu` in TRES, not `Hidden`, `RootOnly`, `INACTIVE` or `DRAIN`, whose AllowGroups holds one of your groups and where, for some account, the association of this cluster for that partition (else the one for all partitions, as Slurm picks) passes AllowAccounts/DenyAccounts with the account or a parent (the lists are hierarchical; a DenyAccounts list needs `parents=complete`) and has a QOS passing AllowQos/DenyQos (an empty QOS list passes only unrestricted partitions). No association data at all: only partitions open to every account and QOS; associations on other clusters only: none |
| `slurm.sacctfmt` (req `sacct`) | `sacct --helpformat` | sacct_fields (enables `StdOut,StdErr` when present) |
| `slurm.gputres` (req `sacct`) | `j=$(sacct -u "$(id -un)" -S now-14days -E now -X -n -P --format=JobIDRaw,AllocTRES,State \| awk -F'\|' '$2 ~ /gres\/gpu/ && $3 !~ /^(RUNNING\|PENDING)/ {j=$1} END {print j}'); [ -n "$j" ] && sacct -j "$j" -n -P --format=JobID,TRESUsageInMax%400; true` | gpu_tres seed |
| `nvidia.version` (req `nvidia-smi`) | `nvidia-smi --version` | driver_version: key `KMD version`, fallback `DRIVER version` |

**gpu_tres** (does the cluster record `gres/gpuutil`?): seeded by `slurm.gputres` — any step row with `gres/gpuutil=`
→ True (k2, vega), rows without it → False (lyra), no recent ended GPU job → None. At runtime only sstat rows of jobs
with `alloc.gpus > 0` count: key present → True; False only after **two distinct GPU jobs** lacked it. CPU-only jobs
(vega `dev` partition) never change it. sstat CPU/RSS fields are used regardless of gpu_tres. Changes go out as
`caps_patch={"gpu_tres": …}`.

**Priority settings** (§5.2.10): the same `slurm.config` grep brings `PriorityType`, every `PriorityWeight*`,
`PriorityFlags`, `PriorityDecayHalfLife`, `PriorityMaxAge` and `PriorityUsageResetPeriod` into `caps.slurm_config`
(no extra RPC, cached with the other capabilities); `DETECT_REV` 2 re-detects capabilities cached without them.

Profile selection (`backends/detect.py: choose_profile(caps, host_cfg) -> tuple[str, ...]`):
1. `backend = "slurm"` in config, or `auto` and `squeue` ∧ `sinfo` ∈ tools → `("slurm", "procfs")` + `"nvidia"` if
   `nvidia-smi` ∈ tools (a login node with GPUs) — but never on k2/vega/lyra today (no nvidia-smi on login nodes).
2. `backend = "workstation"` or `auto` and `nvidia-smi` ∈ tools → `("process", "nvidia", "procfs")`.
3. Otherwise → `("process", "procfs")` (CPU-only Linux box). Non-Linux (`/proc` missing) → host `DEGRADED` "unsupported OS".

Capabilities + profile are cached in SQLite for 24 h; on launch a cached profile is used immediately and the detect probe
re-runs in the background (also every 6 h and after `connect`).

Core (WP1: `prepare`, `collect_once`, `probe_alias`, re-detect) uses exactly these WP2 functions — FROZEN signatures
(each file adds its own imports):

```python
# src/runfleet/backends/registry.py
def collectors() -> dict[str, type[Collector]]: ...
    # BUILTIN plus entry-point group "runfleet.collectors"; a broken plugin is logged and skipped

# src/runfleet/backends/detect.py
def detect_sections(host: HostConfig, collectors: Mapping[str, type[Collector]]) -> list[Section]: ...
    # detect.id + detect.tools + every collector's capability_sections(host); core renders them as one probe
    # (purpose="detect")
def build_capabilities(results: Mapping[str, SectionResult], host: HostConfig, now: datetime,
                       collectors: Mapping[str, type[Collector]]) -> Capabilities: ...
    # pure: parses detect.id (tz_offset from %z; clock_skew = remote epoch - now) and detect.tools; merges each
    # collector's ingest_capabilities(results whose owner is that collector) in registry order; sets profile via
    # choose_profile. Raises errors.RunFleetError if detect.id is not OK or unparseable (core reports it as a
    # REMOTE/PARSE host error and retries per §4.4)
def choose_profile(caps: Capabilities, host: HostConfig) -> tuple[str, ...]: ...
```

### 5.2 Slurm (`backends/slurm/`)

#### 5.2.1 Sections and cadence

| Cadence group (mode) | Section | Interval (visible) / floor / hidden× / focused× | When included |
|---|---|---|---|
| `slurm.jobs` (interval) | `slurm.squeue` | 15 s / 10 s / ×3 / ×1 | always — defines the 15 s tick |
| `procfs.host` (piggyback) | `procfs.host` | 15 s / 10 s / ×3 / ×1 | rides the tick (login-node load) |
| `slurm.jobinfo` (event) | `slurm.scontrol` | — | ids never fetched, PD→R transitions; ≤ 8 ids per probe |
| `slurm.usage` (interval) | `slurm.sstat` | 60 s / 30 s / ×3 / ×0.5 (≥ `JobAcctGatherFrequency`) | ≥ 1 own RUNNING job; ≤ 20 ids, most recently started first |
| `slurm.history` (interval) | `slurm.sacct` | 120 s / 60 s / ×3 / ×1 | always (incremental window) |
| `slurm.vanished` (event) | `slurm.sacct_ids` | — | ids that left squeue and are not yet terminal in history |
| `slurm.logstat` (interval) | `slurm.logstat` | 120 s / 60 s / ×3 / ×0.5 | RUNNING jobs with a known stdout path (≤ 30) |
| `slurm.nodes` (interval) | `slurm.sinfo` | 120 s / 60 s / ×5 / ×1 | partitions known: config `partitions`, else partitions of the user's jobs (live + history) ∪ non-empty association partitions ∪ `caps.gpu_partitions` (GPU partitions you may use but have not used lately, e.g. a lab's paid partition) |
| `slurm.occupants` (interval, **focused only**) | `slurm.sinfo` + `slurm.occupants` | 30 s / 30 s / — / ×1 | the host's detail screen (or a job detail on it) is open and partitions are known; not after the squeue `-o` fallback (§5.2.9) |
| `slurm.names` (event) | `slurm.names` | — | users in the last `slurm.occupants` reading never looked up (≤ 200 per probe) |
| — (on demand, any host kind) | `log.search` | never on a timer: palette › Search logs… / `^f`, one probe per host | the user asked; ≤ 20 own recent log files (§4.7.1) |
| `slurm.priority` (interval) | `slurm.sprio` + `slurm.sshare` | 600 s / 600 s / ×6 / ×1 (failure backoff up to 1 h) | ≥ 1 own job PENDING/REQUEUED and not held, `PriorityType=priority/multifactor` with some `PriorityWeight*` > 0, sprio or sshare present; `slurm.sshare` only when `PriorityWeightFairShare` > 0 (§5.2.10) |

All due groups ride **one** probe per tick, and groups are only evaluated at ticks (§5.4), so a visible Slurm host
costs 4 round trips/min (1.3 hidden); an open host detail adds nothing but two more sections every 30 s, and a pending
job of yours two sections every 10 min (hidden: every hour). Job detail is a separate on-demand probe (`detail_sections`: `slurm.scontrol` for
that id if older than 30 s, `slurm.steps`, `slurm.sstat` if running, `slurm.logstat`).

#### 5.2.2 Exact commands (`backends/slurm/commands.py`)

```python
SEP = "\x1f"   # ASCII unit separator: -O field suffix and sacct --delimiter, so a "|" in a job name cannot split a row.
               # Verified at M3 on all three clusters; if a cluster rejects it, that host falls back to "|" (logged).

SQUEUE_FIELDS = ["JobID", "ArrayJobID", "ArrayTaskID", "Name", "UserName", "State", "StateCompact", "Reason",
    "TimeUsed", "TimeLimit", "TimeLeft", "NumNodes", "NumCPUs", "MinMemory", "Partition", "QOS", "Account",
    "SubmitTime", "StartTime", "EndTime", "NodeList", "BatchHost", "tres-alloc", "tres-per-node", "tres-per-job",
    "RestartCnt", "STDOUT", "STDERR", "WorkDir", "Command"]          # Command MUST stay last

def squeue_cmd(users: list[str] | None, extra: list[str], sep: str = SEP) -> str:
    who = "--me" if not users else "-u " + shlex.quote(",".join(users))
    fmt = ",".join(f"{f}:{sep}" for f in SQUEUE_FIELDS)
    return f"squeue {who} -h -O {shlex.quote(fmt)} {' '.join(map(shlex.quote, extra))}"

# Legacy fallback (only if -O fails with rc≠0; -O works on all three clusters today):
SQUEUE_LEGACY = "%A|%F|%K|%j|%u|%T|%t|%r|%M|%l|%L|%D|%C|%m|%P|%q|%a|%V|%S|%e|%N|%B|%b|%Z|%o"

SACCT_FIELDS = ["JobID", "JobIDRaw", "JobName", "User", "Account", "QOS", "Partition", "State", "ExitCode",
    "DerivedExitCode", "Reason", "Submit", "Start", "End", "Elapsed", "Timelimit", "AllocCPUS", "ReqMem",
    "ReqTRES", "AllocTRES", "NNodes", "NodeList"]  # + ["StdOut","StdErr"] if in caps.sacct_fields, + ["WorkDir"] last

def sacct_window_cmd(since_s: int, fields: list[str], sep: str = SEP) -> str:
    # relative start avoids timezone parsing; -E now is REQUIRED for any state filtering (field probes)
    return (f'sacct -u "$(id -un)" -S now-{since_s}seconds -E now -X -P --delimiter={shlex.quote(sep)} -n '
            f'--format={",".join(fields)}')

def sacct_ids_cmd(ids_raw: list[str], fields: list[str], sep: str = SEP) -> str:
    return f"sacct -j {shlex.quote(','.join(ids_raw))} -X -P --delimiter={shlex.quote(sep)} -n --format={','.join(fields)}"

STEP_FIELDS = "JobID,JobName,State,ExitCode,DerivedExitCode,Reason,Elapsed,Start,End,NodeList,NTasks,MaxRSS,AveCPU," \
              "TRESUsageInMax%400,TRESUsageInTot%400"
STEP_EXTRA = ("TotalCPU",)        # §5.2.8: appended when caps.sacct_fields lists them — detail query only
def step_fields(caps_fields: frozenset[str]) -> str: ...      # STEP_FIELDS + the known STEP_EXTRA
def sacct_steps_cmd(job_id_raw: str, sep: str = SEP, fields: str = STEP_FIELDS) -> str:
    return f"sacct -j {shlex.quote(job_id_raw)} -P --delimiter={shlex.quote(sep)} -n --format={fields}"

def scontrol_cmd(ids_raw: list[str]) -> str:          # one RPC per id; ids validated by JOBID_RE
    return "for j in " + " ".join(map(shlex.quote, ids_raw)) + \
           '; do echo "@@job $j"; scontrol show job -d -o "$j" 2>&1; done; true'

SSTAT_FMT = "JobID,NTasks,AveCPU,AveRSS,MaxRSS,TRESUsageInAve%400,TRESUsageInMax%400,TRESUsageInTot%400"
def sstat_cmd(ids_raw: list[str]) -> str:              # fields never contain "|"
    return f"sstat -a -j {shlex.quote(','.join(ids_raw))} -P -n --format={SSTAT_FMT}"

def logstat_cmd(paths: list[str]) -> str:              # parse with rsplit("|", 2): paths may contain "|"
    return "stat -c '%n|%s|%Y' -- " + " ".join(map(shlex.quote, paths)) + " 2>&1; true"

SINFO_FIELDS = ["NodeList", "Partition", "StateLong", "CPUsState", "Memory", "AllocMem", "FreeMem", "Gres", "GresUsed",
                "Reason"]                                # Reason last
def sinfo_cmd(partitions: list[str], sep: str = SEP) -> str:
    fmt = ",".join(f"{f}:{sep}" for f in SINFO_FIELDS)
    return f"sinfo -h -N -p {shlex.quote(','.join(partitions))} -O {shlex.quote(fmt)}"

# every user's running jobs in the partitions in scope (§5.2.9, host detail only)
OCCUPANT_FIELDS = ["JobID", "ArrayJobID", "ArrayTaskID", "UserName", "Account", "Partition", "StateCompact",
                   "TimeUsed", "TimeLimit", "NumNodes", "NodeList", "tres-alloc", "Name"]   # Name last: free text
def occupants_cmd(partitions: list[str], sep: str = SEP) -> str:
    fmt = ",".join(f"{f}:{sep}" for f in OCCUPANT_FIELDS)
    return f"squeue -a -h -t R,CG -p {shlex.quote(','.join(partitions))} -O {shlex.quote(fmt)}"
def names_cmd(users: list[str]) -> str:               # getent exits 2 when one user is unknown: the rest still print
    return "getent passwd " + " ".join(map(shlex.quote, users)) + " | cut -d: -f1,5; true"

# pending priority evidence (§5.2.10, group slurm.priority): own jobs / own user only, nothing from data interpolated
SPRIO_FORMAT = "%i|%u|%r|%Y|%A|%B|%F|%J|%P|%Q|%T|%N"   # id user partition priority, weighted age assoc fairshare
                                                        # jobsize partition qos tres, nice — one row per job×partition
SPRIO_CMD = f'sprio -h -u "$(id -un)" -o {shlex.quote(SPRIO_FORMAT)}'          # max_stdout 512 000
SSHARE_CMD = 'sshare -h -U -u "$(id -un)" -P -o Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare,LevelFS'
                                                                                # max_stdout 64 000

# on-demand cross-job log search (§4.7.1; every host kind, core/logsearch.py: search_command, not a Slurm command):
# per file  tail -c +<b+1> -- <path> | head -c <n> | grep -a -n -b -o [-i] -E -m <left> -e '.{0,60}(PAT).{0,60}'
#           | head -n <left> | cut -b 1-600      n = min(size, 256 KiB, budget left of 1 MiB); ≤ 20 paths, ≤ 200 lines
```

Incremental history: `since_s` = seconds since `max(last successful sacct − 300 s, now − sacct_window)`.
Validators (`util/shell.py`): `JOBID_RE = r"^\d+(_(\d+|\[[\d,\-%]+\]))?(\.(batch|extern|\d+))?$"`, partition/user/QOS
`^[A-Za-z0-9_.\-]+$`, paths absolute without NUL/newline. Anything else is rejected before rendering. Timeouts: squeue
20 s, sacct 30 s, scontrol loop 15 s, sstat 15 s, logstat 10 s, sinfo 20 s, occupants 20 s, names 10 s, sprio 15 s,
sshare 15 s, log.search 30 s (on demand only; cost per search and host: one round trip, no Slurm RPC, ≤ 1 MiB read).

#### 5.2.3 Parsing rules
- **Rows** are split on `"\n"` only (a trailing `\r` dropped) in every Slurm and ps parser: `str.splitlines()` also
  breaks at `\x0b \x0c \x1c-\x1e \x85 U+2028/9`, which job names, commands and paths may contain.
- **Format fallback** (`|`, then legacy `-o`) only when stderr names the format itself (`format specification`,
  `invalid … field`, `field requested`, `delimiter`); "Invalid user", "Invalid partition", controller / dbd errors never
  downgrade the format.
- **Delimiters**: every `-O`/sacct parser takes `sep` (fixtures use `|`). squeue/sinfo: `line.removesuffix(sep).split(sep)`;
  extras beyond the field count are re-joined into the last field (Command / Reason). With `|`, a `|` inside `Name`
  shifts fields: the row fails validation (unknown State) → `bad_ids` (first field), never "vanished". sacct: a row
  with the wrong field count goes to `bad_lines`/`bad_ids`.
- **Timestamps**: an all-digit value is epoch seconds (UTC) — the live path, via `SLURM_TIME_FORMAT=%s`. Otherwise
  naive ISO is localized with `caps.tz_offset` (fixtures; a cluster that ignored the variable). `N/A`/`Unknown`/`None` →
  None. Durations (`[D-]HH:MM:SS`, `M:SS`, `UNLIMITED`, `INVALID`, `NOT_SET`, `Partition_Limit`) are unaffected.
  Other squeue domains (field probes): `MinMemory` `48G`; `NodeList` empty and `BatchHost` `n/a` for PD;
  `STDERR` `(null)` = merged.
- **Ids**: `job_id_raw` = squeue `JobID` / sacct `JobIDRaw` (an array task's own id, e.g. 31270824; for the pending
  summary row the array job id). Display id (`JobKey.id`) = `JobID` if `ArrayTaskID` is `N/A`, else
  `f"{ArrayJobID}_{ArrayTaskID}"` (`31270818_[1-24%1]` for the pending summary row). scontrol, sstat, `sacct -j` and
  `%j` use `job_id_raw`; scancel/hold/release/requeue use the display id, with any `%N` throttle stripped from a range
  (`31270818_[1-24]`).
- **GPU count**: `gres/gpu` in tres-alloc (running), else tres-per-node × nodes / tres-per-job (pending). **GPU type**,
  first hit wins, recorded in `gpu_type_source`: (1) the single typed key in tres-alloc; (2) scontrol `JOB_GRES=gpu:A30:1`
  or per-node `GRES=gpu:<type>:n(IDX…)` (jobinfo, persisted in jobmeta); (3) tres-per-node / tres-per-job
  (`gres/gpu:nvidia_h200_nvl:2` — k2 2333760 is *running* with an untyped tres-alloc); (4) the single GPU type in the
  sinfo `Gres` of all the job's nodes; (5) jobmeta cache (history rows). Keys are lower-cased (lyra sacct
  `gres/gpu:rtx6000`, sinfo `gpu:RTX6000`).
- **PD start time** = scheduler estimate → `est_start=True` (k2 fixture: StartTime 06:31 for a PD job).
- **States** (`states.py`): squeue/sacct strings → JobState: `PENDING→PD`, `CONFIGURING→CF`, `REQUEUED|REQUEUE_FED→RQ`,
  `REQUEUE_HOLD→RQ (held)`, `RESV_DEL_HOLD→PD (held)`, `RUNNING|RESIZING|SIGNALING|STAGE_OUT→R`, `COMPLETING→CG`,
  `SUSPENDED→S`, `STOPPED→ST`, `COMPLETED→CD`, `FAILED→F`, `SPECIAL_EXIT→RQ (held: slurmctld requeues the job PENDING
  with priority 0)`, `EXPEDITING→RQ` (25.11+), compact `RF→REQUEUE_FED`, `TIMEOUT→TO`,
  `OUT_OF_MEMORY→OOM`, `NODE_FAIL→NF`, `PREEMPTED→PR`, `BOOT_FAIL→BF`, `DEADLINE→DL`, `CANCELLED*→CA` (`CANCELLED by
  10001` → `raw["cancelled_by_uid"]="10001"`), `REVOKED→CA`, unknown → `?` (logged once per string).
  `held = reason in {"JobHeldUser", "JobHeldAdmin"} or raw_state in {"REQUEUE_HOLD", "RESV_DEL_HOLD", "SPECIAL_EXIT"}`.
- **sacct**: `ExitCode "2:0"` → `exit_code=2, exit_signal=0`; `Start` `None`/`Unknown` → None; `NodeList None
  assigned` → None; `End Unknown` while running. Heterogeneous components print as `<het_job_id>+<offset>` (`600+1`,
  steps `600+1.batch`): accepted, and keyed by `JobIDRaw` (601), as squeue keys them.
- **scontrol show job -d -o** (`parse_scontrol.py`): records start at a line beginning `JobId=`; `@@job <id>` lines
  delimit per-id output. A block may hold several records (`scontrol show job <array job id>` prints one per array
  element): the parser keys records by `(JobId, ArrayTaskId)` and the collector takes the record whose `ArrayTaskId`
  equals the row's squeue `ArrayTaskID` (the pending range, e.g. `1-20%1`, for a summary row; no `ArrayTaskId` for
  plain jobs); no matching record → no jobinfo for that row. Any other line inside a block is that id's error (`slurm_load_jobs error: Invalid job id
  specified` → job left the controller). Tokenizing — values may contain spaces (vega/lyra `SubmitLine=sbatch … seed=44
  …`; Hydra-style `key=value` args are common in ML jobs):
  1. Split points are positions where a single space is followed by `Key=` and `Key` matches `[A-Z][A-Za-z0-9_/:]*`
     (Slurm keys are capitalised: `CPUs/Task`, `AllocNode:Sid`, `ReqB:S:C:T`, `JOB_GRES`, `MCS_label`, `CPU_IDs`).
  2. **Greedy keys** `SubmitLine`, `Command`, `Comment`, `AdminComment`, `SystemComment`, `Reason`, `WorkDir`,
     `StdOut`, `StdErr`, `StdIn` only end at the next key from `KNOWN_KEYS` (the keys seen in all fixtures plus those
     `src/scontrol/info_job.c` prints in 25.x/26.x: `SegmentSize`, `CpuFreqMin/Max/Gov`, `GresEnforceBind`,
     `OOMKillStep`, `KillOnInvalidDependent`, `Extra`, …; e.g. `SubmitLine` is always followed by ` WorkDir=`). The path
     keys (`WorkDir`, `StdOut`, `StdErr`, `StdIn`) also end at any CamelCase key, so a key a newer Slurm adds after
     `StdOut` cannot become part of the log path.
  3. The per-node block `Nodes=… CPU_IDs=… Mem=… GRES=…` may repeat (multi-node jobs); collect each occurrence in order
     into `NodeAlloc` (`Nodes` may be a hostlist `n-[01-02]`: expand, same allocation per host). `GRES=gpu:<type>:<n>
     (IDX:0-1)` → `gpu_indices=(0,1)`; several GPU types on one node (`gpu:a100:1(IDX:0),gpu:a100_3g.20gb:1(IDX:1)`) →
     the union of every IDX list; `IDX:N/A` → `()`.
  4. Empty values are legal (`NodeList=`, vega `StdErr=` → merged with stdout; k2 repeats the stdout path → merged).
- **sstat** (`parse_sstat.py`): rows `JobID.step|NTasks|AveCPU|AveRSS|MaxRSS|TRESUsageInAve|TRESUsageInMax|TRESUsageInTot`
  (fixtures lack Tot: use `Ave × NTasks`). Ignore `.extern` (garbage AveCPU `213503982334-14:25:51`). Per job pick, among
  non-extern steps with `gres/gpumem > 0`, the one with the largest NTasks, ties → largest Tot `gres/gpumem` (batch for
  plain jobs, `.0` for srun/torchrun jobs). Then `util_sum_pct = Tot gres/gpuutil`, `util_mean_pct = min(100, sum /
  alloc.gpus)`, `util_peak_pct = min(100, Max gres/gpuutil × NTasks / alloc.gpus)` (Max is per task, i.e. summed over that task's
  GPUs; shown as "peak"), `mem_used_mib = Tot gres/gpumem`
  (`24870M` → MiB), `mem_total_mib = gpu_types.mem_mib(type) × gpus`. GPU fields only when `caps.gpu_tres` is not False;
  AveCPU/MaxRSS always: they become `Job.cpu_pct` (AveCPU × NTasks / elapsed, % of one core, lifetime average) and
  `Job.rss_bytes` (MaxRSS, peak) of RUNNING jobs, and fill the running steps' AveCPU/MaxRSS in the detail Steps tab
  (sacct leaves them empty until the step ends; M3 F8). Note string "Slurm accounting · per job · refreshed every 60 s". The sum-over-tasks semantics is
  verified at M3 (Appendix C).
- **GPU memory by type** (`gpu_types.mem_mib(type) -> float | None`): exact table first (`nvidia_h200_nvl` 143771,
  `nvidia_h100_80gb_hbm3` 81559, `nvidia_h100_nvl` 95830, `nvidia_geforce_rtx_3090` 24576,
  `nvidia_a30`/`a30` 24576, `nvidia_rtx_pro_6000_blackwell_server_edition` 97887); ambiguous names → None (`rtx6000`
  is a 24 GB Quadro elsewhere, while lyra's s71/r71 nodes carry the 96 GB RTX PRO 6000 Blackwell (97887 MiB, Attach
  field probes; the M3 "47884 MiB" was one job's usage) — an Attach sample replaces the type with the
  nvidia-smi model, §7.16.5); then MIG `_<n>g.<m>gb` → m GiB; then a
  generic `(\d+)gb` → GiB (`nvidia_a100-sxm4-40gb`, `nvidia_a100-sxm4-80gb`, `nvidia_a100_80gb_pcie`,
  `tesla_v100-pcie-16gb`); else None (lyra `v100`: memory shown as used only).
- **sinfo nodes** (`parse_sinfo.py`): Reason last. State `mixed-` → base `mixed` + flags `{"-"}` (strip any of
  `*~#!%$@^-+`). `CPUsState` `A/I/O/T`. `Gres`/`GresUsed`: split on commas **outside parentheses**
  (`gpu:tesla_v100-pcie-16gb:3(IDX:0,2-3)`), item `name[:type]:count[(…)]` (lyra has no parens: `gpu:V100:4`). Free per
  type = total − used. **Usable** (counts free GPUs): base ∈ {idle, mixed, allocated, completing, planned} and none of
  the flags `* ~ # ! % $ @ ^`; partition `gpus_free` also skips nodes with no idle CPU (an `allocated` node's free GPUs
  cannot be scheduled; M3 k2: 4 of 25). MIG profiles are separate types and are labelled as such (`A100-1g.10gb`, not
  `A100-80G`). The `-` flag (vega: 30 of 33 nodes `mixed-`) and the `planned` base state mean "planned
  by backfill for a pending job": usable, but their free GPUs also count into `gpus_free_planned` and the UI marks them.
  `reserved`, drain*, down, fail*, maint, future, inval, unknown, power_* → not usable. Rows per (node, partition) are
  merged by node name into `partitions`.
- **Hostlists** (`util/slurmfmt.expand_hostlist`): `n-[01-03,05]`, `lyra-gpu-s71-[00-03]`, multiple bracket groups,
  zero-padding preserved, top-level commas outside brackets; cap 4096 names.

#### 5.2.4 Merging live and history (SlurmCollector state)
```
state: live_prev: dict[id, Job]; history: dict[id, Job]; jobinfo: dict[id, JobInfo]; vanished_at: dict[id, datetime]
on ingest:
  if squeue ok (OK|EMPTY):
      live, bad = parse(squeue)                     # bad = bad_ids
      live = enrich(live, jobinfo, sstat cache, sinfo gpu types)
      for id in bad ∩ live_prev: live[id] = live_prev[id]          # kept, stale; never "vanished"
      for id in live_prev − live:
          if id is an array summary (`500_[1-20%1]`) and its array is still in squeue: skip  # the range shrank
          vanished_at[id] = now; followups += {"slurm.vanished"}
      if any id in live with id ∉ jobinfo or (live_prev[id].state == PD and live[id].state == R):
          followups += {"slurm.jobinfo"}
      live_prev = live
  else: live = live_prev                           # stale; never compute vanished from a failed squeue
  history.update(parse(sacct)); drop history entries with end < now − sacct_window
  jobs = {id: j for id, j in history.items() if j.state.is_terminal}
  jobs.update(live)                                # live always wins for active jobs
  for id, t in vanished_at.items():
      if id in jobs and jobs[id].state.is_terminal: forget id
      elif now − t < 10 min: jobs[id] = replace(last_seen, source="vanished", hints=("ended; waiting for accounting",))
      else: jobs[id] = replace(last_seen, state=UNKNOWN, source="vanished")
  return CollectorOutput(jobs=jobs, …)
```
Log paths for history-only jobs: `jobmeta` cache (paths seen while live, persisted because `scontrol` forgets a job
`MinJobAge` = 300 s after it ends) → sacct `StdOut/StdErr` if supported (they hold the **unexpanded** `--output`
pattern on 25.05/25.11/26.05 — M3 F2 — and are expanded like squeue's, §5.2.5, with JobName, JobIDRaw, User, WorkDir
and `%N` from NodeList) → host `log_templates` → Slurm default
`{workdir}/slurm-{job_id_raw}.out` (`slurm-{array_job}_{task}.out` for arrays). Template/default paths are marked
`source="template"` and verified by `stat` when the detail screen opens.

#### 5.2.5 Log path expansion (`logpaths.py`)
For a squeue `STDOUT`/`STDERR` (or sacct `StdOut`/`StdErr`) pattern, like slurmstepd's `expand_stdio_fields`:
`%%`→`%`, `%A`→array job id (or job id), `%a`→array task id (**NO_VAL `4294967294`** if not an array — what Slurm
writes), `%j`/`%J`→`job_id_raw`, `%N`→first allocated node (short), `%n`→`0`, `%s`→`batch`, `%t`→`0`, `%u`→user,
`%x`→job name, `%b`→task id mod 10 (`4` for a non-array job), `%r`→RestartCnt (squeue; unknown → unexpandable), `%S`
(SLUID, 25.11+) → unexpandable; a width between `%` and the letter zero-pads (`%4a`, capped at 10). A pattern that
contains `\` is not expanded at all, only its backslashes are removed. Relative results are joined to `WorkDir` (lyra
`slurm-%j.out`). A pattern with unexpandable tokens (PD job, `%N` unknown, array summary
`%a`) keeps `pattern` and `path=None` until `scontrol` provides `StdOut=`. `StdErr` = `(null)` (squeue), empty (vega
scontrol) or equal to `StdOut` (k2) → `merged=True`.

#### 5.2.6 Hints and explanations (`states.py`)
- Pending reason → text: `Priority` "waiting for priority", `Resources` "waiting for resources",
  `QOSMaxJobsPerUserLimit` "per-user job cap (QOS)", `AssocGrpGRES` "group GPU cap", `Dependency` "waiting for
  dependency", `DependencyNeverSatisfied` **"will never run: dependency failed"** (red hint), `ReqNodeNotAvail` "requested
  node unavailable", `PartitionTimeLimit` "time limit above partition max — will never run" (red), `BeginTime`
  "scheduled start later", `JobHeldUser`/`JobHeldAdmin` "held". Unknown reasons are shown raw. The full common set and
  its advice: §5.2.8.
- Stall hints for RUNNING Slurm jobs **and ACTIVE tasks**: `log idle {age}` when the stdout mtime is older than
  `stall_after` (30 min); `GPU idle {dur}` when `util_mean_pct < 5` in every sample over ≥ 15 min (sstat for Slurm,
  nvidia-smi for tasks; ≥ 2 samples).
- Exit explanations (detail): `rc ≥ 128` on a step → `128+N = SIG<N>` (143 = SIGTERM, 137 = SIGKILL); `PREEMPTED`
  + `PreemptMode=REQUEUE` (k2, vega) → "preempted; Slurm will requeue it" vs `CANCEL` (lyra) → "preempted and killed;
  resubmit"; `OUT_OF_MEMORY` → "out of memory: exceeded --mem {ReqMem}; MaxRSS {x}" (when the sampled MaxRSS is below
  --mem: "… MaxRSS (sampled) {x} — the spike was likely missed between samples"; a step's `0:125` is Slurm's OOM marker
  and is not reported as "killed by signal 125"; hints already said by an explanation line are not repeated, nor an
  `exit N` hint next to an explanation that names the exit code); `TIMEOUT` → "hit time limit {Timelimit}";
  `CANCELLED by <uid>` → "cancelled by you" when uid == caps.uid else "cancelled by uid N".
- "Why it ended" (Steps tab, UI): for an ended job that did not complete, the failure line of the streamed log tail
  (`util/logscan.exit_cause`: last Traceback/OOM/Killed/`…Error:` line of the last 50, preferring the exception over the
  bare `Traceback (most recent call last):` header) is the first bullet (`log: torch.OutOfMemoryError: …`); a merged
  stdout/stderr file is listed once (`stdout+stderr: <path> (merged)`).

#### 5.2.7 GPU information for Slurm jobs
1. **Always, free**: GPU count and type (§5.2.3) → the GPUS column (`2×H200`).
2. **Background, cheap**: physical indices per node from `scontrol show job -d -o` (once per job: first sight and
   PD→R; persisted) and aggregate utilisation/memory from `sstat` every 60 s where `gpu_tres` (k2, vega). lyra shows
   "n/a — this cluster does not record GPU usage".
3. Per-GPU live readings and the job's process list need compute-node access: **Attach (v0.2, §7.16)**, only while
   JobDetail's GPU/Procs tab is open. Without a live sample (attach off / blocked / not yet read) JobDetail shows steps
   instead (sstat NTasks/AveCPU/MaxRSS/TRESUsageInMax while running — the Steps tab's
   "GPU (max per task)" of a running step comes from the same sstat row — sacct after) with the note
   "per-process view needs compute-node access (v0.2)". CPU: the cores estimate uses the busiest non-batch,
   non-extern step (AveCPU × NTasks / elapsed) when one has numbers — in srun/DDP jobs the batch step is only the
   launcher — else the batch step; CPU use above 999 % of one core reads `≈ 61 cores`, never a clamped `999%`.

#### 5.2.8 Failure triage and efficiency (`util/logscan.py`, `util/efficiency.py`, `states.py`)
Three researcher questions — "why did it die?", "did I over-request?", "why is it still pending?" — answered from data
RunFleet already collects. **No new periodic remote command**: log evidence comes only from log text already fetched
(job detail's stream, the preview's one-shot peek); TotalCPU is added to the on-demand detail `sacct` steps query only,
and only where `caps.sacct_fields` lists it.
- **Cause chip** (`logscan.cause_of(job, detail=None, lines=()) -> Cause | None`; `Cause(code, text, evidence)`): ended
  jobs that did not succeed only. Order: the scheduler state (`OUT_OF_MEMORY` → `OOM-HOST`, `TIMEOUT`/`DEADLINE` →
  `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`/`BOOT_FAIL` → `NODE_FAIL`; a `CANCELLED` record whose log says `DUE TO TIME
  LIMIT` / `PREEMPTION` / `NODE FAIL` is that instead, else `CANCELLED`), then the log by `CAUSE_RULES` (most telling
  first: `OOM-GPU`, `OOM-HOST`, `QUOTA`, `IMPORT`, `SEGV`, `NAN`, `NCCL` — one rank's OOM / crash shows up as NCCL
  timeouts on the others), then the exit status (`rc ≥ 128` → `SIGNAL rc-128`, 139 → `SEGV`, the job's or a non-extern
  step's signal, `0:125` excluded), then `EXIT n`. The evidence is the first matching line (ANSI stripped, last `\r`
  segment, ≤ 200 chars). Job detail keeps the last `TRIAGE_LINES` (2000) streamed lines; the chip line sits under the
  title (tooltip = `Cause.plain()`), `L` opens the log viewer landed on the evidence line. The Jobs preview shows the
  chip from its peek and drops the ended job's summary hint it repeats.
- **Error lines** (`logscan.ERROR_LINE` / `WARN_LINE` / `line_level`): log panes colour error lines red, warnings
  yellow (at write time); the log viewer's `]` / `[` jump between them over the wrapped display lines (a match a wrap
  splits counts on its head; a run of matching display lines is one stop), pausing follow like a search hit.
  `NCCL INFO` chatter is neither.
- **Efficiency card** (`efficiency.efficiency(job, steps) -> Efficiency | None`, `rows()`, `lines()`): Slurm jobs that
  ran. Finished: time used / limit, CPU = TotalCPU (allocation row, else Σ steps, else Σ AveCPU × NTasks over
  non-extern steps) / (Elapsed × AllocCPUS), memory peak = max over work steps of MaxRSS × NTasks on a single-node job
  (per-task MaxRSS on multi-node) / requested (AllocTRES mem, else ReqMem), GPU mean = the busiest work step's
  `TRESUsageInTot gres/gpuutil` / GPUs (clamped to 100; same reading as §5.2.3 sstat). reportseff colouring: CPU < 50 %,
  memory < 25 %, time < 25 % (completed jobs only) amber; GPU < 5 % red with a "GPU idle" note. "next time": COMPLETED →
  `--time` = Elapsed × 1.25 rounded up (15 min steps < 1 h, hours < 1 d, 6 h steps), `--mem` = peak × 1.25 rounded up
  (GiB to 16G, 4G steps to 64G, then 8G) — both capped at what was requested, no `--mem` for multi-node jobs; TIMEOUT
  → limit × 1.25 ("a guess"); OUT_OF_MEMORY → `--mem` × 1.5; other failures → none. Running jobs: live headroom
  (sstat MaxRSS vs requested memory, red ≥ 90 %) as a `room` line in the GPU box. The collector puts `lines()` on
  `JobDetail.efficiency` (plain text, e.g. for a support bundle); job detail shows the coloured rows in the Summary's
  right column in place of an ended job's GPU box ("n/a — job ended"), one `eff …` line in the compact layout.
- **Pending reasons** (`states.REASONS`, `NEVER_RUNS`, `REASON_ADVICE`, `reason_advice`): the common set — Priority,
  Resources, Dependency, DependencyNeverSatisfied, AssocGrp{GRES,CPULimit,MemLimit,NodeLimit,JobsLimit,BillingMinutes,
  CPUMinutesLimit,GRESMinutes}, AssocMaxJobsLimit, QOSMax{GRESPerUser,JobsPerUserLimit,CpuPerUserLimit,MemoryPerUser,
  NodePerUserLimit,WallDurationPerJobLimit}, QOSGrp{GRES,CpuLimit,MemLimit,NodeLimit}, MaxGRESPerAccount,
  Partition{Time,Node}Limit, ReqNodeNotAvail, "Nodes required for job are DOWN, …", Reservation, BeginTime,
  JobHeld{User,Admin}, "launch failed requeued held", JobArrayTaskLimit, BadConstraints, InvalidAccount, InvalidQOS —
  each with a short text and one actionable line (unknown `Assoc…` / `QOS…` limits get a generic one). Never-runs (red):
  DependencyNeverSatisfied, PartitionTimeLimit, PartitionNodeLimit, QOSMaxWallDurationPerJobLimit, BadConstraints,
  InvalidAccount, InvalidQOS (the priority evidence behind `Priority` / `Resources`, read by its own group: §5.2.10).
  The collector copies the advice to `Job.raw["reason_advice"]` (UI must not import
  backends; the UI's short wording lives in `ui/render._REASONS`); `explain()` for a queued job returns the advice and,
  when squeue gave one, the start estimate with its caveat ("Slurm's backfill guess — it moves when running jobs end
  early or higher-priority jobs arrive").


#### 5.2.9 Who runs on a node (`parse_occupants.py`)
"A node is busy and none of it is mine — whose is it, and when does it free up?" Where `PrivateData` does not hide
jobs (none of the three clusters does), `squeue -a` lists every user's jobs, so RunFleet shows them — **only while the
host detail is open** (`slurm.occupants` is a focused-only cadence group, §5.4): the fleet view never pays for it.
- **Reading**: one `occupants_cmd` over the **nodes** in scope (`-w`, from the last sinfo: a node shared with a
  partition outside the scope, such as a lab's private one, shows those jobs too; `-p <partitions>` only before the
  nodes are known or past 1000 nodes), planned **with** `slurm.sinfo` so the GPU counts and the jobs are of one
  moment (sections of one name are planned once per probe). Rows parse into `Occupant`s (display id, user, account,
  partition, compact state, elapsed / limit seconds, expanded node list, job GPUs from tres-alloc: no `gres/gpu` at
  all = 0, a `gres/gpu` that does not parse = unknown). Shown for 3 min (`models.NODE_JOBS_TTL`): every nodes output
  in that window carries them (`Node.jobs`, `Node.jobs_at`), later ones carry `jobs=None`, and the renderers read
  `Node.jobs_seen(now)` (None once the reading is older than the TTL), so a reading never outlives the visit even when
  no new nodes output arrives. A failed reading keeps the last one (within the TTL) and reports a section error; a
  mostly unparsable one is a PARSE error.
- **GPUs per node**: squeue has no per-node split. One node: exact. Several: an even split marked inexact (`~2`);
  an uneven total: unknown (`?`). (`scontrol show job -d` has the split, but one RPC per job is too dear here.)
- **Names**: new users of a reading are looked up once with `getent passwd` (event group `slurm.names`, ≤ 200 per
  probe, cached for the session; a user without an entry is not asked again). GECOS up to the first comma.
- **Display** (host detail, mockup H): the node table's JOBS column lists the user's own job ids (accent), then the
  other users with the GPUs they hold there (`bob×4 · carol×3 +1`, most GPUs first; ends in `+N` when it does not
  fit). Without a reading it falls back to the user's own jobs. A panel under the table lists the jobs on the node
  under the cursor: JOB, USER, NAME (GECOS), ACCOUNT, PARTITION, GPU, ELAPSED / LIMIT, LEFT, JOB NAME — own jobs
  first, then the one that ends soonest; ≤ 8 rows then `+N more`; columns dropped when narrow (account, name,
  partition, time). Times move on from `jobs_at`. Subtitle: `N GPUs free now`, or on a full node
  `next GPU frees in ≤ 2h17m` (time limits are upper bounds; a job whose split is unknown counts as holding GPUs),
  plus `as of 2m ago` once a reading is ≥ 90 s old.
  With `PrivateData=jobs` in `caps.slurm_config` the panel says only the user's jobs are listed.

#### 5.2.10 Pending priority evidence (`priority.py`)
"Why isn't my job starting?" — beyond the reason code (§5.2.8) and squeue's start estimate, RunFleet shows what Slurm
itself says about the job's priority. **Evidence, never a promise**: no start time of its own and no queue position.
- **Settings** (`priority.settings(caps.slurm_config)`): `PriorityType`, the weights (`PriorityWeightTRES` `(null)` → 0,
  a per-TRES list → its sum), flags, `PriorityDecayHalfLife`, `PriorityMaxAge`, `PriorityUsageResetPeriod`, read by
  detection (§5.1). **Ranked** = `priority/multifactor` with some weight > 0. `priority/basic`, or multifactor with every
  weight 0 (all pending jobs have priority ~1), is **not ranked**: no remote command at all, and every own pending job
  gets the hint "this cluster does not rank by priority: pending jobs start in submit order as resources free up
  (backfill can start short jobs earlier)" instead of a row of zero factors. Unknown type (capabilities of an older
  detection) → no evidence.
- **Reading** (cadence group `slurm.priority`, §5.2.1): one probe with `slurm.sprio` (own pending jobs, one row per job ×
  partition) and `slurm.sshare` (the user's associations; left out when the fair-share weight is 0). Due at the first
  tick that has an own pending, not held job, then at most every 600 s (floor; ×6 when hidden; a busy controller
  slows it like every `slurm.*` group); the group disappears when no such job is left, and its last run / backoff
  survive, so a job submitted later waits for the floor. Failures back off (×2 per failure, up to 1 h) and keep the
  last reading; a mostly unparsable answer is a PARSE error. sprio refusing a non-multifactor cluster ("not running a
  supported priority plugin" — the cluster changed since detection) stops the group for the session, without an error.
  sprio's "Unable to find jobs matching user/id(s) specified" is an empty answer, on stdout or (with a non-zero exit)
  on stderr: the job started between squeue and sprio. Both answers are `CollectorOutput.handled`: the poller counts
  them as ok — no error, no backoff, and a refusal does not leave the host DEGRADED by a section that never runs again.
- **Cost**: sprio and sshare are one slurmctld RPC each (priority factors of the user's pending jobs; the association
  tree's share info), ≤ 2 per 10 min per host while a job of yours waits, nothing otherwise — no slurmdbd query, no
  compute node.
- **Evidence** (`priority.evidence(job, rows, shares, settings, sampled_at) -> PendingPriority`, on `Job.priority` of own
  PENDING/REQUEUED jobs; pure and clock-free so a job only changes with a new reading): the sprio row with the highest
  priority (its partition; the other partitions' priorities in `others`), every factor with a weight > 0 (value =
  weighted contribution, weight = the most it can add; a non-zero nice as a negative term), the dominant (largest)
  factor, and the user's sshare rows with the job's account first (`LevelFS` `inf` → None). Hints, most telling first,
  ≤ 4: held ("does not compete until released"); a never-runs reason (§5.2.8: "priority does not matter here", nothing
  else); another blocking reason ("held back by per-user job cap (QOS), not by priority: these factors only count once
  that clears"); the dominant factor and what it means ("fairshare carries most of its priority (3,000 of 3,017): …");
  every factor 0 ("it starts when nothing ranked higher fits — backfill can still start it early if it is short"); not
  in the last reading yet; fair share of the job's account below 0.25 ("fairshare 0.03: your group (lab) used much
  more than its share recently; it recovers with a half-life of 14 days" — "you used more than your share of lab"
  when the user's own LevelFS < 1; with `PriorityDecayHalfLife` 0: "usage does not decay on this cluster; it is reset
  monthly", or "… is reset only by the admins" for `NONE`/`NOW`), or ≥ 0.75 (good standing / no recent usage); age
  still growing ("waiting raises it: age adds up to 1,000 after 7 days in the queue").
- **Display**: job detail of a pending job — the Summary's right column shows a **Priority** box in place of the GPU
  box ("n/a — job pending"), titled `Priority · sprio 4m ago`: `priority 3,017 in gpu-preempt · lab 2,017`, one row per
  factor (neutral bar = share of that factor's weight, `value / weight`, `◀ dominates`), `share` rows per account
  (fair share, usage vs share, "no recent usage"), then the hints; an unranked cluster shows its hint only. The compact
  layout (< 110×30) and the Jobs preview carry one line: `prio 3,017 · fairshare dominates · fairshare 0.03 · 4m ago`
  (`prio not ranked here: submit order as resources free up (+ backfill)`; `prio not read yet …`). `snapshot --json`
  carries it as `jobs[].priority` (additive; `runfleet.snapshot/1` unchanged).

### 5.3 Workstations (`backends/process/`, `backends/common/nvidia.py`, `backends/common/procfs.py`)

#### 5.3.1 Sections and cadence

| Group (mode) | Section (collector) | Interval / floor / hidden× / focused× | Command |
|---|---|---|---|
| `nvidia.gpu` (interval) | `nvidia.gpu` (nvidia, late) | 5 s / 2 s / ×3 / ×0.6 | background wrapper below around `nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,fan.speed,pstate,utilization.memory --format=csv,noheader,nounits` |
| `nvidia.gpu` | `nvidia.apps` (nvidia, late) | same group | wrapper around compute apps, then `ps` of those pids |
| `procfs.host` (interval on workstations) | `procfs.host` (procfs) | 5 s / 2 s / ×3 / ×1 | see below |
| `process.ps` (interval) | `process.ps` (process) | 10 s / 5 s / ×3 / ×0.5 | `date +%s; ps -ww -u "$(id -un)" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args=` |
| `process.ps` | `process.fds` (process) | same group, only pids not yet cached | `for p in <pids>; do c=$(readlink /proc/$p/cwd …); o=$(readlink /proc/$p/fd/1 …); e=$(readlink /proc/$p/fd/2 …); printf '%s\|%s\|%s\|%s\n' "$p" "$c" "$o" "$e"; for t in "$o" "$e"; do case "$t" in /dev/*\|*' (deleted)') ;; /*) stat -c '@@stat %n\|%s\|%Y' -- "$t" 2>/dev/null ;; esac; done; done; true` — the `@@stat` lines give new log paths their size/mtime in the same round (a 2-round `snapshot` never reaches `process.logstat`; M3 F9 residual) |
| `process.logstat` (interval) | `process.logstat` (process) | 60 s / 30 s / ×3 / ×0.5 | `stat -c '%n\|%s\|%Y' -- <task log paths> 2>&1; true` |
| `process.exitscan` (event) | `process.exitscan` (process) | — | per ended task: `tail -c 8192 -- <stdout> [<stderr>]` (§5.3.2 step 8) |

On a workstation the tick is 5 s; `process.*` groups run every second tick. On a Slurm profile `procfs` declares
`procfs.host` as **piggyback** at the jobs interval (it checks `"slurm" in ctx.caps.profile`), so it never adds round
trips to a login node.

```sh
# nvidia.gpu (nvidia.apps uses the same wrapper). rf runs it under `timeout`, but a hung nvidia-smi is usually in
# D state: SIGKILL does not end it and GNU timeout would wait for it. So: background it, poll, abandon.
# Guard: refuse when two of the user's nvidia-smi are already stuck — D state (a hung query survives SIGKILL) or still
# held by a RunFleet wrapper (`rf_nv` in the parent's argv: abandoned in the background). Healthy instances (a job's
# `nvidia-smi -l`, the user's `dmon` in tmux) are not counted: counting them refused every probe (Attach and
# workstations) while two of them ran.
# A `ps` without -u/-o (BusyBox) falls back to pgrep (every instance counts; rc 1 = none); when processes cannot be
# listed at all the query is refused — the guard never fails open.
rf_g() {
  if rf_p=$(ps -ww -u "$(id -un)" -o pid=,ppid=,stat=,args= 2>/dev/null); then
    printf '%s\n' "$rf_p" | awk '{c=$4; sub(/.*\//, "", c); a[$1]=$0} c == "nvidia-smi" {n[$1]=$2; s[$1]=$3} END {k=0; for (p in n) if (s[p] ~ /^D/ || index(a[n[p]], "rf_nv")) k++; print k}' || echo x
    return
  fi
  rf_p=$(pgrep -u "$(id -un)" -x nvidia-smi 2>/dev/null); [ $? -le 1 ] || { echo x; return; }
  printf '%s\n' "$rf_p" | grep -c .
}
rf_k=$(rf_g)
case $rf_k in ''|*[!0-9]*) echo 'nvidia-smi guard: cannot list processes (refusing)' >&2; exit 125;; esac
[ "$rf_k" -ge 2 ] && { echo 'nvidia-smi already running (hung?)' >&2; exit 125; }
[ -n "$RF_T" ] || exec nvidia-smi --query-gpu=… --format=csv,noheader,nounits      # no temp dir: run in foreground
o="$RF_T/nv.$$"
( nvidia-smi --query-gpu=… --format=csv,noheader,nounits >"$o" 2>"$o.e"; echo $? >"$o.rc" ) </dev/null >/dev/null 2>&1 &
i=0; while [ ! -s "$o.rc" ] && [ $i -lt 60 ]; do sleep 0.1 2>/dev/null || sleep 1; i=$((i+1)); done
[ -s "$o.rc" ] || { echo 'nvidia-smi did not answer within 6 s (abandoned)' >&2; exit 124; }
cat "$o"; cat "$o.e" >&2; exit "$(cat "$o.rc")"

# nvidia.apps payload (inside the same wrapper)
a=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits) || exit $?
printf '%s\n' "$a"; echo '@@ps'
p=$(printf '%s\n' "$a" | awk -F', ' 'NF>=2 {print $2}' | sort -u | paste -sd, -)
[ -n "$p" ] && ps -ww -o pid=,ppid=,user:64=,etimes=,pcpu=,pmem=,rss=,stat=,args= -p "$p"; true
# user:64: procps cuts longer names to 7 chars + "+" in a default-width column ("alice000+" ≠ me)

# procfs.host
printf 'ncpu=%s\nepoch=%s\n' "$(nproc 2>/dev/null)" "$(date +%s)"; printf 'load=%s\n' "$(cat /proc/loadavg)"
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo; head -1 /proc/stat
printf 'uptime=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
```
The background child's stdio is detached from the probe, so an abandoned nvidia-smi neither holds the ssh channel open
nor blocks later sections; the probe's EXIT trap removes `$RF_T` (an abandoned writer keeps an unlinked file). Timeouts:
`nvidia.*` 8 s (the wrapper gives up at 6 s), others 5 s. Measured: `--query-gpu` ≈ 100 ms on labstation, 30 ms on lynx.

Parsing:
- `nvidia.gpu`: split on `", "`; field 0 = index, field 1 = uuid; the **last `len(fields) − 3`** values (9 for the query
  above: util.gpu, memory.used, memory.total, temperature.gpu, power.draw, power.limit, fan.speed, pstate,
  utilization.memory) are taken from the right in query order; everything between uuid and them is re-joined as the
  name (lynx: `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`). The parser takes the field list as a
  parameter (fixtures use the 9-field query without the last three). `[N/A]`/`[Not Supported]` → None; `pstate` is a
  string; `memory.*` MiB, `power.*` W.
- `nvidia.apps`: `gpu_uuid, pid, used_memory` rows until `@@ps`, then `ps` rows (`pid ppid user etimes pcpu pmem rss stat
  args` — the first 8 whitespace fields, the rest is args). Join uuid → GPU index. Empty apps output is a valid EMPTY.
  Rows become `Proc(pgid=None)` in `CollectorOutput.gpu_procs`.
- nvidia-smi failures: rc 9 / "couldn't communicate with the NVIDIA driver" → `ErrorKind.REMOTE` "NVIDIA driver not
  loaded" (GPUs unavailable; processes/CPU still work); rc 6 "No devices were found" → zero GPUs; rc 125 (guard) or
  124 (abandoned) → "nvidia-smi hung" — a section failure (DEGRADED), never a transport failure; after 3 consecutive
  such outcomes the `nvidia.gpu` group is suspended for 5 min.
- `procfs.host`: CPU % from the delta of the `cpu` line between two ticks (`(Δtotal − Δidle − Δiowait) / Δtotal`); first
  tick → None. `MemAvailable` drives the mem bar.

#### 5.3.2 Task discovery and grouping (`backends/process/tasks.py`, pure; WP2-internal signature)

Inputs per tick: the `process.ps` sample (own `Proc`s + remote epoch), the `MonitorView` (latest GPU sample and its
`gpu_procs`, used only if `gpus_age ≤ 3 × poll_gpu`), the fd cache, the previous tasks.

1. **Discovery** starts a task from an *interesting* process: an own process that holds a GPU, or an own non-noise
   process with `etime ≥ task_min_age` (60 s) **and** `cpu_pct ≥ task_min_cpu` (5 %).
2. **Noise** (`cfg.ignore`, regexes over `args`, defaults): bare/login shells `^-?(ba|z|da|k|tc|fi)?sh$`, `^sshd`,
   `^(\S*/)?systemd --user`, `^\(sd-pam\)`, `^(tmux|screen|SCREEN)`, `sftp-server|ssh-agent|gpg-agent|dbus-daemon`,
   `\.vscode-server/|code-server|cursor-server`, `^(\S*/)?(nvitop|htop|top|btop|less|man|watch)( |$)`,
   `python\S* (-m )?\S*(nvitop|gpustat)`; the shell and tmux/screen patterns also accept a path prefix
   (`^-?(\S*/)?(ba|z|da|k|tc|fi)?sh$`, `^(\S*/)?(tmux|screen|SCREEN)`) for `/usr/bin/zsh`-style `args`. Shell *runners* with arguments (`bash run.sh`) are not noise. RunFleet's own probe
   processes are always younger than 60 s and so never interesting.
3. **Root**: from each interesting process climb `ppid` while the parent is in the user's process list, is not noise and
   is not pid 1; the topmost such ancestor is the root (`tmux → zsh → python train.py` ⇒ root `python train.py`;
   `nohup python -m orb.launch` re-parented to 1 ⇒ root `orb.launch`, matching the lynx/labstation fixtures: lynx
   642229 launches 642234 and 786345 on GPUs 1 and 0).
4. **Task** = root + all its *current* descendants in this `process.ps` sample (membership is not filtered by
   interest). Roots reached from several processes merge.
5. **Sticky identity**: a task, once discovered, lives as long as its root `(pid, start_epoch ± 5 s)` appears in a
   successful `process.ps` sample — whatever its CPU or GPU use (data prep, checkpoint writes on NFS, a launcher between
   manifest items, a stopped process). `JobKey.id = f"p{pid}-{start_epoch}"` with `start_epoch` fixed at discovery
   (the ID column shows `JobKey.label`, `p642229`), so a reused pid can never collide with an ended task.
6. Other users' GPU processes become single-process tasks (`is_mine=False`, no tree, read-only), shown only when the
   user filter includes others (`u`).
7. Fields: `name` = label(root) + `f" ×{k} ({label(child)})"` when k GPU-holding descendants have a different label;
   label rules: `-m pkg.mod` → `pkg.mod`; first `*.py`/`*.sh` argument → basename; launcher prefix kept for
   `torchrun|accelerate|deepspeed|mpirun`; else basename(argv0). `state`: any member R/S/D → RUNNING, all T → STOPPED.
   `elapsed` = root etime; `start_time` = remote_epoch − etime. `cpu_pct`/`rss_bytes` summed; `alloc.gpus` = distinct
   GPUs of members (type = nvidia-smi name, `gpu_type_source="nvidia-smi"`); `gpu` = `JobGpuUsage(source="nvidia-smi",
   util_mean_pct = mean util of those GPUs, util_peak_pct = max, mem_used_mib = Σ member used memory, note="GPU-level
   utilisation; shared GPUs count other processes too")`. `workdir` = root cwd; `logs` from root fd1/fd2 if they are
   regular file paths (not `/dev/pts/*`, `pipe:`, `socket:`, `/dev/null`), else from the first GPU-holding member;
   `merged` when fd1 == fd2. `pids` = members, root first. Members get `Proc.task` = the task key; `CollectorOutput.procs`
   = all members + all GPU pids; `gpus` = the monitor's GPUs with `GPUProcess.job` annotated. On every tick (also
   those without `process.ps`) `ingest` re-annotates the current `MonitorView.gpus`/`gpu_procs` from its last known
   tasks and returns `gpus` and `procs` whenever `MonitorView.gpus_at` is not None; tasks change only on a `process.ps`
   result.
8. **Ended tasks**: only when `process.ps` was OK/EMPTY **and** the root `(pid, start)` is absent: `end_time` = now,
   follow-up `process.exitscan`. Its last 50 lines matching `Traceback \(most recent call last\)|CUDA out of memory|
   OutOfMemoryError|MemoryError|Killed|Segmentation fault|core dumped|^\S*Error:` → state `FAILED`, raw_state
   `exited (log: <first matching line, 60 chars>)`, red hint; no match or no log → `EXITED` (group ENDED: "ended, exit
   status unknown"). Ended tasks stay `recent_window` (24 h) in the job set and are persisted (`ended_tasks`, 7 days).
9. `process.fds` is only asked for (root, GPU pid) pairs not in the per-host fd cache; steady state costs nothing.

### 5.4 Polling engine and load budget (`core/poller.py`, WP1-internal)

```python
class HostPoller:
    def __init__(self, host: HostConfig, transport: Transport, profile: Sequence[Collector], store: FleetStore,
                 clock: Clock, rng: random.Random, lanes: HostLanes) -> None: ...
    async def run(self) -> None: ...                  # until cancelled; never raises (exceptions -> INTERNAL error)
    def wake(self, reason: str, groups: frozenset[str] = frozenset()) -> None: ...   # manual refresh, post-action
    def set_visibility(self, v: Visibility) -> None: ...
    def set_multiplier(self, m: float) -> None: ...   # global +/- times idle/blur throttle
    def set_paused(self, paused: bool) -> None: ...   # idle pause
```
**Ticks.** Effective period of a group = `interval × visibility factor × multiplier × (1 + (n_hosts − 10)/20 if > 10
hosts, workstations only)`, clamped to `floor`, plus the group's failure backoff. The **tick period** = the minimum
effective period over `interval`-mode groups (Slurm 15 s, workstation 5 s), published as `HostStatus.tick_interval`.
A `focused_only` group (`slurm.occupants`, §5.2.9) exists only while the host is FOCUSED: it neither sets the tick nor
is due otherwise; when the host becomes focused and its last run is older than its period it is forced and the host
woken (3 s floor) — while the host backs off or is paused it is only forced, for the next scheduled tick, and a focus
wake still pending when a probe fails is dropped (the backoff deadline wins).
Jitter: tick k is placed at `anchor + (k + U(0, 0.2)) × period` (fresh draw per tick; the grid is re-anchored when
the period changes, after a wake and after a failure), so gaps vary ±20 % but the mean is exactly one period and
≤ 240 probes/hour holds deterministically (×U(0.9, 1.1) per tick made it a coin flip). Groups are requested per cadence group (`sections(frozenset({group}), ctx)` once per due group), so a failed
section maps to its group's backoff. A manual refresh (`r`) forces only the fast groups (period ≤ 2 ticks), not
history/nodes. At each tick the poller collects every group that is due (interval/piggyback: period
elapsed at this tick ± half a tick; event: requested since the last tick; a collector follow-up may also name an
interval group, which is then due at the next tick too — `slurm.nodes` when the partition set is first known or changes,
`slurm.usage` when an own job starts running, so neither waits a full 60–120 s period), asks each collector for `sections(due)`,
renders **one** `ProbeScript`, and runs it on the poll lane after `check_master()` (mux hosts; failing → ensure master,
skip the tick, `CONNECTING`). Groups are never evaluated between ticks, so a Slurm host makes exactly one round trip
per tick; `wake()` (manual `r`, post-action) serves an immediate tick no sooner than 3 s after the previous probe.
**Routing**: each `SectionResult` goes to the collector named by `Section.owner`; every collector in the profile is
ingested on every tick (empty results map if none of its sections ran); monitors ingest first, the scheduler last with
the host's persistent `MonitorView` (latest successful monitor state + sample times, §3.6, updated from this tick's
monitor outputs before the scheduler runs); outputs merge per the §3.6 rule into `store.apply()`; `HostStatus` is
updated.

**Idle and focus**: no key/mouse input for `idle_after` (10 min) or terminal focus lost (`AppBlur`) → multiplier ×4;
no input for `idle_pause_after` (2 h) → **all polling paused** (`store.idle_paused`, header chip "paused (idle) — any
key resumes"); any input → ×1, resume, immediate refresh of visible hosts. **Second instance**: `fcntl.flock` on
one lock per enabled host's ssh destination, `<mux_dir>/instance.lock.<alias>-<sha1 8>` (pid inside); for the hosts whose
lock is held, toast once "another RunFleet (pid N) is polling k2, lyra — remote load doubles there" (two instances with
disjoint host sets do not warn).

Default load per host (steady state, visible):

| Host type | SSH round trips/min | Remote commands/min | Notes |
|---|---|---|---|
| Slurm login node | 4 (test-enforced: ≤ 240 per simulated hour with no wakes) | squeue 4, sstat ≤ 1, sacct 0.5, sinfo 0.5, stat 0.5, scontrol ≈ 0, sprio + sshare ≤ 0.2 (only while a job of yours pends, §5.2.10) ⇒ **≤ 7 controller/dbd RPCs/min** | ≈ `watch -n 15 squeue --me`; floors: squeue 10 s, sacct 60 s, sprio/sshare 600 s |
| Slurm, hidden | 1.3 | ÷3 | |
| Workstation | 12 (focused 20) | nvidia-smi 2 + ps 0.5–1 + procfs 1 per tick | ≈ 0.04 CPU-core on labstation |
| Any host, Search logs… (§4.7.1) | +1 per search the user starts (none otherwise) | `stat` + `tail`/`head`/`grep` over ≤ 20 files, ≤ 1 MiB read, no Slurm RPC | on demand only, never repeated by a timer |

**Busy controller** (`ControllerBackoff` in the poller): after every tick the controller sections `slurm.squeue`,
`slurm.sacct`, `slurm.sacct_ids` and `slurm.sinfo` are checked. Stderr matching `Socket timed out|Rate limit|
slurm_load_jobs error|Transport endpoint`, or a duration (the section's remote `ms`) above 3× its rolling median (last
20, at least 3 samples) and above 5 s, doubles the host's Slurm factor (×2 … ×8); `BUSY_DECAY` = 3 ticks whose
controller sections are healthy (not busy, at least one answered: a round of failures is no recovery) halve it. The factor multiplies the effective period of every `slurm.*` cadence group
after the floor (so the tick follows), is published as `HostStatus.controller_slowdown` and shown as `controller busy
— slowed 4×` on the host card (instead of the login load), in host detail and in the host error screen. A plain
section failure or timeout stays with the per-group backoff (§9 row 10); no extra remote command is involved.

The Fleet view, roomy or compact (§7.4), adds no section and changes no cadence: it reports `Visibility.VISIBLE` like
before and only reads the store (every user's node occupants stay host-detail-only, `focused_only`).

### 5.5 Caching (`core/cache.py`)
SQLite at `~/.local/state/runfleet/cache.sqlite` (WAL; writes batched every 10 s and at exit via `asyncio.to_thread`;
`--no-cache` disables):

```sql
CREATE TABLE snapshots   (host TEXT PRIMARY KEY, taken_at REAL, json TEXT);          -- warm start (shown as stale)
CREATE TABLE caps        (host TEXT PRIMARY KEY, detected_at REAL, json TEXT);       -- profile + capabilities, 24 h
CREATE TABLE jobmeta     (host TEXT, job_id TEXT, job_id_raw TEXT, name TEXT, workdir TEXT, stdout TEXT, stderr TEXT,
                          merged INTEGER, gpu_type TEXT, per_node_json TEXT, first_seen REAL, last_seen REAL,
                          last_state TEXT, PRIMARY KEY (host, job_id));               -- survives scontrol's MinJobAge
CREATE TABLE ended_tasks (host TEXT, job_id TEXT, ended_at REAL, json TEXT,          -- job_id = "p<pid>-<start>"
                          PRIMARY KEY (host, job_id));
CREATE TABLE host_status (host TEXT PRIMARY KEY, at REAL, state TEXT,                -- for `runfleet status` only
                          error_kind TEXT, error TEXT);
```
`host_status` holds each host's `ConnState` since `at` (with the error's kind and one-liner): FleetService subscribes
to the store and queues a row when (state, error kind, error summary) changes — never NEW / CONNECTING, which are no
observation — so the 10 s batch carries at most one row per host; nothing reads it back into the TUI. Caches of
RunFleet ≤ 0.5.0 lack the table (`CREATE TABLE IF NOT EXISTS` adds it at the next start).
Retention: jobmeta 30 days, ended_tasks 7 days, pruned at startup. Serialization via `models.to_json`/`from_json`.
A failed batch write (`sqlite3.Error`: disk full, read-only filesystem) keeps the batch pending for the next flush,
newer entries winning: ended tasks and jobmeta are queued only on change and would otherwise be lost for good.

---

## 6. Actions (`core/actions.py` + `backends/*/actions.py`)

### 6.1 Catalogue (v0.1)

| Action id | Applies to | Key | Remote command (one probe per host) |
|---|---|---|---|
| `slurm.cancel` | own PD, CF, RQ, R, CG, S (incl. array summaries) | `ctrl+k`, menu | `scancel <ids>`; an array summary whose task list Slurm abbreviated (step range `1-99:2`, truncated `…`) → `scancel --state=PENDING <array job id>` (only the pending tasks the row stands for) |
| `slurm.hold` | own PD, not held; not an abbreviated array summary | menu | `scontrol hold <ids>` |
| `slurm.release` | own held | menu | `scontrol release <ids>` |
| `slurm.requeue` | own R/S; terminal jobs only if ended < `MinJobAge` (else disabled: "no longer in controller memory — resubmit") | menu | `scontrol requeue <ids>` (param `hold` → `scontrol requeuehold`) |
| `task.terminate` | own ACTIVE tasks | `ctrl+k`, menu | §6.2 with TERM, then KILL after `grace_s` (default `kill_grace`, 10 s) if `escalate` (default on) |
| `task.interrupt` | own ACTIVE tasks | menu | §6.2 with INT, no escalation (like Ctrl-C) |
| `task.kill` | own ACTIVE tasks | menu | §6.2 with KILL |

All are `mutating=True, confirm=True`. Ids are the display ids with array throttles stripped (§5.2.3); a command
builder's ValueError becomes an `ActionError` (never a UI crash). Interpretation: per-id error lines map to their
target (array per-task lines `500_7: …` to the summary target); id-less summary lines (`slurm_suspend error:…` after
each scontrol per-id failure) do not fail the other targets. Under
`--readonly` / `general.readonly` they stay listed in the menu but disabled ("read-only mode") and are hidden from the
hint bar. **Protected jobs**: per-host `protected_partitions` / `protected_qos` (k2 `["labpi"]`, the PI's queue)
disable every Slurm action on matching jobs with the reason shown. Other users' jobs/processes never get actions
("owned by <user>"). FleetAPI.actions_for adds host-level reasons (host not OK/DEGRADED, stale data).

Not actions (no probe, no confirm; UI keys calling FleetAPI argv/log methods): shell `s`, pager `p`, log viewer `l`,
copy `y`/`Y` (`App.copy_to_clipboard`, OSC 52; tmux needs `set -g set-clipboard on`).

### 6.2 Task signal script (remote re-enumeration, pid-reuse safe)
`plan_action("task.*")` renders one `process.kill.<root pid>` section per task (unique in the probe; owner `process`). It receives only the root `(pid, start_epoch)`
from the snapshot; the member list is **re-enumerated on the host at execution time**, because the user's launcher
(`python -m orb.launch --manifest … --per-gpu 1`, root on lynx and labstation) is a queue runner that starts the next
manifest item when a child dies, and a 10 s-old snapshot would miss it.

```sh
R=642229 RS=1788696459 SIG=TERM G=10 ESC=1            # from meta: root pid, root start epoch, signal, grace, escalate
tree() {   # current descendants of $1, one per line (one ps snapshot, BFS)
  ps -e -o pid=,ppid= | awk -v r="$1" '{par[$1]=$2} END {q[1]=r; n=1; seen[r]=1
    for (i = 1; i <= n; i++) for (p in par) if (par[p] == q[i] && !(p in seen)) {seen[p]=1; q[++n]=p; print p}}'
}
st() { awk '{sub(/.*\) /, ""); print $20}' "/proc/$1/stat" 2>/dev/null; }   # start time (stat field 22)
ok() { case " $IDS " in *" $1:$(st "$1") "*) ;; *) return 1 ;; esac; }     # still the process recorded frozen
all() { L=; for p in $R $SET; do ok "$p" && L="$L $p"; done; ok "$R" && L="$L $(tree "$R")"
  printf '%s\n' $L | sort -un; }                      # root + known members + current descendants, reused pids out
et=$(ps -o etimes= -p "$R" 2>/dev/null | tr -d ' ')
[ -n "$et" ] || { echo "gone $R"; echo done; exit 0; }
d=$(( $(date +%s) - et - RS )); [ "${d#-}" -le 5 ] || { echo "reused $R"; echo done; exit 0; }
FZ=; trap '[ -z "$FZ" ] || kill -s CONT "$R" 2>/dev/null' EXIT; trap 'exit 129' HUP; trap 'exit 130' INT
trap 'exit 143' TERM; trap '' PIPE                    # never leave the root stopped; a dropped channel cannot stop us
case "$(ps -o stat= -p "$R" 2>/dev/null)" in          # freeze the launcher (it cannot start the next item) — unless
  *+*) ;; *) kill -s STOP "$R" 2>/dev/null && FZ=1 ;; esac   # it is a tty's foreground job: bash would print
                                                      # "Stopped", take the terminal, and a survivor would run on detached
SET=$(tree "$R" | tr '\n' ' ')
IDS=$(for p in $R $SET; do printf '%s:%s ' "$p" "$(st "$p")"; done)   # pid:starttime of every member, while frozen
PG=$(ps -o pgid= -p "$R" | tr -d ' ')                 # also signal the process group if it holds nothing else
OUT=$(ps -e -o pid=,pgid= | awk -v g="$PG" -v s=" $R $SET " '$2 == g && index(s, " " $1 " ") == 0 {print $1}')
GRP=; [ -n "$PG" ] && [ -z "$OUT" ] && GRP=$PG
M=; for p in $SET $R; do ok "$p" && kill -s "$SIG" "$p" 2>/dev/null && M="$M $p"; done
[ -n "$GRP" ] && kill -s "$SIG" -- "-$GRP" 2>/dev/null
[ -n "$FZ" ] && kill -s CONT "$R" 2>/dev/null; FZ=    # a stopped process acts on TERM/INT only once continued
echo "members $R $SET"; for p in $M; do echo "sent $p"; done   # nothing is written while the root is stopped
i=0; while [ "$i" -lt "$G" ]; do
  A=; for p in $(all); do kill -0 "$p" 2>/dev/null && A="$A $p"; done
  [ -z "$A" ] && break; sleep 1; i=$((i + 1))
done
if [ "$ESC" = 1 ]; then
  for p in $(all); do kill -0 "$p" 2>/dev/null && kill -s KILL "$p" 2>/dev/null && echo "killed $p"; done
  [ -n "$GRP" ] && { ! kill -0 "$GRP" 2>/dev/null || ok "$GRP"; } && kill -s KILL -- "-$GRP" 2>/dev/null; sleep 1
fi                                                    # (the group only while its leader is gone or still the same)
for p in $R $SET; do ok "$p" && kill -0 "$p" 2>/dev/null && echo "alive $p"; done
if ok "$R"; then for p in $(tree "$R"); do case " $SET " in *" $p "*) ;; *) echo "new $p" ;; esac; done; fi
echo done
```
`task.terminate`: SIG=TERM, ESC=1 (param `escalate`), G=`grace_s`. `task.interrupt`: SIG=INT, ESC=0. `task.kill`:
SIG=KILL, G=0, ESC=1. Section timeout = G + 15 s. Interpretation per task: `gone` → ok "already exited"; `reused` →
not ok "pid reused — refresh"; any `alive` or `new` line → not ok ("2 processes survived", "1 process started during
termination: 790001"); else ok. All lines go to `ActionResult.details`.

Pid reuse: every member's identity (`pid:starttime`, `/proc/<pid>/stat` field 22) is recorded while the root is frozen;
`ok` rejects a pid whose start time has changed since, so a pid recycled during the grace period is never signalled,
never escalated to SIGKILL and never reported `alive` (the group kill only runs while its leader is gone or unchanged).
Traps: the EXIT trap continues the root if the script dies while it is stopped (HUP/INT/TERM exit through it), and
SIGPIPE is ignored so a dropped ssh channel cannot cut the escalation short; nothing is written while the root is
stopped. The text above is `SIGNAL_SCRIPT` in `backends/process/actions.py` with the first line rendered and comments
added; `tests/integration/test_signal_script.py` runs the real one.

### 6.3 Flow and reporting
1. UI asks `service.actions_for(marked_or_cursor_keys)` → `ActionSpec`s; the action menu (`x`) lists enabled ones
   first and disabled ones greyed with their reason.
2. `service.plan(...)` → one `ActionPlan` per host. **ConfirmScreen** shows title, targets (≤ 8 listed, "+N more"),
   skipped targets with reasons, the exact `preview`, params (escalate checkbox, grace seconds), and *Cancel* (default
   focus) / *Confirm* (`y`). Changing a param calls `service.plan(...)` again (pure, synchronous) and replaces the
   preview. Marked jobs on several hosts → one dialog with per-host groups.
3. `service.execute(plans)` → per host one `probe(purpose="action")` on the on-demand lane → `interpret_action` →
   `ActionResult`s. Slurm stderr `scancel: error: Kill job error on job id <id>: <reason>` maps to that id;
   `Invalid job id specified` → "job already ended".
4. Reporting: success toast ("Cancelled 3 jobs on k2", 4 s); any failure → sticky error toast (10 s) with the first
   stderr/detail line and "details: palette → Action log". The action log keeps the last 200 results with full output
   for the session; every executed action is also written to `runfleet.log`.
5. Affected hosts get `poller.wake("action", {"slurm.jobs", "slurm.history"})` or `{"process.ps"}` (3 s floor), so the
   table reflects the change within ~3 s.

---

## 7. User interface

### 7.1 Principles
- **k9s × lazygit × nvitop.** k9s for scope (`0` all hosts, `1–9` one host) and `/` filter; lazygit for the
  selection-following preview, contextual help and confirmation popups; nvitop for GPU bars, colour thresholds and the
  device/host/process bands of the host screen. Commands beyond the keymap live in Textual's command palette.
- **Everything is a table you can filter, sort and click.** Every row has a stable key (`JobKey`, GPU key, host name);
  refreshes are keyed diffs so the cursor never jumps.
- **Never blank, never crash.** A failing host keeps its last data, dimmed, with an age and a reason; one place
  (header + card/host screen) explains what is wrong and which key fixes it.
- **Screen real estate:** designed at 120×35, fully usable at 80×24, graceful below (≥ 60×16, else a "terminal too
  small" placeholder).

### 7.2 Screens (all v0.1)

| Screen / view | Module | Entered by | Content |
|---|---|---|---|
| MainScreen › **Fleet view** (default, `default_view`) | `screens/main.py`, `widgets/fleet_view.py` | start, `tab` | host cards (≥ 100 cols) or host table (< 100 cols) — the PRD's "immediately see an overview" |
| MainScreen › Jobs view | `widgets/jobs_view.py` | `tab` | merged job table + selection-following preview |
| MainScreen › GPUs view | `widgets/gpus_view.py` | `tab` | every workstation GPU + one row per running Slurm job with GPUs |
| JobDetailScreen | `screens/job_detail.py` | `enter`/double-click on a job, palette | tabs Summary (job + resources + GPU + log follow) / GPU (usage, per-node indices; running Slurm job: Attach live panel, §7.16) / Logs / Steps / Procs (task members; running Slurm job: Attach process list with STEP column) / Raw |
| HostDetailScreen | `screens/host_detail.py` | `enter` on a host card/row, palette | workstation: GPUs + host + processes; cluster: partitions + nodes + my jobs + capabilities |
| LogScreen | `screens/log_viewer.py` | `l`, a log search hit | full-screen log, follow, live line, stream switch; from a hit: streamed from its window, landed on its line |
| HelpScreen (modal) | `screens/help.py` | `?`, `F1` | contextual, filterable key list generated from `keymap.py` |
| Command palette (modal) | Textual built-in + `ui/commands.py` Provider | `ctrl+p`, `:` | fuzzy RunFleet commands (§7.10) |
| ConfirmScreen (modal) | `screens/confirm.py` | every action | title, targets, skipped, exact command, params |
| ActionMenuScreen (modal) | `screens/action_menu.py` | `x`, right-click | `OptionList` of `actions_for()`, disabled ones greyed with reason; optional single-key accelerators (copy menu) |
| HostErrorScreen (modal) | `screens/host_error.py` | `e`, click on a red chip | classified error, stderr excerpt, ssh log tail, next retry; Connect / Retry now / Disable host |
| AttentionScreen (modal) | `screens/attention.py` | `a` in the Fleet view | "needs you": the actionable entries of every card's ranking, fleet-wide, in rank order; `enter` opens the job (§7.4) |
| SaveViewScreen (modal) | `screens/save_view.py` | palette › Save view as… | one-line name prompt for a saved Jobs view (keymap context `prompt`); shows what is saved and why a name cannot be used (§7.6) |
| WatchPromptScreen (modal) | `screens/gpu_watch.py` | palette › Watch free GPUs…, `n` in the Watches list | one-line prompt for a GPU availability watch (keymap context `prompt`): how the text was read, what matches now, why it cannot be saved (§7.14) |
| WatchesScreen (modal) | `screens/gpu_watch.py` | palette › Watches | the GPU watches with what each is doing (waiting, holds 1 of 2, fired, quiet until, paused); `d` removes, `n` adds (§7.14) |
| SearchPromptScreen (modal) | `screens/log_search.py` | palette › Search logs…, `ctrl+f` in the Jobs view, `/` in the results | pattern, regex / match-case options, what would be read per host (keymap context `search_prompt`, §4.7.1) |
| LogSearchScreen (modal) | `screens/log_search.py` | the prompt's `enter` | what was read and every cap; matches per job file with the excerpt highlighted; `enter` pushes LogScreen above it, landed on the hit (keymap context `log_hits`, §4.7.1) |
| SettingsScreen (modal) | `screens/settings.py` | `,`, header `⚙`, palette › Settings, start with a config without hosts | herdr-style panel: Hosts (save / remove / test / add / reorder), General, About (§7.13) |

### 7.3 Mockups (120×35 unless stated)

Glyph legend: `●` ok, `◐` retrying, `✖` unreachable/auth, `○` disabled, `◌` connecting; row markers `▸` cursor,
`+` marked, `!` host has a problem, `~` data is stale. Colours are in §7.11. Data values are illustrative unless a test
in §10.2 pins them.

**A. Fleet view (default start screen)** — card grid, two cards per row (each half of the width left of the
reserved scrollbar column), cards sized to content (Slurm 6 lines + extra free-GPU lines, workstation #GPUs + 3 lines,
capped at 8 GPUs then "+N more"). Free-GPU counts include nodes planned for pending jobs (vega `mixed-`), marked in the
host screen. This fleet stays compact at 120×35 (tier T0, §7.4): lynx's cached card takes 5 lines, which leaves 8 spare
rows, one short of the roomy gate; the demo fleet (lynx without a cache) is roomy at this size, rows [10, 10, 6].

```
 RunFleet ▏ALL▕  ● k2 8s  ● vega 3s  ◐ lyra retry 38s  ● labstation 2s  ✖ lynx auth       R 10  PD 3  F 2/24h   ⟳ 5s  ⚙
  Fleet   Jobs   GPUs                                                                                  scope: all hosts
┏━ 1 k2 · slurm 25.05.8 · ● ok 0.21s · 8s ago ━━━━━━━━━━━━┓┌─ 2 vega · slurm 25.11.4 · ● ok 0.20s · 3s ago ───────────┐
┃ my jobs   R 4   PD 2   F 0   CD 7            (last 24h) ┃│ my jobs   R 4   PD 1   F 0   CD 3            (last 24h)  │
┃ my GPUs   5   A100-40G ×3 · H200 ×2                     ┃│ my GPUs   6   H200 ×6                                    │
┃ job util  ▕████████████████████▌ ▏  93%   mem 166/401G  ┃│ job util  ▕██████████████████▌   ▏  84%   mem 332/846G   │
┃ free GPUs gpu-preempt 31/164 (4 planned) · labpi 2/16   ┃│ free GPUs h200 2/40 (2 planned) · h100 0/9 · a30 3/24    │
┃ next      PD smoke-test · priority                      ┃│ next      PD lmft-x07-trainer-s0 · per-user job cap (Q…  │
┃ login     login02  load 2.3/32                          ┃│ login     login-v01  load 5.1/64                         │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛└──────────────────────────────────────────────────────────┘
┌─ 3 lyra · slurm 26.05.3 · ◐ retrying · last ok 4m ago ──┐┌─ 4 labstation · workstation · ● ok 0.08s · 2s ago ───────┐
│ my jobs   R 0   PD 0   F 2   CD 3            (last 24h) ││ 0 A6000     ▕        ▏   0%  ▕        ▏  0.0/48.0G 30°C  │
│ my GPUs   0                                             ││ 1 A6000     ▕        ▏   0%  ▕▍       ▏  2.1/48.0G 31°C  │
│ job util  n/a — no GPU accounting on this cluster       ││ 2 A6000     ▕        ▏   0%  ▕        ▏  0.0/48.0G 29°C  │
│ free GPUs free-gpu 9/120 · free-gpuxl 4/36              ││ 3 A6000     ▕█████▋  ▏  70%  ▕███▌    ▏ 21.0/48.0G 78°C  │
│ ! ssh: Connection timed out during banner exchange      ││ cpu ▕▏   ▏  3%  mem ▕▌     ▏  9% of 503G  load 2.1/64    │
│   (via jump host) · next retry in 38s · e info          ││ tasks  mine 1 (orb.launch, 1 GPU) · others 1             │
└─────────────────────────────────────────────────────────┘└──────────────────────────────────────────────────────────┘
┌─ 5 lynx · workstation · ✖ needs auth · last ok 14m ago ─┐
│ 0 PRO6000   ▕███████▊▏  97%  ▕█▎      ▏ 14.9/95.6G 55°C │
│ 1 PRO6000   ▕████████▏ 100%  ▕█▎      ▏ 14.9/95.6G 57°C │
│ tasks  mine 1 (orb.launch ×2, 2 GPUs) · others 0        │
│ ✖ Permission denied (publickey,keyboard-interactive)    │
│   press c to log in (Duo / password prompts appear)     │
└─────────────────────────────────────────────────────────┘









 ↵ host detail  space scope  c connect  s shell  t on/off  e info  tab view  ^p cmd  q quit  / filter            ? help
```

**A2. Fleet view, roomy cards (172×62, demo seed 7, tier T1, §7.4)** — the same cards grow into the tall terminal:
each keeps its summary in fixed slots, then lists the jobs that need you (fresh failures, stalled or short on time,
never-run, then the rest by time left), one row per job; workstation GPU lines say who holds the GPU; the empty cell
after the last card is the fleet panel (free GPUs by type, your types first; what changed since your last visit).

```
 RunFleet ▏ALL▕  ● k2 10s  ● vega 10s  ● lyra 10s  ● labstation 0s  ✖ lynx auth                                                               R 18  PD 5  F 6/24h   ⟳ 5s  ⚙
  Fleet   Jobs   GPUs                                                                                                                                      scope: all hosts
┏━ 1 k2 [clusters] · slurm 25.05.8 · ● ok 0.00s · 10s ago ━━━━━━━━━━━━━━━━━━━━━━━━━━┓┌─ 2 vega [clusters] · slurm 25.11.4 · ● ok 0.00s · 10s ago ─────────────────────────┐
┃ my jobs   R 7   PD 2   F 3   CD 3                                      (last 24h) ┃│ my jobs   R 4   PD 2   F 2   CD 1                                      (last 24h)  │
┃ my GPUs   15   A100-80G ×8 · A100-40G ×3 · 3090 ×2 · H200 ×2                      ┃│ my GPUs   6   H200 ×6                                                              │
┃ job util  waiting for sstat                                                       ┃│ job util  waiting for sstat                                                        │
┃ free GPUs gpu-preempt 12/108 +11 MIG (14 planned) · lab 0/8                       ┃│ free GPUs a30 1/8 (1 planned) · h100 0/7 · h200 6/48 (5 planned)                   │
┃ login     login02  load 2.3/32                                                    ┃│ login     login-v01  load 5.1/64                                                   │
┃ failed    F   lmft-x0-lr1e-5             exit 1                           3h ago  ┃│ failed    F   lmft-x07-sampler-s0        exit 1                           5h ago   │
┃           TO  orb-sw…imagenette-w…2--a01 hit time limit                   4h ago  ┃│           F   lmft-x07-ddp-s0            exit 1                           6h ago   │
┃           OOM lmft-x07-trainer-s2        out of memory                    6h ago  ┃│ running   R   lmft-x07-trainer-s1        2×H200                    –  1d15h left   │
┃ running   R   ◉ orb-sw…eval--cifar…--s43 1×3090                    –    34m left  ┃│           R   orb-swp--train--v…s47--a01 1×H200                    –  1d17h left   │
┃           R   ↻1 orb-s…voc-warmup-…--a01 1×3090                    – 20h39m left  ┃│           R   orb-swp--train--v…s48--a01 1×H200                    –  1d17h left   │
┃           R   ddp-scale--imagen…it-b--n2 8×A100                    – 22h47m left  ┃│           R   lmft-x07-sampler-s1        2×H200                    –  1d18h left   │
┃           R   lmft-x07-sampler-s0        2×H200                    –  1d14h left  ┃│ pending   PD  lmft-x07-eval-s0           never: dependency failed        no est.   │
┃           R   orb-swp--…warmup--s43--a02 1×A100                    –  2d19h left  ┃│           PD  lmft-x07-trainer-s0        per-user job cap (QOS)          overdue   │
┃           R   orb-swp--t…const--s43--a02 1×A100                    –  2d19h left  ┃│                                                                                    │
┃           R   orb-s…cosinewarm--s43--a02 1×A100                    –  2d19h left  ┃│                                                                                    │
┃ pending   PD  smoke-test                 priority                         in 34m  ┃│                                                                                    │
┃           PD  orb-sweep-lab--s44         resources                       no est.  ┃│                                                                                    │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛└────────────────────────────────────────────────────────────────────────────────────┘
┌─ 3 lyra [clusters] · slurm 26.05.3 · ● ok 0.00s · 10s ago ────────────────────────┐┌─ 4 labstation [workstations] · workstation · ● ok 0.00s · 0s ago ──────────────────┐
│ my jobs   R 5   PD 1   F 1   CD 2                                      (last 24h) ││ 0 A6000     ▕         ▏   0%  ▕         ▏  0.0/48.0G 30°C  free                    │
│ my GPUs   5   A30 ×3 · RTX6000 ×2                                                 ││ 1 A6000     ▕████████▌▏  94%  ▕█████▋   ▏ 30.0/48.0G 73°C  carol sample_ddpm.py    │
│ job util  n/a — no GPU accounting on this cluster                                 ││ 2 A6000     ▕███▋     ▏  40%  ▕█▏       ▏  6.0/48.0G 47°C  mine eval_ckpt.py       │
│ free GPUs free-gpu 6/48 · free-gpuxl 15/32                                        ││ 3 A6000     ▕██████▏  ▏  68%  ▕███▉     ▏ 21.0/48.0G 63°C  mine orb.launch         │
│ login     login-a02  load 11.4/48                                                 ││ cpu ▕▎   ▏  6%  mem ▕█▏        ▏ 11% of 504G  load 3.8/64                          │
│ failed    PR  orb-s…cosinewarm--s44--a01 preempted, killed               17h ago  ││           swap 2% of 2.0G · up 117d · load 3.8 3.7 3.9                             │
│ running   R   orb-sw…eval--places…p--s45 1×A30                     –    12m left  ││ running   R   orb.launch (orb.train)     gpu 3 · cpu 101%     ▆  68%   18h54m up   │
│           R   orb-swp--…warmup--s45--a01 1×A30                     –  2d21h left  ││           R   eval_ckpt.py               gpu 2 · cpu 94%      ▄  40%      12m up   │
│           R   orb-swp--tr…adam--s46--a01 1×A30                     –  2d23h left  ││ others    R   carol sample_ddpm.py       gpu 1 · 30.0G        █  94%    5h20m up   │
│           R   orb-sweep [3]              1×RTX6000                 –  2d23h left  ││                                                                                    │
│           R   orb-sweep [4]              1×RTX6000                 –  2d23h left  ││                                                                                    │
│ pending   PD  orb-sweep [5-12%2]         array task limit                overdue  ││                                                                                    │
│                                                                                   ││                                                                                    │
│                                                                                   ││                                                                                    │
│                                                                                   ││                                                                                    │
│                                                                                   ││                                                                                    │
│                                                                                   ││                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘└────────────────────────────────────────────────────────────────────────────────────┘
┌─ 5 lynx [workstations] · ✖ needs auth ────────────────────────────────────────────┐┌─ fleet · free GPUs · what changed ─────────────────────────────────────────────────┐
│ no data yet                                                                       ││ free GPUs now · your types first  lynx ?                            sinfo <1m ago  │
│ ✖ permission denied (publickey,keyboard-interactive)                              ││   3090        4   k2 gpu-preempt 4 (2 planned)                                     │
│   press c to log in (Duo / password prompts appear)                               ││   H200        9   vega h200 6 (5 planned) · k2 gpu-preempt 3                       │
│                                                                                   ││   RTX6000    11   lyra free-gpuxl 11                                               │
│                                                                                   ││   A30         6   lyra free-gpu 5 · vega a30 1 (1 planned)                         │
│                                                                                   ││   H100-80G    0                                                                    │
│                                                                                   ││   A100-80G    1   k2 gpu-preempt 1 (1 planned)                                     │
│                                                                                   ││   A100-40G    0                                                                    │
│                                                                                   ││   A6000       1   labstation gpu 0                                                 │
│                                                                                   ││   also free   MIG slices 11 (k2) · L40S 4 · PRO6000 2 · P100-16G 1 · V100 1 · V1…  │
│                                                                                   ││ since 02:42 (you were away): 3 failed · 1 out of memory · 1 timeout · 3 completed  │
│                                                                                   ││   10:21  lyra ✓ orb-sweep [2]                ran 0:41:00                           │
│                                                                                   ││   09:40  lyra ✓ orb-sweep [1]                ran 0:38:00                           │
│                                                                                   ││   07:41  k2   ✖ lmft-x0-lr1e-5               exit 1                                │
│                                                                                   ││   06:55  k2   ✓ orb-ab--train--resnet18-bs5… ran 11:24:33                          │
│                                                                                   ││   05:42  k2   ✖ orb-swp--train--imagenette-… hit time limit                        │
│                                                                                   ││   04:55  vega ✖ lmft-x07-sampler-s0          exit 1                                │
│                                                                                   ││   04:23  vega ✖ lmft-x07-ddp-s0              exit 1                                │
│                                                                                   ││   03:43  k2   ✖ lmft-x07-trainer-s2          out of memory                         │
└───────────────────────────────────────────────────────────────────────────────────┘└────────────────────────────────────────────────────────────────────────────────────┘
 ↵ host detail  space scope  c connect  s shell  t on/off  e info  tab view  ^p cmd  q quit  / filter                                                                ? help
```

**B. Fleet view at 80×24** — one row per host.

```
 RunFleet ALL │ ●k2 ●vega ◐lyra ●labstation ✖lynx │ R13 PD2 F3        ⟳15s
 ▌Fleet▐ Jobs  GPUs
   # HOST        KIND   STATE     JOBS        GPUS  UTIL                 AGE
 ▸ 1 k2          slurm  ● ok      R4 PD1      5     ▕████████▉▏ 94%      8s
   2 vega        slurm  ● ok      R4 PD1      6     ▕████████▍▏ 88%      3s
   3 lyra        slurm  ◐ retry   F1          0     n/a                  4m
   4 labstation  ws     ● ok      1 task      1/4   ▕███████  ▏ 70%      2s
   5 lynx        ws     ✖ auth    1 task      2/2   ▕█████████▉ 99%     14m














 ↵ detail  c connect  tab view  0-9 scope  ? help  q quit
```

**C. Jobs view.** Preview pane at the bottom (width < 150); at ≥ 150 columns it sits on the right. The preview's log
lines come from a cached one-shot peek after 1 s of dwell (§4.7), not a stream.

```
 RunFleet ▏ALL▕  ● k2 8s  ● vega 3s  ◐ lyra retry 38s  ● labstation 2s  ✖ lynx auth     R 13  PD 2  F 3/24h   ⟳ 15s
  Fleet ▌Jobs▐  GPUs     state: fresh (active + ended < 24h)   sort: host ▲, state   group: none          18 of 41 jobs
   HOST       ID       NAME                                 ST  ELAPSED  LEFT      GPUS      GPU%   NODE / REASON
 ▸ k2         2332861  orb-swp--train--cifar-warmup--s43--… R   4:07:12  2-19:52   1×A100    ▇ 98%  n-40-01
   k2         2332862  orb-swp--train--cifar-const--s43--a… R   4:07:12  2-19:52   1×A100    ▇ 97%  n-40-01
   k2         2332863  orb-swp--train--cifar-cosinewarm--s… R   4:07:12  2-19:52   1×A100    ▇ 98%  n-40-01
   k2         2333760  lmft-x07-sampler-s0                  R   9:02:26  1-14:57   2×H200    ▆ 86%  n-38-01
   k2         2334564  smoke-test                           PD  —        1:00:00   1×3090    –      priority · est 06:31
   vega       723954   lmft-x07-sampler-s1                  R   5:54:27  1-18:05   2×H200    ▇ 85%  n-38-01
   vega       723952   lmft-x07-trainer-s1                  R   8:26:24  1-15:33   2×H200    ▇ 91%  n-38-12
   vega       724246   orb-swp--train--voc-cosinewarm--s48… R   6:04:44  1-17:55   1×H200    ▅ 71%  n-38-06
   vega       724245   orb-swp--train--voc-cosinewarm--s47… R   6:05:44  1-17:54   1×H200    ▆ 80%  n-37-12
   vega       724479   lmft-x07-trainer-s0                  PD  —        2-00:00   2×H200    –      per-user job cap (Q…
 ~ lyra       31354789 orb-swp--train--places-cosinewarm--… PR  0:01:04  ended 18h 1×RTX6000 –      preempted, killed
 ~ lyra       31327454 orb-swp--train--warmup--s49--a01     F   0:02:30  ended 2d  1×RTX6000 –      exit 2 · r71-00
   labstation p363942  orb.launch (orb.train)               R   18:54:04 —         1×A6000   ▅ 70%  gpu 3
 ! lynx       p642229  orb.launch ×2 (orb.train)            R   18:45:12 —         2×PRO6000 ▇ 98%  gpu 0,1 · auth
─ k2:2332861 · orb-swp--train--cifar-warmup--s43--a02 ───────────────────────────────── RUNNING · 4:07:12 / 3-00:00:00 ─
 partition gpu-preempt · qos gpu-priority · account lab · node n-40-01 · 8 cpu · 48G · A100-SXM4-40GB idx 2
 GPU (sstat · 30 s)  util ▕█████████▊▏ 98%   mem ▕██████    ▏ 24.3/40.0G     stdout …/logs/…warmup--s43--a02_2332861.out
 │ [2026-09-07 05:56:18] epoch 41/300  train_loss 0.3121  vs_base -2.41%  213 it/s
 │ [2026-09-07 05:56:39] epoch 42/300  train_loss 0.3098  vs_base -2.44%  211 it/s
 │ [2026-09-07 05:57:01] epoch 43/300  train_loss 0.3087  vs_base -2.47%  212 it/s
 │ [2026-09-07 05:57:22] epoch 44/300  train_loss 0.3071  vs_base -2.50%  210 it/s                  log 663 B · 12s ago




 ↵ detail  l logs  x actions  ^k cancel  space mark  f state  o sort  z group  / filter  0-9 host  ? help
```

**D. Jobs view at 80×24** — LEFT, NODE/REASON and PART columns dropped; preview shrinks to a 2-line summary.

```
 RunFleet ALL │ ●k2 ●vega ◐lyra ●labstation ✖lynx │ R13 PD2 F3        ⟳15s
  Fleet ▌Jobs▐ GPUs   fresh · host▲ · 18/41
   HOST      ID       NAME                       ST  ELAPSED  GPUS      GPU%
 ▸ k2        2332861  orb-swp--train--cifar-war… R   4:07:12  1×A100    ▇ 98%
   k2        2332862  orb-swp--train--cifar-con… R   4:07:12  1×A100    ▇ 97%
   k2        2332863  orb-swp--train--cifar-cos… R   4:07:12  1×A100    ▇ 98%
   k2        2333760  lmft-x07-sampler-s0        R   9:02:26  2×H200    ▆ 86%
   k2        2334564  smoke-test                 PD  —        1×3090    –
   vega      723954   lmft-x07-sampler-s1        R   5:54:27  2×H200    ▇ 85%
   vega      723952   lmft-x07-trainer-s1        R   8:26:24  2×H200    ▇ 91%
   vega      724246   orb-swp--train--voc-cosin… R   6:04:44  1×H200    ▅ 71%
   vega      724245   orb-swp--train--voc-cosin… R   6:05:44  1×H200    ▆ 80%
   vega      724479   lmft-x07-trainer-s0        PD  —        2×H200    –
 ~ lyra      31354789 orb-swp--train--places-co… PR  0:01:04  1×RTX6000 –
 ~ lyra      31327454 orb-swp--train--warmup--s… F   0:02:30  1×RTX6000 –
   labstati… p363942  orb.launch (orb.train)     R   18:54:04 1×A6000   ▅ 70%
 ! lynx      p642229  orb.launch ×2 (orb.train)  R   18:45:12 2×PRO6000 ▇ 98%
─ k2:2332861 R 4:07:12 · n-40-01 · A100 idx 2 · GPU 98% · 24.3/40G ───────────
 │ epoch 44/300  train_loss 0.3071  vs_base -2.50%  210 it/s


 ↵ detail  l logs  x act  / filter  f state  ? help
```

**E. Job detail — Summary tab** (≥ 110×30; below that the log moves to its own tab).

```
 RunFleet ▏k2▕ › Jobs › 2332861                                                         ● k2 ok 0.21s · 3s ago   ⟳ 15s
 k2:2332861  orb-swp--train--cifar-warmup--s43--a02                    RUNNING  4:07:12 of 3-00:00:00 (2-19:52 left)
  ▌Summary▐  GPU   Logs   Steps   Procs   Raw
┌ Job ───────────────────────────────────────────────┐┌ Resources ─────────────────────────────────────────────────────┐
│ user       alice          account  lab             ││ nodes   1  n-40-01            cpus 8 (64-71)   mem 48G         │
│ partition  gpu-preempt        qos      gpu-priority││ gpus    1× nvidia_a100-sxm4-40gb   node n-40-01  IDX 2         │
│ submitted  09-06 11:14 (18h ago)                   ││ tres    cpu=8,mem=48G,node=1,billing=8,gres/gpu=1              │
│ started    09-07 01:50 (4h07m ago, waited 14h36m)  │├ GPU · source: sstat (Slurm accounting, every 60 s) ────────────┤
│ ends by    09-10 01:50                             ││ util  ▕█████████▊▏ 98%  (peak 99%)                             │
│ restarts   0          requeue  yes                 ││ mem   ▕██████    ▏ 24.3 / 40.0 G  (peak 24.3 G)                │
│ workdir    /shared/lab/alice/orb                   ││ cpu   AveCPU 4:20:02 (≈ 1.05 of 8 cores)   rss 9.9 / 48 G      │
│ command    /shared/lab/alice/orb/jobs/train.sbatch ││ hints none                                                     │
└────────────────────────────────────────────────────┘└────────────────────────────────────────────────────────────────┘
┌ stdout · /shared/lab/alice/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out · 663 B · 12s ago ───── ▌o▐ e ┐
│ [2026-09-07 01:50:39] host n-40-01  CUDA_VISIBLE_DEVICES=0  torch 2.9.1+cu130                                        │
│ [2026-09-07 01:50:52] config: cifar  model: resnet18  sched=warmup  seed=43                                          │
│ [2026-09-07 05:55:35] epoch 40/300  train_loss 0.3140  vs_base -2.38%  212 it/s                                      │
│ [2026-09-07 05:56:18] epoch 41/300  train_loss 0.3121  vs_base -2.41%  213 it/s                                      │
│ [2026-09-07 05:56:39] epoch 42/300  train_loss 0.3098  vs_base -2.44%  211 it/s                                      │
│ [2026-09-07 05:57:01] epoch 43/300  train_loss 0.3087  vs_base -2.47%  212 it/s                                      │
│ [2026-09-07 05:57:22] epoch 44/300  train_loss 0.3071  vs_base -2.50%  210 it/s                                      │
│                                                                                                    ── following ──   │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 esc back  [ ] tab  o/e stream  F follow  l full log  p pager  x actions  ^k cancel  s shell  y copy id  ? help
```

**F. Job detail — Steps tab of a failed job** (k2 fixture `sacct_failed_job_steps`; log lines illustrative).

```
 k2:2330539  lmft-x0-lr1e-5                                              FAILED  exit 1:0 · ran 0:59:05 · ended 2d ago
  Summary   GPU   Logs  ▌Steps▐  Procs   Raw
┌ Steps (sacct) ───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP              NAME        STATE       EXIT    ELAPSED   NODE      NTASKS  MAXRSS   GPU (max per task)            │
│ 2330539           lmft-x0-l…  FAILED      1:0     0:59:05   n-38-01   —       —        —                             │
│ 2330539.batch     batch       FAILED      1:0     0:59:05   n-38-01   1       24.6G    — (not in this fixture)       │
│ 2330539.extern    extern      COMPLETED   0:0     0:59:05   n-38-01   —       —        —                             │
├ Why it ended ────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • batch step exited with code 1 (the job script / program failed; not a Slurm kill)                                  │
│ • last log lines (illustrative): "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB …"          │
│ • log: <workdir>/logs/lmft-x0-lr1e-5_2330539.out (cached path from jobmeta)                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**G. Host detail — workstation** (nvitop-style; GPU rows get taller bars when space allows).

```
 RunFleet ▏labstation▕ › Hosts › labstation              ● ok 0.08s · 2s ago · Ubuntu 22.04.5 · driver 610.43.02  ⟳ 5s
┌ GPUs ────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  #  NAME        TEMP  PWR        UTIL                          MEM                                   PROCS           │
│  0  RTX A6000   30°C   24/300W   ▕                    ▏   0%   ▕                    ▏   0.0 / 48.0G   —              │
│  1  RTX A6000   31°C   28/300W   ▕                    ▏   0%   ▕                    ▏   0.0 / 48.0G   —              │
│  2  RTX A6000   29°C   20/300W   ▕                    ▏   0%   ▕                    ▏   0.0 / 48.0G   —              │
│  3  RTX A6000   78°C  287/300W   ▕██████████████      ▏  70%   ▕████████▊           ▏  21.0 / 48.0G   alice/363948   │
├ Host ────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  CPU ▕▋                   ▏  3%  of 64    MEM ▕█▊                  ▏  9%  45/503G    load 2.10 1.98 2.04   up 117d   │
├ Processes (GPU: all users · tasks: mine) ──────────────────────────────────── u: others shown · sort: GPU mem ▼ ─────┤
│  GPU  PID       USER      GPU-MEM   %CPU  %MEM   TIME       TASK       COMMAND                                       │
│  3    363948    alice     21.0G     100   0.8    18:54:06   p363942    …/.venv/bin/python -m orb.train --conf…       │
│  -    363942    alice     —         0.0   0.0    18:54:07   p363942    python -m orb.launch --manifest manif…        │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
│                                                                                                                      │
├ Connection ──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ssh labstation → ProxyJump jumpvm · mux: user (~/.ssh/cm-…) · probe 0.12s · sections ok 5/5 · process+nvidia+procfs │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 esc back  ↵ task detail  l log  ^k terminate  x actions  s shell  u others  o sort  ? help
```

**H. Host detail — Slurm cluster.**

```
 RunFleet ▏k2▕ › Hosts › k2                      ● ok 0.21s · 8s ago · Slurm 25.05.8 · cluster k2 · login login02  ⟳ 15s
┌ Partitions (assoc, recent jobs, GPU partitions open to you) ─────────────────────────────────────────────────────────┐
│  PARTITION     NODES  STATES                         GPUS FREE/TOTAL   BY TYPE (free/total)                          │
│  gpu-preempt   24     mixed 17 · alloc 5 · planned 2  27 / 152          3090 12/64 · a100-80g 0/12 · h200 3/4 · …    │
├ Nodes · gpu-preempt ────────────────────────────────────────────────────────────── f: free GPUs only · sort: name ▲ ─┤
│  NODE      STATE       CPU A/I/T      MEM ALLOC/TOTAL   GPU USED/TOTAL       TYPE        JOBS                        │
│  n-31-01   mixed       34/30/64       282/500G          8/8  ████████        3090        bob×4 · carol×3 · dave×1    │
│  n-31-04   mixed       6/50/56        72/125G           3/4  ███░            V100-16G    erin×2 · frank×1            │
│  n-31-05   planned     0/64/64        0/503G            0/12 ░░░░░░░░░░░░    A100 MIG    —                           │
│  n-32-03   mixed-      36/28/64       368/503G          8/8  ████████        3090        bob×4 · grace×~4            │
│  n-38-01   mixed       …              …                 4/4  ████            H200        2333760 · heidi×2           │
│  n-40-01   mixed       …              …                 3/4  ███░            A100-40G    2332861 · 2332862 +1        │
│  …                                                                                                                   │
├ Jobs on n-31-01 ─────────────────────────────────────────────────────────────────────── next GPU frees in ≤ 2h17m ─┤
│  JOB        USER       NAME               ACCOUNT        PARTITION    GPU  ELAPSED / LIMIT     LEFT     JOB NAME     │
│  2333115    carol      Carol Example      lab            gpu-preempt  3    1:42:30 / 4:00:00   2:17:30  llama-ft-lora│
│  2333301    dave       Dave Sample        physics        gpu-preempt  1    0:18:02 / 1-00:00   23:41:58 jupyter      │
│  2331907    bob        Bob Example        lab            gpu-preempt  4    20:11:05 / 2-00:00  1-03:48  ddp-train-r50│
├ My jobs on k2 ───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  2332861 R n-40-01 · 2332862 R n-40-01 · 2332863 R n-40-01 · 2333760 R n-38-01 · 2334564 PD (priority)               │
├ Capabilities / connection ───────────────────────────────────────────────────────────────────────────────────────────┤
│  tools squeue sacct sinfo scontrol sstat less · GPU accounting ✓ (sstat gres/gpuutil) · nvidia-smi ✗ (login node)    │
│  accounting ✓ · PreemptMode REQUEUE · MinJobAge 300 · assoc lab (gpu-priority, normal, labpi*) *protected            │
│  ssh k2 · mux: user ~/.ssh/cm-alice@k2…:22 · probe 0.23s · sections ok 4/4 · next poll 7s                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**I. Log viewer** (header, stream tabs, body, live line, hint bar).

```
 RunFleet ▏k2▕ › Jobs › 2332861 › stdout                 /shared/lab/alice/orb/logs/…warmup--s43--a02_2332861.out
 ▌stdout▐ stderr (merged)      663 B loaded (whole file) · tail -F via ssh · following ●
[2026-09-07 01:50:39] host n-40-01  CUDA_VISIBLE_DEVICES=0  torch 2.9.1+cu130
...
[2026-09-07 05:57:22] epoch 44/300  train_loss 0.3071  vs_base -2.50%  210 it/s
 ↻ epoch 45/300:  37%|███▋      | 211/570 [00:01<00:02, 212 it/s]                                        live line (\r)
 esc back  F follow  g/G top/bottom  o/e stream  p pager (search)  y copy path  ? help
```

**J/K. Confirm dialogs** (default focus on *Cancel*; `y` confirms, `n`/`esc` cancels).

```
                    ┌ Cancel 3 jobs on k2? ───────────────────────────────────────────────┐
                    │  2332861  orb-swp--train--cifar-warmup--s43--a02       R  4:07:12   │
                    │  2332862  orb-swp--train--cifar-const--s43--a02        R  4:07:12   │
                    │  2332863  orb-swp--train--cifar-cosinewarm--s43--a02   R  4:07:12   │
                    │                                                                     │
                    │  will run on k2:  scancel 2332861 2332862 2332863                   │
                    │                                                                     │
                    │                        [ Cancel  (n/esc) ]   [ Confirm  (y) ]       │
                    └─────────────────────────────────────────────────────────────────────┘

                    ┌ Terminate task p642229 on lynx? ────────────────────────────────────┐
                    │  orb.launch ×2 (orb.train) · 3 processes · GPUs 0,1 · 18:45:12      │
                    │  members are re-read on lynx at confirm time; launcher paused first │
                    │                                                                     │
                    │  [x] send SIGKILL to survivors after [10] s                         │
                    │  will run on lynx:  STOP 642229 · TERM its tree · KILL after 10 s   │
                    │                                                                     │
                    │                        [ Cancel  (n/esc) ]   [ Confirm  (y) ]       │
                    └─────────────────────────────────────────────────────────────────────┘
```

**L. Help overlay** (contextual; only keys valid in the current view; type to filter).

```
          ┌ Help · Jobs view ─────────────────────────────────────────────── type to search · esc close ┐
          │ NAVIGATION                         SELECTION & ACTIONS              VIEW                    │
          │ j/k ↑/↓   move                     space   mark / unmark            /     filter            │
          │ g/G       top / bottom             V       mark all visible         f     next state preset │
          │ ^d/^u     half page                x       action menu              o     next sort column  │
          │ ↵         job detail               ^k      cancel / terminate       O     reverse sort      │
          │ l         log viewer               y / Y   copy id / log path       z     group by          │
          │ p         log in $PAGER            s       shell                    u     other users       │
          │ tab       next view                c       connect host             +/-   poll faster/slower│
          │ 0 / 1-9   all hosts / host N       r ^r    refresh scope / all      ^p :  command palette   │
          └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**M. GPUs view.**

```
 RunFleet ▏ALL▕  ● k2 8s  ● vega 3s  ◐ lyra retry 38s  ● labstation 2s  ✖ lynx auth     R 13  PD 2  F 3/24h   ⟳ 15s
  Fleet  Jobs ▌GPUs▐    workstation GPUs (live) + GPUs of my running Slurm jobs (util via sstat)        14 GPUs
   HOST        NODE     IDX  NAME             UTIL               MEM                       TEMP  PWR    JOB / PROCESS
   k2          n-40-01  2    A100-SXM4-40GB   ▕█████████▊▏ 98% * ▕██████    ▏ 24.3/40.0G *  —     —      2332861
   k2          n-38-01  0-1  H200 NVL ×2      ▕████████▌ ▏ 86% * ▕███▍      ▏ 96.1/281G  *  —     —      2333760
   labstation  —        0    RTX A6000        ▕          ▏  0%   ▕          ▏  0.0/48.0G    30°C  24W    —
   labstation  —        3    RTX A6000        ▕███████   ▏ 70%   ▕████▍     ▏ 21.0/48.0G    78°C  287W   p363942 train
 ~ lynx        —        0    RTX PRO 6000 BW  ▕█████████▊▏ 97%   ▕█▌        ▏ 14.9/95.6G    55°C  196W   p642229 train
 * per-job aggregate from Slurm accounting (sstat), not a per-GPU reading · per-GPU Slurm readings are v0.2
```

**N. First run without a config** (plain terminal before the TUI starts, `cli.py`; `runfleet hosts import` is the
non-interactive equivalent).

```
$ runfleet
No config yet (~/.config/runfleet/config.toml). Hosts in ~/.ssh/config:
  1 k2    2 vega    3 lyra    4 labstation    5 lynx    6 jumpvm    7 owl
Monitor which? (e.g. 1-5, k2,lynx, all; empty = quit, d = demo): 1-5
Probing 5 hosts (read-only, BatchMode) …
  k2          ok          slurm 25.05.8               0.21s  mux user
  vega        ok          slurm 25.11.4               0.20s  mux user
  lyra        ok          slurm 26.05.3               1.02s  mux managed
  labstation  ok          workstation 4× RTX A6000    0.08s  mux user
  lynx        needs auth  (saved; press c in RunFleet or run `runfleet connect lynx`)
Wrote ~/.config/runfleet/config.toml (5 hosts).
Starting RunFleet …
```
Lines appear as each probe finishes (at most `max_parallel_ssh` at a time). Hosts that did not answer are followed by
`Not saved (no answer now): jumpvm. Add later with \`runfleet hosts import jumpvm --write\` (slow logins: raise
connect_timeout in the config).`

### 7.4 Responsive rules
- Header: full form ≥ 110 columns; < 110 drops ages; < 90 shows dots + names only; < 70 dots with numbers (`●1 ●2 ◐3`).
- Fleet: cards when width ≥ 100 (2 per row ≥ 120, 3 per row ≥ 180); table otherwise. `#fleet-cards` always reserves
  its scrollbar column (`scrollbar-gutter: stable`) and card text is fitted to `(W − 1) // cols − 4` cells, never
  wrapped (`text-wrap: nowrap; text-overflow: ellipsis`): at 100×30 a wrapped line used to make one card of a row taller.
- **Fleet roomy layout** (`ui/fleet_layout.py`, pure; mockup A2). Today's cards are tier **T0**; a tall view is tier
  **T1**. Heights depend only on geometry: `plan(W, H, shapes, enabled)` is a function of the view size and each card's
  *shape*, never of the data, so cards do not jump when jobs come and go.
  - *Shapes* (content lines, borders excluded): no snapshot (3, 3, 0); Slurm `t0 = 5 + F`, `t1 = 4 + F`, cap 12, with F
    = the lines of the free-GPU block for the partition names and totals with everything free and planned (the widest
    text they can produce), or for today's numbers when those take more (a partition too wide for the block even alone
    folds the widest text into one `+N more` line, its smaller numbers fit on a second); workstation `t0 = G + 3`, `t1 = G + 1`, cap 8, G = GPU lines (≤ 8, + 1 for "+N more", 1
    without GPUs). F and G are per-host high-water marks for the session, reset only when Settings changes the host set
    or order; so a re-plan happens only on a resize, a host-set change, a shape change (first snapshot, more GPUs, a new
    partition name or total) or the palette switch.
  - *Gate*: `W ≥ 100`, view height `H ≥ 32`, card inner width ≥ 52 and `spare = H − Σ(base_row + 2) ≥ 3 × rows`
    (base_row = the tallest `t0` in the row); otherwise T0, today's code path unchanged. ui.json `fleet_layout =
    "compact"` forces T0 (§7.13).
  - *Allocator*: row caps = the tallest `t1 + cap` in the row (the orphan row at least 24, the panel's room); spare
    rows go round-robin, one per row per pass, top row first, until spent or every row is at its cap. An orphan cell
    with ≥ 6 rows takes the rest as the fleet panel (`FleetPanel`, column-span = the empty cells); a full grid gets a
    full-width panel row below it when ≥ 10 rows are left. Demo fleet: 172×62 → rows [17, 17, 19], panel 19;
    120×40 → [11, 12, 8], panel 8; 120×35 → [10, 10, 6]; 100×30 and 80×24 stay T0 / table.
  - *Inside a T1 card* every slot is fixed: the summary (free-GPU block padded to F, never longer: a partition that
    does not fit is counted in `+N more`, never dropped), then `login` (Slurm) or the host
    line (workstation), then a list region of exactly `h − t1` rows. Error lines replace `login` / the host line and
    take the first list row; "next" moves into the list; a workstation adds `swap · up · load` from 8 list rows.
    Lists rank own jobs by need (Slurm: fresh failures < 4 h, running with a warn/red hint or low on time or a red/amber
    runway, never-run or held pending, older failures in `recent_window`, other running by time left, other pending by
    estimated start; workstation: fresh failures, own running with a hint, own running by GPUs, other users' tasks,
    older failures; CD/CA never, a plain EX task is no failure; ties by job id). Shown grouped failed / running /
    pending (others); with more candidates than rows, array siblings of rank 4/5 fold (`orb-sweep [3-4] ×2`), then
    `rows − 1` rows and `+N more · R a, next ends in X · PD b · tab: Jobs`. Wide template (inner ≥ 72): label 10 |
    badge 4 | name inner − 55 | detail 20 | GPU% 6 | time 11; narrow: badge 4 | name inner − 34 | detail 16 (running:
    GPUs 9, a space, GPU% 6) | time 11, so `20h39m left` never moves the GPU% column. Folded array tasks are listed in
    task order (`[3,5,12]`). Time cells change once a minute (`<1m`, `33m left`, `2d19h left`, `5h20m up`, `in 35m`, `overdue`, `3h ago`).
    Workstation GPU lines gain an owner column (22 cells wide, a 6-cell tag narrow): `?` (old reading), `no reading`
    (`?` narrow: neither util nor memory sampled), `mine orb.launch` / `carol sample.py` (`+N` more owners), `in use ·
    no process`, `free`.
  - *Fleet panel* (always the whole fleet, not the scope; which hosts may be counted is `fleet_layout.free_source`,
    shared with the GPU watches of §7.14): free GPUs now, counted once per node with
    `util/slurmfmt.node_free` (an allocated node with an idle GPU counts 0; `parse_sinfo.summarize` uses the same), only
    nodes of the listed partitions, MIG slices apart per host, workstation GPUs not in use and not an old reading;
    problem/stale hosts show as `host ?`, and so does a fresh cluster whose last `slurm.sinfo` failed, is missing (nodes
    from the cache only) or is older than `fleet_layout.sinfo_bound_s` (two periods of the slowest legitimate
    `slurm.nodes` cadence, + 5 s: `poll_nodes` × hidden 5 × idle 4 × the +/- rate × a busy controller's slowdown, at
    least the 600 s backoff cap; 80 min at the defaults);
    your types (GPU types of your active, queued or recent jobs and your tasks' GPUs) first by fleet total, the rest on
    one `also free` line; `sinfo Nm ago` = the oldest counted sinfo. GPU types are labelled by one normaliser for gres
    types and nvidia-smi names (`fleet_layout.type_label`: `nvidia_a100-sxm4-40gb` and `NVIDIA A100-SXM4-40GB` are both
    `A100-40G`; a glued `rtx3090` / `rtx6000` reads like `GeForce RTX 3090` / `Quadro RTX 6000`, an Ada card keeps
    its generation, `6000Ada`, apart from the Turing `6000`; a `Ti` / `SUPER` suffix stays with the number and `GTX` is
    a prefix like `RTX`: `rtx2080ti` and `GeForce RTX 2080 Ti` are `2080TI`, `gtx1070` is `1070`). Then what changed (§7.14), newest first. At most half the
    panel goes to the free part. It never takes focus; arrows and enter skip it, and a click on it (or on a gap of the
    grid) leaves the focus on the selected card (`#fleet-cards` does not focus on click), except that a click on an
    activity row opens that job (below); its legend rows in the help overlay appear only while it is shown.
  - *Opening entries*. Every rendered list row and activity row keeps its `JobKey`, in step with the lines:
    `fleet_layout.list_region` / `activity_region` return `(lines, keys)` (a folded array row keeps its first job, the
    overflow and count lines None), `host_card.card_body` / `fleet_panel.panel_body` carry them to
    `HostCard.row_keys` / `FleetPanel.row_keys` (T0 cards: none). A click maps its content row
    (`Click.get_content_offset`) to that key and opens the job (`App.open_listed_job`: a toast when the store no
    longer lists it); a click on any other line of a card selects it, a double-click opens the host, as before.
    `a` (fleet context, any layout: roomy, compact or the table) opens `AttentionScreen`: `fleet_layout.attention`
    merges the cards' cached rankings (`FleetView._ranked`, also used at T0 and in table mode), whole fleet, keeping
    only rows that need the user — tier 0 fresh failures, 1 running with a warn/red hint or low on time (workstation:
    own task with a hint), 2 pending that will never run or held, 3 older failures in the window (Slurm ranks 0–3;
    workstation ranks 0, 1, 4); sorted by tier, failures newest first across hosts, else host order, then card rank
    order; array siblings never fold. A row reads `host  id  badge  full name  why` (`attention_why`: the card row's
    detail and time cells, `low on time` for a runway-only rank 1), the reason on a second line when the full name
    and it do not fit one; `enter` dismisses with the key and opens the job, `esc` closes.
  - *Cost*: the 1 Hz tick re-renders text only. `FleetView` caches each card's ranking and its `distinct_starts` per
    (jobs dict, minute, window) and until a listed failure turns 4 h old or leaves the window
    (`fleet_layout.rank_expiry`, to the second, so the list agrees with the card's `my jobs` counts); the ranking (a
    `RankedList`) keeps its selection per row count, so folding and selecting run once per ranking (folding is linear
    in the array size). A `fleet_layout.Memo` keeps the panel's scans: the node scan per host until its
    nodes/partitions change, the own-job type scan and the ended-job list until its jobs change, and the activity rows
    until then, a new alert or the moment an end or event leaves the window (`activity_expiry`).
- Jobs table columns (priority 1 = never dropped; the highest priority number goes first, the rightmost first among
  equals). Widths are minimums; the slack goes in tiers: NAME up to 28, NODE / REASON up to 26, NAME up to 40, the rest
  to NODE / REASON (at 120 columns `never: dependency failed` fits). NAME keeps what tells sweep siblings apart: the
  part after the longest prefix shared with its sorted neighbours, backed up to a segment start, is kept whole when it
  fits (`orb-swp--tr…warmup--s43--a02` / `orb-swp--tra…const--s43--a02` / `orb-swp…cosinewarm--s43--a02`); a name
  without such siblings is cut middle-right (keeps the seed/attempt suffix). Cuts are measured in terminal cells; a name
  with emoji sequences (a variation selector, a ZWJ sequence of emoji) is cut at Rich's grapheme boundaries and
  measured per grapheme, never per code point. At 80 columns LEFT and GPUS are dropped
  before NODE / REASON and GPU% (critic: pending reasons and hints were invisible).

| Col | Min width | Priority | Content |
|---|---|---|---|
| marker | 1 | 1 | `+`/`!`/`~` |
| HOST | 8 | 1 | host name (coloured by host colour) |
| ID | 8 | 1 | job id / `p<pid>` |
| NAME | 16 | 1 | job name / task label |
| ST | 3 | 1 | state code, coloured; `H` suffix when held (`PDH`) |
| ELAPSED | 8 | 1 | running/elapsed time (extrapolated each second for ACTIVE), `1-00:15` (minutes) from one day on; `—` for PD |
| GPUS | 7 | 3 | `2×H200`, `—` |
| GPU% | 6 | 2 | mini bar `▇` + %, `–` when unavailable |
| NODE / REASON | 12 | 2 | hint first then node(s) for running (`log idle 49m · n-40-01`), humanized reason/est. start for PD, exit summary for ended |
| LEFT | 8 | 3 | time left / limit for PD / `ended 2d` |
| PART | 10 | 4 | partition (hidden by default; enabled via `jobs_columns`) |
| GMEM | 8 | 4 | job GPU memory |
| CPU, MEM | 5, 6 | 5 | task cpu%/rss or Slurm alloc |
| USER | 9 | auto | shown only when other users' rows are visible (`u`) |
| SUBMIT | 11 | 5 | submit time (relative) |

- Height: < 30 rows → preview collapses to one summary line; < 20 → preview hidden.
- Job detail Summary below 110×30: one 3-line summary box (who/when/workdir; nodes, CPUs, memory, GPUs with indices;
  GPU/CPU use and hints) and the live log under it; the full Job/Resources/GPU boxes are hidden (GPU, Steps and Raw tabs
  keep the details). The title shortens its times (`0:25:06 (0:34:53 left)`) before it cuts the job name.

### 7.5 Jobs table behaviour
- **Default filter "fresh"**: ACTIVE + QUEUED + anything that ended within `recent_window` (24 h). Failed, cancelled and
  ENDED rows within the window stay visible by default.
- **Default sort**: host order (config order), then state group (ACTIVE, QUEUED, FAILED, ENDED, CANCELLED, DONE), then
  submit time descending. `o` cycles the sort column, `O` reverses, clicking a header sorts by it.
- **Group by** (`z`): none → host → array (array tasks under their parent). Group header rows show counts and mean GPU%,
  fold/unfold with `enter`/click, and are never action targets.
- **Marks** (`space`, `V`) are keyed by `JobKey`, survive refreshes, sorting and filtering; actions apply to the marked
  rows that are **visible** (a filter or preset that hides a marked job keeps actions off it; the status line says
  "N marked (M hidden — not acted on)"), else the cursor row. `esc` clears marks when no filter/overlay is active.
- **Updates**: keyed diff (`update_cell` for changed cells, `add_row`/`remove_row` for membership); rebuild only when
  > 30 % of rows changed or the order changed; cursor restored by key **on the same screen row** (the viewport offset
  is kept, so a GPU%/elapsed sort does not shift the table under the user). A 1 Hz timer updates ELAPSED/LEFT of
  visible ACTIVE rows only.
- **Columns**: ID is sized from the visible labels (8–10 cells, cut from the left so array task suffixes stay
  visible: one `31377076_[5-12%2]` must not widen ID for every 7-digit id); LEFT shows the time left for active jobs and the estimated start (`~in 37m`) for queued ones (the limit is
  in the preview); NODE / REASON of a workstation task says `gpu 0,1`, `2 GPUs` (indices unknown) or `cpu only`.
  Header and status line share one sort-direction helper (▲ ascending, ▼ descending; elapsed/GPUs/GPU%/submit sort
  largest-first by default) and the header arrow is never truncated away.
- **Preview pane**: one-line metadata, resource line, GPU line, last 4–8 log lines from `peek_log` (requested after
  1 s of cursor dwell; cached; never a stream). Hidden below 20 rows, one summary line below 30.

### 7.6 Filter (`ui/query.py`, pure)
`/` opens the filter bar (live, 100 ms debounce; `enter`/`tab` returns focus to the table, `esc` clears). Terms are
space-separated and all must match:

| Term | Meaning |
|---|---|
| `word` | substring of `host id name state partition nodes reason user`, or of the name with separators removed (`orbswp` → `orb-swp--…`); no subsequence fuzz (it matched almost every long sweep name) |
| `host:k2` `h:k2,vega` | host in list |
| `state:R` `st:pd,f` / `state:failed` / `st:running` | state code, group name (`queued active done failed cancelled ended`) or state name (`running pending completed timeout cancelled`, aliases `oom canceled done`) |
| `user:me` / `user:alice` | owner |
| `!term` | negation of any term |

State presets (`f` cycles, palette `State: …`): fresh (default) → active → queued → failed (FAILED + ENDED) → all. The
filter text, preset, sort and grouping persist in `~/.local/state/runfleet/ui.json`.

**Saved views** (`ui/saved_views.py`, pure; `ui/screens/save_view.py`). A named set of the five Jobs-view settings,
stored in ui.json `views` = `{name: {filter, preset, sort, reverse, group}}`. Palette only, no key:
- *Save view as…* opens `SaveViewScreen` (the port prompt's pattern, keymap context `prompt`): a summary line of what is
  saved (`failed · sort state ▲ · group host · /k2`), the name input (pre-filled with the active view's name) and a
  live note: why the name cannot be used, that it replaces an existing view, or `N of 20 saved`. `enter` with a usable
  name saves and toasts; `esc` cancels. It saves the Jobs view's settings from any view and does not switch views.
- `View: <name>` (help text = the summary line) switches to the Jobs view and sets all five at once: filter bar
  value and visibility, folded groups cleared, the digest's ended-since mode (§7.14) ended; ui.json gets the five
  values plus `active_view`.
- `Delete view: <name>` removes it (no confirmation; a toast says so).

Names: 1–24 characters of ASCII letters, digits, space, `.`, `_`, `-`, starting with a letter or digit, no double
space and none at the end (the prompt strips the name; a hand-written `gpu ` is skipped); unique ignoring case (saving `GPU` over `gpu` replaces it under the new spelling). At most 20 views: a 21st new
name is refused in the prompt (replacing one is fine). Reading is tolerant: `views` that is not an object, a key that
is not a valid name, a value that is not an object, a filter that is not a one-line string, an unknown preset / sort /
grouping or a non-bool `reverse` make that entry (or `views`) ignored, never an error; a missing field takes the Jobs
view default; of names equal ignoring case only one is used; beyond 20 the rest (by name) are ignored. Writes touch
only the entry asked for, so entries this version cannot read (a newer RunFleet's) stay in the file.

**Active view**: the status line names it — `view: <name>` first in the wide forms, `<name> · ` first below 100
columns and in the shortest fallback — from applying or saving it until any of the five settings changes (or the
digest's `f` sets the preset aside); changing it back does not bring the name back, and deleting the view drops it.
`active_view` in ui.json brings the name back at the next start only when the restored settings still equal the view.

### 7.7 Header, refresh indicators, stale and error presentation
- **FleetHeader** (1 line): `RunFleet`, scope chip `▏ALL▕`/`▏k2▕`, one chip per host (`<dot> <name> <age>`; age = time
  since last successful data; RETRYING `retry 38s`, NEEDS_AUTH `auth`), totals `R n PD n F n/24h` of **my** jobs over
  the scope (like the cards and the default Jobs view; the Jobs view's `u` adds other users'),
  refresh indicator `⟳ 15s` (fastest current tick; a spinner replaces `⟳` while any probe is in flight), `R/O` in
  read-only mode, `idle ×4` when throttled, `paused (idle) — any key resumes` when paused. Clicking a host chip scopes to
  it; clicking a red chip opens HostErrorScreen. The `RunFleet` logo returns to the Fleet view from any screen; on detail screens the host chip and every breadcrumb part but the last (the screen itself) is clickable — `app.crumb(kind, …)`: `("view", jobs|fleet)` shows that main view, `("host", name)` / `("job", host, id)` pop back to that screen when it is further down the stack (state kept), else open it from the main screen; a `⚙` chip at the right edge opens Settings (same as `,`). The
  compact chip forms (name or number only, when the full chips do not fit) end in `:d`/`:a`/`:k` for unreachable,
  needs auth and host key (`✖lynx:a`).
- **Stale data** (`HostState.stale(now)`: state ∉ {OK, DEGRADED}, or data older than 2 × the host's *effective* tick +
  5 s — normal hidden/idle throttling is not stale): rows dimmed with marker `~`; cards/host screens show `last ok 4m
  ago`; detail screens show "data from 4 m ago — lyra is retrying (timeout via jumpvm), next retry in 38 s"; a host
  that never answered reads "no data yet — lynx needs a login … — press c to log in", and its Host detail shows only the
  Connection panel (keys `c`/`e`/`t`) instead of an empty workstation layout.
- **Free GPUs** (cards, partition table): whole GPUs free/total, then free MIG slices apart (`gpu-preempt 13/108 +11
  MIG`; on k2 17 of 22 "free GPUs" were 1g.10gb-style slices, M3). BY TYPE merges types that render the same label
  (`nvidia_a100-sxm4-80gb` + `nvidia_a100_80gb_pcie` → one `A100-80G` entry, summed). Node names that do not fit are
  cut from the left (`…gpu-s71-02`), and the host-detail NODE / GPUs-view NODE columns widen to the longest name in
  scope when room allows (lyra rows all read `lyra-gpu-…`, M3).
- **Hint bar**: a hint that covers a key pair shows the pair (`[/] tab`, `[/] partition`, `o/e stream`, `g/G
  top/bottom`); the help legend's `F 5/24h` example follows `general.recent_window`.
- **Errors**: per-host chip colour + card footer with the classified one-liner and the fixing key (`press c to log in`,
  `e info`). Section-level problems (`DEGRADED`) show `!` and are listed in the host screen's Connection/Capabilities
  box with stderr excerpt and next retry.
- **Unavailable metrics** render `–` (never 0) with the reason in the detail view: "n/a — no GPU accounting on this
  cluster", "n/a — job pending", "unavailable — nvidia-smi hung (retry in 60 s)".
- **Toasts** (`App.notify`): host transitions to/from UNREACHABLE / NEEDS_AUTH / HOSTKEY (one per transition), action
  results, second-instance warning. At most 3 per 10 s; extras coalesce ("+2 more").

### 7.8 Key bindings (`ui/keymap.py` is the single source; help overlay and `docs/USER_GUIDE.md` are generated from it)

Rules: single keys only (no prefix chords in v0.1). Action ids map to keys only in `keymap.py`. `ctrl+c` (quit) and
`ctrl+z` (`suspend_process`) are bound on the App with `priority=True`, overriding Textual's `help_quit`/`copy_text`.
On Windows (no job control; Textual makes the action a no-op) `keymap.bindings` drops `ctrl+z` and the help overlay
does not list it; `keymap.markdown()` (the USER_GUIDE table) is the same on every OS, so `gen_docs --check` is too.
`tab`/`shift+tab` switch main views only on MainScreen (whose only focusables are the table and the filter bar); modals
and detail screens keep Textual's focus traversal. `→`/`right` keeps DataTable's horizontal scroll.

| Context | Keys |
|---|---|
| Global | `?`/`F1` help · `ctrl+p` or `:` command palette · `,` settings · `q` back (MainScreen: quit) · `ctrl+c` quit now · `esc` close overlay → clear filter → clear marks → back · `0` all hosts · `1`–`9` host N · `/` filter · `r` refresh scope (≥ 3 s per host) · `ctrl+r` refresh all · `+`/`-` poll faster/slower (0.5×…4×, floors apply) · `ctrl+z` suspend · `D` since you were away (§7.14) |
| MainScreen | `tab`/`shift+tab` next/previous view (Fleet → Jobs → GPUs) |
| Fleet view | arrows/`hjkl` move between cards · `enter`/double-click host detail · click selects (on a card's job row or a panel activity row: opens that job) · `a` needs you: what needs the user fleet-wide (§7.4) · `space` (or a header chip click) scopes · `c` connect · `C` log in to every host needing auth (§7.15) · `s` host shell · `t` enable/disable polling · `e` error details |
| Jobs view | `j`/`k`/arrows/wheel move · `g`/`G`/`home`/`end` first/last · `ctrl+d`/`ctrl+u`/`pgdn`/`pgup` half page · `enter`/double-click detail · `l` log viewer · `p` log in `$PAGER` · `ctrl+f` search the logs of your recent jobs in scope (§4.7.1) · `x`/right-click action menu · `ctrl+k` cancel (Slurm) / terminate (task) · `space` mark and move down · `V` mark all visible / clear · `A` mark / clear the row's whole array · `y`/`Y` copy id / copy menu (`Y Y` = stdout path, §7.15) · `s` shell · `P` forward a port · `c` connect the row's host · `f` next state preset · `o` next sort column · `O` reverse sort · `z` cycle grouping · `u` toggle other users' GPU processes · `w` watch / unwatch the row or the marked rows (§7.14) |
| GPUs view | navigation as Jobs · `enter` owning job/task (or host) · `x` actions of the owning task · `ctrl+k` terminate owning own task · `c` connect the row's host · `a` copy `CUDA_VISIBLE_DEVICES` for N free GPUs (workstations) · `o`/`O` sort |
| Job detail | `[`/`]` previous/next tab · `o`/`e` stdout/stderr · `F` follow on/off · `l` full log viewer · `L` log viewer landed on the cause chip's line (§5.2.8) · `/` search the log (opens the log viewer with its search box; never the Jobs filter) · `p` pager · `r` refresh detail (reopens a dead stream) · `x`, `ctrl+k`, `s`, `c`, `y`, `Y`, `P`, `w` as in Jobs · `esc`/`q` back |
| Host detail | `enter` on a process opens its task · `ctrl+k` terminate task · `x` actions · `u` other users · `o`/`O` sort · `s` shell · `c` connect · `t` enable/disable · workstation: `a` free-GPU picker · cluster: `f` nodes with free GPUs only, `[`/`]` previous/next partition |
| Log viewer | `F` follow on/off · `G` bottom + follow · `g` top of the buffer · `o`/`e` stream · `/` search the buffer, `n`/`N` next/previous match (`esc` clears) · `]`/`[` next/previous error or warning line (§5.2.8) · `p` pager · `R` reopen the stream · `y` copy path · `c` connect |
| Modals | Confirm: `y` / `enter` on Confirm, `n`/`esc` cancel, `tab` moves between params and buttons · Action menu: `j`/`k`/`enter` · Notifications: `j`/`k`, `enter` opens the job (a GPU watch: its host) · Since you were away: `j`/`k`, `enter` opens a failed job, `f` filters the Jobs view to it · Needs you: `j`/`k`, `enter` opens the job · GPU watches: `j`/`k`, `n` new watch, `d`/`delete` removes one · Port forwards: `enter` copies the URL again, `ctrl+k` cancels · Prompt: `enter` uses the value, `esc` cancels · Search logs: `tab` to the options, `space` toggles, `enter` in the pattern searches; results: `j`/`k`, `enter` opens the log viewer at the match (`esc` there comes back), `/` edits the search · `F1` in any dialog: help with the dialog's keys · Help: typing filters (`q` included); `?`, `F1` and `esc` close; a glyph legend sits under the keys |
| Settings | `tab`/`→` and `shift+tab`/`←` section · `j`/`k`/arrows/wheel move · `space`/`enter` toggle host / next value / open · `t` test host · `a` add ssh target · `p` poll a saved host this run left out · `J`/`K` or `shift+↓`/`↑` reorder saved hosts · `/` filter · `e` config in `$EDITOR` · `?` these keys · `esc`/`q`/`,` close (`esc` first leaves the input line or a pending confirm) |

### 7.9 Mouse
Click selects rows/cards/tabs/header chips (a job row of a roomy Fleet card or an activity row of the fleet panel
opens that job, §7.4); double-click drills in; right-click opens the action menu; clicking a
column header sorts by it (again reverses); the wheel scrolls tables and logs (3 lines), `shift+wheel` scrolls wide
tables horizontally; hint-bar entries are clickable. Text selection: hold `Shift` while dragging (terminal-native in
most emulators), or start with `--no-mouse`; `y`/`Y` copy ids and paths without selecting. Works over SSH and in tmux
(`set -g mouse on`; SGR mouse reporting).

### 7.10 Command palette (`ui/commands.py`)
Textual's built-in palette (`ctrl+p`; `:` opens the same palette) with only the RunFleet `Provider` (`App.COMMANDS =
{RunFleetCommands}`: Textual's Keys/Maximize/Screenshot/Theme competed with `?` and did nothing useful here) yielding
fuzzy-matched commands — no command grammar: `Fleet view`, `Jobs view`,
`GPUs view`, `Scope: all` / `Scope: <host>`, `Host detail: <host>`, `Connect <host>`, `Refresh <host>` / `Refresh all`,
`Enable <host>` / `Disable <host>`, `State preset: fresh|active|queued|failed|all`, `Sort by <column>`, `Group by
none|host|array`, `Save view as…` / `View: <name>` / `Delete view: <name>` (saved Jobs views, §7.6), `Doctor <host>` (runs `FleetAPI.doctor`, shows the report in a modal), `Settings` / `Settings:
general` / `Settings: about`, `Update RunFleet` (§8.6), `Fleet: roomy cards` /
`Fleet: compact cards` (§7.4, §7.13), `Search logs…` (§4.7.1), `Action log`, `Notifications` / `Since you were away` / `Watch free GPUs…` / `Watches` (§7.14), `Log in to all hosts
needing auth` / `Forward a port…` / `Forwards` (§7.15), `RunFleet log`
(`run_external(service.local_pager_argv())`: the local pager on `runfleet.log`, not LogScreen), `Help`, `Quit`.

### 7.11 Colours and thresholds (`ui/theme.py`)
- States: ACTIVE green; QUEUED yellow (held: yellow + `H`, e.g. `PDH`); DONE dim blue; FAILED red (PREEMPTED orange);
  ENDED grey (with a red hint when the exit scan found an error); CANCELLED muted magenta (a solid mix, not SGR 2:
  iTerm2 draws faint text half-transparent); UNKNOWN grey italic.
- GPU bars: partial blocks `▏▎▍▌▋▊▉█`; utilisation < 10 % green, < 75 % yellow, ≥ 75 % red; memory 10/80 (nvitop
  defaults, `[ui]` thresholds).
- Host chips: ok green, degraded yellow `!`, retrying yellow `◐`, unreachable red `✖`, needs_auth/hostkey magenta `✖`,
  disabled dim `○`.
- `NO_COLOR` honoured (bold/reverse instead of colour; never reverse on bar fills — a reverse-video █ is drawn in the
  background colour — high GPU% gets a `!` instead; table cursor rows get `text-style: reverse` via the app class
  `-no-color`, since backgrounds are stripped); transparent default background.
- Body text uses the terminal's default foreground (`color: ansi_default`, DataTables included: Textual's default is
  the theme's); bars with a theme background keep the theme foreground.
- **Mode and background.** `cli.run_tui` asks the terminal for its background before Textual starts
  (`ui/termbg.py`: OSC 11 + a DA1 sentinel on /dev/tty, non-canonical, the DA reply ends the wait (≤ 2 s), the
  /dev/tty descriptor closed on every path, background jobs skipped) and calls `theme.setup(choice, background)`. A
  colour counts only from a complete reply (BEL / ST terminated). With the DA reply in time, typeahead — the input
  minus every OSC sequence and DA reply, complete or cut short — is pushed back with TIOCSTI. Without it the line is
  drained until quiet (≤ 3 s more), flushed, and nothing is pushed back, so a late or split reply never reaches
  Textual as keys. *Deferred (known limits):* where TIOCSTI is disabled (Linux `dev.tty.legacy_tiocsti=0`, OpenBSD)
  keys typed during the query are lost, and a reply arriving after the drain cap reaches Textual as keys. Mode precedence: `RUNFLEET_THEME` (dark|light|ansi) › the Settings › About choice
  (ui.json `theme`; `--demo`: auto) › a 16-colour terminal (Rich's detection → `ansi`) › the background
  (`theme.is_light`: light when black ink has more contrast on it than white, so #999999 is light) › `COLORFGBG` (bg 7 / 9–15 light) › dark. Hues are fixed per mode (dark / light sets); every neutral
  (DIM, GREY, MUTED, FOREGROUND, TRACK, SURFACE, RULE, PANEL, BORDER) and tint (CURSOR, CURSOR_BLUR, HOVER,
  BORDER_PROBLEM, CANCELLED) is `mix(background, ink | hue, t)` — on black exactly the old fixed greys (±1). A
  background contradicting a forced mode is not used as the base. In 256-colour mode the fills snap to the xterm-256
  entry the terminal will draw. `ansi`: named ANSI colours only, `Theme(ansi=True)` with Textual's ansi-dark variables,
  no bar track, host colours from the 9 non-grey slots. Ansi or 16 colours (`PLAIN_CURSOR`): reverse / underlined cursor
  rows (app class `-plain-cursor`), the focused detail tab and dialog menu highlights in reverse video, and pills /
  cause chips are `bold reverse <colour>` (the terminal's own background on the colour: "default on blue" put the
  terminal's foreground on it, 1.2–1.5:1 on some profiles). 256 colours: fills snap to the entry the terminal draws
  (neutrals on the grey ramp), each stepped away from the background and the previous fill; the cursor family
  (CURSOR, CURSOR_BLUR, HOVER, `theme._cursor256`) takes, for each, the xterm entry 16–255 nearest (redmean) to its
  truecolor tint with MUTED (as drawn) ≥ 3.5:1 on it, on the ink's side of the *reported* background, ≥ 1.2 / 1.1 /
  1.02:1 against it and in that order of loudness; only grey (and, for the cursor, the accent's blue side: the nearest
  cube entry to a pale tint is often pink or cyan), measured as Rich draws it. When nothing quieter fits (near white
  few greys are drawable) the blurred cursor / hover share the louder neighbour's entry rather than take another hue.
- **No translucent CSS colours.** Textual blends `$x N%` over `ansi_default` against black (it cannot know the
  background), so frames, cursor rows, hover, scrollbars and panels use solid `$rf-*` theme variables
  (`TEXTUAL_THEME.variables`; `RunFleetApp.get_theme_variable_defaults` provides them before the theme is set), and
  modals have no backdrop. Table cursor rows keep each cell's colours (`cursor_foreground_priority="renderable"`).
  The highlighted option of a menu in a dialog (action menu, digest, forwards) is drawn on `$rf-cursor` with the
  dialog's foreground, not on the accent (its prompts keep their own colours, 1.0–1.6:1 on the accent).
- **Clickable bars keep their colours.** Textual's `link-color` repaints every `@click` span; FleetHeader, HintBar and
  ViewTabs override `Widget.link_style` (`theme.link_style()`: no colour; hover bold on HOVER).
- **Live switch** (`RunFleetApp.set_theme_choice`): `theme.setup` with the background reported at start, register the
  new Textual theme (named per mode and base, so the reactive fires), toggle `-plain-cursor`, call `forget_rendered()` on every widget that has it (tables rebuild their rows,
  `FleetView` its empty-fleet message, `LogView` redraws its kept raw lines — a deque bounded like the RichLog, which
  stores rendered strips — and its status line at the same scroll position and follow state; Rich's `Text ==`
  ignores the base style, so "unchanged, skip it" caches would keep old colours), save ui.json `theme`, then
  `refresh_data(None)` on every FleetScreen except the detail screen of a host Settings removed (still under
  Settings, it has no state; `set_fleet_layout` skips it too, as `_flush_changes` does) and repaint. Settings › About
  shows `choice → MODE` whenever the mode in effect differs from the choice, and `RUNFLEET_THEME` as the source when
  it decides.
- `runfleet doctor` prints a *terminal* block first: TERM, COLORTERM, TERM_PROGRAM, COLORFGBG, the colour system, the
  OSC 11 answer and the resulting theme, with hints for non-truecolor sessions and iTerm2's colour settings.
- Hint bar: `? help` is reserved at the right edge; low-value globals (`tab view`, `^p cmd`, `q quit`) are dropped
  first; the app relabels / hides entries for the current selection (`^k cancel` / `^k terminate`, hidden when only
  ended jobs are selected).
- `ui/render.py` pure functions returning `rich.text.Text` (`gpu_bar(value, width, thresholds) -> Text`,
  `state_badge(state, held) -> Text`, `host_chip(status, age) -> Text`, `mini_util(pct) -> Text`), unit-tested
  including `None` inputs.

### 7.12 Textual implementation notes
- **Key bursts** (slow links deliver typed keys in batches): once `App.exit()` ran, key actions and screen pushes are
  dropped; modals set their titles in `compose()` and tolerate a missing DOM in `on_mount` (a burst can dismiss them
  first); `q` on the main screen within 150 ms of a screen change was typed for that screen and does not quit.
- **Too small** (< 60×16): the placeholder always stays on top — a screen opened meanwhile is pushed below it, every
  action but `ctrl+c`/`ctrl+z` is ignored while it shows, and the resize that pops it reveals the new screen.
- **Partition selection** (cluster host detail) is one-way: rebuilding the partition table never emits RowHighlighted
  and restores the cursor to the screen's partition (a rebuild that restored the old row looped forever).
- **Log panes**: the follow badge of an ended job reads "ended · static"; a stream that stops because the host is
  unreachable / needs auth says "connection lost (…) — retrying".
- The `FleetService` (asyncio tasks per host) is started from `RunFleetApp.on_mount` via `self.run_worker(service.start(),
  exit_on_error=False)`; pollers are plain asyncio tasks owned by the service, so core is testable without Textual.
  Any `@work` used in the UI sets `exit_on_error=False` and reports failures as toasts — **a worker exception must never
  take the app down**.
- Store bridge: `store.subscribe(cb)`; `cb` accumulates `ChangeSet`s and schedules a flush at most every 250 ms
  (`set_timer`), which posts `FleetChanged(changes)` to the active screen; widgets diff from the store.
- `App.suspend()` is used only inside `RunFleetApp.run_external(argv, *, banner=None, fail_log=None) -> int | None` (§4.6);
  connect, shell and pager all go through it; `SuspendNotSupported` → modal with the command to run manually. Tests
  monkeypatch `run_external`.
- Deterministic tests/snapshots: `RunFleetApp(service, clock=ManualClock(...), animations=False)` disables the spinner
  and the 1 Hz elapsed timer.
- Global `sys.excepthook` and Textual's crash path write `~/.local/state/runfleet/crash-<ts>.txt` (traceback +
  versions, no data) and print its path after the terminal is restored.
- `LogView` = `RichLog(max_lines=20000, wrap=True, markup=False)` fed `Text.from_ansi(line)` for each `LogUpdate.lines`
  entry, plus a one-line `Static` under it for `LogUpdate.live`. No re-render of history in v0.1.
- Every key/mouse event calls `service.note_user_activity()` (App-level `on_key`/`on_mouse_down` hooks);
  `AppFocus`/`AppBlur` call `set_app_focus`.
- **Mouse driver**: `RunFleetApp.get_driver_class()` wraps Textual's Linux driver to stay in cell mode — no SGR-pixel
  reporting (1016) and no in-band resize reports (2048; switched off if the terminal already had it on). Multiplexers
  such as herdr answer the 2048 query but keep sending cell coordinates, and Textual then divided every click by the
  cell size (all clicks landed near the top-left corner).
- **Hosts added / removed at runtime** (§7.13): FleetView rebuilds its cards / table rows when the store's host list or
  order changes (card number = position); a removed host drops the scope and its detail screens close once they are on
  top (they get no `FleetChanged` / clock ticks meanwhile). The Settings modal also receives `FleetChanged`
  (`WANTS_FLEET_CHANGES`) for live state chips.

### 7.13 Settings (`ui/screens/settings.py`)
Modelled on herdr's settings popup (`settings_overlay.rs`): a centered panel 76×22 (shrinks to width − 4 × height − 2;
below 18 rows the blank lines and the description go), border in the accent colour on the dialog surface, drawn as one
Rich text: ` settings` (bold; right: the config path, or `demo — not saved` in yellow), the section tabs (active =
accent pill), a thin rule, a bold section heading with a one-line description, the choice list (selected row = accent
bar with `▸`, panel-coloured text), a message line, a key hint line and two centred buttons (`↵ save`/`↵ remove`/
`↵ change`/`↵ open` as an accent pill, `esc close` grey). Tabs, rows, checkboxes, value pills and buttons are clickable
(custom style meta + hit test, not Textual `@click`, which would recolour them as links); the wheel moves the selection.
Opened by `,` (again closes), the header `⚙` chip and the palette; `runfleet` with a config that has no hosts starts in
it. Everything goes through FleetAPI (§3.7).

- **Hosts**: rows = saved hosts (file order) + this run's ad-hoc hosts + the other `~/.ssh/config` aliases. Columns:
  `[x]`/`[ ]` (saved), name, kind (`slurm 25.05.8`, `workstation · 4 GPUs`, `?` = not probed), state (live chip if
  polled; `not in this run` for a saved host an ad-hoc run does not show; `unsaved` for ad-hoc hosts; the test result).
  `space`/`enter`/click on the box: check → `add_host` (saved + polling starts at once); uncheck → `remove_host`
  (polling stops, removed from the file); when the host shows queued or running jobs of mine the first press only says
  "k2 runs 3 jobs of yours — space again removes it from the fleet (the jobs keep running)"; likewise when the table
  has keys beyond name/ssh (`custom_host_keys`: "k2 has its own settings (protected_qos) — space again removes its
  table") or the host is saved but not in this run; checking it again in the session restores the table. Removing
  never crashes or hangs on a write error: the message line shows it. `/` and `a` start the input line in `on_key`
  itself, not via their bindings, so keys queued behind them in one terminal read are typed, not run.
  `t` → `probe_alias` (read-only, BatchMode) with the result inline and in the message line; `a` → an input line for a
  custom ssh target
  (alias, host, `user@host`; name = target with other characters as `_`), `↵` saves and polls it; `J`/`K` swap a saved
  host with its saved neighbour (`reorder_hosts`: card order, number keys, file order); `p` on a saved host this run
  left out (`not in this run`) polls it for this run with its saved table (`add_host`; the file is not written), while
  `space` on it still means remove; `/` filters (shown, with the
  position `3/40`, whenever the list is longer than the panel). A config that cannot be read shows its error in the
  message line and refuses toggles (the file is never overwritten; `e` opens it).
- **General**: default view (fleet/jobs/gpus, at next start), refresh speed (slow ×2 / normal ×1 / fast ×0.5 →
  `refresh_multiplier`, live, floors apply; `+`/`-` still adjust this run and the row shows the rate in effect),
  read-only (live), mouse (live: the driver's mouse reporting is switched on/off), idle pause (30m/2h/8h/24h →
  `idle_pause_after`, live), remember hosts; then a *notifications* group (a heading line, right: whether `[notify]
  command` is set — the hook itself is config.toml-only) with `[notify]` enabled, terminal and events (presets for
  every own job, compared order-insensitively: failures = ended_bad/restarted/time_low = the default, +gpu idle, all;
  any other list shows as `custom N events` and is written only when that row changes). Rows carry their table
  (`Setting.table`); `[notify]` goes through `set_notify` → `ConfigFile.set_table` (a missing table is created above
  the [[host]] tables and the first host's comment block) and is live because `Alerts` reads `config.notify` on every
  batch. A value not in the list shows as `custom …`. The description line explains the highlighted setting.
- **About**: config file (`↵`/`e`: `$VISUAL`/`$EDITOR`/`vi` via `run_external`; hand edits apply at the next start),
  log (`↵`: local pager), state dir, `~/.ssh/config` alias count, version. The version row's note is the update
  state (§8.6): `↵ check for updates` / `checking …` / `<v> available · ↵ update` / `newest · ↵ check again` /
  `check failed · ↵ retry`. `↵` checks (forced) unless a newer version is known; then it arms (`confirm = ":update"`,
  the message names the install's update command) and `↵` again calls `app.request_update()`; `esc` disarms.
- `--demo`: works on an in-memory host list (`config_path` None), says `demo — not saved`, writes nothing.
- The Fleet card size is not a config.toml key and has no General row (that panel is full): palette › *Fleet: roomy
  cards* / *Fleet: compact cards* switch it live and save ui.json `fleet_layout` (`auto`, the default: roomy when the
  view has room, §7.4; `compact`: always today's cards), read at start like the theme choice.
- While the panel is the top screen the base screen's HintBar is hidden (app class `-settings-open`): the panel has its
  own hint line. Right-hand notes keep at least one space to the (truncated) value.
- `F1` (and `?`) inside Settings and every other dialog opens Help with that dialog's own keys (`HELP_CONTEXT`); the
  palette and other app keys wait until the dialog closes. Help itself closes on `?`, `F1` and `esc`, never on `q`
  (typed into its search box). Compact header chips: §7.7 (FleetHeader).

### 7.14 Notifications, watch list, "since you were away", GPU watches (`core/events.py`, `ui/notify.py`, `ui/gpu_watch.py`)
For the researcher who submits a sweep, leaves RunFleet in a tmux pane and comes back hours later. No new remote
command: everything is derived from the snapshots the poller already produces.

- **Detection** (`core/events.py`, pure): `TransitionDetector` keeps the last observed own jobs (`is_mine`; tasks
  included) per host and compares the next observation: *started* (queued → active; a job or task first seen active
  that started after the baseline), *ended* (into a terminal state; carries state, exit code and the collector's short
  summary: `reason_text` / `hints[0]` / a task's `exited (log: …)`), *restarted* (`restarts` up, or back to the queue
  from active / PREEMPTED / NODE_FAIL), *time_low* (RUNNING and left < max(10 % of the limit, 10 min); once per job and
  run), *gpu_idle* (the collector's `GPU idle …` hint first raised; once per run). The first observation of a host is a
  baseline (nothing fires, already-low/idle jobs are marked fired); it is taken only from a live poll (OK/DEGRADED with
  `last_ok`, the squeue/ps section not failing), never from the warm cache; a host in UNREACHABLE / NEEDS_AUTH /
  HOSTKEY / DISABLED (or removed) is reset and baselines again when it is back. Jobs that first appear already ended
  (sacct one tick late) count only if they ended after the baseline (2 min slack). A workstation task that just ended
  while its log is scanned (EXITED with `logscan.PENDING_EXIT_HINT`, `ended; scanning log`) fires nothing yet; the
  verdict a tick later (FAILED → `ended_bad`, or EXITED → `ended_other`) fires the one *ended* event.
- **Selection**: `[notify] events` (default `ended_bad`, `time_low`, `restarted`) for every own job, plus
  `watched_events` (default `started`, `ended_ok`, `ended_other`, `gpu_idle`) for watched ones. `enabled`, `terminal`
  and `events` (as presets) can also be changed in Settings › General (§7.13), live and saved. `ended_bad` = FAILED,
  TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, PREEMPTED, BOOT_FAIL, DEADLINE; `ended_ok` = COMPLETED; `ended_other` =
  CANCELLED / EX.
- **Delivery** (`ui/notify.Alerts`, fed from `RunFleetApp._flush_changes`, i.e. ≤ 4 Hz, plus a 30 s scan for the
  time-based events): selected events go to the history (palette › *Notifications*, 200 entries, `enter` opens the
  job). With `enabled`: per batch and host, ≤ 3 events → one toast each, more → one summary (`k2: 12 failed, 3
  completed`); a terminal notification per notice (`terminal = "auto"`: OSC 9 for iTerm2 / WezTerm / Windows Terminal /
  ghostty / kitty, detected from `TERM_PROGRAM`, `LC_TERMINAL`, `KITTY_WINDOW_ID`, `WEZTERM_PANE`, `WT_SESSION`, …;
  OSC 777 for foot / urxvt / VTE; else BEL, one per batch), wrapped in tmux DCS passthrough when `$TMUX` is set
  (`set -g allow-passthrough on`). The escape is written with `driver.write` from a `call_later` callback on the event
  loop (between frames, never from a worker thread, skipped when headless). `command` runs locally, in its own session,
  stdin/stdout closed, with `RUNFLEET_EVENT` (category or `summary`), `_HOST`, `_JOB`, `_NAME`, `_STATE`, `_SUMMARY`,
  killed with its tree after `command_timeout`, at most 4 at a time; failures are logged, never raised.
- **Watch** (`w` in the Jobs view on the cursor row or the marked rows, and in job detail): `ui.json` `watch` maps
  `host:id` → last time the job was seen; refreshed while present, dropped once gone for 7 days. A watched array
  summary covers its elements. Marker `◉` before the name in the table and the detail title. `--demo`: in memory.
- **Since you were away**: `ui.json` `last_seen` = the last key/click (now when quitting), saved every 5 min and on
  exit. After a start (with a `last_seen`) the digest waits until every enabled host polled successfully after the
  start or needs the user (auth, host key, unreachable), at most 60 s, then shows one toast `since 21:40: 3 failed ·
  12 completed · 1 timeout — D to review` when anything of mine ended (sacct history included). The first key after
  `idle_after` without input does the same for the idle time (immediately, or after fresh polls if polling was idle-
  paused). `D` opens the digest: counts by end state per host, the failures (FAILED group + EX) with their summary,
  `enter` opens one, `f` shows every own job that ended since then in the Jobs view (preset set aside, `esc` clears).
  Without a last visit `D` covers `recent_window`.
- **Fleet panel** (roomy Fleet view, §7.4): *what changed* counts own ends since `digest_since` (else `recent_window`)
  plus this session's time-low warnings (`since 05:12 (you were away): 2 failed · 1 completed · 1 low on time`), then
  rows `HH:MM host glyph name summary` (`Mon 11:00` once older than 20 h, like the head's `since` label; `09-06 11:00`
  from 6 days back), columns padded by terminal cells (CJK names keep the summary column), newest first within
  `recent_window`: own ends from the store (complete, they
  survive a restart; `✖` bad end, `✓` completed with `ran H:MM:SS`, `■` other end) and the history's `time_low` `◷`,
  `restarted` `↻` and `gpu_idle` `◌` events (session only, one row per job and kind; events of hosts no longer in the
  fleet are left out of rows and counts), and GPU watches that fired, `◎` (`HH:MM host ◎ 2× H200 on k2  k2 n-38-01 3
  free`, session only, not in the count line, a click opens nothing). Rows older than `digest_since` are dim.
- **GPU availability watches** (`ui/gpu_watch.py`, pure; `Alerts` runs them; palette › *Watch free GPUs…* /
  *Watches*): "tell me when N free GPUs of type T appear (on host H / partition P)". No new remote command or cadence:
  only data already polled. *Matching* is per node, never a fleet-wide sum (a job needs its GPUs on one node): a
  cluster node counts `util/slurmfmt.node_free` (the fleet panel's count; only nodes of the listed partitions, and of
  the watch's partition when it names one), a workstation is one node of its GPUs not in use and not an old reading.
  A model is one request: the node's best matching gres type (`A100` matches `A100-80G` and `A100-40G` by
  `fleet_layout.type_label`, never `GH200`); "any GPU" adds a node's whole-GPU types up, as an untyped request may.
  MIG slices count only for a `mig` watch (any profile, or one such as `3g.40gb`, one profile per request) and never
  for a whole-GPU watch. GPUs on a node backfill planned for a pending job (`planned` state / `-` flag) are left out
  unless the watch says `planned`. *Readings*: `gpu_watch.readings` per host with the panel's freshness rule
  (`fleet_layout.free_source`: a `host ?` — problem, stale, or a failed / missing / too old sinfo — is not counted;
  a disabled host neither), a workstation also needs a successful `nvidia.gpu` sample; a reading is identified by its
  sample time (`slurm.sinfo` / `nvidia.gpu` `last_ok`) and counted once, node scans are memoised per sample.
  *Evaluation* (`gpu_watch.Evaluator`, per watch and host): a fresh reading that matches extends the host's run, one
  that does not ends it, a host without a fresh reading breaks its run (whether it holds is unknown, so it does not
  re-arm either); the watch fires once a host held on `CONSECUTIVE` = 2 readings in a row, then stays quiet for
  `COOLDOWN_S` = 15 min and is re-armed only after fresh readings show it no longer holds on any host (hosts that
  leave the fleet or the partition are forgotten). An idle pause (`store.idle_paused`) breaks every run; once per pause
  `Alerts` delivers `GPU watches paused: polling stopped after 2h without input, so N watches cannot fire — any key
  in RunFleet resumes` (no false silence), and the Watches list says `paused`. *Delivery*: like a job alert, with
  `[notify] enabled` a toast, a terminal notification (title `RunFleet · <host>`) and the hook with `RUNFLEET_EVENT`
  `gpu_free` / `gpu_watch_paused`, `_HOST` (first matching host), `_NAME` (the watch, `2× H200 on k2`), `_SUMMARY`
  (`free GPUs: 2× H200 on k2 — k2 n-38-01 3 free`), `_JOB` / `_STATE` empty; the `[notify] events` selection does
  not apply (the user asked). Always into `Alerts.watch_events` (200, session): palette › *Notifications* merges it
  with the job alerts by time (`enter` on a watch opens its host) and the fleet panel shows `◎` rows. *Prompt*
  (`gpu_watch.parse`, lenient, against the hosts, listed partitions and GPU types of the store): a count (`2`,
  `2x`, `x2`, glued `2xh200`; a bare `3090` after a count is the model), a model, `mig` or a profile
  (`a100-3g.40gb`), a host (`k2`, `@k2`, `host=k2`), a partition (`k2/gpu-preempt`, `partition gpu`, `p=gpu`, or a
  bare name some host lists; a filler word such as `gpu` that is also a partition gets a note), `planned`; filler
  words are skipped. Problems block saving (unknown word or host, two counts / types / hosts, a partition the host
  does not list, a count outside 1–64, a duplicate, more than `MAX_WATCHES` = 10); notes do not (a model not seen in
  the fleet yet). The prompt shows the reading (`watch for 2× H200 on k2`), what matches now per node (`matches
  now: k2 n-38-01 3 free`, `best now: …`, planned GPUs it leaves out) and the hosts not counted. *Watches list*: each
  watch with its state (`waiting: best now …`, `holds on k2 (1 of 2 readings in a row)`, `fired 14:03 · re-arms once
  it no longer holds`, `quiet until 14:18`, `paused`, `not counted: lyra ?`), refreshed on store batches and every
  5 s. *Storage*: `ui.json` `watches`, a list of `{count, gpu, mig, host, partition, planned}`; an entry this version
  cannot read is skipped and kept; the run / cooldown state is per session (a restart counts afresh).
- **Runway and restarts** (Jobs table, detail title): LEFT is amber below 10 % of the limit or min(30 min, 25 %)
  and red below 2 % or min(5 min, 10 %) (no limit: the minute floors alone); `↻N` before the name when `restarts` (squeue `RestartCnt`, already parsed) is above 0.

### 7.15 Everyday actions (`ui/everyday.py`, `ui/screens/forwards.py`, `core/forwards.py`)
Copy into Slack or another terminal, open TensorBoard / Jupyter of a job, log in after the overnight disconnect, grab
free GPUs, act on a whole array. None mutates a job (all work under `--readonly`); remote commands run only on the
keypress, as ordinary on-demand probes under their timeouts.

- **Copy menu** (`Y`; `y` stays "copy id"): an `ActionMenuScreen` (title "Copy · k2:2332861") of `CopyItem`s — stdout /
  stderr path (merged → the stdout file), job id, `host:jobid`, workdir, the paste-able shell command
  (`ssh -t <alias> '<FleetAPI.job_shell_command>'`, the command `s` runs: `srun --jobid … --pty` on the batch host for
  a running Slurm job) and the support bundle. Stdout comes first (highlighted; disabled entries sink, the first
  enabled one is highlighted). Accelerators `COPY_KEYS`: `o` stdout · `e` stderr · `i` id · `h` host:jobid ·
  `w` workdir · `s` shell command · `b` bundle, plus `Y` → stdout so `Y Y` keeps the old one-key "copy stdout path".
  `ActionMenuScreen(keys={key: action_id})` is generic and optional: the first key of an entry is drawn in front of
  it, a key picks the entry at once (a disabled one toasts its reason and the menu stays), keys of the menu itself
  (`j`/`k`/`q`/`esc`/`F1`/`↵`/cursor keys, `RESERVED_KEYS`) raise `ValueError`; menus without `keys` are unchanged. Copy = `App.copy_to_clipboard` (OSC 52) plus, under `$TMUX`,
  `tmux load-buffer -w -` in a thread (2 s timeout, falls back to without `-w`, failures ignored); the toast names what
  was copied. **Support bundle**: `job_detail()` (cached ≤ 30 s) + log lines RunFleet holds (what job detail streamed, up to
  `TRIAGE_LINES`; preview peeks), else `fetch_log(16 KiB)`; Markdown with cluster, id, user, partition/QOS/account, request, state + exit +
  explanation, the cause chip (`logscan.cause_of(job, detail, lines).plain()`, §5.2.8) and the efficiency card
  (`JobDetail.efficiency`) when there is one,
  nodes with GPU IDX, restarts, the last 30 lines; `scrub()` removes 40-hex wandb keys, `hf_`, `sk-`, `gh?_` tokens,
  AWS key ids / secrets, `token=` values, PEM private-key blocks (to the END line or the end of the fence), `Bearer` /
  `Basic` credentials, URL userinfo and the values of `NAME=` / `NAME:` / JSON assignments whose name ends in
  `password|passwd|passphrase|secret|token|api_key|access_key` (`max_tokens` stays); also written 0600 to `<state>/bundles/<host>-<id>.md` (not in `--demo`).
- **Port forwards** (`P`, palette *Forward a port…*, *Forwards*): node = the running job's batch host, `localhost` for
  tasks and host-level forwards; refused with the reason from `FleetAPI.forward_blocker` (host not OK/DEGRADED, needs
  auth, or `Transport.forward_blocker()`: no master — user mode without ControlPath, `mux = "off"`, `session`
  (until P3), local host).
  Candidates: `find_urls` over the log text RunFleet holds, else the last 1 MiB of stdout/stderr (one `fetch_log`
  each); one per port, a URL with path/token wins; labels by keyword or well-known port; plus *Other port…*
  (`PortPromptScreen`). `forward_port` picks a local port (the remote number when it binds on 127.0.0.1 and no forward
  of ours uses it, else an ephemeral one; one retry if ssh loses the race) and runs `Transport.forward` = `ssh -O
  forward -L 127.0.0.1:<lport>:<node>:<rport>` with the master's `check_opts` (local only, `CHECK_TIMEOUT`); the URL
  `http(s)://localhost:<lport><path?query>` is copied. `ForwardsScreen` lists `FleetAPI.forwards()` (`↵` copy, `^k`
  `-O cancel`). `FleetService.stop()` and host removal cancel every forward (a persisting master would keep them).
  Every `-O forward|cancel` runs in a shielded task tracked in `_forward_ops`: a cancelled caller (a new `P`, the
  screen closed) cannot leave an unrecorded listener — the task records it and `stop()` waits for in-flight ones
  before cancelling. `cancel_forward` drops the record only when ssh cancelled it or the master is gone (no control
  socket); otherwise it raises and the forward stays listed. The same (host, node, port) again re-sends `-O forward`
  (OpenSSH answers an existing forward with OK, a replaced master binds it anew) before returning the record.
  Demo / fake transports simulate it (the demo's sampler job prints a TensorBoard URL).
- **Log in to all** (`C` in Fleet, palette): hosts in NEEDS_AUTH / HOSTKEY / UNREACHABLE (enabled, not connecting):
  `prepare_connect` each (already up → `after_connect(0)`), then **one** suspend — `RunFleetApp.run_external_sequence`
  runs each `connect_argv` after a banner `2/3 vega — approve the Duo push` (Ctrl-C = rc 130 skips, a failure prints
  the ssh log tail, Enter once at the end if any failed) — then `after_connect(host, rc)` each and one summary toast.
- **Free GPUs** (`a` in GPUs view / workstation host detail): host = the cursor's; refused for clusters and stale
  readings; free = no process, < 1 GiB used, < 5 % util, ranked by memory, util, index; a menu `N free GPUs →
  CUDA_VISIBLE_DEVICES=…` copies the first N; the toast names the holders of the others ("GPU 1 alice; GPU 3 you
  (p363942)").
- **Whole array** (`A`, Jobs view): marks the visible rows with the cursor row's host and `array_job_id` (summary rows
  included; rows hidden by filter / fold are not), so cancel / hold / the action menu act on the array; again clears.

### 7.16 Attach — live readings of a running Slurm job (v0.2)
Evidence: the Attach field probes (ssh-to-node and `srun --overlap` reads on all three clusters, with timings,
accounting traces and a leftover audit after each; anonymised cases in `tests/fixtures/attach/`). D7 is the rule: **HostPoller never touches a
compute node**; Attach reads ONE job's node(s) on demand, only while the user looks at that job's GPU or Procs tab.

**Phase-2 file ownership** (three agents in parallel; contracts in §3.3/§3.6/§3.7 are frozen for them):
- *backend* — `backends/slurm/attach.py` (its constants are contract: section names, timeouts, thresholds),
  `backends/slurm/parse_attach.py`, `backends/common/cgroup.py`, the three `SchedulerCollector` attach methods in
  `backends/slurm/collector.py`, GPU-type feedback (§5.2.4 jobmeta), `demo/{world,render,transport}.py` attach
  answers, fixtures `tests/fixtures/attach/` + `tests/backends/slurm/test_attach*.py`, demo tests.
- *core* — `core/attach.py` (AttachService), `core/service.py` (replace the placeholder; `job_detail` fills
  `JobDetail.live`), `core/poller.py` (apply a caps_patch outside a tick: store caps + cache), `probe/framing.py`
  (already has `render(tmp=False)` and purpose `"attach"`), `debugdump.py`, `tests/unit/test_attach*.py`,
  `tests/unit/stubs.py` (StubSlurm attach already present).
- *ui* — `ui/screens/job_detail.py` (GPU/Procs tabs), `ui/widgets/attach_panel.py`, `ui/widgets/proc_table.py` (STEP
  column), `ui/screens/confirm.py` (three-choice srun confirmation), `ui/keymap.py` (+ generated USER_GUIDE section),
  `ui/messages.py`, `tests/ui/fakes.py` (FakeService attach already present), `tests/ui/test_attach*.py`, snapshots.
- Shared, already done in phase 1, change only by editing this section first: `models.py`, `errors.py`,
  `backends/base.py`, `core/store.py`, `config.py`, `docs/CONFIG.md`, `tests/unit/test_contracts.py`.

#### 7.16.1 Modes and policy
| Mode | Path | Trace | Where |
|---|---|---|---|
| `ssh` | login host → `ssh <node>`; pam_slurm_adopt puts the session into the user's job on that node (the **newest** one when several: the user cannot choose) | none in accounting (session CPU counts to the adopted job's `.extern`); a login line in the node's sshd log | k2, vega |
| `srun` | `srun --jobid=J --overlap` one-shot step on the job's node; inherits the job's GPUs | **ONE permanent sacct row** `J.N rfprobe COMPLETED` per sample, not deletable | the only path on lyra (the default `ssh+srun` switches to it); offered on k2/vega as a fallback |
| `exporter` | *future only*: one HTTP read of a node exporter (e.g. a jobstats / Prometheus exporter, where a site runs one) | none on the node | only with the site's permission; not in `AttachMode` |

Config (per host, flat key or `[defaults.slurm]`, §8.2): `attach = "ssh" | "ssh+srun" | "srun" | "off"` (default
`"ssh+srun"`), `attach_refresh` (duration, default `15s`, floor `10s`, ssh only). Core policy, applied to the backend's
technical modes (`SchedulerCollector.attach_modes`: RUNNING job with nodes; no `ssh` when `caps.node_ssh ==
"denied"`; no `srun` without srun/--overlap):

| setting | ssh possible | srun possible | result (`AttachState`) |
|---|---|---|---|
| `off` | – | – | BLOCKED "attach is off for <host>" |
| `ssh` | yes | – | mode ssh: LIVE (auto) |
| `ssh` | no (denied) | (yes) | BLOCKED "<host> refuses ssh to compute nodes" + `hint` = `use attach = "ssh+srun" (the default) for <host> to read via srun; each read adds one permanent sacct row` (only an explicit `attach = "ssh"` gets here) |
| `ssh+srun` | yes | yes | mode ssh: LIVE; `offer` (srun) appears when ssh lands in another job, or when the last ssh sample could not reach a node (HOSTKEY / DNS / REFUSED hop error; reason = that error's summary) — consent only, never run by itself |
| `ssh+srun` / `srun` | no / – | yes | mode srun: NEEDS_CONSENT (IDLE once consent = "job"); the refused ssh sample that led here is the offer's reason, not `error` |
| any srun-only case under `--readonly` | | | BLOCKED "read-only mode: srun adds a permanent sacct row" (no offer) |
| job not RUNNING / not Slurm / host not OK·DEGRADED | | | BLOCKED with that reason |

`node_ssh` is learned from samples (ingest_attach's caps_patch: `"adopt"` after a landing, `"denied"` after an auth
refusal), stored in the host's Capabilities and the cache like every capability (24 h; a re-detect forgets it and the
next sample relearns it). When `"denied"` and the setting allows srun, Attach goes straight to the srun offer.

#### 7.16.2 Lifecycle, refresh, autostop, consent (core, `core/attach.py`)
- **One attach per app.** `attach_start(key)` (the UI calls it when the GPU or Procs tab of `key` becomes visible,
  and on screen resume) stops any other attach (its state → STOPPED "another job attached"). `attach_stop` on tab
  hide / screen pause / dismiss (→ STOPPED "tab closed"; live data, consent and srun_rows stay for the session).
- **ssh auto-refresh** runs only while ALL hold: the tab is open (started, not stopped), mode ssh, `auto` on (`a`
  toggles, default on), the app is focused (`set_app_focus`), polling is not idle-paused, the job is RUNNING in the
  store, the host is OK/DEGRADED, and there was user input in the last **10 min** (`AUTOSTOP_S`). The first sample
  runs immediately on attach_start; the next one `attach_refresh` after the previous one ENDED (never overlapping).
  Blur / idle pause → PAUSED (resumes by itself, sampling at once if overdue); 10 min without input → STOPPED "no input
  for 10 min — press r"; job leaves RUNNING → STOPPED "job is no longer running"; host removed/disabled → STOPPED.
  A failed auto sample backs off (next after `min(2 × attach_refresh, 60 s)`); 3 consecutive failures → STOPPED.
  An auto sample re-checks these rules once it holds the on-demand lane, right before dispatch (the `recheck` of
  `_on_demand`): if one no longer holds (blurred, idle-paused, auto off, tab closed, job ended, autostop) the sample is
  dropped — nothing runs, nothing is recorded (no failure, no error), the state goes where the rules say. A user `r`
  that joins an auto sample in flight makes it the user's (no auto recheck; closing the tab no longer cancels it).
- **srun never auto-refreshes** and is never started by a timer or by opening a tab: only `attach_now` (the user's
  `r`). Without consent `attach_now` raises `AttachConsentRequired(key, offer)`; the UI shows `offer.title/body/preview`
  with the choices **Once** (`consent="once"`) / **Allow for this job (this session)** (`consent="job"`, remembered
  in memory for this key until exit, never persisted) / **Cancel**. There is no "never ask" option.
- `attach_now(key, mode=None)` uses the state's mode; `mode="srun"` is the explicit "read job J's GPUs with srun"
  offer (ssh landed elsewhere / refused / could not reach the node). In ssh mode a sample younger than 3 s **of the
  same node selection** (`node=` or the pin, None = backend default) is returned as is, an in-flight sample of the same
  job, mode and node selection is awaited (another mode or node waits for it, then probes its own), and `r` re-arms
  auto-refresh (also after STOPPED) if the tab is open. Changing the pin (`attach_start(key, node=…)`) cancels an auto
  sample of the old pin in flight and drops the old pin's offers.
- **Shutdown / host removal**: `close()` and `forget_host` mark the service / entries closed BEFORE cancelling their
  tasks; a request waiting behind another sample re-checks after every await and raises `AttachBlocked` instead of
  starting a probe (no srun step after shutdown), and nothing is spawned for a closed service or a dropped entry.
- Samples use the host's **on-demand lane** through `FleetService._on_demand` (master check, lane slot, global
  semaphore; serialized with detail / log peeks on user-master hosts), one probe `framing.make_script(plan.sections,
  "attach")`, local timeout `framing.local_timeout` (≤ 90 s).
- Core ALWAYS calls `ingest_attach(plan, res.sections, ctx, previous=state.live)` — with `idle_streak` emptied when
  the last sample had no data (a failed read ends "N samples in a row"; the counters stay for the `cpu_cores` delta)
  — (every planned section is present,
  NOT_RUN / INCOMPLETE / TIMEOUT when it did not finish, so srun rows are counted uniformly) and appends a
  transport-level `res.error` to `JobLive.errors`. It applies the caps_patch through the poller (store caps + cache),
  adds `JobLive.sacct_rows_added` to `AttachState.srun_rows`, sets `live` only when `JobLive.has_data` (else `error`),
  and calls `FleetStore.notify_live(host, [key.id])` on EVERY state change (status, live, next_at, error, offer).
- `job_detail(key)` returns `dataclasses.replace(detail, live=state.live)` when an AttachState with data exists.

#### 7.16.3 Probe shape (backend, `backends/slurm/attach.py`)
One login probe per sample, sections in this order:

| Section (owner `slurm`) | Command (data values validated + `shlex.quote`d) | Timeout |
|---|---|---|
| `slurm.attach.node<i>` (ssh), `i` = index into `AttachPlan.nodes` | `timeout -k 2 12 ssh -T -a -x -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o "GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts /etc/ssh/ssh_known_hosts2 $HOME/.ssh/known_hosts $HOME/.ssh/known_hosts2" -o UpdateHostKeys=no -o ClearAllForwardings=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 -o LogLevel=ERROR -- <node> "timeout -k 1 10 sh -c 'echo <b64> \| base64 -d \| sh'"` | 14 |
| `slurm.attach.node0` (srun; one node only) | `timeout -k 2 25 srun --jobid=<J> --overlap --job-name=rfprobe -N1 -n1 -w <node> --time=1 --quiet --input=none timeout -k 1 15 sh -c 'echo <b64> \| base64 -d \| sh'` | 27 |
| `slurm.attach.audit` (late, last) | `echo @@steps; squeue -h -s -j <J> -o '%i %j' 2>&1; echo @@ssh; pgrep -u "$(id -un)" -f 'ControlPat[h]=none' \| wc -l; true` (`%T` is invalid with `-s` on all three clusters; the `[h]` keeps pgrep from matching this shell) | 10 |

`<b64>` = base64 of `framing.render(inner, tmp=False)` where `inner` = `AttachPlan.inner[section name]`, a nested
ProbeScript (own nonce, purpose `"attach"`) whose framed output is the outer section's stdout (parse it with
`framing.parse(stdout, inner)`; outer markers use another nonce, so they never collide). ssh mode samples up to
`MAX_SSH_NODES = 4` nodes (batch host first; `node=` pins one); srun mode exactly one node (default batch host),
because every step is one permanent row. `AttachPlan.preview` = node0's exact command with the ~6 KB base64 blob
abbreviated to `<read-only node script, N bytes base64>`; `sacct_rows` = 1 for srun. "srun possible" (attach_modes)
= `--overlap` (Slurm ≥ 20.11) and `srun` in `caps.tools`; `detect.tools` now also looks for `srun` and `ssh`, and a
tool list detected before v0.2 (no `ssh` entry) counts srun as possible (a missing srun is then the section's MISSING).

Nested node sections (owner `node`, each `timeout -k 2 5` via framing; order fixed, `NODE_SECTIONS`):

| Section | Command | Gives |
|---|---|---|
| `node.id` | `hostname -s; cat /proc/self/cgroup; echo "J=${SLURM_JOB_ID:-} G=${SLURM_STEP_GPUS:-}"` | node, landed job, step GPUs (srun) |
| `node.cg` | per job of `meta.jobs_raw` (target first, then the user's other running jobs on that node): `S` = the `*slurmstepd.scope` part of the `0::` line of /proc/self/cgroup (`sed -n -E`; fallbacks `<host>_slurmstepd.scope`, `slurmstepd.scope`), try `/sys/fs/cgroup$S/job_<J>` then `/sys/fs/cgroup$S/<SLUID>`; one block per job: `@@cg <J> <dir>` (`-` = not found), `<file> <value>` for `memory.current memory.peak memory.max`, `memory.events` as is, `grep -E '^(anon\|file\|shmem) ' memory.stat`, `grep '^usage_usec ' cpu.stat` | CgroupStats per job |
| `node.ps` | `echo "$$"; ps -ww -u "$(id -un)" -o pid=,ppid=,pgid=,etimes=,pcpu=,pmem=,rss=,stat=,args=` | Proc rows (user = caps.remote_user) minus the probe's own tree: every own ancestor of the echoed shell (the chain ends at a root-owned sshd / slurmstepd) and all descendants of the topmost one |
| `node.pids` | `for j in <jobs_raw>; do scontrol listpids "$j"; done` | step_of_pid (`"<J>.<step>"`) |
| `node.map` | `grep -H -E 'GPU UUID\|Device Minor\|Bus Location' /proc/driver/nvidia/gpus/*/information` | uuid ↔ bus ↔ minor of every GPU |
| `node.gpu` (late) | `nvidia.wrap("nvidia-smi --query-gpu=<GPU_FIELDS + pci.bus_id> --format=csv,noheader,nounits")` (RF_T empty → foreground; the guard refuses when two of the user's nvidia-smi are already stuck — D state or held by an abandoned `rf_nv` wrapper — rc 125; a job's healthy `nvidia-smi -l` does not count, §5.3.1) | GPU rows |
| `node.apps` (late) | `nvidia.wrap("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")` | GPUProcess rows (joined with node.ps) |
| `node.self` (late, last) | `rf_s() { p=$(ps -ww -u "$(id -un)" -o pid=,ppid=,stat=,etimes=,args=) \|\| return 1; printf '%s\n' "$p" \| awk '{c=$5; sub(/.*\//, "", c)} c == "nvidia-smi"'; }`; `a=$(rf_s) \|\| exit 1`, print `$a`; if a row of `$a` is in D state: `sleep 1`, `echo @@again` and a second `rf_s`; last line `echo @@done` (`attach.NODE_SELF_CMD`) | `NodeInspect.stray_nvidia_smi` = pids in D state on both reads (hung: survives every signal; a query our timeouts ended can only remain this way); `other_nvidia_smi` = the rest of the last read (a job's `nvidia-smi -l`, a query in flight: informational); both None without `@@done` / with a D row lacking its second read (no affirmative answer) |

`meta` (Appendix A) is what the demo/fake transports need to answer without parsing `cmd`.

#### 7.16.4 No leftovers: the five layers (every one is required)
1. **Login-side section**: framing already runs it as `timeout -k 2 T sh -c … </dev/null` (srun/ssh cannot eat the
   script). The login probe (purpose `"attach"`) is rendered without a temp dir (`render` default for that purpose,
   §4.3): no `mktemp`, nothing written on the login host; each hop's stderr is captured in a variable.
2. **login → node hop**: `ssh -T -a -x`, `BatchMode=yes`, **`ControlMaster=no ControlPath=none`** (no socket and no
   ControlPersist process left on the login host), **`StrictHostKeyChecking=accept-new`** with
   **`UserKnownHostsFile=/dev/null`** and the login host's system and user known_hosts files as
   `GlobalKnownHostsFile` (a node key already recorded there must match — a changed key is refused; a node never
   visited is accepted but recorded nowhere, since sites commonly ship no node keys and run `StrictHostKeyChecking no`
   themselves), **`UpdateHostKeys=no`** (nothing appended to the login host's `~/.ssh/known_hosts`), `ClearAllForwardings=yes`,
   `ConnectTimeout=5`, `ServerAliveInterval=5 ServerAliveCountMax=1`, `LogLevel=ERROR`, all under `timeout -k 2 12`.
3. **Node-side script**: read-only; no background processes, no temp dir (`render(tmp=False)`); each section `timeout
   -k 2 5`; the whole script under **`timeout -k 1 10`** on the node itself (a dropped `ssh -T` sends no SIGHUP, so a
   local timeout alone is not enough).
4. **srun step**: exactly `srun --jobid=J --overlap --job-name=rfprobe -N1 -n1 -w NODE --time=1 --quiet --input=none
   timeout -k 1 15 …` under `timeout -k 2 25`; slurmstepd kills the step cgroup at the end and `--time=1` makes
   slurmctld reap it within a minute. **Never `--gres=none`** (the step would see no GPU: "No devices found"),
   **never `--pty`**, never `--exclusive`, never from a timer.
5. **Audit** (`slurm.attach.audit`, same probe, last): clean = no live `rfprobe` step in `squeue -s`, 0 ssh processes
   with `ControlPath=none` on the login host, and an **affirmative** `node.self` with `stray_nvidia_smi == 0` from
   every node the script may have run on: every node with inner markers, and every hop without output that does not
   prove nothing started there — cut by its timeout (`ssh`/`srun timed out`), or any other failure such as a
   connection reset after the remote command began (`ssh failed (rc 255)`). A node whose node.self did not finish,
   failed or is unreadable makes the verdict None, `audit_detail` = `audit inconclusive: no leftover check from
   n-37-08 (node.self did not finish)`. Only hops that provably ran nothing need no check: refused before a session
   (Permission denied, pam_slurm_adopt's message, host key, DNS, `connect to host … port 22: …`), srun that created
   no step, a section that never ran. A bare rc 255 is inconclusive: usually pam_slurm_adopt under LogLevel=ERROR,
   but ssh also exits 255 silently when killed after the remote command started.
   `JobLive.clean`: True / False (with `audit_detail`, logged at WARNING; any leftover found wins over a missing
   answer) / None (audit did not run or inconclusive). The user's healthy nvidia-smi (a job's `nvidia-smi -l`) are
   `other_nvidia_smi`, shown as a dim note, never a leftover. The UI shows `clean ✓ 17:39:15` or a red `leftovers`
   badge with the detail. Residual risk: a D-state nvidia-smi survives every signal; it stays in the job's cgroup
   until the job ends and is reported ("hung nvidia-smi on <node>"); the guard prevents stacking.

#### 7.16.5 Ingest rules (backend, `ingest_attach`)
- **Landed job**: the `job_<N>` component of `/proc/self/cgroup` (or the SLUID directory, mapped back with
  `meta.sluids`), else `$SLURM_JOB_ID`; srun: the target. `gpus_of_target` = landed == target (`job_id_raw`).
  When ssh landed elsewhere the GPU rows are the landed job's (labelled so), cgroups/ps/listpids of the target are
  still valid (k2 can read every own job's cgroup on the node).
- **IDX mapping order** (`GPU.slurm_idx`): (1) srun: `SLURM_STEP_GPUS` paired with the GPUs by **uuid** (nvidia-smi
  index inside a step is always 0..n-1); (2) ssh: `/proc/driver/nvidia` Device Minor = IDX (verified on k2 only),
  paired by uuid/bus; (3) else None (the UI shows bus id and uuid, never a guessed IDX). Never pair by index or PCI
  order (lyra: IDX0 = bus 71:00.0, IDX1 = bus 01:00.0).
- GPUs: `GPU.host` = RunFleet host, `bus_id`, `slurm_idx`, `processes` from node.apps joined with node.ps
  (`GPUProcess.job` = JobKey of the pid's job when listpids knows it). `procs` = node.ps rows (`task=None`).
- **CPU cores** (`CgroupStats.cpu_cores`) = Δusage_usec / Δsampled_at against `previous` for the same (node, job),
  None on the first sample or when Δt < 1 s.
- **Hints** (`JobLive.hints`, target job only; texts ready to show): *GPU idle* — a GPU of the target with util < 5 %
  and memory used > 1 GiB on ≥ 2 consecutive samples (`idle_streak` by uuid, else bus id, carried from `previous`):
  `GPU1 idle (0%) but holding 82.7G` (amounts via `util.humanize.gib`, MiB/1024, like the GPU rows; without a
  Slurm IDX the label is the bus id, `GPU 0001:C5:00.0`); *OOM kills* — target cgroup `oom_kill > 0`: `oom_kill 2 — a process of this
  job was killed for memory`; *memory* — `mem_current > 0.9 × mem_max`: `memory 93% of the job's limit (incl. shmem
  40.6G)`.
- **GPU type feedback**: nvidia-smi `name` + `memory.total` of the target's GPUs become the job's `alloc.gpu_type` with
  `gpu_type_source="nvidia-smi"` on the following ticks (collector state + jobmeta; fixes lyra `rtx6000` = RTX PRO
  6000 Blackwell 96 GB, Attach field probes).
- **caps_patch**: `{"node_ssh": "adopt"}` when an ssh hop landed (node.id ran); `{"node_ssh": "denied"}` when ssh
  failed with `Permission denied (` before any inner marker; nothing for pam_slurm_adopt "no active jobs" (stale
  node), host-key failures, timeouts.
- **sacct rows**: srun plan rows count unless the srun section is NOT_RUN / MISSING or srun's stderr says no step was
  created.
- Errors are `HostError`s with `section` = the outer section name; parse errors never raise.

#### 7.16.6 Failures and fallback
| Situation | Shown / done |
|---|---|
| ssh refused before PAM (lyra) | `node_ssh="denied"`; `ssh+srun`/`srun` → NEEDS_CONSENT offer; `ssh` → BLOCKED + hint |
| ssh lands in another own job (k2, several jobs on a node) | sample kept; GPU rows "GPUs of job 2332863 (ssh can only enter the newest job on n-40-01)"; srun offer if allowed |
| pam_slurm_adopt "you have no active jobs on this node" | error "no job of yours on <node> — job moved or ended?"; core wakes `slurm.jobs`. With `LogLevel=ERROR` OpenSSH hides that banner and "Connection closed": a bare rc 255 without a message reads "ssh to <node> closed without a message — no job of yours there?" (kind SSH) |
| `Host key verification failed` (a recorded node key changed) | error (kind HOSTKEY) "<node>'s host key changed (login host's known_hosts)" — `ssh-keygen -R <node>` there if the node was reinstalled; srun offer (consent, reason = the error summary) if allowed; the same for DNS / REFUSED hop errors; ssh keeps its normal backoff / 3-failure stop |
| nvidia-smi hung (rc 124/125) | GPU rows missing, error "nvidia-smi hung on <node>"; audit dirty if stuck instances remain (D state on both node.self reads), inconclusive when the node timeout cut node.self |
| srun cannot create the step | error with srun's stderr; no row counted |
| login transport failure | JobLive without nodes + error; auto backoff; 3 failures → STOPPED |
| no live data at all | the GPU tab keeps the v0.1 sstat view (vega shows `Σ100% / 2 GPUs`, never "50%") |

#### 7.16.7 UI (`ui/screens/job_detail.py`, `ui/widgets/attach_panel.py`)
- Status line on the GPU tab: `Live · ssh n-37-08 (job 724655) · 21:56:38 · 0.67 s · every 15 s · clean ✓ · r now ·
  a auto on/off`; srun: `Live · srun (1 sample) · 2 sacct rows added this session · r read again (asks)`; paused /
  stopped / blocked / needs-consent states show `AttachState.stopped / blocked / hint / offer.reason`.
  In ssh mode any `offer` adds the line `[<offer.reason> · ]S reads job <id>'s own GPUs with srun on <node> (asks
  first; one permanent sacct row)` — the reason is left out when ssh landed in another job (the GPU rows say so).
- Body (reuses `widgets/gpu_panel.py` bars): per GPU `IDX` (slurm_idx or `?`), short bus id, util bar, memory bar
  used/total, temp, power/limit, `! idle but holding 82.7G` when the GPU's idle_streak ≥ 2; a process line per GPU;
  the cgroup line `mem 143.7 / 274.9G (peak 202.2, shmem 60.4) · oom_kill 0 · CPU ≈ 3.6 / 16 cores`; the sstat line
  `accounting (sstat, 30 s) Σ100% / 2 GPUs · gpumem 176160M`; hints.
- Procs tab: `widgets/proc_table.py` rows from `NodeInspect.procs` with a STEP column (step_of_pid); pids repeat
  across nodes, so a sample with several nodes keys rows, GPU joins and steps by (node, pid) and adds a NODE column
  (one node: plain pid keys, no NODE column); D-state and
  zombies highlighted; a note line when ssh landed in another job.
- Keys (job_detail, keymap.py is the single source): `r` on the GPU/Procs tab of a running Slurm job = `attach_now`
  (plus the detail reload); `a` = `attach_set_auto` toggle. The srun confirmation is a three-choice ConfirmScreen.
- Updates: `FleetChanged.live_keys` (from `ChangeSet.live`) → re-read `attach_state(key)`; next-sample countdown via
  the 1 Hz ClockTick.

#### 7.16.8 Read-only, demo, debug
- `--readonly`: ssh mode allowed (read-only on the node); srun refused everywhere (it writes accounting): no offer,
  BLOCKED with the reason.
- `--demo`: DemoTransport answers the attach sections from `meta` with byte-exact nested framed output (it renders
  the nested script's framing with `meta.inner_nonce`/`meta.inner`): a k2-like node with 3 jobs (ssh lands in the
  newest), a vega-like 2-GPU job with GPU1 at 0 % holding 85 GB, an lyra-like host refusing ssh (`Permission denied
  (publickey,hostbased).`, rc 255) where srun works; audit clean. Nothing is written; srun rows are counted for the
  session only.
- `--debug` dumps attach probes like any other (purpose `attach`); they become fixtures under
  `tests/fixtures/attach/` (cases: k2 3 jobs on one node, vega one idle GPU, lyra SLUID cgroup path, `--gres=none` "No
  devices found", pam_slurm_adopt deny, lyra publickey deny).

---

## 8. Configuration

### 8.1 Files
| Path (platformdirs) | Purpose |
|---|---|
| `~/.config/runfleet/config.toml` | user config (optional) |
| `~/.local/state/runfleet/cache.sqlite` | cache (§5.5) |
| `~/.local/state/runfleet/ui.json` | view, sort, filter text, state preset, grouping, saved Jobs views (`views`, `active_view`, §7.6), theme (auto/dark/light/ansi), Fleet card size (`fleet_layout`: auto/compact), watched jobs and the last visit (`watch`, `last_seen`), GPU availability watches (`watches`, §7.14) |
| `~/.local/state/runfleet/update.json` | update check (§8.6): newest version on main, when asked, the last version announced |
| `~/.local/state/runfleet/runfleet.log` | rotating log (5 MB × 3); `--debug` raises the level and adds `debug/` raw dumps; `--demo` records carry `[demo]` |
| `~/.local/state/runfleet/ssh-<host>.log` | `-E` log of master creation / connect / the session ssh (§4.2, §4.8) |
| `$XDG_RUNTIME_DIR/runfleet/` (else `~/.cache/runfleet/cm/`) | managed ControlMaster sockets, per-host `instance.lock.*` (0700) |

The state dir and `debug/` are created 0700 and every file RunFleet writes there (log and rotations, cache, `ui.json`,
`ssh-<host>.log`, dumps) 0600, whatever the umask: they hold job names, paths and command lines.

**Windows** (`paths.py`, platformdirs with `appauthor=False`, so one `runfleet` level): config
`%APPDATA%\runfleet\config.toml` (roaming), state (cache, `ui.json`, `update.json`, log, ssh logs, bundles, `debug\`)
`%LOCALAPPDATA%\runfleet\`, instance locks `%LOCALAPPDATA%\runfleet\Cache\cm\` (no sockets: no ControlMaster on
Windows, §4.1; no 86-byte limit and no `/tmp` fallback). The session transport (§4.8) writes nothing locally beyond
`ssh-<host>.log` (`-E`, truncated per session start) and remotely only its own `rfa.XXXXXX` temp dir, removed when the
session ends. Modes are not set there (the profile's ACL keeps it private). Instance and
config-edit locks use `msvcrt.locking` on one sentinel byte at 1 MiB (`util/platform.try_lock`), so the pid at
offset 0 stays readable to a second instance; Windows refuses to unlink the open `.config.toml.lock`, so that file
stays (harmless). Host names Windows cannot use as file names (`con`, `nul`, `aux`, `com1`…`lpt9`, a trailing dot)
are a config error there; names made from ssh aliases get a `_`. A log rotation that cannot rename the file (another
instance has it open) keeps writing to it instead of printing tracebacks over the TUI.

### 8.2 Schema by example (three clusters and two workstations, as in the demo fleet)

```toml
# ~/.config/runfleet/config.toml — every key is optional except host.name
[general]
readonly = false
mouse = true
default_view = "fleet"         # fleet | jobs | gpus
recent_window = "24h"          # ended jobs/tasks stay in the default "fresh" filter this long
stall_after = "30m"            # "log idle" hint threshold
idle_after = "10m"             # no input → poll 4× slower
idle_pause_after = "2h"        # no input → pause all polling until a key is pressed
max_parallel_ssh = 8
refresh_multiplier = 1.0
pager = ""                     # default: $PAGER, else "less -R +G"
remember_hosts = true          # `runfleet k2` saves k2 here once it answers (Settings › General)
update_check = true            # once a day, ask GitHub for the newest RunFleet version (§8.6)

[ui]
gpu_util_thresholds = [10, 75]
gpu_mem_thresholds = [10, 80]
jobs_columns = ["host", "id", "name", "state", "elapsed", "left", "gpus", "gpu_util", "node_reason"]

[defaults.slurm]               # applied to every host whose profile is slurm (→ HostConfig.slurm)
poll_jobs = "15s"              # floor 10s
poll_history = "2m"            # floor 60s
poll_nodes = "2m"              # floor 60s
poll_usage = "60s"             # floor 30s (never below JobAcctGatherFrequency)
poll_logstat = "2m"            # floor 60s
sacct_window = "3d"
log_templates = []             # e.g. ["{workdir}/logs/{name}_{jobid}.out"]
attach = "ssh+srun"            # Attach (§7.16): "ssh" | "ssh+srun" | "srun" | "off"
attach_refresh = "15s"         # floor 10s; ssh auto-refresh only (srun never auto-refreshes)

[defaults.workstation]         # → HostConfig.workstation
poll_gpu = "5s"                # floor 2s
poll_procs = "10s"             # floor 5s
poll_logstat = "60s"           # floor 30s
task_min_age = "60s"
task_min_cpu = 5.0             # percent
ignore = []                    # regexes appended to the built-in noise list (§5.3.2)
kill_grace = "10s"

[[host]]
name = "k2"                    # display name; also the ssh alias unless `ssh` is set
protected_partitions = ["labpi"]
protected_qos = ["labpi"]
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]   # a common log-path convention

[[host]]
name = "vega"
connect_timeout = "120s"       # cold login can take minutes (GSSAPI/network-FS/MOTD)
log_templates = ["{workdir}/logs/{name}_{jobid}.out"]

[[host]]
name = "lyra"                  # ~/.ssh/config: ControlMaster false, no ControlPath, ProxyJump jumpvm → mux "managed"
attach = "srun"                # sshd refuses node logins (ssh+srun learns it too); every live read asks first (§7.16)

[[host]]
name = "labstation"            # ProxyJump jumpvm, user ControlPath → mux "user"

[[host]]
name = "lynx"

# [[host]]
# name = "atlas"
# ssh = "local"                # this machine, no ssh
```

### 8.3 Host keys

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | required | unique, `[A-Za-z0-9_.-]+`; order of `[[host]]` = scope numbers |
| `ssh` | str | `name` | ssh alias or `"local"` |
| `enabled` | bool | true | |
| `backend` | `auto`\|`slurm`\|`workstation` | auto | detection (§5.1) |
| `color` | str | auto palette | host colour in tables |
| `group` | str | "" | optional label shown on cards ("clusters", "workstations") |
| `mux` | `auto`\|`user`\|`managed`\|`off`\|`session` | auto | §4.1; `session` = one ssh + agent (§4.8), opt-in on Linux/macOS, the default on Windows |
| `connect_timeout` | duration | 45s | `ssh -fN` master creation |
| `ssh_options` | list[str] | [] | extra `-o Key=Value` for probes and streams (not interactive) |
| `remote_shell` | `sh`\|`login` | sh | `login` = `bash -l -c "exec sh -s"` for sites needing a login profile |
| `forward_agent` | bool | false | adds `-A` to interactive sessions only |
| `users` | list[str] | [] (= `--me`) | Slurm: whose jobs to list (`squeue -u`); "mine" = remote user |
| `partitions` | list[str] | [] (= derived) | sinfo scope |
| `squeue_args` | list[str] | [] | appended to squeue (validated tokens) |
| `sacct_window` | duration | from defaults | |
| `poll_*` | duration | from defaults | clamped to floors with a warning |
| `log_templates` | list[str] | [] | placeholders `{workdir} {name} {jobid} {array_job} {task} {user}` |
| `protected_partitions`, `protected_qos` | list[str] | [] | mutating actions disabled on matching jobs |
| `task_min_age`, `task_min_cpu`, `ignore`, `kill_grace` | see defaults | | workstation only |

Slurm/workstation keys written flat in `[[host]]` (`users`, `partitions`, `poll_jobs`, `kill_grace`, …) override the
matching `[defaults.*]` field of `HostConfig.slurm` / `HostConfig.workstation`; `{jobid}` in templates is `job_id_raw`.

Validation (`config.py`): unknown keys → warning (log + one toast), forward compatible; wrong types / bad durations →
exit code 2 with `config.toml: host[2].poll_jobs: expected duration like "15s", got 5`; durations accept `500ms`,
`15s`, `2m`, `3h`, `3d`.

### 8.4 Zero-config start, ad-hoc hosts, import
- `runfleet` with no config file on a TTY (and no host arguments, no `--demo`) runs the **first-run prompt** in the
  plain terminal before the TUI starts (mockup N): aliases from `~/.ssh/config` (`sshconfig.list_aliases()`, following
  `Include`, skipping patterns with `* ? !`); the user picks by number/name/`all`; selected hosts are probed in parallel
  with `FleetAPI.probe_alias` (detect probe, BatchMode, read-only); `FleetAPI.write_config` writes `[[host]] name=`
  entries plus a commented defaults block; hosts needing auth are saved and show `NEEDS_AUTH`; the prompt ends with
  "From now on `runfleet` alone starts this fleet; add or remove hosts in Settings (press , in RunFleet)". Not a TTY →
  exit 2 with "no config; run `runfleet hosts import`". A config that exists but has no hosts (all unchecked in
  Settings) starts the TUI in Settings › Hosts on a TTY (exit 2 otherwise).
- **Settings** (§7.13) is the everyday way to manage the fleet: checking a host saves it and polls it at once,
  unchecking stops and forgets it; edits keep the file's comments and formatting (`configedit.py`: tomlkit, atomic
  temp-file + rename, mode kept, symlinks written through; a comment block directly above a `[[host]]` header moves
  and disappears with that host).
- `runfleet k2 labstation` uses exactly these hosts (ssh aliases or config names), merging any matching `[[host]]`
  settings, and shows only them this run. Unknown aliases work if `ssh -G` resolves them. Named hosts that are not in
  the config yet are **remembered**: saved (minimal `[[host]]` table) as soon as they answer or need a login, while
  `general.remember_hosts` (default true), with one notice "saved k2, vega — manage in Settings (,)"; a host that
  never answers is not saved. Without a config file this creates it, so the next plain `runfleet` starts that fleet.
- `runfleet hosts import [--all | ALIAS …] [--write]` is the non-interactive equivalent: prints one probe line per
  host as it finishes (alias column as wide as the longest alias + 2); `--write` writes/appends `config.toml` and
  reports aliases already present as "already configured (left unchanged)" instead of claiming they were written.
  Both paths list the hosts that did not answer with the `runfleet hosts import ALIAS … --write` command to add them
  later. `hosts list` without a config points to `runfleet` (picker) or `hosts import --all --write`.

### 8.5 CLI (`cli.py`)
```
runfleet [HOST ...] [-c PATH] [--demo] [--demo-seed N] [--readonly] [--no-mouse] [--no-cache] [--debug] [--version]
runfleet connect HOST                         auth-once with RunFleet's mux policy (ssh -fN in this terminal), then exit;
                                              session hosts (§4.8): only an auth check (press c in the TUI instead)
runfleet hosts list | import [--all | ALIAS ...] [--write]
runfleet doctor [HOST ...] [--reset]
runfleet snapshot [HOST ...] [--json] [--timeout SEC]
runfleet wait HOST:JOBID ... [--timeout SEC] [--json]
runfleet status [HOST ...] [--short | --json] summary from the cache for prompts / tmux; no ssh, no writes
runfleet update [--check] [--force]           update this install to the newest version on GitHub (§8.6)
```
`status` (`status.py`) never runs ssh, never builds a FleetService and never constructs `Cache` (whose constructor
creates, migrates and repairs the file): it opens `paths.cache_file()` with sqlite's read-only URI
(`file:…?mode=ro&immutable=0`, busy timeout 0.5 s) and reads `snapshots` and `host_status` in one read transaction
(WAL: a TUI flushing at that moment neither blocks nor is blocked; sqlite may create the empty `-wal`/`-shm` files of
a WAL database, 0600 like the file). Hosts: those named (configured or cached, else exit 2), else the enabled hosts
of the config in config order (a configured host without data shows "no data"; none enabled: no data, never the
cached hosts it disabled), else — no config, or one that does not load — every cached host by name;
`general.recent_window` comes from the same config (default 24 h). Per host it
counts **my** jobs straight from the snapshot JSON (no `from_json`): R = ACTIVE, PD = QUEUED, F = FAILED/ENDED that
ended inside the window (the header's totals, §7.7). Problem hosts are the header's red chips: NEEDS_AUTH `auth`,
HOSTKEY `hostkey`, UNREACHABLE `down` (`dns`). Age = now − the newest snapshot `taken_at` (no snapshot at all: the
newest `host_status.at`); past 10 min (`STALE_AFTER`) it is marked `stale`. Forms: default a table (HOST, STATE with
the error cut at 40 columns, R, PD, F, AGE per host and a total row); `--short` one line `R4 PD2 F1 · lynx:auth · 3m`
(`… · 14m stale`; the problem part only when there is one); `--json` schema `runfleet.status/1`:
`{"schema", "generated_at", "error", "data_at", "age_s", "stale", "stale_after_s", "recent_window_s", "totals":
{"running", "pending", "failed"}, "problems": [{"host", "problem"}], "hosts": [{"name", "state", "state_since",
"error", "problem", "data_at", "age_s", "stale", "running", "pending", "failed"}]}` (times ISO 8601 UTC; `state` the
`ConnState` value or null when never recorded). Exit 0 a summary (stale or not), 1 no data — no cache file, not a
database, no `snapshots` table, locked past the timeout, or nothing for the selected hosts (`--short` prints `no
data`, the table `no data: <why>`, `--json` the same document with `error` set and no hosts) — 2 usage (`--demo` and
`--no-cache` included: there is nothing to read). Cost ≈ the interpreter + `cli` imports: ~70 ms for the demo
fleet, ~125 ms for 12 hosts × 600 cached jobs (19 MB of JSON) on the dev box.
`wait` (`wait.py`) builds a FleetService for just the targets' hosts (named hosts enabled, `readonly`, no cache: a
cached row is not news, `idle_pause_after` effectively infinite so an unattended wait keeps polling; the ×4 idle
throttle still applies) and starts the normal pollers — no tighter cadence, no extra commands. A `Waiter` looks at the
store on every change (and every 5 s for deadlines) and prints each state change (`--json`: JSON lines, schema
`runfleet.wait/1`, events `state`/`gone`/`done`/`timeout`/`error`). A target is a job id, an array's parent id (all
tasks, summary rows included, must be final) or a task label / root pid. Final = terminal, or a vanished job whose
accounting never answered (`UNKNOWN`). Exit 0 all COMPLETED, 1 any other end (a task's `EX` included: its exit status
is unknown), 2 `--timeout`, 3 error: unknown host, a job absent after two answered polls (an answer = a new
`last_ok` of `slurm.squeue` / `process.ps`; a failed squeue or a metrics-only tick is none), NEEDS_AUTH/HOSTKEY (no
prompt can be answered here), RETRYING/UNREACHABLE for 10 min, and any setup error (config, lock, no ssh; `--json`
prints an `error` event). Targets naming the same job are waited for once. An array target remembers every task it
matched: one that ended (say FAILED) and aged out of the sacct window still decides the exit code.

Exit codes: 0 ok; 1 runtime error; 2 config/usage error; 3 no `ssh` client found (not raised with `--demo`);
`snapshot`: 4 when at least one host failed. `doctor` stores the capabilities it detects in the cache and `snapshot`
reuses capabilities younger than 1 h instead of a detect probe (cached job rows are never printed), so `doctor` +
`snapshot` cost ≤ 3 probe sessions per host (M3 F4; `--no-cache` opts out), plus at most one master login (`-fN`,
with its ProxyCommand `-W` hop) when a RunFleet-managed master has expired; `scripts/smoke_real.sh` budgets the two
separately (`--max-calls 3`, `--max-masters 1`, a hop without a master login fails). With `--debug`, non-TUI
commands print the dump directory on stderr. The plain `snapshot` table counts **my** jobs (`MY JOBS`, like the Fleet
cards) and `MY GPUS`: GPUs allocated to my running jobs on a Slurm host, `in use/total` on a workstation. `collect_once` = prepare (+ detect when no
fresh caps) + a full tick + one more round serving the follow-ups **and** every group that had nothing to do in the
first (sstat, sinfo, logstat once squeue named jobs and paths — M3 F9); workstation log paths come from
`process.fds` in the second round, which stats them itself, so no third round is needed. `runfleet connect HOST` resolves the mux plan
first (`prepare_connect`) and says "already connected" when a working master is up. `snapshot --json` (`FleetAPI.collect_once` → `export.py`) emits schema
`runfleet.snapshot/1`: `{"schema", "generated_at", "hosts": [{"name", "status", "caps", "jobs": [...], "gpus": [...],
"procs": [...], "nodes": [...], "partitions": [...]}]}` with fields as in §3.3 via `models.to_json`.

`runfleet update` is §8.6.

`runfleet doctor` prints `FleetAPI.doctor()` per host: `ssh -G` summary (user, hostname, proxyjump, controlpath,
controlpersist), chosen mux mode and jump rewrite, `-O check`, probe latency, profile, capabilities and missing tools,
timezone/skew, `gpu_tres`, warnings (e.g. "lyra: no GPU accounting — job GPU util unavailable"). `--reset` closes
managed masters.

### 8.6 Updates (`update.py`, `install.sh`, `install.ps1`)
RunFleet installs from GitHub main (no PyPI release). **Newest version** = `__version__` in main's
`src/runfleet/__init__.py`, fetched from `raw.githubusercontent.com` (urllib, 5 s timeout, `User-Agent:
runfleet/<version>`, nothing else sent); "what changed" = the CHANGELOG.md sections newer than this version (≤ 30
lines). Versions compare as `X.Y.Z` integers; anything else is never "newer".

**Install detection** (`detect_install`, from the files beside the interpreter; first match wins):
| Signal | Kind | Update steps |
|---|---|---|
| package dir is `<root>/src/runfleet` with `<root>/.git` and `pyproject.toml` | source | `git -C <root> pull --ff-only` |
| `sys.prefix/uv-receipt.toml`, requirement `git = URL` | uv | `uv tool upgrade runfleet` (keeps the user's `--with`/`--python`) |
| … `git = URL?rev=X` (pinned) | uv | `uv tool install --force git+URL` (an upgrade would keep the pin; the note says it moves to main) |
| … `directory = D` (clone) | uv | `git -C D pull --ff-only`, `uv tool install --force --reinstall D` (uv caches a directory build by version) |
| … `editable = D` | uv | `git -C D pull --ff-only` |
| `sys.prefix/pipx_metadata.json` | pipx | `pipx reinstall runfleet` (`upgrade` does not refetch a git URL of the same version) |
| a prefix part `archive-v*` (uv cache) | uvx | none: `uvx --refresh --from git+URL runfleet` |
| otherwise | pip | none: `<python> -m pip install --upgrade git+URL` |
`uv` is found on PATH, else in `$UV_INSTALL_DIR`, `$XDG_BIN_HOME`, `~/.local/bin`, `~/.cargo/bin`; uv steps run with
`UV_TOOL_DIR=<prefix parent>` and `UV_TOOL_BIN_DIR=<dir of the receipt's runfleet entrypoint>` (dirs chosen at install
time; without them `install --force` would put the script in the default bin dir). A missing program empties the steps (by hand only).

**`runfleet update [--check] [--force]`**: prints version + install, checks (forced), prints what changed; `--check`
stops there (exit 1 only when GitHub could not be asked). Without a newer version it stops (exit 0), or exit 1 when
the check failed, unless `--force`. Then each step runs in the terminal (`cli.run_step`, output visible); a failure
prints the manual command (exit 1). Then `installed_version()` (`<relaunch argv> --version` in a subprocess: this
process is still the old code) and "Updated: a → b".

**In the TUI**: `RunFleetApp(check_updates=)` — `cli.run_tui` passes `not demo and general.update_check`. On mount a
worker thread runs `update.check(now)`: the answer in `<state>/update.json` while younger than 24 h, else a fetch (a
failure keeps the last answer and its error; never raises). A newer version is toasted once per version
(`notified`). Settings › About's version row (§7.13) shows and drives it; palette › *Update RunFleet* opens Settings
on that row and runs its `↵` (checks first when nothing is known). `--demo` never asks (the row says so). Confirming
exits the app with `update.RESTART_CODE` (75, never a process exit code); `cmd_tui` then runs the same steps as
`runfleet update` in the restored terminal and, when they succeed, `os.execv`s `relaunch_argv(original argv)`: the
`runfleet` script it was started as (else `which runfleet`, else `python -m runfleet`), plus `--demo` when the
first-run prompt chose it. A failed update returns 1 with its output on screen. RunFleet never replaces its own files
while the TUI runs: lazily imported modules of the new version would mix with the old.

**`install.sh`** (repo root, `curl -LsSf …/main/install.sh | sh`, POSIX sh): a `runfleet` on PATH or in the usual
bin dirs with `update` → `exec runfleet update`; an older one → `uv tool upgrade runfleet` or `pipx reinstall
runfleet`, whichever lists it. Otherwise uv is installed when missing (Astral's installer via curl or wget), then
`uv tool install git+URL`; when `runfleet` is not on PATH afterwards, `uv tool update-shell` and "open a new
terminal". Warns (does not fail) without `ssh`. Tested with stub programs (`tests/cli/test_install_sh.py`).

**Windows (P1: by hand).** Windows refuses to overwrite or delete a running `.exe` (uv: "os error 32"), and the
`runfleet.exe` launcher that runs `runfleet update` is the file uv would replace. So on Windows `run_update` runs no
step: it prints "quit RunFleet (q) in every window, then run: `<manual>`" and exits 1 (nothing was updated);
`--check` says "Update by hand: …". In the TUI, Settings › About's `↵` on an available version shows the same line
instead of arming, the start-up toast names the command, and `request_update` never exits with `RESTART_CODE`.
Manual commands chain with `; ` instead of `&&` there (Windows PowerShell 5.1 has no `&&`); `find_uv` looks for
`uv.exe` in the fallback dirs (`shutil.which` already honours PATHEXT). `cli.restart` (only reached on POSIX now) has a
Windows branch for P3: no `os.execv` (it would spawn and exit, handing the console back to the shell), but the new
RunFleet as a child with SIGINT ignored in the waiting parent and its exit code passed on. P3 plans the in-place
update: rename the running `runfleet.exe` aside, `uv tool install` without `--force`, put it back on failure, sweep
old copies at start-up.

**`install.ps1`** (repo root, `irm …/main/install.ps1 | iex`, Windows PowerShell 5.1 and PowerShell 7, ASCII only):
the logic of `install.sh` inside one script block (`throw`/`return`, never `exit`, which would close the user's
window): an installed `runfleet` is upgraded right there with `uv tool upgrade runfleet` / `pipx reinstall runfleet`,
whichever lists it — after refusing while a `runfleet` process runs (its .exe cannot be replaced); unlike install.sh it
does not hand over to `runfleet update`, which on Windows only prints those steps (P1); otherwise Git for Windows is required (uv fetches git URLs
with it), uv is installed when missing (Astral's PowerShell installer in a child shell), then `uv tool install
git+URL` and `uv tool update-shell`. CI parses it with PowerShell 5.1 and 7 on Windows and with pwsh on Linux.

---

## 9. Error handling and resilience matrix

Principle: a remote problem is **data** (a `HostError` attached to a host or section), never an exception that reaches
the UI. Each row names the detection rule implemented in `transport/classify.py` (transport) or the collector (section).

| # | Condition | Detection | State | UI | Automatic recovery |
|---|---|---|---|---|---|
| 1 | Auth needed (no key/agent, Duo, password, expired Kerberos) — prompts are impossible in the background (§4.2), including on jump hops | ssh rc 255 + `Permission denied`, `Authentication failed`, `No more authentication methods`, `Too many authentication failures`, `read_passphrase: can't open /dev/tty` | NEEDS_AUTH (polling paused) | magenta `✖ auth` chip; card/host footer "press c to log in"; toast once | none until `c` (suspend → `ssh -fN`) or one manual `r` |
| 2 | Host key unknown / changed | `Host key verification failed`, `REMOTE HOST IDENTIFICATION HAS CHANGED`, `No .* host key is known` | HOSTKEY (paused) | changed key: HostErrorScreen shows `ssh-keygen -R <host>` advice; unknown: `c` lets ssh ask | never auto-accepts |
| 3 | DNS failure | `Could not resolve hostname` | UNREACHABLE (dns) | red chip "dns" | backoff straight to 5 min |
| 4 | Network down / refused / timeout | `Connection timed out`, `Operation timed out`, `Connection refused`, `No route to host`, `Network is unreachable`, `kex_exchange_identification`, `banner exchange`, `Connection closed by` | RETRYING → UNREACHABLE after 3 | yellow `◐ retry Ns` → red; rows `~` | exponential backoff 2× up to 5 min, ±20 % jitter |
| 5 | Jump hop fails | stderr of the rewritten `ProxyCommand` (`-W` hop): mentions the jump host, `channel 0: open failed`, `stdio forwarding failed`, `Connection closed by UNKNOWN` | as #1–#4 with kind JUMP | message names the jump host (`via jumpvm`) | as #4 |
| 6 | Probe hangs | local timeout (§4.2) **before** any section end marker | RETRYING | "probe timed out after 25 s" | managed mux: `reset_master()` after 2 consecutive (session: only when its ping fails too); user mux: suggest `c` restart |
| 6b | Probe partly done | local timeout **after** ≥ 1 end marker | DEGRADED; finished sections used, unfinished TIMEOUT | per-section errors | group backoff; transport backoff untouched |
| 7 | Mux trouble | `mux_client_`, `Control socket connect`, `Session open refused by peer` | DEGRADED/RETRYING (MUX) | host screen shows mux details | on-demand lane → 1 session; managed master reset |
| 8a | Local ssh killed | rc < 0 (signal) before any end marker | RETRYING | "ssh killed by SIGKILL" | as #4 (transient, never paused) |
| 8 | Remote can't run the probe | rc 127 for `sh`, or no markers at all with rc 0 | DEGRADED + paused | "remote shell could not run sh -s (try remote_shell=\"login\")" | none |
| 9 | Command missing | `MISSING` outcome | capability recorded | feature hidden / `n/a (no sstat)` chip | scheduler core command missing (e.g. `squeue`) → re-detect once, maybe switch profile + toast |
| 10 | Slurm controller down/slow | squeue stderr `Unable to contact slurm controller`, `Socket timed out on send/recv`, `slurm_load_jobs error` | DEGRADED (jobs) | jobs kept, `~`, banner "slurmctld unreachable" | `slurm.jobs` backoff ×2 ≤ 5 min; **vanished-detection suppressed** |
| 11 | slurmdbd down | sacct `Problem talking to the database`, `slurmdbd` errors | DEGRADED (history) | finished jobs stale; vanished ones "waiting for accounting" | `slurm.history` backoff ≤ 10 min |
| 12 | rc 0 but only stderr | `STDERR_ONLY` (lyra `--json \| head` lesson) | section failed | stderr excerpt | group backoff |
| 13 | Parse errors | > 50 % bad rows | DEGRADED (PARSE) | "could not parse squeue output (see log)" | keep previous data; raw saved with `--debug`; ≤ 50 % bad → partial rows used; bad ids kept (never "vanished") |
| 14 | Partial output | `INCOMPLETE` / `NOT_RUN` sections | those sections stale | — | next tick |
| 15 | nvidia-smi hung | wrapper rc 124 (abandoned after 6 s) or guard rc 125 | DEGRADED (gpu) | GPUs `unavailable — nvidia-smi hung`; processes/CPU keep updating | group backoff; suspended 5 min after 3; never stacks processes (guard: two stuck instances — D state or held by an abandoned `rf_nv` wrapper; the user's healthy nvidia-smi do not count); never blocks other sections (late + abandoned) |
| 16 | NVIDIA driver broken | rc 9 / "couldn't communicate with the NVIDIA driver" | DEGRADED (gpu) | "driver not loaded" | normal cadence (cheap failure) |
| 17 | Log unreadable / missing | `stat`/`tail` stderr `Permission denied` / `No such file` | LogRef.error | inline in log pane; "waiting for file" for PD jobs | `tail -F` keeps retrying |
| 18 | Action failed | per-target stderr / script lines | — | sticky error toast + palette › Action log | never retried automatically |
| 19 | Timezone / clock skew | `SLURM_TIME_FORMAT=%s` epochs; caps `tz`, `epoch` | info | skew > 60 s noted in host screen | ISO fallback localized with the host offset; elapsed from remote fields |
| 20 | Bug in collector/parser | exception at poller boundary | DEGRADED (INTERNAL) | "internal error (log id 3f2a)" | traceback logged; that group backs off; other hosts unaffected |
| 21 | Bug in UI code | Textual exception | — | crash report path printed after exit | defensive pure renderers + tests; worker `exit_on_error=False` |
| 22 | No local `ssh` | `shutil.which("ssh")` is None at start | fatal | message + exit 3 | `--demo` still works |
| 23 | Config invalid | validation | fatal | message with key path, exit 2 | — |
| 24 | Terminal too small | < 60×16 | — | placeholder screen | resumes on resize |
| 25 | `App.suspend` unsupported | `SuspendNotSupported` | — | modal with manual command | *Done* → `after_connect(host, 0)` |
| 26 | Remote `timeout` missing | `caps.tools` | info | host screen note | local probe timeout still bounds every probe |
| 27 | Cache corrupt / locked | `sqlite3.DatabaseError` | warn | toast once; `runfleet status` prints `no data` (exit 1) | rename to `cache.sqlite.corrupt-<ts>`, start fresh (the TUI; `status` never repairs or writes, §8.5) |
| 28 | MaxSessions / too many sessions | stderr #7; lane limits | — | — | ≤ 3 sessions on a user master, global cap 8, lane drops to 1 on refusal |
| 29 | Second RunFleet instance | a host's `instance.lock.*` held | — | toast once | none (user decides) |
| 30 | Unattended for hours | no input for `idle_pause_after` | all hosts paused | header "paused (idle)" | any key/mouse resumes and refreshes |

---

## 10. Testing strategy

### 10.1 Layers
| Layer | Tooling | Network | Owner |
|---|---|---|---|
| Pure parsers & helpers | pytest, anonymised field fixtures from `tests/fixtures/samples/` | none | WP2 (parsers), WP1 (framing/classify/config), WP3 (query/render/humanize) |
| Collectors | pytest + `FakeTransport`/synthetic `SectionResult`s built from fixtures | none | WP2 |
| Core (poller, store, service, actions, logs) | pytest-asyncio + `ManualClock` + `FakeTransport` | none | WP1 |
| Shell integration | `LocalTransport` running the **real** rendered probe through `/bin/sh` with `tests/mockbin` first on `PATH` | none | WP2 (mockbin) + WP1 (framing) |
| UI | Textual Pilot (`App.run_test`) + `pytest-textual-snapshot` SVGs, on `DemoTransport` + `ManualClock` | none | WP3 (+ WP4 demo) |
| Real hosts | `-m real` tests and `scripts/smoke_real.sh` | yes, read-only, **opt-in, run by hand only** | WP4 |

A session-wide fixture fails any test that spawns `ssh` unless marked `real`; the only exception is `ssh -G` (local
config evaluation, no network) in the jump-rewrite golden test.

### 10.2 Fixtures
`tests/conftest.py` provides `sample(host: str, name: str) -> str` reading `tests/fixtures/samples/<host>/<name>.txt`
byte-exact (no copies). `meta(host, probe) -> dict` parses `_meta*.txt` (rc, ms, stderr); when a probe name appears in
several meta files (`sacct_completed_gpu_tres`: rc 1 `sacct -j NONE` in `_meta_followup`, the real rc 0 run in
`_meta_followup2`), the **last entry in file order** (`_meta`, `_meta_followup`, `_meta_followup2`) wins. Helper
`frame(sections: dict[str, tuple[str, int, str]], nonce) -> str` builds framed probe output. Collector tests that need
stderr inline (the scontrol loop uses `2>&1`) build stdout = fixture stdout + meta stderr (e.g. k2
`scontrol_show_job_finished.txt` is empty because the error went to stderr). A coverage test asserts that every sample
file is referenced by at least one test, except an explicit allowlist (`identity`, `pam_adopt_login`, `*_source`,
`*_json*`, `*_size`, `gpu_query_timing`, `nvidia_smi_full`, `os_release`, `top_procs`, `commands`,
`squeue_me_default`, `scontrol_listpids*` — the latter are Slurm process lists, v0.2 per D7: no parser in v0.1). The workstation
`gpu_pids_ps`/`my_procs` fixtures have headers and a different column set than RunFleet's `ps -o …=` commands; tests
use a tiny header-aware adapter. All fixtures use `|` and ISO timestamps; parsers take `sep` and fall back to ISO, and
synthetic tests cover `\x1f` and epoch input.

Mandated parser assertions (minimum; Appendix B lists file → test module):
- `k2/squeue_me_O`: 5 jobs; `2332861` R, 1×`nvidia_a100-sxm4-40gb` (`gpu_type_source="tres-alloc"`), node `n-40-01`,
  elapsed 4:07:00, limit 3 days, stdout expands to
  `/shared/pilab/pilab/orb/logs/orb-swp--train--cifar-warmup--s43--a02_2332861.out`; `2334564` PD, reason Priority,
  `est_start` 06:31 local (−05:00, ISO fallback); `2333760` R, stderr `(null)` → merged, 2 GPUs, type
  `nvidia_h200_nvl` from tres-per-node (`gpu_type_source="tres-req"`).
- `vega/squeue_me_O`: `724479` PD `QOSMaxJobsPerUserLimit`, `StartTime N/A` → None.
- `lyra/squeue_parts_O`: array summary display id `31270818_[1-24%1]`, `job_id_raw` `31270818`, action id
  `31270818_[1-24]`; relative `slurm-%A_%a.out` resolved against WorkDir with `%a` unexpandable → `pattern` kept;
  Command `./train.sh`. `lyra/squeue_me_O` (empty) → `EMPTY`, 0 jobs. lyra running job: type `a30` from scontrol
  `JOB_GRES=gpu:A30:1`.
- Synthetic: a squeue row whose Name contains `|` (sep `|`) → dropped, id in `bad_ids`, the previous Job kept and not
  vanished; the same row with sep `\x1f` parses. Epoch timestamps `1788696459` → aware UTC.
- `*/squeue_me_legacy`, `*/squeue_parts_legacy` (`squeue -o`; not `_o`: a name differing from `_O` only in case clashes on Windows and macOS file systems): legacy parser with the fixture's field spec
  `%i|%j|%T|%M|%l|%D|%R|%b|%C|%m|%P|%V|%S|%Z|%o` (the legacy parser takes the spec as a parameter).
- `*/sacct_ext`: `CANCELLED by 2100007` → CA + uid; `None assigned` → no nodes; `Start None`; `FAILED 2:0` →
  exit 2; `End Unknown` for RUNNING.
- `lyra/sacct_failed_job_steps`: allocation PREEMPTED, batch `FAILED 143:0` → explanation "SIGTERM (128+15)" +
  "PreemptMode CANCEL: killed, resubmit".
- `k2/sstat_running`: util mean 98 (1 GPU, NTasks 1, Tot = Ave), peak 99, mem 24870 MiB, extern ignored.
  `vega/sstat_running`: 2 GPUs, Ave `gpuutil=100` → sum 100 / mean 50, Max 169 → peak 84.5. Synthetic 4-task step
  with Tot `gpuutil=360` and 4 GPUs → mean 90 (Ave would have under-reported 4×).
- gpu_tres: `vega/sacct_completed_gpu_tres` → True, `lyra/sacct_completed_gpu_tres` → False; a CPU-only job's sstat
  rows never set False; two GPU jobs without the key do.
- `*/scontrol_show_job_d`, `*_running_d`: vega `GRES=…(IDX:0-1)` → `(0, 1)`; vega `StdErr=` → merged; k2
  `StdErr == StdOut` → merged; `SubmitLine` with spaces (vega `--chdir=… --exclude=…`, lyra numeric args) is one value
  and `WorkDir` parses correctly after it; k2 PD job has no per-node block.
- `k2/_meta_followup2` `scontrol_show_job_finished` (rc 1, `Invalid job id specified`) → "left controller".
- `*/sinfo_nodes_O`: k2 `n-31-04` used 3/4 `IDX:0,2-3`; `n-31-05` 4 MIG types, base `planned` → usable, free GPUs
  counted in `gpus_free` and `gpus_free_planned`; lyra `gpu:V100:4` (no parens); vega `mixed-` → flag `-`, usable,
  counted; vega `reserved` node → not usable. Fixture fact: every usable vega node has all GPUs allocated (the only
  idle GPUs are on the `reserved` node), so vega free GPUs **= 0**; the test asserts that and proves `mixed-` counting
  with a one-line edit of the fixture.
- `gpu_types.mem_mib`: `nvidia_a100-sxm4-80gb` and `nvidia_a100_80gb_pcie` → 81920 (generic), MIG `…_3g.40gb` → 40960,
  `v100` → None.
- `*/slurm_config`, `*/sacctmgr_assoc`, `*/sinfo_version` → capabilities.
- `*/slurm_config_priority` (§5.2.10): k2 multifactor (FairShare 100000, half-life 14 days) → ranked; vega half-life
  `00:00:00`, reset `MONTHLY`; lyra every weight 0 → not ranked (the submit-order hint, no sprio/sshare planned).
  `k2/sprio`: priority 1, every factor 0, empty `%T` → the "every factor is 0" hint; `vega/sprio` fairshare 10000 + QOS
  5000. `*/sshare`: `LevelFS` `inf` → None; k2 fair share 0.95 without usage → good standing.
- `labstation/gpu_query`: 4 GPUs, GPU 3 70 %, 21464/49140 MiB; `lynx/gpu_query`: long name intact (9-field list).
  Synthetic 12-field row with a comma in the name and `[N/A]` fan → name intact, fan None, pstate `P0`.
  `*/compute_apps` join → GPU indices. `*/nvidia_smi_version` → `610.43.02` via `KMD version`.
- Tasks: `lynx/my_procs` + `lynx/compute_apps` → exactly one own task, `JobKey.label == "p642229"`, id
  `p642229-<start>`, members {642229, 642234, 786345}, GPUs {0, 1}, name `orb.launch ×2 (orb.train)`;
  systemd/sd-pam/sshd-session/-zsh/nvitop/`bash -s`/ps excluded. `labstation/my_procs` → one task labelled `p363942`;
  the Xvfb and vscode-server processes are not tasks. Synthetic: the task survives a tick in which no member holds a GPU
  or uses CPU; a failed `process.ps` never ends it; a reused root pid with another start is a new task; exit scan with
  a Traceback → FAILED, clean log → EXITED.
- `*/resources`, `*/proc_stat`, `*/meminfo` → metrics; CPU % from the two `cpu` lines.

### 10.3 Transport and core tests (WP1)
- Framing round-trip through real `/bin/sh` (LocalTransport): empty output, no trailing newline, binary-ish bytes,
  marker-like text, stderr capture, rc propagation, `timeout` → 124, missing tool → MISSING, a section reading stdin does
  not eat the script, late sections rendered last, a `SQUEUE_STATES` env var is unset, a TERM to the script exits 143.
- nvidia wrapper: a fake `nvidia-smi` that sleeps 30 s (and ignores TERM) → section rc 124 within ~6 s, later sections
  already done, probe not a transport failure.
- `OpenSSHTransport` argv golden tests for every mux mode and the jump rewrite using canned `ssh -G` outputs; the
  rewritten argv is fed to the real local `ssh -G` to assert `proxycommand` contains `BatchMode=yes` and `proxyjump`
  is not used. `ensure_master` with a fake `ssh` on PATH (records argv; simulates `-O check` failure and `-fN`
  success/auth failure); user-mode masters carry no ServerAlive/ControlPersist options.
- **No tty**: a fake `ssh` that tries to open `/dev/tty` and prompt is spawned via `transport/spawn.py` inside a pty
  harness → it fails (ENXIO) and nothing is written to the pty; the result classifies as AUTH.
- `classify.py` table tests: every regex in §9 rows 1–7.
- **Session transport** (§4.8). `tests/unit/test_agent.py` runs the REAL agent under `sh`, dash, bash and BusyBox ash:
  8 concurrent jobs and a refused 9th, a 5000-byte line byte-exact with every reply line below PIPE_BUF, `\r`
  progress arriving while a stream runs, `K` ending one job only (also with `setsid` hidden: the `ps` walk), `W`,
  stdin EOF and SIGTERM/SIGHUP ending every job (a TERM-ignoring one too) with no leftover process or temp dir, a
  rendered ProbeScript parsing exactly as under plain `sh`. `tests/unit/test_session.py` drives `SessionTransport`
  through a fake ssh (a Python script relaying `sh -c <remote command>` like ssh): prepare/ping, the argv, probes and
  streams, a timeout killing one job, 12 concurrent probes, auth failure and the 30 s throttle, a dropped session
  failing its jobs and reconnecting, the blocking login, and the transport contract run against both OpenSSH and
  session transports. Windows pieces (Job Object, spawn keywords) are tested with `IS_WINDOWS` patched and a fake
  kernel32; the fake-ssh tests are `posix` (Windows CI has no `sh`).
- Poller with `ManualClock`: tick period and evaluation only at ticks, piggyback never alone, event groups via
  follow-ups, hidden/focused factors, jitter bounds, floors, backoff growth/reset, partial-timeout rule, NEEDS_AUTH
  pause, single in-flight, manual refresh floor, idle ×4 and idle pause, persistent MonitorView with ages, merge order.
  **Politeness**: a default Slurm host with FakeTransport and no wakes makes ≤ 240 probes over a simulated hour
  (≤ 4/min on average; a single 60 s window may hold 5 because of the ±20 % grid jitter). Only poll probes count;
  the detect probe runs once at start and every 6 h.
- LogService: 5 MB of CR-only output in 64 KiB chunks → bounded memory, one `live` update stream, no exception; 20 KiB
  line → cut with marker; multi-byte UTF-8 split across chunks.
- Store diffs: `job_material_eq` ignores elapsed-only changes; `HostState.stale` uses the effective tick.
- ActionService: re-plan on param change, readonly/host/stale disabled reasons, ≤ 1 on-demand probe on a user master.

### 10.4 Shell integration with mock Slurm (turm's idea)
`tests/mockbin/{squeue,sacct,scontrol,sinfo,sstat,scancel,sprio,sshare,nvidia-smi}` are small `sh` scripts that print a fixture
selected by `$RF_MOCK_HOST` and the arguments they receive (and append argv to `$RF_MOCK_LOG`). Integration tests run
the real `SlurmCollector.sections()` → `framing.render()` → `LocalTransport` (PATH=`tests/mockbin:$PATH`) → collector
`ingest()`, asserting both the parsed jobs and the exact argv the collector issued (e.g. `sacct … -E now …`,
`squeue --me -h -O …`).
`tests/integration/test_log_search_shell.py` runs the real `log.search` section (§4.7.1) through sh/dash/bash with the
system's GNU grep / tail / head / stat on real files (window, budget, match cap, hostile paths and patterns,
unreadable files, `\r` lines) and checks the demo's play of the script byte for byte against it.
The mockbin also carries a fake `ps`/`kill` world for the §6.2 script: a root that spawns a new child when one dies →
the script STOPs the root first, signals the re-enumerated tree, and reports no survivors.

### 10.5 UI tests (WP3)
Pilot flows on the demo world (seed 7, `ManualClock`, `animations=False`), using WP3's `FakeService` until M2 and the
real stack afterwards:
1. launch → **Fleet view** with 5 cards; `tab` → Jobs (rows == store jobs in scope) → GPUs; `2` scopes to host 2;
   `0` back to all.
2. `/lmft` filters; `esc` clears; `!state:R` negation; `f` cycles presets (failed preset includes ENDED).
3. `o` cycles sort, clicking the GPU% header sorts, `O` reverses; cursor stays on the same job across a refresh.
4. `enter` → JobDetail; `]` cycles tabs; `esc` returns with the cursor on the same row.
5. `ctrl+k` on a running Slurm job → Confirm shows `scancel <id>`; `y` → the world marks it CA; row updates after one
   tick; success toast. `space` marks on 3 jobs → one dialog. Terminate on a task: toggling *escalate* updates the
   preview (re-plan).
6. `--readonly`: `ctrl+k` → toast "read-only", no dialog; the action menu shows disabled reasons. A NEEDS_AUTH host's
   jobs show actions disabled with "needs auth".
7. Fault injection (`world.inject(host, "unreachable")`): header chip red, rows `~`, app alive; recovers to OK.
8. NEEDS_AUTH host → `c` → (`run_external` monkeypatched, rc 0) → host returns to OK.
9. Log viewer: lines stream; CR progress updates the live line; `F` toggles follow; `o`/`e` switch; `p` calls
   `run_external` with the pager argv.
10. Resize 120×35 → 80×24: columns drop per §7.4, preview collapses.
11. Mouse: click selects, double-click opens detail, right-click opens the action menu.
12. Host detail (workstation): processes table; `u` toggles other users.
13. Palette: `ctrl+p`, "conn lynx" → Connect lynx; `:` opens the same palette.
14. Idle: advancing the clock past `idle_pause_after` shows "paused (idle)"; a key resumes.
15. Settings: `,` opens it, `tab`/`←`/`→`/clicks switch sections; `space` and a click on `[ ]` save a host and start
   polling it (a card appears), unchecking removes it (card gone); `t` shows the probe result; `a` adds a custom
   target; `J`/`K` reorder the cards; General values change by key and by click; a broken config shows its error.
Snapshots (SVG): fleet, jobs, job detail, host (ws), host (cluster), log viewer, help, confirm, settings at 120×35;
fleet, jobs, job detail and settings at 80×24; fleet at 100×30 (compact, no wrapped lines), 120×40 and 172×62 (roomy),
and 172×62 with a Slurm host needing auth over its cached data. A stability test runs the demo stack for 60 simulated
minutes in 5 s steps at 172×62 and 120×40: the plan, card and panel heights never change and no line is wider than its
card.

### 10.6 Demo mode (WP4; also the dev environment on a GPU-less dev box)
- `runfleet --demo [--demo-seed N]`: one scenario. Its hosts have the shapes of the field-probed machines and the
  names used throughout this document (the §7.3 mockups and §10.5 flows use them): `k2` (Slurm 25.05, GPU TRES,
  PreemptMode REQUEUE), `vega` (Slurm 25.11, 4-job QOS cap, 2-GPU H200 jobs, `mixed-` nodes), `lyra` (Slurm 26.05, no
  GPU TRES, PreemptMode CANCEL, ProxyJump-shaped latency), `labstation` (4× A6000), `lynx` (2× RTX PRO 6000, starts
  NEEDS_AUTH until connect; so `snapshot --demo` exits 4 unless `lynx` is left out). Identifiers are fictional (user
  `alice`, `/shared/lab/alice/…`, account `lab`, other users `bob`/`carol`, `*.example.edu`). Demo Slurm job ids are
  offset from the fixture ones per cluster (k2 `33xxxxx`, vega `81xxxx`–`82xxxx`, lyra
  `322xxxxx`–`323xxxxx`; new demo jobs count up from there), so a demo id never equals one in the field fixtures. The
  fixtures under `tests/fixtures/` carry fictional ids (the captured ones were replaced per cluster, keeping their
  order and array/step structure), and so do the §7.3 mockups and the §10.2
  assertions, which quote them (`2332861`, `723954`, `31354789`, …). Log records written in demo mode
  carry `[demo]`, so a simulated cancel never reads like a real one.
- **Isolation**: the demo never touches the user's real state — no cache (as before), UI state (`ui.json`) kept in memory
  only (`RunFleetApp(persist_state=False)`), and no instance lock (`FleetService.lock_path = False`), so a demo never
  blocks or is reported by a real session.
- `demo/world.py`: jobs from templates (`orb-swp--train--<bench>-<method>--s<seed>--a01`, arrays, a
  `DependencyNeverSatisfied` job, a `QOSMaxJobsPerUserLimit` job); lifecycle PD → R → CD/F/TO/OOM/PR with seeded
  probabilities; per-job logs that grow (epoch lines, tqdm `\r` progress, a Traceback on failure); GPU util random walk,
  memory ramp, temperatures/power; workstation launcher → GPU child trees (a queue runner that starts the next item),
  plus another user's GPU process. `world.inject(host, fault)` (`unreachable`, `auth`, `nvidia_hang`, `garbage`,
  `slurmdbd_down`) exists for tests; no chaos/big CLI scenarios in v0.1. Priority setups (§5.2.10): k2 ranks by
  fair share (FairShare 100000, half-life 14 days; account `lab` at fair share 0.03), vega's usage never decays
  (half-life 0, reset monthly; age, fair share and QOS weighted), lyra has every weight 0 (not ranked: no sprio/sshare).
  `render.sprio` weights the user's pending jobs from the cluster's own `PriorityWeight*` (age over 7 days, fair share
  from `SimCluster.shares`, job size, `partition_tier`, QOS) and refuses a non-multifactor cluster like the real one.
- `demo/render.py` emits **byte-exact real formats** for every section in Appendix A (squeue `-O` with the separator
  suffix, sacct `--delimiter`, epoch timestamps, scontrol `-o` with `SubmitLine` spaces, sstat TRES incl. Tot, sinfo
  GRES with `mixed-`, nvidia-smi CSV with `[N/A]`, ps). A WP4 test feeds `render` output through WP2's parsers
  (enabled at M2).
- `demo/transport.py`: `DemoTransport(host, world)` implements `Transport`: dispatches on `Section.name` (prefix
  `process.kill.` for task signals) and `meta`,
  simulates 50–300 ms latency, executes actions against the world (scancel → CA, hold/release, requeue, task signals →
  EXITED), streams logs for `log.tail` in chunks (from `meta["start"]` when a search hit opens the viewer), answers
  `log.peek`/`log.fetch` and `log.search` (`render.log_search` plays the §4.7.1 script on the simulated files). `connect_argv()` returns
  `[sys.executable, "-c", "print('demo: login ok')"]`.

### 10.7 Performance budgets (checked by hand at M2; CI enforcement is v0.2)
Startup to first paint < 300 ms with cache; Jobs refresh with 1000 rows < 100 ms headless; idle CPU < 2 % of a core
with 5 hosts; RSS < 150 MB; `runfleet status` < 150 ms in a new process (§8.5).

### 10.8 Commands
`uv run ruff check . && uv run mypy src/runfleet && uv run pytest -q` (CI). `uv run pytest --snapshot-update` to accept
UI changes (review the SVG diff). `uv run pytest -m real` only by hand, with approval.

---

## 11. Implementation plan

Four agents work in parallel in one session after M0. Each owns its files (§11.2) and codes only against the FROZEN
contracts (§3.3–§3.7) and Appendix A; anything else another package owns is reached through those contracts or a fake.

### 11.1 Work packages

**WP1 — Core platform: models, config, probe, transport, core engine**
- Files: §11.2 row WP1.
- Deliverables: contracts verbatim; `errors.py`; config loading/validation/durations/TOML emitter; `sshconfig.py`
  alias listing + `ssh -G` resolver; framing (§4.3); OpenSSH/Local/Fake transports with spawn hygiene, mux policy, jump
  rewrite, ensure-master and classifier (§4.1–4.4, §9); poller (ticks, cadence modes, lanes, backoff, partial rule,
  persistent MonitorView, idle throttle/pause, instance lock, §5.4); store; ActionService; LogService (chunked streams,
  line assembly, peek cache, fetch, pager argv, §4.7); cache (§5.5); doctor; FleetService implementing every
  `FleetAPI` method.
- Acceptance: §10.3 green; a test composes `FleetService` + `FakeTransport` + stub collectors and shows cadence,
  politeness (≤ 240 probes/simulated hour), backoff, actions and logs per §4.4/§5.4/§6.3.

**WP2 — Backends: collectors, parsers, actions**
- Files: §11.2 row WP2.
- Deliverables: §5.1–§5.3 in full (detection, gpu_tres, Slurm, process/tasks, nvidia, procfs), §6 action specs, plans
  (incl. the §6.2 script) and interpretation, detail sections, hints, explanations, exit scan; mockbin.
- Acceptance: every mandated assertion in §10.2; fixture-coverage test; mockbin integration (§10.4) incl. the §6.2
  respawn case; a test that `backends/` never imports `asyncio`, `subprocess` or `textual`.

**WP3 — UI**
- Files: §11.2 row WP3.
- Deliverables: every screen in §7.2 per the mockups, keymap single source + generated help, responsive rules,
  palette provider, confirm/action flows calling `FleetAPI`, suspend wrappers, `tests/ui/fakes.py`
  (`FakeService(FleetAPI)` over a static `FleetStore` built from model objects).
- Acceptance: §10.5 flows + snapshots; keymap test (every binding documented, no duplicate key per context, no prefix
  chords).

**WP4 — CLI, demo, packaging, docs**
- Files: §11.2 row WP4.
- Deliverables: `cli.py` (§8.4–8.5: argparse, first-run prompt, `connect`, `hosts`, `doctor`, `snapshot`, building
  `FleetService` with the right `TransportFactory`, launching `RunFleetApp`), `export.py`, demo world/render/transport
  (§10.6), packaging (§2.2–2.3), README/USER_GUIDE/CONFIG with SVG screenshots from the snapshot tests, CI workflow, a
  read-only smoke script for real hosts that counts SSH calls.
- Acceptance: `uvx --from . runfleet --demo` works in a clean environment; demo render → WP2 parser round-trip test
  (M2); `tests/cli/` covers exit codes and the first-run prompt with a fake `FleetAPI`.

### 11.2 File ownership (exactly one owner each)

| Owner | Paths |
|---|---|
| WP1 | `src/runfleet/{__init__,models,errors,config,sshconfig,paths}.py`, `src/runfleet/py.typed`, `src/runfleet/util/shell.py`, `src/runfleet/probe/`, `src/runfleet/transport/`, `src/runfleet/core/`, `tests/unit/` |
| WP2 | `src/runfleet/backends/`, `src/runfleet/util/slurmfmt.py`, `src/runfleet/util/logscan.py`, `tests/conftest.py`, `tests/backends/`, `tests/integration/`, `tests/mockbin/` |
| WP3 | `src/runfleet/ui/` (incl. `ui/query.py`), `src/runfleet/util/humanize.py`, `tests/ui/` |
| WP4 | `src/runfleet/{__main__,cli,export}.py`, `src/runfleet/demo/`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `LICENSE`, `README.md`, `CHANGELOG.md`, `docs/USER_GUIDE.md`, `docs/CONFIG.md`, `scripts/`, `tests/demo/`, `tests/cli/`, `.github/` |
| Design (edit first, then notify all) | `docs/DESIGN.md`, all **FROZEN** contracts |

`src/runfleet/util/__init__.py` is WP1's; every other package `__init__.py` belongs to the owner of its directory
(`tests/__init__.py`: WP2, with `conftest.py`).

### 11.3 M0 — skeleton commit (before the agents start; exact file list)
The per-file contract blocks of §3.3–§3.7 are copied **verbatim**; the multi-file signature blocks (§3.7 cli entry
points, §5.1 registry/detect) get their imports added. In both, bodies written `...` become `raise
NotImplementedError`, except Protocol members and `@abstractmethod`s (mypy's empty-body check then passes) — this
covers `models.to_json/from_json`, `config.load_config/dump_hosts_toml`, `HostState.stale`, the `FleetStore` methods
and `job_material_eq`. `FleetService` gets one such stub per `FleetAPI` method (so it is concrete and type-checks at
call sites from M1). `core/clock.py` is small enough to be implemented at M0 (the conftest `ManualClock` fixture needs
it); its signatures are unchanged:

| File | Content | Owner |
|---|---|---|
| `src/runfleet/models.py` | §3.3 | WP1 |
| `src/runfleet/errors.py` | `class RunFleetError(Exception)`, `class ConfigError(RunFleetError)` (`key_path: str`), `class ActionError(RunFleetError)` | WP1 |
| `src/runfleet/config.py` | §3.7 settings + `load_config`/`dump_hosts_toml` stubs | WP1 |
| `src/runfleet/probe/section.py` | §3.4 | WP1 |
| `src/runfleet/transport/base.py` | §3.5 | WP1 |
| `src/runfleet/core/clock.py`, `core/store.py`, `core/service.py` | §3.7 | WP1 |
| `src/runfleet/core/cache.py` | `class Cache` with `__init__(self, path: Path \| None) -> None` raising `NotImplementedError` | WP1 |
| `src/runfleet/backends/base.py` | §3.6 | WP2 |
| `src/runfleet/backends/registry.py`, `backends/detect.py` | §5.1 signatures + imports, bodies `raise NotImplementedError` | WP2 |
| `src/runfleet/transport/__init__.py`, `sshconfig.py`, `paths.py` | §3.7 cli entry-point signatures + imports, bodies `raise NotImplementedError` | WP1 |
| `src/runfleet/ui/app.py` | `class RunFleetApp(App[None])` with the §3.7 constructor | WP3 |
| every other `.py` path in §3.2 | a one-line module docstring stating its purpose | owner per §11.2 |
| `src/runfleet/ui/runfleet.tcss` | empty | WP3 |
| `tests/conftest.py` | `sample()`, `meta()` (last entry wins), `frame()`, ManualClock fixture, no-ssh guard | WP2 |
| `tests/unit/test_contracts.py` | imports every module of §3.2 (and fails on modules not listed there); constructs one instance of each contract dataclass with defaults; asserts the §11.5 import rules (no `textual` outside `ui/` and `cli.py`; no `asyncio`/`subprocess`/`textual` in `backends/`; no `backends` import in `ui/` but `backends.base`) | WP1 |
| `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md` (stub), `CHANGELOG.md` (stub), `.github/workflows/ci.yml` | §2.2, §10.8 | WP4 |

Exit criterion: `uv sync && uv run ruff check . && uv run mypy src/runfleet && uv run pytest` green; every import
resolves.

### 11.4 Integration order

| Milestone | Content | Exit criterion |
|---|---|---|
| **M0** | §11.3 | as above |
| **M1 — parallel build** | each WP to its acceptance against fakes: WP1 FakeTransport + stub collectors, WP2 fixtures/mockbin, WP3 FakeService, WP4 world + DemoTransport with its own format tests + CLI against a fake FleetAPI | per-WP acceptance |
| **M2 — demo integration** (WP4 leads) | `runfleet --demo` = DemoTransport → real collectors → real poller/store/service → real UI; render→parse round-trip; all §10.5 flows on the real stack; snapshots baselined; §10.7 budgets measured by hand | full suite green |
| **M3 — real hosts, read-only** (**only with explicit approval**) | `runfleet doctor` + `snapshot --json --debug` on k2, vega, lyra, labstation, lynx; one `--readonly` TUI session; verify Appendix C items 1–6; commit the `--debug` raw dumps as RunFleet-format fixtures | no host errors except real ones; ≤ 3 SSH calls per host for doctor+snapshot |
| **M4 — actions on real targets** (**only with explicit approval**; throwaway targets are created by hand, e.g. a `sleep 600` job and a small `--array=1-3%1` job on k2, and on lynx a `sleep` process plus a queue-runner script that respawns children) | cancel (incl. the array summary → `scancel <id>_[1-3]`), hold/release, requeue; terminate+escalate, interrupt, kill | all actions behave per §6; no orphan survives the queue-runner test |
| **M5 — v0.1.0** | docs, CHANGELOG, final snapshots, `uv tool install` from git | tag `v0.1.0` |

### 11.5 Coordination rules
- FROZEN contracts change only by editing this document first (one change touching `docs/DESIGN.md` + code).
- Section names, owners and `meta` keys (Appendix A) are a contract between WP2 (producer, incl. the `detect`
  sections via `backends/detect.py`), WP4 (DemoTransport) and WP1 (`log` sections only, FakeTransport). New sections are added to Appendix A in the same change.
- Typing: full annotations; `mypy --strict` for `models`, `config`, `probe`, `transport`, `backends`, `core`; Textual
  code may be non-strict. `ruff` per §2.2.
- No `textual` import outside `ui/` and `cli.py`; no `subprocess`/`asyncio` in `backends/`; `ui/` imports nothing from
  `backends/` but the contract types of `backends.base` (pure helpers both need live in `util/`); time only through `Clock`;
  logging via `logging.getLogger("runfleet.<module>")`; no secrets in logs (never log environment values or file
  contents beyond 200-char excerpts).
- Never run anything against real hosts from tests or CI. Manual work on real hosts (M3/M4) needs explicit approval
  and is read-only unless that specific action is approved; nothing in RunFleet submits jobs.
- **v0.2 items are not built in v0.1**, not even as hidden flags or stubs.

---

## Appendix A — Section catalogue (names, owners, `meta` keys)

Owner = `Section.owner` (name prefix). All `meta` values are also interpolated (validated, quoted) into `cmd`.

| Section | `meta` | Purpose |
|---|---|---|
| `detect.id`, `detect.tools` | — | identity, tz, epoch, tools |
| `slurm.version`, `slurm.config`, `slurm.assoc`, `slurm.parts`, `slurm.sacctfmt`, `slurm.gputres` | — | capabilities, gpu_tres seed |
| `nvidia.version` | — | driver version |
| `slurm.squeue` | `users: list[str]`, `extra: list[str]`, `sep: str`, `format: "O"\|"legacy"` | live jobs |
| `slurm.sacct` | `since_s: int`, `fields: list[str]`, `sep: str` | history window |
| `slurm.sacct_ids` | `ids_raw: list[str]`, `fields: list[str]`, `sep: str` | final state of vanished jobs |
| `slurm.scontrol` | `ids_raw: list[str]` | per-job detail, paths, GPU IDX/type |
| `slurm.sstat` | `ids_raw: list[str]` | job GPU/CPU usage |
| `slurm.steps` | `job_id_raw: str`, `sep: str`, `fields: list[str]` | detail: step forensics (+ TotalCPU, §5.2.8) |
| `slurm.logstat` | `paths: list[str]` | size + mtime |
| `slurm.sinfo` | `partitions: list[str]`, `sep: str` | nodes, free GPUs |
| `slurm.occupants` | `partitions: list[str]`, `sep: str` | every user's running jobs per node (§5.2.9) |
| `slurm.names` | `users: list[str]` | GECOS names of those users |
| `slurm.sprio` | `format: str` (`SPRIO_FORMAT`, a constant) | own pending jobs' weighted priority factors (§5.2.10) |
| `slurm.sshare` | `fields: list[str]` (constant) | the user's fair share per account (§5.2.10) |
| `slurm.act.cancel`, `slurm.act.hold`, `slurm.act.release` | `ids: list[str]` (display ids, throttle stripped) | actions |
| `slurm.act.requeue` | `ids: list[str]`, `hold: bool` | action |
| `procfs.host` | — | load, mem, cpu, uptime |
| `nvidia.gpu` (late) | `fields: list[str]` | GPUs |
| `nvidia.apps` (late) | — | GPU processes + their ps rows |
| `process.ps` | — | own processes + remote epoch |
| `process.fds` | `pids: list[int]` | cwd, stdout, stderr |
| `process.logstat` | `paths: list[str]` | size + mtime |
| `process.exitscan` | `tasks: list[tuple[str, list[str]]]` (job id, log paths) | ended-task log scan |
| `process.detail` | `pids: list[int]`, `root: int` | detail: `ps` of members + allow-listed env of the root (`CUDA_VISIBLE_DEVICES VIRTUAL_ENV CONDA_DEFAULT_ENV OMP_NUM_THREADS WANDB_PROJECT WANDB_RUN_ID`) |
| `process.kill.<root pid>` (one per task) | `root: int`, `start: int`, `sig: "TERM"\|"INT"\|"KILL"`, `grace_s: int`, `escalate: bool` | §6.2 |
| `log.tail` (StreamSpec) | `path: str`, `bytes: int`, `start: int` (optional: `tail -c +<start+1>`, a search hit) | follow |
| `log.peek` | `path: str`, `bytes: int` | preview |
| `log.fetch` | `path: str`, `bytes: int` | pager fallback |
| `log.search` | `paths: list[str]`, `pattern: str` (the ERE grep runs), `icase: bool`, `regex: bool`, `max_matches: int`, `per_file: int`, `budget: int` | cross-job log search, on demand (§4.7.1) |
| `slurm.attach.node<i>` (Attach, §7.16) | `mode: "ssh"\|"srun"`, `node: str`, `job_id_raw: str`, `jobs_raw: list[str]` (own running jobs on the node read by node.cg/node.pids, target first), `sluids: list[str]` (parallel to jobs_raw, "" unknown), `inner_nonce: str`, `inner: list[str]` (nested section names in order) | one node hop carrying the nested node script |
| `slurm.attach.audit` (late, Attach) | `mode: str`, `job_id_raw: str`, `nodes: list[str]` | no-leftover audit |
| `node.id` `node.cg` `node.ps` `node.pids` `node.map` `node.gpu` `node.apps` `node.self` (nested, owner `node`) | — (answered via the outer section's meta) | §7.16.3 |

## Appendix B — Fixture → test module map

| Fixture(s) | Test module |
|---|---|
| `*/squeue_me_O`, `*/squeue_parts_O`, `*/squeue_me_legacy`, `*/squeue_parts_legacy`, `*/squeue_parts_states`, `*/squeue_array_expand` | `tests/backends/slurm/test_parse_squeue.py` |
| `*/sacct_basic`, `*/sacct_ext`, `*/sacct_steps`, `*/sacct_running_steps`, `*/sacct_failed_job_steps`, `*/sacct_completed_gpu_tres`, `*/sacct_state_filter_test`, `*/sacct_failed_states` | `tests/backends/slurm/test_parse_sacct.py` |
| `*/scontrol_show_job*`, `*/scontrol_show_node` | `tests/backends/slurm/test_parse_scontrol.py` |
| `*/sinfo_nodes_O`, `*/sinfo_parts`, `*/sinfo_summary` | `tests/backends/slurm/test_parse_sinfo.py` |
| `*/sstat_running` | `tests/backends/slurm/test_parse_sstat.py` |
| `*/slurm_config_priority`, `*/sprio`, `*/sshare` | `tests/backends/slurm/test_priority.py` |
| `*/slurm_config`, `*/slurm_env`, `*/sacctmgr_assoc`, `*/sinfo_version`, `*/stdout_path_stat` | `tests/backends/slurm/test_caps.py` |
| `labstation/*`, `lynx/*` (`gpu_query*`, `compute_apps*`, `nvidia_smi_version`) | `tests/backends/common/test_nvidia.py` |
| `*/resources`, `*/proc_stat`, `*/meminfo` | `tests/backends/common/test_procfs.py` |
| `*/gpu_pids_ps`, `*/my_procs` (+ `compute_apps`) | `tests/backends/process/test_tasks.py` |
| `*/_meta*.txt` | `tests/backends/test_outcomes.py` (STDERR_ONLY/rc mapping) |

## Appendix C — Open items to verify (none block v0.1 code; M3 = read-only, with approval)
1. sstat semantics: `TRESUsageInTot` = sum over tasks, and a task's `gres/gpuutil` = sum over its GPUs (vega 2-GPU
   job: Max 169). Verify at M3 on a multi-GPU job; both readings are unit-tested, flipping is one line.
2. `\x1f` accepted as `-O` field suffix (squeue, sinfo) and by `sacct --delimiter` on Slurm 25.05 / 25.11 / 26.05
   (M3); per-host fallback to `|` exists.
3. `SLURM_TIME_FORMAT=%s` honoured by squeue, sacct and scontrol on all three clusters (M3); ISO fallback exists.
4. Jump rewrite on lyra and labstation: same latency as plain ProxyJump; host-key prompt on `jumpvm` fails cleanly as
   HOSTKEY (M3).
5. `sacct --format=StdOut,StdErr` per cluster — capability-gated, no code risk.
6. sstat on one's own lyra jobs: CPU/RSS without GPU TRES (M3, if a job is running).
7. v0.2 gates, not v0.1: `srun --overlap` deep-inspect flags/latency on k2/vega; pam_slurm_adopt node ssh; slurmrestd.

## Appendix D — Critique disposition (revision 1 → 2)

Numbers are the critique's finding indices (0-based). "Done" = suggested fix applied (or the noted variant).

| # | Sev. | Finding | Resolution | Sections |
|---|---|---|---|---|
| 0 | crit | ProxyJump hop drops `-o`; probe ssh keeps the tty | Done: tty-less spawn (new session, stdin pipe, `SSH_ASKPASS_REQUIRE=never`, no DISPLAY) + explicit BatchMode ProxyCommand rewrite; golden `ssh -G` test and pty no-prompt test. Multi-hop chains: options on the last hop only, earlier hops still tty-less | D9, §4.2, §9 #1/#5, §10.3 |
| 1 | major | User master started with RunFleet keepalives | Done: user master = `-fN BatchMode ControlMaster=yes ConnectTimeout LogLevel -E` only (none affect later sessions; `ControlMaster=yes` is needed when the config says no/ask); keepalives only on managed masters; ControlPersist=no → managed; env inheritance documented | D8, §4.1 |
| 2 | major | `ssh -fN` from suspend writes onto the TUI later | Done: `-E <log> -o LogLevel=ERROR` on connect and on the hop (interactive rewrite); log tail shown on failure and in HostErrorScreen | §4.2, §4.6 |
| 3 | major | Hung nvidia-smi blocks the probe and looks like unreachability | Done: `late` sections rendered last; background + poll + abandon wrapper (6 s); partial-timeout rule makes it DEGRADED, never a transport failure | §3.4, §4.2–4.4, §5.3.1, §9 #6b/#15 |
| 4 | major | procfs polls login nodes every 5 s; no piggyback/event; ambiguous group names | Done: `Cadence.mode` interval/piggyback/event; tick model (groups evaluated only at ticks); groups named `<collector>.<name>`; ≤ 4 probes/min test | §3.6, §5.2.1, §5.3.1, §5.4, §10.3 |
| 5 | major | gpu_tres poisoned by a CPU-only job | Done: learn only from GPU jobs, False after 2 GPU jobs, seeded by `slurm.gputres`; sstat CPU/RSS not gated | §5.1, §5.2.3, §10.2 |
| 6 | major | sstat Ave under-reports multi-task jobs | Done: `TRESUsageInTot` for sum/mean, Max for peak, step chosen by NTasks; synthetic 4-task test; Appendix C #1 | §5.2.2–5.2.3, §3.3 |
| 7 | major | GPU type missing for running jobs | Done: tres-alloc → scontrol → tres-per-node/job → sinfo → cache, with `gpu_type_source`; 2333760 → `nvidia_h200_nvl` asserted (via tres-per-node, which the fixture has) | §3.3, §5.2.3, §10.2 |
| 8 | major | `-` (planned) nodes excluded from free GPUs | Done: `-`/`planned` usable but tracked in `gpus_free_planned`; `reserved` added and excluded; test fixed | §3.3, §5.2.3, §10.2 |
| 9 | major | Tasks flip to EXITED; same-tick MonitorView; pid-reuse id collision | Done: sticky tasks while root `(pid,start)` lives; EXITED only after a successful `process.ps`; persistent MonitorView with ages; id `p<pid>-<start>` with `JobKey.label` | §3.3, §3.6, §5.3.2 |
| 10 | major | Terminate from a stale member list orphans queue-runner children | Done: §6.2 re-enumerates remotely, STOPs the root first, signals tree (+ pgid when exclusive), CONT, grace, KILL, reports `new` pids; M4 queue-runner test | §6.2, §10.4, §11.4 |
| 11 | major | FleetService lacks UI APIs | Done: `FleetAPI` protocol with probe_alias, write_config, doctor, fetch_log, peek_log, stream_log, collect_once, activity/focus hooks; `actions_for` host/stale reasons. Re-planning = calling the pure `plan()` again (no separate `replan`) | §3.7, §6.3 |
| 12 | major | Session budget on user masters; preview stream per cursor stop | Done: preview = cached one-shot `log.peek` after 1 s dwell; streams only in detail/log screens; ≤ 3 sessions on a user master | §4.2, §4.7, §7.5 |
| 13 | major | `readline` 64 KiB limit and CR-only tqdm lines | Done: `StreamHandle.chunks()`, splitter on `\n`/`\r` with a live line, 16 KiB line cap, 5 MB CR-only test | §3.5, §3.7 LogUpdate, §4.7, §10.3 |
| 14 | major | RichLog cannot prepend/search/re-wrap | Done (cut option): RichLog append + live-line Static + `$PAGER` for search; paging/search/wrap/save → v0.2 | §4.7, §7.3 I, §7.12 |
| 15 | major | Fixed UTC offset wrong across DST | Done: `SLURM_TIME_FORMAT=%s`, epoch parsing, ISO fallback only for fixtures; sacct `-S now-<n>seconds` | §4.3, §5.2.2–5.2.3, §9 #19 |
| 16 | major | PRD gaps: Slurm processes, failed vs finished tasks, history, overview | (b)(d) done: ENDED group, exit log scan in v0.1, task stall hints, Fleet is the default view. (a) **deferred**: Slurm process lists need compute-node access (v0.2, D7); v0.1 shows steps with a note. (c) **deferred**: `:history` is v0.2 per the scope decision; `sacct_window` is configurable meanwhile | §1.3, §3.3, §5.2.6–5.2.7, §5.3.2, §7.2 |
| 17 | major | v0.1 overloaded with risky features | Done: lead's scope cut applied throughout | §1.3, §6, §7, §8.5, §10, §11 |
| 18 | minor | `\|` inside a field breaks rows / vanished logic | Done: `\x1f` separator for squeue/sinfo `-O` and sacct `--delimiter` (sstat keeps `\|`: no free text); `bad_ids` never vanish; verification in Appendix C #2 | §5, §5.2.2–5.2.4 |
| 19 | minor | nvidia-smi parse rule wrong for 12 fields | Done: index from the right by field list; 12-field synthetic test | §5.3.1, §10.2 |
| 20 | minor | Section routing / MonitorView / merge ambiguity | Done: `Section.owner` = prefix = collector name (sections renamed `nvidia.*`, `process.*`, `detect.*`, `slurm.act.*`); `MonitorView.gpu_procs`; explicit merge rule (scheduler's gpus win) | §3.1, §3.4, §3.6, App. A |
| 21 | minor | Keymap clashes with Textual defaults; prefix keys | Done differently: no prefix chords at all (`f`/`o` cycle; palette for direct choice); `ctrl+c`/`ctrl+z` priority bindings; `tab` views on MainScreen only; `→` not rebound; `ActionSpec.key` dropped | §3.6, §7.8 |
| 22 | minor | Managed mode overrides a user ControlPath; direct-probe fallback | Done: managed only when no ControlPath (or ControlPersist=no); no probe while ensuring a master; a check→probe race can still cause one direct BatchMode login (accepted, documented) | §4.1 |
| 23 | minor | Stale marker during normal throttling | Done: `HostState.stale(now)` uses `HostStatus.tick_interval` (effective) + 5 s | §3.3, §3.7, §7.7 |
| 24 | minor | Prelude misses row-changing Slurm env | Done: awk loop unsets `^(SQUEUE\|SINFO\|SACCT\|SCONTROL\|SSTAT)_` and `SLURM_TIME_FORMAT/CLUSTERS` | §4.3, §10.3 |
| 25 | minor | Array summary ids passed verbatim; `%j` ambiguity | Done: `Job.job_id_raw` for scontrol/sstat/sacct/`%j`; actions strip the `%N` throttle; M4 array case | §3.3, §5.2.3, §5.2.5, §11.4 |
| 26 | minor | Duplicate meta entries; empty scontrol fixture; mockup F mismatch | Done: `meta()` last entry wins; stderr folded into stdout for loop tests; mockup F rebuilt from the fixture | §10.2, §7.3 F |
| 27 | minor | No idle pause; multi-instance load | Done: `idle_pause_after` (2 h) pauses all polling; `instance.lock` warning (warn only, no coordination) | §3.7, §5.4, §7.7, §9 #29–30 |
| 28 | minor | Consistency nits | Done: one budget (≤ 7 RPCs/min); `recent_window`/`stall_after` only in GeneralSettings; ErrorKind only in models; `*_mib` rule; generic `(\d+)gb` + new types; traps `exit 129/130/143` + final `exit 0`. Readonly vs consent is moot (deep inspect v0.2) | §1.1, §3.3, §3.7, §4.3, §5.2.3, §5.4 |
