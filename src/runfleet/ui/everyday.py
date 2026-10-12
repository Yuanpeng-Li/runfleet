"""[WP3] Everyday actions: copy menu and support bundle, port forwards, log in to all, free-GPU picker.

The pure parts (copy items, the Markdown support bundle with secrets scrubbed, URL candidates, the free-GPU choice)
are module functions the tests call directly; the flows below them take the app and only use what it already offers
(``FleetAPI``, ``ActionMenuScreen``, ``toast``, ``run_external_sequence``). Nothing here mutates a job: they work in
read-only mode too. Remote commands run only on a keypress, as on-demand probes (job detail, a log fetch).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runfleet.backends.base import ActionSpec
from runfleet.core.forwards import UrlCandidate, find_urls
from runfleet.errors import RunFleetError
from runfleet.models import GPU, ConnState, Job, JobDetail, JobKey, JobKind, LogRef, StateGroup
from runfleet.util import humanize
from runfleet.util.logscan import cause_of

if TYPE_CHECKING:
    from runfleet.ui.app import RunFleetApp

log = logging.getLogger("runfleet.ui.everyday")

BUNDLE_LOG_LINES = 30
FETCH_TAIL = 16 << 10          # support bundle: the log's last 16 KiB when RunFleet holds fewer lines
URL_SCAN_BYTES = 1 << 20       # port forward: service URLs are printed at start-up, so look at up to 1 MiB
FREE_MEM_MIB = 1024.0          # a GPU is free below 1 GiB used …
FREE_UTIL_PCT = 5.0            # … and 5 % utilisation, with no process on it
LOGIN_STATES = frozenset({ConnState.NEEDS_AUTH, ConnState.HOSTKEY, ConnState.UNREACHABLE})
LABEL_WIDTH = 60               # menu labels fit one line of the 66-wide menu


# ====================================================================================================== secrets
_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{40}(?![0-9A-Fa-f])"), "<redacted-40hex>"),     # wandb API keys
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-<redacted>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "gh*_<redacted>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<redacted-aws-key-id>"),
    (re.compile(r"(?i)\b(aws_secret_access_key|aws_session_token)(\s*[=:]\s*)\S+"), r"\1\2<redacted>"),
    (re.compile(r"(?i)\b((?:access_)?token=)[^&\s#'\"]+"), r"\1<redacted>"),
    # PEM private keys (to the END line, else to the end of the code fence / text: a tail can cut the block)
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|(?=\n```)|\Z)",
                re.DOTALL), "<redacted-private-key>"),
    (re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/\-]{8,}=*"), r"\1 <redacted>"),
    (re.compile(r"\b([A-Za-z][A-Za-z0-9+.\-]*://)[^/\s:@'\"]+:[^/\s@'\"]+@"), r"\1<redacted>@"),   # user:pw@ in URLs
    # NAME=value / NAME: value / "name": "value" whose name ends in password, secret, token, api key …
    (re.compile(r"(?i)\b([A-Za-z0-9_.\-]*(?:password|passwd|passphrase|secret|token|api_?key|access_?key)"
                r"[\"']?\s*[=:]\s*[\"']?)(?!<redacted)[^\s\"',;&]+"), r"\1<redacted>"),
)


def scrub(text: str) -> str:
    """Secrets out of text that leaves the machine (support bundle): wandb keys (40 hex), ``hf_…``, ``sk-…``,
    ``ghp_…``, AWS key ids and secrets, ``token=`` query parameters, PEM private keys, ``Bearer`` / ``Basic``
    credentials, ``user:password@`` in URLs and values of ``…PASSWORD`` / ``…SECRET`` / ``…TOKEN`` / ``…API_KEY``
    assignments."""
    for rx, repl in _SECRETS:
        text = rx.sub(repl, text)
    return text


# ====================================================================================================== copy menu
@dataclass(frozen=True, slots=True)
class CopyItem:
    id: str
    what: str                    # "job id", "stdout path" (menu label and toast)
    text: str                    # what goes to the clipboard ("" for the bundle, built on demand)
    disabled: str | None = None  # why it cannot be copied ("no stderr path known")

    @property
    def label(self) -> str:
        if self.disabled or not self.text:
            return self.what
        room = max(16, LABEL_WIDTH - len(self.what) - 5)      # 2 more for the accelerator key in front
        return f"{self.what} · {humanize.truncate(self.text.replace(chr(10), ' '), room, keep_tail=True)}"


def paste_shell_command(alias: str, remote_cmd: str | None) -> str:
    """What a user pastes into another terminal: ``ssh -t <alias> '<the command s runs>'`` (just the command for
    ``ssh = "local"``)."""
    if alias == "local":
        return remote_cmd or os.environ.get("SHELL", "sh")
    head = f"ssh -t {shlex.quote(alias)}"
    return f"{head} {shlex.quote(remote_cmd)}" if remote_cmd else head


def _ref(job: Job, stream: str) -> LogRef | None:
    return next((r for r in job.logs if r.stream == stream), None)


def copy_items(job: Job, *, alias: str, shell_cmd: str | None) -> list[CopyItem]:
    """The copy menu of one job, in menu order: the stdout path first (what ``Y`` alone used to copy)."""
    jid = job.key.id if job.kind is JobKind.SLURM else job.job_id_raw
    stdout, stderr = _ref(job, "stdout"), _ref(job, "stderr")
    out = [CopyItem("stdout", "stdout path", stdout.path or "" if stdout else "",
                    None if stdout is not None and stdout.path else "no stdout path known")]
    if stderr is not None and stderr.path and not stderr.merged:
        out.append(CopyItem("stderr", "stderr path", stderr.path))
    elif stdout is not None and stdout.path and (stdout.merged or (stderr is not None and stderr.merged)):
        out.append(CopyItem("stderr", "stderr path (= stdout)", stdout.path))
    else:
        out.append(CopyItem("stderr", "stderr path", "", "no stderr path known"))
    out += [CopyItem("job_id", "job id", jid), CopyItem("host_job", "host:jobid", str(job.key))]
    out.append(CopyItem("workdir", "workdir", job.workdir or "", None if job.workdir else "no workdir known"))
    running = job.kind is JobKind.SLURM and job.state.group is StateGroup.ACTIVE and bool(job.nodes)
    what = "node shell command" if running else "shell command"
    out.append(CopyItem("shell", what, paste_shell_command(alias, shell_cmd)))
    out.append(CopyItem("bundle", "support bundle (Markdown, secrets scrubbed)", ""))
    return out


# copy menu accelerators (key -> item id): mnemonic letters clear of the menu's j / k / q; ``Y`` again (the key that
# opened the menu) copies the stdout path, so ``Y Y`` is the old one-key "copy stdout path"
COPY_KEYS: dict[str, str] = {"o": "stdout", "Y": "stdout", "e": "stderr", "i": "job_id", "h": "host_job",
                             "w": "workdir", "s": "shell", "b": "bundle"}


# ====================================================================================================== support bundle
def _request_line(job: Job) -> str:
    a = job.alloc
    bits = []
    if a.nodes:
        bits.append(f"{a.nodes} node{'s' if a.nodes != 1 else ''}")
    if a.cpus:
        bits.append(f"{a.cpus} CPUs")
    if a.mem_bytes:
        bits.append(humanize.size(a.mem_bytes))
    if a.gpus:
        bits.append(f"{a.gpus}× GPU" + (f" ({a.gpu_type})" if a.gpu_type else ""))
    if job.time_limit is not None:
        bits.append(f"time limit {humanize.clock(job.time_limit)}")
    elif job.kind is JobKind.SLURM:
        bits.append("no time limit")
    return " · ".join(bits) or "—"


def _state_line(job: Job) -> str:
    out = f"{job.state.name} ({job.state.value})"
    if job.raw_state and job.raw_state.upper() != job.state.name:
        out += f" · {job.raw_state}"
    if job.exit_code is not None:
        out += f" · exit {job.exit_code}:{job.exit_signal or 0}"
    if job.derived_exit:
        out += f" · derived {job.derived_exit}"
    if job.held:
        out += " · held"
    if job.reason and job.reason != "None":
        out += f" · reason {job.reason}"
    return out


def _nodes_line(job: Job, detail: JobDetail | None) -> str:
    per_node = (detail.per_node if detail is not None and detail.per_node else job.alloc.per_node)
    if per_node:
        return ", ".join(n.node + (f" (GPU IDX {','.join(map(str, n.gpu_indices))})" if n.gpu_indices else "")
                         for n in per_node)
    return ", ".join(job.nodes) or job.nodelist or "—"


def _cause_line(job: Job, detail: JobDetail | None, lines: Sequence[str]) -> str | None:
    """The cause chip (§5.2.8, ``logscan.cause_of(...).plain()``) from the lines RunFleet holds; never fails the
    bundle."""
    try:
        cause = cause_of(job, detail, lines)
    except Exception:                          # the bundle must come out even if triage breaks
        log.exception("cause_of failed")
        return None
    return cause.plain() if cause is not None else None


def support_bundle(job: Job, detail: JobDetail | None, *, cluster: str, log_lines: Sequence[str], now: datetime,
                   version: str = "") -> str:
    """A Markdown report to paste into Slack / a ticket; secrets scrubbed (§7.15)."""
    md = [f"# RunFleet support bundle · {job.key.host}:{job.key.label}", ""]

    def row(label: str, value: object) -> None:
        if value not in (None, "", ()):
            md.append(f"- **{label}:** {value}")

    row("cluster", cluster)
    row("job", f"{job.key.id} · {job.name}")
    row("user", job.user)
    if job.kind is JobKind.SLURM:
        row("partition / QOS / account", " / ".join(x or "—" for x in (job.partition, job.qos, job.account)))
    row("request", _request_line(job))
    row("state", _state_line(job))
    times = []
    for label, at in (("submitted", job.submit_time), ("started", job.start_time), ("ended", job.end_time)):
        if at is not None:
            times.append(f"{label} {humanize.local_dt(at)}" + (" (estimate)" if label == "started" and job.est_start
                                                               else ""))
    if job.elapsed is not None:
        times.append(f"elapsed {humanize.clock(job.elapsed)}")
    row("times", " · ".join(times))
    row("nodes", _nodes_line(job, detail))
    row("restarts", job.restarts)
    row("workdir", job.workdir)
    for ref in job.logs:
        row(f"{ref.stream}", (ref.path or ref.pattern or "unknown") + (" (merged)" if ref.merged else ""))
    if detail is not None and detail.explanation:
        md += ["", "## Why it ended" if job.state.is_terminal else "## Notes", ""]
        md += [f"- {line}" for line in detail.explanation]
    cause = _cause_line(job, detail, log_lines)
    if cause:
        md += ["", "## Cause", "", f"- {cause}"]
    efficiency = detail.efficiency if detail is not None else ()
    if efficiency:
        md += ["", "## Efficiency", ""] + [f"- {line}" for line in efficiency]
    if job.hints:
        md += ["", "## Hints", ""] + [f"- {h}" for h in job.hints]
    if detail is not None and detail.errors:
        md += ["", "## Incomplete data", ""] + [f"- {e.summary}" for e in detail.errors]
    tail = [humanize_line(ln) for ln in log_lines][-BUNDLE_LOG_LINES:]
    md += ["", f"## Last {len(tail)} log lines" if tail else "## Log", ""]
    md += ["```text", *tail, "```"] if tail else ["(no log lines available)"]
    md += ["", f"_RunFleet {version} · {humanize.local_dt(now)} · secrets scrubbed_".replace("  ", " ")]
    return scrub("\n".join(md) + "\n")


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def humanize_line(line: str) -> str:
    """A log line for Markdown: ANSI escapes gone, the last carriage-return segment (tqdm), no code fence."""
    text = _ANSI.sub("", line).rstrip("\r\n").rsplit("\r", 1)[-1]
    return text.replace("```", "ʼʼʼ")


def bundle_path(directory: Path, key: JobKey) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", f"{key.host}-{key.id}")
    return directory / f"{safe}.md"


# ====================================================================================================== free GPUs
def _holder(gpu: GPU, jobs: Mapping[JobKey, Job]) -> str:
    users: list[str] = []
    for p in sorted(gpu.processes, key=lambda p: -(p.used_mem_mib or 0)):
        job = jobs.get(p.job) if p.job is not None else None
        who = f"you ({job.key.label})" if job is not None and job.is_mine else (p.user or "?")
        if who not in users:
            users.append(who)
    if users:
        return f"GPU {gpu.index} {', '.join(users)}"
    if gpu.util_pct is None or gpu.mem_used_mib is None:
        return f"GPU {gpu.index} no reading"
    return (f"GPU {gpu.index} busy ({gpu.util_pct:.0f}% · {humanize.gib(gpu.mem_used_mib, decimals=1)} "
            "· no process visible)")


def free_gpus(gpus: Iterable[GPU], jobs: Mapping[JobKey, Job] | None = None) -> tuple[list[GPU], list[str]]:
    """(free GPUs idlest first, who holds the others). Free = no process, < 1 GiB used, < 5 % utilisation; ranked
    by memory used, then utilisation, then index."""
    jobs = jobs or {}
    free: list[GPU] = []
    held: list[str] = []
    for g in sorted(gpus, key=lambda g: g.index):
        if (not g.processes and g.mem_used_mib is not None and g.mem_used_mib < FREE_MEM_MIB
                and g.util_pct is not None and g.util_pct < FREE_UTIL_PCT):
            free.append(g)
        else:
            held.append(_holder(g, jobs))
    free.sort(key=lambda g: (g.mem_used_mib or 0.0, g.util_pct or 0.0, g.index))
    return free, held


GPU_SECTIONS = ("nvidia.gpu", "process.ps")   # the GPU table and who runs on it
GPU_MAX_AGE = 60.0                          # a GPU reading older than this (its section failing since) is no answer


def gpu_readings_stale(status: Any, now: datetime) -> bool:
    """The GPU / process sections failed at their last run, or have not answered for ``GPU_MAX_AGE`` (while other
    sections kept the host fresh): the GPUs that look idle may not be."""
    for name in GPU_SECTIONS:
        st = status.sections.get(name)
        if st is None:
            continue
        if not st.ok or st.last_ok is None or (now - st.last_ok).total_seconds() > GPU_MAX_AGE:
            return True
    return False


def cuda_visible(gpus: Sequence[GPU]) -> str:
    return "CUDA_VISIBLE_DEVICES=" + ",".join(str(g.index) for g in gpus)


# ====================================================================================================== port forwards
def forward_node(job: Job | None) -> tuple[str | None, str | None]:
    """(node to forward to, None) or (None, why not): a running Slurm job's batch host; a task / host: localhost."""
    if job is None or job.kind is JobKind.TASK:
        return "localhost", None
    if job.state.group is not StateGroup.ACTIVE or not job.nodes:
        return None, f"{job.key.label} is not running — nothing listens on a node yet"
    return (job.batch_host if job.batch_host in job.nodes else job.nodes[0]), None


def candidate_label(c: UrlCandidate) -> str:
    head = f"{c.port} · {c.label or 'service'} · "
    url = f"{c.scheme}://{c.target}:{c.port}{c.path}"
    return head + humanize.truncate(url, max(16, LABEL_WIDTH - len(head)))


# ====================================================================================================== app flows
def tmux_load_buffer(text: str) -> None:
    """Fallback for terminals that drop OSC 52 inside tmux: the tmux paste buffer (and, with ``-w``, the outer
    terminal's clipboard). Short timeout, failures ignored."""
    for argv in (["tmux", "load-buffer", "-w", "-"], ["tmux", "load-buffer", "-"]):    # -w needs tmux >= 3.2
        try:
            res = subprocess.run(argv, input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=2, check=False)
        except (OSError, subprocess.SubprocessError):
            return
        if res.returncode == 0:
            return


def copy_text(app: RunFleetApp, text: str, what: str, *, message: str | None = None) -> None:
    """OSC 52 (never acknowledged) plus the tmux buffer under $TMUX; the toast says what was copied."""
    app.copy_to_clipboard(text)
    tmux = bool(os.environ.get("TMUX"))
    if tmux:
        app.run_worker(lambda: tmux_load_buffer(text), thread=True, group="tmux", exit_on_error=False)
    if message is None:
        shown = text if len(text) <= 100 and "\n" not in text else f"{len(text.splitlines())} lines"
        message = f"copied {what}: {shown}"
    app.toast(f"{message}  (OSC 52{' + tmux buffer' if tmux else ''})", timeout=8)


def known_log_lines(app: RunFleetApp, job: Job) -> list[str]:
    """Log text RunFleet already holds for the job: what job detail streamed (its cause-chip buffer, up to
    TRIAGE_LINES), else the Jobs-view preview peeks."""
    for screen in reversed(app.screen_stack):
        tail = getattr(screen, "_triage", None) or getattr(screen, "_tail", None)
        if getattr(screen, "key", None) == job.key and tail:
            return [str(x) for x in tail]
    main = app.main_screen()
    peeks: Any = getattr(getattr(main, "jobs", None), "_peeks", None) or {}
    out: list[str] = []
    for ref in job.logs:
        entry = peeks.get((job.key.host, ref.path)) if ref.path else None
        if entry:
            out.extend(entry[0].lines)
    return out


async def fetch_log_lines(app: RunFleetApp, job: Job, nbytes: int, *, streams: Sequence[str] = ("stdout",)) -> \
        list[str]:
    """The last `nbytes` of the job's log files (one on-demand probe per distinct path); [] when unreadable."""
    lines: list[str] = []
    seen: set[str] = set()
    for stream in streams:
        ref = _ref(job, stream)
        if ref is None or not ref.path or ref.path in seen:
            continue
        seen.add(ref.path)
        try:
            path = await app.service.fetch_log(job.key.host, ref, nbytes=nbytes)
        except Exception as exc:               # a missing file is no error here: the menu still offers a port
            log.info("%s: fetch %s: %s", job.key, ref.path, exc)
            continue
        try:
            lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
    return lines


# ---------------------------------------------------------------------------------------------------- copy menu
def copy_menu(app: RunFleetApp) -> None:
    """``Y``: stdout / stderr path, job id, host:jobid, workdir, the node-shell command, a support bundle; each has a
    key (``COPY_KEYS``), ``Y Y`` copies the stdout path."""
    from runfleet.ui.screens.action_menu import ActionMenuScreen

    job = app._current_job()
    if job is None:
        app.toast("no job selected", severity="warning")
        return
    state = app.host_state(job.key.host)
    alias = state.config.ssh if state is not None else job.key.host
    items = copy_items(job, alias=alias, shell_cmd=app.service.job_shell_command(job.key.host, job.key))
    specs = [ActionSpec(i.id, i.label, mutating=False, confirm=False, disabled_reason=i.disabled) for i in items]
    by_id = {i.id: i for i in items}

    def chosen(item_id: str | None) -> None:
        item = by_id.get(item_id or "")
        if item is None:
            return
        if item.id == "bundle":
            app.run_worker(make_bundle(app, job.key), group="bundle", exclusive=True, exit_on_error=False)
        else:
            copy_text(app, item.text, item.what)

    app.push_screen(ActionMenuScreen(specs, 1, title=f"Copy · {job.key.host}:{job.key.label}", keys=COPY_KEYS,
                                     subtitle="↵ / key copy · Y again = stdout path · esc close"), chosen)


async def make_bundle(app: RunFleetApp, key: JobKey) -> None:
    from runfleet import __version__

    job = app.store.job(key)
    if job is None:
        app.toast(f"{key.label} is gone", severity="warning")
        return
    detail: JobDetail | None = None
    try:
        detail = await app.service.job_detail(key)
    except Exception as exc:                   # the bundle still says what the store knows
        log.info("%s: detail for the bundle: %s", key, exc)
    lines = known_log_lines(app, job)
    if len(lines) < BUNDLE_LOG_LINES:
        fetched = await fetch_log_lines(app, job, FETCH_TAIL)
        lines = fetched if len(fetched) > len(lines) else lines
    state = app.host_state(key.host)
    snap = state.snapshot if state is not None else None
    caps = snap.caps if snap is not None else None
    cluster = key.host
    if caps is not None and caps.slurm_version:
        cluster += f" (Slurm {caps.slurm_version}" + (f", cluster {caps.cluster_name}" if caps.cluster_name else "") + ")"
    elif job.kind is JobKind.TASK:
        cluster += " (workstation)"
    text = support_bundle(app.store.job(key) or job, detail, cluster=cluster, log_lines=lines, now=app.now(),
                          version=__version__)
    saved = ""
    directory = app.bundle_dir()
    if directory is not None:
        try:
            from runfleet import paths

            paths.ensure_private_dir(directory)
            target = bundle_path(directory, key)
            paths.write_private(target, text)
            saved = f" · saved {target}"
        except OSError as exc:
            saved = f" · not saved ({exc.strerror or exc})"
    copy_text(app, text, "support bundle",
              message=f"copied support bundle of {key.label} ({len(text.splitlines())} lines){saved}")


# ---------------------------------------------------------------------------------------------------- forwards
def forward_port(app: RunFleetApp) -> None:
    """``P`` / palette: forward a port of the current job (or host) to this machine through the ssh master."""
    job = app._current_job()
    host = job.key.host if job is not None else app._current_host()
    if host is None:
        app.toast("no job or host selected", severity="warning")
        return
    node, why = forward_node(job)
    if node is None:
        app.toast(why or "nothing to forward", severity="warning")
        return
    reason = app.service.forward_blocker(host)
    if reason is not None:
        app.toast(f"port forward unavailable: {reason}", severity="warning")
        return
    app.run_worker(_forward_menu(app, host, node, job), group="forward", exclusive=True, exit_on_error=False)


async def _forward_menu(app: RunFleetApp, host: str, node: str, job: Job | None) -> None:
    from runfleet.ui.screens.action_menu import ActionMenuScreen
    from runfleet.ui.screens.forwards import PortPromptScreen

    cands: list[UrlCandidate] = []
    if job is not None:
        cands = find_urls(known_log_lines(app, job))
        if not cands and job.logs:
            app.toast(f"looking for service URLs in the log of {job.key.label} …")
            cands = find_urls(await fetch_log_lines(app, job, URL_SCAN_BYTES, streams=("stdout", "stderr")))
    specs = [ActionSpec(f"url:{c.port}", candidate_label(c), mutating=False, confirm=False) for c in cands]
    specs.append(ActionSpec("other", "Other port…", mutating=False, confirm=False))
    by_id = {f"url:{c.port}": c for c in cands}
    what = job.key.label if job is not None else host

    def start(port: int | None, cand: UrlCandidate | None = None) -> None:
        if port:
            # not the menu's exclusive group: a new `P` must not cancel a forward ssh is setting up
            app.run_worker(start_forward(app, host, node, port, job, cand), group="forward-start",
                           exit_on_error=False)

    def chosen(choice: str | None) -> None:
        if choice == "other":
            app.push_screen(PortPromptScreen(f"Forward which port of {node}?"), start)
        elif choice in by_id:
            start(by_id[choice].port, by_id[choice])

    app.push_screen(ActionMenuScreen(specs, 1, title=f"Forward a port · {what} → {node}"), chosen)


async def start_forward(app: RunFleetApp, host: str, node: str, port: int, job: Job | None,
                        cand: UrlCandidate | None = None) -> None:
    try:
        fwd = await app.service.forward_port(host, node, port, job=job.key if job is not None else None,
                                             scheme=cand.scheme if cand else "http", path=cand.path if cand else "",
                                             label=cand.label if cand else "")
    except RunFleetError as exc:
        app.toast(f"port forward failed: {exc}", severity="error")
        return
    copy_text(app, fwd.url, "URL", message=f"localhost:{fwd.local_port} → {host} {node}:{port} · copied {fwd.url}")


def show_forwards(app: RunFleetApp) -> None:
    from runfleet.ui.screens.forwards import ForwardsScreen

    app.push_screen(ForwardsScreen())


# ---------------------------------------------------------------------------------------------------- log in to all
def login_targets(app: RunFleetApp) -> list[str]:
    return [h for h, s in app.store.hosts.items() if s.status.state in LOGIN_STATES and s.config.enabled]


async def login_all(app: RunFleetApp) -> None:
    """``C`` / palette: one suspend, every host needing a login in turn ("2/3 vega — approve the Duo push"), then
    re-prepare each and toast a summary."""
    hosts = [h for h in login_targets(app) if h not in app._connecting]
    if not hosts:
        app.toast("no host needs a login")
        return
    app._connecting.update(hosts)
    try:
        steps: list[tuple[str, list[str]]] = []
        already: list[str] = []
        for host in hosts:
            if await app.service.prepare_connect(host):
                already.append(host)
                await app.service.after_connect(host, 0)
                continue
            argv = app.service.connect_argv(host)
            if argv is not None:
                steps.append((host, argv))
        rcs: list[int] = []
        if steps:
            n = len(steps)
            batch = []
            for i, (host, argv) in enumerate(steps, start=1):
                state = app.host_state(host)
                banner = (f"RunFleet → {i}/{n} {host} — approve the Duo push / answer the prompts "
                          "(Ctrl-C skips this host)")
                batch.append((argv, banner, state.status.ssh_log if state is not None else None))
            result = app.run_external_sequence(batch)
            if result is None:                 # no suspend here (web driver): one host at a time instead
                app.toast("cannot suspend this terminal — log in per host (c)", severity="warning")
                return
            rcs = result
        ok = list(already)
        failed: list[str] = []
        for (host, _argv), rc in zip(steps, rcs, strict=False):
            await app.service.after_connect(host, rc)
            if rc == 0:
                ok.append(host)
            else:
                failed.append(f"{host} ({'skipped' if rc == 130 else f'rc {rc}'})")
        parts = []
        if ok:
            parts.append(f"logged in: {', '.join(ok)}")
        if failed:
            parts.append(f"failed: {', '.join(failed)}")
        app.toast(" · ".join(parts) or "nothing to log in", severity="error" if failed else "information")
    finally:
        app._connecting.difference_update(hosts)


# ---------------------------------------------------------------------------------------------------- free GPUs
def free_gpu_menu(app: RunFleetApp) -> None:
    """``a`` (GPUs view, workstation host detail): copy ``CUDA_VISIBLE_DEVICES`` for the N idlest free GPUs."""
    from runfleet.ui.screens.action_menu import ActionMenuScreen
    from runfleet.ui.widgets.host_card import is_cluster

    host = app._current_host()
    state = app.host_state(host) if host else None
    snap = state.snapshot if state is not None else None
    if state is None or snap is None or is_cluster(state) or not snap.gpus:
        app.toast("pick a workstation GPU — Slurm GPUs are handed out by the scheduler", severity="warning")
        return
    if state.stale(app.now()) or gpu_readings_stale(state.status, app.now()):
        app.toast(f"{host}: GPU readings are stale — refresh first (r)", severity="warning")
        return
    free, held = free_gpus(snap.gpus, {j.key: j for j in snap.jobs.values()})
    busy = "; ".join(held)
    if not free:
        app.toast(f"{host}: no free GPU — {busy}", severity="warning", timeout=10)
        return
    specs = [ActionSpec(str(n), f"{n} free GPU{'s' if n != 1 else ''} → {cuda_visible(free[:n])}", mutating=False,
                        confirm=False) for n in range(1, len(free) + 1)]

    def chosen(choice: str | None) -> None:
        if not choice:
            return
        now_state = app.host_state(host) if host else None
        now_snap = now_state.snapshot if now_state is not None else None
        still = {g.index for g in free_gpus(now_snap.gpus)[0]} if now_snap is not None else set()
        taken = [g.index for g in free[: int(choice)] if g.index not in still]
        if taken:                              # a poll while the menu was open: someone started on them
            app.toast(f"{host}: GPU {','.join(map(str, taken))} no longer free — press a again", severity="warning")
            return
        text = cuda_visible(free[: int(choice)])
        copy_text(app, text, "GPUs", message=f"copied {text} on {host}" + (f" · in use: {busy}" if busy else ""))

    app.push_screen(ActionMenuScreen(specs, 1, title=f"Free GPUs on {host} (idlest first)"), chosen)
