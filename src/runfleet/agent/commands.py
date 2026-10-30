"""[WP4] The one dispatcher of `runfleet agent <verb>` (docs/AGENT_INTERFACE.md §3, DESIGN.md §8.7): validates
the parameters, loads the config, builds the fleet view (the cache, or the `--demo` fleet polled in memory),
runs the explicit `--refresh` first, does `log`'s one probe, and returns the agent.view document. Used by the
CLI and by the MCP adapter (agent.mcp) alike; parameters are a plain mapping (JSON-shaped).

Usage errors (unknown verb, host or state, a malformed id or time) raise :class:`UsageError` before any IO.
One end-to-end deadline (``timeout``, default 30 s) covers the refresh, the cache read and the log probe; past it
(plus a grace for cleanup) the answer is a no-data document. Never prompts, never mutates a remote host. Heavy
modules (agent.view / agent.source, transports, the demo) are imported on first use only."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import re
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runfleet.agent.data import Document, FleetView, HostView, Problem, ProblemCode
from runfleet.errors import RunFleetError
from runfleet.util.redact import escape_controls, scrub

if TYPE_CHECKING:
    from runfleet.config import FleetConfig
    from runfleet.core.clock import Clock
    from runfleet.core.logs import LogFiles, LogRead
    from runfleet.core.service import TransportFactory
    from runfleet.models import Job
    from runfleet.wait import Target

VERBS = ("brief", "jobs", "job", "log", "gpus", "host", "events", "help")
DEFAULT_TIMEOUT = 30.0
GRACE = 5.0                          # s past the deadline for cleanup (closing transports) before giving up
JOB_IDS_MAX = 10
LOG_LINES_DEFAULT = 60
LOG_BYTES_DEFAULT, LOG_BYTES_MAX = 32 << 10, 256 << 10
STREAMS = ("auto", "out", "err")
GREP_MAX, GREP_TERMS_MAX = 400, 20     # --grep: the terms are echoed back, inside the log budget
FILE_MAX = 4096                        # --file PATH: characters (PATH_MAX: every listed path fits)
DEMO_SEED = 7
CACHE_THREAD = "runfleet-agent-cache"          # the daemon thread of a cache read (agent.mcp waits for it)
_DUR = re.compile(r"(\d+(?:\.\d+)?)([smhdw])")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


ERROR_TEXT_MAX = 300                 # bytes: an error text that echoes caller input (MCP reply, stderr, runfleet.log)


class UsageError(RunFleetError):
    """A bad verb, parameter, host name, id or time (CLI exit 2); raised before any IO. ``mcp``: the same problem
    in the MCP tools' parameter names when the message names CLI flags (``list_files`` for ``--files``)."""

    def __init__(self, message: str, *, mcp: str | None = None) -> None:
        super().__init__(message)
        self.mcp = mcp


def safe_text(text: str, limit: int = ERROR_TEXT_MAX) -> str:
    """An error text that may echo caller input (a host, id or time as given, a tool or argument name) for an MCP
    client, stderr or runfleet.log: secrets scrubbed (redact.scrub), control characters made visible (newlines,
    ESC, bidi overrides: one log line, nothing a terminal or a model would act on), at most `limit` UTF-8 bytes
    (a cut ends in "…")."""
    # a lone surrogate (valid in JSON: "\\ud800") cannot be encoded: shown as \\ud800, never an exception
    out = escape_controls(scrub(text)).encode("utf-8", "backslashreplace").decode("utf-8")
    raw = out.encode("utf-8")
    if len(raw) <= limit:
        return out
    return raw[:max(0, limit - 3)].decode("utf-8", "ignore") + "…"


# ================================================================================================== parameters
def parse_since(text: str, now: datetime) -> datetime:
    """``2h`` / ``90m`` / ``1d`` / ``1h30m`` before ``now``, or an ISO-8601 time (naive = UTC). ValueError
    otherwise."""
    t = text.strip()
    if t and _DUR.sub("", t) == "":
        seconds = sum(float(n) * _UNIT_S[u] for n, u in _DUR.findall(t))
        try:
            return now - timedelta(seconds=seconds)
        except OverflowError:
            raise ValueError(f"--since {text!r} is too far back") from None
    try:
        at = datetime.fromisoformat(t)
    except ValueError:
        raise ValueError(f"--since expects a duration (2h, 90m, 1d) or an ISO time, got {text!r}") from None
    try:
        return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
    except OverflowError:                    # 0001-01-01T00:00+01:00 has no UTC time
        raise ValueError(f"--since {text!r} is out of range") from None


class _Params:
    """Typed access to the JSON-shaped parameters; a wrong type is a UsageError naming the key."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = raw

    def _get(self, key: str, kinds: type | tuple[type, ...], default: Any) -> Any:
        value = self.raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool) and bool not in (kinds if isinstance(kinds, tuple) else (kinds,)):
            raise UsageError(f"{key}: expected {getattr(kinds, '__name__', kinds)}, got a boolean")
        if not isinstance(value, kinds):
            raise UsageError(f"{key}: expected {getattr(kinds, '__name__', kinds)}, got {type(value).__name__}")
        return value

    def text(self, key: str) -> str | None:
        value: str | None = self._get(key, str, None)
        return value

    def flag(self, key: str) -> bool:
        return bool(self._get(key, bool, False))

    def count(self, key: str, default: int | None, *, minimum: int = 1) -> int | None:
        value: int | None = self._get(key, int, default)
        if value is not None and value < minimum:
            raise UsageError(f"{key}: expected an integer >= {minimum}, got {value}")
        return value

    def names(self, key: str) -> list[str]:
        value = self.raw.get(key)
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, Sequence) or not all(isinstance(v, str) for v in value):
            raise UsageError(f"{key}: expected a list of strings")
        return list(value)

    def seconds(self, key: str, default: float) -> float:
        value = self._get(key, (int, float), default)
        if not math.isfinite(value) or value <= 0:
            raise UsageError(f"{key}: expected a positive number of seconds, got {value}")
        return float(value)


@dataclass(frozen=True, slots=True)
class Request:
    """A validated call: what to load (`hosts`: config names, None = every enabled host) and how to render."""
    verb: str
    hosts: tuple[str, ...] | None
    typed_hosts: tuple[str, ...]              # as given, for "next command" footers
    targets: tuple[Target, ...] = ()          # job / log
    refresh: bool = False
    timeout: float = DEFAULT_TIMEOUT
    limit: int | None = None
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    # the call's own global options, repeated in every printed command (FleetView.config_path / demo_seed)
    config_path: str | None = None            # -c PATH as given (None: the default config, or --demo)
    demo_seed: int | None = None              # --demo-seed N when not the default


def _resolve_hosts(config: FleetConfig, names: Sequence[str]) -> tuple[str, ...]:
    """Config names for HOST arguments (a config name or its ssh alias); UsageError for anything else."""
    out: list[str] = []
    for n in names:
        cfg = next((h for h in config.hosts if h.name == n), None) or \
            next((h for h in config.hosts if h.ssh == n), None)
        if cfg is None:
            known = " ".join(h.name for h in config.hosts) or "none configured"
            raise UsageError(f"unknown host {n!r} (hosts: {known})")
        out.append(cfg.name)
    return tuple(dict.fromkeys(out))


def _targets(config: FleetConfig, texts: Sequence[str]) -> tuple[Target, ...]:
    from runfleet.wait import TargetError, resolve_targets

    try:
        return tuple(resolve_targets(texts, config.hosts))
    except TargetError as exc:
        raise UsageError(str(exc)) from None


def build_request(verb: str, raw: Mapping[str, Any], config: FleetConfig, *,
                  now: datetime | None = None) -> Request:
    """Validate everything a verb takes against the config (host names, ids, states, times; `--since` against
    `now`, the injected clock's time, default the wall clock); UsageError on the first problem. Nothing is read
    or polled here."""
    from runfleet.agent import view

    p = _Params(raw)
    common: dict[str, Any] = {"refresh": p.flag("refresh"), "timeout": p.seconds("timeout", DEFAULT_TIMEOUT),
                              "limit": p.count("limit", None), "params": raw}
    if verb in ("job", "log"):
        texts = p.names("ids") if verb == "job" else [t for t in [p.text("id")] if t]
        if not texts:
            raise UsageError(f"{verb}: give HOST:ID" + (" …" if verb == "job" else ""))
        if verb == "job" and len(texts) > JOB_IDS_MAX:
            raise UsageError(f"job: at most {JOB_IDS_MAX} ids per call, got {len(texts)}")
        targets = _targets(config, texts)
        if verb == "log":
            stream = p.text("stream") or "auto"
            if stream not in STREAMS:
                raise UsageError(f"stream: one of out, err (got {stream!r})")
            p.count("lines", LOG_LINES_DEFAULT)
            p.count("bytes", LOG_BYTES_DEFAULT)
            grep = p.text("grep") or ""
            if len(grep) > GREP_MAX or grep.count("|") >= GREP_TERMS_MAX:
                raise UsageError(f"grep: at most {GREP_TERMS_MAX} alternatives and {GREP_MAX} characters")
            for key in ("head", "path_only"):
                p.flag(key)
            _check_files(p)
        named = tuple(dict.fromkeys(t.host for t in targets))
        return Request(verb, named, named, targets=targets, **common)
    if verb == "host":
        name = p.text("host")
        if not name:
            raise UsageError("host: give HOST")
        return Request(verb, _resolve_hosts(config, [name]), (name,), **common)
    typed = tuple(p.names("hosts"))
    hosts: tuple[str, ...] | None = _resolve_hosts(config, typed) if typed else None
    if verb == "jobs":
        state = p.text("state")
        if state is not None:
            try:
                view.parse_states(state)
            except ValueError as exc:
                raise UsageError(str(exc)) from None
        for key in ("name", "workdir", "partition"):
            p.text(key)
        for key in ("attention", "all_users"):
            p.flag(key)
    elif verb == "events" and not p.text("since"):
        raise UsageError("events: --since is required (a duration like 2h, 90m, 1d, or an ISO time)")
    since = p.text("since") if verb in ("jobs", "events") else None
    if since:
        try:
            parse_since(since, now if now is not None else datetime.now(UTC))
        except ValueError as exc:
            raise UsageError(str(exc)) from None
    elif verb == "gpus":
        p.text("type")
        p.count("min", 0, minimum=0)
    return Request(verb, hosts, typed, **common)


def _check_files(p: _Params) -> None:
    """`log --files` / `--file PATH`: one of them, with the read options that make sense (a UsageError names the
    clash); PATH one line, no NUL, at most FILE_MAX characters (whether it lies under the job's workdir is decided
    once the job is known, before any probe)."""
    listing, file = p.flag("list_files"), p.text("file")
    if file is not None and (not file.strip() or len(file) > FILE_MAX or any(c in file for c in "\x00\n\r")):
        raise UsageError(f"file: a path under the job's workdir (one line, at most {FILE_MAX} characters)")
    if not listing and file is None:
        return
    given = {"list_files": listing, "file": file is not None, "stream": p.text("stream") is not None,
             "path_only": p.flag("path_only"), "grep": bool(p.text("grep")), "head": p.flag("head")}
    allowed = ("list_files",) if listing else ("file", "grep", "head")
    clash = [name for name, on in given.items() if on and name not in allowed]
    if clash:
        mode = "list_files" if listing else "file"
        raise UsageError(f"{_FLAGS[mode]} does not combine with {' '.join(_FLAGS[c] for c in clash)}",
                         mcp=f"{mode} does not combine with {' '.join(clash)}")


_FLAGS = {"list_files": "--files", "file": "--file", "stream": "--stream", "path_only": "--path-only",
          "grep": "--grep", "head": "--head"}       # `agent log` parameters -> their CLI flags (usage errors)


# ================================================================================================== the fleet
def load_config(config_path: Path | None) -> FleetConfig:
    """The config at `config_path` (None: the default file; missing default: no hosts). A named file that does
    not exist is a UsageError; a broken one raises ConfigError (CLI exit 2)."""
    from runfleet.config import load_config as load

    if config_path is not None and not config_path.exists():
        raise UsageError(f"no such config file: {config_path}")
    return load(config_path)


def read_lock_base() -> Path | None:
    """Where instance locks live, for telling who polls a host (source TUI) — without creating the directory
    (a plain read writes nothing): None when it does not exist, so no lock can be held."""
    from runfleet import paths

    try:
        base = paths.mux_dir()
    except OSError:
        return None
    return base / "instance.lock" if base.is_dir() else None


def _ssh_missing(config: FleetConfig, names: Sequence[str] | None) -> bool:
    picked = [h for h in config.hosts if (h.name in names if names else h.enabled)]
    return any(h.ssh != "local" for h in picked) and shutil.which("ssh") is None


def _with_problem(fv: FleetView, problem: Problem) -> FleetView:
    return dataclasses.replace(fv, problems=(*fv.problems, problem))


def _failed_view(config: FleetConfig, now: datetime, error: str, demo: bool) -> FleetView:
    return FleetView(generated_at=now, hosts=(), recent_window=config.general.recent_window, demo=demo, error=error)


def _demo(seed: int, clock: Clock) -> tuple[FleetConfig, TransportFactory]:
    from runfleet.demo.transport import DemoTransport
    from runfleet.demo.world import DemoWorld

    world = DemoWorld(seed=seed, now=clock.now, latency_scale=0.0)       # instant probes: nothing to watch here
    return world.fleet_config(), lambda h: DemoTransport(h, world)


# ================================================================================================== rendering
def render(req: Request, fv: FleetView, read: LogRead | None = None, *, path_only: bool | None = None,
           files: LogFiles | None = None) -> Document:
    """The agent.view document of a request over a fleet view (fv.error set: the verb's no-data document);
    ``files``: the find of `log --files` / `--file`."""
    from runfleet.agent import view

    if req.config_path is not None or req.demo_seed is not None:
        fv = dataclasses.replace(fv, config_path=req.config_path, demo_seed=req.demo_seed)
    p = _Params(req.params)
    if req.verb == "brief":
        return view.brief(fv, hosts=req.typed_hosts)
    if req.verb == "jobs":
        since_text = p.text("since")
        state = p.text("state")
        flt = view.JobFilter(
            states=view.parse_states(state) if state is not None else view.DEFAULT_STATES,
            name_glob=p.text("name"), workdir_prefix=p.text("workdir"), partition=p.text("partition"),
            since=parse_since(since_text, fv.generated_at) if since_text else None,
            attention_only=p.flag("attention"), all_users=p.flag("all_users"), hosts=req.typed_hosts,
            since_text=since_text)
        return view.jobs(fv, flt, **({"limit": req.limit} if req.limit is not None else {}))
    if req.verb == "job":
        return view.job(fv, [(t.host, t.job) for t in req.targets])
    if req.verb == "log":
        t = req.targets[0]
        if p.flag("list_files"):
            return view.log_files(fv, (t.host, t.job), files)
        return view.log(fv, (t.host, t.job), read, lines=p.count("lines", LOG_LINES_DEFAULT) or LOG_LINES_DEFAULT,
                        grep=p.text("grep"), path_only=p.flag("path_only") if path_only is None else path_only,
                        stream=p.text("stream") or "auto", head=p.flag("head"), file=p.text("file"), files=files)
    if req.verb == "gpus":
        return view.gpus(fv, gpu_type=p.text("type"), min_free=p.count("min", 0, minimum=0) or 0,
                         hosts=req.typed_hosts)
    if req.verb == "host":
        assert req.hosts
        return view.host(fv, req.hosts[0])
    if req.verb == "events":
        since_text = p.text("since") or ""
        since = parse_since(since_text, fv.generated_at)
        if req.limit is None:
            return view.events(fv, since, req.typed_hosts, since_text=since_text)
        return view.events(fv, since, req.typed_hosts, since_text=since_text, limit=req.limit)
    raise UsageError(f"unknown verb {req.verb!r}")


# ================================================================================================== dispatch
async def run_async(verb: str, params: Mapping[str, Any], *, config_path: Path | None = None, demo: bool = False,
                    demo_seed: int = DEMO_SEED, clock: Clock | None = None) -> Document:
    """`runfleet agent <verb>` as a document (text, JSON data, exit code). Raises UsageError for bad input and
    ConfigError for a config that does not load; everything else is in the document."""
    from runfleet.agent import view

    if verb not in VERBS:
        raise UsageError(f"unknown verb {verb!r}: one of {' '.join(VERBS)}")
    if verb == "help":
        topic = _Params(params).text("topic")
        try:
            return view.help_text(topic or None)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
    if clock is None:
        from runfleet.core.clock import RealClock
        clock = RealClock()
    transports: TransportFactory | None
    if demo:
        config, transports = _demo(demo_seed, clock)
    else:
        config, transports = load_config(config_path), None
    req = dataclasses.replace(
        build_request(verb, params, config, now=clock.now()),
        config_path=str(config_path) if config_path is not None and not demo else None,
        demo_seed=demo_seed if demo and demo_seed != DEMO_SEED else None)
    if not demo and not config.hosts:
        fv = _failed_view(config, clock.now(), "no hosts configured", demo)
        return render(req, fv)
    try:
        return await asyncio.wait_for(_answer(req, config, demo=demo, clock=clock, transports=transports),
                                      req.timeout + GRACE)
    except TimeoutError:
        fv = _failed_view(config, clock.now(), f"no answer within the {req.timeout:.0f} s deadline", demo)
        return render(req, fv)


def run(verb: str, params: Mapping[str, Any], *, config_path: Path | None = None, demo: bool = False,
        demo_seed: int = DEMO_SEED, clock: Clock | None = None) -> Document:
    """:func:`run_async` in a fresh event loop (the CLI)."""
    return asyncio.run(run_async(verb, params, config_path=config_path, demo=demo, demo_seed=demo_seed,
                                 clock=clock))


async def _answer(req: Request, config: FleetConfig, *, demo: bool, clock: Clock,
                  transports: TransportFactory | None) -> Document:
    from runfleet.agent import source

    started = clock.monotonic()

    def left() -> float:
        return max(0.1, req.timeout - (clock.monotonic() - started))

    names = list(req.hosts) if req.hosts else None
    if demo:
        assert transports is not None
        fv = await source.demo_view(config, transports, clock, names, timeout=left())
        return await _with_log(req, config, fv, clock=clock, transports=transports, timeout=left)
    from runfleet import paths
    from runfleet.core.service import default_lock_base

    cache_path = paths.cache_file()
    outcomes: dict[str, str] | None = None
    no_ssh = False
    if req.refresh:
        if _ssh_missing(config, names):
            no_ssh = True                    # nothing to poll with: the cache as it is, and say why
        else:
            outcomes = await source.refresh(config, names, clock=clock, timeout=left(), cache_path=cache_path,
                                            lock_base=default_lock_base())
    fv = await _in_daemon_thread(lambda: source.load_view(config, names, cache_path=cache_path, now=clock.now(),
                                                          lock_base=read_lock_base(), outcomes=outcomes))
    if no_ssh:
        fv = _with_problem(fv, Problem(ProblemCode.NO_SSH))
    return await _with_log(req, config, fv, clock=clock, transports=None, timeout=left)


async def _in_daemon_thread(fn: Callable[[], FleetView]) -> FleetView:
    """`fn()` in a daemon thread: past the deadline the call is abandoned, and neither asyncio.run's executor
    shutdown nor interpreter exit waits for it (a cache read is bounded by sqlite's busy timeout, its decoding
    is not)."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[FleetView] = loop.create_future()

    def settle(result: FleetView | None, exc: BaseException | None) -> None:
        if future.done():
            return
        if exc is not None:
            future.set_exception(exc)
        else:
            assert result is not None
            future.set_result(result)

    def work() -> None:
        try:
            result = fn()
        except BaseException as exc:         # handed to the awaiting task
            with contextlib.suppress(RuntimeError):         # the loop is gone: nobody waits any more
                loop.call_soon_threadsafe(settle, None, exc)
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(settle, result, None)

    threading.Thread(target=work, name=CACHE_THREAD, daemon=True).start()
    return await future


async def _with_log(req: Request, config: FleetConfig, fv: FleetView, *, clock: Clock,
                    transports: TransportFactory | None, timeout: Callable[[], float]) -> Document:
    """Render; for `log`, first the one bounded read of the log agent.view will show (no probe when the job is
    unknown, ambiguous, someone else's, has no log path, or `path_only`). `--files`: one find under the job's
    workdir instead; `--file PATH`: that find, then the read of PATH only when the find listed it (a PATH outside
    the workdir is refused before any probe)."""
    from runfleet.agent import source, view

    if req.verb != "log" or fv.error is not None:
        return render(req, fv)
    p = _Params(req.params)
    if p.flag("path_only"):
        return render(req, fv)
    t = req.targets[0]
    hv, found, problem = view.lookup(fv, t.host, t.job)
    if problem is not None or hv is None or len(found) != 1 or not found[0].is_mine:
        return render(req, fv)
    file = p.text("file")
    if p.flag("list_files") or file is not None:
        return await _with_files(req, config, fv, hv, found[0], file, clock=clock, transports=transports,
                                 timeout=timeout)
    stream = p.text("stream") or "auto"
    ref = view.pick_log(found[0], stream)
    if ref is None or not ref.path:
        return render(req, fv)
    if transports is None and _ssh_missing(config, [hv.name]):
        return render(req, _with_problem(fv, Problem(ProblemCode.NO_SSH, host=hv.name)), path_only=True)
    nbytes = min(p.count("bytes", LOG_BYTES_DEFAULT) or LOG_BYTES_DEFAULT, LOG_BYTES_MAX)
    read = await source.read_log(config, hv, found[0], stream=stream, nbytes=nbytes, head=p.flag("head"),
                                 clock=clock, timeout=timeout(), transports=transports, log_ref=ref)
    return render(req, fv, read)


async def _with_files(req: Request, config: FleetConfig, fv: FleetView, hv: HostView, job: Job, file: str | None, *,
                      clock: Clock, transports: TransportFactory | None, timeout: Callable[[], float]) -> Document:
    """`log --files` / `--file PATH` of one job of mine: no probe for a job that has not started, has no workdir or
    a PATH outside it, the workdir itself, a directory or another user's job's log (agent.view says why); else the one find (bounded by an
    ended job's end), and for `--file` the read of the listed file."""
    from runfleet.agent import source, view

    p = _Params(req.params)
    workdir = job.workdir
    if not view.started(job) or not workdir or (file is not None and view.file_precheck(job, file, hv) is not None):
        return render(req, fv)
    if transports is None and _ssh_missing(config, [hv.name]):
        return render(req, _with_problem(fv, Problem(ProblemCode.NO_SSH, host=hv.name)))
    files = await source.find_logs(config, hv, job, clock=clock, timeout=timeout(), transports=transports)
    if file is None:
        return render(req, fv, files=files)
    hit, _why = view.file_choice(job, file, files, hv)
    if hit is None:
        return render(req, fv, files=files)
    nbytes = min(p.count("bytes", LOG_BYTES_DEFAULT) or LOG_BYTES_DEFAULT, LOG_BYTES_MAX)
    read = await source.read_log(config, hv, job, nbytes=nbytes, head=p.flag("head"), clock=clock,
                                 timeout=timeout(), transports=transports, log_ref=view.found_ref(hit))
    return render(req, fv, read, files=files)
