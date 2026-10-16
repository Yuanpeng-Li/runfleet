"""[WP1] The one place for operating-system differences (POSIX vs native Windows, DESIGN.md §1.2, §4.2, §8.1).

Every POSIX-only API RunFleet needs locally (``fcntl``, ``os.killpg``, ``os.getuid``, ``signal.SIGKILL``, process
groups) is called here and nowhere else (``tests/unit/test_contracts.py`` enforces it; ``ui/termbg.py`` keeps its own
guarded ``termios`` code). Callers read :data:`IS_WINDOWS` through the module (``platform.IS_WINDOWS``), so tests on
Linux exercise the Windows branches by patching that one attribute. Windows-only modules (``msvcrt``) are imported
when used, so a fake in ``sys.modules`` stands in for them on Linux.

Stdlib only, and `subprocess` only inside the functions that run something: backends (sans-io) import the signal
table from here. Remote hosts are Linux whatever the local OS: signal numbers in Slurm exit codes and remote logs are named from the
static Linux table :data:`LINUX_SIGNALS`, never from the local ``signal`` module.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import re
import shlex
import signal
import sys
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

IS_WINDOWS: bool = sys.platform == "win32"

# Win32 process creation flags (subprocess has them only on Windows)
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# the strongest local signal: SIGKILL on POSIX; Windows has none (os.kill with any other number is TerminateProcess)
KILL_SIGNAL: int = int(getattr(signal, "SIGKILL", signal.SIGTERM))

# msvcrt.locking locks a byte range, and a locked range cannot be read by anyone else: lock one sentinel byte far past
# the pid an instance lock file holds at offset 0, so a second instance can still read whose lock it is
LOCK_OFFSET = 1 << 20
_LK_UNLCK, _LK_NBLCK = 0, 2                           # msvcrt.LK_UNLCK / LK_NBLCK

# Reserved DOS device names: a file or directory called `con` or `nul.txt` cannot be created on Windows, and a
# trailing dot is silently dropped (host names become file and directory names, §8.1)
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))})

LINUX_SIGNALS: dict[int, str] = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL", 5: "SIGTRAP", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE",
    9: "SIGKILL", 10: "SIGUSR1", 11: "SIGSEGV", 12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
    16: "SIGSTKFLT", 17: "SIGCHLD", 18: "SIGCONT", 19: "SIGSTOP", 20: "SIGTSTP", 21: "SIGTTIN", 22: "SIGTTOU",
    23: "SIGURG", 24: "SIGXCPU", 25: "SIGXFSZ", 26: "SIGVTALRM", 27: "SIGPROF", 28: "SIGWINCH", 29: "SIGIO",
    30: "SIGPWR", 31: "SIGSYS", 34: "SIGRTMIN", 64: "SIGRTMAX",
}
SIGSEGV_LINUX = 11


def linux_signal_name(n: int) -> str:
    """`SIGKILL` for 9, ``signal 35`` for a number without a name (Linux numbering, any local OS)."""
    return LINUX_SIGNALS.get(n, f"signal {n}")


# ---------------------------------------------------------------- processes
class Killable(Protocol):
    """What :func:`kill_tree` needs: `asyncio.subprocess.Process` and `subprocess.Popen` both are one."""

    @property
    def pid(self) -> int: ...
    @property
    def returncode(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...
    def kill(self) -> None: ...


def background_spawn_kwargs() -> dict[str, Any]:
    """Popen keywords for a process that must never see the terminal: a new session on POSIX (no controlling tty,
    so nothing can open /dev/tty; its own process group for :func:`kill_tree`); on Windows no console window and its
    own process group (Ctrl-C in the TUI's console is not delivered to it)."""
    if IS_WINDOWS:
        return {"creationflags": CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def login_spawn_kwargs() -> dict[str, Any]:
    """Popen keywords for a session login (§4.8) started under the suspended TUI: it keeps the terminal, so ssh can
    prompt there (POSIX `/dev/tty`, Windows the console via `_getwch`). Windows: its own process group, so a later
    Ctrl-C in the TUI's console (a pager, a shell) is not delivered to it; POSIX: nothing (a new process group could
    not read the terminal: SIGTTIN), see :func:`login_argv` instead."""
    if IS_WINDOWS:
        return {"creationflags": CREATE_NEW_PROCESS_GROUP}
    return {}


def login_argv(argv: Sequence[str]) -> list[str]:
    """`argv` for a session login (§4.8). POSIX: run through `sh -c 'trap "" INT QUIT; exec "$@"'`: ssh stays in
    RunFleet's foreground process group (it must read the terminal) but ignores the SIGINT a later Ctrl-C in a pager
    or shell sends to that group; OpenSSH keeps an ignored SIGINT ignored, and a Ctrl-C during the login itself
    still aborts it (RunFleet gets KeyboardInterrupt and ends it). Windows: as is (:func:`login_spawn_kwargs`)."""
    if IS_WINDOWS:
        return list(argv)
    return ["/bin/sh", "-c", 'trap "" INT QUIT; exec "$@"', "runfleet-login", *argv]


@contextlib.contextmanager
def _sigint_handler(handler: Any) -> Iterator[None]:
    old: Any = None
    if threading.current_thread() is threading.main_thread():
        with contextlib.suppress(ValueError, OSError):
            old = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        if old is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, old)


def _ignore_sigint(signum: int, frame: Any) -> None:
    """A Python-level no-op (not SIG_IGN, which a child would inherit across exec)."""


def ctrl_c_raises() -> contextlib.AbstractContextManager[None]:
    """While suspended (the TUI is not reading keys, the terminal is cooked): Ctrl-C raises KeyboardInterrupt, POSIX
    and Windows alike. `asyncio.run` (Textual's `App.run`) installs a SIGINT handler that only CANCELS the main task
    on the first Ctrl-C (no exception), which would let a blocking login run on and end RunFleet after the resume.
    Main thread only (elsewhere a no-op)."""
    return _sigint_handler(signal.default_int_handler)


def ctrl_c_to_child() -> contextlib.AbstractContextManager[None]:
    """While a foreground command runs in the real terminal (pager, editor, shell): Ctrl-C is the command's alone;
    RunFleet neither ends nor kills it (`less` uses Ctrl-C to stop following a file). The child still gets its
    default SIGINT disposition: a Python handler, unlike SIG_IGN, is not inherited across exec."""
    return _sigint_handler(_ignore_sigint)


# Win32 Job Object (KILL_ON_JOB_CLOSE): every background process RunFleet starts on Windows is assigned to one job
# that RunFleet's process alone holds a handle to; when RunFleet exits, crashes or is killed, Windows closes the
# handle and ends every process still in the job, including the ssh.exe a ProxyCommand hop started (children join
# their parent's job). POSIX needs none: a session's ssh sees EOF on stdin when RunFleet goes away.
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001
_job_handle: int | None = None


def _job_info_type() -> Any:
    import ctypes

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                                                          "OtherOperationCount", "ReadTransferCount",
                                                          "WriteTransferCount", "OtherTransferCount")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    return ExtendedLimits


def _kernel32() -> Any:
    """kernel32 with handle-sized return types (tests replace this function with a fake)."""
    import ctypes

    k32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    return k32


def kill_on_exit(pid: int) -> bool:
    """Windows: put process `pid` into RunFleet's KILL_ON_JOB_CLOSE Job Object (created on first use), so it cannot
    outlive RunFleet. True when assigned; False on POSIX (nothing to do) or when Windows refused (a process that
    already ended, or a job that forbids nesting on Windows 7).
    A child the process started before this call (a ProxyCommand hop in the first milliseconds) stays outside."""
    global _job_handle
    if not IS_WINDOWS:
        return False
    import ctypes

    try:
        k32 = _kernel32()
        if _job_handle is None:
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return False
            info = _job_info_type()()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                               ctypes.sizeof(info)):
                k32.CloseHandle(job)
                return False
            _job_handle = int(job)            # never closed: closing it is what ends the processes, at exit
        proc = k32.OpenProcess(_PROCESS_SET_QUOTA_AND_TERMINATE, False, pid)
        if not proc:
            return False
        try:
            return bool(k32.AssignProcessToJobObject(_job_handle, proc))
        finally:
            k32.CloseHandle(proc)
    except (OSError, AttributeError, TypeError, ValueError):
        return False


def kill_tree(proc: Killable, *, hard: bool) -> None:
    """End `proc` and what it started. POSIX: SIGTERM (`hard`: SIGKILL) to the process group it leads (started with
    :func:`background_spawn_kwargs`), else to the process alone. Windows has no signals worth the name: `taskkill /F
    /T` ends the tree (while `proc` is alive the tree is still linked to it), then TerminateProcess; `hard` is
    implied. Never raises for a process that is already gone."""
    if proc.returncode is not None:
        return
    if IS_WINDOWS:
        taskkill_tree(proc.pid)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            proc.kill()
        return
    sig = KILL_SIGNAL if hard else signal.SIGTERM
    try:
        _killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


def kill_group(pgid: int) -> None:
    """SIGKILL the process group `pgid` even after its leader exited (a shell gone, `cmd &` still running). Windows:
    `taskkill /F /T` on the pid (children of a leader that already exited are not found; the Job Object of
    :func:`kill_on_exit` still ends them when RunFleet exits). Raises ProcessLookupError / PermissionError like
    `os.killpg` on POSIX."""
    if IS_WINDOWS:
        taskkill_tree(pgid)
        return
    _killpg(pgid, KILL_SIGNAL)


def _killpg(pgid: int, sig: int) -> None:
    killpg = getattr(os, "killpg")                    # POSIX only (mypy --platform win32 has no os.killpg)
    killpg(pgid, sig)


def taskkill_tree(pid: int) -> int | None:
    """`taskkill /F /T /PID <pid>` (Windows): its exit code, None if it could not run. Blocks for well under a
    second; only used when a probe, stream or hook must be killed."""
    import subprocess

    kwargs: dict[str, Any] = {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    try:
        done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False, **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode


def current_uid() -> int | None:
    """This user's numeric uid (POSIX); None on Windows, which has no uids."""
    if IS_WINDOWS:
        return None
    getuid = getattr(os, "getuid")
    return int(getuid())


def is_root() -> bool:
    """Running as root (POSIX euid 0): permission tests are meaningless then. Never on Windows."""
    if IS_WINDOWS:
        return False
    return int(getattr(os, "geteuid")()) == 0


# ---------------------------------------------------------------- locks
def try_lock(fd: int) -> bool:
    """Take an exclusive lock on the open file `fd` without waiting; False when another process holds it.

    POSIX: `flock` on the whole file. Windows: `msvcrt.locking` of the byte at :data:`LOCK_OFFSET` (bytes before it,
    e.g. a pid at offset 0, stay readable to others); the file position is back at 0 afterwards. Either lock ends when
    the descriptor is closed (call :func:`unlock` first on Windows to release it at once). Other errors raise OSError.
    """
    if IS_WINDOWS:
        msvcrt = importlib.import_module("msvcrt")
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, _LK_NBLCK, 1)
        except PermissionError:                       # EACCES: locked by another process
            return False
        except OSError as e:
            if e.errno in (13, 36):                   # EACCES, EDEADLOCK (MSVC: 36)
                return False
            raise
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
        return True
    fcntl = importlib.import_module("fcntl")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def unlock(fd: int) -> None:
    """Release a lock :func:`try_lock` took on `fd` (closing `fd` releases it too). Never raises."""
    with contextlib.suppress(OSError):
        if IS_WINDOWS:
            msvcrt = importlib.import_module("msvcrt")
            pos = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            try:
                msvcrt.locking(fd, _LK_UNLCK, 1)
            finally:
                os.lseek(fd, pos, os.SEEK_SET)
            return
        fcntl = importlib.import_module("fcntl")
        fcntl.flock(fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------- command lines
def split_command(text: str, *, comments: bool = False) -> list[str]:
    """A command line from a setting or an environment variable (`$PAGER`, `$EDITOR`, an ssh_config line) as argv.

    POSIX: shell words (`shlex.split`). Windows: backslashes are path separators, not escapes, so the words are
    split without POSIX escapes and a pair of double (or single) quotes around a word is removed
    (`"C:\\Program Files\\Notepad++\\notepad++.exe" -multiInst`). Raises ValueError on unbalanced quotes."""
    if not IS_WINDOWS:
        return shlex.split(text, comments=comments)
    lexer = shlex.shlex(text, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = "#" if comments else ""
    out: list[str] = []
    for word in lexer:
        if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
            word = word[1:-1]
        out.append(word)
    return out


def display_argv(argv: Sequence[str]) -> str:
    """A command as the user would type it in this OS's shell (shell quoting on POSIX, CreateProcess quoting on
    Windows: what both cmd.exe and PowerShell accept for plain paths and words)."""
    if IS_WINDOWS:
        import subprocess

        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def resolve_program(argv: Sequence[str]) -> list[str]:
    """`argv` ready for `subprocess` with a program the user named (pager, editor, a notify hook). POSIX: as is.
    Windows: a bare name is looked up on PATH with PATHEXT (`shutil.which`), because CreateProcess only appends
    `.exe`: `more` is `more.com`, VS Code's `code` is `code.cmd`. Not found: unchanged (the start reports it)."""
    out = list(argv)
    if not IS_WINDOWS or not out or "/" in out[0] or "\\" in out[0]:
        return out
    import shutil

    found = shutil.which(out[0])
    if found:
        out[0] = found
    return out


def chain(commands: Sequence[str]) -> str:
    """Commands to run one after another, as one line to paste: `a && b` (POSIX shells); Windows PowerShell 5.1
    has no `&&`, so `a; b` there."""
    return ("; " if IS_WINDOWS else " && ").join(commands)


# ---------------------------------------------------------------- names
def windows_reserved_name(name: str) -> bool:
    """`name` cannot be a file or directory name on Windows: a DOS device (`con`, `nul.log`, `COM1`) or a trailing
    dot (Windows drops it)."""
    return name.endswith(".") or name.split(".", 1)[0].upper() in _RESERVED


def safe_file_name(name: str) -> str:
    """`name` itself, or on Windows `name_` when Windows would refuse it as a file name (see above)."""
    if IS_WINDOWS and windows_reserved_name(name):
        return re.sub(r"\.+$", "", name) + "_"
    return name


def ssh_version_ok(version_text: str) -> bool | None:
    """`ssh -V` of the Windows OpenSSH client (`OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2`): True from 8.1 on (the
    floor this preview assumes: older clients mishandle a ProxyCommand), False below, None for another ssh (Git for
    Windows, MSYS: not what RunFleet expects on Windows) or an unreadable answer."""
    m = re.search(r"OpenSSH_for_Windows_(\d+)\.(\d+)", version_text)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2))) >= (8, 1)
