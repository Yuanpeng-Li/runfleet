"""[WP2] Shell-integration harness: render real probes with WP1's framing and run them through real shells with the
mock commands of tests/mockbin first on PATH (DESIGN.md §10.4).

`MockEnv` links the requested mocks into a private bin directory, points them at the fixtures and collects their
argv log; `run_sections` renders a probe, feeds it to `<shell> -s` and parses the framed output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from runfleet.probe import framing
from runfleet.probe.section import ProbeScript, Section, SectionResult
from tests.conftest import REPO_ROOT, SAMPLES_DIR

MOCKBIN = REPO_ROOT / "tests" / "mockbin"
SLURM_MOCKS = ("squeue", "sacct", "scontrol", "sinfo", "sstat", "sacctmgr", "scancel", "sprio", "sshare")
# System tools the probes and mocks use. Tests run with PATH = mocks + links to just these, so a real `squeue` or
# `nvidia-smi` on the dev box can never leak into a test.
SYSTEM_TOOLS = ("sh", "dash", "bash", "env", "awk", "grep", "cut", "tr", "sed", "date", "id", "hostname", "mktemp",
                "rm", "cat", "head", "tail", "timeout", "sort", "paste", "wc", "sleep", "ps", "stat", "readlink",
                "nproc", "pgrep", "kill", "dirname", "less", "printf", "true", "uniq")


def shells() -> list[str]:
    """POSIX shells to run probes through: /bin/sh (dash on Debian), dash and bash (/bin/sh on the Rocky clusters)."""
    found = ["/bin/sh"]
    for name in ("dash", "bash"):
        path = shutil.which(name)
        if path and os.path.realpath(path) not in {os.path.realpath(p) for p in found}:
            found.append(path)
    return found


@dataclass
class MockEnv:
    root: Path
    host: str
    mocks: Sequence[str] = SLURM_MOCKS
    extra_env: dict[str, str] = field(default_factory=dict)
    sh: str = "/bin/sh"          # what `sh` means inside the probe (dash on Debian, bash on the Rocky clusters)

    def __post_init__(self) -> None:
        self.bin = self.root / "bin"
        self.bin.mkdir(parents=True, exist_ok=True)
        for name in self.mocks:
            link = self.bin / name
            if not link.exists():
                link.symlink_to(MOCKBIN / name)
        self.sysbin = self.root / "sysbin"
        self.sysbin.mkdir(exist_ok=True)
        for tool in SYSTEM_TOOLS:
            real = os.path.realpath(self.sh) if tool == "sh" else shutil.which(tool)
            link = self.sysbin / tool
            if real and not link.exists():
                link.symlink_to(real)
        self.log = self.root / "mock.log"
        self.log.touch()

    def env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update({
            "PATH": f"{self.bin}{os.pathsep}{self.sysbin}",
            "RF_MOCK_BIN": str(MOCKBIN), "RF_MOCK_SAMPLES": str(SAMPLES_DIR), "RF_MOCK_HOST": self.host,
            "RF_MOCK_LOG": str(self.log), "TMPDIR": str(self.root),
        })
        env.update(self.extra_env)
        env.update(overrides)
        return env

    def calls(self, name: str | None = None) -> list[list[str]]:
        """Logged argv lists (without ENV lines), optionally only of one command."""
        out = []
        for line in self.log.read_text().splitlines():
            parts = line.split("\t")
            if parts[0] != "ENV" and (name is None or parts[0] == name):
                out.append(parts)
        return out

    def env_lines(self) -> list[dict[str, str]]:
        out = []
        for line in self.log.read_text().splitlines():
            parts = line.split("\t")
            if parts[0] == "ENV":
                out.append({"cmd": parts[1], **dict(p.split("=", 1) for p in parts[2:])})
        return out

    def clear(self) -> None:
        self.log.write_text("")


def script_for(sections: Iterable[Section], purpose: str = "poll") -> ProbeScript:
    return framing.make_script(list(sections), purpose)


def run_script(script: ProbeScript, env: Mapping[str, str], *, shell: str = "/bin/sh",
               timeout: float | None = None) -> dict[str, SectionResult]:
    """Render `script`, run it with `<shell> -s` (stdin = the script, like ssh does) and parse the framed output."""
    text = framing.render(script)
    limit = timeout if timeout is not None else framing.local_timeout(script)
    try:
        proc = subprocess.run([shell, "-s"], input=text.encode(), capture_output=True, env=dict(env), timeout=limit,
                              start_new_session=True)
        stdout, timed_out = proc.stdout, False
    except subprocess.TimeoutExpired as e:
        stdout, timed_out = e.stdout or b"", True
    return dict(framing.parse(stdout, script, timed_out=timed_out).sections)


def run_sections(sections: Sequence[Section], env: Mapping[str, str], *, shell: str = "/bin/sh",
                 purpose: str = "poll") -> dict[str, SectionResult]:
    return run_script(script_for(sections, purpose), env, shell=shell)
