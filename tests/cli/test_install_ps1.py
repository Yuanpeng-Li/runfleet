"""install.ps1 (Windows one-line install, README): static checks that hold on any OS.

PowerShell itself parses the script in CI (Windows PowerShell 5.1 and PowerShell 7, `.github/workflows/ci.yml`);
these tests pin what a parser cannot see: the file stays readable by 5.1 (ASCII, no BOM needed), it never closes the
user's window under `irm | iex`, a caller's `$ErrorActionPreference` cannot change its flow, it uses no
PowerShell 7-only operators, and it keeps install.sh's "installed already" logic.
"""

from __future__ import annotations

import re

from tests.conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "install.ps1"


def _code(text: str) -> str:
    """The script with comments and string contents blanked (a rough PowerShell lexer: '…' with '', "…" with
    backtick escapes and $( ) nesting), so operator checks only see code."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "`":
            out.append("  ")
            i += 2
            continue
        if c == "'":
            j = i + 1
            while j < n and not (text[j] == "'" and text[j + 1:j + 2] != "'"):
                j += 2 if text[j:j + 2] == "''" else 1
            out.append("''")
            i = j + 1
            continue
        if c == '"':
            j, depth = i + 1, 0
            while j < n and not (text[j] == '"' and depth == 0):
                if text[j] == "`":
                    j += 1
                elif text[j:j + 2] == "$(":
                    depth += 1
                    j += 1
                elif text[j] == ")" and depth:
                    depth -= 1
                j += 1
            out.append('""')
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def test_readable_by_windows_powershell_5_1() -> None:
    data = SCRIPT.read_bytes()
    assert data.isascii()                      # 5.1 reads a BOM-less script in the ANSI code page
    assert b"\r" not in data                   # byte-exact checkout (.gitattributes `* -text`)


def test_one_block_never_exits_the_users_shell() -> None:
    text = SCRIPT.read_text(encoding="ascii")
    code = _code(text)
    body = [line for line in code.splitlines() if line.strip()]
    assert body[0].strip() == "& {" and body[-1].strip() == "}"       # variables and functions stay local
    assert not re.search(r"\bexit\b", code, re.I)                    # `irm | iex`: exit would close the window
    assert code.count("{") == code.count("}") and code.count("(") == code.count(")")


def test_callers_error_preference_cannot_change_the_flow() -> None:
    code = _code(SCRIPT.read_text(encoding="ascii"))
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    assert lines[:2] == ["& {", "$ErrorActionPreference = ''"]       # first thing in the block ('Continue')
    assert "$ErrorActionPreference = 'Continue'" in SCRIPT.read_text(encoding="ascii")


def test_no_powershell_7_only_syntax() -> None:
    code = _code(SCRIPT.read_text(encoding="ascii"))
    for token in ("&&", "||", "??", "?.", "?[", "-Parallel", "$IsWindows", "-AsHashtable", "-SkipCertificateCheck"):
        assert token not in code, token
    assert not re.search(r"\s\?\s[^:]+\s:\s", code)                   # a ? b : c
    assert not re.search(r"Join-Path([ \t]+[^\s(){}|;]+){3,}", code)     # Join-Path with several children: 6.0+


def test_installed_copy_is_upgraded_here_and_fresh_installs_from_github() -> None:
    ps1 = SCRIPT.read_text(encoding="ascii")
    sh = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "https://github.com/Yuanpeng-Li/runfleet" in ps1 and "github.com/Yuanpeng-Li/runfleet" in sh
    # installed already: upgraded right here (on Windows `runfleet update` only prints the steps), never while running
    assert "& $rf update" not in ps1
    code = ps1[ps1.index("& {"):]                                     # past the header comment
    assert code.index("Get-Process -Name runfleet") < code.index("tool upgrade runfleet")
    # by uv or pipx, whichever installed it; else a fresh `uv tool install git+…`
    assert "tool upgrade runfleet" in ps1 and "reinstall runfleet" in ps1 and 'tool install "git+$Repo"' in ps1
    # Git for Windows is a prerequisite of git installs, checked before uv is installed
    assert ps1.index("Get-Command git") < ps1.index("installing uv")


def test_readme_one_liner_matches_the_script_header() -> None:
    line = ('powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/'
            'install.ps1 | iex"')
    assert line in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert line in SCRIPT.read_text(encoding="ascii")


def test_lexer_blanks_strings_and_comments() -> None:
    assert _code("a # x && y\n'p''q && r' \"s `\" $(t && u) v\" w") == "a \n'' \"\" w"
