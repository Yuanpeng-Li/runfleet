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
    assert body[0].strip() == "param([switch]$Uninstall)"            # `[scriptblock]::Create((irm …)) -Uninstall`
    assert body[1].strip() == "& {" and body[-1].strip() == "}"       # variables and functions stay local
    assert not re.search(r"\bexit\b", code, re.I)                    # `irm | iex`: exit would close the window
    assert code.count("{") == code.count("}") and code.count("(") == code.count(")")


def test_callers_error_preference_cannot_change_the_flow() -> None:
    code = _code(SCRIPT.read_text(encoding="ascii"))
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    assert lines[1:3] == ["& {", "$ErrorActionPreference = ''"]      # first thing in the block ('Continue')
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
    url = "https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.ps1"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for line in (f'powershell -ExecutionPolicy ByPass -c "irm {url} | iex"',
                 f'powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm {url}))) -Uninstall"'):
        assert line in readme and line in SCRIPT.read_text(encoding="ascii"), line


def _blocks(code: str) -> dict[str, str]:
    """The body of each `if (...) {` / `function Name(...) {` block whose header contains a key, by brace matching."""
    out: dict[str, str] = {}
    for key in ("function Update-Skill", "if ($Uninstall)", "$rf = Find-Bin 'runfleet'\n    if ($rf)"):
        start = code.index("{", code.index(key))
        depth, i = 0, start
        while True:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            if depth == 0:
                break
            i += 1
        out[key] = code[start:i + 1]
    return out


def test_skill_is_refreshed_after_every_install_and_upgrade() -> None:
    """§8.7: after `uv tool upgrade` / `pipx reinstall` and after a fresh install, `skill update --if-installed` runs,
    behind a capability check (`skill update --help` mentions --if-installed); its failure only says so."""
    ps1 = SCRIPT.read_text(encoding="ascii")
    blocks = _blocks(ps1[ps1.index("& {"):])
    refresh = blocks["function Update-Skill"]
    assert refresh.index("skill update --help") < refresh.index("skill update --if-installed")
    assert "-match '--if-installed'" in refresh and "Fail" not in refresh
    upgrade = blocks["$rf = Find-Bin 'runfleet'\n    if ($rf)"]
    assert upgrade.index("tool upgrade runfleet") < upgrade.index("Update-Skill $rf") < upgrade.index("return")
    fresh = ps1[ps1.index('tool install "git+$Repo"'):]
    assert fresh.count("Update-Skill") == 2                           # on PATH already, or in the bin dir


def test_uninstall_removes_the_skill_then_the_tool_and_keeps_user_dirs() -> None:
    ps1 = SCRIPT.read_text(encoding="ascii")
    body = _blocks(ps1[ps1.index("& {"):])["if ($Uninstall)"]
    # never while RunFleet runs; the skill (when this version has `skill uninstall`) before the tool
    assert body.index("Get-Process -Name runfleet") < body.index("skill uninstall --help") \
        < body.index("& $rf skill uninstall\n") < body.index("tool uninstall runfleet") < body.index("return")
    assert "-match '--dry-run'" in body and "pipx).Source '' uninstall runfleet" in body
    # config and state are kept: their paths and how to delete them are printed, never deleted here
    assert "$env:APPDATA" in body and "$env:LOCALAPPDATA" in body and "Remove-Item -Recurse -Force -LiteralPath $quoted" in body
    assert "-replace \"'\", \"''\"" in body                         # O'Brien's home: quotes doubled
    assert "Remove-Item" not in _code(body)
    # the uninstall branch comes before anything that installs
    assert ps1.index("if ($Uninstall)") < ps1.index("installing uv")


def test_lexer_blanks_strings_and_comments() -> None:
    assert _code("a # x && y\n'p''q && r' \"s `\" $(t && u) v\" w") == "a \n'' \"\" w"
