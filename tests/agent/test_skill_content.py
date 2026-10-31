"""[WP4] agent.skill content: the packaged `runfleet` skill is portable across coding agents — Agent Skills spec
frontmatter (strict YAML subset, cross-checked with PyYAML when present), agent-neutral body with fallbacks for
Claude-only features, the runfleet-not-found removal fallback, Codex UI metadata, and reference links."""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path

import pytest

from runfleet.agent import skill

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = Path(str(resources.files("runfleet.agent").joinpath("skill")))
FILES = {"SKILL.md", "references/reading-output.md", "references/with-run-skills.md", "agents/openai.yaml"}
SPEC_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
HOST_PREFIX_MAX = 200                     # install caps the host-name prefix of the description at ~200 characters
TOOL = re.compile(r"Bash\(runfleet (agent|wait|status) \*\)|Read")


def _read(rel: str) -> str:
    return (SKILL_DIR / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------- strict YAML subset
_INDICATORS = tuple("-?:,[]{}#&*!|>'\"%@`")
_NON_STRING = re.compile(r"(?i)~|null|true|false|yes|no|on|off|y|n|[-+]?(\d[\d_]*(\.\d*)?|\.\d+)(e[-+]?\d+)?|"
                         r"0x[0-9a-f]+|0o[0-7]+|\.inf|\.nan|\d{4}-\d\d-\d\d.*")


def _plain(value: str, *, flow: bool = False) -> str:
    """`value` as a YAML plain scalar that is a string in every YAML 1.1/1.2 loader, or AssertionError."""
    assert value and value == value.strip(), f"empty or padded scalar {value!r}"
    assert not value.startswith(_INDICATORS), f"scalar starts with an indicator: {value!r}"
    assert ": " not in value and " #" not in value and not value.endswith(":"), f"not plain-safe: {value!r}"
    assert "\t" not in value and value.isprintable(), f"control characters in {value!r}"
    assert not _NON_STRING.fullmatch(value), f"{value!r} would not load as a string"
    if flow:
        assert not re.search(r"[,\[\]{}]", value), f"flow indicator inside {value!r}"
    return value


def _quoted(value: str) -> str:
    assert len(value) >= 2 and value[0] == value[-1] == '"', f"not a double-quoted string: {value!r}"
    inner = value[1:-1]
    assert '"' not in inner and "\\" not in inner and inner.isprintable(), f"unsafe quoted string {value!r}"
    return inner


def _scalar(value: str) -> object:
    if value.startswith("["):
        assert value.endswith("]") and value.count("[") == value.count("]") == 1, f"bad flow sequence {value!r}"
        return [_plain(item, flow=True) for item in value[1:-1].split(", ")]
    return _quoted(value) if value.startswith('"') else _plain(value)


def strict_yaml(text: str) -> dict[str, object]:
    """Parse the narrow YAML subset the skill files use, rejecting anything a loader could read differently:
    top-level `key: scalar` (plain string, double-quoted string, one-line flow sequence of plain strings) and
    `key:` followed by a mapping of 2-space-indented `key: "string"` lines."""
    out: dict[str, object] = {}
    nested: dict[str, str] | None = None
    for line in text.split("\n"):
        assert line == line.rstrip() and "\t" not in line and "\r" not in line, f"whitespace in {line!r}"
        if not line:
            continue
        if line.startswith("  "):
            assert nested is not None, f"indented line outside a mapping: {line!r}"
            k, sep, v = line[2:].partition(": ")
            assert sep and re.fullmatch(r"[a-z][a-z0-9_-]*", k) and k not in nested, f"bad nested line {line!r}"
            nested[k] = _quoted(v)
            continue
        k, sep, v = line.partition(":")
        assert sep and re.fullmatch(r"[a-z][a-z0-9_-]*", k) and k not in out, f"bad line {line!r}"
        if v == "":
            out[k] = nested = {}
            continue
        assert v.startswith(" ") and not v.startswith("  "), f"bad separator in {line!r}"
        out[k], nested = _scalar(v[1:]), None
    assert all(v != {} for v in out.values()), "empty mapping"
    return out


def _split(text: str) -> tuple[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with a frontmatter block"
    head, sep, body = text[4:].partition("\n---\n")
    assert sep, "unterminated frontmatter"
    return head, body


def _front() -> tuple[dict[str, object], str]:
    head, body = _split(_read("SKILL.md"))
    return strict_yaml(head), body


def _yaml_agrees(text: str, parsed: dict[str, object]) -> None:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:                   # not a dev dependency; the strict subset parser is the gate
        return
    assert yaml.safe_load(text) == parsed


@pytest.mark.parametrize("bad", [
    "description: a: b", "description: x #c", "name: 'q'", "metadata:\n  runfleet-skill: 2", "name: yes",
    "allowed-tools:\n  - Read", "allowed-tools: [a, b,c]", "name: a\nname: b", "license:  MIT", "x: [a, {b}]",
    "metadata:", "description: *alias", "name:\trunfleet",
])
def test_strict_parser_rejects_ambiguous_yaml(bad: str) -> None:
    with pytest.raises(AssertionError):
        strict_yaml(bad)


def test_strict_parser_reads_the_supported_forms() -> None:
    text = 'name: run-fleet\ntools: [Bash(x *), Read]\nmetadata:\n  a: "2"\n  b-c: "x y"\nd: "q"'
    parsed = strict_yaml(text)
    assert parsed == {"name": "run-fleet", "tools": ["Bash(x *)", "Read"], "metadata": {"a": "2", "b-c": "x y"},
                      "d": "q"}
    _yaml_agrees(text, parsed)


# ---------------------------------------------------------------- frontmatter: Agent Skills spec
def test_skill_folder_holds_only_the_shipped_files() -> None:
    found = {p.relative_to(SKILL_DIR).as_posix() for p in SKILL_DIR.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    assert found == FILES
    for rel in FILES:
        raw = (SKILL_DIR / rel).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n") and not raw.startswith(b"\xef\xbb\xbf"), rel
        raw.decode("utf-8")


def test_frontmatter_is_strict_yaml_and_spec_compliant() -> None:
    head, _ = _split(_read("SKILL.md"))
    top = strict_yaml(head)
    _yaml_agrees(head, top)
    assert set(top) <= SPEC_KEYS and {"name", "description", "compatibility", "license", "allowed-tools",
                                      "metadata"} <= set(top)
    assert "<" not in head and ">" not in head
    name = top["name"]
    assert isinstance(name, str) and 1 <= len(name) <= 64 and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name)
    assert name == "runfleet" == getattr(skill, "SKILL_NAME", "runfleet")        # = the installed folder's name
    compat = top["compatibility"]
    assert isinstance(compat, str) and 1 <= len(compat) <= 500
    assert "runfleet CLI on PATH" in compat and "read-only" in compat and "any agent" in compat
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert top["license"] == project["license"] == "MIT"


def test_metadata_is_a_string_map_with_the_skill_revision() -> None:
    meta = _front()[0]["metadata"]
    assert isinstance(meta, dict) and meta and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    assert meta["runfleet-skill"] == "3"


def test_allowed_tools_is_a_list_of_narrow_patterns() -> None:
    tools = _front()[0]["allowed-tools"]
    assert isinstance(tools, list) and all(isinstance(t, str) and TOOL.fullmatch(t) for t in tools), tools
    assert {"Bash(runfleet agent *)", "Bash(runfleet wait *)", "Bash(runfleet status *)"} <= set(tools)
    assert "Bash" not in tools and "Bash(*)" not in tools


def test_description_fits_with_a_worst_case_host_prefix() -> None:
    desc = _front()[0]["description"]
    assert isinstance(desc, str) and "\n" not in desc
    assert "READ-ONLY" in desc and "Fleet-wide" in desc and "Never launches, cancels or edits anything" in desc
    prefix = " ".join(f"host{i:02d}" for i in range(40))[:HOST_PREFIX_MAX].rstrip()
    assert len(prefix) >= HOST_PREFIX_MAX - 1
    full = f"{prefix} {desc}"
    assert len(full) <= 1024 and "<" not in full and ">" not in full
    _plain(full)


def test_rendered_skill_with_host_names_still_parses() -> None:
    rendered = skill.render(["k2", "vega", "lyra", "lynx", "labstation"])
    head, body = _split(rendered.text)
    top = strict_yaml(head)
    assert str(top["description"]).startswith("k2 vega lyra lynx labstation runfleet ")
    assert len(str(top["description"])) <= 1024 and body == _front()[1]


# ---------------------------------------------------------------- body: agent-neutral
CLAUDE_ONLY = ("run_in_background", "BashOutput", "Monitor tool", "/permissions", "settings.json", "Task tool",
               "subagent", "CLAUDE.md", "Claude's ")


def _paragraphs(text: str) -> list[str]:
    """Paragraphs and list items (a bullet ends at the next bullet or blank line)."""
    return [p for p in re.split(r"\n\s*\n|\n(?=- )", text) if p.strip()]


@pytest.mark.parametrize("rel", ["SKILL.md", "references/reading-output.md", "references/with-run-skills.md"])
def test_no_claude_only_instruction_without_a_fallback(rel: str) -> None:
    text = _read(rel)
    for phrase in CLAUDE_ONLY:
        assert phrase not in text, f"{rel}: Claude-only {phrase!r}"
    for para in _paragraphs(text):
        if "! runfleet" in para:
            assert "own terminal" in para, f"{rel}: `! runfleet` without the plain-terminal way: {para!r}"
        if "background" in para:
            assert re.search(r"(?i)if your agent supports background|otherwise", para), f"{rel}: {para!r}"


def test_waiting_has_a_foreground_fallback() -> None:
    body = _front()[1]
    assert "if your agent supports background commands" in body
    assert "--timeout 540" in body and "runfleet agent events --since" in body
    runs = _read("references/with-run-skills.md")
    assert "if your agent supports background commands" in runs and "--timeout 540" in runs


def test_budget_advice_names_the_real_cost_fields() -> None:
    """Review: the reference said `share` while `agent cost --json` emits `account.share_pct`; and it had no rule for
    an estimate above what is left."""
    runs = _read("references/with-run-skills.md")
    assert "`share`" not in runs and "`share_pct`" in runs and "`account.exceeds: true`" in runs
    assert "`candidate_accounts[]`" in runs and "`su_min` … `su_max`" in runs
    from runfleet.agent import view
    from tests.agent.test_usage_cost import creq, fleet, lyra, summary
    data = view.cost(fleet(lyra(account=summary())), "lyra", creq(account="pilab_02")).data
    assert {"share_pct", "exceeds", "available_su"} <= set(data["account"]) and "candidate_accounts" in data
    assert {"su_min", "su_max"} <= set(data)


def test_connect_is_left_to_the_user_in_their_own_terminal() -> None:
    body = _front()[1]
    assert "! runfleet" not in body
    assert "`runfleet connect HOST` in their own terminal (it may prompt for Duo/password)" in body
    assert "ask the user to run it in their own terminal" in body


def test_safety_rails_forbid_state_changing_commands() -> None:
    rails = _front()[1].split("## Safety rails (always)", 1)[1]
    for cmd in ("`runfleet connect`", "`doctor --reset`", "`update`", "`hosts import --write`", "`skill install`",
                "`skill update`", "`skill uninstall`"):
        assert cmd in rails, cmd
    assert "Never run scancel/kill yourself" in rails


@pytest.mark.parametrize("rel", ["SKILL.md", "references/reading-output.md", "references/with-run-skills.md"])
def test_code_blocks_only_run_read_only_commands(rel: str) -> None:
    text = _read(rel)
    blocks = re.findall(r"```(?:sh|bash)\n(.*?)```", text, flags=re.S)
    if rel == "SKILL.md":                                     # the skill: sh blocks only (test_agent_cli parses them)
        assert blocks and text.count("```") == 2 * len(blocks)
    for line in (ln.split("  #", 1)[0].strip() for b in blocks for ln in b.splitlines()):
        if line.startswith("runfleet "):
            assert re.match(r"runfleet (agent|wait|status) ", line), f"{rel}: {line!r}"
            assert not re.search(r"\b(skill|connect|update|snapshot|doctor)\b|--write|--yes", line), line


def test_not_found_fallback_first_looks_for_the_binary_then_names_the_removal_steps() -> None:
    body = _front()[1]
    para = next(p for p in _paragraphs(body) if "not found" in p)
    assert para.lstrip().startswith("- `runfleet` not found")
    # first: RunFleet may be installed but not on this agent's PATH
    path_hint = para.index("not be on this agent's PATH")
    assert "`~/.local/bin/runfleet`" in para and "ask the user where" in para and "`uv tool dir --bin`" in para
    # only then: removal, POSIX and Windows
    removal = para.index("Only if RunFleet is really gone")
    assert path_hint < removal < para.index("the user can remove this skill")
    # quoted: a $CLAUDE_CONFIG_DIR with spaces or glob characters must not turn into extra rm targets
    assert '`rm -r -- ~/.agents/skills/runfleet "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/runfleet"`' in para
    assert "$env:CLAUDE_CONFIG_DIR" in para
    assert '`Remove-Item -Recurse "$HOME\\.agents\\skills\\runfleet", "$HOME\\.claude\\skills\\runfleet"`' in para
    assert para.index("`~/.local/bin/runfleet`") < para.index("`rm -r")
    for line in para.splitlines():                       # every command on one line, as an agent copies it
        assert line.count("`") % 2 == 0, line


def test_skill_md_stays_short() -> None:
    text = _read("SKILL.md")
    assert len(text.splitlines()) < 80
    for section in ("## 1. Start here", "## 2. Commands", "## 3. Reading the answers", "## 4. With hpc-run / ws-run",
                    "## Safety rails (always)"):
        assert section in text


def test_references_say_the_skill_is_shared_and_kept_current() -> None:
    reading = _read("references/reading-output.md")
    about = reading.split("## About this skill", 1)[1]
    assert "`~/.agents/skills/runfleet`" in about and "Claude Code reaches it through a link" in about
    assert "RunFleet refreshes it after every upgrade" in about and "Never install, update or remove it" in about
    runs = _read("references/with-run-skills.md")
    assert "one copy shared by every" in runs and "kept current by RunFleet" in runs


def test_every_reference_link_resolves() -> None:
    seen: set[str] = set()
    for rel in FILES - {"agents/openai.yaml"}:
        text = _read(rel)
        for target in re.findall(r"\b(references/[\w.-]+\.md)\b", text):
            assert (SKILL_DIR / target).is_file(), f"{rel}: {target}"
            seen.add(target)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if not re.match(r"[a-z]+://|#", target):
                assert (SKILL_DIR / Path(rel).parent / target).is_file(), f"{rel}: {target}"
    assert seen == {"references/reading-output.md", "references/with-run-skills.md"}


# ---------------------------------------------------------------- agents/openai.yaml (Codex UI)
def test_openai_yaml_is_strict_and_shaped_for_codex() -> None:
    text = _read("agents/openai.yaml")
    parsed = strict_yaml(text)
    _yaml_agrees(text, parsed)
    ui = parsed["interface"]
    assert list(parsed) == ["interface"] and isinstance(ui, dict)
    assert list(ui) == ["display_name", "short_description", "default_prompt"]
    assert ui["display_name"] == "RunFleet"
    assert 25 <= len(ui["short_description"]) <= 64
    assert "$runfleet" in ui["default_prompt"] and "<" not in text and ">" not in text
