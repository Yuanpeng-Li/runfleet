"""[WP4] Text that leaves the machine or reaches a terminal from a remote one: secrets scrubbed (support bundle,
`runfleet agent`), ANSI escape sequences stripped and control characters made visible (untrusted remote strings).

Pure: stdlib only (re), no UI imports, so `agent/` can use it as well as `ui/everyday.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# names whose value is a credential: …password, …secret, …token, …api_key / api-key, …access_key / access-key
_CRED = r"(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key)"
# a value: a whole quoted string (to its closing quote, else to the end of the line: a cut line) or one word
_QUOTED = r"(?P<q>[\"'])(?:\\.|(?!(?P=q))[^\n\\])*(?P=q)?"


def _requote(m: re.Match[str]) -> str:
    """``<redacted>`` inside the value's quotes as they were (an unterminated quote stays unterminated)."""
    q = m.group("q")
    if not q:
        return "<redacted>"
    value = m.string[m.start("q"):m.end()]
    return f"{q}<redacted>{q if len(value) > 1 and value.endswith(q) else ''}"


_SECRETS: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    (re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{40}(?![0-9A-Fa-f])"), "<redacted-40hex>"),     # wandb API keys
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "sk-<redacted>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "gh*_<redacted>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<redacted-aws-key-id>"),
    (re.compile(r"(?i)\b(aws_secret_access_key|aws_session_token)(\s*[=:]\s*)(?:" + _QUOTED + r"|\S+)"),
     lambda m: f"{m.group(1)}{m.group(2)}{_requote(m)}"),
    (re.compile(r"(?i)\b((?:access_)?token=)[^&\s#'\"]+"), r"\1<redacted>"),
    # PEM private keys (to the END line, else to the end of the code fence / text: a tail can cut the block)
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|(?=\n```)|\Z)",
                re.DOTALL), "<redacted-private-key>"),
    (re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/\-]{8,}=*"), r"\1 <redacted>"),
    (re.compile(r"(?<![A-Za-z0-9+.\-])([A-Za-z][A-Za-z0-9+.\-]*://)[^/\s:@'\"]+:[^/\s@'\"]+@"), r"\1<redacted>@"),   # user:pw@ in URLs
    # long command-line flags with a space-separated (or =) value: --password X, --api-key "a b", --token=X
    (re.compile(r"(?i)(?<![\w-])(--(?:[A-Za-z0-9]+[_-])*" + _CRED + r")([ \t]+|[ \t]*=[ \t]*)"
                r"(?:" + _QUOTED + r"|(?!--|<redacted)[^\s\"']+)"),
     lambda m: f"{m.group(1)}{m.group(2)}{_requote(m)}"),
    # NAME="a b c" / "name": "a b c" whose name ends in password, secret, token, api key …: the whole quoted value
    (re.compile(r"(?i)(?<![A-Za-z0-9_.\-])([A-Za-z0-9_.\-]*" + _CRED + r"[\"']?\s*[=:]\s*)" + _QUOTED),
     lambda m: f"{m.group(1)}{_requote(m)}"),
    # NAME=value / NAME: value (unquoted)
    (re.compile(r"(?i)(?<![A-Za-z0-9_.\-])([A-Za-z0-9_.\-]*" + _CRED
                + r"[\"']?\s*[=:]\s*[\"']?)(?!<redacted)[^\s\"',;&]+"),
     r"\1<redacted>"),
)


def scrub(text: str) -> str:
    """Secrets out of text that leaves the machine (support bundle): wandb keys (40 hex), ``hf_…``, ``sk-…``,
    ``ghp_…``, AWS key ids and secrets, ``token=`` query parameters, PEM private keys, ``Bearer`` / ``Basic``
    credentials, ``user:password@`` in URLs, long flags (``--password X``, ``--api-key "a b"``, ``--token=X``) and
    values of ``…PASSWORD`` / ``…SECRET`` / ``…TOKEN`` / ``…API_KEY`` assignments (a quoted value as a whole)."""
    for rx, repl in _SECRETS:
        text = rx.sub(repl, text)
    return text


_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_KEY_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")


def _key_open(text: str) -> bool:
    """A private-key block begins and is not closed before the end of the text."""
    begins = [m.end() for m in _KEY_BEGIN.finditer(text)]
    return bool(begins) and not _KEY_END.search(text, begins[-1])


def scrub_log(lines: Sequence[str], live: str | None = None) -> tuple[list[str], str | None]:
    """Complete log lines plus the unfinished live segment scrubbed as ONE text (a secret can span the boundary),
    then split again. A private-key block still open at the end swallows the live segment: it comes back as
    ``<redacted-private-key>``. A redacted key block inside ``lines`` becomes one line."""
    text = "\n".join(lines)
    body = scrub(text).split("\n") if lines else []
    if live is None:
        return body, None
    whole = f"{text}\n{live}" if lines else live
    out = scrub(whole)
    if out.count("\n") == whole.count("\n"):    # nothing joined lines: the last one is still the live segment
        parts = out.split("\n")
        return parts[:-1], parts[-1]
    if _key_open(whole):
        return body, "<redacted-private-key>"
    # a closed key block joined lines above: the live segment keeps the last complete line as its context
    tail = f"{lines[-1]}\n{live}" if lines else live
    out = scrub(tail)
    return body, out.split("\n", 1)[1] if lines and out.count("\n") == 1 else scrub(live)


# ESC / 8-bit CSI sequences: CSI (SGR colours, cursor moves; private and intermediate bytes), OSC (titles, OSC 8
# hyperlinks; BEL or ST terminated, or cut by the end of the text), DCS / SOS / PM / APC strings, charset selection
# (ESC ( B), then any other two-byte escape (ESC 7, ESC =, ESC c, …). Order matters: `[`, `]`, `P`, `X`, `^`, `_` are
# also two-byte finals.
_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"
    r"|\x1b[ -/]+[0-~]"
    r"|\x1b[0-~]"
)

# C0 except TAB, DEL, C1; plus the Unicode line separators and bidi overrides/isolates, which reorder or break what a
# reader (or a model) sees without being visible themselves ("Trojan Source").
_CONTROLS = re.compile("[\x00-\x08\x0a-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069]")


def _visible(m: re.Match[str]) -> str:
    c = ord(m.group())
    return f"\\x{c:02x}" if c <= 0xFF else f"\\u{c:04x}"


def strip_ansi(text: str) -> str:
    """Terminal escape sequences removed (colours, cursor moves, titles, hyperlinks); the text between them kept."""
    return _ANSI.sub("", text)


def escape_controls(text: str) -> str:
    """Control characters of an untrusted remote string made visible: ESC becomes the four characters ``\\x1b``,
    U+202E the six characters ``\\u202e``; TAB is kept, a newline is escaped too (split lines first). Strip ANSI
    first when the colours are noise rather than evidence."""
    return _CONTROLS.sub(_visible, text)
