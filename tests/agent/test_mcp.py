"""`runfleet mcp` (runfleet.agent.mcp, docs/AGENT_INTERFACE.md §7.6): newline-delimited JSON-RPC over stdio.

The protocol is tested on `Server` / `serve` with fake runners and BytesIO streams; the end-to-end tests run the real
dispatcher on `--demo` in-process and once in a subprocess, with every RunFleet path under tmp_path (no ~/.config,
~/.local/state or real mux directory; no ssh).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from runfleet import cli, paths
from runfleet.agent import commands, mcp
from runfleet.agent.data import Document
from runfleet.errors import ConfigError

DOC = Document("runfleet.agent.brief/1", {"schema": "runfleet.agent.brief/1", "hosts": []}, "# brief", 0)


def _doc(exit_code: int) -> Document:
    return Document(DOC.schema, DOC.data, DOC.text, exit_code)


class Recorder:
    """A runner that records the calls and answers `doc` (or raises `exc`)."""

    def __init__(self, doc: Document = DOC, exc: BaseException | None = None) -> None:
        self.doc, self.exc = doc, exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, verb: str, params: Mapping[str, Any]) -> Document:
        self.calls.append((verb, dict(params)))
        if self.exc is not None:
            raise self.exc
        return self.doc


def req(method: str, params: Any = None, msg_id: Any = 1) -> dict[str, Any]:
    m: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        m["params"] = params
    return m


def call(name: str, arguments: Any = None, msg_id: Any = 1) -> dict[str, Any]:
    return req("tools/call", {"name": name, **({"arguments": arguments} if arguments is not None else {})}, msg_id)


def server(runner: mcp.Runner | None = None, version: str = "2025-06-18") -> mcp.Server:
    s = mcp.Server(runner or Recorder())
    s.handle(req("initialize", {"protocolVersion": version, "capabilities": {}}))
    return s


def result(reply: dict[str, Any] | None) -> dict[str, Any]:
    assert reply is not None and "error" not in reply, reply
    r: dict[str, Any] = reply["result"]
    return r


def error_code(reply: dict[str, Any] | None) -> int:
    assert reply is not None and "result" not in reply, reply
    code: int = reply["error"]["code"]
    return code


def lines(*messages: Any) -> bytes:
    return b"".join((m if isinstance(m, bytes) else json.dumps(m).encode()) + b"\n" for m in messages)


def serve_bytes(data: bytes, runner: mcp.Runner) -> list[Any]:
    out = io.BytesIO()
    assert mcp.serve(io.BytesIO(data), out, runner) == 0
    raw = out.getvalue()
    assert raw == b"" or raw.endswith(b"\n")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


# ====================================================================================================== lifecycle
@pytest.mark.parametrize(("asked", "agreed"), [
    ("2025-06-18", "2025-06-18"), ("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"),
    ("2099-01-01", "2025-06-18"),
])
def test_initialize_negotiates_the_protocol_version(asked: Any, agreed: str) -> None:
    params = {"capabilities": {}, "clientInfo": {"name": "orb", "version": "1"}}
    if asked is not None:
        params["protocolVersion"] = asked
    r = result(mcp.Server(Recorder()).handle(req("initialize", params)))
    assert r["protocolVersion"] == agreed
    assert r["capabilities"] == {"tools": {"listChanged": False}}
    assert r["serverInfo"]["name"] == "runfleet" and r["serverInfo"]["version"]
    assert "fleet_brief" in r["instructions"] and "read-only" in r["instructions"]
    assert "runfleet wait" in r["instructions"]


@pytest.mark.parametrize("params", [{}, {"protocolVersion": 7}, {"protocolVersion": "2025-06-18", "capabilities": []},
                                    {"protocolVersion": "2025-06-18", "clientInfo": "orb"}])
def test_initialize_with_bad_params(params: dict[str, Any]) -> None:
    s = mcp.Server(Recorder())
    assert error_code(s.handle(req("initialize", params))) == mcp.INVALID_PARAMS
    assert s.version == mcp.PROTOCOL_VERSIONS[0]


def test_notifications_get_no_reply_and_ping_answers_empty() -> None:
    s = server()
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}) is None
    assert s.handle({"jsonrpc": "2.0", "method": "no/such/notification"}) is None
    assert s.handle({"jsonrpc": "2.0", "method": "tools/call", "params": "bad"}) is None
    assert s.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None          # a client response: ignored
    assert s.handle(req("ping", msg_id="p-1")) == {"jsonrpc": "2.0", "id": "p-1", "result": {}}


@pytest.mark.parametrize(("message", "code"), [
    (req("resources/list"), mcp.METHOD_NOT_FOUND),
    (req("nope"), mcp.METHOD_NOT_FOUND),
    (req("ping", params=[1]), mcp.INVALID_PARAMS),
    ({"id": 1, "method": "ping"}, mcp.INVALID_REQUEST),                       # no jsonrpc
    ({"jsonrpc": "2.0", "id": 1}, mcp.INVALID_REQUEST),                       # no method
    ({"jsonrpc": "2.0", "id": True, "method": "ping"}, mcp.INVALID_REQUEST),
    ({"jsonrpc": "2.0", "id": [1], "method": "ping"}, mcp.INVALID_REQUEST),
    (3, mcp.INVALID_REQUEST),
    ({}, mcp.INVALID_REQUEST),                                                # no id, but no notification either
    ({"jsonrpc": "2.0", "method": 1}, mcp.INVALID_REQUEST),
    ({"jsonrpc": "1.0", "method": "ping"}, mcp.INVALID_REQUEST),
])
def test_bad_requests_are_json_rpc_errors(message: Any, code: int) -> None:
    reply = server().handle(message)
    assert error_code(reply) == code
    assert reply is not None and reply["jsonrpc"] == "2.0" and reply["error"]["message"]


def test_parse_errors() -> None:
    s = server()
    for raw in (b"{not json", b"\xff\xfe{}", b'{"jsonrpc": "2.0", "id": 1, "method": "ping"',
                b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"x": NaN}}'):
        reply = mcp.handle_line(s, raw)
        assert isinstance(reply, dict) and reply["id"] is None and error_code(reply) == mcp.PARSE_ERROR


def test_batches_only_under_2025_03_26() -> None:
    s = server(version="2025-03-26")
    batch = mcp.handle_line(s, json.dumps([req("ping", msg_id=1), {"jsonrpc": "2.0", "method": "x"},
                                           req("ping", msg_id=2)]).encode())
    assert batch == [{"jsonrpc": "2.0", "id": 1, "result": {}}, {"jsonrpc": "2.0", "id": 2, "result": {}}]
    assert mcp.handle_line(s, b'[{"jsonrpc": "2.0", "method": "x"}]') is None
    empty = mcp.handle_line(s, b"[]")
    assert isinstance(empty, dict) and error_code(empty) == mcp.INVALID_REQUEST
    for version in ("2025-06-18", "2024-11-05"):
        reply = mcp.handle_line(server(version=version), json.dumps([req("ping")]).encode())
        assert isinstance(reply, dict) and reply["id"] is None and error_code(reply) == mcp.INVALID_REQUEST


def test_an_unexpected_failure_is_an_internal_error(monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    def boom(version: str) -> list[dict[str, Any]]:
        raise RuntimeError("lyra-secret")

    monkeypatch.setattr(mcp, "tool_list", boom)
    reply = server().handle(req("tools/list", msg_id=5))
    assert reply is not None and reply["id"] == 5 and error_code(reply) == mcp.INTERNAL_ERROR
    assert "lyra-secret" not in json.dumps(reply) and "lyra-secret" in caplog.text


# ====================================================================================================== tools/list
def _cli_verb_dests(verb: str) -> set[str]:
    def sub(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
        action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        found: argparse.ArgumentParser = action.choices[name]
        return found

    parser = sub(sub(cli.command_parser(), "agent"), verb)
    common = {"help", "config", "demo", "demo_seed", "readonly", "no_mouse", "no_cache", "debug", "json"}
    return {a.dest for a in parser._actions} - common


def test_tools_list_one_read_only_tool_per_verb() -> None:
    tools = result(server().handle(req("tools/list")))["tools"]
    assert [t["name"] for t in tools] == ["fleet_brief", "list_jobs", "job_detail", "read_log", "gpu_availability",
                                          "host_health", "fleet_events", "account_usage", "estimate_cost",
                                          "runfleet_help"]
    assert sorted(t.verb for t in mcp.TOOLS) == sorted(commands.VERBS)
    for t in tools:
        schema = t["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        assert all(p.get("description") or p.get("enum") for p in schema["properties"].values()), t["name"]
        assert t["annotations"] == {"title": t["title"], "readOnlyHint": True, "openWorldHint": True,
                                    "idempotentHint": True}
        assert "Read-only" in t["description"]
        json.dumps(t)                                                         # plain JSON


@pytest.mark.parametrize("tool", mcp.TOOLS, ids=lambda t: t.name)
def test_tool_arguments_mirror_the_cli_flags(tool: mcp.Tool) -> None:
    dests = _cli_verb_dests(tool.verb)
    assert set(tool.properties) <= dests
    # --limit only where it limits rows; help takes no fleet flags
    ignored = {"limit"} if tool.verb not in ("jobs", "events", "usage") else set()
    if tool.verb == "help":
        ignored |= {"refresh", "timeout", "limit"}
    assert dests - ignored <= set(tool.properties)
    cli_required = {a.dest for a in _cli_parser_required(tool.verb)}
    assert set(tool.required) == cli_required
    if not tool.required:                                    # the CLI agrees: the verb runs with no argument at all
        assert cli.parse_args(["agent", tool.verb]).agent_verb == tool.verb


def _cli_parser_required(verb: str) -> list[argparse.Action]:
    parser = cli.command_parser()
    agent = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices["agent"]
    vp = next(a for a in agent._actions if isinstance(a, argparse._SubParsersAction)).choices[verb]
    return [a for a in vp._actions if a.required and not _takes_none(a)]


def _takes_none(action: argparse.Action) -> bool:
    """A positional argparse accepts zero values of (nargs "?" or "*"): never required, whatever `.required` says
    (Python 3.11 marks a nargs="*" positional without a default required=True; 3.13 does not)."""
    return not action.option_strings and action.nargs in (argparse.OPTIONAL, argparse.ZERO_OR_MORE)


def test_older_protocols_get_no_title_or_annotations() -> None:
    old = result(server(version="2024-11-05").handle(req("tools/list")))["tools"]
    assert all(set(t) == {"name", "description", "inputSchema"} for t in old)
    mid = result(server(version="2025-03-26").handle(req("tools/list")))["tools"]
    assert all("annotations" in t and "title" not in t for t in mid)


# ====================================================================================================== tools/call
@pytest.mark.parametrize(("exit_code", "is_error"), [(0, False), (4, False), (1, True), (2, True)])
def test_call_result_and_is_error(exit_code: int, is_error: bool) -> None:
    runner = Recorder(_doc(exit_code))
    r = result(server(runner).handle(call("fleet_brief", {"hosts": ["k2"], "refresh": True})))
    assert r == {"content": [{"type": "text", "text": "# brief"}], "isError": is_error,
                 "structuredContent": dict(DOC.data)}
    assert runner.calls == [("brief", {"hosts": ["k2"], "refresh": True})]


def test_structured_content_only_from_2025_06_18() -> None:
    r = result(server(version="2025-03-26").handle(call("fleet_brief")))
    assert "structuredContent" not in r and r["content"][0]["text"] == "# brief"


def test_arguments_reach_the_dispatcher_under_its_names() -> None:
    runner = Recorder()
    s = server(runner)
    result(s.handle(call("read_log", {"id": "k2:2332861", "lines": 80.0, "stream": "err", "grep": "nan|oom",
                                      "head": False, "timeout": 5, "path_only": None})))
    result(s.handle(call("job_detail", {"ids": ["k2:1", "vega:2"]})))
    result(s.handle(call("runfleet_help", {"topic": "log"})))
    result(s.handle(call("gpu_availability", {"type": "h100", "min": 0})))
    result(s.handle(call("read_log", {"id": "k2:7", "list_files": True})))
    result(s.handle(call("read_log", {"id": "k2:7", "file": "runs/a/train.log", "lines": 5, "list_files": None})))
    assert runner.calls == [
        ("log", {"id": "k2:2332861", "lines": 80, "stream": "err", "grep": "nan|oom", "head": False, "timeout": 5}),
        ("job", {"ids": ["k2:1", "vega:2"]}), ("help", {"topic": "log"}), ("gpus", {"type": "h100", "min": 0}),
        ("log", {"id": "k2:7", "list_files": True}), ("log", {"id": "k2:7", "file": "runs/a/train.log", "lines": 5})]
    assert isinstance(runner.calls[0][1]["lines"], int)


@pytest.mark.parametrize(("name", "arguments"), [
    ("no_such_tool", {}),
    ("list_jobs", {"lines": 3}),                      # additionalProperties: false
    ("list_jobs", {"state": 3}),
    ("list_jobs", {"attention": "yes"}),
    ("list_jobs", {"limit": 0}),
    ("list_jobs", {"limit": 2.5}),
    ("list_jobs", {"hosts": "k2"}),
    ("list_jobs", {"hosts": [1]}),
    ("read_log", {}),                                 # id required
    ("read_log", {"id": "k2:1", "lines": 401}),
    ("read_log", {"id": "k2:1", "lines": 10 ** 400}),
    ("read_log", {"id": "k2:1", "stream": "both"}),
    ("read_log", {"id": "k2:1", "grep": "x" * 401}),
    ("read_log", {"id": "k2:1", "timeout": 0}),
    ("read_log", {"id": "k2:1", "timeout": True}),
    ("read_log", {"id": "k2:1", "timeout": mcp.TIMEOUT_MAX + 1}),
    ("read_log", {"id": "k2:1", "list_files": "yes"}),
    ("read_log", {"id": "k2:1", "file": 3}),
    ("read_log", {"id": "k2:1", "file": "x" * 4097}),
    ("read_log", {"id": "k2:1", "files": True}),                      # the CLI spelling is not an argument
    ("job_detail", {"ids": []}),
    ("job_detail", {"ids": [f"k2:{i}" for i in range(11)]}),
    ("fleet_events", {}),
    ("runfleet_help", {"topic": "help"}),
    ("fleet_brief", [1]),                             # arguments not an object
])
def test_schema_violations_are_invalid_params(name: str, arguments: Any) -> None:
    runner = Recorder()
    reply = server(runner).handle(call(name, arguments))
    assert error_code(reply) == mcp.INVALID_PARAMS
    assert runner.calls == []


@pytest.mark.parametrize(("arguments", "text"), [
    ({"list_files": True, "file": "a.log"}, "list_files does not combine with file"),
    ({"list_files": True, "grep": "x", "head": True}, "list_files does not combine with grep head"),
    ({"list_files": True, "path_only": True}, "list_files does not combine with path_only"),
    ({"file": "a.log", "stream": "err"}, "file does not combine with stream"),
    ({"file": "a.log", "path_only": True}, "file does not combine with path_only"),
])
def test_read_log_usage_errors_name_the_tool_parameters(arguments: dict[str, Any], text: str) -> None:
    """Round 2: `list_files` with `file` said "--files does not combine with --file" (the CLI's flags); the CLI
    keeps its flag names."""
    runner = mcp.command_runner(config_path=None, demo=True, demo_seed=commands.DEMO_SEED, clock=None)
    r = result(server(runner).handle(call("read_log", {"id": "k2:3372131", **arguments})))
    assert r["isError"] is True and r["content"][0]["text"] == f"usage error: {text}"
    with pytest.raises(commands.UsageError) as exc:
        runner("log", {"id": "k2:3372131", **arguments})
    assert str(exc.value).startswith("--files does not combine with --" if "list_files" in arguments else
                                     "--file does not combine with --")


def test_dispatcher_errors_are_tool_errors_without_details(caplog: pytest.LogCaptureFixture) -> None:
    s = server(Recorder(exc=commands.UsageError("unknown host 'zzz' (hosts: k2)")))
    r = result(s.handle(call("host_health", {"host": "zzz"})))
    assert r["isError"] is True and r["content"][0]["text"] == "usage error: unknown host 'zzz' (hosts: k2)"
    r = result(server(Recorder(exc=ConfigError("host[0].ssh", "'jump.alice@vega' is odd"))).handle(call("fleet_brief")))
    assert r["isError"] is True and "host[0].ssh" in r["content"][0]["text"]
    assert "jump.alice@vega" not in r["content"][0]["text"]
    with caplog.at_level(logging.ERROR, logger="runfleet.mcp"):
        r = result(server(Recorder(exc=RuntimeError("jump.alice-secret.example"))).handle(call("fleet_brief")))
    assert r["isError"] is True and "structuredContent" not in r
    assert "jump.alice-secret.example" not in json.dumps(r) and "runfleet.log" in r["content"][0]["text"]
    assert "jump.alice-secret.example" in caplog.text


HF_TOKEN = "hf_" + "Q" * 30                    # fictional, shaped like a Hugging Face token


def test_echoed_error_texts_are_scrubbed_escaped_and_capped(caplog: pytest.LogCaptureFixture) -> None:
    """F6: a UsageError (or a JSON-RPC error) quotes the caller's input as given: a pasted secret, terminal escapes
    or a newline that forges a log line, megabytes of it. What reaches the client and runfleet.log is scrubbed,
    escaped and at most commands.ERROR_TEXT_MAX bytes."""
    evil = f"zzz {HF_TOKEN}\x1b]52;c;cHduZWQ=\x07\n2026-10-18 INFO forged line\u202e" + "A" * 5000
    with caplog.at_level(logging.INFO, logger="runfleet.mcp"):
        r = result(server(Recorder(exc=commands.UsageError(f"unknown host {evil!r} (hosts: k2)"))).handle(
            call("host_health", {"host": "zzz"})))
    text = r["content"][0]["text"]
    assert r["isError"] is True and text.startswith("usage error: unknown host 'zzz hf_<redacted>")
    assert HF_TOKEN not in text and "\x1b" not in text and "\n" not in text and "\u202e" not in text
    assert len(text.encode()) <= len("usage error: ") + commands.ERROR_TEXT_MAX and text.endswith("…")
    logged = [rec.getMessage() for rec in caplog.records if "usage error" in rec.getMessage()]
    assert logged and all(HF_TOKEN not in m and "\n" not in m and "\x1b" not in m for m in logged)
    assert all(len(m.encode()) < commands.ERROR_TEXT_MAX + 100 for m in logged)
    # JSON-RPC errors that echo a method or tool name
    reply = server().handle(req(f"x/{HF_TOKEN}\n" + "B" * 5000))
    assert reply is not None and error_code(reply) == mcp.METHOD_NOT_FOUND
    message = reply["error"]["message"]
    assert HF_TOKEN not in message and "\n" not in message and len(message.encode()) <= commands.ERROR_TEXT_MAX
    reply = server().handle(req("tools/call", {"name": f"{HF_TOKEN}\x1b[2J", "arguments": {}}))
    assert reply is not None and error_code(reply) == mcp.INVALID_PARAMS
    assert HF_TOKEN not in reply["error"]["message"] and "\x1b" not in reply["error"]["message"]


def test_a_lone_surrogate_in_an_echoed_name_never_breaks_the_server() -> None:
    """Codex: `"method": "\\ud800"` is valid JSON; echoing it made encoding raise outside the handler."""
    message = json.loads('{"jsonrpc": "2.0", "id": 1, "method": "\\ud800"}')
    reply = server().handle(message)
    assert reply is not None and error_code(reply) == mcp.METHOD_NOT_FOUND
    assert "\\ud800" in reply["error"]["message"]
    mcp.encode(reply)                                                          # encodable
    assert serve_bytes(b'{"jsonrpc": "2.0", "id": 2, "method": "\\ud800"}\n', Recorder())[0]["id"] == 2


def test_safe_text_cuts_on_a_character_boundary() -> None:
    assert commands.safe_text("plain") == "plain"
    cut = commands.safe_text("é" * 400)
    assert cut.endswith("…") and len(cut.encode()) <= commands.ERROR_TEXT_MAX and cut.count("é") == 148
    assert commands.safe_text("a\tb") == "a\tb"                                  # TAB stays


def test_a_call_past_its_cap_is_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "CAP_MARGIN", 0.0)
    monkeypatch.setattr(commands, "GRACE", 0.0)
    release = threading.Event()

    def stuck(verb: str, params: Mapping[str, Any]) -> Document:
        release.wait(10)
        return DOC

    try:
        s = server(stuck)
        r = result(s.handle(call("fleet_brief", {"timeout": 0.2})))
        assert r["isError"] is True and "no answer within" in r["content"][0]["text"]
        assert s.handle(req("ping")) == {"jsonrpc": "2.0", "id": 1, "result": {}}      # the loop goes on
        r = result(s.handle(call("fleet_brief", {"timeout": 0.2})))                     # but no second worker
        assert r["isError"] is True and "still finishing" in r["content"][0]["text"]
    finally:
        release.set()
    for t in s.leftover:
        t.join(10)
    r = result(s.handle(call("fleet_brief", {"timeout": 0.2})))
    assert r["isError"] is False and s.leftover == []


def test_a_dispatcher_thread_left_behind_also_blocks_the_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "SETTLE", 0.05)
    release = threading.Event()

    def leaves_a_cache_read(verb: str, params: Mapping[str, Any]) -> Document:
        threading.Thread(target=release.wait, args=(10,), name=commands.CACHE_THREAD, daemon=True).start()
        return DOC

    try:
        s = server(leaves_a_cache_read)
        assert result(s.handle(call("fleet_brief")))["isError"] is False
        r = result(s.handle(call("fleet_brief")))
        assert r["isError"] is True and "still finishing" in r["content"][0]["text"]
    finally:
        release.set()
    for t in s.leftover:
        t.join(10)
    assert result(s.handle(call("fleet_brief")))["isError"] is False


def test_unencodable_replies_become_internal_errors(caplog: pytest.LogCaptureFixture) -> None:
    bad = Document(DOC.schema, {"schema": DOC.schema, "at": object()}, "# brief", 0)
    replies = serve_bytes(lines(call("fleet_brief", msg_id=1), req("ping", msg_id=2)), Recorder(bad))
    assert replies[0]["error"]["code"] == mcp.INTERNAL_ERROR and replies[1]["result"] == {}
    raw = mcp._encode_reply([{"jsonrpc": "2.0", "id": 1, "result": {"x": math.inf}},
                             {"jsonrpc": "2.0", "id": 2, "result": {}}])
    assert raw.count(b"\n") == 1
    assert json.loads(raw) == [{"jsonrpc": "2.0", "id": 1, "error": {"code": mcp.INTERNAL_ERROR,
                                                                    "message": "internal error (details in "
                                                                               "runfleet.log)"}},
                               {"jsonrpc": "2.0", "id": 2, "result": {}}]


# ====================================================================================================== the loop
def test_serve_answers_requests_in_order_and_skips_blank_lines() -> None:
    data = lines(req("initialize", {"protocolVersion": "2024-11-05"}, 0),
                 {"jsonrpc": "2.0", "method": "notifications/initialized"}, b"", b"   \r",
                 req("ping", msg_id=1), b"oops", call("fleet_brief", msg_id=2)) + b'{"jsonrpc":"2.0","id":3,' \
        b'"method":"ping"}'                                                    # last line without a newline
    replies = serve_bytes(data, Recorder())
    assert [r["id"] for r in replies] == [0, 1, None, 2, 3]
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert replies[2]["error"]["code"] == mcp.PARSE_ERROR
    assert "structuredContent" not in replies[3]["result"]


def test_serve_crlf_and_oversized_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "MAX_LINE", 64)
    data = json.dumps(req("ping", msg_id=1)).encode() + b"\r\n" + b"[" * 300 + b"\n" + lines(req("ping", msg_id=2))
    replies = serve_bytes(data, Recorder())
    assert [r["id"] for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == mcp.PARSE_ERROR


def test_encode_is_one_utf8_line() -> None:
    raw = mcp.encode({"text": "a\nb · → ", "bad": "\udc80"})
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert json.loads(raw) == {"text": "a\nb · → ", "bad": "\udc80"}
    good = mcp.encode({"text": "· →"})
    assert "· →".encode() in good
    with pytest.raises(ValueError):
        mcp.encode({"x": math.nan})


def test_stray_prints_go_to_the_log(caplog: pytest.LogCaptureFixture) -> None:
    stray = mcp.LogWriter()
    with caplog.at_level(logging.WARNING, logger="runfleet.mcp"), contextlib.redirect_stdout(stray):
        print("hello from orb")
        print("partial", end="")
        stray.flush()
    assert "stray output: hello from orb" in caplog.text and "stray output: partial" in caplog.text


# ====================================================================================================== end to end
def _drop_log_handlers() -> None:
    root = logging.getLogger("runfleet")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def tmp_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    _drop_log_handlers()
    monkeypatch.setattr(paths, "config_file", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(paths, "log_file", lambda: tmp_path / "state" / "runfleet.log")
    monkeypatch.setattr(paths, "cache_file", lambda: tmp_path / "state" / "cache.sqlite")
    monkeypatch.setattr(paths, "mux_dir", lambda: tmp_path / "mux")
    monkeypatch.setattr(paths, "debug_dir", lambda: tmp_path / "state" / "debug")
    yield tmp_path
    _drop_log_handlers()


def run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str], data: bytes) -> tuple[int, bytes, str]:
    stdin = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8")
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    rc = cli.main(argv)
    stdout.flush()
    assert isinstance(stdout.buffer, io.BytesIO)
    return rc, stdout.buffer.getvalue(), stderr.getvalue()


def _session(job_id: str | None = None) -> bytes:
    msgs: list[Any] = [
        req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "orb", "version": "1"}}, 1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        req("tools/list", msg_id=2),
        call("fleet_brief", msg_id=3),
        call("list_jobs", {"state": "failed", "limit": 400}, msg_id=4),
        call("job_detail", {"ids": ["lynx:1"]}, msg_id=5),
        req("ping", msg_id=6),
        req("nope", msg_id=7),
        b"{malformed",
    ]
    if job_id is not None:
        msgs.append(call("read_log", {"id": job_id, "lines": 20}, msg_id=8))
    return lines(*msgs)


def _check_session(replies: list[Any]) -> dict[Any, Any]:
    by_id = {r["id"]: r for r in replies}
    assert len(by_id) == len(replies)
    assert all(r["jsonrpc"] == "2.0" for r in replies)
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    assert len(by_id[2]["result"]["tools"]) == len(mcp.TOOLS)
    brief = by_id[3]["result"]
    assert brief["isError"] is False                                          # demo exit 4 (lynx login): partial
    assert brief["structuredContent"]["schema"] == "runfleet.agent.brief/1"
    assert brief["content"][0]["type"] == "text" and brief["content"][0]["text"].startswith("# runfleet")
    assert by_id[4]["result"]["structuredContent"]["schema"] == "runfleet.agent.jobs/1"
    assert by_id[5]["result"]["isError"] is True                              # no data for lynx (exit 1)
    assert by_id[6]["result"] == {}
    assert by_id[7]["error"]["code"] == mcp.METHOD_NOT_FOUND
    assert by_id[None]["error"]["code"] == mcp.PARSE_ERROR
    return by_id


def _failed_job(jobs_reply: dict[str, Any]) -> str:
    jobs = jobs_reply["result"]["structuredContent"]["jobs"]
    pick: str = next(j["id"] for j in jobs if j["mine"] and j["logs"]["stdout"] and j["kind"] == "slurm")
    return pick


def test_mcp_subcommand_on_the_demo_fleet(monkeypatch: pytest.MonkeyPatch, tmp_paths: Path) -> None:
    rc, out, err = run_main(monkeypatch, ["--demo", "mcp"], _session())
    assert rc == 0 and err == ""
    by_id = _check_session([json.loads(line) for line in out.decode("utf-8").splitlines()])
    job_id = _failed_job(by_id[4])

    rc, out, err = run_main(monkeypatch, ["mcp", "--demo"], lines(call("read_log", {"id": job_id}, 1)))
    assert rc == 0 and err == ""
    log = json.loads(out)["result"]
    assert log["isError"] is False and log["structuredContent"]["schema"] == "runfleet.agent.log/1"
    assert (tmp_paths / "state" / "runfleet.log").exists()
    assert not (tmp_paths / "state" / "cache.sqlite").exists()


def test_mcp_without_a_config_answers_no_data(monkeypatch: pytest.MonkeyPatch, tmp_paths: Path) -> None:
    rc, out, err = run_main(monkeypatch, ["mcp"], lines(call("fleet_brief", msg_id=1)))
    assert rc == 0 and err == ""
    r = json.loads(out)["result"]
    assert r["isError"] is True and not (tmp_paths / "mux").exists()


@pytest.mark.parametrize("argv", [["mcp", "--no-cache"], ["-c", "nope.toml", "mcp"]])
def test_mcp_startup_usage_errors(monkeypatch: pytest.MonkeyPatch, tmp_paths: Path, argv: list[str]) -> None:
    monkeypatch.chdir(tmp_paths)
    rc, out, err = run_main(monkeypatch, argv, lines(req("ping")))
    assert rc == 2 and out == b"" and err.startswith("runfleet: ")


def test_mcp_takes_no_agent_flags(monkeypatch: pytest.MonkeyPatch, tmp_paths: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, ["mcp", "--json"], b"")
    assert exc.value.code == 2


_CHILD = """\
import sys
from pathlib import Path
import runfleet.paths as p
base = Path(sys.argv[1])
p.log_file = lambda: base / 'runfleet.log'
p.cache_file = lambda: base / 'cache.sqlite'
p.config_file = lambda: base / 'config.toml'
p.mux_dir = lambda: base / 'mux'
from runfleet import cli, paths
code = cli.main(['--demo', 'mcp'])
bad = sorted(m for m in sys.modules if m.split('.')[0] == 'textual' or m.startswith('runfleet.ui'))
sys.stderr.write(f'BAD {bad}')
raise SystemExit(code)
"""


def test_mcp_in_a_subprocess(tmp_path: Path) -> None:
    """A real pipe: each reply arrives (flushed) before stdin closes, so read_log can use an id from list_jobs."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    assert proc.stdin is not None and proc.stdout is not None
    got: queue.Queue[bytes] = queue.Queue()
    stdout = proc.stdout

    def pump() -> None:                      # blocking reads off the test thread: every wait below has a timeout
        for line in iter(stdout.readline, b""):
            got.put(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def until(msg_id: int) -> list[bytes]:
        seen = []
        while True:
            raw = got.get(timeout=60)
            seen.append(raw)
            if json.loads(raw).get("id") == msg_id:
                return seen

    try:
        proc.stdin.write(_session())
        proc.stdin.flush()
        raw = until(7) + [got.get(timeout=60)]                                # 7: the last id; then the parse error
        replies = [json.loads(line) for line in raw]
        by_id = _check_session(replies)
        proc.stdin.write(lines(call("read_log", {"id": _failed_job(by_id[4]), "lines": 20}, msg_id=8)))
        proc.stdin.flush()
        raw += until(8)
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0
        reader.join(30)                                                       # stdout read to its end
        assert not reader.is_alive()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        proc.stdout.close()
    assert all(line.endswith(b"\n") and not line.endswith(b"\r\n") for line in raw)
    log = json.loads(raw[-1])["result"]
    assert log["isError"] is False and log["structuredContent"]["schema"] == "runfleet.agent.log/1"
    assert got.empty()                                                        # nothing else on stdout
    assert proc.stderr is not None
    assert proc.stderr.read().decode("utf-8", "replace").strip() == "BAD []"
    assert (tmp_path / "runfleet.log").exists()


def test_mcp_commands_carry_the_global_options(monkeypatch: pytest.MonkeyPatch, tmp_paths: Path) -> None:
    """C5: every printed command repeats --demo-seed N (not the default) — runnable as printed."""
    msgs = lines(call("list_jobs", {"state": "all", "limit": 2}, 1))
    rc, out, err = run_main(monkeypatch, ["--demo", "--demo-seed", "11", "mcp"], msgs)
    assert rc == 0 and err == ""
    doc = json.loads(out)["result"]["structuredContent"]
    assert doc["truncated"]["next"].startswith("runfleet --demo --demo-seed 11 agent jobs ")
    rc, out, err = run_main(monkeypatch, ["--demo", "--demo-seed", "7", "mcp"], msgs)
    doc = json.loads(out)["result"]["structuredContent"]
    assert doc["truncated"]["next"].startswith("runfleet --demo agent jobs ")      # the default: not repeated
