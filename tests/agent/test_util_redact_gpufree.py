"""[WP4] util.redact (scrub moved from ui/everyday, ANSI stripping, control-character escapes) and util.gpufree (the
free-GPU rules moved from ui/render and ui/everyday): same behaviour, old names still re-exported."""

from __future__ import annotations

import dataclasses

from runfleet.models import GPU, GPUProcess
from runfleet.ui import everyday, render
from runfleet.util import gpufree, redact


def _gpu(index: int, util: float | None, mem: float | None, *procs: GPUProcess) -> GPU:
    return GPU(host="labstation", index=index, uuid=f"GPU-{index}", name="NVIDIA RTX A6000", util_pct=util,
               mem_used_mib=mem, mem_total_mib=49140.0, mem_util_pct=None, temp_c=None, power_w=None,
               power_limit_w=None, fan_pct=None, pstate=None, source="demo", processes=procs)


def _proc(user: str = "alice") -> GPUProcess:
    return GPUProcess(pid=642229, gpu_uuid="GPU-0", gpu_index=0, used_mem_mib=300.0, user=user, command="python",
                      etime_s=10, cpu_pct=None, job=None)


# ---------------------------------------------------------------- redact
def test_scrub_is_reexported_unchanged() -> None:
    assert everyday.scrub is redact.scrub
    text = "WANDB_API_KEY=q1w2e3r4t5 hf_abcdefghijklmnopqrstuvwxyz12 https://alice:pw@k2/x token=abc"
    out = redact.scrub(text)
    assert out == "WANDB_API_KEY=<redacted> hf_<redacted> https://<redacted>@k2/x token=<redacted>"


def test_scrub_space_separated_credential_flags() -> None:
    s = redact.scrub
    assert s("python t.py --password hunter2 --lr 3") == "python t.py --password <redacted> --lr 3"
    assert s("x --api-key AbC123 --api_key Zz9") == "x --api-key <redacted> --api_key <redacted>"
    assert s("x --token tk1 --secret sc1 --access-key ak1") == "x --token <redacted> --secret <redacted> " \
        "--access-key <redacted>"
    assert s("x --wandb-token=abc --my-api-key  q") == "x --wandb-token=<redacted> --my-api-key  <redacted>"
    assert s("x --password 'a b c' y") == "x --password '<redacted>' y"
    assert s("x --token --verbose") == "x --token --verbose"              # the next flag is not a value
    assert s("x --password-file /etc/pw -p 22") == "x --password-file /etc/pw -p 22"     # long flags ending in it
    assert s("x --tokenizer gpt2") == "x --tokenizer gpt2"


def test_scrub_quoted_values_whole() -> None:
    s = redact.scrub
    assert s('password="a b c" next') == 'password="<redacted>" next'
    assert s("DB_PASSWORD='x y' z") == "DB_PASSWORD='<redacted>' z"
    assert s('{"api_key": "a b", "lr": 3}') == '{"api_key": "<redacted>", "lr": 3}'
    assert s('secret="unterminated rest of line') == 'secret="<redacted>'
    assert s("password=plain other") == "password=<redacted> other"          # unquoted: one word, as before


def test_scrub_review_gaps() -> None:
    """Codex review: quoted AWS values, escaped quotes, a closed key block before `password:` + live value, and
    no quadratic scan on long untrusted lines."""
    import time
    s = redact.scrub
    assert s('aws_secret_access_key="first second" x') == 'aws_secret_access_key="<redacted>" x'
    assert s('password="first\\"second" y') == 'password="<redacted>" y'
    key = ["-----BEGIN RSA PRIVATE KEY-----", "x", "-----END RSA PRIVATE KEY-----", "password:"]
    assert redact.scrub_log(key, "hunter2") == (["<redacted-private-key>", "password:"], "<redacted>")
    t0 = time.perf_counter()
    for text in ("a." * 20000, "a-" * 20000, "--a-" * 10000, "a.a://" * 5000, '"' * 40000):
        s(text)
    assert time.perf_counter() - t0 < 1.0


def test_scrub_log_treats_lines_and_live_as_one_text() -> None:
    lines = ["ok", "-----BEGIN RSA PRIVATE KEY-----", "MIIEpAIBAAKCAQEA"]
    body, live = redact.scrub_log(lines, "more key material")
    assert body == ["ok", "<redacted-private-key>"] and live == "<redacted-private-key>"
    closed = ["-----BEGIN RSA PRIVATE KEY-----", "x", "-----END RSA PRIVATE KEY-----", "after"]
    assert redact.scrub_log(closed, "epoch 3") == (["<redacted-private-key>", "after"], "epoch 3")
    assert redact.scrub_log(["a", "password:"], "hunter2 epoch") == (["a", "password:"], "<redacted> epoch")
    assert redact.scrub_log([], "token=abc") == ([], "token=<redacted>")
    assert redact.scrub_log(["a", "b"]) == (["a", "b"], None) and redact.scrub_log([]) == ([], None)


def test_strip_ansi_removes_sgr_csi_osc_and_other_escapes() -> None:
    assert redact.strip_ansi("\x1b[1;31mError\x1b[0m: x") == "Error: x"
    assert redact.strip_ansi("\x1b[2K\x1b[1Gepoch 3") == "epoch 3"                         # erase line, column
    assert redact.strip_ansi("\x1b[?25lhidden\x1b[?25h") == "hidden"                        # private mode
    assert redact.strip_ansi("\x1b]0;title\x07after") == "after"                            # OSC, BEL
    assert redact.strip_ansi("\x1b]8;;https://k2/x\x1b\\link\x1b]8;;\x1b\\") == "link"      # OSC 8 hyperlink
    assert redact.strip_ansi("\x1b(Bplain\x1b=") == "plain"                                 # charset, keypad
    assert redact.strip_ansi("\x9b31mred") == "red"                                         # 8-bit CSI
    assert redact.strip_ansi("\x1bP1$r0m\x1b\\done") == "done"                              # DCS string
    assert redact.strip_ansi("no escapes\there") == "no escapes\there"


def test_escape_controls_makes_controls_visible_and_keeps_tabs() -> None:
    assert redact.escape_controls("a\tb") == "a\tb"
    assert redact.escape_controls("\x1b[31mred") == "\\x1b[31mred"
    assert redact.escape_controls("bell\x07 nul\x00 del\x7f c1\x85") == "bell\\x07 nul\\x00 del\\x7f c1\\x85"
    assert redact.escape_controls("x\ny\rz") == "x\\x0ay\\x0dz"
    assert redact.escape_controls("abc\u202edcba") == "abc\\u202edcba"                      # bidi override
    assert redact.escape_controls("naïve ✓ 日本") == "naïve ✓ 日本"                          # printable Unicode
    assert redact.escape_controls(redact.strip_ansi("\x1b[1mok\x1b[0m\x08")) == "ok\\x08"


# ---------------------------------------------------------------- gpufree
def test_gpu_in_use_is_reexported_with_its_thresholds() -> None:
    assert render.gpu_in_use is gpufree.gpu_in_use
    assert (render.GPU_BUSY_UTIL, render.GPU_BUSY_MEM_MIB) == (gpufree.GPU_BUSY_UTIL, gpufree.GPU_BUSY_MEM_MIB)
    idle = _gpu(0, 0.0, 300.0)
    assert not gpufree.gpu_in_use(idle)
    assert gpufree.gpu_in_use(dataclasses.replace(idle, util_pct=5.0))
    assert gpufree.gpu_in_use(dataclasses.replace(idle, mem_used_mib=1024.0))
    assert gpufree.gpu_in_use(_gpu(0, 0.0, 10.0, _proc()))
    assert gpufree.gpu_in_use(_gpu(0, None, None))                       # never sampled: never free
    assert not gpufree.gpu_in_use(_gpu(0, None, 10.0))                   # one reading is enough here …


def test_workstation_pick_is_stricter_and_ranked() -> None:
    assert not gpufree.is_free(_gpu(0, None, 10.0))                     # … but the pick wants both
    assert not gpufree.is_free(_gpu(0, 0.0, None))
    assert gpufree.is_free(_gpu(0, 4.9, 1023.0)) and not gpufree.is_free(_gpu(0, 0.0, 10.0, _proc()))
    gpus = [_gpu(3, 0.0, 900.0), _gpu(1, 50.0, 9000.0), _gpu(0, 2.0, 100.0), _gpu(2, 0.0, 100.0)]
    free, held = gpufree.split_free(gpus)
    assert [g.index for g in free] == [2, 0, 3]                          # memory, then utilisation, then index
    assert [g.index for g in held] == [1]
    assert (everyday.FREE_MEM_MIB, everyday.FREE_UTIL_PCT) == (gpufree.FREE_MEM_MIB, gpufree.FREE_UTIL_PCT)
    e_free, e_held = everyday.free_gpus(gpus)
    assert e_free == free and e_held == ["GPU 1 busy (50% · 8.8G · no process visible)"]
