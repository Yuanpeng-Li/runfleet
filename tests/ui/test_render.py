"""[WP3] ui/render.py and ui/theme.py: pure renderables, thresholds, NO_COLOR."""

from __future__ import annotations

import pytest
from rich.text import Text

from runfleet.models import ConnState, ErrorKind, HostError, HostStatus, JobState, StateGroup
from runfleet.ui import render, theme


def _styles(text: Text) -> set[str]:
    return {str(span.style) for span in text.spans}


@pytest.mark.parametrize(("value", "filled"), [(0, 0), (50, 4), (100, 8)])
def test_gpu_bar_width_and_fill(value: float, filled: int) -> None:
    bar = render.gpu_bar(value, 10)
    assert bar.cell_len == 10
    assert bar.plain[0] == "▕" and bar.plain[-1] == "▏"
    assert bar.plain[1:-1].count("█") == filled


def test_gpu_bar_partial_blocks() -> None:
    bar = render.gpu_bar(55, 10)          # 8 inner cells * 0.55 = 4.4 cells -> 4 full + 3/8 block
    assert bar.plain[1:-1].startswith("████▍")


def test_gpu_bar_none_is_not_zero() -> None:
    bar = render.gpu_bar(None, 12)
    assert "n/a" in bar.plain and "█" not in bar.plain
    assert render.gpu_bar(None, 4).plain.strip("▕▏ ") == ""


def test_gpu_bar_clamps_and_frameless() -> None:
    assert render.gpu_bar(250, 6, frame=False).plain == "██████"
    assert render.gpu_bar(-10, 6, frame=False).plain == "      "


@pytest.mark.parametrize(("value", "lv"), [(None, "low"), (0, "low"), (9.9, "low"), (10, "mid"), (74, "mid"),
                                           (75, "high"), (100, "high")])
def test_levels_nvitop_defaults(value: float | None, lv: str) -> None:
    assert theme.level(value, theme.DEFAULT_UTIL_THRESHOLDS) == lv


def test_bar_colour_follows_threshold() -> None:
    assert theme.RED in " ".join(_styles(render.gpu_bar(90, 10)))
    assert theme.YELLOW in " ".join(_styles(render.gpu_bar(50, 10)))
    assert theme.GREEN in " ".join(_styles(render.gpu_bar(5, 10)))
    assert theme.GREEN in " ".join(_styles(render.gpu_bar(50, 10, (60, 90))))


def test_mini_util() -> None:
    assert render.mini_util(None).plain == "–"
    cell = render.mini_util(98)
    assert cell.plain.endswith(" 98%") and cell.plain[0] in "▁▂▃▄▅▆▇█"
    assert render.mini_util(5).plain[0] == "▁"


def test_state_badge() -> None:
    assert render.state_badge(JobState.PENDING, held=True).plain == "PDH"
    assert render.state_badge(JobState.RUNNING).plain == "R"
    assert render.state_badge(None).plain == "?"
    assert theme.ORANGE in str(render.state_badge(JobState.PREEMPTED).style)
    assert theme.RED in str(render.state_badge(JobState.FAILED).style)


def test_state_word() -> None:
    assert render.state_word(JobState.CANCELLED, "CANCELLED by 610001").plain == "CANCELLED"
    assert render.state_word(None).plain == "UNKNOWN"


def _status(state: ConnState, **kw: object) -> HostStatus:
    return HostStatus(name="lyra", state=state, **kw)  # type: ignore[arg-type]


def test_host_chip_forms() -> None:
    ok = _status(ConnState.OK)
    assert render.host_chip(ok, 8).plain == "● lyra 8s"
    assert render.host_chip(ok, 8, form="short").plain == "● lyra"
    assert render.host_chip(ok, form="name").plain == "●lyra"
    assert render.host_chip(ok, form="num", number=3).plain == "●3"
    assert render.host_chip(_status(ConnState.RETRYING), 240, retry_in=38).plain == "◐ lyra retry 38s"
    assert render.host_chip(_status(ConnState.RETRYING), None).plain == "◐ lyra retry"
    assert render.host_chip(_status(ConnState.NEEDS_AUTH), 800).plain == "✖ lyra auth"
    assert render.host_chip(_status(ConnState.DEGRADED), 3).plain == "! lyra 3s"
    assert render.host_chip(_status(ConnState.DISABLED)).plain == "○ lyra off"
    assert render.host_chip(_status(ConnState.CONNECTING)).plain == "◌ lyra …"


def test_host_chip_dns_vs_down() -> None:
    dns = _status(ConnState.UNREACHABLE, error=HostError(ErrorKind.DNS, "Could not resolve hostname"))
    assert render.host_chip(dns).plain.endswith("dns")
    assert render.host_chip(_status(ConnState.UNREACHABLE)).plain.endswith("down")


def test_host_chip_none_age() -> None:
    assert render.host_chip(_status(ConnState.OK), None).plain == "● lyra"


def test_conn_text_and_styles() -> None:
    assert render.conn_text(ConnState.NEEDS_AUTH).plain == "✖ needs auth"
    assert theme.MAGENTA in theme.conn_style(ConnState.NEEDS_AUTH)
    assert theme.RED in theme.conn_style(ConnState.UNREACHABLE)
    assert theme.YELLOW in theme.conn_style(ConnState.RETRYING)


def test_count_bar() -> None:
    assert render.count_bar(3, 4).plain == "███░"
    assert render.count_bar(0, 12).plain == "░" * 12
    assert render.count_bar(None, 4).plain == ""
    assert len(render.count_bar(10, 40, max_cells=8).plain) == 8


def test_reason_text() -> None:
    assert render.reason_text("Priority").plain == "priority"
    assert render.reason_text("Priority", short=False).plain == "waiting for priority"
    never = render.reason_text("DependencyNeverSatisfied", short=False)
    assert never.plain == "will never run: dependency failed" and theme.RED in str(never.style)
    assert render.reason_text("SomethingNew").plain == "SomethingNew"
    assert render.reason_text(None).plain == ""
    assert render.reason_text("None").plain == ""


def test_fit() -> None:
    assert render.fit(Text("abc"), 5).plain == "abc  "
    assert render.fit(Text("abcdef"), 4).plain == "abc…"


def test_no_color_replaces_colour_with_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    assert theme.no_color()
    assert theme.state_style(JobState.FAILED) == "reverse"
    assert theme.state_style(JobState.RUNNING) == "bold"
    # review: a reverse-video █ renders in the background colour, so the busiest GPU's bar looked empty
    assert theme.level_style(90, theme.DEFAULT_UTIL_THRESHOLDS) == "bold"
    assert not any("reverse" in st for st in _styles(render.gpu_bar(98, 10)))
    assert render.mini_util(98).plain.endswith("!") and not render.mini_util(50).plain.endswith("!")
    assert theme.host_color(None, 0) == "bold"
    assert theme.track() == ""
    assert "#" not in " ".join(_styles(render.gpu_bar(90, 10)))


def test_every_state_group_has_a_style() -> None:
    for state in JobState:
        assert theme.state_style(state)
    for group in StateGroup:
        assert theme.state_style_group(group)


def test_host_color_prefers_config() -> None:
    from runfleet.config import HostConfig

    assert theme.host_color(HostConfig("x", "x", color="#123456"), 0) == "#123456"
    assert theme.host_color(HostConfig("x", "x"), 0) == theme.HOST_PALETTE[0]
    assert theme.host_color(None, len(theme.HOST_PALETTE)) == theme.HOST_PALETTE[0]


def test_temp_style() -> None:
    assert theme.temp_style(None) == ""
    assert theme.temp_style(30) == ""
    assert theme.YELLOW in theme.temp_style(78)
    assert theme.RED in theme.temp_style(85)



def test_light_background_detection() -> None:
    """Review: hard-coded light greys on a transparent background were near-invisible on light terminals."""
    assert theme.light_background({"COLORFGBG": "0;15"}) and theme.light_background({"COLORFGBG": "0;default;7"})
    assert not theme.light_background({"COLORFGBG": "15;0"}) and not theme.light_background({})
    assert theme.light_background({"RUNFLEET_THEME": "light", "COLORFGBG": "15;0"})
    assert not theme.light_background({"RUNFLEET_THEME": "dark", "COLORFGBG": "0;15"})


def test_light_palette_is_dark_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.setenv("RUNFLEET_THEME", "light")
    light = importlib.reload(theme)
    try:
        def luminance(hex_: str) -> float:
            r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        assert light.LIGHT and light.TEXTUAL_THEME.dark is False
        assert all(luminance(c) < 0.5 for c in (light.MUTED, light.GREY, light.YELLOW, light.ACCENT))
    finally:
        monkeypatch.delenv("RUNFLEET_THEME")
        importlib.reload(theme)
