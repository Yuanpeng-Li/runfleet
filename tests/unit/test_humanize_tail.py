"""util.humanize: cutting text from the left never leaves a combining mark without its base (review BACK-47)."""

from __future__ import annotations

from runfleet.util.humanize import _tail, truncate_left


def test_tail_drops_an_orphaned_combining_mark() -> None:
    text = "abcdéA"                                   # decomposed é
    assert _tail(text, 1) == "A"
    assert _tail(text, 2) == "éA"
    assert not truncate_left(text, 2)[1:2] == "́"


def test_tail_drops_a_dangling_zero_width_joiner() -> None:
    assert _tail("x‍\U0001f525", 2) == "\U0001f525"
