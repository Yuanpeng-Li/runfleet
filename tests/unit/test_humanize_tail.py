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


def test_emoji_sequences_are_measured_as_drawn() -> None:
    """Per code point, ``❤️`` (heart + VS16) counted 1 cell and a ZWJ family 6: the result came out too wide (23
    cells in 12, then cut again with a second "…") or short."""
    from rich.cells import cell_len

    from runfleet.util.humanize import truncate, truncate_name

    names = ["✔️" * 12 + "-run-s43--a02", "exp-❤️-heart-very-long-name-s43--a02",
             "exp-👨‍👩‍👧-family-run-s43--a02-and-more"]
    for name in names:
        for width in (8, 12, 16, 20, 26):
            for keep_tail in (False, True):
                out = truncate(name, width, keep_tail=keep_tail)
                assert cell_len(out) <= width and out.count("…") == 1, (name, width, out)
                assert cell_len(out) >= width - 1                      # no blanks where a glyph was over-counted
            assert cell_len(truncate_name(name, width)) <= width
    assert truncate(names[0], 20, keep_tail=True).endswith("…3--a02")
    assert truncate("exp-👨‍👩‍👧-family-run", 9) == "exp-👨‍👩‍👧-f…"
