"""[WP2] util/slurmfmt: placeholders, durations, timestamps (epoch + ISO fallback), memory, TRES, hostlists, states."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runfleet.util.slurmfmt import (expand_hostlist, nullish, parse_duration, parse_exit, parse_index_list,
                                    parse_mem_bytes, parse_mem_mib, parse_timestamp, parse_tres, parse_tres_per,
                                    seconds_prefix, split_outside, split_state, tres_gpu)

UTC_M5 = timedelta(hours=-5)


@pytest.mark.parametrize("value", ["", "N/A", "n/a", "(null)", "None", "None assigned", "Unknown", "[N/A]",
                                   "[Not Supported]", "[Unknown Error]", "  N/A  ", None])
def test_nullish(value: str | None) -> None:
    assert nullish(value) is None


def test_nullish_keeps_values() -> None:
    assert nullish(" n-40-01 ") == "n-40-01"
    assert nullish("0") == "0"


@pytest.mark.parametrize(("text", "expected"), [
    ("4:07:00", timedelta(hours=4, minutes=7)),
    ("3-00:00:00", timedelta(days=3)),
    ("2-19:53:00", timedelta(days=2, hours=19, minutes=53)),
    ("0:00", timedelta(0)),
    ("5:00", timedelta(minutes=5)),
    ("38:18", timedelta(minutes=38, seconds=18)),
    ("00:00.123", timedelta(milliseconds=123)),
    ("1-17:45:42", timedelta(days=1, hours=17, minutes=45, seconds=42)),
])
def test_parse_duration(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["UNLIMITED", "INVALID", "NOT_SET", "Partition_Limit", "N/A", "", "garbage",
                                  "213503982334-14:25:51"])
def test_parse_duration_none(text: str) -> None:
    # the .extern step's garbage AveCPU overflows timedelta and must not raise
    assert parse_duration(text) is None


def test_epoch_timestamps_are_aware_utc() -> None:
    ts = parse_timestamp("1788696459", UTC_M5)
    assert ts == datetime.fromtimestamp(1788696459, tz=UTC)
    assert ts is not None and ts.utcoffset() == timedelta(0)
    assert parse_timestamp("0", UTC_M5) is None


def test_iso_fallback_is_localized() -> None:
    ts = parse_timestamp("2026-09-07T06:31:41", UTC_M5)
    assert ts == datetime(2026, 9, 7, 11, 31, 41, tzinfo=UTC)
    for v in ("N/A", "Unknown", "None", ""):
        assert parse_timestamp(v, UTC_M5) is None


@pytest.mark.parametrize(("text", "unit", "expected"), [
    ("48G", "M", 48 << 30), ("10398988K", "", 10398988 << 10), ("24870M", "M", 24870 << 20), ("0", "M", 0),
    ("4000Mn", "M", 4000 << 20), ("2Gc", "M", 2 << 30), ("512", "M", 512 << 20), ("1.5G", "M", int(1.5 * (1 << 30))),
])
def test_parse_mem(text: str, unit: str, expected: int) -> None:
    assert parse_mem_bytes(text, unit) == expected


def test_parse_mem_mib_and_bad() -> None:
    assert parse_mem_mib("24870M") == 24870.0
    assert parse_mem_bytes("lots") is None
    assert parse_mem_bytes(None) is None


def test_parse_exit() -> None:
    assert parse_exit("2:0") == (2, 0)
    assert parse_exit("143:0") == (143, 0)
    assert parse_exit("0:9") == (0, 9)
    assert parse_exit("") == (None, None)


def test_tres() -> None:
    t = parse_tres("cpu=8,mem=48G,node=1,billing=8,gres/gpu=1,gres/gpu:nvidia_a100-sxm4-40gb=1")
    assert t["mem"] == "48G" and t["gres/gpu"] == "1"
    assert tres_gpu(t) == (1, "nvidia_a100-sxm4-40gb")
    assert tres_gpu(parse_tres("cpu=16,mem=256G,node=1,billing=16,gres/gpu=2")) == (2, None)
    assert tres_gpu(parse_tres("billing=8,cpu=8,gres/gpu:rtx6000=1,mem=48G")) == (1, "rtx6000")
    assert tres_gpu(parse_tres("gres/gpu=3,gres/gpu:a=1,gres/gpu:b=2")) == (3, None)   # heterogeneous: no type
    assert parse_tres("(null)") == {} and parse_tres("") == {}


@pytest.mark.parametrize(("text", "expected"), [
    ("gres/gpu:nvidia_h200_nvl:2", (2, "nvidia_h200_nvl")), ("gres/gpu:1", (1, None)), ("gres:gpu:2", (2, None)),
    ("gres/gpu:nvidia_h100_nvl_1g.12gb:1", (1, "nvidia_h100_nvl_1g.12gb")), ("gres/gpu", (1, None)),
    ("gres/gpu:A30", (1, "a30")), ("N/A", (None, None)), ("cpu=8", (None, None)),
])
def test_tres_per(text: str, expected: tuple[int | None, str | None]) -> None:
    assert parse_tres_per(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("n-[01-03,05]", ["n-01", "n-02", "n-03", "n-05"]),
    ("lyra-gpu-s71-[00-03]", ["lyra-gpu-s71-00", "lyra-gpu-s71-01", "lyra-gpu-s71-02", "lyra-gpu-s71-03"]),
    ("n-40-01", ["n-40-01"]),
    ("n-31-[04-05],n-33-[04-06],n-37-02", ["n-31-04", "n-31-05", "n-33-04", "n-33-05", "n-33-06", "n-37-02"]),
    ("a[1-2]-b[3-4]", ["a1-b3", "a1-b4", "a2-b3", "a2-b4"]),
    ("n[8-10]", ["n8", "n9", "n10"]),
    ("", []),
])
def test_expand_hostlist(text: str, expected: list[str]) -> None:
    assert expand_hostlist(text) == expected


def test_expand_hostlist_cap() -> None:
    assert len(expand_hostlist("n[0-99999]", cap=4096)) == 4096
    assert len(expand_hostlist("a[0-99],b[0-99]", cap=150)) == 150


@pytest.mark.parametrize(("text", "base", "flags"), [
    ("mixed-", "mixed", {"-"}), ("planned", "planned", set()), ("idle*", "idle", {"*"}),
    ("drained~", "drained", {"~"}), ("MIXED+PLANNED", "mixed", {"+"}), ("reserved", "reserved", set()),
    ("allocated#", "allocated", {"#"}),
])
def test_split_state(text: str, base: str, flags: set[str]) -> None:
    assert split_state(text) == (base, frozenset(flags))


def test_misc() -> None:
    assert split_outside("gpu:a:3(IDX:0,2-3),gpu:b:1") == ["gpu:a:3(IDX:0,2-3)", "gpu:b:1"]
    assert parse_index_list("0,2-3") == (0, 2, 3)
    assert parse_index_list("N/A") == ()
    assert parse_index_list("x") == ()
    assert seconds_prefix("300 sec") == 300
    assert seconds_prefix("task=30,energy=0") == 30
    assert seconds_prefix(None) is None


# ---------------------------------------------------------------- review 2026-10
def test_hostlist_expansion_stops_at_the_cap() -> None:
    """BACK-33: the full product / range was built before the cap was applied (`n[0-999999999]` hung the UI)."""
    import time
    t0 = time.monotonic()
    assert len(expand_hostlist("n[0000-4095]x[0000-4095]")) == 4096
    big = expand_hostlist("n[0-999999999]")
    assert len(big) == 4096 and big[:2] == ["n0", "n1"]
    assert time.monotonic() - t0 < 1.0
    assert expand_hostlist("n-[01-03,05]") == ["n-01", "n-02", "n-03", "n-05"]


@pytest.mark.parametrize(("text", "expected"), [("gres/gpu:a100:1,gres/gpu:v100:1", (2, None)),
                                                ("gres/gpu:a100:1,gres/gpu:a100:2", (3, "a100"))])
def test_tres_per_sums_every_gpu_entry(text: str, expected: tuple[int, str | None]) -> None:
    """BACK-18: only the first GPU entry of a mixed request was counted."""
    from runfleet.util.slurmfmt import parse_tres_per
    assert parse_tres_per(text) == expected


def test_tres_gpu_without_aggregate_sums_typed_values() -> None:
    from runfleet.util.slurmfmt import tres_gpu
    assert tres_gpu({"gres/gpu:a100": "1", "gres/gpu:v100": "1"}) == (2, None)
    assert tres_gpu({"gres/gpu": "2", "gres/gpu:a100": "1", "gres/gpu:v100": "1"}) == (2, None)


@pytest.mark.parametrize(("task", "expected"), [
    ("1-9:2%5", [1, 3, 5, 7, 9]), ("[1,3,7-9]", [1, 3, 7, 8, 9]), ("1,3,5,...,45,...%4", None), ("x", None),
    ("1-99999999", None)])
def test_expand_array_tasks(task: str, expected: list[int] | None) -> None:
    from runfleet.util.slurmfmt import compress_tasks, expand_array_tasks
    assert expand_array_tasks(task) == expected
    if expected:
        assert expand_array_tasks(compress_tasks(expected)) == expected
