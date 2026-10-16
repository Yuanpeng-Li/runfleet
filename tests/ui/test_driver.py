"""Mouse stays in cell coordinates: herdr answers the 2048 query but keeps reporting cells (clicks hit (2,0))."""

from __future__ import annotations

import pytest

pytest.importorskip("termios")

from textual.drivers.linux_driver import LinuxDriver  # noqa: E402
from textual.messages import InBandWindowResize  # noqa: E402

from runfleet.ui.app import _cell_mouse_driver  # noqa: E402

pytestmark = pytest.mark.posix                     # Textual's Linux driver (termios)


class _Recorder:
    def __init__(self) -> None:
        self.out: list[str] = []
        self.forwarded: list[object] = []


def _driver() -> tuple[LinuxDriver, _Recorder]:
    cls = _cell_mouse_driver(LinuxDriver)
    assert cls is not LinuxDriver and issubclass(cls, LinuxDriver)
    drv = cls.__new__(cls)          # no terminal needed: only the write/forward paths are exercised
    rec = _Recorder()
    drv._mouse = True               # type: ignore[attr-defined]
    drv.write = rec.out.append      # type: ignore[method-assign]
    drv.flush = lambda: None        # type: ignore[method-assign]
    return drv, rec


def test_never_enables_pixel_mouse_or_in_band_resize() -> None:
    drv, rec = _driver()
    drv._enable_mouse_pixels()
    drv._enable_in_band_window_resize()
    assert rec.out == []


def test_supported_but_off_stays_off() -> None:
    drv, rec = _driver()
    drv.process_message(InBandWindowResize(True, False))
    assert rec.out == []            # neither 2048h nor 1016h


def test_already_enabled_is_switched_off() -> None:
    drv, rec = _driver()
    drv.process_message(InBandWindowResize(True, True))
    assert rec.out == ["\x1b[?2048l"]


def test_other_drivers_untouched() -> None:
    class Other:
        pass
    assert _cell_mouse_driver(Other) is Other  # type: ignore[arg-type]
