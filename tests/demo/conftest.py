"""Shared fixtures for the demo tests: a DemoWorld on a hand-driven clock (no wall time, no latency)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest

from runfleet.demo.world import DemoWorld
from tests.conftest import CLOCK_START


class StepClock:
    """A settable time source for DemoWorld (`now` callable)."""

    def __init__(self, start: datetime = CLOCK_START) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> StepClock:
    return StepClock()


@pytest.fixture
def world(clock: StepClock) -> DemoWorld:
    return DemoWorld(seed=7, now=clock, latency_scale=0.0, stream_poll_s=0.01)


@pytest.fixture
def make_world() -> Callable[..., tuple[DemoWorld, StepClock]]:
    def make(seed: int = 7) -> tuple[DemoWorld, StepClock]:
        clk = StepClock()
        return DemoWorld(seed=seed, now=clk, latency_scale=0.0, stream_poll_s=0.01), clk
    return make
