"""[WP1] FROZEN Clock protocol, RealClock, ManualClock (DESIGN.md §3.7)."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol

class Clock(Protocol):
    def now(self) -> datetime: ...               # aware UTC
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...

class RealClock:                                 # time.time / time.monotonic / asyncio.sleep
    def now(self) -> datetime:
        return datetime.fromtimestamp(time.time(), tz=UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

class ManualClock:                               # tests + deterministic UI snapshots
    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("ManualClock start must be timezone-aware")
        self._start = start.astimezone(UTC)
        self._elapsed = 0.0
        self._seq = itertools.count()
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []   # (deadline, seq, future) heap

    def now(self) -> datetime:
        return self._start + timedelta(seconds=self._elapsed)

    def monotonic(self) -> float:
        return self._elapsed

    async def sleep(self, seconds: float) -> None:   # completes when advance() passes its deadline
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._elapsed + seconds, next(self._seq), fut))
        await fut

    def advance(self, seconds: float) -> None:       # wakes due sleepers in deadline order
        """Move time forward by `seconds` (>= 0) and resolve every sleeper whose deadline is now reached.

        Woken coroutines run when the caller next yields to the event loop (e.g. `await asyncio.sleep(0)`); a sleeper
        that sleeps again from inside its wake-up is measured from the new time.
        """
        if seconds < 0:
            raise ValueError("ManualClock cannot go backwards")
        self._elapsed += seconds
        while self._sleepers and self._sleepers[0][0] <= self._elapsed:
            _, _, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)
