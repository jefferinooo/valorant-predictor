"""In-memory rate limiting for the commentary endpoint (one server process)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import date


class RateLimiter:
    def __init__(self, per_minute: int = 10, daily_cap: int = 500, clock=time.monotonic, today=date.today) -> None:
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._clock = clock
        self._today = today
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = today()
        self._day_count = 0
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            if self._today() != self._day:
                self._day, self._day_count = self._today(), 0
            if self._day_count >= self.daily_cap:
                return False
            now = self._clock()
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            self._day_count += 1
            return True
