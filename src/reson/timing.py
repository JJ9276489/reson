from __future__ import annotations

from collections import deque

from reson.types import EdgeEvent


class MorseTiming:
    def __init__(self, initial_unit_ms: float = 180.0, alpha: float = 0.2, history_size: int = 32):
        self.unit_ms = initial_unit_ms
        self.alpha = alpha
        self._recent: deque[int] = deque(maxlen=history_size)

    def on_press(self, event: EdgeEvent) -> str:
        duration = max(event.duration_ms, 1)
        self._recent.append(duration)
        self.unit_ms = (1.0 - self.alpha) * self.unit_ms + self.alpha * duration
        return "." if duration < (2.0 * self.unit_ms) else "-"

    def on_rest_gap(self, rest_ms: int) -> str:
        if rest_ms >= 7.0 * self.unit_ms:
            return "space"
        if rest_ms >= 3.0 * self.unit_ms:
            return "letter"
        return "none"
