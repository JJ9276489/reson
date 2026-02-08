from __future__ import annotations

from dataclasses import dataclass

from reson.calibration import CalibrationProfile
from reson.types import EdgeEvent, EdgeState, EmgSample


@dataclass
class _Pending:
    state: EdgeState
    since_ms: int


class ThresholdEdgeDetector:
    def __init__(
        self,
        light_threshold: float,
        heavy_threshold: float,
        hysteresis_margin: float,
        min_hold_ms: int = 80,
        min_event_ms: int = 50,
    ):
        self.light_threshold = light_threshold
        self.heavy_threshold = heavy_threshold
        self.hysteresis_margin = hysteresis_margin
        self.min_hold_ms = min_hold_ms
        self.min_event_ms = min_event_ms

        self._stable_state: EdgeState = "rest"
        self._stable_start_ms: int | None = None
        self._pending: _Pending | None = None
        self._events: list[EdgeEvent] = []

    @classmethod
    def from_calibration(cls, profile: CalibrationProfile) -> "ThresholdEdgeDetector":
        return cls(
            light_threshold=profile.light_threshold,
            heavy_threshold=profile.heavy_threshold,
            hysteresis_margin=profile.hysteresis_margin,
        )

    def _candidate_state(self, env: int) -> EdgeState:
        light_enter = self.light_threshold + self.hysteresis_margin
        light_exit = self.light_threshold - self.hysteresis_margin
        heavy_enter = self.heavy_threshold + self.hysteresis_margin
        heavy_exit = self.heavy_threshold - self.hysteresis_margin

        if self._stable_state == "heavy":
            if env < heavy_exit:
                return "light" if env >= light_enter else "rest"
            return "heavy"

        if self._stable_state == "light":
            if env >= heavy_enter:
                return "heavy"
            if env < light_exit:
                return "rest"
            return "light"

        if env >= heavy_enter:
            return "heavy"
        if env >= light_enter:
            return "light"
        return "rest"

    def _maybe_commit_transition(self, new_state: EdgeState, t_ms: int) -> None:
        if new_state == self._stable_state:
            self._pending = None
            return

        if self._pending is None or self._pending.state != new_state:
            self._pending = _Pending(state=new_state, since_ms=t_ms)
            return

        if t_ms - self._pending.since_ms < self.min_hold_ms:
            return

        if self._stable_start_ms is not None:
            event = EdgeEvent(
                state=self._stable_state,
                start_ms=self._stable_start_ms,
                end_ms=t_ms,
                duration_ms=t_ms - self._stable_start_ms,
            )
            if event.duration_ms >= self.min_event_ms:
                self._events.append(event)

        self._stable_state = new_state
        self._stable_start_ms = t_ms
        self._pending = None

    def update(self, sample: EmgSample) -> EdgeState:
        if self._stable_start_ms is None:
            self._stable_start_ms = sample.t_ms

        candidate = self._candidate_state(sample.env)
        self._maybe_commit_transition(candidate, sample.t_ms)
        return self._stable_state

    def pop_events(self) -> list[EdgeEvent]:
        out = self._events
        self._events = []
        return out

    def flush(self, final_t_ms: int) -> list[EdgeEvent]:
        if self._stable_start_ms is None:
            return []

        event = EdgeEvent(
            state=self._stable_state,
            start_ms=self._stable_start_ms,
            end_ms=final_t_ms,
            duration_ms=max(final_t_ms - self._stable_start_ms, 0),
        )
        out = self.pop_events()
        if event.duration_ms >= self.min_event_ms:
            out.append(event)
        return out
