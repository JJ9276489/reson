from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from reson.calibration import CalibrationProfile, default_profile
from reson.features import FeatureSnapshot, RawFeatureEngine
from reson.types import EdgeEvent, EdgeState, EmgSample


class EdgeDetector(Protocol):
    def update(self, sample: EmgSample) -> EdgeState: ...

    def pop_events(self) -> list[EdgeEvent]: ...

    def flush(self, final_t_ms: int) -> list[EdgeEvent]: ...


@dataclass(frozen=True)
class DetectorDebug:
    t_ms: int
    raw: int
    env_in: int
    fast: float
    slow: float
    a: float
    sigma: float
    z: float
    state: EdgeState
    state_code: int
    down: int
    up: int
    press_class: EdgeState | None
    gated_refractory: bool
    gated_rest_gap: bool


@dataclass
class _Pending:
    state: EdgeState
    since_ms: int


class AdaptiveEdgeDetector:
    def __init__(
        self,
        *,
        t_low_enter: float = 2.4,
        t_low_exit: float = 1.7,
        t_high_enter: float = 4.2,
        t_high_exit: float = 3.2,
        min_dwell_ms: int = 90,
        min_event_ms: int = 60,
        min_rest_gap_ms: int = 140,
        refractory_ms: int = 90,
        k_rest: float = 0.8,
        rest_conf_dwell_ms: int = 80,
        tau_fast_ms: float = 35.0,
        tau_slow_ms: float = 1000.0,
        tau_baseline_ms: float = 2000.0,
        sigma_floor: float = 5.0,
        sigma_window_s: float = 3.0,
    ):
        self.t_low_enter = t_low_enter
        self.t_low_exit = t_low_exit
        self.t_high_enter = t_high_enter
        self.t_high_exit = t_high_exit
        self.min_dwell_ms = min_dwell_ms
        self.min_event_ms = min_event_ms
        self.min_rest_gap_ms = min_rest_gap_ms
        self.refractory_ms = refractory_ms
        self.k_rest = k_rest
        self.rest_conf_dwell_ms = rest_conf_dwell_ms

        self.features = RawFeatureEngine(
            tau_fast_ms=tau_fast_ms,
            tau_slow_ms=tau_slow_ms,
            tau_baseline_ms=tau_baseline_ms,
            sigma_floor=sigma_floor,
            sigma_window_s=sigma_window_s,
        )

        self._stable_state: EdgeState = "rest"
        self._stable_start_ms: int | None = None
        self._pending: _Pending | None = None
        self._events: list[EdgeEvent] = []
        self._last_snapshot: FeatureSnapshot | None = None
        self._last_debug: DetectorDebug | None = None

        self._confident_rest = True
        self._rest_conf_start_ms: int | None = None
        self._rest_start_ms: int | None = None
        self._next_press_after_ms = 0
        self._refractory_until_ms = 0

        self._press_start_ms: int | None = None
        self._press_class: EdgeState | None = None
        self._press_peak_z = 0.0

    @classmethod
    def from_profile(cls, profile: CalibrationProfile | None) -> "AdaptiveEdgeDetector":
        p = profile or default_profile()
        return cls(
            t_low_enter=p.t_low_enter,
            t_low_exit=p.t_low_exit,
            t_high_enter=p.t_high_enter,
            t_high_exit=p.t_high_exit,
            min_dwell_ms=p.min_dwell_ms,
            min_event_ms=p.min_event_ms,
            min_rest_gap_ms=p.min_rest_gap_ms,
            refractory_ms=p.refractory_ms,
            k_rest=p.k_rest,
            rest_conf_dwell_ms=p.rest_conf_dwell_ms,
            tau_fast_ms=p.tau_fast_ms,
            tau_slow_ms=p.tau_slow_ms,
            tau_baseline_ms=p.tau_baseline_ms,
            sigma_floor=p.sigma_floor,
            sigma_window_s=p.sigma_window_s,
        )

    def _update_confident_rest(self, sample_t_ms: int, z: float) -> None:
        rest_gate = z < (self.t_low_exit * self.k_rest)
        if self._stable_state == "rest" and rest_gate:
            if self._rest_conf_start_ms is None:
                self._rest_conf_start_ms = sample_t_ms
            self._confident_rest = (sample_t_ms - self._rest_conf_start_ms) >= self.rest_conf_dwell_ms
            return
        self._confident_rest = False
        self._rest_conf_start_ms = None

    def _candidate_state(self, sample_t_ms: int, z: float) -> EdgeState:
        if self._stable_state == "rest":
            if sample_t_ms < self._refractory_until_ms:
                return "rest"
            if sample_t_ms < self._next_press_after_ms:
                return "rest"
            if z >= self.t_high_enter:
                return "heavy"
            if z >= self.t_low_enter:
                return "light"
            return "rest"

        # Latch class until release. No light->heavy escalation in this version.
        strict_rest_gate = z < (self.t_low_exit * self.k_rest)
        if self._stable_state == "heavy":
            return "rest" if strict_rest_gate else "heavy"
        return "rest" if strict_rest_gate else "light"

    def _emit_rest_event(self, end_ms: int) -> None:
        if self._rest_start_ms is None:
            self._rest_start_ms = end_ms
            return
        dur = max(end_ms - self._rest_start_ms, 0)
        if dur <= 0:
            return
        self._events.append(
            EdgeEvent(
                state="rest",
                start_ms=self._rest_start_ms,
                end_ms=end_ms,
                duration_ms=dur,
            )
        )

    def _commit_down(self, new_state: EdgeState, t_ms: int, z: float) -> None:
        self._emit_rest_event(t_ms)
        self._stable_state = new_state
        self._stable_start_ms = t_ms
        self._press_start_ms = t_ms
        self._press_class = new_state
        self._press_peak_z = z
        self._pending = None

        self._events.append(
            EdgeEvent(
                state=new_state,
                start_ms=t_ms,
                end_ms=t_ms,
                duration_ms=0,
                phase="down",
                press_class=new_state,
            )
        )

    def _commit_up(self, t_ms: int) -> None:
        press_start = self._press_start_ms if self._press_start_ms is not None else t_ms
        press_class = self._press_class if self._press_class is not None else "light"
        dur = max(t_ms - press_start, 0)
        looked_real = self._press_peak_z >= self.t_high_enter

        if dur >= self.min_event_ms:
            self._events.append(
                EdgeEvent(
                    state=press_class,
                    start_ms=press_start,
                    end_ms=t_ms,
                    duration_ms=dur,
                    phase="up",
                    press_class=press_class,
                )
            )
            self._next_press_after_ms = t_ms + self.refractory_ms + self.min_rest_gap_ms
        else:
            if looked_real:
                self._next_press_after_ms = t_ms + self.refractory_ms + self.min_rest_gap_ms
            else:
                self._next_press_after_ms = t_ms + self.refractory_ms

        self._refractory_until_ms = t_ms + self.refractory_ms
        self._stable_state = "rest"
        self._stable_start_ms = t_ms
        self._rest_start_ms = t_ms
        self._press_start_ms = None
        self._press_class = None
        self._press_peak_z = 0.0
        self._pending = None

    def _commit_transition(self, new_state: EdgeState, t_ms: int, z: float) -> None:
        if new_state == self._stable_state:
            self._pending = None
            if self._stable_state in ("light", "heavy"):
                self._press_peak_z = max(self._press_peak_z, z)
            return

        if self._pending is None or self._pending.state != new_state:
            self._pending = _Pending(state=new_state, since_ms=t_ms)
            return

        if self._stable_state == "rest" and new_state in ("light", "heavy"):
            if (t_ms - self._pending.since_ms) < self.min_dwell_ms:
                return
            if t_ms < self._next_press_after_ms:
                self._pending = None
                return
            self._commit_down(new_state, t_ms, z)
            return

        if self._stable_state in ("light", "heavy") and new_state == "rest":
            required_release_ms = max(self.min_dwell_ms, self.rest_conf_dwell_ms)
            if (t_ms - self._pending.since_ms) < required_release_ms:
                return
            if z >= (self.t_low_exit * self.k_rest):
                return
            self._commit_up(t_ms)
            return

    def update(self, sample: EmgSample) -> EdgeState:
        if self._stable_start_ms is None:
            self._stable_start_ms = sample.t_ms
            self._rest_start_ms = sample.t_ms

        rest_adapt = self._confident_rest and self._stable_state == "rest" and self._pending is None
        if rest_adapt:
            preview_z = self.features.preview_z(sample)
            if preview_z >= (self.t_low_exit * self.k_rest):
                rest_adapt = False

        snap = self.features.update(sample, rest_adapt=rest_adapt)
        self._last_snapshot = snap

        self._update_confident_rest(sample.t_ms, snap.z)
        candidate = self._candidate_state(sample.t_ms, snap.z)
        self._commit_transition(candidate, sample.t_ms, snap.z)

        gated_refractory = sample.t_ms < self._refractory_until_ms
        gated_rest_gap = sample.t_ms < self._next_press_after_ms
        self._last_debug = DetectorDebug(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            fast=snap.fast,
            slow=snap.slow,
            a=snap.a,
            sigma=snap.sigma,
            z=snap.z,
            state=self._stable_state,
            state_code={"rest": 0, "light": 1, "heavy": 2}[self._stable_state],
            down=1 if self._events and self._events[-1].phase == "down" and self._events[-1].end_ms == sample.t_ms else 0,
            up=1 if self._events and self._events[-1].phase == "up" and self._events[-1].end_ms == sample.t_ms else 0,
            press_class=self._press_class,
            gated_refractory=gated_refractory,
            gated_rest_gap=gated_rest_gap,
        )
        return self._stable_state

    def pop_events(self) -> list[EdgeEvent]:
        out = self._events
        self._events = []
        return out

    def flush(self, final_t_ms: int) -> list[EdgeEvent]:
        out = self.pop_events()
        if self._stable_state == "rest":
            if self._rest_start_ms is not None and final_t_ms > self._rest_start_ms:
                out.append(
                    EdgeEvent(
                        state="rest",
                        start_ms=self._rest_start_ms,
                        end_ms=final_t_ms,
                        duration_ms=final_t_ms - self._rest_start_ms,
                    )
                )
            return out

        press_start = self._press_start_ms if self._press_start_ms is not None else final_t_ms
        press_class = self._press_class if self._press_class is not None else "light"
        dur = max(final_t_ms - press_start, 0)
        if dur >= self.min_event_ms:
            out.append(
                EdgeEvent(
                    state=press_class,
                    start_ms=press_start,
                    end_ms=final_t_ms,
                    duration_ms=dur,
                    phase="up",
                    press_class=press_class,
                )
            )
        return out

    def last_debug(self) -> DetectorDebug | None:
        return self._last_debug


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
    def from_calibration(cls, profile: CalibrationProfile | None) -> "ThresholdEdgeDetector":
        p = profile or default_profile()
        return cls(
            light_threshold=p.light_threshold,
            heavy_threshold=p.heavy_threshold,
            hysteresis_margin=p.hysteresis_margin,
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


def make_detector(mode: str, profile: CalibrationProfile | None) -> EdgeDetector:
    if mode == "threshold":
        return ThresholdEdgeDetector.from_calibration(profile)
    return AdaptiveEdgeDetector.from_profile(profile)
