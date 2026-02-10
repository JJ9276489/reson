from __future__ import annotations

from dataclasses import dataclass
from statistics import median
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
    filtered_raw: float
    fast: float
    slow: float
    a: float
    sigma: float
    z: float
    phase: str
    armed: bool
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
        filter_enabled: bool = True,
        sigma_floor: float = 5.0,
        sigma_window_s: float = 3.0,
        bootstrap_ms: int = 3000,
        quiet_window_ms: int = 80,
        quiet_fraction: float = 0.25,
        hp_hz: float = 20.0,
        lp_hz: float = 230.0,
        notch_hz: float = 60.0,
        notch_q: float = 20.0,
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
        self.bootstrap_ms = bootstrap_ms
        self.quiet_window_ms = quiet_window_ms
        self.quiet_fraction = quiet_fraction

        self.features = RawFeatureEngine(
            tau_fast_ms=tau_fast_ms,
            tau_slow_ms=tau_slow_ms,
            tau_baseline_ms=tau_baseline_ms,
            filter_enabled=filter_enabled,
            sigma_floor=sigma_floor,
            sigma_window_s=sigma_window_s,
            hp_hz=hp_hz,
            lp_hz=lp_hz,
            notch_hz=notch_hz,
            notch_q=notch_q,
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
        self._phase = "BOOTSTRAP" if bootstrap_ms > 0 else "RUNNING"
        self._bootstrap_start_ms: int | None = None
        self._bootstrap_samples: list[FeatureSnapshot] = []

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
            filter_enabled=p.filter_enabled,
            sigma_floor=p.sigma_floor,
            sigma_window_s=p.sigma_window_s,
            bootstrap_ms=p.bootstrap_ms,
            quiet_window_ms=p.quiet_window_ms,
            quiet_fraction=p.quiet_fraction,
            hp_hz=p.hp_hz,
            lp_hz=p.lp_hz,
            notch_hz=p.notch_hz,
            notch_q=p.notch_q,
        )

    def phase(self) -> str:
        return self._phase

    def is_armed(self) -> bool:
        return self._phase == "RUNNING"

    def _bootstrap_init(self, t_ms: int) -> None:
        if not self._bootstrap_samples:
            return
        vals = self._bootstrap_samples
        times = [s.t_ms for s in vals]
        dt_ms = int(median([max(b - a, 1) for a, b in zip(times, times[1:])])) if len(times) > 1 else 4
        win_n = max(int(self.quiet_window_ms / max(dt_ms, 1)), 4)
        windows: list[tuple[float, int, int]] = []
        for i in range(0, max(len(vals) - win_n + 1, 1), max(win_n // 2, 1)):
            chunk = vals[i : i + win_n]
            if not chunk:
                continue
            fr = [s.filtered_raw for s in chunk]
            center = sum(fr) / len(fr)
            rms = sum((x - center) ** 2 for x in fr) / len(fr)
            slope = sum(abs(fr[j] - fr[j - 1]) for j in range(1, len(fr))) / max(len(fr) - 1, 1)
            score = rms + (0.5 * slope)
            windows.append((score, i, i + len(chunk)))
        windows.sort(key=lambda w: w[0])
        keep = max(1, int(len(windows) * max(min(self.quiet_fraction, 0.8), 0.05)))
        keep_windows = windows[:keep]
        quiet_idx: set[int] = set()
        for _, s, e in keep_windows:
            quiet_idx.update(range(s, e))
        quiet = [vals[i] for i in sorted(quiet_idx)] if quiet_idx else vals
        filtered = [s.filtered_raw for s in quiet]
        rects = [s.rect for s in quiet]
        a_vals = [s.a for s in quiet]
        baseline = median(filtered) if filtered else 0.0
        slow = median(rects) if rects else 0.0
        a_sorted = sorted(a_vals) if a_vals else [0.0]
        p50 = a_sorted[int((len(a_sorted) - 1) * 0.5)]
        p95 = a_sorted[int((len(a_sorted) - 1) * 0.95)]
        sigma = max(p95 - p50, self.features.sigma_floor)
        self.features.set_bootstrap_state(
            baseline_raw=baseline,
            slow=slow,
            sigma=sigma,
            t_ms=t_ms,
            a_seed=a_vals[-64:],
        )
        self.features.fast = slow
        self._stable_state = "rest"
        self._stable_start_ms = t_ms
        self._rest_start_ms = t_ms
        self._pending = None
        self._events = []
        self._press_start_ms = None
        self._press_class = None
        self._press_peak_z = 0.0
        self._confident_rest = False
        self._rest_conf_start_ms = None

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
        if self._bootstrap_start_ms is None:
            self._bootstrap_start_ms = sample.t_ms

        if self._phase == "BOOTSTRAP":
            snap = self.features.update(sample, rest_adapt=False)
            self._last_snapshot = snap
            self._bootstrap_samples.append(snap)
            if (sample.t_ms - self._bootstrap_start_ms) >= self.bootstrap_ms:
                self._bootstrap_init(sample.t_ms)
                self._phase = "ARMING"
            debug_state: EdgeState = "heavy" if snap.z >= self.t_high_enter else "light" if snap.z >= self.t_low_enter else "rest"
            self._last_debug = DetectorDebug(
                t_ms=sample.t_ms,
                raw=sample.raw,
                env_in=sample.env,
                filtered_raw=snap.filtered_raw,
                fast=snap.fast,
                slow=snap.slow,
                a=snap.a,
                sigma=snap.sigma,
                z=snap.z,
                phase=self._phase,
                armed=False,
                state=debug_state,
                state_code={"rest": 0, "light": 1, "heavy": 2}[debug_state],
                down=0,
                up=0,
                press_class=None,
                gated_refractory=False,
                gated_rest_gap=False,
            )
            return debug_state

        rest_adapt = self._confident_rest and self._stable_state == "rest" and self._pending is None
        if rest_adapt:
            preview_z = self.features.preview_z(sample)
            if preview_z >= (self.t_low_exit * self.k_rest):
                rest_adapt = False
        snap = self.features.update(sample, rest_adapt=rest_adapt)
        self._last_snapshot = snap

        if self._phase == "ARMING":
            self._stable_state = "rest"
            self._pending = None
            self._update_confident_rest(sample.t_ms, snap.z)
            if self._confident_rest:
                self._phase = "RUNNING"
            debug_state: EdgeState = "heavy" if snap.z >= self.t_high_enter else "light" if snap.z >= self.t_low_enter else "rest"
            self._last_debug = DetectorDebug(
                t_ms=sample.t_ms,
                raw=sample.raw,
                env_in=sample.env,
                filtered_raw=snap.filtered_raw,
                fast=snap.fast,
                slow=snap.slow,
                a=snap.a,
                sigma=snap.sigma,
                z=snap.z,
                phase=self._phase,
                armed=self.is_armed(),
                state=debug_state,
                state_code={"rest": 0, "light": 1, "heavy": 2}[debug_state],
                down=0,
                up=0,
                press_class=None,
                gated_refractory=False,
                gated_rest_gap=False,
            )
            return debug_state

        self._update_confident_rest(sample.t_ms, snap.z)
        candidate = self._candidate_state(sample.t_ms, snap.z)
        self._commit_transition(candidate, sample.t_ms, snap.z)

        gated_refractory = sample.t_ms < self._refractory_until_ms
        gated_rest_gap = sample.t_ms < self._next_press_after_ms
        self._last_debug = DetectorDebug(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            filtered_raw=snap.filtered_raw,
            fast=snap.fast,
            slow=snap.slow,
            a=snap.a,
            sigma=snap.sigma,
            z=snap.z,
            phase=self._phase,
            armed=self.is_armed(),
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
