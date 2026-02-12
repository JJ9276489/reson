from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import Protocol

from reson.calibration import CalibrationProfile, default_profile
from reson.features import FeatureSnapshot, RawFeatureEngine, robust_scale
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
    filtered_raw_hp: float
    rms_state: float
    rest_center: float
    rest_scale: float
    u: float
    lf_energy: float
    artifact_ratio: float
    artifact_score: float
    artifact_gated: bool
    rest_confident: bool
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
        u_light_enter: float = 1.0,
        u_light_exit: float = 0.8,
        u_heavy_enter: float = 2.0,
        u_heavy_exit: float = 1.6,
        u_rest_conf_threshold: float = 0.6,
        min_dwell_ms: int = 90,
        min_event_ms: int = 60,
        min_rest_gap_ms: int = 140,
        refractory_ms: int = 90,
        rest_conf_dwell_ms: int = 80,
        tau_baseline_ms: float = 2000.0,
        filter_enabled: bool = True,
        rest_scale_floor: float = 5.0,
        rms_state_window_ms: int = 180,
        rest_stats_window_s: float = 3.0,
        bootstrap_ms: int = 3000,
        quiet_window_ms: int = 80,
        quiet_fraction: float = 0.25,
        hp_hz: float = 20.0,
        lp_hz: float = 230.0,
        notch_hz: float = 60.0,
        notch_q: float = 20.0,
        artifact_enter: float = 6.0,
        artifact_exit: float = 3.5,
        artifact_holdoff_ms: int = 180,
        artifact_lf_hz: float = 10.0,
        slope_fast_tau_ms: float = 40.0,
        slope_slow_tau_ms: float = 400.0,
        # Legacy args retained for compatibility (unused in v2.4.1 path).
        tau_fast_ms: float | None = None,
        tau_slow_ms: float | None = None,
        sigma_floor: float | None = None,
        sigma_window_s: float | None = None,
        k_rest: float | None = None,
    ):
        self.u_light_enter = u_light_enter
        self.u_light_exit = u_light_exit
        self.u_heavy_enter = u_heavy_enter
        self.u_heavy_exit = u_heavy_exit
        self.u_rest_conf_threshold = min(u_rest_conf_threshold, u_light_exit)
        self.min_dwell_ms = min_dwell_ms
        self.min_event_ms = min_event_ms
        self.min_rest_gap_ms = min_rest_gap_ms
        self.refractory_ms = refractory_ms
        self.rest_conf_dwell_ms = rest_conf_dwell_ms
        self.bootstrap_ms = bootstrap_ms
        self.quiet_window_ms = quiet_window_ms
        self.quiet_fraction = quiet_fraction
        self.artifact_enter = artifact_enter
        self.artifact_exit = artifact_exit
        self.artifact_holdoff_ms = artifact_holdoff_ms

        self.features = RawFeatureEngine(
            tau_baseline_ms=tau_baseline_ms,
            filter_enabled=filter_enabled,
            rest_scale_floor=rest_scale_floor,
            rms_state_window_ms=rms_state_window_ms,
            rest_stats_window_s=rest_stats_window_s,
            hp_hz=hp_hz,
            lp_hz=lp_hz,
            notch_hz=notch_hz,
            notch_q=notch_q,
            artifact_lf_hz=artifact_lf_hz,
            slope_fast_tau_ms=slope_fast_tau_ms,
            slope_slow_tau_ms=slope_slow_tau_ms,
        )

        self._stable_state: EdgeState = "rest"
        self._stable_start_ms: int | None = None
        self._pending: _Pending | None = None
        self._events: list[EdgeEvent] = []
        self._last_snapshot: FeatureSnapshot | None = None
        self._last_debug: DetectorDebug | None = None

        self._confident_rest = False
        self._rest_conf_start_ms: int | None = None
        self._rest_start_ms: int | None = None
        self._next_press_after_ms = 0
        self._refractory_until_ms = 0

        self._press_start_ms: int | None = None
        self._press_class: EdgeState | None = None
        self._press_peak_u = 0.0

        self._phase = "BOOTSTRAP" if bootstrap_ms > 0 else "RUNNING"
        self._bootstrap_start_ms: int | None = None
        self._bootstrap_samples: list[FeatureSnapshot] = []

        self._artifact_gated = False
        self._artifact_hold_until_ms = 0

    @classmethod
    def from_profile(cls, profile: CalibrationProfile | None) -> "AdaptiveEdgeDetector":
        p = profile or default_profile()
        return cls(
            u_light_enter=p.u_light_enter,
            u_light_exit=p.u_light_exit,
            u_heavy_enter=p.u_heavy_enter,
            u_heavy_exit=p.u_heavy_exit,
            u_rest_conf_threshold=p.u_rest_conf_threshold,
            min_dwell_ms=p.min_dwell_ms,
            min_event_ms=p.min_event_ms,
            min_rest_gap_ms=p.min_rest_gap_ms,
            refractory_ms=p.refractory_ms,
            rest_conf_dwell_ms=p.rest_conf_dwell_ms,
            tau_baseline_ms=p.tau_baseline_ms,
            filter_enabled=p.filter_enabled,
            rest_scale_floor=p.rest_scale_floor,
            rms_state_window_ms=p.rms_state_window_ms,
            rest_stats_window_s=p.rest_stats_window_s,
            bootstrap_ms=p.bootstrap_ms,
            quiet_window_ms=p.quiet_window_ms,
            quiet_fraction=p.quiet_fraction,
            hp_hz=p.hp_hz,
            lp_hz=p.lp_hz,
            notch_hz=p.notch_hz,
            notch_q=p.notch_q,
            artifact_enter=p.artifact_enter,
            artifact_exit=p.artifact_exit,
            artifact_holdoff_ms=p.artifact_holdoff_ms,
            artifact_lf_hz=p.artifact_lf_hz,
            slope_fast_tau_ms=p.slope_fast_tau_ms,
            slope_slow_tau_ms=p.slope_slow_tau_ms,
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
            rms_vals = [s.rms_state for s in chunk]
            center = sum(rms_vals) / len(rms_vals)
            variance = sum((x - center) ** 2 for x in rms_vals) / len(rms_vals)
            windows.append((variance, i, i + len(chunk)))

        windows.sort(key=lambda w: w[0])
        keep = max(1, int(len(windows) * max(min(self.quiet_fraction, 0.8), 0.05)))
        quiet_idx: set[int] = set()
        for _, start, end in windows[:keep]:
            quiet_idx.update(range(start, end))

        quiet = [vals[i] for i in sorted(quiet_idx)] if quiet_idx else vals
        quiet_rms = [s.rms_state for s in quiet]
        quiet_filtered = [s.filtered_raw_hp for s in quiet]

        baseline = median(quiet_filtered) if quiet_filtered else 0.0
        rest_center = median(quiet_rms) if quiet_rms else 0.0
        rest_scale = robust_scale(quiet_rms, self.features.rest_scale_floor)

        self.features.set_bootstrap_state(
            baseline_raw=baseline,
            rest_center=rest_center,
            rest_scale=rest_scale,
            t_ms=t_ms,
            rest_seed=quiet_rms,
        )

        self._stable_state = "rest"
        self._stable_start_ms = t_ms
        self._rest_start_ms = t_ms
        self._pending = None
        self._events = []
        self._press_start_ms = None
        self._press_class = None
        self._press_peak_u = 0.0
        self._confident_rest = False
        self._rest_conf_start_ms = None

    def _reset_rest_confidence(self) -> None:
        self._confident_rest = False
        self._rest_conf_start_ms = None

    def _update_confident_rest(self, sample_t_ms: int, u: float) -> None:
        can_be_confident = (not self._artifact_gated) and (u < self.u_rest_conf_threshold)
        if can_be_confident:
            if self._rest_conf_start_ms is None:
                self._rest_conf_start_ms = sample_t_ms
            self._confident_rest = (sample_t_ms - self._rest_conf_start_ms) >= self.rest_conf_dwell_ms
            return

        self._reset_rest_confidence()

    def _rest_learning_allowed(self) -> bool:
        return (
            self._stable_state == "rest"
            and self._pending is None
            and self._confident_rest
            and not self._artifact_gated
        )

    def _enter_artifact_gate(self, t_ms: int) -> None:
        self._artifact_gated = True
        self._artifact_hold_until_ms = t_ms + self.artifact_holdoff_ms
        self._pending = None
        self._reset_rest_confidence()

    def _update_artifact_gate(self, t_ms: int, score: float) -> None:
        was_gated = self._artifact_gated
        if self._artifact_gated:
            if t_ms < self._artifact_hold_until_ms:
                return
            if score <= self.artifact_exit:
                self._artifact_gated = False
            return

        if score >= self.artifact_enter:
            self._enter_artifact_gate(t_ms)
            if not was_gated:
                self._force_rest_from_gate(t_ms)

    def _force_rest_from_gate(self, t_ms: int) -> None:
        self._pending = None
        self._reset_rest_confidence()
        if self._stable_state in ("light", "heavy"):
            self._stable_state = "rest"
            self._stable_start_ms = t_ms
            self._rest_start_ms = t_ms
            self._press_start_ms = None
            self._press_class = None
            self._press_peak_u = 0.0
            self._refractory_until_ms = t_ms + self.refractory_ms
            self._next_press_after_ms = t_ms + self.refractory_ms

    def _candidate_state(self, sample_t_ms: int, u: float) -> EdgeState:
        if self._artifact_gated:
            return "rest"

        if self._stable_state == "rest":
            if sample_t_ms < self._refractory_until_ms or sample_t_ms < self._next_press_after_ms:
                return "rest"
            if u >= self.u_heavy_enter:
                return "heavy"
            if u >= self.u_light_enter:
                return "light"
            return "rest"

        if self._stable_state == "light":
            if u <= self.u_light_exit:
                return "rest"
            return "light"

        # Heavy hysteresis:
        # - stay heavy until u <= u_heavy_exit
        # - drop to rest only once u <= u_light_exit
        if u <= self.u_light_exit:
            return "rest"
        if u <= self.u_heavy_exit:
            return "light"
        return "heavy"

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

    def _commit_down(self, new_state: EdgeState, t_ms: int, u: float) -> None:
        self._emit_rest_event(t_ms)
        self._stable_state = new_state
        self._stable_start_ms = t_ms
        self._press_start_ms = t_ms
        self._press_class = new_state
        self._press_peak_u = u
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
        looked_real = self._press_peak_u >= self.u_heavy_enter

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
        self._press_peak_u = 0.0
        self._pending = None

    def _commit_transition(self, new_state: EdgeState, t_ms: int, u: float) -> tuple[int, int]:
        if self._artifact_gated:
            self._force_rest_from_gate(t_ms)
            return (0, 0)

        if new_state == self._stable_state:
            self._pending = None
            if self._stable_state in ("light", "heavy"):
                self._press_peak_u = max(self._press_peak_u, u)
            return (0, 0)

        if self._pending is None or self._pending.state != new_state:
            self._pending = _Pending(state=new_state, since_ms=t_ms)
            return (0, 0)

        if self._stable_state == "rest" and new_state in ("light", "heavy"):
            if (t_ms - self._pending.since_ms) < self.min_dwell_ms:
                return (0, 0)
            if t_ms < self._next_press_after_ms:
                self._pending = None
                return (0, 0)
            self._commit_down(new_state, t_ms, u)
            return (1, 0)

        if self._stable_state in ("light", "heavy") and new_state == "rest":
            required_release_ms = max(self.min_dwell_ms, self.rest_conf_dwell_ms)
            if (t_ms - self._pending.since_ms) < required_release_ms:
                return (0, 0)
            if not self._confident_rest:
                return (0, 0)
            self._commit_up(t_ms)
            return (0, 1)

        # In-press state changes (e.g. heavy -> light) do not emit events.
        if self._stable_state in ("light", "heavy") and new_state in ("light", "heavy"):
            if (t_ms - self._pending.since_ms) < self.min_dwell_ms:
                return (0, 0)
            self._stable_state = new_state
            self._stable_start_ms = t_ms
            self._pending = None
            self._press_peak_u = max(self._press_peak_u, u)
            return (0, 0)

        return (0, 0)

    def update(self, sample: EmgSample) -> EdgeState:
        if self._stable_start_ms is None:
            self._stable_start_ms = sample.t_ms
            self._rest_start_ms = sample.t_ms
        if self._bootstrap_start_ms is None:
            self._bootstrap_start_ms = sample.t_ms

        snap = self.features.update(sample)
        self._last_snapshot = snap

        if self._phase == "BOOTSTRAP":
            self._bootstrap_samples.append(snap)
            if (sample.t_ms - self._bootstrap_start_ms) >= self.bootstrap_ms:
                self._bootstrap_init(sample.t_ms)
                self._phase = "ARMING"

            debug_state: EdgeState = "heavy" if snap.u >= self.u_heavy_enter else "light" if snap.u >= self.u_light_enter else "rest"
            self._last_debug = DetectorDebug(
                t_ms=sample.t_ms,
                raw=sample.raw,
                env_in=sample.env,
                filtered_raw_hp=snap.filtered_raw_hp,
                rms_state=snap.rms_state,
                rest_center=snap.rest_center,
                rest_scale=snap.rest_scale,
                u=snap.u,
                lf_energy=snap.lf_energy,
                artifact_ratio=snap.artifact_ratio,
                artifact_score=snap.artifact_score,
                artifact_gated=False,
                rest_confident=False,
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

        self._update_artifact_gate(sample.t_ms, snap.artifact_score)
        if self._artifact_gated:
            self._force_rest_from_gate(sample.t_ms)

        self._update_confident_rest(sample.t_ms, snap.u)
        if self._rest_learning_allowed():
            self.features.apply_rest_learning(snap)
            snap = replace(
                snap,
                baseline_raw=self.features.baseline_raw if self.features.baseline_raw is not None else snap.baseline_raw,
                rest_center=self.features.rest_center,
                rest_scale=self.features.rest_scale,
                u=self.features.u,
                slow=self.features.rest_center,
                a=snap.rms_state - self.features.rest_center,
                sigma=self.features.rest_scale,
                z=self.features.u,
            )
            self._last_snapshot = snap
            self._update_confident_rest(sample.t_ms, snap.u)

        if self._phase == "ARMING":
            self._stable_state = "rest"
            self._pending = None
            if self._confident_rest:
                self._phase = "RUNNING"

            debug_state: EdgeState = "heavy" if snap.u >= self.u_heavy_enter else "light" if snap.u >= self.u_light_enter else "rest"
            self._last_debug = DetectorDebug(
                t_ms=sample.t_ms,
                raw=sample.raw,
                env_in=sample.env,
                filtered_raw_hp=snap.filtered_raw_hp,
                rms_state=snap.rms_state,
                rest_center=snap.rest_center,
                rest_scale=snap.rest_scale,
                u=snap.u,
                lf_energy=snap.lf_energy,
                artifact_ratio=snap.artifact_ratio,
                artifact_score=snap.artifact_score,
                artifact_gated=self._artifact_gated,
                rest_confident=self._confident_rest,
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

        candidate = self._candidate_state(sample.t_ms, snap.u)
        down, up = self._commit_transition(candidate, sample.t_ms, snap.u)

        gated_refractory = sample.t_ms < self._refractory_until_ms
        gated_rest_gap = sample.t_ms < self._next_press_after_ms
        self._last_debug = DetectorDebug(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            filtered_raw_hp=snap.filtered_raw_hp,
            rms_state=snap.rms_state,
            rest_center=snap.rest_center,
            rest_scale=snap.rest_scale,
            u=snap.u,
            lf_energy=snap.lf_energy,
            artifact_ratio=snap.artifact_ratio,
            artifact_score=snap.artifact_score,
            artifact_gated=self._artifact_gated,
            rest_confident=self._confident_rest,
            phase=self._phase,
            armed=self.is_armed(),
            state=self._stable_state,
            state_code={"rest": 0, "light": 1, "heavy": 2}[self._stable_state],
            down=down,
            up=up,
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
