from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log
from statistics import median
from typing import Protocol

from reson.calibration import CalibrationProfile, Hmm3Model, default_profile
from reson.features import (
    FeatureFrame,
    FeatureFrameExtractor,
    FeatureSnapshot,
    RawFeatureEngine,
    compute_feature_hash,
    robust_scale,
)
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


@dataclass(frozen=True)
class PressSegmentStats:
    start_ms: int
    last_ms: int
    duration_ms: int
    peak_u: float
    auc_u: float
    mean_u: float

    def update(self, t_ms: int, u: float) -> "PressSegmentStats":
        dt = max(t_ms - self.last_ms, 0)
        auc = self.auc_u + (max(u, 0.0) * dt)
        dur = max(t_ms - self.start_ms, 0)
        peak = max(self.peak_u, u)
        mean_u = auc / max(dur, 1)
        return PressSegmentStats(
            start_ms=self.start_ms,
            last_ms=t_ms,
            duration_ms=dur,
            peak_u=peak,
            auc_u=auc,
            mean_u=mean_u,
        )


@dataclass(frozen=True)
class Hmm3DetectorDebug:
    t_ms: int
    raw: int
    env_in: int
    filtered_raw_hp: float
    rms_state: float
    lf_energy_ratio: float
    slope_burst: float
    waveform_length: float
    p_rest: float
    p_press: float
    p_artifact: float
    decoded_state: str
    phase: str
    armed: bool
    artifact_gated: bool
    down: int
    up: int
    press_class: EdgeState | None
    segment_duration_ms: int
    segment_peak_u: float
    segment_auc: float
    segment_mean_u: float
    segment_class: EdgeState | None


@dataclass
class _HmmPending:
    state: str
    since_frame: int
    since_ms: int


class Hmm3EdgeDetector:
    _STATE_ORDER = ("REST", "PRESS", "ARTIFACT")

    def __init__(self, profile: CalibrationProfile | None = None):
        self.profile = profile or default_profile()
        self.model = Hmm3Model.from_profile(self.profile)
        self.feature_order = list(self.model.feature_config.get("feature_order", []))
        expected_hash = compute_feature_hash(self.feature_order)
        if expected_hash != self.model.feature_hash:
            raise ValueError(
                "hmm3 feature hash mismatch: profile does not match runtime feature ordering"
            )

        self.extractor = FeatureFrameExtractor(
            window_ms=int(self.model.feature_config.get("window_ms", 120)),
            hop_ms=int(self.model.feature_config.get("hop_ms", 30)),
            tau_baseline_ms=self.profile.tau_baseline_ms,
            filter_enabled=self.profile.filter_enabled,
            hp_hz=self.profile.hp_hz,
            lp_hz=self.profile.lp_hz,
            notch_hz=self.profile.notch_hz,
            notch_q=self.profile.notch_q,
            rest_scale_floor=self.profile.rest_scale_floor,
            artifact_lf_hz=self.profile.artifact_lf_hz,
            slope_fast_tau_ms=self.profile.slope_fast_tau_ms,
            slope_slow_tau_ms=self.profile.slope_slow_tau_ms,
        )

        norm = self.model.normalization
        self._norm_center = dict(norm.get("center", {}))
        self._norm_scale = dict(norm.get("scale", {}))
        self._norm_floor = dict(norm.get("floor", {}))
        self._drift_cap = dict(norm.get("drift_cap_per_min", {}))
        self._last_adapt_t_ms: int | None = None

        classifier = self.model.classifier
        self._classes = list(classifier.get("classes", self._STATE_ORDER))
        self._weights = [list(row) for row in classifier.get("weights", [])]
        self._bias = list(classifier.get("bias", []))
        if len(self._weights) != len(self._classes) or len(self._bias) != len(self._classes):
            raise ValueError("hmm3 classifier dimensions are invalid")

        hmm = self.model.hmm
        self._states = list(hmm.get("states", self._STATE_ORDER))
        self._state_to_idx = {s: i for i, s in enumerate(self._states)}
        if any(s not in self._state_to_idx for s in self._STATE_ORDER):
            raise ValueError("hmm3 states must include REST/PRESS/ARTIFACT")
        self._start_logp = list(hmm.get("start_logp", []))
        self._trans_logp = [list(row) for row in hmm.get("transition_logp_final", [])]
        if len(self._start_logp) != len(self._states) or len(self._trans_logp) != len(self._states):
            raise ValueError("hmm3 start/transition matrix dimensions are invalid")
        self._lag_frames = int(hmm.get("lag_frames", 4))

        gates = self.model.decision_gates
        self._enter_dwell_frames = int(gates.get("enter_dwell_frames", 3))
        self._release_dwell_frames = int(gates.get("release_dwell_frames", 3))
        self.min_event_ms = int(gates.get("min_event_ms", self.profile.min_event_ms))
        self.refractory_ms = int(gates.get("refractory_ms", self.profile.refractory_ms))
        self.min_rest_gap_ms = int(gates.get("min_rest_gap_ms", self.profile.min_rest_gap_ms))
        self._rest_conf_frames_min = int(gates.get("rest_conf_frames_min", 3))

        seg = self.model.segment_thresholds
        self._dur_heavy_ms = float(seg.get("dur_heavy_ms", 220.0))
        self._peak_heavy_u = float(seg.get("peak_heavy_u", 2.0))
        self._auc_heavy = float(seg.get("auc_heavy", 260.0))
        self._mean_heavy_u = float(seg.get("mean_heavy_u", 1.3))

        self._phase = "BOOTSTRAP" if self.profile.bootstrap_ms > 0 else "RUNNING"
        self._bootstrap_start_ms: int | None = None
        self._bootstrap_frames: list[FeatureFrame] = []

        self._decoded_stable = "REST"
        self._decoded_pending: _HmmPending | None = None
        self._final_frame_index = -1
        self._rest_conf_frames = 0
        self._artifact_gated = False
        self._await_rearm = False

        self._press_segment: PressSegmentStats | None = None
        self._press_class_latched: EdgeState | None = None

        self._rest_start_ms: int | None = None
        self._next_press_after_ms = 0
        self._refractory_until_ms = 0

        self._events: list[EdgeEvent] = []
        self._last_debug: Hmm3DetectorDebug | None = None

        self._dp_hist: list[list[float]] = []
        self._bp_hist: list[list[int]] = []
        self._frame_hist: list[FeatureFrame] = []
        self._prob_hist: list[list[float]] = []
        self._decoded_upto = -1

    @classmethod
    def from_profile(cls, profile: CalibrationProfile | None) -> "Hmm3EdgeDetector":
        return cls(profile=profile)

    def phase(self) -> str:
        return self._phase

    def is_armed(self) -> bool:
        return self._phase == "RUNNING"

    def _normalize(self, frame: FeatureFrame) -> tuple[list[float], float]:
        values = {
            "rms_state": frame.rms_state,
            "lf_energy_ratio": frame.lf_energy_ratio,
            "slope_burst": frame.slope_burst,
            "waveform_length": frame.waveform_length,
        }
        vec: list[float] = []
        for key in self.feature_order:
            value = float(values[key])
            center = float(self._norm_center.get(key, 0.0))
            scale = float(self._norm_scale.get(key, 1.0))
            floor = float(self._norm_floor.get(key, 1.0))
            vec.append((value - center) / max(scale, floor))
        u_idx = self.feature_order.index("rms_state") if "rms_state" in self.feature_order else 0
        return vec, vec[u_idx]

    def _emission(self, vec: list[float]) -> tuple[list[float], list[float]]:
        scores: list[float] = []
        for cls_idx in range(len(self._classes)):
            dot = sum(w * x for w, x in zip(self._weights[cls_idx], vec))
            scores.append(dot + self._bias[cls_idx])
        m = max(scores)
        exps = [exp(s - m) for s in scores]
        denom = max(sum(exps), 1e-9)
        probs = [v / denom for v in exps]
        logp = [log(max(p, 1e-9)) for p in probs]
        return logp, probs

    def _decode_frame(self, frame: FeatureFrame, emission_logp: list[float], probs: list[float]) -> tuple[int | None, str | None]:
        n_states = len(self._states)
        if not self._dp_hist:
            dp = [self._start_logp[i] + emission_logp[i] for i in range(n_states)]
            bp = [0 for _ in range(n_states)]
        else:
            prev_dp = self._dp_hist[-1]
            dp = [0.0 for _ in range(n_states)]
            bp = [0 for _ in range(n_states)]
            for j in range(n_states):
                candidates = [prev_dp[i] + self._trans_logp[i][j] for i in range(n_states)]
                arg = max(range(n_states), key=lambda i: candidates[i])
                dp[j] = candidates[arg] + emission_logp[j]
                bp[j] = arg

        self._dp_hist.append(dp)
        self._bp_hist.append(bp)
        self._frame_hist.append(frame)
        self._prob_hist.append(probs)

        if len(self._dp_hist) > 512:
            trim = 256
            self._dp_hist = self._dp_hist[trim:]
            self._bp_hist = self._bp_hist[trim:]
            self._frame_hist = self._frame_hist[trim:]
            self._prob_hist = self._prob_hist[trim:]
            self._decoded_upto -= trim

        if len(self._dp_hist) <= self._lag_frames:
            return None, None

        target = len(self._dp_hist) - self._lag_frames - 1
        if target <= self._decoded_upto:
            return None, None

        state = max(range(n_states), key=lambda i: self._dp_hist[-1][i])
        for idx in range(len(self._bp_hist) - 1, target, -1):
            state = self._bp_hist[idx][state]

        self._decoded_upto = target
        return target, self._states[state]

    def _classify_segment(self, seg: PressSegmentStats) -> EdgeState:
        heavy_votes = 0
        heavy_votes += int(seg.duration_ms >= self._dur_heavy_ms)
        heavy_votes += int(seg.peak_u >= self._peak_heavy_u)
        heavy_votes += int(seg.auc_u >= self._auc_heavy)
        heavy_votes += int(seg.mean_u >= self._mean_heavy_u)
        return "heavy" if heavy_votes >= 2 else "light"

    def _update_rest_conf(self) -> None:
        if self._decoded_stable == "REST" and self._decoded_pending is None and not self._artifact_gated:
            self._rest_conf_frames += 1
        else:
            self._rest_conf_frames = 0
        if self._await_rearm and self._rest_conf_frames >= self._rest_conf_frames_min:
            self._await_rearm = False

    def _maybe_adapt_normalization(self, frame: FeatureFrame) -> None:
        if not (
            self._decoded_stable == "REST"
            and self._decoded_pending is None
            and not self._artifact_gated
            and self._rest_conf_frames >= self._rest_conf_frames_min
        ):
            return

        values = {
            "rms_state": frame.rms_state,
            "lf_energy_ratio": frame.lf_energy_ratio,
            "slope_burst": frame.slope_burst,
            "waveform_length": frame.waveform_length,
        }
        alpha = 0.02
        if self._last_adapt_t_ms is None:
            dt_min = 1.0 / 60.0
        else:
            dt_min = max((frame.t_ms - self._last_adapt_t_ms) / 60000.0, 1.0 / 60000.0)
        self._last_adapt_t_ms = frame.t_ms

        for key in self.feature_order:
            value = float(values[key])
            center = float(self._norm_center.get(key, 0.0))
            scale = float(self._norm_scale.get(key, 1.0))
            floor = float(self._norm_floor.get(key, 1.0))
            drift_cap = float(self._drift_cap.get(key, 0.5))
            max_delta = drift_cap * dt_min

            center_target = center + (alpha * (value - center))
            center_delta = max(min(center_target - center, max_delta), -max_delta)
            center_new = center + center_delta

            spread_target = abs(value - center_new)
            scale_target = scale + (alpha * (spread_target - scale))
            scale_delta = max(min(scale_target - scale, max_delta), -max_delta)
            scale_new = max(scale + scale_delta, floor)

            self._norm_center[key] = center_new
            self._norm_scale[key] = scale_new

    def _emit_rest_event(self, end_ms: int) -> None:
        if self._rest_start_ms is None:
            self._rest_start_ms = end_ms
            return
        dur = max(end_ms - self._rest_start_ms, 0)
        if dur <= 0:
            return
        self._events.append(
            EdgeEvent(state="rest", start_ms=self._rest_start_ms, end_ms=end_ms, duration_ms=dur)
        )

    def _start_press(self, t_ms: int, u: float) -> tuple[int, int]:
        self._emit_rest_event(t_ms)
        seg = PressSegmentStats(
            start_ms=t_ms,
            last_ms=t_ms,
            duration_ms=0,
            peak_u=max(u, 0.0),
            auc_u=0.0,
            mean_u=max(u, 0.0),
        )
        self._press_segment = seg
        cls = self._classify_segment(seg)
        self._press_class_latched = cls
        self._events.append(
            EdgeEvent(
                state=cls,
                start_ms=t_ms,
                end_ms=t_ms,
                duration_ms=0,
                phase="down",
                press_class=cls,
            )
        )
        return 1, 0

    def _cancel_press_for_artifact(self, t_ms: int) -> None:
        self._press_segment = None
        self._press_class_latched = None
        self._refractory_until_ms = t_ms + self.refractory_ms
        self._next_press_after_ms = t_ms + self.refractory_ms
        self._await_rearm = True

    def _finish_press(self, t_ms: int) -> tuple[int, int]:
        seg = self._press_segment
        if seg is None:
            self._rest_start_ms = t_ms
            return 0, 0

        cls = self._classify_segment(seg)
        looked_real = seg.peak_u >= self._peak_heavy_u or seg.mean_u >= self._mean_heavy_u
        if seg.duration_ms >= self.min_event_ms:
            self._events.append(
                EdgeEvent(
                    state=cls,
                    start_ms=seg.start_ms,
                    end_ms=t_ms,
                    duration_ms=seg.duration_ms,
                    phase="up",
                    press_class=cls,
                )
            )
            self._next_press_after_ms = t_ms + self.refractory_ms + self.min_rest_gap_ms
            up = 1
        else:
            self._next_press_after_ms = t_ms + self.refractory_ms + (self.min_rest_gap_ms if looked_real else 0)
            up = 0

        self._refractory_until_ms = t_ms + self.refractory_ms
        self._press_segment = None
        self._press_class_latched = None
        self._rest_start_ms = t_ms
        return 0, up

    def _bootstrap_init(self, t_ms: int) -> None:
        if not self._bootstrap_frames:
            return
        keep = max(1, int(len(self._bootstrap_frames) * max(min(self.profile.quiet_fraction, 0.8), 0.05)))
        quiet = sorted(self._bootstrap_frames, key=lambda f: f.rms_state)[:keep]

        by_feature: dict[str, list[float]] = {k: [] for k in self.feature_order}
        for frame in quiet:
            vals = {
                "rms_state": frame.rms_state,
                "lf_energy_ratio": frame.lf_energy_ratio,
                "slope_burst": frame.slope_burst,
                "waveform_length": frame.waveform_length,
            }
            for key in self.feature_order:
                by_feature[key].append(float(vals[key]))

        for key in self.feature_order:
            vals = by_feature[key]
            if not vals:
                continue
            center = median(vals)
            floor = float(self._norm_floor.get(key, 1.0))
            scale = robust_scale(vals, floor)
            self._norm_center[key] = center
            self._norm_scale[key] = scale

        self._phase = "ARMING"
        self._decoded_stable = "REST"
        self._decoded_pending = None
        self._rest_start_ms = t_ms
        self._rest_conf_frames = 0
        self._await_rearm = False

    def _step_state_machine(self, decoded: str, frame: FeatureFrame, frame_u: float) -> tuple[int, int]:
        down = 0
        up = 0

        if self._press_segment is not None and self._decoded_stable == "PRESS":
            self._press_segment = self._press_segment.update(frame.t_ms, frame_u)

        if decoded == self._decoded_stable:
            self._decoded_pending = None
            return down, up

        if self._decoded_pending is None or self._decoded_pending.state != decoded:
            self._decoded_pending = _HmmPending(decoded, self._final_frame_index, frame.t_ms)
            return down, up

        if self._decoded_stable == "REST" and decoded == "PRESS":
            dwell = self._enter_dwell_frames
        elif self._decoded_stable == "PRESS" and decoded == "REST":
            dwell = self._release_dwell_frames
        else:
            dwell = 1

        if (self._final_frame_index - self._decoded_pending.since_frame + 1) < dwell:
            return down, up

        prev = self._decoded_stable
        self._decoded_stable = decoded
        self._decoded_pending = None

        if prev == "REST" and decoded == "PRESS":
            if self._artifact_gated or frame.t_ms < self._refractory_until_ms or frame.t_ms < self._next_press_after_ms:
                self._decoded_stable = "REST"
                return down, up
            if self._await_rearm and self._rest_conf_frames < self._rest_conf_frames_min:
                self._decoded_stable = "REST"
                return down, up
            down, up = self._start_press(frame.t_ms, frame_u)
            return down, up

        if prev == "PRESS" and decoded == "REST":
            down, up = self._finish_press(frame.t_ms)
            return down, up

        if decoded == "ARTIFACT":
            self._artifact_gated = True
            if prev == "PRESS":
                self._cancel_press_for_artifact(frame.t_ms)
            return down, up

        if prev == "ARTIFACT" and decoded == "REST":
            self._artifact_gated = False
            self._rest_start_ms = frame.t_ms
            return down, up

        return down, up

    def _output_state(self) -> EdgeState:
        if self._artifact_gated or self._decoded_stable != "PRESS":
            return "rest"
        if self._press_class_latched is not None:
            return self._press_class_latched
        if self._press_segment is None:
            return "light"
        return self._classify_segment(self._press_segment)

    def _reset_after_stream_restart(self, t_ms: int) -> None:
        # Keep learned profile parameters, but reset runtime decode/segment state.
        self._phase = "ARMING"
        self._bootstrap_start_ms = None
        self._bootstrap_frames = []
        self._decoded_stable = "REST"
        self._decoded_pending = None
        self._final_frame_index = -1
        self._rest_conf_frames = 0
        self._artifact_gated = False
        self._await_rearm = False
        self._press_segment = None
        self._press_class_latched = None
        self._rest_start_ms = t_ms
        self._next_press_after_ms = t_ms
        self._refractory_until_ms = t_ms
        self._events = []
        self._last_debug = None
        self._dp_hist = []
        self._bp_hist = []
        self._frame_hist = []
        self._prob_hist = []
        self._decoded_upto = -1

    def update(self, sample: EmgSample) -> EdgeState:
        snap, frames = self.extractor.update(sample)
        if self.extractor.reset_detected:
            self._reset_after_stream_restart(sample.t_ms)

        if self._rest_start_ms is None:
            self._rest_start_ms = sample.t_ms

        for frame in frames:
            if self._bootstrap_start_ms is None:
                self._bootstrap_start_ms = frame.t_ms

            if self._phase == "BOOTSTRAP":
                self._bootstrap_frames.append(frame)
                if self._bootstrap_start_ms is not None and (frame.t_ms - self._bootstrap_start_ms) >= self.profile.bootstrap_ms:
                    self._bootstrap_init(frame.t_ms)

                self._last_debug = Hmm3DetectorDebug(
                    t_ms=frame.t_ms,
                    raw=sample.raw,
                    env_in=frame.env_in,
                    filtered_raw_hp=frame.filtered_raw_hp,
                    rms_state=frame.rms_state,
                    lf_energy_ratio=frame.lf_energy_ratio,
                    slope_burst=frame.slope_burst,
                    waveform_length=frame.waveform_length,
                    p_rest=0.0,
                    p_press=0.0,
                    p_artifact=0.0,
                    decoded_state="BOOTSTRAP",
                    phase=self._phase,
                    armed=False,
                    artifact_gated=False,
                    down=0,
                    up=0,
                    press_class=None,
                    segment_duration_ms=0,
                    segment_peak_u=0.0,
                    segment_auc=0.0,
                    segment_mean_u=0.0,
                    segment_class=None,
                )
                continue

            vec, frame_u = self._normalize(frame)
            emission_logp, probs = self._emission(vec)
            target_idx, decoded = self._decode_frame(frame, emission_logp, probs)
            if decoded is None or target_idx is None:
                continue

            final_frame = self._frame_hist[target_idx]
            final_probs = self._prob_hist[target_idx]
            final_vec, final_u = self._normalize(final_frame)
            _ = final_vec

            self._final_frame_index += 1
            if self._phase == "ARMING":
                self._decoded_stable = "REST"
                self._decoded_pending = None
                if decoded == "REST" and not self._artifact_gated:
                    self._rest_conf_frames += 1
                else:
                    self._rest_conf_frames = 0
                if self._rest_conf_frames >= self._rest_conf_frames_min:
                    self._phase = "RUNNING"
                    self._rest_start_ms = final_frame.t_ms

                self._last_debug = Hmm3DetectorDebug(
                    t_ms=final_frame.t_ms,
                    raw=sample.raw,
                    env_in=final_frame.env_in,
                    filtered_raw_hp=final_frame.filtered_raw_hp,
                    rms_state=final_frame.rms_state,
                    lf_energy_ratio=final_frame.lf_energy_ratio,
                    slope_burst=final_frame.slope_burst,
                    waveform_length=final_frame.waveform_length,
                    p_rest=final_probs[self._state_to_idx["REST"]],
                    p_press=final_probs[self._state_to_idx["PRESS"]],
                    p_artifact=final_probs[self._state_to_idx["ARTIFACT"]],
                    decoded_state=decoded,
                    phase=self._phase,
                    armed=self.is_armed(),
                    artifact_gated=self._artifact_gated,
                    down=0,
                    up=0,
                    press_class=self._press_class_latched,
                    segment_duration_ms=self._press_segment.duration_ms if self._press_segment else 0,
                    segment_peak_u=self._press_segment.peak_u if self._press_segment else 0.0,
                    segment_auc=self._press_segment.auc_u if self._press_segment else 0.0,
                    segment_mean_u=self._press_segment.mean_u if self._press_segment else 0.0,
                    segment_class=(self._classify_segment(self._press_segment) if self._press_segment else None),
                )
                continue

            down, up = self._step_state_machine(decoded, final_frame, final_u)
            self._update_rest_conf()
            self._maybe_adapt_normalization(final_frame)

            self._last_debug = Hmm3DetectorDebug(
                t_ms=final_frame.t_ms,
                raw=sample.raw,
                env_in=final_frame.env_in,
                filtered_raw_hp=final_frame.filtered_raw_hp,
                rms_state=final_frame.rms_state,
                lf_energy_ratio=final_frame.lf_energy_ratio,
                slope_burst=final_frame.slope_burst,
                waveform_length=final_frame.waveform_length,
                p_rest=final_probs[self._state_to_idx["REST"]],
                p_press=final_probs[self._state_to_idx["PRESS"]],
                p_artifact=final_probs[self._state_to_idx["ARTIFACT"]],
                decoded_state=self._decoded_stable,
                phase=self._phase,
                armed=self.is_armed(),
                artifact_gated=self._artifact_gated,
                down=down,
                up=up,
                press_class=self._press_class_latched,
                segment_duration_ms=self._press_segment.duration_ms if self._press_segment else 0,
                segment_peak_u=self._press_segment.peak_u if self._press_segment else 0.0,
                segment_auc=self._press_segment.auc_u if self._press_segment else 0.0,
                segment_mean_u=self._press_segment.mean_u if self._press_segment else 0.0,
                segment_class=(self._classify_segment(self._press_segment) if self._press_segment else None),
            )

        return self._output_state()

    def pop_events(self) -> list[EdgeEvent]:
        out = self._events
        self._events = []
        return out

    def flush(self, final_t_ms: int) -> list[EdgeEvent]:
        out = self.pop_events()
        if self._press_segment is not None:
            seg = self._press_segment.update(final_t_ms, self._press_segment.mean_u)
            if seg.duration_ms >= self.min_event_ms:
                cls = self._classify_segment(seg)
                out.append(
                    EdgeEvent(
                        state=cls,
                        start_ms=seg.start_ms,
                        end_ms=final_t_ms,
                        duration_ms=seg.duration_ms,
                        phase="up",
                        press_class=cls,
                    )
                )
            return out

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

    def last_debug(self) -> Hmm3DetectorDebug | None:
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
    if mode == "hmm3":
        return Hmm3EdgeDetector.from_profile(profile)
    return AdaptiveEdgeDetector.from_profile(profile)
