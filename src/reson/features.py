from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from statistics import median

from reson.types import EmgSample


def _ema_step(prev: float, value: float, dt_ms: int, tau_ms: float) -> float:
    tau = max(tau_ms, 1.0)
    alpha = max(min(dt_ms / (tau + dt_ms), 1.0), 0.0)
    return prev + alpha * (value - prev)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * p)
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def robust_scale(values: Sequence[float], scale_floor: float) -> float:
    if not values:
        return scale_floor
    vals = list(values)
    p90 = _percentile(vals, 0.90)
    p50 = _percentile(vals, 0.50)
    mad = median([abs(v - p50) for v in vals]) if vals else 0.0
    scale = max(p90 - p50, mad * 1.4826)
    return max(scale, scale_floor)


@dataclass(frozen=True)
class FeatureSnapshot:
    t_ms: int
    raw: int
    env_in: int
    filtered_raw_hp: float
    baseline_raw: float
    rms_state: float
    rest_center: float
    rest_scale: float
    u: float
    lf_energy: float
    artifact_ratio: float
    slope_burst: float
    artifact_score: float

    # Legacy debug compatibility fields (deprecated in v2.4.1).
    filtered_raw: float | None = None
    rect: float | None = None
    fast: float | None = None
    slow: float | None = None
    a: float | None = None
    sigma: float | None = None
    z: float | None = None


class RawFeatureEngine:
    def __init__(
        self,
        tau_baseline_ms: float = 2000.0,
        filter_enabled: bool = True,
        hp_hz: float = 20.0,
        lp_hz: float = 230.0,
        notch_hz: float = 60.0,
        notch_q: float = 20.0,
        rms_state_window_ms: int = 180,
        rest_stats_window_s: float = 3.0,
        sample_hz_hint: int = 250,
        rest_scale_floor: float = 5.0,
        artifact_lf_hz: float = 10.0,
        slope_fast_tau_ms: float = 40.0,
        slope_slow_tau_ms: float = 400.0,
    ):
        self.tau_baseline_ms = tau_baseline_ms
        self.filter_enabled = filter_enabled
        self.hp_hz = hp_hz
        self.lp_hz = lp_hz
        self.notch_hz = notch_hz
        self.notch_q = notch_q
        self.artifact_lf_hz = artifact_lf_hz
        self.slope_fast_tau_ms = slope_fast_tau_ms
        self.slope_slow_tau_ms = slope_slow_tau_ms

        self.rms_state_window_ms = max(rms_state_window_ms, 40)
        self.rest_stats_window_ms = max(int(rest_stats_window_s * 1000.0), 250)
        self.rest_scale_floor = rest_scale_floor

        self._last_t_ms: int | None = None
        self._last_dt_ms = 1
        self._num_updates = 0
        self._fs_est_hz = float(sample_hz_hint)

        # Filter state.
        self._hp_y = 0.0
        self._hp_x_prev = 0.0
        self._lp_y = 0.0
        self._notch_x1 = 0.0
        self._notch_x2 = 0.0
        self._notch_y1 = 0.0
        self._notch_y2 = 0.0
        self._lf_y = 0.0

        # Feature state.
        self.baseline_raw: float | None = None
        self.rest_center: float = 0.0
        self.rest_scale: float = rest_scale_floor
        self.rms_state: float = 0.0
        self.u: float = 0.0
        self.lf_energy: float = 0.0
        self.artifact_ratio: float = 0.0
        self.slope_burst: float = 0.0
        self.artifact_score: float = 0.0

        # Rolling RMS and LF RMS windows.
        self._state_window: deque[tuple[int, float, float]] = deque()
        self._sum_sq_state = 0.0
        self._sum_sq_lf = 0.0

        # REST-only normalization window.
        self._rest_window: deque[tuple[int, float]] = deque(maxlen=max(sample_hz_hint * 12, 128))

        # Slope-burst signal.
        self._last_centered = 0.0
        self._slope_fast = 0.0
        self._slope_slow = 0.0

    def _filter_raw(self, raw_value: float, dt_ms: int) -> float:
        if not self.filter_enabled:
            return raw_value

        dt_s = max(dt_ms / 1000.0, 1e-6)
        fs_hz = 1.0 / dt_s
        self._fs_est_hz = (0.95 * self._fs_est_hz) + (0.05 * fs_hz)

        # First-order high-pass.
        rc_hp = 1.0 / (2.0 * pi * max(self.hp_hz, 0.1))
        alpha_hp = rc_hp / (rc_hp + dt_s)
        hp = alpha_hp * (self._hp_y + raw_value - self._hp_x_prev)
        self._hp_x_prev = raw_value
        self._hp_y = hp

        # Notch biquad (RBJ cookbook).
        omega = 2.0 * pi * min(max(self.notch_hz / max(self._fs_est_hz, 1.0), 0.0), 0.49)
        if omega > 0.0:
            alpha = sin(omega) / (2.0 * max(self.notch_q, 1.0))
            b0 = 1.0
            b1 = -2.0 * cos(omega)
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos(omega)
            a2 = 1.0 - alpha
            notch = (
                (b0 / a0) * hp
                + (b1 / a0) * self._notch_x1
                + (b2 / a0) * self._notch_x2
                - (a1 / a0) * self._notch_y1
                - (a2 / a0) * self._notch_y2
            )
            self._notch_x2 = self._notch_x1
            self._notch_x1 = hp
            self._notch_y2 = self._notch_y1
            self._notch_y1 = notch
        else:
            notch = hp

        # First-order low-pass.
        rc_lp = 1.0 / (2.0 * pi * max(self.lp_hz, 0.1))
        alpha_lp = dt_s / (rc_lp + dt_s)
        self._lp_y = self._lp_y + alpha_lp * (notch - self._lp_y)
        return self._lp_y

    def _update_rms_windows(self, t_ms: int, centered_value: float, dt_ms: int) -> None:
        lf_rc = 1.0 / (2.0 * pi * max(self.artifact_lf_hz, 0.1))
        alpha_lf = (dt_ms / 1000.0) / (lf_rc + (dt_ms / 1000.0))
        self._lf_y = self._lf_y + alpha_lf * (centered_value - self._lf_y)

        state_sq = centered_value * centered_value
        lf_sq = self._lf_y * self._lf_y
        self._state_window.append((t_ms, state_sq, lf_sq))
        self._sum_sq_state += state_sq
        self._sum_sq_lf += lf_sq

        min_t = t_ms - self.rms_state_window_ms
        while self._state_window and self._state_window[0][0] < min_t:
            _, old_state_sq, old_lf_sq = self._state_window.popleft()
            self._sum_sq_state -= old_state_sq
            self._sum_sq_lf -= old_lf_sq

        n = max(len(self._state_window), 1)
        self.rms_state = sqrt(max(self._sum_sq_state / n, 0.0))
        self.lf_energy = sqrt(max(self._sum_sq_lf / n, 0.0))

    def update_rest_stats(self, t_ms: int, rms_state: float) -> None:
        self._rest_window.append((t_ms, rms_state))
        min_t = t_ms - self.rest_stats_window_ms
        while self._rest_window and self._rest_window[0][0] < min_t:
            self._rest_window.popleft()

        values = [v for _, v in self._rest_window]
        if not values:
            return

        self.rest_center = median(values)
        self.rest_scale = robust_scale(values, self.rest_scale_floor)

    def set_bootstrap_state(
        self,
        baseline_raw: float,
        rest_center: float,
        rest_scale: float,
        t_ms: int,
        rest_seed: list[float],
    ) -> None:
        self.baseline_raw = baseline_raw
        self.rest_center = rest_center
        self.rest_scale = max(rest_scale, self.rest_scale_floor)
        self._rest_window.clear()
        seed = rest_seed[-128:]
        for idx, value in enumerate(seed):
            self._rest_window.append((t_ms - (len(seed) - idx), value))

    def apply_rest_learning(self, snapshot: FeatureSnapshot) -> None:
        if self.baseline_raw is None:
            self.baseline_raw = snapshot.filtered_raw_hp
        self.baseline_raw = _ema_step(self.baseline_raw, snapshot.filtered_raw_hp, self._last_dt_ms, self.tau_baseline_ms)
        self.update_rest_stats(snapshot.t_ms, snapshot.rms_state)
        self.u = (snapshot.rms_state - self.rest_center) / max(self.rest_scale, self.rest_scale_floor)

    def update(self, sample: EmgSample) -> FeatureSnapshot:
        dt_ms = 1
        if self._last_t_ms is None:
            self._last_t_ms = sample.t_ms
            self.baseline_raw = float(sample.raw)
            self.rest_center = 0.0
            self.rest_scale = self.rest_scale_floor
            filtered_hp = float(sample.raw)
            centered = 0.0
            self._update_rms_windows(sample.t_ms, centered, dt_ms=1)
        else:
            dt_ms = max(sample.t_ms - self._last_t_ms, 1)
            self._last_dt_ms = dt_ms
            self._last_t_ms = sample.t_ms
            filtered_hp = self._filter_raw(float(sample.raw), dt_ms)

            if self.baseline_raw is None:
                self.baseline_raw = filtered_hp

            centered = filtered_hp - self.baseline_raw
            self._update_rms_windows(sample.t_ms, centered, dt_ms)

        eps = 1e-6
        self.artifact_ratio = self.lf_energy / max(self.rms_state, self.rest_scale_floor, eps)

        slope = abs(centered - self._last_centered)
        self._last_centered = centered
        self._slope_fast = _ema_step(self._slope_fast, slope, dt_ms, self.slope_fast_tau_ms)
        self._slope_slow = _ema_step(self._slope_slow, slope, dt_ms, self.slope_slow_tau_ms)
        self.slope_burst = self._slope_fast / max(self._slope_slow, 1.0)
        self._num_updates += 1
        slope_term = self.slope_burst if self._num_updates >= 5 else 1.0
        self.artifact_score = max(self.artifact_ratio, slope_term)

        denom = max(self.rest_scale, self.rest_scale_floor)
        self.u = (self.rms_state - self.rest_center) / denom

        return FeatureSnapshot(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            filtered_raw_hp=filtered_hp,
            baseline_raw=self.baseline_raw if self.baseline_raw is not None else 0.0,
            rms_state=self.rms_state,
            rest_center=self.rest_center,
            rest_scale=self.rest_scale,
            u=self.u,
            lf_energy=self.lf_energy,
            artifact_ratio=self.artifact_ratio,
            slope_burst=self.slope_burst,
            artifact_score=self.artifact_score,
            filtered_raw=filtered_hp,
            rect=self.rms_state,
            fast=self.rms_state,
            slow=self.rest_center,
            a=self.rms_state - self.rest_center,
            sigma=self.rest_scale,
            z=self.u,
        )
