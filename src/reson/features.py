from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin
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


def robust_scale(values: Sequence[float], sigma_floor: float) -> float:
    if not values:
        return sigma_floor
    vals = list(values)
    p95 = _percentile(vals, 0.95)
    p50 = _percentile(vals, 0.50)
    mad = median([abs(v - p50) for v in vals]) if vals else 0.0
    scale = max(p95 - p50, mad * 1.4826)
    return max(scale, sigma_floor)


@dataclass(frozen=True)
class FeatureSnapshot:
    t_ms: int
    raw: int
    env_in: int
    filtered_raw: float
    baseline_raw: float
    rect: float
    fast: float
    slow: float
    a: float
    sigma: float
    z: float


class RawFeatureEngine:
    def __init__(
        self,
        tau_fast_ms: float = 35.0,
        tau_slow_ms: float = 1000.0,
        tau_baseline_ms: float = 2000.0,
        filter_enabled: bool = True,
        hp_hz: float = 20.0,
        lp_hz: float = 230.0,
        notch_hz: float = 60.0,
        notch_q: float = 20.0,
        sigma_window_s: float = 3.0,
        sample_hz_hint: int = 250,
        sigma_floor: float = 5.0,
    ):
        self.tau_fast_ms = tau_fast_ms
        self.tau_slow_ms = tau_slow_ms
        self.tau_baseline_ms = tau_baseline_ms
        self.filter_enabled = filter_enabled
        self.hp_hz = hp_hz
        self.lp_hz = lp_hz
        self.notch_hz = notch_hz
        self.notch_q = notch_q
        self.sigma_floor = sigma_floor

        self.sigma_window_ms = max(int(sigma_window_s * 1000.0), 250)
        self.a_rest: deque[tuple[int, float]] = deque(maxlen=max(sample_hz_hint * 12, 128))

        self._last_t_ms: int | None = None
        self._fs_est_hz = float(sample_hz_hint)
        self._hp_y = 0.0
        self._hp_x_prev = 0.0
        self._lp_y = 0.0
        self._notch_x1 = 0.0
        self._notch_x2 = 0.0
        self._notch_y1 = 0.0
        self._notch_y2 = 0.0
        self.baseline_raw: float | None = None
        self.fast: float = 0.0
        self.slow: float = 0.0
        self.sigma: float = sigma_floor

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

        # Notch biquad (RBJ cookbook). Falls back gracefully on invalid params.
        omega = 2.0 * pi * min(max(self.notch_hz / max(self._fs_est_hz, 1.0), 0.0), 0.49)
        if omega <= 0.0:
            notch = hp
        else:
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

        # First-order low-pass.
        rc_lp = 1.0 / (2.0 * pi * max(self.lp_hz, 0.1))
        alpha_lp = dt_s / (rc_lp + dt_s)
        self._lp_y = self._lp_y + alpha_lp * (notch - self._lp_y)
        return self._lp_y

    def preview_z(self, sample: EmgSample) -> float:
        if self._last_t_ms is None:
            return 0.0
        baseline = self.baseline_raw if self.baseline_raw is not None else float(sample.raw)
        dt_ms = max(sample.t_ms - self._last_t_ms, 1)
        # Non-mutating preview gate. Use raw proxy to avoid advancing filter state.
        filtered = float(sample.raw)
        rect = abs(filtered - baseline)
        fast_next = _ema_step(self.fast, rect, dt_ms, self.tau_fast_ms)
        a_next = fast_next - self.slow
        return a_next / max(self.sigma, self.sigma_floor)

    def update(self, sample: EmgSample, rest_adapt: bool) -> FeatureSnapshot:
        if self._last_t_ms is None:
            self._last_t_ms = sample.t_ms
            self.baseline_raw = float(sample.raw)
            rect = 0.0
            a = 0.0
            z = 0.0
            return FeatureSnapshot(
                t_ms=sample.t_ms,
                raw=sample.raw,
                env_in=sample.env,
                filtered_raw=float(sample.raw),
                baseline_raw=self.baseline_raw,
                rect=rect,
                fast=self.fast,
                slow=self.slow,
                a=a,
                sigma=self.sigma,
                z=z,
            )

        dt_ms = max(sample.t_ms - self._last_t_ms, 1)
        self._last_t_ms = sample.t_ms

        if self.baseline_raw is None:
            self.baseline_raw = float(sample.raw)

        if rest_adapt:
            filtered_for_baseline = self._filter_raw(float(sample.raw), dt_ms)
            self.baseline_raw = _ema_step(self.baseline_raw, filtered_for_baseline, dt_ms, self.tau_baseline_ms)
            filtered = filtered_for_baseline
        else:
            filtered = self._filter_raw(float(sample.raw), dt_ms)

        rect = abs(filtered - self.baseline_raw)
        self.fast = _ema_step(self.fast, rect, dt_ms, self.tau_fast_ms)
        if rest_adapt:
            self.slow = _ema_step(self.slow, rect, dt_ms, self.tau_slow_ms)

        a = self.fast - self.slow

        if rest_adapt:
            self.a_rest.append((sample.t_ms, a))
            min_t = sample.t_ms - self.sigma_window_ms
            while self.a_rest and self.a_rest[0][0] < min_t:
                self.a_rest.popleft()
            self.sigma = robust_scale([v for _, v in self.a_rest], self.sigma_floor)

        z = a / max(self.sigma, self.sigma_floor)
        return FeatureSnapshot(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            filtered_raw=filtered,
            baseline_raw=self.baseline_raw,
            rect=rect,
            fast=self.fast,
            slow=self.slow,
            a=a,
            sigma=self.sigma,
            z=z,
        )

    def set_bootstrap_state(self, baseline_raw: float, slow: float, sigma: float, t_ms: int, a_seed: list[float]) -> None:
        self.baseline_raw = baseline_raw
        self.slow = slow
        self.sigma = max(sigma, self.sigma_floor)
        self.a_rest.clear()
        for idx, a in enumerate(a_seed):
            self.a_rest.append((t_ms - (len(a_seed) - idx), a))
