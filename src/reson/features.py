from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
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
        sigma_window_s: float = 3.0,
        sample_hz_hint: int = 250,
        sigma_floor: float = 5.0,
    ):
        self.tau_fast_ms = tau_fast_ms
        self.tau_slow_ms = tau_slow_ms
        self.tau_baseline_ms = tau_baseline_ms
        self.sigma_floor = sigma_floor

        self.sigma_window_ms = max(int(sigma_window_s * 1000.0), 250)
        self.a_rest: deque[tuple[int, float]] = deque(maxlen=max(sample_hz_hint * 12, 128))

        self._last_t_ms: int | None = None
        self.baseline_raw: float | None = None
        self.fast: float = 0.0
        self.slow: float = 0.0
        self.sigma: float = sigma_floor

    def preview_z(self, sample: EmgSample) -> float:
        if self._last_t_ms is None:
            return 0.0
        baseline = self.baseline_raw if self.baseline_raw is not None else float(sample.raw)
        dt_ms = max(sample.t_ms - self._last_t_ms, 1)
        rect = abs(float(sample.raw) - baseline)
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
            self.baseline_raw = _ema_step(self.baseline_raw, float(sample.raw), dt_ms, self.tau_baseline_ms)

        rect = abs(float(sample.raw) - self.baseline_raw)
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
            baseline_raw=self.baseline_raw,
            rect=rect,
            fast=self.fast,
            slow=self.slow,
            a=a,
            sigma=self.sigma,
            z=z,
        )
