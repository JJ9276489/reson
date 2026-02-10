from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median


PROFILE_PATH = Path(".reson_profile.json")


@dataclass(frozen=True)
class CalibrationProfile:
    # Legacy threshold fields retained for compatibility.
    rest_max: float = 10.0
    light_threshold: float = 20.0
    heavy_threshold: float = 40.0
    hysteresis_margin: float = 2.0

    # Adaptive detector tuning defaults.
    t_low_enter: float = 2.4
    t_low_exit: float = 1.7
    t_high_enter: float = 4.2
    t_high_exit: float = 3.2
    min_dwell_ms: int = 90
    min_event_ms: int = 60
    min_rest_gap_ms: int = 140
    refractory_ms: int = 90
    rest_conf_dwell_ms: int = 80
    k_rest: float = 0.8
    tau_fast_ms: float = 35.0
    tau_slow_ms: float = 1000.0
    tau_baseline_ms: float = 2000.0
    filter_enabled: bool = True
    sigma_floor: float = 5.0
    sigma_window_s: float = 3.0
    bootstrap_ms: int = 3000
    quiet_window_ms: int = 80
    quiet_fraction: float = 0.25
    hp_hz: float = 20.0
    lp_hz: float = 230.0
    notch_hz: float = 60.0
    notch_q: float = 20.0
    separation_ok: bool = True
    profile_version: int = 2


class CalibrationError(RuntimeError):
    pass


def _percentile(values: list[int], p: float) -> float:
    if not values:
        raise CalibrationError("Calibration sample set is empty")
    idx = int((len(values) - 1) * p)
    sorted_vals = sorted(values)
    return float(sorted_vals[idx])


def build_profile(rest_env: list[int], light_env: list[int], heavy_env: list[int]) -> CalibrationProfile:
    if not rest_env or not light_env or not heavy_env:
        raise CalibrationError("rest/light/heavy calibration sets are required")

    rest_max = _percentile(rest_env, 0.95)
    light_center = float(median(light_env))
    heavy_center = float(median(heavy_env))

    light_threshold = (rest_max + light_center) / 2.0
    heavy_threshold = (light_center + heavy_center) / 2.0
    spread = max(heavy_threshold - light_threshold, 1.0)
    hysteresis_margin = spread * 0.10

    light_sep = light_center - rest_max
    heavy_sep = heavy_center - light_center
    sep_ok = light_sep > 5.0 and heavy_sep > 5.0

    # Convert stage spacing into conservative z-threshold estimates.
    t_low_enter = 2.2 if sep_ok else 2.8
    t_high_enter = 4.0 if sep_ok else 4.8
    t_low_exit = t_low_enter * 0.70
    t_high_exit = t_high_enter * 0.78

    return CalibrationProfile(
        rest_max=rest_max,
        light_threshold=light_threshold,
        heavy_threshold=heavy_threshold,
        hysteresis_margin=hysteresis_margin,
        t_low_enter=t_low_enter,
        t_low_exit=t_low_exit,
        t_high_enter=t_high_enter,
        t_high_exit=t_high_exit,
        separation_ok=sep_ok,
    )


def default_profile() -> CalibrationProfile:
    return CalibrationProfile()


def save_profile(profile: CalibrationProfile, path: Path = PROFILE_PATH) -> None:
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def load_profile(path: Path = PROFILE_PATH) -> CalibrationProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(CalibrationProfile)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return CalibrationProfile(**filtered)
