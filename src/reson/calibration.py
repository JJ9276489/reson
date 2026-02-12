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

    # Adaptive detector tuning defaults (v2.4.1 RMS-state model).
    u_light_enter: float = 1.0
    u_light_exit: float = 0.8
    u_heavy_enter: float = 2.0
    u_heavy_exit: float = 1.6
    u_rest_conf_threshold: float = 0.6
    min_dwell_ms: int = 90
    min_event_ms: int = 60
    min_rest_gap_ms: int = 140
    refractory_ms: int = 90
    rest_conf_dwell_ms: int = 80
    tau_baseline_ms: float = 2000.0
    filter_enabled: bool = True
    rest_scale_floor: float = 5.0
    rms_state_window_ms: int = 180
    rest_stats_window_s: float = 3.0
    bootstrap_ms: int = 3000
    quiet_window_ms: int = 80
    quiet_fraction: float = 0.25
    hp_hz: float = 20.0
    lp_hz: float = 230.0
    notch_hz: float = 60.0
    notch_q: float = 20.0
    artifact_enter: float = 1.9
    artifact_exit: float = 1.4
    artifact_holdoff_ms: int = 180
    artifact_lf_hz: float = 10.0
    slope_fast_tau_ms: float = 40.0
    slope_slow_tau_ms: float = 400.0
    separation_ok: bool = True
    profile_version: int = 3


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

    # Conservative defaults for normalized RMS state thresholds.
    u_light_enter = 1.0 if sep_ok else 1.2
    u_heavy_enter = 2.0 if sep_ok else 2.4
    u_light_exit = u_light_enter * 0.8
    u_heavy_exit = u_heavy_enter * 0.8
    u_rest_conf_threshold = u_light_exit * 0.75

    return CalibrationProfile(
        rest_max=rest_max,
        light_threshold=light_threshold,
        heavy_threshold=heavy_threshold,
        hysteresis_margin=hysteresis_margin,
        u_light_enter=u_light_enter,
        u_light_exit=u_light_exit,
        u_heavy_enter=u_heavy_enter,
        u_heavy_exit=u_heavy_exit,
        u_rest_conf_threshold=u_rest_conf_threshold,
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
