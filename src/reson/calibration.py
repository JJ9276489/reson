from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


PROFILE_PATH = Path(".reson_profile.json")


@dataclass(frozen=True)
class CalibrationProfile:
    rest_max: float
    light_threshold: float
    heavy_threshold: float
    hysteresis_margin: float


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

    return CalibrationProfile(
        rest_max=rest_max,
        light_threshold=light_threshold,
        heavy_threshold=heavy_threshold,
        hysteresis_margin=hysteresis_margin,
    )


def save_profile(profile: CalibrationProfile, path: Path = PROFILE_PATH) -> None:
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def load_profile(path: Path = PROFILE_PATH) -> CalibrationProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationProfile(**data)
