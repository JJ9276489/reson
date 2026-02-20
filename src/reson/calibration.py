from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from math import exp, log
from pathlib import Path
from statistics import median
from typing import Any

from reson.features import compute_feature_hash


PROFILE_PATH = Path(".reson_profile.json")


def _percentile(values: list[int], p: float) -> float:
    if not values:
        raise CalibrationError("Calibration sample set is empty")
    idx = int((len(values) - 1) * p)
    sorted_vals = sorted(values)
    return float(sorted_vals[idx])


def _safe_log(value: float) -> float:
    return log(max(value, 1e-9))


def _log_row(row: list[float]) -> list[float]:
    total = sum(max(v, 0.0) for v in row)
    if total <= 0:
        n = max(len(row), 1)
        return [_safe_log(1.0 / n) for _ in row]
    probs = [max(v, 0.0) / total for v in row]
    return [_safe_log(v) for v in probs]


def blend_transition_probs(
    prior: list[list[float]],
    estimated: list[list[float]],
    prior_weight: float = 0.8,
) -> list[list[float]]:
    prior_weight = min(max(prior_weight, 0.0), 1.0)
    est_weight = 1.0 - prior_weight
    out: list[list[float]] = []
    for row_p, row_e in zip(prior, estimated):
        row = [(prior_weight * p) + (est_weight * e) for p, e in zip(row_p, row_e)]
        total = sum(max(v, 0.0) for v in row)
        if total <= 0:
            n = max(len(row), 1)
            out.append([1.0 / n for _ in row])
        else:
            out.append([max(v, 0.0) / total for v in row])
    return out


def _default_feature_order() -> list[str]:
    return ["rms_state", "lf_energy_ratio", "slope_burst", "waveform_length"]


def _default_feature_config() -> dict[str, Any]:
    return {
        "window_ms": 120,
        "hop_ms": 30,
        "feature_order": _default_feature_order(),
    }


def _default_feature_hash() -> str:
    return compute_feature_hash(_default_feature_order())


def _default_normalization() -> dict[str, Any]:
    return {
        "center": {
            "rms_state": 0.0,
            "lf_energy_ratio": 0.0,
            "slope_burst": 0.0,
            "waveform_length": 0.0,
        },
        "scale": {
            "rms_state": 1.0,
            "lf_energy_ratio": 1.0,
            "slope_burst": 1.0,
            "waveform_length": 1.0,
        },
        "floor": {
            "rms_state": 1.0,
            "lf_energy_ratio": 0.25,
            "slope_burst": 0.25,
            "waveform_length": 1.0,
        },
        "drift_cap_per_min": {
            "rms_state": 0.5,
            "lf_energy_ratio": 0.25,
            "slope_burst": 0.25,
            "waveform_length": 0.5,
        },
    }


def _default_classifier() -> dict[str, Any]:
    # Classes: REST, PRESS, ARTIFACT. Weights align with _default_feature_order.
    return {
        "classes": ["REST", "PRESS", "ARTIFACT"],
        "weights": [
            [-2.2, -0.7, -0.6, -1.4],
            [2.4, -0.8, -0.7, 1.8],
            [0.1, 2.2, 2.4, 0.2],
        ],
        "bias": [1.4, -1.0, -1.2],
    }


def _default_hmm() -> dict[str, Any]:
    states = ["REST", "PRESS", "ARTIFACT"]
    start_probs = [0.90, 0.05, 0.05]
    transition_prior_prob = [
        [0.93, 0.05, 0.02],
        [0.10, 0.85, 0.05],
        [0.35, 0.03, 0.62],
    ]
    transition_est_prob = [
        [0.90, 0.07, 0.03],
        [0.12, 0.82, 0.06],
        [0.40, 0.02, 0.58],
    ]
    transition_final_prob = blend_transition_probs(transition_prior_prob, transition_est_prob, prior_weight=0.8)
    return {
        "states": states,
        "start_logp": [_safe_log(v) for v in start_probs],
        "transition_logp_prior": [_log_row(row) for row in transition_prior_prob],
        "transition_logp_est": [_log_row(row) for row in transition_est_prob],
        "transition_logp_final": [_log_row(row) for row in transition_final_prob],
        "lag_frames": 4,
    }


def _default_segment_thresholds() -> dict[str, float]:
    return {
        "dur_heavy_ms": 220.0,
        "peak_heavy_u": 2.0,
        "auc_heavy": 260.0,
        "mean_heavy_u": 1.3,
    }


def _default_decision_gates() -> dict[str, int]:
    return {
        "enter_dwell_frames": 3,
        "release_dwell_frames": 3,
        "min_event_ms": 60,
        "refractory_ms": 90,
        "min_rest_gap_ms": 140,
        "rest_conf_frames_min": 3,
    }


def _default_metadata() -> dict[str, Any]:
    return {
        "notes": "Default hmm3 model",
    }


@dataclass(frozen=True)
class Hmm3Model:
    model_version: int = 5
    detector_mode: str = "hmm3"
    feature_config: dict[str, Any] = field(default_factory=_default_feature_config)
    feature_hash: str = field(default_factory=_default_feature_hash)
    normalization: dict[str, Any] = field(default_factory=_default_normalization)
    classifier: dict[str, Any] = field(default_factory=_default_classifier)
    hmm: dict[str, Any] = field(default_factory=_default_hmm)
    segment_thresholds: dict[str, Any] = field(default_factory=_default_segment_thresholds)
    decision_gates: dict[str, Any] = field(default_factory=_default_decision_gates)
    metadata: dict[str, Any] = field(default_factory=_default_metadata)

    @classmethod
    def from_profile(cls, profile: CalibrationProfile) -> "Hmm3Model":
        return cls(
            model_version=profile.model_version,
            detector_mode=profile.detector_mode,
            feature_config=profile.feature_config,
            feature_hash=profile.feature_hash,
            normalization=profile.normalization,
            classifier=profile.classifier,
            hmm=profile.hmm,
            segment_thresholds=profile.segment_thresholds,
            decision_gates=profile.decision_gates,
            metadata=profile.metadata,
        )


@dataclass(frozen=True)
class CalibrationProfile:
    # Legacy threshold fields retained for compatibility.
    rest_max: float = 10.0
    light_threshold: float = 20.0
    heavy_threshold: float = 40.0
    hysteresis_margin: float = 2.0

    # Adaptive detector tuning defaults (v2.4.x path).
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

    # Hmm3 model payload (v2.5.2 schema keys).
    model_version: int = 5
    detector_mode: str = "hmm3"
    feature_config: dict[str, Any] = field(default_factory=_default_feature_config)
    feature_hash: str = field(default_factory=_default_feature_hash)
    normalization: dict[str, Any] = field(default_factory=_default_normalization)
    classifier: dict[str, Any] = field(default_factory=_default_classifier)
    hmm: dict[str, Any] = field(default_factory=_default_hmm)
    segment_thresholds: dict[str, Any] = field(default_factory=_default_segment_thresholds)
    decision_gates: dict[str, Any] = field(default_factory=_default_decision_gates)
    metadata: dict[str, Any] = field(default_factory=_default_metadata)

    separation_ok: bool = True
    profile_version: int = 4


class CalibrationError(RuntimeError):
    pass


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
