from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from math import exp, log
from pathlib import Path
from statistics import median
from typing import Any

from reson.features import FeatureFrame, compute_feature_hash


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


def _percentile_float(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * p)
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _robust_center_scale(values: list[float], scale_floor: float) -> tuple[float, float]:
    if not values:
        return 0.0, scale_floor
    center = float(median(values))
    p90 = _percentile_float(values, 0.90)
    p50 = _percentile_float(values, 0.50)
    mad = float(median([abs(v - p50) for v in values])) if values else 0.0
    scale = max(p90 - p50, 1.4826 * mad, scale_floor)
    return center, scale


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [exp(v - m) for v in logits]
    denom = max(sum(exps), 1e-9)
    return [v / denom for v in exps]


def _fit_softmax_classifier(
    features: list[list[float]],
    targets: list[int],
    num_classes: int,
    *,
    epochs: int = 300,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> tuple[list[list[float]], list[float]]:
    if not features or not targets:
        raise CalibrationError("No labeled frames available for classifier fitting")
    dim = len(features[0])
    weights = [[0.0 for _ in range(dim)] for _ in range(num_classes)]
    bias = [0.0 for _ in range(num_classes)]

    class_counts = [0 for _ in range(num_classes)]
    for y in targets:
        class_counts[y] += 1
    total = max(len(targets), 1)
    class_weights = [
        (total / max(num_classes * count, 1)) for count in class_counts
    ]

    for epoch in range(epochs):
        grad_w = [[0.0 for _ in range(dim)] for _ in range(num_classes)]
        grad_b = [0.0 for _ in range(num_classes)]

        for x, y in zip(features, targets):
            logits = [
                sum(weights[c][j] * x[j] for j in range(dim)) + bias[c]
                for c in range(num_classes)
            ]
            probs = _softmax(logits)
            sample_weight = class_weights[y]
            for c in range(num_classes):
                diff = (probs[c] - (1.0 if c == y else 0.0)) * sample_weight
                grad_b[c] += diff
                for j in range(dim):
                    grad_w[c][j] += diff * x[j]

        inv_n = 1.0 / max(len(features), 1)
        step = lr * (0.6 if epoch > (epochs // 2) else 1.0)
        for c in range(num_classes):
            bias[c] -= step * grad_b[c] * inv_n
            for j in range(dim):
                reg_term = l2 * weights[c][j]
                weights[c][j] -= step * ((grad_w[c][j] * inv_n) + reg_term)

    return weights, bias


def _stage_to_hmm_class(stage: str) -> str:
    if stage == "REST":
        return "REST"
    if stage in ("PRESS_LIGHT", "PRESS_HEAVY"):
        return "PRESS"
    if stage == "ARTIFACT":
        return "ARTIFACT"
    raise CalibrationError(f"Unknown calibration stage label: {stage}")


def _estimate_transition_probs(labels: list[int], num_states: int, laplace: float = 1.0) -> list[list[float]]:
    counts = [[laplace for _ in range(num_states)] for _ in range(num_states)]
    for a, b in zip(labels, labels[1:]):
        counts[a][b] += 1.0
    out: list[list[float]] = []
    for row in counts:
        total = sum(row)
        out.append([v / max(total, 1e-9) for v in row])
    return out


def _extract_segments_from_u(
    points: list[tuple[int, float]],
    *,
    enter_u: float = 0.8,
    exit_u: float = 0.4,
) -> list[dict[str, float]]:
    if not points:
        return []
    segs: list[dict[str, float]] = []
    active = False
    start_t = 0
    prev_t = points[0][0]
    peak_u = 0.0
    auc_u = 0.0

    for t_ms, u in points:
        dt = max(t_ms - prev_t, 1)
        prev_t = t_ms
        if not active:
            if u >= enter_u:
                active = True
                start_t = t_ms
                peak_u = max(u, 0.0)
                auc_u = max(u, 0.0) * dt
            continue

        peak_u = max(peak_u, u)
        auc_u += max(u, 0.0) * dt
        if u <= exit_u:
            duration = max(t_ms - start_t, 1)
            segs.append(
                {
                    "duration_ms": float(duration),
                    "peak_u": float(peak_u),
                    "auc_u": float(auc_u),
                    "mean_u": float(auc_u / duration),
                }
            )
            active = False

    if active:
        end_t = points[-1][0]
        duration = max(end_t - start_t, 1)
        segs.append(
            {
                "duration_ms": float(duration),
                "peak_u": float(peak_u),
                "auc_u": float(auc_u),
                "mean_u": float(auc_u / duration),
            }
        )
    return segs


def _between_medians(light_vals: list[float], heavy_vals: list[float], default: float) -> float:
    if not light_vals or not heavy_vals:
        return default
    l_med = float(median(light_vals))
    h_med = float(median(heavy_vals))
    if h_med <= l_med:
        return default
    return (l_med + h_med) / 2.0


def fit_hmm3_profile_from_staged_frames(
    staged_frames: list[tuple[FeatureFrame, str]],
    *,
    base_profile: CalibrationProfile | None = None,
) -> CalibrationProfile:
    if not staged_frames:
        raise CalibrationError("No staged frames available for calibration")

    profile = base_profile or default_profile()
    feature_config = dict(profile.feature_config)
    feature_order = list(feature_config.get("feature_order", _default_feature_order()))
    if not feature_order:
        raise CalibrationError("feature_order is empty; cannot calibrate hmm3")

    floor_cfg = dict(profile.normalization.get("floor", {}))
    center_cfg: dict[str, float] = {}
    scale_cfg: dict[str, float] = {}

    rest_frames = [frame for frame, stage in staged_frames if stage == "REST"]
    if not rest_frames:
        raise CalibrationError("REST stage produced no frames; cannot fit normalization")

    # Fit normalization from REST only.
    for key in feature_order:
        vals = [float(getattr(frame, key)) for frame in rest_frames]
        floor = float(floor_cfg.get(key, 1.0))
        center, scale = _robust_center_scale(vals, floor)
        center_cfg[key] = center
        scale_cfg[key] = scale

    def normalize_frame(frame: FeatureFrame) -> list[float]:
        vec: list[float] = []
        for key in feature_order:
            value = float(getattr(frame, key))
            center = float(center_cfg.get(key, 0.0))
            scale = float(scale_cfg.get(key, 1.0))
            floor = float(floor_cfg.get(key, 1.0))
            vec.append((value - center) / max(scale, floor))
        return vec

    classes = ["REST", "PRESS", "ARTIFACT"]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    features: list[list[float]] = []
    targets: list[int] = []
    class_sequence: list[int] = []
    for frame, stage in staged_frames:
        klass = _stage_to_hmm_class(stage)
        features.append(normalize_frame(frame))
        idx = class_to_idx[klass]
        targets.append(idx)
        class_sequence.append(idx)

    weights, bias = _fit_softmax_classifier(features, targets, len(classes))

    # Transition model estimate from labeled sequence.
    prior_log = profile.hmm.get("transition_logp_prior", _default_hmm()["transition_logp_prior"])
    prior_probs = [[exp(v) for v in row] for row in prior_log]
    est_probs = _estimate_transition_probs(class_sequence, len(classes), laplace=1.0)
    final_probs = blend_transition_probs(prior_probs, est_probs, prior_weight=0.8)

    start_prior_log = profile.hmm.get("start_logp", _default_hmm()["start_logp"])
    start_prior = [exp(v) for v in start_prior_log]
    start_est = [1.0 for _ in classes]
    if class_sequence:
        start_est[class_sequence[0]] += 4.0
    total_start_est = sum(start_est)
    start_est = [v / total_start_est for v in start_est]
    start_final = [(0.8 * p) + (0.2 * e) for p, e in zip(start_prior, start_est)]
    start_final_total = max(sum(start_final), 1e-9)
    start_final = [v / start_final_total for v in start_final]

    # Segment thresholds from PRESS_LIGHT and PRESS_HEAVY stages using normalized u (rms_state z-score).
    rms_idx = feature_order.index("rms_state") if "rms_state" in feature_order else 0
    light_points: list[tuple[int, float]] = []
    heavy_points: list[tuple[int, float]] = []
    for frame, stage in staged_frames:
        vec = normalize_frame(frame)
        u = vec[rms_idx]
        if stage == "PRESS_LIGHT":
            light_points.append((frame.t_ms, u))
        elif stage == "PRESS_HEAVY":
            heavy_points.append((frame.t_ms, u))

    light_segs = _extract_segments_from_u(light_points)
    heavy_segs = _extract_segments_from_u(heavy_points)

    seg_defaults = dict(profile.segment_thresholds)
    segment_thresholds = {
        "dur_heavy_ms": _between_medians(
            [s["duration_ms"] for s in light_segs],
            [s["duration_ms"] for s in heavy_segs],
            float(seg_defaults.get("dur_heavy_ms", 220.0)),
        ),
        "peak_heavy_u": _between_medians(
            [s["peak_u"] for s in light_segs],
            [s["peak_u"] for s in heavy_segs],
            float(seg_defaults.get("peak_heavy_u", 2.0)),
        ),
        "auc_heavy": _between_medians(
            [s["auc_u"] for s in light_segs],
            [s["auc_u"] for s in heavy_segs],
            float(seg_defaults.get("auc_heavy", 260.0)),
        ),
        "mean_heavy_u": _between_medians(
            [s["mean_u"] for s in light_segs],
            [s["mean_u"] for s in heavy_segs],
            float(seg_defaults.get("mean_heavy_u", 1.3)),
        ),
    }

    separation_ok = bool(heavy_segs and light_segs)
    if separation_ok:
        separation_ok = (
            median([s["peak_u"] for s in heavy_segs]) > median([s["peak_u"] for s in light_segs])
        )

    normalization = {
        "center": center_cfg,
        "scale": scale_cfg,
        "floor": floor_cfg,
        "drift_cap_per_min": dict(profile.normalization.get("drift_cap_per_min", {})),
    }
    classifier = {"classes": classes, "weights": weights, "bias": bias}
    hmm = {
        "states": classes,
        "start_logp": [_safe_log(v) for v in start_final],
        "transition_logp_prior": [_log_row(row) for row in prior_probs],
        "transition_logp_est": [_log_row(row) for row in est_probs],
        "transition_logp_final": [_log_row(row) for row in final_probs],
        "lag_frames": int(profile.hmm.get("lag_frames", 4)),
    }

    metadata = dict(profile.metadata)
    metadata.update(
        {
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_frame_count": len(staged_frames),
            "stage_frame_counts": {
                stage: sum(1 for _, s in staged_frames if s == stage)
                for stage in ("REST", "PRESS_LIGHT", "PRESS_HEAVY", "ARTIFACT")
            },
        }
    )

    return replace(
        profile,
        detector_mode="hmm3",
        model_version=5,
        feature_config={
            "window_ms": int(feature_config.get("window_ms", 120)),
            "hop_ms": int(feature_config.get("hop_ms", 30)),
            "feature_order": feature_order,
        },
        feature_hash=compute_feature_hash(feature_order),
        normalization=normalization,
        classifier=classifier,
        hmm=hmm,
        segment_thresholds=segment_thresholds,
        metadata=metadata,
        separation_ok=separation_ok,
    )


def save_profile(profile: CalibrationProfile, path: Path = PROFILE_PATH) -> None:
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")


def load_profile(path: Path = PROFILE_PATH) -> CalibrationProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(CalibrationProfile)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return CalibrationProfile(**filtered)
