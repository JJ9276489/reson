from __future__ import annotations

from dataclasses import replace
from math import log

from reson.calibration import CalibrationProfile, default_profile
from reson.features import compute_feature_hash
from reson.types import EmgSample


def _log_rows(rows: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in rows:
        total = sum(row)
        out.append([log(max(v / total, 1e-9)) for v in row])
    return out


def make_hmm3_test_profile(
    *,
    bootstrap_ms: int = 0,
    rest_conf_frames_min: int = 2,
    lag_frames: int = 1,
) -> CalibrationProfile:
    feature_order = ["rms_state"]
    feature_config = {
        "window_ms": 60,
        "hop_ms": 20,
        "feature_order": feature_order,
    }

    normalization = {
        "center": {"rms_state": 0.0},
        "scale": {"rms_state": 1.0},
        "floor": {"rms_state": 0.5},
        "drift_cap_per_min": {"rms_state": 1.0},
    }

    classifier = {
        "classes": ["REST", "PRESS", "ARTIFACT"],
        # scores:
        #   rest = 2 - x
        #   press = 0
        #   artifact = x - 6
        # yields low-x REST, mid-x PRESS, high-x ARTIFACT.
        "weights": [[-1.0], [0.0], [1.0]],
        "bias": [2.0, 0.0, -6.0],
    }

    trans_prior = [
        [0.93, 0.06, 0.01],
        [0.10, 0.88, 0.02],
        [0.35, 0.01, 0.64],
    ]
    trans_est = [
        [0.90, 0.08, 0.02],
        [0.12, 0.84, 0.04],
        [0.40, 0.01, 0.59],
    ]

    hmm = {
        "states": ["REST", "PRESS", "ARTIFACT"],
        "start_logp": [log(0.99), log(0.005), log(0.005)],
        "transition_logp_prior": _log_rows(trans_prior),
        "transition_logp_est": _log_rows(trans_est),
        "transition_logp_final": _log_rows(
            [[0.8 * p + 0.2 * e for p, e in zip(p_row, e_row)] for p_row, e_row in zip(trans_prior, trans_est)]
        ),
        "lag_frames": lag_frames,
    }

    segment_thresholds = {
        "dur_heavy_ms": 220.0,
        "peak_heavy_u": 2.0,
        "auc_heavy": 260.0,
        "mean_heavy_u": 1.3,
    }
    decision_gates = {
        "enter_dwell_frames": 2,
        "release_dwell_frames": 2,
        "min_event_ms": 50,
        "refractory_ms": 70,
        "min_rest_gap_ms": 80,
        "rest_conf_frames_min": rest_conf_frames_min,
    }

    return replace(
        default_profile(),
        bootstrap_ms=bootstrap_ms,
        filter_enabled=False,
        feature_config=feature_config,
        feature_hash=compute_feature_hash(feature_order),
        normalization=normalization,
        classifier=classifier,
        hmm=hmm,
        segment_thresholds=segment_thresholds,
        decision_gates=decision_gates,
    )


def stream_segment(start_ms: int, end_ms: int, step_ms: int, raw_value: int) -> list[EmgSample]:
    samples: list[EmgSample] = []
    t_ms = start_ms
    while t_ms <= end_ms:
        samples.append(EmgSample(t_ms=t_ms, raw=raw_value, env=0))
        t_ms += step_ms
    return samples
