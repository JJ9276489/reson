from math import exp

from reson.calibration import default_profile, fit_hmm3_profile_from_staged_frames
from reson.features import FeatureFrame, compute_feature_hash


def _frame(t_ms: int, rms: float, lf: float, slope: float, wave: float) -> FeatureFrame:
    return FeatureFrame(
        t_ms=t_ms,
        window_start_ms=t_ms - 120,
        window_end_ms=t_ms,
        env_in=0,
        filtered_raw_hp=rms,
        rms_state=rms,
        lf_energy_ratio=lf,
        slope_burst=slope,
        waveform_length=wave,
    )


def test_fit_hmm3_profile_from_staged_frames_builds_valid_profile():
    staged: list[tuple[FeatureFrame, str]] = []
    t_ms = 0

    # REST baseline.
    for _ in range(80):
        staged.append((_frame(t_ms, rms=1.0, lf=0.25, slope=0.45, wave=4.5), "REST"))
        t_ms += 30

    # PRESS_LIGHT with repeated pulses.
    for i in range(120):
        if i % 14 < 7:
            staged.append((_frame(t_ms, rms=2.8, lf=0.35, slope=0.65, wave=14.0), "PRESS_LIGHT"))
        else:
            staged.append((_frame(t_ms, rms=1.2, lf=0.30, slope=0.50, wave=6.0), "PRESS_LIGHT"))
        t_ms += 30

    # PRESS_HEAVY with stronger repeated pulses.
    for i in range(120):
        if i % 14 < 8:
            staged.append((_frame(t_ms, rms=5.8, lf=0.45, slope=0.95, wave=32.0), "PRESS_HEAVY"))
        else:
            staged.append((_frame(t_ms, rms=1.5, lf=0.32, slope=0.55, wave=8.0), "PRESS_HEAVY"))
        t_ms += 30

    # ARTIFACT with high low-frequency ratio and slope.
    for _ in range(100):
        staged.append((_frame(t_ms, rms=2.2, lf=2.8, slope=2.4, wave=11.0), "ARTIFACT"))
        t_ms += 30

    profile = fit_hmm3_profile_from_staged_frames(staged, base_profile=default_profile())

    assert profile.detector_mode == "hmm3"
    feature_order = profile.feature_config["feature_order"]
    assert profile.feature_hash == compute_feature_hash(feature_order)

    assert profile.classifier["classes"] == ["REST", "PRESS", "ARTIFACT"]
    weights = profile.classifier["weights"]
    bias = profile.classifier["bias"]
    assert len(weights) == 3
    assert len(bias) == 3
    assert all(len(row) == len(feature_order) for row in weights)

    assert profile.hmm["states"] == ["REST", "PRESS", "ARTIFACT"]
    trans_final = profile.hmm["transition_logp_final"]
    for row in trans_final:
        row_prob_sum = sum(exp(v) for v in row)
        assert abs(row_prob_sum - 1.0) < 1e-6

    assert profile.segment_thresholds["dur_heavy_ms"] > 0.0
    assert profile.segment_thresholds["peak_heavy_u"] > 0.0
    assert profile.segment_thresholds["auc_heavy"] > 0.0
    assert profile.segment_thresholds["mean_heavy_u"] > 0.0
