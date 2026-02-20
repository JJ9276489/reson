from reson.edge_detector import FeatureFrame, make_detector

from hmm3_test_utils import make_hmm3_test_profile


def test_hmm3_adaptation_requires_strict_rest_guard_and_respects_drift_cap():
    detector = make_detector("hmm3", make_hmm3_test_profile())
    frame = FeatureFrame(
        t_ms=60000,
        window_start_ms=59940,
        window_end_ms=60000,
        env_in=0,
        filtered_raw_hp=0.0,
        rms_state=10.0,
        lf_energy_ratio=0.0,
        slope_burst=0.0,
        waveform_length=0.0,
    )

    center_start = detector._norm_center["rms_state"]

    detector._decoded_stable = "PRESS"
    detector._decoded_pending = None
    detector._artifact_gated = False
    detector._rest_conf_frames = detector._rest_conf_frames_min
    detector._maybe_adapt_normalization(frame)
    assert detector._norm_center["rms_state"] == center_start

    detector._decoded_stable = "REST"
    detector._decoded_pending = object()  # type: ignore[assignment]
    detector._maybe_adapt_normalization(frame)
    assert detector._norm_center["rms_state"] == center_start

    detector._decoded_pending = None
    detector._artifact_gated = True
    detector._maybe_adapt_normalization(frame)
    assert detector._norm_center["rms_state"] == center_start

    detector._artifact_gated = False
    detector._rest_conf_frames = detector._rest_conf_frames_min
    detector._last_adapt_t_ms = 0
    detector._maybe_adapt_normalization(frame)
    center_after = detector._norm_center["rms_state"]

    # Alpha=0.02 on value=10 gives a target delta of +0.2, below drift cap (1.0 / min).
    assert center_after > center_start
    assert center_after <= center_start + 0.2 + 1e-9
