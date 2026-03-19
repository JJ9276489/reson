from __future__ import annotations

import pytest

from reson.calibration import default_profile
from reson.edge_detector import Hmm3EdgeDetector, make_detector
from reson.features import FeatureFrame


def test_hmm3_wl_only_ablation_masks_other_emission_features():
    detector = make_detector("hmm3", default_profile(), feature_ablation="wl-only")
    assert isinstance(detector, Hmm3EdgeDetector)
    assert detector.active_features() == ("waveform_length",)

    frame = FeatureFrame(
        t_ms=1000,
        window_start_ms=880,
        window_end_ms=1000,
        env_in=0,
        filtered_raw_hp=12.0,
        rms_state=3.0,
        lf_energy_ratio=4.0,
        slope_burst=5.0,
        waveform_length=6.0,
    )
    vec, frame_u = detector._normalize(frame)
    assert frame_u == pytest.approx(3.0)
    assert vec == pytest.approx([0.0, 0.0, 0.0, 6.0])


def test_hmm3_custom_feature_ablation_keeps_requested_features_in_order():
    detector = make_detector(
        "hmm3",
        default_profile(),
        feature_ablation="waveform_length,rms_state",
    )
    assert isinstance(detector, Hmm3EdgeDetector)
    assert detector.active_features() == ("rms_state", "waveform_length")


def test_hmm3_feature_ablation_rejects_unknown_feature():
    with pytest.raises(ValueError, match="unknown feature"):
        make_detector("hmm3", default_profile(), feature_ablation="not_a_feature")
