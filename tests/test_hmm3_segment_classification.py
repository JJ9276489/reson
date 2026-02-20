from reson.edge_detector import PressSegmentStats, make_detector

from hmm3_test_utils import make_hmm3_test_profile


def test_hmm3_segment_classifier_uses_two_of_four_rule():
    detector = make_detector("hmm3", make_hmm3_test_profile())

    light_seg = PressSegmentStats(
        start_ms=0,
        last_ms=180,
        duration_ms=180,
        peak_u=1.5,
        auc_u=180.0,
        mean_u=1.0,
    )
    heavy_seg = PressSegmentStats(
        start_ms=0,
        last_ms=320,
        duration_ms=320,
        peak_u=2.5,
        auc_u=240.0,
        mean_u=0.9,
    )

    assert detector._classify_segment(light_seg) == "light"
    assert detector._classify_segment(heavy_seg) == "heavy"
