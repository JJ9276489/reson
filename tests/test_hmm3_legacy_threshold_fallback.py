from reson.edge_detector import ThresholdEdgeDetector, make_detector


def test_legacy_threshold_detector_mode_still_available():
    detector = make_detector("threshold", None)
    assert isinstance(detector, ThresholdEdgeDetector)
