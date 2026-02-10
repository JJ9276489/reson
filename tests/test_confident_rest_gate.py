from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_borderline_activity_does_not_count_as_confident_rest():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        rest_conf_dwell_ms=80,
        bootstrap_ms=0,
        filter_enabled=False,
    )

    for t in (0, 50, 100, 150):
        detector.update(EmgSample(t, 1000, 0))

    n_before = len(detector.features.a_rest)
    # Borderline-but-active burst should suspend confident-rest adaptation.
    for t in (240, 320, 400, 480):
        detector.update(EmgSample(t, 1900, 0))

    assert len(detector.features.a_rest) <= n_before + 1
