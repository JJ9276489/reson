from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_borderline_activity_does_not_count_as_confident_rest():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        rest_conf_dwell_ms=120,
        u_rest_conf_threshold=0.1,
        bootstrap_ms=0,
        filter_enabled=False,
    )

    # Baseline at 1000, then borderline activity around the rest-confidence line.
    detector.update(EmgSample(0, 1000, 0))
    for t in (60, 120, 180, 240, 300, 360, 420):
        detector.update(EmgSample(t, 1003, 0))

    dbg = detector.last_debug()
    assert dbg is not None
    assert dbg.rest_confident is False
    assert len(detector.features._rest_window) == 0
