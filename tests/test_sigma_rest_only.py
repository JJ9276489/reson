from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_sigma_updates_only_from_rest_buffer():
    detector = AdaptiveEdgeDetector(min_dwell_ms=40, min_event_ms=40, tau_fast_ms=20, tau_slow_ms=500)

    for t in (0, 50, 100, 150):
        detector.update(EmgSample(t, 1000, 0))
    n_before = len(detector.features.a_rest)

    for t in (200, 260, 320, 380, 440):
        detector.update(EmgSample(t, 1900, 0))
    n_after_press = len(detector.features.a_rest)

    assert n_after_press <= n_before + 1
