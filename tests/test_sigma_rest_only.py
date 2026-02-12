from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_rest_scale_updates_only_from_rest_buffer():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=40,
        bootstrap_ms=0,
        filter_enabled=False,
        u_light_enter=3.0,
        u_light_exit=2.2,
        u_heavy_enter=8.0,
        u_heavy_exit=6.0,
    )

    for t in (0, 80, 160, 240, 320, 400):
        detector.update(EmgSample(t, 1000, 0))

    n_before = len(detector.features._rest_window)
    t_before = detector.features._rest_window[-1][0]
    scale_before = detector.features.rest_scale

    # Active segment should not feed REST-only normalization window.
    for t in (480, 560, 640, 720):
        detector.update(EmgSample(t, 1040, 0))

    n_after_press = len(detector.features._rest_window)
    t_after_press = detector.features._rest_window[-1][0]
    scale_after_press = detector.features.rest_scale

    assert n_after_press == n_before
    assert t_after_press == t_before
    assert scale_after_press == scale_before

    # Back in rest, normalization should resume.
    for t in (820, 900, 980, 1060, 1140):
        detector.update(EmgSample(t, 1000, 0))

    assert detector.features._rest_window[-1][0] > t_after_press
