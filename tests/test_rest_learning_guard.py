from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_rest_learning_requires_confident_rest_not_gated_and_no_pending():
    detector = AdaptiveEdgeDetector(
        bootstrap_ms=0,
        filter_enabled=False,
        rest_conf_dwell_ms=80,
        artifact_enter=1.1,
        artifact_exit=1.0,
        artifact_holdoff_ms=220,
        u_light_enter=3.0,
        u_light_exit=2.2,
        u_heavy_enter=8.0,
        u_heavy_exit=6.0,
    )

    # Build confident REST first.
    for t in (0, 80, 160, 240, 320, 400):
        detector.update(EmgSample(t, 1000, 0))

    n_before = len(detector.features._rest_window)

    # Enter artifact gate; subsequent samples should not update rest learning.
    detector.update(EmgSample(480, 1100, 0))
    for t in (560, 640, 720):
        detector.update(EmgSample(t, 1000, 0))

    n_after = len(detector.features._rest_window)
    dbg = detector.last_debug()
    assert dbg is not None
    assert dbg.artifact_gated is True
    assert n_after == n_before
