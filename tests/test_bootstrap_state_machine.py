from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_bootstrap_and_arming_produce_no_events_until_running():
    detector = AdaptiveEdgeDetector(
        bootstrap_ms=300,
        rest_conf_dwell_ms=80,
        filter_enabled=False,
    )

    events = []
    for t, raw in [
        (0, 2600),
        (80, 2400),
        (160, 2200),
        (240, 1800),
        (320, 1300),  # bootstrap ends around here, enters ARMING
        (400, 1100),
        (480, 1050),
        (560, 1000),  # should arm soon after clean rest
    ]:
        detector.update(EmgSample(t, raw, 0))
        events.extend(detector.pop_events())
        if not detector.is_armed():
            assert all(e.phase is None for e in events)

    assert detector.phase() in ("ARMING", "RUNNING")
    assert all(e.phase is None for e in events)


def test_bootstrap_initializes_sigma_above_floor_when_noise_present():
    detector = AdaptiveEdgeDetector(
        bootstrap_ms=300,
        sigma_floor=5.0,
        filter_enabled=False,
    )

    for t, raw in [
        (0, 1000),
        (60, 1020),
        (120, 980),
        (180, 1015),
        (240, 990),
        (320, 1005),
    ]:
        detector.update(EmgSample(t, raw, 0))

    dbg = detector.last_debug()
    assert dbg is not None
    assert dbg.sigma >= 5.0
