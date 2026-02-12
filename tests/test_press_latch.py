from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_press_class_is_latched_until_up():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        bootstrap_ms=0,
        filter_enabled=False,
        u_light_enter=3.0,
        u_light_exit=2.2,
        u_heavy_enter=10.0,
        u_heavy_exit=8.0,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 1030, 0),  # light press candidate
        EmgSample(320, 1030, 0),  # commit down as light
        EmgSample(400, 1060, 0),  # would look heavy if escalation were allowed
        EmgSample(480, 1060, 0),
        EmgSample(560, 1000, 0),
        EmgSample(640, 1000, 0),
        EmgSample(720, 1000, 0),
        EmgSample(800, 1000, 0),
    ]

    events = []
    for sample in stream:
        detector.update(sample)
        events.extend(detector.pop_events())

    downs = [e for e in events if e.phase == "down"]
    ups = [e for e in events if e.phase == "up"]
    assert len(downs) == 1
    assert len(ups) == 1
    assert downs[0].press_class == "light"
    assert ups[0].press_class == "light"
