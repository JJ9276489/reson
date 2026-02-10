from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_press_class_is_latched_until_up():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        bootstrap_ms=0,
        filter_enabled=False,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 1500, 0),  # press candidate
        EmgSample(320, 1500, 0),  # commit down
        EmgSample(400, 2800, 0),  # would look heavier, class must stay latched
        EmgSample(480, 2800, 0),
        EmgSample(600, 1000, 0),
        EmgSample(700, 1000, 0),
        EmgSample(800, 1000, 0),
        EmgSample(900, 1000, 0),
        EmgSample(1000, 1000, 0),
        EmgSample(1100, 1000, 0),
        EmgSample(1200, 1000, 0),
        EmgSample(1300, 1000, 0),  # commit up after strict rest confirmation
    ]

    events = []
    for sample in stream:
        detector.update(sample)
        events.extend(detector.pop_events())

    downs = [e for e in events if e.phase == "down"]
    ups = [e for e in events if e.phase == "up"]
    assert len(downs) == 1
    assert len(ups) == 1
    assert downs[0].press_class in ("light", "heavy")
    assert ups[0].press_class == downs[0].press_class
