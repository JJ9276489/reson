from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_heavy_hysteresis_avoids_release_flicker():
    detector = AdaptiveEdgeDetector(
        bootstrap_ms=0,
        filter_enabled=False,
        min_dwell_ms=40,
        u_light_enter=2.0,
        u_light_exit=1.5,
        u_heavy_enter=4.0,
        u_heavy_exit=3.0,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 1025, 0),
        EmgSample(320, 1025, 0),  # heavy down
        EmgSample(400, 1016, 0),  # around heavy exit zone
        EmgSample(480, 1014, 0),
        EmgSample(560, 1016, 0),
        EmgSample(640, 1000, 0),  # below light exit -> release path
        EmgSample(720, 1000, 0),
        EmgSample(800, 1000, 0),
        EmgSample(880, 1000, 0),
    ]

    events = []
    for s in stream:
        detector.update(s)
        events.extend(detector.pop_events())

    downs = [e for e in events if e.phase == "down"]
    ups = [e for e in events if e.phase == "up"]
    assert len(downs) == 1
    assert len(ups) == 1
