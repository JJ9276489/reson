from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_refractory_and_rest_gap_block_rapid_repeat_presses():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        refractory_ms=90,
        min_rest_gap_ms=260,
        bootstrap_ms=0,
        filter_enabled=False,
        u_light_enter=3.0,
        u_light_exit=2.2,
        u_heavy_enter=8.0,
        u_heavy_exit=6.0,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 1035, 0),
        EmgSample(320, 1035, 0),  # press 1 down
        EmgSample(420, 1035, 0),
        EmgSample(520, 1000, 0),
        EmgSample(620, 1000, 0),
        EmgSample(720, 1000, 0),
        EmgSample(820, 1000, 0),
        EmgSample(920, 1000, 0),  # press 1 up
        EmgSample(980, 1035, 0),  # within refractory/rest-gap -> blocked
        EmgSample(1060, 1035, 0),
        EmgSample(1240, 1000, 0),
        EmgSample(1340, 1000, 0),
        EmgSample(1440, 1000, 0),
        EmgSample(1540, 1035, 0),  # after gap
        EmgSample(1620, 1035, 0),  # press 2 down
    ]

    events = []
    for s in stream:
        detector.update(s)
        events.extend(detector.pop_events())

    downs = [e for e in events if e.phase == "down"]
    assert len(downs) == 2
