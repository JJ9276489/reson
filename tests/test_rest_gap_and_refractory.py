from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_refractory_and_rest_gap_block_rapid_repeat_presses():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=60,
        refractory_ms=90,
        min_rest_gap_ms=260,
        tau_fast_ms=10,
        tau_slow_ms=300,
        bootstrap_ms=0,
        filter_enabled=False,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 2200, 0),
        EmgSample(320, 2200, 0),  # press 1 down
        EmgSample(420, 2200, 0),
        EmgSample(520, 1000, 0),
        EmgSample(620, 1000, 0),
        EmgSample(720, 1000, 0),
        EmgSample(820, 1000, 0),
        EmgSample(920, 1000, 0),  # press 1 up
        EmgSample(980, 2200, 0),  # within refractory/rest-gap -> blocked
        EmgSample(1060, 2200, 0),
        EmgSample(1240, 1000, 0),
        EmgSample(1340, 1000, 0),
        EmgSample(1440, 1000, 0),
        EmgSample(1540, 2200, 0),  # after gap
        EmgSample(1620, 2200, 0),  # press 2 down
    ]

    events = []
    for s in stream:
        detector.update(s)
        events.extend(detector.pop_events())

    downs = [e for e in events if e.phase == "down"]
    assert len(downs) == 2
