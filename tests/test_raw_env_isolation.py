from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def _run(detector: AdaptiveEdgeDetector, env_values: list[int]):
    states = []
    events = []
    raw_values = [1000, 1000, 1000, 1800, 1800, 1800, 1000, 1000]
    t_values = [0, 50, 100, 150, 200, 250, 300, 350]
    for t, raw, env in zip(t_values, raw_values, env_values):
        states.append(detector.update(EmgSample(t, raw, env)))
        events.extend(detector.pop_events())
    return states, [(e.state, e.phase, e.duration_ms) for e in events]


def test_env_input_does_not_change_adaptive_detector_output():
    detector_a = AdaptiveEdgeDetector(min_dwell_ms=20, min_event_ms=40)
    detector_b = AdaptiveEdgeDetector(min_dwell_ms=20, min_event_ms=40)

    states_a, events_a = _run(detector_a, [0] * 8)
    states_b, events_b = _run(detector_b, [1, 50, 99, 1000, 2000, 4095, 10, 3])

    assert states_a == states_b
    assert events_a == events_b
