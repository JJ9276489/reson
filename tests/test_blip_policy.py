from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def _collect(detector: AdaptiveEdgeDetector, samples: list[EmgSample]):
    events = []
    for s in samples:
        detector.update(s)
        events.extend(detector.pop_events())
    return events


def test_blip_policy_noise_uses_refractory_only():
    detector = AdaptiveEdgeDetector(
        min_event_ms=120,
        refractory_ms=90,
        min_rest_gap_ms=260,
    )
    detector._stable_state = "light"
    detector._press_start_ms = 100
    detector._press_class = "light"
    detector._press_peak_z = detector.t_high_enter - 0.1
    detector._commit_up(180)  # short blip (< min_event_ms)
    assert detector._next_press_after_ms == 180 + detector.refractory_ms
    assert detector._refractory_until_ms == 180 + detector.refractory_ms


def test_blip_policy_real_short_enforces_rest_gap():
    detector = AdaptiveEdgeDetector(
        min_event_ms=120,
        refractory_ms=90,
        min_rest_gap_ms=260,
    )

    detector._stable_state = "heavy"
    detector._press_start_ms = 100
    detector._press_class = "heavy"
    detector._press_peak_z = detector.t_high_enter + 0.5
    detector._commit_up(180)  # short but looked real
    assert detector._next_press_after_ms == 180 + detector.refractory_ms + detector.min_rest_gap_ms
    assert detector._refractory_until_ms == 180 + detector.refractory_ms


def test_e2e_real_short_press_blocks_second_down_until_rest_gap():
    detector = AdaptiveEdgeDetector(
        min_dwell_ms=40,
        min_event_ms=120,
        refractory_ms=90,
        min_rest_gap_ms=260,
        tau_fast_ms=10,
        tau_slow_ms=300,
    )

    samples = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 2200, 0),
        EmgSample(320, 2200, 0),  # first down
        EmgSample(360, 1000, 0),
        EmgSample(420, 1000, 0),  # short release => looked real short press
        EmgSample(520, 1000, 0),
        EmgSample(620, 1000, 0),
        EmgSample(700, 2200, 0),
        EmgSample(760, 2200, 0),  # within rest-gap window, should be blocked
        EmgSample(1040, 1000, 0),
        EmgSample(1140, 1000, 0),
        EmgSample(1240, 1000, 0),
        EmgSample(1360, 2200, 0),
        EmgSample(1440, 2200, 0),  # after rest-gap, second down
    ]
    events = _collect(detector, samples)
    assert sum(1 for e in events if e.phase == "down") == 2
