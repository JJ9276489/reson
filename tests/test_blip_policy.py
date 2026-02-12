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
        bootstrap_ms=0,
        filter_enabled=False,
    )
    detector._stable_state = "light"
    detector._press_start_ms = 100
    detector._press_class = "light"
    detector._press_peak_u = detector.u_heavy_enter - 0.1
    detector._commit_up(180)  # short blip (< min_event_ms)
    assert detector._next_press_after_ms == 180 + detector.refractory_ms
    assert detector._refractory_until_ms == 180 + detector.refractory_ms


def test_blip_policy_real_short_enforces_rest_gap():
    detector = AdaptiveEdgeDetector(
        min_event_ms=120,
        refractory_ms=90,
        min_rest_gap_ms=260,
        bootstrap_ms=0,
        filter_enabled=False,
    )

    detector._stable_state = "heavy"
    detector._press_start_ms = 100
    detector._press_class = "heavy"
    detector._press_peak_u = detector.u_heavy_enter + 0.5
    detector._commit_up(180)  # short but looked real
    assert detector._next_press_after_ms == 180 + detector.refractory_ms + detector.min_rest_gap_ms
    assert detector._refractory_until_ms == 180 + detector.refractory_ms


# End-to-end refractory/rest-gap behavior is covered in test_rest_gap_and_refractory.py.
