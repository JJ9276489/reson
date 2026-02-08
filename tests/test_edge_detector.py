from reson.calibration import CalibrationProfile
from reson.edge_detector import ThresholdEdgeDetector
from reson.types import EmgSample


def test_state_transitions_and_events():
    profile = CalibrationProfile(rest_max=10, light_threshold=20, heavy_threshold=40, hysteresis_margin=2)
    detector = ThresholdEdgeDetector.from_calibration(profile)

    samples = [
        EmgSample(0, 0, 5),
        EmgSample(50, 0, 5),
        EmgSample(100, 0, 25),
        EmgSample(190, 0, 25),
        EmgSample(300, 0, 45),
        EmgSample(390, 0, 45),
        EmgSample(520, 0, 5),
        EmgSample(620, 0, 5),
    ]

    for s in samples:
        detector.update(s)

    events = detector.pop_events()
    states = [e.state for e in events]
    assert states == ["rest", "light", "heavy"]


def test_short_spike_filtered_by_hold():
    profile = CalibrationProfile(rest_max=10, light_threshold=20, heavy_threshold=40, hysteresis_margin=2)
    detector = ThresholdEdgeDetector.from_calibration(profile)

    detector.update(EmgSample(0, 0, 5))
    detector.update(EmgSample(10, 0, 30))
    detector.update(EmgSample(30, 0, 5))

    assert detector.pop_events() == []
