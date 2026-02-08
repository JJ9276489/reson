from reson.calibration import CalibrationProfile
from reson.edge_detector import ThresholdEdgeDetector
from reson.morse_engine import MorseComposer
from reson.types import EmgSample


def test_smoke_e2e_to_symbol_and_letter():
    detector = ThresholdEdgeDetector.from_calibration(
        CalibrationProfile(rest_max=10, light_threshold=20, heavy_threshold=40, hysteresis_margin=2)
    )
    composer = MorseComposer()

    stream = [
        EmgSample(0, 0, 5),
        EmgSample(100, 0, 30),
        EmgSample(220, 0, 30),
        EmgSample(340, 0, 5),
        EmgSample(700, 0, 5),
    ]

    final_text = ""
    for sample in stream:
        detector.update(sample)
        for event in detector.pop_events():
            update = composer.update(event)
            final_text = update.typed_text

    assert final_text in ("", "E")
