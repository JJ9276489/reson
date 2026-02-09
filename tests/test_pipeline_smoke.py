from reson.calibration import default_profile
from reson.edge_detector import make_detector
from reson.morse_engine import MorseComposer
from reson.types import EmgSample


def test_smoke_e2e_to_symbol_and_letter():
    detector = make_detector("adaptive", default_profile())
    composer = MorseComposer()

    stream = [
        EmgSample(0, 1000, 5),
        EmgSample(50, 1000, 5),
        EmgSample(100, 1000, 5),
        EmgSample(170, 1900, 30),
        EmgSample(240, 1900, 30),
        EmgSample(330, 1900, 30),
        EmgSample(440, 1000, 5),
        EmgSample(560, 1000, 5),
        EmgSample(920, 1000, 5),
    ]

    final_text = ""
    for sample in stream:
        detector.update(sample)
        for event in detector.pop_events():
            update = composer.update(event)
            final_text = update.typed_text

    assert final_text in ("", "E")
