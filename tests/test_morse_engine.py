from reson.morse_engine import MorseComposer
from reson.types import EdgeEvent


def _evt(state: str, dur: int, start: int = 0):
    return EdgeEvent(state=state, start_ms=start, end_ms=start + dur, duration_ms=dur)


def test_resolves_letter_e():
    composer = MorseComposer()
    composer.update(_evt("light", 100))
    out = composer.update(_evt("rest", 600))
    assert out.typed_text == "E"


def test_backspace_focus_deletes():
    composer = MorseComposer()
    composer.typed_text = "ABC"
    composer.set_focus("backspace")
    out = composer.update(_evt("light", 100))
    assert out.typed_text == "AB"


def test_focus_toggle_token():
    composer = MorseComposer()
    for dur in [100, 100, 100, 300, 100, 300]:
        composer.update(_evt("light" if dur == 100 else "heavy", dur))
    out = composer.update(_evt("rest", 800))
    assert out.focus == "backspace"


def test_clear_buffer_token():
    composer = MorseComposer()
    for _ in range(8):
        composer.update(_evt("light", 100))
    out = composer.update(_evt("rest", 800))
    assert out.symbol_buffer == ""
