from reson.timing import MorseTiming
from reson.types import EdgeEvent


def test_dot_dash_split():
    timing = MorseTiming(initial_unit_ms=100.0, alpha=0.0)
    dot = timing.on_press(EdgeEvent(state="light", start_ms=0, end_ms=100, duration_ms=100))
    dash = timing.on_press(EdgeEvent(state="heavy", start_ms=0, end_ms=250, duration_ms=250))
    assert dot == "."
    assert dash == "-"


def test_gap_boundaries():
    timing = MorseTiming(initial_unit_ms=100.0, alpha=0.0)
    assert timing.on_rest_gap(200) == "none"
    assert timing.on_rest_gap(300) == "letter"
    assert timing.on_rest_gap(700) == "space"
