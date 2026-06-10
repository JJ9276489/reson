from __future__ import annotations

from reson.binary_model import BinaryModelProfile
from reson.clicker import ClickerEngine
from reson.types import EmgSample


def _threshold_profile() -> BinaryModelProfile:
    return BinaryModelProfile(
        model_type="threshold",
        feature_order=["waveform_length"],
        feature_config={"window_ms": 20, "hop_ms": 10},
        model={"feature": "waveform_length", "threshold": 20.0, "softness": 2.0},
        decision={
            "enter_threshold": 0.6,
            "exit_threshold": 0.4,
            "enter_dwell_frames": 1,
            "release_dwell_frames": 1,
            "min_event_ms": 20,
            "refractory_ms": 0,
        },
    )


def test_clicker_counts_completed_press():
    engine = ClickerEngine(_threshold_profile())

    t = 0
    # Long quiet baseline so the high-pass filter's startup transient settles
    # back to rest before we start counting.
    for _ in range(200):
        engine.feed(EmgSample(t, 1000, 0))
        t += 4
    assert engine.is_down is False
    engine.reset_counter()

    for i in range(60):  # active burst -> high waveform length
        engine.feed(EmgSample(t, 1000 if i % 2 == 0 else 1500, 0))
        t += 4
    assert engine.is_down is True

    for _ in range(80):  # back to rest -> release completes the click
        engine.feed(EmgSample(t, 1000, 0))
        t += 4

    assert engine.is_down is False
    assert engine.click_count == 1
    assert engine.last_click_duration_ms is not None


def test_flush_clears_open_press():
    engine = ClickerEngine(_threshold_profile())
    t = 0
    for _ in range(40):  # quiet
        engine.feed(EmgSample(t, 1000, 0))
        t += 4
    for i in range(60):  # ramp into a press and stop mid-press (no rest tail)
        engine.feed(EmgSample(t, 1000 if i % 2 == 0 else 1500, 0))
        t += 4
    assert engine.is_down is True

    engine.flush()
    assert engine.is_down is False  # stream end must not leave the target stuck down


def test_reset_counter():
    engine = ClickerEngine(_threshold_profile())
    engine.click_count = 5
    engine.reset_counter()
    assert engine.click_count == 0
