from pathlib import Path

from reson.binary_model import BinaryModelDetector, BinaryModelProfile, load_binary_profile, save_binary_profile
from reson.types import EmgSample


def test_binary_profile_round_trip(tmp_path):
    profile = BinaryModelProfile(
        model_type="threshold",
        feature_order=["waveform_length"],
        model={"feature": "waveform_length", "threshold": 10.0, "softness": 2.0},
    )
    path = tmp_path / "profile.json"

    save_binary_profile(profile, path)
    loaded = load_binary_profile(path)

    assert loaded.model_type == "threshold"
    assert loaded.feature_order == ["waveform_length"]


def test_binary_model_detector_emits_down_up_for_active_window():
    profile = BinaryModelProfile(
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
    detector = BinaryModelDetector(profile)

    t = 0
    for _ in range(20):
        detector.update(EmgSample(t, 1000, 0))
        t += 4
    for i in range(60):
        raw = 1000 if i % 2 == 0 else 1500
        detector.update(EmgSample(t, raw, 0))
        t += 4
    for _ in range(80):
        detector.update(EmgSample(t, 1000, 0))
        t += 4

    phases = [event.phase for event in detector.pop_events()]
    assert "down" in phases
    assert "up" in phases


def _profile_with_min_event(min_event_ms: int) -> BinaryModelProfile:
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
            "min_event_ms": min_event_ms,
            "refractory_ms": 0,
        },
    )


def _assert_every_down_terminated(events):
    """Invariant: each `down` is followed by exactly one terminal up/cancel."""
    open_press = False
    for event in events:
        if event.phase == "down":
            assert not open_press, "two downs without a terminal event between them"
            open_press = True
        elif event.phase in ("up", "cancel"):
            assert open_press, "terminal event without a preceding down"
            open_press = False
    assert not open_press, "stream ended with an un-terminated down"


def test_short_transient_emits_cancel_not_dangling_down():
    # A press shorter than min_event_ms must still produce a terminal event.
    detector = BinaryModelDetector(_profile_with_min_event(min_event_ms=300))

    t = 0
    for _ in range(200):  # settle the high-pass startup transient
        detector.update(EmgSample(t, 1000, 0))
        t += 4
    detector.pop_events()  # discard whatever the warmup produced

    # Brief activation: a couple of high-variance frames, then back to rest
    # long enough for the press to release naturally (well under min_event_ms).
    for i in range(6):
        detector.update(EmgSample(t, 1000 if i % 2 == 0 else 1500, 0))
        t += 4
    for _ in range(150):
        detector.update(EmgSample(t, 1000, 0))
        t += 4

    events = detector.pop_events()
    phases = [e.phase for e in events]
    assert "down" in phases, "expected a down for the transient"
    assert "up" not in phases, "transient shorter than min_event_ms must not emit up"
    assert "cancel" in phases, "transient must be terminated by a cancel"
    _assert_every_down_terminated(events)
    assert phases.count("down") == phases.count("up") + phases.count("cancel")


def test_flush_terminates_open_press():
    detector = BinaryModelDetector(_profile_with_min_event(min_event_ms=20))

    t = 0
    for _ in range(40):  # quiet
        detector.update(EmgSample(t, 1000, 0))
        t += 4
    for i in range(40):  # ramp into a sustained press and stop mid-press
        detector.update(EmgSample(t, 1000 if i % 2 == 0 else 1500, 0))
        t += 4

    events = detector.pop_events() + detector.flush(t)
    _assert_every_down_terminated(events)
