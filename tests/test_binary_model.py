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
