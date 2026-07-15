from pathlib import Path

from reson import EmgSample, ResonSerialConfig, ResonSwitch, SwitchUpdate
from reson.api import iter_serial_switch_updates
from reson.binary_model import BinaryModelProfile, save_binary_profile


def _profile_with_min_event(min_event_ms: int = 20) -> BinaryModelProfile:
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


def _feed_click_window(switch: ResonSwitch) -> list[SwitchUpdate]:
    updates: list[SwitchUpdate] = []
    t = 0
    for _ in range(20):
        updates.append(switch.feed(EmgSample(t, 1000, 0), host_time_s=1.0))
        t += 4
    for i in range(60):
        raw = 1000 if i % 2 == 0 else 1500
        updates.append(switch.feed(EmgSample(t, raw, 0), host_time_s=1.0))
        t += 4
    for _ in range(80):
        updates.append(switch.feed(EmgSample(t, 1000, 0), host_time_s=1.0))
        t += 4
    return updates


def _phases(updates: list[SwitchUpdate]) -> list[str]:
    return [event.phase for update in updates for event in update.events]


def test_reson_switch_feed_exposes_state_probability_and_events():
    switch = ResonSwitch(_profile_with_min_event())
    updates = _feed_click_window(switch)

    phases = _phases(updates)
    assert "down" in phases
    assert "up" in phases
    assert any(update.is_active for update in updates)
    assert any(update.probability > 0.5 for update in updates)
    assert all(isinstance(update.events, tuple) for update in updates)

    event_payload = next(event for update in updates for event in update.events).to_json_dict(host_time_s=1.0)
    assert event_payload["type"] == "switch"
    assert event_payload["host_time_s"] == 1.0


def test_reson_switch_feed_line_parses_valid_lines_and_ignores_invalid_lines():
    switch = ResonSwitch(_profile_with_min_event())

    assert switch.feed_line("not a sample") is None
    update = switch.feed_line("0 1000 0", host_time_s=2.0)

    assert update is not None
    assert update.t_ms == 0
    assert update.host_time_s == 2.0


def test_reson_switch_from_profile_loads_profile(tmp_path: Path):
    path = tmp_path / "profile.json"
    save_binary_profile(_profile_with_min_event(), path)

    switch = ResonSwitch.from_profile(path)
    update = switch.feed(EmgSample(0, 1000, 0))

    assert isinstance(update, SwitchUpdate)


def test_reson_switch_flush_terminates_open_press():
    switch = ResonSwitch(_profile_with_min_event(min_event_ms=20))
    updates: list[SwitchUpdate] = []
    t = 0
    for _ in range(40):
        updates.append(switch.feed(EmgSample(t, 1000, 0)))
        t += 4
    for i in range(40):
        raw = 1000 if i % 2 == 0 else 1500
        updates.append(switch.feed(EmgSample(t, raw, 0)))
        t += 4

    updates.append(switch.flush(t))
    phases = _phases(updates)

    assert "down" in phases
    assert phases[-1] in {"up", "cancel"}


def test_serial_config_is_public_api_shape():
    config = ResonSerialConfig(port="/dev/cu.usbserial-test", profile_path="models/example.json")

    assert config.baud == 230400
    assert str(config.profile_path) == "models/example.json"
    assert callable(iter_serial_switch_updates)
