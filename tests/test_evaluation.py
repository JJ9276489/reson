from __future__ import annotations

import json

from reson.evaluation import (
    PhaseWindow,
    Press,
    aggregate_scores,
    pair_presses,
    read_phase_windows,
    score_session,
)
from reson.types import SwitchEvent


def _down(t_ms: int) -> SwitchEvent:
    return SwitchEvent(phase="down", t_ms=t_ms, duration_ms=0, source_state="active")


def _up(t_ms: int, duration_ms: int) -> SwitchEvent:
    return SwitchEvent(phase="up", t_ms=t_ms, duration_ms=duration_ms, source_state="active")


def test_pair_presses_handles_unterminated_down():
    presses = pair_presses([_down(100), _up(200, 100), _down(500)])
    assert presses == [Press(down_ms=100, up_ms=200), Press(down_ms=500, up_ms=None)]


def test_read_phase_windows_uses_next_marker_as_end(tmp_path):
    label_path = tmp_path / "labels.jsonl"
    rows = [
        {"type": "session_start", "t_ms": None},
        {"type": "prompt_phase", "phase": "REST", "t_ms": 0, "duration_s": 5.0, "label": None},
        {"type": "prompt_phase", "phase": "CLICK 1/1", "t_ms": 5000, "duration_s": 1.0, "label": "CLICK"},
        {"type": "prompt_phase", "phase": "ARTIFACT_NO_LABEL", "t_ms": 6000, "duration_s": 4.0, "label": None},
    ]
    label_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    windows = read_phase_windows(label_path)
    assert windows[0] == PhaseWindow("REST", None, 0, 5000)
    assert windows[0].is_rest and not windows[0].is_artifact
    assert windows[2].name == "ARTIFACT_NO_LABEL" and windows[2].is_artifact


def test_score_session_detects_click_and_classifies_false_downs():
    clicks = [(5000, 6000)]
    phases = [
        PhaseWindow("REST", None, 0, 5000),
        PhaseWindow("CLICK 1/1", "CLICK", 5000, 6000),
        PhaseWindow("ARTIFACT_NO_LABEL", None, 6000, 10000),
    ]
    events = [
        _down(2000), _up(2100, 100),   # false positive during rest
        _down(5050), _up(6080, 1030),  # the real click (slightly late down, late up)
        _down(7000), _up(7050, 50),    # false positive during artifact
    ]
    score = score_session("s1", events, clicks, phases)

    assert score.n_clicks == 1
    assert score.n_detected == 1
    assert score.n_missed == 0
    assert score.false_downs_rest == 1
    assert score.false_downs_artifact == 1
    assert score.down_latencies_ms == [50.0]
    assert score.up_latencies_ms == [80.0]
    assert score.duration_errors_ms == [30.0]
    assert score.rest_seconds == 5.0
    assert score.artifact_seconds == 4.0


def test_aggregate_scores_computes_per_minute_rates():
    clicks = [(5000, 6000)]
    phases = [
        PhaseWindow("REST", None, 0, 5000),
        PhaseWindow("CLICK 1/1", "CLICK", 5000, 6000),
    ]
    # one false down in 5s of rest -> 12 per minute
    events = [_down(1000), _up(1100, 100), _down(5050), _up(6050, 1000)]
    score = score_session("s1", events, clicks, phases)
    agg = aggregate_scores([score])

    assert agg["detection_rate"] == 1.0
    assert agg["false_downs_rest"] == 1.0
    assert abs(agg["false_downs_rest_per_min"] - 12.0) < 1e-9


def test_missed_click_when_no_down_in_window():
    clicks = [(5000, 6000)]
    phases = [PhaseWindow("CLICK 1/1", "CLICK", 5000, 6000)]
    score = score_session("s1", [], clicks, phases)
    assert score.n_missed == 1
    assert score.n_detected == 0
