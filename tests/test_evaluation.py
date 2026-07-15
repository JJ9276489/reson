from __future__ import annotations

import json

import pytest

import reson.evaluation as evaluation_module
from reson.binary_model import BinaryModelProfile
from reson.evaluation import (
    PhaseWindow,
    Press,
    SessionScore,
    assess_candidate_acceptance,
    aggregate_scores,
    evaluate_nested_decision_selection,
    frozen_decision_grid,
    pair_presses,
    rank_sweep_rows,
    read_phase_windows,
    read_recording_bounds,
    score_session,
)
from reson.training import Dataset, FeatureExample
from reson.types import SwitchEvent


def _down(t_ms: int) -> SwitchEvent:
    return SwitchEvent(phase="down", t_ms=t_ms, duration_ms=0, source_state="active")


def _up(t_ms: int, duration_ms: int) -> SwitchEvent:
    return SwitchEvent(phase="up", t_ms=t_ms, duration_ms=duration_ms, source_state="active")


def _score(
    session: str,
    events: list[SwitchEvent],
    clicks: list[tuple[int, int]],
    phases: list[PhaseWindow],
    **kwargs,
) -> SessionScore:
    return score_session(
        session,
        events,
        clicks,
        phases,
        recording_bounds_ms=(0, 10_000),
        **kwargs,
    )


def test_pair_presses_handles_unterminated_down():
    presses = pair_presses([_down(100), _up(200, 100), _down(500)])
    assert presses == [
        Press(down_ms=100, terminal_ms=200, terminal_phase="up"),
        Press(down_ms=500, terminal_phase="unterminated"),
    ]


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
    score = _score("s1", events, clicks, phases)

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
    score = _score("s1", events, clicks, phases)
    agg = aggregate_scores([score])

    assert agg["detection_rate"] == 1.0
    assert agg["false_downs_rest"] == 1.0
    assert abs(agg["false_downs_rest_per_min"] - 12.0) < 1e-9


def _cancel(t_ms: int, duration_ms: int) -> SwitchEvent:
    return SwitchEvent(phase="cancel", t_ms=t_ms, duration_ms=duration_ms, source_state="active")


def test_cancelled_press_is_false_down_but_not_completed_click():
    clicks = [(5000, 6000)]
    phases = [
        PhaseWindow("REST", None, 0, 5000),
        PhaseWindow("CLICK 1/1", "CLICK", 5000, 6000),
    ]
    # A short transient during rest: down then cancel -> no click delivered.
    events = [_down(2000), _cancel(2030, 30), _down(5050), _up(6050, 1000)]
    score = _score("s1", events, clicks, phases)

    assert score.n_detected == 1
    assert score.false_downs_rest == 1  # the emitted down is still a false activation
    assert score.false_clicks_rest == 0  # but cancel means no completed click
    assert score.n_cancelled == 1


def test_missed_click_when_no_down_in_window():
    clicks = [(5000, 6000)]
    phases = [PhaseWindow("CLICK 1/1", "CLICK", 5000, 6000)]
    score = _score("s1", [], clicks, phases)
    assert score.n_missed == 1
    assert score.n_detected == 0


def test_press_outside_onset_window_is_not_detection_even_if_near_click_interval():
    clicks = [(5000, 6000)]
    phases = [PhaseWindow("REST", None, 0, 5000), PhaseWindow("CLICK", "CLICK", 5000, 6000)]
    score = _score("s1", [_down(4600), _up(4700, 100)], clicks, phases)

    assert score.n_detected == 0
    assert score.false_downs_rest == 1


def test_cancelled_intended_activation_is_not_delivered_or_false():
    clicks = [(5000, 6000)]
    phases = [PhaseWindow("CLICK", "CLICK", 5000, 6000)]
    score = _score("s1", [_down(5050), _cancel(5100, 50)], clicks, phases)

    assert score.n_activated == 1
    assert score.n_detected == 0
    assert score.n_missed == 1
    assert score.n_matched_cancelled == 1
    assert score.n_false_downs == 0


def test_matching_maximizes_count_then_minimizes_total_onset_error():
    clicks = [(100, 150), (200, 250)]
    events = [
        _down(0), _up(10, 10),
        _down(99), _up(109, 10),
        _down(200), _up(210, 10),
    ]
    score = _score("s1", events, clicks, [], pre_tol_ms=200, post_tol_ms=200)

    assert score.n_detected == 2
    assert sorted(score.down_latencies_ms) == [-1.0, 0.0]
    assert score.false_downs_other == 1


def test_matching_maximizes_completed_then_total_matches_in_one_assignment():
    clicks = [(1000, 1050), (1200, 1250)]
    events = [
        _down(850), _cancel(900, 50),  # can match only the first click
        _down(1100), _up(1150, 50),   # can complete either click
    ]
    score = _score("s1", events, clicks, [], pre_tol_ms=200, post_tol_ms=200)

    assert score.n_detected == 1
    assert score.n_activated == 2
    assert score.n_matched_cancelled == 1
    assert score.n_false_downs == 0


def test_phase_boundaries_belong_to_phase_starting_at_boundary():
    phases = [
        PhaseWindow("REST", None, 0, 1000),
        PhaseWindow("ARTIFACT_NO_LABEL", None, 1000, 2000),
    ]
    score = _score("s1", [_down(1000), _up(1100, 100)], [], phases)

    assert score.false_downs_rest == 0
    assert score.false_downs_artifact == 1


def test_overlapping_phase_windows_are_rejected():
    phases = [
        PhaseWindow("REST", None, 0, 1000),
        PhaseWindow("ARTIFACT_NO_LABEL", None, 999, 2000),
    ]
    with pytest.raises(ValueError, match="non-overlapping"):
        _score("s1", [], [], phases)


def test_phase_exposure_cannot_extend_beyond_short_recording():
    phases = [PhaseWindow("ARTIFACT_NO_LABEL", None, 0, 3_600_000)]

    with pytest.raises(ValueError, match="outside recorded coverage"):
        score_session(
            "short",
            [],
            [],
            phases,
            recording_bounds_ms=(0, 1_000),
        )


def test_nonfinite_phase_endpoint_is_rejected():
    phases = [PhaseWindow("REST", None, 0, float("inf"))]  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite integer timestamps"):
        score_session(
            "nonfinite",
            [],
            [],
            phases,
            recording_bounds_ms=(0, 1_000),
        )


def test_read_recording_bounds_uses_first_and_last_usable_raw_samples(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "t_ms,raw,env\n100,1,2\n104,broken,2\n108,3,4\n",
        encoding="utf-8",
    )

    assert read_recording_bounds(raw_path) == (100, 108)


def test_recording_gap_cannot_be_counted_as_continuous_exposure(tmp_path):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "t_ms,raw,env\n0,1,2\n3600000,3,4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuous exposure is not observed"):
        read_recording_bounds(raw_path)


def test_aggregate_includes_other_downs_and_reports_missing_exposure():
    score = _score("s1", [_down(1000), _up(1100, 100)], [], [])
    aggregate = aggregate_scores([score])

    assert aggregate["false_downs_other"] == 1.0
    assert aggregate["false_downs_total"] == 1.0
    assert aggregate["false_downs_rest_per_min"] is None
    assert aggregate["false_downs_artifact_per_min"] is None
    assert aggregate["false_downs_per_negative_min"] is None


def test_sweep_ranking_cannot_hide_false_downs_other():
    common = {
        "detection_rate": 1.0,
        "down_latency_ms_median": 100.0,
        "down_latency_ms_p95_abs": 150.0,
        "false_downs_rest_per_min": 0.0,
        "false_downs_artifact_per_min": 0.0,
        "enter_dwell_frames": 2.0,
        "min_event_ms": 50.0,
    }
    clean = {**common, "id": "clean", "false_downs_per_negative_min": 0.0, "false_downs_total": 0.0, "false_downs_other": 0.0}
    hidden = {**common, "id": "hidden", "false_downs_per_negative_min": 1.0, "false_downs_total": 1.0, "false_downs_other": 1.0}

    assert rank_sweep_rows([hidden, clean])[0]["id"] == "clean"


def test_sweep_ranking_does_not_treat_missing_exposure_as_zero_rate():
    common = {
        "detection_rate": 1.0,
        "down_latency_ms_median": 100.0,
        "down_latency_ms_p95_abs": 150.0,
        "false_downs_total": 0.0,
        "false_downs_other": 0.0,
        "lifecycle_faults": 0.0,
        "enter_dwell_frames": 2.0,
        "min_event_ms": 50.0,
    }
    missing = {
        **common,
        "id": "missing",
        "false_downs_per_negative_min": None,
        "false_downs_rest_per_min": None,
        "false_downs_artifact_per_min": None,
    }
    observed = {
        **common,
        "id": "observed",
        "false_downs_per_negative_min": 1.0,
        "false_downs_rest_per_min": 1.0,
        "false_downs_artifact_per_min": 1.0,
        "false_downs_total": 1.0,
    }

    assert rank_sweep_rows([missing, observed])[0]["id"] == "observed"


def test_frozen_decision_grid_contains_only_predeclared_default_and_tuned_configs():
    grid = frozen_decision_grid()
    assert len(grid) == 2
    assert (grid[0]["enter_threshold"], grid[0]["min_event_ms"]) == (0.6, 50)
    assert (grid[1]["enter_threshold"], grid[1]["min_event_ms"]) == (0.8, 200)


def _contract_score(
    session: str,
    *,
    false_downs: int,
    detected_click_indices: list[int],
    down_latencies_ms: list[float],
) -> SessionScore:
    return SessionScore(
        session=session,
        n_clicks=3,
        n_activated=len(detected_click_indices),
        n_detected=len(detected_click_indices),
        false_downs_rest=false_downs,
        rest_seconds=60.0,
        down_latencies_ms=down_latencies_ms,
        detected_click_indices=detected_click_indices,
    )


def test_frozen_acceptance_contract_passes_only_when_all_criteria_hold():
    baseline = [
        _contract_score("s1", false_downs=5, detected_click_indices=[0, 1], down_latencies_ms=[100, 110]),
        _contract_score("s2", false_downs=5, detected_click_indices=[0, 2], down_latencies_ms=[100, 110]),
    ]
    candidate = [
        _contract_score("s1", false_downs=3, detected_click_indices=[0, 1], down_latencies_ms=[120, 130]),
        _contract_score("s2", false_downs=3, detected_click_indices=[0, 2], down_latencies_ms=[120, 130]),
    ]

    result = assess_candidate_acceptance(baseline, candidate)

    assert result["passed"] is True
    assert result["criteria"] == {
        "false_down_reduction": True,
        "delivered_press_retention": True,
        "per_session_false_down_increase": True,
        "median_onset_regression": True,
    }


def test_acceptance_retention_is_per_click_not_aggregate_detection_count():
    baseline = [
        _contract_score("s1", false_downs=10, detected_click_indices=[0, 1], down_latencies_ms=[100, 110])
    ]
    candidate = [
        _contract_score("s1", false_downs=5, detected_click_indices=[0, 2], down_latencies_ms=[100, 110])
    ]

    result = assess_candidate_acceptance(baseline, candidate)

    assert result["passed"] is False
    assert result["criteria"]["delivered_press_retention"] is False
    assert result["metrics"]["lost_baseline_delivered_click_indices"] == {"s1": [1]}


def test_acceptance_rejects_candidate_specific_exposure_denominator():
    baseline = [
        _contract_score("s1", false_downs=10, detected_click_indices=[0], down_latencies_ms=[100])
    ]
    candidate_score = _contract_score(
        "s1", false_downs=5, detected_click_indices=[0], down_latencies_ms=[100]
    )
    candidate_score.rest_seconds = 120.0

    with pytest.raises(ValueError, match="exposure differs"):
        assess_candidate_acceptance(baseline, [candidate_score])


def test_nested_selection_never_uses_outer_session_to_choose_gate(tmp_path, monkeypatch):
    names = ["s1", "s2", "s3"]
    for name in names:
        session = tmp_path / name
        session.mkdir()
        (session / "raw.csv").write_text(
            "t_ms,raw,env\n0,1,1\n4,1,1\n",
            encoding="utf-8",
        )
        (session / "labels.jsonl").write_text("", encoding="utf-8")

    dataset = Dataset(
        [
            FeatureExample(name, 0, {"waveform_length": float(index + 1)}, index % 2)
            for index, name in enumerate(names)
        ],
        ["waveform_length"],
    )
    train_session_sets: list[frozenset[str]] = []

    def fake_load(*_args, **_kwargs):
        return dataset

    def fake_train(_model_name, train, **_kwargs):
        train_session_sets.append(frozenset(example.session for example in train.examples))
        return BinaryModelProfile(
            model_type="threshold",
            feature_order=["waveform_length"],
            model={"feature": "waveform_length", "threshold": 0.0, "softness": 1.0},
        )

    def fake_replay(profile, _raw_path):
        return [profile.decision]

    def fake_score(session, switch_events, _clicks, _phases, **_kwargs):
        high_gate = float(switch_events[0]["enter_threshold"]) == 0.8
        false_downs = 5 if (session == "s1" and high_gate) else 0 if (session == "s1") else 0 if high_gate else 1
        return SessionScore(
            session=session,
            n_clicks=1,
            n_activated=1,
            n_detected=1,
            false_downs_rest=false_downs,
            rest_seconds=60.0,
            down_latencies_ms=[0.0],
            detected_click_indices=[0],
        )

    monkeypatch.setattr(evaluation_module, "load_interval_sessions", fake_load)
    monkeypatch.setattr(evaluation_module, "_train_fold", fake_train)
    monkeypatch.setattr(evaluation_module, "replay_session", fake_replay)
    monkeypatch.setattr(evaluation_module, "score_session", fake_score)

    def decision(enter):
        return {
            "enter_threshold": enter,
            "exit_threshold": 0.4,
            "enter_dwell_frames": 2,
            "release_dwell_frames": 2,
            "min_event_ms": 50,
            "refractory_ms": 80,
        }

    _scores, aggregate, selections = evaluate_nested_decision_selection(
        tmp_path,
        feature_order=["waveform_length"],
        model_name="threshold",
        decision_grid=[decision(0.6), decision(0.8)],
    )

    selected = {row["outer_session"]: row["decision"]["enter_threshold"] for row in selections}
    assert selected["s1"] == 0.8  # inner s2/s3 prefer high even though outer s1 strongly disfavors it
    assert selected["s2"] == 0.6
    assert selected["s3"] == 0.6
    assert all(row["outer_session"] not in row["inner_sessions"] for row in selections)
    assert frozenset({"s2", "s3"}) in train_session_sets  # final outer-s1 fit
    assert aggregate["all_inner_gates_met"] is False  # fixture has no artifact exposure
    assert aggregate["predeclared_grid"] is False  # custom high gate differs from frozen tuned gate
    assert aggregate["acceptance_passed"] is False
