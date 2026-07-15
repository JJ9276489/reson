from pathlib import Path

import pytest

from reson.training import (
    Dataset,
    FeatureExample,
    _select_threshold_by_f1,
    evaluate_probs,
    load_interval_sessions,
    split_dataset,
    train_logreg_profile,
    train_threshold_profile,
)


def _write_session(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "features.csv").write_text(
        "\n".join(
            [
                "host_time_s,t_ms,window_start_ms,window_end_ms,env_in,filtered_raw_hp,rms_state,lf_energy_ratio,slope_burst,waveform_length",
                "0,0,-120,0,0,0,1,0.1,0.2,2",
                "0,100,-20,100,0,0,1,0.1,0.2,3",
                "0,200,80,200,0,0,8,0.2,0.3,30",
                "0,300,180,300,0,0,9,0.2,0.3,35",
                "0,500,380,500,0,0,1,0.1,0.2,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (d / "labels.jsonl").write_text(
        "\n".join(
            [
                '{"type":"session_start","t_ms":null}',
                '{"type":"label_start","label":"CLICK","t_ms":180}',
                '{"type":"label_end","label":"CLICK","t_ms":340}',
                '{"type":"session_end","t_ms":500}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return d


def test_load_interval_sessions_labels_frames(tmp_path):
    _write_session(tmp_path, "interval-001")

    dataset = load_interval_sessions(tmp_path, feature_order=["waveform_length"], ignore_margin_ms=40)

    assert [ex.label for ex in dataset.examples] == [0, 0, 1, 1, 0]


def test_train_threshold_and_logreg_profiles(tmp_path):
    _write_session(tmp_path, "interval-001")
    dataset = load_interval_sessions(tmp_path, feature_order=["waveform_length"], ignore_margin_ms=40)
    train, val = split_dataset(dataset, val_fraction=0.4, seed=1)

    threshold_profile, threshold_metrics = train_threshold_profile(train, val)
    logreg_profile, logreg_metrics = train_logreg_profile(train, val, epochs=10)

    assert threshold_profile.detector_mode == "binary"
    assert threshold_profile.model_type == "threshold"
    assert "threshold" in threshold_metrics
    assert logreg_profile.model_type == "logreg"
    assert len(logreg_profile.model["weights"]) == 1
    assert "f1" in logreg_metrics


def _threshold_fixture(rows: list[tuple[float, int]]) -> Dataset:
    return Dataset(
        examples=[
            FeatureExample(
                session="fixture",
                t_ms=index,
                features={"waveform_length": value},
                label=label,
            )
            for index, (value, label) in enumerate(rows)
        ],
        feature_order=["waveform_length"],
    )


def _legacy_brute_force_threshold(dataset: Dataset) -> float:
    values = sorted({example.features["waveform_length"] for example in dataset.examples})
    candidates = values if len(values) == 1 else [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    labels = dataset.labels()
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        probabilities = [
            1.0 if example.features["waveform_length"] >= threshold else 0.0
            for example in dataset.examples
        ]
        f1 = evaluate_probs(labels, probabilities)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def test_fast_threshold_selection_matches_legacy_brute_force():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    near_largest = float.fromhex("0x1.ffffffffffffep+1023")
    fixtures = [
        [(1.0, 0), (1.0, 0), (2.0, 1), (3.0, 1)],
        [(1.0, 1), (2.0, 0), (3.0, 1), (4.0, 0)],
        [(1.0, 0), (2.0, 0), (3.0, 0)],  # all candidates tie at F1=0
        [(7.0, 0), (7.0, 1)],  # one distinct value
        [(-3.0, 0), (-1.0, 1), (-1.0, 0), (5.0, 1), (9.0, 1)],
        [(0.0, 0), (near_largest, 1), (largest, 0)],  # midpoint overflows
        [(-largest, 0), (-near_largest, 1), (0.0, 0)],  # negative midpoint overflows
    ]

    for rows in fixtures:
        dataset = _threshold_fixture(rows)
        assert _select_threshold_by_f1(dataset, "waveform_length") == _legacy_brute_force_threshold(dataset)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_threshold_selection_rejects_non_finite_features(value):
    dataset = _threshold_fixture([(0.0, 0), (value, 1)])
    with pytest.raises(ValueError, match="non-finite"):
        _select_threshold_by_f1(dataset, "waveform_length")


def test_threshold_selection_rejects_non_binary_labels():
    dataset = _threshold_fixture([(0.0, 0), (1.0, 2)])
    with pytest.raises(ValueError, match="binary labels"):
        _select_threshold_by_f1(dataset, "waveform_length")


def test_threshold_training_rejects_selected_infinite_midpoint():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    near_largest = float.fromhex("0x1.ffffffffffffep+1023")
    dataset = _threshold_fixture([(near_largest, 0), (largest, 1)])
    with pytest.raises(ValueError, match="overflowed"):
        train_threshold_profile(dataset, Dataset([], dataset.feature_order))
