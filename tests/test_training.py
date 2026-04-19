from pathlib import Path

from reson.training import load_interval_sessions, split_dataset, train_logreg_profile, train_threshold_profile


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
