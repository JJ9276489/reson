from pathlib import Path

from reson.apps.record_app import build_parser, default_session_dir
from reson.recording import BASE_FEATURE_RECORD_FIELDS, RecordingSession, build_session_meta


def test_record_parser_defaults_to_click_label():
    args = build_parser().parse_args([])

    assert args.label_key == "c"
    assert args.quit_key == "q"
    assert args.label == "CLICK"


def test_record_parser_describes_interval_toggle_labels():
    help_text = build_parser().format_help()

    assert "interval click labels" in help_text
    assert "toggles a click interval start/end" in help_text


def test_default_session_dir_uses_sessions_root():
    path = default_session_dir(root=Path("sessions"))

    assert path.parent.name == "sessions"
    assert len(path.name) == len("YYYYMMDD-HHMMSS")


def test_recording_session_writes_shared_schema(tmp_path):
    session_dir = tmp_path / "session"
    meta = build_session_meta(
        port="/dev/cu.test",
        baud=230400,
        label_mode="toggle",
        label="CLICK",
        source="test",
        label_key="c",
        quit_key="q",
    )

    recording = RecordingSession.create(
        session_dir,
        meta=meta,
        feature_fields=BASE_FEATURE_RECORD_FIELDS,
    )
    try:
        recording.write_label({"type": "session_start", "t_ms": None})
        recording.raw_writer.writerow(
            {
                "host_time_s": "1.0",
                "t_ms": 10,
                "raw": 2000,
                "env": 12,
                "line": "10 2000 12",
            }
        )
    finally:
        recording.close()

    assert (session_dir / "meta.json").exists()
    assert (session_dir / "raw.csv").read_text(encoding="utf-8").splitlines()[0] == "host_time_s,t_ms,raw,env,line"
    assert (session_dir / "labels.jsonl").read_text(encoding="utf-8").strip() == '{"type":"session_start","t_ms":null}'
