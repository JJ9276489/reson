from pathlib import Path

from reson.apps.record_app import build_parser, default_session_dir


def test_record_parser_defaults_to_click_label():
    args = build_parser().parse_args([])

    assert args.label_key == "c"
    assert args.quit_key == "q"
    assert args.label == "CLICK"


def test_default_session_dir_uses_sessions_root():
    path = default_session_dir(root=Path("sessions"))

    assert path.parent.name == "sessions"
    assert len(path.name) == len("YYYYMMDD-HHMMSS")
