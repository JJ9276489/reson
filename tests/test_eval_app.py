from __future__ import annotations

import pytest

from reson.apps.clicker_app import build_parser as build_clicker_parser
from reson.apps.eval_app import _parse_configs, _parse_exclude, build_parser as build_eval_parser
from reson.evaluation import list_session_dirs


def test_eval_parser_defaults():
    args = build_eval_parser().parse_args([])
    assert args.sessions == "sessions"
    assert args.configs == "threshold:wl,logreg:wl,logreg:all"
    assert (args.pre_tol_ms, args.post_tol_ms) == (200, 200)


def test_parse_configs():
    assert _parse_configs("threshold:wl, logreg:all") == [("threshold", "wl"), ("logreg", "all")]


def test_parse_configs_rejects_missing_colon():
    with pytest.raises(ValueError):
        _parse_configs("logreg")


def test_parse_exclude():
    assert _parse_exclude("prompt-gui-004, interval") == ("prompt-gui-004", "interval")
    assert _parse_exclude("") == ()


def test_clicker_parser_defaults():
    args = build_clicker_parser().parse_args([])
    assert args.profile == "models/binary_profile.json"
    assert args.replay is None
    assert args.replay_speed == 1.0


def _make_session(root, name):
    d = root / name
    d.mkdir()
    (d / "raw.csv").write_text("host_time_s,t_ms,raw,env,line\n", encoding="utf-8")
    (d / "labels.jsonl").write_text("", encoding="utf-8")
    return d


def test_list_session_dirs_include_glob_and_exclude(tmp_path):
    _make_session(tmp_path, "prompt-gui-001")
    _make_session(tmp_path, "prompt-gui-004")
    _make_session(tmp_path, "interval-001")
    _make_session(tmp_path, "prompt-gui-005-bad-20260610")

    names = [p.name for p in list_session_dirs(tmp_path, include_glob="prompt-gui-*", exclude=("prompt-gui-004",))]
    assert names == ["prompt-gui-001"]  # 004 excluded, bad skipped, interval filtered by glob
