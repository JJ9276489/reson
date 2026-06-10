from __future__ import annotations

import pytest

from reson.apps.clicker_app import build_parser as build_clicker_parser
from reson.apps.eval_app import _parse_configs, build_parser as build_eval_parser


def test_eval_parser_defaults():
    args = build_eval_parser().parse_args([])
    assert args.sessions == "sessions"
    assert args.configs == "threshold:wl,logreg:wl,logreg:all"


def test_parse_configs():
    assert _parse_configs("threshold:wl, logreg:all") == [("threshold", "wl"), ("logreg", "all")]


def test_parse_configs_rejects_missing_colon():
    with pytest.raises(ValueError):
        _parse_configs("logreg")


def test_clicker_parser_defaults():
    args = build_clicker_parser().parse_args([])
    assert args.profile == "models/binary_profile.json"
    assert args.replay is None
    assert args.replay_speed == 1.0
