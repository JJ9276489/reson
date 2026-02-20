import pytest

from reson.calibration import blend_transition_probs


def test_transition_blend_uses_locked_weights():
    prior = [
        [0.9, 0.1, 0.0],
        [0.2, 0.7, 0.1],
        [0.4, 0.1, 0.5],
    ]
    estimated = [
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ]
    blended = blend_transition_probs(prior, estimated, prior_weight=0.8)
    expected = [
        [0.84, 0.14, 0.02],
        [0.18, 0.72, 0.1],
        [0.36, 0.12, 0.52],
    ]

    for row_blended, row_expected in zip(blended, expected):
        assert sum(row_blended) == pytest.approx(1.0, abs=1e-9)
        for got, want in zip(row_blended, row_expected):
            assert got == pytest.approx(want, abs=1e-9)
