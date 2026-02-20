from dataclasses import replace

import pytest

from reson.calibration import default_profile
from reson.edge_detector import make_detector


def test_hmm3_feature_hash_mismatch_raises():
    profile = default_profile()
    bad = replace(profile, feature_hash="not-the-runtime-hash")
    with pytest.raises(ValueError, match="feature hash mismatch"):
        make_detector("hmm3", bad)
