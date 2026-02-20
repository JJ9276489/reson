from reson.features import FeatureFrameExtractor
from reson.types import EmgSample


def test_hmm3_frame_extraction_uses_timestamps_on_irregular_input():
    extractor = FeatureFrameExtractor(window_ms=120, hop_ms=30, filter_enabled=False)
    times = [0, 11, 27, 48, 79, 121, 149, 181, 214, 245, 279, 312, 347, 381]
    frame_ends: list[int] = []

    for t_ms in times:
        _, frames = extractor.update(EmgSample(t_ms=t_ms, raw=10, env=0))
        frame_ends.extend(frame.t_ms for frame in frames)

    assert frame_ends
    assert frame_ends[0] == 120
    diffs = [b - a for a, b in zip(frame_ends, frame_ends[1:])]
    assert all(diff == 30 for diff in diffs)
