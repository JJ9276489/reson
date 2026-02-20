from reson.features import FeatureFrameExtractor
from reson.edge_detector import make_detector
from reson.types import EmgSample
from hmm3_test_utils import make_hmm3_test_profile


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


def test_hmm3_timebase_rewind_restarts_decoding_instead_of_freezing():
    detector = make_detector("hmm3", make_hmm3_test_profile())

    for t_ms in range(0, 1200, 10):
        detector.update(EmgSample(t_ms=t_ms, raw=0, env=0))
    dbg_before = detector.last_debug()
    assert dbg_before is not None
    assert dbg_before.t_ms >= 500

    # Simulate ESP32 reconnect/reset: t_ms restarts near zero.
    for t_ms in range(0, 500, 10):
        detector.update(EmgSample(t_ms=t_ms, raw=0, env=0))

    dbg_after = detector.last_debug()
    assert dbg_after is not None
    assert dbg_after.t_ms < 500
