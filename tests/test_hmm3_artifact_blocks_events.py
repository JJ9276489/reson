from reson.edge_detector import make_detector

from hmm3_test_utils import make_hmm3_test_profile, stream_segment


def test_hmm3_artifact_dominant_window_emits_no_down_or_up():
    detector = make_detector("hmm3", make_hmm3_test_profile())
    samples = []
    samples += stream_segment(0, 500, 10, 0)  # REST
    samples += stream_segment(510, 1200, 10, 10)  # ARTIFACT-like high-amplitude burst

    events = []
    for sample in samples:
        detector.update(sample)
        events.extend(detector.pop_events())

    assert all(event.phase not in ("down", "up") for event in events)
    dbg = detector.last_debug()
    assert dbg is not None
    assert dbg.artifact_gated is True
