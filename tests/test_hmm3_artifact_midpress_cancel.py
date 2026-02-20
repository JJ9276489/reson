from reson.edge_detector import make_detector

from hmm3_test_utils import make_hmm3_test_profile, stream_segment


def test_hmm3_artifact_midpress_cancels_segment_without_up():
    detector = make_detector("hmm3", make_hmm3_test_profile())
    samples = []
    samples += stream_segment(0, 500, 10, 0)  # REST
    samples += stream_segment(510, 850, 10, 3)  # PRESS
    samples += stream_segment(860, 1200, 10, 10)  # ARTIFACT while press active
    samples += stream_segment(1210, 1500, 10, 0)  # REST re-arm

    events = []
    for sample in samples:
        detector.update(sample)
        events.extend(detector.pop_events())

    downs = [event for event in events if event.phase == "down"]
    ups = [event for event in events if event.phase == "up"]
    assert len(downs) == 1
    assert len(ups) == 0
