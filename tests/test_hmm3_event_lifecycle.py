from reson.edge_detector import make_detector

from hmm3_test_utils import make_hmm3_test_profile, stream_segment


def test_hmm3_rest_press_rest_emits_single_down_up_pair():
    detector = make_detector("hmm3", make_hmm3_test_profile())
    samples = []
    samples += stream_segment(0, 600, 10, 0)  # REST
    samples += stream_segment(610, 1050, 10, 3)  # PRESS
    samples += stream_segment(1060, 1600, 10, 0)  # REST

    events = []
    for sample in samples:
        detector.update(sample)
        events.extend(detector.pop_events())

    downs = [event for event in events if event.phase == "down"]
    ups = [event for event in events if event.phase == "up"]

    assert len(downs) == 1
    assert len(ups) == 1
    assert downs[0].start_ms <= ups[0].end_ms
    assert downs[0].state in ("light", "heavy")
    assert ups[0].state in ("light", "heavy")
