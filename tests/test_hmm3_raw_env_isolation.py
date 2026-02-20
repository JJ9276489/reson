from reson.edge_detector import make_detector
from reson.types import EmgSample

from hmm3_test_utils import make_hmm3_test_profile


def test_hmm3_outputs_unchanged_when_only_env_in_changes():
    profile = make_hmm3_test_profile()
    detector_a = make_detector("hmm3", profile)
    detector_b = make_detector("hmm3", profile)

    stream_a = [EmgSample(t_ms=t, raw=3, env=0) for t in range(0, 1200, 10)]
    stream_b = [EmgSample(t_ms=t, raw=3, env=(t % 200)) for t in range(0, 1200, 10)]

    out_a = []
    out_b = []
    for sample in stream_a:
        out_a.append(detector_a.update(sample))
        out_a.extend(f"evt:{e.phase}:{e.state}" for e in detector_a.pop_events())
    for sample in stream_b:
        out_b.append(detector_b.update(sample))
        out_b.extend(f"evt:{e.phase}:{e.state}" for e in detector_b.pop_events())

    assert out_a == out_b
