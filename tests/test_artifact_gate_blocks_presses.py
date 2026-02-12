from reson.edge_detector import AdaptiveEdgeDetector
from reson.types import EmgSample


def test_artifact_like_bursts_do_not_emit_press_events():
    detector = AdaptiveEdgeDetector(
        bootstrap_ms=0,
        filter_enabled=False,
        min_dwell_ms=40,
        artifact_enter=1.1,
        artifact_exit=1.0,
        artifact_holdoff_ms=200,
        u_light_enter=3.0,
        u_light_exit=2.2,
        u_heavy_enter=8.0,
        u_heavy_exit=6.0,
    )

    stream = [
        EmgSample(0, 1000, 0),
        EmgSample(80, 1000, 0),
        EmgSample(160, 1000, 0),
        EmgSample(240, 1080, 0),
        EmgSample(320, 1080, 0),
        EmgSample(400, 1000, 0),
        EmgSample(480, 1080, 0),
        EmgSample(560, 1080, 0),
    ]

    events = []
    gated_any = False
    for sample in stream:
        detector.update(sample)
        dbg = detector.last_debug()
        if dbg is not None and dbg.artifact_gated:
            gated_any = True
        events.extend(detector.pop_events())

    assert gated_any is True
    assert [e for e in events if e.phase in ("down", "up")] == []
