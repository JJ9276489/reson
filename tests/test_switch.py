from reson.switch import edge_event_to_switch_event, edge_events_to_switch_events
from reson.types import EdgeEvent


def test_press_down_edge_becomes_switch_down():
    switch_event = edge_event_to_switch_event(
        EdgeEvent(
            state="active",
            start_ms=100,
            end_ms=100,
            duration_ms=0,
            phase="down",
        )
    )

    assert switch_event is not None
    assert switch_event.phase == "down"
    assert switch_event.t_ms == 100
    assert switch_event.source_state == "active"


def test_press_up_edge_becomes_switch_up_with_duration():
    switch_event = edge_event_to_switch_event(
        EdgeEvent(
            state="active",
            start_ms=100,
            end_ms=240,
            duration_ms=140,
            phase="up",
        )
    )

    assert switch_event is not None
    assert switch_event.phase == "up"
    assert switch_event.t_ms == 240
    assert switch_event.duration_ms == 140
    assert switch_event.source_state == "active"


def test_rest_and_legacy_segment_edges_do_not_emit_switch_events():
    events = [
        EdgeEvent(state="rest", start_ms=0, end_ms=100, duration_ms=100),
        EdgeEvent(state="active", start_ms=100, end_ms=200, duration_ms=100),
    ]

    assert edge_events_to_switch_events(events) == []
