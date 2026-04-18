from __future__ import annotations

from reson.types import EdgeEvent, SwitchEvent


def edge_event_to_switch_event(event: EdgeEvent) -> SwitchEvent | None:
    """Collapse detector press lifecycle events into binary switch events."""

    if event.phase == "down" and event.state != "rest":
        return SwitchEvent(
            phase="down",
            t_ms=event.start_ms,
            duration_ms=0,
            source_state="active",
        )
    if event.phase == "up" and event.state != "rest":
        return SwitchEvent(
            phase="up",
            t_ms=event.end_ms,
            duration_ms=event.duration_ms,
            source_state="active",
        )
    return None


def edge_events_to_switch_events(events: list[EdgeEvent]) -> list[SwitchEvent]:
    out: list[SwitchEvent] = []
    for event in events:
        switch_event = edge_event_to_switch_event(event)
        if switch_event is not None:
            out.append(switch_event)
    return out
