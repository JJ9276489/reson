"""Headless engine behind the demo clicker.

Wraps a `BinaryModelDetector` and tracks the things a "click test" UI cares
about: current probability, whether the switch is held down, and how many
completed clicks have fired. Kept free of any GUI/serial imports so it can be
unit tested and reused by the Qt app in `apps/clicker_app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from reson.binary_model import BinaryModelDetector, BinaryModelProfile
from reson.switch import edge_events_to_switch_events
from reson.types import EmgSample, SwitchEvent


@dataclass(frozen=True)
class ClickerUpdate:
    probability: float
    is_down: bool
    click_count: int
    events: list[SwitchEvent]


class ClickerEngine:
    def __init__(self, profile: BinaryModelProfile):
        self.detector = BinaryModelDetector(profile)
        self.probability = 0.0
        self.is_down = False
        self.click_count = 0
        self.last_down_ms: int | None = None
        self.last_click_duration_ms: int | None = None

    def feed(self, sample: EmgSample) -> ClickerUpdate:
        # The detector's edge state is authoritative for "held down"; a press
        # shorter than min_event_ms emits a `down` but no `up`, so deriving
        # is_down from events alone could leave the flag stuck on.
        state = self.detector.update(sample)
        self.probability = self.detector.last_probability
        events = edge_events_to_switch_events(self.detector.pop_events())
        for event in events:
            if event.phase == "down":
                self.last_down_ms = event.t_ms
            elif event.phase == "up":
                self.click_count += 1
                self.last_click_duration_ms = event.duration_ms
        self.is_down = state == "active"
        return ClickerUpdate(
            probability=self.probability,
            is_down=self.is_down,
            click_count=self.click_count,
            events=events,
        )

    def reset_counter(self) -> None:
        self.click_count = 0
        self.last_click_duration_ms = None
