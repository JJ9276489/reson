"""Headless engine behind the demo clicker.

Wraps a `BinaryModelDetector` and tracks the things a "click test" UI cares
about: current probability, whether the switch is held down, and how many
completed clicks have fired. Kept free of any GUI/serial imports so it can be
unit tested and reused by the Qt app in `apps/clicker_app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from reson.api import ResonSwitch
from reson.binary_model import BinaryModelProfile
from reson.types import EmgSample, SwitchEvent


@dataclass(frozen=True)
class ClickerUpdate:
    probability: float
    is_down: bool
    click_count: int
    events: list[SwitchEvent]


class ClickerEngine:
    def __init__(self, profile: BinaryModelProfile):
        self.switch = ResonSwitch(profile)
        self.detector = self.switch.detector
        self.probability = 0.0
        self.is_down = False
        self.click_count = 0
        self.last_down_ms: int | None = None
        self.last_t_ms: int = 0
        self.last_click_duration_ms: int | None = None

    def _apply(self, events: list[SwitchEvent]) -> None:
        for event in events:
            if event.phase == "down":
                self.last_down_ms = event.t_ms
            elif event.phase == "up":
                self.click_count += 1
                self.last_click_duration_ms = event.duration_ms

    def feed(self, sample: EmgSample) -> ClickerUpdate:
        update = self.switch.feed(sample)
        self.last_t_ms = sample.t_ms
        self.probability = update.probability
        events = list(update.events)
        self._apply(events)
        self.is_down = update.is_active
        return ClickerUpdate(
            probability=self.probability,
            is_down=self.is_down,
            click_count=self.click_count,
            events=events,
        )

    def flush(self) -> ClickerUpdate:
        """Close any press left open when the stream ends (e.g. replay finished)."""
        events = list(self.switch.flush(self.last_t_ms).events)
        self._apply(events)
        self.is_down = False
        return ClickerUpdate(
            probability=self.probability,
            is_down=False,
            click_count=self.click_count,
            events=events,
        )

    def reset_counter(self) -> None:
        self.click_count = 0
        self.last_click_duration_ms = None
