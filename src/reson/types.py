from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmgSample:
    t_ms: int
    raw: int
    env: int


EdgeState = Literal["rest", "active"]
SwitchPhase = Literal["down", "up"]


@dataclass(frozen=True)
class EdgeEvent:
    state: EdgeState
    start_ms: int
    end_ms: int
    duration_ms: int
    phase: str | None = None


@dataclass(frozen=True)
class SwitchEvent:
    phase: SwitchPhase
    t_ms: int
    duration_ms: int
    source_state: str
    confidence: float | None = None

    def to_json_dict(self, *, host_time_s: float | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "switch",
            "phase": self.phase,
            "t_ms": self.t_ms,
            "duration_ms": self.duration_ms,
            "source_state": self.source_state,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if host_time_s is not None:
            payload["host_time_s"] = host_time_s
        return payload
