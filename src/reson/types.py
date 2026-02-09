from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmgSample:
    t_ms: int
    raw: int
    env: int


EdgeState = Literal["rest", "light", "heavy"]


@dataclass(frozen=True)
class EdgeEvent:
    state: EdgeState
    start_ms: int
    end_ms: int
    duration_ms: int
    phase: str | None = None
    press_class: EdgeState | None = None


FocusTarget = Literal["text", "backspace"]


@dataclass(frozen=True)
class MorseUpdate:
    typed_text: str
    symbol_buffer: str
    last_resolved: str | None
    focus: FocusTarget
