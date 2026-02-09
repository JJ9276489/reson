from __future__ import annotations

from reson.morse_map import CLEAR_BUFFER_TOKEN, FOCUS_TOGGLE_TOKEN, MORSE_TO_CHAR
from reson.timing import MorseTiming
from reson.types import EdgeEvent, FocusTarget, MorseUpdate


class MorseComposer:
    def __init__(self):
        self.focus: FocusTarget = "text"
        self.typed_text = ""
        self.symbol_buffer = ""
        self.timing = MorseTiming()

    def set_focus(self, focus: FocusTarget) -> None:
        self.focus = focus

    def _resolve_buffer(self) -> str | None:
        if not self.symbol_buffer:
            return None

        if self.symbol_buffer == FOCUS_TOGGLE_TOKEN:
            self.focus = "backspace" if self.focus == "text" else "text"
            self.symbol_buffer = ""
            return None

        if self.symbol_buffer == CLEAR_BUFFER_TOKEN:
            self.symbol_buffer = ""
            return None

        mapped = MORSE_TO_CHAR.get(self.symbol_buffer)
        if mapped is not None:
            self.typed_text += mapped
            self.symbol_buffer = ""
            return mapped

        self.symbol_buffer = ""
        return None

    def update(self, edge_event: EdgeEvent, focus: FocusTarget | None = None) -> MorseUpdate:
        if focus is not None:
            self.focus = focus

        last_resolved: str | None = None

        # DOWN events are emitted for detector visibility; symbol commit happens on release.
        if edge_event.phase == "down":
            return MorseUpdate(
                typed_text=self.typed_text,
                symbol_buffer=self.symbol_buffer,
                last_resolved=None,
                focus=self.focus,
            )

        if edge_event.state in ("light", "heavy"):
            if self.focus == "backspace":
                if self.typed_text:
                    self.typed_text = self.typed_text[:-1]
                    last_resolved = "<BS>"
            else:
                # Keep timing adaptation but map detector classes directly for v1 behavior.
                _ = self.timing.on_press(edge_event)
                symbol = "." if edge_event.state == "light" else "-"
                self.symbol_buffer += symbol
        elif edge_event.state == "rest":
            gap_action = self.timing.on_rest_gap(edge_event.duration_ms)
            if gap_action == "letter":
                last_resolved = self._resolve_buffer()
            elif gap_action == "space":
                resolved = self._resolve_buffer()
                if resolved is not None:
                    last_resolved = resolved
                if self.typed_text and not self.typed_text.endswith(" "):
                    self.typed_text += " "
                    if last_resolved is None:
                        last_resolved = " "

        return MorseUpdate(
            typed_text=self.typed_text,
            symbol_buffer=self.symbol_buffer,
            last_resolved=last_resolved,
            focus=self.focus,
        )
