from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from reson.parser import parse_line
from reson.serial_io import SerialReader
from reson.switch import edge_events_to_switch_events
from reson.types import EdgeEvent, EdgeState, EmgSample, SwitchEvent


class SwitchDetector(Protocol):
    def update(self, sample: EmgSample) -> EdgeState: ...

    def pop_events(self) -> list[EdgeEvent]: ...

    def flush(self, final_t_ms: int) -> list[EdgeEvent]: ...


def sample_stream(reader: SerialReader) -> Iterator[EmgSample]:
    for line in reader.iter_lines():
        sample = parse_line(line)
        if sample is not None:
            yield sample


def run_switch_pipeline(
    reader: SerialReader,
    detector: SwitchDetector,
) -> Iterator[SwitchEvent]:
    for sample in sample_stream(reader):
        detector.update(sample)
        for switch_event in edge_events_to_switch_events(detector.pop_events()):
            yield switch_event
