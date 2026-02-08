from __future__ import annotations

from collections.abc import Iterator

from reson.edge_detector import ThresholdEdgeDetector
from reson.morse_engine import MorseComposer
from reson.parser import parse_line
from reson.serial_io import SerialReader
from reson.types import EmgSample, MorseUpdate


def sample_stream(reader: SerialReader) -> Iterator[EmgSample]:
    for line in reader.iter_lines():
        sample = parse_line(line)
        if sample is not None:
            yield sample


def run_morse_pipeline(
    reader: SerialReader,
    detector: ThresholdEdgeDetector,
    composer: MorseComposer,
) -> Iterator[MorseUpdate]:
    for sample in sample_stream(reader):
        detector.update(sample)
        for event in detector.pop_events():
            yield composer.update(event)
