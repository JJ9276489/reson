from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import time

from reson.binary_model import BinaryModelDetector, BinaryModelProfile, load_binary_profile
from reson.parser import parse_line
from reson.port_lock import acquire_port_lock
from reson.serial_io import SerialConfig, SerialReader, resolve_port
from reson.switch import edge_events_to_switch_events
from reson.types import EmgSample, SwitchEvent


@dataclass(frozen=True)
class SwitchUpdate:
    """One update from the EMG switch runtime.

    `events` contains only lifecycle edges (`down`, `up`, `cancel`). `is_active`
    and `probability` expose the current detector state even when no edge fired.
    """

    host_time_s: float
    t_ms: int | None
    probability: float
    is_active: bool
    events: tuple[SwitchEvent, ...]


@dataclass(frozen=True)
class ResonSerialConfig:
    """Live serial configuration for the public switch stream API."""

    port: str | None = None
    baud: int = 230400
    profile_path: Path | str = Path("models/binary_profile.json")
    timeout_s: float = 0.1


class ResonSwitch:
    """Public headless API for turning EMG samples into switch events."""

    def __init__(self, profile: BinaryModelProfile):
        self.detector = BinaryModelDetector(profile)
        self.last_t_ms: int | None = None

    @classmethod
    def from_profile(cls, profile_path: Path | str) -> "ResonSwitch":
        return cls(load_binary_profile(Path(profile_path)))

    def feed(self, sample: EmgSample, *, host_time_s: float | None = None) -> SwitchUpdate:
        state = self.detector.update(sample)
        self.last_t_ms = sample.t_ms
        events = tuple(edge_events_to_switch_events(self.detector.pop_events()))
        return SwitchUpdate(
            host_time_s=time.time() if host_time_s is None else host_time_s,
            t_ms=sample.t_ms,
            probability=self.detector.last_probability,
            is_active=state == "active",
            events=events,
        )

    def feed_line(self, line: str, *, host_time_s: float | None = None) -> SwitchUpdate | None:
        sample = parse_line(line)
        if sample is None:
            return None
        return self.feed(sample, host_time_s=host_time_s)

    def flush(self, final_t_ms: int | None = None, *, host_time_s: float | None = None) -> SwitchUpdate:
        t_ms = self.last_t_ms if final_t_ms is None else final_t_ms
        events = tuple(edge_events_to_switch_events(self.detector.flush(t_ms or 0)))
        self.last_t_ms = t_ms
        return SwitchUpdate(
            host_time_s=time.time() if host_time_s is None else host_time_s,
            t_ms=t_ms,
            probability=self.detector.last_probability,
            is_active=False,
            events=events,
        )


def iter_serial_switch_updates(
    config: ResonSerialConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> Iterator[SwitchUpdate]:
    """Yield live switch updates from serial hardware.

    The iterator owns the serial reader and port lock for its lifetime. Status is
    reported through `status_callback`; switch data remains structured updates.
    """

    stop_requested = should_stop or (lambda: False)
    profile_path = Path(config.profile_path)
    switch = ResonSwitch.from_profile(profile_path)
    resolved_port = resolve_port(config.port)
    lock = acquire_port_lock(resolved_port)
    reader: SerialReader | None = None
    try:
        reader = SerialReader(SerialConfig(port=resolved_port, baud=config.baud, timeout_s=config.timeout_s))
        retry_delay = reader.config.reconnect_initial_s
        parse_errors = 0

        if status_callback is not None:
            status_callback(f"port={resolved_port} baud={config.baud} profile={profile_path}")

        while not stop_requested():
            if not reader.is_connected():
                if status_callback is not None:
                    status_callback(f"reconnecting last_error={reader.last_error}")
                if reader.reconnect():
                    retry_delay = reader.config.reconnect_initial_s
                    if status_callback is not None:
                        status_callback("connected")
                    continue
                time.sleep(retry_delay)
                retry_delay = reader.next_reconnect_delay(retry_delay)
                continue

            line = reader.read_line()
            if line is None:
                continue
            update = switch.feed_line(line)
            if update is None:
                parse_errors += 1
                if status_callback is not None and parse_errors % 25 == 0:
                    status_callback(f"parse_errors={parse_errors}")
                continue
            yield update

        final_update = switch.flush()
        if final_update.events:
            yield final_update
    finally:
        if reader is not None:
            reader.close()
        lock.release()
