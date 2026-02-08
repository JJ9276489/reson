from __future__ import annotations

import glob
from dataclasses import dataclass
from enum import Enum
import time

import serial


@dataclass(frozen=True)
class SerialConfig:
    port: str | None = None
    baud: int = 230400
    timeout_s: float = 0.2
    reconnect_initial_s: float = 0.25
    reconnect_max_s: float = 3.0
    reconnect_factor: float = 1.8


class SerialState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTED_RETRYING = "DISCONNECTED_RETRYING"
    FAILED = "FAILED"


def autodetect_port() -> str | None:
    patterns = [
        "/dev/tty.usb*",
        "/dev/tty.SLAB*",
        "/dev/tty.wchusbserial*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "COM*",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_port(port: str | None) -> str:
    resolved = port or autodetect_port()
    if not resolved:
        raise RuntimeError("No serial port found. Pass --port explicitly.")
    return resolved


class SerialReader:
    def __init__(self, config: SerialConfig):
        self.port = resolve_port(config.port)
        self.config = config
        self._serial: serial.Serial | None = None
        self.state: SerialState = SerialState.CONNECTING
        self.last_error: str | None = None
        self._open_serial()

    def _open_serial(self) -> bool:
        try:
            self._serial = serial.Serial(port=self.port, baudrate=self.config.baud, timeout=self.config.timeout_s)
        except serial.SerialException as exc:
            self.last_error = str(exc)
            self.state = SerialState.DISCONNECTED_RETRYING
            self._serial = None
            return False
        self.state = SerialState.CONNECTED
        self.last_error = None
        return True

    def is_connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def reconnect(self) -> bool:
        self.state = SerialState.CONNECTING
        self.close()
        return self._open_serial()

    def next_reconnect_delay(self, current_delay_s: float) -> float:
        return min(current_delay_s * self.config.reconnect_factor, self.config.reconnect_max_s)

    def read_line(self) -> str | None:
        if not self.is_connected():
            return None
        try:
            data = self._serial.readline()  # type: ignore[union-attr]
        except serial.SerialException as exc:
            self.last_error = str(exc)
            self.state = SerialState.DISCONNECTED_RETRYING
            self.close()
            return None
        if not data:
            return None
        return data.decode("utf-8", errors="replace")

    def read_line_nowait(self) -> str | None:
        if not self.is_connected():
            return None
        try:
            if self._serial.in_waiting <= 0:  # type: ignore[union-attr]
                return None
            data = self._serial.readline()  # type: ignore[union-attr]
        except serial.SerialException as exc:
            self.last_error = str(exc)
            self.state = SerialState.DISCONNECTED_RETRYING
            self.close()
            return None
        if not data:
            return None
        return data.decode("utf-8", errors="replace")

    def iter_lines(self):
        while True:
            if not self.is_connected():
                time.sleep(self.config.reconnect_initial_s)
                continue
            line = self.read_line()
            if line is None:
                continue
            yield line

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except serial.SerialException as exc:
                self.last_error = str(exc)
        self._serial = None
        if self.state != SerialState.FAILED:
            self.state = SerialState.DISCONNECTED_RETRYING
