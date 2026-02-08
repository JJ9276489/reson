from __future__ import annotations

import serial

from reson.parser import parse_line
from reson.serial_io import SerialConfig, SerialReader


class _FlakySerial:
    create_count = 0

    def __init__(self, *_, **__):
        type(self).create_count += 1
        self.is_open = True
        self._reads = [b"1 100 5\n", b"bad\n", b"2 110 6\n"]

    @property
    def in_waiting(self):
        return 1

    def readline(self):
        if not self.is_open:
            raise serial.SerialException("closed")
        if self._reads:
            return self._reads.pop(0)
        raise serial.SerialException("disconnect")

    def close(self):
        self.is_open = False


def test_disconnect_then_reconnect_resumes(monkeypatch):
    _FlakySerial.create_count = 0
    monkeypatch.setattr(serial, "Serial", _FlakySerial)

    reader = SerialReader(SerialConfig(port="/dev/cu.fake", timeout_s=0.01))

    parsed = 0
    parse_errors = 0

    for _ in range(8):
        line = reader.read_line()
        if line is None:
            if not reader.is_connected():
                reader.reconnect()
            continue
        sample = parse_line(line)
        if sample is None:
            parse_errors += 1
        else:
            parsed += 1

    assert parse_errors >= 1
    assert parsed >= 2
    assert _FlakySerial.create_count >= 2
