from __future__ import annotations

from pathlib import Path

import serial

from reson.serial_io import SerialConfig, SerialReader, SerialState


class _FakeSerial:
    create_count = 0
    fail_first = False

    def __init__(self, *_, **__):
        type(self).create_count += 1
        if type(self).fail_first and type(self).create_count == 1:
            raise serial.SerialException("open failed")
        self.is_open = True
        self._in_waiting = 1
        self._reads = [b"1 2 3\n", b""]

    @property
    def in_waiting(self):
        return self._in_waiting

    def readline(self):
        if not self.is_open:
            raise serial.SerialException("closed")
        if self._reads:
            data = self._reads.pop(0)
            if data:
                self._in_waiting = 1
            else:
                self._in_waiting = 0
            return data
        return b""

    def close(self):
        self.is_open = False


def test_reconnect_success_after_initial_failure(monkeypatch):
    _FakeSerial.create_count = 0
    _FakeSerial.fail_first = True
    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    reader = SerialReader(SerialConfig(port="/dev/cu.fake", timeout_s=0.01))
    assert reader.state == SerialState.DISCONNECTED_RETRYING
    assert reader.is_connected() is False

    assert reader.reconnect() is True
    assert reader.is_connected() is True
    assert reader.state == SerialState.CONNECTED


def test_idempotent_close(monkeypatch):
    _FakeSerial.create_count = 0
    _FakeSerial.fail_first = False
    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    reader = SerialReader(SerialConfig(port="/dev/cu.fake", timeout_s=0.01))
    assert reader.is_connected() is True

    reader.close()
    reader.close()
    assert reader.is_connected() is False


def test_reconnect_backoff_progression(monkeypatch):
    _FakeSerial.create_count = 0
    _FakeSerial.fail_first = False
    monkeypatch.setattr(serial, "Serial", _FakeSerial)

    cfg = SerialConfig(port="/dev/cu.fake", reconnect_initial_s=0.2, reconnect_factor=2.0, reconnect_max_s=1.0)
    reader = SerialReader(cfg)

    d1 = reader.next_reconnect_delay(0.2)
    d2 = reader.next_reconnect_delay(d1)
    d3 = reader.next_reconnect_delay(d2)

    assert d1 == 0.4
    assert d2 == 0.8
    assert d3 == 1.0
