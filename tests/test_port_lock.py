from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reson.port_lock import PortLockInUseError, acquire_port_lock


def test_acquire_and_release(tmp_path: Path):
    handle = acquire_port_lock("/dev/cu.usbserial-0001", lock_root=tmp_path)
    assert handle.lock_path.exists()
    handle.release()
    assert not handle.lock_path.exists()


def test_second_acquire_conflicts(tmp_path: Path):
    first = acquire_port_lock("/dev/cu.usbserial-0001", lock_root=tmp_path)
    with pytest.raises(PortLockInUseError):
        acquire_port_lock("/dev/cu.usbserial-0001", lock_root=tmp_path)
    first.release()


def test_stale_lock_is_reclaimed(tmp_path: Path):
    lock_path = tmp_path / "_dev_cu_usbserial-0001.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "port": "/dev/cu.usbserial-0001"}), encoding="utf-8")

    handle = acquire_port_lock("/dev/cu.usbserial-0001", lock_root=tmp_path)
    assert handle.lock_path.exists()
    assert handle.pid == os.getpid()
    handle.release()
