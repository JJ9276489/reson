from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time


class PortLockError(RuntimeError):
    pass


class PortLockInUseError(PortLockError):
    def __init__(self, port: str, pid: int, lock_path: Path):
        super().__init__(f"Port {port} is already in use by PID {pid} (lock: {lock_path})")
        self.port = port
        self.pid = pid
        self.lock_path = lock_path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sanitize_port(port: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", port)


@dataclass
class PortLockHandle:
    port: str
    lock_path: Path
    pid: int

    def release(self) -> None:
        if not self.lock_path.exists():
            return
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            owner = int(payload.get("pid", -1))
        except Exception:
            owner = -1
        if owner != self.pid:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return


def acquire_port_lock(port: str, lock_root: Path | None = None) -> PortLockHandle:
    root = lock_root or (Path.cwd() / ".reson_locks")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{_sanitize_port(port)}.lock"

    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            owner = int(payload.get("pid", -1))
        except Exception:
            owner = -1

        if _pid_alive(owner):
            raise PortLockInUseError(port=port, pid=owner, lock_path=lock_path)
        lock_path.unlink(missing_ok=True)

    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    pid = os.getpid()
    payload = {
        "pid": pid,
        "port": port,
        "created_at": time.time(),
        "cmd": " ".join(os.sys.argv),
    }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return PortLockHandle(port=port, lock_path=lock_path, pid=pid)
