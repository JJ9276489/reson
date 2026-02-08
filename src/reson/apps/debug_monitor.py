from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque

from reson.port_lock import PortLockHandle, PortLockInUseError, acquire_port_lock
from reson.qt_runtime import configure_qt_platform_plugin_path, get_qt_plugin_dir, validate_python_runtime
from reson.serial_io import SerialConfig, SerialReader, resolve_port

validate_python_runtime()
configure_qt_platform_plugin_path()

from PySide6.QtCore import QCoreApplication, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg

from reson.parser import parse_line


class SerialWorker(QThread):
    sample_signal = Signal(object)
    bad_line_signal = Signal(str)
    line_signal = Signal()
    status_signal = Signal(str)

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.reader = SerialReader(SerialConfig(port=port, baud=baud, timeout_s=0.1))
        self._running = True

    def run(self) -> None:
        retry_delay = self.reader.config.reconnect_initial_s

        while self._running:
            if not self.reader.is_connected():
                self.status_signal.emit(f"DISCONNECTED_RETRYING (last_error={self.reader.last_error})")
                if self.reader.reconnect():
                    retry_delay = self.reader.config.reconnect_initial_s
                    self.status_signal.emit("CONNECTED")
                    continue
                self.msleep(max(int(retry_delay * 1000), 50))
                retry_delay = self.reader.next_reconnect_delay(retry_delay)
                continue

            line = self.reader.read_line()
            if line is None:
                continue

            self.line_signal.emit()
            sample = parse_line(line)
            if sample is None:
                self.bad_line_signal.emit(line.strip())
                continue
            self.sample_signal.emit(sample)

    def stop(self) -> None:
        self._running = False
        self.reader.close()


class DebugWindow(QWidget):
    def __init__(self, port: str, baud: int, window_sec: float, lock_handle: PortLockHandle):
        super().__init__()
        self.setWindowTitle("Reson Debug Monitor")
        self.window_sec = window_sec
        self.lock_handle = lock_handle
        self.max_samples = max(int(window_sec * 500), 2000)

        self.t_data: deque[float] = deque(maxlen=self.max_samples)
        self.raw_data: deque[int] = deque(maxlen=self.max_samples)
        self.env_data: deque[int] = deque(maxlen=self.max_samples)
        self.parse_errors = 0
        self.line_count = 0
        self.sample_count = 0
        self.t0 = time.time()
        self.last_parse_error_line = ""
        self.parsed_since_tick = 0
        self.connection_state = "CONNECTING"

        layout = QVBoxLayout(self)
        self.stats = QLabel("Initializing...")

        self.raw_plot = pg.PlotWidget(title="Raw")
        self.env_plot = pg.PlotWidget(title="Env")
        self.env_plot.setXLink(self.raw_plot)

        self.raw_curve = self.raw_plot.plot(pen="y")
        self.env_curve = self.env_plot.plot(pen="c")

        layout.addWidget(self.stats)
        layout.addWidget(self.raw_plot)
        layout.addWidget(self.env_plot)

        self.worker = SerialWorker(port=port, baud=baud)
        self.worker.sample_signal.connect(self._on_sample)
        self.worker.bad_line_signal.connect(self._on_bad_line)
        self.worker.line_signal.connect(self._on_line)
        self.worker.status_signal.connect(self._on_status)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _on_status(self, status: str) -> None:
        self.connection_state = status

    def _on_line(self) -> None:
        self.line_count += 1

    def _on_bad_line(self, line: str) -> None:
        self.parse_errors += 1
        self.last_parse_error_line = line

    def _on_sample(self, sample) -> None:
        self.t_data.append(sample.t_ms / 1000.0)
        self.raw_data.append(sample.raw)
        self.env_data.append(sample.env)
        self.sample_count += 1
        self.parsed_since_tick += 1

    def _tick(self) -> None:
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = self.line_count / elapsed
        parsed_this_tick = self.parsed_since_tick
        self.parsed_since_tick = 0

        if self.sample_count > 0:
            x_all = list(self.t_data)
            raw_all = list(self.raw_data)
            env_all = list(self.env_data)
            latest = x_all[-1]
            min_t = max(0.0, latest - self.window_sec)

            start_idx = 0
            for i, t_val in enumerate(x_all):
                if t_val >= min_t:
                    start_idx = i
                    break

            x = x_all[start_idx:]
            raw = raw_all[start_idx:]
            env = env_all[start_idx:]

            self.raw_curve.setData(x, raw)
            self.env_curve.setData(x, env)
            self.stats.setText(
                f"Port: {self.worker.reader.port} | state: {self.connection_state} | lines/s: {rate:.1f} | parsed/tick: {parsed_this_tick} | parse errors: {self.parse_errors} | samples: {self.sample_count}"
            )
            return

        if self.line_count == 0:
            self.stats.setText(f"Port: {self.worker.reader.port} | state: {self.connection_state} | waiting for serial data...")
            return

        tail = self.last_parse_error_line[:60]
        self.stats.setText(
            f"Port: {self.worker.reader.port} | state: {self.connection_state} | lines/s: {rate:.1f} | no valid samples yet | parse errors: {self.parse_errors} | last bad line: {tail}"
        )

    def closeEvent(self, event):  # noqa: N802
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.lock_handle.release()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson serial debug monitor")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-sec", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)

    try:
        lock_handle = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-debug] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    plugin_dir = get_qt_plugin_dir()
    if plugin_dir:
        QCoreApplication.setLibraryPaths([plugin_dir])

    import PySide6

    print(
        f"[reson-debug] python={sys.version.split()[0]} pyside6={PySide6.__version__} "
        f"port={resolved_port} baud={args.baud}",
        file=sys.stderr,
    )

    app = QApplication(sys.argv)

    def _graceful_exit(*_args):
        app.quit()

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    win = DebugWindow(port=resolved_port, baud=args.baud, window_sec=args.window_sec, lock_handle=lock_handle)
    win.resize(1000, 700)
    win.show()

    exit_code = app.exec()
    lock_handle.release()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
