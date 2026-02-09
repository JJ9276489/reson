from __future__ import annotations

import argparse
from collections import deque
import signal
import sys

from reson.port_lock import PortLockHandle, PortLockInUseError, acquire_port_lock
from reson.qt_runtime import configure_qt_platform_plugin_path, get_qt_plugin_dir, validate_python_runtime
from reson.serial_io import SerialConfig, SerialReader, resolve_port

validate_python_runtime()
configure_qt_platform_plugin_path()

from PySide6.QtCore import QCoreApplication, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from reson.calibration import CalibrationProfile, load_profile
from reson.edge_detector import EdgeDetector, make_detector
from reson.morse_engine import MorseComposer
from reson.morse_map import MORSE_TO_CHAR
from reson.parser import parse_line
from reson.types import EmgSample


class SerialWorker(QThread):
    sample_signal = Signal(object)
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

            sample = parse_line(line)
            if sample is not None:
                self.sample_signal.emit(sample)

    def stop(self) -> None:
        self._running = False
        self.reader.close()


class ResonWindow(QMainWindow):
    def __init__(self, port: str, baud: int, lock_handle: PortLockHandle, detector_mode: str):
        super().__init__()
        self.setWindowTitle("Reson Morse Input")
        self.port = port
        self.baud = baud
        self.detector_mode = detector_mode
        self.lock_handle = lock_handle

        self.detector: EdgeDetector | None = None
        self.composer = MorseComposer()
        self.log_symbols: deque[str] = deque(maxlen=64)

        self._build_ui()
        self._load_detector()

        self.worker = SerialWorker(port=port, baud=baud)
        self.worker.sample_signal.connect(self._on_sample)
        self.worker.status_signal.connect(self._on_status)
        self.worker.start()

    def _on_status(self, status: str) -> None:
        self.state_label.setText(f"State: {status}")

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        root.addWidget(self._build_dictionary_box())

        focus_row = QHBoxLayout()
        self.text_box = QLineEdit()
        self.text_box.setReadOnly(True)
        self.backspace_btn = QPushButton("Backspace Target")
        self.backspace_btn.setEnabled(False)
        focus_row.addWidget(self.text_box)
        focus_row.addWidget(self.backspace_btn)
        root.addLayout(focus_row)

        self.symbol_log = QLabel("Symbols: ")
        self.state_label = QLabel("State: waiting")
        root.addWidget(self.symbol_log)
        root.addWidget(self.state_label)

        self.setCentralWidget(central)

    def _build_dictionary_box(self) -> QGroupBox:
        box = QGroupBox("Morse Dictionary (letters, numbers, tab)")
        grid = QGridLayout(box)
        items = sorted(MORSE_TO_CHAR.items(), key=lambda kv: kv[1])
        for idx, (code, ch) in enumerate(items):
            row = idx // 8
            col = idx % 8
            label = "TAB" if ch == "\t" else ch
            grid.addWidget(QLabel(f"{label}: {code}"), row, col)
        return box

    def _load_detector(self) -> None:
        profile: CalibrationProfile | None
        try:
            profile = load_profile()
        except FileNotFoundError:
            profile = None
            QMessageBox.information(
                self,
                "Calibration",
                "No .reson_profile.json found. Using adaptive detector defaults (calibration optional).",
            )
        self.detector = make_detector(self.detector_mode, profile)

    def _on_sample(self, sample: EmgSample) -> None:
        if self.detector is None:
            return

        stable = self.detector.update(sample)
        self.state_label.setText(f"Stable state: {stable}")

        for event in self.detector.pop_events():
            update = self.composer.update(event)
            self.text_box.setText(update.typed_text)
            focus_text = "TEXT" if update.focus == "text" else "BACKSPACE"
            self.backspace_btn.setText(f"Backspace Target ({focus_text})")
            if event.state in ("light", "heavy"):
                self.log_symbols.append("." if event.state == "light" else "-")
            if update.last_resolved is not None:
                self.log_symbols.clear()
            self.symbol_log.setText(f"Symbols: {''.join(self.log_symbols)}")

    def closeEvent(self, event):  # noqa: N802
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.lock_handle.release()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson GUI")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--detector", choices=("adaptive", "threshold"), default="adaptive")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)

    try:
        lock_handle = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-gui] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    plugin_dir = get_qt_plugin_dir()
    if plugin_dir:
        QCoreApplication.setLibraryPaths([plugin_dir])

    import PySide6

    print(
        f"[reson-gui] python={sys.version.split()[0]} pyside6={PySide6.__version__} "
        f"port={resolved_port} baud={args.baud}",
        file=sys.stderr,
    )

    app = QApplication(sys.argv)

    def _graceful_exit(*_args):
        app.quit()

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    win = ResonWindow(port=resolved_port, baud=args.baud, lock_handle=lock_handle, detector_mode=args.detector)
    win.resize(1200, 800)
    win.show()

    exit_code = app.exec()
    lock_handle.release()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
