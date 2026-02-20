from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

from reson.calibration import load_profile
from reson.edge_detector import AdaptiveEdgeDetector, Hmm3EdgeDetector, make_detector
from reson.port_lock import PortLockHandle, PortLockInUseError, acquire_port_lock
from reson.qt_runtime import configure_qt_platform_plugin_path, get_qt_plugin_dir, validate_python_runtime
from reson.serial_io import SerialConfig, SerialReader, resolve_port

validate_python_runtime()
configure_qt_platform_plugin_path()

from PySide6.QtCore import QCoreApplication, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg

from reson.parser import parse_line
from reson.types import EmgSample


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
    CSV_FIELDS = [
        "t_ms",
        "raw",
        "env_in",
        "filtered_raw_hp",
        "rms_state",
        "lf_energy_ratio",
        "slope_burst",
        "waveform_length",
        "p_rest",
        "p_press",
        "p_artifact",
        "decoded_state",
        "phase",
        "armed",
        "artifact_gated",
        "down",
        "up",
        "press_class",
        "segment_duration_ms",
        "segment_peak_u",
        "segment_auc",
        "segment_mean_u",
        "segment_class",
    ]

    def __init__(
        self,
        port: str,
        baud: int,
        window_sec: float,
        lock_handle: PortLockHandle,
        detector_mode: str,
        log_path: Path | None,
    ):
        super().__init__()
        self.setWindowTitle("Reson Debug Monitor")
        self.window_sec = window_sec
        self.lock_handle = lock_handle
        self.max_samples = max(int(window_sec * 500), 2000)
        self.log_path = log_path
        self.log_file = None
        self.log_writer: csv.DictWriter | None = None

        profile = None
        try:
            profile = load_profile()
        except FileNotFoundError:
            profile = None
        self.detector = make_detector(detector_mode, profile)
        self.detector_mode = detector_mode

        self.t_data: deque[float] = deque(maxlen=self.max_samples)
        self.raw_data: deque[int] = deque(maxlen=self.max_samples)
        self.filtered_data: deque[float] = deque(maxlen=self.max_samples)
        self.rms_data: deque[float] = deque(maxlen=self.max_samples)
        self.lf_ratio_data: deque[float] = deque(maxlen=self.max_samples)
        self.slope_data: deque[float] = deque(maxlen=self.max_samples)
        self.waveform_data: deque[float] = deque(maxlen=self.max_samples)
        self.p_rest_data: deque[float] = deque(maxlen=self.max_samples)
        self.p_press_data: deque[float] = deque(maxlen=self.max_samples)
        self.p_art_data: deque[float] = deque(maxlen=self.max_samples)
        self.state_data: deque[int] = deque(maxlen=self.max_samples)
        self.down_data: deque[int] = deque(maxlen=self.max_samples)
        self.up_data: deque[int] = deque(maxlen=self.max_samples)

        self.parse_errors = 0
        self.line_count = 0
        self.sample_count = 0
        self.t0 = time.time()
        self.last_parse_error_line = ""
        self.parsed_since_tick = 0
        self.connection_state = "CONNECTING"
        self.last_row: dict[str, object] | None = None

        layout = QVBoxLayout(self)
        self.stats = QLabel("Initializing...")

        self.raw_plot = pg.PlotWidget(title="Raw + Filtered")
        self.feature_plot = pg.PlotWidget(title="Features")
        self.prob_plot = pg.PlotWidget(title="Emission Probabilities")
        self.state_plot = pg.PlotWidget(title="Decoded State (REST=0 PRESS=1 ARTIFACT=2)")

        self.feature_plot.setXLink(self.raw_plot)
        self.prob_plot.setXLink(self.raw_plot)
        self.state_plot.setXLink(self.raw_plot)

        self.raw_curve = self.raw_plot.plot(pen="y")
        self.filtered_curve = self.raw_plot.plot(pen=pg.mkPen("w", width=1))

        self.rms_curve = self.feature_plot.plot(pen="c", name="rms_state")
        self.lf_curve = self.feature_plot.plot(pen="m", name="lf_energy_ratio")
        self.slope_curve = self.feature_plot.plot(pen=pg.mkPen("g", width=1), name="slope_burst")
        self.waveform_curve = self.feature_plot.plot(pen=pg.mkPen("orange", width=1), name="waveform_length")

        self.p_rest_curve = self.prob_plot.plot(pen=pg.mkPen("w", width=1), name="p_rest")
        self.p_press_curve = self.prob_plot.plot(pen=pg.mkPen("c", width=1), name="p_press")
        self.p_art_curve = self.prob_plot.plot(pen=pg.mkPen("r", width=1), name="p_artifact")

        self.state_curve = self.state_plot.plot(pen="g", stepMode="left")
        self.down_curve = self.state_plot.plot(pen=None, symbol="t", symbolBrush="r")
        self.up_curve = self.state_plot.plot(pen=None, symbol="o", symbolBrush="b")

        layout.addWidget(self.stats)
        layout.addWidget(self.raw_plot)
        layout.addWidget(self.feature_plot)
        layout.addWidget(self.prob_plot)
        layout.addWidget(self.state_plot)

        if self.log_path is not None:
            self.log_file = self.log_path.open("w", encoding="utf-8", newline="")
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.CSV_FIELDS)
            self.log_writer.writeheader()

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

    def _state_code(self, decoded_state: str) -> int:
        if decoded_state == "PRESS":
            return 1
        if decoded_state == "ARTIFACT":
            return 2
        return 0

    def _extract_row(self, sample: EmgSample) -> dict[str, object]:
        state = self.detector.update(sample)
        for _ in self.detector.pop_events():
            pass

        # hmm3 detector debug path.
        if isinstance(self.detector, Hmm3EdgeDetector):
            dbg = self.detector.last_debug()
            if dbg is not None:
                return {
                    "t_ms": dbg.t_ms,
                    "raw": dbg.raw,
                    "env_in": dbg.env_in,
                    "filtered_raw_hp": dbg.filtered_raw_hp,
                    "rms_state": dbg.rms_state,
                    "lf_energy_ratio": dbg.lf_energy_ratio,
                    "slope_burst": dbg.slope_burst,
                    "waveform_length": dbg.waveform_length,
                    "p_rest": dbg.p_rest,
                    "p_press": dbg.p_press,
                    "p_artifact": dbg.p_artifact,
                    "decoded_state": dbg.decoded_state,
                    "phase": dbg.phase,
                    "armed": int(dbg.armed),
                    "artifact_gated": int(dbg.artifact_gated),
                    "down": dbg.down,
                    "up": dbg.up,
                    "press_class": dbg.press_class or "",
                    "segment_duration_ms": dbg.segment_duration_ms,
                    "segment_peak_u": dbg.segment_peak_u,
                    "segment_auc": dbg.segment_auc,
                    "segment_mean_u": dbg.segment_mean_u,
                    "segment_class": dbg.segment_class or "",
                }

        # adaptive fallback path.
        if isinstance(self.detector, AdaptiveEdgeDetector):
            dbg = self.detector.last_debug()
            if dbg is not None:
                decoded = "REST" if dbg.state == "rest" else "PRESS"
                p_rest = 1.0 if decoded == "REST" else 0.0
                p_press = 1.0 - p_rest
                p_art = 1.0 if dbg.artifact_gated else 0.0
                return {
                    "t_ms": dbg.t_ms,
                    "raw": dbg.raw,
                    "env_in": dbg.env_in,
                    "filtered_raw_hp": dbg.filtered_raw_hp,
                    "rms_state": dbg.rms_state,
                    "lf_energy_ratio": dbg.artifact_ratio,
                    "slope_burst": dbg.artifact_score,
                    "waveform_length": 0.0,
                    "p_rest": p_rest,
                    "p_press": p_press,
                    "p_artifact": p_art,
                    "decoded_state": "ARTIFACT" if dbg.artifact_gated else decoded,
                    "phase": dbg.phase,
                    "armed": int(dbg.armed),
                    "artifact_gated": int(dbg.artifact_gated),
                    "down": dbg.down,
                    "up": dbg.up,
                    "press_class": dbg.press_class or "",
                    "segment_duration_ms": 0,
                    "segment_peak_u": 0.0,
                    "segment_auc": 0.0,
                    "segment_mean_u": 0.0,
                    "segment_class": "",
                }

        # threshold fallback with minimal observability.
        decoded = "REST" if state == "rest" else "PRESS"
        return {
            "t_ms": sample.t_ms,
            "raw": sample.raw,
            "env_in": sample.env,
            "filtered_raw_hp": float(sample.raw),
            "rms_state": 0.0,
            "lf_energy_ratio": 0.0,
            "slope_burst": 0.0,
            "waveform_length": 0.0,
            "p_rest": 1.0 if decoded == "REST" else 0.0,
            "p_press": 1.0 if decoded == "PRESS" else 0.0,
            "p_artifact": 0.0,
            "decoded_state": decoded,
            "phase": "N/A",
            "armed": 0,
            "artifact_gated": 0,
            "down": 0,
            "up": 0,
            "press_class": "",
            "segment_duration_ms": 0,
            "segment_peak_u": 0.0,
            "segment_auc": 0.0,
            "segment_mean_u": 0.0,
            "segment_class": "",
        }

    def _on_sample(self, sample: EmgSample) -> None:
        row = self._extract_row(sample)
        self.last_row = row

        self.t_data.append(float(row["t_ms"]) / 1000.0)
        self.raw_data.append(int(row["raw"]))
        self.filtered_data.append(float(row["filtered_raw_hp"]))
        self.rms_data.append(float(row["rms_state"]))
        self.lf_ratio_data.append(float(row["lf_energy_ratio"]))
        self.slope_data.append(float(row["slope_burst"]))
        self.waveform_data.append(float(row["waveform_length"]))
        self.p_rest_data.append(float(row["p_rest"]))
        self.p_press_data.append(float(row["p_press"]))
        self.p_art_data.append(float(row["p_artifact"]))
        self.state_data.append(self._state_code(str(row["decoded_state"])))
        self.down_data.append(int(row["down"]))
        self.up_data.append(int(row["up"]))

        if self.log_writer is not None:
            self.log_writer.writerow(row)

        self.sample_count += 1
        self.parsed_since_tick += 1

    def _tick(self) -> None:
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = self.line_count / elapsed
        parsed_this_tick = self.parsed_since_tick
        self.parsed_since_tick = 0

        if self.sample_count > 0:
            x_all = list(self.t_data)
            latest = x_all[-1]
            min_t = max(0.0, latest - self.window_sec)
            start_idx = next((i for i, t_val in enumerate(x_all) if t_val >= min_t), 0)
            x = x_all[start_idx:]

            raw = list(self.raw_data)[start_idx:]
            filtered = list(self.filtered_data)[start_idx:]
            rms = list(self.rms_data)[start_idx:]
            lf_ratio = list(self.lf_ratio_data)[start_idx:]
            slope = list(self.slope_data)[start_idx:]
            waveform = list(self.waveform_data)[start_idx:]
            p_rest = list(self.p_rest_data)[start_idx:]
            p_press = list(self.p_press_data)[start_idx:]
            p_art = list(self.p_art_data)[start_idx:]
            state = list(self.state_data)[start_idx:]
            down = list(self.down_data)[start_idx:]
            up = list(self.up_data)[start_idx:]

            self.raw_curve.setData(x, raw)
            self.filtered_curve.setData(x, filtered)
            self.rms_curve.setData(x, rms)
            self.lf_curve.setData(x, lf_ratio)
            self.slope_curve.setData(x, slope)
            self.waveform_curve.setData(x, waveform)
            self.p_rest_curve.setData(x, p_rest)
            self.p_press_curve.setData(x, p_press)
            self.p_art_curve.setData(x, p_art)
            self.state_curve.setData(x, state)

            down_x = [tx for tx, val in zip(x, down) if val == 1]
            down_y = [1.1 for _ in down_x]
            up_x = [tx for tx, val in zip(x, up) if val == 1]
            up_y = [0.1 for _ in up_x]
            self.down_curve.setData(down_x, down_y)
            self.up_curve.setData(up_x, up_y)

            dbg_tail = ""
            if self.last_row is not None:
                dbg_tail = (
                    f" | phase={self.last_row['phase']} armed={self.last_row['armed']} "
                    f"decoded={self.last_row['decoded_state']} p_press={float(self.last_row['p_press']):.2f} "
                    f"p_art={float(self.last_row['p_artifact']):.2f}"
                )
            self.stats.setText(
                f"Port: {self.worker.reader.port} | detector={self.detector_mode} | state: {self.connection_state} "
                f"| lines/s: {rate:.1f} | parsed/tick: {parsed_this_tick} | parse errors: {self.parse_errors} "
                f"| samples: {self.sample_count}{dbg_tail}"
            )
            return

        if self.line_count == 0:
            self.stats.setText(f"Port: {self.worker.reader.port} | state: {self.connection_state} | waiting for serial data...")
            return

        tail = self.last_parse_error_line[:60]
        self.stats.setText(
            f"Port: {self.worker.reader.port} | state: {self.connection_state} | lines/s: {rate:.1f} "
            f"| no valid samples yet | parse errors: {self.parse_errors} | last bad line: {tail}"
        )

    def closeEvent(self, event):  # noqa: N802
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        if self.log_file is not None:
            self.log_file.close()
        self.lock_handle.release()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson serial debug monitor")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--detector", choices=("hmm3", "adaptive", "threshold"), default="hmm3")
    parser.add_argument("--log-file", default=None, help="Optional CSV log path for replay/tuning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    log_path = Path(args.log_file) if args.log_file else None

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

    win = DebugWindow(
        port=resolved_port,
        baud=args.baud,
        window_sec=args.window_sec,
        lock_handle=lock_handle,
        detector_mode=args.detector,
        log_path=log_path,
    )
    win.resize(1100, 900)
    win.show()

    exit_code = app.exec()
    lock_handle.release()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
