from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

from reson.calibration import load_profile
from reson.edge_detector import AdaptiveEdgeDetector, DetectorDebug, make_detector
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
        self.center_data: deque[float] = deque(maxlen=self.max_samples)
        self.scale_data: deque[float] = deque(maxlen=self.max_samples)
        self.u_data: deque[float] = deque(maxlen=self.max_samples)
        self.artifact_data: deque[float] = deque(maxlen=self.max_samples)
        self.state_data: deque[int] = deque(maxlen=self.max_samples)
        self.down_data: deque[int] = deque(maxlen=self.max_samples)
        self.up_data: deque[int] = deque(maxlen=self.max_samples)
        self.gate_data: deque[int] = deque(maxlen=self.max_samples)

        self.parse_errors = 0
        self.line_count = 0
        self.sample_count = 0
        self.t0 = time.time()
        self.last_parse_error_line = ""
        self.parsed_since_tick = 0
        self.connection_state = "CONNECTING"
        self.last_dbg: DetectorDebug | None = None

        layout = QVBoxLayout(self)
        self.stats = QLabel("Initializing...")

        self.raw_plot = pg.PlotWidget(title="Raw (+ high-passed filtered overlay)")
        self.rms_plot = pg.PlotWidget(title="RMS State + REST Center/Scale")
        self.u_plot = pg.PlotWidget(title="u (REST-normalized RMS) + thresholds")
        self.state_plot = pg.PlotWidget(title="State (rest=0 light=1 heavy=2)")

        self.rms_plot.setXLink(self.raw_plot)
        self.u_plot.setXLink(self.raw_plot)
        self.state_plot.setXLink(self.raw_plot)

        self.raw_curve = self.raw_plot.plot(pen="y")
        self.filtered_curve = self.raw_plot.plot(pen=pg.mkPen("w", width=1))
        self.rms_curve = self.rms_plot.plot(pen="c", name="rms_state")
        self.center_curve = self.rms_plot.plot(pen="m", name="rest_center")
        self.scale_curve = self.rms_plot.plot(pen=pg.mkPen("g", width=1), name="rest_scale")
        self.u_curve = self.u_plot.plot(pen="w")
        self.artifact_curve = self.u_plot.plot(pen=pg.mkPen("orange", width=1), name="artifact_score")
        self.state_curve = self.state_plot.plot(pen="g", stepMode="left")
        self.down_curve = self.state_plot.plot(pen=None, symbol="t", symbolBrush="r")
        self.up_curve = self.state_plot.plot(pen=None, symbol="o", symbolBrush="b")
        self.gate_curve = self.state_plot.plot(pen=None, symbol="x", symbolBrush="orange")

        self.light_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("orange", width=1))
        self.heavy_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("red", width=1))
        self.u_plot.addItem(self.light_line)
        self.u_plot.addItem(self.heavy_line)

        layout.addWidget(self.stats)
        layout.addWidget(self.raw_plot)
        layout.addWidget(self.rms_plot)
        layout.addWidget(self.u_plot)
        layout.addWidget(self.state_plot)

        if self.log_path is not None:
            self.log_file = self.log_path.open("w", encoding="utf-8", newline="")
            self.log_writer = csv.DictWriter(
                self.log_file,
                fieldnames=[
                    "t_ms",
                    "raw",
                    "env_in",
                    "filtered_raw_hp",
                    "rms_state",
                    "rest_center",
                    "rest_scale",
                    "u",
                    "lf_energy",
                    "artifact_ratio",
                    "artifact_score",
                    "artifact_gated",
                    "rest_confident",
                    "phase",
                    "armed",
                    "state",
                    "down",
                    "up",
                    "press_class",
                ],
            )
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

    def _extract_debug(self, sample: EmgSample) -> DetectorDebug:
        state = self.detector.update(sample)
        for _ in self.detector.pop_events():
            pass

        if isinstance(self.detector, AdaptiveEdgeDetector):
            dbg = self.detector.last_debug()
            if dbg is not None:
                return dbg

        code = {"rest": 0, "light": 1, "heavy": 2}[state]
        return DetectorDebug(
            t_ms=sample.t_ms,
            raw=sample.raw,
            env_in=sample.env,
            filtered_raw_hp=float(sample.raw),
            rms_state=0.0,
            rest_center=0.0,
            rest_scale=1.0,
            u=0.0,
            lf_energy=0.0,
            artifact_ratio=0.0,
            artifact_score=0.0,
            artifact_gated=False,
            rest_confident=False,
            phase="N/A",
            armed=False,
            state=state,
            state_code=code,
            down=0,
            up=0,
            press_class=None,
            gated_refractory=False,
            gated_rest_gap=False,
        )

    def _on_sample(self, sample: EmgSample) -> None:
        dbg = self._extract_debug(sample)
        self.last_dbg = dbg

        self.t_data.append(sample.t_ms / 1000.0)
        self.raw_data.append(sample.raw)
        self.filtered_data.append(dbg.filtered_raw_hp)
        self.rms_data.append(dbg.rms_state)
        self.center_data.append(dbg.rest_center)
        self.scale_data.append(dbg.rest_scale)
        self.u_data.append(dbg.u)
        self.artifact_data.append(dbg.artifact_score)
        self.state_data.append(dbg.state_code)
        self.down_data.append(dbg.down)
        self.up_data.append(dbg.up)
        self.gate_data.append(1 if dbg.artifact_gated else 0)

        if self.log_writer is not None:
            self.log_writer.writerow(
                {
                    "t_ms": sample.t_ms,
                    "raw": sample.raw,
                    "env_in": sample.env,
                    "filtered_raw_hp": f"{dbg.filtered_raw_hp:.6f}",
                    "rms_state": f"{dbg.rms_state:.6f}",
                    "rest_center": f"{dbg.rest_center:.6f}",
                    "rest_scale": f"{dbg.rest_scale:.6f}",
                    "u": f"{dbg.u:.6f}",
                    "lf_energy": f"{dbg.lf_energy:.6f}",
                    "artifact_ratio": f"{dbg.artifact_ratio:.6f}",
                    "artifact_score": f"{dbg.artifact_score:.6f}",
                    "artifact_gated": int(dbg.artifact_gated),
                    "rest_confident": int(dbg.rest_confident),
                    "phase": dbg.phase,
                    "armed": int(dbg.armed),
                    "state": dbg.state,
                    "down": dbg.down,
                    "up": dbg.up,
                    "press_class": dbg.press_class or "",
                }
            )

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
            center = list(self.center_data)[start_idx:]
            scale = list(self.scale_data)[start_idx:]
            u = list(self.u_data)[start_idx:]
            artifact = list(self.artifact_data)[start_idx:]
            state = list(self.state_data)[start_idx:]
            down = list(self.down_data)[start_idx:]
            up = list(self.up_data)[start_idx:]
            gate = list(self.gate_data)[start_idx:]

            self.raw_curve.setData(x, raw)
            self.filtered_curve.setData(x, filtered)
            self.rms_curve.setData(x, rms)
            self.center_curve.setData(x, center)
            self.scale_curve.setData(x, scale)
            self.u_curve.setData(x, u)
            self.artifact_curve.setData(x, artifact)
            self.state_curve.setData(x, state)

            down_x = [tx for tx, val in zip(x, down) if val == 1]
            down_y = [1.1 for _ in down_x]
            up_x = [tx for tx, val in zip(x, up) if val == 1]
            up_y = [0.1 for _ in up_x]
            gate_x = [tx for tx, val in zip(x, gate) if val == 1]
            gate_y = [2.25 for _ in gate_x]
            self.down_curve.setData(down_x, down_y)
            self.up_curve.setData(up_x, up_y)
            self.gate_curve.setData(gate_x, gate_y)

            if isinstance(self.detector, AdaptiveEdgeDetector):
                self.light_line.setValue(self.detector.u_light_enter)
                self.heavy_line.setValue(self.detector.u_heavy_enter)

            dbg_tail = ""
            if self.last_dbg is not None:
                dbg_tail = (
                    f" | u={self.last_dbg.u:.2f} rms={self.last_dbg.rms_state:.1f} "
                    f"phase={self.last_dbg.phase} armed={int(self.last_dbg.armed)} "
                    f"state={self.last_dbg.state} gate={int(self.last_dbg.artifact_gated)} "
                    f"rest_conf={int(self.last_dbg.rest_confident)} rf={int(self.last_dbg.gated_refractory)} "
                    f"gap={int(self.last_dbg.gated_rest_gap)}"
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
    parser.add_argument("--detector", choices=("adaptive", "threshold"), default="adaptive")
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
