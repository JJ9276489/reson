from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from reson.calibration import load_profile
from reson.edge_detector import AdaptiveEdgeDetector, Hmm3EdgeDetector, make_detector
from reson.port_lock import PortLockHandle, PortLockInUseError, acquire_port_lock
from reson.qt_runtime import configure_qt_platform_plugin_path, get_qt_plugin_dir, validate_python_runtime
from reson.serial_io import SerialConfig, SerialReader, resolve_port

validate_python_runtime()
configure_qt_platform_plugin_path()

from PySide6.QtCore import QCoreApplication, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
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
    RAW_RECORD_FIELDS = ["host_time_s", "t_ms", "raw", "env", "line"]
    FEATURE_RECORD_FIELDS = [
        "host_time_s",
        "t_ms",
        "env_in",
        "filtered_raw_hp",
        "rms_state",
        "lf_energy_ratio",
        "slope_burst",
        "waveform_length",
        "decoded_state",
        "down",
        "up",
    ]

    def __init__(
        self,
        port: str,
        baud: int,
        window_sec: float,
        lock_handle: PortLockHandle,
        detector_mode: str,
        feature_ablation: str,
        log_path: Path | None,
        record_dir: Path | None,
    ):
        super().__init__()
        self.setWindowTitle("Reson Debug Monitor")
        self.setFocusPolicy(Qt.StrongFocus)
        self.window_sec = window_sec
        self.lock_handle = lock_handle
        self.max_samples = max(int(window_sec * 500), 2000)
        self.log_path = log_path
        self.log_file = None
        self.log_writer: csv.DictWriter | None = None
        self.record_dir = record_dir
        self.raw_record_file = None
        self.feature_record_file = None
        self.label_record_file = None
        self.raw_record_writer: csv.DictWriter | None = None
        self.feature_record_writer: csv.DictWriter | None = None
        self.record_started_host_s = time.time()
        self.record_sample_count = 0
        self.record_feature_count = 0
        self.record_label_count = 0
        self._last_record_feature_t_ms: int | None = None

        profile = None
        try:
            profile = load_profile()
        except FileNotFoundError:
            profile = None
        self.detector = make_detector(detector_mode, profile, feature_ablation=feature_ablation)
        self.detector_mode = detector_mode
        self.feature_ablation = (
            self.detector.feature_ablation_label()
            if isinstance(self.detector, Hmm3EdgeDetector)
            else "n/a"
        )

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
        self.active_data: deque[int] = deque(maxlen=self.max_samples)
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
        self.record_status = QLabel("Recording: off")
        self.mark_btn = QPushButton("Mark Click (Space/C)")
        self.mark_btn.setEnabled(self.record_dir is not None)
        self.mark_btn.clicked.connect(self._mark_click)

        controls = QHBoxLayout()
        controls.addWidget(self.mark_btn)
        controls.addWidget(self.record_status)

        self.raw_plot = pg.PlotWidget(title="Raw ADC + Filtered")
        self.feature_plot = pg.PlotWidget(title="Binary Features")
        self.state_plot = pg.PlotWidget(title="Binary Switch (ACTIVE=1 REST=0)")

        self.feature_plot.setXLink(self.raw_plot)
        self.state_plot.setXLink(self.raw_plot)

        self.raw_curve = self.raw_plot.plot(pen="y")
        self.filtered_curve = self.raw_plot.plot(pen=pg.mkPen("w", width=1))

        self.waveform_curve = self.feature_plot.plot(pen=pg.mkPen("orange", width=2), name="waveform_length")
        self.rms_curve = self.feature_plot.plot(pen=pg.mkPen("c", width=1), name="rms_state")

        self.state_curve = self.state_plot.plot(pen="g", stepMode="left")
        self.down_curve = self.state_plot.plot(pen=None, symbol="t", symbolBrush="r")
        self.up_curve = self.state_plot.plot(pen=None, symbol="o", symbolBrush="b")

        layout.addWidget(self.stats)
        layout.addLayout(controls)
        layout.addWidget(self.raw_plot)
        layout.addWidget(self.feature_plot)
        layout.addWidget(self.state_plot)

        if self.log_path is not None:
            self.log_file = self.log_path.open("w", encoding="utf-8", newline="")
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.CSV_FIELDS)
            self.log_writer.writeheader()
        if self.record_dir is not None:
            self._open_recording(resolved_dir=self.record_dir, port=port, baud=baud)

        self.worker = SerialWorker(port=port, baud=baud)
        self.worker.sample_signal.connect(self._on_sample)
        self.worker.bad_line_signal.connect(self._on_bad_line)
        self.worker.line_signal.connect(self._on_line)
        self.worker.status_signal.connect(self._on_status)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _open_recording(self, *, resolved_dir: Path, port: str, baud: int) -> None:
        resolved_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "port": port,
            "baud": baud,
            "serial_contract": "t raw env",
            "label": "CLICK",
            "label_keys": ["space", "c"],
            "source": "reson-debug",
            "files": {
                "raw": "raw.csv",
                "features": "features.csv",
                "labels": "labels.jsonl",
            },
        }
        (resolved_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.raw_record_file = (resolved_dir / "raw.csv").open("w", encoding="utf-8", newline="")
        self.feature_record_file = (resolved_dir / "features.csv").open("w", encoding="utf-8", newline="")
        self.label_record_file = (resolved_dir / "labels.jsonl").open("w", encoding="utf-8")
        self.raw_record_writer = csv.DictWriter(self.raw_record_file, fieldnames=self.RAW_RECORD_FIELDS)
        self.feature_record_writer = csv.DictWriter(self.feature_record_file, fieldnames=self.FEATURE_RECORD_FIELDS)
        self.raw_record_writer.writeheader()
        self.feature_record_writer.writeheader()
        self._write_label_record({"type": "session_start", "host_time_s": self.record_started_host_s, "t_ms": None})
        self.record_status.setText(f"Recording: {resolved_dir}")

    def _write_label_record(self, payload: dict[str, object]) -> None:
        if self.label_record_file is None:
            return
        self.label_record_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.label_record_file.flush()

    def _mark_click(self) -> None:
        if self.record_dir is None:
            return
        self.record_label_count += 1
        self._write_label_record(
            {
                "type": "manual_mark",
                "label": "CLICK",
                "host_time_s": time.time(),
                "t_ms": int(self.last_row["t_ms"]) if self.last_row is not None else None,
            }
        )
        self.record_status.setText(f"Recording: {self.record_dir} | labels={self.record_label_count}")

    def keyPressEvent(self, event):  # noqa: N802
        if event.text().lower() == "c" or event.key() == Qt.Key_Space:
            self._mark_click()
            event.accept()
            return
        super().keyPressEvent(event)

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
        host_time_s = time.time()
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
        self.active_data.append(1 if str(row["decoded_state"]) == "PRESS" else 0)
        self.down_data.append(int(row["down"]))
        self.up_data.append(int(row["up"]))

        if self.log_writer is not None:
            self.log_writer.writerow(row)
        if self.raw_record_writer is not None:
            self.raw_record_writer.writerow(
                {
                    "host_time_s": f"{host_time_s:.6f}",
                    "t_ms": sample.t_ms,
                    "raw": sample.raw,
                    "env": sample.env,
                    "line": f"{sample.t_ms} {sample.raw} {sample.env}",
                }
            )
            self.record_sample_count += 1
        if self.feature_record_writer is not None:
            feature_t_ms = int(row["t_ms"])
            if self._last_record_feature_t_ms != feature_t_ms:
                self.feature_record_writer.writerow(
                    {
                        "host_time_s": f"{host_time_s:.6f}",
                        "t_ms": feature_t_ms,
                        "env_in": row["env_in"],
                        "filtered_raw_hp": row["filtered_raw_hp"],
                        "rms_state": row["rms_state"],
                        "lf_energy_ratio": row["lf_energy_ratio"],
                        "slope_burst": row["slope_burst"],
                        "waveform_length": row["waveform_length"],
                        "decoded_state": row["decoded_state"],
                        "down": row["down"],
                        "up": row["up"],
                    }
                )
                self._last_record_feature_t_ms = feature_t_ms
                self.record_feature_count += 1

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
            waveform = list(self.waveform_data)[start_idx:]
            active = list(self.active_data)[start_idx:]
            down = list(self.down_data)[start_idx:]
            up = list(self.up_data)[start_idx:]

            self.raw_curve.setData(x, raw)
            self.filtered_curve.setData(x, filtered)
            self.waveform_curve.setData(x, waveform)
            self.rms_curve.setData(x, rms)
            self.state_curve.setData(x, active)

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
                    f"decoded={self.last_row['decoded_state']} active={int(str(self.last_row['decoded_state']) == 'PRESS')} "
                    f"raw={int(self.last_row['raw'])} wl={float(self.last_row['waveform_length']):.1f} "
                    f"p_press={float(self.last_row['p_press']):.2f}"
                )
            self.stats.setText(
                f"Port: {self.worker.reader.port} | detector={self.detector_mode} | state: {self.connection_state} "
                f"| ablation={self.feature_ablation} "
                f"| lines/s: {rate:.1f} | parsed/tick: {parsed_this_tick} | parse errors: {self.parse_errors} "
                f"| samples: {self.sample_count}{dbg_tail}"
            )
            if self.record_dir is not None:
                self.record_status.setText(
                    f"Recording: {self.record_dir} | raw={self.record_sample_count} "
                    f"features={self.record_feature_count} labels={self.record_label_count}"
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
        if self.label_record_file is not None:
            self._write_label_record(
                {
                    "type": "session_end",
                    "host_time_s": time.time(),
                    "t_ms": int(self.last_row["t_ms"]) if self.last_row is not None else None,
                    "samples": self.record_sample_count,
                    "features": self.record_feature_count,
                    "labels": self.record_label_count,
                    "parse_errors": self.parse_errors,
                }
            )
            self.label_record_file.close()
        if self.raw_record_file is not None:
            self.raw_record_file.close()
        if self.feature_record_file is not None:
            self.feature_record_file.close()
        self.lock_handle.release()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson serial debug monitor")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--detector", choices=("hmm3", "adaptive", "threshold"), default="hmm3")
    parser.add_argument(
        "--feature-ablation",
        default="all",
        help=(
            "hmm3 emission feature subset. Presets: all, wl-only, rms-only. "
            "Custom: comma list (e.g. rms_state,waveform_length)."
        ),
    )
    parser.add_argument("--log-file", default=None, help="Optional CSV log path for replay/tuning.")
    parser.add_argument("--record-dir", default=None, help="Optional visual recording session directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    log_path = Path(args.log_file) if args.log_file else None
    record_dir = Path(args.record_dir) if args.record_dir else None

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
        f"port={resolved_port} baud={args.baud} detector={args.detector} ablation={args.feature_ablation}",
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
        feature_ablation=args.feature_ablation,
        log_path=log_path,
        record_dir=record_dir,
    )
    win.resize(1100, 900)
    win.show()

    exit_code = app.exec()
    lock_handle.release()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
