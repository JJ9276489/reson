from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

from reson.binary_model import BinaryModelDetector, load_binary_profile
from reson.features import FeatureFrameExtractor, FeatureSnapshot
from reson.port_lock import PortLockHandle, PortLockInUseError, acquire_port_lock
from reson.qt_runtime import configure_qt_platform_plugin_path, get_qt_plugin_dir, validate_python_runtime
from reson.recording import (
    DEBUG_FEATURE_RECORD_FIELDS,
    RAW_RECORD_FIELDS as RECORD_RAW_FIELDS,
    RecordingSession,
    build_session_meta,
)
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
    LOG_FIELDS = [
        "host_time_s",
        "t_ms",
        "raw",
        "env_in",
        "filtered_raw_hp",
        "rms_state",
        "lf_energy_ratio",
        "slope_burst",
        "waveform_length",
        "active",
        "probability",
        "down",
        "up",
    ]
    RAW_RECORD_FIELDS = RECORD_RAW_FIELDS
    FEATURE_RECORD_FIELDS = DEBUG_FEATURE_RECORD_FIELDS

    def __init__(
        self,
        port: str,
        baud: int,
        window_sec: float,
        lock_handle: PortLockHandle,
        profile_path: Path | None,
        log_path: Path | None,
        record_dir: Path | None,
    ):
        super().__init__()
        self.setWindowTitle("Reson Debug Monitor")
        self.setFocusPolicy(Qt.StrongFocus)
        self.window_sec = window_sec
        self.lock_handle = lock_handle
        self.max_samples = max(int(window_sec * 500), 2000)
        self.extractor = FeatureFrameExtractor()
        self.detector: BinaryModelDetector | None = None
        self.profile_path = profile_path
        if profile_path is not None:
            self.detector = BinaryModelDetector(load_binary_profile(profile_path))

        self.log_path = log_path
        self.log_file = None
        self.log_writer: csv.DictWriter | None = None
        self.record_dir = record_dir
        self.recording: RecordingSession | None = None
        self.record_started_host_s = time.time()
        self.record_sample_count = 0
        self.record_feature_count = 0
        self.record_label_count = 0
        self.record_label_active = False

        self.t_data: deque[float] = deque(maxlen=self.max_samples)
        self.raw_data: deque[int] = deque(maxlen=self.max_samples)
        self.filtered_data: deque[float] = deque(maxlen=self.max_samples)
        self.rms_data: deque[float] = deque(maxlen=self.max_samples)
        self.waveform_data: deque[float] = deque(maxlen=self.max_samples)
        self.active_data: deque[int] = deque(maxlen=self.max_samples)
        self.prob_data: deque[float] = deque(maxlen=self.max_samples)
        self.down_data: deque[int] = deque(maxlen=self.max_samples)
        self.up_data: deque[int] = deque(maxlen=self.max_samples)

        self.parse_errors = 0
        self.line_count = 0
        self.sample_count = 0
        self.t0 = time.time()
        self.last_parse_error_line = ""
        self.parsed_since_tick = 0
        self.connection_state = "CONNECTING"
        self.last_t_ms: int | None = None
        self.last_row: dict[str, object] | None = None

        layout = QVBoxLayout(self)
        self.stats = QLabel("Initializing...")
        self.record_status = QLabel("Recording: off")
        self.mark_btn = QPushButton("Hold Click Label (Space/C)")
        self.mark_btn.setEnabled(self.record_dir is not None)
        self.mark_btn.pressed.connect(self._start_click_label)
        self.mark_btn.released.connect(self._end_click_label)

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
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.LOG_FIELDS)
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
        meta = build_session_meta(
            port=port,
            baud=baud,
            label_mode="hold",
            label="CLICK",
            source="reson-debug",
            label_keys=["space", "c"],
        )
        self.recording = RecordingSession.create(
            resolved_dir,
            meta=meta,
            feature_fields=self.FEATURE_RECORD_FIELDS,
        )
        self._write_label_record({"type": "session_start", "host_time_s": self.record_started_host_s, "t_ms": None})
        self.record_status.setText(f"Recording: {resolved_dir}")

    def _write_label_record(self, payload: dict[str, object]) -> None:
        if self.recording is None:
            return
        self.recording.write_label(payload)

    def _current_record_t_ms(self) -> int | None:
        return self.last_t_ms

    def _start_click_label(self) -> None:
        if self.record_dir is None or self.record_label_active:
            return
        self.record_label_active = True
        self._write_label_record(
            {"type": "label_start", "label": "CLICK", "host_time_s": time.time(), "t_ms": self._current_record_t_ms()}
        )
        self.record_status.setText(f"Recording: {self.record_dir} | intervals={self.record_label_count} | ACTIVE LABEL")

    def _end_click_label(self, *, closed_by: str | None = None) -> None:
        if self.record_dir is None or not self.record_label_active:
            return
        self.record_label_active = False
        self.record_label_count += 1
        payload: dict[str, object] = {
            "type": "label_end",
            "label": "CLICK",
            "host_time_s": time.time(),
            "t_ms": self._current_record_t_ms(),
        }
        if closed_by is not None:
            payload["closed_by"] = closed_by
        self._write_label_record(payload)
        self.record_status.setText(f"Recording: {self.record_dir} | intervals={self.record_label_count}")

    def keyPressEvent(self, event):  # noqa: N802
        if event.isAutoRepeat():
            event.accept()
            return
        if event.text().lower() == "c" or event.key() == Qt.Key_Space:
            self._start_click_label()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):  # noqa: N802
        if event.isAutoRepeat():
            event.accept()
            return
        if event.text().lower() == "c" or event.key() == Qt.Key_Space:
            self._end_click_label()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _on_status(self, status: str) -> None:
        self.connection_state = status

    def _on_line(self) -> None:
        self.line_count += 1

    def _on_bad_line(self, line: str) -> None:
        self.parse_errors += 1
        self.last_parse_error_line = line

    def _row_from_frame(
        self,
        *,
        host_time_s: float,
        sample: EmgSample,
        snap: FeatureSnapshot,
        active: int,
        probability: float,
        down: int,
        up: int,
        frame_t_ms: int,
        window_start_ms: int,
        window_end_ms: int,
        filtered_raw_hp: float,
        rms_state: float,
        lf_energy_ratio: float,
        slope_burst: float,
        waveform_length: float,
    ) -> dict[str, object]:
        return {
            "host_time_s": f"{host_time_s:.6f}",
            "t_ms": frame_t_ms,
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "raw": sample.raw,
            "env_in": sample.env,
            "filtered_raw_hp": filtered_raw_hp,
            "rms_state": rms_state,
            "lf_energy_ratio": lf_energy_ratio,
            "slope_burst": slope_burst,
            "waveform_length": waveform_length,
            "active": active,
            "probability": probability,
            "down": down,
            "up": up,
        }

    def _on_sample(self, sample: EmgSample) -> None:
        host_time_s = time.time()
        self.last_t_ms = sample.t_ms
        snap, frames = self.extractor.update(sample)
        active = 0
        probability = 0.0
        down = 0
        up = 0
        if self.detector is not None:
            active = 1 if self.detector.update(sample) == "active" else 0
            probability = self.detector.last_probability
            events = self.detector.pop_events()
            down = int(any(event.phase == "down" for event in events))
            up = int(any(event.phase == "up" for event in events))

        if self.recording is not None:
            self.recording.raw_writer.writerow(
                {
                    "host_time_s": f"{host_time_s:.6f}",
                    "t_ms": sample.t_ms,
                    "raw": sample.raw,
                    "env": sample.env,
                    "line": f"{sample.t_ms} {sample.raw} {sample.env}",
                }
            )
            self.record_sample_count += 1

        if not frames:
            row = self._row_from_frame(
                host_time_s=host_time_s,
                sample=sample,
                snap=snap,
                active=active,
                probability=probability,
                down=down,
                up=up,
                frame_t_ms=sample.t_ms,
                window_start_ms=sample.t_ms,
                window_end_ms=sample.t_ms,
                filtered_raw_hp=snap.filtered_raw_hp,
                rms_state=snap.rms_state,
                lf_energy_ratio=snap.artifact_ratio,
                slope_burst=snap.slope_burst,
                waveform_length=0.0,
            )
            self._append_row(row)
        else:
            for frame in frames:
                row = self._row_from_frame(
                    host_time_s=host_time_s,
                    sample=sample,
                    snap=snap,
                    active=active,
                    probability=probability,
                    down=down,
                    up=up,
                    frame_t_ms=frame.t_ms,
                    window_start_ms=frame.window_start_ms,
                    window_end_ms=frame.window_end_ms,
                    filtered_raw_hp=frame.filtered_raw_hp,
                    rms_state=frame.rms_state,
                    lf_energy_ratio=frame.lf_energy_ratio,
                    slope_burst=frame.slope_burst,
                    waveform_length=frame.waveform_length,
                )
                self._append_row(row)
                if self.recording is not None:
                    self.recording.feature_writer.writerow(
                        {key: row[key] for key in self.FEATURE_RECORD_FIELDS}
                    )
                    self.record_feature_count += 1

        self.sample_count += 1
        self.parsed_since_tick += 1

    def _append_row(self, row: dict[str, object]) -> None:
        self.last_row = row
        self.t_data.append(float(row["t_ms"]) / 1000.0)
        self.raw_data.append(int(row["raw"]))
        self.filtered_data.append(float(row["filtered_raw_hp"]))
        self.rms_data.append(float(row["rms_state"]))
        self.waveform_data.append(float(row["waveform_length"]))
        self.active_data.append(int(row["active"]))
        self.prob_data.append(float(row["probability"]))
        self.down_data.append(int(row["down"]))
        self.up_data.append(int(row["up"]))
        if self.log_writer is not None:
            self.log_writer.writerow({key: row[key] for key in self.LOG_FIELDS})

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
            self.raw_curve.setData(x, list(self.raw_data)[start_idx:])
            self.filtered_curve.setData(x, list(self.filtered_data)[start_idx:])
            self.waveform_curve.setData(x, list(self.waveform_data)[start_idx:])
            self.rms_curve.setData(x, list(self.rms_data)[start_idx:])
            self.state_curve.setData(x, list(self.active_data)[start_idx:])

            down = list(self.down_data)[start_idx:]
            up = list(self.up_data)[start_idx:]
            down_x = [tx for tx, val in zip(x, down) if val == 1]
            up_x = [tx for tx, val in zip(x, up) if val == 1]
            self.down_curve.setData(down_x, [1.1 for _ in down_x])
            self.up_curve.setData(up_x, [0.1 for _ in up_x])

            tail = ""
            if self.last_row is not None:
                tail = (
                    f" | active={self.last_row['active']} p={float(self.last_row['probability']):.2f} "
                    f"raw={int(self.last_row['raw'])} wl={float(self.last_row['waveform_length']):.1f}"
                )
            profile = str(self.profile_path) if self.profile_path else "none"
            self.stats.setText(
                f"Port: {self.worker.reader.port} | state: {self.connection_state} | profile={profile} "
                f"| lines/s: {rate:.1f} | parsed/tick: {parsed_this_tick} | parse errors: {self.parse_errors} "
                f"| samples: {self.sample_count}{tail}"
            )
            if self.record_dir is not None:
                active_suffix = " | ACTIVE LABEL" if self.record_label_active else ""
                self.record_status.setText(
                    f"Recording: {self.record_dir} | raw={self.record_sample_count} "
                    f"features={self.record_feature_count} intervals={self.record_label_count}{active_suffix}"
                )
            return

        if self.line_count == 0:
            self.stats.setText(f"Port: {self.worker.reader.port} | state: {self.connection_state} | waiting for serial data...")
            return
        self.stats.setText(
            f"Port: {self.worker.reader.port} | state: {self.connection_state} | lines/s: {rate:.1f} "
            f"| no valid samples yet | parse errors: {self.parse_errors} | last bad line: {self.last_parse_error_line[:60]}"
        )

    def closeEvent(self, event):  # noqa: N802
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        if self.log_file is not None:
            self.log_file.close()
        if self.recording is not None:
            if self.record_label_active:
                self._end_click_label(closed_by="window_close")
            self._write_label_record(
                {
                    "type": "session_end",
                    "host_time_s": time.time(),
                    "t_ms": self.last_t_ms,
                    "samples": self.record_sample_count,
                    "features": self.record_feature_count,
                    "labels": self.record_label_count,
                    "parse_errors": self.parse_errors,
                }
            )
            self.recording.close()
            self.recording = None
        self.lock_handle.release()
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson serial debug monitor")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--profile", default=None, help="Optional trained binary profile for live active/rest overlay.")
    parser.add_argument("--log-file", default=None, help="Optional CSV log path for replay/tuning.")
    parser.add_argument("--record-dir", default=None, help="Optional visual recording session directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    log_path = Path(args.log_file) if args.log_file else None
    record_dir = Path(args.record_dir) if args.record_dir else None
    profile_path = Path(args.profile) if args.profile else None

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
        f"port={resolved_port} baud={args.baud} profile={profile_path or 'none'}",
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
        profile_path=profile_path,
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
