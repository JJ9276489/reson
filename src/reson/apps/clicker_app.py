"""reson-clicker: a small demo target you can click with your muscle.

Loads a trained binary profile and drives a click target from either live
serial or a replayed recording (`--replay sessions/prompt-gui-001`), so you can
feel a model out with or without hardware attached. Each completed press lights
the target, beeps, and bumps a click counter.

`build_parser` and argument handling stay importable without PySide6 (Qt is
imported lazily in `main`) so the CLI surface is testable in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson demo clicker: test a trained model on a click target")
    parser.add_argument("--profile", default="models/binary_profile.json", help="Trained binary profile path")
    parser.add_argument("--port", default=None, help="Serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument(
        "--replay",
        default=None,
        help="Replay a recorded session dir instead of reading serial (offline demo/testing)",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed multiplier")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from collections import deque

    from reson.binary_model import BinaryModelDetector, load_binary_profile
    from reson.clicker import ClickerEngine
    from reson.qt_runtime import configure_qt_platform_plugin_path, validate_python_runtime

    validate_python_runtime()
    configure_qt_platform_plugin_path()

    from PySide6.QtCore import Qt, QThread, QTimer, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    from reson.evaluation import read_raw_samples
    from reson.parser import parse_line
    from reson.serial_io import SerialConfig, SerialReader, resolve_port

    try:
        profile = load_binary_profile(Path(args.profile))
    except FileNotFoundError:
        print(
            f"[reson-clicker] profile not found at {args.profile}. "
            "Train one with `reson-train --sessions sessions --out models/binary_profile.json`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Validate the profile loads into a detector before opening a window.
    BinaryModelDetector(profile)

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
                    self.status_signal.emit("reconnecting…")
                    if self.reader.reconnect():
                        retry_delay = self.reader.config.reconnect_initial_s
                        self.status_signal.emit("connected")
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

    PHASE_COLOR = {"down": "#2e7d32", "up": "#1565c0", "cancel": "#9e9e9e"}

    class ClickerWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.engine = ClickerEngine(profile)
            self.enter_gate = self.engine.detector.enter_threshold
            self.exit_gate = self.engine.detector.exit_threshold
            self._events: deque = deque(maxlen=8)
            self.setWindowTitle("reson clicker")
            self.resize(520, 680)

            self.target = QLabel("READY")
            self.target.setAlignment(Qt.AlignCenter)
            self.target.setFixedHeight(260)
            self._set_target(False)

            self.count_label = QLabel("0")
            self.count_label.setAlignment(Qt.AlignCenter)
            self._count_base = "font-size: 72px; font-weight: bold;"
            self.count_label.setStyleSheet(self._count_base)
            clicks_caption = QLabel("clicks detected")
            clicks_caption.setAlignment(Qt.AlignCenter)
            clicks_caption.setStyleSheet("color: gray;")

            self.prob_bar = QProgressBar()
            self.prob_bar.setRange(0, 100)
            self.prob_bar.setTextVisible(True)
            self.prob_bar.setFixedHeight(40)
            self.prob_bar.setFormat("p = %p%")

            self.gate_label = QLabel(
                f"fires when p ≥ {self.enter_gate:.0%} · releases when p ≤ {self.exit_gate:.0%}"
            )
            self.gate_label.setAlignment(Qt.AlignCenter)
            self.gate_label.setStyleSheet("color: gray; font-size: 12px;")

            self.log_label = QLabel("")
            self.log_label.setTextFormat(Qt.RichText)
            self.log_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.log_label.setFixedHeight(150)
            self.log_label.setStyleSheet(
                "font-family: Menlo, monospace; font-size: 12px; background:#fafafa; "
                "border:1px solid #e0e0e0; border-radius:6px; padding:6px;"
            )

            self.status = QLabel("")
            self.status.setAlignment(Qt.AlignCenter)
            self.status.setStyleSheet("color: gray;")

            reset = QPushButton("Reset count")
            reset.clicked.connect(self._reset)

            layout = QVBoxLayout(self)
            layout.addWidget(self.target)
            layout.addWidget(self.count_label)
            layout.addWidget(clicks_caption)
            layout.addWidget(self.prob_bar)
            layout.addWidget(self.gate_label)
            layout.addWidget(self.log_label)
            row = QHBoxLayout()
            row.addWidget(reset)
            layout.addLayout(row)
            layout.addWidget(self.status)

            source = f"replay {args.replay}" if args.replay else f"port {resolved_port}"
            self.status.setText(f"{Path(args.profile).name} · {profile.model_type} · {source}")

            # UI refresh decoupled from sample arrival.
            self._ui_timer = QTimer(self)
            self._ui_timer.timeout.connect(self._refresh)
            self._ui_timer.start(33)

        def _set_target(self, down: bool, *, flash: bool = False) -> None:
            if down:
                bg = "#43a047" if flash else "#2e7d32"
                self.target.setText("CLICK")
                self.target.setStyleSheet(
                    f"background:{bg}; color:white; font-size:48px; font-weight:bold; border-radius:14px;"
                )
            else:
                self.target.setText("READY")
                self.target.setStyleSheet(
                    "background:#e0e0e0; color:#555; font-size:34px; border-radius:14px;"
                )

        def _flash_count(self) -> None:
            self.count_label.setStyleSheet(self._count_base + " color:#1565c0;")
            QTimer.singleShot(180, lambda: self.count_label.setStyleSheet(self._count_base))

        def on_sample(self, sample) -> None:
            update = self.engine.feed(sample)
            for event in update.events:
                self._events.appendleft((event.t_ms, event.phase))
                if event.phase == "down":
                    QApplication.beep()
                    self._set_target(True, flash=True)
                elif event.phase == "up":
                    self._flash_count()

        def _refresh(self) -> None:
            pct = int(round(self.engine.probability * 100))
            self.prob_bar.setValue(pct)
            active = self.engine.probability >= self.enter_gate
            chunk = "#2e7d32" if active else "#90a4ae"
            self.prob_bar.setStyleSheet(
                "QProgressBar{border:1px solid #cfcfcf; border-radius:6px; text-align:center;}"
                f"QProgressBar::chunk{{background:{chunk}; border-radius:5px;}}"
            )
            self._set_target(self.engine.is_down)
            self.count_label.setText(str(self.engine.click_count))
            self.log_label.setText(self._render_log())

        def _render_log(self) -> str:
            if not self._events:
                return "<span style='color:#999'>waiting for events…</span>"
            lines = []
            for t_ms, phase in self._events:
                color = PHASE_COLOR.get(phase, "#555")
                lines.append(
                    f"<span style='color:#999'>{t_ms:>8d} ms</span> "
                    f"<b style='color:{color}'>{phase.upper()}</b>"
                )
            return "<br>".join(lines)

        def _reset(self) -> None:
            self.engine.reset_counter()
            self._events.clear()
            self.count_label.setText("0")

        def set_status(self, text: str) -> None:
            self.status.setText(text)

    app = QApplication(sys.argv)
    resolved_port = None if args.replay else resolve_port(args.port)
    window = ClickerWindow()

    if args.replay:
        samples = read_raw_samples(Path(args.replay) / "raw.csv")
        if not samples:
            print(f"[reson-clicker] no samples in {args.replay}/raw.csv", file=sys.stderr)
            raise SystemExit(2)
        state = {"i": 0}
        speed = max(args.replay_speed, 0.01)

        feed_timer = QTimer(window)

        def _pump() -> None:
            # Feed samples up to the elapsed (scaled) device time each tick.
            i = state["i"]
            if i >= len(samples):
                feed_timer.stop()
                window.engine.flush()  # close any press open at end of recording
                window.set_status(f"replay finished · {window.engine.click_count} clicks")
                return
            base_t = samples[0].t_ms
            budget = i + 25  # cap per tick so the UI stays responsive
            while i < len(samples) and i < budget:
                window.on_sample(samples[i])
                i += 1
            state["i"] = i

        feed_timer.timeout.connect(_pump)
        feed_timer.start(max(int(10 / speed), 1))
        window.show()
        sys.exit(app.exec())

    worker = SerialWorker(resolved_port, args.baud)
    worker.sample_signal.connect(window.on_sample)
    worker.status_signal.connect(window.set_status)
    worker.start()
    window.show()
    try:
        sys.exit(app.exec())
    finally:
        worker.stop()
        worker.wait(1000)


if __name__ == "__main__":
    main()
