from __future__ import annotations

import argparse
import csv
import json
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from reson.calibration import default_profile, load_profile
from reson.features import FeatureFrameExtractor
from reson.parser import parse_line
from reson.port_lock import PortLockInUseError, acquire_port_lock
from reson.serial_io import SerialConfig, SerialReader, resolve_port


RAW_FIELDS = ["host_time_s", "t_ms", "raw", "env", "line"]
FEATURE_FIELDS = [
    "host_time_s",
    "t_ms",
    "window_start_ms",
    "window_end_ms",
    "env_in",
    "filtered_raw_hp",
    "rms_state",
    "lf_energy_ratio",
    "slope_burst",
    "waveform_length",
]


class TerminalKeyReader:
    def __init__(self, stream):
        self.stream = stream
        self.fd = stream.fileno()
        self.enabled = stream.isatty()
        self._old_settings = None

    def __enter__(self) -> "TerminalKeyReader":
        if self.enabled:
            self._old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([self.stream], [], [], 0)
        if not ready:
            return None
        return self.stream.read(1)


def default_session_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / stamp


def _write_jsonl(handle, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    handle.flush()


def _build_extractor(profile_path: Path) -> FeatureFrameExtractor:
    try:
        profile = load_profile(profile_path)
    except FileNotFoundError:
        profile = default_profile()

    feature_cfg = dict(profile.feature_config)
    return FeatureFrameExtractor(
        window_ms=int(feature_cfg.get("window_ms", 120)),
        hop_ms=int(feature_cfg.get("hop_ms", 30)),
        tau_baseline_ms=profile.tau_baseline_ms,
        filter_enabled=profile.filter_enabled,
        hp_hz=profile.hp_hz,
        lp_hz=profile.lp_hz,
        notch_hz=profile.notch_hz,
        notch_q=profile.notch_q,
        rest_scale_floor=profile.rest_scale_floor,
        artifact_lf_hz=profile.artifact_lf_hz,
        slope_fast_tau_ms=profile.slope_fast_tau_ms,
        slope_slow_tau_ms=profile.slope_slow_tau_ms,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record raw Reson sessions with manual click labels")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--out", default=None, help="Session output directory. Default: sessions/YYYYMMDD-HHMMSS")
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--profile", default=".reson_profile.json")
    parser.add_argument("--label-key", default="c", help="Key that records a manual click mark")
    parser.add_argument("--quit-key", default="q", help="Key that stops recording")
    parser.add_argument("--label", default="CLICK", help="Label written when label-key is pressed")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--status", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    session_dir = Path(args.out) if args.out else default_session_dir(Path(args.sessions_root))
    session_dir.mkdir(parents=True, exist_ok=False)

    running = True

    def _stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        lock = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-record] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    reader: SerialReader | None = None
    started_host_s = time.time()
    last_sample_t_ms: int | None = None
    sample_count = 0
    frame_count = 0
    label_count = 0
    parse_errors = 0

    try:
        meta = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "port": resolved_port,
            "baud": args.baud,
            "serial_contract": "t raw env",
            "label_key": args.label_key,
            "quit_key": args.quit_key,
            "label": args.label,
            "notes": args.notes,
            "files": {
                "raw": "raw.csv",
                "features": "features.csv",
                "labels": "labels.jsonl",
            },
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        extractor = _build_extractor(Path(args.profile))
        reader = SerialReader(SerialConfig(port=resolved_port, baud=args.baud, timeout_s=0.1))
        retry_delay = reader.config.reconnect_initial_s
        next_status = time.monotonic() + 1.0

        with (
            (session_dir / "raw.csv").open("w", encoding="utf-8", newline="") as raw_handle,
            (session_dir / "features.csv").open("w", encoding="utf-8", newline="") as feature_handle,
            (session_dir / "labels.jsonl").open("w", encoding="utf-8") as label_handle,
            TerminalKeyReader(sys.stdin) as keys,
        ):
            raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
            feature_writer = csv.DictWriter(feature_handle, fieldnames=FEATURE_FIELDS)
            raw_writer.writeheader()
            feature_writer.writeheader()
            _write_jsonl(
                label_handle,
                {
                    "type": "session_start",
                    "host_time_s": started_host_s,
                    "t_ms": None,
                },
            )

            print(
                f"[reson-record] writing {session_dir.resolve()} label='{args.label}' "
                f"press '{args.label_key}' to mark, '{args.quit_key}' to stop",
                file=sys.stderr,
                flush=True,
            )

            while running:
                if args.duration_sec is not None and (time.time() - started_host_s) >= args.duration_sec:
                    break

                key = keys.read_key()
                if key == args.quit_key:
                    break
                if key == args.label_key:
                    label_count += 1
                    _write_jsonl(
                        label_handle,
                        {
                            "type": "manual_mark",
                            "label": args.label,
                            "host_time_s": time.time(),
                            "t_ms": last_sample_t_ms,
                        },
                    )
                    print(f"[reson-record] mark {label_count} t_ms={last_sample_t_ms}", file=sys.stderr, flush=True)

                if not reader.is_connected():
                    if reader.reconnect():
                        retry_delay = reader.config.reconnect_initial_s
                        continue
                    time.sleep(retry_delay)
                    retry_delay = reader.next_reconnect_delay(retry_delay)
                    continue

                line = reader.read_line()
                if line is None:
                    continue
                host_time_s = time.time()
                sample = parse_line(line)
                if sample is None:
                    parse_errors += 1
                    continue

                last_sample_t_ms = sample.t_ms
                sample_count += 1
                raw_writer.writerow(
                    {
                        "host_time_s": f"{host_time_s:.6f}",
                        "t_ms": sample.t_ms,
                        "raw": sample.raw,
                        "env": sample.env,
                        "line": line.strip(),
                    }
                )

                _, frames = extractor.update(sample)
                for frame in frames:
                    frame_count += 1
                    feature_writer.writerow(
                        {
                            "host_time_s": f"{host_time_s:.6f}",
                            "t_ms": frame.t_ms,
                            "window_start_ms": frame.window_start_ms,
                            "window_end_ms": frame.window_end_ms,
                            "env_in": frame.env_in,
                            "filtered_raw_hp": frame.filtered_raw_hp,
                            "rms_state": frame.rms_state,
                            "lf_energy_ratio": frame.lf_energy_ratio,
                            "slope_burst": frame.slope_burst,
                            "waveform_length": frame.waveform_length,
                        }
                    )

                if args.status and time.monotonic() >= next_status:
                    print(
                        f"[reson-record] samples={sample_count} frames={frame_count} "
                        f"labels={label_count} parse_errors={parse_errors}",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_status = time.monotonic() + 1.0

            _write_jsonl(
                label_handle,
                {
                    "type": "session_end",
                    "host_time_s": time.time(),
                    "t_ms": last_sample_t_ms,
                    "samples": sample_count,
                    "frames": frame_count,
                    "labels": label_count,
                    "parse_errors": parse_errors,
                },
            )
    finally:
        if reader is not None:
            reader.close()
        lock.release()

    print(
        f"[reson-record] complete samples={sample_count} frames={frame_count} "
        f"labels={label_count} parse_errors={parse_errors}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
