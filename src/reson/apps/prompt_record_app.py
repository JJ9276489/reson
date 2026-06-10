from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from reson.features import FeatureFrameExtractor
from reson.parser import parse_line
from reson.port_lock import PortLockInUseError, acquire_port_lock
from reson.prompt_protocol import build_protocol, phase_at, protocol_as_dicts, protocol_duration_s
from reson.recording import BASE_FEATURE_RECORD_FIELDS, RecordingSession, build_session_meta, default_session_dir
from reson.serial_io import SerialConfig, SerialReader, resolve_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a timed prompted Reson click dataset without keyboard labels")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--out", default=None, help="Session output directory. Default: sessions/YYYYMMDD-HHMMSS")
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--settle-sec", type=float, default=5.0, help="Initial no-label time after starting")
    parser.add_argument("--rest-sec", type=float, default=20.0, help="Initial rest baseline duration")
    parser.add_argument("--trials", type=int, default=20, help="Number of prompted click/clench intervals")
    parser.add_argument("--press-sec", type=float, default=1.0, help="Duration of each prompted click/clench")
    parser.add_argument("--gap-sec", type=float, default=3.0, help="Rest gap between prompted intervals")
    parser.add_argument("--final-rest-sec", type=float, default=10.0)
    parser.add_argument("--artifact-sec", type=float, default=0.0, help="Optional artifact-only block after clicks; no labels")
    parser.add_argument("--notes", default="")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--no-bell", action="store_true", help="Do not ring terminal bell at phase transitions")
    return parser

def _write_feature_row(recording: RecordingSession, host_time_s: float, frame) -> None:
    recording.feature_writer.writerow(
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


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    session_dir = Path(args.out) if args.out else default_session_dir(Path(args.sessions_root))
    phases = build_protocol(
        settle_sec=args.settle_sec,
        rest_sec=args.rest_sec,
        trials=args.trials,
        press_sec=args.press_sec,
        gap_sec=args.gap_sec,
        final_rest_sec=args.final_rest_sec,
        artifact_sec=args.artifact_sec,
    )
    if not phases:
        print("[reson-prompt-record] empty protocol", file=sys.stderr)
        raise SystemExit(2)

    running = True

    def _stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        lock = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-prompt-record] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    reader: SerialReader | None = None
    started_host_s = time.time()
    protocol_started_s: float | None = None
    last_sample_t_ms: int | None = None
    sample_count = 0
    frame_count = 0
    label_count = 0
    parse_errors = 0
    active_label: str | None = None
    current_phase_idx: int | None = None

    try:
        meta = build_session_meta(
            port=resolved_port,
            baud=args.baud,
            label_mode="prompted",
            label="CLICK",
            source="reson-prompt-record",
            notes=args.notes,
        )
        meta["prompt_protocol"] = {
            "settle_sec": args.settle_sec,
            "rest_sec": args.rest_sec,
            "trials": args.trials,
            "press_sec": args.press_sec,
            "gap_sec": args.gap_sec,
            "final_rest_sec": args.final_rest_sec,
            "artifact_sec": args.artifact_sec,
            "phases": protocol_as_dicts(phases),
        }

        extractor = FeatureFrameExtractor()
        reader = SerialReader(SerialConfig(port=resolved_port, baud=args.baud, timeout_s=0.1))
        retry_delay = reader.config.reconnect_initial_s
        next_status = time.monotonic() + 1.0
        recording = RecordingSession.create(session_dir, meta=meta, feature_fields=BASE_FEATURE_RECORD_FIELDS)
        try:
            recording.write_label({"type": "session_start", "host_time_s": started_host_s, "t_ms": None})
            print(
                f"[reson-prompt-record] writing {session_dir.resolve()} duration={protocol_duration_s(phases):.1f}s "
                "Do not touch the keyboard/trackpad during the protocol.",
                file=sys.stderr,
                flush=True,
            )

            while running:
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

                if protocol_started_s is None:
                    protocol_started_s = time.monotonic()

                last_sample_t_ms = sample.t_ms
                sample_count += 1
                recording.raw_writer.writerow(
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
                    _write_feature_row(recording, host_time_s, frame)

                elapsed_s = time.monotonic() - protocol_started_s
                phase_info = phase_at(phases, elapsed_s)
                if phase_info is None:
                    break

                phase_idx = phase_info.index
                phase = phase_info.phase
                if current_phase_idx != phase_idx:
                    if active_label is not None:
                        recording.write_label(
                            {
                                "type": "label_end",
                                "label": active_label,
                                "host_time_s": host_time_s,
                                "t_ms": last_sample_t_ms,
                                "closed_by": "phase_transition",
                            }
                        )
                        label_count += 1
                        active_label = None

                    current_phase_idx = phase_idx
                    recording.write_label(
                        {
                            "type": "prompt_phase",
                            "phase": phase.name,
                            "host_time_s": host_time_s,
                            "t_ms": last_sample_t_ms,
                            "duration_s": phase.duration_s,
                            "label": phase.label,
                        }
                    )
                    if phase.label is not None:
                        active_label = phase.label
                        recording.write_label(
                            {
                                "type": "label_start",
                                "label": active_label,
                                "host_time_s": host_time_s,
                                "t_ms": last_sample_t_ms,
                            }
                        )
                    bell = "" if args.no_bell else "\a"
                    print(
                        f"{bell}[reson-prompt-record] phase {phase_idx + 1}/{len(phases)}: {phase.name} "
                        f"for {phase.duration_s:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )

                if args.status and time.monotonic() >= next_status:
                    print(
                        f"[reson-prompt-record] samples={sample_count} frames={frame_count} "
                        f"labels={label_count} phase={phase.name} parse_errors={parse_errors}",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_status = time.monotonic() + 1.0

            if active_label is not None:
                recording.write_label(
                    {
                        "type": "label_end",
                        "label": active_label,
                        "host_time_s": time.time(),
                        "t_ms": last_sample_t_ms,
                        "closed_by": "session_end",
                    }
                )
                label_count += 1

            recording.write_label(
                {
                    "type": "session_end",
                    "host_time_s": time.time(),
                    "t_ms": last_sample_t_ms,
                    "samples": sample_count,
                    "frames": frame_count,
                    "labels": label_count,
                    "parse_errors": parse_errors,
                }
            )
        finally:
            recording.close()
    finally:
        if reader is not None:
            reader.close()
        lock.release()

    print(
        f"[reson-prompt-record] complete samples={sample_count} frames={frame_count} "
        f"intervals={label_count} parse_errors={parse_errors}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
