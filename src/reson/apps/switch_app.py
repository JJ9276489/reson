from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from reson.binary_model import BinaryModelDetector, load_binary_profile
from reson.parser import parse_line
from reson.port_lock import PortLockInUseError, acquire_port_lock
from reson.serial_io import SerialConfig, SerialReader, resolve_port
from reson.switch import edge_events_to_switch_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson binary switch event stream")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--profile", default="models/binary_profile.json")
    parser.add_argument("--status", action="store_true", help="Print reconnect/parse status to stderr")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)
    running = True

    def _stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        lock = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-switch] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    reader: SerialReader | None = None
    try:
        try:
            detector = BinaryModelDetector(load_binary_profile(Path(args.profile)))
        except FileNotFoundError as exc:
            print(
                f"[reson-switch] trained binary profile not found at {args.profile}. "
                "Run `reson-train --sessions sessions --out models/binary_profile.json`.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        reader = SerialReader(SerialConfig(port=resolved_port, baud=args.baud, timeout_s=0.1))
        retry_delay = reader.config.reconnect_initial_s
        parse_errors = 0

        print(
            f"[reson-switch] port={resolved_port} baud={args.baud} "
            f"profile={args.profile}",
            file=sys.stderr,
            flush=True,
        )

        while running:
            if not reader.is_connected():
                if args.status:
                    print(f"[reson-switch] reconnecting last_error={reader.last_error}", file=sys.stderr, flush=True)
                if reader.reconnect():
                    retry_delay = reader.config.reconnect_initial_s
                    if args.status:
                        print("[reson-switch] connected", file=sys.stderr, flush=True)
                    continue
                time.sleep(retry_delay)
                retry_delay = reader.next_reconnect_delay(retry_delay)
                continue

            line = reader.read_line()
            if line is None:
                continue
            sample = parse_line(line)
            if sample is None:
                parse_errors += 1
                if args.status and parse_errors % 25 == 0:
                    print(f"[reson-switch] parse_errors={parse_errors}", file=sys.stderr, flush=True)
                continue

            detector.update(sample)
            host_time_s = time.time()
            for switch_event in edge_events_to_switch_events(detector.pop_events()):
                print(json.dumps(switch_event.to_json_dict(host_time_s=host_time_s)), flush=True)
    finally:
        if reader is not None:
            reader.close()
        lock.release()


if __name__ == "__main__":
    main()
