from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from reson.api import ResonSerialConfig, iter_serial_switch_updates
from reson.port_lock import PortLockInUseError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reson binary switch event stream")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--profile", default="models/binary_profile.json")
    parser.add_argument("--status", action="store_true", help="Print reconnect/parse status to stderr")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    running = True

    def _stop(*_args) -> None:
        nonlocal running
        running = False

    def _status(message: str) -> None:
        if message.startswith("port=") or args.status:
            print(f"[reson-switch] {message}", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    profile_path = Path(args.profile)
    try:
        config = ResonSerialConfig(port=args.port, baud=args.baud, profile_path=profile_path)
        for update in iter_serial_switch_updates(
            config,
            should_stop=lambda: not running,
            status_callback=_status,
        ):
            for switch_event in update.events:
                print(json.dumps(switch_event.to_json_dict(host_time_s=update.host_time_s)), flush=True)
    except FileNotFoundError as exc:
        if str(profile_path) in str(exc):
            print(
                f"[reson-switch] trained binary profile not found at {args.profile}. "
                "Run `reson-train --sessions sessions --out models/binary_profile.json`.",
                file=sys.stderr,
            )
        else:
            print(f"[reson-switch] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except (PortLockInUseError, RuntimeError) as exc:
        print(f"[reson-switch] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
