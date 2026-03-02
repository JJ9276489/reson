from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from reson.calibration import (
    CalibrationProfile,
    CalibrationError,
    default_profile,
    fit_hmm3_profile_from_staged_frames,
    load_profile,
    save_profile,
)
from reson.features import FeatureFrame, FeatureFrameExtractor
from reson.parser import parse_line
from reson.port_lock import PortLockInUseError, acquire_port_lock
from reson.serial_io import SerialConfig, SerialReader, resolve_port


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...", flush=True)
        time.sleep(1.0)


def _wait_for_enter(prompt: str) -> None:
    print(prompt, flush=True)
    input("Press Enter to start capture...")


def _capture_stage(
    *,
    reader: SerialReader,
    extractor: FeatureFrameExtractor,
    stage_name: str,
    duration_s: float,
    trim_ms: int,
) -> tuple[list[tuple[FeatureFrame, str]], int]:
    frames: list[tuple[FeatureFrame, str]] = []
    parsed_samples = 0
    parse_errors = 0
    started_t_ms: int | None = None
    start_wall = time.monotonic()
    next_progress = start_wall + 1.0

    while (time.monotonic() - start_wall) < duration_s:
        if not reader.is_connected():
            if not reader.reconnect():
                time.sleep(0.1)
                continue

        line = reader.read_line()
        if line is None:
            continue
        sample = parse_line(line)
        if sample is None:
            parse_errors += 1
            continue
        parsed_samples += 1
        if started_t_ms is None:
            started_t_ms = sample.t_ms

        _, new_frames = extractor.update(sample)
        for frame in new_frames:
            if started_t_ms is not None and (frame.t_ms - started_t_ms) < trim_ms:
                continue
            frames.append((frame, stage_name))

        now = time.monotonic()
        if now >= next_progress:
            elapsed = now - start_wall
            print(
                f"  [{stage_name}] {elapsed:4.1f}/{duration_s:4.1f}s "
                f"samples={parsed_samples} frames={len(frames)} parse_errors={parse_errors}",
                flush=True,
            )
            next_progress = now + 1.0

    return frames, parsed_samples


def _build_extractor(profile: CalibrationProfile) -> FeatureFrameExtractor:
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
    parser = argparse.ArgumentParser(description="Guided hmm3 calibration for Reson")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--out", default=".reson_profile.json", help="Output calibration profile path")
    parser.add_argument("--rest-sec", type=float, default=20.0)
    parser.add_argument("--light-sec", type=float, default=10.0)
    parser.add_argument("--heavy-sec", type=float, default=10.0)
    parser.add_argument("--artifact-sec", type=float, default=20.0)
    parser.add_argument("--trim-ms", type=int, default=250, help="Drop early frames each stage to reduce boundary bleed")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--non-interactive", action="store_true", help="Skip Enter prompts between stages")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_port = resolve_port(args.port)

    try:
        lock = acquire_port_lock(resolved_port)
    except PortLockInUseError as exc:
        print(f"[reson-calibrate-hmm3] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    try:
        try:
            base_profile = load_profile(Path(args.out))
            print(f"[reson-calibrate-hmm3] Loaded existing profile: {args.out}")
        except FileNotFoundError:
            base_profile = default_profile()
            print(f"[reson-calibrate-hmm3] No existing profile at {args.out}; using defaults")

        reader = SerialReader(SerialConfig(port=resolved_port, baud=args.baud, timeout_s=0.1))
        extractor = _build_extractor(base_profile)

        stages = [
            (
                "REST",
                args.rest_sec,
                "Stage REST: relax jaw, keep still, minimize motion artifacts.",
            ),
            (
                "PRESS_LIGHT",
                args.light_sec,
                "Stage PRESS_LIGHT: perform repeated light jaw presses/taps.",
            ),
            (
                "PRESS_HEAVY",
                args.heavy_sec,
                "Stage PRESS_HEAVY: perform repeated stronger jaw presses.",
            ),
            (
                "ARTIFACT",
                args.artifact_sec,
                "Stage ARTIFACT: head turns, jaw shifts without clench, cable tug, skin stretch, optional talking/swallowing.",
            ),
        ]

        staged_frames: list[tuple[FeatureFrame, str]] = []
        stage_sample_counts: dict[str, int] = {}
        print(f"[reson-calibrate-hmm3] Port={resolved_port} Baud={args.baud}")

        for name, duration_s, prompt in stages:
            if args.non_interactive:
                print(prompt)
            else:
                _wait_for_enter(prompt)
            if args.countdown > 0:
                _countdown(args.countdown)
            print(f"[reson-calibrate-hmm3] Capturing {name} for {duration_s:.1f}s", flush=True)
            frames, sample_count = _capture_stage(
                reader=reader,
                extractor=extractor,
                stage_name=name,
                duration_s=duration_s,
                trim_ms=max(args.trim_ms, 0),
            )
            staged_frames.extend(frames)
            stage_sample_counts[name] = sample_count
            print(
                f"[reson-calibrate-hmm3] Stage {name} complete: samples={sample_count} frames={len(frames)}",
                flush=True,
            )

        reader.close()

        fitted = fit_hmm3_profile_from_staged_frames(staged_frames, base_profile=base_profile)
        out_path = Path(args.out)
        save_profile(fitted, out_path)

        print("[reson-calibrate-hmm3] Calibration saved", flush=True)
        print(f"  Profile: {out_path.resolve()}", flush=True)
        print(f"  Frames: {len(staged_frames)}", flush=True)
        print(f"  Stage sample counts: {stage_sample_counts}", flush=True)
        print(f"  separation_ok: {fitted.separation_ok}", flush=True)
        print(f"  segment_thresholds: {fitted.segment_thresholds}", flush=True)
        print(
            "  Next: reson-debug --detector hmm3 --port "
            f"{resolved_port} --baud {args.baud}",
            flush=True,
        )
    except CalibrationError as exc:
        print(f"[reson-calibrate-hmm3] Calibration failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        lock.release()


if __name__ == "__main__":
    main()
