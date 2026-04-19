from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reson.binary_model import load_binary_profile, save_binary_profile
from reson.training import (
    load_interval_sessions,
    resolve_feature_order,
    save_profile_and_metrics,
    split_dataset,
    train_logreg_profile,
    train_threshold_profile,
    train_torch_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train binary Reson click/no-click models from interval-labeled sessions")
    parser.add_argument("--sessions", default="sessions", help="Directory containing interval-labeled session folders")
    parser.add_argument("--model", choices=("threshold", "logreg", "cnn", "tcn", "transformer", "all"), default="logreg")
    parser.add_argument("--features", default="wl", help="Feature preset: wl, core, all, or comma list")
    parser.add_argument("--out", default="models/binary_profile.json", help="Output profile path, or output dir for --model all")
    parser.add_argument("--metrics-out", default=None, help="Optional metrics JSON path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--ignore-margin-ms", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-skip-optional", action="store_true", help="Skip optional torch models when torch is unavailable")
    return parser


def _train_one(args, model_name: str, train, val):
    if model_name == "threshold":
        return train_threshold_profile(train, val, feature_name=train.feature_order[0])
    if model_name == "logreg":
        return train_logreg_profile(
            train,
            val,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    return train_torch_profile(
        model_name,
        train,
        val,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden=args.hidden,
        seq_len=args.seq_len,
        seed=args.seed,
    )


def main() -> None:
    args = build_parser().parse_args()
    feature_order = resolve_feature_order(args.features)
    dataset = load_interval_sessions(
        Path(args.sessions),
        feature_order=feature_order,
        ignore_margin_ms=args.ignore_margin_ms,
    )
    if not dataset.examples:
        print(
            "[reson-train] no interval-labeled examples found. Record data with "
            "`reson-debug --record-dir sessions/interval-001` and hold Space/C during clicks.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    train, val = split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)
    if not train.examples:
        print("[reson-train] no training examples after split", file=sys.stderr)
        raise SystemExit(2)

    model_names = ["threshold", "logreg", "cnn", "tcn", "transformer"] if args.model == "all" else [args.model]
    out_path = Path(args.out)
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        try:
            profile, metrics = _train_one(args, model_name, train, val)
        except RuntimeError as exc:
            if args.allow_skip_optional and model_name in {"cnn", "tcn", "transformer"}:
                print(f"[reson-train] skipped {model_name}: {exc}", file=sys.stderr)
                continue
            raise

        if args.model == "all":
            profile_path = out_path / f"{model_name}.json"
            metrics_path = out_path / f"{model_name}.metrics.json"
        else:
            profile_path = out_path
            metrics_path = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".metrics.json")
        save_profile_and_metrics(profile, profile_path, metrics_path, metrics)
        row = {
            "model": model_name,
            "profile": str(profile_path),
            "train_examples": len(train.examples),
            "val_examples": len(val.examples),
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.model == "all":
        summary_path = out_path / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if rows:
            best = max(rows, key=lambda row: float(row.get("f1", 0.0)))
            save_binary_profile(load_binary_profile(Path(str(best["profile"]))), out_path / "best.json")


if __name__ == "__main__":
    main()
