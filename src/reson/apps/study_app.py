from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from reson.training import (
    Dataset,
    load_interval_sessions,
    resolve_feature_order,
    split_dataset,
    train_logreg_profile,
    train_threshold_profile,
    train_torch_profile,
    write_training_report,
)


def _parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Reson binary model scaling studies")
    parser.add_argument("--sessions", default="sessions")
    parser.add_argument("--models", default="threshold,logreg,cnn,tcn,transformer")
    parser.add_argument("--features", default="wl,core,all", help="Comma-separated feature presets")
    parser.add_argument("--fractions", default="0.25,0.5,1.0")
    parser.add_argument("--epochs", default="20,100")
    parser.add_argument("--hidden", default="8,16")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--out", default="studies/binary_scaling.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ignore-margin-ms", type=int, default=80)
    parser.add_argument("--allow-skip-optional", action="store_true", default=True)
    return parser


def _subset(dataset: Dataset, fraction: float) -> Dataset:
    if fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset.examples) * max(min(fraction, 1.0), 0.0)))
    return Dataset(dataset.examples[:n], dataset.feature_order)


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, object]] = []
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    feature_specs = [item.strip() for item in args.features.split(",") if item.strip()]
    fractions = _parse_list(args.fractions, float)
    epochs_list = _parse_list(args.epochs, int)
    hidden_list = _parse_list(args.hidden, int)

    for feature_spec in feature_specs:
        feature_order = resolve_feature_order(feature_spec)
        dataset = load_interval_sessions(Path(args.sessions), feature_order=feature_order, ignore_margin_ms=args.ignore_margin_ms)
        if not dataset.examples:
            print(f"[reson-study] no examples for features={feature_spec}", file=sys.stderr)
            continue
        train_base, val = split_dataset(dataset, seed=args.seed)
        for fraction in fractions:
            train = _subset(train_base, fraction)
            for model_name in models:
                epoch_values = [1] if model_name == "threshold" else epochs_list
                hidden_values = [0] if model_name in {"threshold", "logreg"} else hidden_list
                for epochs in epoch_values:
                    for hidden in hidden_values:
                        try:
                            if model_name == "threshold":
                                _, metrics = train_threshold_profile(train, val, feature_name=train.feature_order[0])
                            elif model_name == "logreg":
                                _, metrics = train_logreg_profile(train, val, epochs=epochs, seed=args.seed)
                            else:
                                _, metrics = train_torch_profile(
                                    model_name,
                                    train,
                                    val,
                                    epochs=epochs,
                                    hidden=hidden,
                                    seq_len=args.seq_len,
                                    seed=args.seed,
                                )
                            row = {
                                "model": model_name,
                                "features": feature_spec,
                                "fraction": fraction,
                                "epochs": epochs,
                                "hidden": hidden,
                                "train_examples": len(train.examples),
                                "val_examples": len(val.examples),
                                **metrics,
                            }
                        except RuntimeError as exc:
                            if args.allow_skip_optional and model_name in {"cnn", "tcn", "transformer"}:
                                row = {
                                    "model": model_name,
                                    "features": feature_spec,
                                    "fraction": fraction,
                                    "epochs": epochs,
                                    "hidden": hidden,
                                    "train_examples": len(train.examples),
                                    "val_examples": len(val.examples),
                                    "skipped": str(exc),
                                }
                            else:
                                raise
                        rows.append(row)
                        print(row, flush=True)

    write_training_report(Path(args.out), rows)


if __name__ == "__main__":
    main()
