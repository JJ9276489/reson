from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from reson.evaluation import SessionScore, evaluate_loso
from reson.training import resolve_feature_order


# The three baselines docs/validation_status.md asks for.
DEFAULT_CONFIGS = [
    ("threshold", "wl"),
    ("logreg", "wl"),
    ("logreg", "all"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leave-one-session-out event-level evaluation of binary switch baselines"
    )
    parser.add_argument("--sessions", default="sessions", help="Directory of labeled session folders")
    parser.add_argument(
        "--configs",
        default="threshold:wl,logreg:wl,logreg:all",
        help="Comma list of model:features pairs to compare",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ignore-margin-ms", type=int, default=80)
    parser.add_argument("--pre-tol-ms", type=int, default=400, help="How early a down may fire before a click start")
    parser.add_argument("--post-tol-ms", type=int, default=600, help="How late a down may fire after a click end")
    parser.add_argument("--report", default=None, help="Optional CSV path for per-config summary rows")
    parser.add_argument("--per-session", action="store_true", help="Also print per-session scores as JSON")
    return parser


def _parse_configs(spec: str) -> list[tuple[str, str]]:
    configs: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad config {item!r}; expected model:features e.g. logreg:wl")
        model, features = item.split(":", 1)
        configs.append((model.strip(), features.strip()))
    return configs


def _session_row(model: str, features: str, score: SessionScore) -> dict[str, object]:
    return {
        "model": model,
        "features": features,
        "session": score.session,
        "clicks": score.n_clicks,
        "detected": score.n_detected,
        "missed": score.n_missed,
        "false_downs_rest": score.false_downs_rest,
        "false_downs_artifact": score.false_downs_artifact,
        "false_downs_other": score.false_downs_other,
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        ("config", 16),
        ("detect", 8),
        ("miss", 6),
        ("fp_rest/min", 12),
        ("fp_artifact/min", 16),
        ("down_lat_ms", 12),
        ("up_lat_ms", 10),
        ("dur_err_ms", 11),
    ]
    line = "".join(name.ljust(width) for name, width in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = [
            str(row["config"]).ljust(16),
            f"{row['detection_rate'] * 100:.0f}%".ljust(8),
            f"{int(row['missed'])}".ljust(6),
            f"{row['false_downs_rest_per_min']:.2f}".ljust(12),
            f"{row['false_downs_artifact_per_min']:.2f}".ljust(16),
            f"{row['down_latency_ms_median']:.0f}".ljust(12),
            f"{row['up_latency_ms_median']:.0f}".ljust(10),
            f"{row['duration_error_ms_median']:.0f}".ljust(11),
        ]
        print("".join(cells))


def main() -> None:
    args = build_parser().parse_args()
    configs = _parse_configs(args.configs)
    sessions_root = Path(args.sessions)

    summary_rows: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []
    for model, features in configs:
        feature_order = resolve_feature_order(features)
        try:
            scores, agg = evaluate_loso(
                sessions_root,
                feature_order=feature_order,
                model_name=model,
                epochs=args.epochs,
                seed=args.seed,
                ignore_margin_ms=args.ignore_margin_ms,
                pre_tol_ms=args.pre_tol_ms,
                post_tol_ms=args.post_tol_ms,
            )
        except ValueError as exc:
            print(f"[reson-eval] {model}:{features} skipped: {exc}", file=sys.stderr)
            continue
        row = {"config": f"{model}:{features}", "model": model, "features": features, **agg}
        summary_rows.append(row)
        if args.per_session:
            session_rows.extend(_session_row(model, features, s) for s in scores)

    if not summary_rows:
        print("[reson-eval] no results; need at least 2 labeled sessions under --sessions", file=sys.stderr)
        raise SystemExit(2)

    _print_table(summary_rows)
    if args.per_session:
        print()
        for row in session_rows:
            print(json.dumps(row, sort_keys=True), flush=True)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in summary_rows for key in row})
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[reson-eval] wrote summary to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
