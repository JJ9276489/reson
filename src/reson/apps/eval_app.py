from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from reson.evaluation import (
    SessionScore,
    evaluate_decision_sweep,
    evaluate_loso,
    evaluate_nested_decision_selection,
    frozen_decision_grid,
    rank_sweep_rows,
)
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
    parser.add_argument("--pre-tol-ms", type=int, default=200, help="How early a down may fire before click onset")
    parser.add_argument("--post-tol-ms", type=int, default=200, help="How late a down may fire after click onset")
    parser.add_argument(
        "--include-glob",
        default="*",
        help="Only evaluate sessions whose dir name matches this glob (e.g. 'prompt-gui-*')",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma list of name substrings to skip (e.g. 'prompt-gui-004'). 'bad' dirs are always skipped.",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        metavar="MODEL:FEATURES",
        help="Sweep decision gates for one model family (e.g. 'threshold:wl') instead of comparing configs",
    )
    parser.add_argument(
        "--nested-sweep",
        default=None,
        metavar="MODEL:FEATURES",
        help="Select gates in inner LOSO folds, then score each untouched outer session once",
    )
    parser.add_argument("--sweep-top", type=int, default=8, help="How many ranked sweep rows to print")
    parser.add_argument(
        "--decision-grid",
        choices=("full", "frozen"),
        default=None,
        help="Gate grid: nested runs default to the two predeclared frozen configs; --sweep defaults to full",
    )
    parser.add_argument("--report", default=None, help="Optional CSV path for per-config summary rows")
    parser.add_argument("--per-session", action="store_true", help="Also print per-session scores as JSON")
    return parser


def _parse_exclude(spec: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in spec.split(",") if item.strip())


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
        "activated": score.n_activated,
        "detected": score.n_detected,
        "missed": score.n_missed,
        "false_downs_rest": score.false_downs_rest,
        "false_downs_artifact": score.false_downs_artifact,
        "false_downs_other": score.false_downs_other,
        "false_downs_total": score.n_false_downs,
        "false_clicks_total": score.n_false_clicks,
        "cancelled": score.n_cancelled,
        "matched_cancelled": score.n_matched_cancelled,
        "unterminated": score.n_unterminated,
    }


def _format_metric(value: object, decimals: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        ("config", 22),
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
            str(row["config"]).ljust(22),
            f"{row['detection_rate'] * 100:.0f}%".ljust(8),
            f"{int(row['missed'])}".ljust(6),
            _format_metric(row["false_downs_rest_per_min"]).ljust(12),
            _format_metric(row["false_downs_artifact_per_min"]).ljust(16),
            f"{row['down_latency_ms_median']:.0f}".ljust(12),
            f"{row['up_latency_ms_median']:.0f}".ljust(10),
            f"{row['duration_error_ms_median']:.0f}".ljust(11),
        ]
        print("".join(cells))


def _print_sweep(rows: list[dict[str, object]], top: int) -> dict[str, object]:
    ranked = rank_sweep_rows(rows)
    headers = (
        "enter exit dwell min_ev | detect rest/min art/min down_ms cancel"
    )
    print(headers)
    print("-" * len(headers))
    for row in ranked[:top]:
        print(
            f"{row['enter_threshold']:.2f}  {row['exit_threshold']:.2f} "
            f"{int(row['enter_dwell_frames'])}     {int(row['min_event_ms']):3d}    | "
            f"{row['detection_rate'] * 100:4.0f}%  "
            f"{_format_metric(row['false_downs_rest_per_min']):>6}  "
            f"{_format_metric(row['false_downs_artifact_per_min']):>6} "
            f"{row['down_latency_ms_median']:6.0f}  {int(row['cancelled'])}"
        )
    best = ranked[0]
    decision = {
        "enter_threshold": best["enter_threshold"],
        "exit_threshold": best["exit_threshold"],
        "enter_dwell_frames": int(best["enter_dwell_frames"]),
        "release_dwell_frames": int(best["release_dwell_frames"]),
        "min_event_ms": int(best["min_event_ms"]),
        "refractory_ms": int(best["refractory_ms"]),
    }
    print("\nbest decision config:")
    print(json.dumps(decision, indent=2))
    return decision


def main() -> None:
    args = build_parser().parse_args()
    if args.sweep and args.nested_sweep:
        raise SystemExit("[reson-eval] choose only one of --sweep or --nested-sweep")
    exclude = _parse_exclude(args.exclude)
    sessions_root = Path(args.sessions)

    if args.nested_sweep:
        model, features = _parse_configs(args.nested_sweep)[0]
        decision_grid = None if args.decision_grid == "full" else frozen_decision_grid()
        scores, aggregate, selections = evaluate_nested_decision_selection(
            sessions_root,
            feature_order=resolve_feature_order(features),
            model_name=model,
            decision_grid=decision_grid,
            epochs=args.epochs,
            seed=args.seed,
            ignore_margin_ms=args.ignore_margin_ms,
            pre_tol_ms=args.pre_tol_ms,
            post_tol_ms=args.post_tol_ms,
            include_glob=args.include_glob,
            exclude=exclude,
        )
        row = {"config": f"nested:{model}:{features}", **aggregate}
        _print_table([row])
        print("\ninner gate selections:")
        for selection in selections:
            print(json.dumps(selection, sort_keys=True), flush=True)
        if args.per_session:
            print("\nouter session scores:")
            for score in scores:
                print(json.dumps(_session_row(model, features, score), sort_keys=True), flush=True)
        print("\nacceptance contract:")
        print(json.dumps(aggregate["acceptance_contract"], sort_keys=True), flush=True)
        if not aggregate["acceptance_passed"]:
            if not aggregate["predeclared_grid"]:
                reason = "the exploratory decision grid is not eligible for frozen acceptance"
            elif not aggregate["all_inner_gates_met"]:
                reason = (
                    "at least one outer fold used a fallback because no inner candidate met "
                    "delivery/latency/exposure requirements"
                )
            else:
                reason = "the nested candidate failed at least one frozen baseline-relative criterion"
            print(
                f"\n[reson-eval] ACCEPTANCE GATE FAILED: {reason}",
                file=sys.stderr,
            )
            raise SystemExit(3)
        return

    if args.sweep:
        model, features = _parse_configs(args.sweep)[0]
        decision_grid = frozen_decision_grid() if args.decision_grid == "frozen" else None
        rows = evaluate_decision_sweep(
            sessions_root,
            feature_order=resolve_feature_order(features),
            model_name=model,
            decision_grid=decision_grid,
            epochs=args.epochs,
            seed=args.seed,
            ignore_margin_ms=args.ignore_margin_ms,
            pre_tol_ms=args.pre_tol_ms,
            post_tol_ms=args.post_tol_ms,
            include_glob=args.include_glob,
            exclude=exclude,
        )
        print(f"[reson-eval] swept {len(rows)} decision configs for {model}:{features}\n")
        _print_sweep(rows, args.sweep_top)
        return

    configs = _parse_configs(args.configs)
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
                include_glob=args.include_glob,
                exclude=exclude,
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
