from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
from pathlib import Path
from statistics import median
from typing import Any

from reson.binary_model import BinaryModelProfile, DEFAULT_BINARY_FEATURES, save_binary_profile
from reson.features import compute_feature_hash


FEATURE_PRESETS = {
    "wl": ["waveform_length"],
    "core": ["waveform_length", "rms_state"],
    "all": list(DEFAULT_BINARY_FEATURES),
}


@dataclass(frozen=True)
class FeatureExample:
    session: str
    t_ms: int
    features: dict[str, float]
    label: int


@dataclass(frozen=True)
class Dataset:
    examples: list[FeatureExample]
    feature_order: list[str]

    def vectors(self) -> list[list[float]]:
        return [[ex.features[name] for name in self.feature_order] for ex in self.examples]

    def labels(self) -> list[int]:
        return [ex.label for ex in self.examples]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return vals[max(0, min(idx, len(vals) - 1))]


def _robust_center_scale(values: list[float], floor: float = 1e-6) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    center = float(median(values))
    mad = float(median([abs(v - center) for v in values]))
    p90 = _percentile(values, 0.90)
    p10 = _percentile(values, 0.10)
    scale = max(1.4826 * mad, (p90 - p10) / 2.563, floor)
    return center, scale


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def resolve_feature_order(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec in FEATURE_PRESETS:
        return list(FEATURE_PRESETS[spec])
    out = [item.strip() for item in spec.split(",") if item.strip()]
    if not out:
        raise ValueError("feature set resolved to no features")
    unknown = [name for name in out if name not in DEFAULT_BINARY_FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s): {', '.join(unknown)}")
    return out


def read_label_intervals(label_path: Path) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    active_start: int | None = None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        typ = payload.get("type")
        t_ms = payload.get("t_ms")
        if t_ms is None:
            continue
        t = int(t_ms)
        if typ == "label_start":
            active_start = t
        elif typ == "label_end" and active_start is not None:
            if t > active_start:
                intervals.append((active_start, t))
            active_start = None
    return intervals


def _frame_label(t_ms: int, intervals: list[tuple[int, int]], ignore_margin_ms: int) -> int | None:
    for start, end in intervals:
        if start <= t_ms <= end:
            return 1
    for start, end in intervals:
        if (start - ignore_margin_ms) <= t_ms <= (end + ignore_margin_ms):
            return None
    return 0


def load_interval_sessions(
    sessions_root: Path,
    *,
    feature_order: list[str],
    ignore_margin_ms: int = 80,
) -> Dataset:
    examples: list[FeatureExample] = []
    session_dirs = sorted(path for path in sessions_root.glob("*") if path.is_dir() and not path.name.startswith("_"))
    for session_dir in session_dirs:
        feature_path = session_dir / "features.csv"
        label_path = session_dir / "labels.jsonl"
        if not feature_path.exists() or not label_path.exists():
            continue
        intervals = read_label_intervals(label_path)
        if not intervals:
            continue
        with feature_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                t_ms = int(float(row["t_ms"]))
                label = _frame_label(t_ms, intervals, ignore_margin_ms)
                if label is None:
                    continue
                feats = {name: float(row[name]) for name in feature_order}
                examples.append(FeatureExample(session=session_dir.name, t_ms=t_ms, features=feats, label=label))
    return Dataset(examples=examples, feature_order=feature_order)


def split_dataset(dataset: Dataset, val_fraction: float = 0.25, seed: int = 7) -> tuple[Dataset, Dataset]:
    by_session: dict[str, list[FeatureExample]] = {}
    for ex in dataset.examples:
        by_session.setdefault(ex.session, []).append(ex)
    sessions = sorted(by_session)
    rng = random.Random(seed)
    rng.shuffle(sessions)
    if len(sessions) >= 2:
        n_val_sessions = max(1, int(round(len(sessions) * val_fraction)))
        val_names = set(sessions[:n_val_sessions])
        train = [ex for ex in dataset.examples if ex.session not in val_names]
        val = [ex for ex in dataset.examples if ex.session in val_names]
    else:
        examples = list(dataset.examples)
        rng.shuffle(examples)
        n_val = max(1, int(round(len(examples) * val_fraction))) if len(examples) > 1 else 0
        val = examples[:n_val]
        train = examples[n_val:]
    return Dataset(train, dataset.feature_order), Dataset(val, dataset.feature_order)


def fit_normalization(dataset: Dataset) -> dict[str, dict[str, float]]:
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    floor: dict[str, float] = {}
    for name in dataset.feature_order:
        values = [ex.features[name] for ex in dataset.examples]
        c, s = _robust_center_scale(values, floor=1e-6)
        center[name] = c
        scale[name] = s
        floor[name] = 1e-6
    return {"center": center, "scale": scale, "floor": floor}


def normalize_vectors(dataset: Dataset, normalization: dict[str, dict[str, float]]) -> list[list[float]]:
    center = normalization["center"]
    scale = normalization["scale"]
    floor = normalization["floor"]
    out: list[list[float]] = []
    for ex in dataset.examples:
        row: list[float] = []
        for name in dataset.feature_order:
            row.append((ex.features[name] - center[name]) / max(scale[name], floor[name]))
        out.append(row)
    return out


def evaluate_probs(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for y, p in zip(labels, probs):
        pred = 1 if p >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 1 and pred == 0:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def train_threshold_profile(
    train: Dataset,
    val: Dataset,
    *,
    feature_name: str = "waveform_length",
) -> tuple[BinaryModelProfile, dict[str, float]]:
    if not train.examples:
        raise ValueError("cannot train threshold model with zero examples")
    values = sorted(set(ex.features[feature_name] for ex in train.examples))
    if len(values) == 1:
        candidates = values
    else:
        candidates = [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    best_threshold = candidates[0]
    best_metric = -1.0
    for threshold in candidates:
        probs = [1.0 if ex.features[feature_name] >= threshold else 0.0 for ex in train.examples]
        metric = evaluate_probs(train.labels(), probs)["f1"]
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold
    below = [ex.features[feature_name] for ex in train.examples if ex.label == 0]
    above = [ex.features[feature_name] for ex in train.examples if ex.label == 1]
    softness = max(abs((median(above) if above else best_threshold) - (median(below) if below else best_threshold)) / 4.0, 1.0)
    profile = BinaryModelProfile(
        model_type="threshold",
        feature_order=[feature_name],
        normalization={"center": {}, "scale": {}, "floor": {}},
        model={"feature": feature_name, "threshold": best_threshold, "softness": softness},
        metadata={"trained_at_utc": datetime.now(timezone.utc).isoformat(), "train_examples": len(train.examples)},
    )
    val_probs = [1.0 if ex.features[feature_name] >= best_threshold else 0.0 for ex in val.examples] if val.examples else []
    metrics = evaluate_probs(val.labels(), val_probs) if val.examples else {}
    metrics["threshold"] = float(best_threshold)
    return profile, metrics


def train_logreg_profile(
    train: Dataset,
    val: Dataset,
    *,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 7,
) -> tuple[BinaryModelProfile, dict[str, float]]:
    if not train.examples:
        raise ValueError("cannot train logistic regression with zero examples")
    rng = random.Random(seed)
    normalization = fit_normalization(train)
    x_train = normalize_vectors(train, normalization)
    y_train = train.labels()
    dim = len(train.feature_order)
    weights = [rng.uniform(-0.01, 0.01) for _ in range(dim)]
    bias = 0.0
    pos = max(sum(y_train), 1)
    neg = max(len(y_train) - sum(y_train), 1)
    pos_weight = len(y_train) / (2.0 * pos)
    neg_weight = len(y_train) / (2.0 * neg)

    for _ in range(max(epochs, 1)):
        grad_w = [0.0 for _ in range(dim)]
        grad_b = 0.0
        for x, y in zip(x_train, y_train):
            p = _sigmoid(sum(w * v for w, v in zip(weights, x)) + bias)
            sample_w = pos_weight if y else neg_weight
            diff = (p - y) * sample_w
            for j, value in enumerate(x):
                grad_w[j] += diff * value
            grad_b += diff
        inv_n = 1.0 / max(len(x_train), 1)
        for j in range(dim):
            weights[j] -= learning_rate * ((grad_w[j] * inv_n) + (l2 * weights[j]))
        bias -= learning_rate * grad_b * inv_n

    def predict(dataset: Dataset) -> list[float]:
        return [_sigmoid(sum(w * v for w, v in zip(weights, x)) + bias) for x in normalize_vectors(dataset, normalization)]

    metrics = evaluate_probs(val.labels(), predict(val)) if val.examples else {}
    profile = BinaryModelProfile(
        model_type="logreg",
        feature_order=list(train.feature_order),
        normalization=normalization,
        model={"weights": weights, "bias": bias},
        metadata={
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_examples": len(train.examples),
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
    )
    return profile, metrics


def write_training_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_profile_and_metrics(profile: BinaryModelProfile, profile_path: Path, metrics_path: Path, metrics: dict[str, float]) -> None:
    save_binary_profile(profile, profile_path)
    payload = {"profile": str(profile_path), "metrics": metrics}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sequence_examples(dataset: Dataset, normalization: dict[str, dict[str, float]], seq_len: int) -> tuple[list[list[list[float]]], list[int]]:
    by_session: dict[str, list[FeatureExample]] = {}
    for ex in dataset.examples:
        by_session.setdefault(ex.session, []).append(ex)
    xs: list[list[list[float]]] = []
    ys: list[int] = []
    for rows in by_session.values():
        rows = sorted(rows, key=lambda item: item.t_ms)
        tmp = Dataset(rows, dataset.feature_order)
        vecs = normalize_vectors(tmp, normalization)
        labels = tmp.labels()
        for idx in range(seq_len - 1, len(vecs)):
            xs.append(vecs[idx - seq_len + 1 : idx + 1])
            ys.append(labels[idx])
    return xs, ys


def train_torch_profile(
    model_type: str,
    train: Dataset,
    val: Dataset,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    hidden: int = 16,
    seq_len: int = 16,
    seed: int = 7,
) -> tuple[BinaryModelProfile, dict[str, float]]:
    try:
        import torch

        from reson.torch_models import build_torch_model
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Torch model training requires `python -m pip install '.[ml]'`") from exc

    if not train.examples:
        raise ValueError(f"cannot train {model_type} with zero examples")
    torch.manual_seed(seed)
    normalization = fit_normalization(train)
    x_train, y_train = _sequence_examples(train, normalization, seq_len)
    x_val, y_val = _sequence_examples(val, normalization, seq_len)
    if not x_train:
        raise ValueError(f"not enough frames to train {model_type}; need at least seq_len={seq_len}")

    cfg = {"input_dim": len(train.feature_order), "hidden": hidden, "seq_len": seq_len}
    model = build_torch_model(model_type, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    pos = max(sum(y_train), 1)
    neg = max(len(y_train) - sum(y_train), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(max(epochs, 1)):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    def predict(xs: list[list[list[float]]]) -> list[float]:
        if not xs:
            return []
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(xs, dtype=torch.float32)).reshape(-1)
            return [float(v) for v in torch.sigmoid(logits).tolist()]

    metrics = evaluate_probs(y_val, predict(x_val)) if y_val else {}
    state_dict = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    cfg.pop("input_dim", None)
    profile = BinaryModelProfile(
        model_type=model_type,
        feature_order=list(train.feature_order),
        normalization=normalization,
        model={"config": cfg, "seq_len": seq_len, "state_dict": state_dict},
        metadata={
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_examples": len(train.examples),
            "train_sequences": len(x_train),
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
    )
    return profile, metrics
