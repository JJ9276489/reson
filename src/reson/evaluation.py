"""Event-level evaluation for trained binary switch profiles.

The unit tests in `tests/test_training.py` and friends check frame-level
behavior. This module instead scores a profile the way it is actually used at
runtime: it replays a recorded session's raw samples through
`BinaryModelDetector`, collects the emitted `down`/`up` switch events, and
compares them against the labeled click intervals and prompt-phase windows.

The metrics mirror `docs/validation_status.md`:

- false `down` events per minute during rest
- false `down` events per minute during artifact-only movement
- missed intended click intervals
- down latency after intended activation
- up latency after intended release
- event duration error relative to interval labels

Everything here is pure-Python and hardware-free so it can run in CI.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

from reson.binary_model import BinaryModelDetector, BinaryModelProfile
from reson.switch import edge_events_to_switch_events
from reson.training import (
    Dataset,
    load_interval_sessions,
    read_label_intervals,
    train_logreg_profile,
    train_threshold_profile,
)
from reson.types import EmgSample, SwitchEvent


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    label: str | None
    start_ms: int
    end_ms: int

    @property
    def is_artifact(self) -> bool:
        return "ARTIFACT" in self.name.upper()

    @property
    def is_rest(self) -> bool:
        # Rest-like = any unlabeled window that is not a deliberate artifact phase
        # (SETTLE and REST both count as "should stay silent" time).
        return self.label is None and not self.is_artifact


@dataclass(frozen=True)
class Press:
    down_ms: int
    up_ms: int | None

    @property
    def duration_ms(self) -> int | None:
        return None if self.up_ms is None else max(self.up_ms - self.down_ms, 0)


@dataclass
class SessionScore:
    session: str
    n_clicks: int = 0
    n_detected: int = 0
    false_downs_rest: int = 0
    false_downs_artifact: int = 0
    false_downs_other: int = 0
    rest_seconds: float = 0.0
    artifact_seconds: float = 0.0
    down_latencies_ms: list[float] = field(default_factory=list)
    up_latencies_ms: list[float] = field(default_factory=list)
    duration_errors_ms: list[float] = field(default_factory=list)

    @property
    def n_missed(self) -> int:
        return self.n_clicks - self.n_detected

    @property
    def n_false_downs(self) -> int:
        return self.false_downs_rest + self.false_downs_artifact + self.false_downs_other


def read_phase_windows(label_path: Path) -> list[PhaseWindow]:
    """Reconstruct phase windows from prompt_phase markers in labels.jsonl.

    Each window runs from its own timestamp to the next phase's timestamp,
    falling back to the recorded duration for the final phase.
    """
    markers: list[tuple[int, str, str | None, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("type") != "prompt_phase":
            continue
        t_ms = payload.get("t_ms")
        if t_ms is None:
            continue
        markers.append(
            (int(t_ms), str(payload.get("phase", "")), payload.get("label"), float(payload.get("duration_s", 0.0)))
        )
    windows: list[PhaseWindow] = []
    for idx, (start_ms, name, label, duration_s) in enumerate(markers):
        if idx + 1 < len(markers):
            end_ms = markers[idx + 1][0]
        else:
            end_ms = start_ms + int(duration_s * 1000.0)
        if end_ms > start_ms:
            windows.append(PhaseWindow(name=name, label=label, start_ms=start_ms, end_ms=end_ms))
    return windows


def read_raw_samples(raw_path: Path) -> list[EmgSample]:
    samples: list[EmgSample] = []
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                samples.append(
                    EmgSample(t_ms=int(float(row["t_ms"])), raw=int(float(row["raw"])), env=int(float(row["env"])))
                )
            except (KeyError, ValueError):
                continue
    return samples


def replay_session(profile: BinaryModelProfile, raw_path: Path) -> list[SwitchEvent]:
    """Feed a recorded session's raw samples through the runtime detector."""
    detector = BinaryModelDetector(profile)
    events = []
    last_t = 0
    for sample in read_raw_samples(raw_path):
        detector.update(sample)
        events.extend(detector.pop_events())
        last_t = sample.t_ms
    events.extend(detector.flush(last_t))
    return edge_events_to_switch_events(events)


def pair_presses(switch_events: list[SwitchEvent]) -> list[Press]:
    presses: list[Press] = []
    pending_down: int | None = None
    for event in switch_events:
        if event.phase == "down":
            if pending_down is not None:
                presses.append(Press(down_ms=pending_down, up_ms=None))
            pending_down = event.t_ms
        elif event.phase == "up" and pending_down is not None:
            presses.append(Press(down_ms=pending_down, up_ms=event.t_ms))
            pending_down = None
    if pending_down is not None:
        presses.append(Press(down_ms=pending_down, up_ms=None))
    return presses


def _classify_false_down(t_ms: int, phases: list[PhaseWindow]) -> str:
    for window in phases:
        if window.start_ms <= t_ms <= window.end_ms:
            if window.is_artifact:
                return "artifact"
            if window.is_rest:
                return "rest"
            return "other"
    return "other"


def score_session(
    session: str,
    switch_events: list[SwitchEvent],
    clicks: list[tuple[int, int]],
    phases: list[PhaseWindow],
    *,
    pre_tol_ms: int = 400,
    post_tol_ms: int = 600,
) -> SessionScore:
    score = SessionScore(session=session, n_clicks=len(clicks))
    for window in phases:
        seconds = (window.end_ms - window.start_ms) / 1000.0
        if window.is_artifact:
            score.artifact_seconds += seconds
        elif window.is_rest:
            score.rest_seconds += seconds

    presses = pair_presses(switch_events)
    used: set[int] = set()
    for start_ms, end_ms in clicks:
        match_idx: int | None = None
        for idx, press in enumerate(presses):
            if idx in used:
                continue
            if (start_ms - pre_tol_ms) <= press.down_ms <= (end_ms + post_tol_ms):
                match_idx = idx
                break
        if match_idx is None:
            continue
        used.add(match_idx)
        press = presses[match_idx]
        score.n_detected += 1
        score.down_latencies_ms.append(float(press.down_ms - start_ms))
        if press.up_ms is not None:
            score.up_latencies_ms.append(float(press.up_ms - end_ms))
            score.duration_errors_ms.append(float(press.duration_ms - (end_ms - start_ms)))

    for idx, press in enumerate(presses):
        if idx in used:
            continue
        bucket = _classify_false_down(press.down_ms, phases)
        if bucket == "artifact":
            score.false_downs_artifact += 1
        elif bucket == "rest":
            score.false_downs_rest += 1
        else:
            score.false_downs_other += 1
    return score


def _per_minute(count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return count / (seconds / 60.0)


def aggregate_scores(scores: list[SessionScore]) -> dict[str, float]:
    n_clicks = sum(s.n_clicks for s in scores)
    n_detected = sum(s.n_detected for s in scores)
    rest_seconds = sum(s.rest_seconds for s in scores)
    artifact_seconds = sum(s.artifact_seconds for s in scores)
    false_rest = sum(s.false_downs_rest for s in scores)
    false_artifact = sum(s.false_downs_artifact for s in scores)
    down_lat = [v for s in scores for v in s.down_latencies_ms]
    up_lat = [v for s in scores for v in s.up_latencies_ms]
    dur_err = [v for s in scores for v in s.duration_errors_ms]

    def _med(values: list[float]) -> float:
        return float(median(values)) if values else 0.0

    def _mean(values: list[float]) -> float:
        return float(mean(values)) if values else 0.0

    return {
        "sessions": float(len(scores)),
        "clicks": float(n_clicks),
        "detected": float(n_detected),
        "missed": float(n_clicks - n_detected),
        "detection_rate": (n_detected / n_clicks) if n_clicks else 0.0,
        "false_downs_rest": float(false_rest),
        "false_downs_artifact": float(false_artifact),
        "false_downs_rest_per_min": _per_minute(false_rest, rest_seconds),
        "false_downs_artifact_per_min": _per_minute(false_artifact, artifact_seconds),
        "down_latency_ms_median": _med(down_lat),
        "up_latency_ms_median": _med(up_lat),
        "down_latency_ms_mean": _mean(down_lat),
        "duration_error_ms_median": _med(dur_err),
        "rest_seconds": rest_seconds,
        "artifact_seconds": artifact_seconds,
    }


def list_session_dirs(sessions_root: Path) -> list[Path]:
    """Session dirs eligible for evaluation: have raw+labels, not hidden/bad."""
    dirs: list[Path] = []
    for path in sorted(sessions_root.glob("*")):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if "bad" in path.name.lower():
            continue
        if (path / "raw.csv").exists() and (path / "labels.jsonl").exists():
            dirs.append(path)
    return dirs


def _train_fold(model_name: str, train: Dataset, *, epochs: int, seed: int) -> BinaryModelProfile:
    empty = Dataset([], train.feature_order)
    if model_name == "threshold":
        profile, _ = train_threshold_profile(train, empty, feature_name=train.feature_order[0])
    elif model_name == "logreg":
        profile, _ = train_logreg_profile(train, empty, epochs=epochs, seed=seed)
    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"unsupported eval model_name={model_name!r}")
    return profile


def evaluate_loso(
    sessions_root: Path,
    *,
    feature_order: list[str],
    model_name: str,
    epochs: int = 200,
    seed: int = 7,
    ignore_margin_ms: int = 80,
    pre_tol_ms: int = 400,
    post_tol_ms: int = 600,
) -> tuple[list[SessionScore], dict[str, float]]:
    """Leave-one-session-out evaluation.

    For each eligible session we train `model_name` on every *other* session and
    replay the held-out session through the runtime detector, so every reported
    number is held-out performance.
    """
    session_dirs = list_session_dirs(sessions_root)
    session_names = {path.name for path in session_dirs}
    dataset = load_interval_sessions(sessions_root, feature_order=feature_order, ignore_margin_ms=ignore_margin_ms)
    dataset = Dataset([ex for ex in dataset.examples if ex.session in session_names], feature_order)
    if len({ex.session for ex in dataset.examples}) < 2:
        raise ValueError("leave-one-session-out evaluation needs at least 2 labeled sessions")

    scores: list[SessionScore] = []
    for held_out in session_dirs:
        name = held_out.name
        train_examples = [ex for ex in dataset.examples if ex.session != name]
        if not train_examples:
            continue
        train = Dataset(train_examples, feature_order)
        profile = _train_fold(model_name, train, epochs=epochs, seed=seed)
        switch_events = replay_session(profile, held_out / "raw.csv")
        clicks = read_label_intervals(held_out / "labels.jsonl")
        phases = read_phase_windows(held_out / "labels.jsonl")
        scores.append(
            score_session(
                name, switch_events, clicks, phases, pre_tol_ms=pre_tol_ms, post_tol_ms=post_tol_ms
            )
        )
    return scores, aggregate_scores(scores)
