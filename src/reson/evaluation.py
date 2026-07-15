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
import dataclasses
import json
from dataclasses import dataclass, field
from itertools import product
from math import ceil, isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any

from reson.binary_model import BinaryDecisionConfig, BinaryModelDetector, BinaryModelProfile
from reson.switch import edge_events_to_switch_events
from reson.training import (
    Dataset,
    load_interval_sessions,
    read_label_intervals,
    train_logreg_profile,
    train_threshold_profile,
)
from reson.types import EmgSample, SwitchEvent


MAX_EVALUATION_SAMPLE_GAP_MS = 100


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
    terminal_ms: int | None = None
    terminal_phase: str | None = None

    @property
    def up_ms(self) -> int | None:
        return self.terminal_ms if self.terminal_phase == "up" else None

    @property
    def is_completed(self) -> bool:
        return self.terminal_phase == "up" and self.terminal_ms is not None

    @property
    def is_cancelled(self) -> bool:
        return self.terminal_phase == "cancel"

    @property
    def is_unterminated(self) -> bool:
        return self.terminal_phase == "unterminated"

    @property
    def duration_ms(self) -> int | None:
        return None if self.terminal_ms is None else max(self.terminal_ms - self.down_ms, 0)


@dataclass
class SessionScore:
    session: str
    n_clicks: int = 0
    n_activated: int = 0
    n_detected: int = 0
    false_downs_rest: int = 0
    false_downs_artifact: int = 0
    false_downs_other: int = 0
    false_clicks_rest: int = 0
    false_clicks_artifact: int = 0
    false_clicks_other: int = 0
    n_cancelled: int = 0
    n_matched_cancelled: int = 0
    n_unterminated: int = 0
    n_matched_unterminated: int = 0
    rest_seconds: float = 0.0
    artifact_seconds: float = 0.0
    down_latencies_ms: list[float] = field(default_factory=list)
    up_latencies_ms: list[float] = field(default_factory=list)
    duration_errors_ms: list[float] = field(default_factory=list)
    detected_click_indices: list[int] = field(default_factory=list)

    @property
    def n_missed(self) -> int:
        return self.n_clicks - self.n_detected

    @property
    def n_activation_missed(self) -> int:
        return self.n_clicks - self.n_activated

    @property
    def n_false_downs(self) -> int:
        return self.false_downs_rest + self.false_downs_artifact + self.false_downs_other

    @property
    def n_false_clicks(self) -> int:
        return self.false_clicks_rest + self.false_clicks_artifact + self.false_clicks_other


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
    marker_times = [marker[0] for marker in markers]
    if any(later <= earlier for earlier, later in zip(marker_times, marker_times[1:])):
        raise ValueError(f"prompt phase timestamps must be strictly increasing in {label_path}")
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


def read_recording_bounds(
    raw_path: Path,
    *,
    max_gap_ms: int = MAX_EVALUATION_SAMPLE_GAP_MS,
) -> tuple[int, int]:
    """Return the finite, ordered time span actually present in ``raw.csv``.

    Phase exposure is only meaningful inside observed recording coverage. A
    labels file therefore cannot extend the evaluation denominator beyond the
    first and last usable raw samples, and a recording with a gap over the
    conservative continuity limit is rejected rather than counting the gap.
    """
    if max_gap_ms <= 0:
        raise ValueError("maximum raw-sample gap must be positive")
    samples = read_raw_samples(raw_path)
    if len(samples) < 2:
        raise ValueError(f"recording needs at least 2 usable raw samples: {raw_path}")
    times = [sample.t_ms for sample in samples]
    if any(later < earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"raw sample timestamps must be ordered in {raw_path}")
    largest_gap_ms = max(later - earlier for earlier, later in zip(times, times[1:]))
    if largest_gap_ms > max_gap_ms:
        raise ValueError(
            f"raw sample gap {largest_gap_ms} ms exceeds {max_gap_ms} ms; "
            f"continuous exposure is not observed in {raw_path}"
        )
    start_ms, end_ms = times[0], times[-1]
    if end_ms <= start_ms:
        raise ValueError(f"recording coverage must have positive duration: {raw_path}")
    return start_ms, end_ms


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
                presses.append(Press(down_ms=pending_down, terminal_phase="unterminated"))
            pending_down = event.t_ms
        elif event.phase == "up" and pending_down is not None:
            presses.append(Press(down_ms=pending_down, terminal_ms=event.t_ms, terminal_phase="up"))
            pending_down = None
        elif event.phase == "cancel" and pending_down is not None:
            presses.append(Press(down_ms=pending_down, terminal_ms=event.t_ms, terminal_phase="cancel"))
            pending_down = None
    if pending_down is not None:
        presses.append(Press(down_ms=pending_down, terminal_phase="unterminated"))
    return presses


def _classify_false_down(t_ms: int, phases: list[PhaseWindow]) -> str:
    for window in phases:
        # Phase windows share endpoints; half-open intervals ensure a boundary
        # event belongs to the phase that starts at that timestamp.
        if window.start_ms <= t_ms < window.end_ms:
            if window.is_artifact:
                return "artifact"
            if window.is_rest:
                return "rest"
            return "other"
    return "other"


def _validate_recording_bounds(recording_bounds_ms: tuple[int, int]) -> tuple[int, int]:
    if len(recording_bounds_ms) != 2:
        raise ValueError("recording bounds must contain exactly (start_ms, end_ms)")
    start_ms, end_ms = recording_bounds_ms
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in recording_bounds_ms):
        raise ValueError("recording bounds must be finite integer timestamps")
    if end_ms <= start_ms:
        raise ValueError("recording coverage must have positive duration")
    return start_ms, end_ms


def _validate_phase_windows(
    phases: list[PhaseWindow],
    recording_bounds_ms: tuple[int, int],
) -> None:
    recording_start_ms, recording_end_ms = _validate_recording_bounds(recording_bounds_ms)
    previous_end: int | None = None
    for window in phases:
        endpoints = (window.start_ms, window.end_ms)
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in endpoints):
            raise ValueError(f"phase window {window.name!r} must use finite integer timestamps")
        if window.end_ms <= window.start_ms:
            raise ValueError(f"invalid phase window {window.name!r}: end must be after start")
        if window.start_ms < recording_start_ms or window.end_ms > recording_end_ms:
            raise ValueError(
                f"phase window {window.name!r} lies outside recorded coverage "
                f"[{recording_start_ms}, {recording_end_ms}]"
            )
        if previous_end is not None and window.start_ms < previous_end:
            raise ValueError("phase windows must be ordered and non-overlapping")
        previous_end = window.end_ms


def _validate_scored_events(
    switch_events: list[SwitchEvent],
    clicks: list[tuple[int, int]],
    recording_bounds_ms: tuple[int, int],
) -> None:
    recording_start_ms, recording_end_ms = _validate_recording_bounds(recording_bounds_ms)
    for start_ms, end_ms in clicks:
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start_ms, end_ms)):
            raise ValueError("click intervals must use finite integer timestamps")
        if end_ms <= start_ms:
            raise ValueError("click interval end must be after start")
        if start_ms < recording_start_ms or end_ms > recording_end_ms:
            raise ValueError("click interval lies outside recorded coverage")
    for event in switch_events:
        if not isinstance(event.t_ms, int) or isinstance(event.t_ms, bool):
            raise ValueError("switch events must use finite integer timestamps")
        if event.t_ms < recording_start_ms or event.t_ms > recording_end_ms:
            raise ValueError("switch event lies outside recorded coverage")


def _optimal_onset_matches(
    clicks: list[tuple[int, int]],
    presses: list[Press],
    click_indices: list[int],
    press_indices: list[int],
    *,
    pre_tol_ms: int,
    post_tol_ms: int,
) -> set[tuple[int, int]]:
    """Maximum-cardinality, minimum-onset-error order-preserving matching."""
    ordered_clicks = sorted(click_indices, key=lambda idx: (clicks[idx][0], idx))
    ordered_presses = sorted(press_indices, key=lambda idx: (presses[idx].down_ms, idx))
    # State: completed matches, all matches, summed absolute onset error,
    # matched original indices. One lexicographic objective prevents a close
    # completed press from consuming the only click an incomplete press could
    # match when assigning the completed press to a later click preserves both.
    empty: tuple[int, int, int, tuple[tuple[int, int], ...]] = (0, 0, 0, ())
    dp = [[empty for _ in range(len(ordered_presses) + 1)] for _ in range(len(ordered_clicks) + 1)]

    def choose(states: list[tuple[int, int, int, tuple[tuple[int, int], ...]]]):
        return min(states, key=lambda state: (-state[0], -state[1], state[2], state[3]))

    for i in range(1, len(ordered_clicks) + 1):
        click_idx = ordered_clicks[i - 1]
        click_start = clicks[click_idx][0]
        for j in range(1, len(ordered_presses) + 1):
            press_idx = ordered_presses[j - 1]
            press_down = presses[press_idx].down_ms
            candidates = [dp[i - 1][j], dp[i][j - 1]]
            if (click_start - pre_tol_ms) <= press_down <= (click_start + post_tol_ms):
                completed, count, cost, pairs = dp[i - 1][j - 1]
                candidates.append(
                    (
                        completed + (1 if presses[press_idx].is_completed else 0),
                        count + 1,
                        cost + abs(press_down - click_start),
                        pairs + ((click_idx, press_idx),),
                    )
                )
            dp[i][j] = choose(candidates)
    return set(dp[-1][-1][3])


def score_session(
    session: str,
    switch_events: list[SwitchEvent],
    clicks: list[tuple[int, int]],
    phases: list[PhaseWindow],
    *,
    recording_bounds_ms: tuple[int, int],
    pre_tol_ms: int = 200,
    post_tol_ms: int = 200,
) -> SessionScore:
    if pre_tol_ms < 0 or post_tol_ms < 0:
        raise ValueError("onset tolerances must be non-negative")
    _validate_phase_windows(phases, recording_bounds_ms)
    _validate_scored_events(switch_events, clicks, recording_bounds_ms)
    score = SessionScore(session=session, n_clicks=len(clicks))
    for window in phases:
        seconds = (window.end_ms - window.start_ms) / 1000.0
        if window.is_artifact:
            score.artifact_seconds += seconds
        elif window.is_rest:
            score.rest_seconds += seconds

    presses = pair_presses(switch_events)
    score.n_cancelled = sum(press.is_cancelled for press in presses)
    score.n_unterminated = sum(press.is_unterminated for press in presses)

    matches = _optimal_onset_matches(
        clicks,
        presses,
        list(range(len(clicks))),
        list(range(len(presses))),
        pre_tol_ms=pre_tol_ms,
        post_tol_ms=post_tol_ms,
    )
    used_presses = {press_idx for _click_idx, press_idx in matches}
    score.n_activated = len(matches)
    score.n_detected = sum(presses[press_idx].is_completed for _click_idx, press_idx in matches)
    score.detected_click_indices = sorted(
        click_idx for click_idx, press_idx in matches if presses[press_idx].is_completed
    )
    score.n_matched_cancelled = sum(presses[press_idx].is_cancelled for _click_idx, press_idx in matches)
    score.n_matched_unterminated = sum(presses[press_idx].is_unterminated for _click_idx, press_idx in matches)

    for click_idx, press_idx in matches:
        start_ms, end_ms = clicks[click_idx]
        press = presses[press_idx]
        if not press.is_completed:
            continue
        assert press.up_ms is not None
        score.down_latencies_ms.append(float(press.down_ms - start_ms))
        score.up_latencies_ms.append(float(press.up_ms - end_ms))
        score.duration_errors_ms.append(float(press.duration_ms - (end_ms - start_ms)))

    # n_detected was initialized from the match count; the loop above only
    # collects per-match timing metrics.
    for idx, press in enumerate(presses):
        if idx in used_presses:
            continue
        bucket = _classify_false_down(press.down_ms, phases)
        if bucket == "artifact":
            score.false_downs_artifact += 1
            if press.is_completed:
                score.false_clicks_artifact += 1
        elif bucket == "rest":
            score.false_downs_rest += 1
            if press.is_completed:
                score.false_clicks_rest += 1
        else:
            score.false_downs_other += 1
            if press.is_completed:
                score.false_clicks_other += 1
    return score


def _per_minute(count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return count / (seconds / 60.0)


def aggregate_scores(scores: list[SessionScore]) -> dict[str, Any]:
    n_clicks = sum(s.n_clicks for s in scores)
    n_activated = sum(s.n_activated for s in scores)
    n_detected = sum(s.n_detected for s in scores)
    rest_seconds = sum(s.rest_seconds for s in scores)
    artifact_seconds = sum(s.artifact_seconds for s in scores)
    false_rest = sum(s.false_downs_rest for s in scores)
    false_artifact = sum(s.false_downs_artifact for s in scores)
    false_other = sum(s.false_downs_other for s in scores)
    false_clicks_rest = sum(s.false_clicks_rest for s in scores)
    false_clicks_artifact = sum(s.false_clicks_artifact for s in scores)
    false_clicks_other = sum(s.false_clicks_other for s in scores)
    down_lat = [v for s in scores for v in s.down_latencies_ms]
    up_lat = [v for s in scores for v in s.up_latencies_ms]
    dur_err = [v for s in scores for v in s.duration_errors_ms]

    def _med(values: list[float]) -> float:
        return float(median(values)) if values else 0.0

    def _mean(values: list[float]) -> float:
        return float(mean(values)) if values else 0.0

    def _p95_abs(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(abs(value) for value in values)
        return float(ordered[max(ceil(0.95 * len(ordered)) - 1, 0)])

    false_total = false_rest + false_artifact + false_other
    false_clicks_total = false_clicks_rest + false_clicks_artifact + false_clicks_other
    negative_seconds = rest_seconds + artifact_seconds
    matched_cancelled = sum(s.n_matched_cancelled for s in scores)
    matched_unterminated = sum(s.n_matched_unterminated for s in scores)
    lifecycle_faults = matched_cancelled + sum(s.n_unterminated for s in scores)

    return {
        "sessions": float(len(scores)),
        "clicks": float(n_clicks),
        "activated": float(n_activated),
        "detected": float(n_detected),
        "missed": float(n_clicks - n_detected),
        "activation_missed": float(n_clicks - n_activated),
        "activation_rate": (n_activated / n_clicks) if n_clicks else 0.0,
        "detection_rate": (n_detected / n_clicks) if n_clicks else 0.0,
        "false_downs_rest": float(false_rest),
        "false_downs_artifact": float(false_artifact),
        "false_downs_other": float(false_other),
        "false_downs_total": float(false_total),
        "false_clicks_rest": float(false_clicks_rest),
        "false_clicks_artifact": float(false_clicks_artifact),
        "false_clicks_other": float(false_clicks_other),
        "false_clicks_total": float(false_clicks_total),
        "cancelled": float(sum(s.n_cancelled for s in scores)),
        "matched_cancelled": float(matched_cancelled),
        "matched_unterminated": float(matched_unterminated),
        "unterminated": float(sum(s.n_unterminated for s in scores)),
        "lifecycle_faults": float(lifecycle_faults),
        "false_downs_rest_per_min": _per_minute(false_rest, rest_seconds),
        "false_downs_artifact_per_min": _per_minute(false_artifact, artifact_seconds),
        "false_downs_per_negative_min": _per_minute(false_total, negative_seconds),
        "false_clicks_per_negative_min": _per_minute(false_clicks_total, negative_seconds),
        "down_latency_ms_median": _med(down_lat),
        "down_latency_ms_p95_abs": _p95_abs(down_lat),
        "early_detection_fraction": (
            sum(value < 0 for value in down_lat) / len(down_lat) if down_lat else 0.0
        ),
        "up_latency_ms_median": _med(up_lat),
        "down_latency_ms_mean": _mean(down_lat),
        "duration_error_ms_median": _med(dur_err),
        "rest_seconds": rest_seconds,
        "artifact_seconds": artifact_seconds,
        "negative_seconds": negative_seconds,
    }


def list_session_dirs(
    sessions_root: Path,
    *,
    include_glob: str = "*",
    exclude: tuple[str, ...] = (),
) -> list[Path]:
    """Session dirs eligible for evaluation: have raw+labels, not hidden/bad.

    `include_glob` keeps only directory names matching the glob (e.g.
    ``prompt-gui-*``). `exclude` drops any session whose name contains one of
    the given substrings (e.g. ``("prompt-gui-004",)``). Directories whose name
    contains ``bad`` are always skipped.
    """
    dirs: list[Path] = []
    for path in sorted(sessions_root.glob(include_glob)):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if "bad" in path.name.lower():
            continue
        if any(token and token in path.name for token in exclude):
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
    pre_tol_ms: int = 200,
    post_tol_ms: int = 200,
    include_glob: str = "*",
    exclude: tuple[str, ...] = (),
) -> tuple[list[SessionScore], dict[str, Any]]:
    """Leave-one-session-out evaluation.

    For each eligible session we train `model_name` on every *other* session and
    replay the held-out session through the runtime detector, so every reported
    number is held-out performance.
    """
    folds = _build_folds(
        sessions_root,
        feature_order=feature_order,
        model_name=model_name,
        epochs=epochs,
        seed=seed,
        ignore_margin_ms=ignore_margin_ms,
        include_glob=include_glob,
        exclude=exclude,
    )
    scores = [
        score_session(
            fold.name, replay_session(fold.profile, fold.raw_path), fold.clicks, fold.phases,
            recording_bounds_ms=fold.recording_bounds_ms,
            pre_tol_ms=pre_tol_ms, post_tol_ms=post_tol_ms,
        )
        for fold in folds
    ]
    return scores, aggregate_scores(scores)


@dataclass(frozen=True)
class _Fold:
    name: str
    profile: BinaryModelProfile
    raw_path: Path
    clicks: list[tuple[int, int]]
    phases: list[PhaseWindow]
    recording_bounds_ms: tuple[int, int]


def _build_folds(
    sessions_root: Path,
    *,
    feature_order: list[str],
    model_name: str,
    epochs: int,
    seed: int,
    ignore_margin_ms: int,
    include_glob: str,
    exclude: tuple[str, ...],
) -> list[_Fold]:
    session_dirs = list_session_dirs(sessions_root, include_glob=include_glob, exclude=exclude)
    dataset = load_interval_sessions(sessions_root, feature_order=feature_order, ignore_margin_ms=ignore_margin_ms)
    return _build_folds_for_dirs(
        session_dirs,
        dataset,
        feature_order=feature_order,
        model_name=model_name,
        epochs=epochs,
        seed=seed,
    )


def _build_folds_for_dirs(
    session_dirs: list[Path],
    dataset: Dataset,
    *,
    feature_order: list[str],
    model_name: str,
    epochs: int,
    seed: int,
) -> list[_Fold]:
    """Build LOSO folds restricted to an explicit, preselected session set."""
    session_names = {path.name for path in session_dirs}
    dataset = Dataset([ex for ex in dataset.examples if ex.session in session_names], feature_order)
    if len({ex.session for ex in dataset.examples}) < 2:
        raise ValueError("leave-one-session-out evaluation needs at least 2 labeled sessions")

    folds: list[_Fold] = []
    for held_out in session_dirs:
        name = held_out.name
        train_examples = [ex for ex in dataset.examples if ex.session != name]
        if not train_examples:
            continue
        profile = _train_fold(model_name, Dataset(train_examples, feature_order), epochs=epochs, seed=seed)
        folds.append(
            _Fold(
                name=name,
                profile=profile,
                raw_path=held_out / "raw.csv",
                clicks=read_label_intervals(held_out / "labels.jsonl"),
                phases=read_phase_windows(held_out / "labels.jsonl"),
                recording_bounds_ms=read_recording_bounds(held_out / "raw.csv"),
            )
        )
    return folds


def default_decision_grid() -> list[dict[str, float]]:
    """A small, sensible grid over the runtime decision gates."""
    grid: list[dict[str, float]] = []
    for enter, exit_, min_event, enter_dwell in product(
        (0.6, 0.7, 0.8), (0.4, 0.5), (50, 120, 200), (2, 3)
    ):
        if exit_ >= enter:
            continue
        grid.append(
            {
                "enter_threshold": enter,
                "exit_threshold": exit_,
                "enter_dwell_frames": enter_dwell,
                "release_dwell_frames": 2,
                "min_event_ms": min_event,
                "refractory_ms": 80,
            }
        )
    return grid


def runtime_default_decision() -> dict[str, float]:
    """Return the detector defaults as the frozen comparison baseline."""
    return dataclasses.asdict(BinaryDecisionConfig())


def frozen_decision_grid() -> list[dict[str, float]]:
    """Predeclared low-multiplicity comparison for resource-safe nested runs.

    The first row is the runtime default. The second is the previously
    documented tuned threshold configuration. Restricting the acceptance run
    to these two already-declared candidates avoids a fresh broad search over
    the same small set of recordings.
    """
    return [
        runtime_default_decision(),
        {
            "enter_threshold": 0.8,
            "exit_threshold": 0.4,
            "enter_dwell_frames": 2,
            "release_dwell_frames": 2,
            "min_event_ms": 200,
            "refractory_ms": 80,
        },
    ]


def evaluate_decision_sweep(
    sessions_root: Path,
    *,
    feature_order: list[str],
    model_name: str,
    decision_grid: list[dict[str, float]] | None = None,
    epochs: int = 200,
    seed: int = 7,
    ignore_margin_ms: int = 80,
    pre_tol_ms: int = 200,
    post_tol_ms: int = 200,
    include_glob: str = "*",
    exclude: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Sweep runtime decision gates for one model family.

    Each fold is trained once; the (config-independent) model is then replayed
    under every decision config. Returns one aggregated row per decision config.
    """
    grid = decision_grid if decision_grid is not None else default_decision_grid()
    folds = _build_folds(
        sessions_root,
        feature_order=feature_order,
        model_name=model_name,
        epochs=epochs,
        seed=seed,
        ignore_margin_ms=ignore_margin_ms,
        include_glob=include_glob,
        exclude=exclude,
    )
    return _evaluate_decisions_on_folds(
        folds,
        grid,
        pre_tol_ms=pre_tol_ms,
        post_tol_ms=post_tol_ms,
    )


def _evaluate_decisions_on_folds(
    folds: list[_Fold],
    grid: list[dict[str, float]],
    *,
    pre_tol_ms: int,
    post_tol_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in grid:
        scores = []
        for fold in folds:
            profile = dataclasses.replace(fold.profile, decision=dict(decision))
            switch_events = replay_session(profile, fold.raw_path)
            scores.append(
                score_session(
                    fold.name, switch_events, fold.clicks, fold.phases,
                    recording_bounds_ms=fold.recording_bounds_ms,
                    pre_tol_ms=pre_tol_ms, post_tol_ms=post_tol_ms,
                )
            )
        rows.append({**decision, **aggregate_scores(scores)})
    return rows


_DECISION_KEYS = (
    "enter_threshold",
    "exit_threshold",
    "enter_dwell_frames",
    "release_dwell_frames",
    "min_event_ms",
    "refractory_ms",
)


def _decision_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {key: row[key] for key in _DECISION_KEYS}


def assess_candidate_acceptance(
    baseline_scores: list[SessionScore],
    candidate_scores: list[SessionScore],
    *,
    min_false_down_reduction: float = 0.30,
    max_session_false_down_increase: int = 1,
    max_median_onset_regression_ms: float = 25.0,
) -> dict[str, Any]:
    """Apply the frozen baseline-relative contract to outer-fold scores.

    Delivery retention is checked per intended-click index, not merely by an
    aggregate count, so a candidate cannot lose one baseline-delivered press
    and compensate by detecting a different press.
    """
    if not 0.0 <= min_false_down_reduction <= 1.0:
        raise ValueError("minimum false-down reduction must be between 0 and 1")

    def by_session(scores: list[SessionScore], label: str) -> dict[str, SessionScore]:
        mapped = {score.session: score for score in scores}
        if len(mapped) != len(scores):
            raise ValueError(f"{label} scores contain duplicate session names")
        for score in scores:
            detected = set(score.detected_click_indices)
            if len(detected) != score.n_detected or any(index < 0 or index >= score.n_clicks for index in detected):
                raise ValueError(f"{label} score has inconsistent delivered-click identities: {score.session}")
        return mapped

    baseline_by_session = by_session(baseline_scores, "baseline")
    candidate_by_session = by_session(candidate_scores, "candidate")
    if not baseline_by_session or baseline_by_session.keys() != candidate_by_session.keys():
        raise ValueError("baseline and candidate must score the same non-empty outer sessions")

    lost_delivered: dict[str, list[int]] = {}
    false_down_increase: dict[str, int] = {}
    for session, baseline in baseline_by_session.items():
        candidate = candidate_by_session[session]
        if baseline.n_clicks != candidate.n_clicks:
            raise ValueError(f"baseline and candidate click counts differ for {session}")
        if (
            baseline.rest_seconds != candidate.rest_seconds
            or baseline.artifact_seconds != candidate.artifact_seconds
        ):
            raise ValueError(f"baseline and candidate exposure differs for {session}")
        lost = sorted(set(baseline.detected_click_indices) - set(candidate.detected_click_indices))
        if lost:
            lost_delivered[session] = lost
        false_down_increase[session] = candidate.n_false_downs - baseline.n_false_downs

    baseline_aggregate = aggregate_scores(baseline_scores)
    candidate_aggregate = aggregate_scores(candidate_scores)
    baseline_rate = baseline_aggregate["false_downs_per_negative_min"]
    candidate_rate = candidate_aggregate["false_downs_per_negative_min"]
    reduction_fraction: float | None = None
    if isinstance(baseline_rate, (int, float)) and isinstance(candidate_rate, (int, float)) and baseline_rate > 0:
        reduction_fraction = (float(baseline_rate) - float(candidate_rate)) / float(baseline_rate)

    baseline_latencies = [value for score in baseline_scores for value in score.down_latencies_ms]
    candidate_latencies = [value for score in candidate_scores for value in score.down_latencies_ms]
    latency_regression_ms: float | None = None
    if baseline_latencies and candidate_latencies:
        latency_regression_ms = float(median(candidate_latencies) - median(baseline_latencies))

    criteria = {
        "false_down_reduction": (
            reduction_fraction is not None and reduction_fraction >= min_false_down_reduction
        ),
        "delivered_press_retention": not lost_delivered,
        "per_session_false_down_increase": all(
            increase <= max_session_false_down_increase for increase in false_down_increase.values()
        ),
        "median_onset_regression": (
            latency_regression_ms is not None
            and latency_regression_ms <= max_median_onset_regression_ms
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": {
            "min_false_down_reduction": min_false_down_reduction,
            "max_session_false_down_increase": max_session_false_down_increase,
            "max_median_onset_regression_ms": max_median_onset_regression_ms,
        },
        "metrics": {
            "baseline_false_downs_per_negative_min": baseline_rate,
            "candidate_false_downs_per_negative_min": candidate_rate,
            "false_down_reduction_fraction": reduction_fraction,
            "lost_baseline_delivered_click_indices": lost_delivered,
            "false_down_increase_by_session": false_down_increase,
            "median_onset_regression_ms": latency_regression_ms,
        },
    }


def evaluate_nested_decision_selection(
    sessions_root: Path,
    *,
    feature_order: list[str],
    model_name: str,
    decision_grid: list[dict[str, float]] | None = None,
    epochs: int = 200,
    seed: int = 7,
    ignore_margin_ms: int = 80,
    pre_tol_ms: int = 200,
    post_tol_ms: int = 200,
    include_glob: str = "*",
    exclude: tuple[str, ...] = (),
) -> tuple[list[SessionScore], dict[str, Any], list[dict[str, Any]]]:
    """Nested LOSO evaluation with decision selection isolated from outer folds.

    For each outer held-out session, decision gates are chosen by an inner LOSO
    sweep over only the remaining sessions. The outer session is replayed once
    with that frozen decision, so it never selects its own runtime gates.
    """
    grid = decision_grid if decision_grid is not None else default_decision_grid()
    if not grid:
        raise ValueError("nested decision selection needs at least one decision config")
    baseline_decision = runtime_default_decision()
    if not any(_decision_from_row(row) == baseline_decision for row in grid):
        raise ValueError("nested decision selection grid must include the runtime-default baseline")
    session_dirs = list_session_dirs(sessions_root, include_glob=include_glob, exclude=exclude)
    dataset = load_interval_sessions(sessions_root, feature_order=feature_order, ignore_margin_ms=ignore_margin_ms)
    selected_names = {path.name for path in session_dirs}
    dataset = Dataset([ex for ex in dataset.examples if ex.session in selected_names], feature_order)
    if len({example.session for example in dataset.examples}) < 3:
        raise ValueError("nested decision selection needs at least 3 labeled sessions")

    outer_scores: list[SessionScore] = []
    baseline_outer_scores: list[SessionScore] = []
    selections: list[dict[str, Any]] = []
    for outer_dir in session_dirs:
        inner_dirs = [path for path in session_dirs if path.name != outer_dir.name]
        inner_folds = _build_folds_for_dirs(
            inner_dirs,
            dataset,
            feature_order=feature_order,
            model_name=model_name,
            epochs=epochs,
            seed=seed,
        )
        inner_rows = _evaluate_decisions_on_folds(
            inner_folds,
            grid,
            pre_tol_ms=pre_tol_ms,
            post_tol_ms=post_tol_ms,
        )
        ranked = rank_sweep_rows(inner_rows, min_detection_rate=1.0)
        selected = ranked[0]
        decision = _decision_from_row(selected)
        gate_met = (
            float(selected.get("detection_rate", 0.0)) >= 1.0
            and float(selected.get("down_latency_ms_median", 1e9)) <= 200.0
            and float(selected.get("down_latency_ms_p95_abs", 1e9)) <= 400.0
            and selected.get("false_downs_rest_per_min") is not None
            and selected.get("false_downs_artifact_per_min") is not None
            and selected.get("false_downs_per_negative_min") is not None
        )

        train_examples = [example for example in dataset.examples if example.session != outer_dir.name]
        profile = _train_fold(
            model_name,
            Dataset(train_examples, feature_order),
            epochs=epochs,
            seed=seed,
        )
        raw_path = outer_dir / "raw.csv"
        clicks = read_label_intervals(outer_dir / "labels.jsonl")
        phases = read_phase_windows(outer_dir / "labels.jsonl")
        recording_bounds_ms = read_recording_bounds(raw_path)
        selected_profile = dataclasses.replace(profile, decision=dict(decision))
        outer_score = score_session(
            outer_dir.name,
            replay_session(selected_profile, raw_path),
            clicks,
            phases,
            recording_bounds_ms=recording_bounds_ms,
            pre_tol_ms=pre_tol_ms,
            post_tol_ms=post_tol_ms,
        )
        outer_scores.append(outer_score)
        if decision == baseline_decision:
            baseline_outer_scores.append(outer_score)
        else:
            baseline_profile = dataclasses.replace(profile, decision=dict(baseline_decision))
            baseline_outer_scores.append(
                score_session(
                    outer_dir.name,
                    replay_session(baseline_profile, raw_path),
                    clicks,
                    phases,
                    recording_bounds_ms=recording_bounds_ms,
                    pre_tol_ms=pre_tol_ms,
                    post_tol_ms=post_tol_ms,
                )
            )
        selections.append(
            {
                "outer_session": outer_dir.name,
                "inner_sessions": [path.name for path in inner_dirs],
                "inner_gate_met": gate_met,
                "selection_status": "accepted" if gate_met else "fallback_no_gate_met",
                "decision": decision,
                "inner_metrics": {
                    key: selected.get(key)
                    for key in (
                        "detection_rate",
                        "false_downs_total",
                        "false_downs_per_negative_min",
                        "down_latency_ms_median",
                        "down_latency_ms_p95_abs",
                    )
                },
            }
        )
    aggregate = aggregate_scores(outer_scores)
    aggregate["all_inner_gates_met"] = all(selection["inner_gate_met"] for selection in selections)
    aggregate["predeclared_grid"] = grid == frozen_decision_grid()
    acceptance = assess_candidate_acceptance(baseline_outer_scores, outer_scores)
    aggregate["acceptance_contract"] = acceptance
    aggregate["acceptance_passed"] = bool(
        aggregate["predeclared_grid"]
        and aggregate["all_inner_gates_met"]
        and acceptance["passed"]
    )
    return outer_scores, aggregate, selections


def rank_sweep_rows(
    rows: list[dict[str, Any]],
    *,
    min_detection_rate: float = 0.9,
    max_down_latency_ms: float = 200.0,
    max_p95_abs_latency_ms: float = 400.0,
) -> list[dict[str, Any]]:
    """Order sweep rows for *usable clicker* behavior.

    Prefer configs that meet the delivery and latency floor, then minimize all
    unmatched down activations. Cancelled and out-of-phase downs remain in that
    primary count, so increasing ``min_event_ms`` cannot game the ranking.
    """

    def metric(row: dict[str, Any], name: str, default: float) -> float:
        value = row.get(name)
        if not isinstance(value, (int, float)) or not isfinite(float(value)):
            return default
        return float(value)

    def key(row: dict[str, Any]) -> tuple:
        detection_rate = metric(row, "detection_rate", 0.0)
        median_latency = metric(row, "down_latency_ms_median", 1e9)
        p95_latency = metric(row, "down_latency_ms_p95_abs", 1e9)
        total_rate = metric(row, "false_downs_per_negative_min", 1e9)
        artifact_rate = metric(row, "false_downs_artifact_per_min", 1e9)
        rest_rate = metric(row, "false_downs_rest_per_min", 1e9)
        meets = (
            detection_rate >= min_detection_rate
            and median_latency <= max_down_latency_ms
            and p95_latency <= max_p95_abs_latency_ms
            and total_rate < 1e9
            and artifact_rate < 1e9
            and rest_rate < 1e9
        )
        false_metrics = (
            total_rate,
            metric(row, "false_downs_total", 1e9),
            metric(row, "lifecycle_faults", 1e9),
            metric(row, "false_downs_other", 1e9),
            artifact_rate,
            rest_rate,
        )
        simple_tie_break = (
            metric(row, "enter_dwell_frames", 1e9),
            metric(row, "min_event_ms", 1e9),
        )
        if meets:
            return (0, *false_metrics, -detection_rate, median_latency, p95_latency, *simple_tie_break)
        # If no row satisfies the gate, preserve sensitivity first rather than
        # selecting an always-off config merely because it has zero false downs.
        return (1, -detection_rate, median_latency, p95_latency, *false_metrics, *simple_tie_break)

    return sorted(rows, key=key)
