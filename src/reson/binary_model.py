from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Any

from reson.features import FeatureFrame, FeatureFrameExtractor, compute_feature_hash
from reson.types import EdgeEvent, EdgeState, EmgSample


DEFAULT_BINARY_FEATURES = ["waveform_length", "rms_state", "slope_burst", "lf_energy_ratio"]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class BinaryDecisionConfig:
    enter_threshold: float = 0.60
    exit_threshold: float = 0.40
    enter_dwell_frames: int = 2
    release_dwell_frames: int = 2
    min_event_ms: int = 50
    refractory_ms: int = 80


@dataclass(frozen=True)
class BinaryModelProfile:
    schema_version: int = 1
    detector_mode: str = "binary"
    model_type: str = "logreg"
    feature_order: list[str] = field(default_factory=lambda: list(DEFAULT_BINARY_FEATURES))
    feature_hash: str = ""
    normalization: dict[str, dict[str, float]] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    feature_config: dict[str, Any] = field(default_factory=lambda: {"window_ms": 120, "hop_ms": 30})
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_hash:
            object.__setattr__(self, "feature_hash", compute_feature_hash(self.feature_order))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BinaryModelProfile":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            detector_mode=str(data.get("detector_mode", "binary")),
            model_type=str(data.get("model_type", "logreg")),
            feature_order=list(data.get("feature_order", DEFAULT_BINARY_FEATURES)),
            feature_hash=str(data.get("feature_hash", "")),
            normalization=dict(data.get("normalization", {})),
            model=dict(data.get("model", {})),
            decision=dict(data.get("decision", {})),
            feature_config=dict(data.get("feature_config", {"window_ms": 120, "hop_ms": 30})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_mode": self.detector_mode,
            "model_type": self.model_type,
            "feature_order": self.feature_order,
            "feature_hash": self.feature_hash,
            "normalization": self.normalization,
            "model": self.model,
            "decision": self.decision,
            "feature_config": self.feature_config,
            "metadata": self.metadata,
        }


def save_binary_profile(profile: BinaryModelProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")


def load_binary_profile(path: Path) -> BinaryModelProfile:
    profile = BinaryModelProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    expected = compute_feature_hash(profile.feature_order)
    if profile.feature_hash != expected:
        raise ValueError("binary profile feature_hash does not match feature_order")
    return profile


def frame_feature_vector(frame: FeatureFrame, feature_order: list[str]) -> list[float]:
    return frame.as_vector(feature_order)


class _TorchSequenceRuntime:
    def __init__(self, profile: BinaryModelProfile):
        try:
            import torch

            from reson.torch_models import build_torch_model
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - exercised only without optional dep at runtime.
            raise RuntimeError("binary torch model requires `python -m pip install '.[ml]'`") from exc

        self.torch = torch
        cfg = dict(profile.model.get("config", {}))
        cfg["input_dim"] = len(profile.feature_order)
        self.seq_len = int(cfg.get("seq_len", profile.model.get("seq_len", 16)))
        self.model = build_torch_model(profile.model_type, cfg)
        state = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in dict(profile.model.get("state_dict", {})).items()
        }
        self.model.load_state_dict(state)
        self.model.eval()
        self.buffer: list[list[float]] = []

    def predict(self, vec: list[float]) -> float:
        self.buffer.append(vec)
        self.buffer = self.buffer[-self.seq_len :]
        if len(self.buffer) < self.seq_len:
            return 0.0
        with self.torch.no_grad():
            x = self.torch.tensor([self.buffer], dtype=self.torch.float32)
            logit = self.model(x).reshape(-1)[0]
            return float(self.torch.sigmoid(logit).item())


class BinaryModelDetector:
    """Frame-level binary detector backed by a trained binary profile."""

    def __init__(self, profile: BinaryModelProfile):
        if profile.detector_mode != "binary":
            raise ValueError("BinaryModelDetector requires detector_mode='binary'")
        self.profile = profile
        self.feature_order = list(profile.feature_order)
        self._center = dict(profile.normalization.get("center", {}))
        self._scale = dict(profile.normalization.get("scale", {}))
        self._floor = dict(profile.normalization.get("floor", {}))
        decision = BinaryDecisionConfig(**{**BinaryDecisionConfig().__dict__, **dict(profile.decision)})
        self.enter_threshold = float(decision.enter_threshold)
        self.exit_threshold = float(decision.exit_threshold)
        self.enter_dwell_frames = int(decision.enter_dwell_frames)
        self.release_dwell_frames = int(decision.release_dwell_frames)
        self.min_event_ms = int(decision.min_event_ms)
        self.refractory_ms = int(decision.refractory_ms)

        feature_cfg = dict(profile.feature_config)
        self.extractor = FeatureFrameExtractor(
            window_ms=int(feature_cfg.get("window_ms", 120)),
            hop_ms=int(feature_cfg.get("hop_ms", 30)),
        )
        self._torch_runtime = (
            _TorchSequenceRuntime(profile)
            if profile.model_type in {"cnn", "tcn", "transformer"}
            else None
        )

        self._stable_active = False
        self._pending_active: bool | None = None
        self._pending_frames = 0
        self._press_start_ms: int | None = None
        self._refractory_until_ms = 0
        self._events: list[EdgeEvent] = []
        self.last_probability = 0.0
        self.last_frame: FeatureFrame | None = None

    def _normalize(self, frame: FeatureFrame) -> list[float]:
        vec = frame_feature_vector(frame, self.feature_order)
        out: list[float] = []
        for key, value in zip(self.feature_order, vec):
            center = float(self._center.get(key, 0.0))
            scale = float(self._scale.get(key, 1.0))
            floor = float(self._floor.get(key, 1.0))
            out.append((float(value) - center) / max(scale, floor))
        return out

    def _predict_probability(self, frame: FeatureFrame) -> float:
        if self.profile.model_type == "threshold":
            feature = str(self.profile.model.get("feature", "waveform_length"))
            value = float(getattr(frame, feature))
            threshold = float(self.profile.model.get("threshold", 0.0))
            softness = max(float(self.profile.model.get("softness", 1.0)), 1e-6)
            return _sigmoid((value - threshold) / softness)

        vec = self._normalize(frame)
        if self.profile.model_type == "logreg":
            weights = [float(v) for v in self.profile.model.get("weights", [])]
            bias = float(self.profile.model.get("bias", 0.0))
            return _sigmoid(sum(w * x for w, x in zip(weights, vec)) + bias)

        if self._torch_runtime is None:
            raise RuntimeError(f"unsupported binary model_type={self.profile.model_type!r}")
        return self._torch_runtime.predict(vec)

    def _target_active(self, probability: float) -> bool:
        if self._stable_active:
            return probability > self.exit_threshold
        return probability >= self.enter_threshold

    def _commit_active(self, active: bool, t_ms: int) -> None:
        if active == self._stable_active:
            return
        self._stable_active = active
        if active:
            self._press_start_ms = t_ms
            self._events.append(
                EdgeEvent(state="active", start_ms=t_ms, end_ms=t_ms, duration_ms=0, phase="down")
            )
            return

        if self._press_start_ms is None:
            return
        duration = max(t_ms - self._press_start_ms, 0)
        # Every committed `down` gets exactly one terminal event. A press that
        # clears min_event_ms ends in `up`; a shorter transient ends in
        # `cancel` so downstream consumers never see a dangling `down`.
        phase = "up" if duration >= self.min_event_ms else "cancel"
        self._events.append(
            EdgeEvent(
                state="active",
                start_ms=self._press_start_ms,
                end_ms=t_ms,
                duration_ms=duration,
                phase=phase,
            )
        )
        self._press_start_ms = None
        self._refractory_until_ms = t_ms + self.refractory_ms

    def update(self, sample: EmgSample) -> EdgeState:
        _, frames = self.extractor.update(sample)
        for frame in frames:
            self.last_frame = frame
            prob = self._predict_probability(frame)
            self.last_probability = prob
            target = self._target_active(prob)
            if target and frame.t_ms < self._refractory_until_ms:
                target = False

            if target == self._stable_active:
                self._pending_active = None
                self._pending_frames = 0
                continue

            if self._pending_active != target:
                self._pending_active = target
                self._pending_frames = 1
            else:
                self._pending_frames += 1

            dwell = self.enter_dwell_frames if target else self.release_dwell_frames
            if self._pending_frames >= dwell:
                self._commit_active(target, frame.t_ms)
                self._pending_active = None
                self._pending_frames = 0

        return "active" if self._stable_active else "rest"

    def pop_events(self) -> list[EdgeEvent]:
        out = self._events
        self._events = []
        return out

    def flush(self, final_t_ms: int) -> list[EdgeEvent]:
        if self._stable_active:
            self._commit_active(False, final_t_ms)
        return self.pop_events()
