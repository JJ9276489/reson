from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


RAW_RECORD_FIELDS = ["host_time_s", "t_ms", "raw", "env", "line"]

BASE_FEATURE_RECORD_FIELDS = [
    "host_time_s",
    "t_ms",
    "window_start_ms",
    "window_end_ms",
    "env_in",
    "filtered_raw_hp",
    "rms_state",
    "lf_energy_ratio",
    "slope_burst",
    "waveform_length",
]

DEBUG_FEATURE_RECORD_FIELDS = BASE_FEATURE_RECORD_FIELDS + ["active", "probability", "down", "up"]


def default_session_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / stamp


def write_jsonl(handle: IO[str], payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    handle.flush()


def build_session_meta(
    *,
    port: str,
    baud: int,
    label_mode: str,
    label: str,
    source: str,
    label_keys: list[str] | None = None,
    label_key: str | None = None,
    quit_key: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "port": port,
        "baud": baud,
        "serial_contract": "t raw env",
        "label_schema": "interval_v1",
        "label_mode": label_mode,
        "label": label,
        "source": source,
        "files": {"raw": "raw.csv", "features": "features.csv", "labels": "labels.jsonl"},
    }
    if label_keys is not None:
        meta["label_keys"] = label_keys
    if label_key is not None:
        meta["label_key"] = label_key
    if quit_key is not None:
        meta["quit_key"] = quit_key
    if notes:
        meta["notes"] = notes
    return meta


@dataclass
class RecordingSession:
    directory: Path
    raw_file: IO[str]
    feature_file: IO[str]
    label_file: IO[str]
    raw_writer: csv.DictWriter
    feature_writer: csv.DictWriter

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        meta: dict[str, Any],
        feature_fields: list[str] | None = None,
    ) -> "RecordingSession":
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        raw_file = (directory / "raw.csv").open("w", encoding="utf-8", newline="")
        feature_file = (directory / "features.csv").open("w", encoding="utf-8", newline="")
        label_file = (directory / "labels.jsonl").open("w", encoding="utf-8")

        raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_RECORD_FIELDS)
        feature_writer = csv.DictWriter(feature_file, fieldnames=feature_fields or BASE_FEATURE_RECORD_FIELDS)
        raw_writer.writeheader()
        feature_writer.writeheader()

        return cls(
            directory=directory,
            raw_file=raw_file,
            feature_file=feature_file,
            label_file=label_file,
            raw_writer=raw_writer,
            feature_writer=feature_writer,
        )

    def write_label(self, payload: dict[str, object]) -> None:
        write_jsonl(self.label_file, payload)

    def close(self) -> None:
        self.label_file.close()
        self.raw_file.close()
        self.feature_file.close()
