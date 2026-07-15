# Reson Data Schema

This document describes the files written by `reson-debug --record-dir ...`, `reson-debug --prompt --record-dir ...`, `reson-prompt-record --out ...`, and `reson-record --out ...`, plus the trained binary profile consumed by `reson-switch`.

## Session Directory

A recording session is a directory with this shape:

```text
sessions/example-session/
  meta.json
  raw.csv
  features.csv
  labels.jsonl
```

Generated session directories are ignored by git by default.

## `meta.json`

Session metadata.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Session schema version. Current value: `1` |
| `created_at_utc` | string | ISO timestamp for session creation |
| `port` | string | Serial port used for recording |
| `baud` | integer | Serial baud rate |
| `serial_contract` | string | Expected firmware line format, currently `t raw env` |
| `label_schema` | string | Current value: `interval_v1` |
| `label_mode` | string | `prompted`, `hold`, or `toggle` |
| `label` | string | Current click label, usually `CLICK` |
| `source` | string | Recording app, for example `reson-debug-prompt`, `reson-prompt-record`, `reson-debug`, or `reson-record` |
| `files` | object | Relative file names for raw, features, and labels |

Optional fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `label_keys` | array | Hold-mode keys, usually `["space", "c"]` |
| `label_key` | string | Toggle-mode label key |
| `quit_key` | string | Toggle-mode quit key |
| `notes` | string | User notes from terminal recording |
| `prompt_protocol` | object | Timed phase configuration for prompted sessions |

## `raw.csv`

One row per parsed serial sample.

| Column | Type | Meaning |
| --- | --- | --- |
| `host_time_s` | float string | Host wall-clock time from Python `time.time()` |
| `t_ms` | integer | ESP32 timestamp in milliseconds |
| `raw` | integer | ESP32 ADC reading |
| `env` | integer | Firmware-side debug envelope |
| `line` | string | Original normalized serial line |

The `env` field is retained for debugging. The current binary model path uses raw-derived Python features for decisions.

## `features.csv`

One row per feature frame. Frames are built by timestamp window, not by assuming an exact serial sample rate.

Base columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `host_time_s` | float string | Host wall-clock time near frame creation |
| `t_ms` | integer | Frame end timestamp |
| `window_start_ms` | integer | Frame window start timestamp |
| `window_end_ms` | integer | Frame window end timestamp |
| `env_in` | integer | Latest firmware envelope inside the frame |
| `filtered_raw_hp` | float | Latest filtered high-pass raw value for the frame |
| `rms_state` | float | RMS over filtered raw values in the frame |
| `lf_energy_ratio` | float | Low-frequency energy ratio feature |
| `slope_burst` | float | Short-vs-long slope burst feature |
| `waveform_length` | float | Sum of absolute frame-to-frame filtered sample movement |

`reson-debug` may add live overlay columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `active` | integer | Current detector state, `1` active and `0` rest |
| `probability` | float | Model active probability |
| `down` | integer | `1` if a down event occurred on this row |
| `up` | integer | `1` if an up event occurred on this row |

## `labels.jsonl`

One JSON object per line. The important labels are explicit intervals.

Session start:

```json
{"type":"session_start","host_time_s":1776400000.0,"t_ms":null}
```

Click interval:

```json
{"type":"label_start","label":"CLICK","host_time_s":1776400001.2,"t_ms":123000}
{"type":"label_end","label":"CLICK","host_time_s":1776400001.5,"t_ms":123340}
```

Prompt phase marker:

```json
{"type":"prompt_phase","phase":"CLICK 4/20","host_time_s":1776400001.2,"t_ms":123000,"duration_s":1.0,"label":"CLICK"}
```

Session end:

```json
{"type":"session_end","host_time_s":1776400030.0,"t_ms":153000,"samples":7500,"frames":996,"labels":20,"parse_errors":0}
```

Training treats frames inside `label_start` / `label_end` intervals as active. Frames near interval boundaries can be ignored with `--ignore-margin-ms` to avoid punishing label timing error.

## Binary Profile

Trained profiles are JSON files, usually written to `models/binary_profile.json`.

Top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Profile schema version. Current value: `1` |
| `detector_mode` | string | Current value: `binary` |
| `model_type` | string | `threshold`, `logreg`, `cnn`, `tcn`, or `transformer` |
| `feature_order` | array | Ordered features expected by the model |
| `feature_hash` | string | Hash guard for feature order compatibility |
| `normalization` | object | Per-feature center, scale, and floor values |
| `model` | object | Model-specific parameters |
| `decision` | object | Runtime threshold/dwell/min-event config |
| `feature_config` | object | Frame window and hop configuration |
| `metadata` | object | Training metadata |

For `threshold`, `model` contains:

```json
{"feature":"waveform_length","threshold":12.3,"softness":4.0}
```

For `logreg`, `model` contains:

```json
{"weights":[1.0,-0.2],"bias":-0.5}
```

For torch sequence models, `model` contains `config`, `seq_len`, and a JSON-serialized `state_dict`.

## Switch Output

`reson-switch` writes JSONL switch events to stdout.

```json
{"type":"switch","phase":"down","t_ms":12345,"duration_ms":0,"source_state":"active","host_time_s":1776400000.123}
{"type":"switch","phase":"up","t_ms":12580,"duration_ms":235,"source_state":"active","host_time_s":1776400000.358}
```

`phase` is one of `down`, `up`, or `cancel`. Every `down` is followed by exactly
one terminal event: `up` when the press cleared `min_event_ms`, or `cancel` for a
shorter transient (a `down`/`cancel` pair delivers no click). Consumers never see
two consecutive `down` events or a `down` without a terminal.

Downstream consumers should depend on this switch schema rather than training profile internals.

For in-process integration, use `reson.api.ResonSwitch`. Its `SwitchUpdate`
returns the same `SwitchEvent` objects plus the current detector `probability`
and `is_active` state. Event-level `confidence` is optional and is not populated
by the current detector; consumers that need a continuous confidence-like value
should read `SwitchUpdate.probability`.
