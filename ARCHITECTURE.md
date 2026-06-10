# Reson Architecture

Reson is currently a binary switch pipeline. It reads `t raw env` samples from ESP32 firmware, extracts timestamp-based features, applies a trained binary model, and emits switch `down` / `up` events for downstream HMI software.

## Product Boundary

```text
ESP32 serial stream
  -> parse sample lines
  -> extract feature frames
  -> run trained binary detector
  -> emit switch events
```

Reson does not own pointer control, eye tracking, UI automation, or click injection. Those belong in downstream software.

## Runtime Flow

```mermaid
flowchart LR
    A["ESP32 firmware"] --> B["SerialReader"]
    B --> C["parse_line"]
    C --> D["FeatureFrameExtractor"]
    D --> E["BinaryModelDetector"]
    E --> F["EdgeEvent"]
    F --> G["SwitchEvent JSONL"]
```

Key files:

| Layer | File | Responsibility |
| --- | --- | --- |
| Firmware | `firmware/` | ESP32 ADC sampling and `t raw env` serial output |
| Serial | `src/reson/serial_io.py` | Open/read/reconnect/close serial ports |
| Parsing | `src/reson/parser.py` | Parse serial text into `EmgSample` |
| Features | `src/reson/features.py` | High-pass/notch/low-pass path and frame features |
| Model runtime | `src/reson/binary_model.py` | Threshold/logreg/optional torch binary detector |
| Switch conversion | `src/reson/switch.py` | Convert edge events to JSON-safe switch events |
| Recording schema | `src/reson/recording.py` | Shared session file fields, metadata, JSONL helpers |
| Training | `src/reson/training.py` | Load interval sessions and fit binary profiles |

## Applications

| Command | File | Purpose |
| --- | --- | --- |
| `reson-debug` | `src/reson/apps/debug_monitor.py` | Qt monitor, live feature plots, optional model overlay, visual prompted/manual interval recording |
| `reson-prompt-record` | `src/reson/apps/prompt_record_app.py` | Terminal timed prompt recorder with automatic interval labels |
| `reson-record` | `src/reson/apps/record_app.py` | Terminal interval recorder |
| `reson-train` | `src/reson/apps/train_app.py` | Train one model or all supported model families |
| `reson-study` | `src/reson/apps/study_app.py` | Run feature/model/data-scale sweeps |
| `reson-switch` | `src/reson/apps/switch_app.py` | Emit trained switch events as JSONL |

## Data And Model Loop

```mermaid
flowchart TD
    A["Collect interval session"] --> B["raw.csv"]
    A --> C["features.csv"]
    A --> D["labels.jsonl"]
    B --> E["reson-train"]
    C --> E
    D --> E
    E --> F["binary_profile.json"]
    F --> G["reson-switch"]
```

The preferred label unit is an interval, not a point click marker. A label interval has explicit `label_start` and `label_end` records. Training labels frames inside intervals as active and frames outside intervals as rest, with a configurable ignore margin around interval boundaries.

## Model Families

Reson currently supports:

| Model | Dependency | Use |
| --- | --- | --- |
| `threshold` | standard install | Sanity baseline, usually waveform-length only |
| `logreg` | standard install | Fast interpretable binary classifier |
| `cnn` | `.[ml]` | Optional sequence model |
| `tcn` | `.[ml]` | Optional sequence model |
| `transformer` | `.[ml]` | Optional sequence model |

Simple baselines are first-class, not placeholders. With limited single-channel biosignal data, a simple waveform-length threshold can be more informative than a larger model. The sequence models exist to support scaling studies once there is enough data to justify them.

## Event Contract

The external switch contract is intentionally small:

```json
{"type":"switch","phase":"down","t_ms":12345,"duration_ms":0,"source_state":"active"}
{"type":"switch","phase":"up","t_ms":12580,"duration_ms":235,"source_state":"active"}
```

Internal `EdgeEvent` records are converted to `SwitchEvent` records before output. Downstream consumers should depend on switch events, not model internals.

## Review Notes

- `env` from firmware is debug-only. Current detector decisions are based on raw-derived Python features.
- AD8232 is ECG-grade hardware and is not an ideal EMG front end. Treat current results as prototype evidence, not final hardware validation.
- The debug monitor is intentionally practical and still broad: it combines UI, recording, plotting, and optional model overlay. The shared recording schema has been extracted, but UI decomposition can continue later.
- `sessions/`, `models/`, and `studies/` are ignored generated artifacts. Expert review should focus on tracked source, firmware, docs, and tests unless sample data is intentionally shared.
