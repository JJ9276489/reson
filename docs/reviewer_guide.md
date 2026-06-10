# Reviewer Guide

This guide is for a technical reviewer trying to understand Reson quickly.

## What This Repo Is

Reson is a narrow binary EMG switch pipeline:

```text
serial -> parser -> feature frames -> trained binary model -> switch down/up events
```

It is not a complete HMI system. It does not own eye tracking, pointer movement, click injection, or UI automation.

## Read These First

1. `README.md`: current status, run commands, limits.
2. `ARCHITECTURE.md`: source-level architecture map.
3. `docs/validation_status.md`: what is implemented/tested/validated/hypothesized.
4. `docs/data_collection_protocol.md`: how data should be collected.
5. `docs/data_schema.md`: files written by recording and training.

## Core Source Files

| Question | File |
| --- | --- |
| What is the serial contract parser? | `src/reson/parser.py` |
| How does serial reconnect work? | `src/reson/serial_io.py` |
| Where are features computed? | `src/reson/features.py` |
| Where is model runtime? | `src/reson/binary_model.py` |
| How are switch events represented? | `src/reson/types.py`, `src/reson/switch.py` |
| How are session files created? | `src/reson/recording.py` |
| How does training load labels? | `src/reson/training.py` |

## Applications

| Command | Purpose | File |
| --- | --- | --- |
| `reson-debug` | Qt signal/features monitor, visual prompted recorder, and manual interval recorder | `src/reson/apps/debug_monitor.py` |
| `reson-prompt-record` | Terminal timed prompt recorder with automatic interval labels | `src/reson/apps/prompt_record_app.py` |
| `reson-record` | Terminal interval recorder | `src/reson/apps/record_app.py` |
| `reson-train` | Train a binary model profile | `src/reson/apps/train_app.py` |
| `reson-study` | Run model/feature/data scaling sweeps | `src/reson/apps/study_app.py` |
| `reson-switch` | Emit switch JSONL events from a trained profile | `src/reson/apps/switch_app.py` |
| `reson-eval` | Leave-one-session-out event-level evaluation | `src/reson/apps/eval_app.py`, `src/reson/evaluation.py` |
| `reson-clicker` | Demo click target driven by a trained model (live or replay) | `src/reson/apps/clicker_app.py`, `src/reson/clicker.py` |

## Minimal Sanity Path

From repo root:

```bash
python -m pytest -q
```

With hardware connected:

```bash
pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw
```

Expected firmware row:

```text
12345 2089 37
```

Record visual prompted data:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --record-dir sessions/review-001 --prompt --prompt-trials 10 --prompt-no-bell
```

Train a simple baseline:

```bash
reson-train --sessions sessions --model threshold --features wl --out models/review_wl_threshold.json
```

Run switch output:

```bash
reson-switch --port /dev/cu.usbserial-XXXX --baud 230400 --profile models/review_wl_threshold.json --status
```

## Core vs Exploratory

Core:

- Serial ingestion.
- Parser.
- Timestamp-based feature extraction.
- Interval recording.
- Simple baseline training.
- Switch JSONL output.

Exploratory:

- CNN/TCN/Transformer models.
- Feature-set scaling studies.
- Current AD8232 hardware as an EMG source.
- Any claim about robust HMI performance.

## Generated Artifacts

The following are intentionally ignored by git:

- `sessions/`: recorded data.
- `models/`: trained profiles and metrics.
- `studies/`: scaling-study outputs.
- `.venv/`, `.pytest_cache/`, `build/`, PlatformIO `.pio/` outputs.

Review source and docs by default. Review generated artifacts only when they are intentionally shared for a specific experiment.

## Main Questions To Ask

Useful review questions:

- Is the serial contract stable and sufficiently tested?
- Are feature frames computed in a timestamp-safe way?
- Are labels represented as intervals rather than point clicks?
- Are simple baselines reported before larger models?
- Does evaluation use held-out sessions rather than random frames from the same session?
- Are hardware limitations clearly separated from software limitations?

Questions not yet answerable from this repo alone:

- Is this reliable enough for production HMI use?
- Does it generalize across users?
- Does it work with a better EMG front end?
- Does it remain usable over long sessions?
