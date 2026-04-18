# reson

EMG-driven binary switch service for ESP32 + AD8232 streams.

Reson is the biosignal switch layer for click/no-click control. The runtime goal is simple:

`serial -> parser -> detector -> switch down/up events`

The public control boundary is binary. Detector internals may still expose richer states for debugging, but consumers should only depend on switch `down/up`.

## ESP32 firmware

Firmware is versioned in this repo:

- `/Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream/esp32_emg_stream.ino`
- `/Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream_cpp/` (PlatformIO C++)
- `/Users/jeraldyuan/dev/reson/firmware/README.md`

Serial contract expected by Reson:
- space-delimited integer rows: `t raw env`
- `baud=230400`
- approximately `250 Hz` stream

Firmware tooling install:

```bash
cd /Users/jeraldyuan/dev/reson
python -m pip install -r requirements-firmware.txt
```

Convenience commands:

```bash
make firmware-upload
make firmware-monitor
```

## Hardware setup

Prototype photo:

![Reson hardware prototype](docs/images/hardware_setup.jpg)

Current wiring (AD8232 -> ESP32):

| AD8232 pin | ESP32 pin |
| --- | --- |
| `3.3V` | `3V3` |
| `GND` | `GND` |
| `OUTPUT` | `GPIO34` |

Notes:
- `LO+` / `LO-` are not used by current firmware.
- Current prototype uses ECG-grade front-end hardware (AD8232) for jaw EMG experimentation.
- This is a non-medical prototype and is not intended for diagnosis or treatment.

Electrode guidance (masseter tests):
- Place one active electrode over the masseter region, one nearby reference, and one reference/ground on a relatively low-motion area.
- Keep cable strain low and skin prep consistent to reduce motion artifacts.

## Install

```bash
cd /Users/jeraldyuan/dev/reson
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install '.[dev]'
```

Supported Python: 3.11 / 3.12.  
Unsupported: 3.14.

Terminal-first guidance:
- On macOS, launch Qt tools like `reson-debug` from the native Terminal app.
- VS Code integrated terminal is best-effort for Qt apps.

## Runtime

Binary switch JSONL stream:

```bash
reson-switch --port /dev/cu.usbserial-XXXX --baud 230400
```

Example output:

```json
{"type":"switch","phase":"down","t_ms":12345,"duration_ms":0,"source_state":"active","host_time_s":1776400000.123}
{"type":"switch","phase":"up","t_ms":12580,"duration_ms":235,"source_state":"active","host_time_s":1776400000.358}
```

This is the intended integration point for `/Users/jeraldyuan/dev/eye-cursor`: consume `down/up` events and let `eye-cursor` own click semantics.

Debug monitor:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10 --detector hmm3
```

Manual labeled recording:

```bash
reson-record --port /dev/cu.usbserial-XXXX --baud 230400 --out sessions/test-001
```

Visual labeled recording with live plots:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3 --feature-ablation wl-only --record-dir sessions/test-visual-001
```

While recording:
- in `reson-record`, press `c` to mark and `q` to stop
- in visual `reson-debug --record-dir`, press `Space`/`c` or click the button to mark, then close the window to stop

Session output:
- `meta.json`: setup metadata
- `raw.csv`: host time, ESP32 `t_ms`, raw, env, original line
- `features.csv`: frame-level features including `waveform_length`
- `labels.jsonl`: manual click marks and session start/end events

Feature ablation, useful for testing whether waveform length is enough:

```bash
reson-switch --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3 --feature-ablation wl-only
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3 --feature-ablation wl-only
```

`--feature-ablation` options:
- `all`
- `wl-only` (default for `reson-switch`)
- `rms-only`
- comma list of feature names (`rms_state,lf_energy_ratio,slope_burst,waveform_length`)

`--port` is optional (auto-detect fallback).  
Only one process may own a serial port at once; Reson enforces lockfiles under `.reson_locks/`.

## Detector modes

- `hmm3`: 3-state decoder (`REST/PRESS/ARTIFACT`) with logistic emissions + fixed-lag Viterbi.
- `adaptive`: raw-first adaptive detector retained as a fallback.

Both modes are collapsed into binary switch events at the app/pipeline boundary:
- detector `DOWN` -> switch `down`
- detector `UP` -> switch `up`
- rest/artifact/noise -> no switch event

`hmm3` still has internal compatibility fields from earlier experiments. They are not part of the product-level output.

## Data collection / profile

Profile file: `.reson_profile.json`

The current priority is collecting reusable binary click data with `reson-record` or `reson-debug --record-dir`. Profiles remain supported for `hmm3`, but new model fitting should be built from recorded sessions rather than one-shot live calibration.

## Debug monitor telemetry

`reson-debug` panes:
1. raw + filtered
2. waveform length + RMS features
3. binary active/rest state with DOWN/UP markers

Optional log:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3 --log-file debug_log.csv
```

CSV order:
`t_ms,raw,env_in,filtered_raw_hp,rms_state,lf_energy_ratio,slope_burst,waveform_length,p_rest,p_press,p_artifact,decoded_state,phase,armed,artifact_gated,down,up,press_class,segment_duration_ms,segment_peak_u,segment_auc,segment_mean_u,segment_class`

## Safe shutdown / recovery

- Prefer normal app close (`Ctrl+C` or window close) before unplugging ESP32.
- On unplug/replug, apps remain alive and auto-retry serial reconnect.
- To inspect stream directly:
  - `pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw`

Recovery checklist:
1. `ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure a single owner:
   `pkill -f "pyserial-miniterm|reson-debug|reson-switch"`
3. Launch one app on one port.

## Tests

```bash
python -m pytest -q
```
