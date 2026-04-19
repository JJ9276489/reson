# reson

EMG-driven binary switch pipeline for ESP32 + AD8232 streams.

Reson's product boundary is intentionally simple:

```text
serial -> parser -> feature frames -> trained binary model -> switch down/up events
```

The intended downstream consumer is `/Users/jeraldyuan/dev/eye-cursor`.

Review references:

- Architecture map: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Session/model schemas: [`docs/data_schema.md`](docs/data_schema.md)

## Install

```bash
cd /Users/jeraldyuan/dev/reson
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install '.[dev]'
```

Supported Python: 3.11 / 3.12. Unsupported: 3.14.

For CNN/TCN/Transformer training:

```bash
python -m pip install -e '.[dev,ml]'
```

## Firmware

Firmware is versioned in this repo:

- `/Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream/esp32_emg_stream.ino`
- `/Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream_cpp/`
- `/Users/jeraldyuan/dev/reson/firmware/README.md`

Serial contract:

```text
t raw env
```

Defaults:

- baud: `230400`
- stream rate: approximately `250 Hz`
- raw: ESP32 ADC reading
- env: firmware-side debug envelope

Firmware helper commands:

```bash
make firmware-upload
make firmware-monitor
```

## Hardware

Prototype photo:

![Reson hardware prototype](docs/images/hardware_setup.jpg)

Current wiring:

| AD8232 pin | ESP32 pin |
| --- | --- |
| `3.3V` | `3V3` |
| `GND` | `GND` |
| `OUTPUT` | `GPIO34` |

Notes:

- `LO+` / `LO-` are not used by current firmware.
- Current prototype uses ECG-grade AD8232 hardware for jaw EMG experimentation.
- This is a non-medical prototype and is not intended for diagnosis or treatment.

## Collect Data

Visual interval recording is the preferred path:

```bash
reson-debug \
  --port /dev/cu.usbserial-XXXX \
  --baud 230400 \
  --record-dir sessions/interval-001
```

During recording:

- hold `Space` or `c` during the intended click/clench
- release when the click/clench ends
- close the window to stop

Terminal-only fallback:

```bash
reson-record \
  --port /dev/cu.usbserial-XXXX \
  --baud 230400 \
  --out sessions/interval-001 \
  --status
```

Terminal mode uses toggle intervals:

- press `c` once for `label_start`
- press `c` again for `label_end`
- press `q` to stop

Session files:

- `meta.json`: setup metadata
- `raw.csv`: host time, ESP32 `t_ms`, raw, env, original line
- `features.csv`: frame-level model features
- `labels.jsonl`: interval labels and session boundaries

Label events:

```json
{"type":"label_start","label":"CLICK","t_ms":123000}
{"type":"label_end","label":"CLICK","t_ms":123340}
```

## Train Models

Train a waveform-length logistic model:

```bash
reson-train \
  --sessions sessions \
  --model logreg \
  --features wl \
  --out models/binary_profile.json
```

Train all currently supported model families:

```bash
reson-train \
  --sessions sessions \
  --model all \
  --features all \
  --epochs 100 \
  --hidden 16 \
  --out models/run-001 \
  --allow-skip-optional
```

Model families:

- `threshold`: single-feature binary threshold baseline
- `logreg`: pure-Python logistic regression over frame features
- `cnn`: optional PyTorch 1D CNN over sliding feature windows
- `tcn`: optional PyTorch temporal convolution model
- `transformer`: optional PyTorch tiny transformer over sliding feature windows

Feature presets:

- `wl`: waveform length only
- `core`: waveform length + RMS
- `all`: waveform length + RMS + slope burst + low-frequency ratio

## Run Switch Output

Run a trained binary model:

```bash
reson-switch \
  --port /dev/cu.usbserial-XXXX \
  --baud 230400 \
  --profile models/binary_profile.json
```

Output is JSONL:

```json
{"type":"switch","phase":"down","t_ms":12345,"duration_ms":0,"source_state":"active","host_time_s":1776400000.123}
{"type":"switch","phase":"up","t_ms":12580,"duration_ms":235,"source_state":"active","host_time_s":1776400000.358}
```

## Scaling Studies

Run a baseline scaling sweep:

```bash
reson-study \
  --sessions sessions \
  --models threshold,logreg,cnn,tcn,transformer \
  --features wl,core,all \
  --fractions 0.25,0.5,1.0 \
  --epochs 20,100 \
  --hidden 8,16 \
  --out studies/binary_scaling.csv
```

Use this to compare:

- data volume
- feature set
- model family
- epochs
- parameter count

If a simple waveform-length threshold wins, that is useful signal: the hardware/data are not yet asking for a larger sequence model.

## Direct Serial Check

```bash
pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw
```

Exit miniterm with `Ctrl+]`.

## Safe Shutdown / Recovery

- Prefer normal app close (`Ctrl+C` or window close) before unplugging ESP32.
- On unplug/replug, apps auto-retry serial reconnect.
- Only one process may own a serial port at once; Reson uses lockfiles under `.reson_locks/`.

Recovery checklist:

```bash
ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null
pkill -f "pyserial-miniterm|reson-debug|reson-switch|reson-record"
```

## Tests

```bash
python -m pytest -q
```
