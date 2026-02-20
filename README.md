# reson

EMG-driven Morse input pipeline for ESP32 + AD8232 streams.

## Pipeline

`serial -> parser -> detector -> timing -> morse -> UI`

## ESP32 firmware

Firmware is versioned in this repo:

- `/Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream/esp32_emg_stream.ino`
- `/Users/jeraldyuan/dev/reson/firmware/README.md`

Serial contract expected by Reson:
- space-delimited integer rows: `t raw env`
- `baud=230400`
- `~250 Hz` stream

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
- On macOS, launch `reson-debug` and `reson-gui` from the native Terminal app.
- VS Code integrated terminal is best-effort for Qt apps.

## Run

Debug monitor:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10 --detector hmm3
```

GUI app:

```bash
reson-gui --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3
```

`--port` is optional (auto-detect fallback).  
Only one process may own a serial port at once; Reson enforces lockfiles under `.reson_locks/`.

## Detector modes

- `hmm3` (default): v2.5.2 3-state decoder (`REST/PRESS/ARTIFACT`) + segment-level `light/heavy` at `UP`.
- `adaptive`: v2.4 raw-first adaptive detector.
- `threshold`: legacy threshold fallback.

## v2.5.2 core algorithm (hmm3)

Signal path:
- Decisions use `raw` only.
- `env_in` is debug/log only.
- `raw -> filtered_raw_hp -> framed features -> logistic emissions -> fixed-lag Viterbi`.

Frame settings:
- Window `120 ms`, hop `30 ms`.
- Timestamp-based framing from `t_ms` (not fixed sample count).

Features (default):
- `rms_state`
- `lf_energy_ratio`
- `slope_burst`
- `waveform_length`

Decode:
- States: `REST`, `PRESS`, `ARTIFACT`.
- Emissions: multiclass logistic regression.
- HMM: fixed-lag Viterbi (`lag_frames=4`).
- Transition blend: `A_final = 0.8 * A_prior + 0.2 * A_estimated`.

Safety/event layer:
- `DOWN` on confirmed `REST->PRESS` (dwell-gated).
- `UP` on confirmed `PRESS->REST` (release dwell-gated).
- Keeps `min_event_ms`, `refractory_ms`, `min_rest_gap_ms`, blip policy.
- If `ARTIFACT` occurs mid-press: cancel active segment, emit no `UP`, require REST re-arm.

Segment-level light/heavy (at `UP` only):
- Segment stats: `duration_ms`, `peak_u`, `auc_u`, `mean_u`.
- Heavy if any 2 of 4 thresholds pass; else light.

Runtime adaptation:
- Normalization drift updates are allowed only when:
  - decoded state is `REST`
  - no pending transition
  - rest-confidence frame minimum is met
  - artifact gate is off
- Drift is bounded by per-feature caps (`drift_cap_per_min`).

Startup phases:
- `BOOTSTRAP` -> `ARMING` -> `RUNNING`
- No Morse lifecycle events before `RUNNING`.

## Calibration / profile

Profile file: `.reson_profile.json`

Calibration is optional for runtime startup, but recommended for quality.  
Profile contains v2.5.2 keys:
- `model_version`
- `detector_mode`
- `feature_config`
- `feature_hash`
- `normalization`
- `classifier`
- `hmm`
- `segment_thresholds`
- `decision_gates`
- `metadata`

Runtime hard-fails if profile `feature_hash` does not match runtime feature ordering.

## Debug monitor telemetry

`reson-debug` panes:
1. raw + filtered
2. feature traces
3. emission probabilities (`p_rest`, `p_press`, `p_artifact`)
4. decoded state with DOWN/UP markers

Optional replay log:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --detector hmm3 --log-file debug_log.csv
```

Locked CSV order:
`t_ms,raw,env_in,filtered_raw_hp,rms_state,lf_energy_ratio,slope_burst,waveform_length,p_rest,p_press,p_artifact,decoded_state,phase,armed,artifact_gated,down,up,press_class,segment_duration_ms,segment_peak_u,segment_auc,segment_mean_u,segment_class`

## Safe shutdown / recovery

- Prefer normal app close (`Ctrl+C` or window close) before unplugging ESP32.
- On unplug/replug, app remains alive and auto-retries serial reconnect.
- To inspect stream directly:
  - `pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw`

Recovery checklist:
1. `ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure a single owner:
   `pkill -f "pyserial-miniterm|reson-debug|reson-gui"`
3. Launch one app on one port.

## Tests

```bash
python -m pytest -q
```
