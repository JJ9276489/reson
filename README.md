# reson

EMG-driven Morse input pipeline for ESP32 + AD8232 streams.

## Pipeline

`serial -> parser -> detector -> timing -> morse -> UI`

## Install

```bash
cd /Users/jeraldyuan/dev/reson # replace with your username
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install '.[dev]'
```

Use Python 3.11 or 3.12. Python 3.14 is unsupported for this project.

Terminal-first guidance:
- On macOS, launch `reson-debug` and `reson-gui` from the native Terminal app.
- VS Code integrated terminal is best-effort for Qt apps and may be unstable depending on environment.

## Run

Debug monitor:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10
```

GUI app:

```bash
reson-gui --port /dev/cu.usbserial-XXXX --baud 230400
```

`--port` is optional. If omitted, reson attempts auto-detect.

Serial ownership rule:
- Only one process can open a serial port at a time.
- Reson enforces this with a lockfile and exits with a clear message if port is already in use.

## Calibration

Calibration is optional in v2.4.1.

- If `.reson_profile.json` exists, detector thresholds/timing/filter defaults are loaded from it.
- If profile is missing, the adaptive detector starts with conservative defaults.
- Profile generation still uses rest/light/heavy capture when generated.

## Safe shutdown / unplug

- Preferred flow: close `reson-debug`/`reson-gui` (or Ctrl+C) before unplugging ESP32.
- If unplug happens while app is running, the app stays alive and auto-retries reconnect.
- Replug the device and wait; streaming should resume without restarting the app.

## Recovery playbook

1. Check serial device names:
`ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure no competing process owns the port:
`pkill -f "pyserial-miniterm|reson-debug|reson-gui"`
3. Start one app only:
- `reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10`
- or `reson-gui --port /dev/cu.usbserial-XXXX --baud 230400`
4. If no lines appear, test raw stream:
`pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw`

## Adaptive detector (v2.4.1)

Adaptive mode is raw-first and deterministic:

- Source of truth: `raw`
- Incoming `env` is debug-only
- Processing path: `raw -> filtered_raw_hp -> rms_state -> u -> state machine`

Feature definitions:
- `filtered_raw_hp`: Python filter path (high-pass, notch, low-pass)
- `rms_state`: rolling RMS of `filtered_raw_hp`
- `rest_center`: robust REST-only center (median)
- `rest_scale`: robust REST-only spread (`max(p90-p50, 1.4826*MAD, rest_scale_floor)`)
- `u = (rms_state - rest_center) / max(rest_scale, rest_scale_floor)`

Strict REST-only adaptation:
- `baseline_raw`, `rest_center`, and `rest_scale` update only when all are true:
  - stable state is `rest`
  - no pending transition
  - `rest_confident` is true
  - artifact gate is off

Artifact gate:
- Uses low-frequency energy ratio + slope burst score
- Has hysteresis and holdoff
- While gated:
  - force candidate state to REST
  - suppress DOWN/UP emission
  - clear pending transition and dwell timers
  - restart rest-confidence accumulation

State/event behavior:
- Stable states: `rest`, `light`, `heavy`
- `DOWN` emitted on confirmed `rest -> light/heavy`
- `UP` emitted on confirmed return to rest
- Press class is latched from DOWN to UP (no intra-press escalation)
- Safety gates remain:
  - min dwell
  - min event duration (blip suppression)
  - rest-gap before next press
  - post-release refractory

Startup phases:
- `BOOTSTRAP`: initialize from lowest-variance quiet RMS windows
- `ARMING`: wait for confident rest
- `RUNNING`: emit DOWN/UP

## Debug monitor telemetry

`reson-debug` shows stacked shared-time plots:

1. raw (+ high-passed filtered overlay)
2. `rms_state`, `rest_center`, `rest_scale`
3. `u` with `u_light_enter` and `u_heavy_enter`
4. state (0 rest, 1 light, 2 heavy) with DOWN/UP + gate markers

Optional replay logging:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --log-file debug_log.csv
```

CSV columns:
`t_ms,raw,env_in,filtered_raw_hp,rms_state,rest_center,rest_scale,u,lf_energy,artifact_ratio,artifact_score,artifact_gated,rest_confident,phase,armed,state,down,up,press_class`

## Tests

```bash
python -m pytest -q
```
