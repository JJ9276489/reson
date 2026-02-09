# reson

EMG-driven Morse input pipeline for ESP32 + AD8232 streams.

## Pipeline

`serial -> parser -> detector -> timing -> morse -> UI`

## Install

```bash
cd /Users/jeraldyuan/dev/reson # replace with your username
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
```

Use Python 3.11 or 3.12. Python 3.14 currently causes unreliable editable installs and Qt runtime issues on macOS for this project.
On some Python 3.11 builds, editable installs (`-e`) may be skipped because `__editable__*.pth` is treated as hidden. If `reson-debug` cannot import `reson`, reinstall non-editable with `pip install '.[dev]'`.
If Qt reports missing `cocoa` plugin, force-reinstall PySide6 in the pinned range:
`python -m pip install --force-reinstall 'PySide6>=6.7,<6.8' 'PySide6_Addons>=6.7,<6.8' 'PySide6_Essentials>=6.7,<6.8' 'shiboken6>=6.7,<6.8'`

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
- Reson now enforces this with a lockfile and exits with a clear message if port is already in use.

## Calibration

Calibration is optional in v2.2.1.

- If `.reson_profile.json` exists, detector thresholds/timing defaults are loaded from it.
- If profile is missing, the adaptive detector starts with built-in conservative defaults.
- Profile still uses 3-stage rest/light/heavy capture when generated.

## Safe shutdown / unplug

- Preferred flow: close `reson-debug`/`reson-gui` (or Ctrl+C) before unplugging ESP32.
- If you unplug accidentally while app is running, the app stays alive and auto-retries reconnect.
- Replug the device and wait; streaming should resume without restarting the app.

## Recovery playbook

1. Check serial device names:
`ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Make sure no competing process owns the port:
`pkill -f \"pyserial-miniterm|reson-debug|reson-gui\"`
3. Start one app only:
- `reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10`
- or `reson-gui --port /dev/cu.usbserial-XXXX --baud 230400`
4. If no lines appear, test raw stream:
`pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw`

## Behavior

- Adaptive detector is raw-first:
  - source of truth: `raw`
  - ESP32 `env` is debug-only
  - features: `rect=abs(raw-baseline_raw)`, `fast EMA`, `slow EMA`, `a=fast-slow`, `z=a/sigma`
  - `sigma` is computed from REST-only `a_rest` buffer
- Edge detector outputs `rest`, `light`, `heavy`.
- Detector emits lifecycle phases:
  - `DOWN` on `rest -> light/heavy` after dwell
  - `UP` on return to rest
  - class is latched from DOWN until UP
- Morse safety gates are enforced:
  - min dwell to enter press states
  - min event duration (blip rejection)
  - required clean rest gap before next accepted press
  - post-release refractory
- Light press produces dot; heavy press produces dash.
- Rest gaps resolve buffered symbols into letters/spaces with adaptive Morse timing.
- Focus target can toggle between text and backspace using a reserved control token.

## Debug monitor telemetry

`reson-debug` now shows stacked shared-time plots:

1. raw
2. fast + slow
3. z with thresholds
4. state (0 rest, 1 light, 2 heavy) with DOWN/UP markers

Optional logging for replay/tuning:

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --log-file debug_log.csv
```

CSV columns:
`t_ms,raw,env_in,fast,slow,a,sigma,z,state,down,up,press_class`

## Tests

```bash
pytest -q
```
