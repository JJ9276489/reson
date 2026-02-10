# AGENTS

## Project purpose

Reson converts ESP32 EMG serial data into Morse-like jaw input.

Pipeline:

`serial -> parser -> detector -> timing -> morse -> UI`

## Canonical run commands

```bash
reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --window-sec 10
reson-gui --port /dev/cu.usbserial-XXXX --baud 230400
```

Run context:
- Use macOS Terminal as canonical launch context for Qt apps.
- VS Code integrated terminal is best-effort and may fail depending on env/plugin state.

## Serial assumptions

- ESP32 emits exactly `t raw env` integer rows at about 250 Hz.
- Only one process can own the serial port at a time.
- Debug app and GUI app are separate entrypoints and should not share a port concurrently.
- Reson uses per-port lockfiles under `.reson_locks/` to enforce single ownership.

## Calibration expectations

- Calibration is optional.
- If `.reson_profile.json` is present, detector tuning defaults are loaded.
- If absent, adaptive detector runs with built-in defaults.
- Profile generation still follows rest/light/heavy stages when used.

## Detector and timing behavior

- Adaptive detector is raw-first:
  - Use `raw` for detection features.
  - Treat incoming `env` as debug-only.
  - Filter raw path in Python (`high-pass -> notch -> low-pass`) before rectification.
  - Compute `fast`, `slow`, `a=fast-slow`, and `z=a/sigma` from filtered raw.
  - Update `baseline_raw`, `slow`, and `sigma` only during confidently REST.
  - `sigma` must be computed from REST-only `a_rest`, never mixed-state buffers.
- Detector startup phases:
  - `BOOTSTRAP`: initialize baseline/sigma from quiet windows; suppress DOWN/UP.
  - `ARMING`: wait for strict confirmed REST.
  - `RUNNING`: normal event emission.
- Detector stable states: `rest`, `light`, `heavy`.
- Lifecycle phases:
  - `DOWN` emitted on `rest -> light/heavy` after dwell.
  - `UP` emitted on return to rest.
  - Press class is latched from DOWN to UP (no light->heavy escalation in v2.2.1).
- Safety gates:
  - min dwell enter criteria
  - min event duration + blip policy
  - min clean rest gap before next accepted press
  - refractory after release
- Timing module uses adaptive Morse unit estimates.
- Gap rules resolve letter and word boundaries.

## Control tokens

- Focus toggle token: `...-.-`
- Clear symbol-buffer token: `........`

## Backspace semantics

- When focus is `backspace`, each completed `light` or `heavy` event removes one character.

## Dev workflow

Install:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
```

Use Python 3.11/3.12 (not 3.14) to avoid Qt plugin/runtime and editable-install issues.

Test:

```bash
pytest -q
```

Run:

```bash
reson-debug --port ... --baud 230400 --detector adaptive
reson-gui --port ... --baud 230400 --detector adaptive
```

Debug replay logging:

```bash
reson-debug --port ... --baud 230400 --log-file debug_log.csv
```

Logged fields:
`t_ms,raw,env_in,filtered_raw,fast,slow,a,sigma,z,phase,armed,state,down,up,press_class`

Safe shutdown:
- Prefer normal app close / Ctrl+C before unplugging ESP32.
- If unplug occurs during runtime, apps should stay alive and auto-reconnect when port returns.

Recovery checklist:
1. `ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure single owner: `pkill -f \"pyserial-miniterm|reson-debug|reson-gui\"`
3. Relaunch one app on one port.
4. Validate stream with `pyserial-miniterm ... --raw` if needed.

## Extensibility notes

- To replace threshold gating with ML, add new detector module(s) in `src/reson/` and keep detector output contract `rest|light|heavy`.
- Keep parser output shape stable (`EmgSample`) unless parser/tests/docs are updated together.

## Guardrails

- Do not change serial protocol parsing without updating parser tests and docs in same change.
- Keep `reson-debug` and `reson-gui` as distinct entrypoints.
- Any behavioral change must include tests or updated acceptance checks.
