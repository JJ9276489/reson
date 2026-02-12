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
- VS Code integrated terminal is best-effort and may fail depending on environment/plugin state.

## Serial assumptions

- ESP32 emits `t raw env` integer rows at about 250 Hz.
- Only one process can own the serial port at a time.
- Debug app and GUI app are separate entrypoints and must not share a port concurrently.
- Reson uses per-port lockfiles under `.reson_locks/`.

## Calibration expectations

- Calibration is optional.
- If `.reson_profile.json` is present, detector tuning defaults are loaded.
- If absent, adaptive detector runs with built-in defaults.
- Profile generation still uses rest/light/heavy stages.

## Detector and timing behavior (v2.4.1)

- Adaptive detector is raw-first:
  - use `raw` for detection
  - treat incoming `env` as debug-only
  - process: `raw -> filtered_raw_hp -> rms_state -> u -> state machine`
- RMS-state normalization:
  - `u = (rms_state - rest_center) / max(rest_scale, rest_scale_floor)`
  - `rest_center` and `rest_scale` are REST-only robust stats
- REST-only learning guard:
  - update `baseline_raw`, `rest_center`, `rest_scale` only when:
    - stable state is `rest`
    - no pending transition
    - `rest_confident` true
    - artifact gate off
- Startup phases:
  - `BOOTSTRAP`: quiet-window initialization
  - `ARMING`: wait for confident REST
  - `RUNNING`: emit DOWN/UP
- Artifact gate:
  - based on low-frequency energy ratio + slope burst
  - while gated: force REST candidate, suppress DOWN/UP, clear pending/dwell, restart rest-confidence timer
- Stable states: `rest`, `light`, `heavy`
- Lifecycle events:
  - `DOWN` on confirmed `rest -> light/heavy`
  - `UP` on confirmed return to REST
  - press class latched from DOWN until UP
- Morse safety gates stay enabled:
  - min dwell
  - min event duration + blip policy
  - min REST gap before next accepted press
  - refractory after release

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
python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install '.[dev]'
```

Use Python 3.11/3.12 (not 3.14).

Test:

```bash
python -m pytest -q
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
`t_ms,raw,env_in,filtered_raw_hp,rms_state,rest_center,rest_scale,u,lf_energy,artifact_ratio,artifact_score,artifact_gated,rest_confident,phase,armed,state,down,up,press_class`

Safe shutdown:
- Prefer normal app close / Ctrl+C before unplugging ESP32.
- If unplug occurs during runtime, apps should stay alive and auto-reconnect.

Recovery checklist:
1. `ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure single owner: `pkill -f "pyserial-miniterm|reson-debug|reson-gui"`
3. Relaunch one app on one port.
4. Validate stream with `pyserial-miniterm ... --raw` if needed.

## Extensibility notes

- New detector modules should keep output contract `rest|light|heavy` and `EdgeEvent` compatibility.
- Keep parser output shape stable (`EmgSample`) unless parser/tests/docs are updated together.

## Guardrails

- Do not change serial protocol parsing without updating parser tests and docs in the same change.
- Keep `reson-debug` and `reson-gui` as distinct entrypoints.
- Any behavioral change must include tests and docs updates.
