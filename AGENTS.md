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

- GUI does 3-stage capture: rest, light, heavy.
- Derived thresholds/hysteresis are saved to `.reson_profile.json`.

## Detector and timing behavior

- Detector stable states: `rest`, `light`, `heavy`.
- Hysteresis + hold times reduce noise flips.
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
reson-debug --port ... --baud 230400
reson-gui --port ... --baud 230400
```

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
