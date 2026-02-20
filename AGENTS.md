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
- Firmware source of truth:
  - `firmware/esp32_emg_stream/esp32_emg_stream.ino`
  - `firmware/esp32_emg_stream_cpp/` (PlatformIO C++)
  - `firmware/README.md`

## Calibration expectations

- Calibration is optional.
- If `.reson_profile.json` is present, detector tuning defaults are loaded.
- If absent, detector uses built-in defaults.

## Detector behavior (v2.5.2 locked)

- Default mode is `hmm3`.
- Decisions use `raw` only; `env_in` is debug-only.
- Decoder states: `REST`, `PRESS`, `ARTIFACT`.
- Processing: timestamp-based framed features -> logistic emissions -> fixed-lag Viterbi.

Feature defaults:
- `window_ms=120`
- `hop_ms=30`
- `feature_order = [rms_state, lf_energy_ratio, slope_burst, waveform_length]`

Runtime normalization/adaptation:
- Runtime adaptation updates only when:
  - decoded state is `REST`
  - pending transition is `None`
  - rest-confidence frame minimum is met
  - artifact gate is off
- Runtime adaptation uses bounded drift caps per minute.

Transition model:
- `A_final = 0.8 * A_prior + 0.2 * A_estimated`.
- ARTIFACT should prefer self/REST, not direct PRESS.

Lifecycle events:
- Emit `DOWN` on confirmed `REST->PRESS` after enter dwell.
- Emit `UP` on confirmed `PRESS->REST` after release dwell.
- Keep min-event/refractory/rest-gap and blip policy.
- If ARTIFACT appears mid-press:
  - cancel active segment
  - emit no `UP`
  - require REST re-arm before next press

Segment class mapping:
- `light/heavy` is decided only at `UP` from segment stats:
  - `duration_ms`, `peak_u`, `auc_u`, `mean_u`
- Heavy if any 2 of 4 heavy thresholds pass; otherwise light.

Startup phases:
- `BOOTSTRAP`: collect quiet windows and initialize normalization.
- `ARMING`: require REST confidence.
- `RUNNING`: allow lifecycle events.

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

Firmware tooling (PlatformIO):

```bash
python -m pip install -r requirements-firmware.txt
```

Test:

```bash
python -m pytest -q
```

Run:

```bash
reson-debug --port ... --baud 230400 --detector hmm3
reson-gui --port ... --baud 230400 --detector hmm3
```

Firmware helpers:

```bash
make firmware-upload
make firmware-monitor
```

Debug replay logging:

```bash
reson-debug --port ... --baud 230400 --log-file debug_log.csv
```

Logged fields:
`t_ms,raw,env_in,filtered_raw_hp,rms_state,lf_energy_ratio,slope_burst,waveform_length,p_rest,p_press,p_artifact,decoded_state,phase,armed,artifact_gated,down,up,press_class,segment_duration_ms,segment_peak_u,segment_auc,segment_mean_u,segment_class`

Safe shutdown:
- Prefer normal app close / Ctrl+C before unplugging ESP32.
- If unplug occurs during runtime, apps should stay alive and auto-reconnect.

Recovery checklist:
1. `ls /dev/cu.usbserial* /dev/tty.usbserial* 2>/dev/null`
2. Ensure single owner: `pkill -f "pyserial-miniterm|reson-debug|reson-gui"`
3. Relaunch one app on one port.
4. Validate stream with `pyserial-miniterm ... --raw` if needed.

## Extensibility notes

- New detector modules must keep external `EdgeEvent` compatibility.
- Keep `make_detector(mode, profile)` stable and preserve `threshold` fallback mode.
- Keep parser output shape stable (`EmgSample`) unless parser/tests/docs are updated together.

## Guardrails

- Do not change serial protocol parsing without updating parser tests and docs in the same change.
- Keep `reson-debug` and `reson-gui` as distinct entrypoints.
- Any behavioral change must include tests and docs updates.
