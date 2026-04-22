# Demo Plan

This is an honest demo plan for the current maturity level. The goal is to show an early but disciplined signal/control loop, not to imply the system is robust or production-ready.

## Demo Goal

Show that Reson can:

1. Stream biosignal samples from ESP32 firmware.
2. Visualize raw-derived features.
3. Record interval labels.
4. Train a simple binary baseline.
5. Emit switch `down` / `up` events from a trained profile.

## Do Not Imply

Do not imply:

- The AD8232 setup is final EMG hardware.
- The current classifier is robust across days or placements.
- The system is a complete HMI.
- Larger ML models have solved artifact problems.
- A single good live run proves reliability.

## Setup Checklist

Before the demo:

- Confirm firmware emits `t raw env` with miniterm.
- Confirm `reson-debug` opens from a normal terminal.
- Confirm the port name, usually `/dev/cu.usbserial-XXXX` on macOS.
- Have a known baseline profile available if training live fails.
- Keep a short known-good recording session available for training demonstration.

## Suggested Demo Sequence

1. Show the hardware and state the limitation: ESP32 + AD8232 ECG front end used as a prototype EMG source.
2. Run direct serial check:

   ```bash
   pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw
   ```

3. Open debug monitor:

   ```bash
   reson-debug --port /dev/cu.usbserial-XXXX --baud 230400
   ```

4. Show raw and feature response during rest, intentional click/clench, and mild artifact.
5. Record a short interval-labeled session:

   ```bash
   reson-debug --port /dev/cu.usbserial-XXXX --baud 230400 --record-dir sessions/demo-001
   ```

6. Inspect generated files:

   ```bash
   ls sessions/demo-001
   wc -l sessions/demo-001/*.csv sessions/demo-001/labels.jsonl
   ```

7. Train a simple baseline:

   ```bash
   reson-train --sessions sessions --model threshold --features wl --out models/demo_wl_threshold.json
   ```

8. Run switch output:

   ```bash
   reson-switch --port /dev/cu.usbserial-XXXX --baud 230400 --profile models/demo_wl_threshold.json --status
   ```

9. Trigger a few intentional clicks and show JSONL `down` / `up` events.
10. Deliberately show one failure mode if it occurs, such as motion artifact or missed click.

## What To Say If It Fails

Reasonable failure explanation:

> This is exactly the uncertainty the repo is designed to expose. The software path is instrumented enough to collect the failure as data, compare simple baselines, and decide whether the next fix is better hardware, better electrode mechanics, IMU gating, or model changes.

Do not hide failures. For this stage, honest failure characterization is more credible than a cherry-picked live run.

## Best Evidence To Show

Best current evidence:

- Stable serial stream.
- Clear feature movement in debug monitor.
- Correct interval labels in `labels.jsonl`.
- A trained baseline model with metrics.
- Switch events appearing in response to intentional activation.

Evidence not yet available:

- Multi-session held-out performance.
- Long-duration false-positive rate.
- Cross-user results.
- Comparison against better EMG hardware.
