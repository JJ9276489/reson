# Data Collection Protocol

This protocol is for generating interval-labeled data that can actually reduce uncertainty about Reson as a binary switch.

## Goal

Collect synchronized raw ADC samples, raw-derived feature frames, and explicit `CLICK` intervals. The first dataset should answer:

- Does the current hardware produce a separable signal for intentional click/clench?
- How often do rest and artifact periods look like clicks?
- Does a simple waveform-length baseline generalize across sessions?

## Current Hardware Assumptions

Current prototype:

- ESP32 board connected over USB serial.
- AD8232 analog output connected to ESP32 `GPIO34`.
- Firmware emits `t raw env` at approximately 250 Hz and 230400 baud.
- Electrodes are placed for jaw/masseter experimentation.

This protocol assumes the current AD8232 setup. If the analog front end changes, keep the same data format if possible and document the hardware change in `meta.json` notes or an external lab note.

## Recommended Command

Use the visual debug monitor when possible:

```bash
reson-debug \
  --port /dev/cu.usbserial-XXXX \
  --baud 230400 \
  --record-dir sessions/session-001
```

Labeling in the debug monitor:

- Hold `Space` or `c` only while intentionally performing the click/clench.
- Release as soon as the intended click/clench ends.
- Do not label head movement, cable movement, talking, swallowing, or jaw shifts unless the intent is to collect a separate artifact-labeled protocol later.

Terminal fallback:

```bash
reson-record \
  --port /dev/cu.usbserial-XXXX \
  --baud 230400 \
  --out sessions/session-001 \
  --status
```

Terminal labels are toggled:

- Press `c` to start a `CLICK` interval.
- Press `c` again to end it.
- Press `q` to stop recording.

## Good First Dataset Target

For the first useful dataset, collect 5-10 sessions. Treat this as a practical starting point, not a statistical guarantee.

Suggested per-session structure:

1. 20-30 seconds quiet rest.
2. 20-40 intentional click/clench intervals with varied duration and intensity.
3. 20-30 seconds rest after clicks.
4. Artifact-only block: head movement, cable tug, jaw shift without click, talking/swallowing if relevant.
5. Another short click block after artifacts.

Recommended variation:

- Re-seat electrodes between at least some sessions.
- Record on at least two different days before claiming stability.
- Include sessions where the system performs badly; those failures are useful.

## Labeling Rules

Use interval labels, not point labels.

Correct:

```json
{"type":"label_start","label":"CLICK","t_ms":123000}
{"type":"label_end","label":"CLICK","t_ms":123280}
```

Avoid:

- Pressing the label key after the clench is already over.
- Labeling only the peak of the signal.
- Labeling motion artifacts as clicks.
- Mixing different behaviors under the same label without notes.

Boundary precision will never be perfect. Training uses an ignore margin around interval edges to reduce the penalty for human label timing error.

## Files To Check After Recording

Each session should contain:

```text
meta.json
raw.csv
features.csv
labels.jsonl
```

Quick checks:

```bash
wc -l sessions/session-001/raw.csv sessions/session-001/features.csv sessions/session-001/labels.jsonl
head sessions/session-001/raw.csv
head sessions/session-001/features.csv
cat sessions/session-001/labels.jsonl
```

Expected signs of a usable session:

- `raw.csv` has thousands of rows for a normal session.
- `features.csv` has many frame rows.
- `labels.jsonl` has matching `label_start` / `label_end` pairs.
- `session_end` reports low parse errors.
- Click intervals are not all extremely short.

## Common Failure Modes

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| No serial rows | ESP32 not streaming or wrong port | Press ESP32 reset, check `/dev/cu.usbserial*`, run miniterm |
| App says port locked | Another process owns the port | Close `reson-debug`, `reson-switch`, `reson-record`, or miniterm |
| `raw.csv` exists but `features.csv` is tiny | Recording ended too quickly or stream froze | Re-record; verify serial stream first |
| Labels have starts without ends | App closed during active label | The app tries to close active labels; inspect `closed_by` fields |
| Many parse errors | Serial corruption or wrong firmware output format | Verify firmware still emits exactly `t raw env` |
| Click labels do not match visible signal | Human label timing issue | Re-record slower, with deliberate holds |
| Rest produces large feature spikes | Motion/electrode/cable artifact | Record artifact-only segments and consider hardware stabilization |

## Training After Collection

Start simple:

```bash
reson-train \
  --sessions sessions \
  --model threshold \
  --features wl \
  --out models/wl_threshold.json
```

Then compare:

```bash
reson-study \
  --sessions sessions \
  --models threshold,logreg,cnn,tcn,transformer \
  --features wl,core,all \
  --fractions 0.25,0.5,1.0 \
  --epochs 20,100 \
  --hidden 8,16 \
  --out studies/binary_scaling.csv \
  --allow-skip-optional
```

Do not interpret one random frame split as real validation. Reserve complete sessions for held-out evaluation.
