# Validation Status

This document separates implemented functionality from actual validation evidence.

## Status Summary

Reson is an early-stage biosignal switch prototype. The software path is implemented well enough to collect data, train preliminary models, and emit switch events. The control claim is not yet validated.

## Implemented

| Area | Current status |
| --- | --- |
| Firmware stream | ESP32 firmware emits `t raw env` at approximately 250 Hz |
| Parser | Python parser accepts valid rows and rejects malformed rows |
| Serial lifecycle | Runtime has reconnect behavior and per-port lockfiles |
| Feature extraction | Timestamp-windowed raw-derived features are implemented |
| Recording | Prompted, visual, and terminal interval-label recorders write session directories |
| Training | Threshold, logistic regression, and optional torch sequence model training paths exist |
| Runtime | `reson-switch` emits JSONL `down` / `up` switch events from a trained profile |
| Tests | Unit/smoke tests cover parser, serial doubles, recording helpers, training fixtures, and switch events |

## Tested

The test suite checks software behavior against synthetic fixtures and test doubles. It does not validate EMG control performance.

Covered by tests:

- Parser behavior for serial lines.
- Binary profile loading and event emission on controlled synthetic samples.
- Training data loading from interval labels.
- Basic threshold/logistic model fitting on fixture data.
- Port lock behavior.
- Serial reconnect behavior with simulated failures.
- Recording schema helper behavior.

Not covered by tests:

- Real electrode placement variability.
- Real motion artifact rejection.
- Multi-session generalization.
- Human-in-the-loop click throughput or fatigue.
- Downstream pointer control usability.

## Current Evidence

Current evidence should be treated as prototype evidence:

- The hardware can stream ADC samples into the Python stack.
- The debug monitor can visualize raw and feature traces.
- Interval-labeled sessions can be recorded.
- Preliminary models can be trained from local sessions.
- Waveform length appears to be a useful feature in early local inspection.

This is not enough evidence to claim robust EMG switching.

## Preliminary Held-Out Evaluation

`reson-eval` runs leave-one-session-out, event-level evaluation: for each
session it trains on the others and replays the held-out session through the
runtime detector. The numbers below are over **4 clean prompted sessions
(`prompt-gui-001/002/003/005`), all recorded in a single sitting on one day**,
excluding the contaminated `prompt-gui-004` and the older heterogeneous
`interval-001`/`prompt-001` recordings. This is a within-sitting check, not a
cross-day or cross-user result.

Default decision gates:

| Config | Detect | FP rest/min | FP artifact/min | Down latency (ms) |
| --- | --- | --- | --- | --- |
| threshold:wl | 100% | 3.41 | 21.0 | 188 |
| logreg:wl | 100% | 5.69 | 20.0 | 168 |
| logreg:all | 98% | 4.23 | 29.0 | 168 |

A decision-gate sweep (`reson-eval --sweep`) over both single-feature families
selects the same gates, with `threshold:wl` best:

> enter_threshold 0.8, exit_threshold 0.4, enter_dwell 2, release_dwell 2,
> min_event_ms 200, refractory_ms 80

Tuned `threshold:wl` (held-out): **100% detection, 0.49 rest false downs/min,
6.0 artifact false downs/min, 198 ms down latency.** Raising `min_event_ms` and
the enter threshold cut rest false positives ~7x and artifact false positives
~3.5x while keeping every click and staying under 200 ms.

Caveats: single sitting, one wearer, four sessions; artifact false positives
(~6/min) are still the dominant failure mode; held-out across *days* and across
*users* remains untested. These are encouraging prototype numbers, not a
validated control claim.

## Hypotheses

Current working hypotheses:

- Waveform length may capture most of the useful single-channel activation signal.
- Simple threshold or logistic regression baselines may be more appropriate than larger sequence models until more labeled data exists.
- Motion artifacts and cable/electrode disturbances are likely dominant failure modes.
- IMU or mechanical stabilization may eventually reduce false positives more than larger ML models.

These are hypotheses, not established results.

## Hardware Limits

The current AD8232 board is ECG-grade hardware. It is being used experimentally for jaw EMG because it is available and easy to wire, not because it is the right final front end.

Expected limits:

- Motion artifact sensitivity.
- Electrode placement sensitivity.
- Cable movement sensitivity.
- Limited bandwidth/control over analog conditioning.
- Unclear separation between jaw activation and non-click facial movement.

The software should be evaluated with these limits in mind.

## What Good Performance Means

Frame accuracy on one session is not sufficient. A useful binary switch should be judged by event-level and session-level behavior.

Primary metrics:

- False `down` events per minute during rest.
- False `down` events per minute during artifact-only movement.
- Missed intended click intervals per minute.
- Down latency after intended activation.
- Up latency after intended release.
- Event duration error relative to interval labels.
- Held-out-session performance.

Secondary metrics:

- Precision/recall/F1 on frame labels.
- Stability across port reconnects and app restarts.
- Sensitivity to threshold/decision-gate settings.

## Minimum Evidence Before Stronger Claims

Before making stronger claims, collect a dataset with:

- At least 5-10 sessions.
- Reattachment or repositioning between some sessions.
- Rest-only segments.
- Intentional click/clench intervals.
- Artifact-only segments.
- Held-out sessions reserved for final comparison.

A useful comparison should include at least:

- Waveform-length threshold baseline.
- Waveform-length logistic regression.
- All-feature logistic regression.
- Optional sequence models only if there is enough data.

## Current Validation Boundary

Reasonable claim today:

> Reson is an implemented early-stage binary EMG switch pipeline with recording, training, and runtime paths. It is ready to generate evidence.

Not yet reasonable:

> Reson is a validated reliable EMG click interface.
