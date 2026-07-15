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

## Adversarial Held-Out Evaluation

The earlier preliminary table and its 100%-detection statement have been
withdrawn. An adversarial audit found that the legacy scorer used an overly
broad match window, excluded cancelled `down` activations, omitted
`false_downs_other` from aggregate ranking, and selected runtime gates after
inspecting the same outer folds used for the reported result.

The corrected scorer matches click onset within `[-200, +200] ms`, counts every
unmatched emitted `down`, and reports completed false clicks as a subset. Gate
selection now has a nested LOSO path. Results over all five main prompted
sessions, including the formerly excluded `prompt-gui-004`, are:

| Threshold gates | Delivered | False downs | Rest/min | Artifact/min | Other | Onset median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime default (`0.6`, 50 ms minimum) | 53/100 | 156 | 9.10 | 18.00 | 62 | 136 ms |
| Previously tuned (`0.8`, 200 ms minimum) | 46/100 | 79 | 0.91 | 6.00 | 64 | 142 ms |

The tuned gates reduce false activations but lose seven clicks already
delivered by the weak default baseline, so they fail the frozen acceptance
contract. In the nested run, neither candidate reached the 100% inner delivery
gate in any outer fold. No improvement was accepted.

The evaluator now applies that contract programmatically to outer-fold
runtime-default and candidate scores. It checks delivered-click identities,
not just aggregate detection counts, and refuses a passing exit status unless
both the inner eligibility gates and all four baseline-relative criteria pass.

See `docs/adversarial_evaluation.md` for the frozen criteria, exact commands,
metric semantics, critic findings, and remaining validation boundary. These
are retrospective results from one wearer and one sitting, not cross-day or
cross-user evidence.

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
