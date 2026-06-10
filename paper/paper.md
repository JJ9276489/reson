---
title: "Reson: an open-source EMG binary switch pipeline on low-cost hardware"
tags:
  - Python
  - electromyography
  - assistive technology
  - accessibility
  - human-machine interface
authors:
  - name: Jerald Yuan
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

Reson is an open-source pipeline that turns a single surface
electromyography (EMG) channel into a binary switch signal — discrete
`down`/`up` events suitable for use as a "click" in assistive
human-machine interfaces such as eye-tracking cursor systems. The
pipeline covers the full path from hardware to control signal: ESP32
firmware streaming ADC samples over serial at approximately 250 Hz, a
fault-tolerant Python ingestion layer with automatic reconnection,
timestamp-windowed feature extraction, prompted and interval-labeled
data collection applications, training paths for threshold, logistic
regression, and optional sequence models, and a runtime that emits
switch events as JSONL.

The reference hardware is deliberately inexpensive: an ESP32
development board and an AD8232 analog front end, totaling under $50.

<!-- TODO(validation): one-sentence headline result from the
multi-session validation study, e.g. false-activation rate at rest and
during artifact-only movement, missed-click rate, and down latency on
held-out sessions. -->

# Statement of need

Commercial assistive switches and EMG interfaces are expensive,
closed, or both, which limits experimentation by researchers,
clinicians, and end users. People with motor impairments who retain
reliable muscle activation (for example jaw clench) can use a binary
switch as a high-value input channel, but building one today requires
assembling firmware, signal processing, labeling, training, and
runtime components from scratch.

Reson packages that path end to end with an explicit, documented
validation methodology. The project separates implemented
functionality from validated performance claims
(`docs/validation_status.md`), defines event-level metrics
(false activations per minute at rest and under artifact, missed
intended clicks, down/up latency), and ships a prompted data-collection
protocol that avoids keyboard-labeling artifacts during signal
windows.

<!-- TODO(validation): summarize the dataset (number of sessions,
reattachment between sessions, rest-only and artifact-only segments)
and the baseline comparison (waveform-length threshold vs.
waveform-length logistic regression vs. all-feature logistic
regression), with held-out-session results. Cite the Zenodo dataset
DOI. -->

# Functionality

- **Firmware**: ESP32 sketch streaming `t raw env` rows at ~250 Hz.
- **Ingestion**: serial reader with reconnect behavior and per-port
  lockfiles.
- **Feature extraction**: timestamp-windowed features derived from raw
  ADC samples, including waveform length.
- **Data collection**: `reson-record` (interval labels),
  `reson-prompt-record` (prompted protocol with settle/rest/click/
  artifact phases), and `reson-debug` (live visualization).
- **Training**: `reson-train` fits threshold and logistic-regression
  profiles, with an optional PyTorch sequence model.
- **Runtime**: `reson-switch` loads a trained profile and emits JSONL
  `down`/`up` events.
- **Testing**: the behavior of the parser, serial lifecycle, recording
  schema, training loaders, and event emission is covered by a
  hardware-free test suite run in CI.

# Acknowledgements

<!-- TODO: anyone who tested hardware, gave feedback, or reviewed. -->

# References
