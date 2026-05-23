# ROI Tremor Architecture Notes

## Decision Summary

MotionBloom should not use MediaPipe landmark trajectories as the primary tremor measurement signal. MediaPipe is kept as a coarse hand ROI anchor, while tremor is measured from ROI-local optical flow and signal processing.

## Current Implementation Status

Implemented test-first ROI tremor foundation:

- `motionbloom/analysis/tremor_signal.py` — explainable tremor-band analysis over generic x/y motion signals.
- `motionbloom/analysis/quality_gate.py` — confidence labels and reusable quality gates.
- `motionbloom/tracking/optical_flow.py` — sparse Lucas-Kanade optical-flow tracker using median/MAD aggregation.
- `motionbloom/tracking/roi_tracker.py` — ROI flow wrapper that subtracts optional background/global motion.
- `tests/test_roi_tremor_synthetic.py` — known-truth synthetic validation harness with printed diagnostic objects.

No `app.py` integration has been added yet.

## Intermediate Palm-Relative Landmark Path

Before wiring ROI optical flow into the app, MotionBloom now uses a simpler intermediate signal path:

- The green palm-center dot remains visible as a macro-motion diagnostic. It does not freeze the previous score, but hard physical travel gates now veto tremor scoring when palm motion is too large to plausibly be tremor.
- Raw MediaPipe landmark trajectories are not used directly for tremor scoring when palm-relative data is available.
- Primary tremor candidate signals are normalized palm-relative motion from index, middle, and ring fingertips only.
- Thumb, pinky, and MCP relative streams are kept as supporting/debug signals, not primary scoring inputs.
- Each primary fingertip is analyzed separately; at least 2 of 3 must pass peak/SNR/cycle checks and agree within about 1 Hz.

This reduces whole-hand translation artifacts while keeping normal palm motion, drift, or settling visible as diagnostics. Wide cross-screen or hand-size-relative palm travel is rejected before tremor scoring as physically implausible tremor. It does not replace ROI optical flow; it is the cleaner landmark-based experiment to ship first.

## Physical Impossibility Gates

Palm-center motion is checked before PSD/SNR tremor scoring. The current hard gates are:

- `screen_travel_ratio = palm path length / frame diagonal`; reject above `0.08`.
- `net_screen_displacement_ratio = palm start/end distance / frame diagonal`; reject above `0.05`.
- `hand_relative_travel = palm path length / median hand-box diagonal`; reject above `0.50`.
- `hand_relative_net = palm start/end distance / median hand-box diagonal`; reject above `0.35`.
- `tremor_amp_ratio = tremor peak-to-peak displacement / median hand-box diagonal`; reject above `0.12`, and treat below `0.002` without very strong SNR as no reliable tremor signal.

Rejected windows use the user-facing reason `Wide hand movement — tremor analysis paused` and never receive a research-valid tremor score.

## Architecture

```text
MediaPipe / coarse detector
        ↓
Hand ROI anchor only
        ↓
Sparse LK optical flow inside hand ROI
        ↓
Median/MAD robust motion aggregation
        ↓
Optional background/global flow subtraction
        ↓
ROI-local x/y motion time series
        ↓
Welch PSD + tremor-band diagnostics
        ↓
Quality-gated confidence label
```

## Confidence Labels

The ROI tremor subsystem uses these user-facing confidence labels:

- `Unusable recording`
- `No reliable tremor signal`
- `No tremor detected`
- `Possible rhythmic tremor`
- `Likely rhythmic tremor`

The label is not the whole result. Each analysis window also returns diagnostics including peak frequency, SNR, peak width, band power, gross power, cycles, x/y agreement, valid optical-flow points, track survival rate, FPS, window duration, and reasons.

## Synthetic Validation Cases

The synthetic harness validates these required cases before UI wiring:

1. `0 Hz tremor + 1 Hz gross motion → No tremor detected`
2. `5 Hz tremor + no gross motion → Likely rhythmic tremor`
3. `5 Hz tremor + 1 Hz gross motion → Likely rhythmic tremor if quality passes`
4. `random jitter/noise → No reliable tremor signal` or `No tremor detected`
5. `low FPS / dropped frames → Unusable recording`
6. `5 Hz global/camera jitter equally on hand/background → not Likely rhythmic tremor`

Run validation with:

```zsh
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
PYTHONPATH=/Users/aharshi/MotionBloomAppVersion/motionbloomtremor /Users/aharshi/MotionBloomAppVersion/motionbloomtremor/venv/bin/python -m unittest tests.test_roi_tremor_synthetic -v
```

## Integration Constraint

Do not wire this subsystem into `motionbloom/app.py` until the synthetic tests pass and diagnostic metrics print for all cases. Existing MediaPipe-based exercise/gross-movement scoring can remain separate from ROI optical-flow tremor confidence.
