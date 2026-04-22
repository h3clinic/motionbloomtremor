# MotionBloom TremorLab

Real-time hand tremor detector. Webcam-only, runs 100% locally — no video leaves your machine.

![status](https://img.shields.io/badge/status-beta-orange) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.10--3.12-informational)

## Features
- Live hand tracking (MediaPipe) with tremor score 0–100
- FFT + Welch spectrum, dominant frequency, band-ratio (3–12 Hz), SNR, peak sharpness
- Voluntary-motion rejection — waving or raising your hand no longer inflates the score
- Guided exercise gates with random tasks, self-view camera mirror, object-grip detection
- Personal adaptive threshold: passes when you are near your own running average
- Built-in focus video player (seek, play/pause, mute, volume, skip ±10s)

## Quick start (source)

Requires Python **3.10–3.12** (MediaPipe wheels).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tremorlab_run.py
```

macOS may prompt for camera permission on first launch. On macOS you also need Tk: `brew install python-tk@3.12`.

## Downloads

Prebuilt binaries for macOS and Windows are attached to each [GitHub Release](../../releases). Download, unzip, and run — no Python install needed.

## Store distribution

See [packaging/STORES.md](packaging/STORES.md) for step-by-step Microsoft Store (MSIX) submission and Apple notarization setup.

## Privacy

All processing is local. See [PRIVACY.md](PRIVACY.md).

## Build your own binary

```bash
pip install pyinstaller
pyinstaller packaging/tremorlab.spec
# output in dist/TremorLab/
```

## How it works
1. MediaPipe Hands (21 landmarks) + Pose (arm geometry) at ~30 fps.
2. Fingertip trajectory → resample → detrend → Hann window.
3. Welch PSD over 1–15 Hz gives the dominant tremor frequency, band power, and peak sharpness.
4. Low-band (0.3–2.5 Hz) power is measured separately; when it dominates we classify as voluntary motion and suppress the score.
5. Adaptive personal baseline learns your running average and sets the pass threshold ≈ avg + 8.

## Disclaimer
Technical demo — **not a medical device**, not a diagnostic tool. Do not use for clinical decisions.

## License
MIT — see [LICENSE](LICENSE).
