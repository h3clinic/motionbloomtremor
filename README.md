# MotionBloom MotionBloom

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
python motionbloom_run.py
```

macOS may prompt for camera permission on first launch. On macOS you also need Tk: `brew install python-tk@3.12`.

## Electron UI (Chromium switch)

MotionBloom now includes an Electron desktop UI path under `electron/` with a Python bridge (`motionbloom/electron_bridge.py`) that streams live metric events.

### Install Electron UI dependencies

```bash
cd electron
npm install
```

### Run Electron UI

```bash
cd electron
npm start
```

### Optional: choose Python interpreter for bridge

```bash
cd electron
MB_PYTHON=../venv/bin/python npm start
```

### Current migration scope

- Electron renderer provides a Chromium-based dashboard shell and local camera preview.
- Python bridge provides start/stop status + live tremor metric stream.
- Legacy Tkinter/PyQt paths remain available during transition.

### Verify the UI actually rendered (screenshot, not process checks)

Process counts (`ps aux`, `npm start` exit code) do NOT prove the UI rendered.
With the app running, capture and validate a real window screenshot:

```bash
# from repo root, with the app already running via `npm start`
venv/bin/python scripts/verify_ui_render.py
# saves agent-runs/latest-ui.png and checks it is non-empty
```

Renderer wiring, the exact file Electron loads, the startup command, and the
screenshot-verification workflow are documented in `agent-runs/NOTES.md`.

### Logos, mascot, and UI icons (AI image generation)

All UI art (logo, the Bloom mascot, streak/XP/heart badges, exercise icons,
camera/report icons) lives in `motionbloom/assets/duolingo/`. The UI uses real
PNG artwork — no emojis. Prompts live in
`motionbloom/assets/duolingo/assets_manifest.json`.

Generate polished AI artwork with Google's image models:

```bash
pip install google-genai pillow
export GEMINI_API_KEY=<your-key>          # or GOOGLE_API_KEY
python scripts/generate_gemini_assets.py            # fill missing
python scripts/generate_gemini_assets.py --force    # regenerate all
python scripts/generate_gemini_assets.py --only motionbloom_logo bloom_idle
```

The generator tries Imagen (`imagen-3.0-generate-002`) first, then falls back to
the Gemini image model (`gemini-2.5-flash-image-preview`), and normalizes every
result to the size declared in the manifest.

Until you add an API key, on-brand red/white vector placeholders are drawn
locally (no network, no emojis) so the UI is never blank or broken:

```bash
python scripts/make_placeholder_icons.py            # fill missing
python scripts/make_placeholder_icons.py --force    # redraw all
```



## UI Improvement Loop

This repository includes a small UI iteration logger using Python standard library only. Screenshot capture is manual unless you pass an existing screenshot file.

### Run the app

```bash
python3 motionbloom_run.py
```

Local venv option:

```bash
./venv/bin/python motionbloom_run.py
```

### Record a UI iteration

```bash
python3 scripts/run_ui_iteration.py \
	--target-file motionbloom/ui/pyqt_integrated_app.py \
	--app-command "python3 motionbloom_run.py" \
	--notes "Initial UI loop smoke test"
```

### Record a manual screenshot

```bash
python3 scripts/capture_ui.py \
	--target-file motionbloom/ui/pyqt_integrated_app.py \
	--screenshot-path /path/to/screenshot.png \
	--app-command "python3 motionbloom_run.py" \
	--notes "Manual screenshot after UI pass"
```

If `--screenshot-path` is omitted or invalid, `screenshot_path` is recorded as `null`.

### Score the UI

Open the generated `agent-runs/<timestamp>/ui_score.json` and fill all 9 rubric categories (1-10), then set `overall_score`.

Pass threshold:

```text
overall_score >= 8.5
```

## Downloads

Prebuilt binaries for macOS and Windows are attached to each [GitHub Release](../../releases). Download, unzip, and run — no Python install needed.

## Store distribution

See [packaging/STORES.md](packaging/STORES.md) for step-by-step Microsoft Store (MSIX) submission and Apple notarization setup.

## Privacy

All processing is local. See [PRIVACY.md](PRIVACY.md).

## Build your own binary

```bash
pip install pyinstaller
pyinstaller packaging/motionbloom.spec
# output in dist/MotionBloom/
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
