# MotionBloom Smoke Tests

Automated smoke tests for video audio playback validation.

## Quick Start

### Run the smoke test via VS Code Tasks

1. Open this workspace in VS Code
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
3. Type: `Tasks: Run Task`
4. Select: `MotionBloom: Video Audio Smoke Test`
5. Enter path to a known MP4 with audio (default: `/Users/aharshi/Downloads/Adafruit Plotter.mp4`)

### Run the smoke test via terminal

```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python scripts/smoke_video_audio.py "/path/to/video.mp4"
```

## What the smoke test validates

The smoke test (`scripts/smoke_video_audio.py`) checks:

- ✅ Video file exists
- ✅ App initializes without crashing
- ✅ Video loads without crashing
- ✅ VLC backend is selected (not OpenCV fallback)
- ✅ `video_player` instance is created
- ✅ `debug_state()` method exists and returns data
- ✅ Player is **not muted** (`muted=False` or `muted=0`)
- ✅ Volume is **greater than 0** (`volume=100` expected)
- ✅ Audio stream is detected (`has_audio_stream=True`)
- ✅ Duration is valid (not zero or negative)
- ✅ App closes cleanly

## Available VS Code Tasks

### 1. MotionBloom: Compile

Compiles `motionbloom/app.py` and `motionbloom/video_player.py` to catch syntax errors.

```bash
venv/bin/python -m py_compile motionbloom/app.py motionbloom/video_player.py
```

### 2. MotionBloom: Video Audio Smoke Test

Runs the full video audio smoke test. Depends on compilation passing first.

### 3. MotionBloom: Launch App

Launches the full MotionBloom app in the background.

```bash
PYTHONPATH=${workspaceFolder} venv/bin/python -m motionbloom
```

## Using with GitHub Copilot Agent Mode

After every video-player change, run:

```text
Tasks: Run Task → MotionBloom: Video Audio Smoke Test
```

**Do not mark the task complete unless:**

- ✅ Compile passes
- ✅ Upload/load does not crash
- ✅ `backend=VLC`
- ✅ `muted=False`
- ✅ `volume > 0`
- ✅ `has_audio_stream=True`
- ✅ App closes cleanly

## Expected Output

Successful smoke test output:

```text
[STARTUP] MotionBloom customer UI active
[STARTUP] Tremor pipeline, palm-relative tracking, and diagnostics enabled
VIDEO_AUDIO_STATE label=app_after_750ms backend=VLC audio_supported=True path=/Users/aharshi/Downloads/Adafruit_Plotter.mp4 duration_ms=27029 time_ms=2158 muted=0 volume=100 audio_track=1 audio_track_count=2 has_audio_stream=True state=State.Playing
VIDEO_AUDIO_STATE {'backend': 'VLC', 'audio_supported': True, 'path': '/Users/aharshi/Downloads/Adafruit Plotter.mp4', 'duration_ms': 27029, 'time_ms': 4905, 'muted': 0, 'volume': 100, 'audio_track': 1, 'audio_track_count': 2, 'has_audio_stream': True, 'state': 'State.Playing'}
OK: Video audio smoke test passed
```

## Troubleshooting

### Smoke test fails with "Video not found"

Make sure you're using a valid path to an MP4 file with audio. Update the default path in `.vscode/tasks.json`:

```json
"default": "/path/to/your/video.mp4"
```

### Smoke test fails with "No audio stream detected"

The video file may not have an audio track. Test with a known-good MP4:

```bash
ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of default=noprint_wrappers=1 /path/to/video.mp4
```

Expected output: `codec_type=audio`

### Smoke test fails with "Expected VLC backend, got OpenCV preview"

VLC is not properly installed or configured. Install VLC:

```bash
brew install --cask vlc
pip install python-vlc
```

## Philosophy

**AI can edit, but tests must judge.**

This smoke test provides deterministic validation instead of manual clicking. Every video-player change should pass this test before being considered complete.
