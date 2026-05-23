# Agent Testing Rules for MotionBloom

**The agent must test MotionBloom directly. The user should not be required to manually validate routine changes.**

## Required Test Command

From repo root:

```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python scripts/smoke_video_audio.py "/Users/aharshi/Downloads/Adafruit Plotter.mp4"
```

If that file does not exist, search for a known `.mp4` in `/Users/aharshi/Downloads`.

## Required Validation

**Do not mark a video-player fix complete unless the smoke test confirms:**

- ✅ Compile passes
- ✅ Upload/load does not crash
- ✅ `backend=VLC`
- ✅ `muted=False` or `muted=0`
- ✅ `volume > 0`
- ✅ `has_audio_stream=True`
- ✅ `state=Playing` or equivalent
- ✅ App closes cleanly

## Failure Policy

If the smoke test fails:

1. Read the traceback
2. Identify the exact failing method/file
3. Patch **only** the failing area
4. Rerun the smoke test
5. Repeat until passing

**Do not move to a new task until the current test passes.**

## VLC Diagnostic Safety

These methods **must never crash**:

- `debug_state()`
- `has_audio_stream()`
- `get_volume()`
- `set_volume()`
- `is_muted()`
- `set_muted()`
- `get_duration_ms()`
- `get_time_ms()`
- `get_position_ratio()`

All VLC calls must be wrapped safely. Unknown VLC state should be logged as `"unknown"`, not allowed to crash upload/playback.

## Completion Standard

The agent may only say "complete" after showing the final passing smoke-test output.

Expected output:

```text
OK: Video audio smoke test passed
```

## Agent Workflow

```text
edit code
  ↓
run smoke test
  ↓
inspect output
  ↓
test passed? → report complete
  ↓ no
fix issue
  ↓
rerun test
  ↓
(repeat until passing)
```

## Example Test Execution

```bash
# Agent should run this automatically after every video-player change
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python scripts/smoke_video_audio.py "/Users/aharshi/Downloads/Adafruit Plotter.mp4" 2>&1
```

## Philosophy

**AI can edit, but tests must judge.**

The agent is a CI worker, not just a code suggester. It must:

- Run tests itself
- Read failures
- Fix issues
- Rerun
- Only report success when tests pass

**No "I think this should work" — show passing test output.**
