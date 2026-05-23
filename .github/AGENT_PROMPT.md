# Agent Prompt for MotionBloom Testing

Copy-paste this into the VS Code Chat panel to establish the testing contract:

---

## Initial Prompt

```text
You are responsible for testing MotionBloom after every code change.

Goal:
Stop making me manually validate video upload, VLC playback, audio state, and crash behavior.

Rules:
1. Run the MotionBloom smoke test yourself after every video/player/app change.
2. Do not ask me to manually upload the video unless absolutely impossible.
3. Use the existing VS Code task if available:
   MotionBloom: Video Audio Smoke Test
4. If the VS Code task is inconvenient, run the terminal command directly from the repo root:
   cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
   ./venv/bin/python scripts/smoke_video_audio.py "/Users/aharshi/Downloads/Adafruit Plotter.mp4"
5. If that MP4 path does not exist, search Downloads for an .mp4 file and use the first valid one.
6. After running the test, inspect the output.
7. If the test fails, fix the code and rerun.
8. Repeat until the test passes.
9. Do not claim success unless the output confirms:
   - compile passes
   - upload/load does not crash
   - backend=VLC
   - muted=False or muted=0
   - volume > 0
   - has_audio_stream=True
   - state=Playing or equivalent
   - app closes cleanly
10. Record the final result and changed files.

Important:
Diagnostics must never crash the app. debug_state(), has_audio_stream(), volume/mute methods, and VLC track calls must be safe. If VLC returns unknown values, log "unknown" instead of crashing.
```

---

## Follow-Up Command

After setting the rules, immediately test:

```text
Before changing anything else, run the smoke test now and show me the exact terminal output. If it fails, fix only the failing part and rerun.
```

---

## Permanent Agent Configuration

For ongoing work:

```text
Read AGENT_TESTING_RULES.md and follow it permanently for this repo. You are responsible for running the smoke test yourself after every video-related change.
```

---

## Expected Agent Behavior

**Before:** "I updated the code. Please test it manually."

**After:** "I updated the code. Running smoke test... [output]. Test passed. Changes complete."

**On Failure:** "Test failed with [error]. Fixing [specific issue]. Rerunning... [output]. Test passed."

---

## Test Commands for Agent

### Full smoke test
```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python scripts/smoke_video_audio.py "/Users/aharshi/Downloads/Adafruit Plotter.mp4"
```

### Compile only
```bash
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python -m py_compile motionbloom/app.py motionbloom/video_player.py
```

### Find MP4 in Downloads
```bash
find /Users/aharshi/Downloads -name "*.mp4" -type f | head -1
```

---

## Success Criteria

Agent may only report "complete" when it shows:

```text
OK: Video audio smoke test passed
```

Not before.
