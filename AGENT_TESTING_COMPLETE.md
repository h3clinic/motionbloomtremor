# Agent Testing Infrastructure — Complete

Created a complete agent testing contract to force VS Code AI agents to run tests themselves instead of asking humans to validate.

## Files Created

### Core Testing
- ✅ `scripts/smoke_video_audio.py` (2.4KB) — Automated video audio smoke test
- ✅ `.vscode/tasks.json` (968B) — VS Code tasks for compile/test/launch
- ✅ `scripts/README.md` — Full smoke test documentation
- ✅ `scripts/list_tasks.py` (739B) — Helper to list available tasks
- ✅ `TESTING.md` (1.6KB) — Quick reference for testing

### Agent Contract
- ✅ `AGENT_TESTING_RULES.md` (2.2KB) — **Agent testing contract**
- ✅ `.github/AGENT_PROMPT.md` (2.9KB) — **Copy-paste prompts for agent setup**

## How to Use

### 1. Configure the Agent (One Time)

Open VS Code Chat panel and paste:

```text
Read AGENT_TESTING_RULES.md and follow it permanently for this repo.
You are responsible for running the smoke test yourself after every video-related change.
```

### 2. Agent Workflow (Every Change)

The agent must:

```
edit code → run smoke test → inspect output → fix if needed → rerun → report complete
```

**Not allowed:** "I updated the code. Please test it."

**Required:** "I updated the code. Running smoke test... [output]. Test passed."

### 3. Validation Criteria

Agent may only say "complete" when smoke test shows:

```text
VIDEO_AUDIO_STATE ... backend=VLC muted=0 volume=100 has_audio_stream=True state=State.Playing
OK: Video audio smoke test passed
```

## Current Status

✅ **Smoke test passes**

```text
VIDEO_AUDIO_STATE {'backend': 'VLC', 'audio_supported': True, 
  'path': '/Users/aharshi/Downloads/Adafruit Plotter.mp4', 
  'duration_ms': 27029, 'time_ms': 4151, 'muted': 0, 'volume': 100, 
  'audio_track': 1, 'audio_track_count': 2, 'has_audio_stream': True, 
  'state': 'State.Playing'}
OK: Video audio smoke test passed
```

## Philosophy

**AI can edit, but tests must judge.**

The agent is now a CI worker that:
- Runs tests automatically
- Reads failures
- Fixes issues
- Reruns until passing
- Only reports success when tests confirm it

No more "vibes-based validation" — executable tests judge correctness.

## Quick Commands

```bash
# Run smoke test
cd /Users/aharshi/MotionBloomAppVersion/motionbloomtremor
./venv/bin/python scripts/smoke_video_audio.py "/Users/aharshi/Downloads/Adafruit Plotter.mp4"

# Compile check
./venv/bin/python -m py_compile motionbloom/app.py motionbloom/video_player.py

# List tasks
./venv/bin/python scripts/list_tasks.py

# VS Code task
Cmd+Shift+P → Tasks: Run Task → MotionBloom: Video Audio Smoke Test
```

## Next Steps

1. Open VS Code Chat panel (right side)
2. Paste the agent configuration from `.github/AGENT_PROMPT.md`
3. Ask agent to run smoke test immediately
4. Verify agent follows the test-first workflow

The bottleneck is no longer coding — it's now deterministic validation enforced by contract.
