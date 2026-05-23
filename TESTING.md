# Testing Quick Reference

## Automated Smoke Tests

Run deterministic tests instead of manual clicking.

### Run via VS Code Tasks

```text
Cmd+Shift+P (macOS) or Ctrl+Shift+P (Windows/Linux)
→ Tasks: Run Task
→ MotionBloom: Video Audio Smoke Test
```

### Run via Terminal

```bash
# Video audio smoke test
./venv/bin/python scripts/smoke_video_audio.py "/path/to/video.mp4"

# Compile check
./venv/bin/python -m py_compile motionbloom/app.py motionbloom/video_player.py

# Launch app
PYTHONPATH=$PWD ./venv/bin/python -m motionbloom
```

## Expected Results

Smoke test should output:

```text
VIDEO_AUDIO_STATE ... muted=0 volume=100 audio_track_count=2 has_audio_stream=True state=State.Playing
OK: Video audio smoke test passed
```

## Available Tasks

Run `./venv/bin/python scripts/list_tasks.py` to see all VS Code tasks.

## Philosophy

**AI can edit, but tests must judge.**

Every video-player change should:
1. ✅ Pass compilation (`py_compile`)
2. ✅ Pass smoke test (upload/load/play/audio validation)
3. ✅ Show correct diagnostic state

## For AI Agents

**The agent must run tests, not ask humans to validate.**

See `AGENT_TESTING_RULES.md` for the complete agent test contract.

Quick setup for VS Code agent:

```text
Read AGENT_TESTING_RULES.md and follow it permanently for this repo.
You are responsible for running the smoke test yourself after every video-related change.
```

Copy-paste prompts from `.github/AGENT_PROMPT.md`.

## Documentation

- `scripts/README.md` — Full smoke test documentation
- `AGENT_TESTING_RULES.md` — Agent testing contract
- `.github/AGENT_PROMPT.md` — Copy-paste prompts for agent setup
