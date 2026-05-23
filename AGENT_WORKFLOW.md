# Agent Testing Workflow

## Before (Manual Validation)

```
┌─────────────┐
│ Agent Edits │
│    Code     │
└──────┬──────┘
       │
       v
┌─────────────┐
│   Agent:    │
│ "I updated  │
│  the code.  │
│ Please test │
│     it."    │
└──────┬──────┘
       │
       v
┌─────────────┐
│    Human    │
│  manually   │
│   uploads   │
│   video     │
└──────┬──────┘
       │
       v
┌─────────────┐
│    Human    │
│  inspects   │
│   output    │
└──────┬──────┘
       │
       v
┌─────────────┐
│   Human     │
│   reports   │
│   result    │
└─────────────┘

PROBLEM: Slow, error-prone, not repeatable
```

## After (Automated Agent Testing)

```
┌─────────────┐
│ Agent Edits │
│    Code     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────┐
│      Agent Runs Test:           │
│  ./venv/bin/python scripts/     │
│  smoke_video_audio.py           │
│  "/path/to/video.mp4"           │
└──────┬──────────────────────────┘
       │
       v
┌──────────────┐
│ Test Output  │
└──────┬───────┘
       │
       ├───── PASS ─────┐
       │                │
       │                v
       │         ┌──────────────┐
       │         │    Agent:    │
       │         │ "Test passed │
       │         │  Changes     │
       │         │  complete."  │
       │         └──────────────┘
       │
       └───── FAIL ─────┐
                        │
                        v
                 ┌──────────────┐
                 │ Agent Reads  │
                 │  Traceback   │
                 └──────┬───────┘
                        │
                        v
                 ┌──────────────┐
                 │ Agent Fixes  │
                 │  Specific    │
                 │   Issue      │
                 └──────┬───────┘
                        │
                        v
                 ┌──────────────┐
                 │ Agent Reruns │
                 │     Test     │
                 └──────┬───────┘
                        │
                        v
                   (Loop until pass)

SOLUTION: Fast, repeatable, deterministic
```

## Test Contract

### Agent Must

- ✅ Run smoke test automatically after every change
- ✅ Read test output and failures
- ✅ Fix failing code
- ✅ Rerun until passing
- ✅ Show passing output before reporting "complete"

### Agent Must Not

- ❌ Ask human to manually upload video
- ❌ Ask human to verify results
- ❌ Report "complete" without showing passing test
- ❌ Say "I think this should work"
- ❌ Skip test reruns after fixes

## Success Criteria

Agent may only report complete when it shows:

```text
VIDEO_AUDIO_STATE ... backend=VLC muted=0 volume=100 
  audio_track_count=2 has_audio_stream=True state=State.Playing
OK: Video audio smoke test passed
```

## Configuration

Paste this into VS Code Chat:

```text
Read AGENT_TESTING_RULES.md and follow it permanently for this repo.
You are responsible for running the smoke test yourself after every 
video-related change.
```

## Philosophy

**"AI can edit, but tests must judge."**

The agent is a CI worker, not just a code suggester.
