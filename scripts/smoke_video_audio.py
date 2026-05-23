#!/usr/bin/env python3
"""Smoke test for video audio playback.

Usage:
    python scripts/smoke_video_audio.py /path/to/video.mp4

This script validates:
- VLC backend is selected
- Video loads without crashing
- Audio is not muted
- Volume is set correctly
- Audio stream is detected
- debug_state() returns valid data
"""

import sys
import time
from pathlib import Path
from tkinter import Tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionbloom.app import App


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> None:
    video_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if not video_arg:
        fail("Pass a known MP4 with audio: python scripts/smoke_video_audio.py /path/to/video.mp4")

    video = Path(video_arg).expanduser().resolve()
    if not video.exists():
        fail(f"Video not found: {video}")

    root = Tk()
    root.geometry("900x650+100+100")

    app = App(root)
    app._stop_startup_intro()
    
    # Stop camera to avoid threading conflicts
    if hasattr(app.tracker, 'thread') and app.tracker.thread is not None:
        app.tracker.stop()
        time.sleep(0.2)

    root.update()
    app._set_video_file(str(video))

    for _ in range(20):
        root.update()
        time.sleep(0.1)

    player = getattr(app, "video_player", None)
    if player is None:
        fail("video_player was not created")

    backend = getattr(player, "backend_name", "unknown")
    if backend != "VLC":
        fail(f"Expected VLC backend, got {backend}")

    if hasattr(player, "debug_state"):
        state = player.debug_state()
        print("VIDEO_AUDIO_STATE", state)

        muted = state.get("muted")
        volume = state.get("volume")
        audio_track_count = state.get("audio_track_count")
        has_audio_stream = state.get("has_audio_stream")

        if muted not in (False, 0):
            fail(f"Player is muted: {muted}")

        if isinstance(volume, int) and volume <= 0:
            fail(f"Volume is zero: {volume}")

        if has_audio_stream is False:
            fail(f"No audio stream detected. audio_track_count={audio_track_count}")

    else:
        fail("Player has no debug_state() method")

    duration = player.get_duration_ms()
    if isinstance(duration, int) and duration <= 0:
        fail(f"Invalid duration: {duration}")

    app._pause_local_video()
    app._stop_local_video(clear_frame=True)
    root.destroy()

    ok("Video audio smoke test passed")


if __name__ == "__main__":
    main()
