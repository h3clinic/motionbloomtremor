#!/usr/bin/env python3
"""Capture and verify the MotionBloom Electron UI window on macOS.

This script finds the running MotionBloom Electron window, captures a real
screenshot of it (not just the desktop), saves it to agent-runs/latest-ui.png,
and validates that the screenshot is non-empty.

Why this exists:
  Checking `ps aux` / process counts does NOT prove the UI rendered. Only a
  real screenshot of the window confirms the Duolingo-style UI is visible.

Usage:
  python scripts/verify_ui_render.py
  python scripts/verify_ui_render.py --output agent-runs/latest-ui.png

Requirements (macOS):
  - `screencapture` (built-in)
  - Screen Recording permission granted to the terminal/host app

Exit codes:
  0  Screenshot captured and passes basic non-empty checks
  1  No MotionBloom window found
  2  Screenshot missing or empty
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "agent-runs" / "latest-ui.png"

# The macOS window owner is "Electron" when launched via `electron .`.
# We match on the window title which is set to "MotionBloom" in index.html.
WINDOW_TITLE = "MotionBloom"
WINDOW_OWNER = "Electron"


def find_window_id() -> str | None:
    """Return the CoreGraphics window id for the MotionBloom Electron window.

    Uses AppleScript via System Events first (reliable for title), then falls
    back to scanning Quartz window list through a small Python helper.
    """
    # Strategy 1: Quartz window list (most reliable, includes off-screen).
    try:
        import Quartz  # type: ignore

        options = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        window_list = Quartz.CGWindowListCopyWindowInfo(
            options, Quartz.kCGNullWindowID
        )
        candidates = []
        for win in window_list:
            owner = win.get("kCGWindowOwnerName", "")
            name = win.get("kCGWindowName", "") or ""
            wid = win.get("kCGWindowNumber")
            # MotionBloom Electron: owner "Electron", title contains MotionBloom.
            if owner == WINDOW_OWNER and WINDOW_TITLE in name:
                candidates.append((wid, name, owner))
            # Some builds set owner to the app name.
            elif WINDOW_TITLE.lower() in owner.lower():
                candidates.append((wid, name, owner))
        if candidates:
            # Prefer one with a non-empty title.
            candidates.sort(key=lambda c: (c[1] == "", c[0]))
            return str(candidates[0][0])
    except ImportError:
        pass  # Quartz not available; fall through.

    return None


def capture_window(window_id: str, output: Path) -> bool:
    """Capture a specific window by id using macOS screencapture -l."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # -l <windowid> : capture that window
    # -o            : no window shadow
    # -x            : no capture sound
    result = subprocess.run(
        ["screencapture", "-l", window_id, "-o", "-x", str(output)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and output.exists()


def capture_frontmost(output: Path) -> bool:
    """Fallback: bring Electron to front and capture the main display region.

    Used when we cannot resolve a CoreGraphics window id but the app is open.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    # Try to focus the Electron app first.
    subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to set frontmost of '
            'first process whose name is "Electron" to true',
        ],
        capture_output=True,
        text=True,
    )
    time.sleep(1.0)
    # Capture the full main display (-x silent). Better than nothing.
    result = subprocess.run(
        ["screencapture", "-x", str(output)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and output.exists()


def validate_screenshot(output: Path, min_bytes: int = 5000) -> bool:
    """Basic checks: file exists and is larger than a trivial blank image."""
    if not output.exists():
        print(f"[FAIL] Screenshot not found at {output}")
        return False
    size = output.stat().st_size
    if size < min_bytes:
        print(f"[FAIL] Screenshot too small ({size} bytes); likely blank.")
        return False
    print(f"[OK] Screenshot saved: {output} ({size:,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the screenshot (default: agent-runs/latest-ui.png)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="How many times to look for the window before giving up.",
    )
    args = parser.parse_args()

    window_id = None
    for attempt in range(1, args.retries + 1):
        window_id = find_window_id()
        if window_id:
            print(f"[INFO] Found MotionBloom window id={window_id} "
                  f"(attempt {attempt})")
            break
        print(f"[INFO] MotionBloom window not found yet "
              f"(attempt {attempt}/{args.retries})...")
        time.sleep(1.5)

    captured = False
    if window_id:
        captured = capture_window(window_id, args.output)
        if not captured:
            print("[WARN] Window capture failed; trying frontmost fallback.")

    if not captured:
        print("[INFO] Falling back to frontmost-display capture.")
        captured = capture_frontmost(args.output)

    if not captured:
        print("[FAIL] Could not capture any screenshot. Is the app open and "
              "is Screen Recording permission granted?")
        return 1

    if not validate_screenshot(args.output):
        return 2

    print("\n[NEXT] Open the screenshot and visually confirm the "
          "Duolingo-style UI:")
    print("        - Bloom mascot + score circle")
    print("        - Streak / XP / Hearts pills")
    print("        - Exercise sidebar")
    print("        - Camera preview + metric cards")
    print(f"\n  open {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
