#!/usr/bin/env python3
"""
Record a UI screenshot artifact for a MotionBloom UI iteration.

Standard-library only.
This does NOT fake automatic screenshots.
If --screenshot-path is provided and exists, it copies that file into the run folder.
Otherwise, screenshot_path is null and the manifest records that capture is manual.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a manual UI screenshot artifact.")
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--screenshot-path", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--app-command", default=None)
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / "agent-runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    copied_screenshot = None
    screenshot_note = "Screenshot capture is manual. No screenshot was provided."

    if args.screenshot_path:
        source = Path(args.screenshot_path).expanduser()
        if source.exists() and source.is_file():
            destination = run_dir / source.name
            shutil.copy2(source, destination)
            copied_screenshot = str(destination.relative_to(repo_root))
            screenshot_note = "Screenshot was copied into the run folder."
        else:
            screenshot_note = f"Provided screenshot path does not exist: {args.screenshot_path}"

    notes = args.notes
    if copied_screenshot is None:
        notes = (
            f"{notes}\n\nScreenshot capture is manual. Provide --screenshot-path with an existing file to attach one."
            if notes
            else "Screenshot capture is manual. Provide --screenshot-path with an existing file to attach one."
        )

    manifest = {
        "timestamp": timestamp,
        "run_dir": str(run_dir.relative_to(repo_root)),
        "target_file": args.target_file,
        "app_command": args.app_command,
        "screenshot_path": copied_screenshot,
        "notes": notes,
        "screenshot_note": screenshot_note,
    }

    (run_dir / "iteration_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Created screenshot run: {run_dir}")
    print("Created:")
    print(f"- {run_dir / 'iteration_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())