#!/usr/bin/env python3
"""
Record a MotionBloom UI improvement iteration.

Standard-library only.
Creates:
- agent-runs/<timestamp>/iteration_manifest.json
- agent-runs/<timestamp>/ui_score.json
- agent-runs/<timestamp>/notes.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


RUBRIC_KEYS = [
    "layout_hierarchy",
    "spacing_consistency",
    "typography_clarity",
    "color_intentionality",
    "component_consistency",
    "responsive_behavior",
    "accessibility",
    "visual_distinctiveness",
    "product_clarity",
]


def run_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {
            "command": " ".join(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"Command not found: {command[0]}",
        }


def changed_files_from_status(status_text: str) -> list[str]:
    files = []
    for line in status_text.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a UI improvement iteration.")
    parser.add_argument("--target-file", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--app-command", default=None)
    parser.add_argument("--screenshot-path", default=None)
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / "agent-runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    git_status = run_command(["git", "status", "--short"])
    git_diff_stat = run_command(["git", "diff", "--stat"])
    changed_files = changed_files_from_status(git_status["stdout"])

    screenshot_path = args.screenshot_path if args.screenshot_path else None

    manifest = {
        "timestamp": timestamp,
        "run_dir": str(run_dir.relative_to(repo_root)),
        "target_file": args.target_file,
        "app_command": args.app_command,
        "screenshot_path": screenshot_path,
        "notes": args.notes,
        "changed_files": changed_files,
        "git_status": git_status,
        "git_diff_stat": git_diff_stat,
    }

    ui_score = {key: None for key in RUBRIC_KEYS}
    ui_score["overall_score"] = None
    ui_score["pass_threshold"] = 8.5

    notes_md = f"""# UI Iteration Notes

Timestamp: {timestamp}

Target file: {args.target_file or "Not specified"}

App command: {args.app_command or "Not specified"}

Screenshot path: {screenshot_path or "Not provided"}

## Notes

{args.notes or "No notes provided."}

## Changed Files

{chr(10).join(f"- {file}" for file in changed_files) if changed_files else "No changed files detected by git status."}

## Manual Scoring Instructions

Open `ui_score.json` and fill in each category from 1–10.

Pass threshold: 8.5+

If the score is below 8.5, run another UI improvement iteration.
"""

    (run_dir / "iteration_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (run_dir / "ui_score.json").write_text(
        json.dumps(ui_score, indent=2),
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text(notes_md, encoding="utf-8")

    print(f"Created UI iteration run: {run_dir}")
    print("Created:")
    print(f"- {run_dir / 'iteration_manifest.json'}")
    print(f"- {run_dir / 'ui_score.json'}")
    print(f"- {run_dir / 'notes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
