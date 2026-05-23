#!/usr/bin/env python3
"""List all available VS Code tasks for this workspace."""

import json
from pathlib import Path

tasks_file = Path(__file__).parent.parent / ".vscode" / "tasks.json"

if not tasks_file.exists():
    print("No .vscode/tasks.json found")
    raise SystemExit(1)

with open(tasks_file) as f:
    data = json.load(f)

print("Available VS Code Tasks:\n")

for task in data.get("tasks", []):
    label = task.get("label", "Unknown")
    command = task.get("command", "")
    depends = task.get("dependsOn", [])
    
    print(f"📋 {label}")
    print(f"   Command: {command}")
    if depends:
        print(f"   Depends on: {', '.join(depends)}")
    print()

print("\nRun tasks with: Cmd+Shift+P → Tasks: Run Task")
