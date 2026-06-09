#!/usr/bin/env python3
"""
MotionBloom PyQt6 Launcher
Launches the new premium Duolingo-style UI
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment for PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Headless if no display

from motionbloom.ui.pyqt_app import launch_pyqt6_app

if __name__ == "__main__":
    launch_pyqt6_app()
