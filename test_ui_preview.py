#!/usr/bin/env python3
"""Quick UI preview test - no camera access needed."""

import sys
from PyQt6.QtWidgets import QApplication
from motionbloom.ui.pyqt_integrated_app import MotionBloomMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MotionBloomMainWindow()
    window.show()
    sys.exit(app.exec())
