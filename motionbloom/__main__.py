"""Entry point for running motionbloom as a module (python -m motionbloom).

Supports both legacy Tkinter UI and new PyQt6 Duolingo-style UI.
Use MOTIONBLOOM_PYQT6=1 environment variable to launch PyQt6 UI.
"""

import os
from tkinter import Tk


def main():
    """Launch MotionBloom with selected UI framework."""
    # Check environment flag for UI choice
    use_pyqt6 = os.getenv("MOTIONBLOOM_PYQT6", "0") == "1"
    
    if use_pyqt6:
        print("[STARTUP] Launching PyQt6 UI with tremor backend integration...")
        try:
            from motionbloom.ui.pyqt_integrated_app import launch_pyqt6_app
            launch_pyqt6_app()
        except ImportError as e:
            print(f"ERROR: PyQt6 not available ({e}). Falling back to Tkinter UI.")
            from motionbloom.app import App
            root = Tk()
            app = App(root)
            root.mainloop()
    else:
        print("[STARTUP] Launching legacy Tkinter UI...")
        from motionbloom.app import App
        root = Tk()
        app = App(root)
        root.mainloop()


if __name__ == "__main__":
    main()
