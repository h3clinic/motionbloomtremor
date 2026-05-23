"""Entry point for running motionbloom as a module (python -m motionbloom)."""

from tkinter import Tk
from .app import App


def main():
    """Launch the MotionBloom tremor detection application."""
    root = Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
