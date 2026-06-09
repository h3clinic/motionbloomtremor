"""MotionBloom asset loading and management.

Handles loading PNG assets for Duolingo-style UI elements (mascot, icons).
Gracefully falls back to None if assets missing or in headless context.
"""

from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk


# Asset directory
DUOLINGO_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "duolingo"


def load_asset_image(
    name: str, size: Optional[tuple[int, int]] = None
) -> Optional[ImageTk.PhotoImage]:
    """Load asset image from duolingo assets directory.

    Args:
        name: Asset name (e.g. 'bloom_idle', 'streak_flame', 'xp_gem')
        size: Optional (width, height) tuple to resize image

    Returns:
        ImageTk.PhotoImage if found and Tk initialized, else None (placeholder will be used)
    """
    asset_path = DUOLINGO_ASSETS_DIR / f"{name}.png"

    if not asset_path.exists():
        return None

    try:
        img = Image.open(asset_path)
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        # ImageTk.PhotoImage requires a Tk root; returns None if in headless context
        try:
            return ImageTk.PhotoImage(img)
        except Exception:
            # Headless context or Tk not initialized yet
            return None
    except Exception:
        return None
