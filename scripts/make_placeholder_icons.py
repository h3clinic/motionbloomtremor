#!/usr/bin/env python3
"""Draw clean red/white vector-style placeholder icons with Pillow.

These are NOT emojis and NOT blank squares -- they are simple, on-brand shapes
(flame, gem, heart, hand, camera, chart, flower logo, mascot blob) so the UI
looks polished immediately. Run `generate_gemini_assets.py` with an API key to
replace any of these with fully AI-generated artwork.

Usage:
    python scripts/make_placeholder_icons.py            # fill only missing
    python scripts/make_placeholder_icons.py --force    # redraw everything
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "motionbloom" / "assets" / "duolingo"
LOGO = REPO_ROOT / "motionbloom" / "assets" / "motionbloom_logo.png"

RED = (230, 57, 70, 255)
RED_SOFT = (255, 107, 119, 255)
WHITE = (255, 255, 255, 255)


def canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def rrect(d, box, r, fill, outline=None, width=0):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_flame(s=128):
    img, d = canvas(s)
    cx = s / 2
    pts = [
        (cx, s * 0.10), (s * 0.74, s * 0.42), (s * 0.80, s * 0.66),
        (cx, s * 0.92), (s * 0.20, s * 0.66), (s * 0.26, s * 0.42),
    ]
    d.polygon(pts, fill=RED)
    d.ellipse([s * 0.42, s * 0.52, s * 0.58, s * 0.78], fill=WHITE)
    return img


def draw_gem(s=128):
    img, d = canvas(s)
    top, bot = s * 0.22, s * 0.86
    d.polygon([(s*0.5, top), (s*0.84, s*0.42), (s*0.5, bot),
               (s*0.16, s*0.42)], fill=RED)
    d.line([(s*0.34, s*0.42), (s*0.66, s*0.42)], fill=WHITE, width=max(2, s//40))
    d.line([(s*0.5, top), (s*0.5, bot)], fill=(255, 255, 255, 120),
           width=max(1, s//64))
    return img


def draw_heart(s=128):
    img, d = canvas(s)
    r = s * 0.22
    d.ellipse([s*0.20, s*0.22, s*0.20+2*r, s*0.22+2*r], fill=RED)
    d.ellipse([s*0.58-2*r+2*r, 0, 0, 0]) if False else None
    d.ellipse([s*0.56, s*0.22, s*0.56+2*r, s*0.22+2*r], fill=RED)
    d.polygon([(s*0.16, s*0.46), (s*0.84, s*0.46), (s*0.5, s*0.86)], fill=RED)
    d.ellipse([s*0.30, s*0.34, s*0.40, s*0.44], fill=(255, 255, 255, 120))
    return img


def draw_hand(s=128, wave=False):
    img, d = canvas(s)
    palm = [s*0.32, s*0.42, s*0.70, s*0.86]
    rrect(d, palm, r=s*0.14, fill=RED)
    fw = s * 0.085
    base_x = s * 0.34
    for i in range(4):
        x = base_x + i * (fw + s * 0.018)
        top = s * (0.20 if i in (1, 2) else 0.26)
        rrect(d, [x, top, x + fw, s*0.50], r=fw/2, fill=RED)
    # thumb
    rrect(d, [s*0.20, s*0.46, s*0.34, s*0.70], r=s*0.07, fill=RED)
    if wave:
        d.arc([s*0.66, s*0.10, s*0.92, s*0.40], 200, 340, fill=RED,
              width=max(3, s//26))
    return img


def draw_nose(s=128):
    img, d = canvas(s)
    # simple face profile + fingertip
    d.arc([s*0.20, s*0.18, s*0.78, s*0.82], 300, 120, fill=RED,
          width=max(4, s//22))
    d.polygon([(s*0.30, s*0.50), (s*0.20, s*0.60), (s*0.32, s*0.62)], fill=RED)
    d.ellipse([s*0.10, s*0.52, s*0.24, s*0.66], fill=RED_SOFT)  # fingertip
    return img


def draw_head(s=128):
    img, d = canvas(s)
    d.ellipse([s*0.30, s*0.34, s*0.70, s*0.78], fill=RED)        # head
    rrect(d, [s*0.30, s*0.18, s*0.70, s*0.34], r=s*0.08, fill=RED_SOFT)  # hand
    for i in range(3):
        x = s*0.36 + i*s*0.11
        rrect(d, [x, s*0.10, x+s*0.06, s*0.22], r=s*0.03, fill=RED_SOFT)
    return img


def draw_hold(s=128):
    img, d = canvas(s)
    rrect(d, [s*0.40, s*0.30, s*0.66, s*0.62], r=s*0.05, fill=RED)  # cup
    d.arc([s*0.60, s*0.34, s*0.78, s*0.56], 300, 60, fill=RED,
          width=max(3, s//26))  # handle
    rrect(d, [s*0.30, s*0.60, s*0.74, s*0.74], r=s*0.06, fill=RED_SOFT)  # hand
    return img


def draw_camera(s=128):
    img, d = canvas(s)
    rrect(d, [s*0.16, s*0.34, s*0.84, s*0.76], r=s*0.10, fill=RED)
    rrect(d, [s*0.36, s*0.26, s*0.56, s*0.36], r=s*0.03, fill=RED)
    d.ellipse([s*0.40, s*0.42, s*0.60, s*0.62], fill=WHITE)
    d.ellipse([s*0.45, s*0.47, s*0.55, s*0.57], fill=RED)
    return img


def draw_reports(s=128):
    img, d = canvas(s)
    base = s * 0.78
    heights = [0.30, 0.48, 0.22, 0.40]
    for i, h in enumerate(heights):
        x = s*0.22 + i*s*0.16
        rrect(d, [x, base - s*h, x+s*0.10, base], r=s*0.02, fill=RED)
    d.line([(s*0.16, base+ s*0.02), (s*0.84, base + s*0.02)], fill=RED,
           width=max(2, s//40))
    return img


def draw_flower_logo(s=512):
    img, d = canvas(s)
    cx = cy = s / 2
    pr = s * 0.20
    for k in range(6):
        ang = math.radians(60 * k)
        px = cx + math.cos(ang) * s * 0.22
        py = cy + math.sin(ang) * s * 0.22
        d.ellipse([px-pr, py-pr*0.7, px+pr, py+pr*0.7],
                  fill=RED_SOFT if k % 2 else RED)
    d.ellipse([cx-s*0.12, cy-s*0.12, cx+s*0.12, cy+s*0.12], fill=WHITE)
    d.ellipse([cx-s*0.07, cy-s*0.07, cx+s*0.07, cy+s*0.07], fill=RED)
    return img


def draw_mascot(s=256, mood="idle"):
    img, d = canvas(s)
    # rounded blob body
    rrect(d, [s*0.24, s*0.22, s*0.76, s*0.84], r=s*0.26, fill=RED)
    # eyes
    ey = s*0.42
    for ex in (s*0.40, s*0.60):
        d.ellipse([ex-s*0.06, ey-s*0.06, ex+s*0.06, ey+s*0.06], fill=WHITE)
        d.ellipse([ex-s*0.025, ey-s*0.025, ex+s*0.025, ey+s*0.025], fill=RED)
    # mouth by mood
    if mood == "sad":
        d.arc([s*0.40, s*0.60, s*0.60, s*0.76], 200, 340, fill=WHITE,
              width=max(3, s//40))
    elif mood == "cheer":
        d.chord([s*0.38, s*0.54, s*0.62, s*0.74], 0, 180, fill=WHITE)
    else:
        d.arc([s*0.40, s*0.56, s*0.60, s*0.72], 20, 160, fill=WHITE,
              width=max(3, s//40))
    if mood == "wave":
        rrect(d, [s*0.74, s*0.30, s*0.86, s*0.52], r=s*0.05, fill=RED_SOFT)
    return img


JOBS = {
    "streak_flame": (draw_flame, 128),
    "xp_gem": (draw_gem, 128),
    "heart": (draw_heart, 128),
    "ex_free": (lambda s: draw_hand(s, wave=True), 128),
    "ex_nose": (draw_nose, 128),
    "ex_head": (draw_head, 128),
    "ex_hold": (draw_hold, 128),
    "ui_camera": (draw_camera, 128),
    "ui_reports": (draw_reports, 128),
    "bloom_idle": (lambda s: draw_mascot(s, "idle"), 256),
    "bloom_wave": (lambda s: draw_mascot(s, "wave"), 256),
    "bloom_cheer": (lambda s: draw_mascot(s, "cheer"), 256),
    "bloom_sad": (lambda s: draw_mascot(s, "sad"), 256),
    "bloom_analyzing": (lambda s: draw_mascot(s, "idle"), 256),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Redraw even if file exists")
    args = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)

    made = 0
    for name, (fn, size) in JOBS.items():
        out = ASSETS / f"{name}.png"
        # Always redraw the tiny 230-byte blank placeholders.
        is_blank = out.exists() and out.stat().st_size < 600
        if out.exists() and not args.force and not is_blank:
            continue
        fn(size).save(out, "PNG")
        print(f"drew {out.name}")
        made += 1

    if args.force or not LOGO.exists() or LOGO.stat().st_size < 4000:
        draw_flower_logo(512).save(LOGO, "PNG")
        print(f"drew {LOGO.name}")
        made += 1

    print(f"\nDone: {made} icon(s) drawn. Run generate_gemini_assets.py with an "
          f"API key to upgrade to AI art.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
