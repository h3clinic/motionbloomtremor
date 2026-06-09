#!/usr/bin/env python3
"""Render a 360 turntable of the original hand and build an HTML gallery."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = Path("input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb")
DEFAULT_OUT_DIR = Path("output/turntable_360")
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.environ.get("HAND_MODEL", str(DEFAULT_MODEL)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", DEFAULT_BLENDER))
    parser.add_argument("--skip-render", action="store_true", help="rebuild gallery HTML from existing frames only")
    return parser.parse_args()


def render_gallery_html(names, views):
    cards = "\n".join(
        f'<a class="card" href="{n}.png" target="_blank" rel="noreferrer">'
        f'<img src="{n}.png" alt="{n}" loading="lazy" /><span>{n}</span></a>'
        for n in names
    )
    frames_js = ", ".join(f'"{n}.png"' for n in names)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hand 360 Turntable</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: linear-gradient(180deg, #1d2027, #0d0e11); color: #f4f4f5; }}
    .wrap {{ max-width: 1280px; margin: 24px auto; padding: 0 16px 40px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(26px, 4vw, 46px); }}
    p {{ margin: 0 0 18px; color: #cbd5e1; }}
    .player {{ display: flex; flex-direction: column; align-items: center; gap: 12px; margin-bottom: 28px; }}
    .stage {{ width: min(520px, 90vw); aspect-ratio: 1/1; background: #000; border-radius: 18px; overflow: hidden; border: 1px solid rgba(255,255,255,.12); box-shadow: 0 18px 48px rgba(0,0,0,.4); }}
    .stage img, .stage video {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .controls {{ display: flex; align-items: center; gap: 14px; }}
    button {{ background: #2563eb; color: #fff; border: 0; border-radius: 10px; padding: 10px 16px; font-weight: 700; cursor: pointer; }}
    input[type=range] {{ width: min(420px, 80vw); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }}
    .card {{ display: block; border-radius: 12px; overflow: hidden; text-decoration: none; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); }}
    .card img {{ display: block; width: 100%; aspect-ratio: 1/1; object-fit: cover; background: #000; }}
    .card span {{ display: block; padding: 6px 8px; color: #f8fafc; font-size: 12px; font-weight: 600; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Hand 360 Turntable</h1>
    <p>{views} views around the original rigged hand (rest pose).</p>
    <div class="player">
      <div class="stage">
        <video id="vid" src="hand_turntable_360.mp4" autoplay loop muted playsinline controls></video>
      </div>
      <div class="controls">
        <button id="toggle">Pause</button>
        <input id="scrub" type="range" min="0" max="{len(names) - 1}" value="0" />
      </div>
    </div>
    <section class="grid">
      {cards}
    </section>
  </main>
  <script>
    const frames = [{frames_js}];
    const vid = document.getElementById('vid');
    const scrub = document.getElementById('scrub');
    const toggle = document.getElementById('toggle');
    let playing = true;
    toggle.addEventListener('click', () => {{
      playing = !playing;
      if (playing) {{ vid.play(); toggle.textContent = 'Pause'; }}
      else {{ vid.pause(); toggle.textContent = 'Play'; }}
    }});
    vid.addEventListener('timeupdate', () => {{
      if (!vid.duration) return;
      const frac = (vid.currentTime % (vid.duration / 4)) / (vid.duration / 4);
      scrub.value = Math.round(frac * (frames.length - 1));
    }});
    scrub.addEventListener('input', () => {{
      vid.pause(); playing = false; toggle.textContent = 'Play';
      const f = parseInt(scrub.value, 10) / frames.length;
      vid.currentTime = f * (vid.duration || 1);
    }});
  </script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    model = Path(args.input)
    blender = Path(args.blender_bin)
    out_dir = Path(args.output_dir)

    if not model.exists():
        print(f"Missing model: {model}")
        return 2
    if not blender.exists():
        print(f"Missing Blender binary: {blender}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_render:
        cmd = [
            str(blender),
            "--background",
            "--python",
            "scripts/blender_turntable_360.py",
            "--",
            "--input",
            str(model),
            "--output-dir",
            str(out_dir),
            "--views",
            str(args.views),
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print("Blender turntable render failed")
            return proc.returncode

    names = sorted(p.stem for p in out_dir.glob("view_*.png"))
    if not names:
        print("No frames rendered")
        return 1

    (out_dir / "index.html").write_text(render_gallery_html(names, args.views), encoding="utf-8")
    print(f"Wrote turntable gallery to {out_dir} ({len(names)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
