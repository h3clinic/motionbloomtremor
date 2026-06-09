#!/usr/bin/env python3
"""Render tremor sequences across multiple intensities and build a comparison gallery.

Usage:
  python3 scripts/render_tremor_dataset.py \
    --input input/hand_base/.../hand.glb \
    --output-dir output/tremor_dataset \
    --intensities 1,20,40,60,80,100 \
    --frames 60 --fps 60

For each intensity this script:
  1. Calls Blender headless to bake + render the PNG frame sequence.
  2. Encodes the sequence to an H.264 MP4 with ffmpeg.
  3. Reads metadata.json to extract physical parameters.
Builds a final HTML gallery with:
  - A 1–100 intensity slider that switches between the pre-rendered videos.
  - Side-by-side grid of all intensity videos with amplitude/frequency labels.
  - Download links for MP4s.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL   = Path("input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb")
DEFAULT_OUT_DIR = Path("output/tremor_dataset")
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_INTENSITIES = "1-100"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--intensities", default=DEFAULT_INTENSITIES,
              help="Comma-separated values and/or ranges, e.g. 1-100 or 1,20,40")
    parser.add_argument("--frames",      type=int, default=90,
                        help="Frames per sequence (90 @ 30fps = 3 sec loop)")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--resolution",  type=int, default=512,
                        help="Render resolution (use 512 for speed, 1024 for quality)")
    parser.add_argument("--rig-map",     default="rig_map.json")
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", DEFAULT_BLENDER))
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ffmpeg",      default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip Blender; rebuild HTML from existing frames/videos")
    return parser.parse_args()


def intensity_dir(out_dir: Path, intensity: int) -> Path:
    return out_dir / f"intensity_{intensity:03d}"


def parse_intensities(spec: str) -> list[int]:
  values = set()
  for token in [t.strip() for t in spec.split(",") if t.strip()]:
    if "-" in token:
      lo_s, hi_s = token.split("-", 1)
      lo = int(lo_s.strip())
      hi = int(hi_s.strip())
      if lo > hi:
        lo, hi = hi, lo
      for v in range(lo, hi + 1):
        if 0 <= v <= 100:
          values.add(v)
    else:
      v = int(token)
      if 0 <= v <= 100:
        values.add(v)
  return sorted(values)


def encode_video(ffmpeg_bin: str, frames_dir: Path, fps: int, mp4_path: Path) -> bool:
    """Encode frame_NNNN.png sequence → H.264 MP4 looped once."""
    # Check frames exist
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"  [WARN] No frames found in {frames_dir}")
        return False

    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def load_meta(frames_dir: Path) -> dict:
    meta_path = frames_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def render_gallery_html(items: list[dict]) -> str:
    """items: list of {intensity, rel_video, amplitude_deg, frequency_hz, mp4_name}"""

    # Build data arrays for the slider JS
    intensities_js = ", ".join(str(d["intensity"]) for d in items)
    videos_js = ", ".join(f'"{d["rel_video"]}"' for d in items)
    labels_js = ", ".join(f'"{d["label"]}"' for d in items)

    video_tags = []
    for d in items:
        video_tags.append(
            f'<div class="vid-cell">'
            f'<div class="vid-label">Intensity {d["intensity"]}</div>'
            f'<video src="{d["rel_video"]}" autoplay loop muted playsinline></video>'
            f'<div class="vid-meta">{d["label"]}</div>'
            f'<a class="dl" href="{d["rel_video"]}" download>Download MP4</a>'
            f'</div>'
        )
    grid_html = "\n".join(video_tags)

    # Intensity → nearest prerendered video (for the slider)
    intensity_to_idx = "[\n"
    for i in range(1, 101):
        closest = min(range(len(items)), key=lambda k: abs(items[k]["intensity"] - i))
        intensity_to_idx += f"  {closest},\n"
    intensity_to_idx += "]"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tremor Intensity Dataset</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Roboto, Arial, sans-serif;
           background: linear-gradient(180deg, #1a1d23, #0c0d10); color: #f0f0f2; min-height: 100vh; }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ font-size: clamp(24px, 4vw, 44px); margin: 0 0 6px; }}
    .sub {{ color: #8b93a4; margin: 0 0 28px; font-size: 15px; }}

    /* Slider demo */
    .demo {{ background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
             border-radius: 20px; padding: 24px; margin-bottom: 36px; }}
    .demo h2 {{ margin: 0 0 16px; font-size: 20px; }}
    .demo-inner {{ display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }}
    .stage {{ width: min(340px, 90vw); aspect-ratio: 1/1; border-radius: 14px; overflow: hidden;
              background: #000; border: 1px solid rgba(255,255,255,.15); flex-shrink: 0; }}
    .stage video {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .slider-panel {{ flex: 1; min-width: 220px; }}
    .slider-panel label {{ display: block; font-size: 13px; color: #8b93a4; margin-bottom: 8px; }}
    input[type=range] {{ width: 100%; accent-color: #3b82f6; }}
    .intensity-val {{ font-size: 52px; font-weight: 800; line-height: 1;
                      background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent; }}
    .phys-info {{ margin-top: 16px; display: flex; flex-direction: column; gap: 8px; }}
    .pill {{ display: inline-flex; gap: 8px; align-items: center; background: rgba(255,255,255,.08);
             border-radius: 999px; padding: 5px 14px; font-size: 13px; }}
    .pill-key {{ color: #8b93a4; }}
    .pill-val {{ font-weight: 700; color: #f0f0f2; }}

    /* Grid */
    h2.section {{ margin: 0 0 16px; font-size: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
    .vid-cell {{ background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
                border-radius: 14px; overflow: hidden; }}
    .vid-label {{ padding: 10px 12px 4px; font-weight: 700; font-size: 14px; }}
    .vid-cell video {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; background: #000; }}
    .vid-meta {{ padding: 6px 12px 2px; font-size: 12px; color: #8b93a4; }}
    .dl {{ display: block; padding: 6px 12px 10px; font-size: 12px; color: #60a5fa; text-decoration: none; }}
    .dl:hover {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Tremor Intensity Dataset</h1>
    <p class="sub">Clinically-mapped tremor scale 1–100 rendered onto the anatomical hand rig.
      Amplitude scales to 12 cm peak-to-peak at intensity 100; frequency decays from 10 Hz to 4 Hz.</p>

    <div class="demo">
      <h2>Live Intensity Selector</h2>
      <div class="demo-inner">
        <div class="stage"><video id="demo-vid" autoplay loop muted playsinline></video></div>
        <div class="slider-panel">
          <label for="intensity-slider">Tremor Intensity</label>
          <div class="intensity-val" id="intensity-display">50</div>
          <input id="intensity-slider" type="range" min="1" max="100" value="50" />
          <div class="phys-info">
            <div class="pill">
              <span class="pill-key">Amplitude</span>
              <span class="pill-val" id="amp-val">—</span>
            </div>
            <div class="pill">
              <span class="pill-key">Frequency</span>
              <span class="pill-val" id="freq-val">—</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <h2 class="section">All Rendered Intensities</h2>
    <div class="grid">
      {grid_html}
    </div>
  </main>

  <script>
    const intensityToIdx = {intensity_to_idx};
    const videos = [{videos_js}];
    const labels = [{labels_js}];
    const intensities = [{intensities_js}];

    // Pre-load all videos invisibly so switching is instant
    const vidCache = videos.map(src => {{
      const v = document.createElement('video');
      v.src = src; v.autoplay = true; v.loop = true; v.muted = true; v.playsInline = true;
      v.load();
      return v;
    }});

    const demoVid = document.getElementById('demo-vid');
    const slider  = document.getElementById('intensity-slider');
    const dispEl  = document.getElementById('intensity-display');
    const ampEl   = document.getElementById('amp-val');
    const freqEl  = document.getElementById('freq-val');

    function ampForIntensity(i) {{
      const ampRad = (i / 100.0) * 0.15;
      const ampCm = i * 0.12;
      return ampCm.toFixed(2) + ' cm  (' + (ampRad * 180 / Math.PI).toFixed(2) + '°)';
    }}
    function freqForIntensity(i) {{
      return (10.0 - (i / 100.0) * 6.0).toFixed(2) + ' Hz';
    }}

    function update(intensity) {{
      const idx = intensityToIdx[Math.max(1, Math.min(100, intensity)) - 1];
      const src = videos[idx];
      if (demoVid.src !== new URL(src, location.href).href) {{
        const t = demoVid.currentTime;
        demoVid.src = src;
        demoVid.load();
        demoVid.play();
      }}
      dispEl.textContent = intensity;
      ampEl.textContent  = ampForIntensity(intensity);
      freqEl.textContent = freqForIntensity(intensity);
    }}

    slider.addEventListener('input', () => update(parseInt(slider.value, 10)));
    update(50);
  </script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    model   = Path(args.input)
    out_dir = Path(args.output_dir)
    blender = Path(args.blender_bin)
    ffmpeg  = args.ffmpeg

    if not model.exists():
        print(f"Missing model: {model}")
        return 2
    if not args.skip_render and not blender.exists():
        print(f"Missing Blender binary: {blender}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    intensities = parse_intensities(args.intensities)

    items = []
    for intensity in intensities:
        print(f"\n--- Intensity {intensity:3d} ---")
        idir = intensity_dir(out_dir, intensity)
        idir.mkdir(parents=True, exist_ok=True)
        mp4_path = idir / "tremor.mp4"

        if not args.skip_render:
            cmd = [
                str(blender), "--background", "--python",
                "scripts/blender_tremor_sequence.py", "--",
                "--input",      str(model),
                "--output-dir", str(idir),
                "--intensity",  str(intensity),
                "--frames",     str(args.frames),
                "--fps",        str(args.fps),
                "--rig-map",    args.rig_map,
                "--resolution", str(args.resolution),
                "--seed",       str(args.seed),
            ]
            print("  Blender:", " ".join(cmd))
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"  [ERROR] Blender failed for intensity {intensity}")
                continue

        print(f"  Encoding MP4…")
        ok = encode_video(ffmpeg, idir, args.fps, mp4_path)
        if not ok:
            print(f"  [WARN] ffmpeg encoding failed for intensity {intensity}")
            continue

        meta = load_meta(idir)
        labels = meta.get("labels", {})
        phys = labels.get("physical_metrics", {})
        clinical = labels.get("clinical_binned_references", {})

        amp_deg = phys.get("calculated_peak_amplitude_deg", round(intensity / 100 * 8.5944, 2))
        amp_cm = phys.get("calculated_peak_amplitude_cm", round(intensity * 0.12, 2))
        freq_hz = phys.get("simulated_frequency_hz", round(10.0 - intensity / 100 * 6.0, 2))
        classification = clinical.get("classification", "")
        rel_video = f"intensity_{intensity:03d}/tremor.mp4"
        class_suffix = f" · {classification}" if classification else ""
        label = f"{amp_cm:.2f} cm ({amp_deg:.2f}°) · {freq_hz:.2f} Hz{class_suffix}"
        items.append({"intensity": intensity, "rel_video": rel_video,
                "label": label, "amplitude_deg": amp_deg, "frequency_hz": freq_hz})
        print(f"  Done: {label}")

    if not items:
        print("No intensities completed successfully.")
        return 1

    html = render_gallery_html(items)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\nWrote tremor dataset gallery to {out_dir} ({len(items)} intensities)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
