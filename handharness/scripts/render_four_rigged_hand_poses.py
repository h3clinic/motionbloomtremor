#!/usr/bin/env python3
"""Render four distinct poses from one rigged hand model using Blender."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = Path("input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb")
DEFAULT_RIG_MAP = Path("rig_map.json")
DEFAULT_OUT_DIR = Path("output/four_hand_poses")
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
MODEL_URL_HINT = "Set HAND_MODEL or pass --input to a rigged .glb/.fbx model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_MODEL))
    parser.add_argument("--rig-map", default=str(DEFAULT_RIG_MAP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--blender-bin", default=os.environ.get("BLENDER_BIN", DEFAULT_BLENDER))
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def chain(base_x, base_y, base_angle_deg, joints_deg, lengths):
    points = [(base_x, base_y, 0.0)]
    angle = math.radians(base_angle_deg)
    x, y = base_x, base_y
    for idx, length in enumerate(lengths):
        if idx > 0:
            angle += math.radians(joints_deg[idx - 1])
        x += length * math.sin(angle)
        y -= length * math.cos(angle)
        points.append((round(x, 5), round(y, 5), 0.0))
    return points


def make_pose(name: str, finger_specs: dict, thumb_spec: dict):
    wrist = (0.50, 0.84, 0.0)
    thumb_base = (0.43, 0.76)
    index_base = (0.48, 0.69)
    middle_base = (0.54, 0.68)
    ring_base = (0.60, 0.70)
    pinky_base = (0.66, 0.73)

    thumb = chain(thumb_base[0], thumb_base[1], thumb_spec["angle"], thumb_spec["joints"], thumb_spec["lengths"])
    index = chain(index_base[0], index_base[1], finger_specs["index"]["angle"], finger_specs["index"]["joints"], finger_specs["index"]["lengths"])
    middle = chain(middle_base[0], middle_base[1], finger_specs["middle"]["angle"], finger_specs["middle"]["joints"], finger_specs["middle"]["lengths"])
    ring = chain(ring_base[0], ring_base[1], finger_specs["ring"]["angle"], finger_specs["ring"]["joints"], finger_specs["ring"]["lengths"])
    pinky = chain(pinky_base[0], pinky_base[1], finger_specs["pinky"]["angle"], finger_specs["pinky"]["joints"], finger_specs["pinky"]["lengths"])

    # MediaPipe order: wrist, thumb(cmc/mcp/ip/tip), index(mcp/pip/dip/tip), middle, ring, pinky.
    pts = [
        wrist,
        thumb[0], thumb[1], thumb[2], thumb[3],
        index[0], index[1], index[2], index[3],
        middle[0], middle[1], middle[2], middle[3],
        ring[0], ring[1], ring[2], ring[3],
        pinky[0], pinky[1], pinky[2], pinky[3],
    ]
    return {
        "source": "synthetic",
        "coordinate_space": "normalized",
        "handedness": "right",
        "name": name,
        "landmark_count": 21,
        "landmarks": [{"x": x, "y": y, "z": z} for x, y, z in pts],
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_gallery_html(items):
    cards = []
    for item in items:
        cards.append(
            f'<a class="card" href="{item["name"]}.png" target="_blank" rel="noreferrer">'
            f'<img src="{item["name"]}.png" alt="{item["name"]}" loading="lazy" />'
            f'<span>{item["name"]}</span></a>'
        )
    card_html = "\n".join(cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>4 Hand Poses</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: linear-gradient(180deg, #20232a, #101114); color: #f4f4f5; }}
    .wrap {{ max-width: 1260px; margin: 24px auto; padding: 0 16px 32px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 52px); }}
    p {{ margin: 0 0 18px; color: #cbd5e1; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ display: block; border-radius: 16px; overflow: hidden; text-decoration: none; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 12px 32px rgba(0,0,0,.3); }}
    img {{ display: block; width: 100%; aspect-ratio: 1/1; object-fit: cover; background: #000; }}
    span {{ display: block; padding: 10px 12px; color: #f8fafc; font-weight: 700; letter-spacing: .01em; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>4 Different Hand Poses</h1>
    <p>Rendered from the attached rigged hand model using one shared rig map.</p>
    <section class="grid">
      {card_html}
    </section>
  </main>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    model = Path(args.input)
    rig_map = Path(args.rig_map)
    blender = Path(args.blender_bin)
    out_dir = Path(args.output_dir)

    if not model.exists():
        print(f"Missing model: {model}\n{MODEL_URL_HINT}")
        return 2
    if not rig_map.exists():
        print(f"Missing rig map: {rig_map}")
        return 2
    if not blender.exists():
        print(f"Missing Blender binary: {blender}")
        return 2

    ensure_dir(out_dir)
    for child in out_dir.glob("*"):
        if child.is_file():
            child.unlink()

    poses = [
        make_pose(
            "01_open",
            {
                "index": {"angle": 0, "joints": [2, 2, 0], "lengths": [0.10, 0.09, 0.08]},
                "middle": {"angle": 0, "joints": [2, 2, 0], "lengths": [0.11, 0.10, 0.08]},
                "ring": {"angle": 2, "joints": [2, 2, 0], "lengths": [0.10, 0.09, 0.07]},
                "pinky": {"angle": 6, "joints": [3, 2, 0], "lengths": [0.09, 0.08, 0.06]},
            },
            {"angle": -28, "joints": [3, 2, 0], "lengths": [0.06, 0.045, 0.035]},
        ),
        make_pose(
            "02_peace",
            {
                "index": {"angle": 0, "joints": [2, 2, 0], "lengths": [0.11, 0.10, 0.08]},
                "middle": {"angle": 0, "joints": [2, 2, 0], "lengths": [0.12, 0.10, 0.08]},
                "ring": {"angle": 10, "joints": [85, 95, 60], "lengths": [0.08, 0.07, 0.06]},
                "pinky": {"angle": 12, "joints": [85, 95, 60], "lengths": [0.07, 0.06, 0.05]},
            },
            {"angle": -40, "joints": [55, 50, 30], "lengths": [0.06, 0.045, 0.035]},
        ),
        make_pose(
            "03_fist",
            {
                "index": {"angle": 4, "joints": [85, 95, 60], "lengths": [0.07, 0.06, 0.05]},
                "middle": {"angle": 5, "joints": [88, 98, 62], "lengths": [0.075, 0.065, 0.05]},
                "ring": {"angle": 7, "joints": [88, 98, 62], "lengths": [0.07, 0.06, 0.05]},
                "pinky": {"angle": 9, "joints": [88, 98, 62], "lengths": [0.065, 0.055, 0.045]},
            },
            {"angle": -48, "joints": [50, 55, 35], "lengths": [0.05, 0.042, 0.03]},
        ),
        make_pose(
            "04_thumb_out",
            {
                "index": {"angle": 6, "joints": [80, 92, 58], "lengths": [0.07, 0.06, 0.05]},
                "middle": {"angle": 6, "joints": [82, 94, 58], "lengths": [0.075, 0.065, 0.05]},
                "ring": {"angle": 8, "joints": [82, 94, 58], "lengths": [0.07, 0.06, 0.05]},
                "pinky": {"angle": 10, "joints": [82, 94, 58], "lengths": [0.065, 0.055, 0.045]},
            },
            {"angle": -62, "joints": [2, 1, 0], "lengths": [0.07, 0.06, 0.05]},
        ),
    ]

    pose_jsons = []
    for pose in poses:
        pose_json = out_dir / f"{pose['name']}.json"
        pose_png = out_dir / f"{pose['name']}.png"
        pose_glb = out_dir / f"{pose['name']}.glb"
        write_json(pose_json, pose)
        pose_jsons.append(pose_json)

        cmd = [
            str(blender),
            "--background",
            "--python",
            "scripts/blender_apply_one_mediapipe_pose.py",
            "--",
            "--input",
            str(model),
            "--pose-json",
            str(pose_json),
            "--rig-map",
            str(rig_map),
            "--output-glb",
            str(pose_glb),
            "--output-png",
            str(pose_png),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    (out_dir / "index.html").write_text(render_gallery_html(poses), encoding="utf-8")
    print(f"Wrote four-pose gallery to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
