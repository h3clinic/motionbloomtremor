#!/usr/bin/env python3
"""Generate one synthetic MediaPipe-like peace-sign landmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="output/mediapipe/peace_pose_dummy.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Approximate right-hand peace sign in normalized coordinates.
    pts = [
        (0.50, 0.78, 0.00),  # 0 wrist
        (0.44, 0.73, -0.01), (0.40, 0.67, -0.02), (0.38, 0.61, -0.02), (0.39, 0.56, -0.02),  # thumb
        (0.49, 0.63, -0.02), (0.49, 0.52, -0.03), (0.49, 0.40, -0.03), (0.49, 0.29, -0.03),  # index up
        (0.55, 0.63, -0.02), (0.55, 0.51, -0.03), (0.55, 0.39, -0.03), (0.55, 0.30, -0.03),  # middle up
        (0.60, 0.66, -0.01), (0.61, 0.62, -0.01), (0.61, 0.59, -0.01), (0.60, 0.57, -0.01),  # ring curled
        (0.65, 0.69, -0.01), (0.66, 0.66, -0.01), (0.66, 0.63, -0.01), (0.65, 0.61, -0.01),  # pinky curled
    ]

    payload = {
        "source": "synthetic",
        "coordinate_space": "normalized",
        "handedness": "right",
        "landmark_count": 21,
        "landmarks": [{"x": x, "y": y, "z": z} for (x, y, z) in pts],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
