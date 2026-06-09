"""
hand_landscape/regions.py

Assign each tracked dot to an anatomical region of the hand using MediaPipe
hand landmarks, so tremor can be decomposed per finger / palm.

MediaPipe Hands landmark indices
--------------------------------
     0           wrist
     1  2  3  4  thumb   (CMC, MCP, IP, TIP)
     5  6  7  8  index   (MCP, PIP, DIP, TIP)
     9 10 11 12  middle
    13 14 15 16  ring
    17 18 19 20  pinky

A dot is assigned to the finger whose *distal* landmarks (PIP→TIP) it is
nearest to, unless it is closer to the palm centroid — then it is "palm".
"""

from __future__ import annotations

import numpy as np

REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")

# distal landmarks that best identify each finger's moving segment
_FINGER_DISTAL = {
    "thumb":  (2, 3, 4),
    "index":  (6, 7, 8),
    "middle": (10, 11, 12),
    "ring":   (14, 15, 16),
    "pinky":  (18, 19, 20),
}
# palm anchor landmarks (wrist + finger bases)
_PALM_ANCHORS = (0, 1, 5, 9, 13, 17)


def assign_regions(points_xy, landmark_points) -> list[str]:
    """Return a region label per point.

    points_xy        : (N, 2) pixel positions of tracked dots
    landmark_points  : (21, 2) pixel positions of MediaPipe landmarks, or None

    Falls back to "hand" for every point when landmarks are unavailable.
    """
    if points_xy is None or len(points_xy) == 0:
        return []
    pts = np.asarray(points_xy, dtype=np.float64)
    if landmark_points is None or len(landmark_points) < 21:
        return ["hand"] * len(pts)

    lm = np.asarray(landmark_points, dtype=np.float64)
    palm_centroid = np.mean(lm[list(_PALM_ANCHORS)], axis=0)

    # Precompute finger distal landmark coordinate sets.
    finger_pts = {name: lm[list(idx)] for name, idx in _FINGER_DISTAL.items()}

    labels: list[str] = []
    for p in pts:
        best_name = "palm"
        best_dist = float(np.linalg.norm(p - palm_centroid))
        for name, fpts in finger_pts.items():
            d = float(np.min(np.linalg.norm(fpts - p, axis=1)))
            if d < best_dist:
                best_dist = d
                best_name = name
        labels.append(best_name)
    return labels
