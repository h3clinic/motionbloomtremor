"""
hand_landscape/observation.py

All evidence extracted from one video frame.

The landscape uses this to score its candidate states. The observation is
built by the bridge's main loop and passed into HandStateLandscape.update().

optical_flow_points  : (N, 2) float64 – new tracked positions
optical_flow_deltas  : (N, 2) float64 – (new - old) per-point displacement
landmark_points      : (K, 2) float64 | None – MediaPipe landmarks if available
detection_confidence : float in [0, 1] – MediaPipe model confidence
detected_scale_z     : float | None   – sqrt(box_area / ref_area) or None
frame_quality        : float in [0, 1] – brightness/blur heuristic
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """Evidence extracted from one frame."""

    # Hand-centre position from the detector (pixel coords)
    detected_center_x: float | None = None
    detected_center_y: float | None = None

    # Pseudo-depth scale: sqrt(box_area / reference_area)
    detected_scale_z: float | None = None

    # MediaPipe box itself – (x, y, w, h) in pixels
    detected_box: tuple[int, int, int, int] | None = None

    # Optical-flow evidence
    optical_flow_points: np.ndarray | None = None   # (N, 2) new positions
    optical_flow_deltas: np.ndarray | None = None   # (N, 2) = new – old

    # Landmark positions if still used (K, 2) or None
    landmark_points: np.ndarray | None = None

    # MediaPipe's own confidence in detecting the hand
    detection_confidence: float = 0.0

    # Simple frame-quality heuristic (0 = black/blurry, 1 = ideal)
    frame_quality: float = 1.0

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    @property
    def has_detection(self) -> bool:
        return self.detected_center_x is not None and self.detected_center_y is not None

    @property
    def detected_center_xy(self) -> np.ndarray | None:
        if not self.has_detection:
            return None
        return np.array([self.detected_center_x, self.detected_center_y], dtype=np.float64)

    @property
    def macro_flow_vector(self) -> np.ndarray | None:
        """Median optical-flow vector across all tracked points.

        This is the most reliable single-frame estimate of the hand's bulk
        motion — robust to individual-point noise.
        """
        if self.optical_flow_deltas is None or len(self.optical_flow_deltas) < 3:
            return None
        return np.median(self.optical_flow_deltas, axis=0)

    @property
    def n_flow_points(self) -> int:
        if self.optical_flow_points is None:
            return 0
        return len(self.optical_flow_points)
