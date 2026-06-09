"""
hand_landscape/state.py

One candidate hand state hypothesis.

Every frame the landscape maintains N of these, each with a
slightly different position, velocity, acceleration, and pseudo-depth
(scale_z). The landscape scores them, normalises their weights, prunes
the weak ones, and merges near-duplicates. The result is a probability
cloud over possible hand states rather than a single tracked box.

scale_z is NOT true depth. On a monocular webcam it is estimated from
hand-box area relative to a reference area:

    scale_z = sqrt(box_area / reference_area)     (≈1 when at init distance)

Larger box → scale_z > 1 (hand closer).
Smaller box → scale_z < 1 (hand farther).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandState:
    """One hypothesis about the hand's current kinematic state."""

    # Spatial position in image-pixel coordinates
    center_x: float = 0.0
    center_y: float = 0.0

    # Pseudo-depth derived from apparent box size (1.0 = reference distance)
    scale_z: float = 1.0

    # First-order dynamics in each axis
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0  # change in scale per frame

    # Second-order dynamics
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_z: float = 0.0

    # Particle-filter weight (un-normalised; normalised by landscape)
    weight: float = 1.0

    # Quality indicators
    confidence: float = 0.5   # [0, 1] local tracking confidence
    age: int = 0              # frames this state has been alive

    # ------------------------------------------------------------------ #
    # Derived / convenience                                                #
    # ------------------------------------------------------------------ #

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)

    @property
    def velocity_xy(self) -> np.ndarray:
        return np.array([self.velocity_x, self.velocity_y], dtype=np.float64)

    @property
    def velocity_3d(self) -> np.ndarray:
        return np.array([self.velocity_x, self.velocity_y, self.velocity_z], dtype=np.float64)

    @property
    def acceleration_xy(self) -> np.ndarray:
        return np.array([self.acceleration_x, self.acceleration_y], dtype=np.float64)

    @property
    def speed(self) -> float:
        return math.hypot(self.velocity_x, self.velocity_y)

    @property
    def acceleration_magnitude(self) -> float:
        return math.hypot(self.acceleration_x, self.acceleration_y)

    # ------------------------------------------------------------------ #
    # Propagation                                                          #
    # ------------------------------------------------------------------ #

    def predicted_center(self) -> tuple[float, float]:
        """Return (x, y) one frame into the future using the kinematic model."""
        px = self.center_x + self.velocity_x + 0.5 * self.acceleration_x
        py = self.center_y + self.velocity_y + 0.5 * self.acceleration_y
        return (px, py)

    def predicted_scale(self) -> float:
        return self.scale_z + self.velocity_z + 0.5 * self.acceleration_z

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "scale_z": round(self.scale_z, 4),
            "velocity_x": round(self.velocity_x, 3),
            "velocity_y": round(self.velocity_y, 3),
            "velocity_z": round(self.velocity_z, 4),
            "acceleration_x": round(self.acceleration_x, 3),
            "acceleration_y": round(self.acceleration_y, 3),
            "confidence": round(self.confidence, 3),
            "weight": round(self.weight, 4),
            "age": self.age,
        }
