"""
hand_landscape/dynamics.py

Generates a spread of candidate next-states from the current weighted state.

Goal: do NOT produce one predicted future. Instead produce a *spread* of
plausible futures that brackets all likely hand behaviours for this frame
interval. The landscape then scores each candidate against the actual frame
evidence and discards the bad ones.

Variants produced around the weighted state:

  smooth      : continue at current velocity, no acceleration change
  accel+      : slight positive acceleration (hand speeding up)
  accel-      : slight deceleration
  right/left  : lateral deviation ± lateral_sigma pixels
  up/down     : vertical deviation ± lateral_sigma pixels
  closer      : scale_z increases (hand moving toward camera)
  farther     : scale_z decreases (hand moving away)
  recovery    : low-confidence state that preserves trajectory but shrinks

The caller chooses N (default 32 to 128). The variants are repeated with
small Gaussian jitter to fill the particle budget.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .state import HandState


# ─── tunables ────────────────────────────────────────────────────────────────
LATERAL_SIGMA_PX = 4.0    # 1-σ spatial jitter around the deterministic variants
SCALE_SIGMA = 0.04        # 1-σ jitter on scale_z
ACCEL_DELTA = 1.2         # px/frame² added/removed for accel variants
VELOCITY_DAMPING = 0.94   # velocity decay each propagation step
ACCEL_DAMPING = 0.82      # acceleration decay each propagation step

# Relative weight of each variant (before re-normalisation)
_VARIANT_WEIGHTS = {
    "smooth":   1.6,
    "accel+":   0.9,
    "accel-":   0.9,
    "right":    0.7,
    "left":     0.7,
    "up":       0.7,
    "down":     0.7,
    "closer":   0.5,
    "farther":  0.5,
    "recovery": 0.4,
}


def _propagate_base(state: HandState) -> HandState:
    """Apply one kinematic step to produce a single predicted state."""
    s = HandState(
        center_x=state.center_x + state.velocity_x + 0.5 * state.acceleration_x,
        center_y=state.center_y + state.velocity_y + 0.5 * state.acceleration_y,
        scale_z=state.scale_z + state.velocity_z + 0.5 * state.acceleration_z,
        velocity_x=state.velocity_x * VELOCITY_DAMPING + state.acceleration_x,
        velocity_y=state.velocity_y * VELOCITY_DAMPING + state.acceleration_y,
        velocity_z=state.velocity_z * VELOCITY_DAMPING + state.acceleration_z,
        acceleration_x=state.acceleration_x * ACCEL_DAMPING,
        acceleration_y=state.acceleration_y * ACCEL_DAMPING,
        acceleration_z=state.acceleration_z * ACCEL_DAMPING,
        confidence=state.confidence,
        weight=state.weight,
        age=state.age + 1,
    )
    # Clamp scale to a plausible range
    s.scale_z = max(0.2, min(3.0, s.scale_z))
    return s


def _jitter(state: HandState, rng: np.random.Generator, spatial_sigma: float = LATERAL_SIGMA_PX) -> HandState:
    """Add Gaussian noise to position and scale."""
    dx, dy = rng.normal(0.0, spatial_sigma, 2)
    dz = rng.normal(0.0, SCALE_SIGMA)
    s = HandState(
        center_x=state.center_x + dx,
        center_y=state.center_y + dy,
        scale_z=max(0.2, min(3.0, state.scale_z + dz)),
        velocity_x=state.velocity_x,
        velocity_y=state.velocity_y,
        velocity_z=state.velocity_z,
        acceleration_x=state.acceleration_x,
        acceleration_y=state.acceleration_y,
        acceleration_z=state.acceleration_z,
        confidence=state.confidence,
        weight=state.weight,
        age=state.age,
    )
    return s


def generate_candidates(
    weighted_state: HandState,
    n_candidates: int = 64,
    rng: np.random.Generator | None = None,
) -> list[HandState]:
    """Return n_candidates candidate states spread around the kinematic prediction.

    Parameters
    ----------
    weighted_state:
        The current weighted (expected) hand state from the landscape.
    n_candidates:
        Target number of candidate particles.  Minimum useful value is 16.
    rng:
        Optional RNG for reproducibility in tests.
    """
    if rng is None:
        rng = np.random.default_rng()

    base = _propagate_base(weighted_state)

    # --- Deterministic variant templates ---------------------------------
    vx, vy = base.velocity_x, base.velocity_y
    speed = math.hypot(vx, vy)
    # Unit direction (capped to avoid divide-by-zero)
    if speed > 0.5:
        nx, ny = vx / speed, vy / speed
    else:
        nx, ny = 1.0, 0.0

    variants: dict[str, HandState] = {}

    # smooth – unmodified propagation
    variants["smooth"] = base

    # accel+ – slightly faster
    ap = _propagate_base(weighted_state)
    ap.velocity_x += ACCEL_DELTA * nx
    ap.velocity_y += ACCEL_DELTA * ny
    variants["accel+"] = ap

    # accel- – slightly slower
    am = _propagate_base(weighted_state)
    am.velocity_x -= ACCEL_DELTA * nx * 0.7
    am.velocity_y -= ACCEL_DELTA * ny * 0.7
    variants["accel-"] = am

    # lateral variants – perpendicular to current heading
    perp_x, perp_y = -ny, nx
    for name, sign in [("right", 1.0), ("left", -1.0)]:
        s = _propagate_base(weighted_state)
        s.center_x += sign * LATERAL_SIGMA_PX * 2.0 * perp_x
        s.center_y += sign * LATERAL_SIGMA_PX * 2.0 * perp_y
        variants[name] = s

    # up/down
    for name, sign in [("up", -1.0), ("down", 1.0)]:
        s = _propagate_base(weighted_state)
        s.center_y += sign * LATERAL_SIGMA_PX * 2.0
        variants[name] = s

    # closer / farther in pseudo-depth
    sc = _propagate_base(weighted_state)
    sc.scale_z = max(0.2, min(3.0, sc.scale_z + SCALE_SIGMA * 3.0))
    variants["closer"] = sc

    sf = _propagate_base(weighted_state)
    sf.scale_z = max(0.2, min(3.0, sf.scale_z - SCALE_SIGMA * 3.0))
    variants["farther"] = sf

    # recovery – preserve prior trajectory but with low confidence/weight
    rec = _propagate_base(weighted_state)
    rec.confidence = max(0.05, weighted_state.confidence * 0.6)
    rec.weight = 0.4
    variants["recovery"] = rec

    # --- Distribute n_candidates across variants proportionally ----------
    total_prior = sum(_VARIANT_WEIGHTS.values())
    candidates: list[HandState] = []

    for vname, template in variants.items():
        prior = _VARIANT_WEIGHTS[vname]
        count = max(1, round(n_candidates * prior / total_prior))
        # Determine per-variant spatial sigma (recovery uses larger jitter)
        sigma = LATERAL_SIGMA_PX * (2.0 if vname == "recovery" else 1.0)
        for _ in range(count):
            c = _jitter(template, rng, sigma)
            c.weight = prior / total_prior
            candidates.append(c)

    # Trim or top-up to exactly n_candidates using the smooth variant
    while len(candidates) > n_candidates:
        candidates.pop()
    while len(candidates) < n_candidates:
        c = _jitter(base, rng, LATERAL_SIGMA_PX)
        c.weight = _VARIANT_WEIGHTS["smooth"] / total_prior
        candidates.append(c)

    return candidates
