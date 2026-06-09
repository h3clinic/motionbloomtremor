"""
hand_landscape/scoring.py

Scores each candidate HandState against the current frame Observation.

The score is a product of (clipped) likelihood terms:

  1. Spatial proximity   – how close is the candidate to the detected hand centre?
  2. Optical-flow match  – does the candidate's velocity agree with the bulk flow?
  3. Scale consistency   – does the candidate's pseudo-depth match the box size?
  4. Acceleration penalty– penalise physically impossible accelerations.
  5. Prior weight        – carry forward confidence from the previous frame.

Each term is in (0, 1]. The final weight is the product × candidate.weight.

Design decisions
----------------
- We do NOT use exponential likelihoods with hand-tuned σ that the caller must
  tune. Instead we use a smooth sigmoid / tanh clamp so the scorer degrades
  gracefully when the detector is absent.
- When detection_confidence is low (detector uncertain), the spatial term is
  down-weighted and the flow term is up-weighted.
- When n_flow_points is small, the flow term is also down-weighted.
"""

from __future__ import annotations

import math

import numpy as np

from .observation import Observation
from .state import HandState


# ─── tunables ────────────────────────────────────────────────────────────────
SPATIAL_SIGMA_PX = 35.0   # spatial proximity half-width in pixels
FLOW_SIGMA_PXF  = 4.0     # optical-flow agreement half-width in px/frame
SCALE_SIGMA     = 0.18    # pseudo-depth agreement half-width
MAX_ACCEL_PX    = 18.0    # threshold for penalising impossible acceleration
MIN_FLOW_PTS    = 5       # below this, flow term collapses to neutral


def _gaussian_like(distance: float, sigma: float) -> float:
    """Gaussian likelihood: exp(-0.5*(d/σ)²) clamped to (1e-6, 1)."""
    return max(1e-6, math.exp(-0.5 * (distance / max(sigma, 1e-9)) ** 2))


def score_candidate(candidate: HandState, obs: Observation) -> float:
    """Return a positive weight proportional to how well the candidate fits the frame.

    Returns a float in (0, 1].
    """
    det_conf = obs.detection_confidence  # [0, 1]
    flow_pts = obs.n_flow_points

    # ── 1. Spatial proximity ──────────────────────────────────────────────────
    if obs.has_detection and det_conf > 0.15:
        dx = candidate.center_x - (obs.detected_center_x or 0.0)
        dy = candidate.center_y - (obs.detected_center_y or 0.0)
        dist = math.hypot(dx, dy)
        # Detector confidence scales the effective sigma: uncertain detector
        # allows states farther from its centre to survive.
        eff_sigma = SPATIAL_SIGMA_PX / max(0.3, det_conf)
        spatial_term = _gaussian_like(dist, eff_sigma)
    else:
        spatial_term = 0.3   # neutral when detector is absent

    # ── 2. Optical-flow agreement ─────────────────────────────────────────────
    macro_flow = obs.macro_flow_vector  # (2,) or None
    if macro_flow is not None and flow_pts >= MIN_FLOW_PTS:
        pred_vx, pred_vy = candidate.velocity_x, candidate.velocity_y
        flow_dx = float(macro_flow[0]) - pred_vx
        flow_dy = float(macro_flow[1]) - pred_vy
        flow_dist = math.hypot(flow_dx, flow_dy)
        flow_term = _gaussian_like(flow_dist, FLOW_SIGMA_PXF)
        # Trust flow more when detector is uncertain
        flow_weight = 0.6 + 0.4 * (1.0 - det_conf)
    else:
        flow_term = 0.6   # neutral
        flow_weight = 0.5

    # ── 3. Scale consistency ──────────────────────────────────────────────────
    if obs.detected_scale_z is not None:
        ds = candidate.scale_z - obs.detected_scale_z
        scale_term = _gaussian_like(abs(ds), SCALE_SIGMA)
    else:
        scale_term = 0.7   # neutral

    # ── 4. Acceleration penalty ───────────────────────────────────────────────
    accel_mag = math.hypot(candidate.acceleration_x, candidate.acceleration_y)
    if accel_mag > MAX_ACCEL_PX:
        # Soft penalty that grows with excess acceleration
        accel_penalty = math.exp(-0.15 * (accel_mag - MAX_ACCEL_PX) / max(MAX_ACCEL_PX, 1.0))
    else:
        accel_penalty = 1.0

    # ── 5. Prior weight (confidence carry-forward) ────────────────────────────
    prior = max(0.05, candidate.confidence)

    # ── Combine ────────────────────────────────────────────────────────────────
    # Spatial gets 40%, flow gets (30% × flow_weight), scale gets 15%
    combined = (
        0.40 * spatial_term
        + 0.30 * flow_weight * flow_term
        + 0.15 * scale_term
        + 0.15 * prior
    ) * accel_penalty

    return max(1e-9, combined)


def score_all_candidates(candidates: list[HandState], obs: Observation) -> None:
    """Update candidate.weight in-place by scoring each against obs.

    Normalises so weights sum to 1.
    """
    for c in candidates:
        c.weight = c.weight * score_candidate(c, obs)

    total = sum(c.weight for c in candidates)
    if total < 1e-12:
        # All candidates collapsed – give equal weight (recovery)
        for c in candidates:
            c.weight = 1.0 / len(candidates)
    else:
        for c in candidates:
            c.weight /= total
