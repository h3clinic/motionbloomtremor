"""
hand_landscape/landscape.py

The core particle-filter-inspired hand state landscape.

Maintains N candidate HandState objects. Every frame it:

  1. Propagates candidates forward using the dynamics model.
  2. Scores them against the frame Observation.
  3. Normalises, prunes, and merges near-duplicate states.
  4. Resamples to restore particle diversity.
  5. Exposes the best_state (highest weight) and weighted_state (mean).

The weighted_state is the best single estimate of the hand's position,
used to compute the macro_delta for tremor residual extraction.

Usage example
-------------
    landscape = HandStateLandscape(n_candidates=64)

    # per frame:
    obs = Observation(...)
    landscape.update(obs)
    best  = landscape.best_state
    wmean = landscape.weighted_state
    unc   = landscape.state_uncertainty
    valid = landscape.validity_status
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .dynamics import generate_candidates
from .observation import Observation
from .scoring import score_all_candidates
from .state import HandState


# ─── tunables ────────────────────────────────────────────────────────────────
N_CANDIDATES_DEFAULT = 64
PRUNE_WEIGHT_THRESHOLD = 0.003   # states below this fraction are culled
MERGE_DISTANCE_PX = 8.0          # merge two states closer than this in 2-D
RESAMPLE_NOISE_SIGMA = 2.0       # spatial jitter when resampling
CONFIDENCE_DECAY = 0.08          # confidence lost per frame without detection
CONFIDENCE_GAIN  = 0.12          # confidence gained per frame with detection
LOST_TOLERANCE   = 12            # frames without detection before HAND_LOST
UNCERTAINTY_DIFFUSE_THRESHOLD = 40.0  # px – landscape considered diffuse above this


class HandStateLandscape:
    """Multi-hypothesis hand tracking via weighted particle landscape."""

    def __init__(
        self,
        n_candidates: int = N_CANDIDATES_DEFAULT,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_candidates = n_candidates
        self._rng = rng or np.random.default_rng()

        self._candidates: list[HandState] = []
        self._frame_count: int = 0
        self._lost_frames: int = 0

        # Smoothed state estimates (updated by _compute_estimates)
        self._best_state: HandState | None = None
        self._weighted_state: HandState | None = None
        self._prev_weighted_state: HandState | None = None

        # Diagnostics
        self._state_uncertainty: float = 0.0    # spread of candidates in px
        self._confidence: float = 0.0
        self._validity_status: str = "INSUFFICIENT_DURATION"
        self._detected_scale_ref: float | None = None  # reference box area for scale_z

        # Rolling history for duration-based checks
        self._span_buffer: deque = deque(maxlen=2)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def best_state(self) -> HandState | None:
        return self._best_state

    @property
    def weighted_state(self) -> HandState | None:
        return self._weighted_state

    @property
    def prev_weighted_state(self) -> HandState | None:
        return self._prev_weighted_state

    @property
    def state_uncertainty(self) -> float:
        """2-D spatial spread of candidates in pixels (weighted std dev)."""
        return self._state_uncertainty

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def validity_status(self) -> str:
        return self._validity_status

    @property
    def candidates(self) -> list[HandState]:
        return self._candidates

    def update(self, obs: Observation, timestamp: float | None = None) -> None:
        """Ingest one frame's observation and advance the landscape.

        Parameters
        ----------
        obs:
            Evidence extracted from the current frame.
        timestamp:
            Optional wall-clock time used for duration tracking.
        """
        self._frame_count += 1
        if timestamp is not None:
            self._span_buffer.append(timestamp)

        self._prev_weighted_state = self._weighted_state

        # ── Bootstrap on first detection ─────────────────────────────────────
        if not self._candidates:
            if obs.has_detection:
                self._bootstrap(obs)
                return
            else:
                self._validity_status = "INSUFFICIENT_DURATION"
                return

        # ── Track lost ───────────────────────────────────────────────────────
        if not obs.has_detection:
            self._lost_frames += 1
            self._confidence = max(0.0, self._confidence - CONFIDENCE_DECAY)
        else:
            self._lost_frames = 0
            self._confidence = min(1.0, self._confidence + CONFIDENCE_GAIN)
            # Establish/update reference scale for scale_z computation
            if obs.detected_scale_z is not None and self._detected_scale_ref is None:
                self._detected_scale_ref = obs.detected_scale_z

        # ── Generate candidate futures ────────────────────────────────────────
        ref = self._weighted_state or self._best_state or self._candidates[0]
        new_candidates = generate_candidates(ref, self.n_candidates, self._rng)

        # ── Score against observation ─────────────────────────────────────────
        score_all_candidates(new_candidates, obs)

        # ── Replace, prune, merge ─────────────────────────────────────────────
        self._candidates = new_candidates
        self._prune()
        self._merge()
        self._resample_if_needed()

        # ── Update derived estimates ──────────────────────────────────────────
        self._compute_estimates()
        self._update_validity(obs)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _bootstrap(self, obs: Observation) -> None:
        """Initialise the particle cloud from the first valid detection."""
        cx = obs.detected_center_x or 0.0
        cy = obs.detected_center_y or 0.0
        sz = obs.detected_scale_z or 1.0
        self._detected_scale_ref = sz
        macro_flow = obs.macro_flow_vector

        initial = HandState(
            center_x=cx, center_y=cy, scale_z=sz,
            velocity_x=float(macro_flow[0]) if macro_flow is not None else 0.0,
            velocity_y=float(macro_flow[1]) if macro_flow is not None else 0.0,
            confidence=obs.detection_confidence,
            weight=1.0,
            age=0,
        )
        self._candidates = generate_candidates(initial, self.n_candidates, self._rng)
        for c in self._candidates:
            c.weight = 1.0 / self.n_candidates
        self._confidence = obs.detection_confidence
        self._lost_frames = 0
        self._compute_estimates()
        self._validity_status = "INSUFFICIENT_DURATION"

    def _prune(self) -> None:
        """Remove candidates whose weight is negligibly small."""
        if not self._candidates:
            return
        threshold = PRUNE_WEIGHT_THRESHOLD
        alive = [c for c in self._candidates if c.weight >= threshold]
        if len(alive) < 4:
            alive = sorted(self._candidates, key=lambda c: -c.weight)[:max(4, len(self._candidates) // 4)]
        # Renormalise
        total = sum(c.weight for c in alive)
        if total > 0:
            for c in alive:
                c.weight /= total
        self._candidates = alive

    def _merge(self) -> None:
        """Merge near-duplicate candidates into one weighted state."""
        if len(self._candidates) < 2:
            return
        merged: list[HandState] = []
        used = [False] * len(self._candidates)
        # Sort by weight desc so high-weight states are chosen as cluster centres
        order = sorted(range(len(self._candidates)), key=lambda i: -self._candidates[i].weight)
        for i in order:
            if used[i]:
                continue
            cluster_states = [self._candidates[i]]
            total_w = self._candidates[i].weight
            for j in order:
                if i == j or used[j]:
                    continue
                ci = self._candidates[i]
                cj = self._candidates[j]
                d = math.hypot(ci.center_x - cj.center_x, ci.center_y - cj.center_y)
                if d < MERGE_DISTANCE_PX:
                    cluster_states.append(cj)
                    total_w += cj.weight
                    used[j] = True
            used[i] = True
            # Compute weighted centroid of cluster
            if len(cluster_states) == 1:
                merged.append(cluster_states[0])
            else:
                rep = cluster_states[0]
                wsum = total_w
                cx = sum(s.center_x * s.weight for s in cluster_states) / wsum
                cy = sum(s.center_y * s.weight for s in cluster_states) / wsum
                sz = sum(s.scale_z * s.weight for s in cluster_states) / wsum
                vx = sum(s.velocity_x * s.weight for s in cluster_states) / wsum
                vy = sum(s.velocity_y * s.weight for s in cluster_states) / wsum
                merged_state = HandState(
                    center_x=cx, center_y=cy, scale_z=sz,
                    velocity_x=vx, velocity_y=vy,
                    velocity_z=sum(s.velocity_z * s.weight for s in cluster_states) / wsum,
                    acceleration_x=rep.acceleration_x,
                    acceleration_y=rep.acceleration_y,
                    confidence=sum(s.confidence * s.weight for s in cluster_states) / wsum,
                    weight=total_w,
                    age=rep.age,
                )
                merged.append(merged_state)
        # Renormalise
        total = sum(c.weight for c in merged)
        if total > 0:
            for c in merged:
                c.weight /= total
        self._candidates = merged

    def _resample_if_needed(self) -> None:
        """Systematic resampling when particle diversity collapses."""
        n = len(self._candidates)
        if n >= self.n_candidates // 2:
            return  # plenty of diversity
        if not self._candidates:
            return
        ref = max(self._candidates, key=lambda c: c.weight)
        extras = generate_candidates(ref, self.n_candidates - n, self._rng)
        for e in extras:
            e.weight = 1.0 / (self.n_candidates * 4)  # low prior for new particles
        self._candidates.extend(extras)
        total = sum(c.weight for c in self._candidates)
        if total > 0:
            for c in self._candidates:
                c.weight /= total

    def _compute_estimates(self) -> None:
        """Compute best_state, weighted_state, and uncertainty."""
        if not self._candidates:
            return
        self._best_state = max(self._candidates, key=lambda c: c.weight)

        # Weighted mean
        total_w = sum(c.weight for c in self._candidates)
        if total_w <= 0:
            self._weighted_state = self._best_state
            self._state_uncertainty = 999.0
            return

        wmx = sum(c.center_x * c.weight for c in self._candidates) / total_w
        wmy = sum(c.center_y * c.weight for c in self._candidates) / total_w
        wmz = sum(c.scale_z  * c.weight for c in self._candidates) / total_w
        wvx = sum(c.velocity_x * c.weight for c in self._candidates) / total_w
        wvy = sum(c.velocity_y * c.weight for c in self._candidates) / total_w
        wvz = sum(c.velocity_z * c.weight for c in self._candidates) / total_w
        wax = sum(c.acceleration_x * c.weight for c in self._candidates) / total_w
        way = sum(c.acceleration_y * c.weight for c in self._candidates) / total_w
        wconf = sum(c.confidence * c.weight for c in self._candidates) / total_w

        self._weighted_state = HandState(
            center_x=wmx, center_y=wmy, scale_z=wmz,
            velocity_x=wvx, velocity_y=wvy, velocity_z=wvz,
            acceleration_x=wax, acceleration_y=way,
            confidence=wconf,
            weight=total_w,
            age=self._best_state.age,
        )

        # Uncertainty = weighted root-mean-square distance of candidates from the mean
        wvar = sum(
            c.weight * ((c.center_x - wmx) ** 2 + (c.center_y - wmy) ** 2)
            for c in self._candidates
        ) / total_w
        self._state_uncertainty = math.sqrt(max(0.0, wvar))

    def _update_validity(self, obs: Observation) -> None:
        """Set self._validity_status based on current landscape state.

        Emits only landscape-owned states. Tracking states (FLOW_LAG,
        RELATCHING) and session states (LOW_FPS, INSUFFICIENT_DURATION) are
        merged in by the bridge, which has the optical-flow + timing context.
        """
        if self._weighted_state is None:
            self._validity_status = "INSUFFICIENT_DURATION"
            return
        if self._lost_frames > LOST_TOLERANCE:
            self._validity_status = "HAND_LOST"
            return
        # Low confidence manifests as an uncertain / diffuse landscape.
        if self._state_uncertainty > UNCERTAINTY_DIFFUSE_THRESHOLD or self._confidence < 0.25:
            self._validity_status = "LANDSCAPE_DIFFUSE"
            return
        if obs.n_flow_points < 8:
            self._validity_status = "LOW_POINTS"
            return
        self._validity_status = "VALID"

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def debug(self) -> dict:
        ws = self._weighted_state
        bs = self._best_state
        return {
            "landscape_candidates": len(self._candidates),
            "landscape_uncertainty_px": round(self._state_uncertainty, 2),
            "landscape_confidence": round(self._confidence, 3),
            "landscape_lost_frames": self._lost_frames,
            "best_state": bs.to_dict() if bs else None,
            "weighted_state": ws.to_dict() if ws else None,
        }
