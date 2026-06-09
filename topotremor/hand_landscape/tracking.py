"""
hand_landscape/tracking.py

Tracking-quality monitor — the GATE, never the tremor source.

Why this module exists
-----------------------
Sparse optical-flow points are *reactive*: when the hand moves, the feature
points latch onto texture slightly late, then "catch up" a frame later. That
catch-up shows up in the per-point displacement as a coherent backward bias
followed by a forward correction — i.e. a fake oscillation at the frequency of
the hand's macro acceleration. If that residual is fed to the tremor scorer it
is misclassified as biological tremor.

The fix: point behaviour must only affect *confidence / validity*, never the
tremor score. This monitor turns raw point behaviour into:

    flow_lag_px        – how far the flow median lags the landscape macro motion
    relatch_rate       – fraction of recent frames that forced a full re-seed
    point_survival     – median tracked-point count, normalised
    tracking_quality   – [0, 1] gate applied to tremor confidence
    tracking_state     – VALID / FLOW_LAG / RELATCHING / LOW_POINTS

Key idea
--------
The MediaPipe box centre is a *per-frame* detection: it does NOT lag. The
landscape's weighted state is anchored to that box centre, so the landscape
macro delta is essentially lag-free. The optical-flow median, by contrast,
lags. Therefore:

    flow_lag = | landscape_macro_delta  -  flow_median_delta |

A large, sustained flow_lag means the points cannot keep up → FLOW_LAG → the
tremor residual is contaminated and must NOT be trusted.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ─── thresholds ───────────────────────────────────────────────────────────────
LAG_SUSPECT_PX      = 2.5    # per-frame flow lag above this is "suspect"
LAG_FAIL_PX         = 4.5    # sustained flow lag above this → FLOW_LAG
RELATCH_FAIL_RATE   = 0.30   # >30% of frames re-seeding → RELATCHING
LOW_POINTS_THRESH   = 12     # below this median point count → LOW_POINTS
GOOD_POINTS_TARGET  = 40     # point count that saturates the survival term
LAG_QUALITY_MAX_PX  = 6.0    # flow lag that drives the lag-quality term to 0
HISTORY_FRAMES      = 24     # rolling window (~2 s at 12 fps preview cadence)


class TrackingQualityMonitor:
    """Stateful monitor converting point behaviour into a confidence gate."""

    def __init__(self, history: int = HISTORY_FRAMES) -> None:
        self._lag_hist: deque = deque(maxlen=history)
        self._relatch_hist: deque = deque(maxlen=history)
        self._points_hist: deque = deque(maxlen=history)

        self._flow_lag_px: float = 0.0
        self._relatch_rate: float = 0.0
        self._point_survival: float = 0.0
        self._tracking_quality: float = 0.0
        self._tracking_state: str = "LOW_POINTS"

    # ------------------------------------------------------------------ #
    # Update                                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        macro_delta: np.ndarray | None,
        flow_median_delta: np.ndarray | None,
        n_points: int,
        relatched: bool,
    ) -> None:
        """Ingest one frame of tracking evidence.

        Parameters
        ----------
        macro_delta:
            (2,) landscape weighted-state centre delta this frame (lag-free).
        flow_median_delta:
            (2,) median optical-flow displacement this frame (laggy).
        n_points:
            Number of surviving tracked points this frame.
        relatched:
            True if this frame forced a full feature re-seed (discontinuity).
        """
        # Per-frame flow lag = how far the flow median trails the macro motion.
        if macro_delta is not None and flow_median_delta is not None:
            lag_vec = np.asarray(macro_delta, dtype=np.float64) - np.asarray(
                flow_median_delta, dtype=np.float64
            )
            lag = float(np.hypot(lag_vec[0], lag_vec[1]))
        else:
            lag = 0.0

        self._lag_hist.append(lag)
        self._relatch_hist.append(1.0 if relatched else 0.0)
        self._points_hist.append(int(n_points))

        # Smoothed, outlier-robust aggregates over the rolling window.
        self._flow_lag_px = float(np.median(self._lag_hist)) if self._lag_hist else 0.0
        self._relatch_rate = float(np.mean(self._relatch_hist)) if self._relatch_hist else 0.0
        median_points = float(np.median(self._points_hist)) if self._points_hist else 0.0
        self._point_survival = min(1.0, median_points / GOOD_POINTS_TARGET)

        self._recompute_quality(median_points)

    def _recompute_quality(self, median_points: float) -> None:
        # Component qualities, each in [0, 1].
        q_lag = max(0.0, 1.0 - self._flow_lag_px / LAG_QUALITY_MAX_PX)
        q_relatch = max(0.0, 1.0 - self._relatch_rate / 0.5)  # 50% reseed → 0
        q_points = self._point_survival

        self._tracking_quality = float(
            0.40 * q_lag + 0.30 * q_relatch + 0.30 * q_points
        )

        # State machine: most-severe tracking problem wins.
        if median_points < LOW_POINTS_THRESH:
            self._tracking_state = "LOW_POINTS"
        elif self._relatch_rate > RELATCH_FAIL_RATE:
            self._tracking_state = "RELATCHING"
        elif self._flow_lag_px > LAG_FAIL_PX:
            self._tracking_state = "FLOW_LAG"
        else:
            self._tracking_state = "VALID"

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    @property
    def flow_lag_px(self) -> float:
        return self._flow_lag_px

    @property
    def relatch_rate(self) -> float:
        return self._relatch_rate

    @property
    def point_survival(self) -> float:
        return self._point_survival

    @property
    def tracking_quality(self) -> float:
        """[0, 1] gate applied to tremor confidence."""
        return self._tracking_quality

    @property
    def tracking_state(self) -> str:
        return self._tracking_state

    @property
    def is_lagging(self) -> bool:
        return self._tracking_state == "FLOW_LAG"

    @property
    def is_relatching(self) -> bool:
        return self._tracking_state == "RELATCHING"

    @property
    def suppresses_tremor(self) -> bool:
        """True when tracking problems make the tremor residual untrustworthy."""
        return self._tracking_state in ("FLOW_LAG", "RELATCHING")

    def debug(self) -> dict:
        return {
            "tracking_quality_num": round(self._tracking_quality * 100.0, 1),
            "flow_lag_px": round(self._flow_lag_px, 2),
            "relatch_rate": round(self._relatch_rate, 3),
            "point_survival": round(self._point_survival, 3),
            "tracking_state": self._tracking_state,
        }
