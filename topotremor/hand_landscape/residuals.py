"""
hand_landscape/residuals.py

Extracts the tremor residual signal from per-frame optical flow.

The key insight:

    observed_delta[i]  = where tracked point i actually moved this frame
    macro_delta        = where the WEIGHTED hand state moved (model prediction)
    residual_delta[i]  = observed_delta[i] - macro_delta

`residual_delta` isolates micro-oscillations — the non-macro component of
hand motion — which is where physiological tremor lives.

We take the *median* of residuals across all valid tracked points.
The median is robust to individual point track failures, landmark drift,
and finger-interior sliding that would inflate the mean.

Both x and y channels are returned. The tremor analysis typically uses the
x-channel (largest signal for essential tremor), but the y-channel is also
available for direction-specific analysis.

Reference frame
---------------
The macro_delta is computed in the same pixel coordinate system as the
optical-flow displacements, so no unit conversion is needed:

    macro_delta = (weighted_state.center_x - prev_weighted_state.center_x,
                   weighted_state.center_y - prev_weighted_state.center_y)
"""

from __future__ import annotations

import numpy as np

from .state import HandState


def compute_residuals(
    optical_flow_deltas: np.ndarray,
    weighted_state: HandState,
    prev_weighted_state: HandState | None,
) -> tuple[float, float, np.ndarray]:
    """Compute per-point and summary residual micro-motion.

    Parameters
    ----------
    optical_flow_deltas:
        Shape (N, 2). The per-point (new_pos - old_pos) displacement for
        this frame from Lucas-Kanade optical flow.
    weighted_state:
        The current frame's weighted hand state.
    prev_weighted_state:
        The *previous* frame's weighted hand state.  If None (first frame),
        the macro_delta is taken from the state's own velocity estimate.

    Returns
    -------
    residual_x:
        Median residual x-component across all valid points (px/frame).
    residual_y:
        Median residual y-component across all valid points (px/frame).
    per_point_residuals:
        Shape (N, 2) – per-point [residual_x, residual_y]. Useful for
        diagnostics and heatmap generation.
    """
    if optical_flow_deltas is None or len(optical_flow_deltas) == 0:
        return 0.0, 0.0, np.zeros((0, 2), dtype=np.float64)

    deltas = np.asarray(optical_flow_deltas, dtype=np.float64)

    # Macro delta: how much did the weighted state centre move this frame?
    if prev_weighted_state is not None:
        macro_x = weighted_state.center_x - prev_weighted_state.center_x
        macro_y = weighted_state.center_y - prev_weighted_state.center_y
    else:
        # First frame after bootstrap: use velocity as proxy
        macro_x = weighted_state.velocity_x
        macro_y = weighted_state.velocity_y

    macro_delta = np.array([macro_x, macro_y], dtype=np.float64)

    # Residual = observed point motion – macro model motion
    per_point = deltas - macro_delta[np.newaxis, :]

    # Median across points (robust estimator)
    residual_x = float(np.median(per_point[:, 0]))
    residual_y = float(np.median(per_point[:, 1]))

    return residual_x, residual_y, per_point
