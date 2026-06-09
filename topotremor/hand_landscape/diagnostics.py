"""
hand_landscape/diagnostics.py

Builds the 3D landscape diagnostics payload emitted to the Electron renderer.

The payload is designed so the UI can render:

  - A 3D point cloud (candidate states)
  - A best/selected state peak
  - A weighted-mean state
  - Confidence surface data (state uncertainty)
  - Tremor energy surface
  - Validity status

The UI is explicitly NOT given bounding-box overlay instructions.
The data describes the state landscape; the renderer decides how to display it.

Output schema (subset of the "metrics" JSON):
{
  "validity_status": str,
  "live_score": int,             # 0–100
  "tremor_score": int,           # 0–100
  "tremor_power_ratio": float,   # 0–1
  "peak_hz": float,
  "band_rms": float,
  "h1_lifetime": float,
  "signal": [float, …],          # last 200 bandpassed residuals
  "embed_delay": int,
  "landscape": {
    "candidates": [              # up to 32 states for the cloud
      {"x": float, "y": float, "z": float, "weight": float, "confidence": float},
      …
    ],
    "selected_state": {x, y, z, vx, vy, vz, ax, ay, confidence, weight, age},
    "weighted_state": {…},
    "uncertainty_px": float,
    "n_candidates": int,
    "confidence": float,
    "residual_energy": float,    # mean(|residuals|²) this frame
    "tremor_energy": float,      # tremor_energy from tremor analysis [0, 1]
  },
  "macro_motion_score": int,     # 0–100
  "tracking_quality": str,       # "NN%"
  "confidence": str,             # "High" / "Medium" / "Low"
  "status_message": str,
  …
}
"""

from __future__ import annotations

import numpy as np

from .landscape import HandStateLandscape
from .state import HandState
from .dots import DEBUG_MODE

# Maximum number of candidate states to include in the serialised cloud
MAX_CLOUD_STATES = 32


def build_landscape_diagnostics(
    landscape: HandStateLandscape,
    tremor_result: dict,
    residual_xy: tuple[float, float],
    per_point_residuals: np.ndarray | None,
    n_flow_points: int,
    validity_status: str,
    tracking_debug: dict | None = None,
    dot_metrics: dict | None = None,
) -> tuple[dict, str]:
    """Build the full metrics + landscape payload.

    Parameters
    ----------
    landscape:
        The updated HandStateLandscape for this frame.
    tremor_result:
        Output of hand_landscape.tremor.run_tremor_analysis().
    residual_xy:
        (residual_x, residual_y) median residuals for this frame.
    per_point_residuals:
        (N, 2) array of per-point residuals, or None.
    n_flow_points:
        Number of tracked optical-flow points this frame.
    validity_status:
        String from landscape.validity_status (may be overridden by caller).
    dot_metrics:
        Output of DotFieldAnalyzer.analyze() — the per-dot micro-oscillator
        field. When present it is the AUTHORITATIVE tremor classifier; the
        aggregate tremor_result is retained only for residual / topology
        diagnostics.

    Returns
    -------
    metrics:
        Dict ready for JSON serialisation.
    status_message:
        Human-readable status string.
    """
    ws = landscape.weighted_state
    bs = landscape.best_state
    candidates = landscape.candidates
    dot_metrics = dot_metrics or {}

    # ── Candidate cloud (top MAX_CLOUD_STATES by weight) ──────────────────────
    sorted_cands = sorted(candidates, key=lambda c: -c.weight)[:MAX_CLOUD_STATES]
    cloud = [
        {
            "x": round(c.center_x, 1),
            "y": round(c.center_y, 1),
            "z": round(c.scale_z, 3),
            "weight": round(c.weight, 5),
            "confidence": round(c.confidence, 3),
        }
        for c in sorted_cands
    ]

    # ── Residual energy ───────────────────────────────────────────────────────
    if per_point_residuals is not None and len(per_point_residuals) > 0:
        residual_energy = float(np.mean(
            per_point_residuals[:, 0] ** 2 + per_point_residuals[:, 1] ** 2
        ))
    else:
        residual_energy = 0.0

    # ── Status message ────────────────────────────────────────────────────────
    peak_hz = tremor_result.get("peak_hz", 0.0)
    h1 = tremor_result.get("h1_lifetime", 0.0)
    band_rms = tremor_result.get("band_rms", 0.0)
    tremor_score = tremor_result.get("tremor_score", 0)

    # ── Per-dot field overrides the headline tremor classification ────────────
    # The per-dot micro-oscillator vote is the authority: it requires rhythmic
    # residual oscillation, in-band frequency, multi-dot coherence and tracking
    # validity. The aggregate spectrum is kept only for residual / H1 context.
    dot_score = None
    if dot_metrics:
        dot_score = int(dot_metrics.get("global_tremor_score", 0))
        g_freq = float(dot_metrics.get("global_tremor_frequency_hz", 0.0))
        g_amp = float(dot_metrics.get("global_tremor_amplitude", 0.0))
        if g_freq > 0:
            peak_hz = g_freq
        if g_amp > 0:
            band_rms = g_amp

    coherent_n = int(dot_metrics.get("coherent_dot_count", 0)) if dot_metrics else 0
    confirmed_n = int(dot_metrics.get("tremor_confirmed_count", 0)) if dot_metrics else 0
    valid_n = int(dot_metrics.get("valid_dot_count", 0)) if dot_metrics else 0
    field_validity = dot_metrics.get("field_validity", "OK") if dot_metrics else "OK"

    if validity_status == "HAND_LOST":
        status_msg = "Hand lost — show hand to camera"
    elif validity_status == "INSUFFICIENT_DURATION":
        status_msg = "Warming up — keep hand steady"
    elif validity_status == "LOW_FPS":
        status_msg = "Low frame rate — check camera"
    elif validity_status == "FLOW_LAG":
        status_msg = "Tracking lag — tremor not measured (move slower)"
    elif validity_status == "RELATCHING":
        status_msg = "Re-acquiring hand — tremor not measured"
    elif validity_status == "LOW_POINTS":
        status_msg = "Few tracked points — move to better lighting"
    elif validity_status == "LANDSCAPE_DIFFUSE":
        status_msg = "Tracking uncertain — hold hand still briefly"
    elif dot_metrics and field_validity == "SPARSE_FIELD":
        status_msg = f"Building dot field… ({valid_n} stable dots)"
    elif dot_score is not None and confirmed_n >= 1 and coherent_n >= 3:
        status_msg = f"Coherent tremor ~{peak_hz:.1f} Hz across {coherent_n} dots"
    elif dot_score is not None and dot_score >= 30:
        status_msg = f"Possible tremor ~{peak_hz:.1f} Hz ({coherent_n} coherent dots)"
    elif h1 > 0.4 and band_rms > 0.5:
        status_msg = f"Cyclic tremor ~{peak_hz:.1f} Hz (H1={h1:.2f})"
    elif band_rms > 0.3:
        status_msg = "Motion detected (non-cyclic)"
    else:
        status_msg = "Steady — minimal tremor-band motion"

    # ── Tremor trust gate ─────────────────────────────────────────────────────
    # Only a VALID frame yields a trusted tremor score. FLOW_LAG, RELATCHING,
    # LANDSCAPE_DIFFUSE, LOW_POINTS, etc. mean the residual is contaminated by
    # tracking artefacts, so the displayed tremor score is forced to 0 and
    # flagged untrusted. The raw value is preserved for debugging only.
    tracking_debug = tracking_debug or {}
    tremor_trusted = (validity_status == "VALID")
    if dot_score is not None:
        tremor_score_raw = dot_score
        tremor_confidence = float(dot_metrics.get("tremor_confidence", 0.0))
    else:
        tremor_score_raw = int(tremor_result.get("tremor_score", 0))
        tremor_confidence = float(tremor_result.get("tremor_confidence", 0.0))
    band_power_ratio = float(dot_metrics.get("global_band_power_ratio", tremor_result.get("tremor_power_ratio", 0.0)))
    peak_stability = float(dot_metrics.get("peak_stability", tremor_result.get("peak_stability", 0.0)))
    frequency_confidence = float(np.clip(max(tremor_confidence, peak_stability), 0.0, 1.0))
    coherent_dot_ratio = float(coherent_n / max(valid_n, 1))
    tremor_evidence = float(np.clip(band_power_ratio * frequency_confidence * coherent_dot_ratio, 0.0, 1.0))

    # Backward-compatible temporary score from measurement evidence when no
    # stable score exists yet.
    if tremor_score_raw <= 0:
        tremor_score_raw = int(round(100.0 * tremor_evidence))
    else:
        tremor_evidence = float(np.clip(tremor_score_raw / 100.0, 0.0, 1.0))

    # Displayed score is validity-gated; raw evidence is always emitted.
    displayed_tremor_score = tremor_score_raw if validity_status == "VALID" else 0
    tremor_score = displayed_tremor_score
    reveal = tremor_trusted or DEBUG_MODE

    # ── Confidence label (only meaningful when trusted) ───────────────────────
    conf_val = landscape.confidence
    if tremor_trusted and tremor_score >= 60 and h1 > 0.35:
        confidence_label = "High"
    elif tremor_trusted and (tremor_score >= 30 or band_rms > 0.3):
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # ── Scores ────────────────────────────────────────────────────────────────
    # macro_motion_score = how confidently we track the hand's bulk movement.
    macro_motion_score = int(min(100, conf_val * 100.0))
    # tracking_quality comes from the dedicated monitor (point survival, flow
    # lag, relatch rate) — NOT from anything that could be mistaken for tremor.
    if "tracking_quality_num" in tracking_debug:
        tracking_quality = int(round(tracking_debug["tracking_quality_num"]))
    else:
        tracking_quality = int(min(100, max(0, conf_val * 70.0 + min(n_flow_points * 2.5, 30.0))))

    # ── Landscape sub-dict ────────────────────────────────────────────────────
    landscape_dict = {
        "candidates": cloud,
        "selected_state": bs.to_dict() if bs else None,
        "weighted_state": ws.to_dict() if ws else None,
        "uncertainty_px": round(landscape.state_uncertainty, 2),
        "n_candidates": len(candidates),
        "confidence": round(conf_val, 3),
        "residual_energy": round(residual_energy, 4),
        "tremor_energy": tremor_result.get("tremor_energy", 0.0) if reveal else 0.0,
    }

    metrics = {
        # Core scores
        "live_score": tremor_score,
        "final_score": None,
        "tremor_score": tremor_score,
        "tremor_score_raw": tremor_score_raw,
        "displayed_tremor_score": displayed_tremor_score,
        "tremor_evidence": round(tremor_evidence, 4),
        "tremor_trusted": tremor_trusted,
        "tremor_confidence": round(tremor_confidence, 4),
        "macro_motion_score": macro_motion_score,
        "tracking_quality": f"{tracking_quality}%",
        "tracking_quality_num": tracking_quality,
        "confidence": confidence_label,
        # Tremor analysis
        "peak_hz": f"{peak_hz:.2f} Hz" if peak_hz else "-",
        "peak_hz_num": round(float(peak_hz), 2),
        "dominant_frequency_hz": round(float(peak_hz), 2),
        "band_ratio": f"{tremor_result.get('tremor_power_ratio', 0.0) * 100:.0f}%",
        "band_power_ratio": round(float(band_power_ratio), 4),
        "amp_mm": f"{band_rms:.2f} px",
        "residual_amplitude_px": round(float(band_rms), 4),
        "snr_db": "-",
        "h1_lifetime": f"{h1:.3f}",
        "power_ratio": f"{tremor_result.get('tremor_power_ratio', 0.0):.3f}",
        "embedding_delay": tremor_result.get("embedding_delay", 1),
        "live_score_num": tremor_score,
        # Residual diagnostics (the actual tremor evidence)
        "residual_rms": tremor_result.get("residual_rms", 0.0),
        "residual_peak_frequency_hz": tremor_result.get("residual_peak_frequency_hz", 0.0),
        "residual_band_power": tremor_result.get("residual_band_power", 0.0),
        "tremor_band_power": tremor_result.get("residual_band_power", 0.0),
        "peak_stability": dot_metrics.get("peak_stability", tremor_result.get("peak_stability", 0.0)),
        "spike_fraction": tremor_result.get("spike_fraction", 0.0),
        # ── Per-dot micro-oscillator field (authoritative tremor classifier) ──
        "global_tremor_amplitude": round(float(dot_metrics.get("global_tremor_amplitude", 0.0)), 4) if reveal else 0.0,
        "global_tremor_frequency_hz": round(float(dot_metrics.get("global_tremor_frequency_hz", 0.0)), 2) if reveal else 0.0,
        "dominant_tremor_frequency_hz": round(float(dot_metrics.get("dominant_tremor_frequency_hz", 0.0)), 2) if reveal else 0.0,
        "global_band_power_ratio": round(float(dot_metrics.get("global_band_power_ratio", 0.0)), 3),
        "median_dot_amplitude": round(float(dot_metrics.get("median_dot_amplitude", 0.0)), 4),
        "topk_dot_amplitude": round(float(dot_metrics.get("topk_dot_amplitude", 0.0)), 4) if reveal else 0.0,
        "coverage_factor": round(float(dot_metrics.get("coverage_factor", 0.0)), 3),
        "field_validity": dot_metrics.get("field_validity", "OK"),
        "valid_dot_count": int(dot_metrics.get("valid_dot_count", 0)),
        "warming_dot_count": int(dot_metrics.get("warming_dot_count", 0)),
        "positive_dot_count": int(dot_metrics.get("positive_dot_count", 0)),
        "coherent_dot_count": int(dot_metrics.get("coherent_dot_count", 0)),
        "tremor_confirmed_count": int(dot_metrics.get("tremor_confirmed_count", 0)),
        "noise_dot_count": int(dot_metrics.get("noise_dot_count", 0)),
        "relatching_dot_count": int(dot_metrics.get("relatching_dot_count", 0)),
        "total_dot_count": int(dot_metrics.get("total_dot_count", 0)),
        "min_valid_dots": int(dot_metrics.get("min_valid_dots", 0)),
        # ── Score decomposition (EVIDENCE vs TRUST — debug-visible) ───────────
        "debug_mode": bool(DEBUG_MODE),
        "displayed_tremor_score": displayed_tremor_score,
        "raw_evidence_score": int(dot_metrics.get("raw_evidence_score", 0)),
        "global_dot_score": round(float(dot_metrics.get("global_dot_score", 0.0)), 4),
        "trust_factor": round(float(dot_metrics.get("trust_factor", 0.0)), 4),
        "trust_score": int(round(100.0 * float(dot_metrics.get("trust_factor", 0.0)))),
        "coherence_factor": round(float(dot_metrics.get("coherence_factor", 0.0)), 3),
        "invalid_reason_counts": dot_metrics.get("invalid_reason_counts", {}),
        "p75_dot_amplitude": round(float(dot_metrics.get("p75_dot_amplitude", 0.0)), 4),
        "p90_dot_amplitude": round(float(dot_metrics.get("p90_dot_amplitude", 0.0)), 4),
        "max_dot_amplitude": round(float(dot_metrics.get("max_dot_amplitude", 0.0)), 4),
        "median_band_power_ratio": round(float(dot_metrics.get("median_band_power_ratio", 0.0)), 4),
        "median_prominence": round(float(dot_metrics.get("median_prominence", 0.0)), 4),
        "median_flatness": round(float(dot_metrics.get("median_flatness", 1.0)), 4),
        "stable_dot_count": int(dot_metrics.get("stable_dot_count", 0)),
        "candidate_dot_count": int(dot_metrics.get("candidate_dot_count", 0)),
        "tracking_noise_dot_count": int(dot_metrics.get("tracking_noise_dot_count", 0)),
        "region_tremor": dot_metrics.get("region_tremor", {}),
        "per_dot": dot_metrics.get("per_dot", []),

        # Tracking diagnostics (the gate — never tremor)
        "flow_lag_px": tracking_debug.get("flow_lag_px", 0.0),
        "relatch_rate": tracking_debug.get("relatch_rate", 0.0),
        "point_survival": tracking_debug.get("point_survival", 0.0),
        "tracking_state": tracking_debug.get("tracking_state", "UNKNOWN"),
        # Signal for phase-space rendering
        "signal": tremor_result.get("filtered", []),
        "embed_delay": tremor_result.get("embedding_delay", 1),
        # Validity
        "validity_status": validity_status,
        "validity_issues": _validity_issues(landscape, n_flow_points, tracking_debug),
        # 3D landscape data
        "landscape": landscape_dict,
        # Residual
        "residual_x": round(residual_xy[0], 4),
        "residual_y": round(residual_xy[1], 4),
    }
    return metrics, status_msg


def build_waiting_diagnostics(
    landscape: HandStateLandscape | None,
    validity_status: str,
    status_msg: str = "Show your hand to the camera",
    tracking_debug: dict | None = None,
) -> tuple[dict, str]:
    """Build a placeholder metrics dict when no tremor analysis is possible."""
    ws = landscape.weighted_state if landscape else None
    bs = landscape.best_state if landscape else None
    candidates = landscape.candidates if landscape else []

    sorted_cands = sorted(candidates, key=lambda c: -c.weight)[:MAX_CLOUD_STATES]
    cloud = [
        {"x": round(c.center_x, 1), "y": round(c.center_y, 1),
         "z": round(c.scale_z, 3), "weight": round(c.weight, 5),
         "confidence": round(c.confidence, 3)}
        for c in sorted_cands
    ]

    landscape_dict = {
        "candidates": cloud,
        "selected_state": bs.to_dict() if bs else None,
        "weighted_state": ws.to_dict() if ws else None,
        "uncertainty_px": round(landscape.state_uncertainty, 2) if landscape else 0.0,
        "n_candidates": len(candidates),
        "confidence": round(landscape.confidence, 3) if landscape else 0.0,
        "residual_energy": 0.0,
        "tremor_energy": 0.0,
    }

    metrics = {
        "live_score": 0,
        "final_score": None,
        "tremor_score": 0,
        "tremor_score_raw": 0,
        "displayed_tremor_score": 0,
        "tremor_evidence": 0.0,
        "tremor_trusted": False,
        "tremor_confidence": 0.0,
        "macro_motion_score": 0,
        "tracking_quality": f"{int(round((tracking_debug or {}).get('tracking_quality_num', 0.0)))}%",
        "tracking_quality_num": int(round((tracking_debug or {}).get('tracking_quality_num', 0.0))),
        "confidence": "Low",
        "peak_hz": "-",
        "peak_hz_num": 0.0,
        "dominant_frequency_hz": 0.0,
        "band_ratio": "0%",
        "band_power_ratio": 0.0,
        "amp_mm": "0.00 px",
        "residual_amplitude_px": 0.0,
        "snr_db": "-",
        "h1_lifetime": "0.000",
        "power_ratio": "0.000",
        "embedding_delay": 1,
        "live_score_num": 0,
        "residual_rms": 0.0,
        "residual_peak_frequency_hz": 0.0,
        "residual_band_power": 0.0,
        "tremor_band_power": 0.0,
        "peak_stability": 0.0,
        "spike_fraction": 0.0,
        "global_tremor_amplitude": 0.0,
        "global_tremor_frequency_hz": 0.0,
        "dominant_tremor_frequency_hz": 0.0,
        "global_band_power_ratio": 0.0,
        "median_dot_amplitude": 0.0,
        "topk_dot_amplitude": 0.0,
        "coverage_factor": 0.0,
        "field_validity": "SPARSE_FIELD",
        "valid_dot_count": 0,
        "warming_dot_count": 0,
        "positive_dot_count": 0,
        "coherent_dot_count": 0,
        "tremor_confirmed_count": 0,
        "noise_dot_count": 0,
        "relatching_dot_count": 0,
        "total_dot_count": 0,
        "min_valid_dots": 0,
        "debug_mode": bool(DEBUG_MODE),
        "raw_evidence_score": 0,
        "global_dot_score": 0.0,
        "trust_factor": 0.0,
        "trust_score": 0,
        "coherence_factor": 0.0,
        "invalid_reason_counts": {},
        "p75_dot_amplitude": 0.0,
        "p90_dot_amplitude": 0.0,
        "max_dot_amplitude": 0.0,
        "median_band_power_ratio": 0.0,
        "median_prominence": 0.0,
        "median_flatness": 1.0,
        "stable_dot_count": 0,
        "candidate_dot_count": 0,
        "tracking_noise_dot_count": 0,
        "region_tremor": {},
        "per_dot": [],
        "flow_lag_px": (tracking_debug or {}).get("flow_lag_px", 0.0),
        "relatch_rate": (tracking_debug or {}).get("relatch_rate", 0.0),
        "point_survival": (tracking_debug or {}).get("point_survival", 0.0),
        "tracking_state": (tracking_debug or {}).get("tracking_state", "UNKNOWN"),
        "signal": [],
        "embed_delay": 1,
        "validity_status": validity_status,
        "validity_issues": [],
        "landscape": landscape_dict,
        "residual_x": 0.0,
        "residual_y": 0.0,
    }
    return metrics, status_msg


def _validity_issues(landscape: HandStateLandscape, n_flow_points: int,
                     tracking_debug: dict | None = None) -> list[str]:
    issues = []
    if landscape.confidence < 0.3:
        issues.append("low_landscape_confidence")
    if landscape.state_uncertainty > 40.0:
        issues.append("diffuse_landscape")
    if n_flow_points < 8:
        issues.append("insufficient_texture")
    td = tracking_debug or {}
    state = td.get("tracking_state")
    if state == "FLOW_LAG":
        issues.append("flow_lag")
    elif state == "RELATCHING":
        issues.append("relatching")
    if td.get("relatch_rate", 0.0) > 0.3:
        issues.append("high_relatch_rate")
    return issues
