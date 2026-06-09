"""
Headless tremor analysis engine.

Extracts the full analysis pipeline from `motionbloom.app.App._refresh_analysis`
into a reusable class so both the legacy Tkinter UI and the new PyQt6 UI
can use IDENTICAL backend logic.

Inputs:  TremorTracker with active capture.
Outputs: AnalysisResult dataclass with every field the UI needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .signal import (
    PALM_DRIFT_RATIO_THRESHOLD,
    PALM_STEADY_GROSS_RATIO,
    PALM_STEADY_FRAME_MOTION,
    PALM_UNCERTAIN_GROSS_RATIO,
    AdaptiveBaseline,
    TaskMode,
    assess_trial_quality,
    compute_metrics,
    compute_palm_center_motion_gate,
    movement_residual_features,
    resample_uniform,
)
from .tracker import CAM_HEIGHT, CAM_WIDTH


# Match legacy app.py constants
WINDOW_SECONDS = 1.5
FLOW_MAX_PER_FRAME_DISPLACEMENT = 0.08
FLOW_GROSS_FRAME_FRACTION_MAX = 0.20
FLOW_MIN_QUALITY = 0.50
FLOW_MIN_VALID_POINTS = 8

HAND_SETTLING_SECONDS = 1.0
HAND_MOVING_WARNING_SECONDS = 1.5

HAND_STATE_NO_HAND = "NO_HAND"
HAND_STATE_MOVING = "HAND_MOVING"
HAND_STATE_MOVING_WIDE = "HAND_MOVING_WIDE_RANGE"
HAND_STATE_DRIFTING = "HAND_DRIFTING"
HAND_STATE_SETTLING = "HAND_SETTLING"
HAND_STATE_STEADY = "HAND_STEADY"
HAND_STATE_ACTIVE = "TREMOR_ANALYSIS_ACTIVE"

HAND_MOVING_STATES = {HAND_STATE_MOVING, HAND_STATE_MOVING_WIDE, HAND_STATE_DRIFTING}


def gate_flow_for_microtremor(body_dx, body_dy, *,
                              max_per_frame=FLOW_MAX_PER_FRAME_DISPLACEMENT,
                              gross_fraction_max=FLOW_GROSS_FRAME_FRACTION_MAX):
    """Reject flow windows containing physiologically impossible motion."""
    dx = np.asarray(body_dx, dtype=np.float64)
    dy = np.asarray(body_dy, dtype=np.float64)
    if dx.size == 0 or dy.size == 0:
        return False, "no flow samples", 0.0, dx, dy
    mag = np.hypot(dx, dy)
    gross_mask = mag > max_per_frame
    gross_fraction = float(gross_mask.mean())
    clip = float(max_per_frame)
    cdx = np.clip(dx, -clip, clip)
    cdy = np.clip(dy, -clip, clip)
    if gross_fraction > gross_fraction_max:
        return (False, f"Too much hand movement ({int(gross_fraction*100)}% gross)",
                gross_fraction, cdx, cdy)
    return True, "flow within physiological range", gross_fraction, cdx, cdy


def select_tracking_source(flow_snapshot, flow_status, palm_relative_available, *,
                            min_quality=FLOW_MIN_QUALITY,
                            min_valid_points=FLOW_MIN_VALID_POINTS):
    """Pick the live tremor signal source."""
    if flow_snapshot is None or flow_status is None:
        if palm_relative_available:
            return "mediapipe_palm_body", "Optical flow warming up"
        return "fallback", "Waiting for tracking"
    usable = bool(flow_status.get("usable", False))
    quality = float(flow_status.get("flow_quality", 0.0))
    pts = int(flow_status.get("valid_points", 0))
    if usable and quality >= min_quality and pts >= min_valid_points:
        return "optical_flow", f"Optical flow (q={quality:.2f}, {pts} pts)"
    if palm_relative_available:
        return "mediapipe_palm_body", f"Landmark fallback (q={quality:.2f})"
    return "fallback", "No usable tracking"


def classify_hand_motion_state(palm_gate, now, previous_state,
                               previous_state_since, steady_since, moving_since):
    """Advance palm-center motion state machine."""
    if palm_gate is None:
        new_state = HAND_STATE_NO_HAND
        state_since = previous_state_since if previous_state == new_state else now
        return {"state": new_state, "state_since": state_since,
                "steady_since": None, "moving_since": None,
                "analysis_active": False, "message": "No hand detected."}

    gross_ratio = float(palm_gate.get("gross_motion_ratio", 0.0))
    velocity_p95 = float(palm_gate.get("per_frame_velocity_p95", 0.0))
    drift_ratio = float(palm_gate.get("drift_ratio", 0.0))
    gate_state = str(palm_gate.get("state", "unknown"))

    if gate_state == "unknown":
        new_state = HAND_STATE_SETTLING
        new_steady_since = steady_since if steady_since is not None else now
        message = "Hold your hand still. Getting ready."
    elif gross_ratio > PALM_UNCERTAIN_GROSS_RATIO:
        new_state = HAND_STATE_MOVING_WIDE
        new_steady_since = None
        message = "Too much hand movement."
    elif velocity_p95 > PALM_STEADY_FRAME_MOTION or gross_ratio >= PALM_STEADY_GROSS_RATIO:
        new_state = HAND_STATE_MOVING
        new_steady_since = None
        message = "Your hand is moving."
    elif drift_ratio > PALM_DRIFT_RATIO_THRESHOLD:
        new_state = HAND_STATE_DRIFTING
        new_steady_since = None
        message = "Your hand is slowly drifting."
    else:
        if (previous_state in HAND_MOVING_STATES
                or previous_state == HAND_STATE_NO_HAND
                or steady_since is None):
            new_steady_since = now
        else:
            new_steady_since = steady_since
        steady_duration = now - new_steady_since
        if steady_duration < HAND_SETTLING_SECONDS:
            new_state = HAND_STATE_SETTLING
            remaining = max(0.0, HAND_SETTLING_SECONDS - steady_duration)
            message = f"Hold steady. Reading in {remaining:.1f}s."
        else:
            new_state = HAND_STATE_ACTIVE
            message = "Hand is still. Reading is ready."

    state_since = previous_state_since if previous_state == new_state else now
    if new_state in HAND_MOVING_STATES:
        if previous_state in HAND_MOVING_STATES and moving_since is not None:
            new_moving_since = moving_since
        else:
            new_moving_since = now
        if now - new_moving_since >= HAND_MOVING_WARNING_SECONDS:
            message = "Too much movement. Hold still."
    else:
        new_moving_since = None

    return {"state": new_state, "state_since": state_since,
            "steady_since": new_steady_since, "moving_since": new_moving_since,
            "analysis_active": new_state == HAND_STATE_ACTIVE,
            "message": message}


@dataclass
class AnalysisResult:
    """Everything a UI needs to display one tick of analysis."""
    # Top-line
    live_score: int = 0                      # ALWAYS shown (0-100)
    final_tremor_score: Optional[int] = None # Only when research_valid
    candidate_score: int = 0                 # What score would be

    # Status
    status_message: str = "Place your hand in view."
    verdict_message: str = ""
    verdict_type: str = "info"               # info|success|warning|error
    research_valid: bool = False
    confidence_level: str = "low"            # low|medium|high

    # Spectral
    peak_hz: float = 0.0
    band_ratio: float = 0.0
    rms_amp: float = 0.0
    rms_amp_mm: float = 0.0
    snr_db: float = 0.0
    regularity: float = 0.0
    peak_sharpness: float = 0.0
    peak_prominence: float = 0.0

    # Quality / motion
    quality_status: str = "invalid"
    motion_classification: str = "unknown"
    fps_cv: float = 0.0
    velocity_p95: float = 0.0
    path_ratio: float = 0.0
    center_drift: float = 0.0
    tracking_quality_pct: int = 0
    palm_motion_label: str = "-"
    palm_gross_motion_ratio: float = 0.0

    # Source
    tracking_source: str = "fallback"
    tracking_source_label: str = "Warming up"
    flow_status_label: str = "Flow: warming up"
    mode_label: str = "-"                    # Palm-Relative / Box / Multi / Single / Optical-Flow

    # Stream stats
    fs: float = 0.0
    samples: int = 0
    fps_meas: float = 0.0

    # Reason (when not valid)
    reason: str = ""
    hand_state: str = HAND_STATE_NO_HAND

    # Buffering progress (when no snap yet)
    buffering: bool = False
    buffer_progress: tuple[int, int] = (0, 0)  # (have, need)


class TremorAnalysisEngine:
    """Stateful analysis engine — one instance per session.

    Mirrors `App._refresh_analysis` exactly. Call `tick(tracker)` periodically
    (e.g. every 200ms) to get an updated `AnalysisResult`.
    """

    def __init__(self, task_mode: str = TaskMode.POSTURAL_GENERAL):
        self.task_mode = task_mode
        self.baseline = AdaptiveBaseline()
        self.baseline_rms: Optional[float] = None

        # Hand motion state machine
        now = time.time()
        self.hand_motion_state = HAND_STATE_NO_HAND
        self.hand_motion_state_since = now
        self.hand_steady_since: Optional[float] = None
        self.hand_moving_since: Optional[float] = None

        # History
        self.last_stable_score: Optional[int] = None
        self.last_metrics = None

    def reset(self) -> None:
        now = time.time()
        self.hand_motion_state = HAND_STATE_NO_HAND
        self.hand_motion_state_since = now
        self.hand_steady_since = None
        self.hand_moving_since = None
        self.last_stable_score = None
        self.last_metrics = None

    def _advance_state(self, palm_gate, now):
        st = classify_hand_motion_state(
            palm_gate, now, self.hand_motion_state,
            self.hand_motion_state_since, self.hand_steady_since,
            self.hand_moving_since,
        )
        self.hand_motion_state = st["state"]
        self.hand_motion_state_since = st["state_since"]
        self.hand_steady_since = st["steady_since"]
        self.hand_moving_since = st["moving_since"]
        return st

    def tick(self, tracker) -> AnalysisResult:
        """Run one analysis tick. Returns an AnalysisResult."""
        result = AnalysisResult()
        result.fps_meas = getattr(tracker, "fps_meas", 0.0) or 0.0

        # === Snapshots from tracker ===
        snap_box = tracker.snapshot_box_micro(WINDOW_SECONDS)
        snap_palm = tracker.snapshot_palm_center(WINDOW_SECONDS)
        snap_relative = tracker.snapshot_palm_relative(WINDOW_SECONDS)
        snap_flow = tracker.snapshot_flow(WINDOW_SECONDS)
        try:
            flow_status = tracker.get_flow_status()
        except Exception:
            flow_status = None

        # Tracking source decision
        tracking_source, tracking_reason = select_tracking_source(
            snap_flow, flow_status,
            palm_relative_available=snap_relative is not None,
        )
        result.tracking_source = tracking_source
        result.tracking_source_label = tracking_reason
        if flow_status is not None:
            result.flow_status_label = (
                f"Flow q={flow_status.get('flow_quality', 0.0):.2f}  "
                f"pts={int(flow_status.get('valid_points', 0))}  "
                f"surv={flow_status.get('survival_rate', 0.0):.2f}"
            )

        # Palm gate
        palm_gate = None
        if snap_palm is not None:
            _pt, raw_palm_x, raw_palm_y, _pw, _ph, raw_hand_box_size, _pq = snap_palm
            palm_gate = compute_palm_center_motion_gate(
                raw_palm_x, raw_palm_y, raw_hand_box_size,
                frame_width=CAM_WIDTH, frame_height=CAM_HEIGHT,
                hand_box_width=_pw, hand_box_height=_ph,
            )
        hand_state = self._advance_state(palm_gate, time.time())
        result.hand_state = hand_state["state"]
        result.status_message = hand_state["message"]

        # Pick a signal source
        snap_multi = tracker.snapshot_multi_finger(WINDOW_SECONDS)
        use_palm_relative = snap_relative is not None
        use_box_micro = snap_box is not None and not use_palm_relative
        use_multi_finger = (snap_multi is not None
                            and not use_palm_relative and not use_box_micro)

        relative_count = len(getattr(tracker, "palm_relative_samples", []))
        movement_count = len(getattr(tracker, "box_micro_samples", []))
        multi_count = len(getattr(tracker, "multi_finger_samples", []))
        single_count = len(getattr(tracker, "samples", []))

        local_x = local_y = global_x = global_y = None
        box_width = box_height = box_area = None
        relative_fingertip_signals = None
        raw_fingertip_power = 0.0
        raw_primary_x = raw_primary_y = None
        confidence = None
        tracking_quality_value = None
        ref = None
        t = None
        x = y = None
        mode_str = "-"

        if use_palm_relative:
            (t, ix_rel, iy_rel, mid_rel_x, mid_rel_y, ring_rel_x, ring_rel_y,
             thumb_rel_x, thumb_rel_y, pinky_rel_x, pinky_rel_y,
             index_mcp_rel_x, index_mcp_rel_y,
             middle_mcp_rel_x, middle_mcp_rel_y,
             rel_palm_x, rel_palm_y, rel_hand_size, quality) = snap_relative
            x = (ix_rel + mid_rel_x + ring_rel_x) / 3.0
            y = (iy_rel + mid_rel_y + ring_rel_y) / 3.0
            global_x = rel_palm_x; global_y = rel_palm_y
            local_x = x; local_y = y
            confidence = quality
            tracking_quality_value = (
                float(np.median(quality)) if len(quality) else 1.0
            )
            ref = rel_hand_size
            raw_index_x = ix_rel * rel_hand_size + rel_palm_x
            raw_index_y = iy_rel * rel_hand_size + rel_palm_y
            raw_middle_x = mid_rel_x * rel_hand_size + rel_palm_x
            raw_middle_y = mid_rel_y * rel_hand_size + rel_palm_y
            raw_ring_x = ring_rel_x * rel_hand_size + rel_palm_x
            raw_ring_y = ring_rel_y * rel_hand_size + rel_palm_y
            raw_primary_x = (raw_index_x + raw_middle_x + raw_ring_x) / 3.0
            raw_primary_y = (raw_index_y + raw_middle_y + raw_ring_y) / 3.0
            mode_str = "Palm-Relative"
        elif use_box_micro:
            (t, local_x, local_y, global_x, global_y,
             box_width, box_height, box_area, quality) = snap_box
            x = global_x; y = global_y
            confidence = quality
            tracking_quality_value = (
                float(np.median(quality)) if len(quality) else 1.0
            )
            ref = box_width
            mode_str = "Box-Normalized"
        elif use_multi_finger:
            t, ix, iy, mx, my, rx, ry, ref, conf = snap_multi
            x = (ix + mx + rx) / 3.0
            y = (iy + my + ry) / 3.0
            confidence = conf
            mode_str = "Multi-Finger"
        else:
            snap = tracker.snapshot(WINDOW_SECONDS)
            if snap is None:
                required = max(16, int(WINDOW_SECONDS * 25))
                have = max(relative_count, movement_count, multi_count, single_count)
                result.buffering = True
                result.buffer_progress = (have, required)
                result.status_message = (
                    f"Hold your hand still. Getting ready ({have}/{required})."
                )
                result.verdict_message = result.status_message
                result.verdict_type = "warning"
                result.mode_label = "buffering"
                if self.last_stable_score is not None:
                    result.live_score = self.last_stable_score
                return result
            t, x, y, ref = snap
            confidence = None
            mode_str = "Single-Landmark"

        # Optical flow override
        if tracking_source == "optical_flow" and snap_flow is not None:
            (ft, fdx, fdy, fpts, fsurv, fq, fbgx, fbgy) = snap_flow
            allow_flow, gate_reason, gross_fraction, fdx, fdy = \
                gate_flow_for_microtremor(fdx, fdy)
            result.flow_status_label += f"  gross={int(gross_fraction*100)}%"
            if not allow_flow:
                result.status_message = gate_reason
                result.verdict_message = gate_reason
                result.verdict_type = "warning"
                result.mode_label = "paused"
                result.hand_state = HAND_STATE_MOVING_WIDE
                if self.last_stable_score is not None:
                    result.live_score = self.last_stable_score
                return result
            t_old = np.asarray(t, dtype=np.float64)
            t = ft
            x = np.cumsum(fdx); y = np.cumsum(fdy)
            confidence = fq
            mode_str = "Optical-Flow"
            if use_palm_relative:
                if local_x is not None and local_y is not None:
                    local_x = np.interp(t, t_old, np.asarray(local_x))
                    local_y = np.interp(t, t_old, np.asarray(local_y))
                if global_x is not None and global_y is not None:
                    global_x = np.interp(t, t_old, np.asarray(global_x))
                    global_y = np.interp(t, t_old, np.asarray(global_y))
                if isinstance(ref, np.ndarray) and ref.shape == t_old.shape:
                    ref = np.interp(t, t_old, ref)
            if tracking_quality_value is None and len(fq):
                tracking_quality_value = float(np.median(fq))

        result.mode_label = mode_str

        # Trial quality
        try:
            quality_result = assess_trial_quality(t, confidence)
        except Exception:
            quality_result = {"quality_status": "invalid", "quality_score": 0.0,
                              "fps_cv": 0.0, "reasons": []}
        quality_status = quality_result.get("quality_status", "invalid")
        fps_cv = quality_result.get("fps_cv", 0.0)
        reasons = quality_result.get("reasons", [])
        result.quality_status = quality_status
        result.fps_cv = fps_cv

        dur = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
        if dur <= 0.5:
            result.reason = "Need a bit more data."
            return result
        fs_est = float(np.clip(len(t) / dur, 15.0, 60.0))
        res = resample_uniform(t, x, y, fs_est)
        if res is None:
            result.reason = "Tracking gaps. Hold steady."
            return result
        tu, xu, yu = res

        local_xu = local_yu = global_xu = global_yu = None
        box_width_u = box_height_u = box_area_u = None
        palm_xu = palm_yu = palm_w_u = palm_h_u = hand_box_size_u = None

        if use_palm_relative:
            local_res = resample_uniform(t, local_x, local_y, fs_est)
            global_res = resample_uniform(t, global_x, global_y, fs_est)
            if local_res is None or global_res is None:
                result.reason = "Tracking gaps."
                return result
            _, local_xu, local_yu = local_res
            _, global_xu, global_yu = global_res
            try:
                relative_fingertip_signals = {
                    "index_tip": (np.interp(tu, t, ix_rel), np.interp(tu, t, iy_rel)),
                    "middle_tip": (np.interp(tu, t, mid_rel_x), np.interp(tu, t, mid_rel_y)),
                    "ring_tip": (np.interp(tu, t, ring_rel_x), np.interp(tu, t, ring_rel_y)),
                }
            except Exception:
                pass
            if raw_primary_x is not None and raw_primary_y is not None:
                try:
                    raw_xu = np.interp(tu, t, raw_primary_x)
                    raw_yu = np.interp(tu, t, raw_primary_y)
                    raw_feats = movement_residual_features(raw_xu, raw_yu, fs_est)
                    raw_fingertip_power = float(raw_feats.get("tremor_power", 0.0))
                except Exception:
                    pass
        elif use_box_micro:
            local_res = resample_uniform(t, local_x, local_y, fs_est)
            global_res = resample_uniform(t, global_x, global_y, fs_est)
            if local_res is None or global_res is None:
                result.reason = "Tracking gaps."
                return result
            _, local_xu, local_yu = local_res
            _, global_xu, global_yu = global_res
            box_width_u = np.interp(tu, t, box_width)
            box_height_u = np.interp(tu, t, box_height)
            box_area_u = np.interp(tu, t, box_area)

        if snap_palm is not None:
            palm_t, palm_x, palm_y, palm_w, palm_h, hand_box_size, palm_quality = snap_palm
            palm_xu = np.interp(tu, palm_t, palm_x)
            palm_yu = np.interp(tu, palm_t, palm_y)
            hand_box_size_u = np.interp(tu, palm_t, hand_box_size)
            palm_w_u = np.interp(tu, palm_t, palm_w)
            palm_h_u = np.interp(tu, palm_t, palm_h)
            if tracking_quality_value is None and len(palm_quality):
                tracking_quality_value = float(np.median(palm_quality))

        hand_ref = float(np.median(ref)) if ref is not None else None

        # === Identical compute_metrics call as legacy ===
        metrics = compute_metrics(
            xu, yu, fs_est,
            hand_ref_pixels=hand_ref,
            baseline_rms=self.baseline_rms,
            task_mode=self.task_mode,
            local_xu=local_xu, local_yu=local_yu,
            global_xu=global_xu, global_yu=global_yu,
            tracking_quality=tracking_quality_value,
            box_width=box_width_u, box_height=box_height_u, box_area=box_area_u,
            palm_center_x=palm_xu, palm_center_y=palm_yu,
            hand_box_size=hand_box_size_u,
            frame_width=CAM_WIDTH, frame_height=CAM_HEIGHT,
            hand_box_width=palm_w_u, hand_box_height=palm_h_u,
            relative_fingertip_signals=relative_fingertip_signals,
            raw_fingertip_tremor_power=raw_fingertip_power,
        )
        if metrics is None:
            result.reason = "Computing…"
            return result

        # Override quality from assess_trial_quality
        metrics.quality_status = quality_status
        metrics.quality_reason = ", ".join(reasons) if reasons else ""
        metrics.fps_cv = fps_cv
        motion_valid = (metrics.motion_classification == "valid_tremor")
        quality_valid = (quality_status == "valid")
        metrics.research_valid = motion_valid and quality_valid
        if not metrics.research_valid:
            metrics.final_tremor_score = None
            if quality_status == "invalid":
                metrics.reason = reasons[0] if reasons else "Invalid quality"
            elif quality_status == "low_quality":
                metrics.reason = reasons[0] if reasons else "Low quality"
            else:
                metrics.reason = metrics.motion_reason

        self.last_metrics = metrics

        # Feed adaptive baseline
        try:
            self.baseline.update(metrics.rms_amp, metrics.band_ratio, metrics.score)
            if (self.baseline_rms is None and self.baseline.rms is not None
                    and self.baseline.samples > 30):
                self.baseline_rms = self.baseline.rms
        except Exception:
            pass

        # === Fill result ===
        result.live_score = int(metrics.live_motion_score)
        result.candidate_score = int(metrics.tremor_candidate_score)
        result.final_tremor_score = (
            int(metrics.final_tremor_score)
            if metrics.final_tremor_score is not None else None
        )
        result.research_valid = bool(metrics.research_valid)
        result.confidence_level = metrics.confidence_level
        result.motion_classification = metrics.motion_classification

        result.peak_hz = float(metrics.peak_hz)
        result.band_ratio = float(metrics.band_ratio)
        result.rms_amp = float(metrics.rms_amp)
        result.rms_amp_mm = float(metrics.rms_amp_mm)
        result.snr_db = float(metrics.snr_db)
        result.regularity = float(metrics.regularity)
        result.peak_sharpness = float(metrics.peak_sharpness)
        result.peak_prominence = float(metrics.peak_prominence)

        result.velocity_p95 = float(metrics.velocity_p95)
        result.path_ratio = float(metrics.path_ratio)
        result.center_drift = float(metrics.center_drift)
        result.tracking_quality_pct = int(metrics.tracking_quality * 100)
        result.palm_gross_motion_ratio = float(metrics.palm_gross_motion_ratio)

        # Palm motion friendly label
        palm_state = metrics.palm_motion_state
        if palm_state == "wide_range_hand_movement":
            result.palm_motion_label = "Too much movement"
        elif palm_state == "motion_too_high_for_tremor":
            result.palm_motion_label = "Moving"
        elif palm_state == "steady":
            result.palm_motion_label = "Still enough"
        else:
            result.palm_motion_label = palm_state.replace("_", " ").title()

        result.fs = float(metrics.fs)
        result.samples = int(metrics.samples)

        # Verdict + message
        if metrics.research_valid and metrics.final_tremor_score is not None:
            score = int(metrics.final_tremor_score)
            self.last_stable_score = score
            result.status_message = "✓ Steady reading captured"
            if score < 30:
                result.verdict_message = "Excellent — very steady hand"
                result.verdict_type = "success"
            elif score < 60:
                result.verdict_message = "Good — slight movement"
                result.verdict_type = "success"
            elif score < 80:
                result.verdict_message = "Notable tremor detected"
                result.verdict_type = "warning"
            else:
                result.verdict_message = "Significant tremor detected"
                result.verdict_type = "error"
        else:
            # Live feedback (not certified)
            live = result.live_score
            if live < 15:
                result.verdict_message = "Very little movement"
                result.verdict_type = "success"
            elif live < 40:
                result.verdict_message = (
                    f"Light movement at {result.peak_hz:.1f} Hz"
                )
                result.verdict_type = "warning"
            elif live < 70:
                result.verdict_message = (
                    f"Moderate movement at {result.peak_hz:.1f} Hz"
                )
                result.verdict_type = "warning"
            else:
                result.verdict_message = (
                    f"Strong movement at {result.peak_hz:.1f} Hz"
                )
                result.verdict_type = "error"
            result.reason = metrics.reason or ""
            if result.reason:
                result.status_message = result.reason

        return result
