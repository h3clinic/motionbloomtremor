"""
TopoTremor live bridge — 3D Hand State Landscape Edition.

Architecture
------------
MediaPipe per-frame hand box
  ↓
Observation extraction (detected centre, scale, optical flow)
  ↓
HandStateLandscape.update(obs)       ← multi-hypothesis particle landscape
  ↓
compute_residuals()                  ← residual = observed - macro_state_delta
  ↓
run_tremor_analysis()                ← bandpass → PSD → peak_hz → H1 topology
  ↓
build_landscape_diagnostics()        ← 3D cloud + scores → JSON to renderer
  ↓
stdout JSON stream to Electron

Emits:
  {"type":"status","running":true/false,"message":…}
  {"type":"frame","jpg":…,"w":…,"h":…,"points":…,"landscape":{…}}
  {"type":"metrics","metrics":{…},"status_message":…}

Run:
  .venv311/bin/python topo_bridge.py            # live webcam
  .venv311/bin/python topo_bridge.py --dry-run  # no camera
"""

from __future__ import annotations

import argparse
import base64
import json
import signal as _signal_module
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hand_landscape import (
    HandStateLandscape,
    Observation,
    TrackingQualityMonitor,
    TremorAnalyzer,
    DotFieldAnalyzer,
    compute_residuals,
    assign_regions,
    build_landscape_diagnostics,
    build_waiting_diagnostics,
)
from hand_landscape.dots import (
    TREMOR_CONFIRMED, TREMOR_CANDIDATE, TRACKING_NOISE, RELATCHING as DOT_RELATCHING,
    STABLE_NO_TREMOR, INVALID as DOT_INVALID, DEBUG_MODE,
)

# Per-dot preview styling: (BGR colour, radius). Confirmed tremor dots are
# intensified; tracking-noise / relatching dots are dimmed; stable dots soft.
_DOT_STATE_STYLE = {
    TREMOR_CONFIRMED: ((40, 40, 255), 3),     # intense red, large
    TREMOR_CANDIDATE: ((0, 140, 255), 2),     # orange
    STABLE_NO_TREMOR: ((130, 200, 130), 1),   # soft green
    TRACKING_NOISE:   ((90, 90, 90), 1),      # dim grey
    DOT_RELATCHING:   ((70, 70, 70), 1),      # dimmer grey
    DOT_INVALID:      ((55, 55, 55), 1),      # faint
}

# ─── camera + flow parameters ─────────────────────────────────────────────────
CAM_WIDTH        = 640
CAM_HEIGHT       = 480
WINDOW_SECONDS   = 5.0
ANALYSIS_INTERVAL = 0.25
MIN_SECONDS_TO_ANALYZE = 2.0
N_LANDSCAPE_CANDIDATES = 64

# Corner detector tuned to pack a DENSE ~500-dot field into the (small) hand
# box: more corners, finer block, lower quality floor and a small min spacing.
FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.005, minDistance=4, blockSize=5)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
# Dense residual-dot field: maintain ~TARGET_DOTS stable dots inside the HAND
# bounding box ONLY (the MediaPipe-hand box spans wrist→fingertips, never the
# arm / elbow). Seeding is masked to the box and any dot that drifts outside is
# culled every frame, so all nodes act strictly within the hand. The field is
# refilled GRADUALLY (≤ REFILL_PER_FRAME per frame, deduped against survivors)
# so freshly spawned dots warm up before they can vote.
TARGET_DOTS    = 500
REFILL_PER_FRAME = 40
REFILL_MIN_DIST  = 4.0
RESEED_BELOW   = 80      # full re-seed only if the field collapses below this
# Lucas-Kanade per-point tracking error that maps to zero per-dot quality.
# Typical good-match LK errors are < 10; > ~25 means an unreliable point.
LK_ERR_MAX     = 25.0

FRAME_W        = 480
FRAME_H        = 360
FRAME_INTERVAL = 1.0 / 12.0
JPEG_QUALITY   = 50

MOTION_DECAY   = 0.6
MOTION_BLUR    = 5


def _scale_box(box, sx: float, sy: float):
    if box is None:
        return None
    x, y, w, h = box
    return [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]


def box_mask_for(shape, box) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    x, y, w, h = box
    m[y:y + h, x:x + w] = 255
    return m


def in_box(pts: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return (
        (pts[:, 0] >= x) & (pts[:, 0] < x + w)
        & (pts[:, 1] >= y) & (pts[:, 1] < y + h)
    )


def _box_scale_z(box, reference_area):
    if box is None:
        return None
    _, _, bw, bh = box
    area = float(bw * bh)
    if reference_area is None or reference_area <= 0:
        return 1.0
    return float(np.sqrt(area / reference_area))


def detect_hand_box_mediapipe(frame_bgr: np.ndarray, hands):
    """Return (box, landmarks_px) for the largest detected hand, else (None, None).

    landmarks_px is a (21, 2) float array of MediaPipe landmark pixel positions,
    used downstream to assign each tracked dot to a finger / palm region.
    """
    if hands is None:
        return None, None
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out = hands.process(rgb)
    if not out.multi_hand_landmarks:
        return None, None
    best = None
    best_lm = None
    best_area = 0.0
    for hand_lm in out.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        x1 = min(w, int(max(xs)))
        y1 = min(h, int(max(ys)))
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        pad = int(0.08 * max(bw, bh))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        area = float(bw * bh)
        if area > best_area:
            best_area = area
            best = (x0, y0, bw, bh)
            best_lm = np.array([[lm.x * w, lm.y * h] for lm in hand_lm.landmark],
                               dtype=np.float64)
    return best, best_lm


def emit_preview_frame(frame, raw_box, landscape, latest_pts, per_point_residuals,
                       tracking_debug=None, dot_states=None, dot_amps=None):
    vis = frame.copy()
    if raw_box is not None:
        rx, ry, rw, rh = raw_box
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 100, 180), 1)
    if latest_pts is not None and len(latest_pts):
        if dot_states is not None and len(dot_states) == len(latest_pts):
            # Dense residual-dot field. Each dot is coloured by its state and
            # INTENSIFIED by its filtered residual amplitude; invalid / noise /
            # relatching dots are dimmed so only real tremor energy stands out.
            amps = None
            amp_ref = 0.0
            if dot_amps is not None and len(dot_amps) == len(latest_pts):
                amps = np.asarray(dot_amps, dtype=np.float64)
                if np.any(amps > 0):
                    amp_ref = max(float(np.percentile(amps[amps > 0], 90)), 1e-6)
            for i, ((px, py), state) in enumerate(zip(latest_pts, dot_states)):
                base, radius = _DOT_STATE_STYLE.get(state, ((90, 90, 90), 1))
                if state in (DOT_INVALID, DOT_RELATCHING, TRACKING_NOISE):
                    # Dim invalid / relatching / tracking-noise dots.
                    color = tuple(int(c * 0.40) for c in base)
                    r = 1
                else:
                    t = float(np.clip(amps[i] / amp_ref, 0.0, 1.0)) if (amps is not None and amp_ref > 0) else 0.5
                    scale = 0.45 + 0.55 * t          # intensify by amplitude
                    color = tuple(int(c * scale) for c in base)
                    r = radius if t > 0.6 else max(1, radius - 1)
                cv2.circle(vis, (int(px), int(py)), r, color, -1)
        elif per_point_residuals is not None and len(per_point_residuals) == len(latest_pts):
            magnitudes = np.linalg.norm(per_point_residuals, axis=1)
            max_mag = max(float(np.percentile(magnitudes, 90)), 1.0)
            for (px, py), mag in zip(latest_pts, magnitudes):
                t = min(1.0, float(mag) / max_mag)
                if t < 0.5:
                    color = (int(255 * (1 - 2 * t)), 255, 200)
                else:
                    color = (0, int(255 * (2 - 2 * t)), int(200 * (2 * t - 1)))
                cv2.circle(vis, (int(px), int(py)), 2, color, -1)
        else:
            for px, py in latest_pts:
                cv2.circle(vis, (int(px), int(py)), 2, (0, 230, 30), -1)
    small = cv2.resize(vis, (FRAME_W, FRAME_H))
    ok_jpg, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok_jpg:
        return
    sx = FRAME_W / float(vis.shape[1])
    sy = FRAME_H / float(vis.shape[0])
    emit({
        "type": "frame",
        "jpg": base64.b64encode(buf).decode("ascii"),
        "w": FRAME_W, "h": FRAME_H,
        "points": int(0 if latest_pts is None else len(latest_pts)),
        "raw_box": _scale_box(raw_box, sx, sy),
        "landscape": landscape.debug(),
        "tracking": tracking_debug or {},
    })


def refill_dots(gray, seed_mask, good_new, good_ids, next_id):
    """Top the dense dot field back up toward TARGET_DOTS, gradually.

    Surviving dots keep their identity (and history). New corners are deduped
    against survivors (≥ REFILL_MIN_DIST px away) and capped at REFILL_PER_FRAME
    so the field is replenished slowly — fresh dots then warm up before they can
    vote, instead of a sudden refill injecting a burst of fake residual motion.

    Returns (prev_pts (M,1,2) float32, prev_ids (M,), next_id).
    """
    base_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
    base_ids = good_ids
    n = len(good_new)
    deficit = TARGET_DOTS - n
    if deficit <= 0:
        return base_pts, base_ids, next_id
    extra = cv2.goodFeaturesToTrack(gray, mask=seed_mask, **FEATURE_PARAMS)
    if extra is None or not len(extra):
        return base_pts, base_ids, next_id
    extra = extra.reshape(-1, 2).astype(np.float32)
    if n > 0:
        dists = np.linalg.norm(extra[:, None, :] - good_new[None, :, :], axis=2)
        extra = extra[np.min(dists, axis=1) >= REFILL_MIN_DIST]
    take = int(min(len(extra), deficit, REFILL_PER_FRAME))
    if take <= 0:
        return base_pts, base_ids, next_id
    extra = extra[:take]
    extra_ids = np.arange(next_id, next_id + take)
    next_id += take
    prev_pts = np.vstack([base_pts, extra.reshape(-1, 1, 2)])
    prev_ids = (np.concatenate([base_ids, extra_ids])
                if base_ids is not None else extra_ids)
    return prev_pts, prev_ids, next_id


def emit(payload: dict) -> None:
    try:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, ValueError):
        raise SystemExit(0)


def merge_validity(landscape_status, tracking_state, span, fs, n_flow_points):
    """Merge landscape, tracking, and session validity into one status.

    Priority (most severe first). Only VALID yields a trusted tremor score;
    FLOW_LAG / RELATCHING / LANDSCAPE_DIFFUSE / LOW_POINTS all suppress tremor.
    """
    if landscape_status == "HAND_LOST":
        return "HAND_LOST"
    if fs <= 8.0:
        return "LOW_FPS"
    if span < MIN_SECONDS_TO_ANALYZE:
        return "INSUFFICIENT_DURATION"
    if tracking_state == "RELATCHING":
        return "RELATCHING"
    if tracking_state == "FLOW_LAG":
        return "FLOW_LAG"
    if landscape_status == "LANDSCAPE_DIFFUSE":
        return "LANDSCAPE_DIFFUSE"
    if n_flow_points < 8 or tracking_state == "LOW_POINTS" or landscape_status == "LOW_POINTS":
        return "LOW_POINTS"
    if landscape_status == "INSUFFICIENT_DURATION":
        return "INSUFFICIENT_DURATION"
    return "VALID"


def run_dry() -> int:
    emit({"type": "status", "running": False,
          "message": "Dry run ok: hand_landscape + cv2 loaded."})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TopoTremor live bridge")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        return run_dry()

    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    _signal_module.signal(_signal_module.SIGINT, _stop)
    _signal_module.signal(_signal_module.SIGTERM, _stop)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        emit({"type": "status", "running": False, "message": "Camera open failed"})
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    ok, frame = cap.read()
    if not ok:
        emit({"type": "status", "running": False, "message": "Camera read failed"})
        cap.release()
        return 1

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        import mediapipe as mp
    except Exception:
        emit({"type": "status", "running": False,
              "message": "MediaPipe is required. Run: .venv311/bin/pip install mediapipe"})
        cap.release()
        return 1

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landscape = HandStateLandscape(n_candidates=N_LANDSCAPE_CANDIDATES)
    tracking_monitor = TrackingQualityMonitor()
    tremor_analyzer = TremorAnalyzer()
    dot_field = DotFieldAnalyzer()
    _reference_area = None

    residual_x_buf: deque = deque(maxlen=600)
    ts_buf: deque = deque(maxlen=600)
    tracked_buf: deque = deque(maxlen=600)

    prev_pts = None
    prev_ids = None          # persistent dot IDs aligned with prev_pts
    next_dot_id = 0          # monotonically increasing dot-ID allocator
    latest_pts = None
    latest_dot_states = None  # per-dot state aligned with latest_pts (for preview)
    latest_dot_amps = None    # per-dot filtered amplitude aligned with latest_pts
    per_point_residuals = None
    motion_accum = None
    last_analysis = 0.0
    last_frame = 0.0
    raw_box = None
    raw_lm = None            # MediaPipe landmark pixels for region assignment
    last_dot_metrics = {}

    emit({"type": "status", "running": True,
          "message": "3D Hand State Landscape started"
                     + (" · DEBUG (evidence revealed, thresholds relaxed)" if DEBUG_MODE else "")})

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None and prev_gray.shape == gray.shape:
                d = cv2.absdiff(gray, prev_gray).astype(np.float32)
                if motion_accum is None or motion_accum.shape != d.shape:
                    motion_accum = d
                else:
                    motion_accum = MOTION_DECAY * motion_accum + (1.0 - MOTION_DECAY) * d

            raw_box, raw_lm = detect_hand_box_mediapipe(frame, hands)

            if raw_box is not None and _reference_area is None:
                _, _, bw, bh = raw_box
                _reference_area = float(bw * bh)

            scale_z = _box_scale_z(raw_box, _reference_area)
            det_conf = 0.85 if raw_box is not None else 0.0

            search_box = raw_box
            seed_mask = box_mask_for(gray.shape, search_box) if search_box is not None else None

            n = 0
            good_new = np.empty((0, 2), dtype=np.float32)
            good_old = np.empty((0, 2), dtype=np.float32)
            good_ids = None
            good_err = None

            if seed_mask is None:
                prev_pts = None
                prev_ids = None
                latest_pts = None
                latest_dot_states = None
                latest_dot_amps = None
                per_point_residuals = None
            else:
                if prev_pts is None or len(prev_pts) < RESEED_BELOW:
                    # (Re)seed features. This frame has no flow yet (needs two
                    # frames), so update the landscape from detection only and
                    # keep the preview alive so the UI doesn't freeze.
                    prev_pts = cv2.goodFeaturesToTrack(gray, mask=seed_mask, **FEATURE_PARAMS)
                    if prev_pts is not None and len(prev_pts):
                        prev_ids = np.arange(next_dot_id, next_dot_id + len(prev_pts))
                        next_dot_id += len(prev_pts)
                    else:
                        prev_ids = None
                    prev_gray = gray
                    if raw_box is not None:
                        cx = raw_box[0] + raw_box[2] / 2.0
                        cy = raw_box[1] + raw_box[3] / 2.0
                        obs = Observation(
                            detected_center_x=cx, detected_center_y=cy,
                            detected_scale_z=scale_z, detected_box=raw_box,
                            detection_confidence=det_conf,
                        )
                        landscape.update(obs, timestamp=now)
                    if now - last_frame >= FRAME_INTERVAL:
                        emit_preview_frame(frame, raw_box, landscape, latest_pts,
                                           per_point_residuals, tracking_monitor.debug(),
                                           latest_dot_states, latest_dot_amps)
                        last_frame = now
                    continue

                nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
                if nxt is not None and st is not None:
                    mask1d = st.flatten() == 1
                    gn = nxt[mask1d].reshape(-1, 2)
                    go = prev_pts[mask1d].reshape(-1, 2)
                    gi = prev_ids[mask1d] if prev_ids is not None else None
                    ge = err.flatten()[mask1d] if err is not None else None
                    if search_box is not None and len(gn):
                        keep = in_box(gn, search_box)
                        gn = gn[keep]
                        go = go[keep]
                        if gi is not None:
                            gi = gi[keep]
                        if ge is not None:
                            ge = ge[keep]
                    good_new = gn
                    good_old = go
                    good_ids = gi
                    good_err = ge
                    n = len(good_new)

            prev_gray = gray

            if raw_box is not None:
                cx = raw_box[0] + raw_box[2] / 2.0
                cy = raw_box[1] + raw_box[3] / 2.0
            else:
                cx = cy = None

            flow_deltas = (good_new - good_old) if n > 0 else None

            obs = Observation(
                detected_center_x=cx,
                detected_center_y=cy,
                detected_scale_z=scale_z,
                detected_box=raw_box,
                optical_flow_points=good_new if n > 0 else None,
                optical_flow_deltas=flow_deltas,
                detection_confidence=det_conf,
            )
            landscape.update(obs, timestamp=now)

            if n >= 10 and flow_deltas is not None:
                ws = landscape.weighted_state
                prev_ws = landscape.prev_weighted_state
                if ws is not None:
                    res_x, res_y, per_point_res = compute_residuals(flow_deltas, ws, prev_ws)
                else:
                    res_x, res_y, per_point_res = 0.0, 0.0, None

                # ── Tracking-quality evidence (NOT tremor) ────────────────────
                # The landscape macro delta is lag-free (MediaPipe-anchored);
                # the flow median lags. A large gap => FLOW_LAG, which lowers
                # confidence and suppresses tremor — it must never raise it.
                if ws is not None and prev_ws is not None:
                    macro_delta = np.array(
                        [ws.center_x - prev_ws.center_x, ws.center_y - prev_ws.center_y],
                        dtype=np.float64,
                    )
                else:
                    macro_delta = None
                flow_median_delta = np.median(flow_deltas, axis=0).astype(np.float64)
                tracking_monitor.update(macro_delta, flow_median_delta, n, relatched=False)

                residual_x_buf.append(res_x)
                ts_buf.append(now)
                tracked_buf.append(n)
                latest_pts = good_new
                per_point_residuals = per_point_res

                # ── Per-dot micro-oscillator sensing ──────────────────────────
                # Route each point's residual to its persistent dot. LK error
                # becomes per-point tracking quality (low error = trustworthy).
                if good_ids is not None and per_point_res is not None and len(good_ids) == len(per_point_res):
                    if good_err is not None and len(good_err) == n:
                        q_pts = np.clip(1.0 - good_err / LK_ERR_MAX, 0.0, 1.0)
                    else:
                        q_pts = np.ones(n, dtype=np.float64)
                    regions = assign_regions(good_new, raw_lm)
                    dot_field.update(good_ids, good_new[:, 0], good_new[:, 1],
                                     per_point_res, q_pts, now, regions)
                    dot_field.prune(now)
                    latest_dot_states = [dot_field.state_of(int(i)) for i in good_ids]
                    latest_dot_amps = [dot_field.amplitude_of(int(i)) for i in good_ids]
                else:
                    latest_dot_states = None
                    latest_dot_amps = None

                # Gradual, deduped refill back toward the dense ~500-dot target.
                prev_pts, prev_ids, next_dot_id = refill_dots(
                    gray, seed_mask, good_new, good_ids, next_dot_id)
            else:
                # Too few points survived → hard relatch coming next frame.
                relatched = seed_mask is not None  # only a relatch if we HAD a hand
                tracking_monitor.update(None, None, n, relatched=relatched)
                prev_pts = None
                prev_ids = None
                latest_pts = None
                latest_dot_states = None
                latest_dot_amps = None
                per_point_residuals = None

            if now - last_frame >= FRAME_INTERVAL:
                emit_preview_frame(frame, raw_box, landscape, latest_pts,
                                   per_point_residuals, tracking_monitor.debug(),
                                   latest_dot_states, latest_dot_amps)
                last_frame = now

            while len(ts_buf) > 2 and (ts_buf[-1] - ts_buf[0]) > WINDOW_SECONDS:
                ts_buf.popleft()
                residual_x_buf.popleft()
                tracked_buf.popleft()

            if now - last_analysis >= ANALYSIS_INTERVAL:
                tdebug = tracking_monitor.debug()
                if len(ts_buf) > 3:
                    span = ts_buf[-1] - ts_buf[0]
                    fs = (len(ts_buf) - 1) / span if span > 0 else 30.0
                    n_pts = int(np.median(tracked_buf)) if tracked_buf else 0
                    validity = merge_validity(
                        landscape.validity_status, tracking_monitor.tracking_state,
                        span, fs, n_pts,
                    )

                    if validity in ("HAND_LOST", "INSUFFICIENT_DURATION", "LOW_FPS"):
                        _msg = {
                            "HAND_LOST": "Hand lost — show hand to camera",
                            "INSUFFICIENT_DURATION": "Warming up…",
                            "LOW_FPS": "Low frame rate — check camera",
                        }.get(validity, "Analyzing…")
                        metrics, status = build_waiting_diagnostics(landscape, validity, _msg, tdebug)
                    else:
                        try:
                            # Always compute the residual spectrum so the UI can
                            # show flow_lag / residual diagnostics, but suppress
                            # the tremor SCORE whenever validity is not VALID.
                            suppress = (validity != "VALID")
                            tremor = tremor_analyzer.analyze(
                                list(residual_x_buf), fs, span,
                                tracking_quality=tracking_monitor.tracking_quality,
                                suppress=suppress,
                            )
                            # Per-dot micro-oscillator field — authoritative.
                            # NOTE: zeroing tracking_quality on suppress also
                            # zeroes the dot EVIDENCE at the source. In debug we
                            # keep the real quality so evidence survives and is
                            # merely flagged UNTRUSTED downstream.
                            dot_tq = tracking_monitor.tracking_quality / 100.0
                            if suppress and not DEBUG_MODE:
                                dot_tq = 0.0
                            last_dot_metrics = dot_field.analyze(fs, tracking_quality=dot_tq)
                            last_res = list(residual_x_buf)[-10:]
                            res_xy = (float(np.median(last_res)) if last_res else 0.0, 0.0)
                            metrics, status = build_landscape_diagnostics(
                                landscape, tremor, res_xy, per_point_residuals,
                                n_pts, validity, tdebug, last_dot_metrics,
                            )
                        except Exception as exc:
                            metrics, status = build_waiting_diagnostics(
                                landscape, validity,
                                f"Analysis error ({exc.__class__.__name__})", tdebug,
                            )

                    emit({"type": "metrics", "metrics": metrics, "status_message": status})
                else:
                    metrics, status = build_waiting_diagnostics(
                        landscape, "INSUFFICIENT_DURATION", "Locking onto hand…", tdebug,
                    )
                    emit({"type": "metrics", "metrics": metrics, "status_message": status})
                last_analysis = now

    finally:
        cap.release()
        hands.close()
        emit({"type": "status", "running": False, "message": "Landscape analysis stopped"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
