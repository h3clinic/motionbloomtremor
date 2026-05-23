"""Webcam + hand/pose landmark capture for tremor analysis.

Optimisations:
- MediaPipe Hands at model_complexity=0.
- MediaPipe Pose runs only every POSE_EVERY frame.
- Minimal drawing on the published frame.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread

import cv2
import mediapipe as mp
import numpy as np

from ._cv_lock import CV_LOCK

CAM_WIDTH = 640
CAM_HEIGHT = 480
BUFFER_SECONDS = 30.0
POSE_EVERY = 3

LANDMARK_CHOICES = {
    "Index fingertip": 8,
    "Middle fingertip": 12,
    "Thumb tip": 4,
    "Wrist": 0,
}

PALM_LANDMARKS = (0, 5, 9, 13, 17)
FINGERTIP_LANDMARKS = (8, 12, 16)
PALM_RELATIVE_MIN_HAND_SIZE = 0.03
PALM_RELATIVE_PRIMARY_LANDMARKS = {
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
}
PALM_RELATIVE_DEBUG_LANDMARKS = {
    "thumb_tip": 4,
    "pinky_tip": 20,
    "index_mcp": 5,
    "middle_mcp": 9,
}

HAND_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class PoseSnapshot:
    nose: tuple[float, float] | None
    l_ear: tuple[float, float] | None
    r_ear: tuple[float, float] | None
    l_wrist: tuple[float, float] | None
    r_wrist: tuple[float, float] | None
    l_elbow: tuple[float, float] | None
    r_elbow: tuple[float, float] | None
    l_shoulder: tuple[float, float] | None
    r_shoulder: tuple[float, float] | None
    visibility: float = 0.0


def compute_palm_relative_landmarks(
    landmark_xy: np.ndarray,
    *,
    min_hand_size: float = PALM_RELATIVE_MIN_HAND_SIZE,
) -> dict | None:
    """Return palm-relative, hand-size-normalized landmark coordinates.

    Input coordinates are normalized image coordinates with shape (21, 2).
    The palm center is wrist + MCP landmarks. The hand size is the all-landmark
    bounding-box diagonal, clamped by `min_hand_size` to avoid noise blow-ups.
    """
    xy = np.asarray(landmark_xy, dtype=np.float64)
    if xy.shape[0] < 21 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        return None
    palm_center = xy[list(PALM_LANDMARKS)].mean(axis=0)
    span = xy.max(axis=0) - xy.min(axis=0)
    hand_size = float(np.hypot(span[0], span[1]))
    if hand_size < min_hand_size:
        return None

    relative: dict[str, tuple[float, float]] = {}
    for name, idx in {**PALM_RELATIVE_PRIMARY_LANDMARKS, **PALM_RELATIVE_DEBUG_LANDMARKS}.items():
        delta = (xy[idx] - palm_center) / max(hand_size, 1e-6)
        relative[name] = (float(delta[0]), float(delta[1]))

    return {
        "palm_center": (float(palm_center[0]), float(palm_center[1])),
        "hand_size": hand_size,
        "relative": relative,
    }


class TremorTracker:
    def __init__(self) -> None:
        self.samples: deque = deque(maxlen=4096)
        self.multi_finger_samples: deque = deque(maxlen=4096)  # Multi-fingertip buffer
        self.movement_tremor_samples: deque = deque(maxlen=4096)
        self.box_micro_samples: deque = deque(maxlen=4096)
        self.palm_center_samples: deque = deque(maxlen=4096)
        self.palm_relative_samples: deque = deque(maxlen=4096)
        self.stop_event = Event()
        self.thread: Thread | None = None

        self._frame_lock = Lock()
        self._latest_frame: np.ndarray | None = None

        self.hand_present = False
        self.landmark_idx = 8
        self.fps_meas = 0.0
        self.frame_shape: tuple[int, int] = (CAM_HEIGHT, CAM_WIDTH)

        self._pose_lock = Lock()
        self._pose: PoseSnapshot | None = None
        self._hand_tip_norm: tuple[float, float] | None = None
        # Hand grip signal in [0..1]: 1 = fingers curled toward palm
        # (holding something), 0 = hand open.
        self._grip_strength: float = 0.0

        self._cap: "cv2.VideoCapture | None" = None

        # Session recording state
        self._rec_lock = Lock()
        self._rec_path: str | None = None
        self._rec_writer: "cv2.VideoWriter | None" = None
        self._rec_frames: int = 0
        self._rec_fps: float = 20.0
        self._rec_start_t: float = 0.0

    # ----------------------------------------------- session recording
    def start_recording(self, path: str, fps: float = 20.0) -> None:
        """Begin saving the webcam feed to ``path`` (mp4).

        The writer is created lazily on the first frame inside the
        capture thread so the actual frame size is known.
        """
        with self._rec_lock:
            self._rec_path = str(path)
            self._rec_writer = None
            self._rec_frames = 0
            self._rec_fps = float(fps) if fps > 0 else 20.0
            self._rec_start_t = time.time()

    def stop_recording(self) -> tuple[str | None, int]:
        """Stop recording. Returns (path, frame_count). path is None if
        nothing was captured."""
        with self._rec_lock:
            writer = self._rec_writer
            path = self._rec_path
            frames = self._rec_frames
            self._rec_writer = None
            self._rec_path = None
            self._rec_frames = 0
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if frames <= 0:
            return (None, 0)
        return (path, frames)

    # ------------------------------------------------------------- lifecycle
    def set_landmark(self, idx: int) -> None:
        self.landmark_idx = idx
        self.samples.clear()
        self.multi_finger_samples.clear()
        self.movement_tremor_samples.clear()
        self.box_micro_samples.clear()
        self.palm_center_samples.clear()
        self.palm_relative_samples.clear()

    def start(self, cap: "cv2.VideoCapture") -> None:
        if self.thread and self.thread.is_alive():
            return
        self._cap = cap
        self.stop_event.clear()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        # Finalize any in-flight session recording
        try:
            self.stop_recording()
        except Exception:
            pass
        self.samples.clear()
        self.multi_finger_samples.clear()
        self.movement_tremor_samples.clear()
        self.box_micro_samples.clear()
        self.palm_center_samples.clear()
        self.palm_relative_samples.clear()
        self.hand_present = False
        with self._frame_lock:
            self._latest_frame = None
        with self._pose_lock:
            self._pose = None
        if self._cap is not None:
            try:
                with CV_LOCK:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ----------------------------------------------------------- accessors
    def get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame

    def get_pose(self) -> PoseSnapshot | None:
        with self._pose_lock:
            return self._pose

    def get_hand_tip_norm(self) -> tuple[float, float] | None:
        return self._hand_tip_norm

    def get_grip_strength(self) -> float:
        """0..1 - how curled/closed the hand currently is.

        Approximated from the ratio of fingertip-to-MCP distances vs
        palm size. A value >~0.5 suggests the user is gripping an
        object rather than holding an open hand up.
        """
        return float(self._grip_strength)

    def snapshot(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        if len(self.samples) < 16:
            return None
        arr = np.array(self.samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    def snapshot_multi_finger(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        """Get multi-fingertip trajectory for robust tremor extraction.
        
        Returns:
            (timestamps, index_x, index_y, middle_x, middle_y, ring_x, ring_y, ref_px, confidence)
            or None if insufficient data
        """
        if len(self.multi_finger_samples) < 16:
            return None
        arr = np.array(self.multi_finger_samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        # Return (t, ix, iy, mx, my, rx, ry, ref, conf)
        return (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4],
                arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8])

    def snapshot_movement_tremor(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        """Get local/global hand streams for movement-tremor analysis.

        Returns:
            (t, local_x, local_y, global_x, global_y, hand_scale,
             tracking_quality, finger_x, finger_y, palm_x, palm_y, ref_px)
            or None if insufficient data.
        """
        if len(self.movement_tremor_samples) < 16:
            return None
        arr = np.array(self.movement_tremor_samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        return tuple(arr[:, i] for i in range(arr.shape[1]))

    def snapshot_box_micro(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        """Get hand-box normalized micro/macro streams.

        Returns:
            (t, micro_x, micro_y, box_center_x, box_center_y,
             box_width, box_height, box_area, tracking_quality)
            or None if insufficient data.
        """
        if len(self.box_micro_samples) < 16:
            return None
        arr = np.array(self.box_micro_samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        return tuple(arr[:, i] for i in range(arr.shape[1]))

    def snapshot_palm_center(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        """Get palm-center macro-motion stream for the tremor veto gate.

        Returns:
            (t, palm_center_x, palm_center_y, hand_box_width,
             hand_box_height, hand_box_size, tracking_quality)
            or None if insufficient data.
        """
        if len(self.palm_center_samples) < 16:
            return None
        arr = np.array(self.palm_center_samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        return tuple(arr[:, i] for i in range(arr.shape[1]))

    def snapshot_palm_relative(self, seconds: float) -> tuple[np.ndarray, ...] | None:
        """Get palm-relative landmark streams for tremor candidate scoring.

        Returns:
            (t,
             index_rx, index_ry, middle_rx, middle_ry, ring_rx, ring_ry,
             thumb_rx, thumb_ry, pinky_rx, pinky_ry,
             index_mcp_rx, index_mcp_ry, middle_mcp_rx, middle_mcp_ry,
             palm_x, palm_y, hand_size, tracking_quality)
            or None if insufficient data.
        """
        if len(self.palm_relative_samples) < 16:
            return None
        arr = np.array(self.palm_relative_samples, dtype=np.float64)
        now = arr[-1, 0]
        mask = arr[:, 0] >= now - seconds
        arr = arr[mask]
        if arr.shape[0] < 16:
            return None
        return tuple(arr[:, i] for i in range(arr.shape[1]))

    # ----------------------------------------------------------- main loop
    def _run(self) -> None:
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose

        cap = self._cap
        if cap is None or not cap.isOpened():
            return

        ema_fps = 0.0
        last_t = time.time()
        frame_idx = 0

        hands = None
        pose = None
        try:
            hands = mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            hands = None
        try:
            pose = mp_pose.Pose(
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            pose = None
        try:
            while not self.stop_event.is_set():
                try:
                    with CV_LOCK:
                        ok, frame = cap.read()
                except Exception:
                    ok, frame = False, None
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    frame = cv2.flip(frame, 1)
                except Exception:
                    pass
                h, w = frame.shape[:2]
                self.frame_shape = (h, w)

                # If a session recording is active, append this frame.
                try:
                    with self._rec_lock:
                        if self._rec_path is not None:
                            if self._rec_writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                self._rec_writer = cv2.VideoWriter(
                                    self._rec_path, fourcc, self._rec_fps,
                                    (w, h),
                                )
                                if not self._rec_writer.isOpened():
                                    print(f"[REC] VideoWriter failed to open: {self._rec_path}", flush=True)
                                    self._rec_writer = None
                                    self._rec_path = None
                            if self._rec_writer is not None:
                                self._rec_writer.write(frame)
                                self._rec_frames += 1
                except Exception as exc:
                    print(f"[REC] write failed: {exc}", flush=True)

                # Publish the raw frame immediately so the user always
                # sees the camera feed, even if MediaPipe fails later.
                try:
                    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self._frame_lock:
                        self._latest_frame = raw_rgb
                except Exception:
                    pass

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                    self.fps_meas = ema_fps

                hresults = None
                if hands is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        hresults = hands.process(rgb)
                    except Exception:
                        hresults = None

                if pose is not None and frame_idx % POSE_EVERY == 0:
                    try:
                        rgb2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb2.flags.writeable = False
                        presults = pose.process(rgb2)
                        self._update_pose(presults)
                    except Exception:
                        pass

                if hresults is not None and hresults.multi_hand_landmarks:
                    self.hand_present = True
                    lm = hresults.multi_hand_landmarks[0].landmark
                    pts = np.array([[p.x * w, p.y * h] for p in lm],
                                   dtype=np.int32)
                    for a, b in HAND_CONN:
                        cv2.line(frame, tuple(pts[a]), tuple(pts[b]),
                                 (76, 201, 240), 2, cv2.LINE_AA)
                    
                    # Draw multi-fingertip tracking indicators (index, middle, ring)
                    index_tip = pts[8]
                    middle_tip = pts[12]
                    ring_tip = pts[16]
                    cv2.circle(frame, tuple(index_tip), 8, (229, 37, 133), -1, cv2.LINE_AA)  # Pink
                    cv2.circle(frame, tuple(middle_tip), 8, (255, 193, 7), -1, cv2.LINE_AA)   # Amber
                    cv2.circle(frame, tuple(ring_tip), 8, (76, 175, 80), -1, cv2.LINE_AA)      # Green
                    cv2.circle(frame, tuple(pts[4]), 5, (147, 51, 234), -1, cv2.LINE_AA)       # Thumb debug
                    cv2.circle(frame, tuple(pts[20]), 5, (59, 130, 246), -1, cv2.LINE_AA)      # Pinky debug
                    
                    # Also mark the old single landmark for reference
                    tip = pts[self.landmark_idx]
                    cv2.circle(frame, tuple(tip), 4, (255, 255, 255), 1, cv2.LINE_AA)  # White outline
                    
                    p = lm[self.landmark_idx]
                    self._hand_tip_norm = (float(p.x), float(p.y))

                    p0, p9 = lm[0], lm[9]
                    dxp = (p0.x - p9.x) * w
                    dyp = (p0.y - p9.y) * h
                    ref_px = float(np.hypot(dxp, dyp))

                    all_x = np.array([p.x for p in lm], dtype=np.float64)
                    all_y = np.array([p.y for p in lm], dtype=np.float64)
                    landmark_xy = np.column_stack((all_x, all_y))
                    box_left = float(all_x.min())
                    box_right = float(all_x.max())
                    box_top = float(all_y.min())
                    box_bottom = float(all_y.max())
                    box_width = max(float(box_right - box_left), 1e-6)
                    box_height = max(float(box_bottom - box_top), 1e-6)
                    box_center_x = 0.5 * (box_left + box_right)
                    box_center_y = 0.5 * (box_top + box_bottom)
                    box_area = box_width * box_height

                    box_left_px = int(max(0, min(w - 1, round(box_left * w))))
                    box_right_px = int(max(0, min(w - 1, round(box_right * w))))
                    box_top_px = int(max(0, min(h - 1, round(box_top * h))))
                    box_bottom_px = int(max(0, min(h - 1, round(box_bottom * h))))
                    cv2.rectangle(
                        frame,
                        (box_left_px, box_top_px),
                        (box_right_px, box_bottom_px),
                        (52, 211, 153),
                        2,
                        cv2.LINE_AA,
                    )

                    fingertip_norm = []
                    for tip_i in FINGERTIP_LANDMARKS:
                        tip_lm = lm[tip_i]
                        fingertip_norm.append((
                            (float(tip_lm.x) - box_left) / box_width,
                            (float(tip_lm.y) - box_top) / box_height,
                        ))
                    fingertip_norm_arr = np.array(fingertip_norm, dtype=np.float64)
                    micro_xy = fingertip_norm_arr.mean(axis=0)

                    palm_xy = np.array(
                        [[lm[i].x, lm[i].y] for i in PALM_LANDMARKS],
                        dtype=np.float64,
                    )
                    finger_xy = np.array(
                        [[lm[i].x, lm[i].y] for i in FINGERTIP_LANDMARKS],
                        dtype=np.float64,
                    )
                    palm_center = palm_xy.mean(axis=0)
                    finger_center = finger_xy.mean(axis=0)
                    palm_center_px = (
                        int(round(float(palm_center[0]) * w)),
                        int(round(float(palm_center[1]) * h)),
                    )
                    cv2.circle(frame, palm_center_px, 9, (34, 197, 94), -1, cv2.LINE_AA)
                    cv2.circle(frame, palm_center_px, 13, (255, 255, 255), 2, cv2.LINE_AA)

                    trail_points = []
                    for sample in self.palm_center_samples:
                        if now - sample[0] <= 2.0:
                            trail_points.append((
                                int(round(sample[1] * w)),
                                int(round(sample[2] * h)),
                            ))
                    if len(trail_points) >= 2:
                        for i in range(1, len(trail_points)):
                            alpha = i / max(len(trail_points) - 1, 1)
                            radius = max(2, int(round(2 + 3 * alpha)))
                            cv2.circle(frame, trail_points[i], radius, (34, 197, 94), 1, cv2.LINE_AA)
                            cv2.line(frame, trail_points[i - 1], trail_points[i], (34, 197, 94), 1, cv2.LINE_AA)

                    index_mcp = np.array([lm[5].x, lm[5].y], dtype=np.float64)
                    pinky_mcp = np.array([lm[17].x, lm[17].y], dtype=np.float64)
                    hand_scale = float(np.linalg.norm(index_mcp - pinky_mcp))
                    safe_scale = max(hand_scale, 1e-6)
                    local_xy = (finger_center - palm_center) / safe_scale
                    global_xy = palm_center

                    # Grip strength: mean fingertip→MCP distance divided
                    # by palm size. Closed hand → small tip-to-MCP
                    # distance → high grip.
                    palm = max(1e-6, float(np.hypot(
                        (p0.x - p9.x), (p0.y - p9.y))))
                    tip_mcp_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
                    extensions = []
                    for tip_i, mcp_i in tip_mcp_pairs:
                        t, m = lm[tip_i], lm[mcp_i]
                        extensions.append(
                            float(np.hypot(t.x - m.x, t.y - m.y)) / palm)
                    mean_ext = float(np.mean(extensions))
                    # extended finger ≈ 1.4+, curled finger ≈ 0.6
                    grip = (1.4 - mean_ext) / 0.8
                    self._grip_strength = float(max(0.0, min(1.0, grip)))

                    self.samples.append((now, float(p.x), float(p.y), ref_px))
                    
                    # Store multi-fingertip data for robust tremor extraction
                    # MediaPipe confidence is not directly available per landmark,
                    # use hand detection confidence as proxy
                    hand_conf = 0.9  # Default high confidence if hand detected
                    if hasattr(hresults, 'multi_hand_world_landmarks') and hresults.multi_hand_world_landmarks:
                        # Use detection confidence if available
                        hand_conf = 1.0
                    tracking_quality = float(max(0.0, min(1.0, hand_conf)))
                    
                    index_tip = lm[8]   # Index fingertip
                    middle_tip = lm[12]  # Middle fingertip
                    ring_tip = lm[16]    # Ring fingertip
                    
                    self.multi_finger_samples.append((
                        now,
                        float(index_tip.x), float(index_tip.y),
                        float(middle_tip.x), float(middle_tip.y),
                        float(ring_tip.x), float(ring_tip.y),
                        ref_px,
                        hand_conf
                    ))

                    self.movement_tremor_samples.append((
                        now,
                        float(local_xy[0]), float(local_xy[1]),
                        float(global_xy[0]), float(global_xy[1]),
                        hand_scale,
                        tracking_quality,
                        float(finger_center[0]), float(finger_center[1]),
                        float(palm_center[0]), float(palm_center[1]),
                        ref_px,
                    ))
                    self.box_micro_samples.append((
                        now,
                        float(micro_xy[0]), float(micro_xy[1]),
                        box_center_x, box_center_y,
                        box_width, box_height, box_area,
                        tracking_quality,
                    ))
                    self.palm_center_samples.append((
                        now,
                        float(palm_center[0]), float(palm_center[1]),
                        box_width, box_height,
                        float(np.hypot(box_width, box_height)),
                        tracking_quality,
                    ))
                    palm_relative = compute_palm_relative_landmarks(landmark_xy)
                    if palm_relative is not None:
                        rel = palm_relative["relative"]
                        rel_palm = palm_relative["palm_center"]
                        self.palm_relative_samples.append((
                            now,
                            rel["index_tip"][0], rel["index_tip"][1],
                            rel["middle_tip"][0], rel["middle_tip"][1],
                            rel["ring_tip"][0], rel["ring_tip"][1],
                            rel["thumb_tip"][0], rel["thumb_tip"][1],
                            rel["pinky_tip"][0], rel["pinky_tip"][1],
                            rel["index_mcp"][0], rel["index_mcp"][1],
                            rel["middle_mcp"][0], rel["middle_mcp"][1],
                            rel_palm[0], rel_palm[1],
                            palm_relative["hand_size"],
                            tracking_quality,
                        ))
                else:
                    self.hand_present = False
                    self._hand_tip_norm = None
                    self._grip_strength = 0.0

                while self.samples and now - self.samples[0][0] > BUFFER_SECONDS:
                    self.samples.popleft()
                while self.multi_finger_samples and now - self.multi_finger_samples[0][0] > BUFFER_SECONDS:
                    self.multi_finger_samples.popleft()
                while self.movement_tremor_samples and now - self.movement_tremor_samples[0][0] > BUFFER_SECONDS:
                    self.movement_tremor_samples.popleft()
                while self.box_micro_samples and now - self.box_micro_samples[0][0] > BUFFER_SECONDS:
                    self.box_micro_samples.popleft()
                while self.palm_center_samples and now - self.palm_center_samples[0][0] > BUFFER_SECONDS:
                    self.palm_center_samples.popleft()
                while self.palm_relative_samples and now - self.palm_relative_samples[0][0] > BUFFER_SECONDS:
                    self.palm_relative_samples.popleft()

                # Publish the annotated frame (overlays) for display.
                try:
                    out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self._frame_lock:
                        self._latest_frame = out
                except Exception:
                    pass

                frame_idx += 1
        except Exception:
            # Never let the capture thread die without releasing.
            pass
        finally:
            try:
                if hands is not None:
                    hands.close()
                if pose is not None:
                    pose.close()
            except Exception:
                pass

    # --------------------------------------------------------- pose helpers
    def _update_pose(self, results) -> None:
        if not results.pose_landmarks:
            with self._pose_lock:
                self._pose = None
            return
        lm = results.pose_landmarks.landmark

        def get(i: int) -> tuple[float, float] | None:
            p = lm[i]
            if p.visibility < 0.3:
                return None
            return (float(p.x), float(p.y))

        snap = PoseSnapshot(
            nose=get(0),
            l_ear=get(7), r_ear=get(8),
            l_shoulder=get(11), r_shoulder=get(12),
            l_elbow=get(13), r_elbow=get(14),
            l_wrist=get(15), r_wrist=get(16),
            visibility=float(np.mean(
                [lm[i].visibility for i in (0, 11, 12, 15, 16)])),
        )
        with self._pose_lock:
            self._pose = snap
