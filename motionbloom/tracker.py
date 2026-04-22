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


class TremorTracker:
    def __init__(self) -> None:
        self.samples: deque = deque(maxlen=4096)
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

    # ------------------------------------------------------------- lifecycle
    def set_landmark(self, idx: int) -> None:
        self.landmark_idx = idx
        self.samples.clear()

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
        self.samples.clear()
        self.hand_present = False
        with self._frame_lock:
            self._latest_frame = None
        with self._pose_lock:
            self._pose = None
        if self._cap is not None:
            try:
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
        """0..1 — how curled/closed the hand currently is.

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

        hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose = mp_pose.Pose(
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                self.frame_shape = (h, w)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                hresults = hands.process(rgb)

                if frame_idx % POSE_EVERY == 0:
                    presults = pose.process(rgb)
                    self._update_pose(presults)

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                    self.fps_meas = ema_fps

                if hresults.multi_hand_landmarks:
                    self.hand_present = True
                    lm = hresults.multi_hand_landmarks[0].landmark
                    pts = np.array([[p.x * w, p.y * h] for p in lm],
                                   dtype=np.int32)
                    for a, b in HAND_CONN:
                        cv2.line(frame, tuple(pts[a]), tuple(pts[b]),
                                 (76, 201, 240), 2, cv2.LINE_AA)
                    tip = pts[self.landmark_idx]
                    cv2.circle(frame, tuple(tip), 8, (229, 37, 133), -1,
                               cv2.LINE_AA)
                    p = lm[self.landmark_idx]
                    self._hand_tip_norm = (float(p.x), float(p.y))

                    p0, p9 = lm[0], lm[9]
                    dxp = (p0.x - p9.x) * w
                    dyp = (p0.y - p9.y) * h
                    ref_px = float(np.hypot(dxp, dyp))

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
                else:
                    self.hand_present = False
                    self._hand_tip_norm = None
                    self._grip_strength = 0.0

                while self.samples and now - self.samples[0][0] > BUFFER_SECONDS:
                    self.samples.popleft()

                out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._latest_frame = out

                frame_idx += 1
        finally:
            try:
                hands.close()
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
