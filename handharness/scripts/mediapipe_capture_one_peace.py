#!/usr/bin/env python3
"""Capture one peace-sign hand pose with MediaPipe Tasks and save 21 landmarks."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


Landmark = Tuple[float, float, float]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/mediapipe/peace_pose.json")
    parser.add_argument("--annotated", default="output/mediapipe/peace_annotated.png")
    parser.add_argument("--model", default="output/mediapipe/hand_landmarker.task")
    parser.add_argument("--image", default="")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--require-peace", type=int, default=1)
    return parser.parse_args()


def ensure_model(model_path: Path):
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {model_path}")
    urllib.request.urlretrieve(MODEL_URL, str(model_path))


def vec(a: Landmark, b: Landmark) -> Landmark:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def norm(v: Landmark) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def angle_at(a: Landmark, b: Landmark, c: Landmark) -> float:
    ba = vec(b, a)
    bc = vec(b, c)
    n1 = norm(ba)
    n2 = norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    cos_v = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.acos(cos_v)


def finger_extension_score(lms: Sequence[Landmark], mcp: int, pip: int, dip: int, tip: int) -> float:
    a1 = angle_at(lms[mcp], lms[pip], lms[dip])
    a2 = angle_at(lms[pip], lms[dip], lms[tip])
    return 0.5 * (a1 + a2)


def is_peace_sign(lms: Sequence[Landmark]) -> bool:
    idx = finger_extension_score(lms, 5, 6, 7, 8)
    mid = finger_extension_score(lms, 9, 10, 11, 12)
    rng = finger_extension_score(lms, 13, 14, 15, 16)
    pky = finger_extension_score(lms, 17, 18, 19, 20)
    tips_up = (lms[8][1] < lms[16][1]) and (lms[12][1] < lms[20][1])
    return idx > 2.35 and mid > 2.30 and rng < 2.15 and pky < 2.10 and tips_up


def to_xyz_list(mp_lms) -> List[Landmark]:
    return [(lm.x, lm.y, lm.z) for lm in mp_lms]


def draw_landmarks(frame, lms: Sequence[Landmark]):
    h, w = frame.shape[:2]
    for i, j in HAND_CONNECTIONS:
        x1 = int(lms[i][0] * w)
        y1 = int(lms[i][1] * h)
        x2 = int(lms[j][0] * w)
        y2 = int(lms[j][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (90, 220, 90), 2, cv2.LINE_AA)
    for x, y, _ in lms:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (50, 180, 255), -1, cv2.LINE_AA)


def save_pose(path: Path, landmarks: Sequence[Landmark], handedness: str, source: str, space: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "coordinate_space": space,
        "handedness": handedness,
        "landmarks": [{"x": p[0], "y": p[1], "z": p[2]} for p in landmarks],
        "landmark_count": len(landmarks),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_landmarker(model_path: str, image_mode: bool):
    running_mode = vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=running_mode,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_one_result(result):
    if not result.hand_landmarks:
        return None
    lms_norm = to_xyz_list(result.hand_landmarks[0])

    handed = "unknown"
    if result.handedness and result.handedness[0]:
        handed = result.handedness[0][0].category_name.lower()

    if result.hand_world_landmarks:
        lms_world = to_xyz_list(result.hand_world_landmarks[0])
        return (lms_norm, lms_world, handed)
    return (lms_norm, None, handed)


def detect_from_image(args: argparse.Namespace, landmarker) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Unable to read image: {image_path}")
        return 2

    mp_image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(mp_image)
    sample = extract_one_result(result)
    if not sample:
        print("No hand detected in image")
        return 1

    lms_norm, lms_world, handed = sample
    if args.require_peace and not is_peace_sign(lms_norm):
        print("Hand detected, but peace sign condition not satisfied")
        return 1

    draw_landmarks(frame, lms_norm)

    out_pose = Path(args.output)
    out_img = Path(args.annotated)
    if lms_world:
        save_pose(out_pose, lms_world, handed, "image", "world")
    else:
        save_pose(out_pose, lms_norm, handed, "image", "normalized")

    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), frame)
    print(f"Saved peace pose: {out_pose}")
    print(f"Saved annotated image: {out_img}")
    return 0


def detect_from_webcam(args: argparse.Namespace, landmarker) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}")
        return 2

    out_pose = Path(args.output)
    out_img = Path(args.annotated)

    frame_idx = 0
    status = 1
    try:
        while frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_idx * 33)
            sample = extract_one_result(result)

            if sample:
                lms_norm, lms_world, handed = sample
                draw_landmarks(frame, lms_norm)
                peace = is_peace_sign(lms_norm)

                cv2.putText(
                    frame,
                    "PEACE" if peace else "HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (40, 220, 40) if peace else (255, 220, 40),
                    2,
                    cv2.LINE_AA,
                )

                if peace or not args.require_peace:
                    if lms_world:
                        save_pose(out_pose, lms_world, handed, "webcam", "world")
                    else:
                        save_pose(out_pose, lms_norm, handed, "webcam", "normalized")

                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_img), frame)
                    print(f"Saved peace pose: {out_pose}")
                    print(f"Saved annotated image: {out_img}")
                    status = 0
                    break

            cv2.imshow("MediaPipe Peace Capture (press q to cancel)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                status = 1
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if status != 0:
        print("Did not capture a valid peace sign pose")
    return status


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    ensure_model(model_path)

    with make_landmarker(str(model_path), image_mode=bool(args.image)) as landmarker:
        if args.image:
            return detect_from_image(args, landmarker)
        return detect_from_webcam(args, landmarker)


if __name__ == "__main__":
    raise SystemExit(main())
