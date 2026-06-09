"""Landmark Extraction from Rendered Videos.

Runs MediaPipe Hands on each rendered video to extract 21 3D hand landmarks
per frame. This is the first stage of the training pipeline:

    video → MediaPipe → landmarks → features → model

The extracted landmarks are saved as NumPy arrays (.npy) alongside the videos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

SYNTH_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = SYNTH_ROOT / "outputs"


def extract_landmarks_from_video(
    video_path: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe hand landmarks from a video file.

    Args:
        video_path: Path to .mp4 video.
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence: MediaPipe tracking threshold.

    Returns:
        landmarks: np.ndarray shape (num_frames, 21, 3) — normalized x,y,z
        detection_mask: np.ndarray shape (num_frames,) — bool, True if hand detected
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_landmarks = []
    detection_mask = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
            hand = result.multi_hand_landmarks[0]
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                dtype=np.float32,
            )
            all_landmarks.append(frame_landmarks)
            detection_mask.append(True)
        else:
            # No hand detected — store zeros
            all_landmarks.append(np.zeros((21, 3), dtype=np.float32))
            detection_mask.append(False)

    cap.release()
    hands.close()

    landmarks = np.array(all_landmarks, dtype=np.float32)
    mask = np.array(detection_mask, dtype=bool)

    return landmarks, mask


def normalize_landmarks_to_hand_box(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks to the hand's bounding box per frame.

    This removes macro hand translation/scale, isolating tremor-only signal.
    The bounding box is defined by the detected hand landmarks themselves.

    Args:
        landmarks: Shape (num_frames, 21, 3).

    Returns:
        Normalized landmarks, same shape. Each frame is centered at (0,0,0)
        and scaled so the hand spans approximately [-0.5, 0.5] in each axis.
    """
    normalized = np.zeros_like(landmarks)

    for i in range(landmarks.shape[0]):
        frame = landmarks[i]
        if np.all(frame == 0):
            continue

        # Compute bounding box
        min_xyz = frame.min(axis=0)
        max_xyz = frame.max(axis=0)
        center = (min_xyz + max_xyz) / 2
        extent = max_xyz - min_xyz
        scale = extent.max()

        if scale < 1e-6:
            continue

        # Normalize: center at origin, scale to [-0.5, 0.5]
        normalized[i] = (frame - center) / scale

    return normalized


def process_dataset(
    dataset_dir: Path,
    output_dir: Optional[Path] = None,
):
    """Process all videos in a dataset directory.

    Args:
        dataset_dir: Directory containing videos/ subfolder.
        output_dir: Where to save landmarks (default: dataset_dir/landmarks/).
    """
    video_dir = dataset_dir / "videos"
    if output_dir is None:
        output_dir = dataset_dir / "landmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"Processing {len(video_files)} videos...")

    summary = {"total": len(video_files), "success": 0, "no_hand": 0, "error": 0}

    for i, video_path in enumerate(video_files):
        video_id = video_path.stem
        out_path = output_dir / f"{video_id}.npz"

        if out_path.exists():
            summary["success"] += 1
            continue

        try:
            landmarks, mask = extract_landmarks_from_video(video_path)
            normalized = normalize_landmarks_to_hand_box(landmarks)

            # Save as compressed numpy
            np.savez_compressed(
                out_path,
                landmarks_raw=landmarks,
                landmarks_normalized=normalized,
                detection_mask=mask,
            )

            detection_rate = mask.sum() / len(mask) if len(mask) > 0 else 0
            if detection_rate < 0.5:
                summary["no_hand"] += 1
            else:
                summary["success"] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(video_files)}] "
                      f"detection_rate={detection_rate:.1%}")

        except Exception as e:
            print(f"  ERROR {video_id}: {e}")
            summary["error"] += 1

    print(f"\nLandmark extraction complete:")
    print(f"  Success: {summary['success']}")
    print(f"  No hand: {summary['no_hand']}")
    print(f"  Errors:  {summary['error']}")

    # Save summary
    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from videos")
    parser.add_argument("--dataset-dir", type=str,
                        default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    process_dataset(dataset_dir, output_dir)


if __name__ == "__main__":
    main()
