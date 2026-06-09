"""
extract_landmarks.py – MediaPipe landmark extraction from synthetic tremor videos.

Reads videos from a dataset directory, runs MediaPipe Hand Landmarker,
and outputs per-frame landmark JSON for each video.

Usage:
    python handharness/extract_landmarks.py \
        --dataset datasets/synth_tremor_smoke \
        --model handharness/hand_landmarker.task \
        --out datasets/synth_tremor_smoke/landmarks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)


# 21 MediaPipe hand landmarks
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def extract_video_landmarks(
    video_path: str,
    detector,
    min_detection_confidence: float = 0.3,
) -> Dict:
    """Extract per-frame landmarks from a video file.

    Returns:
        {
            "video_id": str,
            "total_frames": int,
            "detected_frames": int,
            "detection_rate": float,
            "fps": float,
            "frames": [
                {
                    "frame_idx": int,
                    "timestamp_ms": float,
                    "detected": bool,
                    "confidence": float,
                    "landmarks_3d": [[x,y,z], ...],  # 21 landmarks
                    "landmarks_2d": [[x,y], ...],    # 21 landmarks (normalized)
                }
            ]
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_data = []
    detected_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = frame_idx * (1000.0 / fps) if fps > 0 else 0

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        result = detector.detect_for_video(mp_image, int(timestamp_ms))

        frame_record = {
            "frame_idx": frame_idx,
            "timestamp_ms": round(timestamp_ms, 2),
            "detected": False,
            "confidence": 0.0,
            "landmarks_3d": None,
            "landmarks_2d": None,
        }

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            landmarks_3d = [[lm.x, lm.y, lm.z] for lm in hand]
            landmarks_2d = [[lm.x, lm.y] for lm in hand]

            # World landmarks if available
            if result.hand_world_landmarks and len(result.hand_world_landmarks) > 0:
                world_hand = result.hand_world_landmarks[0]
                landmarks_3d = [[lm.x, lm.y, lm.z] for lm in world_hand]

            confidence = 1.0  # MediaPipe Tasks API doesn't expose per-frame confidence easily
            if hasattr(result, 'handedness') and result.handedness:
                confidence = result.handedness[0][0].score if result.handedness[0] else 1.0

            frame_record["detected"] = True
            frame_record["confidence"] = round(confidence, 4)
            frame_record["landmarks_3d"] = [[round(v, 6) for v in lm] for lm in landmarks_3d]
            frame_record["landmarks_2d"] = [[round(v, 6) for v in lm] for lm in landmarks_2d]
            detected_count += 1

        frames_data.append(frame_record)
        frame_idx += 1

    cap.release()

    detection_rate = detected_count / max(1, frame_idx)

    return {
        "video_id": Path(video_path).stem,
        "video_path": video_path,
        "total_frames": frame_idx,
        "detected_frames": detected_count,
        "detection_rate": round(detection_rate, 4),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from video dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory (with videos/ subdir)")
    parser.add_argument("--model", required=True, help="Path to hand_landmarker.task model file")
    parser.add_argument("--out", help="Output directory for landmarks (default: <dataset>/landmarks)")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min detection confidence")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with existing landmarks")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    out_dir = Path(args.out).resolve() if args.out else dataset_dir / "landmarks"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found: {dataset_dir}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        print(f"ERROR: No videos/ directory in {dataset_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos
    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"ERROR: No .mp4 files found in {videos_dir}")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Landmark Extraction                        ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Model:    {model_path.name}")
    print(f"  Videos:   {len(video_files)}")
    print(f"  Output:   {out_dir}")
    print()

    # Initialize MediaPipe detector options (will create per-video)
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )

    # Process videos
    results_summary = []
    total_detected = 0
    total_frames = 0
    t_start = time.time()

    for i, video_path in enumerate(video_files):
        video_id = video_path.stem
        out_file = out_dir / f"{video_id}.json"

        if args.skip_existing and out_file.exists():
            print(f"  [{i+1}/{len(video_files)}] {video_id} → SKIPPED (exists)")
            continue

        # Create fresh detector per video (timestamp must be monotonically increasing)
        detector = vision.HandLandmarker.create_from_options(options)
        result = extract_video_landmarks(str(video_path), detector, args.min_confidence)
        detector.close()

        if "error" in result:
            print(f"  [{i+1}/{len(video_files)}] {video_id} → ERROR: {result['error']}")
            results_summary.append({"video_id": video_id, "error": result["error"]})
            continue

        # Save landmarks
        out_file.write_text(json.dumps(result, separators=(',', ':')))

        detection_rate = result["detection_rate"]
        total_detected += result["detected_frames"]
        total_frames += result["total_frames"]

        status = "OK" if detection_rate > 0.5 else "LOW" if detection_rate > 0 else "NONE"
        print(f"  [{i+1}/{len(video_files)}] {video_id} → {result['detected_frames']}/{result['total_frames']} frames ({detection_rate*100:.0f}%) [{status}]")

        results_summary.append({
            "video_id": video_id,
            "total_frames": result["total_frames"],
            "detected_frames": result["detected_frames"],
            "detection_rate": result["detection_rate"],
        })

    elapsed = time.time() - t_start
    overall_rate = total_detected / max(1, total_frames)

    # Write extraction summary
    summary = {
        "total_videos": len(video_files),
        "processed_videos": len(results_summary),
        "total_frames": total_frames,
        "total_detected_frames": total_detected,
        "overall_detection_rate": round(overall_rate, 4),
        "elapsed_seconds": round(elapsed, 1),
        "per_video": results_summary,
    }
    (out_dir / "_extraction_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"✓ Extraction complete")
    print(f"  Videos processed:      {len(results_summary)}")
    print(f"  Overall detection:     {total_detected}/{total_frames} frames ({overall_rate*100:.1f}%)")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print(f"  Output:                {out_dir}")

    if overall_rate < 0.3:
        print(f"\n⚠ WARNING: Detection rate is below 30%. Consider improving rendering quality.")
        print(f"  See: handharness/render_pose_grid.py for lighting/material tweaks")


if __name__ == "__main__":
    main()
