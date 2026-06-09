"""
check_detectability.py – Validate MediaPipe detection rate on synthetic renders.

Runs MediaPipe hand landmark detection on rendered videos and produces a
detectability report with per-video and aggregate metrics.

Acceptance criteria:
- Per-video valid frame rate: ≥80%
- Dataset mean valid frame rate: ≥85%
- No camera-angle group below 75%
- No skin-tone group below 75%
- No pose-family group below 70%

Usage:
    python handharness/check_detectability.py \
        --dataset datasets/synth_tremor_smoke \
        --model handharness/hand_landmarker.task \
        --min-valid-frame-rate 0.80
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

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


def check_video_detectability(
    video_path: str,
    detector,
    min_confidence: float = 0.3,
) -> Dict:
    """Run MediaPipe on a video and compute detectability metrics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detected_frames = []
    bbox_areas = []
    consecutive_valid = 0
    max_consecutive_valid = 0
    fail_frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = frame_idx * (1000.0 / fps) if fps > 0 else frame_idx * 33

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect_for_video(mp_image, int(timestamp_ms))

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            # Compute hand bounding box area (normalized)
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            bbox_area = bbox_w * bbox_h
            bbox_areas.append(bbox_area)

            detected_frames.append(frame_idx)
            consecutive_valid += 1
            max_consecutive_valid = max(max_consecutive_valid, consecutive_valid)
        else:
            fail_frames.append(frame_idx)
            consecutive_valid = 0

        frame_idx += 1

    cap.release()

    valid_frame_rate = len(detected_frames) / max(1, frame_idx)
    mean_bbox_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0

    return {
        "total_frames": frame_idx,
        "detected_frames": len(detected_frames),
        "valid_frame_rate": round(valid_frame_rate, 4),
        "consecutive_valid_run_max": max_consecutive_valid,
        "mean_hand_bbox_area": round(mean_bbox_area, 4),
        "detection_fail_frames": len(fail_frames),
        "fail_frame_indices": fail_frames[:20],  # First 20 only
        "image_size": [width, height],
    }


def main():
    parser = argparse.ArgumentParser(description="Check MediaPipe detectability on synthetic renders")
    parser.add_argument("--dataset", required=True, help="Dataset directory (with videos/ and metadata.jsonl)")
    parser.add_argument("--model", required=True, help="Path to hand_landmarker.task")
    parser.add_argument("--min-valid-frame-rate", type=float, default=0.80, help="Minimum per-video detection rate")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min detection confidence")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    min_rate = args.min_valid_frame_rate

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

    # Load metadata for video attributes
    metadata_map = {}
    metadata_path = dataset_dir / "metadata.jsonl"
    if metadata_path.exists():
        for line in metadata_path.read_text().strip().split("\n"):
            m = json.loads(line)
            metadata_map[m["video_id"]] = m

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"ERROR: No .mp4 files in {videos_dir}")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Detectability Check                        ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Dataset:     {dataset_dir}")
    print(f"  Videos:      {len(video_files)}")
    print(f"  Model:       {model_path.name}")
    print(f"  Min rate:    {min_rate:.0%}")
    print()

    # MediaPipe options
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )

    # Check each video
    per_video_results = []
    accepted = 0
    rejected = 0

    # Group tracking
    by_camera = defaultdict(list)
    by_skin_tone = defaultdict(list)
    by_pose_family = defaultdict(list)

    t_start = time.time()

    for i, video_path in enumerate(video_files):
        video_id = video_path.stem

        # Fresh detector per video (timestamps must be monotonically increasing)
        detector = vision.HandLandmarker.create_from_options(options)
        result = check_video_detectability(str(video_path), detector, args.min_confidence)
        detector.close()

        if "error" in result:
            print(f"  [{i+1}/{len(video_files)}] {video_id} → ERROR: {result['error']}")
            continue

        # Get metadata attributes
        meta = metadata_map.get(video_id, {})
        camera_angle = meta.get("camera_angle", "unknown")
        skin_tone = meta.get("skin_tone_id", "unknown")
        pose_family = meta.get("pose_family", "unknown")

        valid_rate = result["valid_frame_rate"]
        passed = valid_rate >= min_rate

        status = "✓ PASS" if passed else "✗ FAIL"
        if passed:
            accepted += 1
        else:
            rejected += 1

        print(f"  [{i+1}/{len(video_files)}] {video_id} | {valid_rate:.0%} valid | "
              f"bbox={result['mean_hand_bbox_area']:.3f} | {camera_angle:12s} | {pose_family:14s} | {status}")

        video_report = {
            "video_id": video_id,
            "valid_frame_rate": valid_rate,
            "total_frames": result["total_frames"],
            "detected_frames": result["detected_frames"],
            "consecutive_valid_run_max": result["consecutive_valid_run_max"],
            "mean_hand_bbox_area": result["mean_hand_bbox_area"],
            "detection_fail_frames": result["detection_fail_frames"],
            "camera_angle": camera_angle,
            "pose_family": pose_family,
            "skin_tone_id": skin_tone,
            "accepted_for_training": passed,
        }
        per_video_results.append(video_report)

        # Group tracking
        by_camera[camera_angle].append(valid_rate)
        by_skin_tone[skin_tone].append(valid_rate)
        by_pose_family[pose_family].append(valid_rate)

    elapsed = time.time() - t_start

    # Aggregate metrics
    all_rates = [r["valid_frame_rate"] for r in per_video_results]
    mean_rate = float(np.mean(all_rates)) if all_rates else 0.0
    min_rate_actual = float(np.min(all_rates)) if all_rates else 0.0

    # Group aggregates
    detection_by_camera = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_camera.items())}
    detection_by_skin_tone = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_skin_tone.items())}
    detection_by_pose_family = {k: round(float(np.mean(v)), 4) for k, v in sorted(by_pose_family.items())}

    # Check acceptance criteria
    # Minimum accepted ratio: if count >= 50, require >= 90% accepted (e.g., 45/50)
    min_accepted_ratio = 0.90 if len(video_files) >= 50 else 1.0
    min_accepted_count = int(np.ceil(len(video_files) * min_accepted_ratio))
    enough_accepted = accepted >= min_accepted_count

    dataset_passes = (
        mean_rate >= 0.85 and
        enough_accepted and
        (all(r["valid_frame_rate"] >= min_rate for r in per_video_results) if per_video_results else False)
    )

    camera_all_pass = all(v >= 0.75 for v in detection_by_camera.values())
    skin_all_pass = all(v >= 0.75 for v in detection_by_skin_tone.values())
    pose_all_pass = all(v >= 0.70 for v in detection_by_pose_family.values())

    # Write report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(dataset_dir),
        "total_videos": len(video_files),
        "accepted": accepted,
        "rejected": rejected,
        "elapsed_seconds": round(elapsed, 1),
        "aggregate": {
            "mean_valid_frame_rate": round(mean_rate, 4),
            "min_valid_frame_rate": round(min_rate_actual, 4),
            "dataset_passes_85_threshold": dataset_passes,
            "all_videos_pass_80_threshold": accepted == len(video_files),
            "min_accepted_count": min_accepted_count,
            "enough_accepted": enough_accepted,
        },
        "group_detection": {
            "by_camera_angle": detection_by_camera,
            "by_skin_tone": detection_by_skin_tone,
            "by_pose_family": detection_by_pose_family,
            "camera_all_above_75": camera_all_pass,
            "skin_all_above_75": skin_all_pass,
            "pose_all_above_70": pose_all_pass,
        },
        "acceptance_criteria": {
            "per_video_min": args.min_valid_frame_rate,
            "dataset_mean_min": 0.85,
            "camera_group_min": 0.75,
            "skin_tone_group_min": 0.75,
            "pose_family_group_min": 0.70,
        },
        "per_video": per_video_results,
    }

    report_path = dataset_dir / "detectability_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DETECTABILITY REPORT")
    print(f"{'='*60}")
    print(f"  Mean valid frame rate:  {mean_rate:.1%}")
    print(f"  Min valid frame rate:   {min_rate_actual:.1%}")
    print(f"  Accepted for training:  {accepted}/{len(video_files)}")
    print(f"  Rejected:               {rejected}/{len(video_files)}")
    print()
    print(f"  By camera angle:")
    for k, v in detection_by_camera.items():
        status = "✓" if v >= 0.75 else "✗"
        print(f"    {k:16s} {v:.1%} {status}")
    print()
    print(f"  By skin tone:")
    for k, v in detection_by_skin_tone.items():
        status = "✓" if v >= 0.75 else "✗"
        print(f"    {k:16s} {v:.1%} {status}")
    print()
    print(f"  By pose family:")
    for k, v in detection_by_pose_family.items():
        status = "✓" if v >= 0.70 else "✗"
        print(f"    {k:16s} {v:.1%} {status}")
    print()

    # Final verdict
    overall_pass = dataset_passes and camera_all_pass and skin_all_pass and pose_all_pass
    if overall_pass:
        print(f"  ✓ DATASET PASSES ALL ACCEPTANCE CRITERIA")
        print(f"    Accepted {accepted}/{len(video_files)} (need ≥{min_accepted_count})")
        print(f"    Ready to proceed with full generation.")
    else:
        print(f"  ✗ DATASET FAILS ACCEPTANCE CRITERIA")
        if not dataset_passes:
            print(f"    - Mean rate {mean_rate:.1%} < 85% or not all pass ≥80%")
        if not enough_accepted:
            print(f"    - Accepted {accepted}/{len(video_files)} < required {min_accepted_count}")
        if not camera_all_pass:
            failing = [k for k, v in detection_by_camera.items() if v < 0.75]
            print(f"    - Camera groups below 75%: {failing}")
        if not skin_all_pass:
            failing = [k for k, v in detection_by_skin_tone.items() if v < 0.75]
            print(f"    - Skin tone groups below 75%: {failing}")
        if not pose_all_pass:
            failing = [k for k, v in detection_by_pose_family.items() if v < 0.70]
            print(f"    - Pose groups below 70%: {failing}")
        print(f"\n    DO NOT proceed to full generation until this passes.")

    print(f"\n  Report: {report_path}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
