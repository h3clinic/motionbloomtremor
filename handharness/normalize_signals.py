"""
normalize_signals.py – Box-normalize landmarks and extract tremor signals.

Takes per-frame landmark JSON (from extract_landmarks.py) and produces:
1. Box-normalized landmark sequences (0-1 range per frame)
2. Displacement signals (frame-to-frame landmark motion)
3. Velocity and acceleration signals
4. Per-landmark and aggregate signal files

Usage:
    python handharness/normalize_signals.py \
        --landmarks datasets/synth_tremor_smoke/landmarks \
        --out datasets/synth_tremor_smoke/signals
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Fingertip indices for key signal extraction
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
WRIST_INDEX = 0


def box_normalize_frame(landmarks_3d: List[List[float]]) -> np.ndarray:
    """Normalize landmarks to bounding-box [0,1] range.

    Centers on wrist, scales by max extent.
    Returns (21, 3) array.
    """
    pts = np.array(landmarks_3d, dtype=np.float64)  # (21, 3)

    # Center on wrist
    wrist = pts[WRIST_INDEX]
    pts_centered = pts - wrist

    # Scale by max absolute extent
    max_extent = np.abs(pts_centered).max()
    if max_extent > 1e-8:
        pts_normalized = pts_centered / max_extent
    else:
        pts_normalized = pts_centered

    # Shift to [0, 1] range
    pts_normalized = (pts_normalized + 1.0) / 2.0

    return pts_normalized


def compute_displacement(landmarks_seq: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame displacement.

    Input: (T, 21, 3) normalized landmarks
    Output: (T-1, 21, 3) displacement vectors
    """
    return np.diff(landmarks_seq, axis=0)


def compute_velocity(displacement: np.ndarray, fps: float) -> np.ndarray:
    """Compute velocity (displacement / dt).

    Input: (T-1, 21, 3) displacement
    Output: (T-1, 21, 3) velocity
    """
    dt = 1.0 / fps
    return displacement / dt


def compute_acceleration(velocity: np.ndarray, fps: float) -> np.ndarray:
    """Compute acceleration (diff of velocity / dt).

    Input: (T-1, 21, 3) velocity
    Output: (T-2, 21, 3) acceleration
    """
    dt = 1.0 / fps
    return np.diff(velocity, axis=0) / dt


def compute_jerk(acceleration: np.ndarray, fps: float) -> np.ndarray:
    """Compute jerk (diff of acceleration / dt).

    Input: (T-2, 21, 3) acceleration
    Output: (T-3, 21, 3) jerk
    """
    dt = 1.0 / fps
    return np.diff(acceleration, axis=0) / dt


def compute_magnitude(vectors: np.ndarray) -> np.ndarray:
    """Compute L2 magnitude along last axis.

    Input: (..., 3) vectors
    Output: (...) magnitudes
    """
    return np.linalg.norm(vectors, axis=-1)


def interpolate_missing_frames(
    landmarks_seq: List[Optional[np.ndarray]],
    detected_flags: List[bool],
) -> Tuple[np.ndarray, List[bool]]:
    """Linearly interpolate missing landmark frames.

    Returns (T, 21, 3) array with interpolated values where possible.
    """
    T = len(landmarks_seq)
    result = np.zeros((T, 21, 3), dtype=np.float64)
    valid = [False] * T

    # Fill in detected frames
    for i in range(T):
        if detected_flags[i] and landmarks_seq[i] is not None:
            result[i] = landmarks_seq[i]
            valid[i] = True

    # Linear interpolation for gaps
    for lm_idx in range(21):
        for axis in range(3):
            # Find valid anchor points
            anchors = [(i, result[i, lm_idx, axis]) for i in range(T) if valid[i]]
            if len(anchors) < 2:
                continue

            for gap_start in range(T):
                if valid[gap_start]:
                    continue
                # Find prev and next valid frames
                prev_anchor = None
                next_anchor = None
                for ai, (idx, val) in enumerate(anchors):
                    if idx < gap_start:
                        prev_anchor = (idx, val)
                    elif idx > gap_start and next_anchor is None:
                        next_anchor = (idx, val)
                        break

                if prev_anchor and next_anchor:
                    # Linear interpolation
                    t_ratio = (gap_start - prev_anchor[0]) / (next_anchor[0] - prev_anchor[0])
                    result[gap_start, lm_idx, axis] = (
                        prev_anchor[1] + t_ratio * (next_anchor[1] - prev_anchor[1])
                    )

    return result, valid


def process_video_landmarks(landmark_data: Dict) -> Optional[Dict]:
    """Process a single video's landmark data into normalized signals.

    Returns signal dictionary or None if insufficient data.
    """
    frames = landmark_data.get("frames", [])
    fps = landmark_data.get("fps", 30)
    video_id = landmark_data.get("video_id", "unknown")

    if not frames:
        return None

    # Extract landmark sequences
    raw_landmarks = []
    detected_flags = []

    for frame in frames:
        if frame["detected"] and frame["landmarks_3d"]:
            raw_landmarks.append(np.array(frame["landmarks_3d"]))
            detected_flags.append(True)
        else:
            raw_landmarks.append(None)
            detected_flags.append(False)

    detection_rate = sum(detected_flags) / max(1, len(detected_flags))
    if detection_rate < 0.1:
        return {"video_id": video_id, "error": "insufficient_detection", "detection_rate": detection_rate}

    # Box-normalize detected frames
    normalized_landmarks = []
    for i, (lm, detected) in enumerate(zip(raw_landmarks, detected_flags)):
        if detected and lm is not None:
            normalized_landmarks.append(box_normalize_frame(lm))
        else:
            normalized_landmarks.append(None)

    # Interpolate missing frames
    landmarks_interp, valid_flags = interpolate_missing_frames(normalized_landmarks, detected_flags)

    T = len(landmarks_interp)
    if T < 4:
        return {"video_id": video_id, "error": "too_few_frames", "frame_count": T}

    # Compute kinematic signals
    displacement = compute_displacement(landmarks_interp)       # (T-1, 21, 3)
    velocity = compute_velocity(displacement, fps)              # (T-1, 21, 3)
    acceleration = compute_acceleration(velocity, fps)          # (T-2, 21, 3)

    # Magnitudes
    disp_mag = compute_magnitude(displacement)                  # (T-1, 21)
    vel_mag = compute_magnitude(velocity)                       # (T-1, 21)
    accel_mag = compute_magnitude(acceleration)                 # (T-2, 21)

    # Jerk (if enough frames)
    if T >= 5:
        jerk = compute_jerk(acceleration, fps)                  # (T-3, 21, 3)
        jerk_mag = compute_magnitude(jerk)                      # (T-3, 21)
    else:
        jerk_mag = np.zeros((max(0, T - 4), 21))

    # Aggregate signals (mean across fingertips)
    fingertip_disp = disp_mag[:, FINGERTIP_INDICES].mean(axis=1)   # (T-1,)
    fingertip_vel = vel_mag[:, FINGERTIP_INDICES].mean(axis=1)     # (T-1,)
    fingertip_accel = accel_mag[:, FINGERTIP_INDICES].mean(axis=1) # (T-2,)
    wrist_disp = disp_mag[:, WRIST_INDEX]                           # (T-1,)
    wrist_vel = vel_mag[:, WRIST_INDEX]                             # (T-1,)

    return {
        "video_id": video_id,
        "fps": fps,
        "total_frames": T,
        "detected_frames": sum(detected_flags),
        "detection_rate": round(detection_rate, 4),
        "landmarks_normalized": landmarks_interp.tolist(),          # (T, 21, 3)
        "displacement_magnitude": disp_mag.tolist(),                # (T-1, 21)
        "velocity_magnitude": vel_mag.tolist(),                     # (T-1, 21)
        "acceleration_magnitude": accel_mag.tolist(),               # (T-2, 21)
        "jerk_magnitude": jerk_mag.tolist(),                        # (T-3, 21)
        "aggregate_signals": {
            "fingertip_displacement": fingertip_disp.tolist(),
            "fingertip_velocity": fingertip_vel.tolist(),
            "fingertip_acceleration": fingertip_accel.tolist(),
            "wrist_displacement": wrist_disp.tolist(),
            "wrist_velocity": wrist_vel.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Box-normalize landmarks and extract tremor signals")
    parser.add_argument("--landmarks", required=True, help="Directory with per-video landmark .json files")
    parser.add_argument("--out", required=True, help="Output directory for signals")
    parser.add_argument("--min-detection", type=float, default=0.1, help="Minimum detection rate to process")
    args = parser.parse_args()

    landmarks_dir = Path(args.landmarks).resolve()
    out_dir = Path(args.out).resolve()

    if not landmarks_dir.exists():
        print(f"ERROR: Landmarks directory not found: {landmarks_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find landmark files
    landmark_files = sorted(landmarks_dir.glob("MB_SYNTH_*.json"))
    if not landmark_files:
        print(f"ERROR: No landmark files found in {landmarks_dir}")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Signal Normalization                       ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Input:     {landmarks_dir}")
    print(f"  Output:    {out_dir}")
    print(f"  Files:     {len(landmark_files)}")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for i, lm_file in enumerate(landmark_files):
        video_id = lm_file.stem
        out_file = out_dir / f"{video_id}.json"

        # Load landmarks
        data = json.loads(lm_file.read_text())

        if data.get("detection_rate", 0) < args.min_detection:
            print(f"  [{i+1}/{len(landmark_files)}] {video_id} → SKIP (detection {data.get('detection_rate', 0)*100:.0f}% < {args.min_detection*100:.0f}%)")
            skipped += 1
            continue

        # Process
        result = process_video_landmarks(data)

        if result is None or "error" in result:
            err_msg = result.get("error", "unknown") if result else "no_result"
            print(f"  [{i+1}/{len(landmark_files)}] {video_id} → ERROR: {err_msg}")
            errors += 1
            continue

        # Save (compact JSON to save space)
        out_file.write_text(json.dumps(result, separators=(',', ':')))
        processed += 1

        if (i + 1) % 10 == 0 or i == len(landmark_files) - 1:
            print(f"  [{i+1}/{len(landmark_files)}] {video_id} → OK ({result['detected_frames']}/{result['total_frames']} frames)")

    # Summary
    summary = {
        "total_files": len(landmark_files),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }
    (out_dir / "_normalization_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"✓ Signal normalization complete")
    print(f"  Processed:  {processed}")
    print(f"  Skipped:    {skipped}")
    print(f"  Errors:     {errors}")
    print(f"  Output:     {out_dir}")


if __name__ == "__main__":
    main()
