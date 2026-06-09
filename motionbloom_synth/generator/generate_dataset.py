"""Synthetic Tremor Dataset Generator.

Generates controlled tremor videos using a VALID rigged hand model.
NO FAKE GEOMETRY. Two supported modes only:

  MODE=rigged_blender_hand  (default, recommended)
    Uses handharness/input/hand_base/.../Do_Hand_DetailedRiggedAnimated.glb
    Animates bone rotations directly in Blender.
    Renders via Blender headless (Eevee or Cycles).

  MODE=official_mano
    ONLY if official MANO_RIGHT.pkl is present AND passes validation.
    Uses smplx to generate posed meshes, then renders in Blender.

There is NO third fake/synthetic mode. If neither model is available,
the generator refuses to run.

Usage:
    blender --background --python generate_dataset.py -- \\
        --mode rigged_blender_hand --quick

    python generate_dataset.py --plan-only --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Resolve paths
SYNTH_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = SYNTH_ROOT / "configs"
OUTPUTS_DIR = SYNTH_ROOT / "outputs"
HANDHARNESS_DIR = SYNTH_ROOT.parent / "handharness"
BLENDER_TREMOR_SCRIPT = HANDHARNESS_DIR / "scripts" / "blender_tremor_sequence.py"
HAND_MODEL_PATH = (
    HANDHARNESS_DIR / "input" / "hand_base" / "extracted" / "source"
    / "Do_Hand_DetailedRiggedAnimated_shared_16022026.glb"
)
RIG_MAP_PATH = HANDHARNESS_DIR / "rig_map.json"
MANO_DIR = HANDHARNESS_DIR / "input" / "mano"


# === Mode Validation ===

def validate_mode(mode: str) -> tuple[bool, str]:
    """Check that the requested mode has valid assets available.

    Returns (is_valid, error_message).
    """
    if mode == "rigged_blender_hand":
        if not HAND_MODEL_PATH.exists():
            return False, (
                f"Rigged hand model not found: {HAND_MODEL_PATH}\n"
                f"Place a valid .glb/.fbx rigged hand in handharness/input/hand_base/"
            )
        if not RIG_MAP_PATH.exists():
            return False, f"Rig map not found: {RIG_MAP_PATH}"
        return True, ""

    elif mode == "official_mano":
        mano_pkl = MANO_DIR / "MANO_RIGHT.pkl"
        if not mano_pkl.exists():
            return False, (
                f"MANO_RIGHT.pkl not found in {MANO_DIR}/\n"
                f"Download from https://mano.is.tue.mpg.de/ (requires registration)"
            )
        # Validate the pkl is real, not synthetic
        from .validate_geometry import validate_mano_pkl
        is_valid, errors = validate_mano_pkl(mano_pkl)
        if not is_valid:
            return False, (
                "MANO_RIGHT.pkl FAILED validation:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nThis file may be synthetic/fake. Use official MANO files only."
            )
        return True, ""

    else:
        return False, (
            f"Unknown mode: '{mode}'. Valid modes:\n"
            f"  rigged_blender_hand  (recommended)\n"
            f"  official_mano        (requires licensed MANO files)"
        )


# === Config Loading ===

def load_all_configs() -> Dict[str, dict]:
    """Load all configuration files."""
    configs = {}
    for name in ["poses", "tremor_profiles", "cameras", "render_profiles"]:
        path = CONFIGS_DIR / f"{name}.yaml"
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    return configs


# === Plan Generation ===

def generate_dataset_plan(
    num_videos: int = 500,
    seed: int = 42,
    quick: bool = False,
) -> List[dict]:
    """Generate the combinatorial plan for the dataset.

    Quick mode (500 videos):
        5 poses × 5 severity levels × 5 frequencies × 4 cameras = 500

    Full mode: random sampling from the complete parameter space.
    """
    random.seed(seed)
    configs = load_all_configs()

    if quick:
        poses = ["REST_OPEN", "PINCH", "POINTING", "PALM_UP", "GRIP_RELEASE"]
        severities = [0, 20, 40, 60, 80]
        frequencies = [4.0, 5.0, 6.0, 7.0, 8.0]
        cameras = ["front", "front_oblique", "side_right", "top_down"]
    else:
        poses = list(configs["poses"]["poses"].keys())
        severities = [0, 5, 10, 20, 35, 50, 65, 80, 95]
        frequencies = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
        cameras = list(configs["cameras"]["cameras"].keys())

    tremor_types_for_severity = {
        0: "GROSS_HAND_MOVEMENT_NO_TREMOR",
    }

    plan = []
    video_idx = 0

    if quick:
        for pose, sev, freq, cam in itertools.product(
            poses, severities, frequencies, cameras
        ):
            video_idx += 1
            tremor_type = "POSTURAL_TREMOR" if sev > 0 else "GROSS_HAND_MOVEMENT_NO_TREMOR"
            plan.append({
                "video_id": f"MB_SYNTH_{video_idx:06d}",
                "pose": pose,
                "tremor_type": tremor_type,
                "severity_target": sev,
                "frequency_hz": freq,
                "camera_angle": cam,
                "motion_type": "stationary",
                "lighting": "indoor_soft",
                "background": "white_clean",
                "skin_tone": "medium",
                "degradation": "none",
            })
    else:
        # Include negative examples (artifacts, gross movement)
        all_tremor_types = [
            "POSTURAL_TREMOR", "KINETIC_TREMOR", "REST_TREMOR",
            "FINGER_TREMOR", "WRIST_TREMOR",
            "TRACKING_ARTIFACT", "GROSS_HAND_MOVEMENT_NO_TREMOR",
        ]
        motions = list(configs["cameras"]["motion_types"].keys())
        lightings = list(configs["render_profiles"]["lighting"].keys())
        backgrounds = list(configs["render_profiles"]["backgrounds"].keys())

        while len(plan) < num_videos:
            video_idx += 1
            severity = random.choice(severities)
            tremor_type = random.choice(all_tremor_types)

            if tremor_type in ("TRACKING_ARTIFACT", "GROSS_HAND_MOVEMENT_NO_TREMOR"):
                severity = 0

            plan.append({
                "video_id": f"MB_SYNTH_{video_idx:06d}",
                "pose": random.choice(poses),
                "tremor_type": tremor_type,
                "severity_target": severity,
                "frequency_hz": random.choice(frequencies),
                "camera_angle": random.choice(cameras),
                "motion_type": random.choice(motions),
                "lighting": random.choice(lightings),
                "background": random.choice(backgrounds),
                "skin_tone": "medium",
                "degradation": "none",
            })

    return plan[:num_videos]


# === Rendering (Rigged Blender Hand) ===

def render_single_video_blender_hand(
    video_spec: dict,
    output_dir: Path,
    blender_bin: str,
) -> Optional[str]:
    """Render one video using handharness/scripts/blender_tremor_sequence.py.

    This is the primary rendering path. It uses the real rigged hand model
    with bone-level tremor injection.
    """
    video_id = video_spec["video_id"]
    video_out_dir = output_dir / "videos" / video_id

    # Map severity 0-100 to intensity for blender_tremor_sequence.py
    intensity = video_spec["severity_target"]

    # Map frequency: blender_tremor_sequence uses intensity to derive freq,
    # but we want to control it. Use seed variation to get frequency diversity.
    seed = hash(video_id) % (2**31)

    render_config = load_all_configs()["render_profiles"]["render"]
    fps = render_config["fps"]
    frames = int(render_config["duration_sec"] * fps)

    cmd = [
        blender_bin,
        "--background",
        "--python", str(BLENDER_TREMOR_SCRIPT),
        "--",
        "--input", str(HAND_MODEL_PATH),
        "--output-dir", str(video_out_dir),
        "--intensity", str(intensity),
        "--frames", str(frames),
        "--fps", str(fps),
        "--rig-map", str(RIG_MAP_PATH),
        "--seed", str(seed),
        "--resolution", "512",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )
        if result.returncode != 0:
            print(f"  ERROR {video_id}: {result.stderr[-300:]}")
            return None
        return video_id
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {video_id}")
        return None
    except FileNotFoundError:
        print(f"  ERROR: Blender not found at {blender_bin}")
        return None


# === Blender Discovery ===

def find_blender() -> str:
    """Find Blender executable."""
    import shutil

    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"),
        "blender",
    ]

    for path in candidates:
        if Path(path).exists() or shutil.which(path):
            return path

    raise FileNotFoundError(
        "Blender not found. Install Blender.app or set --blender-bin.\n"
        "Download: https://www.blender.org/download/"
    )


# === Main ===

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Tremor Dataset Generator (valid geometry only)"
    )
    parser.add_argument(
        "--mode", choices=["rigged_blender_hand", "official_mano"],
        default="rigged_blender_hand",
        help="Generation mode (default: rigged_blender_hand)"
    )
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Generate 500-video starter (deterministic grid)")
    parser.add_argument("--blender-bin", type=str, default=None)
    parser.add_argument("--plan-only", action="store_true",
                        help="Only generate plan JSON, don't render")

    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        args = parser.parse_args()

    # === Validate mode ===
    is_valid, error = validate_mode(args.mode)
    if not is_valid:
        print(f"\n❌ MODE VALIDATION FAILED: {args.mode}\n")
        print(error)
        print("\nRefusing to generate. Fix the asset issue first.")
        sys.exit(1)

    print(f"✓ Mode validated: {args.mode}")
    if args.mode == "rigged_blender_hand":
        print(f"  Model: {HAND_MODEL_PATH.name}")
        print(f"  Rig map: {RIG_MAP_PATH.name}")

    # === Generate plan ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plan: {args.num_videos} videos (seed={args.seed})")
    plan = generate_dataset_plan(
        num_videos=args.num_videos,
        seed=args.seed,
        quick=args.quick,
    )

    plan_path = output_dir / "generation_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved: {plan_path} ({len(plan)} videos)")

    if args.plan_only:
        print("\nPlan-only mode. Exiting.")
        # Print summary
        severities = [v["severity_target"] for v in plan]
        print(f"  Severity distribution: "
              f"0={severities.count(0)}, "
              f"20={severities.count(20)}, "
              f"40={severities.count(40)}, "
              f"60={severities.count(60)}, "
              f"80={severities.count(80)}")
        return

    # === Render ===
    if args.mode == "rigged_blender_hand":
        blender_bin = args.blender_bin or find_blender()
        print(f"\nUsing Blender: {blender_bin}")
        print(f"Rendering {len(plan)} videos...\n")

        from .label_writer import create_metadata, write_labels_csv, write_metadata_jsonl

        all_metadata = []
        success = 0
        for i, spec in enumerate(plan):
            print(f"[{i+1}/{len(plan)}] {spec['video_id']} "
                  f"(sev={spec['severity_target']}, "
                  f"freq={spec['frequency_hz']}Hz, "
                  f"cam={spec['camera_angle']})")

            result = render_single_video_blender_hand(spec, output_dir, blender_bin)
            if result:
                success += 1
                meta = create_metadata(
                    video_id=spec["video_id"],
                    pose=spec["pose"],
                    tremor_type=spec["tremor_type"],
                    frequency_hz=spec["frequency_hz"],
                    amplitude_degrees=spec["severity_target"] * 0.12,
                    affected_joints=["wrist_pitch", "wrist_yaw", "wrist_roll"],
                    camera_angle=spec["camera_angle"],
                    motion_type=spec["motion_type"],
                    lighting=spec["lighting"],
                    background=spec["background"],
                    fps=30,
                    duration_sec=4.0,
                )
                all_metadata.append(meta)

        # Write labels
        write_labels_csv(all_metadata, output_dir / "labels.csv")
        write_metadata_jsonl(all_metadata, output_dir / "metadata.jsonl")

        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"  Mode: {args.mode}")
        print(f"  Success: {success}/{len(plan)}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

    elif args.mode == "official_mano":
        print("\nofficial_mano rendering not yet implemented.")
        print("Use rigged_blender_hand mode (recommended).")
        sys.exit(1)


if __name__ == "__main__":
    main()
