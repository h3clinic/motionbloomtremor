"""Synthetic Tremor Dataset Generator.

Main orchestrator that combines:
  - Hand rig controller (pose library)
  - Tremor injection engine (per-joint oscillation)
  - Camera engine (multi-view placement)
  - Render pipeline (Blender headless)
  - Label writer (ground-truth metadata)

Usage (inside Blender):
    blender --background --python generate_dataset.py -- \\
        --num-videos 500 --output-dir outputs/dataset_v1

Usage (standalone — spawns Blender subprocesses):
    python generate_dataset.py --num-videos 500 --output-dir outputs/dataset_v1
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


def load_all_configs() -> Dict[str, dict]:
    """Load all configuration files."""
    configs = {}
    for name in ["poses", "tremor_profiles", "cameras", "render_profiles"]:
        path = CONFIGS_DIR / f"{name}.yaml"
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    return configs


def generate_dataset_plan(
    num_videos: int = 500,
    seed: int = 42,
    quick: bool = False,
) -> List[dict]:
    """Generate the combinatorial plan for the dataset.

    For the 500-video starter:
        5 poses × 5 severity levels × 5 frequencies × 4 cameras = 500

    Args:
        num_videos: Target number of videos.
        seed: Random seed.
        quick: If True, generate minimal 500-video starter set.

    Returns:
        List of dicts, each specifying one video's parameters.
    """
    random.seed(seed)
    configs = load_all_configs()

    if quick:
        # Deterministic 500-video starter
        poses = ["REST_OPEN", "PINCH", "POINTING", "PALM_UP", "GRIP_RELEASE"]
        severities = [0, 20, 40, 60, 80]
        frequencies = [4.0, 5.0, 6.0, 7.0, 8.0]
        cameras = ["front", "front_oblique", "side_right", "top_down"]
        tremor_types = ["POSTURAL_TREMOR"]
        motions = ["stationary"]
        lightings = ["indoor_soft"]
        backgrounds = ["white_clean"]
        skin_tones = ["medium"]
        degradations = ["none"]
    else:
        # Full combinatorial (will be subset-sampled to num_videos)
        poses = list(configs["poses"]["poses"].keys())
        severities = [0, 5, 10, 20, 35, 50, 65, 80, 95]
        frequencies = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
        cameras = list(configs["cameras"]["cameras"].keys())
        tremor_types = [
            "POSTURAL_TREMOR", "KINETIC_TREMOR", "REST_TREMOR",
            "FINGER_TREMOR", "WRIST_TREMOR",
            "TRACKING_ARTIFACT", "GROSS_HAND_MOVEMENT_NO_TREMOR",
        ]
        motions = list(configs["cameras"]["motion_types"].keys())
        lightings = list(configs["render_profiles"]["lighting"].keys())
        backgrounds = list(configs["render_profiles"]["backgrounds"].keys())
        skin_tones = list(configs["render_profiles"]["skin_tones"].keys())
        degradations = list(configs["render_profiles"]["degradation"].keys())

    plan = []
    video_idx = 0

    if quick:
        # Exhaustive grid for starter set
        for pose, sev, freq, cam in itertools.product(
            poses, severities, frequencies, cameras
        ):
            video_idx += 1
            plan.append({
                "video_id": f"MB_SYNTH_{video_idx:06d}",
                "pose": pose,
                "tremor_type": "POSTURAL_TREMOR" if sev > 0 else "GROSS_HAND_MOVEMENT_NO_TREMOR",
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
        # Random sampling from full space
        while len(plan) < num_videos:
            video_idx += 1
            severity = random.choice(severities)
            tremor_type = random.choice(tremor_types)

            # Override severity for non-tremor types
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
                "skin_tone": random.choice(skin_tones),
                "degradation": random.choice(degradations),
            })

    return plan[:num_videos]


def render_single_video_blender(video_spec: dict, output_dir: Path):
    """Render a single video inside Blender's Python environment.

    This function is called when running inside Blender (--background mode).
    """
    # These imports only work inside Blender
    from . import hand_rig, tremor_engine, camera_engine, render_blender, label_writer

    configs = load_all_configs()
    render_config = configs["render_profiles"]["render"]
    fps = render_config["fps"]
    duration = render_config["duration_sec"]

    # 1. Import hand model
    armature = hand_rig.import_hand_model()
    rig_map = hand_rig.load_rig_map()

    # 2. Load base pose
    poses = hand_rig.load_poses()
    base_pose = poses.get(video_spec["pose"], poses["REST_OPEN"])

    # 3. Create tremor profile
    profile = tremor_engine.create_tremor_profile(
        tremor_type=video_spec["tremor_type"],
        severity_score=video_spec["severity_target"],
        frequency_hz=video_spec["frequency_hz"],
        seed=hash(video_spec["video_id"]) % (2**31),
    )

    # 4. Generate animation frames (base pose + tremor injection)
    animation = tremor_engine.generate_animation_curve(
        profile=profile,
        base_pose=base_pose,
        duration_sec=duration,
        fps=fps,
    )

    # 5. Apply animation to armature
    hand_rig.apply_animation_to_armature(armature, animation, rig_map, fps=fps)

    # 6. Apply gross hand motion (if any)
    camera_engine.apply_hand_motion(
        armature, video_spec["motion_type"], duration, fps
    )

    # 7. Set up camera
    camera_engine.setup_camera_blender(video_spec["camera_angle"])

    # 8. Set up lighting and materials
    render_blender.setup_scene(
        resolution=render_config["resolution"],
        fps=fps,
        duration_sec=duration,
        background=video_spec["background"],
    )
    render_blender.setup_lighting(video_spec["lighting"])

    # Apply skin tone to mesh objects
    import bpy
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            render_blender.setup_skin_material(obj, video_spec["skin_tone"])

    # 9. Render video
    video_path = output_dir / "videos" / f"{video_spec['video_id']}.mp4"
    render_blender.render_video(video_path)

    # 10. Apply degradation (post-render)
    render_blender.apply_degradation(video_path, video_spec["degradation"])

    # 11. Write metadata
    metadata = label_writer.create_metadata(
        video_id=video_spec["video_id"],
        pose=video_spec["pose"],
        tremor_type=video_spec["tremor_type"],
        frequency_hz=profile.frequency_hz,
        amplitude_degrees=profile.amplitude_degrees,
        affected_joints=profile.affected_joints,
        camera_angle=video_spec["camera_angle"],
        motion_type=video_spec["motion_type"],
        lighting=video_spec["lighting"],
        background=video_spec["background"],
        skin_tone=video_spec["skin_tone"],
        degradation=video_spec["degradation"],
        fps=fps,
        duration_sec=duration,
    )
    label_writer.write_metadata_json(metadata, output_dir)

    return metadata


def spawn_blender_render(video_spec: dict, output_dir: Path, blender_bin: str):
    """Spawn Blender subprocess to render one video.

    Used when running the generator from outside Blender.
    """
    script_path = Path(__file__).resolve()
    spec_json = json.dumps(video_spec)

    cmd = [
        blender_bin,
        "--background",
        "--python", str(script_path),
        "--",
        "--single-video",
        "--spec", spec_json,
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  ERROR rendering {video_spec['video_id']}: {result.stderr[-500:]}")
        return None

    return video_spec["video_id"]


def find_blender() -> str:
    """Find Blender executable."""
    import shutil

    # Check common locations
    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"),
        "blender",  # PATH lookup
    ]

    for path in candidates:
        if Path(path).exists() or shutil.which(path):
            return path

    raise FileNotFoundError(
        "Blender not found. Install Blender or set BLENDER_BIN environment variable."
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Synthetic Tremor Dataset Generator")
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Generate 500-video starter (deterministic grid)")
    parser.add_argument("--blender-bin", type=str, default=None)
    parser.add_argument("--plan-only", action="store_true",
                        help="Only generate plan JSON, don't render")
    # Internal: called by spawned Blender subprocess
    parser.add_argument("--single-video", action="store_true")
    parser.add_argument("--spec", type=str, default=None)

    # Parse args after -- (Blender passes everything before -- to itself)
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Single video mode (called by Blender subprocess) ---
    if args.single_video and args.spec:
        spec = json.loads(args.spec)
        render_single_video_blender(spec, output_dir)
        return

    # --- Plan generation ---
    print(f"Generating dataset plan: {args.num_videos} videos (seed={args.seed})")
    plan = generate_dataset_plan(
        num_videos=args.num_videos,
        seed=args.seed,
        quick=args.quick,
    )

    # Save plan
    plan_path = output_dir / "generation_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved: {plan_path} ({len(plan)} videos)")

    if args.plan_only:
        print("Plan-only mode. Exiting.")
        return

    # --- Render all videos ---
    blender_bin = args.blender_bin or find_blender()
    print(f"Using Blender: {blender_bin}")

    from .label_writer import write_labels_csv, write_metadata_jsonl, create_metadata

    all_metadata = []
    for i, spec in enumerate(plan):
        print(f"[{i+1}/{len(plan)}] Rendering {spec['video_id']} "
              f"(pose={spec['pose']}, sev={spec['severity_target']}, "
              f"freq={spec['frequency_hz']}Hz, cam={spec['camera_angle']})")

        result = spawn_blender_render(spec, output_dir, blender_bin)
        if result:
            # Read back metadata
            meta_path = output_dir / f"{spec['video_id']}.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta_dict = json.load(f)
                # Reconstruct for CSV/JSONL writing
                from .label_writer import VideoMetadata
                meta = VideoMetadata(**meta_dict)
                all_metadata.append(meta)

    # Write aggregate label files
    write_labels_csv(all_metadata, output_dir / "labels.csv")
    write_metadata_jsonl(all_metadata, output_dir / "metadata.jsonl")

    print(f"\nDataset generation complete!")
    print(f"  Videos: {output_dir / 'videos'}")
    print(f"  Labels: {output_dir / 'labels.csv'}")
    print(f"  Metadata: {output_dir / 'metadata.jsonl'}")
    print(f"  Total: {len(all_metadata)} videos rendered successfully")


if __name__ == "__main__":
    main()
