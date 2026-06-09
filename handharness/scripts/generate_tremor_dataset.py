#!/usr/bin/env python3
"""Generate a multi-view hand tremor dataset of short temporal clips in Blender headless mode.

Creates N distinct procedural hand poses. For each pose the finger configuration is held
static while the wrist undergoes a phase-consistent tremor oscillation sampled across a
short high-FPS burst (default 16 frames @ 60 FPS ~= 0.27 s). Each frame is rendered from
6 views (4 equatorial + 2 axial), producing real motion sequences instead of phase-corrupted
single frames. Per-frame wrist signal and labels are written to dataset_manifest.json.

Usage:
  blender --background --python scripts/generate_tremor_dataset.py -- \
    --input input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb \
    --rig-map rig_map.json \
    --output-dir output/dataset \
    --num-positions 3000 \
    --clip-frames 16 --clip-fps 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Pass args after --")
    argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--rig-map", default="rig_map.json")
    parser.add_argument("--output-dir", default="output/dataset")
    parser.add_argument("--num-positions", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--camera-radius", type=float, default=0.40)
    parser.add_argument("--target-size", type=float, default=0.20,
                        help="Normalize longest hand dimension to this size in Blender units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--intensity-mode", choices=["random", "uniform"], default="uniform")
    parser.add_argument("--clip-frames", type=int, default=16,
                        help="Temporal frames per pose burst (16 @ 60fps ~= 0.27s, ~1.5-2.5 tremor cycles)")
    parser.add_argument("--clip-fps", type=int, default=60,
                        help="Sampling rate of the temporal burst in frames per second")
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(input_path: str):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_path)
    else:
        raise RuntimeError(f"Unsupported model extension: {ext}")


def load_rig_map(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def find_armature(preferred_name: str = ""):
    if preferred_name:
        obj = bpy.data.objects.get(preferred_name)
        if obj and obj.type == "ARMATURE":
            return obj
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found")


def clear_animation_data(armature):
    armature.animation_data_clear()
    if armature.data is not None:
        armature.data.animation_data_clear()
        armature.data.pose_position = "POSE"

    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        if obj.type == "MESH" and obj.data is not None:
            sk = getattr(obj.data, "shape_keys", None)
            if sk is not None:
                sk.animation_data_clear()

    for pbone in armature.pose.bones:
        for con in list(pbone.constraints):
            pbone.constraints.remove(con)


def ensure_render_scene(resolution: int):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    if scene.world is None:
        scene.world = bpy.data.worlds.new("DatasetWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.10, 0.10, 0.11, 1.0)
        bg.inputs[1].default_value = 1.0

    # Lighting
    key = bpy.data.lights.new("DatasetSun", type="SUN")
    key.energy = 2.8
    key_obj = bpy.data.objects.new("DatasetSun", key)
    key_obj.rotation_euler = (math.radians(55), 0.0, math.radians(35))
    scene.collection.objects.link(key_obj)

    fill = bpy.data.lights.new("DatasetFill", type="AREA")
    fill.energy = 1300
    fill.size = 2.0
    fill_obj = bpy.data.objects.new("DatasetFill", fill)
    fill_obj.location = (0.0, -2.0, 2.0)
    scene.collection.objects.link(fill_obj)


def mesh_bounds_center_size():
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for c in obj.bound_box:
            points.append(obj.matrix_world @ Vector(c))

    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0

    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (min_v + max_v) * 0.5
    size = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)
    return center, max(size, 1e-3)


def create_camera():
    cam_data = bpy.data.cameras.new("DatasetCam")
    cam_data.lens = 70
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1000.0
    cam_obj = bpy.data.objects.new("DatasetCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_camera_position(camera_obj, target_loc: Vector, radius: float, azimuth_deg: float, elevation_deg: float):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    x = target_loc.x + radius * math.cos(elevation) * math.sin(azimuth)
    y = target_loc.y + radius * math.cos(elevation) * math.cos(azimuth)
    z = target_loc.z + radius * math.sin(elevation)

    camera_obj.location = (x, y, z)

    direction = target_loc - camera_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()


def tremor_physical_params(intensity: int):
    i = max(1, min(100, int(intensity)))
    amplitude_rad = (i / 100.0) * 0.15
    amplitude_cm = i * 0.12
    frequency_hz = 10.0 - ((i / 100.0) * 6.0)
    return amplitude_rad, amplitude_cm, frequency_hz


def clinical_bin(intensity: int):
    i = max(0, min(100, int(intensity)))
    if i == 0:
        return "Absent (Normal)", 0, 0.0
    if i <= 15:
        return "Slight", 1, 1.0
    if i <= 35:
        return "Mild", 2, 2.0
    if i <= 65:
        return "Moderate", 3, 3.0
    if i <= 85:
        return "Marked", 4, 4.0
    return "Severe / Extreme", 4, 4.5


def bone_group(name: str):
    n = name.lower()
    if "thumb" in n:
        return "thumb"
    if "index" in n:
        return "index"
    if "midd" in n or "middle" in n:
        return "middle"
    if "ring" in n:
        return "ring"
    if "pinky" in n or "little" in n:
        return "pinky"
    return "other"


def segment_type(name: str):
    n = name.lower()
    if "meta" in n:
        return "meta"
    if "prox" in n:
        return "prox"
    if "midd" in n:
        return "midd"
    if "dist" in n:
        return "dist"
    if "trapez" in n:
        return "trapez"
    return "other"


def apply_random_finger_pose(armature, finger_groups, rng: random.Random):
    """Apply a static random finger configuration. Held fixed across the temporal clip."""
    # Clear transforms first.
    for b in armature.pose.bones:
        b.rotation_mode = "XYZ"
        b.rotation_euler = (0.0, 0.0, 0.0)

    # Pose randomization constrained to vetted finger chains from rig_map.
    for finger, chain in finger_groups.items():
        valid = [n for n in chain if n in armature.pose.bones]
        if not valid:
            continue

        is_thumb = finger == "thumb"
        spread_range = (-0.30, 0.30) if is_thumb else (-0.18, 0.18)

        for idx, bone_name in enumerate(valid[:3]):
            b = armature.pose.bones[bone_name]
            b.rotation_mode = "XYZ"

            # Index in chain approximates meta/prox/midd.
            if is_thumb:
                flex_limits = [(0.05, 0.55), (0.05, 0.80), (0.05, 0.70)]
            else:
                flex_limits = [(0.00, 0.60), (0.00, 1.00), (0.00, 0.90)]

            lo, hi = flex_limits[min(idx, len(flex_limits) - 1)]
            b.rotation_euler[0] = rng.uniform(lo, hi)
            if idx == 0:
                b.rotation_euler[2] = rng.uniform(spread_range[0], spread_range[1])


def compute_wrist_tremor_frame(amp_rad: float, freq_hz: float, t: float, rng: random.Random):
    """Return phase-consistent (x, y, z) wrist rotation in radians for absolute time t.

    Sampling consecutive t values yields a continuous oscillation, so amplitude and
    frequency are recoverable from the rendered sequence (unlike a single random phase).
    """
    p = 2.0 * math.pi * freq_hz * t

    primary = math.sin(p)
    secondary = 0.18 * math.sin(2.0 * p + 0.5)

    x = amp_rad * (primary + secondary)
    y = amp_rad * math.sin(p + 1.2)
    z = amp_rad * math.sin(p + 2.4)

    # Micro-irregularity: inter-cycle muscle-firing variability.
    x += rng.uniform(-0.05, 0.05) * amp_rad
    y += rng.uniform(-0.05, 0.05) * amp_rad

    return x, y, z


def apply_wrist_tremor(armature, wrist_name: str, x: float, y: float, z: float) -> bool:
    """Set the wrist bone rotation for a single frame of the burst."""
    wrist = armature.pose.bones.get(wrist_name)
    if wrist is None:
        return False
    wrist.rotation_mode = "XYZ"
    wrist.rotation_euler[0] = x
    wrist.rotation_euler[1] = y
    wrist.rotation_euler[2] = z
    return True


def choose_intensity(idx: int, total: int, mode: str, rng: random.Random):
    if mode == "random":
        return rng.randint(1, 100)
    # Uniform spread over 1..100 across the full run.
    if total <= 1:
        return 50
    return int(round(1 + (idx / (total - 1)) * 99))


def build_views():
    return {
        "equatorial_front": (0.0, 0.0),
        "equatorial_left": (90.0, 0.0),
        "equatorial_back": (180.0, 0.0),
        "equatorial_right": (270.0, 0.0),
        "axial_superior": (0.0, 90.0),
        "axial_inferior": (0.0, -90.0),
        # Angled high-front view looking down onto the dorsal surface and fingertips.
        "dorsal_fingertip": (0.0, 55.0),
    }


def normalize_model_to_origin(target_size: float):
    """Translate model center to origin and scale longest dimension to target_size.

    IMPORTANT: apply transforms only to root objects to avoid double-transforming
    armature + skinned mesh child hierarchies.
    """
    center, size = mesh_bounds_center_size()
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]

    for obj in roots:
        obj.location = obj.location - center

    bpy.context.view_layer.update()
    _, size_after_center = mesh_bounds_center_size()
    if size_after_center <= 1e-8:
        return {"scale_factor": 1.0, "original_size": size, "normalized_size": size_after_center}

    scale_factor = target_size / size_after_center
    for obj in roots:
        obj.scale = obj.scale * scale_factor

    bpy.context.view_layer.update()
    _, final_size = mesh_bounds_center_size()
    return {
        "scale_factor": float(scale_factor),
        "original_size": float(size),
        "normalized_size": float(final_size),
    }


def write_manifest(path: Path, payload: list):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rig_map = load_rig_map(args.rig_map)

    reset_scene()
    import_model(args.input)

    armature_name = rig_map.get("armature_name", "")
    wrist_name = rig_map.get("wrist", "")
    finger_groups = (rig_map.get("fingers", {}) if rig_map else {})

    armature = find_armature(armature_name)
    clear_animation_data(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    normalization = normalize_model_to_origin(args.target_size)
    ensure_render_scene(args.resolution)
    center, size = mesh_bounds_center_size()
    camera = create_camera()
    radius = max(args.camera_radius, size * 2.0)

    views = build_views()

    manifest = {
        "type": "synthetic_hand_tremor_dataset",
        "format": "temporal_clips",
        "num_positions": args.num_positions,
        "views_per_position": list(views.keys()),
        "clip": {
            "frames": args.clip_frames,
            "fps": args.clip_fps,
            "duration_sec": round(args.clip_frames / max(1, args.clip_fps), 6),
            "filename_pattern": "pose_{pose_id:05d}_{view}_frame_{frame:02d}.png",
        },
        "camera": {
            "single_dynamic_camera": True,
            "radius": radius,
            "target": [round(center.x, 6), round(center.y, 6), round(center.z, 6)],
        },
        "normalization": normalization,
        "generation": {
            "intensity_mode": args.intensity_mode,
            "seed": args.seed,
            "resolution": args.resolution,
            "target_size": args.target_size,
            "clip_frames": args.clip_frames,
            "clip_fps": args.clip_fps,
        },
        "samples": [],
    }

    manifest_path = out_dir / "dataset_manifest.json"

    for pose_id in range(args.num_positions):
        intensity = choose_intensity(pose_id, args.num_positions, args.intensity_mode, rng)
        amp_rad, amp_cm, freq_hz = tremor_physical_params(intensity)

        # Static anatomy: finger pose is fixed for the whole clip; only the wrist oscillates.
        apply_random_finger_pose(armature, finger_groups, rng)
        # Random window offset so poses don't all start at the same phase of the cycle.
        phase_offset_t = rng.uniform(0.0, 10.0)

        cls_name, updrs, tetras = clinical_bin(intensity)

        sample = {
            "pose_id": pose_id,
            "labels": {
                "continuous_tremor_intensity": intensity,
                "physical_metrics": {
                    "calculated_peak_amplitude_cm": round(amp_cm, 4),
                    "simulated_frequency_hz": round(freq_hz, 4),
                },
                "clinical_binned_references": {
                    "classification": cls_name,
                    "mds_updrs_equivalent": updrs,
                    "tetras_equivalent": tetras,
                },
            },
            "tremor_state": {
                "peak_amplitude_rad": round(amp_rad, 6),
                "phase_offset_t": round(phase_offset_t, 6),
                "clip_frames": args.clip_frames,
                "clip_fps": args.clip_fps,
            },
            "per_frame_signal": [],
            "renders": {view_name: [] for view_name in views},
        }

        for f_idx in range(args.clip_frames):
            t = phase_offset_t + f_idx / max(1, args.clip_fps)
            x, y, z = compute_wrist_tremor_frame(amp_rad, freq_hz, t, rng)
            apply_wrist_tremor(armature, wrist_name, x, y, z)
            bpy.context.view_layer.update()

            sample["per_frame_signal"].append({
                "frame": f_idx,
                "t": round(t, 6),
                "wrist_rotation_rad": [round(x, 6), round(y, 6), round(z, 6)],
            })

            for view_name, (azimuth, elevation) in views.items():
                setup_camera_position(camera, center, radius, azimuth, elevation)
                filename = f"pose_{pose_id:05d}_{view_name}_frame_{f_idx:02d}.png"
                filepath = out_dir / filename
                bpy.context.scene.render.filepath = str(filepath)
                bpy.ops.render.render(write_still=True)
                sample["renders"][view_name].append(str(filepath))

        manifest["samples"].append(sample)

        if pose_id % max(1, args.save_every) == 0:
            write_manifest(manifest_path, manifest)
            print(f"Saved checkpoint: pose {pose_id}/{args.num_positions - 1} "
                  f"({args.clip_frames} frames x {len(views)} views)")

    write_manifest(manifest_path, manifest)
    total_renders = args.num_positions * args.clip_frames * len(views)
    print(f"Generation complete. Wrote {args.num_positions} temporal clips "
          f"({total_renders} frames) to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
