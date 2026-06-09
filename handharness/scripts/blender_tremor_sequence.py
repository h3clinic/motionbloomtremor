#!/usr/bin/env python3
"""Bake and render a tremor animation sequence for a given intensity (1–100).

Usage:
  blender -b --python scripts/blender_tremor_sequence.py -- \
    --input  input/hand_base/.../hand.glb \
    --output-dir  output/tremor_dataset/intensity_050 \
    --intensity  50 \
    --frames  60 \
    --fps  60 \
    --rig-map  rig_map.json \
    --seed  42

Outputs:
  frame_0000.png … frame_NNNN.png  (1024×1024 each)
  metadata.json                     (intensity, physical params, per-frame signal)
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


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Pass script args after --")
    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--intensity",   type=float, default=50.0,
                        help="Tremor intensity on 1-100 clinical scale")
    parser.add_argument("--frames",      type=int,   default=60,
                        help="Total frames to render (60 frames @ 60fps = 1 sec)")
    parser.add_argument("--fps",         type=int,   default=60)
    parser.add_argument("--rig-map",     default="rig_map.json")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="RNG seed for reproducible micro-irregularity noise")
    parser.add_argument("--resolution",  type=int,   default=1024)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene helpers (shared logic with existing pipeline)
# ---------------------------------------------------------------------------

def rad(v: float) -> float:
    return math.radians(v)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise RuntimeError(f"Unsupported model extension: {ext}")


def find_armature(rig_map=None):
    preferred = (rig_map or {}).get("armature_name", "")
    if preferred:
        obj = bpy.data.objects.get(preferred)
        if obj and obj.type == "ARMATURE":
            return obj
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found in model")


def load_rig_map(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def clear_all_animation(armature):
    armature.animation_data_clear()
    if armature.data:
        armature.data.animation_data_clear()
        armature.data.pose_position = "POSE"
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        if obj.type == "MESH" and obj.data:
            sk = getattr(obj.data, "shape_keys", None)
            if sk:
                sk.animation_data_clear()
    for pbone in armature.pose.bones:
        for con in list(pbone.constraints):
            pbone.constraints.remove(con)


def set_pose_mode(armature):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")


def reset_and_freeze_pose(armature, wrist_name: str):
    """Set all bones to identity and insert keyframe at frame 0 to freeze them."""
    for pbone in armature.pose.bones:
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        pbone.location = (0.0, 0.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)
        if pbone.name != wrist_name:
            # Freeze non-wrist bones at identity for the whole sequence.
            pbone.keyframe_insert(data_path="rotation_euler", frame=0)


# ---------------------------------------------------------------------------
# Tremor math
# ---------------------------------------------------------------------------

def clinical_bin(intensity: float):
    """Return clinical band + coarse score references for a 0-100 intensity."""
    i = max(0.0, min(100.0, intensity))
    if i <= 0:
        return {
            "classification": "Absent (Normal)",
            "disp_min_cm": 0.0,
            "disp_max_cm": 0.0,
            "freq_band_hz": [0.0, 0.0],
            "mds_updrs_equivalent": 0,
            "tetras_equivalent": 0.0,
        }
    if i <= 15:
        return {
            "classification": "Slight",
            "disp_min_cm": 0.0,
            "disp_max_cm": 0.5,
            "freq_band_hz": [7.5, 11.0],
            "mds_updrs_equivalent": 1,
            "tetras_equivalent": 1.0,
        }
    if i <= 35:
        return {
            "classification": "Mild",
            "disp_min_cm": 0.5,
            "disp_max_cm": 1.5,
            "freq_band_hz": [6.5, 8.0],
            "mds_updrs_equivalent": 2,
            "tetras_equivalent": 2.0,
        }
    if i <= 65:
        return {
            "classification": "Moderate",
            "disp_min_cm": 1.5,
            "disp_max_cm": 4.5,
            "freq_band_hz": [5.5, 7.0],
            "mds_updrs_equivalent": 3,
            "tetras_equivalent": 3.0,
        }
    if i <= 85:
        return {
            "classification": "Marked",
            "disp_min_cm": 4.5,
            "disp_max_cm": 8.0,
            "freq_band_hz": [4.5, 6.0],
            "mds_updrs_equivalent": 4,
            "tetras_equivalent": 4.0,
        }
    return {
        "classification": "Severe / Extreme",
        "disp_min_cm": 8.0,
        "disp_max_cm": 12.0,
        "freq_band_hz": [3.5, 5.0],
        "mds_updrs_equivalent": 4,
        "tetras_equivalent": 4.5,
    }


def tremor_params(intensity: float):
    """Map intensity to physical parameters used by the synthetic generator."""
    i = max(0.0, min(100.0, intensity))
    max_angle_rad = 0.15
    amplitude_rad = (i / 100.0) * max_angle_rad
    amplitude_cm = i * 0.12
    frequency_hz = 10.0 - (i / 100.0) * 6.0
    return amplitude_rad, frequency_hz, amplitude_cm


def compute_tremor_rotation(intensity: float, frame: int, fps: int, rng: random.Random):
    """Return (rx, ry, rz) rotational offsets in radians for this frame."""
    amplitude, frequency, _ = tremor_params(intensity)
    t = frame / fps

    # Primary wrist axes: flexion/extension (x), pronation/supination (y), radial deviation (z)
    # Secondary harmonics at 2× the base frequency → pathological character
    primary = math.sin(2 * math.pi * frequency * t)
    secondary = 0.18 * math.sin(4 * math.pi * frequency * t + 0.5)
    vib_x = amplitude * (primary + secondary)
    vib_y = amplitude * math.sin(2 * math.pi * frequency * t + 1.2)
    vib_z = amplitude * math.sin(2 * math.pi * frequency * t + 2.5)

    # Micro-irregularity: simulates muscle-firing noise / inter-cycle variability
    vib_x += rng.uniform(-0.05, 0.05) * amplitude
    vib_y += rng.uniform(-0.05, 0.05) * amplitude

    return vib_x, vib_y, vib_z


# ---------------------------------------------------------------------------
# Keyframe baking
# ---------------------------------------------------------------------------

def bake_tremor_keyframes(armature, wrist_name: str, intensity: float,
                          n_frames: int, fps: int, seed: int):
    """Insert per-frame rotation keyframes on the wrist bone for the full sequence."""
    rng = random.Random(seed)
    wrist = armature.pose.bones.get(wrist_name)
    if wrist is None:
        print(f"[WARN] Wrist bone '{wrist_name}' not found — skipping tremor")
        return []

    wrist.rotation_mode = "XYZ"

    signal = []
    for f in range(n_frames):
        bpy.context.scene.frame_set(f)
        rx, ry, rz = compute_tremor_rotation(intensity, f, fps, rng)
        wrist.rotation_euler[0] = rx
        wrist.rotation_euler[1] = ry
        wrist.rotation_euler[2] = rz
        wrist.keyframe_insert(data_path="rotation_euler", frame=f)
        signal.append({"frame": f, "rx": round(rx, 6), "ry": round(ry, 6), "rz": round(rz, 6)})

    return signal


# ---------------------------------------------------------------------------
# Camera / lighting / material
# ---------------------------------------------------------------------------

def mesh_world_bounds():
    pts = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    return pts


def setup_scene_for_render(resolution: int):
    scene = bpy.context.scene

    # Camera
    pts = mesh_world_bounds()
    if pts:
        min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        center = (min_v + max_v) * 0.5
        size   = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)
        dist   = max(1.6, size * 2.4)
    else:
        center, dist = Vector((0, 0, 0)), 2.0

    cam_data = bpy.data.cameras.new("TremorCam")
    cam = bpy.data.objects.new("TremorCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (center.x, center.y - dist, center.z + dist * 0.5)
    cam.rotation_euler = (math.radians(72), 0.0, 0.0)
    scene.camera = cam

    # Key sun
    sun_data = bpy.data.lights.new("KeySun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("KeySun", sun_data)
    sun_obj.rotation_euler = (math.radians(55), 0.0, math.radians(35))
    bpy.context.scene.collection.objects.link(sun_obj)

    # Fill area
    area_data = bpy.data.lights.new("Fill", type="AREA")
    area_data.energy = 1200
    area_obj = bpy.data.objects.new("Fill", area_data)
    area_obj.location = (center.x, center.y - dist * 1.3, center.z + dist * 1.1)
    bpy.context.scene.collection.objects.link(area_obj)

    # World
    if scene.world is None:
        scene.world = bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.08, 0.08, 0.09, 1.0)
        bg.inputs[1].default_value = 1.0

    # Skin material
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mat = bpy.data.materials.new(f"SkinMat_{obj.name}")
        mat.use_nodes = True
        p = mat.node_tree.nodes.get("Principled BSDF")
        if p:
            p.inputs[0].default_value = (0.77, 0.62, 0.52, 1.0)
            if len(p.inputs) > 7:
                p.inputs[7].default_value = 0.4
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    # Render settings
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("frame_*.png"):
        p.unlink()

    rig_map = load_rig_map(args.rig_map) if args.rig_map else {}
    wrist_name = rig_map.get("wrist", "")

    reset_scene()
    import_model(args.input)
    armature = find_armature(rig_map)
    clear_all_animation(armature)
    set_pose_mode(armature)
    reset_and_freeze_pose(armature, wrist_name)
    signal = bake_tremor_keyframes(armature, wrist_name, args.intensity,
                                   args.frames, args.fps, args.seed)
    bpy.ops.object.mode_set(mode="OBJECT")

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end   = args.frames - 1
    scene.render.fps  = args.fps

    setup_scene_for_render(args.resolution)

    scene.render.filepath = str(out_dir / "frame_")
    bpy.ops.render.render(animation=True)

    # Write metadata JSON with physical parameters + per-frame signal
    amplitude, frequency, amplitude_cm = tremor_params(args.intensity)
    band = clinical_bin(args.intensity)
    meta = {
        "pose_source": "rest_pose_original_hand",
        "labels": {
            "continuous_tremor_intensity": round(float(args.intensity), 4),
            "physical_metrics": {
                "calculated_peak_amplitude_cm": round(amplitude_cm, 4),
                "calculated_peak_amplitude_rad": round(amplitude, 6),
                "calculated_peak_amplitude_deg": round(math.degrees(amplitude), 4),
                "simulated_frequency_hz": round(frequency, 4),
            },
            "clinical_binned_references": {
                "classification": band["classification"],
                "displacement_band_cm": [band["disp_min_cm"], band["disp_max_cm"]],
                "target_frequency_band_hz": band["freq_band_hz"],
                "mds_updrs_equivalent": band["mds_updrs_equivalent"],
                "tetras_equivalent": band["tetras_equivalent"],
            },
        },
        "generator": {
            "fps": args.fps,
            "frames": args.frames,
            "seed": args.seed,
            "wrist_bone": wrist_name,
        },
        "per_frame_signal": signal,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Tremor sequence done: intensity={args.intensity}, "
        f"A={amplitude:.4f}rad ({amplitude_cm:.2f}cm), f={frequency:.2f}Hz"
    )


if __name__ == "__main__":
    main()
