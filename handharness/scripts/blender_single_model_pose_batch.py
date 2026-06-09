"""
Single-model hand pose batch generator for Blender.

Usage:
  blender -b --python scripts/blender_single_model_pose_batch.py -- \
    --input /absolute/path/to/hand.glb \
    --output /absolute/path/to/output/poses \
    --count 500 \
    --seed 42 \
    --export-glb 1 \
    --render-png 0

Notes:
- This script uses ONE source hand model only.
- The model should be rigged (armature + hand mesh).
- Bone name matching is keyword-based to support common rigs.
"""

import argparse
import math
import os
import random
import sys

import bpy


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Pass args after --")
    argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-glb", type=int, default=1)
    parser.add_argument("--render-png", type=int, default=0)
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(input_path: str):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=input_path)
    else:
        raise RuntimeError(f"Unsupported model extension: {ext}")


def find_armature():
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found. Use a rigged hand model.")


def bind_meshes_to_armature(armature):
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh found in imported model.")

    for mesh in meshes:
        if mesh.parent is None:
            mesh.parent = armature
    return meshes


def set_pose_mode(armature):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")


def reset_pose(armature):
    for pbone in armature.pose.bones:
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = (0.0, 0.0, 0.0)


def deg(v):
    return math.radians(v)


def pick_finger_bones(armature):
    names = [b.name for b in armature.pose.bones]

    groups = {
        "thumb": [n for n in names if "thumb" in n.lower()],
        "index": [n for n in names if any(k in n.lower() for k in ["index", "fore", "pointer"])],
        "middle": [n for n in names if "middle" in n.lower()],
        "ring": [n for n in names if "ring" in n.lower()],
        "pinky": [n for n in names if any(k in n.lower() for k in ["pinky", "little"])],
    }

    # Keep deterministic ordering from proximal to distal by name length + lexical sort.
    for finger in groups:
        groups[finger] = sorted(groups[finger], key=lambda s: (len(s), s))

    return groups


def apply_finger_pose(armature, finger_groups):
    # Human-ish limits (degrees).
    flex = {
        "thumb": [(-10, 25), (0, 50), (0, 70)],
        "index": [(-10, 70), (0, 100), (0, 80)],
        "middle": [(-10, 70), (0, 105), (0, 85)],
        "ring": [(-10, 70), (0, 105), (0, 85)],
        "pinky": [(-10, 70), (0, 100), (0, 80)],
    }
    spread = {
        "thumb": (-40, 20),
        "index": (-20, 25),
        "middle": (-10, 10),
        "ring": (-15, 18),
        "pinky": (-20, 25),
    }

    for finger, bones in finger_groups.items():
        if not bones:
            continue

        max_bones = min(3, len(bones))
        spread_deg = random.uniform(spread[finger][0], spread[finger][1])

        for i in range(max_bones):
            b = armature.pose.bones[bones[i]]
            low, high = flex[finger][i]
            curl_deg = random.uniform(low, high)
            # X = bend, Z = spread-like fan for many rigs.
            b.rotation_euler[0] = deg(curl_deg)
            if i == 0:
                b.rotation_euler[2] = deg(spread_deg)


def ensure_output_dirs(base):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "glb"), exist_ok=True)
    os.makedirs(os.path.join(base, "png"), exist_ok=True)


def export_glb(filepath):
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format="GLB",
        use_visible=True,
        export_apply=True,
    )


def ensure_camera():
    cam = bpy.data.objects.get("PoseCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("PoseCam")
        cam = bpy.data.objects.new("PoseCam", cam_data)
        bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (0.0, -1.8, 1.2)
    cam.rotation_euler = (deg(70), 0.0, 0.0)
    return cam


def ensure_light():
    light = bpy.data.objects.get("PoseLight")
    if light is None:
        data = bpy.data.lights.new(name="PoseLight", type="AREA")
        data.energy = 1000
        light = bpy.data.objects.new("PoseLight", data)
        bpy.context.scene.collection.objects.link(light)
    light.location = (0.0, -1.2, 1.8)


def render_png(filepath):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = filepath
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    reset_scene()
    import_model(args.input)

    armature = find_armature()
    bind_meshes_to_armature(armature)
    set_pose_mode(armature)
    finger_groups = pick_finger_bones(armature)

    ensure_output_dirs(args.output)

    if args.render_png:
        ensure_camera()
        ensure_light()

    for i in range(args.count):
        reset_pose(armature)
        apply_finger_pose(armature, finger_groups)

        if args.export_glb:
            out_glb = os.path.join(args.output, "glb", f"pose_{i:05d}.glb")
            export_glb(out_glb)

        if args.render_png:
            out_png = os.path.join(args.output, "png", f"pose_{i:05d}.png")
            render_png(out_png)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"Single-model pose batch complete: {args.output}")


if __name__ == "__main__":
    main()
