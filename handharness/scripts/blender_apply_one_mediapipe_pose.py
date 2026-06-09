#!/usr/bin/env python3
"""
Apply one MediaPipe hand pose JSON onto one rigged hand model in Blender.

Usage:
  blender -b --python scripts/blender_apply_one_mediapipe_pose.py -- \
    --input /absolute/path/to/hand.glb \
    --pose-json /absolute/path/to/output/mediapipe/peace_pose.json \
    --output-glb /absolute/path/to/output/mediapipe/peace_wrapped.glb \
    --output-png /absolute/path/to/output/mediapipe/peace_wrapped.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    parser.add_argument("--pose-json", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-png", default="")
    parser.add_argument("--armature-name", default="")
    parser.add_argument("--rig-map", default="")
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


def find_armature(preferred_name: str = ""):
    if preferred_name:
        obj = bpy.data.objects.get(preferred_name)
        if obj and obj.type == "ARMATURE":
            return obj
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found in model")


def load_rig_map(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Rig map not found: {path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload


def clear_animation(armature):
    # The source rig is animated; baked fcurves would override our manual pose.
    armature.animation_data_clear()
    if armature.data is not None:
        armature.data.animation_data_clear()
        armature.data.pose_position = "POSE"
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.animation_data_clear()
            if obj.data is not None and getattr(obj.data, "shape_keys", None) is not None:
                obj.data.shape_keys.animation_data_clear()
    # Remove pose-bone constraints that could re-pose the rig at evaluation time.
    for pbone in armature.pose.bones:
        for con in list(pbone.constraints):
            pbone.constraints.remove(con)


def set_pose_mode(armature):
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")


def reset_pose(armature):
    for pbone in armature.pose.bones:
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = (0.0, 0.0, 0.0)


def deg(v: float) -> float:
    return math.degrees(v)


def rad(v: float) -> float:
    return math.radians(v)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_at(a: Vector, b: Vector, c: Vector) -> float:
    ba = (a - b)
    bc = (c - b)
    if ba.length < 1e-8 or bc.length < 1e-8:
        return 0.0
    return ba.angle(bc)


def pick_finger_bones(armature):
    names = [b.name for b in armature.pose.bones]
    groups = {
        "thumb": [n for n in names if "thumb" in n.lower()],
        "index": [n for n in names if any(k in n.lower() for k in ["index", "fore", "pointer"])],
        "middle": [n for n in names if "middle" in n.lower()],
        "ring": [n for n in names if "ring" in n.lower()],
        "pinky": [n for n in names if any(k in n.lower() for k in ["pinky", "little"])],
    }
    for finger in groups:
        groups[finger] = sorted(groups[finger], key=lambda s: (len(s), s))
    return groups


def pick_finger_bones_from_rig_map(armature, rig_map):
    fingers = (rig_map or {}).get("fingers", {})
    groups = {}
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        names = fingers.get(finger, [])
        valid = [n for n in names if n in armature.pose.bones]
        groups[finger] = valid
    return groups


def pick_wrist_bone(armature):
    candidates = []
    for b in armature.pose.bones:
        n = b.name.lower()
        if any(k in n for k in ["wrist", "hand", "palm"]):
            candidates.append(b.name)
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: (len(s), s))[0]


def pick_wrist_bone_from_rig_map(armature, rig_map):
    if not rig_map:
        return None
    wrist = rig_map.get("wrist", "")
    if wrist and wrist in armature.pose.bones:
        return wrist
    return None


def ensure_parent_dirs(*paths):
    for p in paths:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_landmarks(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lms = payload.get("landmarks", [])
    if len(lms) != 21:
        raise RuntimeError(f"Expected 21 landmarks, got {len(lms)}")
    return [Vector((float(p["x"]), float(p["y"]), float(p["z"]))) for p in lms]


def finger_joint_curls(lms):
    idx = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    out = {}
    for finger, (a, b, c, d) in idx.items():
        ang0 = angle_at(lms[a], lms[b], lms[c])
        ang1 = angle_at(lms[b], lms[c], lms[d])

        # Straight finger => angle near pi => curl near 0.
        curl0 = clamp(deg(math.pi - ang0) * 1.05, 0.0, 95.0)
        curl1 = clamp(deg(math.pi - ang1) * 1.10, 0.0, 100.0)
        curl2 = clamp(curl1 * 0.85, 0.0, 85.0)
        out[finger] = (curl0, curl1, curl2)
    return out


def finger_spreads(lms):
    # Relative to middle MCP to approximate fan spread.
    base_x = lms[9].x
    mcp = {
        "thumb": lms[1],
        "index": lms[5],
        "middle": lms[9],
        "ring": lms[13],
        "pinky": lms[17],
    }
    out = {}
    for finger, p in mcp.items():
        out[finger] = clamp((p.x - base_x) * 180.0, -22.0, 22.0)
    return out


def wrist_euler_deg(lms):
    wrist = lms[0]
    to_middle = (lms[9] - wrist)
    to_index = (lms[5] - wrist)
    to_pinky = (lms[17] - wrist)
    if to_middle.length < 1e-8:
        return (0.0, 0.0, 0.0)
    to_middle.normalize()
    side = (to_pinky - to_index)
    if side.length > 1e-8:
        side.normalize()

    curl = clamp(to_middle.z * 55.0, -14.0, 16.0)
    yaw = clamp(to_middle.x * 70.0, -18.0, 18.0)
    roll = clamp(side.z * 45.0, -10.0, 10.0)
    return (curl, yaw, roll)


def _bone_rest_dir(armature, bone_name):
    """Bone head->tail direction in armature space (rest pose)."""
    b = armature.pose.bones[bone_name].bone
    return (b.matrix_local.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()


def _bone_rest_head(armature, bone_name):
    return armature.pose.bones[bone_name].bone.matrix_local.translation.copy()


def compute_palm_normal(armature, finger_groups, wrist_name):
    """Estimate the palm (dorsal) normal in armature space from rest bone heads."""
    from mathutils import Vector as V

    def first_head(finger):
        bones = finger_groups.get(finger, [])
        return _bone_rest_head(armature, bones[0]) if bones else None

    idx = first_head("index")
    pky = first_head("pinky")
    mid = first_head("middle")
    wrist = _bone_rest_head(armature, wrist_name) if wrist_name else None

    if idx is not None and pky is not None and mid is not None:
        across = (pky - idx)
        along = (mid - (wrist if wrist is not None else idx))
        n = across.cross(along)
        if n.length > 1e-8:
            return n.normalized()
    return V((0.0, 0.0, 1.0))


def flexion_axis_local(armature, bone_name, palm_normal):
    """Local-space axis (in the bone's rest basis) that flexes the finger toward the palm."""
    from mathutils import Quaternion

    bone = armature.pose.bones[bone_name].bone
    rest3 = bone.matrix_local.to_3x3()
    bone_dir = (rest3 @ Vector((0.0, 1.0, 0.0))).normalized()

    axis_world = bone_dir.cross(palm_normal)
    if axis_world.length < 1e-6:
        # Bone parallel to palm normal; fall back to a sane perpendicular.
        axis_world = bone_dir.cross(Vector((1.0, 0.0, 0.0)))
        if axis_world.length < 1e-6:
            axis_world = Vector((1.0, 0.0, 0.0))
    axis_world.normalize()

    # Choose the sign so a positive angle curls the tail toward the palm (-normal/volar side).
    test = Quaternion(axis_world, rad(10.0)) @ bone_dir
    if test.dot(-palm_normal) < bone_dir.dot(-palm_normal):
        axis_world = -axis_world

    local_axis = (rest3.inverted() @ axis_world)
    if local_axis.length > 1e-8:
        local_axis.normalize()
    return local_axis


def apply_pose(armature, lms, rig_map=None):
    from mathutils import Quaternion

    finger_groups = pick_finger_bones_from_rig_map(armature, rig_map) if rig_map else pick_finger_bones(armature)

    # Fallback to auto-matching if rig map is partial or empty.
    if not any(finger_groups.values()):
        finger_groups = pick_finger_bones(armature)

    wrist_name = pick_wrist_bone_from_rig_map(armature, rig_map) if rig_map else None
    if not wrist_name:
        wrist_name = pick_wrist_bone(armature)

    palm_normal = compute_palm_normal(armature, finger_groups, wrist_name)

    curls = finger_joint_curls(lms)
    spreads = finger_spreads(lms)

    for finger, bones in finger_groups.items():
        if not bones:
            continue
        c0, c1, c2 = curls[finger]
        s0 = spreads[finger]
        values = [c0, c1, c2]

        for i, bone_name in enumerate(bones[:3]):
            b = armature.pose.bones[bone_name]
            b.rotation_mode = "QUATERNION"

            flex_axis = flexion_axis_local(armature, bone_name, palm_normal)
            rot = Quaternion(flex_axis, rad(values[i]))

            if i == 0:
                # Spread (abduction) about the palm normal at the knuckle.
                rest3 = b.bone.matrix_local.to_3x3()
                spread_axis = rest3.inverted() @ palm_normal
                if spread_axis.length > 1e-8:
                    spread_axis.normalize()
                    rot = rot @ Quaternion(spread_axis, rad(s0))

            b.rotation_quaternion = rot

    if wrist_name:
        c, y, r = wrist_euler_deg(lms)
        wb = armature.pose.bones[wrist_name]
        wb.rotation_mode = "XYZ"
        wb.rotation_euler[0] = rad(c)
        wb.rotation_euler[1] = rad(y)
        wb.rotation_euler[2] = rad(r)


def ensure_camera_light():
    cam = bpy.data.objects.get("PoseCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("PoseCam")
        cam = bpy.data.objects.new("PoseCam", cam_data)
        bpy.context.scene.collection.objects.link(cam)

    light = bpy.data.objects.get("PoseLight")
    if light is None:
        data = bpy.data.lights.new(name="PoseLight", type="AREA")
        data.energy = 1300
        light = bpy.data.objects.new("PoseLight", data)
        bpy.context.scene.collection.objects.link(light)

    cam.location = (0.0, -1.7, 1.1)
    cam.rotation_euler = (rad(68), 0.0, 0.0)
    light.location = (0.0, -1.2, 1.8)
    bpy.context.scene.camera = cam

    sun_data = bpy.data.lights.get("PreviewSun")
    if sun_data is None:
        sun_data = bpy.data.lights.new(name="PreviewSun", type="SUN")
        sun_data.energy = 2.0
    sun = bpy.data.objects.get("PreviewSun")
    if sun is None:
        sun = bpy.data.objects.new("PreviewSun", sun_data)
        bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (rad(50), 0.0, rad(35))


def mesh_world_bounds():
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    return points


def frame_camera_to_mesh(cam):
    points = mesh_world_bounds()
    if not points:
        return

    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (min_v + max_v) * 0.5
    size = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)
    distance = max(1.6, size * 2.2)

    cam.location = (center.x, center.y - distance, center.z + distance * 0.55)
    cam.rotation_euler = (rad(72), 0.0, 0.0)
    bpy.context.scene.camera = cam


def export_glb(path: str):
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_visible=True,
        export_apply=True,
    )


def render_png(path: str):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    if scene.world is None:
        scene.world = bpy.data.worlds.new("PreviewWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.12, 0.12, 0.13, 1.0)
        bg.inputs[1].default_value = 1.2

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mat = bpy.data.materials.get(f"PreviewMat_{obj.name}")
        if mat is None:
            mat = bpy.data.materials.new(name=f"PreviewMat_{obj.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            principled = nodes.get("Principled BSDF")
            if principled is not None:
                principled.inputs[0].default_value = (0.77, 0.62, 0.52, 1.0)
                if len(principled.inputs) > 7:
                    principled.inputs[7].default_value = 0.35
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    cam = bpy.data.objects.get("PoseCam")
    if cam is not None:
        frame_camera_to_mesh(cam)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    ensure_parent_dirs(args.output_glb, args.output_png)

    rig_map = load_rig_map(args.rig_map) if args.rig_map else None
    preferred_armature = args.armature_name
    if rig_map and not preferred_armature:
        preferred_armature = rig_map.get("armature_name", "")

    reset_scene()
    import_model(args.input)
    armature = find_armature(preferred_armature)
    clear_animation(armature)
    set_pose_mode(armature)
    reset_pose(armature)

    lms = load_landmarks(args.pose_json)
    apply_pose(armature, lms, rig_map=rig_map)

    bpy.ops.object.mode_set(mode="OBJECT")
    export_glb(args.output_glb)

    if args.output_png:
        ensure_camera_light()
        render_png(args.output_png)

    print(f"Wrote posed GLB: {args.output_glb}")
    if args.output_png:
        print(f"Wrote preview PNG: {args.output_png}")


if __name__ == "__main__":
    main()
