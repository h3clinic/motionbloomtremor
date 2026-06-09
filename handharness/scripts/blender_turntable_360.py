#!/usr/bin/env python3
"""Render a 360-degree turntable of the original rigged hand (rest pose).

Usage:
  blender -b --python scripts/blender_turntable_360.py -- \
    --input /absolute/path/to/hand.glb \
    --output-dir output/turntable_360 \
    --views 32

Imports the model in its REST pose (animation cleared, no posing), orbits a
camera 360 degrees in N evenly spaced steps, and writes one PNG per view.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--elevation", type=float, default=12.0, help="camera elevation in degrees")
    parser.add_argument("--resolution", type=int, default=1024)
    return parser.parse_args(argv)


def rad(v: float) -> float:
    return math.radians(v)


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


def clear_all_animation():
    # Keep the model in its rest pose: strip baked animation everywhere.
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        if obj.type == "ARMATURE" and obj.data is not None:
            obj.data.animation_data_clear()
            obj.data.pose_position = "REST"
        if obj.type == "MESH" and obj.data is not None:
            sk = getattr(obj.data, "shape_keys", None)
            if sk is not None:
                sk.animation_data_clear()


def mesh_world_bounds():
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    return points


def compute_center_radius():
    points = mesh_world_bounds()
    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (min_v + max_v) * 0.5
    size = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)
    return center, max(size, 1e-3)


def make_target(center: Vector):
    target = bpy.data.objects.new("OrbitTarget", None)
    target.location = center
    bpy.context.scene.collection.objects.link(target)
    return target


def make_camera(target):
    cam_data = bpy.data.cameras.new("TurntableCam")
    cam = bpy.data.objects.new("TurntableCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return cam


def make_lights(center: Vector, radius: float):
    sun = bpy.data.lights.new(name="KeySun", type="SUN")
    sun.energy = 3.0
    sun_obj = bpy.data.objects.new("KeySun", sun)
    sun_obj.rotation_euler = (rad(55), 0.0, rad(35))
    bpy.context.scene.collection.objects.link(sun_obj)

    area = bpy.data.lights.new(name="FillArea", type="AREA")
    area.energy = max(800.0, radius * 1200.0)
    area.size = radius * 3.0
    area_obj = bpy.data.objects.new("FillArea", area)
    area_obj.location = (center.x, center.y - radius * 2.5, center.z + radius * 2.0)
    bpy.context.scene.collection.objects.link(area_obj)


def setup_world():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("TurntableWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.12, 0.12, 0.13, 1.0)
        bg.inputs[1].default_value = 1.2


def apply_preview_material():
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mat = bpy.data.materials.new(name=f"PreviewMat_{obj.name}")
        mat.use_nodes = True
        principled = mat.node_tree.nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs[0].default_value = (0.77, 0.62, 0.52, 1.0)
            if len(principled.inputs) > 7:
                principled.inputs[7].default_value = 0.4
        if len(obj.data.materials) > 0:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def configure_render(resolution: int):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False


def render_turntable(out_dir: Path, views: int, center: Vector, radius: float, elevation_deg: float):
    target = make_target(center)
    cam = make_camera(target)

    distance = max(1.6, radius * 2.6)
    elev = rad(elevation_deg)
    z = center.z + distance * math.sin(elev)
    horiz = distance * math.cos(elev)

    names = []
    for i in range(views):
        theta = (2.0 * math.pi) * (i / views)
        cam.location = (
            center.x + horiz * math.sin(theta),
            center.y - horiz * math.cos(theta),
            z,
        )
        name = f"view_{i:02d}"
        bpy.context.scene.render.filepath = str(out_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        names.append(name)
        print(f"Rendered {name} at {math.degrees(theta):.1f} deg")
    return names


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.glob("*.png"):
        child.unlink()

    reset_scene()
    import_model(args.input)
    clear_all_animation()

    center, radius = compute_center_radius()
    setup_world()
    apply_preview_material()
    make_lights(center, radius)
    configure_render(args.resolution)

    names = render_turntable(out_dir, args.views, center, radius, args.elevation)
    print(f"Wrote {len(names)} turntable views to {out_dir}")


if __name__ == "__main__":
    main()
