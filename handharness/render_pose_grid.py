"""
render_pose_grid.py – Blender final-render pose grid for MotionBloom training.

Produces realistic shaded 3D hand renders with:
- Smooth shading (no triangle/wireframe artifacts)
- Principled BSDF skin material
- Area lights + ambient occlusion
- Perspective camera at multiple angles
- Subdivision surface for smooth geometry
- Shadows + depth

Usage (from repo root):
    python handharness/render_pose_grid.py \
        --asset handharness/input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb \
        --rig-map handharness/rig_map.json \
        --out handharness/output/pose_grid

This script calls Blender in background mode. If Blender is not installed,
it fails with a clear error — no fallback to debug rendering.
"""

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path


def find_blender() -> str:
    """Find Blender executable. Fail if not available."""
    paths = [
        shutil.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/homebrew/bin/blender",
        "/usr/local/bin/blender",
    ]
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    print("ERROR: Blender not found. Cannot produce training-quality renders.")
    print("Install via: brew install --cask blender")
    sys.exit(1)


# Blender Python script (runs inside Blender's Python environment)
BLENDER_SCRIPT = r'''
import bpy
import bmesh
import math
import json
import os
import sys
from pathlib import Path
from mathutils import Euler, Vector

# ─── Parse Args ───────────────────────────────────────────────────────────
argv = sys.argv
argv = argv[argv.index("--") + 1:]
asset_path = argv[0]
rig_map_path = argv[1]
out_dir = argv[2]
skin_tone = argv[3] if len(argv) > 3 else "medium"

os.makedirs(out_dir, exist_ok=True)

with open(rig_map_path) as f:
    rig_map = json.load(f)

# ─── Skin Tone Presets ────────────────────────────────────────────────────
SKIN_TONES = {
    "light":  {"base": (0.93, 0.78, 0.68, 1.0), "subsurface": (0.90, 0.60, 0.45)},
    "medium": {"base": (0.76, 0.57, 0.44, 1.0), "subsurface": (0.70, 0.40, 0.28)},
    "tan":    {"base": (0.62, 0.42, 0.30, 1.0), "subsurface": (0.55, 0.30, 0.18)},
    "dark":   {"base": (0.38, 0.24, 0.16, 1.0), "subsurface": (0.30, 0.15, 0.08)},
}

# ─── Scene Setup ─────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB
bpy.ops.import_scene.gltf(filepath=asset_path)

# Find the mesh and armature by known names
mesh_obj = None
armature_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.name == 'Do_HandObject':
        mesh_obj = obj
    elif obj.type == 'ARMATURE' and obj.name == 'Do_HandRigged':
        armature_obj = obj

# Fallback: largest mesh if names don't match
if not mesh_obj:
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if meshes:
        mesh_obj = max(meshes, key=lambda o: len(o.data.vertices))
if not armature_obj:
    armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if armatures:
        armature_obj = armatures[0]

if not mesh_obj:
    print("FATAL: No mesh found in GLB")
    sys.exit(1)

# Delete any stray objects (Icosphere, etc.)
for obj in list(bpy.context.scene.objects):
    if obj.type == 'MESH' and obj != mesh_obj:
        bpy.data.objects.remove(obj, do_unlink=True)

print(f"Mesh: {mesh_obj.name} ({len(mesh_obj.data.vertices)} verts, {len(mesh_obj.data.polygons)} faces)")
if armature_obj:
    print(f"Armature: {armature_obj.name} ({len(armature_obj.data.bones)} bones)")

# ─── Smooth Shading ──────────────────────────────────────────────────────
# Set all faces to smooth shading
mesh_obj.data.polygons.foreach_set("use_smooth", [True] * len(mesh_obj.data.polygons))
mesh_obj.data.update()

# Recalculate normals (outside)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# Add subdivision surface for smoother geometry
subsurf = mesh_obj.modifiers.new(name="Subdivision", type='SUBSURF')
subsurf.levels = 1
subsurf.render_levels = 2

# Add weighted normals modifier
wnorm = mesh_obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
wnorm.mode = 'FACE_AREA'
wnorm.weight = 50

# ─── Skin Material ───────────────────────────────────────────────────────
# Remove existing materials
mesh_obj.data.materials.clear()

mat = bpy.data.materials.new(name="SkinMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output_node = nodes.new(type='ShaderNodeOutputMaterial')
output_node.location = (400, 0)

bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)

tone = SKIN_TONES.get(skin_tone, SKIN_TONES["medium"])
bsdf.inputs["Base Color"].default_value = tone["base"]
bsdf.inputs["Roughness"].default_value = 0.45  # slightly less rough for better highlights
bsdf.inputs["Specular IOR Level"].default_value = 0.4

# Subsurface scattering for skin realism
bsdf.inputs["Subsurface Weight"].default_value = 0.2
bsdf.inputs["Subsurface Radius"].default_value = tone["subsurface"]

links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
mesh_obj.data.materials.append(mat)

# ─── Compute Mesh Bounding Box FIRST (needed for lighting + camera) ──────
bbox = [mesh_obj.matrix_world @ Vector(v) for v in mesh_obj.bound_box]
bbox_center = sum(bbox, Vector()) / 8
bbox_size = max((max(v[i] for v in bbox) - min(v[i] for v in bbox)) for i in range(3))
cam_distance = bbox_size * 1.5  # hand fills ~60-80% for MediaPipe detection

print(f"  BBox center: {bbox_center}")
print(f"  BBox size: {bbox_size:.1f}")
print(f"  Cam distance: {cam_distance:.1f}")

# ─── Lighting (Sun + Point for reliable illumination) ────────────────────
cx, cy, cz = bbox_center.x, bbox_center.y, bbox_center.z
light_dist = bbox_size * 1.5

# Sun light - directional, distance-independent, reliable
sun_light = bpy.data.lights.new(name="SunKey", type='SUN')
sun_light.energy = 12.0  # strong key for well-lit skin
sun_light.angle = math.radians(15)  # soft shadow edge
sun_light.color = (1.0, 0.97, 0.93)
sun_obj = bpy.data.objects.new("SunKey", sun_light)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = Euler((math.radians(-50), math.radians(15), math.radians(25)))

# Second sun for fill (opposite side, warm)
sun_fill = bpy.data.lights.new(name="SunFill", type='SUN')
sun_fill.energy = 6.0
sun_fill.angle = math.radians(30)
sun_fill.color = (0.90, 0.93, 1.0)
sun_fill_obj = bpy.data.objects.new("SunFill", sun_fill)
bpy.context.collection.objects.link(sun_fill_obj)
sun_fill_obj.rotation_euler = Euler((math.radians(-30), math.radians(-20), math.radians(-15)))

# Third sun from below for shadow fill
sun_under = bpy.data.lights.new(name="SunUnder", type='SUN')
sun_under.energy = 3.0
sun_under.angle = math.radians(45)
sun_under.color = (0.95, 0.95, 1.0)
sun_under_obj = bpy.data.objects.new("SunUnder", sun_under)
bpy.context.collection.objects.link(sun_under_obj)
sun_under_obj.rotation_euler = Euler((math.radians(60), 0, math.radians(-10)))

# Point light near the hand for warmth
point_light = bpy.data.lights.new(name="PointWarm", type='POINT')
point_light.energy = 8000
point_light.color = (1.0, 0.95, 0.90)
point_light.shadow_soft_size = 3.0
point_obj = bpy.data.objects.new("PointWarm", point_light)
bpy.context.collection.objects.link(point_obj)
point_obj.location = (cx, cy - light_dist * 0.6, cz + light_dist * 0.4)

# World/ambient - medium dark for good contrast with lit skin
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.18, 0.18, 0.20, 1.0)
bg.inputs["Strength"].default_value = 1.0

# Add a ground plane behind the hand (gives MediaPipe spatial context)
bpy.ops.mesh.primitive_plane_add(size=bbox_size * 5, location=(cx, cy + bbox_size * 1.2, cz))
ground = bpy.context.active_object
ground.name = "BackPlane"
ground.rotation_euler = Euler((math.radians(90), 0, 0))
ground_mat = bpy.data.materials.new(name="GroundMat")
ground_mat.use_nodes = True
ground_bsdf = ground_mat.node_tree.nodes["Principled BSDF"]
ground_bsdf.inputs["Base Color"].default_value = (0.25, 0.24, 0.23, 1.0)
ground_bsdf.inputs["Roughness"].default_value = 0.9
ground.data.materials.append(ground_mat)

# ─── Render Settings ─────────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 720
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.film_transparent = False

# Eevee settings
scene.eevee.use_shadows = True

# ─── Camera Setup ────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.type = 'PERSP'
cam_data.lens = 50  # standard portrait lens
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Camera angles
CAMERA_ANGLES = [
    {"name": "front",          "azim": 0,   "elev": 5,   "desc": "Front palm view"},
    {"name": "front_oblique",  "azim": 30,  "elev": 15,  "desc": "Front oblique"},
    {"name": "three_quarter",  "azim": 45,  "elev": 20,  "desc": "3/4 perspective"},
    {"name": "side",           "azim": 90,  "elev": 5,   "desc": "Side view"},
    {"name": "back",           "azim": 180, "elev": 5,   "desc": "Back of hand"},
    {"name": "top",            "azim": 0,   "elev": 70,  "desc": "Top down"},
]

# ─── Pose Definitions ────────────────────────────────────────────────────
# All poses include finger splay (Z-rotation) for clear finger separation
POSES = {
    "rest_open": {
        # Slight spread so fingers are clearly separated from front view
        "index_meta": (0, 0, -8), "midd_meta": (0, 0, -3),
        "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 12),
        "thumb_meta": (5, 0, -15),
    },
    "fist": {
        "index_meta": (70, 0, -5), "index_prox": (85, 0, 0), "index_midd": (75, 0, 0),
        "midd_meta": (70, 0, 0), "midd_prox": (85, 0, 0), "midd_midd": (75, 0, 0),
        "ring_meta": (70, 0, 5), "ring_prox": (85, 0, 0), "ring_midd": (75, 0, 0),
        "pinky_meta": (70, 0, 10), "pinky_prox": (85, 0, 0), "pinky_midd": (75, 0, 0),
        "thumb_meta": (30, 0, -20), "thumb_prox": (50, 0, 0), "thumb_dist": (40, 0, 0),
    },
    "pinch": {
        "index_prox": (45, 0, -5), "index_midd": (50, 0, 0),
        "thumb_meta": (20, 0, -30), "thumb_prox": (40, 0, -10), "thumb_dist": (30, 0, 0),
        "midd_meta": (15, 0, 3), "midd_prox": (20, 0, 0),
        "ring_meta": (20, 0, 8), "ring_prox": (25, 0, 0),
        "pinky_meta": (25, 0, 14), "pinky_prox": (30, 0, 0),
    },
    "pointing": {
        "index_meta": (0, 0, -5),  # index stays extended, slightly splayed
        "midd_meta": (70, 0, 3), "midd_prox": (85, 0, 0), "midd_midd": (75, 0, 0),
        "ring_meta": (70, 0, 8), "ring_prox": (85, 0, 0), "ring_midd": (75, 0, 0),
        "pinky_meta": (70, 0, 12), "pinky_prox": (85, 0, 0), "pinky_midd": (75, 0, 0),
        "thumb_meta": (30, 0, -15), "thumb_prox": (40, 0, 0),
    },
    "palm_up": {
        "radius_ulna": (0, 0, 180),
        "index_meta": (0, 0, -8), "midd_meta": (0, 0, -3),
        "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 12),
    },
    "palm_down": {
        "index_meta": (0, 0, -8), "midd_meta": (0, 0, -3),
        "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 12),
    },
    "wrist_flexion": {
        "radius_ulna": (45, 0, 0),
        "index_meta": (0, 0, -6), "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 10),
    },
    "wrist_extension": {
        "radius_ulna": (-35, 0, 0),
        "index_meta": (0, 0, -6), "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 10),
    },
}

# ─── Pose Application ────────────────────────────────────────────────────
def apply_pose(armature, pose_rotations):
    """Apply bone rotations for a specific pose."""
    if not armature:
        return
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Reset all bones
    for pbone in armature.pose.bones:
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler = (0, 0, 0)
    
    # Apply pose rotations
    for bone_name, (rx, ry, rz) in pose_rotations.items():
        if bone_name in armature.pose.bones:
            pbone = armature.pose.bones[bone_name]
            pbone.rotation_mode = 'XYZ'
            pbone.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()


def set_camera(azim_deg, elev_deg, distance, target):
    """Position camera at given spherical coordinates looking at target."""
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    
    x = target.x + distance * math.cos(elev) * math.sin(azim)
    y = target.y - distance * math.cos(elev) * math.cos(azim)
    z = target.z + distance * math.sin(elev)
    
    cam_obj.location = (x, y, z)
    
    # Point at target
    direction = target - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


# ─── Render Loop ─────────────────────────────────────────────────────────
manifest = {
    "model": os.path.basename(asset_path),
    "skin_tone": skin_tone,
    "render_engine": "EEVEE",
    "resolution": "512x512",
    "poses": [],
}

total_renders = len(POSES) * len(CAMERA_ANGLES)
render_count = 0

for pose_name, pose_rots in POSES.items():
    apply_pose(armature_obj, pose_rots)
    
    for cam_cfg in CAMERA_ANGLES:
        set_camera(cam_cfg["azim"], cam_cfg["elev"], cam_distance, bbox_center)
        
        filename = f"{pose_name}_{cam_cfg['name']}.png"
        filepath = os.path.join(out_dir, filename)
        scene.render.filepath = filepath
        
        bpy.ops.render.render(write_still=True)
        render_count += 1
        print(f"  [{render_count}/{total_renders}] {filename}")
        
        manifest["poses"].append({
            "file": filename,
            "pose": pose_name,
            "camera": cam_cfg["name"],
            "camera_azimuth": cam_cfg["azim"],
            "camera_elevation": cam_cfg["elev"],
            "description": cam_cfg["desc"],
        })

# Save manifest
manifest_path = os.path.join(out_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone: {render_count} renders saved to {out_dir}/")
print(f"Manifest: {manifest_path}")
'''


def main():
    parser = argparse.ArgumentParser(description="Render pose grid using Blender final render mode")
    parser.add_argument("--asset", required=True, help="Path to rigged hand .glb/.blend")
    parser.add_argument("--rig-map", required=True, help="Path to rig_map.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--skin-tone", default="medium", choices=["light", "medium", "tan", "dark"])
    args = parser.parse_args()

    asset = Path(args.asset).resolve()
    rig_map = Path(args.rig_map).resolve()
    out_dir = Path(args.out).resolve()

    if not asset.exists():
        print(f"ERROR: Asset not found: {asset}")
        sys.exit(1)
    if not rig_map.exists():
        print(f"ERROR: Rig map not found: {rig_map}")
        sys.exit(1)

    blender = find_blender()
    print(f"Blender: {blender}")
    print(f"Asset: {asset}")
    print(f"Rig map: {rig_map}")
    print(f"Output: {out_dir}")
    print(f"Skin tone: {args.skin_tone}")
    print()

    # Write the Blender script to a temp file
    script_path = out_dir / "_render_script.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(BLENDER_SCRIPT)

    # Run Blender in background
    cmd = [
        blender,
        "--background",
        "--python", str(script_path),
        "--",
        str(asset),
        str(rig_map),
        str(out_dir),
        args.skin_tone,
    ]

    print(f"Running Blender render...")
    print(f"  {' '.join(cmd[:4])} ...")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Cleanup temp script
    if script_path.exists():
        script_path.unlink()

    if result.returncode != 0:
        print(f"\nERROR: Blender render failed (exit code {result.returncode})")
        print("This script does NOT fall back to debug/matplotlib rendering.")
        sys.exit(1)

    # Validate outputs exist
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: No manifest.json produced — render failed silently")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    rendered = len(manifest.get("poses", []))
    print(f"\n✓ Pose grid complete: {rendered} renders")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
