"""
render_tremor_dataset.py – Blender final-render tremor video dataset generator.

Produces MP4 videos of a rigged hand with bone-level tremor injection:
- Smooth shading + skin material + lighting (same as pose_grid)
- Bone-driven sinusoidal tremor with clinical parameters
- 30fps MP4 videos, 3-5 seconds each
- Per-video metadata.jsonl with pose, camera, tremor params
- Validation report

Usage (from repo root):
    python handharness/render_tremor_dataset.py \
        --asset handharness/input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb \
        --rig-map handharness/rig_map.json \
        --count 100 --fps 30 --duration 4 \
        --out datasets/synth_tremor_v1

Requires Blender. Fails if Blender is not installed.
"""

import argparse
import json
import subprocess
import sys
import shutil
import random
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


# ─── Blender Python Script ───────────────────────────────────────────────────
BLENDER_SCRIPT = r'''
import bpy
import bmesh
import math
import json
import os
import sys
import random
from pathlib import Path
from mathutils import Euler, Vector

# ─── Parse Args ───────────────────────────────────────────────────────────
argv = sys.argv
argv = argv[argv.index("--") + 1:]
asset_path = argv[0]
rig_map_path = argv[1]
out_dir = argv[2]
config_json = argv[3]

with open(config_json) as f:
    config = json.load(f)

os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

with open(rig_map_path) as f:
    rig_map = json.load(f)

# ─── Skin Tones ───────────────────────────────────────────────────────────
SKIN_TONES = {
    "light":  {"base": (0.93, 0.78, 0.68, 1.0), "subsurface": (0.90, 0.60, 0.45)},
    "medium": {"base": (0.76, 0.57, 0.44, 1.0), "subsurface": (0.70, 0.40, 0.28)},
    "tan":    {"base": (0.62, 0.42, 0.30, 1.0), "subsurface": (0.55, 0.30, 0.18)},
    "dark":   {"base": (0.38, 0.24, 0.16, 1.0), "subsurface": (0.30, 0.15, 0.08)},
}

# ─── Tremor Parameters ────────────────────────────────────────────────────
TREMOR_PROFILES = {
    "rest_4hz":     {"freq": 4.0, "amp_range": (2, 6),  "label": "rest"},
    "rest_5hz":     {"freq": 5.0, "amp_range": (3, 8),  "label": "rest"},
    "postural_8hz": {"freq": 8.0, "amp_range": (1, 4),  "label": "postural"},
    "postural_10hz":{"freq": 10.0,"amp_range": (1, 3),  "label": "postural"},
    "intention_3hz":{"freq": 3.0, "amp_range": (4, 12), "label": "intention"},
    "essential_6hz":{"freq": 6.0, "amp_range": (2, 7),  "label": "essential"},
    "no_tremor":    {"freq": 0.0, "amp_range": (0, 0),  "label": "none"},
}

POSES = {
    "rest_open": {},
    "palm_up": {"radius_ulna": (0, 0, 180)},
    "postural_hold": {"radius_ulna": (-15, 0, 0)},
    "fist_light": {
        "index_prox": (40, 0, 0), "index_midd": (35, 0, 0),
        "midd_prox": (40, 0, 0), "midd_midd": (35, 0, 0),
        "ring_prox": (40, 0, 0), "ring_midd": (35, 0, 0),
        "pinky_prox": (40, 0, 0), "pinky_midd": (35, 0, 0),
        "thumb_prox": (20, 0, 0),
    },
    "pointing": {
        "midd_prox": (75, 0, 0), "midd_midd": (70, 0, 0),
        "ring_prox": (75, 0, 0), "ring_midd": (70, 0, 0),
        "pinky_prox": (75, 0, 0), "pinky_midd": (70, 0, 0),
        "thumb_meta": (25, 0, -15),
    },
}

CAMERA_ANGLES = [
    {"name": "front", "azim": 0, "elev": 5},
    {"name": "oblique", "azim": 35, "elev": 15},
    {"name": "side", "azim": 80, "elev": 5},
    {"name": "top", "azim": 0, "elev": 65},
]

# ─── Scene Setup ─────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=asset_path)

mesh_obj = None
armature_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.name == 'Do_HandObject':
        mesh_obj = obj
    elif obj.type == 'ARMATURE' and obj.name == 'Do_HandRigged':
        armature_obj = obj

# Fallback: largest mesh
if not mesh_obj:
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if meshes:
        mesh_obj = max(meshes, key=lambda o: len(o.data.vertices))
if not armature_obj:
    armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if armatures:
        armature_obj = armatures[0]

if not mesh_obj:
    print("FATAL: No mesh found")
    sys.exit(1)

# Remove stray objects
for obj in list(bpy.context.scene.objects):
    if obj.type == 'MESH' and obj != mesh_obj:
        bpy.data.objects.remove(obj, do_unlink=True)

# ─── Smooth Shading ──────────────────────────────────────────────────────
mesh_obj.data.polygons.foreach_set("use_smooth", [True] * len(mesh_obj.data.polygons))
mesh_obj.data.update()

bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

subsurf = mesh_obj.modifiers.new(name="Subdivision", type='SUBSURF')
subsurf.levels = 1
subsurf.render_levels = 2

# ─── Skin Material ───────────────────────────────────────────────────────
mesh_obj.data.materials.clear()
mat = bpy.data.materials.new(name="SkinMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output_node = nodes.new(type='ShaderNodeOutputMaterial')
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

tone = SKIN_TONES.get(config.get("skin_tone", "medium"), SKIN_TONES["medium"])
bsdf.inputs["Base Color"].default_value = tone["base"]
bsdf.inputs["Roughness"].default_value = 0.55
bsdf.inputs["Specular IOR Level"].default_value = 0.3
bsdf.inputs["Subsurface Weight"].default_value = 0.15
bsdf.inputs["Subsurface Radius"].default_value = tone["subsurface"]

links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
mesh_obj.data.materials.append(mat)

# ─── Lighting (Sun-based for reliable illumination) ──────────────────────
bbox = [mesh_obj.matrix_world @ Vector(v) for v in mesh_obj.bound_box]
bbox_center = sum(bbox, Vector()) / 8
bbox_size = max((max(v[i] for v in bbox) - min(v[i] for v in bbox)) for i in range(3))
cx, cy, cz = bbox_center.x, bbox_center.y, bbox_center.z
light_dist = bbox_size * 1.5

sun_light = bpy.data.lights.new(name="SunKey", type='SUN')
sun_light.energy = 12.0
sun_light.angle = math.radians(15)
sun_obj = bpy.data.objects.new("SunKey", sun_light)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = Euler((math.radians(-50), math.radians(15), math.radians(25)))

sun_fill = bpy.data.lights.new(name="SunFill", type='SUN')
sun_fill.energy = 6.0
sun_fill.angle = math.radians(30)
sun_fill_obj = bpy.data.objects.new("SunFill", sun_fill)
bpy.context.collection.objects.link(sun_fill_obj)
sun_fill_obj.rotation_euler = Euler((math.radians(-30), math.radians(-20), math.radians(-15)))

sun_under = bpy.data.lights.new(name="SunUnder", type='SUN')
sun_under.energy = 3.0
sun_under_obj = bpy.data.objects.new("SunUnder", sun_under)
bpy.context.collection.objects.link(sun_under_obj)
sun_under_obj.rotation_euler = Euler((math.radians(60), 0, math.radians(-10)))

point_light = bpy.data.lights.new(name="PointWarm", type='POINT')
point_light.energy = 8000
point_obj = bpy.data.objects.new("PointWarm", point_light)
bpy.context.collection.objects.link(point_obj)
point_obj.location = (cx, cy - light_dist * 0.6, cz + light_dist * 0.4)

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs["Color"].default_value = (0.18, 0.18, 0.20, 1.0)
bg_node.inputs["Strength"].default_value = 1.0

# ─── Render Settings ─────────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.fps = config["fps"]
# Render as PNG sequence (Blender 5.1 ffmpeg integration changed)
# We'll combine to MP4 with ffmpeg after rendering
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'

scene.eevee.use_shadows = True

# Camera
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.type = 'PERSP'
cam_data.lens = 50
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Target
cam_distance = bbox_size * 1.5


def set_camera(azim_deg, elev_deg):
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    x = bbox_center.x + cam_distance * math.cos(elev) * math.sin(azim)
    y = bbox_center.y - cam_distance * math.cos(elev) * math.cos(azim)
    z = bbox_center.z + cam_distance * math.sin(elev)
    cam_obj.location = (x, y, z)
    direction = bbox_center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


def compute_tremor_rotation(frame, freq, amplitude, phase, fps):
    """Compute tremor rotation at given frame with harmonics."""
    if freq == 0 or amplitude == 0:
        return 0.0
    t = frame / fps
    # Primary frequency
    val = amplitude * math.sin(2 * math.pi * freq * t + phase)
    # Second harmonic (30% amplitude)
    val += amplitude * 0.3 * math.sin(2 * math.pi * freq * 2 * t + phase * 1.3)
    # Micro noise
    val += amplitude * 0.05 * math.sin(2 * math.pi * 17.3 * t + phase * 2.7)
    return val


# ─── Render Videos ────────────────────────────────────────────────────────
videos_config = config["videos"]
metadata_lines = []

for vi, vid_cfg in enumerate(videos_config):
    video_id = f"MB_SYNTH_{vi+1:06d}"
    pose_name = vid_cfg["pose"]
    tremor_profile = vid_cfg["tremor_profile"]
    cam_cfg = vid_cfg["camera"]
    duration = vid_cfg["duration"]
    
    profile = TREMOR_PROFILES[tremor_profile]
    freq = profile["freq"]
    amp = vid_cfg.get("amplitude", random.uniform(*profile["amp_range"]))
    
    total_frames = int(duration * config["fps"])
    scene.frame_start = 1
    scene.frame_end = total_frames
    
    # Set camera
    set_camera(cam_cfg["azim"], cam_cfg["elev"])
    
    # Base pose
    base_pose = POSES.get(pose_name, {})
    
    # Tremor-affected bones (typically distal joints)
    tremor_bones = vid_cfg.get("tremor_bones", ["index_midd", "midd_midd", "ring_midd", "pinky_midd", "thumb_dist"])
    tremor_axis = vid_cfg.get("tremor_axis", "X")
    phase = random.uniform(0, 2 * math.pi)
    
    # Keyframe the tremor
    if armature_obj:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        
        # Clear all animation data
        if armature_obj.animation_data:
            armature_obj.animation_data_clear()
        
        for frame in range(1, total_frames + 1):
            scene.frame_set(frame)
            
            # Apply base pose + tremor
            for pbone in armature_obj.pose.bones:
                pbone.rotation_mode = 'XYZ'
                bone_name = pbone.name
                
                # Base rotation
                base_rot = list(base_pose.get(bone_name, (0, 0, 0)))
                
                # Add tremor to affected bones
                if bone_name in tremor_bones:
                    tremor_val = compute_tremor_rotation(frame, freq, amp, phase, config["fps"])
                    axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(tremor_axis, 0)
                    base_rot[axis_idx] += tremor_val
                
                pbone.rotation_euler = (
                    math.radians(base_rot[0]),
                    math.radians(base_rot[1]),
                    math.radians(base_rot[2]),
                )
                pbone.keyframe_insert(data_path="rotation_euler", frame=frame)
        
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Render as PNG frame sequence
    frames_dir = os.path.join(out_dir, "frames", video_id)
    os.makedirs(frames_dir, exist_ok=True)
    scene.render.filepath = os.path.join(frames_dir, "frame_")
    bpy.ops.render.render(animation=True)
    
    # Combine frames to MP4 using ffmpeg
    video_path = os.path.join(out_dir, "videos", f"{video_id}.mp4")
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(config["fps"]),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", video_path
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)
    
    # Save first frame as preview
    import shutil
    first_frame = os.path.join(frames_dir, "frame_0001.png")
    preview_path = os.path.join(out_dir, "frames", f"{video_id}_frame001.png")
    if os.path.exists(first_frame):
        shutil.copy2(first_frame, preview_path)
    
    # Metadata
    meta = {
        "video_id": video_id,
        "file": f"videos/{video_id}.mp4",
        "preview": f"frames/{video_id}_frame001.png",
        "pose": pose_name,
        "camera_angle": cam_cfg["name"],
        "camera_azimuth": cam_cfg["azim"],
        "camera_elevation": cam_cfg["elev"],
        "tremor_type": profile["label"],
        "tremor_frequency_hz": freq,
        "tremor_amplitude_deg": round(amp, 2),
        "tremor_bones": tremor_bones,
        "tremor_axis": tremor_axis,
        "duration_sec": duration,
        "fps": config["fps"],
        "total_frames": total_frames,
        "skin_tone": config.get("skin_tone", "medium"),
        "severity_score": min(100, int(amp * freq * 1.8)),
    }
    metadata_lines.append(json.dumps(meta))
    
    print(f"  [{vi+1}/{len(videos_config)}] {video_id} | {pose_name} | {tremor_profile} | amp={amp:.1f}° | {duration}s")

# Write metadata
with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
    f.write("\n".join(metadata_lines) + "\n")

# Write labels CSV
with open(os.path.join(out_dir, "labels.csv"), "w") as f:
    f.write("video_id,pose,tremor_type,frequency_hz,amplitude_deg,severity_score\n")
    for line in metadata_lines:
        m = json.loads(line)
        f.write(f"{m['video_id']},{m['pose']},{m['tremor_type']},{m['tremor_frequency_hz']},{m['tremor_amplitude_deg']},{m['severity_score']}\n")

print(f"\nDone: {len(videos_config)} videos rendered to {out_dir}/")
'''


def generate_video_configs(count: int, fps: int, duration: float) -> list:
    """Generate randomized video configurations."""
    profiles = list({
        "rest_4hz", "rest_5hz", "postural_8hz", "postural_10hz",
        "intention_3hz", "essential_6hz", "no_tremor"
    })
    poses = list({"rest_open", "palm_up", "postural_hold", "fist_light", "pointing"})
    cameras = [
        {"name": "front", "azim": 0, "elev": 5},
        {"name": "oblique", "azim": 35, "elev": 15},
        {"name": "side", "azim": 80, "elev": 5},
        {"name": "top", "azim": 0, "elev": 65},
    ]
    
    configs = []
    for i in range(count):
        tremor_profile = random.choice(profiles)
        cfg = {
            "pose": random.choice(poses),
            "tremor_profile": tremor_profile,
            "camera": random.choice(cameras),
            "duration": duration + random.uniform(-0.5, 0.5),
            "tremor_bones": random.sample(
                ["index_midd", "midd_midd", "ring_midd", "pinky_midd", "thumb_dist",
                 "index_prox", "midd_prox", "ring_prox", "pinky_prox"],
                k=random.randint(3, 6)
            ),
            "tremor_axis": random.choice(["X", "Z"]),
        }
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Render tremor video dataset with Blender")
    parser.add_argument("--asset", required=True, help="Path to rigged hand .glb/.blend")
    parser.add_argument("--rig-map", required=True, help="Path to rig_map.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=4.0, help="Video duration (seconds)")
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
    print(f"Output: {out_dir}")
    print(f"Videos: {args.count} @ {args.fps}fps × {args.duration}s")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    # Generate video configs
    video_configs = generate_video_configs(args.count, args.fps, args.duration)
    
    # Write config for Blender
    run_config = {
        "fps": args.fps,
        "skin_tone": args.skin_tone,
        "videos": video_configs,
    }
    config_path = out_dir / "_run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2))

    # Write Blender script
    script_path = out_dir / "_render_script.py"
    script_path.write_text(BLENDER_SCRIPT)

    # Run Blender
    cmd = [
        blender,
        "--background",
        "--python", str(script_path),
        "--",
        str(asset),
        str(rig_map),
        str(out_dir),
        str(config_path),
    ]

    print(f"Running Blender render ({args.count} videos)...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Cleanup
    for tmp in [script_path, config_path]:
        if tmp.exists():
            tmp.unlink()

    if result.returncode != 0:
        print(f"\nERROR: Blender render failed (exit code {result.returncode})")
        sys.exit(1)

    # Validate
    metadata_path = out_dir / "metadata.jsonl"
    if not metadata_path.exists():
        print("ERROR: No metadata.jsonl — render failed")
        sys.exit(1)

    lines = metadata_path.read_text().strip().split("\n")
    print(f"\n✓ Dataset complete: {len(lines)} videos")
    print(f"  Videos:   {out_dir / 'videos'}")
    print(f"  Frames:   {out_dir / 'frames'}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Labels:   {out_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
