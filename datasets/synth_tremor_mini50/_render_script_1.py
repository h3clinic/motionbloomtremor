
import bpy
import math
import json
import os
import sys
import random
import shutil
import subprocess
from pathlib import Path
from mathutils import Euler, Vector

# Parse arguments after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:]
asset_path = argv[0]
rig_map_path = argv[1]
out_dir = argv[2]
config_path = argv[3]

with open(config_path) as f:
    config = json.load(f)

os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "pose_previews"), exist_ok=True)

# ─── Scene Setup ─────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=asset_path)

mesh_obj = None
armature_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.name == 'Do_HandObject':
        mesh_obj = obj
    elif obj.type == 'ARMATURE' and obj.name == 'Do_HandRigged':
        armature_obj = obj

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

# Remove stray objects (e.g. Icosphere)
for obj in list(bpy.context.scene.objects):
    if obj.type == 'MESH' and obj != mesh_obj:
        bpy.data.objects.remove(obj, do_unlink=True)

# ─── Smooth Shading + SubSurf ────────────────────────────────────────────────
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

# ─── Bounding Box Calculation ────────────────────────────────────────────────
bbox = [mesh_obj.matrix_world @ Vector(v) for v in mesh_obj.bound_box]
bbox_center = sum(bbox, Vector()) / 8
bbox_size = max((max(v[i] for v in bbox) - min(v[i] for v in bbox)) for i in range(3))
cx, cy, cz = bbox_center.x, bbox_center.y, bbox_center.z
light_dist = bbox_size * 2.0
# Camera distance: hand should occupy ~50-60% of frame
# With 35mm lens at 512px, 2.2x bbox_size gives ~50% framing
cam_distance = bbox_size * 2.2

# ─── Lighting (soft studio setup for MediaPipe compatibility) ─────────────────
# Key light: large soft area light from front-above
key_light = bpy.data.lights.new(name="KeyArea", type='AREA')
key_light.energy = 800
key_light.size = bbox_size * 3.0  # Very large for soft shadows
key_light.color = (1.0, 0.97, 0.93)
key_obj = bpy.data.objects.new("KeyArea", key_light)
bpy.context.collection.objects.link(key_obj)
key_obj.location = (cx, cy - light_dist * 1.2, cz + light_dist * 1.0)
key_obj.rotation_euler = Euler((math.radians(-40), 0, 0))

# Fill light: area light from opposite side (reduces shadows)
fill_light = bpy.data.lights.new(name="FillArea", type='AREA')
fill_light.energy = 500
fill_light.size = bbox_size * 4.0  # Even larger for very soft fill
fill_light.color = (0.95, 0.97, 1.0)
fill_obj = bpy.data.objects.new("FillArea", fill_light)
bpy.context.collection.objects.link(fill_obj)
fill_obj.location = (cx + light_dist * 0.8, cy - light_dist * 0.5, cz + light_dist * 0.3)
fill_obj.rotation_euler = Euler((math.radians(-25), math.radians(35), 0))

# Rim/back light: subtle edge definition
rim_light = bpy.data.lights.new(name="RimArea", type='AREA')
rim_light.energy = 300
rim_light.size = bbox_size * 2.0
rim_light.color = (1.0, 0.98, 0.95)
rim_obj = bpy.data.objects.new("RimArea", rim_light)
bpy.context.collection.objects.link(rim_obj)
rim_obj.location = (cx - light_dist * 0.5, cy + light_dist * 0.8, cz + light_dist * 0.6)
rim_obj.rotation_euler = Euler((math.radians(-20), math.radians(-150), 0))

# Under-fill: prevents completely dark underside
under_light = bpy.data.lights.new(name="UnderFill", type='AREA')
under_light.energy = 200
under_light.size = bbox_size * 3.0
under_light.color = (0.9, 0.92, 1.0)
under_obj = bpy.data.objects.new("UnderFill", under_light)
bpy.context.collection.objects.link(under_obj)
under_obj.location = (cx, cy - light_dist * 0.3, cz - light_dist * 0.8)
under_obj.rotation_euler = Euler((math.radians(70), 0, 0))

# World background: medium gray for contrast (not black, not white)
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
# Neutral gray background - enough contrast with skin, not confusing for detector
bg_node.inputs["Color"].default_value = (0.42, 0.44, 0.46, 1.0)
bg_node.inputs["Strength"].default_value = 1.0

# ─── Render Settings ─────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.eevee.use_shadows = True

# Camera: 35mm lens for wider FOV - hand occupies ~55% of frame
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.type = 'PERSP'
cam_data.lens = 35
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# ─── Helper Functions ─────────────────────────────────────────────────────────

SKIN_TONES_DATA = {"very_light": {"base": [0.96, 0.87, 0.8, 1.0], "subsurface": [0.95, 0.7, 0.55], "roughness": 0.5}, "light": {"base": [0.93, 0.78, 0.68, 1.0], "subsurface": [0.9, 0.6, 0.45], "roughness": 0.52}, "light_warm": {"base": [0.89, 0.72, 0.6, 1.0], "subsurface": [0.85, 0.55, 0.4], "roughness": 0.53}, "medium_light": {"base": [0.82, 0.63, 0.5, 1.0], "subsurface": [0.78, 0.48, 0.33], "roughness": 0.54}, "medium": {"base": [0.76, 0.57, 0.44, 1.0], "subsurface": [0.7, 0.4, 0.28], "roughness": 0.55}, "medium_warm": {"base": [0.72, 0.48, 0.34, 1.0], "subsurface": [0.65, 0.35, 0.22], "roughness": 0.56}, "tan": {"base": [0.62, 0.42, 0.3, 1.0], "subsurface": [0.55, 0.3, 0.18], "roughness": 0.57}, "brown": {"base": [0.5, 0.33, 0.22, 1.0], "subsurface": [0.42, 0.22, 0.12], "roughness": 0.58}, "dark_brown": {"base": [0.4, 0.26, 0.17, 1.0], "subsurface": [0.32, 0.16, 0.08], "roughness": 0.59}, "deep_brown": {"base": [0.32, 0.2, 0.12, 1.0], "subsurface": [0.25, 0.12, 0.06], "roughness": 0.6}, "cool_dark": {"base": [0.28, 0.18, 0.14, 1.0], "subsurface": [0.2, 0.1, 0.06], "roughness": 0.61}, "warm_dark": {"base": [0.35, 0.22, 0.14, 1.0], "subsurface": [0.28, 0.14, 0.07], "roughness": 0.6}}


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


def set_skin_material(tone_data):
    mesh_obj.data.materials.clear()
    mat = bpy.data.materials.new(name="SkinMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Add subtle noise to base color for skin-like variation
    noise_tex = nodes.new(type='ShaderNodeTexNoise')
    noise_tex.inputs['Scale'].default_value = 15.0
    noise_tex.inputs['Detail'].default_value = 4.0
    noise_tex.inputs['Roughness'].default_value = 0.6
    
    # Mix noise with base color for subtle variation
    mix_rgb = nodes.new(type='ShaderNodeMix')
    mix_rgb.data_type = 'RGBA'
    mix_rgb.inputs[0].default_value = 0.08  # Very subtle variation
    mix_rgb.inputs[6].default_value = tone_data["base"]
    # Slightly darker variation
    darker = tuple(max(0, c - 0.06) for c in tone_data["base"][:3]) + (1.0,)
    mix_rgb.inputs[7].default_value = darker
    links.new(noise_tex.outputs['Fac'], mix_rgb.inputs[0])
    links.new(mix_rgb.outputs[2], bsdf.inputs["Base Color"])
    
    bsdf.inputs["Roughness"].default_value = tone_data["roughness"]
    # Adaptive material properties based on skin luminance
    # Dark tones need more specular highlight to look skin-like to detector
    skin_luminance = tone_data["base"][0] * 0.3 + tone_data["base"][1] * 0.6 + tone_data["base"][2] * 0.1
    if skin_luminance < 0.35:  # Dark tones
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
        bsdf.inputs["Subsurface Weight"].default_value = 0.2
    elif skin_luminance > 0.8:  # Very light tones
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
        bsdf.inputs["Subsurface Weight"].default_value = 0.5
    else:  # Medium tones
        bsdf.inputs["Specular IOR Level"].default_value = 0.3
        bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Radius"].default_value = tone_data["subsurface"]
    bsdf.inputs["Subsurface Scale"].default_value = 0.1
    
    # Adaptive background: ensure contrast with skin tone
    # Darker bg for lighter skin, lighter bg for darker skin
    if skin_luminance < 0.4:
        bg_val = (0.55, 0.56, 0.57, 1.0)  # Lighter bg for dark skin
    elif skin_luminance > 0.75:
        bg_val = (0.30, 0.32, 0.34, 1.0)  # Darker bg for light skin
    else:
        bg_val = (0.42, 0.44, 0.46, 1.0)  # Default gray
    world = bpy.context.scene.world
    if world and world.node_tree:
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = bg_val
    
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    mesh_obj.data.materials.append(mat)


def compute_tremor_rotation(frame, freq, amplitude, phase, fps):
    if freq == 0 or amplitude == 0:
        return 0.0
    t = frame / fps
    val = amplitude * math.sin(2 * math.pi * freq * t + phase)
    val += amplitude * 0.3 * math.sin(2 * math.pi * freq * 2 * t + phase * 1.3)
    val += amplitude * 0.05 * math.sin(2 * math.pi * 17.3 * t + phase * 2.7)
    return val


# ─── Render All Videos ────────────────────────────────────────────────────────
videos_config = config["videos"]
metadata_lines = []
failed_count = 0

for vi, vid_cfg in enumerate(videos_config):
    video_id = f"MB_SYNTH_{vi+1:06d}"
    pose_family = vid_cfg["pose_family"]
    pose_rotations = vid_cfg["pose_rotations"]
    skin_tone_id = vid_cfg["skin_tone_id"]
    tremor_profile = vid_cfg["tremor_profile"]
    freq = vid_cfg["frequency_hz"]
    amp = vid_cfg["amplitude_degrees"]
    affected_joints = vid_cfg["affected_joints"]
    motion_type = vid_cfg["motion_type"]
    duration = vid_cfg["duration"]
    render_seed = vid_cfg["render_seed"]
    cam_cfg = vid_cfg["camera"]

    fps = config["fps"]
    total_frames = int(duration * fps)
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.fps = fps

    # Skin material
    tone_data = SKIN_TONES_DATA[skin_tone_id]
    set_skin_material(tone_data)

    # Camera position
    set_camera(cam_cfg["azim"], cam_cfg["elev"])

    # Phase + axis randomization
    phase = random.Random(render_seed).uniform(0, 2 * math.pi)
    tremor_axis = random.Random(render_seed + 1).choice([0, 2])

    # Keyframe bones
    if armature_obj:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        if armature_obj.animation_data:
            armature_obj.animation_data_clear()

        for frame in range(1, total_frames + 1):
            scene.frame_set(frame)
            for pbone in armature_obj.pose.bones:
                pbone.rotation_mode = 'XYZ'
                bone_name = pbone.name
                base_rot = list(pose_rotations.get(bone_name, (0, 0, 0)))

                # Add tremor to affected joints
                if bone_name in affected_joints:
                    tremor_val = compute_tremor_rotation(frame, freq, amp, phase, fps)
                    base_rot[tremor_axis] += tremor_val

                # Add macro motion (wrist rotation) - limited to ±8° to keep hand in detectable range
                if bone_name == "radius_ulna" and motion_type == "wrist_rotation":
                    t = frame / fps
                    base_rot[1] += 8.0 * math.sin(0.6 * math.pi * t)

                pbone.rotation_euler = (
                    math.radians(base_rot[0]),
                    math.radians(base_rot[1]),
                    math.radians(base_rot[2]),
                )
                pbone.keyframe_insert(data_path="rotation_euler", frame=frame)

        bpy.ops.object.mode_set(mode='OBJECT')

    # Render frames to temp dir
    frames_tmp = os.path.join(out_dir, "_tmp_frames", video_id)
    os.makedirs(frames_tmp, exist_ok=True)
    scene.render.filepath = os.path.join(frames_tmp, "frame_")
    bpy.ops.render.render(animation=True)

    # Combine frames → MP4
    video_path = os.path.join(out_dir, "videos", f"{video_id}.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_tmp, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", video_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True)
    valid = os.path.exists(video_path) and os.path.getsize(video_path) > 1000

    # Save first frame as preview
    first_frame = os.path.join(frames_tmp, "frame_0001.png")
    preview_path = os.path.join(out_dir, "pose_previews", f"{video_id}.png")
    if os.path.exists(first_frame):
        shutil.copy2(first_frame, preview_path)

    # Cleanup temp frames
    shutil.rmtree(frames_tmp, ignore_errors=True)

    if not valid:
        failed_count += 1

    # Metadata record
    meta = {
        "video_id": video_id,
        "pose_id": f"POSE_{vi+1:06d}",
        "pose_family": pose_family,
        "skin_tone_id": skin_tone_id,
        "base_color_rgb": list(tone_data["base"][:3]),
        "material_seed": render_seed,
        "camera_angle": cam_cfg["name"],
        "camera_azimuth": cam_cfg["azim"],
        "camera_elevation": cam_cfg["elev"],
        "motion_type": motion_type,
        "tremor_type": vid_cfg["tremor_type"],
        "tremor_profile": tremor_profile,
        "frequency_hz": freq,
        "amplitude_degrees": amp,
        "affected_joints": affected_joints,
        "affected_joint_type": vid_cfg["affected_joint_type"],
        "severity_score_1_100": vid_cfg["severity_score"],
        "fps": fps,
        "duration_sec": duration,
        "frame_count": total_frames,
        "render_seed": render_seed,
        "valid_render": valid,
        "validation_flags": [],
    }
    metadata_lines.append(json.dumps(meta))
    print(f"  [{vi+1}/{len(videos_config)}] {video_id} | {pose_family:16s} | {tremor_profile:22s} | amp={amp:5.1f} | score={vid_cfg['severity_score']:3d} | {'OK' if valid else 'FAIL'}")

# Cleanup
tmp_dir = os.path.join(out_dir, "_tmp_frames")
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir, ignore_errors=True)

# Write metadata.jsonl
with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
    f.write("\n".join(metadata_lines) + "\n")

# Write labels.csv
with open(os.path.join(out_dir, "labels.csv"), "w") as f:
    header = "video_id,filepath,severity_score_1_100,tremor_type,frequency_hz,amplitude_degrees,pose_family,skin_tone_id,camera_angle,motion_type,affected_joint_type,valid_render"
    f.write(header + "\n")
    for line in metadata_lines:
        m = json.loads(line)
        f.write(f"{m['video_id']},videos/{m['video_id']}.mp4,{m['severity_score_1_100']},{m['tremor_type']},{m['frequency_hz']},{m['amplitude_degrees']},{m['pose_family']},{m['skin_tone_id']},{m['camera_angle']},{m['motion_type']},{m['affected_joint_type']},{m['valid_render']}\n")

# Write validation report
report = {
    "total_attempted": len(videos_config),
    "total_valid": len(videos_config) - failed_count,
    "total_failed": failed_count,
    "success_rate": round((len(videos_config) - failed_count) / max(1, len(videos_config)) * 100, 1),
}
with open(os.path.join(out_dir, "validation_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
print(f"Done: {len(videos_config) - failed_count}/{len(videos_config)} valid videos")
print(f"Failed: {failed_count}")
