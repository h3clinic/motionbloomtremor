"""Camera Engine.

Manages camera placement, animation, and multi-view rendering setup
for the synthetic tremor dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_camera_config() -> dict:
    """Load camera and motion type configurations."""
    config_path = CONFIGS_DIR / "cameras.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_camera_position(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    target: Tuple[float, float, float] = (0, 0, 0),
) -> Tuple[float, float, float]:
    """Compute camera world position from spherical coordinates.

    Args:
        azimuth_deg: Horizontal angle (0=front, 90=right).
        elevation_deg: Vertical angle (0=level, 90=above).
        distance: Distance from target in meters.
        target: Point the camera looks at.

    Returns:
        (x, y, z) camera position.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = target[0] + distance * math.cos(el) * math.sin(az)
    y = target[1] - distance * math.cos(el) * math.cos(az)
    z = target[2] + distance * math.sin(el)

    return (x, y, z)


def setup_camera_blender(
    camera_name: str,
    camera_config: Optional[dict] = None,
):
    """Set up a Blender camera from config.

    Must be called inside Blender's Python environment.

    Args:
        camera_name: Key from cameras.yaml (e.g., 'front', 'side_right').
        camera_config: Override config dict (if None, loads from YAML).

    Returns:
        The Blender camera object.
    """
    import bpy
    from mathutils import Vector

    if camera_config is None:
        config = load_camera_config()
        camera_config = config["cameras"][camera_name]

    azimuth = camera_config["azimuth"]
    elevation = camera_config["elevation"]
    distance = camera_config["distance"]
    fov = camera_config.get("fov", 50)

    # Compute position
    pos = get_camera_position(azimuth, elevation, distance)

    # Create or get camera
    cam_data = bpy.data.cameras.get("SynthCamera")
    if cam_data is None:
        cam_data = bpy.data.cameras.new("SynthCamera")

    cam_obj = bpy.data.objects.get("SynthCameraObj")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("SynthCameraObj", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)

    # Position and aim at origin
    cam_obj.location = Vector(pos)
    direction = Vector((0, 0, 0)) - Vector(pos)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set FOV
    cam_data.angle = math.radians(fov)

    # Make active camera
    bpy.context.scene.camera = cam_obj

    return cam_obj


def apply_hand_motion(
    armature,
    motion_type: str,
    duration_sec: float,
    fps: int = 30,
):
    """Apply gross hand motion (translation/rotation) to the armature root.

    This is NOT tremor — it's deliberate hand movement during the clip.

    Args:
        armature: Blender armature object.
        motion_type: Key from cameras.yaml motion_types.
        duration_sec: Clip duration.
        fps: Frames per second.
    """
    import bpy
    from mathutils import Vector, Euler

    config = load_camera_config()
    motion_config = config["motion_types"].get(motion_type)

    if motion_config is None or motion_type == "stationary":
        return  # No motion to apply

    num_frames = int(duration_sec * fps)
    trans_amp = motion_config.get("translation_amplitude", 0)
    trans_freq = motion_config.get("translation_frequency", 0)
    trans_axis = motion_config.get("translation_axis", [1, 0, 0])
    rot_amp = motion_config.get("rotation_amplitude", 0)
    rot_freq = motion_config.get("rotation_frequency", 0)
    rot_axis = motion_config.get("rotation_axis", "pitch")

    base_location = Vector(armature.location)
    base_rotation = list(armature.rotation_euler)

    for frame_idx in range(num_frames):
        t = frame_idx / fps
        bpy.context.scene.frame_set(frame_idx + 1)

        # Translation
        if trans_amp > 0 and trans_freq > 0:
            offset = trans_amp * math.sin(2 * math.pi * trans_freq * t)
            armature.location = base_location + Vector([
                offset * trans_axis[0],
                offset * trans_axis[1],
                offset * trans_axis[2],
            ])
            armature.keyframe_insert(data_path="location", frame=frame_idx + 1)

        # Rotation
        if rot_amp > 0 and rot_freq > 0:
            angle = math.radians(rot_amp * math.sin(2 * math.pi * rot_freq * t))
            euler = list(base_rotation)
            if rot_axis == "pitch":
                euler[0] += angle
            elif rot_axis == "yaw":
                euler[2] += angle
            elif rot_axis == "roll":
                euler[1] += angle
            armature.rotation_euler = Euler(euler)
            armature.keyframe_insert(
                data_path="rotation_euler", frame=frame_idx + 1
            )
