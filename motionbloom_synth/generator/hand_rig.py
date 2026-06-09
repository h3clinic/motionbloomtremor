"""Hand Rig Controller.

Applies bone rotations to a Blender armature.
Reads the pose library from configs/poses.yaml and maps joint names
to the actual bone names defined in handharness/rig_map.json.

Designed to run inside Blender's Python environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
HANDHARNESS_DIR = Path(__file__).parent.parent.parent / "handharness"


def load_poses() -> Dict[str, Dict[str, float]]:
    """Load all poses from the pose library YAML.

    Returns:
        Dict mapping pose_name -> {joint_name: angle_degrees}.
    """
    poses_path = CONFIGS_DIR / "poses.yaml"
    with open(poses_path) as f:
        data = yaml.safe_load(f)

    poses = {}
    for name, pose_data in data["poses"].items():
        # Extract only numeric bone angles (skip 'description')
        angles = {k: v for k, v in pose_data.items() if isinstance(v, (int, float))}
        poses[name] = angles

    return poses


def load_rig_map() -> dict:
    """Load the rig bone mapping from handharness.

    Returns:
        Dict with armature_name, wrist bone, and finger bone lists.
    """
    rig_map_path = HANDHARNESS_DIR / "rig_map.json"
    with open(rig_map_path) as f:
        return json.load(f)


# Mapping from our generic joint names to rig bone names.
# This bridges poses.yaml joints → actual Blender bone names.

def build_joint_to_bone_map(rig_map: dict) -> Dict[str, str]:
    """Build mapping from generic joint names to actual armature bone names.

    Generic names: thumb_meta, thumb_prox, thumb_dist,
                   index_meta, index_prox, index_midd, etc.
                   wrist_pitch, wrist_yaw, wrist_roll

    Args:
        rig_map: Loaded rig_map.json content.

    Returns:
        Dict mapping generic_joint_name -> blender_bone_name.
    """
    mapping = {}
    fingers = rig_map.get("fingers", {})

    # Map each finger's 3 bones
    finger_suffixes = ["meta", "prox", "midd"]  # dist handled for thumb
    for finger_name, bone_list in fingers.items():
        for i, bone_name in enumerate(bone_list):
            if i < len(finger_suffixes):
                generic = f"{finger_name}_{finger_suffixes[i]}"
            else:
                generic = f"{finger_name}_dist"
            mapping[generic] = bone_name

    # Special case: thumb has meta/prox/dist in our naming
    if "thumb" in fingers and len(fingers["thumb"]) >= 3:
        mapping["thumb_meta"] = fingers["thumb"][0]
        mapping["thumb_prox"] = fingers["thumb"][1]
        mapping["thumb_dist"] = fingers["thumb"][2]

    # Wrist — maps to the rig wrist bone
    wrist_bone = rig_map.get("wrist", "wrist")
    mapping["wrist_pitch"] = wrist_bone
    mapping["wrist_yaw"] = wrist_bone
    mapping["wrist_roll"] = wrist_bone

    return mapping


def get_hand_model_path() -> Path:
    """Get path to the rigged hand model .glb file."""
    model_path = (
        HANDHARNESS_DIR
        / "input"
        / "hand_base"
        / "extracted"
        / "source"
        / "Do_Hand_DetailedRiggedAnimated_shared_16022026.glb"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Hand model not found: {model_path}")
    return model_path


def apply_pose_to_armature(
    armature,
    pose_angles: Dict[str, float],
    rig_map: dict,
    frame: int = 1,
):
    """Apply joint angles to a Blender armature and keyframe.

    Must be called from within Blender's Python environment.

    Args:
        armature: bpy.types.Object (armature).
        pose_angles: Dict of generic_joint_name -> angle_degrees.
        rig_map: Loaded rig_map.json.
        frame: Frame number to keyframe at.
    """
    import bpy
    from mathutils import Euler

    joint_to_bone = build_joint_to_bone_map(rig_map)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    for joint_name, angle_deg in pose_angles.items():
        bone_name = joint_to_bone.get(joint_name)
        if bone_name is None:
            continue

        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue

        angle_rad = math.radians(angle_deg)

        # Determine rotation axis based on joint type
        if "pitch" in joint_name:
            axis_idx = 0  # X rotation
        elif "yaw" in joint_name:
            axis_idx = 2  # Z rotation
        elif "roll" in joint_name:
            axis_idx = 1  # Y rotation
        else:
            # Default: finger bones rotate around X (flexion/extension)
            axis_idx = 0

        # Apply rotation
        pose_bone.rotation_mode = "XYZ"
        euler = list(pose_bone.rotation_euler)
        euler[axis_idx] = angle_rad
        pose_bone.rotation_euler = Euler(euler)

        # Insert keyframe
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")


def apply_animation_to_armature(
    armature,
    animation_frames: List[Dict[str, float]],
    rig_map: dict,
    fps: int = 30,
):
    """Apply full animation (from tremor engine) to armature.

    Args:
        armature: Blender armature object.
        animation_frames: List of per-frame angle dicts from tremor_engine.
        rig_map: Loaded rig_map.json.
        fps: Frames per second.
    """
    import bpy

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = len(animation_frames)

    for frame_idx, angles in enumerate(animation_frames):
        apply_pose_to_armature(armature, angles, rig_map, frame=frame_idx + 1)


def import_hand_model():
    """Import the rigged hand model into the current Blender scene.

    Returns:
        The imported armature object.
    """
    import bpy

    model_path = get_hand_model_path()

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=str(model_path))

    # Find the armature
    rig_map = load_rig_map()
    armature_name = rig_map.get("armature_name", "Armature")

    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        # Try to find any armature
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                armature = obj
                break

    if armature is None:
        raise RuntimeError(
            f"No armature found after importing {model_path}. "
            f"Expected '{armature_name}'."
        )

    return armature
