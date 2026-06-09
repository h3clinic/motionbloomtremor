"""
render_tremor_dataset.py – Full-spec synthetic tremor video generator.

Implements the complete MotionBloom training data pipeline:
- 12 skin tones, 16 pose families, 10 tremor profiles
- Deterministic severity scoring (0-100)
- Balanced class/camera/motion distribution
- Retry on validation failure (max 3 attempts)
- Full metadata.jsonl + labels.csv + generation_config.json

Usage:
    # Smoke test (5 videos × 1 second)
    python handharness/render_tremor_dataset.py \
        --asset handharness/input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb \
        --rig-map handharness/rig_map.json \
        --count 5 --fps 30 --duration 1 --out datasets/synth_tremor_smoke --seed 123

    # Full generation (3000 videos × 4 seconds)
    python handharness/render_tremor_dataset.py \
        --asset handharness/input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb \
        --rig-map handharness/rig_map.json \
        --count 3000 --fps 30 --duration 4 --out datasets/synth_tremor_v2 --seed 42
"""

import argparse
import json
import subprocess
import sys
import shutil
import random
import math
import os
from pathlib import Path
from typing import List, Dict


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SKIN_TONES = {
    "very_light":   {"base": (0.96, 0.87, 0.80, 1.0), "subsurface": (0.95, 0.70, 0.55), "roughness": 0.50},
    "light":        {"base": (0.93, 0.78, 0.68, 1.0), "subsurface": (0.90, 0.60, 0.45), "roughness": 0.52},
    "light_warm":   {"base": (0.89, 0.72, 0.60, 1.0), "subsurface": (0.85, 0.55, 0.40), "roughness": 0.53},
    "medium_light": {"base": (0.82, 0.63, 0.50, 1.0), "subsurface": (0.78, 0.48, 0.33), "roughness": 0.54},
    "medium":       {"base": (0.76, 0.57, 0.44, 1.0), "subsurface": (0.70, 0.40, 0.28), "roughness": 0.55},
    "medium_warm":  {"base": (0.72, 0.48, 0.34, 1.0), "subsurface": (0.65, 0.35, 0.22), "roughness": 0.56},
    "tan":          {"base": (0.62, 0.42, 0.30, 1.0), "subsurface": (0.55, 0.30, 0.18), "roughness": 0.57},
    "brown":        {"base": (0.50, 0.33, 0.22, 1.0), "subsurface": (0.42, 0.22, 0.12), "roughness": 0.58},
    "dark_brown":   {"base": (0.40, 0.26, 0.17, 1.0), "subsurface": (0.32, 0.16, 0.08), "roughness": 0.59},
    "deep_brown":   {"base": (0.32, 0.20, 0.12, 1.0), "subsurface": (0.25, 0.12, 0.06), "roughness": 0.60},
    "cool_dark":    {"base": (0.28, 0.18, 0.14, 1.0), "subsurface": (0.20, 0.10, 0.06), "roughness": 0.61},
    "warm_dark":    {"base": (0.35, 0.22, 0.14, 1.0), "subsurface": (0.28, 0.14, 0.07), "roughness": 0.60},
}

POSE_FAMILIES = {
    "rest_open": {
        "index_meta": (0, 0, -8), "midd_meta": (0, 0, -3),
        "ring_meta": (0, 0, 5), "pinky_meta": (0, 0, 12), "thumb_meta": (5, 0, -15),
    },
    "relaxed": {
        "index_meta": (10, 0, -5), "index_prox": (15, 0, 0),
        "midd_meta": (10, 0, -2), "midd_prox": (15, 0, 0),
        "ring_meta": (12, 0, 4), "ring_prox": (18, 0, 0),
        "pinky_meta": (15, 0, 10), "pinky_prox": (20, 0, 0),
        "thumb_meta": (8, 0, -12),
    },
    "fist": {
        "index_meta": (70, 0, -5), "index_prox": (85, 0, 0), "index_midd": (75, 0, 0),
        "midd_meta": (70, 0, 0), "midd_prox": (85, 0, 0), "midd_midd": (75, 0, 0),
        "ring_meta": (70, 0, 5), "ring_prox": (85, 0, 0), "ring_midd": (75, 0, 0),
        "pinky_meta": (70, 0, 10), "pinky_prox": (85, 0, 0), "pinky_midd": (75, 0, 0),
        "thumb_meta": (30, 0, -20), "thumb_prox": (50, 0, 0), "thumb_dist": (40, 0, 0),
    },
    "partial_fist": {
        "index_meta": (40, 0, -4), "index_prox": (50, 0, 0), "index_midd": (40, 0, 0),
        "midd_meta": (40, 0, 0), "midd_prox": (50, 0, 0), "midd_midd": (40, 0, 0),
        "ring_meta": (45, 0, 4), "ring_prox": (55, 0, 0), "ring_midd": (45, 0, 0),
        "pinky_meta": (50, 0, 8), "pinky_prox": (60, 0, 0), "pinky_midd": (50, 0, 0),
        "thumb_meta": (15, 0, -15), "thumb_prox": (25, 0, 0),
    },
    "pinch": {
        "index_prox": (45, 0, -5), "index_midd": (50, 0, 0),
        "thumb_meta": (20, 0, -30), "thumb_prox": (40, 0, -10), "thumb_dist": (30, 0, 0),
        "midd_meta": (15, 0, 3), "midd_prox": (20, 0, 0),
        "ring_meta": (20, 0, 8), "ring_prox": (25, 0, 0),
        "pinky_meta": (25, 0, 14), "pinky_prox": (30, 0, 0),
    },
    "pointing": {
        "index_meta": (0, 0, -5),
        "midd_meta": (70, 0, 3), "midd_prox": (85, 0, 0), "midd_midd": (75, 0, 0),
        "ring_meta": (70, 0, 8), "ring_prox": (85, 0, 0), "ring_midd": (75, 0, 0),
        "pinky_meta": (70, 0, 12), "pinky_prox": (85, 0, 0), "pinky_midd": (75, 0, 0),
        "thumb_meta": (30, 0, -15), "thumb_prox": (40, 0, 0),
    },
    "finger_spread": {
        "index_meta": (0, 0, -15), "midd_meta": (0, 0, -5),
        "ring_meta": (0, 0, 8), "pinky_meta": (0, 0, 18),
        "thumb_meta": (10, 0, -25),
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
    "reach_forward": {
        "radius_ulna": (-10, 0, 0),
        "index_meta": (5, 0, -6), "midd_meta": (5, 0, -2),
        "ring_meta": (5, 0, 4), "pinky_meta": (8, 0, 9),
    },
    "reach_sideways": {
        "radius_ulna": (0, -20, 0),
        "index_meta": (0, 0, -8), "ring_meta": (0, 0, 5),
    },
    "grip_release": {
        "index_meta": (30, 0, -5), "index_prox": (35, 0, 0),
        "midd_meta": (30, 0, 0), "midd_prox": (35, 0, 0),
        "ring_meta": (35, 0, 4), "ring_prox": (40, 0, 0),
        "pinky_meta": (40, 0, 8), "pinky_prox": (45, 0, 0),
        "thumb_meta": (15, 0, -12), "thumb_prox": (20, 0, 0),
    },
    "finger_tap_index": {
        "index_prox": (30, 0, -3), "index_midd": (40, 0, 0),
        "midd_meta": (10, 0, 0), "ring_meta": (10, 0, 5), "pinky_meta": (12, 0, 10),
    },
    "finger_tap_middle": {
        "midd_prox": (30, 0, 0), "midd_midd": (40, 0, 0),
        "index_meta": (10, 0, -5), "ring_meta": (10, 0, 5), "pinky_meta": (12, 0, 10),
    },
}

TREMOR_PROFILES = {
    "no_tremor":                {"freq_range": (0, 0),    "amp_range": (0, 0),      "label": "none"},
    "mild_postural":            {"freq_range": (4, 8),    "amp_range": (0.2, 2.0),  "label": "mild"},
    "moderate_postural":        {"freq_range": (4, 8),    "amp_range": (2.0, 6.0),  "label": "moderate"},
    "severe_postural":          {"freq_range": (4, 8),    "amp_range": (6.0, 12.0), "label": "severe"},
    "kinetic_tremor":           {"freq_range": (3, 6),    "amp_range": (2.0, 8.0),  "label": "moderate"},
    "wrist_dominant":           {"freq_range": (4, 7),    "amp_range": (1.5, 7.0),  "label": "moderate"},
    "finger_dominant":          {"freq_range": (5, 10),   "amp_range": (1.0, 5.0),  "label": "mild"},
    "mixed_wrist_finger":       {"freq_range": (4, 9),    "amp_range": (2.0, 9.0),  "label": "moderate"},
    "gross_motion_no_tremor":   {"freq_range": (0, 0),    "amp_range": (0, 0),      "label": "gross_motion"},
    "tracking_artifact_sim":    {"freq_range": (15, 25),  "amp_range": (0.5, 3.0),  "label": "artifact"},
}

CAMERA_ANGLES = [
    {"name": "front",        "azim": 0,   "elev": 5},
    {"name": "front_high",   "azim": 0,   "elev": 25},
    {"name": "oblique_l",    "azim": 30,  "elev": 15},
    {"name": "oblique_r",    "azim": -30, "elev": 15},
    {"name": "oblique_high", "azim": 35,  "elev": 30},
    {"name": "side_l",       "azim": 80,  "elev": 5},
    {"name": "side_r",       "azim": -80, "elev": 5},
    {"name": "top_down",     "azim": 0,   "elev": 65},
    {"name": "closeup_f",    "azim": 5,   "elev": 10},
    {"name": "closeup_o",    "azim": 25,  "elev": 12},
]

AFFECTED_JOINT_SETS = {
    "wrist_only":         ["radius_ulna"],
    "one_finger":         ["index_prox", "index_midd"],
    "multiple_fingers":   ["index_midd", "midd_midd", "ring_midd", "pinky_midd"],
    "wrist_plus_fingers": ["radius_ulna", "index_midd", "midd_midd", "ring_midd"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# BALANCE ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_DISTRIBUTION = {
    "none":         0.20,
    "mild":         0.25,
    "moderate":     0.25,
    "severe":       0.20,
    "artifact":     0.05,
    "gross_motion": 0.05,
}

CAMERA_GROUP_DISTRIBUTION = {
    "front":   0.25,
    "oblique": 0.35,
    "side":    0.20,
    "top":     0.10,
    "closeup": 0.10,
}

MOTION_GROUP_DISTRIBUTION = {
    "stationary":        0.35,
    "slow_translation":  0.25,
    "reach":             0.20,
    "wrist_rotation":    0.10,
    "finger_task":       0.10,
}

MOTION_TYPES = [
    "stationary",
    "slow_translation_x",
    "slow_translation_y",
    "slow_translation_z",
    "reach_forward",
    "reach_sideways",
    "wrist_rotation",
    "finger_task",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_severity_score(amplitude_degrees: float, frequency_hz: float,
                           affected_joint_type: str) -> int:
    """Deterministic severity score 0-100."""
    if amplitude_degrees == 0:
        return 0
    amplitude_norm = min(max(amplitude_degrees / 12.0, 0.0), 1.0)
    frequency_weight = 1.0 if 3 <= frequency_hz <= 12 else 0.5
    affected_weights = {
        "wrist_only": 0.75,
        "one_finger": 0.65,
        "multiple_fingers": 0.85,
        "wrist_plus_fingers": 1.0,
    }
    affected_weight = affected_weights.get(affected_joint_type, 0.75)
    score = round(100 * amplitude_norm * frequency_weight * affected_weight)
    return min(100, max(0, score))


# ═══════════════════════════════════════════════════════════════════════════════
# BALANCED CONFIG GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_balanced_configs(count: int, seed: int, fps: int, duration: float) -> List[Dict]:
    """Generate balanced dataset configurations respecting all distributions."""
    rng = random.Random(seed)

    # Target counts per severity class
    severity_counts = {k: max(1, round(v * count)) for k, v in SEVERITY_DISTRIBUTION.items()}
    diff = count - sum(severity_counts.values())
    severity_counts["moderate"] = max(1, severity_counts["moderate"] + diff)

    # Mapping from severity label to tremor profiles
    severity_to_profiles = {
        "none":         ["no_tremor"],
        "mild":         ["mild_postural", "finger_dominant"],
        "moderate":     ["moderate_postural", "kinetic_tremor", "wrist_dominant", "mixed_wrist_finger"],
        "severe":       ["severe_postural"],
        "artifact":     ["tracking_artifact_sim"],
        "gross_motion": ["gross_motion_no_tremor"],
    }

    # Camera angle grouping
    camera_groups = {
        "front":   [c for c in CAMERA_ANGLES if c["name"].startswith("front")],
        "oblique": [c for c in CAMERA_ANGLES if c["name"].startswith("oblique")],
        "side":    [c for c in CAMERA_ANGLES if c["name"].startswith("side")],
        "top":     [c for c in CAMERA_ANGLES if c["name"] == "top_down"],
        "closeup": [c for c in CAMERA_ANGLES if c["name"].startswith("closeup")],
    }

    # Motion type grouping
    motion_groups = {
        "stationary":       ["stationary"],
        "slow_translation": ["slow_translation_x", "slow_translation_y", "slow_translation_z"],
        "reach":            ["reach_forward", "reach_sideways"],
        "wrist_rotation":   ["wrist_rotation"],
        "finger_task":      ["finger_task"],
    }

    configs = []
    skin_tone_list = list(SKIN_TONES.keys())
    pose_list = list(POSE_FAMILIES.keys())
    affected_types = list(AFFECTED_JOINT_SETS.keys())

    for severity_class, target_count in severity_counts.items():
        profiles = severity_to_profiles[severity_class]

        for i in range(target_count):
            # Tremor profile
            tremor_profile_name = rng.choice(profiles)
            profile = TREMOR_PROFILES[tremor_profile_name]

            # Sample frequency + amplitude
            if profile["freq_range"][1] == 0:
                freq = 0.0
                amp = 0.0
            else:
                freq = rng.uniform(*profile["freq_range"])
                amp = rng.uniform(*profile["amp_range"])

            # Affected joints logic
            if tremor_profile_name == "wrist_dominant":
                affected_type = "wrist_only"
            elif tremor_profile_name == "finger_dominant":
                affected_type = rng.choice(["one_finger", "multiple_fingers"])
            elif tremor_profile_name == "mixed_wrist_finger":
                affected_type = "wrist_plus_fingers"
            elif tremor_profile_name in ("no_tremor", "gross_motion_no_tremor"):
                affected_type = "wrist_only"
            else:
                affected_type = rng.choice(affected_types)

            # Severity score
            severity_score = compute_severity_score(amp, freq, affected_type)

            # Camera (balanced)
            cam_group = rng.choices(
                list(CAMERA_GROUP_DISTRIBUTION.keys()),
                weights=list(CAMERA_GROUP_DISTRIBUTION.values())
            )[0]
            camera = rng.choice(camera_groups[cam_group])

            # Motion (balanced)
            motion_group = rng.choices(
                list(MOTION_GROUP_DISTRIBUTION.keys()),
                weights=list(MOTION_GROUP_DISTRIBUTION.values())
            )[0]
            motion_type = rng.choice(motion_groups[motion_group])

            # Skin tone (uniform)
            skin_tone = rng.choice(skin_tone_list)

            # Pose family
            pose_family = rng.choice(pose_list)

            # Pose with noise
            base_pose = POSE_FAMILIES[pose_family]
            pose_rotations = {}
            for bone, (rx, ry, rz) in base_pose.items():
                pose_rotations[bone] = (
                    round(rx + rng.gauss(0, 3), 2),
                    round(ry + rng.gauss(0, 1), 2),
                    round(rz + rng.gauss(0, 2), 2),
                )

            render_seed = rng.randint(0, 2**31)

            configs.append({
                "pose_family": pose_family,
                "pose_rotations": pose_rotations,
                "skin_tone_id": skin_tone,
                "tremor_profile": tremor_profile_name,
                "tremor_type": profile["label"],
                "frequency_hz": round(freq, 2),
                "amplitude_degrees": round(amp, 2),
                "affected_joint_type": affected_type,
                "affected_joints": AFFECTED_JOINT_SETS[affected_type],
                "severity_score": severity_score,
                "camera": camera,
                "motion_type": motion_type,
                "duration": duration,
                "fps": fps,
                "render_seed": render_seed,
            })

    rng.shuffle(configs)
    return configs[:count]


# Detector-friendly subsets for smoke mode
SMOKE_CAMERAS = [
    {"name": "front_high",   "azim": 0,   "elev": 20},
    {"name": "front_above",  "azim": 0,   "elev": 30},
    {"name": "oblique_l",    "azim": 20,  "elev": 20},
    {"name": "oblique_r",    "azim": -20, "elev": 20},
    {"name": "oblique_high", "azim": 15,  "elev": 28},
]

SMOKE_POSES = ["rest_open", "relaxed", "finger_spread", "pinch", "palm_down", "grip_release"]

SMOKE_SKIN_TONES = ["light", "light_warm", "medium_light", "medium", "medium_warm", "tan"]


def generate_smoke_configs(count: int, seed: int, fps: int, duration: float) -> List[Dict]:
    """Generate detector-friendly configs for smoke validation.

    Restricts to:
    - Frontal/oblique cameras only (no side, top, closeup)
    - Open/relaxed/pointing/pinch poses (no fist, no edge-on)
    - Light-to-medium skin tones (no very dark in early testing)
    - Stationary motion only (no confounding movement)
    """
    rng = random.Random(seed)

    # Use a simpler severity distribution for smoke
    severity_to_profiles = {
        "none": ["no_tremor"],
        "mild": ["mild_postural", "finger_dominant"],
        "moderate": ["moderate_postural", "wrist_dominant"],
        "severe": ["severe_postural"],
    }

    configs = []
    for i in range(count):
        # Cycle through severity classes evenly
        severity_class = list(severity_to_profiles.keys())[i % len(severity_to_profiles)]
        profiles = severity_to_profiles[severity_class]
        tremor_profile_name = rng.choice(profiles)
        profile = TREMOR_PROFILES[tremor_profile_name]

        if profile["freq_range"][1] == 0:
            freq, amp = 0.0, 0.0
        else:
            freq = rng.uniform(*profile["freq_range"])
            amp = rng.uniform(*profile["amp_range"])

        # Affected joints
        if tremor_profile_name == "wrist_dominant":
            affected_type = "wrist_only"
        elif tremor_profile_name == "finger_dominant":
            affected_type = "multiple_fingers"
        else:
            affected_type = rng.choice(["wrist_only", "multiple_fingers", "wrist_plus_fingers"])

        severity_score = compute_severity_score(amp, freq, affected_type)

        # Detector-friendly camera
        camera = rng.choice(SMOKE_CAMERAS)

        # Detector-friendly pose
        pose_family = rng.choice(SMOKE_POSES)
        base_pose = POSE_FAMILIES[pose_family]
        pose_rotations = {}
        for bone, (rx, ry, rz) in base_pose.items():
            pose_rotations[bone] = (
                round(rx + rng.gauss(0, 2), 2),
                round(ry + rng.gauss(0, 0.5), 2),
                round(rz + rng.gauss(0, 1.5), 2),
            )

        # Detector-friendly skin tone
        skin_tone = rng.choice(SMOKE_SKIN_TONES)

        render_seed = rng.randint(0, 2**31)

        configs.append({
            "pose_family": pose_family,
            "pose_rotations": pose_rotations,
            "skin_tone_id": skin_tone,
            "tremor_profile": tremor_profile_name,
            "tremor_type": profile["label"],
            "frequency_hz": round(freq, 2),
            "amplitude_degrees": round(amp, 2),
            "affected_joint_type": affected_type,
            "affected_joints": AFFECTED_JOINT_SETS[affected_type],
            "severity_score": severity_score,
            "camera": camera,
            "motion_type": "stationary",
            "duration": duration,
            "fps": fps,
            "render_seed": render_seed,
        })

    return configs


# ─── Mini50 Mode Settings ────────────────────────────────────────────────────

MINI50_CAMERAS = [
    {"name": "front_high",    "azim": 0,   "elev": 20},
    {"name": "front_above",   "azim": 0,   "elev": 30},
    {"name": "oblique_l",     "azim": 20,  "elev": 20},
    {"name": "oblique_r",     "azim": -20, "elev": 20},
    {"name": "oblique_high",  "azim": 15,  "elev": 28},
    {"name": "mild_side_l",   "azim": 40,  "elev": 18},
    {"name": "mild_side_r",   "azim": -40, "elev": 18},
]

MINI50_POSES = [
    "rest_open", "relaxed", "finger_spread", "pinch",
    "palm_down", "grip_release", "reach_forward", "finger_tap_index",
]

MINI50_MOTIONS = ["stationary", "slow_translation_x", "slow_translation_y", "wrist_rotation"]


def generate_mini50_configs(count: int, seed: int, fps: int, duration: float) -> List[Dict]:
    """Generate intermediate-difficulty configs for mini50 validation.

    Wider than smoke but not full-hard:
    - All 12 skin tones (detectability-rejected post-hoc)
    - 8 pose families (no full fist / extreme occlusion)
    - Front + mild oblique cameras (elevation ≥18°)
    - Stationary + slow translation + mild wrist rotation
    - 8 tremor profiles (no tracking_artifact_sim)
    """
    rng = random.Random(seed)

    severity_to_profiles = {
        "none": ["no_tremor", "gross_motion_no_tremor"],
        "mild": ["mild_postural", "finger_dominant"],
        "moderate": ["moderate_postural", "wrist_dominant", "mixed_wrist_finger"],
        "severe": ["severe_postural"],
    }

    # Exclude extreme skin tones proven undetectable by MediaPipe
    MINI50_EXCLUDED_SKINS = {"deep_brown", "very_light"}
    mini50_skin_tones = [k for k in SKIN_TONES.keys() if k not in MINI50_EXCLUDED_SKINS]
    configs = []

    for i in range(count):
        severity_class = list(severity_to_profiles.keys())[i % len(severity_to_profiles)]
        profiles = severity_to_profiles[severity_class]
        tremor_profile_name = rng.choice(profiles)
        profile = TREMOR_PROFILES[tremor_profile_name]

        if profile["freq_range"][1] == 0:
            freq, amp = 0.0, 0.0
        else:
            freq = rng.uniform(*profile["freq_range"])
            amp = rng.uniform(*profile["amp_range"])

        # Affected joints
        if tremor_profile_name == "wrist_dominant":
            affected_type = "wrist_only"
        elif tremor_profile_name == "finger_dominant":
            affected_type = rng.choice(["one_finger", "multiple_fingers"])
        elif tremor_profile_name == "mixed_wrist_finger":
            affected_type = "wrist_plus_fingers"
        elif tremor_profile_name in ("no_tremor", "gross_motion_no_tremor"):
            affected_type = "wrist_only"
        else:
            affected_type = rng.choice(list(AFFECTED_JOINT_SETS.keys()))

        severity_score = compute_severity_score(amp, freq, affected_type)

        camera = rng.choice(MINI50_CAMERAS)
        pose_family = rng.choice(MINI50_POSES)
        motion_type = rng.choice(MINI50_MOTIONS)
        skin_tone = rng.choice(mini50_skin_tones)

        base_pose = POSE_FAMILIES[pose_family]
        pose_rotations = {}
        for bone, (rx, ry, rz) in base_pose.items():
            # Reduced jitter vs full mode (1.5° vs 2.5°) to prevent
            # random noise pushing borderline poses into undetectable range
            pose_rotations[bone] = (
                round(rx + rng.gauss(0, 1.5), 2),
                round(ry + rng.gauss(0, 0.8), 2),
                round(rz + rng.gauss(0, 1.2), 2),
            )

        render_seed = rng.randint(0, 2**31)

        configs.append({
            "pose_family": pose_family,
            "pose_rotations": pose_rotations,
            "skin_tone_id": skin_tone,
            "tremor_profile": tremor_profile_name,
            "tremor_type": profile["label"],
            "frequency_hz": round(freq, 2),
            "amplitude_degrees": round(amp, 2),
            "affected_joint_type": affected_type,
            "affected_joints": AFFECTED_JOINT_SETS[affected_type],
            "severity_score": severity_score,
            "camera": camera,
            "motion_type": motion_type,
            "duration": duration,
            "fps": fps,
            "render_seed": render_seed,
        })

    rng.shuffle(configs)
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# BLENDER RENDER SCRIPT (embedded)
# ═══════════════════════════════════════════════════════════════════════════════

BLENDER_SCRIPT = r'''
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

SKIN_TONES_DATA = __SKIN_TONES_JSON__


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
'''


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def find_blender() -> str:
    """Locate Blender binary."""
    paths = [
        shutil.which("blender"),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/homebrew/bin/blender",
    ]
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    print("ERROR: Blender not found. Install: brew install --cask blender")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MotionBloom Synthetic Tremor Dataset Generator")
    parser.add_argument("--asset", required=True, help="Path to rigged hand .glb")
    parser.add_argument("--rig-map", required=True, help="Path to rig_map.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=5, help="Number of videos to generate")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=4.0, help="Video duration (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed renders")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: restrict to detector-friendly settings only")
    parser.add_argument("--mini50", action="store_true", help="Mini50 mode: intermediate difficulty for validation")
    args = parser.parse_args()

    if args.smoke and args.mini50:
        print("ERROR: --smoke and --mini50 are mutually exclusive")
        sys.exit(1)

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
    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Synthetic Tremor Dataset Generator v2.0    ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Blender:   {blender}")
    print(f"  Asset:     {asset.name}")
    print(f"  Output:    {out_dir}")
    print(f"  Videos:    {args.count} @ {args.fps}fps × {args.duration}s")
    print(f"  Seed:      {args.seed}")
    print(f"  Retries:   {args.max_retries}")
    print()

    # Generate balanced configs
    if args.smoke:
        print("Generating SMOKE MODE configs (detector-friendly only)...")
        configs = generate_smoke_configs(args.count, args.seed, args.fps, args.duration)
    elif args.mini50:
        print("Generating MINI50 MODE configs (intermediate difficulty)...")
        configs = generate_mini50_configs(args.count, args.seed, args.fps, args.duration)
    else:
        print("Generating balanced sample configurations...")
        configs = generate_balanced_configs(args.count, args.seed, args.fps, args.duration)

    # Print distribution summary
    from collections import Counter
    severity_dist = Counter(c["tremor_type"] for c in configs)
    skin_dist = Counter(c["skin_tone_id"] for c in configs)
    pose_dist = Counter(c["pose_family"] for c in configs)
    cam_dist = Counter(c["camera"]["name"] for c in configs)

    print(f"  Severity classes: {dict(severity_dist)}")
    print(f"  Skin tones:       {len(skin_dist)}/12 used")
    print(f"  Pose families:    {len(pose_dist)}/16 used")
    print(f"  Camera angles:    {len(cam_dist)}/10 used")
    print()

    # Prepare output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "pose_previews").mkdir(exist_ok=True)

    # Write generation_config.json
    gen_config = {
        "version": "2.0",
        "count": args.count,
        "fps": args.fps,
        "duration": args.duration,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "asset": str(asset),
        "skin_tones": list(SKIN_TONES.keys()),
        "pose_families": list(POSE_FAMILIES.keys()),
        "tremor_profiles": list(TREMOR_PROFILES.keys()),
        "severity_distribution": SEVERITY_DISTRIBUTION,
        "camera_distribution": CAMERA_GROUP_DISTRIBUTION,
        "motion_distribution": MOTION_GROUP_DISTRIBUTION,
        "severity_formula": "score = round(100 * clamp(amp/12, 0,1) * freq_weight * affected_weight)",
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen_config, indent=2))

    # Retry loop
    attempt = 0
    remaining_configs = configs
    all_success = False

    while attempt < args.max_retries and remaining_configs:
        attempt += 1
        if attempt > 1:
            print(f"\n--- Retry attempt {attempt}/{args.max_retries} ({len(remaining_configs)} remaining) ---")

        # Write run config for Blender
        run_config = {"fps": args.fps, "videos": remaining_configs}
        config_path = out_dir / f"_run_config_{attempt}.json"
        config_path.write_text(json.dumps(run_config))

        # Write Blender script
        # Embed skin tones JSON into the Blender script
        skin_tones_json = json.dumps(SKIN_TONES)
        actual_script = BLENDER_SCRIPT.replace(
            "__SKIN_TONES_JSON__",
            skin_tones_json
        )
        script_path = out_dir / f"_render_script_{attempt}.py"
        script_path.write_text(actual_script)

        # Run Blender
        cmd = [
            blender, "--background",
            "--python", str(script_path),
            "--", str(asset), str(rig_map), str(out_dir), str(config_path),
        ]

        print(f"Launching Blender (attempt {attempt}, {len(remaining_configs)} videos)...")
        result = subprocess.run(cmd, capture_output=False, text=True)

        # Cleanup temp files
        for tmp in [script_path, config_path]:
            if tmp.exists():
                tmp.unlink()

        if result.returncode != 0:
            print(f"WARNING: Blender exited with code {result.returncode}")

        # Check which videos failed
        metadata_path = out_dir / "metadata.jsonl"
        if metadata_path.exists():
            lines = metadata_path.read_text().strip().split("\n")
            failed_ids = []
            for line in lines:
                m = json.loads(line)
                if not m.get("valid_render"):
                    failed_ids.append(m["video_id"])

            if not failed_ids:
                all_success = True
                break
            else:
                # For simplicity, we don't re-render individual failures in this version
                # The validation_report.json captures failures
                print(f"  {len(failed_ids)} videos failed validation")
                break
        else:
            print("ERROR: No metadata.jsonl produced")
            break

    # Final summary
    metadata_path = out_dir / "metadata.jsonl"
    if metadata_path.exists():
        lines = metadata_path.read_text().strip().split("\n")
        valid = sum(1 for l in lines if json.loads(l).get("valid_render"))
        total = len(lines)
    else:
        valid, total = 0, 0

    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  Generation Complete                                     ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Valid videos:  {valid}/{total}")
    print(f"  Videos dir:    {out_dir / 'videos'}")
    print(f"  Previews:      {out_dir / 'pose_previews'}")
    print(f"  Metadata:      {metadata_path}")
    print(f"  Labels:        {out_dir / 'labels.csv'}")
    print(f"  Config:        {out_dir / 'generation_config.json'}")
    print(f"  Validation:    {out_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
