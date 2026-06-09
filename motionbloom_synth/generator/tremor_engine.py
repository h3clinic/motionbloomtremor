"""Tremor Injection Engine.

Injects physiologically realistic tremor at the bone/joint transform level.

Core model:
    joint_angle(t) = base_angle + tremor_component + noise_component

Tremor function with biological imperfection:
    theta = A*sin(2π*f*t + φ)
          + 0.25*A*sin(2π*1.7*f*t + φ2)
          + gaussian_noise
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class TremorParams:
    """Parameters for a single joint's tremor oscillation."""

    frequency_hz: float
    amplitude_degrees: float
    phase: float = 0.0
    phase2: float = 0.0  # Second harmonic phase
    # Noise model
    harmonic_ratio: float = 1.7
    harmonic_amplitude_fraction: float = 0.25
    gaussian_noise_std: float = 0.0
    # Envelope modulation
    amplitude_mod_hz: float = 0.3
    amplitude_mod_depth: float = 0.15
    phase_drift_rate: float = 0.3


@dataclass
class TremorProfile:
    """Complete tremor specification for one video clip."""

    tremor_type: str
    severity_score: int  # 0–100
    frequency_hz: float
    amplitude_degrees: float
    affected_joints: List[str]
    joint_params: Dict[str, TremorParams] = field(default_factory=dict)
    joint_coherence: float = 0.7
    onset_pattern: str = "constant"


def load_tremor_config() -> dict:
    """Load tremor profiles YAML config."""
    config_path = CONFIGS_DIR / "tremor_profiles.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_tremor_profile(
    tremor_type: str,
    severity_score: int,
    frequency_hz: Optional[float] = None,
    affected_joints: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> TremorProfile:
    """Create a complete tremor profile from type and severity.

    Args:
        tremor_type: One of the defined tremor types in config.
        severity_score: 0–100 severity.
        frequency_hz: Override frequency (None = sample from type's range).
        affected_joints: Override joint list (None = use type defaults).
        seed: Random seed for reproducibility.

    Returns:
        TremorProfile with per-joint parameters.
    """
    if seed is not None:
        random.seed(seed)

    config = load_tremor_config()
    type_config = config["tremor_types"][tremor_type]
    noise_config = config["noise_model"]

    # Determine frequency
    freq_range = type_config["frequency_range"]
    if frequency_hz is None:
        frequency_hz = random.uniform(freq_range[0], freq_range[1])
    else:
        frequency_hz = max(freq_range[0], min(freq_range[1], frequency_hz))

    # Determine amplitude from severity
    amplitude = _severity_to_amplitude(severity_score, config)

    # Handle non-tremor types
    if tremor_type == "GROSS_HAND_MOVEMENT_NO_TREMOR":
        amplitude = 0.0
    elif tremor_type == "TRACKING_ARTIFACT":
        # Override amplitude to tiny noise values
        amplitude = random.uniform(0.1, 0.5)

    # Determine affected joints
    if affected_joints is None:
        affected_joints = list(type_config["typical_joints"])

    coherence = type_config["joint_coherence"]

    # Generate per-joint tremor parameters
    # Base phase (shared across coherent joints)
    base_phase = random.uniform(0, 2 * math.pi)
    base_phase2 = random.uniform(0, 2 * math.pi)

    joint_params: Dict[str, TremorParams] = {}
    for joint in affected_joints:
        # Phase variation based on coherence
        if coherence >= 1.0:
            phase = base_phase
            phase2 = base_phase2
        else:
            phase_spread = (1.0 - coherence) * math.pi
            phase = base_phase + random.gauss(0, phase_spread)
            phase2 = base_phase2 + random.gauss(0, phase_spread)

        # Amplitude variation per joint (±20% around target)
        joint_amplitude = amplitude * random.uniform(0.8, 1.2)

        # Frequency micro-variation (±5%)
        joint_freq = frequency_hz * random.uniform(0.95, 1.05)

        joint_params[joint] = TremorParams(
            frequency_hz=joint_freq,
            amplitude_degrees=joint_amplitude,
            phase=phase,
            phase2=phase2,
            harmonic_ratio=noise_config["harmonic_ratio"],
            harmonic_amplitude_fraction=noise_config["harmonic_amplitude_fraction"],
            gaussian_noise_std=noise_config["gaussian_noise_fraction"] * joint_amplitude,
            amplitude_mod_hz=noise_config["amplitude_modulation_hz"],
            amplitude_mod_depth=noise_config["amplitude_modulation_depth"],
            phase_drift_rate=noise_config["phase_drift_rate"],
        )

    return TremorProfile(
        tremor_type=tremor_type,
        severity_score=severity_score,
        frequency_hz=frequency_hz,
        amplitude_degrees=amplitude,
        affected_joints=affected_joints,
        joint_params=joint_params,
        joint_coherence=coherence,
        onset_pattern=type_config["onset_pattern"],
    )


def compute_tremor_angle(params: TremorParams, t: float) -> float:
    """Compute instantaneous tremor angle offset at time t.

    Args:
        params: Joint tremor parameters.
        t: Time in seconds.

    Returns:
        Angle offset in degrees to add to base pose.
    """
    f = params.frequency_hz
    A = params.amplitude_degrees

    if A == 0:
        return 0.0

    # Phase drift over time
    phi = params.phase + params.phase_drift_rate * t
    phi2 = params.phase2 + params.phase_drift_rate * 0.7 * t

    # Amplitude envelope modulation
    envelope = 1.0 + params.amplitude_mod_depth * math.sin(
        2 * math.pi * params.amplitude_mod_hz * t
    )

    # Primary oscillation
    primary = A * math.sin(2 * math.pi * f * t + phi)

    # Second harmonic (biological imperfection)
    harmonic_amp = params.harmonic_amplitude_fraction * A
    harmonic = harmonic_amp * math.sin(
        2 * math.pi * params.harmonic_ratio * f * t + phi2
    )

    # Gaussian noise
    noise = random.gauss(0, params.gaussian_noise_std) if params.gaussian_noise_std > 0 else 0.0

    return envelope * (primary + harmonic) + noise


def compute_frame_angles(
    profile: TremorProfile,
    base_pose: Dict[str, float],
    t: float,
) -> Dict[str, float]:
    """Compute all joint angles for a single frame.

    Args:
        profile: Tremor profile with per-joint params.
        base_pose: Base bone rotations (degrees) from pose library.
        t: Time in seconds.

    Returns:
        Dict of joint_name -> final_angle_degrees.
    """
    result = dict(base_pose)

    for joint_name, params in profile.joint_params.items():
        base_angle = base_pose.get(joint_name, 0.0)
        tremor_offset = compute_tremor_angle(params, t)
        result[joint_name] = base_angle + tremor_offset

    return result


def generate_animation_curve(
    profile: TremorProfile,
    base_pose: Dict[str, float],
    duration_sec: float,
    fps: int = 30,
) -> List[Dict[str, float]]:
    """Generate full animation: list of per-frame joint angle dicts.

    Args:
        profile: Tremor profile.
        base_pose: Static base pose angles.
        duration_sec: Video length in seconds.
        fps: Frames per second.

    Returns:
        List of dicts (one per frame), each mapping joint -> angle.
    """
    num_frames = int(duration_sec * fps)
    frames = []

    for frame_idx in range(num_frames):
        t = frame_idx / fps
        angles = compute_frame_angles(profile, base_pose, t)
        frames.append(angles)

    return frames


def _severity_to_amplitude(severity_score: int, config: dict) -> float:
    """Interpolate amplitude from severity score using config levels."""
    levels = config["severity_levels"]
    sorted_scores = sorted(levels.keys())

    # Clamp
    if severity_score <= sorted_scores[0]:
        return levels[sorted_scores[0]]["amplitude_degrees"]
    if severity_score >= sorted_scores[-1]:
        return levels[sorted_scores[-1]]["amplitude_degrees"]

    # Find bracketing levels and interpolate
    for i in range(len(sorted_scores) - 1):
        lo = sorted_scores[i]
        hi = sorted_scores[i + 1]
        if lo <= severity_score <= hi:
            frac = (severity_score - lo) / (hi - lo)
            amp_lo = levels[lo]["amplitude_degrees"]
            amp_hi = levels[hi]["amplitude_degrees"]
            return amp_lo + frac * (amp_hi - amp_lo)

    return 0.0
