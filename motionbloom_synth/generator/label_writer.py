"""Label Writer.

Generates ground-truth labels and metadata for each synthetic video.
Since we control the generation parameters, labels are deterministic —
no manual annotation required.

Scoring formula:
    severity = 0.55*amplitude_norm + 0.25*band_power_norm
             + 0.10*frequency_confidence + 0.10*duration_consistency
    score = round(100 * severity)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


@dataclass
class VideoMetadata:
    """Complete metadata record for one generated video."""

    video_id: str
    pose: str
    tremor_type: str
    severity_score: int
    frequency_hz: float
    amplitude_degrees: float
    affected_joints: List[str]
    camera_angle: str
    motion_type: str
    lighting: str
    background: str
    skin_tone: str
    degradation: str
    fps: int
    duration_sec: float
    # Derived fields
    band_power_norm: float = 0.0
    frequency_confidence: float = 0.0
    duration_consistency: float = 0.0
    validity_class: str = "VALID_TREMOR"


def compute_severity_score(
    amplitude_degrees: float,
    frequency_hz: float,
    tremor_type: str,
    duration_sec: float,
) -> int:
    """Compute deterministic severity score from generation parameters.

    Args:
        amplitude_degrees: Peak tremor amplitude.
        frequency_hz: Dominant tremor frequency.
        tremor_type: Type of tremor generated.
        duration_sec: How long tremor is present.

    Returns:
        Integer score 0–100.
    """
    config = _load_scoring_config()
    weights = config["scoring_weights"]
    norms = config["scoring_normalization"]

    # Handle non-tremor types
    if tremor_type == "GROSS_HAND_MOVEMENT_NO_TREMOR":
        return 0
    if tremor_type == "TRACKING_ARTIFACT":
        return 0

    # Amplitude normalization (0–1)
    amp_max = norms["amplitude_max_degrees"]
    amplitude_norm = min(amplitude_degrees / amp_max, 1.0)

    # Band power normalization
    # Tremor in the 3–12 Hz band: higher amplitude → more power
    band_range = norms["band_power_hz_range"]
    in_band = band_range[0] <= frequency_hz <= band_range[1]
    band_power_norm = amplitude_norm if in_band else amplitude_norm * 0.3

    # Frequency confidence: peaks near typical tremor frequencies score higher
    # Optimal range: 4–8 Hz for most clinical tremors
    if 4.0 <= frequency_hz <= 8.0:
        frequency_confidence = 1.0
    elif 3.0 <= frequency_hz <= 12.0:
        frequency_confidence = 0.7
    else:
        frequency_confidence = 0.3

    # Duration consistency: longer presence = higher confidence
    duration_consistency = min(duration_sec / 4.0, 1.0)

    # Weighted sum
    severity = (
        weights["amplitude_norm"] * amplitude_norm
        + weights["band_power_norm"] * band_power_norm
        + weights["frequency_confidence"] * frequency_confidence
        + weights["duration_consistency"] * duration_consistency
    )

    return round(100 * severity)


def determine_validity_class(tremor_type: str, severity_score: int) -> str:
    """Determine Model A classification target.

    Returns one of:
        NO_HAND, TRACKING_UNSTABLE, GROSS_MOVEMENT_ONLY,
        VALID_TREMOR, ARTIFACT
    """
    if tremor_type == "TRACKING_ARTIFACT":
        return "ARTIFACT"
    if tremor_type == "GROSS_HAND_MOVEMENT_NO_TREMOR":
        return "GROSS_MOVEMENT_ONLY"
    if severity_score == 0:
        return "GROSS_MOVEMENT_ONLY"
    return "VALID_TREMOR"


def create_metadata(
    video_id: str,
    pose: str,
    tremor_type: str,
    frequency_hz: float,
    amplitude_degrees: float,
    affected_joints: List[str],
    camera_angle: str,
    motion_type: str,
    lighting: str,
    background: str,
    skin_tone: str = "medium",
    degradation: str = "none",
    fps: int = 30,
    duration_sec: float = 4.0,
) -> VideoMetadata:
    """Create complete metadata record with computed severity score.

    All parameters come from the generation pipeline — nothing is estimated.
    """
    severity_score = compute_severity_score(
        amplitude_degrees, frequency_hz, tremor_type, duration_sec
    )
    validity_class = determine_validity_class(tremor_type, severity_score)

    # Derived confidence metrics
    config = _load_scoring_config()
    norms = config["scoring_normalization"]
    amp_max = norms["amplitude_max_degrees"]
    band_range = norms["band_power_hz_range"]

    amplitude_norm = min(amplitude_degrees / amp_max, 1.0)
    in_band = band_range[0] <= frequency_hz <= band_range[1]
    band_power_norm = amplitude_norm if in_band else amplitude_norm * 0.3

    if 4.0 <= frequency_hz <= 8.0:
        frequency_confidence = 1.0
    elif 3.0 <= frequency_hz <= 12.0:
        frequency_confidence = 0.7
    else:
        frequency_confidence = 0.3

    duration_consistency = min(duration_sec / 4.0, 1.0)

    return VideoMetadata(
        video_id=video_id,
        pose=pose,
        tremor_type=tremor_type,
        severity_score=severity_score,
        frequency_hz=round(frequency_hz, 2),
        amplitude_degrees=round(amplitude_degrees, 2),
        affected_joints=affected_joints,
        camera_angle=camera_angle,
        motion_type=motion_type,
        lighting=lighting,
        background=background,
        skin_tone=skin_tone,
        degradation=degradation,
        fps=fps,
        duration_sec=duration_sec,
        band_power_norm=round(band_power_norm, 4),
        frequency_confidence=frequency_confidence,
        duration_consistency=round(duration_consistency, 4),
        validity_class=validity_class,
    )


def write_metadata_json(metadata: VideoMetadata, output_dir: Optional[Path] = None):
    """Write single video metadata as JSON sidecar file.

    Args:
        metadata: Video metadata record.
        output_dir: Directory to write to (default: outputs/).
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{metadata.video_id}.json"
    with open(filepath, "w") as f:
        json.dump(asdict(metadata), f, indent=2)


def write_labels_csv(
    metadata_list: List[VideoMetadata],
    output_path: Optional[Path] = None,
):
    """Write all video labels to a single CSV file.

    Args:
        metadata_list: All generated video metadata records.
        output_path: CSV file path (default: outputs/labels.csv).
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata_list:
        return

    fieldnames = [
        "video_id", "pose", "tremor_type", "severity_score",
        "frequency_hz", "amplitude_degrees", "camera_angle",
        "motion_type", "lighting", "background", "skin_tone",
        "degradation", "fps", "duration_sec", "validity_class",
        "band_power_norm", "frequency_confidence", "duration_consistency",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metadata_list:
            row = asdict(m)
            row["affected_joints"] = "|".join(row["affected_joints"])
            writer.writerow({k: row[k] for k in fieldnames})


def write_metadata_jsonl(
    metadata_list: List[VideoMetadata],
    output_path: Optional[Path] = None,
):
    """Write all metadata as JSON Lines (one JSON object per line).

    Args:
        metadata_list: All generated video metadata records.
        output_path: JSONL file path (default: outputs/metadata.jsonl).
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "metadata.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for m in metadata_list:
            f.write(json.dumps(asdict(m)) + "\n")


def _load_scoring_config() -> dict:
    """Load scoring weights and normalization from tremor_profiles.yaml."""
    config_path = CONFIGS_DIR / "tremor_profiles.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
