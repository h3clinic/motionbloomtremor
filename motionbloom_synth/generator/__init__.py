"""Generator package for synthetic tremor dataset."""

from .tremor_engine import (
    TremorParams,
    TremorProfile,
    create_tremor_profile,
    compute_tremor_angle,
    compute_frame_angles,
    generate_animation_curve,
)
from .label_writer import (
    VideoMetadata,
    compute_severity_score,
    create_metadata,
    write_labels_csv,
    write_metadata_jsonl,
)
from .hand_rig import load_poses, load_rig_map, get_hand_model_path
from .camera_engine import load_camera_config, get_camera_position

__all__ = [
    "TremorParams",
    "TremorProfile",
    "create_tremor_profile",
    "compute_tremor_angle",
    "compute_frame_angles",
    "generate_animation_curve",
    "VideoMetadata",
    "compute_severity_score",
    "create_metadata",
    "write_labels_csv",
    "write_metadata_jsonl",
    "load_poses",
    "load_rig_map",
    "get_hand_model_path",
    "load_camera_config",
    "get_camera_position",
]
