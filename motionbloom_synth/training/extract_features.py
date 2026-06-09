"""Feature Extraction from Normalized Landmarks.

Converts box-normalized hand landmarks into tremor-specific features:
  - Fingertip displacement signal (time series)
  - Spectral features (dominant frequency, band power)
  - Temporal features (amplitude, consistency)

Pipeline position:
    video → landmarks → **features** → model

This is where the actual tremor signal lives — not in raw pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

SYNTH_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = SYNTH_ROOT / "outputs"

# MediaPipe hand landmark indices for fingertips
FINGERTIP_INDICES = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}

# MCP joint indices (for MCP-to-tip displacement)
MCP_INDICES = {
    "index_mcp": 5,
    "middle_mcp": 9,
    "ring_mcp": 13,
    "pinky_mcp": 17,
}

# Wrist landmark
WRIST_INDEX = 0

# Tremor frequency band
TREMOR_BAND_HZ = (3.0, 12.0)


def extract_fingertip_signals(
    landmarks_normalized: np.ndarray,
    detection_mask: np.ndarray,
    fps: int = 30,
) -> Dict[str, np.ndarray]:
    """Extract per-fingertip displacement time series.

    Uses box-normalized landmarks so macro hand movement is removed.
    The signal is the frame-to-frame displacement of each fingertip.

    Args:
        landmarks_normalized: Shape (num_frames, 21, 3), box-normalized.
        detection_mask: Shape (num_frames,), True where hand detected.
        fps: Video frame rate.

    Returns:
        Dict mapping fingertip_name -> displacement signal (num_frames-1,).
    """
    signals = {}

    for name, idx in FINGERTIP_INDICES.items():
        tip_positions = landmarks_normalized[:, idx, :]  # (T, 3)

        # Frame-to-frame displacement magnitude
        diff = np.diff(tip_positions, axis=0)  # (T-1, 3)
        displacement = np.linalg.norm(diff, axis=1)  # (T-1,)

        # Zero out where detection failed
        valid = detection_mask[:-1] & detection_mask[1:]
        displacement[~valid] = 0.0

        signals[name] = displacement

    return signals


def extract_wrist_signal(
    landmarks_normalized: np.ndarray,
    detection_mask: np.ndarray,
) -> np.ndarray:
    """Extract wrist displacement signal (macro movement indicator).

    Args:
        landmarks_normalized: Shape (num_frames, 21, 3).
        detection_mask: Shape (num_frames,).

    Returns:
        Wrist displacement signal (num_frames-1,).
    """
    wrist = landmarks_normalized[:, WRIST_INDEX, :]
    diff = np.diff(wrist, axis=0)
    displacement = np.linalg.norm(diff, axis=1)

    valid = detection_mask[:-1] & detection_mask[1:]
    displacement[~valid] = 0.0

    return displacement


def compute_spectral_features(
    signal_data: np.ndarray,
    fps: int = 30,
    band_hz: Tuple[float, float] = TREMOR_BAND_HZ,
) -> Dict[str, float]:
    """Compute frequency-domain features from a displacement signal.

    Args:
        signal_data: 1D time series (frame-to-frame displacement).
        fps: Sampling rate.
        band_hz: Frequency band of interest (tremor band).

    Returns:
        Dict with spectral features.
    """
    n = len(signal_data)
    if n < fps:  # Need at least 1 second of data
        return {
            "dominant_frequency_hz": 0.0,
            "band_power": 0.0,
            "total_power": 0.0,
            "band_power_ratio": 0.0,
            "spectral_entropy": 0.0,
        }

    # Compute PSD using Welch's method
    nperseg = min(n, fps * 2)  # 2-second windows
    freqs, psd = scipy_signal.welch(
        signal_data, fs=fps, nperseg=nperseg, noverlap=nperseg // 2
    )

    # Total power
    total_power = np.trapz(psd, freqs)

    # Band power (tremor band)
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    band_power = np.trapz(psd[band_mask], freqs[band_mask]) if band_mask.any() else 0.0

    # Band power ratio
    band_power_ratio = band_power / total_power if total_power > 1e-10 else 0.0

    # Dominant frequency in tremor band
    if band_mask.any() and band_power > 1e-10:
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]
        dominant_idx = np.argmax(band_psd)
        dominant_frequency = band_freqs[dominant_idx]
    else:
        dominant_frequency = 0.0

    # Spectral entropy (measure of how "peaked" the spectrum is)
    psd_norm = psd / (psd.sum() + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    return {
        "dominant_frequency_hz": float(dominant_frequency),
        "band_power": float(band_power),
        "total_power": float(total_power),
        "band_power_ratio": float(band_power_ratio),
        "spectral_entropy": float(spectral_entropy),
    }


def compute_temporal_features(
    signal_data: np.ndarray,
    fps: int = 30,
) -> Dict[str, float]:
    """Compute time-domain features from displacement signal.

    Args:
        signal_data: 1D displacement time series.
        fps: Frame rate.

    Returns:
        Dict with temporal features.
    """
    if len(signal_data) == 0:
        return {
            "mean_amplitude": 0.0,
            "max_amplitude": 0.0,
            "std_amplitude": 0.0,
            "rms_amplitude": 0.0,
            "duration_above_threshold": 0.0,
            "zero_crossing_rate": 0.0,
        }

    mean_amp = float(np.mean(signal_data))
    max_amp = float(np.max(signal_data))
    std_amp = float(np.std(signal_data))
    rms_amp = float(np.sqrt(np.mean(signal_data ** 2)))

    # Duration above threshold (fraction of time tremor is "active")
    threshold = mean_amp + 0.5 * std_amp
    duration_above = float(np.mean(signal_data > threshold))

    # Zero-crossing rate of the de-meaned signal
    centered = signal_data - mean_amp
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    zcr = crossings / len(signal_data) * fps  # Crossings per second

    return {
        "mean_amplitude": mean_amp,
        "max_amplitude": max_amp,
        "std_amplitude": std_amp,
        "rms_amplitude": rms_amp,
        "duration_above_threshold": duration_above,
        "zero_crossing_rate": float(zcr),
    }


def extract_features_for_video(
    landmarks_path: Path,
    fps: int = 30,
) -> Dict[str, float]:
    """Extract complete feature vector for one video.

    Args:
        landmarks_path: Path to .npz file with extracted landmarks.
        fps: Video frame rate.

    Returns:
        Flat dict of feature_name -> value.
    """
    data = np.load(landmarks_path)
    landmarks_norm = data["landmarks_normalized"]
    mask = data["detection_mask"]

    features = {}

    # Detection quality
    detection_rate = float(mask.sum() / len(mask)) if len(mask) > 0 else 0
    features["detection_rate"] = detection_rate

    # Per-fingertip features
    fingertip_signals = extract_fingertip_signals(landmarks_norm, mask, fps)

    for name, sig in fingertip_signals.items():
        spectral = compute_spectral_features(sig, fps)
        temporal = compute_temporal_features(sig, fps)

        for k, v in spectral.items():
            features[f"{name}_{k}"] = v
        for k, v in temporal.items():
            features[f"{name}_{k}"] = v

    # Aggregate fingertip features (average across all fingertips)
    all_tips = np.stack(list(fingertip_signals.values()))  # (5, T-1)
    combined_signal = all_tips.mean(axis=0)

    agg_spectral = compute_spectral_features(combined_signal, fps)
    agg_temporal = compute_temporal_features(combined_signal, fps)

    for k, v in agg_spectral.items():
        features[f"aggregate_{k}"] = v
    for k, v in agg_temporal.items():
        features[f"aggregate_{k}"] = v

    # Wrist signal (macro movement indicator)
    wrist_signal = extract_wrist_signal(landmarks_norm, mask)
    wrist_temporal = compute_temporal_features(wrist_signal, fps)
    for k, v in wrist_temporal.items():
        features[f"wrist_{k}"] = v

    # Tremor-to-movement ratio
    tremor_power = features.get("aggregate_band_power", 0)
    macro_power = features.get("wrist_rms_amplitude", 0)
    features["tremor_to_movement_ratio"] = (
        tremor_power / (macro_power + 1e-10) if macro_power > 0 else tremor_power
    )

    return features


def process_dataset(
    dataset_dir: Path,
    output_dir: Optional[Path] = None,
    fps: int = 30,
):
    """Extract features for all videos in a dataset.

    Args:
        dataset_dir: Dataset directory (must contain landmarks/ subfolder).
        output_dir: Where to save features (default: dataset_dir/features/).
        fps: Video frame rate.
    """
    landmarks_dir = dataset_dir / "landmarks"
    if output_dir is None:
        output_dir = dataset_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    landmark_files = sorted(landmarks_dir.glob("*.npz"))
    print(f"Extracting features from {len(landmark_files)} landmark files...")

    all_features = []

    for i, lm_path in enumerate(landmark_files):
        video_id = lm_path.stem

        try:
            features = extract_features_for_video(lm_path, fps)
            features["video_id"] = video_id

            # Save individual feature file
            feat_path = output_dir / f"{video_id}.json"
            with open(feat_path, "w") as f:
                json.dump(features, f, indent=2)

            all_features.append(features)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(landmark_files)}] processed")

        except Exception as e:
            print(f"  ERROR {video_id}: {e}")

    # Save combined features as JSON Lines
    features_path = output_dir / "features.jsonl"
    with open(features_path, "w") as f:
        for feat in all_features:
            f.write(json.dumps(feat) + "\n")

    print(f"\nFeature extraction complete: {len(all_features)} videos")
    print(f"  Output: {features_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract tremor features from landmarks")
    parser.add_argument("--dataset-dir", type=str,
                        default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    process_dataset(Path(args.dataset_dir), 
                    Path(args.output_dir) if args.output_dir else None,
                    args.fps)


if __name__ == "__main__":
    main()
