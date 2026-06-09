"""
extract_features.py – Compute tremor features from normalized signals.

Takes signal files (from normalize_signals.py) and computes per-video features:
- Dominant frequency (FFT peak in 3-12 Hz band)
- Band power (4-8 Hz, 8-12 Hz, 12+ Hz)
- RMS displacement / velocity / acceleration
- Jerk (smoothness metric)
- Macro path length (slow drift)
- Synchronization index (inter-finger correlation)
- Amplitude stability (coefficient of variation)
- Wrist-to-fingertip ratio

Usage:
    python handharness/extract_features.py \
        --signals datasets/synth_tremor_smoke/signals \
        --labels datasets/synth_tremor_smoke/labels.csv \
        --out datasets/synth_tremor_smoke/features.csv
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def dominant_frequency(signal: np.ndarray, fps: float, band: tuple = (3, 12)) -> float:
    """Find dominant frequency in specified band using FFT."""
    if len(signal) < 8:
        return 0.0

    # Detrend
    signal = signal - np.mean(signal)

    # FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Band filter
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not band_mask.any():
        return 0.0

    band_freqs = freqs[band_mask]
    band_mags = fft_mag[band_mask]

    if band_mags.max() < 1e-10:
        return 0.0

    return float(band_freqs[np.argmax(band_mags)])


def band_power(signal: np.ndarray, fps: float, low: float, high: float) -> float:
    """Compute signal power in frequency band."""
    if len(signal) < 8:
        return 0.0

    signal = signal - np.mean(signal)
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(signal))

    band_mask = (freqs >= low) & (freqs <= high)
    if not band_mask.any():
        return 0.0

    # Power = sum of squared magnitudes in band
    return float(np.sum(fft_mag[band_mask] ** 2) / n)


def rms(signal: np.ndarray) -> float:
    """Root mean square."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def mean_jerk(jerk_signal: np.ndarray) -> float:
    """Mean absolute jerk (smoothness metric)."""
    if len(jerk_signal) == 0:
        return 0.0
    return float(np.mean(np.abs(jerk_signal)))


def macro_path_length(displacement_signal: np.ndarray) -> float:
    """Cumulative path length (total displacement over time)."""
    if len(displacement_signal) == 0:
        return 0.0
    return float(np.sum(np.abs(displacement_signal)))


def amplitude_stability(signal: np.ndarray) -> float:
    """Coefficient of variation of envelope (amplitude consistency).

    High CV = inconsistent tremor, Low CV = regular tremor.
    """
    if len(signal) < 10:
        return 0.0

    # Compute envelope using absolute value + smoothing
    envelope = np.abs(signal)
    # Simple moving average (window = ~0.5s worth of samples)
    window = max(3, len(envelope) // 8)
    kernel = np.ones(window) / window
    smoothed = np.convolve(envelope, kernel, mode='valid')

    if len(smoothed) < 3:
        return 0.0

    mean_env = np.mean(smoothed)
    if mean_env < 1e-10:
        return 0.0

    return float(np.std(smoothed) / mean_env)


def synchronization_index(landmark_signals: np.ndarray, indices: List[int]) -> float:
    """Mean pairwise correlation between selected landmark signals.

    High sync = all fingers moving together (typical tremor).
    Low sync = independent finger movements.
    """
    if landmark_signals.shape[0] < 10:
        return 0.0

    selected = landmark_signals[:, indices]  # (T, N)
    n = selected.shape[1]

    if n < 2:
        return 0.0

    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            sig_i = selected[:, i] - np.mean(selected[:, i])
            sig_j = selected[:, j] - np.mean(selected[:, j])
            denom = np.sqrt(np.sum(sig_i**2) * np.sum(sig_j**2))
            if denom > 1e-10:
                corr = np.sum(sig_i * sig_j) / denom
                correlations.append(abs(corr))

    return float(np.mean(correlations)) if correlations else 0.0


def wrist_to_fingertip_ratio(wrist_rms: float, fingertip_rms: float) -> float:
    """Ratio of wrist motion to fingertip motion."""
    if fingertip_rms < 1e-10:
        return 0.0
    return float(wrist_rms / fingertip_rms)


def extract_video_features(signal_data: Dict) -> Optional[Dict]:
    """Extract all features from a single video's signal data."""
    video_id = signal_data.get("video_id", "unknown")
    fps = signal_data.get("fps", 30)
    agg = signal_data.get("aggregate_signals", {})

    if not agg:
        return None

    # Get aggregate signals
    ft_disp = np.array(agg.get("fingertip_displacement", []))
    ft_vel = np.array(agg.get("fingertip_velocity", []))
    ft_accel = np.array(agg.get("fingertip_acceleration", []))
    w_disp = np.array(agg.get("wrist_displacement", []))
    w_vel = np.array(agg.get("wrist_velocity", []))

    # Displacement magnitude per landmark (T-1, 21)
    disp_mag = np.array(signal_data.get("displacement_magnitude", []))
    vel_mag = np.array(signal_data.get("velocity_magnitude", []))
    jerk_mag_arr = np.array(signal_data.get("jerk_magnitude", []))

    if len(ft_disp) < 8:
        return None

    # Fingertip indices
    fingertip_idx = [4, 8, 12, 16, 20]

    features = {
        "video_id": video_id,

        # Frequency features
        "dom_freq_fingertip": dominant_frequency(ft_disp, fps, band=(3, 12)),
        "dom_freq_wrist": dominant_frequency(w_disp, fps, band=(3, 12)),
        "dom_freq_vel": dominant_frequency(ft_vel, fps, band=(3, 12)),

        # Band power
        "power_4_8hz": band_power(ft_disp, fps, 4, 8),
        "power_8_12hz": band_power(ft_disp, fps, 8, 12),
        "power_12_25hz": band_power(ft_disp, fps, 12, 25),
        "power_low_0_3hz": band_power(ft_disp, fps, 0, 3),

        # RMS features
        "rms_fingertip_disp": rms(ft_disp),
        "rms_fingertip_vel": rms(ft_vel),
        "rms_fingertip_accel": rms(ft_accel),
        "rms_wrist_disp": rms(w_disp),
        "rms_wrist_vel": rms(w_vel),

        # Jerk (smoothness)
        "mean_jerk": mean_jerk(jerk_mag_arr.mean(axis=1)) if len(jerk_mag_arr) > 0 else 0.0,

        # Path length
        "macro_path_fingertip": macro_path_length(ft_disp),
        "macro_path_wrist": macro_path_length(w_disp),

        # Amplitude stability
        "amp_stability_fingertip": amplitude_stability(ft_disp),
        "amp_stability_wrist": amplitude_stability(w_disp),

        # Synchronization
        "sync_index_fingertips": synchronization_index(disp_mag, fingertip_idx) if len(disp_mag) > 10 else 0.0,

        # Ratio
        "wrist_fingertip_ratio": wrist_to_fingertip_ratio(rms(w_disp), rms(ft_disp)),

        # Signal stats
        "max_fingertip_disp": float(ft_disp.max()) if len(ft_disp) > 0 else 0.0,
        "max_wrist_disp": float(w_disp.max()) if len(w_disp) > 0 else 0.0,
        "std_fingertip_disp": float(ft_disp.std()) if len(ft_disp) > 0 else 0.0,
        "std_wrist_disp": float(w_disp.std()) if len(w_disp) > 0 else 0.0,

        # Meta
        "detection_rate": signal_data.get("detection_rate", 0.0),
        "frame_count": signal_data.get("total_frames", 0),
    }

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract tremor features from normalized signals")
    parser.add_argument("--signals", required=True, help="Directory with signal .json files")
    parser.add_argument("--labels", help="Path to labels.csv (to merge ground truth)")
    parser.add_argument("--out", required=True, help="Output features CSV path")
    args = parser.parse_args()

    signals_dir = Path(args.signals).resolve()
    out_path = Path(args.out).resolve()

    if not signals_dir.exists():
        print(f"ERROR: Signals directory not found: {signals_dir}")
        sys.exit(1)

    signal_files = sorted(signals_dir.glob("MB_SYNTH_*.json"))
    if not signal_files:
        print(f"ERROR: No signal files found in {signals_dir}")
        sys.exit(1)

    # Load labels if provided
    labels_map = {}
    if args.labels:
        labels_path = Path(args.labels).resolve()
        if labels_path.exists():
            import csv
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels_map[row["video_id"]] = row

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Feature Extraction                         ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Signals:   {signals_dir}")
    print(f"  Labels:    {args.labels or 'none'}")
    print(f"  Output:    {out_path}")
    print(f"  Files:     {len(signal_files)}")
    print()

    all_features = []
    errors = 0

    for i, sig_file in enumerate(signal_files):
        video_id = sig_file.stem
        data = json.loads(sig_file.read_text())

        features = extract_video_features(data)
        if features is None:
            errors += 1
            continue

        # Merge ground truth labels
        if video_id in labels_map:
            lbl = labels_map[video_id]
            features["severity_score_gt"] = int(lbl.get("severity_score_1_100", 0))
            features["tremor_type_gt"] = lbl.get("tremor_type", "unknown")
            features["frequency_hz_gt"] = float(lbl.get("frequency_hz", 0))
            features["amplitude_degrees_gt"] = float(lbl.get("amplitude_degrees", 0))

        all_features.append(features)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(signal_files)}] processed")

    if not all_features:
        print("ERROR: No features extracted")
        sys.exit(1)

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = list(all_features[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    # Also write JSON for easier programmatic access
    json_path = out_path.with_suffix('.json')
    json_path.write_text(json.dumps(all_features, indent=2))

    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete")
    print(f"  Videos processed:  {len(all_features)}")
    print(f"  Errors:            {errors}")
    print(f"  Features per video: {len(fieldnames)}")
    print(f"  Output CSV:        {out_path}")
    print(f"  Output JSON:       {json_path}")

    # Print feature statistics
    print(f"\n  Feature summary (first 5 videos):")
    for feat in all_features[:5]:
        print(f"    {feat['video_id']}: dom_freq={feat['dom_freq_fingertip']:.1f}Hz, "
              f"rms_disp={feat['rms_fingertip_disp']:.4f}, "
              f"sync={feat['sync_index_fingertips']:.3f}")


if __name__ == "__main__":
    main()
