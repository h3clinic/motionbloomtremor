"""Model B: Tremor Severity Regressor.

Trains a model to predict severity score (1–100) from extracted features.
Only runs on samples classified as VALID_TREMOR by Model A.

Predicts:
    - score: 1–100 severity
    - dominant_frequency_hz
    - confidence

Architecture options:
    1. Gradient Boosted Trees (XGBoost/LightGBM) — fast, interpretable
    2. Small MLP — if more capacity is needed
    3. 1D CNN on landmark time series — for sequence modeling

Starts with option 1 (most data-efficient for 500-video starter).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

SYNTH_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = SYNTH_ROOT / "outputs"

# Features to use for regression (subset of all extracted features)
REGRESSOR_FEATURES = [
    # Aggregate spectral
    "aggregate_dominant_frequency_hz",
    "aggregate_band_power",
    "aggregate_band_power_ratio",
    "aggregate_spectral_entropy",
    "aggregate_total_power",
    # Aggregate temporal
    "aggregate_mean_amplitude",
    "aggregate_max_amplitude",
    "aggregate_rms_amplitude",
    "aggregate_std_amplitude",
    "aggregate_duration_above_threshold",
    "aggregate_zero_crossing_rate",
    # Per-finger (index as representative)
    "index_tip_dominant_frequency_hz",
    "index_tip_band_power_ratio",
    "index_tip_rms_amplitude",
    # Thumb
    "thumb_tip_dominant_frequency_hz",
    "thumb_tip_band_power_ratio",
    "thumb_tip_rms_amplitude",
    # Wrist (macro movement)
    "wrist_mean_amplitude",
    "wrist_rms_amplitude",
    # Ratio
    "tremor_to_movement_ratio",
    # Quality
    "detection_rate",
]


def load_training_data(
    dataset_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load features and labels, filtering to VALID_TREMOR only.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y_score: Severity scores (n_samples,)
        y_freq: Dominant frequencies (n_samples,)
        video_ids: List of video IDs
    """
    features_path = dataset_dir / "features" / "features.jsonl"
    labels_path = dataset_dir / "labels.csv"

    # Load labels
    import csv
    labels = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    # Load features
    features_list = []
    with open(features_path) as f:
        for line in f:
            features_list.append(json.loads(line))

    # Filter to VALID_TREMOR only (Model B prerequisite)
    X_rows = []
    y_scores = []
    y_freqs = []
    video_ids = []

    for feat in features_list:
        vid = feat["video_id"]
        label = labels.get(vid)
        if label is None:
            continue
        if label["validity_class"] != "VALID_TREMOR":
            continue

        # Build feature vector
        row = []
        for fname in REGRESSOR_FEATURES:
            row.append(float(feat.get(fname, 0.0)))

        X_rows.append(row)
        y_scores.append(int(label["severity_score"]))
        y_freqs.append(float(label["frequency_hz"]))
        video_ids.append(vid)

    X = np.array(X_rows, dtype=np.float32)
    y_score = np.array(y_scores, dtype=np.float32)
    y_freq = np.array(y_freqs, dtype=np.float32)

    return X, y_score, y_freq, video_ids


def train_severity_regressor(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[GradientBoostingRegressor, StandardScaler, Dict[str, float]]:
    """Train gradient boosted regressor for severity score.

    Args:
        X: Feature matrix.
        y: Target severity scores.
        test_size: Fraction held out for test.
        seed: Random seed.

    Returns:
        model: Trained regressor.
        scaler: Fitted feature scaler.
        metrics: Dict of evaluation metrics.
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    # Clamp predictions to [0, 100]
    y_pred = np.clip(y_pred, 0, 100)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
    }

    # Cross-validation MAE
    cv_scores = cross_val_score(
        model, scaler.transform(X), y,
        cv=5, scoring="neg_mean_absolute_error"
    )
    metrics["cv_mae_mean"] = float(-cv_scores.mean())
    metrics["cv_mae_std"] = float(cv_scores.std())

    # Feature importance
    importance = dict(zip(REGRESSOR_FEATURES, model.feature_importances_))
    metrics["top_features"] = dict(
        sorted(importance.items(), key=lambda x: -x[1])[:10]
    )

    return model, scaler, metrics


def train_frequency_regressor(
    X: np.ndarray,
    y_freq: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[GradientBoostingRegressor, StandardScaler, Dict[str, float]]:
    """Train regressor for dominant tremor frequency.

    Same architecture as severity, different target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_freq, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 30)  # Clamp to reasonable Hz range

    metrics = {
        "mae_hz": float(mean_absolute_error(y_test, y_pred)),
        "rmse_hz": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return model, scaler, metrics


def save_model(
    model,
    scaler: StandardScaler,
    metrics: Dict,
    output_dir: Path,
    name: str = "severity_regressor",
):
    """Save trained model, scaler, and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    with open(output_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Scaler
    with open(output_dir / f"{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Metrics
    with open(output_dir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Feature names
    with open(output_dir / f"{name}_features.json", "w") as f:
        json.dump(REGRESSOR_FEATURES, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train tremor severity regressor (Model B)")
    parser.add_argument("--dataset-dir", type=str,
                        default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUTS_DIR / "models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    print("Loading training data...")
    X, y_score, y_freq, video_ids = load_training_data(dataset_dir)
    print(f"  Samples (VALID_TREMOR): {len(video_ids)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Score range: [{y_score.min():.0f}, {y_score.max():.0f}]")
    print(f"  Frequency range: [{y_freq.min():.1f}, {y_freq.max():.1f}] Hz")

    # Train severity regressor
    print("\nTraining severity regressor...")
    model_sev, scaler_sev, metrics_sev = train_severity_regressor(
        X, y_score, seed=args.seed
    )
    print(f"  MAE: {metrics_sev['mae']:.2f}")
    print(f"  RMSE: {metrics_sev['rmse']:.2f}")
    print(f"  R²: {metrics_sev['r2']:.4f}")
    print(f"  CV MAE: {metrics_sev['cv_mae_mean']:.2f} ± {metrics_sev['cv_mae_std']:.2f}")

    save_model(model_sev, scaler_sev, metrics_sev, output_dir, "severity_regressor")

    # Train frequency regressor
    print("\nTraining frequency regressor...")
    model_freq, scaler_freq, metrics_freq = train_frequency_regressor(
        X, y_freq, seed=args.seed
    )
    print(f"  MAE: {metrics_freq['mae_hz']:.2f} Hz")
    print(f"  R²: {metrics_freq['r2']:.4f}")

    save_model(model_freq, scaler_freq, metrics_freq, output_dir, "frequency_regressor")

    print(f"\nModels saved to: {output_dir}")


if __name__ == "__main__":
    main()
