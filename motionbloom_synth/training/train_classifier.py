"""Model A: Tremor Validity Classifier.

Classifies each video into one of:
    NO_HAND          — no hand detected (< 50% detection rate)
    TRACKING_UNSTABLE — hand detected but landmarks too noisy
    GROSS_MOVEMENT_ONLY — large hand movement without tremor oscillation
    VALID_TREMOR     — real tremor present (pass to Model B)
    ARTIFACT         — tracking artifact / jitter (not real tremor)

This gate prevents Model B from scoring bad inputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

SYNTH_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = SYNTH_ROOT / "outputs"

# Features relevant for validity classification
CLASSIFIER_FEATURES = [
    # Detection quality
    "detection_rate",
    # Aggregate spectral (distinguishes tremor from noise/movement)
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
    "aggregate_zero_crossing_rate",
    # Wrist movement (separates gross movement from tremor)
    "wrist_mean_amplitude",
    "wrist_rms_amplitude",
    "wrist_max_amplitude",
    # Ratio (key discriminator)
    "tremor_to_movement_ratio",
    # Index finger (representative)
    "index_tip_dominant_frequency_hz",
    "index_tip_band_power_ratio",
    "index_tip_rms_amplitude",
]

# Classes
VALIDITY_CLASSES = [
    "NO_HAND",
    "TRACKING_UNSTABLE",
    "GROSS_MOVEMENT_ONLY",
    "VALID_TREMOR",
    "ARTIFACT",
]


def load_training_data(
    dataset_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features and validity labels for all videos.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Encoded class labels (n_samples,)
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

    X_rows = []
    y_labels = []
    video_ids = []

    for feat in features_list:
        vid = feat["video_id"]
        label = labels.get(vid)
        if label is None:
            continue

        # Determine validity class
        validity = label.get("validity_class", "VALID_TREMOR")

        # Additional heuristic: low detection → NO_HAND
        detection_rate = float(feat.get("detection_rate", 1.0))
        if detection_rate < 0.3:
            validity = "NO_HAND"
        elif detection_rate < 0.6:
            validity = "TRACKING_UNSTABLE"

        # Build feature vector
        row = []
        for fname in CLASSIFIER_FEATURES:
            row.append(float(feat.get(fname, 0.0)))

        X_rows.append(row)
        y_labels.append(validity)
        video_ids.append(vid)

    X = np.array(X_rows, dtype=np.float32)

    # Encode labels
    le = LabelEncoder()
    le.fit(VALIDITY_CLASSES)
    y = le.transform(y_labels)

    return X, y, video_ids


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[GradientBoostingClassifier, StandardScaler, LabelEncoder, Dict]:
    """Train the validity classifier.

    Args:
        X: Feature matrix.
        y: Encoded labels.
        test_size: Test split fraction.
        seed: Random seed.

    Returns:
        model: Trained classifier.
        scaler: Feature scaler.
        label_encoder: For decoding predictions.
        metrics: Evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    le = LabelEncoder()
    le.fit(VALIDITY_CLASSES)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=VALIDITY_CLASSES,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        model, scaler.transform(X), y, cv=5, scoring="accuracy"
    )

    metrics = {
        "accuracy": float(accuracy),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(VALIDITY_CLASSES),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": VALIDITY_CLASSES,
    }

    # Feature importance
    importance = dict(zip(CLASSIFIER_FEATURES, model.feature_importances_))
    metrics["top_features"] = dict(
        sorted(importance.items(), key=lambda x: -x[1])[:10]
    )

    return model, scaler, le, metrics


def save_model(
    model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    metrics: Dict,
    output_dir: Path,
):
    """Save classifier model and artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "validity_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "validity_classifier_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "validity_classifier_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open(output_dir / "validity_classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(output_dir / "validity_classifier_features.json", "w") as f:
        json.dump(CLASSIFIER_FEATURES, f, indent=2)


def predict(
    model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    features: Dict[str, float],
) -> Tuple[str, float]:
    """Predict validity class for a single sample.

    Args:
        model: Trained classifier.
        scaler: Feature scaler.
        label_encoder: Label decoder.
        features: Extracted feature dict for one video.

    Returns:
        predicted_class: One of VALIDITY_CLASSES.
        confidence: Prediction probability.
    """
    row = [float(features.get(fname, 0.0)) for fname in CLASSIFIER_FEATURES]
    X = np.array([row], dtype=np.float32)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    predicted_class = label_encoder.inverse_transform([pred])[0]
    confidence = float(proba[pred])

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Train tremor validity classifier (Model A)"
    )
    parser.add_argument("--dataset-dir", type=str,
                        default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--output-dir", type=str,
                        default=str(OUTPUTS_DIR / "models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    print("Loading training data...")
    X, y, video_ids = load_training_data(dataset_dir)

    le = LabelEncoder()
    le.fit(VALIDITY_CLASSES)

    # Class distribution
    print(f"  Total samples: {len(video_ids)}")
    for i, cls in enumerate(VALIDITY_CLASSES):
        count = (y == i).sum()
        print(f"    {cls}: {count} ({100*count/len(y):.1f}%)")

    print("\nTraining validity classifier...")
    model, scaler, encoder, metrics = train_classifier(X, y, seed=args.seed)

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print(f"\n  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"    {VALIDITY_CLASSES}")
    for i, row in enumerate(cm):
        print(f"    {VALIDITY_CLASSES[i]:25s} {row}")

    save_model(model, scaler, encoder, metrics, output_dir)
    print(f"\nModel saved to: {output_dir}")


if __name__ == "__main__":
    main()
