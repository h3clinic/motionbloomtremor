"""
train_feature_model.py – Train RandomForest baseline models on extracted features.

Trains two models:
1. Classifier: severity class (none/mild/moderate/severe/artifact/gross_motion)
2. Regressor: severity score (0-100)

Usage:
    python handharness/train_feature_model.py \
        --features datasets/synth_tremor_smoke/features.csv \
        --out models/feature_baseline
"""

import argparse
import json
import sys
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix, mean_absolute_error,
        mean_squared_error, r2_score, accuracy_score
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: Required packages not installed.")
    print("  pip install pandas scikit-learn")
    sys.exit(1)


# Feature columns used for training (exclude metadata columns)
EXCLUDE_COLS = {
    "video_id", "severity_score_gt", "tremor_type_gt",
    "frequency_hz_gt", "amplitude_degrees_gt",
    "detection_rate", "frame_count",
}

# Severity class mapping
SEVERITY_CLASSES = ["none", "mild", "moderate", "severe", "artifact", "gross_motion"]


def load_features(features_path: str) -> pd.DataFrame:
    """Load features CSV and validate."""
    df = pd.read_csv(features_path)

    # Check required columns
    if "severity_score_gt" not in df.columns:
        print("WARNING: No ground truth severity_score_gt column. Model will not train properly.")
    if "tremor_type_gt" not in df.columns:
        print("WARNING: No ground truth tremor_type_gt column.")

    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Prepare feature matrix and labels.

    Returns: (X, y_class, y_score, feature_names)
    """
    # Feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].fillna(0).values.astype(np.float64)
    feature_names = feature_cols

    # Class labels
    if "tremor_type_gt" in df.columns:
        y_class = df["tremor_type_gt"].fillna("unknown").values
    else:
        y_class = np.array(["unknown"] * len(df))

    # Score labels
    if "severity_score_gt" in df.columns:
        y_score = df["severity_score_gt"].fillna(0).values.astype(np.float64)
    else:
        y_score = np.zeros(len(df))

    return X, y_class, y_score, feature_names


def train_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100) -> Tuple[RandomForestClassifier, Dict]:
    """Train severity class classifier with cross-validation."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation (if enough samples)
    n_classes = len(np.unique(y_encoded))
    n_samples = len(y_encoded)

    if n_samples >= 10 and n_classes >= 2:
        n_splits = min(5, n_samples // n_classes)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring='accuracy')
            cv_result = {"mean": float(cv_scores.mean()), "std": float(cv_scores.std()), "scores": cv_scores.tolist()}
        else:
            cv_result = {"mean": 0, "std": 0, "scores": []}
    else:
        cv_result = {"mean": 0, "std": 0, "scores": []}

    # Train on full data
    clf.fit(X, y_encoded)
    y_pred = clf.predict(X)

    # Metrics (on training data)
    metrics = {
        "accuracy": float(accuracy_score(y_encoded, y_pred)),
        "cv_accuracy": cv_result,
        "classes": le.classes_.tolist(),
        "classification_report": classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True),
    }

    return clf, metrics, le


def train_regressor(X: np.ndarray, y: np.ndarray, n_estimators: int = 100) -> Tuple[RandomForestRegressor, Dict]:
    """Train severity score regressor."""
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    n_samples = len(y)
    if n_samples >= 10:
        n_splits = min(5, n_samples // 2)
        if n_splits >= 2:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(reg, X, y, cv=cv, scoring='neg_mean_absolute_error')
            cv_result = {"mean_mae": float(-cv_scores.mean()), "std": float(cv_scores.std())}
        else:
            cv_result = {"mean_mae": 0, "std": 0}
    else:
        cv_result = {"mean_mae": 0, "std": 0}

    # Train on full data
    reg.fit(X, y)
    y_pred = reg.predict(X)

    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)) if len(np.unique(y)) > 1 else 0.0,
        "cv_mae": cv_result,
    }

    return reg, metrics


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest tremor severity models")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--out", required=True, help="Output directory for models")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    args = parser.parse_args()

    features_path = Path(args.features).resolve()
    out_dir = Path(args.out).resolve()

    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Feature Model Training                     ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Features:     {features_path}")
    print(f"  Output:       {out_dir}")
    print(f"  Estimators:   {args.n_estimators}")
    print()

    # Load data
    df = load_features(str(features_path))
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    X, y_class, y_score, feature_names = prepare_data(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Classes: {np.unique(y_class)}")
    print(f"  Score range: [{y_score.min():.0f}, {y_score.max():.0f}]")
    print()

    # Train classifier
    print("Training severity classifier...")
    clf, clf_metrics, label_encoder = train_classifier(X, y_class, args.n_estimators)
    print(f"  Training accuracy: {clf_metrics['accuracy']:.3f}")
    print(f"  CV accuracy: {clf_metrics['cv_accuracy']['mean']:.3f} ± {clf_metrics['cv_accuracy']['std']:.3f}")

    # Feature importances
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top features:")
    for fname, imp in top_features:
        print(f"    {fname:30s} {imp:.4f}")
    print()

    # Train regressor
    print("Training severity score regressor...")
    reg, reg_metrics = train_regressor(X, y_score, args.n_estimators)
    print(f"  Training MAE: {reg_metrics['mae']:.2f}")
    print(f"  Training RMSE: {reg_metrics['rmse']:.2f}")
    print(f"  Training R²: {reg_metrics['r2']:.3f}")
    print(f"  CV MAE: {reg_metrics['cv_mae']['mean_mae']:.2f}")
    print()

    # Save models
    with open(out_dir / "classifier.pkl", 'wb') as f:
        pickle.dump(clf, f)
    with open(out_dir / "regressor.pkl", 'wb') as f:
        pickle.dump(reg, f)
    with open(out_dir / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save metadata
    model_meta = {
        "model_type": "RandomForest",
        "n_estimators": args.n_estimators,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_samples": len(df),
        "classes": label_encoder.classes_.tolist(),
        "classifier_metrics": clf_metrics,
        "regressor_metrics": reg_metrics,
        "top_features": [{"name": n, "importance": float(v)} for n, v in top_features],
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(model_meta, indent=2))

    print(f"{'='*60}")
    print(f"✓ Models saved to {out_dir}")
    print(f"  classifier.pkl      ({os.path.getsize(out_dir / 'classifier.pkl') / 1024:.0f} KB)")
    print(f"  regressor.pkl       ({os.path.getsize(out_dir / 'regressor.pkl') / 1024:.0f} KB)")
    print(f"  label_encoder.pkl")
    print(f"  model_metadata.json")


if __name__ == "__main__":
    main()
