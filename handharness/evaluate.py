"""
evaluate.py – Comprehensive model evaluation for MotionBloom tremor models.

Evaluates both feature-based and sequence models with:
- MAE / RMSE for severity score regression
- Accuracy / F1 / confusion matrix for classification
- Per-class metrics
- Calibration analysis

Usage:
    python handharness/evaluate.py \
        --features datasets/synth_tremor_smoke/features.csv \
        --signals datasets/synth_tremor_smoke/signals \
        --labels datasets/synth_tremor_smoke/labels.csv \
        --feature-model models/feature_baseline \
        --sequence-model models/sequence_cnn \
        --out evaluation/
"""

import argparse
import json
import sys
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional

try:
    import pandas as pd
    from sklearn.metrics import (
        classification_report, confusion_matrix, mean_absolute_error,
        mean_squared_error, r2_score, accuracy_score, f1_score
    )
except ImportError:
    print("ERROR: sklearn/pandas not installed. Run: pip install pandas scikit-learn")
    sys.exit(1)


SEVERITY_CLASSES = ["none", "mild", "moderate", "severe", "artifact", "gross_motion"]


def evaluate_feature_model(features_path: str, model_dir: str) -> Dict:
    """Evaluate RandomForest feature model."""
    df = pd.read_csv(features_path)

    # Load model
    with open(Path(model_dir) / "classifier.pkl", 'rb') as f:
        clf = pickle.load(f)
    with open(Path(model_dir) / "regressor.pkl", 'rb') as f:
        reg = pickle.load(f)
    with open(Path(model_dir) / "label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)

    # Prepare features
    exclude = {"video_id", "severity_score_gt", "tremor_type_gt",
               "frequency_hz_gt", "amplitude_degrees_gt",
               "detection_rate", "frame_count"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0).values.astype(np.float64)

    # Ground truth
    y_true_class = df["tremor_type_gt"].fillna("unknown").values
    y_true_score = df["severity_score_gt"].fillna(0).values.astype(np.float64)

    # Predictions
    y_pred_class_encoded = clf.predict(X)
    y_pred_class = le.inverse_transform(y_pred_class_encoded)
    y_pred_score = reg.predict(X)

    # Classification metrics
    valid_mask = np.isin(y_true_class, le.classes_)
    if valid_mask.sum() > 0:
        y_true_valid = y_true_class[valid_mask]
        y_pred_valid = y_pred_class[valid_mask]
        accuracy = float(accuracy_score(y_true_valid, y_pred_valid))
        f1_macro = float(f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0))
        f1_weighted = float(f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0))
        report = classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0)
    else:
        accuracy = 0
        f1_macro = 0
        f1_weighted = 0
        report = {}

    # Regression metrics
    mae = float(mean_absolute_error(y_true_score, y_pred_score))
    rmse = float(np.sqrt(mean_squared_error(y_true_score, y_pred_score)))
    r2 = float(r2_score(y_true_score, y_pred_score)) if len(np.unique(y_true_score)) > 1 else 0.0

    # Per-class MAE
    per_class_mae = {}
    for cls in le.classes_:
        mask = y_true_class == cls
        if mask.sum() > 0:
            per_class_mae[cls] = float(mean_absolute_error(y_true_score[mask], y_pred_score[mask]))

    return {
        "model_type": "RandomForest",
        "n_samples": len(df),
        "classification": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class": report,
        },
        "regression": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "per_class_mae": per_class_mae,
        },
    }


def evaluate_sequence_model(signals_dir: str, labels_path: str, model_dir: str) -> Optional[Dict]:
    """Evaluate temporal Conv1D model."""
    try:
        import torch
        from train_sequence_model import TremorConv1D, TremorSequenceDataset
    except ImportError:
        # Try relative import
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            import torch
            from train_sequence_model import TremorConv1D, TremorSequenceDataset
        except ImportError:
            print("WARNING: Cannot evaluate sequence model (torch or module not available)")
            return None

    model_path = Path(model_dir) / "best_model.pt"
    if not model_path.exists():
        model_path = Path(model_dir) / "final_model.pt"
    if not model_path.exists():
        print(f"WARNING: No model found in {model_dir}")
        return None

    # Load metadata
    meta_path = Path(model_dir) / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        max_frames = meta.get("max_frames", 120)
    else:
        max_frames = 120

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = TremorSequenceDataset(signals_dir, labels_path, max_frames=max_frames)
    if len(dataset) == 0:
        return None

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    model = TremorConv1D(input_dim=63, n_classes=len(SEVERITY_CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_true_class = []
    all_pred_class = []
    all_true_score = []
    all_pred_score = []

    with torch.no_grad():
        for x, y_class, y_score in loader:
            x = x.to(device)
            class_logits, score_pred = model(x)
            preds = class_logits.argmax(dim=1).cpu().numpy()
            scores = score_pred.cpu().numpy() * 100

            all_true_class.extend(y_class.numpy())
            all_pred_class.extend(preds)
            all_true_score.extend(y_score.numpy() * 100)
            all_pred_score.extend(scores)

    all_true_class = np.array(all_true_class)
    all_pred_class = np.array(all_pred_class)
    all_true_score = np.array(all_true_score)
    all_pred_score = np.array(all_pred_score)

    # Metrics
    accuracy = float(accuracy_score(all_true_class, all_pred_class))
    mae = float(mean_absolute_error(all_true_score, all_pred_score))
    rmse = float(np.sqrt(mean_squared_error(all_true_score, all_pred_score)))

    return {
        "model_type": "TremorConv1D",
        "n_samples": len(dataset),
        "classification": {
            "accuracy": accuracy,
        },
        "regression": {
            "mae": mae,
            "rmse": rmse,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MotionBloom tremor models")
    parser.add_argument("--features", help="Path to features.csv")
    parser.add_argument("--signals", help="Directory with signal .json files")
    parser.add_argument("--labels", help="Path to labels.csv")
    parser.add_argument("--feature-model", help="Path to feature model directory")
    parser.add_argument("--sequence-model", help="Path to sequence model directory")
    parser.add_argument("--out", required=True, help="Output directory for evaluation results")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Model Evaluation                           ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}

    # Evaluate feature model
    if args.features and args.feature_model:
        feature_model_dir = Path(args.feature_model).resolve()
        if feature_model_dir.exists() and Path(args.features).exists():
            print("Evaluating feature model (RandomForest)...")
            feat_results = evaluate_feature_model(args.features, str(feature_model_dir))
            results["feature_model"] = feat_results
            print(f"  Accuracy:     {feat_results['classification']['accuracy']:.3f}")
            print(f"  F1 (macro):   {feat_results['classification']['f1_macro']:.3f}")
            print(f"  Score MAE:    {feat_results['regression']['mae']:.1f}")
            print(f"  Score RMSE:   {feat_results['regression']['rmse']:.1f}")
            print(f"  Score R²:     {feat_results['regression']['r2']:.3f}")
            print()

    # Evaluate sequence model
    if args.signals and args.labels and args.sequence_model:
        seq_model_dir = Path(args.sequence_model).resolve()
        if seq_model_dir.exists():
            print("Evaluating sequence model (Conv1D)...")
            seq_results = evaluate_sequence_model(args.signals, args.labels, str(seq_model_dir))
            if seq_results:
                results["sequence_model"] = seq_results
                print(f"  Accuracy:     {seq_results['classification']['accuracy']:.3f}")
                print(f"  Score MAE:    {seq_results['regression']['mae']:.1f}")
                print(f"  Score RMSE:   {seq_results['regression']['rmse']:.1f}")
                print()

    # Summary comparison
    if len(results) > 1:
        print("Model Comparison:")
        print(f"  {'Model':<20} {'Accuracy':<12} {'MAE':<10} {'RMSE':<10}")
        print(f"  {'-'*52}")
        for name, res in results.items():
            acc = res.get("classification", {}).get("accuracy", 0)
            mae = res.get("regression", {}).get("mae", 0)
            rmse = res.get("regression", {}).get("rmse", 0)
            print(f"  {name:<20} {acc:<12.3f} {mae:<10.1f} {rmse:<10.1f}")
        print()

    # Save results
    (out_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))

    print(f"{'='*60}")
    print(f"✓ Evaluation complete → {out_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
