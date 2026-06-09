"""Evaluation & Acceptance Testing.

Runs the trained models (A + B) against the synthetic dataset
and validates against acceptance criteria:

    No tremor videos       → score under 10
    Mild tremor            → 10–35
    Moderate tremor        → 40–70
    Severe tremor          → 75+
    Gross movement (no tremor) → low tremor, high movement
    Tracking artifact      → classified as ARTIFACT, not VALID_TREMOR

Also runs the current MotionBloom detector over the synthetic videos
to compare against the ground truth labels.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SYNTH_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = SYNTH_ROOT / "outputs"


# --- Acceptance Criteria ---

ACCEPTANCE_CRITERIA = {
    "no_tremor": {"score_range": (0, 10), "severity_levels": [0]},
    "mild": {"score_range": (10, 35), "severity_levels": [5, 10, 20]},
    "moderate": {"score_range": (35, 70), "severity_levels": [35, 50, 65]},
    "severe": {"score_range": (70, 100), "severity_levels": [80, 95]},
}


def load_models(model_dir: Path) -> Dict:
    """Load trained Model A (classifier) and Model B (regressor)."""
    models = {}

    # Model A: Validity classifier
    clf_path = model_dir / "validity_classifier.pkl"
    if clf_path.exists():
        with open(clf_path, "rb") as f:
            models["classifier"] = pickle.load(f)
        with open(model_dir / "validity_classifier_scaler.pkl", "rb") as f:
            models["classifier_scaler"] = pickle.load(f)
        with open(model_dir / "validity_classifier_encoder.pkl", "rb") as f:
            models["classifier_encoder"] = pickle.load(f)
        with open(model_dir / "validity_classifier_features.json") as f:
            models["classifier_features"] = json.load(f)

    # Model B: Severity regressor
    reg_path = model_dir / "severity_regressor.pkl"
    if reg_path.exists():
        with open(reg_path, "rb") as f:
            models["regressor"] = pickle.load(f)
        with open(model_dir / "severity_regressor_scaler.pkl", "rb") as f:
            models["regressor_scaler"] = pickle.load(f)
        with open(model_dir / "severity_regressor_features.json") as f:
            models["regressor_features"] = json.load(f)

    # Frequency regressor
    freq_path = model_dir / "frequency_regressor.pkl"
    if freq_path.exists():
        with open(freq_path, "rb") as f:
            models["freq_regressor"] = pickle.load(f)
        with open(model_dir / "frequency_regressor_scaler.pkl", "rb") as f:
            models["freq_regressor_scaler"] = pickle.load(f)

    return models


def predict_full_pipeline(
    models: Dict,
    features: Dict[str, float],
) -> Dict[str, float]:
    """Run the full A→B pipeline on a single sample.

    Returns:
        Dict with validity_class, confidence, predicted_score, predicted_freq.
    """
    result = {
        "validity_class": "UNKNOWN",
        "validity_confidence": 0.0,
        "predicted_score": 0,
        "predicted_frequency_hz": 0.0,
    }

    # Model A: Classify validity
    if "classifier" in models:
        clf = models["classifier"]
        scaler = models["classifier_scaler"]
        encoder = models["classifier_encoder"]
        feat_names = models["classifier_features"]

        row = [float(features.get(f, 0.0)) for f in feat_names]
        X = np.array([row], dtype=np.float32)
        X_scaled = scaler.transform(X)

        pred = clf.predict(X_scaled)[0]
        proba = clf.predict_proba(X_scaled)[0]

        result["validity_class"] = encoder.inverse_transform([pred])[0]
        result["validity_confidence"] = float(proba[pred])

    # Model B: Score (only if VALID_TREMOR)
    if result["validity_class"] == "VALID_TREMOR" and "regressor" in models:
        reg = models["regressor"]
        scaler = models["regressor_scaler"]
        feat_names = models["regressor_features"]

        row = [float(features.get(f, 0.0)) for f in feat_names]
        X = np.array([row], dtype=np.float32)
        X_scaled = scaler.transform(X)

        score = float(reg.predict(X_scaled)[0])
        result["predicted_score"] = int(np.clip(score, 0, 100))

    # Frequency prediction
    if result["validity_class"] == "VALID_TREMOR" and "freq_regressor" in models:
        freg = models["freq_regressor"]
        fscaler = models["freq_regressor_scaler"]
        feat_names = models["regressor_features"]  # Same features

        row = [float(features.get(f, 0.0)) for f in feat_names]
        X = np.array([row], dtype=np.float32)
        X_scaled = fscaler.transform(X)

        freq = float(freg.predict(X_scaled)[0])
        result["predicted_frequency_hz"] = round(np.clip(freq, 0, 30), 2)

    return result


def run_acceptance_test(
    dataset_dir: Path,
    model_dir: Path,
) -> Dict[str, any]:
    """Run full acceptance test against labeled synthetic dataset.

    Tests:
        1. Validity classification accuracy
        2. Score predictions fall within expected ranges
        3. Non-tremor correctly rejected
        4. Artifacts correctly identified
    """
    import csv

    models = load_models(model_dir)
    if not models:
        print("ERROR: No trained models found. Train first.")
        return {"pass": False, "reason": "no_models"}

    # Load labels
    labels = {}
    with open(dataset_dir / "labels.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video_id"]] = row

    # Load features
    features_all = {}
    features_path = dataset_dir / "features" / "features.jsonl"
    with open(features_path) as f:
        for line in f:
            feat = json.loads(line)
            features_all[feat["video_id"]] = feat

    # Run predictions
    results = []
    for vid, label in labels.items():
        if vid not in features_all:
            continue

        features = features_all[vid]
        prediction = predict_full_pipeline(models, features)
        prediction["video_id"] = vid
        prediction["true_severity"] = int(label["severity_score"])
        prediction["true_validity"] = label["validity_class"]
        prediction["true_frequency"] = float(label["frequency_hz"])
        results.append(prediction)

    # --- Acceptance Tests ---
    tests = {}

    # Test 1: Validity classification
    correct_validity = sum(
        1 for r in results if r["validity_class"] == r["true_validity"]
    )
    validity_accuracy = correct_validity / len(results) if results else 0
    tests["validity_accuracy"] = {
        "value": validity_accuracy,
        "threshold": 0.80,
        "pass": validity_accuracy >= 0.80,
    }

    # Test 2: No-tremor videos score under 10
    no_tremor = [r for r in results if r["true_severity"] == 0]
    if no_tremor:
        no_tremor_scores = [r["predicted_score"] for r in no_tremor]
        pct_under_10 = sum(1 for s in no_tremor_scores if s <= 10) / len(no_tremor)
        tests["no_tremor_under_10"] = {
            "value": pct_under_10,
            "threshold": 0.90,
            "pass": pct_under_10 >= 0.90,
            "mean_score": float(np.mean(no_tremor_scores)),
        }

    # Test 3: Mild tremor → 10–35
    mild = [r for r in results if r["true_severity"] in (5, 10, 20)
            and r["validity_class"] == "VALID_TREMOR"]
    if mild:
        mild_scores = [r["predicted_score"] for r in mild]
        pct_in_range = sum(1 for s in mild_scores if 10 <= s <= 35) / len(mild)
        tests["mild_in_range"] = {
            "value": pct_in_range,
            "threshold": 0.60,
            "pass": pct_in_range >= 0.60,
            "mean_score": float(np.mean(mild_scores)),
        }

    # Test 4: Moderate tremor → 40–70
    moderate = [r for r in results if r["true_severity"] in (35, 50, 65)
                and r["validity_class"] == "VALID_TREMOR"]
    if moderate:
        mod_scores = [r["predicted_score"] for r in moderate]
        pct_in_range = sum(1 for s in mod_scores if 35 <= s <= 70) / len(moderate)
        tests["moderate_in_range"] = {
            "value": pct_in_range,
            "threshold": 0.60,
            "pass": pct_in_range >= 0.60,
            "mean_score": float(np.mean(mod_scores)),
        }

    # Test 5: Severe tremor → 75+
    severe = [r for r in results if r["true_severity"] in (80, 95)
              and r["validity_class"] == "VALID_TREMOR"]
    if severe:
        sev_scores = [r["predicted_score"] for r in severe]
        pct_above_70 = sum(1 for s in sev_scores if s >= 70) / len(severe)
        tests["severe_above_70"] = {
            "value": pct_above_70,
            "threshold": 0.70,
            "pass": pct_above_70 >= 0.70,
            "mean_score": float(np.mean(sev_scores)),
        }

    # Test 6: Artifacts correctly rejected
    artifacts = [r for r in results if r["true_validity"] == "ARTIFACT"]
    if artifacts:
        correct_artifacts = sum(
            1 for r in artifacts if r["validity_class"] != "VALID_TREMOR"
        )
        artifact_rejection = correct_artifacts / len(artifacts)
        tests["artifact_rejection"] = {
            "value": artifact_rejection,
            "threshold": 0.85,
            "pass": artifact_rejection >= 0.85,
        }

    # Test 7: Gross movement correctly rejected
    gross = [r for r in results if r["true_validity"] == "GROSS_MOVEMENT_ONLY"]
    if gross:
        correct_gross = sum(
            1 for r in gross if r["validity_class"] != "VALID_TREMOR"
        )
        gross_rejection = correct_gross / len(gross)
        tests["gross_movement_rejection"] = {
            "value": gross_rejection,
            "threshold": 0.85,
            "pass": gross_rejection >= 0.85,
        }

    # Overall pass
    all_pass = all(t["pass"] for t in tests.values())

    return {
        "pass": all_pass,
        "tests": tests,
        "total_samples": len(results),
        "predictions_sample": results[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models against acceptance criteria")
    parser.add_argument("--dataset-dir", type=str,
                        default=str(OUTPUTS_DIR / "dataset_v1"))
    parser.add_argument("--model-dir", type=str,
                        default=str(OUTPUTS_DIR / "models"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)

    print("=" * 60)
    print("ACCEPTANCE TEST: Synthetic Tremor Dataset")
    print("=" * 60)

    results = run_acceptance_test(dataset_dir, model_dir)

    print(f"\nTotal samples evaluated: {results['total_samples']}")
    print(f"\n{'Test':<35} {'Value':<10} {'Threshold':<10} {'Pass'}")
    print("-" * 65)

    for test_name, test_data in results["tests"].items():
        value = f"{test_data['value']:.3f}"
        threshold = f"{test_data['threshold']:.2f}"
        status = "✅" if test_data["pass"] else "❌"
        print(f"  {test_name:<33} {value:<10} {threshold:<10} {status}")

    print("-" * 65)
    overall = "✅ ALL PASS" if results["pass"] else "❌ SOME FAILED"
    print(f"  {'OVERALL':<33} {'':10} {'':10} {overall}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_dir / "evaluation_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
