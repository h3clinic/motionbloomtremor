#!/usr/bin/env bash
# run_pipeline.sh – Full MotionBloom synthetic training pipeline
#
# Usage:
#   ./handharness/run_pipeline.sh smoke   # Quick test (5 videos × 1s)
#   ./handharness/run_pipeline.sh full    # Full dataset (3000 videos × 4s)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/Users/aharshi/MotionBloomAppVersion/.venv/bin/python"

ASSET="$SCRIPT_DIR/input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb"
RIG_MAP="$SCRIPT_DIR/rig_map.json"
MODEL="$SCRIPT_DIR/hand_landmarker.task"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    COUNT=5; DURATION=1; SEED=123
    DATASET="$PROJECT_DIR/datasets/synth_tremor_smoke"
    ;;
  full)
    COUNT=3000; DURATION=4; SEED=42
    DATASET="$PROJECT_DIR/datasets/synth_tremor_v2"
    ;;
  *)
    echo "Usage: $0 {smoke|full}"
    exit 1
    ;;
esac

FPS=30
FEATURES_DIR="$DATASET/features.csv"
MODEL_DIR="$PROJECT_DIR/models"

echo "════════════════════════════════════════════════════════════"
echo "  MotionBloom Pipeline: $MODE mode"
echo "  Count: $COUNT | FPS: $FPS | Duration: ${DURATION}s | Seed: $SEED"
echo "════════════════════════════════════════════════════════════"
echo

# Phase 1: Render videos
echo "▶ Phase 1: Rendering videos..."
"$PYTHON" "$SCRIPT_DIR/render_tremor_dataset.py" \
  --asset "$ASSET" \
  --rig-map "$RIG_MAP" \
  --count "$COUNT" --fps "$FPS" --duration "$DURATION" \
  --out "$DATASET" --seed "$SEED"
echo

# Phase 2: Extract landmarks
echo "▶ Phase 2: Extracting landmarks..."
"$PYTHON" "$SCRIPT_DIR/extract_landmarks.py" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --out "$DATASET/landmarks"
echo

# Phase 3: Normalize signals
echo "▶ Phase 3: Normalizing signals..."
"$PYTHON" "$SCRIPT_DIR/normalize_signals.py" \
  --landmarks "$DATASET/landmarks" \
  --out "$DATASET/signals" \
  --min-detection 0.1
echo

# Phase 4: Extract features
echo "▶ Phase 4: Extracting features..."
"$PYTHON" "$SCRIPT_DIR/extract_features.py" \
  --signals "$DATASET/signals" \
  --labels "$DATASET/labels.csv" \
  --out "$FEATURES_DIR"
echo

# Phase 5: Train feature model
echo "▶ Phase 5: Training feature model..."
mkdir -p "$MODEL_DIR/feature_baseline"
"$PYTHON" "$SCRIPT_DIR/train_feature_model.py" \
  --features "$FEATURES_DIR" \
  --out "$MODEL_DIR/feature_baseline" \
  --n-estimators 100 || echo "  ⚠ Feature model training failed (may need more samples)"
echo

# Phase 6: Train sequence model (requires PyTorch)
echo "▶ Phase 6: Training sequence model..."
mkdir -p "$MODEL_DIR/sequence_cnn"
"$PYTHON" "$SCRIPT_DIR/train_sequence_model.py" \
  --signals "$DATASET/signals" \
  --labels "$DATASET/labels.csv" \
  --out "$MODEL_DIR/sequence_cnn" \
  --epochs 50 --batch-size 16 || echo "  ⚠ Sequence model training failed (may need PyTorch or more samples)"
echo

# Phase 7: Evaluate
echo "▶ Phase 7: Evaluating models..."
mkdir -p "$PROJECT_DIR/evaluation"
"$PYTHON" "$SCRIPT_DIR/evaluate.py" \
  --features "$FEATURES_DIR" \
  --signals "$DATASET/signals" \
  --labels "$DATASET/labels.csv" \
  --feature-model "$MODEL_DIR/feature_baseline" \
  --sequence-model "$MODEL_DIR/sequence_cnn" \
  --out "$PROJECT_DIR/evaluation" || echo "  ⚠ Evaluation failed"
echo

echo "════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Dataset:     $DATASET"
echo "  Models:      $MODEL_DIR"
echo "  Evaluation:  $PROJECT_DIR/evaluation"
echo "════════════════════════════════════════════════════════════"
