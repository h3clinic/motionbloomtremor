"""
train_sequence_model.py – Train temporal Conv1D model on landmark sequences.

Takes normalized landmark sequences (from normalize_signals.py) and trains
a 1D convolutional neural network for severity classification and regression.

Architecture:
    Input: (T, 21*3=63) normalized landmark positions per frame
    → Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)
    → Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool(2)
    → Conv1D(128, k=3) → BatchNorm → ReLU → GlobalAvgPool
    → Dense(64) → Dropout(0.3) → Dense(output)

Usage:
    python handharness/train_sequence_model.py \
        --signals datasets/synth_tremor_smoke/signals \
        --labels datasets/synth_tremor_smoke/labels.csv \
        --out models/sequence_cnn \
        --epochs 50 --batch-size 16
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_CLASSES = ["none", "mild", "moderate", "severe", "artifact", "gross_motion"]


class TremorSequenceDataset(Dataset):
    """Dataset of normalized landmark sequences with labels."""

    def __init__(self, signals_dir: str, labels_path: str, max_frames: int = 120):
        self.max_frames = max_frames
        self.samples = []
        self.labels_class = []
        self.labels_score = []

        # Load labels
        import csv
        labels_map = {}
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels_map[row["video_id"]] = row

        # Load signal files
        signals_path = Path(signals_dir)
        for sig_file in sorted(signals_path.glob("MB_SYNTH_*.json")):
            video_id = sig_file.stem
            if video_id not in labels_map:
                continue

            data = json.loads(sig_file.read_text())
            landmarks = data.get("landmarks_normalized")
            if landmarks is None or len(landmarks) < 4:
                continue

            # Convert to numpy array (T, 21, 3)
            seq = np.array(landmarks, dtype=np.float32)
            T = seq.shape[0]

            # Flatten to (T, 63)
            seq_flat = seq.reshape(T, -1)

            # Pad or truncate to max_frames
            if T > max_frames:
                seq_flat = seq_flat[:max_frames]
            elif T < max_frames:
                pad = np.zeros((max_frames - T, 63), dtype=np.float32)
                seq_flat = np.vstack([seq_flat, pad])

            self.samples.append(seq_flat)

            # Labels
            lbl = labels_map[video_id]
            tremor_type = lbl.get("tremor_type", "unknown")
            severity_score = int(lbl.get("severity_score_1_100", 0))

            class_idx = SEVERITY_CLASSES.index(tremor_type) if tremor_type in SEVERITY_CLASSES else 0
            self.labels_class.append(class_idx)
            self.labels_score.append(severity_score / 100.0)  # Normalize to [0, 1]

        self.samples = np.array(self.samples)  # (N, max_frames, 63)
        self.labels_class = np.array(self.labels_class)
        self.labels_score = np.array(self.labels_score, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (T, 63)
        y_class = torch.tensor(self.labels_class[idx], dtype=torch.long)
        y_score = torch.tensor(self.labels_score[idx], dtype=torch.float32)
        return x, y_class, y_score


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TremorConv1D(nn.Module):
    """1D CNN for temporal landmark sequences → severity prediction."""

    def __init__(self, input_dim: int = 63, n_classes: int = 6):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.shared_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Linear(64, n_classes)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x):
        """
        x: (batch, T, 63) → permute to (batch, 63, T) for Conv1d
        """
        x = x.permute(0, 2, 1)  # (B, 63, T)

        x = self.conv_block1(x)   # (B, 64, T/2)
        x = self.conv_block2(x)   # (B, 128, T/4)
        x = self.conv_block3(x)   # (B, 128, T/4)
        x = self.global_pool(x)   # (B, 128, 1)
        x = x.squeeze(-1)         # (B, 128)

        shared = self.shared_fc(x)  # (B, 64)

        class_logits = self.classifier(shared)  # (B, n_classes)
        score_pred = self.regressor(shared).squeeze(-1)  # (B,)

        return class_logits, score_pred


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, class_criterion, reg_criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y_class, y_score in dataloader:
        x, y_class, y_score = x.to(device), y_class.to(device), y_score.to(device)

        optimizer.zero_grad()
        class_logits, score_pred = model(x)

        loss_class = class_criterion(class_logits, y_class)
        loss_reg = reg_criterion(score_pred, y_score)
        loss = loss_class + 0.5 * loss_reg  # Combined loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = class_logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, class_criterion, reg_criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds_class = []
    all_preds_score = []
    all_true_class = []
    all_true_score = []

    with torch.no_grad():
        for x, y_class, y_score in dataloader:
            x, y_class, y_score = x.to(device), y_class.to(device), y_score.to(device)

            class_logits, score_pred = model(x)

            loss_class = class_criterion(class_logits, y_class)
            loss_reg = reg_criterion(score_pred, y_score)
            loss = loss_class + 0.5 * loss_reg

            total_loss += loss.item() * x.size(0)
            preds = class_logits.argmax(dim=1)
            correct += (preds == y_class).sum().item()
            total += x.size(0)

            all_preds_class.extend(preds.cpu().numpy())
            all_preds_score.extend(score_pred.cpu().numpy())
            all_true_class.extend(y_class.cpu().numpy())
            all_true_score.extend(y_score.cpu().numpy())

    metrics = {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "preds_class": all_preds_class,
        "preds_score": all_preds_score,
        "true_class": all_true_class,
        "true_score": all_true_score,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train temporal Conv1D model for tremor severity")
    parser.add_argument("--signals", required=True, help="Directory with signal .json files")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--out", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-frames", type=int, default=120, help="Max sequence length")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    signals_dir = Path(args.signals).resolve()
    labels_path = Path(args.labels).resolve()
    out_dir = Path(args.out).resolve()

    if not signals_dir.exists():
        print(f"ERROR: Signals directory not found: {signals_dir}")
        sys.exit(1)
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  MotionBloom Temporal CNN Training                      ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"  Signals:     {signals_dir}")
    print(f"  Labels:      {labels_path}")
    print(f"  Output:      {out_dir}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Max frames:  {args.max_frames}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TremorSequenceDataset(str(signals_dir), str(labels_path), max_frames=args.max_frames)
    print(f"  Loaded {len(dataset)} samples")

    if len(dataset) < 4:
        print("ERROR: Not enough samples to train. Need at least 4.")
        sys.exit(1)

    # Class distribution
    from collections import Counter
    class_counts = Counter(dataset.labels_class.tolist())
    print(f"  Class distribution: {dict(class_counts)}")
    print()

    # Train/Val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    n_classes = len(SEVERITY_CLASSES)
    model = TremorConv1D(input_dim=63, n_classes=n_classes).to(device)

    # Class weights for imbalanced data
    class_weights = np.ones(n_classes)
    for cls_idx, count in class_counts.items():
        if count > 0:
            class_weights[cls_idx] = len(dataset) / (n_classes * count)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    class_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            class_criterion, reg_criterion, device)
        val_metrics = evaluate(model, val_loader, class_criterion, reg_criterion, device)

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Final evaluation
    print("\nFinal evaluation on validation set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    final_metrics = evaluate(model, val_loader, class_criterion, reg_criterion, device)

    # Score metrics
    true_scores = np.array(final_metrics["true_score"]) * 100
    pred_scores = np.array(final_metrics["preds_score"]) * 100
    mae = float(np.mean(np.abs(true_scores - pred_scores)))

    print(f"  Val Accuracy:  {final_metrics['accuracy']:.3f}")
    print(f"  Val Score MAE: {mae:.1f}")

    # Save model + metadata
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    model_meta = {
        "architecture": "TremorConv1D",
        "input_dim": 63,
        "max_frames": args.max_frames,
        "n_classes": n_classes,
        "classes": SEVERITY_CLASSES,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "n_train_samples": train_size,
        "n_val_samples": val_size,
        "best_val_loss": float(best_val_loss),
        "final_val_accuracy": float(final_metrics["accuracy"]),
        "final_val_score_mae": mae,
        "history": history,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(model_meta, indent=2))

    print(f"\n{'='*60}")
    print(f"✓ Model saved to {out_dir}")
    print(f"  best_model.pt          (best validation)")
    print(f"  final_model.pt         (last epoch)")
    print(f"  model_metadata.json    (config + metrics)")


if __name__ == "__main__":
    main()
