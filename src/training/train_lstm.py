import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, Tuple
# Ensure project root is always in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.dataset.audio_dataset_mfcc import AudioMFCCDataset
from src.models.lstm import AudioLSTMClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
def compute_class_weights(
    manifest_path: str = "data/processed/manifest.csv",
) -> Tuple[float, float]:
    """Computes balanced sample weights for REAL (0) and FAKE (1) classes
    based on the training manifest distribution.
    """
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    total = len(train_df)
    n_real = int((train_df["label"] == 0).sum())
    n_fake = int((train_df["label"] == 1).sum())
    w_real = total / (2.0 * n_real)
    w_fake = total / (2.0 * n_fake)
    print(
        f"Class distribution -> REAL (0): {n_real:,} | FAKE (1): {n_fake:,}"
    )
    print(
        f"Computed class weights -> w_real (0): {w_real:.4f} | w_fake (1): {w_fake:.4f}"
    )
    return float(w_real), float(w_fake)
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    w_real: float,
    w_fake: float,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the model on the validation dataset."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Validating", leave=False):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features).squeeze(-1)
            unweighted_loss = criterion(logits, targets)
            # Per-sample weighting
            sample_weights = targets * w_fake + (1.0 - targets) * w_real
            loss = (unweighted_loss * sample_weights).mean()
            total_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, preds)
    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        roc_auc = 0.5
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
def train(
    manifest_path: str = "data/processed/manifest.csv",
    save_dir: str = "models",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    num_workers: int = 0,
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")
    os.makedirs(save_dir, exist_ok=True)
    # 1. Compute class weights
    w_real, w_fake = compute_class_weights(manifest_path)
    # 2. Datasets and DataLoaders
    print("\nLoading datasets...")
    train_dataset = AudioMFCCDataset(manifest_path=manifest_path, split="train")
    val_dataset = AudioMFCCDataset(manifest_path=manifest_path, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # 3. Model, Loss, Optimizer
    model = AudioLSTMClassifier(
        input_dim=40,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
    )
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_f1 = -1.0
    best_model_path = os.path.join(save_dir, "lstm_best.pt")
    print(
        f"\nStarting training for {epochs} epochs (Batch Size: {batch_size}, LR: {learning_rate})...\n"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            unweighted_loss = criterion(logits, targets)
            # Per-sample weighting for class imbalance
            sample_weights = targets * w_fake + (1.0 - targets) * w_real
            loss = (unweighted_loss * sample_weights).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        avg_train_loss = train_loss / len(train_dataset)
        val_metrics = evaluate(
            model, val_loader, criterion, w_real, w_fake, device
        )
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )
        # Save best model checkpoint based on Validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )
            print(
                f" ⭐ Best model saved! (Epoch {epoch}, Val F1: {best_val_f1:.4f}) -> {best_model_path}"
            )
    print(f"\nTraining completed! Best Validation F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Branch B LSTM Deepfake Audio Classifier"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/processed/manifest.csv"
    )
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    train(
        manifest_path=args.manifest,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
    )