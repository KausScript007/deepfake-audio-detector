import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from src.dataset.audio_dataset_mfcc import AudioMFCCDataset
from src.models.lstm import AudioLSTMClassifier
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and the optimal EER threshold.

    Standard biometric metric for ASVspoof audio anti-spoofing.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    # The EER point is where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    return float(eer), float(eer_threshold)


def evaluate_model(
    model_path: str = "models/lstm_best.pt",
    manifest_path: str = "data/processed/manifest.csv",
    split: str = "val",
    batch_size: int = 32,
    results_dir: str = "results",
):
    """Evaluates the saved model checkpoint and generates evaluation reports."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Branch B (LSTM) on '{split}' split ===")
    print(f"Device: {device}")
    print(f"Model checkpoint: {model_path}")

    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Dataset
    dataset = AudioMFCCDataset(manifest_path=manifest_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"Dataset size: {len(dataset):,} samples")

    # 2. Instantiate and load model weights
    model = AudioLSTMClassifier(
        input_dim=40,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.3,
    )
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 3. Inference loop
    all_targets = []
    all_probs = []

    print("\nRunning inference...")
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Inference"):
            features = features.to(device)
            # Use predict_proba
            probs = model.predict_proba(features).squeeze(-1)
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    eer, eer_thresh = compute_eer(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "split": split,
        "total_samples": len(y_true),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "eer": round(float(eer), 4),
        "eer_threshold": round(float(eer_thresh), 4),
        "confusion_matrix": {
            "tn_real": int(cm[0, 0]),
            "fp_fake": int(cm[0, 1]),
            "fn_real": int(cm[1, 0]),
            "tp_fake": int(cm[1, 1]),
        },
    }

    # 5. Display Summary
    print("\n" + "=" * 50)
    print("         BRANCH B (LSTM) EVALUATION REPORT        ")
    print("=" * 50)
    print(f" Accuracy:       {acc * 100:.2f}%")
    print(f" Precision:      {prec:.4f}")
    print(f" Recall:         {rec:.4f}")
    print(f" F1-Score:       {f1:.4f}")
    print(f" ROC-AUC:        {roc_auc:.4f}")
    print(f" EER:            {eer * 100:.2f}% (at threshold {eer_thresh:.4f})")
    print("\nConfusion Matrix:")
    print(f"  [TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}]  <- Actual REAL (0)")
    print(f"  [FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}]  <- Actual FAKE (1)")
    print("=" * 50)

    # 6. Save JSON report
    report_path = os.path.join(results_dir, f"lstm_evaluation_{split}.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics summary -> {report_path}")

    # 7. Plot and Save Confusion Matrix
    cm_plot_path = os.path.join(results_dir, f"lstm_confusion_matrix_{split}.png")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL (0)", "FAKE (1)"],
        yticklabels=["REAL (0)", "FAKE (1)"],
    )
    plt.title(f"Branch B (BiLSTM) Confusion Matrix ({split.upper()})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot -> {cm_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Branch B BiLSTM Classifier"
    )
    parser.add_argument(
        "--model_path", type=str, default="models/lstm_best.pt"
    )
    parser.add_argument(
        "--manifest", type=str, default="data/processed/manifest.csv"
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        manifest_path=args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
    )