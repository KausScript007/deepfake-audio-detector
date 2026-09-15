import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.dataset.audio_dataset import AudioDataset
from src.models.cnn import AudioCNN


# -------------------------
# Configuration
# -------------------------

MANIFEST = "data/processed/manifest.csv"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "cnn_best.pt"


# REAL is the minority class.
REAL_WEIGHT = 8.837209302325581


# -------------------------
# Evaluation
# -------------------------

def evaluate(model, loader, device):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for mel, labels in loader:

            mel = mel.to(device)

            logits = model(mel).squeeze(1)

            probabilities = torch.sigmoid(logits)
            all_probabilities.extend(
            probabilities.cpu().numpy()
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    return accuracy, precision, recall, f1, auc


# -------------------------
# Training
# -------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = AudioDataset(
        MANIFEST,
        split="train"
    )

    val_dataset = AudioDataset(
        MANIFEST,
        split="val"
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------------------------
    # Model
    # -------------------------

    model = AudioCNN().to(device)

    # -------------------------
    # Loss
    # -------------------------

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0

    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for mel, labels in train_loader:

            mel = mel.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(
                mel
            ).squeeze(1)

            losses = criterion(
                logits,
                labels
            )

            sample_weights = torch.where(
                labels == 0,
                torch.tensor(
                    REAL_WEIGHT,
                    device=device
                ),
                torch.tensor(
                    1.0,
                    device=device
                )
            )

            loss = (
                losses * sample_weights
            ).mean()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss
            / len(train_loader)
        )

        accuracy, precision, recall, f1, auc = (
            evaluate(
                model,
                val_loader,
                device
            )
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {average_loss:.4f}"
        )

        print(
            f"Val Accuracy: {accuracy:.4f}"
        )

        print(
            f"Val Precision: {precision:.4f}"
        )

        print(
            f"Val Recall: {recall:.4f}"
        )

        print(
            f"Val F1: {f1:.4f}"
        )

        print(
            f"Val ROC-AUC: {auc:.4f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                f"Saved best model → "
                f"{BEST_MODEL_PATH}"
            )


    print(
        "\nTraining complete."
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )


if __name__ == "__main__":
    main()