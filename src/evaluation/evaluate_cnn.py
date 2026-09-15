import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

from src.dataset.audio_dataset import AudioDataset
from src.models.cnn import AudioCNN


MANIFEST = "data/processed/manifest.csv"
MODEL_PATH = "models/cnn_best.pt"

BATCH_SIZE = 16


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -------------------------
    # Dataset
    # -------------------------

    dataset = AudioDataset(
        MANIFEST,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    # -------------------------
    # Model
    # -------------------------

    model = AudioCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    # -------------------------
    # Predictions
    # -------------------------

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for mel, labels in loader:

            mel = mel.to(device)

            logits = model(
                mel
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    all_predictions = (
        all_probabilities >= 0.5
    ).astype(int)

    # -------------------------
    # Metrics
    # -------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities,
    )

    print()
    print("CNN Validation Results")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # -------------------------
    # Confusion Matrix
    # -------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print()
    print("Confusion Matrix")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "REAL",
            "FAKE",
        ],
    ).plot()

    plt.title(
        "CNN Validation Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "results/cnn_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # -------------------------
    # ROC Curve
    # -------------------------

    fpr, tpr, _ = roc_curve(
        all_labels,
        all_probabilities,
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "CNN Validation ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/cnn_roc_curve.png",
        dpi=300,
    )

    plt.close()

    print()
    print(
        "Saved evaluation plots to results/"
    )


if __name__ == "__main__":
    main()
