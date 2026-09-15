import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    PyTorch Dataset for REAL/FAKE audio classification.

    Loads precomputed Mel-spectrograms from disk.
    """

    def __init__(
        self,
        manifest_path,
        split,
    ):
        self.manifest = pd.read_csv(manifest_path)

        self.manifest = self.manifest[
            self.manifest["split"] == split
        ].copy()

        self.split = split

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):

        row = self.manifest.iloc[index]

        # Cache filename matches the manifest row index.
        # Find the original global index using the manifest index.
        global_index = row.name

        mel_path = (
            f"data/processed/mels/"
            f"{self.split}_{global_index}.npy"
        )

        mel = np.load(mel_path)

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # Add CNN channel dimension.
        mel = mel.unsqueeze(0)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return mel, label