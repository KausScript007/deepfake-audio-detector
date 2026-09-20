from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.contract import LABEL_FAKE, LABEL_REAL, SAMPLE_RATE, SEGMENT_DURATION
from src.preprocessing.audio import load_audio, segment_audio
from src.preprocessing.mfcc import extract_mfcc


class AudioMFCCDataset(Dataset):
    def __init__(
        self,
        manifest_path: str = "data/processed/manifest.csv",
        split: str = "train",
        sr: int = SAMPLE_RATE,
        segment_duration: int = SEGMENT_DURATION,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.split = split
        self.sr = sr
        self.segment_duration = segment_duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = int(row["label"])

        # 1. Load audio using shared loader (FFmpeg-backed, mono, 16kHz, trimmed & normalized)
        audio, _ = load_audio(audio_path, sample_rate=self.sr)

        # 2. Segment audio to 4.0s (64,000 samples)
        segments = segment_audio(
            audio,
            sample_rate=self.sr,
            segment_duration=self.segment_duration
        )
        primary_segment = segments[0]  # shape: (64000,)

        # 3. Extract MFCC sequence -> shape: (126, 40)
        mfcc_seq = extract_mfcc(
            audio=primary_segment,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # 4. Convert to PyTorch Tensors
        feature_tensor = torch.from_numpy(mfcc_seq).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return feature_tensor, label_tensor