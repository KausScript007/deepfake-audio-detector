import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.audio import (
    load_audio,
    segment_audio,
    extract_mel_spectrogram,
)


MANIFEST = Path("data/processed/manifest.csv")
OUTPUT_DIR = Path("data/processed/mels")


def cache_mels():

    manifest = pd.read_csv(MANIFEST)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(manifest)

    for index, row in manifest.iterrows():

        audio_path = Path(row["audio_path"])
        split = row["split"]

        output_file = (
            OUTPUT_DIR
            / f"{split}_{index}.npy"
        )

        if output_file.exists():
            continue

        audio, sr = load_audio(
            audio_path,
            sample_rate=16000
        )

        segments = segment_audio(
            audio,
            sample_rate=sr,
            segment_duration=4
        )

        # Baseline: use the first 4-second segment.
        audio_segment = segments[0]

        mel = extract_mel_spectrogram(
            audio_segment,
            sample_rate=sr
        )

        np.save(
            output_file,
            mel.astype(np.float32)
        )

        if (index + 1) % 100 == 0:
            print(
                f"Processed "
                f"{index + 1}/{total}"
            )

    print()
    print("Mel-spectrogram caching complete.")


if __name__ == "__main__":
    cache_mels()
