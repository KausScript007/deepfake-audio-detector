from pathlib import Path
import pandas as pd
import librosa
from tqdm import tqdm

# Configuration

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/dataset_inventroy.csv")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Helper Function

def get_label(file_path: Path) -> int: #Assigns the label based on the parent folder
    folder_name = file_path.parent.name.lower()

    if folder_name == "real":
        return 0

    if folder_name == "fake":
        return 1

    raise ValueError(
        f"Unknown class folder '{folder_name}' for file: {file_path}"
    )

# Main inventory function

def create_inventory():
    audio_files=[
        file for file in RAW_DATA_DIR.rglob("*")
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print(f"FOUND {len(audio_files)} audio files.")

    records = []

    for file_path in tqdm(audio_files, desc ="Scanning audio files"):
        try:
            # Load only enough information to determine duration and sampling rate
            audio, sample_rate = librosa.load(
                file_path, sr=None, mono = True
            )

            duration = len(audio) / sample_rate

            label = get_label(file_path)

            records.append({
                "filename": file_path.name,
                "filepath": str(file_path),
                "label": label,
                "class": "real" if label ==0 else "fake",
                "duration_seconds": round(duration, 3),
                "sample_rate": sample_rate
            })

        except Exception as e:
            print(f"\nCould not process {file_path}")
            print(f"Reason: {e}")

    df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nInventory created successfully.")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nClass distribution:")
    print(df["class"].value_counts())

    print("\nSampling rates:")
    print(df["sample_rate"].value_counts())

    print("\nDuration statistics:")
    print(df["duration_seconds"].describe())


if __name__ == "__main__":
    create_inventory()