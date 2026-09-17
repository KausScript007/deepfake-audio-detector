import csv
from pathlib import Path


DATA_ROOT = Path("data/raw/asvspoof2019/LA")
OUTPUT_FILE = Path("data/processed/manifest.csv")


PROTOCOLS = {
    "train": DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt",

    "val": DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt",
}


AUDIO_DIRS = {
    "train": DATA_ROOT / "ASVspoof2019_LA_train" / "flac",
    "val": DATA_ROOT / "ASVspoof2019_LA_dev" / "flac",
}


LABELS = {
    "bonafide": 0,
    "spoof": 1,
}


def read_protocol(protocol_file, split, audio_dir):
    """
    Read an ASVspoof CM protocol and create manifest rows.

    The protocol is the source of truth for:
        - which files belong to the split
        - REAL/FAKE labels
    """

    rows = []

    with open(protocol_file, "r") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            speaker_id = parts[0]
            utterance_id = parts[1]
            label_name = parts[4]

            if label_name not in LABELS:
                raise ValueError(
                    f"Unknown label: {label_name}"
                )

            audio_path = audio_dir / f"{utterance_id}.flac"

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            rows.append(
                {
                    "audio_path": str(audio_path),
                    "label": LABELS[label_name],
                    "split": split,
                }
            )

    return rows


def create_manifest():
    """
    Create the shared train/validation manifest.
    """

    rows = []

    for split, protocol_file in PROTOCOLS.items():
        audio_dir = AUDIO_DIRS[split]

        split_rows = read_protocol(
            protocol_file,
            split,
            audio_dir
        )

        rows.extend(split_rows)

        print(
            f"{split}: {len(split_rows)} samples"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "audio_path",
                "label",
                "split"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Manifest created: {OUTPUT_FILE}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    create_manifest()