import csv
from pathlib import Path


DATA_ROOT = Path(
    "data/raw/asvspoof2021"
    "/ASVspoof2021_DF_eval"
)

METADATA_FILE = Path(
    "data/raw/asvspoof2021"
    "/keys/DF/CM/trial_metadata.txt"
)

OUTPUT_FILE = Path(
    "data/processed/asvspoof2021_df_manifest.csv"
)


LABELS = {
    "bonafide": 0,
    "spoof": 1,
}


def create_manifest():

    rows = []

    with open(METADATA_FILE) as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) < 6:
                continue

            utterance_id = parts[1]
            label_name = parts[5]

            if label_name not in LABELS:
                raise ValueError(
                    f"Unknown label: {label_name}"
                )

            audio_path = (
                DATA_ROOT
                / "flac"
                / f"{utterance_id}.flac"
            )

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            rows.append(
                {
                    "audio_path": str(audio_path),
                    "label": LABELS[label_name],
                    "split": "external_test",
                }
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
                "split",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Manifest created: {OUTPUT_FILE}"
    )

    print(
        f"Total samples: {len(rows)}"
    )

    print(
        "REAL samples:",
        sum(row["label"] == 0 for row in rows)
    )

    print(
        "FAKE samples:",
        sum(row["label"] == 1 for row in rows)
    )


if __name__ == "__main__":
    create_manifest()
