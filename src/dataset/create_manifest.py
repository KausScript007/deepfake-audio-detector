import csv
from pathlib import Path


DATA_ROOT = Path("data/raw/asvspoof2019")
OUTPUT_FILE = Path("data/processed/manifest.csv")


def create_manifest():
    """
    Create a shared dataset manifest.

    The manifest will contain:
        audio_path,label,split

    Labels:
        0 = REAL / bona fide
        1 = FAKE / spoof

    Splits:
        train = ASVspoof 2019 LA training set
        val   = ASVspoof 2019 LA development set
    """

    rows = []

    # Dataset-specific parsing will be added
    # after we inspect the extracted ASVspoof files.

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["audio_path", "label", "split"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest created: {OUTPUT_FILE}")
    print(f"Number of entries: {len(rows)}")


if __name__ == "__main__":
    create_manifest()
