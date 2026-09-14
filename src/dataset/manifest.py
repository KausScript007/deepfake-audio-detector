import pandas as pd

from pathlib import Path


MANIFEST_FILE = Path("data/processed/manifest.csv")


def load_manifest():
    """
    Load the shared dataset manifest.
    """

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    return pd.read_csv(MANIFEST_FILE)


def get_split(manifest, split):
    """
    Return only samples belonging to the requested split.

    Valid splits:
        train
        val
    """

    if split not in {"train", "val"}:
        raise ValueError(
            "split must be either 'train' or 'val'"
        )

    return manifest[
        manifest["split"] == split
    ].reset_index(drop=True)
