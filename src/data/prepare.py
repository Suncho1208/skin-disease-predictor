"""Build the 20k-benign subset, a patient-level split, and an optional dummy hdf5.

Local machines do not need the 400k-image bundle. `--dummy` writes a tiny hdf5
with JPEG bytes under the real `isic_id` keys so Dataset/DataLoader can be
tested here. On Colab, skip --dummy and point at the real train-image.hdf5.
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "train-metadata.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KEEP_COLS = ["isic_id", "patient_id", "target"]
BENIGN_N = 20_000
VAL_SIZE = 0.2
SEED = 42
DUMMY_N = 128
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=RAW_CSV)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--benign-n", type=int, default=BENIGN_N)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Also write a tiny dummy hdf5 for local DataLoader tests.",
    )
    parser.add_argument("--dummy-n", type=int, default=DUMMY_N)
    return parser.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download train-metadata.csv into data/raw/ first."
        )
    df = pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)
    df["isic_id"] = df["isic_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["target"] = df["target"].astype(int)
    return df


def build_subset(df: pd.DataFrame, benign_n: int, seed: int) -> pd.DataFrame:
    malignant = df[df["target"] == 1]
    benign = df[df["target"] == 0]
    if len(benign) < benign_n:
        raise ValueError(f"Asked for {benign_n} benign images, found {len(benign)}")
    sampled = benign.sample(n=benign_n, random_state=seed)
    subset = pd.concat([malignant, sampled], ignore_index=True)
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def patient_split(
    subset: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by patient, stratified on whether that patient has any malignant image.

    Grouping prevents the same person appearing in train and val. Stratifying on
    "this patient has at least one malignant lesion" keeps rare positives in both
    sides — a plain random patient split could unluckily put all 259 malignant
    patients in train.
    """
    patient_has_mal = subset.groupby("patient_id")["target"].max()
    train_patients, val_patients = train_test_split(
        patient_has_mal.index.to_numpy(),
        test_size=val_size,
        random_state=seed,
        stratify=patient_has_mal.to_numpy(),
    )
    train_patients = set(train_patients)
    val_patients = set(val_patients)
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Patient leakage: {len(overlap)} patients in both splits")

    train_df = subset[subset["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = subset[subset["patient_id"].isin(val_patients)].reset_index(drop=True)
    return train_df, val_df


def summarize(name: str, frame: pd.DataFrame) -> str:
    n_mal = int((frame["target"] == 1).sum())
    n_patients = frame["patient_id"].nunique()
    return (
        f"{name}: {len(frame):,} images | {n_mal} malignant | {n_patients:,} patients"
    )


def write_dummy_hdf5(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    path: Path,
    dummy_n: int,
    seed: int,
    image_size: int = IMAGE_SIZE,
) -> pd.DataFrame:
    """Write JPEG bytes under real isic_id keys. Prefer keeping some malignant IDs."""
    rng = np.random.default_rng(seed)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    malignant = combined[combined["target"] == 1]
    benign = combined[combined["target"] == 0]

    n_mal = min(len(malignant), max(16, dummy_n // 8))
    n_ben = min(len(benign), dummy_n - n_mal)
    dummy = pd.concat(
        [
            malignant.sample(n=n_mal, random_state=seed),
            benign.sample(n=n_ben, random_state=seed),
        ],
        ignore_index=True,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    with h5py.File(path, "w") as handle:
        for isic_id in dummy["isic_id"]:
            color = tuple(int(x) for x in rng.integers(0, 256, size=3))
            image = Image.new("RGB", (image_size, image_size), color)
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=85)
            payload = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            handle.create_dataset(str(isic_id), data=payload)

    return dummy


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata)
    subset = build_subset(df, benign_n=args.benign_n, seed=args.seed)
    train_df, val_df = patient_split(subset, val_size=args.val_size, seed=args.seed)

    train_path = args.out_dir / "train.csv"
    val_path = args.out_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(summarize("full CSV", df))
    print(summarize("subset  ", subset))
    print(summarize("train   ", train_df))
    print(summarize("val     ", val_df))
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")

    if args.dummy:
        dummy_path = args.out_dir / "dummy_images.hdf5"
        dummy = write_dummy_hdf5(
            train_df, val_df, dummy_path, dummy_n=args.dummy_n, seed=args.seed
        )
        dummy_train = train_df[train_df["isic_id"].isin(dummy["isic_id"])]
        dummy_val = val_df[val_df["isic_id"].isin(dummy["isic_id"])]
        dummy_train.to_csv(args.out_dir / "dummy_train.csv", index=False)
        dummy_val.to_csv(args.out_dir / "dummy_val.csv", index=False)
        print(
            f"wrote {dummy_path} "
            f"({len(dummy)} fake images: "
            f"{int((dummy['target'] == 1).sum())} malignant keys)"
        )
        print(summarize("dummy train", dummy_train))
        print(summarize("dummy val  ", dummy_val))

    return 0


if __name__ == "__main__":
    sys.exit(main())
