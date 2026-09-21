"""Local DataLoader smoke test. Training on the A100 will land here later.

With --dummy this reads the tiny hdf5 from `python src/data/prepare.py --dummy`.
On Colab, drop --dummy and pass the real hdf5 + the same train.csv / val.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import SkinLesionDataset, build_transforms

PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--hdf5", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/data/prepare.py --dummy"
        )
    return pd.read_csv(path)


def main() -> int:
    args = parse_args()
    processed = args.processed_dir

    if args.dummy:
        train_csv = processed / "dummy_train.csv"
        val_csv = processed / "dummy_val.csv"
        hdf5_path = args.hdf5 or processed / "dummy_images.hdf5"
    else:
        train_csv = processed / "train.csv"
        val_csv = processed / "val.csv"
        hdf5_path = args.hdf5
        if hdf5_path is None:
            raise ValueError("Pass --hdf5 path/to/train-image.hdf5 (or use --dummy).")

    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing image file: {hdf5_path}")

    train_ds = SkinLesionDataset(
        load_split(train_csv), hdf5_path, transform=build_transforms(train=True)
    )
    val_ds = SkinLesionDataset(
        load_split(val_csv), hdf5_path, transform=build_transforms(train=False)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print(f"hdf5: {hdf5_path}")
    print(f"train dataset: {len(train_ds)}  val dataset: {len(val_ds)}")
    print(
        "train batch "
        f"image={tuple(train_batch['image'].shape)} "
        f"dtype={train_batch['image'].dtype} "
        f"target={tuple(train_batch['target'].shape)}"
    )
    print(
        "val batch   "
        f"image={tuple(val_batch['image'].shape)} "
        f"dtype={val_batch['image'].dtype} "
        f"target={tuple(val_batch['target'].shape)}"
    )
    print("train targets in this batch:", train_batch["target"].tolist())
    print("val targets in this batch:  ", val_batch["target"].tolist())
    print("smoke test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
