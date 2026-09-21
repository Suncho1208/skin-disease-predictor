"""Train EfficientNet-B0 on the ISIC subset (dummy locally, real hdf5 on Colab).

Examples:
  python src/train.py --dummy --epochs 1
  python src/train.py --hdf5 path/to/train-image.hdf5 --epochs 5 --batch-size 64
  python src/train.py --dummy --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import SkinLesionDataset, build_transforms
from models.classifier import build_classifier

PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only pull one train/val batch (no training).",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--hdf5", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/data/prepare.py --dummy"
        )
    return pd.read_csv(path)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
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
    return train_csv, val_csv, hdf5_path


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hdf5_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SkinLesionDataset(
        train_df, hdf5_path, transform=build_transforms(train=True)
    )
    val_ds = SkinLesionDataset(
        val_df, hdf5_path, transform=build_transforms(train=False)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def pos_weight_from_targets(targets: pd.Series, device: torch.device) -> torch.Tensor:
    """Weight the rare malignant class so BCE does not ignore it.

    pos_weight = n_benign / n_malignant. A false negative then costs that many
    times more than a false positive. Train has ~299 positives vs ~16k negatives,
    so this is ~50x, not a small tweak.
    """
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    if n_pos == 0:
        raise ValueError("Training split has zero malignant images.")
    weight = n_neg / n_pos
    print(f"pos_weight={weight:.2f}  (benign={n_neg:,} / malignant={n_pos:,})")
    return torch.tensor([weight], dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    desc: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_seen = 0
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)

            with torch.autocast(device_type=amp_device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                assert optimizer is not None and scaler is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_n = images.size(0)
            total_loss += loss.item() * batch_n
            n_seen += batch_n
            logits_all.append(logits.detach().float().cpu())
            targets_all.append(targets.detach().cpu())

    probs = torch.sigmoid(torch.cat(logits_all)).numpy().ravel()
    y_true = torch.cat(targets_all).numpy().ravel()
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc = float("nan")
    return total_loss / max(n_seen, 1), auroc


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_auroc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auroc": val_auroc,
            "backbone": args.backbone,
        },
        path,
    )


def smoke_test(train_loader: DataLoader, val_loader: DataLoader, hdf5_path: Path) -> int:
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print(f"hdf5: {hdf5_path}")
    print(
        f"train dataset: {len(train_loader.dataset)}  "
        f"val dataset: {len(val_loader.dataset)}"
    )
    print(
        "train batch "
        f"image={tuple(train_batch['image'].shape)} "
        f"target={tuple(train_batch['target'].shape)}"
    )
    print(
        "val batch   "
        f"image={tuple(val_batch['image'].shape)} "
        f"target={tuple(val_batch['target'].shape)}"
    )
    print("smoke test ok")
    return 0


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    use_amp = device.type == "cuda"
    print(f"device={device}  amp={use_amp}")

    train_csv, val_csv, hdf5_path = resolve_paths(args)
    train_df = load_split(train_csv)
    val_df = load_split(val_csv)
    train_loader, val_loader = make_loaders(
        train_df,
        val_df,
        hdf5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"hdf5: {hdf5_path}")
    print(f"train={len(train_df):,}  val={len(val_df):,}")

    if args.smoke:
        return smoke_test(train_loader, val_loader, hdf5_path)

    model = build_classifier(
        backbone=args.backbone, pretrained=not args.no_pretrained
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from_targets(train_df["target"], device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auroc = float("-inf")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auroc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            use_amp,
            scaler,
            desc=f"train {epoch}/{args.epochs}",
        )
        val_loss, val_auroc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            use_amp=use_amp,
            scaler=None,
            desc=f"val {epoch}/{args.epochs}",
        )
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_auroc={train_auroc:.4f}  "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}"
        )
        save_checkpoint(
            args.checkpoint_dir / "last.pt",
            epoch,
            model,
            optimizer,
            val_auroc,
            args,
        )
        if val_auroc == val_auroc and val_auroc >= best_auroc:
            best_auroc = val_auroc
            save_checkpoint(
                args.checkpoint_dir / "best.pt",
                epoch,
                model,
                optimizer,
                val_auroc,
                args,
            )
            print(f"  saved best.pt (val_auroc={val_auroc:.4f})")

    print(f"done. best val_auroc={best_auroc:.4f}  dir={args.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
