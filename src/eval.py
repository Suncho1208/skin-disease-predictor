"""Evaluate a saved checkpoint on the val split. No training.

Writes ROC + confusion-matrix figures and a metrics JSON. Default threshold
is the most conservative cutoff that still catches ~80% of malignancies
(TPR >= 0.80), which matches the ISIC high-sensitivity story.

Examples:
  python src/eval.py --hdf5 /content/data/train-image.hdf5 \\
    --checkpoint /content/drive/MyDrive/skin-disease-predictor/checkpoints/best_auroc094.pt \\
    --out-dir /content/drive/MyDrive/skin-disease-predictor/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import SkinLesionDataset, build_transforms
from models.classifier import build_classifier

PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MIN_TPR = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--hdf5", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If omitted, use the highest threshold that still reaches TPR>=0.80.",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_val_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    processed = args.processed_dir
    if args.dummy:
        val_csv = processed / "dummy_val.csv"
        hdf5_path = args.hdf5 or processed / "dummy_images.hdf5"
    else:
        val_csv = processed / "val.csv"
        hdf5_path = args.hdf5
        if hdf5_path is None:
            raise ValueError("Pass --hdf5 path/to/train-image.hdf5 (or use --dummy).")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing {val_csv}")
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing image file: {hdf5_path}")
    return val_csv, hdf5_path


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{checkpoint} is not a train.py checkpoint (missing 'model').")
    backbone = payload.get("backbone", "efficientnet_b0")
    model = build_classifier(backbone=backbone, pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    logged = payload.get("val_auroc")
    print(f"loaded {checkpoint}  backbone={backbone}  ckpt_val_auroc={logged}")
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    isic_ids: list[str] = []
    patient_ids: list[str] = []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
        batch_probs = torch.sigmoid(logits.float()).cpu().numpy().ravel()
        probs.append(batch_probs)
        targets.append(batch["target"].numpy().ravel())
        isic_ids.extend(batch["isic_id"])
        patient_ids.extend(batch["patient_id"])
    return np.concatenate(probs), np.concatenate(targets), isic_ids, patient_ids


def isic_pauc(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = MIN_TPR) -> float:
    """ISIC 2024-style partial AUC in the TPR >= min_tpr region.

    sklearn's roc_auc_score(max_fpr=...) is a low-FPR slice. The challenge
    cares about high sensitivity, so labels/scores are flipped first. The
    rescale matches the public ISIC 2024 metric implementation.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.min() == y_true.max():
        return float("nan")
    v_gt = np.abs(y_true - 1.0)
    v_pred = 1.0 - y_score
    max_fpr = 1.0 - min_tpr
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return float(
        0.5 * max_fpr**2
        + (max_fpr - 0.5 * max_fpr**2)
        / (1.0 - 0.5 * max_fpr)
        * (partial_auc_scaled - 0.5 * max_fpr**2)
    )


def threshold_for_min_tpr(
    y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = MIN_TPR
) -> float:
    """Highest threshold that still reaches min_tpr (catch that fraction of cancers)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ok = np.where(tpr >= min_tpr)[0]
    if len(ok) == 0:
        return float(np.min(y_score))
    idx = ok[0]
    return float(thresholds[idx]) if idx < len(thresholds) else 0.0


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, auroc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"model (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="always-benign / random (0.50)")
    ax.axhline(MIN_TPR, linestyle=":", color="black", alpha=0.6, label=f"TPR={MIN_TPR:.0%}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["benign (0)", "malignant (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Validation confusion matrix (threshold={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = pick_device()
    print(f"device={device}")

    val_csv, hdf5_path = resolve_val_paths(args)
    val_df = pd.read_csv(val_csv)
    loader = DataLoader(
        SkinLesionDataset(val_df, hdf5_path, transform=build_transforms(train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, device)
    y_score, y_true, isic_ids, patient_ids = predict(model, loader, device)

    auroc = float(roc_auc_score(y_true, y_score))
    pauc = isic_pauc(y_true, y_score)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    baseline_acc = n_neg / n

    threshold = (
        args.threshold
        if args.threshold is not None
        else threshold_for_min_tpr(y_true, y_score)
    )
    y_hat = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    metrics = {
        "n": n,
        "n_malignant": n_pos,
        "n_benign": n_neg,
        "auroc": auroc,
        "pauc_tpr_ge_0.80": pauc,
        "always_benign_accuracy": baseline_acc,
        "always_benign_auroc": 0.5,
        "threshold": threshold,
        "threshold_rule": (
            "user-supplied" if args.threshold is not None else "highest cutoff with TPR>=0.80"
        ),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "checkpoint": str(args.checkpoint),
        "hdf5": str(hdf5_path),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "val_predictions.csv"
    pd.DataFrame(
        {
            "isic_id": isic_ids,
            "patient_id": patient_ids,
            "target": y_true.astype(int),
            "prob_malignant": y_score,
        }
    ).to_csv(pred_path, index=False)

    roc_path = args.out_dir / "roc_curve.png"
    cm_path = args.out_dir / "confusion_matrix.png"
    metrics_path = args.out_dir / "metrics.json"
    plot_roc(y_true, y_score, auroc, roc_path)
    plot_confusion(cm, threshold, cm_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"val images={n:,}  malignant={n_pos}  benign={n_neg}")
    print(f"AUROC={auroc:.4f}  (expect ~0.936 if this is best_auroc094.pt)")
    print(f"pAUC (TPR>=0.80)={pauc:.4f}  [ISIC-style; usually much lower than AUROC]")
    print(
        f"always-benign baseline: accuracy={baseline_acc:.4f}  AUROC=0.5000"
    )
    print(f"threshold={threshold:.4f} ({metrics['threshold_rule']})")
    print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(
        f"at that threshold: acc={metrics['accuracy']:.4f}  "
        f"precision={metrics['precision']:.4f}  "
        f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}"
    )
    print(f"wrote {metrics_path}")
    print(f"wrote {roc_path}")
    print(f"wrote {cm_path}")
    print(f"wrote {pred_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
