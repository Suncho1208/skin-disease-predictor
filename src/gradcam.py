"""Grad-CAM overlays for a few val images. No training.

Hooks the last conv layer of EfficientNet-B0, backprops the malignant logit,
and paints a heatmap over the resized lesion. Default picks: one TP, FN, FP, TN
from eval's val_predictions.csv using the TPR>=0.80 threshold (0.2845).

Examples:
  python src/gradcam.py --dummy --checkpoint checkpoints/best.pt
  python src/gradcam.py \\
    --hdf5 /content/data/train-image.hdf5 \\
    --checkpoint /content/drive/MyDrive/skin-disease-predictor/checkpoints/best_auroc094.pt \\
    --predictions /content/drive/MyDrive/skin-disease-predictor/eval/val_predictions.csv \\
    --out-dir /content/drive/MyDrive/skin-disease-predictor/gradcam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import IMAGE_SIZE, build_transforms, decode_hdf5_image
from models.classifier import build_classifier

PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_THRESHOLD = 0.2845


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--hdf5", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "gradcam")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--isic-ids",
        type=str,
        default="",
        help="Optional comma-separated isic_id list. Overrides auto-picking.",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    processed = args.processed_dir
    if args.dummy:
        val_csv = processed / "dummy_val.csv"
        hdf5_path = args.hdf5 or processed / "dummy_images.hdf5"
    else:
        val_csv = processed / "val.csv"
        hdf5_path = args.hdf5
        if hdf5_path is None:
            raise ValueError("Pass --hdf5 path/to/train-image.hdf5 (or use --dummy).")
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing image file: {hdf5_path}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing {val_csv}")
    return val_csv, hdf5_path


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{checkpoint} is not a train.py checkpoint (missing 'model').")
    backbone = payload.get("backbone", "efficientnet_b0")
    model = build_classifier(backbone=backbone, pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    print(f"loaded {checkpoint}  backbone={backbone}")
    return model


def target_conv(model: nn.Module) -> nn.Module:
    """Last spatial conv on timm EfficientNet; otherwise the last Conv2d."""
    if hasattr(model, "conv_head"):
        return model.conv_head
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return convs[-1]


class GradCAM:
    """Standard Grad-CAM: channel weights = GAP(gradients), then ReLU(weighted maps)."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.features: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer.register_forward_hook(self._on_forward)
        layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self.features = out

    def _on_backward(
        self, _module: nn.Module, _gin: tuple, gout: tuple
    ) -> None:
        self.gradients = gout[0]

    def __call__(self, image: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(image)
        logit.reshape(-1)[0].backward()
        if self.features is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.features).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().float().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        prob = float(torch.sigmoid(logit.detach().float()).reshape(-1)[0].cpu())
        return cam, prob


def load_rgb(hdf5_path: Path, isic_id: str, size: int = IMAGE_SIZE) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as handle:
        rgb = decode_hdf5_image(handle[isic_id][()])
    return np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))


def overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = plt.cm.jet(cam)[..., :3]
    rgb_f = rgb.astype(np.float32) / 255.0
    mixed = 0.5 * rgb_f + 0.5 * heat
    return (np.clip(mixed, 0, 1) * 255).astype(np.uint8)


def classify_row(target: int, prob: float, threshold: float) -> str:
    pred = int(prob >= threshold)
    if target == 1 and pred == 1:
        return "tp"
    if target == 1 and pred == 0:
        return "fn"
    if target == 0 and pred == 1:
        return "fp"
    return "tn"


def pick_rows(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One example per bucket, preferring extreme probabilities."""
    preds = preds.copy()
    preds["bucket"] = [
        classify_row(int(t), float(p), threshold)
        for t, p in zip(preds["target"], preds["prob_malignant"])
    ]
    chosen: list[pd.Series] = []
    for bucket, ascending in (("tp", False), ("fn", True), ("fp", False), ("tn", True)):
        part = preds[preds["bucket"] == bucket]
        if part.empty:
            print(f"no {bucket.upper()} examples at threshold={threshold:.3f}; skipping")
            continue
        chosen.append(part.sort_values("prob_malignant", ascending=ascending).iloc[0])
    if not chosen:
        raise ValueError("Could not pick any Grad-CAM examples from predictions.")
    return pd.DataFrame(chosen)


def save_pair(out_dir: Path, stem: str, rgb: np.ndarray, cam: np.ndarray) -> None:
    Image.fromarray(rgb).save(out_dir / f"{stem}_orig.png")
    Image.fromarray(overlay(rgb, cam)).save(out_dir / f"{stem}_cam.png")
    heat = (plt.cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(heat).save(out_dir / f"{stem}_heat.png")


def save_grid(out_dir: Path, panels: list[dict]) -> None:
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for row, panel in enumerate(panels):
        axes[row, 0].imshow(panel["rgb"])
        axes[row, 1].imshow(panel["heat"])
        axes[row, 2].imshow(panel["overlay"])
        axes[row, 0].set_ylabel(
            f"{panel['bucket'].upper()}\n{panel['isic_id']}\n"
            f"y={panel['target']} p={panel['prob']:.2f}",
            fontsize=8,
        )
        for ax, title in zip(axes[row], ("image", "Grad-CAM", "overlay")):
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "grid.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = pick_device()
    print(f"device={device}")
    val_csv, hdf5_path = resolve_paths(args)
    val_df = pd.read_csv(val_csv)
    val_df["isic_id"] = val_df["isic_id"].astype(str)

    if args.isic_ids:
        ids = [x.strip() for x in args.isic_ids.split(",") if x.strip()]
        picked = val_df[val_df["isic_id"].isin(ids)].copy()
        if args.predictions and args.predictions.exists():
            preds = pd.read_csv(args.predictions)
            preds["isic_id"] = preds["isic_id"].astype(str)
            picked = picked.merge(
                preds[["isic_id", "prob_malignant"]], on="isic_id", how="left"
            )
        else:
            picked["prob_malignant"] = np.nan
        picked["bucket"] = "custom"
    elif args.predictions and args.predictions.exists():
        preds = pd.read_csv(args.predictions)
        preds["isic_id"] = preds["isic_id"].astype(str)
        picked = pick_rows(preds, args.threshold)
        picked = picked.merge(val_df[["isic_id", "patient_id"]], on="isic_id", how="left")
    else:
        if args.dummy:
            picked = val_df.head(4).copy()
            picked["prob_malignant"] = np.nan
            picked["bucket"] = "dummy"
        else:
            raise ValueError(
                "Pass --predictions path/to/val_predictions.csv (from eval.py) "
                "or --isic-ids ISIC_xxx,ISIC_yyy."
            )

    model = load_model(args.checkpoint, device)
    cam_fn = GradCAM(model, target_conv(model))
    transform = build_transforms(train=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    panels: list[dict] = []
    for _, row in picked.iterrows():
        isic_id = str(row["isic_id"])
        rgb = load_rgb(hdf5_path, isic_id)
        tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
        cam, prob = cam_fn(tensor)
        target = int(row["target"])
        bucket = str(row.get("bucket", classify_row(target, prob, args.threshold)))
        stem = f"{bucket}_{isic_id}"
        save_pair(args.out_dir, stem, rgb, cam)
        panels.append(
            {
                "isic_id": isic_id,
                "bucket": bucket,
                "target": target,
                "prob": prob,
                "rgb": rgb,
                "heat": (plt.cm.jet(cam)[..., :3] * 255).astype(np.uint8),
                "overlay": overlay(rgb, cam),
            }
        )
        print(f"{bucket.upper():6} {isic_id}  target={target}  p={prob:.4f}  → {stem}_cam.png")

    save_grid(args.out_dir, panels)
    print(f"wrote {args.out_dir / 'grid.png'}  ({len(panels)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
