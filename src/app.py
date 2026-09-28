"""Gradio demo: upload a lesion photo → P(malignant) + Grad-CAM overlay.

Not a medical device. Run locally with the frozen talk checkpoint:

  python src/app.py --checkpoint checkpoints/best_auroc094.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import IMAGE_SIZE, build_transforms
from gradcam import GradCAM, load_model, overlay, pick_device, target_conv

PROJECT_ROOT = SRC_DIR.parent
DEFAULT_THRESHOLD = 0.2845
DISCLAIMER = (
    "Research demo only — not a medical device, not a diagnosis, not FDA-cleared. "
    "Do not use this to make health decisions. Dataset license is CC BY-NC 4.0."
)


def resolve_checkpoint(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    candidates.extend(
        [
            ckpt_dir / "best_auroc094.pt",
            ckpt_dir / "best.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found. Copy best_auroc094.pt from Drive into checkpoints/ "
        "or pass --checkpoint /path/to/best_auroc094.pt"
    )


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    return np.array(pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))


class Predictor:
    def __init__(self, checkpoint: Path, threshold: float) -> None:
        self.device = pick_device()
        self.threshold = threshold
        self.model = load_model(checkpoint, self.device)
        self.cam = GradCAM(self.model, target_conv(self.model))
        self.transform = build_transforms(train=False)
        print(f"demo checkpoint={checkpoint}  device={self.device}  threshold={threshold}")

    def __call__(self, image: np.ndarray | None) -> tuple[np.ndarray | None, str, str]:
        if image is None:
            raise gr.Error("Upload a photo first.")
        rgb = prepare_rgb(image)
        tensor = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device)
        cam, prob = self.cam(tensor)
        flagged = prob >= self.threshold
        flag = (
            f"Flag for review  (score ≥ {self.threshold:.2f})"
            if flagged
            else f"Below screening threshold  (score < {self.threshold:.2f})"
        )
        return overlay(rgb, cam), f"{prob:.1%}", flag


def build_ui(predictor: Predictor) -> gr.Blocks:
    with gr.Blocks(title="Skin lesion risk score") as demo:
        gr.Markdown(
            "# Skin lesion risk score\n"
            "Upload a close-up photo of a single lesion. "
            "The model returns **P(malignant)** and a Grad-CAM overlay "
            "(coarse heatmap of pixels that pushed the score up).\n\n"
            f"**{DISCLAIMER}**"
        )
        with gr.Row():
            inp = gr.Image(type="numpy", label="Lesion photo", sources=["upload"])
            out_img = gr.Image(label="Grad-CAM overlay", type="numpy")
        with gr.Row():
            out_prob = gr.Textbox(label="P(malignant)", interactive=False)
            out_flag = gr.Textbox(label=f"Flag at threshold {predictor.threshold:.2f}", interactive=False)
        gr.Button("Score photo", variant="primary").click(
            predictor, inputs=inp, outputs=[out_img, out_prob, out_flag]
        )
        gr.Markdown(
            "Threshold 0.28 is the operating point from the talk "
            "(~80% of cancers caught on the held-out patient val set). "
            "Precision there was ~14%: most flags are still benign."
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--share", action="store_true", help="Gradio public link (expires).")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    predictor = Predictor(checkpoint, args.threshold)
    demo = build_ui(predictor)
    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
