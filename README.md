# Skin lesion classifier (ISIC 2024 / SLICE-3D)

Image-only binary classifier for the [ISIC 2024](https://www.kaggle.com/competitions/isic-2024-challenge) SLICE-3D dataset. Fine-tunes pretrained **EfficientNet-B0** (`timm` / PyTorch) to output P(malignant). Built for a DevConf.US 2026 lightning talk.

This is a student research project, **not a medical device**. Dataset license is CC BY-NC 4.0.

## Results (locked)

Patient-level train/val split (no shared patients). All **393** malignant images plus **20,000** sampled benign images.

| | Always predict benign | This model |
|---|---:|---:|
| Val accuracy | 97.6% | 87.4% (at TPR ≥ 80% threshold 0.28) |
| Val AUROC | 0.50 | **0.94** |

At that cutoff: **76 / 94** cancers caught, **18** missed, **465** false alarms (precision 14%).

Figures: [`results/eval/roc_curve.png`](results/eval/roc_curve.png), [`results/eval/confusion_matrix.png`](results/eval/confusion_matrix.png), Grad-CAM grid [`results/gradcam/grid.png`](results/gradcam/grid.png). Run log: [`results/run_2026-09-20.md`](results/run_2026-09-20.md).

## Why these choices

- **Split by patient, not image.** One person can have thousands of photos. A random image split leaks the same mole into train and val.
- **AUROC, not accuracy.** ~0.1% of images are malignant. Always-benign is ~98% accurate and useless.
- **Image-only EfficientNet-B0.** Small enough to train in ~10 minutes on a Colab A100. No tabular fusion or stacking.

## Repo layout

```text
src/data/prepare.py   # 20k-benign subset + patient split (+ dummy hdf5 locally)
src/data/dataset.py   # hdf5 Dataset + Albumentations
src/models/           # EfficientNet-B0, 1 logit
src/train.py          # train / val loop, pos_weight, checkpoints
src/eval.py           # AUROC, pAUC, ROC, confusion matrix
src/gradcam.py        # Grad-CAM overlays on val examples
src/app.py            # Gradio demo: upload a photo → score + heatmap
notebooks/            # Colab training notebook (A100)
data/processed/       # train.csv / val.csv (ids + labels only)
```

Images live in Kaggle `train-image.hdf5` (~2 GB). They are not in git. The checkpoint `best_auroc094.pt` lives on Drive / local `checkpoints/` (gitignored).

## Demo (upload a photo)

Copy `best_auroc094.pt` from Drive into `checkpoints/`, then:

```bash
python src/app.py --checkpoint checkpoints/best_auroc094.pt
```

Opens a local Gradio page: image in, P(malignant) + Grad-CAM out. Not a medical device. Do not commit the `.pt` file.

## Local dummy path (no GPU, no 400k images)

```bash
python src/data/prepare.py --dummy
python src/train.py --dummy --epochs 1
python src/eval.py --dummy --checkpoint checkpoints/best.pt
```

## Colab (real images)

Use `notebooks/colab_train.ipynb` on an A100: Kaggle secret `KAGGLE_API_TOKEN`, download `train-image.hdf5`, then:

```bash
python src/train.py --hdf5 /content/data/train-image.hdf5 --epochs 8 --batch-size 64 --num-workers 2
python src/eval.py --hdf5 /content/data/train-image.hdf5 \
  --checkpoint path/to/best_auroc094.pt \
  --out-dir ./results/eval
```

## License

Code: MIT (see `LICENSE`). Data: ISIC SLICE-3D, CC BY-NC 4.0.
