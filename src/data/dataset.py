"""Image-only ISIC 2024 Dataset and Albumentations transforms.

The Kaggle image bundle stores one JPEG per `isic_id` inside a single .hdf5
file. We open that file lazily so DataLoader worker processes each get their
own handle (h5py file objects cannot be pickled and sent to workers).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, image_size: int = IMAGE_SIZE) -> A.Compose:
    """Training gets flips/rotation/color jitter; validation only resize+normalize."""
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def decode_hdf5_image(payload: np.ndarray) -> np.ndarray:
    """Turn one hdf5 value into an RGB uint8 array (H, W, 3).

    The official Kaggle file stores JPEG bytes. A dummy file we create locally
    uses the same format. If someone stored a raw HWC array instead, accept that too.
    """
    if getattr(payload, "ndim", 0) == 3:
        image = np.asarray(payload)
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {image.shape}")
        return image
    raw = payload.tobytes() if hasattr(payload, "tobytes") else bytes(payload)
    return np.array(Image.open(BytesIO(raw)).convert("RGB"))


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        hdf5_path: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        needed = {"isic_id", "patient_id", "target"}
        missing = needed - set(metadata.columns)
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")

        self.metadata = metadata.reset_index(drop=True)
        self.hdf5_path = str(hdf5_path)
        self.transform = transform
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.metadata)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        isic_id = str(row["isic_id"])
        image = decode_hdf5_image(self._file()[isic_id][()])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {
            "image": image,
            "target": np.float32(row["target"]),
            "isic_id": isic_id,
            "patient_id": str(row["patient_id"]),
        }
