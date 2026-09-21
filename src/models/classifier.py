"""Image-only binary classifier: pretrained EfficientNet-B0, one logit.

`num_classes=1` means the model returns a raw score (logit) per image, not a
two-class softmax. We turn that into P(malignant) with `torch.sigmoid` later.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=1)
