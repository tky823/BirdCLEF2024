from typing import Optional

import torch
import torch.nn as nn
from audyn.utils.data.birdclef.birdclef2024 import num_primary_labels
from audyn.utils.data.birdclef.birdclef2024.models.baseline import BaselineModel as _BaselineModel
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

__all__ = [
    "BaselineModel",
    "LargeBaselineModel",
]


class BaselineModel(_BaselineModel):
    """Wrapper class of _BaselineModel in audyn."""


class LargeBaselineModel(nn.Module):
    """Baseline model.

    Implementation is ported from https://www.kaggle.com/code/awsaf49/birdclef24-kerascv-starter-train.
    Backbone architecture is EfficientNet B7.
    """  # noqa: E501

    def __init__(
        self,
        weights: Optional[EfficientNet_B7_Weights] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_classes is None:
            num_classes = num_primary_labels

        efficientnet = efficientnet_b7(weights=weights)
        self.backbone = efficientnet.features
        self.avgpool = efficientnet.avgpool

        last_block = self.backbone[-1]
        last_conv2d: nn.Conv2d = last_block[0]
        dropout: nn.Dropout = efficientnet.classifier[0]
        num_features = last_conv2d.out_channels
        dropout_p = dropout.p

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of baseline model.

        Args:
            input (torch.Tensor): Mel-spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Logit of shape (batch_size, num_classes).

        """
        assert input.dim() == 3, "Only 3D input is supported."

        x = input.unsqueeze(dim=-3)
        x = x.expand((-1, 3, -1, -1))
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.squeeze(dim=(-2, -1))
        output = self.classifier(x)

        return output
