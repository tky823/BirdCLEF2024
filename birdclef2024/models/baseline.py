import os
from typing import Optional

import torch
import torch.nn as nn
from audyn.utils import instantiate_model
from audyn.utils.data.birdclef.birdclef2024 import num_primary_labels
from audyn.utils.data.birdclef.birdclef2024.models.baseline import BaselineModel as _BaselineModel
from omegaconf import OmegaConf
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    efficientnet_b0,
    efficientnet_b7,
)

__all__ = [
    "BaselineModel",
    "SmallBaselineModel",
    "LargeBaselineModel",
]


class BaselineModel(_BaselineModel):
    """Wrapper class of _BaselineModel in audyn."""

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_classes: Optional[int] = None,
    ) -> "BaselineModel":
        path = pretrained_model_name_or_path

        if os.path.exists(path):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model_state_dict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            model: BaselineModel = instantiate_model(resolved_config.model)
            model.load_state_dict(model_state_dict)

            dropout: nn.Dropout = model.classifier[0]
            linear: nn.Linear = model.classifier[-1]

            if num_classes is not None and linear.out_features != num_classes:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout.p),
                    nn.Linear(linear.in_features, num_classes),
                )
        else:
            raise FileNotFoundError(f"{path} is not found.")

        return model


class SmallBaselineModel(nn.Module):
    """Baseline model.

    Implementation is ported from https://www.kaggle.com/code/awsaf49/birdclef24-kerascv-starter-train.
    Backbone architecture is EfficientNet B0.
    """  # noqa: E501

    def __init__(
        self,
        weights: Optional[EfficientNet_B0_Weights] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_classes is None:
            num_classes = num_primary_labels

        efficientnet = efficientnet_b0(weights=weights)
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
