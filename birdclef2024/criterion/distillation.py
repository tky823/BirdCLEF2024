import torch
import torch.nn as nn

__all__ = [
    "DistillationCrossEntropyLoss",
]


class DistillationCrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss for knowledge distillation.

    Unlike nn.CrossEntropyLoss, this class can compute exp of target
    when ``is_logit_target=True``.
    """

    def __init__(
        self,
        *args,
        is_logit_target: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.is_logit_target = is_logit_target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_logit_target:
            target = torch.softmax(target, dim=-1)

        output = super().forward(input, target)

        return output
