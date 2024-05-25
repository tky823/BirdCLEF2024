import math
import warnings
from typing import Any, Dict

from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["LinearWarmupCosineAnnealingLR"]


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warm-up + linear cool-down of learning rate.

    This learning rate schduler is used to train baseline model of BirdCLEF2024.

    Args:
        optimizer (Optimizer): Optimizer to adjust learning rate.
        warmup_steps (int): Number of exponential warm-up steps.
        suspend_steps (int): Number of constant learning rate steps between warm-up and cool-down.
        cooldown_steps (int): Number of linear cool-down steps after constant learning rate.
        initial_lr (float): Initial learning rate.
        eta_min (float): Minimum learning rate in cosine annealing.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        suspend_steps: int,
        cooldown_steps: int,
        initial_lr: float = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.suspend_steps = suspend_steps
        self.cooldown_steps = cooldown_steps
        self.initial_lr = initial_lr
        self.eta_min = eta_min

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def get_lr(self):
        warmup_steps = self.warmup_steps
        suspend_steps = self.suspend_steps
        cooldown_steps = self.cooldown_steps
        initial_lr = self.initial_lr
        eta_min = self.eta_min
        last_epoch = self.last_epoch

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        def _compute_lr(base_lr: float, group: Dict[str, Any]) -> float:
            if last_epoch < warmup_steps:
                normalized_steps = last_epoch / warmup_steps
                lr = (base_lr - initial_lr) * normalized_steps + initial_lr
            elif last_epoch < warmup_steps + suspend_steps:
                lr = base_lr
            else:
                step_after_suspend = last_epoch - (warmup_steps + suspend_steps)
                phase = math.pi * step_after_suspend / cooldown_steps
                lr = (base_lr - eta_min) * 0.5 * (1 + math.cos(phase)) + eta_min

            return lr

        return [
            _compute_lr(base_lr, group)
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
