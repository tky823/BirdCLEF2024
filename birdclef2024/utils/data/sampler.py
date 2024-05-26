import warnings
from typing import Dict, List, Optional

from torch.utils.data import WeightedRandomSampler

__all__ = [
    "BirdCLEF2024WeightedRandomSampler",
]


class BirdCLEF2024WeightedRandomSampler(WeightedRandomSampler):
    """Weighted random sampler for BirdCLEF2024.

    Args:
        label_mapping (dict): Dictionary that maps filename to primary label.
        num_samples (int): Number of samples at each epoch.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. In [#koutini2022efficient]_, ``1000``
            is used. Default: ``1``.
        filenames (list, optional): Filenames of audio. If ``None``, order of filenames are
            determined by alphabetical order using built-in ``sorted`` function.

    """

    def __init__(
        self,
        label_mapping: Dict[str, str],
        num_samples: int,
        replacement: bool = True,
        smooth: float = 1,
        filenames: Optional[List[str]] = None,
        generator=None,
    ) -> None:
        weights_per_sample = _get_sampling_weights(label_mapping, smooth=smooth)

        if filenames is None:
            warnings.warn(
                "It is highly recommended to set filenames to align orders between "
                "sampler and other modules.",
                UserWarning,
                stacklevel=2,
            )
            filenames = sorted(list(weights_per_sample.keys()))

        # from dict to list
        weights = []

        for filename in filenames:
            weight = weights_per_sample[filename]
            weights.append(weight)

        super().__init__(
            weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )


def _get_sampling_weights(label_mapping: Dict[str, str], smooth: float) -> Dict[str, float]:
    frequency_per_label = {}
    weight_per_sample = {}

    for label in label_mapping.values():
        if label not in frequency_per_label:
            frequency_per_label[label] = smooth

        frequency_per_label[label] += 1

    for filename, label in label_mapping.items():
        weight_per_sample[filename] = 1 / frequency_per_label[label]

    return weight_per_sample
