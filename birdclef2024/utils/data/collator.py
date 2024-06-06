import warnings
from typing import Any, Dict, List, Optional

import torchvision
from audyn.utils.data.birdclef.birdclef2024.collator import (
    BirdCLEF2024BaselineCollator as _BirdCLEF2024BaselineCollator,
)
from audyn.utils.data.collator import Collator
from audyn.utils.data.dataset import Composer

from .composer import (
    BirdCLEF2024AudioChunkingComposer,
    BirdCLEF2024SharedAudioComposer,
    BirdCLEF2024VadBasedSharedAudioComposer,
)

__all__ = [
    "BirdCLEF2024BaselineCollator",
    "BirdCLEF2024PretrainCollator",
    "BirdCLEF2024AudioChunkingCollator",
]


class BirdCLEF2024BaselineCollator(_BirdCLEF2024BaselineCollator):
    """Alias of audyn.utils.data.birdclef.birdclef2024.collator.BirdCLEF2024BaselineCollator"""


class BirdCLEF2024PretrainCollator(BirdCLEF2024BaselineCollator):
    def __init__(
        self,
        composer: Composer | None = None,
        melspectrogram_key: str = "melspectrogram",
        label_index_key: str = "label_index",
        alpha: float = 0.4,
    ) -> None:
        super().__init__(
            composer=composer,
            melspectrogram_key=melspectrogram_key,
            label_index_key=label_index_key,
            alpha=alpha,
        )

        from . import num_birdclef2024_pretrain_primary_labels

        try:
            from torchvision.transforms.v2 import MixUp
        except ImportError:
            raise ImportError(f"MixUp is not supported by torchvision=={torchvision.__version__}")

        self.mixup = MixUp(
            alpha=alpha,
            num_classes=num_birdclef2024_pretrain_primary_labels,
        )


class BirdCLEF2024AudioChunkingCollator(Collator):
    """Collator for BirdCLEF2024, which supports chunking.

    This class can be used for inference.
    """

    def __init__(
        self,
        composer: Optional[Composer] = None,
        filename_key: str = "filename",
        sample_rate_key: str = "sample_rate",
        waveform_key: str = "waveform",
        melspectrogram_key: str = "melspectrogram",
    ) -> None:
        if not isinstance(
            composer,
            (
                BirdCLEF2024AudioChunkingComposer,
                BirdCLEF2024SharedAudioComposer,
                BirdCLEF2024VadBasedSharedAudioComposer,
            ),
        ):
            warnings.warn(
                f"{type(composer)} is not supported by BirdCLEF2024AudioChunkingCollator, "
                "which may cause unexpected behavior.",
                stacklevel=2,
            )

        super().__init__(composer=composer)

        self.filename_key = filename_key
        self.sample_rate_key = sample_rate_key
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        filename_key = self.filename_key
        sample_rate_key = self.sample_rate_key
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key

        dict_batch = super().__call__(batch)
        filename = dict_batch[filename_key]
        sample_rate = dict_batch[sample_rate_key]
        waveform = dict_batch[waveform_key]
        melspectrogram = dict_batch[melspectrogram_key]

        assert len(filename) == 1, "Pseudo batch size should be 1."
        assert waveform.size(0) == 1, "Pseudo batch size should be 1."
        assert sample_rate.size(0) == 1, "Pseudo batch size should be 1."
        assert melspectrogram.size(0) == 1, "Pseudo batch size should be 1."

        # remove pseudo dimension
        dict_batch[filename_key] = filename[0]
        dict_batch[sample_rate_key] = sample_rate[0]
        dict_batch[waveform_key] = waveform[0]
        dict_batch[melspectrogram_key] = melspectrogram[0]

        return dict_batch
