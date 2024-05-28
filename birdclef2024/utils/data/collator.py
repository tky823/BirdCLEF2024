import warnings
from typing import Any, Dict, List, Optional

from audyn.utils.data.birdclef.birdclef2024.collator import (
    BirdCLEF2024BaselineCollator as _BirdCLEF2024BaselineCollator,
)
from audyn.utils.data.collator import Collator
from audyn.utils.data.dataset import Composer

from .composer import BirdCLEF2024AudioChunkingComposer

__all__ = [
    "BirdCLEF2024BaselineCollator",
    "BirdCLEF2024ChunkingCollator",
]


class BirdCLEF2024BaselineCollator(_BirdCLEF2024BaselineCollator):
    """Alias of audyn.utils.data.birdclef.birdclef2024.collator.BirdCLEF2024BaselineCollator"""


class BirdCLEF2024ChunkingCollator(Collator):
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
        if isinstance(composer, BirdCLEF2024AudioChunkingComposer):
            warnings.warn(
                f"{type(composer)} is not supported by BirdCLEF2024ChunkingCollator, "
                "which may cause unexpected behavior of chunking.",
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
