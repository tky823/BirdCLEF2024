import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as aT
from audyn.utils.data.birdclef.birdclef2024 import primary_labels as birdclef2024_primary_labels
from audyn.utils.data.birdclef.birdclef2024.composer import (
    BirdCLEF2024AudioComposer as _BirdCLEF2024AudioComposer,
)
from audyn.utils.data.birdclef.birdclef2024.composer import (
    BirdCLEF2024PrimaryLabelComposer as _BirdCLEF2024PrimaryLabelComposer,
)
from audyn.utils.data.composer import Composer
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = [
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024AudioComposer",
    "BirdCLEF2024AudioChunkingComposer",
]


class BirdCLEF2024PrimaryLabelComposer(_BirdCLEF2024PrimaryLabelComposer):
    """Alias of audyn.utils.data.birdclef.birdclef2024.composer.BirdCLEF2024PrimaryLabelComposer."""  # noqa: E501


class BirdCLEF2024AudioComposer(_BirdCLEF2024AudioComposer):
    """Alias of audyn.utils.data.birdclef.birdclef2024.composer.BirdCLEF2024AudioComposer."""


class BirdCLEF2024AudioChunkingComposer(Composer):
    """Composer for BirdCLEF2024, which supports chunking.

    This class can be used for inference.

    Args:
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        filename_key (str): Key of filename in given sample.
        waveform_key (str): Key of waveform to add to given sample.
        melspectrogram_key (str): Key of Mel-spectrogram to add to given sample.
        sample_rate (int): Target sampling rate. Default: ``32000``.
        max_duration (float, optional): Max duration. Since some test samples are longer
            than 240s, this option is useful.
        chunk_duration (float): Duration of chunked audio. Default: ``5``.
        hop_duration (float, optional): Duration of hop size. If ``None``,
            ``chunk_duration`` is used.
        pad_duration (_size_2t): Duration of padding. Default: ``0``.
        decode_audio_as_waveform (bool): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. This parameter is given to Composer class.
            When composer is specified, this parameter is not used. Default: ``True``.
        decode_audio_as_monoral (bool): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. When composer is specified, this parameter is not used.
            Default: ``True``.
        append_end_time_to_filename (bool): If ``True``, end time is appended to filename
            as suffix.

    .. note::

        ``birdclef2024.utils.data.BirdCLEF2024ChunkingCollator`` is expected
        to be used as collator.

    .. note::

        When ``append_end_time_to_filename=True``, end time is determined by ``hop_duration``.

    """

    def __init__(
        self,
        melspectrogram_transform: Union[
            aT.MelSpectrogram,
            nn.Module,
        ],
        audio_key: str,
        sample_rate_key: str,
        filename_key: str = "filename",
        waveform_key: str = "waveform",
        melspectrogram_key: str = "melspectrogram",
        sample_rate: int = 32000,
        max_duration: Optional[float] = None,
        chunk_duration: float = 5,
        hop_duration: Optional[float] = None,
        pad_duration: _size_2_t = 0,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
        append_end_time_to_filename: bool = True,
        training: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        if hop_duration is None:
            hop_duration = chunk_duration

        self.melspectrogram_transform = melspectrogram_transform

        self.audio_key = audio_key
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.waveform_key = waveform_key
        self.melspectrogram_key = melspectrogram_key

        self.primary_labels = birdclef2024_primary_labels
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.pad_duration = _pair(pad_duration)
        self.training = training

        assert hasattr(self.melspectrogram_transform, "train")
        assert callable(self.melspectrogram_transform.train)
        assert hasattr(self.melspectrogram_transform, "eval")
        assert callable(self.melspectrogram_transform.eval)

        if self.training:
            self.melspectrogram_transform.train()
        else:
            self.melspectrogram_transform.eval()

        self.append_end_time_to_filename = append_end_time_to_filename

        if chunk_duration is None:
            raise ValueError("chunk_duration is required.")

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        audio_key = self.audio_key
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key
        waveform_key = self.waveform_key
        melspectrogram_key = self.melspectrogram_key
        target_sample_rate = self.sample_rate
        max_duration = self.max_duration
        chunk_duration = self.chunk_duration
        hop_duration = self.hop_duration
        pad_duration = self.pad_duration

        sample = super().process(sample)

        audio = sample[audio_key]
        sample_rate = sample[sample_rate_key]
        filename = sample[filename_key]
        sample_rate_dtype = sample_rate.dtype
        sample_rate = sample_rate.item()

        assert isinstance(audio, torch.Tensor), f"{type(audio)} is not supported."

        if sample_rate != target_sample_rate:
            audio = aF.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        if max_duration is not None:
            max_length = int(max_duration * sample_rate)

            if audio.size(-1) > max_length:
                trimming = audio.size(-1) - max_length

                if self.training:
                    trimming_left = torch.randint(0, trimming, ()).item()
                else:
                    trimming_left = trimming // 2

                trimming_right = trimming - trimming_left
                audio = F.pad(audio, (-trimming_left, -trimming_right))

        chunk_length = int(sample_rate * chunk_duration)
        hop_length = int(sample_rate * hop_duration)
        pad_duration_left, pad_duration_right = pad_duration
        padding_left = int(sample_rate * pad_duration_left)
        padding_right = int(sample_rate * pad_duration_right)
        audio = F.pad(audio, (padding_left, padding_right))

        num_chunks = math.ceil((audio.size(-1) - chunk_length) / hop_length) + 1

        # additional padding
        padding = (num_chunks - 1) * hop_length + chunk_length - audio.size(-1)

        if padding > 0:
            if self.training:
                padding_left = torch.randint(0, padding, ()).item()
            else:
                padding_left = padding // 2

            padding_right = padding - padding_left
            audio = F.pad(audio, (padding_left, padding_right))

        *batch_shape, length = audio.size()
        audio = audio.view(-1, 1, 1, length)
        audio = F.unfold(audio, kernel_size=(1, chunk_length), stride=(1, hop_length))
        audio = audio.view(*batch_shape, num_chunks, chunk_length)

        n_dims = audio.dim()
        dims = tuple(range(n_dims))
        dims = dims[-2:-1] + dims[:-2] + dims[-1:]

        # num_chunks is imposed to batch dimension.
        audio = audio.permute(*dims).contiguous()
        sample_rate = torch.full(
            (num_chunks,),
            fill_value=sample_rate,
            dtype=sample_rate_dtype,
        )

        sample[audio_key] = audio

        # filename
        filenames = []

        for chunk_idx in range(num_chunks):
            if self.append_end_time_to_filename:
                end = (chunk_idx + 1) * hop_duration
                filenames.append(f"{filename}_{end}")
            else:
                filenames.append(filename)

        sample[filename_key] = filenames

        melspectrogram = self.melspectrogram_transform(audio)

        output = {
            waveform_key: audio,
            melspectrogram_key: melspectrogram,
            sample_rate_key: sample[sample_rate_key],
            filename_key: sample[filename_key],
        }

        return output
