import glob
import os
from typing import Optional

from audyn.utils.data.birdclef.birdclef2024.dataset import (
    BirdCLEF2024AudioDataset as _BirdCLEF2024AudioDataset,
)

__all__ = [
    "BirdCLEF2024AudioDataset",
]


class BirdCLEF2024AudioDataset(_BirdCLEF2024AudioDataset):
    """Dataset for inference of bird classification model.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``soundscape_ABCDE``.
        feature_dir (str): Path to dataset containing ``sample_submission.csv`` file,
            ``test_soundscapes`` directory, and so on.
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        filename_key (str): Key of filename in given sample.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    .. note::

        If test set is unavailable, training set of unlabeled dataset is used.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        audio_key: str = "audio",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        decode_audio_as_waveform: Optional[bool] = None,
        decode_audio_as_monoral: Optional[bool] = None,
    ) -> None:
        super().__init__(
            list_path,
            feature_dir,
            audio_key=audio_key,
            sample_rate_key=sample_rate_key,
            filename_key=filename_key,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        audio_paths = sorted(glob.glob(os.path.join(self.audio_root, "*.ogg")))

        if len(audio_paths) == 0:
            # fall back to unlabeled dataset
            self.fallback_to_unlabeled_audio(list_path, feature_dir)

    def fallback_to_unlabeled_audio(self, list_path: str, feature_dir: str) -> None:
        list_dir = os.path.dirname(list_path)

        audio_root = os.path.join(feature_dir, "unlabeled_soundscapes")
        list_path = os.path.join(list_dir, "unlabeled_validation.txt")

        filenames = []

        with open(list_path) as f:
            for line in f:
                filename = line.strip()
                filenames.append(filename)

        self.audio_root = audio_root
        self.list_path = list_path
        self.filenames = filenames
